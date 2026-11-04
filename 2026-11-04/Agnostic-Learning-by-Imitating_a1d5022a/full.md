# Agnostic Learning by Imitating

S´ılvia Casacuberta<sup>∗</sup> Stanford University

Omer Reingold<sup>‡</sup> Stanford University

Parikshit Gopalan Apple

Konstantinos Stavropoulos<sup>§</sup> Institute for Advanced Study

Varun Kanade<sup>†</sup> Oxford University

Pranay Tankala<sup>¶</sup> Harvard University

November 3, 2026

## Abstract

Given a class of machine learning models and sample access to a data distribution with binary labels, which computational task is harder: training a discriminative model that distinguishes positive and negative labels as well as the best member of the class, or training a generative model whose outputs are indistinguishable from real data by all members of the class? We prove that these two fundamental tasks in computational learning theory are not just closely related, but actually equivalent to each other in complexity. One direction of this equivalence was previously known. Indeed, the powerful regularity lemma in complexity theory gives an eficient oracle reduction from the indistinguishability task (a.k.a. multiaccuracy) to the distinguishing task (a.k.a. agnostic learning) in the style of boosting algorithms. Our main contribution is an eficient oracle reduction in the other direction.

Taking this idea further, we show that a multiaccuracy oracle is a remarkably powerful primitive, achieving not just agnostic learning but also a wide range of seemingly more demanding learning objectives via similarly eficient and direct constructions. Our results crucially use the power of making adaptive queries to the oracle, and stand in stark contrast to prior work, which showed that post-processing a single multiaccurate predictor is insuficient for even weak forms of learning. On a technical level, we draw a connection between our (ofline) problem and a version of no-regret learning in the online setting, which allows us to use follow-the-leader type algorithms to modify the label distribution and aggregate the predictions of the multiaccuracy oracle.

## Contents

1 Introduction   
1.1 Main Result . . 2   
1.2 Why This Direction Is Harder . 3   
1.3 Algorithm Overview 5   
1.4 Analysis Overview 10   
1.5 Related Work . 13   
1.6 AI Methodology 14   
2 Preliminaries 14   
3 Upper Bounds 16   
3.1 Reweighting and Conditioning . 17   
3.2 Achieving Agnostic Learning 17   
3.3 Achieving Omniprediction and Multicalibration 18   
3.4 Deferred Lemmas on FTL and Smoothing 20   
4 Lower Bounds 22   
4.1 Lower Bounds for Omniprediction 22   
4.2 Lower Bounds for Multicalibration 23   
5 Derandomization 25   
5.1 Degree-2 Calibration . 26   
5.2 General Derandomization 29   
A Deferred Proofs From Section 3 40   
B Deferred Proofs From Section 4 44   
C Deferred Proofs From Section 5 47   
D Constructing a Partition Function 50   
E Implementation using Statistical Queries 51

## 1 Introduction

Suppose we are given a class C of machine learning models. These can either be viewed abstractly as an arbitrary set of functions, or more concretely as a set of decision trees (depth $\leqslant d )$ , neural networks (parameters $\leqslant p )$ , or Boolean circuits $( { \mathrm { s i z e } } \leqslant s )$ , to name a few examples. Suppose also that we have been given sample access to a ground-truth distribution D over labeled data points $( { \pmb x } , { \pmb y } )$ , where $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ is a feature vector and $\pm  \in \{ \pm 1 \}$ is a binary label. In this setting, there are two fundamental and natural computational goals that one might attempt to achieve:

1. Train a discriminative model that distinguishes positively and negatively labeled data points from D as well as (or better than) the best model in the baseline class $\mathcal { C } .$

2. Train a generative model which produces synthetic labels for each point, such that the resulting joint distribution of points and labels is indistinguishable from real labeled data drawn from D by all observers in the class C.

Although these two tasks sound related, they are not quite the same, and they lie at the core of two diferent formulations of the question: What does it mean to learn optimally, relative to a baseline class? The first task, namely the distinguishing task with respect to a restricted class of competitors, corresponds to the agnostic model for learning, introduced by [Hau92] and [KSS94], extends Valiant’s influential theory of Probably Approximately Correct (PAC) learning [Val84] beyond the realizable setting. The second task, which frames learning as ensuring indistinguishability with respect to a restricted class of distinguishers, has its roots in the imitation game proposed by Alan Turing as a test of machine intelligence, now known as the Turing test [Tur50]. The form of indistinguishability that is relevant to us is multiaccuracy, which was first defined in the context of complexity theory [TTV09], and later in the setting of algorithmic fairness [HKRR18].

In this work, we ask: What is the complexity of these two learning tasks relative to one another? That is, which of the two tasks is easier and which is harder in terms of its computational complexity? Also, does the answer change depending on how restricted or how expressive the class C is, or depending on the properties of the underlying data distribution $\mathcal { D } ?$

Before stating our results, let us briefly state the formal definitions of the versions of the distinguishing and indistinguishability tasks that we consider.

Distinguishing. We model the distinguishing task as agnostic learning [Hau92, KSS94]. Here, we are given independent samples from D and a description of a class C of functions $c : \mathcal { X }  [ - 1 , + 1 ]$ and we train a discriminative model or predictor $p : \mathcal { X }  [ - 1 , + 1 ]$ . Each output of $p$ is a number between −1 and +1, which should be interpreted as the mean of a distribution over the two-element set of binary labels $\{ - 1 , + 1 \}$ . Our goal is for the expected absolute value of the error $\pmb { y } - \pmb { p } ( \pmb { x } )$ to be not much worse than that of the best model in the class $\mathcal { C } \colon$

$$
\mathbb { E } \big [ | { \pmb y } - { \boldsymbol p } ( { \pmb x } ) | \big ] \leqslant \operatorname* { m i n } _ { { \boldsymbol c } \in { \mathcal { C } } } \mathbb { E } \big [ | { \pmb y } - { \boldsymbol c } ( { \pmb x } ) | \big ] + { \boldsymbol \varepsilon } .\tag{1}
$$

Informally, our model should “beat” the baseline class $\mathcal { C }$ at loss minimization. Our algorithm should succeed with probability at least $1 - \delta$ regardless of the conditional distribution of $\textbf {  { y } }$ given x, but we treat the marginal distribution $\mathcal { D } _ { \mathcal { X } }$ over the features x as fixed and known in advance.

Although we have stated the definition of agnostic learning in terms of $\ell _ { 1 }$ loss minimization, it can be stated just as easily in terms of distinguishing two distributions over the feature space $x ,$ corresponding to positively and negatively labeled examples. The first distribution is the conditional distribution $\mathcal { D } _ { + }$ of x given $y = + 1$ and the second is the conditional distribution $\mathcal { D } _ { - }$ of x given $y = - 1$ . If labels are balanced, meaning $\mathbb { E } [ \pmb { y } ] = 0$ , then a model $p$ satisfies inequality (1) if and only if its advantage at distinguishing $\mathcal { D } _ { + }$ from $\mathcal { D } _ { - }$ is at most $2 \varepsilon$ worse than the best distinguishing advantage of any function in C. Here, distinguishing advantage has the usual definition of

$$
\begin{array} { r } { \operatorname { A d v } _ { f } ( \mathcal { P } , \mathcal { Q } ) = \operatorname { \mathbb { E } } _ { { \pmb x } \sim \mathcal { P } } \big [ f ( { \pmb x } ) \big ] - \operatorname { \mathbb { E } } _ { { \pmb x } \sim \mathcal { Q } } \big [ f ( { \pmb x } ) \big ] , } \end{array}
$$

for probability distributions $\mathcal { P }$ and Q on X and a test $f : \mathcal { X }  [ - 1 , + 1 ]$

Indistinguishability. We model the indistinguishability task as multiaccuracy [TTV09, HKRR18]. Here, D and C are the same as in agnostic learning, but we now view $p : \mathcal { X }  [ - 1 , + 1 ]$ as a generative model, a perspective that we will soon elaborate. Our goal is for the error, still $\pmb { y } - \pmb { p } ( \pmb { x } )$ , to be approximately uncorrelated with all models in the class $\mathcal { C } \colon$

$$
\operatorname* { m a x } _ { c \in \mathcal { C } } \Big | \mathbb { E } \big [ c ( \pmb { x } ) ( \pmb { y } - p ( \pmb { x } ) ) \big ] \Big | \leqslant \varepsilon .\tag{2}
$$

Informally, our model should “fool” observers in the class C into thinking that the labels it generates are the same as real labels. Once again, we allow a small failure probability $\delta ,$ treat the conditional distribution of y given x as unknown, and treat the marginal distribution of x as known.

Although we have stated the definition of multiaccuracy in terms of correlation, it can be stated just as easily in terms of distinguishing the ground-truth distribution $\mathcal { D }$ over real data points $( { \pmb x } , { \pmb y } )$ from the modeled distribution $\tilde { \mathcal { D } }$ over simulated data points $( \pmb { x } , \tilde { \pmb { y } } )$ , where $\tilde { y }$ is generated from $p$ in the sense that $\mathbb { E } [ \tilde { \pmb { y } } | \pmb { x } ] = p ( \pmb { x } )$ . Indeed, a two-input distinguisher $A ( x , y )$ taking values in $[ - 1 , + 1 ]$ achieves absolute advantage at most ε at distinguishing $\mathcal { D }$ from $\tilde { \mathcal { D } }$ if and only if $p$ satisfies inequality (2) with respect to the function $c _ { A } ( x ) = ( A ( x , + 1 ) - A ( x , - 1 ) ) / 2$

The imitation in the title refers to multiaccuracy, since the only guarantee we have is that the real and simulated distributions look the same to observers modeled by functions in $\mathcal { C } .$

## 1.1 Main Result

Our main result is that distinguishing (agnostic learning) and indistinguishability (multiaccuracy) are statistically and computationally equivalent in complexity for all feature distributions $\mathcal { D } _ { \mathcal { X } }$ and all model classes C. By “equivalent”, we mean that an eficient algorithm for either one of the two problems can be transformed into an eficient algorithm for the other using a simple and elegant oracle reduction with $\mathrm { p o l y } ( 1 / \varepsilon )$ overhead.

Theorem 1.1 (Main, Informal). Agnostic learning and multiaccuracy are statistically and computationally equivalent in the sense that for all feature distributions $\mathcal { D } _ { \mathcal { X } }$ and model classes $\mathcal { C } .$

• There exists a $( \mathcal { C } , \varepsilon )$ -agnostic learning algorithm for $\mathcal { D } _ { \mathcal { X } }$ that only accesses the data distribution via ${ \cal O } ( 1 / \varepsilon ^ { 5 } )$ calls to a $( \mathcal { C } , \Omega ( \varepsilon ^ { 2 } ) ,$ )-multiaccuracy oracle for $\mathcal { D } _ { \mathcal { X } }$

• (Conversely, there exists a $( \mathcal { C } , \varepsilon )$ -multiaccuracy algorithm for $\mathcal { D } _ { \mathcal { X } }$ that accesses the data distribution via ${ \cal O } ( 1 / \varepsilon ^ { 2 } )$ calls to a $( \mathcal { C } , \Omega ( \varepsilon ) )$ -agnostic learning oracle for $\mathcal { D } \chi \ / T T V O 9 , H K R R 1 8 ] . )$

The converse direction of this equivalence was previously known. Indeed, the powerful regularity lemma of $\left[ \mathrm { T T V 0 9 } \right]$ states that eficient $( \mathcal { C } , \varepsilon )$ -multiaccurate models always exist in the form of clipped linear combinations of ${ \cal O } ( 1 / \varepsilon ^ { 2 } )$ functions from $\mathcal { C } , ^ { 1 }$ and it was later shown that these component functions can be eficiently found using adaptive calls to a (weak) agnostic learning oracle [HKRR18].<sup>2</sup> Agnostic learning turns out to be a simple yet powerful learning primitive. Other examples of goals that are known to be achievable using an agnostic learning oracle include multicalibration, a stronger version of (2) that conditions on the level sets of $p ,$ and omniprediction, a stronger version of (1) that requires small error not just for the $\ell _ { 1 }$ loss but also simultaneously for a multitude of other loss functions in a set $\mathcal { L }$ [HKRR18, GKR<sup>+</sup>22]. Provided $\ell _ { 1 } \in { \mathcal { L } }$ , omniprediction implies agnostic learning.

The forward direction was open prior to our work. Positive answers were known under stronger assumptions on the predictor, typically that it satisfies calibration guarantees [Daw85] of varying strengths. It was known that if a model is both C-multiaccurate and calibrated, then it must be an agnostic learner for the class $\mathcal { C }$ [CGKR25]. The stronger guarantee of multicalibration was known to imply omniprediction for a rich family of convex losses [GKR<sup>+</sup>22], and hence agnostic learning. These assumptions can be viewed as requiring stronger indistinguishability guarantees between the real and simulated data distributions, via the outcome indistinguishability framework [DKR<sup>+</sup>21].

The above reductions to agnostic learning were one-shot reductions, where we take the predictor p satisfying calibrated multiaccuracy, and output sign $( p ( x ) )$ as our learner. It was shown by [CGKR25] that such reductions will not sufice for multiaccuracy: there are multiaccurate predictors where even conditioned on the output of the predictor, the true label is uniformly random (a condition they called anti-calibration). This implies that no post-processing of the predictor can beat random guessing. But it leaves open the possibility that a more sophisticated reduction to multiaccuracy (for example, with repeated oracle calls) could still yield agnostic learning. Note that the reduction in the other direction crucially uses multiple rounds with adaptive calls to the agnostic learner to get a multiaccurate predictor.

In this work, we show that access to a multiaccuracy oracle for a class C is indeed suficient for agnostic learning of C. We prove this by designing a statistically and computationally eficient oracle reduction from agnostic learning to multiaccuracy. By “multiaccuracy oracle”, we mean an algorithm that receives as input n samples from an arbitrary labeled data distribution with fixed marginal $\mathcal { D } _ { \mathcal { X } }$ over features and outputs a predictor that is $( \mathcal { C } , \varepsilon )$ -multiaccurate with respect to this distribution. We find this main result surprising given the aforementioned negative results regarding the post-processing of a single multiaccurate predictor. Moreover, our reduction holds even in the statistical query (SQ) framework of [Kea98]. This follows by standard arguments for implementing boosting-style procedures using statistical queries [Fel09]. Importantly, this implies that SQ lower bounds for agnostic learning also extend to learning multiaccurate predictors. See the related work section, as well as Appendix E, for more details.

## 1.2 Why This Direction Is Harder

The standard reduction from multiaccuracy to weak agnostic learning can be seen as a boosting algorithm that proceeds in rounds: at round $t ,$ assuming we have not achieved multiaccuracy, we use the weak learner to find $c _ { t }$ that is correlated with the residual error ${ \pmb y } - p _ { t } ( { \pmb x } )$ of our predictor. We think of $( { \pmb y } - p _ { t } ( { \pmb x } ) ) / 2$ as the expectation of a Bernoulli random variable z, which $c _ { t }$ is guaranteed to predict better than random guessing. We update our predictor using $c _ { t }$ so as to reduce the squared error. A standard potential argument shows that this cannot proceed for too many rounds.

Why does boosting or some similar procedure not work in the reverse direction? It is because we work with a diferent primitive: At stage t, we are given a C-multiaccurate predictor $p _ { t }$ , which knows all the structure in the target distribution (that it was trained on) that is observable to the class C. Yet, the key challenge is that $p _ { t }$ need not be any better at predicting y than random guessing. The following example illustrates this point.

A Guiding Example. Consider a data distribution with uniform n-bit features $\mathbf { x } \sim \{ \pm 1 \} ^ { n }$ and binary labels ${ \pmb y } = { \pmb x } _ { n }$ . Let C be the class of symmetric functions $c : \{ \pm 1 \} ^ { n }  [ \pm 1 ]$ , which are invariant under permutations of the input bits. In this setting, there are many perfectly multiaccurate predictors. For example, consider the predictors $p _ { i } ( x ) = x _ { i }$ for each index $i \neq n$ By symmetry of the $i ^ { \mathrm { t h } }$ and $n ^ { \mathrm { t h } }$ bits, $p _ { i }$ satisfies (C, 0)-multiaccuracy. In fact, it does so even after conditioning on the predictions made by $p _ { 1 } , \dotsc , p _ { i - 1 } \colon$

$$
\mathbb { E } \big [ c ( \pmb { x } ) ( \pmb { x } _ { n } - \pmb { x } _ { i } ) \big | \pmb { x } _ { 1 } , \dots , \pmb { x } _ { i - 1 } \big ] = 0 .
$$

Indeed, from the point of view of symmetric tests, $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ is a fine model for ${ \pmb y } = { \pmb x } _ { n }$ . However, since the n coordinates of x are independent, the predictors $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n - 1 }$ <sub>−1</sub> are essentially random guesses for the label!

Assume that our goal is to achieve squared loss comparable to the best symmetric function. Each $p _ { i }$ gives a squared loss of 2 since

$$
\begin{array} { r } { \mathbb { E } [ ( { \pmb y } - { \pmb x } _ { i } ) ^ { 2 } ] = 2 - 2 \mathbb { E } [ { \pmb x } _ { i } { \pmb x } _ { n } ] = 2 . } \end{array}
$$

In contrast, the best symmetric function $\begin{array} { r } { s _ { n } ( { \pmb x } ) = \sum _ { i } x _ { i } / n } \end{array}$ achieves error $1 - 1 / n \colon$

$$
\mathbb { E } \bigl [ ( \pmb { y } - s _ { n } ( \pmb { x } ) ) ^ { 2 } \bigr ] = \mathbb { E } \biggl [ \biggl ( \pmb { x } _ { n } - \frac { \pmb { x } _ { 1 } + \dots + \pmb { x } _ { n } } { n } \biggr ) ^ { 2 } \biggr ] = 1 - \frac { 1 } { n } .\tag{3}
$$

Now assume that n is very large compared to the number of queries t we can make to the multiaccuracy oracle. Given $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { t }$ for $t \ll n$ in response to our queries, can we combine these random guesses to improve on the squared loss of each individual guess? The answer is yes: we output the average $\textstyle s _ { t } ( { \pmb x } ) = \sum _ { i < t } { \pmb x } _ { i } / t$ , whose squared error is seen to be bounded by $1 + 1 / t$ . Hence for $t = O ( 1 / \varepsilon )$ , we are within $\bar { 1 / n } + 1 / t = O ( \varepsilon )$ of the optimal squared error, since $t < n$

What can we learn from this? Note that $s _ { t } ( \pmb { x } )$ is also independent of ${ \bf { x } } _ { n } ,$ just like each $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ for $i \leq t$ . It is also C-multiaccurate, being the average of multiaccurate predictors. However, unlike the $\mathbf { \Delta } _ { \mathbf { x } _ { i } \mathrm { s } }$ and like $s _ { n }$ , it has low calibration error. By the result of [GKR23, CGKR25], multiaccuracy together with calibration implies agnostic learning. So one lesson is to try and make calibrated predictions, while not losing multiaccuracy.

Another key idea comes from considering the agnostic learning setting, where the goal is to achieve correlation comparable to the best hypothesis in C. As [CGKR25] showed, there are distributions D and classes C where the optimal hypothesis from the class c satisfies $\mathbb { E } [ c ( { \pmb x } ) { \pmb y } ] ~ \ge$ $1 / 2$ , whereas there exist multiaccurate predictors where $\mathbb { E } [ p ( { \pmb x } ) { \pmb y } ] = 0 ;$ in fact the stronger anticalibration condition $\mathbb { E } [ { \pmb { y } } | { \pmb { p } } ( { \pmb { x } } ) ] = 0$ holds. This anti-calibration condition in their example also shows that na¨ıvely attempting to calibrate a multiaccurate predictor doesn’t work as we simply get the constant zero function after calibrating. We can reconstruct a similar example by fixing an odd dimension $n \geq 3$ , setting ${ \pmb y } = { \pmb x } _ { n }$ , and letting $\alpha : = \mathbb { E } [ \mathrm { M a j } _ { n } ( { \pmb x } ) { \pmb y } ] = \Theta ( 1 / \sqrt { n } )$ , where $\operatorname { M a j } _ { n }$ is the Majority function on n variables. Then, $\operatorname { M a j } _ { n }$ achieves correlation α, which is optimal for C (all symmetric functions). But every convex combination of $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n - 1 }$ is perfectly C-multiaccurate, yet has 0 correlation with ${ \pmb x } _ { n }$ . Our aim is to learn a hypothesis with correlation close to $\alpha .$ . Clearly, we cannot achieve this by averaging predictors that have 0 correlation with $\mathbf { \pmb { y } } .$ . So we need the multiaccuracy oracle to not just repeat combinations of $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n - 1 }$ , but to eventually reveal ${ \mathbf { \mathcal { x } } } _ { n }$ In general, we want to extract from the oracle a diverse set of predictors, including ones that have non-trivial correlation. These observations inspire our main reduction, which we describe in the next section.

Algorithm 1 Weighted Multiaccuracy   
Require: Multiaccuracy oracle MA with sample size $n _ { \mathrm { : } }$ reweighting measure m : $\mathcal { X }  [ 0 , 1 ]$ , labeled   
data points $( x _ { t i } , y _ { t i } ) \in \mathcal { X } \times \{ \pm 1 \}$ for $t \in [ T ]$ and $i \in [ n ]$   
1: $p _ { 0 } = 0$ ▷ Learning   
2: for $t = 1 , \dots , T \colon$   
3: Sample $z _ { t i } \in \{ \pm 1 \} { \mathrm { ~ w i t h ~ } } \mathbb { E } [ z _ { t i } | x _ { t i } , y _ { t i } ] = m ( x _ { t i } ) \cdot y _ { t i } + ( 1 - m ( x _ { t i } ) ) \cdot p _ { t - 1 } ( x _ { t i } )$   
4: $p _ { t } = \mathsf { M A } ( ( x _ { t i } , z _ { t i } ) _ { i \in [ n ] } )$   
5: return $\textstyle p = { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } p _ { t }$ ▷ Averaging

## 1.3 Algorithm Overview

High-Level Outline. We query the multiaccuracy oracle to get predictors $p _ { 1 } , \ldots , p _ { t }$ that are all C-multiaccurate for the target distribution D. At each stage, we ensure that the predictor $p _ { t }$ is weakly independent of the predictors seen so far. We then output a randomization/weighted average of the various $p _ { t }$ as our final predictor. If we think of each $p _ { t }$ as having a structured component along C and an orthogonal noise component, then averaging preserves the structure but the noise cancels out. Over multiple rounds, we prove that randomizing over the predictors yields a predictor that is both multiaccurate and calibrated, and this implies agnostic learning. Indeed, this approach naturally yields the much stronger primitive of swap multicalibration, which is known to guarantee swap omniprediction, a strong generalization of agnostic learning [GKR23].

The question that remains is finding the right notion of weak independence. We require that the predictor $p _ { t }$ is multiaccurate conditioned on the level sets of $\textstyle \sum _ { i < t } w _ { i } p _ { i } / ( \sum _ { i < t } w _ { i } )$ , suitably discretized, for some suitably chosen weights $w _ { i } \geq 0$ . We show how to obtain such a predictor eficiently using calls to a standard multiaccuracy oracle, changing only the conditional distribution over the labels y, while leaving the marginal distribution over x fixed. By stitching together the multiaccurate predictors for each level set, we ensure that the overall predictor is multiaccurate for the target distribution D that we wish to learn agnostically.

The analysis of this algorithm is fairly subtle; it combines ideas from classic algorithms like Follow-the-leader in the online learning literature [Han57], with the technique of loss outcome indistinguishability from more recent work on omniprediction $[ \mathrm { G H K ^ { + } 2 3 } ]$ . Randomization over the predictors $p _ { i }$ , which is typical in online learning, is in fact essential for achieving strong notions like omniprediction or multicalibration that give guarantees for multiple losses or distinguishers, since diferent $p _ { i } \mathrm { s }$ might do well for diferent losses. In contrast, for agnostic learning (single loss) or degree-2 calibration (single distinguisher), one can pick the single best predictor over the iterations.

We now present a more formal outline of the reduction from agnostic learning to multiaccuracy, which has several steps. It is simple to state what each step aims to accomplish; however the analysis is far from obvious. For this reason, we first explain the steps of the algorithm and defer all analysis to the technical overview in Section 1.4.

Step 1: Reweighting. The first step, described in Algorithm 1, converts a $( \mathcal { C } , \varepsilon )$ -multiaccuracy oracle for a fixed feature distribution $\mathcal { D } _ { \mathcal { X } }$ into a multiaccuracy oracle for any reweighting of $\mathcal { D } _ { \mathcal { X } }$ The algorithm takes as input a function m $: \mathcal { X }  [ 0 , 1 ]$ , which we view as a measure that is allowed to decrease (but not increase) the weight of some feature vectors $x \in \mathcal { X }$ . If we seek multiaccuracy conditional on the level sets of an arbitrary function $f : \mathcal { X }  \mathcal { V }$ of the features, then running

Algorithm 2 Agnostic Learning   
Require: Conditional multiaccuracy oracle CMA, grid size $\gamma ^ { - 1 } \in \mathbb { N } .$ , function $f : [ \pm 1 ] \to [ \pm B ]$   
1: for $t = 1 , 2 , \dots , T \colon$   
2: $\begin{array} { r } { q _ { t } = \sum _ { s < t } ( s \cdot r _ { s } ) / { \binom { t } { 2 } } } \end{array}$ ▷ Response aggregation $( q _ { 1 } = 0 )$   
3: $r _ { t } = \mathsf { C M A } ( R _ { \gamma } \circ f \circ q _ { t } )$ ▷ $R _ { \gamma }$ rounds to multiples $o f \gamma$   
4: $\textstyle p = \sum _ { t = 1 } ^ { T } ( t \cdot ( f \circ q _ { t } ) ) / { \binom { T + 1 } { 2 } }$ ▷ Averaging   
return p

Algorithm 1 on each of the {0, 1}-valued level-set indicator functions $m = \mathbf { 1 } _ { f ^ { - 1 } ( v ) }$ achieves the following guarantee, which we call $( \mathcal { C } , \varepsilon )$ -conditional multiaccuracy.

Theorem 1.2 (Conditional Multiaccuracy). For all $f : \mathcal { X }  \mathcal { V }$ , running Algorithm 1 for $T =$ $O ( 1 / \varepsilon )$ rounds with a $( \mathcal { C } , \varepsilon )$ -multiaccuracy oracle on each level set of f yields a predictor $p : \mathcal { X } $ [±1] satisfying

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { c \in \mathcal { C } } \Big | \mathbb { E } \big [ c ( \pmb { x } ) ( \pmb { y } - p ( \pmb { x } ) ) \mid f ( \pmb { x } ) \big ] \Big | \right] \leqslant O \left( | \mathcal { V } | \varepsilon \right) .
$$

If the oracle has failure probability δ, then the resulting Conditional Multiaccuracy Algorithm has failure probability at most $O \left( | \mathcal { V } | \delta / \varepsilon \right)$

Algorithm 1 operates by making a sequence of queries to the (unweighted) multiaccuracy oracle, using the same feature distribution but diferent label distributions at each time step. Specifically, at time $t \leqslant T$ , our algorithm uses a label distribution that preserves the true label of $x \in \mathcal { X }$ with probability $m ( x )$ , and otherwise replaces it with the prediction $p _ { t - 1 } ( x )$ of the oracle’s previous output; since binary labels are required, we randomly round the prediction to {±1} accordingly.

Step 2: Agnostic Learning. The next part of our reduction, described in Algorithm 2, also makes iterative calls to an oracle, but this time to a conditional multiaccuracy oracle, such as the one in Theorem 1.2. The following theorem states the agnostic learning guarantee that it achieves.

Theorem 1.3 (Agnostic Learning). Algorithm 2 with a $( \mathcal { C } , \gamma \varepsilon )$ -conditional multiaccuracy oracle, soft-thresholding $f = \mathrm { s i g n } _ { \varepsilon }$ , grid size $\gamma = \varepsilon$ , and time horizon $T = O ( 1 / \varepsilon ^ { 2 } )$ outputs p satisfying

$$
\mathbb { E } \big [ | { \pmb y } - { \pmb p } ( { \pmb x } ) | \big ] \leqslant \operatorname* { m i n } _ { c \in \mathcal { C } } \mathbb { E } \big [ | { \pmb y } - { \pmb c } ( { \pmb x } ) | \big ] + O ( \varepsilon ) .
$$

If the oracle has failure probability δ, then Algorithm 2 has failure probability at most $T \delta$

At each time step $t \leqslant T$ , Algorithm 2 first computes a weighted average $q _ { t } : \mathcal { X } \to [ \pm 1 ]$ of the oracle responses $r _ { 1 } , \ldots , r _ { t - 1 }$ received so far, placing slightly more weight on more recent responses. It then post-processes $q _ { t }$ with the carefully chosen soft-thresholding function

$$
f ( v ) = \mathrm { s i g n } _ { \varepsilon } ( v ) = { \left\{ \begin{array} { l l } { - 1 } & { { \mathrm { i f ~ } } v \leqslant - \varepsilon , } \\ { + 1 } & { { \mathrm { i f ~ } } v \geqslant + \varepsilon , } \\ { v / \varepsilon } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

Next, the algorithm queries the conditional multiaccuracy oracle on $f \circ q _ { t }$ , rounded to multiples of $\gamma ,$ to receive a new response $r _ { t }$ . Finally, the algorithm outputs a nonuniform weighted average of $f \circ q _ { t }$ over time steps $t \in [ T ]$

Algorithm 3 Multicalibration and Omniprediction   
Require: Conditional multiaccuracy oracle CMA, grid size $\gamma ^ { - 1 } \in \mathbb { N }$ , learning weights $a _ { 1 } , \dots , a _ { T } \geqslant$   
0, mixing weights $b _ { 1 } , \dots , b _ { T } \geqslant 0 ,$ noise function $N : [ \pm 1 ]  [ \pm 1 ]$   
1: for $t = 1 , 2 , \dots , T \colon$   
2: $q _ { t } = ( a _ { 1 } r _ { 1 } + \cdot \cdot \cdot + a _ { t - 1 } r _ { t - 1 } ) / ( a _ { 1 } + \cdot \cdot \cdot + a _ { t - 1 } )$ ▷ Response aggregation $( q _ { 1 } = 0 )$   
3: $r _ { t } = \mathsf { C M A } ( R _ { \gamma } \circ q _ { t } )$ $\textsf { D } R _ { \gamma }$ rounds to multiples of γ   
4: def $p ( x ) ;$ ▷ Randomization   
5: $\mathbf { \Delta } t \sim \mathbf { \mathrm { \Delta } } b$   
6: return $N ( q _ { t } ( x ) )$   
7: return p

Note that Theorem 1.3 implies that if the optimal correlation with C is a constant α, then the algorithm gets correlation $\alpha / 2$ in $O ( 1 / \alpha ^ { 2 } )$ rounds, hence in constantly many rounds. In our guiding example with $n = \Theta ( 1 / \alpha ^ { 2 } )$ variables, it implies that within $O ( n )$ rounds, the conditional multiaccuracy oracle must return a predictor that depends on ${ \mathbf { \mathcal { x } } } _ { n }$

Note that the averaging over stopping times t is a strength of the algorithm, not a weakness. Essentially, our guarantee states that the post-processed iterates $q _ { t }$ perform well on average. In particular, if one were to output the one with the lowest $\ell _ { 1 }$ loss, which can be identified with a handful of additional labeled samples, this would also perform well. We also remark that the nonuniform weighting in both the response aggregation and final averaging is not strictly necessary; as our proof will show, if we were to use a uniform weighting for both the average over $r _ { t }$ in the inner loop and the average over $f \circ q _ { t }$ at the end, it would only cost an additional factor of log(1/ε) in the time horizon T.

Step 3: Omniprediction and Multicalibration. At this point, we have shown that multiaccuracy is suficient to achieve agnostic learning, which was previously known to be suficient to achieve strong learning-theoretic goals like omniprediction and multicalibration. Our next result shows that with minor modifications to Algorithm 2, we can actually achieve direct conversions from multiaccuracy into omniprediction and multicalibration that are much more eficient than going through agnostic learning as an intermediate step and then composing with known reductions. As a concrete comparison, for the case of multicalibration, the algorithm of $[ \mathrm { G K R ^ { + } 2 2 } ]$ uses ${ \cal O } ( 1 / \varepsilon ^ { 6 } )$ calls to a weak agnostic learner. Each of these calls can in turn be implemented at the required accuracy using ${ \cal O } ( 1 / \varepsilon ^ { 2 } )$ conditional multiaccuracy calls, by our Theorem 1.3. This composition therefore uses a total of ${ \cal O } ( 1 / \varepsilon ^ { 8 } )$ conditional multiaccuracy calls to achieve multicalibration. In comparison, our Theorem 1.4 below achieves multicalibration (and the stronger guarantee of swap multicalibration) at the same accuracy using only ${ \cal O } ( 1 / \varepsilon ^ { 3 } )$ conditional multiaccuracy calls.

The notion of omniprediction we consider is with respect to a class of loss functions that we call naturally parametrized losses. To define this class, we need some notation. Consider a real-valued loss function $\ell ( y , a )$ , which takes as input a label $y \in \{ \pm 1 \}$ and an action a in some arbitrary action set. Given a value $p \in [ \pm 1 ]$ , which we identify with the unique probability distribution over {±1}

with mean p, we define

$$
\ell ( p , a ) = \underset { { \pmb y } \sim p } { \mathbb { E } } [ \ell ( { \pmb y } , a ) ] ,
$$

(expected loss)

$$
k _ { \ell } ( p ) \in \mathop { \mathrm { a r g m i n } } _ { a } \ell ( p , a ) .
$$

Note that $k _ { \ell } ( p )$ does not necessarily exist for an arbitrary loss ℓ and value $p \in [ \pm 1 ]$ , and even when it does exist, it might not be unique. We also define the discrete partial derivative

$$
\partial \ell ( \eta ) = \frac { \ell ( + 1 , \eta ) - \ell ( - 1 , \eta ) } { 2 } .
$$

We say that ℓ is naturally parametrized if its action set is R and $\partial \ell ( \eta ) = - \eta$ . This class includes, in particular, all generalized linear model (GLM) loss functions, a class of loss functions that are convex in this natural parameter, which have also arisen in prior work on omniprediction [GHK<sup>+</sup>23], but the losses we consider need not be convex. We let $\mathcal { L } _ { B }$ be the class of naturally parametrized loss functions with bounded best responses $k _ { \ell } ( p ) \in [ \pm B ]$ and we let $\mathcal { L } _ { B , \beta } \subseteq \mathcal { L } _ { B }$ further require that $k _ { \ell }$ is β-Lipschitz. For examples of such loss functions, see Section 2.

Algorithm 3 difers from Algorithm 2 in a few ways. Now, at step t, we split the domain by the level sets of $q _ { t } ,$ which is loss-independent, as opposed to splitting by the level sets of $f \circ q _ { t }$ for some loss-dependent post-processing $f .$ It also outputs a randomized predictor derived from the entire sequence of iterates $q _ { 1 } , \ldots , q _ { T }$ , rather than any particular chosen $q _ { t }$ from the sequence. We also sometimes add additional noise.<sup>3</sup> Theorem 1.4 below summarizes the various learning guarantees attained by Algorithm 3. We aim for omniprediction with respect to all suitably bounded naturally parametrized loss functions, with better rates when $k _ { \ell }$ is Lipschitz. We also target the challenging “swap” versions of multicalibration, along with a novel “smooth swap” version. We defer the formal definitions of omniprediction and multicalibration to the next section.

Theorem 1.4. Algorithm 3 with a $( \mathcal { C } , \gamma \varepsilon )$ -conditional multiaccuracy oracle outputs a randomized predictor with any one of the learning guarantees listed below, for some appropriate choices of the algorithm’s parameters $a _ { t } , ~ b _ { t }$ , and N (the choices depend on the particular guarantee):

(C, $O ( \varepsilon ) )$ -smooth omniprediction, using $T = O ( 1 / \varepsilon )$ oracle calls with $\gamma = \Omega ( \varepsilon )$ ;

(C, O(ε))- omniprediction, using $T = { \cal O } ( 1 / \varepsilon ^ { 2 } )$ oracle calls with $\gamma = \Omega ( \varepsilon ^ { 2 } )$ ;

(C, O(ε))-smooth swap multicalibration, using $T = O ( 1 / \varepsilon ^ { 2 } )$ oracle calls with $\gamma = \Omega ( \varepsilon ) ;$

(C, O(ε))-swap multicalibration, using $T = { \cal O } ( 1 / \varepsilon ^ { 3 } )$ oracle calls with $\gamma = \Omega ( \varepsilon ^ { 2 } )$

In each case, the predictor is also $( \mathcal { C } , O ( \varepsilon ) )$ -multiaccurate. Omniprediction is with respect to $\mathcal { L } _ { 1 }$ or $\mathcal { L } _ { 1 , 1 }$ . If the oracle has failure probability δ, then Algorithm 3 has failure probability at most Tδ.

We prove Theorem 1.4 in Section 3, where we also prove certain quantitative improvements in the case of a single loss function ℓ (not necessarily $\ell _ { 1 } )$ . We show in Section 4 that many of the rates in Theorem 1.4 are tight for our algorithm by constructing several specific hard inputs. We remark that omniprediction also has a challenging “swap” variant in the literature, which Theorem 1.4 does not explicitly attempt to achieve. However, it was shown in [GKR23] that swap omniprediction is equivalent up to a small change of parameters to swap multicalibration, which our algorithm indeed achieves.<sup>4</sup>

## 1.3.1 Additional Definitions

We now formally define multicalibration [HKRR18, GKR23] and omniprediction $[ \mathrm { G K R ^ { + } 2 2 } ]$ . As discussed, multiaccuracy is a notion of indistinguishability, and multicalibration strengthens it by applying it to each level set of the predictor. When the class C contains only constant functions, multicalibration collapses to the much older notion of calibration [Daw82]. In some cases, we allow a diferent distinguisher from C to be applied to each level set, which we call swapping. In our proofs, the notion of smoothness will play a central role, which is why we present a new notion of smooth swap multicalibration alongside the existing notions. For a unified definition, we use the framework of sample-access outcome indistinguishability $( O I ) \ [ \mathrm { D K R ^ { + } 2 1 } ]$

Definition 1.5 (Multiaccuracy and Multicalibration [HKRR18, GKR23]). Given a function class ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to [ \pm 1 ] \}$ , let $\mathcal { C } ^ { \mathrm { s w a p } }$ be the class of all functions $c : [ \pm 1 ] \times \mathcal { X }  [ \pm 1 ]$ such that $c _ { v } ( \cdot ) =$ $c ( v , \cdot ) \in \mathcal { C }$ for every value $v \in [ \pm 1 ]$ . A predictor $p : \mathcal { X }  [ \pm 1 ]$ is $( \mathcal { C } , \varepsilon )$ -swap multicalibrated with respect to a distribution D over labeled data points $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \{ \pm 1 \}$ if for all $c \in \mathcal { C } ^ { \mathrm { s w a p } }$

$$
\mathbb { E } \big [ c _ { p ( { \pmb x } ) } ( { \pmb x } ) ( { \pmb y } - p ( { \pmb x } ) ) \big ] \leqslant \varepsilon .\tag{4}
$$

There are several special cases, depending on whether or not c is $L - L i p s c h i t z .$ meaning $c ( v , x )$ is L-Lipschitz in v for every $x \in \mathcal { X }$ , and whether or not c factors, meaning $c ( v , x ) = w ( v ) c ^ { \prime } ( x )$ for some $w : [ \pm 1 ] \to [ \pm 1 ]$ and some $c ^ { \prime } \in { \mathcal { C } }$ . If (4) holds...

...when c is 1-Lipschitz,

then p is $( \mathcal { C } , \varepsilon )$ -smooth swap multicalibrated;

...when c factors,

then p is (C, ε)-multicalibrated;

...when c is 1-Lipschitz and factors, then p is (C, ε)-smooth multicalibrated;

...when c ignores its first input, then p is (C, ε)-multiaccurate.

We only consider base classes C that contain the constant 1 function and are closed under negation and convex combinations. We omit the prefix “multi” when C only contains the constant functions (e.g., ε-calibration instead of $( \mathcal { C } , \varepsilon )$ -multicalibration or $( \mathcal { C } , \varepsilon )$ -swap multicalibration, and ε- smooth calibration instead of $( \mathcal { C } , \varepsilon )$ )-smooth multicalibration or $( \mathcal { C } , \varepsilon )$ -smooth swap multicalibration). When p is a randomized predictor, the expectation in (4) is also taken over its internal randomness.

We remark that the definition of smooth swap multicalibration may be less intuitive than the others since it may be harder to imagine natural examples of functions $c \in \mathcal { C } ^ { \mathrm { { s w a p } } }$ that are Lipschitz but do not factor. For an example of where such functions naturally arise, one might consider the expected advantage of distinguishers $c ( \pmb { v } , \pmb { x } )$ that observe only an unbiased noisy estimate v of $p ( { \pmb x } )$ , rather than the prediction $p ( { \pmb x } )$ itself.

We now turn our attention to loss minimization and omniprediction $[ \mathrm { G K R ^ { + } 2 2 } ]$ , with an emphasis on naturally parametrized loss functions. While most parts of the following definition are standard in the literature, the smooth variant is new.

Definition 1.6 (Omnipredictor [GKR<sup>+</sup>22]). Given a function class C and a class of loss functions L, a predictor p is an $( \mathcal { L } , \mathcal { C } , \varepsilon )$ -omnipredictor if for all $c \in { \mathcal { C } }$ and $\ell \in { \mathcal { L } }$ ，

$$
\begin{array} { r } { \mathbb { E } \big [ \ell ( { \pmb y } , k _ { \ell } ( p ( { \pmb x } ) ) ) \big ] \leqslant \mathbb { E } \big [ \ell ( { \pmb y } , c ( { \pmb x } ) ) \big ] + \varepsilon . } \end{array}\tag{5}
$$

If inequality (5) only holds when the function $\partial \ell ( k _ { \ell } ( v ) )$ is 1-Lipschitz, then we instead say that p is an $( \mathcal { L } , \mathcal { C } , \varepsilon )$ -smooth omnipredictor.

By default, in this paper, L refers to the class of naturally parametrized loss functions, meaning that $\partial \ell ( \eta ) = - \eta$ , with $[ \pm 1 ]$ -valued best responses $k _ { \ell } ( p )$ . When this is clear from context, we simply say that p is a $( \mathcal { C } , \varepsilon )$ -omnipredictor or $( \mathcal { C } , \varepsilon )$ -smooth omnipredictor.

## 1.4 Analysis Overview

In this section, we present the ideas behind the analysis of our main oracle reductions from agnostic learning (and beyond) to multiaccuracy. There are multiple ways to frame our arguments. The most natural approach is perhaps through the lens of online learning, even though we study the batch setting. To begin, let us consider the following simple game played repeatedly between two players, named the Predictor and Nature.

Definition 1.7 (Deterministic Online Prediction Game). At each time step $t = 1 , \dots , T$

1. The Predictor selects and announces a prediction $q _ { t } \in [ \pm 1 ]$ . We require $q _ { t } = s ( r _ { 1 } , \ldots , r _ { t - 1 } )$ to be a deterministic function s of Nature’s previous responses $r _ { 1 } , \ldots , r _ { t - 1 }$ . We call the function s the Predictor’s strategy.

2. After seeing $q _ { t }$ , Nature adaptively selects and announces its response $r _ { t } \in [ \pm 1 ]$

Ultimately, the Predictor has one of two possible goals. The first possible goal of the Predictor is to minimize its external regret for a class of loss functions ${ \mathcal { L } } ,$ which measures its excess loss incurred with respect to the best fixed action $a \in [ \pm 1 ]$ in hindsight:

$$
R _ { \mathcal { L } } ( s ) = \operatorname* { m a x } _ { \ell \in \mathcal { L } } \bigg ( \sum _ { t = 1 } ^ { T } \ell ( r _ { t } , k _ { \ell } ( q _ { t } ) ) - \operatorname* { m i n } _ { a \in [ \pm 1 ] } \sum _ { t = 1 } ^ { T } \ell ( r _ { t } , a ) \bigg ) .
$$

As always, we focus on loss functions $\ell ( \cdot , \cdot )$ that are afine in their first input. We emphasize that both the Predictor’s action and Nature’s response are real values in the interval [±1], rather than discrete values in $\{ \pm 1 \}$

The second possible goal for the Predictor is to minimize its calibration error with respect to a class W of weight functions $w : [ \pm 1 ] \to [ \pm 1 ]$ . Intuitively, online calibration asks that the error $r _ { t } - q _ { t }$ should be uncorrelated with $w ( q _ { t } )$ . In the extreme, W contains all functions, in which case W-calibration requires that $\mathbb { E } [ r _ { t } | q _ { t } ] = q _ { t }$ for a random time step $t \sim$ Uniform $\{ 1 , \ldots , T \}$ . For a given weight class $\mathcal { W } ,$ we define the calibration error as

$$
\mathrm { C E } _ { \mathcal { W } } ( s ) = \operatorname* { m a x } _ { w \in \mathcal { W } } \sum _ { t = 1 } ^ { T } w ( q _ { t } ) ( r _ { t } - q _ { t } ) .
$$

## 1.4.1 Multiaccuracy Oracle Reductions by Online Regret/Calibration

The fundamental observation of our work is that any eficiently computable online strategy s for the Predictor with low regret $R _ { \mathcal { L } } ( s )$ can be converted into a computationally eficient oracle reduction from $( \mathcal { L } , \mathcal { C } )$ -omniprediction to C-multiaccuracy. Similarly, if the Predictor’s strategy achieves low online calibration error $\operatorname { C E } _ { \mathcal { W } } ( s )$ , then we can convert it into an eficient oracle reduction from ${ \mathcal { C } } { \mathrm { - s w a p } }$ multicalibration (for weight functions in W) to C-multiaccuracy.

Suppose we are given access to an oracle that takes as input a query predictor $q : \mathcal { X } \to [ \pm 1 ]$ and produces a response $r : \mathcal { X } \to [ \pm 1 ]$ that is multiaccurate on each level set of $q .$ Let us call this conditional multiaccuracy oracle repeatedly across time steps $t = 1 , \dots , T$ , where at each round t we submit the following query, defined pointwise for each $x \in \mathcal { X }$ as

$$
q _ { t } ( x ) = s ( r _ { 1 } ( x ) , \ldots , r _ { t - 1 } ( x ) ) .
$$

If the Predictor has regret $R _ { \mathcal { L } } ( s ) \leqslant \delta T$ , then applying this bound pointwise for each $x \in \mathcal { X }$ yields

$$
\sum _ { t = 1 } ^ { T } \ell ( r _ { t } ( x ) , k _ { \ell } ( q _ { t } ( x ) ) ) \leqslant \operatorname* { m i n } _ { c : \mathcal { X } \to [ \pm 1 ] } \sum _ { t = 1 } ^ { T } \ell ( r _ { t } ( x ) , c ( x ) ) + \delta T .\tag{6}
$$

Note that since the bound has been applied pointwise for each $x \in \mathcal { X }$ , we may choose a diferent fixed action $c ( x )$ for each $x \in \mathcal { X }$ on the right-hand side. Next, since ℓ is naturally parametrized, we have the special identity

$$
\ell ( r , \eta ) - \ell ( y , \eta ) = \eta ( y - r ) .
$$

This identity tells us that when the correlation of $\eta$ with $y - r$ is small, the action η incurs similar loss on a real label $y \in \{ \pm 1 \}$ as on a simulated label generated from $r \in [ \pm 1 ]$ . By $( \mathcal { C } , \varepsilon )$ -conditional multiaccuracy,<sup>5</sup> we know that simulated labels generated from the model $r _ { t }$ are indistinguishable from real labels by both the entire class $\mathcal { C }$ and by any post-processing of $q _ { t }$ in expectation over $( { \pmb x } , { \pmb y } )$ . Accordingly, we may replace $r _ { t } ( \pmb { x } )$ with $\textbf {  { y } }$ in inequality (6) whenever $c \in { \mathcal { C } }$ , which gives

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } [ \ell ( \pmb { y } , k _ { \ell } ( q _ { t } ( \pmb { x } ) ) ) ] \leqslant \operatorname* { m i n } _ { c \in \mathcal { C } } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \ell ( \pmb { y } , c ( \pmb { x } ) ) ] + ( 2 \varepsilon + \delta ) T .
$$

This is exactly $( \mathcal { L } , \mathcal { C } , 2 \varepsilon + \delta )$ -omniprediction, not for any single predictor $q _ { t }$ in the sequence, but rather for the randomized predictor $\mathbfit { p }$ that on input x outputs $q _ { t } ( x )$ for a freshly sampled time index $t \sim [ T ]$ . If the set ${ \mathcal { L } } = \{ \ell \}$ contains a single convex loss $\ell ,$ as is the case for agnostic learning, then this randomization is not even necessary, since we may simply output the average of the (post-processed) predictors $q _ { t }$ in the sequence.

Similarly, if the Predictor has online calibration error $\mathrm { C E } _ { \mathcal { W } } ( s ) \leqslant \delta T$ , then we get the following pointwise bound, where $c ( v , x )$ is any function such that $c ( \cdot , x ) \in \mathcal { W }$ for all $x \in \mathcal { X }$

$$
\sum _ { t = 1 } ^ { T } c ( q _ { t } ( x ) , x ) ( r _ { t } ( x ) - q _ { t } ( x ) ) \leqslant \delta T .
$$

Once again, by applying $( \mathcal { C } , \varepsilon )$ -conditional multiaccuracy, we may replace $r _ { t } ( \pmb { x } )$ with y whenever $c ( v , \cdot ) \in \mathcal { C }$ for all values $v \in [ \pm 1 ]$ . This gives us either $( \mathcal { C } , \varepsilon + \delta )$ -swap multicalibration or $( { \mathcal { C } } , \varepsilon + \delta ) .$ smooth swap multicalibration, depending on whether the class $\mathcal { W }$ contains all possible bounded weight functions or just those bounded weight functions that are 1-Lipschitz:

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } [ c ( q _ { t } ( x ) , \pmb { x } ) ( \pmb { y } - q _ { t } ( \pmb { x } ) ) ] \leqslant ( \delta + \varepsilon ) T .
$$

## 1.4.2 Online Regret/Calibration by Following the Leader

At this point, all that remains is to determine which loss classes $\mathcal { L }$ and which weight classes W admit eficient strategies for the Predictor with low regret and calibration error. For this, we use a classic and extremely simple algorithm in online learning, known as Follow the Leader $( F T L )$ (see [Han57, CBL06]). At each time step t, this algorithm makes a prediction that would have been the best fixed response in hindsight to the $t - 1$ rounds completed so far.

In order to analyze this algorithm, it is helpful to introduce two quantities, which we call the entropy and divergence, that can be defined for any loss function ℓ:

$$
\begin{array} { r l r } & { } & { h _ { \ell } ( p ) = \underset { a } { \mathrm { i n f } } \ \ell ( p , a ) , \eqno ( \mathrm { e n t r o p y } ) } \\ & { } & { D _ { \ell } ( p , a ) = \ell ( p , a ) - h _ { \ell } ( p ) . \hphantom { \rule { 0 ex } { 0 ex } } } & { \qquad \mathrm { ( d i v e r g e n c e ) } } \end{array}
$$

The entropy is a concave function measuring the irreducible or inherent loss when the true distribution is given by $p .$ The divergence is a nonnegative function measuring the excess loss of using a suboptimal action $a . ^ { 6 }$ The two parts of this decomposition of the loss into $\ell ( p , a ) = h _ { \ell } ( p ) + D _ { \ell } ( p , a )$ are sometimes called aleatoric uncertainty and epistemic uncertainty $[ \mathrm { A G G ^ { + } 2 5 } ]$ . For example, $\ell _ { 1 } ( y , \eta ) = | y - \eta |$ has entropy $h _ { \ell _ { 1 } } ( p ) = 1 - | p |$ . For more examples, see Section 2.

It turns out that FTL has ${ \cal O } ( \log T )$ regret if the loss function’s entropy $h _ { \ell }$ is smooth, which is roughly equivalent to the requirement that $\partial \ell \circ k _ { \ell }$ is Lipschitz. In fact, using a weighted variant of FTL, one can improve this to $O ( 1 )$ regret. For non-smooth entropy functions like $h _ { \ell _ { 1 } }$ , we use a smoothing trick based on the addition of a small amount of random noise to transfer our results from the smooth-entropy case. For more details, see the proof of our main upper bound in Section 3.

We show that, conveniently, the FTL update rule also works for online smooth calibration, which we again transfer to full calibration with a noise trick. To our knowledge, this use case of Follow the Leader was not previously known. Concretely, we fix $0 < \lambda \leq 1$ and have the Predictor play

$$
q _ { t } = ( 1 - \lambda ) q _ { t - 1 } + \lambda r _ { t - 1 } .
$$

By modifying the proof of the familiar regret bound for Follow the Leader, we show that this algorithm achieves small smooth calibration error:

$$
\operatorname* { m a x } _ { 1 \mathrm { - L i p s c h i t z } } \mathrm { C E } _ { w } \big ( \mathrm { F o l l o w ~ t h e ~ L e a d e r } \big ) = O \Big ( \frac { 1 } { \lambda } + \lambda T \Big ) .
$$

If $\lambda = 1 / \sqrt { T }$ , then the online error is $O ( \sqrt { T } )$ . We emphasize that deterministic online algorithms for smooth calibration were previously known and that the $O ( \sqrt { T } )$ rate is known to be achievable [FH18, QZ24, ACRS25] and in fact optimal [GKS26]. We choose to use Follow the Leader for its explicit and straightforward handling of real-valued responses $r _ { t } \in [ \pm 1 ]$ from Nature, where previous results generally considered only discrete {±1}-valued responses.

## 1.4.3 Lower Bounds and Derandomization

All of our upper bounds based on Algorithms 2 and 3 make $T = \mathrm { p o l y } ( 1 / \varepsilon )$ queries of granularity $\gamma$ to a $( \mathcal { C } , \gamma \varepsilon )$ -conditional multiaccuracy oracle, which is in turn implemented using $1 / \gamma ^ { 2 } \varepsilon$ calls to an ordinary (C, γε)-multiaccuracy oracle. Although these reductions are eficient, they raise the question of whether the number of oracles calls can be improved further. In Section 4, we construct several instances of classes C and distributions over pairs $( { \pmb x } , { \pmb y } )$ for which many calls to a conditional multiaccuracy oracle are required to achieve agnostic learning. Some of these instances are specific to our particular oracle algorithm, while others are not. For more detail, see the discussion in Section 4 and Lemmas 4.3 to 4.5.

In some cases, our oracle reduction outputs a randomized predictor, partly due to mixing across a sequence of predictors and partly due to explicit noise injection into the predictor’s output. We show in Section 5 that these predictors can be derandomized in certain cases, either by avoiding adding the noise to begin with (Theorem 5.2) or by fixing the random coin tosses in advance (Theorem 5.7).

## 1.5 Related Work

As discussed, the notion of agnostic learning dates back to the work of [Hau92, KSS94]. The related notion of weak agnostic learning, as well as techniques for agnostic boosting, play an important role in this literature [BDLM01, MM02, KMV08, KK09, Fel10]. In particular, these techniques arise in the proof of the complexity-theoretic regularity lemma, or equivalently, the construction of multiaccurate predictors, as discussed earlier [TTV09, HKRR18, KGZ19]. These works are also responsible for initiating the now quite substantial line of work on multicalibration. Especially relevant to us in this body of work are the notions of outcome indistinguishability [DKR<sup>+</sup>21] and omniprediction, which concerns the simultaneous optimization of multiple loss functions [GKR<sup>+</sup>22, GHK<sup>+</sup>23, DLLT23, GJRR24, LSS25, DHI<sup>+</sup>25, GT26, OKK25]. Besides their relevance to omniprediction and other machine learning tasks, multiaccuracy and multicalibration have recently seen a surprisingly rich collection of applications to complexity theory and cryptography [DLLT23, HLNR23, CDV24, MPV25, CGKR25, DT26].

Our work can be seen as the culmination of a line of research starting in [HKRR18, GKR<sup>+</sup>22], and continued in $[ \mathrm { G H K ^ { + } 2 3 }$ , GKR23, OKK25, DLLT23, CGKR25, $\mathrm { C G K ^ { + } 2 6 ] }$ that aims to understand the learning power implied by multigroup fairness notions of varying strength. For a fixed class ${ \mathcal { C } } ,$ , we know that multicalibration yields omniprediction $\mathrm { [ G K R ^ { + } 2 2 }$ , GKR23] for convex losses, while calibrated multiaccuracy yields agnostic learning $[ \mathrm { G H K ^ { + } 2 3 }$ , CGKR25], both via one-shot reductions that post-process the predictions. One-shot multiaccuracy is a much weaker learning primitive, that does not yield (even) weak agnostic learning [CGKR25]. With oracle access to multiaccuracy and adaptive queries over multiple rounds, we show that one recovers the full power of omniprediction and multicalibration. [GHN26] recently proposed $\mathrm { a }$ “feature-augmented” multicalibration algorithm that avoids discretization during boosting, which uses a squared-loss minimization oracle over an augmented concept class. A call to such an oracle is stronger than a call to the multiaccuracy oracle we assume, given that any function which beats random guessing in squared error already gives a weak agnostic learner [CGKR25].

Turing formulates the imitation game as a game between a machine and a human, where winning the game means convincing the distinguisher that you are the human [Tur50]. Turing considers the possibility that a machine might win the imitation game using a policy that is diferent from imitating the behavior of a human. He considers this unlikely, and proceeds by explicitly assuming that the best strategy is indeed imitation. Recent studies cast doubt on this assumption, by showing that LLMs are sometimes more likely to be labeled human than humans themselves, which suggests that imitation is not the best strategy [JB26]. However, the name imitation suggests that perhaps a revised goal of the imitation game ought to be indistinguishability, for which there is no better strategy than imitation. The underlying question of what learning goals imitation achieves is what drives our work.

Computational Complexity of Multiaccuracy. Our main result (Theorem 1.1) establishes that multiaccurate prediction and agnostic learning have essentially equivalent computational complexity in the fixed-distribution setting. One direction of this equivalence was established by [HKRR18], who showed that agnostic learning implies multiaccurate prediction via an eficient reduction. Our converse reduction shows that computational lower bounds for agnostic learning also apply to multiaccurate prediction.

This equivalence is particularly informative in the statistical query (SQ) model [Kea98], where the complexity of agnostic learning is closely connected to polynomial approximation. The seminal work of [KKMS08] showed that low-degree polynomial approximators under the underlying feature distribution yield eficient agnostic learning algorithms, extending the classical result of [LMN93] for the realizable (noise-free) setting. Conversely, [DSFT<sup>+</sup>15] established that polynomial approximation degree characterizes the SQ complexity of agnostic learning for a broad family of concept classes under the uniform distribution on the Boolean hypercube. This characterization was subsequently extended to Gaussian feature distributions by [DKPZ21]. Since our reduction can equally be implemented in the SQ framework, these characterizations now extend to multiaccurate prediction. For a short survey of quantitative bounds on the complexity of agnostic learning obtained by combining these characterizations with results from approximation theory, see, e.g., [Sta26, Appendix C].

## 1.6 AI Methodology

Some steps of this paper’s proofs were incrementally accrued through repeated rounds of interactive discussion between the authors and ChatGPT-5.6 Sol. The model suggested conditioning by level sets, which we do in Step 1, as well as preliminary versions of Step 2, and preliminary versions of the hard inputs constructed in Section 4. The model’s provided version of Step 2 initially had some issues: it required exponentially many oracle calls, was specific to the $\ell _ { 1 }$ loss, and had a somewhat opaque proof using a mysterious potential function. With further prompting, the model was able to bypass the first issue. The authors addressed the second issue with the introduction of Step $^ { 3 , }$ which used tools from convex analysis to generalize the proof to loss functions besides $\ell _ { 1 }$ and also achieve omniprediction and swap multicalibration. The authors addressed the third issue using a connection to online learning, which significantly shortened, simplified, and modularized the proof. Similarly, the authors simplified and clarified the examples used in Section 4, the model’s versions of which were more complicated than necessary (e.g. three-dimensional instead of two-dimensional).

Acknowledgements. KS would like to thank Arsen Vasilyan for useful discussions regarding the derandomization procedure presented in Section 5.2.

## 2 Preliminaries

Throughout this paper, we write variables in boldface to indicate that they are random variables, such as a random feature vector $\textbf { \em x } \in ~ \mathcal { X }$ or a randomized predictor $p : \mathcal { X }  [ \pm 1 ]$ . We use the abbreviations $\{ \pm 1 \} = \{ - 1 , 1 \}$ and $[ \pm 1 ] = [ - 1 , 1 ]$

Definition 2.1 (Convexity). A function $f : \mathbb { R } \to \mathbb { R }$ is convex if

$$
f ( \lambda x + ( 1 - \lambda ) y ) \leqslant \lambda f ( x ) + ( 1 - \lambda ) f ( y )
$$

for all $x , y \in \mathbb { R }$ and $\lambda \in [ 0 , 1 ]$ . We say that $g : \mathbb { R }  \mathbb { R }$ is a subderivative of $f$ if

$$
f ( y ) \geqslant f ( x ) + g ( x ) ( y - x )
$$

for all $x , y \in \mathbb { R }$ . Similarly, we say that $- f$ is concave and that $- g$ is a superderivative of $- f$ Definition 2.2 (Bregman Divergence). Given a convex function $f$ and subderivative $f ^ { \prime }$ , let

$$
D _ { f } ( y , x ) = f ( y ) - f ( x ) - f ^ { \prime } ( x ) ( y - x ) .
$$

Note that $f$ is β-smooth if and only if $D _ { f } ( y , x ) \leqslant ( \beta / 2 ) ( y - x ) ^ { 2 }$ . If h is a concave function, we sometimes write $D _ { h }$ instead of $D _ { - h }$ when the meaning is clear from context.<sup>7</sup>

Definition 2.3 (Conjugacy). The convex conjugate of a function $f : \mathbb { R } \to \mathbb { R }$ is

$$
{ \hat { f } } ( y ) = \operatorname* { s u p } _ { x } x y - f ( x ) .
$$

It is known that the biconjugate $\hat { \hat { f } }$ is the greatest convex lower-semicontinuous lower bound on $f .$

Definition 2.4 (Loss Function Miscellany). Given a loss function $\ell : \{ \pm 1 \} \times \mathcal { A }  \mathbb { R }$ , let

$$
\ell ( p , a ) = \underset { { \pmb y } \sim p } { \mathbb { E } } [ \ell ( { \pmb y } , a ) ] ,
$$

(expected loss)

$$
k _ { \ell } ( p ) \in \mathop { \mathrm { a r g m i n } } _ { a } \ell ( p , a ) ,
$$

(best response)

$$
h _ { \ell } ( p ) = \operatorname* { i n f } _ { a } \ \ell ( p , a ) ,
$$

(entropy)

$$
D _ { \ell } ( p , a ) = \ell ( p , a ) - h _ { \ell } ( p ) .
$$

(divergence)

Example 2.5 (Entropy Functions).

• (Absolute Loss) The loss $\ell _ { 1 } ( y , \eta ) = | y - \eta |$ has a piecewise linear entropy

$$
h _ { \ell _ { 1 } } ( p ) = 1 - | p | .
$$

• (Squared Loss) The loss $\ell _ { 2 } ( y , \eta ) = ( y - \eta ) ^ { 2 } / 2$ has a quadratic entropy

$$
h _ { \ell _ { 2 } } ( p ) = { \frac { 1 } { 2 } } ( 1 - p ^ { 2 } ) .
$$

Also, $D _ { \ell _ { 2 } } ( p , \eta ) = ( p - \eta ) ^ { 2 } / 2$ is squared distance.

• (Logistic Loss) The loss $\ell ( y , \eta ) = \log ( 1 + e ^ { - 2 y \eta } )$ has (rescaled) Shannon entropy

$$
h _ { \ell } ( p ) = \frac { 1 + p } { 2 } \cdot \log \Bigl ( \frac { 2 } { 1 + p } \Bigr ) + \frac { 1 - p } { 2 } \cdot \log \Bigl ( \frac { 2 } { 1 - p } \Bigr ) .
$$

Definition 2.6 (Proper Loss $[ \mathrm { M c C 5 6 } , \mathrm { S a v 7 1 } ] )$ ). A loss function ℓ is proper if $\mathcal { A } = [ \pm 1 ]$ and

$$
\ell ( p , q ) \geqslant \ell ( p , p )
$$

for all $p , q \in [ \pm 1 ]$ . Equivalently, there is some superderivative $h _ { \ell } ^ { \prime }$ of $h _ { \ell }$ such that

$$
\ell ( p , q ) = h _ { \ell } ( q ) + h _ { \ell } ^ { \prime } ( q ) ( p - q ) .
$$

Equivalently, $D _ { \ell } ( p , q )$ is a Bregman divergence of $h _ { \ell }$ . Equivalently, $\ell ( p , p ) = h _ { \ell } ( p )$

Definition 2.7 (Natural Parametrization). We let $\mathcal { L } _ { B }$ be the class of loss functions ℓ with $\mathcal { A } = \mathbb { R }$ and $\partial \ell ( \eta ) = - \eta$ and $[ \pm B ]$ -bounded best responses $k _ { \ell } ( p )$ for all $p \in [ \pm 1 ]$ . We let $\mathcal { L } _ { B , \beta } \subseteq \mathcal { L } _ { B }$ further require $k _ { \ell } ( p )$ to be β-Lipschitz in $p \in [ \pm 1 ]$

Example 2.8 (Natural Parametrization Examples).

• (Absolute Loss) If we define $\ell _ { 1 } ( y , \eta ) = | y - \eta |$ for $y \in \{ \pm 1 \}$ , then for $\eta \in [ \pm 1 ]$ , we have

$$
\ell _ { 1 } ( p , \eta ) = 1 - p \eta .
$$

This can be extended to all $\eta \in \mathbb { R }$ as follows:

$$
\ell ( p , \eta ) = \operatorname* { m a x } ( 1 , | \eta | ) - p \eta .
$$

This extension is naturally parametrized and has $k _ { \ell } ( p ) = \mathrm { s i g n } ( p )$ as a [±1]-valued bestresponse (note that $\mathrm { s i g n } ( 0 ) \in [ \pm 1 ]$ may be defined arbitrarily).

• (Squared Loss) If we define $\ell _ { 2 } ( y , \eta ) = ( y - \eta ) ^ { 2 } / 2$ for $y \in \{ \pm 1 \}$ , then

$$
\ell _ { 2 } ( p , \eta ) = \frac { 1 + \eta ^ { 2 } } { 2 } - p \eta .
$$

This is naturally parametrized with $k _ { \ell _ { 2 } } ( p ) = p$ as $\mathrm { ~ a ~ } [ \pm 1 ]$ -valued best response.

• (Logistic Loss) If we define $\ell ( y , \eta ) = \log ( 1 + e ^ { - 2 y \eta } )$ for $y \in \{ \pm 1 \}$ , then

$$
\ell ( p , \eta ) = \log ( e ^ { \eta } + e ^ { - \eta } ) - p \eta .
$$

This is naturally parametrized with best-response $k _ { \ell } ( p ) = \mathrm { t a n h } ^ { - 1 } ( p )$ , which can be made $[ \pm B ]$ -valued by modifying $\mathrm { o g } ( e ^ { \eta } + e ^ { - \eta } )$ to instead be linear with slope −1 for $\eta < - B$ and linear with slope 1 for $\eta > B$

Definition 2.9 (Calibration Error [Daw85, GKSZ22]). Given a class of weight functions $\mathcal { W } \subseteq$ $\{ [ \pm 1 ]  [ \pm 1 ] \}$ , the weight-restricted calibration error of a predictor $p : \mathcal { X }  [ \pm 1 ]$ with respect to W (and a distribution over labeled data points ${ \pmb x } , { \pmb y } )$ is

$$
\operatorname { C E } _ { \mathcal { W } } ( p ) = \operatorname* { m a x } _ { w \in \mathcal { W } } \mathbb { E } [ w ( p ( \pmb { x } ) ) ( \pmb { y } - p ( \pmb { x } ) ) ] .
$$

When W contains only the identity function, we instead write $\mathrm { C E _ { 2 } } .$ , called degree-2 calibration error.   
When W contains all 1-Lipschitz functions, we instead write smCE, called smooth calibration error.   
When W contains all functions, we instead write ECE, called expected calibration error.

Definition 2.10 (Pseudodimension [Hau92]). The pseudodimension of a class ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to \mathbb { R } \}$ is the VC dimension of the following class over the domain $\mathcal { X } \times \mathbb { R }$

$$
\left\{ ( x , y ) \mapsto \mathbf { 1 } [ c ( x ) > y ] | c \in { \mathcal { C } } \right\} .
$$

## 3 Upper Bounds

In this section, we analyze our main oracle reduction from agnostic learning to multiaccuracy. As discussed, our reduction also achieves several nominally stronger guarantees, such as (swap) omniprediction and (swap) multicalibration, in one fell swoop. We follow the template outlined in the technical overview in Section 1.4.

## 3.1 Reweighting and Conditioning

As described in the introduction, the first step of the reduction, described in Algorithm 1, allows us to transform a multiaccuracy oracle for the feature distribution $\mathcal { D } _ { \mathcal { X } }$ into a conditional multiaccuracy oracle with respect to an arbitrary function $f : \mathcal { X }  \mathcal { V } .$ . Theorem 1.2 states the guarantee for this step of the algorithm. To prove it, we first prove the following, slightly more general theorem.

Theorem 3.1. Algorithm 1 with $a \left( \mathcal { C } , \varepsilon \right)$ -multiaccuracy oracle and $T = O ( 1 / \varepsilon )$ outputs p satisfying

$$
\operatorname* { m a x } _ { c \in \mathcal { C } } \Big | \mathbb { E } \big [ m ( \pmb { x } ) \cdot c ( \pmb { x } ) ( \pmb { y } - p ( \pmb { x } ) ) \big ] \Big | \leqslant O ( \varepsilon ) .
$$

If the oracle has failure probability δ, then Algorithm 1 has failure probability at most $T \delta$

Proof. At each step $t \leqslant T$ , the $( \mathcal { C } , \varepsilon )$ -multiaccuracy oracle returns $p _ { t }$ satisfying

$$
\operatorname* { m a x } _ { c \in { \mathcal { C } } } \left| \mathbb { E } [ m ( { \pmb x } ) \cdot c ( { \pmb x } ) ( { \pmb y } - p _ { t } ( { \pmb x } ) ) ] + \mathbb { E } [ ( 1 - m ( { \pmb x } ) ) \cdot c ( { \pmb x } ) ( p _ { t - 1 } ( { \pmb x } ) - p _ { t } ( { \pmb x } ) ) ] \right| \leqslant \varepsilon .\tag{7}
$$

In particular, for any fixed $c \in { \mathcal { C } }$ and for every $t \leq T$ , we have,

$$
\mathbb { E } [ m ( { \pmb x } ) \cdot c ( { \pmb x } ) ( { \pmb y } - p _ { t } ( { \pmb x } ) ) ] + \mathbb { E } [ ( 1 - m ( { \pmb x } ) ) \cdot c ( { \pmb x } ) ( p _ { t - 1 } ( { \pmb x } ) - p _ { t } ( { \pmb x } ) ) ] \leqslant \varepsilon .
$$

Let $\bar { p } _ { T } = ( p _ { 1 } + \cdot \cdot \cdot + p _ { T } ) / T$ . Taking the average of the above inequality over all time steps $t \leqslant T$ causes the second term on the left-hand side to telescope. This gives

$$
\mathbb { E } [ m ( { \pmb x } ) \cdot { \pmb c } ( { \pmb x } ) ( { \pmb y } - \bar { p } _ { T } ( { \pmb x } ) ) ] + \frac { 1 } { T } \mathbb { E } [ ( 1 - m ( { \pmb x } ) ) \cdot { \pmb c } ( { \pmb x } ) ( p _ { 0 } ( { \pmb x } ) - p _ { T } ( { \pmb x } ) ) ] \leqslant \varepsilon .\tag{8}
$$

Likewise, using Eq. (7) in the other direction, we can conclude that for any $c \in C$

$$
\mathbb { E } [ m ( { \pmb x } ) \cdot { \pmb c } ( { \pmb x } ) ( { \pmb y } - { \bar { p } } _ { T } ( { \pmb x } ) ) ] + \frac { 1 } { T } \mathbb { E } [ ( 1 - m ( { \pmb x } ) ) \cdot { \pmb c } ( { \pmb x } ) ( p _ { 0 } ( { \pmb x } ) - p _ { T } ( { \pmb x } ) ) ] \geqslant - \varepsilon .\tag{9}
$$

As Eqns. (8) and (9) apply to each $c \in { \mathcal { C } }$ , rearranging the terms and using the triangle inequality, we conclude that

$$
\operatorname* { m a x } _ { c \in \mathcal { C } } \left| \mathbb { E } [ m ( \pmb { x } ) \cdot c ( \pmb { x } ) ( \pmb { y } - \bar { p } _ { T } ( \pmb { x } ) ) ] \right| \leqslant \varepsilon + \frac { 1 } { T } .
$$

Proof of Theorem 1.2. Apply Theorem 3.1 to each $m = { \bf 1 } _ { f ^ { - 1 } ( v ) }$ for each $v \in f ( \mathcal { X } )$

## 3.2 Achieving Agnostic Learning

We now analyze our reduction from agnostic learning to (conditional) multiaccuracy, described in Algorithm 2. As explained in the technical overview, the key tools that we use are the bounds on the online regret and online calibration error of Follow the Leader. We also use some smoothing tricks along the way. We defer these helper lemmas on FTL and smoothing to Section 3.4. We actually prove the following, stronger version of Theorem 1.3, which handles not just $\ell _ { 1 }$ loss but any particular naturally parametrized loss satisfying mild conditions:

Theorem 3.2. If ℓ is naturally parametrized and convex in its second argument with an $O ( 1 )$ bounded and O(1)-Lipschitz best response function $k _ { \ell ; \ell }$ , then Algorithm 2 with a $( \mathcal { C } , \gamma \varepsilon )$ -conditional multiaccuracy oracle, the function $f = k _ { \ell }$ , grid size $\gamma = \varepsilon _ { \mathrm { { ; } } }$ , and $T = O ( 1 / \varepsilon )$ outputs p satisfying

$$
\mathbb { E } [ \ell ( \pmb { y } , p ( \pmb { x } ) ) ] \leqslant \operatorname* { m i n } _ { { \boldsymbol { c } } \in \mathcal { C } } \mathbb { E } [ \ell ( \pmb { y } , { \boldsymbol { c } } ( \pmb { x } ) ) ] + O ( \varepsilon ) .
$$

Even if $k _ { \ell }$ is not $O ( 1 )$ -Lipschitz, a modified choice of f and slightly longer sequence of $T = { \cal O } ( 1 / \varepsilon ^ { 2 } )$ calls sufice to achieve the inequality above. In both cases, if the oracle has failure probability $\delta ,$ then Algorithm 2 has failure probability at most Tδ.

In the statement of Theorem 3.2, we also assumed that the loss function is convex, but we emphasize that this is $o n l y$ for simplicity. The two real assumptions are (1) natural parametrization and (2) bounded best responses. If the loss function satisfies these two assumptions but is nonconvex, the error rates in Theorem 3.2 still hold verbatim, but for a modified version of the algorithm that replaces the final averaging over iterates $t \in [ T ]$ with a random mixture over $t \in [ T ]$ and additional random noise (hence, the modified algorithm for a nonconvex loss outputs a randomized predictor). This will be apparent from the following proof.

Proof of Theorems 1.3 and 3.2. Let $\pmb { t } \in [ T ]$ be a random time index with $\operatorname* { P r } [ t = s ] \propto s$ . Let $q _ { t }$ and $r _ { t }$ denote, respectively, the queries to and responses from the conditional multiaccuracy oracle in Algorithm 2. Suppose first that $k _ { \ell }$ is L-Lipschitz and set $f = k _ { \ell }$ . By Lemmas 3.4 and 3.5, with linear weights $a _ { t } = t ,$ , we have the following regret bound, which holds pointwise for all $x \in \mathcal { X }$

$$
\mathbb { E } [ \ell ( r _ { t } ( x ) , ( k _ { \ell } \circ q _ { t } ) ( x ) ) ] \leqslant \operatorname* { m i n } _ { c : \mathcal { X } \to [ \pm 1 ] } \mathbb { E } [ \ell ( r _ { t } ( x ) , c ( x ) ) ] + O \Big ( \frac { L } { T } \Big ) .
$$

Next, we apply indistinguishability, as described in the technical overview. Indeed, since ℓ is naturally parametrized, we have $\ell ( r , a ) - \ell ( y , a ) = a ( y - r )$ . Therefore, on the left-hand side, where the predictor is $k _ { \ell } \circ q _ { t }$ , we may replace $r _ { t } ( \pmb { x } )$ with y by multiaccuracy conditional on $R _ { \gamma } \circ k _ { \ell } \circ q _ { t }$ while introducing error ${ \cal O } ( \varepsilon + \gamma ) . ^ { 8 }$ On the right-hand side, we may similarly replace $r _ { t } ( \pmb { x } )$ with y whenever $c \in { \mathcal { C } }$ while introducing error $O ( \varepsilon )$ . Ultimately, these substitutions yield

$$
\underset { t , x , y } { \mathbb { E } } [ \ell ( { \pmb y } , ( k _ { \ell } \circ q _ { \ell } ) ( { \pmb x } ) ) ] \leqslant \underset { c \in \mathcal { C } } { \operatorname* { m i n } } \ \underset { t , x , y } { \mathbb { E } } [ \ell ( { \pmb y } , c ( { \pmb x } ) ) ] + O \Big ( \frac { L } { T } + \gamma + \varepsilon \Big ) .
$$

In the special case that $L = O ( 1 )$ , the error terms on the right-hand side are $O ( \varepsilon )$ when we set $\gamma = \varepsilon$ and $T = 1 / \varepsilon$ . Otherwise, we apply Lemma 3.8 to obtain an approximation $\ell _ { \sigma }$ of ℓ for which $k _ { \ell _ { \sigma } }$ is $O ( 1 / \sigma )$ -Lipschitz. Letting $f = k _ { \ell _ { \sigma } }$ and applying the approximation guarantee from Lemma 3.8,

$$
\underset { t , x , y } { \mathbb { E } } [ \ell ( y , ( k _ { \ell } \circ N _ { \sigma } \circ q _ { t } ) ( x ) ) ] \leqslant \underset { c \in \mathcal { C } } { \operatorname* { m i n } } \ \underset { t , x , y } { \mathbb { E } } [ \ell ( y , c ( x ) ) ] + O \Big ( \frac { 1 } { \sigma T } + \sigma + \gamma + \varepsilon \Big ) .
$$

The error terms on the right-hand side are $O ( \varepsilon )$ when we set $\sigma = \gamma = \varepsilon$ and $T = 1 / \varepsilon ^ { 2 }$

Finally, suppose in addition that ℓ is convex in its second input. Then, in the smooth case, Jensen’s inequality allows us to move the expectation over t as follows:

$$
\underset { x , y } { \mathbb { E } } \left[ \ell \big ( y , \underset { t } { \mathbb { E } } [ ( k _ { \ell } \circ q _ { t } ) ( x ) ] \big ) \right] \leqslant \underset { c \in \mathcal { C } } { \operatorname* { m i n } } \ \underset { t , x , y } { \mathbb { E } } [ \ell ( y , c ( x ) ) ] + O \Big ( \frac { L } { T } + \gamma + \varepsilon \Big ) .
$$

Similarly, in the nonsmooth case, Jensen’s inequality allows us to move both the expectation over t and the expectation over the random noise:

$$
\underset { x , y } { \mathbb { E } } \left[ \ell \big ( y , \underset { t } { \mathbb { E } } [ ( k _ { \ell \sigma } \circ q _ { t } ) ( \pmb { x } ) ] \big ) \right] \leqslant \underset { c \in \mathcal { C } } { \operatorname* { m i n } } \ \underset { t , x , y } { \mathbb { E } } [ \ell ( \pmb { y } , c ( \pmb { x } ) ) ] + O \Big ( \frac { 1 } { \sigma T } + \sigma + \gamma + \varepsilon \Big ) .
$$

## 3.3 Achieving Omniprediction and Multicalibration

Finally, we show how to go beyond agnostic learning to (smooth) omniprediction and (smooth) swap multicalibration, as described in Algorithm 3, while still only accessing the data distribution through a multiaccuracy oracle. The proof of Theorem 1.4, which analyzes Algorithm 3, follows roughly the same template as the previous proof, with a few additional steps.

Proof of Theorem $1 . 4 \ ( O m n i p r e d i c t i o n )$ . As in the preceding proof, let $t \in [ T ]$ be a random time index with $\mathrm { P r } [ t = s ]$ ∝ s, let $q _ { t }$ and $r _ { t }$ denote the queries to and responses from the oracle, and suppose that $k _ { \ell }$ is L-Lipschitz. (As usual, we also assume that ℓ is naturally parametrized and that $k _ { \ell }$ is [±1]-valued.) As in the previous proof, we apply Lemmas 3.4 and 3.5 pointwise, which gives the following regret bound for linear weights $a _ { t } = t \colon$

$$
\mathbb { E } [ \ell ( r _ { t } ( x ) , ( k _ { \ell } \circ q _ { t } ) ( x ) ) ] \leqslant \operatorname* { m i n } _ { c : \mathcal { X } \to [ \pm 1 ] } \mathbb { E } [ \ell ( r _ { t } ( x ) , c ( x ) ) ] + O \Big ( \frac { L } { T } \Big ) .
$$

Next, we apply indistinguishability as in the previous proof, but with some slight modifications. Once again, since ℓ is naturally parametrized, we have $\ell ( r , a ) - \ell ( y , a ) = a ( y - r )$ . Therefore, on the left-hand side, where the predictor is $k _ { \ell } \circ q _ { t }$ , we may replace $r _ { t } ( \pmb { x } )$ with y by multiaccuracy conditional on the (loss-independent) quantity $R _ { \gamma } \circ q _ { t }$ while introducing error $O ( \varepsilon + L \gamma )$ . Here, we have used the fact that $k _ { \ell }$ is L-Lipschitz. On the right-hand side, we may similarly replace $r _ { t } ( \pmb { x } )$ with y while introducing error $O ( \varepsilon )$ whenever $c \in { \mathcal { C } }$ . Ultimately, these substitutions yield

$$
\underset { t , x , y } { \mathbb { E } } [ \ell ( { \pmb y } , ( k _ { \ell } \circ q _ { \ell } ) ( { \pmb x } ) ) ] \leqslant \operatorname* { m i n } _ { c \in \mathcal { C } } \ \underset { t , x , y } { \mathbb { E } } [ \ell ( { \pmb y } , c ( { \pmb x } ) ) ] + O \Big ( \frac { L } { T } + L \gamma + \varepsilon \Big ) .
$$

In the special case that $L = O ( 1 )$ , the error terms on the right-hand side are $O ( \varepsilon )$ when we set $\gamma = \varepsilon$ and $T = O ( 1 / \varepsilon )$ . Otherwise, we apply Lemma 3.8 to obtain an approximation $\ell _ { \sigma }$ of ℓ for which $k _ { \ell _ { \sigma } }$ is $O ( 1 / \sigma )$ -Lipschitz. Applying the approximation guarantee from Lemma 3.8,

$$
\mathbb { E } _ { t , x , y , N _ { \sigma } } [ \ell ( { \pmb y } , ( k _ { \ell } \circ N _ { \sigma } \circ q _ { t } ) ( { \pmb x } ) ) ] \leqslant \operatorname* { m i n } _ { c \in { \mathcal { C } } } \ \mathbb { E } _ { t , x , y } [ \ell ( { \pmb y } , c ( { \pmb x } ) ) ] + O \Big ( \frac { 1 } { \sigma T } + \frac { \gamma } { \sigma } + \sigma + \varepsilon \Big ) .
$$

The error terms on the right-hand side are all $O ( \varepsilon )$ when we set $\sigma = \varepsilon$ and $\gamma = \varepsilon ^ { 2 }$ and $T = 1 / \varepsilon ^ { 2 }$ Since the loss function ℓ was arbitrary and no step of the algorithm depended on its choice, we conclude that Algorithm 3 achieves omniprediction. □

Proof of Theorem 1.4 (Multicalibration). Departing slightly from the previous proof, let $\pmb { t } \in [ T ]$ be uniform and let $q _ { t }$ and $r _ { t }$ denote the queries to and responses from the oracle with geometrically increasing weights $a _ { t } / A _ { t } = \lambda$ for $t \geqslant 2$ . By Lemma 3.7, our online calibration error bound, we have the following inequality for all $x \in \mathcal { X }$ whenever the weight function w : $: [ \pm 1 ]  [ \pm 1 ]$ is L-Lipschitz:

$$
\big | \mathbb { E } [ w ( q _ { t } ( x ) ) ( r _ { t } ( x ) - q _ { t } ( x ) ) ] \big | \leqslant O \Big ( \frac { 1 } { \lambda T } + L \lambda \Big ) .
$$

By applying the same indistinguishability argument as in the preceding two proofs, we see that for all $c \in \mathcal { C } ^ { \mathrm { { s w a p } } }$ such that $c ( \cdot , x )$ is L-Lipschitz for all $x \in \mathcal { X }$ and $c ( v , \cdot ) \in \mathcal { C }$ for all $v \in [ \pm 1 ]$ ，

$$
\big | \operatorname { \frac { \mathbb { E } } { \hbar , x , y } } _ { t , x , y } [ c ( q _ { t } ( x ) , x ) ( y - q _ { t } ( x ) ) ] \big | \leqslant O \Big ( \frac { 1 } { \lambda T } + L \lambda + L \gamma + \varepsilon \Big ) .
$$

Once again, note that there is an $O ( L \gamma )$ term that arises because we round $q _ { t }$ to multiples of $\gamma$ before calling the conditional multiaccuracy oracle.

In the case of smooth swap multicalibration, where $L = O ( 1 )$ , the error terms on the righthand side are $O ( \varepsilon )$ when we set $\lambda = \gamma = \varepsilon$ and $T = 1 / \varepsilon ^ { 2 }$ . For (full) swap multicalibration, we apply Lemma 3.9, which uses the noise operator $N _ { \sigma }$ to mimic the efect of smoothing an initially nonsmooth weight function. Substituting $L = O ( 1 / \sigma )$ , the lemma implies that

$$
\left| \mathbb { E } [ c ( N _ { \sigma } ( q _ { t } ( x ) ) , x ) ( y - N _ { \sigma } ( q _ { t } ( x ) ) ) ] \right| \leqslant O \Bigl ( \frac { 1 } { \lambda T } + \frac { \lambda } { \sigma } + \frac { \gamma } { \sigma } + \sigma + \varepsilon \Bigr ) .
$$

The error terms on the right-hand side are $O ( \varepsilon )$ when we set $\sigma = \varepsilon , \lambda = \gamma = \varepsilon ^ { 2 }$ , and $T = 1 / \varepsilon ^ { 3 }$ □

## 3.4 Deferred Lemmas on FTL and Smoothing

We now state the various helper lemmas that we used in the preceding sections. Some of these lemmas are essentially standard results in the literature with known proofs, but others are genuinely new. We defer all proofs to Appendix A.

We begin with a helpful lemma relating the entropy function $h _ { \ell }$ and best-response function $k _ { \ell }$ of a given loss function $\ell .$ In particular, the lemma will imply that Lipschitzness of $\partial \ell \circ k _ { \ell }$ implies smoothness of $h _ { \ell } .$ . To state the lemma, recall that for a given loss function $\ell ,$ we define its associated entropy to be the concave function $h _ { \ell } ( p ) = \operatorname* { i n f } _ { a } \ell ( p , a )$ and its divergence to be the nonnegative function $D _ { \ell } ( p , a ) = \ell ( p , a ) - h _ { \ell } ( p )$

Lemma 3.3. If $k _ { \ell }$ is a best-response function of ℓ, then $h _ { \ell } ^ { \prime } = \partial \ell \circ k _ { \ell }$ is a superderivative of $h _ { \ell }$ and

$$
\begin{array} { r } { D _ { \ell } ( p , k _ { \ell } ( q ) ) = D _ { h _ { \ell } } ( p , q ) , } \end{array}
$$

where $D _ { h _ { \ell } }$ denotes the Bregman divergence from p to q with respect to $h _ { \ell } , h _ { \ell } ^ { \prime }$ (Definition $\it { 2 . 2 ) }$

Next, we present the standard regret bound for Follow the Leader. As described in the technical overview, Follow the Leader is an online algorithm which, at time t, performs the best action in hindsight for the $t - 1$ previous time steps. The regret bound is folklore in the literature, but since it is usually stated with rather diferent notation, we include a self-contained proof in our context for the reader’s convenience.

Lemma 3.4 (Follow the Leader’s Regret). Given a loss function ℓ, weights $w _ { t } \geqslant 0$ , distribution means $p _ { 1 } , \dotsc , p _ { T } \in [ \pm 1 ]$ , define the prefix sums $W _ { t } = w _ { 1 } + \cdot \cdot \cdot + w _ { t }$ and weighted averages $\bar { p } _ { t } =$ $( w _ { 1 } p _ { 1 } + \cdot \cdot \cdot + w _ { t } p _ { t } ) / W _ { t } \ ( l e t \bar { p } _ { 0 } = 0 )$ . In this setting, Follow the Leader, which in round t plays the best response $k _ { \ell } ( \bar { p } _ { t - 1 } )$ in hindsight to the first $t - 1$ rounds, achieves the following regret bound:

$$
\sum _ { t = 1 } ^ { T } w _ { t } \ell ( p _ { t } , k _ { \ell } ( \bar { p } _ { t - 1 } ) ) - \operatorname* { m i n } _ { a \in \mathcal { A } } \sum _ { t = 1 } ^ { T } w _ { t } \ell ( p _ { t } , a ) = \sum _ { t = 1 } ^ { T } W _ { t } D _ { \ell } ( \bar { p } _ { t } , k _ { \ell } ( \bar { p } _ { t - 1 } ) ) .
$$

While Lemma 3.4 gives a clean exact expression for the regret, there is no reason to believe that this regret is small a priori. The following lemma, which is also more or less standard up to changes in notation, outlines an important special case where this regret can be bounded.

Lemma 3.5. In the setting of Lemma $\ 3 . 4 \ i$ , if $\partial \ell \circ k _ { \ell }$ is β-Lipschitz, then the regret satisfies

$$
\sum _ { t = 1 } ^ { T } W _ { t } D _ { \ell } ( \bar { p } _ { t } , k _ { \ell } ( \bar { p } _ { t - 1 } ) ) \leqslant O \Bigl ( \beta \sum _ { t = 1 } ^ { T } \frac { w _ { t } ^ { 2 } } { W _ { t } } \Bigr ) .
$$

The preceding lemma bounded the regret of Follow the Leader in terms of a sum $\textstyle \sum _ { t } w _ { t } ^ { 2 } / W _ { t }$ depending on the chosen weights $w _ { t }$ . The next technical lemma shows that this bound is indeed small for many choices of the weights, such as uniform weights $w _ { t } = 1$ , linearly increasing weights $w _ { t } = t$ , and geometrically increasing weights $w _ { t }$ with $w _ { t } / W _ { t } = \lambda$ for all $t \geqslant 2$ . In each case, we normalize the weights so that they sum to $W _ { T } = 1$ . To our knowledge, the lemma is new, but nevertheless quite straightforward.

Lemma 3.6 (Step Sizes). Given $w _ { t } \geqslant 0$ , let $W _ { t } = w _ { 1 } + \cdot \cdot \cdot + w _ { t }$ and $\begin{array} { r } { \delta = \sum _ { t = 1 } ^ { T } w _ { t } ^ { 2 } / W _ { t } W _ { T } } \end{array}$ . Then,

• (Uniform) If $w _ { t } = 1$ for all $t \geqslant 1$ , then $\delta = O ( \log ( T ) / T )$

• (Linear) If $w _ { t } = t$ for all $t \geqslant 1$ , then $\delta = { \cal O } ( 1 / T )$

$$
\bullet ( G e o m e t r i c ) \ I f w _ { t } / W _ { t } = \lambda = \log ( T ) / T \ f o r \ t \geqslant 2 , \ t h e n \ \delta = \lambda + ( 1 - \lambda ) ^ { T } = O ( \log ( T ) / T ) .
$$

Now, we shift our attention to the online smooth calibration error of Follow the Leader, as opposed to its regret for a given loss function. As discussed, deterministic online algorithms for smooth calibration were previously known, but to our knowledge, the fact that such a simple algorithm as Follow the Leader achieves this goal was not previously known. The following lemma uses geometrically increasing weights.

Lemma 3.7 (Follow the Leader’s Calibration Error). Given a β-Lipschitz function w $: [ \pm 1 ]  [ \pm 1 ]$ and distribution means $p 1 , \ldots , p _ { T } \in [ \pm 1 ]$ , define the weighted Follow the Leader iterates $\bar { p } _ { 0 } = 0 .$ $\bar { p } _ { 1 } = p _ { 1 } , 0 < \lambda \leq 1$ , and $\bar { p } _ { t } = ( 1 - \lambda ) \bar { p } _ { t - 1 } + \lambda p _ { t }$ for $t = 2 , \ldots , T$ . Then, we have the following online calibration bound:

$$
\Big | \sum _ { t = 1 } ^ { T } w ( \bar { p } _ { t - 1 } ) ( p _ { t } - \bar { p } _ { t - 1 } ) \Big | \leqslant O \Big ( \frac { 1 } { \lambda } + \beta \lambda T \Big ) .
$$

Next, we prove some useful smoothing lemmas, which will allow us to transfer results for smooth omniprediction to full omniprediction, and similarly transfer results for smooth multicalibration to full multicalibration. This is relevant, in particular, for $\ell _ { 1 }$ loss’s non-Lipschitz best response, needed for our main result on agnostic learning.

For loss minimization, the smoothing operation that we study can be viewed in two equivalent ways: either as a Lipschitz approximation to the best-response function $k _ { \ell }$ or as a small addition of noise to the prediction itself. While the former view has the advantage of being deterministic, the second is more convenient for omniprediction since it is not specific to the loss ℓ under consideration.

Lemma 3.8 (Loss Smoothing). Let ℓ be a naturally parametrized loss function with $a \ [ \pm B ] - v a l u e d$ best-response function $k _ { \ell }$ . Then, for all $\sigma ^ { - 1 } \in \mathbb { N }$ , there exists a naturally parametrized loss function $\ell _ { \sigma }$ with the following two properties:

• (Smoothness) The best response function $k _ { \ell _ { \sigma } }$ is $[ \pm B ]$ -valued and $O ( B / \sigma )  – L i p s c h i t z$

• (Approximation) For all $p \in [ \pm 1 ]$ and $\eta \in \mathbb R$

$$
\ell _ { \sigma } ( p , \eta ) \leqslant \ell ( p , \eta ) + O ( B \sigma ) .
$$

Also, for all $p , q \in [ \pm 1 ]$ ，

$$
\left| \mathbb { E } [ \ell ( p , k _ { \ell } ( N _ { \sigma } ( q ) ) ) ] - \ell _ { \sigma } ( p , k _ { \ell _ { \sigma } } ( q ) ) \vert \right. \leqslant { \cal O } ( B \sigma ) .
$$

Concretely, $\ell _ { \sigma }$ and $k _ { \ell _ { \sigma } }$ are derived from ℓ and $k _ { \ell }$ by the formula

$$
k _ { \ell _ { \sigma } } ( p ) = \mathbb { E } [ k _ { \ell } ( N _ { \sigma } ( p ) ) ] .
$$

Here, $N _ { \sigma } : [ \pm 1 ]  [ \pm 1 ]$ is the (randomized) noise function that randomly rounds its input either up or down to the nearest multiple of σ while preserving its input’s value in expectation.

For example, starting from the $\ell _ { 1 }$ loss with best response function $\mathrm { s i g n } ( p )$ the smoothing lemma constructs a modified loss whose best response function is $\mathbb { E } [ \mathrm { s i g n } ( N _ { \sigma } ( p ) ) ]$ . If we use the convention $\mathrm { s i g n } ( 0 ) = 0$ , then this is precisely the piecewise linear function $\mathrm { s i g n } _ { \sigma } ( p )$ defined in the introduction, which agrees with $\mathrm { s i g n } ( p )$ when $| p | \geqslant \sigma$ and increases linearly from −1 to +1 over the interval $[ \pm \sigma ]$

Finally, we state our lemma for smoothing calibration weight functions by adding random noise. This lemma is essentially known in the recent literature [BN24, HWY25, GSTT26], but for slightly diferent noise functions than the one we consider. Our chosen noise function, which we denote by $N _ { \sigma }$ , has the advantage of keeping the predictions discrete-valued in a grid over the interval [±1].

Lemma 3.9 (Weight Smoothing). Suppose that p has calibration error $\varepsilon _ { L }$ for weight functions $c \in \mathcal { C } ^ { \mathrm { { s w a p } } }$ that are [±1]-valued and L-Lipschitz in their first input, meaning that

$$
\begin{array} { r } { \mathbb { E } [ c ( p ( { \pmb x } ) , { \pmb x } ) ( { \pmb y } - p ( { \pmb x } ) ) ] \leqslant \varepsilon _ { L } . } \end{array}
$$

Then, $N _ { \sigma } \circ p$ has calibration error $\varepsilon _ { O ( 1 / \sigma ) } + O ( \sigma )$ for arbitrary [±1]-valued weights $c \in { \mathcal { C } } ^ { \mathrm { s w a p } }$

## 4 Lower Bounds

In this section, we give examples showing that the number of calls that we make to the conditional multiaccuracy oracle is essentially tight in several important cases. We emphasize that our lower bounds are on the number of calls to a conditional multiaccuracy oracle. Proving lower bounds on the required number of calls to the underlying multiaccuracy oracle, as opposed to the conditiona multiaccuracy gadget, is an interesting direction for future work.

## 4.1 Lower Bounds for Omniprediction

We start by showing via the guiding example from Section 1.2 that the $O ( 1 / \varepsilon )$ upper bound for smooth omniprediction and the ${ \cal O } ( 1 / \varepsilon ^ { 2 } )$ upper bound for (full) omniprediction are both tight up to constant factors. As our argument will show, they are tight not only for our specific algorithm, but also for any algorithm whose queries and final output are restricted to be pointwise functions of the oracle responses of a conditional multiaccuracy oracle. In fact, this example rules out improvements even for $\ell _ { 2 }$ and $\ell _ { 1 }$ loss minimization, which are clearly prerequisites for smooth omniprediction and (full) omniprediction, respectively.

Theorem 4.1 (Large $\ell _ { 1 } , \ell _ { 2 } )$ . Consider the class of algorithms with access to a conditional multiaccuracy oracle, whose queries and final output are restricted to be pointwise functions of the oracle responses received so far. Among such algorithms, $\Omega ( 1 / \varepsilon )$ queries are required to achieve ε-approximate $\ell _ { 2 }$ loss minimization, and $\Omega ( 1 / \varepsilon ^ { 2 } )$ queries instead for $\ell _ { 1 }$ loss.

Proof. Recall the guiding example in Section 1.2, where the data distribution consists of uniform $n { \mathrm { - b i t } }$ features ${ \pmb x } \sim \{ \pm 1 \} ^ { n }$ for odd $n ,$ and binary labels ${ \pmb y } = { \pmb x } _ { n }$ , while C is the class of symmetric functions $c : \{ \pm 1 \} ^ { n }  [ \pm 1 ]$ , which are invariant under permutations of the input bits. The predictors $p _ { i } ( { \pmb x } ) = { \pmb x } _ { i }$ for $i \le n - 1$ are each independent of $^ { y , }$ and moreover the same is true for any post-processing function $f : [ \pm 1 ] ^ { n - 1 } \to [ \pm 1 ]$ used to combine these $n - 1$ predictors:

$$
\mathbb { E } [ { \pmb x } _ { n } f ( { \pmb x } _ { 1 } , . . . , { \pmb x } _ { n - 1 } ) ] = 0 .
$$

The $\ell _ { 1 }$ loss of any function $f$ of $p 1 , \ldots , p _ { n - 1 }$ is 1 since

$$
\begin{array} { r } { \mathbb { E } \big [ | { \pmb x } _ { n } - f ( { \pmb x } _ { 1 } , \ldots , { \pmb x } _ { n - 1 } ) | \big ] = 1 - \mathbb { E } [ { \pmb x } _ { n } f ( { \pmb x } _ { 1 } , \ldots , { \pmb x } _ { n - 1 } ) ] = 1 . } \end{array}
$$

On the other hand, the Majority function $\operatorname { M a j } _ { n }$ , which is clearly symmetric, achieves correlation $\Omega ( 1 / { \sqrt { n } } ) . ^ { 9 }$ Using the standard fact that its degree-1 Fourier coeficients are all $\Omega ( 1 / \sqrt { n } )$

$$
\mathbb { E } \big [ | { \pmb x } _ { n } - \mathrm { M a j } _ { n } ( { \pmb x } ) | \big ] = 1 - \Omega \Big ( \frac { 1 } { \sqrt { n } } \Big ) .
$$

Unless $n > \Omega ( 1 / \varepsilon ^ { 2 } )$ , the $\Omega ( 1 / \sqrt { n } )$ gap between the loss sufered by $f ( x _ { 1 } , \ldots , x _ { n - 1 } )$ and $\operatorname { M a j } _ { n }$ exceeds ε.

We now consider the $\ell _ { 2 }$ loss, which can be bounded for any $f : \{ \pm 1 \} ^ { n - 1 }  [ \pm 1 ]$ as

$$
\begin{array} { r } { \mathbb { E } \big [ ( { \pmb x } _ { n } - f ( { \pmb x } _ { 1 } , \dots , { \pmb x } _ { n - 1 } ) ) ^ { 2 } \big ] = 1 - 2 \mathbb { E } [ { \pmb x } _ { n } f ( { \pmb x } _ { 1 } , \dots , { \pmb x } _ { n - 1 } ) ] + \mathbb { E } [ f ( { \pmb x } _ { 1 } , \dots , { \pmb x } _ { n - 1 } ) ^ { 2 } ] \ge 1 . } \end{array}
$$

The optimal symmetric function is $\begin{array} { r } { s _ { n } ( { \pmb x } ) = \sum _ { i } { \pmb x } _ { i } / n } \end{array}$ , which is perfectly calibrated and whose sign is Majority. Computing its loss, we see that

$$
\operatorname* { m i n } _ { c \in { \mathcal { C } } } \mathbb { E } \big [ ( { \pmb y } - c ( { \pmb x } ) ) ^ { 2 } \big ] \leqslant \mathbb { E } \bigg [ \Big ( { \pmb x } _ { n } - \frac { { \pmb x } _ { 1 } + \dots + { \pmb x } _ { n } } { n } \Big ) ^ { 2 } \bigg ] = 1 - \frac { 1 } { n } .
$$

Unless $n > \Omega ( 1 / \varepsilon )$ , the gap between the $\ell _ { 2 }$ loss sufered by $f$ and $s _ { n }$ exceeds $\varepsilon .$

## 4.2 Lower Bounds for Multicalibration

What remains is to show that our ${ \cal O } ( 1 / \varepsilon ^ { 2 } )$ upper bound for smooth swap multicalibration and ${ \cal O } ( 1 / \varepsilon ^ { 3 } )$ upper bound for (full) swap multicalibration are also tight up to constant factors. In fact, our examples will rule out improvements even for smooth calibration and (full) calibration, where there is efectively no class C. As a bonus, we will also show that the randomization over iterates performed at the end of Algorithm 3 is necessary, in the sense that there may not exist any particular iterate, besides the initial 0 predictor, that is smoothly calibrated. We emphasize that the following lower bounds, unlike Theorem 4.1, are quite specific to our specific oracle algorithm, Algorithm 3.

To state our various results, it will be helpful to define the notion of a valid sequence of queries $q _ { 0 } , \ldots , q _ { T }$ to a conditional multiaccuracy oracle.<sup>10</sup> These are query sequences that could have arisen over the course of Algorithm 3 for some underlying data distribution and some adversarially chosen sequence of oracle responses. The hard problem instances that we construct do not assume the class C contains any functions besides the constant functions, nor do they assume that any error, such as statistical or rounding error, is present in the oracle responses. We will, however, place a restriction on the types of queries submitted to the oracle; we require these to be weighted averages of the previous responses with coeficients determined in advance by a constant step size $\lambda .$ This restriction was not needed for Theorem 4.1.

Definition 4.2 (Valid Query Sequence). A sequence of query predictors $q _ { 1 } , \dots , q _ { T } : \mathcal { X } \to [ \pm 1 ]$ is valid for $\lambda \in ( 0 , 1 ]$ if there exists a distribution D over pairs $( { \pmb x } , { \pmb y } ) \in \mathcal { X } \times \{ \pm 1 \}$ and a sequence of response predictors $r _ { 1 } , \dots , r _ { T } : \mathcal { X } \to [ \pm 1 ]$ such that the following conditions hold for all $t = 1 , \dots , T \colon$

• (Update Rule) if $\lambda _ { 1 } = 1$ and $\lambda _ { t } = \lambda$ for $t \geqslant 2 .$ , then

$$
q _ { t } = ( 1 - \lambda _ { t } ) q _ { t - 1 } + \lambda _ { t } r _ { t } ;
$$

• (Conditional Accuracy in Expectation)

$$
\begin{array} { r } { \mathbb { E } [ { \pmb { y } } - r _ { t } ( { \pmb { x } } ) \ | \ q _ { t - 1 } ( { \pmb { x } } ) ] = 0 . } \end{array}
$$

We use the convention $q _ { 0 } = 0$

Our first two results of this section, Lemmas 4.3 and 4.4, state that there exist long valid sequences with large calibration error. By $\mathrm { ^ { 6 6 } l a r g e , } ^ { \mathrm { 7 } }$ we mean greater than $\varepsilon ,$ and by “long,” we mean $T = \Omega ( 1 / \varepsilon ^ { 2 } )$ for degree-2 calibration or $T = \Omega ( 1 / \varepsilon ^ { 3 } )$ for (full) calibration. These lemmas show that our $\dot { T } = \cal { O } ( 1 / \varepsilon ^ { 2 } )$ upper bound for smooth swap multicalibration and $T = { \cal O } ( 1 / \varepsilon ^ { 3 } )$ upper bound for (full) swap multicalibration are tight. We defer all further proofs in this section to Appendix B.

Lemma 4.3 (Large CE<sub>2</sub>). For all $\lambda \in ( 0 , 1 ]$ and $T \in \mathbb { N }$ , there exist valid $q _ { 1 } , \ldots , q _ { T }$ for λ such that

$$
\operatorname* { m i n } _ { t \in [ T ] } | \mathrm { C E } _ { 2 } ( q _ { t } ) | = \Omega \Big ( \frac { 1 } { \sqrt { T } } \Big ) .
$$

In fact, we have the stronger lower bound $\Omega ( \operatorname* { m i n } \{ 1 , 1 / ( \lambda T ) + \lambda \} )$ , which also holds for the degree-2 calibration error of any randomized predictor defined as a random mixture of $q _ { 1 } , \ldots , q _ { T }$ composed with additional unbiased random noise N : $[ \pm 1 ]  [ \pm 1 ]$

Lemma 4.4 (Large ECE). For all $\lambda \in ( 0 , 1 ]$ and $T \in \mathbb { N }$ , there exist valid $q _ { 1 } , \ldots , q _ { T }$ for λ such that

$$
\operatorname* { m i n } _ { t \in [ T ] } \operatorname { E C E } ( q _ { t } ) = \Omega \Big ( \frac { 1 } { \sqrt [ 3 ] { T } } \Big ) .
$$

In fact, we have the stronger lower bound $\Omega ( \operatorname* { m i n } \{ 1 , 1 / ( \lambda T ) + \sqrt { \lambda } \} )$ , which also holds for the expected calibration error of any randomized predictor defined as a random mixture of $q _ { 1 } , \ldots , q _ { T }$ composed with additional unbiased random noise N : $[ \pm 1 ]  [ \pm 1 ]$

Our next result, Lemma 4.5, shows that the random mixture over time steps at the end of Algorithm 3 is necessary, in the sense that one cannot hope to simply output a single, deterministic iterate $q _ { t }$ from the sequence. Crucially, the hard instances that we construct apply when there are multiple weight functions of interest. This is not a coincidence. As we saw in Section 3 and will discuss further in Section 5.1, when there is only one weight function or loss function of interest (e.g. agnostic learning) Algorithm 3 can indeed be modified to output a single iterate $q _ { t } ,$ possibly with additional noise and post-processing.

Lemma 4.5 (Smooth Calibration Tradeof). For all $\lambda \in ( 0 , 1 ]$ , there exist valid $q _ { 1 } , q _ { 2 } , . . . .$ . such that

$$
\operatorname* { m i n } _ { t \in \mathbb { N } } \operatorname* { m a x } _ { i \in [ 2 ] } | \mathrm { C E } _ { w _ { i } } ( q _ { t } ) | = \Omega ( 1 )
$$

for two diferent 1-Lipschitz and [±1]-valued weight functions $w _ { 1 } , w _ { 2 } : [ \pm 1 ]  [ \pm 1 ]$

Although we defer the proofs of Lemmas 4.3 to 4.5 to Appendix B, we sketch the main ideas here.

The results all rely on a certain simple family of predictors, which we will use for the Bayesoptimal $p ^ { * }$ , the queries $q _ { t }$ , and the responses $r _ { t }$ . In this family, the feature space is ${ \mathcal { X } } = { \sqrt { n } } \cdot S ^ { n - 1 }$ the sphere of radius $\sqrt { n }$ centered at $0 \in \mathbb { R } ^ { n }$ , and each $a \in \mathbb { R } ^ { n }$ defines a predictor

$$
a ( x ) = \langle a , x \rangle .
$$

This predictor is [±1]-valued as long as $\| a \| \leqslant 1 / { \sqrt { n } }$ . For clarity of presentation, our examples will sometimes involve vectors $p ^ { \ast } , q _ { t } , r _ { t } \in \mathbb { R } ^ { n }$ with norm greater than $1 / \sqrt { n }$ , but still $O ( 1 )$ . We keep in mind that these vectors must eventually be scaled down to have norm at most $1 / { \sqrt { n } }$ , but this scaling will not afect any of our results since all our examples use $n = 2$ . Intuitively, the sequences $q _ { t }$ that we construct will resemble circular or oscillating paths in the plane.

There are a few key properties of these predictors that we use repeatedly. They pertain to expectations of products, conditional expectations, and expected absolute values. We state the properties for arbitrary dimensions $n \in \mathbb { N }$ , but to reiterate, we only really need the case $n = 2$

Lemma 4.6. Fix $n \geqslant 2$ and sample x ∼ Uniform $( { \sqrt { n } } \cdot S ^ { n - 1 } )$ . Then for all $a , b \in \mathbb { R } ^ { n }$ with $\| b \| = 1$

$$
\begin{array} { r l } & { \mathbb { E } \big [ \langle a , \pmb { x } \rangle \langle b , \pmb { x } \rangle \big ] = \langle a , b \rangle , } \\ & { \mathbb { E } \big [ \langle a , \pmb { x } \rangle \mid \langle b , \pmb { x } \rangle \big ] = \langle a , b \rangle \langle b , \pmb { x } \rangle . } \end{array}
$$

In particular, $\mathbb { E } [ \langle a , { \pmb x } \rangle \mid \langle b , { \pmb x } \rangle ] = 0$ if and only $i f \left. a , b \right. = 0$ . Also,

$$
\mathbb { E } \big [ | \langle b , \pmb { x } \rangle | \big ] = C _ { n } ,
$$

where $3 / 4 \leqslant C _ { n } \leqslant 1$ is a constant depending only on $n ,$ and not the direction of b.

The first key observation is that in the linear predictor family, degree-2 calibration error $\mathrm { C E _ { 2 } }$ measures the squared distance to the point $p ^ { * } / 2$ . Indeed, by Lemma 4.6, for any $p ^ { * } , q \in \mathbb { R } ^ { n }$ ,

$$
\operatorname { C E } _ { 2 } ( q ) = \operatorname { \mathbb { E } } [ q ( { \pmb x } ) ( { \pmb y } - q ( { \pmb x } ) ) ] = \langle q , p ^ { * } - q \rangle = \Big \| \frac { p ^ { * } } { 2 } \Big \| ^ { 2 } - \Big \| q - \frac { p ^ { * } } { 2 } \Big \| ^ { 2 } .\tag{10}
$$

In other words, if $n = 2$ and a sequence of points $q _ { 1 } , \ldots , q _ { T }$ lies entirely on a circle centered at the point $p ^ { * } / 2$ , then the sequence has constant degree-2 calibration error (in n ⩾ 3 dimensions, this set becomes a sphere). For a non-zero prediction vector, Lemma 4.6 implies a similar formula for expected calibration error:

$$
\operatorname { E C E } ( q ) = \mathbb { E } { \big [ } | \mathbb { E } [ y | q ( \pmb { x } ) ] - q ( \pmb { x } ) | { \big ] } = \mathbb { E } { \bigg [ } { \bigg | } { \frac { \langle p ^ { * } , q \rangle } { \| q \| ^ { 2 } } } \langle q , \pmb { x } \rangle - \langle q , \pmb { x } \rangle { \bigg | } { \bigg ] } = C _ { n } { \frac { | \langle q , p ^ { * } - q \rangle | } { \| q \| } } .\tag{11}
$$

Here, $C _ { n } \in [ 3 / 4 , 1 ]$ is the constant from Lemma 4.6. In particular, for the family of predictors under consideration, degree-2 and expected calibration error essentially difer by a factor of $\| q \| ;$

$$
\operatorname { E C E } ( q ) = C _ { n } { \frac { | \mathrm { C E } _ { 2 } ( q ) | } { \| q \| } } .
$$

For the zero predictor, both calibration errors are zero. We also have the general relationship $\mathrm { E C E } ( q ) \geqslant | \mathrm { C E } _ { 2 } ( q ) |$ |, which is not specific to the linear families under consideration. For the remaining details of the constructions, see Appendix B.

## 5 Derandomization

In several cases, our learning algorithms output randomized predictors $p : \mathcal { X }  [ \pm 1 ]$ , which may output diferent values in [±1] on the same input $x \in \mathcal { X }$ when invoked multiple times. In practice, this trait is arguably undesirable, raising the question of whether these predictors can be derandomized while retaining their various learning guarantees. In this section, we study several cases where this derandomization can indeed be performed.

We already saw that our algorithm does not require randomness to achieve loss minimization with respect to any single loss function ℓ with Lipschitz $k _ { \ell }$ . Next, in Section 5.1, we similarly show that our algorithm can achieve calibration with respect to any single Lipschitz weight function without introducing unnecessary randomness. To this end, we first use the weight function $w ( p ) = p$ $( \mathrm { i . e . }$ , degree-2 calibration $\mathrm { C E _ { 2 } } )$ as a case study, to convey the intuition behind the proof, and then state the derandomization result for the more general case of L-Lipschitz weight functions w. Next, in Section 5.2, we give a recipe for derandomizing a predictor in a black-box fashion while maintaining a wide range of other learning guarantees.

## 5.1 Degree-2 Calibration

Suppose throughout that the query-response pairs $\left( q _ { t } , r _ { t } \right)$ satisfy $\begin{array} { r } { \mathbb { E } [ { \pmb { y } } - r _ { t + 1 } ( { \pmb { x } } ) \ | \ q _ { t } ( { \pmb { x } } ) ] = 0 \ ( \mathrm { i . e . } } \end{array}$ they satisfy the conditional accuracy in expectation condition from Definition 4.2). Recall that the geometric update is defined as

$$
q _ { t + 1 } = ( 1 - \lambda ) q _ { t } + \lambda r _ { t + 1 } = q _ { t } + \lambda ( r _ { t + 1 } - q _ { t } ) ,
$$

where $q _ { t } , r _ { t } : \mathcal { X }  [ \pm 1 ]$ . In this subsection only, we ignore oracle approximation and predictionrounding errors for simplicity, and work with exact conditional accuracy.

At each step t, we denote the degree-2 calibration error by $\alpha _ { t } : = \mathrm { C E } _ { 2 } ( q _ { t } ) = \mathbb { E } [ q _ { t } ( \pmb { x } ) ( \pmb { y } - q _ { t } ( \pmb { x } ) ) ]$ Our goal is to show that every suficiently long consecutive sequence of query predictors $\left\{ q _ { t } \right\}$ contains at least one (deterministic) predictor whose absolute degree-2 calibration error satisfies $| \alpha _ { t } | \le \alpha$ , and hence is degree-2 calibrated with error parameter α.

The proof proceeds in two steps:

1. Consecutive violations are close (Claim 5.1). Per the geometric update, each successive update changes the query predictor $q _ { t }$ by at most 2λ, i.e., $| q _ { t + 1 } ( x ) - q _ { t } ( x ) | \leq 2 \lambda$ . Since the degree-2 calibration violation is Lipschitz in the prediction (specifically, the map $v \mapsto v ( y - v )$ is 3-Lipschitz), we can translate the closeness between successive $q _ { t }$ updates to closeness between successive $\alpha _ { t }$ errors. This means that consecutive violations cannot switch from substantially positive to substantially negative (or vice versa) without passing through the approximately calibrated region $[ - \alpha , \alpha ]$

2. Telescoping bounds the average signed violation (Theorem 5.2). Summing the identity in Lemma 5.3 over N consecutive rounds and telescoping gives

$$
\left| \frac { 1 } { N } \sum _ { t } \alpha _ { t } \right| \le O \Bigl ( \frac { 1 } { \lambda N } + \lambda \Bigr ) .
$$

Under the step size $\lambda$ and sequence length assumption in Theorem 5.2, the RHS is upperbounded by $\alpha .$ . This alone does not imply that any one $\left| \alpha _ { t } \right|$ violation is small. However, Claim 5.1 is what excludes this possibility. Indeed, suppose that $| \alpha _ { t } | > \alpha$ for every round in these N consecutive rounds. By Claim 5.1, consecutive violations cannot jump across the approximately calibrated region. Since none lies inside that region, then they must all either satisfy $\alpha _ { t } > \alpha$ or all satisfy $\alpha _ { t } < - \alpha$ . Their average would likewise also be greater than α or less than $- \alpha$ , respectively. In either case,

$$
\left| \frac { 1 } { N } \sum _ { t } \alpha _ { t } \right| > \alpha ,
$$

which contradicts the initial telescoping inequality.

Note that it is not necessarily true that the absolute violation $\left| \alpha _ { t } \right|$ decreases at every round. What we show is that the query predictors cannot all have large absolute violation for too many consecutive rounds.

The first step of the proof can be formally stated as follows:

Claim 5.1. For all $t \geq 1 , \left| \alpha _ { t + 1 } - \alpha _ { t } \right| \leq 6 \lambda$

The proof of this claim follows from Lemma 7.4 in [BGHN23], adapted to the [±1] range of our predictors. The proof requires expanding the $q _ { t + 1 } ( x ) ( y - q _ { t + 1 } ( x ) ) - q _ { t } ( x ) ( y - q _ { t } ( x ) )$ expression point-wise, factoring out the $q _ { t + 1 } - q _ { t }$ term, and using the fact that all of $y , q _ { t } , q _ { t + 1 } , r _ { t + 1 }$ output values in [±1].

Next, we use a potential-based argument to show that in any long-enough sequence of iterates $\left\{ q _ { t } \right\}$ , at least one of the predictors is α-degree-2 calibrated. Formally:

Theorem 5.2. Fix $\alpha > 0$ and assume $\lambda \le \alpha / 4$ . Suppose that $\left( q _ { t } , r _ { t } \right)$ are a valid sequence of query-response pairs that satisfy the geometric update rule. Let $s \geq 1$ be any starting round. If $N > 1 / ( \lambda \alpha )$ , then there exists $t \in \left\{ s , \ldots , s + N - 1 \right\}$ such that

$$
| \mathrm { C E } _ { 2 } ( q _ { t } ) | = \left| \mathbb { E } [ q _ { t } ( \pmb { x } ) ( \pmb { y } - q _ { t } ( \pmb { x } ) ) ] \right| \leq \alpha .
$$

Equivalently, no consecutive block of more than $1 / ( \lambda \alpha )$ iterates can remain entirely outside the α-degree-2 calibration region.

To prove this theorem, we use an appropriate potential function, defined using the $q _ { t }$ values. Before proving Theorem 5.2, we define and prove a key identity of this potential function:

Lemma 5.3. Let $\Phi _ { t } = \mathbb { E } [ q _ { t } ( \pmb { x } ) ^ { 2 } ]$ . Then,

$$
\Phi _ { t + 1 } - \Phi _ { t } = 2 \lambda \alpha _ { t } + \lambda ^ { 2 } \mathbb { E } [ ( r _ { t + 1 } ( \pmb { x } ) - q _ { t } ( \pmb { x } ) ) ^ { 2 } ] ,
$$

where $\lambda$ is defined in the geometric update and $\alpha _ { t } = \mathbb { E } [ q _ { t } ( \pmb { x } ) ( \pmb { y } - q _ { t } ( \pmb { x } ) ) ]$

Proof. By the query-response pairs property, it follows that

$$
\begin{array} { r l } & { \mathbb { E } [ q _ { t } ( \pmb { x } ) ( \pmb { y } - r _ { t + 1 } ( \pmb { x } ) ) ] = \mathbb { E } [ q _ { t } ( \pmb { x } ) \cdot \mathbb { E } \big [ \pmb { y } - r _ { t + 1 } ( \pmb { x } ) | q _ { t } ( \pmb { x } ) \big ] \big ] } \\ & { \qquad = 0 , } \end{array}\tag{12}
$$

since on each level set of $q _ { t }$ (on which $q _ { t }$ is, by definition, constant), the residual $\boldsymbol { y } - \boldsymbol { r } _ { t + 1 }$ has mean equal to 0. Therefore, $\alpha _ { t } = \mathbb { E } [ q _ { t } ( \pmb { x } ) ( r _ { t + 1 } ( \pmb { x } ) - q _ { t } ( \pmb { x } ) ) ]$ . By adding and subtracting $r _ { t + 1 }$ , we have:

$$
\begin{array} { r l r } { \alpha _ { t } = \mathbb { E } [ q _ { t } ( \pmb { x } ) \cdot ( \pmb { y } - r _ { t + 1 } ( \pmb { x } ) + r _ { t + 1 } ( \pmb { x } ) - q _ { t } ( \pmb { x } ) ) ] } \\ { } & { } & { = \mathbb { E } [ q _ { t } ( \pmb { x } ) \cdot ( \pmb { y } - r _ { t + 1 } ( \pmb { x } ) ) ] + \mathbb { E } [ q _ { t } ( \pmb { x } ) \cdot ( r _ { t + 1 } ( \pmb { x } ) - q _ { t } ( \pmb { x } ) ) ] } \\ { } & { } & { = 0 + \mathbb { E } [ q _ { t } ( \pmb { x } ) \cdot ( r _ { t + 1 } ( \pmb { x } ) - q _ { t } ( \pmb { x } ) ) ] ~ \mathrm { b y ~ e q u a t i o n ~ ( 1 2 ) } } \end{array}
$$

Pointwise, by expanding and applying the geometric update definition, we have that

$$
\begin{array} { c } { { q _ { t + 1 } ( x ) ^ { 2 } - q _ { t } ( x ) ^ { 2 } = \left( q _ { t } ( x ) + \lambda ( r _ { t + 1 } ( x ) - q _ { t } ( x ) ) \right) ^ { 2 } - q _ { t } ( x ) ^ { 2 } } } \\ { { = 2 \lambda q _ { t } ( x ) \cdot ( r _ { t + 1 } ( x ) - q _ { t } ( x ) ) + \lambda ^ { 2 } \cdot ( r _ { t + 1 } ( x ) - q _ { t } ( x ) ) ^ { 2 } . } } \end{array}
$$

By taking expectations and using the fact that $\alpha _ { t } = \mathbb { E } [ q _ { t } ( \pmb { x } ) ( r _ { t + 1 } ( \pmb { x } ) - q _ { t } ( \pmb { x } ) ) ]$ , it follows that

$$
\Phi _ { t + 1 } - \Phi _ { t } = 2 \lambda \alpha _ { t } + \lambda ^ { 2 } \mathbb { E } [ ( r _ { t + 1 } ( \pmb { x } ) - q _ { t } ( \pmb { x } ) ) ^ { 2 } ] ,
$$

as we wanted to show.

Summing the identity in Lemma 5.3 over N consecutive rounds and re-arranging gives

$$
\frac { 1 } { N } \sum _ { t = s } ^ { s + N - 1 } \alpha _ { t } = \frac { \Phi _ { s + N } - \Phi _ { s } } { 2 \lambda N } - \frac { \lambda } { 2 N } \sum _ { t = s } ^ { s + N - 1 } \mathbb { E } [ ( r _ { t + 1 } ( x ) - q _ { t } ( x ) ) ^ { 2 } ] .
$$

Since the potential takes values between zero and one, and each squared diference is at most four, taking absolute values yields

$$
\left| \frac { 1 } { N } \sum _ { t } \alpha _ { t } \right| \leq \frac { 1 } { 2 \lambda N } + 2 \lambda ,
$$

which is the expression we used in the proof intuition at the beginning of Section 5.1.

With this lemma in hand, we now prove Theorem 5.2:

Proof of Theorem 5.2. Suppose toward contradiction that

$$
| \alpha _ { t } | > \alpha \qquad \mathrm { f o r ~ e v e r y ~ } t \in \{ s , \ldots , s + N - 1 \} .\tag{13}
$$

By Claim 5.1,

$$
\left| { { \alpha } _ { t + 1 } } - { { \alpha } _ { t } } \right| \le 6 \lambda \le 2 \alpha .
$$

Therefore, the values in equation (13) cannot change sign. Indeed, moving from a value greater than α to a value smaller than −α would require a change strictly larger than $2 \alpha$ . Consequently, either

$$
\alpha _ { t } > \alpha \qquad \mathrm { f o r ~ e v e r y ~ } t \in \{ s , \ldots , s + N - 1 \} ,\tag{14}
$$

or

$$
\alpha _ { t } < - \alpha \qquad \mathrm { f o r ~ e v e r y ~ } t \in \{ s , \ldots , s + N - 1 \} .\tag{15}
$$

Suppose first that equation (14) holds. By Lemma 5.3, $\Phi _ { t + 1 } - \Phi _ { t } > 2 \lambda \alpha$ . Summing over $t = s , \ldots , s + N - 1$ gives $\Phi _ { s + N } - \Phi _ { s } > 2 N \lambda \alpha$ . Given that by assumption $N > 1 / ( \lambda \alpha )$ , it follows that the right-hand side is greater than 2. This is impossible because $\Phi _ { s } , \Phi _ { s + N } \in [ 0 , 1 ]$ , and hence $\Phi _ { s + N } - \Phi _ { s } \leq 1$

Suppose instead that equation (15) holds. Since $( r _ { t + 1 } ( x ) - q _ { t } ( x ) ) ^ { 2 } \leq 4$ , Lemma 5.3 implies that

$$
\begin{array} { c } { { \Phi _ { t + 1 } - \Phi _ { t } < - 2 \lambda \alpha + 4 \lambda ^ { 2 } } } \\ { { \leq - \lambda \alpha , } } \end{array}
$$

since $\lambda \le \alpha / 4$ . Summing over $t = s , \ldots , s + N - 1$ gives $\Phi _ { s + N } - \Phi _ { s } < - N \lambda \alpha$ . We again reach a contradiction, since $N > 1 / ( \lambda \alpha )$ implies that the left-hand side is smaller than −1, which is impossible since $\Phi _ { s } , \Phi _ { s + N } \in [ 0 , 1 ]$ , and hence $\Phi _ { s + N } - \Phi _ { s } \geq - 1$

Hence equation (13) cannot hold, proving that at least one of the $q _ { t }$ predictors for some $t \in$ $\left\{ s , \ldots , s + N - 1 \right\}$ is α-degree-2 calibrated. □

## 5.1.1 Generalization to Fixed w-Calibration

Beyond 2-calibration error, for a weight function $w : [ \pm 1 ]  [ \pm 1 ]$ , define the w-calibration violation of the query predictor at round t by

$$
\alpha _ { t } : = \mathrm { C E } _ { w } ( q _ { t } ) = \mathbb { E } [ w ( q _ { t } ( \pmb { x } ) ) ( \pmb { y } - q _ { t } ( \pmb { x } ) ) ] .
$$

In particular, taking $w ( p ) = p$ recovers the degree-2 calibration definition.

We generalize our derandomization result of the degree-2 calibration case (Theorem 5.2) to a general L-Lipschitz weight function w:

Theorem 5.4. Let $s \geq 1$ be any starting round and let $w : [ \pm 1 ]  [ \pm 1 ]$ be L-Lipschitz. Suppose that $( q _ { t } , r _ { t } )$ are a valid sequence of query-response pairs that satisfy the geometric update rule. Fix $\alpha > 0$ and assume $\lambda \le \alpha / ( 4 L + 2 )$ and $( T - 1 ) ( \lambda \alpha / 2 ) > 2$ . Then there exists a fixed $t \in \{ s , \ldots , s + T - 1 \}$ such that

$$
| \mathrm { C E } _ { w } ( q _ { t } ) | = | \mathbb { E } [ w ( q _ { t } ( \pmb { x } ) ) ( \pmb { y } - q _ { t } ( \pmb { x } ) ) ] | \leq \alpha .
$$

The proof is very similar to the case of degree-2 calibration: we use the same geometric update and exact conditional-accuracy assumption as in the degree-2 case. For this reason, we defer the formal proofs to the appendix.

The proof of Theorem 5.4 then follows the same two steps:

1. Consecutive violations are close (Lemma C.1). The geometric update is unchanged, so consecutive query predictions still difer by at most 2λ. What changes is the Lipschitz constant: the map $v \mapsto w ( v ) ( y - v )$ is now Lipschitz with constant $2 L + 1$ . Consequently, $| \alpha _ { t + 1 } - \alpha _ { t } | \leq 2 ( 2 L + 1 ) \lambda$ . Hence if λ is small enough relative to $\alpha ,$ consecutive violations cannot jump across the approximately calibrated region $[ - \alpha , \alpha ]$

2. Telescoping bounds the average signed violation (Theorem 5.4). The same argument as with the degree-2 calibration case holds, this time with the telescoping expression

$$
\left| \frac { 1 } { T - 1 } \sum _ { t } \alpha _ { t } \right| \leq O \Bigl ( \frac { 1 } { \lambda ( T - 1 ) } + L \lambda \Bigr ) .
$$

Under the step size λ and sequence length assumption in Theorem 5.4, the RHS is upperbounded by $\alpha$

## 5.2 General Derandomization

We will now provide a general and simple recipe for derandomizing a randomized predictor. The main idea is to partition the space into sets of suficiently small mass and fix, for each piece, an independent predictor drawn from the original distribution over predictors. Similar ideas have appeared in prior work in diferent contexts (see, e.g., [LV25, NR26]), but we provide all the relevant statements and proofs here for completeness. The proof is based on a simple Hoefding bound.

The proposed derandomization procedure requires the existence of a partition of the feature space into low-mass regions. This assumption is captured by the following definition.

Definition 5.5 (Partition Function). Let $\mathcal { D } _ { \mathcal { X } }$ be a distribution over X and let $f : \mathcal { X }  [ V ]$ be measurable and total, where $V \in  { \mathbb { N } }$ . We say that $f$ is a γ-partition function with respect to $\mathcal { D } _ { \mathcal { X } }$ if the following inequality holds for all $v \in [ V ]$

$$
\operatorname* { P r } _ { { \boldsymbol x } \sim { \mathcal { D } } _ { \boldsymbol \chi } } [ f ( { \boldsymbol x } ) = v ] \leqslant \gamma .
$$

Given a suficiently refined partition function, we use the following simple, black-box derandomization procedure.

Definition 5.6 (Piecewise Derandomization). Let f be a γ-partition function with respect to a distribution $\mathcal { D } _ { \mathcal { X } }$ over $x ,$ and let p be a randomized predictor, i.e., a random variable over the space of functions mapping $\mathcal { X }$ to $[ \pm 1 ]$ . The f-piecewise derandomization of $\mathbfit { p }$ is the following randomized procedure that outputs a deterministic predictor $p ^ { f }$

• For each $v \in [ V ]$ , let $\mathbf { \nabla } p _ { v }$ be an independent copy of $\mathbf { \nabla } _ { \pmb { p } } .$

• Let $\begin{array} { r } { p ^ { f } ( x ) = \sum _ { v \in [ V ] } p _ { v } ( x ) \mathbf { 1 } _ { f ( x ) = v } , } \end{array}$ , where $( p _ { v } ) _ { v }$ is a realization of $( p _ { v } ) _ { v }$

Our main derandomization result states that the piecewise derandomization procedure defined above yields with high probability a predictor that fools any fixed and bounded function $\varphi _ { : }$ in expectation over $\mathbf { \boldsymbol { x } } \sim \mathcal { D } _ { \mathcal { X } }$ . The corresponding accuracy scales polynomially with the refinement of the partition function.

Theorem 5.7. Let $\gamma , \delta \in ( 0 , 1 )$ , let $\varphi : \mathcal { X } \times [ \pm 1 ] \to [ \pm 1 ]$ and let f be a γ-partition function with respect to the distribution $\mathcal { D } _ { \mathcal { X } }$ over X . Let p be any randomized predictor. Then, the f-piecewise derandomization of p satisfies the following with probability at least $1 - \delta$

$$
\Big | \underset { x , p } { \mathbb { E } } [ \varphi ( \pmb { x } , \pmb { p } ( \pmb { x } ) ) ] - \frac { \mathbb { E } } { x } [ \varphi ( \pmb { x } , p ^ { f } ( \pmb { x } ) ) ] \Big | \leqslant O \big ( \sqrt { \gamma \log ( e / \delta ) } \big )
$$

Proof. We view $p ^ { f }$ as a random variable depending on the realization of $( p _ { v } ) _ { v }$ during the derandomization procedure. Accordingly, we also define the random variable $\pmb { \Delta }$ as follows:

$$
\begin{array} { r } { \pmb { \Delta } = \Big | \underset { \pmb { x } , \pmb { p } } { \mathbb { E } } [ \varphi ( \pmb { x } , \pmb { p } ( \pmb { x } ) ) ] - \frac { \mathbb { E } } { \pmb { x } } [ \varphi ( \pmb { x } , \pmb { p } ^ { f } ( \pmb { x } ) ) ] \Big | } \end{array}
$$

We will show that with probability at least $1 - \delta$ over the random choice of $( p _ { v } ) _ { v }$ , we have $\Delta \leqslant$ $O ( \sqrt { \gamma \log ( e / \delta ) } )$ , as desired. To this end, we decompose the terms of the diference as follows:

$$
\begin{array} { r l } & { \frac { \mathbb { E } } { x } [ \varphi ( x , p ^ { f } ( x ) ) ] = \displaystyle \sum _ { v \in [ V ] } \mathbb { E } [ \varphi ( x , p _ { v } ( x ) ) \mathbf { 1 } _ { f ( x ) = v } ] = : \displaystyle \sum _ { v \in [ V ] } Z _ { v } } \\ & { \frac { \mathbb { E } } { x , p } [ \varphi ( x , p ( x ) ) ] = \displaystyle \sum _ { v \in [ V ] } \mathbb { E } _ { x , p } [ \varphi ( x , p ( x ) ) \mathbf { 1 } _ { f ( x ) = v } ] = : \displaystyle \sum _ { v \in [ V ] } \mu _ { v } } \end{array}
$$

Observe that each of the terms in the first equation is a random variable whose expectation with respect to $( { \pmb p } _ { v } ) _ { v }$ equals the corresponding term of the second equation. In particular:

$$
\underset { ( p _ { v } ) _ { v } } { \mathbb { E } } \left[ \pmb { Z } _ { v } \right] = \underset { ( p _ { v } ) _ { v } } { \mathbb { E } } \frac { \mathbb { E } } { \mathbb { E } } [ \varphi ( \pmb { x } , p _ { v } ( \pmb { x } ) ) \mathbf { 1 } _ { f ( \pmb { x } ) = v } ] = \underset { \pmb { x } , p } { \mathbb { E } } [ \varphi ( \pmb { x } , p ( \pmb { x } ) ) \mathbf { 1 } _ { f ( \pmb { x } ) = v } ] = \mu _ { v }
$$

Moreover, we have that $| Z _ { v } | \leqslant \operatorname* { P r } _ { \pmb { x } } [ f ( \pmb { x } ) = v ] \leqslant \gamma$ with probability 1 over the choice of $( { \pmb p } _ { v } ) _ { \imath }$ , due to the definition of function $f ,$ and the bound on the values of $\varphi .$ . As a consequence, we have that $| Z _ { v } | \leqslant z _ { v }$ almost surely for values $z _ { v }$ satisfying

$$
\sum _ { v \in [ V ] } z _ { v } ^ { 2 } \leqslant \sum _ { v \in [ V ] } \left( \operatorname* { P r } _ { \mathbf { \Phi } } [ f ( \mathbf { \Phi } x ) = v ] \right) ^ { 2 } \leqslant \gamma \sum _ { v \in [ V ] } \operatorname* { P r } _ { \mathbf { \Phi } } [ f ( \mathbf { \Phi } x ) = v ] = \gamma
$$

The result follows due to a standard Hoefding bound, by additionally observing that the random variables $( Z _ { v } ) _ { \ v }$ are independent, by the definition of $( { \pmb p } _ { v } ) _ { \imath }$ . □

Constructing the Partition Function We have shown that it is possible to derandomize any predictor with respect to any fixed target measure $\mathcal { D } _ { \mathcal { X } }$ and function $\varphi ,$ provided that there exists a γ-partition function with respect to $\mathcal { D } _ { \mathcal { X } }$ . In $\mathrm { A p } ]$ pendix D, we show that a partition function can be eficiently constructed with sample access to $\mathcal { D } _ { \mathcal { X } }$ essentially whenever $\mathcal { D } _ { \mathcal { X } }$ does not have point masses larger than $\gamma .$ . We call such distributions γ-divisible.

Example 5.8. Any distribution over $\mathbb { R } ^ { d }$ that admits a density is γ-divisible for any $\gamma \in ( 0 , 1 )$ and a γ-partition function can be constructed with probability at least $1 - \delta$ in time $\tilde { O } ( d / \gamma ) \log ( 1 / \delta )$ and using $\tilde { O } ( 1 / \gamma ) \log ( 1 / \delta )$ samples from $\mathcal { D } _ { \mathcal { X } }$ . The same is true for any discrete distribution over $\mathbb { R } ^ { d }$ that does not assign mass more than $\gamma$ to any singleton.

Applications We will now show that our main results on smooth swap multicalibration and smooth omniprediction (see Theorem 1.4) can be derandomized in a black-box way, as long as $\mathcal { D } _ { \mathcal { X } }$ is γ-divisible for some appropriate $\gamma = \mathrm { p o l y } ( \varepsilon ) / \log ( 1 / \delta )$

Remark 5.9. Note that the divisibility assumption is almost without loss of generality: If $\mathcal { D } _ { \mathcal { X } }$ had heavy atoms $x _ { h } \in \mathcal { X }$ , then we would be able to compute accurate estimates of the Bayes-optimal values $\mathbb { E } [ { \pmb y } | { \pmb x } = { \pmb x } _ { h } ]$ using $\mathrm { p o l y } ( 1 / \gamma )$ samples, and apply our main algorithm on the truncation of $\mathcal { D } _ { \mathcal { X } }$ that does not have heavy atoms. Note that this requires labeled samples (not just unlabeled) and changes the marginal distribution $\mathcal { D } _ { \mathcal { X } }$ over features, but any reweighting such as this one is already addressed by the very first step of our algorithm.

The idea is to view the aforementioned guarantees from Theorem 1.4 as bounds on quantities of the form:

$$
\operatorname* { s u p } _ { \varphi \in \Phi } \big | \mathbb { E } _ { \pmb { x } , \pmb { y } } [ \varphi ( \pmb { x } , \pmb { y } , p ( \pmb { x } ) ) ] ,
$$

where the function family $\Phi$ can be represented by a sparse subset $\Phi ^ { \prime }$ of small cardinality. Therefore, it sufices to fool only the members of the family $\Phi ^ { \prime }$ , and the result follows by combining Theorem 5.7 with a union bound.

We now define the notion of sparsification that sufices for our purposes.

Definition 5.10. Let Φ be a collection of functions of the form $\varphi : \mathcal { X } \times \{ \pm 1 \} \times [ \pm 1 ] \to \mathbb { R }$ . We say that Φ is $( \varepsilon , N )$ -sparsifiable with respect to a distribution $\mathcal { D }$ over $\mathcal { X } \times \left\{ \pm 1 \right\}$ if there is a set $\Phi ^ { \prime } \subseteq \Phi$ with cardinality $\vert \Phi ^ { \prime } \vert \leqslant N$ such that for all $\varphi \in \Phi$ , there exists $\varphi ^ { \prime } \in \Phi ^ { \prime }$ such that

$$
\operatorname* { s u p } _ { p : \mathcal { X } \to [ \pm 1 ] } \Big | \operatorname* { \mathbb { E } } _ { { \boldsymbol x } , { \boldsymbol y } } [ \varphi ^ { \prime } ( { \boldsymbol x } , { \boldsymbol y } , p ( { \boldsymbol x } ) ) ] - \operatorname* { \mathbb { E } } _ { { \boldsymbol x } , { \boldsymbol y } } [ \varphi ( { \boldsymbol x } , { \boldsymbol y } , p ( { \boldsymbol x } ) ) ] \Big | \leqslant \varepsilon .
$$

Application I: Smooth Swap Multicalibration We first focus on smooth swap multicalibration, where the class C is assumed to have bounded pseudodimension (see Definition 2.10).

Lemma 5.11 (Sparsifying Smooth Swap Multicalibration Witnesses). Let ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to [ \pm 1 ] \}$ have pseudo-dimension d, and let $\Phi _ { \mathcal { C } , L } ^ { \mathrm { s w a p } }$ be the class of functions

$$
\varphi _ { c } ( x , y , v ) : = c ( v , x ) ( y - v ) ,
$$

where $c ( v , \cdot ) \in \mathcal { C }$ for every $v \in [ \pm 1 ]$ and $c ( \cdot , x )$ is L-Lipschitz for every x $\in \mathcal { X }$ for some $L \geqslant 1$ . Then, for every distribution D over $\mathcal { X } \times \{ \pm 1 \}$ and every $\varepsilon \in ( 0 , 1 )$ , the class $\Phi _ { \mathcal { C } , L } ^ { \mathrm { s w a p } }$ is $( \varepsilon , N )$ -sparsifiable with respect to D, where

$$
\log N = O \Bigl ( \frac { d L } { \varepsilon } \log \Bigl ( \frac { L } { \varepsilon } \Bigr ) \Bigr ) .
$$

Proof. Let $\mathcal { D } _ { \mathcal { X } }$ denote the X-marginal of $\mathcal { D } .$ . Choose a grid $v _ { 1 } , \ldots , v _ { K }$ in [±1] such that every $v \in [ \pm 1 ]$ is within $\varepsilon / ( 8 L )$ of some grid point, where $K = \Theta ( L / \varepsilon )$ . By the standard pseudo-dimension covering bound (see, $\mathrm { e . g . , \ [ H a u 9 2 ] ) }$ , C has an $L _ { 1 } ( \mathcal { D } _ { \mathcal { X } } )$ -cover of radius $r : = \varepsilon / 8 K = \Theta ( \varepsilon ^ { 2 } / L )$ and size at most $( 1 / r ) ^ { O ( d ) } = ( \bar { L ^ { } } / \varepsilon ) ^ { O ( d ) }$

Associate with every c the tuple of cover elements approximating $c ( v _ { 1 } , \cdot ) , \ldots , c ( v _ { K } , \cdot )$ , and retain one representative $c ^ { \prime }$ for every realized tuple. The number of retained representatives is bounded by the size of the net corresponding to each coordinate, raised to the number of coordinates K. Therefore, it is at most

$$
\left( ( 1 / r ) ^ { O ( d ) } \right) ^ { K } = \left( \frac { L } { \varepsilon } \right) ^ { O \left( d L / \varepsilon \right) } .
$$

If c and $c ^ { \prime }$ have the same tuple, then

$$
\underset { \ b { x } } { \mathbb { E } } \left[ \big | c ( p ( \pmb { x } ) , \pmb { x } ) - c ^ { \prime } ( p ( \pmb { x } ) , \pmb { x } ) \big | \right] \leqslant 2 L \frac { \varepsilon } { 8 L } + 2 K r \leqslant \frac { \varepsilon } { 2 }
$$

for every predictor $p : \mathcal { X } \to [ \pm 1 ]$ . Since $| { \pmb y } - p ( { \pmb x } ) | \leqslant 2$ , it follows that

$$
\big | \mathbb { E } \left[ \varphi _ { c } ( { \pmb x } , { \pmb y } , p ( { \pmb x } ) ) - \varphi _ { c ^ { \prime } } ( { \pmb x } , { \pmb y } , p ( { \pmb x } ) ) \right] \big | \leqslant \varepsilon .
$$

Taking the supremum over p proves the result.

We are ready to provide a derandomization result for smooth swap multicalibration.

Corollary 5.12 (Derandomizing Smooth Swap Multicalibration). Let ${ \mathcal { C } } \subseteq \{ \mathcal { X } \to [ \pm 1 ] \}$ have pseudo-dimension $d ,$ and suppose that p is $( \mathcal { C } , \varepsilon )$ -smooth swap multicalibrated with respect to $\mathcal { D }$ There is a universal constant $c > 0$ such that, for any $\gamma \leqslant$ c min $\{ 1 , \varepsilon ^ { 3 } / ( d \log ( e / \varepsilon ) + \varepsilon \log ( e / \delta ) ) \}$ , if $\mathcal { D } _ { \mathcal { X } }$ is γ-divisible, then the piecewise derandomization $p ^ { f }$ obtained from a γ-partition function f is $( \mathcal { C } , O ( \varepsilon ) )$ -smooth swap multicalibrated with probability at least $1 - \delta$ . Moreover, f can be constructed using $O ( ( 1 / \gamma ) \log ( 1 / \gamma \delta ) )$ unlabeled samples from $\mathcal { D } _ { \mathcal { X } }$

Proof. Apply Lemma 5.11 with accuracy $\varepsilon / 8$ to obtain a sparsifying set $\Phi ^ { \prime }$ of size:

$$
| \Phi ^ { \prime } | \leqslant 2 ^ { O ( ( d / \varepsilon ) \log ( e / \varepsilon ) ) }
$$

Conditional on f being a γ-partition function, the partition-function concentration bound (Theorem 5.7) and a union bound over $\Phi ^ { \prime }$ show that the expectations of all its functions change by at most $O ( \varepsilon )$ under derandomization. Sparsifiability extends the guarantee to every smooth swap witness, at an additional error of at most $\varepsilon / 4$ . The guarantee for $_ p$ therefore transfers to $p ^ { f }$ . Finally, the construction of the partition function succeeds with probability at least $1 - \delta / 2 ;$ a union bound completes the proof. □

Application II: Smooth Omniprediction Next, we focus on smooth omniprediction, obtaining the following derandomization result.

Corollary 5.13 (Derandomizing Smooth Omniprediction). Let p be an $( \mathcal { L } _ { B , \beta } , \mathcal { C } , \varepsilon )$ -omnipredictor with respect to ${ \mathcal { D } } ,$ where $\beta > 0$ . There is a universal constant $c > 0$ such that, for any $\gamma \leqslant$ c min $\{ 1 , \varepsilon ^ { 3 } / ( \beta ^ { 3 } + \beta ^ { 2 } \varepsilon \log ( e / \delta ) ) \}$ , if D<sub>X</sub> is γ-divisible, then the piecewise derandomization $p ^ { f }$ obtained from a γ-partition function f is an $( \mathcal { L } _ { B , \beta } , \mathcal { C } , O ( \varepsilon ) )$ -omnipredictor with probability at least $1 - \delta$ Moreover, $f$ can be constructed using $O ( ( 1 / \gamma ) \log ( 1 / \gamma \delta ) )$ unlabeled samples from $\mathcal { D } _ { \mathcal { X } }$

As usual, our approach will be to express the guarantee we want to preserve after derandomization as a bound on a quantity of the following form:

$$
\operatorname* { s u p } _ { \varphi \in \Phi } \big | \mathbb { E } _ { \pmb { x } , \pmb { y } } [ \varphi ( \pmb { x } , \pmb { y } , p ( \pmb { x } ) ) ] ,
$$

where the set $\Phi$ is sparsifiable. To this end, a first idea would be to choose $\varphi _ { \ell } ( x , y , v ) = \ell ( y , k _ { \ell } ( v ) )$ However, the class of loss functions $\mathcal { L } _ { B , \beta }$ we consider is not guaranteed to only contain Lipschitz losses $\ell$ and it is not clear whether $\mathcal { L } _ { B , \beta }$ is sparsifiable.

Instead, the guarantee we have is that the optimal response function $k _ { \ell }$ is β-Lipschitz. In the following lemma, we show that this sufices for our purposes, by showing that there is a choice of $\varphi _ { \ell }$ that preserves the omniprediction guarantee, and leads to a sparsifiable family $\Phi$

Lemma 5.14 (Sparsifying loss functions for smooth omniprediction). Let $\mathcal { L } _ { B , \beta }$ be the class of naturally parametrized loss functions with $a \ [ \pm B ]$ -bounded and $\beta - L i p s c h i t z$ best response function $k _ { \ell }$ . For $\ell \in \mathcal { L } _ { B , \beta }$ , define

$$
\varphi _ { \ell } ( x , y , v ) : = \ell ( y , k _ { \ell } ( v ) ) - \ell ( y , k _ { \ell } ( 0 ) ) ,
$$

and let

$$
\Phi _ { B , \beta } ^ { \mathrm { o m n i } } : = \{ \varphi _ { \ell } : \ell \in \mathcal { L } _ { B , \beta } \} .
$$

Then, for every distribution $\mathcal { D }$ over $\mathcal { X } \times \left\{ \pm 1 \right\}$ and every $\varepsilon > 0$ , the class $\Phi _ { B , \beta } ^ { \mathrm { o m n i } } \ i s \ ( \varepsilon , N ) \ – s p a r s i f i a b l e$ with respect to $\mathcal { D }$ , where

$$
\log N = O \left( 1 + { \frac { \beta } { \varepsilon } } \right) .
$$

In fact, the same sparsifying set works simultaneously for every distribution $\mathcal { D } .$

Proof. For $\ell \in \mathcal { L } _ { B , \beta }$ , let

$$
r _ { \ell } ( v ) : = k _ { \ell } ( v ) - k _ { \ell } ( 0 ) .
$$

Then $r _ { \ell } ( 0 ) = 0$ and $r _ { \ell }$ is $\beta { \cdot } \mathrm { I }$ ipschitz.

We first express $\varphi _ { \ell }$ in terms of $r _ { \ell }$ . Fix $s < t .$ . Since $k _ { \ell } ( t )$ minimizes $\ell ( t , \cdot )$ and $k _ { \ell } ( s )$ minimizes $\ell ( s , \cdot )$ , natural parametrization gives

$$
\ell ( t , k _ { \ell } ( t ) ) \leqslant \ell ( t , k _ { \ell } ( s ) ) = \ell ( s , k _ { \ell } ( s ) ) - ( t - s ) k _ { \ell } ( s ) ,
$$

and similarly

$$
\ell ( s , k _ { \ell } ( s ) ) \leqslant \ell ( s , k _ { \ell } ( t ) ) = \ell ( t , k _ { \ell } ( t ) ) + ( t - s ) k _ { \ell } ( t ) .
$$

Therefore

$$
- k _ { \ell } ( t ) \leqslant { \frac { \ell ( t , k _ { \ell } ( t ) ) - \ell ( s , k _ { \ell } ( s ) ) } { t - s } } \leqslant - k _ { \ell } ( s ) .
$$

Since $k _ { \ell }$ is Lipschitz, hence continuous, it follows that

$$
{ \frac { d } { d v } } \ell ( v , k _ { \ell } ( v ) ) = - k _ { \ell } ( v ) .
$$

Equivalently, recall from Lemma 3.3 that $h _ { \ell } ( v ) = \ell ( v , k _ { \ell } ( v ) )$ , viewed as a function of $v ,$ is concave with superderivative $h _ { \ell } ^ { \prime } ( v ) = - k _ { \ell } ( v )$ , which in fact must be a derivative by $\beta \mathrm { . }$ smoothness. Thus,

$$
\ell ( v , k _ { \ell } ( v ) ) - \ell ( 0 , k _ { \ell } ( 0 ) ) = - \int _ { 0 } ^ { v } k _ { \ell } ( t ) d t .
$$

Using natural parametrization once more,

$$
\ell ( v , k _ { \ell } ( 0 ) ) = \ell ( 0 , k _ { \ell } ( 0 ) ) - v k _ { \ell } ( 0 ) ,
$$

so

$$
\ell ( v , k _ { \ell } ( v ) ) - \ell ( v , k _ { \ell } ( 0 ) ) = - \int _ { 0 } ^ { v } r _ { \ell } ( t ) d t .
$$

Also,

$$
\ell ( y , a ) - \ell ( v , a ) = ( v - y ) a .
$$

Hence

$$
\varphi _ { \ell } ( x , y , v ) = - \int _ { 0 } ^ { v } r _ { \ell } ( t ) d t + ( v - y ) r _ { \ell } ( v ) .
$$

Consequently, for any $\ell , \ell ^ { \prime } \in \mathcal L _ { B , \beta }$

$$
\| \varphi _ { \ell } - \varphi _ { \ell ^ { \prime } } \| _ { \infty } \leqslant 3 \| r _ { \ell } - r _ { \ell ^ { \prime } } \| _ { \infty } ,
$$

since $| v | \leqslant 1$ and $| v - y | \leqslant 2$

The class of β-Lipschitz functions $r : [ \pm 1 ]  \mathbb { R }$ satisfying $r ( 0 ) = 0$ has an $L _ { \infty } \mathrm { - c o v e r }$ of radius $\varepsilon / 6$ and size

$$
\exp \left( O \left( 1 + { \frac { \beta } { \varepsilon } } \right) \right) .
$$

Indeed, discretize $[ \pm 1 ]$ at spacing $\Theta ( \varepsilon / \beta )$ and quantize function values at resolution $\Theta ( \varepsilon )$ . Since $r ( 0 ) = 0$ and $r$ is β-Lipschitz, there are only $O ( 1 )$ possibilities for each successive quantized value, and there are $O ( 1 + \beta / \varepsilon )$ grid points.

Partition the losses according to the closest cover element and retain one loss from every nonempty part. If $\ell ^ { \prime }$ is the representative of the part containing $\ell ,$ then

$$
\| r _ { \ell } - r _ { \ell ^ { \prime } } \| _ { \infty } \leqslant \frac { \varepsilon } { 3 } ,
$$

and hence

$$
\| \varphi _ { \ell } - \varphi _ { \ell ^ { \prime } } \| _ { \infty } \leqslant \varepsilon .
$$

Therefore, for every predictor $p : \mathcal { X }  [ \pm 1 ]$ ,

$$
\begin{array} { r } { | \mathbb { E } \left[ \varphi _ { \ell } ( \pmb { x } , \pmb { y } , p ( \pmb { x } ) ) - \varphi _ { \ell ^ { \prime } } ( \pmb { x } , \pmb { y } , p ( \pmb { x } ) ) \right] | \leqslant \varepsilon . } \end{array}
$$

This proves the claim. Since the cover does not depend on $\mathcal { D } ,$ the same sparsifying set works simultaneously for every distribution $\mathcal { D }$ □

We complete the proof of Corollary 5.13 by using Lemma 5.14.

Proof of Corollary 5.13. Apply Lemma 5.14 with accuracy $\varepsilon / 8$ to the centered postprocessed losses. The resulting sparsifying set has size

$$
\exp \left( O \left( 1 + { \frac { \beta } { \varepsilon } } \right) \right) ,
$$

and each of its functions is bounded in absolute value by $O ( \beta )$ . The partition-function concentration bound (Theorem 5.7) and a union bound therefore show that, with probability at least $1 - \delta / 2$ conditional on $f$ being a γ-partition function, every function in the sparsifying set changes by at most $O ( \varepsilon )$ under derandomization. By Lemma 5.14, this extends to every $\ell \in \mathcal L _ { B , \beta }$ , so simultaneously for all such $\ell ,$

$$
\begin{array} { r } { \mathbb { E } \left[ \varphi _ { \ell } ( \boldsymbol { x } , \boldsymbol { y } , p ^ { f } ( \boldsymbol { x } ) ) \right] \leqslant \mathbb { E } \left[ \varphi _ { \ell } ( \boldsymbol { x } , \boldsymbol { y } , p ( \boldsymbol { x } ) ) \right] + O ( \varepsilon ) . } \end{array}
$$

Recalling that

$$
\varphi _ { \ell } ( x , y , v ) = \ell ( y , k _ { \ell } ( v ) ) - \ell ( y , k _ { \ell } ( 0 ) ) ,
$$

we obtain

$$
\begin{array} { r } { \mathbb { E } \left[ \ell ( \pmb { y } , k _ { \ell } ( p ^ { f } ( \pmb { x } ) ) ) - \ell ( \pmb { y } , k _ { \ell } ( 0 ) ) \right] \leqslant \mathbb { E } \left[ \ell ( \pmb { y } , k _ { \ell } ( \pmb { p } ( \pmb { x } ) ) ) - \ell ( \pmb { y } , k _ { \ell } ( 0 ) ) \right] + O ( \varepsilon ) . } \end{array}
$$

The centering term is the same on both sides, and therefore cancels:

$$
\begin{array} { r } { \mathbb { E } \left[ \ell ( \pmb { y } , k _ { \ell } ( p ^ { f } ( \pmb { x } ) ) ) \right] \leqslant \mathbb { E } \left[ \ell ( \pmb { y } , k _ { \ell } ( p ( \pmb { x } ) ) ) \right] + O ( \varepsilon ) . } \end{array}
$$

Since $_ p$ is an $( \mathcal { L } _ { B , \beta } , \mathcal { C } , \varepsilon )$ -omnipredictor,

$$
\mathbb { E } \left[ \ell ( \pmb { y } , k _ { \ell } ( \pmb { p } ( \pmb { x } ) ) ) \right] \leqslant \operatorname* { m i n } _ { c \in \mathcal { C } } \mathbb { E } \left[ \ell ( \pmb { y } , c ( \pmb { x } ) ) \right] + \varepsilon .
$$

Combining the previous two inequalities gives

$$
\mathbb { E } \left[ \ell ( \pmb { y } , k _ { \ell } ( p ^ { f } ( \pmb { x } ) ) ) \right] \leqslant \operatorname* { m i n } _ { c \in \mathcal { C } } \mathbb { E } \left[ \ell ( \pmb { y } , c ( \pmb { x } ) ) \right] + O ( \varepsilon )
$$

for every $\ell \in \mathcal { L } _ { B , \beta }$ . Hence $p ^ { f }$ is an $( \mathcal { L } _ { B , \beta } , \mathcal { C } , O ( \varepsilon ) )$ -omnipredictor.

Finally, by the partition-function construction lemma, f can be constructed with probability at least $1 - \delta / 2$ using

$$
O ( \gamma ^ { - 1 } ( \log ( 1 / \gamma ) + \log ( 1 / \delta ) ) )
$$

unlabeled samples from $\mathcal { D } _ { \mathcal { X } }$ . A union bound with the derandomization event above gives overall success probability at least $1 - \delta$ □

## References

[ACRS25] Eshwar Ram Arunachaleswaran, Natalie Collina, Aaron Roth, and Mirah Shi. An elementary predictor obtaining distance to calibration. In Yossi Azar and Debmalya Panigrahi, editors, Proceedings ofthe 2025 Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2025, New Orleans, LA, USA, January 12-15, 2025, pages 1366– 1370. SIAM, 2025.

[AGG<sup>+</sup>25] Gustaf Ahdritz, Aravind Gollakota, Parikshit Gopalan, Charlotte Peale, and Udi Wieder. Provable uncertainty decomposition via higher-order calibration. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

[BDLM01] Shai Ben-David, Philip M Long, and Yishay Mansour. Agnostic boosting. In International Conference on Computational Learning Theory, pages 507–516. Springer, 2001.

[BGHN23] Jaros law B lasiok, Parikshit Gopalan, Lunjia Hu, and Preetum Nakkiran. A unifying theory of distance from calibration. In Proceedings ofthe 55th Annual ACM Symposium on Theory of Computing, pages 1727–1740, 2023.

[BN24] Jaroslaw Blasiok and Preetum Nakkiran. Smooth ECE: Principled reliability diagrams via kernel smoothing. In The Twelfth International Conference on Learning Representations (ICLR), 2024.

[CBL06] Nicol\`o Cesa-Bianchi and G´abor Lugosi. Prediction, learning, and games. Cambridge University Press, 2006.

[CDV24] S´ılvia Casacuberta, Cynthia Dwork, and Salil Vadhan. Complexity-theoretic implications of multicalibration. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing (STOC), pages 1071–1082, 2024.

[CGK<sup>+</sup>26] S´ılvia Casacuberta, Parikshit Gopalan, Varun Kanade, Omer Reingold, Konstantinos Stavropoulos, and Pranay Tankala. Multigroup fairness and omniprediction: Separations and equivalences. In The Annual Conference on Neural Information Processing Systems (NeurIPS’26), 2026.

[CGKR25] S´ılvia Casacuberta, Parikshit Gopalan, Varun Kanade, and Omer Reingold. How global calibration strengthens multiaccuracy. In 2025 IEEE 66th Annual Symposium on Foundations of Computer Science (FOCS), pages 1198–1227. IEEE, 2025.

[Daw82] A. P. Dawid. Objective probability forecasts. University College London, Dept. of Statistical Science. Research Report 14, 1982.

[Daw85] A Philip Dawid. Calibration-based empirical probability. The Annals of Statistics, pages 1251–1274, 1985.

[DHI<sup>+</sup>25] Cynthia Dwork, Chris Hays, Nicole Immorlica, Juan C. Perdomo, and Pranay Tankala. From fairness to infinity: Outcome-indistinguishable (omni)prediction in evolving graphs. In The Thirty Eighth Annual Conference on Learning Theory (COLT), volume 291 of Proceedings of Machine Learning Research, pages 1564–1637. PMLR, 2025.

[DKPZ21] Ilias Diakonikolas, Daniel M Kane, Thanasis Pittas, and Nikos Zarifis. The optimality of polynomial regression for agnostic learning under gaussian marginals in the SQ model. In Proceedings of The 34<sup>th</sup> Conference on Learning Theory, COLT, pages 1552–1584. PMLR, 2021.

[DKR<sup>+</sup>21] Cynthia Dwork, Michael P. Kim, Omer Reingold, Guy N. Rothblum, and Gal Yona. Outcome indistinguishability. In ACM Symposium on Theory of Computing (STOC), 2021.

[DLLT23] Cynthia Dwork, Daniel Lee, Huijia Lin, and Pranay Tankala. From pseudorandomness to multi-group fairness and back. In The Thirty-Sixth Annual Conference on Learning Theory (COLT), volume 195 of Proceedings of Machine Learning Research, pages 3566– 3614. PMLR, 2023.

[DSFT<sup>+</sup>15] Dana Dachman-Soled, Vitaly Feldman, Li-Yang Tan, Andrew Wan, and Karl Wimmer. Approximate resilience, monotonicity, and the complexity of agnostic learning. In Piotr Indyk, editor, Proceedings of the twenty-sixth annual ACM-SIAM symposium on Discrete algorithms, pages 498–511. SIAM, 2015.

[DT26] Cynthia Dwork and Pranay Tankala. Eficient and private property testing via indistinguishability. In Symposium on Foundations of Computer Science (FOCS). IEEE, 2026.

[Fel09] Vitaly Feldman. A complete characterization of statistical query learning with applications to evolvability. In 2009 50th Annual IEEE Symposium on Foundations of Computer Science, pages 375–384, 2009.

[Fel10] Vitaly Feldman. Distribution-specific agnostic boosting. In Innovations in Computer Science - ICS 2010, pages 241–250, 2010.

[FGR<sup>+</sup>17] Vitaly Feldman, Elena Grigorescu, Lev Reyzin, Santosh S Vempala, and Ying Xiao. Statistical algorithms and a lower bound for detecting planted cliques. Journal of the ACM (JACM), 64(2):1–37, 2017.

[FGV17] V. Feldman, C. Guzm´an, and S. S. Vempala. Statistical query algorithms for mean vector estimation and stochastic convex optimization. In Philip N. Klein, editor, Proceedings of the Twenty-Eighth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2017, pages 1265–1277. SIAM, 2017.

[FH18] Dean P. Foster and Sergiu Hart. Smooth calibration, leaky forecasts, finite recall, and Nash dynamics. Games Econ. Behav., 109:271–293, 2018.

[FK96] A. Frieze and R. Kannan. The regularity lemma and approximation schemes for dense problems. In Proceedings of 37th Conference on Foundations of Computer Science, pages 12–20, October 1996. ISSN: 0272-5428.

[GHK<sup>+</sup>23] Parikshit Gopalan, Lunjia Hu, Michael P. Kim, Omer Reingold, and Udi Wieder. Loss minimization through the lens of outcome indistinguishability. In Innovations in Theoretical Computer Science (ITCS), 2023.

[GHN26] Ira Globus-Harris and Inbal Livni Navon. Augmented feature boosting for multicalibration. arXiv preprint arXiv:2609.33891, 2026.

[GJRR24] Sumegha Garg, Christopher Jung, Omer Reingold, and Aaron Roth. Oracle eficient online multicalibration and omniprediction. In David P. Woodruf, editor, Proceedings of the 2024 ACM-SIAM Symposium on Discrete Algorithms, SODA 2024, Alexandria, VA, USA, January 7-10, 2024, pages 2725–2792. SIAM, 2024.

[GKR<sup>+</sup>22] Parikshit Gopalan, Adam Tauman Kalai, Omer Reingold, Vatsal Sharan, and Udi Wieder. Omnipredictors. In Innovations in Theoretical Computer Science (ITCS), 2022.

[GKR23] Parikshit Gopalan, Michael Kim, and Omer Reingold. Swap agnostic learning, or characterizing omniprediction via multicalibration. Advances in Neural Information Processing Systems, 36, 2023.

[GKS26] Parikshit Gopalan, Satyen Kale, and Spandan Senapati. Personal communication, 2026.

[GKSZ22] Parikshit Gopalan, Michael P. Kim, Mihir Singhal, and Shengjia Zhao. Low-degree multicalibration. In Conference on Learning Theory, 2-5 July 2022, London, UK, volume 178 of Proceedings of Machine Learning Research, pages 3193–3234. PMLR, 2022.

[GSTT26] Parikshit Gopalan, Konstantinos Stavropoulos, Kunal Talwar, and Pranay Tankala. The importance of being smoothly calibrated. In Huijia (Rachel) Lin, editor, 7th Symposium on Foundations of Responsible Computing, FORC 2026, Harvard University, Cambridge, MA, USA, June 3-5, 2026, volume 368 of LIPIcs, pages 21:1–21:22. Schloss Dagstuhl - Leibniz-Zentrum f¨ur Informatik, 2026.

[GT26] Isaac Gibbs and Ryan J. Tibshirani. Sample-eficient omniprediction for proper losses. In Steve Hanneke and Tor Lattimore, editors, The Thirty Ninth Annual Conference on Learning Theory (COLT), volume 336 of Proceedings of Machine Learning Research, pages 2679–2719. PMLR, 2026.

[Han57] James Hannan. Approximation to Bayes risk in repeated play. In Melvin Dresher, A. W. Tucker, and Philip Wolfe, editors, Contributions to the Theory ofGames, Volume III, number 39 in Annals of Mathematics Studies, pages 97–139. Princeton University Press, 1957.

[Hau92] David Haussler. Decision theoretic generalizations of the PAC model for neural net and other learning applications. Information and Computation, 100(1):78–150, 1992.

[HKRR18] Ursula H´ebert-Johnson, Michael P. Kim, Omer Reingold, and Guy N. Rothblum. Mul-<sup>´</sup> ticalibration: Calibration for the (computationally-identifiable) masses. In Proceedings of the 35th International Conference on Machine Learning (ICML), pages 1939–1948. PMLR, 2018.

[HLNR23] Lunjia Hu, Inbal Rachel Livni Navon, and Omer Reingold. Generative models of huge objects. In 38th Computational Complexity Conference (CCC 2023), volume 264 of Leibniz International Proceedings in Informatics (LIPIcs), pages 5:1–5:20, 2023.

[HWY25] Jason D. Hartline, Yifan Wu, and Yunran Yang. Smooth calibration and decision making. In 6th Symposium on Foundations of Responsible Computing, FORC 2025, volume 329 of LIPIcs, pages 16:1–16:26, 2025.

[JB26] Cameron R. Jones and Benjamin K. Bergen. Large language models pass a standard three-party Turing test. Proceedings of the National Academy of Sciences, 123(21):e2524472123, 2026.

[Kea98] Michael Kearns. Eficient noise-tolerant learning from statistical queries. J. ACM, 45(6):983–1006, November 1998.

[KGZ19] Michael P. Kim, Amirata Ghorbani, and James Zou. Multiaccuracy: Black-box postprocessing for fairness in classification. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, pages 247–254, 2019.

[KK09] Adam Kalai and Varun Kanade. Potential-based agnostic boosting. In Advances in Neural Information Processing Systems, volume 22. Curran Associates, Inc., 2009.

[KKMS08] Adam Tauman Kalai, Adam R. Klivans, Yishay Mansour, and Rocco A. Servedio. Agnostically learning halfspaces. SIAM J. Comput., 37(6):1777–1805, 2008.

[KMV08] Adam Tauman Kalai, Yishay Mansour, and Elad Verbin. On agnostic boosting and parity learning. In Proceedings of the 40th Annual ACM Symposium on Theory of Computing, pages 629–638. ACM, 2008.

[KSS94] Michael J Kearns, Robert E Schapire, and Linda M Sellie. Toward eficient agnostic learning. Machine Learning, 17(2):115–141, 1994.

[KZ20] Andrey Kupavskii and Nikita Zhivotovskiy. When are epsilon-nets small? Journal of Computer and System Sciences, 110:22–36, 2020.

[LMN93] Nathan Linial, Yishay Mansour, and Noam Nisan. Constant depth circuits, Fourier transform, and learnability. Journal of the ACM (JACM), 40(3):607–620, July 1993.

[LSS25] Haipeng Luo, Spandan Senapati, and Vatsal Sharan. Improved bounds for swap multicalibration and swap omniprediction. arXiv preprint arXiv:2505.20885, 2025.

[LV25] Jane Lange and Arsen Vasilyan. Robust learning of halfspaces under log-concave marginals. Advances in Neural Information Processing Systems, 38:100700–100744, 2025.

[McC56] John McCarthy. Measures of the value of information. Proceedings of the National Academy of Sciences, 42(9):654–655, 1956.

[MM02] Yishay Mansour and David McAllester. Boosting using branching programs. Journal of Computer and System Sciences, 64(1):103–112, 2002.

[MPV25] Cassandra Marcussen, Aaron Putterman, and Salil Vadhan. Characterizing the distinguishability of product distributions through multicalibration. In 40th Computational Complexity Conference (CCC 2025), 2025.

[NR26] Georgy Noarov and Aaron Roth. Optimal deterministic multicalibration and omniprediction, 2026.

[OKK25] Princewill Okoroafor, Robert Kleinberg, and Michael P. Kim. Near-optimal algorithms for omniprediction. In 2025 IEEE 66th Annual Symposium on Foundations of Computer Science (FOCS), pages 1595–1609. IEEE, 2025.

[QZ24] Mingda Qiao and Letian Zheng. On the distance from calibration in sequential prediction. In Shipra Agrawal and Aaron Roth, editors, The Thirty Seventh Annual Conference on Learning Theory, June 30 - July 3, 2024, Edmonton, Canada, volume 247 of Proceedings of Machine Learning Research, pages 4307–4357. PMLR, 2024.

[Sav71] Leonard J Savage. Elicitation of personal probabilities and expectations. Journal of the American Statistical Association, 66(336):783–801, 1971.

[Sta26] Konstantinos Stavropoulos. Computational foundations of reliable learning. PhD thesis, 2026.

[Sze75] Endre Szemeredi. Regular partitions of graphs. Technical report, Stanford University, Stanford, CA, USA, March 1975.

[TTV09] Luca Trevisan, Madhur Tulsiani, and Salil P. Vadhan. Regularity, boosting, and eficiently simulating every high-entropy distribution. In Proceedings of the 24th Annual IEEE Conference on Computational Complexity (CCC), pages 126–136. IEEE Computer Society, 2009.

[Tur50] Alan M. Turing. Computing machinery and intelligence. Mind, LIX(236):433–460, 1950.

[Val84] L. G. Valiant. A theory of the learnable. Commun. ACM, 27(11):1134–1142, November 1984.

## A Deferred Proofs From Section 3

Proof of Lemma 3.3. First, observe that the proper loss function $\ell ( p , k _ { \ell } ( q ) )$ is afine in $p$ with coeficient $\partial \ell ( k _ { \ell } ( q ) )$ ). Also, by definition, $h _ { \ell } ( q ) = \ell ( q , k _ { \ell } ( q ) )$ . These two simple constraints imply that $\ell ( p , k _ { \ell } ( q ) )$ , viewed as a function of $p ,$ is an afine upper bound on $h _ { \ell } ( p )$ of the form

$$
\ell ( p , k _ { \ell } ( q ) ) = h _ { \ell } ( q ) + h _ { \ell } ^ { \prime } ( q ) ( p - q ) ,
$$

where $h _ { \ell } ^ { \prime } = \partial \ell \circ k _ { \ell }$ . Therefore, $h _ { \ell } ^ { \prime }$ is a superderivative of $h _ { \ell } .$ . In this notation, it is easy to see that the divergence $D _ { \ell }$ satisfies

$$
\begin{array} { r l } & { D _ { \ell } ( p , k _ { \ell } ( q ) ) = \ell ( p , k _ { \ell } ( q ) ) - h _ { \ell } ( p ) } \\ & { \qquad = h _ { \ell } ( q ) + h _ { \ell } ^ { \prime } ( q ) ( p - q ) - h _ { \ell } ( p ) } \\ & { \qquad = D _ { h _ { \ell } } ( p , q ) . } \end{array}
$$

Proof of Lemma 3.4. The definition of $D _ { \ell }$ states that for any action $a ^ { \prime } { . }$

$$
\ell ( \bar { p } _ { T } , a ^ { \prime } ) = \operatorname* { m i n } _ { a } \ell ( \bar { p } _ { T } , a ) + D _ { \ell } ( \bar { p } _ { T } , a ^ { \prime } ) .
$$

Since ℓ is afine in its first input and $\bar { p } _ { T } = ( w _ { 1 } p _ { 1 } + \cdot \cdot \cdot + w _ { T } p _ { T } ) / W _ { T }$ , we may rewrite this as

$$
\sum _ { t = 1 } ^ { T } w _ { t } \ell ( p _ { t } , a ^ { \prime } ) = \operatorname* { m i n } _ { a } \sum _ { t = 1 } ^ { T } w _ { t } \ell ( p _ { t } , a ) + W _ { T } D _ { \ell } ( \bar { p } _ { T } , a ^ { \prime } ) .
$$

Separating the final term from the sum on the left gives

$$
\sum _ { t = 1 } ^ { T - 1 } w _ { t } \ell ( p _ { t } , a ^ { \prime } ) + w _ { T } \ell ( p _ { T } , a ^ { \prime } ) = \operatorname* { m i n } _ { a } \sum _ { t = 1 } ^ { T } w _ { t } \ell ( p _ { t } , a ) + W _ { T } D _ { \ell } ( \bar { p } _ { T } , a ^ { \prime } ) .
$$

If we choose $a ^ { \prime } = k _ { \ell } ( \bar { p } _ { T - 1 } )$ to minimize the sum of the first $T - 1$ terms on the left, this becomes

$$
\operatorname* { m i n } _ { a } \sum _ { t = 1 } ^ { T - 1 } w _ { t } \ell ( p _ { t } , a ) + w _ { T } \ell \big ( p _ { T } , k _ { \ell } ( \bar { p } _ { T - 1 } ) \big ) = \operatorname* { m i n } _ { a } \sum _ { t = 1 } ^ { T } w _ { t } \ell ( p _ { t } , a ) + W _ { T } D _ { \ell } \big ( \bar { p } _ { T } , k _ { \ell } ( \bar { p } _ { T - 1 } ) \big ) .
$$

Continuing by induction on $T$ completes the proof.

Although the preceding proof of the Follow the Leader regret bound is the standard one, a helpful observation is that the proof can alternatively be phrased in terms of a potential function $W _ { t } h ( { \bar { p } } _ { t } )$ , whose change in value we track across iterations. This point of view will prove to be useful for the lower bound constructions. There, we construct examples where the algorithm requires many iterations by carefully controlling the potential gap from one iteration to the next.

Alternate Proof of Lemma $\ 3 . 4 \cdot$ Consider the superderivative $h ^ { \prime } = \partial \ell \circ k _ { \ell }$ of the concave function $h = h _ { \ell }$ . For any two points $u , v \in \mathbb { R }$ , we have the first-order approximation

$$
h ( v ) - h ( u ) = h ^ { \prime } ( u ) ( v - u ) - D _ { h } ( v , u ) .
$$

Substituting $u = \bar { p } _ { t - 1 }$ and $v = \bar { p } _ { t }$ and the update rule $\bar { p } _ { t } - \bar { p } _ { t - 1 } = ( w _ { t } / W _ { t } ) ( p _ { t } - \bar { p } _ { t - 1 } )$ yields

$$
h ( \bar { p } _ { t } ) - h ( \bar { p } _ { t - 1 } ) = \frac { w _ { t } } { W _ { t } } h ^ { \prime } ( \bar { p } _ { t - 1 } ) ( p _ { t } - \bar { p } _ { t - 1 } ) - D _ { h } ( \bar { p } _ { t } , \bar { p } _ { t - 1 } ) .
$$

Rearranging terms and using the fact that $\ell ( p , k _ { \ell } ( q ) ) = h ( q ) + h ^ { \prime } ( q ) ( p - q )$ (Lemma 3.3) yields

$$
W _ { t } \ : h ( \bar { p } _ { t } ) - W _ { t - 1 } \ : h ( \bar { p } _ { t - 1 } ) = w _ { t } \ell ( p _ { t } , k _ { \ell } ( \bar { p } _ { t - 1 } ) ) - W _ { t } \ : D _ { h } ( \bar { p } _ { t } , \bar { p } _ { t - 1 } ) .
$$

Summing over $t = 1 , \dots , T$ , the left-hand side telescopes, leaving

$$
W _ { T } h ( \bar { p } _ { T } ) = \sum _ { t = 1 } ^ { T } w _ { t } \ell ( p _ { t } , k _ { \ell } ( \bar { p } _ { t - 1 } ) ) - \sum _ { t = 1 } ^ { T } W _ { t } D _ { h } ( \bar { p } _ { t } , \bar { p } _ { t - 1 } ) .
$$

Finally, observe that $\begin{array} { r } { W _ { T } h ( \bar { p } _ { T } ) = \operatorname* { m i n } _ { a } \sum _ { t = 1 } ^ { T } w _ { t } \ell ( p _ { t } , a ) } \end{array}$ by the definition of h. □

Proof of Lemma 3.5. By Lemma 3.3, the proper loss $\ell ( p , k _ { \ell } ( q ) )$ can be written in the form

$$
\ell ( p , k _ { \ell } ( q ) ) = h _ { \ell } ( q ) + h _ { \ell } ^ { \prime } ( q ) ( p - q ) ,
$$

where $h _ { \ell } ^ { \prime } = \partial \ell \circ k _ { \ell }$ is a superderivative of $h _ { \ell } .$ . In the special case that $h _ { \ell } ^ { \prime }$ is β-Lipschitz, the function $h _ { \ell }$ must be β-smooth, meaning that

$$
D _ { \ell } ( p , k _ { \ell } ( q ) ) = D _ { h _ { \ell } } ( p , q ) \leqslant { \frac { \beta } { 2 } } ( p - q ) ^ { 2 } .
$$

Using the update rule $\bar { p } _ { t } - \bar { p } _ { t - 1 } = ( w _ { t } / W _ { t } ) ( p _ { t } - \bar { p } _ { t - 1 } )$ and the fact that predictions are [±1]-valued,

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } W _ { t } D _ { \ell } \big ( \bar { p } _ { t } , k _ { \ell } \big ( \bar { p } _ { t - 1 } \big ) \big ) \leqslant \displaystyle \sum _ { t = 1 } ^ { T } W _ { t } \cdot \frac { \beta } { 2 } \big ( \bar { p } _ { t } - \bar { p } _ { t - 1 } \big ) ^ { 2 } } \\ & { \displaystyle \leqslant \displaystyle \sum _ { t = 1 } ^ { T } W _ { t } \cdot \frac { \beta } { 2 } \Big ( \frac { w _ { t } } { W _ { t } } \big ( p _ { t } - \bar { p } _ { t - 1 } \big ) \Big ) ^ { 2 } } \\ & { \displaystyle \leqslant 2 \beta \displaystyle \sum _ { t = 1 } ^ { T } \frac { w _ { t } ^ { 2 } } { W _ { t } } . } \end{array}
$$

Proof of Lemma 3.6. We consider the three cases separately.

• (Uniform) When $w _ { t } = 1$ , we have

$$
\delta = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { 1 } { t } \leqslant \frac { 1 + \log ( T ) } { T } .
$$

• (Linear) When $w _ { t } = t$ , we have

$$
\delta = \frac { 2 } { T ( T + 1 ) } \sum _ { t = 1 } ^ { T } \frac { 2 t } { t + 1 } \leqslant \frac { 4 } { T + 1 } .
$$

• (Geometric) When we normalize so that $W _ { T } = 1$ , we have

$$
\begin{array} { r } { w _ { t } = \left\{ \begin{array} { l l } { ( 1 - \lambda ) ^ { T - 1 } } & { \mathrm { i f ~ } t = 1 , } \\ { \lambda ( 1 - \lambda ) ^ { T - t } } & { \mathrm { i f ~ } t \geqslant 2 , } \end{array} \right. } \end{array}
$$

and hence

$$
\begin{array} { l } { \displaystyle \delta = ( 1 - \lambda ) ^ { T - 1 } + \sum _ { t = 2 } ^ { T } \lambda ^ { 2 } ( 1 - \lambda ) ^ { T - t } } \\ { = ( 1 - \lambda ) ^ { T - 1 } + \lambda ^ { 2 } \Big ( \frac { 1 - ( 1 - \lambda ) ^ { T - 1 } } { 1 - ( 1 - \lambda ) } \Big ) } \\ { = \lambda + ( 1 - \lambda ) ^ { T } . } \end{array}
$$

This is at most $\lambda + e ^ { - \lambda T }$ , which is $O ( \log ( T ) / T )$ for $\lambda = \log ( T ) / T$

Proof of Lemma 3.7. Let $\begin{array} { r } { W ( p ) = \int _ { 0 } ^ { p } w ( q ) d q } \end{array}$ . Since w is β-Lipschitz, W is β-smooth. Using the update rule

$$
\bar { p } _ { t } - \bar { p } _ { t - 1 } = \lambda ( p _ { t } - \bar { p } _ { t - 1 } ) ,
$$

which holds for $t \geqslant 2$ , along with β-smoothness of $W .$ , we see that

$$
\begin{array} { l } { { \displaystyle W ( \bar { p } _ { t } ) - W ( \bar { p } _ { t - 1 } ) \geqslant w ( \bar { p } _ { t - 1 } ) ( \bar { p } _ { t } - \bar { p } _ { t - 1 } ) - \frac { \beta } { 2 } ( \bar { p } _ { t } - \bar { p } _ { t - 1 } ) ^ { 2 } \ ~ } } \\ { { \displaystyle \qquad \geqslant \ \lambda w ( \bar { p } _ { t - 1 } ) ( p _ { t } - \bar { p } _ { t - 1 } ) - \frac { \beta \lambda ^ { 2 } } { 2 } ( p _ { t } - \bar { p } _ { t - 1 } ) ^ { 2 } } . } \end{array}
$$

Summing over t so that the left-hand side telescopes, and then rearranging terms, yields

$$
\sum _ { t = 2 } ^ { T } w ( \bar { p } _ { t - 1 } ) ( p _ { t } - \bar { p } _ { t - 1 } ) \leqslant \frac { W ( \bar { p } _ { T } ) - W ( \bar { p } _ { 1 } ) } { \lambda } + 2 \beta \lambda ( T - 1 ) .
$$

Note that the numerator on the right-hand side is bounded since W is bounded. Also, the same argument goes through with −w in place of w, completing the proof. (Note that the update rule is actually slightly diferent for the edge case $t = 1$ , but this diference is inconsequential, contributing an additional error of at most $O ( 1 )$ in the worst case, or $O ( 1 / T )$ after dividing through by T.)

Proof of Lemma 3.8. As in the proof of Lemma 3.5, if the loss function has the form

$$
\ell ( p , \eta ) = g ( \eta ) - p \eta
$$

for some function $^ { g , }$ then the proper loss $\ell ( p , k _ { \ell } ( q ) )$ can be written as

$$
\ell ( p , k _ { \ell } ( q ) ) = h ( q ) + h ^ { \prime } ( q ) ( p - q ) ,
$$

where $h ( q ) = \operatorname* { i n f } _ { \eta } \ell ( q , \eta )$ is the negation of the convex conjugate of $g$ and where $h ^ { \prime } ( q ) = - k _ { \ell } ( q )$ is a superderivative of $h .$ Our goal is to construct slightly modified functions $g _ { \sigma } , h _ { \sigma } , \ell _ { \sigma }$ that satisfy the following corresponding pair of identities:

$$
\begin{array} { c } { { \ell _ { \sigma } ( p , \eta ) = g _ { \sigma } ( \eta ) - p \eta } } \\ { { \ell _ { \sigma } ( p , k _ { \ell _ { \sigma } } ( q ) ) = h _ { \sigma } ( q ) + h _ { \sigma } ^ { \prime } ( q ) ( p - q ) . } } \end{array}
$$

We construct these modified functions in three steps:

1. First, we set $h _ { \sigma } ^ { \prime } ( p ) = \mathbb { E } [ h ^ { \prime } ( N _ { \sigma } ( p ) ) ]$

2. Next, we let $h _ { \sigma }$ be the antiderivative of $h _ { \sigma } ^ { \prime }$ with $h _ { \sigma } ( 0 ) = h ( 0 )$

3. Finally, we let $g _ { \sigma }$ be the convex conjugate of $- h _ { \sigma }$ , or more precisely

$$
g _ { \sigma } ( \eta ) = \operatorname* { s u p } _ { p \in [ \pm 1 ] } p \eta + h _ { \sigma } ( p ) .
$$

This also uniquely determines $\ell _ { \sigma }$

To build intuition for the construction of $\ell _ { \sigma }$ from $\ell ,$ observe that $h _ { \sigma } ^ { \prime }$ is a Lipschitz and piecewiselinear approximation of $h ^ { \prime } .$ Specifically, we obtain $h _ { \sigma } ^ { \prime }$ from $h ^ { \prime }$ by interpolating points on the graph of $h ^ { \prime }$ at horizontal spacing $\sigma .$ Clearly, if $h ^ { \prime }$ is [±B]-valued, then this interpolation $h _ { \sigma } ^ { \prime }$ is $2 B / \sigma \mathrm { - }$ Lipschitz. Accordingly, the antiderivative $h _ { \sigma }$ is $2 B /$ σ-smooth. Moreover, one can check that h and $h _ { \sigma }$ difer pointwise by at most $B \sigma$ . Indeed, by concavity, $h ^ { \prime }$ and $h _ { \sigma } ^ { \prime }$ are both nonincreasing, so if we set $k = \sigma ^ { - 1 }$ , then

$$
\begin{array} { c l } { { | h ( p ) - h _ { \sigma } ( p ) | \leqslant \displaystyle \sum _ { j = - k } ^ { k - 1 } \displaystyle \int _ { j \sigma } ^ { ( j + 1 ) \sigma } | h ^ { \prime } ( q ) - h _ { \sigma } ^ { \prime } ( q ) | d q } } \\ { { \leqslant \displaystyle \sum _ { j = - k } ^ { k - 1 } \displaystyle \frac { \sigma } { 2 } \big ( h ^ { \prime } ( j \sigma ) - h ^ { \prime } ( ( j + 1 ) \sigma ) \big ) } } \\ { { = \displaystyle \frac { \sigma } { 2 } ( h ^ { \prime } ( - 1 ) - h ^ { \prime } ( 1 ) ) } } \\ { { \leqslant B \sigma . } } \end{array}
$$

Since $h$ and $h _ { \sigma }$ difer pointwise by at most $B \sigma .$ , so do their convex conjugates. In particular, $g _ { \sigma }$ difers pointwise by at most Bσ from a convex lower bound on $g \colon$

$$
g _ { \sigma } ( \eta ) \leqslant \operatorname* { s u p } _ { p \in [ \pm 1 ] } p \eta + h ( p ) + B \sigma \leqslant g ( \eta ) + B \sigma .
$$

Thus, for all inputs $p \in [ \pm 1 ]$ and $\eta \in \mathbb { R }$ , we have

$$
\ell _ { \sigma } ( p , \eta ) \leqslant \ell ( p , \eta ) + B \sigma .
$$

Similarly, it is easy to check that for all inputs $p , q \in [ \pm 1 ]$ , the expected value of $\ell ( p , k _ { \ell } ( N _ { \sigma } ( q ) ) )$ over the random noise introduced by $N _ { \sigma }$ is close to the smoothed proper loss $\ell _ { \sigma } ( p , k _ { \ell _ { \sigma } } ( q ) )$

$$
\begin{array} { r l r l } { \mathbb { E } [ \ell ( p , k _ { \ell } ( N _ { \sigma } ( q ) ) ) ] = \mathbb { E } \big [ h ( N _ { \sigma } ( q ) ) + h ^ { \prime } ( N _ { \sigma } ( q ) ) ( p - N _ { \sigma } ( q ) ) \big ] } \\ & { \approx _ { \mathcal { O } ( B \sigma ) } h ( q ) + \mathbb { E } [ h ^ { \prime } ( N _ { \sigma } ( q ) ) ] ( p - q ) } & & { \mathrm { ~ s i n c e ~ } h \mathrm { ~ i s ~ } B \mathrm { - } \mathrm { L i p s c h i t z } } \\ & { \approx _ { \mathcal { O } ( B \sigma ) } h _ { \sigma } ( q ) + h _ { \sigma } ^ { \prime } ( q ) ( p - q ) } & & { \mathrm { ~ b y ~ d e f i n i t i o n ~ o f ~ } h _ { \sigma } , h _ { \sigma } ^ { \prime } } \\ & { = \ell _ { \sigma } ( p , k _ { \ell _ { \sigma } } ( q ) ) . } & & { \square } \end{array}
$$

Proof of Lemma 3.9. Given $c \in \mathcal { C } ^ { \mathrm { { s w a p } } }$ and $v \in [ \pm 1 ]$ and $x \in \mathcal { X }$ , define

$$
c _ { \sigma } ( v , x ) = \mathbb { E } [ c ( N _ { \sigma } ( v ) , x ) ] .
$$

Intuitively, $c _ { \sigma } ( v , x )$ is a piecewise-linear approximation of c obtained by interpolating points on the graph of c at horizontal spacing σ. Also, the function $c _ { \sigma }$ belongs to $\mathcal { C } ^ { \mathrm { s w a p } }$ by our assumption throughout this paper that C is closed under convex combinations. Clearly, $c _ { \sigma }$ is $O ( 1 / \sigma )$ -Lipschitz in its first input and

$$
\left| \mathbb { E } [ c ( N _ { \sigma } ( v ) , x ) ( y - N _ { \sigma } ( v ) ) ] - c _ { \sigma } ( v , x ) ( y - v ) \right| \leqslant O ( \sigma ) .
$$

## B Deferred Proofs From Section 4

Proof of Lemma $4 . 6 .$ By symmetry, $\mathbb { E } [ x _ { i } ^ { 2 } ] = 1$ and $\mathbb { E } [ { \pmb x } _ { i } { \pmb x } _ { j } ] = 0$ for $i \neq j$ . Therefore,

$$
\begin{array} { r } { \mathbb { E } \big [ \langle a , { \pmb x } \rangle \langle b , { \pmb x } \rangle \big ] = \mathbb { E } [ a ^ { \top } { \pmb x } { \pmb x } ^ { \top } b ] = a ^ { \top } I _ { n } b = \langle a , b \rangle . } \end{array}
$$

Next, we decompose $a = a ^ { \parallel } + a ^ { \perp }$ where $a ^ { | | } = \langle a , b \rangle b$ is parallel to b and $a ^ { \perp }$ is perpendicular to $b .$ By symmetry, $\langle a , \pmb { x } \rangle = \langle a ^ { \| } + a ^ { \perp } , \pmb { x } \rangle$ has the same distribution as $\langle a ^ { \parallel } - a ^ { \perp } , { \pmb x } \rangle$ , so

$$
\mathbb { E } \big [ \langle a , \pmb { x } \rangle \bigm | \langle b , \pmb { x } \rangle \bigm ] = \mathbb { E } \Big [ \frac { \langle a ^ { \lVert } + a ^ { \perp } , \pmb { x } \rangle } { 2 } + \frac { \langle a ^ { \lVert } - a ^ { \perp } , \pmb { x } \rangle } { 2 } \bigm | \langle b , \pmb { x } \rangle \Big ] = \langle a ^ { \lVert } , \pmb { x } \rangle .
$$

Finally, we use the standard fact that $\langle b , \pmb { x } \rangle ^ { 2 } / n \sim \mathrm { B e t a } ( 1 / 2 , ( n - 1 ) / 2 )$ and hence

$$
C _ { n } = \mathbb { E } \big [ | \langle b , \pmb { x } \rangle | \big ] = \sqrt { \frac { n } { \pi } } \cdot \frac { \Gamma ( n / 2 ) } { \Gamma ( ( n + 1 ) / 2 ) } \in \Big [ \sqrt { \frac { 2 } { \pi } } , 1 \Big ] .
$$

For predictors in the linear family, there is a clean characterization of valid sequences. We state it in the following lemma, which has a simple proof using Lemma 4.6.

Lemma B.1. Suppose that $q _ { 1 } , \dotsc , q _ { T } \in \mathbb { R } ^ { n }$ satisfy the following conditions with respect to $p ^ { * } \in \mathbb { R } ^ { n }$

• (Boundedness) $\| q _ { t } \| \leqslant 1$ for all t and $\| q _ { t } - q _ { t - 1 } \| \leqslant \lambda$ for all $t \geqslant 2$

• (Quadratic Potential Gap) $\begin{array} { r } { \frac 1 2 \| \boldsymbol { q } _ { t } \| ^ { 2 } - \frac 1 2 \| \boldsymbol { q } _ { t - 1 } \| ^ { 2 } = \lambda \left. \boldsymbol { q } _ { t - 1 } , \boldsymbol { p } ^ { * } - \boldsymbol { q } _ { t - 1 } \right. + \frac 1 2 \| \boldsymbol { q } _ { t } - \boldsymbol { q } _ { t - 1 } \| ^ { 2 } . } \end{array}$

Then the sequence $q _ { t } ( x ) = \langle q _ { t } , x \rangle$ is valid as per Definition 4.2 for $\pmb { x } \sim \mathrm { U n i f o r m } ( \sqrt { n } \cdot S ^ { n - 1 } )$ , up to rescaling by a factor $o f O ( 1 / { \sqrt { n } } )$

Proof of Lemma B.1. Define $r _ { 1 } ~ = ~ q _ { 1 }$ and $r _ { t } = q _ { t - 1 } + ( q _ { t } - q _ { t - 1 } ) / \lambda$ for $t = 2 , \dots , T$ , which is [±1]-valued because

$$
\lVert r _ { t } \rVert \leqslant \lVert q _ { t } \rVert + \frac { 1 } { \lambda } \lVert q _ { t } - q _ { t - 1 } \rVert \leqslant 1 + 1 = 2 .
$$

Next, consider the following algebraic identity, which holds for all points $q _ { t } , q _ { t - 1 } \colon$

$$
\frac 1 2 \| q _ { t } \| ^ { 2 } - \frac 1 2 \| q _ { t - 1 } \| ^ { 2 } = \langle q _ { t - 1 } , q _ { t } - q _ { t - 1 } \rangle + \frac 1 2 \| q _ { t } - q _ { t - 1 } \| ^ { 2 } .
$$

Using the definition of $r _ { t }$ , we can rewrite this as

$$
\frac { 1 } { 2 } \| q _ { t } \| ^ { 2 } - \frac { 1 } { 2 } \| q _ { t - 1 } \| ^ { 2 } = \lambda \left. q _ { t - 1 } , r _ { t } - q _ { t - 1 } \right. + \frac { 1 } { 2 } \| q _ { t } - q _ { t - 1 } \| ^ { 2 } .
$$

Subtracting this from the potential gap condition in this lemma’s statement yields

$$
0 = \lambda \langle q _ { t - 1 } , p ^ { * } - r _ { t } \rangle ,
$$

which by Lemma 4.6 is precisely the condition for a valid $q _ { t }$ sequence from Definition 4.2. □

We now present a unified proof of our various lower bounds. We begin with two examples that give lower bounds of $\Omega ( \lambda )$ and Ω(min $\{ 1 , 1 / ( \lambda T ) \} )$ on degree-2 calibration error and $\Omega ( { \sqrt { \lambda } } )$ on expected calibration error. Together, they imply Lemmas 4.3 and 4.4.

![](images/6e02635a8aea0c073ffe1702ef304260b06fb92725031186ebed6938a4e6c6ef.jpg)  
Figure 1: Examples B.2 to B.4.

Example B.2 (Infinite Sequence with Small Error). Let $q _ { 1 } , q _ { 2 } , . . .$ . lie on a circle of radius $\rho = \sqrt { \lambda }$ centered at the origin in $\mathbb { R } ^ { 2 }$ . Obtain $q _ { t }$ by rotating $q _ { t - 1 }$ counterclockwise about the origin by an angle $\theta ,$ where $\cos ( \theta ) = 1 - \lambda$ . By Lemma B.1, this is a valid sequence because

$$
\begin{array} { r l } & { \| q _ { t } - q _ { t - 1 } \| = \sqrt { \| q _ { t } \| ^ { 2 } - 2 \langle q _ { t } , q _ { t - 1 } \rangle + \| q _ { t - 1 } \| ^ { 2 } } } \\ & { \qquad = \rho \sqrt { 2 ( 1 - \cos ( \theta ) ) } } \\ & { \qquad \leqslant O ( \lambda ) } \end{array}
$$

and, letting $p ^ { * } = 0$ and $r _ { t } = q _ { t - 1 } + ( q _ { t } - q _ { t - 1 } ) / \lambda$ as always,

$$
\begin{array} { l } { \langle q _ { t - 1 } , p ^ { * } - r _ { t } \rangle = - \Big ( \| q _ { t - 1 } \| ^ { 2 } + \frac { \langle q _ { t } , q _ { t - 1 } \rangle - \| q _ { t - 1 } \| ^ { 2 } } { \lambda } \Big ) } \\ { = - \rho ^ { 2 } \Big ( 1 + \frac { \cos ( \theta ) - 1 } { \lambda } \Big ) } \\ { = 0 . } \end{array}
$$

By equations (10) and (11), this sequence has degree-2 calibration error $| \mathrm { C E _ { 2 } } ( q _ { t } ) | = \rho ^ { 2 } = \Omega ( \lambda )$ and expected calibration error $\operatorname { E C E } ( q _ { t } ) = C _ { 2 } \rho = \Omega ( { \sqrt { \lambda } } )$ . Clearly, lower bounds on the norm $\| q _ { t } \|$ are preserved by mixing over a random time index $t \in [ T ]$ , regardless of its distribution, so if we define $p ( x ) = q _ { t } ( x )$ , it follows that $| \mathrm { C E } _ { 2 } ( \pmb { p } ) | = \Omega ( \lambda )$ and $\begin{array} { r } { \dot { \mathrm { E C E } } ( \pmb { p } ) = \Omega ( \sqrt { \lambda } ) } \end{array}$ , as well. Similarly, for any unbiased randomized noise function $N : [ \pm 1 ]  [ \pm 1 ]$ , since $p ^ { * } = 0$ , Jensen’s inequality implies

$$
| \mathrm { C E } _ { 2 } ( N \circ p ) | = \mathbb { E } [ N ( p ( x ) ) ^ { 2 } ] \geqslant \mathbb { E } \big [ \mathbb { E } [ N ( p ( x ) ) | p ( x ) | ^ { 2 } ] = \mathbb { E } [ p ( x ) ^ { 2 } ] = | \mathrm { C E } _ { 2 } ( p ) |
$$

and

$$
\begin{array} { r } { \mathrm { E C E } ( N \circ p ) = \mathbb { E } [ | N ( p ( x ) ) | ] \geqslant \mathbb { E } \big [ \big | \mathbb { E } [ N ( p ( x ) ) | p ( x ) ] \big | \big ] = \mathbb { E } [ | p ( x ) | ] = \mathrm { E C E } ( p ) , } \end{array}
$$

so this noise addition preserves the lower bounds on $| \mathrm { C E _ { 2 } } |$ and ECE, as well.

Example B.3 (Finite Sequence with Large Error). Let $q _ { 1 } , \ldots , q _ { T }$ be a sequence of points on a circle of radius $\rho = { \sqrt { 1 + \varepsilon } }$ centered at $p ^ { * } / 2 = e _ { 1 } = ( 1 , 0 )$ in $\mathbb { R } ^ { 2 }$ . By equation (10), this circle is the set of points $q$ with $\mathrm { C E } _ { 2 } ( q ) = - \varepsilon$ . For points on this circle, $\langle q , p ^ { * } \rangle - \| q \| ^ { 2 } = - \varepsilon$ , so the potential gap condition from Lemma B.1 becomes

$$
\langle e _ { 1 } , q _ { t } - q _ { t - 1 } \rangle = - \lambda \varepsilon + \frac { 1 } { 2 } \| q _ { t } - q _ { t - 1 } \| ^ { 2 } .
$$

A simple continuity argument shows that for any point $q _ { t - 1 }$ in, say, the upper one-third of the circle, there exists a solution $q _ { t }$ to this equation that also lies on the circle at distance $O ( \lambda \varepsilon )$ from $q _ { t - 1 } . ^ { 1 1 }$ Thus, for a suficiently small universal constant $c > 0$ , we may choose $\varepsilon = c \operatorname* { m i n } \{ 1 , 1 / ( \lambda T ) \}$ and remain in the upper one-third of the circle for $T$ time steps. Therefore, after rescaling, this is a valid sequence whose degree-2 calibration error is negative, constant across rounds, and has absolute value $\Omega ( \operatorname* { m i n } \{ 1 , 1 / ( \lambda T ) \} )$ . As in the previous example, this lower bound on the absolute degree-2 calibration error is preserved even if we consider any randomized predictor p defined as a mixture of $q _ { 1 } , \ldots , q _ { T }$ . Similarly, for any unbiased randomized noise function $N : [ \pm 1 ]  [ \pm 1 ]$ , we have

$$
\begin{array} { r l } & { \mathrm { C E } _ { 2 } ( N \circ p ) = \mathbb { E } [ N ( p ( x ) ) ( y - N ( p ( x ) ) ) ] } \\ & { \qquad = \mathbb { E } [ p ( x ) ( y - p ( x ) ) ] - \mathbb { E } [ N ( p ( x ) ) ^ { 2 } - p ( x ) ^ { 2 } ] } \\ & { \qquad = \mathrm { C E } _ { 2 } ( p ) - \mathbb { E } [ \mathrm { V a r } ( N ( p ( x ) ) | p ( x ) ) ] . } \end{array}
$$

Since the degree-2 calibration error was negative without noise, and since variance is always nonnegative, the above calculation shows that this noise addition preserves the lower bound on $| \mathrm { C E _ { 2 } } |$ as well.

Proof of Lemma $4 . 3 .$ The $\Omega ( \lambda )$ lower bound comes from Example B.2, and the $\Omega ( \operatorname* { m i n } \{ 1 , 1 / ( \lambda T ) \} )$ lower bound comes from Example B.3. Choosing the stronger of the two examples gives that Ω(min $\{ 1 , 1 / ( \lambda T ) + \lambda \} = \Omega ( T ^ { - 1 / 2 } )$ □

Proof of Lemma $4 { \cdot } 4 .$ The $\Omega ( { \sqrt { \lambda } } )$ lower bound comes from Example B.2, and the $\Omega ( \operatorname* { m i n } \{ 1 , 1 / ( \lambda T ) \} )$ lower bound comes from Example B.3. Choosing the stronger of the two examples gives that $\Omega ( \operatorname* { m i n } \{ 1 , 1 / ( \lambda T ) + \sqrt { \lambda } \} ) = \Omega ( T ^ { - 1 / 3 } )$ □

Having proved Lemmas 4.3 and 4.4, we now give an example of a valid predictor sequence where the degree-2 calibration error oscillates. This will prove to be a useful gadget for showing that randomization over the iterates $q _ { t }$ is necessary.

Example B.4 (Infinite Sequence with Oscillating Error). Let $q _ { 1 } , q _ { 2 } , . . .$ . lie on a circle of radius $\rho$ centered at $c \in \mathbb { R } ^ { 2 }$ . Let $I \in \mathbb { R } ^ { 2 \times 2 }$ denote the identity matrix and $R _ { \theta } \in \mathbb { R } ^ { 2 \times 2 }$ denote counterclockwise rotation by θ. Define $q _ { t }$ by rotating $q _ { t - 1 }$ counterclockwise about c by an angle $\theta ,$ and define

$$
p ^ { * } = \Big ( 2 I + \frac { R _ { - \theta } - I } { \lambda } \Big ) c .
$$

As long as $\| c \| = O ( 1 )$ and $\rho = O ( 1 )$ and $\theta = O ( \lambda )$ , we have the required boundedness conditions $\| p ^ { * } \| \leqslant O ( 1 )$ and $\| q _ { t } \| \leqslant O ( 1 )$ and $\| q _ { t } - q _ { t - 1 } \| \leqslant O ( \lambda )$ . If we additionally set $\rho = \| c \|$ and $z _ { t } = q _ { t } - c ,$ with $r _ { t }$ defined the same way as always, then we have conditional accuracy in expectation:

$$
\begin{array} { l } { \displaystyle { \langle q _ { t - 1 } , p ^ { * } - r _ { t } \rangle = \Big \langle c + z _ { t - 1 } , p ^ { * } - c - \Big ( I + \frac { R _ { \theta } - I } { \lambda } \Big ) z _ { t - 1 } \Big \rangle } } \\ { \displaystyle { \qquad = \big ( \| c \| ^ { 2 } - \rho ^ { 2 } \big ) \Big ( 1 + \frac { \cos ( \theta ) - 1 } { \lambda } \Big ) } } \\ { \displaystyle { \qquad = 0 . } } \end{array}
$$

Therefore, this is a valid sequence. Moreover, its degree-2 calibration error oscillates:

$$
\begin{array} { l } { \operatorname { C E } _ { 2 } ( q _ { t } ) = \langle q _ { t } , p ^ { * } - q _ { t } \rangle } \\ { \qquad = \| c \| ^ { 2 } - \rho ^ { 2 } + \Bigl \langle q _ { t } , \frac { R _ { - \theta } - I } { \lambda } c \Bigr \rangle . } \end{array}
$$

For example, if $c = ( 1 , 0 )$ and $\rho = 1$ and $q _ { t } = ( 1 + \cos ( \theta t ) , \sin ( \theta t ) )$ for $\theta = \lambda$ then

$$
\mathrm { C E } _ { 2 } ( q _ { t } ) = - \sin ( \lambda t ) \pm O ( \lambda ) ,
$$

which oscillates from +1 to −1 back to +1 over periods of length ${ \cal O } ( 1 / \lambda )$ . In this example, $p ^ { * }$ is close to $( 2 , - 1 )$ for small λ. (The predictors $p ^ { * }$ and $q _ { t }$ are $O ( 1 )$ -bounded but not quite $[ \pm 1 ]$ -valued. Scaling down by a constant factor fixes the issue.)

Proof of Lemma 4.5. For each sign $\sigma \in \{ \pm 1 \}$ , choose $p _ { \sigma } ^ { * }$ and a valid sequence $q _ { \sigma t }$ such that

$$
\left. q _ { \sigma t } , p _ { \sigma } ^ { * } - q _ { \sigma t } \right. = a \cdot \sin \Bigl ( \sigma \frac { \pi } { 4 } + \lambda t \Bigr ) \pm O ( \lambda )
$$

for some constant $a > 0$ . This is possible by Example B.4. Note that at any time t, one of the two predictors $q _ { \sigma t }$ must have Ω(1) degree-2 calibration error for its respective $p _ { \sigma } ^ { * }$ , simply because sin $( \pi / 4 + \lambda t )$ and sin $\scriptstyle 1 \left( - { \pi } / 4 + \lambda t \right)$ are of by a $\pi / 2 = 9 0 ^ { \circ }$ phase shift and are therefore never close to 0 at the same time t. To combine these examples, define a new space $\mathcal { X } ^ { \prime } = \mathcal { X } \times \{ \pm 1 \}$ with

$$
\begin{array} { l } { { \displaystyle p ^ { * } ( x , \sigma ) = \frac { p _ { \sigma } ^ { * } ( x ) + \sigma } { 2 } } } \\ { { \displaystyle q _ { t } ( x , \sigma ) = \frac { q _ { \sigma t } ( x ) + \sigma } { 2 } . } } \end{array}
$$

Intuitively, the new space $\mathcal { X } ^ { \prime }$ has two groups labeled $\sigma = + 1$ and $\sigma = - 1$ , where both $p ^ { * }$ and $q _ { t }$ take on values in the range $[ 0 , + 1 ]$ and $[ - 1 , 0 ]$ , respectively. Define the 1-Lipschitz weight function

$$
w _ { \sigma } ( p ) = \mathrm { p r o j } _ { \left[ \pm \frac { 1 } { 2 } \right] } \Big ( p - \frac { \sigma } { 2 } \Big ) .
$$

Intuitively, the clipping to $[ \pm 1 / 2 ]$ ensures that $w _ { + 1 }$ ignores behavior in the group $\sigma = - 1$ and $w _ { - 1 }$ ignores behavior in the group $\sigma = + 1$ . More formally, if the group index $\sigma \in \{ \pm 1 \}$ is uniform, then

$$
\mathrm { C E } _ { w _ { \sigma } } ( q _ { t } ) = \frac { 1 } { 8 } \langle q _ { \sigma t } , p _ { \sigma } ^ { * } - q _ { \sigma t } \rangle .
$$

Taking the maximum over $\sigma \in \{ \pm 1 \}$ gives

$$
\operatorname* { m a x } _ { \sigma \in \{ \pm 1 \} } | \mathrm { C E } _ { w _ { \sigma } } ( q _ { t } ) | = \Omega ( 1 ) .
$$

## C Deferred Proofs From Section 5

Proof of Theorem $5 . 4 .$ First, we prove the following helper lemma:

Lemma C.1. Let $w : [ \pm 1 ]  [ \pm 1 ]$ be L-Lipschitz, and define $\begin{array} { r } { W ( v ) : = \int _ { 0 } ^ { v } w ( u ) } \end{array}$ du. Then, for every $a , b \in [ \pm 1 ]$

$$
| W ( b ) - W ( a ) - w ( a ) ( b - a ) | \leq \frac { L } { 2 } ( b - a ) ^ { 2 } .\tag{16}
$$

Moreover, for every $y \in \{ \pm 1 \}$ , the function $v \mapsto w ( v ) ( y - v )$ is $( 2 L + 1 ) – L i p s c h i t z$ .

Proof. For the first statement,

$$
W ( b ) - W ( a ) - w ( a ) ( b - a ) = \int _ { a } ^ { b } ( w ( u ) - w ( a ) ) d u .
$$

Since w is L-Lipschitz,

$$
| w ( u ) - w ( a ) | \leq L | u - a | ,
$$

and integrating gives equation (16). For the second statement, for every $u , v \in [ \pm 1 ]$

$$
\begin{array} { r l } & { | w ( v ) ( y - v ) - w ( u ) ( y - u ) | \leq | w ( v ) - w ( u ) | | y - v | + | w ( u ) | | v - u | } \\ & { \qquad \leq ( 2 L + 1 ) | v - u | , } \end{array}
$$

where we used $\left| y - v \right| \leq 2$ and $| w ( u ) | \leq 1$

We can now prove Theorem 5.4. Analogously to Theorem 5.2, we define the calibration error on the t-th iteration:

$$
\alpha _ { t } = \mathbb { E } [ w ( q _ { t } ( \pmb { x } ) ) ( \pmb { y } - q _ { t } ( \pmb { x } ) ) ] .
$$

By Lemma C.1 the function $v \mapsto w ( v ) ( y - v )$ is (2L + 1)-Lipschitz. Since by the geometric update,

$$
| q _ { t + 1 } ( x ) - q _ { t } ( x ) | = \lambda | r _ { t + 1 } ( x ) - q _ { t } ( x ) | \leq 2 \lambda ,
$$

we have

$$
| \alpha _ { t + 1 } - \alpha _ { t } | \leq ( 2 L + 1 ) \cdot 2 \lambda \leq \alpha .\tag{17}
$$

Suppose toward contradiction that $| \alpha _ { t } | > \alpha$ for every $t \in \{ s , \ldots , s + T - 1 \}$ . Equation (17) implies that the $\alpha _ { t } { ' } \mathrm { s }$ cannot change sign. Indeed, moving from a value greater than α to a value smaller than $- \alpha$ would require a change greater than 2α. Therefore, either

$$
\alpha _ { t } > \alpha \qquad \mathrm { f o r ~ e v e r y ~ } t \in \{ s , \ldots , s + T - 1 \} ,\tag{18}
$$

or

$$
\alpha _ { t } < - \alpha \qquad \mathrm { f o r ~ e v e r y ~ } t \in \{ s , \ldots , s + T - 1 \} .\tag{19}
$$

Here we use the following potential function to track the $q _ { t }$ changes, which generalizes the potential function that we used in the proof of Theorem 5.2. Namely, define

$$
\Psi _ { t } : = \mathbb { E } [ W ( q _ { t } ( x ) ) ] , \qquad W ( v ) : = \int _ { 0 } ^ { v } w ( u ) d u .
$$

By Lemma C.1 and the geometric update rule,

$$
W ( q _ { t + 1 } ( x ) ) = W ( q _ { t } ( x ) ) + \lambda w ( q _ { t } ( x ) ) ( r _ { t + 1 } ( x ) - q _ { t } ( x ) ) + e _ { t } ( x ) ,
$$

where $| e _ { t } ( x ) | \leq 2 L \lambda ^ { 2 }$ pointwise.

By the query-response pair property $\mathbb { E } [ { \pmb y } - r _ { t + 1 } ( { \pmb x } ) \ | \ q _ { t } ( { \pmb x } ) ] = 0$ , it follows that for every bounded function w $: [ \pm 1 ]  \mathbb { R }$ 2

$$
\begin{array} { r l } & { \mathbb { E } [ w ( q _ { t } ( \pmb { x } ) ) ( \pmb { y } - r _ { t + 1 } ( \pmb { x } ) ) ] = \mathbb { E } [ \mathbb { E } [ w ( q _ { t } ( \pmb { x } ) ) ( \pmb { y } - r _ { t + 1 } ( \pmb { x } ) ) \mid q _ { t } ( \pmb { x } ) ] ] } \\ & { \quad \quad \quad \quad = \mathbb { E } [ w ( q _ { t } ( \pmb { x } ) ) \mathbb { E } [ \pmb { y } - r _ { t + 1 } ( \pmb { x } ) \mid q _ { t } ( \pmb { x } ) ] ] } \\ & { \quad \quad \quad = 0 . } \end{array}
$$

Hence,

$$
\begin{array} { r l } & { \alpha _ { t } = \mathbb { E } [ w ( q _ { t } ( \pmb { x } ) ) ( \pmb { y } - q _ { t } ( \pmb { x } ) ) ] } \\ & { \quad = \mathbb { E } [ w ( q _ { t } ( \pmb { x } ) ) ( \pmb { y } - r _ { t + 1 } ( \pmb { x } ) ) ] + \mathbb { E } [ w ( q _ { t } ( \pmb { x } ) ) ( r _ { t + 1 } ( \pmb { x } ) - q _ { t } ( \pmb { x } ) ) ] } \\ & { \quad = \mathbb { E } [ w ( q _ { t } ( \pmb { x } ) ) ( r _ { t + 1 } ( \pmb { x } ) - q _ { t } ( \pmb { x } ) ) ] . } \end{array}
$$

Then, taking expectations, and by the definition of the potential function,

$$
\begin{array} { r l } & { \Psi _ { t + 1 } - \Psi _ { t } = \lambda \mathbb { E } [ w ( q _ { t } ( \pmb { x } ) ) ( r _ { t + 1 } ( \pmb { x } ) - q _ { t } ( \pmb { x } ) ) ] + \mathbb { E } [ e _ { t } ( \pmb { x } ) ] } \\ & { \qquad = \lambda \alpha _ { t } + \mathbb { E } [ e _ { t } ( \pmb { x } ) ] . } \end{array}
$$

Moreover, $| \mathbb { E } [ e _ { t } ( { \pmb x } ) ] | \leq \mathbb { E } [ | e _ { t } ( { \pmb x } ) | ] \leq 2 L \lambda ^ { 2 }$ . Therefore,

$$
\lambda \alpha _ { t } - 2 L \lambda ^ { 2 } \leq \Psi _ { t + 1 } - \Psi _ { t } \leq \lambda \alpha _ { t } + 2 L \lambda ^ { 2 } .\tag{20}
$$

As in the case of degree-2 calibration, summing the potential identity over consecutive rounds makes the intermediate potential values cancel:

$$
\sum _ { t = s } ^ { s + T - 2 } ( \Psi _ { t + 1 } - \Psi _ { t } ) = \Psi _ { s + T - 1 } - \Psi _ { s } .
$$

Consequently, rearranging gives

$$
\frac { 1 } { T - 1 } \sum _ { t = s } ^ { s + T - 2 } \alpha _ { t } = \frac { \Psi _ { s + T - 1 } - \Psi _ { s } } { \lambda ( T - 1 ) } - \frac { 1 } { \lambda ( T - 1 ) } \sum _ { t = s } ^ { s + T - 2 } \mathbb { E } [ e _ { t } ( x ) ] .
$$

Since the weight function is bounded in absolute value by one, its integral is 1-Lipschitz. Together with the bounded prediction range and the remainder estimate above, this gives

$$
| \Psi _ { s + T - 1 } - \Psi _ { s } | \leq 2 , \qquad | \mathbb { E } [ e _ { t } ( x ) ] | \leq 2 L \lambda ^ { 2 } .
$$

Taking absolute values therefore yields

$$
\begin{array} { r l r } {  {  \frac { 1 } { T - 1 } \sum _ { t = s } ^ { s + T - 2 } \alpha _ { t }  \le \frac {  \Psi _ { s + T - 1 } - \Psi _ { s }  } { \lambda ( T - 1 ) } + \frac { 1 } { \lambda ( T - 1 ) } \sum _ { t = s } ^ { s + T - 2 }  \mathbb { E } [ e _ { t } ( x ) ]  } } \\ & { } & { \le \frac { 2 } { \lambda ( T - 1 ) } + 2 L \lambda , } \end{array}
$$

which is the bound used in the proof sketch in Section 5.1.1.

By assumption, $\lambda \le \alpha / ( 4 L + 2 )$ , and hence $2 L \lambda ^ { 2 } \leqslant \lambda \alpha / 2$ . Suppose first that equation (18) holds. From equation (20) it follows that

$$
\begin{array} { c } { \displaystyle \Psi _ { t + 1 } - \Psi _ { t } \geq \lambda \alpha _ { t } - 2 L \lambda ^ { 2 } } \\ { \displaystyle > \lambda \alpha - 2 L \lambda ^ { 2 } } \\ { \displaystyle \geq \frac { \lambda \alpha } { 2 } . } \end{array}
$$

Hence, summing over $t = s , \ldots , s + T - 2$ gives $\Psi _ { s + T - 1 } - \Psi _ { s } > ( T - 1 ) ( \lambda \alpha / 2 ) > 2$ . This is impossible: since $| w | \leq 1$ , the function W is 1-Lipschitz on [±1], and so for every $u , v \in [ \pm 1 ]$ $| W ( v ) - W ( u ) | \leq | u - v | \leq 2$ . Hence,

$$
\begin{array} { r l } & { | \Psi _ { s + T - 1 } - \Psi _ { s } | = | \mathbb { E } \left[ W ( q _ { s + T - 1 } ( \pmb { x } ) ) - W ( q _ { s } ( \pmb { x } ) ) \right] | } \\ & { \qquad \le \mathbb { E } \left[ | W ( q _ { s + T - 1 } ( \pmb { x } ) ) - W ( q _ { s } ( \pmb { x } ) ) | \right] } \\ & { \qquad \le 2 , } \end{array}
$$

and hence it cannot be that $\Psi _ { s + T - 1 } - \Psi _ { s } > 2 .$

Symmetrically, if we instead suppose that equation (19) holds, from equation (20) we get that

$$
\Psi _ { t + 1 } - \Psi _ { t } \leq - \frac { \lambda \alpha } { 2 } .
$$

Summing over $t = s , \ldots , s + T - 2$ gives $\Psi _ { s + T - 1 } - \Psi _ { s } < - ( T - 1 ) \frac { \lambda \alpha } 2 < - 2$ . This again contradicts the fact that $| \Psi _ { s + T - 1 } - \Psi _ { s } | \leq 2$ . Hence the entire block cannot satisfy $| \alpha _ { t } | > \alpha$

Therefore, there exists some $t \in \{ s , \ldots , s + T - 1 \}$ such that

$$
| \mathbb { E } [ w ( q _ { t } ( { \pmb x } ) ) ( { \pmb y } - q _ { t } ( { \pmb x } ) ) ] | \le \alpha ,
$$

as we wanted to show.

## D Constructing a Partition Function

We now argue that partition functions (Definition 5.5) exist and can be constructed eficiently from unlabeled samples from $\mathcal { D } _ { \mathcal { X } }$ under the following mild conditions.

Definition D.1. We say that a distribution $\mathcal { D } _ { \mathcal { X } }$ is γ-divisible if:

• (No Heavy Atoms) For all $x \in \mathcal { X } , \operatorname* { P r } _ { \pmb { x } } [ \pmb { x } = \pmb { x } ] \leqslant \gamma$

• (Total Order) There is a total order ≺ on X admitting an eficient comparison oracle.<sup>12</sup>

Essentially, the only non-trivial requirement we impose on the distribution $\mathcal { D } _ { \mathcal { X } }$ is that there are no points with probability mass larger than γ. The existence of a total order is, for example, always true for distributions over $\mathbb { R } ^ { d }$ that admit a density or a probability mass function. Under these conditions, we show that a γ-partition function can be constructed as follows.

Lemma D.2. Suppose that $\mathcal { D } _ { \mathcal { X } }$ is γ-divisible. Then, there is an algorithm that, upon receiving $m = O ( ( 1 / \gamma ) \log ( 1 / \gamma \delta ) )$ i.i.d. samples from $\mathcal { D } _ { \mathcal { X } }$ , outputs with probability at least $1 - \delta \ a \ \gamma \mathbf { - } p a r t i t i o n$ function. The algorithm runs in time $O ( m t \log ( m ) )$ , where t is the time required for a single evaluation of the comparison oracle on X .

Proof. Let S be the set of m i.i.d. samples from $\mathcal { D } _ { \mathcal { X } }$ . Let R be the set of singletons and intervals with respect to the order ≺, i.e., sets of the form:

$$
\{ x \in \mathcal { X } : x _ { - } \prec x \prec x _ { + } \} \quad \mathrm { o r } \quad \{ x _ { 0 } \} , \quad \mathrm { ~ w h e r e } \quad x _ { - } , x _ { + } \in \mathcal { X } \cup \{ - \infty , \infty \} , \quad x _ { 0 } \in \mathcal { X }
$$

Observe that the VC dimension of R is 2. Therefore, due to the standard VC γ-net theorem (see, e.g., [KZ20] and references therein), we have that S is a γ-net for $( \mathcal { X } , \mathcal { R } )$ with probability at least $1 - \delta$

Let $x _ { 1 } \prec x _ { 2 } \prec \dots \prec x _ { m }$ be a sorting of the elements of S, which can be computed in time $O ( m t \log m )$ . Without loss of generality, we assume that there are no duplicates, since we can otherwise remove them. We define f as follows:

$$
f ( x ) = { \left\{ \begin{array} { l l } { 1 , { \mathrm { ~ i f ~ } } x \prec x _ { 1 } } \\ { i , { \mathrm { ~ i f ~ } } x _ { i - 1 } \prec x \prec x _ { i } } \\ { m + 1 , { \mathrm { ~ i f ~ } } x _ { m } \prec x } \\ { m + i + 1 , { \mathrm { ~ i f ~ } } x = x _ { i } } \end{array} \right. }
$$

Observe that $f$ is measurable and total. Moreover, for any $i \in [ m ]$ , f takes the value $m + i + 1$ with probability at most γ, because $\mathcal { D } _ { \mathcal { X } }$ does not have heavy atoms.

To complete the argument, note that S does not contain any points in the intervals of the form $( - \infty , x _ { 1 } ) , ( x _ { i - 1 } , x _ { i } ) , ( x _ { m } , \infty )$ . Since $S$ is $\mathrm { a \ { \gamma - n e t } \ ( w . h . p . ) }$ , this implies that the mass of $\mathcal { D } _ { \mathcal { X } }$ on each of these intervals is at most γ. □

Example D.3. Any distribution over $\mathbb { R } ^ { d }$ that admits a density is γ-divisible for any $\gamma \in ( 0 , 1 )$ We may use the lexicographic order, whose comparisons can be performed in time $O ( d )$

## E Implementation using Statistical Queries

In this section, we demonstrate that all of our main results in Theorems 1.3 and 1.4 continue to hold in the statistical query framework of [Kea98]. The statistical query framework is a restricted computation model for statistical problems. It captures a wide range of algorithms like moment methods, convex relaxations, polynomial regression, and gradient descent [FGR<sup>+</sup>17, FGV17]. Concretely, it restricts the access of the algorithm to the underlying distribution D as follows.

Definition E.1 (Statistical Query Algorithm, [Kea98]). For a labeled distribution D over $\mathcal { X } \times \left\{ \pm 1 \right\}$ and tolerance $\tau > 0 ,$ , we say that an algorithm is a statistical query algorithm with tolerance τ if it has access to $\mathcal { D }$ only through adaptive statistical queries $\varphi : \mathcal { X } \times \{ \pm 1 \} \to [ - 1 , 1 ]$ of tolerance τ. Concretely, in response to $\varphi ,$ the algorithm receives a value v satisfying

$$
\begin{array} { r } { \left| v - \mathbb { E } [ \varphi ( \pmb { x } , \pmb { y } ) ] \right| \leqslant \tau . } \end{array}
$$

In order to show that our reductions are SQ-preserving, we observe that the only part where we access the distribution D is when we invoke the conditional multiaccuracy oracle of Theorem 1.2. It therefore sufices to prove the following lemma.

Lemma E.2. Suppose that we have a $( \mathcal { C } , \varepsilon )$ -multiaccuracy oracle that can be implemented with τ -tolerant statistical queries. Then, we have an algorithm for conditional multiaccuracy with the specifications of Theorem 1.2, that can be implemented with τ-tolerant statistical queries.

Proof. Suppose that we have access to statistical queries of tolerance τ under a labeled distribution D over pairs $( { \pmb x } , { \pmb y } )$ . The conditional multiaccuracy oracle of Theorem 1.2 works by calling Algorithm 1 a number of times using the same distribution $\mathcal { D } ,$ but diferent label reweightings m. For each reweighting, the multiaccuracy oracle is called on distributions of the form (x, z), where:

$$
\begin{array} { r } { \mathbb { E } [ z | x ] = m ( x ) \cdot \mathbb { E } [ { \pmb { y } } | { \pmb { x } } ] + ( 1 - m ( x ) ) \cdot { \pmb { p } } ( x ) , } \end{array}
$$

for some known $p : \mathcal { X }  [ - 1 , 1 ]$

Therefore, to implement a τ -tolerant statistical query $\varphi$ requested by the multiaccuracy oracle, we observe:

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \varphi ( { \pmb x } , z ) ] = \mathbb { E } \Big [ \frac { 1 + z } { 2 } \cdot \varphi ( { \pmb x } , 1 ) + \frac { 1 - z } { 2 } \cdot \varphi ( { \pmb x } , - 1 ) \Big ] } \\ { \displaystyle \qquad = \frac { 1 } { 2 } \mathbb { E } \Big [ \varphi ( { \pmb x } , 1 ) + \varphi ( { \pmb x } , - 1 ) + \Big ( m ( { \pmb x } ) \cdot y + ( 1 - m ( { \pmb x } ) ) \cdot p ( { \pmb x } ) \Big ) \cdot \Big ( \varphi ( { \pmb x } , 1 ) - \varphi ( { \pmb x } , - 1 ) \Big ) \Big ] } \end{array}
$$

Note that the quantity inside the expectation is a function of $( { \pmb x } , { \pmb y } )$ , and is pointwise bounded by 2. Therefore, if we define $\varphi ^ { \prime }$ as follows:

$$
\varphi ^ { \prime } ( x , y ) = { \frac { 1 } { 2 } } \cdot { \Big ( } \varphi ( x , 1 ) + \varphi ( x , - 1 ) + { \Big ( } m ( x ) \cdot y + ( 1 - m ( x ) ) \cdot p ( x ) { \Big ) } \cdot { \Big ( } \varphi ( x , 1 ) - \varphi ( x , - 1 ) { \Big ) } { \Big ) } \ ,
$$

then we have $| \varphi ^ { \prime } ( { \pmb x } , { \pmb y } ) | \leqslant 1$ , and we can obtain a τ -tolerant query for $\varphi ^ { \prime }$ using the available oracle. Let $v ^ { \prime } \in [ - 1 , 1 ]$ be the response of the oracle. Then:

$$
\Big | v ^ { \prime } - \mathbb { E } [ \varphi ( { \pmb x } , { \pmb z } ) ] \Big | = \Big | v ^ { \prime } - \mathbb { E } [ \varphi ^ { \prime } ( { \pmb x } , { \pmb y } ) ] \Big | \leqslant \tau
$$

Therefore, we can implement a statistical query for $( x , z )$ using a statistical query for $( { \pmb x } , { \pmb y } )$ with the same tolerance. □

As an immediate consequence of Lemma E.2, we obtain that our main results in Theorems 1.3 and 1.4 are SQ-preserving.