# Agnostic Marginal-Nonuniform Learning

Tanmay Devale Steve Hanneke <sup>†</sup> Amirreza Shaeiri <sup>‡</sup>

November 10, 2026

## Abstract

We study the problem of binary classification in the marginal-nonuniform learning framework in the agnostic setting. In this model, the constants in the learning rate may depend on the marginal distribution of the unlabeled data, but the guarantee must hold uniformly over all conditional distributions of the labels. We also study fine-grained rates.

We consider two criteria. (1) The standard agnostic setting, where the learner competes with the best concept in the class. In this setting, an immediate consequence of standard agnostic PAC learning results gives us the optimal rate of $\scriptstyle { \frac { 1 } { \sqrt { n } } }$ for every class H with finite $\mathrm { V C } ( \mathscr { H } )$ and shows that arbitrarily slow rates are necessary otherwise. This also gives us the optimal fine-grained rate of $\sqrt { \frac { \mathrm { V C } ( \mathcal { H } ) } { n } }$ . Therefore, the dependence on the marginal distribution does not help, which is in contrast to the realizable setting. (2) In the relaxed agnostic setting, where the learner competes with (1 + λ) times the error of the best concept for a fixed $\lambda \in ( 0 , 1 ]$ , the picture from the realizable setting reappears. We obtain a trichotomy of the achievable optimal rates: exponential rates of the form $e ^ { - n }$ arise precisely when the class is finite; linear rates of the form $\textstyle { \frac { 1 } { n } }$ arise when the class is infinite but VC(H) is finite; and arbitrarily slow rates occur when $\mathrm { V C } ( \mathcal { H } ) \stackrel { \cdot \cdot } { = } \infty$ . We also obtain the optimal fine-grained rate of $\frac { \mathrm { u V C E } ( \mathcal { H } ) } { \lambda n }$ , where $\operatorname { u V C E } ( { \mathcal { H } } )$ is a recently introduced combinatorial complexity dimension called the uncentered VC-eluder dimension. <sup>1</sup>

## 1 Introduction

A central question in statistical learning theory is how many labeled examples are needed to predict well on unseen data. The classical PAC model [Vapnik and Chervonenkis, 1971, Valiant, 1984] answers the uniform case: one rate has to hold for every marginal distribution over the instance space and every target concept in the class. This gives a clean worst-case theory, in which the VC dimension characterizes learnability, but it cannot tell an easy data distribution from a hard one. Marginal-nonuniform learning [Ben-David et al., 1995, Hanneke et al., 2025b] relaxes exactly this point. The constants in the rate may depend on the marginal distribution $\mathcal { D } _ { \mathcal { X } }$ of the unlabeled data, but once $\mathcal { D } _ { \mathcal { X } }$ is fixed, the guarantee must hold uniformly over everything else. It sits between uniform PAC learning and universal learning [Bousquet et al., 2021], where the constants may depend on the whole data distribution.

Marginal-nonuniform learning has a classical counterpart in concept-nonuniform learning, introduced by Benedek and Itai [1994] and usually called simply nonuniform learning. In this framework, the roles are exchanged: the constants in the rate may depend on the target concept, but the guarantee must hold uniformly over all marginal distributions. This model makes a much larger family of classes learnable, namely all countable unions of VC classes, and it underlies structural risk minimization, Occam’s razor and the minimum description length principle [Vapnik, 1998, Blumer et al., 1987, Rissanen,

1978, Lugosi and Zeger, 1996]. It distinguishes concepts that are easy to learn from concepts that are hard to learn, whereas marginal-nonuniform learning distinguishes easy marginals from hard ones. Together with uniform and universal learning, these are the four natural notions of learnability obtained by letting the rate depend on the marginal, on the target, on both, or on neither [Ben-David et al., 1995]. We recall them, in the form needed for this paper, in Section 1.1.

In the realizable case, the possible marginal-nonuniform rates are by now well understood. Hanneke et al. [2025b] proved a trichotomy for binary classification: every class is learnable at an exponential rate, at a linear rate, or only at arbitrarily slow rates, and in the linear case the optimal rate is $d / n$ where d is the VC-eluder dimension of Hanneke and Xu [2024]. This dimension can be much smaller than the VC dimension. The work of Brukhim et al. [2026] extended the characterization to multiclass classification, using the uncentered DS-eluder dimension. All of these results assume that some concept in the class labels the data perfectly.

Agnostic marginal-nonuniform learning. Consider an email provider that stores a large archive of messages. The prediction task is chosen only later: filtering spam, detecting phishing, or flagging messages as urgent. The messages, and hence the marginal $\mathcal { D } _ { \mathcal { X } } .$ , are fixed before the task is known, which is the situation marginal-nonuniform guarantees are designed for. But the labels will come from users, who click carelessly and who disagree with each other: the same newsletter is spam to one user and wanted by another. No concept in the class will label the data perfectly. What the provider can hope for is a filter that is nearly as accurate as the best concept in the class, whatever the task and the behavior of the users turn out to be.

This is the step from realizable to agnostic learning [Haussler, 1992, Kearns et al., 1994]. It is the standard second step in PAC learning: one first studies the realizable case, where the optimal rate is $\mathrm { V C } ( \mathscr { H } ) / n$ , and then drops the assumption on the labels, where the optimal rate becomes $\sqrt { \mathrm { V C } ( \mathcal { H } ) / n }$ We take the same step in the marginal-nonuniform setting. The learner competes with $\mathrm { i n f } _ { h \in \mathcal { H } } \mathrm { e r } _ { \mathcal { D } } ( h )$ the constants may depend on the marginal $\mathcal { D } _ { \mathcal { X } }$ , and the guarantee must hold for every distribution D over $\mathcal { X } \times \mathcal { V }$ with that marginal. In the example, the archive fixes $\mathcal { D } _ { \mathcal { X } }$ , while the task and the behavior of the annotators, i.e., the conditional distribution of the label given the message, are left completely free.

If $\mathrm { V C } ( \mathscr { H } ) < \infty .$ , the classical agnostic PAC upper bound applies directly: it holds for every distribution D over $\mathcal { X } \times \mathcal { V }$ , with constants that do not depend on ${ \mathcal { D } } ,$ so in particular it holds for every distribution with a given marginal $\mathcal { D } _ { \mathcal { X } }$ . The classical agnostic PAC lower bound also carries over. Its hard distributions all have the same marginal, namely the uniform distribution on a set of $\mathrm { V C } ( \mathscr { H } )$ points shattered by $\mathcal { H } ,$ , and only the conditional distribution of the labels depends on the sample size and on the learner. It is therefore a lower bound in the marginal-nonuniform model as well. Combining the two, H is agnostically marginal-nonuniformly learnable at exact rate $\scriptstyle { \frac { 1 } { \sqrt { n } } }$ whenever $\mathrm { V C } ( \mathcal { H } ) < \infty$ . It also provides a fine-grained rate of $\sqrt { \frac { \mathrm { V C } ( \mathcal { H } ) } { n } }$ . Therefore, we could say that the dependence on the marginal does not help.

Relaxed agnostic marginal-nonuniform learning. The $\scriptstyle { \frac { 1 } { \sqrt { n } } }$ rate is the price of competing with the best concept exactly. In many applications this is more than one needs. Suppose that in the example above the best rule in the class disagrees with the user on 1% of the messages. A guarantee of the form “error at most 1.1%, plus a term of order $1 / n ^ { \prime \prime }$ is then far more useful than “error at most 1%, plus a term of order $1 / \sqrt { n } ^ { \prime \prime }$ . When the best concept is already good, a small multiplicative loss costs almost nothing, while the additive $1 / \sqrt { n }$ term dominates the bound.

This motivates the relaxed agnostic criterion: for a fixed $\lambda \in ( 0 , 1 ]$ , the learner competes with $( 1 +$ $\lambda ) \operatorname* { i n f } _ { h \in { \mathcal { H } } } \operatorname { e r } _ { { \mathcal { D } } } ( h )$ . Guarantees of this multiplicative form are common. The weighted majority algorithm competes with a constant factor times the loss of the best expert [Littlestone and Warmuth, 1994]; efficient algorithms for learning halfspaces with adversarial label noise under suitable distributional assumptions typically guarantee error $( 1 + \lambda ) { \mathrm { O P T } } + \varepsilon { \mathrm { o r } } C \cdot { \mathrm { O P T } } + \varepsilon [ { \mathrm { D a n i e l y } } ]$ , 2015, Awasthi et al., 2017]; and bounds that improve when the best error is small are known as optimistic or first-order bounds [Srebro et al., 2010]. The relaxed criterion also interpolates between the two classical settings: when the data are realizable it is the realizable criterion, and as $\lambda \to 0$ it becomes the agnostic one.

In this case we show that under the relaxed criterion the realizable picture returns, both for the rates and for the fine-grained rates. The achievable rates form the same trichotomy as in the realizable case [Hanneke et al., 2025b]: exponential for finite classes, linear for infinite classes of finite VC dimension, and arbitrarily slow otherwise. Note that even a finite class only has rate $\scriptstyle { \frac { 1 } { \sqrt { n } } }$ under the standard criterion, so the multiplicative slack already matters there. The fine-grained rate is of order $\frac { d } { \lambda n }$ , where d is the uncentered VC-eluder dimension of the class, and the dependence on the marginal is confined to an exponentially small term. So the leading term is again governed by the uncentered VC-eluder dimension rather than by the VC dimension. The $\sqrt { \mathrm { V C } ( \mathcal { H } ) / n }$ barrier of the agnostic case is therefore caused entirely by insisting on the constant 1 in front of the best error.

## 1.1 Main results

We first place our setting among the four learning paradigms mentioned above. A distribution D over $\mathcal { X } \times \mathcal { V }$ consists of its marginal $\mathcal { D } _ { \mathcal { X } }$ and its conditional distribution of the label given the instance, which we denote $\mathcal { D } _ { \mathcal { V } | \mathcal { X } }$ . In the realizable case, $\mathcal { D } _ { \mathcal { V } | \mathcal { X } }$ is given by a target concept; in general it plays the role that the target concept plays there. For $\lambda \in [ \dot { 0 } , 1 ]$ and a learner ${ \mathcal { A } } ,$ write

$$
\mathcal { E } _ { \lambda } ( \boldsymbol { \mathcal { A } } , \mathcal { D } , n ) = \mathbb { E } _ { S \sim \mathcal { D } ^ { n } } [ \mathrm { e r } _ { \mathcal { D } } ( \boldsymbol { \mathcal { A } } _ { S } ) ] - ( 1 + \lambda ) \operatorname* { i n f } _ { h \in \mathcal { H } } \mathrm { e r } _ { \mathcal { D } } ( h )
$$

for the excess error, where $\lambda = 0$ is the standard agnostic criterion and $\lambda > 0$ the relaxed one. Adapting Bousquet et al. [2021] and Hanneke et al. [2025b], the four paradigms differ only in the order of the quantifiers. Throughout, we extend every rate function $R : \mathbb { N }  [ 0 , 1 ]$ to positive real arguments by setting $R ( t ) = R ( \lceil t \rceil )$ for all $t > 0$

Definition 1.1. Let $R : \mathbb { N }  [ 0 , 1 ]$ be a ratefunction with lim $\operatorname { 1 } _ { n \to \infty } R ( n ) = 0$ and let $\lambda \in [ 0 , 1 ]$

• H is λ-agnostically uniformly learnable at rate R if

$\exists { \mathcal { A } } \ \exists C , c > 0$ for all $\mathcal { D } _ { \mathcal { V } | \mathcal { X } }$ for all $\mathcal { D } _ { \mathcal { X } }$ for all $n \in \mathbb { N } \ \mathcal { E } _ { \lambda } ( A , \mathcal { D } , n ) \leq C R { ( c n ) }$

• H is λ-agnostically universally learnable at rate R if

∃ A for all $\mathcal { D } _ { \mathcal { V } | \mathcal { X } }$ for all $\mathcal { D } _ { \mathcal { X } } \exists C , c > 0$ for all $n \in \mathbb { N } \ \mathcal { E } _ { \lambda } ( A , \mathcal { D } , n ) \leq C R { ( c n ) }$

• H is λ-agnostically concept-nonuniformly learnable at rate R if

∃ A for all $\mathcal { D } _ { \mathcal { V } | \mathcal { X } } \perp C , c > 0$ for all $\mathcal { D } _ { \mathcal { X } }$ for all $n \in \mathbb { N } \ \mathcal { E } _ { \lambda } ( A , \mathcal { D } , n ) \leq C R { ( c n ) }$

• H is λ-agnostically marginal-nonuniformly learnable at rate R if

∃ A for all $\mathcal { D } _ { \mathcal { X } } \exists C , c > 0$ for all $\mathcal { D } _ { \mathcal { V } | \mathcal { X } }$ for all $n \in \mathbb { N } \ \mathcal { E } _ { \lambda } ( A , \mathcal { D } , n ) \leq C R { ( c n ) }$

Here D is the distribution over $\mathcal { X } \times \mathcal { V }$ with marginal $\mathcal { D } _ { \mathcal { X } }$ and conditional label distribution $\mathcal { D } _ { \mathcal { V } | \mathcal { X } }$

Uniform learning is the classical agnostic PAC model: the constants depend on nothing. In universal learning they may depend on the whole distribution. The two nonuniform models sit in between and are incomparable: in the concept-nonuniform model the constants may depend on how the labels are generated but not on the marginal, and in the marginal-nonuniform model it is the other way round. This paper is about the last paradigm. It is the natural one when the unlabeled data are fixed in advance and the labeling task is not. We state our results at two levels of detail. The first is the rate in the sense of Definition 1.1, which describes the shape of the learning curve. Throughout we assume $| \mathcal { H } | \geq 3$ , which only excludes trivial classes [Hanneke et al., 2025b]. Our first result is regarding the rates for standard agnostic and relaxed agnostic settings.

Theorem 1.2. [Rates, informal] Let H be a concept class with $| \mathcal { H } | \geq 3$

• Standard agnostic learning.

1. H is 0-agnostically marginal-nonuniformly learnable at exact rate $\scriptstyle { \frac { 1 } { \sqrt { n } } }$ if and only $i f \mathrm { V C } ( \mathcal { H } ) <$ ∞.

2. H requires arbitrarily slow rates to be 0-agnostically marginal-nonuniformly learnable if and only $i f \mathrm { V C } ( \mathcal { H } ) = \infty$

• Relaxed agnostic learning. For every fixed $0 < \lambda \leq 1$

1. H is λ-agnostically marginal-nonuniformly learnable at exact rate $e ^ { - n }$ if and only $i f \mathcal { H }$ is finite.

2. H is λ-agnostically marginal-nonuniformly learnable at exact rate $\frac { 1 } { n }$ if and only if H is infinite and $\mathrm { V C } ( \mathcal { H } ) < \infty$

3. H requires arbitrarily slow rates to be λ-agnostically marginal-nonuniformly learnable if and only $i f \mathrm { V C } ( \mathcal { H } ) = \infty .$

For the standard criterion, the rates are exactly those of uniform agnostic PAC learning. For the relaxed criterion, they are exactly those of realizable marginal-nonuniform learning [Hanneke et al., 2025b].

Rates alone say nothing about the constants. A bound $C R ( c n )$ in which C and c may depend on the marginal does not tell us how the class and the marginal together determine the error. This is addressed by the fine-grained rates of Bousquet et al. [2023], which separate the two dependences into different additive terms: one asks for a bound of the form

$$
\mathcal { E } _ { \lambda } ( A , \mathcal { D } , n ) \leq K R ( n ) + \delta ( n ) ,
$$

where K is a universal constant, the leading term $R ( n )$ depends only on the class (and on λ), and all dependence on the marginal is confined to a lower-order term $\delta ( n ) = o ( R ( n ) )$ . A matching lower bound asks that a universal constant multiple of $R ( n )$ cannot be avoided. The marginal then determines only how long one has to wait before the class-dependent rate takes over, and not the rate itself. The precise definitions are in Section 3.

Theorem 1.3 (Fine-grained rates, informal). 1. $I f \mathrm { V C } ( \mathscr { H } ) < \infty$ , then H is 0-agnostically marginalnonuniformly learnable at exact fine-grained rate $\sqrt { \frac { \mathrm { V C } ( \mathcal { H } ) } { n } }$

2. $I f 1 \leq \mathrm { u V C E } ( \mathcal { H } ) < \infty ,$ , then for every $0 < \lambda \leq 1$ , H is λ-agnostically marginal-nonuniformly learnable at exact fine-grained rate $\frac { \mathrm { u V C E } ( \mathcal { H } ) } { \lambda n }$

## 2 Related Works

Agnostic universal and nonuniform learning. The work of Hanneke and Moran [2026] extends the universal learning theory of Bousquet et al. [2021] to the agnostic setting, showing that the optimal universal rate is $e ^ { - \bar { n } } , e ^ { - o ( \bar { n } ) } , o ( n ^ { - 1 / 2 } )$ , or arbitrarily slow, and Hanneke and Xu [2025] characterize the agnostic universal rates of ERMs. Since the constants there may depend on the entire data distribution, every finite class is learnable at rate $e ^ { - n }$ even under the standard criterion, whereas in the marginalnonuniform framework two concepts already force rate $1 / \sqrt { n }$ unless the criterion is relaxed. In classical comparator-dependent nonuniform learning <sup>2</sup>, the agnostic case is handled by structural risk minimization [Vapnik, 1998, Lugosi and Zeger, 1996, Shalev-Shwartz and Ben-David, 2014], and Hanneke et al.

[2025c] characterize nonuniform multiclass learnability in both the realizable and the agnostic settings, under both full-information and bandit feedback. To our knowledge, marginal-nonuniform learning has so far been studied only in the realizable case [Hanneke et al., 2025b, Brukhim et al., 2026].

Multiplicative slack in the uniform setting. In the uniform learning framework, it is well known that a multiplicative slack yields fast rates. Relative deviation and optimistic bounds [Vapnik, 1998, Srebro et al., 2010] show that the error of ERM on n examples, over a class of VC dimension d, is at most $( 1 + \lambda )$ times the best error in the class plus $O \big ( d \log ( n ) / ( \lambda n ) \big )$ , and the first-order learner of Mathiasen et al. [2026] (see also Hanneke et al. [2024]) removes the logarithmic factor. Additionally, the work of Asilis et al. [2025] gives a learner whose error is at most c times the best error in the class, for a constant $c \leq 2 . 1$ , plus the optimal additive term. In statistics, bounds of this form, with a leading constant larger than one, are called nonexact oracle inequalities, and they can give fast rates where exact ones only give slow rates [Lecué and Mendelson, 2012]. Our results for the relaxed criterion are the marginalnonuniform analogue, with the VC dimension replaced by the uncentered VC-eluder dimension, which can be much smaller, and with exponential rates for finite classes; neither follows from the uniform bounds.

Learning with a known marginal. Marginal-nonuniform learning is related to learning under a fixed marginal that is known to the learner [Benedek and Itai, 1991]. This is also the idealized limit of semisupervised learning, where the value of unlabeled data has been studied extensively [Ben-David et al., 2008, Darnstädt et al., 2013, Göpfert et al., 2019]. Our fine-grained lower bounds use a single marginal, so they hold even for learners that know $\mathcal { D } _ { \mathcal { X } }$ . In particular, under the standard criterion, knowing the marginal does not improve the fine-grained rate $\sqrt { \mathrm { V C } ( \mathcal { H } ) / n }$

## 3 Preliminaries

Let X denote a non-empty set which we call the instance space, let $\mathcal { V } = \{ 0 , 1 \}$ denote our label space, and let $\mathcal { H } \subseteq \mathcal { V } ^ { \mathcal { X } }$ be our concept class. The measure-theoretical technicalities have been addressed in appendix B. A data distribution D is a distribution over $\mathcal { X } \times \mathcal { V }$ . For every data distributions D we denote its marginal distribution over $\mathcal { X }$ by $\mathcal { D } _ { \mathcal { X } }$ . A learning algorithm A maps a dataset $S \in \textstyle \bigcup _ { n = 1 } ^ { \infty } ( { \mathcal { X } } \times { \mathcal { Y } } ) ^ { n }$ to a classifier $\mathcal { A } _ { S } : \mathcal { X }  \mathcal { Y }$ . We allow learning algorithms to use internal randomness, independent of the data; then $A _ { S }$ is a random classifier, and $\mathrm { e r } _ { \mathcal { D } } ( \mathcal { A } _ { S } )$ and all expectations below are also taken over this randomness.

The error rate of a classifier $h : \mathcal { X }  \mathcal { Y }$ with respect to a data distribution D is $\mathrm { e r } _ { \mathscr { D } } ( h ) ~ =$ $\mathbb { P } _ { ( x , y ) \sim \mathcal { D } } [ h ( x ) \ \neq \ y ]$ Similarly, given a dataset $S ~ \in ~ ( { \mathcal { X } } \times { \mathcal { Y } } ) ^ { n }$ the error rate $A _ { S }$ is defined as $\mathrm { e r } _ { \mathcal { D } } ( \mathcal { A } _ { S } ) = \mathbb { P } _ { ( x , y ) \sim \mathcal { D } } [ \mathcal { A } _ { S } ( x ) \neq y ]$ and for a concept $h \in \mathcal { H } , \mathrm { e r } _ { \mathcal { D } } ( h ) = \mathbb { P } [ h ( x ) \neq y ]$ . The works of Hanneke et al. [2025b], Brukhim et al. [2026] focused on the realizable case: namely those distributions satisfying in $\dot { \bar { \boldsymbol { h } } } \in \mathcal { H } \operatorname { e r } _ { \mathcal { D } } ( \boldsymbol { h } ) = 0$ and study the rate of convergence of the expected error rate $\mathbb { E } [ \mathrm { e r } _ { \mathcal { D } } ( \mathcal { A } _ { S } ) ]$ holding uniformly over all label conditionals with the given marginal $\mathcal { D } _ { \mathcal { X } }$ . In contrast, in the context of λ-agnostic learning for a fixed $\lambda \in [ 0 , 1 ]$ our interest is in characterizing the optimal rates of convergence of the expected excess error rate $\mathbb { E } [ \operatorname { e r } _ { \mathcal { D } } ( \boldsymbol { \mathcal { A } } _ { S } ) ] - ( 1 + \lambda ) \operatorname { i n f } _ { h \in \mathcal { H } } \operatorname { e r } _ { \mathcal { D } } ( h )$ holding uniformly over all concept in our concept class for the distribution D (without realizability restrictions). The case $\lambda = 0$ recovers the standard agnostic setting. For $\lambda > 0$ , we call this the relaxed agnostic setting. We extend the definitions of achievable rates from the works of Hanneke et al. [2025b], Brukhim et al. [2026] to apply in the context of λ-agnostic learning. Importantly, in the following definitions, note that the order of the quantifiers allows for the constants $C , c$ to be $\mathcal { D } _ { \mathcal { X } }$ -dependent (which is what distinguishes marginal-nonuniform rates from uniform rates).

Now, we present the definition of agnostic marginal-nonuniform learnability

Definition 3.1. Let $\mathcal { H } \subseteq \mathcal { V } ^ { \mathcal { X } }$ be a concept class and let $\mathcal { R } : \mathbb { N }  [ 0 , 1 ]$ be a rate function such that $\begin{array} { r } { \operatorname* { l i m } _ { n  \infty } \mathcal { R } ( n ) = 0 } \end{array}$ . We say

1. H is λ-agnostically marginal-nonuniformly learnable at rate R if there is a learning algorithm A such that for all marginal distributions $\mathcal { D } _ { \mathcal { X } }$ there exists $C , c > 0$ such that for all distributions D with marginal distribution $\begin{array} { r } { \mathcal { D } _ { \mathcal { X } , \mathbb { E } _ { S \sim \mathcal { D } ^ { n } } } [ \mathrm { e r } _ { \mathcal { D } } ( \mathcal { A } _ { S } ) ] - ( 1 + \lambda ) \operatorname* { i n f } _ { h \in \mathcal { H } } \mathrm { e r } _ { \mathcal { D } } ( h ) \le C \mathcal { R } ( c n ) } \end{array}$ for all $n \in \mathbb { N } .$

2. H is not λ-agnostically marginal-nonuniformly learnable at rate faster than ${ \mathcal { R } } ,$ if for every learning algorithm A, there exists a marginal distribution $\mathcal { D } _ { \mathcal { X } }$ such that there exists $C , c > 0$ there exists infinitely many $n \in \mathbb { N } f o r$ each ofwhich there exists a distribution $\mathcal { D } _ { n }$ with marginal $\mathcal { D } _ { \mathcal { X } }$ such that $\begin{array} { r } { \mathbb { E } _ { S \sim \mathcal { D } _ { n } ^ { n } } [ \mathrm { e r } _ { \mathcal { D } _ { n } } ( \mathcal { A } _ { S } ) ] - ( 1 + \lambda ) \operatorname* { i n f } _ { h \in \mathcal { H } } \mathrm { e r } _ { \mathcal { D } _ { n } } ( h ) \ge C \mathcal { R } ( c n ) } \end{array}$

3. H is λ-agnostically marginal-nonuniformly learnable at exact rate R if H is λ-agnostically marginalnonuniformly learnable at R and notfaster than rate R.

4. H requires at least arbitrarily slow rates to be λ-agnostically marginal-nonuniformly learnable if for every rate function $\mathcal { R } ^ { \prime } ,$ the class H is not λ-agnostically marginal-nonuniformly learnable at ratefaster than $\mathcal { R } ^ { \prime }$

We also define the fine-grained learning rates as defined in Bousquet et al. [2023].

Definition 3.2 (Marginal-nonuniform fine-grained rates). Let H be a concept class, let $\lambda \in [ 0 , 1 ]$ , and let $R : \mathbb { N } \to ( 0 , \infty )$ be a function with li $\begin{array} { r } { { 1 } _ { n \to \infty } R ( n ) = 0 } \end{array}$ that depends only on H and λ (and not on the distribution).

1. H is λ-agnostically marginal-nonuniformly learnable at fine-grained rate R ifthere are a universal constant $K > 0$ and a learning algorithm A such that for every marginal distribution $\mathcal { D } _ { \mathcal { X } }$ there is afunction $\delta _ { \mathcal { D } _ { \mathcal { X } } } : \mathbb { N } \to [ 0 , \infty )$ with l $\begin{array} { r } { \operatorname* { i m } _ { n  \infty } \delta _ { D _ { \mathcal { X } } } ( n ) / R ( n ) = 0 , } \end{array}$ , such that for every distribution D with marginal $\mathcal { D } _ { \mathcal { X } }$ and every $n \in \mathbb { N } ,$

$$
\mathbb { E } _ { S \sim \mathcal { D } ^ { n } } [ \mathrm { e r } _ { \mathcal { D } } ( \mathcal { A } _ { S } ) ] - ( 1 + \lambda ) \operatorname* { i n f } _ { h \in \mathcal { H } } \mathrm { e r } _ { \mathcal { D } } ( h ) \le K R ( n ) + \delta _ { \mathcal { D } _ { \mathcal { X } } } ( n ) .
$$

2. H is not λ-agnostically marginal-nonuniformly learnable at fine-grained rate faster than R if there is a universal constant $\kappa ~ > ~ 0$ such that for every learning algorithm A there is a marginal distribution $\mathcal { D } _ { \mathcal { X } }$ such thatfor infinitely many $n \in \mathbb N$ there is a distribution $\mathcal { D } _ { n }$ with marginal $\mathcal { D } _ { \mathcal { X } }$ satisfying

$$
\mathbb { E } _ { S \sim \mathcal { D } _ { n } ^ { n } } [ \mathrm { e r } _ { \mathcal { D } _ { n } } ( A _ { S } ) ] - ( 1 + \lambda ) \operatorname* { i n f } _ { h \in \mathcal { H } } \mathrm { e r } _ { \mathcal { D } _ { n } } ( h ) \ge \kappa R ( n ) .
$$

3. H is λ-agnostically marginal-nonuniformly learnable at exact fine-grained rate R ifboth 1. and 2. hold.

Our main results are based on two combinatorial dimensions; namely, the Vapnik-Chervonenkis (VC) dimension as defined in Vapnik and Chervonenkis [1971] and the uncentered VC-eluder dimension essentially defined in Brukhim et al. [2026].

Definition 3.3. (H VC-shatters S) We say H VC-shatters a set S ifthe projection ofH onto $\textit { S i . e . , } \mathcal { H } | _ { S }$ has size $2 ^ { | S | }$

(VC dimension) The VC dimension of H, denoted $\mathrm { V C } ( \mathscr { H } )$ , is the size of the largest set that H VC-shatters, $o r \infty i f$ no such largest set exists.

Definition 3.4. (uncentered VC-eluder sequence) We say that H has an infinite uncentered d-VCeluder sequence $B _ { 1 } , B _ { 2 } , \ldots i f$ each $B _ { k } \in \mathcal { X } ^ { d }$ is VC-shattered by concepts in H that are consistent with each other over $\cup _ { j < k } B _ { j }$

(uncentered VC-eluder dimension) The uncentered VC-eluder dimension ofH, denoted by uVCE(H) is defined as largest d for which H has an infinite uncentered d-VC-eluder sequence. We refer to uncentered 1-VC-eluder sequences as infinite u-eluder sequence. If no infinite sequence exists then $\mathrm { u V C E } ( \mathcal { H } ) = 0 a n d \mathrm { u V C E } ( \mathcal { H } ) = \infty i f$ there exists an infinite d-VC-eluder sequencefor every $d \in \mathbb { N } .$

The centered version of the above dimension can be defined analogously by fixing a concept $h ^ { \ast } \in$ $y x$ and replacing the consistency condition by the requirement that every concept in the corresponding witnessing family agrees with $h ^ { * }$ on all previously selected points. See Appendix C and Brukhim et al. [2026] for a detailed comparison.

We now recall a few other standard notions. For every concept class $\mathcal { H } \subseteq \mathcal { V } ^ { \mathcal { X } }$ , every finite sequence of instances $\mathcal { U } = ( x _ { 1 } , \ldots , x _ { n } ) \in \mathcal { X } ^ { n }$ and every concept $h \in \mathcal { V } ^ { \mathcal { X } }$ we define the corresponding version space by $\mathcal { V } _ { \mathcal { U } , h } = \{ f \in \mathcal { H } : f ( x _ { i } ) = h ( x _ { i } )$ for all $i \in [ n ] \}$ . Similarly, we can do so, if we have a finite sequence of instance-label pairs. In addition, we define the disagreement region of a class H by $\mathrm { D I S } ( \mathcal { H } ) = \{ x \in \mathcal { X } |$ there exists $f , g \in { \mathcal { H } }$ such that $f ( x ) \neq g ( x ) \}$ . Finally, for every class $\nu \subseteq \mathcal { y } ^ { \mathcal { X } }$ and $n \in \mathbb N$ , let $\operatorname { S H } ( { \mathcal { V } } , n ) = \{ ( x _ { 1 } , \dots , x _ { n } ) \in { \mathcal { X } } ^ { n } : \{ ( h ( x _ { 1 } ) , \dots , h ( x _ { n } ) ) : h \in { \mathcal { V } } \} = \{ 0 , 1 \} ^ { n } \}$ be the set of n-tuples that are VC-shattered by V. A tuple with a repeated point is never shattered.

Definition 3.5 (Distributional VC dimension). For $\nu \subseteq \mathcal { H }$ , a marginal distribution $\mathcal { D } _ { \mathcal { X } }$ and an integer $k \geq 0 ,$ , we write $\begin{array} { r } { \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } ) \leq k i f \mathcal { D } _ { \mathcal { X } } ^ { k + 1 } ( \mathrm { S H } ( \mathcal { V } , k + 1 ) ) = 0 , } \end{array}$ , i.e., an i.i.d. sample of $k + 1$ points from $\mathcal { D } _ { \mathcal { X } }$ is almost surely not VC-shattered by V.

If $\mathrm { V C } _ { { D } _ { \mathcal { X } } } ( \mathcal { V } ) \leq k .$ , then almost surely the projection of V onto an i.i.d. sample from $\mathcal { D } _ { \mathcal { X } }$ of any size has VC dimension at most k, and the same holds for every subclass of V. In particular $\mathrm { V C } _ { { D } _ { \mathcal { X } } } ( \mathcal { V } ^ { \prime } ) \le$ $\mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } )$ whenever $\nu ^ { \prime } \subseteq \nu$ . Our upper bounds use a first-order learner whose guarantee depends only on the distributional VC dimension; Appendix A explains why this is needed and shows that such a learner exists.

For $\nu \subseteq \varkappa$ write $\begin{array} { r } { \tau _ { \mathcal { D } , \mathcal { V } } = \operatorname* { i n f } _ { h \in \mathcal { V } } \operatorname { e r } _ { \mathcal { D } } ( h ) \ \mathrm { a n d } \ \tau _ { \mathcal { D } } = \tau _ { \mathcal { D } , \mathcal { H } } . } \end{array}$

Definition 3.6 (First-order learner). A first-order learner is a (possibly randomized) learning rule B that takes a class $\nu \subseteq \varkappa$ and a sample S, and outputs a classifier $B _ { \nu } ( S )$ , such that for a universal constant $B \geq 1$ the following holds. For every class V that is H or a version space of H, every distribution D over $\mathcal { X } \times \mathcal { V } _ { : }$ , every $k \geq 1$ with $\begin{array} { r } { \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } ) \leq k , } \end{array}$ and every $n \geq 1$

$$
\mathbb { E } _ { S \sim \mathcal { D } ^ { n } } [ \mathrm { e r } _ { \mathcal { D } } ( \mathcal { B } _ { \mathcal { V } } ( S ) ) ] \le \tau _ { \mathcal { D } , \mathcal { V } } + B \left( \sqrt { \frac { \tau _ { \mathcal { D } , \mathcal { V } } k } { n } } + \frac { k } { n } \right) .\tag{1}
$$

The rule uses only V and S; it does not know ${ \mathcal { D } } _ { \mathcal { X } } ,$ , k or $\tau _ { \mathcal { D } , \mathcal { V } }$ . The expectation includes the internal randomness ofB. The existence ofsuch a learner has been discussed in Appendix A.

## 4 Standard Agnostic Setting

In this section we note that in the standard agnostic setting the dependence on the marginal brings no benefit: both the rates and the fine-grained rates are those of uniform agnostic PAC learning. The upper bounds are uniform bounds, so they hold in particular in the marginal-nonuniform model. We also note that the classical lower bound can be realized with a single marginal i.e. the uniform distribution, so that it also holds in the marginal-nonuniform model. We first prove the fine-grained statement and then deduce the rates from it.

Theorem 4.1 (Fine-grained rate, standard agnostic). Let H be a concept class with $d = \mathrm { V C } ( \mathscr { H } ) < \infty$

1. (Upper bound.) There is a learner A such that for every distribution D and every $n \geq 1$

$$
\mathbb { E } _ { S \sim \mathcal { D } ^ { n } } [ \mathrm { e r } _ { \mathcal { D } } ( \boldsymbol { \mathcal { A } } _ { S } ) ] - \tau _ { \mathcal { D } } \leq B \left( \sqrt { \frac { d } { n } } + \frac { d } { n } \right) ,
$$

where B is the constant ofthefirst-order learner in Definition 3.6.

2. (Lower bound.) There is a marginal distribution $\mathcal { D } _ { \mathcal { X } }$ such that for every learner A and every $n \geq d$ there is a distribution D with marginal $\mathcal { D } _ { \mathcal { X } }$ satisfying

$$
\mathbb { E } _ { S \sim { D } ^ { n } } [ \mathrm { e r } _ { \mathcal { D } } ( \mathcal { A } _ { S } ) ] - \tau _ { \mathcal { D } } \geq c \sqrt { \frac { d } { n } } ,
$$

where $c > 0$ is a universal constant.

In particular, H is 0-agnostically marginal-nonuniformly learnable at exact fine-grained rate ${ \sqrt { \frac { d } { n } } } .$

Proof. Upper bound. Run the first-order learner on the whole class, ${ \cal A } = B _ { \varkappa }$ . Every marginal $\mathcal { D } _ { \mathcal { X } }$ satisfies $\begin{array} { r } { \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { H } ) \leq \mathrm { V C } ( \mathcal { H } ) = d , } \end{array}$ so (1) applies with $\nu = \varkappa$ and $k = d .$ With $\tau _ { \mathcal { D } } \leq 1$ , Definition 3.6 gives the bound. The leading term $B \sqrt { d / n }$ depends only on $d ,$ and the remaining term $B d / n$ is of lower order, so this is a fine-grained upper bound.

Lower bound. This is the classical lower bound for agnostic PAC learning [Shalev-Shwartz and Ben-David, 2014, Section 28.2]. The point is that its proof already works with a single marginal. For any class H with ${ \mathrm { V C } } ( \mathcal { H } ) = d ,$ all the hard distributions used there have the same marginal: the uniform distribution $\mathcal { D } _ { \mathcal { X } }$ on a fixed set $\{ x _ { 1 } , \ldots , x _ { d } \}$ that is VC-shattered by $\mathcal { H } .$ . Only the conditional distribution of the labels, which is slightly biased towards a labeling of the shattered set, is chosen depending on n and on the learner ${ \mathcal { A } } .$ Hence for this one marginal $\mathcal { D } _ { \mathcal { X } }$ , for every learner A and every $n \geq d ,$ , there is a distribution D with marginal $\mathcal { D } _ { \mathcal { X } }$ whose expected excess error is at least $c \sqrt { d / n }$ . In the notation of Definition 3.2, the upper bound gives item 1 with $R ( n ) = \sqrt { d / n } , K = B$ and $\begin{array} { r } { \delta ( n ) = B \frac { d } { n } = o ( R ( n ) ) } \end{array}$ and the lower bound gives item 2 with $\kappa = c ,$ since it holds for every $n \geq d .$ □

Theorem 4.2 (Rates, standard agnostic). Let H be a concept class with $| \mathcal { H } | \geq 3 .$

1. H is 0-agnostically marginal-nonuniformly learnable at exact rate ${ \begin{array} { l } { { \frac { 1 } { \sqrt { n } } } } \end{array} } i f$ and only $i f \mathrm { V C } ( \mathcal { H } ) < \infty$

2. H requires arbitrarily slow rates to be 0-agnostically marginal-nonuniformly learnable if and only $i f \mathrm { V C } ( \mathcal { H } ) = \infty$

Proof. Since $| \mathcal { H } | \geq 2$ , some point receives two different labels from concepts in $\mathcal { H } ,$ so $\mathrm { V C } ( \mathscr { H } ) \geq 1$

Finite VC dimension. Let $d = \mathrm { V C } ( \mathcal { H } ) < \infty$ . By the upper bound of Theorem $4 . 1$ , for $n \geq d$ the excess error is at most $B \Big ( \sqrt { d / n } + d / n \Big ) \leq 2 B \sqrt { d / n }$ , and for $n < d$ it is at most $1 \leq \sqrt { d / n } \leq$ $2 B \sqrt { d / n } ,$ , since $B \geq 1$ . So for every n the excess error is at most $2 B { \sqrt { d } } \cdot ( 1 / { \sqrt { n } } )$ , and H is learnable at rate $1 / \sqrt { n }$ , with constants that do not even depend on the marginal. By the lower bound of Theorem 4.1, a single marginal forces excess error at least $c { \sqrt { d } } \cdot ( 1 / { \sqrt { n } } )$ at every $n \geq d ,$ , so H is not learnable at a rate faster than $1 / \sqrt { n }$ . This proves the $\mathrm { \Omega ^ { 6 6 } \vec { 1 } f { ' } }$ direction of the first item, and it also shows that H does not require arbitrarily slow rates.

Infinite VC dimension. Every realizable distribution, i.e., one whose labels are given by some $f \in { \mathcal { H } } ,$ is allowed in our model and has $\tau _ { \mathscr { D } } = 0$ , so for it the excess error is the error itself. Hence any lower bound for realizable marginal-nonuniform learning is also a lower bound here. If $\mathrm { V C } ( \mathcal { H } ) = \infty$ , the class requires arbitrarily slow rates already in the realizable case [Hanneke et al., 2025b, Theorem 3], hence also in the 0-agnostic case. In particular it is not learnable at rate $\scriptstyle { \frac { 1 } { \sqrt { n } } }$ , which proves the “only $\mathrm { i f } ^ { \dag }$ direction of the first item. □

## 5 Relaxed Agnostic Setting

Theorem 5.1 (Rates, relaxed agnostic). Let H be a concept class with $| \mathscr { H } | \geq 3$ and let $0 < \lambda \leq 1$

1. H is λ-agnostically marginal-nonuniformly learnable at exact rate $e ^ { - n }$ ifand only ifH isfinite.

2. H is λ-agnostically marginal-nonuniformly learnable at exact rate $\begin{array} { l } { { \frac { 1 } { n } } } \end{array} i f$ and only if H is infinite and $\mathrm { V C } ( \mathcal { H } ) < \infty$

3. H requires arbitrarily slow rates to be λ-agnostically marginal-nonuniformly learnable if and only $i f \mathrm { V C } ( \mathcal { H } ) = \infty .$

The three cases are exhaustive and mutually exclusive. For the first two cases, we establish matching upper and lower bounds; the third requires only a lower bound.

Lower bounds. As in the standard setting, every realizable distribution has $\tau _ { \mathscr D } = 0$ , so for it the relaxed excess error $\mathbb { E } [ \mathrm { e r } _ { \mathcal { D } } ( \mathcal { A } _ { S } ) ] - ( 1 + \lambda ) \tau _ { \mathcal { D } }$ is the error itself, whatever λ is. Hence the lower bounds of the realizable trichotomy [Hanneke et al., 2025b, Theorem 3] carry over: a class with $| \mathcal { H } | \geq 3$ is not learnable faster than $e ^ { - n }$ , an infinite class is not learnable faster than $\textstyle { \frac { 1 } { n } }$ , and a class with $\mathrm { V C } ( \mathcal { H } ) = \infty$ requires arbitrarily slow rates.

Upper bounds. For a finite class we prove an exponential rate in Theorem 5.2 below. For an infinite class with finite VC dimension, Lemma C.1 gives $1 \leq \mathrm { u V C E } ( \mathcal { H } ) < \infty$ , and the upper bound of Theorem 5.3 gives excess error at most $K \frac { \mathrm { u V C E } ( \mathcal { H } ) } { \lambda n } + C e ^ { - c n }$ , which is a rate $\frac { 1 } { n }$

## 5.1 Finite classes

For a finite class, a plain ERM already achieves an exponential rate. The idea is that ERM compares two concepts only on the points where they disagree. On the disagreement region of the best concept and a concept with error more than $( 1 + \lambda ) \tau _ { \mathscr D }$ , the best concept is correct noticeably more often than not, and for a fixed marginal this region has probability bounded away from 0. So ERM prefers the worse concept only with exponentially small probability. This uses $\lambda > 0$ in an essential way: under the standard criterion even a class of two concepts has rate $\scriptstyle { \frac { 1 } { \sqrt { n } } }$

Theorem 5.2 (Exponential rate for finite classes). Let H be a finite concept class and $0 < \lambda \leq 1$ . Let A be any ERM over H. For every marginal $\mathcal { D } _ { \mathcal { X } } ,$ , let

$$
\delta _ { \mathcal { D } _ { \mathcal { X } } } = \operatorname* { m i n } \{ \mathcal { D } _ { \mathcal { X } } ( \mathrm { D I S } ( \{ h , g \} ) ) : h , g \in \mathcal { H } , \mathcal { D } _ { \mathcal { X } } ( \mathrm { D I S } ( \{ h , g \} ) ) > 0 \} ,
$$

with $\delta _ { D _ { \mathcal { X } } } = 1$ if this set is empty. Then for every distribution D with marginal $\mathcal { D } _ { \mathcal { X } }$ and every $n \geq 1$

$$
\mathbb { E } _ { S \sim \mathcal { D } ^ { n } } [ \mathrm { e r } _ { \mathcal { D } } ( A _ { S } ) ] - ( 1 + \lambda ) \tau _ { \mathcal { D } } \leq | \mathcal { H } | \exp \left( - \frac { \lambda ^ { 2 } \delta _ { \mathcal { D } _ { x } } } { 1 9 } n \right) .
$$

The constants $| \mathcal { H } |$ and $\frac { \lambda ^ { 2 } \delta _ { \mathcal { D } _ { \mathcal { X } } } } { 1 9 }$ depend on $\mathcal { H } , \lambda$ and $\mathcal { D } _ { \mathcal { X } }$ but not on the conditional distribution of the labels, so this is an exponential marginal-nonuniform rate.

Proof. Fix D with marginal $\mathcal { D } _ { \mathcal { X } }$ , and let $h ^ { \ast } \in \mathcal { H }$ have $\mathrm { e r } _ { \mathcal { D } } ( h ^ { * } ) = \tau _ { \mathcal { D } } ;$ it exists because H is finite. Call a concept $h \in \mathcal H$ bad if $\mathrm { e r } _ { \mathcal { D } } ( h ) > ( 1 + \lambda ) \tau _ { \mathcal { D } }$ . If ERM outputs a concept that is not bad, its error is at most $( 1 + \lambda ) \tau _ { D } ;$ otherwise we bound its error by 1. Hence

$$
\mathbb { E } [ \mathrm { e r } _ { \mathcal { D } } ( \boldsymbol { A } _ { S } ) ] \le ( 1 + \lambda ) \tau _ { \mathcal { D } } + \mathbb { P } [ \boldsymbol { A } _ { S } \mathrm { i s } \mathrm { b a d } ] ,\tag{2}
$$

and it remains to show that ERM rarely outputs a bad concept. Fix a bad concept $h .$

Only the disagreement region matters. Let $R = { \mathrm { D I S } } ( \{ h , h ^ { * } \} )$ be the set of points where h and $h ^ { * }$ disagree, and let $s = \mathcal { D } _ { \mathcal { X } } ( R )$ . Outside R the two concepts make exactly the same mistakes. Inside R the labels are binary, so exactly one of them is wrong: h makes a mistake exactly when $h ^ { * }$ is correct. Let $q = \mathbb { P } _ { ( X , Y ) \sim \mathcal { D } } [ h ^ { * } ( X ) \neq Y \mid X \in R ]$ be the probability that $h ^ { * }$ errs on a point of $R$ (this is well defined, since $s > 0$ as we show next). Then

$$
\mathrm { e r } _ { \mathcal { D } } ( h ) - \mathrm { e r } _ { \mathcal { D } } ( h ^ { * } ) = s ( 1 - q ) - s q = s ( 1 - 2 q ) .
$$

On the disagreement region, $h ^ { * }$ is right noticeably more often than wrong. Since h is bad, $\mathrm { e r } _ { \mathcal { D } } ( h ) - \mathrm { e r } _ { \mathcal { D } } ( h ^ { \ast } ) > \lambda \tau _ { \mathcal { D } } \geq 0$ , so $s > 0$ . Also $\tau _ { \mathscr D } = \mathrm { e r } _ { \mathscr D } ( h ^ { * } ) \geq s q .$ , because the mistakes of $h ^ { * }$ on R are part of all its mistakes. Hence $s ( 1 - 2 q ) > \lambda s q$ , i.e., $\begin{array} { r } { q < \frac { 1 } { 2 + \lambda } } \end{array}$ , and with $\lambda \leq 1$

$$
\frac { 1 } { 2 } - q > \frac { 1 } { 2 } - \frac { 1 } { 2 + \lambda } = \frac { \lambda } { 2 ( 2 + \lambda ) } \geq \frac { \lambda } { 6 } .\tag{3}
$$

Moreover, $s > 0$ means that the set in the definition of $\delta _ { \mathcal { D } _ { \mathcal { X } } }$ is nonempty and contains s, so $s \geq \delta _ { D _ { \mathcal { X } } }$

ERM rarely prefers h to $h ^ { * }$ . Let N be the number of sample points that fall in R, and let M be the number of those at which $h ^ { * }$ makes a mistake. On the sample, outside R the two concepts have the same mistakes, and inside $R , h ^ { * }$ makes M mistakes and $h$ makes $N - M$ . ERM can output h only if h makes at most as many mistakes on $S$ as $h ^ { * }$ , that is, only if $M \geq N / 2$ . We bound this probability in two steps.

• Given N. Conditioned on which sample points fall in R, these points are i.i.d. draws from D conditioned on $X \in R$ , so M is a sum of N independent indicators with mean q. By Hoeffding’s inequality and (3),

$$
\mathbb { P } \big [ M \ge \frac { N } { 2 } \big | N \big ] \le \exp \Big ( { - 2 N \big ( \frac { 1 } { 2 } - q \big ) ^ { 2 } } \Big ) \le e ^ { - t N } , \mathrm { w h e r e ~ } t = \frac { \lambda ^ { 2 } } { 1 8 } .
$$

This also holds when $N = 0$ , where the right side equals 1.

• Averaging over N. Each sample point falls in R independently with probability s, so N is binomial with parameters n and s. Hence, using $1 - u \leq e ^ { - u }$

$$
\begin{array} { r } { \mathbb { E } \big [ e ^ { - t N } \big ] = \big ( 1 - s \big ( 1 - e ^ { - t } \big ) \big ) ^ { n } \le \exp \big ( - s n \big ( 1 - e ^ { - t } \big ) \big ) . } \end{array}
$$

Since $t \leq \frac { 1 } { 1 8 }$ , we have $\textstyle 1 - e ^ { - t } \geq { \frac { t } { 1 + t } } \geq { \frac { 1 8 } { 1 9 } } t = { \frac { \lambda ^ { 2 } } { 1 9 } }$

Combining the two steps and $s \geq \delta _ { D _ { \mathcal { X } } }$

$$
\begin{array} { r } { \mathbb { P } [ \mathcal { A } _ { S } = h ] \le \mathbb { P } \big [ M \ge \frac { N } { 2 } \big ] \le \exp \left( - \frac { \lambda ^ { 2 } s } { 1 9 } n \right) \le \exp \left( - \frac { \lambda ^ { 2 } \delta _ { \mathcal D _ { \mathcal X } } } { 1 9 } n \right) . } \end{array}
$$

Union bound. There are at most |H| bad concepts, so $\mathbb { P } [ \mathcal { A } _ { S }$ is bad] $\leq | \mathcal { H } | \exp \bigl ( - \lambda ^ { 2 } \delta _ { \mathcal { D } _ { \mathcal { X } } } n / 1 9 \bigr )$ Together with (2) this proves the theorem. (If no concept is bad, the bound holds trivially.) □

## 5.2 Fine-grained rates

We now turn to the fine-grained rates for infinite classes of finite VC dimension. We state the result here and prove the upper and lower bounds in the next two subsections.

Theorem 5.3 (Fine-grained rate, relaxed agnostic). Let $0 < \lambda \leq 1$ and $1 \leq \mathrm { u V C E } ( \mathcal { H } ) < \infty$ . Then H is λ-agnostically marginal-nonuniformly learnable at exact fine-grained rate $\frac { \mathrm { u V C E } ( \dot { \mathcal { H } } ) } { \lambda n }$

Theorem 5.3 follows from Theorems 5.4 and 5.9, with $\begin{array} { r } { R ( n ) = \frac { \mathrm { u V C E } ( \mathcal { H } ) } { \lambda n } } \end{array}$ , which depends only on H and λ. The upper bound gives item 1 of Definition 3.2 with the constant K of Theorem 5.4, which depends only on the universal constant B of the first-order learner, and with $\delta _ { \mathcal { D } _ { \mathcal { X } } } ( n ) = C e ^ { - c n }$ , which is $o ( R ( n ) )$ ). The lower bound gives item 2 with $\kappa = \textstyle { \frac { 1 } { 2 0 4 8 } }$ , along the infinite sequence $n _ { 1 } < n _ { 2 } < \cdots$ of Theorem 5.9.

## 5.3 Fine-grained upper bound

Theorem 5.4. Let $0 < \lambda \leq 1$ and suppose $1 \leq d = \mathrm { u V C E } ( \mathcal { H } ) < \infty$ . There are a learner $\mathcal { A } _ { \lambda }$ and a constant K depending only on B from the first-order learner as in Definition 3.6 such that for every marginal $\mathcal { D } _ { \mathcal { X } }$ there are $C , c > 0$ , depending on $\mathcal { D } _ { \mathcal { X } } ,$ , λ and H only, with

$$
\mathbb { E } _ { S \sim \mathcal { D } ^ { n } } [ \mathrm { e r } _ { \mathcal { D } } ( \mathcal { A } _ { \lambda , S } ) ] - ( 1 + \lambda ) \tau _ { \mathcal { D } } \leq K \frac { d } { \lambda n } + C e ^ { - c n }
$$

for every D with marginal $\mathcal { D } _ { \mathcal { X } }$ and every n $\geq 1$ .

Idea. In the realizable case Brukhim et al. [2026], the learner uses part of the sample to form a version space. After enough samples this version space is “simple”: a fresh sample from $\mathcal { D } _ { \mathcal { X } }$ almost never contains d + 1 points that it shatters. A good learner is then run inside this simple class. In the agnostic case the sample is not consistent with any single concept, so we cannot form this version space. However, a good concept h makes few mistakes, so a small random subsample is likely to contain none of them. The version space of such a clean subsample is again simple, and it still contains $h .$ We do not know which subsamples are clean, so we try all of them: every subsample gives one candidate predictor, obtained by running the first-order learner inside its version space. Finally, we choose among the candidates on fresh data using exponential weights. The factor $( 1 + \lambda )$ is what pays for this last selection step.

## 5.3.1 The learner

Fix $\begin{array} { r } { \eta = \frac { \lambda } { 1 6 } } \end{array}$ and $\begin{array} { r } { p = \frac { \lambda ^ { 2 } } { 1 0 2 4 } } \end{array}$ . For $n < 6$ the learner outputs a fixed $h \in { \mathcal { H } }$ . For $n \geq 6$ let $m = \lfloor n / 3 \rfloor$ and $r = n - 2 m .$ , so that ${ \begin{array} { l } { { \frac { n } { 4 } } \leq m \leq r } \end{array} }$ , and split S into three independent parts, one for each stage of the idea above:

$$
\underbrace { S ^ { ( 0 ) } = ( ( X _ { i } , Y _ { i } ) ) _ { i = 1 } ^ { m } } _ { \mathrm { t o l o c a l i z e } } , \qquad \underbrace { S ^ { ( 1 ) } \mathrm { o f ~ s i z e } m } _ { \mathrm { t o ~ t r a i n ~ t h e ~ c a n d i d a t e s } } , \qquad \underbrace { S ^ { ( 2 ) } \mathrm { o f ~ s i z e } r } _ { \mathrm { t o ~ c h o o s e ~ a m o n g ~ t h e m } } .
$$

1. Candidates (experts). Let

$$
\mathcal { E } = \{ \star \} \cup \{ I \subseteq [ m ] : ( X _ { i } , Y _ { i } ) _ { i \in I } \mathrm { { i s } ~ r e a l i z a b l e ~ b y ~ \mathcal { H } \} } .
$$

For $I \in \mathcal { E } \setminus \{ \star \}$ , form the version space $\mathcal { V } _ { I } = \mathcal { V } _ { ( X _ { i } , Y _ { i } ) _ { i \in I } }$ and train the first-order learner inside $\mathrm { i t } , g _ { I } = B _ { \mathcal { V } _ { I } } ( S ^ { ( 1 ) } )$ . For the special expert ⋆, let $g _ { \star }$ be an ERM over $\mathcal { H }$ on $S ^ { ( 1 ) }$ . The experts are the elements of $\varepsilon \colon$ two different experts may output the same classifier, and we still treat them as different experts. This use of realizable subsamples follows the agnostic-to-realizable reductions of David et al. [2016], Hopkins et al. [2022], and Hanneke et al. [2025a].

2. Prior. Let $w ( I ) = p ^ { | I | } ( 1 - p ) ^ { m - | I | }$ . This is the probability that a random subset of $[ m ]$ , which contains each index independently with probability $p ,$ is exactly I. Let $\begin{array} { r } { Z = \sum _ { I \in { \mathcal E } \backslash \{ \star \} } w ( I ) } \end{array}$ . The empty subsample is realizable, so $Z \geq w ( \emptyset ) = ( 1 - p ) ^ { m } > 0 .$ , and $Z \leq 1$ . Set

$$
\pi ( \star ) = \frac { 1 } { 2 } , \qquad \pi ( I ) = \frac { w ( I ) } { 2 Z } \geq \frac { w ( I ) } { 2 } \quad \mathrm { f o r } I \in \mathcal { E } \setminus \{ \star \} .
$$

So $\pi$ is a probability distribution on $\mathcal { E }$ that favors small subsamples.

3. Aggregation. Let $( X _ { t } ^ { \prime } , Y _ { t } ^ { \prime } ) , t \in [ r ]$ , be the examples of $S ^ { ( 2 ) }$ , and for $e \in { \mathcal { E } }$ let $\ell _ { t } ( e ) \in [ 0 , 1 ]$ be the probability that $g _ { e }$ errs on $( X _ { t } ^ { \prime } , Y _ { t } ^ { \prime } )$ (over the internal randomness of $g _ { e }$ , if any). Run exponential weights with rate η starting from π:

$$
w _ { 1 } = \pi , \qquad w _ { t + 1 } ( e ) = { \frac { w _ { t } ( e ) e ^ { - \eta \ell _ { t } ( e ) } } { \sum _ { e ^ { \prime } \in { \mathcal { E } } } w _ { t } ( e ^ { \prime } ) e ^ { - \eta \ell _ { t } ( e ^ { \prime } ) } } } .
$$

Output the randomized classifier $\widehat { h }$ that draws $t \in [ r ]$ uniformly, then an expert $e \in { \mathcal { E } }$ from $w _ { t }$ and predicts $g _ { e } ( x )$ . Its error is $\begin{array} { r } { \operatorname { e r } _ { \mathcal { D } } ( \widehat { h } ) = \frac { 1 } { r } \sum _ { t = 1 } ^ { r } \sum _ { e \in \mathcal { E } } w _ { t } ( e ) \operatorname { e r } _ { \mathcal { D } } ( g _ { e } ) } \end{array}$

## 5.3.2 Version spaces become simple

The first tool is taken from Brukhim et al. [2026]: after enough samples from $\mathcal { D } _ { \mathcal { X } }$ , all the version spaces are simple at the same time.

Lemma 5.5 (Exhaustion). If H has no infinite uncentered (d + 1)-VC-eluder sequence, then for every marginal $\mathcal { D } _ { \mathcal { X } }$

$$
\operatorname* { l i m } _ { m  \infty } \mathbb { P } _ { \mathcal { U } \sim \mathcal { D } _ { \mathcal { X } } ^ { m } } [ t h e r e \ i s \ h \in \mathcal { H } \ w i t h \ \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } _ { \mathcal { U } , h } ) > d ] = 0 .
$$

This is Corollary 20 of Brukhim et al. [2026], stated at width $d + 1$ . We include the proof, which follows their Lemma 18.

Proof. Suppose, towards a contradiction, that the limit is not 0 for some $\mathcal { D } _ { \mathcal { X } }$ . Let $X _ { 1 } , X _ { 2 } , . . .$ . be i.i.d.   
from $\mathcal { D } _ { \mathcal { X } }$ and write $\mathcal { U } _ { m } = ( X _ { 1 } , \ldots , X _ { m } )$ , with $\mathcal { U } _ { 0 }$ empty.

Version spaces depend only on labels. For $\sigma \in \{ 0 , 1 \} ^ { m }$ let $\mathcal { V } _ { \mathcal { U } _ { m , \sigma } } = \{ f \in \mathcal { H } : f ( X _ { i } ) = \sigma _ { i }$ for all $i \in [ m ] \}$ . For every $h \in \mathcal H$ we have $\mathcal { V } _ { \mathcal { U } _ { m } , h } = \mathcal { V } _ { \mathcal { U } _ { m } , \sigma }$ with $\sigma = ( h ( X _ { 1 } ) , \dots , h ( X _ { m } ) )$ . So, although H may be uncountable, there are at most $2 ^ { m }$ different version spaces on $\boldsymbol { \mathcal { U } } _ { m }$

A bad event that never goes away. Let $E _ { m }$ be the event that some $\sigma \in \{ 0 , 1 \} ^ { m }$ satisfies $\mathcal { V } _ { \mathcal { U } _ { m } , \sigma } \ne \emptyset$ and $\mathcal { D } _ { \mathcal { X } } ^ { d + 1 } ( \mathrm { S H } ( \mathcal { V } _ { \mathcal { U } _ { m } , \sigma } , d + 1 ) ) > 0$ . This is exactly the event that some $h \in \mathcal H$ has $\mathrm { V C } _ { \mathscr { D } _ { \mathscr { X } } } ( \mathcal { V } _ { \mathscr { U } _ { m } , h } ) > d ,$ and it is a finite union of measurable events. If $\sigma ^ { \prime } \in \{ 0 , 1 \} ^ { m + 1 }$ witnesses $E _ { m + 1 }$ , then its first m coordinates σ witness $E _ { m }$ , because $\mathcal { V } _ { \mathcal { U } _ { m + 1 } , \sigma ^ { \prime } } \subseteq \mathcal { V } _ { \mathcal { U } _ { m } , \sigma }$ and a smaller class shatters fewer tuples. Hence $E _ { m + 1 } \subseteq E _ { m }$ . By assumption li $\mathbb { 1 } _ { m } \mathbb { P } [ E _ { m } ] > 0$ , so $\begin{array} { r } { E _ { \infty } = \bigcap _ { m } E _ { m } } \end{array}$ has positive probability.

A set of shattered tuples of positive mass. On $E _ { m }$ , let $\sigma _ { m }$ be the lexicographically first $\sigma \in \{ 0 , 1 \}$ m that witnesses $E _ { m }$ , and let $A _ { m } = \mathrm { S H } ( \mathcal { V } _ { \mathcal { U } _ { m } , \sigma _ { m } } , d + 1 )$ , so that $\mathcal { D } _ { \mathcal { X } } ^ { d + 1 } ( A _ { m } ) > 0$ . Off $E _ { m }$ , let $A _ { m } = \emptyset$ Since $\sigma _ { m }$ is chosen among finitely many candidates by a fixed rule, $A _ { m }$ is determined by $\boldsymbol { \mathcal { U } } _ { m }$ in a measurable way; we never choose a concept from $\mathcal { H }$

Future samples hit this set. Cut the future samples $X _ { m + 1 } , X _ { m + 2 } , . . .$ . into consecutive disjoint blocks of $d + 1$ points. These blocks are i.i.d. from $\bar { \mathcal { D } } _ { \mathcal { X } } ^ { d + 1 }$ and independent of $\boldsymbol { \mathcal { U } } _ { m }$ . Conditioned on $\mathcal { U } _ { m } .$ , each block lies in $A _ { m }$ independently with probability $\mathcal { D } _ { \mathcal { X } } ^ { d + 1 } ( A _ { m } )$ , which is positive on $E _ { m }$ . By the second Borel–Cantelli lemma, almost surely on $E _ { m }$ infinitely many blocks lie in $A _ { m }$ . There are countably many m, so almost surely this holds for all m at once.

Build the forbidden sequence. Fix a sample path in $E _ { \infty }$ on which the conclusion of the previous step holds for all $m ;$ the rest is deterministic. Set $t _ { 0 } = 0$ . Given $t _ { r - 1 }$ , pick a block after position $t _ { r - 1 }$ that lies in $A _ { t _ { r - 1 } } ,$ call its set of points $B _ { r }$ , and let $t _ { r }$ be the position of its last point. By definition of $A _ { t _ { r - 1 } }$ , the set $B _ { r }$ is VC-shattered by $\mathscr { V } _ { \mathcal { U } _ { t _ { r - 1 } } , \sigma _ { t _ { r - 1 } } } .$ All members of this version space have the labels $\sigma _ { t _ { r - 1 } } \mathbf { o n } \mathcal { U } _ { t _ { r - 1 } }$ , so they agree with each other there, and $\mathcal { U } _ { t _ { r - 1 } }$ contains $B _ { 1 } , \ldots , B _ { r - 1 }$ . Hence $B _ { 1 } , B _ { 2 } , \ldots$ is an infinite uncentered $( d + 1 )$ )-VC-eluder sequence, a contradiction. □

Since $\operatorname { u V C E } ( \mathcal { H } ) = d ,$ the class has no infinite uncentered (d + 1)-VC-eluder sequence and Lemma 5.5 applies. For the rest of this section fix a marginal $\mathcal { D } _ { \mathcal { X } }$ . By the lemma there is $q$ which depends on $\mathcal { D } _ { \mathcal { X } }$ such that with

$$
\mathbb { P } _ { \mathcal { U } \sim \mathcal { D } _ { \mathcal { X } } ^ { q } } [ \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } _ { \mathcal { U } , h } ) \le d \mathrm { ~ f o r ~ e v e r y ~ } h \in \mathcal { H } ] \ge \frac { 3 } { 4 } .\tag{4}
$$

We call a q-tuple U good if the event in (4) holds, and we set

$$
\rho = \frac { 1 } { 4 q } , \qquad \beta = \frac { 3 p } { 4 } \Bigl ( 1 - 2 ^ { - 1 / q } \Bigr ) > 0 .
$$

These quantities are used only in the analysis; the learner does not know $q , \rho$ or $\beta _ { - }$

## 5.3.3 Clean subsamples localize

Lemma 5.5 is about samples from $\mathcal { D } _ { \mathcal { X } }$ . The next lemma transfers it to what the learner can actually use: a random subsample on which a fixed good concept h makes no mistake.

Lemma 5.6 (Localization from a clean subsample). Let $h \in \mathcal H$ with $a = \mathrm { e r } _ { \mathcal { D } } ( h ) \leq \rho .$ . Draw $S ^ { ( 0 ) } \sim \mathcal { D } ^ { m }$ and independent $b _ { 1 } , \dots , b _ { m } \sim \operatorname { B e r } ( p )$ . Keep the indices that are selected and on which h is correct,

$$
I = \left\{ i \in [ m ] : b _ { i } = 1 a n d Y _ { i } = h ( X _ { i } ) \right\} , \qquad \mathcal { U } _ { I } = ( X _ { i } ) _ { i \in I } .
$$

Then

$$
\mathbb { P } [ \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } _ { \mathcal { U } _ { I } , h } ) > d ] \le 2 e ^ { - \beta m } .
$$

Proof. The kept points are i.i.d.. The triples $( X _ { i } , Y _ { i } , b _ { i } )$ are i.i.d., and index i is kept with probability $p ( 1 - a )$ . A kept point has the law $\nu _ { h }$ of X conditioned on $Y = h ( X )$ . Hence the number of kept points is $J = | I | \sim \mathrm { B i n } ( m , p ( 1 - a ) )$ , and given J the kept points are i.i.d. from $\nu _ { h }$

$\nu _ { h }$ is close to $\mathcal { D } _ { \mathcal { X } }$ . Conditioning on whether h is correct gives $\mathcal { D } _ { \mathcal { X } } = ( 1 - a ) \nu _ { h } + a \nu ^ { \prime } .$ , where $\nu ^ { \prime }$ is the law of X given $Y \neq h ( X )$ . So $\mathrm { T V } ( \nu _ { h } , { \cal D } _ { \mathcal { X } } ) \leq a ,$ and for q-tuples $\begin{array} { r } { \mathrm { T V } ( \nu _ { h } ^ { q } , { \mathcal { D } _ { \mathcal { X } } ^ { q } } ) \leq q a \leq q \rho = \frac { 1 } { 4 } } \end{array}$ $\mathrm { B y } \left( 4 \right)$ , a q-tuple drawn from $\nu _ { h } ^ { q }$ is therefore good with probability at least $\frac { 3 } { 4 } - \frac { 1 } { 4 } = \frac { 1 } { 2 }$

One good block is enough. Split the kept points into $\lfloor J / q \rfloor$ disjoint blocks of q points. If some block $\mathcal { U } ^ { \prime }$ is good, then $\begin{array} { r } { \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } _ { \mathcal { U } ^ { \prime } , h } ) \leq d . } \end{array}$ , and since $\mathcal { V } _ { \mathcal { U } _ { I } , h } \subseteq \mathcal { V } _ { \mathcal { U } ^ { \prime } , h }$ also $\begin{array} { r } { \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } _ { \mathcal { U } _ { I } , h } ) \leq d . } \end{array}$ . The blocks are independent and each fails to be good with probability at most $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ , so

$$
\mathbb { P } [ \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } _ { \mathcal { U } _ { I } , h } ) > d \mid J ] \le \left( \frac { 1 } { 2 } \right) ^ { \lfloor J / q \rfloor } \le 2 \cdot 2 ^ { - J / q } .
$$

Average over J. For $J \sim \mathrm { B i n } ( m , p ^ { \prime } )$ and $x \in [ 0 , 1 ]$ we have $\mathbb { E } \big [ { \boldsymbol { x } } ^ { J } \big ] \ = \ ( 1 - { p ^ { \prime } ( 1 - { x } ) } ) ^ { m } \ \leq$ $e ^ { - m p ^ { \prime } ( 1 - x ) }$ . With $\begin{array} { r } { p ^ { \prime } = p ( 1 - a ) \ge \frac { 3 } { 4 } p } \end{array}$ (as $a \leq \rho \leq { \frac { 1 } { 4 } } )$ and $x = 2 ^ { - 1 / q }$ this gives E $\left[ 2 ^ { - J / q } \right] \leq e ^ { - \beta m }$ , and the claim follows from the previous step. □

## 5.3.4 Choosing among the candidates

The next lemma is the standard guarantee of exponential weights [Freund and Schapire, 1997], stated against a distribution $Q$ over experts instead of a single expert and combined with the usual online-tobatch conversion [Cesa-Bianchi et al., 2004]. The price for competing with $Q$ is $\operatorname { D } _ { \mathrm { K L } } ( Q \lVert \boldsymbol { \pi } )$ , which is small whenever the prior puts enough mass where $Q$ lives.

Lemma 5.7 (Exponential weights against a distribution). Let $a _ { \eta } = 1 - e ^ { - \eta }$ and $b _ { \eta } = \eta / a _ { \eta }$ . For every distribution Q on $\mathcal { E }$ that is determined $b y \ S ^ { ( 0 ) }$ and $S ^ { ( 1 ) }$

$$
\mathbb { E } \Big [ \mathrm { e r } _ { \mathcal { D } } \big ( \widehat { h } \big ) \Big | S ^ { ( 0 ) } , S ^ { ( 1 ) } \Big ] \leq b _ { \eta } \sum _ { e \in \mathcal { E } } Q ( e ) \mathrm { e r } _ { \mathcal { D } } ( g _ { e } ) + \frac { \mathrm { D } _ { \mathrm { K L } } ( Q \| \pi ) } { r a _ { \eta } } ,
$$

where log is the natural logarithm, and for a randomized expert, er<sub>D</sub>(g<sub>e</sub>) includes its internal randomness. $H Q = \pi ( \cdot \mid G )$ for a set $G \subseteq { \mathcal { E } }$ with $\pi ( G ) > 0$ , then $\begin{array} { r } { \mathrm { D } _ { \mathrm { K L } } ( Q \| \pi ) = \log \frac { 1 } { \pi ( G ) } } \end{array}$

Proof. Condition on $S ^ { ( 0 ) }$ and $S ^ { ( 1 ) }$ , so that the experts, π and $Q$ are fixed. Let $\begin{array} { r } { L _ { e } = \sum _ { t = 1 } ^ { r } \ell _ { t } ( e ) } \end{array}$ be the total loss of expert $e _ { \mathrm { : } }$ , and consider the potential $\begin{array} { r } { W _ { t } = \sum _ { e \in \mathcal { E } } \pi ( e ) \exp \bigl ( - \eta \sum _ { s < t } \ell _ { s } ( e ) \bigr ) } \end{array}$ , so that $W _ { 1 } = 1$ and $\begin{array} { r } { w _ { t } ( e ) = { \pi } ( e ) \exp \left( - \eta \sum _ { s < t } \ell _ { s } ( e ) \right) / W _ { t } } \end{array}$

The potential drops when the mixture makes a mistake. Using $e ^ { - \eta x } \leq 1 - a _ { \eta } x$ for $x \in [ 0 , 1 ]$ and $1 - u \leq e ^ { - u }$ ，

$$
\frac { W _ { t + 1 } } { W _ { t } } = \sum _ { e \in \mathcal { E } } w _ { t } ( e ) e ^ { - \eta \ell _ { t } ( e ) } \leq 1 - a _ { \eta } \langle w _ { t } , \ell _ { t } \rangle \leq e ^ { - a _ { \eta } \langle w _ { t } , \ell _ { t } \rangle } ,
$$

so

$$
\log W _ { r + 1 } \leq - a _ { \eta } \sum _ { t = 1 } ^ { r } \langle w _ { t } , \ell _ { t } \rangle .
$$

The potential stays large if Q is good. By Jensen’s inequality,

$$
\log W _ { r + 1 } = \log \sum _ { \epsilon \in \mathcal { E } } \pi ( \epsilon ) e ^ { - \eta L _ { \epsilon } } \geq \sum _ { \epsilon \in \mathcal { E } } Q ( \epsilon ) \log \frac { \pi ( \epsilon ) e ^ { - \eta L _ { \epsilon } } } { Q ( \epsilon ) } = - \mathrm { D } _ { \mathrm { K L } } ( Q \| \pi ) - \eta \sum _ { \epsilon \in \mathcal { E } } Q ( \epsilon ) L _ { \epsilon } ,
$$

where the sums over e with $Q ( e ) = 0$ are dropped.

Combine and take expectations. The previous two steps give $\begin{array} { r } { a _ { \eta } \sum _ { t } \langle w _ { t } , \ell _ { t } \rangle \leq \eta \sum _ { e \in \mathcal { E } } Q ( e ) L _ { e } + } \end{array}$ $\operatorname { D } _ { \mathrm { K L } } ( Q \| \pi )$ . The weights $w _ { t }$ depend only on $S ^ { ( 0 ) } , S ^ { ( 1 ) }$ and the first $t - 1$ examples of $S ^ { ( 2 ) }$ , and the t-th example of $S ^ { ( 2 ) }$ is a fresh draw from $\mathcal { D } ,$ independent of these. $\mathrm { { S o . } }$ conditioned on $S ^ { ( 0 ) }$ and $S ^ { ( 1 ) }$ the expectation of $\langle w _ { t } , \ell _ { t } \rangle$ is the expectation of $\begin{array} { r } { \sum _ { e \in \mathcal { E } } w _ { t } ( e ) \exp ( g _ { e } ) } \end{array}$ , and the expectation of $\ell _ { t } ( e )$ is $\mathrm { e r } _ { \mathcal { D } } ( g _ { e } )$ . Summing over $t ,$ the left side becomes $a _ { \eta } r \mathbb { E } \Big [ \mathrm { e r } _ { \mathcal { D } } ( \widehat { h } ) \vert S ^ { ( 0 ) } , S ^ { ( 1 ) } \Big ]$ and the right side becomes $\begin{array} { r } { \eta r \sum _ { e \in \mathcal { E } } Q ( e ) \mathrm { e r } _ { \mathcal { D } } ( g _ { e } ) + \mathrm { D } _ { \mathrm { K L } } ( Q \| \pi ) } \end{array}$ . Divide by $r a _ { \eta }$

For $Q = \pi ( \cdot \mid G )$ with $\pi ( G ) > 0$ , the ratio $Q ( e ) / \pi ( e )$ equals $\frac { 1 } { \pi ( G ) }$ on $G$ and $Q ( e ) = 0$ off $G ,$ which gives the last claim. □

Finally, when $\tau _ { D }$ is not small we will not need localization at all; a plain ERM is good enough.

Lemma 5.8 (ERM at fixed accuracy). Suppose $v = \mathrm { V C } ( \mathcal { H } ) < \infty .$ . For every $\zeta \in \mathsf { \Gamma } ( 0 , 1 ]$ there are $A , b > 0$ , depending only on v and $\zeta ,$ such that for every distribution D and every $m \geq 1$ , an $E R M g _ { \star }$ over H on $S \sim \mathcal { D } ^ { m }$ satisfies

$$
\begin{array} { r } { \mathbb { E } _ { S \sim \mathcal { D } ^ { m } } [ \mathrm { e r } _ { \mathcal { D } } ( g _ { \star } ) ] \le \tau _ { \mathcal { D } } + \zeta + A e ^ { - b m } . } \end{array}
$$

Proof. Write $\mathrm { e r } _ { S } ( h )$ for the fraction of examples in S that h misclassifies.

Uniform convergence. By the VC inequality [Devroye et al., 1996, Theorem 12.5] and Sauer’s lemma, for $m \geq v .$

$$
\mathbb { P } \bigg [ \operatorname* { s u p } _ { h \in \mathcal { H } } | \mathrm { e r } _ { \mathcal { D } } ( h ) - \mathrm { e r } _ { S } ( h ) | > \frac { \zeta } { 2 } \bigg ] \leq 8 \Big ( \frac { e m } { v } \Big ) ^ { v } e ^ { - m \zeta ^ { 2 } / 1 2 8 } .
$$

ERM is accurate outside this event. If $\begin{array} { r } { \dot { \mathbf { \sigma } } | \mathrm { e r } _ { \mathcal { D } } ( h ) - \mathrm { e r } _ { S } ( h ) | \leq \frac { \zeta } { 2 } } \end{array}$ for every $h \in \mathcal H$ , then, since $g _ { \star }$ minimizes the empirical error, for every $h \in \mathcal H$

$$
\operatorname { e r } _ { { \mathcal { D } } } ( g _ { \star } ) \leq \operatorname { e r } _ { S } ( g _ { \star } ) + \frac { \zeta } { 2 } \leq \operatorname { e r } _ { S } ( h ) + \frac { \zeta } { 2 } \leq \operatorname { e r } _ { { \mathcal { D } } } ( h ) + \zeta .
$$

Taking the infimum over $h \in \mathcal H$ gives $\mathrm { e r } _ { \mathscr { D } } ( g _ { \star } ) \leq \tau _ { \mathscr { D } } + \zeta$ . On the complementary event we bound $\mathrm { e r } _ { { D } } ( g _ { \star } ) \le 1$ . Hence, for $m \geq v .$

$$
\mathbb { E } [ \mathrm { e r } _ { \mathcal { D } } ( g _ { \star } ) ] \le \tau _ { \mathcal { D } } + \zeta + 8 \Big ( \frac { e m } { v } \Big ) ^ { v } e ^ { - m \zeta ^ { 2 } / 1 2 8 } .
$$

Absorbing the polynomial factor. Let $b = \zeta ^ { 2 } / 2 5 6$ . The function $m \mapsto 8 ( e m / v ) ^ { v } e ^ { - m \zeta ^ { 2 } / 2 5 6 } { \mathrm { ~ i s } }$ bounded on $[ v , \infty ) ;$ ; let $A _ { 1 }$ be its supremum. Then for $m \geq v$ the last term is at most $A _ { 1 } e ^ { - b m }$ . For $m < v$ we use the trivial bound $\mathbb { E } [ \mathrm { e r } _ { \mathcal { D } } ( g _ { \star } ) ] \le 1 \le e ^ { b v } e ^ { - b m }$ . The lemma follows with $A = \operatorname* { m a x } \{ A _ { 1 } , e ^ { b v } \}$ which depends only on v and $\zeta .$ □

## 5.3.5 Proof of the upper bound

Proof of Theorem 5.4. Fix a distribution D with marginal $\mathcal { D } _ { \mathcal { X } }$ . For $n < 6$ the learner outputs a fixed concept and the bound holds trivially, since $\begin{array} { r } { K \frac { d } { \lambda n } \geq \frac { \bar { K } } { 6 } \geq 1 } \end{array}$ . We assume $n \geq 6$ from now on. We will use the elementary bounds

$$
b _ { \eta } \leq 1 + \eta , \qquad a _ { \eta } \geq \frac \eta 2 , \qquad \log \frac { 1 } { 1 - p } \leq 2 p , \qquad \frac { n } { 4 } \leq m \leq r .\tag{5}
$$

The proof has two cases. If the best concept in H has small error, we compete with the candidates trained in clean version spaces (Case 1). Otherwise the ERM candidate $g _ { \star }$ already suffices (Case 2).

Case $\begin{array} { r } { \mathbf { 1 } \colon \tau _ { \mathcal { D } } \le \rho / 2 . } \end{array}$

A comparator. Fix any $h \in \mathcal H$ with $a = \mathrm { e r } _ { \mathcal { D } } ( h ) \leq \rho ;$ such h exist because $\tau _ { D } < \rho .$ . Let

$$
E _ { h } = \{ i \leq m : Y _ { i } \neq h ( X _ { i } ) \} , \qquad K _ { h } = | E _ { h } | ,
$$

be the positions of the mistakes of h on $S ^ { ( 0 ) }$ and their number, so $\mathbb { E } [ K _ { h } ] = m a$ . A subsample I is clean if $I \cap E _ { h } = \emptyset$ . For a clean I the labels on I are those of $h .$ , so $I \in { \mathcal { E } }$ , and

$$
\mathcal { V } _ { I } = \mathcal { V } _ { \mathcal { U } _ { I } , h } \ni h .\tag{6}
$$

The good experts. Define

$$
G _ { h } = \left\{ I \in \mathcal { E } \setminus \left\{ \star \right\} : I \mathrm { ~ i s ~ c l e a n ~ a n d ~ } \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } _ { \mathcal { U } _ { I } , h } ) \leq d \right\} .
$$

These are the experts trained in a version space that is simple and contains h. The set $G _ { h } \subseteq { \mathcal { E } }$ depends on $S ^ { ( 0 ) }$ and h only. We show that every good expert is accurate and that the good experts together have enough prior mass; exponential weights then finds them in the following steps.

Each good expert is accurate. Let $I \in G _ { h }$ . By (6), $\mathcal { V } _ { I }$ is a version space of H with $\tau _ { \mathcal { D } , \mathcal { V } _ { I } } \leq a$ and $\begin{array} { r } { \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } _ { I } ) \leq d . } \end{array}$ Since $S ^ { ( 1 ) }$ is independent of $S ^ { ( 0 ) }$ , we can apply Definition $3 . 6$ to the fixed class $\mathcal { V } _ { I }$ with $k = d .$ . Its right-hand side is increasing in $\tau _ { \mathcal { D } , \mathcal { V } _ { I } }$ , so writing $z = d / m$

$$
\begin{array} { r } { \mathbb { E } \left[ \mathrm { e r } _ { \mathcal { D } } ( g _ { I } ) \Big | S ^ { ( 0 ) } \right] \leq a + B \sqrt { a z } + B z , } \end{array}
$$

where the expectation is over $S ^ { ( 1 ) }$ and the internal randomness of B. To separate a from $z \textrm { \textbf { W e } }$ use $\begin{array} { r } { B \sqrt { a z } \le \varepsilon a + \frac { B ^ { 2 } z } { 4 \varepsilon } } \end{array}$ with $\begin{array} { r } { \varepsilon = \frac { \lambda } { 1 6 } } \end{array}$

$$
\mathbb { E } \Big [ \mathrm { e r } _ { \mathcal { D } } ( g _ { I } ) \mid S ^ { ( 0 ) } \Big ] \leq ( 1 + \varepsilon ) a + \frac { ( B + B ^ { 2 } ) z } { \varepsilon } .\tag{7}
$$

This is the only place where the first-order learner is used, and it is why we need a first-order bound: a term $\sqrt { z }$ in place of $\sqrt { a z }$ could not be absorbed into εa.

The good experts have enough prior mass. Recall that w is the law of a random subset $R \subseteq$ [m] that contains each index independently with probability $p .$ This subset is clean with probability $\begin{array} { r } { ( 1 - p ) ^ { K _ { h } } , \thinspace \mathrm { s o } \sum _ { I \mathrm { c l e a n } } w ( I ) = ( 1 - p ) ^ { K _ { h } } } \end{array}$ . Let

$$
b _ { h } = \mathbb { P } \Big [ \mathrm { V C } _ { \mathscr { D } _ { \mathcal { X } } } ( \mathscr { V } _ { \mathscr { U } _ { R } , h } ) > d  { \mathrm { ~ | ~ } } S ^ { ( 0 ) } , ~ R \mathrm { i s ~ c l e a n } \Big ]
$$

be the fraction of the clean mass that is not good. Then, using $\pi ( I ) \geq { \frac { 1 } { 2 } } w ( I )$

$$
\pi ( G _ { h } ) = \sum _ { I \in G _ { h } } \pi ( I ) \geq \frac { 1 } { 2 } \sum _ { I \in G _ { h } } w ( I ) = \frac { 1 } { 2 } ( 1 - p ) ^ { K _ { h } } ( 1 - b _ { h } ) .\tag{8}
$$

Conditioned on $S ^ { ( 0 ) }$ and on R being clean, R has the same law as the set I of Lemma 5.6 given $S ^ { ( 0 ) }$ each index outside $E _ { h }$ is included independently with probability $p .$

Hence $b _ { h } = \mathbb { P } \big [ \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } _ { \mathcal { U } _ { I } , h } ) > d \mid S ^ { ( 0 ) } \big ]$ , and taking expectations over $S ^ { ( 0 ) }$ , Lemma $5 . 6$ gives $\mathbb { E } [ b _ { h } ] \le 2 e$ <sup>−βm</sup>. By Markov’s inequality, the event $T _ { h } = \left\{ b _ { h } \le \frac { 1 } { 2 } \right\}$ satisfies $\mathbb { P } [ T _ { h } ^ { c } ] \le 4 e ^ { - \beta m }$ . On $T _ { h } , ( 8 )$ gives $\pi ( G _ { h } ) \geq { \textstyle { \frac { 1 } { 4 } } } ( 1 - p ) ^ { K _ { h } }$ . In particular $\pi ( G _ { h } ) > 0 , \ : \mathrm { s o } \ : Q ^ { - } = \pi ( \cdot \mid G _ { h } )$ is well defined. The factor $( 1 - p ) ^ { K _ { h } }$ is the reason for the choice of prior: every mistake of $h$ costs only a factor $( 1 - p )$ , so the cost below is proportional to the error of $h .$

Aggregate. The event $T _ { h }$ and the distribution $Q = \pi ( \cdot \mid G _ { h } )$ depend on $S ^ { ( 0 ) }$ only. On $T _ { h }$ we apply Lemma 5.7 with this $Q ;$ by the previous steps and (5),

$$
\mathrm { D } _ { \mathrm { K L } } ( Q \| \pi ) = \log { \frac { 1 } { \pi ( G _ { h } ) } } \le \log 4 + K _ { h } \log { \frac { 1 } { 1 - p } } \le \log 4 + 2 p K _ { h } .
$$

On $T _ { h } ^ { c }$ we use the trivial bound $\mathrm { e r } _ { \mathcal { D } } ( \widehat { h } ) \leq 1$ . Taking expectations, using (7) for every expert in the support of $Q$ , and $\mathbb { E } [ K _ { h } ] = m a$

$$
\mathbb { E } \bigg [ \mathrm { e r } _ { \mathcal { D } } ( \widehat { h } ) \bigg ] \leq b _ { \eta } \bigg ( ( 1 + \varepsilon ) a + \frac { ( B + B ^ { 2 } ) z } { \varepsilon } \bigg ) + \frac { \log 4 + 2 p m a } { r a _ { \eta } } + 4 e ^ { - \beta m } .\tag{9}
$$

Simplify. We collect the terms of (9) using (5), $\begin{array} { r } { \eta = \varepsilon = \frac { \lambda } { 1 6 } } \end{array}$ and $\begin{array} { r } { p = \frac { \lambda ^ { 2 } } { 1 0 2 4 } } \end{array}$

• Terms with a. Their coefficient is at most

$$
( 1 + \eta ) ( 1 + \varepsilon ) + \frac { 2 p m } { r } \cdot \frac { 2 } { \eta } \leq \left( 1 + \frac { \lambda } { 1 6 } \right) ^ { 2 } + \frac { 4 p } { \eta } = \left( 1 + \frac { \lambda } { 1 6 } \right) ^ { 2 } + \frac { \lambda } { 1 6 } \leq 1 + \frac { \lambda } { 4 } .
$$

• Terms oforder ${ \frac { 1 } { n } } .$ . Since $\textstyle m , r \geq { \frac { n } { 4 } }$

$$
b _ { \eta } \frac { ( B + B ^ { 2 } ) z } { \varepsilon } + \frac { \log 4 } { r a _ { \eta } } \leq \frac { 1 7 } { 1 6 } \cdot \frac { 1 6 ( B + B ^ { 2 } ) } { \lambda } \cdot \frac { 4 d } { n } + \frac { 1 2 8 \log 4 } { \lambda n } \leq K \frac { d } { \lambda n } ,
$$

where $K = 6 8 ( B + B ^ { 2 } ) + 1 7 8$

• Exponential term. 4e<sup>−</sup> $^ { - \beta m } < 4 e ^ { - \beta n / 4 }$

Therefore,

$$
\mathbb { E } \Big [ \mathrm { e r } _ { \mathcal { D } } ( \widehat { h } ) \Big ] \leq \left( 1 + \frac { \lambda } { 4 } \right) \mathrm { e r } _ { \mathcal { D } } ( h ) + K \frac { d } { \lambda n } + 4 e ^ { - \beta n / 4 } .\tag{10}
$$

Remove the comparator. The learner does not depend on $h ,$ , and (10) holds for every $h \in \mathcal H$ with $\mathrm { e r } _ { \mathcal { D } } ( h ) \leq \rho$ . Taking such h with $\mathrm { e r } _ { \mathcal { D } } ( h )$ arbitrarily close to $\tau _ { D }$ gives

$$
\mathbb { E } \Big [ \mathrm { e r } _ { \mathcal { D } } \big ( \widehat { h } \big ) \Big ] \leq ( 1 + \lambda ) \tau _ { \mathcal { D } } + K \frac { d } { \lambda n } + 4 e ^ { - \beta n / 4 } .
$$

Case 2: $\tau _ { \mathscr D } > \rho / 2$ . Now $\tau _ { D }$ is bounded below by a constant, so an additive error that is a small constant can be charged to $\lambda \tau _ { D }$ . Since $\mathrm { u V C E } ( \mathscr { H } ) < \infty$ we have $v = \mathrm { V C } ( \mathcal { H } ) < \infty$ by Lemma C.1. Apply Lemma 5.8 to the ERM expert $g _ { \star }$ <sub>⋆</sub> with $\begin{array} { r } { \zeta = \frac { \lambda \rho } { 6 4 } < \frac { \lambda \tau _ { D } } { 3 2 } } \end{array}$

$$
\mathbb { E } [ \mathrm { e r } _ { \mathcal { D } } ( g _ { \star } ) ] \leq \left( 1 + \frac { \lambda } { 3 2 } \right) \tau _ { \mathcal { D } } + A e ^ { - b m } .
$$

Then apply Lemma 5.7 with Q the point mass on the expert ⋆. It is fixed in advance, so it is determined by $S ^ { ( 0 ) }$ and $S ^ { ( 1 ) }$ , and $\begin{array} { r } { D _ { \mathrm { K L } } ( Q \| \pi ) = \log \frac { 1 } { \pi ( \star ) } = \log 2 } \end{array}$ . Taking expectations and using (5),

$$
\mathbb { E } \Big [ \exp ( \widehat { h } ) \Big ] \leq \left( 1 + \frac { \lambda } { 1 6 } \right) \left( 1 + \frac { \lambda } { 3 2 } \right) \tau _ { \mathscr { D } } + 2 A e ^ { - b n / 4 } + \frac { 1 2 8 \log 2 } { \lambda n } \leq ( 1 + \lambda ) \tau _ { \mathscr { D } } + K \frac { d } { \lambda n } + 2 A e ^ { - b n / 4 } .
$$

Conclude. In both cases

$$
\mathbb { E } \Big [ \mathrm { e r } _ { \mathcal { D } } ( \widehat { h } ) \Big ] - ( 1 + \lambda ) \tau _ { \mathcal { D } } \le K \frac { d } { \lambda n } + C e ^ { - c n } , \qquad C = 4 + 2 A , \quad c = \frac { \operatorname* { m i n } \{ \beta , b \} } { 4 } .
$$

The constants $\beta ,$ A and b depend on $\mathcal { D } _ { \mathcal { X } }$ (through q), on λ and on $\mathcal { H } ,$ but not on the conditional distribution of Y given X. So C and c are the same for all D with marginal $\mathcal { D } _ { \mathcal { X } }$ □

Remark 1. The result is for each fixed $\lambda > 0 ;$ the marginal-dependent remainder is not controlled uniformly as $\lambda \downarrow 0 ,$ so the theorem does not permit choosing λ as a function of n to recover an agnostic bound with uVCE(H).

## 5.4 Fine-grained lower bound

Theorem 5.9. Let $0 < \lambda \leq 1$ and let H be a concept class that has an infinite uncentered d-VC-eluder sequence for some d $\geq 1$ . There exist a marginal $\mathcal { D } _ { \mathcal { X } }$ and sample sizes $n _ { 1 } < n _ { 2 } < \cdots$ such that for every learner A and every k there is a distribution D with marginal $\mathcal { D } _ { \mathcal { X } }$ satisfying

$$
\mathbb { E } _ { S \sim \mathcal { D } ^ { n _ { k } } } [ \mathrm { e r } _ { \mathcal { D } } ( \mathcal { A } _ { S } ) ] - ( 1 + \lambda ) \tau _ { \mathcal { D } } \ge \frac { 1 } { 2 0 4 8 } \cdot \frac { d } { \lambda n _ { k } } .
$$

Idea. The idea is similar to the fine-grained lower bound for agnostic PAC learning. We spread the marginal over infinitely many shattered blocks with decaying masses. For each sample size $n _ { k }$ we look at one block that has little probability mass, so the sample contains only a few points from it. In this block we hide $d$ bits in noisy labels, in the spirit of Assouad’s lemma [Assouad, 1983, Yu, 1997]. With so few points from the block, the learner cannot figure out most of the bits, and it pays for each bit it gets wrong. The noise level is chosen proportional to $\lambda ,$ so that the price of the wrong bits is larger than the slack that the factor $( 1 + \lambda )$ gives the learner. The earlier blocks do not help, since all the concepts we use agree there, and the later blocks have too little mass to matter.

Proof. We first set up the three ingredients for the proof: the blocks, the witnesses, and the marginal.

Setup. Fix an infinite uncentered d-VC-eluder sequence $B _ { 1 } , B _ { 2 } , \ldots$ . with $B _ { k } = \{ x _ { k , 1 } , \ldots , x _ { k , d } \}$ . By Definition 3.4, for each k there is $F _ { k } \subseteq { \mathcal { H } }$ that VC-shatters $B _ { k }$ and whose members all agree with each other on the earlier blocks $B _ { 1 } \cup \dots \cup B _ { k - 1 }$ . We record three consequences.

• For each $\theta \in \{ 0 , 1 \} ^ { d }$ there is a witness $h _ { k , \theta } \in F _ { k }$ with $h _ { k , \theta } ( x _ { k , j } ) = \theta _ { j }$ for all $j \in [ d ]$

• There is a function $c _ { k }$ on $B _ { 1 } \cup \dots \cup B _ { k - 1 }$ such that $h ( x ) = c _ { k } ( x )$ for every $h \in F _ { k }$ and every x in an earlier block. In words: on earlier blocks all witnesses carry the same labels, so these labels say nothing about θ.

• The blocks are pairwise disjoint: a point of an earlier block has the same label under every member of $F _ { k }$ , so it cannot belong to the shattered set $B _ { k }$

Let $\begin{array} { r } { s = \frac { \lambda } { 1 2 8 } } \end{array}$ , and let $\mathcal { D } _ { \mathcal { X } }$ be the marginal that gives each point of $B _ { k }$ mass $\textstyle { \frac { p _ { k } } { d } }$ , where $p _ { k } = ( 1 - s ) s ^ { k - 1 }$ is the total mass of block k (note $\textstyle \sum _ { k } p _ { k } = 1 )$ . The total mass of the blocks after $B _ { k }$ , which we call the tail, is

$$
t _ { k } = \sum _ { \ell > k } p _ { \ell } = \frac { s } { 1 - s } p _ { k } .
$$

The geometric decay is what lets one marginal serve all sample sizes: block k has mass $p _ { k } ,$ , so it is the block that a sample of size about $\frac { 1 } { p _ { k } }$ has barely explored. For the kth sample size we hide d bits in block $k .$

Parameters. Take the noise level

$$
\gamma = \frac { \lambda } { 1 + 2 \lambda } , \qquad \mathrm { s o ~ t h a t } \qquad 0 < \gamma \leq \frac { 1 } { 3 } \quad \mathrm { a n d } \quad \frac { 1 } { 2 } - \gamma = \frac { 1 } { 2 ( 1 + 2 \lambda ) } ,
$$

and the sample sizes $\begin{array} { r } { n _ { k } = \left\lfloor \frac { d } { 1 2 8 p _ { k } \gamma ^ { 2 } } \right\rfloor } \end{array}$ for $k \geq 2$ . We use three facts about these choices.

1. The tail is negligible: since $\begin{array} { r } { s \le \frac { 1 } { 2 } , t _ { k } = \frac { s } { 1 - s } p _ { k } \le 2 s p _ { k } = \frac { \lambda } { 6 4 } p _ { k } } \end{array}$

2. $n _ { k }$ is not too small: for $k \geq 2$ we have $\begin{array} { r } { p _ { k } \le s = \frac { \lambda } { 1 2 8 } } \end{array}$ and $\gamma \leq \lambda \leq 1$ , so 128 $p _ { k } \gamma ^ { 2 } \leq 1 \leq d ,$ i.e., $\begin{array} { r } { \frac { d } { 1 2 8 p _ { k } \gamma ^ { 2 } } \geq 1 . \mathrm { A s } \lfloor u \rfloor \geq u / 2 } \end{array}$ for $u \geq 1$

$$
\frac { d } { 2 5 6 p _ { k } \gamma ^ { 2 } } \leq n _ { k } \leq \frac { d } { 1 2 8 p _ { k } \gamma ^ { 2 } } .\tag{11}
$$

3. The $n _ { k }$ increase: $\textstyle p _ { k + 1 } = s p _ { k } \leq { \frac { p _ { k } } { 1 2 8 } }$ , so by (11) $\begin{array} { r } { n _ { k + 1 } \geq \frac { d } { 2 5 6 p _ { k + 1 } \gamma ^ { 2 } } \geq 6 4 \cdot \frac { d } { 1 2 8 p _ { k } \gamma ^ { 2 } } \geq 6 4 n _ { k } > n _ { k } } \end{array}$ Fix a learner A and $k \geq 2$

Construction. For $\theta \in \{ 0 , 1 \} ^ { d }$ let $\mathcal { D } _ { \theta }$ be the distribution with marginal $\mathcal { D } _ { \mathcal { X } }$ in which the label y of a point x is drawn as follows:

• if x lies in an earlier block $( B _ { 1 } , \dots , B _ { k - 1 } ) \colon y = c _ { k } ( x )$

• if $\boldsymbol { x } = \boldsymbol { x } _ { k , j }$ lies in the current block: $y = \theta _ { j }$ with probability ${ \frac { 1 } { 2 } } + \gamma$ , and $y = 1 - \theta _ { j }$ otherwise;

• if x lies in a later block: $y = 0$

Only the labels in the current block depend on θ. This is the reason for labeling the tail by the constant 0 rather than by $h _ { k , \theta } { : }$ the values of $h _ { k , \theta }$ on later blocks are not controlled by the definition and could reveal θ to the learner.

The comparator is small. The witness $h _ { k , \theta }$ never makes a mistake on earlier blocks (there it equals $c _ { k } )$ . It makes a mistake with probability $\begin{array} { r l r } {  { \frac { 1 } { 2 } - \gamma } } \end{array}$ on each point of $B _ { k }$ (there it predicts $\theta _ { j } )$ , and may make a mistake on the tail, whose total mass is $t _ { k }$ . Since $h _ { k , \theta } \in \mathcal { H }$ , we get

$$
\tau _ { \mathcal { D } _ { \theta } } \leq \mathrm { e r } _ { \mathcal { D } _ { \theta } } ( h _ { k , \theta } ) \leq p _ { k } \bigg ( \frac { 1 } { 2 } - \gamma \bigg ) + t _ { k } .\tag{12}
$$

The learner’s error on the current block. Let ξ denote the internal randomness of ${ \mathcal { A } } ,$ which is independent of the sample, and let $q _ { j } ( \theta ) = \mathbb { P } [ \mathcal { A } _ { S } ( x _ { k , j } ) \neq \theta _ { j } ]$ be the probability that the learner gets bit $j$ wrong, where $S \sim \mathcal { D } _ { \theta } ^ { n _ { k } }$ and the probability is also over $\xi .$ . By construction, the label at $x _ { k , j }$ equals $\theta _ { j }$ with probability ${ \frac { 1 } { 2 } } + \gamma$ . So a classifier $g : \mathcal { X }  \mathcal { Y }$ errs at $\boldsymbol { x } _ { k , j }$ with probability ${ \frac { 1 } { 2 } } - \gamma$ if $g ( x _ { k , j } ) = \theta _ { j }$ , and with probability $\textstyle { \frac { 1 } { 2 } } + \gamma$ otherwise; in both cases the error probability at $x _ { k , j } ~ \mathrm { i s } ~ \frac { 1 } { 2 } - \gamma +$ $2 \gamma \mathbb { 1 } [ g ( x _ { k , j } ) \neq \theta _ { j } ]$ . Each point of $B _ { k }$ has mass $\textstyle { \frac { p _ { k } } { d } }$ , and the error on the other blocks is nonnegative, so $\begin{array} { r } { \operatorname { e r } _ { \mathcal { D } _ { \theta } } ( g ) \geq \sum _ { j = 1 } ^ { d } \frac { p _ { k } } { d } \left( \frac { 1 } { 2 } - \gamma + 2 \gamma { \mathbb 1 } [ g ( x _ { k , j } ) \neq \theta _ { j } ] \right) } \end{array}$ . Applying this to $g = A _ { S }$ (which need not belong to $\mathcal { H } )$ and taking the expectation over $S \sim \mathcal { D } _ { \theta } ^ { n _ { k } }$ and $\xi ,$

$$
\mathbb { E } _ { S \sim \mathcal { D } _ { \theta } ^ { n _ { k } } } [ \mathrm { e r } _ { \mathcal { D } _ { \theta } } ( \mathcal { A } _ { S } ) ] \ge \sum _ { j = 1 } ^ { d } \frac { p _ { k } } { d } \bigg ( \frac { 1 } { 2 } - \gamma + 2 \gamma q _ { j } ( \theta ) \bigg ) .\tag{13}
$$

It remains to show that the learner gets many bits wrong, i.e., that the $q _ { j } ( \theta )$ are large on average.

The learner cannot tell $\theta _ { j } = 0$ from $\theta _ { j } \ = \ 1$ . Fix $j ~ \in ~ [ d ]$ and $\theta \in \{ 0 , 1 \} ^ { d }$ , and let $\theta ^ { ( j ) }$ denote θ with its jth bit flipped. Let $P _ { \theta }$ be the joint law of $( S , \xi )$ when $S \sim \mathcal { D } _ { \theta } ^ { n _ { k } }$ , and consider the event $E = \{ ( S , \xi ) : \mathcal { A } _ { S } ( x _ { k , j } ) \neq \theta _ { j } \}$ . The learner is wrong about bit $j$ under $\mathcal { D } _ { \theta }$ exactly when $( S , \xi ) \in E$ and, because $\theta _ { j } ^ { ( j ) } = 1 - \theta _ { j }$ , it is wrong about bit $j$ under $\mathcal { D } _ { \theta ^ { ( j ) } }$ exactly when $( S , \xi ) \notin E$ . Therefore

$$
q _ { j } ( \theta ) + q _ { j } \left( \theta ^ { ( j ) } \right) = P _ { \theta } ( E ) + 1 - P _ { \theta ^ { ( j ) } } ( E ) \geq 1 - \mathrm { T V } ( P _ { \theta } , P _ { \theta ^ { ( j ) } } ) = 1 - \mathrm { T V } \left( \mathcal { D } _ { \theta } ^ { n _ { k } } , \mathcal { D } _ { \theta ^ { ( j ) } } ^ { n _ { k } } \right) ,\tag{14}
$$

where the last equality holds because $\xi$ has the same law under both and is independent of S. So it suffices to show that the total variation distance $\mathrm { T V } \Big ( \mathscr { D } _ { \theta } ^ { n _ { k } } , \mathscr { D } _ { \theta ^ { ( j ) } } ^ { n _ { k } } \Big ) = \operatorname* { s u p } _ { E ^ { \prime } } \Big | \mathscr { D } _ { \theta } ^ { n _ { k } } ( E ^ { \prime } ) - \mathscr { D } _ { \theta ^ { ( j ) } } ^ { n _ { k } } ( E ^ { \prime } ) \Big |$ is small. We bound it through the KL divergence.

The distributions $\mathcal { D } _ { \theta }$ and $\mathcal { D } _ { \theta ^ { ( j ) } }$ have the same marginal and differ only in the label at $\mathit { x } _ { k , j } \colon$ there the label equals $\theta _ { j }$ with probability ${ \frac { 1 } { 2 } } + \gamma$ under $\mathcal { D } _ { \theta }$ and with probability ${ \frac { 1 } { 2 } } - \gamma$ under $\mathcal { D } _ { \theta ^ { ( j ) } }$ . The labels on earlier and later blocks do not depend on θ at all. Since $x _ { k , j }$ has mass $\textstyle { \frac { p _ { k } } { d } }$ , the chain rule for the KL divergence gives

$$
\mathrm { D } _ { \mathrm { K L } } ( \mathcal { D } _ { \theta } \parallel \mathcal { D } _ { \theta ^ { ( j ) } } ) = \frac { p _ { k } } { d } \mathrm { D } _ { \mathrm { K L } } \bigl ( \mathrm { B e r } \bigl ( \frac { 1 } { 2 } + \gamma \bigr ) \parallel \mathrm { B e r } \bigl ( \frac { 1 } { 2 } - \gamma \bigr ) \bigr ) = \frac { 2 \gamma p _ { k } } { d } \ln \frac { 1 + 2 \gamma } { 1 - 2 \gamma } \leq \frac { 8 \gamma ^ { 2 } } { 1 - 4 \gamma ^ { 2 } } \frac { p _ { k } } { d } \leq \frac { 1 6 \gamma ^ { 2 } p _ { k } } { d } ,
$$

where we used ln $\begin{array} { r } { \frac { 1 + u } { 1 - u } \le \frac { 2 u } { 1 - u ^ { 2 } } } \end{array}$ with $u = 2 \gamma$ , and $1 - 4 \gamma ^ { 2 } \geq \frac { 5 } { 9 } \geq \frac { 1 } { 2 }$ since $\begin{array} { r } { \gamma \le \frac { 1 } { 3 } } \end{array}$ . The KL divergence is additive over the $n _ { k }$ i.i.d. samples, so by the upper bound in $\begin{array} { r } { ( 1 1 ) , \mathrm { D } _ { \mathrm { K L } } \Big ( \mathcal { D } _ { \theta } ^ { n _ { k } } \parallel \mathcal { D } _ { \theta ^ { ( j ) } } ^ { n _ { k } } \Big ) \leq \frac { 1 6 n _ { k } p _ { k } \gamma ^ { 2 } } { d } \leq } \end{array}$ $\frac { 1 } { 8 }$ . By Pinsker’s inequality, TV $\begin{array} { r } { { ' } \Big ( \mathscr { D } _ { \theta } ^ { n _ { k } } , \mathscr { D } _ { \theta ^ { ( j ) } } ^ { n _ { k } } \Big ) \leq \sqrt { \frac { 1 } { 2 } \operatorname { D } _ { \mathrm { K L } } \Big ( \mathscr { D } _ { \theta } ^ { n _ { k } } \left. \mathscr { D } _ { \theta ^ { ( j ) } } ^ { n _ { k } } \right) } \leq \sqrt { \frac { 1 } { 1 6 } } = \frac { 1 } { 4 } } \end{array}$ , and therefore $\begin{array} { r } { q _ { j } ( \theta ) + q _ { j } \left( \theta ^ { ( j ) } \right) \geq \frac { 3 } { 4 } } \end{array}$ by (14).

Average over θ. The vectors $\theta \in \{ 0 , 1 \} ^ { d }$ split into $2 ^ { d - 1 }$ pairs $\left\{ \theta , \theta ^ { \left( j \right) } \right\}$ , and by the previous step each pair contributes at least $\frac 3 4$ . Hence

$$
{ \frac { 1 } { 2 ^ { d } } } \sum _ { \theta \in \{ 0 , 1 \} ^ { d } } q _ { j } ( \theta ) = { \frac { 1 } { 2 ^ { d } } } \sum _ { \theta : \theta _ { j } = 0 } \left( q _ { j } ( \theta ) + q _ { j } { \Big ( } \theta ^ { ( j ) } { \Big ) } \right) \geq { \frac { 1 } { 2 ^ { d } } } \cdot 2 ^ { d - 1 } \cdot { \frac { 3 } { 4 } } = { \frac { 3 } { 8 } } \quad { \mathrm { f o r ~ e v e r y ~ } } j \in [ d ] .
$$

Averaging (13) over a uniformly random θ therefore gives

$$
\frac { 1 } { 2 ^ { d } } \sum _ { \theta } \mathbb { E } _ { S \sim \mathcal { D } _ { \theta } ^ { n _ { k } } } [ \mathrm { e r } _ { \mathcal { D } _ { \theta } } ( A _ { S } ) ] \geq p _ { k } \bigg ( \frac { 1 } { 2 } - \gamma \bigg ) + 2 \gamma p _ { k } \cdot \frac { 3 } { 8 } = p _ { k } \bigg ( \frac { 1 } { 2 } - \gamma \bigg ) + \frac { 3 } { 4 } \gamma p _ { k } .\tag{15}
$$

Subtract the relaxed comparator. Subtracting $( 1 + \lambda )$ times (12) from (15), the term $p _ { k } { \bigl ( } { \frac { 1 } { 2 } } - \gamma { \bigr ) }$ cancels once and

$$
\frac { 1 } { 2 ^ { d } } \sum _ { \theta } \Bigl ( \mathbb { E } _ { S \sim \mathcal { D } _ { \theta } ^ { n _ { k } } } [ \mathrm { e r } _ { \mathcal { D } _ { \theta } } ( A _ { S } ) ] - ( 1 + \lambda ) \tau _ { \mathcal { D } _ { \theta } } \Bigr ) \ge \underbrace { \frac { 3 } { 4 } \gamma p _ { k } } _ { \mathrm { w r o n g ~ b i s } } - \underbrace { \lambda \biggl ( \frac { 1 } { 2 } - \gamma \biggr ) p _ { k } } _ { \mathrm { s l a c k ~ o n ~ t h e ~ b l o c k } } - \underbrace { ( 1 + \lambda ) t _ { k } } _ { \mathrm { t a l } } .
$$

This is where the choice of $\gamma$ is used. Since $\begin{array} { r } { \gamma = \frac { \lambda } { 1 + 2 \lambda } } \end{array}$ and $\begin{array} { r } { \frac { 1 } { 2 } - \gamma = \frac { 1 } { 2 ( 1 + 2 \lambda ) } } \end{array}$ , the wrong bits beat the slack:

$$
\frac { 3 } { 4 } \gamma p _ { k } - \lambda \bigg ( \frac { 1 } { 2 } - \gamma \bigg ) p _ { k } = \frac { 3 \lambda p _ { k } } { 4 ( 1 + 2 \lambda ) } - \frac { \lambda p _ { k } } { 2 ( 1 + 2 \lambda ) } = \frac { \lambda p _ { k } } { 4 ( 1 + 2 \lambda ) } .
$$

For the tail, $1 + \lambda \leq 2$ and $\begin{array} { r } { t _ { k } \le \frac { \lambda } { 6 4 } p _ { k } } \end{array}$ give $\begin{array} { r } { ( 1 + \lambda ) t _ { k } \le \frac { \lambda p _ { k } } { 3 2 } \le \frac { \lambda p _ { k } } { 8 ( 1 + 2 \lambda ) } } \end{array}$ , using $1 + 2 \lambda \leq 3 \leq 4$ . Hence the average excess error is at least

$$
\frac { \lambda p _ { k } } { 4 ( 1 + 2 \lambda ) } - \frac { \lambda p _ { k } } { 8 ( 1 + 2 \lambda ) } = \frac { \lambda p _ { k } } { 8 ( 1 + 2 \lambda ) } .
$$

Finally, the lower bound in (11) gives $\begin{array} { r } { p _ { k } \ge \frac { d } { 2 5 6 \gamma ^ { 2 } n _ { k } } = \frac { d ( 1 + 2 \lambda ) ^ { 2 } } { 2 5 6 \lambda ^ { 2 } n _ { k } } } \end{array}$ , so

$$
\frac { \lambda p _ { k } } { 8 ( 1 + 2 \lambda ) } \geq \frac { d \left( 1 + 2 \lambda \right) } { 2 0 4 8 \lambda n _ { k } } \geq \frac { d } { 2 0 4 8 \lambda n _ { k } } .
$$

Some $\theta$ attains at least the average; take $\mathcal { D } = \mathcal { D } _ { \theta }$

The sample sizes $n _ { k }$ are defined and increasing for $k \geq 2$ . Relabeling them as $n _ { 1 } < n _ { 2 } < \cdots$ gives the statement of the theorem. □

## 6 Conclusion

We studied marginal-nonuniform learning in the agnostic case, for binary classification, at the level of both rates and fine-grained rates. In the standard agnostic setting, allowing the constants to depend on the marginal does not help: a class is learnable at rate $\scriptstyle { \frac { 1 } { \sqrt { n } } }$ exactly when its VC dimension is finite, and the fine-grained rate is $\sqrt { \mathrm { V C } ( \mathcal { H } ) / n }$ , as in uniform PAC learning, with a lower bound attained by a single marginal. In the relaxed setting, where the learner competes with $\begin{array} { r } { ( 1 + \lambda ) \operatorname* { i n f } _ { h \in \mathcal { H } } \operatorname { e r } _ { \mathcal { D } } ( h ) } \end{array}$ , the picture of the realizable case returns. The rates form the same trichotomy as in the realizable case: exponential for finite classes, linear for infinite classes of finite VC dimension, and arbitrarily slow otherwise. The fine-grained rate is of order $\frac { d } { \lambda n }$ , where d is the uncentered VC-eluder dimension, and the marginal only affects an exponentially small term. So the slow rate of the agnostic setting comes only from competing with the best concept exactly: any constant multiplicative slack restores the realizable rates, and even makes finite classes learnable at an exponential rate.

We expect the ideas of this paper to extend to multiclass classification with a bounded label space. The lower bounds only use the labels of a shattered set, and the upper bound is built from three ingredients that are not specific to two labels: the exhaustion lemma, which Brukhim et al. [2026] already prove in the multiclass setting; the version spaces of clean subsamples; and aggregation by exponential weights. With the VC dimension replaced by a suitable multiclass dimension and a multiclass first-order learner in place of the binary one, we believe the same arguments go through, with constants that depend on the number of labels. The main open problem is the case of an infinite label space, where two dimensions play a role [Cohen et al., 2026].

## References

Julian Asilis, Mikael Møller Høgsgaard, and Grigoris Velegkas. On agnostic PAC learning in the small error regime. In Advances in Neural Information Processing Systems, volume 38, 2025.

Patrice Assouad. Deux remarques sur l’estimation. Comptes Rendus de l’Académie des Sciences de Paris, Série I, 296:1021–1024, 1983.

Pranjal Awasthi, Maria Florina Balcan, and Philip M. Long. The power of localization for efficiently learning linear separators with noise. Journal of the ACM, 63(6):50:1–50:27, 2017.

Shai Ben-David, Gyora M. Benedek, and Yishay Mansour. A parameterization scheme for classifying models of PAC learnability. Inf. Comput., 120(1):11–21, July 1995. ISSN 0890-5401. doi: 10.1006/ inco.1995.1094. URL https://doi.org/10.1006/inco.1995.1094.

Shai Ben-David, Tyler Lu, and Dávid Pál. Does unlabeled data provably help? Worst-case analysis of the sample complexity of semi-supervised learning. In Proceedings ofthe 21st Annual Conference on Learning Theory (COLT), pages 33–44, 2008.

Gyora M. Benedek and Alon Itai. Learnability with respect to fixed distributions. Theoretical Computer Science, 86(2):377–389, 1991.

Gyora M. Benedek and Alon Itai. Nonuniform learnability. Journal of Computer and System Sciences, 48(2):311–323, 1994.

Anselm Blumer, Andrzej Ehrenfeucht, David Haussler, and Manfred K. Warmuth. Occam’s razor. Information Processing Letters, 24(6):377–380, 1987.

Olivier Bousquet, Steve Hanneke, Shay Moran, Ramon van Handel, and Amir Yehudayoff. A theory of universal learning. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing (STOC), pages 532–541, 2021.

Olivier Bousquet, Steve Hanneke, Shay Moran, Jonathan Shafer, and Ilya Tolstikhin. Fine-grained distribution-dependent learning curves. In Proceedings of the 36th Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 5890–5924, 2023.

Nataly Brukhim, Steve Hanneke, Amirreza Shaeiri, and Maximilian Thiessen. Marginal-nonuniform multiclass learning. In Advances in Neural Information Processing Systems, 2026.

Nicolò Cesa-Bianchi, Alex Conconi, and Claudio Gentile. On the generalization ability of on-line learning algorithms. IEEE Transactions on Information Theory, 50(9):2050–2057, 2004.

Alon Cohen, Liad Erez, Steve Hanneke, Tomer Koren, Yishay Mansour, Shay Moran, and Qian Zhang. Sample complexity of agnostic multiclass classification: Natarajan dimension strikes back. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing, STOC ’26, page 722–733, New York, NY, USA, 2026. Association for Computing Machinery. ISBN 9798400725364. doi: 10.1145/3798129.3800787. URL https://doi.org/10.1145/3798129.3800787.

Amit Daniely. A PTAS for agnostically learning halfspaces. In Proceedings ofThe 28th Conference on Learning Theory, volume 40 of Proceedings ofMachine Learning Research, pages 484–502. PMLR, 2015.

Malte Darnstädt, Hans Ulrich Simon, and Balázs Szörényi. Unlabeled data does provably help. In 30th International Symposium on Theoretical Aspects ofComputer Science (STACS), volume 20 of Leibniz International Proceedings in Informatics (LIPIcs), pages 185–196, 2013.

Ofir David, Shay Moran, and Amir Yehudayoff. Supervised learning through the lens of compression. In Advances in Neural Information Processing Systems, volume 29, 2016.

Luc Devroye, László Györfi, and Gábor Lugosi. A Probabilistic Theory of Pattern Recognition, volume 31 of Applications of Mathematics. Springer, 1996.

Yoav Freund and Robert E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal ofComputer and System Sciences, 55(1):119–139, 1997.

Christina Göpfert, Shai Ben-David, Olivier Bousquet, Sylvain Gelly, Ilya Tolstikhin, and Ruth Urner. When can unlabeled data improve the learning rate? In Proceedings ofthe Thirty-Second Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 1500–1518. PMLR, 2019.

Steve Hanneke and Shay Moran. A theory of universal agnostic learning. arXiv preprint arXiv:2601.20961, 2026.

Steve Hanneke and Mingyue Xu. Universal rates of empirical risk minimization. In Advances in Neural Information Processing Systems, volume 37, 2024.

Steve Hanneke and Mingyue Xu. Universal rates of ERM for agnostic learning. In Proceedings of the Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 2666–2703. PMLR, 2025.

Steve Hanneke, Kasper Green Larsen, and Nikita Zhivotovskiy. Revisiting agnostic PAC learning. In Proceedings of the 65th IEEE Annual Symposium on Foundations of Computer Science (FOCS), 2024.

Steve Hanneke, Qinglin Meng, and Amirreza Shaeiri. Representation preserving multiclass agnostic to realizable reduction. In ICML, 2025a.

Steve Hanneke, Shay Moran, and Maximilian Thiessen. Marginal-nonuniform PAC learnability. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b. URL https: //openreview.net/forum?id=aoVCFtox89.

Steve Hanneke, Amirreza Shaeiri, and Hongao Wang. Non-uniform multiclass learning with bandit feedback. In Advances in Neural Information Processing Systems, volume 38, 2025c.

David Haussler. Decision theoretic generalizations of the PAC model for neural net and other learning applications. Information and Computation, 100(1):78–150, 1992. ISSN 0890-5401. doi: 10.1016/ 0890-5401(92)90010-d. URL http://dx.doi.org/10.1016/0890-5401(92)90010-D.

Max Hopkins, Daniel M. Kane, Shachar Lovett, and Gaurav Mahajan. Realizable learning is all you need. In Proceedings of the 35th Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, 2022.

Michael J. Kearns, Robert E. Schapire, and Linda M. Sellie. Toward efficient agnostic learning. Machine Learning, 17(2–3):115–141, 1994.

Guillaume Lecué and Shahar Mendelson. General nonexact oracle inequalities for classes with a subexponential envelope. The Annals ofStatistics, 40(2):832–860, 2012.

Nick Littlestone and Manfred K. Warmuth. The weighted majority algorithm. Information and Computation, 108(2):212–261, 1994.

Gábor Lugosi and Kenneth Zeger. Concept learning using complexity regularization. IEEE Transactions on Information Theory, 42(1):48–54, 1996.

Markus Engelund Mathiasen, Jian Qian, and Nikita Zhivotovskiy. An optimal agnostic PAC algorithm, 2026. URL https://arxiv.org/abs/2608.06363.

Jorma Rissanen. Modeling by shortest data description. Automatica, 14(5):465–471, 1978.

Shai Shalev-Shwartz and Shai Ben-David. Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press, New York, NY, USA, 2014. ISBN 978-1-107-05713-5.

Nathan Srebro, Karthik Sridharan, and Ambuj Tewari. Smoothness, low noise and fast rates. In Advances in Neural Information Processing Systems, volume 23, pages 2199–2207, 2010.

L. G. Valiant. A theory of the learnable. Commun. ACM, 27(11):1134–1142, November 1984. ISSN 0001-0782. doi: 10.1145/1968.1972. URL https://doi.org/10.1145/1968.1972.

V. N. Vapnik and A. Ya. Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. Theory of Probability & Its Applications, 16(2):264–280, 1971. doi: 10.1137/ 1116025. URL https://doi.org/10.1137/1116025.

Vladimir N. Vapnik. Statistical Learning Theory. Wiley-Interscience, 1998.

Bin Yu. Assouad, Fano, and Le Cam. In Festschrift for Lucien Le Cam, pages 423–435. Springer, 1997.

## A Existence of the first-order learner

Our upper bounds run a learner inside a version space of $\mathcal { H } ,$ and they need a guarantee whose lowerorder term scales with the complexity of that version space. Known first-order guarantees for agnostic PAC learning [Hanneke et al., 2024, Mathiasen et al., 2026] assume that the class has VC dimension at most k. This is not what we have. The version spaces used in section 5.3 are only simple in the distributional sense, $\mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } ) \le k$ , and their VC dimension can be much larger than $k .$ For instance, let $\mathcal { X } = \{ z _ { 1 } , . . . , z _ { M } \} \cup \mathbb { N }$ , and let H consist of all labelings of $z _ { 1 } , \dots , z _ { M }$ that are 0 on N, together with the indicator functions $\mathbb { 1 } _ { \{ x \} }$ of the points $x \in \mathbb { N }$ . Then $\mathrm { V C } ( \mathcal { H } ) = M$ while $\operatorname { u V C E } ( \mathcal { H } ) = 1 . \operatorname { I f } \mathcal { D } \chi$ puts no mass on the points $z _ { j }$ , then no sample contains them, so every version space still VC-shatters $\{ z _ { 1 } , \dotsc , z _ { M } \}$ . Every version space thus has VC dimension M, while its distributional VC dimension is at most 1. Applying a first-order guarantee with k = M would give a bound of order $\frac { \mathrm { V C } ( \mathcal { H } ) } { \lambda n }$ instead of $\frac { \mathrm { u V C E } ( \mathcal { H } ) } { \lambda n }$ . Moreover, the learner does not know $\mathcal { D } _ { \mathcal { X } }$ , so it cannot discard the part of the class that lives on a set of probability zero. We therefore need a first-order learner whose guarantee depends only on the distributional VC dimension. Definition 3.6 asks for such a learner, and Lemma A.1 shows that it exists. The point is that a shattered set of probability zero almost surely never appears among the sample points or the test point, so a learner whose guarantee depends only on the points it sees is not affected by it.

Lemma A.1. Afirst-order learner in the sense ofDefinition 3.6 exists.

Proof. We use a rule $\boldsymbol { B }$ with the following leave-one-out guarantee [Mathiasen et al., 2026]: there are universal constants $c _ { 1 } , c _ { 2 }$ such that for every N, every labeled sequence $z _ { 1 } , \dots , z _ { N }$ with $z _ { i } = ( x _ { i } , y _ { i } )$ and every class V whose trace on $( x _ { 1 } , \ldots , x _ { N } )$ has VC dimension at most $k ,$

$$
\frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { P } [ \mathcal { B } _ { \mathcal { V } } ( z _ { - i } ) ( x _ { i } ) \neq y _ { i } ] \leq L ^ { * } + c _ { 1 } \sqrt { \frac { L ^ { * } k } { N } } + c _ { 2 } \frac { k } { N } , \qquad L ^ { * } = \operatorname* { m i n } _ { h \in \mathcal { V } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } [ h ( x _ { i } ) \neq y _ { i } ] ,
$$

where $z _ { - i }$ is the sequence with $z _ { i }$ removed and the probability is over the randomness of $B .$

Now let $S \sim \mathcal { D } ^ { n }$ and draw a fresh $( X , Y ) \sim { \mathcal { D } }$ , so that $z _ { 1 } , \ldots , z _ { n + 1 }$ are i.i.d. from $\mathcal { D } _ { \ell }$

The trace condition holds almost surely. If $\begin{array} { r } { \mathrm { V C } _ { \mathcal { D } _ { \mathcal { X } } } ( \mathcal { V } ) \leq k . } \end{array}$ , then each of the $\textstyle { \binom { n + 1 } { k + 1 } }$ sub-tuples of size $k + 1$ of $\left( x _ { 1 } , \ldots , x _ { n + 1 } \right)$ is shattered by $\nu$ with probability 0, so the trace of $\nu$ on these points has VC dimension at most k almost surely.

Exchangeability. Since the $z _ { i }$ are i.i.d., each term of the leave-one-out average has the same expectation as the error of $B _ { \nu } ( S )$ on the fresh point, so $\mathbb { E } [ \exp ( B \nu ( S ) ) ]$ equals the expected left-hand side above with $N = n + 1$

Jensen. Taking expectations and using concavity of the square root,

$$
\mathbb { E } [ \exp ( \mathcal { B } _ { \mathcal { V } } ( S ) ) ] \leq \mathbb { E } [ L ^ { * } ] + c _ { 1 } \sqrt { \frac { \mathbb { E } [ L ^ { * } ] k } { n + 1 } } + c _ { 2 } \frac { k } { n + 1 } .
$$

Finally, for every $h \in \mathcal V$ we have $\mathbb { E } [ L ^ { * } ] \ \leq \ \mathrm { e r } _ { \mathcal { D } } ( h )$ , hence $\mathbb { E } [ L ^ { * } ] \le \tau _ { \mathcal { D } , \mathcal { V } }$ . This gives (1) with $B =$ max $\{ c _ { 1 } , c _ { 2 } , 1 \}$ □

Remark 2. Hanneke et al. [2024] were thefirst to obtain afirst-order bound with the optimal $\sqrt { \tau _ { D , \mathcal { V } } k / n }$ term, with an additional polylogarithmicfactor in the lower-order term. Using their learner in place of $B ,$ the same argument gives $\begin{array} { r } { \mathbb { E } [ \mathrm { e r } _ { \mathcal { D } } ( A _ { \lambda , S } ) ] - ( 1 + \lambda ) \tau _ { \mathcal { D } } \leq K \frac { d \mathrm { p o l y l o g } ( n / d ) } { \lambda n } + C e ^ { - c n } } \end{array}$ . The optimal learner of Mathiasen et al. [2026] removes the logarithmic factor, which is what gives the exact fine-grained rate $\frac { d } { \lambda n }$ and the linear rate in Theorem 5.1.

## B Measurability

As in Bousquet et al. [2021] and Hanneke et al. [2025b], we assume that X is a Polish space with its Borel σ-algebra and that H is measurable in the following sense: there are a Polish space Θ and a Borel map $( \theta , x ) \mapsto h _ { \theta } ( x )$ such that $\mathcal { H } = \{ h _ { \theta } : \theta \in \Theta \}$ . Under this assumption, every event in this paper that is defined by an existential quantifier over H (for example, that a tuple of points is VC-shattered by a version space) is an analytic set, hence universally measurable, so all the probabilities we write are well defined. We also assume that the learning rules we use, namely ERM and the first-order learner, are universally measurable functions of the data; for ERM such a choice exists by the Jankov–von Neumann uniformization theorem. If H is countable, all of this holds trivially.

## C The uncentered VC-eluder dimension

In this appendix we collect the properties of uVCE that we use, and we show that for binary classification it coincides with the VC-eluder dimension of Hanneke and Xu [2024], which characterizes the realizable marginal-nonuniform rate [Hanneke et al., 2025b]. All statements here are purely combinatorial.

We first recall the centered notion. Following Hanneke and Xu [2024], an infinite d-VC-eluder sequence for H is a sequence of labeled points $( x _ { 1 } , y _ { 1 } ) , ( x _ { 2 } , y _ { 2 } ) , . . .$ such that every finite prefix is realizable by H and, for every $k \geq 1$ , the block $\{ x _ { ( k - 1 ) d + 1 } , \ldots , x _ { k d } \}$ is VC-shattered by the version space $\{ h \in { \mathcal { H } } : h ( x _ { i } ) = y _ { i }$ for all $i \leq ( k - 1 ) d \}$ The VC-eluder dimension ${ \mathrm { V C E } } ( { \mathcal { H } } )$ is the largest d for which such a sequence exists, with $\mathrm { V C E } ( \mathscr { H } ) = 0$ if there is none for $d = 1$ and $\mathrm { V C E } ( \mathcal { H } ) = \infty$ if there is one for every d.

The two notions differ in what the witnesses must agree on. In the centered version, all concepts that shatter a block agree with one fixed labeling of the earlier blocks. In the uncentered version (Definition 3.4), the concepts that shatter block k only need to agree with each other on the earlier blocks, and different blocks may use different labelings of the past. For binary labels this extra freedom does not help.

Proposition 1. For every concept class $\mathcal { H } \subseteq \{ 0 , 1 \} ^ { \mathcal { X } }$ and every $d \geq 1$ , H has an infinite uncentered d-VC-eluder sequence ifand only ifit has an infinite d-VC-eluder sequence. In particular, $\operatorname { u V C E } ( \mathcal { H } ) =$ VCE(H).

Proof. Centered implies uncentered. Given an infinite d-VC-eluder sequence, let $B _ { k }$ be its kth block and $\mathcal { F } _ { k }$ the version space of the labeled points before $B _ { k }$ . Then $\mathcal { F } _ { k }$ shatters $B _ { k }$ , and all its members agree with the labels $y _ { i }$ on the earlier blocks, so in particular they agree with each other there. Hence $B _ { 1 } , B _ { 2 } , \ldots$ . is an infinite uncentered d-VC-eluder sequence.

Uncentered implies centered. Let $B _ { 1 } , B _ { 2 } , \ldots$ . be an infinite uncentered d-VC-eluder sequence. For each k let $\mathcal { F } _ { k } \subseteq \mathcal { H }$ shatter $B _ { k }$ with all members agreeing with each other on $B _ { 1 } \cup \dots \cup B _ { k - 1 }$ , and let $c _ { k }$ be their common labeling there. We pass to a subsequence along which these labelings are consistent. Let $N _ { 0 } = \mathbb { N }$ . Given an infinite set $N _ { j - 1 } \subseteq \mathbb { N } _ { }$ , let $k _ { j } =$ min $N _ { j - 1 }$ . For every $k \in N _ { j - 1 }$ with $k > k _ { j }$ , the restriction of $c _ { k }$ to $B _ { k _ { j } }$ is one of the $2 ^ { d }$ labelings of $B _ { k _ { i } }$ , so some labeling $\sigma _ { j }$ occurs for infinitely many such $k .$ . Let $N _ { j }$ be the set of these $k ;$ it is infinite, and $\dot { N } _ { j } \subseteq N _ { j - 1 }$ . Since $k _ { j } \in N _ { j - 1 } \subseteq N _ { i }$ for all $i < j ,$ we have

$$
c _ { k _ { j } } = \sigma _ { i } \mathrm { o n } B _ { k _ { i } } \qquad \mathrm { f o r } \mathrm { a l l } i < j .\tag{16}
$$

Now list the blocks $B _ { k _ { 1 } } , B _ { k _ { 2 } } , . . .$ . in this order, labeling the points of $B _ { k _ { i } }$ by $\sigma _ { i }$ . We check that this is an infinite d-VC-eluder sequence.

• Every block is shattered by the current version space. Let $\nu _ { j }$ be the set of concepts that equal $\sigma _ { i }$ on $B _ { k _ { i } }$ for all $i < j$ . By (16), every member of $\mathcal { F } _ { k _ { j } }$ lies in $\nu _ { j }$ , and $\mathcal { F } _ { k _ { j } }$ shatters $B _ { k _ { j } }$ . Hence $\nu _ { j }$ shatters $B _ { k _ { j } }$

• Everyfinite prefix is realizable. Any member of $\mathcal { F } _ { k _ { j + 1 } }$ equals $\sigma _ { i }$ on $B _ { k _ { i } }$ for all $i \leq j ,$ by (16) with $j + 1$ in place of $j .$ So the labeled blocks $B _ { k _ { 1 } } , \ldots , B _ { k _ { j } }$ , and hence every prefix of the sequence, are realizable by H.

Both notions are monotone in d (dropping one point from every block gives a sequence of width $d - 1 )$ ), so the equivalence for every d gives $\mathrm { u V C E } ( \mathcal { H } ) = \mathrm { V C E } ( \mathcal { H } )$ , including the cases 0 and $\infty$ □

The only place where binary labels enter is the pigeonhole step: a block of size d has finitely many labelings. The same argument works for any finite label space, with $| { \mathcal { V } } | ^ { d }$ in place of $2 ^ { d }$ , but not for an infinite label space. This is why we state our results with the uncentered notion, which is also the form used in the multiclass setting [Brukhim et al., 2026], and which is the form our proofs use directly (Lemma 5.5 and Theorem 5.9).

Remark 3. By Proposition 1, the fine-grained rate $\frac { \mathrm { u V C E } ( \mathcal { H } ) } { \lambda n }$ of Theorem 5.3 equals $\frac { \mathrm { V C E } ( \mathcal { H } ) } { \lambda n }$ . The realizablefine-grained rate is $\frac { \mathrm { V C E } ( \mathcal { H } ) } { n }$ [Hanneke et al., 2025b]. So under the relaxed criterion, the agnostic fine-grained rate is governed by the same dimension as the realizable one, and only an additionalfactor $\frac { 1 } { \lambda }$ is paid.

We next relate uVCE to the size and the VC dimension of the class.

Lemma C.1. For every concept class H:

1. $\operatorname { u V C E } ( \mathcal { H } ) \leq \operatorname { V C } ( \mathcal { H } )$

2. $\mathrm { u V C E } ( \mathscr { H } ) = 0$ if and only if H is finite;

3. $\operatorname { u V C E } ( \mathcal { H } ) < \infty$ if and only $i f \mathrm { V C } ( \mathscr { H } ) < \infty$

Proof. 1. The first block of an infinite uncentered d-VC-eluder sequence is a VC-shattered set of size $d .$

2, finite classes. Suppose H has an infinite uncentered 1-VC-eluder sequence $B _ { 1 } , B _ { 2 } , \ldots$ . with witnesses $\mathcal { F } _ { k }$ . Since $\mathcal { F } _ { k }$ shatters $B _ { k }$ , it contains two concepts $f _ { k } \neq g _ { k }$ that disagree on $B _ { k }$ . For $k < k ^ { \prime }$ , the concepts $f _ { k ^ { \prime } }$ and $g _ { k ^ { \prime } }$ agree on $B _ { k }$ , while $f _ { k }$ and $g _ { k }$ do not, so $\left\{ f _ { k } , g _ { k } \right\} \ne \left\{ f _ { k ^ { \prime } } , g _ { k ^ { \prime } } \right\}$ . Hence H contains infinitely many distinct pairs of concepts, and $\mathcal { H }$ is infinite. So a finite class has $\mathrm { u V C E } ( \mathscr { H } ) = 0$

2, infinite classes. We build an infinite uncentered 1-VC-eluder sequence. Let $\mathcal { H } _ { 0 } = \mathcal { H }$ . Given an infinite class $\mathcal { H } _ { r - 1 } \subseteq \mathcal { H }$ whose members agree with each other on the points $x _ { 1 } , \ldots , x _ { r - 1 }$ chosen so far, pick two members of $\mathcal { H } _ { r - 1 }$ and a point $x _ { r }$ where they disagree. Then $\{ x _ { r } \}$ is VC-shattered by $\mathcal { H } _ { r - 1 }$ and $x _ { r }$ is a new point, since all members of $\mathcal { H } _ { r - 1 }$ agree on the earlier points. One of the two classes $\{ h \in \mathcal { H } _ { r - 1 } : h ( x _ { r } ) = 0 \}$ and $\{ h \in \mathcal { H } _ { r - 1 } : h ( x _ { r } ) = 1 \}$ is infinite; call it $\mathcal { H } _ { r }$ . Its members agree on $x _ { 1 } , \ldots , x _ { r } , \ldots$ we can continue. The singletons $\{ x _ { 1 } \} , \{ x _ { 2 } \} , . . .$ . form an infinite uncentered 1-VC-eluder sequence.

3. If $\mathrm { V C } ( \mathcal { H } ) < \infty$ , then $\mathrm { u V C E } ( \mathscr { H } ) < \infty$ by item 1. Conversely, suppose $\mathrm { V C } ( \mathcal { H } ) = \infty$ and fix $d \geq 1$ ; we build an infinite uncentered d-VC-eluder sequence. The argument is the same as for item 2, with “infinite VC dimension” in place of “infinite”. Let $\mathcal { H } _ { 0 } \ : = \ : \mathcal { H }$ . Given $\mathcal { H } _ { r - 1 } \subseteq \mathcal { H }$ with $\mathrm { V C } ( \mathcal { H } _ { r - 1 } ) = \infty$ whose members agree with each other on $B _ { 1 } \cup \dots \cup B _ { r - 1 }$ , pick a set $B _ { r }$ of size d that is VC-shattered by $\mathcal { H } _ { r - 1 }$ . It is disjoint from the earlier blocks, since all members of $\mathcal { H } _ { r - 1 }$ agree there. Split $\mathcal { H } _ { r - 1 }$ into the $2 ^ { d }$ classes of concepts with a given labeling of $B _ { r }$ . Since the VC dimension of a union of two classes is at most the sum of their VC dimensions plus one, a finite union of classes of finite VC dimension has finite VC dimension. So one of these $2 ^ { d }$ classes has infinite VC dimension; call it $\mathcal { H } _ { r }$ . Its members agree on $B _ { 1 } \cup \cdots \cup B _ { r }$ , so we can continue. Hence $\operatorname { u V C E } ( \mathcal { H } ) \geq d$ for every $d ,$ $\mathrm { i . e . , u V C E } ( \mathcal { H } ) = \infty$ □

By Proposition 1, items 2 and 3 are also consequences of the corresponding facts for VCE [Hanneke and Xu, 2024]; we include the short proofs to keep the paper self-contained.

Finally, uVCE(H) can be much smaller than $\mathrm { V C } ( \mathscr { H } )$ , which is where the relaxed criterion gains over the standard one.

Example C.2. Let $M \geq 2 ,$ , let $\mathcal { X } = \{ z _ { 1 } , . . . , z _ { M } \} \cup \mathbb { N } ,$ , and let H consist of

• all labelings of $z _ { 1 } , \dots , z _ { M }$ that are 0 on N, and

• the indicatorfunctions $\mathbb { 1 } _ { \{ x \} }$ ofthe points $x \in \mathbb { N } .$

The set $\{ z _ { 1 } , \dots , z _ { M } \}$ is VC-shattered. No set that contains a point of N together with another point is VC-shattered, since no concept labels both points 1. Hence ${ \mathrm { V C } } ( \mathcal { H } ) = M .$ . The class is infinite, so $\mathrm { u V C E } ( \mathscr { H } ) \geq 1$ by Lemma C.1. In an infinite uncentered 2-VC-eluder sequence, every block would be a shattered pair, hence a subset of $\{ z _ { 1 } , \dots , z _ { M } \}$ , and the blocks are pairwise disjoint, so there are at most M/2 ofthem. Hence $\mathrm { u V C E } ( \mathcal { H } ) = 1 . ~ F o r$ this class, the standard agnostic fine-grained rate is $\sqrt { M / n }$ (Theorem 4.1), while the relaxed one is $\frac { 1 } { \lambda n }$ (Theorem 5.3), whatever M is. The finite part of the class only affects the marginal-dependent lower-order term.