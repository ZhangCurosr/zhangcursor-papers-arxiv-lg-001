# Anytime-valid simulation-based hypothesis testing

Patrick Forré

University of Amsterdam

p.d.forre@uva.nl

Lydia Brenner Nikhef

lbrenner@nikhef.nl

## Abstract

For a given data distribution $( X _ { t } ) _ { t \in \mathbb { N } } \stackrel { \mathrm { i . i . d . } } { \sim } Q ,$ we investigate the hypothesis testing problem: $H _ { 0 } : Q = P _ { 0 } { \mathrm { ~ v s . ~ } } H _ { 1 } : Q = P _ { 1 }$ , for two diferent model probability distributions $P _ { 0 }$ and $P _ { 1 }$ In contrast to the standard setting, where analytic densities $p _ { 0 }$ and $p _ { 1 }$ are given, here, we consider the density-free setting, where we only have access to simulations $( Z _ { t } ^ { 0 } ) _ { t \in \mathbb { N } } \stackrel { \mathrm { i . i . d . } } { \sim } P _ { 0 }$ and $( Z _ { t } ^ { 1 } ) _ { t \in \mathbb { N } } \stackrel { \mathrm { i . i . d . } } { \sim } P _ { 1 }$ . For this simulation-based hypothesis testing setting, we construct an e-test martingale, resulting in a sequential test with anytime-valid type-I error guarantees, approximate growth optimality, geometrically decaying type-II error bounds, and asymptotic power one. Most ingredients used in our constructions are variants of well known concepts. The value of this paper lies in the compact presentation of an efective, anytime-valid solution for the density-free simulation-based sequential hypothesis testing case.

## 1 Introduction

Experimental particle physics generally focuses on distinguishing between competing hypotheses about the underlying processes responsible for the observed data. Often this is done in the form of testing an alternate hypothesis $H _ { 1 }$ versus the so called Standard Model (SM) of particle physics, which acts as the null hypothesis $H _ { 0 }$ . At the Large Hadron Collider (LHC), this is done both for searches for new phenomena and in precision measurements of the Standard Model. The observed events are the result of a complex sequence of stochastic processes, particle interactions with the detector, and the reconstruction of measurable quantities. Statistical hypothesis testing then provides the framework through which theoretical predictions are tested based on actual experimental observations $x = ( x _ { n } ) _ { n \in [ N ] }$ that originated from the underlying true physical distribution $Q .$

In particle physics, the alternative hypothesis $H _ { 1 }$ usually consists only of a single model $P _ { 1 }$ that arises as an extended version of the Standard Model $P _ { 0 }$ . In such a setting, where one only has a simple null hypothesis $H _ { 0 } : Q = P _ { 0 }$ and a simple alternative $H _ { 1 } : Q = P _ { 1 }$ , the standard way of testing $H _ { 0 }$ versus $H _ { 1 }$ is through the likelihood ratio:

$$
\Lambda ( x ) : = \frac { p _ { 1 } ( x ) } { p _ { 0 } ( x ) } ,\tag{1}
$$

where $p _ { 1 }$ and $p _ { 0 }$ are the corresponding probability densities of $H _ { 1 }$ and $H _ { 0 } .$ , respectively. The reason for this lies in the Neyman-Pearson lemma, which states that, for the simple-vs-simple setting, a test based on a threshold applied to the likelihood ratio $\Lambda ( x )$ is the most powerful test at each fixed significance level $\alpha \in ( 0 , 1 ]$ , see [NP33, FC98].

For example, in the LHC experiments such as ATLAS [ATL08] and CMS [CMS08], in the search for the Higgs boson, which was discovered in 2012, see [ATL12, CMS12], a likelihood ratio was used between the null-hypothesis $H _ { 0 }$ of “no Higgs boson present” and the alternate hypothesis $H _ { 1 }$ of the “existence of a Higgs boson” in the data. In collider analyses, related likelihood-ratio and profile-likelihood constructions form the basis of widely used procedures for hypothesis testing, parameter estimation, and the treatment of nuisance parameters, see [CCGV11, CER11].

However, in realistic LHC analyses, the probability densities $p _ { 0 }$ and $p _ { 0 }$ entering the likelihood ratio Λ are rarely available in analytic or closed form. The mapping from the underlying theory parameters to the reconstructed detector-level data involves multiple stages of particle production and detector response, making direct evaluation of the high-dimensional likelihood ratio computationally dificult or intractable, see [BC20]. Monte Carlo simulation therefore plays a central role in modern collider physics. Event generators model the underlying particle interactions and subsequent evolution of the event, while detailed detector simulations model the propagation of particles through the experimental apparatus and its response. The resulting simulated events provide samples from the distributions predicted by a given physical hypothesis and allow theoretical models to be compared with data at the level at which measurements are actually performed. In consequence, simulations are used not only for estimating signal and background distributions but also for modelling detector eficiencies and acceptance, validating analysis procedures, and evaluating systematic uncertainties. The connection between likelihood-ratio optimality and simulation is particularly relevant when the likelihood itself cannot be evaluated explicitly. If independent samples can be generated under competing hypotheses, the problem of distinguishing them can instead be formulated directly in terms of the simulated distributions.

With more modern methods, such as deep learning, a classifier can be trained to distinguish samples drawn from $P _ { 0 }$ and $P _ { 1 } ,$ under appropriate conditions and with suficient capacity and training data, to learn an approximation Λ<sup>ˆ</sup> of the likelihood ratio Λ, which then can be used as a surrogate in the testing procedure and which allows to calculate p-values as a measure of significance.

However, the classical hypothesis testing approach based on p-values has further drawbacks, especially when the sample size needed for a targeted efect size is a priori unknown. The test is generally only valid for a chosen sample size fixed in advance: no “peeking”, no optional stopping or optional continuation is allowed. In addition, one can not easily aggregate evidence against the null hypothesis $H _ { 0 }$ from diferent experiments, or conduct multiple hypothesis tests at once, etc. As a remedy, the anytime-valid sequential testing procedure testing by betting was recently developed, see [SSVV11, VW21, GdHK20, WRB20, HZ21, RGVS23, RW25]. Also in this framework, the standard approaches require, in the simple-vs-simple setting, the likelihood ratio Λ, which we do not have in the density-free, simulation-based context.

In this paper, we want to address both points at once: 1.) the density-free/simulation-based hypothesis testing setting (without analytic densities $p _ { 1 } , p _ { 0 } )$ , and: 2.) building an anytime-valid sequential hypothesis testing procedure. Our construction will result in a sequential test, which comes with anytime-valid type-I error guarantees, approximate growth optimality, geometrically decaying type-II error bounds and asymptotic power one. We also show that the test statistics will implicitly approximate the likelihood ratio Λ.

## 2 Anytime-valid hypothesis testing with conditional e-variables

In this section, we start by giving an overview over the main results from the standard literature of anytimevalid hypothesis testing, see, for instance, [SSVV11, VW21, GdHK20, WRB20, HZ21, RGVS23, RW25], etc.

In the following, let Ω be a measurable space with σ-algebra B, which we consider the underlying sample space, on which all occurring random variables live. We will endow (Ω, B) with several diferent probability distributions: $Q ,$ the true distribution, $P _ { 0 }$ the standard model distribution, and $P _ { 1 }$ the alternative model distribution. Based on a (finite or infinite) sequence of data points $( X _ { t } ) _ { t \in \mathbb { N } } \sim Q$ we want to test the hypothesis if the true distribution $Q$ rather coincides with the standard model distribution $P _ { 0 }$ or with the alternative model distribution $P _ { 1 }$ :

$$
H _ { 0 } : Q = P _ { 0 } , \qquad \mathrm { v s . } \qquad H _ { 1 } : Q = P _ { 1 } .\tag{2}
$$

The safe hypothesis testing framework is based on the following concepts:

Definition 2.1 (e-variables). A non-negative measurable map $E : \Omega \to [ 0 , \infty ]$ is called an e-variable for $H _ { 0 }$ if:

$$
\mathbb { E } _ { P _ { 0 } } [ E ] \leq 1 .\tag{3}
$$

Note that the Markov inequality shows that every e-variable E for $H _ { 0 }$ induces a p-variable $\pi : = \operatorname* { m i n } ( 1 , 1 / E )$ for $H _ { 0 } { \mathrm { : } }$

$$
P _ { 0 } ( \pi \leq \alpha ) = P _ { 0 } \left( E \geq { \frac { 1 } { \alpha } } \right) \stackrel { ! } { \leq } \alpha , \qquad \forall \alpha \in ( 0 , 1 ] .\tag{4}
$$

In this sense, e-variables can be considered more restrictive, so called post-hoc, (inverses of) p-variables, see [Kon23, Grü24, RW25, CLRG26], and, they directly produce a decision rule for every confidence level $\alpha \in ( 0 , 1 ] \colon$ “Reject $H _ { 0 }$ if $\begin{array} { r } { E \geq \frac { 1 } { \alpha } . { \stackrel { \mathfrak { N } } { . } } } \end{array}$ . The inequality above shows that the type-I error is then bounded by $\alpha .$ . We remain to investigate the type-II error, the power, resp., for $H _ { 1 }$ . However, in contrast to the classical concept of power $\beta : = P _ { 1 } \left( \pi \leq c \right)$ of a test, which depends on the chosen threshold c, in the safe hypothesis testing framework, one introduces the e-power instead, which quantifies on logarithmic scale how much evidence one can expect to gather against $H _ { 0 }$ under the alternative model $P _ { 1 }$

Definition 2.2 (e-power). Let $E : \Omega \to [ 0 , \infty ]$ be an e-variable for $H _ { 0 }$ . Then its e-power for $H _ { 1 }$ is given by:

$$
\mathbb { E } _ { P _ { 1 } } [ \log E ] .\tag{5}
$$

In the simple-vs-simple hypothesis test situation, which we are considering in this paper, we can even point out the most e-powerful e-variable:

Lemma 2.3 (Neyman-Pearson lemma for the optimal e-power). Under the assumption of absolute continuity: $P _ { 1 } \ll P _ { 0 }$ , the $P _ { 0 }$ -essentially unique e-variable for $H _ { 0 }$ with maximal e-power for $H _ { 1 }$ is given by the likelihoodratio, i.e. the Radon-Nikodym derivative:

$$
E ^ { \ast } ( x ) : = \frac { d P _ { 1 } } { d P _ { 0 } } ( x ) ,\tag{6}
$$

which has the e-power:

$$
\begin{array} { r } { \mathbb { E } _ { P _ { 1 } } [ \log E ^ { * } ] \stackrel { ! } { = } \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) , } \end{array}\tag{7}
$$

the Kullback-Leibler divergence between $P _ { 1 }$ and $P _ { 0 }$ .

Since, in this paper, we are in the situation, where the ratio $\frac { d P _ { 1 } } { d P _ { 0 } }$ is not given to us in an analytic form or as an algorithmic function, we cannot directly use it in practice as our e-variable. In the next section we thus aim to construct other e-variables for $H _ { 0 }$ with an e-power close to the maximal one of $\mathrm { K L } ( P _ { 1 } \| P _ { 0 } )$ and which are only based on simulations from $P _ { 0 }$ and $P _ { 1 }$

Before we go there, first note, that the concepts of e-variable E and its e-value $E ( x )$ , in the presented form, are only restricted to either one realization $x \sim Q$ or an array of data points of fixed given length $\boldsymbol { x } = ( x _ { 1 } , \dots , x _ { t } )$ with $x _ { i } \sim Q$ . To turn this into a flexible framework where the number of data points $x _ { t } \sim Q$ can grow over time $t \to \infty$ and where we can build on information from the past, we have to introduce the following concept:

Definition 2.4 (Conditional e-variables and conditional e-power). Let ${ \mathcal { F } } \subseteq B$ be a sub-σ-algebra of $\boldsymbol { B } ^ { 1 }$ on $\Omega .$ A non-negative measurable map $E : \Omega \to [ 0 , \infty ]$ is called a conditional e-variable for $H _ { 0 }$ conditional on ${ \mathcal { F } } i f .$

$$
\begin{array} { r } { \mathbb { E } _ { P _ { 0 } } [ E | \mathcal { F } ] \le 1 \qquad P _ { 0 } - a . s . } \end{array}\tag{8}
$$

Its conditional e-power for $H _ { 1 }$ conditional on $\mathcal { F }$ is defined by:

$$
\mathbb { E } _ { P _ { 1 } } [ \log E | \mathcal { F } ] .\tag{9}
$$

Note that a conditional e-variable for $H _ { 0 }$ conditional to $\mathcal { F }$ is also a (unconditional) e-variable for $H _ { 0 }$ . More generally, we can consider a sequence of $( E _ { t } ) _ { t \in \mathbb { N } _ { 0 } }$ conditional e-variables for $H _ { 0 }$ , each constructed by using past information. Formally, we introduce:

Notations 2.5. Let $( E _ { t } ) _ { t \in \mathbb { N } _ { \mathrm { C } } }$ be a sequence of conditional e-variables for $H _ { 0 } , E _ { t } : \Omega \to [ 0 , \infty ] , E _ { 0 } : = 1$ adapted to the filtration $\mathcal { F } = ( \mathcal { F } _ { t } ) _ { t \in \mathbb { N } _ { 0 } }$ of sub-σ-algebras $\mathcal { F } _ { t } \subseteq B$

$$
\mathcal { F } _ { 0 } \subseteq \mathcal { F } _ { 1 } \subseteq \mathcal { F } _ { 2 } \subseteq \cdot \cdot \cdot \subseteq \mathcal { F } _ { t } \subseteq \cdot \cdot \cdot \subseteq B ,\tag{10}
$$

each conditional on its own past, that is, for every $t \in \mathbb { N } _ { 1 }$ the map $E _ { t } : \Omega \to [ 0 , \infty ]$ is $\mathcal { F } _ { t }$ -measurable and:

$$
\begin{array} { r } { \mathbb { E } _ { P _ { 0 } } [ E _ { t } | \mathcal { F } _ { t - 1 } ] \le 1 \qquad P _ { 0 ^ { - } } a . s . } \end{array}\tag{11}
$$

We further construct non-negative variables $W _ { t } : \Omega \to [ 0 , \infty ]$ as follows for $t \in \mathbb { N } _ { 1 }$ by multiplication:

$$
W _ { t } : = E _ { t } \cdot W _ { t - 1 } = E _ { t } \cdot E _ { t - 1 } \cdot \cdot \cdot E _ { 1 } ,
$$

$$
W _ { 0 } : = 1 .\tag{12}
$$

With these notations we have the following:

Lemma 2.6. With the Notations 2.5, each $W _ { t }$ for $t \geq 0$ is an (unconditional) e-variable for $H _ { 0 }$ and the sequence $( W _ { t } ) _ { t \in \mathbb { N } _ { 0 } }$ is a non-negative supermartingale adapted to the filtration $\mathcal { F } = ( \mathcal { F } _ { t } ) _ { t \in \mathbb { N } _ { 0 } }$ w.r.t. $P _ { 0 }$ . More concretely, $f o r t \in \mathbb { N } _ { 1 }$ we have:

$$
\begin{array} { r } { \mathbb { E } _ { P _ { 0 } } [ W _ { t } | \mathcal { F } _ { t - 1 } ] \le W _ { t - 1 } \qquad P _ { 0 ^ { - } } a . s . , } \end{array}
$$

$$
\mathbb { E } _ { P _ { 0 } } [ W _ { 0 } ] = 1 .\tag{13}
$$

The above sequence $( W _ { t } ) _ { t \in \mathbb { N } _ { 0 } }$ now allows us to define a sequential decision rule with $\alpha \in ( 0 , 1 ]$ as follows:

“Reject $H _ { 0 }$ if there ever appears any $t \in  { \mathbb { N } } _ { 0 }$ with $\begin{array} { r } { W _ { t } > \frac { 1 } { \alpha } . > } \end{array}$

Remarkably, this sequential decision rule has the following anytime-valid type-I error guarantee, provided by Ville’s inequality for non-negative supermartingales, see [Vil39] or Theorem A.1:

Theorem 2.7 (Anytime-valid type-I error bounds). With the Notations 2.5, we have the following upper bound on the anytime-valid type-I error for the above sequential test for every $\alpha \in ( 0 , 1 ]$

$$
P _ { 0 } \left( \exists t \in \mathbb { N } _ { 0 } . W _ { t } > \frac { 1 } { \alpha } \right) = P _ { 0 } \left( \operatorname* { s u p } _ { t \in \mathbb { N } _ { 0 } } W _ { t } > \frac { 1 } { \alpha } \right) \overset { ! } { \leq } \alpha .\tag{14}
$$

Under boundedness conditions on the logarithms of the $E _ { t } { ' } \mathrm { s }$ and suficiently big conditional e-power, we also get an asymptotic type-II error control with help of the Azuma–Hoefding inequality for bounded, filtration adapted sequences of random variables, see [Azu67, For26b] or Theorem A.2:

Theorem 2.8 (Geometrically decaying type-II error bounds for log-bounded conditional e-variables). With the Notations 2.5, assume that for every $t \in  { \mathbb { N } } _ { 1 }$ there are constants $a _ { t } , b _ { t } , c _ { t } \in \mathbb { R }$ such that:

$$
a _ { t } \leq \log E _ { t } \leq b _ { t } \quad P _ { 1 } \ – a . s . ,
$$

$$
c _ { t } \leq \mathbb { E } _ { P _ { 1 } } [ \log E _ { t } | \mathcal { F } _ { t - 1 } ] \quad P _ { 1 } { - } a . s . .\tag{15}
$$

Then for every $\alpha \in ( 0 , 1 ]$ and $t \in  { \mathbb { N } } _ { 1 }$ we have the following type-II error bound<sup>2</sup>:

$$
P _ { 1 } \left( W _ { t } \leq \frac { 1 } { \alpha } \right) \leq \exp \left( - \frac { 2 \left( \sum _ { n = 1 } ^ { t } c _ { n } - \log \frac { 1 } { \alpha } \right) ^ { 2 } } { \sum _ { n = 1 } ^ { t } ( b _ { n } - a _ { n } ) ^ { 2 } } \right) .\tag{16}
$$

In particular, if the bounds are the same $( a _ { t } = a , b _ { t } = b , c _ { t } = c )$ for all $t \in  { \mathbb { N } } _ { 1 }$ with positive (conditional) e-power lower bound $c > 0$ for $H _ { 1 }$ then the type-II error bound simplifies and converges to zero with asymptotically geometric rate:

$$
P _ { 1 } \left( W _ { t } \leq \frac { 1 } { \alpha } \right) \leq \exp \left( - 2 t \cdot \frac { \left( c - \frac { 1 } { t } \log \frac { 1 } { \alpha } \right) _ { + } ^ { 2 } } { ( b - a ) ^ { 2 } } \right) \stackrel { t \to \infty } { \longrightarrow } 0 .\tag{17}
$$

As a consequence, we then have asymptotic power one:

$$
P _ { 1 } \left( \operatorname* { s u p } _ { t \in \mathbb { N } _ { 0 } } W _ { t } > \frac { 1 } { \alpha } \right) = P _ { 1 } \left( \operatorname* { s u p } _ { t \in \mathbb { N } _ { 0 } } W _ { t } = \infty \right) = 1 .\tag{18}
$$

Remark 2.9 (Clipping log-unbounded e-variables). $H E$ is an e-variable for $H _ { 0 }$ , but for which log E might not be bounded, then for every $\gamma \in ( 0 , 1 ]$ we can introduce its γ-clipped version:

$$
E _ { \gamma } ( x ) : = \gamma + ( 1 - \gamma ) \cdot \operatorname* { m i n } ( E ( x ) , \gamma ^ { - 1 } ) \leq 1 + E ( x ) ,\tag{19}
$$

which is again an e-variable for $H _ { 0 }$ and for which we then have the bounds:

$$
- \log \frac { 1 } { \gamma } \leq \log E _ { \gamma } ( x ) \leq \log \frac { 1 } { \gamma } ,\tag{20}
$$

together with the pointwise convergence:

$$
\operatorname* { l i m } _ { \gamma  0 } E _ { \gamma } ( x ) = E ( x ) ,\tag{21}
$$

which is monotone non-decreasing if $E ( x ) \geq 1$ and monotone non-increasing if $E ( x ) \leq 1$ for $\gamma  0$ . This then implies the following convergences of the integrals for arbitrary measures $P .$

$$
\operatorname* { l i m } _ { \gamma \to 0 } \int E _ { \gamma } d P = \int E d P , \qquad \operatorname* { l i m } _ { \gamma \to 0 } \int \log E _ { \gamma } d P = \int \log E d P .\tag{22}
$$

The latter limit holds, whenever the integral on the right is well-defined (possibly $\pm \infty$ , but excluding the $\infty - \infty \ c a s e )$ . The convergence of the integrals can be shown with help of the monotone convergence theorem, applied on the sets $\{ E \leq 1 \}$ and $\{ E > 1 \}$ separately.

This allows us to take an e-variable E for $H _ { 0 }$ with positive e-power $\mathbb { E } _ { P _ { 1 } }$ [log $E ] > 0$ for $H _ { 1 }$ (but with possible unbounded log E), and clip it with some small $\gamma > 0 . \ I f \gamma > 0$ is suficiently small then also $E _ { \gamma }$ will have positive e-power $\mathbb { E } _ { P _ { 1 } }$ [log $E _ { \gamma } ] > 0$ for $H _ { 1 }$ and, in addition, the geometrically decaying type-II error bounds of Theorem 2.8 will hold with the bounds $\begin{array} { r } { a : = - \log \frac { 1 } { \gamma } , b : = \log \frac { 1 } { \gamma } } \end{array}$ and $c : = \mathbb { E } _ { P _ { 1 } } [ \log E _ { \gamma } ] > 0$

Remark 2.10 (Rejection time). With the Notations $2 . 5 ,$ we introduce the rejection time of the test supermartingale $( W _ { t } ) _ { t \in \mathbb { N } }$ w.r.t. $\alpha \in ( 0 , 1 ]$ :

$$
\tau _ { \alpha } : = \operatorname* { i n f } \left\{ t \in \mathbb { N } _ { 0 } \bigg | W _ { t } > \frac { 1 } { \alpha } \right\} .\tag{23}
$$

To get an intuition about the average rejection time, assume, for simplicity, that $( x _ { t } ) _ { t \in \mathbb { N } } \sim P _ { 1 }$ is i.i.d. and that we use the same e-variable at each time: $E _ { t } = E ( x _ { t } ) \ f o r \ t \in \mathbb { N } .$ . Via Wald’s equation, under integrability conditions<sup>3</sup>, see $\vert W a l 4 4 ;$ , Dur19], we then get the following close approximation for the average rejection time:

$$
\mathbb { E } _ { P _ { 1 } } [ \tau _ { \alpha } ] = \frac { \mathbb { E } _ { P _ { 1 } } \left[ \log W _ { \tau _ { \alpha } } \right] } { \mathbb { E } _ { P _ { 1 } } \left[ \log E \right] } \~ \gtrapprox \ \frac { \log \frac { 1 } { \alpha } } { \mathbb { E } _ { P _ { 1 } } \left[ \log E \right] } ,\tag{24}
$$

which can be interpreted as the average number of samples needed to obtain a rejection. The above formula shows that, if we want to minimize the average rejection time, we need to maximize the e-power of E, which by Lemma 2.3 is uniquely maximized by the likelihood ratio: $\begin{array} { r } { E ^ { * } = \frac { d P _ { 1 } } { d P _ { 0 } } } \end{array}$ , for which we then get the minimal average rejection time:

$$
\mathbb { E } _ { P _ { 1 } } [ \tau _ { \alpha } ^ { * } ] \ \gtrapprox \ \frac { \log \frac { 1 } { \alpha } } { \mathrm { K L } ( P _ { 1 } \Vert P _ { 0 } ) } .\tag{25}
$$

## 3 Construction of e-variables in the density-free, simulation-based setting

In this section, we now want to construct e-powerful e-variables E for a sequential i.i.d. hypothesis testing problem:

$$
H _ { 0 } : Q = P _ { 0 } , \qquad \mathrm { v s . } \qquad H _ { 1 } : Q = P _ { 1 } ,\tag{26}
$$

where we do not have access to the densities of $P _ { 0 }$ or $P _ { 1 }$ , or even just the likelihood ratio $\frac { d P _ { 1 } } { d P _ { 0 } }$ . Instead, we aim to construct e-variables that are constructed only from i.i.d. samples $( z _ { t } ^ { 0 } ) _ { t \in \mathbb { N } } \sim P _ { 0 }$ and $( z _ { t } ^ { 1 } ) _ { t \in \mathbb { N } } \sim P _ { 1 }$ but are close to the optimal e-variable: $\begin{array} { r } { E ^ { * } : = \frac { d P _ { 1 } } { d P _ { 0 } } } \end{array}$

To construct e-variables, consider an arbitrary function: $h : \Omega \to [ 0 , \infty ]$ with $0 < \mathbb { E } _ { P _ { 0 } } [ h ] < \infty$ . Then a simple normalization turns h into a valid e-variable for $H _ { 0 }$

$$
\tilde { h } ( z ) : = \frac { h ( z ) } { \mathbb { E } _ { P _ { 0 } } [ h ] } ,
$$

$$
\mathbb { E } _ { P _ { 0 } } [ { \tilde { h } } ] = 1 .\tag{27}
$$

The problem is that we usually cannot analytically solve the integral: $\begin{array} { r } { \mathbb { E } _ { P _ { 0 } } [ h ] = \int h d P _ { 0 } } \end{array}$ . If we instead approximate this integral with i.i.d. samples from $z _ { 1 } ^ { 0 } , . . . , z _ { K } ^ { 0 } \sim P _ { 0 }$ , we get:

$$
{ \hat { h } } _ { K } ( z ) : = { \frac { h ( z ) } { { \frac { 1 } { K } } \sum _ { k = 1 } ^ { K } h ( z _ { k } ^ { 0 } ) } } { \quad \begin{array} { l l } { { \xrightarrow [ { K  \infty } ] { K  \infty } } } & { { \quad { \hat { h } } ( z ) } } \\ { { \xrightarrow [ { \mathrm { a . s . } } ] { } } } & { { \quad { \overline { { \mathbb { E } } } } _ { P _ { 0 } } [ h ] } = { \tilde { h } } ( z ) . } \end{array} }\tag{28}
$$

However, this might not be a valid e-variable for $H _ { 0 }$ anymore for fixed finite $K \geq 1$ , as expectation and quotients don’t commute in general:

$$
\mathbb { E } \left[ \hat { h } _ { K } \right] = \mathbb { E } \left[ \frac { h ( Z ) } { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } h ( Z _ { k } ^ { 0 } ) } \right] \stackrel { \mathrm { i . g . } } { \neq } \frac { \mathbb { E } \left[ h ( Z ) \right] } { \mathbb { E } \left[ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } h ( Z _ { k } ^ { 0 } ) \right] } = 1 .\tag{29}
$$

A commonly known trick in statistics and machine learning is to turn such a quantity into a classification probability (or variants thereof), see [Qin98, BBS09, SNK<sup>+</sup>08, SKS<sup>+</sup>09, NWJ10, KSS10, GH10, SSK12, $\mathrm { G P A M ^ { + } 1 4 }$ , CPL15, CPLB16, LC18, BCLP18, OLV18, HFLM<sup>+</sup>19, POvdO<sup>+</sup>19, HBL20, DMP20, CKNH20, WRB20, KRSW21, FDF<sup>+</sup>20, FTF21, FRF23, FFTV24, $\mathrm { M C F ^ { + } 2 1 }$ , MWF22, MFWF23, DHR<sup>+</sup>22, DMF<sup>+</sup>23, PBNF24, PFRS24, LMAF26, DER26], where it is known that a maximizer of the underlying classification task is proportional to the (log) density ratio $\frac { d P _ { 1 } } { d P _ { 0 } }$ . More concretely, we simply add the nominator $h ( z )$ to (the average of) the denominator above and introduce the quantity:

$$
E ^ { ( K , h ) } ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \dots , z _ { K } ^ { 0 } ) : = \frac { h ( z ^ { 1 } ) } { \frac { 1 } { K + 1 } \left( h ( z ^ { 1 } ) + \sum _ { k = 1 } ^ { K } h ( z _ { k } ^ { 0 } ) \right) } .\tag{30}
$$

If $H _ { 0 } : Q = P _ { 0 }$ is true, then $z ^ { 1 } \sim P _ { 0 }$ together with $z _ { 1 } ^ { 0 } , . . . , z _ { K } ^ { 0 } \sim P _ { 0 }$ i.i.d. of the same distribution $P _ { 0 }$ . This joint i.i.d. assumption then implies the same distributions for each of the ratios:

$$
{ \frac { h ( z ^ { 1 } ) } { h ( z ^ { 1 } ) + h ( z _ { 1 } ^ { 0 } ) + \cdots + h ( z _ { k } ^ { 0 } ) } } \sim { \frac { h ( z _ { 1 } ^ { 0 } ) } { h ( z ^ { 1 } ) + h ( z _ { 1 } ^ { 0 } ) + \cdots + h ( z _ { k } ^ { 0 } ) } } \sim \cdots \sim { \frac { h ( z _ { K } ^ { 0 } ) } { h ( z ^ { 1 } ) + h ( z _ { 1 } ^ { 0 } ) + \cdots + h ( z _ { k } ^ { 0 } ) } } .\tag{31}
$$

In particular, each ratio has the same expectation value:

$$
\mathbb { E } \left[ \frac { h ( Z ^ { 1 } ) } { h ( Z ^ { 1 } ) + h ( Z _ { 1 } ^ { 0 } ) + \cdots + h ( Z _ { k } ^ { 0 } ) } \right] = \cdots = \mathbb { E } \left[ \frac { h ( Z _ { K } ^ { 0 } ) } { h ( Z ^ { 1 } ) + h ( Z _ { 1 } ^ { 0 } ) + \cdots + h ( Z _ { k } ^ { 0 } ) } \right] ,\tag{32}
$$

while they all add up to one:

$$
\mathbb { E } \left[ \frac { h ( Z ^ { 1 } ) } { h ( Z ^ { 1 } ) + h ( Z _ { 1 } ^ { 0 } ) + \cdots + h ( Z _ { k } ^ { 0 } ) } \right] + \cdots + \mathbb { E } \left[ \frac { h ( Z _ { K } ^ { 0 } ) } { h ( Z ^ { 1 } ) + h ( Z _ { 1 } ^ { 0 } ) + \cdots + h ( Z _ { k } ^ { 0 } ) } \right] = 1 ,\tag{33}
$$

which thus implies:

$$
\mathbb { E } \left[ { \frac { h ( Z ^ { 1 } ) } { h ( Z ^ { 1 } ) + h ( Z _ { 1 } ^ { 0 } ) + \cdots + h ( Z _ { k } ^ { 0 } ) } } \right] = { \frac { 1 } { K + 1 } } .\tag{34}
$$

Multiplying by $( K + 1 )$ then shows that $E ^ { ( K , h ) }$ is a valid e-variable for $H _ { 0 }$ . Formally, note, that we implicitly endowed all probability distributions with the K-fold independent product of $P _ { 0 }$ in the expectation value. So, we actually investigated the equivalent hypothesis testing problem:

$$
H _ { 0 } : Q \otimes P _ { 0 } ^ { \otimes K } = P _ { 0 } \otimes P _ { 0 } ^ { \otimes K } = : \mathbb { P } _ { 0 } , \qquad { \mathrm { ~ v s . ~ } } \qquad H _ { 1 } : Q \otimes P _ { 0 } ^ { \otimes K } = P _ { 1 } \otimes P _ { 0 } ^ { \otimes K } = : \mathbb { P } _ { 1 } .\tag{35}
$$

In the rest of this section, we will mostly stay in this setting, but we might move between these diferent views without further indication.

To summarize the above reasoning, we formally state the following:

Lemma 3.1 (e-value construction). Let $K \geq 1$ and $h : \Omega \to [ 0 , \infty )$ be any map for which $0 < h ( Z ^ { 1 } )$ + $\begin{array} { r } { \sum _ { k = 1 } ^ { K } h ( Z _ { k } ^ { 0 } ) < \infty \mathbb { P } _ { 1 } } \end{array}$ -almost-surely, then the map $E ^ { ( K , h ) }$ given by:

$$
E ^ { ( K , h ) } ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \dots , z _ { K } ^ { 0 } ) : = \frac { h ( z ^ { 1 } ) } { \frac { 1 } { K + 1 } \left( h ( z ^ { 1 } ) + \sum _ { k = 1 } ^ { K } h ( z _ { k } ^ { 0 } ) \right) } ,\tag{36}
$$

is a valid e-variable for $H _ { 0 }$ with:

$$
E ^ { ( K , h ) } \ge 0 ,
$$

$$
\mathbb { E } _ { \mathbb { P } _ { 0 } } \left[ E ^ { ( K , h ) } \right] = 1 .\tag{37}
$$

We now want to investigate, which choice of h actually does optimize e-power for $H _ { 1 }$ . For this we have the following theoretical result:

Proposition 3.2 (Optimal e-power for fixed $K \geq 1$ , see Proof B.1). Let $P _ { 0 } , P _ { 1 }$ be two probability distributions on Ω with $P _ { 1 } \ll P _ { 0 }$ and $K \geq 1$ be a fixed natural number. Then consider all measurable functions $h : \Omega \to [ 0 , \infty )$ for which $\begin{array} { r } { 0 < h ( Z ^ { 1 } ) + \sum _ { k = 1 } ^ { K } h ( Z _ { k } ^ { 0 } ) < \infty \mathbb { P } _ { 1 } } \end{array}$ -almost-surely, and, the following e-power objective:

$$
J _ { K } ( h ) : = \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \log E ^ { ( K , h ) } \right] .\tag{38}
$$

Then each maximizer $h ^ { * }$ of $J _ { K }$ is $P _ { 0 }$ -almost-surely of the form:

$$
h ^ { * } ( z ) = c \cdot \frac { d P _ { 1 } } { d P _ { 0 } } ( z ) ,\tag{39}
$$

for some constant $c \in \mathbb { R } _ { > 0 }$ . Conversely, those functions maximize $J _ { K }$ (for all $K \geq 1$ simultaneously). The optimal e-variable of our given type is thus given by:

$$
E ^ { ( K , * ) } ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \dots , z _ { K } ^ { 0 } ) : = \frac { \frac { d P _ { 1 } } { d P _ { 0 } } ( z ^ { 1 } ) } { \frac { 1 } { K + 1 } \left( \frac { d P _ { 1 } } { d P _ { 0 } } ( z ^ { 1 } ) + \sum _ { k = 1 } ^ { K } \frac { d P _ { 1 } } { d P _ { 0 } } ( z _ { k } ^ { 0 } ) \right) } \quad \xrightarrow [ a . s . ] { K \to \infty } \quad \frac { d P _ { 1 } } { d P _ { 0 } } ( z ^ { 1 } ) = : E ^ { * } ( z ^ { 1 } ) ,\tag{40}
$$

which (for each fixed z<sup>1</sup>) converges $P _ { 0 } ^ { \otimes \mathbb { N } _ { 1 } }$ -almost-surely to the overall most e-powerful e-variable $E ^ { * } ( z ^ { 1 } ) =$ $\begin{array} { r } { \frac { d P _ { 1 } } { d P _ { 0 } } \left( z ^ { 1 } \right) } \end{array}$ for $H _ { 0 }$ vs. $H _ { 1 }$ , see Lemma 2.3, for $K  \infty$ . Furthermore, for the e-power of $E ^ { ( K , * ) }$ we have the following Jensen inequality:

$$
\begin{array} { r c l c l } { \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) } & { \geq } & { \mathbb { E } _ { P _ { 1 } } [ \log \cal E ^ { ( K , * ) } ] } & { \stackrel { ! } { \sum } } & { \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) - \log \left( 1 + \frac { \chi ^ { 2 } ( P _ { 1 } \| P _ { 0 } ) } { K + 1 } \right) } & { \stackrel { K \to \infty } { \longrightarrow } } & { \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) , } \end{array}\tag{41}
$$

provided $\chi ^ { 2 } ( P _ { 1 } \| P _ { 0 } ) < \infty$ $H ,$ even more, $\frac { d P _ { 1 } } { d P _ { 0 } }$ is bounded, then we have the uniform $( i n \ z ^ { 1 } ) \ P _ { 0 } ^ { \otimes \mathbb { N } _ { 1 } }$ -almost-sure convergence:

$$
\operatorname* { s u p } _ { z ^ { 1 } \in \Omega } \left| \log E ^ { ( K , * ) } ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \ldots , z _ { K } ^ { 0 } ) - \log E ^ { * } ( z ^ { 1 } ) \right| \stackrel { K \to \infty } { a . s . } \quad 0 .\tag{42}
$$

We now focus on a way to learn to approximate such optimal e-variables with simulations from $P _ { 1 }$ and $P _ { 0 }$ Notations 3.3. To approximate the most e-powerful e-variable for $H _ { 0 }$ vs. $H _ { 1 }$ , we make use of a flexible, optimizable paramterized function class $g _ { \theta } : \Omega  \mathbb { R }$ , for example, deep neural networks of some architecture, with parameters $\theta \in \Theta$ . We then put $h _ { \theta } ( z ) : = \exp ( g _ { \theta } ( z ) )$ to use inside $E ^ { ( K , h ) }$ and get the parameterized e-variables for $H _ { 0 }$ :

$$
\begin{array} { r } { E ^ { ( K , \theta ) } ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \dots , z _ { K } ^ { 0 } ) : = \frac { \exp ( g _ { \theta } ( z ^ { 1 } ) ) } { \frac { 1 } { K + 1 } \left( \exp ( g _ { \theta } ( z ^ { 1 } ) ) + \sum _ { k = 1 } ^ { K } \exp ( g _ { \theta } ( z _ { k } ^ { 0 } ) ) \right) } . } \end{array}\tag{43}
$$

We will optimize the e-power of $E ^ { ( K , \theta ) }$ by using an empirical estimate of the e-power objective $J _ { K }$ from above, using i.i.d. samples from $P _ { 0 }$ and $P _ { 1 }$ . For this first sample suficiently many:

$$
( z _ { n } ^ { 1 } ) _ { n \in [ N ] } \stackrel { i . i . d . } { \sim } P _ { 1 } ,
$$

$$
\left( z _ { n , k } ^ { 0 } \right) _ { n \in [ N ] } \stackrel { i . i . d . } { \sim } P _ { 0 } .\tag{44}
$$

We then consider the empirical objective:

$$
J _ { K , N } ( \theta ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \log E ^ { ( K , \theta ) } \left( z _ { n } ^ { 1 } , ( z _ { n , k } ^ { 0 } ) _ { k \in [ K ] } \right) .\tag{45}
$$

We then maximize this objective to receive the following estimator:

$$
{ \widehat { \theta } } _ { K , N } : \in \mathop { \mathrm { a r g m a x } } _ { \theta \in \Theta } J _ { K , N } ( \theta ) ,\tag{46}
$$

or, more generally, a sequence of near-maximizing estimators $( { \widehat { \theta } } _ { K , N } ) _ { N \in \mathbb { N } _ { 1 } }$ that might come with some optimization gap $( \hat { \xi } _ { K , N } ) _ { N \in \mathbb { N } _ { 1 } } , ~ e . g .$ . a gradient optimizer stuck in a local, non-global, maximum, etc., satisfying:

$$
J _ { K , N } ( \widehat { \theta } _ { K , N } ) \geq \operatorname* { s u p } _ { \theta \in \Theta } J _ { K , N } ( \theta ) - \xi _ { K , N } , \qquad \ \xi _ { K } : = \operatorname* { l i m } _ { N  \infty } \operatorname* { s u p } _ { \xi _ { K , N } \in \mathbb { R } } \mathbb { e } _ { \geq 0 } .\tag{47}
$$

The resulting e-variable for $H _ { 0 }$ is then:

$$
E ^ { ( K , \hat { \theta } _ { K , N } ) } ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \dots , z _ { K } ^ { 0 } ) = \frac { \exp ( g _ { \hat { \theta } _ { K , N } } ( z ^ { 1 } ) ) } { \frac { 1 } { K + 1 } \left( \exp ( g _ { \hat { \theta } _ { K , N } } ( z ^ { 1 } ) ) + \sum _ { k = 1 } ^ { K } \exp ( g _ { \hat { \theta } _ { K , N } } ( z _ { k } ^ { 0 } ) ) \right) } .\tag{48}
$$

We can use M-estimation theory under standard Wald-type conditions, see [Wal49, vdV98], to get the following results on the asymptotic approximately optimal e-power of the e-variable $E ^ { ( K , \widehat { \theta } _ { K , N } ) }$ for $H _ { 0 }$ vs. $H _ { 1 }$ for the limit $N \to \infty$

Theorem 3.4 (Approximate growth optimality, see Proof B.2). Let the setting be like in Notations 3.3. Assume $P _ { 1 } \ll P _ { 0 }$ and let $g _ { \theta } : \Omega  \mathbb { R }$ be a family of B-measurable functions, $\theta \in \Theta$ , with the following properties:

1. Compactness: $\Theta$ is a compact metric space.

2. Continuity: for every $z \in \Omega$ the map $\theta \mapsto g _ { \theta } ( z )$ is continuous.

3. Non-degeneracy: there exists a $\theta \in \Theta$ with $J _ { K } ( \theta ) > - \infty$

4. Uniform integrable envelopes:

$$
\mathbb { E } _ { P _ { 0 } } \left[ \operatorname* { s u p } _ { \theta \in \Theta } g _ { \theta } ( Z ^ { 0 } ) _ { + } \right] < \infty ,
$$

$$
\mathbb { E } _ { P _ { 1 } } \left[ \operatorname* { s u p } _ { \theta \in \Theta } g _ { \theta } ( Z ^ { 1 } ) _ { - } \right] < \infty .\tag{49}
$$

Then, $f o r$ fixed $K \geq 1$ , we have the following uniform strong law of large numbers for the objective, that is the $\mathbb { P } _ { 1 }$ -almost-sure convergence:

$$
\operatorname* { s u p } _ { \theta \in \Theta } \left| J _ { K , N } ( \theta ) - J _ { K } ( \theta ) \right| \overset { N \to \infty } { \longrightarrow } 0 \qquad \mathbb { P } _ { 1 } - a . s .\tag{50}
$$

Furthermore, for every near-maximizing sequence $\begin{array} { r l } { \big ( \widehat { \theta } _ { K , N } \big ) _ { N \in \mathbb { N } _ { 1 } } } & { { } o f \ \big ( J _ { K , N } \big ) _ { N \in \mathbb { N } _ { 1 } } } \end{array}$ with optimization gaps $\left( \xi _ { K , N } \right) _ { N \in \mathbb { N } _ { 1 } }$ we have the $\mathbb { P } _ { 1 }$ -almost-sure convergences<sup>4</sup>:

$$
\operatorname* { l i m } _ { N \to \infty } \operatorname* { i n f } _ { \mathbb { P } _ { 1 } } [ \log E ^ { ( K , \hat { \theta } _ { K , N } ) } ] \geq \operatorname* { s u p } _ { \theta \in \Theta } J _ { K } ( \theta ) - \xi _ { K } \qquad \mathbb { P } _ { 1 } \boldsymbol { - a } . s .\tag{51}
$$

Now consider the maximizers $g _ { c } ^ { * } : \Omega \to \mathbb { R } \cup \{ - \infty \}$ of $J _ { K }$ from Proposition 3.2 (in log space) for $c \in \mathbb { R }$ :

$$
g _ { c } ^ { \ast } ( z ) : = g ^ { \ast } ( z ) + c ,
$$

$$
g ^ { \ast } ( z ) : = \log \frac { d P _ { 1 } } { d P _ { 0 } } ( z ) .\tag{52}
$$

We now assume, in addition to the above, that we have the following properties:

5. Approximating class: there exist $\theta \in \Theta$ and $\epsilon , c \in \mathbb { R }$ with $0 \leq \epsilon <$ ∞ such that:

$$
\displaystyle \operatorname* { s u p } _ { z \in \Omega } | g _ { \theta } ( z ) - g _ { c } ^ { * } ( z ) | \leq \epsilon .\tag{53}
$$

6. Finite $\chi ^ { 2 } .$ -divergence: we have: $P _ { 1 } \ll P _ { 0 }$ with $\chi ^ { 2 } ( P _ { 1 } \| P _ { 0 } ) < \infty$

Then we have approximate growth optimality (AGRO) expressed as the following $\mathbb { P } _ { 1 }$ -almost-sure inequalities:

(54)

$$
\begin{array} { r l } { \underbrace { \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) } _ { m a x i m a l \ k \cdot p o w e r } \ge \underbrace { \operatorname* { l i m i n f } _ { N \to \infty } { \mathbb { I } } _ { \boldsymbol { \phi } _ { 1 } } [ \log E ^ { ( K , \delta _ { K , N } ) } ] } _ { a s y m p t o t i c \ l e a r n e d \ k \cdot p o w e r } } & { } \\ { \ge \underbrace { \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) } _ { m a x i m a l \ E \cdot p o w e r } - \underbrace { \log \left( 1 + \frac { \chi ^ { 2 } ( P _ { 1 } \| P _ { 0 } ) } { K + 1 } \right) } _ { p e n a l t y \ f o r \ f i n i t e \ K } \underbrace { - \frac { 1 } { 2 } \epsilon ^ { 2 } } _ { a p p r o x i m a t i o n \ g a r \ o p t i m i z a t i o n \ g a p } . } \end{array}\tag{55}
$$

In particular, if $\begin{array} { r } { \mathrm { \ddot { K L } } ( P _ { 1 } \| P _ { 0 } ) > \frac { 1 } { 2 } \epsilon ^ { 2 } + \operatorname* { s u p } _ { K \in \mathbb { N } _ { 1 } } \xi _ { K } } \end{array}$ , then we can find $K , N \ge 1$ such that we achieve positive e-power E<sub>P</sub> [log $E ^ { ( K , \widehat { \theta } _ { K , N } ) } ] > 0$ for $H _ { 1 }$ with high probability.

After training, we have thus found the e-variable $E ^ { ( K , \widehat { \theta } _ { K , N } ) }$ for $H _ { 0 }$ with optimized e-power for $H _ { 1 }$ or at least with positive e-power $E _ { \mathbb { P } _ { \lambda } }$ <sub>1</sub> [log $E ^ { ( K , \widehat { \theta } _ { K , N } ) } ] > 0$ for $H _ { 1 }$ with high probability.

For newly sampled points at each time step $t \geq 1$

$$
x _ { t } \sim Q ,
$$

$$
( z _ { t , k } ^ { 0 } ) _ { k \in [ K ] } \stackrel { \mathrm { i . i . d . } } { \sim } P _ { 0 } ,\tag{56}
$$

we can thus finally define the (conditional) e-variable at time $t \geq 1$

$$
E _ { t } : = E ^ { ( K , \widehat { \theta } _ { K , N } ) } ( x _ { t } , z _ { t , 1 } ^ { 0 } , \ldots , z _ { t , K } ^ { 0 } ) ,\tag{57}
$$

and recursively the e-test martingale for $t \geq 1 { : }$

$$
W _ { t } : = E _ { t } \cdot W _ { t - 1 } ,
$$

$$
W _ { 0 } : = 1 .\tag{58}
$$

We thus reject $H _ { 0 }$ in favor of $H _ { 1 }$ whenever $W _ { t }$ crosses the threshold $\textstyle { \frac { 1 } { \alpha } }$ and we can stop. Recall, that in real physical experiments, the standard is to use the so called 5σ-rule with $\textstyle { \frac { 1 } { \alpha } } \approx 3 . 5 \cdot 1 0 ^ { 6 }$ , that is: log $\textstyle { \frac { 1 } { \alpha } } \approx 1 5$ Remark 3.5 (Deep neural networks, see [GBC16, Bis23, Mur22]). For deep neural networks $g _ { \theta }$ the map:

$$
( \theta , z ) \mapsto g _ { \theta } ( z ) ,\tag{59}
$$

is continuous and one usually even has a Lipschitz type inequality, based on the used architecture:

$$
| g _ { \theta } ( z ) | \le C ( \theta ) \cdot ( 1 + \| z \| ) ,\tag{60}
$$

with a constant $C ( \theta )$ that is continuous in θ. Furthermore, the class of deep neural networks is a universal approximator. $S o ,$ on compact Ω, we can adjust the architecture and the parameterization such that:

$$
g _ { \boldsymbol { \theta } } \approx g _ { c } ^ { * } ,\tag{61}
$$

for some θ from some suficiently $b i g ,$ but compact Θ. So, the remaining envelope properties from Theorem 3.4 reduce for deep neural networks to simple moment conditions:

$$
\mathbb { E } _ { P _ { 1 } } [ \lVert Z ^ { 1 } \rVert ] < \infty ,
$$

$$
\begin{array} { r } { \mathbb { E } _ { P _ { 0 } } [ \| Z ^ { 0 } \| ] < \infty . } \end{array}\tag{62}
$$

$S o ,$ for deep neural networks, in practice, we can argue that Theorem 3.4 should hold on compact Ω. For noncompact, but σ-compact, Ω one can exhaust the space Ω with compacta and apply Theorem $\it 3 . 4$ individually on each compact subspace. We do not provide further technical details for this generalization here.

Remark 3.6 (Removing the ambiguity of a constant). In practice, we can remove the ambiguity of the constant $c \in \mathbb { R }$ in the optimal function in Proposition 3.2 and Theorem 3.4 by introducing the following regularizer:

$$
R _ { K , N } ( \theta ) : = \left| \log \frac { 1 } { N K } \sum _ { n = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \exp ( g _ { \theta } ( z _ { n , k } ^ { 0 } ) ) \right| ^ { 2 } ,\tag{63}
$$

which enforces to keep $c \approx 1 \ ( o r , \ c \approx 0$ in log-space, resp.). We then minimize the following joint objective with a small regularization parameter $\kappa > 0$

$$
\hat { \theta } _ { K , N } : \in \mathop { \operatorname { a r g m a x } } _ { \theta \in \Theta } \left( J _ { K , N } ( \theta ) - \kappa \cdot R _ { K , N } ( \theta ) \right) .\tag{64}
$$

Alternatively, we can sample separate i.i.d. samples $( z _ { m } ^ { 0 } ) _ { m \in [ M ] } \sim P _ { 0 }$ with $M \gg 1$ after training (with or without such regularization) and compute:

$$
\hat { c } : = \log \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \exp \left( g _ { \hat { \theta } _ { K , N } } ( z _ { m } ^ { 0 } ) \right) ,
$$

$$
\tilde { g } _ { \hat { \theta } _ { K , N } } ( x ) : = g _ { \hat { \theta } _ { K , N } } ( x ) - \hat { c } .\tag{65}
$$

With the above methods, we can remove the additive constant c and achieve that: $g _ { \hat { \theta } _ { K , N } }$ ≈ log $\frac { d P _ { 1 } } { d P _ { 0 } }$ . Note that the constant, anyways, does not afect the e-variable $E ^ { ( K , \theta ) }$ of our form Equation (43), and only helps to interpret $g _ { \hat { \theta } _ { K , N } }$ as a log-likelihood ratio.

Remark 3.7 (Clipping - again). In order to prevent a learned e-variable $E ^ { ( K , \widehat { \theta } _ { K , N } ) }$ , which might by chance become close to zero (log-unbounded from below), to quickly destroy the wealth process $W _ { t }$ , we could introduce a regularizing clipping parameter $\lambda \in [ 0 , 1 ]$ , as described in Remark 2.9:

$$
E ^ { ( K , \theta , \lambda ) } ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \ldots , z _ { K } ^ { 0 } ) : = ( 1 - \lambda ) + \lambda \cdot E ^ { ( K , \theta ) } ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \ldots , z _ { K } ^ { 0 } )\tag{66}
$$

$$
\exp ( g _ { \theta } ( z ^ { 1 } ) )
$$

$$
= ( 1 - \lambda ) + \lambda \cdot \frac { \exp ( g _ { \theta } ( z ^ { 0 } ) ) } { \frac { 1 } { K + 1 } \left( \exp ( g _ { \theta } ( z ^ { 1 } ) ) + \sum _ { k = 1 } ^ { K } \exp ( g _ { \theta } ( z _ { k } ^ { 0 } ) ) \right) } .\tag{67}
$$

This would, as mentioned in Remark 2.9, restore the explicit type-II error bounds from Theorem 2.8. However, there are diferent ways to choose $\lambda ,$ as we will indicate in the following.

a.) A typical, simple and efective choice for λ would be to just set $\begin{array} { r } { \lambda _ { 1 } : = \frac { 1 } { 2 } } \end{array}$

b.) Based on the observation, that we already have the deterministic upper bound:

$$
\log E ^ { ( K , \theta ) } \leq \log ( K + 1 ) ,\tag{68}
$$

a symmetric lower bound could be achieved by the choice $\textstyle \lambda _ { K } : = { \frac { K } { K + 1 } }$ . Then we would just get:

$$
E ^ { \left( K , \theta , \lambda _ { K } \right) } = { \frac { 1 } { K + 1 } } + { \frac { K } { K + 1 } } \cdot E ^ { \left( K , \theta \right) } = { \frac { 1 } { K + 1 } } + { \frac { \exp \left( g \theta \left( z ^ { 1 } \right) \right) } { { \frac { 1 } { K } } \left( \exp ( g \theta \left( z ^ { 1 } \right) ) + \sum _ { k = 1 } ^ { K } \exp ( g \theta \left( z _ { k } ^ { 0 } \right) ) \right) } } ,\tag{69}
$$

with the symmetric lower and upper bounds:

$$
- \log ( K + 1 ) \leq \log E ^ { ( K , \theta , \lambda _ { K } ) } \leq \log ( K + 1 ) .\tag{70}
$$

c.) Alternatively, we could jointly fit θ and $\lambda ,$ to maximize e-power, similarly to before:

$$
( \widehat { \theta } _ { K , N } , \widehat { \lambda } _ { K , N } ) : \in \underset { \lambda \in [ 0 , 1 ] } { \operatorname { r g m a x } } J _ { K , N } ( \theta , \lambda ) , \qquad J _ { K , N } ( \theta , \lambda ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \log E ^ { ( K , \theta , \lambda ) } \left( z _ { n } ^ { 1 } , ( z _ { n , k } ^ { 0 } ) _ { k \in [ K ] } \right) .\tag{71}
$$

Since λ might, in this fitting procedure, still become too overconfident $( \hat { \lambda } _ { K , N } \approx 1 )$ , we could afterwards scale it down by a fixed hyperparameter $\rho \in [ 0 , 1 )$ , e.g. again $\begin{array} { r } { \rho _ { 1 } = \frac { 1 } { 2 } , \rho _ { 2 } = \frac { 2 } { 3 } , \rho _ { 3 } = \frac { 3 } { 4 } , \rho _ { 4 } = \frac { 4 } { 5 } \ o r \ \rho _ { K } = \frac { K } { K + 1 } } \end{array}$ , and put:

$$
\tilde { \lambda } _ { K , N } : = \rho \cdot \hat { \lambda } _ { K , N } ,\tag{72}
$$

and use the e-variable

$$
E ^ { ( K , \rho , N ) } : = E ^ { ( K , \hat { \theta } _ { K , N } , \tilde { \lambda } _ { K , N } ) } ,\tag{73}
$$

which then always produces the deterministic bounds:

$$
\log ( 1 - \rho ) \leq \log E ^ { ( K , \rho , N ) } \leq \log ( K + 1 ) .\tag{74}
$$

d.) There are further options, like providing a whole schedule $( \lambda _ { t } ) _ { t \in \mathbb { N } _ { 0 } }$ over time, or even learning such a schedule (but only on past data). All these choices would not violate the conditional e-variable conditions and all guarantees from Theorem 2.7 and Theorem 2.8 would still hold with the corresponding numbers.

Remark 3.8 (The choice of the number K). There are two considerations for the choice of the number $K \geq 1$ that restrict the power of $E ^ { ( K , \theta ) }$

a.) The first obstruction is the deterministic upper bound:

$$
E ^ { ( K , \theta ) } \leq K + 1 ,\tag{75}
$$

which restricts the e-power to:

$$
\mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \log E ^ { ( K , \theta ) } \right] \leq \log ( K + 1 ) .\tag{76}
$$

If we even want to have a chance to be approximately close to the upper limit of $\mathrm { K L } ( P _ { 1 } \| P _ { 0 } )$ then at least the upper bound should be close to or even exceed it:

$$
\begin{array} { r } { \log ( K + 1 ) \gtrsim \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) , } \end{array}\tag{77}
$$

leading to the following condition for the choice of K:

$$
K + 1 \gtrsim \exp ( \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) ) \approx \exp \left( \frac { 1 } { M } \sum _ { m = 1 } ^ { M } g _ { \hat { \theta } _ { K , N } } ( z _ { m } ^ { 1 } ) \right) , \qquad ( z _ { m } ^ { 1 } ) _ { m \in [ M ] } \overset { i . i . d . } { \sim } P _ { 1 } .\tag{78}
$$

which one could track and adjust during training.

b.) The second obstruction is the lower bound from Theorem 3.4 for the optimized limit e-variable:

$$
\mathbb { E } _ { \mathbb { P } _ { 1 } } [ \log \cal { E } ^ { ( K , \hat { \theta } ) } ] \geq \quad \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) - \log \left( 1 + \frac { \chi ^ { 2 } ( P _ { 1 } \| P _ { 0 } ) } { K + 1 } \right) - \frac { 1 } { 2 } \epsilon ^ { 2 } .\tag{79}
$$

If we wanted the term for the penalty for finite K to become of size smaller than $\delta > 0$ we could require:

$$
\delta \stackrel { ! } { \gtrsim } \frac { \chi ^ { 2 } ( P _ { 1 } \| P _ { 0 } ) } { K + 1 } \geq \log \left( 1 + \frac { \chi ^ { 2 } ( P _ { 1 } \| P _ { 0 } ) } { K + 1 } \right) .\tag{80}
$$

This would lead to the following criterion:

$$
K + 1 \gtrsim \frac { \chi ^ { 2 } ( P _ { 1 } \| P _ { 0 } ) } { \delta } \approx \frac { 1 } { \delta } \cdot \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left( \exp ( g _ { \bar { \theta } _ { K , N } } ( z _ { m } ^ { 0 } ) ) - 1 \right) ^ { 2 } , \qquad ( z _ { m } ^ { 0 } ) _ { m \in [ M ] } \overset { i . i . d . } { \sim } P _ { 0 }\tag{81}
$$

$$
\approx \frac { 1 } { \delta } \cdot \left( \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \exp ( g _ { \hat { \theta } _ { K , N } } ( z _ { m } ^ { 1 } ) ) - 1 \right) ,
$$

$$
( z _ { m } ^ { 1 } ) _ { m \in [ M ] } \stackrel { i . i . d . } { \sim } P _ { 1 } .\tag{82}
$$

As before, also this quantity could be checked and adjusted during training.

Remark 3.9 (Alternative training objective). Another popular and rather stable training objective in machine learning for learning density ratios comes from another variational form of the KL-divergence, see [NWJ10, BBR<sup>+</sup>18, POvdO<sup>+</sup>19]:

$$
\mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) = \operatorname* { s u p } _ { g } \left\{ \mathbb { E } _ { P _ { 1 } } [ g ] - \mathbb { E } _ { P _ { 0 } } [ \exp ( g ) ] + 1 \right\} , \qquad w i t h \ u n i q u e \ m a x i m i z e r : \qquad g ^ { * } ( z ) = \log \frac { d P _ { 1 } } { d P _ { 0 } } ( z ) .\tag{83}
$$

Here, we then would, again, maximize the corresponding empirical version with samples $( z _ { n } ^ { 1 } ) _ { n \in [ N _ { 1 } ] } \stackrel { i . i . d . } { \sim } P _ { 1 }$ and $( z _ { n } ^ { 0 } ) _ { n \in [ N _ { 0 } ] } \stackrel { i . i . d . } { \sim } P _ { 0 }$ :

$$
\widehat { \theta } _ { N _ { 1 } , N _ { 0 } } : \in \mathop { \mathrm { a r g m a x } } _ { \theta \in \Theta } I _ { N _ { 1 } , N _ { 0 } } ( \theta ) , \qquad I _ { N _ { 1 } , N _ { 0 } } ( \theta ) : = \frac { 1 } { N _ { 1 } } \sum _ { n = 1 } ^ { N _ { 1 } } g _ { \theta } ( z _ { n } ^ { 1 } ) - \frac { 1 } { N _ { 0 } } \sum _ { n = 1 } ^ { N _ { 0 } } \exp ( g _ { \theta } ( z _ { n } ^ { 0 } ) ) + 1 .\tag{84}
$$

The estimator is known to have less bias, but higher variance. After training, one can use $g _ { \hat { \theta } _ { N _ { 1 } , N _ { 0 } } }$ in our e-variable from Equation (43) with some chosen $K \geq 1$

The above concepts and constructions can also be combined with the following.

Remark 3.10 (Finite time horizon and inverse temperature). Recall that in Section 3 we used the construction from Equation (30) to approximate the e-variable from Equation (27):

$$
\tilde { h } ( z ^ { 1 } ) : = \frac { h ( z ^ { 1 } ) } { \mathbb { E } _ { P _ { 0 } } [ h ] } \approx \frac { h ( z ^ { 1 } ) } { \frac { 1 } { K + 1 } \left( h ( z ^ { 1 } ) + \sum _ { k = 1 } ^ { K } h ( z _ { k } ^ { 0 } ) \right) } = : E ^ { ( K , h ) } ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \ldots , z _ { K } ^ { 0 } ) ,\tag{85}
$$

where we then optimized h in Theorem $\it 3 . 4$ to receive: $h = \exp ( g _ { \hat { \theta } _ { K , N } } )$ with $g _ { \hat { \theta } _ { K , N } }$ ≈ log $\frac { d P _ { 1 } } { d P _ { 0 } }$ , which is approximately growth optimal in the long run and minimizes average rejection time, see Remark ${ 2 . 1 0 } .$ In contrast, following $[ F o r { \mathcal { Q } } 6 a ] ,$ we now want to consider the case where the time horizon $T \in \mathbb { N } f o r$ the real experiments $( x _ { t } ) _ { t \in [ T ] } \stackrel { i . i . d . } { \sim } Q$ is a priori known and finite. For this we want to fix h from above, but we now modify it with an inverse temperature regularization parameter $\beta \in \mathbb { R } _ { > 0 } .$

$$
h _ { \beta } ( z ^ { 1 } ) : = \frac { h ( z ^ { 1 } ) ^ { \beta } } { \mathbb { E } _ { P _ { 0 } } [ h ^ { \beta } ] } \approx \frac { h ( z ^ { 1 } ) ^ { \beta } } { \frac { 1 } { K + 1 } \left( h ( z ^ { 1 } ) ^ { \beta } + \sum _ { k = 1 } ^ { K } h ( z _ { k } ^ { 0 } ) ^ { \beta } \right) } = : E ^ { ( K , h , \beta ) } ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \ldots , z _ { K } ^ { 0 } ) ,\tag{86}
$$

in order to minimize the type-II error at the one future time $T ,$ when at present time $t \leq T$ , when we have already observed $W _ { t - 1 }$

$$
\beta _ { t - 1 } : \in \operatorname * { a r g m i n } _ { \beta > 0 } P _ { 1 } \left( W _ { T } \leq \frac { 1 } { \alpha } \biggm | \mathcal { F } _ { t - 1 } \right) .\tag{87}
$$

Assume $f o r$ now, that we are in the idealized case, where we have directly used $h _ { \beta }$ as e-variables; then the above reduces to:

$$
P _ { 1 } \left( W _ { T } \leq { \frac { 1 } { \alpha } } { \bigg | } { \mathcal { F } } _ { t - 1 } \right) = P _ { 1 } \left( \sum _ { n = t } ^ { T } \log h _ { \beta } ( z _ { t } ^ { 1 } ) \leq \log { \frac { 1 } { \alpha } } - \log W _ { t - 1 } { \bigg | } { \mathcal { F } } _ { t - 1 } \right)\tag{88}
$$

$$
= P _ { 1 } \left( \frac { 1 } { T - t + 1 } \sum _ { n = t } ^ { T } \log h ( z _ { t } ^ { 1 } ) \leq \frac { 1 } { \beta } \left( \frac { \log \frac { 1 } { \alpha } - \log W _ { t - 1 } } { T - t + 1 } + \log \mathbb { E } _ { P _ { 0 } } [ h ^ { \beta } ] \right) \Bigg | \mathcal { F } _ { t - 1 } \right) .\tag{89}
$$

Note that only the rhs depends on $\beta ,$ , so minimizing that type-II error reduces to minimizing the rhs w.r.t. $\beta _ { i }$ which we only need for the case when we have not rejected $H _ { 1 }$ yet, that is, $\begin{array} { r } { i f W _ { t - 1 } \le \frac { 1 } { \alpha } } \end{array}$ . We thus arrive at the exact objective:

$$
\beta _ { t - 1 } \in \underset { \beta > 0 } { \operatorname { a r g m i n } } \frac { 1 } { \beta } \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \left( \underbrace { \log \frac { 1 } { \alpha } - \log W _ { t - 1 } } _ { T - t + 1 } + \underbrace  \log \mathbb { E } _ { P _ { 0 } } \left[ h ^ { \beta } \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \right) \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \right) ,\tag{90}
$$

where we interpret $\ell _ { t - 1 }$ as the residual target evidence for rejecting $H _ { 1 }$ per remaining time step until time T. The optimization problem can then numerically be solved with Newton iteration:

$$
\beta ^ { ( i + 1 ) } : = \beta ^ { ( i ) } - \frac { \beta ^ { ( i ) } \cdot \psi ^ { \prime } ( \beta ^ { ( i ) } ) - \psi ( \beta ^ { ( i ) } ) - \ell _ { t - 1 } } { \beta ^ { ( i ) } \cdot \psi ^ { \prime \prime } ( \beta ^ { ( i ) } ) } , \qquad w i t h .\tag{91}
$$

$$
\psi ( \beta ) = \log \mathbb { E } _ { P _ { 0 } } [ h ^ { \beta } ] , \qquad \psi ^ { \prime } ( \beta ) = \frac { \mathbb { E } _ { P _ { 0 } } [ h ^ { \beta } \log h ] } { \mathbb { E } _ { P _ { 0 } } [ h ^ { \beta } ] } \qquad \psi ^ { \prime \prime } ( \beta ) = \frac { \mathbb { E } _ { P _ { 0 } } [ h ^ { \beta } ( \log h ) ^ { 2 } ] } { \mathbb { E } _ { P _ { 0 } } [ h ^ { \beta } ] } - \psi ^ { \prime } ( \beta ) ^ { 2 } ,\tag{92}
$$

where each expectation value is replaced by the empirical counterparts based on the same one batch of i.i.d. samples $( z _ { m } ^ { 0 } ) _ { m \in [ M ] } \stackrel { i . i . d . } { \sim } P _ { 0 }$ and with $h = \exp ( g _ { \hat { \theta } _ { K , N } } )$ . After convergence, $i  \infty ,$ , practically after $3 - 5$ rounds of iterations, one has found an approximate $\beta _ { t - 1 } : = \beta ^ { ( \infty ) }$ , which one can use, at time $t \in [ T ]$ , for the conditional e-variable:

$$
E _ { t } ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \dots , z _ { K } ^ { 0 } ) : = { \frac { \exp ( \beta _ { t - 1 } \cdot g _ { \hat { \theta } _ { K , N } } ( z ^ { 1 } ) ) } { { \frac { 1 } { K + 1 } } \left( \exp ( \beta _ { t - 1 } \cdot g _ { \hat { \theta } _ { K , N } } ( z ^ { 1 } ) ) + \sum _ { k = 1 } ^ { K } \exp ( \beta _ { t - 1 } \cdot g _ { \hat { \theta } _ { K , N } } ( z _ { k } ^ { 0 } ) ) \right) } } .\tag{93}
$$

At the next time step $t + 1$ , this can be repeated, leading to an adaptive predictive sequence $( \beta _ { t } ) _ { t \in [ T ] }$ of inverse temperatures. This, of course, can be combined with the clipping regularization from Remark 3.7. Note that, even with these β values in place, each of which is only dependent on their own past, the anytime type-I error guarantees from Theorem 2.7 still hold.

## 4 An alternative testing scenario

Consider the scenario, where we want to test:

$$
H _ { 0 } : Q = P _ { 0 } , \mathrm { v s . } H _ { 1 } : Q \neq P _ { 0 } ,\tag{94}
$$

For this we pre-train $E ^ { ( K , \theta ) }$ as before with help of some alternative $P _ { 1 } \neq P _ { 0 }$ first, if possible. However, to adjust to the real data distribution coming from $Q ,$ which we suspect to deviate from $P _ { 0 } .$ , we will then train $E ^ { ( K , \theta ) }$ on all past observations $x _ { n } \sim Q$ with $n = 1 , \ldots , t - 1$ , explicitly excluding $x _ { t }$ (and futute $x _ { n } )$ for training at time t, and suficiently many simulations from $P _ { 0 }$ as before (and/or stored simulations from the past). More concretely, at each time $t \geq 1$ , we optimize the empirical objective with data from times until t−1:

$$
( \widehat { \theta } _ { K , t - 1 } , \widehat { \lambda } _ { K , t - 1 } ) : \in \underset { \Delta \in [ 0 , 1 ] } { \arg \operatorname* { m a x } } \left\{ \frac { 1 } { t - 1 } \sum _ { n = 1 } ^ { t - 1 } \log E ^ { ( K , \theta , \lambda ) } \left( x _ { n } , ( z _ { n , k } ^ { 0 } ) _ { k \in [ K ] } \right) - \kappa \cdot R _ { K , t - 1 } ( \theta ) \right\} ,\tag{95}
$$

and plug the new point $x _ { t } \sim Q$ together with fresh simulations $( z _ { t , k } ^ { 0 } ) _ { k \in [ K ] } \stackrel { \mathrm { i . i . d . } } { \sim } P _ { 0 }$ into it:

$$
\begin{array} { r } { E _ { t } : = E ^ { ( K , \hat { \theta } _ { K , t - 1 } , \tilde { \lambda } _ { K , t - 1 } , \beta _ { t - 1 } ) } \left( x _ { t } , ( z _ { t , k } ^ { 0 } ) _ { k \in [ K ] } \right) . } \end{array}\tag{96}
$$

This will still be a valid conditional e-variable for $H _ { 0 }$ and the anytime type-I error guarantees will still hold. Note that the trained e-variable $E ^ { ( K , \theta ) }$ will then rather learn to approximate $\frac { d Q } { d P _ { 0 } }$ than $\frac { d P _ { 1 } } { d P _ { 0 } }$ , which, in this testing scenario, is more optimal, if $H _ { 1 }$ is true.

## 5 Discussion

For the simulation-based hypothesis testing setting, $H _ { 0 } : Q = P _ { 0 } { \mathrm { ~ v s . ~ } } H _ { 1 } : Q = P _ { 1 }$ , where we do not have densities and can only sample from $P _ { 0 }$ and $P _ { 1 }$ , we constructed a sequential test based on an e-test martingale, which comes with anytime-valid type-I error guarantees, approximate growth optimality, geometrically decaying type-II error bounds, and asymptotic power one.

Future work could address further generalizations for the anytime-valid simulation-based hypothesis testing setting, where we have, instead of simple, composite hypothesis classes $\mathcal { H } _ { 0 } , \mathcal { H } _ { 1 } \subseteq \mathcal { P } ( \Omega )$ , leading to the statistical test $H _ { 0 } : Q \in \mathcal { H } _ { 0 }$ vs. $H _ { 1 } : Q \in \mathcal { H } _ { 1 }$ , and/or the setting of (filtration-adapted) dependent data streams $( X _ { t } ) _ { t \in \mathbb { N } _ { 1 } } \sim Q$ , instead of i.i.d. data. Furthermore, a more realistic approximate growth optimality theorem could be proven for more realistic norms, non-compact Ω and Θ, and (universally approximating) function classes of more practical relevance.

## References

[ATL08] ATLAS Collaboration. The ATLAS Experiment at the CERN Large Hadron Collider. JINST, 3:S08003, 2008. Also published by CERN Geneva in 2010.

[ATL12] ATLAS Collaboration. Observation of a new particle in the search for the standard model higgs boson with the atlas detector at the lhc. Physics Letters B, 716:1–29, 2012.

[Azu67] Kazuoki Azuma. Weighted sums of certain dependent random variables. Tohoku Mathematical Journal, 19(3):357–367, 1967.

[BBR<sup>+</sup>18] Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeshwar, et al. Mutual Information Neural Estimation. In International Conference on Machine Learning, pages 531–540. PMLR, 2018.

[BBS09] Stefen Bickel, Michael Brückner, and Tobias Schefer. Discriminative Learning under Covariate Shift. Journal of Machine Learning Research, 10:2137–2155, 2009.

[BC20] Johann Brehmer and Kyle Cranmer. Simulation-based inference methods for particle physics, 2020.

[BCLP18] Johann Brehmer, Kyle Cranmer, Gilles Louppe, and Juan Pavez. Likelihood-Free Inference with Neural Networks. Physical Review D, 98(5):052004, 2018.

[Bis23] Christopher M. Bishop. Deep Learning: Foundations and Concepts. Springer, 2023.

[CCGV11] Glen Cowan, Kyle Cranmer, Eilam Gross, and Ofer Vitells. Asymptotic formulae for likelihoodbased tests of new physics. The European Physical Journal C, 71(2), February 2011.

[CER11] CERN. Proceedings of the PHYSTAT 2011 Workshop on Statistical Issues Related to Discovery Claims in Search Experiments and Unfolding, Geneva, 2011. CERN.

[CKNH20] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geofrey Hinton. A Simple Framework for Contrastive Learning of Visual Representations. In International Conference on Machine Learning, pages 1597–1607. PMLR, 2020.

[CLRG26] Ben Chugg, Tyron Lardy, Aaditya Ramdas, and Peter Grünwald. On Admissibility in Post-hoc Hypothesis Testing. International Journal of Approximate Reasoning, 191:109634:1–109634:33, 2026.

[CMS08] CMS Collaboration. The CMS experiment at the CERN LHC. JINST, 3:S08004, 2008. Also published by CERN Geneva in 2010.

[CMS12] CMS Collaboration. Observation of a new boson at a mass of 125 gev with the cms experiment at the lhc. Physics Letters B, 716:30–61, 2012.

[CPL15] Kyle Cranmer, Juan Pavez, and Gilles Louppe. Approximating Likelihood Ratios with Classifiers. arXiv preprint arXiv:1506.02169, 2015.

[CPLB16] Kyle Cranmer, Juan Pavez, Gilles Louppe, and William K. Brooks. Experiments Using Machine Learning to Approximate Likelihood Ratios for Mixture Models. In Journal of Physics: Conference Series, volume 762, page 012034. IOP Publishing, 2016.

[DER26] Alexander Dombowsky, Barbara E. Engelhardt, and Aaditya Ramdas. Besag-Cliford E-Values for Unnormalized Testing. arXiv preprint arXiv:2603.15845, 2026.

[DHR<sup>+</sup>22] Arnaud Delaunoy, Joeri Hermans, François Rozet, Antoine Wehenkel, and Gilles Louppe. Towards Reliable Simulation-Based Inference with Balanced Neural Ratio Estimation. In Advances in Neural Information Processing Systems, volume 35, 2022.

[DMF<sup>+</sup>23] Arnaud Delaunoy, Benjamin Kurt Miller, Patrick Forré, Christoph Weniger, and Gilles Louppe. Balancing Simulation-based Inference for Conservative Posteriors. In Proceedings of the 5th Symposium on Advances in Approximate Bayesian Inference (AABI), Honolulu, Hawaii, USA, July 2023.

[DMP20] Conor Durkan, Iain Murray, and George Papamakarios. Contrastive Learning for Neural Posterior Estimation. In International Conference on Learning Representations, 2020.

[Dur19] Richard Durrett. Probability: Theory and Examples. Cambridge University Press, 5 edition, 2019.

[FC98] Gary J. Feldman and Robert D. Cousins. A Unified approach to the classical statistical analysis of small signals. Phys. Rev. D, 57:3873–3889, 1998.

[FDF<sup>+</sup>20] Marco Federici, Anjan Dutta, Patrick Forré, Nate Kushman, and Zeynep Akata. Learning Robust Representations via Multi-View Information Bottleneck. In International Conference on Learning Representations, 2020.

[FFTV24] Marco Federici, Patrick Forré, Ryota Tomioka, and Bastiaan S. Veeling. Latent Representation and Simulation of Markov Processes via Time-Lagged Information Bottleneck. In International Conference on Learning Representations, 2024.

[For26a] Patrick Forré. The Type-II Error of Test Supermartingales: e-Power versus the Chernof-Stein Exponent. arXiv preprint arXiv:2609.27765, 2026.

[For26b] Patrick Forré. Type-II Error Bounds of Test Supermartingales from Lower-Tail Hypothesis. arXiv preprint arXiv:2609.27766, 2026.

[FRF23] Marco Federici, David Ruhe, and Patrick Forré. On the Efectiveness of Hybrid Mutual Information Estimation. arXiv preprint arXiv:2306.00608, 2023.

[FTF21] Marco Federici, Ryota Tomioka, and Patrick Forré. An Information-theoretic Approach to Distribution Shifts. In Advances in Neural Information Processing Systems, 2021.

[GBC16] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016.

[GdHK20] Peter Grünwald, Rianne de Heide, and Wouter M. Koolen. Safe Testing. Journal of the Royal Statistical Society: Series B, 82(3):543–564, 2020.

[GH10] Michael U. Gutmann and Aapo Hyvärinen. Noise-Contrastive Estimation: A New Estimation Principle for Unnormalized Statistical Models. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, pages 297–304. PMLR, 2010.

[GPAM<sup>+</sup>14] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Nets. In Advances in Neural Information Processing Systems, volume 27, pages 2672–2680. Curran Associates, Inc., 2014.

[Grü24] Peter Grünwald. Beyond Neyman–Pearson: E-values Enable Hypothesis Testing with a Data-Driven Alpha. Proceedings of the National Academy of Sciences, 121(39):e2302098121, 2024.

[HBL20] Joeri Hermans, Volodimir Begy, and Gilles Louppe. Likelihood-free MCMC with Amortized Approximate Ratio Estimation. In International Conference on Machine Learning, pages 4239– 4248. PMLR, 2020.

[HFLM<sup>+</sup>19] R. Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Philip Bachman, Adam Trischler, and Yoshua Bengio. Learning Deep Representations by Mutual Information Estimation and Maximization. In International Conference on Learning Representations, 2019.

[HZ21] Alexander Henzi and Johanna F. Ziegel. Valid Post-Selection Inference with E-Values. Electronic Journal of Statistics, 2021.

[Kon23] Nick W. Koning. Post-hoc α Hypothesis Testing and the Post-hoc p-value. arXiv preprint arXiv:2312.08040, 2023.

[KRSW21] Ilmun Kim, Aaditya Ramdas, Aarti Singh, and Larry Wasserman. Classification Accuracy as a Proxy for Two-Sample Testing. The Annals of Statistics, 49(1):411–434, 2021.

[KSS10] Takafumi Kanamori, Taiji Suzuki, and Masashi Sugiyama. Theoretical Analysis of Density Ratio Estimation. IEICE Transactions on Fundamentals, 93(4):787–798, 2010.

[LC18] Alix Lhéritier and Frédéric Cazals. A sequential non-parametric multivariate two-sample test. IEEE Transactions on Information Theory, 64(5):3361–3370, 2018.

[LMAF26] Maximilian Lipp, Benjamin Kurt Miller, Lyubov Amitonova, and Patrick Forré. Generalizing Coverage Plots for Simulation-based Inference. Transactions on Machine Learning Research, 2026.

[MCF<sup>+</sup>21] Benjamin Kurt Miller, Alex Cole, Patrick Forré, Gilles Louppe, and Christoph Weniger. Truncated Marginal Neural Ratio Estimation. In Advances in Neural Information Processing Systems, volume 34, pages 17707–17726, 2021.

[MFWF23] Benjamin Kurt Miller, Marco Federici, Christoph Weniger, and Patrick Forré. Simulationbased Inference with the Generalized Kullback-Leibler Divergence. In ICML 2023 Workshop on Synergy of Scientific and Machine Learning Modeling (SynS & ML), 2023.

[Mur22] Kevin P. Murphy. Probabilistic Machine Learning: An Introduction. MIT Press, 2022.

[MWF22] Benjamin Kurt Miller, Christoph Weniger, and Patrick Forré. Contrastive Neural Ratio Estimation (for Simulation-based Inference). In Advances in Neural Information Processing Systems, volume 35, pages 4643–4658, 2022.

[NP33] Jerzy Neyman and Egon Sharpe Pearson. Ix. on the problem of the most eficient tests of statistical hypotheses. Philosophical Transactions of the Royal Society of London, Series A: Containing Papers of a Mathematical or Physical Character, 231(694-706):289–337, 02 1933.

[NWJ10] XuanLong Nguyen, Martin J. Wainwright, and Michael I. Jordan. Estimating Divergence Functionals and the Likelihood Ratio by Convex Risk Minimization. IEEE Transactions on Information Theory, 56(11):5847–5861, 2010.

[OLV18] Aäron van den Oord, Yazhe Li, and Oriol Vinyals. Representation Learning with Contrastive Predictive Coding. In arXiv preprint arXiv:1807.03748, 2018.

[PBNF24] Teodora Pandeva, Tim Bakker, Christian A. Naesseth, and Patrick Forré. E-Valuating Classifier Two-Sample Tests. Transactions on Machine Learning Research, 2024.

[PFRS24] Teodora Pandeva, Patrick Forré, Aaditya Ramdas, and Shubhanshu Shekhar. Deep Anytime-Valid Hypothesis Testing. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 622–630. PMLR, 2024.

[POvdO<sup>+</sup>19] Ben Poole, Sherjil Ozair, Aaron van den Oord, Alexander A. Alemi, and George Tucker. On Variational Bounds of Mutual Information. International Conference on Machine Learning, pages 5171–5180, 2019.

[Qin98] Jing Qin. Inferences for Case-Control and Semiparametric Two-Sample Density Ratio Models. Biometrika, 85(3):619–630, 1998.

[RGVS23] Aaditya Ramdas, Peter Grünwald, Vladimir Vovk, and Glenn Shafer. Game-Theoretic Statistics and Safe Anytime-Valid Inference. Statistical Science, 38(4):576–601, 2023.

[RW25] Aaditya Ramdas and Ruodu Wang. Hypothesis Testing with E-Values. Foundations and Trends in Statistics, 1(1–2):1–390, 2025.

[SKS<sup>+</sup>09] Masashi Sugiyama, Takafumi Kanamori, Taiji Suzuki, Shohei Hido, Jun Sese, Ichiro Takeuchi, and Liwei Wang. A Density-ratio Framework for Statistical Data Processing. IPSJ Transactions on Computer Vision and Applications, 1:183–208, 2009.

[SNK<sup>+</sup>08] Masashi Sugiyama, Shinichi Nakajima, Hisashi Kashima, Paul von Bünau, and Motoaki Kawanabe. Direct Importance Estimation with Model Selection and Its Application to Covariate Shift Adaptation. In Advances in Neural Information Processing Systems, 2008.

[SSK12] Masashi Sugiyama, Taiji Suzuki, and Takafumi Kanamori. Density Ratio Estimation in Machine Learning. Cambridge University Press, 2012.

[SSVV11] Glenn Shafer, Alexander Shen, Nikolai Vereshchagin, and Vladimir Vovk. Testing by Betting: A Strategy for Statistical and Scientific Communication. Journal of the Royal Statistical Society: Series A, 174(1):193–227, 2011.

[vdV98] Aad W. van der Vaart. Asymptotic Statistics, volume 3 of Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 1998.

[Vil39] Jean Ville. Étude Critique de la Notion de Collectif. Bull. Amer. Math. Soc, 45(11):824, 1939.

[VW21] Vladimir Vovk and Ruodu Wang. E-values: Calibration, Combination, and Applications. The Annals of Statistics, 49(3):1736–1754, 2021.

[Wal44] Abraham Wald. On Cumulative Sums of Random Variables. The Annals of Mathematical Statistics, 15(3):283–296, September 1944.

[Wal49] Abraham Wald. Note on the Consistency of the Maximum Likelihood Estimate. The Annals of Mathematical Statistics, 20(4):595–601, 1949.

[WRB20] Larry Wasserman, Aaditya Ramdas, and Sivaraman Balakrishnan. Universal Inference. Proceedings of the National Academy of Sciences, 117(29):16880–16890, 2020.

## Appendix

## A Theorems

Theorem A.1 (Ville’s inequality, see [Vil39]). Let $( W _ { t } ) _ { t \in \mathbb { N } _ { 0 } }$ be a non-negative supermartingale, $W _ { t }$ $( \Omega , \ B , P _ { 0 } )  [ 0 , \infty ] , \ t \in \mathbb { N } _ { 0 }$ , adapted to the filtration $\mathcal { F } = ( \mathcal { F } _ { t } ) _ { t \in \mathbb { N } _ { 0 } } , \mathcal { F } _ { t } \subseteq B$ and w.r.t. probability measure $P _ { 0 }$ Then for every $\alpha \in ( 0 , 1 ]$ we have the inequality:

$$
P _ { 0 } \left( \exists t \in \mathbb { N } _ { 0 } . W _ { t } > \frac { 1 } { \alpha } \right) \leq \alpha \cdot \mathbb { E } _ { P _ { 0 } } [ W _ { 0 } ] .\tag{97}
$$

Theorem A.2 (Azuma–Hoefding inequality, see $ { \left[ \mathrm { A z u 6 7 } \right] } )$ . Let $( L _ { t } ) _ { t \in \mathbb { N } _ { 0 } }$ be sequence of random variables $L _ { t } : ( \Omega , \mathcal { B } , P _ { 1 } ) \to \mathbb { R } , t \in \mathbb { N } _ { 0 } , L _ { 0 } : = 0$ , adapted to the filtration $\mathcal { F } = ( \mathcal { F } _ { t } ) _ { t \in \mathbb { N } _ { 0 } } , \mathcal { F } _ { t } \subseteq B$ and w.r.t. probability measure $P _ { 1 }$ . Further assume that for every $t \in  { \mathbb { N } } _ { 1 }$ there are constants $a _ { t } , b _ { t } , c _ { t } \in \mathbb { R }$ such that:

$$
a _ { t } \leq L _ { t } \leq b _ { t } \quad P _ { 1 } { \mathrm { - } } a . s . ,
$$

$$
c _ { t } \leq \mathbb { E } _ { P _ { 1 } } [ L _ { t } | \mathcal { F } _ { t - 1 } ] \quad P _ { 1 } { \mathrm { - } } a . s . .\tag{98}
$$

Then for every $\ell \in \mathbb { R }$ and $t \in \mathbb { N } _ { 1 }$ we have:

$$
P _ { 1 } \left( \sum _ { n = 1 } ^ { t } L _ { n } \leq \ell \right) \leq \exp \left( - \frac { 2 \left( \sum _ { n = 1 } ^ { t } c _ { n } - \ell \right) _ { + } ^ { 2 } } { \sum _ { n = 1 } ^ { t } ( b _ { n } - a _ { n } ) ^ { 2 } } \right) .\tag{99}
$$

If, in particular, $b _ { t } - a _ { t } \leq d > 0$ and $c _ { t } \geq c$ for all $t \in  { \mathbb { N } } _ { 1 }$ then we get:

$$
P _ { 1 } \left( \sum _ { n = 1 } ^ { t } L _ { n } \leq \ell \right) \leq \exp \left( - 2 t \cdot \frac { \left( c - \frac { 1 } { t } \ell \right) _ { + } ^ { 2 } } { d ^ { 2 } } \right) .\tag{100}
$$

## B Proofs

Proof B.1 (Proposition 3.2). In the following we abbreviate $\mathcal { X } : = \Omega ^ { K + 1 }$ and $x = ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \dotsc , z _ { K } ^ { 0 } ) \in \mathcal { X }$ . We abbreviate: $\mathcal { Y } : = \{ 0 , 1 , \ldots , K \}$ and the uniform distribution on Y:

$$
P ( Y = k ) : = \frac { 1 } { K + 1 } .\tag{101}
$$

For $k \in \mathcal { V }$ we then abbreviate the probability distribution on X with random variable $\boldsymbol { X } = \left( X _ { 0 } , \ldots , X _ { K } \right)$

$$
\mathbb { P } ( X | Y = k ) : = P _ { 0 } ( X _ { 0 } ) \otimes \cdots \otimes P _ { 0 } ( X _ { k - 1 } ) \otimes P _ { 1 } ( X _ { k } ) \otimes P _ { 0 } ( X _ { k + 1 } ) \otimes \cdot \otimes P _ { 0 } ( X _ { K } ) .\tag{102}
$$

We further put:

$$
Q ^ { ( K , h ) } ( Y = k | X = x ) : = \frac { h ( x _ { k } ) } { \sum _ { j = 0 } ^ { K } h ( x _ { j } ) } .\tag{103}
$$

Then we have:

$$
\mathbb { E } _ { P ( X | Y ) \otimes P ( Y ) } \left[ \log Q ^ { ( K , h ) } ( Y | X ) \right] = \frac { 1 } { K + 1 } \sum _ { k = 0 } ^ { K } \mathbb { E } _ { P ( X | Y = k ) } \left[ \log \left( \frac { h ( X _ { k } ) } { \sum _ { j = 0 } ^ { K } h ( X _ { j } ) } \right) \right]\tag{104}
$$

$$
= \mathbb { E } _ { P ( X | Y = 0 ) } \left[ \log \left( \frac { h ( X _ { 0 } ) } { \sum _ { j = 0 } ^ { K } h ( X _ { j } ) } \right) \right]\tag{105}
$$

$$
= \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \log E ^ { ( K , h ) } \right] - \log ( K + 1 ) .\tag{106}
$$

With this we get:

$$
\underset { h } { \operatorname { a r g m a x } } \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \log E ^ { ( K , h ) } \right] = \underset { h } { \operatorname { a r g m a x } } \mathbb { E } _ { P ( X | Y ) \otimes P ( Y ) } \left[ \log Q ^ { ( K , h ) } ( Y | X ) \right]\tag{107}
$$

$$
\mathbf { \Sigma } = \underset { h } { \operatorname { a r g m i n } } \mathbb { E } _ { P ( X ) } \left[ \operatorname { K L } ( P ( Y | X ) \| Q ^ { ( K , h ) } ( Y | X ) ) \right] .\tag{108}
$$

Since KL is a proper divergence, the rhs is minimized if and only if it equals zero if and only if:

$$
P ( Y | X ) = Q ^ { ( K , h ) } ( Y | X ) \qquad P ( X ) - a . s .\tag{109}
$$

Since we assumed $P _ { 1 } \ll P _ { 0 }$ , we also have $P ( X | Y = k ) \ll P _ { 0 } ^ { \otimes ( K + 1 ) }$ and $P ( X ) \ll P _ { 0 } ^ { \otimes ( K + 1 ) }$ . So we can express the $l h s$ via Bayes’s theorem in terms of densities $p ( x | k )$ and p(x):

$$
P ( Y = k | X = x ) = { \frac { p ( x | k ) } { p ( x ) } } P ( Y = k ) = { \frac { { \frac { d P _ { 1 } } { d P _ { 0 } } } ( x _ { k } ) } { { \frac { 1 } { K + 1 } } \sum _ { j = 0 } ^ { K } { \frac { d P _ { 1 } } { d P _ { 0 } } } ( x _ { j } ) } } { \frac { 1 } { K + 1 } } = { \frac { { \frac { d P _ { 1 } } { d P _ { 0 } } } ( x _ { k } ) } { \sum _ { j = 0 } ^ { K } { \frac { d P _ { 1 } } { d P _ { 0 } } } ( x _ { j } ) } } .\tag{110}
$$

$S o ,$ the condition for minimization holds if and only if for P(X)-almost-all $x \in \mathcal { X }$ and all $k \in \mathcal { V }$ we have:

$$
{ \frac { { \frac { d P _ { 1 } } { d P _ { 0 } } } ( x _ { k } ) } { \sum _ { j = 0 } ^ { K } { \frac { d P _ { 1 } } { d P _ { 0 } } } ( x _ { j } ) } } = { \frac { h ( x _ { k } ) } { \sum _ { j = 0 } ^ { K } h ( x _ { j } ) } } .\tag{111}
$$

Taking ratios for $k , l \in \mathcal { D }$ gives P(X)-almost-all $x \in \mathcal { X }$

$$
h ( x _ { k } ) \cdot \frac { d P _ { 1 } } { d P _ { 0 } } ( x _ { l } ) = h ( x _ { l } ) \cdot \frac { d P _ { 1 } } { d P _ { 0 } } ( x _ { k } ) .\tag{112}
$$

Inegrating over $P _ { 0 } ( d x _ { l } )$ gives for every $k \in \mathcal { V }$

$$
h ( x _ { k } ) = h ( x _ { k } ) \cdot \underbrace { { \mathbb { E } } _ { P _ { 0 } } \left[ { \frac { d P _ { 1 } } { d P _ { 0 } } } \right] } _ { = 1 } = \underbrace { { \mathbb { E } } _ { P _ { 0 } } \left[ h \right] } _ { = : c > 0 } \cdot { \frac { d P _ { 1 } } { d P _ { 0 } } } ( x _ { k } ) = c \cdot { \frac { d P _ { 1 } } { d P _ { 0 } } } ( x _ { k } ) ,\tag{113}
$$

which implies that for some $c > 0$ $P _ { 0 }$ -almost-all $z \in \Omega$

$$
h ( z ) = c \cdot \frac { d P _ { 1 } } { d P _ { 0 } } ( z ) .\tag{114}
$$

Note that Equation (114) also directly implies Equation (111). This then shows that such an h maximizes the e-power $\hat { \mathbb { E } _ { \mathbb { P } _ { 1 } } } \ \left[ \log E ^ { ( \dot { K } , h ) } \right]$ . This shows the equivalence.

Proof B.2 (Theorem 3.4). 1.) In the following we abbreviate $\mathcal { X } : = \Omega ^ { K + 1 }$ and $x = ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \dotsc , z _ { K } ^ { 0 } ) \in \mathcal { X }$ First note that we have for every $x \in \mathcal { X }$ the uniform bound:

$$
E ^ { ( K , \theta ) } ( x ) = \frac { \exp ( g _ { \theta } ( z ^ { 1 } ) ) } { \frac { 1 } { K + 1 } \left( \exp ( g _ { \theta } ( z ^ { 1 } ) ) + \sum _ { k = 1 } ^ { K } \exp ( g _ { \theta } ( z _ { k } ^ { 0 } ) ) \right) } \leq K + 1 .\tag{115}
$$

We further abbreviate:

$$
\ell _ { \theta } ( x ) : = \log E ^ { ( K , \theta ) } ( x ) = g _ { \theta } ( z ^ { 1 } ) - \log \left( \exp ( g _ { \theta } ( z ^ { 1 } ) ) + \sum _ { k = 1 } ^ { K } \exp ( g _ { \theta } ( z _ { k } ^ { 0 } ) ) \right) + \log ( K + 1 ) .\tag{116}
$$

Note that we have the bounds:

$$
\begin{array} { r } { \log ( K + 1 ) \ge \ell _ { \theta } ( x ) \ge g _ { \theta } ( z ^ { 1 } ) - \operatorname* { m a x } \left( g _ { \theta } ( z ^ { 1 } ) , g _ { \theta } ( z _ { 1 } ^ { 0 } ) , \dots , g _ { \theta } ( z _ { K } ^ { 0 } ) \right) } \end{array}\tag{117}
$$

$$
\geq - g _ { \theta , - } ( z ^ { 1 } ) - \operatorname* { m a x } _ { k \in [ K ] } g _ { \theta , + } ( z _ { k } ^ { 0 } ) ,\tag{118}
$$

which implies the two bounds:

$$
\ell _ { \theta , + } ( x ) \leq \log ( K + 1 ) < \infty , \qquad \ell _ { \theta , - } ( x ) \leq g _ { \theta , - } ( z ^ { 1 } ) + \operatorname* { m a x } _ { k \in [ K ] } g _ { \theta , + } ( z _ { k } ^ { 0 } ) .\tag{119}
$$

For the latter we take $\operatorname { s u p } _ { \theta \in \Theta }$ and then $\mathbb { E } _ { \mathbb { P } _ { \lambda } }$ and get:

$$
\mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \operatorname* { s u p } _ { \theta \in \Theta } \ell _ { \theta , - } ( X ) \right] \le \mathbb { E } _ { P _ { 1 } } \left[ \operatorname* { s u p } _ { \theta \in \Theta } g _ { \theta , - } ( Z ^ { 1 } ) \right] + \mathbb { E } _ { P _ { 0 } } \left[ \operatorname* { m a x } _ { k \in [ K ] } \operatorname* { s u p } _ { \theta \in \Theta } g _ { \theta , + } ( Z _ { k } ^ { 0 } ) \right]\tag{120}
$$

$$
\leq \mathbb { E } _ { P _ { 1 } } \left[ \operatorname* { s u p } _ { \theta \in \Theta } g _ { \theta , - } ( Z ^ { 1 } ) \right] + \sum _ { k \in [ K ] } \mathbb { E } _ { P _ { 0 } } \left[ \operatorname* { s u p } _ { \theta \in \Theta } g _ { \theta , + } ( Z _ { k } ^ { 0 } ) \right]\tag{121}
$$

$$
^ { 4 } _ { < } \infty ,\tag{122}
$$

where the latter holds by assumption $\it 4$ about the integrable envelopes. Together we get the finite integrable envelope bound:

$$
\mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \operatorname* { s u p } _ { \theta \in \Theta } | \ell _ { \theta } ( X ) | \right] < \infty .\tag{123}
$$

2.) Also note that by assumption ${ \mathit { 2 } } , \ \ell _ { \theta } ( x )$ is continuous in θ for every fixed $x \in \mathcal { X }$ $B y$ the integrability condition from Equation (123) we can use the dominated convergence theorem to show that also:

$$
J _ { K } ( \theta ) = \mathbb { E } _ { \mathbb { P } _ { 1 } } [ \ell _ { \theta } ( X ) ]\tag{124}
$$

is continuous in $\theta \in \Theta$ . Since Ω is compact by assumption 1 we even get that $J _ { K }$ is uniformly continuous. $\it 3 . 7$ We now want to prove the uniform law of large numbers:

$$
\operatorname* { s u p } _ { \theta \in \Theta } \left| J _ { K , N } ( \theta ) - J _ { K } ( \theta ) \right| \overset { N \to \infty } { \longrightarrow } 0 \qquad \mathbb { P } _ { 1 } - a . s .\tag{125}
$$

To show this, let $\epsilon > 0$ be fixed.

$a . )$ Now recall:

$$
J _ { K , N } ( \theta ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \ell _ { \theta } ( X _ { n } ) ,
$$

$$
J _ { K } ( \theta ) = \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ J _ { K , N } ( \theta ) \right] .\tag{126}
$$

Since $J _ { K }$ is uniformly continuous, there exists $\eta _ { 1 } > 0$ such that for all $\theta ^ { \prime } , \theta ^ { \prime } \in \Theta$ we have:

$$
d ( \theta ^ { \prime } , \theta ^ { \prime \prime } ) \leq \eta _ { 1 } \implies | J _ { K } ( \theta ^ { \prime } ) - J _ { K } ( \theta ^ { \prime \prime } ) | < \epsilon / 4 .\tag{127}
$$

b.) As a next step, for $\eta > 0$ and $x \in \mathcal { X }$ define:

$$
D _ { \eta } ( x ) : = \operatorname* { s u p } _ { \theta ^ { \prime } , \theta ^ { \prime \prime } \in \Theta \atop d ( \theta ^ { \prime } , \theta ^ { \prime \prime } ) \leq \eta } | \ell _ { \theta ^ { \prime } } ( x ) - \ell _ { \theta ^ { \prime \prime } } ( x ) | .\tag{128}
$$

Note that $D _ { \eta } ( x )$ in monotone non-decreasing in η for each fixed $x \in \mathcal { X }$ with:

$$
\operatorname* { l i m } _ { \eta  0 } D _ { \eta } ( x ) = 0 ,\tag{129}
$$

due to the uniform continuity of $\ell _ { \theta } ( x )$ in $\theta ,$ , combining the assumptions of continuity 2 and compactness 1. Further note that we have the bound:

$$
0 \leq D _ { \eta } ( x ) \leq 2 \cdot \operatorname* { s u p } _ { \theta \in \Theta } | \ell _ { \theta } ( x ) | ,\tag{130}
$$

where the upper bound is integrable by Equation (123). The dominated convergence theorem thus implies:

$$
\operatorname* { l i m } _ { \eta  0 } \mathbb { E } _ { \mathbb { P } _ { 1 } } [ D _ { \eta } ( X ) ] = 0 .\tag{131}
$$

$S o ,$ we can choose $\eta > 0$ small enough and with $\eta _ { 1 } > \eta > 0$ such that:

$$
\begin{array} { r } { \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ D _ { \eta } ( X ) \right] < \epsilon / 4 . } \end{array}\tag{132}
$$

$B y$ the strong law of large numbers we have the $\mathbb { P } _ { 1 }$ -almost-sure convergence:

$$
\frac { 1 } { N } \sum _ { n = 1 } ^ { N } D _ { \eta } ( X _ { n } ) \stackrel { N  \infty } { \longrightarrow } \mathbb { E } _ { \mathbb { P } _ { 1 } } [ D _ { \eta } ( X ) ] < \epsilon / 4 .\tag{133}
$$

$S o ,$ for $\mathbb { P } _ { 1 }$ -almost-all realizations $\omega \in \Omega$ there exists an $N _ { 0 } ( \omega ) \in \mathbb { N }$ such that for all $N \geq N _ { 0 } ( \omega )$ we have:

$$
\frac { 1 } { N } \sum _ { n = 1 } ^ { N } D _ { \eta } ( X _ { n } ( \omega ) ) < \epsilon / 2 .\tag{134}
$$

$\boldsymbol { c . } \boldsymbol { \jmath } \ B \boldsymbol { y }$ assumption 1 the space Ω is a compact metric space with some metric d. $S o ,$ , there are finitely many $\theta _ { 1 } , \dots , \theta _ { M } \in \Omega$ with:

$$
\Omega \subseteq \bigcup _ { m \in [ M ] } B ( \theta _ { m } , \eta ) ,\tag{135}
$$

where $\eta > 0$ was fixed in the previous points and $B ( \theta _ { m } , \eta )$ is the (closed) ball of radius η around $\theta _ { m }$ . $S o ,$ for every $\theta \in \Omega$ we can pick an index $j ( \theta ) \in [ M ]$ such that:

$$
d ( \theta , \theta _ { j ( \theta ) } ) \leq \eta ,\tag{136}
$$

which then implies:

$$
| J _ { K } ( \theta ) - J _ { K } ( \theta _ { j ( \theta ) } ) | < \epsilon / 4 .\tag{137}
$$

$d . )$ Now consider $\theta \in \Omega$ and $m = j ( \theta ) \in [ M ]$ . Then we have:

$$
| J _ { K , N } ( \boldsymbol \theta ) - J _ { K , N } ( \boldsymbol \theta _ { m } ) | = \left| \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( \ell _ { \boldsymbol \theta } ( X _ { n } ) - \ell _ { \boldsymbol \theta _ { m } } ( X _ { n } ) ) \right|\tag{138}
$$

$$
\leq \frac { 1 } { N } \sum _ { n = 1 } ^ { N } | \ell _ { \boldsymbol { \theta } } ( X _ { n } ) - \ell _ { \boldsymbol { \theta } _ { m } } ( X _ { n } ) |\tag{139}
$$

$$
\leq \frac { 1 } { N } \sum _ { n = 1 } ^ { N } D _ { \eta } ( X _ { n } )\tag{140}
$$

$$
< \epsilon / 2 ,\tag{141}
$$

for P -almost-all realizations $\omega \in \Omega$ and all $N \geq N _ { 0 } ( \omega )$

$e . { \big / }$ The strong law of large numbers shows that for every m $\in [ M ]$ we have the $\mathbb { P } _ { 1 }$ -almost-sure convergence:

$$
J _ { K , N } ( \theta _ { m } ) \stackrel { N  \infty } { \longrightarrow } J _ { K } ( \theta _ { m } ) ,\tag{142}
$$

and since M is finite also the $\mathbb { P } _ { 1 }$ -almost-sure convergence:

$$
\operatorname* { m a x } _ { m \in [ M ] } | J _ { K , N } ( \theta _ { m } ) - J _ { K } ( \theta _ { m } ) | \stackrel { N \to \infty } { \longrightarrow } 0 ,\tag{143}
$$

$S o ,$ for $\mathbb { P } _ { 1 }$ -almost-all realizations $\omega \in \Omega$ and all $N \geq N _ { 0 } ( \omega )$ we have:

$$
\operatorname* { m a x } _ { m \in [ M ] } | J _ { K , N } ( \theta _ { m } ) - J _ { K } ( \theta _ { m } ) | < \epsilon / 4 .\tag{144}
$$

f.) Putting everything together shows that for fixed $\epsilon > 0$ and $\mathbb { P } _ { 1 }$ -almost-all realizations $\omega \in \Omega$ , all $\theta \in \Theta$ and all $N \geq N _ { 0 } ( \omega )$ , and with $m = j ( \theta ) \in [ M ]$ , we have:

$$
\left| J _ { K , N } ( \boldsymbol \theta ) - J _ { K } ( \boldsymbol \theta ) \right| \leq \underbrace { \left| J _ { K , N } ( \boldsymbol \theta ) - J _ { K , N } ( \boldsymbol \theta _ { m } ) \right| } _ { \leq \frac { 1 } { N } \sum _ { n = 1 } ^ { N } D _ { \eta } ( X _ { n } ) < \epsilon / 2 } + \underbrace { \left| J _ { K , N } ( \boldsymbol \theta _ { m } ) - J _ { K } ( \boldsymbol \theta _ { m } ) \right| } _ { < \epsilon / 4 } + \underbrace { \left| J _ { K } ( \boldsymbol \theta _ { m } ) - J _ { K } ( \boldsymbol \theta ) \right| } _ { < \epsilon / 4 } < \epsilon .\tag{145}
$$

Note that, due to the uniform bound involving $D _ { \eta }$ , the bound $N _ { 0 } ( \omega )$ depends only on the realization $\omega \in \Omega$ (and ϵ), but crucially not on θ. We thus get:

$$
\operatorname* { s u p } _ { \theta \in \Theta } | J _ { K , N } ( \theta ) - J _ { K } ( \theta ) | \leq \epsilon .\tag{146}
$$

This thus shows the uniform law of large numbers:

$$
\Delta _ { K , N } : = \operatorname* { s u p } _ { \theta \in \Theta } \left| J _ { K , N } ( \theta ) - J _ { K } ( \theta ) \right| \overset { N \to \infty } { \longrightarrow } 0 \quad \quad \mathbb { P } _ { 1 } \cdot a . s .\tag{147}
$$

4.) We now further have the (near-)maximizing sequence $( { \widehat { \theta } } _ { K , N } ) _ { N \in \mathbb { N } }$ , i.e. for all $N \in \mathbb { N }$

$$
J _ { K , N } ( \widehat { \theta } _ { K , N } ) \geq \operatorname* { s u p } _ { \theta \in \Theta } J _ { K , N } ( \theta ) - \xi _ { K , N } \qquad \operatorname* { l i m } _ { N \to \infty } \operatorname* { s u p } _ { K , N } = \xi _ { K } \in \mathbb { R } _ { \geq 0 } \qquad \mathbb { P } _ { 1 } - a . s .\tag{148}
$$

By definition of $\Delta _ { K , N }$ we now have:

$$
\operatorname* { s u p } _ { \theta \in \Theta } J _ { K } ( \theta ) \geq J _ { K } ( \widehat { \theta } _ { K , N } ) = J _ { K , N } ( \widehat { \theta } _ { K , N } ) + J _ { K } ( \widehat { \theta } _ { K , N } ) - J _ { K , N } ( \widehat { \theta } _ { K , N } )\tag{149}
$$

$$
\begin{array} { r l } { \phantom { \sum } } & { { } \geq J _ { K , N } ( \widehat { \theta } _ { K , N } ) - \left| J _ { K } ( \widehat { \theta } _ { K , N } ) - J _ { K , N } ( \widehat { \theta } _ { K , N } ) \right| } \end{array}\tag{150}
$$

$$
\ge J _ { K , N } ( \hat { \theta } _ { K , N } ) - \Delta _ { K , N }\tag{151}
$$

$$
\ge \operatorname* { s u p } _ { \theta \in \Theta } J _ { K , N } ( \theta ) - \Delta _ { K , N } - \xi _ { K , N }\tag{152}
$$

$$
\begin{array} { l } { \displaystyle \frac { \ d \mathbf { \sigma } ( * ) } { \ d t \geq \sum \limits _ { \theta \in \Theta } J _ { K } ( \theta ) - 2 \Delta _ { K , N } - \xi _ { K , N } . } } \end{array}\tag{153}
$$

To get (∗) observe that we have:

$$
J _ { K , N } ( \theta ) = J _ { K } ( \theta ) + J _ { K , N } ( \theta ) - J _ { K } ( \theta )\tag{154}
$$

$$
\ge J _ { K } ( \theta ) - | J _ { K , N } ( \theta ) - J _ { K } ( \theta ) |\tag{155}
$$

$$
\ge J _ { K } ( \theta ) - \Delta _ { K , N } ,\tag{156}
$$

which gives:

$$
\operatorname* { s u p } _ { \theta \in \Theta } J _ { K , N } ( \theta ) \ge \operatorname* { s u p } _ { \theta \in \Theta } J _ { K } ( \theta ) - \Delta _ { K , N } .\tag{157}
$$

Altogether, we have:

$$
\operatorname* { s u p } _ { \theta \in \Theta } J _ { K } ( \theta ) \geq J _ { K } ( \widehat \theta _ { K , N } ) \geq \operatorname* { s u p } _ { \theta \in \Theta } J _ { K } ( \theta ) - 2 \Delta _ { K , N } - \xi _ { K , N } \qquad \mathbb { P } _ { 1 } - a . s . ,\tag{158}
$$

with:

$$
\operatorname* { l i m } _ { N \to \infty } \Delta _ { K , N } = 0 ,
$$

$$
\operatorname* { l i m } _ { N \to \infty } \operatorname* { s u p } _ { { \xi } _ { K , N } } = { \xi } _ { K } , \qquad { \mathbb P } _ { 1 ^ { - } } a . s . .\tag{159}
$$

This then implies:

$$
\operatorname* { l i m i n f } _ { N \to \infty } J _ { K } ( { \widehat { \theta } } _ { K , N } ) \geq \operatorname* { s u p } _ { \theta \in \Theta } J _ { K } ( \theta ) - \xi _ { K } \qquad \mathbb { P } _ { 1 } - a . s .\tag{160}
$$

This shows another main claim of Theorem $\ 3 . 4 \cdot$

${ 5 . } { \mathord {  / } } \ { g . } { \mathord { ) } }$ For the final main claim, we need to show the inequality:

$$
J _ { K } ( g ) \geq J _ { K } ( g _ { c } ^ { * } ) - { \frac { 1 } { 2 } } \epsilon ^ { 2 } , \qquad w h e n e v e r \qquad \| g - g _ { c } ^ { * } \| _ { \infty } \leq \epsilon .\tag{161}
$$

For this we introduce the curve in function space, $s \in \mathbb { R } .$

$$
g _ { s } ( z ) : = g _ { c } ^ { * } ( z ) + s \cdot v ( z ) ,
$$

$$
v ( z ) : = g ( z ) - g _ { c } ^ { * } ( z ) \in [ - \epsilon , \epsilon ] .\tag{162}
$$

and

$$
x : = ( z ^ { 1 } , z _ { 1 } ^ { 0 } , \dots , z _ { K } ^ { 0 } ) ,
$$

$$
\pi _ { k } ( s , x ) : = \frac { \exp ( g _ { s } ( x _ { k } ) ) } { \sum _ { j = 0 } ^ { K } \exp ( g _ { s } ( x _ { j } ) ) } , .\tag{163}
$$

We then introduce:

$$
J ( s ) : = J _ { K } ( g _ { s } ) = \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ g _ { s } ( X _ { 0 } ) - \log \left( \sum _ { k = 0 } ^ { K } \exp ( g _ { s } ( X _ { k } ) ) \right) \right] + \log ( K + 1 ) .\tag{164}
$$

With this we get the derivatives:

$$
J ^ { \prime } ( s ) = \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ v ( X _ { 0 } ) - \sum _ { k = 0 } ^ { K } \pi _ { k } ( s , X ) \cdot v ( X _ { k } ) \right] ,\tag{165}
$$

$$
J ^ { \prime \prime } ( s ) = - \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \sum _ { k = 0 } ^ { K } \pi _ { k } ( s , X ) \cdot \left( v ( X _ { k } ) - \sum _ { j = 0 } ^ { K } \pi _ { j } ( s , X ) \cdot v ( X _ { j } ) \right) \cdot v ( X _ { k } ) \right]\tag{166}
$$

$$
= - \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \sum _ { k = 0 } ^ { K } \pi _ { k } ( s , X ) \cdot v ( X _ { k } ) ^ { 2 } - \left( \sum _ { k = 0 } ^ { K } \pi _ { k } ( s , X ) \cdot v ( X _ { k } ) \right) ^ { 2 } \right]\tag{167}
$$

$$
= - \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \sum _ { k = 0 } ^ { K } \pi _ { k } ( s , X ) \cdot v ( X _ { k } ) ^ { 2 } \right] + \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \left( \sum _ { k = 0 } ^ { K } \pi _ { k } ( s , X ) \cdot v ( X _ { k } ) \right) ^ { 2 } \right]\tag{168}
$$

$$
\begin{array} { r l } & { \geq - \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \displaystyle \sum _ { k = 0 } ^ { K } \pi _ { k } ( s , X ) \cdot \underbrace { v ( X _ { k } ) ^ { 2 } } _ { \leq \epsilon ^ { 2 } } \right] } \\ & { \geq - \epsilon ^ { 2 } . } \end{array}\tag{169}
$$

(170)

Then note that $g _ { 0 } = g _ { c } ^ { * }$ is a maximizer of $J _ { K }$ , so at $s = 0$ we get vanishing first derivative:

$$
J ^ { \prime } ( 0 ) = 0 .\tag{171}
$$

Then we consider the Taylor expansion around $s = 0$ and get:

$$
J _ { K } ( g ) = J ( 1 ) = J ( 0 ) + \underbrace { J ^ { \prime } ( 0 ) } _ { = 0 } + \int _ { 0 } ^ { 1 } ( 1 - u ) \cdot \underbrace { J ^ { \prime \prime } ( u ) } _ { > - \epsilon ^ { 2 } } d u\tag{172}
$$

$$
\geq J ( 0 ) - \epsilon ^ { 2 } \cdot \int _ { 0 } ^ { 1 } ( 1 - u ) d u\tag{173}
$$

$$
= J ( 0 ) - \frac { 1 } { 2 } \epsilon ^ { 2 }\tag{174}
$$

$$
= J _ { K } ( g _ { c } ^ { * } ) - \frac { 1 } { 2 } \epsilon ^ { 2 } ,\tag{175}
$$

implying the inequality:

$$
J _ { K } ( g ) \geq J _ { K } ( g _ { c } ^ { * } ) - { \frac { 1 } { 2 } } \epsilon ^ { 2 } , \qquad w h e n e v e r \qquad \| g - g _ { c } ^ { * } \| _ { \infty } \leq \epsilon .\tag{176}
$$

$B y$ the approximation assumption 5 we thus get the inequalities :

$$
\operatorname { K L } ( P _ { 1 } \| P _ { 0 } ) \geq J _ { K } ( g _ { c } ^ { * } ) \geq \operatorname* { s u p } _ { \theta \in \Theta } J _ { K } ( g _ { \theta } ) \geq J _ { K } ( g _ { c } ^ { * } ) - \frac { 1 } { 2 } \epsilon ^ { 2 } .\tag{177}
$$

$h . )$ Finally, $i f$ we assume $\chi ^ { 2 } ( P _ { 1 } \| P _ { 0 } ) < \infty$ , then a simple Jensen inequality shows:

$$
\begin{array} { r } { J _ { K } ( g _ { c } ^ { * } ) = \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \log \left( \frac { \frac { d P _ { 1 } } { d P _ { 0 } } ( Z ^ { 1 } ) } { \frac { 1 } { K + 1 } \left( \frac { d P _ { 1 } } { d P _ { 0 } } ( Z ^ { 1 } ) + \sum _ { k = 1 } ^ { K } \frac { d P _ { 1 } } { d P _ { 0 } } ( Z _ { k } ^ { 0 } ) \right) } \right) \right] } \end{array}\tag{178}
$$

$$
= \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) - \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \log \left( \frac { 1 } { K + 1 } \left( \frac { d P _ { 1 } } { d P _ { 0 } } ( Z ^ { 1 } ) + \sum _ { k = 1 } ^ { K } \frac { d P _ { 1 } } { d P _ { 0 } } ( Z _ { k } ^ { 0 } ) \right) \right) \right]\tag{179}
$$

$$
\geq \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) - \log \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \frac { 1 } { K + 1 } \left( \frac { d P _ { 1 } } { d P _ { 0 } } ( Z ^ { 1 } ) + \sum _ { k = 1 } ^ { K } \frac { d P _ { 1 } } { d P _ { 0 } } ( Z _ { k } ^ { 0 } ) \right) \right]\tag{180}
$$

$$
= \mathrm { K L } ( P _ { 1 } \Vert P _ { 0 } ) - \log \left( \frac { 1 } { K + 1 } \left( \mathbb { E } _ { P _ { 1 } } \left[ \frac { d P _ { 1 } } { d P _ { 0 } } ( Z ^ { 1 } ) \right] + \sum _ { k = 1 } ^ { K } \mathbb { E } _ { P _ { 0 } } \left[ \frac { d P _ { 1 } } { d P _ { 0 } } ( Z _ { k } ^ { 0 } ) \right] \right) \right)\tag{181}
$$

$$
= \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) - \log \left( \frac { 1 } { K + 1 } \left( 1 + \chi ^ { 2 } ( P _ { 1 } \| P _ { 0 } ) + \sum _ { k = 1 } ^ { K } 1 \right) \right)
$$

$$
= \mathrm { K L } ( P _ { 1 } \| P _ { 0 } ) - \log \left( 1 + \frac { \chi ^ { 2 } ( P _ { 1 } \| P _ { 0 } ) } { K + 1 } \right) .\tag{182}
$$

(183)

With this all claims are shown in Theorem 3.4.