# ON THE DISINTEGRATION OF THE STOCHASTIC MAJORITY VOTE: FROM PAC-BAYESIAN BOUNDS TO A SELF-BOUNDING ALGORITHM

Julien Bastian Universite Jean Monnet Saint-´ Etienne, CNRS, Institut d Optique Graduate School,<sup>´</sup> Laboratoire Hubert Curien UMR 5516, F-42023, Saint-Etienne, France julien.bastian@univ-st-etienne.fr

Benjamin Leblanc, Pascal Germain Departement d’informatique et de g´ enie logiciel, Universit´ e Laval, Qu´ ebec, Canada´ benjamin.leblanc.2@ulaval.ca pascal.germain@ulaval.ca

Amaury Habrard Universite Jean Monnet Saint- ´ Etienne, CNRS, Institut d Optique Graduate School,<sup>´</sup> Laboratoire Hubert Curien UMR 5516, Inria, F-42023, Saint-Etienne, France Institut Universitaire de France amaury.habrard@univ-st-etienne.fr

Guillaume Metzler Universite Lumi´ ere Lyon 2, Universite Claude Bernard Lyon 1, ERIC, 69007, Lyon, France\` guillaume.metzler@univ-lyon2.fr

Emilie Morvant Universite Jean Monnet Saint-´ Etienne, CNRS, Institut d Optique Graduate School,<sup>´</sup> Laboratoire Hubert Curien UMR 5516, F-42023, Saint-Etienne, France emilie.morvant@univ-st-etienne.fr

Paul Viallard Univ Rennes, Inria, CNRS IRISA - UMR 6074, F35000 Rennes, France paul.viallard@inria.fr

## ABSTRACT

Weighted majority votes are central to many successful ensemble methods. PAC-Bayesian theory provides tight generalization guarantees for such models by analyzing the expected risk of stochastic classifiers, while analyzing the risk of deterministic majority votes relies on surrogate bounds. To avoid these surrogates, Zantedeschi et al. (2021) introduced guarantees for stochastic majority votes, but the resulting models remain randomized. In this paper, we propose a derandomization framework for stochastic majority votes. To do so, we apply recent advances in disintegrated PAC-Bayesian theory directly to the space of majority vote weight vectors, transforming stochastic guarantees into certificates for a single deterministic majority vote. We derive two families of high-probability generalization bounds, covering both data-independent and data-dependent constructions of the ensemble, which naturally lead to a self-bounding learning algorithm optimizing deterministic majority vote guarantees.

## 1 Introduction

Ensemble methods (Dietterich, 2000) based on weighted majority votes are among the most classical and widely used approaches in machine learning, especially when dealing with weak base learners (also called voters). Their empirical success comes from their ability to exploit voter diversity by aggregating them into a single decision rule (see, e.g., Kuncheva, 2004), such as for Bagging (Breiman, 1996), random forest (Breiman, 2001), boosting (Freund & Schapire, 1996).

The PAC-Bayesian statistical learning theory (Shawe-Taylor & Williamson, 1997; McAllester, 1998) provides a principled framework for analyzing generalization in ensemble methods. The core idea is to consider a posterior (weights) distribution over a set of voters and to provide generalization guarantees for the weighted-average risk of these voters (e.g., McAllester, 1998; Seeger, 2002; Catoni, 2007). A particularly appealing characteristic of PAC-Bayesian bounds is that they can directly lead to learning algorithms. This has been explored through the notion of self-bounding algorithms of Freund (1998) (e.g., Langford & Blum, 2003; Ambroladze et al., 2006; Dziugaite & Roy, 2017; Germain et al., 2009; Viallard et al., 2021, 2024; Atbir et al., 2026), where the training objective explicitly minimizes a generalization bound. Classical PAC-Bayesian bounds control the weighted-average risk of the voters, which coincides with the expected risk of the stochastic classifier associated with the posterior distribution (often called the Gibbs classifier). However, since practitioners are usually interested in a single deterministic classifier, an important line of research has focused on transferring PAC-Bayesian bounds from the stochastic classifier to the deterministic weighted majority vote induced by the same posterior distribution (see, e.g., Langford & Shawe-Taylor, 2002; Lacasse et al., 2006; Roy et al., 2011; Germain et al., 2015; Masegosa et al., 2020; Viallard et al., 2021). A common strategy is to upper-bound the vote’s risk using surrogate quantities relying on the stochastic classifier’s risk. The simplest example is the “factor-two” bound (stating that the majority vote’s risk is upper-bounded by twice the stochastic classifier’s risk, see Langford & Shawe-Taylor, 2002), while tighter analyses have been obtained through the binomial bound (Shawe-Taylor & Hardoon, 2009; Lacasse et al., 2010), the second-order bound (Masegosa et al., 2020) or the C-bound (Breiman, 2001; Lacasse et al., 2006; Roy et al., 2011; Germain et al., 2015; Laviolette et al., 2017; Viallard et al., 2021). Even if these approaches have led to PAC-Bayesian self-bounding algorithms (e.g., Masegosa et al., 2020; Viallard et al., 2021), the surrogate usually does not directly account for the risk of the deterministic majority vote, leading to looser generalization guarantees.

To tackle this, Zantedeschi et al. (2021) proposed a stochastic majority vote approach, where the ensemble itself is sampled from a probability distribution. This introduces a second level of randomization: instead of sampling a single voter from a posterior distribution, one samples a whole majority vote, that is, an entire set of weights. In particular, modeling the distribution over weight vectors with a Dirichlet distribution provides a flexible probabilistic model on the simplex (non-negative values summing to one) while preserving tractability. Remarkably, in binary classification, the expected 0-1 loss of the resulting stochastic majority vote admits a closed-form expression, enabling the direct optimization of the associated PAC-Bayesian bound. This leads to a self-bounding learning algorithm that directly minimizes the bound with tight certificates. Despite this advantage, the learned predictor remains stochastic: a new majority vote must be sampled at each prediction, leading to variability and making the deployment difficult.

In this paper, we extend the stochastic majority vote paradigm of Zantedeschi et al. (2021) to disintegrated PAC-Bayesian bounds, that is, bounds that control the risk of a single model drawn from the posterior distribution (Catoni, 2007; Blanchard & Fleuret, 2007). More specifically, we build on two recent advances in this line of research (Rivasplata et al., 2020; Viallard et al., 2024) to derive bounds for deterministic majority vote predictors that are optimizable. This allows us to first transform PAC-Bayesian guarantees for a stochastic model into guarantees for a deterministic one, and then to design corresponding self-bounding learning algorithms that directly optimize these guarantees. Experimental results on classification tasks show that our approach yields deterministic majority votes together with competitive PAC-Bayesian guarantees. Compared to stochastic majority votes, our method (i) produces deterministic majority votes, (ii) requires less training time, and (iii) remains competitive in terms of accuracy and certificate tightness. Moreover, it substantially improves upon classical PAC-Bayesian majority vote bounds based on surrogate anal yses. Our work provides a principled bridge between stochastic majority vote learning and deterministic certification.

Organization of the paper. Section 2 gives basics on classical and disintegrated PAC-Bayes, and Section 3 recalls PAC-Bayes analyses of deterministic and stochastic majority vote. Section 4 states our contribution, from which we derive a self-bounding algorithm in Section 5, empirically evaluated in Section 6.

## 2 Basics on PAC-Bayesian Theory

## 2.1 Classical PAC-Bayesian Bounds

We focus on classification tasks from an input space $\mathcal { X }$ to an output space Y. We denote by D the unknown data-generating distribution over $\mathcal { X } \times \mathcal { V }$ . A learning set $\textit { S } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ consists of n examples drawn i.i.d. from $\mathcal { D } ;$ we denote by $\mathcal { D } ^ { n }$ the distribution of such an n-sample. We consider a finite<sup>1</sup> set of hypotheses/voters $\mathcal { H } = \{ h _ { 1 } , \ldots , h _ { | \mathcal { H } | } \}$ , where each $h \in \mathcal H$ maps X to Y. Given $\ell : \widehat { \mathcal { V } } { \times } \mathcal { V }  [ 0 , 1 ]$ a loss function, where $\widehat { \mathcal { V } }$ denotes the prediction space, the true risk on D and empirical risk on S of a hypothesis $h \in \mathcal H$ respectively are

$$
R _ { \mathcal { D } } ( h ) = \underset { ( x , y ) \sim \mathcal { D } } { \mathbb { E } } \ell ( h ( x ) , y ) , \qquad \mathrm { a n d } \qquad \widehat { R } _ { S } ( h ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell ( h ( x _ { i } ) , y _ { i } ) .
$$

Since $\mathcal { D }$ is unknown, statistical learning theory provides tools to estimate how close the empirical risk $\widehat { R } _ { S } ( h )$ is to the true risk $R _ { \mathcal { D } } ( h )$ . In particular, Probably Approximately Correct (PAC) learning theory (Valiant, 1984) provides high-probability generalization bounds that relate the true risk to the empirical risk. Such guarantees are usually expressed through a deviation function $\Psi ( \widehat { R } _ { S } ( h ) , R _ { \cal D } ( h ) )$ , often referred to as the generalization gap. For instance, when $\Psi ( a , b ) = { \overset { \vartriangle } { | } } a - b { \vline }$ , a PAC bound takes the general form

$$
\operatorname* { l } _ { S \sim \mathcal { D } ^ { n } } \bigg [ \Big | \widehat { R } _ { S } ( h ) - R _ { \mathcal { D } } ( h ) \Big | \leq \epsilon ( h , n , \delta ) \bigg ] \geq 1 - \delta , \quad \mathrm { f o r ~ s o m e ~ p o s i t i v e ~ f u n c t i o n } \epsilon .
$$

Put into words, with high probability over the random draw of the learning set S of size n, a hypothesis will exhibit good generalization guarantees when the generalization gap between the true and the empirical risk is low, i.e., the value of $\epsilon ( h , n , \delta )$ should be as small as possible.

We focus on PAC-Bayesian generalization bounds, introduced by Shawe-Taylor & Williamson (1997); McAllester (1998). Unlike classical PAC bounds, which provide guarantees for a fixed hypothesis, PAC-Bayes studies the risk of the stochastic (Gibbs) classifier obtained by sampling, for each data point, a hypothesis from a learned distribution over H. Let M denote the set of probability distributions over its input and $\mathcal { M } ^ { * }$ the set of strictly positive distributions. PAC-Bayes relies on two distributions over H: a prior distribution $\pmb { \pi } \in \mathcal { M } ^ { * } ( \mathcal { H } )$ , encoding prior knowledge before observing S, and a learned posterior distribution $\pmb { \rho } \in \mathcal { M } ( \mathcal { H } )$ . The goal is to upper-bound the true risk of the stochastic classifier $\mathbb { E } _ { h \sim \rho } R _ { \mathcal { D } } ( h )$ through the generalization gap $\Psi \big ( \mathbb { E } _ { h \sim \rho } \widehat { R } _ { S } ( h ) , \mathbb { E } _ { h \sim \rho } R _ { \mathcal { D } } ( h ) \big )$ . We recall below the general PAC-Bayesian theorem of Germain et al. (2009).

Theorem 1 (General PAC-Bayesian Theorem of Germain et al. (2009), modified). For any distribution $\mathcal { D } ,$ hypothesis set $\mathcal { H } ,$ prior distribution $\pi \in { \mathcal { M } } ^ { * } ( { \mathcal { H } } )$ , measurablefunction $\varphi : \mathcal { H } \times ( \mathcal { X } \times \mathcal { Y } ) ^ { n } \to \mathbb { R }$ , and $\dot { \delta } \in ( 0 , 1 ]$ , we have

$$
\underset { S \sim \mathcal { D } ^ { n } } { \mathbb { P } } \left[ \forall \rho \in \mathcal { M } ( \mathcal { H } ) , \quad \underset { h \sim \rho } { \mathbb { E } } \varphi ( h , S ) \leq \mathrm { D } _ { \mathrm { K L } } ( \rho \| \pi ) + \ln \left( \frac { 1 } { \delta } \underset { S ^ { \prime } \sim \mathcal { D } ^ { n } } { \mathbb { E } } \frac { \mathbb { E } } { h ^ { \prime } \sim \pi } e ^ { \varphi ( h ^ { \prime } , S ^ { \prime } ) } \right) \right] \geq 1 - \delta ,
$$

where $\begin{array} { r } { \operatorname { D } _ { \mathrm { K L } } ( \pmb { \rho } \| \pmb { \pi } ) = \operatorname { \mathbb { E } } _ { h \sim \pmb { \rho } } \ln \frac { \pmb { \rho } ( h ) } { \pmb { \pi } ( h ) } } \end{array}$ is the Kullback-Leibler divergence between ρ and π.

Theorem 1 holds for any posterior $\pmb { \rho } \in \mathcal { M } ( \mathcal { H } )$ and is penalized by the KL divergence between $\rho$ and $\pi \cdot$ the closer $\rho$ is to π, the tighter the resulting guarantee. Importantly, from this generic result, one can recover many classical $\mathrm { P A C _ { - } }$ Bayes bounds through an appropriate choice of $\varphi ( e . g $ ., McAllester, 1998; Seeger, 2002; Catoni, 2007; Maurer, 2004). For example, choosing $\varphi ( h , S ) \triangleq n \Psi ( \widehat { R } _ { S } ( h ) , R _ { \widehat { D } } ( h ) )$ (where Ψ is convex) leads to bounds controlling the generalization gap between empirical and true risks. We now recall its instantiation with $\varphi ( h , S ) \triangleq n \mathbf { k l } ( \widehat { R } _ { S } ( h ) \| R _ { D } ( h ) )$ , where $\begin{array} { r } { \operatorname { k l } ( p \| q ) { \triangleq } p \ln { \frac { p } { q } } + ( 1 - p ) } \end{array}$ ln $\frac { 1 - p } { 1 - q }$ is the KL divergence between two Bernoulli distributions with parameters p and $q .$

Theorem 2 (Seeger (2002); Maurer (2004)). For any distribution ${ \mathcal { D } } ,$ hypothesis set $\mathcal { H } ,$ , prior distribution $\pmb { \pi } \in \mathcal { M } ^ { * } ( \mathcal { H } )$ and $\delta \in ( 0 , 1 ]$ ], we have

$$
\underset { S \sim \mathcal { P } ^ { n } } { \mathbb { P } } \left[ \forall \rho \in \mathcal { M } ( \mathcal { H } ) , \ \mathrm { ~ k l ~ } \left( \underset { h \sim \rho } { \mathbb { E } } \ \widehat { R } _ { S } ( h ) \Big | \Big | _ { h \sim \rho } R _ { \mathcal { P } } ( h ) \right) \leq \frac { 1 } { n } \left[ \mathrm { D } _ { \mathrm { K L } } ( \rho \| \pi ) + \ln \left( \frac { 2 \sqrt { n } } { \delta } \right) \right] \right] \geq 1 - \delta .
$$

PAC-Bayesian bounds are particularly appealing since (i) their data-dependent nature generally leads to tight and computable generalization bounds, and (ii) their ability to lead to algorithms that directly minimize the bound gives rise to self-bounding algorithms (Freund, 1998).

## 2.2 Disintegrated PAC-Bayesian Bounds

A limitation of classical PAC-Bayesian bounds is that they provide guarantees in expectation with respect to the posterior distribution. Consequently, they certify the performance of a stochastic classifier rather than that of a single deterministic one, the latter being often required in practice. For instance, in medical diagnosis (Naji et al., 2021), a patient must receive a single and reproducible diagnosis: randomly changing decisions would be undesirable. In response to such a lack in classical PAC-Bayesian bounds, another line of research known as disintegrated PAC-Bayes bounds was initiated by Catoni (2007); Blanchard & Fleuret (2007). Instead of bounding the risk of a stochastic classifier with respect to the posterior distribution, disintegrated PAC-Bayesian bounds upper-bound the risk of a single classifier sampled from a posterior distribution $\rho _ { S }$ learned by an algorithm $A : ( \mathcal { X } \times \mathcal { \dot { y } } ) ^ { n } \times \mathcal { M } ^ { * } ( \mathcal { H } ) \to \mathcal { M } ( \mathcal { H } )$ involving both a dataset and a prior distribution. We recall below the general disintegrated PAC-Bayesian bound proposed by Rivasplata et al. (2020) that is a derandomization of the general PAC-Bayesian Theorem 1.

Theorem 3 (General Disintegrated PAC-Bayesian Theorem of Rivasplata et al. (2020)). For any distribution ${ \mathcal { D } } ,$ hypothesis set H, prior distribution $\pi \in \mathcal { M } ^ { * } ( \mathcal { H } )$ , measurable function $\varphi : \mathcal { H } \times ( \mathcal { X } \times \mathcal { Y } ) ^ { \dot { n } }  \mathbb { R }$ , algorithm $\mathring { A } : ( \mathcal { X } \times \mathcal { Y } ) ^ { n } \times \mathcal { M } ^ { * } ( \mathcal { H } )  \mathcal { M } ( \mathcal { H } )$ , and $\delta \in ( 0 , 1 ]$ , we have

$$
\operatorname* { P } _ { \stackrel { S \sim \mathcal { D } ^ { n } } { h \sim \rho _ { S } } } \left[ \varphi ( h , S ) \leq \ln \left[ \frac { \rho _ { S } ( h ) } { \pi ( h ) } \right] + \ln \left[ \frac { 1 } { \delta S ^ { \prime } \sim \mathcal { D } ^ { n } h ^ { \prime } \sim \pi } \mathbb { E } \exp { ( \varphi ( h ^ { \prime } , S ^ { \prime } ) ) } \right] \right] \geq 1 - \delta , \quad w i t h \quad \rho _ { S } \triangleq A ( S , \pi ) .
$$

Compared to Theorem 1, the expectation over the posterior is moved outside the probability statement. Then, the bound holds with high probability for a single model h sampled from the learned posterior $\rho _ { S }$ . Moreover, the KL divergence is replaced by its “disintegrated” version that only involves the density ratio between $\rho _ { S } ( h )$ and $\pi ( h )$ More recently, Viallard et al. (2024) proposed the following alternative theorem, where the penalty is expressed through the Renyi divergence between´ $\rho _ { S }$ and π, and thus depends on the entire $\mathcal { H } .$

Theorem 4 (General Disintegrated PAC-Bayesian Theorem of Viallard et al. (2024)). Under the same assumptions as Theorem 3, assuming additionally that $\varphi : \mathcal { H } \times ( \mathcal { X } \times \mathcal { Y } ) ^ { n } \to \mathbb { R } _ { + } ^ { * }$ and $\lambda > 1$ , we have

$$
\underset { h \sim \rho _ { S } } { \underbrace { \mathbb { P } } } \left[ \frac { \lambda } { \lambda - 1 } \ln ( \varphi ( h , S ) ) \leq \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac { 2 } { \delta } + \mathsf { D } _ { \lambda } ( \rho _ { S } \parallel \pi ) + \ln \left[ \underset { S ^ { \prime } \sim \mathcal { D } ^ { n } h ^ { \prime } \sim \pi } { \mathbb { E } } \left( \varphi \left( h ^ { \prime } , S ^ { \prime } \right) ^ { \frac { \lambda } { \lambda - 1 } } \right) \right] \right] \geq 1 - \delta \int _ { \mathbb { R } } d \rho _ { S } .
$$

where $\rho _ { S } \triangleq A ( S , \pi )$ , and $\operatorname { D } _ { \lambda } ( \pmb { \rho } _ { S } \| \pmb { \pi } ) \triangleq \frac { 1 } { \lambda - 1 } \ln [ \mathbb { E } _ { h \sim \pmb { \pi } } [ \frac { \pmb { \rho } _ { S } ( h ) } { \pmb { \pi } ( h ) } ] ^ { \lambda } ]$ is the Renyi divergence.´

These results motivate a learning procedure: (i) learn a posterior $\rho _ { S }$ from $S$ using an algorithm $A ; ( i i )$ sample a single model $h \sim \rho _ { S }$ for deployment; and (iii) obtain a high-probability generalization guarantee for this specific deterministic h. In the next section, we recall another common approach in PAC-Bayes to get bounds to study the risk of the deterministic majority vote induced by the posterior distribution $\rho .$

## 3 PAC-Bayes for Majority Vote

## 3.1 The Deterministic Weighted Majority Vote

In practice, the stochastic Gibbs classifier associated with a posterior $\rho$ is rarely used directly at prediction time. Instead, one can aggregate the predictions of the voters in H according to their posterior weights, yielding a deterministic weighted majority vote, denoted by $\mathrm { M V } _ { \rho }$ . Given a posterior distribution $\rho$ over $\mathcal { H } ,$ , the weighted majority vote predicts the label with the largest total posterior weight, it is defined by

$$
\begin{array} { r } { \mathbf { M } \mathbf { V } _ { \rho } ( x ) = \underset { y \in \mathcal { V } } { \arg \operatorname* { m a x } } \left\{ \underset { h \sim \rho } { \mathbb { E } } \left[ h ( x ) = y \right] \right\} , } \end{array}\tag{1}
$$

where $[ [ a ] = 1 { \mathrm { i f } } a$ is true and 0 otherwise. Its true risk and empirical risk are, respectively,

$$
R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) = \operatorname* { \mathbb { E } } _ { ( x , y ) \sim \mathcal { D } } \ell \left( \mathbf { M } \mathbf { V } _ { \rho } ( x ) , y \right) , \quad \mathrm { a n d } \quad \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \rho } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell \left( \mathbf { M } \mathbf { V } _ { \rho } ( x _ { i } ) , y _ { i } \right) .\tag{2}
$$

Majority votes play a central role in PAC-Bayes, as they often achieve good empirical performance while still benefiting from PAC-Bayesian guarantees through their connection with the stochastic Gibbs classifier. However, the challenge is that the majority vote involves a non-differentiable aggregation of the voters, making its risk hard to analyze directly. For this reason, a part of the literature has focused on the specialization of PAC-Bayes bounds to majority vote. One of the most classical approaches consists in upper-bounding the risk of the majority vote with surrogate quantities that depend on the risk of the stochastic classifier $( e . g .$ , Langford & Shawe-Taylor, 2002; Shawe-Taylor & Hardoon, 2009; Lacasse et al., 2010, 2006; Masegosa et al., 2020). As a consequence, most PAC-Bayesian analyses do not directly control the generalization gap for the risk of the majority vote itself. Instead, they control surrogate quantities from which guarantees on the majority vote can be derived with varying degrees of tightness. We recall below the classical surrogates instantiated with the 0-1 loss, defined by $\ell ( h ( x ) , y ) = \operatorname { I } [ h ( x ) \neq y ]$

Factor-two bound (or first-order bound, Langford & Shawe-Taylor, 2002). Using the first-order Markov inequality, the risk of the majority vote satisfies

$$
R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) \leq 2 \underset { h \sim \rho } { \mathbb { E } } R _ { \mathcal { D } } ( h ) .\tag{3}
$$

This inequality relates the deterministic majority vote risk to the risk $\mathbb { E } _ { h \sim \rho } R _ { \mathcal { D } } ( h )$ Despite its simplicity, this surrogate lacks precision, since (i) when $\begin{array} { r } { \mathbb { E } _ { h \sim \rho } R _ { \mathcal { D } } ( h ) } \end{array}$ exceeds $^ { 1 / 2 , }$ , the bound exceeds 1 and becomes uninformative, and (ii) the surrogate is close to 0 only if the risk of the stochastic classifier is close to 0 too. In particular, it does not exploit the diversity or error correlations among voters in $\mathcal { H } ,$ which is key when learning with a majority vote: if all the voters were always right, then one does not need to combine them.

Binomial bound (Shawe-Taylor & Hardoon, 2009; Lacasse et al., 2010). It estimates the majority vote risk by sampling N base voters and computing the probability that at least half of them make an error:

$$
b _ { \mathcal { D } } ^ { N } ( \rho ) \triangleq \underset { ( x , y ) \sim \mathcal { D } } { \mathbb { E } } \left[ \sum _ { j = \lceil \frac { N } { 2 } \rceil } ^ { N } { \binom { N } { j } } w _ { \rho } ( x , y ) ^ { j } \left( 1 - w _ { \rho } ( x , y ) \right) ^ { ( N - j ) } \right] ,
$$

with $w _ { \pmb { \rho } } ( x , y ) \ \triangleq \ \mathbb { E } _ { h \sim \pmb { \rho } } \operatorname { I } [ h ( x ) \neq y ]$ the posterior weight assigned to misclassifying voters. As N increases, $b _ { \mathcal { D } } ^ { N }$ provides a tighter approximation of the majority vote risk, but the corresponding PAC-Bayes bound becomes looser since its penalty term scales linearly with N. The resulting surrogate relies on a factor-two inequality:

$$
R _ { \mathcal { D } } ( \mathbf { M V } _ { \rho } ) \leq 2 b _ { \mathcal { D } } ^ { N } ( \pmb { \rho } ) .\tag{4}
$$

Second-order bound (Masegosa et al., 2020). One solution to account for the diversity of voters with slightly greater precision is to use the joint error defined by

$$
\mathcal { E } _ { \mathcal { D } } ( \boldsymbol { \rho } ) = \underset { ( x , y ) \sim \mathcal { D } ( h , h ^ { \prime } ) \sim \rho ^ { 2 } } { \mathbb { E } } { \mathrm { \mathbb { E } } } [ h ( x ) \neq y ] \operatorname { I } [ h ^ { \prime } ( x ) \neq y ] .
$$

More precisely, by applying second-order Markov’s inequality, we have

$$
R _ { \mathcal { D } } ( \mathbf { M V } _ { \rho } ) \leq 4 ~ \mathcal { E } _ { \mathcal { D } } ( \rho ) .\tag{5}
$$

This surrogate reflects the idea that, for a majority vote to perform well, voters must be sufficiently diverse, and if they make mistakes, those mistakes must occur for different predictions. However, if the joint error $\mathcal { E } _ { \mathcal { D } } ( { \boldsymbol \rho } )$ exceeds $^ { 1 / 4 , }$ the surrogate exceeds 1 and is uninformative.

C-bound<sup>2</sup> (Lacasse et al., 2006). This bound involves both the joint error and the disagreement between voters, allowing the diversity/complementarity of voters to be taken into account explicitly. It follows from the Cantelli-Chebyshev inequality and is expressed as

$$
\mathrm { i f } \underset { h \sim \rho } { \mathbb { E } } R _ { \mathcal { D } } ( h ) \leq \frac { 1 } { 2 } \quad \mathrm { w e ~ h a v e } \quad R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) \leq 1 - \frac { \left( 1 - \left( 2 \mathcal { E } _ { \mathcal { D } } ( \rho ) + \mathfrak { D } _ { \mathcal { D } } ( \rho ) \right) \right) ^ { 2 } } { 1 - 2 \mathfrak { D } _ { \mathcal { D } } ( \rho ) } \triangleq \mathcal { C } _ { \mathcal { D } } ( \rho ) ,
$$

where ${ \mathfrak { D } } _ { { \mathcal { D } } } ( \rho ) = \underset { ( x , y ) \sim { \mathcal { D } } ( h , h ^ { \prime } ) \sim \rho ^ { 2 } } { \mathbb { E } } \operatorname { I } [ h ( x ) \neq h ^ { \prime } ( x ) ]$ is the disagreement.

Relationships between the surrogates. For any distribution D, for any voter set H, for any distribution $\rho$ on H, if $\begin{array} { r } { \mathbb { E } _ { h \sim \rho } R _ { \mathcal { D } } ( h ) < \frac { 1 } { 2 } } \end{array}$ , we have

$$
\begin{array} { r l } & { ( i ) \ R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) \ \leq \ \mathcal { C } _ { \mathcal { D } } ( \rho ) \ \leq \ 4 \mathcal { E } _ { \mathcal { D } } ( \rho ) \ \leq \ 2 \ \underset { h \sim \rho } { \mathbb { E } } \ R _ { \mathcal { D } } ( h ) , \quad \mathrm { i f } \ \underset { h \sim \rho } { \mathbb { E } } \ R _ { \mathcal { D } } ( h ) \ \leq \ \mathfrak { D } _ { \mathcal { D } } ( \rho ) , } \\ & { ( i i ) \ R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) \ \leq \ 2 \ \underset { h \sim \rho } { \mathbb { E } } \ R _ { \mathcal { D } } ( h ) \ \leq \ \mathcal { C } _ { \mathcal { D } } ( \rho ) \ \leq \ 4 \mathcal { E } _ { \mathcal { D } } ( \rho ) , \quad \mathrm { o t h e r w i s e } . } \end{array}
$$

Interestingly, it suggests that if the voters are sufficiently diverse, $i . e . , \mathrm { i f } \mathbb { E } _ { h \sim \rho } R _ { \mathcal { D } } ( h ) \le \mathfrak { D } _ { \mathcal { D } } ( \rho )$ , one should consider the C-bound as a surrogate to minimize.

Importantly, these approaches do not necessarily lead to tight guarantees for the majority vote, as they rely on surrogate quantities. Recently, Zantedeschi et al. (2021) proposed a different perspective allowing PAC-Bayes to reason directly on the risk of majority votes.

## 3.2 The PAC-Bayesian Stochastic Majority Vote

The idea of Zantedeschi et al. (2021) is to shift the PAC-Bayesian randomization from the voter space H to the space of majority votes itself. Instead of defining a posterior distribution over individual voters $h \in { \mathcal { H } }$ , they consider a posterior over the weight vectors that define weighted majority votes, i.e., over the weights induced by $\rho .$ Then, the stochasticity no longer comes from sampling a voter but from sampling a majority vote. Concretely, let $\mathcal { W } \subseteq \mathbb { R } ^ { | \mathcal { H } | }$ denote the set of admissible weight vectors. For any $\pmb \rho \in \mathcal W$ , a weighted majority vote of Equation (1) is

$$
\mathrm { M V } _ { \rho } ( x ) = \underset { y \in \mathcal { V } } { \arg \operatorname* { m a x } } \left\{ \sum _ { h \in \mathcal { H } } \rho _ { h } \mathrm { I } [ h ( x ) = y ] \right\} .\tag{6}
$$

Then, they consider a hyper-prior distribution $P ~ \in ~ { \mathcal { M } } ^ { * } ( { \mathcal { W } } )$ and a hyper-posterior distribution $Q \ \in \ \mathcal { M } ( \mathcal { W } )$ defined on the weight space $\mathcal { W } .$ . Sampling $\rho \sim Q$ amounts to sampling one deterministic majority vote $\mathrm { M V } _ { \rho } .$ Therefore, the stochastic majority vote is the randomized predictor obtained by drawing such a weight vector from the hyper-posterior before applying the corresponding majority vote. Its true risk is given by $\mathbb { E } _ { \pmb { \rho } \sim Q } R _ { \mathcal { D } } ( \mathbf { M V } _ { \pmb { \rho } } )$ . Under this framework, they derived the following PAC-Bayesian generalization bound.

Theorem 5 (PAC-Bayesian generalization bound for the stochastic majority vote of Zantedeschi et al. (2021)). For any distribution D,finite hypothesis set H, hyper-prior distribution $P \in \bar { \mathcal { M } } ^ { * } ( \dot { \mathcal { W } } )$ and $\delta \in ( 0 , 1 ]$ , we have

$$
\begin{array} { r } { \underset { S \sim \mathcal { D } ^ { n } } { \mathbb { P } } \Bigg [ \forall Q \in \mathcal { M } ( \mathcal { W } ) , \underset { \rho \smile Q } { \mathbb { E } } R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) \leq \mathrm { k l } ^ { - 1 } \Bigg ( \underset { \rho \smile Q } { \mathbb { E } } \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \rho } ) \Big \lVert \frac { \mathrm { D } _ { \mathrm { K L } } ( Q \lVert P ) + \ln \frac { 2 \sqrt { n } } { \delta } } { n } \Bigg ) \Bigg ] \geq 1 - \delta , } \end{array}
$$

where ${ \mathrm { k l } } ^ { - 1 } ( q \| \epsilon ) = \operatorname* { m a x } \left\{ p \in [ 0 , 1 ] \vert \mathrm { k l } ( q , p ) \leq \epsilon \right\}$ is the inverse of the binary KL divergence.

Theorem 5 is appealing since it is expressed directly in terms of majority votes, without relying on a surrogate quantity as in Section 3.1. However, the empirical risk $\mathbb { E } _ { \pmb { \rho } \sim Q } \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \rho } )$ is not straightforward to compute or optimize, since it involves an expectation over majority votes sampled from Q. To make the problem tractable, Zantedeschi et al. (2021) consider the set $\mathcal { W } \triangleq \{ \pmb { \rho } \in [ 0 , 1 ] ^ { | \mathcal { H } | } \ | \ \| \ \pmb { \rho } \| _ { 1 } = 1 \}$ and model distributions over W with Dirichlet distributions. Specifically, a weight vector is sampled as $\rho \sim \operatorname { D i r } ( \alpha )$ with $\alpha \in R _ { > 0 } ^ { | \mathcal { H } | }$ . In binary classification, this choice is particularly interesting since the expected 0-1 loss admits a closed-form expression. As a consequence, the empirical risk and the penalty can be combined into an explicit objective over the Dirichlet parameters. The hyper-posterior $Q$ can then be learned by directly minimizing the resulting PAC-Bayesian bound, leading to a self-bounding learning algorithm for stochastic majority votes.

Nevertheless, the certified predictor remains stochastic, since the guarantee controls the average risk of majority votes drawn from $Q .$ . This implies a drawback that has a consequence not only for computing the bound, but also at prediction time: for each data point, one has to sample a weight vector $\rho \sim Q$ and then use the induced majority vote $\mathrm { M V } _ { \rho } .$ In the following, we show how to derandomize the stochastic majority vote by applying disintegrated PAC-Bayesian theory directly on the weight space W.

## 4 Disintegrated stochastic majority vote

We now present our contributions in this section. To obtain a certified deterministic majority vote, we leverage the recent advances in disintegrated PAC-Bayesian theory (Section 2.2). Since each weight vector $\pmb { \rho } \in \mathcal { W }$ defines a deterministic majority vote $\mathrm { M V } _ { \rho } .$ , we apply the disintegration directly to the hyper-posterior over majority vote weights. Consequently, if a learning algorithm outputs a hyper-posterior $\dot { Q } _ { S } \in \mathcal { M } ( \dot { \mathcal { W } } )$ ), sampling $\rho \sim Q _ { S }$ amounts to selecting a single deterministic majority vote, for which disintegrated PAC-Bayesian theory provides high-probability generalization guarantees. Importantly, this construction preserves the possibility of directly optimizing the resulting bounds in a self-bounding learning procedure (see Section 5). More precisely, our objective is to transform the guarantee of Theorem 5 for a stochastic classifier into a guarantee for a single majority vote. Given a learning set $S ,$ the learner outputs a hyper-posterior $Q _ { S }$ over W, from which a single $\rho$ weight vector is drawn. We then seek a high-probability upper bound on $R _ { \mathcal { D } } ( \mathbf { M V } _ { \rho } )$

We consider two common settings in PAC-Bayes. In Section 4.1, we assume the data-independent setting, for which the set of voters H, the weight space W, and the hyper-prior $P$ are fixed independently of the learning set. This yields two disintegrated PAC-Bayesian guarantees: Theorem 6, derived from Rivasplata et al. (2020), and Theorem 7, derived from Viallard et al. (2024). Then, in Section 4.2, we extend our analysis to data-dependent voters and priors through a sample-splitting strategy, leading to Theorems 8 and 9. This allows the voters and the hyper-priors to be learned from data while preserving PAC-Bayesian validity.

## 4.1 Data-Independent Base Voters and Priors

In the context of stochastic majority votes, the standard data-independent setting means that the set of voters $\mathcal { H } ,$ from which the majority votes are built, is fixed before observing the data in S. Consequently, the weight space W is fixed as well, and the hyper-prior distribution $P \in { \mathcal { M } } ^ { * } ( \mathcal { W } )$ is chosen independently of S. The learning algorithm A then uses the learning set $S \sim \mathcal { D } ^ { n }$ to output a hyper-posterior $Q _ { S } \in \mathcal { M } ( \mathcal { W } )$ ).

Technically, the bounds derivation consists of applying the disintegrated PAC-Bayesian theorems of Section 2.2 with the weight space $\mathcal { W }$ “playing the role” of the hypothesis space. Therefore, the bounds hold for the drawing of a single weight vector $\pmb { \rho } \in \mathcal { W }$ that parametrizes the deterministic majority vote $\mathrm { M V } _ { \rho }$ . Consequently, the empirical and true risks appearing in the bounds are $\widehat { R } _ { S } ( \mathbf { M V } _ { \rho } )$ and $R _ { \mathcal { D } } ( \mathbf { M V } _ { \rho } )$ .

First, in Theorem $^ { 6 , }$ we leverage Theorem 3 (Rivasplata et al., 2020), in which the penalty term takes the form of a pointwise log-density ratio ln $\frac { Q _ { S } ( \pmb { \rho } ) } { P ( \pmb { \rho } ) }$ between the learned hyper-posterior and the hyper-prior, evaluated at the sampled weight vector $\rho .$ This term compares how likely it is to sample $\rho$ under the learned hyper-posterior and under the hyper-prior.

Theorem 6 (Disintegrated bound for stochastic majority votes). For any distribution D, finite hypothesis set ${ \mathcal { H } } ,$ majority votes weight space $\mathcal { W } \subseteq \mathbb { R } ^ { | \mathcal { H } | }$ , hyper-prior $P \in { \mathcal { M } } ^ { * } ( \mathcal { W } )$ , loss $\ell : \widehat { \mathcal { V } } \times \mathcal { V }  [ 0 , 1 ]$ , algorithm $A : ( { \mathcal { X } } { \times } { \mathcal { Y } } ) ^ { n } \ \times$ $\mathcal { M } ^ { \ast } ( \mathcal { W } ) \overset { \cdot } {  } \bar { \mathcal { M } } ( \mathcal { W } )$ , and $\delta \in ( 0 , 1 ]$ , we have

$$
\operatorname* { l i p } _ { \rho \sim \mathcal { D } _ { S } ^ { n } } \left[ \mathbf { k } \mathbf { | } ( \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \rho } ) \mathbf { | } R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) ) \leq \frac { 1 } { n } \bigg ( \ln \frac { Q _ { S } ( \rho ) } { P ( \rho ) } + \ln \frac { 2 \sqrt { n } } { \delta } \bigg ) \right] \geq 1 - \delta , \ w i t h \ Q _ { S } \triangleq A ( S , P ) .
$$

Proof. Theorem 3 with $\mathbf { \hat { \mu } } \varkappa \triangleq \mathcal { W } ^ { \prime }$ and $\varphi ( \pmb { \rho } , S ) \triangleq n \mathbf { k } \mathbf { l } \big ( \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \pmb { \rho } } ) \big \| R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \pmb { \rho } } ) \big )$ leads to

$$
\operatorname* { P } _ { s \sim \mathcal { P } _ { s } } \left[ n \operatorname { k l } ( \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \rho } ) \| R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) ) \leq \ln \left[ \frac { Q _ { S } ( \rho ) } { P ( \rho ) } \right] + \ln \left( \frac { 1 } { \delta } \operatorname* { \mathbb { E } } _ { s ^ { \prime } \sim \mathcal { P } ^ { n } } \operatorname* { \mathbb { E } } _ { \rho ^ { \prime } \sim \mathcal { P } } e ^ { n \mathsf { k l } \left( \widehat { R } _ { S ^ { \prime } } ( \mathbf { M } \mathbf { V } _ { \rho ^ { \prime } } ) \| R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho ^ { \prime } } ) \right) } \right) \right] \geq 1 - \delta .
$$

For any $\pmb { \rho } \in \mathcal { W }$ , the risk $\widehat { R } _ { S } ( \mathbf { M V } _ { \rho } )$ is the empirical mean of $i . i . d .$ random variables in [0, 1]. Thus, by applying an exponential moment inequality for the empirical KL divergence of bounded i.i.d. variables (Maurer, 2004), we have $\begin{array} { r } { \mathbb { E } _ { S ^ { \prime } \sim \mathcal { D } ^ { n } } \mathbb { E } _ { \pmb { \rho ^ { \prime } } \sim P } e ^ { n \mathbf { k l } ( \widehat { R } _ { S ^ { \prime } } ( \mathbf { M V } _ { \pmb { \rho ^ { \prime } } } ) \parallel R _ { \mathcal { D } } ( \mathbf { M V } _ { \pmb { \rho ^ { \prime } } } ) ) } \leq 2 \sqrt { n } } \end{array}$ . Plugging into the previous expression leads to

$$
\operatorname* { P } _ { S \sim \mathcal { D } ^ { n } } \left[ n \operatorname { k l } \Big ( \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \rho } ) \| R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) \Big ) \leq \ln \frac { Q _ { S } ( \rho ) } { P ( \rho ) } + \ln \frac { 2 \sqrt { n } } { \delta } \right] \geq 1 - \delta .
$$

Dividing by n concludes the proof.

Second, we derive in Theorem 7 a guarantee based on Theorem 4 (Viallard et al., 2024). Here, the penalty term is not evaluated using the sampled weight vector; instead, it depends on the Renyi divergence´ $\operatorname { D } _ { \lambda } ( Q _ { S } \| { \bar { P } } )$ between the whole distributions $Q _ { S }$ and ${ \bar { P } } .$

Theorem 7 (Renyi divergence-based disintegrated bound for stochastic majority votes)´ . For any distribution D, finite hypothesis set H, majority votes weight space $\mathcal { W } \subseteq \mathbb { R } ^ { | \mathcal { H } | }$ , hyper-prior $P \in { \mathcal { M } } ^ { * } ( \mathcal { W } )$ , loss $\ell : \widehat { \mathcal { V } } \times \mathcal { Y }  [ 0 , 1 ] , \ : \lambda > 1$ algorithm $A : ( \mathcal { X } \times \mathcal { Y } ) ^ { n } \times \mathcal { M } ^ { * } ( \mathcal { W } ) \to \bar { \mathcal { M } } ( \mathcal { W } )$ , and $\delta \in ( 0 , 1 ] ,$ , we have

$$
\underset { \rho \right. Q _ { S } } { \overset { \mathbb { P } } { \operatorname* { P } } } \left[ \mathbf { k l } \left( \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \rho } ) \left. \mathbf { \Phi } R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) \right) \leq \frac { 1 } { n } \left( \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac { 2 } { \delta } + \mathrm { D } _ { \lambda } ( Q _ { S } \Vert P ) + \mathrm { I n } ( 2 \sqrt { n } ) \right) \right] \geq 1 - \delta , w i t h \ Q _ { S } \triangleq A ( S , P ) .
$$

Proof. First, we apply Theorem 4 with $\mathbf { \hat { \mu } } ^ { \mathrm { 6 6 } } \mathcal { H } \triangleq \mathcal { W } ^ { \prime }$ and $\varphi ( \pmb { \rho } , S ) \triangleq e ^ { n \frac { \lambda - 1 } { \lambda } \mathbf { k } \mathbf { l } } \big ( \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \pmb { \rho } } ) \| R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \pmb { \rho } } ) \big )$ . For any $\pmb { \rho } ^ { \prime } \in \mathcal { W }$ , the r.v. $\widehat { R } _ { S } ( \mathbf { M V } _ { \rho ^ { \prime } } )$ is an empirical mean of i.i.d. bounded variables in $[ 0 , 1 ]$ . Hence, as in Theorem 6, we have

$$
\underset { S ^ { \prime } \sim \mathcal { D } ^ { n } } { \mathbb { E } } \ \underset { \rho ^ { \prime } \sim P } { \mathbb { E } } ( e ^ { n \frac { \lambda - 1 } { \lambda } \mathbf { k l } ( \widehat { R } _ { S ^ { \prime } } ( \mathbf { M } \mathbf { V } _ { \rho ^ { \prime } } ) )  R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho ^ { \prime } } ) ) } ) ^ { \frac { \lambda } { \lambda - 1 } } = \underset { S ^ { \prime } \sim \mathcal { D } ^ { n } } { \mathbb { E } } \ \underset { \rho ^ { \prime } \sim P } { \mathbb { E } } e ^ { n \mathbf { k l } ( \widehat { R } _ { S ^ { \prime } } ( \mathbf { M } \mathbf { V } _ { \rho ^ { \prime } } )  R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho ^ { \prime } } ) ) } \leq 2 \sqrt { n } .
$$

Combining this inequality with Theorem 4 we have

$$
\operatorname* { P } _ { \rho \sim \mathscr { D } _ { S } ^ { n } } \left[ n \mathrm { k l } \big ( \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \rho } ) \| R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) \big ) \leq \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac { 2 } { \delta } + \mathrm { D } _ { \lambda } ( Q _ { S } \| P ) + \ln ( 2 \sqrt { n } ) \right] \geq 1 - \delta .
$$

Dividing by n concludes the proof.

As mentioned earlier, the difference between Theorems 6 and 7 lies in the penalty term. Indeed, Theorem 6 involves a pointwise density ratio evaluated at the sampled weight vector, meaning its value depends only on the sampled $\rho \sim Q _ { S }$ . In particular, it can be small when $\rho$ has similar densities under the hyper-prior and the hyper-posterior, potentially leading to a tighter certificate. However, from an algorithmic point of view, this can make it more difficult to optimize. In contrast, Theorem 7 evaluates the penalty for the entire hyper-prior and hyper-posterior distributions, which generally leads to a more stable and tractable optimization objective, since it does not depend on a particular sampled weight vector. In both cases, the result is a PAC-Bayesian bound on the risk of the deterministic majority vote $\mathrm { M V } _ { \rho }$ associated with a weight vector $\rho \sim Q _ { S }$

## 4.2 Data-Dependent Base Voters and Priors

The data-independent bounds of Section 4.1 fall within the most classical PAC-Bayesian framework, which can be used when the learning task is simple enough that a fixed, data-independent family of voters can provide a sufficiently rich representation. However, for more complex tasks $( e . g .$ , multiclass classification or with high-dimensional data), initializing voters that correctly cover the feature space while allowing for every possible label would require a large number of base voters. This would increase the dimension of the hypothesis set, and thus the dimension of the associated weight space W, which may deteriorate the PAC-Bayesian bound and the ability to optimize the bound through the penalty term. To overcome this drawback, we propose, following Mhammedi et al. (2019), to consider data-dependent base voters using a cross-bounding certificate<sup>3</sup>. To do so, we split the learning set S into two disjoint subsets $\mathbf { \bar { \mathit { S } } } _ { 1 } = \{ ( x _ { i } , y _ { i } ) \} _ { i = } ^ { n _ { 1 } }$ and $\mathbf { \bar { \mathbf { \mathit { S } } } _ { 2 } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n _ { 2 } }$ for $n _ { 1 } + n _ { 2 } = n$ . The key idea is that each subset is used to learn a base voter set and a hyper-prior that will be evaluated on the other subset; for instance, $S _ { 2 }$ is used to learn the base voters $\mathcal { H } _ { 1 }$ , to define the weight space $\mathcal { W } _ { 1 }$ , and to instantiate the hyper-prior $P _ { 1 }$ . This ensures that $\mathcal { H } _ { 1 } , \mathcal { W } _ { 1 }$ and $P _ { 1 }$ are independent of $S _ { 1 }$ . Symmetrically, $S _ { 1 }$ is used to learn $\mathcal { H } _ { 2 }$ , to define $\mathcal { W } _ { 2 }$ , and to instantiate $P _ { 2 } { \mathrm { . } }$ , so that they are independent of $S _ { 2 }$ . Thus, unlike the data-independent setting, both the base classifiers and their posterior weights can be learned from data. Concretely, the algorithm (i) learns from $S _ { 1 }$ a hyper-posterior $Q _ { S _ { 1 } }$ , then samples a weight vector $\pmb { \rho } _ { 1 }$ to define the deterministic majority vote $\mathrm { M V } _ { \rho _ { 1 } }$ , and $( i i )$ learns from $S _ { 2 }$ a hyper-posterior $Q _ { S _ { 2 } }$ , then samples $\pmb { \rho } _ { 2 }$ defining $ { \mathrm { M V } _ { \rho _ { 2 } } }$ . The resulting bounds control a convex combination of the risks of the two sampled majority votes, weighted by a parameter $\tau \in [ \bar { 0 } , 1 ]$ We now extend the data-independent guarantees to data-dependent priors and base voters. More precisely, Theorem 8 extends Theorem 6, while Theorem 9 extends Theorem 7.

Theorem 8 (Data-dependent disintegrated bound for stochastic majority votes). For any distribution D, sizes $( n _ { 1 } , n _ { 2 } )$ such that $n _ { 1 } + n _ { 2 } = n ,$ finite hypothesis sets $\mathcal { H } _ { 1 }$ and $\mathcal { H } _ { 2 } ,$ majority votes weight spaces $\mathcal { W } _ { 1 } \subseteq \mathbb { R } ^ { | \mathcal { H } _ { 1 } | }$ and $\mathcal { W } _ { 2 } \subseteq \mathbb { R } ^ { | \mathcal { H } _ { 2 } | }$

On the Disintegration of the Stochastic Majority Vote: From PAC-Bayesian Bounds to a Self-Bounding Algorithm

hyper-prior distributions $P _ { 1 } ~ \in ~ { \mathcal { M } } ^ { * } ( { \mathcal { W } } _ { 1 } )$ and $P _ { 2 } ~ \in ~ \mathcal { M } ^ { * } ( \mathcal { W } _ { 2 } )$ , loss $\ell : \widehat { \mathcal { V } } \times \mathcal { Y } \ :  \ : \lceil 0 , 1 \rceil$ , algorithms $A _ { 1 } : \pentagon$ $( \mathcal { \hat { X } } { \times } \mathcal { \hat { V } } ) ^ { n _ { 1 } } { \times } \mathcal { M } ^ { * } ( \mathcal { W } _ { 1 } ) { \to } \mathcal { M } ( \mathcal { W } _ { 1 } )$ and $\dot { A } _ { 2 } : ( \mathcal { X } \times \mathcal { Y } ) ^ { n _ { 2 } } \times \mathcal { M } ^ { * } ( \mathcal { W } _ { 2 } ) {  } \mathcal { M } ( \mathcal { W } _ { 2 } )$ , and $\delta \in ( 0 , 1 ]$ and $\dot { \tau } \in [ 0 , 1 ]$ , we have

$$
\operatorname* { P } _ { \rho _ { 1 } \sim \mathcal { D } _ { S _ { 1 } } ^ { n } } \left[ \mathrm { k l } \left( \tau \widehat { R } _ { S _ { 1 } } ( \mathsf { M V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) \widehat { R } _ { S _ { 2 } } ( \mathsf { M V } _ { \rho _ { 2 } } ) \right. \tau R _ { \mathcal { D } } ( \mathsf { M V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) R _ { \mathcal { D } } ( \mathsf { M V } _ { \rho _ { 2 } } ) \right)  \\  \overset { S \sim \widehat { D } ^ { n } } { \underset { \rho _ { 1 } \sim Q _ { S _ { 1 } } } { \sim \sim } } \left[ \mathrm { k l } \left( \tau \frac { \tau } { R _ { 1 } } ( \rho _ { 1 } ) + \ln \frac { 4 \sqrt { n _ { 1 } } } { \mathcal { P } _ { 1 } ( \rho _ { 1 } ) } + \ln \frac { 1 - \tau } { \delta } \right) + \frac { 1 - \tau } { n _ { 2 } } \bigg [ \ln \frac { Q _ { S _ { 2 } } ( \rho _ { 2 } ) } { \mathcal { P } _ { 2 } ( \rho _ { 2 } ) } + \ln \frac { 4 \sqrt { n _ { 2 } } } { \delta } \bigg ] \right] \geq 1 - \delta ,
$$

where $Q _ { S _ { 1 } } \triangleq A _ { 1 } ( S _ { 1 } , P _ { 1 } )$ and $Q _ { S _ { 2 } } \triangleq A _ { 2 } ( S _ { 2 } , P _ { 2 } )$

Theorem 9 (Data-dependent Renyi-based disintegrated bound for stochastic majority votes)´ . Under the same assumptions as Theorem 8 and with $\dot { \lambda } > 1$ , we have

$$
\operatorname* { P } _ { \stackrel { \beta \to \infty } { \rho _ { 1 } \sim Q _ { s _ { 2 } } } } ^ { \mathbb { P } } [ \frac { \mathrm { k } 1 ( \tau \widehat { R } _ { S _ { 1 } } ( \mathsf { M } \mathsf { V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) \widehat { R } _ { S _ { 2 } } ( \mathsf { M } \mathsf { V } _ { \rho _ { 2 } } )   \tau R ( \mathsf { M } \mathsf { V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) R _ { \mathcal { D } } ( \mathsf { M } \mathsf { V } _ { \rho _ { 2 } } )  ) \leq } { \sigma _ { 1 } \sim Q ^ { \mathrm { B } } } ] ,
$$

Proof. The proofs of Theorems 8 and 9 are similar and follow the proof scheme of Zantedeschi et al. (2021). See, respectively, Appendices A.1 and A.2. □

The interpretation of Theorems 8 and 9 is similar to that of the data-independent case, except that the guarantee is applied separately to the two complementary splits of S. Each split is used to evaluate a majority vote whose voters and hyper-prior have been constructed from the other split.

This cross-dependence ensures that the hyper-prior remains independent of the data used to compute the empirical risk, thereby preserving PAC-Bayesian validity despite data-dependent voters. The main advantage of this construction is that it extends our framework beyond fixed hypothesis sets. In particular, the base voters can themselves be learned from data $( e . g .$ ., using random forests or other multiclass predictors), after which our guarantees apply to the resulting learned voter sets. Thus, Theorems 8 and 9 extend the disintegration principle to practical settings where both the voters and the hyper-posterior over their weights are learned from data.

In both the data-independent (Section 4.1) and data-dependent (Section 4.2) settings, the guarantees apply to deterministic majority votes. The former bounds the risk of a single sampled majority vote, whereas the latter bounds the risk of a convex combination of the risks of two sampled majority votes drawn from independent data splits. Unlike the stochastic majority vote bound of Theorem 5, neither result involves an expectation over majority votes: once the weight vectors have been sampled, the predictors are fixed and deterministic. This provides the theoretical foundation of our self-bounding algorithm introduced below.

## 5 Optimization Algorithm

A key characteristic of the bounds of Section 4 is that they can serve as objective functions to minimize during training, leading to a self-bounding procedure where the generalization bound is used as the optimization objective. Note that self-bounding algorithms have recently regained interest in PAC-Bayes (e.g., Viallard et al., 2021; Zantedeschi et al., 2021; Biggs et al., 2022; Rivasplata, 2022; Viallard, 2023; Atbir et al., 2026).

Although our disintegration analysis is not restricted to a specific family of hyper-prior and hyper-posterior distributions, its practical instantiation depends on the distribution chosen over the weight space W. The bounds apply to any pair of distributions for which the penalty term and empirical objective can be evaluated and optimized. Following Zantedeschi et al. (2021), we use Dirichlet distributions and restrict the admissible weight vectors to the simplex $\mathcal { W } = \{ \pmb { \rho } \in [ 0 , 1 ] ^ { | \mathcal { H } | } | \| \pmb { \rho } \| _ { 1 } = 1 \}$ (for completeness, we provide the instantiation in Section B). Under this choice, our penalty terms admit closed-form expressions.

Since each theorem induces a specific learning objective (Equations (14) to (17) of Section B), we denote by $B ( \alpha ; S , P , \delta )$ the bound to minimize, where α parametrizes the Dirichlet hyper-posterior, S is the learning set, $P$ the hyper-prior, and δ the confidence parameter. Algorithm 1 presents our generic learning procedure.

Algorithm 1 Self-bounding optimization of the Dirichlet hyper-posterior   
1: Input: learning set $S ,$ voters ${ \mathcal { H } } ,$ hyper-prior $P = \operatorname { D i r } ( \beta )$ with $\beta \in R _ { > 0 } ^ { | \mathcal { H } | }$ , objective $B ,$ number of epochs $T ,$   
learning rate $\eta ,$ confidence parameter δ   
2: Output: learned hyper-posterior $Q _ { S } = \operatorname { D i r } ( \alpha )$ and deterministic majority vote $\mathrm { M V } _ { \rho }$   
3: Initialize the Dirichlet parameter $\alpha \in R _ { > 0 } ^ { | \mathcal { H } | }$   
4: for each epoch $t = 1 , \ldots , T$ do   
5: for each mini-batch $\mathcal { U } \subseteq S$ do   
6: Draw a weight vector $\rho \sim \operatorname { D i r } ( \alpha )$   
7: Build the deterministic majority vote $\mathrm { M V } _ { \rho }$   
8: Compute the empirical risk $\widehat { R } _ { \mathcal { U } } ( \mathbf { M } \mathbf { V } _ { \pmb { \rho } } )$   
9: Compute the self-bounding objective $\mathbf { \dot { \theta } } ( \alpha ; \mathcal { U } , P , \delta )$   
10: Update α using gradient descent   
11: end for   
12: end for   
13: Draw a final weight vector $\rho \sim \operatorname { D i r } ( \alpha )$   
14: Return: $Q _ { S } = \bar { \mathrm { D i r } } ( \alpha )$ and $\mathrm { M V } _ { \rho }$

Once the hyper-posterior $Q _ { S }$ has been learned, the final sampled weight vector $\rho \sim Q _ { S }$ defines the deterministic classifier $\mathrm { M V } _ { \rho }$ used at prediction time. The same PAC-Bayesian guarantee that was minimized during training provides an upper bound on the true risk of this sampled deterministic majority vote. An important computational advantage of this learning procedure, when compared to the stochastic majority vote, is that it does not require evaluating the expected empirical risk over all possible majority votes (as in Theorem 5 and Zantedeschi et al. (2021)), but simply the empirical risk of the drawn majority vote. Indeed, the expected empirical risk consists of an indefinite integral and thus must be approximated using either integration estimation or the Monte Carlo technique. By contrast, computing the empirical risk of a drawn majority vote is straightforward and computationally cheap

## 6 Experiments

Compared methods. Following the experimental protocol of Zantedeschi et al. (2021), we compare our DISintegrated methods with their Stochastic Majority Vote ones. DIS-R denotes our pointwise-density-ratio-based bound of Theorem 6, DIS-V the Renyi divergence-based bound of Theorem´ 7, SMV-EXACT the closed-form stochastic majority vote objective of Zantedeschi et al. (2021), and SMV-MC its Monte Carlo approximation. Unlike SMV-EXACT and SMV-MC, which certify the expected risk of a stochastic model, DIS-R and DIS-V certify a single deterministic model sampled from the learned hyper-posterior. For completeness, we compare against three classical PAC-Bayesian majority vote methods: the First-Order method (FO) relies on the factor-two bound of Equation (3), while the Binomial method (BIN) employs the binomial bound of Equation (4) and the Second-Order method (SO) uses the joint-error bound of Equation (5). The C-bound is not included as available optimization approaches are limited to binary classification and do not readily extend to large datasets.

Datasets (see Section C.1 for details). We use binary and multiclass classification datasets from UCI Machine Learning Repository (Dua et al., 2017), LIBSVM repository, and Fashion-MNIST (Xiao et al., 2017).

Optimization. All methods are optimized using Adam (Kingma & Ba, 2015), with exponential moving-average coefficients (0.9, 0.999). The learning rate is initialized to 0.1 and is divided by 10 after 2 consecutive epochs without improvement. Training is limited to 100 epochs, with early stopping after 25 epochs without improvement. The data are split into 80% training and a 20% test set, using mini-batches of size 128 for binary datasets and 1, 024 for multiclass datasets. For SMV-MC, the expected empirical risk is estimated using 10 Monte Carlo samples. For all methods involving Monte Carlo sampling or disintegrated guarantees, the 0-1 loss is approximated during training using a sigmoid surrogate with slope parameter $c = 1 0 0$ . All bounds are computed with confidence parameter $\delta { = } 0 . 0 { \bar { 5 } }$ , and the final bound is evaluated on the entire training set using the empirical 0-1 loss. Dirichlet hyper-priors are initialized with concentration parameter $\beta = [ 0 . 5 , 0 . 5 , \bar { . . . } ]$ and the hyper-posterior parameters are initialized independently and uniformly in [0.01, 2]. For FO, SO and BIN priors are categorical distributions and posterior parameters are initialized uniformly in [0.01, 2] before normalization, and the $\mathbf { \bar { \rho } } _ { N }$ parameter of BIN is fixed at $N = 1 0 0$ . For DIS-V, the Renyi divergence order is fixed to ´ $\lambda { = } 1 . 5$ . Each experiment is repeated 10 times, and we report the mean and standard deviation of the bound value, empirical test risk, and training time.

![](images/64d8be93c66d7fd902dc5195a2f86c9ca57b11408a0ea99fa2328c38e1489af6.jpg)

![](images/4e6264113ab5ab05fccfbf9cd2d17720413af77b18bc7182ee852560fd665330.jpg)

![](images/937acab5c2ac45cd40454b693ff00ff14ec927b68d5f09e94b9ce2cf8befa4ee.jpg)

![](images/c29e1b05fbf9d539047c9da4ead1f8995faa91553af98132e51fae82d318e806.jpg)  
Figure 1: Test-error and PAC-Bayes bounds averaged over 10 runs. Vertical lines show standard deviations.

Base voters. For binary classification, we consider the data-independent setting of Section 4.1. The voter set is composed of decision stumps. For each input feature, 10 decision thresholds are evenly distributed over the range of the feature space. The resulting voter set, weight space, and hyper-prior are fixed independently of the learning set. The hyper-posterior is then learned using the complete learning set. For multiclass classification, we consider the data-dependent setting of Section 4.2. The training sample is split into two equally sized subsets, and the value of the weighting parameter<sup>4</sup> τ is set to 1/2. Each subset is used to learn the voters and hyper-prior evaluated on the complementary subset. For each split, the voters are the M = 100 decision trees of a random forest trained on the opposite subset. At each node, d features are randomly selected among the d input features, the split is chosen by minimizing the Gini impurity, and the tree depth is left unconstrained.

Results—Test error and bound tightness. We report the results in Figure 1. Overall, all methods achieve comparable test errors. In contrast, the behavior of the associated generalization bounds differs significantly between the methods that directly control the majority vote risk (DIS-R, DIS-V, SMV-EXACT, SMV-MC) and the classical surrogatebased methods (FO, SO, BIN). Across all datasets, DIS-R, DIS-V, SMV-EXACT, and SMV-MC consistently yield substantially tighter bounds than the surrogate-based methods, whose bounds even become vacuous on the PROTEIN dataset.

Comparing the stochastic majority vote with its disintegration, we observe that SMV-EXACT and SMV-MC lead to the tightest bounds, with DIS-R outperforming them only on the SPLICE dataset. This is expected, as they certify the expected risk of a stochastic predictor. Nevertheless, our proposed disintegrated approaches (DIS-R and DIS-V) remain highly competitive while providing guarantees for a single deterministic majority vote, which constitutes a substantially stronger certification. As expected from the theory, DIS-R consistently produces tighter bounds than DIS-V. Overall, these results show that certifying a single deterministic majority vote through disintegration instead of a stochastic majority vote only induces a very limited loss in bound tightness without degrading predictive performance. The complete numerical results are reported in Section C.2.

Results—Training times. We report in Figure 2 the training times of SMV-EXACT, SMV-MC, DIS-V and DIS-R. As expected, our disintegrated method DIS-V is faster on all datasets than the stochastic majority vote methods (SMV-EXACT and SMV-MC). Moreover, DIS-R is also faster on most datasets, with the exception of MUSHROOMS, PHISHING, and SPLICE, where it is the slowest method. The computational advantage of the disintegration is expected since our disintegrated objectives require optimizing only a single sampled majority vote at each mini-batch. In contrast, SMV-MC approximates an expectation using multiple sampled majority votes, while SMV-EXACT explicitly computes this expectation, resulting in a significantly higher computational cost.

Compared with DIS-V, the training time of DIS-R is more variable and is often slightly higher on the multiclass datasets. This behavior is consistent with the additional optimization difficulty introduced by its sample-dependent pointwise density-ratio term, which can lead to a less stable optimization process and longer runtime on the MUSHROOMS, PHISHING, and SPLICE datasets.

Overall, these results highlight a trade-off between optimization efficiency and certificate tightness. While DIS-V provides the most computationally efficient objective, DIS-R tends to produce tighter certificates at the price of a more challenging optimization problem.

![](images/0e52d6516a5dfe1d5e9a46965443e6b3b86befb499fe81c4d95c7446e3b14914.jpg)

![](images/450c0bcd83686ac174d5325a363d8d2b2566f6804c55f34157d28d059e123e41.jpg)

![](images/d21f10bc017db85d14c57b7211512941b6715da490805232dde65ce3f954e3e8.jpg)

![](images/4e85754dd6629a0922a3db29117297836bfca2ec3894bcbb65cf394c935d2e8e.jpg)

![](images/5ff25b1dce5e2fcb2baf1ba8074bce0f0ed3f6f7712b480c22b014b981363c22.jpg)

![](images/0a8a9f0b8270e8b98bdb2315e38ab9b5b69ba969211b06683abb09b5befa9225.jpg)  
Figure 2: Training time across datasets, averaged over 10 runs. Vertical lines show standard deviation.

## 7 Conclusion

In this paper, we introduce the first disintegrated PAC-Bayesian framework for stochastic majority votes. Unlike classical PAC-Bayesian analyses of majority votes, which rely on surrogate quantities, our framework directly certifies the true risk of deterministic majority votes. By applying recent disintegrated PAC-Bayesian theorems (Rivasplata et al., 2020; Viallard et al., 2024) to the space of majority vote weight vectors, we transform the stochastic majority vote guarantees of Zantedeschi et al. (2021) into guarantees applied to a single sampled deterministic weighted majority vote, together with a self-bounding algorithm. Empirically, our approach yields significantly tighter certificates than classical surrogate-based approaches while remaining competitive with stochastic majority votes and generally requiring less training time. A first natural extension is to investigate other families of distributions over predictor parameters beyond the Dirichlet family, which could further improve the flexibility of the learned majority votes while preserving tractable optimization.

More broadly and beyond majority votes, our work confirms that disintegrated PAC-Bayesian bounds naturally extend to higher levels of stochasticity, where the random object is no longer a base voter but an entire parameterization of the final predictor. We believe that this perspective paves the way to new PAC-Bayesian self-bounding algorithms for modern machine learning models, where randomness can be introduced over latent representations, architectures, or other high-level model parameters rather than over individual predictors.

## Acknowledgments.

This work has been partly funded by public grants from the French National Research Agency (ANR), namely the Famous project (ANR-23-CE23-0019), the DATeS project (ANR-25-CE23-6035). Pascal Germain is supported by the NSERC Discovery grant RGPIN-2020-07223. Benjamin Leblanc is supported by a Mitacs Acceleration grant, in partnership with Intact Financial Corporation. Paul Viallard is partially funded through Inria with the associate team PACTOL and the exploratory action HYPE.

## References

Ambroladze, A., Parrado-Hernandez, E., and Shawe-Taylor, J. Tighter PAC-Bayes Bounds. In´ Advances in Neural Information Processing Systems, 2006.

Atbir, H., Cherfaoui, F., Metzler, G., Morvant, E., and Viallard, P. PAC-Bayesian Bounds on Constrained f-Entropic Risk Measures. In International Conference on Artificial Intelligence and Statistics, 2026.

Biggs, F., Zantedeschi, V., and Guedj, B. On Margins and Generalisation for Voting Classifiers. In Advances in Neural Information Processing Systems, 2022.

Blanchard, G. and Fleuret, F. Occam’s Hammer. In Bshouty, N. H. and Gentile, C. (eds.), Conference on Learning Theory, Lecture Notes in Computer Science, 2007.

Breiman, L. Bagging Predictors. Machine Learning, 1996.

Breiman, L. Random forests. Machine Learning, 2001.

Catoni, O. PAC-Bayesian Supervised Classification: The Thermodynamics of Statistical Learning. Institute ofMathematical Statistics Lecture Notes Monograph Series, 2007.

Dietterich, T. G. Ensemble Methods in Machine Learning. In Multiple Classifier Systems, 2000.

Dua, D., Graff, C., et al. UCI machine learning repository, 2017.

Dziugaite, G. K. and Roy, D. M. Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data. In Conference on Uncertainty in Artificial Intelligence, 2017.

Dziugaite, G. K. and Roy, D. M. Data-dependent PAC-Bayes priors via differential privacy. In Advances in Neural Information Processing Systems, 2018.

Freund, Y. Self bounding learning algorithms. In Conference on Learning Theory, 1998.

Freund, Y. and Schapire, R. E. Experiments with a New Boosting Algorithm. In International Conference on Machine Learning, 1996.

Germain, P., Lacasse, A., Laviolette, F., and Marchand, M. PAC-Bayesian learning of linear classifiers. In International Conference on Machine Learning, 2009.

Germain, P., Lacasse, A., Laviolette, F., Marchand, M., and Roy, J. Risk bounds for the majority vote: from a PAC-Bayesian analysis to a learning algorithm. Journal ofMachine Learning Research, 2015.

Gil, M., Alajaji, F., and Linder, T. Renyi divergence measures for commonly used univariate continuous distributions.´ Information Sciences, 2013.

Kingma, D. P. and Ba, J. Adam: A Method for Stochastic Optimization. In International Conference on Learning Representations, 2015.

Kuncheva, L. I. Combining Pattern Classifiers: Methods and Algorithms. Wiley, 2004.

Lacasse, A., Laviolette, F., Marchand, M., Germain, P., and Usunier, N. PAC-Bayes Bounds for the Risk of the Majority Vote and the Variance of the Gibbs Classifier. In Advances in Neural Information Processing Systems, 2006.

Lacasse, A., Laviolette, F., Marchand, M., and Turgeon-Boutin, F. Learning with Randomized Majority Votes. In European Conference on Machine Learning and Knowledge Discovery in Databases, 2010.

Langford, J. and Blum, A. Microchoice Bounds and Self Bounding Learning Algorithms. Machine Learning, 2003.

Langford, J. and Shawe-Taylor, J. PAC-Bayes & Margins. In Advances in Neural Information Processing Systems, 2002.

Laviolette, F., Morvant, E., Ralaivola, L., and Roy, J. Risk upper bounds for general ensemble methods with an application to multiclass classification. Neurocomputing, 2017.

Letarte, G., Germain, P., Guedj, B., and Laviolette, F. Dichotomize and Generalize: PAC-Bayesian Binary Activated Deep Neural Networks. In Advances in Neural Information Processing Systems, 2019.

Masegosa, A., Lorenzen, S. S., Igel, C., and Seldin, Y. Second order PAC-Bayesian bounds for the weighted majority vote. In Advances in Neural Information Processing Systems, 2020.

Maurer, A. A note on the PAC Bayesian theorem. arXiv, cs.LG/0411099, 2004.

McAllester, D. A. Some PAC-Bayesian Theorems. Machine Learning, 1998.

Mhammedi, Z., Grunwald, P., and Guedj, B. PAC-Bayes un-expected Bernstein inequality. In¨ Advances in Neural Information Processing Systems, 2019.

Naji, M. A., Filali, S. E., Aarika, K., Benlahmar, E. H., Abdelouhahid, R. A., and Debauche, O. Machine Learning Algorithms For Breast Cancer Prediction And Diagnosis. Procedia Computer Science, 2021.

Perez-Ortiz, M., Rivasplata, O., Shawe-Taylor, J., and Szepesv´ ari, C. Tighter Risk Certificates for Neural Networks.´ Journal ofMachine Learning Research, 2021.

Rivasplata, O. PAC-Bayesian Computation. PhD thesis, University College London, United Kingdom, 2022.

Rivasplata, O., Kuzborskij, I., Szepesvari, C., and Shawe-Taylor, J. PAC-Bayes Analysis Beyond the Usual Bounds.´ In Advances in Neural Information Processing Systems, 2020.

Roy, J., Laviolette, F., and Marchand, M. From PAC-Bayes Bounds to Quadratic Programs for Majority Votes. In International Conference on Machine Learning, 2011.

Seeger, M. W. PAC-Bayesian Generalisation Error Bounds for Gaussian Process Classification. Journal of Machine Learning Research, 2002.

Shawe-Taylor, J. and Hardoon, D. Pac-bayes analysis of maximum entropy classification. In International Conference on Artificial Intelligence and Statistics, 2009.

Shawe-Taylor, J. and Williamson, R. C. A PAC Analysis of a Bayesian Estimator. In Conference on Learning Theory, 1997.

Thiemann, N., Igel, C., Wintenberger, O., and Seldin, Y. A Strongly Quasiconvex PAC-Bayesian Bound. In International Conference on Algorithmic Learning Theory, 2017.

Valiant, L. G. A Theory of the Learnable. Communications ofthe ACM, 1984.

Viallard, P. PAC-Bayesian Bounds and Beyond: Self-Bounding Algorithms and New Perspectives on Generalization in Machine Learning. PhD thesis, University Jean Monnet Saint-Etienne, France, 2023.

Viallard, P., Germain, P., Habrard, A., and Morvant, E. Self-bounding majority vote learning algorithms by the direct minimization of a tight PAC-Bayesian C-bound. In European Conference on Machine Learning and Knowledge Discovery in Databases, 2021.

Viallard, P., Germain, P., Habrard, A., and Morvant, E. A general framework for the practical disintegration of PAC-Bayesian bounds. Machine Learning, 2024.

Xiao, H., Rasul, K., and Vollgraf, R. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv, abs/1708.07747, 2017.

Zantedeschi, V., Viallard, P., Morvant, E., Emonet, R., Habrard, A., Germain, P., and Guedj, B. Learning Stochastic Majority Votes by Minimizing a PAC-Bayes Generalization Bound. In Advances in Neural Information Processing Systems, 2021.

## A Proofs of the main results

The two proofs rely on the cross-bounding argument described in Section $4 . 2$ . Recall that $\mathcal { H } _ { 1 } , ~ \mathcal { W } _ { 1 }$ , and $P _ { 1 }$ are constructed from $S _ { 2 }$ and are therefore independent of $S _ { 1 }$ . Symmetrically, $\mathcal { H } _ { 2 } , \mathcal { W } _ { 2 } .$ , and $P _ { 2 }$ are constructed from $S _ { 1 }$ and are independent of $S _ { 2 }$ . We can thus condition on the subset used to construct the voters and hyper-prior, apply the corresponding data-independent bound to the other subset with confidence parameter $\delta / 2 .$ , and combine the two resulting guarantees using a union bound and the joint convexity of the binary KL divergence.

## A.1 Proof of Theorem 8

Theorem 8 (Data-dependent disintegrated bound for stochastic majority votes). For any distribution D, sizes $( n _ { 1 } , n _ { 2 } )$ such that $n _ { 1 } + n _ { 2 } = n ,$ finite hypothesis sets $\mathcal { H } _ { 1 }$ and $\mathcal { H } _ { 2 } ,$ , majority votes weight spaces $\mathcal { W } _ { 1 } \subseteq \mathbb { R } ^ { | \mathcal { H } _ { 1 } | }$ and $\dot { \mathcal { W } } _ { 2 } \subset \mathbb { R } ^ { | \mathcal { H } _ { 2 } | }$ hyper-prior distributions $P _ { 1 } ~ \in ~ \mathcal { M } ^ { * } ( \mathcal { W } _ { 1 } )$ and $P _ { 2 } ~ \in ~ \mathcal { M } ^ { * } ( \mathcal { W } _ { 2 } )$ , loss $\ell : \widehat { \mathcal { V } } { \times } \mathcal { Y } \  \ \lceil 0 , 1 \rceil$ , algorithms $A _ { 1 } : \pentagon$ $( \bar { \mathcal { X } } { \times } \bar { \mathcal { Y } } ) ^ { n _ { 1 } } { \times } \bar { \mathcal { M } } ^ { * } ( \mathcal { W } _ { 1 } ) { \to } \bar { \mathcal { M } } ( \mathcal { W } _ { 1 } )$ and $\dot { A } _ { 2 } : ( \mathcal { X } \times \mathcal { Y } ) ^ { n _ { 2 } } \times \mathcal { M } ^ { * } ( \mathcal { W } _ { 2 } ) {  } \mathcal { M } ( \mathcal { W } _ { 2 } )$ , and $\delta \in ( 0 , 1 ]$ and $\dot { \tau } \in [ 0 , 1 ]$ , we have

$$
\operatorname* { P } _ { \rho _ { 1 } \sim \mathcal { D } _ { S _ { 1 } } ^ { n } } \left[ \mathrm { k l } \left( \tau \widehat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) \widehat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \right) \Big | \tau R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \Big ) \right] \geq 1 - \delta ,
$$

where $Q _ { S _ { 1 } } \triangleq A _ { 1 } ( S _ { 1 } , P _ { 1 } )$ and $Q _ { S _ { 2 } } \triangleq A _ { 2 } ( S _ { 2 } , P _ { 2 } )$

Proof. Recall that $\mathcal { H } _ { 1 } , \mathcal { W } _ { 1 }$ , and $P _ { 1 }$ are constructed from $S _ { 2 }$ and are therefore independent of $S _ { 1 }$ . We can apply Theorem 6 to $S _ { 1 }$ with confidence parameter $\delta / 2$ . We obtain

$$
\operatorname* { P } _ { S _ { 1 } \sim \mathcal { D } ^ { n _ { 1 } } } \left[ \mathrm { k l } \left( \widehat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \Big | \Big | R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \right) \leq \frac { 1 } { n _ { 1 } } \left[ \ln \frac { Q _ { S _ { 1 } } ( \rho _ { 1 } ) } { P _ { 1 } ( \rho _ { 1 } ) } + \ln \frac { 4 \sqrt { n _ { 1 } } } { \delta } \right] \right] \geq 1 - \frac { \delta } { 2 } .
$$

On the Disintegration of the Stochastic Majority Vote: From PAC-Bayesian Bounds to a Self-Bounding Algorithm

Adjoining the draw $\rho _ { 2 } \sim Q _ { S _ { 2 } }$ gives

$$
\operatorname* { P } _ { \stackrel { S \sim \mathcal { D } ^ { n } } { \rho _ { 1 } \sim Q _ { S _ { 1 } } } } \left[ \mathrm { k l } \left( \widehat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \Big \| R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \right) \leq \frac { 1 } { n _ { 1 } } \left[ \mathrm { l n } \frac { Q _ { S _ { 1 } } ( \rho _ { 1 } ) } { P _ { 1 } ( \rho _ { 1 } ) } + \mathrm { l n } \frac { 4 \sqrt { n _ { 1 } } } { \delta } \right] \right] \geq 1 - \frac { \delta } { 2 } .\tag{7}
$$

Symmetrically, $\mathcal { H } _ { 2 } , \mathcal { W } _ { 2 }$ , and $P _ { 2 }$ are constructed from $S _ { 1 }$ and are therefore independent of $S _ { 2 }$ , applying Theorem 6 to $S _ { 2 }$ with confidence parameter $\delta / 2$ gives

$$
\operatorname* { P } _ { S _ { 2 } \sim \mathcal { D } ^ { n _ { 2 } } } \left[ \mathrm { k l } \left( \widehat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \Big \| R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \right) \leq \frac { 1 } { n _ { 2 } } \left[ \ln \frac { Q _ { S _ { 2 } } ( \rho _ { 2 } ) } { P _ { 2 } ( \rho _ { 2 } ) } + \ln \frac { 4 \sqrt { n _ { 2 } } } { \delta } \right] \right] \geq 1 - \frac { \delta } { 2 } .
$$

Adjoining the draw $\rho _ { 1 } \sim Q _ { S _ { 1 } }$ yields

$$
\operatorname* { P } _ { \stackrel { S \sim \mathcal { D } ^ { n } } { \rho _ { 1 } \sim Q _ { S _ { 1 } } } } \left[ \mathrm { k l } \left( \widehat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \Big \| R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \right) \leq \frac { 1 } { n _ { 2 } } \left[ \mathrm { l n } \frac { Q _ { S _ { 2 } } ( \rho _ { 2 } ) } { P _ { 2 } ( \rho _ { 2 } ) } + \mathrm { l n } \frac { 4 \sqrt { n _ { 2 } } } { \delta } \right] \right] \geq 1 - \frac { \delta } { 2 } .\tag{8}
$$

By a union bound, Equations (7) and (8) hold simultaneously with probability at least $1 - \delta$ . Therefore,

$$
\operatorname* { P } _ { \stackrel { S \sim \mathcal { D } ^ { n } } { \rho _ { 1 } \sim Q _ { 3 1 } } } ^ { \mathbb { P } } [ \tau \mathrm { k l } ( \widehat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \Big | \Big | R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) ) + ( 1 - \tau ) \mathrm { k l } ( \widehat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \Big | \Big | R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) )   \\   \stackrel { S \sim \mathcal { D } ^ { n } } { \rho _ { 1 } \sim Q _ { S _ { 2 } } } | ^ { \tau } \leq \frac { \tau } { n _ { 1 } } [ \ln \frac { Q _ { S _ { 1 } } ( \rho _ { 1 } ) } { P _ { 1 } ( \rho _ { 1 } ) } + \ln \frac { 4 \sqrt { n _ { 1 } } } { \delta } ] + \frac { 1 - \tau } { n _ { 2 } } [ \ln \frac { Q _ { S _ { 2 } } ( \rho _ { 2 } ) } { P _ { 2 } ( \rho _ { 2 } ) } + \ln \frac { 4 \sqrt { n _ { 2 } } } { \delta } ] ] \geq 1 - \delta .
$$

Finally, by the convexity of the KL divergence,

$$
\begin{array} { r l } & { \tau \mathbf { k } \mathbf { l } \left( \widehat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \Big | \Big | R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \right) + ( 1 - \tau ) \mathbf { k } \mathbf { l } \left( \widehat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \Big | \Big | R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \right) } \\ & { \geq \mathbf { k } \mathbf { l } \left( \tau \widehat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) \widehat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \Big | \Big | \tau R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \right) . } \end{array}
$$

Substituting this inequality into the preceding probability statement gives the claimed result.

## A.2 Proof of Theorem 9

Theorem 9 (Data-dependent Renyi-based disintegrated bound for stochastic majority votes) ´ . Under the same assumptions as Theorem 8 and with $\lambda > 1$ , we have

$$
\operatorname* { P } _ { \stackrel { \beta \to \infty } { \rho _ { 1 } \sim Q _ { s _ { 2 } } } } ^ { \mathbb { P } } [ \frac { \mathrm { k } 1 ( \tau \widehat { R } _ { S _ { 1 } } ( \mathsf { M V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) \widehat { R } _ { S _ { 2 } } ( \mathsf { M V } _ { \rho _ { 2 } } )   \tau R ( \mathsf { M V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) R _ { \mathcal { D } } ( \mathsf { M V } _ { \rho _ { 2 } } )  ) \leq } { \sigma _ { 1 } \sim Q _ { s _ { 1 } } ^ { \mathbb { P } } } [ \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac { 4 } { \delta } + \mathrm { D } _ { \lambda } ( Q _ { S _ { 1 } } \| P _ { 1 } ) + \ln ( 2 \sqrt { n _ { 1 } } ) ] + \frac { 1 - \tau } { n _ { 2 } } [ \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac { 4 } { \delta } + \mathrm { D } _ { \lambda } ( Q _ { S _ { 2 } } \| P _ { 2 } ) + \ln ( 2 \sqrt { n _ { 2 } } ) ] ] \geq 1 - \delta .
$$

Proof. Recall that $\mathcal { H } _ { 1 } , \mathcal { W } _ { 1 }$ , and $P _ { 1 }$ are constructed from $S _ { 2 }$ and are therefore independent of $S _ { 1 }$ . We can apply Theorem $7 \mathrm { t o } S _ { 1 }$ with confidence parameter $\delta / 2$ . We obtain

$$
\operatorname* { l i p } _ { \rho _ { 1 } \sim \mathcal { D } _ { S _ { 1 } } ^ { n _ { 1 } } } \left[ \mathbf { k } \mathbf { l } \left( \widehat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \Big \| R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \right) \leq \frac { 1 } { n _ { 1 } } \left[ \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac { 4 } { \delta } + \mathbf { D } _ { \lambda } \left( Q _ { S _ { 1 } } \| P _ { 1 } \right) + \ln ( 2 \sqrt { n _ { 1 } } ) \right] \right] \geq 1 - \frac { \delta } { 2 } .
$$

Adjoining the draw $\rho _ { 2 } \sim Q _ { S _ { 2 } }$ gives

$$
\operatorname* { P } _ { \rho _ { 1 } \sim \rho _ { 2 s _ { 1 } } ^ { n } } [ \mathbf k | ( \widehat R _ { S _ { 1 } } ( \mathbf M \mathbf V _ { \rho _ { 1 } } ) ) ] R _ { D } ( \mathbf M \mathbf V _ { \rho _ { 1 } } ) ) \le \frac { 1 } { n _ { 1 } } [ \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac 4 \delta + \operatorname { D } _ { \lambda } ( Q _ { S _ { 1 } } \| P _ { 1 } ) + \ln ( 2 \sqrt { n _ { 1 } } ) ] ] \ge 1 - \frac \delta 2 .\tag{9}
$$

Symmetrically, $\mathcal { H } _ { 2 } , \mathcal { W } _ { 2 }$ , and $P _ { 2 }$ are constructed from $S _ { 1 }$ and are therefore independent of $S _ { 2 }$ , applying Theorem $7$ to $S _ { 2 }$ with confidence parameter $\delta / 2$ gives

$$
\operatorname* { l i m } _ { \substack { \rho _ { 2 } \sim \mathcal { P } _ { \infty _ { 2 } } ^ { n _ { 2 } } } } \bigg [ \mathrm { k l } \left( \widehat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \Big \| R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \right) \leq \frac { 1 } { n _ { 2 } } \left[ \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac { 4 } { \delta } + \mathrm { D } _ { \lambda } \left( Q _ { S _ { 2 } } \| P _ { 2 } \right) + \ln ( 2 \sqrt { n _ { 2 } } ) \right] \bigg ] \geq 1 - \frac { \delta } { 2 } .
$$

Adjoining the draw $\rho _ { 1 } \sim Q _ { S _ { 1 } }$ <sub>1</sub> yields

$$
\operatorname* { P } _ { \rho _ { 1 } \sim \rho _ { 2 s _ { 2 } } ^ { n } } [ \mathbf k | ( \widehat { R } _ { S _ { 2 } } ( \mathbf M \mathbf V _ { \rho _ { 2 } } ) ) ] R _ { \mathcal { D } } ( \mathbf M \mathbf V _ { \rho _ { 2 } } ) ) \le \frac { 1 } { n _ { 2 } } [ \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac 4 \delta + \mathrm { D } _ { \lambda } ( Q _ { S _ { 2 } } \| P _ { 2 } ) + \ln ( 2 \sqrt { n _ { 2 } } ) ] ] \ge 1 - \frac \delta 2 .\tag{10}
$$

By a union bound, Equations (9) and (10) hold simultaneously with probability at least $1 - \delta .$ . Therefore,

$$
\begin{array}{c} \operatorname* { l y } _ { \stackrel { S \sim \mathcal { D } ^ { n } } { \rho _ { 1 } \sim Q s _ { 1 } } } ^ { \mathbb { P } } \left[ \stackrel { \sum \mathrm { k l } } { R } \left( \stackrel { \widehat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) } { R } \Big | \Big | R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \Big ) + ( 1 - \tau ) \mathrm { k l } \left( \widehat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \Big | \Big | R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \right) \right] \right. ~  \\ { \left. \stackrel { S \sim \mathcal { D } ^ { n } } { \rho _ { 1 } \sim Q s ^ { n } } \right| \leq \frac { \tau } { n _ { 1 } } \left[ \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac { 4 } { \delta } + \mathrm { D } _ { \lambda } \left( Q s _ { 1 } \left| \right| P _ { 1 } \right) + \ln ( 2 \sqrt { n _ { 1 } } ) \right] ~ } \\ { \left. \stackrel { \rho _ { 1 } \sim Q s _ { 1 } } { \rho _ { 2 } \sim Q s _ { 2 } } \right| ~ + \frac { 1 - \tau } { n _ { 2 } } \left[ \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac { 4 } { \delta } + \mathrm { D } _ { \lambda } \left( Q s _ { 2 } \left| \right| P _ { 2 } \right) + \ln ( 2 \sqrt { n _ { 2 } } ) \right] ~ } \end{array}
$$

Finally, by the convexity of the KL divergence,

$$
\begin{array} { r l } & { \tau \mathbf { k } \mathbf { l } \left( \widehat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \Big | \Big | R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) \right) + ( 1 - \tau ) \mathbf { k } \mathbf { l } \left( \widehat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \Big | \Big | R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \right) } \\ & { \geq \mathbf { k } \mathbf { l } \left( \tau \widehat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) \widehat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \Big | \Big | \tau R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) + ( 1 - \tau ) R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \right) . } \end{array}
$$

Substituting this inequality into the preceding probability statement gives the claimed result.

## B Dirichlet Specializations and Optimization Objectives

We specialize our bounds to Dirichlet hyper-priors and hyper-posteriors, and derive the corresponding self-bounding optimization objectives for both the data-independent and data-dependent settings

## B.1 Preliminaries on Dirichlet Distributions

Let $\mathscr { W } = \{ \pmb { \rho } \in [ 0 , 1 ] ^ { | \mathscr { H } | } : \| \pmb { \rho } \| _ { 1 } = 1 \}$ . For any $\alpha \in \mathbb { R } _ { > 0 } ^ { | \mathcal { H } | }$ , the density of $\operatorname { D i r } ( \alpha )$ is

$$
\operatorname { D i r } ( \pmb { \rho } | \alpha ) = \frac { 1 } { B ( \alpha ) } \prod _ { j = 1 } ^ { | \mathcal { H } | } \rho _ { j } ^ { \alpha _ { j } - 1 } , \qquad B ( \alpha ) = \frac { \prod _ { j = 1 } ^ { | \mathcal { H } | } \Gamma ( \alpha _ { j } ) } { \Gamma \left( \sum _ { j = 1 } ^ { | \mathcal { H } | } \alpha _ { j } \right) } .
$$

Let $P = \operatorname { D i r } ( \beta )$ and $Q _ { S } = \operatorname { D i r } ( \alpha )$ . Their pointwise log-density ratio is

$$
\ln \frac { Q _ { S } ( \pmb { \rho } ) } { P ( \pmb { \rho } ) } = \ln \frac { B ( \pmb { \beta } ) } { B ( \pmb { \alpha } ) } + \sum _ { j = 1 } ^ { | \mathcal { H } | } ( \alpha _ { j } - \beta _ { j } ) \ln ( \rho _ { j } ) .\tag{11}
$$

We refer to Gil et al. (2013) for the expression of the Renyi divergence for the Dirichlet distribution. With´ $\pi = \operatorname { D i r } ( \beta )$ the prior distribution and ${ \pmb \rho } _ { S } = \mathrm { D i r } ( { \pmb \alpha } )$ we have that for any $\lambda > 1$

$$
\operatorname { D } _ { \lambda } ( \pmb { \rho } _ { S } \| \pmb { \pi } ) = \ln \frac { B ( \pmb { \beta } ) } { B ( \pmb { \alpha } ) } + \frac { 1 } { \lambda - 1 } \ln \frac { B ( \lambda \pmb { \alpha } + ( 1 - \lambda ) \pmb { \beta } ) } { B ( \pmb { \alpha } ) }\tag{12}
$$

This expression is finite provided that

$$
\lambda \alpha _ { j } + ( 1 - \lambda ) \beta _ { j } > 0 , \qquad j = 1 , \ldots , | \mathcal { H } | ,
$$

or, equivalently,

$$
\alpha _ { j } > \frac { \lambda - 1 } { \lambda } \beta _ { j } , \qquad j = 1 , \ldots , | \mathcal { H } | .
$$

Finally, we define

$$
\operatorname { k l } ^ { - 1 } ( q \| \epsilon ) \triangleq \operatorname* { m a x } \left\{ p \in [ q , 1 ] : \operatorname { k l } ( q \| p ) \leq \epsilon \right\} .
$$

Thus, $\mathop { \mathrm { k l } } ( q \| p ) \leq \epsilon$ implies $p \leq \mathbf { k l } ^ { - 1 } ( q \| \epsilon )$

## B.2 Data-Independent Setting

## B.2.1 Pointwise Density-Ratio Bound

Corollary 1 ( Theorem 6 with Dirichlet distributions ). For any distribution D, finite hypothesis set ${ \mathcal { H } } ,$ majority-vote weight space $\mathcal { W } = \{ \pmb { \rho } \in [ 0 , 1 ] ^ { | \mathcal { H } | } : \| \pmb { \rho } \| _ { 1 } = 1 \}$ , hyper-prior distribution $P = \operatorname { D i r } ( \beta )$ , hyper-posterior distribution $Q _ { S } = \operatorname { D i r } ( \alpha )$ , loss $\ell : \hat { \mathcal { y } } \times \mathcal { y }  [ 0 , 1 ]$ , algorithm $A : ( \mathcal { X } \times \mathcal { Y } ) ^ { n } \times \mathcal { M } ^ { * } ( \mathcal { W } ) \to \mathcal { M } ( \mathcal { W } )$ , and $\delta \in \mathsf { \Gamma } ( 0 , 1 ]$ , with $\alpha , \beta \in \mathbb { R } _ { > 0 } ^ { | \mathcal { H } | }$ , we have

$$
\operatorname* { P } _ { \rho \sim \mathrm { N i r } ( \alpha ) } \left[ \mathrm { k l } \left( \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \rho } ) \Big | \Big | R _ { \mathcal { P } } ( \mathbf { M } \mathbf { V } _ { \rho } ) \right) \leq \frac { 1 } { n } \left( \ln \frac { B ( \beta ) } { B ( \alpha ) } + \sum _ { j = 1 } ^ { | \mathcal { H } | } ( \alpha _ { j } - \beta _ { j } ) \ln ( \rho _ { j } ) + \ln \frac { 2 \sqrt { n } } { \delta } \right) \right] \geq 1 - \delta .
$$

where $Q _ { S } \triangleq { \mathcal { A } } ( S , P )$

Proof. Recall that the Dirichlet density is defined as

$$
\operatorname { D i r } ( \pmb { \rho } \mid \pmb { \alpha } ) = \frac { 1 } { B ( \pmb { \alpha } ) } \prod _ { j = 1 } ^ { | \mathcal { H } | } \rho _ { j } ^ { \alpha _ { j } - 1 } .
$$

Then, for any $\rho \sim \rho _ { S }$ we obtain the following decomposition:

$$
\begin{array} { l } { \displaystyle \ln \frac { \rho _ { S } ( \rho ) } { \pi ( \rho ) } = \ln \rho _ { S } ( \rho ) - \ln \pi ( \rho ) } \\ { \displaystyle \qquad = \Big [ - \ln B ( \alpha ) + \sum _ { j = 1 } ^ { | \mathcal { H } | } ( \alpha _ { j } - 1 ) \ln \rho _ { j } \Big ] - \Big [ - \ln B ( \beta ) + \sum _ { j = 1 } ^ { | \mathcal { H } | } ( \beta _ { j } - 1 ) \ln \rho _ { j } \Big ] } \\ { \displaystyle \qquad = \sum _ { j = 1 } ^ { | \mathcal { H } | } ( \alpha _ { j } - \beta _ { j } ) \ln \rho _ { j } + \big ( \ln B ( \beta ) - \ln B ( \alpha ) \big ) . } \end{array}\tag{13}
$$

Substituting Equation (13) into Theorem 6 yields the stated result.

The corresponding self-bounding optimization objective is

$$
\mathbf { k l } ^ { - 1 } ( \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \rho } ) \| \frac { 1 } { n } [ \ln \frac { B ( \beta ) } { B ( \alpha ) } + \sum _ { j = 1 } ^ { | \mathcal { H } | } ( \alpha _ { j } - \beta _ { j } ) \ln ( \rho _ { j } ) + \ln \frac { 2 \sqrt { n } } { \delta } ] ) .\tag{14}
$$

## B.2.2 Renyi Divergence-Based Bound´

Corollary 2 ( Theorem 7 with Dirichlet distributions ). For any distribution D, finite hypothesis set ${ \mathcal { H } } ,$ majority-vote weight space $\mathcal { W } = \{ \pmb { \rho } \in [ 0 , 1 ] ^ { | \mathcal { H } | } : \| \pmb { \rho } \| _ { 1 } = 1 \}$ , hyper-prior distribution $P = \operatorname { D i r } ( \beta )$ , hyper-posterior distribution $Q _ { S } = \operatorname { D i r } ( \alpha )$ , loss $\ell : \widehat { \mathcal { V } } \times \mathcal { V }  [ 0 , 1 ]$ , algorithm $A : ( \mathcal { X } \times \mathcal { Y } ) ^ { n } \times \mathcal { M } ^ { * } ( \mathcal { W } ) \to \mathcal { M } ( \mathcal { W } ) , \lambda > 1$ , and $\delta \in ( 0 , 1 ]$ , with

$\alpha , \beta \in \mathbb { R } _ { > 0 } ^ { | \mathcal { H } | }$ , we have

$$
\operatorname* { P } _ { \rho \sim \mathrm { N i r } ( \alpha ) } \left[ \mathrm { k i } \left( \widehat { R } _ { S } ( \mathbf { M } \mathbf { V } _ { \rho } ) \Big | \Big | R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho } ) \right) \leq \right. ~
$$

where $Q _ { S } \triangleq { \mathcal { A } } ( S , P )$

Proof. Substituting the Renyi divergence of Equation ( ´ 12) into Theorem 7 gives the result.

The corresponding self-bounding optimization objective is

$$
\operatorname { k l } ^ { - 1 } ( { \widehat { R } } _ { S } ( \operatorname { M V } _ { \rho } ) \ \| \ { \frac { 1 } { n } } [ { \frac { 2 \lambda - 1 } { \lambda - 1 } } \ln { \frac { 2 } { \delta } } + \ln { \frac { B ( \beta ) } { B ( \alpha ) } } + { \frac { 1 } { \lambda - 1 } } \ln { \frac { B ( \lambda \alpha + ( 1 - \lambda ) \beta ) } { B ( \alpha ) } } + \ln ( 2 { \sqrt { n } } ) ] ) .\tag{15}
$$

## B.3 Data-Dependent Setting

We consider $n _ { 1 } = n _ { 2 } = n / 2$ and $\tau = 1 / 2$ , assuming that n is even. We instantiate the hyper-priors and hyperposteriors as

$$
P _ { 1 } = \mathrm { D i r } ( \beta ^ { ( 1 ) } ) , \qquad Q _ { S _ { 1 } } = \mathrm { D i r } ( \alpha ^ { ( 1 ) } ) , \qquad P _ { 2 } = \mathrm { D i r } ( \beta ^ { ( 2 ) } ) , \qquad Q _ { S _ { 2 } } = \mathrm { D i r } ( \alpha ^ { ( 2 ) } ) ,
$$

where

$$
\begin{array} { r } { { \pmb \alpha } ^ { ( 1 ) } , \beta ^ { ( 1 ) } \in \mathbb { R } _ { > 0 } ^ { | \mathcal { H } _ { 1 } | } , \qquad { \pmb \alpha } ^ { ( 2 ) } , \beta ^ { ( 2 ) } \in \mathbb { R } _ { > 0 } ^ { | \mathcal { H } _ { 2 } | } . } \end{array}
$$

## B.3.1 Pointwise Density-Ratio Bound

Corollary 3 ( Theorem 8 with Dirichlet distributions ). For any distribution D, finite hypothesis sets $\mathcal { H } _ { 1 }$ and $\mathcal { H } _ { 2 }$ majority votes weight spaces $\mathcal { W } _ { 1 } = \{ \rho \in [ 0 , 1 ] ^ { | \mathcal { H } _ { 1 } | } : \| \rho \| _ { 1 } \stackrel { \cdot } { = } 1 \} , \mathcal { W } _ { 2 } = \{ \rho \stackrel { \cdot } { \in } [ 0 , 1 ] ^ { | \mathcal { H } _ { 2 } | } : \| \rho \| _ { 1 } = \bar { 1 } \}$ , hyperprior distributions $P _ { 1 } = \mathrm { D i r } ( \beta ^ { ( 1 ) } ) , P _ { 2 } = \mathrm { D i r } ( \beta ^ { ( 2 ) } )$ , hyper-posterior $Q _ { S _ { 1 } } = \mathrm { D i r } ( { \pmb \alpha } ^ { ( 1 ) } ) , Q _ { S _ { 2 } } = \mathrm { D i r } ( { \pmb \alpha } ^ { ( 2 ) } )$ , loss $\ell : \widehat { \mathcal { V } } \times \mathcal { V }  [ 0 , 1 ] ,$ , algorithms $A _ { 1 } : ( \mathcal { X } \times \mathcal { Y } ) ^ { n _ { 1 } } \times \mathcal { M } ^ { * } ( \mathcal { W } _ { 1 } ) \to \mathcal { M } ( \mathcal { W } _ { 1 } )$ and $A _ { 2 } : ( \mathcal { X } \times \mathcal { Y } ) ^ { n _ { 2 } } \times \mathcal { M } ^ { * } ( \mathcal { W } _ { 2 } ) \to \mathcal { M } ( \mathcal { W } _ { 2 } )$ and $\delta \in ( 0 , 1 ]$ , with $\pmb { \alpha } ^ { ( 1 ) } , \pmb { \beta } ^ { ( 1 ) } \in \mathbb { R } _ { > 0 } ^ { | \mathcal { H } _ { 1 } | } , \pmb { \alpha } ^ { ( 2 ) } , \pmb { \beta } ^ { ( 2 ) } \in \mathbb { R } _ { > 0 } ^ { | \mathcal { H } _ { 2 } | } , n _ { 1 } = n _ { 2 } = n / 2$ and $\tau = 1 / 2 ,$ , we have

$$
\begin{array}{c} \operatorname* { l i m } _ { \rho _ { 1 } \sim \operatorname { N } ( \mathbf { r } ) } [ \mathrm { k l } ( \frac { 1 } { 2 } \hat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) + \frac { 1 } { 2 } \hat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) ) ]  \frac { 1 } { 2 } R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) + \frac { 1 } { 2 } R _ { \mathcal { D } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) )  ~  \\ {  \frac { \mathbb { P } } { \rho _ { 1 } \sim \operatorname { N } ( \mathbf { r } ) }  ~ } \\ {  \rho _ { 1 } \sim \operatorname { N i r } ( \alpha ^ { ( 1 ) } ) ( \alpha ^ { ( 1 ) } )  ~ } \\ {  \rho _ { 2 } \sim \operatorname { N i r } ( \alpha ^ { ( 2 ) } ) ( } \end{array} \leq \frac { 1 } { n } [ \ln \frac { B ( \beta ^ { ( 1 ) } ) B ( \beta ^ { ( 2 ) } ) } { B ( \alpha ^ { ( 1 ) } ) B ( \alpha ^ { ( 2 ) } ) } + \sum _ { j = 1 } ^ { \vert \mathcal { H } _ { 1 } \vert } ( \alpha _ { j } ^ { ( 1 ) } - \beta _ { j } ^ { ( 1 ) } ) \ln ( \rho _ { 1 } ) _ { j } +  ~  \\   \sum _ { j = 1 } ^ { \vert \mathcal { H } _ { 2 } \vert } ( \alpha _ { j } ^ { ( 2 ) } - \beta _ { j } ^ { ( 2 ) } ) \ln ( \rho _ { 2 } ) _ { j } + \ln \frac { 8 n } { \delta ^ { 2 } } ] ] \geq 1 - \delta .
$$

where $Q _ { S _ { 1 } } \triangleq A _ { 1 } ( S _ { 1 } , P _ { 1 } )$ and $Q _ { S _ { 2 } } \triangleq A _ { 2 } ( S _ { 2 } , P _ { 2 } )$ .

Proof. Apply Equation (11) to the pairs $( Q _ { S _ { 1 } } , P _ { 1 } )$ and $( Q _ { S _ { 2 } } , P _ { 2 } )$ in Theorem 8. Since

$$
{ \frac { \tau } { n _ { 1 } } } = { \frac { 1 - \tau } { n _ { 2 } } } = { \frac { 1 } { n } } \qquad { \mathrm { a n d } } \qquad 2 \ln { \frac { 4 { \sqrt { n / 2 } } } { \delta } } = \ln { \frac { 8 n } { \delta ^ { 2 } } } ,
$$

the result follows.

The corresponding self-bounding optimization objective is

$$
\begin{array} { r l } & { \displaystyle \mathbf { k } \mathbf { l } ^ { - 1 } \Bigg ( \frac { 1 } { 2 } \widehat { R } _ { S _ { 1 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 1 } } ) + \frac { 1 } { 2 } \widehat { R } _ { S _ { 2 } } ( \mathbf { M } \mathbf { V } _ { \rho _ { 2 } } ) \Bigg | \Bigg | \frac { 1 } { n } \Bigg [ \ln \frac { B ( { \boldsymbol { \beta } } ^ { ( 1 ) } ) B ( { \boldsymbol { \beta } } ^ { ( 2 ) } ) } { B ( \alpha ^ { ( 1 ) } ) B ( \alpha ^ { ( 2 ) } ) } } \\ & { \qquad \quad + \displaystyle \sum _ { j = 1 } ^ { | \mathcal { H } _ { 1 } | } \left( \alpha _ { j } ^ { ( 1 ) } - \beta _ { j } ^ { ( 1 ) } \right) \ln ( { \boldsymbol { \rho } } ) _ { j } + \displaystyle \sum _ { j = 1 } ^ { | \mathcal { H } _ { 2 } | } \left( \alpha _ { j } ^ { ( 2 ) } - \beta _ { j } ^ { ( 2 ) } \right) \ln ( { \boldsymbol { \rho } } ) _ { j } + \ln \frac { 8 n } { \delta ^ { 2 } } \Bigg ] \Bigg ) . } \end{array}\tag{16}
$$

## B.3.2 Renyi Divergence-Based Bound´

Corollary 4 ( Theorem 9 with Dirichlet distributions ). For any distribution D, finite hypothesis sets $\mathcal { H } _ { 1 }$ and $\mathcal { H } _ { 2 }$ majority votes weight spaces $\mathcal { W } _ { 1 } = \{ \rho \in [ 0 , 1 ] ^ { | \mathcal { H } _ { 1 } | } : \| \rho \| _ { 1 } = \mathrm { i } \} , \mathcal { W } _ { 2 } = \{ \rho \in [ 0 , 1 ] ^ { | \mathcal { H } _ { 2 } | } : \| \rho \| _ { 1 } = 1 \}$ , hyper-prior distributions $P _ { 1 } = \mathrm { D i r } ( \beta ^ { ( 1 ) } ) , P _ { 2 } = \mathrm { D i r } ( \beta ^ { ( 2 ) } )$ , loss $\ell : \widehat { \mathcal { V } } \times \mathcal { V }  [ 0 , 1 ]$ , algorithms $A _ { 1 } : ( \mathcal { X } \times \mathcal { Y } ) ^ { n _ { 1 } } \times \mathcal { M } ^ { * } ( \mathcal { W } _ { 1 } ) \to$ $\mathcal { M } ( \mathcal { W } _ { 1 } )$ and $A _ { 2 } : ( \mathcal { X } \times \mathcal { Y } ) ^ { n _ { 2 } } \times \mathcal { M } ^ { * } ( \mathcal { W } _ { 2 } ) \to \mathcal { M } ( \mathcal { W } _ { 2 } )$ $\lambda > 1$ and $\delta \in ( 0 , 1 ]$ , with ${ \pmb \alpha } ^ { ( 1 ) } , { \pmb \beta } ^ { ( 1 ) } \in \mathbb { R } _ { > 0 } ^ { | \mathcal { H } _ { 1 } | } , { \pmb \alpha } ^ { ( 2 ) } , { \pmb \beta } ^ { ( 2 ) } \in $ $\mathbb { R } _ { > 0 } ^ { | \mathcal { H } _ { 2 } | } , n _ { 1 } = n _ { 2 } = n / 2 a n d \tau = 1 / 2 ,$ , we have

$$
\operatorname* { l y } _ { \rho _ { 1 } \sim \mathrm { N i r } ( \alpha ^ { ( 1 ) } ) } \left[ \mathrm { k l } \left( \frac { 1 } { 2 } \widehat { R } _ { S _ { 1 } } ( \mathsf { M V } _ { \rho _ { 1 } } ) + \frac { 1 } { 2 } \widehat { R } _ { S _ { 2 } } ( \mathsf { M V } _ { \rho _ { 2 } } ) \bigg | \bigg | \frac { 1 } { 2 } R _ { \mathcal { D } } ( \mathsf { M V } _ { \rho _ { 1 } } ) + \frac { 1 } { 2 } R _ { \mathcal { D } } ( \mathsf { M V } _ { \rho _ { 2 } } ) \right) \right.  \\ { \left. \operatorname* { l n p } _ { \rho _ { 1 } \sim \mathrm { N i r } ( \alpha ^ { ( 1 ) } ) } \right] \leq \frac { 1 } { n } \bigg [ 2 \frac { 2 \lambda - 1 } { \lambda - 1 } \ln \frac { 4 } { \delta } + \ln ( 2 n ) + \ln \frac { B ( \beta ^ { ( 1 ) } ) B ( \beta ^ { ( 2 ) } ) } { B ( \alpha ^ { ( 1 ) } ) B ( \alpha ^ { ( 2 ) } ) } \qquad \bigg ] \geq 1 - \delta . } \\ { \left. \rho _ { 2 } \sim \mathrm { N i r } ( \alpha ^ { ( 2 ) } ) \right| } \\  + \left. \frac { 1 } { \lambda - 1 } \ln \frac { B \left( \lambda \alpha ^ { ( 1 ) } + ( 1 - \lambda ) \beta ^ { ( 1 ) } \right) B \left( \lambda \alpha ^ { ( 2 ) } + ( 1 - \lambda ) \beta ^ { ( 2 ) } \right) } { B ( \alpha ^ { ( 1 ) } ) B ( \alpha ^ { ( 2 ) } ) } \bigg | \bigg | \begin{array} { l } { \sum 1 - \delta . } \\ { \sum 1 - \lambda . } \end{array} \right.
$$

where $Q _ { S _ { 1 } } \triangleq A _ { 1 } ( S _ { 1 } , P _ { 1 } )$ and $Q _ { S _ { 2 } } \triangleq A _ { 2 } ( S _ { 2 } , P _ { 2 } )$

Proof. Apply Equation (12) to the pairs $( Q _ { S _ { 1 } } , P _ { 1 } )$ and $( Q _ { S _ { 2 } } , P _ { 2 } )$ in Theorem 9. Since

$$
{ \frac { \tau } { n _ { 1 } } } = { \frac { 1 - \tau } { n _ { 2 } } } = { \frac { 1 } { n } } , \qquad \ln ( 2 \sqrt { n _ { 1 } } ) + \ln ( 2 \sqrt { n _ { 2 } } ) = \ln ( 2 n ) ,
$$

the result follows.

The corresponding self-bounding optimization objective is

$$
\begin{array} { l } { { \displaystyle { \bf k } ^ { 1 - 1 } ( \frac { 1 } { 2 } \widehat { R } _ { S _ { 1 } } ( { \bf M } { \bf V } _ { \rho _ { 1 } } ) + \frac { 1 } { 2 } \widehat { R } _ { S _ { 2 } } ( { \bf M } { \bf V } _ { \rho _ { 2 } } ) \|  \frac { 1 } { n } [ 2 \frac { 2 \lambda - 1 } { \lambda - 1 } { \ln \frac { 4 } { \delta } } + \ln ( 2 n ) + \ln \frac { B ( \beta ^ { ( 1 ) } ) B ( \beta ^ { ( 2 ) } ) } { B ( \alpha ^ { ( 1 ) } ) B ( \alpha ^ { ( 2 ) } ) }   }  }  \\ { { \displaystyle   + \frac { 1 } { \lambda - 1 } \ln \frac { B ( \lambda { \bf \alpha } ^ { ( 1 ) } + ( 1 - \lambda ) \beta ^ { ( 1 ) } ) B ( \lambda { \bf \alpha } ^ { ( 2 ) } + ( 1 - \lambda ) \beta ^ { ( 2 ) } ) } { B ( \alpha ^ { ( 1 ) } ) B ( \alpha ^ { ( 2 ) } ) } ] ) . } } \end{array}\tag{17}
$$

## C Experimental Details

## C.1 Datasets

We consider nine binary and six multiclass classification datasets obtained from UCI (Dua et al., 2017), LIBSVM, and Zalando (Xiao et al., 2017). Tables 1 and 2 report the number of instances, the number of input features after preprocessing, the number of classes, and the associated prediction task.

Table 1: Binary classification datasets.
<table><tr><td>Dataset</td><td>Source</td><td>n</td><td>d</td><td>Prediction task</td></tr><tr><td>Haberman</td><td>UCI</td><td>306</td><td>3</td><td>Survival after surgery.</td></tr><tr><td>TicTacToe</td><td>UCI</td><td>958</td><td>9</td><td>Winning configurations for player x.</td></tr><tr><td>Svmguide1</td><td>LIBSVM</td><td>7,089</td><td>4</td><td>Binary classification without further descrip- tion.</td></tr><tr><td>Mushrooms</td><td>UCI</td><td>8,124</td><td>22</td><td>Edibility from categorical appearance features.</td></tr><tr><td>Phishing</td><td>LIBSVM</td><td>2,456</td><td>68</td><td>Detection of phishing websites.</td></tr><tr><td>Adult</td><td>LIBSVM</td><td>32,561</td><td>123</td><td>Prediction of whether annual income exceeds $50K.</td></tr><tr><td>CodRNA</td><td>LIBSVM</td><td>59,535</td><td>8</td><td>Detection of non-coding RNAs.</td></tr><tr><td>Splice</td><td>UCI</td><td>3,190</td><td>60</td><td>Prediction of DNA splice junctions.</td></tr><tr><td>Australian</td><td>LIBSVM</td><td>690</td><td>14</td><td>Credit approval prediction.</td></tr></table>

Table 2: Multiclass classification datasets.
<table><tr><td>Dataset</td><td>Source</td><td>n</td><td> $d$ </td><td>Classes</td><td>Prediction task</td></tr><tr><td>Pendigits</td><td>UCI</td><td>10,992</td><td>16</td><td>10</td><td>Recognition of handwritten digits.</td></tr><tr><td>Protein</td><td>LIBSVM</td><td>24,387</td><td>357</td><td>3</td><td>Classification of protein structures.</td></tr><tr><td>Shuttle</td><td>UCI</td><td>58,000</td><td>9</td><td>7</td><td>Classification of space-shuttle condi- tions.</td></tr><tr><td>Sensorless</td><td>LIBSVM</td><td>58,509</td><td>48</td><td>11</td><td>Prediction of motor operating condi-</td></tr><tr><td>MNIST</td><td>LIBSVM</td><td>70,000</td><td>784</td><td>10</td><td>tions. Recognition of handwritten digits.</td></tr><tr><td>Fashion-MNIST</td><td>Zalando</td><td>70,000</td><td>784</td><td>10</td><td>Recognition of clothing articles.</td></tr></table>

## C.2 Complete Numerical Results

Tables 3, 4 and 7 report the bound values, test errors, and training times, respectively, on the binary datasets. The corresponding results on the multiclass datasets are reported in Tables 5, 6 and 8. All results are averaged over 10 independent runs and reported with their standard deviations. Bound values and test errors are expressed as percentages, whereas training times are expressed in seconds.
<table><tr><td>Method</td><td>MUSHROOMS</td><td>TICTACTOE</td><td>SVMGUIDE</td><td>HABERMAN</td><td>PHISHING</td><td>CODRNA</td><td>ADULT</td><td>SPLICE</td><td>AUSTRALIAN</td></tr><tr><td>FO</td><td> $1 2 . 3 0 \pm 0 . 1 9$ </td><td> $7 6 . 4 7 \pm 1 . 4 5$ </td><td> $1 7 . 8 1 \pm 0 . 2 1$ </td><td> $7 5 . 7 3 \pm 1 . 7 3$ </td><td> $2 5 . 7 3 \pm 0 . 3 1$ </td><td> $4 9 . 4 5 \pm 0 . 4 7$ </td><td> $4 6 . 9 3 \pm 0 . 1 8$ </td><td> $5 4 . 6 7 \pm 0 . 9 3$ </td><td> $4 3 . 3 2 \pm 1 . 1 9$ </td></tr><tr><td>SO</td><td> $2 1 . 0 2 \pm 0 . 2 4$ </td><td> $1 0 0 . 2 0 \pm 1 . 0 2$ </td><td> $2 8 . 5 6 \pm 0 . 2 7$ </td><td> $1 1 0 . 5 8 \pm 1 . 4 2$ </td><td> $3 5 . 6 4 \pm 0 . 2 2$ </td><td> $6 0 . 5 5 \pm 0 . 2 1$ </td><td> $5 3 . 2 7 \pm 0 . 0 9$ </td><td> $6 3 . 6 5 \pm 0 . 4 8$ </td><td> $6 9 . 4 6 \pm 1 . 2 3 $ </td></tr><tr><td>Bin</td><td> $1 1 . 4 4 \pm 0 . 2 7$ </td><td> $7 9 . 5 0 \pm 0 . 8 8$ </td><td> $2 2 . 3 0 \pm 0 . 2 7$ </td><td> $8 0 . 3 6 \pm 0 . 7 3$ </td><td> $2 4 . 6 3 \pm 0 . 2 0$ </td><td> $3 6 . 2 0 \pm 0 . 1 3$ </td><td> $4 1 . 0 4 \pm 0 . 1 3$ </td><td> $4 6 . 4 0 \pm 0 . 4 4$ </td><td> $5 4 . 4 5 \pm 0 . 9 3$ </td></tr><tr><td>SMV-Exact</td><td> $\mathbf { 4 . 6 8 \pm 0 . 0 9 }$ </td><td> ${ \bf 4 2 . 7 4 \pm 0 . 6 3 }$ </td><td> ${ \bf 9 . 2 6 \pm 0 . 1 5 }$ </td><td> ${ \bf 4 2 . 1 7 \pm 0 . 8 4 }$ </td><td> ${ \bf 1 3 . 5 6 \pm 0 . 1 1 }$ </td><td> ${ \bf 1 5 . 9 0 \pm 2 . 9 9 }$ </td><td> ${ \bf 2 4 . 8 2 \pm 1 . 7 3 }$ </td><td> $3 2 . 0 7 \pm 2 2 . 6 5$ </td><td> ${ \bf 2 8 . 5 7 \pm 0 . 5 6 }$ </td></tr><tr><td>SMV-MC</td><td> $5 . 3 4 \pm 0 . 1 2$ </td><td> $4 3 . 3 6 \pm 0 . 7 0$ </td><td> $9 . 4 4 \pm 0 . 1 5$ </td><td> $4 4 . 2 6 \pm 1 . 5 2$ </td><td> $1 5 . 2 5 \pm 0 . 1 2$ </td><td> $1 7 . 1 5 \pm 2 . 5 9$ </td><td> $2 6 . 7 4 \pm 0 . 0 2$ </td><td> $3 5 . 8 2 \pm 2 1 . 4 0$ </td><td> $2 9 . 4 7 \pm 0 . 7 6$ </td></tr><tr><td>DIS-V</td><td> $6 . 6 0 \pm 1 . 1 5$ </td><td> $4 8 . 6 9 \pm 1 . 3 2$ </td><td> $1 2 . 4 0 \pm 3 . 8 7$ </td><td> $5 4 . 7 7 \pm 4 . 7 9$ </td><td> $1 6 . 4 7 \pm 0 . 7 6$ </td><td> $2 0 . 1 8 \pm 4 . 3 8$ </td><td> $2 7 . 3 2 \pm 0 . 2 8$ </td><td> $3 1 . 7 2 \pm 1 . 2 3$ </td><td> $3 5 . 9 1 \pm 2 . 0 5$ </td></tr><tr><td>DIS-R</td><td> $5 . 9 5 \pm 1 . 2 7$ </td><td> $4 5 . 5 8 \pm 2 . 6 3$ </td><td> $9 . 8 0 \pm 0 . 8 4$ </td><td> $5 1 . 4 5 \pm 7 . 6 8$ </td><td> $1 5 . 2 6 \pm 0 . 3 3$ </td><td> $1 8 . 1 6 \pm 3 . 5 5$ </td><td> $2 6 . 7 9 \pm 0 . 2 5$ </td><td> ${ \bf 2 9 . 2 5 \pm 1 . 3 2 }$ </td><td> $3 3 . 0 0 \pm 3 . 0 9$ </td></tr></table>

Table 3: Mean bound value and standard deviation over 10 independent runs on binary datasets.

<table><tr><td>Method</td><td>MUSHROOMS</td><td>TICTACTOE</td><td>SVMGUIDE</td><td>HABERMAN</td><td>PHISHING</td><td>CODRNA</td><td>ADULT</td><td>SPLICE</td><td>AUSTRALIAN</td></tr><tr><td>FO</td><td> $4 . 6 0 \pm 0 . 3 6$ </td><td> $2 9 . 2 2 \pm 2 . 7 7$ </td><td> $6 . 2 6 \pm 0 . 3 7$ </td><td> $2 5 . 4 8 \pm 3 . 2 1$ </td><td> $1 1 . 2 6 \pm 0 . 5 8$ </td><td> $2 3 . 5 2 \pm 0 . 3 5$ </td><td> $2 2 . 1 4 \pm 0 . 3 6$ </td><td> $2 2 . 2 8 \pm 1 . 8 0$ </td><td> $1 6 . 0 9 \pm 2 . 0 4$ </td></tr><tr><td>SO</td><td> $4 . 6 0 \pm 0 . 3 6$ </td><td> $3 0 . 0 0 \pm 3 . 8 4$ </td><td> $6 . 2 6 \pm 0 . 3 7$ </td><td> ${ \bf 2 4 . 3 5 \pm 3 . 3 4 }$ </td><td> $1 0 . 1 4 \pm 0 . 7 0$ </td><td> $2 2 . 9 3 \pm 1 . 4 3$ </td><td> $1 6 . 9 6 \pm 0 . 3 0$ </td><td> $1 2 . 7 9 \pm 1 . 3 3$ </td><td> $1 6 . 0 9 \pm 2 . 0 4$ </td></tr><tr><td>Bin</td><td> ${ \bf 1 . 0 8 \pm 0 . 2 2 }$ </td><td> $\mathbf { 2 7 . 7 1 \pm 3 . 6 0 }$ </td><td> $5 . 2 9 \pm 0 . 4 5$ </td><td> $2 6 . 4 5 \pm 1 . 2 9$ </td><td> ${ \bf 6 . 8 2 \pm 0 . 4 1 }$ </td><td> ${ \bf 1 2 . 3 8 \pm 0 . 2 5 }$ </td><td> ${ \bf 1 6 . 2 1 \pm 0 . 3 6 }$ </td><td> ${ \bf 7 . 1 0 \pm 1 . 0 5 }$ </td><td> ${ \bf 1 5 . 9 4 \pm 2 . 1 5 }$ </td></tr><tr><td>SMV-Exact</td><td> $1 . 3 4 \pm 0 . 2 1$ </td><td> $3 0 . 7 5 \pm 2 . 3 1$ </td><td> ${ \bf 5 . 0 8 \pm 0 . 3 4 }$ </td><td> $2 6 . 6 7 \pm 2 . 7 5$ </td><td> $8 . 1 7 \pm 0 . 3 5$ </td><td> $1 2 . 8 3 \pm 3 . 5 3$ </td><td> $2 2 . 2 5 \pm 2 . 9 7$ </td><td> $1 4 . 3 9 \pm 1 2 . 3 1$ </td><td> $1 6 . 9 9 \pm 1 . 9 1$ </td></tr><tr><td>SMV-MC</td><td> $1 . 2 7 \pm 0 . 2 3$ </td><td> $3 0 . 5 7 \pm 2 . 3 0$ </td><td> $5 . 1 1 \pm 0 . 4 1$ </td><td> $2 7 . 1 0 \pm 1 . 8 5$ </td><td> $7 . 9 8 \pm 0 . 4 0$ </td><td> $1 3 . 3 9 \pm 3 . 3 6$ </td><td> $2 4 . 0 7 \pm 0 . 0 0$ </td><td> $1 4 . 4 6 \pm 1 2 . 3 0$ </td><td> $1 6 . 5 6 \pm 2 . 0 1$ </td></tr><tr><td>DIS-V</td><td> $1 . 9 8 \pm 1 . 4 6 $ </td><td> $2 9 . 9 0 \pm 3 . 7 4$ </td><td> $6 . 8 0 \pm 2 . 9 1$ </td><td> $2 8 . 5 5 \pm 5 . 8 6$ </td><td> $8 . 2 2 \pm 0 . 6 7$ </td><td> $1 6 . 8 7 \pm 5 . 3 4$ </td><td> $2 4 . 0 7 \pm 0 . 0 0$ </td><td> $1 0 . 6 1 \pm 1 . 6 6$ </td><td> $1 6 . 2 3 \pm 2 . 1 3$ </td></tr><tr><td>DIS-R</td><td> $1 . 9 3 \pm 1 . 3 3$ </td><td> $2 9 . 9 0 \pm 2 . 8 8$ </td><td> $5 . 4 3 \pm 0 . 6 6$ </td><td> $2 6 . 7 7 \pm 1 . 0 7$ </td><td> $7 . 9 3 \pm 0 . 6 5$ </td><td> $1 4 . 6 7 \pm 4 . 2 4$ </td><td> $2 4 . 0 7 \pm 0 . 0 0$ </td><td> $1 0 . 0 9 \pm 1 . 4 8$ </td><td> $1 7 . 0 3 \pm 2 . 4 9$ </td></tr></table>

Table 4: Mean test error and standard deviation over 10 independent runs on binary datasets.

<table><tr><td>Method</td><td>PENDIGITS</td><td>SHUTTLE</td><td> $\mathrm { S E N S O R L E S S }$ </td><td>PROTEIN</td><td>MNIST</td><td>FASHION-MNIST</td></tr><tr><td>FO</td><td> $2 1 . 6 9 \pm 0 . 3 0$ </td><td> $0 . 3 6 \pm 0 . 0 2$ </td><td> $2 . 1 9 \pm 0 . 1 8$ </td><td> $1 1 2 . 6 4 \pm 0 . 4 7$ </td><td> $4 7 . 1 6 \pm 0 . 3 0$ </td><td> $5 3 . 2 0 \pm 0 . 2 6$ </td></tr><tr><td>SO</td><td> $1 7 . 4 1 \pm 0 . 2 0$ </td><td> $0 . 4 7 \pm 0 . 0 2$ </td><td> $1 . 9 5 \pm 0 . 0 4$ </td><td> $1 3 8 . 5 7 \pm 0 . 3 7$ </td><td> $4 4 . 8 1 \pm 0 . 1 8$ </td><td> $5 8 . 6 9 \pm 0 . 1 9$ </td></tr><tr><td>Bin</td><td> $9 . 0 1 \pm 0 . 2 8$ </td><td> $0 . 3 0 \pm 0 . 0 2$ </td><td> $1 . 0 5 \pm 0 . 0 6$ </td><td> $1 2 7 . 2 9 \pm 3 . 0 2$ </td><td> $3 0 . 7 8 \pm 0 . 2 2$ </td><td> $4 4 . 2 5 \pm 0 . 2 4$ </td></tr><tr><td>SMV-Exact</td><td> ${ \bf 4 . 8 0 \pm 0 . 1 3 }$ </td><td> $\mathbf { 0 . 1 6 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 5 5 \pm 0 . 0 3 }$ </td><td> $5 9 . 0 0 \pm 0 . 7 2$ </td><td> ${ \bf 1 5 . 9 4 \pm 0 . 1 1 }$ </td><td> ${ \bf 2 2 . 5 6 \pm 0 . 1 2 }$ </td></tr><tr><td>SMV-MC</td><td> $4 . 8 6 \pm 0 . 1 6$ </td><td> $\mathbf { 0 . 1 6 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 5 5 \pm 0 . 0 3 }$ </td><td> ${ \bf 5 8 . 7 7 \pm 0 . 4 1 }$ </td><td> $1 5 . 9 8 \pm 0 . 1 1$ </td><td> ${ \bf 2 2 . 5 6 \pm 0 . 1 2 }$ </td></tr><tr><td>DIS-V</td><td> $5 . 9 7 \pm 0 . 2 2$ </td><td> $0 . 2 7 \pm 0 . 0 2$ </td><td> $0 . 8 2 \pm 0 . 0 6$ </td><td> $6 0 . 2 7 \pm 0 . 4 2$ </td><td> $1 6 . 6 0 \pm 0 . 1 6$ </td><td> $2 3 . 2 6 \pm 0 . 1 8$ </td></tr><tr><td>DIS-R</td><td> $5 . 1 0 \pm 0 . 1 3$ </td><td> $0 . 1 8 \pm 0 . 0 2$ </td><td> $0 . 5 8 \pm 0 . 0 3$ </td><td> $5 9 . 1 7 \pm 0 . 5 0$ </td><td> $1 6 . 0 4 \pm 0 . 1 5$ </td><td> $2 2 . 6 1 \pm 0 . 1 6$ </td></tr></table>

Table 5: Mean bound value and standard deviation over 10 independent runs on multiclass datasets.

<table><tr><td>Method</td><td>PENDIGITS</td><td>SHUTTLE</td><td>SENSORLESS</td><td>PROTEIN</td><td>MNIST</td><td>FASHION-MNIST</td></tr><tr><td>FO</td><td> $8 . 2 6 \pm 1 . 0 7$ </td><td> $0 . 0 7 \pm 0 . 0 2$ </td><td> $0 . 6 8 \pm 0 . 1 1$ </td><td> ${ \bf 5 4 . 3 7 \pm 0 . 5 9 }$ </td><td> $2 2 . 4 1 \pm 0 . 3 3$ </td><td> $2 5 . 3 1 \pm 0 . 3 9$ </td></tr><tr><td>SO</td><td> ${ \bf 3 . 0 0 \pm 0 . 1 9 }$ </td><td> ${ \bf 0 . 0 4 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 2 0 \pm 0 . 0 6 }$ </td><td> $6 3 . 4 7 \pm 0 . 3 8$ </td><td> ${ \bf 1 4 . 3 4 \pm 0 . 1 7 }$ </td><td> ${ \bf 2 1 . 1 1 \pm 0 . 3 8 }$ </td></tr><tr><td>Bin</td><td> $3 . 1 8 \pm 0 . 1 9$ </td><td> $0 . 0 6 \pm 0 . 0 2$ </td><td> $0 . 2 1 \pm 0 . 0 6$ </td><td> $5 9 . 3 3 \pm 4 . 8 5$ </td><td> $1 4 . 4 8 \pm 0 . 1 9$ </td><td> $2 1 . 2 8 \pm 0 . 3 7$ </td></tr><tr><td>SMV-Exact</td><td> $3 . 4 4 \pm 0 . 2 2$ </td><td> $0 . 0 6 \pm 0 . 0 2$ </td><td> $0 . 2 3 \pm 0 . 0 5$ </td><td> $5 4 . 7 9 \pm 0 . 5 0$ </td><td> $1 4 . 8 7 \pm 0 . 1 8$ </td><td> $2 1 . 4 3 \pm 0 . 3 2$ </td></tr><tr><td>SMV-MC</td><td> $3 . 4 4 \pm 0 . 2 2$ </td><td> $0 . 0 6 \pm 0 . 0 2$ </td><td> $0 . 2 3 \pm 0 . 0 5$ </td><td> $5 4 . 9 1 \pm 0 . 4 6$ </td><td> $1 4 . 8 7 \pm 0 . 1 8$ </td><td> $2 1 . 4 3 \pm 0 . 3 2$ </td></tr><tr><td>DIS-V</td><td> $3 . 4 3 \pm 0 . 2 7$ </td><td> $0 . 0 6 \pm 0 . 0 2$ </td><td> $0 . 2 3 \pm 0 . 0 6$ </td><td> $5 5 . 0 0 \pm 0 . 4 6$ </td><td> $1 4 . 8 1 \pm 0 . 2 0$ </td><td> $2 1 . 3 8 \pm 0 . 2 9$ </td></tr><tr><td>DIS-R</td><td> $3 . 4 2 \pm 0 . 2 2$ </td><td> $0 . 0 6 \pm 0 . 0 2$ </td><td> $0 . 2 2 \pm 0 . 0 5$ </td><td> $5 5 . 4 3 \pm 0 . 6 9$ </td><td> $1 4 . 8 0 \pm 0 . 2 0$ </td><td> $2 1 . 3 5 \pm 0 . 3 0$ </td></tr></table>

Table 6: Mean test error and standard deviation over 10 independent runs on multiclass datasets.

<table><tr><td>Method</td><td>MUSHROOMS</td><td>TICTACTOE</td><td>SVMGUIDE</td><td>HABERMAN</td><td>PHISHING</td><td>CODRNA</td><td>ADULT</td><td>SPLICE</td><td>AUSTRALIAN</td></tr><tr><td>FO</td><td> $1 4 2 . 8 9 \pm 4 . 2 1$ </td><td> $5 0 . 9 5 \pm 6 . 2 3$ </td><td> $9 1 . 6 6 \pm 7 . 9 4$ </td><td> $4 7 . 7 4 \pm 2 . 6 9$ </td><td> $2 1 5 . 8 1 \pm 8 . 6 4$ </td><td> $2 0 4 6 . 8 6 \pm 4 7 . 4 6$ </td><td> $1 1 9 4 . 0 2 \pm 7 4 . 8 4$ </td><td> $5 8 . 2 2 \pm 8 . 3 5$ </td><td> $4 9 . 7 9 \pm 1 . 3 2$ </td></tr><tr><td>SO</td><td> $\mathbf { 1 1 5 . 5 1 \pm 2 . 0 9 }$ </td><td> $4 3 . 5 2 \pm 8 . 9 0$ </td><td> $8 7 . 7 0 \pm 8 . 8 3$ </td><td> $4 3 . 1 0 \pm 5 . 2 6$ </td><td> $2 0 1 . 5 3 \pm 4 . 4 1$ </td><td> $1 8 5 8 . 7 3 \pm 3 4 . 2 7$ </td><td> $9 6 7 . 9 4 \pm 1 7 . 2 9$ </td><td> ${ \bf 5 5 . 9 8 \pm 1 0 . 2 5 }$ </td><td> ${ \bf 4 1 . 6 3 \pm 8 . 7 8 }$ </td></tr><tr><td>Bin</td><td> $3 8 6 . 1 2 \pm 7 3 . 4 3$ </td><td> $9 4 . 3 7 \pm 1 6 . 0 0$ </td><td> $3 9 9 . 5 5 \pm 1 . 5 8$ </td><td> $6 3 . 1 1 \pm 6 . 1 9$ </td><td> $5 9 5 . 1 8 \pm 2 . 0 1$ </td><td> $3 3 6 5 . 2 7 \pm 2 0 . 9 5$ </td><td> $1 4 5 6 . 3 6 \pm 4 . 3 5$ </td><td> $1 0 0 . 0 1 \pm 7 . 9 1$ </td><td> $8 6 . 9 3 \pm 1 7 . 9 3$ </td></tr><tr><td>SMV-Exact</td><td> $3 3 5 . 3 6 \pm 7 1 . 9 6$ </td><td>101.23 ± 1.35</td><td> $3 5 7 . 8 5 \pm 5 4 . 9 6$ </td><td>62.91 ± 4.64</td><td> $5 6 0 . 9 7 \pm 2 . 7 1$ </td><td> $3 3 1 1 . 6 8 \pm 3 2 2 . 5 2$ </td><td>1795.34 ± 238.56</td><td> $1 2 2 . 1 4 \pm 2 4 . 5 7$ </td><td>62.81 ± 12.50</td></tr><tr><td>SMV-MC</td><td> $1 5 5 . 5 9 \pm 0 . 6 7$ </td><td> $5 9 . 4 0 \pm 1 . 1 8$ </td><td> $1 1 3 . 5 3 \pm 0 . 9 6$ </td><td> $5 1 . 3 2 \pm 0 . 3 8$ </td><td> $2 8 3 . 5 6 \pm 0 . 8 4$ </td><td> $8 5 7 . 4 0 \pm 3 4 9 . 0 1$ </td><td> $1 1 5 9 . 2 5 \pm 5 . 1 7$ </td><td> $8 0 . 5 1 \pm 1 0 . 8 9$ </td><td> $4 9 . 9 2 \pm 0 . 7 3$ </td></tr><tr><td>DIS-V</td><td> $1 2 9 . 0 5 \pm 1 . 0 3$ </td><td> $5 4 . 5 3 \pm 0 . 6 3$ </td><td> $1 0 0 . 6 0 \pm 1 . 0 0$ </td><td> $4 8 . 9 3 \pm 0 . 3 5$ </td><td> $\mathbf { 1 5 3 . 5 8 \pm 0 . 8 5 }$ </td><td> $7 8 2 . 1 3 \pm 2 9 6 . 3 9$ </td><td> $\mathbf { 4 3 7 . 9 9 \pm 2 . 5 4 }$ </td><td> $6 5 . 7 7 \pm 0 . 8 6$ </td><td> $4 8 . 8 8 \pm 0 . 8 9$ </td></tr><tr><td>DIS-R</td><td> $4 5 5 . 1 6 \pm 6 . 8 1$ </td><td> ${ \bf 2 3 . 7 3 \pm 0 . 2 4 }$ </td><td> ${ \bf 5 9 . 7 3 \pm 0 . 9 3 }$ </td><td> ${ \bf 1 7 . 5 9 \pm 0 . 2 1 }$ </td><td> $6 4 3 . 2 2 \pm 2 6 . 9 6$ </td><td> $\mathbf { 5 4 8 . 7 3 \pm 2 1 1 . 7 2 }$ </td><td> $6 0 2 . 6 7 \pm 9 . 8 5$ </td><td> $1 5 9 . 1 5 \pm 2 5 . 6 3$ </td><td> $5 0 . 7 1 \pm 0 . 7 0$ </td></tr></table>

Table 7: Mean training time (s) and standard deviation over 10 independent runs on binary datasets.

<table><tr><td>Method</td><td>PENDIGITS</td><td>SHUTTLE</td><td>SENSORLESS</td><td>PROTEIN</td><td>MNIST</td><td>FASHION-MNIST</td></tr><tr><td>FO</td><td> $2 2 6 . 8 0 \pm 2 9 . 2 7$ </td><td> $6 4 6 . 6 1 \pm 1 6 3 . 6 6$ </td><td> $3 6 2 . 6 1 \pm 6 6 . 3 4$ </td><td> $5 9 6 . 6 6 \pm 2 . 5 6$ </td><td> $2 2 5 0 . 3 5 \pm 6 4 . 3 3$ </td><td> $1 7 9 7 . 5 1 \pm 1 0 . 4 9$ </td></tr><tr><td>SO</td><td> $\mathbf { 2 0 3 . 5 1 \pm 4 1 . 4 9 }$ </td><td> ${ \bf 4 2 9 . 3 7 \pm 1 1 9 . 3 8 }$ </td><td> $\mathbf { 3 0 5 . 8 0 \pm 7 2 . 9 5 }$ </td><td> $6 2 0 . 8 6 \pm 5 . 4 2$ </td><td> $2 1 6 6 . 8 2 \pm 1 3 . 1 5$ </td><td> $1 7 1 5 . 1 4 \pm 5 . 8 2$ </td></tr><tr><td>Bin</td><td> $5 6 5 . 9 9 \pm 1 4 6 . 3 7$ </td><td> $1 7 5 1 . 1 5 \pm 5 6 3 . 6 6$ </td><td> $8 9 4 . 9 4 \pm 2 0 5 . 5 6$ </td><td> $1 4 8 6 . 8 1 \pm 3 2 3 . 8 1$ </td><td> $5 4 4 2 . 7 3 \pm 7 0 . 2 3$ </td><td> $4 8 5 5 . 8 0 \pm 3 9 . 7 5$ </td></tr><tr><td>SMV-Exact</td><td> $6 1 7 . 5 1 \pm 6 9 . 3 9$ </td><td> $1 8 8 8 . 1 7 \pm 3 4 2 . 5 1$ </td><td> $9 1 5 . 5 3 \pm 2 3 3 . 4 5$ </td><td> $1 5 4 2 . 9 8 \pm 1 5 5 . 4 6$ </td><td> $5 0 4 3 . 0 8 \pm 1 3 . 0 2$ </td><td> $4 5 9 0 . 2 6 \pm 2 1 . 4 6$ </td></tr><tr><td>SMV-MC</td><td> $2 5 9 . 1 4 \pm 3 . 0 9$ </td><td> $8 2 6 . 0 3 \pm 3 . 3 8$ </td><td> $4 3 8 . 1 4 \pm 2 . 1 9$ </td><td> $6 3 3 . 4 1 \pm 3 . 7 3$ </td><td> $2 2 6 3 . 8 6 \pm 2 4 . 8 1$ </td><td> $1 8 4 5 . 1 7 \pm 6 . 8 9$ </td></tr><tr><td>DIS-V</td><td> $2 4 6 . 1 3 \pm 4 . 0 9$ </td><td> $7 6 8 . 0 9 \pm 1 5 . 0 7$ </td><td> $4 1 3 . 0 5 \pm 2 . 7 2$ </td><td> $\mathbf { 5 8 8 . 3 4 \pm 2 . 5 6 }$ </td><td> $\mathbf { 2 1 5 9 . 5 7 \pm 2 0 . 5 2 }$ </td><td> $\mathbf { 1 6 5 7 . 7 6 \pm 9 . 3 2 }$ </td></tr><tr><td>DIS-R</td><td> $2 4 0 . 0 2 \pm 3 . 0 7$ </td><td> $7 6 3 . 8 8 \pm 8 . 9 7$ </td><td> $4 5 8 . 3 8 \pm 7 . 5 4$ </td><td> $6 9 2 . 6 2 \pm 8 . 6 5$ </td><td> $2 3 3 7 . 8 3 \pm 4 4 . 6 3$ </td><td> $1 7 8 5 . 8 5 \pm 1 0 . 0 4$ </td></tr></table>

Table 8: Mean training time (s) and standard deviation over 10 independent runs on multiclass datasets.