# Adversarial Training of Linear Models under Stealthy Attacks

Lovisa Eriksson, Dave Zachariah, and Andre M. H. Teixeira´

Abstract—Predictive models are widely used in many fields, but are vulnerable to false data injection attacks. To address this, detection schemes and adversarial training have been proposed, but such approaches lack guarantees against stealthy attacks. We therefore propose a detector-based switched model, in which optimal attack strategies are stealthy. For linear prediction models, we derive a convex formulation of the resulting adversarial risk. The model incorporates protected features and introduces a hyperparameter modelling attack probability, enabling an explicit performance trade-off between clean and attacked data regimes. Numerical simulations on real and synthetic data show improved performance on partially attacked data, even for misspecified attack probabilities.

Index Terms—Adversarial machine learning, False data injection attacks, Detection, Robust estimation

## I. INTRODUCTION

A basic idea for most modern machine learning algorithms is that a predictive model can be optimized on a set of past data such that it performs well on new data. However, in reality, the new data may not be drawn from the same distribution as the past data, due to natural variations [1], or false data injection attacks [2]. Applying a model to data subject to attacks presents a particular challenge since the new data is intentionally crafted to deteriorate the model performance [3].

False data injection attacks have been extensively studied, both in machine learning [3], [4], [5] and adjacent fields such as control theory [2]. To make the problem feasible, different assumptions are put on the attacker. Existing work either imposes a limit on the size of attack perturbations [3], [5] or imposes stealthiness [6], [7]. Classical robust statistics studies estimators under bounded contamination and worst-case deviations from nominal assumptions [1], but without analytically characterizing the resulting performance trade-off’s across regimes. Common defence mechanisms include adversarial training and detection schemes. Adversarial training [5], [8] improves robustness through the generation of static perturbed examples, thus limiting the defence against unforeseen and adapting attacks. Detection schemes against adversarial examples has been shown infective against stronger, tailored attacks [4].

To address these gaps simultaneously, we consider the use of a predictive model that switches mode depending on whether an attack detector triggers an alarm or not. Importantly, the attack can be stealthy and we consider the possibility that some features of the new data can be protected. Formal results are derived for linear models. Our main contributions are (i) a method that makes explicit trade-offs between accuracy on clean and attacked data, (ii) deriving adversarial stealthiness as a consequence of the proposed prediction model, and (iii) providing explicit convex formulations for adversarial training of linear models under stealthy attacks. Numerical experiments illustrating the performance use both synthetic and real data.

Notation: Throughout the paper we use ∥ · ∥ to denote the euclidean norm, $\mathbb { E } [ x ]$ to denote expectation of $x$ and $\mathbb { V } [ x , x ^ { \prime } ]$ is the covariance between x and $x ^ { \prime }$ , with $\mathbb { V } [ x ] \equiv \mathbb { V } [ x , x ]$ . The natural logarithm is denoted log and the transpose of a vector x is denoted $x ^ { \top }$ . We use $p ( x )$ to denote the probability density function of $x ,$ and $p ( x = k )$ to denote the probability of event $k .$

The code is available at GitHub:

LovisaLuna/code Adversarial-Training-of-Linear-Modelsunder-Stealthy-Attacks

## II. PROBLEM FORMULATION

Let $( x , y ) \in \mathcal { X } \times \mathcal { Y } \subseteq \mathbb { R } ^ { d } \times$ R denote a pair of features and outcomes of interest. Let $f _ { \theta } ( x )$ be a model parametrized by $\theta$ that predicts y from x with loss $\ell _ { \boldsymbol { \theta } } ( x , y )$ . Specifically, we study two cases: (i) regression $\mathcal { V } = \mathbb { R }$ using the standard linear model with squared error loss

$$
f _ { \boldsymbol { \theta } } ( x ) = { \boldsymbol { \theta } } ^ { \top } { \boldsymbol { x } } , \quad \ell _ { \boldsymbol { \theta } } ( x , y ) = | y - f _ { \boldsymbol { \theta } } ( x ) | ^ { 2 } , ~ \mathrm { a n d }\tag{1}
$$

(ii) binary classification $\mathcal { V } = \{ - 1 , 1 \}$ using logistic regression with logistic loss

$$
f _ { \theta } ( x ) = \frac { 1 - e ^ { - \theta ^ { \top } x } } { 1 + e ^ { - \theta ^ { \top } x } } , \quad \ell _ { \theta } ( x , y ) = \log \Big ( 1 + e ^ { - y \theta ^ { \top } x } \Big ) .\tag{2}
$$

During training, we observe n samples $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ drawn independently from $p ( x , y )$ . At test time, however, we observe the features $\widetilde { x }$ which could be manipulated by an attack, indicated by an unobserved variable $a \in \{ 0 , 1 \}$ that occurs with probability $\gamma ~ \in ~ [ 0 , 1 ]$ . We assume that features are partitioned into protected and unprotected components, which we express as:

$$
\begin{array} { r } { x = \left[ x _ { \mathrm { p } } ^ { \top } \quad x _ { \mathrm { u } } ^ { \top } \right] ^ { \top } \mathrm { a n d } \widetilde x = \left[ x _ { \mathrm { p } } ^ { \top } \quad \widetilde x _ { \mathrm { u } } ^ { \top } \right] ^ { \top } . } \end{array}\tag{3}
$$

Definition 1. An attack event $a \ = \ 1$ is adversarial if the observation follows

$$
\widetilde { x } = \left\{ \begin{array} { l l } { x , } & { \mathrm { i f ~ } a = 0 , } \\ { \arg \operatorname* { m a x } _ { z \in { \mathcal { Z } } ( x _ { \mathrm { p } } ) } \ell _ { \theta } ( z , y ) , } & { \mathrm { i f ~ } a = 1 , } \end{array} \right.\tag{4}
$$

where ${ \mathcal { Z } } ( x _ { \mathrm { p } } ) \subseteq \{ z \in { \mathcal { X } } : z _ { p } = x _ { p } \}$ . That is, the unprotected features in $\widetilde { x }$ are designed to maximize the loss of a given model $f _ { \theta }$

![](images/dab022d0ce6aa0aa9e5a0711c16621fa3303ccd6b4c6aecb1d9ce6be59e52a80.jpg)

Under this attack, we can express the process that generates the observation xe using an acyclic graph, illustrated to the left. The process induces a (marginal) distribution $p ( \widetilde { x } , y )$ at test time.

Definition 2. A (deterministic) anomaly detector is a function $\delta ( \widetilde { \boldsymbol { x } } ) : \mathbb { R } ^ { d }  \{ 0 , 1 \}$ and has a false alarm rate $p ( \delta ( \tilde { x } ) = 1 | a =$ 0). For notational brevity, the detector output will be written as $\delta = \delta ( \widetilde { x } )$ whenever it is unambiguous.

The detector splits the feature space $\mathcal { X }$ into two parts depending on whether $\delta = 0 \ \mathrm { o r } \ \delta = 1$ . Therefore, we propose a switched model approach that uses a nominal model $f _ { \theta _ { 0 } }$ and a recovery model $f _ { \theta _ { 1 } }$ , depending on δ. We can express the above approach using the switched model

$$
f _ { \theta _ { 0 } , \theta _ { 1 } } ( \widetilde { x } , \delta ) = \left\{ \begin{array} { l l } { f _ { \theta _ { 0 } } ( \widetilde { x } ) , } & { \mathrm { ~ i f ~ } \delta = 0 } \\ { f _ { \theta _ { 1 } } ( \widetilde { x } ) , } & { \mathrm { ~ i f ~ } \delta = 1 . } \end{array} \right.\tag{5}
$$

The loss of the switched model can then be expressed as

$$
\ell _ { \theta _ { 0 } , \theta _ { 1 } } ( \widetilde x , y ) : = \delta ( \widetilde x ) \ell _ { \theta _ { 1 } } ( \widetilde x , y ) + ( 1 - \delta ( \widetilde x ) ) \ell _ { \theta _ { 0 } } ( \widetilde x , y ) ,\tag{6}
$$

such that its expected loss, or $r i s k ,$ is

$$
R ( \theta _ { 0 } , \theta _ { 1 } ) = \mathbb { E } [ \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( \widetilde { x } , y ) ] .\tag{7}
$$

It should be noted that minimizing the risk of the switched model can be no worse than using only one of the models $f _ { \theta _ { 0 } }$ and $f _ { \theta _ { 1 } }$ , i.e.,

$$
\operatorname* { m i n } _ { \theta _ { 0 } , \theta _ { 1 } } R ( \theta _ { 0 } , \theta _ { 1 } ) \leq \operatorname* { m i n } _ { \theta _ { 0 } = \theta _ { 1 } } R ( \theta _ { 0 } , \theta _ { 1 } ) = \operatorname* { m i n } _ { \theta } \mathbb { E } [ \ell _ { \theta } ( \widetilde { x } , y ) ] .\tag{8}
$$

However, this does not guarantee that attacked data is handled by the recovery model. If an attacker has complete knowledge of the detector function $\delta ( \cdot )$ , an attack event can be designed to evade detection, i.e. to be stealthy. Assuming full model knowledge of the adversary is a standard assumption in security to make sure the security of proposed methods rely only on theoretical principles, rather than lack of equipment or knowledge of the adversary [9].

Definition 3. An adversarial attack event $a = 1$ is stealthy w.r.t. detector δ if the latter misses the attack with probability $p ( \delta ( \widetilde { x } ) = 0 | a = 1 ) = 1$ . This occurs when the set of feasible attacks is further restricted to exploit the detector, by setting

$$
{ \mathcal { Z } } ( x _ { \mathfrak { p } } ) \subseteq \{ z \in { \mathcal { X } } : z _ { p } = x _ { p } \} \cap \{ z \in { \mathcal { X } } : \delta ( z ) = 0 \} .\tag{9}
$$

The goal of the paper is to learn a risk-minimizing switched model $f _ { \theta _ { 0 } , \theta _ { 1 } }$ , using clean training data, that is robust against adversarial and stealthy attacks at test time, given a detector with false alarm rate no greater than α¯.

## III. THEORETICAL RESULTS

We now show that under certain conditions on the recovery model $f _ { \theta _ { 1 } }$ , the attack (4) will be essentially equivalent to an stealthy adversarial attack. We formalise the required conditions as $f _ { \theta _ { 1 } }$ being secure.

Definition 4. The recovery model $f _ { \boldsymbol { \theta } _ { 1 } } ( \boldsymbol { x } )$ is said to be secure w.r.t δ and $f _ { \theta _ { 0 } }$ if there exists an imputation function $g ( x _ { \mathfrak { p } } ) = { \widehat x }$ such that, for all $\tilde { x } = \left[ x _ { p } ^ { \top } \quad \tilde { x } _ { u } ^ { \top } \right] ^ { \top }$

$$
f _ { \theta _ { 1 } } ( \widetilde { x } ) = f _ { \theta _ { 0 } } ( g ( x _ { p } ) ) = f _ { \theta _ { 0 } } ( \widehat { x } ) \quad \mathrm { a n d } \quad \delta ( \widehat { x } ) = 0 .\tag{10}
$$

Thus the secure $f _ { \theta _ { 1 } }$ imputes $\widetilde { x } _ { \mathrm { u } }$ and reverts to the prediction of $f _ { \theta _ { 0 } }$ , rather than taking the unprotected features at face value. With a secure model, an attack event can never cause more harm when detected.

Proposition 1. Let $f _ { \theta _ { 0 } , \theta _ { 1 } } ( \widetilde { x } , \delta )$ be a model (5) such that $f _ { \theta _ { 1 } }$ is secure w.r.t δ and $f _ { \theta _ { 0 } }$ . Then, any adversarial attack event against this model yields the same risk as a stealthy adversarial attack. That is, letting ${ \mathcal { Z } } ( x _ { p } ) = \{ z \in { \mathcal { X } } : z _ { p } = x _ { p } \}$

$$
\operatorname* { m a x } _ { z \in \mathcal { Z } ( x _ { \mathsf { p } } ) } \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( z , y ) \equiv \operatorname* { m a x } _ { z \in \mathcal { Z } ( x _ { \mathsf { p } } ) , \delta ( z ) = 0 } \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( z , y )\tag{11}
$$

Proof. Note that using the constraint $\delta ( z ) = 0$

$$
\operatorname* { m a x } _ { z \in \mathcal { Z } ( x _ { \mathsf { p } } ) } \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( z , y ) \geq \operatorname* { m a x } _ { z \in \mathcal { Z } ( x _ { \mathsf { p } } ) , \delta ( z ) = 0 } \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( z , y ) .\tag{12}
$$

It is then sufficient to prove that

$$
\operatorname* { m a x } _ { z \in \mathcal { Z } ( x _ { \mathsf { p } } ) } \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( z , y ) \leq \operatorname* { m a x } _ { z \in \mathcal { Z } ( x _ { \mathsf { p } } ) , \delta ( z ) = 0 } \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( z , y ) .\tag{13}
$$

Let $x ^ { * } = \arg \operatorname* { m a x } _ { z \in { \mathcal { Z } } ( x _ { \mathrm { p } } ) } \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( z , y )$ . When $\delta ( x ^ { * } ) = 0 .$ , (13) holds trivially. Now assume $\delta ( x ^ { * } ) = 1$ . Then, by security of $f _ { \theta _ { 1 } }$ as per Definition 4, there exists a mapping $g ( x _ { p } ) = \widehat { x }$ such that $\delta ( \widehat { x } ) = 0$ and

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { z \in \mathcal { Z } ( x _ { \mathrm { p } } ) } \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( z , y ) = \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( x ^ { * } , y ) } & { } \\ { = \ell _ { \theta _ { 1 } } ( x ^ { * } , y ) = \ell _ { \theta _ { 0 } } ( \widehat { x } , y ) } & { } \\ { = \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( \widehat { x } , y ) \le \displaystyle \operatorname* { m a x } _ { z \in \mathcal { Z } ( x _ { \mathrm { p } } ) , \delta ( z ) = 0 } \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( z , y ) . } & { } \end{array}\tag{14}
$$

Since we will use a secure model, we can from here on assume that all adversarial attacks are stealthy.

For the rest of this section, as well as for the numerical experiments, we consider the imputation function to be

$$
g ( x _ { p } ) = [ \frac { x _ { p } } { \widehat { x } _ { u } } ] = [ \begin{array} { c } { I } \\ { \mathbb { V } [ x _ { u } , x _ { p } ] \mathbb { V } [ x _ { p } ] ^ { - 1 } ] x _ { p } } \end{array}\tag{15}
$$

where $\widehat { x } _ { u }$ is the optimal linear predictor [10], and use the energy detector

$$
\delta ( \widetilde { x } ) = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { ~ i f ~ } \| \Sigma ^ { - \frac { 1 } { 2 } } \left( \widetilde { x } _ { \mathrm { u } } - \widehat { x } _ { \mathrm { u } } \right) \| _ { 2 } ^ { 2 } \leq \tau _ { \bar { \alpha } } } \\ { 1 , } & { \mathrm { ~ o t h e r w i s e } } \end{array} \right. ,\tag{16}
$$

where $\widehat { x } _ { \mathrm { u } }$ from (15), α¯ a desired upper bound on the false alarm rate, and

$$
\Sigma = \mathbb { V } [ x _ { u } ] - \mathbb { V } [ x _ { u } , x _ { p } ] \mathbb { V } [ x _ { p } ] ^ { - 1 } \mathbb { V } [ x _ { u } , x _ { p } ] ^ { \top } .\tag{17}
$$

is the error covariance of $\widehat { x } _ { u }$ in (15).

The threshold $\tau _ { \bar { \alpha } }$ can be chosen to guarantee a set false alarm rate in different ways: (i) For a general distribution, $\begin{array} { r } { \tau _ { \bar { \alpha } } = \frac { d _ { u } } { \bar { \alpha } } } \end{array}$ guarantees a false alarm bounded by α¯, through the Chebychev inequality [11]; (ii) for a Gaussian distribution, the detector corresponds to a chi-test and the exact false alarm rate of a given τ can thus be determined.

Using this detector together with linear or logistic regression in (5), we show that there is a choice of $\theta _ { 1 }$ that guarantees $f _ { \theta _ { 1 } }$ to be secure w.r.t. δ and $f _ { \theta _ { 0 } }$

Lemma 1. Consider the switched model $f _ { \theta _ { 0 } , \theta _ { 1 } } ~ ( { 5 } )$ , with anomaly detector δ (16). Then, letting

$$
\boldsymbol { \theta } _ { 1 } ^ { \top } \equiv \boldsymbol { \theta } _ { 0 } ^ { \top } \left[ \begin{array} { c c } { I } & { 0 } \\ { \mathbb { V } \left[ \boldsymbol { x } _ { u } , \boldsymbol { x } _ { p } \right] \mathbb { V } \left[ \boldsymbol { x } _ { p } \right] ^ { - 1 } } & { 0 } \end{array} \right]\tag{18}
$$

ensures that $f _ { \theta _ { 1 } }$ is secure w.r.t. δ and $f _ { \theta _ { 0 } }$

Proof. Using $g ( x _ { p } ) = \left[ x _ { p } ^ { \top } \widehat { x } _ { \mathrm { { u } } } \right] ^ { \top }$ in (15) with a linear regression model, we have

$$
f _ { \theta _ { 1 } } ( \widetilde { \boldsymbol { x } } ) = \theta _ { 0 } ^ { \top } \left[ \begin{array} { c c } { I } & { 0 } \\ { \mathbb { V } \left[ \boldsymbol { x _ { u } } , \boldsymbol { x _ { p } } \right] \mathbb { V } \left[ \boldsymbol { x _ { p } } \right] ^ { - 1 } } & { 0 } \end{array} \right] \left[ \boldsymbol { x _ { p } } \right]\tag{19}
$$

$$
= \theta _ { 0 } ^ { \top } g ( x _ { p } ) = f _ { \theta _ { 0 } } ( g ( x _ { p } ) ) ,\tag{20}
$$

and $\delta ( \widehat { x } ) = 0$ , thus satisfying the conditions in Definition 4. A corresponding argument can be made for a logistic model.

Now, we show that the risk (7) under attack is convex, guaranteeing that the global optima can be found by numerical optimization. For notational convenience, let $M = \left\lceil 0 \quad I \right\rceil ^ { \top } \Sigma ^ { \frac { 1 } { 2 } }$

Proposition 2. Consider the linear regression model (1) with anomaly detector (16). The risk (7) under stealthy adversarial attacks, with attack probability $\gamma _ { - }$ , is

$$
\begin{array} { r l } & { R ( \theta _ { 0 } , \theta _ { 1 } ) = \gamma \mathbb { E } \left[ \big ( \vert y - { \theta } _ { 0 } ^ { \top } g ( x _ { p } ) \vert + \sqrt { \tau _ { \bar { \alpha } } } \left. { \theta } _ { 0 } ^ { \top } { M } \right. \big ) ^ { 2 } \right] } \\ & { + ( 1 - \gamma ) \mathbb { E } \big [ \delta \Vert y - { \theta } _ { 1 } ^ { \top } x \Vert ^ { 2 } + ( 1 - \delta ) \Vert y - { \theta } _ { 0 } ^ { \top } x \Vert ^ { 2 } \big ] . } \end{array}\tag{21}
$$

The risk is a convex function of the model parameters $\theta _ { 0 }$ and $\theta _ { 1 }$ .

Proof. The expectation (7) can be divided into two cases depending on the attack event a, so that

$$
\begin{array} { r l } & { R ( \theta _ { 0 } , \theta _ { 1 } ) } \\ & { \ = \gamma \mathbb { E } [ \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( \widetilde { x } , y ) | a = 1 ] + ( 1 - \gamma ) \mathbb { E } [ \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( \widetilde { x } , y ) | a = 0 ] } \\ & { \ = \gamma \mathbb { E } [ \underset { z \in \mathcal { Z } ( \widetilde { x } _ { \mathsf { p } } ) , \delta ( z ) = 0 } { \operatorname* { m a x } } \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( z , y ) ] + ( 1 - \gamma ) \mathbb { E } [ \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( x , y ) ] . } \end{array}
$$

By (6), the second term becomes

$$
( 1 - \gamma ) \mathbb { E } [ \delta \| y - \theta _ { 1 } ^ { \top } x \| ^ { 2 } + ( 1 - \delta ) \| y - \theta _ { 0 } ^ { \top } x \| ^ { 2 } ] .\tag{22}
$$

Now, we can focus on the first term. Since the adversary is stealthy,

$$
\operatorname* { m a x } _ { z \in \mathcal { Z } ( \widetilde { x } _ { \mathrm { p } } ) , \delta ( z ) = 0 } \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( z , y ) = \operatorname* { m a x } _ { z \in \mathcal { Z } ( \widetilde { x } _ { \mathrm { p } } ) , \delta ( z ) = 0 } \ell _ { \theta _ { 0 } } ( z , y ) ,\tag{23}
$$

and plugging in the squared error loss this becomes

$$
\operatorname* { m a x } _ { z \in \mathcal { Z } ( \widetilde { x } _ { \mathfrak { p } } ) , \delta ( z ) = 0 } \Vert y - \theta _ { 0 } ^ { \top } z \Vert _ { 2 } ^ { 2 } ]\tag{24}
$$

$$
= \operatorname* { m a x } _ { \delta ( z ) = 0 } \lVert y - \theta _ { 0 , \mathrm { p } } ^ { \top } x _ { \mathrm { p } } - \theta _ { 0 , \mathrm { u } } ^ { \top } z _ { u } \rVert _ { 2 } ^ { 2 } ,\tag{25}
$$

where $\begin{array} { r l } { \theta _ { 0 } = \left[ \theta _ { 0 , \mathrm { p } } \quad \theta _ { 0 , \mathrm { u } } \right] ^ { \top } } \end{array}$ , and we can rewrite

$$
\theta _ { 0 , { \bf u } } ^ { \top } z _ { u } = \theta _ { 0 , { \bf u } } ^ { \top } ( \Sigma ^ { \frac { 1 } { 2 } } \Sigma ^ { - \frac { 1 } { 2 } } ( z _ { \bf u } - \widehat { x } _ { \bf u } ) + \widehat { x } _ { \bf u } ) .\tag{26}
$$

Now, letting $\xi = \Sigma ^ { - \frac { 1 } { 2 } } ( \widetilde { x } _ { \mathrm { u } } - \widehat { x } _ { \mathrm { u } } )$ , we get

$$
\operatorname* { m a x } _ { \| \boldsymbol { \xi } \| _ { 2 } \le \sqrt { \tau _ { \alpha } } } \| y - \theta _ { 0 , \mathrm { p } } ^ { \top } x _ { \mathrm { p } } - \theta _ { 0 , \mathrm { u } } ^ { \top } \Sigma ^ { \frac { 1 } { 2 } } \boldsymbol { \xi } - \theta _ { 0 , \mathrm { u } } ^ { \top } \widehat x _ { \mathrm { u } } \| _ { 2 } ^ { 2 }\tag{27}
$$

By collecting the constant terms in $\begin{array} { r } { c = y - \theta _ { 0 , \mathrm { p } } ^ { \top } x _ { \mathrm { p } } - \theta _ { 0 , \mathrm { u } } ^ { \top } \widehat x _ { \mathrm { u } } . } \end{array}$ the expression can be rewritten as

$$
c ^ { 2 } + \operatorname* { m a x } _ { \| \xi \| _ { 2 } \leq \sqrt { \tau _ { \bar { \alpha } } } } ( \theta _ { 0 , \mathrm { u } } ^ { \top } \Sigma ^ { \frac { 1 } { 2 } } \xi ) ^ { 2 } - 2 c \theta _ { 0 , \mathrm { u } } ^ { \top } \Sigma ^ { \frac { 1 } { 2 } } \xi .\tag{28}
$$

Define $r = \theta _ { 0 , \mathrm { u } } ^ { \top } \Sigma ^ { \frac { 1 } { 2 } } \xi$ . We can choose ξ in the direction of the main singular vector of $\theta _ { \mathrm { 0 , u } } ^ { \top } \Sigma ^ { \frac { 1 } { 2 } }$ , such that

$$
\begin{array} { r } { | r | = \| \theta _ { 0 , \mathrm { u } } ^ { \top } \Sigma ^ { \frac { 1 } { 2 } } \xi \| = \| \theta _ { 0 , \mathrm { u } } ^ { \top } \Sigma ^ { \frac { 1 } { 2 } } \| \| \xi \| = \sqrt { \tau _ { \bar { \alpha } } } \| \theta _ { 0 , \mathrm { u } } ^ { \top } \Sigma ^ { \frac { 1 } { 2 } } \| , } \end{array}\tag{29}
$$

and $\mathrm { s i g n } ( r ) = \mathrm { s i g n } ( - c )$ , which gives us the maximum as

$$
c ^ { 2 } + \tau _ { \bar { \alpha } } \| \theta _ { 0 , { \bf u } } ^ { \top } \Sigma ^ { \frac { 1 } { 2 } } \| ^ { 2 } - 2 c \sqrt { \tau _ { \bar { \alpha } } } \| \theta _ { 0 , { \bf u } } ^ { \top } \Sigma ^ { \frac { 1 } { 2 } } \| .\tag{30}
$$

The expression can be compactly written as

$$
\left( | y - \theta _ { 0 } ^ { \top } \widehat { x } | + \sqrt { \tau _ { \bar { \alpha } } } \| \theta _ { 0 } ^ { \top } M \| \right) ^ { 2 } .\tag{31}
$$

This formulation is convex in $\theta _ { 0 }$ since it is a monotonically increasing function of a weighted sum of convex functions. Finally, the full risk (21) thus is another weighted sum of convex functions in $\theta _ { 0 }$ and $\theta _ { 1 }$ □

Similarly, the risk under attack for logistic regression can be shown to be convex.

Proposition 3. Consider the logistic regression model (2) with anomaly detector (16). The risk (7) under stealthy adversarial attacks with attack probability γ, is

$$
\begin{array} { r l } & { R ( \theta _ { 0 } , \theta _ { 1 } ) = \gamma \mathbb { E } \left[ \log \left( 1 + e ^ { - y \theta _ { 0 } ^ { \top } g ( x _ { p } ) + \sqrt { \tau _ { \bar { \alpha } } } \| \theta _ { 0 } ^ { \top } M \| } \right) \right] } \\ & { ~ + ( 1 - \gamma ) \mathbb { E } [ \delta \log \left( 1 + e ^ { - y \theta _ { 1 } ^ { \top } x } \right) } \\ & { ~ + ( 1 - \delta ) \log \left( 1 + e ^ { - y \theta _ { 0 } ^ { \top } x } \right) ] . } \end{array}\tag{32}
$$

This risk is convex in the model parameters θ.

Proof. The proof uses the same main ideas as the one for Proposition 2 First, the risk is divided into cases depending on whether an attack is present, and the second term follows directly from plugging in the logistic loss. Now, consider the first term, where an attack is present. Note that by monotonicity of the logarithm and exponential, maximizing the logistic loss is equivalent to maximizing

$$
- y \theta _ { 0 } ^ { \top } z = - y \theta _ { 0 , p } ^ { \top } x _ { \mathrm { p } } - y \theta _ { 0 , \mathrm { u } } ^ { \top } z _ { u } .\tag{33}
$$

Letting $\xi = \Sigma ^ { - \frac { 1 } { 2 } } ( z _ { u } - \widehat { x } _ { u } )$ , we get

$$
\operatorname* { m a x } _ { \xi \leq \sqrt { \tau _ { \widehat { \alpha } } } } - y \theta _ { 0 } \widehat { x } + \theta _ { 0 } ^ { \top } M \xi .\tag{34}
$$

By choosing ξ in the proper direction, we obtain the maximum loss

$$
\log \Big ( 1 + e ^ { - y \theta _ { 0 } ^ { \top } \widehat { x } + \sqrt { \tau _ { \bar { \alpha } } } \| \theta _ { 0 } ^ { \top } M \| } \Big ) ,\tag{35}
$$

and the desired formulation is obtained. The convexity follows straight-forward from the convexity of the logistic loss and norms. □

It can be noted that as $\gamma  0$ in (21) and (32), the risk minimizing $\begin{array} { r l } { \theta _ { 0 } ^ { \top } = \big \lceil \theta _ { p } ^ { \top } } & { { } \theta _ { u } ^ { \top } \big \rceil } \end{array}$ sets $\theta _ { u } = 0$ , as this minimizes $\lVert \theta _ { 0 } ^ { \top } M \rVert$ and the rank of $\theta _ { 0 } ^ { \top } \bar { g } ( x _ { p } )$ allows $\theta _ { u }$ to be treated as a free variable.

## IV. METHOD

The adverserial training is achieved by minimizing the empirical expectation $\begin{array} { r l r } { R _ { n } ( \theta _ { 0 } , \theta _ { 1 } ) } & { { } = } & { \mathbb { E } _ { n } [ \ell _ { \theta _ { 0 } , \theta _ { 1 } } ( \widetilde { x } , y ) ] \quad = } \end{array}$ $\begin{array} { r } { \frac { 1 } { n } \dot { \sum _ { i = 1 } ^ { n } } \ell _ { \theta _ { 0 } , \theta _ { 1 } } \dot { ( x ^ { ( i ) } , y ^ { ( i ) } ) } } \end{array}$ in (21) or (32), with constraint (18), using the clean training data. We use stochastic gradient descent with momentum and decreasing step size. The method requires specifying an upper bound on false alarm $\bar { \alpha } ,$ and on the attack probability, denoted $\gamma _ { \mathrm { t r a i n } }$

## V. NUMERICAL RESULTS

For comparison, we use two different baselines. First, a fully secure model denoted $f _ { \theta _ { s } , \theta _ { s } }$ , with the restriction $\theta _ { s } = $ $\begin{array} { r l } { \left[ \theta _ { p } ^ { \dagger } \right. } & { { } \left. 0 _ { u } \right] ^ { \top } } \end{array}$ , so that the unprotected features are discarded and the model is secure w.r.t. any detector $\delta$ and itself by Definition 4. Second, a standard switched model denoted $f _ { \boldsymbol { \theta } _ { n } , \boldsymbol { \theta } _ { s } }$ with $\begin{array} { r } { \theta _ { n } = \arg \operatorname* { m i n } _ { \theta _ { 0 } } \mathbb { E } [ \ell _ { \theta _ { 0 } } ( x , y ) ] } \end{array}$ , i.e. trained to minimize the risk on clean data, with constraint (18), which we denote $f _ { \theta _ { n } , \theta _ { s } }$

All models are evaluated with respect to their (empirical) risk under data that is under stealthy adversarial attacks with probability $\gamma _ { \mathrm { t e s t } }$ . The data is split 50-50 into a train and a test set, and we evaluate the method on both synthetic and real world data.

## A. Synthetic data

We evaluate our methodology for linear regression where $d = 4$ and the label is given by $\textstyle y = \sum _ { i = 1 } ^ { 4 } x _ { i } .$ . The distribution $p ( x )$ is a zero mean Gaussian with $\mathbb { V } [ x _ { i } ] = 1$ for all i and $\mathbb { V } [ x _ { 1 } , x _ { 3 } ] = \mathbb { V } [ x _ { 2 } , x _ { 4 } ] = 0 . 8$ (remaining covariances are zero). We use $n = 2 0 0 0$ training samples.

Fig. 1 illustrate the results. First, stealthy attacks are clearly detrimental to the standard model since no attacks are detected! Second, when $\gamma _ { \mathrm { t r a i n } } = \gamma _ { \mathrm { t e s t } } = 0 .$ , the proposed switched model has the same performance but converges to that of the fully secure model as $\gamma _ { \mathrm { t e s t } }$ increases. Between these points it shows an improved performance. Third, even as the assumed upper bound is incorrect, $\gamma _ { \mathrm { t r a i n } } < \gamma _ { \mathrm { t r a i n } }$ , there is still a range in which the risk is lower than the secure model and is consistently lower than the standard model.

Simulations for other parameters of the Gaussian, and for logistic regression, show the same overall behaviour with variation only in convergence rate, and plots for these are therefore excluded from this paper.

## B. Real world data

We further evaluate our training scheme on a real word Credit Card Fraud Detection data set [12] with $n \ : = \ : 5 6 9 k$ samples. The feature ‘amount’ is treated as protected and 21 features are treated as unprotected (7 features removed due to high correlation with other features). The results are shown in Fig. 2, where we see that the method also works for complex real word data with a more significant number of features.

![](images/15e41b5451343cd0b8542bbcbdec0db5f39d4d723c6405e0ab6a71d57bf806cf.jpg)  
(a)

![](images/36fc6bbbc2748526ea768ddccd69de08b251c0c0d998c6bb582e7a9485a40bb2.jpg)  
(b)

Fig. 1: (a) Risk versus attack probability $\gamma _ { \mathrm { t e s t } } .$ including misspecified $\gamma _ { \mathrm { t r a i n } } = 0 . 0 5$ . (b) Risk versus $\gamma _ { \mathrm { t r a i n } } .$ where red dots mark when the model is correctly specified and the dotted line is the correctly specified model from (a). Both (a) and (b) also show the standard switched model $f _ { \theta _ { n } , \theta _ { s } }$ , and the fully secure model $f _ { \theta _ { s } , \theta _ { s } }$ . Both cases set $\bar { \alpha } = 0 . 0 1$ yielding an actual false alarm rate of 1.3%.  
![](images/905cf33d7b3a9c584df01a12c9808c1e67466ad00bb93fc1313092e87929b5c5.jpg)  
(a)

![](images/c09e7c12c5d460be52d630faef508e0ada195e10be787cd37be5ddddb8767bb6.jpg)  
(b)  
Fig. 2: (a) Risk versus attack probability $\gamma _ { \mathrm { t e s t } } .$ . (b) Same as (a) but also includes the standard switched model. In this case we set the bound α¯ yielding an actual false alarm rate of 0.01%.

## VI. CONCLUSION

We have shown that using an attack detector and aligning the recovery model with the nominal model, adversarial attacks are forced to be stealthy. A convex risk for this setup is derived for linear and logistic regression, and numerical results support the optimality of the resulting model. Future work include extension to more complex models and detectors, extension to neural networks, and further investigating the relation between our method, adversarial training, and regularisation.

## REFERENCES

[1] P. J. Huber, “Robust Estimation of a Location Parameter,” The Annals of Mathematical Statistics, vol. 35, no. 1, pp. 73 – 101, 1964.

[2] G. Liang, J. Zhao, F. Luo, S. R. Weller, and Z. Y. Dong, “A review of false data injection attacks against modern power systems,” IEEE Transactions on Smart Grid, vol. 8, no. 4, pp. 1630–1638, 2017.

[3] A. H. Ribeiro and T. B. Schon, “Overparameterized linear regression¨ under adversarial attacks,” Trans. Sig. Proc., vol. 71, p. 601–614, jan 2023.

[4] N. Carlini and D. Wagner, “Adversarial examples are not easily detected: Bypassing ten detection methods,” in Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security, ser. AISec ’17. New York, NY, USA: Association for Computing Machinery, 2017, p. 3–14.

[5] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” in Conference Track Proceedings of the 3rd International Conference on Learning Representations, ICLR 2015, 2015.

[6] D. I. Urbina, J. A. Giraldo, A. A. Cardenas, N. O. Tippenhauer, J. Valente, M. Faisal, J. Ruths, R. Candell, and H. Sandberg, “Limiting the impact of stealthy attacks on industrial control systems.” New York, NY, USA: Association for Computing Machinery, 2016.

[7] Y. Mao, H. Jafarnejadsani, P. Zhao, E. Akyol, and N. Hovakimyan, “Novel stealthy attack and defense strategies for networked control systems,” IEEE Transactions on Automatic Control, vol. 65, no. 9, pp. 3847–3862, 2020.

[8] W. Zhao, S. Alwidian, and Q. H. Mahmoud, “Adversarial training methods for deep learning: A systematic review,” Algorithms, vol. 15, no. 8, 2022.

[9] C. E. Shannon, “Communication theory of secrecy systems,” The Bell System Technical Journal, vol. 28, no. 4, pp. 656–715, 1949.

[10] T. Kailath, A. H. Sayed, and B. Hassibi, Linear Estimation. New Jersey, Prentice Hall, 2000.

[11] X. Liu, D. Zachariah, and P. Stoica, “Robust prediction when features are missing,” IEEE Signal Processing Letters, vol. 27, pp. 720–724, 2020.

[12] N. Elgiriyewithana, “Credit card fraud detection dataset 2023,” 2023. [Online]. Available: https://www.kaggle.com/dsv/6492730