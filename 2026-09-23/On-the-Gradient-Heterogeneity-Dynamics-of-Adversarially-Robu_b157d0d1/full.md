# On the Gradient Heterogeneity Dynamics of Adversarially Robust Federated Regression

Leonardo F. Toso<sup>∗1</sup>, James Anderson<sup>1</sup>, Nirupam Gupta<sup>2</sup>, and Rafael Pinot<sup>3</sup>

<sup>1</sup>Columbia University, USA

<sup>2</sup>University of Copenhagen, Denmark

<sup>3</sup>Sorbonne Universit´e and Universit´e Paris Cit´e, CNRS, LPSM, France

## Abstract

Federated learning (FL) is intrinsically heterogeneous: honest clients may have diferent data-generating models. On top of that, adversarial clients can make heterogeneity even more pronounced by sharing arbitrary updates. Existing analyses typically control the interaction between statistical heterogeneity and adversarial behavior through gradientdissimilarity conditions. However, the underlying bound is imposed a priori and may yield conservative guarantees even for least-squares regression. We instead derive the gradient heterogeneity from the statistical model of linear and nonlinear regression with fresh data samples at every round. Our bounds separate heterogeneity among the honest clients’ groundtruth model parameters, finite-sample label noise, and initialization. We then demonstrate that, for any (f, κ)-robust aggregator with coeficient κ = O(f/n), where f is the number of adversarial clients and n the total number of clients (with $f / n < 1 / 2 )$ , convergence holds after an explicit sample burn-in.

## 1 Introduction

Federated learning (FL) allows for a collection of clients to train a shared model while keeping their data locally [MMR<sup>+</sup>17, KM21]. This distributed architecture is particularly attractive when data are sensitive, geographically dispersed, or expensive to centralize. At the same time, it creates a fundamental robustness dificulty: a subset of clients may deviate arbitrarily from the prescribed protocol due to corrupted data, software or hardware faults, and even malicious behavior. Byzantine-robust FL aims to protect the learning process from such adversarial clients by replacing the aggregation average with (f, κ)-robust aggregators [CSX17, BEMGS17, YCKB18, EMGR18, GGP24], where f is the number of adversarial clients and κ > 0 is the robust aggregator coeficient. Therefore, any useful guarantee in the adversarial learning setting must account for optimization, statistical heterogeneity, and the underlying adversarial behavior.

Statistical heterogeneity makes the adversarial learning problem even more subtle. Honest clients generally have diferent data-generating models and may therefore share gradients that are genuinely far apart. A server must then distinguish this honest heterogeneity from an adversarial behavior without knowing which clients are honest. Prior analyses characterize this dificulty through a gradient-dissimilarity condition. The G-dissimilarity uniformly bounds the heterogeneity of honest gradients [KHJ20, WMA22, AFG<sup>+</sup>23, FTA25], while the $( G , B ) \cdot$ dissimilarity (formally defined in (3)) allows the heterogeneity to grow with the norm of the honest-average gradient [LSZ<sup>+</sup>20, KKM<sup>+</sup>20, AGG<sup>+</sup>23, GHRG23]. These have been shown to be useful heterogeneity abstractions, however, the bounds on G and B are imposed a priori, without any interpretability of such in terms of the underlying client model parameters, the client’s data sample size, or the label noise level, resulting in conservative guarantees.

More precisely, $\mathrm { [ A G G ^ { + } 2 3 ] }$ demonstrate that $( G , B )$ -dissimilarity provides a tight breakdown point at $1 / ( 2 + B ^ { 2 } )$ and a corresponding lower bound on the optimization error. Their upperbound analysis requires $\kappa B ^ { 2 } < 1$ , for an arbitrarily large B, which is sharp for the class of objectives satisfying the (G, B)-dissimilarity. However, it leaves open how G and B arise from the statistical learning problem and whether such restriction on $\kappa = \mathcal O ( f / n )$ , with n being the total number of clients, is necessary when gradient heterogeneity is characterized from first principles on the explicit statistical model (here we work with regression).

We address this question by studying the dynamics of gradient heterogeneity over the iterations of adversarially robust federated regression. For linear regression, for example, the gradient admits an exact decomposition into the client’s model parameter heterogeneity, the finite-sample covariance concentration, and the label-noise level. The covariance concentration is the term that couples the current optimization error to the gradient heterogeneity, with coeficient decreasing as $1 / \tau .$ , where τ is the number of fresh samples per client and round. Therefore, the analogue of $B ^ { 2 }$ in our analysis is not an arbitrary large constant set a priori: it is a concen tration parameter that can be controlled through the batch size τ .

Informal result. Let H be the set of honest clients, let T be the number of iterations/rounds, and assume that $\tau \geq \tau _ { \mathrm { b u r n - i n } }$ (with $\tau _ { \mathsf { b u r n - i n } }$ specified in Sections 3 and 4). Let $L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) : =$ $\begin{array} { r } { \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } L _ { i } \big ( \theta ^ { ( t ) } \big ) } \end{array}$ , with $L _ { i } ( \theta )$ being the regression loss evaluated at the model parameter θ. Then, for a robust coeficient $\kappa = \mathcal { O } ( f / n )$ , with $f / n < 1 / 2$ , the convergence bound holds

$$
\begin{array} { r l } & { \frac { 1 } { T } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \| \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \| _ { 2 } ^ { 2 } \lesssim \underbrace { \frac { \mathrm { i n i t i a l i z a t i o n ~ e r r o r } } { T } } _ { \mathrm { o p t i m i z a t i o n } } + \left( \kappa + \frac { \omega ^ { 2 } } { | \mathcal { H } | } \right) \times \underbrace { \mathrm { m o d e l ~ h e t e r o g e n e i t y } } _ { \mathrm { b i a s } } } \\ & { \quad \quad \quad + \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \times \underbrace { \mathrm { s t a t i s t i c a l ~ e r r o r } } _ { \mathrm { f i n i t e s a m p l e s } } , } \end{array}
$$

where $\omega ^ { 2 } = \mathcal { O } ( 1 / \tau )$ is the covariate concentration parameter. The statistical error decreases as $1 / \tau$ , up to the problem dimension, logarithmic, and conditioning factors (omitted in $\lesssim )$ stated in Sections 3 and 4. Hence, any fixed κ is admissible once the fresh-batch size is suficiently large (i.e., larger than $\tau _ { \mathsf { b u r n - i n } } )$

## 1.1 Contributions

Our contributions are summarized as follows:

• We provide a first-principles characterization of gradient heterogeneity in adversarially robust federated regression, demonstrating that it decomposes into (i) honest client’s model parameter heterogeneity, (ii) finite-sample statistical error, and (iii) an initialization-dependent error (Lemmas 3.1 and 4.1).

• Our main contribution is that, for any fixed $\kappa = \mathcal { O } ( f / n )$ , the non-asymptotic guarantees hold once each client uses a suficiently large fresh data batch (Theorems 3.1 and 4.1). In contrast, prior work $\mathrm { \left[ A G G ^ { + } 2 3 \right] }$ requires $\kappa B ^ { 2 } < 1$ , with $B > 0$ being potentially prohibitively large. The price of removing such condition on κ is explicit: it requires more samples per client.<sup>1</sup>

• We establish non-asymptotic parameter recovery error bounds that explicitly capture the interplay among honest client’s model parameter heterogeneity, client’s sample size, initialization, and robust aggregation (Corollaries 3.1 and 4.1).

Our contributions connect the underlying statistical model of regression to the gradient heterogeneity abstraction in adversarially robust FL. Before formalizing our regression model, we collect the notation used throughout the paper.

## 1.2 Notation

We use $\lVert \cdot \rVert$ to denote the Euclidean norm for vectors and the spectral/operator norm for matrices, and $\left\| \cdot \right\| _ { F }$ to denote the Frobenius norm. $\mathbb { S } _ { F } ^ { p \times q } : = \{ U \in \mathbb { R } ^ { p \times q } : \| U \| _ { F } = 1 \}$ denotes the unit sphere in $\mathbb { R } ^ { p \times q }$ equipped with the Frobenius norm. For a real-valued random variable $X$ its sub-Gaussian norm [Ver19, Definition 2.6.4] is defined as follows:

$$
\| X \| _ { \psi _ { 2 } } : = \operatorname* { i n f } \left\{ t > 0 : \mathbb { E } \left[ \exp \left( \frac { X ^ { 2 } } { t ^ { 2 } } \right) \right] \leq 2 \right\} .
$$

The random variable X is said to be sub-Gaussian if $\| X \| _ { \psi _ { 2 } } < \infty$ . For a random vector $Z \in \mathbb { R } ^ { d }$ and a random matrix $M \in \mathbb { R } ^ { p \times q }$ , we define

$$
\| Z \| _ { \psi _ { 2 } } : = \operatorname* { s u p } _ { u \in \mathbb { S } ^ { d - 1 } } \| \langle u , Z \rangle \| _ { \psi _ { 2 } } ,
$$

$$
\| M \| _ { \psi _ { 2 } } : = \operatorname* { s u p } _ { U \in \mathbb { S } _ { F } ^ { p \times q } } \| \langle U , M \rangle _ { F } \| _ { \psi _ { 2 } } .
$$

$Z$ and M are sub-Gaussian when its corresponding ψ<sub>2</sub>-norm is finite.

We use the asymptotic notation, $h = \mathcal { O } ( g )$ and $h = \Omega ( g )$ denote upper and lower bounds up to positive constants for suficiently large arguments, respectively. We write $h \lesssim g \mathrm { i f } h \leq C g$ for a constant $C > 0$

Next we introduce our statistical model and the adversarially robust FL setting with the $( f , \kappa )$ robust aggregation definition leveraged to control adversarial clients updates.

## 2 Problem Formulation

We study a synchronous server-client FL architecture in which honest clients compute gradients from fresh local data batches and the server robustly aggregates them. The objective is the population honest-average loss $L _ { \mathcal { H } } ( \cdot )$ , whose identities are unknown to the server.

Clients and adversaries. There are n clients indexed by $[ n ] = \{ 1 , \dots , n \}$ . An unknown subset $B \subset [ n ]$ of size $| B | = f$ is adversarial, with $\begin{array} { r } { f < { \frac { n } { 2 } } } \end{array}$ . The remaining clients, indexed by the set $\mathcal { H } = [ n ] \setminus B$ , are honest.

Data and samples. At every iteration $t ,$ each honest client $i \in \mathcal { H }$ draws a fresh batch of $\tau$ i.i.d. samples $\{ ( \boldsymbol { X } _ { i , k } ^ { ( t ) } , \boldsymbol { Y } _ { i , k } ^ { ( t ) } ) \} _ { k = 1 } ^ { \tau }$ from a client-specific distribution $\mathcal { D } _ { i }$ on $\mathbb { R } ^ { d } \times \mathbb { R } ^ { q }$ . The batches are independent across clients and iterations. The fresh-sample assumption ensures that, conditionally on the current iterate $\boldsymbol { \theta } ^ { ( t ) }$ , the samples used to compute the next update remain independent. For simplicity, we suppress the iteration index on the samples whenever we analyze a fixed iteration.

Losses. For a parameter θ $\ b ( \theta \in \mathbb { R } ^ { q \times d }$ for linear regression, and $\theta \in \mathbb { R } ^ { p }$ for nonlinear regression), we define the population loss of client i and the honest-average population loss as follows:

$$
L _ { i } ( \theta ) : = \mathbb { E } _ { ( X , Y ) \sim { \mathcal { D } } _ { i } } { \big [ } \ell ( \theta ; ( X , Y ) ) { \big ] } ,
$$

$$
L _ { \mathcal { H } } ( \theta ) : = \frac { 1 } { \vert \mathcal { H } \vert } \sum _ { i \in \mathcal { H } } L _ { i } ( \theta ) .
$$

At iteration $t ,$ client i computes the fresh-batch empirical loss

$$
\hat { L } _ { i } ^ { ( t ) } ( \theta ) : = \frac { 1 } { \tau } \sum _ { k = 1 } ^ { \tau } \ell \big ( \theta ; ( X _ { i , k } ^ { ( t ) } , Y _ { i , k } ^ { ( t ) } ) \big ) ,
$$

$$
\hat { L } _ { \mathcal { H } } ^ { ( t ) } ( \theta ) : = \frac { 1 } { \lvert \mathcal { H } \rvert } \sum _ { i \in \mathcal { H } } \hat { L } _ { i } ^ { ( t ) } ( \theta ) ,
$$

where throughout this paper we consider the squared losses

Linear:

$$
\ell ( \theta ; ( X , Y ) ) : = \| Y - \theta X \| _ { 2 } ^ { 2 } ,
$$

Nonlinear:

$$
\ell ( \theta ; ( X , Y ) ) : = \| Y - h _ { \theta } ( X ) \| _ { 2 } ^ { 2 } ,
$$

where $h _ { \theta } ( X )$ is a nonlinear function of the covariates $X \in \mathbb { R } ^ { d }$ parameterized by $\theta \in \mathbb { R } ^ { p }$

Assumption 2.1. $\Sigma _ { i } : = \mathbb { E } _ { X _ { i } \sim \mathcal { D } _ { i } } [ X _ { i } X _ { i } ^ { \top } ] = I _ { d } ,$ for all honest clients $i \in \mathcal { H }$

The isotropic data assumption is standard in non-asymptotic analyses of linear and nonlinear regression and can be obtained by whitening when the population covariance is known and nonsingular [Ver19, OS19]. It allows us to isolate heterogeneity in the client ground-truth model parameter rather than in the conditioning of their covariates. For nonlinear-regression, we additionally assume a common covariate marginal across honest clients.

Next we describe how the server combines the client gradients in the presence of arbitrary adversarial updates.

## 2.1 Adversarially Robust FL

We consider the class of $( f , \kappa )$ -robust aggregators defined below.

Definition 2.1 $( ( f , \kappa )$ -robust aggregator). An aggregation rule $\mathsf { F } : ( \mathbb { R } ^ { p } ) ^ { n } \to \mathbb { R } ^ { p }$ is called $( f , \kappa )$ robust, if for every set of inputs $a _ { 1 } , \dots , a _ { n } \in \mathbb { R } ^ { p }$ and every honest set H with $\vert \mathcal { H } \vert = n - f$

$$
\| \mathsf { F } ( a _ { 1 } , \ldots , a _ { n } ) - \bar { a } _ { \mathcal { H } } \| ^ { 2 } \leq \frac { \kappa } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \| a _ { i } - \bar { a } _ { \mathcal { H } } \| ^ { 2 } .
$$

The parameter κ measures how strongly the aggregation error reacts to the deviation of the honest inputs $( \mathrm { i . e . } ,$ , gradient updates). The definition is deterministic and does not restrict how adversarial clients choose their updates. Therefore, once the gradient heterogeneity is controlled, the same optimization argument applies to any aggregation rule satisfying Definition 2.1, that is, the statistical model enters only through our bound on the honest-gradient heterogeneity.

Well-known instances of $( f , \kappa )$ -robust aggregators include Krum [BEMGS17], coordinatewise trimmed mean (CWTM) [YCKB18], geometric median (GM), minimum diameter averaging, etc.. In addition, combined with the nearest neighbor mixing (NNM) approach of [AFG<sup>+</sup>23], any $( f , \kappa )$ -robust aggregator achieves $\kappa = \mathcal O ( f / n )$ , which is information-theoretically optimal as proved in $\left[ \mathrm { A F G ^ { + } 2 3 } \right]$

The adversarially robust gradient descent update of $\theta ^ { ( t ) }$ , for all iterations $t \in \{ 0 , 1 , \ldots , T - 1 \}$ , is given by

$$
\boldsymbol { \theta } ^ { ( t + 1 ) } = \boldsymbol { \theta } ^ { ( t ) } - \eta \mathsf { F } \big ( \nabla \hat { L } _ { 1 } ^ { ( t ) } ( \boldsymbol { \theta } ^ { ( t ) } ) , \dots , \nabla \hat { L } _ { n } ^ { ( t ) } ( \boldsymbol { \theta } ^ { ( t ) } ) \big ) .\tag{1}
$$

For simplicity, we use $\mathsf { F } ^ { ( t ) }$ to denote the aggregated fresh-batch gradients at iteration t in (1).

## 2.2 The Role of Gradient Heterogeneity

An important quantity in the convergence analysis of adversarially robust FL is the gradient heterogeneity and its average over iterations given by

$$
\begin{array} { l } { \displaystyle { G ^ { ( t ) } : = \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \left\| \nabla _ { \theta } \hat { L } _ { i } ^ { ( t ) } ( \theta ^ { ( t ) } ) - \nabla _ { \theta } \hat { L } _ { \mathcal { H } } ^ { ( t ) } ( \theta ^ { ( t ) } ) \right\| ^ { 2 } } . } \\ { \displaystyle { G _ { T } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } G ^ { ( t ) } } . } \end{array}\tag{2}
$$

When controlling

$$
Q _ { T } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \left. \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \right. ^ { 2 }
$$

through our ergodic convergence analysis, $G _ { T }$ will appear multiplied by the robustness coeficient κ of the aggregation rule, i.e., after leveraging the definition of $( f , \kappa )$ -robust aggregators. Therefore, understanding the structure of $G _ { T }$ is of utmost importance for our analysis.

As discussed previously, prior analyses set the bound for $G _ { T }$ with a priori fixed constants $G$ and B. For instance, G-dissimilarity [BEMGS17, YCKB18], assumes there exists a uniform $G ^ { 2 } < \infty$ such that $G ^ { ( t ) } \leq G ^ { 2 }$ for all iterations $t = 0 , 1 , \ldots , T - 1$ and all $\boldsymbol { \theta } ^ { ( t ) }$ . As $\mathrm { \left[ A G G ^ { + } 2 3 \right] }$ points out, G-dissimilarity fails for least-squares regression whenever clients have diferent datagenerating distributions (even when $f = 0 )$ , as the gradient heterogeneity grows when $\theta ^ { ( t ) }$ moves away from the optima of diferent client’s model parameters $\theta _ { i } ^ { \star }$ , for $i \in \mathcal { H }$

Therefore, $\mathrm { [ A G G ^ { + } 2 3 ] }$ , following $[ \mathrm { L S Z ^ { + } 2 0 }$ 2 $\mathrm { K K M ^ { + } 2 0 } ]$ , leverage $( G , B )$ -gradient dissimilarity:

$$
G _ { T } \leq G ^ { 2 } + \frac { B ^ { 2 } } { T } \sum _ { t = 0 } ^ { T - 1 } \left\| \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \right\| ^ { 2 } ,\tag{3}
$$

which allows the gradient heterogeneity to grow with the honest averaged gradient norm at rate $B ^ { 2 }$ . Under this dissimilarity condition, $\mathrm { [ A G G ^ { + } 2 3 ] }$ prove a tight lower bound Ω $\left( \frac { f / n } { ( 1 - ( 2 + B ^ { 2 } ) f / n ) } G ^ { 2 } \right)$ on the optimization error of any robust algorithm, and demonstrate that the largest fraction of adversarial clients should be $\frac { 1 } { 2 + B ^ { 2 } }$ rather than $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$

Here, we will shed light on the interpretability of $G$ and B. We explicitly characterize the structure of $G _ { T }$ for linear and nonlinear regression under isotropic data. In particular, for linear regression, we will demonstrate that

$$
G ^ { 2 } \lesssim \frac { 1 } { | \mathcal { H } | ^ { 2 } } \sum _ { i \in \mathcal { H } } \sum _ { j \in \mathcal { H } } \big \Vert \theta _ { i } ^ { \star } - \theta _ { j } ^ { \star } \big \Vert ^ { 2 } + \mathrm { s t a t i s t i c a l ~ e r r o r }
$$

and $B ^ { 2 } = \mathcal { O } \left( 1 / \tau \right)$ , therefore, demonstrating that with a suficiently large amount of data samples per client, the condition on $\kappa B ^ { 2 } < 1$ can be guaranteed by the number of data samples per client $\tau$ and not by the robust aggregator coeficient $\kappa .$ Such interpretability of $G$ and $B$ is the result of our first-principles analysis of the gradient dynamics in adversarially robust federated regression.

We first focus on the linear regression problem, where the honest gradients admit an exact decomposition that makes the source of each error term transparent in the analysis.

## 3 Linear Regression

The linear setting provides the cleanest illustration of our approach. We first control the empirical gradient heterogeneity, then use this control in the descent recursion, and finally translate approximate stationarity into parameter recovery error.

Each honest client $i \in \mathcal { H }$ has an unknown ground-truth parameter $\theta _ { i } ^ { \star } \in \mathbb { R } ^ { q \times d }$ and is assumed to generate τ samples according to

$$
Y _ { i , k } = \theta _ { i } ^ { \star } X _ { i , k } + V _ { i , k } { \mathrm { ~ f o r ~ a l l ~ } } k = 1 , \dots , \tau ,\tag{4}
$$

where $X _ { i , k } \in \mathbb { R } ^ { d }$ satisfies $\| X _ { i , k } \| \le R$ almost surely. Here $V _ { i , k } \in \mathbb { R } ^ { q }$ denotes the vector-valued label noise.

Assumption 3.1. For each honest client $i ,$ sample $k ,$ , and iteration t, the noise $V _ { i , k } ^ { ( t ) } \in \mathbb { R } ^ { q }$ is conditionally mean-zero and σ<sup>2</sup>-sub-Gaussian given $X _ { i , k } ^ { ( t ) } , i . e .$ , for every $u \in \mathbb { S } ^ { q - 1 }$ ，

$$
\begin{array} { r } { \mathbb { E } \left[ \exp \left( \lambda u ^ { \top } V _ { i , k } ^ { ( t ) } \right) \mid X _ { i , k } ^ { ( t ) } \right] \leq \exp \left( \frac { \lambda ^ { 2 } \sigma ^ { 2 } } { 2 } \right) , \forall \lambda \in \mathbb { R } . } \end{array}
$$

We emphasize that conditional mean-zero sub-Gaussian noise is standard in high-dimensional regression and yields dimension-explicit concentration for the empirical gradient [Ver19].

At iteration t, the fresh-batch empirical squared loss for client $i \in \mathcal { H }$ is given by

$$
\hat { L } _ { i } ^ { ( t ) } ( \theta ) : = \frac { 1 } { \tau } \sum _ { k = 1 } ^ { \tau } \left\| Y _ { i , k } ^ { ( t ) } - \theta X _ { i , k } ^ { ( t ) } \right\| ^ { 2 } ,
$$

and our goal is to minimize the honest-average population loss

$$
\operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { q \times d } } L _ { \mathcal { H } } ( \theta ) : = \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \mathbb { E } \left[ \| Y _ { i } - \theta X _ { i } \| ^ { 2 } \right] .
$$

Let $\begin{array} { r } { \Gamma _ { \sf I i n } : = \frac { 1 } { | \mathcal { H } | ^ { 2 } } \sum _ { i \in \mathcal { H } } \sum _ { j \in \mathcal { H } } \left\| \theta _ { i } ^ { \star } - \theta _ { j } ^ { \star } \right\| ^ { 2 } < \infty } \end{array}$ , denote the model heterogeneity.

Gradient Decomposition. We fix any iteration $t \in \{ 0 , 1 , \ldots , T - 1 \}$ and model parameter $\theta ^ { ( t ) }$ , and write

$$
\begin{array} { l } { \nabla _ { \theta } \hat { L } _ { i } ^ { ( t ) } ( \theta ^ { ( t ) } ) - \nabla _ { \theta } \hat { L } _ { \mathcal { H } } ^ { ( t ) } ( \theta ^ { ( t ) } ) = \displaystyle \frac { 2 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } ( \theta _ { j } ^ { \star } - \theta _ { i } ^ { \star } ) \Sigma _ { i } + \displaystyle \frac { 2 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } ( \theta _ { j } ^ { \star } - \theta _ { i } ^ { \star } ) ( \widehat { \Sigma } _ { i } - \Sigma _ { i } ) - ( \xi _ { i } - \bar { \xi } ) } \\ { + \displaystyle \frac { 2 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \left[ ( \theta ^ { ( t ) } - \theta _ { j } ^ { \star } ) ( \widehat { \Sigma } _ { i } - \Sigma _ { i } ) + ( \theta ^ { ( t ) } - \theta _ { j } ^ { \star } ) ( \Sigma _ { j } - \widehat { \Sigma } _ { j } ) \right] , } \end{array}
$$

where $\begin{array} { r } { \widehat { \Sigma } _ { i } = \frac { 1 } { \tau } \sum _ { k = 1 } ^ { \tau } X _ { i , k } X _ { i , k } ^ { \top } , \forall i \in \mathcal { H } } \end{array}$ . The noise term is composed of $\begin{array} { r } { \xi _ { i } : = \frac { 2 } { \tau } \sum _ { k = 1 } ^ { \tau } V _ { i , k } X _ { i , k } ^ { \top } \in \dag } \end{array}$ $\mathbb { R } ^ { q \times d }$ with the honest-average noise term given by $\begin{array} { r } { \bar { \xi } = \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \xi _ { i } } \end{array}$

In addition, let $\Delta L ^ { ( 0 ) } : = L _ { \mathcal { H } } ( \theta ^ { ( 0 ) } ) - L _ { \mathcal { H } } ( \theta ^ { \star } )$ , where $\theta ^ { \star }$ minimizes the honest-average population loss $L _ { { \mathcal { H } } } ( \theta )$

Lemma 3.1 (Gradient Heterogeneity). Suppose that Assumptions 2.1 and 3.1 hold. Let the step-size be such that $\begin{array} { r } { \eta \le { \frac { 1 } { 2 } } } \end{array}$ . Given constants $C _ { 0 } , C _ { 1 } > 0$ , for every $\delta \in ( 0 , 1 )$ , suppose that the number of fresh samples per client satisfies $\tau \geq$ τ<sub>burn-in</sub> with

$$
\frac { \tau _ { b u r n - i n } } { C _ { 0 } C _ { 1 } } : = ( R ^ { 2 } + 1 ) ^ { 2 } \operatorname* { m a x } \left\{ 1 , \kappa + \frac { 1 } { | \mathcal { H } | } \right\} \log \left( \frac { 2 | \mathcal { H } | T ( d + q ) } { \delta } \right) .
$$

For some $\omega > 0$ , let

$$
\omega ^ { 2 } : = C _ { 1 } ( R ^ { 2 } + 1 ) ^ { 2 } \left( \frac { \log ( 2 | \mathcal { H } | T ( d + q ) / \delta ) } { \tau } \right) .
$$

Then, with probability at least $1 - \delta$ , it holds that

$$
G _ { T } \lesssim \Gamma _ { I i n } + \omega ^ { 2 } Q _ { T } + R ^ { 2 } \sigma ^ { 2 } \left( \frac { d q + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) ,\tag{5}
$$

and consequently, it holds

$$
G _ { T } \lesssim \Gamma _ { I i n } + \omega ^ { 2 } \frac { \Delta L ^ { ( 0 ) } } { T } + R ^ { 2 } \sigma ^ { 2 } \left( \frac { d q + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) .
$$

Proof sketch. We apply the matrix Bernstein inequality [Tro12, Theorem 1.4] to the meanzero matrices $X _ { i , k } X _ { i , k } ^ { \top } - I _ { d }$ , which yields $\Vert \widehat { \Sigma } _ { i } - \Sigma _ { i } \Vert \leq \omega / 2$ simultaneously over the honest clients and iterations. For the noise matrices $V _ { i , k } X _ { i , k } ^ { \top }$ , we apply the sub-Gaussian Hoefding inequality $[ \mathrm { V e r 1 9 } ,$ , Theorem 2.7.3] to their scalar projections and take a union bound over a net of the Frobenius unit sphere. By substituting these bounds into the gradient decomposition above, then squaring and averaging over $i \in \mathcal { H }$ , gives

$$
G ^ { ( t ) } \lesssim \Gamma _ { \mathrm { l i n } } + \omega ^ { 2 } \Delta ^ { ( t ) } + R ^ { 2 } \sigma ^ { 2 } \left( \frac { d q + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) ,
$$

where $\begin{array} { r } { \Delta ^ { ( t ) } : = \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \| \theta ^ { ( t ) } - \theta _ { j } ^ { \star } \| ^ { 2 } } \end{array}$ . Then, under isotropic data, the variance decomposition and $\nabla L _ { { \mathcal { H } } } ( \theta ) = 2 ( \theta - { \bar { \theta } } ^ { \star } )$ imply $\begin{array} { r } { \Delta ^ { ( t ) } = \frac { 1 } { 4 } \| \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \| ^ { 2 } + \frac { 1 } { 2 } \Gamma _ { \mathsf { l i n } } } \end{array}$ . Averaging over t yields (5), and substituting the convergence bound below gives the second expression. The complete proof is provided in Appendix B.

Discussion of Lemma 3.1. We emphasize that this lemma is the mechanism behind our main result. Note that the iterate-dependent term (i.e., second term) in (5) is multiplied by $\omega ^ { 2 } = \mathcal { O } ( 1 / \tau )$ , while model heterogeneity and label noise enter additively. Therefore, for any fixed $\kappa = \mathcal O ( f / n )$ , with $f / n < 1 / 2$ , increasing the fresh-batch size makes the term later multiplied by κ small enough to be absorbed in the convergence argument, namely, when we bound $Q _ { T }$

With the heterogeneity term controlled, we can insert it into the robust descent recursion and obtain the following ergodic convergence guarantee.

Theorem 3.1 (Convergence Bound). Suppose that the conditions of Lemma 3.1 hold with $\begin{array} { r } { \eta = \frac { 1 } { 2 } } \end{array}$ . Then, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta _ { i }$ , it holds that

$$
Q _ { T } \lesssim \frac { \Delta L ^ { ( 0 ) } } { T } + \left( \kappa + \frac { \omega ^ { 2 } } { | \mathcal { H } | } \right) \Gamma _ { l i n } + R ^ { 2 } \sigma ^ { 2 } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \left( \frac { d q + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) .
$$

Proof. The proof is provided in Appendix B.

Discussion of Theorem 3.1. This theorem makes our tradeof explicit: $\kappa$ is not required to lie below a fixed threshold, $\mathrm { e . g . , } \ \kappa B ^ { 2 } < 1$ as in $\mathrm { [ A G G ^ { + } 2 3 ] }$ . For any fixed $\kappa ,$ the burn-in condition in Lemma 3.1 is suficient, and a larger κ is compensated by a larger sample size $\tau .$ After this burn-in, the remaining error consists of the model parameter heterogeneity bias, a statistical term that decreases with $\tau _ { : }$ , and an initialization term that goes away with $T .$

Key Takeaway. Our bounds separate the irreducible model parameter heterogeneity $\Gamma _ { \mathsf { I i n } }$ from the finite-sample statistical error and initialization-dependent error. The burn-in condition (17) demonstrates explicitly how the required fresh-batch size grows with the robust coeficient κ. More precisely, the averaged recovery error contains three diferent contributions. The term $\Delta L ^ { ( 0 ) } / T$ is transient and characterizes the efect of initialization. The term proportional to $\Gamma _ { \mathsf { I i n } }$ is irreducible when honest clients have diferent ground-truth parameters, even without adversaries (and it goes away when $\tau \to \infty \mathrm { a s } \omega \to 0 )$ . The last term is statistical and decreases as $1 / \tau$ , up to logarithmic factors. We also note that robust aggregation amplifies the two non-transient terms through κ.

The comparison with $\mathrm { [ A G G ^ { + } 2 3 ] }$ is the most relevant to this work. Their $( G , B )$ -dissimilarity condition controls the iterate-dependent heterogeneity through an a priori fixed coeficient $B ^ { 2 }$ and therefore requires $\kappa B ^ { 2 } < 1$ . Here, the corresponding coeficient $\omega ^ { 2 }$ scales as $\mathcal { O } ( 1 / \tau )$ , up to problem-dependent and logarithmic factors. We also refer the reader to (27), where we write our resulting gradient heterogeneity bound in the $( G , B )$ form:

$$
G _ { T } \lesssim \underbrace { \Gamma _ { \mathrm { l i n } } + R ^ { 2 } \sigma ^ { 2 } \left( \frac { d q + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) } _ { : = G ^ { 2 } } + \underbrace { \omega ^ { 2 } } _ { : = B ^ { 2 } } \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \left\| \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \right\| ^ { 2 } .
$$

Moreover, the burn-in condition ensures directly that $\kappa \omega ^ { 2 }$ is suficiently small. Thus, every fixed κ is admissible after a suficiently large sample burn-in. For aggregators with κ scaling as $\mathcal { O } ( f / n )$ , the adversarial contribution has the optimal linear dependence on the adversarial fraction, while the usual requirement $f < n / 2$ remains for the existence of such robust aggregator.

The bound on $Q _ { T }$ also provides a direct parameter recovery error bound under isotropic data as given in the corollary below.

Corollary 3.1 (Parameter Recovery Error). Suppose that the conditions of Theorem 3.1 hold. Then, with probability at least $1 - \delta$

$$
\begin{array} { r l r } {  { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \Big \| \theta ^ { ( t ) } - \theta _ { j } ^ { \star } \Big \| ^ { 2 } = \frac { 1 } { 4 } Q _ { T } + \frac { 1 } { 2 } \Gamma _ { I i n } } } \\ & { } & { \lesssim \frac { \Delta L ^ { ( 0 ) } } { T } + ( 1 + \kappa + \frac { \omega ^ { 2 } } { | \mathcal { H } | } ) \Gamma _ { I i n } } \\ & { } & { + R ^ { 2 } \sigma ^ { 2 } ( \kappa + \frac { 1 } { | \mathcal { H } | } ) ( \frac { d q + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } ) . } \end{array}
$$

Proof. We refer the reader to Appendix B for the proof.

## 4 Nonlinear Regression

We now extend the preceding argument beyond the linear parameterization. The proof follows the same high-level sequence, but model parameter heterogeneity is replaced by heterogeneity

between the client nonlinear functions and covariance concentration $\| \Sigma _ { i } - \widehat \Sigma _ { i } \|$ , for all $i \in \mathcal { H }$ , is replaced by the concentration of the residual-Jacobian process.

Each honest client $i \in \mathcal { H }$ has ground-truth parameter $\theta _ { i } ^ { \star } \in \mathbb { R } ^ { p }$ (i.e., a vector now) and generates data according to

$$
Y _ { i , k } = h _ { \theta _ { i } ^ { \star } } ( X _ { i , k } ) + V _ { i , k } , \forall k = 1 , \dots , \tau ,
$$

where $h _ { \theta } : \mathbb { R } ^ { d }  \mathbb { R } ^ { q }$ is a nonlinear function parameterized by $\theta \in \mathbb { R } ^ { p } , \ V _ { i , k } \in \mathbb { R } ^ { q }$ satisfies Assumption 3.1, and $\| X _ { i , k } \| \le R$ almost surely. The empirical loss is then given by

$$
L _ { i } ( \theta ) : = \frac { 1 } { \tau } \sum _ { k = 1 } ^ { \tau } \| Y _ { i , k } - h _ { \theta } ( X _ { i , k } ) \| ^ { 2 } .
$$

For the nonlinear analysis, we assume that the covariates of all honest clients have the same marginal distribution. The client heterogeneity therefore enters through the ground-truth functions $h _ { \theta _ { i } ^ { \star } } ( \cdot )$ , while every expectation denoted by $\mathbb { E } _ { X }$ is taken with respect to this common covariate distribution.

Assumption 4.1. The Jacobian given by $J _ { \theta } ( X ) : = \nabla _ { \theta } h _ { \theta } ( X ) \in \mathbb { R } ^ { p \times q }$ satisfies $\| J _ { \theta } ( X ) \| \leq \bar { J }$ for all $\theta , X$

Let $E _ { j } ^ { ( t ) } : = \mathbb { E } _ { X } \big [ \| h _ { \theta ^ { ( t ) } } ( X ) - h _ { \theta _ { j } ^ { \star } } ( X ) \| ^ { 2 } \big ] , \mathrm { f o r } j \in \mathcal { H } .$

Assumption 4.2. For every honest client $j \in \mathcal { H }$ and every iteration t, conditionally on the current iterate $\theta ^ { ( t ) }$ , the residual

$$
r _ { j } ^ { ( t ) } : = \| h _ { \theta ^ { ( t ) } } ( X ) - h _ { \theta _ { j } ^ { \star } } ( X ) \| ,
$$

is sub-Gaussian, such that

$$
\| r _ { j } ^ { ( t ) } \| _ { \psi _ { 2 } } \leq C _ { 2 } \left( \mathbb { E } _ { X } \left[ \left( r _ { j } ^ { ( t ) } \right) ^ { 2 } \right] \right) ^ { 1 / 2 } = C _ { 2 } { \sqrt { E _ { j } ^ { ( t ) } } } ,
$$

for some constant $C _ { 2 } > 0$

We note that along with the bounded Jacobian assumption (Assumption 4.1) this implies that matrices

$$
\boldsymbol { W } _ { j , k } ^ { ( t ) } - \mathbb { E } \boldsymbol { W } _ { j , k } ^ { ( t ) } \mathrm { ~ w i t h ~ } \boldsymbol { W } _ { j , k } ^ { ( t ) } : = \boldsymbol { r } _ { j } ^ { ( t ) } ( \boldsymbol { X } _ { j , k } ) \boldsymbol { J } _ { \theta ^ { ( t ) } } ( \boldsymbol { X } _ { j , k } ) ^ { \top }
$$

satisfy

$$
\left. W _ { j , k } ^ { ( t ) } - \mathbb { E } W _ { j , k } ^ { ( t ) } \right. _ { \psi _ { 2 } } \leq C _ { 3 } \bar { J } \sqrt { E _ { j } ^ { ( t ) } } ,
$$

for some constant $C _ { 3 } > 0$

We emphasize that Assumptions 4.1 and 4.2 are standard regularity conditions in nonlinear statistical learning and empirical stochastic process analyses. In particular, Assumption 4.2 corresponds to the sub-Gaussian condition on a function class, under which the ψ<sub>2</sub>-norm of function diferences is controlled by their L -norm [LM13, Men15]. The bounded Jacobian assumption imposes a uniform Lipschitz condition on the model with respect to the parameter [OS19, SH20] and holds, for example, for bounded inputs and bounded or Lipschitz neural networks on compact parameter sets.

Assumption 4.3. The honest-average population loss $L _ { \mathcal { H } }$ has an L<sup>′</sup>-Lipschitz gradient, i.e., for all $\theta , \theta ^ { \prime } \in \mathbb { R } ^ { p }$

$$
\begin{array} { r } { \left\| \nabla L _ { { \mathcal { H } } } ( { \boldsymbol { \theta } } ^ { \prime } ) - \nabla L _ { { \mathcal { H } } } ( { \boldsymbol { \theta } } ) \right\| \leq L ^ { \prime } \left\| { \boldsymbol { \theta } } ^ { \prime } - { \boldsymbol { \theta } } \right\| . } \end{array}
$$

Assumption 4.4. The honest-average loss $L _ { { \mathcal { H } } } ( \theta )$ satisfies the PL condition with constant $\mu ^ { \prime } >$ $0 , i . e .$

$$
\begin{array} { r } { \| \nabla _ { \theta } L _ { \mathcal { H } } ( \theta ) \| ^ { 2 } \geq 2 \mu ^ { \prime } \big ( L _ { \mathcal { H } } ( \theta ) - L _ { \mathcal { H } } ( \theta ^ { \star } ) \big ) , \forall \theta \in \mathbb { R } ^ { p } . } \end{array}\tag{6}
$$

Assumption 4.4 is standard in nonlinear optimization and strictly weaker than strong convexity, i.e., every µ-strongly convex function satisfies PL with $\mu ^ { \prime } = \mu$ , but not vice versa. In the linear setting $h _ { \theta } ( X ) = \theta X$ , the loss $L _ { { \mathcal { H } } } ( \theta )$ is 2-strongly convex under isotropy, hence satisfies (6) with $\mu ^ { \prime } = 2$ . In the nonlinear setting, the PL condition holds for suficiently overparameterized models whenever the neural tangent kernel is positive definite [JGH18].

$$
\Gamma _ { \sf n o n l i n } : = \frac { 1 } { | \mathcal { H } | ^ { 2 } } \sum _ { i \in \mathcal { H } } \sum _ { j \in \mathcal { H } } \mathbb { E } _ { X } \left[ \left\| h _ { \theta _ { i } ^ { \star } } ( X ) - h _ { \theta _ { j } ^ { \star } } ( X ) \right\| ^ { 2 } \right] < \infty ,
$$

denote the honest client’s model heterogeneity in nonlinear function space.

With these regularity assumptions in place, we are now ready to bound the empirical gradient heterogeneity in terms of the function-space heterogeneity $\Gamma _ { \mathsf { n o n l i n } } ,$ the statistical error, and $Q _ { T }$ (which can also be made in terms of a initialization-dependent term).

Lemma 4.1 (Gradient Heterogeneity). Suppose that Assumptions 3.1, 4.1, 4.2, 4.3, and $4 { \cdot } 4$ hold. Let the step-size be selected as $\eta \leq 1 / L ^ { \prime }$ . Given constants $C _ { 0 } , C _ { 1 } > 0$ . For every $\delta \in ( 0 , 1 )$ suppose that the number of fresh samples per client satisfies $\tau \geq$ τ<sub>burn-in</sub>,

$$
\frac { \tau _ { b u r n - i n } } { C _ { 0 } C _ { 1 } } : = \frac { \bar { J } ^ { 2 } } { \mu ^ { \prime } } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \left( p + \log \left( \frac { 2 | \mathcal { H } | T } { \delta } \right) \right) .
$$

In addition, for some $\bar { \omega } > 0$ , let

$$
\bar { \omega } ^ { 2 } : = C _ { 1 } \bar { J } ^ { 2 } \left( \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) .
$$

Then, with probability at least $1 - \delta$ , it holds

$$
G _ { T } \lesssim ( \bar { J } ^ { 2 } + \bar { \omega } ^ { 2 } ) \Gamma _ { n o n l i n } + \frac { \bar { \omega } ^ { 2 } Q _ { T } } { 2 \mu ^ { \prime } } + \bar { J } ^ { 2 } \sigma ^ { 2 } \left( \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) .\tag{7}
$$

and consequently, it holds that

$$
G _ { T } \lesssim ( \bar { J } ^ { 2 } + \bar { \omega } ^ { 2 } ) \Gamma _ { n o n l i n } + \frac { \bar { \omega } ^ { 2 } \Delta L ^ { ( 0 ) } } { \mu ^ { \prime } T } + \bar { J } ^ { 2 } \sigma ^ { 2 } \left( \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) .
$$

Proof. We defer the proof to Appendix C.

Discussion of Lemma 4.1. The nonlinear heterogeneity bound mirrors the linear decomposition in function space. The first term, ${ \bar { J } } ^ { 2 } \Gamma _ { \mathsf { n o n l i n } }$ , measures the heterogeneity among the honest ground-truth functions and persists even with infinite data. The label-noise term decreases with the fresh-batch size. The middle term is transient: $\varpi ^ { 2 }$ is an empirical-process concentration coeficient of order $1 / \tau .$ , while $\Delta L ^ { ( 0 ) } / ( \mu ^ { \prime } T )$ measures the initialization-dependent error. Thus, also in the nonlinear setting, the iterate-dependent coeficient can be made suficiently small for any fixed κ by having $\tau \geq$ τ<sub>burn-in</sub>.

Theorem 4.1 (Convergence Bound). Suppose that the conditions of Lemma 4.1 hold. Then, with probability at least 1 − δ, it holds that

$$
Q _ { T } { \lesssim } \frac { L ^ { \prime } { \Delta } L ^ { ( 0 ) } } { T } + \left[ \kappa . { \bar { J } } ^ { 2 } + { \bar { \omega } } ^ { 2 } \left( \kappa + \frac { 1 } { \vert \mathcal { H } \vert } \right) \right] \Gamma _ { n o n l i n } + \bar { J } ^ { 2 } \sigma ^ { 2 } \left( \kappa + \frac { 1 } { \vert \mathcal { H } \vert } \right) \left( \frac { p + \log ( 2 \vert \mathcal { H } \vert T / \delta ) } { \tau } \right) .
$$

Proof. We provide the proof in Appendix C.

Discussion of Theorem 4.1. The convergence theorem demonstrates that the conclusion from the linear setting persists beyond a linear parameterization. The sample-size condition makes $\bar { \omega } ^ { 2 } ( \kappa + 1 / | \mathcal { H } | ) / \mu ^ { \prime }$ small enough to absorb the second term of (7) in the inequality of $Q _ { T }$ when we conduct the ergodic convergence analysis. Therefore, for any fixed $\kappa = \mathcal O ( f / n )$ , with $f / n < 1 / 2$ , a suficiently large fresh batch removes the need for an additional upper bound on κ as required in [AGG<sup>+</sup>23]. The remaining terms consist of honest client’s function heterogeneity bias and label noise term, where the later goes away with more data samples τ per client.

Corollary 4.1 (Parameter Recovery Error). Suppose that the conditions of Theorem $\it 4 . 1$ hold. Then, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ , it holds

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \displaystyle \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \mathbb { E } _ { X } \left[ \left\| h _ { \theta ^ { ( t ) } } ( X ) - h _ { \theta _ { j } ^ { * } } ( X ) \right\| ^ { 2 } \right] \lesssim \frac { L ^ { \prime } \Delta L ^ { ( 0 ) } } { \mu ^ { \prime } T } + \left( 1 + \frac { \bar { J } ^ { 2 } \kappa } { \mu ^ { \prime } } + \frac { \bar { \omega } ^ { 2 } } { \mu ^ { \prime } } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \right) \Gamma _ { n o n l i n } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad + \frac { \bar { J } ^ { 2 } \sigma ^ { 2 } } { \mu ^ { \prime } } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \left( \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) . } \end{array}
$$

Proof. The proof is detailed in Appendix C.

Discussion of Corollary 4.1. The corollary translates approximate stationarity, i.e., the bound on $Q _ { T }$ , into recovery of the honest client functions. The term $\Gamma _ { \mathsf { n o n l i n } }$ is unavoidable: a single global model parameter $\theta \in \mathbb { R } ^ { p }$ cannot simultaneously characterize heterogeneous groundtruth functions. The additional price of adversarial robustness is explicit inflated through $\kappa ,$ while the statistical contribution vanishes with the number of fresh samples per client τ . In particular, for any fixed $\kappa ,$ homogeneous client functions and increasing $\tau$ yield vanishing averaged prediction error at the optimization rate.

Taken together, the linear and nonlinear results demonstrate that the irreducible bias is determined by the heterogeneity among honest client models $\Gamma _ { \mathsf { I n } } <$ ∞ and $\Gamma _ { \sf n o n l i n } < \infty$ , while the finite-sample and initialization-dependent component of gradient heterogeneity can be reduced through the fresh-batch size τ and the number of iterations $T ,$ respectively.

## 5 Related Work

Finally, we position our results relative to federated optimization under heterogeneity and Byzantine-robust aggregation.

FL under heterogeneity. Client heterogeneity is a crucial dificulty in federated optimization. FedAvg and its variants reduce communication but may sufer from client drift when local objectives difer [MMR<sup>+</sup>17, GHR21]. FedProx controls this drift through a proximal local objective [LSZ<sup>+</sup>20], while SCAFFOLD uses control variates to correct the bias induced by heterogeneous local updates [KKM<sup>+</sup>20]. Their analyses, and much of the subsequent literature, use bounded gradient assumptions to characterize client heterogeneity. They focus primarily on the optimization consequences of non-identical objectives in the absence of Byzantine behavior.

Our focus is complementary: we study how the empirical gradient heterogeneity itself evolves under the statistical regression model and how this quantity propagates through the adversarially robust guarantees.

Byzantine-robust distributed learning. A broad line of work designs aggregators that tolerate arbitrary client updates. Examples include Krum [BEMGS17], coordinate-wise median and trimmed mean [YCKB18], geometric-median-based aggregation [CSX17, PKH22], Bulyan [EMGR18], and momentum or bucketing-based methods [FGG<sup>+</sup>22, KHJ20, AFG<sup>+</sup>23]. These methods difer computationally and statistically, but their convergence analyses must all control the deviation between the aggregate and the mean update of the honest clients (Definition 2.1). Moreover, nearest neighbor mixing (NNM) $\left[ \mathrm { A F G ^ { + } 2 3 } \right]$ further pre-processes aggregators to have $\kappa = \mathcal O ( f / n )$ . Pre-aggregation gradient clipping is also commonly used to reduce the efect of unusually large updates. $[ \mathrm { A G G ^ { + } 2 5 } ]$ demonstrate that a fixed clipping threshold need not preserve $( f , \kappa )$ -robustness and propose adaptive clipping. Accordingly, our sample burn-in applies to any aggregator satisfying Definition 2.1.

A recent line of work also studies high-dimensional statistical rates [DD21, $\mathrm { Z W P ^ { + } 2 3 } ]$ gradient splitting $[ \mathrm { L C L ^ { + } 2 3 } ]$ , label skewness [BWH24], client subsampling and local updates $\mathrm { [ A F G ^ { + } 2 4 ] }$ , and communication compression [RGF<sup>+</sup>24] under Byzantine attacks. These works develop algorithms and guarantees for particular sources of heterogeneity. Our goal with this work is to expose the statistical structure of the gradient-heterogeneity term that appears in such robust federated optimization analyses, here with an emphasis to the regression problem.

Gradient dissimilarity. Uniform G-dissimilarity is standard in Byzantine-robust federated learning [KHJ20, DD21, AFG<sup>+</sup>23], while related bounded-dissimilarity conditions appear in federated optimization [LSZ<sup>+</sup>20, KKM<sup>+</sup>20]. However, a uniform G bound may fail even for simple regression problems. [AGG<sup>+</sup>23] address this limitation establishing tight lower and upper bounds for (G, B)-dissimilarity. Our analysis is built upon the observation that G-dissimilarity fails, but takes a diferent route: rather than assuming fixed constants G and B, we derive the gradient heterogeneity from the regression model, sample size, and noise distribution, and shed the light to the interpretation of these constants.

For linear and nonlinear regression, we derive a trajectory-dependent bound in which the irreducible bias term G is in the order of the pairwise honest client’s model parameter or function heterogeneity, while the multiplicative term B is in the order of a covariance concentration term. Such multiplicative coeficient scales inversely with τ . The condition needed to close the convergence recursion is therefore an explicit sample-size burn-in, rather than a fixed constraint on $\kappa , \mathrm { i . e . , } \kappa B ^ { 2 } < 1$ as required in $\mathrm { \left[ A G G ^ { + } 2 3 \right] }$

## 6 Conclusions and Future Work

We studied the gradient heterogeneity dynamics of adversarially robust federated regression. Rather than assuming a uniform dissimilarity bound with a priori selected constants G and B, as in (G, B)-dissimilarity, we derived the honest-gradient heterogeneity from the statistical data-generating model. For linear and nonlinear regression, the resulting bound separates model parameter heterogeneity, finite-sample noise, and an initialization-dependent term. This yields non-asymptotic convergence and recovery bounds in which the restriction on the robust aggregation coeficient imposed in prior work $\mathrm { \left[ A G G ^ { + } 2 3 \right] }$ is replaced by an explicit sample burn-in condition on τ .

Future work would consider a fixed local dataset over iterations and would control the dependence between the iterates and reused samples in the analysis. Moreover, extending the analysis to account for local model updates, partial client participation, and time-varying adversarial sets would bring the theory closer to practical federated systems.

## 7 Acknowledgments

Leonardo F. Toso is funded by the Center for AI and Responsible Financial Innovation (CAIRFI) Fellowship and the Columbia Presidential Fellowship. James Anderson is partially funded by NSF grants EECS 2144634 and CNS 2535097 and the Center of AI Technology (CAIT) in collaboration with Amazon.

## References

[AFG<sup>+</sup>23] Youssef Allouah, Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafa¨el Pinot, and John Stephan. Fixing by mixing: A recipe for optimal byzantine ml under heterogeneity. In International Conference on Artificial Intelligence and Statistics, pages 1232–1300. PMLR, 2023.

[AFG<sup>+</sup>24] Youssef Allouah, Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, Geovani Rizk, and Sasha Voitovych. Byzantine-robust federated learning: Impact of client subsampling and local updates. arXiv preprint arXiv:2402.12780, 2024.

[AGG<sup>+</sup>23] Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, and Geovani Rizk. Robust distributed learning: Tight error bounds and breakdown point under data heterogeneity. Advances in neural information processing systems, 36:45744– 45776, 2023.

[AGG<sup>+</sup>25] Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Ahmed Jellouli, Geovani Rizk, and John Stephan. Adaptive gradient clipping for robust federated learning. In International Conference on Learning Representations, volume 2025, pages 84251–84295, 2025.

[BEMGS17] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. Machine learning with adversaries: Byzantine tolerant gradient descent. Advances in neural information processing systems, 30, 2017.

[BWH24] Wenxuan Bao, Jun Wu, and Jingrui He. Boba: Byzantine-robust federated learning with label skewness. In International Conference on Artificial Intelligence and Statistics, pages 892–900. PMLR, 2024.

[CSX17] Yudong Chen, Lili Su, and Jiaming Xu. Distributed statistical machine learning in adversarial settings: Byzantine gradient descent. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 1(2):1–25, 2017.

[DD21] Deepesh Data and Suhas Diggavi. Byzantine-resilient high-dimensional sgd with local iterations on heterogeneous data. In International Conference on Machine Learning, pages 2478–2488. PMLR, 2021.

[EMGR18] El-Mahdi El-Mhamdi, Rachid Guerraoui, and S´ebastien Rouault. The hidden vulnerability of distributed learning in byzantium. In International conference on machine learning, pages 3521–3530. PMLR, 2018.

[FGG<sup>+</sup>22] Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, and John Stephan. Byzantine machine learning made easy by resilient averaging of momentums. In International Conference on Machine Learning, pages 6246–6283. PMLR, 2022.

[FTA25] Kasra Fallah, Leonardo F Toso, and James Anderson. Adversarially robust multitask adaptive control. arXiv preprint arXiv:2511.05444, 2025.

[GGP24] Rachid Guerraoui, Nirupam Gupta, and Rafael Pinot. Byzantine machine learning: A primer. ACM Computing Surveys, 56(7):1–39, 2024.

[GHR21] Eduard Gorbunov, Filip Hanzely, and Peter Richt´arik. Local sgd: Unified theory and new eficient methods. In International Conference on Artificial Intelligence and Statistics, pages 3556–3564. PMLR, 2021.

[GHRG23] Eduard Gorbunov, Samuel Horv´ath, Peter Richt´arik, and Gauthier Gidel. Variance reduction is an antidote to byzantine workers: Better rates, weaker assumptions and communication compression as a cherry on the top. In International Conference on Learning Representations, ICLR, 2023.

[JGH18] Arthur Jacot, Franck Gabriel, and Cl´ement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31, 2018.

[KHJ20] Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Byzantine-robust learning on heterogeneous datasets via bucketing. arXiv preprint arXiv:2006.09365, 2020.

[KKM<sup>+</sup>20] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scafold: Stochastic controlled averaging for federated learning. In International conference on machine learning, pages 5132– 5143. PMLR, 2020.

[KM21] Peter Kairouz and H Brendan McMahan. Advances and open problems in federated learning. Foundations and trends in machine learning, 14(1-2):1–210, 2021.

[LCL<sup>+</sup>23] Yuchen Liu, Chen Chen, Lingjuan Lyu, Fangzhao Wu, Sai Wu, and Gang Chen. Byzantine-robust learning on heterogeneous data via gradient splitting. In International Conference on Machine Learning, pages 21404–21425. PMLR, 2023.

[LM13] Guillaume Lecu´e and Shahar Mendelson. Learning subgaussian classes: Upper and minimax bounds. arXiv preprint arXiv:1305.4825, 2013.

[LSZ<sup>+</sup>20] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. Proceedings of Machine learning and systems, 2:429–450, 2020.

[Men15] Shahar Mendelson. Learning without concentration. Journal of the ACM (JACM), 62(3):1–25, 2015.

[MMR<sup>+</sup>17] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-eficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pages 1273–1282. Pmlr, 2017.

[OS19] Samet Oymak and Mahdi Soltanolkotabi. Overparameterized nonlinear learning: Gradient descent takes the shortest path? In International Conference on Machine Learning, pages 4951–4960. PMLR, 2019.

[PKH22] Krishna Pillutla, Sham M Kakade, and Zaid Harchaoui. Robust aggregation for federated learning. IEEE Transactions on Signal Processing, 70:1142–1154, 2022.

[RGF<sup>+</sup>24] Ahmad Rammal, Kaja Gruntkowska, Nikita Fedin, Eduard Gorbunov, and Peter Richt´arik. Communication compression for byzantine robust learning: New efficient algorithms and improved rates. In International Conference on Artificial Intelligence and Statistics, pages 1207–1215. PMLR, 2024.

[SH20] Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with ReLU activation function. The Annals of Statistics, 48(4):1875 – 1897, 2020.

[Tro12] Joel A Tropp. User-friendly tail bounds for sums of random matrices. Foundations of computational mathematics, 12(4):389–434, 2012.

[Ver19] Roman Vershynin. High-dimensional probability. Cambridge Series in Statistical and Probabilistic Mathematics, 47, 2019.

[WMA22] Han Wang, Siddartha Marella, and James Anderson. Fedadmm: A federated primal-dual algorithm allowing partial participation. In 2022 IEEE 61st Conference on Decision and Control (CDC), pages 287–294. IEEE, 2022.

[YCKB18] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Byzantinerobust distributed learning: Towards optimal statistical rates. In International conference on machine learning, pages 5650–5659. Pmlr, 2018.

[ZWP<sup>+</sup>23] Banghua Zhu, Lun Wang, Qi Pang, Shuai Wang, Jiantao Jiao, Dawn Song, and Michael I Jordan. Byzantine-robust federated learning with optimal statistical rates. In International Conference on Artificial Intelligence and Statistics, pages 3151–3178. PMLR, 2023.

## A Appendix Roadmap

The appendix is organized as follows. Appendix B reports the linear-regression analysis. We first recall the regression model and the gradients in Section B.1, and collect the supporting inequalities and concentration inequalities in Section B.2. Sections B.3 and B.4 establish the concentration and noise bounds used throughout the proofs. We then derive the gradient heterogeneity decomposition in Section B.5, the optimization error recursion in Section B.6, and the convergence guarantee in Section B.7. Finally, Section B.8 provides the resulting nonasymptotic parameter recovery error bounds.

Appendix C reports the nonlinear-regression analysis. Section C.1 recalls the nonlinear regression model and its gradients. Sections C.2, C.3, and C.4 derive the gradient heterogeneity bound, while in Section C.5 we establish the bound on the gradient heterogeneity and the corresponding convergence guarantee. Section C.6 concludes with the non-asymptotic functionrecovery error bound.

## B Linear Regression

We consider the parametric linear regression problem and derive non-asymptotic parameterrecovery guarantees for adversarially robust federated learning. Our analysis will combine a firstprinciples derivation of the per-iteration gradient heterogeneity with the parameter-optimization error recursion. We also establish a non-asymptotic ergodic convergence bound and compare the conditions on the robust aggregation coeficient κ obtained in our analysis to those in $\mathrm { [ A G G ^ { + } 2 3 ] }$

## B.1 Regression Model and Gradients

We recall that each honest client $i \in \mathcal { H }$ has an unknown ground-truth parameter $\theta _ { i } ^ { \star } \in \mathbb { R } ^ { q \times d }$ and is assumed to generate data according to

$$
Y _ { i , k } = \theta _ { i } ^ { \star } X _ { i , k } + V _ { i , k } \quad \mathrm { ~ f o r ~ a l l ~ } k = 1 , \ldots , \tau ,
$$

where $X _ { i , k } \in \mathbb { R } ^ { d }$ satisfies $\| X _ { i , k } \| \le R$ almost surely. Here $V _ { i , k } \in \mathbb { R } ^ { q }$ denotes a vector-valued label noise. We also recall that the empirical squared loss for client $i \in \mathcal { H }$ is given by

$$
\hat { L } _ { i } ( \theta ) : = \frac { 1 } { \tau } \sum _ { k = 1 } ^ { \tau } \left\| Y _ { i , k } - \theta X _ { i , k } \right\| ^ { 2 } .
$$

At a fixed iteration t, we suppress the iteration index on the fresh samples and define the empirical covariance and noise terms as follows:

$$
\widehat { \Sigma } _ { i } : = \frac { 1 } { \tau } \sum _ { k = 1 } ^ { \tau } X _ { i , k } X _ { i , k } ^ { \top } \in \mathbb { R } ^ { d \times d } \mathrm { ~ a n d ~ } \xi _ { i } : = \frac { 2 } { \tau } \sum _ { k = 1 } ^ { \tau } V _ { i , k } X _ { i , k } ^ { \top } \in \mathbb { R } ^ { q \times d } .
$$

We then have the following gradient expression:

$$
\nabla _ { \boldsymbol { \theta } } \hat { L } _ { i } ^ { ( t ) } ( \boldsymbol { \theta } ) = 2 ( \boldsymbol { \theta } - \boldsymbol { \theta } _ { i } ^ { \star } ) \widehat { \Sigma } _ { i } - \xi _ { i } .\tag{8}
$$

We note that, under the isotropic data assumption $\Sigma _ { i } = I _ { d }$ (Assumption 2.1), and by expanding the gradient at iteration $t \in \{ 0 , 1 , \ldots , T - 1 \}$ , we obtain

$$
\nabla _ { \boldsymbol { \theta } } \widehat { L } _ { i } ^ { ( t ) } ( { \boldsymbol { \theta } } ^ { ( t ) } ) = 2 ( { \boldsymbol { \theta } } ^ { ( t ) } - { \boldsymbol { \theta } } _ { i } ^ { \star } ) \Sigma _ { i } + 2 ( { \boldsymbol { \theta } } ^ { ( t ) } - { \boldsymbol { \theta } } _ { i } ^ { \star } ) ( \widehat { \Sigma } _ { i } - \Sigma _ { i } ) - \xi _ { i } .\tag{9}
$$

Then, the honest-average gradient is as follows:

$\nabla _ { \boldsymbol { \theta } } \hat { L } _ { \mathcal { H } } ^ { ( t ) } ( { \boldsymbol { \theta } } ^ { ( t ) } ) = \frac { 2 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } ( { \boldsymbol { \theta } } ^ { ( t ) } - { \boldsymbol { \theta } } _ { i } ^ { \star } ) \widehat { \Sigma } _ { i } - \bar { \xi } ,$ with the honest-average noise term $\bar { \xi } : = \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \xi _ { i } .$

(10)

Remark B.1 (G-gradient dissimilarity fails for linear regression). In general, G-gradient dissimilarity [KHJ20] does not hold for the linear regression model (4) under heterogeneous client parameters. Indeed, from (8) and the isotropic data assumption, we have a per-iteration gradient heterogeneity bound for

$$
G ^ { ( t ) } = \frac { 1 } { \vert \mathcal { H } \vert } \sum _ { i \in \mathcal { H } } \left. \nabla _ { \theta } L _ { i } ( \theta ^ { ( t ) } ) - \nabla _ { \theta } L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \right. ^ { 2 } ,
$$

that has a term of order $\begin{array} { r } { \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \| \theta _ { j } ^ { \star } - \theta _ { i } ^ { \star } \| ^ { 2 } } \end{array}$ from the model parameter heterogeneity, and additional terms depending on $\lVert { \boldsymbol { \theta } } ^ { ( t ) } - { \boldsymbol { \theta } } _ { j } ^ { \star } \rVert ^ { 2 }$ through the covariance deviations $\Vert \widehat { \Sigma } _ { i } - \Sigma _ { i } \Vert ^ { 2 }$ . As $\theta ^ { ( t ) }$ may move away from the optimum, the latter terms grow unboundedly, and thus no finite uniform $G ^ { 2 }$ can bound $G ^ { ( t ) }$ for all θ<sup>(t)</sup>. This is consistent with $[ A G G ^ { + } { \mathcal { Q } } { \mathcal { 3 } } ] ;$ , that discusses this limitation of G-dissimilarity for a two-client, one-dimensional motivating example.

## B.2 Supporting Inequalities

We also list some standard supporting inequalities that are leveraged throughout the derivations.

Lemma B.1 (Cauchy-Schwarz). For any nonnegative random variable $U , \mathbb { E } [ U ] \leq \sqrt { \mathbb { E } [ U ^ { 2 } ] }$

Lemma B.2 (Young’s inequality). For any $a , b \in \mathbb { R }$ and any $\gamma > 0$ 2

$$
2 a b \leq \gamma a ^ { 2 } + \frac { 1 } { \gamma } b ^ { 2 } .
$$

In addition, for any $u , v \in \mathbb { R }$ , it is also equivalent to write

$$
( u + v ) ^ { 2 } \leq ( 1 + \gamma ) u ^ { 2 } + \left( 1 + \frac { 1 } { \gamma } \right) v ^ { 2 } .
$$

Lemma B.3. Let $a _ { 1 } , \ldots , a _ { n } \in \mathbb { R } ^ { d }$ and define their average

$$
{ \bar { a } } : = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } a _ { i } .
$$

Then, the following identity holds:

$$
{ \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \| a _ { i } - { \bar { a } } \| ^ { 2 } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \| a _ { i } \| ^ { 2 } - \| { \bar { a } } \| ^ { 2 } .
$$

Lemma B.4 (Jensen’s inequality). Let X be a random variable and let $\phi$ be a convex function such that the expectations below are well defined. Then,

$$
\phi \left( \mathbb { E } [ X ] \right) \leq \mathbb { E } \left[ \phi ( X ) \right] .
$$

As a special case, for any random vector $X \in \mathbb { R } ^ { d }$

$$
\left\| \mathbb { E } [ X ] \right\| ^ { 2 } \leq \mathbb { E } \left[ \left\| X \right\| ^ { 2 } \right] .
$$

## B.3 Supporting Concentration Inequalities for Linear Regression

We now state standard concentration inequalities that are leveraged throughout our derivations.

Lemma B.5 (Matrix Bernstein - Theorem 1.4 from [Tro12]). Consider a finite sequence $\{ Z _ { k } \} _ { k }$ of independent, random, self-adjoint matrices with dimension d. We assume that each random matrix satisfies

$$
\begin{array} { r } { \mathbb { E } Z _ { k } = 0 ~ a n d ~ \lambda _ { \operatorname* { m a x } } ( Z _ { k } ) \leq L ~ a l m o s t ~ s u r e l y . } \end{array}
$$

Then, for all $t \geq 0$

$$
\mathbb { P } \left\{ \lambda _ { \operatorname* { m a x } } \left( \sum _ { k } Z _ { k } \right) \geq t \right\} \leq d \exp \left( \frac { - t ^ { 2 } / 2 } { v + L t / 3 } \right) ,
$$

where $\begin{array} { r } { \boldsymbol { v } : = \| \sum _ { k } \mathbb { E } \left( Z _ { k } ^ { 2 } \right) \| } \end{array}$

Lemma B.6 (Average of sub-Gaussian random matrices). Let $M _ { 1 } , \dots , M _ { m } \in \mathbb { R } ^ { p \times q }$ be independent, mean-zero random matrices satisfying

$$
\| M _ { j } \| _ { \psi _ { 2 } } : = \operatorname* { s u p } _ { U \in \mathbb { S } _ { F } ^ { p \times q } } \| \langle U , M _ { j } \rangle \| _ { \psi _ { 2 } } \leq K , \forall j = 1 , \ldots , m .
$$

Then there exists a constant $C _ { 1 } > 0$ such that, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ , the following bound holds:

$$
\left\| \frac { 1 } { m } \sum _ { j = 1 } ^ { m } M _ { j } \right\| _ { F } \leq C _ { 1 } K \sqrt { \frac { p q + \log ( 1 / \delta ) } { m } } .
$$

Proof. This result follows by treating each matrix $M _ { j } \in \mathbb { R } ^ { p \times q }$ as a vector in $\mathbb { R } ^ { p q }$ . For any fixed $U \in \mathbb { S } _ { F } ^ { p \times q }$ , the random variables $\langle U , M _ { j } \rangle _ { F }$ are independent, mean-zero, and sub-Gaussian with ψ<sub>2</sub>-norm bounded by K. Therefore, the sub-Gaussian Hoefding inequality [Ver19, Theorem 2.7.3] yields the following probability bound

$$
\mathbb { P } \left( \left| \left. U , \frac { 1 } { m } \sum _ { j = 1 } ^ { m } M _ { j } \right. _ { F } \right| \geq t \right) \leq 2 \exp \left( - \frac { c m t ^ { 2 } } { K ^ { 2 } } \right) .\tag{11}
$$

Next, let $\mathcal { N } _ { \xi }$ be a ξ-net [Ver19, Definition 4.2.1] of the Frobenius unit sphere. A standard volumetric estimate yields $\left| \mathcal { N } _ { \xi } \right| \leq \left( 1 + \frac { 2 } { \xi } \right) ^ { p q }$ , while the net approximation lemma given in [Ver19, Lemma 4.4.1] provides

$$
\left\| \frac { 1 } { m } \sum _ { j = 1 } ^ { m } M _ { j } \right\| _ { F } \leq \frac { 1 } { 1 - \xi } \operatorname* { m a x } _ { U \in \cal N _ { \xi } } \left\| \left. U , \frac { 1 } { m } \sum _ { j = 1 } ^ { m } M _ { j } \right. _ { F } \right\|
$$

Hence, by taking a union bound over $\mathcal { N } _ { \xi }$ , with $\begin{array} { r } { \xi = \frac { 1 } { 2 } } \end{array}$ , and by selecting $t \asymp K \sqrt { \frac { p q + \log ( 1 / \delta ) } { m } }$ ， with probability at least $1 - \delta _ { \pmb { \mathscr { s } } }$ we have $\begin{array} { r } { \left\| \frac { 1 } { m } \sum _ { j = 1 } ^ { m } M _ { j } \right\| _ { F } \leq C _ { 1 } K \sqrt { \frac { p q + \log ( 1 / \delta ) } { m } } } \end{array}$ □

Lemma B.7 (Covariance concentration). Suppose that the isotropic data assumption $( i . e . ,$ $\Sigma _ { i } = I _ { d } , \forall i \in \mathcal { H }$ in Assumption 2.1) holds. Then, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ , simultaneously for all $i \in \mathcal { H }$ and $t = 0 , \dots , T - 1$ , we have

$$
\Big \| \widehat { \Sigma } _ { i } - \Sigma _ { i } \Big \| _ { 2 } \leq 2 ( R ^ { 2 } + 1 ) \left( \sqrt { \frac { \log ( 2 | \mathcal { H } | T d / \delta ) } { \tau } } + \frac { \log ( 2 | \mathcal { H } | T d / \delta ) } { \tau } \right) .\tag{12}
$$

Proof. We begin by defining the mean-zero self-adjoint matrices $Z _ { i , k } : = X _ { i , k } X _ { i , k } ^ { \top } - I _ { d } .$ , such that $\begin{array} { r } { \widehat { \Sigma } _ { i } - \Sigma _ { i } = \frac { 1 } { \tau } \sum _ { k = 1 } ^ { \tau } Z _ { i , k } } \end{array}$ . The spectral norm of each $Z _ { i , k }$ is bounded as $\| Z _ { i , k } \| _ { 2 } \leq R ^ { 2 } + 1$ . We also define the variance $\begin{array} { r } { \boldsymbol { v } = \left. \sum _ { k } \mathbb { E } [ Z _ { i , k } ^ { 2 } ] \right. _ { 2 } } \end{array}$ . Note that since $Z _ { i , k } ^ { 2 } \preceq ( R ^ { 4 } + 2 R ^ { 2 } + 1 ) I _ { d }$ , we obtain $v \leq ( R ^ { 2 } + 1 ) ^ { 2 } \tau$ . Therefore, applying the matrix Bernstein inequality in Lemma B.5 (see also [Tro12, Theorem 1.4]) to $Z _ { i , k }$ and $- Z _ { i , k }$ and union bounding over honest clients $i \in \mathcal { H }$ and iterations $t = 0 , \ldots , T - 1$ , yield the expression in (12). □

## B.4 Bound for the Noise

We now turn our attention to bound the σ-sub-Gaussian noise term.

Lemma B.8. Suppose that Assumption 3.1 and $\| X _ { i , k } \| \le R$ hold almost surely. Then there exists a constant $C _ { 1 } > 0$ such that, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta _ { i }$ simultaneously for all honest clients and iterations, we have

$$
\| \xi _ { i } \| + \left\| \bar { \xi } \right\| \leq C _ { 1 } R \sigma \sqrt { \frac { d q + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } } \left( 1 + \frac { 1 } { \sqrt { | \mathcal { H } | } } \right) .
$$

Then, by triangle inequality, $\left\| \xi _ { i } - \bar { \xi } \right\| \leq \| \xi _ { i } \| + \left\| \bar { \xi } \right\|$ , which provides the same bound.

Proof. We apply Lemma B.6 to the matrices $\begin{array} { r } { \xi _ { i } = \frac { 2 } { \tau } \sum _ { k = 1 } ^ { \tau } V _ { i , k } X _ { i , k } ^ { \top } \in \mathbb { R } ^ { q \times d } } \end{array}$ , for all $i \in \mathcal { H }$ . For this, let us fix $U \in \mathbb { S } _ { F } ^ { q \times d }$ and write

$$
\langle U , \xi _ { i } \rangle = \frac { 2 } { \tau } \sum _ { k = 1 } ^ { \tau } V _ { i , k } ^ { \top } U X _ { i , k } .
$$

Note that as $\left. U X _ { i , k } \right. _ { 2 } \leq \left. U \right. _ { F } \left. X _ { i , k } \right. _ { 2 } \leq R .$ the scalar $V _ { i , k } ^ { \top } U X _ { i , k }$ is mean-zero sub-Gaussian with $\left. V _ { i , k } ^ { \top } U X _ { i , k } \right. _ { \psi _ { 2 } } \leq R \sigma$ . By the Hoefding inequality for sub-Gaussian random variables (11) (see also [Ver19, Theorem 2.2.1]), we obtain $\| \xi _ { i } \| _ { \psi _ { 2 } } \leq 2 C R \sigma / \sqrt { \tau }$ for some $C > 0$ . Therefore, applying Lemma B.6 with $m = \tau , p = q$ , and $q = d ,$ we obtain the following bound:

$$
\| \xi _ { i } \| _ { F } \leq C _ { 1 } R \sigma \sqrt { \frac { d q + \log ( 1 / \delta ) } { \tau } } .\tag{13}
$$

In addition, as $\{ \xi _ { j } \} _ { j \in \mathcal { H } }$ are independent across honest clients, $\bar { \xi }$ is an average of $| { \mathcal { H } } |$ independent mean-zero sub-Gaussian matrices with the same sub-Gaussian norm. Therefore, Lemma B.6 provides

$$
\big \| \bar { \xi } \big \| _ { F } \leq C _ { 1 } R \sigma \sqrt { \frac { d q + \log ( 1 / \delta ) } { \tau | \mathcal { H } | } } .\tag{14}
$$

The proof is then completed by combining (13)-(14) and absorbing any constants in $C _ { 1 }$

## B.5 Gradient Heterogeneity Decomposition

We begin by subtracting (10) from (9) and using the isotropic data assumption $\Sigma _ { i } = I _ { d }$ for all $i \in \mathcal { H }$ (Assumption 2.1) as follows:

$$
\nabla _ { \boldsymbol { \theta } } \hat { L } _ { i } ^ { ( t ) } ( \boldsymbol { \theta } ^ { ( t ) } ) - \nabla _ { \boldsymbol { \theta } } \hat { L } _ { \mathcal { H } } ^ { ( t ) } ( \boldsymbol { \theta } ^ { ( t ) } ) = \frac { 2 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } ( \boldsymbol { \theta } _ { j } ^ { \star } - \boldsymbol { \theta } _ { i } ^ { \star } ) \Sigma _ { i } - ( \xi _ { i } - \bar { \xi } )
$$

$$
\begin{array} { l } { { \displaystyle + \frac { 2 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } ( \theta ^ { ( t ) } - \theta _ { j } ^ { \star } ) ( \widehat \Sigma _ { i } - \Sigma _ { i } ) } } \\ { { \displaystyle + \frac { 2 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } ( \theta ^ { ( t ) } - \theta _ { j } ^ { \star } ) ( \Sigma _ { j } - \widehat \Sigma _ { j } ) } } \\ { { \displaystyle + \frac { 2 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } ( \theta _ { j } ^ { \star } - \theta _ { i } ^ { \star } ) ( \widehat \Sigma _ { i } - \Sigma _ { i } ) . } } \end{array}
$$

Therefore, by taking norms and applying the triangle inequality, we obtain

$$
\begin{array} { r l } { \left\| \nabla \sigma \widehat { u } _ { \xi } ^ { ( t ) } ( \theta ^ { ( t ) } ) - \nabla \circ \widehat { u } _ { \xi } \widehat { L } _ { \mathcal { H } } ^ { ( t ) } ( \theta ^ { ( t ) } ) \right\| \leq \displaystyle \frac { 2 } { | M | } \displaystyle \sum _ { j \in M } \left\| \beta _ { j } ^ { ( t ) } - \theta _ { i } ^ { * } \right\| - \left\| \underline { { \xi } } _ { \mathcal { S } } - \overline { { \xi } } \right\| } & { } \\ & { \qquad \times \displaystyle \operatorname* { m a x } _ { i \in M } \left| \nabla \circ \widehat { u } _ { \xi } \widehat { L } _ { \mathcal { S } } \right| } \\ & { + \displaystyle \frac { 2 } { | M | } \displaystyle \sum _ { j \in M } \left\| z _ { j } ^ { ( t ) } - \widehat { L } _ { \xi } \right\| \left\| \theta ^ { ( t ) } - \theta _ { j } ^ { * } \right\| } \\ & { \displaystyle \quad \times \displaystyle \operatorname* { m a x } _ { i \in M } \left( \operatorname* { c a v o u n } _ { i \in M + 1 } \right) } \\ & { + \displaystyle \frac { 2 } { | M | } \displaystyle \sum _ { j \in M } \left\| z _ { j } ^ { ( t ) } - \widehat { L } _ { \xi } \right\| \left\| \theta ^ { ( t ) } - \theta _ { j } ^ { * } \right\| } \\ & { + \displaystyle \frac { 2 } { | M | } \displaystyle \sum _ { j \in M } \left\| z _ { j } ^ { ( t ) } - \widehat { L } _ { \xi } \right\| \left\| \theta _ { j } ^ { ( t ) } - \theta _ { j } ^ { * } \right\| } \\ & { + \displaystyle \frac { 2 } { | M | } \displaystyle \sum _ { j \in M } \left\| z _ { j } ^ { ( t ) } - \widehat { L } _ { \xi } \right\| \left\| \theta _ { j } ^ { * } - \theta _ { j } ^ { * } \right\| . } \end{array}\tag{15}
$$

Burn-in time condition. Fix $\delta \in ( 0 , 1 )$ and, for some $\omega > 0$ , let

$$
\omega ^ { 2 } : = C _ { 1 } ( R ^ { 2 } + 1 ) ^ { 2 } \Bigg ( \frac { \log ( 2 | \mathcal { H } | T ( d + q ) / \delta ) } { \tau } \Bigg ) .\tag{16}
$$

$$
\tau \geq C _ { 0 } C _ { 1 } ( R ^ { 2 } + 1 ) ^ { 2 } \operatorname* { m a x } \bigg \{ 1 , \kappa + \frac { 1 } { | \mathcal { H } | } \bigg \} \log \bigg ( \frac { 2 | \mathcal { H } | T ( d + q ) } { \delta } \bigg ) .\tag{17}
$$

Here, $C _ { 0 } > 0$ is a suficiently large constant. Thus, (17) guarantees $\omega \leq 1$ and

$$
\omega ^ { 2 } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \leq \frac { 1 } { C _ { 0 } } ,
$$

where $C _ { 0 }$ can be chosen large enough to absorb all constants appearing below.

Therefore, under the burn-in time condition (17), Lemma B.7 yields $\begin{array} { r } { \left\| \widehat { \Sigma } _ { i } - \Sigma _ { i } \right\| \leq \frac { \omega } { 2 } } \end{array}$ and $\begin{array} { r } { \left\| \widehat { \Sigma } _ { j } - \Sigma _ { j } \right\| \leq \frac { \omega } { 2 } } \end{array}$ , for all $i , j \in { \mathcal { H } }$ . Then the covariance concentration and cross terms in (15) are absorbed as follows:

$$
\left. \nabla _ { \theta } \hat { L } _ { i } ( \theta ^ { ( t ) } ) - \nabla _ { \theta } \hat { L } _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \right. \leq \frac { 3 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \left. \theta _ { j } ^ { \star } - \theta _ { i } ^ { \star } \right. + \left. \xi _ { i } - \bar { \xi } \right. + \frac { 2 \omega } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \left. \theta ^ { ( t ) } - \theta _ { j } ^ { \star } \right. .\tag{18}
$$

where we consider $\omega \leq 1$ for the first term. We reemphasize that, as discussed in Remark B.1, the gradient heterogeneity bound has a model-heterogeneity term of order $\begin{array} { r } { \frac { 1 } { \left| \mathcal { H } \right| } \sum _ { j \in \mathcal { H } } \left\| \theta _ { j } ^ { \star } - \theta _ { i } ^ { \star } \right\| } \end{array}$ as well as an additional iterate-dependent term that scales as $\begin{array} { r } { \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \left| \left| \theta ^ { ( t ) } - \theta _ { j } ^ { \star } \right| \right| } \end{array}$ . Therefore,

if $\boldsymbol { \theta } ^ { ( t ) }$ moves arbitrarily far from the optimum $\theta _ { j } ^ { \star }$ of any honest client $j \in \mathcal { H }$ , then even in this simple linear regression setting, the gradient heterogeneity bound need not remain uniformly bounded.

To complete the bound for $G ^ { ( t ) }$ , we square both sides of (18), apply $\left\| a + b + c \right\| ^ { 2 } \leq 3 \left\| a \right\| ^ { 2 } +$ $3 \| b \| ^ { 2 } + 3 \| c \| ^ { 2 }$ , and average over honest clients $i \in \mathcal { H }$ . Therefore, the per-iteration gradient heterogeneity satisfies

$$
G ^ { ( t ) } \leq \underbrace { \frac { 2 7 } { \vert \mathcal { H } \vert ^ { 2 } } \sum _ { i \in \mathcal { H } } \sum _ { j \in \mathcal { H } } \Vert \theta _ { j } ^ { \star } - \theta _ { i } ^ { \star } \Vert ^ { 2 } } _ { = 2 \mathcal { T } \Gamma _ { \mathsf { f m } } } + 3 C _ { 1 } R ^ { 2 } \sigma ^ { 2 } \left( \frac { d q + \log ( 2 \vert \mathcal { H } \vert / \delta ) } { \tau } \right) \left( 1 + \frac { 1 } { \vert \mathcal { H } \vert } \right) + \frac { 3 \omega ^ { 2 } } { \vert \mathcal { H } \vert } \sum _ { j \in \mathcal { H } } \left. \theta ^ { ( t ) } - \theta _ { j } ^ { \star } \right. ^ { 2 } ,\tag{19}
$$

where we recall that the model heterogeneity as follows:

$$
\Gamma _ { \mathsf { l i n } } : = \frac { 1 } { | \mathcal { H } | ^ { 2 } } \sum _ { i \in \mathcal { H } } \sum _ { j \in \mathcal { H } } \big \| \theta _ { i } ^ { \star } - \theta _ { j } ^ { \star } \big \| ^ { 2 } .
$$

## B.6 Optimization Error Recursion

We now relate the parameter error, i.e., $\left\| \theta ^ { ( t ) } - \theta _ { j } ^ { \star } \right\|$ , directly to the gradient of the population loss. Let

$$
\bar { \theta } ^ { \star } : = \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \theta _ { i } ^ { \star } .
$$

We note that under isotropy and conditional mean-zero noise, we have

$$
\nabla L _ { { \mathcal { H } } } ( \theta ) = 2 ( \theta - { \bar { \theta } } ^ { \star } ) .\tag{20}
$$

Moreover, the variance decomposition in Lemma B.3 provides the following identity.

$$
\Delta ^ { ( t ) } : = \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \Big \| \theta ^ { ( t ) } - \theta _ { j } ^ { \star } \Big \| ^ { 2 } = \frac { 1 } { 4 } \left\| \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \right\| ^ { 2 } + \frac { 1 } { 2 } \Gamma _ { \mathsf { l i n } } .\tag{21}
$$

We then substitute (21) into (19) to obtain

$$
G ^ { ( t ) } \lesssim \Gamma _ { \mathrm { l i n } } + \omega ^ { 2 } \left\| \nabla L _ { { \mathcal { H } } } ( \theta ^ { ( t ) } ) \right\| ^ { 2 } + R ^ { 2 } \sigma ^ { 2 } \left( \frac { d q + \log ( 2 | { \mathcal { H } } | T / \delta ) } { \tau } \right) ,\tag{22}
$$

where all constants are adsorbed into $\lesssim$

Let us also define the honest-average gradient error

$$
\begin{array} { r } { Z ^ { ( t ) } : = \left. \nabla \hat { L } _ { \mathcal { H } } ^ { ( t ) } ( { \boldsymbol { \theta } } ^ { ( t ) } ) - \nabla L _ { \mathcal { H } } ( { \boldsymbol { \theta } } ^ { ( t ) } ) \right. ^ { 2 } . } \end{array}\tag{23}
$$

To bound this quantity, we first subtract the population gradient (20) from the honest-average fresh-batch gradient. This yields the following expression:

$$
\nabla \hat { L } _ { \mathcal { H } } ^ { ( t ) } ( \theta ^ { ( t ) } ) - \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) = \frac { 2 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } ( \theta ^ { ( t ) } - \theta _ { i } ^ { \star } ) ( \hat { \Sigma } _ { i } - I _ { d } ) - \bar { \xi } .
$$

Therefore, using $\left\| a + b \right\| ^ { 2 } \leq 2 \left\| a \right\| ^ { 2 } + 2 \left\| b \right\| ^ { 2 }$ , we obtain

$$
Z ^ { ( t ) } \leq 8 \left\| \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } ( \theta ^ { ( t ) } - \theta _ { i } ^ { \star } ) ( \hat { \Sigma } _ { i } - I _ { d } ) \right\| ^ { 2 } + 2 \left\| \bar { \xi } \right\| ^ { 2 } .
$$

We proceed to bound the first term. To do so, let us define the following quantities $A _ { i } ^ { ( t ) } : = $ $\theta ^ { ( t ) } - \theta _ { i } ^ { \star }$ and recall from Lemma B.7 that $Z _ { i , k } : = X _ { i , k } X _ { i , k } ^ { \top } - I _ { d }$ , and $M _ { i , k } ^ { ( t ) } : = A _ { i } ^ { ( t ) } Z _ { i , k }$ . Then, we can write

$$
\frac { 1 } { \vert \mathcal { H } \vert } \sum _ { i \in \mathcal { H } } A _ { i } ^ { ( t ) } ( \widehat { \Sigma } _ { i } - I _ { d } ) = \frac { 1 } { \vert \mathcal { H } \vert \tau } \sum _ { i \in \mathcal { H } } \sum _ { k = 1 } ^ { \tau } M _ { i , k } ^ { ( t ) } .
$$

We then recall that the fresh-sample assumption makes the matrices $M _ { i , k } ^ { ( t ) }$ independent conditionally on $\boldsymbol { \theta } ^ { ( t ) }$ , while the data isotropy assumption implies

$$
\mathbb { E } [ Z _ { i , k } \ | \ \theta ^ { ( t ) } ] = 0 \ \mathrm { a n d } \ \mathbb { E } [ M _ { i , k } ^ { ( t ) } \ | \ \theta ^ { ( t ) } ] = 0 .
$$

We can also define the self-adjoint matrix

$$
\begin{array} { r } { \mathcal { S } ( M _ { i , k } ^ { ( t ) } ) : = \left( \begin{array} { c c } { 0 } & { M _ { i , k } ^ { ( t ) } } \\ { M _ { i , k } ^ { ( t ) \top } } & { 0 } \end{array} \right) , } \end{array}
$$

where we have $\lVert S ( M _ { i , k } ^ { ( t ) } ) \rVert = \lVert M _ { i , k } ^ { ( t ) } \rVert$ . As we assume that $\| X _ { i , k } \| \le R$ , we have

$$
\| Z _ { i , k } \| \leq R ^ { 2 } + 1 \ \mathrm { a n d } \ \| M _ { i , k } ^ { ( t ) } \| \leq ( R ^ { 2 } + 1 ) \| A _ { i } ^ { ( t ) } \| .
$$

In addition, note that we can write

$$
\begin{array} { r l } & { M _ { i , k } ^ { ( t ) } ( M _ { i , k } ^ { ( t ) } ) ^ { \top } = A _ { i } ^ { ( t ) } Z _ { i , k } ^ { 2 } ( A _ { i } ^ { ( t ) } ) ^ { \top } , } \\ & { ( M _ { i , k } ^ { ( t ) } ) ^ { \top } M _ { i , k } ^ { ( t ) } = Z _ { i , k } ( A _ { i } ^ { ( t ) } ) ^ { \top } A _ { i } ^ { ( t ) } Z _ { i , k } , } \end{array}
$$

which implies that both diagonal blocks of $\mathbb { E } [ S ( M _ { i , k } ^ { ( t ) } ) ^ { 2 } \mid \theta ^ { ( t ) } ]$ have norm at most $( R ^ { 2 } + 1 ) ^ { 2 } \Vert A _ { i } ^ { ( t ) } \Vert ^ { 2 }$ Thus, we can write

$$
v _ { t } : = \left\| \sum _ { i \in \mathcal { H } } \sum _ { k = 1 } ^ { \tau } \mathbb { E } \left[ S ( M _ { i , k } ^ { ( t ) } ) ^ { 2 } \mid \theta ^ { ( t ) } \right] \right\| \leq \tau b ^ { 2 } \sum _ { i \in \mathcal { H } } \| A _ { i } ^ { ( t ) } \| ^ { 2 } .
$$

Let us then set $\begin{array} { r } { \ell : = \log \left( \frac { 2 | \mathcal { H } | T ( d + q ) } { \delta } \right) } \end{array}$ . We are now ready to Lemma B.5 and take union bound over $t = 0 , \ldots , T - 1$ to obtain

$$
\left\| \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } A _ { i } ^ { ( t ) } ( \hat { \Sigma } _ { i } - I _ { d } ) \right\| \lesssim b \sqrt { \frac { \ell } { | \mathcal { H } | \tau } \left( \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \| A _ { i } ^ { ( t ) } \| ^ { 2 } \right) } + b \frac { \ell } { | \mathcal { H } | \tau } \operatorname* { m a x } _ { i \in \mathcal { H } } \| A _ { i } ^ { ( t ) } \| .
$$

Moreover, we have

$$
\operatorname* { m a x } _ { i \in \mathcal { H } } \| A _ { i } ^ { ( t ) } \| \leq \left( \sum _ { i \in \mathcal { H } } \| A _ { i } ^ { ( t ) } \| ^ { 2 } \right) ^ { 1 / 2 } = \sqrt { | \mathcal { H } | \Delta ^ { ( t ) } } .
$$

We then note that under the burn-in condition (17), the second term is dominated by the first. Therefore, we have

$$
\left\| \frac { 1 } { m } \sum _ { i \in \mathcal { H } } A _ { i } ^ { ( t ) } ( \hat { \Sigma } _ { i } - I _ { d } ) \right\| \lesssim b \sqrt { \frac { \ell } { | \mathcal { H } | \tau } \Delta ^ { ( t ) } } .
$$

By squaring this inequality and using the definition of $\omega ^ { 2 } ~ ( 1 6 )$ , we obtain

$$
\left\| \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } ( \theta ^ { ( t ) } - \theta _ { i } ^ { \star } ) ( \hat { \Sigma } _ { i } - I _ { d } ) \right\| ^ { 2 } \lesssim \frac { ( R ^ { 2 } + 1 ) ^ { 2 } \ell } { \tau | \mathcal { H } | } \Delta ^ { ( t ) } \lesssim \frac { \omega ^ { 2 } } { | \mathcal { H } | } \Delta ^ { ( t ) } .
$$

In addition, we recall that Lemma B.8 implies

$$
\left\| \bar { \xi } \right\| ^ { 2 } \lesssim R ^ { 2 } \sigma ^ { 2 } \left( \frac { d q + \log ( 2 T / \delta ) } { \tau | \mathcal { H } | } \right) ,
$$

Therefore, combining these bounds and substituting (21), we obtain

$$
Z ^ { ( t ) } \lesssim \frac { \omega ^ { 2 } } { \vert \mathscr { H } \vert } \left. \nabla L _ { \mathscr { H } } ( \theta ^ { ( t ) } ) \right. ^ { 2 } + \frac { \omega ^ { 2 } } { \vert \mathscr { H } \vert } \Gamma _ { \mathsf { l i n } } + R ^ { 2 } \sigma ^ { 2 } \left( \frac { d q + \log ( 2 T / \delta ) } { \tau \vert \mathscr { H } \vert } \right) .\tag{24}
$$

## B.7 Convergence Analysis

Let $e ^ { ( t ) } : = \mathsf { F } ^ { ( t ) } - \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } )$ . By $( f , \kappa )$ -robustness (Definition 2.1) and (23), we have

$$
\left\| e ^ { ( t ) } \right\| ^ { 2 } \leq 2 \kappa G ^ { ( t ) } + 2 Z ^ { ( t ) } .\tag{25}
$$

The population loss is 2-smooth under isotropy. Thus, for $\eta \leq 1 / 2$ , we have

$$
L _ { { \mathcal { H } } } ( \boldsymbol { \theta } ^ { ( t + 1 ) } ) - L _ { { \mathcal { H } } } ( \boldsymbol { \theta } ^ { ( t ) } ) \leq - \frac { \eta } { 2 } \left\| \nabla L _ { { \mathcal { H } } } ( \boldsymbol { \theta } ^ { ( t ) } ) \right\| ^ { 2 } + \frac { \eta } { 2 } \left\| \boldsymbol { e } ^ { ( t ) } \right\| ^ { 2 } .\tag{26}
$$

Theorem B.1. Suppose that Assumptions 2.1 and 3.1 hold, $\eta \leq 1 / 2$ , and the burn-in condition (17) holds. Then, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$

$$
Q _ { T } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \left\| \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \right\| ^ { 2 } \lesssim \frac { \Delta L ^ { ( 0 ) } } { \eta T } + \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \Gamma _ { l i n } + R ^ { 2 } \sigma ^ { 2 } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \left( \frac { d q + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right)
$$

Proof. We begin by summing (26) over $t = 0 , \ldots , T - 1$ to write

$$
\frac { \eta } { 2 } \sum _ { t = 0 } ^ { T - 1 } \left\| \nabla L \varkappa ( \theta ^ { ( t ) } ) \right\| ^ { 2 } \leq L _ { \mathcal { H } } ( \theta ^ { ( 0 ) } ) - L _ { \mathcal { H } } ( \theta ^ { ( T ) } ) + \frac { \eta } { 2 } \sum _ { t = 0 } ^ { T - 1 } \left\| e ^ { ( t ) } \right\| ^ { 2 }
$$

$$
\leq \Delta L ^ { ( 0 ) } + \frac { \eta } { 2 } \sum _ { t = 0 } ^ { T - 1 } \left. e ^ { ( t ) } \right. ^ { 2 } ,
$$

where we recall that $\Delta L ^ { ( 0 ) } : = L _ { { \mathcal { H } } } ( \theta ^ { ( 0 ) } ) - L _ { { \mathcal { H } } } ( \theta ^ { \star } )$ . The second inequality uses the optimality of $\theta ^ { \star }$ . Then, dividing by $\eta T / 2$ and applying (25), we obtain

$$
Q _ { T } \leq \frac { 2 \Delta L ^ { ( 0 ) } } { \eta T } + \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \Big \Vert e ^ { ( t ) } \Big \Vert ^ { 2 } \leq \frac { 2 \Delta L ^ { ( 0 ) } } { \eta T } + 2 \kappa G _ { T } + 2 Z _ { T } ,
$$

where $\begin{array} { r } { Z _ { T } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } Z ^ { ( t ) } } \end{array}$ . By averaging (22) and (24) over the iterations gives

$$
\begin{array} { r l } & { G _ { T } \lesssim \Gamma _ { \mathrm { l i n } } + R ^ { 2 } \sigma ^ { 2 } \left( \frac { d q + \log \left( 2 \left| \mathcal { H } \right| T / \delta \right) } { \tau } \right) + \omega ^ { 2 } Q _ { T } , } \\ & { Z _ { T } \lesssim \frac { \omega ^ { 2 } } { \left| \mathcal { H } \right| } Q _ { T } + \frac { \omega ^ { 2 } } { \left| \mathcal { H } \right| } \Gamma _ { \mathrm { l i n } } + R ^ { 2 } \sigma ^ { 2 } \left( \frac { d q + \log \left( 2 \left| \mathcal { H } \right| T / \delta \right) } { \tau \left| \mathcal { H } \right| } \right) . } \end{array}\tag{27}
$$

Therefore, by substituting these two bounds into the preceding inequality, we obtain

$$
Q _ { T } \lesssim \frac { \Delta L ^ { ( 0 ) } } { \eta T } + \left( \kappa + \frac { \omega ^ { 2 } } { | \mathcal { H } | } \right) \Gamma _ { \mathrm { f i n } } + R ^ { 2 } \sigma ^ { 2 } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \left( \frac { d q + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) + \omega ^ { 2 } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) Q _ { T } .
$$

By (17) we have $\omega ^ { 2 } ( \kappa + 1 / | \mathcal { H } | ) \leq 1 / C _ { 0 }$ , and choosing $C _ { 0 }$ suficiently large makes the coeficient of $Q _ { T }$ strictly smaller than one. We can therefore move the final term to the left-hand side and absorb the resulting numerical factor into the implicit constant absorbed into $\lesssim$ . This proves the stated bound. □

## B.8 Non-Asymptotic Parameter Recovery Error Bound

We now state the recovery guarantee separately from the convergence analysis. We first note that under data isotropy, parameter error and prediction error coincide because, for any $\theta \in$ $\mathbb { R } ^ { q \times d }$ we can write

$$
\operatorname { \mathbb { E } } _ { X } \left[ \left\| ( \theta - \theta _ { j } ^ { \star } ) X \right\| ^ { 2 } \right] = \operatorname { T r } \left( ( \theta - \theta _ { j } ^ { \star } ) ^ { \top } ( \theta - \theta _ { j } ^ { \star } ) \mathbb { E } [ X X ^ { \top } ] \right) = \left\| \theta - \theta _ { j } ^ { \star } \right\| ^ { 2 } .
$$

Corollary B.1 (Linear Parameter Recovery Bound). Suppose that the conditions of Theorem B.1 hold. Then, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ 1

$$
\begin{array} { r l r } {  { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \| \theta ^ { ( t ) } - \theta _ { j } ^ { * } \| ^ { 2 } = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \mathbb { E } _ { X } [ \| ( \theta ^ { ( t ) } - \theta _ { j } ^ { * } ) X \| ^ { 2 } ] } } \\ & { } & { \lesssim \frac { \Delta L ^ { ( 0 ) } } { \eta T } + ( 1 + \kappa + \frac { 1 } { | \mathcal { H } | } ) \Gamma _ { \mathit { i n } } + R ^ { 2 } \sigma ^ { 2 } ( \kappa + \frac { 1 } { | \mathcal { H } | } ) ( \frac { d q + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } ) . } \end{array}
$$

Proof. For every iteration t, adding and subtracting $\bar { \theta } ^ { \star }$ and applying Lemma B.3 give

$$
\frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \left\| \theta ^ { ( t ) } - \theta _ { j } ^ { \star } \right\| ^ { 2 } = \left\| \theta ^ { ( t ) } - \bar { \theta } ^ { \star } \right\| ^ { 2 } + \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \left\| \theta _ { j } ^ { \star } - \bar { \theta } ^ { \star } \right\| ^ { 2 } .
$$

The population-gradient identity (20) implies

$$
\left\| \theta ^ { ( t ) } - \bar { \theta } ^ { \star } \right\| ^ { 2 } = \frac { 1 } { 4 } \left\| \nabla L _ { { \mathcal { H } } } ( \theta ^ { ( t ) } ) \right\| ^ { 2 } .
$$

Moreover, applying the pairwise form of the variance decomposition to the client parameters yields

$$
\frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \big \| \theta _ { j } ^ { \star } - \bar { \theta } ^ { \star } \big \| ^ { 2 } = \frac { 1 } { 2 | \mathcal { H } | ^ { 2 } } \sum _ { i \in \mathcal { H } } \sum _ { j \in \mathcal { H } } \big \| \theta _ { i } ^ { \star } - \theta _ { j } ^ { \star } \big \| ^ { 2 } = \frac { 1 } { 2 } \Gamma _ { \mathsf { l i n } } .
$$

Hence, combining the last three bounds and averaging over the iterations yields

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \Big \Vert \theta ^ { ( t ) } - \theta _ { j } ^ { \star } \Big \Vert ^ { 2 } = \frac { 1 } { 4 } Q _ { T } + \frac { 1 } { 2 } \Gamma _ { \mathsf { l i n } } .
$$

Finally, substituting the bound on $Q _ { T }$ from Theorem B.1 and absorbing numerical constants proves the result. □

## C Nonlinear Regression

In this section, we extend our analysis to the nonlinear regression setting and derive nonasymptotic guarantees for adversarially robust federated learning. Our analysis follows the same underlying structure in linear regression, while accounting for the additional intricacies introduced by the nonlinear parameterization. In particular, we first characterize the gradient decomposition across honest clients in function space and derive a bound on the resulting gradient heterogeneity under standard regularity and concentration conditions. We then combine this bound with the optimization-error recursion to establish a non-asymptotic ergodic convergence bound and the corresponding function-recovery guarantee.

## C.1 Regression Model and Gradients

We recall that each honest client $i \in \mathcal { H }$ has a ground-truth parameter $\theta _ { i } ^ { \star } \in \mathbb { R } ^ { p }$ and generates data according to

$$
Y _ { i , k } = h _ { \theta _ { i } ^ { \star } } ( X _ { i , k } ) + V _ { i , k } , \forall k = 1 , \dots , \tau ,
$$

where $h _ { \theta } : \mathbb { R } ^ { d }  \mathbb { R } ^ { q }$ is a nonlinear function parameterized by $\theta \in \mathbb { R } ^ { p }$ (note that now θ is a vector that parameterizes the nonlinear function $h _ { \theta } ( \cdot ) ) , V _ { i , k } \in \mathbb { R } ^ { q }$ satisfies Assumption 3.1, and $\| X _ { i , k } \| \le R$ almost surely. At a fixed iteration, with the iteration index on the fresh samples suppressed, we recall that the empirical loss is

$$
\hat { L } _ { i } ^ { ( t ) } ( \theta ) : = \frac { 1 } { \tau } \sum _ { k = 1 } ^ { \tau } \| Y _ { i , k } - h _ { \theta } ( X _ { i , k } ) \| ^ { 2 } .
$$

We then define the residual $r _ { i , k } ( \theta ) : = h _ { \theta } ( X _ { i , k } ) - Y _ { i , k }$ , and write

$$
\nabla _ { \boldsymbol { \theta } } \hat { L } _ { i } ^ { ( t ) } ( \boldsymbol { \theta } ) = \frac { 2 } { \tau } \sum _ { k = 1 } ^ { \tau } r _ { i , k } ( \boldsymbol { \theta } ) J _ { \boldsymbol { \theta } } ( X _ { i , k } ) ^ { \top } .
$$

and by substituting the ground-truth model $Y _ { i , k } = h _ { \theta _ { i } ^ { \star } } ( X _ { i , k } ) + V _ { i , k }$ , we have

$$
\nabla _ { \theta } \hat { L } _ { i } ^ { ( t ) } ( \theta ^ { ( t ) } ) = \frac { 2 } { \tau } \sum _ { k = 1 } ^ { \tau } \bigl ( h _ { \theta ^ { ( t ) } } \bigl ( X _ { i , k } \bigr ) - h _ { \theta _ { i } ^ { \star } } \bigl ( X _ { i , k } \bigr ) \bigr ) J _ { \theta ^ { ( t ) } } \bigl ( X _ { i , k } \bigr ) ^ { \top } - \frac { 2 } { \tau } \sum _ { k = 1 } ^ { \tau } V _ { i , k } J _ { \theta ^ { ( t ) } } \bigl ( X _ { i , k } \bigr ) ^ { \top } .
$$

## C.2 Gradient Heterogeneity Decomposition

At iteration t, let

$$
\begin{array} { r l } & { \hat { s } _ { i } ^ { ( t ) } : = \displaystyle \frac { 1 } { \tau } \sum _ { k = 1 } ^ { \tau } \bigl ( h _ { \theta ^ { ( t ) } } ( X _ { i , k } ) - h _ { \theta _ { i } ^ { \star } } ( X _ { i , k } ) \bigr ) J _ { \theta ^ { ( t ) } } ( X _ { i , k } ) ^ { \top } , } \\ & { s _ { i } ^ { ( t ) } : = \mathbb { E } _ { X } \left[ \bigl ( h _ { \theta ^ { ( t ) } } ( X ) - h _ { \theta _ { i } ^ { \star } } ( X ) \bigr ) J _ { \theta ^ { ( t ) } } ( X ) ^ { \top } \right] . } \end{array}
$$

Here and below, the iteration index on the fresh samples is suppressed. We also define the client noise term and its honest average as

$$
\xi _ { i } ^ { ( t ) } : = \frac { 2 } { \tau } \sum _ { k = 1 } ^ { \tau } V _ { i , k } J _ { \theta ^ { ( t ) } } \big ( X _ { i , k } \big ) ^ { \top } \mathrm { ~ a n d ~ } \bar { \xi } ^ { ( t ) } : = \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \xi _ { i } ^ { ( t ) } .
$$

Then, we have

$$
\nabla \hat { L } _ { i } ^ { ( t ) } ( \theta ^ { ( t ) } ) - \nabla \hat { L } _ { \mathcal { H } } ^ { ( t ) } ( \theta ^ { ( t ) } ) = 2 \left( \hat { s } _ { i } ^ { ( t ) } - \frac { 1 } { \vert \mathcal { H } \vert } \sum _ { j \in \mathcal { H } } \hat { s } _ { j } ^ { ( t ) } \right) - \left( \xi _ { i } ^ { ( t ) } - \bar { \xi } ^ { ( t ) } \right) .\tag{28}
$$

## C.3 Concentration Inequalities for Nonlinear Regression

The fresh-sample assumption ensures that, conditionally on $\boldsymbol { \theta } ^ { ( t ) }$ , the samples used at iteration t are independent of the current iterate.

Lemma C.1. Suppose that Assumptions $\it 4 . 1$ and 4.2 hold. Then, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta _ { i }$ , simultaneously for all $i \in \mathcal { H }$ and $t = 0 , \ldots , T - 1$ , it holds that

$$
\left\| \hat { s } _ { i } ^ { ( t ) } - s _ { i } ^ { ( t ) } \right\| _ { F } \leq C _ { 1 } \bar { J } \sqrt { E _ { i } ^ { ( t ) } } \sqrt { \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } } .\tag{29}
$$

Proof. We begin by noting that conditionally on $\boldsymbol { \theta } ^ { ( t ) }$ , the matrices

$$
\begin{array} { r } { W _ { i , k } ^ { ( t ) } - \mathbb { E } W _ { i , k } ^ { ( t ) } \mathrm { ~ w i t h ~ } W _ { i , k } ^ { ( t ) } : = \left( h _ { \theta ^ { ( t ) } } ( X _ { i , k } ) - h _ { \theta _ { i } ^ { \star } } ( X _ { i , k } ) \right) J _ { \theta ^ { ( t ) } } ( X _ { i , k } ) ^ { \top } , } \end{array}
$$

are independent and sub-Gaussian from Assumption 4.2. Then, Assumptions 4.1 and 4.2 yield

$$
\left. W _ { i , k } ^ { ( t ) } - \mathbb { E } W _ { i , k } ^ { ( t ) } \right. _ { \psi _ { 2 } } \leq C _ { 1 } \bar { J } \sqrt { E _ { i } ^ { ( t ) } } .
$$

Therefore, the result follows from Lemma B.6 and a union bound over the honest clients and iterations. □

The same argument, together with Assumption 3.1, yields

$$
\begin{array} { r l } & { \left\| \xi _ { i } ^ { ( t ) } \right\| _ { F } \le C _ { 1 } \bar { J } \sigma \sqrt { \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } } , } \\ & { \left\| \bar { \xi } ^ { ( t ) } \right\| _ { F } \le C _ { 1 } \bar { J } \sigma \sqrt { \frac { p + \log ( 2 T / \delta ) } { \tau | \mathcal { H } | } } , } \end{array}\tag{30}
$$

simultaneously for all honest clients and iterations, with probability at least $1 - \delta$

## C.4 Per-iteration Gradient Heterogeneity Bound

We recall that

$$
\begin{array} { r l r } {  { \Gamma _ { \mathfrak { n o n l i n } } : = \frac { 1 } { | \mathcal { H } | ^ { 2 } } \sum _ { i \in \mathcal { H } } \sum _ { j \in \mathcal { H } } \mathbb { E } _ { X } [ \| h _ { \theta _ { i } ^ { \star } } ( X ) - h _ { \theta _ { j } ^ { \star } } ( X ) \| ^ { 2 } ] , } } \\ & { } & { \bar { E } ^ { ( t ) } : = \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \mathbb { E } _ { X } [ \| h _ { \theta ^ { ( t ) } } ( X ) - h _ { \theta _ { i } ^ { \star } } ( X ) \| ^ { 2 } ] , } \end{array}
$$

and, for some $\bar { \omega } > 0$ , let

$$
\bar { \omega } ^ { 2 } : = C _ { 1 } \bar { J } ^ { 2 } \left( \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) .\tag{31}
$$

Lemma C.2. Suppose that Assumptions $ 4 . 1 , \ 4 . 2 ,$ and 3.1 hold. Then, with probability at least $1 - \delta ,$ simultaneously for all $t = 0 , \ldots , T - 1$

$$
G ^ { ( t ) } \lesssim \bar { J } ^ { 2 } \Gamma _ { n o n l i n } + \bar { \omega } ^ { 2 } \bar { E } ^ { ( t ) } + \bar { J } ^ { 2 } \sigma ^ { 2 } \left( \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) .
$$

Proof. By adding and subtracting $s _ { i } ^ { ( t ) }$ and $\begin{array} { r } { \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } s _ { j } ^ { ( t ) } } \end{array}$ yields

$$
\hat { s } _ { i } ^ { ( t ) } - \frac { 1 } { \vert \mathcal { H } \vert } \sum _ { j \in \mathcal { H } } \hat { s } _ { j } ^ { ( t ) } = s _ { i } ^ { ( t ) } - \frac { 1 } { \vert \mathcal { H } \vert } \sum _ { j \in \mathcal { H } } s _ { j } ^ { ( t ) } + \hat { s } _ { i } ^ { ( t ) } - s _ { i } ^ { ( t ) } - \frac { 1 } { \vert \mathcal { H } \vert } \sum _ { j \in \mathcal { H } } \left( \hat { s } _ { j } ^ { ( t ) } - s _ { j } ^ { ( t ) } \right) .
$$

The first term contains only the model heterogeneity since the terms involving $h _ { \theta ^ { ( t ) } } ( X )$ cancel. By Jensen’s inequality and Assumption 4.1,

$$
\frac { 1 } { \left| \mathcal { H } \right| } \sum _ { i \in \mathcal { H } } \left\| s _ { i } ^ { ( t ) } - \frac { 1 } { \left| \mathcal { H } \right| } \sum _ { j \in \mathcal { H } } s _ { j } ^ { ( t ) } \right\| _ { F } ^ { 2 } \leq \bar { J } ^ { 2 } \Gamma _ { \mathsf { n o n l i n } } .
$$

Lemma C.1, Jensen’s inequality, (29), and (30) bound the remaining terms. We then substitute these bounds into (28), squaring, and averaging over the honest clients proves the result.

Let

$$
\begin{array} { r } { Z ^ { ( t ) } : = \left. \nabla \hat { L } _ { \mathcal { H } } ^ { ( t ) } ( { \boldsymbol { \theta } } ^ { ( t ) } ) - \nabla L _ { \mathcal { H } } ( { \boldsymbol { \theta } } ^ { ( t ) } ) \right. ^ { 2 } . } \end{array}\tag{32}
$$

We now derive the bound on $Z ^ { ( t ) }$ explicitly. By the definitions of $\hat { s } _ { i } ^ { ( t ) } , s _ { i } ^ { ( t ) }$ , and $\bar { \xi } ^ { ( t ) }$ , we have

$$
\nabla \hat { L } _ { \mathcal { H } } ^ { ( t ) } ( \boldsymbol { \theta } ^ { ( t ) } ) - \nabla L _ { \mathcal { H } } ( \boldsymbol { \theta } ^ { ( t ) } ) = \frac { 2 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \left( \hat { s } _ { i } ^ { ( t ) } - s _ { i } ^ { ( t ) } \right) - \bar { \xi } ^ { ( t ) } .
$$

It follows from $\left\| a + b \right\| ^ { 2 } \leq 2 \left\| a \right\| ^ { 2 } + 2 \left\| b \right\| ^ { 2 }$ that

$$
Z ^ { ( t ) } \leq 8 \left\| { \frac { 1 } { | { \mathcal { H } } | } } \sum _ { i \in { \mathcal { H } } } \left( { \hat { s } } _ { i } ^ { ( t ) } - s _ { i } ^ { ( t ) } \right) \right\| ^ { 2 } + 2 \left\| { \bar { \xi } } ^ { ( t ) } \right\| ^ { 2 } .
$$

We note that conditionally on $\theta ^ { ( t ) }$ , the matrices $\hat { s } _ { i } ^ { ( t ) } - s _ { i } ^ { ( t ) }$ are independent and mean-zero across honest clients. In addition, for every $U \in \mathbb { S } _ { F } ^ { p \times 1 }$ , conditional sub-Gaussian Hoefding [Ver19, Theorem 2.7.3] implies

$$
\mathbb { P } \left( \left| \left. U , \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } ( \hat { s } _ { i } ^ { ( t ) } - s _ { i } ^ { ( t ) } ) \right. _ { F } \right| \geq u \middle | \theta ^ { ( t ) } \right) \leq 2 \exp \left( - \frac { c \tau | \mathcal { H } | ^ { 2 } u ^ { 2 } } { \bar { J } ^ { 2 } \sum _ { i \in \mathcal { H } } E _ { i } ^ { ( t ) } } \right) .
$$

Therefore, by taking a union bound over a fixed net of $\mathbb { S } _ { F } ^ { p \times 1 }$ and over the iterations yields the following expression

$$
\left. \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } \left( \hat { s } _ { i } ^ { ( t ) } - s _ { i } ^ { ( t ) } \right) \right. ^ { 2 } \lesssim \frac { \bar { \omega } ^ { 2 } } { | \mathcal { H } | } \left( \frac { 1 } { | \mathcal { H } | } \sum _ { i \in \mathcal { H } } E _ { i } ^ { ( t ) } \right) = \frac { \bar { \omega } ^ { 2 } } { | \mathcal { H } | } \bar { E } ^ { ( t ) } .
$$

Moreover, the second inequality in (30) provides the bound for the second term in the bound of $Z ^ { ( t ) }$ , i.e.,

$$
\left\| \bar { \xi } ^ { ( t ) } \right\| ^ { 2 } \lesssim \bar { J } ^ { 2 } \sigma ^ { 2 } \left( \frac { p + \log ( 2 T / \delta ) } { \tau | \mathcal { H } | } \right) .
$$

Therefore, combining these bounds proves

$$
Z ^ { ( t ) } \lesssim \frac { \bar { \omega } ^ { 2 } } { | \mathcal { H } | } \bar { E } ^ { ( t ) } + \bar { J } ^ { 2 } \sigma ^ { 2 } \left( \frac { p + \log ( 2 T / \delta ) } { \tau | \mathcal { H } | } \right) ,\tag{33}
$$

simultaneously over the iterations, with probability at least $1 - \delta$

## C.5 Bound on $G _ { T }$ and Convergence Analysis

Let $\theta ^ { \star }$ be a minimizer of the honest-average population loss $L _ { { \mathcal { H } } } ( \theta )$ . We note that conditional mean-zero noise implies

$$
\bar { E } ^ { ( t ) } = L \varkappa ( \theta ^ { ( t ) } ) - L \varkappa ( \theta ^ { \star } ) + \bar { E } ^ { \star } \mathrm { ~ w h e r e ~ w e ~ } \bar { E } ^ { \star } : = \frac { 1 } { \vert \mathcal { H } \vert } \sum _ { i \in \mathcal { H } } \mathbb { E } _ { X } \left[ \left. h _ { \theta ^ { \star } } ( X ) - h _ { \theta _ { i } ^ { \star } } ( X ) \right. ^ { 2 } \right]\tag{34}
$$

As $\theta ^ { \star }$ minimizes $L _ { { \mathcal { H } } } ( \theta )$ , by evaluating the loss at each client optimum and averaging provides $\bar { E } ^ { \star } \leq \Gamma _ { \mathsf { n o n l i n } }$ . Assumption 4.4 therefore yields

$$
\bar { E } ^ { ( t ) } \leq \Gamma _ { \mathrm { n o n l i n } } + \frac { 1 } { 2 \mu ^ { \prime } } \left\| \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \right\| ^ { 2 } .\tag{35}
$$

Theorem C.1. Suppose that Assumptions 3.1, 4.1, 4.2, 4.3, and $4 . 4$ hold. Let $\eta \leq 1 / L ^ { \prime }$ and suppose that the number of fresh samples per client satisfies

$$
\tau \geq \frac { 4 C _ { 0 } C _ { 1 } \bar { J } ^ { 2 } } { \mu ^ { \prime } } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \left( p + \log \left( \frac { 2 | \mathcal { H } | T } { \delta } \right) \right) .\tag{36}
$$

Then, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$

$$
\begin{array} { l } { \displaystyle Q _ { T } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } { \left\| \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \right\| ^ { 2 } } \lesssim \frac { \Delta L ^ { ( 0 ) } } { \eta T } + \left[ \kappa \bar { J } ^ { 2 } + \bar { \omega } ^ { 2 } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \right] \Gamma _ { n o n l i n } } \\ { \displaystyle ~ + ~ \bar { J } ^ { 2 } \sigma ^ { 2 } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \left( \frac { p + \log \left( 2 | \mathcal { H } | T / \delta \right) } { \tau } \right) , } \end{array}
$$

where $\Delta L ^ { ( 0 ) } : = L _ { \mathcal { H } } ( \theta ^ { ( 0 ) } ) - L _ { \mathcal { H } } ( \theta ^ { \star } )$

Proof. Let $e ^ { ( t ) } : = \mathsf { F } ^ { ( t ) } - \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } )$ . We note that the robustness of the aggregation rule and (32) imply

$$
\left\| e ^ { ( t ) } \right\| ^ { 2 } \leq 2 \kappa G ^ { ( t ) } + 2 Z ^ { ( t ) } .\tag{37}
$$

By L<sup>′</sup>-smoothness and $\eta \leq 1 / L ^ { \prime }$

$$
L _ { { \mathcal { H } } } ( \boldsymbol { \theta } ^ { ( t + 1 ) } ) - L _ { { \mathcal { H } } } ( \boldsymbol { \theta } ^ { ( t ) } ) \leq - \frac { \eta } { 2 } \left\| \nabla L _ { { \mathcal { H } } } ( \boldsymbol { \theta } ^ { ( t ) } ) \right\| ^ { 2 } + \frac { \eta } { 2 } \left\| \boldsymbol { e } ^ { ( t ) } \right\| ^ { 2 } .
$$

Then, we sum the above inequality over the iterations and use the optimality of $\theta ^ { \star }$ to obtain

$$
Q _ { T } \leq \frac { 2 \Delta L ^ { ( 0 ) } } { \eta T } + \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \Big \Vert e ^ { ( t ) } \Big \Vert ^ { 2 } \leq \frac { 2 \Delta L ^ { ( 0 ) } } { \eta T } + 2 \kappa G _ { T } + 2 Z _ { T } ,
$$

where we have $\begin{array} { r } { Z _ { T } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } Z ^ { ( t ) } } \end{array}$ . By averaging Lemma C.2 and (33), we obtain

$$
\begin{array} { r l } & { G _ { T } \lesssim \bar { J } ^ { 2 } \Gamma _ { \mathsf { n o n l i n } } + \bar { \omega } ^ { 2 } \bar { E } _ { T } + \bar { J } ^ { 2 } \sigma ^ { 2 } \left( \frac { p + \log \left( 2 | \mathcal { H } | T / \delta \right) } { \tau } \right) , } \\ & { Z _ { T } \lesssim \frac { \bar { \omega } ^ { 2 } } { | \mathcal { H } | } \bar { E } _ { T } + \bar { J } ^ { 2 } \sigma ^ { 2 } \left( \frac { p + \log \left( 2 | \mathcal { H } | T / \delta \right) } { \tau | \mathcal { H } | } \right) , } \end{array}
$$

where $\begin{array} { r } { \bar { E } _ { T } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \bar { E } ^ { ( t ) } } \end{array}$ . In addition, we (35) over iterations to obtain

$$
\bar { E } _ { T } \leq \Gamma _ { \mathsf { n o n l i n } } + \frac { Q _ { T } } { 2 \mu ^ { \prime } } .
$$

Then, by combining the preceding bounds, we have

$$
\begin{array} { r l r } {  { Q _ { T } \lesssim \frac { \Delta L ^ { ( 0 ) } } { \eta T } + [ \kappa \bar { J } ^ { 2 } + \bar { \omega } ^ { 2 } ( \kappa + \frac { 1 } { | \mathcal { H } | } ) ] \Gamma _ { \mathfrak { n o n } \mid \mathfrak { i } } } } \\ & { } & { + ~ \bar { J } ^ { 2 } \sigma ^ { 2 } ( \kappa + \frac { 1 } { | \mathcal { H } | } ) ( \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } ) + \frac { \bar { \omega } ^ { 2 } } { \mu ^ { \prime } } ( \kappa + \frac { 1 } { | \mathcal { H } | } ) Q _ { T } . ~ } \end{array}
$$

Finally, substituting (31) into (36) yields

$$
\frac { { \bar { \omega } } ^ { 2 } } { \mu ^ { \prime } } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \leq \frac { 1 } { C _ { 0 } } ,
$$

As $C _ { 0 }$ is chosen suficiently large relative to the universal constants hidden above, this allows us to move the final term to the left-hand side and absorb the resulting numerical factor into the constant absorbed in $\lesssim$ □

We now combine Lemma C.2, (35), and Theorem C.1 to write the following bound.

Lemma C.3. Suppose that the conditions of Theorem C.1 hold. Then, with probability at least $1 - \delta$

$$
G _ { T } \lesssim ( \bar { J } ^ { 2 } + \bar { \omega } ^ { 2 } ) \Gamma _ { n o n l i n } + \frac { \bar { \omega } ^ { 2 } \Delta { \cal L } ^ { ( 0 ) } } { \mu ^ { \prime } \eta T } + \bar { J } ^ { 2 } \sigma ^ { 2 } \left( \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) .
$$

Proof. We begin by averaging the per-iteration bound in Lemma C.2 gives

$$
G _ { T } \lesssim \bar { J } ^ { 2 } \Gamma _ { \mathsf { n o n l i n } } + \bar { \omega } ^ { 2 } \bar { E } _ { T } + \bar { J } ^ { 2 } \sigma ^ { 2 } \left( \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) .
$$

By the averaged form of (35), we then have

$$
\bar { E } _ { T } \leq \Gamma _ { \mathsf { n o n l i n } } + \frac { Q _ { T } } { 2 \mu ^ { \prime } } .
$$

Therefore, by substituting this inequality and then applying Theorem C.1, we obtain

$$
\begin{array} { l } { { \displaystyle G _ { T } \lesssim ( \bar { J } ^ { 2 } + \bar { \omega } ^ { 2 } ) \Gamma _ { \mathrm { n o n l i n } } + \frac { \bar { \omega } ^ { 2 } } { \mu ^ { \prime } } \left( \frac { \Delta L ^ { ( 0 ) } } { \eta T } + \left[ \kappa \bar { J } ^ { 2 } + \bar { \omega } ^ { 2 } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \right] \Gamma _ { \mathrm { n o n l i n } } \right) } } \\ { { \displaystyle ~ + \bar { J } ^ { 2 } \sigma ^ { 2 } \left( 1 + \frac { \bar { \omega } ^ { 2 } } { \mu ^ { \prime } } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \right) \left( \frac { p + \log \left( 2 | \mathcal { H } | T / \delta \right) } { \tau } \right) . } } \end{array}
$$

The sample-size condition (36) bounds the factors involving $\varpi ^ { 2 }$ , and absorbing universal numerical constants yields the stated result. □

## C.6 Non-Asymptotic Function Recovery Error Bound

We conclude the nonlinear analysis by restating the function-recovery guarantee and providing its complete derivation.

Corollary C.1 (Parameter Recovery Bound). Suppose that the conditions of Theorem C.1 hold. Then, for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ , it holds that

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \displaystyle \frac { 1 } { | \mathcal { H } | } \sum _ { j \in \mathcal { H } } \mathbb { E } _ { X } \left[ \left\| h _ { \theta ^ { ( t ) } } ( X ) - h _ { \theta _ { j } ^ { * } } ( X ) \right\| ^ { 2 } \right] } \\ & { \displaystyle \lesssim \frac { \Delta L ^ { ( 0 ) } } { \mu ^ { \prime } \eta T } + \left[ 1 + \frac { \kappa \bar { J } ^ { 2 } } { \mu ^ { \prime } } + \frac { \bar { \omega } ^ { 2 } } { \mu ^ { \prime } } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \right] \Gamma _ { n o n i n } + \frac { \bar { J } ^ { 2 } \sigma ^ { 2 } } { \mu ^ { \prime } } \left( \kappa + \frac { 1 } { | \mathcal { H } | } \right) \left( \frac { p + \log ( 2 | \mathcal { H } | T / \delta ) } { \tau } \right) . } \end{array}
$$

Proof. Recall from (34) that

$$
\bar { E } ^ { ( t ) } = L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) - L _ { \mathcal { H } } ( \theta ^ { \star } ) + \bar { E } ^ { \star } .
$$

To control $\bar { E } ^ { \star }$ , evaluate the average functional error at each client optimum and then average over these choices. As $\theta ^ { \star }$ minimizes the honest-average population loss, we have

$$
\begin{array} { r l } & { \bar { E } ^ { \star } \le \displaystyle \frac { 1 } { | \mathcal { H } | } \displaystyle \sum _ { i \in \mathcal { H } } \left( \frac { 1 } { | \mathcal { H } | } \displaystyle \sum _ { j \in \mathcal { H } } \mathbb { E } _ { X } \left[ \left\| h _ { \theta _ { i } ^ { \star } } ( X ) - h _ { \theta _ { j } ^ { \star } } ( X ) \right\| ^ { 2 } \right] \right) } \\ & { \displaystyle = \frac { 1 } { | \mathcal { H } | ^ { 2 } } \displaystyle \sum _ { i \in \mathcal { H } } \displaystyle \sum _ { j \in \mathcal { H } } \mathbb { E } _ { X } \left[ \left\| h _ { \theta _ { i } ^ { \star } } ( X ) - h _ { \theta _ { j } ^ { \star } } ( X ) \right\| ^ { 2 } \right] : = \Gamma _ { \mathsf { n o n l i n } } . } \end{array}
$$

We note that the PL condition (Assumption 4.4) implies

$$
L _ { { \mathcal { H } } } ( \theta ^ { ( t ) } ) - L _ { { \mathcal { H } } } ( \theta ^ { \star } ) \leq \frac { 1 } { 2 \mu ^ { \prime } } \left\| \nabla L _ { { \mathcal { H } } } ( \theta ^ { ( t ) } ) \right\| ^ { 2 } .
$$

Therefore, by combining the previous bounds, we obtain

$$
\bar { E } ^ { ( t ) } \leq \Gamma _ { \mathrm { n o n l i n } } + \frac { 1 } { 2 \mu ^ { \prime } } \left\| \nabla L _ { \mathcal { H } } ( \theta ^ { ( t ) } ) \right\| ^ { 2 } .
$$

Finally, averaging over $t = 0 , \ldots , T - 1$ yields

$$
{ \frac { 1 } { T } } \sum _ { t = 0 } ^ { T - 1 } { \bar { E } } ^ { ( t ) } \leq \Gamma _ { \mathsf { n o n l i n } } + { \frac { Q _ { T } } { 2 \mu ^ { \prime } } } .
$$

Hence, substituting the bound on $Q _ { T }$ from Theorem C.1 and absorbing numerical constants proves the result. □