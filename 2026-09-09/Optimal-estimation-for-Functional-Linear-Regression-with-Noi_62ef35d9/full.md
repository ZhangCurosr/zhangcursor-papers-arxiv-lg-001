# Optimal estimation for Functional Linear Regression with Noisy Discretized Data

Sixtine Sphabmixay

Université Paris Cité, CNRS, MAP5, F-75006 Paris, France

September 9, 2026

## Abstract

In this paper, we consider the scalar-on-function linear regression model under a realistic sampling scheme in which the functional covariates are observed on a regular grid and contaminated by additive noise. We propose a two-step estimation procedure: first, the underlying curves are reconstructed from the discrete noisy observations using a Fourier-based projection method; second, the slope function is estimated by a penalized least-squares criterion over finite-dimensional trigonometric spaces, with data-driven selection of the model dimension. We establish oracle-type inequalities for the prediction error, both with respect to the reconstructed curves and to the true latent curves. Under regularity assumptions on the slope function and polynomial decay of the eigenvalues of the covariate, we derive convergence rates for the prediction error and show that our estimator attains the minimax rate when the number of grid points is suficiently large. Finally, the proposed method is illustrated on simulated data and on a real meteorological dataset.

## 1 Introduction

Functional data analysis (FDA), as formalized and popularized by Ramsay and Silverman [14], is a statistical framework designed to analyze data that are naturally expressed as functions rather than finite-dimensional vectors, which are commonly considered in classical statistical methods. This methodology is particularly well suited to modern datasets, where observations evolve continuously over domains such as time or space. Moreover, advances in data collection and storage have also led to a substantial increase in the availability of functional data. Consequently, FDA has been widely applied across various disciplines, including medicine (Frøslie et al. [10]), biology (Dieng et al. [9]), and finance (Kokoszka et al. [13]).

A central inferential tool in FDA is the functional linear regression model, which extends classical regression by relating scalar or functional responses to functional predictors. In this paper, we focus on the scalar-on-function case. More precisely, we assume that a functional covariate X taking values in a Hilbert space X and a real-valued response variable $Y$ in R are linked through the model,

$$
Y = \mu _ { Y } + \int _ { 0 } ^ { 1 } \beta ( t ) ( X ( t ) - \mu ( t ) ) d t + \epsilon ,\tag{1}
$$

where $\mu _ { Y } = \operatorname { \mathbb { E } } ( Y ) , \mu ( t ) = \operatorname { \mathbb { E } } ( X ( t ) )$ for all $t \in [ 0 , 1 ]$ and ϵ is a random variable which is centered and of finite variance $\sigma ^ { 2 }$

This model was popularized in the FDA literature by Ramsay and Silverman [14] and has since been extensively developed, with numerous approaches proposed for estimating the coeficient function $\beta .$ For instance, Cardot et al. [5] introduced an estimator based on functional principal component analysis. James et al. [12] proposed the FLiRTI method, designed to produce interpretable estimates of $\beta$ that highlight regions where the relationship between $Y$ and X is negligible. More recently, Grollemund et al. [11] developed a Bayesian approach, known as Bliss, which leverages the conditional distribution $\mathrm { ~ \bar { ~ } Y ~ } | \ X , \beta \sim \mathcal N ( \langle \beta , \dot { X } \rangle _ { \mathscr X } , \sigma ^ { 2 } )$ . This approach also aims to identify the time intervals that have the greatest influence on the response.

Studies of the statistical properties of existing estimators have also been developed under the assumption that the functional data are observed continuously and without noise, as in Cardot et al. [6] who derived minimax rates and oracle-type inequalities. However, this idealized setting is rarely encountered in practice, since observations are typically available only on a finite grid, which may be regularly or irregularly spaced, deterministic or random. This more realistic setting has received comparatively less attention. To our knowledge, Cardot et al. [7] were among the first to study it. Although their approach is more practically relevant, it requires the grid size to be suficiently large for their results to hold.

In this paper, we aim to to extend the approach of Brunel and Roche [4] to the more realistic setting where observations are available on a regular grid and are corrupted by noise. As them, we assume that the covariates are centered, periodic, and second-order stationary, with $X ( 0 ) = X ( 1 )$ The sample $( X _ { i } ) _ { i = 1 , \ldots , n }$ consists of independent copies of $X$ . Here we further assume that the functional covariates are observed on a common regular grid $t _ { h } = h / p$ for all $h \in \{ 0 , \ldots , p - 1 \}$ , and that the observations are contaminated by noise. Thus, rather than observing the full curves $X _ { i } ,$ we observe

$$
Z _ { i } ( t _ { h } ) = X _ { i } ( t _ { h } ) + \eta _ { i , h } ,\tag{2}
$$

where the noise variables $( \eta _ { i , h } )$ are centered, i.i.d., with variance $\tau ^ { 2 }$ , and independent of the processes $X _ { i }$ . For technical simplicity, we assume throughout this paper that $\sigma ^ { 2 }$ , the variance of ϵ defined in (1) is known.

In the idealized setting of continuously observed noiseless curves, Brunel and Roche [4] proposed an estimator of the slope function $\beta$ based on a least-squares contrast over finite-dimensional spaces generated by trigonometric bases, together with a penalization procedure for selecting the optimal dimension of the basis. They proved an oracle inequality and obtained convergence rates when the eigenvalues of the covariance operator associated to the true curves decay in either a polynomial or exponential way, and that $\beta$ belongs to a periodic Sobolev space,

$$
W ^ { \mathrm { p e r } } ( k , L ) : = \{ f \in W _ { 2 } ^ { k } ( L ) , \forall j = 0 , . . . , k - 1 , f ^ { ( j ) } ( 0 ) = f ^ { ( j ) } ( 1 ) \} ,\tag{3}
$$

with

$W _ { 2 } ^ { k } ( L ) : = \{ f : [ 0 , 1 ] \to \mathbb { R } , f ^ { ( k - 1 ) }$ is absolutely continuous and $\| f ^ { ( k ) } \| _ { L ^ { 2 } } \leq L \}$

for k a positive integer and $L$ a positive real number. Here $\| \cdot \| _ { L ^ { 2 } }$ is defined as

$$
\| f \| _ { L ^ { 2 } } = \left( \int _ { 0 } ^ { 1 } f ( t ) ^ { 2 } d t \right) ^ { 1 / 2 }
$$

for any square-integrable function f on [0, 1].

Our approach follows a similar philosophy. We estimate $\beta$ by minimizing a least-squares criterion after a preliminary step that reconstructs the underlying curve X from the noisy discrete observations. This reconstruction is based on a Riemann-sum approximation and a projection onto a finite-dimensional space spanned by a truncated Fourier basis. We then introduce a penalization procedure to automatically select the optimal number of Fourier terms used in the estimation of $\beta .$ This strategy difers from the work of Cardot et al. [7] where they proposed a smoothing splines estimator to estimate the unknown slope function. We derive two oracle-type inequalities for the prediction error, one with respect to the reconstructed curves and one with respect to the true curves. We then establish convergence rates for the prediction error with respect to the true curves when $\beta$ belongs to $W ^ { \mathrm { p e r } } ( k , L )$ and the eigenvalues of the covariance operator decay polynomially. We recover the same rate as Brunel and Roche [4] when p is large enough but not necessarily infinite, namely that the prediction error decays as $\stackrel { - } { n } ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } }$ , where a denotes the decay parameter of the covariance eigenvalues. In this regime, we also establish a matching lower bound, showing that observing the trajectories only on a suficiently fine grid incurs no minimax loss compared to the continuous setting. When $p$ is moderately large, the number of observations on the grid also plays a role, and the prediction error decays as $n ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } } + p ^ { - \frac { 2 a - 1 } { 2 a } }$ . When $p$ is small the error decays as $p ^ { - { \frac { 2 a - 1 } { 2 a } } }$ . Finally, we apply our method on both simulated data and real-world meteorological observations.

Outline of the paper. Section 2 introduces the estimation procedure for $\beta$ and the risk measure considered in this work. Section 3 establishes oracle-type inequalities. Section 4 derives the convergence rates over periodic Sobolev spaces. Section 5 presents the lower bound in the case where $p$ is large. Section $6$ reports numerical simulation results on artificial datasets and Section $7$ on a true meteorological dataset.

Notation: We denote by $L ^ { 2 }$ the space of square-integrable functions on the interval [0, 1] equipped with the inner product $\begin{array} { r } { \langle f , g \rangle _ { L ^ { 2 } } = \int _ { 0 } ^ { 1 } f ( t ) g ( t ) d t } \end{array}$ and the associated norm $\| f \| _ { L ^ { 2 } } = \langle f , f \rangle _ { L ^ { 2 } } ^ { 1 / 2 }$ . For any set ${ \mathcal { A } } ,$ we denote by $\mathbb { 1 } _ { A }$ the indicator function and for all non-negative integers i and $j$ we define $\delta _ { i j }$ as the Kronecker delta. For any square matrix A we define the operator norm by $\| A \| _ { o p } = \operatorname* { s u p } _ { \| a \| _ { 2 } = 1 } \| A a \| _ { 2 }$ , where for any vector $u , \| u \| _ { 2 }$ denotes the Euclidean norm. In particular, we consider the space $l ^ { 2 }$ consisting of all sequences $c = ( c _ { l } ) _ { l \ge 1 }$ , such that $\| c \| _ { 2 } ^ { 2 } < \infty .$ . We also consider $\langle \cdot , \cdot \rangle _ { 2 }$ the standard Euclidean inner product. Moreover for any $a , b \in \mathbb { Z }$ and m $\in \mathbb { N } \backslash \{ 0 \}$ , we write $a \equiv b [ m ]$ to indicate that a and b are congruent modulo $m .$

## 2 Estimation procedure

## 2.1 Curve reconstruction

We consider an i.i.d. sample $( Y _ { i } , X _ { i } ) _ { i = 1 , \dots , n }$ drawn from the distribution of $( Y , X )$ . Thus, for each $i = 1 , \ldots , n$ , the pair $( Y _ { i } , X _ { i } )$ satisfies

$$
Y _ { i } = \int _ { 0 } ^ { 1 } \beta ( t ) X _ { i } ( t ) \mathrm { d } t + \epsilon _ { i } ,\tag{4}
$$

where the errors $( \epsilon _ { i } ) _ { i = 1 , \dots , n }$ are assumed to be i.i.d., centered, with finite variance $\sigma ^ { 2 }$ , and independent of the covariates $( X _ { i } ) _ { i = 1 , \ldots , n }$

The first step of our methodology is to reconstruct the true unobserved curves $X _ { i }$ , from their noisy and discrete observations, $Z _ { i }$ . To achieve this, we approximate each curve $X _ { i }$ by a finite series expansion in a suitable basis. Given our assumptions that the true curves are periodic, centered, second-order stationary and in $L ^ { 2 } ( [ 0 , 1 ] )$ , the Fourier basis provides a natural and eficient choice for this decomposition. This choice is further justified by the discussion in the next section (see Section 2.3). Let us first define $N _ { n , p } \in \mathbb { N } \backslash \{ 0 \}$ and $\mathcal { M } _ { n , p } = \{ 1 , . . . , N _ { n , p } \}$ . We denote

$$
S _ { m } : = \mathrm { S p a n } \{ \phi _ { 1 } , \phi _ { 2 } , \dots , \phi _ { 2 m + 1 } \} ,\tag{5}
$$

for all $m \in \mathcal { M } _ { n , p }$ with

$$
\phi _ { 1 } = 1 , \quad \phi _ { 2 j } ( \cdot ) = \sqrt { 2 } \cos \bigl ( 2 \pi j \cdot \bigr ) , \quad \phi _ { 2 j + 1 } ( \cdot ) = \sqrt { 2 } \sin \bigl ( 2 \pi j \cdot \bigr ) .
$$

We also define $D _ { m }$ the dimension of $S _ { m }$ (in particular, $D _ { m } = 2 m + 1 )$ . As we only consider $D _ { m }$ elements, $S _ { m }$ represents a finite dimension space spanned by a truncated Fourier basis.

Since the $Z _ { i } \mathrm { ^ { * } s }$ are only known at discrete points $t _ { h }$ , it’s impossible to compute $\langle X _ { i } , \phi _ { j } \rangle _ { L ^ { 2 } }$ directly. Hence we define

$$
\widetilde { x } _ { i , j } = \frac { 1 } { p } \sum _ { h = 0 } ^ { p - 1 } Z _ { i } ( t _ { h } ) \phi _ { j } ( t _ { h } ) .\tag{6}
$$

This quantity is a discrete approximation of the inner product $\langle X _ { i } , \phi _ { j } \rangle _ { L ^ { 2 } }$ , based on noisy observations on a regular grid, and can be interpreted as a Riemann sum in the ideal continuous setting. This particular scheme and choice of grid create a convenient discrete orthogonality property for the chosen Fourier basis under the condition that ${ \cal D } _ { N _ { n , p } } ~ < ~ p .$ This significantly simplifies the mathematical proofs required for the main results which are detailed in the appendix (Lemma B.1 and Lemma C.1). Then, we can define the reconstructed curves as follows :

$$
\widetilde X _ { i } ( t ) : = \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } \widetilde x _ { i , j } \phi _ { j } ( t ) , \qquad \mathrm { f o r ~ a l l ~ } i = 1 , \dots , n \quad \mathrm { a n d ~ f o r ~ a l l ~ } t \in [ 0 , 1 ] .\tag{7}
$$

Thus, $\widetilde { X } _ { i }$ is a random element of the finite-dimensional space $S _ { N _ { n , p } ; \ l }$ entirely determined by the discrete noisy sample $\left( Z _ { i } ( t _ { 0 } ) , \ldots , Z _ { i } ( t _ { p - 1 } ) \right)$

We also denote by $\widetilde { X }$ a generic random element having the same distribution as any $\widetilde { X } _ { i }$ . Equivalently, if $( X , \eta _ { 0 } , \ldots , \eta _ { p - 1 } )$ is a generic copy of $\left( X _ { i } , \eta _ { i , 0 } , \ldots , \eta _ { i , p - 1 } \right)$ , with $Z ( t _ { h } ) = X ( t _ { h } ) + \eta _ { h }$ 2 then

$$
\widetilde X ( t ) = \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } \widetilde x _ { j } \phi _ { j } ( t ) , \qquad t \in [ 0 , 1 ] ,
$$

with

$$
\widetilde x _ { j } = \frac { 1 } { p } \sum _ { h = 0 } ^ { p - 1 } Z ( t _ { h } ) \phi _ { j } ( t _ { h } ) .
$$

In particular, $\widetilde { X } _ { 1 } , \ldots , \widetilde { X } _ { n }$ are i.i.d. copies of $\widetilde { X }$

## 2.2 Estimating the slope function

In this section, we build an estimator of the true slope function $\beta .$ . Let $m \in \mathcal { M } _ { n , p }$ . First of all, we define the following least-square criterion :

$$
\gamma _ { n } ( f ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( Y _ { i } - \langle f , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } ) ^ { 2 } , \qquad f \in S _ { m } .\tag{8}
$$

Because the trajectories $X _ { i }$ are unobserved, the ideal least-squares contrast (defined using $X _ { i }$ rather than $\widetilde { X } _ { i }$ in (8)) cannot be directly minimized. We therefore use the least-square criterion $\gamma _ { n }$ based on the reconstructed curves $( \widetilde { X } _ { i } ) ^ { \mathfrak { s } }$ which are the only available curves. This induces a diference with the ideal model, which is quantified by the reconstruction error terms appearing in the oracle inequalities introduced in Section 3.

We wish to minimize $\gamma _ { n }$ to get an estimate of $\beta .$ More particularly, we define ${ \widehat { \beta } } _ { m }$ as

$$
\widehat { \beta } _ { m } \in \mathop { \mathrm { a r g m i n } } _ { f \in S _ { m } } \gamma _ { n } ( f ) .\tag{9}
$$

Any $f \in S _ { m }$ can be rewritten as $\begin{array} { r } { f = \sum _ { j = 1 } ^ { D _ { m } } v _ { j } \phi _ { j } } \end{array}$ where $v = ( v _ { 1 } , v _ { 2 } , \ldots , v _ { D _ { m } } )$ is in $\mathbb { R } ^ { D _ { m } }$ . Therefore, minimizing $\gamma _ { n }$ in $S _ { m }$ is equivalent to minimizing

$$
F ( v ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( Y _ { i } - \sum _ { j = 1 } ^ { D _ { m } } v _ { j } \langle \phi _ { j } , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } \right) ^ { 2 } \qquad \mathrm { i n ~ } \mathbb { R } ^ { D _ { m } } .\tag{10}
$$

Considering

$$
\Phi _ { m } : = \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \widetilde { x } _ { i , k } \widetilde { x } _ { i , j } \right) _ { 1 \leq j , k \leq D _ { m } } ,\tag{11}
$$

and

$$
b : = \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Y _ { i } \widetilde { x } _ { i , k } \right) _ { 1 \leq k \leq D _ { m } } ,\tag{12}
$$

we have the following result,

$$
\nabla F ( \boldsymbol { v } ) = - 2 b + 2 \Phi _ { m } \boldsymbol { v } ,\tag{13}
$$

where $\nabla F$ denotes the gradient of $F$ . The computations that lead to this result are detailed in Lemma A.1. If $\Phi _ { m }$ is invertible we find that the minimum of $F , { \widetilde { \alpha } } ,$ , is such that $\widetilde { \alpha } = ( \Phi _ { m } ) ^ { - 1 } b$ and $\begin{array} { r } { \widehat { \beta } _ { m } = \sum _ { j = 1 } ^ { D _ { m } } \widetilde { \alpha } _ { j } \phi _ { j } } \end{array}$

However, $\Phi _ { m }$ is built from only n observations projected onto a $D _ { m }$ -basis. If the model dimension exceeds the sample size $( D _ { m } > n )$ or if some projected coordinates are linearly dependent in the data, $\Phi _ { m }$ loses rank, giving at least one zero eigenvalue so its determinant vanishes and it cannot

be inverted. Even when $D _ { m } \leq n$ , any direction that shows nearly zero empirical variance still makes the matrix singular or ill-conditioned.

To deal with this issue, let us define the set $G _ { m } : = \{ \widehat { \lambda } _ { m } ^ { ( p ) } \geq s _ { n } \}$ where $\widehat { \lambda } _ { m } ^ { \left( p \right) }$ is the smallest eigenvalue of $\Phi _ { m }$ and

$$
s _ { n } = { \frac { 2 } { n ^ { \alpha } } } \left( 1 - { \frac { 1 } { \sqrt { \ln n } } } \right) \qquad \alpha > 0 .\tag{14}
$$

This form of $s _ { n }$ is inspired by the one used in Brunel and Roche [4] introduced in Section 3.2. This condition guarantees positive definiteness and hence invertibility of $\Phi _ { m }$ on $G _ { m }$ . Moreover it enables us to show that the probability of the event $G _ { m }$ not happening is small, for all $m \in \mathcal { M } _ { n , p }$

Now, let $\begin{array} { r } { \overline { { G } } : = \bigcap _ { m \in { \mathcal { M } } _ { n , p } } G _ { m } } \end{array}$ . On G we can compute the least squares estimate ${ \widehat { \beta } } _ { m }$ of $\beta$ in $S _ { m }$ for all $m \in \mathcal { M } _ { n , p }$ . Hence on this set, we can define an integer

$$
\widehat { m } \in \mathop { \mathrm { a r g } } _ { m \in \mathcal { M } _ { n , p } } ( \gamma _ { n } ( \widehat { \beta } _ { m } ) + \mathrm { p e n } ( m ) ) ,
$$

where

$$
\mathrm { p e n } ( m ) = \theta ( 1 + \delta ) D _ { m } \sigma ^ { 2 } / n \qquad \mathrm { f o r } \ \theta > 4 \ \mathrm { a n d } \ \delta > 0 .\tag{15}
$$

This penalty is crucial for model selection as it accounts for model complexity and thus helps to avoid overfitting. Moreover, we chose this particular penalty because its structure is specifically designed to control and bound certain terms that appear in the derivation of the oracle inequality (see Theorem 3.1). It is also inspired by the work of Brunel and Roche ([4], Equation (8)) and Brunel et al. ([3], Section 1.4).

Finally, we set the estimate of the true slope function $\beta$ as

$$
\widetilde { \beta } = \left\{ \begin{array} { l l } { \widehat { \beta } _ { \widehat { m } } } & { \mathrm { o n } \overline { { G } } , } \\ { 0 } & { \mathrm { o n } \overline { { G } } ^ { c } . } \end{array} \right.\tag{16}
$$

## 2.3 Risk measure

This section introduces the criterion used to evaluate the performance of the proposed estimator. We start by introducing the covariance operator of the true unobserved random curve X:

$$
\Gamma f ( \cdot ) = \mathbb { E } \left( \langle f , X \rangle _ { L ^ { 2 } } X ( \cdot ) \right) , \qquad f \in L ^ { 2 } ( [ 0 , 1 ] ) .\tag{17}
$$

Under our assumptions that the curves $( X _ { i } )$ are periodic, centered and second-order stationary, the eigenfunctions of Γ coincide with the Fourier basis (as discussed in Section 2.1 of Comte and Johannes [8]). Denoting by $( \lambda _ { j } ) _ { j \geq 1 }$ the associated eigenvalues, we have for each $j \geq 1$

$$
\Gamma \phi _ { j } = \lambda _ { j } \phi _ { j } , \qquad \langle \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } = \delta _ { j k } .
$$

Let us define the kernel K associated with the operator Γ :

$$
K ( s , t ) = \mathbb { E } \left( X ( s ) X ( t ) \right) , \qquad s , t \in [ 0 , 1 ] .\tag{18}
$$

We also introduce the empirical version of this operator, that we denote $\Gamma _ { n }$ , and which is associated with the kernel

$$
K _ { n } ( s , t ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } X _ { i } ( s ) X _ { i } ( t ) , \qquad s , t \in [ 0 , 1 ] .\tag{19}
$$

In particular for all $f \in L ^ { 2 } ( [ 0 , 1 ] )$ ,

$$
\Gamma _ { n } f ( \cdot ) = \int _ { 0 } ^ { 1 } K _ { n } ( \cdot , t ) f ( t ) d t = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle f , X _ { i } \rangle _ { L ^ { 2 } } X _ { i } ( \cdot ) .\tag{20}
$$

We next define the covariance kernel of the reconstructed process $\widetilde { X }$ by

$$
\widetilde K ( s , t ) = \mathbb { E } \left( \widetilde X ( s ) \widetilde X ( t ) \right) , \qquad s , t \in [ 0 , 1 ] ,\tag{21}
$$

and its associated covariance operator

$$
\widetilde \Gamma f ( \cdot ) = \int _ { 0 } ^ { 1 } \widetilde K ( \cdot , t ) f ( t ) d t = \mathbb { E } \left( \langle f , \widetilde X \rangle _ { L ^ { 2 } } \widetilde X ( \cdot ) \right) , \qquad f \in L ^ { 2 } ( [ 0 , 1 ] ) .\tag{22}
$$

Given the reconstructed sample $( \widetilde { X } _ { i } ) _ { i = 1 , \dots , n }$ , we also consider the empirical covariance kernel

$$
\widetilde { K } _ { n } ( s , t ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \widetilde { X } _ { i } ( s ) \widetilde { X } _ { i } ( t ) , \qquad s , t \in [ 0 , 1 ] ,\tag{23}
$$

together with the corresponding empirical covariance operator $\widetilde { \Gamma } _ { n }$ , defined for any $f \in L ^ { 2 } ( [ 0 , 1 ] )$ by

$$
\widetilde { \Gamma } _ { n } f ( \cdot ) = \int _ { 0 } ^ { 1 } \widetilde { K } _ { n } ( \cdot , t ) f ( t ) d t = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle f , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } \widetilde { X } _ { i } ( \cdot ) .\tag{24}
$$

These operators naturally induce the following bilinear forms (and associated seminorms)

$$
\langle f , g \rangle _ { \Gamma _ { n } } = \langle \Gamma _ { n } f , g \rangle _ { L ^ { 2 } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle f , X _ { i } \rangle _ { L ^ { 2 } } \langle g , X _ { i } \rangle _ { L ^ { 2 } } ,\tag{25}
$$

$$
\langle f , g \rangle _ { \widetilde { \Gamma } _ { n } } = \langle \widetilde { \Gamma } _ { n } f , g \rangle _ { L ^ { 2 } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle f , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } \langle g , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } ,\tag{26}
$$

$$
\langle f , g \rangle _ { \widetilde { \Gamma } } = \langle \widetilde { \Gamma } f , g \rangle _ { L ^ { 2 } } = \mathbb { E } \left( \langle f , \widetilde { X } \rangle _ { L ^ { 2 } } \langle g , \widetilde { X } \rangle _ { L ^ { 2 } } \right) ,\tag{27}
$$

$$
\langle f , g \rangle _ { \Gamma } = \langle \Gamma f , g \rangle _ { L ^ { 2 } } = \mathbb { E } \left( \langle f , X \rangle _ { L ^ { 2 } } \langle g , X \rangle _ { L ^ { 2 } } \right) .\tag{28}
$$

To quantify the performance of the estimator, we study the prediction error under the distribution of the true process. More specifically, we consider a new curve $X _ { n + 1 }$ with a scalar response $Y _ { n + 1 }$ independent of the sample with $( X _ { n + 1 } , Y _ { n + 1 }$ satisfying (1). We wish to quantify the following error

$$
\begin{array} { r l } & { \mathbb { E } \left( \left( \langle \widetilde { \beta } , X _ { n + 1 } \rangle _ { L ^ { 2 } } - \mathbb { E } \left( Y _ { n + 1 } | X _ { n + 1 } \right) \right) ^ { 2 } | X _ { 1 } , \cdots , X _ { n } \right) = \mathbb { E } \left( \left( \langle \widetilde { \beta } - \beta , X _ { n + 1 } \rangle _ { L ^ { 2 } } \right) ^ { 2 } | X _ { 1 } , \cdots , X _ { n } \right) } \\ & { \qquad = \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } , } \end{array}\tag{29}
$$

This criterion quantifies the average discrepancy between predictions based on $\widetilde { \beta }$ and those based on $\beta _ { \mathrm { {  } } }$ , when both are evaluated on the same unobserved curve.

Remark 1. In the functional linear model, the quantity of interest is the prediction $\langle \beta , X \rangle _ { L ^ { 2 } }$ ， rather than the function $\beta$ itself. For this reason, the natural risk is

$$
\mathbb { E } \big ( \langle \widetilde { \beta } - \beta , X \rangle _ { L ^ { 2 } } ^ { 2 } \big ) = \mathbb { E } \left( \lVert \widetilde { \beta } - \beta \rVert _ { \Gamma } ^ { 2 } \right) ,
$$

which measures the prediction error. The $L ^ { 2 }$ -norm does not take into account the distribution of the covariate $X ,$ and in particular directions corresponding to small eigenvalues of the covariance operator Γ have little influence on the prediction, but may contribute significantly to the $L ^ { 2 }$ -error.

## 3 Main results

In this section, we derive an upper bound of $\mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } )$ and $\mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } )$ in the oracle setting. Before stating our assumptions, we recall the notion of a sub-Gaussian random variable from Vershynin [16], used throughout this section. A random variable A is sub-Gaussian with variance factor v if for all $t \in \mathbb { R }$

$$
\mathbb { E } \left( e ^ { t ( A - \mathbb { E } ( A ) ) } \right) \le e ^ { \frac { t ^ { 2 } v } { 2 } } .
$$

Throughout this section, we work under the following assumptions:

• (H1) For all $i \in \{ 1 , . . . , n \}$ and for all $\begin{array} { r } { ( c _ { l } ) _ { l \geq 1 } \in l ^ { 2 } , \sum _ { l > 1 } c _ { l } \langle X _ { i } , \phi _ { l } \rangle _ { L ^ { 2 } } } \end{array}$ is sub-Gaussian with variance factor $K _ { 1 } ^ { 2 } \sum _ { l > 1 } c _ { l } ^ { 2 } \lambda _ { l }$ , where $K _ { 1 }$ is a positive constant.

• (H2) For all $i = 1 , . . . , n$ and $h = 0 , . . . , p - 1 , \eta _ { i , h }$ is sub-Gaussian with variance factor $\tau ^ { 2 }$

• (H3) We assume that the eigenvalues of Γ, namely the $\lambda _ { j } \mathrm { ^ { \circ } s }$ decrease in a polynomial way : there exist two constants $c ^ { \prime } > 0$ and $a > 1 / 2$ such that for all $j \geq 1$

$$
\lambda _ { j } \le c ^ { \prime } j ^ { - 2 a } .
$$

Assumption (H1) is satisfied under the following set of suficient conditions:

Assume that, for all $i \in \{ 1 , \ldots , n \}$ and all $j \geq 1$ , the Fourier coeficients $\langle X _ { i } , \phi _ { j } \rangle _ { L ^ { 2 } }$ are sub-Gaussian with variance factor $\lambda _ { j }$ . This assumption appears repeatedly in the works of Brunel and Roche [4] and Brunel et al. [3]. If in addition, for each i, the random variables $\langle X _ { i } , \phi _ { j } \rangle _ { L ^ { 2 } }$ and $\langle X _ { i } , \phi _ { k } \rangle _ { L ^ { 2 } }$ are independent whenever $j \neq k$ , then Assumption (H1) holds. We emphasize however, that these conditions may be restrictive in practice. In particular, the independence of the Fourier coeficients is a strong requirement that is generally satisfied only in specific settings, such as Gaussian processes represented in their Karhunen–Loève basis. Nevertheless, this independence assumption can be dispensed when Hypothesis (H3) holds for some $a > 1$ , thereby providing an alternative set of suficient conditions for Assumption (H1).

Assumption (H2) is primarily introduced to derive concentration inequalities. It is a mild and reasonable condition in practice when the observational noise is light-tailed.

Assumption (H3) imposes a polynomial decay on the eigenvalues of the covariance operator Γ. Such a condition is standard in functional data analysis (see, e.g., Brunel and Roche [4], Cardot and Johannes [6], and Comte and Johannes [8]) and reflects the regularity of the stochastic process X. In particular, it implies that most of the variability of X is captured by the first Fourier modes, whereas the contribution of higher-frequency components decreases progressively. Moreover, larger values of the parameter a correspond to smoother sample paths of $X$ . For example, Brownian motion satisfies this assumption with $a = 1$ , while integrated Brownian motion corresponds to $a = 2$ , consistent with its smoother sample paths.

Remark 2. Since both Γ and $\widetilde \Gamma$ are diagonal operators on $\operatorname { S p a n } \{ \phi _ { 1 } , \phi _ { 2 } , \dots , \phi _ { 2 N _ { n , p } + 1 } \}$ with strictly positive eigenvalues, the orthogonal projection of $\beta$ onto $S _ { m }$ with respect to the scalar product $\langle \cdot , \cdot \rangle _ { \Gamma }$ coincides with its orthogonal projection with respect to $\langle \cdot , \cdot \rangle _ { \widetilde { \Gamma } }$ . We denote this common projection by $\beta ^ { ( m ) }$ . The detailed proof of this result is established in Appendix E.

First let us state the following result which will be useful to derive oracle inequalities :

Proposition 3.1. Assume that there exists $l > 4$ such that $v _ { l } : = \mathbb { E } ( | \boldsymbol { \epsilon } | ^ { l } ) <$ ∞ and that E $\left( \langle \beta , X \rangle _ { L ^ { 2 } } ^ { 4 } \right) <$ ∞. Moreover we suppose that assumptions $( H 1 ) , ( H 2 ) , ( H 3 )$ are verified. If

$$
{ \cal D } _ { N _ { n , p } } < \operatorname* { m i n } { \left( \frac { n } { \ln ^ { 2 } n } , p \right) } ,
$$

$$
\lambda _ { D _ { N _ { n , p } } } \geq \frac { 2 } { n ^ { \alpha } }
$$

and $\alpha > 2 a$ in the definition of $s _ { n }$ (see in Equation (14)), then for all slope $\beta \in L ^ { 2 } ( [ 0 , 1 ] )$ we have

$$
\begin{array} { r l r } {  { \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } ) \le C _ { 0 } \bigg [ \operatorname* { m i n } _ { m \in \mathcal { M } _ { n , p } } \Big ( \mathbb { E } ( \| \beta ^ { ( m ) } - \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } ) + \mathrm { p e n } ( m ) \Big ) } } \\ & { } & { + \| \beta \| _ { L ^ { 2 } } ^ { 2 } \mathbb { E } ( \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 2 } ) } \\ & { } & { + \frac { \| \beta \| _ { L ^ { 2 } } ^ { 2 } } { n } \Big ( \mathbb { E } \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } + C _ { \beta } \Big ( \frac { 1 } { n } + \frac { 1 } { n p } \Big ) \bigg ] , } \end{array}\tag{30}
$$

where pen(m) is defined in $( 1 5 ) , C _ { 0 }$ is a positive constant which depends on $K _ { 1 } , \theta , l , \tau _ { l } , \delta , \sigma ^ { 2 }$ . Moreover we have:

$$
C _ { \beta } = \mathbb { E } ( \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } ) ^ { 1 / 2 } + \| \beta \| _ { L ^ { 2 } } ^ { 2 } + 1 .\tag{31}
$$

The following result provides an oracle inequality with respect to the norm $\| \cdot \| _ { \widetilde { \Gamma } }$ , thereby evaluating the prediction risk based on the reconstructed curves.

Theorem 3.1. Suppose that the assumptions of Proposition 3.1 hold. Then for all slope $\beta \in$ $L ^ { 2 } ( [ 0 , 1 ] )$ ,

$$
\begin{array} { r l r } {  { \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } ) \le C _ { 1 } \bigg [ \operatorname* { m i n } _ { m \in \mathcal { M } _ { n , p } } \Big ( \mathbb { E } ( \| \beta ^ { ( m ) } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } ) + \mathrm { p e n } ( m ) \Big ) } } \\ & { } & { + \| \beta \| _ { L ^ { 2 } } ^ { 2 } \mathbb { E } ( \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 2 } ) } \\ & { } & { + \frac { \| \beta \| _ { L ^ { 2 } } ^ { 2 } } { n } \Big ( \mathbb { E } \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } + C _ { \beta } ( \frac { 1 } { n } + \frac { 1 } { n p } ) \bigg ] , } \end{array}\tag{32}
$$

where pen(m) is defined in $( 1 5 ) , C _ { 1 }$ is a positive constant which depends on $\theta , l , \tau _ { l } , \delta , \sigma ^ { 2 }$ and $C _ { \beta }$ is defined in (31).

The following result provides an oracle inequality with respect to the norm $\| \cdot \| _ { \Gamma }$ , thereby evaluating the prediction risk based on the true curves.

Theorem 3.2. Suppose that the assumptions of Theorem 3.1 are satisfied. Then for all slope $\beta \in L ^ { 2 } ( [ 0 , 1 ] )$ we have,

$$
\begin{array} { r l } & { \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } ) \le C _ { 2 } \bigg [ \underset { m \in \mathcal { M } _ { n , p } } { \operatorname* { m i n } } \Big ( \| \beta ^ { ( m ) } - \beta \| _ { \Gamma } ^ { 2 } + \| \beta ^ { ( m ) } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } + \mathrm { p e n } ( m ) \Big ) } \\ & { \quad \quad \quad \quad + \| \beta \| _ { L ^ { 2 } } ^ { 2 } \mathbb { E } ( \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 2 } ) } \\ & { \quad \quad \quad \quad + \frac { \| \beta \| _ { L ^ { 2 } } ^ { 2 } } { n } \Big ( \mathbb { E } \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } + C _ { \beta } \bigg ( \frac { 1 } { n } + \frac { 1 } { n p } \bigg ) \bigg ] , } \end{array}\tag{33}
$$

where $\mathrm { p e n } ( m )$ is defined in $( 1 5 ) , C _ { 2 }$ is a positive constant which depends on $K _ { 1 } , \theta , l , \tau _ { l } , \delta , \sigma ^ { 2 } , c ^ { \prime } ;$ a and $C _ { \beta }$ is defined in (31).

If we assume furthermore that $\beta \in W ^ { p e r } ( k , L )$ we obtain the following oracle inequality :

$$
\begin{array} { r l r } {  { \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } ) \leq C _ { 3 } \bigg [ \operatorname* { m i n } _ { m \in \mathscr { M } _ { n , p } } \Big ( \| \beta ^ { ( m ) } - \beta \| _ { \Gamma } ^ { 2 } + \mathrm { p e n } ( m ) \Big ) } } \\ & { } & { + \mathbb { E } ( \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 2 } ) } \\ & { } & { + \frac { 1 } { n } \Big ( \mathbb { E } \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } + \frac { 1 } { p } + \frac { 1 } { n } + \frac { 1 } { n p } \bigg ] , } \end{array}\tag{34}
$$

where $C _ { 3 }$ is a positive constant which depends on $K _ { 1 } , \theta , l , \tau _ { l } , \delta , \sigma ^ { 2 } , c ^ { \prime } , a .$

The sequence of Proposition 3.1 and Theorems 3.1–3.2 reflects a progressive refinement of the risk bounds. Proposition 3.1 establishes an oracle type inequality expressed in terms of the empirical covariance operator of the reconstructed data, which is the most directly observable quantity in our framework. Theorem 3.1 then replaces this empirical operator by the covariance operator of the reconstructed process, thereby removing the efect of sampling variability. Finally, Theorem 3.2 transfers the result to the covariance operator of the true underlying process by controlling the discrepancy between the reconstructed and true covariance structures.

The bounds obtained in Theorems 3.1 and 3.2 are of oracle type. They show that the estimator $\widetilde { \beta }$ performs, up to a multiplicative constant and additional remainder terms, nearly as well as the best estimator in the collection $( S _ { m } ) _ { m \in \mathcal { M } _ { n , p } } .$ More precisely, the leading term in the bound of Theorem 3.1 is governed by the oracle criterion

$$
\operatorname* { m i n } _ { m \in \mathcal { M } _ { n , p } } \left( \| \beta ^ { ( m ) } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } + \mathrm { p e n } ( m ) \right) ,
$$

whereas Theorem 3.2 involves

$$
\operatorname* { m i n } _ { m \in \mathcal { M } _ { n , p } } \left( \| \beta ^ { ( m ) } - \beta \| _ { \Gamma } ^ { 2 } + \mathrm { p e n } ( m ) \right) .
$$

The additional terms involving $\| \widetilde { X } - X \| _ { L ^ { 2 } }$ quantify the error induced by reconstructing the curves from discrete noisy observations and therefore measure the impact of the preliminary smoothing step on the final estimation procedure. The remaining terms, of order $1 / n , 1 / p$ , and $1 / ( n p )$ , arise from the proof of the oracle inequalities.

## 4 Convergence rates over Sobolev spaces

In this part, we derive convergence rates for the error $\mathbb { E } \left( \Vert \widetilde { \beta } - \beta \Vert _ { \Gamma } ^ { 2 } \right)$

Theorem 4.1. Suppose that the assumptions of Theorem 3.2 are $s a t i s f i e d ,$ with the exception that we consider $a \ge 1$ instead of $a > 1 / 2$ in (H3). Then, for all $\beta \in W ^ { p e r } ( k , L )$ :

$\begin{array} { r } { I f p \gtrsim \left( \frac { n } { \ln ^ { 2 } n } \right) ^ { 2 a } } \end{array}$ then,

$$
\mathbb { E } \bigg ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } \bigg ) = \mathcal { O } \Big ( n ^ { - \frac { 2 a + 2 k } { 2 a + 2 k + 1 } } \Big ) .\tag{35}
$$

$\begin{array} { r } { I f n ^ { \frac { 2 a } { 2 a + 2 k + 1 } } \lesssim p \lesssim \left( \frac { n } { \ln ^ { 2 } n } \right) ^ { 2 a } } \end{array}$ then,

$$
\mathbb { E } \Big ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } \Big ) = \mathcal { O } \Big ( n ^ { - \frac { 2 a + 2 k } { 2 a + 2 k + 1 } } + p ^ { - \frac { 2 a - 1 } { 2 a } } \Big ) .\tag{36}
$$

$I f p \lesssim n ^ { \frac { 2 a } { 2 a + 2 k + 1 } } t h e n ;$

$$
\mathbb { E } \bigg ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } \bigg ) = \mathcal { O } \Big ( p ^ { - \frac { 2 a - 1 } { 2 a } } \Big ) .\tag{37}
$$

The stronger assumption $a \ge 1$ is required in Theorem 4.1 to control the contribution of the reconstruction error and to ensure that the tail sum of the eigenvalues decays suficiently fast. Moreover, the three regimes described in Theorem 4.1 reveal a phase transition phenomenon that depends on the relative magnitudes of $p$ and n. When $p$ is large, the problem behaves essentially as if the curves were fully observed. In contrast, when $p$ is small, the estimation error is dominated by the discretization error, and increasing the sample size n alone is not suficient to improve the convergence rate. Furthermore, when the number of observation points is suficiently large, namely

$$
p \gtrsim \left( \frac { n } { \ln ^ { 2 } n } \right) ^ { 2 a } ,
$$

the convergence rate obtained in Theorem 4.1 coincides with the optimal rate established in the literature for functional linear models with fully observed curves. In particular, we recover the rate derived by Brunel and Roche [4]. Finally, as shown in the proof of Theorem 4.1, the dependence of the rate on $p$ is primarily driven by the observation noise rather than by the reconstruction procedure itself.

## 5 Lower Bound for the case p large

In this section we derive a lower bound for the case 1 of Theorem 4.1.

Theorem 5.1. Let us assume that there exist two constants $c ^ { \prime } > 0$ and $a > 1 / 2$ such that for all $j \geq 1$ 2

$$
j ^ { - 2 a } / c ^ { \prime } \leq \lambda _ { j } \leq c ^ { \prime } j ^ { - 2 a } .
$$

We also make the further assumption that

$$
\epsilon _ { i } \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } ) \qquad \forall i \in \{ 1 , \dots , n \} ,
$$

and that

$$
\eta _ { i , h } \sim { \mathcal { N } } ( 0 , \tau ^ { 2 } ) \qquad \forall i \in \{ 1 , \dots , n \} , \forall h \in \{ 0 , \dots , p - 1 \} .
$$

Then for $\textstyle p \geq \left( { \frac { n } { \ln ^ { 2 } n } } \right) ^ { 2 a }$ and for all $\beta \in W ^ { p e r } ( k , L )$ ，

$$
\mathbb { E } \Big ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } \Big ) \geq C _ { 4 } n ^ { - \frac { 2 a + 2 k } { 2 a + 2 k + 1 } } ,
$$

where $C _ { 4 }$ is a positive constant which depends on $k , L , c ^ { \prime } , \sigma$

In Theorem 4.1, only the upper bound $\lambda _ { j } \ \le \ c ^ { \prime } j ^ { - 2 a }$ is required. Indeed, the proof relies on controlling approximation and reconstruction errors, which involve tail sums of the eigenvalues and therefore only require an upper estimate on their decay. In contrast, the proof of Theorem 5.1 is based on a minimax lower-bound construction. To ensure that the candidate slope functions defined in (81) belong to $W ^ { \mathrm { p e r } } ( k , L )$ and remain suficiently separated in the prediction norm $\Vert \cdot \Vert _ { \Gamma } ,$ it is necessary to control both $\lambda _ { j }$ and $1 / \lambda _ { j }$ . This requires the two-sided condition $\begin{array} { r } { \frac { 1 } { c ^ { \prime } } j ^ { - 2 a } \leq \overbrace { \lambda _ { j } } \leq } \end{array}$ $c ^ { \prime } j ^ { - 2 a }$ . Furthermore, Theorem 5.1 establishes that, when $p$ is suficiently large, the rate n<sup>−</sup> 2a+2k+1 2a+2k constitutes a lower bound for the prediction risk over the class $W ^ { \mathrm { p e r } } ( k , L )$ . Combined with the upper bound derived in Theorem 4.1, this shows that our estimator attains the minimax rate over this class.

## 6 Simulations

Across the next four paragraphs, we present a framework for evaluating our estimator on a simulated dataset. This framework consists of generating the true functional curves, computing the corresponding scalar outputs, creating noisy discrete observations, reconstructing the curves, choosing adequate values for $\theta$ and $\delta ,$ estimating the slope function, and finally assessing the model’s performance. We study here three slope functions

$$
\beta _ { 1 } ( t ) = t ( t - 1 ) , \qquad t \in [ 0 , 1 ] ,\tag{38}
$$

which is in $W ^ { \mathrm { p e r } } ( 1 , 1 )$ and also studied in Brunel and Roche [4]. Since for all $t \in [ 0 , 1 ] \ \beta _ { 1 } ^ { ( 1 ) } ( t ) =$ $2 t - 1$ , therefore $\beta _ { 1 } ^ { ( 1 ) } ( 0 ) \neq \beta _ { 1 } ^ { ( 1 ) } ( 1 )$ and $\beta _ { 1 } \notin W ^ { \mathrm { p e r } } ( 2 , 1 )$

$$
\beta _ { 2 } ( t ) = 3 \exp \left( - \frac { ( t - 0 . 3 0 ) ^ { 2 } } { 2 \cdot 0 . 0 7 ^ { 2 } } \right) - 2 \exp \left( - \frac { ( t - 0 . 7 0 ) ^ { 2 } } { 2 \cdot 0 . 0 8 ^ { 2 } } \right) , \qquad t \in [ 0 , 1 ] ,\tag{39}
$$

which is defined as a linear combination of two Gaussian functions and

$$
\beta _ { 3 } ( t ) = 4 \sin ( 4 \pi t ) - \mathrm { s i g n } ( t - 0 . 3 ) - \mathrm { s i g n } ( 0 . 7 2 - t ) , \qquad t \in [ 0 , 1 ] ,\tag{40}
$$

which corresponds to the heavisine function.

![](images/a6b62adbc3fcc2dff3d41cbfff1a4f3e4ca90f06e7d53b6ce79c88cffe8ba5e7.jpg)

![](images/653c87b0c8b32db0da39a4c1cc066c0e26a2fb3f50dc1dbfa0f02d7ecd05e22c.jpg)

![](images/9b5b4d43b14b4fb96debb7b0b41768b7d04dd1b2cb00d115802710b1d5c9a285.jpg)  
Figure 1: Plot of $\beta _ { 1 }$ (left), $\beta _ { 2 }$ (middle), $\beta _ { 3 }$ (right)

## 6.1 Simulation setup and data generation

The simulation study begins with the generation of the true functional curves, denoted $X _ { i } ,$ representing the underlying functional processes. These curves are constructed using a truncated Karhunen–Loève decomposition:

$$
X ( t ) = \sum _ { j = 1 } ^ { 2 J + 1 } \xi _ { j } \phi _ { j } ( t ) , \qquad t \in [ 0 , 1 ] .\tag{41}
$$

Here, the sequence of independent, centered random variables $\{ \xi _ { 1 } , . . . , \xi _ { 2 J + 1 } \}$ has variances $\mathrm { V a r } ( \xi _ { j } ) =$ 二 $\lambda _ { j }$ . We set $J = 5 0 0$ and consider the $\xi _ { j } \mathrm { \bar { s } }$ to be Gaussian.

To generate the true curves $( X _ { i } ) _ { i = 1 , \ldots , n }$ , we first construct a high-resolution grid $G _ { 1 }$ over the observation interval [0, 1], defined as:

$$
G _ { 1 } : = \left\{ \frac { j } { p _ { \mathrm { s i m } } - 1 } \biggm | j = 0 , \ldots , p _ { \mathrm { s i m } } - 1 \right\} .
$$

This grid consists of $p _ { \mathrm { s i m } } = 1 0 0 0 0$ equally spaced points and provides a discrete approximation of the underlying continuous functions. Because it is not feasible to simulate the curves at every point in continuous time, this fine, regular grid serves as a practical compromise between approximation accuracy and computational cost. Next, we define a lower-resolution observation grid, G<sub>2</sub>, consisting of $p$ regularly spaced points where the noisy and discrete data are actually observed. This grid is defined as:

$$
G _ { 2 } : = \left\{ t _ { h } = { \frac { h } { p } } \biggm | h = 0 , \ldots , p - 1 \right\} .
$$

Because these observation points do not necessarily coincide with those of the high-resolution grid, we augment the latter to include $G _ { 2 }$ . The true curves are generated on this combined grid, $G _ { 1 } \cup G _ { 2 }$

Next, independent Gaussian noise terms $\eta _ { i , h }$ , with zero mean and variance $\tau ^ { 2 }$ are added to the discretized observations. The value of $\tau ^ { 2 }$ will vary and thus will be precised in each sections. This step aims to mimic measurement error in practical settings, yielding the observed sample $( Z _ { i } ) _ { i = 1 , \ldots , n } ,$ defined as:

$$
Z _ { i } ( t _ { h } ) = X _ { i } ^ { o b s } ( t _ { h } ) + \eta _ { i , h } , \qquad t _ { h } \in { \cal G } _ { 2 } .\tag{42}
$$

Once the functional predictors are generated, the corresponding scalar response $Y _ { i }$ for each curve is simulated according to the functional linear model defined in (4). To compute the integral term $\textstyle \int _ { 0 } ^ { 1 } X _ { i } ( t ) \beta ( t ) d t$ , we employ a discrete numerical approximation evaluated over the grid $G _ { 1 }$ . In particular,

$$
\int _ { 0 } ^ { 1 } X _ { i } ( t ) \beta ( t ) d t \approx \frac { 1 } { p _ { \mathrm { s i m } } - 1 } \sum _ { j = 0 } ^ { p _ { \mathrm { s i m } } - 1 } X _ { i } ( t _ { j } ) \beta ( t _ { j } ) .
$$

Finally, the noise variables $\epsilon _ { i }$ are generated independently from a zero-mean Gaussian distribution with variance $\sigma ^ { 2 } = 0 . 1$

## 6.2 Estimation of the slope function

Prior to estimating $\beta ,$ the functional predictors are reconstructed from the noisy and discretized observations. Following the procedure described in Section 2.1, we first compute the coeficients $( \widetilde { x } _ { i , j } ) _ { i = 1 , \dots , n } ~ ; ~ j = 1 , \dots , D _ { N _ { n , p } }$ as defined in Equation (6). These coeficients are then used to reconstruct the curves $\widetilde { X } _ { i }$ according to Equation (7). This procedure yields a sample of reconstructed curves $( \widetilde { X } _ { i } ) _ { i = 1 , \dots , n } .$ which can be interpreted as smoothed versions of the original observations. The choice of the basis dimension $D _ { N _ { n , p } }$ matters. In particular, $D _ { N _ { n , p } }$ must be chosen so as to ensure the invertibility of the empirical covariance matrix with high probability, and therefore to be able to compute $\widetilde { \beta }$ defined in (16). Theorem 3.1 shows that this condition is satisfied whenever

$$
{ \cal D } _ { N _ { n , p } } < \operatorname* { m i n } \left( \frac { n } { \ln ^ { 2 } n } , p \right) .
$$

Accordingly, in our simulations, we select the largest value of $D _ { N _ { n , p } }$ satisfying this constraint in order to achieve the most accurate curve reconstruction possible.

Based on the reconstructed curves, we proceed to estimate the slope function $\beta .$ To this end, we adopt a penalized model selection approach to determine the appropriate model complexity, as described in Section 2. The performance of the resulting estimator is supported by an oracle inequality (Theorem 3.1), provided that the penalty parameters satisfy $\theta > 4$ and $\delta > 0$ . In practice, these parameters must be calibrated; the corresponding selection procedure is detailed in Section 6.4.

![](images/6c29a2420a3098af02b53b3208e3008d810f9aee1a544b1fffcaa8b6ee17cd71.jpg)

Observed Noisy Data (Z)  
![](images/65da83b349e0237dd00d6704f668ad12ab6a7375aa451564623eebf916450825.jpg)

Reconstructed Curves (X-tilde)  
![](images/fc9bdd8ee33d237c5606be9df4be35111fc67a017261bba4ef369a597e4d227f.jpg)  
Figure 2: Simulation of a functional curve: true, observed, and reconstructed. Here $p = 5 0$ p<sub>sim</sub> = 10000, D<sub>N</sub> = 11, τ<sup>2</sup> = 0.1 and $\lambda _ { j } = 1 / j ^ { 4 }$ for all $j = 1 , \ldots , 2 J + 1$

## 6.3 Prediction error

As discussed in the previous sections, we focus on the prediction error of the proposed estimator. More specifically, we aim to analyze the behavior of

$$
\mathbb { E } \left( \lVert \widetilde { \beta } - \beta \rVert _ { \Gamma } ^ { 2 } \right) .
$$

As $\begin{array} { r } { \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } = \sum _ { j \geq 1 } \lambda _ { j } \langle \widetilde { \beta } - \beta , \phi _ { j } \rangle _ { L ^ { 2 } } ^ { 2 } } \end{array}$ , we approximate this quantity by using a finite number of elements in the Fourier basis,

$$
\| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } \approx \sum _ { j = 1 } ^ { 2 J + 1 } \lambda _ { j } \langle \widetilde { \beta } - \beta , \phi _ { j } \rangle _ { L ^ { 2 } } ^ { 2 } \approx \sum _ { j = 1 } ^ { 2 J + 1 } \lambda _ { j } \left( ( \widetilde { \alpha } _ { j } - \beta _ { j } ) ^ { 2 } \mathbb { 1 } _ { j \leq D _ { \widehat { m } } } + \beta _ { j } ^ { 2 } \mathbb { 1 } _ { j > D _ { \widehat { m } } } \right) ,
$$

where the last equality is derived from the definition of $\widetilde { \beta }$ given in Equation (16). Finally we estimate our prediction error using a Monte-Carlo method with $n _ { M C }$ independent samples. In particular we derive $n _ { M C }$ estimators of $\beta$ that we denote as $\widetilde { \beta } ^ { \left( l \right) }$ for $l = 1 , . . . , n _ { M C }$ . For each replication $l \in \{ 1 , \dots , n _ { \mathrm { M C } } \}$ , we obtain an estimator $\widetilde { \beta } ^ { \left( l \right) }$ defined by Equation (16) and its definition is given on G by $\begin{array} { r } { \widetilde { \beta } ^ { ( l ) } = \sum _ { j = 1 } ^ { D _ { \widehat { m } } ( l ) } \widetilde { \alpha } _ { j } ^ { ( l ) } \phi _ { j } } \end{array}$ , where $\widehat { m } ^ { ( l ) }$ denotes the model selected at iteration l. We then compute an estimate of the prediction error:

$$
\mathbb { E } \left( \Vert \widetilde { \beta } - \beta \Vert _ { \Gamma } ^ { 2 } \right) \approx \frac { 1 } { n _ { M C } } \sum _ { l = 1 } ^ { n _ { M C } } \widehat { e } _ { l } ,
$$

where $\begin{array} { r } { \widehat { e } _ { l } = \sum _ { j = 1 } ^ { 2 J + 1 } \lambda _ { j } \left( ( \widetilde { \alpha } _ { j } ^ { ( l ) } - \beta _ { j } ) ^ { 2 } \mathbb { 1 } _ { j \leq D _ { \widehat { m } ^ { ( l ) } } } + \beta _ { j } ^ { 2 } \mathbb { 1 } _ { j > D _ { \widehat { m } ^ { ( l ) } } } \right) } \end{array}$ . For our simulations we use $n _ { M C } = 5 0$

## 6.4 Calibration of θ and δ

The constant $\kappa : = \theta ( 1 + \delta )$ , which appears in the penalty term defined by (15), must be calibrated. To determine a suitable value for it, we evaluate its predictive performance across a grid of candidate values ranging from 1 to 20 (with a step size of 0.5), holding the sample size $n = 1 0 0 0$ and the number of observation points $p = 1 0 0 0$ fixed. Specifically, we conduct a Monte Carlo simulation study to assess the candidate values across the three distinct slope functions $\beta _ { 1 } , \beta _ { 2 }$ , and $\beta _ { 3 }$ . For each slope function, the prediction error, which is approximated via the procedure detailed in Section 6.3, is averaged over 50 independent replications. We then calculate the overall mean error across all three scenarios to identify a single, universal κ that minimizes the aggregate prediction error. The results of this calibration process are visualized in Figure 3. As indicated by the vertical dashed line, the aggregate prediction error attains its minimum at $\kappa = 3$ . However, given the constraints $\theta > 4$ and $\delta > 0$ , the definition of the penalty parameter strictly requires $\kappa > 4$ Consequently, we select $\kappa = 4 . 0 1$ , a value situated appropriately above this theoretical boundary but close to the lowest aggregate prediction error, for all subsequent simulations.

![](images/9356b9399a2aabb63975522e36f8c3a8723067e1b20d977bb289a87c2cb81aae.jpg)  
Figure 3: Average prediction error as a function of the penalty parameter $\kappa \in [ 1 , 2 0 ]$ across the three slope functions $\beta _ { 1 } , \beta _ { 2 } , \beta _ { 3 }$

## 6.5 Results

We now study the performance of our estimator when various slope functions are considered.

## 6.5.1 Study of $\beta _ { 1 }$

Now we consider $\beta _ { 1 }$ as the true slope function, which belongs to $W ^ { \mathrm { p e r } } ( 1 , 1 )$ and $\lambda _ { j } = 1 / j ^ { 4 }$ for all $j = 1 , \ldots , 2 J + 1$

In Figures 4a, 4b we analyze the impact of the sample size n by setting a fixed $p = 7 5 0 0$ . The variance of the noises $\tau ^ { 2 }$ and $\sigma ^ { 2 }$ are both set to 0.1. The sample size, $n ,$ varies across the set {200, 400, 800, 1600, 3200, 6400}. The resulting log-log plot yields a slope of −0.92, which is quite close to the value that we can expect given the convergence rate established in Theorem 4.1. Indeed, here we have $k = 1$ and $a \ = \ 2$ hence we expect to have an error which decreases in $n ^ { - ( 2 k + 2 a ) / ( 2 a + 2 k + 1 ) } = n ^ { - 6 / 7 }$

In Figures $^ { \mathrm { 4 c , } }$ 4d we study the impact of the size of the grid p by setting a fixed $n = 1 0 0 0$ . The variance of the noise $\sigma ^ { 2 } = 0 .$ 1 but we use here a much higher variance for the noise of the observation $\tau ^ { 2 } = 4$ . The number of observations, $p ,$ on the grid varies across the set {50, 100, 200, 300, 400, 500}. The resulting log-log plot yields a slope of $- 0 . 7 9$ which matches the rate that we expect from Theorem 4.1. Indeed, here we have $a \ : = \ : 2$ hence we expect to have an error which decreases in $p ^ { - ( 2 a - 1 ) / ( 2 a ) } = p ^ { - 3 / 4 }$ . However, for smaller values of $\tau _ { : }$ the rate of decrease is significantly higher. This could be explained by the fact that, when $\tau$ is small, the influence of noise on the reconstruction error of the curves is reduced; instead, the error is primarily driven by the choice of reconstruction scheme.

We also studied the case where the true curves are less smooth. In particular we consider now $\lambda _ { j } = 1 / j ^ { 2 }$ for all $j = 1 , \ldots , 2 J + 1$ . All the parameters and considered function remain unchanged.

![](images/444f0094abc87a8fddc8f0ebaad332d6438729d7b425730d7964185f00fa8d68.jpg)  
a. Average prediction error vs. sample size n

![](images/5fd25e53564c6ae48db5b43c7a3db98f077738bf5578f4fa3be62988a2bd1c8e.jpg)  
b. Log-log plot of average prediction error vs. sample size n

![](images/e83233125a84d629812ffc3db9c1664f9a3f4e7464b07b4fc6728f5213898946.jpg)  
c. Average prediction error vs. number of observation points p

![](images/b6c8864fb0d6cdca3979fef95b0c7dc7a371250efef2e20949db2f0d7eb68463.jpg)  
d. Log-log plot of average prediction error vs. number of observation points p

Figure 4: Monte Carlo approximation of the mean squared Γ-norm error as a function of the sample size n and the grid density p. The solid red curves represent the empirical error averaged over the Monte Carlo replications. In panels a. and c. the blue dashed curves represent the 10th and 90th empirical deciles of the Monte Carlo errors. Panels b. and d. display the corresponding mean errors on a log-log scale; in these panels, the black dashed lines represent linear fits used to assess the empirical rate of decay of the error.

On Figure 5 we observe that the resulting log-log plot for the case p fixed - varying n yields a slope

![](images/5e7d99b3c74541376742f077ee118f4308b8ee613510b2451da06e03aaf65e42.jpg)  
a. Average prediction error vs. sample size n

![](images/65df88b6f51d52496b69dda2b7437c7d064573d1191f440fb538f7b75bda3d94.jpg)  
b. Log-log plot of average prediction error vs. sample size n

![](images/75766e3fd0b677352612aa72c87d6655bec3b7f39a1d37914127bcda889839d3.jpg)  
c. Average prediction error vs. number of observation points p

![](images/5c1b05569dfb5712c75b3a7dc2ea7f501998d55634988f03306d8bae295de09d.jpg)  
d. Log-log Plot of average prediction error vs. number of observation points p

Figure 5: Monte Carlo approximation of the mean squared Γ-norm error as a function of the sample size n and the grid density p. The solid red curves represent the empirical error averaged over the Monte Carlo replications. In panels a. and c. the blue dashed curves represent the 10th and 90th empirical deciles of the Monte Carlo errors. Panels b. and d. display the corresponding mean errors on a log-log scale; in these panels, the black dashed lines represent linear fits used to assess the empirical rate of decay of the error.

of −0.76, which is close to the value that we can expect given the convergence rate established in Theorem 4.1. Indeed, here we have $k = 1$ and $a = 1$ therefore we expect to have an error which decreases in $n ^ { - ( 2 k + 2 a ) / ( 2 a + 2 k + 1 ) } = n ^ { - 0 . 8 }$ . For the case n fixed - varying p the log-log plot gives an estimated slope of approximately −0.55, which indicates that the empirical decay of the prediction error is quite consistent with the expected $p ^ { - ( 2 a - 1 ) / ( 2 a ) } = p ^ { - 1 / 2 }$ behavior.

## 6.5.2 Study of $\beta _ { 2 }$

In this section, we consider the true slope function $\beta _ { 2 }$ . Because $\beta _ { 2 }$ does not perfectly satisfy the periodic boundary condition $\beta _ { 2 } ( 0 ) = \beta _ { 2 } ( 1 )$ , this setup allows us to evaluate how our estimator perform even if the true slope function does not perfectly lie in the space $W ^ { \mathrm { p e r } }$ . In this part we pick $a = 2$ and aim to compare our result with the first study of $\beta _ { 1 } .$ . The same parameters are used for the simulations. The log-log plot in Figures 6b and 6d yield respectively slopes of −0.86 and −0.77, which are very close to the one that we got previously and which match our theoretical results.

## 6.5.3 Study of $\beta _ { 3 }$

In this section, we evaluate the estimator’s ability to handle change points by examining its behavior around the two discontinuities located at $t _ { 1 } = 0 . 3 0$ and $t _ { 2 } = 0 . 7 2$ of the true slope function $\beta _ { 3 }$ defined in Equation (40). Instead of using a threshold-based detection method, we adopt a visual approach to better understand the local behavior of our estimator. We plot the true slope function against 10 estimated curves obtained from independent samples. To illustrate the asymptotic

![](images/56d7df5c7a0872f40433cbfd7c48b7ff6a79e4142129ffbec1afda4c2f9669bd.jpg)  
a. Average prediction error vs. sample size n

![](images/f1147834b4558b941bd46bead8b661cc4301b728c8c9fab56fce116b937d60eb.jpg)  
b. Log-log plot of average prediction error vs. sample size n

![](images/2005dad8a517fd3505e1053be7b580311babaf62562cb5cc87aa842d5c5888dd.jpg)  
c. Average prediction error vs. number of observation points p

![](images/9833b4387e96152a5647d9f87e6616abcbb89e535b4f324200d0d75d57409c0b.jpg)  
d. Log-log plot of average prediction error vs. number of observation points p

Figure 6: Monte Carlo approximation of the mean squared Γ-norm error as a function of the sample size n and the grid density p. The solid red curves represent the empirical error averaged over the Monte Carlo replications. In panels a. and c. the blue dashed curves represent the 10th and 90th empirical deciles of the Monte Carlo errors. Panels b. and d. display the corresponding mean errors on a log-log scale; in these panels, the black dashed lines represent linear fits used to assess the empirical rate of decay of the error.

behavior of the estimator, we compare two distinct settings: a moderate sample size $( n = 8 0 0 ;$ $p = 5 0 0 0 , D _ { N _ { n , p } } = 1 5 )$ and a large sample size $( n = 6 4 0 0 , p = 5 0 0 0 , D _ { N _ { n , p } } = 8 1 )$ Figure

Estimation of the Slope Function $\beta _ { 3 }$  
![](images/513121d7138075db3b6c6a5e25ecef1268ce67cbbb063807dd9a5f29c261bc3d.jpg)  
a. n = 800, p = 5000, and $D _ { N _ { n , p } } = 1 5$

Estimation of the Slope Function $\beta _ { 3 }$  
![](images/8a66435d37b99400672df7e533a0947728a37a2bb9a6d25759107a5218ecf9df.jpg)  
b. n = 6400, p = 5000, and $D _ { N _ { n , p } } = 8 1$  
Figure 7: Estimation of the slope function $\beta _ { 3 }$ for $p = 5 0 0 0$ and diferent sample sizes n.

7a displays the results for the setting with $n = 8 0 0$ . The estimated curves (in red) capture the global trend of $\beta _ { 3 }$ but completely smooth over the discontinuities. On the other hand, Figure 7b illustrates the results for $n = 6 4 0 0$ . With a larger sample size and a higher dimension for the basis $( D _ { N _ { n , p } } = 8 1 )$ , the variance of the estimators is significantly reduced, yielding a good fit on the continuous segments of $\beta _ { 3 }$ . More importantly, around the jump location $t _ { 2 }$ , the estimators actively attempt to fit the discontinuities as it benefits from an increased flexibility provided by $D _ { N _ { n , p } } = 8 1$

## 6.6 Efect of the unknown variance $\sigma ^ { 2 }$

Throughout this work we have assumed that $\sigma ^ { 2 }$ , the variance of ϵ defined in Equation (1), is known. In practice however, this is generally not the case and σ must therefore be estimated. To this end, for each model dimension $m _ { \colon }$ , we consider the empirical variance estimator

$$
{ \widehat { \sigma } } _ { m } ^ { 2 } : = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \left( Y _ { i } - \left. { \widehat { \beta } } _ { m } , { \widetilde { X } } _ { i } \right. _ { L ^ { 2 } } \right) ^ { 2 } .\tag{43}
$$

We then select the dimension $\widehat { m } _ { \mathrm { p l u g } }$ by minimizing the following penalized criterion:

$$
\widehat { m } _ { \mathrm { p l u g } } \in \mathop { \mathrm { a r g } } _ { m \in \mathcal { M } _ { n , p } } \left( \gamma _ { n } ( \widehat { \beta } _ { m } ) + \theta ( 1 + \delta ) \frac { D _ { m } \widehat { \sigma } _ { m } ^ { 2 } } { n } \right) .\tag{44}
$$

Table 1: Approximate mean squared Γ-norm prediction errors for the slope function $\beta _ { 1 }$ with variance $\sigma ^ { 2 }$ known or unknown. The upper panel reports errors as n varies with $p = 7 5 0 0$ fixed, while the lower panel reports errors as p varies with $n = 1 0 0 0$ fixed.

$$
\overline { { \mathrm { F i x e d } ~ p = 7 5 0 0 } }
$$

$$
\beta _ { 1 }
$$

$$
\overline { { \sigma ^ { 2 } } }
$$

$$
\overline { { n = 2 0 0 } }
$$

$$
\overline { { 1 . 4 2 \times 1 0 ^ { - 4 } } }
$$

$$
n = 4 0 0
$$

$$
1 . 2 7 \times 1 0 ^ { - 4 }
$$

$$
\overline { { 6 . 9 0 \times 1 0 ^ { - 5 } } }
$$

$$
n = 8 0 0
$$

$$
8 . 5 6 \times 1 0 ^ { - 5 }
$$

$$
\overline { { 3 . 5 2 \times 1 0 ^ { - 5 } } }
$$

$$
\beta _ { 1 }
$$

$$
n = 1 6 0 0
$$

$$
3 . 7 3 \times 1 0 ^ { - 5 }
$$

$$
\overline { { 1 . 7 3 \times 1 0 ^ { - 5 } } }
$$

$$
\overline { { n = 3 2 0 0 } }
$$

$$
1 . 8 3 \times 1 0 ^ { - 5 }
$$

$$
\overline { { n = 6 4 0 0 } }
$$

$$
\overline { { 1 . 0 9 \times 1 0 ^ { - 5 } } }
$$

$$
8 . 6 8 \times 1 0 ^ { - 6 }
$$

$$
\overline { { 6 . 5 5 \times 1 0 ^ { - 6 } } }
$$

$$
6 . 1 1 \times 1 0 ^ { - 6 }
$$

$$
\overline { { { \mathrm { F i x e d ~ } } n = 1 0 0 0 } }
$$

$$
\beta _ { 1 }
$$

$$
\overline { { \sigma ^ { 2 } } }
$$

$$
p = 5 0
$$

$$
2 . 2 4 \times 1 0 ^ { - 4 }
$$

$$
\beta _ { 1 }
$$

$$
p = 1 0 0
$$

$$
2 . 1 8 \times 1 0 ^ { - 4 }
$$

$$
p = 2 0 0
$$

$$
\overline { { 1 . 0 6 \times 1 0 ^ { - 4 } } }
$$

$$
9 . 3 2 \times 1 0 ^ { - 5 }
$$

$$
p = 3 0 0
$$

$$
\overline { { 6 . 6 5 \times 1 0 ^ { - 5 } } }
$$

$$
4 . 9 6 \times 1 0 ^ { - 5 }
$$

$$
5 . 0 9 \times 1 0 ^ { - 5 }
$$

$$
4 . 7 2 \times 1 0 ^ { - 5 }
$$

$$
p = 4 0 0
$$

$$
\overline { { 3 . 9 2 \times 1 0 ^ { - 5 } } }
$$

$$
\overline { { p = 5 0 0 } }
$$

$$
3 . 8 7 \times 1 0 ^ { - 5 }
$$

$$
\overline { { 3 . 9 4 \times 1 0 ^ { - 5 } } }
$$

$$
3 . 4 4 \times 1 0 ^ { - 5 }
$$

Table 2: Approximate mean squared Γ-norm prediction errors for the irregular slope function $\beta _ { 2 }$ with variance $\sigma ^ { 2 }$ known or unknown. The upper panel reports errors as n varies with $p = 7 5 0 0$ fixed, while the lower panel reports errors as $p$ varies with $n = 1 0 0 0$ fixed.
<table><tr><td colspan="9">Fixed  $p = 7 5 0 0$ </td></tr><tr><td>Slope</td><td> $\overline { { \sigma ^ { 2 } } }$ </td><td> $n = 2 0 0$ </td><td> $n = 4 0 0$ </td><td> $n = 8 0 0$ </td><td> $n = 1 6 0 0$ </td><td> $\overline { { n = 3 2 0 0 } }$ </td><td></td><td> $\overline { { n = 6 4 0 0 } }$ </td></tr><tr><td> $\overline { { \beta _ { 2 } } }$ </td><td>known</td><td> $\overline { { 2 . 6 0 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 1 . 6 4 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 1 . 0 2 \times 1 0 ^ { - 3 } } }$ </td><td> $5 . 1 8 \times 1 0 ^ { - 4 }$ </td><td> $2 . 3 6 \times 1 0 ^ { - 4 }$ </td><td></td><td> $\overline { { 1 . 4 8 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td> $\beta _ { 2 }$ </td><td>unknown</td><td> $3 . 3 1 \times 1 0 ^ { - 3 }$ </td><td> $1 . 8 4 \times 1 0 ^ { - 3 }$ </td><td> $1 . 0 7 \times 1 0 ^ { - 3 }$ </td><td> $4 . 5 6 \times 1 0 ^ { - 4 }$ </td><td> $2 . 6 5 \times 1 0 ^ { - 4 }$ </td><td></td><td> $1 . 5 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td colspan="9">Fixed  $\overline { { n = 1 0 0 0 } }$ </td></tr><tr><td>Slope</td><td> $\overline { { \sigma ^ { 2 } } }$ </td><td> $\overline { { p = 5 0 } }$ </td><td> $p = 1 0 0$ </td><td> $p = 2 0 0$ </td><td> $p = 3 0 0$ </td><td> $p = 4 0 0$ </td><td></td><td> $p = 5 0 0$ </td></tr><tr><td> $\beta _ { 2 }$ </td><td>known</td><td> $5 . 9 3 \times 1 0 ^ { - 3 }$ </td><td> $\overline { { 2 . 6 6 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 1 . 6 6 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 1 . 1 1 \times 1 0 ^ { - 3 } } }$ </td><td></td><td> $\overline { { 1 . 0 1 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 1 . 0 6 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td> $\beta _ { 2 }$ </td><td>unknown</td><td> $5 . 6 3 \times 1 0 ^ { - 3 }$ </td><td> $2 . 8 1 \times 1 0 ^ { - 3 }$ </td><td> $1 . 5 1 \times 1 0 ^ { - 3 }$ </td><td> $1 . 2 2 \times 1 0 ^ { - 3 }$ </td><td></td><td> $1 . 0 7 \times 1 0 ^ { - 3 }$ </td><td> $9 . 1 9 \times 1 0 ^ { - 4 }$ </td></tr></table>

Tables 1 and 2 compare the mean squared Γ-norm prediction errors obtained when $\sigma ^ { 2 }$ is known and when it is estimated using (43). The comparison is performed for $\beta _ { 1 }$ (38) and $\beta _ { 2 } ~ ( 3 9 )$ . As in the previous simulations, we consider two asymptotic regimes: first, n varies while $p = 7 5 0 0$ is fixed; second, $p$ varies while $n = 1 0 0 0$ is fixed. The results show that replacing $\textstyle { \bar { \sigma } } ^ { 2 }$ by the residual-based estimator $\widehat { \sigma } _ { m } ^ { 2 }$ has little impact on the performance of the estimator. Indeed, in the two tables the prediction errors obtained with known and unknown variance remain of the same order of magnitude and exhibit similar decreasing behavior as n or p increases. This suggests that the procedure remains reliable in the more realistic setting where the regression noise variance is unknown.

## 7 Application to meteorological data

We apply our estimation procedure to meteorological data provided by Météo-France, originally covering the period 1950–2024. For this application, we retain observations from the recent period 2019–2024 and average them over time to avoid making the estimated curve depend on the particular weather realization of a single year. After removing observations with missing values and excluding February 29 to obtain a common yearly calendar, the final dataset consists of $n = 3 0 5$ temperature curves, each corresponding to one location, and $p = 3 6 5$ daily measurements. The annual average precipitation recorded at each site is used as the scalar response variable, while the corresponding annual temperature profile is treated as a functional covariate. This yields a scalar-on-function regression framework, whose objective is to investigate how the shape of the temperature curve is associated with annual rainfall across the selected French departments highlighted in Figure 8a. Figures 8b and 8c show examples of the functional data for four stations located in Finistère. The initial step involves partitioning the dataset into a training set (80% of observations) and a test set (20% of observations) to ensure an unbiased evaluation of the final model. We denote as trainIDX and testIDX the indexes of the data present in respectively the train and test set, as well as $n _ { \mathrm { t r a i n } }$ and $n _ { \mathrm { t e s t } }$ their dimensions. Both sets are subsequently centered to standardize the data. More particularly, the empirical means are computed exclusively from the training set. For the scalar response, we define

$$
\overline { { Y } } _ { \mathrm { t r a i n } } = \frac { 1 } { n _ { \mathrm { t r a i n } } } \sum _ { i \in \mathrm { t r a i n I D X } } Y _ { i } ,
$$

while, for each time point $t _ { j } .$ , the mean temperature curve is given by

$$
\overline { { Z } } _ { \mathrm { t r a i n } } ( t _ { j } ) = \frac { 1 } { n _ { \mathrm { t r a i n } } } \sum _ { i \in \mathrm { t r a i n I D X } } Z _ { i } ( t _ { j } ) .
$$

Both the training and test observations are then centered using these training-set means. Thus, the centered training data are defined as

$$
\widetilde { Y } _ { \mathrm { t r a i n } } = Y _ { \mathrm { t r a i n } } - \overline { { Y } } _ { \mathrm { t r a i n } } ,
$$

and for all $j = 0 , \ldots , p - 1$

$$
\begin{array} { r } { \widetilde { Z } _ { \mathrm { t r a i n } } ( t _ { j } ) = Z _ { \mathrm { t r a i n } } ( t _ { j } ) - \overline { { Z } } _ { \mathrm { t r a i n } } ( t _ { j } ) . } \end{array}
$$

Similarly, the test data are centered using the same means computed from the training set:

$$
\widetilde { Y } _ { \mathrm { t e s t } } = Y _ { \mathrm { t e s t } } - \overline { { Y } } _ { \mathrm { t r a i n } } ,
$$

and for all $j = 0 , \ldots , p - 1$

$$
\begin{array} { r } { \widetilde { Z } _ { \mathrm { t e s t } } ( t _ { j } ) = Z _ { \mathrm { t e s t } } ( t _ { j } ) - \overline { { Z } } _ { \mathrm { t r a i n } } ( t _ { j } ) . } \end{array}
$$

We then reconstruct the curves with the same methodology as in Section 2.1. Afterward we estimate the variance $\sigma ^ { 2 }$ using only the elements of training set $\widetilde { Z } _ { \mathrm { t r a i n } }$ and $\widetilde { Y } _ { \mathrm { t r a i n } }$ , by using the estimator defined in Equation (43). The functional linear regression model is subsequently fitted on the training set, using the estimator of $\sigma ^ { 2 }$ to select the dimension $\widehat { m } _ { \mathrm { p l u g } }$ (see (44)). After this step we obtain an estimator of the slope function of the form

$$
\widetilde { \beta } ( t ) = \sum _ { j = 1 } ^ { \widehat { m } _ { \mathrm { p l u g } } } \widetilde { \alpha } _ { j } \phi _ { j } ( t ) ,
$$

and its predictive performance is evaluated on the held-out test set by calculating the root mean squared error (RMSE). Since the true curves are not available, we use the reconstructed ones to make prediction. More particularly, in order to do so, we reconstruct the curves from the noisy and discrete data of the test set $( \widetilde { Z } _ { \mathrm { t e s t } } )$ . Thus we obtain

$$
\widetilde { X } _ { \mathrm { t e s t } _ { i } } ( t ) = \sum _ { j = 1 } ^ { D _ { N _ { n } , p } } \widetilde { x } _ { i , j } \phi _ { j } ( t ) , \qquad \mathrm { f o r ~ a l l ~ } i \in \mathrm { t e s t I D X ~ a n d ~ f o r ~ a l l ~ } t \in [ 0 , 1 ] ,
$$

and then, for each observation i in the test set we predict

$$
\widetilde { Y } _ { \mathrm { p r e d } _ { i } } = \sum _ { j = 1 } ^ { \widehat { m } _ { \mathrm { p l u g } } } \widetilde { x } _ { i , j } \widetilde { \alpha } _ { j } .
$$

In Figure 9 we can see that the estimated slope function exhibits two extrema. One is negative during winter and early spring, with its lowest values around February–March, which indicates that higher temperatures during this period are associated with lower annual precipitation. The coeficient then plateau around 0 during the summer. Thus at this time the temperature curve does not impact the annual precipitation. The estimated slope function then increase quickly, reaching its maximum around October-November. This indicates a positive association between autumnal temperatures and annual precipitation. It then decreases during winter. Figure 10a shows a positive relationship between the observed and predicted centred values, which indicates that the model is able to reproduce the overall trend in annual precipitation. Figure 10b confirm this. In particular, it shows that the prediction errors are mostly centred around zero therefore we don’t have a systematic bias. However, the lower whisker is longer than the upper one, suggesting a slightly greater dispersion among negative residuals. Hence, some locations are more strongly overpredicted than underpredicted.

![](images/b5eb904198598509d3273cb1054be07ae3b9045e432ea2d7a98c8b7dcf190b83.jpg)

a. French departments included in the meteorological dataset. Evolution of the temperature by year  
![](images/819fe5de1e12fe276a7170a16522ed059a9941bec5a1212103d7656c251b8156.jpg)

b. Averaged annual temperature curves across four stations in Finistère. Evolution of the temperature by year for each city  
![](images/13dd6b1ab6bab4e5fa253e5deb5802e322964a59c64b9e90327a73045ad8fce8.jpg)  
c. Averaged annual temperature curves for the selected stations in Finistère.  
Figure 8: Meteorological dataset map and the associated temperature curves in Finistère.

![](images/fcb1c03a170e1323219830462ce1741520518e2f4b5b00db04044144741ab7e2.jpg)  
Figure 9: Estimated functional coeficient $\widetilde { \beta } ( t )$ , describing the efect of the annual temperature profile on average annual precipitation.

![](images/119285592d21ef70b9cdac33423389b23611ba81287012eca9608ea35ee3f65b.jpg)  
a. Plot of the actual vs predicted values, the red dashed line is $y = x .$

![](images/16bfe7d64238e8b43026caeaca138b9485ca57cd06c9eee10a28ccef5b292adf.jpg)  
b. Boxplot of the model’s residuals, showing the distribution of the prediction errors on the test set.  
Figure 10: Performance assessment of the estimator on the test set.

The model achieves a RMSE of 0.58, which is quite low. However, this performance is only a small improvement over the baseline method where we use the average rainfall quantity with all the data from the training set to predict the average rainfall of the test set. This method yields a RMSE of 0.91. A primary reason for this small diference could be that the baseline method is already quite efective. Given that the target variable is the annual average rainfall, it is a stable and predictable value. The variations from year to year might be minimal, meaning that a simple prediction of the overall average is a already a strong starting point.

## A Gradient Calculation for the Least Squares Minimization

Lemma A.1. The function F which is such that

$$
F ( v ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( Y _ { i } - \sum _ { j = 1 } ^ { D _ { m } } v _ { j } \langle \phi _ { j } , \widetilde X _ { i } \rangle _ { L ^ { 2 } } \right) ^ { 2 } , \qquad f o r \ a l l \ v \in \mathbb { R } ^ { D _ { m } } ,
$$

is diferentiable and its gradient is

$$
\nabla F ( \boldsymbol { v } ) = - 2 b + 2 \Phi _ { m } \boldsymbol { v } ,
$$

where $\Phi _ { m }$ is defined by Equation (11) and b by Equation (12).

Proof. The function F being convex and $\mathcal { C } ^ { 1 }$ , finding a minimizer $\widetilde { \alpha } \in \mathbb { R } ^ { D _ { m } }$ of $F$ is equivalent to   
finding an $\widetilde { \alpha }$ in $\mathbb { R } ^ { D _ { m } }$ such that $\nabla F ( \widetilde { \alpha } ) = 0$ . Let us compute $\nabla F$   
First let us consider $k \in \{ 1 , \ldots , D _ { m } \}$ , then

$$
\begin{array} { r l } & { \displaystyle \frac { \partial F ( \boldsymbol { v } ) } { \partial v _ { k } } = - \frac { 2 } { n } \sum _ { i = 1 } ^ { n } \langle \phi _ { k } , \tilde { X } _ { i } \rangle _ { L ^ { 2 } } \left( Y _ { i } - \sum _ { j = 1 } ^ { n _ { m } } v _ { j } \langle \phi _ { j } , \tilde { X } _ { i } \rangle _ { L ^ { 2 } } \right) } \\ & { \quad \quad = - \frac { 2 } { n } \sum _ { i = 1 } ^ { n } Y _ { i } \left( \ \sum _ { l = 1 } ^ { D _ { N _ { m , p } } } \tilde { x } _ { i , l } \langle \phi _ { k } , \phi _ { l } \rangle _ { L ^ { 2 } } \right) } \\ & { \quad \quad \quad + \frac { 2 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n _ { m } } v _ { j } \left( \ \sum _ { l = 1 } ^ { D _ { N _ { m , p } } } \tilde { x } _ { i , l } \langle \phi _ { k } , \phi _ { l } \rangle _ { L ^ { 2 } } \right) \left( \sum _ { \nu = 1 } ^ { D _ { N _ { m , p } } } \tilde { x } _ { i , \nu } \langle \phi _ { j } , \phi _ { \nu } \rangle _ { L ^ { 2 } } \right) } \\ & { \quad \quad = - \frac { 2 } { n } \sum _ { i = 1 } ^ { n } Y _ { i } \tilde { x } _ { i , k } + \frac { 2 } { n } \sum _ { i = 1 } ^ { n } \tilde { x } _ { i , k } \sum _ { j = 1 } ^ { D _ { m } } v _ { j } \tilde { x } _ { i , j } . } \end{array}
$$

The transition from the first to second line is obtained by replacing the $\widetilde { X } _ { i } ^ { \mathrm { ~ } } \mathrm { { s } }$ by their definition given in (7) and the one from the second to third by using the fact that the $\phi _ { j } '$ s are orthonormal. Hence,

$$
\nabla F ( v ) = - 2 b + 2 \Phi _ { m } v .
$$

## B Proof of Section 3

## B.1 Proof of Proposition 3.1

Let us start first by decomposing $\mathbb { E } ( \} | \widetilde { \beta } - \beta | | _ { \widetilde { \Gamma } _ { n } } ^ { 2 } )$ into two terms :

$$
\begin{array} { r l } & { \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } ) = \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \mathbb { 1 } _ { \overline { { G } } } ) + \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \mathbb { 1 } _ { \overline { { G } } ^ { c } } ) } \\ & { \qquad = \mathbb { E } ( \| \widehat { \beta } _ { \widehat { m } } - \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \mathbb { 1 } _ { \overline { { G } } } ) + \mathbb { E } ( \| \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \mathbb { 1 } _ { \overline { { G } } ^ { c } } ) . } \end{array}
$$

In a first hand we will focus on the left term $\mathbb { E } ( \Vert \widehat { \beta } _ { \widehat { m } } - \beta \Vert _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \mathbb { 1 } _ { \overline { { G } } } )$ . Let $m \in \mathcal { M } _ { n , p }$ be fixed. Consider $\beta ^ { ( m ) }$ the orthogonal projection of $\beta$ over $S _ { m }$ with respect to the scalar product $\langle \cdot , \cdot \rangle _ { \widetilde { \Gamma } }$ . By definition of ${ \widehat { \beta } } _ { m }$ , we have that $\gamma _ { n } \big ( \widehat { \beta } _ { m } \big ) \leq \gamma _ { n } \big ( \beta ^ { ( m ) } \big )$ . Moreover, the definition of mb implies that

$$
\gamma _ { n } ( \widehat { \beta } _ { \widehat { m } } ) + \mathrm { p e n } ( \widehat { m } ) \leq \gamma _ { n } ( \widehat { \beta } _ { m } ) + \mathrm { p e n } ( m ) \leq \gamma _ { n } ( \beta ^ { ( m ) } ) + \mathrm { p e n } ( m ) .
$$

Therefore, replacing the $\gamma _ { n } \mathrm { \dot { s } }$ by their definition given in (8) we have :

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( Y _ { i } - \langle \widehat { \beta } _ { \widehat { m } } , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } ) ^ { 2 } + \mathrm { p e n } ( \widehat { m } ) \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( Y _ { i } - \langle \beta ^ { ( m ) } , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } ) ^ { 2 } + \mathrm { p e n } ( m ) .
$$

As for all $i = 1 , . . . , n , Y _ { i } = \langle \beta , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } + \langle \beta , X _ { i } - \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } + \epsilon _ { i }$ the previous equation is equivalent to the following one :

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big ( \big \langle \beta - \widehat \beta _ { \widehat m } , \widetilde X _ { i } \big \rangle _ { L ^ { 2 } } + \epsilon _ { i } + \big \langle \beta , X _ { i } - \widetilde X _ { i } \big \rangle _ { L ^ { 2 } } \Big ) ^ { 2 } + \mathrm { p e n } ( \widehat m ) } \\ & { \qquad \leq \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big ( \big \langle \beta - \beta ^ { ( m ) } , \widetilde X _ { i } \big \rangle _ { L ^ { 2 } } + \epsilon _ { i } + \big \langle \beta , X _ { i } - \widetilde X _ { i } \big \rangle _ { L ^ { 2 } } \Big ) ^ { 2 } + \mathrm { p e n } ( m ) . } \end{array}
$$

Simplifying the terms leads then to this inequality,

$$
\begin{array} { r } { \| \beta - \widehat { \beta } _ { \widehat { m } } \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \leq \| \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } + 2 \nu _ { n } ( \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } ) + 2 r _ { n } ( \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } ) + \mathtt { p e n } ( m ) - \mathtt { p e n } ( \widehat { m } ) . } \end{array}
$$

where,

$$
\nu _ { n } ( f ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \epsilon _ { i } \big < f , \widetilde { X } _ { i } \big > _ { L ^ { 2 } } ,
$$

$$
r _ { n } ( f ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bigl < f , \widetilde { X } _ { i } \bigr > _ { L ^ { 2 } } \bigl < \beta , X _ { i } - \widetilde { X } _ { i } \bigr > _ { L ^ { 2 } } .
$$

Let us focus on the $\nu _ { n }$ part. As $\nu _ { n }$ is a linear process, we have that

$$
2 \nu _ { n } ( \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } ) \leq 2 \| \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } _ { n } } \operatorname* { s u p } _ { f \in S _ { m \vee \widehat { m } } ^ { \widetilde { \Gamma } _ { n } } } \nu _ { n } ( f ) ,
$$

with $S _ { m \vee \widehat { m } } ^ { \widetilde { \Gamma } _ { n } } = \{ f \in S _ { m \vee \widehat { m } } , \| f \| _ { \widetilde { \Gamma } _ { n } } = 1 \}$ . Using the inequality $2 x y \le \textstyle \frac { 1 } { \theta } x ^ { 2 } + \theta y ^ { 2 }$ for all $x , y$ and for all $\theta > 0$ , we get that

$$
2 \nu _ { n } ( \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } ) \leq \frac { 1 } { \theta } \| \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } \| _ { { \widetilde { \Gamma } _ { n } } } ^ { 2 } + \theta \operatorname* { s u p } _ { f \in S _ { m \vee \widehat { m } } ^ { \widetilde { \Gamma } _ { n } } } ( \nu _ { n } ( f ) ) ^ { 2 } .
$$

Let us focus now on the $r _ { n }$ part. Using the Cauchy–Schwarz inequality for the scalar product in $\mathbb { R } ^ { n }$ we get that

$$
r _ { n } ( \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } ) \leq \Big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } ^ { 2 } \Big ) ^ { 1 / 2 } \Big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle \beta , X _ { i } - \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } ^ { 2 } \Big ) ^ { 1 / 2 } .
$$

Using now the Cauchy–Schwarz inequality for the scalar product $\langle \cdot , \cdot \rangle _ { L ^ { 2 } }$ , we get that

$$
r _ { n } ( \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } ) \leq \| \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } \| _ { { \widetilde { \Gamma } } _ { n } } \Big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| \beta \| _ { L ^ { 2 } } ^ { 2 } \| X _ { i } - { \widetilde { X } } _ { i } \| _ { L ^ { 2 } } ^ { 2 } \Big ) ^ { 1 / 2 } .
$$

Therefore,

$$
2 r _ { n } ( \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } ) \leq \frac { 1 } { \theta } \| \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } \| _ { { \widetilde { \Gamma } } _ { n } } ^ { 2 } + \theta \frac { \| \beta \| _ { L ^ { 2 } } ^ { 2 } } { n } \sum _ { i = 1 } ^ { n } \| X _ { i } - { \widetilde X } _ { i } \| _ { L ^ { 2 } } ^ { 2 } .
$$

These two steps lead to the following inequality,

$$
\| \widehat { \beta } _ { \widehat { m } } - \beta \| _ { { \widetilde { \Gamma } _ { n } } } ^ { 2 } \leq \| \beta ^ { ( m ) } - \beta \| _ { { \widetilde { \Gamma } _ { n } } } ^ { 2 } + \frac { 2 } { \theta } \| \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } \| _ { { \widetilde { \Gamma } _ { n } } } ^ { 2 } + \theta \operatorname* { s u p } _ { f \in S _ { m \sqrt { \widehat { m } } } ^ { \widehat { \Gamma } _ { n } } } ( \nu _ { n } ( f ) ) ^ { 2 }
$$

$$
+ \frac { \theta \| \beta \| _ { L ^ { 2 } } ^ { 2 } } { n } \sum _ { i = 1 } ^ { n } \| X _ { i } - \widetilde { X } _ { i } \| _ { L ^ { 2 } } ^ { 2 } + \mathrm { p e n } ( m ) - \mathrm { p e n } ( \widehat { m } ) .
$$

Now, as $\| \widehat { \beta } _ { \widehat { m } } - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \leq 2 \| \widehat { \beta } _ { \widehat { m } } - \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } + 2 \| \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } _ { \eta } } ^ { 2 }$ , we obtain for all $\theta > 4 \mathrm { : }$

$$
\left( 1 - \frac { 4 } { \theta } \right) \| \widehat { \beta } _ { \widehat { m } } - \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \leq \left( 1 + \frac { 4 } { \theta } \right) \| \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } + \theta \operatorname* { s u p } _ { f \in S _ { m \sqrt { \widehat { m } } } ^ { \widetilde { \Gamma } } } ( \nu _ { n } ( f ) ) ^ { 2 }
$$

$$
+ \frac { \theta \| \beta \| _ { L ^ { 2 } } ^ { 2 } } { n } \sum _ { i = 1 } ^ { n } \| X _ { i } - \widetilde { X } _ { i } \| _ { L ^ { 2 } } ^ { 2 } + \mathrm { p e n } ( m ) - \mathrm { p e n } ( \widehat { m } ) ,
$$

which leads to :

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } \big ( \| \widehat { \beta } _ { \widehat { m } } - \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \mathbb { 1 } _ { \overline { { G } } } \big ) \leq \frac { \theta + 4 } { \theta - 4 } \mathbb { E } \big ( \| \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \big ) + \frac { \theta ^ { 2 } } { \theta - 4 } \mathbb { E } \Big ( \operatorname* { s u p } _ { f \in S _ { m v \widehat { m } } ^ { \tilde { \Gamma } _ { n } } } ( \nu _ { n } ( f ) ) ^ { 2 } \Big ) } \\ & { \quad \quad \quad \quad \quad \quad + \frac { \theta ^ { 2 } } { \theta - 4 } \frac { \| \beta \| _ { L ^ { 2 } } ^ { 2 } } { n } \mathbb { E } \Big ( \displaystyle \sum _ { i = 1 } ^ { n } \| X _ { i } - \widetilde { X } _ { i } \| _ { L ^ { 2 } } ^ { 2 } \Big ) + \frac { \theta } { \theta - 4 } \mathbb { E } \big ( \mathrm { p e n } ( m ) - \mathrm { p e n } ( \widehat { m } ) \big ) . } \end{array}
$$

Now, for $x > 0$ and $Z$ a random variable we have that $Z = ( Z - x ) _ { + } + x - ( x - Z ) _ { + } \leq ( Z - x ) _ { + } + x .$ Therefore,

$$
\operatorname* { s u p } _ { f \in S _ { m \sqrt { m } } ^ { \widehat { \Gamma } _ { n } } } ( \nu _ { n } ( f ) ) ^ { 2 } \leq \Big ( \operatorname* { s u p } _ { f \in S _ { m \sqrt { m } } ^ { \widehat { \Gamma } _ { n } } } ( \nu _ { n } ( f ) ) ^ { 2 } - p ( m , \widehat { m } ) \Big ) _ { + } + p ( m , \widehat { m } ) ,
$$

with $\begin{array} { r } { p ( m , \widehat { m } ) = ( 1 + \delta ) \sigma ^ { 2 } D _ { m \vee \widehat { m } } / n = \frac { 1 } { \theta } \mathrm { p e n } ( m \vee \widehat { m } ) . \mathrm { O b v i o u s l y , f o r ~ a l l } \ x , y \geq 0 } \end{array}$ $x , y \geq 0 , \operatorname* { m a x } \left( x , y \right) \leq x + y .$ Thus $\mathrm { p e n } ( m ) - \mathrm { p e n } ( \widehat { m } ) + \theta p ( m , \widehat { m } ) \leq \mathrm { p e n } ( m ) - \mathrm { p e n } ( \widehat { m } ) + \mathrm { p e n } ( m ) + \mathrm { p e n } ( \widehat { m } ) = 2 \mathrm { p e n } ( m )$ . Therefore,

$$
\begin{array} { r l } { \mathbb { E } \big ( \| \widehat { \beta } _ { \widehat { m } } - \beta \| _ { \widetilde { \mathbf { F } } _ { n } } ^ { 2 } \mathbb { 1 } _ { \overline { { G } } } \big ) \leq \displaystyle \frac { \theta + 4 } { \theta - 4 } \mathbb { E } \big ( \| \beta - \beta ^ { ( m ) } \| _ { \widetilde { \mathbf { F } } _ { n } } ^ { 2 } \big ) } & { } \\ { \quad } & { \quad + \displaystyle \frac { \theta ^ { 2 } } { \theta - 4 } \sum _ { m ^ { \prime } \in { \mathcal { M } _ { n , p } } } \mathbb { E } \Big ( \Big [ \displaystyle \operatorname* { s u p } _ { f \in S _ { m \vee m ^ { \prime } } ^ { \widehat { \mathbf { r } } _ { n } } } ( \nu _ { n } ( f ) ) ^ { 2 } - p ( m , m ^ { \prime } ) \Big ] _ { + } \Big ) } \\ { \quad } & { \quad + \displaystyle \frac { \theta ^ { 2 } } { \theta - 4 } \frac { \| \beta \| _ { L ^ { 2 } } ^ { 2 } } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } \big ( \| X _ { i } - \widetilde { X } _ { i } \| _ { L ^ { 2 } } ^ { 2 } \big ) } \\ { \quad } & { \quad + \displaystyle \frac { 2 \theta } { \theta - 4 } \mathrm { p e n } ( m ) . } \end{array}
$$

Using the Lemma 1 of Brunel et al. [3], we find that

$$
\sum _ { m ^ { \prime } \in \mathcal { M } _ { n , p } } \mathbb { E } \Big ( [ \operatorname* { s u p } _ { f \in S _ { m \vee m ^ { \prime } } ^ { \tilde { \Gamma } _ { n } } } ( \nu _ { n } ( f ) ) ^ { 2 } - p ( m , m ^ { \prime } ) ] _ { + } \Big ) \leq \frac { C ( l , \delta ) } { n } \sigma ^ { 2 } .
$$

(The proof of this result is based on the Corollary 5.1 of Baraud [1]). To conclude this proof we need to control the term $\mathbb { E } ( \| \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \mathbb { 1 } _ { \overline { { G } } ^ { c } } )$ . Applying Cauchy–Schwarz inequality for the scalar product $\langle Z _ { 1 } , Z _ { 2 } \rangle = \mathbb { E } ( Z _ { 1 } Z _ { 2 } )$ we have that,

$$
\mathbb { E } ( \| \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \mathbb { 1 } _ { \overline { { G } } ^ { c } } ) \leq \Big [ \mathbb { E } ( \| \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 4 } ) \mathbb { P } ( \overline { { G } } ^ { c } ) \Big ] ^ { 1 / 2 } .
$$

Using the Lemma B.3 in the appendix leads to,

$$
\mathbb { P } ( \overline { { G } } ^ { c } ) \leq D _ { N _ { n , p } } \exp \Big ( { - \frac { n } { 4 \ln n ( K _ { 1 } ^ { 2 } + 1 ) ^ { 2 } } } \Big ) .
$$

This quantity is negligible for a suficiently large n as we chose $D _ { N _ { n , p } } < n / \ln ^ { 2 } n$ . Now, by definition of the norm $\| \cdot \| _ { \widetilde { \Gamma } _ { n } }$ we have,

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left( \| \beta \| _ { \tilde { \Gamma } _ { n } } ^ { 4 } \right) = \mathbb { E } \left( \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle \beta , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } ^ { 2 } \right) ^ { 2 } \right) } \\ { \displaystyle = \mathbb { E } \left( \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \langle \beta , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } ^ { 4 } \right) + \mathbb { E } \left( \frac { 2 } { n ^ { 2 } } \sum _ { 1 \leq i < i ^ { \prime } \leq n } \langle \beta , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } ^ { 2 } \langle \beta , \widetilde { X } _ { i ^ { \prime } } \rangle _ { L ^ { 2 } } ^ { 2 } \right) } \\ { \displaystyle = \frac { 1 } { n } \mathbb { E } \left( \langle \beta , \widetilde { X } _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) + \frac { n - 1 } { n } \| \beta \| _ { \tilde { \Gamma } } ^ { 4 } . } \end{array}
$$

Using the fact that ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ for all $x , y \geq 0$ and that n is a positive integer we obtain

$$
\begin{array} { r l } & { \mathbb { E } \left( \| \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 4 } \right) ^ { 1 / 2 } \leq \left( \frac { 1 } { n } \mathbb { E } \left( \langle \beta , \widetilde { X } _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) + \frac { n - 1 } { n } \| \beta \| _ { \widetilde { \Gamma } } ^ { 4 } \right) ^ { 1 / 2 } } \\ & { \quad \leq \displaystyle \frac { 1 } { \sqrt { n } } \mathbb { E } \left( \langle \beta , \widetilde { X } _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 2 } + \| \beta \| _ { \widetilde { \Gamma } } ^ { 2 } } \\ & { \quad \leq \widetilde { C } _ { \beta } . } \end{array}
$$

where

$$
\widetilde { C } _ { \beta } = \mathbb { E } \left( \langle \beta , \widetilde { X } _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 2 } + \| \beta \| _ { \widetilde { \Gamma } } ^ { 2 } + 1 .\tag{45}
$$

Hence since $\begin{array} { r } { D _ { N _ { n , p } } < \frac { n } { \ln ^ { 2 } n } } \end{array}$ we have for n large enough

$$
\begin{array} { r l } & { \left[ \mathbb { E } ( \| \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 4 } ) \mathbb { P } ( \overline { { G } } ^ { c } ) \right] ^ { 1 / 2 } \leq \widetilde { C } _ { \beta } \displaystyle \frac { n ^ { 1 / 2 } } { \ln n } \exp \left( - \frac { n } { 8 \ln n ( K _ { 1 } ^ { 2 } + 1 ) ^ { 2 } } \right) } \\ & { \qquad \leq \displaystyle \frac { \widetilde { C } _ { \beta } } { n } . } \end{array}
$$

Lemma B.9 allows us to conclude the proof of this proposition.

Lemma B.1. Let us define for all $l , j \in \mathbb { N } \backslash \{ 0 \}$

$$
\langle \phi _ { l } , \phi _ { j } \rangle _ { p } : = \frac { 1 } { p } \sum _ { h = 0 } ^ { p - 1 } \phi _ { l } ( t _ { h } ) \phi _ { j } ( t _ { h } ) ,
$$

where we recall that $\begin{array} { r } { t _ { h } = \frac { h } { p } } \end{array}$ and $h = 0 , \ldots , p - 1$ . We have

$$
\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = \left\{ \begin{array} { l l } { 1 } & { i f l = j = 1 , } \\ { 1 _ { \{ l \equiv j [ 2 p ] \} } + 1 _ { \{ l \equiv - j [ 2 p ] \} } } & { i f l ~ a n d ~ j ~ a r e ~ e v e n ~ , } \\ { 1 _ { \{ l = j [ 2 p ] \} } - 1 _ { \{ l \equiv - j + 2 [ 2 p ] \} } } & { i f l ~ a n d ~ j ~ a r e ~ o d d ~ a n d ~ s t r i c t l y ~ l a r g e r ~ t h a n ~ I ~ , } \\ { \sqrt { 2 } 1 _ { \{ l = 0 [ 2 p ] \} } } & { i f j = 1 ~ a n d l ~ i s ~ e v e n , } \\ { \sqrt { 2 } 1 _ { \{ j \equiv 0 [ 2 p ] \} } } & { i f l = 1 ~ a n d ~ j ~ i s ~ e v e n , } \\ { 0 } & { o t h e r w i s e . } \end{array} \right.
$$

Moreover, $i f l$ and $j$ are strictly lower than p then $\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = \delta _ { l j }$

Proof. We distinguish four cases.

Case 1: j and l are even, meaning that there exist two positive integers a and b such that $j = 2 a$ and $l = 2 b$

Using the fact that for all $x \in \mathbb { R } , \cos ( x ) = ( e ^ { i x } + e ^ { - i x } ) / 2 .$ , we have that

$$
\sum _ { h = 0 } ^ { p - 1 } \cos \Bigl ( { \frac { 2 \pi a h } { p } } \Bigr ) \cos \Bigl ( { \frac { 2 \pi b h } { p } } \Bigr ) = { \frac { 1 } { 4 } } \sum _ { h = 0 } ^ { p - 1 } \Bigl ( e ^ { \frac { i 2 \pi ( a - b ) h } { p } } + e ^ { - { \frac { i 2 \pi ( a - b ) h } { p } } } + e ^ { \frac { i 2 \pi ( a + b ) h } { p } } + e ^ { - { \frac { i 2 \pi ( a + b ) h } { p } } } \Bigr ) .
$$

Now, let us define $\begin{array} { r } { S ( r ) : = \sum _ { h = 0 } ^ { p - 1 } e ^ { i \frac { 2 \pi r h } { p } } } \end{array}$ for all integer r. $\mathrm { ~ I f ~ } r \equiv 0 [ p ]$ then $\begin{array} { r } { S ( r ) = \sum _ { h = 0 } ^ { p - 1 } 1 = p . } \end{array}$ Otherwise,

$$
S ( r ) = { \frac { 1 - e ^ { i 2 \pi r } } { 1 - e ^ { \frac { i 2 \pi r } { p } } } } = 0 .
$$

Moreover $S ( - r ) = \overline { { S ( r ) } } = S ( r )$ as for all integer $r , S ( r )$ is a real number. Hence,

$$
\sum _ { h = 0 } ^ { p - 1 } \cos \Bigl ( { \frac { 2 \pi a h } { p } } \Bigr ) \cos \Bigl ( { \frac { 2 \pi b h } { p } } \Bigr ) = { \frac { p } { 2 } } \Bigl [ 1 _ { \{ b \equiv a [ p ] \} } + 1 _ { \{ b \equiv - a [ p ] \} } \Bigr ] ,
$$

and

$$
\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = \mathbb { 1 } _ { \{ l \equiv j [ 2 p ] \} } + \mathbb { 1 } _ { \{ l \equiv - j [ 2 p ] \} } .
$$

Case 2: j and l are odd and strictly bigger than 1, meaning that there exist two positive integer a and b such that $j = 2 a + 1$ and $l = 2 b + 1$ . Using the fact that for all $x \in \mathbb { R } , \sin ( x ) =$ $( e ^ { i x } - e ^ { - i x } ) / ( 2 i )$ we have that

$$
\sum _ { h = 0 } ^ { p - 1 } \sin \Bigl ( \frac { 2 \pi a h } { p } \Bigr ) \sin \Bigl ( \frac { 2 \pi b h } { p } \Bigr ) = - \frac { 1 } { 4 } \sum _ { h = 0 } ^ { p - 1 } [ - e ^ { \frac { i 2 \pi ( a - b ) h } { p } } - e ^ { - \frac { i 2 \pi ( a - b ) h } { p } } + e ^ { \frac { i 2 \pi ( a + b ) h } { p } } + e ^ { - \frac { i 2 \pi ( a + b ) h } { p } } ] .
$$

Therefore,

$$
\sum _ { h = 0 } ^ { p - 1 } \sin \Bigl ( { \frac { 2 \pi a h } { p } } \Bigr ) \sin \Bigl ( { \frac { 2 \pi b h } { p } } \Bigr ) = { \frac { p } { 2 } } \bigl ( \mathbb { 1 } _ { \{ b \equiv a [ p ] \} } - \mathbb { 1 } _ { \{ b \equiv - a [ p ] \} } \bigr ) ,
$$

and

$$
\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = \mathbb { 1 } _ { \{ l \equiv j [ 2 p ] \} } - \mathbb { 1 } _ { \{ l \equiv - j + 2 [ 2 p ] \} } .
$$

Case 3: j is equal to 1. If l is also equal to 1 then $\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = 1$ . If l is odd and strictly larger than 1, then there exists a positive integer b such that $l = 2 b + 1$ and

$$
\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = \frac { \sqrt { 2 } } { p } \sum _ { h = 0 } ^ { p - 1 } \sin \Bigl ( \frac { 2 \pi b h } { p } \Bigr ) = 0 .
$$

If l is even then there exists a positive integer b such that $l = 2 b$ and

$$
\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = { \frac { \sqrt { 2 } } { p } } \sum _ { h = 0 } ^ { p - 1 } \cos \Bigl ( { \frac { 2 \pi b h } { p } } \Bigr ) = { \frac { \sqrt { 2 } } { p } } \sum _ { h = 0 } ^ { p - 1 } 1 = \sqrt { 2 }
$$

provided that $b \equiv 0 [ p ]$ , and $\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = 0$ otherwise. Consequently, we have

$$
\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = \sqrt { 2 }
$$

whenever $l \equiv 0 [ 2 p ]$ and $j = 1$ . By symmetry, the case $l = 1$ is treated analogously.

Case 4: j is odd and strictly larger than 1, and l is even. In this situation there exist two positive integers a and b such that $j = 2 a + 1$ and $l = 2 b .$ . Then,

$$
\sum _ { h = 0 } ^ { p - 1 } \sin \Bigl ( \frac { 2 \pi a h } { p } \Bigr ) \cos \Bigl ( \frac { 2 \pi b h } { p } \Bigr ) = \frac { 1 } { 2 } \sum _ { h = 0 } ^ { p - 1 } \Bigl [ \sin \Bigl ( \frac { 2 \pi ( a + b ) h } { p } \Bigr ) + \sin \Bigl ( \frac { 2 \pi ( a - b ) h } { p } \Bigr ) \Bigr ] .
$$

For any integer $r , \textstyle \sum _ { h = 0 } ^ { p - 1 }$ sin $. ( ( 2 \pi r h ) / p ) = 0$ , hence

$$
\sum _ { h = 0 } ^ { p - 1 } \sin \Bigl ( \frac { 2 \pi a h } { p } \Bigr ) \cos \Bigl ( \frac { 2 \pi b h } { p } \Bigr ) = 0 ,
$$

and $\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = 0$ . By symmetry of the roles of l and $j$ we have the same result if l is odd and strictly bigger than 1, and $j$ is even.

Lemma B.2. Let us assume that $D _ { N _ { n , p } } < p _ { ; }$ , then we have for all $j , k \in \{ 1 , \ldots , D _ { N _ { n , p } } \}$

$$
\mathbb { E } ( \widetilde { x } _ { i , j } \widetilde { x } _ { i , k } ) = \delta _ { j k } \widetilde { \lambda } _ { j } ,
$$

where

$$
\widetilde { \lambda } _ { j } = \lambda _ { j } + \overline { { { \lambda } } } _ { j } + \frac { \tau ^ { 2 } } { p } ,\tag{46}
$$

and

$$
\begin{array} { l } { { \overline { { \lambda } } _ { j } = \mathbb { 1 } _ { \{ j \ e v e n \} } \displaystyle \sum _ { q \geq 1 } \bigl ( \lambda _ { j + 2 q p } + \lambda _ { 2 q p - j } \bigr ) } } \\ { { \mathrm { } ~ + \mathbb { 1 } _ { \{ j \ o d d , j > 1 \} } \displaystyle \sum _ { q \geq 1 } \bigl ( \lambda _ { j + 2 q p } + \lambda _ { 2 q p - j + 2 } \bigr ) } } \\ { { \mathrm { } ~ + \ 2 \mathbb { 1 } _ { \{ j = 1 \} } \displaystyle \sum _ { q \geq 1 } \lambda _ { 2 q p } . } } \end{array}
$$

Proof. For all $i \in \{ 1 , \ldots , n \}$ and $j \in \{ 1 , \dots , D _ { N _ { n , p } } \}$ , let $\widetilde { x } _ { i , j } = \overline { { x } } _ { i , j } + \overline { { \eta } } _ { i , j }$ where

$$
\overline { { x } } _ { i , j } = \frac { 1 } { p } \sum _ { h = 0 } ^ { p - 1 } X _ { i } ( t _ { h } ) \phi _ { j } ( t _ { h } ) \quad \mathrm { a n d } \quad \overline { { \eta } } _ { i , j } = \frac { 1 } { p } \sum _ { h = 0 } ^ { p - 1 } \eta _ { i , h } \phi _ { j } ( t _ { h } ) .\tag{47}
$$

Then, as the $\eta _ { i , h }$ ’s are supposed to be i.i.d, independent of everything else, centered and of variance $\tau ^ { 2 }$ we have for all $j , k \in \{ 1 , \ldots , D _ { N _ { n , p } } \}$ ,

$$
\mathbb { E } ( \widetilde { x } _ { i , j } \widetilde { x } _ { i , k } ) = \mathbb { E } ( \overline { { x } } _ { i , j } \overline { { x } } _ { i , k } ) + \mathbb { E } ( \overline { { \eta } } _ { i , j } \overline { { \eta } } _ { i , k } ) .
$$

Moreover $\begin{array} { r } { \mathbb { E } ( \overline { { \eta } } _ { i , j } \overline { { \eta } } _ { i , k } ) = \frac { \tau ^ { 2 } } { p } \langle \phi _ { j } , \phi _ { k } \rangle _ { p } } \end{array}$ as the $\eta _ { i , h } ;$ s are supposed to be i.i.d, centered and of variance $\tau ^ { 2 }$ . As $j , k \le D _ { N _ { n , p } } < p$ the previous lemma leads to

$$
\mathbb { E } ( \overline { { \eta } } _ { i , j } \overline { { \eta } } _ { i , k } ) = \frac { \tau ^ { 2 } } { p } \delta _ { j k } .
$$

We can then write $\overline { { x } } _ { i , j }$ as

$$
\overline { { x } } _ { i , j } = x _ { i , j } + e _ { i , j } \quad \mathrm { w h e r e } \quad x _ { i , j } = \langle X _ { i } , \phi _ { j } \rangle _ { L ^ { 2 } }\tag{48}
$$

Therefore $\mathbb { E } ( \overline { { x } } _ { i , j } \overline { { x } } _ { i , k } ) = \mathbb { E } ( x _ { i , j } x _ { i , k } ) + \mathbb { E } ( x _ { i , j } e _ { i , k } ) + \mathbb { E } ( x _ { i , k } e _ { i , j } ) + \mathbb { E } ( e _ { i , k } e _ { i , j } ) .$

The first term can be rewritten as $\mathbb { E } ( x _ { i , j } x _ { i , k } ) = \lambda _ { j } \delta _ { j k }$ where the $\lambda _ { j } \mathrm { ^ { \circ } s }$ are the eigenvalues of $\Gamma .$ Indeed, for any $j , k \geq 1$ and for any $i \in \{ 1 , \ldots , n \}$ we have, using the fact that the $X _ { i } { } ^ { \ ' } \mathrm { s }$ are periodic and second order stationary,

$$
\begin{array} { r l } & { \mathbb { E } ( x _ { i , j } x _ { i , k } ) = \mathbb { E } ( \langle X _ { i } , \phi _ { j } \rangle _ { L ^ { 2 } } \langle X _ { i } , \phi _ { k } \rangle _ { L ^ { 2 } } ) } \\ & { \quad \quad \quad = \langle \mathbb { E } \left( \langle \phi _ { j } , X _ { i } \rangle _ { L ^ { 2 } } X _ { i } \right) , \phi _ { k } \rangle _ { L ^ { 2 } } } \\ & { \quad \quad \quad = \langle \Gamma \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } } \\ & { \quad \quad \quad = \langle \lambda _ { j } \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } } \\ & { \quad \quad \quad = \lambda _ { j } \delta _ { j k } . } \end{array}\tag{49}
$$

Moreover for all $i \in \{ 1 , \ldots , n \}$ and $\begin{array} { r } { t \in [ 0 , 1 ] , X _ { i } ( t ) = \sum _ { l \geq 1 } x _ { i , l } \phi _ { l } ( t ) } \end{array}$ , therefore $\begin{array} { r } { X _ { i } ( t _ { h } ) = \sum _ { l > 1 } x _ { i , l } \phi _ { l } ( t _ { h } ) } \end{array}$ and using Fubini we get,

$$
\begin{array} { r l } { \mathbb { E } \big ( x _ { i , j } e _ { i , k } \big ) } & { = \mathbb { E } \left( x _ { i , j } \Big ( \frac { p - 1 } { p } X _ { i } ( t _ { h } ) \phi _ { k } ( t _ { h } ) - x _ { i , k } \Big ) \right) } \\ & { = \mathbb { E } \left( x _ { i , j } \displaystyle \sum _ { i \geq 1 } x _ { i , i } \Big ( \frac { p - 1 } { p } \phi _ { i } ( t _ { h } ) \phi _ { k } ( t _ { h } ) - \delta _ { i k } \Big ) \right) } \\ & { = \displaystyle \sum _ { l \geq 1 } \mathbb { E } \big ( x _ { i , j } x _ { i , l } \Big ) \left( \frac { p - 1 } { p } \phi _ { l } ( t _ { h } ) \phi _ { k } ( t _ { h } ) - \delta _ { i k } \right) } \\ & { = \lambda _ { j } \Big ( \langle \phi _ { j } , \phi _ { k } \rangle _ { p } - \delta _ { j k } \Big ) } \\ & { = 0 , } \end{array}
$$

as $j , k \le D _ { N _ { n . v } } < p$ . Similarly, we obtain $\mathbb { E } ( x _ { i , k } e _ { i , j } ) = 0$ . To compute $\mathbb { E } ( e _ { i , j } e _ { i , k } )$ , we first express $e _ { i , j }$ for any $j \in \{ 1 , \ldots , D _ { N _ { n } } \}$ as

$$
e _ { i , j } = \sum _ { l \ge 1 } x _ { i , l } \big ( \langle \phi _ { l } , \phi _ { j } \rangle _ { p } - \delta _ { l j } \big ) .
$$

Using Lemma B.1, we have $\mathrm { I f } \ j = 1$

$$
\langle \phi _ { l } , \phi _ { 1 } \rangle _ { p } = { \left\{ \begin{array} { l l } { { \sqrt { 2 } } } & { { \mathrm { i f ~ } } l \in ( 2 p q ) _ { q \geq 1 } , } \\ { 1 } & { { \mathrm { i f ~ } } l = j , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

If $j > 1$ and $j$ is even,

$$
\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } l \in ( j + 2 q p ) _ { q \geq 0 } { \mathrm { ~ o r ~ } } l \in ( - j + 2 q p ) _ { q \geq 1 } , } \\ { 0 } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

If $j > 1$ and $j$ is odd,

$$
\langle \phi _ { l } , \phi _ { j } \rangle _ { p } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } l \in ( j + 2 q p ) _ { q \geq 0 } , } \\ { - 1 } & { { \mathrm { i f ~ } } l \in ( - j + 2 + 2 q p ) _ { q \geq 0 } , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Therefore,

$$
e _ { i , j } = \left\{ \begin{array} { l l } { \sqrt { 2 } \sum _ { q \geq 1 } x _ { i , 2 q p } } & { \mathrm { i f ~ } j = 1 , } \\ { \sum _ { q \geq 1 } [ x _ { i , 2 s + 2 q p } + x _ { i , - 2 s + 2 q p } ] } & { \mathrm { i f ~ } j = 2 s \mathrm { ~ w i t h ~ } s \in \{ 1 , \ldots , \lfloor \frac { D _ { N _ { n , p } } } { 2 } \rfloor \} , } \\ { \sum _ { q \geq 1 } [ x _ { i , 2 s + 1 + 2 q p } - x _ { i , 2 q p + 2 - ( 2 s + 1 ) } ] } & { \mathrm { i f ~ } j = 2 s + 1 \mathrm { ~ w i t h ~ } s \in \{ 1 , \ldots , \lfloor \frac { D _ { N _ { n , p } } } { 2 } \rfloor \} . } \end{array} \right.\tag{50}
$$

Given Equation (49)), we notice that, when expanding $\mathbb { E } ( e _ { i , j } e _ { i , k } )$ into sums of terms of the form $\mathbb { E } ( x _ { i , t } x _ { i , s } )$ every term vanishes unless the indices coincide. In other words, $\mathbb { E } ( e _ { i , j } e _ { i , k } ) = 0$ as soon as the index sets used in $e _ { i , j }$ and $e _ { i , k }$ are disjoint.

Let us first describe the sets of indices involved:

• For j = 1:

$$
T _ { 1 } = \{ 2 q p : q \geq 1 \} , \quad \mathrm { a l l ~ e v e n ~ m u l t i p l e s ~ o f } \ p .
$$

• For $j = 2 s$ with $1 \leq s \leq \lfloor ( p - 1 ) / 2 \rfloor$

$$
T _ { 2 s } = \{ 2 s + 2 q p , - 2 s + 2 q p : q \geq 1 \} , \quad \mathrm { a l l ~ e v e n } .
$$

• For $j = 2 s + 1$ with $1 \leq s \leq \lfloor ( p - 1 ) / 2 \rfloor$

$$
T _ { 2 s + 1 } = \{ 2 s + 1 + 2 q p , 2 q p + 1 - 2 s : q \geq 1 \} , \quad \mathrm { a l l ~ o d d } .
$$

Trivial cases . From this, two trivial vanishing cases appear: when $j$ and k don’t have the same parity and when $j = 1$ and $k > 1 ( \mathrm { o r } k = 1$ and $j > 1 )$ . As the role of j and k are symmetric, we can assume that j is odd and k is even. Then $T _ { j }$ contains only odd indices while $T _ { k }$ contains only even ones. Hence $T _ { j } \cap T _ { k } = \emptyset$ and

$$
\mathbb { E } ( e _ { i , j } e _ { i , k } ) = 0 .
$$

For the second case, let us consider $j = 1$ versus $k > 1$ . If k is odd, this falls into the previous parity case. $\mathrm { ~ I f ~ } k = 2 s$ is even, we must check $T _ { 1 } \cap T _ { 2 s } = \emptyset$ . Let us suppose that there exist $q , q ^ { \prime } \geq 1$ such that

$$
2 q p = 2 s + 2 q ^ { \prime } p \quad \mathrm { o r } \quad 2 q p = - 2 s + 2 q ^ { \prime } p .
$$

This would imply $2 s \equiv 0 [ 2 p ] , \mathrm { o r } s \equiv 0 [ p ]$ . But since $1 \leq s \leq \lfloor \frac { D _ { N _ { n , p } } } { 2 } \rfloor \leq \lfloor ( p - 1 ) / 2 \rfloor$ , this is impossible. Therefore $T _ { 1 } \cap T _ { 2 m } = \emptyset$ and

$$
\mathbb { E } ( e _ { i , 1 } e _ { i , 2 s } ) = 0 .
$$

To conclude, j and k have diferent parity, or if one of them is 1 and the other is strictly larger than 1, then $\mathbb { E } ( e _ { i , j } e _ { i , k } ) = 0$

j and k are even . Then there exist $\begin{array} { r } { a , b \in \{ 1 , \dotsc , \lfloor \frac { D _ { N _ { n , p } } } { 2 } \rfloor \} } \end{array}$ with $j = 2 a$ and $k = 2 b$ . Using the notation introduced above, the index sets are

$$
\begin{array} { r l r } { T _ { 2 a } = \{ 2 a + 2 q p , - 2 a + 2 q p : q \geq 1 \} , } & { { } } & { T _ { 2 b } = \{ 2 b + 2 q ^ { \prime } p , - 2 b + 2 q ^ { \prime } p : q ^ { \prime } \geq 1 \} . } \end{array}
$$

To determine whether $\mathbb { E } ( e _ { i , 2 a } e _ { i , 2 b } )$ can be nonzero we must look for coincidences between elements of $T _ { 2 a }$ and $T _ { 2 b }$ . Thus we consider equalities of the form (with $q , q ^ { \prime } \geq 1 )$ ):

$$
2 a + 2 q p = 2 b + 2 q ^ { \prime } p ,\tag{I}
$$

$$
2 a + 2 q p = - 2 b + 2 q ^ { \prime } p ,\tag{II}
$$

$$
- 2 a + 2 q p = 2 b + 2 q ^ { \prime } p ,\tag{III}
$$

$$
- 2 a + 2 q p = - 2 b + 2 q ^ { \prime } p .\tag{IV}
$$

• Cases (I) and (IV) : From (I) we get

$$
2 ( a - b ) = 2 p ( q ^ { \prime } - q ) \quad \Longleftrightarrow \quad a - b = p ( q ^ { \prime } - q ) .
$$

Since $| a - b | \le \lfloor \frac { D _ { N _ { n , p } } } { 2 } \rfloor \le \lfloor ( p - 1 ) / 2 \rfloor < p$ , the only multiple of p in that range is 0. Hence $a - b = 0 ;$ , or $a = b .$ If $a = b$ then (I) reduces to $2 q p = 2 q ^ { \prime } p , \mathrm { o r } q = q ^ { \prime }$ . The same reasoning applied to (IV) yields the identical conclusion: (IV) can occur only when $a = b$ and $q = q ^ { \prime }$

• Cases (II) and (III) : From (II) we obtain

$$
2 ( a + b ) = 2 p ( q ^ { \prime } - q ) \quad \Longleftrightarrow \quad a + b = p ( q ^ { \prime } - q ) .
$$

But $\begin{array} { r } { 2 \leq a + b \leq 2 \lfloor \frac { D _ { N _ { n , p } } } { 2 } \rfloor \leq 2 \lfloor ( p - 1 ) / 2 \rfloor \leq p - 1 } \end{array}$ , therefore $1 \leq a + b \leq p - 1$ and therefore $a + b$ cannot equal a nonzero multiple of p. Thus (II) is impossible. The same argument applied to $( \mathrm { I I I } ) , - 2 ( a + b ) = 2 p ( q ^ { \prime } - q )$ , shows (III) is impossible as well.

Combining the four cases we conclude that if $a \neq b$ then no equality between elements of $T _ { 2 a }$ and $T _ { 2 b }$ can occur, therefore $T _ { 2 a } \cap T _ { 2 b } = \emptyset$ and

$$
\mathbb { E } ( e _ { i , 2 a } e _ { i , 2 b } ) = 0 \qquad ( a \neq b ) .
$$

If $a = b$ the only coincidences are the trivial ones coming from matching identical indices $2 a + 2 q p$ with $2 a + 2 q p$ and $- 2 a + 2 q p$ with $- 2 a + 2 q p$ . Using Equation (49) we get

$$
\mathbb { E } ( e _ { i , 2 a } ^ { 2 } ) = \sum _ { q \geq 1 } \lambda _ { 2 a + 2 q p } + \sum _ { q \geq 1 } \lambda _ { - 2 a + 2 q p } .
$$

Hence,

$$
\mathbb { E } ( e _ { i , j } e _ { i , k } ) = \left\{ \sum _ { q \geq 1 } ^ { 0 , } \lambda _ { j + 2 q p } + \sum _ { q \geq 1 } \lambda _ { - j + 2 q p } , \quad j = k . \right.
$$

j and k are odd and strictly bigger than 1. Then there exist $\begin{array} { r } { a , b \in \{ 1 , \dotsc , \lfloor \frac { D _ { N _ { n , p } } } { 2 } \rfloor \} } \end{array}$ with $j = 2 a + 1$ and $k = 2 b + 1$ . Recalling the notation introduced above, the index sets are

$$
\begin{array} { r l } & { T _ { 2 a + 1 } = \{ 2 a + 1 + 2 q p , 2 q p + 1 - 2 a : q \geq 1 \} , } \\ & { T _ { 2 b + 1 } = \{ 2 b + 1 + 2 q ^ { \prime } p , 2 q ^ { \prime } p + 1 - 2 b : q ^ { \prime } \geq 1 \} . } \end{array}
$$

To determine whether $\mathbb { E } ( e _ { i , 2 a + 1 } e _ { i , 2 b + 1 } )$ can be nonzero we must look for coincidences between elements of $T _ { 2 a + 1 }$ and $T _ { 2 b + 1 }$ . There are four types of equalities to consider (with $q , q ^ { \prime } \geq 1 )$ :

$$
2 a + 1 + 2 q p = 2 b + 1 + 2 q ^ { \prime } p ,\tag{I}
$$

$$
2 a + 1 + 2 q p = 2 q ^ { \prime } p + 1 - 2 b ,\tag{II}
$$

$$
2 q p + 1 - 2 a = 2 b + 1 + 2 q ^ { \prime } p ,\tag{III}
$$

$$
2 q p + 1 - 2 a = 2 q ^ { \prime } p + 1 - 2 b .\tag{IV}
$$

We treat each case separately.

• Cases (I) and (IV) : From (I) we get

$$
2 ( a - b ) = 2 p ( q ^ { \prime } - q ) \quad \Longleftrightarrow \quad a - b = p ( q ^ { \prime } - q ) .
$$

Since $\begin{array} { r } { | a - b | \le \lfloor \frac { D _ { N _ { n , p } } } { 2 } \rfloor \le \lfloor ( p - 1 ) / 2 \rfloor < p , } \end{array}$ the only multiple of p in that range is 0. Hence $a - b = 0 $ , or a = b. If a = b then (I) reduces to $2 q p = 2 q ^ { \prime } p .$ , hence $q = q ^ { \prime }$ . Thus (I) can occur only when $a = b$ and $q = q ^ { \prime }$ . The same reasoning applied to (IV) yields the identical conclusion $: ~ \mathrm { ( I V ) }$ can occur only when $a = b$ and $q = q ^ { \prime }$

• Cases (II) and (III) : From (II) we obtain

$$
2 ( a + b ) = 2 p ( q ^ { \prime } - q ) \quad \Longleftrightarrow \quad a + b = p ( q ^ { \prime } - q ) .
$$

But $\begin{array} { r } { 2 \leq a + b \leq 2 \lfloor \frac { D _ { N _ { n , p } } } { 2 } \rfloor \leq 2 \lfloor ( p - 1 ) / 2 \rfloor \leq p - 1 } \end{array}$ , thus $1 \leq a + b \leq p - 1$ and therefore $a + b$ cannot equal a nonzero multiple of p. Hence (II) is impossible for any $a , b , q , q ^ { \prime }$ . The same reasoning applied to (III) yields the identical conclusion.

Combining the four cases we conclude that if $a \neq b$ then no equality between elements of $T _ { 2 a + 1 }$ and $T _ { 2 b + 1 }$ can occur, therefore $T _ { 2 a + 1 } \cap T _ { 2 b + 1 } = \emptyset$ and

$$
\mathbb { E } ( e _ { i , 2 a + 1 } e _ { i , 2 b + 1 } ) = 0 \qquad ( a \neq b )
$$

If $a = b$ the only coincidences are the trivial ones coming from matching identical indices $2 a + 1 + 2 q p$ with itself and $2 q p + 1 - 2 a$ with itself. Using Equation (49) we therefore get the expression

$$
\mathbb { E } ( e _ { i , 2 a + 1 } ^ { 2 } ) = \sum _ { q \geq 1 } \lambda _ { 2 a + 1 + 2 q p } + \sum _ { q \geq 1 } \lambda _ { 2 q p + 1 - 2 a }
$$

Hence,

$$
\mathbb { E } ( e _ { i , j } e _ { i , k } ) = \left\{ \sum _ { q \geq 1 } ^ { 0 , } \lambda _ { j + 2 q p } + \sum _ { q \geq 1 } \lambda _ { 2 q p + 2 - j } , \quad j = k . \right.
$$

$j$ and k are equal to 1 . Let us recall that $e _ { i , 1 } = { \sqrt { 2 } } \sum _ { q \geq 1 } x _ { i , 2 q p }$ and the index set is $T _ { 1 } = \{ 2 q p : q \geq 1 \}$ . To check possible coincidences between indices in the product $e _ { i , 1 } ^ { 2 }$ we look for solutions $q , q ^ { \prime } \geq 1$ of

$$
2 q p = 2 q ^ { \prime } p \qquad \Longleftrightarrow \qquad q = q ^ { \prime } .
$$

Hence we get

$$
\mathbb { E } ( e _ { i , 1 } ^ { 2 } ) = 2 \sum _ { q \geq 1 } \mathbb { E } \big ( x _ { i , 2 q p } ^ { 2 } \big ) = 2 \sum _ { q \geq 1 } \lambda _ { 2 q p } .
$$

Finally get that $\begin{array} { r } { \mathbb { E } ( \widetilde { x } _ { i , j } \widetilde { x } _ { i , k } ) = \delta _ { j k } \left( \lambda _ { j } + \overline { { \lambda } } _ { j } + \frac { \tau ^ { 2 } } { p } \right) = \delta _ { j k } \widetilde { \lambda } _ { j } \mathrm { ~ a n d ~ } \mathrm { V a r } ( \widetilde { x } _ { i , j } ) = \widetilde { \lambda } _ { j } . } \end{array}$

Lemma B.3. Let us assume that $\begin{array} { l } { \lambda _ { D _ { N _ { n , p } } } \ \ge \ \frac { 2 } { n ^ { \alpha } } } \end{array}$ and $\alpha \mathrm { ~ > ~ }$ 2a in the definition of $s _ { n }$ defined in Equation (14). We also suppose that $\begin{array} { r } { D _ { N _ { n , p } } < \operatorname* { m i n } \left( \frac { n } { \ln ^ { 2 } n } , p \right) } \end{array}$ and that (H1) and (H2) hold. Then we have

$$
\mathbb { P } ( \overline { { G } } ^ { c } ) \leq D _ { N _ { n , p } } \exp \left( - \frac { n } { 4 ( K _ { 1 } ^ { 2 } + 1 ) ^ { 2 } \ln n } \right) .
$$

Proof. The proofs for this lemma and the succeeding one were developed by Brunel and Roche [4], based on their work in Lemma 6. We only adapt them to our setting. First of all, we have

$$
\mathbb { P } ( \overline { { G } } ^ { c } ) = \mathbb { P } \left( \bigcup _ { m \in \mathcal { M } _ { n , p } } G _ { m } ^ { c } \right) \leq \sum _ { m \in \mathcal { M } _ { n , p } } \mathbb { P } ( \widehat { \lambda } _ { m } ^ { ( p ) } < s _ { n } ) ,
$$

where $\widehat { \lambda } _ { m } ^ { \left( p \right) }$ is the smallest eigenvalue of $\Phi _ { m }$ and $\begin{array} { r } { s _ { n } = \frac { 2 } { n ^ { \alpha } } ( 1 - \frac { 1 } { \sqrt { \ln n } } ) } \end{array}$ with $\alpha > 2 a$ . The last two quantities are defined in Section 2.2. By Lemma B.4 we have

$$
\mathbb { P } ( \widehat { \lambda } _ { m } ^ { ( p ) } < s _ { n } ) \leq \mathbb { P } \left( \widehat { \mu } _ { m } ^ { ( p ) } < \frac { s _ { n } } { \operatorname* { m i n } _ { 1 \leq j \leq D _ { m } } \widetilde { \lambda } _ { j } } \right) ,
$$

where $\widehat { \mu } _ { m } ^ { \left( p \right) }$ is the smallest eigenvalue of the matrix $\Psi _ { m }$ defined in Lemma B.4. Since for all $j \in \{ 1 , \dots , D _ { N _ { n , p } } \}$ we have

$$
\widetilde { \lambda } _ { j } = \lambda _ { j } + \overline { { { \lambda } } } _ { j } + \frac { \tau ^ { 2 } } { p } \ge \lambda _ { j } ,
$$

then using the fact that the eigenvalues decrease we get

$$
\operatorname* { m i n } _ { 1 \le j \le D _ { N _ { n , p } } } \widetilde { \lambda } _ { j } \ge \operatorname* { m i n } _ { 1 \le j \le D _ { N _ { n , p } } } \lambda _ { j } = \lambda _ { D _ { N _ { n , p } } } .
$$

Using the assumption that $\begin{array} { r } { \lambda _ { D _ { N _ { n , p } } } \geq \frac { 2 } { n ^ { \alpha } } } \end{array}$ and the definition of $s _ { n }$ given in (14) we obtain,

$$
\frac { s _ { n } } { \operatorname* { m i n } _ { 1 \le j \le D _ { m } } \widetilde { \lambda } _ { j } } \le 1 - \frac { 1 } { \sqrt { \ln n } } .
$$

We can then apply Lemma B.6 with $\omega = 1 - 1 / { \sqrt { \ln n } }$ and get

$$
\mathbb { P } ( \widehat { \lambda } _ { m } ^ { ( p ) } < s _ { n } ) \leq 2 \exp \left( - \frac { n } { 4 ( K _ { 1 } ^ { 2 } + 1 ) ^ { 2 } \ln n } \right) .
$$

As $N _ { n , p } \leq D _ { N _ { n , p } } / 2$

$$
\sum _ { m \in \mathcal { M } _ { n , p } } \mathbb { P } ( \widehat { \lambda } _ { m } ^ { ( p ) } < s _ { n } ) \leq D _ { N _ { n , p } } \exp \left( - \frac { n } { 4 ( K _ { 1 } ^ { 2 } + 1 ) ^ { 2 } \ln n } \right) .
$$

Combining this with the first inequality of this proof gives the desired result.

Lemma B.4. For $m \in \mathcal { M } _ { n , p } , l e t \widehat { \lambda } _ { m }$ be the smallest eigenvalue of the matrix $\Phi _ { m }$ defined in Section 2.2, and $\widehat { \mu } _ { m } ^ { \left( p \right) }$ be the smallest eigenvalue of the matrix

$$
\Psi _ { m } : = \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { \widetilde { x } _ { i , j } } { \sqrt { \widetilde { \lambda } _ { j } } } \frac { \widetilde { x } _ { i , k } } { \sqrt { \widetilde { \lambda } _ { k } } } \right) _ { 1 \leq j , k \leq D _ { m } } .\tag{51}
$$

Then,

$$
\frac { \widehat { \lambda } _ { m } ^ { ( p ) } } { \rho ( \widetilde { \Gamma } ) } \leq \widehat { \mu } _ { m } ^ { ( p ) } \leq \widehat { \lambda } _ { m } ^ { ( p ) } \left( \operatorname* { m i n } _ { 1 \leq j \leq D _ { m } } \widetilde { \lambda } _ { j } \right) ^ { - 1 } ,
$$

where $\rho ( \widetilde { \Gamma } )$ is the spectral radius of the operator $\widetilde \Gamma$

Proof. Now let us define

$$
\Lambda _ { m } : = \mathrm { d i a g } ( \sqrt { \widetilde { \lambda } _ { 1 } } , \ldots , \sqrt { \widetilde { \lambda } _ { D _ { m } } } ) .\tag{52}
$$

We have that

$$
\Phi _ { m } = \Lambda _ { m } \Psi _ { m } \Lambda _ { m } .
$$

Recall that $\widehat { \mu } _ { m } ^ { \left( p \right) }$ is the smallest eigenvalue of the matrix $\Psi _ { m }$ . We notice that $\widehat { \mu } _ { m } ^ { \left( p \right) } > 0$ directly implies that $\Psi _ { m }$ is a positive definite matrix which by definition is invertible. As a consequence, det $( \Psi _ { m } ^ { ( p ) } ) \neq 0$ . Moreover, $\Lambda _ { m }$ is invertible as it is a diagonal matrix with positive coeficient. Therefore det $( \Lambda _ { m } ^ { ( p ) } ) \neq 0$ . As

$$
\operatorname* { d e t } \left( \Phi _ { m } \right) = \operatorname* { d e t } \left( \Lambda _ { m } \right) \cdot \operatorname* { d e t } \left( \Psi _ { m } \right) \cdot \operatorname* { d e t } \left( \Lambda _ { m } \right) ,
$$

thus we get that det $( \Phi _ { m } ) \neq 0$ and that $\Phi _ { m }$ is invertible. Hence, $\widehat { \mu } _ { m } ^ { \left( p \right) } > 0$ implies that both $\Phi _ { m }$ and $\Psi _ { m }$ are invertible and we have

$$
\widehat { \mu } _ { m } ^ { ( p ) } = \rho ( \Psi _ { m } ^ { - 1 } ) ^ { - 1 } \qquad \mathrm { a n d } \qquad \widehat { \lambda } _ { m } ^ { ( p ) } = \rho ( \Phi _ { m } ^ { - 1 } ) ^ { - 1 } ,
$$

where $\rho$ is the spectral radius. We recall that for all square matrix A

$$
\| A \| _ { o p } = \operatorname* { s u p } _ { \| a \| _ { 2 } = 1 } \| A a \| _ { 2 } .
$$

If A is symmetric, $\rho ( A ) = \| A \| _ { o p }$ and we get, by using the definition of the spectral radius as well as the sub-multiplicative property of matrix norms,

$$
\begin{array} { r l } & { \rho ( \Phi _ { m } ^ { - 1 } ) = \| \Phi _ { m } ^ { - 1 } \| _ { o p } } \\ & { \qquad = \| \Lambda _ { m } ^ { - 1 } \Psi _ { m } ^ { - 1 } \Lambda _ { m } ^ { - 1 } \| _ { o p } } \\ & { \qquad \le \| \Lambda _ { m } ^ { - 1 } \| _ { o p } ^ { 2 } \| \Psi _ { m } ^ { - 1 } \| _ { o p } } \\ & { \qquad = \rho ( \Lambda _ { m } ^ { - 1 } ) ^ { 2 } \rho ( \Psi _ { m } ^ { - 1 } ) . } \end{array}
$$

Hence,

$$
\widehat { \mu } _ { m } ^ { ( p ) } \cdot \operatorname* { m i n } _ { 1 \leq j \leq D _ { m } } \widetilde { \lambda } _ { j } \leq \widehat { \lambda } _ { m } ^ { ( p ) } .
$$

As $\Psi _ { m } = \Lambda _ { m } ^ { - 1 } \Phi _ { m } \Lambda _ { m } ^ { - 1 }$ , we have, using the same reasoning that

$$
\widehat { \mu } _ { m } ^ { ( p ) - 1 } \leq \widehat { \lambda } _ { m } ^ { ( p ) - 1 } \operatorname* { m a x } _ { 1 \leq j \leq D _ { m } } \widetilde { \lambda } _ { j } \leq \rho ( \widetilde { \Gamma } ) \widehat { \lambda } _ { m } ^ { ( p ) - 1 } .
$$

The second inequality can be deduced from Lemma B.5.

Lemma B.5. Let us assume that $D _ { N _ { n , p } } < p$ . For all $k \in \{ 1 , . . . , D _ { N _ { n , p } } \}$

$$
\begin{array} { r } { \widetilde \Gamma \phi _ { k } = \widetilde \lambda _ { k } \phi _ { k } , } \end{array}
$$

where $\widetilde { \lambda } _ { k } \ : \dot { \ : } s$ defined in $( 4 6 )$ are the eigenvalues $o f \widetilde \Gamma$ associated to $\phi _ { k }$ , and for all integer $k > D _ { N _ { n , p } } .$

$$
\widetilde { \Gamma } \phi _ { k } = 0 .
$$

Proof. For all $s , t \in [ 0 , 1 ]$ we have using the definition (21) and Lemma B.2,

$$
\begin{array} { l } { \displaystyle \widetilde { K } ( s , t ) = \mathbb { E } \left( \widetilde { X } ( s ) \widetilde { X } ( t ) \right) } \\ { = \displaystyle \sum _ { j = 1 } ^ { D _ { N _ { n , p } } D _ { N _ { n , p } } } \phi _ { j } ( t ) \phi _ { k } ( s ) \mathbb { E } ( \widetilde { x } _ { j } \widetilde { x } _ { k } ) } \\ { = \displaystyle \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } \widetilde { \lambda } _ { j } \phi _ { j } ( s ) \phi _ { j } ( t ) . } \end{array}
$$

Hence, for all integer $k \in \{ 1 , . . . , D _ { N _ { n , p } } \}$

$$
\begin{array} { l } { \widetilde { \Gamma } \phi _ { k } ( s ) = \displaystyle \int _ { 0 } ^ { 1 } \widetilde { K } ( s , t ) \phi _ { k } ( t ) d t } \\ { = \displaystyle \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } \widetilde { \lambda } _ { j } \phi _ { j } ( s ) \displaystyle \int _ { 0 } ^ { 1 } \phi _ { j } ( t ) \phi _ { k } ( t ) d t } \\ { = \widetilde { \lambda } _ { k } \phi _ { k } ( s ) , } \end{array}
$$

and for all integer $k > D _ { N _ { n , p } }$

$$
\begin{array} { l } { \displaystyle \widetilde { \Gamma } \phi _ { k } ( s ) = \int _ { 0 } ^ { 1 } \widetilde { K } ( s , t ) \phi _ { k } ( t ) d t } \\ { = \displaystyle \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } \widetilde { \lambda } _ { j } \phi _ { j } ( s ) \int _ { 0 } ^ { 1 } \phi _ { j } ( t ) \phi _ { k } ( t ) d t } \\ { = 0 . } \end{array}
$$

Lemma B.6. Let ω be a real number such that $0 < \omega < 1$ . For m $\in \mathcal { M } _ { n , p } ,$ consider the smallest eigenvalue $\widehat { \mu } _ { m } ^ { \left( p \right) }$ of the matrix $\Psi _ { m }$ . Then, under Assumptions $( H 1 ) , ( H 2 )$ , we have

$$
\mathbb { P } ( \widehat { \mu } _ { m } ^ { ( p ) } < \omega ) \leq 2 \exp \left( - n \frac { ( 1 - \omega ) ^ { 2 } } { 4 ( K _ { 1 } ^ { 2 } + 1 ) ^ { 2 } } \right) .
$$

Proof. We have

$$
\begin{array} { r } { \{ \widehat { \mu } _ { m } ^ { ( p ) } < \omega \} = \{ 1 - \widehat { \mu } _ { m } ^ { ( p ) } > 1 - \omega \} . } \end{array}
$$

Since $1 - \omega > 0 ,$

$$
\begin{array} { r } { \big \{ | 1 - \widehat { \mu } _ { m } ^ { ( p ) } | > 1 - \omega \big \} = \big \{ \| \Psi _ { m } - I \| _ { o p } > 1 - \omega \big \} . } \end{array}
$$

Thus our goal is now to control $\mathbb { P } \left( \Vert \Psi _ { m } - I \Vert _ { o p } > 1 - \omega \right)$ . Let

$$
U _ { i } ^ { ( m ) } = \left( \frac { \widetilde { x } _ { i , 1 } } { \sqrt { \widetilde { \lambda } _ { 1 } } } , . . . , \frac { \widetilde { x } _ { i , D _ { m } } } { \sqrt { \widetilde { \lambda } _ { D _ { m } } } } \right) ^ { T } \in \mathbb { R } ^ { D _ { m } } , \qquad i \in \{ 1 , . . . , n \} ,\tag{53}
$$

and

$$
A _ { m } = \left( U _ { 1 } ^ { ( m ) T } , \ldots , U _ { n } ^ { ( m ) T } \right) ^ { T } \in \mathbb { R } ^ { n \times D _ { m } } .\tag{54}
$$

We can easily check that

$$
A _ { m } ^ { T } A _ { m } = \left( \sum _ { i = 1 } ^ { n } \frac { \widetilde { x } _ { i , j } } { \sqrt { \widetilde { \lambda } _ { j } } } \frac { \widetilde { x } _ { i , k } } { \sqrt { \widetilde { \lambda } _ { k } } } \right) _ { 1 \leq j , k \leq D _ { m } } ,
$$

and thus

$$
\Psi _ { m } = \frac { 1 } { n } A _ { m } ^ { T } A _ { m } .
$$

Our aim is to apply Theorem 4.6.1 of Vershynin [16]. In order to do this, we must check that the vectors $U _ { i } ^ { ( m ) } \mathrm { { ^ , } \mathrm { { s } } }$ are independent, mean-zero, isotropic and sub-gaussian. As we assumed that for all $i \in \{ 1 , \ldots , n \}$ the true curves $X _ { i }$ are independent, that for all $i \in \{ 1 , \ldots , n \} , h \in \{ 0 , \ldots , p - 1 \}$ the noises $\eta _ { i , h }$ are also independent and that for all $i \in \{ 1 , \ldots , n \} , h \in \{ 0 , \ldots , p - 1 \} \ X _ { i } \ \mathrm { a n d } \ \eta _ { i , h }$ are independent we have that the $U _ { i } ^ { ( m ) } \mathrm { { s } }$ are also independent. Moreover, as we made the assumption that the $X _ { i } { } ^ { \ ' } \mathrm { s }$ and the $\eta _ { i , h }$ ’s are centered we have that

$$
\begin{array} { l } { \mathbb { E } ( \widetilde { x } _ { i , j } ) = \mathbb { E } \left( \displaystyle \frac { 1 } { p } \sum _ { h = 0 } ^ { p - 1 } X _ { i } ( t _ { h } ) \phi _ { j } ( t _ { h } ) + \displaystyle \frac { 1 } { p } \sum _ { h = 0 } ^ { p - 1 } \eta _ { i , h } \phi _ { j } ( t _ { h } ) \right) } \\ { \displaystyle = \frac { 1 } { p } \sum _ { h = 0 } ^ { p - 1 } \mathbb { E } \left( X _ { i } ( t _ { h } ) \right) \phi _ { j } ( t _ { h } ) + \displaystyle \frac { 1 } { p } \sum _ { h = 0 } ^ { p - 1 } \mathbb { E } ( \eta _ { i , h } ) \phi _ { j } ( t _ { h } ) } \\ { \displaystyle = 0 . } \end{array}
$$

Finally we can easily check that

$$
U _ { i } ^ { ( m ) } U _ { i } ^ { ( m ) T } = \left( \frac { \widetilde { x } _ { i , j } } { \sqrt { \widetilde { \lambda } _ { j } } } \cdot \frac { \widetilde { x } _ { i , k } } { \sqrt { \widetilde { \lambda } _ { k } } } \right) _ { 1 \leq j , k \leq D _ { m } } ,
$$

and Lemma B.2 gives us that $\mathbb { E } ( \widetilde { x } _ { i , j } \widetilde { x } _ { i , k } ) = \widetilde { \lambda } _ { j } \delta _ { j k }$ . Therefore,

$$
\mathbb { E } \left( U _ { i } ^ { ( m ) } U _ { i } ^ { ( m ) T } \right) = I _ { D _ { m } } .
$$

Lemma B.7 bellow gives us that the $U _ { i } ^ { ( m ) } \mathrm { { ^ , } \mathrm { { s } } }$ are sub-gaussian. Thus all the conditions needed to apply Theorem 4.6.1 of Vershynin [16] are satisfied and we get that

$$
\begin{array} { r } { \mathbb { P } \left( \lVert \Psi _ { m } - I \rVert _ { o p } > \widetilde { K } ^ { 2 } \operatorname* { m a x } ( \widetilde { \delta } , \widetilde { \delta } ^ { 2 } ) \right) \leq 2 \exp \left( - t ^ { 2 } \right) , } \end{array}
$$

where $\begin{array} { r } { \widetilde { \delta } = C ^ { \prime } \sqrt { \frac { D _ { m } } { n } } + \frac { t } { \sqrt { n } } } \end{array}$ and $\widetilde { K } ^ { 2 } = K _ { 1 } ^ { 2 } + 1$ . Let us now find which t allows us to have $\widetilde { K } \widetilde { \delta } \le 1 - \omega$ Replacing $\widetilde { \delta }$ by its definition we get that

$$
\begin{array} { c } { { \widetilde { K } ^ { 2 } \widetilde { \delta } \le 1 - \omega \iff C ^ { \prime } \sqrt { \frac { D _ { m } } { n } } + \frac { t } { \sqrt { n } } \le \frac { 1 - \omega } { \widetilde { K } ^ { 2 } } } } \\ { { \iff t \le \sqrt { n } \left( \frac { 1 - \omega } { \widetilde { K } ^ { 2 } } - C ^ { \prime } \sqrt { \frac { D _ { m } } { n } } \right) . } } \end{array}
$$

We get that for n large enough $\begin{array} { r } { \frac { 1 } { \ln n } < \frac { 1 - \omega } { 2 C ^ { \prime } \widetilde { K } ^ { 2 } } } \end{array}$ , therefore in this setting $\begin{array} { r } { D _ { m } < \frac { n } { \ln ^ { 2 } n } < \left( \frac { 1 - \omega } { 2 C \widetilde { K } ^ { 2 } } \right) ^ { 2 } n } \end{array}$ which means that $\begin{array} { r } { C ^ { \prime } \sqrt { \frac { D _ { m } } { n } } < \frac { 1 - \omega } { 2 \widetilde { K } ^ { 2 } } } \end{array}$ . Thus we can pick

$$
t = \frac { 1 } { 2 } \left( \frac { 1 - \omega } { \widetilde { K } ^ { 2 } } \right) \sqrt { n } .
$$

Then $\begin{array} { r } { \widetilde { \delta } = C ^ { \prime } \sqrt { \frac { D _ { m } } { n } } + \frac { t } { \sqrt { n } } \le \Big ( \frac { 1 - \omega } { \widetilde { K } ^ { 2 } } \Big ) . \mathrm { ~ A s ~ } 1 - \omega < 1 \le \widetilde { K } ^ { 2 } } \end{array}$ we have that $\frac { 1 - \omega } { \widetilde K ^ { 2 } } < 1$ and $\operatorname* { m a x } ( \widetilde { \delta } , \widetilde { \delta } ^ { 2 } )$ = δe. Therefore,

$$
\mathbb { P } \left( \lVert \Psi _ { m } - I \rVert _ { o p } > 1 - \omega \right) \leq 2 \exp \left( - \frac { ( 1 - \omega ) ^ { 2 } n } { 4 \widetilde K ^ { 4 } } \right) .
$$

Lemma B.7. Under hypotheses (H1) and (H2) we have for all $i \in \{ 1 , . . . , n \}$ that $U _ { i } ^ { ( m ) }$ defined in (53) is sub-Gaussian with variance factor $K _ { 1 } ^ { 2 } + 1$

Proof. Let $v \in \mathbb { R } ^ { D _ { m } }$ such that $\| v \| _ { 2 } = 1$ . By definition of the $U _ { i } ^ { ( m ) }$ given in (53) we have that for all $t \in \mathbb { R }$

$$
\begin{array} { r } { \mathbb { E } \left( e ^ { t \langle v , U _ { i } ^ { ( m ) } \rangle _ { 2 } } \right) = \mathbb { E } \left( \exp \left( t \sum _ { j = 1 } ^ { D _ { m } } v _ { j } U _ { i , j } ^ { ( m ) } \right) \right) } \\ { = \mathbb { E } \left( \exp \left( t \sum _ { j = 1 } ^ { D _ { m } } v _ { j } \frac { \widetilde { x } _ { i , j } } { \sqrt { \widetilde { \lambda } _ { j } } } \right) \right) . } \end{array}
$$

As defined in equation (47) we have that for all $j \in \{ 1 , \dots , D _ { N _ { n , p } } \} \widetilde { x } _ { i , j } = \overline { { x } } _ { i , j } + \overline { { \eta } } _ { i , j }$ . Let us focus on E $\begin{array} { r } { \left( \exp \left( t \sum _ { j = 1 } ^ { D _ { m } } \overline { { \eta } } _ { i , j } \frac { v _ { j } } { \widetilde { \lambda } _ { j } } \right) \right) } \end{array}$ . As we assumed that the $\eta _ { i , h }$ ’s are independent, we have that for all $t \in \mathbb { R }$

$$
\begin{array} { r } { \mathbb { E } \left( \exp \left( t \sum _ { j = 1 } ^ { D _ { m } } \overline { { \eta } } _ { i , j } \frac { v _ { j } } { \widetilde { \lambda } _ { j } } \right) \right) = \mathbb { E } \left( \exp \left( \frac { t } { p } \displaystyle \sum _ { h = 0 } ^ { p - 1 } \eta _ { i , h } \displaystyle \sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } \phi _ { j } \left( t _ { h } \right) } { \widetilde { \lambda } _ { j } } \right) \right) } \\ { = \displaystyle \prod _ { h = 0 } ^ { p - 1 } \mathbb { E } \left( \exp \left( \frac { t } { p } \eta _ { i , h } \displaystyle \sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } \phi _ { j } \left( t _ { h } \right) } { \sqrt { \widetilde { \lambda } _ { j } } } \right) \right) , } \end{array}
$$

and as we assumed (H2) we have that for all $t ^ { \prime } \in \mathbb { R } , i \in \{ 1 , \dots , n \}$ and $h \in \{ 0 , \ldots , p - 1 \}$

$$
\mathbb { E } \left( \exp \left( t ^ { \prime } \eta _ { i , h } \right) \right) \leq \exp \left( \frac { \tau ^ { 2 } t ^ { \prime 2 } } { 2 } \right) .
$$

Hence,

$$
\begin{array} { r } { \mathbb { E } \left( \exp \left( t \sum _ { j = 1 } ^ { D _ { m } } \overline { { \eta } } _ { i , j } \frac { v _ { j } } { \widetilde { \lambda } _ { j } } \right) \right) \le \displaystyle \prod _ { h = 0 } ^ { p - 1 } \exp \left( \tau ^ { 2 } t ^ { 2 } \left( \sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } \phi _ { j } \left( t _ { h } \right) } { \sqrt { \widetilde { \lambda } _ { j } } } \right) ^ { 2 } \right) } \\ { = \exp \left( \displaystyle \frac { \tau ^ { 2 } t ^ { 2 } } { 2 p ^ { 2 } } \displaystyle \sum _ { h = 0 } ^ { p - 1 } \left( \sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } \phi _ { j } \left( t _ { h } \right) } { \sqrt { \widetilde { \lambda } _ { j } } } \right) ^ { 2 } \right) . } \end{array}
$$

Using Lemma B.1 we get that

$$
\begin{array} { c } { \displaystyle \frac { 1 } { p ^ { 2 } } \sum _ { h = 0 } ^ { p - 1 } \left( \sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } \phi _ { j } ( t _ { h } ) } { \sqrt { \widetilde { \lambda } _ { j } } } \right) ^ { 2 } = \displaystyle \frac { 1 } { p ^ { 2 } } \sum _ { h = 0 } ^ { p - 1 } \sum _ { j , k = 1 } ^ { D _ { m } } \frac { v _ { j } } { \sqrt { \widetilde { \lambda } _ { j } } } \frac { v _ { k } } { \sqrt { \widetilde { \lambda } _ { k } } } \phi _ { j } ( t _ { h } ) \phi _ { k } ( t _ { h } ) } \\ { = \displaystyle \frac { 1 } { p } \sum _ { j , k = 1 } ^ { D _ { m } } \frac { v _ { j } } { \sqrt { \widetilde { \lambda } _ { j } } } \frac { v _ { k } } { \sqrt { \widetilde { \lambda } _ { k } } } \langle \phi _ { j } , \phi _ { k } \rangle _ { p } } \\ { = \displaystyle \frac { 1 } { p } \sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } ^ { 2 } } { \widetilde { \lambda } _ { j } } . } \end{array}
$$

Hence,

$$
\mathbb { E } \left( \exp \left( t \sum _ { j = 1 } ^ { D _ { m } } \overline { { \eta } } _ { i , j } \frac { v _ { j } } { \widetilde { \lambda } _ { j } } \right) \right) \le \exp \left( \frac { \tau ^ { 2 } t ^ { 2 } } { 2 p } \sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } ^ { 2 } } { \widetilde { \lambda } _ { j } } \right) .
$$

As $\begin{array} { r } { \frac { \tau ^ { 2 } } { p } \leq \widetilde { \lambda } _ { j } } \end{array}$ for all $j \in \{ 1 , . . . , D _ { N _ { n , p } } \}$ we have that $\begin{array} { r } { \frac { \tau ^ { 2 } } { p \widetilde { \lambda } _ { j } } \leq 1 } \end{array}$ , therefore

$$
\mathbb { E } \left( \exp \left( t \sum _ { j = 1 } ^ { D _ { m } } \bar { \eta } _ { i , j } \frac { v _ { j } } { \widetilde { \lambda } _ { j } } \right) \right) \le \exp \left( \frac { t ^ { 2 } } { 2 } \| v \| _ { 2 } ^ { 2 } \right) = \exp \left( \frac { t ^ { 2 } } { 2 } \right) .
$$

Now let us focus on E $\begin{array} { r l } {  { ( \exp ( t \sum _ { j = 1 } ^ { D _ { m } } \overline { { x } } _ { i , j } \frac { v _ { j } } { \sqrt { \widetilde { \lambda } _ { j } } } ) ) } } \end{array}$ . Using the definition of $\overline { { x } } _ { i , j }$ given in (47) and the fact that for all $i \in \{ 1 , . . . , n \}$ and for all $\begin{array} { r l } { \dot { s } \in \lbrack 0 , 1 ] } & { { } X _ { i } ( s ) = \sum _ { j \geq 1 } x _ { i , j } \phi _ { j } ( s ) } \end{array}$ we have,

$$
\begin{array} { c } { \displaystyle \sum _ { j = 1 } ^ { D _ { m } } \overline { { x } } _ { i , j } \frac { v _ { j } } { \sqrt { \widetilde { \lambda } _ { j } } } = \displaystyle \sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } } { \sqrt { \widetilde { \lambda } _ { j } } } \left[ \frac { 1 } { j } \displaystyle \sum _ { h = 0 } ^ { p - 1 } \left( \displaystyle \sum _ { l \ge 1 } x _ { i , l } \phi _ { l } ( t _ { h } ) \right) \phi _ { j } ( t _ { h } ) \right] } \\ { \displaystyle = \sum _ { l \ge 1 } x _ { i , l } \displaystyle \sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } } { \sqrt { \widetilde { \lambda } _ { j } } } \langle \phi _ { j } , \phi _ { l } \rangle _ { p } . } \end{array}\tag{55}
$$

Let us consider $\begin{array} { r } { c _ { l } : = \sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } } { \sqrt { \widetilde \lambda _ { j } } } \langle \phi _ { j } , \phi _ { l } \rangle _ { p } . } \end{array}$ As we assumed that (H1) holds, we get that

$$
\begin{array} { r l r } {  { \mathbb { E } ( \exp ( t \sum _ { j = 1 } ^ { D _ { m } } \overline { { x } } _ { i , j } \frac { v _ { j } } { \sqrt { \widetilde { \lambda } _ { j } } } ) ) = \mathbb { E } ( \exp ( t \sum _ { l \geq 1 } x _ { i , l } c _ { l } ) ) } } \\ & { } & { \leq \exp ( \frac { t ^ { 2 } K _ { 1 } ^ { 2 } } { 2 } ( \sum _ { l \geq 1 } \lambda _ { l } c _ { l } ^ { 2 } ) ) . } \end{array}
$$

Using Fubini’s theorem we get that

$$
\begin{array} { r l r } {  { \mathbb { E } ( [ \sum _ { l \geq 1 } x _ { i , l } c _ { l } ] ^ { 2 } ) = \sum _ { l , r \geq 1 } \mathbb { E } ( x _ { i , l } x _ { i , r } ) c _ { l } c _ { r } } } \\ & { } & { = \sum _ { l \geq 1 } \lambda _ { l } c _ { l } ^ { 2 } , } \end{array}
$$

where we used the fact that $\mathbb { E } ( x _ { i , l } x _ { i , r } ) = \lambda _ { l } \delta _ { l r }$ for all $l , r \geq 1$ (see Equation (49)) to get from the last equality. On an other hand, the same computations as in (55) give that

$$
\sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } } { \sqrt { \widetilde { \lambda } _ { j } } } \overline { { x } } _ { i , j } = \sum _ { l \ge 1 } x _ { i , l } d _ { l } .
$$

Therefore,

$$
\begin{array} { l } { { \displaystyle \sum _ { l \geq 1 } \lambda _ { l } c _ { l } ^ { 2 } = \mathbb E \left( \left[ \sum _ { l \geq 1 } x _ { i , l } c _ { l } \right] ^ { 2 } \right) } } \\ { ~ } \\ { { \displaystyle \qquad = \mathbb E \left( \left[ \sum _ { j = 1 } ^ { D _ { m } } \frac { \upsilon _ { j } } { \sqrt { \lambda _ { j } } } \overline { { x } } _ { i , j } \right] ^ { 2 } \right) } } \\ { { \displaystyle \qquad = \sum _ { j , k = 1 } ^ { D _ { m } } \frac { \upsilon _ { j } } { \sqrt { \lambda _ { j } } } \frac { \upsilon _ { k } } { \sqrt { \widetilde { \lambda _ { j } } } } \mathbb E ( \overline { { x } } _ { i , j } \overline { { x } } _ { i , k } ) . } } \end{array}
$$

As for all $i \in \{ 1 , \ldots , n \}$ and $j \in \{ 1 , \dots , D _ { N _ { n , p } } \} , \overline { { x } } _ { i , j } = \widetilde { x } _ { i , j } - \overline { { \eta } } _ { i , j }$ , that the $\eta _ { i , h } ?$ s are independent of everything else and centered, we get

$$
\mathbb { E } ( \overline { { x } } _ { i , j } \overline { { x } } _ { i , k } ) = \mathbb { E } ( \widetilde { x } _ { i , j } \widetilde { x } _ { i , k } ) - \mathbb { E } ( \overline { { \eta } } _ { i , j } \overline { { \eta } } _ { i , k } ) = \delta _ { j k } \left( \widetilde { \lambda } _ { j } - \frac { \tau ^ { 2 } } { p } \right) .
$$

The last inequality can be deduced from the reasoning in the proof of Lemma B.2. We then have

$$
\begin{array} { r l } & { \displaystyle \sum _ { l \geq 1 } \lambda _ { l } c _ { l } ^ { 2 } = \sum _ { j = 1 } ^ { D _ { m } } \frac { v _ { j } ^ { 2 } } { \widetilde { \lambda } _ { j } } \left( \widetilde { \lambda } _ { j } - \frac { \tau ^ { 2 } } { p } \right) } \\ & { \qquad \leq \displaystyle \sum _ { j = 1 } ^ { D _ { m } } v _ { j } ^ { 2 } } \\ & { \qquad = \| v \| _ { 2 } ^ { 2 } } \\ & { \qquad = 1 . } \end{array}
$$

Therefore,

$$
\mathbb { E } \left( \exp { \left( t \sum _ { j = 1 } ^ { D _ { m } } \overline { { x } } _ { i , j } \frac { v _ { j } } { \sqrt { \widetilde { \lambda } _ { j } } } \right) } \right) \le \exp { \left( \frac { t ^ { 2 } K _ { 1 } ^ { 2 } } { 2 } \right) } .
$$

Using the fact that the $\eta _ { i , h } \mathrm { ^ { \circ } s }$ are independent of the $X _ { i } \mathrm { ^ { 5 , } }$ , we get that for all $i \in \{ 1 , \ldots , n \}$

$$
\begin{array} { r l } & { \mathbb { E } \left( \exp \left( t \langle v , U _ { i } ^ { ( m ) } \rangle _ { 2 } \right) \right) = \mathbb { E } \left( \exp \left( t \sum _ { j = 1 } ^ { D _ { m } } \overline { { x } } _ { i , j } \frac { v _ { j } } { \sqrt { \widetilde { \lambda } _ { j } } } \right) \right) \mathbb { E } \left( \exp \left( t \sum _ { j = 1 } ^ { D _ { m } } \overline { { \eta } } _ { i , j } \frac { v _ { j } } { \sqrt { \widetilde { \lambda } _ { j } } } \right) \right) } \\ & { \qquad \leq \exp \left( \frac { t ^ { 2 } ( K _ { 1 } ^ { 2 } + 1 ) } { 2 } \right) . } \end{array}
$$

Therefore, for all $i \in \{ 1 , \ldots , n \} \ U _ { i } ^ { ( m ) }$ is sub-gaussian with variance factor $K _ { 1 } ^ { 2 } + 1$

Lemma B.8. Let us assume that (H3) holds. Recall that for all $j \in \{ 1 , \dots , D _ { N _ { n , p } } \}$

$$
\overline { { \lambda } } _ { j } = \left\{ \begin{array} { l l } { { 2 \displaystyle \sum _ { q \geq 1 } \lambda _ { 2 q p } , } } & { { i f j = 1 , } } \\ { { \displaystyle \sum _ { q \geq 1 } \left( \lambda _ { j + 2 q p } + \lambda _ { 2 q p - j + 2 } \right) , } } & { { i f j > 1 \ i s \ o d d , } } \\ { { \displaystyle \sum _ { q \geq 1 } \left( \lambda _ { j + 2 q p } + \lambda _ { 2 q p - j } \right) , } } & { { i f j \ i s \ e v e n . } } \end{array} \right.
$$

Then there exist a positive constant $C ( \boldsymbol { a } , \boldsymbol { c } ^ { \prime } )$ such that for every $p \geq 2$ and every $j \in \{ 1 , \dots , p - 1 \}$ ,

$$
\begin{array} { r } { \overline { { \lambda } } _ { j } \leq C ( { a } , { c } ^ { \prime } ) p ^ { - 2 a } . } \end{array}
$$

Proof. We first note the following inequalities valid for every $q \geq 1$ and $1 \le j < p$

$$
\begin{array} { c } { { 2 q p \leq 2 q p + j \leq ( 2 q + 1 ) p , } } \\ { { } } \\ { { ( 2 q - 1 ) p \leq 2 q p - j \leq 2 q p , } } \\ { { } } \\ { { ( 2 q - 1 ) p + 2 \leq 2 q p - j + 2 \leq ( 2 q + 1 ) p . } } \end{array}
$$

Since $\left( \lambda _ { k } \right)$ is non-increasing, if $u \leq v$ then $\lambda _ { u } \geq \lambda _ { v }$ . We repeatedly use this together with (H3) Case 1. j even. We have

$$
\overline { { { \lambda } } } _ { j } = \sum _ { q \geq 1 } \left( \lambda _ { 2 q p + j } + \lambda _ { 2 q p - j } \right) .
$$

Since 2qp $+ j \geq 2 q p$ and 2qp $- j \geq ( 2 q - 1 ) p$

$$
\lambda _ { 2 q p + j } \leq c ^ { \prime } ( 2 q p ) ^ { - 2 a } , \qquad \lambda _ { 2 q p - j } \leq c ^ { \prime } ( ( 2 q - 1 ) p ) ^ { - 2 a } .
$$

Hence

$$
{ \overline { { \lambda } } } _ { j } \leq c ^ { \prime } p ^ { - 2 a } \sum _ { q \geq 1 } { \bigl ( } ( 2 q ) ^ { - 2 a } + ( 2 q - 1 ) ^ { - 2 a } { \bigr ) } = c ^ { \prime } \zeta ( 2 a ) p ^ { - 2 a } ,
$$

where $\zeta$ denotes the Riemann function. Thus

$$
\begin{array} { r } { \overline { { \lambda } } _ { j } \leq C ( { a } , { c } ^ { \prime } ) p ^ { - 2 a } . } \end{array}
$$

Case 2. j odd and $j > 1$ . We have

$$
\overline { { { \lambda } } } _ { j } = \sum _ { q \geq 1 } \left( \lambda _ { 2 q p + j } + \lambda _ { 2 q p - j + 2 } \right) .
$$

Since 2qp $+ j \ge$ 2qp and $2 q p - j + 2 \geq ( 2 q - 1 ) p$

$$
\begin{array} { r } { \overline { { \lambda } } _ { j } \leq c ^ { \prime } \zeta ( 2 a ) p ^ { - 2 a } . } \end{array}
$$

we conclude that $\overline { { \lambda } } _ { j } \le C ( a , c ^ { \prime } ) p ^ { - 2 a }$

Case 3. $j = 1$ . We have

$$
\overline { { { \lambda } } } _ { 1 } = 2 \sum _ { q \geq 1 } \lambda _ { 2 q p } .
$$

Using assumption (H3) we get,

$$
\overline { { \lambda } } _ { 1 } \leq 2 c ^ { \prime } \sum _ { q \geq 1 } ( 2 q p ) ^ { - 2 a } .
$$

But

$$
\sum _ { q \geq 1 } ( 2 q p ) ^ { - 2 a } = p ^ { - 2 a } 2 ^ { - 2 a } \zeta ( 2 a ) ,
$$

therefore

$$
\begin{array} { r } { \overline { { \lambda } } _ { 1 } \leq C ( { a } , { c } ^ { \prime } ) p ^ { - 2 { a } } . } \end{array}
$$

Combining the three cases yields the result.

Lemma B.9. Suppose that $D _ { N _ { n , p } } < p , \mathbb { E } \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \big ) < \infty$ and that (H3) holds. Then,

$$
\frac { \widetilde { C } _ { \beta } } { n } \leq C ( a , c ^ { \prime } , \tau ) C _ { \beta } \left( \frac { 1 } { n } + \frac { 1 } { n p } \right) + \frac { 2 \| \beta \| _ { L ^ { 2 } } ^ { 2 } \Big ( \mathbb { E } \| \widetilde { X } _ { 1 } - X _ { 1 } \| _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } } { n } ,\tag{56}
$$

where ${ \widetilde { C } } _ { \beta }$ is defined in $( 4 5 )$ and $C _ { \beta }$ in (31).

Proof. First we can rewrite $\langle \beta , \widetilde { X } _ { 1 } \rangle _ { L ^ { 2 } }$ as

$$
\langle \beta , \widetilde { X } _ { 1 } \rangle _ { L ^ { 2 } } = \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } + \langle \beta , \widetilde { X } _ { 1 } - X _ { 1 } \rangle _ { L ^ { 2 } } .
$$

By Minkowski’s inequality in $L ^ { 4 }$ we have,

$$
\begin{array} { r } { \left( \mathbb { E } \langle \beta , \widetilde { X } _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 2 } \leq \left( \left( \mathbb { E } \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 4 } + \left( \mathbb { E } | \langle \beta , \widetilde { X } _ { 1 } - X _ { 1 } \rangle _ { L ^ { 2 } } | ^ { 4 } \right) ^ { 1 / 4 } \right) ^ { 2 } . } \end{array}
$$

Cauchy–Schwarz inequality gives,

$$
| \langle \beta , \widetilde { X } _ { 1 } - X _ { 1 } \rangle _ { L ^ { 2 } } | \leq \| \beta \| _ { L ^ { 2 } } \| \widetilde { X } _ { 1 } - X _ { 1 } \| _ { L ^ { 2 } } ,
$$

Therefore,

$$
\begin{array} { r } { \big ( \mathbb { E } \vert \langle \beta , \widetilde { X } _ { 1 } - X _ { 1 } \rangle _ { L ^ { 2 } } \vert ^ { 4 } \big ) ^ { 1 / 4 } \leq \| \beta \| _ { L ^ { 2 } } \big ( \mathbb { E } \| \widetilde { X } _ { 1 } - X _ { 1 } \| _ { L ^ { 2 } } ^ { 4 } \big ) ^ { 1 / 4 } . } \end{array}
$$

Hence,

$$
\begin{array} { r } { \left( \mathbb { E } \langle \beta , \widetilde { X } _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 2 } \leq \left( \left( \mathbb { E } \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 4 } + \| \beta \| _ { L ^ { 2 } } \left( \mathbb { E } \| \widetilde { X } _ { 1 } - X _ { 1 } \| _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 4 } \right) ^ { 2 } . } \end{array}
$$

Using now $( u + v ) ^ { 2 } \leq 2 u ^ { 2 } + 2 v ^ { 2 } , u , v \in \mathbb R$ , we obtain the simpler bound

$$
\begin{array} { r } { \Big ( \mathbb { E } \langle \beta , \widetilde { X } _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } \leq 2 \Big ( \mathbb { E } \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } + 2 \| \beta \| _ { L ^ { 2 } } ^ { 2 } \Big ( \mathbb { E } \| \widetilde { X } _ { 1 } - X _ { 1 } \| _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } . } \end{array}
$$

Using the definition of $\| . \| _ { \widetilde { \Gamma } }$ given in (27), Lemma B.5 and Lemma B.8 we get

$$
\begin{array} { r l r } {  { \| \beta \| _ { \widetilde { \Gamma } } ^ { 2 } = \langle \widetilde { \Gamma } \beta , \beta \rangle _ { L ^ { 2 } } = \langle \displaystyle \sum _ { j \geq 1 } \beta _ { j } \widetilde { \Gamma } \phi _ { j } , \displaystyle \sum _ { k \geq 1 } \beta _ { k } \phi _ { k } \rangle _ { L ^ { 2 } } = \sum _ { j , k \geq 1 } \beta _ { j } \beta _ { k } \langle \widetilde { \Gamma } \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } = } } & { \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } \widetilde { \lambda } _ { j } \beta _ { j } ^ { 2 } } & \\ & { } & \\ & { } & { \leq \operatorname* { m a x } _ { 1 \leq j \leq D _ { N _ { n , p } } } \widetilde { \lambda } _ { j } \| \beta \| _ { L ^ { 2 } } ^ { 2 } \leq \| \beta \| _ { L ^ { 2 } } ^ { 2 } ( \displaystyle \frac { \tau ^ { 2 } } { p } + \displaystyle \frac { C _ { 1 } } { p ^ { 2 a } } + c ^ { \prime } ) \leq C ( a , c ^ { \prime } , \tau ) \| \beta \| _ { L ^ { 2 } } ^ { 2 } ( \displaystyle \frac { 1 } { p } + 1 ) . } & \end{array}\tag{57}
$$

Hence,

$$
\frac { \widetilde { C } _ { \beta } } { n } \leq \frac { 2 \Big ( \mathbb { E } \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } } { n } + \frac { 2 \| \beta \| _ { L ^ { 2 } } ^ { 2 } \Big ( \mathbb { E } \| \widetilde { X } _ { 1 } - X _ { 1 } \| _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } } { n } + \frac { C ( a , c ^ { \prime } , \tau ) \| \beta \| _ { L ^ { 2 } } ^ { 2 } } { n } \left( \frac { 1 } { p } + 1 \right) + \frac { 1 } { n } .
$$

Therefore we have that

$$
\frac { \widetilde { C } _ { \beta } } { n } \leq C ( a , c ^ { \prime } , \tau ) C _ { \beta } \left( \frac { 1 } { n } + \frac { 1 } { n p } \right) + \frac { 2 \| \beta \| _ { L ^ { 2 } } ^ { 2 } \Big ( \mathbb { E } \| \widetilde { X } _ { 1 } - X _ { 1 } \| _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } } { n } .
$$

## B.2 Proof of Theorem 3.1

Proof. We have that

$$
\mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } ) = \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) } } ) + \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } } ) ,\tag{58}
$$

where $\Omega ^ { ( n ) }$ is such that

$$
\Omega ^ { ( n ) } : = \bigcap _ { m \in { \mathcal { M } } _ { n } } \Omega _ { m } ,\tag{59}
$$

with

$$
\Omega _ { m } : = \left\{ \operatorname* { s u p } _ { f \in S _ { m } } \left. \frac { \lvert \lvert f \rvert \rvert _ { \widetilde { \Gamma } _ { n } } ^ { 2 } } { \lvert \lvert f \rvert \rvert _ { \widetilde { \Gamma } } ^ { 2 } } - 1 \right. \leq \widetilde { \omega } \right\} ,\tag{60}
$$

where $0 < \widetilde { \omega } < 1$ . The second term of the sum in Equation (58) is controlled by Lemma B.10. Let us focus now on the first term of the sum. Let $\beta ^ { ( m ) }$ be the projection of $\beta$ onto $S _ { m }$ with respect to the semi-norm $\langle \cdot , \cdot \rangle _ { \Gamma }$ , for all $m \in \mathcal { M } _ { n , p } .$ . Then,

$$
\begin{array} { l } { \displaystyle \| \widetilde \beta - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) } } = \| \widetilde \beta - \beta ^ { ( m ) } + \beta ^ { ( m ) } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) } } } \\ { \leq 2 \| \widetilde \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) } } + 2 \| \beta ^ { ( m ) } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) } } } \\ { \leq \displaystyle \frac { 2 } { 1 - \widetilde { \omega } } \| \widetilde \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } + 2 \| \beta ^ { ( m ) } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } } \\ { \leq \displaystyle \frac { 4 } { 1 - \widetilde { \omega } } \| \widetilde \beta - \beta \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } + \displaystyle \frac { 4 } { 1 - \widetilde { \omega } } \| \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } + 2 \| \beta ^ { ( m ) } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } , } \end{array}
$$

where we used the definition of $\Omega ^ { ( n ) }$ to go from the second to third line. Taking the expectation, using the results of Proposition 3.1 we get that

$$
\begin{array} { r l r } {  { \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } ) \le C _ { 1 } \bigg [ \operatorname* { m i n } _ { m \in \mathcal { M } _ { n , p } } \Big ( \mathbb { E } ( \| \beta ^ { ( m ) } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } ) + \mathrm { p e n } ( m ) \Big ) } } \\ & { } & { + \| \beta \| _ { L ^ { 2 } } ^ { 2 } \mathbb { E } ( \| \widetilde { X } _ { 1 } - X _ { 1 } \| _ { L ^ { 2 } } ^ { 2 } ) } \\ & { } & { + \frac { \| \beta \| _ { L ^ { 2 } } ^ { 2 } } { n } \Big ( \mathbb { E } \| \widetilde { X } _ { 1 } - X _ { 1 } \| _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } + C _ { \beta } ( \frac { 1 } { n } + \frac { 1 } { n p } ) \bigg ] , } \end{array}
$$

where $C _ { \beta } = \mathbb { E } ( \langle \beta , X _ { 1 } \rangle ^ { 4 } ) ^ { 1 / 2 } + \| \beta \| _ { L ^ { 2 } } ^ { 2 } + 1$ and $C _ { 1 } > 0$ depends on $\theta , K , l , \tau _ { l } , \delta , \sigma ^ { 2 }$

Lemma B.10. Under the assumptions that $\begin{array} { r } { D _ { N _ { n , p } } < \operatorname* { m i n } \left( \frac { n } { \ln ^ { 2 } n } , p \right) } \end{array}$ and that (H1), (H2) and (H3) hold, there exist a constant $C _ { 0 } ^ { \prime }$ depending on ρ, K, σ, a and $c ^ { \prime }$ such that

$$
\mathbb { E } \left( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega _ { n } ^ { c } } \right) \leq C _ { 0 } ^ { \prime } \left( \frac { 1 } { n } + \frac { 1 } { n p } \right) \left( \mathbb { E } ( \langle \beta , X _ { 1 } \rangle ^ { 4 } ) ^ { 1 / 2 } + \| \beta \| _ { L ^ { 2 } } ^ { 2 } + 1 \right) .
$$

Proof. The proof of this Lemma was developped by Brunel and Roche [4] in their Lemma 5. Here we just adapt it to our setting. Using the triangular inequality and the definition of $\widetilde { \beta }$ we have,

$$
\begin{array} { r l } & { \mathbb { E } \left( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } } \right) \leq 2 \mathbb { E } \left( ( \| \widetilde { \beta } \| _ { \widetilde { \Gamma } } ^ { 2 } + \| \beta \| _ { \widetilde { \Gamma } } ^ { 2 } ) \mathbb { 1 } _ { \Omega ^ { ( n ) c } } \right) } \\ & { \qquad = 2 \mathbb { E } \left( \| \widehat { \beta } _ { \widehat { m } } \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } \cap \overline { { G } } } \right) + 2 \| \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { P } ( \Omega ^ { ( n ) c } ) . } \end{array}\tag{61}
$$

Lemma B.4 and the definition of $\overline { G }$ allow us to derive the following inclusions :

$$
\overline { { G } } \subset \{ \widehat { \lambda } _ { N _ { n , p } } ^ { \left( p \right) } > s _ { n } \} \subset \left\{ \widehat { \mu } _ { N _ { n , p } } > \frac { s _ { n } } { \rho ( \widetilde { \Gamma } ) } \right\} ,\tag{62}
$$

where $\rho$ denotes the spectral radius of the operator. By using the same reasoning as detailed in the proof of Lemma B.11 bellow, we have

$$
\operatorname* { i n f } _ { f \in S _ { N _ { n , p } } } \frac { \| f \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } } { \| f \| _ { \widetilde { \Gamma } } ^ { 2 } } = \operatorname* { i n f } _ { b \neq 0 , \| b \| = 1 } b ^ { T } \Psi _ { N _ { n , p } } b .\tag{63}
$$

If we assume that $\widehat { \mu } _ { N _ { n , p } } > 0 , \Psi _ { N _ { n , p } }$ is a symmetric matrix which is positive and there exists an orthogonal matrix $U$ such that $U ^ { \tilde { T } } \Psi _ { N _ { n , p } } \breve { U }$ is a diagonal matrix whose diagonal entries are the eigenvalues of $\Psi _ { N _ { n , p } } .$ Therefore,

$$
\widehat { \mu } _ { N _ { n , p } } = \operatorname* { i n f } _ { \substack { b \neq 0 , \| b \| = 1 } } b ^ { T } U ^ { T } \Psi _ { N _ { n , p } } U b = \operatorname* { i n f } _ { \substack { c \neq 0 , \| c \| = 1 } } c ^ { T } \Psi _ { N _ { n , p } } c .\tag{64}
$$

Combining the results from Equations (62), (63) and (64), we have for all $f \in S _ { N _ { n , p } }$ and $f \neq 0$

$$
\| f \| _ { \widetilde { \Gamma } } ^ { 2 } < \frac { \rho ( \widetilde { \Gamma } ) \| f \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } } { s _ { n } } .
$$

Taking $f = \widehat { \beta } _ { \widehat { m } }$ we get

$$
\mathbb { E } \left( \| \widehat { \beta } _ { \widehat { m } } \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } \cap \overline { { G } } } \right) \leq \frac { \rho ( \widetilde { \Gamma } ) } { s _ { n } } \mathbb { E } \left( \| \widehat { \beta } _ { \widehat { m } } \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } \cap \overline { { G } } } \right) .\tag{65}
$$

As ${ \widehat { \beta } } _ { \widehat { m } }$ is a mean-square-type estimator, the vector $\left( \langle \widehat { \beta } _ { \widehat { m } } , \widetilde { X } _ { 1 } \rangle _ { L ^ { 2 } } , \dots , \langle \widehat { \beta } _ { \widehat { m } } , \widetilde { X } _ { n } \rangle _ { L ^ { 2 } } \right) ^ { T }$ can be seen as the orthogonal projection with respect to the euclidean scalar product on $\mathbb { R } ^ { n }$ of the vector $\left( Y _ { 1 } , \ldots , Y _ { n } \right) ^ { T }$ on the subspace $\left\{ ( \langle f , \widetilde { X _ { 1 } } \rangle _ { L ^ { 2 } } , \ldots , \langle f , \widetilde { X } _ { n } \rangle _ { L ^ { 2 } } ) ^ { T } , f \in S _ { \widehat { m } } \right\}$ . Therefore,

$$
\sum _ { i = 1 } ^ { n } \langle \widehat \beta _ { \widehat m } , \widetilde X _ { i } \rangle _ { L ^ { 2 } } ^ { 2 } \leq \sum _ { i = 1 } ^ { n } Y _ { i } ^ { 2 } ,
$$

and

$$
n \| \widehat { \beta } _ { \widehat { m } } \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \leq \sum _ { i = 1 } ^ { n } Y _ { i } ^ { 2 } .
$$

Replacing Y<sub>i</sub> by its definition, $Y _ { i } = \langle \beta , X _ { i } \rangle _ { L ^ { 2 } } + \epsilon _ { i }$ we get

$$
\| \widehat { \beta } _ { \widehat { m } } \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } \leq 2 \| \beta \| _ { \Gamma _ { n } } ^ { 2 } + \frac { 2 } { n } \sum _ { i = 1 } ^ { n } \epsilon _ { i } ^ { 2 } .\tag{66}
$$

Using Equations (65) and (66) we get

$$
\mathbb { E } \left( \| \widehat \beta _ { \widehat { m } } \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } \cap \overline { { G } } } \right) \leq \frac { 2 \rho ( \widetilde { \Gamma } ) } { s _ { n } } \mathbb { E } \left( \big ( \| \beta \| _ { \Gamma _ { n } } ^ { 2 } + \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \epsilon _ { i } ^ { 2 } \big ) \mathbb { 1 } _ { \Omega ^ { ( n ) c } } \right) .\tag{67}
$$

As the $\epsilon _ { i } ^ { \phantom { } } \mathrm { { s } }$ are independent of the $\widetilde { X } _ { i } ^ { \mathrm { ~ } } \mathrm { { s } }$ and that the set $\Omega _ { n } ^ { c }$ depends only on the $\widetilde { X } _ { i } ^ { \mathrm { ~ } } \mathrm { { s } }$ we have

$$
\mathbb { E } \left( \mathbb { 1 } _ { \Omega ^ { ( n ) c } } { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \epsilon _ { i } ^ { 2 } \right) = \sigma ^ { 2 } \mathbb { P } ( \Omega ^ { ( n ) c } ) .\tag{68}
$$

Cauchy–Schwarz inequality gives,

$$
\begin{array} { r } { \mathbb { E } \left( \| \beta \| _ { \Gamma _ { n } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } } \right) \leq \mathbb { E } \left( \| \beta \| _ { \Gamma _ { n } } ^ { 4 } \right) ^ { 1 / 2 } \mathbb { P } ( \Omega ^ { ( n ) c } ) ^ { 1 / 2 } . } \end{array}
$$

By definition of the norm $\Vert \cdot \Vert _ { \Gamma _ { n } }$ we have,

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left( \| \beta \| _ { \Gamma _ { n } } ^ { 4 } \right) = \mathbb { E } \left( \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle \beta , X _ { i } \rangle _ { L ^ { 2 } } ^ { 2 } \right] ^ { 2 } \right) } \\ { \displaystyle \quad = \mathbb { E } \left( \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \langle \beta , X _ { i } \rangle _ { L ^ { 2 } } ^ { 4 } \right) + \mathbb { E } \left( \frac { 2 } { n ^ { 2 } } \sum _ { 1 \leq i < i ^ { \prime } \leq n } \langle \beta , X _ { i } \rangle _ { L ^ { 2 } } ^ { 2 } \langle \beta , X _ { i ^ { \prime } } \rangle _ { L ^ { 2 } } ^ { 2 } \right) } \\ { \displaystyle \quad = \frac { 1 } { n } \mathbb { E } \left( \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) + \frac { n - 1 } { n } \| \beta \| _ { \Gamma } ^ { 4 } . } \end{array}
$$

Hence,

$$
\mathbb { E } \left( \| \beta \| _ { \Gamma _ { n } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } } \right) \leq \left( \frac { 1 } { n } \mathbb { E } \left( \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) + \frac { n - 1 } { n } \| \beta \| _ { \Gamma } ^ { 4 } \right) ^ { 1 / 2 } \mathbb { P } ( \Omega ^ { ( n ) c } ) ^ { 1 / 2 } .
$$

Using the fact that for all $x , y \geq 0 \ { \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ we have

$$
\mathbb { E } \left( \| \beta \| _ { \Gamma _ { n } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } } \right) \leq \left( \frac { 1 } { \sqrt { n } } \mathbb { E } \left( \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 2 } + \| \beta \| _ { \Gamma } ^ { 2 } \right) \mathbb { P } ( \Omega ^ { ( n ) c } ) ^ { 1 / 2 } .\tag{69}
$$

Combining Equations (67), (68) and (69),

$$
\mathbb { E } \left( \| \widehat { \beta } _ { \widehat { m } } \| _ { \tilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } \cap \overline { { G } } } \right) \leq \frac { 2 \mathbb { P } ( \Omega ^ { ( n ) c } ) ^ { 1 / 2 } } { s _ { n } } \left( \rho ( \widetilde { \Gamma } ) \left( \frac { 1 } { \sqrt { n } } \mathbb { E } \left( \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 2 } + \| \beta \| _ { \Gamma } ^ { 2 } + \sigma ^ { 2 } \right) \right) .
$$

As $\mathbb { P } ( \Omega ^ { ( n ) c } ) \leq \mathbb { P } ( \Omega ^ { ( n ) c } ) ^ { 1 / 2 }$ and $s _ { n } \leq 2$ we get, combining the previous result and Equation (61) :

$$
\mathbb { E } \left( \Vert \widetilde { \beta } - \beta \Vert _ { \Gamma } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } } \right) \leq \frac { 4 \mathbb { P } ( \Omega ^ { ( n ) c } ) ^ { 1 / 2 } } { s _ { n } } \left( \rho ( \widetilde { \Gamma } ) \left( \frac { 1 } { \sqrt { n } } \mathbb { E } \left( \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 2 } + \Vert \beta \Vert _ { \Gamma } ^ { 2 } + \sigma ^ { 2 } \right) + \Vert \beta \Vert _ { \Gamma } ^ { 2 } \right) .\tag{70}
$$

Using Lemma B.11 and the assumption that $D _ { N _ { n , p } } < n / \ln ^ { 2 } \prime$ we have

$$
\begin{array} { r l } & { \frac { \mathbb { P } ( \Omega ^ { ( n ) c } ) ^ { 1 / 2 } } { s _ { n } } \leq \frac { 1 / 2 } { 1 - 1 / \sqrt { \ln n } } ( \ln n ) ^ { - 1 } n ^ { \alpha + 1 / 2 } \exp \left( - \frac { \widetilde { \omega } ^ { 2 } n } { 8 ( K _ { 1 } ^ { 2 } + 1 ) ^ { 2 } } \right) } \\ & { \qquad \leq C \exp \left( \left( \alpha + \frac { 1 } { 2 } \right) \ln n - C ^ { \prime } n \right) } \\ & { \qquad \leq \frac { C ^ { \prime \prime } } { n } , } \end{array}\tag{71}
$$

with $C , C ^ { \prime } , C ^ { \prime \prime }$ depending on $K _ { 1 }$ . Now, using Lemma B.5 and Lemma B.8 we have that

$$
\rho ( \widetilde \Gamma ) = \operatorname* { m a x } _ { 1 \leq j \leq D _ { N _ { n , p } } } \widetilde \lambda _ { j } = \frac { \tau ^ { 2 } } { p } + \frac { C _ { 1 } } { p ^ { 2 a } } + c ^ { \prime } \leq C _ { 2 } \left( \frac 1 p + 1 \right) ,
$$

where $C _ { 1 } , C _ { 2 } > 0$ depends on $c ^ { \prime } , \ a$ and $\tau \ ( \mathrm { f o r } \ C _ { 2 } )$ . We recall that here $\rho$ denotes the spectral radius. Let’s now define the following quantity :

$$
A _ { n } : = \frac { 1 } { \sqrt { n } } \mathbb { E } \left( \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 2 } + \| \beta \| _ { \Gamma } ^ { 2 } + \sigma ^ { 2 } .
$$

Then using Equation (70) and (71) we get,

$$
\mathbb { E } \left( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } } \right) \leq C _ { 3 } \left[ \frac { 1 } { n } A _ { n } + \frac { 1 } { n p } A _ { n } + \frac { 1 } { n } \| \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \right] ,
$$

where $C _ { 3 }$ depends on $K _ { 1 } , a$ and $c ^ { \prime } .$ Using the definition of the λe<sub>j</sub>’s and the fact that the $\lambda _ { j } \mathrm { ^ { \circ } s }$ decrease in a polynomial way (see (H3)), we get using the same computations as used in (57)

$$
\begin{array} { r l r } {  { \| \beta \| _ { \widetilde { \Gamma } } ^ { 2 } = \sum _ { j = 1 } ^ { D _ { N _ { n } , p } } \widetilde { \lambda } _ { j } \beta _ { j } ^ { 2 } } } \\ & { } & { \le _ { 1 \le j \le D _ { N _ { n , p } } } \widetilde { \lambda } _ { j } \| \beta \| _ { L ^ { 2 } } ^ { 2 } } \\ & { } & { \le \| \beta \| _ { L ^ { 2 } } ^ { 2 } ( \frac { \tau ^ { 2 } } { p } + \frac { C _ { 1 } } { p ^ { 2 a } } + c ^ { \prime } ) } \\ & { } & { \le C _ { 4 } \| \beta \| _ { L ^ { 2 } } ^ { 2 } ( \frac { 1 } { p } + 1 ) . } \end{array}
$$

Hence,

$$
\mathbb { E } \left( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } } \right) \leq C _ { 5 } \left[ \frac { 1 } { n } A _ { n } + \frac { 1 } { n p } A _ { n } + \frac { 1 } { n } \| \beta \| _ { L ^ { 2 } } ^ { 2 } + \frac { 1 } { n p } \| \beta \| _ { L ^ { 2 } } ^ { 2 } \right] ,
$$

where $C _ { 5 } > 0$ depends on $\widetilde { \omega } , K _ { 1 } , a$ and $c ^ { \prime } .$ . Therefore we obtain

$$
\mathbb { E } \left( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } } \right) \leq C _ { 6 } \left( \frac { 1 } { n } + \frac { 1 } { n p } \right) \left( \mathbb { E } \left( \langle \beta , X _ { 1 } \rangle _ { L ^ { 2 } } ^ { 4 } \right) ^ { 1 / 2 } + \| \beta \| _ { \Gamma } ^ { 2 } + \| \beta \| _ { L ^ { 2 } } ^ { 2 } + 1 \right) ,
$$

where $C _ { 6 } > 0$ is a constant which depends on $K _ { 1 } , a$ and $c ^ { \prime } .$ . Finally, as

$$
\begin{array} { r l r } {  { \| \beta \| _ { \Gamma } ^ { 2 } = \langle \Gamma \beta , \beta \rangle _ { L ^ { 2 } } = \langle \sum _ { j \geq 1 } \beta _ { j } \Gamma \phi _ { j } , \displaystyle \sum _ { k \geq 1 } \beta _ { k } \phi _ { k } \rangle _ { L ^ { 2 } } = \displaystyle \sum _ { j , k \geq 1 } \beta _ { j } \beta _ { k } \langle \Gamma \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } = \sum _ { j \geq 1 } \lambda _ { j } \beta _ { j } ^ { 2 } } } \\ & { } & { \leq \operatorname* { m a x } _ { j \geq 1 } \lambda _ { j } \| \beta \| _ { L ^ { 2 } } ^ { 2 } \leq c ^ { \prime } \| \beta \| _ { L ^ { 2 } } ^ { 2 } , ~ } \end{array}\tag{72}
$$

we get that

$$
\mathbb { E } \left( \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \mathbb { 1 } _ { \Omega ^ { ( n ) c } } \right) \leq C _ { 0 } ^ { \prime } \left( \frac { 1 } { n } + \frac { 1 } { n p } \right) \left( \mathbb { E } ( \langle \beta , X _ { 1 } \rangle ^ { 4 } ) ^ { 1 / 2 } + \| \beta \| _ { L ^ { 2 } } ^ { 2 } + 1 \right) ,
$$

where $C _ { 0 } ^ { \prime } > 0$ depends on $K _ { 1 } , \sigma ,$ a and $c ^ { \prime }$

□

Lemma B.11. Let us assume that (H1), (H2) hold and that $D _ { N _ { n , p } } < p$ . Then,

$$
\mathbb { P } \left( \Omega ^ { ( n ) c } \right) \leq D _ { N _ { n , p } } \exp \left( - \frac { n \widetilde { \omega } ^ { 2 } } { 4 ( K _ { 1 } ^ { 2 } + 1 ) ^ { 2 } } \right) ,
$$

where $\Omega ^ { ( n ) c }$ is defined in (58).

Proof. Let $f \in S _ { m }$ . We can write f as $\begin{array} { r } { f = \sum _ { j = 1 } ^ { D _ { m } } f _ { j } \phi _ { j } } \end{array}$ . Therefore, by using the definition (26), we have

$$
\begin{array} { l } { \displaystyle \| f \| _ { \mathbf { F } _ { n } } ^ { 2 } = \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \zeta f , \widetilde { X } _ { i } \rangle _ { L ^ { 2 } } ^ { 2 } } \\ { \displaystyle = \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { D _ { m } } \sum _ { k = 1 } ^ { D _ { \mathbf { v } _ { k , v } } } \widetilde { X } _ { i , k } \phi _ { k } \rangle _ { I , \lambda } ^ { 2 } } \\ { \displaystyle = \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \Big ( \displaystyle \sum _ { j = 1 } ^ { D _ { m } } f _ { j } \widetilde { X } _ { i , j } \Big ) ^ { 2 } } \\ { \displaystyle = \sum _ { j = 1 } ^ { D _ { m } } f _ { j } f _ { k } \Big ( \displaystyle \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \widetilde { X } _ { i , j } \widetilde { X } _ { i , k } \Big ) } \\ { \displaystyle = \sum _ { j = k = 1 } ^ { D _ { m } } f _ { j } f _ { k } \Big ( \displaystyle \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \widetilde { X } _ { i , j } \widetilde { X } _ { i , k } \Big ) } \\ { \displaystyle = h ^ { T } \widetilde { \Phi } ^ { ( p ) } h , } \end{array}
$$

where $\boldsymbol { h } \ : = \ : ( f _ { 1 } , \ldots , f _ { D _ { m } } ) ^ { T }$ , and $\Phi _ { m } ^ { ( p ) }$ was defined in (11). We used the fact that the $\phi _ { j } \mathrm { ^ { \circ } s }$ are orthonormal to get the third line. Moreover, using the definition (27) we get

$$
\begin{array} { r l } { \| f \| _ { \Gamma } ^ { 2 } = \mathbb { E } \bigg ( \langle \widetilde { X } , f \rangle _ { L ^ { 2 } } ^ { 2 } \bigg ) } \\ & { \quad = \mathbb { E } \bigg ( \langle \sum _ { \kappa = 1 } ^ { D _ { \operatorname* { m } , r } } \widetilde { x } _ { k } \phi _ { k } \sum _ { j = 1 } ^ { D _ { \operatorname* { m } } } f _ { j } \phi _ { j } \rangle _ { L ^ { 2 } } ^ { 2 } \bigg ) } \\ & { \quad = \displaystyle \sum _ { j , k = 1 } ^ { D _ { \operatorname* { m } } } f _ { j } f _ { k } \mathbb { E } ( \widetilde { x } _ { k } \widetilde { x } _ { j } ) } \\ & { \quad = \displaystyle \sum _ { j = 1 } ^ { D _ { \operatorname* { m } } } f _ { j } ^ { 2 } \widetilde { X } _ { j } } \\ & { \quad = \displaystyle \sum _ { j = 1 } ^ { D _ { \operatorname* { m } } } \sum _ { r _ { j } , k _ { m } , h , \atop j } h , } \end{array}
$$

where $\Lambda _ { m }$ was defined by (52). We get the last line by using Lemma B.2. Using the matrix $\Psi _ { m }$ defined in (51), we get the following equation,

$$
\| \ b { f } \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } = h ^ { T } \Phi _ { m } h = h ^ { T } \Lambda _ { m } ^ { T } \Psi _ { m } \Lambda _ { m } h = b ^ { T } \Psi _ { m } \boldsymbol { b } ,
$$

with $b = \Lambda _ { m } h$ , and

$$
\| \boldsymbol { f } \| _ { \widetilde { \Gamma } } ^ { 2 } = h ^ { T } \Lambda _ { m } ^ { T } \Lambda _ { m } h = \boldsymbol { b } ^ { T } \boldsymbol { b } .
$$

Therefore,

$$
\frac { \| f \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } } { \| f \| _ { \widetilde { \Gamma } } ^ { 2 } } = \frac { b ^ { T } \Psi _ { m } b } { b ^ { T } b } ,\tag{73}
$$

and

$$
\operatorname* { s u p } _ { f \in S _ { m } } \left. \frac { \| f \| _ { \widetilde { \Gamma } _ { n } } ^ { 2 } } { \| f \| _ { \widetilde { \Gamma } } ^ { 2 } } - 1 \right. = \operatorname* { s u p } _ { b \neq 0 } \left. \frac { b ^ { T } ( \Psi _ { m } - I ) b } { b ^ { T } b } \right. = \| \Psi _ { m } - I \| _ { o p } ,
$$

as $\Psi _ { m }$ is symmetric. Therefore we can rewrite $\Omega _ { m }$ as

$$
\Omega _ { m } = \left\{ \| \Psi _ { m } - I \| _ { o p } \leq \widetilde { \omega } \right\} .
$$

As discussed in the proof of Lemma B.6 we can apply after some computations Theorem 4.6.1 of Vershynin [16] to get

$$
\mathbb { P } \left( \Omega _ { m } ^ { c } \right) = \mathbb { P } ( \left. \Psi _ { m } - I \right. _ { o p } > \widetilde { \omega } ) \le 2 \exp { \left( - \frac { n \widetilde { w } ^ { 2 } } { 4 ( K _ { 1 } ^ { 2 } + 1 ) ^ { 2 } } \right) } ,
$$

and using the fact that $N _ { n , p } \leq D _ { N _ { n , p } } / 2$ we get that

$$
\mathbb { P } \left( \Omega ^ { ( n ) c } \right) = \sum _ { m \in \mathcal { M } _ { n } } \mathbb { P } ( \Omega _ { m } ^ { c } ) \leq D _ { N _ { n , p } } \exp \left( - \frac { n \widetilde { \omega } ^ { 2 } } { 4 ( K _ { 1 } ^ { 2 } + 1 ) ^ { 2 } } \right) .
$$

## B.3 Proof of Theorem 3.2

Proof. Lemma B.5 gives us that the eigenvalues of $\widetilde \Gamma$ in $S _ { N _ { n , p } }$ are the $\widetilde { \lambda } _ { j } ^ { \mathrm { ~ } } \mathrm { { s } }$ which are defined in (46). Now for all $j \in \{ 1 , \dots , D _ { N _ { n , p } } \}$ we have that $\lambda _ { j } \leq \widetilde { \lambda } _ { j }$ , therefore for all $f \in S _ { N _ { n , p } } $

$$
\| f \| _ { \Gamma } \leq \| f \| _ { \widetilde { \Gamma } } .
$$

For all $m \in \mathcal { M } _ { n , p }$ and considering $\beta ^ { ( m ) }$ the projection of $\beta$ on $S _ { m }$ we get that,

$$
\begin{array} { r l } & { \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } = \| \widetilde { \beta } - \beta ^ { ( m ) } + \beta ^ { ( m ) } - \beta \| _ { \Gamma } ^ { 2 } } \\ & { \qquad \leq 2 \| \widetilde { \beta } - \beta ^ { ( m ) } \| _ { \Gamma } ^ { 2 } + 2 \| \beta ^ { ( m ) } - \beta \| _ { \Gamma } ^ { 2 } } \\ & { \qquad \leq 2 \| \widetilde { \beta } - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } } ^ { 2 } + 2 \| \beta ^ { ( m ) } - \beta \| _ { \Gamma } ^ { 2 } } \\ & { \qquad \leq 4 \| \widetilde { \beta } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } + 4 \| \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } } ^ { 2 } + 2 \| \beta ^ { ( m ) } - \beta \| _ { \Gamma } ^ { 2 } . } \end{array}
$$

Taking the expectation in the equation just above and using Theorem 3.1, we get that

$$
\begin{array} { r l } & { \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } ) \le C _ { 2 } \bigg [ \underset { m \in \mathcal { M } _ { n , p } } { \operatorname* { m i n } } \Big ( \| \beta ^ { ( m ) } - \beta \| _ { \Gamma } ^ { 2 } + \| \beta ^ { ( m ) } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } + p e n ( m ) \Big ) } \\ & { \quad \quad \quad \quad + \| \beta \| _ { L ^ { 2 } } ^ { 2 } \mathbb { E } ( \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 2 } ) } \\ & { \quad \quad \quad \quad + \frac { \| \beta \| _ { L ^ { 2 } } ^ { 2 } } { n } \Big ( \mathbb { E } \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 4 } \Big ) ^ { 1 / 2 } + C _ { \beta } \bigg ( \frac { 1 } { n } + \frac { 1 } { n p } \bigg ) \bigg ] , } \end{array}
$$

To derive Equation (34) we combine this result with Lemma A.3 of Tsybakov [15].

Lemma B.12. Let us suppose that (H3) hold, $D _ { N _ { n , p } } < p$ and that $\beta \in W ^ { p e r } ( k , L )$ . Then for all $m \in \mathcal { M } _ { n , p }$

$$
\| \beta ^ { ( m ) } - \beta \| _ { \widetilde { \Gamma } } ^ { 2 } \leq C _ { 2 } ^ { \prime \prime } \Big ( \| \beta ^ { ( m ) } - \beta \| _ { \Gamma } ^ { 2 } + \frac { 1 } { p } \Big ) ,
$$

where $C _ { 2 } ^ { \prime \prime }$ is a positive constant which depends on $L , k , c ^ { \prime } , a , \tau .$

Proof. We have, denoting as $\beta _ { j }$ the coeficients of $\beta$ in the Fourier basis,

$$
\begin{array} { c } { \displaystyle \| \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } } ^ { 2 } = \langle \widetilde { \Gamma } ( \beta - \beta ^ { ( m ) } ) , \beta - \beta ^ { ( m ) } \rangle _ { L ^ { 2 } } } \\ { = \displaystyle \sum _ { j = D _ { m } + 1 } ^ { D _ { N _ { n , p } } } \widetilde { \lambda } _ { j } \beta _ { j } ^ { 2 } , } \end{array}
$$

since Lemma B.5 gives us that the eigenvalues of $\widetilde \Gamma$ associated with the elements of the Fourier basis are the $\widetilde { \lambda } _ { j }$ for $j \in \{ 1 , . . . , D _ { N _ { n , p } } \}$ , and are 0 for $j > D _ { N _ { n , p } }$ . Using now the definition of the $\widetilde { \lambda } _ { j } ^ { \mathrm { ~ } } \mathrm { { s } }$ defined in (46) and our assumption that $\beta \in W ^ { \mathrm { p e r } } ( k , L )$ for a certain positive integer k and a positive real number L we get that

$$
\begin{array} { r l } {  { \mathbb { E } ( \| \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } } ^ { 2 } ) = \sum _ { j = D _ { m } + 1 } ^ { D _ { N _ { n , p } } } \lambda _ { j } \beta _ { j } ^ { 2 } + \sum _ { j = D _ { m } + 1 } ^ { D _ { N _ { n , p } } } \overline { { \lambda } } _ { j } \beta _ { j } ^ { 2 } + \sum _ { j = D _ { m } + 1 } ^ { D _ { N _ { n , p } } } \overline { { \boldsymbol { p } } } ^ { 2 } \beta _ { j } ^ { 2 } } } \\ & { \leq \sum _ { j > D _ { m } } ^ { \sum } \lambda _ { j } \beta _ { j } ^ { 2 } + \operatorname* { s u p } _ { D _ { m } < l \leq D _ { N _ { n , p } } } \overline { { \lambda } } _ { l } \sum _ { j > D _ { m } } \beta _ { j } ^ { 2 } + \sum _ { j > D _ { m } } \overline { { \boldsymbol { p } } } ^ { 2 } \beta _ { j } ^ { 2 } } \\ & { \leq C _ { 3 } ^ { \prime } \big ( \| \beta - \beta ^ { ( m ) } \| _ { \Gamma } ^ { 2 } + D _ { m } ^ { - 2 k } { p ^ { - 2 a } } + \frac { \tau ^ { 2 } } { p } D _ { m } ^ { - 2 k } \big ) , } \end{array}\tag{74}
$$

where $C _ { 3 } ^ { \prime } > 0$ depends on $a , c ^ { \prime } , L , k .$ . Let us give more details on how we derived the last inequality. Let us focus first on the term $\begin{array} { r } { \operatorname* { s u p } _ { D _ { m } < l \leq D _ { N _ { n , p } } } \overline { { \lambda } } _ { l } \sum _ { j > D _ { m } } \beta _ { j } ^ { 2 } } \end{array}$ . Lemma B.8 allows us to deduce that $\overline { { \lambda } } _ { j } = \mathcal { O } ( p ^ { - 2 a } )$ for all $j = 1 , . . . , D _ { N _ { n } }$ . Hence by Lemma A.3 of Tsybakov [15] we have

$$
\operatorname* { s u p } _ { D _ { m } < l \leq D _ { N _ { n , p } } } \bar { \lambda } _ { l } \sum _ { j > D _ { m } } \beta _ { j } ^ { 2 } \leq C ^ { \prime } p ^ { - 2 a } D _ { m } ^ { - 2 k } ,
$$

where $C ^ { \prime }$ is a positive constant which depends on $c ^ { \prime } , a , L , k$ . Using Lemma A.3 of Tsybakov [15] again we get that $\begin{array} { r } { \frac { \tau ^ { 2 } } { p } \sum _ { j > D _ { m } } \beta _ { j } ^ { 2 } \le C ^ { \prime \prime } \frac { \tau ^ { 2 } } { p } \bar { D } _ { m } ^ { - 2 k } } \end{array}$ where $C ^ { \prime \prime }$ is a positive constant which depends on $L , k$ . Hence, for all $\mathbf { \bar { \rho } } _ { m } \in \mathcal { M } _ { n , p } ,$

$$
\mathbb { E } ( \| \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } } ^ { 2 } ) \le C _ { 3 } ^ { \prime } \big ( \| \beta - \beta ^ { ( m ) } \| _ { \Gamma } ^ { 2 } + \operatorname* { s u p } _ { l \in \mathcal { M } _ { n , p } } D _ { l } ^ { - 2 k } p ^ { - 2 a } + \frac { \tau ^ { 2 } } { p } D _ { l } ^ { - 2 k } \big ) .
$$

Since for all $l \in \mathcal { M } _ { n , p } \ D _ { l } \geq 1$ we have $\begin{array} { r } { \operatorname* { s u p } _ { l \in { \mathcal { M } } _ { n , p } } D _ { l } ^ { - 2 k } p ^ { - 2 a } + \frac { \tau ^ { 2 } } { p } D _ { l } ^ { - 2 k } = \left( p ^ { - 2 a } + \frac { \tau ^ { 2 } } { p } \right) } \end{array}$ . Now using the fact that $p ^ { - 2 a } \leq p ^ { - 1 }$ as $a > 1 / 2$ , we find that

$$
\begin{array} { r } { \mathbb { E } ( \| \beta - \beta ^ { ( m ) } \| _ { \widetilde { \Gamma } } ^ { 2 } ) \le C _ { 2 } ^ { \prime \prime } \big ( \| \beta - \beta ^ { ( m ) } \| _ { \Gamma } ^ { 2 } + p ^ { - 1 } \big ) . } \end{array}
$$

□

## C Proof of Theorem 4.1

Using the result of Theorem 3.2, we get that

$$
\begin{array} { r l r } {  { \mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } ) \leq C _ { 3 } \bigg [ \operatorname* { m i n } _ { m \in \mathcal { M } _ { n , p } } \Big ( \| \beta ^ { ( m ) } - \beta \| _ { \Gamma } ^ { 2 } + p e n ( m ) \Big ) } } \\ & { } & { + \mathbb { E } ( \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 2 } ) } \\ & { } & { + \frac { 1 } { n } ( \mathbb { E } \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 4 } ) ^ { 1 / 2 } + \frac { 1 } { p } + \frac { 1 } { n } + \frac { 1 } { n p } \bigg ] . } \end{array}
$$

With the assumption that the $\lambda _ { j } { ' } \mathfrak { s }$ s decay in a polynomial way (see (H3)) and that $\beta$ is in $W ^ { \mathrm { p e r } } ( k , L )$ , we can use the result of the Theorem $2$ of Brunel and Roche [4] which gives that

$$
\| \beta - \beta ^ { ( m ) } \| _ { \Gamma } ^ { 2 } = \sum _ { j > D _ { m } } \beta _ { j } ^ { 2 } \lambda _ { j } \leq C D _ { m } ^ { - 2 a - 2 k } .
$$

Combining this reasoning with Lemmas C.1 and C.2 leads to the following equation,

$$
\mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } ) \le C \operatorname* { m i n } _ { 1 \le D _ { N _ { n , p } } < p \land \sqrt { \frac { n } { \ln ^ { 3 } n } } , \atop 1 \le D _ { m } \le D _ { N _ { n , p } } } \bigl ( f ( D _ { m } ) + g ( D _ { N _ { n , p } } ) \bigr ) ,\tag{75}
$$

where for all $x \geq 1$

$$
f ( x ) = x ^ { - 2 k - 2 a } + x \frac { \sigma ^ { 2 } } { n } \qquad \mathrm { a n d } \qquad g ( x ) = \frac { 1 } { n } + \frac { 1 } { p } + \frac { 1 } { n p } + \left( x ^ { - ( 2 a - 1 ) } + \frac { x } { p } \right) \left( \frac { 1 } { n } + 1 \right) .
$$

Let us focus on minimizing the function $g$ first. $\mathrm { A s \ g }$ is strictly convex, coercive and $C ^ { 1 } , g$ admits a unique minimum which is given by solving $g ^ { \prime } = 0$ . For all $x \ge 1$ , we have

$$
g ^ { \prime } ( x ) = \left[ ( 1 - 2 a ) x ^ { - 2 a } + { \frac { 1 } { p } } \right] \cdot \left[ { \frac { 1 } { n } } + 1 \right] .
$$

Therefore, $g ^ { \prime } ( x ) = 0 \iff x = { \bigl ( } ( 2 a - 1 ) p { \bigr ) } ^ { 1 / ( 2 a ) }$ . Taking into account the fact that $D _ { N _ { n , p } }$ must satisfy the following inequality $\begin{array} { r } { D _ { N _ { n , p } } < \operatorname* { m i n } ( p , \frac { n } { \ln ^ { 2 } n } ) } \end{array}$ , we get that

$$
D _ { N _ { n , p } } ^ { * } \approx \left\{ { p \atop { \ln ( n ) ^ { 2 } } } \right. \sp { \frac { 1 } { 2 a } } \left. { \mathrm { i f } \ p \atop { \mathrm { o t h e r w i s e } \ . } } \right. \sp { \sum } \left( { \frac { n } { \ln ( n ) ^ { 2 } } } \right) \sp { 2 a }
$$

Remark : We derive the condition $\begin{array} { r } { p \lesssim \left( \frac { n } { \ln ( n ) ^ { 2 } } \right) ^ { 2 a } } \end{array}$ since if min $\begin{array} { r } { ( p , \frac { n } { \ln ^ { 2 } n } ) = p } \end{array}$ then $p ^ { \frac { 1 } { 2 a } } \leq p \mathrm { a s } a > 1 / 2$ and $D _ { N _ { n , p } } ^ { * } \approx p ^ { 1 / ( 2 a ) }$ in this situation. Otherwise, if min $\begin{array} { r } { ( p , \frac { n } { \ln ^ { 2 } n } ) = \frac { n } { \ln ^ { 2 } n } } \end{array}$ then

$$
p ^ { 1 / ( 2 a ) } \lesssim \frac { n } { \ln ^ { 2 } n } \iff p \lesssim \left( \frac { n } { \ln ^ { 2 } n } \right) ^ { 2 a } .
$$

$\begin{array} { r } { \operatorname { I f } p \gtrsim \left( \frac { n } { \ln ^ { 2 } n } \right) ^ { 2 a } } \end{array}$ then the best value that $D _ { N _ { n , p } }$ can reach is $\frac { n } { \ln ( n ) ^ { 2 } }$

Now let’s focus on $f .$ We can easily verify that $f$ is $C ^ { 1 }$ , strictly convex and coercive therefore the function admits a unique minimum. For all $x \geq 1$ we have

$$
f ^ { \prime } ( x ) = - ( 2 k + 2 a ) x ^ { - ( 2 k + 2 a + 1 ) } + \frac { \sigma ^ { 2 } } { n } .
$$

Therefore, $\begin{array} { r } { f ^ { \prime } ( x ) = 0 \iff x = \left( \frac { n ( 2 k + 2 a ) } { \sigma ^ { 2 } } \right) ^ { 1 / ( 2 k + 2 a + 1 ) } } \end{array}$ . Thus,

$$
D _ { m } ^ { * } \approx n ^ { 1 / ( 2 a + 2 k + 1 ) } .
$$

Now let’s check when $D _ { m } ^ { * } \leq D _ { N _ { n , p } } ^ { * }$ . We distinguish two cases :

$\mathrm { I f ~ } D _ { N _ { n , p } } ^ { * } \approx p ^ { 1 / ( 2 a ) }$ (or the case where $\begin{array} { r } { p \lesssim \bigl ( \frac { n } { \ln ( n ) ^ { 2 } } \bigr ) ^ { 2 a } \bigr ) } \end{array}$ then,

$$
D _ { m } ^ { * } \leq D _ { N _ { n , p } } ^ { * } \iff n ^ { \frac { 1 } { 2 a + 2 k + 1 } } \leq p ^ { \frac { 1 } { 2 a } } \iff n ^ { \frac { 2 a } { 2 a + 2 k + 1 } } \leq p .
$$

• If $\begin{array} { r } { D _ { N _ { n , p } } ^ { * } \approx \frac { n } { \ln ( n ) ^ { 2 } } } \end{array}$ (or the case where $\begin{array} { r } { p \gtrsim \left( \frac { n } { \ln ( n ) ^ { 2 } } \right) ^ { 2 a } ) } \end{array}$ then,

$$
D _ { m } ^ { * } \leq D _ { N _ { n , p } } ^ { * } \iff n ^ { \frac { 1 } { 2 a + 2 k + 1 } } \leq \frac { n } { \ln ^ { 2 } ( n ) } .
$$

As $a > 0$ and $k \geq 1$ , the last inequality and the condition $\begin{array} { r } { p \gtrsim \left( \frac { n } { \ln ( n ) ^ { 2 } } \right) ^ { 2 a } } \end{array}$ are always compatible for n large enough.

From these two cases, we can deduce the following ones :

• If $\begin{array} { r } { n ^ { \frac { 2 a } { 2 a + 2 k + 1 } } \lesssim p \lesssim \left( \frac { n } { \ln ( n ) ^ { 2 } } \right) ^ { 2 a } } \end{array}$ , then

$$
\mathbb { E } ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } ) \le C _ { 1 } \left[ n ^ { \frac { - 2 k - 2 a } { 2 k + 2 a + 1 } } + p ^ { \frac { - ( 2 a - 1 ) } { 2 a } } + p ^ { \frac { - ( 2 a - 1 ) } { 2 a } } n ^ { - 1 } + \frac { 1 } { n } + \frac { 1 } { p } + \frac { 1 } { n p } \right] ,
$$

where $C _ { 1 } > 0$ . Since $k \geq 1 , a > 0 , ( - 2 k - 2 a ) / ( 2 k + 2 a + 1 )$ lies between -1 and 0. Therefore $n ^ { \frac { - 2 k - 2 a } { 2 k + 2 a + 1 } }$ converges slower than $n ^ { - 1 }$ . Moreover the fact that $- 1 \leq - ( 2 a - 1 ) / ( 2 a )$ allows us to deduce that $p ^ { - 1 }$ is dominated by $p ^ { - ( 2 a - 1 ) / ( 2 a ) }$ . We also have,

$$
\frac { n ^ { - 1 } p ^ { - 1 } } { n ^ { - 1 } p ^ { - \frac { ( 2 a - 1 ) } { 2 a } } } = p ^ { - \frac { 1 } { 2 a } } \xrightarrow [ p  \infty ] 0 ,
$$

and

$$
\frac { n ^ { - 1 } p ^ { - \frac { ( 2 a - 1 ) } { 2 a } } } { n ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } } } \lesssim n ^ { - \frac { 2 a } { 2 a + 2 k + 1 } } \xrightarrow [ n  \infty ] { } 0 ,
$$

where we used the fact that $p ^ { - 1 } \lesssim n ^ { - \frac { 2 a } { 2 a + 2 k + 1 } }$ and thus that $p ^ { - { \frac { ( 2 a - 1 ) } { 2 a } } } \lesssim n ^ { - { \frac { 2 a - 1 } { 2 a + 2 k + 1 } } }$ to derive the last inequality. This allows us to conclude that $n ^ { - 1 } p ^ { - \frac { ( 2 a - 1 ) } { 2 a } }$ and $n ^ { - 1 } p ^ { - 1 }$ are dominated by $n ^ { - \frac { 2 a + 2 k } { 2 a + 2 k + 1 } }$ . Hence the dominant rate is $\mathcal { O } ( n ^ { \frac { - 2 k - 2 a } { 2 k + 2 a + 1 } } + p ^ { \frac { - ( 2 a - 1 ) } { 2 a } } )$ , and

$$
\mathbb { E } \Big ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } \Big ) = \mathcal { O } \Big ( n ^ { - \frac { 2 a + 2 k } { 2 a + 2 k + 1 } } + p ^ { - \frac { 2 a - 1 } { 2 a } } \Big ) .
$$

• If $\begin{array} { r } { p \gtrsim \left( \frac { n } { \ln ( n ) ^ { 2 } } \right) ^ { 2 a } } \end{array}$ , then

$$
\begin{array} { r l } & { \mathbb { E } \left( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } \right) \le C _ { 2 } \Big [ n ^ { \frac { - 2 k - 2 a } { 2 k + 2 a + 1 } } + n ^ { - ( 2 a - 1 ) } \ln ^ { - 2 ( 2 a - 1 ) } ( n ) + n ^ { - 2 a } \ln ^ { - 2 ( 2 a - 1 ) } ( n ) } \\ & { \qquad + \displaystyle \frac { 1 } { p } \frac { n } { \ln ^ { 2 } ( n ) } + \frac { 1 } { n p } \frac { n } { \ln ^ { 2 } ( n ) } + \frac { 1 } { n } + \displaystyle \frac { 1 } { p } + \frac { 1 } { n p } \Big ] , } \end{array}
$$

where $C _ { 2 } > 0$ . We now compare the diferent terms. First, let’s denote

$$
\begin{array} { r } { T _ { 1 } : = n ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } } , \qquad T _ { 2 } : = n ^ { - ( 2 a - 1 ) } \ln ^ { - 2 ( 2 a - 1 ) } ( n ) , \qquad T _ { 3 } : = n ^ { - 2 a } \ln ^ { - 2 ( 2 a - 1 ) } ( n ) , } \end{array}
$$

$$
T _ { 4 } : = \frac { n } { p \ln ^ { 2 } n } , \qquad T _ { 5 } : = \frac { 1 } { p \ln ^ { 2 } n } , \qquad T _ { 6 } : = \frac { 1 } { n } , \qquad T _ { 7 } : = \frac { 1 } { p } , \qquad T _ { 8 } : = \frac { 1 } { n p } .
$$

Then the bound may be rewritten as

$$
\begin{array} { r } { \mathbb { E } \left( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } \right) \leq C _ { 2 } ( T _ { 1 } + T _ { 2 } + T _ { 3 } + T _ { 4 } + T _ { 5 } + T _ { 6 } + T _ { 7 } + T _ { 8 } ) . } \end{array}
$$

We first observe that

$$
T _ { 3 } = n ^ { - 2 a } \ln ^ { - 2 ( 2 a - 1 ) } ( n ) = { \frac { 1 } { n } } n ^ { - ( 2 a - 1 ) } \ln ^ { - 2 ( 2 a - 1 ) } ( n ) = { \frac { 1 } { n } } T _ { 2 } .
$$

Hence $T _ { 3 } = o ( T _ { 2 } )$ and thus $T _ { 3 }$ is negligible compared with $T _ { 2 }$ . In a similar way we get that

$$
T _ { 5 } = \frac { 1 } { p \ln ^ { 2 } n } = \frac { 1 } { n } \frac { n } { p \ln ^ { 2 } n } = \frac { 1 } { n } T _ { 4 } ,
$$

therefore $T _ { 5 } = o ( T _ { 4 } )$ . Moreover,

$$
T _ { 8 } = \frac { 1 } { n p } = \frac { 1 } { n } \frac { 1 } { p } = \frac { 1 } { n } T _ { 7 } ,
$$

thus $T _ { 8 } = o ( T _ { 7 } )$ . Next, let’s compare $T _ { 7 }$ and $T _ { 4 }$ :

$$
\frac { T _ { 4 } } { T _ { 7 } } = \frac { n / ( p \ln ^ { 2 } n ) } { 1 / p } = \frac { n } { \ln ^ { 2 } n } \xrightarrow [ n  \infty ] { } \infty .
$$

Therefore $T _ { 7 } = o ( T _ { 4 } )$ , and consequently $T _ { 8 } = o ( T _ { 4 } )$ as well. Finally, let us compare $T _ { 6 }$ and $T _ { 1 }$ . Since $\begin{array} { r } { \frac { 2 k + 2 a } { 2 k + 2 a + 1 } < 1 } \end{array}$ , we have that $T _ { 1 } = n ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } }$ decays more slowly than $n ^ { - 1 }$ . Hence $T _ { 6 } = o ( T _ { 1 } )$ . Therefore the bound simplifies to

$$
\mathbb { E } \left( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } \right) = \mathcal { O } \left( n ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } } + n ^ { - ( 2 a - 1 ) } \ln ^ { - 2 ( 2 a - 1 ) } ( n ) + \frac { n } { p \ln ^ { 2 } n } \right) .
$$

As we are in the situation where $\begin{array} { r } { p \gtrsim \left( \frac { n } { \ln ^ { 2 } n } \right) ^ { 2 a } } \end{array}$ we have that

$$
T _ { 4 } = { \frac { n } { p \ln ^ { 2 } n } } \lesssim { \frac { n } { \left( n / \ln ^ { 2 } n \right) ^ { 2 a } \ln ^ { 2 } n } } = { \frac { n } { n ^ { 2 a } \ln ^ { - 4 a } n \cdot \ln ^ { 2 } n } } = n ^ { 1 - 2 a } \ln ^ { 4 a - 2 } ( n ) .
$$

Since $1 - 2 a = - ( 2 a - 1 )$ , we obtain

$$
T _ { 4 } = \mathcal { O } \Big ( n ^ { - ( 2 a - 1 ) } \ln ^ { 4 a - 2 } ( n ) \Big ) .
$$

Therefore,

$$
\begin{array} { r } { \mathbb { E } \left( \Vert \widetilde { \beta } - \beta \Vert _ { \Gamma } ^ { 2 } \right) = { \mathcal O } \left( n ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } } + n ^ { - ( 2 a - 1 ) } \ln ^ { - 2 ( 2 a - 1 ) } ( n ) + n ^ { - ( 2 a - 1 ) } \ln ^ { 4 a - 2 } ( n ) \right) . } \end{array}
$$

We compare now the last two terms. Since for all $n \geq 2 , \ln ^ { 4 a - 2 } ( n ) \geq \ln ^ { - 2 ( 2 a - 1 ) } ( n )$ the term $n ^ { - ( 2 a - 1 ) } \ln ^ { 4 a - 2 } ( n )$ is larger than $n ^ { - ( 2 a - 1 ) } \ln ^ { - 2 ( 2 a - 1 ) } ( n )$ . Thus $T _ { 2 }$ is dominated by $T _ { 4 }$ under the present lower bound on $p .$ Therefore,

$$
\begin{array} { r } { \mathbb E \left( \| \widetilde \beta - \beta \| _ { \Gamma } ^ { 2 } \right) = \mathcal O \left( n ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } } + n ^ { - ( 2 a - 1 ) } \ln ^ { 4 a - 2 } ( n ) \right) . } \end{array}
$$

We now compare

$$
n ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } } \qquad \mathrm { a n d } \qquad n ^ { - ( 2 a - 1 ) } \ln ^ { 4 a - 2 } ( n ) .
$$

The dominant term is determined first by the powers of $n .$ We therefore study whether $\begin{array} { r } { \frac { 2 k + 2 a } { 2 k + 2 a + 1 } < 2 a - 1 } \end{array}$ . We have

$$
\begin{array} { r c l } { \frac { 2 k + 2 a } { 2 k + 2 a + 1 } < 2 a - 1 \iff 2 k + 2 a < ( 2 a - 1 ) ( 2 k + 2 a + 1 ) } \\ { \iff 2 k + 2 a < 4 a k + 4 a ^ { 2 } - 2 k - 1 } \\ { \iff 0 < 4 a ^ { 2 } + 4 a k - 2 a - 4 k - 1 } \\ { \iff 0 < 4 a ( a - 1 ) + 4 k ( a - 1 ) + ( 2 a - 1 ) } \\ { \iff 0 < ( a - 1 ) ( 4 a + 4 k ) + ( 2 a - 1 ) . } \end{array}
$$

Since $a \ge 1$ and $k \geq 0$ , it follows that

$$
( a - 1 ) ( 4 a + 4 k ) \geq 0 \qquad \mathrm { a n d } \qquad 2 a - 1 > 0 ,
$$

and thus $( a - 1 ) ( 4 a + 4 k ) + ( 2 a - 1 ) > 0$ . Hence,

$$
\frac { 2 k + 2 a } { 2 k + 2 a + 1 } < 2 a - 1 .
$$

Therefore $n ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } }$ decays more slowly and is the dominant term. We conclude that

$$
\mathbb { E } \left( \Vert \widetilde { \beta } - \beta \Vert _ { \Gamma } ^ { 2 } \right) = \mathcal { O } \left( n ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } } \right) .
$$

Let us focus now on the case where $p \lesssim n ^ { \frac { 2 a } { 2 a + 2 k + 1 } }$ . In this situation we can’t have $D _ { m } ^ { * } \leq D _ { N _ { n , p } } ^ { * }$ and $D _ { m } ^ { * } \approx n ^ { 1 / ( 2 a + 2 k + 1 ) }$ . Here the best value possible that minimizes f is attained by taking $D _ { m } ^ { * } = \widetilde { D } _ { N _ { n , p } } ^ { * } \approx p ^ { 1 / ( 2 a ) }$ . We then get

$$
\begin{array} { r } { \mathbb { E } \Big ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } \Big ) \leq C _ { 3 } \left( p ^ { - \frac { 2 a + 2 k } { 2 a } } + n ^ { - 1 } p ^ { \frac { 1 } { 2 a } } + p ^ { - \frac { 2 a - 1 } { 2 a } } + p ^ { - \frac { 2 a - 1 } { 2 a } } n ^ { - 1 } + n ^ { - 1 } + p ^ { - 1 } + n ^ { - 1 } p ^ { - 1 } \right) . } \end{array}
$$

Following the previous discussion, we have that $p ^ { - 1 }$ is dominated by $p ^ { - { \frac { 2 a - 1 } { 2 a } } } . { \mathrm { ~ A s ~ } } a > 0$ and $k \geq 1$ we have $- ( 2 a + 2 k ) / ( 2 a ) \leq - ( 2 a - 1 ) / ( 2 a )$ and $\stackrel { - } { p } ^ { - \frac { 2 a + 2 k } { 2 a } }$ is dominated by $p ^ { - \frac { 2 a - 1 } { 2 a } }$

Since we are in the framework of $p \lesssim n ^ { \frac { 2 a } { 2 a + 2 k + 1 } }$ , we have $n ^ { - 1 } \lesssim p ^ { - \frac { 2 a + 2 k + 1 } { 2 a } }$ and obvious $\mathrm { | y , ~ - ( 2 a + }$ $2 k + 1 ) / ( 2 a ) \leq - ( 2 a - 1 ) / ( 2 a )$ . Hence $p ^ { - ( 2 a - 1 ) ^ { ' } ( 2 a ) }$ dominates $n ^ { - 1 }$

We also have, $n ^ { - 1 } p ^ { \frac { 1 } { 2 a } } \lesssim p ^ { - { \frac { 2 a + 2 k } { 2 a } } }$ and using a previous argument, we get that $p ^ { - { \frac { 2 a - 1 } { 2 a } } }$ dominates $n ^ { - 1 } p ^ { \frac { 1 } { 2 \alpha } }$ . Moreover,

$$
\frac { p ^ { - \frac { ( 2 a - 1 ) } { 2 a } } n ^ { - 1 } } { p ^ { - \frac { ( 2 a - 1 ) } { 2 a } } } \lesssim p ^ { - \frac { 2 a + 2 k + 1 } { 2 a } } \xrightarrow [ p  \infty ] { } 0
$$

and

$$
\frac { n ^ { - 1 } p ^ { - 1 } } { p ^ { - \frac { ( 2 a - 1 ) } { 2 a } } } \lesssim p ^ { - \frac { 2 a + 2 k } { 2 a } } \xrightarrow [ p  \infty ] 0 .
$$

Therefore, $p ^ { - { \frac { ( 2 a - 1 ) } { 2 a } } } n ^ { - 1 }$ and $n ^ { - 1 } p ^ { - 1 }$ are dominated by $p ^ { - { \frac { ( 2 a - 1 ) } { 2 a } } }$ . Hence,

$$
\mathbb { E } \bigg ( \| \widetilde { \beta } - \beta \| _ { \Gamma } ^ { 2 } \bigg ) = \mathcal { O } \Big ( p ^ { - \frac { 2 a - 1 } { 2 a } } \Big ) .
$$

Lemma C.1. Assume that (H3) holds. Then

$$
\mathbb { E } \Big ( \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 2 } \Big ) \leq C ^ { \prime } \left( \frac { D _ { N _ { n , p } } } { p } + D _ { N _ { n , p } } ^ { - ( 2 a - 1 ) } \right) ,
$$

where $C ^ { \prime }$ is a positive constant which depends on $\tau , c ^ { \prime } , a$

Proof. We can express X in the Fourier basis : $X = \Sigma _ { 1 \leq l }$ x<sub>l</sub>ϕ<sub>l</sub>, where $x _ { l } = \langle X , \phi _ { l } \rangle _ { L ^ { 2 } }$ . Using the definition of X and $\widetilde { X }$ we get that

$$
\begin{array} { l } { \displaystyle \mathbb { E } ( \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 2 } ) = \mathbb { E } \Big ( \int _ { 0 } ^ { 1 } ( \widetilde { X } ( t ) - X ( t ) ) ^ { 2 } d t \Big ) } \\ { \displaystyle \qquad = \mathbb { E } \Big ( \int _ { 0 } ^ { 1 } \big | \ \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } \widetilde { x } _ { j } \phi _ { j } ( t ) - \displaystyle \sum _ { 1 \leq j } x _ { j } \phi _ { j } ( t ) | ^ { 2 } d t \Big ) } \\ { \displaystyle \qquad = \mathbb { E } \Big ( \int _ { 0 } ^ { 1 } \big | \ \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } ( \widetilde { x } _ { j } - x _ { j } ) \phi _ { j } ( t ) - \displaystyle \sum _ { D _ { N _ { n , p } } < j } x _ { j } \phi _ { j } ( t ) | ^ { 2 } d t \Big ) . } \end{array}
$$

As the $( \phi _ { j } ) _ { 1 \leq j }$ are orthonormal, we then get

$$
\mathbb { E } ( \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 2 } ) = \mathbb { E } \Big ( \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } ( \widetilde { x } _ { j } - x _ { j } ) ^ { 2 } \Big ) + \mathbb { E } \Big ( \sum _ { D _ { N _ { n , p } } < j } x _ { j } ^ { 2 } \Big ) .
$$

As $\widetilde { x } _ { j } = \overline { { x } } _ { j } + \overline { { \eta } } _ { j }$ for all $j \in \{ 1 , \dots , D _ { N _ { n , p } } \}$ , where $\overline { { x } } _ { j }$ and $\overline { { \eta } } _ { j }$ are defined in (47), we get that $( \widetilde { x } _ { j } - x _ { j } ) ^ { 2 } = ( \overline { { x } } _ { j } - x _ { j } ) ^ { 2 } + \overline { { \eta } } _ { j } ^ { 2 } + 2 \overline { { \eta } } _ { j } ( \overline { { x } } _ { j } - x _ { j } )$ . Now, using the fact that $\begin{array} { r } { \mathbb { E } ( \overline { { \eta } } _ { j } ) = 0 , \mathbb { E } ( \overline { { \eta } } _ { j } ^ { 2 } ) = \frac { \tau ^ { 2 } } { p } } \end{array}$ by Lemma B.2 and that $( \eta _ { j } ) _ { j = 1 , \dots , D _ { N _ { n } } }$ are independent of $X$ , we get the following equation :

$$
\mathbb { E } \Big ( \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 2 } \Big ) \leq \frac { D _ { N _ { n , p } } \tau ^ { 2 } } { p } + \mathbb { E } \Big ( \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } ( \overline { { x } } _ { j } - x _ { j } ) ^ { 2 } \Big ) + \mathbb { E } \Big ( \sum _ { D _ { N _ { n , p } } < j } x _ { j } ^ { 2 } \Big ) .\tag{76}
$$

Let us focus on the second term of the right part of the inequality. Using Fubini’s theorem we have,

$$
\mathbb { E } \Big ( \sum _ { D _ { N _ { n , p } } < j } x _ { j } ^ { 2 } \Big ) = \sum _ { D _ { N _ { n , p } } < j } \mathbb { E } \left( x _ { j } ^ { 2 } \right) .
$$

Now using Fubini’s theorem and the definition of the kernel K associated with the norm Gamma mentioned in (18) we get,

$$
\begin{array} { l } { \mathbb { E } \left( x _ { j } ^ { 2 } \right) = \mathbb { E } \left( \langle X , \phi _ { j } \rangle _ { L ^ { 2 } } ^ { 2 } \right) } \\ { = \displaystyle \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \phi _ { j } ( t ) \phi _ { j } ( s ) \mathbb { E } \left( X ( s ) X ( t ) \right) d s d t } \\ { = \displaystyle \int _ { 0 } ^ { 1 } \phi _ { j } ( t ) \int _ { 0 } ^ { 1 } K ( s , t ) \phi _ { j } ( s ) d s d t } \\ { = \langle \Gamma \phi _ { j } , \phi _ { j } \rangle _ { L ^ { 2 } } } \\ { = \lambda _ { j } . } \end{array}
$$

Hence using (H3),

$$
\begin{array} { r l r } {  { \mathbb { E } \Big ( \sum _ { D _ { N _ { n , p } } < j } x _ { j } ^ { 2 } \Big ) = \sum _ { D _ { N _ { n , p } } < j } \mathbb { E } ( x _ { j } ^ { 2 } ) } } \\ & { } & { = \sum _ { N _ { n , p } < j } \lambda _ { j } } \\ & { } & { \leq c ^ { \prime } \sum _ { D _ { N _ { n , p } } < j } j ^ { - 2 a } } \\ & { } & { \quad D _ { N _ { n , p } < j } } \\ & { } & { \leq \frac { c ^ { \prime } } { 2 a - 1 } D _ { N _ { n , p } } ^ { - ( 2 a - 1 ) } . } \end{array}\tag{77}
$$

Let us focus now on $\mathbb { E } \Big ( \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } ( \overline { { x } } _ { j } - x _ { j } ) ^ { 2 } \Big )$ . In Lemma B.2 we defined for all $j \in \{ 1 , . . . , D _ { N _ { n , p } } \}$

$$
e _ { j } = { \overline { { x } } } _ { j } - x _ { j } ,
$$

and proved that

$$
\mathbb { E } \left( e _ { j } ^ { 2 } \right) = \overline { { \lambda } } _ { j } .
$$

Using now Lemma B.8 we get that

$$
\forall j \in \{ 1 , \ldots , D _ { N _ { n , p } } \} \qquad \overline { { { \lambda } } } _ { j } = \mathcal { O } ( p ^ { - 2 a } ) .
$$

Hence,

$$
\mathbb { E } \left( \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } ( \overline { { x } } _ { j } - x _ { j } ) ^ { 2 } \right) \leq C D _ { N _ { n , p } } p ^ { - 2 a } .\tag{78}
$$

Combining Equations (76), (77) and (78) and using the fact that $p ^ { - 2 a } \leq p ^ { - 1 }$ as $a > 1 / 2$ , we get

$$
\mathbb { E } \Big ( \| \widetilde { X } - X \| _ { L ^ { 2 } } ^ { 2 } \Big ) \leq C ^ { \prime } \left( \frac { D _ { N _ { n , p } } } { p } + D _ { N _ { n , p } } ^ { - ( 2 a - 1 ) } \right) .
$$

□

Lemma C.2. Assume that (H1), (H2), (H3) hold. Then,

$$
\begin{array} { r } { { \mathbb { E } ^ { 1 / 2 } } \left( \| X - \widetilde { X } \| _ { L ^ { 2 } } ^ { 4 } \right) \leq C ( { a } , { c } ^ { \prime } , \tau ) \left( D _ { N _ { n , p } } ^ { - ( 2 a - 1 ) } + \frac { D _ { N _ { n , p } } } { p } \right) . } \end{array}
$$

Proof. By definition of $X _ { 1 }$ and $\widetilde { X } _ { 1 }$ we have

$$
\| X - \widetilde { X } \| _ { L ^ { 2 } } ^ { 2 } = \sum _ { j \leq D _ { N _ { n , p } } } ( \widetilde { x } _ { j } - x _ { j } ) ^ { 2 } + \sum _ { j > D _ { N _ { n , p } } } x _ { j } ^ { 2 } .
$$

Using the inequality $( x + y ) ^ { 2 } \leq 2 x ^ { 2 } + 2 y ^ { 2 }$ first, and then the inequality ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ we get that

$$
\begin{array} { r l r } {  { \mathbb { E } ^ { 1 / 2 } ( \| X - \widetilde { X } \| _ { L ^ { 2 } } ^ { 4 } ) \le \sqrt { 2 } [ \mathbb { E } \Big ( \big ( \sum _ { j \le D _ { N _ { n , p } } } ( \widetilde { x } _ { j } - x _ { j } ) ^ { 2 } \big ) ^ { 2 } \Big ) + \mathbb { E } \Big ( \big ( \sum _ { j > D _ { N _ { n , p } } } x _ { j } ^ { 2 } \big ) ^ { 2 } \Big ) ] ^ { 1 / 2 } } } \\ & { } & { \le \sqrt { 2 } [ \mathbb { E } ^ { 1 / 2 } \Big ( \big ( \sum _ { j \le D _ { N _ { n , p } } } ( \widetilde { x } _ { j } - x _ { j } ) ^ { 2 } \big ) ^ { 2 } \Big ) + \mathbb { E } ^ { 1 / 2 } \Big ( \big ( \sum _ { j > D _ { N _ { n , p } } } x _ { j } ^ { 2 } \big ) ^ { 2 } \Big ) ] . } \end{array}
$$

Let us first study the term $\begin{array} { r } { \mathbb { E } \Big ( \big ( \sum _ { j \leq D _ { N _ { n , p } } } ( \widetilde { x } _ { j } - x _ { j } ) ^ { 2 } \big ) ^ { 2 } \Big ) ^ { 1 / 2 } } \end{array}$ . Let us recall that we can write $\widetilde { x } _ { j }$ as $\widetilde { \boldsymbol { x } } _ { j } = \boldsymbol { x } _ { j } + \boldsymbol { e } _ { j } + \boldsymbol { \overline { { \eta } } } _ { j }$ as defined in Equations (47) and (48). Therefore, applying Cauchy–Schwarz inequality to the vector $( 1 , 1 , \dots , 1 ) \in \mathbb { R } ^ { D _ { N _ { n , p } } }$ and $\left( ( e _ { j } + \overline { { \eta } } _ { j } ) ^ { 2 } \right) _ { j = 1 , \dots , D _ { N _ { n , p } } }$ we get

$$
\begin{array} { r l r } {  { ( \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } ( \widetilde { x } _ { j } - x _ { j } ) ^ { 2 } ) ^ { 2 } = ( \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } ( e _ { j } + \overline { { \eta } } _ { j } ) ^ { 2 } ) ^ { 2 } } } \\ & { } & { \leq D _ { N _ { n , p } } \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } ( e _ { j } + \overline { { \eta } } _ { j } ) ^ { 4 } } \\ & { } & { \leq 8 D _ { N _ { n , p } } ( \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } e _ { j } ^ { i } + \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } \overline { { \eta } } _ { j } ^ { 4 } ) . } \end{array}
$$

Let us study the term $\mathbb { E } \left( \sum _ { j = 1 } ^ { D _ { N _ { n , p } } } e _ { j } ^ { 4 } \right)$ . Hypothesis (H1) implies that the e<sub>j</sub>’s are sub-Gaussian with variance factor $\overline { { \lambda } } _ { j }$ . Theorem 2.1 of Boucheron et al. [2] gives us that for all $q ^ { \prime } \geq 1$ and for all $1 \le j \le D _ { N _ { n , p } }$

$$
\mathbb { E } \left( e _ { j } ^ { 2 q ^ { \prime } } \right) \leq q ^ { \prime } ! ( 4 \overline { { \lambda } } _ { j } ) ^ { q ^ { \prime } } ,
$$

and taking $q ^ { \prime } = 2$ leads to

$$
\mathbb { E } \left( e _ { j } ^ { 4 } \right) \leq 3 2 \overline { { \lambda } } _ { j } ^ { 2 } .
$$

Lemma B.8 gives that for all $1 \leq j \leq D _ { N _ { n , p } } , \overline { { { \lambda } } } _ { j } \leq C ( a , c ^ { \prime } ) p ^ { - 2 a }$ , hence

$$
\mathbb { E } \left( e _ { j } ^ { 4 } \right) \leq C ( a , c ^ { \prime } ) p ^ { - 4 a } .
$$

Now, let us focus on the term $\mathbb { E } \left( \overline { { \eta } } _ { i } ^ { 4 } \right)$ . In a similar way, hypothesis (H2) gives us that $\overline { { \eta } } _ { j }$ is sub-Gaussian with variance factor $\tau ^ { 2 } \dot { / } p .$ . Theorem 2.1 of Boucheron et al. [2] gives that for all $q ^ { \prime } \geq 1$ and for all $1 \leq j \leq D _ { N _ { n , p } ; }$

$$
\mathbb { E } \left( \overline { { \eta } } _ { j } ^ { 2 q ^ { \prime } } \right) \leq q ^ { \prime } ! \left( 4 \frac { \tau ^ { 2 } } { p } \right) ^ { q ^ { \prime } } .
$$

Taking $q ^ { \prime } = 2$ we get that

$$
\mathbb { E } \left( \overline { { \eta } } _ { j } ^ { 4 } \right) \leq 3 2 \frac { \tau ^ { 4 } } { p ^ { 2 } } .
$$

This allows us to get that

$$
\begin{array} { r l r } {  { \mathbb { E } ^ { 1 / 2 } \Big ( \big ( \sum _ { j \leq D _ { N _ { n , p } } } ( \widetilde { x } _ { j } - x _ { j } ) ^ { 2 } \big ) ^ { 2 } \Big ) \leq D _ { N _ { n , p } } ( \frac { C ( c ^ { \prime } , a ) } { p ^ { 4 a } } + \frac { C ( \tau ) } { p ^ { 2 } } ) ^ { 1 / 2 } } } \\ & { } & \\ & { } & { \leq D _ { N _ { n , p } } \frac { C ( c ^ { \prime } , a , \tau ) } { p } , } \end{array}\tag{79}
$$

where we used the fact that ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ and that $a > 1 / 2$ to get the last inequality. Lastly, let’s focus on $\mathbb { E } ^ { 1 / 2 } \Big ( \big ( \sum _ { j > D _ { N _ { n , p } } } x _ { j } ^ { 2 } \big ) ^ { 2 } \Big )$ . By using Tonelli’s theorem, we have

$$
\begin{array} { r } { \mathbb { E } \Big ( \big ( \displaystyle \sum _ { j > D _ { N _ { n , p } } } x _ { j } ^ { 2 } \big ) ^ { 2 } \Big ) = \mathbb { E } \left( \displaystyle \sum _ { j > D _ { N _ { n , p } } } \displaystyle \sum _ { k > D _ { N _ { n , p } } } x _ { j } ^ { 2 } x _ { k } ^ { 2 } \right) } \\ { = \displaystyle \sum _ { j > D _ { N _ { n , p } } } \displaystyle \sum _ { k > D _ { N _ { n , p } } } \mathbb { E } \left( x _ { j } ^ { 2 } x _ { k } ^ { 2 } \right) . } \end{array}
$$

Applying Cauchy–Schwarz inequality to $x _ { j } ^ { 2 }$ and $x _ { k } ^ { 2 }$ we find

$$
\begin{array} { r l r } { \mathbb { E } \Big ( \big ( \displaystyle \sum _ { j > D _ { N _ { n , p } } } x _ { j } ^ { 2 } \big ) ^ { 2 } \Big ) \leq \displaystyle \sum _ { j > D _ { N _ { n , p } } } \displaystyle \sum _ { k > D _ { N _ { n , p } } } \mathbb { E } ( x _ { j } ^ { 4 } ) ^ { 1 / 2 } \mathbb { E } ( x _ { k } ^ { 4 } ) ^ { 1 / 2 } } & { } & \\ { \quad \quad \quad \quad = \left( \displaystyle \sum _ { j > D _ { N _ { n , p } } } \mathbb { E } ( x _ { j } ^ { 4 } ) ^ { 1 / 2 } \right) ^ { 2 } . } \end{array}
$$

Hypothesis (H1) and the fact that the $\lambda _ { j } \mathrm { ^ { \circ } s }$ decrease in a polynomial way, give us that

$$
\begin{array} { r l r } {  { \mathbb { E } \Big ( \big ( \sum _ { j > D _ { N _ { n , p } } } x _ { j } ^ { 2 } \big ) ^ { 2 } \Big ) \leq 3 2 ( \sum _ { j > D _ { N _ { n , p } } } \lambda _ { j } ) ^ { 2 } } } \\ & { } & { \leq 3 2 c ^ { \prime 2 } ( \sum _ { j > D _ { N _ { n , p } } } j ^ { - 2 a } ) ^ { 2 } } \\ & { } & { \leq C ( a , c ^ { \prime } ) D _ { N _ { n , p } } ^ { - 2 ( 2 a - 1 ) } , } \end{array}
$$

and

$$
{  { \mathbb E } } ^ { 1 / 2 } \Big ( \big ( \sum _ { j > D _ { N _ { n , p } } } x _ { j } ^ { 2 } \big ) ^ { 2 } \Big ) \leq C ( a , c ^ { \prime } ) D _ { N _ { n , p } } ^ { - ( 2 a - 1 ) } .\tag{80}
$$

Combining (79) and (80) we find that

$$
\mathbb { E } \left( \| X - \widetilde { X } \| _ { L ^ { 2 } } ^ { 4 } \right) \leq C ( a , c ^ { \prime } , \tau ) \left( D _ { N _ { n , p } } ^ { - ( 2 a - 1 ) } + \frac { D _ { N _ { n , p } } } { p } \right) .
$$

## D Proof of Theorem 5.1

This section is divided into four parts. The first part presents the three conditions of Theorem 2.7 from Tsybakov [15], while the remaining three sections provide the results used to prove them. The proof follows the methodology introduced in Section 2.6 of Tsybakov [15]. The first step of the proof is to build test functions. We define $M \in \mathbb { N } ^ { * }$ and for all $l = 1 , \ldots , M$ we consider $w ^ { ( l ) } \in \{ 0 , 1 \} ^ { d }$ . We set $w ^ { ( 0 ) } = ( 0 , \ldots , 0 ) \in \mathbb { R } ^ { d }$ . Then we consider the following test functions

$$
\beta _ { w ^ { ( l ) } } ( t ) = \sum _ { j = 1 } ^ { d } w _ { j } ^ { ( l ) } u _ { j } \phi _ { j } ( t ) \quad \mathrm { f o r ~ a l l } \quad l = 0 , \ldots , M ,\tag{81}
$$

where

$$
u _ { j } ^ { 2 } = \frac { C } { n \lambda _ { j } } \mathrm { ~ w i t h ~ } C = \operatorname* { m i n } \left( L ^ { 2 } ( 2 \pi ) ^ { - 2 k } c ^ { \prime - 1 } , \sigma ^ { 2 } \frac { \alpha ^ { \prime } \log ( 2 ) } { 4 } \right) .\tag{82}
$$

We set $0 < \alpha ^ { \prime } < 1 / 8$ and $0 < \kappa < 1$ such that

$$
n \geq ( 1 - \kappa ) ^ { - ( 2 a + 2 k + 1 ) } .\tag{83}
$$

We also define

$$
d = \left\lceil \kappa n ^ { \frac { 1 } { 2 a + 2 k + 1 } } \right\rceil\tag{84}
$$

with n such that $d \geq 8$

Remark 3. The parameter M corresponds to the number of hypotheses in the packing set. Its introduction is motivated by the Varshamov–Gilbert lemma (Lemma 2.9 in Tsybakov $[ 1 5 ] )$ , which ensures the existence of a large subset of binary vectors with pairwise Hamming distance bounded below. This large family of well-separated alternatives is the key ingredient for applying Tsybakov’s lower-bound theorem.

We then consider the following setting :

• A class of functions containing the true slope function $\beta$ that we want to estimate : $W ^ { \mathrm { p e r } } ( k , L )$ which is given by Equation (3).

• A sample $( Z _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ which is such that for all $h = 0 , \ldots , p - 1$ , for all $i = 1 , \ldots , n$

$$
Z _ { i } ( t _ { h } ) = X _ { i } ( t _ { h } ) + \eta _ { i , h } ,\tag{85}
$$

where the $\eta _ { i , h } ^ { \ } \mathrm { : }$ are i.i.d, Gaussian, centered and of variance $\tau ^ { 2 }$ . For each $l = 0 , \ldots , M$ we consider

$$
Y _ { i } = \langle X _ { i } , \beta _ { w ^ { ( l ) } } \rangle _ { L ^ { 2 } } + \epsilon _ { i } ,\tag{86}
$$

where $\beta _ { w ^ { ( l ) } } \in W ^ { \mathrm { p e r } } ( k , L )$ and the $\epsilon _ { i } ^ { \phantom { } } \mathrm { { s } }$ are i.i.d, are Gaussian, centered and of variance $\sigma ^ { 2 }$ Both the $\epsilon _ { i } ^ { \phantom { } } \mathrm { { s } }$ and $\eta _ { i , h }$ ’s are independent of everything else. We denote the joint distribution of this sample by $P _ { l } = P _ { w ^ { ( l ) } }$

• A semi distance $\| \cdot \| _ { \Gamma }$ which is defined in (28).

In order to apply Theorem 2.7 of Tsybakov [15] we need to check the three following conditions:

$$
1 . \forall l = 0 , \ldots , M , \beta _ { w ^ { ( l ) } } \in W ^ { \mathrm { p e r } } ( k , L ) .
$$

2. $P _ { j } \ll P _ { 0 }$ for all $j = 1 , \dots , M$ and

$$
\frac { 1 } { M } \sum _ { j = 1 } ^ { M } K ( P _ { j } \| P _ { 0 } ) \le \alpha ^ { \prime } \log ( M ) ,
$$

with $0 < \alpha ^ { \prime } < 1 / 8$ and $P _ { j } = P _ { w ^ { ( j ) } }$ for $j = 0 , 1 , \dotsc , M$ . Here $K ( \cdot \| \cdot )$ denotes the Kullback– Leibler divergence.

$$
3 . \ \forall \ 0 \leq l < l ^ { \prime } \leq M
$$

$$
\begin{array} { r } { \| \beta _ { w ^ { ( l ) } } - \beta _ { w ^ { ( l ^ { \prime } ) } } \| _ { \Gamma } ^ { 2 } \geq 2 C _ { 4 } n ^ { - \frac { 2 k + 2 a } { 2 k + 2 a + 1 } } , } \end{array}
$$

where $C _ { 4 } > 0$ is a positive constant which will be defined bellow.

Proposition D.1, D.2 and D.3 allow us to apply this theorem and therefore to prove the result.

## D.1 Condition 1. of the proof

Proposition D.1. Consider $M \in \mathbb { N } \backslash \{ 0 \}$ and for all $l = 1 , \ldots , M$ , define $w ^ { ( l ) } \in \{ 0 , 1 \} ^ { d }$ , as well $a s \ w ^ { ( 0 ) } = ( 0 , 0 , \ldots , 0 ) \in \mathbb { R } ^ { d }$ where d is defined by $( 8 4 )$ . Consider for all $l = 0 , 1 , \ldots , M$ , the test functions $\beta _ { w ^ { ( l ) } }$ given by (81). Then, for all $l = 0 , 1 , \ldots , M$

$$
\beta _ { w ^ { ( l ) } } \in W ^ { p e r } ( k , L ) .
$$

Proof. As for all $t \in [ 0 , 1 ] \ \beta _ { w ^ { ( 0 ) } } ( t ) = 0$ we have that $\beta _ { w ^ { ( 0 ) } }$ is trivially in $W ^ { \mathrm { p e r } } ( k , L )$ . Let us focus on the case where $l = 1 , 2 , \ldots , M$ . We need to prove that for all $s = 1 , \ldots , k , \beta _ { w ^ { ( l ) } } ^ { ( s ) }$ is 1-periodic and that $\| \beta _ { w ^ { ( l ) } } ^ { ( s ) } \| _ { L ^ { 2 } } ^ { 2 } \leq L ^ { 2 }$ . For all $s = 1 , \ldots , k$ we have, by Lemma D.1

$$
\begin{array} { l } { { \displaystyle { \beta _ { w ^ { ( l ) } } ^ { ( s ) } ( t ) = \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor } ( 2 \pi j ) ^ { s } \Big ( \big [ a _ { s } w _ { 2 j } ^ { ( l ) } u _ { 2 j } + c _ { s } \mathbf { 1 } _ { \{ 2 j + 1 \leq d \} } w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } \big ] \phi _ { 2 j } ( t ) } } } \\ { { \qquad + \left[ b _ { s } w _ { 2 j } ^ { ( l ) } u _ { 2 j } + a _ { s } \mathbf { 1 } _ { \{ 2 j + 1 \leq d \} } w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } \right] \phi _ { 2 j + 1 } ( t ) \Big ) . } } \end{array}
$$

As each real Fourier basis function is 1-periodic we have that $\beta _ { w ^ { ( l ) } } ^ { ( s ) }$ is 1-periodic for all $s = 1 , . . . , M$ With the same argument we have, for $s = 0$ that the function is also 1-periodic. To conclude the proof we then need to prove that $\| \beta _ { w ^ { ( l ) } } ^ { ( k ) } \| _ { L ^ { 2 } } ^ { 2 } \leq L ^ { 2 }$ . By Lemma D.1 we have

$$
\| \boldsymbol { \beta } _ { w ^ { ( l ) } } ^ { ( k ) } \| _ { L ^ { 2 } } ^ { 2 } \leq ( 2 \pi ) ^ { 2 k } \sum _ { j = 1 } ^ { d } j ^ { 2 k } ( w _ { j } u _ { j } ) ^ { 2 } .
$$

Using the definition of $u _ { j }$ given by (82) and the fact that $\lambda _ { j }$ decrease in a polynomial way we have

$$
\Vert \boldsymbol { \beta } _ { w ^ { ( l ) } } ^ { ( k ) } \Vert _ { L ^ { 2 } } ^ { 2 } \leq \frac { L ^ { 2 } } { n } \sum _ { j = 1 } ^ { d } j ^ { 2 k + 2 a } .
$$

As $a > 1 / 2$ and k is a positive integer, $2 k + 2 a \geq 1$ and we get

$$
\frac { L ^ { 2 } } { n } \sum _ { j = 1 } ^ { d } j ^ { 2 k + 2 a } \leq \frac { L ^ { 2 } } { n } d ^ { 2 k + 2 a + 1 } .
$$

Using now the definition of d and equation (83) we find for n large enough

$$
\| \beta _ { w ^ { ( l ) } } ^ { ( k ) } \| _ { L ^ { 2 } } ^ { 2 } \leq L ^ { 2 } .
$$

Therefore, for all $l = 0 , 1 , \ldots , M$

$$
\beta _ { w ^ { ( l ) } } \in W ^ { \mathrm { p e r } } ( k , L ) .
$$

Lemma D.1. Consider $M \in \mathbb { N } \backslash \{ 0 \}$ and for all $l = 1 , \ldots , M$ , define $w ^ { ( l ) } \in \{ 0 , 1 \} ^ { d }$ where d is defined $\boldsymbol { b } y \ ( \boldsymbol { \delta } \mathscr { \boldsymbol { \mathscr { 4 } } } )$ . Consider for all $l = 1 , \ldots , M$ , the test functions $\beta _ { w ^ { ( l ) } }$ given by (81). Then, for all $l = 1 , \ldots , M$ and for all integer s $\ge 1 , \beta _ { w ^ { ( l ) } }$ is s-times diferentiable and

$$
\begin{array} { l } { { \displaystyle { \boldsymbol { \beta } } _ { w ^ { ( l ) } } ^ { ( s ) } ( t ) = \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor } ( 2 \pi j ) ^ { s } \Big ( \big [ a _ { s } w _ { 2 j } ^ { ( l ) } u _ { 2 j } + c _ { s } \mathbf { 1 } _ { \{ 2 j + 1 \leq d \} } w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } \big ] \phi _ { 2 j } ( t ) } } \\ { { \displaystyle ~ + \left[ b _ { s } w _ { 2 j } ^ { ( l ) } u _ { 2 j } + a _ { s } \mathbf { 1 } _ { \{ 2 j + 1 \leq d \} } w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } \right] \phi _ { 2 j + 1 } ( t ) \Big ) , } } \end{array}
$$

with $\begin{array} { r } { a _ { s } = \cos { \left( \frac { s \pi } { 2 } \right) } , b _ { s } = - \sin { \left( \frac { s \pi } { 2 } \right) } } \end{array}$ and $\begin{array} { r } { c _ { s } = \sin \left( \frac { s \pi } { 2 } \right) } \end{array}$ . Moreover,

$$
\| \boldsymbol { \beta } _ { w ^ { ( l ) } } ^ { ( s ) } \| _ { L ^ { 2 } } ^ { 2 } \leq ( 2 \pi ) ^ { 2 k } \sum _ { j = 1 } ^ { d } j ^ { 2 k } ( w _ { j } u _ { j } ) ^ { 2 } .
$$

Proof. Let $s \in \{ 1 , \ldots , k \}$ . Then, $\beta _ { w ^ { ( l ) } }$ is s-times diferentiable as a sum of s-times diferentiable functions and

$$
\beta _ { w ^ { ( l ) } } ^ { ( s ) } ( t ) = \sum _ { j = 1 } ^ { d } w _ { j } ^ { ( l ) } u _ { j } \phi _ { j } ^ { ( s ) } ( t ) = \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor } w _ { 2 j } ^ { ( l ) } u _ { 2 j } \phi _ { 2 j } ^ { ( s ) } ( t ) + \sum _ { j = 1 } ^ { \lfloor ( d - 1 ) / 2 \rfloor } w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } \phi _ { 2 j + 1 } ^ { ( s ) } ( t ) .
$$

Now,

$$
\sqrt { 2 } \frac { d ^ { s } \cos ( 2 \pi j t ) } { d t ^ { s } } = \sqrt { 2 } ( 2 \pi j ) ^ { s } \cos \left( 2 \pi j t + \frac { s \pi } { 2 } \right) ,
$$

$$
{ \sqrt { 2 } } { \frac { d ^ { s } \sin ( 2 \pi j t ) } { d t ^ { s } } } = { \sqrt { 2 } } ( 2 \pi j ) ^ { s } \sin \left( 2 \pi j t + { \frac { s \pi } { 2 } } \right) .
$$

The trigonometric formulas lead to,

$$
\begin{array} { c } { { \sqrt { 2 } \cos \left( 2 \pi j t + \displaystyle \frac { s \pi } { 2 } \right) = \sqrt { 2 } \cos \left( \displaystyle \frac { s \pi } { 2 } \right) \cos ( 2 \pi j t ) - \sqrt { 2 } \sin \left( \displaystyle \frac { s \pi } { 2 } \right) \sin ( 2 \pi j t ) } } \\ { { = a _ { s } \phi _ { 2 j } ( t ) + b _ { s } \phi _ { 2 j + 1 } ( t ) , } } \end{array}
$$

and

$$
\begin{array} { c l l } { { \sqrt 2 \sin \left( 2 \pi j t + \displaystyle \frac { s \pi } { 2 } \right) = \sqrt 2 \sin \left( \displaystyle \frac { s \pi } { 2 } \right) \cos ( 2 \pi j t ) + \sqrt 2 \cos \left( \displaystyle \frac { s \pi } { 2 } \right) \sin ( 2 \pi j t ) } } & { { } } & { { } } \\ { { = c _ { s } \phi _ { 2 j } ( t ) + a _ { s } \phi _ { 2 j + 1 } ( t ) , } } & { { } } & { { } } \end{array}
$$

with $\begin{array} { r } { a _ { s } = \cos { \left( \frac { s \pi } { 2 } \right) } , b _ { s } = - \sin { \left( \frac { s \pi } { 2 } \right) } } \end{array}$ and $\begin{array} { r } { c _ { s } = \sin \left( \frac { s \pi } { 2 } \right) } \end{array}$ . More particularly these coeficients have the following values depending on s:

<table><tr><td rowspan=1 colspan=1>s mod 4</td><td rowspan=1 colspan=1> $a _ { s }$ </td><td rowspan=1 colspan=1> $b _ { s }$ </td><td rowspan=1 colspan=1> $c _ { s }$ </td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>-1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>-1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>-1</td></tr></table>

Using the previous result, we get that

$$
\begin{array} { l } { { \displaystyle \beta _ { w ^ { ( 1 ) } } ^ { ( s ) } ( t ) = \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor } ( 2 \pi j ) ^ { s } \big [ a _ { s } w _ { 2 j } ^ { ( l ) } u _ { 2 j } \phi _ { 2 j } ( t ) + b _ { s } w _ { 2 j } ^ { ( l ) } u _ { 2 j } \phi _ { 2 j + 1 } ( t ) \big ] } } \\ { { \mathrm { } ~ + ~ \sum _ { j = 1 } ^ { \lfloor ( d - 1 ) / 2 \rfloor } ( 2 \pi j ) ^ { s } \big [ c _ { s } w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } \phi _ { 2 j } ( t ) + a _ { s } w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } \phi _ { 2 j + 1 } ( t ) \big ] } } \\ { { \mathrm { } ~ = \displaystyle \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor } ( 2 \pi j ) ^ { s } \Big ( \big [ a _ { s } w _ { 2 j } ^ { ( l ) } u _ { 2 j } + c _ { s } \mathbf { 1 } _ { \{ 2 j + 1 \leq d \} } w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } \big ] \phi _ { 2 j } ( t ) } } \\ { { \mathrm { } ~ + \left[ b _ { s } w _ { 2 j } ^ { ( l ) } u _ { 2 j } + a _ { s } \mathbf { 1 } _ { \{ 2 j + 1 \leq d \} } w _ { 2 j + 1 } ^ { ( l ) } w _ { 2 j + 1 } \right] \phi _ { 2 j + 1 } ( t ) \Big ) , } } \end{array}
$$

which proves the first point of the Lemma. Now let us focus on the second one. Let’s consider

$$
\begin{array} { r l } & { A _ { j } : = ( 2 \pi j ) ^ { k } \left( a _ { k } w _ { 2 j } ^ { ( l ) } u _ { 2 j } + c _ { k } \mathbf { 1 } _ { \{ 2 j + 1 \leq d \} } w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } \right) , } \\ & { B _ { j } : = ( 2 \pi j ) ^ { k } \left( b _ { k } w _ { 2 j } ^ { ( l ) } u _ { 2 j } + a _ { k } \mathbf { 1 } _ { \{ 2 j + 1 \leq d \} } w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } \right) , } \end{array}
$$

as well as,

$$
x _ { j } : = w _ { 2 j } ^ { ( l ) } u _ { 2 j } , \qquad y _ { j } : = \mathbf { 1 } _ { \left\{ 2 j + 1 \leq d \right\} } w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } .
$$

This gives the following expressions for $A _ { j }$ and $B _ { j }$ :

$$
A _ { j } = ( 2 \pi j ) ^ { k } ( a _ { k } x _ { j } + c _ { k } y _ { j } ) , \qquad B _ { j } = ( 2 \pi j ) ^ { k } ( b _ { k } x _ { j } + a _ { k } y _ { j } ) .
$$

From this we can deduce that

$$
\begin{array} { r l r } {  { A _ { j } ^ { 2 } + B _ { j } ^ { 2 } = ( 2 \pi j ) ^ { 2 k } \big [ ( a _ { k } x _ { j } + c _ { k } y _ { j } ) ^ { 2 } + ( b _ { k } x _ { j } + a _ { k } y _ { j } ) ^ { 2 } \big ] } } \\ & { } & { = ( 2 \pi j ) ^ { 2 k } \big [ ( a _ { k } ^ { 2 } + b _ { k } ^ { 2 } ) x _ { j } ^ { 2 } + ( c _ { k } ^ { 2 } + a _ { k } ^ { 2 } ) y _ { j } ^ { 2 } + 2 x _ { j } y _ { j } ( a _ { k } c _ { k } + a _ { k } b _ { k } ) \big ] . } \end{array}
$$

Now we can notice that

$$
\begin{array} { r } { a _ { k } ^ { 2 } + b _ { k } ^ { 2 } = \cos ^ { 2 } \left( \frac { k \pi } { 2 } \right) + \sin ^ { 2 } \left( \frac { k \pi } { 2 } \right) = 1 , \qquad c _ { k } ^ { 2 } + a _ { k } ^ { 2 } = \sin ^ { 2 } \left( \frac { k \pi } { 2 } \right) + \cos ^ { 2 } \left( \frac { k \pi } { 2 } \right) = 1 , } \end{array}
$$

and

$$
a _ { k } c _ { k } + a _ { k } b _ { k } = a _ { k } ( c _ { k } + b _ { k } ) = 0 .
$$

Therefore the cross term vanishes and both quadratic coeficients are equal to 1, which gives

$$
A _ { j } ^ { 2 } + B _ { j } ^ { 2 } = ( 2 \pi j ) ^ { 2 k } ( x _ { j } ^ { 2 } + y _ { j } ^ { 2 } ) = ( 2 \pi j ) ^ { 2 k } \left( ( w _ { 2 j } ^ { ( l ) } u _ { 2 j } ) ^ { 2 } + { \bf 1 } _ { \{ 2 j + 1 \leq d \} } ( w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } ) ^ { 2 } \right) .\tag{87}
$$

By orthonormality of the real Fourier basis $\{ \phi _ { j } \} _ { j \ge 1 }$ in $L ^ { 2 } ( 0 , 1 )$ , we have $\langle \phi _ { i } , \phi _ { j } \rangle _ { L ^ { 2 } } = \delta _ { i j }$ . Hence, when $\beta _ { w ^ { ( l ) } } ^ { ( k ) }$ is written as a linear combination

$$
\boldsymbol { \beta } _ { w ^ { ( l ) } } ^ { ( k ) } ( t ) = \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor } \big ( A _ { j } \phi _ { 2 j } ( t ) + B _ { j } \phi _ { 2 j + 1 } ( t ) \big ) ,
$$

its squared $L ^ { 2 }$ norm is simply the sum of the squared coeficients :

$$
\| \beta _ { w ^ { ( l ) } } ^ { ( k ) } \| _ { L ^ { 2 } } ^ { 2 } = \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor } { \big ( } A _ { j } ^ { 2 } + B _ { j } ^ { 2 } { \big ) } ,
$$

since all cross terms $\langle \phi _ { i } , \phi _ { j } \rangle _ { L ^ { 2 } }$ with $i \neq j$ vanish. Substituting equation (87) we obtain

$$
\begin{array} { r l r } {  { \| \beta _ { w ^ { ( l ) } } ^ { ( k ) } \| _ { L ^ { 2 } } ^ { 2 } = ( 2 \pi ) ^ { 2 k } \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor } j ^ { 2 k } ( ( w _ { 2 j } ^ { ( l ) } u _ { 2 j } ) ^ { 2 } + \mathbf { 1 } _ { \{ 2 j + 1 \leq d \} } ( w _ { 2 j + 1 } ^ { ( l ) } u _ { 2 j + 1 } ) ^ { 2 } ) } } \\ & { } & { = ( 2 \pi ) ^ { 2 k } ( \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor } j ^ { 2 k } ( w _ { 2 j } u _ { 2 j } ) ^ { 2 } + \sum _ { j = 1 } ^ { \lfloor ( d - 1 ) / 2 \rfloor } j ^ { 2 k } ( w _ { 2 j + 1 } u _ { 2 j + 1 } ) ^ { 2 } ) , } \end{array}\tag{88}
$$

where the indicator function has been replaced by the upper limit $\lfloor ( d - 1 ) / 2 \rfloor$ . Let us treat the first sum. We set $r = 2 j$ . Then $j = r / 2$ , and when $j = 1$ we have $r = 2 .$ while when $j = \lfloor d / 2 \rfloor$ ， $r = 2 \lfloor d / 2 \rfloor \leq d .$ Therefore

$$
\sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor } j ^ { 2 k } ( w _ { 2 j } u _ { 2 j } ) ^ { 2 } = \sum _ { r = 2 , 4 , \ldots \atop r \leq d } \left( \frac { r } { 2 } \right) ^ { 2 k } ( w _ { r } u _ { r } ) ^ { 2 } .
$$

Since $r / 2 \leq r ,$ we get the bound

$$
\sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor } j ^ { 2 k } ( w _ { 2 j } u _ { 2 j } ) ^ { 2 } \leq \sum _ { r = 2 , 4 , \ldots \atop r \leq d } r ^ { 2 k } ( w _ { r } u _ { r } ) ^ { 2 } .
$$

Let us now focus on the second sum of equation (88). We set $r = 2 j + 1$ . Then $j = ( r - 1 ) / 2$ , and as $j$ runs from 1 to $\lfloor ( d - 1 ) / 2 \rfloor$ , r runs over all odd integers $r = 3 , 5 , \ldots , 2 \lfloor ( d - 1 ) / 2 \rfloor + 1 \le d .$ Hence,

$$
\sum _ { j = 1 } ^ { \lfloor ( d - 1 ) / 2 \rfloor } j ^ { 2 k } ( w _ { 2 j + 1 } u _ { 2 j + 1 } ) ^ { 2 } = \sum _ { r = 1 , 3 , \ldots \atop r < d } \left( \frac { r - 1 } { 2 } \right) ^ { 2 k } ( w _ { r } u _ { r } ) ^ { 2 } .
$$

Since $( r - 1 ) / 2 \leq r .$ , we obtain

$$
\sum _ { j = 1 } ^ { \lfloor ( d - 1 ) / 2 \rfloor } j ^ { 2 k } ( w _ { 2 j + 1 } u _ { 2 j + 1 } ) ^ { 2 } \leq \sum _ { r = 1 , 3 , \ldots \atop r \leq d } r ^ { 2 k } ( w _ { r } u _ { r } ) ^ { 2 } .
$$

Adding the even and odd contributions gives

$$
\Vert \beta _ { w ^ { ( l ) } } ^ { ( k ) } \Vert _ { L ^ { 2 } } ^ { 2 } \leq ( 2 \pi ) ^ { 2 k } \left[ \sum _ { r = 2 , 4 , \ldots \atop r \leq d } r ^ { 2 k } ( w _ { r } u _ { r } ) ^ { 2 } + \sum _ { r = 1 , 3 , \ldots \atop r \leq d } r ^ { 2 k } ( w _ { r } u _ { r } ) ^ { 2 } \right] .
$$

The union of even and odd indices from 1 to d is simply $\{ 1 , 2 , \ldots , d \}$ , therefore we can merge the sums:

$$
\| \beta _ { w ^ { ( l ) } } ^ { ( k ) } \| _ { L ^ { 2 } } ^ { 2 } \leq ( 2 \pi ) ^ { 2 k } \sum _ { r = 1 } ^ { d } r ^ { 2 k } ( w _ { r } u _ { r } ) ^ { 2 } .
$$

## D.2 Condition 2. of the proof

Proposition D.2. Let M $\in \mathbb { N } \setminus \{ 0 \}$ and, for each $l = 1 , \ldots , M$ , let $w ^ { ( l ) } \in \{ 0 , 1 \} ^ { d }$ and $w ^ { ( 0 ) } =$ $( 0 , \ldots , 0 )$ , where d is defined in $( 8 4 ) . \ F o r \ j \in \{ 0 , l \}$ , let $\beta _ { w ^ { ( j ) } }$ be defined by $( 8 1 )$ . Let $P _ { j }$ denote the joint law of the observed sample $\{ ( Z _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ under the model associated with $\beta _ { w ^ { ( j ) } }$ . Then, for every $l = 1 , \ldots , M$

$$
K ( P _ { 0 } \| P _ { l } ) \le \alpha ^ { \prime } \log M ,
$$

for some $\ 0 < \alpha ^ { \prime } < \frac { 1 } { 8 }$

Proof. We first work at the level of a single observation $( X , Z , Y )$ , and later use the fact that $( X _ { i } , Z _ { i } , Y _ { i } ) _ { i = 1 , \ldots , n }$ are i.i.d to recover the result for the full sample. Using Lemma D.2, we have

$$
\begin{array} { r } { K \big ( P _ { 0 } ^ { 1 } \| P _ { l } ^ { 1 } \big ) \le \mathbb { E } _ { X \sim P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( X ) } \left[ K \big ( P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( Y \mid X ) \| P _ { \mathrm { f u l l } } ^ { ( l ) , 1 } ( Y \mid X ) \big ) \right] , } \end{array}\tag{89}
$$

where $P _ { \mathrm { f u l l } } ^ { ( l ) , 1 }$ is the probability of the triplet $( X , Z , Y )$ under model l. We now compute the righthand side. Under $P _ { \mathrm { f u l l } } ^ { ( l ) , 1 }$ , conditionally on X, we have

$$
Y \mid X \sim { \mathcal { N } } { \big ( } \langle X , \beta _ { w ^ { ( l ) } } \rangle _ { L ^ { 2 } } , \sigma ^ { 2 } { \big ) } .
$$

Define

$$
\nu ^ { ( l ) } : = \langle X , \beta _ { w ^ { ( l ) } } \rangle _ { L ^ { 2 } } = \sum _ { j = 1 } ^ { d } w _ { j } ^ { ( l ) } u _ { j } \langle X , \phi _ { j } \rangle _ { L ^ { 2 } } .\tag{90}
$$

and

$$
\nu ^ { ( 0 ) } : = 0 .
$$

Since the conditional distributions are Gaussian with the same variance, we obtain

$$
\frac { d P _ { \mathrm { \scriptsize { f u l l } } } ^ { ( 0 ) , 1 } ( Y \mid X ) } { d P _ { \mathrm { \scriptsize { f u l l } } } ^ { ( l ) , 1 } ( Y \mid X ) } = \exp \left( \frac { 1 } { 2 \sigma ^ { 2 } } \left[ ( Y - \nu ^ { ( l ) } ) ^ { 2 } - ( Y - \nu ^ { ( 0 ) } ) ^ { 2 } \right] \right) ,
$$

and after developing the square we get

$$
\frac { d P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( Y \mid X ) } { d P _ { \mathrm { f u l l } } ^ { ( l ) , 1 } ( Y \mid X ) } = \exp \left( \frac { 1 } { 2 \sigma ^ { 2 } } \left[ \left( \nu ^ { ( l ) } \right) ^ { 2 } - 2 Y \nu ^ { ( l ) } \right] \right) = \exp \left( \frac { \left( \nu ^ { ( l ) } \right) ^ { 2 } } { 2 \sigma ^ { 2 } } - \frac { Y \nu ^ { ( l ) } } { \sigma ^ { 2 } } \right) .
$$

By taking expectation under $P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( Y \mid X )$ , we obtain

$$
\mathbb { E } _ { P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } } [ \log \frac { d P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( Y \mid X ) } { d P _ { \mathrm { f u l l } } ^ { ( l ) , 1 } ( Y \mid X ) } | X ] = \frac { 1 } { 2 \sigma ^ { 2 } } \Big [ ( \nu ^ { ( l ) } ) ^ { 2 } - 2 \nu ^ { ( l ) } \mathbb { E } ( Y \mid X ) \Big ] .
$$

Now, under $P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 }$ , we have $\mathbb { E } ( Y \mid X ) = \nu ^ { ( 0 ) } = 0$ , hence

$$
\mathbb { E } _ { P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } } [ \log \frac { d P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( Y \mid X ) } { d P _ { \mathrm { f u l l } } ^ { ( l ) , 1 } ( Y \mid X ) } ] X ] = \frac { 1 } { 2 \sigma ^ { 2 } } ( \nu ^ { ( l ) } ) ^ { 2 } .
$$

Using the definition of $\nu ^ { ( l ) }$ given in (90) we have that

$$
\left( \nu ^ { ( l ) } \right) ^ { 2 } = \sum _ { j , k = 1 } ^ { d } w _ { j } ^ { ( l ) } w _ { k } ^ { ( l ) } u _ { j } u _ { k } \left. X , \phi _ { j } \right. _ { L ^ { 2 } } \left. X , \phi _ { k } \right. _ { L ^ { 2 } } .
$$

Taking expectation with respect to $X \sim P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( X )$ , we obtain

$$
\mathbb { E } \left[ \left( \nu ^ { ( l ) } \right) ^ { 2 } \right] = \sum _ { j , k = 1 } ^ { d } w _ { j } ^ { ( l ) } w _ { k } ^ { ( l ) } u _ { j } u _ { k } \mathbb { E } \big [ \langle X , \phi _ { j } \rangle _ { L ^ { 2 } } \langle X , \phi _ { k } \rangle _ { L ^ { 2 } } \big ] .
$$

Using now equation (49) we get that

$$
\mathbb { E } \left[ \left( \nu ^ { ( l ) } \right) ^ { 2 } \right] = \sum _ { j = 1 } ^ { d } \lambda _ { j } u _ { j } ^ { 2 } \big ( w _ { j } ^ { ( l ) } \big ) ^ { 2 } .
$$

Combining with (89), we get for one observation

$$
K \big ( P _ { 0 } ^ { 1 } \| P _ { l } ^ { 1 } \big ) \le \frac { 1 } { 2 \sigma ^ { 2 } } \sum _ { j = 1 } ^ { d } \lambda _ { j } u _ { j } ^ { 2 } \big ( w _ { j } ^ { ( l ) } \big ) ^ { 2 } ,
$$

Since the observations are independent, we have

$$
K \big ( P _ { 0 } \| P _ { l } \big ) = n K \big ( P _ { 0 } ^ { 1 } \| P _ { l } ^ { 1 } \big ) ,
$$

and therefore

$$
K ( P _ { 0 } \| P _ { l } ) \leq \frac { n } { 2 \sigma ^ { 2 } } \sum _ { j = 1 } ^ { d } \lambda _ { j } u _ { j } ^ { 2 } \big ( w _ { j } ^ { ( l ) } \big ) ^ { 2 } .
$$

Using the definition $\begin{array} { r } { u _ { j } ^ { 2 } = \frac { C } { n \lambda _ { j } } } \end{array}$ and the bound $C \leq \sigma ^ { 2 } \frac { \alpha ^ { \prime } \log ( 2 ) } { 4 }$ , we obtain

$$
\frac { n } { 2 \sigma ^ { 2 } } \sum _ { j = 1 } ^ { d } \lambda _ { j } u _ { j } ^ { 2 } \bigl ( w _ { j } ^ { ( l ) } \bigr ) ^ { 2 } \leq \frac { \alpha ^ { \prime } \log ( 2 ) d } { 8 } .
$$

Finally, by the Varshamov–Gilbert bound (Lemma 2.9 in [15]), $M \geq 2 ^ { d / 8 }$ , hence $\begin{array} { r } { d \le \frac { 8 \log ( M ) } { \log ( 2 ) } } \end{array}$ which concludes the proof. □

Lemma D.2. Let $l \in \{ 1 , \ldots , M \}$ . Let $P _ { 0 } ^ { 1 }$ and $P _ { l } ^ { 1 }$ denote the joint laws $o f \left( Z , Y \right)$ under the models associated with $\beta _ { w ^ { ( 0 ) } }$ and $\beta _ { w ^ { ( l ) } }$ , respectively. Let $P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 }$ and $P _ { \mathrm { f u l l } } ^ { ( l ) , 1 }$ denote the corresponding joint laws of $( X , Z , Y )$ . Then

$$
\begin{array} { r } { K ( P _ { 0 } ^ { 1 } \| P _ { l } ^ { 1 } ) \le \mathbb { E } _ { X \sim P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( X ) } \left[ K \left( P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( Y | X ) \| P _ { \mathrm { f u l l } } ^ { ( l ) , 1 } ( Y | X ) \right) \right] . } \end{array}
$$

Proof. We work with the full joint model for each $j \in \{ 0 , l \}$ , denoted $P _ { \mathrm { f u l l } } ^ { ( j ) , 1 }$ , governing the triplet $( X , Z , Y )$ . The observed law $P _ { j } ^ { 1 }$ is the $( Z , Y )$ -marginal of $P _ { \mathrm { f u l l } } ^ { ( j ) , 1 }$

Using Lemma D.3 stated below and its associated notation, we have

$$
\begin{array} { r } { K \big ( P _ { 0 } ^ { 1 } \| P _ { l } ^ { 1 } \big ) = K \big ( P _ { 0 } ^ { 1 , Z } \| P _ { l } ^ { 1 , Z } \big ) + \mathbb { E } _ { Z \sim P _ { 0 } ^ { 1 , Z } } \left[ K \big ( P _ { 0 } ^ { 1 } ( \cdot \mid Z ) \| P _ { l } ^ { 1 } ( \cdot \mid Z ) \big ) \right] . } \end{array}\tag{91}
$$

Since the marginal law of $Z$ does not depend on $j$ for $j \in \{ 0 , l \}$ (the distribution of $( X , \eta )$ being identical across models), we have $P _ { 0 } ^ { 1 , Z } = P _ { l } ^ { 1 , Z }$ and thus

$$
K \big ( P _ { 0 } ^ { 1 , Z } \| P _ { l } ^ { 1 , Z } \big ) = 0 .
$$

Now fix z. For $j \in \{ 0 , l \}$ we can define under the full model $P _ { \mathrm { f u l l } } ^ { ( j ) , 1 }$ ，

$$
\mu _ { z } ( d x ) : = P _ { \mathrm { f u l l } } ^ { ( j ) , 1 } ( X \in d x \mid Z = z )
$$

be a regular conditional distribution of $X$ given $Z = z$ . Note that $\mu _ { z }$ does not depend on $j ,$ since the joint law of $( X , Z )$ is the same for all $j .$ . Let us denote by $\mu ( d x )$ the law of $X$ . We have using Bayes’s formula that the density of $\mu _ { z } ( d x )$ with respect to $\mu ( d x )$ is

$$
\mu _ { z } ( d x ) = \frac { f _ { Z | X = x } ( z ) } { \int f _ { Z | X = u } ( z ) \mu ( d u ) } \mu ( d x ) ,\tag{92}
$$

where $f _ { Z \mid X = x }$ is defined in equation (95). Using the definition of the conditional density (98) we have for $\dot { j } \in \{ 0 , l \}$ that

$$
p _ { j } ( y \mid z ) = \frac { p _ { j } ( z , y ) } { p _ { j } ^ { Z } ( z ) } = \frac { \int f _ { Z | X = x } ( z ) p _ { \mathrm { f u l l } } ^ { ( j ) } ( y \mid x ) \mu ( d x ) } { \int f _ { Z | X = x } ( z ) \mu ( d x ) } .
$$

Now using the definition of $\mu _ { z }$ given in equation (92) we find for $j \in \{ 0 , l \}$ ,

$$
\begin{array} { r l } & { \displaystyle \int p _ { \mathrm { f u l l } } ^ { ( j ) } ( y \mid x ) \mu _ { z } ( d x ) = \int p _ { \mathrm { f u l l } } ^ { ( j ) } ( y \mid x ) \frac { f _ { Z | X = x } ( z ) } { \int f _ { Z | X = u } ( z ) \mu ( d u ) } \mu ( d x ) } \\ & { \quad \quad \quad = \frac { \int f _ { Z | X = x } ( z ) p _ { \mathrm { f u l l } } ^ { ( j ) } ( y \mid x ) \mu ( d x ) } { \int f _ { Z | X = x } ( z ) \mu ( d x ) } } \\ & { \quad \quad = p _ { j } ( y \mid z ) . } \end{array}
$$

Let us still consider $z$ fixed, and let us compute $K \big ( P _ { 0 } ^ { 1 } ( \cdot \mid Z = z ) \| P _ { l } ^ { 1 } ( \cdot \mid Z = z ) \big )$ . Using the definition of the Kullback–Leibler divergence and the previous representation for $p _ { j } ( y \mid z )$ , we have

$$
\begin{array} { r l } & { K \big ( P _ { 0 } ^ { 1 } ( \cdot \mid Z = z ) \| P _ { l } ^ { 1 } ( \cdot \mid Z = z ) \big ) = \displaystyle \int p _ { 0 } ( y \mid z ) \log \frac { p _ { 0 } ( y \mid z ) } { p _ { l } ( y \mid z ) } d y } \\ & { \quad \quad \quad \quad = \displaystyle \int \left( \int p _ { \mathrm { f u l l } } ^ { ( 0 ) } ( y \mid x ) \mu _ { z } ( d x ) \right) \log \frac { \int p _ { \mathrm { f u l l } } ^ { ( 0 ) } ( y \mid x ) \mu _ { z } ( d x ) } { \int p _ { \mathrm { f u l l } } ^ { ( l ) } ( y \mid x ) \mu _ { z } ( d x ) } d y . } \end{array}
$$

Now let us fix y and set

$$
a ( x ) = p _ { \mathrm { f u l l } } ^ { ( 0 ) } ( y \mid x ) , \qquad b ( x ) = p _ { \mathrm { f u l l } } ^ { ( l ) } ( y \mid x ) , \qquad s = \int b ( x ) \mu _ { z } ( d x ) .
$$

Let us define the probability measure $\pi ( d x ) = \frac { b ( x ) \mu _ { z } ( d x ) } { s }$ and the ratio $r ( x ) = a ( x ) / b ( x )$ . For $\varphi ( t ) = t$ log t which is a convex function on $( 0 , \infty )$ we have by Jensen’s inequality,

$$
\begin{array} { l } { \displaystyle \int a ( x ) \log \left( \frac { a ( x ) } { b ( x ) } \right) \mu _ { z } ( d x ) = \int b ( x ) \varphi ( r ( x ) ) \mu _ { z } ( d x ) } \\ { \displaystyle \qquad = s \int \varphi ( r ( x ) ) \pi ( d x ) } \\ { \displaystyle \qquad \geq s \varphi \left( \int r ( x ) \pi ( d x ) \right) } \\ { \displaystyle \qquad = s \left( \frac { \int a ( x ) \mu _ { z } ( d x ) } { s } \right) \log \left( \frac { \int a ( x ) \mu _ { z } ( d x ) } { s } \right) . } \end{array}
$$

As $\begin{array} { r } { s = \int b ( x ) \mu _ { z } ( d x ) } \end{array}$ , we obtain for each $y ,$

$$
\Big ( \int p _ { \mathrm { f u l l } } ^ { ( 0 ) } ( y \mid x ) \mu _ { z } ( d x ) \Big ) \log \frac { \int p _ { \mathrm { f u l l } } ^ { ( 0 ) } ( y \mid x ) \mu _ { z } ( d x ) } { \int p _ { \mathrm { f u l l } } ^ { ( l ) } ( y \mid x ) \mu _ { z } ( d x ) } \le \int p _ { \mathrm { f u l l } } ^ { ( 0 ) } ( y \mid x ) \log \frac { p _ { \mathrm { f u l l } } ^ { ( 0 ) } ( y \mid x ) } { p _ { \mathrm { f u l l } } ^ { ( l ) } ( y \mid x ) } \mu _ { z } ( d x ) .
$$

Integrating over y and applying Fubini’s theorem gives

$$
K \big ( P _ { 0 } ^ { 1 } ( Y \mid Z = z ) \| P _ { l } ^ { 1 } ( Y \mid Z = z ) \big ) \leq \int \left( \int p _ { \mathrm { f u l l } } ^ { ( 0 ) } ( y \mid x ) \log \frac { p _ { \mathrm { f u l l } } ^ { ( 0 ) } ( y \mid x ) } { p _ { \mathrm { f u l l } } ^ { ( l ) } ( y \mid x ) } d y \right) \mu _ { z } ( d x ) .
$$

The inner integral is the Kullback–Leibler divergence $K \big ( P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( Y \mid X = x ) \| P _ { \mathrm { f u l l } } ^ { ( l ) , 1 } ( Y \mid X = x ) \big )$ Hence

$$
K \big ( P _ { 0 } ^ { 1 } ( Y \mid Z = z ) \| P _ { l } ^ { 1 } ( Y \mid Z = z ) \big ) \leq \mathbb { E } _ { X \sim P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( X \mid Z = z ) } \left[ K \big ( P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( Y \mid X ) \| P _ { \mathrm { f u l l } } ^ { ( l ) , 1 } ( Y \mid X ) \big ) \right] .
$$

Taking expectations over $Z \sim P _ { 0 } ^ { 1 , Z }$ and using (91) we $\mathrm { g e t }$

$$
\begin{array} { r } { K \big ( P _ { 0 } ^ { 1 } \| P _ { l } ^ { 1 } \big ) \le \mathbb { E } _ { X \sim P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( X ) } \left[ K \big ( P _ { \mathrm { f u l l } } ^ { ( 0 ) , 1 } ( Y \mid X ) \| P _ { \mathrm { f u l l } } ^ { ( l ) , 1 } ( Y \mid X ) \big ) \right] . } \end{array}\tag{93}
$$

Lemma D.3. Let $P _ { 0 } ^ { 1 }$ and $P _ { l } ^ { 1 } , l = 1 , \dots , M$ , be two probability measures on $\mathbb { R } ^ { p } \times$ R corresponding to the joint distribution of $( Z , Y )$ . In particular, for all model $j \in \{ 0 , l \}$ ,

$$
Y = \langle X , \beta _ { w ^ { ( j ) } } \rangle _ { L ^ { 2 } } + \epsilon ,\tag{94}
$$

where $\beta _ { w ^ { ( j ) } }$ given by $( 8 1 )$ and $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . Assume that $P _ { 0 } ^ { 1 } \ll P _ { l } ^ { 1 }$ and let us denote by $P _ { 0 } ^ { 1 , Z }$ and $P _ { l } ^ { 1 , Z }$ the marginal laws of $Z$ . Then,

$$
K ( P _ { 0 } ^ { 1 } \| P _ { l } ^ { 1 } ) = K ( P _ { 0 } ^ { 1 , Z } \| P _ { l } ^ { 1 , Z } ) + \mathbb { E } _ { Z \sim P _ { 0 } ^ { 1 , Z } } \left[ K \big ( P _ { 0 } ^ { 1 } ( \cdot \mid Z ) \| P _ { l } ^ { 1 } ( \cdot \mid Z ) \big ) \right] .
$$

$P r o o f .$ We first show that $P _ { 0 } ^ { 1 }$ and $P _ { l } ^ { 1 }$ admit densities with respect to the Lebesgue measure on $\mathbb { R } ^ { p + 1 }$ . By definition of $Z$ and using the fact that the $\eta _ { i , h } ^ { \ } \stackrel { , } { }$ s are Gaussian, centered and of variance $\tau ^ { 2 }$ , we have that

$$
Z \mid X = x \sim { \mathcal { N } } ( x ^ { \mathrm { g r i d } } , \tau ^ { 2 } I _ { p } ) ,
$$

where $x ^ { \mathrm { g r i d } } = ( x ( h / p ) ) _ { h = 0 , \ldots , p - 1 }$ . Therefore $Z \mid X = x$ admits a density $f _ { Z \mid X = x }$ which is such that

$$
f _ { Z | X = x } ( z ) = { \frac { 1 } { ( 2 \pi \tau ^ { 2 } ) ^ { p / 2 } } } \exp \left( - { \frac { \| z - x ^ { \mathrm { g r i d } } \| _ { 2 } ^ { 2 } } { 2 \tau ^ { 2 } } } \right) .\tag{95}
$$

Moreover, under $P _ { j } ^ { 1 } , j \in \{ 0 , l \}$ , the response variable Y is described by equation (94). Therefore for all $j \in \{ 0 , l \}$ we have conditional on $X = x$ ，

$$
Y \mid X = x \sim { \mathcal { N } } ( \langle x , \beta _ { w ^ { ( j ) } } \rangle _ { L ^ { 2 } } , \sigma ^ { 2 } ) .
$$

Hence it admits a density $p _ { \mathrm { f u l l } } ^ { ( j ) } ( \cdot \mid x )$ which is such that

$$
p _ { \mathrm { f u l l } } ^ { ( j ) } ( y \mid x ) = \frac { 1 } { ( 2 \pi \sigma ^ { 2 } ) ^ { 1 / 2 } } \exp \left( - \frac { ( y - \langle x , \beta _ { w ^ { ( j ) } } \rangle _ { L ^ { 2 } } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .
$$

As the $\epsilon _ { i } ^ { \phantom { } } \mathrm { { s } }$ and $\eta _ { i , h } \mathrm { ^ { \circ } s }$ are assumed to be independent we have that for all $j \in \{ 0 , l \} , l = 1 , \ldots , M$ $( Z , Y ) \mid X = x$ admits a density which is given by

$$
f _ { ( Z , Y ) \mid X = x } ^ { ( j ) } ( z , y ) = f _ { Z \mid X = x } ( z ) p _ { \mathrm { f u l l } } ^ { ( j ) } ( y \mid x ) .
$$

Let us now integrate out with respect to the law of X that we denote as $\mu ( d x )$ . We obtain that the density of $P _ { j } ^ { 1 }$ is given by

$$
\begin{array} { l } { { p } _ { j } ( z , y ) = \displaystyle \int f _ { ( Z , Y ) \mid X = x } ^ { ( j ) } ( z , y ) \mu ( d x ) } \\ { = \displaystyle \int f _ { Z \mid X = x } ( z ) p _ { \mathrm { f u l l } } ^ { ( j ) } ( y \mid x ) \mu ( d x ) . } \end{array}\tag{96}
$$

Finally, we get that $P _ { j } ^ { 1 , Z }$ admits a density too which is given by

$$
\begin{array} { l } { \displaystyle p _ { j } ^ { Z } ( z ) = \int p _ { j } ( z , y ) d y } \\ { \displaystyle \qquad = \int \int f _ { Z | X = x } ( z ) p _ { \mathrm { f u l l } } ^ { ( j ) } ( y \mid x ) \mu ( d x ) d y } \\ { \displaystyle \qquad = \int f _ { Z | X = x } ( z ) \mu ( d x ) , } \end{array}\tag{97}
$$

where we used Fubini’s theorem and the definition of a density to derive the last line. Thus the conditional density of Y given Z under $P _ { j } ^ { 1 }$ is given by

$$
\begin{array} { r l } & { p _ { j } ( y \mid z ) = \frac { p _ { j } ( z , y ) } { p _ { j } ^ { Z } ( z ) } } \\ & { \quad \quad = \frac { \int f _ { Z | X = x } ( z ) p _ { \mathrm { f u l l } } ^ { ( j ) } ( y \mid x ) \mu ( d x ) } { \int f _ { Z | X = x } ( z ) \mu ( d x ) } , } \end{array}\tag{98}
$$

By definition of the Kullback–Leibler divergence, we have

$$
\begin{array} { l } { { \displaystyle K \left( P _ { 0 } ^ { 1 } \| P _ { l } ^ { 1 } \right) = \int p _ { 0 } ( z , y ) \log \left( \frac { p _ { 0 } ( z , y ) } { p _ { l } ( z , y ) } \right) d z d y } } \\ { { \displaystyle \quad \quad = \int p _ { 0 } ( z , y ) \log \left( \frac { p _ { 0 } ( y \mid z ) p _ { 0 } ^ { Z } ( z ) } { p _ { l } ( y \mid z ) p _ { l } ^ { Z } ( z ) } \right) d z d y } } \\ { { \displaystyle \quad \quad = \int p _ { 0 } ( z , y ) \log \left( \frac { p _ { 0 } ^ { Z } ( z ) } { p _ { l } ^ { Z } ( z ) } \right) d z d y + \int p _ { 0 } ( z , y ) \log \left( \frac { p _ { 0 } ( y \mid z ) } { p _ { l } ( y \mid z ) } \right) d z d y , } } \end{array}
$$

where we used the definition of the conditional density given by equation (98) to get the second line and the property of the log to get the last one. Now let’s deal with these two terms one by one. For the first one we have, using Fubini’s theorem and the definition of the marginal density that

$$
\begin{array} { c l l } { { \displaystyle { \int p _ { 0 } ( z , y ) \log \left( \frac { p _ { 0 } ^ { Z } ( z ) } { p _ { l } ^ { Z } ( z ) } \right) d z d y } = \int \log \left( \frac { p _ { 0 } ^ { Z } ( z ) } { p _ { l } ^ { Z } ( z ) } \right) \left( \int p _ { 0 } ( z , y ) d y \right) d z } } \\ { { { } } } & { { { } = \displaystyle { \int p _ { 0 } ^ { Z } ( z ) \log \left( \frac { p _ { 0 } ^ { Z } ( z ) } { p _ { l } ^ { Z } ( z ) } \right) d z } } } \\ { { { } } } & { { { } = K \left( P _ { 0 } ^ { 1 , Z } \Vert P _ { l } ^ { 1 , Z } \right) . } } \end{array}
$$

For the second term, we get using the definition of conditional densities given in (98) and Fubini’s theorem that

$$
\begin{array} { r l } { \displaystyle \int p _ { 0 } ( z , y ) \log \bigg ( \frac { p _ { 0 } ( y \mid z ) } { p _ { l } ( y \mid z ) } \bigg ) d z d y = \int p _ { 0 } ^ { z } ( z ) p _ { 0 } ( y \mid z ) \log \bigg ( \frac { p _ { 0 } ( y \mid z ) } { p _ { l } ( y \mid z ) } \bigg ) d z d y } & { } \\ { = \displaystyle \int p _ { 0 } ^ { z } ( z ) \left( \int p _ { 0 } ( y \mid z ) \log \bigg ( \frac { p _ { 0 } ( y \mid z ) } { p _ { l } ( y \mid z ) } \bigg ) d y \right) d z } & { } \\ { = \mathbb { E } _ { Z \sim P _ { 0 } ^ { 1 , z } } \left( K ( P _ { 0 } ^ { 1 } ( \cdot \mid Z ) \| P _ { l } ^ { 1 } ( \cdot \mid Z ) ) \right) . } \end{array}
$$

Combining these two results yields the desired property.

## D.3 Condition 3. of the proof

Proposition D.3. Consider d defined by Equation $( 8 4 )$ , the test functions given by Equation (81), the semi-norm described by Equation (28). Then,

$$
\| \beta _ { w ^ { ( l ) } } - \beta _ { w ^ { ( l ^ { \prime } ) } } \| _ { \Gamma } ^ { 2 } \geq 2 C _ { 4 } n ^ { - \frac { 2 a + 2 k } { 2 a + 2 k + 1 } } ,
$$

where $C _ { 4 }$ is a positive constant which depends on $k , L , c ^ { \prime } , \sigma$

Proof. Using the definition of Γ given in (17) we find that

$$
\begin{array} { r l } {  { \big \| \beta _ { w ^ { ( t ) } } - \beta _ { w ^ { ( t ^ { \prime } ) } } \big \| _ { \Gamma } ^ { 2 } = \langle \Gamma \big ( \beta _ { w ^ { ( t ) } } - \beta _ { w ^ { ( t ^ { \prime } ) } } \big ) , \beta _ { w ^ { ( t ) } } - \beta _ { w ^ { ( t ^ { \prime } ) } } \rangle _ { L ^ { 2 } } } } \\ & { = \big \langle \displaystyle \sum _ { j = 1 } ^ { d } u _ { j } ( w _ { j } ^ { ( l ) } - w _ { j } ^ { ( t ^ { \prime } ) } ) \Gamma \phi _ { j } \displaystyle \sum _ { k = 1 } ^ { d } u _ { k } \big ( w _ { k } ^ { ( l ) } - w _ { k } ^ { ( l ^ { \prime } ) } \big ) \phi _ { k } \big \rangle _ { L ^ { 2 } } } \\ & { = \displaystyle \sum _ { j , k = 1 } ^ { d } u _ { j } u _ { k } \big ( w _ { j } ^ { ( l ) } - w _ { j } ^ { ( l ^ { \prime } ) } \big ) ( w _ { k } ^ { ( l ) } - w _ { k } ^ { ( l ^ { \prime } ) } ) \langle \Gamma \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } } \\ & { = \displaystyle \sum _ { j = 1 } ^ { d } \lambda _ { j } \big ( w _ { j } ^ { ( l ) } - w _ { j } ^ { ( l ^ { \prime } ) } \big ) ^ { 2 } u _ { j } ^ { 2 } , } \end{array}\tag{99}
$$

as the $\lambda _ { j }$ ’s are eigenvalues of Γ associated to the eigenvectors $\phi _ { j } \mathrm { ^ { \circ } s }$ . Now using the definition of $u _ { j }$ given by (82) we get

$$
\| \beta _ { w ^ { ( l ) } } - \beta _ { w ^ { ( l ^ { \prime } ) } } \| _ { \Gamma } ^ { 2 } \geq \frac { C } { n } \sum _ { j = 1 } ^ { d } ( w _ { j } ^ { ( l ) } - w _ { j } ^ { ( l ^ { \prime } ) } ) ^ { 2 } .
$$

Now, as

$$
\sum _ { j = 1 } ^ { d } ( w _ { j } ^ { ( l ) } - w _ { j } ^ { ( l ^ { \prime } ) } ) ^ { 2 } = \sum _ { j = 1 } ^ { d } \mathbb { 1 } _ { \{ w _ { j } ^ { ( l ) } \neq w _ { j } ^ { ( l ^ { \prime } ) } \} } ,
$$

we can use the Varshamov–Gilbert bound (Lemma 2.9 of Tsybakov [15]) and get that

$$
\Vert \beta _ { w ^ { ( l ) } } - \beta _ { w ^ { ( l ^ { \prime } ) } } \Vert _ { \Gamma } ^ { 2 } \geq \frac { C d } { 8 n } .
$$

Using the definition of d we find that

$$
\begin{array} { r } { \| \beta _ { w ^ { ( l ) } } - \beta _ { w ^ { ( l ^ { \prime } ) } } \| _ { \Gamma } ^ { 2 } \geq 2 C _ { 4 } n ^ { - \frac { 2 a + 2 k } { 2 a + 2 k + 1 } } , } \end{array}
$$

where $\begin{array} { r } { C _ { 4 } = \frac { C \kappa } { 1 6 } } \end{array}$ which is the desired result.

## E Orthogonal projection of $\beta$ onto $S _ { m }$ with respect to $\langle \cdot , \cdot \rangle _ { \Gamma }$ and $\langle \cdot , \cdot \rangle _ { \widetilde { \Gamma } }$

As $\beta$ is in $L ^ { 2 } ( [ 0 , 1 ] )$ and the Fourier basis forms a basis of this space, we can write

$$
\beta = \sum _ { j \geq 1 } \beta _ { j } \phi _ { j } ,
$$

where $\beta _ { j } = \langle \beta , \phi _ { j } \rangle _ { L ^ { 2 } }$ . We first determine $\beta _ { \Gamma } ^ { ( m ) }$ the orthogonal projection of $\beta$ onto $S _ { m }$ with respect to $\langle \cdot , \cdot \rangle _ { \Gamma }$ . Using the definition of $S _ { m }$ defined by equation (5) we have that $( \phi _ { 1 } , \phi _ { 2 } , \ldots , \phi _ { D _ { m } } )$ form a basis of $S _ { m }$ . By definition of the orthogonal projection $\beta _ { \Gamma } ^ { ( m ) }$ is in $S _ { m }$ , there exist real numbers $\mu _ { 1 } , \mu _ { 2 } , \ldots , \mu _ { D _ { m } }$ , such that

$$
\beta _ { \Gamma } ^ { ( m ) } = \sum _ { j = 1 } ^ { D _ { m } } \mu _ { j } \phi _ { j } ,
$$

and for the same argument we have that $\beta - \beta _ { \Gamma } ^ { ( m ) } \in S _ { m } ^ { \perp } . \mathrm { \ A s \ } ( \phi _ { 1 } , \phi _ { 2 } , \dots , \phi _ { D _ { m } } )$ is a basis of $S _ { m }$ this implies that for all $k = 1 , \ldots , D _ { m }$

$$
\langle \beta - \beta _ { \Gamma } ^ { ( m ) } , \phi _ { k } \rangle _ { \Gamma } = 0 .
$$

Using the fact that for all $j , k = 1 , \ldots , D _ { m }$

$$
\langle \phi _ { j } , \phi _ { k } \rangle _ { \Gamma } = \langle \Gamma \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } = \langle \lambda _ { j } \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } = \lambda _ { j } \delta _ { j k } ,
$$

with $\lambda _ { j } > 0$ for all $j = 1 , \ldots , D _ { m }$ , and Fubini’s theorem to justify the following computation, we obtain

$$
\begin{array} { r l } { \langle \beta - \beta _ { \Gamma } ^ { ( m ) } , \phi _ { k } \rangle _ { \Gamma } = \langle \displaystyle \sum _ { j > D _ { m } } \beta _ { j } \Gamma \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } + \langle \displaystyle \sum _ { j = 1 } ^ { D _ { m } } ( \beta _ { j } - \mu _ { j } ) \Gamma \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } } & { = \lambda _ { k } ( \beta _ { k } - \mu _ { k } ) . } \end{array}
$$

Therefore, $\langle \beta - \beta _ { \Gamma } ^ { ( m ) } , \phi _ { k } \rangle _ { \Gamma } = 0 \Longleftrightarrow \lambda _ { k } ( \beta _ { k } - \mu _ { k } ) = 0$ for all $k = 1 , \ldots , D _ { m }$ . And as $\lambda _ { k } > 0$ we find that $\langle \beta - \beta _ { \Gamma } ^ { ( m ) } , \phi _ { k } \rangle _ { \Gamma } = 0 \iff \beta _ { k } = \mu _ { k }$ . Hence, $\begin{array} { r } { \beta _ { \Gamma } ^ { ( m ) } = \sum _ { j = 1 } ^ { D _ { m } } \beta _ { j } \phi _ { j } } \end{array}$ . We proceed in a similar way to determine $\beta _ { \widetilde { \Gamma } } ^ { ( m ) }$ the orthogonal projection of $\beta$ onto $S _ { m }$ with respect to $\langle \cdot , \cdot \rangle _ { \widetilde { \Gamma } }$ . By the definition of the orthogonal projection, we must have $\beta _ { \widetilde { \Gamma } } ^ { ( m ) } \in S _ { m }$ , therefore there exist $\widetilde { \mu } _ { 1 } , \ldots , \widetilde { \mu } _ { D _ { m } }$ such that

$$
\beta _ { \widetilde { \Gamma } } ^ { ( m ) } = \sum _ { j = 1 } ^ { D _ { m } } \widetilde { \mu } _ { j } \phi _ { j } .
$$

We also have that for all $k = 1 , \ldots , D _ { m }$

$$
\beta - \beta _ { \widetilde { \Gamma } } ^ { ( m ) } = 0 .
$$

Using Lemma B.5 and Fubini’s theorem we obtain,

$$
\langle \beta - \beta _ { \widetilde { \Gamma } } ^ { ( m ) } , \phi _ { k } \rangle _ { \widetilde { \Gamma } } = \langle \sum _ { j > D _ { m } } \beta _ { j } \widetilde { \Gamma } \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } + \langle \sum _ { j = 1 } ^ { D _ { m } } ( \beta _ { j } - \widetilde { \mu } _ { j } ) \widetilde { \Gamma } \phi _ { j } , \phi _ { k } \rangle _ { L ^ { 2 } } \quad = \widetilde { \lambda } _ { k } ( \beta _ { k } - \widetilde { \mu } _ { k } ) .
$$

Hence, $\langle \beta - \beta _ { \widetilde { \Gamma } } ^ { ( m ) } , \phi _ { k } \rangle _ { \widetilde { \Gamma } } = 0 \iff \lambda _ { k } ( \beta _ { k } - \widetilde { \mu } _ { k } ) = 0$ for all $k = 1 , \ldots , D _ { m }$ . And as $\widetilde { \lambda } _ { k } > 0$ we find that $\langle \beta - \beta _ { \widetilde { \Gamma } } ^ { ( m ) } , \phi _ { k } \rangle _ { \widetilde { \Gamma } } = 0 \iff \beta _ { k } = \widetilde { \mu } _ { k }$ . Therefore, $\begin{array} { r } { \beta _ { \widetilde { \Gamma } } ^ { ( m ) } = \sum _ { j = 1 } ^ { D _ { m } } \beta _ { j } \phi _ { j } } \end{array}$ and $\beta _ { \widetilde { \Gamma } } ^ { ( m ) } = \beta _ { \Gamma } ^ { ( m ) }$

## Acknowledgments

I am deeply grateful to my supervisors, Gaëlle Chagny, Vincent Rivoirard and Angelina Roche, for their insightful comments and constructive feedback.

The author also acknowledge the support of the French Agence Nationale de la Recherche (ANR) under reference ANR-24-CE40-2439 (FUNMathStat project).

## References

[1] Yannick Baraud. “Model Selection for Regression on a Fixed Design”. In: Probability Theory and Related Fields 117 (2000), pp. 467–493. doi: 10.1007/PL00008731.

[2] Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration Inequalities: A Nonasymptotic Theory ofIndependence. Oxford, UK: Oxford University Press, 2013. isbn: 9780199535255. doi: 10.1093/acprof:oso/9780199535255.001.0001.

[3] Elodie Brunel, André Mas, and Angelina Roche. “Non-Asymptotic Adaptive Prediction in Functional Linear Models”. In: Journal of Multivariate Analysis 143 (2016), pp. 208–232. issn: 0047-259X. doi: 10.1016/j.jmva.2015.09.008.

[4] Elodie Brunel and Angelina Roche. “Penalized Contrast Estimation in Functional Linear Models with Circular Data”. In: Statistics 49.6 (2015), pp. 1298–1321. doi: 10 . 1080 / 02331888.2014.993986.

[5] Hervé Cardot, Frédéric Ferraty, and Pascal Sarda. “Functional Linear Model”. In: Statistics & Probability Letters 45.1 (1999), pp. 11–22. doi: 10.1016/S0167-7152(99)00036-X.

[6] Hervé Cardot and Jan Johannes. “Thresholding Projection Estimators in Functional Linear Models”. In: Journal of Multivariate Analysis 101.2 (2010), pp. 395–408. issn: 0047-259X. doi: 10.1016/j.jmva.2009.03.001.

[7] Hervé Cardot et al. “Smoothing Splines Estimators in Functional Linear Regression with Errors-in-Variables”. In: Computational Statistics & Data Analysis 51.10 (2007).

[8] F. Comte and J. Johannes. “Adaptive Estimation in Circular Functional Linear Models”. In: Mathematical Methods of Statistics 19.1 (2010), pp. 42–63.

[9] S. Dieng et al. “Application of Functional Data Analysis to Identify Patterns of Malaria Incidence, to Guide Targeted Control Strategies”. In: International Journal of Environmental Research and Public Health 17.11 (2020), p. 4168. doi: 10.3390/ijerph17114168.

[10] K. F. Frøslie, J. Røislien, E. Qvigstad, et al. “Shape Information from Glucose Curves: Functional Data Analysis Compared with Traditional Summary Measures”. In: BMC Medical Research Methodology 13.6 (2013). doi: 10.1186/1471-2288-13-6.

[11] Paul-Marie Grollemund et al. “Bayesian Functional Linear Regression with Sparse Step Functions”. In: Bayesian Analysis 14.1 (2019), pp. 111–135. doi: 10.1214/18-BA1095.

[12] Gareth M. James, Jing Wang, and Ji Zhu. “Functional Linear Regression That’s Interpretable”. In: The Annals of Statistics 37.5A (2009), pp. 2083–2108. doi: 10 . 1214 / 08 - AOS641.

[13] Piotr Kokoszka, Hong Miao, and Ben Zheng. “Testing for Asymmetry in Betas of Cumulative Returns: Impact of the Financial Crisis and Crude Oil Price”. In: Statistics & Risk Modeling 34.1–2 (2017), pp. 33–53. doi: 10.1515/strm-2016-0010.

[14] J. O. Ramsay and B. W. Silverman. Functional Data Analysis. 2nd ed. Springer Series in Statistics. Springer, 2005.

[15] Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. New York, NY: Springer, 2009. isbn: 978-0-387-79051-0. doi: 10.1007/b13794.

[16] Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. 2nd ed. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge: Cambridge University Press, 2026.