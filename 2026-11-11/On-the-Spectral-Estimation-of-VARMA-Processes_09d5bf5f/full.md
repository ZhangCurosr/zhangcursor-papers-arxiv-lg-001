# On the Spectral Estimation of VARMA Processes

Bin Zhu and Mattia Zorz

Abstract—We consider the problem of estimating the power spectral density of a Vector-valued Autoregressive Moving-Average (VARMA) process. A well-established approach to accomplishing this task consists of formulating a moment problem that selects the spectral density maximizing an entropy-like index while matching certain moments of the underlying process. However, existing methods assume that the moving-average (MA) component is either fixed or scalar, thereby reducing the class of ARMA models over which the optimal spectral density is sought. In this paper, we propose a new moment problem, based on the notion of transportation entropy, for learning the spectral density of a VARMA process without fixing the MA component or restricting it to be scalar.

Index Terms—Spectrum approximation, moment problem, transportation entropy, W-cepstral coefficients, convex optimization.

## I. INTRODUCTION

Autoregressive Moving-Average (ARMA) processes have wide applications in different domains, including systems and control [1], [2], signal processing [3], time series analysis [4], and finance [5]. From the perspective of identification and control, these processes have rational spectra which can approximate any continuous spectral density if the order of the model is sufficiently large. Moreover, standard realization theory can convert the ARMA model into a state-space form which is convenient, e.g., for the purpose of filtering.

It is often the case that the spectral density of ARMA processes must be identified from data. For scalar processes, a well-posed problem of jointly estimating the AR and MA parameters was formulated in [6], [7] as maximizing the entropy of the process with equality constraints of covariance and cepstral matching (i.e., the moment equations). The sample version of the latter can readily be obtained from a finite-length realization of the underlying process. It was shown that the correspondence between the covariances and the AR parameters is a smooth bijection, and the same holds for the correspondence between the cepstral coefficients and the MA parameters. The cepstral coefficients are also called logarithmic moments because they are the Fourier coefficients of log Φ where Φ (assumed to be positive) denotes the spectral density of the process.

For vector-valued ARMA (VARMA) processes, the model parameters become matrices, and the estimation problem becomes more challenging. A number of attempts have been made to extend the Maximum Entropy framework to the multivariate case. The first setup considered the case where the MA part is fixed a priori, and certain divergence indices are minimized subject to matricial covariance constraints; see, e.g., [8]–[10]. However, if we want to jointly estimate the matricial AR and MA parameters from the data, the standard Maximum Entropy method [11] works only with scalar logarithmic moments [1], [12], [13], bringing about VARMA models with scalar MA parts, i.e., the MA parameters have the form of a scalar times the identity matrix. We remark that the scalar logarithmic moments of a vector-valued process with a matricial spectral density Φ (assumed to be positive definite) were first defined in [14] as the Fourier coefficients of log det Φ. These quantities were utilized in the spectral estimation of VARMA process in later works that are cited above. It is worth noting that assuming the MA component to be fixed or scalar reduces the class of ARMA models over which the optimal spectral density is sought, potentially resulting in a spectral estimator that is not always satisfactory.

One may be tempted to use the matricial logarithm for positive definite matrices to define a kind of matricial cepstral coefficients and hope to generalize the multivariate spectral estimation method with the scalar logarithmic moments. As we will see, however, it is not possible to go through the Lagrange dual analysis of the corresponding Maximum Entropy problem, because the optimal primal variable (which is a function) cannot be expressed in terms of the dual variable. Notice that a successful dual analysis is the key to obtaining a finite-dimensional dual problem, which serves as a practical tool for estimating the parameters characterizing the spectral density. The same happens if we consider the more general notion of ν-entropy (parametrized by a positive integer ν) proposed in [15], [16] for the scalar case.

Hence, the major obstacle in estimating a VARMA process by means of a moment problem is to find a suitable definition of the cepstral coefficients for a vector-valued process, and this is precisely the motivation behind the current work. Inspired by recent results on optimal transportation between Gaussian processes [8], [17], [18], we define a new concept of matricial cepstral coefficients using a spectral factor W of the spectral density of the process. For this reason, they are termed “Wcepstral coefficients”. These coefficients are naturally compatible with the transportation entropy derived from the transportation (Hellinger) distance between two spectral densities. We show that this new Maximum Entropy formulation is amenable to Lagrange dual analysis, and the dual problem becomes well-posed after incorporating an appropriate design of regularization. Then we derive a spectral estimator, based on the proposed Maximum Entropy problem, which uses only a finite sample path of the underlying process. The advantage of our VARMA spectral estimator is illustrated by numerical experiments.

The rest of the paper is organized as follows. Sec. II introduces the VARMA spectral estimation problem and two possible Maximum Entropy formulations, and explains the difficulty. Sec. III introduces the transportation entropy and the W-cepstral coefficients and formulates the new Maximum Entropy problem. A detailed dual analysis is carried out. Sec. IV regularizes the dual optimization problem and investigates its well-posedness property. Sec. V provides a spectral estimation procedure based on the regularized dual problem and a finite sample path of a random process, with a discussion of the statistical consistency of the estimator. Sec. VI illustrates our VARMA spectral estimator in comparison with some alternative ones through numerical simulations. Sec. VII concludes the paper.

Notation. Let $\mathbb { T } : = [ 0 , 2 \pi )$ be the interval for normalized angular frequencies which is isomorphic to the unit circle $\{ z \in$ $\mathbb { C } : | z | = \bar { 1 } \} = \{ e ^ { i \vartheta } : \vartheta \in \mathbb { T } \}$ in the complex plane. The integral of a function f on T with respect to the normalized Lebesgue measure, namely $\textstyle \int _ { \mathbb { T } } f ( \vartheta ) { \frac { \mathrm { d } \vartheta } { 2 \pi } }$ , is often abbreviated as $\textstyle \int f .$ The symbol $\mathbb { L } _ { m \times n } ^ { p } ( \mathbb { T } )$ Rwith $0 < p \le \infty$ represents the vector space of $m \times n$ complex matrix-valued functions on T whose entries are complex-valued functions in $\mathbb { L } ^ { p } ( \mathbb { T } )$ . The set $\mathcal { S } _ { m , + }$ consists of (Hermitian) positive semidefinite matrixvalued functions in $\mathbb { L } _ { m \times m } ^ { 1 } ( \mathbb { T } )$

## II. VARMA SPECTRUM APPROXIMATION

We consider the problem of finding a power spectral density $\Phi \in S _ { m , + }$ describing a vector-valued ARMA Gaussian process assuming that some moments of its spectrum are known. In the univariate case, i.e., $m = 1$ , a solution can be obtained from the maximum entropy problem [7]:

$$
\begin{array} { l } { { \displaystyle \operatorname* { m a x } _ { \Phi \in { \cal S } _ { m , + } } \int \log \operatorname* { d e t } \Phi } } \\ { { \mathrm { s . t . } \int e ^ { i \vartheta k } \Phi = R _ { k } , \quad k = 0 , \ldots , n } } \\ { { \displaystyle \int e ^ { i \vartheta k } \log \Phi = C _ { k } , \quad k = 1 , \ldots , n } } \end{array}\tag{1}
$$

where the objective function denotes the entropy rate [19] of the Gaussian process with a spectrum Φ, and in the moment constraints $R _ { k } \in \mathbb { R } ^ { m \times m }$ , with $k = 0 \dots n$ , and $C _ { k } \in \mathbb { R } ^ { m \times m }$ with $k = 1 \ldots n .$ , are the covariance lags and the cepstral coefficients of the spectrum, respectively.

In the multivariate case, i.e., $m > 1$ , we can still consider (1) where log Φ denotes the matricial logarithm which is well defined when Φ is positive definite pointwise for each $\vartheta \in \mathbb { T } .$ In that case, Φ admits a spectral decomposition $\Phi = X \Lambda X ^ { * }$ in which X is a unitary matrix of eigenvectors and Λ is a diagonal matrix of positive real eigenvalues, both of which are functions of ϑ. Then log $\Phi : = X ( \log \Lambda ) X ^ { * }$ where log Λ is a diagonal matrix obtained by taking the logarithm of each diagonal entry of Λ. As a first attempt, we try to solve Problem (1). Let Q be an element of the vector space

$$
\mathfrak { B } : = \left\{ \{ Q _ { k } \} _ { k = 0 } ^ { n } : Q _ { k } \in \mathbb { R } ^ { m \times m } \mathrm { ~ a n d ~ } Q _ { 0 } ^ { T } = Q _ { 0 } \right\}\tag{2}
$$

and P an element of

$$
\mathfrak { B } _ { o } : = \left\{ \{ P _ { k } \} _ { k = 1 } ^ { n } : P _ { k } \in \mathbb { R } ^ { m \times m } \right\} .\tag{3}
$$

The Lagrangian of (1) is

$$
\begin{array} { l } { { \displaystyle { \cal L } ( \Phi , { \bf P } , { \bf Q } ) = \int \log \operatorname * { d e t } \Phi - \mathrm { t r } \left[ \sum _ { k = 0 } ^ { n } Q _ { k } ^ { T } \left( \int e ^ { i \vartheta k } \Phi - R _ { k } \right) \right] } \ ~ } \\ { { \displaystyle ~ + \mathrm { t r } \left[ \sum _ { k = 1 } ^ { n } P _ { k } ^ { T } \left( \int e ^ { i \vartheta k } \log \Phi - C _ { k } \right) \right] } \ ~ } \\ { { \displaystyle ~ = \mathrm { t r } \left( \int P \log \Phi - Q \Phi \right) + \mathrm { t r } \sum _ { k = 0 } ^ { n } Q _ { k } ^ { T } R _ { k } - \mathrm { t r } \sum _ { k = 1 } ^ { n } P _ { k } ^ { T } C _ { k } } } \end{array}
$$

where

$$
\begin{array} { r } { Q ( e ^ { i \vartheta } ) : = Q _ { 0 } + \displaystyle \frac { 1 } { 2 } \sum _ { k = 1 } ^ { n } \left( Q _ { k } e ^ { - i \vartheta k } + Q _ { k } ^ { T } e ^ { i \vartheta k } \right) } \\ { \mathrm { ~ a n d ~ } P ( e ^ { i \vartheta } ) : = I + \displaystyle \frac { 1 } { 2 } \sum _ { k = 1 } ^ { n } \left( P _ { k } e ^ { - i \vartheta k } + P _ { k } ^ { T } e ^ { i \vartheta k } \right) } \end{array}\tag{4}
$$

are polynomial matrices constructed from Q and P, respectively. Under the assumption that P is positive definite on T, it is not difficult to see that $L ( \cdot , { \bf P } , { \bf Q } )$ is concave, and the arguments are similar to those in [20]. Then the unconstrained maximizer is also a stationary point, i.e., it is obtained by setting equal to zero the first variation of L along the direction $\delta \Phi \in \mathbf { \bar { L } } _ { m \times m } ^ { 1 } ( \mathbb { T } )$

$$
\begin{array} { l } { { \displaystyle \delta L ( \Phi , { \bf P } , { \bf Q } ; \delta \Phi ) = \mathrm { t r } \int \left[ P \int _ { 0 } ^ { \infty } ( \Phi + t I ) ^ { - 1 } \right. } } \\ { { \displaystyle \left. \times \delta \Phi ( \Phi + t I ) ^ { - 1 } \mathrm { d } t - Q \delta \Phi \right] } } \end{array}
$$

where we exploited [10, Proposition A.2]. Accordingly, the stationary point satisfies the condition

$$
\int _ { 0 } ^ { \infty } ( \Phi + t I ) ^ { - 1 } P ( \Phi + t I ) ^ { - 1 } \mathrm { d } t { = } Q .\tag{5}
$$

It is clear that it is not possible to find a closed-form expression for Φ in terms of the Lagrange multipliers, and thus we cannot carry out the dual analysis.

Remark 1. If, in Problem (1), we replace log Φ with log det Φ (and thus also $C _ { k } \in \mathbb { R } ^ { m \times m }$ with $c _ { k } \in \mathbb { R } )$ , i.e., we consider the scalar logarithmic moments of a vector-valued process defined in [14], then the matricial polynomial P defined in (4) is replaced with a scalar trigonometric polynomial

$$
p ( e ^ { i \vartheta } ) = 1 + \sum _ { k = 1 } ^ { n } p _ { k } \cos ( \vartheta k ) , \quad p _ { k } \in \mathbb { R }\tag{6}
$$

and the stationarity condition in (5) reduces to

$$
p \Phi ^ { - 1 } = Q .
$$

Moreover, it is possible to show, through the dual problem, that the solution to the primal problem exists and is of the form [1]:

$$
\Phi ^ { \circ } = p Q ^ { - 1 } .\tag{7}
$$

However, the spectrum in (7) describes a special class of ARMA processes. Indeed, let $\begin{array} { r } { b ( z ) = \sum _ { k = 0 } ^ { n } b _ { k } z ^ { - k } , b _ { k } \in \mathbb { R } , } \end{array}$ and $\begin{array} { r } { A ( z ) = \sum _ { k = 0 } ^ { n } A _ { k } z ^ { - k } , ~ A _ { k } \in \mathbb { R } ^ { m \times m } } \end{array}$ , be the minimum Pphase spectral factors of $p$ and Q, i.e., $p ( z ) \ : = \ : b ( z ^ { - 1 } ) b ( z )$ and $Q ( z ) = A ( z ^ { - 1 } ) ^ { T } A ( z )$ . Then the process $\{ y _ { t } \} _ { t \in \mathbb { Z } }$ with a spectrum (7) admits the unilateral ARMA representation

$$
\sum _ { k = 0 } ^ { k } A _ { k } y _ { t - k } = \sum _ { k = 0 } ^ { n } b _ { k } e _ { t - k }\tag{8}
$$

where $\{ e _ { t } \} _ { t \in \mathbb { Z } }$ is a normalized white Gaussian noise. We conclude that such a process is very special, since its MA component is scalar, i.e., the coefficients $b _ { k }$ are scalar.

Another possible route is to consider a general definition of entropy, called ν-entropy, which is given in [15], [16] for the scalar case, i.e. $m = 1$ , with an integer parameter $\nu \geq 1$ . In the case of $\nu = 1$ , it reduces to the classic entropy in (1). Its extension to the matricial case $( m > 1 )$ with $\nu \geq 2$ is

$$
\frac { \nu ^ { 2 } } { \nu - 1 } \mathrm { t r } \int \Phi ^ { \frac { \nu - 1 } { \nu } } .
$$

Now in the moment problem, we can replace the entropy rate with the ν-entropy above, and the cepstral coefficients with the matricial ν-cespral coefficients, and obtain

$$
\begin{array} { l } { { \displaystyle { \operatorname* { m a x } _ { \Phi \in S _ { m , + } } \frac { \nu ^ { 2 } } { \nu - 1 } \mathrm { t r } \int \Phi ^ { \frac { \nu - 1 } { \nu } } } } } \\ { { \mathrm { s . t . } \ \displaystyle { \int e ^ { i \vartheta k } \Phi = R _ { k } , } \quad k = 0 , \ldots , n } } \\ { { \displaystyle { \frac { \nu ^ { 2 } } { \nu - 1 } \int e ^ { i \vartheta k } \Phi ^ { \frac { \nu - 1 } { \nu } } = C _ { k } , \quad k = 1 , \ldots , n . } } } \end{array}\tag{9}
$$

The notion of matricial ν-cespral coefficient is an extension of the one given for scalar processes $( m = 1 )$ in [15]. More precisely, the matricial power function with an exponent $\alpha >$ 0 is defined in a way similar to the matricial logarithm. In fact, we let $\Phi ^ { \alpha } : = \dot { X } \Lambda ^ { \alpha } X ^ { * }$ where X and Λ come from the (pointwise) spectral decomposition $\Phi = X \Lambda X ^ { * }$ and the power function $( \cdot ) ^ { \alpha }$ is applied to each (nonnegative) eigenvalue on the diagonal of Λ. The Lagrangian of (9) is

$$
\begin{array} { r l } & { L ( \Phi , { \bf P } , { \bf Q } ) = \frac { \nu ^ { 2 } } { \nu - 1 } \mathrm { t r } \int \Phi ^ { \frac { \nu - 1 } { \nu } } } \\ & { \qquad - \mathrm { t r } \left[ \displaystyle \sum _ { k = 0 } ^ { n } Q _ { k } ^ { T } \left( \int e ^ { i \vartheta k } \Phi - R _ { k } \right) \right] } \\ & { \qquad + \mathrm { t r } \left[ \displaystyle \sum _ { k = 1 } ^ { n } P _ { k } ^ { T } \left( \frac { \nu ^ { 2 } } { \nu - 1 } \int e ^ { i \vartheta k } \Phi ^ { \frac { \nu - 1 } { \nu } } - C _ { k } \right) \right] } \\ & { \qquad = \mathrm { t r } \left( \displaystyle \int \frac { \nu ^ { 2 } } { \nu - 1 } P \Phi ^ { \frac { \nu - 1 } { \nu } } - Q \Phi \right) + \mathrm { t r } \displaystyle \sum _ { k = 0 } ^ { n } Q _ { k } ^ { T } R _ { k } - \mathrm { t r } \displaystyle \sum _ { k = 1 } ^ { n } P _ { k } ^ { T } C _ { k } } \end{array}
$$

where $\mathbf { P } \in \mathfrak { B } _ { o }$ and $\mathbf { Q } \in \mathfrak { B }$ . We associate the polynomial matrices $P$ and Q, defined in (4), to P and Q, respectively. Under the assumption that P is positive definite on T, it can be shown that $L ( \cdot , { \bf P } , { \bf Q } )$ is concave [10], [21]. Then, the unconstrained maximizer is also a stationary point, i.e., it is obtained by setting equal to zero the first variation of L along the direction $\delta \Phi \in \mathbb { L } _ { m \times m } ^ { \bar { 1 } } ( \mathbb { T } )$ :

$$
\begin{array} { l } { { \displaystyle \delta { \cal L } ( \Phi , { \bf P } , { \bf Q } ; \delta \Phi ) = \mathrm { t r } \int \left[ \nu P \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { \infty } \Phi ^ { \frac { \nu - 1 } { \nu } ( 1 - \tau ) } ( \Phi + t I ) ^ { - 1 } \right. } } \\ { { \displaystyle \left. \times \delta \Phi ( \Phi + t I ) ^ { - 1 } \Phi ^ { \frac { \nu - 1 } { \nu } \tau } \mathrm { d } t \mathrm { d } \tau - Q \delta \Phi \right] } } \end{array}
$$

where we exploited [10, Proposition A.3]. Accordingly, the stationary point satisfies the condition

$$
\nu \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { \infty } ( \Phi + t I ) ^ { - 1 } \Phi ^ { \frac { \nu - 1 } { \nu } \tau } P \Phi ^ { \frac { \nu - 1 } { \nu } ( 1 - \tau ) } ( \Phi + t I ) ^ { - 1 } \mathrm { d } t \mathrm { d } \tau = Q .
$$

Also in this case it is not possible to find a closed-form expression for Φ in terms of the Lagrange multipliers and thus we cannot continue the dual analysis.

Remark 2. One could replace the matricial ν-cespral coefficient with its scalar version for vector-valued processes, $\mathrm { i . e . , }$ we replace $\Phi ^ { \frac { \nu - 1 } { \nu } }$ with $\operatorname { t r } [ \Phi ^ { \frac { \nu - 1 } { \nu } } ]$ (and thus also $C _ { k } \in \mathbb { R } ^ { n }$ ×m with $c _ { k } \in \mathbb { R } )$ . The Lagrangian in this case reduces to

$$
\mathrm { t r } \left( \int \frac { \nu ^ { 2 } } { \nu - 1 } p \Phi ^ { \frac { \nu - 1 } { \nu } } - Q \Phi \right) + \mathrm { t r } \sum _ { k = 0 } ^ { n } Q _ { k } ^ { T } R _ { k } - \mathrm { t r } \sum _ { k = 1 } ^ { n } p _ { k } c _ { k }
$$

where the scalar polynomial $p$ is defined as in (6). Under the assumption that p is positive on T, it is not difficult to prove that the Lagrangian above is concave with respect to Φ and thus its unconstrained maximum is obtained by imposing the stationarity condition, which turns out to be

$$
\int ( \nu p \Phi ^ { - \frac { 1 } { \nu } } - Q ) \delta \Phi = 0 , \quad \forall \delta \Phi \in \mathbb { L } _ { m \times m } ^ { 1 } ( \mathbb { T } ) .
$$

Under the assumption that $Q$ is positive definite on T we have the optimal form of the primal variable

$$
\Phi ^ { \circ } = \nu ^ { \nu } p ^ { \nu } Q ^ { - \nu } .\tag{10}
$$

Thus, the spectrum in (10) describes a special class of ARMA processes which admit a time-domain representation similar to the one in (8).

## III. OPTIMAL TRANSPORT FORMULATION

It is worth noting that an equivalent formulation of problem (1) is to minimize $\mathbf { D } _ { \mathrm { I S } } ( \Phi , I )$ under the moment constraints and

$$
{ \bf D } _ { \mathrm { I S } } ( \Phi , \Psi ) = \int - \log \operatorname * { d e t } \Phi + \log \operatorname * { d e t } \Psi + \mathrm { t r } ( \Phi \Psi ^ { - 1 } ) - m\tag{11}
$$

denotes the Itakura–Saito distance between the spectral densities $\Phi$ and Ψ, [9]. Indeed,

$$
{ \bf D } _ { \mathrm { I S } } ( \Phi , I ) = \int - \log \operatorname * { d e t } \Phi + \mathrm { t r } ( \Phi ) - m ,\tag{12}
$$

and the last two terms are constant over the feasible set characterized by the moment constraints. Similarly, an equivalent formulation of Problem (9) is to minimize $\mathbf { D } _ { \beta } ( \Phi , I )$ , with $\beta = 1 - \nu ^ { - 1 }$ , under the moment constraints and

$$
\mathbf { D } _ { \beta } ( \Phi , \Psi ) = \operatorname { t r } \int { \frac { 1 } { \beta - 1 } } ( \Phi ^ { \beta } - \Phi \Psi ^ { \beta - 1 } ) - { \frac { 1 } { \beta } } ( \Phi ^ { \beta } - \Psi ^ { \beta } )
$$

denotes the β-divergence between the spectral densities Φ and Ψ [10]. Drawing inspiration from that, we set up the moment

problem using the optimal transportation between Gaussian stationary processes [18] that has been previously introduced in [8]:

$$
\begin{array} { l } { { { \bf D } _ { \mathrm { T D } } ( \Phi , \Psi ) ^ { 2 } = \displaystyle \operatorname* { m i n } _ { W _ { \Phi } , W _ { \Psi } \in \mathbb { L } _ { m \times m } ^ { 2 } ( \mathbb { T } ) } \mathrm { t r } \displaystyle \int ( W _ { \Phi } - W _ { \Psi } ) ( W _ { \Phi } - W _ { \Psi } ) ^ { * } } } \\ { { \mathrm { s . t . } W _ { \Phi } W _ { \Phi } ^ { * } = \Phi } } \\ { { W _ { \Psi } W _ { \Psi } ^ { * } = \Psi } } \end{array}
$$

where $W _ { \Phi }$ and W<sub>Ψ</sub> represent a left spectral factor of Φ and Ψ, respectively. Then we define the transportation entropy as

$$
\begin{array} { r l } & { \mathbf { H } _ { \mathrm { T } } ( \Phi ) : = - \left( \mathbf { D } _ { \mathrm { T D } } ( \Phi , I ) ^ { 2 } - \mathrm { t r } \int \Phi - m \right) } \\ & { \qquad = \underset { W \in \mathbb { L } _ { m \times m } ^ { 2 } ( \mathbb { T } ) } { \mathrm { m a x } } \mathrm { t r } \int W + W ^ { * } } \\ & { \qquad \mathrm { s . t . ~ } W W ^ { * } = \Phi . } \end{array}\tag{14}
$$

Notice that the term $- \operatorname { t r } \int \Phi - m$ can be treated as a constant Rin the constrained maximization problem above, and we added it only to simplify the subsequent analysis. Indeed, if we define another functional $\tilde { \bf H } _ { \mathrm { T } } ( \Phi ) : = - { \bf D } _ { \mathrm { T D } } ( \Phi , I ) ^ { 2 }$ , then for each fixed Φ, the corresponding maximization problem will admit an optimal spectral factor $W ^ { \circ }$ that is identical to the maximizer of (14). The maximum values of the two objective functionals, however, will clearly be different.

Before formulating the problem, we need to define the cepstral coefficients.

Definition 1. Let W be a left spectral factor of the spectral density Φ. We define as the k-th matricial W-cepstral coefficient of Φ the matrix

$$
C _ { k } = \frac { 1 } { 2 } \int e ^ { i \vartheta k } ( W + W ^ { \ast } ) , \quad k \in \mathbb { Z } .\tag{15}
$$

It is clear that the definition above depends on the particular spectral factor. If we take $W = \Phi ^ { 1 / 2 }$ , then we have

$$
C _ { k } = \int e ^ { i \vartheta k } \Phi ^ { 1 / 2 }\tag{16}
$$

which coincides (up to a constant factor) with the ν-cepstral coefficient in (9) with $\nu = 2$

Thus, we obtain the moment problem:

$$
\begin{array} { l } { \displaystyle { \operatorname* { m a x } _ { { \bf { m } } \times { \bf { m } } ( { \mathbb { T } } ) } { \bf { H } } _ { \mathrm { { T } } } ( \Phi ) } } \\ { \displaystyle { \mathfrak { W } \in } \mathbb { E } _ { m \times m } ^ { 1 } ( { \mathbb { T } } ) } \\ { \displaystyle \quad \mathrm { s . t . } \ \Phi = W W ^ { * } } \\ { \displaystyle \quad \displaystyle \int e ^ { i \vartheta k } \Phi = R _ { k } , \quad k = 0 , \ldots , n } \\ { \displaystyle \displaystyle \frac { 1 } { 2 } \int e ^ { i \vartheta k } ( W + W ^ { * } ) = C _ { k } , \quad k = 1 , \ldots , n . } \end{array}\tag{17}
$$

Due to the presence of the constraint $\int \Phi \ = \ R _ { 0 }$ with $R _ { 0 }$ fixed, the term tr $\textstyle \int \Phi$ in (14) becomes a constant for the optimization problem, and we can use the objective functional

in the maximization problem of (14). In fact, the optimization problem (17) can be rewritten in terms of W as

$$
\begin{array} { r l } & { \quad \displaystyle \operatorname* { m a x } _ { W \in \mathbb { L } _ { m \times m } ^ { 2 } ( \mathbb { T } ) } \mathrm { t r } \int W + W ^ { * } } \\ & { \quad \mathrm { s . t . } \int e ^ { i \vartheta k } W W ^ { * } = R _ { k } , \quad k = 0 , \ldots , n } \\ & { \quad \quad \displaystyle \frac { 1 } { 2 } \int e ^ { i \vartheta k } ( W + W ^ { * } ) = C _ { k } , \quad k = 1 , \ldots , n . } \end{array}\tag{18}
$$

The Lagrangian of the above problem is

$$
\begin{array} { l } { { \displaystyle { \cal L } ( W , { \bf P } , { \bf Q } ) = \mathrm { t r } \int W + W ^ { * } } \ ~ } \\ { { \displaystyle ~ - \mathrm { t r } \left[ \sum _ { k = 0 } ^ { n } Q _ { k } ^ { r } \left( \int e ^ { i \phi k } W W ^ { * } - R _ { k } \right) \right] } \ ~ } \\ { { \displaystyle ~ + \mathrm { t r } \left[ \sum _ { k = 1 } ^ { n } P _ { k } ^ { r } \left( \int e ^ { i \phi k } ( W + W ^ { * } ) - 2 C _ { k } \right) \right] } \ ~ } \\ { { \displaystyle ~ = \mathrm { t r } \int [ ( W + W ^ { * } ) P - W W ^ { * } Q ] } \ ~ } \\ { { \displaystyle ~ + \mathrm { t r } \sum _ { k = 0 } ^ { n } Q _ { k } ^ { r } R _ { k } - 2 \mathrm { t r } \sum _ { k = 1 } ^ { n } P _ { k } ^ { r } C _ { k } } , } \end{array}\tag{19}
$$

where $( \mathbf { P } , \mathbf { Q } ) \in \mathfrak { B } _ { o } \times \mathfrak { B }$ are the Lagrange multipliers. Next, we consider the unconstrained maximization problem

$$
\operatorname* { s u p } _ { W \in \mathbb { L } _ { m \times m } ^ { 2 } ( \mathbb { T } ) } L ( W , \mathbf { P } , \mathbf { Q } ) .\tag{20}
$$

We first show that in order for the supremum above to be finite, the polynomial matrix $Q$ must be positive semidefinite everywhere on T. To this end, let us suppose the contrary, i.e., there exist $\bar { \vartheta } \in \mathbb { T }$ and a vector $v \in \mathbb { C } ^ { m }$ such that $\lVert \boldsymbol { v } \rVert = 1$ and $v ^ { * } Q ( e ^ { i \bar { \vartheta } } ) v < 0$ . Then, according to continuity, there exists a neighborhood of centered in ϑ<sup>¯</sup> such that

$$
v ^ { * } Q ( e ^ { i \vartheta } ) v < 0 \quad \forall \vartheta \in { \cal B } .\tag{21}
$$

We take the Hermitian spectral factor

$$
W _ { \alpha } ( e ^ { i \vartheta } ) : = \left\{ \begin{array} { l l } { \alpha \left| v ^ { * } Q ( e ^ { i \vartheta } ) v \right| v v ^ { * } , } & { \vartheta \in \mathcal { B } } \\ { O , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{22}
$$

with $\alpha \in \mathbb { R }$ . Plugging $W _ { \alpha }$ into the Lagrangian (19), we have

$$
\begin{array} { l } { { \displaystyle { \cal L } ( W _ { \alpha } , { \bf P } , { \bf Q } ) = \mathrm { t r } \int _ { \mathcal { B } } [ ( W _ { \alpha } + W _ { \alpha } ^ { * } ) P - W _ { \alpha } W _ { \alpha } ^ { * } Q ] + \kappa } } \\ { { \displaystyle ~ = \int _ { \mathcal { B } } \{ 2 \alpha v ^ { * } P ( e ^ { i \vartheta } ) v \left| v ^ { * } { \cal Q } ( e ^ { i \vartheta } ) v \right| - \alpha ^ { 2 } \underbrace { [ v ^ { * } { \cal Q } ( e ^ { i \vartheta } ) v ] ^ { 3 } } _ { < 0 } \} + \kappa } } \end{array}
$$

where κ denotes the terms that do not depend on $W _ { \alpha }$ . Thus, $L ( W _ { \alpha } , \mathbf { P } , \mathbf { Q } ) \to \infty \ \mathrm { a s } \ | \alpha | \to \infty$ . Therefore, the supremum in (20) is finite only if $\mathbf { Q } \in \overline { { \mathfrak { B } _ { + } } }$ where

$$
\mathfrak { B } _ { + } : = \{ \mathbf { Q } \in \mathfrak { B } : Q > 0 \mathrm { ~ o n ~ } \mathbb { T } \} .\tag{23}
$$

Next, we handle the singularity of Q. More precisely, let us assume that Q is always singular on T. In other words, the polynomial det $Q ( e ^ { i \vartheta } ) \equiv 0$ for any $\vartheta \in \mathbb { T }$ . Suppose that there exists $\bar { \vartheta } \in \mathbb { T }$ such that $Q ( e ^ { i \bar { \vartheta } } )$ has a normalized eigenvector v¯ associated to the zero eigenvalue and $\bar { v } ^ { * } { \cal P } ( e ^ { i \bar { \vartheta } } ) \bar { v } \neq 0 .$ According to [22, Thm. 6.3.12, p. 409], the eigenvector of a polynomial matrix $Q ( e ^ { i \vartheta } )$ associated with the smallest eigenvalue $\lambda ~ = ~ 0$ can be chosen continuously; see also [23], [24]. Hence $Q ( e ^ { i \vartheta } )$ admits an eigenvector function $v ( \vartheta )$ associated with the zero eigenvalue that satisfies $v ( \bar { \vartheta } ) = \bar { v }$ . By continuity, there exists a neighborhood centered in ϑ<sup>¯</sup> such that

$$
v ( \vartheta ) ^ { * } P ( e ^ { i \vartheta } ) v ( \vartheta ) \neq 0 \quad \forall \vartheta \in \mathcal { B } .
$$

We can now take

$$
W _ { \alpha } ( e ^ { i \vartheta } ) : = \left\{ \begin{array} { l l } { \alpha [ v ( \vartheta ) ^ { \ast } P ( e ^ { i \vartheta } ) v ( \vartheta ) ] v ( \vartheta ) v ( \vartheta ) ^ { \ast } , } & { \vartheta \in \mathcal { B } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{24}
$$

Then, we have

$$
\begin{array} { l } { { { \displaystyle { \cal L } ( W _ { \alpha } , { \bf P } , { \bf Q } ) = \mathrm { t r } \int _ { { \cal B } } [ ( W _ { \alpha } + W _ { \alpha } ^ { * } ) P - W _ { \alpha } W _ { \alpha } ^ { * } Q ] + \kappa } \ ~ } } \\ { { { \displaystyle ~ = \int _ { { \cal B } } 2 \alpha [ v ( \vartheta ) ^ { * } P ( e ^ { i \vartheta } ) v ( \vartheta ) ] ^ { 2 } + \kappa  \infty } } } \end{array}
$$

as $\alpha  \infty$ . The second equality above holds because $v ( \vartheta ) ^ { * } Q ( e ^ { i \vartheta } ) v ( \vartheta ) \equiv 0 $ . We conclude that the supremum of the Lagrangian is finite only if

$$
v \in \ker Q ( e ^ { i \vartheta } ) \implies v ^ { * } P ( e ^ { i \vartheta } ) v = 0 \quad \forall \vartheta \in \mathbb { T } .\tag{25}
$$

If P is also positive semidefinite (or negative semidefinite) pointwise, then the condition (25) boils down to

$$
\ker Q ( e ^ { i \vartheta } ) \subseteq \ker P ( e ^ { i \vartheta } ) \quad \forall \vartheta \in \mathbb { T } ,\tag{26}
$$

or equivalently,

$$
\operatorname { R a n g e } P ( e ^ { i \vartheta } ) \subseteq \operatorname { R a n g e } Q ( e ^ { i \vartheta } ) \quad \forall \vartheta \in \mathbb { T } .\tag{27}
$$

In general, the condition (26) or (27) implies (25) but the converse may not hold when P is indefinite. Under conditions $\mathbf { Q } \in \overline { { \mathfrak { B } _ { + } } }$ and (25), the Lagrangian $L ( \cdot , { \bf P } , { \bf Q } )$ is concave in W and its maxima are also stationary points, i.e., they are obtained by setting equal to zero the first variation of L along an arbitrary direction $\delta W \in \mathbb { L } _ { m \times m } ^ { 2 } ( \mathbb { T } )$

$$
\begin{array} { l } { { \displaystyle \delta { \cal L } \big ( W , { \bf P } , { \bf Q } ; \delta W \big ) } \ ~ } \\ { { \displaystyle ~ = \mathrm { t r } \int ( \delta W + \delta W ^ { * } ) { \cal P } - \delta W W ^ { * } Q - W \delta W ^ { * } Q } \ ~ } \\ { { \displaystyle ~ = 2 \mathrm { R e } \left( \mathrm { t r } \int { \cal P } \delta W ^ { * } - Q W \delta W ^ { * } \right) . } } \end{array}\tag{28}
$$

Clearly, in order to have $\delta L ( W , { \bf P } , { \bf Q } ; \delta W ) = 0$ for any $\delta W \in$ $\mathbb { L } _ { m \times m } ^ { 2 } ( \mathbb { T } )$ , we must set

$$
Q W - P \equiv O .\tag{29}
$$

Therefore, the unconstrained maximizers of the Lagrangian are of the form

$$
W ^ { \circ } = Q ^ { + } P + X\tag{30}
$$

where $( \cdot ) ^ { + }$ stands for the Moore–Penrose pseudoinverse, and Range $\dot { X } ( e ^ { i \vartheta } ) \subseteq [ \mathrm { R a n g e } Q ( e ^ { i \vartheta } ) ] ^ { \bot }$ for all $\vartheta \in \mathbb { T }$ , or equivalently, $Q X \equiv O$ pointwise. Thus, the dual function is

$$
\begin{array} { l } { { \displaystyle { \cal J } ( { \bf P } , { \bf Q } ) : = { \cal L } ( W ^ { \circ } , { \bf P } , { \bf Q } ) } \ ~ } \\ { { \displaystyle ~ = \mathrm { t r } \int Q ^ { + } P ^ { 2 } + \mathrm { t r } \sum _ { k = 0 } ^ { n } Q _ { k } ^ { T } R _ { k } - 2 \mathrm { t r } \sum _ { k = 1 } ^ { n } P _ { k } ^ { T } C _ { k } } } \end{array}\tag{31}
$$

and the dual problem is

$$
\begin{array} { r l } & { \underset { { \bf { P } } , { \bf { Q } } } { \operatorname* { m i n } } J ( { \bf { P } } , { \bf { Q } } ) } \\ & { \mathrm { ~ s . t . ~ } { \bf { P } } \in \mathfrak { B } _ { o } } \\ & { \qquad { \bf { Q } } \in \overline { { \mathfrak { B } _ { + } } } } \\ & { \qquad \mathrm { ~ R a n g e } P ( e ^ { i \vartheta } ) \subseteq \mathrm { R a n g e } Q ( e ^ { i \vartheta } ) , \forall \vartheta \in \mathbb { T } . } \end{array}\tag{32a}
$$

(32b)

Notice that we have used the stronger constraint (32b) in place of (25). However, it is not obvious that the dual problem admits a solution. Next, we consider a special situation. Let

$$
\mathfrak { B } _ { o , + } : = \{ \mathbf { P } \in \mathfrak { B } _ { o } : P > 0 \mathrm { ~ o n ~ } \mathbb { T } \} .\tag{33}
$$

Let $( \mathbf { P } , \mathbf { Q } )$ be a point in the feasible set of the dual problem (32). According to (27), if $\mathbf { P } \in \mathfrak { B } _ { o , + }$ , then $\mathbf { Q } \in { \mathfrak { B } } _ { + }$ . In this case,

$$
W ^ { \circ } = Q ^ { - 1 } P\tag{34}
$$

and the dual function evaluated for $( \mathbf { P } , \mathbf { Q } ) \in \mathfrak { B } _ { o , + } \times \mathfrak { B } _ { + }$ becomes

$$
J ( { \bf P } , { \bf Q } ) = \mathrm { t r } \int Q ^ { - 1 } P ^ { 2 } + \mathrm { t r } \sum _ { k = 0 } ^ { n } Q _ { k } ^ { T } R _ { k } - 2 \mathrm { t r } \sum _ { k = 1 } ^ { n } P _ { k } ^ { T } C _ { k } .\tag{35}
$$

Proposition 1. If the dual problem (32) admits a solution $( \mathbf { P } , \mathbf { Q } ) \in \mathfrak { B } _ { + } \times \mathfrak { B } _ { o , + }$ , then $\bar { \Phi ^ { } } = Q ^ { - 1 } P ^ { 2 } Q ^ { - 1 }$ solves the primal problem (17).

Proof. The function J in (31) is convex because it is a dual function. Since (P, Q) belongs to the interior of the feasible set, it follows that $( \mathbf { P } , \mathbf { Q } )$ must be a stationary point for J. Finally, it is not difficult to see that these stationary-point conditions coincide with the moment constraints in (17) with $W = Q ^ { - 1 } P$ , i.e., the duality gap between the primal and dual problems is equal to zero. □

Remark 3. If the polynomial matrix $Q$ has only isolated singularity, i.e. if det Q is not identically zero on T, then the optimal spectral factor in (34) is still valid almost everywhere. Consequently, the formula for the dual function (35) makes sense provided that the zeros of det Q (which form a finite set) are excluded from the domain of integration. In this case, however, the dual function should be interpreted as an extended real-valued function, and its values at boundary points should be defined with care. This point will be made clear in the next section.

## IV. REGULARIZED DUAL PROBLEM

As pointed out in the previous section, when $\mathbf { P } \in \mathfrak { B } _ { o , + }$ , any feasible point $( \mathbf { P } , \mathbf { Q } )$ of the dual problem belongs to $\mathfrak { B } _ { o , + } \times$ ${ \mathfrak { B } } _ { + }$ where we have Range $P ( e ^ { i \bar { \vartheta } } ) = \mathrm { R a n g e } \bar { Q ( } e ^ { i \vartheta } ) = \bar { \mathbb { C } } ^ { m }$ for all $\vartheta \in \mathbb { T }$ , and the dual function takes the simpler form (35). The idea is to consider a regularized version of the dual problem that explicitly enforces P to belong to $\mathfrak { B } _ { o , + } .$

$$
\begin{array} { r l } & { \underset { \mathbf { P } , \mathbf { Q } } { \operatorname* { m i n } } \tilde { J } _ { \boldsymbol { \tau } } ( \mathbf { P } , \mathbf { Q } ) } \\ & { \mathrm { s . t . } \mathbf { P } \in \mathfrak { B } _ { o , + } , \mathbf { Q } \in \mathfrak { B } _ { + } , } \end{array}\tag{36}
$$

where

$$
\begin{array} { l } { { \displaystyle { \tilde { J } } _ { \tau } ( { \bf P } , { \bf Q } ) : = { J } ( { \bf P } , { \bf Q } ) + \tau \mathrm { t r } \int P ^ { - 1 } } \ ~ } \\ { { \displaystyle = \mathrm { t r } \int Q ^ { - 1 } P ^ { 2 } + \mathrm { t r } \sum _ { k = 0 } ^ { n } Q _ { k } ^ { T } R _ { k } - 2 \mathrm { t r } \sum _ { k = 1 } ^ { n } P _ { k } ^ { T } C _ { k } + \tau \mathrm { t r } \int P ^ { - 1 } } } \end{array}\tag{37}
$$

and $\tau > 0$ is a regularization parameter.

As pointed out in the proof of Proposition 1, the dual function J is convex. Based on this fact, it is not difficult to show the convexity of the regularized dual function ${ \tilde { J } } _ { \tau }$ However, a direct analysis with ${ \breve { \tilde { J } } } _ { \tau }$ can yield strict convexity, which is a stronger result. We will need the next lemma.

Lemma 1. Fix a real number $\tau > 0$ . For two (Hermitian) positive definite matrices A and $B ,$ define a real-valued function

$$
g ( A , B ) : = \mathrm { t r } ( B ^ { - 1 } A ^ { 2 } ) + \tau \mathrm { t r } ( A ^ { - 1 } ) .\tag{38}
$$

Then g is strictly convex in $( A , B )$

Proof. Let us compute the first- and second-order differentials of $g \colon$

$$
\begin{array} { r l } & { \delta g ( A , B ; \delta A , \delta B ) = \mathrm { t r } \left[ - B ^ { - 1 } \delta B B ^ { - 1 } A ^ { 2 } \right. } \\ & { \qquad \left. + B ^ { - 1 } ( \delta A A + A \delta A ) - \tau A ^ { - 1 } \delta A A ^ { - 1 } \right] . } \end{array}\tag{(39}
$$

$$
\begin{array} { r } { \delta ^ { 2 } g ( A , B ; \delta A _ { 1 } , \delta B _ { 1 } , \delta A _ { 2 } , \delta B _ { 2 } ) = \mathrm { t r } \left[ B ^ { - 1 } ( Y + Y ^ { * } ) \right. } \\ { \left. + \tau ( Z + Z ^ { * } ) \right] , } \end{array}\tag{40}
$$

where $Y : = ( \delta B _ { 1 } B ^ { - 1 } A - \delta A _ { 1 } ) ( A B ^ { - 1 } \delta B _ { 2 } - \delta A _ { 2 } )$ and $Z : =$ $A ^ { - 1 } \delta A _ { 2 } A ^ { - 1 } \delta A _ { 1 } A ^ { - 1 }$ . By letting $( \delta A _ { 1 } , \delta B _ { 1 } ) = ( \delta A _ { 2 } , \delta B _ { 2 } ) =$ (δA, δB), we obtain the second variation

$$
\begin{array} { r l } & { \quad \delta ^ { 2 } g ( A , B ; \delta A , \delta B ) } \\ & { = 2 \mathrm { t r } \left[ \left( A B ^ { - 1 } \delta B - \delta A \right) B ^ { - 1 } ( \delta B B ^ { - 1 } A - \delta A ) \right. } \\ & { \quad \left. + \tau A ^ { - 1 } \delta A A ^ { - 1 } \delta A A ^ { - 1 } \right] \geq 0 , } \end{array}\tag{41}
$$

where the last inequality holds because both $A ^ { - 1 } > 0$ and $B ^ { - 1 } > 0$ . Now if we impose $\delta ^ { 2 } g ( A , B ; \delta A , \delta B ) = 0 .$ , then the trace of both terms in the bracket above must vanish, implying $\delta A A ^ { - 1 } = 0$ and $\delta B B ^ { - 1 } A - \delta A = 0$ which further yield $\delta A = \delta B = 0$ . Hence, the second variation of g is positive definite which leads to strict convexity. □

Proposition 2. The regularized dual function ${ \tilde { J } } _ { \tau }$ is strictly convex in $\mathfrak { B } _ { o , + } \times \mathfrak { B } _ { + }$

Proof. The claim can be proved by showing the strict convexity of the function

$$
f _ { \tau } ( { \bf P } , { \bf Q } ) : = \mathrm { t r } \int Q ^ { - 1 } P ^ { 2 } + \tau \mathrm { t r } \int P ^ { - 1 }\tag{42}
$$

since ${ \tilde { J } } _ { \tau }$ and $f _ { \tau }$ only differs from linear terms in $( \mathbf { P } , \mathbf { Q } )$ Using the function g defined in (38), we can rewrite $f _ { \tau }$ as $\textstyle \int g ( { \bar { P } } ( e ^ { i \vartheta } ) , Q ( e ^ { i \vartheta } ) )$ ). Furthermore, for any $t \in ( 0 , 1 )$ and any

two points $( \mathbf { P } _ { 1 } , \mathbf { Q } _ { 1 } )$ and $( \mathbf { { P } } _ { 2 } , \mathbf { { Q } } _ { 2 } )$ in $\mathfrak { B } _ { o , + } \times \mathfrak { B } _ { + }$ , Lemma 1 implies that

$$
\begin{array} { r l } & { \quad f _ { \tau } \left( t ( \mathbf { P } _ { 1 } , \mathbf { Q } _ { 1 } ) + ( 1 - t ) ( \mathbf { P } _ { 2 } , \mathbf { Q } _ { 2 } ) \right) } \\ & { = \displaystyle \int g \left( t ( P _ { 1 } ( e ^ { i \vartheta } ) , Q _ { 1 } ( e ^ { i \vartheta } ) ) + ( 1 - t ) ( P _ { 2 } ( e ^ { i \vartheta } ) , Q _ { 2 } ( e ^ { i \vartheta } ) ) \right) } \\ & { < t \displaystyle \int g \left( P _ { 1 } ( e ^ { i \vartheta } ) , Q _ { 1 } ( e ^ { i \vartheta } ) \right) + ( 1 - t ) \displaystyle \int g \left( P _ { 2 } ( e ^ { i \vartheta } ) , Q _ { 2 } ( e ^ { i \vartheta } ) \right) } \\ & { = t f _ { \tau } \left( \mathbf { P } _ { 1 } , \mathbf { Q } _ { 1 } \right) + ( 1 - t ) f _ { \tau } \left( \mathbf { P } _ { 2 } , \mathbf { Q } _ { 2 } \right) , } \end{array}\tag{43}
$$

which is by definition the strong convexity of $f _ { \tau }$ . The strict inequality holds in the above formula because of the continuity of the integrands. □

Feasibility assumption: There exists a spectral density $\bar { \Phi }$ which is positive definite on an open interval in T and satisfies the moment constraints

$$
\int e ^ { i \vartheta k } \bar { \Phi } = R _ { k } , \quad k = 0 , \ldots , n .\tag{44}
$$

Notice that under the feasibility assumption, we have

$$
\mathrm { t r } \sum _ { k = 0 } ^ { n } Q _ { k } ^ { T } R _ { k } = \mathrm { t r } \int \bar { \Phi } Q \geq 0 .\tag{45}
$$

In what follows, we always suppose that the feasibility assumption holds. In practice, the feasibility assumption can be guaranteed by suitably estimating the covariances using e.g., the periodogram; see [25], [26]. This point will be made precise in the next section.

Next, we shall characterize the behavior of ${ \tilde { J } } _ { \tau }$ near the boundary of the feasible set. Before to proceed, let us define the norm of P and $\mathbf { Q }$ as

$$
\| \mathbf { P } \| ^ { 2 } : = \frac { 1 } { 2 } \sum _ { k = 1 } ^ { n } \mathrm { t r } ( P _ { k } P _ { k } ^ { T } ) = \int \| P ( e ^ { i \vartheta } ) - I \| _ { \mathrm { F } } ^ { 2 } ,\tag{46}
$$

$$
\| { \bf Q } \| ^ { 2 } : = \mathrm { t r } ( Q _ { 0 } Q _ { 0 } ^ { T } ) + \frac { 1 } { 2 } \sum _ { k = 1 } ^ { n } \mathrm { t r } ( Q _ { k } Q _ { k } ^ { T } ) = \int \| Q ( e ^ { i \vartheta } ) \| _ { \mathrm { F } } ^ { 2 } ,\tag{47}
$$

and $\| ( \mathbf { P } , \mathbf { Q } ) \| ^ { 2 } : = \| \mathbf { P } \| ^ { 2 } + \| \mathbf { Q } \| ^ { 2 }$ . Here the factor $1 / 2$ comes from the definition of the polynomial matrices in (4), and $\Vert \cdot \Vert _ { \mathrm { F } }$ denotes the Frobenius norm of a matrix. The following lemma will be useful in the subsequent development.

Lemma 2. The set $\mathfrak { B } _ { o , + }$ in (33) is bounded, i.e., there exists a constant $C > 0$ such that any $\mathbf { P } \in \mathfrak { B } _ { o , + }$ satisfies $\| \mathbf { P } \| \leq C$ Proof. Let us take $\mathbf { P } \in \mathfrak { B } _ { o , + }$ . The following computation is straightforward:

$$
\begin{array} { l } { { \displaystyle \| P _ { k } \| _ { \mathrm { F } } ^ { 2 } = \mathrm { t r } ( P _ { k } P _ { k } ^ { T } ) = \mathrm { t r } \left[ \left( 2 \int e ^ { i \vartheta k } P \right) \left( 2 \int e ^ { - i \varphi k } P \right) \right] } } \\ { { \displaystyle \qquad = 4 \int \int e ^ { i ( \vartheta - \varphi ) k } \mathrm { t r } \left[ P ( e ^ { i \vartheta } ) P ( e ^ { i \varphi } ) \right] , } } \end{array}\tag{48}
$$

where the two integrals are taken with respect to different integration variables, $\vartheta \in \mathbb { T }$ and $\varphi \in \mathbb { T }$ . The last trace term in

(48) is nonnegative for any $\vartheta$ and $\varphi .$ Taking the modulus (of a complex number) on both sides of the equation, we have

$$
\begin{array} { l } { \displaystyle \| P _ { k } \| _ { \mathrm { F } } ^ { 2 } \leq 4 \int \int \mathrm { t r } \left[ P ( e ^ { i \vartheta } ) P ( e ^ { i \varphi } ) \right] } \\ { = 4 \int \displaystyle \int \sum _ { i , j = 1 } ^ { m } p _ { i j } ( e ^ { i \vartheta } ) p _ { i j } ^ { \ast } ( e ^ { i \varphi } ) } \\ { = 4 \displaystyle \sum _ { i , j = 1 } ^ { m } \int p _ { i j } ( e ^ { i \vartheta } ) \int p _ { i j } ^ { \ast } ( e ^ { i \varphi } ) = 4 m , } \end{array}\tag{49}
$$

where the last equality follows from the fact that

$$
\int p _ { i j } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } i = j } \\ { 0 } & { { \mathrm { i f ~ } } i \neq j } \end{array} \right. }
$$

is the constant term of the scalar polynomial $p _ { i j }$ , the $( i , j )$ entry of the polynomial matrix P whose constant term is I. In view of (46), we have $\| \mathbf { P } \| \leq { \sqrt { 2 m n } } = : C .$ □

Proposition 3. Consider a sequence $\{ ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) \} _ { l \geq 1 } \subset \mathfrak { B } _ { o , + } \times$ ${ \mathfrak { B } } _ { + }$ that converges to a point $( \bar { \mathbf { P } } , \bar { \mathbf { Q } } ) \in \partial ( \mathfrak { B } _ { o , + } \times \mathfrak { B } _ { + } )$ as l diverges. Then,

$$
\operatorname* { l i m } _ { l  \infty } \tilde { J } _ { \tau } ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) = \infty ,\tag{50}
$$

i.e., $\{ ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) \} _ { l \geq 1 }$ is not an infimizing sequence.

Proof. Since $\mathbf { P } ^ { l } \in \mathfrak { B } _ { o , + }$ , we know that $\mathbf { P } ^ { l }$ is bounded by Lemma 2, and this further implies that the inner product $\mathrm { t r } \sum _ { k = 1 } ^ { n } ( P _ { k } ^ { l } ) ^ { T } C _ { k }$ is also bounded. Let us first consider the Pcase in which $( \bar { \mathbf { P } } , \bar { \mathbf { Q } } ) \in \partial \mathfrak { B } _ { o , + } \times \overline { { \mathfrak { B } _ { + } } }$ . In view of (45) we have that

$$
\tilde { J } _ { \tau } ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) \geq - 2 \mathrm { t r } \sum _ { k = 1 } ^ { n } ( P _ { k } ^ { l } ) ^ { T } C _ { k } + \tau \int \mathrm { t r } [ ( P ^ { l } ) ^ { - 1 } ] .\tag{51}
$$

The term $\mathrm { t r } [ ( P ^ { l } ) ^ { - 1 } ]$ is a nonnegative rational function and it has at least one pole tending to the unit circle as $l  \infty$ Accordingly, the second term in (51) tends to infinity as $l $ $\infty .$ . Moreover, the latter dominates the first term in (51) and thus we conclude that ${ \tilde { J } } _ { \tau }$ diverges when $\mathbf { P } ^ { l }$ tends to $\bar { \textbf { P } } \in$ $\partial \mathfrak { B } _ { o , + }$

It remains to show that ${ \tilde { J } } _ { \tau }$ also diverges when $( \bar { \bf P } , \bar { \bf Q } ) \in$ $\mathfrak { B } _ { o , + } \times \partial \mathfrak { B } _ { + }$ . Let $\varepsilon _ { l } \geq 0$ denote the minimum over T of the minimum eigenvalue of the polynomial matrix $P ^ { l } ( e ^ { i \vartheta } )$ . Since the latter converges to $\bar { P } \in \mathfrak { B } _ { o , + }$ , there exist two constants $\epsilon > 0$ and $\bar { l } \in \mathbb { N }$ such that $\varepsilon _ { l } \geq \varepsilon$ for $l \geq { \bar { l } } .$ Therefore,

$$
\mathrm { t r } \int ( Q ^ { l } ) ^ { - 1 } ( P ^ { l } ) ^ { 2 } \geq \varepsilon ^ { 2 } \mathrm { t r } \int ( Q ^ { l } ) ^ { - 1 } ,
$$

and thus

$$
\tilde { J } _ { \tau } ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) \geq \varepsilon ^ { 2 } \int \mathrm { t r } [ ( Q ^ { l } ) ^ { - 1 } ] - 2 \mathrm { t r } \sum _ { k = 1 } ^ { n } ( P _ { k } ^ { l } ) ^ { T } C _ { k } ,\tag{52}
$$

where the first term on the right-hand side dominates the second one because it diverges since t $\mathrm { r } [ ( Q ^ { l } ) ^ { - 1 } ]$ is a nonnegative rational function having at least one pole tending to the unit circle as $l  \infty$ □

The next proposition shows the behavior of ${ \tilde { J } } _ { \tau }$ when the norm of its argument tends to infinity.

Proposition 4. Consider a sequence $\{ ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) \} _ { l \geq 1 } \subset \mathfrak { B } _ { o , + } \times$ ${ \mathfrak { B } } _ { + }$ such that $\| ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) \|  \infty$ as $l  \infty .$ . Then,

$$
\operatorname* { l i m } _ { l  \infty } \tilde { J } _ { \tau } ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) = \infty .\tag{53}
$$

Proof. According to Lemma $2 , { \bf P } ^ { l }$ is bounded in the feasible set and thus $\| ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) \|  \infty$ implies that $\| \mathbf { Q } ^ { l } \|  \infty$ . By (45), we have

$$
\begin{array} { r l } {  { \tilde { J } _ { \tau } ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) \geq \operatorname { t r } \sum _ { k = 0 } ^ { n } ( Q _ { k } ^ { l } ) ^ { T } R _ { k } - 2 \mathrm { t r } \sum _ { k = 1 } ^ { n } ( P _ { k } ^ { l } ) ^ { T } C _ { k } } } \\ & { = \operatorname { t r } \int \bar { \Phi } Q ^ { l } - 2 \mathrm { t r } \sum _ { k = 1 } ^ { n } ( P _ { k } ^ { l } ) ^ { T } C _ { k } } \end{array}\tag{54}
$$

where the second term remains bounded as $l  \infty$ . Let $\tilde { \mathbf { Q } } ^ { l } : =$ $\mathbf { Q } ^ { l } / \vert \vert \mathbf { Q } ^ { l } \vert \vert$ and take a subsequence $\tilde { \mathbf { Q } } ^ { l _ { s } }$ , with $s \in \mathbb { N } .$ , such that $\begin{array} { r } { \operatorname* { i m } _ { s \to \infty } \mathrm { t r } \int \bar { \Phi } \tilde { Q } ^ { l _ { s } } = \eta : = } \end{array}$ lim inf $l \to \infty \mathrm { t r } \int \bar { \Phi } \tilde { Q } ^ { l } \geq 0$ . Since such a subsequence belongs to the surface of the (compact) unit ball, then there exists a convergent subsubsequence $\mathbf { \tilde { Q } } ^ { l _ { t } }$ with $t \in \mathbb { N }$ and let

$$
\tilde { \mathbf { Q } } = \operatorname* { l i m } _ { t  \infty } \tilde { \mathbf { Q } } ^ { l _ { t } } \in \overline { { \mathfrak { B } _ { + } } } .\tag{55}
$$

Then, $\eta = \mathrm { t r } \int \bar { \Phi } \tilde { Q }$ and we claim that $\eta > 0$ . Indeed, assume by contradiction that $\eta = 0$ . Since $\mathrm { t r } ( \bar { \Phi } \tilde { Q } ) \geq 0 \ \mathrm { a . e }$ . on T, $\eta = 0$ implies that $\mathrm { t r } ( \bar { \Phi } \tilde { Q } ) = 0 ~ \mathrm { a . e } .$ . on T. By the feasibility assumption, $\tilde { Q }$ must vanish in some open interval in T and thus $\tilde { \mathbf { Q } } \equiv 0$ . This is a contradiction because $\lVert \tilde { \mathbf { Q } } \rVert = 1$ . We conclude that $\eta > 0$ . Since by definition $\begin{array} { r } { \eta = \operatorname* { l i m } \operatorname* { i n f } _ { l \to \infty } \operatorname { t r } \int \bar { \Phi } \tilde { Q } ^ { l } } \end{array}$ , there exists a positive integer <sup>¯</sup>l such that tr $\int \bar { \Phi } \tilde { Q } ^ { l } > \eta / 2$ for all $l \geq { \bar { l } } .$ Consequently,

$$
\begin{array} { l } { { \displaystyle \mathrm { t r } \int \bar { \Phi } Q ^ { l } = \| { \bf Q } ^ { l } \| \mathrm { t r } \int \bar { \Phi } \tilde { Q } ^ { l } } } \\ { { \displaystyle ~ > \frac { \eta } { 2 } \| { \bf Q } ^ { l } \|  \infty ~ \mathrm { a s } ~ l  \infty , } } \end{array}\tag{56}
$$

where the inequality holds for all $l \geq \bar { l } .$ In view of (54), the above formula implies $\tilde { J } _ { \tau } ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } )  \infty , \mathrm { i . e . , ~ } \{ ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) \} _ { l \geq 1 }$ is not an infimizing sequence.

Theorem 1. The regularized dual problem (36) admits a unique solution, say $( \mathbf { P } , \mathbf { Q } )$ . Moreover, the spectral factor $W ^ { \circ } = Q ^ { - 1 } P$ satisfies the moment conditions:

$$
\begin{array} { l } { { \displaystyle \int e ^ { i \vartheta k } W ^ { \circ } ( W ^ { \circ } ) ^ { \ast } = R _ { k } , \quad k = 0 , \ldots , n } } \\ { { \displaystyle \frac { 1 } { 2 } \int e ^ { i \vartheta k } \left[ W ^ { \circ } + ( W ^ { \circ } ) ^ { \ast } \right] = C _ { k } + E _ { k } , \quad k = 1 , \ldots , n } } \end{array}\tag{57}
$$

with the error of cepstral matching

$$
E _ { k } : = { \frac { \tau } { 2 } } \int e ^ { i \vartheta k } P ^ { - 2 } .
$$

Proof. Consider the sublevel set of the regularized dual function

$$
\tilde { J } _ { \tau } ^ { - 1 } ( - \infty , \beta ] : = \{ ( \mathbf { P } , \mathbf { Q } ) \in \mathfrak { B } _ { o , + } \times \mathfrak { B } _ { + } : \tilde { J } _ { \tau } ( \mathbf { P } , \mathbf { Q } ) \leq \beta \}\tag{58}
$$

where $\beta \in \mathbb { R }$ is sufficiently large such that the set above is nonempty. For example, one can take $\beta = \tilde { J } _ { \tau } ( \mathbf { P } _ { 0 } , \mathbf { Q } _ { 0 } )$ for an arbitrarily chosen $( \mathbf { P } _ { 0 } , \mathbf { Q } _ { 0 } ) \in \mathfrak { B } _ { o , + } \times \mathfrak { B } _ { + }$ . The aim is to show that the sublevel set is compact, i.e., closed and bounded because (P, Q) resides in a finite-dimensional space. For the closedness, suppose that there is a sequence $\{ ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } ) \} _ { l \geq 1 } \subset$ $\tilde { J } _ { \tau } ^ { - 1 } ( - \infty , \beta ]$ that converges to $( \bar { \mathbf { P } } , \bar { \mathbf { Q } } ) \in \overline { { \mathfrak { B } _ { o , + } } } \times \overline { { \mathfrak { B } _ { + } } } .$ . We need to show that (P<sup>¯</sup> , Q<sup>¯</sup> ) are in fact in the open set $\mathfrak { B } _ { o , + } \times \mathfrak { B } _ { + }$ so that (P<sup>¯</sup> , Q<sup>¯</sup> ) belongs to the sublevel set by continuity. To this end, suppose that $( \bar { \mathbf { P } } , \bar { \mathbf { Q } } ) \in \partial ( \mathfrak { B } _ { o , + } \times \mathfrak { B } _ { + } )$ . Then according to Proposition 3, the sequence of function values $\tilde { J } _ { \boldsymbol { \tau } } ( \mathbf { P } ^ { l } , \mathbf { Q } ^ { l } )$ will eventually become larger than $\beta ,$ which means that $( \mathbf { P } ^ { l } , \dot { \mathbf { Q } } ^ { l } ) \notin$ $\tilde { J } _ { \tau } ^ { - 1 } ( - \infty , \beta ]$ for all sufficiently large l, a contradiction. Hence the limit point (P<sup>¯</sup> , Q<sup>¯</sup> ) must be an interior point and also a member of the sublevel set. The boundedness of $\tilde { J } _ { \tau } ^ { - 1 } ( - \infty , \beta ]$ can be proved in a similar way with reference to Proposition 4. Accordingly, the regularized dual problem admits a point of minimum on the compact sublevel set by Weierstrass’ theorem. The uniqueness follows from the fact that ${ \tilde { J } } _ { \tau }$ is strictly convex in $\mathfrak { B } _ { o , + } \times \mathfrak { B } _ { + }$ (Proposition 2). Finally, such a minimizer must be a stationary point of ${ \tilde { J } } _ { \tau }$ . The stationarity conditions are obtained by imposing

$$
\begin{array} { l } { { \displaystyle \delta { \tilde { J } } _ { \tau } ( { \bf P } , { \bf Q } , \delta { \bf P } , \delta { \bf Q } ) } \ ~ } \\ { { \displaystyle = 2 \mathrm { R e } \mathrm { t r } \left\{ \frac 1 2 \sum _ { k = 0 } ^ { n } \delta Q _ { k } ^ { T } \left( R _ { k } - \int e ^ { i \vartheta k } Q ^ { - 1 } P ^ { 2 } Q ^ { - 1 } \right) \right. } \ ~ } \\ { { \displaystyle \left. + \sum _ { k = 1 } ^ { n } \delta P _ { k } ^ { T } \left[ \frac 1 2 \int e ^ { i \vartheta k } ( P Q ^ { - 1 } + Q ^ { - 1 } P - \tau P ^ { - 2 } ) - C _ { k } \right] \right\} } } \end{array}\tag{59}
$$

$$
\mathbf { \sigma } = 0 , \forall \delta \mathbf { P } \in \mathfrak { B } _ { o } , \delta \mathbf { Q } \in \mathfrak { B } ,
$$

which clearly implies the moment constraints in (57) with $W ^ { \circ } = Q ^ { - 1 } \dot { P }$ □

## V. SPECTRAL ESTIMATION

Let $y _ { 1 } , \ldots , y _ { N }$ be a finite-length realization obtained from a process with spectral density Φ. We can compute a coarse estimate of Φ from the data by means of the f-periodogram [26]:

$$
\hat { \Phi } _ { N } ( e ^ { i \vartheta } ) = \sum _ { k = - f ( N ) } ^ { f ( N ) } e ^ { - i \vartheta k } \hat { R } _ { k }\tag{60}
$$

where

$$
\hat { R } _ { k } = \frac { 1 } { N } \sum _ { t = 1 } ^ { N - k } y _ { t + k } y _ { t } ^ { T }\tag{61}
$$

is the standard biased covariance estimator. Moreover, we assume that $\hat { \Phi } _ { N }$ is positive definite a.e. on T. Let $\hat { W } _ { N }$ be a canonical left spectral factor<sup>1</sup> of the f-periodogram, and thus $\hat { \Phi } _ { N } = \hat { W } _ { N } \hat { W } _ { N } ^ { * }$ . We can take $\hat { W } _ { N }$ as the causal and minimum phase spectral factor which is of the form

$$
\hat { W } _ { N } ( e ^ { i \vartheta } ) = \sum _ { k = 0 } ^ { f ( N ) } e ^ { - i \vartheta k } \hat { S } _ { k } .\tag{62}
$$

Another possible choice is to take $\hat { W } _ { N }$ as the pointwise Cholesky factor, i.e., $\hat { W } _ { N }$ is a continuous matrix function such that $\hat { W _ { N } } ( e ^ { i \vartheta } )$ is lower triangular and $\hat { W } _ { N } ( e ^ { i \vartheta } ) \hat { W } _ { N } ( e ^ { i \vartheta } ) ^ { * } = $ Φˆ $\cdot _ { N } ( e ^ { i \vartheta } )$ for all ϑ. Notice that, the spectral factor is not unique. More precisely, all the left spectral factors of $\hat { \Phi } _ { N }$ are obtained from $\hat { W } _ { N }$ as

$$
\hat { W } = \hat { W } _ { N } U\tag{63}
$$

where U is an inner matrix function, i.e. $U ( e ^ { i \vartheta } ) U ( e ^ { i \vartheta } ) ^ { \ast } = I$ for all $\vartheta \in \mathbb { T }$ . For the moment, we assume that U is fixed. Then, an estimator of Φ is obtained by solving the regularized dual problem (36) where $R _ { k }$ is replaced by $\hat { R } _ { k }$ and $C _ { k }$ by

$$
\hat { C } _ { k } = \frac { 1 } { 2 } \int e ^ { i \vartheta k } ( \hat { W } _ { N } U + U ^ { * } \hat { W } _ { N } ^ { * } ) .\tag{64}
$$

It is worth noting that the feasibility condition (44) is satisfied by taking $\bar { \Phi } = \bar { \Phi } _ { N }$ . By (60) and (64), the regularized dual function in (37) can be rewritten as

$$
\begin{array} { l } { { \displaystyle \tilde { \cal J } _ { \tau } ( { \bf P } , { \bf Q } , U ) = } } \\ { { \displaystyle \mathrm { t r } \int \left[ Q ^ { - 1 } P ^ { 2 } + Q \hat { \Phi } _ { N } - ( P - I ) ( \hat { W } _ { N } U + U ^ { * } \hat { W } _ { N } ^ { * } ) + \tau P ^ { - 1 } \right] } } \end{array}\tag{65}
$$

where we have explicitly included the inner function U as an additional argument. Then, the minimization of ${ \tilde { J } } _ { \tau }$ with respect to $( \mathbf { P } , \mathbf { Q } )$ is equivalent to minimizing the function

$$
\begin{array} { l } { { \displaystyle { \check { J } } _ { \tau } ( { \bf P } , { \bf Q } ) = { \tilde { J } } _ { \tau } ( { \bf P } , { \bf Q } ) - \mathrm { t r } \int \left( { \hat { W } } _ { N } U + U ^ { * } { \hat { W } } _ { N } ^ { * } \right) } \ ~ } \\ { { \displaystyle = \mathrm { t r } \int \left[ Q ^ { - 1 } P ^ { 2 } + Q { \hat { \Phi } } _ { N } - P ( { \hat { W } } _ { N } U + U ^ { * } { \hat { W } } _ { N } ^ { * } ) + \tau P ^ { - 1 } \right] } } \end{array}
$$

as the two functions differ from a term not depending on $( \mathbf { P } , \mathbf { Q } )$ . Moreover, notice that

$$
\begin{array} { l } { { \displaystyle { \tilde { J } } _ { \tau } ( { \bf P } , { \bf Q } ) } } \\ { { \displaystyle = \mathrm { t r } \int \left[ P ^ { 2 } Q ^ { - 1 } + Q \hat { \Phi } _ { N } - Q \hat { W } _ { N } U P Q ^ { - 1 } - P U ^ { * } \hat { W } _ { N } ^ { * } + \tau P ^ { - 1 } \right] } } \\ { ~ } \\ { { \displaystyle = \mathrm { t r } \int \left[ Q ( Q ^ { - 1 } P ^ { 2 } Q ^ { - 1 } + \hat { \Phi } _ { N } - \hat { W } _ { N } U P Q ^ { - 1 } - Q ^ { - 1 } P U ^ { * } \hat { W } _ { N } ^ { * } ) \right. } } \\ { { \displaystyle \left. + \tau P ^ { - 1 } \right] = \int \| Q ^ { - 1 } P - \hat { W } _ { N } U \| _ { Q } ^ { 2 } + \tau \mathrm { t r } \int P ^ { - 1 } , } } \end{array}\tag{66}
$$

where $\| X \| _ { Q } ~ : = ~ \mathrm { t r } ( X ^ { * } Q X )$ with matrices X and $Q$ of compatible size and $Q > 0$ . In plain words, the dual problem finds the spectral factor $W = Q ^ { - 1 } P$ which is the closest to $\hat { W } _ { N } U$ according to a weighted norm. As we already noticed, such solution depends on the particular spectral factor that we have chosen, i.e., it depends on the inner function U. Moreover, $\check { J } _ { \tau } ( \mathbf { P } , \mathbf { Q } ) \ \geq \ 0$ and, neglecting the effect of the regularization term, $\check { J } _ { \tau } ( \mathbf { P } , \mathbf { Q } ) \approx 0$ only when $Q ^ { - 1 } P \approx { \hat { W } } _ { N } U$ Note that, if we take $U \equiv I ,$ , then $\hat { W } _ { N } U$ is a causal spectral factor, while ${ \cal Q } ^ { - 1 } { \cal P }$ is not. Accordingly, the idea is to learn jointly (P, Q) and U through an alternating minimization scheme. Then, at Stage $\ell \in \mathbb { N } _ { + }$ of the procedure we perform

the following steps:

Spectral factor optimization

$$
\begin{array} { r } { ( \mathbf { P } ^ { ( \ell ) } , \mathbf { Q } ^ { ( \ell ) } ) = \underset { \mathbf { P } , \mathbf { Q } } { \mathrm { a r g m i n } } \breve { J } _ { \tau } ( \mathbf { P } , \mathbf { Q } , U ^ { ( \ell - 1 ) } ) } \\ { \mathrm { s . t . } \mathbf { P } \in \mathfrak { B } _ { o , + } , \mathbf { Q } \in \mathfrak { B } _ { + } } \end{array}\tag{67}
$$

Inner function optimization

$$
\begin{array} { r l } & { U ^ { ( \ell ) } = \underset { U } { \mathrm { a r g m i n } } \breve { J } _ { \tau } ( \mathbf { P } ^ { ( \ell ) } , \mathbf { Q } ^ { ( \ell ) } , U ) } \\ & { ~ \mathrm { s . t . } ~ U U ^ { * } = I \quad \forall \vartheta \in \mathbb { T } . } \end{array}\tag{68}
$$

The alternating minimization procedure can be initialized with $U ^ { ( 0 ) } \equiv { \bar { I } }$ . Clearly, the problem (67) admits a unique global minimum. If also (68) admits a global minimum, and if $( \mathbf { P } ^ { ( \ell ) } , \mathbf { Q } ^ { ( \ell ) } , U ^ { ( \ell ) } )$ converges, then such a limit is a coordinatewise point of minimum for $\check { J } _ { \tau } ( \mathbf { P } , \mathbf { Q } , U )$ . A point of minimum of (68) can be characterized as follows.

Proposition 5. A solution to the minimization problem (68) is

$$
U ( e ^ { i \vartheta } ) = \hat { V } ( e ^ { i \vartheta } ) V ( e ^ { i \vartheta } ) ^ { * }\tag{69}
$$

where

$$
P ^ { ( \ell ) } ( e ^ { i \vartheta } ) \hat { W } _ { N } ( e ^ { i \vartheta } ) = V ( e ^ { i \vartheta } ) D ( e ^ { i \vartheta } ) \hat { V } ( e ^ { i \vartheta } ) ^ { \ast }\tag{70}
$$

is a pointwise singular value decomposition of $P ^ { ( \ell ) } \hat { W } _ { N }$ with $V V ^ { \bar { * } } = \hat { V } \hat { V } ^ { * } = I$ and $D > 0$ diagonal.

Proof. By (66), the optimization problem (68) is equivalent to minimizing

$$
- \mathrm { t r } \int \hat { W } _ { N } U P ^ { ( \ell ) } + P ^ { ( \ell ) } U ^ { * } \hat { W } _ { N } ^ { * } = - 2 \mathrm { R e } \mathrm { t r } \int P ^ { ( \ell ) } \hat { W } _ { N } U ,
$$

which is the same as maximizing Re tr $\int P ^ { ( \ell ) } \hat { W } _ { N } U$ under the constraint $U U ^ { * } = I .$ R. In view of (70), we have the following chain of inequalities:

$$
\begin{array} { l } { \displaystyle \mathrm { R e } \mathrm { t r } \int P ^ { ( \ell ) } \hat { W } _ { N } U \leq \left| \mathrm { t r } \int D \tilde { U } \right| } \\ { \displaystyle \qquad \leq \int \left| \mathrm { t r } ( D \tilde { U } ) \right| \leq \int \mathrm { t r } D } \end{array}\tag{71}
$$

where the first inequality comes from the fact $\mathrm { R e } z \le | z |$ for any $z \in \mathbb { C }$ and $\tilde { U } : \stackrel { \cdot } { = } \hat { V } ^ { * } U V$ is pointwise unitary. Clearly, all the inequalities above become equalities if one takes ${ \tilde { U } } \equiv I .$ Therefore, a minimizer of (68) is (69) and it is an all-pass function because it is a product of two such functions $\hat { V }$ and V. 口

Remark 4. In practice, upon the completion of Stage $\ell ,$ we can compute the values of the optimal solution $U ^ { ( \ell ) }$ to (68) over a grid in [0, 2π) and then we use these values to update the W-cepstral coefficients through

$$
\hat { C } _ { k } ^ { ( \ell + 1 ) } = \frac { 1 } { 2 } \int e ^ { i \vartheta k } ( \hat { W } _ { N } U ^ { ( \ell ) } + ( U ^ { ( \ell ) } ) ^ { * } \hat { W } _ { N } ^ { * } ) , \quad k = 1 , \dots , n\tag{72}
$$

where the integral is approximated with a (finite) Riemann sum. Then in Stage $\ell + 1$ , we can solve the minimization problem (67) with the updated W-cepstral coefficients $\{ \hat { C } _ { k } ^ { ( \ell + 1 ) } \} _ { k = 1 } ^ { n }$

Next, we analyze the consistency of the proposed estimator. Data generation assumption: The data $y _ { 1 } , \ldots , y _ { N }$ have been generated by a process having spectral density

$$
\Phi _ { \mathrm { t r u e } } = Q _ { \mathrm { t r u e } } ^ { - 1 } P _ { \mathrm { t r u e } } ^ { 2 } Q _ { \mathrm { t r u e } } ^ { - 1 }\tag{73}
$$

with $\mathbf { P _ { t r u e } } \in \mathfrak { B } _ { o , + }$ and $\mathbf { Q } _ { \mathrm { t r u e } } \in \mathfrak { B } _ { + }$

Let $\mathcal { A } _ { m , + }$ denote the set of $\mathbb { C } ^ { m \times m }$ -valued functions of $z \in$ C which are analytic in an annulus containing the unit circle and are positive definite for all $z = e ^ { i \vartheta }$ with $\vartheta \in \mathbb { T }$ . Consider the continuous map for constructing a left canonical spectral factor of an element in $A _ { m , + }$

$$
\begin{array} { c } { \mathcal { T } \colon \mathcal { A } _ { m , + } \to \mathbb { L } _ { m \times m } ^ { 2 } ( \mathbb { T } ) } \\ { \Phi \mapsto W } \end{array}
$$

such that $\Phi = W W ^ { * }$ and

$$
W ( z ) = \sum _ { k = 0 } ^ { \infty } W _ { k } z ^ { - k }
$$

is analytic outside a disk with radius smaller than one. Since $\Phi _ { \mathrm { t r u e } }$ is positive definite in T, the true process can be generated by a bounded input-bounded output (BIBO) stable system fed by white noise. Choosing $f ( N )$ such that $f ( N ) \to \infty$ and $\dot { f } ( N ) / \sqrt { N } \to 0$ as $N  \infty ,$ by [26, Theorem 1] we have that

$$
{ \hat { \Phi } } _ { N } \ { \xrightarrow { \mathrm { m . s . } } } \ \Phi _ { \mathrm { t r u e } } \quad { \mathrm { a s } } \quad N \to \infty ,\tag{74}
$$

where m.s. indicate mean square convergence. Therefore, for N sufficiently large $\hat { W } _ { N } = \mathcal { T } ( \hat { \Phi } _ { N } )$ is well defined and by the continuous mapping theorem we have that

$$
\begin{array} { r } { \hat { W } _ { N } \xrightarrow { \mathrm { m . s . } } W _ { \mathrm { t r u e } } \quad \mathrm { a s } \quad N  \infty } \end{array}
$$

with $W _ { \mathrm { t r u e } } = \mathcal { T } ( \Phi _ { \mathrm { t r u e } } )$ . Let $\check { J } _ { 0 , N } ( \mathbf { P } , \mathbf { Q } , U )$ denote the function in (66) with $\tau = 0 .$ . Then, taking the limit $N \to \infty$ we obtain

$$
\check { J } _ { 0 , \infty } ( \mathbf { P } , \mathbf { Q } , U ) = \int \| Q ^ { - 1 } P - W _ { \mathrm { t r u e } } U \| _ { Q } ^ { 2 } \geq 0\tag{75}
$$

and it is not difficult to see that

$$
\check { J } _ { 0 , \infty } ( \mathbf { P } _ { \mathrm { t r u e } } , \mathbf { Q } _ { \mathrm { t r u e } } , U ^ { \star } ) = 0\tag{76}
$$

where $\mathbf { P } _ { \mathrm { t r u e } } , \mathbf { Q } _ { \mathrm { t r u e } }$ correspond to $P _ { \mathrm { t r u e } } , Q _ { \mathrm { t r u e } }$ , respectively, and $U ^ { \star } = W _ { \mathrm { t r u e } } ^ { - 1 } Q _ { \mathrm { t r u e } } ^ { - 1 } P _ { \mathrm { t r u e } } = W _ { \mathrm { t r u e } } ^ { - 1 } W _ { \mathrm { t r u e } } ^ { \circ }$ is all pass. Notice that passing to the limit as $N \to \infty$ in the above equalities involves an interchange of the limit with the integral which can be justified by the dominated convergence theorem. In fact, one can argue along the lines of the proof of Theorem 1 to show that the eventual optimal solution $( { \bf P } ^ { ( \infty ) } , { \bf Q } ^ { ( \infty ) } )$ must remain in a compact (sublevel) set. In summary, without regularization a global point of minimum for $\check { J } _ { \tau , \infty }$ corresponds to the true model.

## VI. SIMULATIONS

In this section, we present some numerical simulations to assess the performance of the spectral estimator proposed in Section V. More precisely, we implement an estimation procedure starting from a finite sample path of a stationary process and test the alternating minimization scheme (67) and (68). Notice that the numerical solution of the regularized dual problem (36) by means of Newton’s method was presented in the conference paper [27, Sec. 4] along with many implementation details, including the construction of a basis for the vector space containing the dual variables, the computation of the gradient and the Hessian of the regularized dual function, and the initialization of the dual variables. Therefore, these details are omitted here.

## A. Parameters setting

In order to showcase the spectral estimation procedure described in Section V, we consider a vector-valued process $\{ y _ { t } \} _ { t \in \mathbb { Z } }$ with $m \ : = \ : 3$ scalar components which admits the unilateral ARMA representation of order two:

$$
\sum _ { k = 0 } ^ { 2 } A _ { k } y _ { t - k } = \sum _ { k = 0 } ^ { 2 } B _ { k } w _ { t - k } , \quad t \in \mathbb { Z }\tag{77}
$$

where

$$
A _ { 0 } = \left[ { \begin{array} { c c c } { 2 . 3 9 } & { 0 } & { 0 } \\ { - 0 . 2 8 } & { 2 . 5 2 } & { 0 } \\ { - 0 . 1 7 } & { - 0 . 3 2 } & { 0 . 9 7 } \end{array} } \right] ,
$$

$$
A _ { 1 } = \left[ 0 . 4 4 0 . 0 1 0 . 0 6 \right] ,\tag{78}
$$

$$
A _ { 2 } = \left[ { \begin{array} { c c c } { - 1 . 1 3 } & { 0 . 4 4 } & { - 0 . 1 9 } \\ { - 2 . 5 5 } & { 1 } & { - 0 . 4 9 } \\ { 0 . 5 8 } & { - 0 . 7 1 } & { 0 . 4 3 } \end{array} } \right] ,
$$

$$
\begin{array} { r l } & { B _ { 0 } = \mathrm { d i a g } ( 1 , 1 , 0 . 9 5 ) , } \\ & { B _ { 1 } = \mathrm { d i a g } ( - 1 . 6 6 7 4 , 0 . 9 0 9 6 , - 0 . 5 2 1 3 ) , ~ \mathrm { a n d } } \\ & { B _ { 2 } = \mathrm { d i a g } ( 0 . 9 0 2 5 , 0 . 9 0 2 5 , 0 . 2 3 7 5 ) } \end{array}
$$

are the matricial parameters, and $\{ w _ { t } \} _ { t \in \mathbb { Z } }$ is a twodimensional white Gaussian noise process of unit variance, i.e., $\mathbb { E } w _ { t } w _ { t } ^ { T } ~ = ~ I$ . The spectral density of the process y<sub>t</sub> is $\begin{array} { l l l } { \Phi ^ { \circ } ( e ^ { i \vartheta } ) ^ { \circ } \ = } & { A ( z ) ^ { - 1 } \bar { B ( z ) } B ( z ^ { - 1 } ) ^ { T } A ( z ^ { - 1 } ) ^ { - T } { \big | } _ { \substack { \scriptscriptstyle { \mathscr { r } = \mathscr { e } ^ { i \vartheta } } } } ^ { \circ } } \end{array}$ with $\begin{array} { r } { A ( z ) ~ = ~ \sum _ { k = 0 } ^ { 2 } A _ { k } z ^ { - k } } \end{array}$ and $\begin{array} { r } { B ( z ) ~ = ~ \sum _ { k = 0 } ^ { 2 } \bar { B _ { k } } z ^ { - k } } \end{array}$ The peculiarity of this model is that two polynomials in the main diagonal of $B ( z )$ have $\mathrm { z e r o s } ^ { 2 }$ very close to the unit circle (which are $0 . 9 \dot { 5 } e ^ { \pm . 5 i }$ and $0 . 9 5 e ^ { \pm 2 . 0 7 i }$ , respectively). Thus, it is difficult to approximate such process using an AR process. By construction the system (77) is BIBO stable. Thus a sample path $y _ { 1 } , \ldots , y _ { N }$ of $y _ { t }$ of length $N = 5 0 0$ is generated from (77) via standard filtering techniques. The f-periodogram in (60) can then be evaluated with $f ( N ) ~ = ~ \sqrt [ 3 ] { N }$ . To ensure that $\hat { \Phi } _ { N }$ after the truncation is positive definite on $\mathbb { T } ,$ a positive multiple of the identity matrix is added to it whenever necessary.

For the alternating minimization scheme (67) and (68), we terminate the stage when the distance between two successive iterates $( \mathbf { P } ^ { ( \ell + 1 ) } , \mathbf { \bar { Q } } ^ { ( \ell + 1 ) } )$ and $( \mathbf { P } ^ { ( \ell ) } , \mathbf { Q } ^ { ( \ell ) } )$ , as measured by the norm of their difference, is less than $1 0 ^ { - 3 }$

![](images/00f2e3964a9ee194e2d9ce5e3de80399e83d2bf1bf3dac934eb58f250b6bb2c2.jpg)  
Figure 1: Relative error in the spectrum reconstruction as a function of $\vartheta$ which is averaged over 100 trials of the Monte Carlo simulation.

## B. Simulation results

We compute an estimate of $\Phi ^ { \circ }$ from the sample path using the following four estimators:

• VARMA which denotes the vector-valued ARMA spectral estimator proposed in Section V with $U ^ { ( 0 ) } \equiv I$ and $\hat { W } _ { N }$ in (62) is computed using Bauer’s method [28];

• VAR-ma which denotes the vector-valued ARMA spectral estimator whose MA part is scalar [1], [12], [13]; see also Remark 1;

• VAR which denotes the Hellinger spectral estimator with a noninformative prior Ψ = I [8], [29], [30] such that the spectrum corresponds to a vector-valued AR model;

${ \mathrm { V A R } } _ { \mathrm { M E } }$ which denotes the maximum entropy spectral estimator [11], [31], i.e., the spectrum corresponds to a vector-valued AR model;

Notice that VAR and ${ \tt V A R } _ { \mathrm { M E } }$ use only the covariances lags $\{ \hat { R } _ { k } \} _ { k = 0 } ^ { n }$ , while in addition, VARMA uses the matricial Wcepstral coefficients $\{ \hat { C } _ { k } \} _ { k = 1 } ^ { n }$ (see Definition 1) and VARma uses the scalar cepstral coefficients $\{ \hat { c } _ { k } \} _ { k = 1 } ^ { n } ( \mathrm { i . e }$ ., the scalar logarithmic moments mentioned in Remark 1). The regularization parameter for the cepstral matching is set as $\tau = 1 0 ^ { - 3 }$ for both VARMA and VAR-ma. For all estimators, the number of retained terms n, for both covariance lags and cepstral coefficients, is set equal to 2. The W-cepstral coefficients used by VARMA are computed via (72) and the scalar logarithmic moments used by VAR-ma are computed by substituting the true spectrum with the f-periodogram in the standard formula.

We perform a Monte Carlo simulation of 100 trials using the true model (77) and (78). In each trial, we generate a sample path of $y _ { t }$ of length 500 from which we can compute the covariance estimates, the periodogram, and the cepstral coefficients. Then the four spectral estimators listed above are constructed and compared. Two criteria of comparison are adopted. One is the relative error of the reconstructed spectral

![](images/90e2ddc8b82a890b737552d8a29b40c5925ffb863abbfdcaa797c12f6816e2f6.jpg)  
Figure 2: Boxplot of the integrated relative error for the four spectral estimators.

density $\hat { \Phi }$ which is defined as

$$
\frac { \lVert \Phi ^ { \circ } ( e ^ { i \vartheta } ) - \hat { \Phi } ( e ^ { i \vartheta } ) \rVert _ { \mathrm { F } } } { \lVert \Phi ^ { \circ } ( e ^ { i \vartheta } ) \rVert _ { \mathrm { F } } }
$$

and is a function of $\vartheta .$ The other is called “integrated relative error” which is simply

$$
\int \frac { \| \Phi ^ { \circ } - \hat { \Phi } \| _ { \mathrm { F } } } { \| \Phi ^ { \circ } \| _ { \mathrm { F } } }
$$

and is real-valued. Fig. 1 shows the relative error averaged over the 100 trials, and Fig. 2 shows the boxplot of the integrated relative error for each one of the four spectral estimators. We can clearly see that VARMA outperform the other estimators. In particular, it is interesting to note that although VAR-ma estimates the spectral density of an ARMA process, its inferior performance is due to the fact that the latter searches the process over a special class of ARMA processes, as pointed out in Remark 1.

We have also tested the impact of $U ^ { ( 0 ) }$ , the initial inner function for the alternating minimization scheme (67) and (68). More precisely, we fix a sample path of length 500 of the true model in Subsec. VI-A. Then we do another Monte Carlo simulations of 100 trials to compute the VARMA estimate, and each trial starts from a different $\mathbf { \widehat { \boldsymbol { U } } } ^ { ( 0 ) }$ whose value is a constant orthogonal matrix for all $\vartheta \in \mathbb { T }$ . Such an orthogonal matrix is generated by applying the QR factorization to a random square matrix with i.i.d. standard normal entries [32]. In the end, we compute the difference between two estimators $\hat { \Phi } _ { j }$ and $\hat { \Phi } _ { k }$ where the subscripts indicate the index of a trial. We find that $\begin{array} { r } { \int \| \hat { \Phi } _ { j } - \hat { \Phi } _ { k } \| _ { \mathrm { F } } \approx 1 0 ^ { - 4 } } \end{array}$ for all $j \neq k$ , which means that all resulting spectral estimates are essentially the same.

## VII. CONCLUSIONS

We have developed a new paradigm for spectral estimation of vector-valued ARMA (VARMA) processes based on the philosophy of Maximum Entropy. A new transportation entropy is employed as the objective function and a new concept of matricial cepstral coefficients, along with the covariances, is used in the equality constraints to select the spectrum. The optimization problem is treated via Lagrange dual analysis. We show that the dual problem can be made well-posed when a suitable regularization term is added. A spectral estimation procedure based on a finite sample path of a random process has also been proposed. Numerical simulations confirm the advantage of our estimator when the true spectrum has a nontrivial zero structure.

For future work, it is possible to exploit the subspace inclusion condition (26) or (27) via techniques in semidefinite programming in order to obtain an estimation theory for lowrank spectral densities [33], [34].

## REFERENCES

[1] M. Zorzi, “On the identification of ARMA graphical models,” IEEE Transactions on Automatic Control, vol. 70, no. 1, pp. 403–414, 2025.

[2] W. Cao and A. Li, “ARMAX identification of low rank graphical models,” arXiv preprint: 2501.09616, 2025.

[3] P. Stoica and R. Moses, Spectral Analysis of Signals. Upper Saddle River, NJ: Pearson Prentice Hall, 2005.

[4] D. R. Brillinger, Time Series: Data Analysis and Theory. SIAM, 2001.

[5] R. S. Tsay, Analysis of Financial Time Series. John Wiley & Sons, 2005.

[6] C. Byrnes, P. Enqvist, and A. Lindquist, “Cepstral coefficients, covariance lags, and pole-zero models for finite data strings,” IEEE Transactions on Signal Processing, vol. 49, no. 4, pp. 677–693, 2001.

[7] P. Enqvist, “A convex optimization approach to ARMA(n, m) model design from covariance and cepstral data,” SIAM Journal on Control and Optimization, vol. 43, no. 3, pp. 1011–1036, 2004.

[8] A. Ferrante, M. Pavon, and F. Ramponi, “Hellinger versus Kullback– Leibler multivariable spectrum approximation,” IEEE Transactions on Automatic Control, vol. 53, no. 4, pp. 954–967, 2008.

[9] A. Ferrante, C. Masiero, and M. Pavon, “Time and spectral domain relative entropy: A new approach to multivariate spectral estimation,” IEEE Transactions on Automatic Control, vol. 57, no. 10, pp. 2561– 2575, 2012.

[10] M. Zorzi, “A new family of high-resolution multivariate spectral estimators,” IEEE Transactions on Automatic Control, vol. 59, no. 4, pp. 892–904, 2014.

[11] J. P. Burg, “Maximum entropy spectral analysis,” Ph.D. dissertation, Department of Geophysics, Stanford University, 1975.

[12] A. Lindquist, C. Masiero, and G. Picci, “On the multivariate circulant rational covariance extension problem,” in IEEE 52nd Annual Conference on Decision and Control (CDC), 2013, pp. 7155–7161.

[13] A. Lindquist and G. Picci, “Modeling of stationary periodic time series by ARMA representations,” in Optimization and Its Applications in Control and Data Sciences: In Honor ofBoris T. Polyak’s 80th Birthday. Springer, 2016, pp. 281–314.

[14] A. Blomqvist, A. Lindquist, and R. Nagamune, “Matrix-valued Nevanlinna–Pick interpolation with complexity constraint: An optimization approach,” IEEE Transactions on Automatic Control, vol. 48, no. 12, pp. 2172–2190, 2003.

[15] B. Zhu and M. Zorzi, “A well-posed multidimensional rational covariance and generalized cepstral extension problem,” SIAM Journal on Control and Optimization, vol. 61, no. 3, pp. 1532–1556, 2023.

[16] ——, “On the statistical consistency of a generalized cepstral estimator,” IEEE Transactions on Automatic Control, vol. 69, no. 9, pp. 5775–5787, 2024.

[17] M. Zorzi, “Optimal transport between Gaussian random fields,” in the 29th Mediterranean Conference on Control and Automation (MED), 2021, pp. 186–191.

[18] ——, “Optimal transport between Gaussian stationary processes,” IEEE Transactions on Automatic Control, vol. 66, no. 10, pp. 4939–4944, 2020.

[19] T. Cover and A. Thomas, Elements ofInformation Theory. Wiley, 2006.

[20] A. Ringh, J. Karlsson, and A. Lindquist, “Multidimensional rational covariance extension with applications to spectral estimation and image compression,” SIAM Journal on Control and Optimization, vol. 54, no. 4, pp. 1950–1982, 2016.

[21] M. Zorzi, “Rational approximations of spectral densities based on the Alpha divergence,” Mathematics of Control, Signals, and Systems, vol. 26, no. 2, pp. 259–278, 2014.

[22] R. A. Horn and C. R. Johnson, Matrix Analysis, 2nd ed. Cambridge University Press, 2013.

[23] T. Kato, Perturbation Theory for Linear Operators, 2nd ed., ser. Classics in Mathematics. Springer Science & Business Media, 2012, vol. 132.

[24] P. D. Lax, Linear Algebra and Its Applications, 2nd ed. Wiley, 2013.

[25] B. Zhu, A. Ferrante, J. Karlsson, and M. Zorzi, “M<sup>2</sup>-spectral estimation: A relative entropy approach,” Automatica, vol. 125, March 2021.

[26] L. Falconi, A. Ferrante, and M. Zorzi, “Mean-square consistency of the f-truncated M<sup>2</sup>-periodogram,” Automatica, vol. 147, p. 110672, 2023.

[27] B. Zhu and M. Zorzi, “Multivariate spectral estimation using the Wcepstral coefficients,” Accepted for presentation at the 23rd IFAC World Congress (IFAC 2026), 2026.

[28] A. H. Sayed and T. Kailath, “A survey of spectral factorization methods,” Numerical Linear Algebra with Applications, vol. 8, no. 6-7, pp. 467– 496, 2001.

[29] F. Ramponi, A. Ferrante, and M. Pavon, “A globally convergent matricial algorithm for multivariate spectral estimation,” IEEE Transactions on Automatic Control, vol. 54, no. 10, pp. 2376–2388, 2009.

[30] ——, “On the well-posedness of multivariate spectrum approximation and convergence of high-resolution spectral estimators,” Systems & Control Letters, vol. 59, no. 3, pp. 167–172, 2010.

[31] T. T. Georgiou, “Spectral analysis based on the state covariance: the maximum entropy spectrum and linear fractional parametrization,” IEEE Transactions on Automatic Control, vol. 47, no. 11, pp. 1811–1823, 2002.

[32] P. Diaconis, “Patterns in eigenvalues: The 70th Josiah Willard Gibbs lecture,” Bulletin of the American Mathematical Society, vol. 40, no. 2, pp. 155–178, 2003.

[33] W. Cao, G. Picci, and A. Lindquist, “Identification of low rank vector processes,” Automatica, vol. 151, p. 110938, 2023.

[34] W. Cao, A. Lindquist, and G. Picci, “Modeling of low rank time series,” IEEE Transactions on Automatic Control, vol. 68, no. 12, pp. 7270– 7285, 2023.