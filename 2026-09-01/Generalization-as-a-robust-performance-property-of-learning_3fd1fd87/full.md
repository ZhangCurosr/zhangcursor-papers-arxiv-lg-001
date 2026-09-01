# Generalization as a robust performance property of learning-enabled dynamical systems

Filippo Fabiani

## Abstract

By focusing on algorithmic stability as a means of establishing out-of-sample bounds, we provide a system-theoretic interpretation of generalization in learning-enabled dynamical systems arising in data-driven optimization and feedback control approximation. Given two neighboring datasets, we specifically model sample replacement as an exogenous disturbance acting on a sensitivity system, while the incremental behavior of the data-dependent operator is encoded through an integral quadratic constraint. By relying on dissipativity arguments, we establish a matrix inequality-based certificate and a uniform stability bound that separates the one-sample sensitivity of the learned operator, and an algorithmdependent dynamical gain. The latter can then be optimized, offering a tractable tool for certifying and comparing generalization capabilities of learning dynamics. We show that our results recover classical ones for gradient descent, apply naturally to momentum-based methods such as heavy-ball and Nesterov acceleration, and extend to data-driven control.

## I. INTRODUCTION

In machine learning, generalization refers to the ability of a certain learning-based model to perform well on new, unseen data, rather than only on the samples used for training [1]. Thus, ensuring excellent generalization capabilities is recommended in modern dynamical systems whose evolution is governed by a learning policy $\hat { \Phi } ( \cdot , S _ { N } )$ trained on dataset $ { \boldsymbol { S } } _ { N }$ with N samples.

A prominent example includes iterative methods for data-driven optimization and empirical risk minimization (ERM) [2], [3], which may overfit in case of poor generalization. Another suitable instance stems from feedback control approximation, where data-driven proxies are employed to steer the system evolution [4], [5]. Here, poor out-of-sample capabilities may lead to, e.g., degraded performance, or even instability.

In this context, algorithmic stability [6] is a central tool to explain the generalization behavior of learning algorithms. In its classical form, stability quantifies the change in an algorithm’s output, measured through a prescribed loss function, when a single training sample is replaced. When this change is small uniformly over neighboring datasets $ { \boldsymbol { S } } _ { N }$ and $ { \mathcal { S } } _ { N } ^ { j }$ , where the latter differs from the former in the j-th sample, one obtains high-probability generalization guarantees through concentration arguments [7]. A standard approach to perform the stability analysis of iterative procedures consists in deriving algorithm-specific recursive bounds on the distance between two trajectories generated from neighboring datasets, and then rely on the contraction/nonexpansive property of the data-driven mapping $\hat { \Phi } ( \cdot , S _ { N } )$ [8]–[15]. Although effective, such arguments do not explicitly capture how the data perturbation is attenuated or amplified by the algorithmic dynamics over iterations, thereby obscuring the inherent input-output (IO) structure underlying the algorithmic stability mechanism.

This is precisely the perspective we offer. By focusing on a broad class of discrete-time learning dynamics driven by a data-dependent operator $\hat { \Phi } ( \cdot , S _ { N } )$ , we compare the system trajectories corresponding to two neighboring datasets $ { \boldsymbol { S } } _ { N }$ and $S _ { N } ^ { j }$ , obtaining a data sensitivity system whose induced operator discrepancy can be decomposed into two components. The first consists of the sensitivity propagated through $\hat { \Phi } ( \cdot , S _ { N } )$ itself, while the second one is the extra sensitivity injected due to sample replacement. Acting as an exogenous disturbance channel, the latter enables our interpretation of algorithmic stability as a robust performance property from this data perturbation channel to the loss function. Our analysis hence relies on dissipativity arguments, where the available information on the regularity of $\hat { \Phi } ( \cdot , S _ { N } )$ is encoded through an incremental quadratic constraint (IQC) [16], while the effect of replacing one sample is captured by a uniform data-injection bound. A storage function then certifies a finite gain from the data perturbation to the terminal (i.e., after T steps) state sensitivity. Capitalizing on this, we develop an optimizationbased procedure to quantify the algorithmic stability bound, and hence generalization capabilities, of the learning-enabled system at hand. In particular, the (uniform) stability coefficient decomposes in two main terms: i) the one-sample sensitivity of the data-dependent operator, which scales as $1 / N$ in several relevant settings, and ii) an algorithm-dependent dynamical sensitivity gain certified by a dissipativity inequality. Such a decomposition is our main conceptual message: generalization depends on both how strongly a single sample perturbs $\hat { \Phi } ( \cdot , S _ { N } )$ , and how the closed-loop dynamics dissipates or amplifies the injected sensitivity.

## A. Related work

To the best of our knowledge, only a few recent works have considered algorithmic stability under a system-theoretic lens.

Of particular relevance, [17] focused on the stability of Nesterov accelerated gradient (NAG) method, and established a sharp distinction between quadratic and general smooth convex costs. As a main conclusion, it turned out that acceleration may strongly amplify small perturbations and that convergence guarantees alone do not determine the algorithmic stability of an optimization dynamics. The present work, instead, offers a complementary vision by formulating the propagation of data perturbations as an IO property of a general sensitivity system. In our framework, the amplification phenomena identified in [17] are naturally reflected in the dynamical gain from the data injection channel to the terminal hypothesis.

In [18] it was shown, instead, that a contractive iterative scheme in a suitable Riemannian metric is uniformly stable, obtaining bounds proportional to the inverse of both the contraction rate and the number of samples. Although similar in the spirit, our approach differs in the robustness property being certified. In fact, [18] derived stability from the contraction of neighboring trajectories, while in this paper sample replacement is explicitly represented as an exogenous input, and uniform stability is obtained from a dissipativity certificate bounding the resulting induced effect. This allows one to separate the sensitivity of $\hat { \Phi } ( \cdot , S _ { N } )$ to the dataset from the ability of the dynamics to dissipate or amplify the resulting perturbation.

In [19] the authors studied uniform stability of first-order accelerated schemes in the smooth, strongly convex quadratic regime. Here, accelerated methods were represented as Lur’e-type feedback interconnections between a linear dynamical system and a gradient oracle, encoded smoothness and strong convexity through sector IQCs, and searched for quadratic Lyapunov functions via (LMIs). Besides adopting a similar state-space and IQC machinery, our contribution starts from a different decomposition of the data sensitivity dynamics, including the disturbance channel caused by the sample replacement, which is not absorbed into an optimization-specific recursion. This also leads to a broader scope. Unlike [19], our analysis is indeed not limited to first-order optimization, since $\hat { \Phi } ( \cdot , S _ { N } )$ may correspond to the empirical gradient or a data-driven monotone operator evaluated along the algorithmic trajectory or, more generally, an operator learned from data before the dynamics is deployed, for instance through least-squares (LS) or kernel ridge regression (KRR).

## B. Summary of contributions and paper organization

Our framework allows one to establish a modular connection between statistical generalization and robust control, as it essentially shows that a learning-based dynamics generalizes when the sensitivity injected by a one-sample replacement is sufficiently dissipated by the closed-loop interconnection.

In summary, this paper makes the following contribution:

i) We recast algorithmic stability of learning-enabled dynamics as an IO robustness property. Specifically, we show that uniform stability can be interpreted as a bounded-gain property from the exogenous disturbance generated by sample replacement to the loss discrepancy;

ii) We derive a dissipativity condition, in the form of a (MI), that allows one to certify such a bounded-gain property. This yields a uniform stability bound that separates the statistical sensitivity of $\hat { \Phi } ( \cdot , S _ { N } )$ from the amplification or dissipation induced by the dynamics;

iii) We formulate the computation of the dynamical sensitivity gain as a semidefinite programming problem (SDP) complemented by a scalar search, thereby providing a systematic way to optimize and compare the stability property, and hence generalization, of different dynamics;

iv) We show that our framework recovers standard contractivity-based stability mechanisms for gradient descent (GD), while also applying naturally to lifted methods with memory, such as heavy ball (HB) and NAG, for which contractivity may fail. Interestingly, numerical experiments show that nominal convergence and data sensitivity are distinct design objectives.

In addition, we show that several mappings $\hat { \Phi } ( \cdot , S _ { N } )$ of practical interest are characterized by one-sample perturbation scaling as $1 / N$ , allowing one to obtain consistent, exponential, highprobability bounds. Finally, we emphasize how our results apply to a broad class of data-dependent operators $\hat { \Phi } ( \cdot , S _ { N } )$ , including those learned offline via LS [20] or KRR [4], and illustrate them on a kernel-based residual-controller example.

The rest of the paper is organized as follows: in §II we formalize the problem addressed, provide basic notions of algorithmic stability and a system-theoretic connection. In §III we develop a dissipativity-based approach to optimize uniform stability, which can be interpreted as a robust control problem, and discuss the computational aspects. In §IV and V, instead, we apply our technique to characterize the generalization properties of popular data-based optimization and learning-based dynamics, also performing numerical experiments. Finally, in Appendix A are reported examples of common operators characterized by one-sample perturbation scaling as $1 / N$

## II. PROBLEM FORMULATION AND PRELIMINARIES

Throughout this paper, we will consider the following control-affine, nonlinear, discrete-time dynamics:

$$
\left\{ \begin{array} { l } { { \displaystyle x _ { k + 1 } = A x _ { k } + B \hat { \Phi } ( y _ { k } , S _ { N } ) } , } \\ { { \displaystyle y _ { k } = C x _ { k } + D \hat { \Phi } ( y _ { k } , S _ { N } ) } , } \end{array} \right.\tag{1}
$$

where the vector of state variables $x \in \mathbb { R } ^ { n }$ evolves according to an autonomous term through $A \in \mathbb { R } ^ { n \times n }$ , and a deterministic mapping $\hat { \Phi } : \mathbb { R } ^ { p } \times \Xi ^ { N }  \mathbb { R } ^ { m }$ , premultiplied by $B \in \mathbb { R } ^ { n \times m }$ . The output equation determining the measurement vector $y \in \mathbb { R } ^ { p }$ , instead, is obtained with matrices $C \in \mathbb { R } ^ { p \times n }$ and $D \in \mathbb { R } ^ { p \times m }$

Given a training set $\begin{array} { r } { S _ { N } : = \{ \xi ^ { ( i ) } \} _ { i = 1 } ^ { N } } \end{array}$ consisting of N independent and identically distributed (i.i.d.) samples $\boldsymbol { \xi } ^ { ( i ) } \in \Xi \subseteq \mathbb { R } ^ { d }$ with unknown statistics $\mathbb { P } ,$ the operator $\hat { \Phi }$ either directly manipulates these data as in, e.g., sample-average approximation (SAA) and kernel-based techniques, or amounts to a data-driven map learned offline via, e.g., LS or KRR.

For the well-posedness of the implicit interconnection occurring in the output of (1), we will assume the following:

Standing Assumption 2.1. For every $N \geq 1 , S _ { N } \in \Xi ^ { N }$ and $x \in \mathbb { R } ^ { n } , y = C x + D \hat { \Phi } \left( y , S _ { N } \right)$ admits a unique solution. □

While trivially satisfied when $D = 0$ , also verifying that $y \mapsto C x + D \hat { \Phi } ( y , S _ { N } )$ is a contraction uniformly in x allows one to meet the requirement in Standing Assumption 2.1.

Note that the intrinsic dependency of $\hat { \Phi }$ on $ { \boldsymbol { S } } _ { N }$ may produce substantially different evolutions for the data-enabled dynamics in (1). We will thus explore the generalization properties possessed by the underlying dynamics, and establish rigorous certificates on the sensitivity of the resulting trajectories with respect to (w.r.t.) changes in the training set $ { \boldsymbol { S } } _ { N }$ . Specifically, we will analyze algorithmic stability as a possible means for generalization, and develop suitable connections bridging this concept with standard system-theoretic dissipativity notions.

## A. Preliminaries on algorithmic stability

We define a learning algorithm as a mapping $\mathcal { A } _ { N } : \mathbb { R } ^ { n } \times \Xi ^ { N } \to \mathbb { R } ^ { n }$ , taking some $N -$ multisample and returning a hypothesis $H _ { N } : = \mathcal { A } _ { N } ( S _ { N } ) \in \mathbb { R } ^ { n }$ . By identifying $A _ { N } ( S _ { N } )$ with the dynamics in (1) observed for T iterations starting from some $x _ { 0 } .$ , we simply have $H _ { N } =$ $x _ { T }$ . We call $ { \mathcal { S } } _ { N } ^ { j }$ neighboring set, as it is obtained by replacing the j-th element of $S _ { N } , \mathrm { i } . e .$ $\begin{array} { r } {  { \mathcal { S } } _ { N } ^ { j } = \{  { \xi } ^ { ( 1 ) } ,  { \mathrm { ~ \dots ~ } } ,  { \xi } ^ { ( j - 1 ) } ,  { \xi } ^ { \prime } ,  { \xi } ^ { ( j + 1 ) } ,  { \mathrm { ~ \dots ~ } } ,  { \xi } ^ { ( N ) } \} } \end{array}$ , where $\xi ^ { \prime } \in \Xi$ is drawn according to $\mathbb { P } _ { \mathrm { : } }$ , i.i.d. w.r.t. $S _ { N } \setminus \{ \xi ^ { ( j ) } \}$ . In this case, $H _ { N ^ { j } } : = \mathcal { A } _ { N } ( \boldsymbol { S } _ { N } ^ { j } )$ , which in respect to (1) yields $H _ { N ^ { j } } = x _ { T } ^ { \prime }$

The accuracy of the algorithm prediction (i.e., the hypothesis produced) is generally measured according to some loss function $\ell : \mathbb { R } ^ { n } \times \Xi \to \mathbb { R } _ { > 0 }$ . The generalization error is typically attached to the loss, and coincides with the following random variable (the hypothesis $x _ { T }$ indeed depends on ${ \cal { S } } _ { N } )$ :

$$
r ( \mathcal { A } _ { N } ) = \mathbb { E } _ { \mathbb { P } } \left[ \ell ( H _ { N } , \xi ) \right] .\tag{2}
$$

Note that computing (2) would require $\mathbb { P } ,$ which is unavailable in the considered framework. Nevertheless, the simplest estimator for (2) amounts to the empirical error, defined as:

$$
\hat { r } ( A _ { N } , S _ { N } ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \ell ( H _ { N } , \xi ^ { ( j ) } ) .\tag{3}
$$

Definition 2.2 ([6, Def. 6] Uniform stability). An algorithm $\mathcal { A } _ { N }$ has uniform stability $\eta \geq 0$ w.r.t. the loss function ℓ if for all $N \geq 1 , S _ { N } \in \Xi ^ { N }$ , and $j \in \{ 1 , \ldots , N \}$ ,

$$
\operatorname* { s u p } _ { \xi \in \Xi } | \ell ( H _ { N } , \xi ) - \ell ( H _ { N ^ { j } } , \xi ) | \leq \eta .
$$

In words, an algorithm is said to be uniformly stable if changing one sample $\xi ^ { ( j ) }$ in the training dataset does not substantially alter the output w.r.t. ℓ. In general, the smaller the η the better the stability features possessed by the algorithm at hand. The following result links the uniform stability and the generalization properties of $\mathcal { A } _ { N }$ , by providing a probabilistic bound on the generalization gap, $r ( \mathcal { A } _ { N } ) - \hat { r } ( \mathcal { A } _ { N } , \mathcal { S } _ { N } )$ :

Lemma 2.3 ([21, Th. 3.2]). Let $\mathcal { A } _ { N }$ be η-uniformly stable w.r.t. a loss function $0 \le \ell ( H _ { N } , \xi ) \le M$ $M \geq 0 ,$ , for all $\xi \in \Xi$ and ${ \cal S } _ { N } \in \Xi ^ { N }$ . Then, for any $N \geq 1$ and $\delta \in ( 0 , 1 )$ , it holds that:

$$
\mathbb { P } ^ { N } \left\{ \mathcal { S } _ { N } \in \Xi ^ { N } : r ( A _ { N } ) - \hat { r } ( A _ { N } , \mathcal { S } _ { N } ) \leq \eta + ( N \eta + M ) \sqrt { 2 \ln ( 1 / \delta ) / N } \right\} \geq 1 - \delta .
$$

Specifically, Lemma 2.3 provides with an upper bound on the risk associated to hypothesis $H _ { N } = \mathcal { A } _ { N } ( \mathcal { S } _ { N } )$ , which holds with arbitrary confidence $1 - \delta .$ , such that i) if $\eta$ scales proportional to $1 / N$ , vanishes as $N  \infty$ , and ii) it is exponential in the confidence parameter $\delta .$ . The former condition is called consistency property, and provides a “tight” certificate in the sense that the upper bound in Lemma 2.3 vanishes as $N \to \infty$

## B. An incremental IO interpretation of algorithmic stability

Algorithmic stability is thus an attractive property, as it allows one to directly verify the finite difference condition required to apply the McDiarmid’s concentration inequality [7] and, consequently, to derive generalization bounds. However, the application of this notion to the data-enabled dynamics in (1), and that obtained with $ { \mathcal { S } } _ { N } ^ { j }$ in place of $ { \boldsymbol { S } } _ { N }$ , turns into:

$$
\operatorname* { s u p } _ { \xi \in \Xi } | \ell ( x _ { T } , \xi ) - \ell ( x _ { T } ^ { \prime } , \xi ) | \leq \eta ,
$$

which basically concentrates on a “snapshot” of the last step of the resulting T-long trajectories, passed through ℓ. From a system-theoretic perspective, this has limited interpretation, since the available information on the transient, which depends on how sensitive is the learned mapping $\hat { \Phi }$ to data is wasted.

With this regard, we note that incremental input-output stability (i-IOS) [22] is the systemtheoretic stability notion closer in the spirit to Definition 2.2. In a generic nonlinear framework, one then considers the following IO dynamics:

$$
\left\{ \begin{array} { c } { x _ { k + 1 } = \Psi ( x _ { k } , u _ { k } ) , } \\ { y _ { k } = h ( x _ { k } ) , } \end{array} \right.\tag{4}
$$

with $\Psi : \mathbb { R } ^ { n } \times \mathbb { R } ^ { m }  \mathbb { R } ^ { n }$ and $h : \mathbb { R } ^ { n }  \mathbb { R } ^ { p }$

Definition 2.4. The system in (4) is i-IOS if there exist class-KL function θ and $c l a s s { - } \mathcal { K } _ { \infty }$ function $\gamma ^ { 1 }$ such that:

$$
\left\| y _ { k } - y _ { k } ^ { \prime } \right\| \leq \theta ( \left\| x _ { 0 } - x _ { 0 } ^ { \prime } \right\| , k ) + \gamma \left( \operatorname* { s u p } _ { 0 \leq i \leq k } \ \left\| u _ { i } - u _ { i } ^ { \prime } \right\| \right) ,
$$

at every $k \geq 0$ , where the output trajectories are generated by (4) for any $x _ { 0 } , \ x _ { 0 } ^ { \prime }$ and input sequences $u , u ^ { \prime } .$ □

Proposition 2.5. For any $y \in \mathbb { R } ^ { p } , \ N \geq 1 , \ S _ { N } \in \Xi ^ { N }$ and $j \in \{ 1 , \ldots , N \}$ , let $\begin{array} { r } { \big \| \hat { \Phi } \big ( y , S _ { N } ^ { j } \big ) - } \end{array}$ $\hat { \Phi } ( y , S _ { N } ) \vert \vert$ be bounded. If $D = 0$ and (1) is i-IOS w.r.t. the loss-output map $z = h ( x ) : = \ell ( x , \cdot )$ then it is also uniformly stable. □

Proof. The control-affine nonlinear dynamics in (1) with $D = 0$ can be equivalently absorbed by $\hat { \Psi } ( x _ { k } , S _ { N } ) + B u _ { k } : = A x _ { k } + B ( \hat { \Phi } ( C x _ { k } , S _ { N } ) + u _ { k } )$ with $u _ { k } = 0$ for all $k \geq 0$ , which yields (4) once redefined the output equation as $z _ { k } : = h ( x _ { k } )$ , with $h : \mathbb { R } ^ { n }  \mathcal { V }$ denoting the loss-output mapping, formally defined as $h ( x ) : = \ell ( x , \cdot )$ . Specifically, Y is the space of bounded real-valued functions on $\Xi ,$ , equipped with supremum norm, namely $\begin{array} { r } { \| h ( x ) \| _ { \mathcal { V } } : = \operatorname* { s u p } _ { \xi \in \Xi } | \ell ( x , \xi ) | } \end{array}$

Then, pick $j \in \{ 1 , \ldots , N \}$ , and consider the following dynamics compatible with $\hat { \Psi } ( x _ { k } , S _ { N } ) +$ $B u _ { k }$ and $S _ { N } ^ { j }$ :

$$
\left\{ \begin{array} { l l } { x _ { k + 1 } ^ { \prime } = \hat { \Psi } ( x _ { k } ^ { \prime } , S _ { N } ^ { j } ) = \hat { \Psi } ( x _ { k } ^ { \prime } , S _ { N } ) + B u _ { k } ^ { \prime } , } \\ { \quad \quad z _ { k } ^ { \prime } = h ( x _ { k } ^ { \prime } ) , } \end{array} \right.
$$

initialized at some $x _ { 0 } ^ { \prime } = x _ { 0 }$ , with $u _ { k } ^ { \prime } : = \hat { \Phi } ( C x _ { k } ^ { \prime } , S _ { N } ^ { j } ) - \hat { \Phi } ( C x _ { k } ^ { \prime } , S _ { N } )$ . Notice that, by definition, $\begin{array} { r } { \| h ( x ) - h ( x ^ { \prime } ) \| _ { \mathcal { V } } = \operatorname* { s u p } _ { \xi \in \Xi } | \ell ( x , \xi ) - \ell ( x ^ { \prime } , \xi ) | } \end{array}$ . Then, if (1) is i-IOS with the redefined output equation, one can readily conclude that it is also uniformly stable, since at the T-th iteration, $H _ { N } = \mathcal { A } _ { N } ( S _ { N } ) = x _ { T }$ , thereby obtaining:

$$
\begin{array} { r l r } {  { \| z _ { T } - z _ { T } ^ { \prime } \| _ { \mathcal { V } } = \| h ( x _ { T } ) - h ( x _ { T } ^ { \prime } ) \| _ { \mathcal { V } } = \underset { \xi \in \Xi } { \operatorname* { s u p } } \ | \ell ( x _ { T } , \xi ) - \ell ( x _ { T } ^ { \prime } , \xi ) | } } \\ & { } & { \quad \leq \gamma ( \underset { 0 \leq k \leq T } { \operatorname* { s u p } } \ \| B ( \hat { \Phi } ( C x _ { k } ^ { \prime } , S _ { N } ^ { j } ) - \hat { \Phi } ( C x _ { k } ^ { \prime } , S _ { N } ) ) \| ) , } \end{array}
$$

which, in view of the properties of the function of class $\kappa _ { \infty } .$ , yields a finite bound on the loss discrepancy in case $\| \Phi ( C x _ { k } ^ { \prime } , S _ { N } ^ { j } ) - \hat { \Phi } ( C x _ { k } ^ { \prime } , S _ { N } ) \|$ is bounded. ■

While i-IOS represents a condition that is difficult to verify in practice, we observe that the resulting stability coefficient in Proposition 2.5 captures the data perturbation effect producing an exogenous disturbance, $u _ { k } ^ { \prime } = \hat { \Psi } ( x _ { k } ^ { \prime } , S _ { N } ^ { j } ) - \hat { \Psi } ( x _ { k } ^ { \prime } , S _ { N } ) = \hat { \Phi } ( C x _ { k } ^ { \prime } , S _ { N } ^ { j } ) - \hat { \Phi } ( C x _ { k } ^ { \prime } , S _ { N } )$ , through the evolution of two compatible trajectories. The combination of these insights motivates us to resort to a more tractable dissipativity-based approach, which will allow us to directly control the sensitivity of the dynamics in (1) w.r.t. data perturbations, and hence the uniform stability of the resulting learning-enabled dynamics.

## III. DISSIPATIVITY-BASED CERTIFICATION OF ALGORITHMIC STABILITY

Inspired by the previous discussion, we will explore next how algorithmic stability shall not be considered just a static property of learning-enabled dynamics, but rather the output of a data sensitivity system. We will make this interpretation explicit by treating the effect of replacing one training sample as an exogenous disturbance acting on a sensitivity system, and show that the resulting stability coefficient decomposes into a data-injection term and a dynamical sensitivity gain.

## A. Sensitivity dynamics and uniform stability certificates

Given neighboring datasets $ { \boldsymbol { S } } _ { N }$ and $ { \mathcal { S } } _ { N } ^ { j }$ , let us then consider the two trajectories $( x _ { k } , y _ { k } )$ and $( x _ { k } ^ { \prime } , y _ { k } ^ { \prime } )$ generated by (1) and

$$
\left\{ \begin{array} { r } { x _ { k + 1 } ^ { \prime } = A x _ { k } ^ { \prime } + B \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } ^ { j } ) , } \\ { y _ { k } ^ { \prime } = C x _ { k } ^ { \prime } + D \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } ^ { j } ) , } \end{array} \right.
$$

respectively, with the same initial condition $x _ { 0 } = x _ { 0 } ^ { \prime }$ . By defining $\Delta x _ { k } : = x _ { k } - x _ { k } ^ { \prime }$ and $\Delta y _ { k } : =$ $y _ { k } - y _ { k } ^ { \prime }$ , one can immediately decompose the input difference as follows:

$$
\begin{array} { r l } & { \Delta u _ { k } : = \hat { \Phi } ( y _ { k } , S _ { N } ) - \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } ^ { j } ) } \\ & { \qquad = \underbrace { \hat { \Phi } ( y _ { k } , S _ { N } ) - \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } ) } _ { = : \nu _ { k } } + \underbrace { \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } ) - \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } ^ { j } ) } _ { = : w _ { k } } . } \end{array}
$$

Here, $\nu _ { k }$ is the sensitivity propagated through the same data-based mapping, whereas $w _ { k }$ is the new sensitivity injected by replacing one sample. Hence, the sensitivity system reads as:

$$
\left\{ \begin{array} { r } { \Delta x _ { k + 1 } = A \Delta x _ { k } + B ( \nu _ { k } + w _ { k } ) , } \\ { \Delta y _ { k } = C \Delta x _ { k } + D ( \nu _ { k } + w _ { k } ) . } \end{array} \right.\tag{5}
$$

Next, we make suitable assumptions on the mapping $\hat { \Phi }$

Standing Assumption 3.1. For any $N \geq 1$ and ${ \cal S } _ { N } \in \Xi ^ { N }$ , the following conditions hold true:

(i) $y \mapsto \hat { \Phi } ( y , S _ { N } )$ satisfies an incremental IQC, i.e., there exists a multiplier $\Pi \in \mathbb { S } ^ { m + p }$ such that

$$
\left[ \Delta y \right] ^ { \top } \Pi \left[ \begin{array} { c } { \Delta y } \\ { \nu } \end{array} \right] \leq 0 ,\tag{6}
$$

for all admissible IO pairs satisfying $\Delta y = y - y ^ { \prime } \in \mathbb { R } ^ { p } , \ \nu = \hat { \Phi } ( y , S _ { N } ) - \hat { \Phi } ( y ^ { \prime } , S _ { N } ) \in \mathbb { R } ^ { m } ,$

(ii) $\exists \zeta _ { N } > 0 \ s . t .$ , for any $j \in \{ 1 , \ldots , N \}$ and $y \in \mathbb { R } ^ { p } , \| \hat { \Phi } ( y , S _ { N } ) - \hat { \Phi } ( y , S _ { N } ^ { j } ) \| \le \zeta _ { N } .$

The first requirement is a standard assumption in dissipativity-based analysis [24], [25] that will be crucial to let the sensitivity dynamics be dissipative w.r.t. the supply rate defined by Π, a key factor to control the algorithmic stability. The term $w _ { k }$ , instead, is merely treated as an exogenous data perturbation channel. The second condition depends on how the mapping $\hat { \Phi }$ is obtained, or manipulates the samples directly. Remarkably, $\zeta _ { N }$ happens to be proportional to $1 / N$ in several relevant cases, making the generalization bound in Lemma 2.3 tight—see Appendix A for suitable examples and main derivations. We note that the proofs below also apply to the local case in which the neighboring trajectories evolve within some local set $\mathcal { V } \subseteq \mathbb { R } ^ { p }$ , and the developed bounds will hold true with:

$$
\zeta _ { \cal { N } } = \zeta _ { \cal { N } } ( \mathcal { Y } ) : = \operatorname* { s u p } _ { j \in \{ 1 , \dots , N \} } \operatorname* { s u p } _ { y \in \mathcal { Y } } \| \hat { \Phi } ( y , S _ { \cal { N } } ) - \hat { \Phi } ( y , S _ { \cal { N } } ^ { j } ) \| .
$$

Standing Assumption 3.2. The loss function $H _ { N } \mapsto \ell ( H _ { N } , \xi )$ is κ-Lipschitz continuous and so that $0 \le \ell ( H _ { N } , \xi ) \le M$ □

Standing Assumption 3.2 holds, for instance, for absolute, Huber-type or clipped losses, and squared loss functions restricted to compact subsets of the hypothesis space, $\mathcal { X } \subset \mathbb { R } ^ { n }$

Theorem 3.3. Let $( \Delta x _ { k } , \Delta y _ { k } , \nu _ { k } , w _ { k } )$ satisfy the sensitivity dynamics (5) and the IQC (6). Define

$$
\begin{array} { r } { \mathcal { A } : = \left[ A ~ B ~ B \right] , \qquad \mathcal { E } : = \left[ C ~ D ~ D \right] . } \end{array}
$$

Suppose there exist $P \succ 0 , \ \rho , \ \gamma _ { 1 } , \ \gamma _ { 2 } \ \ge \ 0$ such that

$$
\mathcal { M } ( P , \rho , \gamma _ { 1 } , \gamma _ { 2 } ) : = \mathcal { A } ^ { \top } P \mathcal { A } - \mathrm { d i a g } ( \rho ^ { 2 } P , 0 , 0 ) - \gamma _ { 1 } \mathcal { O } ^ { \top } \Pi \mathcal { E } - \mathrm { d i a g } ( 0 , 0 , \gamma _ { 2 } I ) \prec 0 ,\tag{7}
$$

Then, for every $T \geq 1$

$$
\| \Delta x _ { T } \| \leq \sqrt { \frac { \gamma _ { 2 } } { \lambda _ { \operatorname* { m i n } } ( P ) } } \left( \sum _ { k = 0 } ^ { T - 1 } \rho ^ { 2 ( T - k - 1 ) } \| w _ { k } \| ^ { 2 } \right) ^ { 1 / 2 } .\tag{8}
$$

Proof. Given a set of samples $ { \boldsymbol { S } } _ { N }$ , Standing Assumption 2.1 guarantees that the dynamics in (1) is well-posed, and hence for any $x _ { 0 }$ and $ { \boldsymbol { S } } _ { N }$ there exists a unique trajectory $\{ x _ { k } \} _ { k = 0 } ^ { T }$ . The same argument can be likewise applied to $ { \mathcal { S } } _ { N } ^ { j }$ to eventually establish the well-posedness of the sensitivity dynamics in (5).

Then, let us define a storage function $V ( \Delta x ) : = \Delta x ^ { \top } P \Delta x$ , along with $s _ { k } : = \mathrm { c o l } ( \Delta x _ { k } , \nu _ { k } , w _ { k } )$ With matrices $\mathcal { A }$ and $\mathcal { E } ^ { \mathcal { O } }$ defined in the main statement, one readily obtains:

$$
\Delta x _ { k + 1 } = \mathcal { A } s _ { k } , \qquad \left[ \begin{array} { c } { \Delta y _ { k } } \\ { \nu _ { k } } \end{array} \right] = \mathcal { E } s _ { k } .
$$

With the introduced notation, it is thus immediate to verify that the MI in (7) is equivalent to the pointwise dissipativity inequality:

$$
V ( \Delta x _ { k + 1 } ) - \rho ^ { 2 } V ( \Delta x _ { k } ) \leq \gamma _ { 1 } \left[ \Delta y _ { k } \right] ^ { \top } \Pi \left[ \Delta y _ { k } \right] + \gamma _ { 2 } \| w _ { k } \| ^ { 2 } .
$$

Along admissible operator $\hat { \Phi }$ trajectories, the IQC term is nonpositive by (6) in Standing Assumption 3.1.(i). Since $\gamma _ { 1 } \geq 0$ , we immediately obtain:

$$
V ( \Delta x _ { k + 1 } ) \leq \rho ^ { 2 } V ( \Delta x _ { k } ) + \gamma _ { 2 } \| w _ { k } \| ^ { 2 } .
$$

To meet Definition 2.2, the two trajectories have the same initial condition by construction, so $\Delta x _ { 0 } = 0$ , which implies $V ( \Delta x _ { 0 } ) = 0$ . Telescoping the relation above leaves us with:

$$
V ( \Delta x _ { T } ) \leq \gamma _ { 2 } \sum _ { k = 0 } ^ { T - 1 } \rho ^ { 2 ( T - k - 1 ) } \| w _ { k } \| ^ { 2 } .
$$

Finally, by noticing that $V ( \Delta x _ { T } ) \geq \lambda _ { \operatorname* { m i n } } ( P ) \Vert \Delta x _ { T } \Vert ^ { 2 }$ , we directly obtain (8).

Remark 3.4. In case the multiplier class in (6) is a cone, the positive scalar $\gamma _ { 1 }$ can be absorbed in Π directly and one may set $\gamma _ { 1 } = 1$ without loss of generality. If Π is fixed or normalized, however, $\gamma _ { 1 } \geq 0$ remains a decision variable. □

The robust performance object is hence the induced gain from $w$ to the terminal state sensitivity $\Delta x _ { T }$ . The result above isolates this gain before imposing any specific pointwise bound on $w _ { k }$ . It is useful to introduce the weighted data-injection norm $\begin{array} { r } { \| w \| _ { \rho , T } : = \left( \sum _ { k = 0 } ^ { T - 1 } \rho ^ { 2 ( T - 1 - k ) } \| w _ { k } \| ^ { 2 } \right) ^ { 1 / 2 } } \end{array}$ which turns the bound in Theorem 3.3 as $\| \Delta x _ { T } \| \leq \sqrt { \gamma _ { 2 } / \lambda _ { \operatorname* { m i n } } ( P ) } \| w \| _ { \rho , T }$

A uniform stability bound is recovered by bounding the data-injection signal uniformly, yielding the following corollary:

Corollary 3.5. Under the same hypotheses of Theorem 3.3, and in case (7) is solvedfor $\rho \in [ 0 , 1 )$ the algorithm $\mathcal { A } _ { N } ( S _ { N } ) = x _ { T }$ associated with the dynamics in (1) is uniformly stable with

$$
\eta \leq \kappa \zeta _ { N } \sqrt { \frac { \gamma _ { 2 } } { \lambda _ { \operatorname* { m i n } } ( P ) ( 1 - \rho ^ { 2 } ) } } .\tag{9}
$$

Proof. By Standing Assumption 3.1.(ii), $\| w _ { k } \| \leq \zeta _ { N }$ for all $k \geq 0$ . Since $\rho < 1$

$$
\| w \| _ { \rho , T } \leq \zeta _ { N } \left( \sum _ { k = 0 } ^ { T - 1 } \rho ^ { 2 ( T - k - 1 ) } \right) ^ { 1 / 2 } \leq \frac { \zeta _ { N } } { \sqrt { 1 - \rho ^ { 2 } } } .
$$

With $\mathcal { A } _ { N } ( S _ { N } ^ { j } ) = x _ { T } ^ { \prime }$ , for any $j \in \{ 1 , \ldots , N \}$ , from Theorem 3.3 one directly obtains:

$$
\| x _ { T } - x _ { T } ^ { \prime } \| \leq \zeta _ { N } \sqrt { \frac { \gamma _ { 2 } } { \lambda _ { \operatorname* { m i n } } ( P ) ( 1 - \rho ^ { 2 } ) } } .
$$

Using the Lipschitz property of the loss function in Standing Assumption 3.2, su $\mathrm { p } _ { \xi \in \Xi } | \ell ( x _ { T } , \xi ) -$ $\ell ( x _ { T } ^ { \prime } , \xi ) | \leq \kappa \| x _ { T } - x _ { T } ^ { \prime } \|$ , which immediately proves (9). ■

Our dissipativity-based perspective allows one to identify three distinct terms contributing to the stability coefficient, i.e.,

$$
\eta \leq \underbrace { \kappa } _ { \mathrm { l o s s ~ r e a d o u t } } \underbrace { \zeta _ { N } } _ { \mathrm { d a t a ~ i n j e c t i o n } } \underbrace { \sqrt { \lambda _ { \mathrm { m i n } } ( P ) ( 1 - \rho ^ { 2 } ) } } _ { \mathrm { d y n a m i c a l ~ s e n s i t i v i t y ~ g a i n } } .
$$

![](images/faec516d7368854fde6eec35025c35670abe99c6e7cdbce457ceedb45e6654ec.jpg)  
Fig. 1: Feedback interconnection of the sensitivity system (5). Uniform stability is a bounded-gain condition on $w _ { k } \mapsto \Delta z _ { T }$

It turns out that algorithmic stability is directly controlled by how well the sensitivity dynamics either dissipates or amplifies the new data sensitivity injected at each step. Then, to obtain tight generalization bounds consistent with N, e.g., Lemma 2.3, notice that one must have $\zeta _ { N } \propto 1 / N$ which happens to be the case for several learning mappings of interest—see Appendix A.

We conclude by stating the following result that will be useful for the momentum-based schemes discussed in §IV:

Corollary 3.6. Let the hypothesis be $H _ { N } = R x _ { T } f o r$ some matrix $R \in \mathbb { R } ^ { q \times n }$ , and suppose the assumptions of Theorem 3.3 hold true. If $P \succeq R ^ { \top } R$ , then

$$
\operatorname* { s u p } _ { \xi \in \Xi } | \ell ( R x _ { T } , \xi ) - \ell ( R x _ { T } ^ { \prime } , \xi ) | \leq \kappa \zeta _ { N } \sqrt { \gamma _ { 2 } \frac { 1 - \rho ^ { 2 T } } { 1 - \rho ^ { 2 } } } ,
$$

which in case $\rho \in [ 0 , 1 )$ turns into:

$$
\operatorname* { s u p } _ { \xi \in \Xi } | \ell ( R x _ { T } , \xi ) - \ell ( R x _ { T } ^ { \prime } , \xi ) | \leq \kappa \zeta _ { N } \sqrt { \frac { \gamma _ { 2 } } { 1 - \rho ^ { 2 } } } .
$$

## B. Robust performance interpretation

In our analysis the object of central interest coincides with the finite-horizon IO mapping:

$$
w \mapsto \Delta z _ { T } , \qquad \Delta z _ { T } ( \xi ) : = \ell ( x _ { T } , \xi ) - \ell ( x _ { T } ^ { \prime } , \xi ) ,\tag{10}
$$

where $w _ { k } = \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } ) - \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } ^ { j } )$ is the exogenous disturbance introduced by replacing one training sample, and $\Delta z _ { T }$ is the resulting terminal loss difference. Uniform algorithmic stability requires $\begin{array} { r } { \mathrm { s u p } _ { \xi \in \Xi } | \Delta z _ { T } ( \xi ) | \leq \eta } \end{array}$ for every neighbouring dataset pair $ { \boldsymbol { S } } _ { N }$ and $S _ { N } ^ { j }$ . Equivalently, this can be interpreted as a bounded-gain condition on the mapping in (10).

This is precisely the structure of a standard robust-performance problem in the sense of $\mathcal { H } _ { \infty } / \mathrm { I Q C }$ theory [24], [26]. Here, a linear system is driven by an exogenous disturbance $w ,$ with a structured uncertainty $\nu _ { k } = \hat { \Phi } ( y _ { k } , S _ { N } ) - \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } )$ in the feedback connection, and a performance output $\Delta z _ { T }$ whose induced gain must be certified. The dissipativity MI in (7) is the corresponding robust performance certificate.

Figure 1 illustrates the feedback interconnection of the sensitivity system (5), which consists of the following blocks:

i) Linear system: The realization $( A , B , C , D )$ governs how sensitivity $\Delta x _ { k }$ propagates across iteration;

ii) Structured uncertainty: The increment $\nu _ { k } = \hat { \Phi } ( y _ { k } , S _ { N } ) - \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } )$ , constrained by the IQC (6) with multiplier Π, constitutes the feedback nonlinearity;

iii) Exogenous perturbation: The signal $w _ { k } = \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } ) - \hat { \Phi } ( y _ { k } ^ { \prime } , S _ { N } ^ { j } )$ is the data-injection channel, carrying the sensitivity introduced by replacing sample $\xi ^ { ( j ) }$ at every $k .$

Two design parameters directly shape performance in $\mathcal { M }$ in (7). Specifically, the multiplier Π encodes different prior knowledge about the learning mapping, while the rate $\rho$ controls how aggressively past sensitivity is discounted. Note that different choices of the Π may reduce conservatism (e.g., Zames-Falb conditions) at the cost of a higher-dimensional MI compared to slope-restricted/sector constraints, cocoercivity, or monotonicity, all corresponding to a recognized class of IQC [16], [27]. The resulting sensitivity gain is then a function of $( P , \rho , \gamma _ { 1 } , \gamma _ { 2 } )$ , which can be optimized over to obtain the tightest bound for the chosen multiplier class Π—see §III-C. Notice that the same argument used in Theorem 3.3, specific to static pointwise incremental IQCs, also applies to the augmented sensitivity system including the state of the multiplier filter, and dynamic hard IQCs over $T$ can be incorporated directly.

## C. Optimizing the dynamical sensitivity gain

The adopted dissipativity-based perspective therefore offers the opportunity to directly optimize over the sensitivity gain, since it merely depends on quantities related with the storage function and supply rate of the exogenous disturbance $w _ { k }$ . For a feasible tuple $( P , \rho , \gamma _ { 1 } , \gamma _ { 2 } )$ of (7), define the sensitivity gain:

$$
G ( P , \rho , \gamma _ { 2 } ) : = \sqrt { \frac { \gamma _ { 2 } } { \lambda _ { \operatorname* { m i n } } ( P ) ( 1 - \rho ^ { 2 } ) } } .
$$

Since the MI is homogeneous in $( P , \gamma _ { 1 } , \gamma _ { 2 } )$ , one can then impose a normalization such as $P \succcurlyeq I$ For a fixed $\rho \in [ 0 , 1 )$ , let us then define

$$
\begin{array} { r l } { \gamma _ { 2 } ^ { \star } ( \rho ) : = \underset { P , \gamma _ { 1 } , \gamma _ { 2 } } { \operatorname* { m i n } } } & { \gamma _ { 2 } } \\ { \mathrm { s . t . } } & { \mathcal { M } ( P , \rho , \gamma _ { 1 } , \gamma _ { 2 } ) \preccurlyeq 0 , } \end{array}\tag{11}
$$

$$
P \succcurlyeq I , \gamma _ { 1 } , \gamma _ { 2 } \geq 0 .
$$

Note that (11) amounts to an SDP when $\rho$ is fixed. The optimized sensitivity gain then reads as:

$$
G ^ { \star } : = \operatorname* { i n f } _ { \rho \in [ 0 , 1 ) } \sqrt { \frac { \gamma _ { 2 } ^ { \star } ( \rho ) } { 1 - \rho ^ { 2 } } } ,\tag{12}
$$

which on the other hand requires scalar search over [0, 1) to apply Corollaries 3.5 or 3.6. The resulting uniform stability coefficient is thus $\eta \leq \kappa \zeta _ { N } G ^ { \star }$ , with optimal sensitivity gain $G ^ { \star }$ being the tightest bound the chosen multiplier class Π can certify for the realization $( A , B , C , D )$ at hand.

Armed with this, the generalization bound in Lemma 2.3 yields, with probability $1 - \delta$ over the multisample extraction:

$$
r ( x _ { T } ) - \hat { r } ( x _ { T } , \mathcal { S } _ { N } ) \leq \kappa \zeta _ { N } G ^ { \star } + ( N \kappa \zeta _ { N } G ^ { \star } + M ) \sqrt { \frac { 2 \ln ( 1 / \delta ) } { N } } ,
$$

which quantifies the generalization capabilities of the observed learning-enabled dynamics, measured by the loss function ℓ.

## IV. DATA-BASED OPTIMIZATION DYNAMICS

We now apply the proposed dissipativity-based perspective to the analysis of the algorithmic stability of data-based optimization dynamics. Specifically, we consider here algorithms tailored

to SAA for ERM, where the latter denotes the empirical counterpart of the stochastic optimization problem:

$$
\operatorname* { m i n } _ { \boldsymbol { x } \in \mathbb { R } ^ { n } } \mathbb { E } _ { \mathbb { P } } \left[ f ( \boldsymbol { x } , \boldsymbol { \xi } ) \right] ,
$$

with $f : \mathbb { R } ^ { n } \times \Xi \to \mathbb { R }$ , whose sample-based approximation relying on i.i.d. samples gives the resulting empirical risk:

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { n } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } f ( x , \xi ^ { ( i ) } ) .\tag{13}
$$

## A. Gradient descent method

Given a step-size $\alpha > 0$ , applying the GD to (13) yields:

$$
x _ { k + 1 } = x _ { k } - \alpha \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla f ( x _ { k } , \xi ^ { ( i ) } ) \right) ,
$$

which trivially amounts to a special case of the feedback form in (1) with system matrices:

$$
A = I , \qquad B = - \alpha I , \qquad C = I , \qquad D = 0 ,
$$

and data-driven operator $\begin{array} { r } { \hat { \Phi } ( y , S _ { N } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla f ( y , \xi ^ { ( i ) } ) } \end{array}$ . Thus, the general sensitivity system (5) specializes to the sensitivity dynamics of GD. Here, in view of the structure of the output matrix $C ,$ , the data-injection disturbance generated by the replacement of one sample satisfies:

$$
w _ { k } = \hat { \Phi } \big ( x _ { k } ^ { \prime } , S _ { N } \big ) - \hat { \Phi } \big ( x _ { k } ^ { \prime } , S _ { N } ^ { j } \big ) = \frac { 1 } { N } \left( \nabla f ( x _ { k } ^ { \prime } , \xi ^ { ( j ) } ) - \nabla f ( x _ { k } ^ { \prime } , \xi ^ { \prime } ) \right) .
$$

Therefore, according to Corollary A.2, if there exists $B > 0$ such that $\| \nabla f ( x , \xi ) \| \leq B$ for all $x \in \mathbb { R } ^ { n }$ and $\xi \in \Xi$ , then $\lVert w _ { k } \rVert \leq 2 B / N$ for all $k \geq 0$ , i.e., $\zeta _ { N } \leq 2 B / N$ in the notation of Theorem 3.3. The remaining part of the stability estimate is purely dynamical, as it depends on how the GD dynamics dissipates the injected sensitivity. For completeness, we next state and prove a popular result available in the literature [8]:

Corollary 4.1. Given any $N \geq 1$ and $S _ { N } \in \Xi ^ { N }$ , let $x \mapsto f ( x , \xi )$ be m-strongly convex and L-smooth. Choose $\alpha \in ( 0 , 2 / ( L + m ) ]$ , and define $q : = \operatorname* { m a x } \{ | 1 - \alpha m | , | 1 - \alpha L | \} < 1$ Then, the GD is uniformly stable with

$$
\eta \leq \frac { 2 \alpha \kappa B } { N ( 1 - q ) } .\tag{14}
$$

If $\alpha \in ( 0 , 1 / L )$ , then $q = 1 - \alpha m$ , and $\eta \leq 2 \kappa B / m N .$

Proof. Pick any $j \in \{ 1 , \ldots , N \}$ , and let $ { \mathcal { S } } _ { N } ^ { j }$ be the associated neighboring dataset of $ { \boldsymbol { S } } _ { N }$ . In addition, let $x _ { k } , \ x _ { k } ^ { \prime }$ be the corresponding state trajectories of the GD initialized from the same $x _ { 0 } = x _ { 0 } ^ { \prime }$ . By adding and subtracting $\begin{array} { r } { \frac { 1 } { N } \big ( \sum _ { i = 1 , i \neq j } ^ { N } \nabla f ( x _ { k } ^ { \prime } , \xi ^ { ( i ) } ) + \nabla f ( x _ { k } ^ { \prime } , \xi ^ { \prime } ) \big ) } \end{array}$ to $\Delta x _ { k } = x _ { k } - x _ { k } ^ { \prime }$ one immediately obtains $\Delta x _ { k + 1 } = \Delta x _ { k } - \alpha ( \nu _ { k } + w _ { k } )$ , with exactly the same definitions for signals $\nu _ { k }$ and $w _ { k }$ given in §III. Since $\begin{array} { r } { x \mapsto \frac { 1 } { N } \sum _ { i = 1 } ^ { N } f ( x , \xi ^ { ( i ) } ) } \end{array}$ is m-strongly convex and L-smooth, the gradient step $\begin{array} { r } { \mathcal T _ { N } ( x ) : = x - \frac { \alpha } { N } \sum _ { i = 1 } ^ { N } \nabla f ( x , \xi ^ { ( i ) } ) } \end{array}$ is contractive with rate $q .$ . Therefore, one directly obtains:

$$
\| \Delta x _ { k + 1 } \| = \| \mathcal { T } _ { N } ( x _ { k } ) - \mathcal { T } _ { N } ^ { j } ( x _ { k } ^ { \prime } ) \| \leq \| \mathcal { T } _ { N } ( x _ { k } ) - \mathcal { T } _ { N } ( x _ { k } ^ { \prime } ) \| + \| \mathcal { T } _ { N } ( x _ { k } ^ { \prime } ) - \mathcal { T } _ { N } ^ { j } ( x _ { k } ^ { \prime } ) \|
$$

$$
\leq q \| \Delta x _ { k } \| + \alpha \| w _ { k } \| .
$$

Moreover, $\begin{array} { r } { w _ { k } = \frac { 1 } { N } \left( \nabla f ( x _ { k } ^ { \prime } , \xi ^ { ( j ) } ) - \nabla f ( x _ { k } ^ { \prime } , \xi ^ { \prime } ) \right) } \end{array}$ , so $\lVert w _ { k } \rVert \leq 2 B / N$ . Hence $\| \Delta x _ { k + 1 } \| \leq q \| \Delta x _ { k } \| +$ $2 \alpha B / N$ . Since $\Delta x _ { 0 } = 0$ , unrolling the recursion over $T$ immediately leaves us with

$$
\| \Delta x _ { T } \| \leq \frac { 2 \alpha B } { N } \sum _ { k = 0 } ^ { T - 1 } q ^ { k } = \frac { 2 \alpha B } { N } \frac { 1 - q ^ { T } } { 1 - q } \leq \frac { 2 \alpha B } { N ( 1 - q ) } .
$$

Using the κ-Lipschitz continuity of the loss function proves (14), while the remaining claim follows by simply considering that for $\alpha \in ( 0 , 1 / L ) , q = 1 - \alpha m$ ■

The bounds in Corollary 4.1 admit the same interpretation as the general results in §III-A. Specifically, the replacement of one sample injects at most $\alpha \| w _ { k } \| \le 2 \alpha B / N$ units of state sensitivity at each iteration. The contraction factor $q$ determines how much of this injected sensitivity remains after consecutive iterations. The stability coefficient is therefore the product of three factors, which in the strongly convex case generalizes because it forgets one-sample perturbations geometrically, and the accumulated sensitivity saturates. If, instead, $q = 1$ , one recovers the linear-in-time estimate $\eta \le 2 \kappa \alpha B T / N$ . Finally, with $q > 1$ , sample perturbations may be amplified.

Notice that if $\textstyle x \mapsto { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } f ( x , \xi ^ { ( i ) } )$ is m-strongly convex and L-smooth, then the map $\begin{array} { r } { x \mapsto \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla f ( x , \xi ^ { ( i ) } ) } \end{array}$ is slope-restricted in the sector $[ m , L ]$ . In view of the structure of the output matrix $C = I$ , the pair $( \Delta x _ { k } , \nu _ { k } )$ then satisfies the IQC:

$$
\left[ \Delta x _ { k } \right] ^ { \top } \left[ \begin{array} { c c } { m L } & { - ( L + m ) / 2 } \\ { - ( L + m ) / 2 } & { 1 } \end{array} \right] \left[ \begin{array} { c c } { \Delta x _ { k } } \\ { \nu _ { k } } \end{array} \right] \leq 0 .\tag{15}
$$

Therefore, the general MI-based certificate in (7) applies directly. In this simple case, the resulting sensitivity gain is consistent with the explicit contraction calculation above: the dissipativity-based theorem recovers the classical contractivity-based uniform stability bound while assigning a systems-theoretic meaning to each factor. GD is perhaps the simplest instance illustrating the main message conveyed by this paper: algorithms that dissipate data sensitivity admit stabilitybased generalization guarantees, while algorithms that accumulate sensitivity may exhibit poor generalization.

## B. Heavy-ball algorithm

Given strictly positive step-size α and momentum term $\beta ,$ the HB iteration applied to the minimization of (13) reads as:

$$
x _ { k + 1 } = x _ { k } - \alpha \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla f ( x _ { k } , \xi ^ { ( i ) } ) \right) + \beta ( x _ { k } - x _ { k - 1 } ) .
$$

With the lifted state $\tilde { x } _ { k } : = \mathrm { c o l } ( x _ { k } , x _ { k - 1 } )$ , the resulting dynamics can be written in control-affine form as [24]:

$$
\left\{ \begin{array} { l } { \tilde { x } _ { k + 1 } = A _ { \mathrm { H B } } \tilde { x } _ { k } + B _ { \mathrm { H B } } \hat { \Phi } ( y _ { k } , S _ { N } ) , } \\ { ~ y _ { k } = C _ { \mathrm { H B } } \tilde { x } _ { k } , } \end{array} \right.
$$

with nonlinear map $\begin{array} { r } { \hat { \Phi } ( y _ { k } , S _ { N } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla f ( y _ { k } , \xi ^ { ( i ) } ) } \end{array}$ , and

$$
A _ { \mathrm { H B } } = \left[ \begin{array} { c c } { ( 1 + \beta ) I } & { - \beta I } \\ { I } & { 0 } \end{array} \right] , \quad B _ { \mathrm { H B } } = \left[ \begin{array} { c } { - \alpha I } \\ { 0 } \end{array} \right] , \quad C _ { \mathrm { H B } } = \left[ I \quad 0 \right] , \quad D _ { \mathrm { H B } } = 0 .
$$

As in the GD case, the neighboring-data perturbation associated to the sensitivity system amounts to $\begin{array} { r } { w _ { k } = \frac { 1 } { N } ( \nabla f ( x _ { k } ^ { \prime } , \xi ^ { ( j ) } ) - \nabla f ( x _ { k } ^ { \prime } , \xi ^ { \prime } ) ) } \end{array}$ , and if $\| \nabla f ( x , \xi ) \| \le B$ for all $x \in \mathbb { R } ^ { n } , \xi \in \Xi$ , one obtains $\lVert w _ { k } \rVert \leq 2 B / N$ for all $k \geq 0$ . Hence, Corollary 3.6 applied to hypothesis $H _ { N } = [ I \ 0 ] \tilde { x } _ { T }$ leaves us with:

$$
\eta _ { \mathrm { H B } } \leq \frac { 2 \kappa B } { N } G _ { \mathrm { H B } } ^ { \star } ,
$$

where $G _ { \mathrm { H B } } ^ { \star }$ follows by solving (11)–(12) with $( A , B , C , D ) = ( A _ { \mathrm { H B } } , B _ { \mathrm { H B } } , C _ { \mathrm { H B } } , D _ { \mathrm { H B } } )$

Akin to the proof of Corollary 4.1, we note that standard algorithmic stability arguments would implicitly require a pointwise one-step estimate in a prescribed norm as, e.g.,

$$
\| \Delta x _ { k + 1 } \| \le q \| \Delta x _ { k } \| + b \| w _ { k } \| , \mathrm { ~ w i t h ~ } q < 1 .
$$

Such a condition immediately implies a finite-gain bound, but it is stronger. The robust performance certificate of Theorem 3.3 only requires the existence of a storage function for which the closedloop sensitivity interconnection dissipates the data injection supply. This distinction becomes crucial for algorithms such as HB. In fact, after lifting the state the dynamics may be Schur stable and have finite IO gain even though the lifted update is not a contraction in the Euclidean norm.

Example 4.2. Let $\begin{array} { r } { f ( x , \xi ) = \frac { 1 } { 2 } \varrho ( \xi ) x ^ { 2 } } \end{array}$ , with $0 < m \le \varrho ( \xi ) \le L$ for all $\xi \in \Xi ,$ , and define the empirical curvature as $\begin{array} { r } { \hat { \varrho } _ { N } : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \varrho ( \xi ^ { ( i ) } ) } \end{array}$ . Then the empirical gradient mapping coincides with $\hat { \Phi } ( x , S _ { N } ) = \hat { \varrho } _ { N } x$ . For a neighboring dataset $ { \mathcal { S } } _ { N } ^ { j }$ , let $\hat { \varrho } _ { N } ^ { j }$ denote the corresponding empirical curvature. In this quadratic, strongly convex setting, the HB dynamics can be recasted into the linear lifted sensitivity dynamics by means of matrices:

$$
A _ { \mathrm { H B } } ( \hat { \varrho } _ { N } ) = \left[ \begin{array} { c c } { 1 + \beta - \alpha \hat { \varrho } _ { N } } & { - \beta } \\ { 1 } & { 0 } \end{array} \right] , \qquad B _ { \mathrm { H B } } = \left[ \begin{array} { c c } { - \alpha } \\ { 0 } \end{array} \right] ,
$$

and $w _ { k } = ( \hat { \varrho } _ { N } - \hat { \varrho } _ { N } ^ { j } ) x _ { k } ^ { \prime } = ( \varrho ( \xi ^ { ( j ) } ) - \varrho ( \xi ^ { \prime } ) x _ { k } ^ { \prime } / N$ . In particular, $i f \left| x _ { k } ^ { \prime } \right| \le R$ for all $k \geq 0 ,$ , then $| w _ { k } | \leq ( L - m ) R / N$ . However, note that there exist $( \alpha , \beta )$ for which $A _ { \mathrm { H B } } ( \underline { { \hat { \varrho } } } _ { N } )$ is Schur stable but not a strict Euclidean contraction, $e . g . , \ \beta = 1 / 4$ and $\alpha = 3 / 4 \hat { \varrho } _ { N }$ , yielding

$$
A _ { \mathrm { H B } } = \left[ \begin{array} { c c } { 1 / 2 } & { - 1 / 4 } \\ { 1 } & { 0 } \end{array} \right] .
$$

The spectral radius is max $\{ | \lambda _ { 1 } ( A _ { \mathrm { H B } } ) | , | \lambda _ { 2 } ( A _ { \mathrm { H B } } ) | \} = 1 / 2 < 1$ , while $\| A _ { \mathrm { H B } } \| > 1$ . Hence, the HB sensitivity dynamics has finite data sensitivity gain although the contractivity argument fails. In view of the Schur stability of $A _ { \mathrm { H B } } ,$ , standard Lyapunov arguments guarantee the existence of some $P \succ 0$ satisfying $A _ { \mathrm { H B } } ^ { \top } P A _ { \mathrm { H B } } - P = - Q ,$ , for some $Q \succ 0$ . Thus, for any $\rho \in \left( 1 / 2 , 1 \right)$ , one can find $P \succ 0$ such that $A _ { \mathrm { H B } } ^ { \top } P A _ { \mathrm { H B } } - \rho ^ { 2 } P \prec 0 .$ , giving the dissipativity certificate and hence a finite gain. □

For given positive step-size α and momentum term $\beta ,$ , the NAG iteration applied to the minimization of (13) reads as:

$$
\left\{ \begin{array} { c } { \displaystyle z _ { k } = x _ { k } + \beta ( x _ { k } - x _ { k - 1 } ) , } \\ { \displaystyle x _ { k + 1 } = z _ { k } - \frac { \alpha } { N } \sum _ { i = 1 } ^ { N } \nabla f ( \boldsymbol { z } _ { k } , \boldsymbol { \xi } ^ { ( i ) } ) . } \end{array} \right.
$$

With the lifted state $\tilde { x } _ { k }$ as for HB, we readily obtain a linear system representation with nonlinear input and system matrices:

$$
\begin{array} { r l r } & { A _ { \mathrm { N A G } } = \left[ ( 1 + \beta ) I \mathrm {  ~ \Psi ~ } - \beta I \right] , } & { B _ { \mathrm { N A G } } = \left[ - \alpha I \right] , } & { C _ { \mathrm { N A G } } = \left[ ( 1 + \beta ) I \mathrm {  ~ \Psi ~ } - \beta I \right] , \quad D _ { \mathrm { N A G } } = 0 . } \end{array}
$$

Akin to the GD and HB methods, one can bound $\lVert w _ { k } \rVert \leq 2 B / N$ for all $k \geq 0$ , and obtain a uniform stability bound once considered the hypothesis $\begin{array} { r l } { H _ { N } = [ I } & { { } 0 ] \tilde { x } _ { T } } \end{array}$ as:

$$
\eta _ { \mathrm { N A G } } \leq \frac { 2 \kappa B } { N } G _ { \mathrm { N A G } } ^ { \star } ,
$$

where $G _ { \mathrm { N A G } } ^ { \star }$ is obtained by solving (11)–(12) with $( A , B , C , D ) = ( A _ { \mathrm { N A G } } , B _ { \mathrm { N A G } } , C _ { \mathrm { N A G } } , D _ { \mathrm { N A G } } )$ The example provided next confirms that, also for the lifted NAG dynamics where contractivity may not hold, our dissipativity-based angle allows one to optimize over the dynamical sensitivity gain:

Example 4.3. With the same ingredients as in Example 4.2, the (lifted) NAG iteration is characterized by

$$
A _ { \mathrm { N A G } } ( \hat { \varrho } _ { N } ) = \left[ \begin{array} { c c } { ( 1 - \alpha \hat { \varrho } _ { N } ) ( 1 + \beta ) } & { - ( 1 - \alpha \hat { \varrho } _ { N } ) \beta } \\ { 1 } & { 0 } \end{array} \right] ,
$$

$B _ { \mathrm { N A G } } ~ = ~ B _ { \mathrm { H B } }$ , and $\begin{array} { r } { w _ { k } = \frac { \varrho ( \xi ^ { ( j ) } ) - \varrho ( \xi ^ { \prime } ) } { N } z _ { k } ^ { \prime } . } \end{array}$ . As before, $i f \ | z _ { k } ^ { \prime } | \ \leq \ R$ for all $k \geq 0 ,$ , then $| w _ { k } | \le$ $( L - m ) R / N$ . Also in this case, one can find suitable choices of $( \alpha , \beta )$ for which $A _ { \mathrm { N A G } } ( \hat { \varrho } _ { N } )$ is Schur stable but not a strict Euclidean contraction, $e . g . , \ \beta = 1 / 2$ and $\alpha = 1 / 2 \hat { \varrho } _ { N }$ , which yields a spectral radius max $\{ | \lambda _ { 1 } ( A _ { \mathrm { N A G } } ) | , | \lambda _ { 2 } ( A _ { \mathrm { N A G } } ) | \} = 1 / 2 < 1$ , while $\| A _ { \mathrm { N A G } } \| > 1$ , and the same conclusion for HB also applies to NAG. □

![](images/342881ec277a17ef8e937b7bd639638a3434d3c1e6653d7b3fbd0d44478c5dd2.jpg)

![](images/af70d85667a0474e6b7214915682decc21014963f2953f922564d4ea948aabb6.jpg)  
Fig. 2: Top: Certified data sensitivity gain (logarithmic scale). Bottom: Optimized discount factor as functions of the momentum parameter $\beta .$ . Coloured lines are shown only over those values of $\beta$ for which the SDP certificate is feasible.

Remark 4.4. Theorem 3.3, together with Corollaries 3.5 and 3.6, produces algorithm-specific stability bounds, i.e.,

$$
\eta _ { \mathrm { G D } } \leq \frac { 2 \kappa B } { N } G _ { \mathrm { G D } } ^ { \star } , \eta _ { \mathrm { H B } } \leq \frac { 2 \kappa B } { N } G _ { \mathrm { H B } } ^ { \star } , \eta _ { \mathrm { N A G } } \leq \frac { 2 \kappa B } { N } G _ { \mathrm { N A G } } ^ { \star } .
$$

Thus the underlying framework allows one to compare iterative algorithms not only through convergence rate, but also through their generalization capabilities. □

## D. Numerical experiments

We start by evaluating the dynamical component of the stability coefficient. Inspired by Remark 4.4, we fix the class of nonlinearity for $\hat { \Phi }$ by imposing the same sector IQC $[ m , L ]$ for all methods, i.e., GD, HB, and NAG, and compare the certified gains $G ^ { \star }$ obtained by solving (11)-(12). The simulations are run in MATLAB with MOSEK [28] as solver, on a laptop with an Apple M2 chip featuring an 8-core CPU and 16 Gb RAM.

![](images/75df619ca55e6c3095844e5faf50c6577423ffa15dfedb408563c6d597326beb.jpg)  
Fig. 3: Convergence-sensitivity comparison. Each marker corresponds to a value of the momentum parameter $\beta .$

In a scalar setting, i.e., $n = 1$ , by choosing $m = 1$ $L = 1 0 .$ , and $\alpha = 0 . 1$ , Fig. 2 reports the certified sensitivity gain $G ^ { \star }$ and the corresponding optimized discount factor $\rho ^ { \star }$ as functions of the momentum parameter. Notice that, since the loss Lipschitz constant κ and the data-injection size $\zeta _ { N }$ are common across the methods, differences in the resulting stability certificates are entirely due to the algorithmic realization $( A , B , C , D )$ . For small momentum, the gains of HB and NAG are close to the GD baseline. As $\beta$ increases, the certified gain grows rapidly, indicating that the algorithmic dynamics increasingly amplifies one-sample perturbations. In particular, HB loses feasibility earlier than NAG, which suggests a smaller certified robustness margin under the chosen quadratic storage and sector multiplier. The increase of $\rho ^ { \star }$ toward one further indicates that large momentum induces a longer memory of data perturbations.

Following the approach illustrated in Examples 4.2 and 4.3, we set $q _ { \mathrm { n o m } } : = \mathrm { m a x } _ { t \in [ m , L ] } \mathrm { m a x } _ { i } | \lambda _ { i } ( A _ { \mathrm { a l g } } ( t ) ) |$ as the nominal worst-case convergence factor, where $A _ { \mathrm { a l g } }$ is the dynamical matrix obtained in the quadratic setting, $\mathrm { a l g } \in \{ \mathrm { G D } , \mathrm { H B } , \mathrm { N A G } \}$ , and in Fig. 3 we compare the certified data sensitivity gain $G ^ { \star }$ against it. While GD appears as a single point, HB and NAG trace distinct curves as $\beta$ varies. Interestingly, nominal convergence and data sensitivity appear distinct properties, and methods with comparable values of $q _ { \mathrm { n o m } }$ may have substantially different certified gains. In

![](images/30ab21b8d7ad5279d750b648ec791fc2f62b73a4fb9f18e3607839cf55b4c5c0.jpg)

![](images/ae6ca265bef8432ce7f3c7220e00f0a22a7ea7b3e3f3da7a6a497b40ade6f6bf.jpg)  
Fig. 4: Top: IQC-certified finite-horizon gain $G _ { T } ^ { \star } ( \alpha , \beta )$ for HB from the data-injection channel to the terminal current-iterate sensitivity. Bottom: certified sensitivity gain for HB compared with a nominal quadratic convergence proxy.

particular, NAG attains smaller nominal convergence factors than GD, while maintaining moderate sensitivity over a wide range of $\beta ,$ whereas HB exhibits a rapid increase in $G ^ { \star }$ , as its certified robustness margin deteriorates. This is consistent with the robust performance interpretation provided in §III-B, where algorithmic stability amounts to a property of the data sensitivity system rather than as a consequence of nominal convergence alone.

Along the same line, Figs. 4 and 5 report the certified finite-horizon gain $G _ { T } ^ { \star } = G _ { T } ^ { \star } ( \alpha , \beta ) : =$ $\sqrt { \gamma _ { 2 } ^ { \star } ( 1 - \rho ^ { \star ^ { 2 T } } ) / ( 1 - \rho ^ { \star ^ { 2 } } ) } , T = 5 0$ , which is further contrasted with $q _ { \mathrm { n o m } }$ in the scatter plot, for HB and NAG, respectively. Both heatmaps show that $G _ { T } ^ { \star } ( \alpha , \beta )$ significantly increases as the parameters approach the high-momentum boundary of the feasible region, with NAG exhibiting a larger set of combinations of $( \alpha , \beta )$ guaranteeing the feasibility of (11)-(12). Even if the term $\zeta _ { N }$ scales as $O ( 1 / N )$ , notice that the dynamical gain multiplying it can vary significantly across acceleration parameters, with larger values of $\alpha$ that substantially reduce the range of $\beta$ for which the certificate remains small. The scatter plots, instead, further confirm that nominal convergence and data sensitivity are distinct design objectives. Parameter choices yielding more favorable, i.e., smaller, $q _ { \mathrm { n o m } }$ may exhibit substantially larger certified sensitivity gain, whereas the smallest sensitivity gains occur in more conservative regimes where $q _ { \mathrm { n o m } }$ is close to one.

![](images/bbbc5b475816706e27325b2f2d052660cfa6ef86cf1ff2929c9e3477c5c8b52c.jpg)

![](images/d4a05b58fa8475e8966ceddbb0379ec1eff4bebac9690dd2dcf6db53c96a4537.jpg)  
Fig. 5: Top: IQC-certified finite-horizon gain $G _ { T } ^ { \star } ( \alpha , \beta )$ for NAG from the data-injection channel to the terminal current-iterate sensitivity. Bottom: certified sensitivity gain for NAG compared with a nominal quadratic convergence proxy.

## V. APPLICATION TO LEARNED FEEDBACK CONTROL

When the data-driven map $\hat { \Phi } ( \cdot , S _ { N } )$ is constructed to replicate a certain control action, its ability to generalize beyond the finite dataset $ { \boldsymbol { S } } _ { N }$ used for training is essential, since it is deployed online within the feedback interconnection in (1).

Consequently, the relevant inputs to $\hat { \Phi }$ are not only the samples contained in $\boldsymbol { S _ { N } }$ , but the outputs generated by the closed-loop system itself. These outputs may differ from the empirical training distribution, and approximation errors of Φ<sup>ˆ</sup> are recursively propagated through the dynamics. $\hat { \Phi }$ Therefore, a small empirical error on $ { \boldsymbol { S } } _ { N }$ is in general insufficient to guarantee satisfactory closedloop behavior, since poor out-of-sample performance may lead to, e.g., degraded performance, constraint violations, or instability at worst.

This motivates the need for finite-sample generalization guarantees on the learned operator.

![](images/89e747c8eb68e2ea9dd4246f73f868ddb2ffcfd8498ea4006eccca3c90ca0686.jpg)  
Fig. 6: Top: Empirical one-sample sensitivity $\widehat { \zeta } _ { N }$ for different sample sizes. Bottom: Terminal closed-loop sensitivity between trajectories generated by neighbouring learned controllers.

For instance, by writing

$$
\hat { \Phi } ( y , S _ { N } ) = \Phi ( y ) + e _ { N } ( y ) ,
$$

the learning error $\boldsymbol { e } _ { N } : \mathbb { R } ^ { p } \times \Xi ^ { N } \to \mathbb { R } ^ { m }$ enters the state equation as a feedback-dependent disturbance,

$$
x _ { k + 1 } = A x _ { k } + B \Phi ( y _ { k } ) + B e _ { N } ( y _ { k } ) .
$$

Hence, stability and performance properties depend on controlling $e _ { N } ( y )$ over the set of outputs that can be visited by the closed-loop system, rather than only at the training samples $ { \boldsymbol { S } } _ { N }$ Generalization is thus the mechanism that connects finite offline data to reliable online closedloop guarantees.

## A. KRR-based residual feedback control

We consider here a control-oriented example in which the data-dependent operator is learned offline from demonstrations. The plant is a discrete-time linear system $x _ { k + 1 } = A x _ { k } + B u _ { k }$ equipped with a stabilizing nominal linear feedback $u = K x$ . In addition, we assume that an expert operator applies a nonlinear residual correction of the form $\operatorname { r } ( x ) = - a \operatorname { t a n h } ( v ^ { \top } x )$ , so that the measured input is actually $u ^ { ( i ) } = K x ^ { ( i ) } + \mathrm { r } ( x ^ { ( i ) } )$ . The residual map is thus learned via KRR from samples $\begin{array} { r } { S _ { N } = \{ ( \boldsymbol { x } ^ { ( i ) } , \boldsymbol { \mathrm { r } } ( \boldsymbol { x } ^ { ( i ) } ) \} _ { i = 1 } ^ { N } , } \end{array}$ , yielding a data-dependent, state feedback residual controller $\hat { \Phi } ( \cdot , S _ { N } )$ . The learned closed loop dynamics hence reads as:

$$
x _ { k + 1 } = ( A + B K ) x _ { k } + B \hat { \Phi } ( x _ { k } , S _ { N } ) .
$$

Replacing one sample then produces a neighbouring dataset $S _ { N } ^ { j }$ , a different learned controller $\hat { \Phi } ( \cdot , S _ { N } ^ { j } )$ , and hence a different closed-loop trajectory. This is precisely the data sensitivity mechanism captured by our framework discussed in §III, where the perturbation $\hat { \Phi } ( x _ { k } ^ { \prime } , S _ { N } ) \mathrm { ~ - ~ }$ $\hat { \Phi } ( x _ { k } ^ { \prime } , S _ { N } ^ { j } )$ acts as an exogenous disturbance injected into the closed-loop sensitivity dynamics.

The simulations are conducted by considering $n = 2 , m = 1$ and linear controller $K =$ $[ - 0 . 5 5 ~ - 0 . 8 5 ]$ , with matrices

$$
A = { \left[ \begin{array} { l l } { 1 . 1 } & { 0 . 2 } \\ { 0 } & { 0 . 9 5 } \end{array} \right] } ~ , ~ B = { \left[ \begin{array} { l } { 0 } \\ { 1 } \end{array} \right] } ~ .
$$

In addition, we set $a ~ = ~ 0 . 6$ and $v = [ 1 - 1 ] ^ { \top }$ , and to fill the dataset $ { \boldsymbol { S } } _ { N }$ we uniformly sample the set $[ - 2 \ 2 ] ^ { 2 } = : \mathcal { G }$ so that each $\boldsymbol { x } ^ { ( i ) } \in \boldsymbol { \mathcal { G } }$ . For the KRR technique, we use standard Gaussian radial basis functions with bandwidth 0.75, and regularization term 0.001. For each $N \in \{ 2 5 , 5 0 , 1 0 0 , 2 0 0 , 4 0 0 , 8 0 0 , 1 0 0 0 \}$ , we estimate the empirical one-sample sensitivity over 30 possible replacements of the learned residual controller as follows:

$$
\widehat { \zeta } _ { N } : = \operatorname * { m a x } _ { j \in \{ 1 , \ldots , N \} } \operatorname * { m a x } _ { x \in \mathcal { G } } | \widehat { \Phi } ( x , S _ { N } ) - \widehat { \Phi } ( x , S _ { N } ^ { j } ) | .
$$

Regardless of the specific sample replacement, both trajectories are initialized with $x _ { 0 } =$ $[ 1 . 5 ~ - 1 ] ^ { \top }$ . For $T = 4 0$ , we thus estimate the corresponding terminal closed-loop sensitivity as:

$$
\Delta x _ { T } ( N ) : = \operatorname* { m a x } _ { j \in \{ 1 , \ldots , N \} } \ \left\| x _ { T } - x _ { T } ^ { \prime } \right\| .
$$

The two quantities are reported in Fig. 6, corroborating the statistical-dynamical decomposition underlying the proposed bound. From the top plot, the learned-controller sensitivity $\widehat { \zeta } _ { N }$ decreases approximately with the reference rate $1 / N$ , as predicted by the one-sample sensitivity of regularized kernel methods—see Corollary A.8. The terminal closed-loop sensitivity exhibits the same samplesize decay, showing that the reduction in the statistical perturbation is inherited by the closed-loop trajectories. Even in a control setting with a nonlinear learned residual feedback, the observed terminal sensitivity follows the decay of the data injection channel.

## VI. CONCLUSION

We developed a system-theoretic interpretation of generalization in learning-enabled dynamical systems by modeling sample replacement as an exogenous disturbance acting on a data sensitivity system. The operator increment is encoded through an IQC, while dissipativity yields a finite-gain certificate from data injection to terminal loss discrepancy. The resulting uniform stability bound separates the learned operator’s one-sample sensitivity from the dynamical amplification induced by the interconnection. Computable via SDP, the certificate provides a tractable tool for comparing learning dynamics. It recovers classical results for GD, extends to HB and NAG when Euclidean contractivity fails, and shows that convergence and data sensitivity are distinct design objectives. A KRR residual-control example further illustrates its applicability to operators learned offline and deployed in feedback.

Besides considering richer dynamic IQC multipliers, nonquadratic storage functions, or focusing on more general nonlinear system realizations, an interesting direction to investigate consists in the algorithm synthesis for generalization. Since the algorithm realization enters the MI (7) directly, one can in principle co-design parameters according to both convergence and data sensitivity objectives, as well as its generalization certificate, by means of a joint, possibly convex, program.

## REFERENCES

[1] S. Shalev-Shwartz and S. Ben-David, Understanding machine learning: From theory to algorithms. Cambridge University Press, 2014.

[2] V. Vapnik, “Principles of risk minimization for learning theory,” Advances in Neural Information Processing Systems, vol. 4, 1991.

[3] A. J. Kleywegt, A. Shapiro, and T. Homem-de Mello, “The sample average approximation method for stochastic discrete optimization,” SIAM Journal on Optimization, vol. 12, no. 2, pp. 479–502, 2002.

[4] A. Care, R. Carli, A. Dalla Libera, D. Romeres, and G. Pillonetto, “Kernel methods and gaussian processes for system identification and control: A road map on regularized kernel-based learning for control,” IEEE Control Systems Magazine, vol. 43, no. 5, pp. 69–110, 2023.

[5] F. Fabiani and P. J. Goulart, “Reliably-stabilizing piecewise-affine neural network controllers,” IEEE Transactions on Automatic Control, vol. 68, no. 9, pp. 5201–5215, 2023.

[6] O. Bousquet and A. Elisseeff, “Stability and generalization,” The Journal of Machine Learning Research, vol. 2, pp. 499–526, 2002.

[7] C. McDiarmid, “On the method of bounded differences,” Surveys in Combinatorics, vol. 141, no. 1, pp. 148–188, 1989.

[8] M. Hardt, B. Recht, and Y. Singer, “Train faster, generalize better: Stability of stochastic gradient descent,” in International Conference on Machine Learning. PMLR, 2016, pp. 1225–1234.

[9] Y. Chen, C. Jin, and B. Yu, “Stability and convergence trade-off of iterative optimization algorithms,” arXiv preprint arXiv:1804.01619, 2018.

[10] Z. Charles and D. Papailiopoulos, “Stability and generalization of learning algorithms that converge to global optima,” in International Conference on Machine Learning. PMLR, 2018, pp. 745–754.

[11] W. Mou, L. Wang, X. Zhai, and K. Zheng, “Generalization bounds of sgld for non-convex learning: Two theoretical viewpoints,” in Conference on Learning Theory. PMLR, 2018, pp. 605–638.

[12] F. Farnia and A. Ozdaglar, “Train simultaneously, generalize better: Stability of gradient-based minimax learners,” in International Conference on Machine Learning. PMLR, 2021, pp. 3174–3185.

[13] N. Ho, K. Khamaru, R. Dwivedi, M. J. Wainwright, M. I. Jordan, and B. Yu, “Instability, computational efficiency and statistical accuracy,” Journal of Machine Learning Research, vol. 26, no. 65, pp. 1–68, 2025.

[14] F. Fabiani and B. Franci, “Finite-sample guarantees for data-driven forward-backward operator methods,” IEEE Transactions on Automatic Control, 2026.

[15] ——, “Certifying ε-equilibria in stochastic games with limited data availability,” European Journal of Control, p. 101558, 2026.

[16] V. A. Yakubovich, “S-procedure in nonlinear control theory,” Vestnik Leningrad University, pp. 62–77, 1971, (In Russian).

[17] A. Attia and T. Koren, “Algorithmic instabilities of accelerated gradient descent,” Advances in Neural Information Processing Systems, vol. 34, pp. 1204–1214, 2021.

[18] L. Kozachkov, P. Wensing, and J.-J. Slotine, “Generalization as dynamical robustness–The role of Riemannian contraction in supervised learning,” Transactions on Machine Learning Research, 2023.

[19] D. Li and D. Daescu, “A unified Lyapunov-IQC framework for uniform stability of smooth quadratic first-order accelerated optimizers,” arXiv preprint arXiv:2605.08488, 2026.

[20] F. Fabiani, A. Simonetto et al., “Concentration inequalities for semidefinite least squares based on data,” IEEE Signal Processing Letters, vol. 33, pp. 326–330, 2026.

[21] S. Kutin and P. Niyogi, “Almost-everywhere algorithmic stability and generalization error,” in Proceedings of the 18th Conference on Uncertainty in Artificial Intelligence, 2002, pp. 275–282.

[22] E. D. Sontag and Y. Wang, “Notions of input to output stability,” Systems & Control Letters, vol. 38, no. 4-5, pp. 235–248, 1999.

[23] C. M. Kellett, “A compendium of comparison function results,” Mathematics of Control, Signals, and Systems, vol. 26, no. 3, pp. 339–374, 2014.

[24] L. Lessard, B. Recht, and A. Packard, “Analysis and design of optimization algorithms via integral quadratic constraints,” SIAM Journal on Optimization, vol. 26, no. 1, pp. 57–95, 2016.

[25] L. Lessard, “The analysis of optimization algorithms: A dissipativity approach,” IEEE Control Systems Magazine, vol. 42, no. 3, pp. 58–72, 2022.

[26] A. Megretski and A. Rantzer, “System analysis via integral quadratic constraints,” IEEE Transactions on Automatic Control, vol. 42, no. 6, pp. 819–830, 1997.

[27] C. W. Scherer, “Dissipativity and integral quadratic constraints: Tailored computational robustness tests for complex interconnections,” IEEE Control Systems Magazine, vol. 42, no. 3, pp. 115–139, 2022.

TABLE I: Main examples of mappings satisfying $\zeta _ { N } \propto 1 / N$
<table><tr><td>Mapping class</td><td>Assumption</td><td>Bound</td></tr><tr><td>SAA mapping</td><td> $\| \phi ( y , \xi ) \| \le B$ </td><td> $2 B / N$ </td></tr><tr><td>Empirical moments</td><td>bounded moments</td><td> $\begin{array} { r } { 2 L ( \sum _ { k } B _ { k } ^ { 2 } ) ^ { 1 / 2 } / N } \end{array}$ </td></tr><tr><td>Regularized ERM</td><td>µ-strong convexity</td><td> $2 B / \mu N$ </td></tr><tr><td>KRR in RKHS</td><td> $\begin{array} { r l r } { \operatorname { k } ( y , y ) } & { { } \le } & { B ^ { 2 } , } \end{array}$ </td><td> $\begin{array} { r } { \frac { 2 B ^ { 2 } U } { \mu N } \left( 1 + \frac { B } { \sqrt { \mu } } \right) } \end{array}$ </td></tr><tr><td></td><td> $\| u \| \leq U$ </td><td></td></tr><tr><td>Contractive</td><td>contraction  $q < 1$ </td><td> $2 L B / ( 1 - q ) N$ </td></tr></table>

[28] MOSEK ApS, The MOSEK optimization toolbox for MATLAB manual. Version 10.1., 2024. [Online]. Available: http://docs.mosek.com/latest/toolbox/index.htm

## APPENDIX

## A. One-sample sensitivity bounds for data-dependent operators

Given a collection of samples ${ \cal S } _ { N } \in \Xi ^ { N }$ , associated one-sample replacement dataset $ { \mathcal { S } } _ { N } ^ { j }$ , for any $j \in \{ 1 , \ldots , N \}$ , and mapping $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { n }$ , define the empirical averaging operator:

$$
\widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f ( \xi ) \right] : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } f ( \xi ^ { ( i ) } ) ,
$$

and the corresponding empirical average after replacing the j-th sample by $\xi ^ { \prime } \colon$

$$
\widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ f ( \xi ) \right] : = \frac { 1 } { N } \left( \sum _ { i = 1 , i \neq j } ^ { N } f ( \xi ^ { ( i ) } ) + f ( \xi ^ { \prime } ) \right) .
$$

The main classes and realated bounds are summarized in Tab. I. We then start by considering SAA mappings:

Proposition A.1. Let us consider $\hat { \Phi } ( y , S _ { N } ) = g ( \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f _ { 1 } ( \xi ) \right] , \ldots , \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f _ { s } ( \xi ) \right] )$ , where each $f _ { k }$ : $\mathbb { R } ^ { d } \to \mathbb { R } ^ { n }$ , and $g : \mathbb { R } ^ { s n }  \mathbb { R } ^ { m }$ is L-Lipschitz. Assume that there exist constants $B _ { 1 } , \ldots , B _ { s } > 0$ such that $\| f _ { k } ( \xi ) \| \le B _ { k }$ for all $k = 1 , \dots , s$ and $\xi \in \Xi .$ . Then, $\begin{array} { r } { \zeta _ { N } \le 2 L \sqrt { \sum _ { k = 1 } ^ { s } B _ { k } ^ { 2 } } / N . } \end{array}$ □

Proof. For every $k = 1 , \dots , s$ , we have $\begin{array} { r } { \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f _ { k } ( \xi ) \right] - \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ f _ { k } ( \xi ) \right] = \frac { 1 } { N } \left( f _ { k } ( \xi ^ { ( j ) } ) - f _ { k } ( \xi ^ { \prime } ) \right) } \end{array}$ , which yields:

$$
| | \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f _ { k } ( \xi ) \right] - \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ f _ { k } ( \xi ) \right] | | \leq \frac { 1 } { N } \left( | | f _ { k } ( \xi ^ { ( j ) } ) | | + | | f _ { k } ( \xi ^ { \prime } ) | | \right) \leq \frac { 2 B _ { k } } { N } .
$$

Hence, $\begin{array} { r } { \sqrt { \sum _ { k = 1 } ^ { s } \| \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f _ { k } ( \xi ) \right] - \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ f _ { k } ( \xi ) \right] \| ^ { 2 } } \leq \frac { 2 } { N } \sqrt { \sum _ { k = 1 } ^ { s } B _ { k } ^ { 2 } } } \end{array}$ . Using the L-Lipschitz continuity of g, one directly obtains:

$$
\begin{array} { r l r } {  { \| \Phi ( \boldsymbol { y } , S _ { N } ) - \Phi ( \boldsymbol { y } , S _ { N } ^ { j } ) \| = \| g ( \widehat { \mathbb { E } } _ { \mathbb { P } } [ f _ { 1 } ( \boldsymbol { \xi } ) ] , \ldots , \widehat { \mathbb { E } } _ { \mathbb { P } } [ f _ { s } ( \boldsymbol { \xi } ) ] ) - g ( \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } [ f _ { k } ( \boldsymbol { \xi } ) ] , \ldots , \widehat { \mathbb { E } } _ { \mathbb { P } } [ f _ { s } ( \boldsymbol { \xi } ) ] ) \| } } \\ & { } & { \leq L ( \displaystyle \sum _ { k = 1 } ^ { s } \| \widehat { \mathbb { E } } _ { \mathbb { P } } [ f _ { k } ( \boldsymbol { \xi } ) ] - \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } [ f _ { k } ( \boldsymbol { \xi } ) ] \| ^ { 2 } ) ^ { 1 / 2 } \leq \frac { 2 L } { N } ( \displaystyle \sum _ { k = 1 } ^ { s } B _ { k } ^ { 2 } ) ^ { 1 / 2 } , } \end{array}
$$

concluding the proof.

Corollary A.2 (SAA mappings). If $f ( \xi ) = \phi ( y , \xi )$ , with $\phi : \mathbb { R } ^ { p } \times \Xi \to \mathbb { R } ^ { n }$ and $\| \phi ( y , \xi ) \| \leq B$ for all $y \in \mathbb { R } ^ { p } , \xi \in \Xi ,$ , then $\zeta _ { N } \leq 2 B / N$ □

Corollary A.3 (Lipschitz transformations of SAA). If $f ( \xi ) = \phi ( y , \xi )$ , with $\phi : \mathbb { R } ^ { p } \times \Xi \to \mathbb { R } ^ { n }$ and $\| \phi ( x , \xi ) \| \le B$ , for all $y \in \mathbb { R } ^ { p } , \xi \in \Xi ,$ , and L-Lipschitz $\widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f ( \xi ) \right] \mapsto g ( y , \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f ( \xi ) \right] )$ , then $\zeta _ { N } \leq 2 L B / N$ □

Corollary A.4 (Finite collections of empirical moments). If $f _ { k } ( \xi ) ~ = ~ \phi _ { k } ( y , \xi )$ , with $\phi _ { k } \ :$ $\mathbb { R } ^ { p } \times \Xi \to \mathbb { R } ^ { n }$ and $\| \phi _ { k } ( y , \xi ) \| \le B _ { k } ,$ for all $y \in \mathbb { R } ^ { p } , \xi \in \Xi , k = 1 , \dots , s ,$ and L-Lipschitz $( \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f _ { 1 } ( \xi ) \right] , \ldots , \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f _ { s } ( \xi ) \right] ) \mapsto g ( y , \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f _ { 1 } ( \xi ) \right] , \ldots , \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ f _ { s } ( \xi ) \right] )$ , then $\zeta _ { N } \leq 2 L \sqrt { \sum _ { k = 1 } ^ { s } B _ { k } ^ { 2 } } / N$ . □

Corollary A.5 (Kernel mean embeddings). Let H be a Hilbert space. $I f f ( \xi ) = k _ { y } ( \xi ) \in \mathcal { H } ,$ , with $\| k _ { y } ( \xi ) \| _ { \mathcal { H } } \le B$ for all $y \in \mathbb { R } ^ { p } , \xi \in \Xi$ , then $\zeta _ { N } \leq 2 B / N$ □

We now discuss the family of empirical minimizers:

Proposition A.6. Let $\theta \mapsto g ( y , \theta ) + \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ \phi ( y , \theta , \xi ) \right]$ be µ-strongly convex for all $y \in \mathbb { R } ^ { p } , \xi \in \Xi ,$ and replacement $\xi ^ { \prime } \in \Xi ,$ , with $g : \mathbb { R } ^ { p } \times \mathbb { R } ^ { q } \to \mathbb { R } , \ \phi : \mathbb { R } ^ { p } \times \mathbb { R } ^ { q } \times \Xi \to \mathbb { R }$ , and $\hat { \Phi } ( y , S _ { N } ) : =$ arg $\begin{array} { r } { \operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { q } } \left\{ g ( y , \theta ) + \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ \phi ( y , \theta , \xi ) \right] \right\} } \end{array}$ . In addition, assume each sample loss being differentiable with uniformly bounded gradient, $\| \nabla \phi ( y , \theta , \xi ) \| \le B ,$ , for all $y \in \mathbb { R } ^ { p } , \ \theta \in \mathbb { R } ^ { q } , \ \xi \in \Xi$ . Then, $\zeta _ { N } \leq 2 B / \mu N$ □

Proof. Let $\theta _ { N }$ be the unique minimizers for $\theta \mapsto g ( y , \theta ) + \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ \phi ( y , \theta , \xi ) \right]$ ], which satisfies: $\begin{array} { r } { \nabla g ( y , \theta _ { N } ) + \nabla \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ \phi ( y , \theta _ { N } , \xi ) \right] = \nabla g ( y , \theta _ { N } ) + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla \phi ( y , \theta _ { N } , \xi ^ { ( i ) } ) = 0 } \end{array}$ . Similarly, $\theta _ { N } ^ { j }$ is the unique minimizer of $\theta \mapsto g ( y , \theta ) + \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ \phi ( y , \theta , \xi ) \right]$ so that: $\nabla g ( y , \theta _ { N } ^ { j } ) + \nabla \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ \phi ( y , \theta _ { N } ^ { j } , \xi ) \right] =$ $\begin{array} { r } { \nabla g ( y , \theta _ { N } ^ { j } ) + \frac { 1 } { N } \left( \sum _ { i \neq j } ^ { N } \nabla \phi ( y , \theta _ { N } ^ { j } , \xi ^ { ( i ) } ) + \nabla \phi ( y , \theta _ { N } ^ { j } , \xi ^ { \prime } ) \right) = 0 } \end{array}$ . In view of the µ-strong monotonicity of $\theta \mapsto \nabla g ( y , \dot { \theta } ) + \nabla \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ \phi ( y , \theta , \xi ) \right]$ , consequence of the µ-strong convexity of the underlying function, and the application of the Cauchy-Schwarz inequality we readily obtain:

$$
\begin{array} { r l } & { \mu \| \theta _ { N } - \theta _ { N } ^ { j } \| = \mu \| \hat { \Phi } ( y , S _ { N } ) - \hat { \Phi } ( y , S _ { N } ^ { j } ) \| } \\ & { \qquad \leq \Big \| \nabla g ( y , \theta _ { N } ) + \nabla \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ \phi ( y , \theta _ { N } , \xi ) \right] - \nabla g ( y , \theta _ { N } ^ { j } ) - \nabla \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ \phi ( y , \theta _ { N } ^ { j } , \xi ) \right] \Big \| } \\ & { \qquad = \| \nabla g ( y , \theta _ { N } ) + \nabla \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ \phi ( y , \theta _ { N } , \xi ) \right] \| . } \end{array}
$$

By exploiting the fact that $\nabla g ( y , \theta _ { N } ) + \nabla \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ \phi ( y , \theta _ { N } , \xi ) \right] = 0 :$

$$
\begin{array} { r l } & { \| \nabla g ( y , \theta _ { N } ) + \nabla \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ \phi ( y , \theta _ { N } , \xi ) \right] \| = \Big \| \nabla g ( y , \theta _ { N } ) + \nabla \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ \phi ( y , \theta _ { N } , \xi ) \right] } \\ & { \qquad - \nabla g ( y , \theta _ { N } ) - \nabla \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ \phi ( y , \theta _ { N } , \xi ) \right] \Big \| } \\ & { \qquad \leq \displaystyle \frac { 1 } { N } \left( \| \nabla \phi ( x , \theta _ { N } , \xi ^ { \prime } ) \| + \| \nabla \phi ( x , \theta _ { N } , \xi ^ { ( j ) } ) \| \right) \leq \displaystyle \frac { 2 B } { N } , } \end{array}
$$

since the two gradients differ only from the replaced sample, directly yielding the desired bound on $\zeta _ { N }$ ■

Corollary A.7 (Regularized empirical minimizers). $I f g ( y , \theta ) = \mu \left\| \theta \right\| ^ { 2 } / 2$ and $\theta \mapsto \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ \phi ( y , \theta , \xi ) \right]$ is convex for all $y \in \mathbb { R } ^ { p } , \xi \in \Xi ,$ and $\xi ^ { \prime } \in \Xi ,$ , then $\zeta _ { N } \leq 2 B / \mu N$ □

Corollary A.8 (KRR in RKHS). Let H be the reproducing kernel Hilbert space (RKHS) induced by a kernel k : $\mathbb { R } ^ { p } \times \mathbb { R } ^ { p } \to \mathbb { R } _ { \geq 0 }$ , and let ${ \mathcal { H } } ^ { m }$ denote the corresponding product space for vector-valued maps $f : \mathbb { R } ^ { p }  \mathbb { R } ^ { m }$ . In addition, let $\mathcal { V } \subseteq \mathbb { R } ^ { p }$ be such that $\begin{array} { r } { \operatorname* { s u p } _ { y \in \mathcal { y } } \operatorname { k } ( y , y ) \le B ^ { 2 } } \end{array}$ With sample $\xi = \operatorname { c o l } ( y , u ) \in \mathcal { V } \times \mathbb { R } ^ { m }$ satisfying $\| u \| \leq U$ , for $\mu > 0$ we define the KRR estimator as

$$
\hat { f } _ { N } : = \underset { f \in \mathcal { H } ^ { m } } { \mathrm { a r g m i n } } \left. \frac { \mu } { 2 } \| f \| _ { \mathcal { H } ^ { m } } ^ { 2 } + \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \| f ( y ^ { ( i ) } ) - u ^ { ( i ) } \| ^ { 2 } \right. .
$$

With $\hat { \Phi } ( y , S _ { N } ) : = \hat { f } _ { N } ( y )$ , for every neighboring dataset $ { \mathcal { S } } _ { N } ^ { j }$ ,

$$
\operatorname* { s u p } _ { y \in \mathcal { V } } \| \hat { \Phi } ( y , S _ { N } ) - \hat { \Phi } ( y , S _ { N } ^ { j } ) \| \leq \frac { 2 B ^ { 2 } U } { \mu N } \left( 1 + \frac { B \sqrt { \mu } } { \mu } \right) .
$$

Proof. Let $\hat { f } _ { N }$ and $\hat { f } _ { N } ^ { j }$ denote the KRR minimizers associated with $ { \boldsymbol { S } } _ { N }$ and $ { \mathcal { S } } _ { N } ^ { j }$ , respectively. Then, define the cost function as $J _ { N } ( f ) : = \mu \| f \| _ { \mathcal H ^ { m } } ^ { 2 } / 2 + \sum _ { i = 1 } ^ { N } \| f ( y ^ { ( i ) } ) - u ^ { ( i ) } \| ^ { 2 } / 2 N$ , along with the analogous counterpart $J _ { N } ^ { j }$ obtained after replacing $\xi ^ { ( j ) } = \mathrm { c o l } ( y ^ { ( j ) } , u ^ { ( j ) } )$ with $\xi ^ { \prime } = \mathrm { c o l } ( y ^ { \prime } , u ^ { \prime } )$ Since $J _ { N }$ is µ-strongly convex on ${ \mathcal { H } } ^ { m }$ , its minimizer is unique. Moreover, $J _ { N } ( \hat { f } _ { N } ) \le J _ { N } ( 0 ) =$ $\begin{array} { r } { \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \| u ^ { ( i ) } \| ^ { 2 } \leq \frac { U ^ { 2 } } { 2 } } \end{array}$ , which directly implies $\| \hat { f } _ { N } \| _ { \mathcal { H } ^ { m } } \leq U / \sqrt { \mu }$ . For a generic sample $\xi =$ co $. ( y , u )$ , let $\phi ( f , \xi ) : = \| f ( y ) - u \| ^ { 2 } / 2$ . By the reproducing property, the gradient of $\phi$ with respect to $f \in \mathcal { H } ^ { m }$ satisfies

$$
\| \nabla \phi ( \hat { f } _ { N } , \xi ) \| _ { \mathcal { H } ^ { m } } \leq \sqrt { \mathrm { k } ( y , y ) } \| \hat { f } _ { N } ( y ) - u \| .
$$

Since $y \in \mathcal { V } , \operatorname { k } ( y , y ) \leq B ^ { 2 }$ . Again by the reproducing property,

$$
\| \hat { f } _ { N } ( y ) \| \leq B \| \hat { f } _ { N } \| _ { \mathcal { H } ^ { m } } \leq \frac { B U } { \sqrt { \mu } } ,
$$

which yields $\begin{array} { r } { \| \nabla \phi ( \hat { f } _ { N } , \xi ) \| _ { \mathcal { H } ^ { m } } \leq B U \left( 1 + \frac { B } { \sqrt { \mu } } \right) } \end{array}$

Since $\hat { f } _ { N } ^ { j }$ minimizes $J _ { N } ^ { j }$ and $J _ { N } ^ { j }$ is µ-strongly convex, the same argument used in Proposition A.6 gives

$$
\begin{array} { r } { \mu \| \hat { f } _ { N } - \hat { f } _ { N } ^ { j } \| _ { \mathcal { H } ^ { m } } \leq \| \nabla J _ { N } ^ { j } ( \hat { f } _ { N } ) - \nabla J _ { N } ( \hat { f } _ { N } ) \| _ { \mathcal { H } ^ { m } } . } \end{array}
$$

Because $J _ { N }$ and $J _ { N } ^ { j }$ differ only in the replaced sample,

$$
\| \nabla J _ { N } ^ { j } ( \hat { f } _ { N } ) - \nabla J _ { N } ( \hat { f } _ { N } ) \| _ { \mathcal { H } ^ { m } } \leq \frac { 1 } { N } \left( \| \nabla \phi ( \hat { f } _ { N } , \xi ^ { \prime } ) \| _ { \mathcal { H } ^ { m } } + \| \nabla \phi ( \hat { f } _ { N } , \xi ^ { ( j ) } ) \| _ { \mathcal { H } ^ { m } } \right) \leq \frac { 2 B U } { N } \left( 1 + \frac { B } { \sqrt { \mu } } \right) ,
$$

and thus $\begin{array} { r } { \| \hat { f } _ { N } - \hat { f } _ { N } ^ { j } \| _ { \mathcal { H } ^ { m } } \leq \frac { 2 B U } { \mu N } \left( 1 + \frac { B } { \sqrt { \mu } } \right) } \end{array}$ . Finally, using the reproducing property once more, for every $y \in \mathcal { V }$

$$
\| \hat { \Phi } ( y , S _ { N } ) - \hat { \Phi } ( y , S _ { N } ^ { j } ) \| = \| \hat { f } _ { N } ( y ) - \hat { f } _ { N } ^ { j } ( y ) \| \leq B \| \hat { f } _ { N } - \hat { f } _ { N } ^ { j } \| _ { \mathcal { H } ^ { m } } ,
$$

which allows one to establish the desired bound.

We conclude by discussing the case in which $\hat { \Phi }$ consists of a contractive mapping driven by empirical averages.

Proposition A.9. Suppose that $\hat { \Phi } ( y , S _ { N } ) ~ = ~ z _ { N }$ where $z _ { N }$ is the unique solution $t o \ z \ =$ $\mathcal { T } \left( y , z , \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ \phi ( y , \xi ) \right] \right)$ . Assume that $\tau$ is contractive in $z , i . e .$ , there exists $q \in [ 0 , 1 )$ such that

$$
\begin{array} { r } { \| \mathcal { T } ( y , z , u ) - \mathcal { T } ( y , z ^ { \prime } , u ) \| \le q \| z - z ^ { \prime } \| , } \end{array}
$$

for all $y \in \mathbb { R } ^ { p } , \ u \in \mathbb { R } ^ { m }$ , and $u \mapsto \mathcal T ( y , z , u )$ being L-Lipschitz. $H ,$ in addition, $\| \phi ( y , \xi ) \| \leq B$ for all $y \in \mathbb { R } ^ { p } , \xi \in \Xi ,$ then $\zeta _ { N } \leq 2 L B / ( 1 - q ) N$ □

Proof. In view of Corollary A.2, $\| \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ \phi ( y , \xi ) \right] - \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ \phi ( y , \xi ) \right] \| \leq 2 B / N$ . Let $z _ { N }$ and $z _ { N } ^ { j }$ be the fixed points corresponding to $\mathcal { T } ( y , z , \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ \phi ( y , \xi ) \right] )$ and $\mathcal { T } ( y , z , \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ \phi ( y , \xi ) \right] )$ , respectively. Thus, we immediately obtain inequalities:

$$
\begin{array} { r l } & { \| z _ { N } - z _ { N } ^ { j } \| = \| { \mathcal { T } } ( y , z _ { N } , { \widehat { \mathbb { E } } } _ { \mathbb { F } } [ \phi ( y , \xi ) ] ) - { \mathcal { T } } ( y , z _ { N } ^ { j } , { \widehat { \mathbb { E } } } _ { \mathbb { P } } ^ { j } [ \phi ( y , \xi ) ] ) \| } \\ & { \qquad \leq \| { \mathcal { T } } ( y , z _ { N } , { \widehat { \mathbb { E } } } _ { \mathbb { P } } [ \phi ( y , \xi ) ] ) - { \mathcal { T } } ( y , z _ { N } ^ { j } , { \widehat { \mathbb { E } } } _ { \mathbb { P } } [ \phi ( y , \xi ) ] ) \| } \\ & { \qquad ~ + \| { \mathcal { T } } ( y , z _ { N } ^ { j } , { \widehat { \mathbb { E } } } _ { \mathbb { P } } [ \phi ( y , \xi ) ] ) - { \mathcal { T } } ( y , z _ { N } ^ { j } , { \widehat { \mathbb { E } } } _ { \mathbb { P } } ^ { j } [ \phi ( y , \xi ) ] ) | } \\ & { \qquad \leq q \| z _ { N } - z _ { N } ^ { j } \| + L \| { \widehat { \mathbb { E } } } _ { \mathbb { P } } [ \phi ( y , \xi ) ] - { \widehat { \mathbb { E } } } _ { \mathbb { P } } ^ { j } [ \phi ( y , \xi ) ] \| . } \end{array}
$$

Rearranging the terms, $( 1 - q ) \| z _ { N } - z _ { N } ^ { j } \| \leq L \| \widehat { \mathbb { E } } _ { \mathbb { P } } \left[ \phi ( y , \xi ) \right] - \widehat { \mathbb { E } } _ { \mathbb { P } } ^ { j } \left[ \phi ( y , \xi ) \right] \| \leq 2 L B / N$ , which together with $\| z _ { N } - z _ { N } ^ { j } \| = \| \hat { \Phi } ( y , S _ { N } ) - \hat { \Phi } ( y , S _ { N } ^ { j } ) \|$ concludes the proof. ■