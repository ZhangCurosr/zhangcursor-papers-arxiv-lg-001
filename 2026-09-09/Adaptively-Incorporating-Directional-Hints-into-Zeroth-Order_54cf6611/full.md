# Adaptively Incorporating Directional Hints into Zeroth-Order Optimization

Alexander Ryabchenko University of Toronto and Vector Institute alex.rbch.research@gmail.com

Jian Qian University of Hong Kong jianqian@hku.hk

Wenlong Mou University of Toronto and Vector Institute wenlong.mou@utoronto.ca

## Abstract

We study zeroth-order optimization of non-convex functions with the aid of directional hints, which are cheap but potentially inaccurate approximations of the true gradient direction, given by linear subspaces at each iteration. To leverage these hints adaptively while maintaining robustness to their quality, we introduce Control-Variate Zeroth-Order Descent (CV-ZOD), a new framework that refines the classical zeroth-order gradient estimator with a control variate that can be set based on the directional hints. We first show that the oracle algorithm that optimally sets the reference vector and step size at each iteration achieves a convergence rate that interpolates between the first-order O(1/T) rate and the zeroth-order O(d/T) rate, depending on the quality of the hints along the trajectory. We then develop a practical variant of CV-ZOD that achieves the same oracle guarantee up to logarithmic factors, without any prior knowledge of the hint quality. We validate the method empirically on simulation-based scientific optimization tasks, demonstrating sustained progress on nonconvex landscapes where zeroth-order descent is slower and existing guided methods stall as guidance deteriorates.

## 1 Introduction

Many real-world optimization problems involve non-convex objectives whose gradients are unavailable or prohibitively expensive to compute. In scientific applications, the objective may be defined by a simulator or experimental pipeline that cannot be differentiated end-to-end [LMW19]. In machine learning, the model itself may be accessible only through an API, as in black-box adversarial attacks on deployed classifiers [CZS<sup>+</sup>17, IEAL18], prompt optimization for API-gated language models [ZCD<sup>+</sup>24], or memory-efficient fine-tuning of large models, where backpropagation is replaced by forward-pass-only updates [MGN<sup>+</sup>23]. Zeroth-order methods, which rely only on function evaluations, are a natural tool in these settings. However, their convergence rates typically scale with the dimension d, making them impractical in high-dimensional problems.

At the same time, auxiliary information about the objective is often available. A pretrained surrogate, a smaller related model, or a low-rank approximation may suggest promising descent directions. For example, the gradient of a distilled model can provide a cheap proxy for the gradient of a large target model in prompt optimization, while a differentiable foundation model can guide black-box optimization in scientific experiments. Using such information effectively, however, is delicate. Blindly following the suggested directions can lead to rapid early progress when they are well aligned with the true gradient, but the resulting bias may ultimately limit convergence. Conversely, ignoring this information altogether reverts to the standard dimension-dependent cost of zeroth-order optimization. Ideally, we would like to use directional hints when they are informative and adaptively fall back to unbiased zeroth-order descent when they are not, without any prior knowledge of their quality along the trajectory.

The idea of incorporating prior directional information into zeroth-order optimization has been explored in several works [MMT<sup>+</sup>19, CWZ21]. These methods bias the search distribution toward the suggested directions, which can accelerate convergence when the guidance is well aligned with the true gradient. However, this modification also biases the resulting gradient estimators: they no longer target the gradient of the original objective, but rather a reweighted version of it. Consequently, the optimization dynamics are distorted by the guidance, which can limit progress. Moreover, existing convergence analyses for these approaches are restricted to convex objectives. This motivates the central question: Can we incorporate directional hints into non-convex zeroth-order optimization in a way that adapts to their quality along the trajectory without altering the underlying optimization dynamics?

In this paper, we answer this question affirmatively. We study a general interaction model in which the hints are revealed as low-dimensional linear subspaces at each iteration, allowing different sources of directional information to be treated within a common framework. Our contributions are as follows:

1. Control-Variate Zeroth-Order Descent (CV-ZOD) framework (Section 3). We introduce a new zerothorder descent scheme, CV-ZOD, built on a control-variate gradient estimator [AL24, Owe13, GBB04] that uses a reference vector to reduce variance while remaining unbiased for the smoothed gradient. The variance of the estimator is governed by the accuracy of the reference vector relative to the true gradient. By instantiating the reference vector through projection of the true gradient onto the hint subspace and choosing the optimal stepsize, the resulting oracle algorithm interpolates between the first-order $O ( 1 / T )$ rate when the hints are well-aligned and the zeroth-order $O ( d / T )$ rate when they are uninformative.

2. A fully zeroth-order adaptive instantiation (Section 4). In practice, the oracle information of projected gradient and optimal stepsize is unavailable. We develop a practical variant of CV-ZOD that estimates these quantities using only function evaluations, and show that it achieves the same adaptive convergence guarantee, up to logarithmic factors, with $O ( k + \log T )$ queries per iteration. As a result, the adaptive version of CV-ZOD can achieve the interpolating convergence rates without any prior knowledge of the hint quality along the trajectory.

3. Simulation-based scientific optimization tasks (Section 5). We validate the framework on scientific optimization tasks arising from fluid dynamics and computational chemistry, using cheap surrogate models to generate directional hints. We show that CV-ZOD achieves significant speedup over existing zeroth-order methods, and much lower final error than existing guided methods that rely on biased gradient estimation.

Related work. Prior theoretical work has explored incorporating surrogate gradient information into zeroth-order optimization, but the existing analyses have so far been limited to convex objectives and relied on biased gradient estimators that reshape the optimization dynamics around the surrogate; the resulting algorithms do not follow the gradient flow of $f ,$ but one shaped by the surrogate. Specifically, $[ \mathbf { M } \mathbf { M } \mathbf { T } ^ { + } 1 9 ]$ introduce Guided Evolutionary Strategies (GES), which bias the search distribution along the guiding subspace and analyze the bias-variance tradeoff of the resulting estimator, but provide no convergence guarantees. [CWZ21] analyze the Prior-Guided Random Gradient-Free (PRGF) method, which uses a similar biased gradient estimator, and obtain convergence rates that improve with the cosine similarity between the prior and the true gradient; however, their analysis relies essentially on convexity and a bounded level set assumption, precluding a direct extension to non-convex objectives. See Appendix A for a more detailed discussion.

A complementary approach in scientific optimization is Bayesian optimization, which selects experiments through a probabilistic model of objective values. Applications include reaction optimization [SSL<sup>+</sup>21] and the discovery of improved photocatalyst formulations $[ \mathrm { B M G } ^ { + } 2 0 ]$ . CV-ZOD instead uses directional hints to reduce the variance of a zeroth-order gradient estimator while retaining its smoothed-gradient expectation. The black-box machine-learning applications discussed above provide further settings in which cheap surrogate gradients may supply such hints.

## 2 Problem Setting

This paper considers the unconstrained minimization problem

$$
f ^ { * } : = \operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } f ( x ) ,
$$

where $d \in \mathbb { N }$ is the dimension and $f \colon  { \mathbb { R } ^ { d } } \to$ R is the differentiable objective. Throughout, $f$ is taken to be B-Lipschitz, i.e., $\| \nabla f ( x ) \| \leq B$ for all $x \in \mathbb { R } ^ { d }$ , and L-smooth, i.e.,

$$
\begin{array} { r l r } { \big | f ( y ) - f ( x ) - \langle \nabla f ( x ) , y - x \rangle \big | \leq \frac { L } { 2 } \| y - x \| ^ { 2 } } & { } & { \mathrm { ~ f o r ~ a l l ~ } x , y \in { \mathbb R } ^ { d } . } \end{array}
$$

We study iterative algorithms that access $f$ only through a zeroth-order oracle, which returns the scalar value $f ( x )$ for any query $x \in \mathbb { R } ^ { d }$ , and that, at each iteration, are provided with a directional hint in the form of a low-dimensional linear subspace of $\mathbb { R } ^ { d }$ . Figure 1 depicts the resulting interaction.

Formally, the algorithm is initialized at an arbitrary point $x _ { 1 } \in \mathbb { R } ^ { d }$ and runs for $T \in \mathbb { N }$ iterations. At iteration $t \in [ T ]$ , the environment reveals a hint subspace $S _ { t } \in \operatorname { G r } _ { k } ( \mathbb { R } ^ { d } )$ , where $\operatorname { G r } _ { k } ( \mathbb { R } ^ { d } )$ denotes the set of k-dimensional linear subspaces of $\mathbb { R } ^ { d }$ . The algorithm then issues a batch of zeroth-order queries and selects the next iterate $x _ { t + 1 }$ After T iterations, it returns an output point $\bar { x } \in \mathbb { R } ^ { d }$ based on the information collected during the interaction. For the descent schemes considered below, every step size $\alpha _ { t }$ is strictly positive.

Zeroth-Order Optimization with Directional Hints   
Parameters: horizon T, hint subspace dimension $k \in [ d ] .$   
Input: initial point $x _ { 1 } \in \mathbb { R } ^ { d } .$   
For each iteration $t = 1 , 2 , \dots , T \colon$   
1. The environment reveals a hint subspace $S _ { t } \in \operatorname { G r } _ { k } ( \mathbb { R } ^ { d } )$   
2. The algorithm makes queries to the zeroth-order oracle and selects the next iterate $x _ { t + 1 }$   
Output: point $\bar { x } \in \mathbb { R } ^ { d }$ selected using the information collected over the $T$ iterations.  
Figure 1: Zeroth-order optimization with directional hints.

Performance is measured by the stationarity of the output: specifically, we seek to control $\mathbb { E } [ \| \nabla f ( \bar { x } ) \| ^ { 2 } ]$ . Here, the expectation is with respect to all randomness that can affect the output point x¯: the algorithm’s internal randomization and the hint-generation process that generates hints $( S _ { t } ) _ { t = 1 } ^ { T }$ , as specified in the next paragraph.

Hint generation model. We impose no distributional or structural assumptions on the mechanism that produces hint subspaces. The sequence $( \boldsymbol { \bar { S _ { t } } } ) _ { t = 1 } ^ { T }$ may be deterministic or random, and may be chosen adaptively, subject only to a non-anticipation condition. Formally, let $( \mathcal { F } _ { t } ) _ { t = 1 } ^ { T }$ be a filtration such that $\mathcal { F } _ { t }$ captures all randomness available before iteration $t ,$ including the iterate $x _ { t }$ . We require only that $S _ { t }$ be $\mathcal { F } _ { t }$ -measurable. Equivalently, the hint revealed at iteration t may depend on the current iterate and all past information, but not on the algorithm’s internal randomness within the same iteration. For example, given access to differentiable surrogates $f _ { 1 } ^ { \prime } , \ldots , f _ { k } ^ { \prime }$ our framework accommodates the natural hint choice ${ \cal { S } } _ { t } = \operatorname { s p a n } \{ \nabla f _ { 1 } ^ { \prime } ( x _ { t } ) , \ldots , \nabla f _ { k } ^ { \prime } ( x _ { t } ) \}$

We quantify the quality of the hint $S _ { t }$ via the principal angle between $S _ { t }$ and the current gradient, defined as

$$
\begin{array} { r } { \theta _ { t } : = \angle ( \nabla f ( x _ { t } ) , S _ { t } ) = \operatorname { a r c c o s } \left( \frac { \| P _ { S _ { t } } \nabla f ( x _ { t } ) \| } { \| \nabla f ( x _ { t } ) \| } \right) , } \end{array}
$$

with the convention $\theta _ { t } : = 0$ when $\nabla f ( x _ { t } ) = 0$ . The angle $\theta _ { t }$ measures how much of the current gradient is captured by $S _ { t }$ . The sequence $( \theta _ { t } ) _ { t = 1 } ^ { T }$ is itself a stochastic process adapted to $( \mathcal { F } _ { t } ) _ { t = 1 } ^ { T } .$ , and our convergence guarantees are stated in terms of its realized distribution along the trajectory rather than any worst-case bound.

Additional notation. We write $[ n ] : = \{ 1 , \dots , n \}$ for $n \in \mathbb { N } .$ . For a subspace $S \subseteq \mathbb { R } ^ { d } ,$ , we denote by $P _ { S }$ the orthogonal projector onto $s$ and by $\boldsymbol { S ^ { \perp } }$ its orthogonal complement, with the shorthand $P _ { S } ^ { \bot } : = P _ { S ^ { \bot } }$ . For vectors $v _ { 1 } , \ldots , v _ { m } \in \mathbb { R } ^ { d }$ , we write $\operatorname { s p a n } \{ v _ { 1 } , \dots , v _ { m } \}$ for their linear span. For $k \in \{ 0 , \ldots , d \}$ , we denote by $\operatorname { G r } _ { k } ( \mathbb { R } ^ { d } )$ the Grassmannian of k-dimensional linear subspaces of $\mathbb { R } ^ { d }$ , with $\operatorname { G r } _ { 0 } ( \mathbb { R } ^ { d } )$ consisting of the trivial subspace {0} alone.

## 3 Control-Variate Framework for Zeroth-Order Descent

This section introduces the Control-Variate Zeroth-Order Descent (CV-ZOD) framework. Section 3.1 constructs a gradient estimator (2) that, given an arbitrary reference vector, refines it into an unbiased estimate of the smoothed gradient with variance controlled by how close the reference vector is to the true gradient. Section 3.2 embeds this estimator into an iterative descent scheme in which a fresh reference vector and step size are chosen at each iteration, and establishes a convergence guarantee (Theorem 3.4) governed by the interplay between the two along the trajectory. The framework is agnostic to how reference vectors and step sizes are chosen; Section 3.3 specializes it to directional hints and identifies the oracle choices that serve as a benchmark for Section 4, where we match this benchmark using only function evaluations.

## 3.1 Gradient Estimation with Control Variates

We recall standard zeroth-order tools [GL13, NS17, BG19] before introducing our control-variate estimator.

Preliminaries: Gaussian smoothing and the classical gradient estimator. For a smoothing parameter $\tau > 0 .$ we define the τ-smoothed objective by

$$
f ^ { \tau } ( \boldsymbol { x } ) : = \mathbb { E } _ { u \sim \mathcal { N } ( 0 , I _ { d } ) } \left[ f ( \boldsymbol { x } + \tau \boldsymbol { u } ) \right] \qquad \mathrm { f o r ~ a l l ~ } \boldsymbol { x } \in \mathbb { R } ^ { d } .
$$

The following lemma records the basic approximation properties of this Gaussian smoothing.

Lemma 3.1 (Theorems 1-3 in [NS17]) The function $f ^ { \tau }$ is differentiable, L-smooth, and, for all $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , satisfies

$$
| f ^ { \tau } ( x ) - f ( x ) | \leq \frac { \tau ^ { 2 } } { 2 } L d \qquad a n d \qquad \| \nabla f ^ { \tau } ( x ) - \nabla f ( x ) \| \leq \frac { \tau } { 2 } L ( d + 3 ) ^ { 3 / 2 } .
$$

Moreover, for all $x \in \mathbb { R } ^ { d } , \nabla f ^ { \tau } ( x ) = \mathbb { E } _ { u \sim { \mathcal { N } } ( 0 , I _ { d } ) } \left[ \frac { f ( x + \tau u ) } { \tau } u \right]$

By taking $\tau$ sufficiently small, the smoothed objective $f ^ { \tau }$ and its gradient $\nabla f ^ { \tau }$ can be made arbitrarily close to $f$ and $\nabla f$ , respectively. The final identity expresses $\nabla f ^ { \tau } ( x )$ through function evaluations alone and motivates the classical (two-point) zeroth-order gradient estimator:

$$
G ( x ; u ) : = \frac { f ( x + \tau u ) - f ( x ) } { \tau } u , \qquad \mathrm { w h e r e } u \sim \mathcal { N } ( 0 , I _ { d } ) ,\tag{1}
$$

whose properties are summarized below.

Lemma 3.2 (Theorem 3.1 in [GL13]) For all $x \in \mathbb { R } ^ { d } , \mathbb { E } _ { u \sim \mathcal { N } ( 0 , I _ { d } ) } [ G ( x ; u ) ] = \nabla f ^ { \tau } ( x ) a n d$

$$
\begin{array} { r } { \mathbb { E } _ { u \sim \mathcal { N } ( 0 , I _ { d } ) } \left[ \| G ( x ; u ) - \nabla f ( x ) \| ^ { 2 } \right] \leq 4 ( d + 5 ) \| \nabla f ( x ) \| ^ { 2 } + \tau ^ { 2 } L ^ { 2 } ( d + 6 ) ^ { 3 } . } \end{array}
$$

The $d -$ factor in the first term of the mean-squared error bound reflects the cost of isotropic exploration: (1) probes all directions equally, without any mechanism to prioritize the gradient direction.

The control-variate gradient estimator. For every $x \in \mathbb { R } ^ { d }$ , define the control-variate gradient estimator parameterized by a reference vector $m \in \mathbb { R } ^ { d }$ as

$$
G ( x , m ; u ) : = m + \frac { f ( x + \tau u ) - f ( x ) - \langle m , \tau u \rangle } { \tau } u , \qquad \mathrm { w h e r e ~ } u \sim \mathcal { N } ( 0 , I _ { d } ) .\tag{2}
$$

The construction applies the classical zeroth-order estimator to $f ( \cdot ) - \langle m , \cdot \rangle$ , estimating the residual $\nabla f ( x ) - m$ and adding m back. Equivalently, rearranging gives $G ( x , m ; u ) = G ( x ; u ) + ( I _ { d } - u u ^ { \top } ) m$ . The term $( I _ { d } - u u ^ { \top } ) m$ is a zero-mean control variate: it leaves the bias unchanged while reducing variance when m approximates $\nabla f ( x )$ The next lemma makes this precise.

Lemma 3.3 For all $x , m \in \mathbb { R } ^ { d }$ , it holds that $\begin{array} { r } { \mathbb E _ { u \sim \mathcal { N } ( 0 , I _ { d } ) } \left[ G ( x , m ; u ) \right] = \nabla f ^ { \tau } ( x ) } \end{array}$ . Moreover,

$$
\begin{array} { r } { {  { \mathbb E } } _ { u \sim \mathcal { N } ( 0 , I _ { d } ) } \left[ \| G ( x , m ; u ) - \nabla f ( x ) \| ^ { 2 } \right] \le 2 ( d + 1 ) \| m - \nabla f ( x ) \| ^ { 2 } + 8 \tau ^ { 2 } L ^ { 2 } d ^ { 3 } } \end{array}
$$

When m closely approximates $\nabla f ( x )$ , the leading term $d \| m - \nabla f ( x ) \| ^ { 2 }$ becomes small, substantially improving over Lemma 3.2 in which the corresponding term scales as $d \| \nabla f ( x ) \| ^ { 2 }$

## 3.2 Descent Scheme and General Convergence Guarantee

We now embed the control-variate estimator into an iterative descent scheme (Algorithm 1). At each iteration t, the algorithm selects a reference vector $m _ { t }$ and a step size $\alpha _ { t } ,$ , draws a fresh exploration direction $u _ { t } \sim \mathcal { N } ( 0 , I _ { d } )$ , and takes a gradient step using the control-variate estimator $G ( x _ { t } , m _ { t } ; u _ { t } )$ . The final output is sampled from the iterates with probabilities proportional to the step sizes taken.

```powershell
Algorithm 1: Control-Variate Zeroth-Order Descent (CV-ZOD)
Input : Initial point $x _ { 1 } \in \mathbb { R } ^ { d } .$
For each iteration $t = 1 , 2 , \dots , T \colon$
1. Select the reference vector $m _ { t }$ and step-size $\alpha _ { t } > 0 .$
2. Update $x _ { t + 1 } = x _ { t } - \alpha _ { t } G ( x _ { t } , m _ { t } ; u _ { t } )$ for independently sampled $u _ { t } \sim \mathcal { N } ( 0 , I _ { d } )$
Output $: { \bar { x } } = x _ { R } ,$ , where $R \in [ T ]$ is sampled conditionally on the realized run with probabilities $\textstyle \alpha _ { t } / \sum _ { s = 1 } ^ { T } \alpha _ { s } .$
```

The exploration direction $u _ { t }$ is sampled from $\mathcal { N } ( 0 , I _ { d } )$ independently of all information available when it is drawn, including $\left( { { x } _ { t } } , { { m } _ { t } } , { { \alpha } _ { t } } \right)$ . The reference vector and step size may depend arbitrarily on the history, including the current iterate and any external available information.

The convergence of Algorithm 1 is governed by the interplay between the step sizes $\alpha _ { t }$ and the quality of the reference vectors $m _ { t }$ . By Lemma 3.3, taking a step of size $\alpha _ { t }$ incurs estimation variance proportional to $\alpha _ { t } ^ { 2 } d \parallel m _ { t } - \nabla f ( x _ { t } ) \parallel ^ { 2 }$ up to smoothing. Larger step sizes amplify this variance, while more accurate reference vectors reduce it. To quantify whether the step sizes are justified by the reference quality along the trajectory, we introduce the variance-descent balance

$$
\begin{array} { r } { \mathcal { B } _ { T } ^ { \gamma } : = \sum _ { t = 1 } ^ { T } \left( \alpha _ { t } ^ { 2 } d \| m _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } - \frac { \alpha _ { t } } { \gamma L } \| \nabla f ( x _ { t } ) \| ^ { 2 } \right) , } \end{array}\tag{3}
$$

which accumulates, over iterations, the variance cost of each step minus a descent credit, with the scale parameter $\gamma \geq 1$ controlling the relative weight between the two. We fix $\gamma = C _ { 0 } \log ( 2 T ^ { 2 } )$ for most of our applications, where $C _ { 0 }$ is the absolute constant from the following theorem. This theorem shows that, for a B-Lipschitz and L-smooth objective $f ,$ the sign and magnitude of $B _ { T } ^ { \gamma }$ govern the convergence rate of Algorithm 1.

Theorem 3.4 (Convergence of CV-ZOD) Suppose Algorithm 1 uses a fresh standard Gaussian direction $u _ { t } ,$ conditionally on all information available before its draw, for every $t \in [ T ]$ , and that ma $\mathrm { x } _ { t \in [ T ] } \| m _ { t } \| \leq 2 B$ almost surely. Then there exist absolute constants $C _ { 0 } , C > 0$ such that the following holds. For any $\delta \in ( 0 , 1 )$ , set $\iota : = \log ( 2 T / \delta )$ , and suppose $d \geq \iota$ and $\begin{array} { r } { 0 < \alpha _ { t } \le \frac { 1 } { C _ { 0 } L \iota } } \end{array}$ for all $t \in [ T ]$ . Then, $f o r \gamma = C _ { 0 } \iota$ , with probability at least $1 - \delta ,$

$$
\begin{array} { r } { \frac { 1 } { L } \sum _ { t = 1 } ^ { T } \alpha _ { t } \| \nabla f ( x _ { t } ) \| ^ { 2 } \leq C \left( D _ { f } ^ { 2 } + \frac { B ^ { 2 } } { L ^ { 2 } } + \tau ^ { 2 } T d ^ { 3 } + \mathcal { B } _ { T } ^ { \gamma } \iota \right) , } \end{array}
$$

where $\begin{array} { r } { D _ { f } ^ { 2 } : = \frac { 2 } { L } \left( f ( x _ { 1 } ) - f ^ { * } \right) } \end{array}$ . Moreover, $i f d \ge \log ( 2 T ^ { 2 } ) , \gamma = C _ { 0 } \log ( 2 T ^ { 2 } ) , 0 < \alpha _ { t } \le 1 / ( \gamma L ) f o r a l l t \in [ T ]$ and $\begin{array} { r } { \tau \leq \frac { B / L } { \sqrt { T d ^ { 3 } } } } \end{array}$ , then x satisfies

$$
\begin{array} { r } { \frac { 1 } { L } \mathbb { E } \bigl [ \| \nabla f ( \overline { { x } } ) \| ^ { 2 } \bigr ] \leq C \mathbb { E } \biggl [ \frac { D _ { f } ^ { 2 } + B ^ { 2 } / L ^ { 2 } + B _ { T } ^ { \gamma } \log ( 2 T ^ { 2 } ) } { A _ { T } } \biggr ] , } \end{array}
$$

where $\textstyle A _ { T } : = \sum _ { t = 1 } ^ { T } \alpha _ { t }$ denotes the cumulative step size.

The proof, which appears in Appendix $\mathrm { C } ,$ is based on careful control of self-normalized martingales arising from the interaction between the step sizes and the control-variate estimation error. The condition $d \geq \iota = \log ( 2 T / \delta )$ is mild in the high-dimensional regime we consider; when it fails, one may replace d by $d + \iota$ in all bounds. The bound interpolates between two familiar regimes. When the reference vectors are uninformative $( m _ { t } = 0$ for all $t ) .$ taking $\alpha _ { t } = 1 / ( \gamma L d )$ ensures $B _ { T } ^ { \gamma } \leq 0$ and recovers the standard zeroth-order rate $O ( d / T )$ up to the factor $\gamma ,$ which is logarithmic for our choice $\gamma \overset { \cdot } { = } C _ { 0 } \log ( 2 T ^ { 2 } )$ . When the reference vectors match the gradient $( m _ { t } = \nabla f ( x _ { t } )$ for all t), taking $\alpha _ { t } = 1 / ( \gamma L )$ again gives $B _ { T } ^ { \gamma } \leq 0$ and recovers the first-order rate $O ( 1 / T )$ up to the same logarithmic factor. In general, the quality of the reference vectors determines how large the step sizes can be while keeping $B _ { T } ^ { \gamma }$ controlled, and thus the degree of acceleration beyond the zeroth-order baseline.

Remark 3.5 (The $B ^ { 2 } / L ^ { 2 }$ term) The quantities $\begin{array} { r } { D _ { f } ^ { 2 } = \frac { 2 } { L } ( f ( x _ { 1 } ) - f ^ { * } ) } \end{array}$ and $B ^ { 2 } / L ^ { 2 }$ are not generally comparable. The former is the standard initial suboptimality and satisfies $D _ { f } ^ { 2 } \geq \| \nabla f ( x _ { 1 } ) \| ^ { 2 } / L ^ { 2 }$ . The latter arises from the high-probability martingale concentration used in the proof.

## 3.3 Locally optimal choices from directional hints

Theorem 3.4 holds for arbitrary reference vectors and step sizes. To understand the best achievable rate given the hint subspaces $( S _ { t } ) _ { t = 1 } ^ { T } .$ , consider the idealized choice $m _ { t } = P _ { S _ { t } } \nabla f ( x _ { t } )$ , which minimizes $\| m - \nabla f ( x _ { t } ) \| ^ { 2 }$ over $m \in S _ { t }$ . Under this choice, the residual satisfies $\| m _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } = \sin ^ { 2 } \theta _ { t } \| \nabla f ( x _ { t } ) \| ^ { 2 }$ , and each summand of $B _ { T } ^ { \gamma }$ is non-positive whenever $\begin{array} { r } { \alpha _ { t } \leq \frac { 1 } { \gamma L d \sin ^ { 2 } \theta _ { t } } } \end{array}$

This motivates defining, for any realization of the trajectory, the oracle reference vector and step size:

$$
\begin{array} { r } { m _ { t } ^ { * } : = P _ { S _ { t } } \nabla f ( x _ { t } ) , \qquad \alpha _ { t } ^ { * } : = \frac { 1 } { L } \operatorname* { m i n } \Bigl \{ 1 , \frac { 1 } { d \sin ^ { 2 } \theta _ { t } } \Bigr \} \in \Bigl [ \frac { 1 } { L d } , \frac { 1 } { L } \Bigr ] \ . } \end{array}\tag{4}
$$

They serve as a natural benchmark: $\alpha _ { t } ^ { * } / \gamma$ is the largest step size for which the t-th summand of $B _ { T } ^ { \gamma }$ remains non-positive under the best reference vector from $S _ { t }$ . An algorithm using $( m _ { t } ^ { * } , \alpha _ { t } ^ { * } / \gamma )$ at each iteration would ensure $B _ { T } ^ { \gamma } \leq 0$ , and Theorem 3.4 would yield

$$
\begin{array} { r } { \frac { 1 } { L } \mathbb { E } \left[ \| \nabla f ( \overline { { x } } ) \| ^ { 2 } \right] \leq C \mathbb { E } \left[ \frac { \gamma ( D _ { f } ^ { 2 } + B ^ { 2 } / L ^ { 2 } ) } { A _ { T } ^ { * } } \right] , } \end{array}\tag{5}
$$

where $\begin{array} { r } { \mathcal { A } _ { T } ^ { * } : = \sum _ { t = 1 } ^ { T } \alpha _ { t } ^ { * } } \end{array}$ . This rate interpolates with the hint quality: when $\theta _ { t } = 0$ for all $t ,$ we have $\alpha _ { t } ^ { * } = 1 / L$ and recover $O ( 1 / T ) ;$ ; when $\theta _ { t } = \pi / 2$ , we have $\alpha _ { t } ^ { * } = 1 / ( L d )$ and recover $O ( d / T )$

The quantities $m _ { t } ^ { * }$ and $\alpha _ { t } ^ { * }$ can be viewed as locally optimal in the sense that they minimize the variance-descent balance at each iteration given only the current pair $( x _ { t } , S _ { t } )$ , without knowledge of the future trajectory. While they depend on the unknown gradient and cannot be computed directly, approximating them from function evaluations is a natural goal. The following section shows that this can be done.

## 4 Main Results: Adaptive Convergence from Directional Hints

This section constructs a fully zeroth-order instantiation of CV-ZOD. Section 4.1 estimates the projected gradient and the squared norms of the full gradient and its component orthogonal to the hint subspace. Section 4.2 uses these estimates to choose the reference vector and an adaptive step size. Theorem 4.4 establishes a convergence bound governed by the cumulative oracle step size $\mathcal { A } _ { T } ^ { \ast }$ evaluated along the algorithm’s realized trajectory. Proofs appear in Appendix D.

## 4.1 Zeroth-Order Estimation Primitives

Estimating the in-subspace gradient component. To construct the reference vector, we estimate the projected gradient $P _ { S } \nabla f ( x )$ by probing the function once along each vector in an orthonormal basis of S. Algorithm 2 therefore uses $k + 1$ oracle queries, and its approximation guarantee is deterministic under L-smoothness.

```latex
Algorithm 2: $P _ { f } ^ { \tau } ( x , S )$
Input : Point $x \in \mathbb { R } ^ { d } .$ , subspace $s$ of dimension $k \in [ d ]$
1. Construct $B = [ b _ { 1 } , \dots , b _ { k } ] \in \mathbb { R } ^ { d \times k }$ with orthonormal columns spanning ${ \mathcal { S } } .$
2. For $j \in [ k ]$ , compute $\begin{array} { r } { \widehat { y _ { j } } = \frac { \dot { f } ( x + \tau b _ { j } ) - f ( x ) } { \tau } } \end{array}$
3. Output $B { \widehat { y } } ,$ where $\widehat { \boldsymbol { y } } = ( \widehat { y } _ { 1 } , \dots , \widehat { y } _ { k } ^ { \prime } ) ^ { \intercal }$
Lemma 4.1 (Deterministic projection estimation) For every $x \in \mathbb { R } ^ { d }$ and $S \in \operatorname { G r } _ { k } ( \mathbb { R } ^ { d } )$ , Algorithm $^ 2$ satisfies
$\Vert P _ { f } ^ { \tau } ( x , S ) - P s \nabla f ( x ) \Vert ^ { 2 } \leq \frac { k L ^ { 2 } \tau ^ { 2 } } { 4 } .$
The bound follows directly by applying the smoothness remainder to each basis direction; in particular, no
concentration argument or logarithmic oversampling is needed.
Estimating gradient norms. To set the step size adaptively, we estimate $\| \nabla f ( x _ { t } ) \| ^ { 2 }$ and $\| P _ { S _ { t } ^ { \perp } } \nabla f ( x _ { t } ) \| ^ { 2 }$
Both quantities are squared norms of gradient projections, which can be approximated via a common primitive:
Algorithm 3 computes a constant-factor approximation by averaging estimates from N Gaussian probes.
```

```latex
Algorithm 3: $N _ { f } ^ { \tau } ( x , S , N )$
Input : Point $x \in \mathbb { R } ^ { d }$ , linear subspace $S \operatorname { o f } \mathbb { R } ^ { d }$ , number of sampled directions $N .$
1. For $i \in [ N ]$ , sample $u _ { i } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , I _ { d } )$ and compute $\begin{array} { r } { \widehat { y } _ { i } = \frac { f ( x + \tau P _ { S } u _ { i } ) - f ( x ) } { \tau } } \end{array}$
2. Output $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \widehat { y } _ { i } ^ { 2 } . } \end{array}$
Lemma 4.2 (Norm estimation) There exist absolute constants $C , c > 0$ such that for every $x \in \mathbb { R } ^ { d } ,$ , linear
subspace S $o f \mathbb { R } ^ { d } , \tau > 0 , \delta \in ( 0 , 1 )$ , and $N \geq C \log ( 2 / \delta )$ , it holds with probability at least $1 - \delta$ that
$\begin{array} { r } { N _ { f } ^ { \tau } ( x , S , N ) \in \left[ \frac { 1 } { 2 } \| P _ { S } \nabla f ( x ) \| ^ { 2 } - c \varepsilon , \frac { 3 } { 2 } \| P _ { S } \nabla f ( x ) \| ^ { 2 } + c \varepsilon \right] } \end{array}$
where $\varepsilon = \tau ^ { 2 } L ^ { 2 } ( d ^ { 2 } + \log ^ { 2 } ( 2 N / \delta ) ) .$
The constant-factor accuracy suffices: the adaptive rule below uses these estimates only to identify the correct scale
of $\alpha _ { t } ^ { * }$ up to a constant factor.
```

## 4.2 Adaptive guarantees

We now combine the two estimation primitives. The reference vector $m _ { t }$ is set using Algorithm 2. The next lemma defines the adaptive step size $\alpha _ { t }$ via the norm estimates from Algorithm 3 and establishes two properties: it tracks the oracle step size $\alpha _ { t } ^ { * }$ up to a constant factor, and it keeps the descent balance controlled.

Lemma 4.3 There exist absolute constants $C , c _ { 0 } , c _ { 1 } > 0$ such that thefollowing holds. For any $\delta \in ( 0 , 1 )$ and $N \geq C \log ( 2 T / \delta ) , s e t \varepsilon : = \tau ^ { 2 } L ^ { 2 } ( d ^ { 2 } + \log ^ { 2 } ( 2 N T / \delta ) )$ and define

$$
\begin{array} { r } { \alpha _ { t } : = \frac { 1 } { \gamma L } \operatorname* { m i n } \left\{ 1 , \frac { N _ { f } ^ { \tau } ( x _ { t } , \mathbb { R } ^ { d } , N ) + c _ { 1 } \varepsilon } { 1 6 d \operatorname* { m a x } \{ 0 , N _ { f } ^ { \tau } ( x _ { t } , S _ { t } ^ { \bot } , N ) - c _ { 1 } \varepsilon \} } \right\} , } \end{array}\tag{6}
$$

If the denominator is zero, define $\alpha _ { t } : = 1 / ( \gamma L )$ . Then with probability at least 1 − δ, for all $t \in [ T ]$ simultaneously: 1. (Local Optimality) $\begin{array} { r } { \alpha _ { t } \geq \frac { \alpha _ { t } ^ { * } } { c _ { 0 } \gamma } ; } \end{array}$

$$
\begin{array} { r l } { 2 . \ ( B a l a n c e \ C o n t r o l ) } & { \ \alpha _ { t } ^ { 2 } d \| P _ { S _ { t } } ^ { \bot } \nabla f ( x _ { t } ) \| ^ { 2 } \leq \frac { \alpha _ { t } } { 2 \gamma L } \| \nabla f ( x _ { t } ) \| ^ { 2 } + c _ { 0 } \tau ^ { 2 } ( d ^ { 3 } + d \log ^ { 2 } ( 2 N T / \delta ) ) . } \end{array}
$$

Local optimality ensures that the realized step sizes are within a constant factor of the locally optimal ones (oracle $\alpha _ { t } ^ { * } )$ ; balance control ensures that $B _ { T } ^ { \gamma }$ is dominated by smoothing terms. Substituting both $m _ { t }$ and $\alpha _ { t }$ into Theorem 3.4 yields our main convergence result.

Theorem 4.4 There exist absolute constants $C , c > 0$ such that, $i f d \ge \log ( 2 T ^ { 2 } )$ , Algorithm 1 is run with $m _ { t } = P _ { f } ^ { \tau } ( x _ { t } , S _ { t } )$ and step sizes $\alpha _ { t } f r o m \left( 6 \right)$ with $\gamma = C _ { 0 } \log ( 2 T ^ { 2 } )$ and $N = \lceil C \log ( 2 T ) \rceil$ , and the smoothing radius satisfies $\begin{array} { r } { \tau \leq \frac { B / L } { \sqrt { T ( d ^ { 3 } + d k ) \log ( 2 T ^ { 2 } ) } } } \end{array}$ , then

$$
\begin{array} { r } { \frac { 1 } { L } \mathbb { E } \big [ \| \nabla f ( \overline { { x } } ) \| ^ { 2 } \big ] \leq c \mathbb { E } \Big [ \frac { ( D _ { f } ^ { 2 } + B ^ { 2 } / L ^ { 2 } ) \log ( 2 T ) } { A _ { T } ^ { * } } \Big ] , } \end{array}
$$

where $\begin{array} { r } { \mathcal { A } _ { T } ^ { * } = \sum _ { t = 1 } ^ { T } \alpha _ { t } ^ { * } f o r \alpha _ { t } ^ { * } } \end{array}$ in (4).

Thus, the algorithm matches the locally optimal rate up to logarithmic factors. The cumulative oracle step size $\boldsymbol { \mathcal { A } } _ { T } ^ { * }$ governs the rate: the bound interpolates between $O ( 1 / T )$ and $O ( d / T )$ depending on how well the hint subspaces align with the gradient along the trajectory. Algorithm 2 uses $k + 1$ queries, while the norm estimates in Algorithm 3 use ${ \cal O } ( \log T )$ queries. Hence the total per-iteration query cost beyond the single control-variate gradient query is $O ( k + \log T ) -$ modest when the hint dimension k is small relative to d.

Remark 4.5 (Variance reduction by batching) Our analysis uses a single perturbation $u _ { t }$ per iteration for the control-variate gradient step, with the remaining queries allocated to the estimation primitives. The framework extends directly to a batch of b independent directions $u _ { t } ^ { 1 } , \ldots , u _ { t } ^ { b } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , I _ { d } )$ . From Lemmas 3.1 and 3.3,

$$
\begin{array} { r } { \mathbb { E } \bigg [ \Big \| \frac { 1 } { b } \sum _ { i = 1 } ^ { b } G ( x _ { t } , m _ { t } ; u _ { t } ^ { i } ) - \nabla f ( x _ { t } ) \Big \| ^ { 2 } \bigg | \ x _ { t } , m _ { t } \bigg ] \leq \frac { 4 d } { b } \| m _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } + 2 4 \tau ^ { 2 } L ^ { 2 } d ^ { 3 } . } \end{array}
$$

Thus batching reduces estimator’s variance by a factor of $b ,$ up to Gaussian-smoothing bias. The convergence analysis can be modified accordingly with a proportional improvement in the convergence bound.

## 5 Simulations

We evaluate CV-ZOD on two simulation-based optimization tasks where a cheap surrogate provides directional hints of varying quality: a fluid inverse problem (Section 5.1) and molecular geometry optimization (Section 5.2). In both, the optimization updates access the expensive objective f through function evaluations, while the differentiable surrogate $f ^ { \prime }$ supplies rank-one hint subspaces $S _ { t } = \operatorname { s p a n } \{ \nabla f ^ { \prime } ( x _ { t } ) \}$ . We compare against classical zeroth-order descent (ZOD) [GL13], which uses no hints, and Guided Evolutionary Strategies (GES) $[ \mathbf { M } \mathbf { M } \mathbf { T } ^ { + } 1 9 ]$ , which incorporates the hint subspace through a biased estimator. All these methods share the same per-iteration query budget. For reference, we also include surrogate gradient descent (SurGD), which takes gradient steps along $\nabla f ^ { \prime } ( x _ { t } )$ , effectively optimizing the surrogate rather than $f .$

Remark 5.1 (Implementation of the adaptive step size) In practice, we found it more stable and query-efficient to smooth the norm estimates over time using exponential discounting. Let $a _ { t } : = N _ { f } ^ { \tau } ( x _ { t } , \mathbb { R } ^ { d } , N _ { \mathrm { f u l l } } )$ and $b _ { t } : =$

$N _ { f } ^ { \tau } ( x _ { t } , S _ { t } ^ { \bot } , N _ { \bot } )$ , where $N _ { \mathrm { f u l l } }$ and $N _ { \perp }$ are the numbers of probes allocated to the two norm estimates, and fix a discount factor $\gamma _ { \mathrm { d i s c } } \in [ 0 , 1 )$ ). Starting from $A _ { 0 } = B _ { 0 } = 0$ , we update

$$
A _ { t } = \gamma _ { \mathrm { d i s c } } A _ { t - 1 } + a _ { t } , \qquad B _ { t } = \gamma _ { \mathrm { d i s c } } B _ { t - 1 } + b _ { t } , \qquad \alpha _ { t } ^ { \mathrm { e x p } } = \frac { A _ { t } } { B _ { t } } .
$$

In our simulations, we use this smoothed, recency-biased approximation to the adaptive rule. We support this implementation choice with an ablation experiment in Appendix B.

## 5.1 Fluid Inverse Problem

Fluid dynamics model. Recovering initial conditions from terminal observations in fluid systems is a classical PDE-constrained inverse problem with applications in geophysics and weather forecasting [Gun03]. We study a concrete instance: a passive dye is released into a two-dimensional incompressible fluid governed by the Navier– Stokes equations, discretized on an $n \times n$ spatial grid via the semi-Lagrangian advection–projection scheme of [Sta99]. The dye density $\rho _ { s } \in \mathbb { R } ^ { n \times n }$ evolves jointly with a velocity field $v _ { s } \in \mathbb { R } ^ { n \times n \times 2 }$ over $s = 0 , \ldots , T _ { \mathrm { s i m } }$ time steps. Given a fixed initial density $\rho _ { 0 }$ and a prescribed target $\rho ^ { \star } \in \mathbb { R } ^ { n \times n }$ , the goal is to find an initial velocity field $v _ { 0 } \in \mathbb { R } ^ { n \times n \times 2 }$ that minimizes $f ( v _ { 0 } ) : = \| \rho _ { T _ { \mathrm { s i m } } } ( v _ { 0 } ) - \rho ^ { \star } \| _ { \mathrm { F } } ^ { 2 }$ , where $\rho _ { T _ { \mathrm { s i m } } } ( v _ { 0 } )$ is the terminal density obtained by evolving the dynamics from $( \rho _ { 0 } , v _ { 0 } )$ . We identify the decision variable $v _ { 0 }$ with a vector $x \in \mathbb { R } ^ { d }$ for $d = 2 n ^ { 2 }$ and view $f \colon  { \mathbb { R } ^ { d } } \to$ R accordingly.

Oracle and surrogate. To simulate the dynamics, we use the differentiable fluid solver provided by NVIDIA Warp [Mac22]. The solver supports reverse-mode automatic differentiation, but backpropagating through the full $T _ { \mathrm { s i m } }$ -step rollout naively requires storing all intermediate density and velocity fields, at $O ( T _ { \mathrm { s i m } } \cdot n ^ { 2 } )$ memory cost. Gradient checkpointing [GW00] can reduce this to $O ( \sqrt { T _ { \mathrm { s i m } } } \cdot n ^ { 2 } )$ at the cost of additional forward recomputation, but the memory overhead remains substantial for long time horizons. Forward passes, used by our algorithm as zeroth-order oracle calls, require no intermediate storage. To construct the directional hints, we use a coarse surrogate $f ^ { \prime } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ defined by the same solver over a shorter time horizon $T _ { \mathrm { s u r } } \ll T _ { \mathrm { s i m } }$ . Since $T _ { \mathrm { s u r } }$ is small, backpropagation through $f ^ { \prime }$ is cheap.

Experimental setup. We set the grid size to $n = 6 4 \ : ( d = 2 n ^ { 2 } = 8 1 9 2 )$ , the simulation horizon to $T _ { \mathrm { s i m } } = 6 4$ steps, and the surrogate horizon to $T _ { \mathrm { s u r } } = 1 6$ steps. Each zeroth-order method receives a per-iteration budget of 16 function evaluations. ZOD and GES use all 16 queries for batching, while CV-ZOD allocates 12 to the batched control-variate gradient step, 2 to the gradient projection (Algorithm 2), and 2 to the norm estimates used by the adaptive step-size rule (Remark 5.1). Appendix B reports ablations over the discount factor and query allocation for this fluid problem; we present the best-performing configuration here.

We use 15 paired seeds and run each trajectory for 100 optimization steps. All runs start from a low-variance Gaussian perturbation of zero initial velocity to avoid the nondifferentiable stationary point. The initial density $\rho _ { 0 }$ and target $\rho ^ { \star }$ are a vertical and horizontal rectangle, respectively, centered on the grid (Figure 2, right).

![](images/a71915fdb11fa40d345abfe4430c4b320b718d05cfec3c3f7f4bcd59f749fe9e.jpg)  
Figure 2: Fluid inverse problem. Left: objective averaged over 15 runs (top) and cosine similarity between the surrogate and true gradients averaged over the same runs (bottom). Right: sample terminal density fields produced by each method, alongside the initial density and target.

Results. Figure 2 (left) shows the objective value averaged over 15 paired runs, alongside $\cos ( \theta _ { t } )$ for CV-ZOD and GES. Figure 2 (right) shows terminal density fields from a common seed. Early in optimization, when the surrogate gradient is well-aligned with the true gradient, CV-ZOD descends at a rate comparable to GES and surrogate descent. As the alignment deteriorates, GES plateaus and SurGD loses much of its early improvement. CV-ZOD, by contrast, transitions smoothly to zeroth-order exploration and reaches the lowest final objective.

## 5.2 Molecular Geometry Optimization

Conformation problem and energy surrogates. Finding low-energy molecular conformations is a central task in computational chemistry [Haw17]. Given a molecule with N atoms, the goal is to find atomic positions $x \in  { \mathbb { R } } ^ { 3 N }$ minimizing the potential energy. In practice, the true energy surface is accessed through expensive quantummechanical calculations. For our experiments, we substitute the MACE-OFF23 machine-learned interatomic potential [KMB<sup>+</sup>25] as a realistic proxy, setting $f ( x ) : = E _ { \mathrm { M A C E } } ( x )$ and treating it as a zeroth-order oracle. As surrogate, we use the MMFF94 classical force field [Hal96], an analytic potential with closed-form gradients. Since MACE-OFF23 is itself a differentiable PyTorch model, its gradients are available as a diagnostic, allowing us to compute the true principal angle $\angle ( \nabla f ( x _ { t } ) , S _ { t } )$ along each trajectory for post-hoc analysis.

Experimental setup. We select three molecules spanning a range of dimensions $( d = 3 N )$ : efavirenz $( N = 3 0 )$ adenosine $( N = 3 2 )$ ), and benzylpenicillin $( N = 4 1 )$ . Starting geometries are taken from Wiggle150, a benchmark of highly strained molecular conformations $[ \mathrm { B N B } ^ { + } 2 5 ]$ , with ten conformations per molecule. Each trajectory runs for 100 optimization steps, with starting coordinates and seeds shared across the algorithms and the SurGD reference. We test the same algorithms with the same query allocations as in the fluid experiments (Section 5.1).

Results. Figure 3 shows energy changes and running cosine similarities for each molecule, averaged over 10 paired runs. In this setting, the surrogate gradient retains useful alignment with the true gradient throughout optimization, and both CV-ZOD and GES outperform unguided ZOD. The SurGD reference steadily reduces the true energy, illustrating the quality of the surrogate. CV-ZOD achieves a similar energy reduction and outperforms GES.

![](images/4f41135e1a48df3556322856136c082cdae5f9dcac432717d31dbd89bdc0c192.jpg)  
Figure 3: Molecular geometry optimization. MACE energy change from initialization (top; lower is better) and cosine similarity between surrogate and true gradients along CV-ZOD and GES trajectories (bottom) for efavirenz $( d = 9 0 )$ , adenosine $( d = 9 6 )$ , and benzylpenicillin $( d = 1 2 3 )$ . Curves average ten paired runs, with shared vertical scales across molecules. CV-ZOD remains close to the surrogate-only reference (SurGD, dotted).

## 6 Discussion and Future Work

We introduced CV-ZOD, a framework for incorporating directional hints into zeroth-order optimization through control-variate gradient estimation. The key property of the framework is that hints affect only the variance of the gradient estimator, not its bias, so the optimization trajectory tracks the true objective regardless of hint quality. The adaptive step-size rule matches the oracle rate up to logarithmic factors without requiring any prior knowledge of the alignment between the hints and the gradient along the trajectory, at a per-iteration query cost of only $O ( k + \log T )$

Several directions remain open. First, the current framework processes a single low-dimensional hint subspace per iteration. In practice, one may have access to multiple surrogates whose gradients accumulate over iterations; naively taking their span as the hint subspace causes the dimension k to grow, eroding low-dimensionality and increasing the required number of queries. A natural question is whether one can adaptively aggregate a fixed number of surrogate gradients in an optimal way—for instance, through an online learning scheme over the surrogates. Second, the most compelling test of the framework’s scalability would be in large-scale settings such as memory-efficient fine-tuning or prompt optimization of language models, where zeroth-order methods are already in use and surrogate gradients from smaller or distilled models arise naturally. Validating CV-ZOD in these high-dimensional regimes is an important direction for future work.

## References

[AL24] Sohei Arisaka and Qianxiao Li. Accelerating legacy numerical solvers by non-intrusive gradient-based meta-solving. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 1689–1708. PMLR, 2024.

[BG19] Krishnakumar Balasubramanian and Saeed Ghadimi. Zeroth-order nonconvex stochastic optimization: Handling constraints, high-dimensionality and saddle-points, 2019.

[BMG<sup>+</sup>20] Benjamin Burger, Phillip M. Maffettone, Vladimir V. Gusev, Catherine M. Aitchison, Yang Bai, Xiaoyan Wang, Xiaobo Li, Ben M. Alston, Buyi Li, Rob Clowes, Nicola Rankin, Brandon Harris, Reiner Sebastian Sprick, and Andrew I. Cooper. A mobile robotic chemist. Nature, 583(7815):237– 241, 2020.

[BNB<sup>+</sup>25] Rebecca R. Brew, Ian A. Nelson, Meruyert Binayeva, Amlan S. Nayak, Wyatt J. Simmons, Joseph J. Gair, and Corin C. Wagen. Wiggle150: Benchmarking density functionals and neural network potentials on highly strained conformers. Journal ofChemical Theory and Computation, 21(8):3922– 3929, 2025. https://doi.org/10.1021/acs.jctc.5c00015.

[CWZ21] Shuyu Cheng, Guoqiang Wu, and Jun Zhu. On the convergence of prior-guided zeroth-order optimization algorithms. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pages 14620–14631, 2021.

[CZS<sup>+</sup>17] Pin-Yu Chen, Huan Zhang, Yash Sharma, Jinfeng Yi, and Cho-Jui Hsieh. ZOO: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models. In Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security (AISec), pages 15–26, 2017.

[FGL15] Xiequan Fan, Ion Grama, and Quansheng Liu. Exponential inequalities for martingales with applications. Electronic Journal ofProbability, 20(1):1–22, 2015.

[GBB04] Evan Greensmith, Peter L. Bartlett, and Jonathan Baxter. Variance reduction techniques for gradient estimates in reinforcement learning. Journal ofMachine Learning Research, 5:1471–1530, 2004.

[GL13] Saeed Ghadimi and Guanghui Lan. Stochastic first- and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization, 23(4):2341–2368, 2013.

[Gun03] Max D. Gunzburger. Perspectives in Flow Control and Optimization, volume 5 of Advances in Design and Control. SIAM, Philadelphia, 2003.

[GW00] Andreas Griewank and Andrea Walther. Algorithm 799: Revolve: An implementation of checkpointing for the reverse or adjoint mode of computational differentiation. ACM Transactions on Mathematical Software, 26(1):19–45, 2000.

[Hal96] Thomas A Halgren. Merck molecular force field. I. Basis, form, scope, parameterization, and performance of MMFF94. Journal ofComputational Chemistry, 17(5–6):490–519, 1996.

[Haw17] Paul CD Hawkins. Conformation generation: The state of the art. Journal ofChemical Information and Modeling, 57(8):1747–1756, 2017.

[IEAL18] Andrew Ilyas, Logan Engstrom, Anish Athalye, and Jessy Lin. Black-box adversarial attacks with limited queries and information. In Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80 of Proceedings of Machine Learning Research, pages 2137–2146. PMLR, 2018.

[KMB<sup>+</sup>25] Dávid Péter Kovács, J. Harry Moore, Nicholas J. Browning, Ilyes Batatia, Joshua T. Horton, Yixuan Pu, Venkat Kapil, William C. Witt, Ioan-Bogdan Magdau, Daniel J. Cole, and Gábor Csányi. MACE-OFF:˘ Short-range transferable machine learning force fields for organic molecules. Journal ofthe American Chemical Society, 147(21):17598–17611, 2025.

[LMW19] Jeffrey Larson, Matt Menickelly, and Stefan M. Wild. Derivative-free optimization methods. Acta Numerica, 28:287–404, 2019.

[Mac22] Miles Macklin. Warp: A high-performance Python framework for GPU simulation and graphics. https://github.com/nvidia/warp, March 2022. NVIDIA GPU Technology Conference (GTC).

[MGN<sup>+</sup>23] Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, and Sanjeev Arora. Fine-tuning language models with just forward passes. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[MMT<sup>+</sup>19] Niru Maheswaranathan, Luke Metz, George Tucker, Dami Choi, and Jascha Sohl-Dickstein. Guided evolutionary strategies: Augmenting random search with surrogate gradients. In Proceedings ofthe 36th International Conference on Machine Learning (ICML), volume 97, pages 4264–4273. PMLR, 2019.

[NS17] Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions. Foundations ofComputational Mathematics, 17(2):527–566, 2017.

[Owe13] Art B. Owen. Monte Carlo Theory, Methods and Examples. 2013.

[SSL<sup>+</sup>21] Benjamin J. Shields, Jason Stevens, Jun Li, Marvin Parasram, Farhan Damani, Jesus I. Martinez Alvarado, Jacob M. Janey, Ryan P. Adams, and Abigail G. Doyle. Bayesian reaction optimization as a tool for chemical synthesis. Nature, 590(7844):89–96, 2021.

[Sta99] Jos Stam. Stable fluids. In Proceedings ofthe 26th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), pages 121–128, 1999.

[ZCD<sup>+</sup>24] Heshen Zhan, Congliang Chen, Tian Ding, Ziniu Li, and Ruoyu Sun. Unlocking black-box prompt tuning efficiency via zeroth-order optimization. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 14825–14838, 2024.

## A Prior Work on Zeroth-Order Optimization with Directional Hints

This section analyzes the gradient-estimation techniques used by the two prior approaches for incorporating directional guidance into zeroth-order optimization: Guided Evolutionary Strategies (GES) [MMT<sup>+</sup>19] and Prior Guided Random Gradient-Free (PRGF) methods [CWZ21]. Both reduce exploration variance along a guided subspace by modifying the search distribution, but in doing so introduce bias into the gradient estimator. We show that this bias creates a fundamental step-size dilemma that the control-variate construction in CV-ZOD avoids.

Guided Evolutionary Strategies. The approach of $[ \mathbf { M } \mathbf { M } \mathbf { T } ^ { + } 1 9 ]$ replaces isotropic perturbations with anisotropic ones whose covariance allocates more variance to directions in the hint subspace $S \in \operatorname { G r } _ { k } ( \mathbb { R } ^ { d } )$ . Concretely, one uses the same estimator $G ( x ; u )$ , but samples perturbations according to

$$
\begin{array} { r } { u \sim { \mathcal N } ( 0 , \Sigma ) , \qquad \mathrm { w h e r e } \Sigma : = \gamma I _ { d } + ( 1 - \gamma ) \frac { d } { k } P s , } \end{array}
$$

with $\gamma \in [ 0 , 1 ]$ interpolating between isotropic exploration $( \gamma = 1 )$ and exploration supported entirely on $\boldsymbol { S } \left( \gamma = 0 \right)$ while preserving the total variance $\operatorname { t r } ( \Sigma ) = d .$

The main drawback is that this introduces an anisotropic bias. Defining $f _ { \Sigma } ^ { \tau } ( x ) : = \mathbb { E } _ { u \sim \mathcal { N } ( 0 , \Sigma ) } [ f ( x + \tau u ) ]$ , a direct calculation shows that

$$
\begin{array} { r } { \mathbb { E } _ { u \sim \mathcal { N } ( 0 , \Sigma ) } [ G ( x ; u ) ] = \Sigma \nabla f _ { \Sigma } ^ { \tau } ( x ) \xrightarrow { \tau  0 } \Sigma \nabla f ( x ) = \gamma \nabla f ( x ) + ( 1 - \gamma ) \frac { d } { k } P _ { S } \nabla f ( x ) . } \end{array}
$$

Thus the estimator no longer targets $\nabla f ( x )$ itself, but applies different gains to its $s$ and $\mathcal { S } ^ { \perp }$ components, amplifying the former by a factor of order $d / k .$ . This creates a fundamental step-size dilemma: a step size chosen to remain stable along $s$ is overly conservative on $\mathcal { S } ^ { \perp }$ , while one tuned for $\boldsymbol { S ^ { \perp } }$ is too aggressive along S. Anisotropic exploration does not simply reduce variance; it changes the optimization geometry itself, making it ill-suited as a robust mechanism for incorporating directional hints whose alignment with the gradient may vary across iterations.

Prior-Guided Random Gradient-Free methods. [CWZ21] propose a related approach in a more restrictive setting. Their guidance takes the form of a single vector $p \in \mathbb { R } ^ { d }$ , or equivalently the one-dimensional subspace $S = \operatorname { s p a n } \{ p \}$ , and they assume access to directional derivatives $\langle \nabla f ( x ) , u \rangle$ rather than zeroth-order feedback $f ( x )$ . Their estimator with batch size $q \in \mathbb { N }$ can be written as

$$
G _ { \mathrm { P R G F } } ( \boldsymbol { x } , \boldsymbol { S } ; \{ u _ { i } \} _ { i = 1 } ^ { q } ) = P s \nabla f ( \boldsymbol { x } ) + \sum _ { i = 1 } ^ { q } \langle \nabla f ( \boldsymbol { x } ) , u _ { i } \rangle u _ { i } ,
$$

where $u _ { 1 } , \ldots , u _ { q }$ follow a Haar-uniform orthonormal q-frame in $\boldsymbol { S ^ { \perp } }$ . This estimator is generally biased:

$$
\begin{array} { r } { \mathbb { E } _ { U } \left[ G _ { \mathrm { P R G F } } ( x , \mathcal { S } ; U ) \right] = P _ { \mathcal { S } } \nabla f ( x ) + \frac { q } { d - 1 } P _ { \mathcal { S } } ^ { \perp } \nabla f ( x ) = \frac { q } { d - 1 } \nabla f ( x ) + \left( 1 - \frac { q } { d - 1 } \right) P _ { \mathcal { S } } \nabla f ( x ) . } \end{array}
$$

As with GES, the estimator applies different effective gains to the $s$ and $\mathcal { S } ^ { \perp }$ gradient components.

Summary. Both GES and PRGF incorporate directional guidance by modifying the search distribution, which unavoidably biases the gradient estimator. The bias introduces two coupled scales in the update — one along S and one along $s ^ { \perp } -$ that cannot be simultaneously controlled by a single step size. By contrast, the control-variate estimator in CV-ZOD remains unbiased for all reference vectors, so the step size controls a single scale (the residual variance), enabling the adaptive tuning developed in Section 4.

## B Ablations of the Discount Factor and Query Allocation

Using the same fluid setup as in Section 5.1, we vary ${ \gamma _ { \mathrm { d i s c } } \in \{ 0 , 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 \} }$ and compare query partitions $1 2 / 2 / 2 , 7 / 7 / 2 .$ , and $2 / 1 2 / 2$

Table 1: Final fluid objective: mean ± sample standard deviation over 15 paired seeds. Columns give query allocations (update / norm / projection). Lower is better.
<table><tr><td rowspan="2"> $\gamma _ { \mathrm { d i s c } }$ </td><td colspan="3">CV-ZOD</td><td>ZOD</td></tr><tr><td> $\overline { { 1 2 / 2 / 2 } }$ </td><td> $\overline { { 7 / 7 / 2 } }$ </td><td> $\overline { { 2 / 1 2 / 2 } }$ </td><td> $\overline { { 1 6 / 0 / 0 } }$ </td></tr><tr><td>0</td><td> $0 . 0 7 6 4 \pm 0 . 0 2 1 9$ </td><td> $0 . 0 6 8 6 \pm 0 . 0 2 7 4$ </td><td> $0 . 0 8 4 6 \pm 0 . 0 0 0 3$ </td><td> $0 . 0 2 5 8 \pm 0 . 0 0 4 0$ </td></tr><tr><td>0.6</td><td> $0 . 0 1 4 9 \pm 0 . 0 0 2 9$ </td><td> $0 . 0 1 6 9 \pm 0 . 0 0 4 1$ </td><td> $0 . 0 7 9 7 \pm 0 . 0 1 4 9$ </td><td> $0 . 0 2 5 8 \pm 0 . 0 0 4 0$ </td></tr><tr><td>0.7</td><td> $\mathbf { 0 . 0 1 3 6 \pm 0 . 0 0 2 5 }$ </td><td> $0 . 0 1 5 3 \pm 0 . 0 0 2 8$ </td><td> $0 . 0 7 9 6 \pm 0 . 0 1 3 7$ </td><td> $0 . 0 2 5 8 \pm 0 . 0 0 4 0$ </td></tr><tr><td>0.8</td><td> $0 . 0 1 5 3 \pm 0 . 0 0 3 7$ </td><td> $0 . 0 1 4 9 \pm 0 . 0 0 3 0$ </td><td> $0 . 0 7 4 9 \pm 0 . 0 1 9 7$ </td><td> $0 . 0 2 5 8 \pm 0 . 0 0 4 0$ </td></tr><tr><td>0.9</td><td> $0 . 0 1 7 6 \pm 0 . 0 0 6 3$ </td><td> $0 . 0 1 5 0 \pm 0 . 0 0 2 2$ </td><td> $0 . 0 7 5 9 \pm 0 . 0 1 7 5$ </td><td> $0 . 0 2 5 8 \pm 0 . 0 0 4 0$ </td></tr></table>

Table 1 shows that temporal smoothing has its largest benefit when enough queries remain for the control-variate update. With the $1 2 / 2 / 2$ allocation, setting $\gamma _ { \mathrm { d i s c } } = 0 . 7$ reduces the mean final objective from 0.07635 to 0.01363; all 15 runs finish below 0.020. For both $1 2 / 2 / 2$ and $7 / 7 / 2 .$ , every displayed positive discount factor gives a lower mean final objective than ZOD. Increasing the norm-estimation allocation to 12 leaves only two update directions and gives much poorer results throughout the sweep. Among the displayed configurations, the $1 2 / 2 / 2$ allocation with $\gamma _ { \mathrm { d i s c } } = 0 . 7$ has both the lowest mean final objective and the earliest mean crossing of 0.020, at update 61. We use this setting for the fluid comparison in Figure 2; selection and evaluation use the same 15 seeds.

## C Proofs for Section 3: CV-ZOD Framework

This section contains the proofs of Lemma 3.3 and Theorem 3.4. The proof of Theorem 3.4 relies on two concentration results for self-normalized martingales (Theorem C.1 and Corollary C.2), which adapt the exponential inequalities of [FGL15] to our setting via a peeling argument; these are stated and proved after the main argument.

## C.1 Proof of Lemma 3.3

Lemma 3.3 (Restated) For all $x , m \in \mathbb { R } ^ { d } ,$ , it holds that $\begin{array} { r } { \mathbb E _ { u \sim \mathcal { N } ( 0 , I _ { d } ) } \left[ G ( x , m ; u ) \right] = \nabla f ^ { \tau } ( x ) } \end{array}$ . Moreover,

$$
\begin{array} { r } { {  { \mathbb E } } _ { u \sim \mathcal { N } ( 0 , I _ { d } ) } \left[ \| G ( x , m ; u ) - \nabla f ( x ) \| ^ { 2 } \right] \le 2 ( d + 1 ) \| m - \nabla f ( x ) \| ^ { 2 } + 8 \tau ^ { 2 } L ^ { 2 } d ^ { 3 } } \end{array}
$$

Proof Fix $x , m \in \mathbb { R } ^ { d }$ . The unbiasedness follows immediately from Lemma 3.2 and Fact E.1:

$$
\begin{array} { r } { \mathbb { E } _ { u } \left[ G ( x , m ; u ) \right] = \mathbb { E } _ { u } \left[ G ( x ; u ) \right] + \mathbb { E } _ { u } \left[ ( I _ { d } - u u ^ { \top } ) m \right] = \nabla f ^ { \tau } ( x ) + 0 . } \end{array}
$$

To bound the mean-squared estimation error, we write

$$
G ( x , m ; u ) - \nabla f ( x ) = \left( I _ { d } - u u ^ { \top } \right) ( m - \nabla f ( x ) ) + \frac { f ( x + \tau u ) - f ( x ) - \left. \nabla f ( x ) , \tau u \right. } { \tau } u .
$$

Consequently, we have

$$
\begin{array} { r l } & { \mathbb { E } _ { u } \left[ \| G ( x , m ; u ) - \nabla f ( x ) \| ^ { 2 } \right] \leq 2 ( m - \nabla f ( x ) ) ^ { \top } \mathbb { E } _ { u } \left[ ( I _ { d } - u u ^ { \top } ) ^ { 2 } \right] ( m - \nabla f ( x ) ) } \\ & { \phantom { \qquad \quad } + 2 \mathbb { E } _ { u } \left[ \left( \frac { f ( x + \tau u ) - f ( x ) - \langle \nabla f ( x ) , \tau u \rangle } { \tau } \right) ^ { 2 } \| u \| ^ { 2 } \right] } \\ & { \phantom { \qquad \quad } \stackrel { { ( ) } } { \leq } 2 ( d + 1 ) \| m - \nabla f ( x ) \| ^ { 2 } + \frac { 1 } { 2 } \tau ^ { 2 } L ^ { 2 } \mathbb { E } [ \| u \| ^ { 6 } ] } \\ & { \phantom { \qquad \quad } \stackrel { { ( ) } } { \leq } 2 ( d + 1 ) \| m - \nabla f ( x ) \| ^ { 2 } + 8 \tau ^ { 2 } L ^ { 2 } d ^ { 3 } , } \end{array}
$$

where (a) follows from L-smoothness and Fact E.1 and (b) from Fact E.2.

## C.2 Proof of Theorem 3.4

Theorem 3.4 (Restated) Suppose Algorithm 1 uses afresh standard Gaussian direction $u _ { t } ,$ conditionally on all information available before its draw,for every $t \in [ T ]$ , and that $\begin{array} { r } { \operatorname* { m a x } _ { t \in [ T ] } \| m _ { t } \| \leq 2 B } \end{array}$ almost surely. Then there

exist absolute constants $C _ { 0 } , C > 0$ such that the following holds. For any $\delta \in ( 0 , 1 )$ , set $\iota : = \log ( 2 T / \delta )$ , and suppose $d \geq \iota$ and $\begin{array} { r } { 0 < \alpha _ { t } \le \frac { 1 } { C _ { 0 } L \iota } } \end{array}$ for all $t \in [ T ]$ . Then, for $\gamma = C _ { 0 } \iota$ , with probability at least $1 - \delta ,$

$$
\begin{array} { r } { \frac { 1 } { L } \sum _ { t = 1 } ^ { T } \alpha _ { t } \| \nabla f ( x _ { t } ) \| ^ { 2 } \leq C \left( D _ { f } ^ { 2 } + \frac { B ^ { 2 } } { L ^ { 2 } } + \tau ^ { 2 } T d ^ { 3 } + \mathcal { B } _ { T } ^ { \gamma } \iota \right) , } \end{array}
$$

where $\begin{array} { r } { D _ { f } ^ { 2 } : = \frac { 2 } { L } \left( f ( x _ { 1 } ) - f ^ { * } \right) } \end{array}$ . Moreover, $i f d \geq \log ( 2 T ^ { 2 } ) , \gamma = C _ { 0 } \log ( 2 T ^ { 2 } ) , 0 < \alpha _ { t } \leq 1 / ( \gamma L )$ for all $t \in [ T ]$ and $\begin{array} { r } { \tau \leq \frac { B / L } { \sqrt { T d ^ { 3 } } } } \end{array}$ , then x satisfies

$$
\begin{array} { r } { \frac { 1 } { L } \mathbb { E } \bigl [ \| \nabla f ( \overline { { x } } ) \| ^ { 2 } \bigr ] \leq C \mathbb { E } \biggl [ \frac { D _ { f } ^ { 2 } + B ^ { 2 } / L ^ { 2 } + B _ { T } ^ { \gamma } \log ( 2 T ^ { 2 } ) } { A _ { T } } \biggr ] , } \end{array}
$$

where $\textstyle A _ { T } : = \sum _ { t = 1 } ^ { T } \alpha _ { t }$ denotes the cumulative step size.

Proof For each $t \in [ T ]$ , let $\mathcal { F } _ { t }$ contain all information available before iteration t, including the current iterate and hint, and let $\mathcal { F } _ { T + 1 }$ contain all information after the final iteration. Let $\mathcal { G } _ { t }$ additionally contain all information available when $\alpha _ { t }$ and $m _ { t }$ have been chosen, just before drawing $u _ { t }$ . These sigma-algebras are nested as $\mathcal { F } _ { t } \subseteq C$ $\mathcal G _ { t } \subseteq \mathcal F _ { t + 1 }$ . Throughout, set $g _ { t } : = \nabla f ^ { \tau } ( x _ { t } ) , w _ { t } : = m _ { t } - \nabla f ( x _ { t } ) , \beta _ { t } : = \alpha _ { t } - L \alpha _ { t } ^ { 2 } , G _ { t } : = G ( x _ { t } , m _ { t } ; u _ { t } )$ , and $\Delta _ { t } : = G _ { t } - g _ { t }$ . Thus $x _ { t } , \alpha _ { t } ,$ , and $m _ { t }$ are G -measurable, while $u _ { t }$ is conditionally standard Gaussian given $\mathcal { G } _ { t }$ Hence Lemma 3.3 gives $\mathbb { E } [ \Delta _ { t } \ | \ \mathcal { G } _ { t } ] = 0$ . Since $\iota \geq \log 2$ and we take $C _ { 0 } \geq 4 .$ , the cap $\begin{array} { r } { \alpha _ { t } \le \frac { 1 } { C _ { 0 } L \iota } \le \frac { 1 } { 2 L } } \end{array}$ implies $\beta _ { t } \in [ \frac { 1 } { 2 } \alpha _ { t } , \alpha _ { t } ]$ ]. We also write

$$
\begin{array} { r } { S _ { T } : = \sum _ { t = 1 } ^ { T } \alpha _ { t } \| \nabla f ( x _ { t } ) \| ^ { 2 } , \qquad V _ { T } : = d \sum _ { t = 1 } ^ { T } \alpha _ { t } ^ { 2 } \| w _ { t } \| ^ { 2 } , \qquad H : = D _ { f } ^ { 2 } + B ^ { 2 } / L ^ { 2 } . } \end{array}
$$

We record two approximation bounds from Lemma 3.1 used repeatedly:

$$
\operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \| \nabla f ^ { \tau } ( x ) - \nabla f ( x ) \| \leq c _ { 0 } \tau L d ^ { 3 / 2 } ,\tag{7}
$$

$$
\begin{array} { r } { f ^ { \tau } ( x _ { 1 } ) - f ^ { * } \leq \frac { L } { 2 } D _ { f } ^ { 2 } + \frac { \tau ^ { 2 } L d } { 2 } . } \end{array}\tag{8}
$$

Step 1: Descent inequality. By L-smoothness of $f ^ { \tau }$ and $G _ { t } = g _ { t } + \Delta _ { t }$

$$
\begin{array} { r l } & { f ^ { \tau } ( x _ { t + 1 } ) \leq f ^ { \tau } ( x _ { t } ) - \alpha _ { t } \langle g _ { t } , G _ { t } \rangle + \frac { L \alpha _ { t } ^ { 2 } } { 2 } \| G _ { t } \| ^ { 2 } } \\ & { \qquad = f ^ { \tau } ( x _ { t } ) - \underbrace { ( \alpha _ { t } - \frac { L \alpha _ { t } ^ { 2 } } { 2 } ) } _ { \geq \alpha _ { t } / 2 } \| g _ { t } \| ^ { 2 } - \underbrace { ( \alpha _ { t } - L \alpha _ { t } ^ { 2 } ) } _ { = \beta _ { t } } \langle g _ { t } , \Delta _ { t } \rangle + \frac { L \alpha _ { t } ^ { 2 } } { 2 } \| \Delta _ { t } \| ^ { 2 } . } \end{array}
$$

Telescoping over $t = 1 , \dots , T$ and using $f ^ { \tau } ( x _ { T + 1 } ) \geq f ^ { * }$ with (8),

$$
\begin{array} { r } { \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \alpha _ { t } \| g _ { t } \| ^ { 2 } \leq \frac { L } { 2 } D _ { f } ^ { 2 } + \frac { \tau ^ { 2 } L d } { 2 } - M _ { T } + \frac { L } { 2 } Q _ { T } , } \end{array}\tag{9}
$$

where $\begin{array} { r } { M _ { T } : = \sum _ { t = 1 } ^ { T } \beta _ { t } \langle g _ { t } , \Delta _ { t } \rangle } \end{array}$ and $\begin{array} { r } { Q _ { T } : = \sum _ { t = 1 } ^ { T } \alpha _ { t } ^ { 2 } \| \Delta _ { t } \| ^ { 2 } } \end{array}$

## Step 2: Pathwise bound on $Q _ { T }$ . The proof of Lemma 3.3 gives

$$
\begin{array} { r } { G _ { t } - \nabla f ( x _ { t } ) = \left( I _ { d } - u _ { t } \boldsymbol { u } _ { t } ^ { \top } \right) \boldsymbol { w } _ { t } + \epsilon _ { t } \boldsymbol { u } _ { t } , \qquad | \epsilon _ { t } | \leq \frac { L \tau } { 2 } \| \boldsymbol { u } _ { t } \| ^ { 2 } . } \end{array}
$$

Since $\Delta _ { t } = \left( G _ { t } - \nabla f ( x _ { t } ) \right) + \left( \nabla f ( x _ { t } ) - g _ { t } \right)$ , we have

$$
\begin{array} { r } { \| \Delta _ { t } \| ^ { 2 } \leq 2 \| \big ( I _ { d } - u _ { t } u _ { t } ^ { \top } \big ) w _ { t } + \epsilon _ { t } u _ { t } \| ^ { 2 } + 2 c _ { 0 } ^ { 2 } \tau ^ { 2 } L ^ { 2 } d ^ { 3 } . } \end{array}\tag{10}
$$

We establish pathwise control on $u _ { t }$ . Since $\lVert u _ { t } \rVert ^ { 2 } \sim \chi _ { d } ^ { 2 }$ conditioned on $\mathcal { G } _ { t }$ , the Laurent–Massart inequality (Lemma E.3) gives $\operatorname* { P r } ( \| u _ { t } \| ^ { 2 } > d + 2 \sqrt { d \ell ^ { \prime } } + 2 \ell ^ { \prime } ) \le e ^ { - \ell ^ { \prime } }$ for all $\ell ^ { \prime } > 0$ . Similarly, $\langle w _ { t } , u _ { t } \rangle \mid \mathcal { G } _ { t } \sim \mathcal { N } ( 0 , \| w _ { t } \| ^ { 2 } )$ so the standard Gaussian tail yields $\operatorname* { P r } ( | \langle w _ { t } , u _ { t } \rangle | > \| w _ { t } \| \sqrt { 2 \ell ^ { \prime } } ) \leq 2 e ^ { - \ell ^ { \prime } }$ . Set $\ell : = \log ( 1 2 T / \delta ) \leq 4 \iota$ . A union bound over $t \in [ T ]$ gives total failure probability at most $3 T e ^ { - \ell } = \delta / 4$ . Thus the following event E holds with probability at least $1 - \delta / 4 \mathrm { : }$ for all $t \in [ T ]$ simultaneously,

$$
\| u _ { t } \| ^ { 2 } \leq 4 ( d + \ell ) \qquad \mathrm { a n d } \qquad \langle w _ { t } , u _ { t } \rangle ^ { 2 } \leq 2 \| w _ { t } \| ^ { 2 } \ell .\tag{11}
$$

We work on $\mathcal { E }$ for the remainder of the step.

Bounding the two components. Expanding and applying (11):

$$
\begin{array} { r l } & { \| ( I _ { d } - u _ { t } u _ { t } ^ { \top } ) w _ { t } \| ^ { 2 } = \| w _ { t } \| ^ { 2 } - 2 \langle w _ { t } , u _ { t } \rangle ^ { 2 } + \langle w _ { t } , u _ { t } \rangle ^ { 2 } \| u _ { t } \| ^ { 2 } } \\ & { \qquad \leq \| w _ { t } \| ^ { 2 } + 8 \| w _ { t } \| ^ { 2 } \ell ( d + \ell ) } \\ & { \qquad \leq c \| w _ { t } \| ^ { 2 } d \iota , } \end{array}
$$

and

$$
\begin{array} { r } { \epsilon _ { t } ^ { 2 } \| u _ { t } \| ^ { 2 } \leq \frac { L ^ { 2 } \tau ^ { 2 } } { 4 } \| u _ { t } \| ^ { 6 } \leq c L ^ { 2 } \tau ^ { 2 } d ^ { 3 } . } \end{array}
$$

Substituting into (10),

$$
\| \Delta _ { t } \| ^ { 2 } \leq c d \iota \| w _ { t } \| ^ { 2 } + c \tau ^ { 2 } L ^ { 2 } d ^ { 3 } .\tag{12}
$$

Multiplying by $\alpha _ { t } ^ { 2 }$ and summing:

$$
\begin{array} { r } { Q _ { T } \leq c d \iota \sum _ { t = 1 } ^ { T } \alpha _ { t } ^ { 2 } \| w _ { t } \| ^ { 2 } + c \tau ^ { 2 } L ^ { 2 } d ^ { 3 } \sum _ { t = 1 } ^ { T } \alpha _ { t } ^ { 2 } . } \end{array}
$$

Since $\textstyle \sum _ { t } \alpha _ { t } ^ { 2 } \leq T / L ^ { 2 }$ , this gives

$$
Q _ { T } \leq c \iota V _ { T } + c \tau ^ { 2 } T d ^ { 3 } .\tag{13}
$$

We retain the nonnegative quantity $V _ { T }$ until the final step, where we use the identity $\imath B _ { T } ^ { \gamma } = \imath V _ { T } - S _ { T } / ( C _ { 0 } L )$

Step 3: Martingale concentration. Dominant martingale. Define

$$
\xi _ { t } : = \beta _ { t } \left( \langle \nabla f ( x _ { t } ) , w _ { t } \rangle - \langle \nabla f ( x _ { t } ) , u _ { t } \rangle \langle w _ { t } , u _ { t } \rangle \right) .
$$

Since $\beta _ { t } , \nabla f ( x _ { t } ) , w _ { t }$ are $\mathcal { G } _ { t }$ -measurable and $\mathbb { E } [ \langle \nabla f ( x _ { t } ) , u _ { t } \rangle \langle w _ { t } , u _ { t } \rangle \mid \mathcal { G } _ { t } ] = \langle \nabla f ( x _ { t } ) , w _ { t } \rangle$ , each $\xi _ { t }$ is conditionally mean zero. It is measurable with respect to $\mathcal { F } _ { t + 1 } ,$ so the partial sums of $\widetilde M _ { T } : = \textstyle \sum _ { t = 1 } ^ { T } \xi _ { t }$ form a martingale with respect to the post-draw filtration $( \mathcal G _ { t + 1 } ) _ { t = 0 } ^ { T } ,$ , where $\mathcal G _ { T + 1 } : = \mathcal F _ { T + 1 }$ . By Lemma E.4 with $a = \nabla f ( x _ { t } )$ and $\Sigma = w _ { t } w _ { t } ^ { \top }$

$$
\begin{array} { r } { \mathbb { E } [ \xi _ { t } ^ { 2 } \mid \mathcal { G } _ { t } ] = \beta _ { t } ^ { 2 } \big ( \| \nabla f ( x _ { t } ) \| ^ { 2 } \| w _ { t } \| ^ { 2 } + \langle \nabla f ( x _ { t } ) , w _ { t } \rangle ^ { 2 } \big ) < \infty . } \end{array}
$$

MGF condition. Conditioned on $\mathcal { G } _ { t } , \xi _ { t }$ is a centered degree-two polynomial in $u _ { t } \sim \mathcal { N } ( 0 , I _ { d } )$ . By the Hanson– Wright inequality, there exists an absolute constant $c _ { 1 } > 0$ such that

$$
\begin{array} { r l r } { \log \mathbb { E } \big [ e ^ { \lambda \xi _ { t } } \mid \mathcal { G } _ { t } \big ] \leq c _ { 1 } ^ { 2 } \lambda ^ { 2 } \beta _ { t } ^ { 2 } \| \nabla f ( x _ { t } ) \| ^ { 2 } \| w _ { t } \| ^ { 2 } } & { } & { \mathrm { f o r } \left| \lambda \right| \leq \frac { 1 } { c _ { 1 } \beta _ { t } \| \nabla f ( x _ { t } ) \| \| w _ { t } \| } . } \end{array}
$$

Since $e ^ { x } \leq 1 +$ 2x for $x \in [ 0 , 1 ]$ and $\mathbb { E } [ \xi _ { t } ^ { 2 } \mid \mathcal { G } _ { t } ] \ge \beta _ { t } ^ { 2 } \| \nabla f ( x _ { t } ) \| ^ { 2 } \| w _ { t } \| ^ { 2 }$ by Lemma $\mathrm { E . 4 . }$

$$
\begin{array} { r } { \mathbb { E } \big [ e ^ { \lambda \xi _ { t } } \mid \mathcal { G } _ { t } \big ] \leq 1 + 2 c _ { 1 } ^ { 2 } \lambda ^ { 2 } \ \mathbb { E } [ \xi _ { t } ^ { 2 } \mid \mathcal { G } _ { t } ] \qquad \mathrm { f o r } \left| \lambda \right| \leq \frac { 1 } { c _ { 1 } \beta _ { t } \| \nabla f ( x _ { t } ) \| \| w _ { t } \| } . } \end{array}
$$

If $\beta _ { t } \| \nabla f ( x _ { t } ) \| \| w _ { t } \| = 0$ , then $\xi _ { t } = 0$ and the MGF bound holds for every λ. Otherwise the displayed range applies. The boundedness assumptions give $\lVert w _ { t } \rVert \leq 3 B$ and hence $c _ { 1 } \beta _ { t } \| \nabla f ( x _ { t } ) \| \| w _ { t } \| \leq 3 c _ { 1 } B ^ { 2 } / ( C _ { 0 } L \iota )$ . Set

$$
\varepsilon : = \frac { \operatorname* { m a x } \{ 3 c _ { 1 } , 1 8 \} B ^ { 2 } } { C _ { 0 } L \iota } .
$$

The hypothesis of Theorem C.1 then holds with this ε and its constant $c _ { 0 } = 4 c _ { 1 } ^ { 2 }$

Bounding $\widetilde { M } _ { T }$ . Using $\beta _ { t } \le \alpha _ { t } , \| w _ { t } \| \le 3 B$ , and $\alpha _ { t } \leq 1 / ( C _ { 0 } L \iota )$

$$
\begin{array} { r } { \mathbb { E } [ \xi _ { t } ^ { 2 } \ | \ { \mathcal G } _ { t } ] \le 1 8 \alpha _ { t } ^ { 2 } \| \nabla f ( x _ { t } ) \| ^ { 2 } B ^ { 2 } \le \frac { 1 8 B ^ { 2 } } { C _ { 0 } L L } \cdot \alpha _ { t } \| \nabla f ( x _ { t } ) \| ^ { 2 } \le \varepsilon \cdot \alpha _ { t } \| \nabla f ( x _ { t } ) \| ^ { 2 } . } \end{array}
$$

Setting $r _ { t } : = \alpha _ { t } \| \nabla f ( x _ { t } ) \| ^ { 2 }$ , Corollary C.2 gives, with probability at least $1 - \delta / 2$

$$
\begin{array} { r } { | \widetilde { M } _ { T } | \leq \frac { 1 } { 8 } \sum _ { t = 1 } ^ { T } \alpha _ { t } \| \nabla f ( x _ { t } ) \| ^ { 2 } + \frac { c B ^ { 2 } } { L } , } \end{array}\tag{14}
$$

where we used ε log $\begin{array} { r } { \mathfrak { s } ( 2 e / \delta ) \le c B ^ { 2 } / L . } \end{array}$ , since $\log ( 2 e / \delta ) \leq c \iota$

Smoothing residual. Since $\Delta _ { t } = ( I _ { d } - u _ { t } u _ { t } ^ { \top } ) w _ { t } + \epsilon _ { t } u _ { t } + ( \nabla f ( x _ { t } ) - g _ { t } ) ,$

$$
\begin{array} { r } { \beta _ { t } \langle g _ { t } , \Delta _ { t } \rangle - \xi _ { t } = \beta _ { t } \langle g _ { t } - \nabla f ( x _ { t } ) , \Delta _ { t } \rangle + \beta _ { t } \langle \nabla f ( x _ { t } ) , ( \nabla f ( x _ { t } ) - g _ { t } ) + \epsilon _ { t } u _ { t } \rangle . } \end{array}
$$

On E, using (7), (12), $| \epsilon _ { t } | \leq c L \tau ( d + \iota ) , \| u _ { t } \| \leq c \sqrt { d + \iota } ,$ and $\beta _ { t } \le \alpha _ { t } \le 1 / ( C _ { 0 } L \iota )$

$$
\begin{array} { r } { | \beta _ { t } \langle g _ { t } , \Delta _ { t } \rangle - \xi _ { t } | \leq c \alpha _ { t } \tau L d ^ { 3 / 2 } \| \Delta _ { t } \| + c \alpha _ { t } \| \nabla f ( x _ { t } ) \| \tau L ( d + \iota ) ^ { 3 / 2 } . } \end{array}
$$

Applying Young’s inequality to the first term with quadratic contribution $L \alpha _ { t } ^ { 2 } \| \Delta _ { t } \| ^ { 2 } / 2$ and to the second with contribution $\alpha _ { t } \| \nabla f ( x _ { t } ) \| ^ { 2 } / 3 2$ , then summing and using $\textstyle \sum _ { t } { \alpha _ { t } \leq T / L }$ , gives

$$
\begin{array} { r } { | M _ { T } - \widetilde { M } _ { T } | \leq \frac { L } { 2 } Q _ { T } + \frac { 1 } { 3 2 } S _ { T } + c \tau ^ { 2 } T d ^ { 3 } L . } \end{array}\tag{15}
$$

Combining (14) and (15):

$$
\begin{array} { r } { | M _ { T } | \leq \frac { 5 } { 3 2 } S _ { T } + \frac { c B ^ { 2 } } { L } + \frac { L } { 2 } Q _ { T } + c \tau ^ { 2 } T d ^ { 3 } L . } \end{array}\tag{16}
$$

Step 4: Combining. Intersecting E with the martingale event gives an event H with probability at least $1 - 3 \delta / 4 \geq$ $1 - \delta .$ . On H, substituting (16) into (9) and using $- M _ { T } \le | M _ { T } |$ gives

$$
\begin{array} { r } { \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \alpha _ { t } \| g _ { t } \| ^ { 2 } \leq \frac { L } { 2 } D _ { f } ^ { 2 } + \frac { c B ^ { 2 } } { L } + \frac { 5 } { 3 2 } S _ { T } + L Q _ { T } + c \tau ^ { 2 } T d ^ { 3 } L . } \end{array}
$$

The gradient approximation (7) and $\textstyle \sum _ { t } \alpha _ { t } \leq T / L$ imply

$$
\begin{array} { r } { \frac 1 2 \sum _ { t = 1 } ^ { T } \alpha _ { t } \| g _ { t } \| ^ { 2 } \geq \frac { 1 } { 4 } S _ { T } - c \tau ^ { 2 } T d ^ { 3 } L . } \end{array}
$$

Consequently,

$$
\begin{array} { r } { \frac { 3 } { 3 2 } S _ { T } \leq \frac { L } { 2 } D _ { f } ^ { 2 } + \frac { c B ^ { 2 } } { L } + L Q _ { T } + c \tau ^ { 2 } T d ^ { 3 } L . } \end{array}
$$

Applying (13) and dividing by L therefore gives an absolute constant $K \geq 1$ , independent of $C _ { 0 }$ once $C _ { 0 } \geq 4 .$ such that

$$
\begin{array} { r } { \frac { S _ { T } } { L } \leq K \left( H + \tau ^ { 2 } T d ^ { 3 } + \iota V _ { T } \right) . } \end{array}\tag{17}
$$

Set $K _ { 2 } : = 2 K + 1$ , and choose $C _ { 0 } \geq \operatorname* { m a x } \{ 4 , 2 K _ { 2 } \}$ and $C : = 2 K _ { 2 }$ . The balance identity and (17) give, on $\mathcal { H } .$

$$
\begin{array} { r l } & { H + \tau ^ { 2 } T d ^ { 3 } + \iota B _ { T } ^ { \gamma } = H + \tau ^ { 2 } T d ^ { 3 } + \iota V _ { T } - \frac { S _ { T } } { C _ { 0 } L } } \\ & { \qquad \geq \left( 1 - K / C _ { 0 } \right) \left( H + \tau ^ { 2 } T d ^ { 3 } + \iota V _ { T } \right) } \\ & { \qquad \geq \frac 1 2 \left( H + \tau ^ { 2 } T d ^ { 3 } + \iota V _ { T } \right) . } \end{array}
$$

Combining this inequality with (17) proves the high-probability conclusion with the chosen $C .$

For the expected conclusion, first suppose $T \geq 2$ and apply the preceding argument with $\delta = 1 / T , \mathrm { { s o } } \iota = \log ( 2 T ^ { 2 } )$ Since the step sizes are positive, $A _ { T } > 0$ . Define

$$
Y : = \frac { S _ { T } } { L \mathcal { A } _ { T } } , \qquad X : = \frac { H + \iota V _ { T } } { \mathcal { A } _ { T } } , \qquad W : = \frac { H + \iota B _ { T } ^ { \gamma } } { \mathcal { A } _ { T } } = X - \frac { Y } { C _ { 0 } } .
$$

The output rule gives $\mathbb { E } Y = L ^ { - 1 } \mathbb { E } \| \nabla f ( \overline { { x } } ) \| ^ { 2 }$ . The smoothing condition implies $\tau ^ { 2 } T d ^ { 3 } \leq B ^ { 2 } / L ^ { 2 }$ , so (17) gives $Y \le 2 K X$ on $\mathcal { H } .$ . Moreover, $X \geq 0$ and $0 \le Y \le B ^ { 2 } / L$ everywhere. Taking expectations while retaining the nonnegative X, we obtain

$$
\mathbb { E } Y \le 2 K \mathbb { E } X + \frac { B ^ { 2 } } { L T } \le K _ { 2 } \mathbb { E } X .
$$

The last inequality uses $\mathcal { A } _ { T } \leq T / ( C _ { 0 } \iota L )$ , which implies $X \geq B ^ { 2 } / ( L ^ { 2 } A _ { T } ) \geq B ^ { 2 } / ( L T )$ . If E $X < \infty$ , then

$$
\begin{array} { r } { \mathbb { E } W = \mathbb { E } X - \frac { \mathbb { E } Y } { C _ { 0 } } \ge ( 1 - K _ { 2 } / C _ { 0 } ) \mathbb { E } X \ge \frac { 1 } { 2 } \mathbb { E } X . } \end{array}
$$

Thus $\mathbb { E } Y \le C \mathbb { E } W$ , as claimed. $\operatorname { I f } \operatorname { \mathbb { E } } X = \infty$ , the same conclusion holds in the extended sense because $Y$ is bounded and $W \ge - B ^ { 2 } / ( C _ { 0 } L )$

Finally, when $T = 1$ , the output is $x _ { 1 }$ . With $\iota = \log 2$ and the same definitions of $Y , X , W$ , positivity and the step-size cap give

$$
W \geq \frac { B ^ { 2 } } { L ^ { 2 } \alpha _ { 1 } } - \frac { Y } { C _ { 0 } } \geq \left( C _ { 0 } \log 2 - \frac { 1 } { C _ { 0 } } \right) \frac { B ^ { 2 } } { L } \geq Y .
$$

This proves the expected conclusion for every $T \in \mathbb { N }$ and completes the proof.

Theorem C.1 Let $( S _ { k } , \mathcal { F } _ { k } ) _ { k = 0 , . . . , n }$ be a martingale with $S _ { 0 } = 0$ and differences $\xi _ { i } = S _ { i } - S _ { i - 1 }$ . Let $c _ { 0 } > 0 .$ Suppose there exists $\varepsilon > 0$ such that for all $i \in [ n ]$ and all $| \lambda | \leq 1 / \varepsilon$

$$
\begin{array} { r } { \mathbb { E } \left[ e ^ { \lambda \xi _ { i } } \mid \mathcal { F } _ { i - 1 } \right] \leq 1 + \frac { c _ { 0 } \lambda ^ { 2 } } { 2 } \ \mathbb { E } \left[ \xi _ { i } ^ { 2 } \mid \mathcal { F } _ { i - 1 } \right] . } \end{array}
$$

Then for all $x , w > 0 ,$

$$
\begin{array} { r } { \operatorname* { P r } ( S _ { n } \geq x \ a n d \langle S \rangle _ { n } \leq w ) \leq \exp \left( - \frac { x ^ { 2 } } { 2 ( c _ { 0 } w + \varepsilon x ) } \right) , } \end{array}
$$

where $\begin{array} { r } { \langle S \rangle _ { n } = \sum _ { i = 1 } ^ { n } \mathbb { E } [ \xi _ { i } ^ { 2 } \mid \mathcal { F } _ { i - 1 } ] . } \end{array}$

Proof We apply Theorem 2.1 of [FGL15] with $\begin{array} { r } { g ( \lambda ) = 0 , f ( \lambda ) = \frac { c _ { 0 } \lambda ^ { 2 } } { 2 } , V _ { i - 1 } = \mathbb { E } [ \xi _ { i } ^ { 2 } \mid \mathcal { F } _ { i - 1 } ] } \end{array}$ , and $v  \infty$ . Their bound (2.3) gives, for any $\lambda \in ( 0 , 1 / \varepsilon )$

$$
\begin{array} { r } { \operatorname* { P r } ( S _ { n } \ge x \mathrm { ~ a n d ~ } \langle S \rangle _ { n } \le w ) \le \exp \Bigl ( - \lambda x + \frac { c _ { 0 } \lambda ^ { 2 } } { 2 } w \Bigr ) . } \end{array}
$$

Substituting $\begin{array} { r } { \lambda = \frac { x } { c _ { 0 } w + \varepsilon x } \in ( 0 , 1 / \varepsilon ) } \end{array}$

$$
\begin{array} { l } { \displaystyle - \lambda x + \frac { c _ { 0 } \lambda ^ { 2 } } { 2 } w = - \frac { x ^ { 2 } } { c _ { 0 } w + \varepsilon x } + \frac { x ^ { 2 } } { ( c _ { 0 } w + \varepsilon x ) ^ { 2 } } \cdot \frac { c _ { 0 } w } { 2 } } \\ { \displaystyle \qquad \leq - \frac { x ^ { 2 } } { c _ { 0 } w + \varepsilon x } + \frac { x ^ { 2 } } { 2 ( c _ { 0 } w + \varepsilon x ) } = - \frac { x ^ { 2 } } { 2 ( c _ { 0 } w + \varepsilon x ) } , } \end{array}
$$

which yields the result.

Corollary C.2 Under the conditions of Theorem $C . l ,$ suppose further that there exist non-negative $\mathcal { F } _ { i - 1 } -$ measurable random variables $r _ { 1 } , \ldots , r _ { n }$ such that

$$
\mathbb { E } \left[ \xi _ { i } ^ { 2 } \mid { \mathcal { F } } _ { i - 1 } \right] \leq \varepsilon r _ { i } \qquad f o r a l l i \in [ n ] a l m o s t s u r e l y .
$$

Let $\textstyle R = \sum _ { i = 1 } ^ { n } r _ { i }$ . Then for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$

$$
| S _ { n } | \leq { \frac { R } { 8 } } + c \varepsilon \log ( e / \delta ) ,
$$

where $c > 0$ depends only on $c _ { 0 } .$

Proof It suffices to prove the upper-tail bound, since the lower-tail bound follows by applying the same argument to $- S _ { n }$ . For $j \geq 1$ , define

$$
E _ { j } = \{ 2 ^ { j - 1 } \varepsilon < R \leq 2 ^ { j } \varepsilon \} , \qquad x _ { j } = \frac { 2 ^ { j - 1 } \varepsilon } 8 + r , \qquad w _ { j } = \varepsilon ^ { 2 } 2 ^ { j } ,
$$

and also set $E _ { 0 } = \{ R \leq \varepsilon \} , x _ { 0 } = r$ , and $w _ { 0 } = \varepsilon ^ { 2 }$ . On $E _ { j }$ , we have $\langle S \rangle _ { n } \leq \varepsilon R \leq w _ { j }$ and $R / 8 + r \geq x _ { j }$ Therefore, by Theorem C.1,

$$
\operatorname* { P r } ( S _ { n } \ge R / 8 + r ) \le \sum _ { j \ge 0 } \exp \left( - \frac { x _ { j } ^ { 2 } } { 2 ( c _ { 0 } w _ { j } + \varepsilon x _ { j } ) } \right) .
$$

We now bound the summands. For $j \geq 1$ j ≥ , since $x _ { j } \geq 2 ^ { j } \varepsilon / 1 6 .$ we have

$$
c _ { 0 } w _ { j } = c _ { 0 } \varepsilon ^ { 2 } 2 ^ { j } \leq 1 6 c _ { 0 } \varepsilon x _ { j } .
$$

Hence, for a constant $C > 0$ depending only on $c _ { 0 }$ ,

$$
\frac { x _ { j } ^ { 2 } } { 2 ( c _ { 0 } w _ { j } + \varepsilon x _ { j } ) } \geq \frac { x _ { j } } { C \varepsilon } \geq \frac { 2 ^ { j - 1 } } { C } + \frac { r } { C \varepsilon } .
$$

For the term $j = 0$ , after increasing C if necessary, the same bound gives

$$
\frac { x _ { 0 } ^ { 2 } } { 2 ( c _ { 0 } w _ { 0 } + \varepsilon x _ { 0 } ) } = \frac { r ^ { 2 } } { 2 ( c _ { 0 } \varepsilon ^ { 2 } + \varepsilon r ) } \geq \frac { r } { C \varepsilon } ,
$$

provided $r \geq c _ { 0 } \varepsilon$ , which will be true for the choice of r below. Thus

$$
\begin{array} { r l } & { \operatorname* { P r } ( S _ { n } \geq R / 8 + r ) \leq \exp \left( - \frac { r } { C \varepsilon } \right) \left( 1 + \sum _ { j \geq 1 } \exp ( - \frac { 2 ^ { j - 1 } } { C } ) \right) } \\ & { \qquad \leq C ^ { \prime } \exp \left( - \frac { r } { C \varepsilon } \right) , } \end{array}
$$

where $C ^ { \prime } > 0$ depends only on $c _ { 0 }$ . Taking

$$
r = C \varepsilon \log ( 2 C ^ { \prime } / \delta )
$$

gives

$$
\operatorname* { P r } ( S _ { n } \geq R / 8 + r ) \leq \delta / 2 .
$$

The same argument applied $\mathrm { t o } - S _ { n }$ gives the lower tail, and a union bound yields

$$
\operatorname* { P r } ( | S _ { n } | \geq R / 8 + r ) \leq \delta .
$$

Finally, increasing the constant once more, $r \leq c \varepsilon \log ( e / \delta )$

## D Proof of Theorem 4.4

This section proves the adaptive convergence results of Section 4. We first prove Theorem 4.4 using Theorem 3.4 and the estimation primitives, then prove the supporting Lemmas 4.1, 4.2, and 4.3.

Theorem 4.4 (Restated) There exist absolute constants $C , c > 0$ such that, $i f d \geq \log ( 2 T ^ { 2 } )$ , Algorithm 1 is run with $m _ { t } = P _ { t } ^ { \tau } ( x _ { t } , S _ { t } )$ and step sizes α<sub>t</sub> from (6) with confidence parameter $\delta = 1 / ( 2 T ) , \gamma = C _ { 0 } \log ( 2 T ^ { 2 } )$ , and $N = \lceil C \log ( 2 T ) \rceil$ , and the smoothing radius satisfies $\begin{array} { r } { \tau \leq \frac { B / L } { \sqrt { T \left( d ^ { 3 } + d k \right) \log \left( 2 T ^ { 2 } \right) } } , } \end{array}$ , then

$$
\begin{array} { r } { \frac { 1 } { L } \mathbb { E } \big [ \| \nabla f ( \overline { { x } } ) \| ^ { 2 } \big ] \leq c \mathbb { E } \Big [ \frac { ( D _ { f } ^ { 2 } + B ^ { 2 } / L ^ { 2 } ) \log ( 2 T ) } { A _ { T } ^ { * } } \Big ] , } \end{array}
$$

where $\begin{array} { r } { \mathcal { A } _ { T } ^ { * } = \sum _ { t = 1 } ^ { T } \alpha _ { t } ^ { * } f o r \alpha _ { t } ^ { * } } \end{array}$ in (4).

Proof When $T = 1$ , the output is $x _ { 1 }$ and the conclusion follows directly from $\| \nabla f ( x _ { 1 } ) \| ^ { 2 } \leq B ^ { 2 } , \mathcal { A } _ { 1 } ^ { \ast } \leq 1 / L .$ , and $c \geq 1 / \log 2$ . We therefore assume $T \geq 2 . \mathrm { S e t } \delta : = 1 / T$ and $\iota : = \log ( 2 T / \delta ) = \log ( 2 T ^ { 2 } )$ , so that $\gamma = C _ { 0 } \iota$

Step 0: High-probability events. Choose the sample constant $C$ at least twice the corresponding constant in Lemma 4.3; then $N = \lceil C \log ( 2 T ) \rceil$ meets its sample requirement at failure probability $1 / ( 2 T )$ , since $\log ( 4 T ^ { 2 } ) =$ $2 \log ( 2 T )$ . Let $\mathcal { E } _ { 1 }$ be the event from Lemma 4.3 applied with failure probability $\delta / 2 = 1 / ( 2 T )$ , on which local optimality and balance control hold simultaneously for all $t \in [ T ]$ . By Lemma 4.1, deterministically,

$$
\| m _ { t } - P _ { S _ { t } } \nabla f ( x _ { t } ) \| ^ { 2 } \leq { \frac { k L ^ { 2 } \tau ^ { 2 } } { 4 } } \qquad { \mathrm { f o r ~ a l l ~ } } t \in [ T ] .
$$

The assumed upper bound on τ also ensures

$$
\Vert m _ { t } \Vert \leq \Vert P _ { S _ { t } } \nabla f ( x _ { t } ) \Vert + \frac { L \tau \sqrt { k } } { 2 } \leq 2 B ,
$$

so the reference-vector condition in Theorem 3.4 holds. Let ${ \mathcal { E } } _ { 2 }$ be the event from that theorem applied with failure probability δ; its logarithmic parameter is precisely $\iota .$ Thus, for $\mathcal { E } : = \mathcal { E } _ { 1 } \cap \mathcal { E } _ { 2 }$ , we have $\operatorname* { P r } ( \mathcal { E } ) \geq 1 - 3 \delta / 2$

Step 1: Controlling $B _ { T } ^ { \gamma }$ on ${ \mathcal { E } } _ { 1 }$ . Because $m _ { t } - P _ { S _ { t } } \nabla f ( x _ { t } ) \in S _ { t }$ and $P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \in S _ { t } ^ { \perp }$ , for each $t \in [ T ]$

$$
\begin{array} { r l } & { \| m _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } = \| m _ { t } - P _ { S _ { t } } \nabla f ( x _ { t } ) \| ^ { 2 } + \| P _ { S _ { t } } ^ { \bot } \nabla f ( x _ { t } ) \| ^ { 2 } } \\ & { \qquad \leq \displaystyle \frac { k L ^ { 2 } \tau ^ { 2 } } { 4 } + \| P _ { S _ { t } } ^ { \bot } \nabla f ( x _ { t } ) \| ^ { 2 } . } \end{array}
$$

On $\mathcal { E } _ { 1 }$ , substituting this bound and applying the balance control from Lemma 4.3 with failure probability $\delta / 2$ gives

$$
\begin{array} { r l } & { \alpha _ { t } ^ { 2 } d \left\| m _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \alpha _ { t } ^ { 2 } d \left\| P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \right\| ^ { 2 } + \frac { \alpha _ { t } ^ { 2 } d k L ^ { 2 } \tau ^ { 2 } } { 4 } } \\ & { \qquad \leq \frac { \alpha _ { t } } { 2 \gamma L } \| \nabla f ( x _ { t } ) \| ^ { 2 } + c \tau ^ { 2 } \big ( d ^ { 3 } + d \log ^ { 2 } ( 4 N T / \delta ) + d k \big ) . } \end{array}
$$

Here we also used $\alpha _ { t } \leq 1 / ( \gamma L )$ . Since $N = O ( \log T )$ and $d \geq \iota .$ , we have $\log ( 4 N T / \delta ) \leq c d$ . Dropping the remaining negative descent credit from each summand and summing over t therefore yields

$$
B _ { T } ^ { \gamma } \leq c \tau ^ { 2 } T ( d ^ { 3 } + d k ) .\tag{18}
$$

Step 2: Applying Theorem 3.4 on $\mathcal { E } .$ . Deterministically, the step sizes satisfy $\alpha _ { t } \leq 1 / ( \gamma L ) = 1 / ( C _ { 0 } \iota L )$ , which is precisely the hypothesis of Theorem 3.4. On ${ \mathcal { E } } _ { 2 }$ , the high-probability bound gives

$$
\begin{array} { r } { \frac { 1 } { L } \sum _ { t = 1 } ^ { T } \alpha _ { t } \| \nabla f ( x _ { t } ) \| ^ { 2 } \leq C \left( D _ { f } ^ { 2 } + \frac { B ^ { 2 } } { L ^ { 2 } } + \tau ^ { 2 } T d ^ { 3 } + \mathcal { B } _ { T } ^ { \gamma } \iota \right) . } \end{array}
$$

Substituting (18) and using $\iota = \log ( 2 T ^ { 2 } )$

$$
\begin{array} { r } { \frac { 1 } { L } \sum _ { t = 1 } ^ { T } \alpha _ { t } \| \nabla f ( x _ { t } ) \| ^ { 2 } \leq c \left( D _ { f } ^ { 2 } + \frac { B ^ { 2 } } { L ^ { 2 } } + \tau ^ { 2 } T ( d ^ { 3 } + d k ) \log ( 2 T ^ { 2 } ) \right) . } \end{array}\tag{19}
$$

Step 3: Taking expectations. On $\mathcal { E }$ (probability $\geq 1 - 3 / ( 2 T ) )$ , local optimality gives $\alpha _ { t } \geq \alpha _ { t } ^ { * } / ( c _ { 0 } \gamma )$ for all $t ,$ so $\mathcal { A } _ { T } \geq \mathcal { A } _ { T } ^ { \ast } / ( \gamma c _ { 0 } )$ . Dividing (19) by A<sub>T</sub> and using the output rule,

$$
\begin{array} { r } { \mathbb { I } _ { \mathcal { E } } \cdot \mathbb { E } _ { R } \big [ \| \nabla f ( \overline { { x } } ) \| ^ { 2 } \big ] \leq \frac { c \gamma L } { \mathcal { A } _ { T } ^ { \ast } } \left( D _ { f } ^ { 2 } + \frac { B ^ { 2 } } { L ^ { 2 } } + \tau ^ { 2 } T ( d ^ { 3 } + d k ) \log ( 2 T ^ { 2 } ) \right) . } \end{array}
$$

On $\mathcal { E } ^ { c }$ (probability $\leq 3 / ( 2 T ) )$ , we bound $\| \nabla f ( \overline { { x } } ) \| ^ { 2 } \leq B ^ { 2 }$ . Taking full expectations,

$$
\begin{array} { r } { \frac { 1 } { L } \mathbb { E } \big [ \| \nabla f ( \overline { { x } } ) \| ^ { 2 } \big ] \leq c \gamma \mathbb { E } \Bigg [ \frac { D _ { f } ^ { 2 } + \frac { B ^ { 2 } } { L ^ { 2 } } + \tau ^ { 2 } T ( d ^ { 3 } + d k ) \log ( 2 T ^ { 2 } ) } { A _ { T } ^ { \ast } } \Bigg ] + \frac { 3 B ^ { 2 } } { 2 L T } . } \end{array}
$$

The assumed bound on τ makes the smoothing term at most $B ^ { 2 } / L ^ { 2 }$ . Finally, $\gamma = O ( \log ( 2 T ) )$ and $\mathcal { A } _ { T } ^ { \ast } \leq T / L$ , so the failure-event term is absorbed into the first term, proving the claim. ■

## D.1 Proof of Lemma 4.1

Lemma 4.1 (Restated) For every $x \in \mathbb { R } ^ { d }$ and $S \in \operatorname { G r } _ { k } ( \mathbb { R } ^ { d } )$ , Algorithm 2 satisfies

$$
\Vert P _ { f } ^ { \tau } ( x , S ) - P s \nabla f ( x ) \Vert ^ { 2 } \leq \frac { k L ^ { 2 } \tau ^ { 2 } } { 4 } .
$$

Proof Let $B = [ b _ { 1 } , \ldots , b _ { k } ]$ be the orthonormal basis matrix used by Algorithm 2, and let $\boldsymbol { y } : = \boldsymbol { B } ^ { \intercal } \nabla f ( \boldsymbol { x } )$ . For each $j \in [ k ]$ , L-smoothness gives

$$
| \widehat { y } _ { j } - \langle \nabla f ( x ) , b _ { j } \rangle | = \frac { 1 } { \tau } \left| f ( x + \tau b _ { j } ) - f ( x ) - \tau \langle \nabla f ( x ) , b _ { j } \rangle \right| \leq \frac { L \tau } { 2 } ,
$$

where we used $\| b _ { j } \| = 1$ . Since $P s = B B ^ { \top }$ and $B ^ { \top } B = I _ { k }$

$$
\| P _ { f } ^ { \tau } ( x , S ) - P _ { S } \nabla f ( x ) \| ^ { 2 } = \| B ( \widehat { y } - y ) \| ^ { 2 } = \sum _ { j = 1 } ^ { k } | \widehat { y } _ { j } - \langle \nabla f ( x ) , b _ { j } \rangle | ^ { 2 } \leq \frac { k L ^ { 2 } \tau ^ { 2 } } { 4 } .
$$

## D.2 Proofs of Lemmas 4.2 and 4.3

We first establish the norm estimation guarantee (Lemma 4.2), then use it to prove Lemma $4 . 3 .$

Lemma 4.2 (Restated) There exist absolute constants $C , c > 0$ such that for every $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , linear subspace S of $\mathbb { R } ^ { d } , \tau > 0 , \delta \in ( 0 , 1 )$ , and $N \geq C \log ( 2 / \delta )$ , it holds with probability at least $1 - \delta$ that

$$
\begin{array} { r } { N _ { f } ^ { \tau } ( x , S , N ) \in \left[ \frac { 1 } { 2 } \| P _ { S } \nabla f ( x ) \| ^ { 2 } - c \varepsilon , \frac { 3 } { 2 } \| P _ { S } \nabla f ( x ) \| ^ { 2 } + c \varepsilon \right] , } \end{array}
$$

where $\varepsilon = \tau ^ { 2 } L ^ { 2 } ( d ^ { 2 } + \log ^ { 2 } ( 2 N / \delta ) )$

Proof Let $v = P s \nabla f ( x )$ . By L-smoothness of $f ,$

$$
\widehat { y } _ { i } = \left. \nabla f ( \boldsymbol { x } ) , P _ { S } \boldsymbol { u } _ { i } \right. + \tau \epsilon _ { i } = \left. \boldsymbol { v } , \boldsymbol { u } _ { i } \right. + \tau \epsilon _ { i } ,
$$

with $\begin{array} { r } { | \epsilon _ { i } | \le \frac { L } { 2 } \| P s u _ { i } \| ^ { 2 } \le \frac { L } { 2 } \| u _ { i } \| ^ { 2 } } \end{array}$ . Decompose the estimator as follows

$$
\frac { 1 } { N } \sum _ { i = 1 } ^ { N } \widehat { y } _ { i } ^ { 2 } \ = \ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \langle v , u _ { i } \rangle ^ { 2 } \ + \ \frac { 2 \tau } { N } \sum _ { i = 1 } ^ { N } \langle v , u _ { i } \rangle \epsilon _ { i } \ + \ \frac { \tau ^ { 2 } } { N } \sum _ { i = 1 } ^ { N } \epsilon _ { i } ^ { 2 } .
$$

Let $\begin{array} { r } { S _ { 1 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \langle v , u _ { i } \rangle ^ { 2 } } \end{array}$ and $\begin{array} { r } { S _ { 2 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lVert u _ { i } \rVert ^ { 4 } } \end{array}$ . Using the fact that $\begin{array} { r } { \epsilon _ { i } ^ { 2 } \leq \frac { L ^ { 2 } } { 4 } \| u _ { i } \| ^ { 4 } } \end{array}$

$$
\left| \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \widehat { y } _ { i } ^ { 2 } - S _ { 1 } \right| \leq \frac { S _ { 1 } } { 4 } + \frac { 5 \tau ^ { 2 } } { N } \sum _ { i = 1 } ^ { N } \epsilon _ { i } ^ { 2 } \leq \frac { S _ { 1 } } { 4 } + \frac { 5 \tau ^ { 2 } L ^ { 2 } S _ { 2 } } { 4 } .\tag{20}
$$

Step 1: Concentrating $S _ { 1 }$ around $\| v \| ^ { 2 }$ . When $v = 0 , S _ { 1 } = 0$ holds almost surely. When v $\neq 0 , \langle v , u _ { i } \rangle ^ { 2 } / \| v \| ^ { 2 }$ <sup>iid</sup>∼ $\chi _ { 1 } ^ { 2 }$ . By Laurent-Massart (Lemma E.3), for $N \geq 2 5 6 \log ( 4 / \delta )$

$$
\begin{array} { r } { \mathbb { P } \left( | S _ { 1 } - \| v \| ^ { 2 } | \leq \frac { 1 } { 5 } \| v \| ^ { 2 } \right) \geq 1 - \delta / 2 . } \end{array}
$$

Step 2: Bounding $S _ { 2 } .$ . Since $\| u _ { i } \| ^ { 2 } \overset { \mathrm { i i d } } { \sim } \chi _ { d } ^ { 2 }$ , by the union bound of Laurent-Massart (Lemma E.3),

$$
\operatorname* { P r } \left( \operatorname* { m a x } _ { i \in [ N ] } | \| u _ { i } \| ^ { 2 } - d | \leq 2 \sqrt { d \log ( 4 N / \delta ) } + 2 \log ( 4 N / \delta ) \right) \geq 1 - \delta / 2 .
$$

Consequently, as $\begin{array} { r } { S _ { 2 } \leq \operatorname* { m a x } _ { i \in [ N ] } \| u _ { i } \| ^ { 4 } = ( \operatorname* { m a x } _ { i \in [ N ] } \| u _ { i } \| ^ { 2 } ) ^ { 2 } } \end{array}$ , we have

$$
\operatorname* { P r } \left( S _ { 2 } \leq 1 8 ( d ^ { 2 } + \log ^ { 2 } ( 4 N / \delta ) ) \right) \geq 1 - \delta / 2 .
$$

Conclusion. Combining (20) with a union-bound over events in Steps 1 and 2, we have that with probability at least $1 - \delta ,$

$$
\begin{array} { r l r } {  { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \hat { y } _ { i } ^ { 2 } \in [ \frac { 3 S _ { 1 } } { 4 } - \frac { 5 \tau ^ { 2 } L ^ { 2 } S _ { 2 } } { 4 } , \frac { 5 S _ { 1 } } { 4 } + \frac { 5 \tau ^ { 2 } L ^ { 2 } S _ { 2 } } { 4 } ] } } \\ & { } & { \qquad \subset [ \frac { \| v \| ^ { 2 } } { 2 } - 2 5 \tau ^ { 2 } L ^ { 2 } ( d ^ { 2 } + \log ^ { 2 } ( 4 N / \delta ) ) , \frac { 3 \| v \| ^ { 2 } } { 2 } + 2 5 \tau ^ { 2 } L ^ { 2 } ( d ^ { 2 } + \log ^ { 2 } ( 4 N / \delta ) ) ] . } \end{array}
$$

Here $( 3 / 4 ) ( 4 / 5 ) = 3 / 5 \geq 1 / 2 , ( 5 / 4 ) ( 6 / 5 ) = 3 / 2$ , and $( 5 / 4 ) \cdot 1 8 \leq 2 5$ . Since $\log ( 4 / \delta ) \leq 2 \log ( 2 / \delta )$ and $\log ( 4 N / \delta ) \le 2 \log ( 2 N / \delta )$ , the stated constants may be taken as $C = 5 1 2$ and $c = 1 0 0$ . This proves the lemma.

Lemma 4.3 (Restated) There exist absolute constants $C , c _ { 0 } , c _ { 1 } > 0$ such that the following holds. For any $\delta \in ( 0 , 1 )$ and $N \ge C \log ( 2 T / \delta )$ , set $\varepsilon : = \tau ^ { 2 } L ^ { 2 } ( d ^ { 2 } + \log ^ { 2 } ( 2 N T / \delta ) )$ ) and define

$$
\begin{array} { r } { \alpha _ { t } : = \frac { 1 } { \gamma L } \operatorname* { m i n } \left\{ 1 , \frac { N _ { f } ^ { \tau } ( x _ { t } , \mathbb { R } ^ { d } , N ) + c _ { 1 } \varepsilon } { 1 6 d \operatorname* { m a x } \{ 0 , N _ { f } ^ { \tau } ( x _ { t } , S _ { t } ^ { \bot } , N ) - c _ { 1 } \varepsilon \} } \right\} , } \end{array}
$$

If the denominator is zero, define $\alpha _ { t } : = 1 / ( \gamma L )$ . Then with probability at least $1 - \delta ,$ , for all $t \in [ T ]$ simultaneously:

1. (Local Optimality) $\begin{array} { r } { \alpha _ { t } \geq \frac { \alpha _ { t } ^ { * } } { c _ { 0 } \gamma } , } \end{array}$

2. (Balance Control) $\begin{array} { r } { \alpha _ { t } ^ { 2 } d \| P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 } \le \frac { \alpha _ { t } } { 2 \gamma L } \| \nabla f ( x _ { t } ) \| ^ { 2 } + c _ { 0 } \tau ^ { 2 } \big ( d ^ { 3 } + d \log ^ { 2 } ( 2 N T / \delta ) \big ) . } \end{array}$

Proof Let $c _ { \mathrm { N E } }$ be the error constant in Lemma 4.2, take $c _ { 1 } = 4 c _ { \mathrm { N E } }$ , and abbreviate $c : = c _ { 1 }$ within this proof. Write $a _ { t } : = N _ { t } ^ { \tau } ( x _ { t } , \mathbb { R } ^ { d } , N )$ and $b _ { t } : = N _ { t } ^ { \tau } ( x _ { t } , S _ { t } ^ { \bot } , N )$ , which target $\| \nabla f ( x _ { t } ) \| ^ { 2 }$ and $\| P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 }$ , respectively. Apply Lemma $\check { 4 } . 2$ conditionally on the information available before each call, with failure probability $\delta / ( 2 T )$ . Its error radius is at most $4 \varepsilon .$ , since $\log ( 4 N T / \delta ) \leq 2 \log ( 2 N T / \delta )$ . A union bound over the $2 T$ calls therefore gives, for $N \ge C \log ( 2 T / \delta )$ with C sufficiently large, with probability at least $1 - \delta .$

$$
\begin{array} { r } { a _ { t } \in \left[ \frac { 1 } { 2 } \| \nabla f ( x _ { t } ) \| ^ { 2 } - c \varepsilon , \frac { 3 } { 2 } \| \nabla f ( x _ { t } ) \| ^ { 2 } + c \varepsilon \right] } \end{array}
$$

and similarly for $b _ { t }$ targeting $\| P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 }$ , simultaneously for all $t \in [ T ]$ . We work on this event throughout the rest of the proof.

Local optimality. The bounds $\begin{array} { r } { a _ { t } + c \varepsilon \geq \frac { 1 } { 2 } \| \nabla f ( x _ { t } ) \| ^ { 2 } } \end{array}$ and max $\begin{array} { r } { \{ 0 , b _ { t } - c \varepsilon \} \leq \| P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 } } \end{array}$ yield

$$
\begin{array} { r } { \alpha _ { t } \geq \frac { 1 } { \gamma L } \operatorname* { m i n } \left\{ 1 , \frac { \| \nabla f ( x _ { t } ) \| ^ { 2 } } { 6 4 d \| P _ { \mathcal { S } _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 } } \right\} \geq \frac { \alpha _ { t } ^ { * } } { 6 4 \gamma } . } \end{array}
$$

(When $b _ { t } - c \varepsilon \leq 0$ or $\nabla f ( x _ { t } ) = 0 , \alpha _ { t } = 1 / ( \gamma L ) \geq \alpha _ { t } ^ { * } / \gamma$ directly.)

Balance Control. We bound $\alpha _ { t } ^ { 2 } d \parallel P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \parallel ^ { 2 }$ in two cases.

Case 1: $\begin{array} { r } { c \varepsilon \leq \| P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 } } \end{array}$ . Then $\begin{array} { r } { b _ { t } - c \varepsilon \ge \frac { 1 } { 4 } \| P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 } } \end{array}$ and $a _ { t } + c \varepsilon \leq { \textstyle { \frac { 3 } { 2 } } } \| \nabla f ( x _ { t } ) \| ^ { 2 } + 2 c \varepsilon$ , which guarantees

$$
\begin{array} { r } { \alpha _ { t } \leq \frac { 1 } { \gamma L } \operatorname* { m i n } \left\{ 1 , \frac { \| \nabla f ( x _ { t } ) \| ^ { 2 } } { 2 d \| P _ { \mathcal { S } _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 } } \right\} + \frac { c \varepsilon } { 2 \gamma L d \| P _ { \mathcal { S } _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 } } . } \end{array}
$$

Multiplying by $\alpha _ { t } d \| P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 }$ and using $\alpha _ { t } \leq 1 / ( \gamma L )$

$$
\begin{array} { r } { \alpha _ { t } ^ { 2 } d \| P _ { S _ { t } } ^ { \bot } \nabla f ( x _ { t } ) \| ^ { 2 } \leq \frac { \alpha _ { t } } { 2 \gamma L } \| \nabla f ( x _ { t } ) \| ^ { 2 } + \frac { c \varepsilon } { 2 L ^ { 2 } } \leq \frac { \alpha _ { t } } { 2 \gamma L } \| \nabla f ( x _ { t } ) \| ^ { 2 } + c \tau ^ { 2 } \big ( d ^ { 2 } + \log ^ { 2 } ( 2 N T / \delta ) \big ) . } \end{array}
$$

Case 2: $\begin{array} { r } { c \varepsilon > \frac { 1 } { 8 } \| P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 } } \end{array}$ . Then $\| P _ { S _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 } <$ 8cε and, using $\alpha _ { t } \leq 1 / ( \gamma L ) \leq 1 / L$

$$
\begin{array} { r } { \alpha _ { t } ^ { 2 } d \| P _ { \mathcal { S } _ { t } } ^ { \perp } \nabla f ( x _ { t } ) \| ^ { 2 } \le \frac { 1 } { \gamma ^ { 2 } L ^ { 2 } } \cdot d \cdot 8 c \varepsilon \le 8 c \tau ^ { 2 } ( d ^ { 3 } + d \log ^ { 2 } ( 2 N T / \delta ) ) . } \end{array}
$$

Combining both cases and taking $c _ { 0 } = \operatorname* { m a x } \{ 6 4 , 8 c \}$ proves both claims of the lemma.

## E Technical Preliminaries

Fact E.1 Let $u \sim \mathcal { N } ( 0 , I _ { d } )$ . For $k \in \mathbb N ,$ , it holds that

$$
\mathbb { E } \big [ ( u u ^ { \top } ) ^ { k } \big ] = \prod _ { j = 1 } ^ { k - 1 } ( d + 2 j ) I _ { d } .
$$

Fact E.2 Let $u \sim \mathcal { N } ( 0 , I _ { d } )$ . For $k > - d ,$ it holds that

$$
\mathbb { E } \big [ \| u \| ^ { k } \big ] \leq 2 ^ { k / 2 } \frac { \Gamma ( \frac { d + k } { 2 } ) } { \Gamma ( \frac { d } { 2 } ) } \leq ( d + k ) ^ { k / 2 } .
$$

In particular,for $m \in \mathbb { N } , \mathbb { E } \big [ \| u \| ^ { 2 m } \big ] = d ( d + 2 ) \ldots ( d + 2 m - 2 )$

Lemma E.3 (Laurent–Massart) Let $X _ { 1 } , \ldots , X _ { N } \stackrel { \mathrm { i i d } } { \sim } \chi _ { d } ^ { 2 }$ and $\begin{array} { r } { \bar { X } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } X _ { i } } \end{array}$ . For every $\delta \in ( 0 , 1 )$ ,

$$
\mathbb { P } \Bigg ( \vert \bar { X } - d \vert \leq 2 \sqrt { \frac { d \log ( 2 / \delta ) } { N } } + \frac { 2 \log ( 2 / \delta ) } { N } \Bigg ) \geq 1 - \delta .
$$

Lemma E.4 (Isserlis) For fixed $a \in \mathbb { R } ^ { d }$ and $\Sigma \in \mathbb { R } ^ { d \times d }$

$$
\begin{array} { r } { \mathbb E _ { z \sim \mathcal N ( 0 , I _ { d } ) } \left[ ( \boldsymbol a ^ { \top } \boldsymbol z ) ^ { 2 } ( \boldsymbol z ^ { \top } \Sigma \boldsymbol z ) \right] = \| \boldsymbol a \| ^ { 2 } \operatorname { t r } ( \Sigma ) + 2 \boldsymbol a ^ { \top } \Sigma \boldsymbol a . } \end{array}
$$

Proof Write $\begin{array} { r } { ( a ^ { \top } z ) ^ { 2 } ( z ^ { \top } \Sigma z ) = \sum _ { i , j , k , l } a _ { i } a _ { j } \Sigma _ { k l } z _ { i } z _ { j } z _ { k } z _ { l } } \end{array}$ . Since $z \sim \mathcal { N } ( 0 , I _ { d } )$ , Isserlis’ theorem gives $\mathbb { E } [ z _ { i } z _ { j } z _ { k } z _ { l } ] =$ $\delta _ { i j } \delta _ { k l } + \delta _ { i k } \delta _ { j l } + \delta _ { i l } \delta _ { j k }$ . Substituting and summing over each pairing:

$$
\sum _ { i , j , k , l } a _ { i } a _ { j } \Sigma _ { k l } \delta _ { i j } \delta _ { k l } = \left( \sum _ { i } a _ { i } ^ { 2 } \right) \left( \sum _ { k } \Sigma _ { k k } \right) = \| a \| ^ { 2 } \ \mathrm { t r } ( \Sigma ) ,
$$

$$
\sum _ { i , j , k , l } a _ { i } a _ { j } \Sigma _ { k l } \delta _ { i k } \delta _ { j l } = \sum _ { i , j } a _ { i } a _ { j } \Sigma _ { i j } = a ^ { \top } \Sigma a ,
$$

$$
\sum _ { i , j , k , l } a _ { i } a _ { j } \Sigma _ { k l } \delta _ { i l } \delta _ { j k } = \sum _ { i , j } a _ { i } a _ { j } \Sigma _ { j i } = a ^ { \top } \Sigma a ,
$$

where the last equality uses symmetry of Σ. Summing the three terms yields the result.