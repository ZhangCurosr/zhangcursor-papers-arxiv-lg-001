# Optimal Last-Iterate Bounds for Nonexpansive Stochastic Approximation

Ramanan Tamizholi<sup>1</sup>, Siddharth Chandak<sup>2</sup>, and Rahul Singh<sup>3</sup>

<sup>1</sup>Department of Computer Science and Automation, Indian Institute of Science, Bengaluru, India. ramanant@iisc.ac.in

<sup>2</sup>Department of Electrical Engineering, Stanford University, Stanford, California, USA. chandaks@stanford.edu <sup>3</sup>EPITA, Laboratoire Recherche de l’EPITA, Paris, France. rahulsingh0188@gmail.com

## Abstract

We establish last-iterate mean-square residual bounds for nonexpansive stochastic approximation (SA), or stochastic Krasnoselskii–Mann (KM) iterations. In particular, we consider fixed-point mappings that are nonexpansive with respect to the Euclidean norm, and focus on polynomially decaying stepsizes. We study three conditions on the conditional second moment of the noise, and obtain bounds that are optimal up to logarithmic factors in each case. When it scales linearly with the squared residual, the best rate is $O ( 1 / k ^ { 1 - \epsilon } )$ for any � > 0. In this case, for all considered stepsizes, our bounds match those of the noiseless KM iteration. When it is bounded by a constant, the best rate is $O ( \log ^ { 2 } k / \sqrt { k } )$ . When it scales afinely with the squared norm of the iterate, the best rate is $O ( 1 / k ^ { 1 / 2 - \epsilon } )$ , and the mean-square residual can diverge if the stepsizes are not square-summable. Thus, unlike for contractive SA, square-summable stepsizes are necessary for nonexpansive SA. We demonstrate how various algorithms with cocoercive maps, such as stochastic gradient descent, linear SA, and learning in cocoercive games, fit into our framework, and how the diferent noise conditions arise in each of them.

Keywords: Stochastic approximation, nonexpansive maps, Krasnoselskii–Mann iteration, last-iterate convergence, finite-time analysis, cocoercive operators

## 1 Introduction

Stochastic approximation (SA) is a class of iterative algorithms to find fixed points of an operator given its noisy realizations [28]. These algorithms have found applications in various areas, including optimization, reinforcement learning [33], communications and stochastic control [2]. Consider the SA iteration

$$
x _ { k + 1 } = x _ { k } + \beta _ { k } \left( f ( x _ { k } ) - x _ { k } + M _ { k + 1 } \right) .
$$

Here, $\boldsymbol { x } _ { k } \in \mathbb { R } ^ { d }$ is the iterate, $\beta _ { k }$ is the stepsize, $\{ M _ { k + 1 } \}$ is a martingale diference noise sequence, and $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is the mapping for which we wish to find a fixed point $x ^ { \star } , \mathrm { i . e . , a }$ point satisfying $f ( x ^ { \star } ) = x ^ { \star }$

Most finite-time analyses of SA assume that � is a contraction [8, 9, 11, 32]. However, this assumption does not hold in many problems of interest, e.g., stochastic gradient descent (SGD) for convex functions that are not strongly convex, where the gradient step is only nonexpansive. In this work, we study SA where � is nonexpansive with respect to the Euclidean norm. These iterations are also known as stochastic Krasnoselskii–Mann (KM) iterations [22, 26]. As nonexpansive mappings need not have a unique fixed point, we bound the last-iterate mean-square residual $\mathbb { E } \left[ \| f ( x _ { k } ) - x _ { k } \| ^ { 2 } \right]$ rather than the distance to a fixed point.

While there has been recent interest in stochastic fixed-point computation for nonexpansive mappings, existing works difer from ours in two ways. First, some works consider mappings that are nonexpansive with respect to general norms [1, 4]. Due to their more general scope, their bounds are weaker than what can be obtained in the Euclidean setting. Second, other works modify the iteration, either through Halpern-type anchoring [5, 6, 16, 31] or through modifications of the KM iteration such as minibatches or inertia [19, 35]. In contrast, we focus on the standard stochastic KM iteration with polynomially decaying stepsizes under the Euclidean norm.

We study three conditions on the conditional second moment of the noise. Under each condition, we obtain the first last-iterate mean-square residual bounds that are optimal up to logarithmic factors.

(R) Residual-dependent: bounded by a multiple of the squared residual $\| f ( x _ { k } ) - x _ { k } \| ^ { 2 }$

– Best rate: $O ( 1 / k ^ { 1 - \epsilon } )$ , where $\epsilon > 0$ can be arbitrarily small.

(C) Bounded by a Constant.

– Best rate: $\widetilde { O } ( 1 / \sqrt { k } )$ , where $\widetilde { o }$ hides logarithmic factors.

(I) Iterate-dependent: bounded by an afine function of the squared norm of the iterate $\| x _ { k } \| ^ { 2 }$

– Best rate: $O ( 1 / k ^ { 1 / 2 - \epsilon } )$ , where $\epsilon > 0$ can be arbitrarily small.

While most finite-time guarantees for stochastic KM iterations are obtained under condition C [4, 5], condition I is the standard noise model in the SA literature [2]. We additionally include condition R for two reasons. First, it arises in applications, e.g., as the strong growth condition for SGD [30, 34]. Second, it shows that noise vanishing at the fixed points does not slow down convergence: under condition R, our bounds match those for the noiseless KM iteration for all polynomially decaying stepsizes, which is not the case under condition C or I. Moreover, under condition I, we show that square-summable stepsizes are necessary, as the mean-square residual can otherwise diverge. This requirement is absent for contractive SA, where convergence is achieved even for non-square-summable stepsizes [8, 11], and thus just moving from contractive to nonexpansive mappings brings it back. We explain the relevance of these conditions through three applications: SGD, linear SA, and learning in games.

Finally, we briefly outline our proof technique. Bounding the ergodic residual, i.e., a weighted average of past residuals, is well understood from the analysis of inexact KM iterations [15, 24]. To obtain the last-iterate bound, we compare the stochastic iterates with a noiseless auxiliary iteration restarted from an earlier stochastic iterate, and bound how far the two trajectories can drift apart.

## 1.1 Related Work

In recent years, stochastic KM iterations have been studied for mappings that are nonexpansive with respect to general norms. In this setting, almost sure convergence and finite-time last-iterate expected residual bounds were established in [4], and extended to Markovian noise in [1]. In contrast, we focus on the Euclidean norm, which allows us to obtain sharper bounds on the mean-square residual. Moreover, while the analyses in [1, 4] also rely on an auxiliary iteration, it is defined through an averaged noise sequence rather than a noiseless iteration.

Much of the remaining literature on nonexpansive mappings either modifies the KM iteration or studies variants ofHalpern iterations. The former includes using increasing batch sizes [19] and introducing inertia [35], and the latter includes Halpern iterations combined with minibatches [5, 20], variance reduction [6, 16, 17, 21], or tuned anchoring schedules [31].

Bounds on the convergence rates of two-time-scale SA with a nonexpansive mapping in the slower time-scale have also been obtained in [7, 29]. While the slower time-scale in this setting is also a stochastic inexact KM iteration, its rates are further limited by the time-scale separation.

Closest to our work, [14] studies stochastic KM iterations in Hilbert spaces. It assumes that each sampled function is nonexpansive and that the noise has finite variance at a fixed point, and establishes almost sure convergence and finite-time last-iterate mean-square residual bounds. In contrast, we require only the underlying function $f$ to be nonexpansive and impose moment conditions directly on the noise, without requiring it to arise from sampled nonexpansive functions. Moreover, our bounds improve on those of [14] for some stepsizes and match them for others.

## 1.2 Outline and Notation

The paper is organized as follows. Section 2 sets up the problem and presents our upper bounds under each noise condition, followed by remarks on the choice of stepsizes and the optimality of the bounds. Section 3 gives a proof sketch of the upper bounds. Section 4 presents the lower bounds, which establish optimality, and the divergence result under condition I. Section 5 discusses three applications and how our noise conditions arise in them. Finally, Section 6 concludes the paper.

Throughout, $\| \cdot \|$ and $\langle \cdot , \cdot \rangle$ denote the Euclidean norm and inner product on $\mathbb { R } ^ { d }$ . For sequences, $O ( \cdot )$ and $\Omega ( \cdot )$ denote the standard order notation, and $\widetilde { O } ( \cdot )$ additionally hides logarithmic factors.

## 2 Problem Formulation and Main Results

Consider the following iteration.

$$
x _ { k + 1 } = x _ { k } + \beta _ { k } \left( f ( x _ { k } ) - x _ { k } + M _ { k + 1 } \right) .\tag{1}
$$

Here, $\boldsymbol { x } _ { k } \in \mathbb { R } ^ { d }$ is the iterate and $\beta _ { k }$ is the stepsize at time $k \geq 0$ . The function $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ denotes the mapping whose fixed point we wish to find, and $M _ { k + 1 }$ denotes the martingale diference noise sequence.

The first assumption states that the mapping $f$ is nonexpansive and has at least one fixed point.

Assumption 1. The function � is nonexpansive, i.e., for all $x _ { 1 } , x _ { 2 } \in \mathbb { R } ^ { d }$

$$
\| f ( x _ { 1 } ) - f ( x _ { 2 } ) \| \leq \| x _ { 1 } - x _ { 2 } \| .
$$

Moreover, the set of fixed points $X ^ { \star } = \{ x \in \mathbb { R } ^ { d } : f ( x ) = x \}$ is nonempty.

Unlike a contraction, a nonexpansive mapping can have multiple fixed points. Hence, instead of the distance $\| x _ { k } - x ^ { \star } \|$ to a particular fixed point $x ^ { \star }$ , we measure convergence through the residual $\| f ( x _ { k } ) - x _ { k } \|$ , which is the standard measure of convergence for nonexpansive SA and KM iterations. Our objective is to obtain finite-time bounds on the last-iterate mean-square residual E $\cdot \left[ \| f ( x _ { k } ) - x _ { k } \| ^ { 2 } \right]$

We next state the assumption on the noise sequence $\{ M _ { k + 1 } \}$ . We consider three diferent conditions on its conditional second moment, and obtain a separate bound under each of them.

Assumption 2. Define the family of �-fields $\mathcal { F } _ { k } = \sigma ( x _ { 0 } , M _ { i } , i \le k )$ . Then $\{ M _ { k + 1 } \}$ is a martingale diference sequence with respect to $\mathcal { F } _ { k } ,$ $i . e . , \mathbb { E } \left[ M _ { k + 1 } \mid \mathcal { F } _ { k } \right] = 0$ . Moreover,for some constant $\sigma > 0$ , one ofthefollowing conditions holds for all $k \geq 0$

(R) Conditional second moment is Residual-dependent:

$$
\begin{array} { r } { \mathbb { E } \left[ \| M _ { k + 1 } \| ^ { 2 } \vert \mathcal { F } _ { k } \right] \leq \sigma ^ { 2 } \| f ( x _ { k } ) - x _ { k } \| ^ { 2 } . } \end{array}
$$

(C) Conditional second moment is bounded by a Constant:

$$
\mathbb { E } \left[ \lVert M _ { k + 1 } \rVert ^ { 2 } \ | \ \mathcal { F } _ { k } \right] \leq \sigma ^ { 2 } .
$$

(I) Conditional second moment is afinely bounded by the Iterate:

$$
\begin{array} { r } { \mathbb { E } \left[ \| M _ { k + 1 } \| ^ { 2 } \vert \mathcal { F } _ { k } \right] \leq \sigma ^ { 2 } \left( 1 + \| x _ { k } \| ^ { 2 } \right) . } \end{array}
$$

Under condition R, the noise scales with the residual, and hence vanishes at the fixed points of �. For example, for SGD iterations, noise whose conditional second moment scales with the squared norm of the gradient satisfies condition R [3, 30, 34]. Condition I is the most general of the three, and is the standard noise model in the $\mathrm { S A }$ literature [2, 23]. It clearly subsumes condition C, and also subsumes condition R, albeit with $\sigma ^ { 2 }$ replaced by $8 \sigma ^ { 2 }$ max $\{ 1 , \| x ^ { \star } \| ^ { 2 } \}$ for any $x ^ { \star } \in \mathcal { X } ^ { \star }$ . However, the more restrictive conditions lead to better rates, and we therefore treat the three conditions separately.

Finally, we state the assumption on the stepsize sequence.

Assumption 3. The stepsize sequence is of the form

$$
\beta _ { k } = \frac { \beta } { ( k + K _ { 0 } ) ^ { b } } ,
$$

where $b \in ( 0 , 1 ) , 0 < \beta \leq 1 / 2$ , and $K _ { 0 } \ge 1$ . Moreover, under condition R in Assumption 2, $( 1 + \sigma ^ { 2 } ) \beta _ { 0 } < 1$

The conditions $\beta \leq 1 / 2$ and $K _ { 0 } \ge 1$ ensure that $\beta _ { k } \le 1 / 2$ for all $k \geq 0$ . The additional condition under noise condition R ensures that the stepsize is suficiently small relative to the noise magnitude, starting from time $k = 0$ . We do not consider stepsizes of the form $\beta _ { k } = \Theta ( 1 / k )$ , as they lead to bounds of the form $O ( 1 / \log k )$ , which are weaker than the bounds obtained for $b < 1$

We now state our main results, one for each noise condition in Assumption 2. An outline of their proofs is presented in Section 3, and the complete proofs are given in Appendix A. The lower bounds showing optimality of these results are presented in Section 4. Each theorem is followed by a short discussion of the resulting rate and the choice of stepsize.

Theorem 1. Suppose Assumptions 1–3 hold, with noise condition R in Assumption 2. Then there exists a constant ${ \mathfrak { c } } _ { 1 } > 0$ such that, for all $k \geq 0 _ { ; }$

$$
\mathbb { E } \left[ \Vert f ( x _ { k } ) - x _ { k } \Vert ^ { 2 } \right] \leq \frac { \mathfrak { c } _ { 1 } } { ( k + 1 ) ^ { 1 - b } } .
$$

Moreover, this bound is optimal (see Remark 1).

Best rate. A slowly decreasing stepsize, i.e., a smaller �, gives a better rate. In fact, we achieve the same rate as that achieved for noiseless KM iterations [13, 15]. The best rate is $O ( 1 / k ^ { 1 - \epsilon } )$ , where $\epsilon > 0$ can be arbitrarily small, and is achieved by choosing $b = \epsilon$

Theorem 2. Suppose Assumptions 1–3 hold, with noise condition C in Assumption 2. Then there exist constants $\mathbf { c } _ { 2 } , \mathbf { c } _ { 3 } > 0$ such that,for all $k \geq 0 _ { ; }$

$$
\mathbb { E } \left[ \Vert f ( x _ { k } ) - x _ { k } \Vert ^ { 2 } \right] \leq \left\{ \begin{array} { l l } { \frac { \mathfrak { c } _ { 2 } \log ^ { 2 } ( k + 2 ) } { ( k + 1 ) ^ { b } } , } & { 0 < b \leq \frac { 1 } { 2 } , } \\ { \frac { \mathfrak { c } _ { 3 } } { ( k + 1 ) ^ { 1 - b } } , } & { \frac { 1 } { 2 } < b < 1 . } \end{array} \right.
$$

Moreover, this bound is optimal up to a logarithmic factor (see Remark 1).

Non-monotone dependence on the stepsize. Unlike in Theorem 1, the rate is no longer monotone in �. For $b > 1 / 2 .$ , a smaller � gives a better rate, as before. For $b < 1 / 2$ , the noise dominates, and the bound decays only as $( k + 1 ) ^ { - b }$ , up to logarithmic factors. Hence, the best rate of $\widetilde { O } ( 1 / \sqrt { k } )$ is achieved by choosing $b = 1 / 2$

Theorem 3. Suppose Assumptions 1–3 hold, with noise condition I in Assumption 2, and let $\textstyle { \frac { 1 } { 2 } } < b < 1$ . Then there exists a constant ${ \mathfrak { c } } _ { 4 } > 0$ such that,for all $k \geq 0$

$$
\mathbb { E } \left[ \Vert f ( x _ { k } ) - x _ { k } \Vert ^ { 2 } \right] \leq \frac { \mathfrak { t } _ { 4 } } { ( k + 1 ) ^ { 1 - b } } .
$$

Moreover, this bound is optimal (see Remark 1). However, this result cannot be extended to $b \leq 1 / 2 .$ : for every $0 < b \leq 1 / 2$ , there exists an instance satisfying Assumptions 1–3 with noise condition I for which $\mathbb { E } [ \Vert f ( x _ { k } ) - x _ { k } \Vert ^ { 2 } ] $ ∞ as $k \to \infty$ (see Proposition 3).

Stepsize choice and necessity of square-summable stepsizes. For $b > 1 / 2$ , the rate matches that of Theorems 1 and 2. Since this result cannot be extended to $b \leq 1 / 2$ , the best rate is $O ( 1 / k ^ { 1 / 2 - \epsilon } )$ , where $\epsilon > 0$ can be arbitrarily small, and is achieved by choosing $b = 1 / 2 + \epsilon$ . The condition $b > 1 / 2$ is equivalent to the stepsizes being square-summable. While this is the standard requirement for the convergence analysis of SA without additional assumptions on the operator [2, 23, 28], it has been shown that for contractive fixed-point iterations, convergence is achieved for all $0 < b \leq 1 [ 8 , 1 1 ]$ . Theorem 3 shows that just by moving from contractive to nonexpansive mappings, the requirement of square-summable stepsizes comes back. This is not an artifact of our analysis: Proposition 3 gives a single instance for which the mean-square residual grows as exp $\left( \Omega ( k ^ { 1 - 2 b } ) \right)$ for $0 < b < 1 / 2$ , and polynomially for $b = 1 / 2$ . In contrast, Theorems 1 and 2 hold for all $0 < b < 1$ , and do not require square-summable stepsizes. In fact, under noise condition C, the best rate is achieved by $b = 1 / 2$

The following remarks discuss the optimality of the above bounds, and the implied bounds on the ergodic residual.

Remark 1 (Optimality of the bounds). By optimality in Theorems 1–3, we mean the following. For every $k \geq 1$ , there exist a mapping � satisfying Assumption 1, an initial iterate � , and a noise sequence satisfying the corresponding noise condition, for which E $\left[ \| f ( x _ { k } ) - x _ { k } \| ^ { 2 } \right]$ is lower bounded by the upper bound at time �, up to a constant independent of � (and up to a logarithmic factor under noise condition C with $b \leq 1 / 2 )$ Here, the instance is allowed to depend on �. The corresponding instances are given in Propositions 1 and 2.

Remark 2 (Ergodic bounds). Another object of interest when studying nonexpansive SA is the ergodic residual $\begin{array} { r } { \frac { 1 } { k + 1 } \sum _ { i = 0 } ^ { \bar { k } } \mathbb { E } \left[ \| f ( x _ { i } ) - x _ { i } \| ^ { 2 } \right] } \end{array}$ , or the minimum residual min $_ { 0 \leq i \leq k } \mathbb { E } \left[ \Vert f ( x _ { i } ) - x _ { i } \Vert ^ { 2 } \right] [ 1 5 ]$ . We do not highlight bounds on these quantities separately here, because they follow directly from our last-iterate bounds. Both quantities are bounded by the average of the last-iterate bounds in Theorems 1–3, and hence satisfy the same upper bounds, up to constants.

## 3 Proof Outline for the Upper Bounds

In this section, we present an outline of the proofs of the upper bounds in Theorems 1–3 through a series of lemmas. The complete proofs are given in Appendix A. Throughout this section, we fix a fixed point $x ^ { \star } \in X ^ { \star }$ , and for brevity, we write $r _ { i } : = \mathbb { E } \left[ \| f ( x _ { i } ) - x _ { i } \| ^ { 2 } \right]$ for the mean-square residual at time $i ,$ and

$$
\bar { r } _ { \ell , k } : = \frac { \sum _ { i = \ell } ^ { k } \beta _ { i } r _ { i } } { \sum _ { i = \ell } ^ { k } \beta _ { i } } , \qquad 0 \le \ell \le k ,
$$

for the weighted average of the residuals between times ℓ and �. Note that $\bar { r } _ { k , k } = r _ { k }$ is the last-iterate residual.

The proof consists of two steps. In the first step, we bound the ergodic residual $\bar { r } _ { 0 , k }$ . This follows from a standard Lyapunov drift argument for the squared distance $\| x _ { k } - x ^ { \star } \| ^ { 2 }$ , although some care is needed in handling the noise under each noise condition. In the second step, we convert this ergodic bound into a last-iterate bound on $r _ { k }$ . To do so, we compare the stochastic iterates with a noiseless trajectory restarted from an intermediate iterate. The residual along a noiseless KM trajectory is nonincreasing, and we show that the residuals of the two trajectories remain close in a suitable averaged sense.

The majority of the proof is common to the three noise conditions. The noise condition only enters through the bound on the accumulated noise $\begin{array} { r l } { \sum _ { i } \beta _ { i } ^ { 2 } \mathbb { E } \left[ \| M _ { i + 1 } \| ^ { 2 } \right] } \end{array}$ , which appears in both steps.

## 3.1 Ergodic Bound

The first lemma is a one-step drift inequality for the squared distance of the iterates from $x ^ { \star }$ . It follows from the identity $\| ( 1 - \beta ) a + \beta b \| ^ { 2 } = ( 1 - \beta ) \| a \| ^ { 2 } + \beta \| b \| ^ { 2 } - \beta ( 1 - \beta ) \| a - b \| ^ { 2 }$ , the nonexpansiveness of $f ,$ , and the martingale diference property of the noise.

Lemma 1. For every $k \geq 0 ,$

$$
\begin{array} { r } { \mathbb { E } \left[ \Vert x _ { k + 1 } - x ^ { \star } \Vert ^ { 2 } \right] \leq \mathbb { E } \left[ \Vert x _ { k } - x ^ { \star } \Vert ^ { 2 } \right] - \beta _ { k } ( 1 - \beta _ { k } ) \mathbb { E } \left[ \Vert f ( x _ { k } ) - x _ { k } \Vert ^ { 2 } \right] + \beta _ { k } ^ { 2 } \mathbb { E } \left[ \Vert M _ { k + 1 } \Vert ^ { 2 } \right] . } \end{array}\tag{2}
$$

When summed over $k .$ , the squared distance terms telescope, and dividing by $\textstyle \sum _ { i = 0 } ^ { k } \beta _ { i }$ gives a bound on the ergodic residual $\bar { r } _ { 0 , k }$ in terms of $\| x _ { 0 } - x ^ { \star } \| ^ { 2 }$ and the accumulated noise $\textstyle \sum _ { i = 0 } ^ { k } \beta _ { i } ^ { 2 } \dot { \mathbb { E } } \left[ \| M _ { i + 1 } \| ^ { 2 } \right]$ . The three noise conditions difer only in how the accumulated noise is handled.

• Under noise condition R, the noise term $\sigma ^ { 2 } \beta _ { i } ^ { 2 } r _ { i }$ is absorbed into the drift term $- \beta _ { i } ( 1 - \beta _ { i } ) r _ { i }$ , which is possible since $( 1 + \sigma ^ { 2 } ) \beta _ { i } \leq ( 1 + \sigma ^ { 2 } ) \beta _ { 0 } < 1$ by Assumption 3. The resulting ergodic bound is of the order of $1 / { \textstyle \sum _ { i = 0 } ^ { k } \beta _ { i } }$ , the same as for the noiseless KM iteration.

• Under noise condition C, the noise contributes the additional term $\scriptstyle \sigma ^ { 2 } \sum _ { i = 0 } ^ { k } \beta _ { i } ^ { 2 }$ . For $b > 1 / 2$ , this sum is bounded, and the ergodic bound is of the order of $1 / \textstyle \sum _ { i = 0 } ^ { k } \beta _ { i }$ , as under noise condition R. For $b < 1 / 2$ it grows as $( k + 1 ) ^ { 1 - 2 b }$ and dominates, giving an ergodic bound of the order of $( k + 1 ) ^ { - b }$ , i.e., of the order of the stepsize $\beta _ { k }$

• Under noise condition I, we first show that for $b > 1 / 2$ , the iterates are bounded in expectation, i.e., $\mathbb { E } \left[ \| x _ { k } - x ^ { \star } \| ^ { 2 } \right]$ is bounded uniformly in �. This is the only step where $b > 1 / 2$ is needed, and it cannot be extended to $b \leq 1 / 2$ . Indeed, since $\| f ( x ) - x \| \leq 2 \| x - x ^ { \star } \|$ , a uniform bound on $\mathbb { E } \left[ \| x _ { k } - x ^ { \star } \| ^ { 2 } \right]$ would also give a uniform bound on the residual $r _ { k }$ , while Proposition 3 shows that $r _ { k }$ can diverge for $b \leq 1 / 2$ . Then, the noise second moment is bounded by a constant, and the argument reduces to that under noise condition C.

## Lemma 2. For every $k \geq 0 _ { ; }$ , the following hold.

(a) Under noise condition R,

$$
\bar { r } _ { 0 , k } \leq \frac { \| x _ { 0 } - x ^ { \star } \| ^ { 2 } } { [ 1 - ( 1 + \sigma ^ { 2 } ) \beta _ { 0 } ] \sum _ { i = 0 } ^ { k } \beta _ { i } } .\tag{3}
$$

(b) Under noise condition C,

$$
\bar { r } _ { 0 , k } \leq \frac { 2 \| x _ { 0 } - x ^ { \star } \| ^ { 2 } + 2 \sigma ^ { 2 } \sum _ { i = 0 } ^ { k } \beta _ { i } ^ { 2 } } { \sum _ { i = 0 } ^ { k } \beta _ { i } } .\tag{4}
$$

(c) Let $\textstyle \Gamma _ { 1 } : = \sum _ { i = 0 } ^ { \infty } \beta _ { i } ^ { 2 }$ . Under noise condition I with $1 / 2 \mathit { \Theta } < \mathit { b } < 1$ , there exists a constant $\Gamma _ { 2 } > 0 ;$ independent of �, such that $\mathbb { E } \left[ \| x _ { k } - x ^ { \star } \| ^ { 2 } \right] \leq \Gamma _ { 2 } ,$ , and

$$
\bar { r } _ { 0 , k } \leq \frac { 2 \left( \| x _ { 0 } - x ^ { \star } \| ^ { 2 } + \sigma ^ { 2 } \Gamma _ { 1 } ( 1 + 2 \| x ^ { \star } \| ^ { 2 } + 2 \Gamma _ { 2 } ) \right) } { \sum _ { i = 0 } ^ { k } \beta _ { i } } .\tag{5}
$$

## 3.2 Last-Iterate Bound

The key step in going from the ergodic residual to the last iterate is an auxiliary noiseless iteration, restarted at an arbitrary iterate of the stochastic trajectory. As discussed above, the residual along a noiseless trajectory is nonincreasing. We use this auxiliary iteration to show that the residuals are approximately nonincreasing on average, in the sense that $\bar { r } _ { \ell + 1 , k }$ exceeds $\bar { r } _ { \ell , k }$ by at most a noise-dependent term. For a fixed $\ell \geq 0$ , define the auxiliary iteration

$$
y _ { \ell } = x _ { \ell } , \qquad y _ { k + 1 } = y _ { k } + \beta _ { k } \left( f ( y _ { k } ) - y _ { k } \right) , \quad k \geq \ell .\tag{6}
$$

Since the two trajectories agree at time ℓ, any discrepancy between them after time ℓ is due to the noise injected from time ℓ onward. The following lemma quantifies this discrepancy. Its first inequality is the analogue of Lemma 1 for the diference $x _ { k } - y _ { k }$ . In fact, as the constant sequence $y _ { k } = x ^ { \star }$ is itself a noiseless trajectory, Lemma 1 is the special case of this inequality with $y _ { k } = x ^ { \star }$ . Summing the first inequality, as before, gives the second inequality: the weighted sum of the squared diferences between the residuals of the two trajectories is bounded by the noise injected after time $\ell .$ The final statement shows that the residual along the noiseless trajectory is nonincreasing, and is a consequence of the nonexpansiveness of $f .$

## Lemma 3. For every $k \geq \ell ,$

$$
\begin{array} { r } { \mathbb { E } \left[ \Vert x _ { k + 1 } - y _ { k + 1 } \Vert ^ { 2 } \right] \leq \mathbb { E } \left[ \Vert x _ { k } - y _ { k } \Vert ^ { 2 } \right] - \beta _ { k } ( 1 - \beta _ { k } ) \mathbb { E } \left[ \Vert f ( x _ { k } ) - x _ { k } - f ( y _ { k } ) + y _ { k } \Vert ^ { 2 } \right] + \beta _ { k } ^ { 2 } \mathbb { E } \left[ \Vert M _ { k + 1 } \Vert ^ { 2 } \right] . } \end{array}\tag{7}
$$

and

$$
\sum _ { i = \ell } ^ { k } \beta _ { i } \mathbb { E } \left[ \Vert f ( x _ { i } ) - x _ { i } - f ( y _ { i } ) + y _ { i } \Vert ^ { 2 } \right] \leq 2 \sum _ { i = \ell } ^ { k } \beta _ { i } ^ { 2 } \mathbb { E } \left[ \Vert M _ { i + 1 } \Vert ^ { 2 } \right] .\tag{8}
$$

Moreover,

$$
\| f ( y _ { k } ) - y _ { k } \| \leq \| f ( y _ { \ell } ) - y _ { \ell } \| = \| f ( x _ { \ell } ) - x _ { \ell } \| .
$$

To move from the weighted average $\bar { r } _ { m , k }$ to the last-iterate residual $\bar { r } _ { k , k } = r _ { k }$ , we need to bound how $\bar { r } _ { \ell , k }$ grows as ℓ increases. Removing the first term from the weighted average gives the exact identity

$$
\bar { r } _ { \ell + 1 , k } = \bar { r } _ { \ell , k } + \frac { \beta _ { \ell } } { \sum _ { i = \ell + 1 } ^ { k } \beta _ { i } } \left( \bar { r } _ { \ell , k } - r _ { \ell } \right) .
$$

In the absence of noise, the monotonicity of the residual gives $\bar { r } _ { \ell , k } \le r _ { \ell }$ , and hence the second term is nonpositive. In the presence of noise, this term can be bounded using Lemma 3, which gives the following lemma.

Lemma 4. For any $0 \leq \ell < k ,$

$$
\bar { r } _ { \ell + 1 , k } \leq \bar { r } _ { \ell , k } + 2 \sqrt { 2 } \cdot \frac { \beta _ { \ell } } { \sum _ { i = \ell + 1 } ^ { k } \beta _ { i } } \left( \bar { r } _ { \ell , k } \right) ^ { 1 / 2 } \left( \frac { \sum _ { i = \ell } ^ { k } \beta _ { i } ^ { 2 } \mathbb { E } \left[ \| M _ { i + 1 } \| ^ { 2 } \right] } { \sum _ { i = \ell } ^ { k } \beta _ { i } } \right) ^ { 1 / 2 } .\tag{9}
$$

Fix a starting time $0 \leq m \leq k$ , which we choose later. Applying Lemma 4 successively for $\ell =$ $m , m + 1 , \ldots , k - 1$ and accumulating the error terms gives a bound on $r _ { k }$ in terms of $\bar { r } _ { m , k }$ . Under noise condition R, the error term in Lemma 4 is at most a multiple of $\bar { r } _ { \ell , k }$ itself, and hence the resulting bound takes the form of a product, as in part (a) of the following lemma.

Lemma 5. For any $0 \leq m \leq k ,$

(a) Under noise condition R,

$$
r _ { k } \leq \bar { r } _ { m , k } \prod _ { i = m } ^ { k - 1 } \left[ 1 + 2 \sqrt { 2 } \sigma \frac { \beta _ { i } \sqrt { \beta _ { i } } } { \sum _ { j = i + 1 } ^ { k } \beta _ { j } } \right] .\tag{10}
$$

(b) Under noise condition C with $0 < b < 1$ , or I with $1 / 2 < b < 1$ , there exists a constant $\Gamma _ { 3 } > 0 ;$ independent of � and $k ,$ such that

$$
r _ { k } \leq \Gamma _ { 3 } \left( \bar { r } _ { m , k } + \left( \sum _ { i = m } ^ { k - 1 } \frac { \beta _ { i } \sqrt { \beta _ { i } } } { \sum _ { j = i + 1 } ^ { k } \beta _ { j } } \right) ^ { 2 } \right) .\tag{11}
$$

The choice of the starting time � involves a trade-of. On the one hand, Lemma 5 bounds $r _ { k }$ in terms of $\bar { r } _ { m , k }$ , while Lemma 2 only bounds $\bar { r } _ { 0 , k }$ . Hence, we need $\bar { r } _ { m , k }$ to be of the same order as $\bar { r } _ { 0 , k }$ . Since $r _ { i } \geq 0$ we have $\begin{array} { r } { \bar { r } _ { m , k } \leq \bar { r } _ { 0 , k } \sum _ { i = 0 } ^ { k } \beta _ { i } / \sum _ { i = m } ^ { k } \beta _ { i } } \end{array}$ , and hence this holds as long as the ratio $\textstyle \sum _ { i = 0 } ^ { k } \beta _ { i } / \sum _ { i = m } ^ { k } \beta _ { i }$ is bounded. This rules out � close to �. On the other hand, the error terms in Lemma 5 involve the sum of the ratios $\beta _ { i } ^ { 3 / 2 } / \Sigma _ { j = i + 1 } ^ { k } \beta _ { j }$ over $m \leq i < k$ , which only increases as � decreases, since starting earlier adds terms with larger stepsizes. We take $m = \lceil k / 2 \rceil$ , which balances the two, as shown in the following lemma.

Lemma 6. Let $m = \lceil k / 2 \rceil$ . There exist constants $\Gamma _ { 4 } , \Gamma _ { 5 } > 0$ such that,for every $k \geq 0 ,$

$$
\bar { r } _ { m , k } \leq \Gamma _ { 4 } \bar { r } _ { 0 , k } , a n d \sum _ { i = m } ^ { k - 1 } \frac { \beta _ { i } \sqrt { \beta _ { i } } } { \sum _ { j = i + 1 } ^ { k } \beta _ { j } } \leq \frac { \Gamma _ { 5 } \log ( k + 2 ) } { ( k + 1 ) ^ { b / 2 } } .\tag{12}
$$

Taking $m = \lceil k / 2 \rceil$ and combining Lemmas 2, 5, and 6 gives the upper bounds in Theorems 1–3. In particular, under noise condition R, the product in Lemma 5(a) is bounded by a constant, and hence the last-iterate residual is of the same order as the ergodic residual.

## 4 Optimality of the Upper Bounds

In this section, we present the instances showing that the upper bounds in Theorems 1–3 are optimal, and that Theorem 3 cannot be extended to $b \leq 1 / 2$ . For each result, we first describe the instance, and then discuss what it shows. Throughout this section, the stepsizes satisfy Assumption 3, and the complete proofs are given in Appendix B.

The first two instances are built from rotations. For $\theta \in ( 0 , \pi ]$ , let

$$
R _ { \theta } = \left( \begin{array} { c c } { { \cos \theta } } & { { - \sin \theta } } \\ { { \sin \theta } } & { { \cos \theta } } \end{array} \right)
$$

denote the rotation of $\mathbb { R } ^ { 2 }$ by the angle �. Since $R _ { \theta }$ is orthogonal, the map $x \mapsto R _ { \theta } x$ is nonexpansive, and since det $( R _ { \theta } - I ) = 2 ( 1 - \cos \theta ) > 0$ , its unique fixed point is the origin. Moreover, for every $\boldsymbol { x } \in \mathbb { R } ^ { 2 }$ and $\beta \in [ 0 , 1 ]$

$$
\| R _ { \theta } x - x \| ^ { 2 } = 2 ( 1 - \cos \theta ) \| x \| ^ { 2 } , \qquad \| ( 1 - \beta ) x + \beta R _ { \theta } x \| ^ { 2 } = [ 1 - 2 ( 1 - \cos \theta ) \beta ( 1 - \beta ) ] \| x \| ^ { 2 } .\tag{13}
$$

That is, the squared residual is a fixed fraction $2 ( 1 - \cos \theta )$ of the squared norm, and each noiseless KM step shrinks the squared norm by the factor $1 - 2 ( 1 - \cos \theta ) \beta ( 1 - \beta )$ . A small angle therefore makes the residual small relative to the squared norm, but also makes the iterates contract slowly. Our instances exploit this trade-of by choosing the angle depending on the time horizon �.

## 4.1 Lower Bound without Noise

Example 1 (Slow rotation). Fix $k \geq 1$ . Let $d = 2 , x _ { 0 } = ( 1 , 0 ) , M _ { i + 1 } = 0$ for all $i \geq 0 ,$ , and $f ( x ) = R _ { \theta } x$ , where � is chosen such that

$$
2 ( 1 - \cos \theta ) = \frac { 1 } { 1 + 2 \sum _ { j = 0 } ^ { k - 1 } \beta _ { j } } .
$$

To see why this choice of the angle is natural, write $\gamma = 2 ( 1 - \cos \theta )$ and $\begin{array} { r } { \tau = \sum _ { j = 0 } ^ { k - 1 } \beta _ { j } } \end{array}$ . By (13), the squared residual at time � is $\begin{array} { r } { \gamma \prod _ { j = 0 } ^ { k - 1 } [ 1 - \gamma \beta _ { j } ( 1 - \beta _ { j } ) ] \| x _ { 0 } \| ^ { 2 } } \end{array}$ , which is roughly $\gamma e ^ { - \gamma \tau } \| x _ { 0 } \| ^ { 2 }$ . As a function of $\gamma ,$ this is maximized at $\gamma = 1 / \tau$ , where it equals $\| x _ { 0 } \| ^ { 2 } / ( e \tau )$ . The choice $\gamma = 1 / ( 1 + 2 \tau )$ in Example 1 is of this order, and hence gives a residual of the order of $1 / \tau$ . Since there is no noise, this instance satisfies all three noise conditions in Assumption 2. Example 1 is essentially the instance used in [29] to lower bound the residual of noiseless KM iterations, and we include it here for completeness.

Proposition 1. There exists a constant ${ \mathfrak { c } } _ { 5 } > 0$ such that, for every $k \geq 1$ , the instance in Example 1 satisfies

$$
\mathbb { E } \left[ \Vert f ( x _ { k } ) - x _ { k } \Vert ^ { 2 } \right] \geq \frac { \mathfrak { c } _ { 5 } } { ( k + 1 ) ^ { 1 - b } } .
$$

Proposition 1 shows that the upper bounds in Theorem 1, in Theorem 2 for $b > 1 / 2$ , and in Theorem 3 are optimal.

## 4.2 Lower Bound for Noise Condition C

Example 2 (Rotations at multiple scales). Fix $k \geq 1$ . Let $d = 2 k$ , and group the coordinates of $\mathbb { R } ^ { 2 k }$ into � consecutive pairs, indexed by $p \in \{ 1 , \ldots , k \}$ . Let $x _ { 0 } = 0$ , and let $f ( x ) = \mathrm { d i a g } ( R _ { \theta _ { 1 } } , \dots , R _ { \theta _ { k } } ) x$ act as the rotation $R _ { \theta _ { p } }$ on the �-th pair, where $\theta _ { p }$ is chosen such that

$$
2 ( 1 - \cos \theta _ { p } ) = \frac { 1 } { 1 + 2 \sum _ { j = p } ^ { k - 1 } \beta _ { j } } , \qquad 1 \leq p \leq k .
$$

Let $\{ \varepsilon _ { i } \} _ { i \ge 1 }$ be independent random variables taking the values 1 and −1 with probability $1 / 2$ each. Let $M _ { i + 1 }$ have a single nonzero coordinate, equal to $\sigma \varepsilon _ { i + 1 }$ , in the first coordinate of the $( i + 1 )$ )-th pair for $0 \leq i < k$ and of the first pair for $i \geq k$

Only the noise injected before time � afects $x _ { k }$ . The noise injected from time � onward is defined only so that $\{ M _ { i + 1 } \}$ is a noise sequence for all $i \geq 0$ . The map $f$ is nonexpansive with the origin as its unique fixed point, and the noise satisfies noise condition C with equality. The proof of the following proposition consists of two steps.

• Each pair is a copy ofExample 1. Before time �, the <sub>�</sub>-th pair receives noise only at time $p - 1$ , and is zero up to that time. Hence, from time � onward, it evolves as a copy of Example 1, started with squared norm $\sigma ^ { 2 } \beta _ { p - 1 } ^ { 2 }$ , and with the angle $\theta _ { p }$ tuned to the remaining horizon $\begin{array} { r } { \sum _ { j = p } ^ { k - 1 } \beta _ { j } } \end{array}$ . As discussed after Example 1, this choice of the angle maximizes, up to a constant factor, the residual of the �-th pair at time �, which is then of the order of $\sigma ^ { 2 } \beta _ { p - 1 } ^ { 2 } / ( 1 + 2 \sum _ { j = p } ^ { k - 1 } \beta _ { j } )$ . Since $f$ acts separately on each pair and the pairs are orthogonal, the squared residual at time $\dot { k }$ is the sum of these contributions. That is, the instance attains the worst case for each noise injection simultaneously.

• Evaluating the sum. Since all terms in the sum are nonnegative, we can lower bound it by restricting it to the noise injected at times $k / 2 \leq i < k$ . The restricted sum is of the order of $\sigma ^ { 2 } \beta _ { k }$ log �, i.e., of the order of $\sigma ^ { 2 } \log { k } / k ^ { b }$

Proposition 2. Let $0 < b \leq 1 / 2$ . There exists a constant ${ \mathfrak { c } } _ { 6 } > 0$ such that, for every $k \geq 1$ , the instance in Example 2 satisfies

$$
\mathbb { E } \left[ \Vert f ( x _ { k } ) - x _ { k } \Vert ^ { 2 } \right] \geq \frac { \mathfrak { c } _ { 6 } \sigma ^ { 2 } \log ( k + 1 ) } { ( k + 1 ) ^ { b } } .
$$

Proposition 2 shows that the upper bound in Theorem 2 for $b \leq 1 / 2$ is optimal up to a logarithmic factor.

## 4.3 Divergence under Noise Condition I

Example 3 (Reflection with noise condition I). Let $d = 2$ , write $x = ( u , \nu ) \in \mathbb { R } ^ { 2 }$ , and let $f ( u , \nu ) = ( u , - \nu )$ and $x _ { 0 } = ( 1 , 0 )$ . Writing $\boldsymbol { x } _ { i } = \left( \boldsymbol { u } _ { i } , \boldsymbol { \nu } _ { i } \right)$ , let the two coordinates of $M _ { i + 1 }$ , conditionally on $\mathscr { F } _ { i }$ , independently take the values $\sigma u _ { i } / \sqrt { 2 }$ and $- \sigma u _ { i } / \sqrt { 2 }$ with probability $1 / 2$ each.

In this instance, � is a reflection, and is hence nonexpansive. Its fixed points form the line $\{ ( u , 0 ) : u \in \mathbb { R } \}$ and $\| f ( x ) - x \| ^ { 2 } \ = \ 4 \nu ^ { 2 }$ The noise satisfies $\begin{array} { r } { \mathbb { E } \left[ \| \bar { M } _ { i + 1 } \| ^ { 2 } \vert \mathcal { F } _ { i } \right] = \bar { \sigma ^ { 2 } } u _ { i } ^ { 2 } \le \sigma ^ { 2 } ( 1 + \| x _ { i } \| ^ { 2 } ) } \end{array}$ , and hence noise condition I holds. Along the line of fixed points, the iteration has no drift, and the �-coordinate evolves as $u _ { i + 1 } = u _ { i } + \beta _ { i } M _ { i + 1 } ^ { ( 1 ) }$ . As the noise scales with $u _ { i } ,$ this is a multiplicative random walk, with $\mathbb { E } [ u _ { i + 1 } ^ { 2 } ] = ( 1 + \sigma ^ { 2 } \beta _ { i } ^ { 2 } / 2 ) \mathbb { E } [ u _ { i } ^ { 2 } ]$ . Hence, $\mathbb { E } [ u _ { k } ^ { 2 } ]$ grows as exp $\textstyle { \big ( } \Omega ( \sum _ { i = 0 } ^ { k - 1 } \beta _ { i } ^ { 2 } ) { \big ) }$ , which diverges when $\textstyle \sum _ { i } \beta _ { i } ^ { 2 } = \infty$ Since the noise in the �-coordinate also scales with $u _ { i }$ , this growth is transferred to the residual, up to a polynomial factor.

Proposition 3. The instance in Example 3 satisfies the following.

(a) $H 0 < b < 1 / 2$ , there exist constants ${ \mathfrak { c } } _ { 7 } , { \mathfrak { c } } _ { 8 } > 0$ such that, for all $k \geq 1$

$$
\mathbb { E } \left[ \Vert f ( x _ { k } ) - x _ { k } \Vert ^ { 2 } \right] \geq \frac { \mathfrak { c } _ { 7 } } { ( k + 1 ) ^ { 2 b } } \exp \left( \mathfrak { c } _ { 8 } ( k + 1 ) ^ { 1 - 2 b } \right) .
$$

(b) $H b = 1 / 2$ and $\sigma \beta > { \sqrt { 2 } } ,$ , there exists a constant ${ \mathfrak { c } } _ { 9 } > 0$ such that, for all $k \geq 1$

$$
\begin{array} { r } { \mathbb { E } \left[ \Vert { f ( x _ { k } ) } - x _ { k } \Vert ^ { 2 } \right] \geq { \mathfrak { c } } _ { 9 } ( k + 1 ) ^ { \sigma ^ { 2 } \beta ^ { 2 } / 2 - 1 } . } \end{array}
$$

In particular, in both cases, $\mathbb { E } \left[ \Vert f ( x _ { k } ) - x _ { k } \Vert ^ { 2 } \right] \to \infty a s k \to \infty .$

Proposition 3 shows that Theorem 3 cannot be extended to $b \leq 1 / 2$ . The condition $\sigma \beta > \sqrt { 2 }$ in part (b) is allowed, as noise condition I places no restriction on $\sigma$ . Unlike Examples 1 and 2, this is a single instance that does not depend on �. As discussed above, the divergence is driven by the noise scaling with the iterate along the line of fixed points. Under noise condition C, the �-coordinate would instead perform an additive random walk, whose second moment grows only as $\textstyle \sum _ { i = 0 } ^ { k - 1 } { \beta _ { i } ^ { 2 } }$ , and the residual would still be controlled by Theorem 2. Note that the above result only shows that the mean-square residual diverges, and not that the iterates diverge. In Example 3, the divergence is driven by rare trajectories along which $u _ { k }$ grows, and the iterates may still converge almost surely.

## 5 Applications

In this section, we discuss three applications of our results: stochastic gradient descent for convex functions, linear stochastic approximation, and learning in cocoercive games. For each application, our focus is on what the noise conditions in Assumption 2 mean, and on where they arise. Noise condition C is the most common noise model, and arises whenever the noisy evaluations are obtained from samples with bounded variance, such as stochastic gradients with bounded variance or noisy observations of a fixed quantity. In contrast, whether noise conditions R and I hold depends on the structure of the problem, and we discuss these conditions wherever applicable.

Root-finding formulation. Many applications of nonexpansive SA arise in the root-finding formulation, where the goal is to find a zero of an operator $F : \mathbb { R } ^ { d }  \bar { \mathbb { R } } ^ { d }$ using noisy evaluations $F ( x _ { k } ) + \zeta _ { k + 1 }$ , where $\mathbb { E } [ \zeta _ { k + 1 } \mid \mathcal { F } _ { k } ] = 0$ , via the iteration

$$
x _ { k + 1 } = x _ { k } - \beta _ { k } ^ { \prime } \left( F ( x _ { k } ) + \zeta _ { k + 1 } \right) .\tag{14}
$$

Fix $\eta > 0$ , and write $\beta _ { k } ^ { \prime } = \eta \beta _ { k }$ . Then, (14) can be rewritten as

$$
x _ { k + 1 } = x _ { k } + \beta _ { k } \left( f ( x _ { k } ) - x _ { k } + M _ { k + 1 } \right) , \qquad { \mathrm { w h e r e ~ } } f ( x ) = x - \eta F ( x ) { \mathrm { ~ a n d ~ } } M _ { k + 1 } = - \eta \zeta _ { k + 1 } ,
$$

which is iteration (1). The zeros of � are exactly the fixed points of $f .$ . Moreover, the residual is $\| f ( x _ { k } ) - x _ { k } \| ^ { 2 } = \eta ^ { 2 } \| F ( x _ { k } ) \| ^ { 2 }$ , and hence our results give last-iterate bounds on $\mathbb { E } \left[ \| F ( x _ { k } ) \| ^ { 2 } \right]$ . The fixedpoint formulation is nonexpansive when $F$ is cocoercive. An operator � is said to be $( 1 / L )$ -cocoercive if

$$
\langle F ( x ) - F ( y ) , x - y \rangle \geq \frac { 1 } { L } \| F ( x ) - F ( y ) \| ^ { 2 } , \qquad \forall x , y \in \mathbb { R } ^ { d } .
$$

If � is $( 1 / L )$ -cocoercive and has at least one zero, and $0 < \eta \leq 2 / L$ , then $f$ satisfies Assumption 1. Indeed, for all $x , y \in \mathbb { R } ^ { d }$

$$
\begin{array} { r l } & { \| f ( x ) - f ( y ) \| ^ { 2 } = \| x - y \| ^ { 2 } - 2 \eta \langle F ( x ) - F ( y ) , x - y \rangle + \eta ^ { 2 } \| F ( x ) - F ( y ) \| ^ { 2 } } \\ & { \qquad \le \| x - y \| ^ { 2 } - \eta \left( \displaystyle \frac { 2 } { L } - \eta \right) \| F ( x ) - F ( y ) \| ^ { 2 } \le \| x - y \| ^ { 2 } . } \end{array}
$$

All three applications below are of this form.

## 5.1 Stochastic Gradient Descent for Convex Functions

Consider the problem of minimizing $F _ { 0 } ( x ) = \mathbb { E } _ { \xi } [ \ell ( x ; \xi ) ]$ , where ℓ is a loss function and $\xi$ is a random sample. Suppose that $F _ { 0 } : \mathbb { R } ^ { d } $ R is convex and �-smooth, and its set of minimizers is nonempty. In this case, $\nabla F _ { 0 } \mathrm { i s } \left( 1 / L \right)$ -cocoercive [27, Theorem 2.1.5]. Stochastic gradient descent (SGD) is given by

$$
\begin{array} { r } { x _ { k + 1 } = x _ { k } - \beta _ { k } ^ { \prime } \nabla \ell ( x _ { k } ; \xi _ { k + 1 } ) , } \end{array}
$$

where $\xi _ { 1 } , \xi _ { 2 } , \ldots$ . are i.i.d. samples. Since $\nabla F _ { 0 } ( x ) = \mathbb { E } _ { \xi } [ \nabla \ell ( x ; \xi ) ]$ , this is of the form (14) with $F = \nabla F _ { 0 }$ and $\zeta _ { k + 1 } = \nabla \ell ( x _ { k } ; \xi _ { k + 1 } ) - \nabla F _ { 0 } ( x _ { k } )$ , and our results give last-iterate bounds on $\mathbb { E } \left[ \Vert \nabla F _ { 0 } ( x _ { k } ) \Vert ^ { 2 } \right]$

• Noise condition R is equivalent to the strong growth condition [34],

$$
\begin{array} { r } { \mathbb { E } _ { \xi } \left[ \| \nabla \ell ( x ; \xi ) \| ^ { 2 } \right] \leq \rho \| \nabla F _ { 0 } ( x ) \| ^ { 2 } \quad \mathrm { f o r ~ a l l ~ } x \in \mathbb { R } ^ { d } , } \end{array}
$$

with $\sigma ^ { 2 } = \rho - 1$ . That is, the gradients of the individual losses are small whenever their average is small. This holds, for example, under the stronger condition $\begin{array} { r } { \operatorname* { s u p } _ { \xi } \| \nabla \ell ( x ; \xi ) \| \le B \| \nabla F _ { 0 } ( x ) \| } \end{array}$ for some $B \geq 0$ , as previously considered in [30].

• Noise condition I is much more general than noise condition R, as it does not require the stochastic gradients to vanish at the minimizers. For example, it holds when, for some $L ^ { \prime } , G \ge 0 , \nabla \ell ( \cdot ; \xi )$ is �<sup>′</sup>-Lipschitz and $\| \nabla \ell ( 0 ; \xi ) \| \le G$ for every �. In this case, $\| \nabla \ell ( x ; \xi ) \| \le G + L ^ { \prime } \| x \|$ , and hence the stochastic gradients grow at most linearly in ∥�∥.

## 5.2 Linear Stochastic Approximation

Consider the problem of solving $A x = b$ , where the system has at least one solution, $A \in \mathbb { R } ^ { d \times d }$ and $b \in \mathbb { R } ^ { d }$ and � is $( 1 / L )$ -cocoercive, i.e., $\begin{array} { r } { \langle x , A x \rangle \geq \frac { 1 } { L } \| A x \| ^ { 2 } } \end{array}$ for all �. For example, this holds for any symmetric positive semidefinite matrix � with largest eigenvalue at most � [18, p. 79]. In particular, � may be singular, in which case there are multiple solutions. At each time �, we observe conditionally unbiased estimates $\tilde { A } _ { k + 1 }$ and $\tilde { b } _ { k + 1 }$ of � and �, and update

$$
x _ { k + 1 } = x _ { k } + \beta _ { k } ^ { \prime } \left( \tilde { b } _ { k + 1 } - \tilde { A } _ { k + 1 } x _ { k } \right) .
$$

This is of the form (14) with $F ( x ) = A x - b$ and $\zeta _ { k + 1 } = ( \tilde { A } _ { k + 1 } - A ) x _ { k } - ( \tilde { b } _ { k + 1 } - b )$ , and our results give last-iterate bounds on $\mathbb { E } \left[ \| A x _ { k } - b \| ^ { 2 } \right]$ Noise condition I is the standard noise condition for linear SA [12, 32]. It holds whenever $\tilde { A } _ { k + 1 }$ and $\tilde { b } _ { k + 1 }$ have bounded conditional second moments, since the noise term $( \tilde { A } _ { k + 1 } - A ) x _ { k }$ grows linearly in $x _ { k }$

## 5.3 Learning in Cocoercive Games

Consider an �-player game, where each player $n \in \{ 1 , \ldots , N \}$ chooses an action $x _ { n } \in \mathbb { R } ^ { d }$ and incurs a cost $J _ { n } ( \mathbf { x } )$ . Here, $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { N } ) \in \mathbb { R } ^ { N d }$ denotes the concatenation of the actions of all players, and $J _ { n }$ is convex and diferentiable in the player’s own action $x _ { n }$ . Let

$$
F ( \mathbf { x } ) = \left( \nabla _ { x _ { 1 } } J _ { 1 } ( \mathbf { x } ) , \ldots , \nabla _ { x _ { N } } J _ { N } ( \mathbf { x } ) \right) \in \mathbb { R } ^ { N d }
$$

denote the game gradient operator. The Nash equilibria of the game are the zeros of �. When � is $( 1 / L )$ )-cocoercive, the game is called a cocoercive game [10, 25]. We assume that the game is cocoercive and has at least one Nash equilibrium. In stochastic gradient play, each player updates its action using a noisy estimate $\tilde { F } _ { n , k + 1 }$ of its own gradient $\nabla _ { x _ { n } } J _ { n } ( \mathbf { x } _ { k } )$ ,

$$
x _ { n , k + 1 } = x _ { n , k } - \beta _ { k } ^ { \prime } \tilde { F } _ { n , k + 1 } , \qquad n = 1 , \dots , N .
$$

Stacking the updates of all players, we get

$$
\begin{array} { r } { \mathbf { x } _ { k + 1 } = \mathbf { x } _ { k } - \beta _ { k } ^ { \prime } \tilde { F } _ { k + 1 } , } \end{array}
$$

where $\mathbf { x } _ { k } \ = \ ( x _ { 1 , k } , \ldots , x _ { N , k } ) \ \in \ \mathbb { R } ^ { N d }$ and $\tilde { F } _ { k + 1 } = ( \tilde { F } _ { 1 , k + 1 } , \ldots , \tilde { F } _ { N , k + 1 } )$ . This is of the form (14) with $\zeta _ { k + 1 } = \tilde { F } _ { k + 1 } - F ( \mathbf { x } _ { k } )$ , and our results give last-iterate bounds on $\mathbb { E } \left[ \Vert F ( \mathbf { x } _ { k } ) \Vert ^ { 2 } \right]$ . As the game may have multiple Nash equilibria, the residual is the natural measure of convergence in this setting.

• Noise condition R holds when players update at random times. Suppose each player updates at time � with probability $q \in ( 0 , 1 )$ , independently of the other players and of the past, and scales its exact gradient by $1 / q$ when it updates. Then, $\tilde { F } _ { k + 1 }$ is conditionally unbiased, and $\mathbb { E } \left[ \Vert \zeta _ { k + 1 } \Vert ^ { 2 } | \mathcal { F } _ { k } \right] = ( 1 / q - 1 ) \Vert F ( \mathbf { x } _ { k } ) \Vert ^ { 2 }$

• Noise condition I holds when the costs depend on random data, as in linear SA. For example, in quadratic games where the cost matrices are observed only through unbiased samples, the noise grows linearly in the actions.

## 6 Conclusion

In this paper, we studied nonexpansive SA under the Euclidean norm. We obtained last-iterate mean-square residual bounds under three conditions on the conditional second moment of the noise. We also obtained matching lower bounds, up to logarithmic factors, and further showed that the mean-square residual may diverge under iterate-dependent noise if the stepsizes are not square-summable. Our analysis relies on a noiseless auxiliary iteration, which we use to convert an ergodic residual bound into a last-iterate bound. Future directions include obtaining optimal last-iterate rates under general norms and for two-time-scale SA with nonexpansive mappings, where it remains open whether the existing bounds [1, 4, 7, 29] can be improved.

## Disclosure of AI Use

The authors formulated the problem and the three noise conditions, wrote preliminary versions of the proofs of the upper bounds, and conjectured that, under noise condition I, square-summable stepsizes are necessary for convergence. The large language models GPT-6 Astra (OpenAI) and Claude Opus 5.5 (Anthropic) were then used to strengthen the upper bounds, and to construct the instances in Section 4. The tools were also used to assist with writing and editing the paper. The authors have checked all results, proofs, and references, and take full responsibility for the content of this paper.

## A Proofs of the Upper Bounds

Throughout this appendix, $x ^ { \star } \in \mathcal { X } ^ { \star }$ is fixed, and we use the notation $r _ { i }$ and $\bar { r } _ { \ell , k }$ introduced in Section 3. We also repeatedly use the identity

$$
\begin{array} { r } { \| ( 1 - \beta ) a + \beta b \| ^ { 2 } = ( 1 - \beta ) \| a \| ^ { 2 } + \beta \| b \| ^ { 2 } - \beta ( 1 - \beta ) \| a - b \| ^ { 2 } , \qquad a , b \in \mathbb { R } ^ { d } , \beta \in \mathbb { R } . } \end{array}\tag{15}
$$

## A.1 Proof of Lemma 1

Subtracting $x ^ { \star }$ from both sides of (1) and using $f ( x ^ { \star } ) = x ^ { \star }$ , we get

$$
x _ { k + 1 } - x ^ { \star } = ( 1 - \beta _ { k } ) \bigl ( x _ { k } - x ^ { \star } \bigr ) + \beta _ { k } \bigl ( f ( x _ { k } ) - f ( x ^ { \star } ) \bigr ) + \beta _ { k } M _ { k + 1 } .
$$

Applying (15) with $a = x _ { k } - x ^ { \star }$ and $b = f ( x _ { k } ) - f ( x ^ { \star } )$ , we have

$$
\begin{array} { r l } & { \| ( 1 - \beta _ { k } ) ( x _ { k } - x ^ { \star } ) + \beta _ { k } ( f ( x _ { k } ) - f ( x ^ { \star } ) ) \| ^ { 2 } = ( 1 - \beta _ { k } ) \| x _ { k } - x ^ { \star } \| ^ { 2 } + \beta _ { k } \| f ( x _ { k } ) - f ( x ^ { \star } ) \| ^ { 2 } } \\ & { \qquad - \beta _ { k } ( 1 - \beta _ { k } ) \| f ( x _ { k } ) - x _ { k } \| ^ { 2 } } \\ & { \qquad \leq \| x _ { k } - x ^ { \star } \| ^ { 2 } - \beta _ { k } ( 1 - \beta _ { k } ) \| f ( x _ { k } ) - x _ { k } \| ^ { 2 } . } \end{array}
$$

Here, the inequality follows from the nonexpansiveness of $f .$ Expanding the square after adding $\beta _ { k } M _ { k + 1 }$ , we get

$$
\begin{array} { r l } & { \| x _ { k + 1 } - x ^ { \star } \| ^ { 2 } \leq \| x _ { k } - x ^ { \star } \| ^ { 2 } - \beta _ { k } ( 1 - \beta _ { k } ) \| f ( x _ { k } ) - x _ { k } \| ^ { 2 } + \beta _ { k } ^ { 2 } \| M _ { k + 1 } \| ^ { 2 } } \\ & { \qquad + 2 \beta _ { k } \langle ( 1 - \beta _ { k } ) ( x _ { k } - x ^ { \star } ) + \beta _ { k } ( f ( x _ { k } ) - x ^ { \star } ) , M _ { k + 1 } \rangle . } \end{array}
$$

Since $x _ { k }$ is $\mathcal { F } _ { k }$ -measurable and $\{ M _ { k + 1 } \}$ is a martingale diference sequence with respect to $\mathcal { F } _ { k }$ , the conditional expectation of the last term given $\mathcal { F } _ { k }$ is zero. Taking expectations completes the proof of Lemma 1.

## A.2 Proof of Lemma 2

For part (a), taking expectations in noise condition R gives E $\left[ \Vert M _ { i + 1 } \Vert ^ { 2 } \right] \leq \sigma ^ { 2 } r _ { i }$ . Substituting this bound in Lemma 1, we get

$$
\begin{array} { r } { \mathbb { E } \left[ \Vert x _ { i + 1 } - x ^ { \star } \Vert ^ { 2 } \right] \leq \mathbb { E } \left[ \Vert x _ { i } - x ^ { \star } \Vert ^ { 2 } \right] - \beta _ { i } \left[ 1 - ( 1 + \sigma ^ { 2 } ) \beta _ { i } \right] r _ { i } \leq \mathbb { E } \left[ \Vert x _ { i } - x ^ { \star } \Vert ^ { 2 } \right] - [ 1 - ( 1 + \sigma ^ { 2 } ) \beta _ { 0 } ] \beta _ { i } r _ { i } . } \end{array}
$$

Here, the second inequality follows from the fact that the stepsizes are nonincreasing. Summing from $i = 0$ to �, the squared distance terms telescope, and we get

$$
[ 1 - ( 1 + \sigma ^ { 2 } ) \beta _ { 0 } ] \left( \sum _ { i = 0 } ^ { k } \beta _ { i } \right) \bar { r } _ { 0 , k } = [ 1 - ( 1 + \sigma ^ { 2 } ) \beta _ { 0 } ] \sum _ { i = 0 } ^ { k } \beta _ { i } r _ { i } \le \| x _ { 0 } - x ^ { \star } \| ^ { 2 } - \mathbb { B } \left[ \| x _ { k + 1 } - x ^ { \star } \| ^ { 2 } \right] \le \| x _ { 0 } - x ^ { \star } \| ^ { 2 } .
$$

Since $( 1 + \sigma ^ { 2 } ) \beta _ { 0 } < 1$ , dividing both sides by $\begin{array} { r } { \left[ 1 - ( 1 + \sigma ^ { 2 } ) \beta _ { 0 } \right] \sum _ { i = 0 } ^ { k } \beta _ { i } } \end{array}$ completes the proof for part (a).

For part (b), noise condition C gives $\mathbb { E } \left[ \lVert M _ { i + 1 } \rVert ^ { 2 } \right] \leq \sigma ^ { 2 }$ . Using this bound and $1 - \beta _ { i } \ge 1 / 2$ in Lemma 1, we get

$$
\frac { 1 } { 2 } \beta _ { i } r _ { i } \le \mathbb { E } \left[ \left\| x _ { i } - x ^ { \star } \right\| ^ { 2 } \right] - \mathbb { E } \left[ \left\| x _ { i + 1 } - x ^ { \star } \right\| ^ { 2 } \right] + \sigma ^ { 2 } \beta _ { i } ^ { 2 } .
$$

Summing from $i = 0$ to � as before gives $\begin{array} { r } { \frac 1 2 \left( \sum _ { i = 0 } ^ { k } \beta _ { i } \right) \bar { r } _ { 0 , k } \leq \| x _ { 0 } - x ^ { \star } \| ^ { 2 } + \sigma ^ { 2 } \sum _ { i = 0 } ^ { k } \beta _ { i } ^ { 2 } } \end{array}$ , which completes the proof for part (b).

For part (c), using $\| x _ { i } \| ^ { 2 } \leq 2 \| x _ { i } - x ^ { \star } \| ^ { 2 } + 2 \| x ^ { \star } \| ^ { 2 }$ in noise condition I, we get

$$
\begin{array} { r } { \mathbb { E } \left[ \Vert M _ { i + 1 } \Vert ^ { 2 } \right] \leq \sigma ^ { 2 } \left( 1 + \mathbb { E } \left[ \Vert x _ { i } \Vert ^ { 2 } \right] \right) \leq \sigma ^ { 2 } \left( 1 + 2 \Vert x ^ { \star } \Vert ^ { 2 } + 2 \mathbb { E } \left[ \Vert x _ { i } - x ^ { \star } \Vert ^ { 2 } \right] \right) . } \end{array}\tag{16}
$$

Substituting this bound in Lemma 1 and dropping the residual term, we get

$$
\begin{array} { r } { \mathbb { E } \left[ \| x _ { i + 1 } - x ^ { \star } \| ^ { 2 } \right] \leq ( 1 + 2 \sigma ^ { 2 } \beta _ { i } ^ { 2 } ) \mathbb { E } \left[ \| x _ { i } - x ^ { \star } \| ^ { 2 } \right] + \sigma ^ { 2 } ( 1 + 2 \| x ^ { \star } \| ^ { 2 } ) \beta _ { i } ^ { 2 } . } \end{array}
$$

Recall that $\begin{array} { r } { \Gamma _ { 1 } = \sum _ { i = 0 } ^ { \infty } \beta _ { i } ^ { 2 } } \end{array}$ , which is finite since $b > 1 / 2$ . Unrolling the above recursion, for all $k \geq 0$

$$
\begin{array} { r l } & { \displaystyle \mathbb { B } \left[ \| x _ { k } - x ^ { \star } \| ^ { 2 } \right] \leq \| x _ { 0 } - x ^ { \star } \| ^ { 2 } \prod _ { i = 0 } ^ { k - 1 } ( 1 + 2 \sigma ^ { 2 } \beta _ { i } ^ { 2 } ) + \sigma ^ { 2 } ( 1 + 2 \| x ^ { \star } \| ^ { 2 } ) \sum _ { i = 0 } ^ { k - 1 } \beta _ { i } ^ { 2 } \prod _ { j = i + 1 } ^ { k - 1 } ( 1 + 2 \sigma ^ { 2 } \beta _ { j } ^ { 2 } ) } \\ & { \qquad \leq e ^ { 2 \sigma ^ { 2 } \Gamma _ { 1 } } \left( \| x _ { 0 } - x ^ { \star } \| ^ { 2 } + \sigma ^ { 2 } ( 1 + 2 \| x ^ { \star } \| ^ { 2 } ) \Gamma _ { 1 } \right) = : \Gamma _ { 2 } . } \end{array}
$$

Here, the second inequality follows from $1 + u \leq e ^ { u }$ . This proves the bound on the iterates in part (c). Substituting it in (16) gives $\begin{array} { r } { \mathbb { E } \left[ \| M _ { i + 1 } \| ^ { 2 } \right] \leq \sigma ^ { 2 } ( 1 + 2 \| x ^ { \star } \| ^ { 2 } + 2 \Gamma _ { 2 } ) } \end{array}$ for all $i \geq 0$ . The rest of the proof is identical to that of part (b), with $\sigma ^ { 2 }$ replaced by $\sigma ^ { 2 } ( 1 + 2 \| x ^ { \star } \| ^ { 2 } + 2 \Gamma _ { 2 } )$ , and using $\begin{array} { r } { \sum _ { i = 0 } ^ { k } \beta _ { i } ^ { 2 } \le \Gamma _ { 1 } } \end{array}$ . This completes the proof of Lemma 2.

## A.3 Proof of Lemma 3

Throughout this proof and the next, we write $\Delta _ { i } : = f ( x _ { i } ) - x _ { i } - f ( y _ { i } ) + y _ { i }$ . Subtracting (6) from (1), for $k \geq \ell$

$$
x _ { k + 1 } - y _ { k + 1 } = ( 1 - \beta _ { k } ) ( x _ { k } - y _ { k } ) + \beta _ { k } ( f ( x _ { k } ) - f ( y _ { k } ) ) + \beta _ { k } M _ { k + 1 } .
$$

Applying (15) with $a = x _ { k } - y _ { k }$ and $b = f ( x _ { k } ) - f ( y _ { k } )$ , and using the nonexpansiveness of � as in the proof of Lemma 1, we get

$$
\begin{array} { r } { \| ( 1 - \beta _ { k } ) ( x _ { k } - y _ { k } ) + \beta _ { k } ( f ( x _ { k } ) - f ( y _ { k } ) ) \| ^ { 2 } \leq \| x _ { k } - y _ { k } \| ^ { 2 } - \beta _ { k } ( 1 - \beta _ { k } ) \| \Delta _ { k } \| ^ { 2 } . } \end{array}
$$

Since $y _ { k }$ is a deterministic function of $x _ { \ell } .$ , it is $\mathcal { F } _ { k }$ -measurable. Hence, expanding $\| x _ { k + 1 } - y _ { k + 1 } \| ^ { 2 }$ and taking expectations as in the proof of Lemma 1 proves (7).

Next, summing (7) from $i = \ell$ to $k .$ , the squared distance terms telescope. Since $y _ { \ell } = x _ { \ell }$ , we get

$$
\sum _ { i = \ell } ^ { k } \beta _ { i } ( 1 - \beta _ { i } ) \mathbb { E } \left[ \left. \Delta _ { i } \right. ^ { 2 } \right] \leq \sum _ { i = \ell } ^ { k } \beta _ { i } ^ { 2 } \mathbb { E } \left[ \left. M _ { i + 1 } \right. ^ { 2 } \right] - \mathbb { E } \left[ \left. x _ { k + 1 } - y _ { k + 1 } \right. ^ { 2 } \right] \leq \sum _ { i = \ell } ^ { k } \beta _ { i } ^ { 2 } \mathbb { E } \left[ \left. M _ { i + 1 } \right. ^ { 2 } \right] .
$$

Using $1 - \beta _ { i } \ge 1 / 2$ proves (8).

Finally, (6) gives $y _ { k + 1 } - y _ { k } = \beta _ { k } ( f ( y _ { k } ) - y _ { k } )$ , and hence $f ( y _ { k + 1 } ) - y _ { k + 1 } = f ( y _ { k + 1 } ) - f ( y _ { k } ) + ( 1 -$ $\beta _ { k } ) ( f ( y _ { k } ) - y _ { k } )$ . Then,

$$
\| f ( y _ { k + 1 } ) - y _ { k + 1 } \| \leq \| y _ { k + 1 } - y _ { k } \| + ( 1 - \beta _ { k } ) \| f ( y _ { k } ) - y _ { k } \| = \| f ( y _ { k } ) - y _ { k } \| .
$$

Here, the inequality follows from the triangle inequality and the nonexpansiveness of $f .$ Iterating from $k = \ell$ and using $y _ { \ell } = x _ { \ell }$ completes the proof of Lemma 3.

## A.4 Proof of Lemma 4

Fix $0 \leq \ell < k$ , and let $\{ y _ { i } \} _ { i \geq \ell }$ be the auxiliary iteration (6) restarted at time ℓ. For $i \geq \ell ,$ , note that $f ( y _ { i } ) - y _ { i } = ( f ( x _ { i } ) - x _ { i } ) - \Delta _ { i }$ . Then, by Lemma 3,

$$
\begin{array} { r l } & { \| f ( x _ { \ell } ) - x _ { \ell } \| ^ { 2 } \geq \| f ( y _ { i } ) - y _ { i } \| ^ { 2 } = \| f ( x _ { i } ) - x _ { i } \| ^ { 2 } - 2 \langle f ( x _ { i } ) - x _ { i } , \Delta _ { i } \rangle + \| \Delta _ { i } \| ^ { 2 } } \\ & { \qquad \geq \| f ( x _ { i } ) - x _ { i } \| ^ { 2 } - 2 \langle f ( x _ { i } ) - x _ { i } , \Delta _ { i } \rangle . } \end{array}
$$

Taking expectations, multiplying by $\beta _ { i }$ , and summing from $i = \ell$ to $k ,$ , we get

$$
\begin{array} { r l r } {  { ( \sum _ { i = \ell } ^ { k } \beta _ { i } ) ( \bar { r } _ { \ell , k } - r _ { \ell } ) \le 2 \sum _ { i = \ell } ^ { k } \beta _ { i } \mathbb { E } [  f ( x _ { i } ) - x _ { i } , \Delta _ { i }  ] \le 2 ( \displaystyle \sum _ { i = \ell } ^ { k } \beta _ { i } r _ { i } ) ^ { 1 / 2 } ( \displaystyle \sum _ { i = \ell } ^ { k } \beta _ { i } \mathbb { E } [ \| \Delta _ { i } \| ^ { 2 } ] ) ^ { 1 / 2 } } } \\ & { } & { \le 2 \sqrt { 2 } ( \displaystyle \sum _ { i = \ell } ^ { k } \beta _ { i } ) ^ { 1 / 2 } ( \bar { r } _ { \ell , k } ) ^ { 1 / 2 } ( \displaystyle \sum _ { i = \ell } ^ { k } \beta _ { i } ^ { 2 } \mathbb { E } [ \| M _ { i + 1 } \| ^ { 2 } ] ) ^ { 1 / 2 } . } \end{array}
$$

Here, the second inequality follows from the Cauchy–Schwarz inequality, first for the expectation and then for the weighted sum, and the third inequality follows from (8). Dividing by $\textstyle \sum _ { i = \ell } ^ { k } \beta _ { i }$ , we get

$$
\bar { r } _ { \ell , k } - r _ { \ell } \leq 2 \sqrt 2 \left( \bar { r } _ { \ell , k } \right) ^ { 1 / 2 } \left( \frac { \sum _ { i = \ell } ^ { k } \beta _ { i } ^ { 2 } \mathbb { E } \left[ \| M _ { i + 1 } \| ^ { 2 } \right] } { \sum _ { i = \ell } ^ { k } \beta _ { i } } \right) ^ { 1 / 2 } .\tag{17}
$$

Next, splitting of the first term of the weighted sum gives $\begin{array} { r } { \left( \sum _ { i = \ell } ^ { k } \beta _ { i } \right) \bar { r } _ { \ell , k } = \beta _ { \ell } r _ { \ell } + \left( \sum _ { i = \ell + 1 } ^ { k } \beta _ { i } \right) \bar { r } _ { \ell + 1 , k } } \end{array}$ . Writing $\begin{array} { r } { \sum _ { i = \ell } ^ { k } \beta _ { i } = \beta _ { \ell } + \sum _ { i = \ell + 1 } ^ { k } \beta _ { i } } \end{array}$ and rearranging, we get

$$
\bar { r } _ { \ell + 1 , k } = \bar { r } _ { \ell , k } + \frac { \beta _ { \ell } } { \sum _ { i = \ell + 1 } ^ { k } \beta _ { i } } \left( \bar { r } _ { \ell , k } - r _ { \ell } \right) .
$$

Substituting (17) completes the proof of Lemma 4.

## A.5 Proof of Lemma 5

Fix $0 \leq m \leq k$ . When $m = k$ , we have $r _ { k } = \bar { r } _ { k , k }$ , and hence part (a) holds with equality, and part (b) holds for any $\Gamma _ { 3 } \geq 1$ . We henceforth assume that $m < k$ , and write

$$
c _ { i } : = \frac { \beta _ { i } \sqrt { \beta _ { i } } } { \sum _ { j = i + 1 } ^ { k } \beta _ { j } } , \quad m \leq i < k , \qquad \mathrm { a n d ~ } \qquad S : = \sum _ { i = m } ^ { k - 1 } c _ { i } > 0 .
$$

For part (a), taking expectations in noise condition R gives $\mathbb { E } \left[ \| M _ { j + 1 } \| ^ { 2 } \right] \leq \sigma ^ { 2 } r _ { j }$ . Since the stepsizes are nonincreasing, for $m \leq i < k$

$$
\sum _ { j = i } ^ { k } \beta _ { j } ^ { 2 } \mathbb { E } \left[ \left. M _ { j + 1 } \right. ^ { 2 } \right] \leq \sigma ^ { 2 } \beta _ { i } \sum _ { j = i } ^ { k } \beta _ { j } r _ { j } = \sigma ^ { 2 } \beta _ { i } \left( \sum _ { j = i } ^ { k } \beta _ { j } \right) \bar { r } _ { i , k } .
$$

Substituting this bound in Lemma 4 with $\ell = i ,$ we get ${ \bar { r } } _ { i + 1 , k } \leq \left( 1 + 2 \sqrt { 2 } \sigma c _ { i } \right) { \bar { r } } _ { i , k }$ for $m \leq i < k$ . Iterating this inequality from $i = m$ to $k - 1$ and using $\bar { r } _ { k , k } = r _ { k }$ completes the proof for part (a).

For part (b), suppose that $\mathbb { E } \left[ \lVert M _ { i + 1 } \rVert ^ { 2 } \right] \leq \bar { \sigma } ^ { 2 }$ for all $i \geq 0$ , for some $\bar { \sigma } > 0$ . Under noise condition C, this holds with $\bar { \sigma } ^ { 2 } = \sigma ^ { \bar { 2 } }$ . Under noise condition I with $1 / 2 < b < 1$ , Lemma 2(c) and (16) show that this holds with $\bar { \sigma } ^ { 2 } = \sigma ^ { 2 } ( 1 + 2 \| x ^ { \star } \| ^ { 2 } + 2 \Gamma _ { 2 } )$ . Since the stepsizes are nonincreasing, $\begin{array} { r } { \sum _ { j = i } ^ { k } \beta _ { j } ^ { 2 } \mathbb { E } \left[ \| M _ { j + 1 } \| ^ { 2 } \right] \leq \bar { \sigma } ^ { 2 } \beta _ { i } \sum _ { j = i } ^ { k } \beta _ { j } } \end{array}$ Substituting this bound in Lemma 4 with $\ell = i ,$ and using the AM–GM inequality, we get

$$
\bar { r } _ { i + 1 , k } \leq \bar { r } _ { i , k } + 2 \sqrt { 2 } \bar { \sigma } c _ { i } \left( \bar { r } _ { i , k } \right) ^ { 1 / 2 } \leq \left( 1 + \frac { c _ { i } } { S } \right) \bar { r } _ { i , k } + 2 \bar { \sigma } ^ { 2 } c _ { i } S , \qquad m \leq i < k .
$$

Iterating this inequality from $i = m$ to $k - 1$ , and bounding the products multiplying the additive terms by the full product, as every multiplicative factor is at least one, we get

$$
r _ { k } = \bar { r } _ { k , k } \leq \prod _ { i = m } ^ { k - 1 } \left( 1 + \frac { c _ { i } } { S } \right) \left( \bar { r } _ { m , k } + 2 \bar { \sigma } ^ { 2 } S \sum _ { i = m } ^ { k - 1 } c _ { i } \right) \leq e \left( \bar { r } _ { m , k } + 2 \bar { \sigma } ^ { 2 } S ^ { 2 } \right) .
$$

Here, the second inequality follows from $1 + u \leq e ^ { u }$ and $\textstyle \sum _ { i = m } ^ { k - 1 } c _ { i } = S$ . This completes the proof for part (b) with $\Gamma _ { 3 } = e \operatorname* { m a x } \{ 1 , 2 \bar { \sigma } ^ { 2 } \}$

## A.6 Proof of Lemma 6

Recall that $m = \lceil k / 2 \rceil$ and $\beta _ { i } = \beta / ( i + K _ { 0 } ) ^ { b }$ . Since $r _ { i } \geq 0$ for all $i ,$

$$
\bar { r } _ { m , k } = \frac { \sum _ { i = m } ^ { k } \beta _ { i } r _ { i } } { \sum _ { i = m } ^ { k } \beta _ { i } } \leq \frac { \sum _ { i = 0 } ^ { k } \beta _ { i } r _ { i } } { \sum _ { i = m } ^ { k } \beta _ { i } } = \frac { \sum _ { i = 0 } ^ { k } \beta _ { i } } { \sum _ { i = m } ^ { k } \beta _ { i } } \bar { r } _ { 0 , k } .
$$

Hence, it sufices to bound the ratio of the stepsize sums. Since $k - m + 1 \geq ( k + 1 ) / 2$ and the stepsizes are nonincreasing,

$$
\sum _ { i = m } ^ { k } \beta _ { i } \geq ( k - m + 1 ) \beta _ { k } \geq \frac { ( k + 1 ) \beta } { 2 ( k + K _ { 0 } ) ^ { b } } .
$$

Moreover, comparing the sum with an integral, we get

$$
\sum _ { i = 0 } ^ { k } \beta _ { i } \le \frac { \beta } { K _ { 0 } ^ { b } } + \beta \int _ { K _ { 0 } } ^ { k + K _ { 0 } } { s ^ { - b } } d s \le \frac { \beta } { K _ { 0 } ^ { b } } + \frac { \beta ( k + K _ { 0 } ) ^ { 1 - b } } { 1 - b } .
$$

Combining the above two bounds and using $k + K _ { 0 } \le K _ { 0 } ( k + 1 )$ , which holds since $K _ { 0 } \ge 1$ , we get

$$
\frac { \sum _ { i = 0 } ^ { k } \beta _ { i } } { \sum _ { i = m } ^ { k } \beta _ { i } } \leq \frac { 2 } { k + 1 } \left( \frac { k + K _ { 0 } } { K _ { 0 } } \right) ^ { b } + \frac { 2 ( k + K _ { 0 } ) } { ( 1 - b ) ( k + 1 ) } \leq 2 + \frac { 2 K _ { 0 } } { 1 - b } = : \Gamma _ { 4 } .
$$

This proves the first inequality in (12).

For the second inequality, the sum is empty for $k \leq 1$ , and we assume that $k \geq 2$ . For $m \leq i < k$ , note that $\beta _ { i } / \beta _ { k } = ( ( k + K _ { 0 } ) / ( i + K _ { 0 } ) ) ^ { b } \le 2 ^ { b }$ and $\begin{array} { r } { \sum _ { j = i + 1 } ^ { k } \beta _ { j } \ge ( k - i ) \beta _ { k } } \end{array}$ . Therefore,

$$
\sum _ { i = m } ^ { k - 1 } \frac { \beta _ { i } \sqrt { \beta _ { i } } } { \sum _ { j = i + 1 } ^ { k } \beta _ { j } } \leq 2 ^ { 3 b / 2 } \sqrt { \beta _ { k } } \sum _ { i = m } ^ { k - 1 } \frac { 1 } { k - i } \leq 2 ^ { 3 b / 2 } \sqrt { \beta _ { k } } ( 1 + \log ( k - m ) ) \leq \frac { 3 \cdot 2 ^ { 3 b / 2 } \sqrt { \beta } \log ( k + 2 ) } { ( k + 1 ) ^ { b / 2 } } .
$$

Here, the second inequality follows by comparing the harmonic sum with an integral, and the third inequality follows from $\sqrt { \beta _ { k } } \le \sqrt { \beta } / ( k + 1 ) ^ { b / 2 }$ and $1 + \log ( k - m ) \leq 3 \log ( k + 2 )$ . This completes the proof of Lemma 6 with $\Gamma _ { 5 } = 3 \cdot 2 ^ { 3 b / 2 } \sqrt { \beta }$

## A.7 Proof of Theorem 1

Throughout the proofs of Theorems 1–3, we take $m = \lceil k / 2 \rceil$ . Since $K _ { 0 } \ge 1$ , we have $( k + K _ { 0 } ) ^ { b } \leq K _ { 0 } ^ { b } ( k + 1 ) ^ { b }$ and hence $\begin{array} { r } { \sum _ { i = 0 } ^ { k } \beta _ { i } \ge ( k + 1 ) \beta _ { k } \ge \beta ( k + 1 ) ^ { 1 - b } / K _ { 0 } ^ { b } } \end{array}$ . Combining this with the first inequality in Lemma 6, we get

$$
\bar { r } _ { m , k } \leq \Gamma _ { 4 } \bar { r } _ { 0 , k } \leq \frac { \Gamma _ { 6 } } { ( k + 1 ) ^ { 1 - b } } \left( \sum _ { i = 0 } ^ { k } \beta _ { i } \right) \bar { r } _ { 0 , k } , \qquad \mathrm { w h e r e ~ } \Gamma _ { 6 } : = \frac { \Gamma _ { 4 } K _ { 0 } ^ { b } } { \beta } .\tag{18}
$$

Under noise condition R, Lemma 2(a) then gives $\bar { r } _ { m , k } \leq \Gamma _ { 6 } \Vert x _ { 0 } - x ^ { \star } \Vert ^ { 2 } / ( [ 1 - ( 1 + \sigma ^ { 2 } ) \beta _ { 0 } ] ( k + 1 ) ^ { 1 - b } )$ . For the product in Lemma 5(a), using $1 + u \leq e ^ { u }$ and the second inequality in Lemma 6, we get

$$
\prod _ { i = m } ^ { k - 1 } \left[ 1 + 2 \sqrt { 2 } \sigma \frac { \beta _ { i } \sqrt { \beta _ { i } } } { \sum _ { j = i + 1 } ^ { k } \beta _ { j } } \right] \leq \exp \left( \frac { 2 \sqrt { 2 } \sigma \Gamma _ { 5 } \log ( k + 2 ) } { ( k + 1 ) ^ { b / 2 } } \right) \leq \exp \left( 2 \sqrt { 2 } \sigma \Gamma _ { 5 } \left( \log 2 + \frac { 2 } { b e } \right) \right) .
$$

Here, the second inequality follows from $\begin{array} { r } { \log ( k + 2 ) \leq \log 2 + \log ( k + 1 ) } \end{array}$ and $\mathrm { s u p } _ { t \geq 1 } \log t / t ^ { b / 2 } = 2 / ( b e )$ Substituting the above two bounds in Lemma 5(a), we get

$$
r _ { k } \le \frac { \Gamma _ { 6 } \| x _ { 0 } - x ^ { \star } \| ^ { 2 } } { 1 - ( 1 + \sigma ^ { 2 } ) \beta _ { 0 } } \exp \left( 2 \sqrt { 2 } \sigma \Gamma _ { 5 } \left( \log 2 + \frac { 2 } { b e } \right) \right) \frac { 1 } { ( k + 1 ) ^ { 1 - b } } .
$$

This completes the proof of the upper bound in Theorem 1 with

$$
\mathfrak { c } _ { 1 } = \frac { \Gamma _ { 6 } \| x _ { 0 } - x ^ { \star } \| ^ { 2 } } { 1 - ( 1 + \sigma ^ { 2 } ) \beta _ { 0 } } \exp \left( 2 \sqrt { 2 } \sigma \Gamma _ { 5 } \left( \log 2 + \frac { 2 } { b e } \right) \right) .
$$

The optimality of this bound follows from Proposition 1.

## A.8 Proof of Theorem 2

We first bound the sum of squared stepsizes. Since $K _ { 0 } \ge 1$ , comparing the sum with an integral gives

$$
\sum _ { i = 0 } ^ { k } \beta _ { i } ^ { 2 } \le \beta ^ { 2 } \left( 1 + \int _ { 1 } ^ { k + 1 } s ^ { - 2 b } d s \right) \le \left\{ \begin{array} { l l } { \beta ^ { 2 } ( k + 1 ) ^ { 1 - 2 b } / ( 1 - 2 b ) , } & { 0 < b < \frac 1 2 , } \\ { 3 \beta ^ { 2 } \log ( k + 2 ) , } & { b = \frac 1 2 , } \\ { \Gamma _ { 1 } , } & { \frac 1 2 < b < 1 , } \end{array} \right.
$$

where $\Gamma _ { 1 } = \textstyle \sum _ { i = 0 } ^ { \infty } \beta _ { i } ^ { 2 }$ , and the second case uses $1 + \log ( k + 1 ) \leq 3 \log ( k + 2 )$ . Using Lemma 5(b) with $\Gamma _ { 3 } = e \operatorname* { m a x } \{ 1 , 2 \sigma ^ { 2 } \} , ( 1 8 )$ , Lemma 2(b), and the second inequality in Lemma 6, we get

$$
r _ { k } \leq \Gamma _ { 3 } \left( \frac { \Gamma _ { 6 } } { ( k + 1 ) ^ { 1 - b } } \left( 2 \| x _ { 0 } - x ^ { \star } \| ^ { 2 } + 2 \sigma ^ { 2 } \sum _ { i = 0 } ^ { k } \beta _ { i } ^ { 2 } \right) + \frac { \Gamma _ { 5 } ^ { 2 } \log ^ { 2 } ( k + 2 ) } { ( k + 1 ) ^ { b } } \right) .\tag{19}
$$

For $0 < b < 1 / 2$ , using the first case of the bound on $\textstyle \sum _ { i = 0 } ^ { k } \beta _ { i } ^ { 2 }$ and $( k + 1 ) ^ { 1 - b } \geq ( k + 1 ) ^ { b }$ in (19), we get

$$
\begin{array} { c } { { r _ { k } \le \displaystyle \frac { \Gamma _ { 3 } } { ( k + 1 ) ^ { b } } \left( \Gamma _ { 6 } \left( 2 \| x _ { 0 } - x ^ { \star } \| ^ { 2 } + \displaystyle \frac { 2 \sigma ^ { 2 } \beta ^ { 2 } } { 1 - 2 b } \right) + \Gamma _ { 5 } ^ { 2 } \log ^ { 2 } ( k + 2 ) \right) } } \\ { { \le \Gamma _ { 3 } \left( \displaystyle \frac { \Gamma _ { 6 } } { \log ^ { 2 } 2 } \left( 2 \| x _ { 0 } - x ^ { \star } \| ^ { 2 } + \displaystyle \frac { 2 \sigma ^ { 2 } \beta ^ { 2 } } { 1 - 2 b } \right) + \Gamma _ { 5 } ^ { 2 } \right) \displaystyle \frac { \log ^ { 2 } ( k + 2 ) } { ( k + 1 ) ^ { b } } . } } \end{array}
$$

For $b = 1 / 2$ , using the second case of the bound on $\textstyle \sum _ { i = 0 } ^ { k } \beta _ { i } ^ { 2 }$ in (19), we similarly get

$$
r _ { k } \le \Gamma _ { 3 } \left( \frac { 2 \Gamma _ { 6 } \| x _ { 0 } - x ^ { \star } \| ^ { 2 } } { \log ^ { 2 } 2 } + \frac { 6 \Gamma _ { 6 } \sigma ^ { 2 } \beta ^ { 2 } } { \log 2 } + \Gamma _ { 5 } ^ { 2 } \right) \frac { \log ^ { 2 } ( k + 2 ) } { ( k + 1 ) ^ { 1 / 2 } } .
$$

This proves the first case of Theorem 2 with

$$
\mathfrak { c } _ { 2 } = \left\{ \begin{array} { l l } { \displaystyle \Gamma _ { 3 } \left( \frac { \Gamma _ { 6 } } { \log ^ { 2 } 2 } \left( 2 \| \boldsymbol { x } _ { 0 } - \boldsymbol { x } ^ { \star } \| ^ { 2 } + \frac { 2 \sigma ^ { 2 } \beta ^ { 2 } } { 1 - 2 b } \right) + \Gamma _ { 5 } ^ { 2 } \right) , } & { 0 < b < \frac { 1 } { 2 } , } \\ { \displaystyle \Gamma _ { 3 } \left( \frac { 2 \Gamma _ { 6 } \| \boldsymbol { x } _ { 0 } - \boldsymbol { x } ^ { \star } \| ^ { 2 } } { \log ^ { 2 } 2 } + \frac { 6 \Gamma _ { 6 } \sigma ^ { 2 } \beta ^ { 2 } } { \log 2 } + \Gamma _ { 5 } ^ { 2 } \right) , } & { b = \frac { 1 } { 2 } . } \end{array} \right.
$$

For $1 / 2 < b < 1$ , we first note that

$$
\frac { \log ^ { 2 } ( k + 2 ) } { ( k + 1 ) ^ { 2 b - 1 } } \leq \frac { 2 \log ^ { 2 } 2 + 2 \log ^ { 2 } ( k + 1 ) } { ( k + 1 ) ^ { 2 b - 1 } } \leq 2 \log ^ { 2 } 2 + \frac { 8 } { e ^ { 2 } ( 2 b - 1 ) ^ { 2 } } .\tag{20}
$$

Here, the first inequality follows from $\log ( k + 2 ) \leq \log 2 + \log ( k + 1 )$ , and the second inequality follows from $\begin{array} { r } { \operatorname* { s u p } _ { t \geq 1 } \log ^ { 2 } t / t ^ { 2 b - 1 } = 4 / ( e ^ { 2 } ( 2 b - 1 ) ^ { 2 } ) } \end{array}$ . Using the third case of the bound on $\textstyle \sum _ { i = 0 } ^ { k } \beta _ { i } ^ { 2 }$ and (20) in (19), we get

$$
r _ { k } \leq \Gamma _ { 3 } \left( \Gamma _ { 6 } ( 2 \| x _ { 0 } - x ^ { \star } \| ^ { 2 } + 2 \sigma ^ { 2 } \Gamma _ { 1 } ) + \Gamma _ { 5 } ^ { 2 } \left( 2 \log ^ { 2 } 2 + \frac { 8 } { e ^ { 2 } ( 2 b - 1 ) ^ { 2 } } \right) \right) \frac { 1 } { ( k + 1 ) ^ { 1 - b } } .
$$

This completes the proof of the upper bound in Theorem 2 with

$$
\mathfrak { c } _ { 3 } = \Gamma _ { 3 } \left( \Gamma _ { 6 } ( 2 \| \boldsymbol { x } _ { 0 } - \boldsymbol { x } ^ { \star } \| ^ { 2 } + 2 \sigma ^ { 2 } \Gamma _ { 1 } ) + \Gamma _ { 5 } ^ { 2 } \left( 2 \log ^ { 2 } 2 + \frac { 8 } { e ^ { 2 } ( 2 b - 1 ) ^ { 2 } } \right) \right) .
$$

The optimality of this bound, up to a logarithmic factor, follows from Propositions 1 and 2.

## A.9 Proof of Theorem 3

The proof is identical to that of the case $1 / 2 < b < 1$ in Theorem 2, with Lemma 2(c) in place of Lemma 2(b), and with $\Gamma _ { 3 } = e \operatorname* { m a x } \{ 1 , 2 \sigma ^ { 2 } ( 1 + 2 \| x ^ { \star } \| ^ { 2 } + 2 \Gamma _ { 2 } ) \}$ in Lemma 5(b). Using (18), Lemma 2(c), the second inequality in Lemma 6, and (20), we get

$$
r _ { k } \leq \Gamma _ { 3 } \left( 2 \Gamma _ { 6 } \left( \| x _ { 0 } - x ^ { \star } \| ^ { 2 } + \sigma ^ { 2 } \Gamma _ { 1 } ( 1 + 2 \| x ^ { \star } \| ^ { 2 } + 2 \Gamma _ { 2 } ) \right) + \Gamma _ { 5 } ^ { 2 } \left( 2 \log ^ { 2 } 2 + \frac { 8 } { e ^ { 2 } ( 2 b - 1 ) ^ { 2 } } \right) \right) \frac { 1 } { ( k + 1 ) ^ { 1 - b } } .
$$

This completes the proof of the upper bound in Theorem 3 with

$$
\mathfrak { c } _ { 4 } = \Gamma _ { 3 } \left( 2 \Gamma _ { 6 } \left( \| x _ { 0 } - x ^ { \star } \| ^ { 2 } + \sigma ^ { 2 } \Gamma _ { 1 } ( 1 + 2 \| x ^ { \star } \| ^ { 2 } + 2 \Gamma _ { 2 } ) \right) + \Gamma _ { 5 } ^ { 2 } \left( 2 \log ^ { 2 } 2 + \frac { 8 } { e ^ { 2 } ( 2 b - 1 ) ^ { 2 } } \right) \right) .
$$

The optimality of this bound follows from Proposition 1, and the divergence for $0 < b < 1 / 2$ follows from Proposition 3.

## B Proofs of the Optimality Results

Recall that $r _ { i } = \mathbb { E } \left[ \Vert f ( x _ { i } ) - x _ { i } \Vert ^ { 2 } \right]$ denotes the mean-square residual at time �. Throughout this appendix, we use the facts about the rotation $R _ { \theta }$ stated at the beginning of Section 4.

## B.1 Proof of Proposition 1

Fix $k \geq 1$ , consider the instance in Example 1, and write $\textstyle \gamma : = 2 ( 1 - \cos \theta ) = 1 / ( 1 + 2 \sum _ { j = 0 } ^ { k - 1 } \beta _ { j } )$ . Since $M _ { i + 1 } = 0$ , the iteration is $x _ { i + 1 } = ( 1 - \beta _ { i } ) x _ { i } + \beta _ { i } R _ { \theta } x _ { i }$ . Applying (13) repeatedly, we get

$$
r _ { k } = \gamma \| x _ { k } \| ^ { 2 } = \gamma \prod _ { j = 0 } ^ { k - 1 } \left[ 1 - \gamma \beta _ { j } ( 1 - \beta _ { j } ) \right] .
$$

Since $\gamma \leq 1$ and $\beta _ { j } \leq 1 / 2$ , we have $0 \leq \gamma \beta _ { j } ( 1 - \beta _ { j } ) \leq 1 / 2$ . Using $1 - u \geq e ^ { - 2 u }$ for $0 \leq u \leq 1 / 2$ , we get

$$
\prod _ { j = 0 } ^ { k - 1 } \left[ 1 - \gamma \beta _ { j } ( 1 - \beta _ { j } ) \right] \geq \exp \left( - 2 \gamma \sum _ { j = 0 } ^ { k - 1 } \beta _ { j } \right) \geq e ^ { - 1 } .\tag{21}
$$

Here, the second inequality follows from $\begin{array} { r } { 2 \gamma \sum _ { j = 0 } ^ { k - 1 } \beta _ { j } = 2 \sum _ { j = 0 } ^ { k - 1 } \beta _ { j } / ( 1 + 2 \sum _ { j = 0 } ^ { k - 1 } \beta _ { j } ) \le 1 } \end{array}$ . Moreover, since $K _ { 0 } \ge 1$ , comparing the sum with an integral gives

$$
\sum _ { j = 0 } ^ { k - 1 } \beta _ { j } \leq \beta \sum _ { j = 0 } ^ { k - 1 } ( j + 1 ) ^ { - b } \leq \beta \left( 1 + \int _ { 1 } ^ { k } s ^ { - b } d s \right) \leq { \frac { \beta ( k + 1 ) ^ { 1 - b } } { 1 - b } } ,
$$

and hence $\gamma \geq \Gamma _ { 7 } / ( k + 1 ) ^ { 1 - b }$ , where $\Gamma _ { 7 } : = ( 1 + 2 \beta / ( 1 - b ) ) ^ { - 1 }$ . Combining the above bounds gives $r _ { k } \ge \Gamma _ { 7 } / ( e ( k + 1 ) ^ { 1 - b } )$ . This completes the proof of Proposition 1 with ${ \mathfrak { c } } _ { 5 } = \Gamma _ { 7 } / e$

## B.2 Proof of Proposition 2

Fix $k \geq 1$ , consider the instance in Example 2, and write $\gamma _ { p } : = 2 ( 1 - \cos \theta _ { p } )$ . Since $\varepsilon _ { i + 1 }$ is independent of $\mathcal { F } _ { i } .$ , we have $\\mathbb { E } \left[ M _ { i + 1 } \ | \ \mathcal { F } _ { i } \right] = 0$ and $\mathbb { E } \left[ \Vert \bar { M } _ { i + 1 } \Vert ^ { 2 } \ | \ \mathcal { F } _ { i } \right] = \sigma ^ { 2 }$ , and hence noise condition C holds. Moreover, the noise injected at times $i \geq k$ does not afect $x _ { k }$ . For $1 \leq p \leq k$ , let $x _ { j } ^ { ( p ) } \in \mathbb { R } ^ { 2 }$ denote the �-th pair of coordinates of $x _ { j }$

We first compute the contribution of each pair to the residual at time �. Fix $1 \leq p \leq k$ . Since $x _ { 0 } = 0$ , and before time � the �-th pair receives noise only at time $p - 1$ , we have $x _ { j } ^ { ( p ) } = \overset { \cdot } { 0 }$ for $j \le p - 1$ $x _ { p } ^ { ( p ) } = \sigma \beta _ { p - 1 } \varepsilon _ { p } ( 1 , 0 )$ , and $x _ { j + 1 } ^ { ( p ) } = ( 1 - \beta _ { j } ) x _ { j } ^ { ( p ) } + \beta _ { j } R _ { \theta _ { p } } x _ { j } ^ { ( p ) }$ for $p \leq j \leq k - 1$ . Since $\varepsilon _ { p } ^ { 2 } = 1$ , applying (13) repeatedly gives

$$
\left\| R _ { \theta _ { p } } x _ { k } ^ { ( p ) } - x _ { k } ^ { ( p ) } \right\| ^ { 2 } = \gamma _ { p } \sigma ^ { 2 } \beta _ { p - 1 } ^ { 2 } \prod _ { j = p } ^ { k - 1 } \left[ 1 - \gamma _ { p } \beta _ { j } ( 1 - \beta _ { j } ) \right] \geq \frac { \gamma _ { p } \sigma ^ { 2 } \beta _ { p - 1 } ^ { 2 } } { e } .
$$

Here, the inequality follows as in (21), since $\begin{array} { r } { 2 \gamma _ { p } \sum _ { j = p } ^ { k - 1 } \beta _ { j } \le 1 } \end{array}$ . Since $f$ acts separately on each pair and the pairs are orthogonal, $\begin{array} { r } { \| f ( x _ { k } ) - x _ { k } \| ^ { 2 } = \sum _ { p = 1 } ^ { k } \| R _ { \theta _ { p } } x _ { k } ^ { ( \bar { p } ) } - x _ { k } ^ { ( p ) } \| ^ { 2 } } \end{array}$ . Summing the above bound over the pairs and substituting $i = p - 1$ , we get

$$
r _ { k } \geq \frac { \sigma ^ { 2 } } { e } \sum _ { i = 0 } ^ { k - 1 } \frac { \beta _ { i } ^ { 2 } } { 1 + 2 \sum _ { j = i + 1 } ^ { k - 1 } \beta _ { j } } .\tag{22}
$$

It remains to lower bound the sum in (22). For $k = 1$ , we have $\gamma _ { 1 } = 1$ , and the above computation gives $r _ { 1 } = \sigma ^ { 2 } \beta _ { 0 } ^ { 2 }$ exactly. For $k \geq 2$ , let $m = \lceil k / 2 \rceil$ . Since all terms in (22) are nonnegative, we can restrict the sum to $m \leq i < k$ . For $m \leq i < k$ , we have $\beta _ { i } \geq \beta _ { k }$ and $\begin{array} { r } { \sum _ { j = i + 1 } ^ { k - 1 } \beta _ { j } \le ( k - i - 1 ) \beta _ { m } } \end{array}$ . Hence,

$$
r _ { k } \geq \frac { \sigma ^ { 2 } \beta _ { k } ^ { 2 } } { e } \sum _ { j = 0 } ^ { k - m - 1 } \frac { 1 } { 1 + 2 j \beta _ { m } } \geq \frac { \sigma ^ { 2 } \beta _ { k } ^ { 2 } } { e } \int _ { 0 } ^ { k - m } \frac { d s } { 1 + 2 s \beta _ { m } } = \frac { \sigma ^ { 2 } \beta _ { k } ^ { 2 } } { 2 e \beta _ { m } } \log \left( 1 + 2 ( k - m ) \beta _ { m } \right) .
$$

We now bound the two factors on the right-hand side. Since $m + K _ { 0 } \ge ( k + K _ { 0 } ) / 2$ and $k + K _ { 0 } \le K _ { 0 } ( k + 1 )$

$$
\frac { \beta _ { k } ^ { 2 } } { \beta _ { m } } = \frac { \beta ( m + K _ { 0 } ) ^ { b } } { ( k + K _ { 0 } ) ^ { 2 b } } \geq \frac { \beta } { 2 ^ { b } ( k + K _ { 0 } ) ^ { b } } \geq \frac { \Gamma _ { 8 } } { ( k + 1 ) ^ { b } } , \qquad \mathrm { w h e r e ~ } \Gamma _ { 8 } : = \frac { \beta } { 2 ^ { b } K _ { 0 } ^ { b } } .
$$

Moreover, since $k - m \geq ( k + 1 ) / 4$ for $k \geq 2$ , we have $2 ( k - m ) \beta _ { m } \geq ( k + 1 ) \beta _ { k } / 2 \geq \beta ( k + 1 ) ^ { 1 - b } / ( 2 K _ { 0 } ^ { b } )$ Using log $( 1 + c x ) \geq \operatorname* { m i n } \{ 1 , c \} \log ( 1 + x )$ for $c , x \geq 0 .$ , we get

$$
\log \left( 1 + 2 ( k - m ) \beta _ { m } \right) \geq \operatorname* { m i n } \left\{ 1 , \frac { \beta } { 2 K _ { 0 } ^ { b } } \right\} \log \left( 1 + ( k + 1 ) ^ { 1 - b } \right) \geq \Gamma _ { 9 } \log ( k + 1 ) ,
$$

where $\Gamma _ { 9 } : = ( 1 - b ) \operatorname* { m i n } \{ 1 , \beta / ( 2 K _ { 0 } ^ { b } ) \}$ . Combining the above bounds gives $r _ { k } \geq \sigma ^ { 2 } \Gamma _ { 8 } \Gamma _ { 9 } \log ( k + 1 ) / ( 2 e ( k +$ $1 ) ^ { b } )$ for $k \geq 2$ . Together with the case $k = 1$ , this completes the proof of Proposition 2 with

$$
{ \mathfrak { c } } _ { 6 } = \operatorname* { m i n } \left\{ { \frac { \Gamma _ { 8 } \Gamma _ { 9 } } { 2 e } } , { \frac { 2 ^ { b } \beta _ { 0 } ^ { 2 } } { \log 2 } } \right\} .
$$

## B.3 Proof of Proposition 3

Consider the instance in Example 3. Since $f ( x ) - x = ( 0 , - 2 \nu )$ , the coordinates of the iteration (1) evolve as

$$
u _ { i + 1 } = u _ { i } + \beta _ { i } M _ { i + 1 } ^ { ( 1 ) } , \qquad \nu _ { i + 1 } = ( 1 - 2 \beta _ { i } ) \nu _ { i } + \beta _ { i } M _ { i + 1 } ^ { ( 2 ) } .
$$

Since the coordinates of $M _ { i + 1 }$ have conditional mean zero and conditional second moment $\sigma ^ { 2 } u _ { i } ^ { 2 } / 2$ , taking expectations gives

$$
\mathbb { E } \left[ u _ { i + 1 } ^ { 2 } \right] = \left( 1 + \frac { \sigma ^ { 2 } \beta _ { i } ^ { 2 } } { 2 } \right) \mathbb { E } \left[ u _ { i } ^ { 2 } \right] , \quad \quad \mathbb { E } \left[ \nu _ { i + 1 } ^ { 2 } \right] = ( 1 - 2 \beta _ { i } ) ^ { 2 } \mathbb { E } \left[ \nu _ { i } ^ { 2 } \right] + \frac { \sigma ^ { 2 } \beta _ { i } ^ { 2 } } { 2 } \mathbb { E } \left[ u _ { i } ^ { 2 } \right] .
$$

Since $u _ { 0 } = 1$ , iterating the first recursion and using log $( 1 + z ) \geq z / ( 1 + z )$ with $z = \sigma ^ { 2 } \beta _ { j } ^ { 2 } / 2 \le \sigma ^ { 2 } / 8$ , we get

$$
\mathbb { E } \left[ u _ { i } ^ { 2 } \right] = \prod _ { j = 0 } ^ { i - 1 } \left( 1 + \frac { \sigma ^ { 2 } \beta _ { j } ^ { 2 } } { 2 } \right) \ge \exp \left( \frac { \sigma ^ { 2 } / 2 } { 1 + \sigma ^ { 2 } / 8 } \sum _ { j = 0 } ^ { i - 1 } \beta _ { j } ^ { 2 } \right) .
$$

Dropping the nonnegative first term in the recursion for $\mathbb { E } [ \nu _ { k } ^ { 2 } ]$ , for every $k \geq 1$

$$
r _ { k } = 4 \mathbb { E } \left[ \nu _ { k } ^ { 2 } \right] \geq 2 \sigma ^ { 2 } \beta _ { k - 1 } ^ { 2 } \mathbb { E } \left[ u _ { k - 1 } ^ { 2 } \right] \geq 2 \sigma ^ { 2 } \beta _ { k - 1 } ^ { 2 } \exp \left( \frac { \sigma ^ { 2 } / 2 } { 1 + \sigma ^ { 2 } / 8 } \sum _ { j = 0 } ^ { k - 2 } \beta _ { j } ^ { 2 } \right) .\tag{23}
$$

Part (a). We bound the two factors on the right-hand side of (23). Since $k - 1 + K _ { 0 } \le K _ { 0 } ( k + 1 )$ , we have $\beta _ { k - 1 } ^ { 2 } \ge \Gamma _ { 1 0 } / ( k + 1 ) ^ { 2 b }$ , where $\Gamma _ { 1 0 } : = \beta ^ { \bar { 2 } } / K _ { 0 } ^ { 2 b }$ . For $k \geq 2$ , the monotonicity of the stepsizes and $k - 1 \geq ( k + 1 ) / 3$ give

$$
\sum _ { j = 0 } ^ { k - 2 } \beta _ { j } ^ { 2 } \geq ( k - 1 ) \beta _ { k - 2 } ^ { 2 } \geq \frac { ( k - 1 ) \beta ^ { 2 } } { K _ { 0 } ^ { 2 b } ( k + 1 ) ^ { 2 b } } \geq \Gamma _ { 1 1 } ( k + 1 ) ^ { 1 - 2 b } , \quad \quad \mathrm { ~ w h e r e ~ } \Gamma _ { 1 1 } : = \frac { \beta ^ { 2 } } { 3 K _ { 0 } ^ { 2 b } } .
$$

Since the sum is empty for $k = 1$ , we have $\begin{array} { r } { \sum _ { i = 0 } ^ { k - 2 } \beta _ { i } ^ { 2 } \ge \Gamma _ { 1 1 } ( ( k + 1 ) ^ { 1 - 2 b } - 2 ^ { 1 - 2 b } ) } \end{array}$ for all $k \geq 1$ . Substituting the above bounds in (23) completes the proof of Proposition 3 with

$$
\mathfrak c _ { 8 } = \frac { \sigma ^ { 2 } \Gamma _ { 1 1 } } { 2 ( 1 + \sigma ^ { 2 } / 8 ) } , \qquad \mathfrak c _ { 7 } = 2 \sigma ^ { 2 } \Gamma _ { 1 0 } \exp \left( - 2 ^ { 1 - 2 b } \mathfrak c _ { 8 } \right) .
$$

Finally, since $1 - 2 b > 0$ , the exponential term dominates the polynomial term, and hence $r _ { k } $ ∞ as $k \to \infty$ Part (b). Let $b = 1 / 2$ and $p : = \sigma ^ { 2 } \beta ^ { 2 } / 2 > 1$ . Then $\beta _ { j } ^ { 2 } = \beta ^ { 2 } / ( j + K _ { 0 } )$ . Using log $( 1 + z ) \geq z - z ^ { 2 } / 2$ for $z \geq 0$ , we get

$$
\mathbb { E } \left[ u _ { i } ^ { 2 } \right] = \prod _ { j = 0 } ^ { i - 1 } \left( 1 + \frac { p } { j + K _ { 0 } } \right) \ge \exp \left( p \sum _ { j = 0 } ^ { i - 1 } \frac { 1 } { j + K _ { 0 } } - \frac { p ^ { 2 } } { 2 } \sum _ { j = 0 } ^ { \infty } \frac { 1 } { ( j + K _ { 0 } ) ^ { 2 } } \right) \ge e ^ { - p ^ { 2 } } \left( \frac { i + K _ { 0 } } { K _ { 0 } } \right) ^ { p } ,
$$

where we used $\begin{array} { r } { \sum _ { j = 0 } ^ { i - 1 } 1 / ( j + K _ { 0 } ) \geq \log ( ( i + K _ { 0 } ) / K _ { 0 } ) } \end{array}$ and $\begin{array} { r } { \sum _ { j = 0 } ^ { \infty } 1 / ( j + K _ { 0 } ) ^ { 2 } \le 1 / K _ { 0 } ^ { 2 } + 1 / K _ { 0 } \le 2 } \end{array}$ . Substituting this with $i = k - 1$ in the first inequality in (23), and using $\beta _ { k - 1 } ^ { 2 } = \beta ^ { 2 } / ( k - 1 + K _ { 0 } )$ and $k - 1 + K _ { 0 } \ge ( k + 1 ) / 2$ we get

$$
r _ { k } \geq 2 \sigma ^ { 2 } \beta ^ { 2 } e ^ { - p ^ { 2 } } K _ { 0 } ^ { - p } ( k - 1 + K _ { 0 } ) ^ { p - 1 } \geq \mathfrak { c } _ { 9 } ( k + 1 ) ^ { p - 1 } , \qquad \mathrm { w h e r e ~ } \mathfrak { c } _ { 9 } : = 2 ^ { 2 - p } \sigma ^ { 2 } \beta ^ { 2 } e ^ { - p ^ { 2 } } K _ { 0 } ^ { - p } .
$$

Since $p > 1$ , we get $r _ { k } \to \infty { \mathrm { a s ~ } } k \to \infty$

## References

[1] Ethan Blaser and Shangtong Zhang. Asymptotic and finite sample analysis of nonexpansive stochastic approximations with Markovian noise. Proceedings ofthe AAAI Conference on Artificial Intelligence, 40(24):19764–19772, 2026.

[2] Vivek S. Borkar. Stochastic Approximation: A Dynamical Systems Viewpoint, volume 48 of Texts and Readings in Mathematics. Hindustan Book Agency, second edition, 2022.

[3] Léon Bottou, Frank E. Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. SIAM Review, 60(2):223–311, 2018.

[4] Mario Bravo and Roberto Cominetti. Stochastic fixed-point iterations for nonexpansive maps: Convergence and error bounds. SIAM Journal on Control and Optimization, 62(1):191–219, 2024.

[5] Mario Bravo and Juan Pablo Contreras. Stochastic Halpern iteration in normed spaces and applications to reinforcement learning. Mathematical Programming, 2026.

[6] Xufeng Cai, Chaobing Song, Cristóbal Guzmán, and Jelena Diakonikolas. Stochastic Halpern iteration with variance reduction for stochastic monotone inclusions. In Advances in Neural Information Processing Systems, volume 35, 2022.

[7] Siddharth Chandak. Nonexpansive mappings in two-time-scale stochastic approximation: Finite-time analysis. SIAM Journal on Control and Optimization, 64(4):3024–3049, 2026.

[8] Siddharth Chandak. Concentration and mean-square bounds for contractive stochastic approximation: A unified elementary approach, 2026. arXiv:2607.17595.

[9] Siddharth Chandak, Vivek S. Borkar, and Parth Dodhia. Concentration of contractive stochastic approximation and reinforcement learning. Stochastic Systems, 12(4):411–430, 2022.

[10] Siddharth Chandak, Ramanan Tamizholi, and Nicholas Bambos. Last-iterate guarantees for learning in co-coercive games, 2026. arXiv:2604.19065.

[11] Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. Finite-sample analysi of contractive stochastic approximation using smooth convex envelopes. In Advances in Neural Information Processing Systems, volume 33, 2020.

[12] Zaiwei Chen, Siva Theja Maguluri, and Martin Zubeldia. Concentration of contractive stochastic approximation: Additive and multiplicative noise. The Annals ofApplied Probability, 35(2):1298–1352, 2025.

[13] Roberto Cominetti, José A. Soto, and José Vaisman. On the rate of convergence of Krasnosel’skii–Mann iterations and their connection with sums of Bernoullis. Israel Journal ofMathematics, 199(2):757–772, 2014.

[14] Daniel Cortild, Coralia Cartis, and Juan Peypouquet. Stochastic Krasnoselskii–Mann iterations: Convergence without uniformly bounded variance, 2026. arXiv:2604.22581.

[15] Damek Davis and Wotao Yin. Convergence rate analysis of several splitting schemes. In Roland Glowinski, Stanley J. Osher, and Wotao Yin, editors, Splitting Methods in Communication, Imaging, Science, and Engineering, Scientific Computation, pages 115–163. Springer, 2016.

[16] Jelena Diakonikolas. Solving stochastic fixed-point equations with high probability, 2026. arXiv:2607.09097.

[17] Jelena Diakonikolas, Cristóbal Guzmán, and David Martínez-Rubio. Oracle complexity of stochastic fixed-point equations with nonexpansive maps, 2026. arXiv:2609.09524.

[18] Francisco Facchinei and Jong-Shi Pang. Finite-Dimensional Variational Inequalities and Complementarity Problems, volume I of Springer Series in Operations Research. Springer, 2003.

[19] Hideaki Iiduka. Mini-batch stochastic Krasnosel’skii–Mann algorithm for nonexpansive fixed point problems, 2026. arXiv:2604.06909.

[20] Hideaki Iiduka. Mini-batch stochastic Halpern algorithm for nonexpansive fixed point problems, 2026. arXiv:2604.21443.

[21] Ege C. Kaya, Arda Fazla, M. Berk Sahin, and Abolfazl Hashemi. A Banach-space theory of Markovian Halpern iteration for non-expansive maps, 2026. arXiv:2608.15966.

[22] Mark Aleksandrovich Krasnosel’ski˘ı. Two remarks on the method of successive approximations. Uspekhi Matematicheskikh Nauk, 10(1):123–127, 1955.

[23] Harold J. Kushner and G. George Yin. Stochastic Approximation and Recursive Algorithms and Applications, volume 35 of Stochastic Modelling and Applied Probability. Springer, second edition, 2003.

[24] Jingwei Liang, Jalal Fadili, and Gabriel Peyré. Convergence rates with inexact non-expansive operators. Mathematical Programming, 159(1–2):403–434, 2016.

[25] Tianyi Lin, Zhengyuan Zhou, Panayotis Mertikopoulos, and Michael I. Jordan. Finite-time last-iterate convergence for multi-agent learning in games. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pages 6161–6171, 2020.

[26] W. Robert Mann. Mean value methods in iteration. Proceedings of the American Mathematical Society, 4(3): 506–510, 1953.

[27] Yurii Nesterov. Lectures on Convex Optimization, volume 137 of Springer Optimization and Its Applications. Springer, 2018.

[28] Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics, 22(3):400–407, 1951.

[29] Dhruv Sarkar and Vaneet Aggarwal. Non-expansive two-time-scale stochastic approximation: A fixed-schedule one-quarter barrier and bias-corrected acceleration, 2026. arXiv:2607.13414.

[30] Mark Schmidt and Nicolas Le Roux. Fast convergence of stochastic gradient descent under a strong growth condition, 2013. arXiv:1308.6370.

[31] Yekini Shehu. Adaptivity, anchoring, and the exact oracle complexity of stochastic fixed-point iterations, 2026. arXiv:2610.00295.

[32] R. Srikant and Lei Ying. Finite-time error bounds for linear stochastic approximation and TD learning. In Proceedings of the 32nd Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 2803–2830. PMLR, 2019.

[33] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, second edition, 2018.

[34] Sharan Vaswani, Francis Bach, and Mark Schmidt. Fast and faster convergence of SGD for over-parameterized models and an accelerated perceptron. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings ofMachine Learning Research, pages 1195–1204, 2019.

[35] Tong Yang, Tao Jiang, Yuejie Chi, Ashok Cutkosky, and Lin Xiao. Stochastic inertial Krasnosel’skii–Mann iteration achieves near-optimal sample complexity, 2026. arXiv:2609.28543.