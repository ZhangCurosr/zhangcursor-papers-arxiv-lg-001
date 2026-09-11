# AdamX: Cosine similarity meets gradient descent

Francisco Caldas<sup>1[0000−0001−5090−0216]</sup>, Ruben Belo<sup>1[0009−0006−8516−7732]</sup>, and Cláudia Soares<sup>1[0000−0003−3071−6627]</sup>

NOVA School of Science of Technology Universidade Nova de Lisboa, Caparica, Portugal f.caldas@campus.fct.unl.pt

Abstract. We introduce AdamX, a first-order optimizer that incorporates cosine similarity as an adaptive mechanism for controlling update magnitudes. The proposed method is scalable, model-agnostic, and straightforward to integrate into existing training pipelines. We further introduce a variance rectification scheme that promotes smoother optimization during the early stages of training. Overall, we provide empirical evidence that AdamX achieves competitive convergence rates across a range of benchmark datasets and architectures. Performance is evaluated in terms of the number of epochs required to reach predefined performance thresholds under a fixed hyperparameter budget. Code and Experiments available at: https://github.com/FranciscoCaldas/adamX

Keywords: First Order Optimizer · Online Convex Optimization · Cosine Similarity.

## 1 Introduction

Gradient-based optimization is central to modern machine learning, where model training requires minimizing high-dimensional and generally non-convex objectives. While stochastic gradient descent (SGD) remains a foundational approach [4], adaptive first-order methods have become widely used because they adjust update magnitudes according to observed gradient statistics. Prominent examples include AdaGrad [8], RMSProp [1], and Adam [12].

Adam combines exponential moving averages of the gradient and its coordinatewise squared magnitude to produce momentum-based, adaptively normalized updates [12]. This combination has made Adam a practical default for many deep learning tasks, as it often provides stable training behavior with limited task-specific tuning. However, its adaptive normalization can also lead to problematic update dynamics and, in some settings, a failure to converge [15].

Existing Adam-type methods primarily improve the treatment of magnitude information in the optimization trajectory. AMSGrad enforces a monotone second-moment envelope to recover convergence guarantees in online convex optimization [15]; AdamW decouples weight decay from adaptive updates [14]; RAdam addresses instability in the early variance estimate [13]; and AdaBelief modifies the second-moment statistic to reflect deviation from the predicted gradient direction [23]. More recent optimizers, such as Lion [5] and Muon [11], further reconsider the form of the update rule or its preconditioning structure. Despite these developments, the alignment between successive gradients remains comparatively underused as a direct mechanism for modulating update magnitudes.

Directional alignment provides a computationally inexpensive signal for adapting update magnitudes. Consecutive gradients that point in similar directions indicate locally consistent optimization progress, whereas poorly aligned or opposing gradients may indicate oscillation or rapidly changing trajectory information. Cosine similarity captures this signal independently of gradient scale. Closely related to this motivation, GALA [10] adapts the learning rate using consecutive-gradient alignment together with a local curvature estimate, formulated through a one-dimensional online learning problem. AdamX instead incorporates alignment through a bounded multiplicative cosine controller within an Adam/AMSGrad-style coordinate-wise adaptive update. This design preserves the practical structure of adaptive moment methods while explicitly exploiting directional consistency.

Second-order and preconditioned optimization methods also exploit geometric information to improve training dynamics. Methods such as Shampoo and SOAP construct richer approximations to curvature or preconditioning structure, and can improve optimization performance in large-scale learning problems [9,20]. In contrast, our objective is to investigate whether a lightweight scalar signal derived from consecutive gradient directions can provide useful geometric adaptivity while retaining the implementation simplicity and scalability of first-order Adam-type methods.

Our contributions are threefold. First, we introduce AdamX, an adaptive first-order optimizer that integrates a bounded cosine-similarity controller into an Adam-style moment-normalized update with a monotone second-moment envelope. Second, we provide an OCO analysis of a simplified momentum-free AdamX variant, showing that a bounded cosine controller can be incorporated into an adaptive projected-gradient scheme without worsening the standard convex regret rate. Third, we evaluate AdamX across benchmark datasets and architectures, with default settings, measuring the number of epochs required to reach predefined test-performance thresholds.

## 2 Method

We consider stochastic optimization of an expected loss over parameters $\theta \in \mathbb { R } ^ { d }$ Let D denote the data distribution and let $B \sim \mathcal { D }$ be a randomly sampled mini-batch. The training objective is

$$
\operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { d } } \mathbb { E } _ { B \sim D } \left[ \mathcal { L } ( \theta ; B ) \right] .\tag{1}
$$

At iteration $t ,$ the optimizer observes the stochastic gradient

$$
g _ { t } = \nabla _ { \theta } \mathcal { L } ( \theta _ { t } ; B _ { t } ) .\tag{2}
$$

AdamX builds on the moment-normalized update used by Adam. Specifically, it maintains exponential moving averages of the stochastic gradient and its coordinate-wise square

$$
m _ { t } = \beta _ { 1 } m _ { t - 1 } + ( 1 - \beta _ { 1 } ) g _ { t } , \qquad v _ { t } = \beta _ { 2 } v _ { t - 1 } + ( 1 - \beta _ { 2 } ) g _ { t } ^ { 2 } ,\tag{3}
$$

where $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ and all squares are taken element-wise. The corresponding bias-corrected estimates are

$$
\hat { m } _ { t } = \frac { m _ { t } } { 1 - \beta _ { 1 } ^ { t } } , \qquad \hat { v } _ { t } = \frac { v _ { t } } { 1 - \beta _ { 2 } ^ { t } } .\tag{4}
$$

AdamX addresses two complementary aspects of adaptive optimization. First, Adam may fail to converge even in simple convex settings because its adaptive denominator can produce unfavorable efective stepsizes. AMSGrad addresses this issue by retaining a coordinate-wise maximum of past second-moment estimates [15]. Second, adaptive learning rates may exhibit high variance during the early stages of training, motivating rectification mechanisms such as RAdam [13]. AdamX retains an AMSGrad-style monotone denominator and augments it with a bounded controller derived from directional agreement between consecutive gradients.

The distinctive component of AdamX is a cosine-similarity controller that modulates the magnitude of updates. For $t \geq 2$ , we define

$$
c _ { t } = \frac { \langle g _ { t } , g _ { t - 1 } \rangle } { \operatorname* { m a x } \left\{ \| g _ { t } \| _ { 2 } \| g _ { t - 1 } \| _ { 2 } , \delta \right\} } , \qquad \gamma _ { t } = \exp ( \lambda c _ { t } ) ,\tag{5}
$$

where $\lambda \geq 0$ controls the strength of the adaptation and $\delta > 0$ δ > prevents division by zero. We set $\gamma _ { 1 } ~ = ~ 1$ , since no preceding gradient is available at the first iteration. Because $c _ { t } \in [ - 1 , 1 ]$ , the controller is bounded as

$$
e ^ { - \lambda } \leq \gamma _ { t } \leq e ^ { \lambda } .\tag{6}
$$

Thus, aligned consecutive gradients increase the efective update magnitude, while opposing gradients reduce it, without making the controller dependent on gradient scale. Also note that, for each parameter group $k , \gamma _ { t } ^ { ( k ) }$ is computed from the cosine similarity between the current and previous gradients of the group.

Gradient alignment has previously been used to adapt learning rates in hypergradient-based methods [2,17,3]. AdamX uses this signal in a diferent optimizer structure: cosine similarity acts as a bounded multiplicative controller on top of an Adam-style moment-normalized update with a monotone secondmoment envelope. Consequently, setting $\lambda = 0$ removes the alignment controller and recovers the AMSGrad algorithm.

The normalization component of AdamX uses an AMSGrad-style variance envelope. We define

$$
\tilde { v } _ { t } = \operatorname* { m a x } \left\{ \tilde { v } _ { t - 1 } , \hat { v } _ { t } \right\} ,\tag{7}
$$

where the maximum is evaluated coordinate-wise. This construction ensures that the adaptive denominator is coordinate-wise non-decreasing, preventing increases in efective coordinate-wise stepsizes that arise solely from decreases in the second-moment estimate.

The resulting AdamX update combines moment normalization, the monotone variance envelope, and the cosine controller. Given a base learning rate $\eta > 0 .$ the parameters are updated as $\begin{array} { r } { \theta _ { t } = \theta _ { t - 1 } - \eta \gamma _ { t } \frac { \hat { m } _ { t } } { \sqrt { \tilde { v } _ { t } } + \epsilon { \bf 1 } } } \end{array}$ , where all vector operations in the denominator are coordinate-wise.

Algorithm 1 AdamX Optimizer   
Require: Learning rate η, decay rates $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ , epsilon ϵ   
Require: Scaling parameter $\lambda ,$ initial parameters $\theta _ { 0 }$   
1: $m _ { 0 } \gets 0 , v _ { 0 } \gets 0 , \gamma _ { 1 } \gets 1$   
2: g<sub>prev</sub> ← 1 {Initialize previous gradient}   
3: for $t = 1$ to $T$ do   
4: $g _ { t } \gets \nabla _ { \theta } f _ { t } ( \theta _ { t - 1 } )$ {Get gradients w.r.t. stochastic objective at $t \}$   
5: $m _ { t } \gets \beta _ { 1 } m _ { t - 1 } + ( 1 - \beta _ { 1 } ) g _ { t }$   
6: $v _ { t }  \beta _ { 2 } v _ { t - 1 } + ( 1 - \beta _ { 2 } ) g _ { t } ^ { 2 }$   
7: $\hat { m } _ { t } \gets m _ { t } / ( 1 - \beta _ { 1 } ^ { t } )$   
8: $\hat { v } _ { t }  v _ { t } / ( 1 - \beta _ { 2 } ^ { t } )$   
9: if $t > 1$ then   
10: γ<sub>t</sub> ← exp (λ · cosinesimilarity $( g _ { t } , g _ { \mathrm { p r e v } } ) )$   
11: end if   
12: $\tilde { v } _ { t } \gets \operatorname* { m a x } ( \tilde { v } _ { t } , \hat { v } _ { t } )$ {variance envelope}   
13: $\begin{array} { r } { \theta _ { t }  \theta _ { t - 1 } - \eta \cdot \frac { \gamma _ { t } \hat { m } _ { t } } { \sqrt { \tilde { v } _ { t } } + \epsilon { \bf 1 } } } \end{array}$ {Update parameters}   
14: g<sub>prev</sub> $\gets g _ { t }$ {Store gradient for next iteration}   
15: end for   
16: return $\theta _ { t }$

Our regret analysis considers a modified AdamX update designed for online convex optimization. In particular, the analyzed variant removes momentum, uses the current subgradient as the update direction, projects onto a convex feasible set in an adaptive diagonal metric, and controls the alignment-scaled learning rate through a non-increasing envelope. These modifications isolate the efect of the bounded cosine controller while enabling a standard adaptive onlinelearning analysis.

## 3 Regret Guarantees for AdamX-OCO

Scope of the analysis. The practical AdamX optimizer in Algorithm 1 uses momentum and applies the raw alignment multiplier $\gamma _ { t }$ to the update magnitude. To obtain a transparent regret guarantee, we analyze an OCO variant that removes momentum, projects onto a convex feasible set using an adaptive diagonal metric, and replaces the raw alignment-scaled step size with a non-increasing envelope. This variant isolates the efect of the bounded cosine controller while retaining the AMSGrad-style monotone second-moment envelope.

Algorithm 2 AdamX for OCO   
Require: Convex compact set $\mathcal { K } \subset \mathbb { R } ^ { d }$ , sequence $\alpha _ { t } > 0 ,$ parameters $\beta _ { 2 } \in [ 0 , 1 ) , \epsilon > 0$   
$\lambda \ge 0 , \rho \in [ 0 , 1 ]$   
Require: Initial point $\theta _ { 1 } \in \mathcal { K }$   
1: v<sub>0</sub> ← 0, v¯<sub>0</sub> ← 0, q<sub>0</sub> ← +∞, g<sub>0</sub> ← ⊥, γ<sub>1</sub> ← 1   
2: for t = 1 to T do   
3: Play θ<sub>t</sub> and observe $g _ { t } \in \partial f _ { t } ( \theta _ { t } )$   
4: $v _ { t }  \beta _ { 2 } v _ { t - 1 } + ( 1 - \beta _ { 2 } ) g _ { t } ^ { 2 }$   
5: $\hat { v } _ { t }  v _ { t } / ( 1 - \beta _ { 2 } ^ { t } )$   
6: v˜<sub>t</sub> ← max $\{ \tilde { v } _ { t - 1 } , \hat { v } _ { t } \}$   
7: if t > 1 then   
8: $c _ { t } \gets \langle g _ { t } , g _ { t - 1 } \rangle / ( \lVert g _ { t } \rVert _ { 2 } \lVert g _ { t - 1 } \rVert _ { 2 } )$   
9: γ<sub>t</sub> ← exp(λc<sub>t</sub>) {AdamX}   
10: end if   
11: $q _ { t } \gets \operatorname* { m i n } \{ q _ { t - 1 } , \alpha _ { t } \gamma _ { t } \}$ {OCO Adaptation}   
12: $H _ { t } \gets \mathrm { d i a g } ( \sqrt { \tilde { v } _ { t } } + \epsilon \mathbf { 1 } )$   
13: $\theta _ { t + 1 } \gets \mathrm { a r g m i n } _ { \theta \in \mathcal { K } } \left\| \dot { \theta } - ( \theta _ { t } - q _ { t } H _ { t } ^ { - 1 } g _ { t } ) \right\| _ { H _ { t } } ^ { 2 }$   
14: end for   
15: return $\theta _ { T + 1 }$

Adaptive projected-gradient interpretation. Define the weighted norm $\| x \| _ { H _ { t } } ^ { 2 } =$ $x ^ { \top } H _ { t } x$ and the efective metric

$$
A _ { t } = \frac { H _ { t } } { q _ { t } } .\tag{8}
$$

Since multiplication of the projection metric by a positive scalar does not change the projection, Algorithm 2 can equivalently be written as

$$
\theta _ { t + 1 } = \mathrm { a r g m i n } _ { \theta \in \mathcal { K } } \left\{ \langle g _ { t } , \theta \rangle + \frac { 1 } { 2 } \left. \theta - \theta _ { t } \right. _ { A _ { t } } ^ { 2 } \right\} .\tag{9}
$$

The monotone envelope $q _ { t }$ is introduced solely for analysis: together with the monotone second-moment envelope, it ensures that $A _ { t }$ is coordinate-wise nondecreasing.

## Convex Regret Guarantee

Online convex optimization setting. At round t, the learner chooses $\theta _ { t } \in \mathcal { K } .$ observes a convex loss $f _ { t } : { \mathcal { K } }  \mathbb { R }$ , and receives a subgradient $g _ { t } \in \partial f _ { t } ( \theta _ { t } )$ . For any comparator $u \in \kappa$ , the regret is

$$
R _ { T } ( u ) = \sum _ { t = 1 } ^ { T } \left( f _ { t } ( \theta _ { t } ) - f _ { t } ( u ) \right) .\tag{10}
$$

By convexity,

$$
R _ { T } ( u ) \leq \sum _ { t = 1 } ^ { T } \left. g _ { t } , \theta _ { t } - u \right. .\tag{11}
$$

Assumption 1 (Bounded domain and gradients) There exist constants $D _ { \infty } >$ 0 and $G _ { \infty } > 0$ such that, for every $\theta , u \in \mathcal { K }$ and every t,

$$
\left\| \theta - u \right\| _ { \infty } \leq D _ { \infty } , \qquad \left\| g _ { t } \right\| _ { \infty } \leq G _ { \infty } .\tag{12}
$$

Bounded alignment controller. The stabilized cosine similarity satisfies $c _ { t } ~ \in$ $[ - 1 , 1 ]$ , and therefore the AdamX alignment multiplier obeys

$$
e ^ { - \lambda } \leq \gamma _ { t } = \exp ( \lambda c _ { t } ) \leq e ^ { \lambda }\tag{13}
$$

for every t (with $\gamma _ { 1 } = 1$ by definition).

Lemma 1 (One-step adaptive projected-gradient bound). For every $u \in$ $\kappa ,$

$$
\langle g _ { t } , \theta _ { t } - u \rangle \le \frac { 1 } { 2 } \left( \left. \theta _ { t } - u \right. _ { A _ { t } } ^ { 2 } - \left. \theta _ { t + 1 } - u \right. _ { A _ { t } } ^ { 2 } \right) + \frac { 1 } { 2 } \left. g _ { t } \right. _ { A _ { t } ^ { - 1 } } ^ { 2 } .\tag{14}
$$

Equivalently,

$$
\left. g _ { t } , \theta _ { t } - u \right. \leq \frac { 1 } { 2 q _ { t } } \left( \left\| \theta _ { t } - u \right\| _ { H _ { t } } ^ { 2 } - \left\| \theta _ { t + 1 } - u \right\| _ { H _ { t } } ^ { 2 } \right) + \frac { q _ { t } } { 2 } \left\| g _ { t } \right\| _ { H _ { t } ^ { - 1 } } ^ { 2 } .\tag{15}
$$

Proof. The optimality condition of the projected update gives

$$
\langle g _ { t } + A _ { t } ( \theta _ { t + 1 } - \theta _ { t } ) , u - \theta _ { t + 1 } \rangle \geq 0 .
$$

Combining this inequality with

$$
2 \left. \theta _ { t } - \theta _ { t + 1 } , A _ { t } ( \theta _ { t } - u ) \right. = \left\| \theta _ { t } - \theta _ { t + 1 } \right\| _ { A _ { t } } ^ { 2 } + \left\| \theta _ { t } - u \right\| _ { A _ { t } } ^ { 2 } - \left\| \theta _ { t + 1 } - u \right\| _ { A _ { t } } ^ { 2 } .\tag{16}
$$

Applying Young’s inequality to $\langle g _ { t } , \theta _ { t } - \theta _ { t + 1 } \rangle$ yields the result.

Convex Regret Bound for AdamX-OCO

Theorem 1 (Convex OCO regret). Suppose Assumption 1 holds. Run Algorithm 2 with $\alpha _ { t } = \eta / \sqrt { t }$ for some $\eta > 0$ . Then for every $u \in \kappa$

$$
R _ { T } ( u ) \leq \frac { D _ { \infty } ^ { 2 } } { 2 q _ { T } } \sum _ { i = 1 } ^ { d } H _ { T , i } + \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } q _ { t } \sum _ { i = 1 } ^ { d } \frac { g _ { t , i } ^ { 2 } } { H _ { t , i } } .\tag{17}
$$

Moreover, using the coarse bounds $\epsilon \leq H _ { t , i } \leq G _ { \infty } + \epsilon ,$

$$
R _ { T } ( u ) \leq \frac { d D _ { \infty } ^ { 2 } ( G _ { \infty } + \epsilon ) } { 2 \eta e ^ { - \lambda } } \sqrt { T } + \frac { \eta e ^ { \lambda } d G _ { \infty } ^ { 2 } } { \epsilon } \sqrt { T } .\tag{18}
$$

Consequently,

$$
R _ { T } ( u ) = O ( { \sqrt { T } } ) .
$$

Proof. By convexity,

$$
R _ { T } ( u ) \leq \sum _ { t = 1 } ^ { T } \left. g _ { t } , \theta _ { t } - u \right. .
$$

Applying Lemma 1 and summing over t gives

$$
R _ { T } ( u ) \leq \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \Big ( \| \theta _ { t } - u \| _ { A _ { t } } ^ { 2 } - \| \theta _ { t + 1 } - u \| _ { A _ { t } } ^ { 2 } \Big ) + \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \| g _ { t } \| _ { A _ { t } ^ { - 1 } } ^ { 2 } .
$$

Because $\tilde { v } _ { t }$ is coordinatewise nondecreasing and $q _ { t }$ is nonincreasing, the matrix sequence $A _ { t } = H _ { t } / q _ { t }$ is positive semidefinite nondecreasing. Therefore the first sum telescopes with an additional nonnegative metric-growth term and can be bounded as

$$
\frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \Big ( \left. \theta _ { t } - u \right. _ { A _ { t } } ^ { 2 } - \left. \theta _ { t + 1 } - u \right. _ { A _ { t } } ^ { 2 } \Big ) \leq \frac { 1 } { 2 } \left. \theta _ { 1 } - u \right. _ { A _ { 1 } } ^ { 2 } + \frac { 1 } { 2 } \sum _ { t = 2 } ^ { T } \left. \theta _ { t } - u \right. _ { A _ { t } - A _ { t - 1 } } ^ { 2 } .
$$

Since $A _ { t }$ is diagonal and $\| \theta _ { t } - u \| _ { \infty } \leq D _ { \infty }$ , this is at most

$$
\frac { D _ { \infty } ^ { 2 } } { 2 } \sum _ { i = 1 } ^ { d } { A _ { T , i } } = \frac { D _ { \infty } ^ { 2 } } { 2 q _ { T } } \sum _ { i = 1 } ^ { d } { H _ { T , i } } .
$$

The second term is

$$
\frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \left. g _ { t } \right. _ { A _ { t } ^ { - 1 } } ^ { 2 } = \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } q _ { t } \sum _ { i = 1 } ^ { d } \frac { g _ { t , i } ^ { 2 } } { H _ { t , i } } .
$$

This proves the first bound.

Since $\gamma _ { t } \in [ e ^ { - \lambda } , e ^ { \lambda } ]$ and $\alpha _ { t } = \eta / \sqrt { t } .$ the envelope satisfies

$$
q _ { T } \geq \frac { \eta e ^ { - \lambda } } { \sqrt { T } } , \qquad q _ { t } \leq \frac { \eta e ^ { \lambda } } { \sqrt { t } } .
$$

Also $H _ { t , i } \geq \epsilon$ and, under $\| g _ { t } \| _ { \infty } \leq G _ { \infty }$ , the second-moment and scalar rectification terms are bounded so that $H _ { t , i } \leq G _ { \infty } + \epsilon$ . Hence

$$
\frac { D _ { \infty } ^ { 2 } } { 2 q _ { T } } \sum _ { i = 1 } ^ { d } H _ { T , i } \leq \frac { d D _ { \infty } ^ { 2 } ( G _ { \infty } + \epsilon ) } { 2 \eta e ^ { - \lambda } } \sqrt { T } .
$$

For the second term,

$$
\frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } q _ { t } \sum _ { i = 1 } ^ { d } \frac { g _ { t , i } ^ { 2 } } { H _ { t , i } } \leq \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \frac { \eta e ^ { \lambda } } { \sqrt { t } } \frac { d G _ { \infty } ^ { 2 } } { \epsilon } \leq \frac { \eta e ^ { \lambda } d G _ { \infty } ^ { 2 } } { \epsilon } \sqrt { T } .
$$

Combining the two inequalities gives the stated result.

Remark 1 (Efect of the alignment parameter). The regret rate is unchanged by the alignment factor, but the constants scale with $e ^ { \lambda }$ . This is expected: the multiplier $\gamma _ { t }$ is bounded between $e ^ { - \lambda }$ and $e ^ { \lambda }$

Using Raw Alignment Instead of the Envelope The monotone envelope $q _ { t } ~ =$ min $\{ q _ { t - 1 } , \alpha _ { t } \gamma _ { t } \}$ is theoretically convenient, but it removes part of the intended behavior of AdamX: when gradients become strongly aligned, the method cannot re-increase the efective step size if the envelope has already decreased.

If one instead uses the raw efective step size

$$
\boldsymbol { r } _ { t } = \boldsymbol { \alpha } _ { t } \gamma _ { t } ,
$$

and defines

$$
A _ { t } = \frac { H _ { t } } { r _ { t } } ,
$$

then $A _ { t }$ need not be monotone, even when $H _ { t }$ is monotone. The proof still yields a data-dependent variation bound.

Proposition 1 (Variation-dependent regret with raw alignment). Consider the AdamX-OCO update with $r _ { t } = \alpha _ { t } \gamma _ { t }$ instead of the monotone envelope $q _ { t }$ . Then for every $u \in \kappa$

$$
R _ { T } ( u ) \leq \frac { 1 } { 2 } \left. \theta _ { 1 } - u \right. _ { A _ { 1 } } ^ { 2 } + \frac { 1 } { 2 } \sum _ { t = 2 } ^ { T } \left. \theta _ { t } - u \right. _ { ( A _ { t } - A _ { t - 1 } ) _ { + } } ^ { 2 } + \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \left. g _ { t } \right. _ { A _ { t } ^ { - 1 } } ^ { 2 } ,\tag{19}
$$

where $\left( A _ { t } - A _ { t - 1 } \right) _ { + }$ denotes the positive part of the symmetric matrix $A _ { t } - A _ { t - 1 }$

This statement is more faithful to the practical optimizer. It says that AdamX-OCO keeps sublinear regret when the metric variation induced by the denominator and the alignment factor is controlled. In adversarial sequences, however, the cosine signal can oscillate, and the variation term may be large.

Momentum and the Full AdamX Algorithm The full AdamX update uses $\hat { m } _ { t }$ rather than $g _ { t }$ . In $\mathrm { O C O } .$ convexity gives

$$
f _ { t } ( \theta _ { t } ) - f _ { t } ( u ) \leq \langle g _ { t } , \theta _ { t } - u \rangle ,
$$

whereas the projected update controls a term involving $\hat { m } _ { t }$ . Consequently,

$$
\left. g _ { t } , \theta _ { t } - u \right. = \left. \hat { m } _ { t } , \theta _ { t } - u \right. + \left. g _ { t } - \hat { m } _ { t } , \theta _ { t } - u \right. .
$$

The first term can be handled by adaptive mirror descent. The second is a momentum-bias term. A direct bound gives

$$
\sum _ { t = 1 } ^ { T } \left. g _ { t } - \hat { m } _ { t } , \theta _ { t } - u \right. \leq D _ { \infty } \sum _ { t = 1 } ^ { T } \left. g _ { t } - \hat { m } _ { t } \right. _ { 1 } ,\tag{20}
$$

which can be linear in T for adversarial gradient sequences [7].

## 4 Experiments

We evaluate the proposed algorithm against comparable optimizers using an experimental protocol inspired by DeepOBS [19] and AlgoPerf [6]. When designing empirical evaluations for deep learning optimizers, we focus on three key aspects. (1) Generalization. The goal of optimization in deep learning is to learn models that generalize well to unseen data. Although some prior studies focus primarily on training metrics, improvements in training loss do not necessarily translate into better test performance. Accordingly, our evaluation emphasizes test-set performance throughout. (2) Stochasticity. The observed performance can vary substantially due to random initialization. To mitigate the influence of these sources of randomness and ensure fair comparisons, all optimizers are evaluated using the same set of five random seeds, and results are reported as averages across runs. (3) Realistic evaluation setting. Optimizer performance is highly dependent on the model architecture and dataset. Consequently, we adopt established benchmark architectures from DeepOBS [19] together with widely used datasets, ensuring evaluation on representative and commonly studied tasks. For consistency, we evaluate all optimizers, including ours, with the default hyperparameters [18].

Following this principles, the main evaluation tool is the number of epochs necessary to achieve a predetermined test set accuracy. Unlike AlgoPerf[6], which is more focused on algorithmic speed, we do not evaluate on wall-clock runtime, which has well-known drawbacks, such as dependency on hardware or weak reproducibility. By evaluating on epochs, we evaluate performance against the number of gradient evaluations, which typically dominates the total computational costs.

Baseline Algorithms To evaluate AdamX, we compare against widely used first-order optimizers spanning adaptive, momentum-based, and non-adaptive methods. Specifically, we consider SGD [16]; Adagrad, which accumulates squared historical gradients [8]; RMSProp, which replaces Adagrad’s cumulative statistic with an exponential moving average [1]; Adam [12]; AdamW, which decouples weight decay from adaptive updates [14]; AMSGrad, which enforces a non-decreasing second-moment estimate [15]; RAdam, which introduces variance rectification during early training [13]; Yogi, which controls excessive growth of the variance estimate [22]; Lion, which updates parameters using the sign of the momentum vector [5]; and Adan, which incorporates Nesterov-style momentum into adaptive moment estimation [21].

MNIST On MNIST, we use a three-layer CNN with default hyperparameters and measure the epochs required to reach 0.994 test accuracy, up to 100 epochs. Figure 1 shows that AdamX is competitive with the best-performing optimizers, AMSGrad and Yogi. SGD does not reach the target, RMSprop fails in all runs due to gradient collapse, and Adagrad shows the largest variance across seeds. The training-loss curves in Figure 2 are consistent with these results, with AMSGrad, AdamX, and Yogi among the fastest methods to reach the target.

![](images/14704b88e5cefab766ed0ae644d225bf9970618307cf37bd474d312a6bc5ea9c.jpg)

![](images/14fd967b60308af808ec47fb941eed0581ae563f5f63a3b6d44d3ba983d43f9a.jpg)  
Fig. 1. Number of epochs to reach the test accuracy target. Each optimizer is evaluated over five seeds. Yogi, AMS-Grad, and AdamX achieve the best performance, while SGD and RMSProp fail to reach the target.  
Fig. 2. Training loss over 100 epochs on MNIST. Methods with better generalization (AMSGrad, Yogi, AdamX) also exhibit lower training loss. RMSProp is omitted due to significantly higher loss values.

CIFAR-10 CIFAR-10 is more challenging than MNIST; to focus on optimizer behavior, we use the fixed CifarNet architecture [11]. The target test accuracy is 0.84, with a maximum of 100 epochs.

Figure 3 shows that five out of eleven optimizers fail to reach the target, indicating that the threshold captures a demanding training regime. AdamX, AMSGrad, and Adam are the strongest methods, with AdamX achieving the lowest mean number of epochs and outperforming the closely related AMSGrad baseline. Figure 4 further shows that AdamX, AMSGrad, and Adan exhibit smoother training-loss trajectories, whereas RAdam, Adam, and AdamW display larger oscillations.

Table 1 summarizes the results. Overall, AdamX and AMSGrad require the fewest gradient evaluations to reach the target, with AdamX comparing favorably on CIFAR-10. The results also illustrate that lower training loss does not necessarily imply better generalization; for example, Adagrad obtains a low MNIST training loss but requires more epochs to reach the test-accuracy target.

## 5 Conclusions

We presented AdamX, an adaptive first-order optimizer that augments Adam/AMSGradstyle updates with a bounded cosine similarity controller. A simplified OCO analysis shows that, under a monotone envelope on the cosine-scaled step size, the alignment mechanism is compatible with standard adaptive regret guarantees. Empirically, AdamX is competitive with ten established optimizers and

![](images/0786fbed71bf93517fa756c6b5d06d4b2f94a356d6b1dc58927f468449e6fb80.jpg)

![](images/a07448df4c11579809ac6963ec1161092dee080abc202f1a95f04852b216845b.jpg)  
Fig. 3. Number of epochs to reach the test accuracy target. Each optimizer is evaluated over five seeds. AdamX obtains the lowest mean number of epochs to reach the target, with similar values ob tained by AMSGrad and Adam.  
Fig. 4. Training loss over 100 epochs on CIFAR-10. Methods with better generalization (AMSGrad, Yogi, AdamX) also exhibit lower training loss. RMSProp is omitted due to significantly higher loss values.

Table 1. Number of epochs until target accuracy, and training loss at 100 Epochs. Lower is better. Maximum number of runs is 100. Best, second-best, and third-best results are highlighted.
<table><tr><td></td><td colspan="2">MNIST</td><td colspan="2">CIFAR-10</td></tr><tr><td>Optimizer</td><td>Epochs (↓) Train Loss (↓)</td><td> $( \times 1 0 ^ { - 5 } )$ </td><td></td><td>Epochs (↓) Train Loss(↓)</td></tr><tr><td>AdamX (Ours)</td><td> $1 3 . 2 \pm 2 . 0 0$ </td><td> $\it { 2 . 1 7 \pm 0 . 4 1 }$ </td><td> ${ \bf 6 . 6 \pm 0 . 5 4 }$ </td><td> $\it { 1 . 9 6 e  – 0 3 }$ </td></tr><tr><td>Yogi</td><td> ${ \bf 9 . 2 \pm 0 . 7 3 4 }$ </td><td> $\underline { { 0 . 7 9 \pm 0 . 1 3 } }$ </td><td>100</td><td>9.59e-03</td></tr><tr><td>SGD</td><td>100</td><td> $2 5 . 0 0 \pm 0 . 5 2$ </td><td>100</td><td>2.30</td></tr><tr><td>RMSProp</td><td>100</td><td> $1 6 0 9 . 9 7 \pm 2 9 5 . 4 6$ </td><td> $2 4 . 2 \pm 2 8 . 9 4$ </td><td>2.64e-03</td></tr><tr><td>Radam</td><td> $2 5 . 2 \pm 6 . 6 7$ </td><td> $9 0 . 3 9 \pm 1 3 . 2 9$ </td><td> $3 0 . 2 \pm 2 4 . 9 3$ </td><td>2.72e-03</td></tr><tr><td>Lion</td><td> $4 2 . 0 \pm 1 0 . 3 6$ </td><td> $3 9 7 . 4 0 \pm 4 . 0 7$ </td><td>100.0</td><td>6.42e-03</td></tr><tr><td>AMSGrad</td><td> ${ \bf 9 . 2 \pm 1 . 3 5 }$ </td><td> ${ \bf 0 . 4 4 \pm 0 . 0 2 }$ </td><td> $\frac { 7 . 8 \pm 1 . 3 0 } { 1 0 0 }$ </td><td>2.34e-03</td></tr><tr><td>Adan</td><td> $4 4 . 4 \pm 1 0 . 1 9$ </td><td> $8 9 . 9 9 \pm 1 2 . 0 0$ </td><td></td><td>2.16e-03</td></tr><tr><td>AdamW</td><td> $2 4 . 2 \pm 2 . 7 3$ </td><td> $1 1 2 . 0 3 \pm 6 7 . 3 9$ </td><td> $1 6 . 6 \pm 1 1 . 6 5$ </td><td>1.60e-03</td></tr><tr><td>Adam</td><td> $2 8 . 0 \pm 4 . 0 9$ </td><td> $1 0 0 . 3 3 \pm 3 5 . 0 5$ </td><td> $g . 6 \pm \ : 5 . 8 6$ </td><td>1.72e-03</td></tr><tr><td>Adagrad</td><td> $6 4 . 4 \pm 2 1 . 3 9$ </td><td> $7 . 3 2 \pm 1 . 6 4 5$ </td><td>100</td><td>5.08e-02</td></tr></table>

achieves the best result in the considered CIFAR-10 setting. These results indicate that directional alignment is a promising lightweight source of adaptivity, motivating future work on hyperparameter robustness, second-order extensions, and larger-scale training regimes.

![](images/2a191043f51826996e080b7020eb227b98d5a9c6334d64fe4cd9c6714f4feb31.jpg)

Acknowledgments. This work was partially supported by NOVA LINCS (UID/04516) funded by FCT IP, and the Neuraspace AI Fights Space Debris project (C626449889- 00463050), co-funded by Recovery and Resilience Plan and NextGeneration EU Funds, www.recuperarportugal.gov.pt. The authors have no competing interests to declare that are relevant to the content of this article.

REPÚBLICA PORTUGUESA

## References

1. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks machine learning 4(2), 26 (2012)

2. Almeida, L.B., Langlois, T., Amaral, J.D., Plakhov, A.: Parameter adaptation in stochastic optimization. In: On-Line Learning Neural Networks. CUP (1998)

3. Baydin, A., Cornish, R., Rubio, D., Schmidt, M., Wood, F.: Online learning rate adaptation with hypergradient descent. In: ICLR (2018)

4. Cauchy, A.L.: Méthode générale pour la résolution des systèmes d’équations simultanées. Comptes Rendus Hebd Seances Acad Sci 25, 536–538 (1847)

5. Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong, X., Luong, T., Hsieh, C.J., Lu, Y., Le, Q.V.: Symbolic discovery of optimization algorithms. In: NeurIPS (2023)

6. Dahl, G.E., Schneider, F., Nado, Z., Agarwal, N., Sastry, C.S., Hennig, P., Medapati, S., Eschenhagen, R., Kasimbeg, P., Suo, D., Bae, J., Gilmer, J., Peirson, A.L., Khan, B., Anil, R., Rabbat, M., Krishnan, S., Snider, D., Amid, E., Chen, K., Maddison, C.J., Vasudev, R., Badura, M., Garg, A., Mattson, P.: Benchmarking Neural Network Training Algorithms. arXiv preprint arXiv:2306.07179 (2023)

7. Défossez, A., Bottou, L., Bach, F., Usunier, N.: A simple convergence proof of Adam and Adagrad. TMLR (2022)

8. Duchi, J., Hazan, E., Singer, Y.: Adaptive subgradient methods for online learning and stochastic optimization. JMLR 12, 2121–2159 (2011)

9. Gupta, V., Koren, T., Singer, Y.: Shampoo: Preconditioned stochastic tensor optimization. In: ICML. pp. 1842–1850. PMLR (2018)

10. Jiang, R., Kavis, A., Mokhtari, A.: Online learning-guided learning rate adaptation via gradient alignment. arXiv preprint arXiv:2506.08419 (2025)

11. Jordan, K., Jin, Y., Boza, V., Jiacheng, Y., Cesista, F., Newhouse, L., Bernstein, J.: Muon: An optimizer for hidden layers in neural networks (2024)

12. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: ICLR (2015)

13. Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., Han, J.: On the variance of the adaptive learning rate and beyond. In: ICLR (2020)

14. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019)

15. Reddi, S.J., Kale, S., Kumar, S.: On the convergence of adam and beyond. In: ICLR (2018)

16. Robbins, H., Monro, S.: A stochastic approximation method. Annals Mathematical Statistics 22(3), 400–407 (1951)

17. Rubio, D.M.: Convergence Analysis of an Adaptive Method of Gradient Descent. Msc thesis, U. Oxf. (2017)

18. Schmidt, R.M., Schneider, F., Hennig, P.: Descending through a crowded valley - benchmarking deep learning optimizers. In: ICML (2021)

19. Schneider, F., Balles, L., Hennig, P.: DeepOBS: A deep learning optimizer benchmark suite. In: ICLR (2019)

20. Vyas, N., Morwani, D., Zhao, R., Shapira, I., Brandfonbrener, D., Janson, L., Kakade, S.: SOAP: Improving and stabilizing shampoo using adam for language modeling. In: ICLR (2025)

21. Xie, X., Zhou, P., Li, H., Lin, Z., Yan, S.: Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models. IEEE TPAMI (2024)

22. Zaheer, M., Reddi, S., Sachan, D., Kale, S., Kumar, S.: Adaptive methods for nonconvex optimization. In: NeurIPS. vol. 31 (2018)

23. Zhuang, J., Tang, T., Ding, Y., Tatikonda, S.C., Dvornek, N., Papademetris, X., Duncan, J.: Adabelief optimizer: Adapting stepsizes by the belief in observed gradients. NeurIPS (2020)