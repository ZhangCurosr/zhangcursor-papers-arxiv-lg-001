# AURA: Angular Update Rate Adaptation for training complex-valued neural networks

Enrico Ballini<sup>1</sup>, Allan Peter Engsig-Karup<sup>2</sup>, and Tito Andriollo<sup>1</sup>

Abstract—Complex-valued neural networks (CVNNs) are increasingly adopted for complex-valued data; however, they are often trained with first-order optimizers inherited from the realvalued case. The efficiency of these methods depends largely on the step size, and their step-size rules ignore the angular information available in the complex plane. We address stepsize adaptation in the complex domain by introducing AURA (Angular Update Rate Adaptation), a per-parameter step-size adaptation that can be added on top of any first-order optimizer, and removed from it, without altering its update direction. AURA measures the agreement between consecutive updates of each complex parameter, in length, alignment, and sense of rotation, and enlarges the step when they are consistent and reduces it when they are not. It requires no additional gradient evaluations and only inexpensive vector operations per step. We combine AURA with Adam and Muon and compare the resulting methods with well-known first-order optimizers on four test cases of increasing complexity, ranging from the approximation of scalar complex functions to physics-informed training. Fully connected neural networks are used throughout this work. All hyperparameters other than the step size are held fixed across test cases; for one case, we also tune the hyperparameters of each optimizer under the same budget. Our empirical tests show that AURA improves the convergence of its base optimizer in most cases with a small per-step overhead, and we identify the conditions under which it fails to do so.

Index Terms—Complex-valued neural networks, adaptive opti mization, step-size adaptation

## I. INTRODUCTION

Neural networks are trained with iterative optimizers, ranging from first- to second-order methods and from heuristic rules to principled ones [1]. First-order methods are the standard in deep learning because each step is cheap, but their efficiency depends largely on the step size. The classical levers are the gradient history and its reliability: momentum lets noise cancel and consistent directions accumulate, while the signal-to-noise ratio (SNR) of the gradient decides how far a step can be trusted, as in AdaBelief [2].

A further device to adjust the step size is the agreement between consecutive gradients or updates; for real-valued networks, AngularGrad [3] rescales the Adam step with an angle-like measure of consecutive gradients, diffGrad [4] with their difference, and AdaSmooth [5] sets the averaging window of the RMSProp denominator from the ratio between the net and the total displacement of a weight. Hypergradient descent [6], applied earlier to complex-valued adaptive filters [7], [8], learns the step size itself by gradient descent, and its multiplicative form updates the step size with the cosine between the current gradient and the previous update. Adaptive methods such as AdaGrad [9] and AdaDelta [10] keep one step size per parameter, and CoRe [11], [12] shows that adding an RPROP-like sign test to Adam at this level can be effective in real-valued networks.

Whether such rules actually speed up training can only be judged through fair comparisons. Benchmarking practices are set in, e.g., [13] and [14], based on standard test problems and explicit tuning budgets, and take into account that the hyperparameter search space can change the ranking [15].

The above works focused on the real-valued case, see also [16]. For complex-valued neural networks (CVNNs), [17] compare adaptive optimizers in online training only, and [18] show that the specific optimizer changes how CVNNs compare with real-valued neural networks (RVNNs).

CVNNs process complex-valued data directly and are used in image classification [19], image reconstruction, signal classification, speech enhancement, wind prediction, and control systems [20]. They are a natural choice in telecommunications [17], and in MRI reconstruction they outperform real-valued CNNs with the same number of parameters [21]. Their advantage is not universal, however: [22] find RVNNs as good or better in most cases, and the optimizer is part of the reason [18].

Several training algorithms have been designed for CVNNs, from metacognitive learning in complex RBF networks [23] to conjugate gradients and selectable search directions with Armijo [24] and Wolfe [25] line searches. Adaptive optimizers, instead, have been ported from the real-valued case by treating the real and imaginary parts separately: [26] extend Adam in this way, and [17] extend AdaGrad [9], RMSProp [27], AdaMax [28], AMSGrad [29], SAMSGrad [30], Nadam [31], and diffGrad [4]. Their step-size rules therefore act on each part independently and measure no angle in the complex plane.

The complex plane also allows a complex step size, which rotates the update and extends the search from a ray to a half-plane: [32] show that the complex step size approximates Hessian information better than a real one, and [33] and [34] adapt it during training. The authors of [35] propose to adjust a complex step size, shared by all parameters, according to its phase, obtained by comparing gradient norms at three trial points; the angle thus comes from flatness rather than from consecutive updates, at the cost of three extra gradient evaluations per step. Methods based on curvature or line searches are also more expensive per step: HCSCGM [36] and its extension with an adaptive complex step size, HCSCGACS [37], rely on complex inexact line searches, AS-CNAG [38] sets the step from an approximate Hessian, and [39], [40] develop adaptive complex L-BFGS methods. We therefore focus on optimizers that use one stochastic gradient per step and inexpensive vector operations.

This work builds on the idea that the angle between consecutive updates is an informative signal for step-size control in CVNNs. We propose AURA, a per-parameter stepsize multiplier applicable on top of any optimizer, which enlarges the step when consecutive update directions are aligned and of similar length and reduces it when they are opposed or keep rotating in the same sense. We combine AURA with Adam [28] and Muon [41] and evaluate it on four problems with fully connected neural networks, from scalar complex functions to more complicated physics-informed training. Unlike realvalued rules, which can only test whether consecutive updates agree in sign and magnitude, AURA also measures how much they rotate in the complex plane, in which sense they rotate, and how similar they are in magnitude. The building blocks of AURA have already been used in the cited literature, but their combination for complex parameters is new. Section II presents the AURA algorithm, Section III relates it to the closest optimizers, Section IV reports the case studies, and Section V concludes.

## II. AURA

The method we propose scales the update direction of a base optimizer by a per-parameter multiplier, which, in a nutshell, increases when consecutive directions agree in the complex plane and decreases when they do not. This section fixes the notation, states the algorithm, and explains the building blocks of the algorithm.

a) Notation: Let $w _ { j , t } = x _ { j , t } + \mathrm { i } y _ { j , t } \in \mathbb { C } , j = 1 , \dots , N$ be the N trainable parameters at iteration $t \in \mathbb { N } .$ , and $\mathcal { L } _ { t }$ $\mathbb { C } ^ { N }  \mathbb { R }$ the loss at step t, e.g., the mini-batch loss (12). Being real-valued, $\mathcal { L } _ { t }$ is not holomorphic and is differentiated as a map $\mathbb { R } ^ { 2 N } \to \mathbb { R }$ under $\mathbb { C } ^ { N } \simeq \mathbb { R } ^ { 2 N }$ . The gradient

$$
g _ { j , t } = \frac { \partial \mathcal { L } _ { t } } { \partial x _ { j , t } } + \mathrm { i } \frac { \partial \mathcal { L } _ { t } } { \partial y _ { j , t } } ,\tag{1}
$$

combines the two real partial derivatives into a single complex number, and $- \mathit { g _ { j , t } }$ is the steepest-descent direction of $\mathcal { L } _ { t }$ in $w _ { j , t }$

b) Algorithm: Given the direction $d _ { j , t }$ of a base optimizer , AURA computes a multiplier $\gamma _ { j , t } > 0$ and takes the step $\alpha \gamma _ { j , t } d _ { j , t }$ . With the initial values $\dot { d } _ { j , 0 } = 0 , \ : \tilde { \zeta } _ { j , 0 } = 0$ , and $\gamma _ { j , 0 } = 1$ , and suitable choices for the hyperparameters listed in Table I and discussed below, the AURA algorithm takes the following form for every $j$ and $t \geq 1$

$$
\begin{array}{c} \begin{array} { r l } & { \overbrace { \frac { \partial } { \partial } \underbrace { \dot { \bigtriangledown } } } ^ { \updownarrow } } \\ & { \overbrace { \frac { \partial } { \partial } \underbrace { \dot { \bigtriangledown } } } ^ { \updownarrow } } \\ & { \overbrace { \frac { \partial } { \partial } \underbrace { \dot { \bigtriangledown } } } ^ { \bigstar \bigtriangledown } } \\ & { \overbrace { \frac { \partial } { \partial } \geq } ^ { \bigstar \bigtriangledown } } \\ & { \overbrace { \frac { \partial } { \partial } \geq } ^ { \bigstar \bigtriangledown } } \\ & { \overbrace { \bigtriangleup } ^ { \bigstar } } \\ & { \overbrace { \frac { \bigtriangledown } { \bigtriangleup } } ^ { \bigstar } } \end{array} d _ { j , t } = \mathcal { G } _ { j } \big ( \big \{ g _ { k , s } \big \} _ { k \leq N , s \leq t } \big ) ,  \end{array}\tag{2a}
$$

$$
\begin{array} { r l } & { \stackrel { \triangledown } { \overbrace { \sum } } \underset { \geq } { \underbrace { \frac { \partial } { \partial } } } \left( \begin{array} { l } { \zeta _ { j , t } = \frac { 2 d _ { j , t } \overline { { d _ { j , t - 1 } } } } { | d _ { j , t } | ^ { 2 } + | d _ { j , t - 1 } | ^ { 2 } + \varepsilon _ { \mathrm { E } } } , \qquad | \zeta _ { j , t } | < 1 , } \\ { \tilde { \overline { { \Xi } } } \frac { \partial } { \partial \mathbf { \Sigma } } \frac { \partial } { \partial \mathbf { \Sigma } } } \\ { \tilde { \overline { { \Xi } } } \varepsilon \cdot \overleftarrow { \Xi } } \\ { \tilde { \overline { { \Xi } } } ^ { \ast } } \end{array} \right) } \\ & { \stackrel { \triangledown } { \overbrace { \Xi } } \left( \begin{array} { l } { \zeta _ { j , t } = \mathcal { \mathrm { R } } \mathrm { e } ( \widehat { \zeta } _ { j , t - 1 } + ( 1 - \beta _ { \zeta } ) \zeta _ { j , t } , \qquad \widehat { \zeta } _ { j , t } = \frac { \tilde { \zeta } _ { j , t } } { 1 - \beta _ { \zeta } ^ { t } } , } \\ { \chi _ { j , t } = \mathrm { R e } ( \widehat { \zeta } _ { j , t } ) \in ( - 1 , 1 ) , } \\ { \Psi _ { j , t } = \mathrm { I m } ( \widehat { \zeta } _ { j , t } ) \in ( - 1 , 1 ) , } \end{array} \right) } \end{array}\tag{2b}
$$

$$
\begin{array}{c} \begin{array}{c} \begin{array} { r l } & { \frac { \partial } { \partial \mathbf { \phi } } } \\ & { \frac { \partial } { \partial \mathbf { \phi } } } \\ & { \frac { \hat { \mathbf { r } } } { \partial \mathbf { \phi } } } \\ & { \frac { \hat { \mathbf { r } } } { \partial \mathbf { \phi } } } \\ & { \frac { \partial } { \partial \mathbf { \phi } } } \end{array} ( \begin{array} { l } { \sigma _ { j , t } = ( \chi _ { j , t } \leq \chi _ { \mathrm { o } } ) \mathrm { ~ o r ~ } ( | \Psi _ { j , t } | \geq \Psi _ { \mathrm { o } } ) , } \\ { \frac { \partial } { \partial \mathbf { \phi } } } \\ { \frac { \partial } { \partial \mathbf { \phi } } } \end{array} ) \rho _ { j , t } = ( \chi _ { j , t } \geq \chi _ { \mathrm { a } } ) \mathrm { ~ a n d ~ } ( | \Psi _ { j , t } | \leq \Psi _ { \mathrm { a } } ) ,  \\ & { \frac { \partial } { \partial \mathbf { \phi } } } \\ & { \frac { \partial } { \partial \mathbf { \phi } } } \\ & { \frac { \partial } { \partial \mathbf { \phi } } } \end{array} ) \rho _ { j , t } = \{ \begin{array} { l l } { \operatorname* { m a x } \{ \eta _ { - } \gamma _ { j , t - 1 } , \gamma _ { \mathrm { m i n } } \} , } & { \ \sigma _ { j , t } = 1 , } \\ { \operatorname* { m i n } \{ \eta _ { + } \gamma _ { j , t - 1 } , \gamma _ { \mathrm { m a x } } \} , } & { \ \rho _ { j , t } = 1 , } \\ { \gamma _ { j , t - 1 } , } & { \ \mathrm { o t h e r w i s e } , } \end{array}   \end{array}\tag{2c}
$$

$$
w _ { j , t + 1 } = w _ { j , t } - \alpha \gamma _ { j , t } d _ { j , t } - \alpha \lambda w _ { j , t } .\tag{2d}
$$

c) Base optimizer: Line (2a) is any first-order optimizer, e.g., Adam [28] or Muon [41]; the subsequent blocks depend on $d _ { j , t }$ only. With $\gamma _ { j , t } = 1$ for all $j$ and t, AURA reduces to , which shows that the proposed algorithm acts as a plug-in to the base optimizer.

d) Consistency variable: Block (2b) compares $d _ { j , t }$ with $d _ { j , t - 1 }$ through the complex variable $\zeta _ { j , t } .$ , a Dice-like measure [42] (Section A). Under $\mathbb { C } \simeq \mathbb { R } ^ { 2 }$ , the real part of its numerator is the inner product of the two directions and the imaginary part their signed area; the denominator normalizes by their squared lengths, so that $\zeta _ { j , t } \to 1$ for equal directions, $\zeta _ { j , t } \to - 1$ for opposite ones, $\zeta _ { j , t } \to e ^ { \mathrm { i } \theta }$ for a rotation by θ at equal length, and $| \zeta _ { j , t } |$ decreases as the two lengths separate. Unlike a cosine similarity, $\zeta _ { j , t }$ therefore captures length, angle, and sense of rotation simultaneously; the safeguard $\varepsilon _ { \mathrm { { E } } } > 0$ only prevents division by zero. The moving average $\bar { \zeta } _ { j , t } ,$ with decay rate $\beta _ { \zeta }$ , retains the memory of past directions and, since it starts from zero, is bias-corrected. The real part $\chi _ { j , t }$ of $\widehat { \zeta } _ { j , t }$ is the smoothed alignment of consecutive directions, the imaginary part $\Psi _ { j , t }$ their smoothed signed rotation: rotations of the same sense accumulate in $\Psi _ { j , t }$ , alternating ones cancel.

e) Multiplier and update: Block (2c) converts the two statistics into $\gamma _ { j , t }$ by a three-way test, conceptually similar to the strategies delta-bar-delta [43] and RPROP [44]: the multiplier is increased by $\eta _ { + } > 1$ when the flag $\rho _ { j , t }$ holds, decreased by $\eta _ { - } \in ( 0 , 1 )$ when $\sigma _ { j , t }$ holds, and left unchanged otherwise, always within [γ<sub>min</sub>, γ<sub>max</sub>]. Alignment alone is not sufficient for an increase: a sequence of small rotations of constant sense keeps $\chi _ { j , t }$ close to 1 but accumulates in $\Psi _ { j , t } .$ An increase therefore requires both $\chi _ { j , t } \geq \chi _ { \mathrm { a } }$ and $| \Psi _ { j , t } | \leq \Psi _ { \mathrm { a } } ,$ whereas either $\chi _ { j , t } \leq \chi _ { \mathrm { o } }$ or $| \Psi _ { j , t } | \geq \Psi _ { \mathrm { o } }$ forces a decrease;

TABLE I  
HYPERPARAMETERS OF AURA
<table><tr><td>Symbol</td><td>Range Role</td></tr><tr><td colspan="2">Consistency variable (2b)</td></tr><tr><td> $\beta _ { \zeta }$   $[ 0 , 1 )$   $\varepsilon _ { \mathrm { E } }$   $> 0$ </td><td>Decay rate of the average  $\tilde { \zeta } _ { j , t }$  Safeguard on the denominator of</td></tr><tr><td colspan="2">&quot;Delta-bar-delta&quot;-style α adjustment (2c)</td></tr><tr><td> $\chi _ { \mathrm { a } }$   $( \chi _ { \mathrm { o } } , 1 ]$   $\chi _ { \mathrm { o } }$   $[ - 1 , \chi _ { \mathrm { a } } )$ </td><td>Alignment threshold for growth Opposition threshold forcing decrease</td></tr><tr><td> $\Psi _ { \mathrm { a } }$   $[ 0 , \Psi _ { \mathrm { o } } )$   $\left( \Psi _ { \mathrm { a } } , 1 \right]$ </td><td>Rotation tolerance for growth</td></tr><tr><td> $\Psi _ { \mathrm { o } }$   $( 0 , 1 )$ </td><td>Rotation threshold forcing decrease Decrease factor of</td></tr><tr><td> $\eta _ { - }$   $\eta _ { + }$   $> 1$ </td><td> $\gamma _ { j , t }$  Increase factor of</td></tr><tr><td> $\gamma _ { \mathrm { m i n } }$ </td><td> $\gamma _ { j , t }$ </td></tr><tr><td></td><td>Floor on the multiplier</td></tr><tr><td> $\gamma _ { \mathrm { m a x } }$ </td><td>Ceiling on the multiplier</td></tr><tr><td colspan="2"> $\geq 1$ </td></tr><tr><td>Update (2d)</td><td></td></tr></table>

the two flags are exclusive since $\chi _ { \mathrm { o } } < \chi _ { \mathrm { a } }$ and $\Psi _ { \mathrm { a } } < \Psi _ { \mathrm { o } }$ . The update (2d) applies the scaled direction with base step size $\alpha > 0$ and decoupled weight decay $\lambda \geq 0 ,$ , as in AdamW [45].

f) Hyperparameters and cost: AURA adds the ten hyperparameters of Table I, plus the weight decay $\lambda ;$ the numerical values used in the test cases (Section IV) are given in Table II. Most have a clear interpretation and should not require heavy tuning: ε is a safeguard, $\gamma _ { \mathrm { m i n } }$ and $\gamma _ { \mathrm { m a x } }$ bound the multiplier, and $\eta _ { \pm }$ set its rate of change. The decay rate $\beta _ { \zeta }$ and the four thresholds can be more difficult to tune. Per parameter, AURA stores $d _ { j , t - 1 } , \tilde { \zeta } _ { j , t }$ , and $\gamma _ { j , t }$ , i.e., five real numbers against the three of Adam (a complex first moment and a real second moment), and adds a few elementwise operations per step; it needs no extra gradient evaluations. The measured overhead is reported in Section IV.

## III. RELATED WORK

This section relates AURA to the optimizers it builds on or resembles, with a focus on how each method measures the agreement between successive gradients or updates and on how it uses that measurement.

a) Adam and Muon: Adam [28] and Muon [41] are the base optimizers used in this work because they are established methods with known good performance, although other optimizers could be chosen. Adam normalizes the first moment of the gradient by the square root of its second moment, which Balles and Hennig [46] interpret as a damping of each step by the signal-to-noise ratio (SNR) of the corresponding gradient component. Muon applies momentum and orthogonalizes the update of each weight matrix by Newton–Schulz iterations; it carries no per-parameter variance estimate. In both cases, the SNR does not enter $\zeta _ { j , t }$ , which depends only on the directions $d _ { j , t }$ and $d _ { j , t - 1 } \colon$ AURA inherits whatever variance adaptation the base optimizer provides and adds none of its own. This could represent a limitation for noisy updates, since $\zeta _ { j , t }$ then measures the agreement between noisy directions (see also Section IV-D).

b) Step-size adaptation by sign information: The idea of adapting the step size from the agreement of successive derivatives dates back to the rules of Kesten, Saridis, and Barto and Sutton [47]–[49], and was systematized by Jacobs [43] in the delta-bar-delta rule, which increases the step size of a weight while consecutive derivatives share their sign and decreases it when they differ. AURA rests on the same principle, with the sign test replaced by a test on the complex quantity $\widehat { \zeta } _ { j , t }$

c) RPROP: The method proposed by [44], [50] assigns each weight its own update value, which is increased by a factor $\eta _ { + } ~ > ~ 1$ while the partial derivative keeps its sign and decreased by a factor $\eta _ { - } ~ < ~ 1$ when the sign changes; the weight then moves by this value against the sign of the derivative, irrespective of its magnitude. As in AURA, the agreement of successive derivatives sets the size of the next step. Complex RPROP [51], [52] applies the same rule to the real and imaginary parts separately, and therefore measures no angle in the complex plane.

d) CoRe: The CoRe optimizer [11], [12] combines an Adam direction $u _ { j , t } \in \mathbb { R }$ , the counterpart of $d _ { j , t }$ in (2a), with an RPROP-style per-weight step size $s _ { j , t } \in \mathbb { R }$ , as AURA does, and differs from it in three respects. First, it is defined for real-valued networks: its step-size rule depends only on the sign of the product of two consecutive first moments, the realdomain counterpart of the sign of $\chi _ { j , t }$ . Second, the rule is gated by a binary plasticity factor $P _ { j , t } \in \{ 0 , 1 \}$ , which freezes the weights with a high importance score. The update reads

$$
\begin{array} { r } { w _ { j , t } = w _ { j , t - 1 } - \underbrace { u _ { j , t } { P } _ { j , t } { s } _ { j , t } } _ { \mathrm { u p d a t e } } - \underbrace { d _ { \chi } \left| u _ { j , t } \right| { P } _ { j , t } { s } _ { j , t } w _ { j , t - 1 } } _ { \mathrm { r e g u l a r i z a t i o n } } , } \end{array}\tag{3}
$$

where $d _ { \chi } \in \mathbb { R } _ { \geq 0 }$ is a hyperparameter. Since $P _ { j , t }$ <sub>t</sub> also multiplies the update, a weight can remain unchanged for a whole step even when $u _ { j , t }$ is nonzero. AURA has no such switch: $\gamma _ { j , t }$ is bounded below by $\gamma _ { \mathrm { m i n } } > 0 ,$ , so the step in (2d) is never suppressed while $d _ { j , t }$ is nonzero. Third, the regularization term in (3) is a weight decay proportional to the magnitude $| u _ { j , t } | P _ { j , t } s _ { j , t }$ of the update, with a group-specific coefficient $d _ { \chi } ,$ , so that stable weights are decayed the least. Overfitting control is not a target of AURA, which uses the decoupled weight decay of Loshchilov and Hutter [45] and is compatible with other strategies.

e) AdaRem: AdaRem [53] multiplies the step size of each parameter by $a _ { t , i } ~ = ~ 1 + \lambda _ { t } b _ { t , i } ,$ where $\begin{array} { r l } { b _ { t , i } } & { { } = } \end{array}$ $g _ { t , i } m _ { t , i } / ( | g _ { t , i } |$ max $| m _ { t } | ~ + ~ \epsilon )$ and m<sub>t</sub> is the exponential moving average of the gradient. Up to $\epsilon , b _ { t , i } =$ $\mathrm { s i g n } ( g _ { t , i } m _ { t , i } ) | m _ { t , i } | /$ max $\lvert m _ { t } \rvert :$ for two scalars, Salton’s cosine reduces to the sign alone, which AdaRem weights by the relative size of the momentum. As in AURA, the step increases when the current gradient agrees with the past direction and decreases when it opposes it.

f) TAM: The algorithm TAM [54] computes the cosine similarity (Salton’s measure, Section $\mathbf { A } )$ between the previous momentum direction and the current gradient, an ingredient close to $\chi _ { j , t }$ ; this quantity, however, modifies the momentum rather than the step size.

g) diffGrad: diffGrad [4] multiplies the Adam direction by a “friction coefficient” $\xi _ { j , t } = \left( 1 + e ^ { - | g _ { j , t } - g _ { j , t - 1 } | } \right) ^ { - 1 } \in \left[ { \textstyle { \frac { 1 } { 2 } } } , 1 \right)$ so that “large changes in the gradient incur less friction, whereas small changes in the gradient incur more friction” [4]. Its behavior is therefore, to some extent, opposite to that of AURA.

h) AngularGrad: AngularGrad [3] rescales the Adam direction by an angle-like measure of consecutive gradients, the same kind of quantity that drives AURA. The multiplier is $\begin{array} { r l r } { \phi _ { j , t } ^ { \mathrm { A n g } } } & { { } = } & { \frac { \mathrm { i } } { 2 } + \frac { 1 } { 2 } \mathrm { \hat { t a n h } } \big | f ( A _ { j , \mathrm { m i n } } ) \big | } \end{array}$ , where $A _ { j , t } \ =$ arctan $\left| ( g _ { j , t } - g _ { j , t - 1 } ) / ( 1 + g _ { j , t } g _ { j , t - 1 } ) \right|$ is the angle between two consecutive scalar gradients interpreted as slopes, $A _ { j , \operatorname* { m i n } } =$ min $\{ A _ { j , t - 1 } , A _ { j , t } \}$ , and $f$ is either the cosine or the tangent. Three differences from AURA stand out. First, AngularGrad is designed and validated on real-valued networks only. Second, $A _ { j , t }$ is an unoriented slope angle, rather than the oriented phase difference of complex updates. Third, $A _ { j , t }$ hardly separates concordant from opposite gradients, because it compares slopes and a sign change of a small gradient is a small angle: with $f = \cos .$ , the pair $g _ { j , t - 1 } = - 0 . 0 1 , g _ { j , t } = 0 . 0 1$ gives $A _ { j , t } = 0 . 0 2$ , two equal gradients give $A _ { j , t } = 0$ , and both lead to $\phi _ { j , t } ^ { \mathrm { A n g } } \approx 0 . 8 8 1$

i) Hypergradient descent: Hypergradient descent [6] differentiates the loss with respect to the step size. Although conceptually distant from AURA, its multiplicative variant shares an important ingredient. As reported by Baydin et al. [6], the multiplicative rule is

$$
\alpha _ { t } = \alpha _ { t - 1 } \left( 1 - \beta ^ { \prime } \cos \phi _ { t } \right) ,\tag{4}
$$

where

$$
\cos \phi _ { t } = \frac { \widetilde { \nabla } \mathcal { L } ( w _ { t - 1 } ) ^ { \top } \nabla _ { \alpha } u ( \mathcal { W } _ { t - 2 } , \alpha _ { t - 1 } ) } { \left\| \widetilde { \nabla } \mathcal { L } ( w _ { t - 1 } ) \right\| \left\| \nabla _ { \alpha } u ( \mathcal { W } _ { t - 2 } , \alpha _ { t - 1 } ) \right\| } .\tag{5}
$$

Here $\beta ^ { \prime } \in \mathbb { R }$ is a hyperparameter, $\mathcal { L } \colon \mathbb { R } ^ { N }  \mathbb { R }$ is the loss, $\tilde { \nabla } \mathcal { L }$ its stochastic estimator, and u the update rule of the base method, $w _ { t } = u ( \mathscr { W } _ { t - 1 } , \alpha ) \in \mathbb { R } ^ { N }$ with ${ \mathcal W } _ { t } = \{ { w } _ { i } \} _ { i = 0 } ^ { t }$ . The step size is thus updated according to the angle between two vectors, as in $\operatorname { A U R A } ;$ the vectors, however, differ in meaning. Since $\nabla _ { \alpha } u ( \mathcal { W } _ { t - 2 } , \alpha _ { t - 1 } )$ is the previous update direction, hypergradient descent correlates the current gradient with the previous update, whereas AURA correlates two consecutive update directions.

j) Phase of the complex step size: FGACGD [35] also relies on an angle, but of a different kind: the phase $\phi _ { t } \in$ $[ - \pi / 4 , \pi / 4 ]$ of a single complex step size $| \eta _ { t } | e ^ { \mathrm { i } \phi _ { t } }$ shared by all trainable parameters. The phase is selected by comparing the gradient norms at three trial points, reached with step sizes $\rho e ^ { \mathrm { i 0 } }$ and $\rho e ^ { \pm \mathrm { i } \pi / 4 }$ , and steers the update toward flatter regions; no angle between consecutive directions is measured.

k) Summary: The methods above measure the agreement of successive gradients or updates in two ways. Sign-based rules [11], [43], [44] detect only whether the direction is reversed, and angle-based rules [3], [6], [54] detect an unoriented angle computed from real-valued gradients. AURA operates per parameter in the complex plane and measures how much consecutive directions differ – in length and in angle – and in which sense they rotate. This combination is the novelty claimed here; its ingredients are established: per-parameter multiplicative step control [44], angle-sensitive modulation [3], [6], and the combination of Adam with RPROP [11].

## IV. CASE STUDIES

This section compares AURA with the baseline optimizers on four test cases, selected to span different common learning tasks, architectures, and training regimes rather than to reach the best possible performance on any of them. The first three approximate a non-holomorphic (Section IV-C), a holomorphic (Section IV-D), and a multivariate (Section IV-E) complex function with fully connected networks; they are inexpensive and serve for sensitivity studies on the step size, the moment decay rates, the mini-batch size, and the floating-point precision. The last one tests whether the conclusions carry over to a physics-informed loss with full-batch training (Section IV-F). Since the object of study is the optimizer and not the task, reference architectures and standard setups are used throughout.

The protocol follows, as far as practicable, the recommendations of [13], [14], [55] on the fair comparison of optimizers. Within each case, all methods start from the same initialization, are trained on the same data, and are run for several random seeds; they are compared through the training and test loss curves and the three scalar metrics of Section IV-A: the best training loss reached, the area under the logarithmic learning curve, and the wall-clock time per step relative to plain stochastic gradient descent. Since the hyperparameter search space can by itself determine the ranking of optimizers [15], the hyperparameters of AURA are fixed, although their number would allow per-case tuning; this choice penalizes AURA, if anything, and tests at the same time whether the method requires such tuning. Section IV-F complements this with a comparison in which every optimizer is tuned under limited budget.

Four cases cannot cover every training scenario. The aim is to show that AURA is advantageous on a reasonably broad set of problems under a fair protocol, to quantify its overhead, and to document a failure mode (Section IV-D); extending the comparison to further tasks and larger models is left to future work.

All the test cases were run on a single A100 40GB graphics processing unit (GPU).

## A. Metrics for comparison

In each test case, every method m of the compared set is trained from the same initialization and on the same data, independently for $N _ { \mathrm { s e e d } }$ random seeds $s = 1 , \ldots , N _ { \mathrm { s e e d } } ;$ the value of $N _ { \mathrm { s e e d } }$ is given in the settings table of each case. Each run records the training loss $\mathcal { L } _ { \mathrm { t r a i n } } ^ { ( m , s ) } ( t )$ at every optimizer step t, namely the loss of the current mini-batch, or of the whole training set in the full-batch case, and the test-set loss $\mathcal { L } _ { \mathrm { t e s t } } ^ { ( m , s ) } ( t )$ at regular intervals. Methods are compared through the loss curves and through three scalar metrics, both summarized over seeds by the median and the 25th and 75th percentiles $Q _ { 2 5 }$ and $Q _ { 7 5 } ,$ , computed by linear interpolation; the tables report a metric with median x as $x _ { - ( x - Q _ { 2 5 } ) } ^ { + ( Q _ { 7 5 } - x ) }$ . Steps at which the loss is not finite, which occur only in diverged runs, are excluded from the scalar metrics but highlighted in the results summary tables. The three metrics are listed below.

Min training loss. The best training loss reached within a run, ${ \mathcal { L } } _ { \mathrm { m i n } } ^ { ( m , \tilde { s } ) } = \operatorname* { m i n } _ { t } { \mathcal { L } } _ { \mathrm { t r a i n } } ^ { ( m , s ) } ( t )$ , summarized over seeds by its median and percentiles,

$$
\mathcal { L } _ { \mathrm { m i n } } ^ { ( m ) } = \operatorname * { m e d i a n } _ { s = 1 , \ldots , N _ { \mathrm { s e e d } } } \mathcal { L } _ { \mathrm { m i n } } ^ { ( m , s ) } .\tag{6}
$$

It summarizes a run by a single number that does not depend on the choice of a stopping step. It is used to assess how effectively the optimizer minimizes the loss; the median reduces the influence of unusually favorable or unfavorable runs.

Area under the log learning curve. To account for the whole trajectory rather than for its best point only, the training-loss curve is averaged on a base-10 logarithmic scale,

$$
A ^ { ( m , s ) } = \frac { 1 } { T + 1 } \sum _ { t = 0 } ^ { T } \Bigl ( C + \log _ { 1 0 } \mathcal { L } _ { \mathrm { t r a i n } } ^ { ( m , s ) } ( t ) \Bigr ) ,\tag{C = 10,}
$$

(7)

where T is the final optimizer step, common to all methods within a case. $A ^ { ( m , s ) }$ is the normalized area between the log learning curve and a floor at loss $1 0 ^ { - C }$ , placed below the smallest loss reached in any case so that the area is non-negative; equivalently, it is $C$ plus the base-10 logarithm of the geometric-mean training loss, and the constant C affects no comparison. Since every decade of loss reduction has the same weight, $A ^ { ( m , s ) }$ rewards methods that reduce the loss early and keep it low, and is not dominated by the large losses of the initial transient. This metric is used as it provides a quantitative assessment of the whole training curve. A complementary qualitative assessment is provided by the training-curve plots.

Training time. The wall-clock time $\tau ^ { ( m , s ) }$ of the training loop, excluding the diagnostic instrumentation, such as the routines that store the data from which the figures are generated, and the just-in-time compilation of the first step. Since a time in seconds is tied to the GPU it was measured on, following [13], we use the ratio

$$
\tilde { \tau } ^ { ( m , s ) } = \tau ^ { ( m , s ) } / \tau ^ { ( \mathrm { S G D } ) } ,\tag{8}
$$

summarized over seeds by its median $\tilde { \tau } ^ { ( m ) }$ and percentiles, where $\tau ^ { \mathrm { ( S G D ) } }$ is the median time of the plain stochastic gradient descent update, $w _ { j , t + 1 } ~ = ~ w _ { j , t } - \alpha g _ { j , t }$ , over $N _ { \mathrm { S G D } } = 5$ seeds, run with the same architecture, precision, step budget, and GPU. As SGD is the cheapest first-order step, $\tilde { \tau } ^ { ( m ) }$ is the (almost) hardware-independent factor by which method m is slower than it. A separate SGD baseline is measured for each setting whose per-step cost differs, i.e., for each architecture and for complex64 versus complex128. SGD serves only as a unit of time and is not part of . We adopt this metric because the actual training time is an important practical feature of the optimizer reflecting its computational efficiency.

## B. Optimizer settings

The baseline optimizers are chosen as follows. RPROP [44] is included because, as in AURA, the agreement of successive derivatives sets the size of the next step. Adam [28] and NadamW, that is, Nadam [31] with decoupled weight decay [45], are included because they are widely used and perform well for training complex-valued networks [17]; Adam is, in addition, the base optimizer of Adam-AURA, so that the comparison between the two isolates the effect of AURA. CvAMSGrad, the split-complex extension of AMSGrad [29] proposed in [17], is included because it attains a lower steadystate error than SGD on the two complex-valued benchmark problems of [17], at a moderate increase in computational cost. Muon [41] is included because it is designed specifically for the weight matrices of the hidden layers of neural networks, and it is the base optimizer of Muon-AURA.

The default hyperparameters for each optimizer are listed in Table II. Where applicable, these are adjusted as described in the relevant test case. The ten hyperparameters specific to AURA are held fixed across all test cases, except in Section $\mathrm { I V \mathrm { - } F } ,$ where its principal ones are tuned together with those of the baselines. The resulting comparison is pragmatic and fair: if anything, this constraint disadvantages AURA, which nonetheless outperforms the baselines in many cases.

C. TEST 1: univariate scalar complex non-holomorphic function

a) Rationale: This test provides a lightweight analytical benchmark for sensitivity and ablation studies. The goal is to approximate a non-holomorphic function, complementing the holomorphic case considered in a separate test. We empirically investigate the sensitivity of the optimizers to their main hyperparameters by comparing their performance across two fully connected neural network architectures, different step sizes, and different exponential moving average (EMA) decay rates.

b) Learning task: The domain is the square $\Omega _ { \zeta } = \{ \zeta \in$ $\mathbb { C } : | \mathrm { R e } \zeta | \leq 1 , \ | \mathrm { I m } \zeta | \leq 1 \}$ . The target function is

$$
\begin{array} { r l } & { f ( \zeta ) = \exp \bigl ( ( 0 . 3 0 - 0 . 2 0 \mathrm { i } ) \zeta \overline { { \zeta } } \bigr ) + 0 . 2 5 \sin ( \zeta ) \cos ( \overline { { \zeta } } ) } \\ & { \qquad + 0 . 1 0 \zeta ^ { 2 } \overline { { \zeta } } + 0 . 0 8 \zeta \overline { { \zeta } } ^ { 2 } - 0 . 0 5 \mathrm { i } ( \zeta \overline { { \zeta } } ) ^ { 2 } . } \end{array}\tag{9}
$$

TABLE II  
DEFAULT HYPERPARAMETERS USED IN THE BENCHMARK CASES IN SECTION IV.  
Optimizer Hyperparameters   
RPROP $\eta _ { - } = 0 . 5 , \eta _ { + } = 1 . 2 , \Delta \in [ 1 \times 1 0 ^ { - 6 } , 5 0 ] ; \Delta _ { 0 } = \alpha$   
Adam $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \varepsilon _ { \mathrm { A } } = 1 \times 1 0 ^ { - 8 }$   
Adam (variable LR) Same β<sub>1</sub>, β<sub>2</sub>, ε<sub>A</sub> as Adam; learning rate α for the first 50% of updates, then $\alpha / 1 0$   
NadamW $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \varepsilon _ { \mathrm { A } } = 1 \times 1 0 ^ { - 8 }$ , weight decay: $1 \times 1 0 ^ { - 4 }$   
CvAMSGrad $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \varepsilon _ { \mathrm { A } } = 1 \times 1 0 ^ { - 8 } .$ ; bias correction included   
Muon $\beta = 0 . 9 5 ,$ Newton–Schulz steps: 5, Nesterov: yes, weight decay: 0; matrix step 10 α; bias fallback AdamW at   
$\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \varepsilon _ { \mathrm { A } } = \mathrm { i } \stackrel { \cdot } { \times } 1 0 ^ { - 8 }$   
Adam-AURA $\beta _ { \zeta } = 0 . 9 5 , ~ \varepsilon _ { \mathrm { E } } = 1 \times 1 0 ^ { - 6 } , ~ \chi _ { \mathrm { a } } = 0 . 7 , ~ \chi _ { \mathrm { o } } = 0 . 4 , ~ \Psi _ { \mathrm { a } } = 0 . 0 1 5 , ~ \Psi _ { \mathrm { o } } = 0 . 3 , ~ \eta _ { - } = 0 . 9 9 , ~ \eta _ { + } = 1 . 0 1 , ~ \gamma \in \mathbb { R } ^ { 3 } .$   
$[ 1 ^ { \cdot } \times 1 0 ^ { - 3 } , 1 0 0 0 ] , \lambda = 1 \times 1 0 ^ { - }$   
Muon-AURA β = 0.95, Newton–Schulz steps: 5, Nesterov: yes, matrix step $1 0 \alpha ; \beta _ { \zeta } = 0 . 9 5 , \varepsilon _ { \mathrm { E } } = 1 \times 1 0 ^ { - 6 } , \chi _ { \mathrm { a } } = 0 . 7 5 , \chi _ { \mathrm { o } } = 0 . 4 ,$   
$\Psi _ { \mathrm { a } } = 0 . 0 1 , \Psi _ { \mathrm { o } } = 0 . 2 , \eta _ { \mathrm { - } } \stackrel {  } { = } 0 . 9 9 , \eta _ { \mathrm { + } } = \dot { 1 } . 0 1 , \gamma \in [ 1 \stackrel { \cdot } { \times } 1 0 ^ { - 3 } , \dot { 1 } 0 0 0 ] , \lambda = 1 \times 1 0 ^ { - 4 } ;$ ; bias direction Adam at   
$\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \varepsilon _ { \mathrm { A } } = 1 \times 1 0 ^ { - 8 }$

The training and test sets consist of mutually independent samples drawn uniformly from $\Omega _ { \zeta } \colon$

$$
\begin{array} { r l r } { \mathcal { Z } _ { \mathrm { t r a i n } } = \{ \zeta _ { i } \} _ { i = 1 } ^ { n _ { \mathrm { t r a i n } } } , } & { \zeta _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } \mathrm { U n i f } ( \Omega _ { \zeta } ) , } & { \quad ( 1 0 ) } \\ { \mathcal { Z } _ { \mathrm { t e s t } } = \{ \zeta _ { i } ^ { \prime } \} _ { i = 1 } ^ { n _ { \mathrm { t e s t } } } , } & { \zeta _ { i } ^ { \prime } \stackrel { \mathrm { i . i . d . } } { \sim } \mathrm { U n i f } ( \Omega _ { \zeta } ) , } & { \mathcal { Z } _ { \mathrm { t e s t } } \cap \mathcal { Z } _ { \mathrm { t r a i n } } = \emptyset , } \end{array}\tag{11}
$$

with $n _ { \mathrm { t e s t } } = \mathrm { r o u n d } ( 0 . 2 n _ { \mathrm { t r a i n } } )$ . The real and imaginary parts of each sample are drawn independently from Unif([ 1, 1]). Denoting the neural network by $f _ { \mathcal { W } }$ and a nonempty mini-batch by $B \subseteq \mathcal { Z } _ { \mathrm { t r a i n } }$ , training minimizes the mean squared error

$$
\mathcal { L } _ { B } ( \mathcal { W } ) = \frac { 1 } { | \mathcal { B } | } \sum _ { \zeta \in B } \bigl | f _ { \mathcal { W } } ( \zeta ) - f ( \zeta ) \bigr | ^ { 2 } .\tag{12}
$$

c) Network architectures: The networks are compositions of affine maps and componentwise nonlinear activations:

$$
f _ { \mathcal { W } } = A _ { L } \circ \varsigma \circ A _ { L - 1 } \circ \cdot \cdot \cdot \circ \varsigma \circ A _ { 1 } ,
$$

with $A _ { \ell } ( z ) = W _ { \ell } z + b _ { \ell } , \ell = 1 , \dots , L ,$ where $W _ { \ell } \in \mathbb { C } ^ { n _ { \ell } \times n _ { \ell - 1 } }$ $b _ { \ell } \in \mathbb { C } ^ { n _ { \ell } } , n _ { 0 } = n _ { L } = 1$ , and $\mathcal { W } = \{ W _ { \ell } , b _ { \ell } \} _ { \ell = 1 } ^ { L }$ . The activation

$$
\begin{array} { r } { \varsigma ( \zeta ) = \mathrm { S i L U } ( \mathrm { R e } \zeta ) + \mathrm { i } \mathrm { \ S i L U } ( \mathrm { I m } \zeta ) , } \end{array}
$$

is applied componentwise in the hidden layers. This activation is non-holomorphic, so the networks are generally nonholomorphic.

The complete benchmark configuration is given in Table III.

TABLE III  
CONFIGURATION OF THE NON-HOLOMORPHIC OPTIMIZER BENCHMARK.  
```csv
Setting Value
Primary architecture (1, 32, 32, 32, 32, 1), 3,265 trainable param
eters
Secondary (1, 128, 128, 128, 128, 128, 1), 66,433 train
architecture able parameters
Hidden activation $\mathrm { S i L U } ( \mathrm { R e } \zeta ) + \mathrm { i } \mathrm { S i L U } ( \mathrm { I m } \zeta )$
Training / test set 2,500 / 500 points (test: 20%)
Mini-batch size 256 points
Updates / seeds $1 2 , 0 0 0 / \left\{ 0 , \ldots , 4 \right\}$
Arithmetic complex64 (32-bit real components)
Initialization Glorot/Xavier normal rule [56], applied sep
arately to the real and imaginary parts; zero
biases
Base learning rate α $5 \times 1 0 ^ { - 4 }$
(primary)
Secondary- $5 \times 1 0 ^ { - 5 }$
architecture learning
rate
Optimizer default values; see Table II
hyperparameters
```

d) Results – α sensitivity: Table IV reports the three metrics for the step sizes $\alpha / 1 0 , \alpha .$ , and 10α, and Figure 1 shows the corresponding training- and test-loss curves for both architectures. Over this 100-fold range, $\mathcal { L } _ { \operatorname* { m i n } } ^ { ( m ) }$ varies by more than two orders of magnitude for Adam and its variants and by more than three for Muon at the primary architecture and by more than one at the secondary, but by less than one for Adam-AURA, and it remains of order $1 0 ^ { - 7 }$ or below for Muon-AURA. RPROP is similarly insensitive to $\alpha .$ , but its minimum loss is 6 to 50 times that of Adam-AURA. Muon-AURA attains the lowest $\mathcal { L } _ { \operatorname* { m i n } } ^ { ( m ) }$ and $A ^ { ( m ) }$ in all six settings; since $A ^ { ( m ) }$ is C plus the base-10 logarithm of the geometric-mean training loss (7), its margin of about 2 to 3 over the best baseline corresponds to a loss roughly two to three orders of magnitude lower along the whole trajectory, not only at its best point. This improvement comes at a small cost: the time ratio (8) of Adam-AURA exceeds that of Adam by at most 0.1, within the spread over seeds, and that of Muon-AURA is even slightly below that of Muon, so the higher ratio of Muon-AURA, up to

TABLE IV  
STEP-SIZE SENSITIVITY FOR THE NON-HOLOMORPHIC BENCHMARK.
<table><tr><td></td><td></td><td colspan="3"> $\mathcal { L } _ { \mathrm { m i n } } ^ { ( m ) } ~ ( \times 1 0 ^ { - 6 } )$ </td><td colspan="3"> $A ^ { ( m ) }$ </td><td>Training time (×SGD)</td></tr><tr><td>Arch.</td><td>Optimizer</td><td> $\alpha / 1 0$ </td><td>α</td><td>10α</td><td> $\alpha / 1 0$ </td><td>α</td><td>10α</td><td>α</td></tr><tr><td rowspan="8">Primary</td><td>RPROP</td><td> $3 4 _ { - 8 } ^ { + 1 9 }$ </td><td> $2 0 3 _ { - 9 3 } ^ { + 6 }$ </td><td> $4 1 _ { - 1 , i } ^ { + 6 }$ </td><td> $5 . 9 _ { - 0 . 1 } ^ { + 0 . 3 }$ </td><td> $6 . 6 _ { - 0 . 0 } ^ { + 0 . 0 1 }$ </td><td> $6 . 1 _ { - 0 . 3 } ^ { + 0 . 1 }$ </td><td> $1 . 6 7 _ { - 0 . 0 9 } ^ { + 0 . 0 0 0 2 }$ </td></tr><tr><td>Adam</td><td> $1 6 7 _ { - 5 2 } ^ { + 9 2 }$ </td><td> $4 . 6 _ { - 0 . 1 } ^ { + 0 . 5 }$ </td><td> $1 . 3 _ { - 0 . 2 } ^ { + 0 . 1 }$ </td><td> $7 . 3 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td>-0.2  $5 . 8 _ { - 0 . 0 3 } ^ { + 0 . 1 }$ </td><td> $5 . 4 _ { - 0 . 0 4 } ^ { + 0 . 0 1 }$ </td><td> $1 . 5 9 _ { - 0 . 0 4 } ^ { + 0 . 0 \bar { 0 } 5 }$ </td></tr><tr><td>Adam (variable LR)</td><td> $6 1 4 _ { - 1 1 6 } ^ { + 1 0 6 }$ </td><td> $7 . 6 _ { - 0 . 4 } ^ { + 2 . 5 }$ </td><td> $1 . 5 _ { - 0 . 0 3 } ^ { + 0 . 2 }$ </td><td> $7 . 4 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $5 . 8 _ { - 0 . 0 1 } ^ { + 0 . 1 }$ </td><td> $5 . 1 _ { - 0 . 0 0 1 } ^ { + 0 . 0 0 2 }$ </td><td> $\mathbf { 1 . 5 8 0 } _ { - 0 . 0 1 2 } ^ { + 0 . 0 0 6 }$ </td></tr><tr><td>NadamW</td><td> $1 5 1 ^ { + 9 9 }$  49</td><td> $5 . 6 _ { - } ^ { + 0 . 7 }$ </td><td> $1 . 1 ^ { + 0 . 1 } \dot { }$ </td><td> $7 . 3 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $5 . 9 _ { - 0 . 0 5 } ^ { + 0 . 1 }$ </td><td> $5 . 8 ^ { + 0 . 0 \dot { 2 } }$ </td><td> $1 . 5 9 _ { - 0 . 0 1 } ^ { + 0 . { \dot { 0 4 } } }$ </td></tr><tr><td>CvAMSGrad</td><td> $8 9 3 _ { - 9 4 } ^ { + 1 1 9 }$ </td><td> $1 6 ^ { + 0 . 3 }$ </td><td> $\stackrel { \bullet \cdot \mathbf { \scriptscriptstyle 4 } } { \mathbf { \scriptscriptstyle 4 } } \stackrel { - 0 . 0 2 } { + 0 . 3 }$   $2 . 2 _ { - 0 . 0 2 } ^ { \phantom { + } }$ </td><td> $7 . 4 _ { \scriptscriptstyle - 0 . 0 0 2 } ^ { \scriptscriptstyle + 0 . 0 5 }$ </td><td> $6 . 1 ^ { + 0 . 0 5 }$   $6 . 1 _ { - 0 . 1 } ^ { + 0 . 0 5 }$ </td><td> $\yen 123$   $5 . 1 _ { , - 0 . 0 4 } ^ { \tau \mathrm { v . v . } }$ </td><td> $1 . 6 _ { - 0 \mathrm { ~ 2 ~ } } ^ { + 0 . 1 }$ </td></tr><tr><td>Muon</td><td> $0 . 2 _ { - \mathrm { ~ n ~ o ~ n ~ } } ^ { + 0 . 4 }$ </td><td> $\mathbf { ^ { x 0 } - 4 }$   $3 2 ^ { + 3 }$ </td><td> $2 8 3 _ { - 4 4 } ^ { + 6 0 }$ </td><td> $5 . 3 ^ { \overline { { + } } \overline { { 0 . 0 2 } } ^ { \overline { { \angle } } } }$   $5 . 3 _ { - 0 . 0 0 3 } ^ { + 0 . 0 2 }$ </td><td></td><td> $7 . 0 _ { - 0 . 0 1 } ^ { + 0 . 0 0 1 }$ </td><td> $1 . 8 8 _ { - 0 . 0 2 } ^ { + 0 . 0 0 0 2 }$ </td></tr><tr><td>Adam-AURA</td><td> $3 . 0 _ { - 0 . 4 } ^ { + 2 . 1 }$ </td><td> $3 . 9 _ { - 1 . 9 } ^ { + 0 . 4 }$ </td><td> $1 . 7 _ { - 0 . 0 4 } ^ { + 0 . 8 ^ { \circ } }$ </td><td> $5 . 3 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $\begin{array} { c } { 6 . 0 _ { - 0 . 0 0 2 } ^ { + 0 . 0 1 } } \\ { 5 . 1 _ { - 0 . 1 } ^ { + 0 . 1 } } \end{array}$ </td><td> $4 . 8 _ { - 0 . 1 } ^ { + 0 . 0 3 }$ </td><td> $1 . 6 4 _ { - 0 . 0 1 } ^ { + 0 . 1 1 }$ </td></tr><tr><td>Muon-AURA</td><td> $\mathbf { 0 . 1 } _ { - \mathbf { n . n 0 0 2 } } ^ { + \mathbf { 0 . 0 0 3 } }$ </td><td> $\mathbf { 0 . 0 4 } _ { - 0 . 0 0 5 } ^ { + 0 . { \overset { . } { 0 } } 0 4 }$ </td><td> $\mathbf { 0 . 0 1 } _ { - \mathbf { 0 . 0 0 0 1 } } ^ { + \mathbf { 0 . 0 0 1 } }$ </td><td> $\mathbf { 3 . 4 } _ { - 0 . 0 0 3 } ^ { + 0 . 1 }$ </td><td> $\mathbf { 3 . 1 } _ { - 0 . 1 } ^ { + 0 . \bar { 0 } 4 }$ </td><td> $\mathbf { 2 . 7 _ { - 0 . 0 1 } ^ { + 0 . 0 0 3 } }$ </td><td> $1 . 8 0 _ { - 0 . 0 3 } ^ { + 0 . 0 9 }$ </td></tr><tr><td rowspan="8">Secondary</td><td>RPROP</td><td> $9 . 2 _ { - 0 . 2 } ^ { + 1 9 . 1 }$ </td><td> $1 6 _ { - 7 } ^ { + 4 }$ </td><td> $1 7 _ { - 1 } ^ { + 1 }$ </td><td> $5 . 4 _ { - \Omega \ \Omega , 4 } ^ { + 0 . 4 }$ </td><td> $5 . 7 _ { - 0 . 1 } ^ { + 0 . 0 3 }$ </td><td> $5 . 6 _ { - 0 . 0 5 } ^ { + 0 . 1 }$ </td><td> $1 . 7 0 _ { - \sim } ^ { + 0 . 0 2 }$ </td></tr><tr><td>Adam</td><td> $1 5 0 5 _ { - 1 0 3 } ^ { + 4 }$ </td><td> $2 8 _ { - 8 } ^ { + \overline { { 4 } } }$ </td><td> ${ 3 . 4 } _ { - 0 . 1 } ^ { + 0 . 2 }$ </td><td> $8 . 0 _ { - 0 . 0 4 } ^ { + 0 . 0 1 }$ </td><td> $6 . 8 _ { - 0 . 0 5 } ^ { + 0 . { \bar { 0 } } 2 }$ </td><td> $5 . 9 _ { - 0 . 1 } ^ { + 0 . 0 1 }$ </td><td> $1 ~ 6 5 ^ { + 0 . 0 8 }$ </td></tr><tr><td>Adam (variable LR)</td><td> $1 7 8 6 _ { - 6 } ^ { + 2 1 }$ </td><td> $3 0 5 _ { - 2 3 } ^ { + \mathrm { { 1 2 } } }$ </td><td> $4 . 4 _ { - 0 . 6 } ^ { + 0 . 3 }$ </td><td> $8 . 0 _ { - 0 . 0 4 } ^ { + 0 . 0 1 }$ </td><td> $7 . 0 _ { - 0 . 0 2 } ^ { + 0 . 0 2 }$ </td><td>0.01  $5 . 7 _ { - 0 . 2 } ^ { + 0 . 0 1 }$ </td><td>-0.01  $\mathbf { 1 . 6 4 } _ { - 0 . 0 2 } ^ { + 0 . 0 1 }$ </td></tr><tr><td>NadamW</td><td> $1 4 9 4 _ { - 9 3 } ^ { + 9 }$ </td><td> $2 7 _ { - 5 } ^ { + 1 }$ </td><td> $3 . 7 _ { - 0 . 0 1 } ^ { + 0 . 8 }$ </td><td> $8 . 0 _ { - 0 . 0 4 } ^ { + 0 . 0 1 }$ </td><td> $6 . 8 _ { - 0 . 0 3 } ^ { + 0 . 0 2 }$ </td><td> $6 . 1 _ { - 0 . 0 2 } ^ { + 0 . 0 1 }$ </td><td> $1 . 6 4 _ { - 0 ~ 0 2 } ^ { + 0 . 0 1 }$ </td></tr><tr><td>CvAMSGrad</td><td> $1 7 8 5 _ { - 2 4 } ^ { + 1 6 }$ </td><td> $2 8 7 _ { \mathrm { ~ \bf ~ 1 ~ } A } ^ { + 1 8 }$ </td><td> $4 . 1 _ { - 0 . 7 } ^ { + 0 . 2 } $ </td><td>+0.04</td><td> $7 . 0 _ { \frac { - 0 . 0 1 } { \cdot } } ^ { + 0 . 0 4 }$ </td><td> $5 . 7 _ { \scriptscriptstyle - 0 . 1 . } ^ { \scriptscriptstyle + 0 . 0 3 }$ </td><td> $1 . 6 6 _ { - 0 . 0 2 } ^ { + 0 . 0 0 2 }$ </td></tr><tr><td>Muon</td><td> $5 0 _ { - 1 A } ^ { + 6 }$ </td><td> $3 . 5 _ { - 0 \mathrm { ~ n ~ 2 ~ } } ^ { + 0 . 3 }$ </td><td> $5 5 _ { - 0 . 3 } ^ { + 0 . 1 }$ </td><td> $\begin{array} { c } { { 7 . 9 _ { - 0 . 0 5 } ^ { + 0 . 0 1 } } } \\ { { 8 . 4 ^ { + 0 . 0 1 } } } \end{array}$ </td><td> $5 . 2 _ { - 0 . 0 1 } ^ { + 0 . 0 2 }$ </td><td> $6 . 1 _ { - 0 . 0 1 } ^ { + 0 . 0 1 }$ </td><td> $2 . 2 1 8 _ { - 0 . 0 0 0 3 } ^ { + 0 . 0 1 0 }$ </td></tr><tr><td>Adam-AURA</td><td> $1 . 5 _ { - 0 . 2 } ^ { + 0 . 1 }$ </td><td> $1 . 6 _ { - 0 . 4 } ^ { + 0 . 2 }$ </td><td> $0 . 6 _ { - 0 . 2 } ^ { + 0 . 2 }$ </td><td> $5 . 4 _ { - 0 . 1 } ^ { - 0 . 1 }$ </td><td> $4 . 9 _ { - 0 . 0 1 } ^ { + 0 . 2 }$ </td><td> $4 . 6 _ { - 0 . 0 1 } ^ { + 0 . 1 }$ </td><td> $1 . 6 8 _ { - 0 . 0 5 } ^ { + 0 . 0 5 }$ </td></tr><tr><td>Muon-AURA</td><td> $\mathbf { 0 . 0 2 } _ { - 0 . 0 0 1 } ^ { + 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 0 1 } _ { - 0 . 0 0 0 3 } ^ { + 0 . 0 \mathrm { \bar { 0 } } \mathbf { 0 } 4 }$ </td><td> $\mathbf { 0 . 0 1 _ { - 0 . 0 0 0 0 4 } ^ { + 0 . 0 0 0 4 } }$ </td><td> $\mathbf { 2 . 9 } _ { - 0 . 0 2 } ^ { + \bar { 0 . 0 3 } }$ </td><td> $\mathbf { 2 . 8 _ { - 0 . 0 1 } ^ { + 0 . 0 2 } }$ </td><td> $\mathbf { 2 . 6 _ { - 0 . 0 4 } ^ { + 0 . 0 1 } }$ </td><td> $2 . 1 4 7 _ { - 0 . 0 0 3 } ^ { + 0 . 0 1 5 }$ </td></tr></table>

2.1, stems from the Muon direction rather than from AURA.

![](images/9b1df39233b7e98834b6cd484eac0a5e50adbfcc7a23193cf4bd4712e9c8a3cd.jpg)  
Fig. 1. Training-set and test-set mean squared errors for the Non-holomorphic optimizer benchmark. The two architecture blocks show the network trained at the primary (top) and secondary (bottom) architecture (Table III). Within each block, the rows sweep the base learning rate α across α/10, α and 10α from top to bottom. Solid curves show the pointwise median over all 5 seeds; shaded regions span the interquartile range (25th–75th percentile) over seeds, whose edges are also drawn as dotted curves.

TABLE V  
$\beta _ { 1 } / \beta _ { 2 }$ SENSITIVITY FOR THE NON-HOLOMORPHIC BENCHMARK.
<table><tr><td> $\beta _ { 1 }$ </td><td> $\beta _ { 2 }$ </td><td>RPPROP</td><td>Adam</td><td>Ada ab val)</td><td>NamW</td><td></td><td>CSrad</td><td>Muon</td><td>Ad-ARA</td><td>MU-AARA</td></tr><tr><td></td><td colspan="8"> $\mathcal { L } _ { \mathrm { m i n } } ^ { ( m ) } ~ ( \times 1 0 ^ { - 6 } )$ </td><td></td><td></td></tr><tr><td>0.85</td><td>0.99</td><td> $1 1 0 _ { - 1 5 } ^ { + 4 9 }$ </td><td> $4 . 4 _ { - 0 . 4 } ^ { + 0 . 2 }$ </td><td> $5 . 5 _ { - 0 . 7 } ^ { + 0 . 4 }$ </td><td> $5 . 2 _ { - 0 . 2 } ^ { + 1 . 9 }$ </td><td> $5 5 _ { - 5 } ^ { + 3 7 }$ </td><td> $4 1 _ { - 4 } ^ { + 0 . 2 }$ </td><td> $5 . 8 _ { - 1 . 0 } ^ { + 1 . 1 }$ </td><td> $\mathbf { 0 . 0 4 } _ { - \mathbf { n . 0 0 } } ^ { + \mathbf { 0 . 0 1 } }$ </td></tr><tr><td>0.9</td><td>0.99</td><td> $1 1 0 _ { - 1 5 } ^ { + 5 0 }$ </td><td> $3 . 8 _ { - 0 . 4 } ^ { + 0 . 0 4 }$ </td><td> $4 . 9 _ { - 0 . 5 } ^ { + 0 . 0 3 }$ </td><td> $5 . 1 _ { - 0 . 8 } ^ { + 0 . 4 }$ </td><td> $5 8 _ { - 0 . 1 } ^ { + 4 5 }$ </td><td> $3 4 _ { - 2 } ^ { + 2 }$ </td><td> $2 . 0 _ { - 0 . 3 } ^ { + 0 . 6 }$ </td><td> $\mathbf { 0 . 0 4 } _ { - 0 . 0 0 3 } ^ { + 0 . 0 0 2 }$ </td></tr><tr><td>0.95</td><td>0.99</td><td> $1 1 0 _ { - 1 5 } ^ { + 5 0 }$ </td><td> $2 . 6 _ { - 0 . 0 3 } ^ { + 0 . 4 }$ </td><td> $4 . 1 _ { - 0 . 4 } ^ { + 0 . 1 }$ </td><td> $3 . 3 _ { - 0 . 4 } ^ { + 0 . 2 }$ </td><td> $1 5 1 _ { - 3 7 } ^ { + 2 9 }$ </td><td> $3 0 _ { - 0 . 1 } ^ { + 1 }$ </td><td> $0 . 5 _ { - 0 . 0 4 } ^ { + 0 . 8 }$ </td><td> $\mathbf { 0 . 0 3 _ { - 0 . 0 0 2 } ^ { + 0 . 0 0 3 } }$ </td></tr><tr><td>0.85</td><td>0.999</td><td> $1 1 0 _ { - 1 5 } ^ { + 5 0 }$ </td><td> $5 . 7 _ { - 0 . 4 } ^ { + 0 . 2 }$ </td><td> $8 . 0 _ { - 0 . 1 } ^ { + 1 . 4 }$ </td><td> $7 . 5 _ { - 0 . 4 } ^ { + 2 . 2 }$ </td><td> $1 5 _ { - 2 } ^ { + 2 }$ </td><td> $3 9 _ { - 3 } ^ { + 1 }$ </td><td> $5 . 3 _ { - 1 . 0 } ^ { + 2 . 2 }$ </td><td> $\mathbf { 0 . 0 4 } _ { - 0 . 0 0 3 } ^ { + 0 . 0 1 }$ </td></tr><tr><td>0.9</td><td>0.999</td><td> $1 2 8 _ { - 2 4 } ^ { + 4 1 }$ </td><td> $4 . 9 _ { - 0 . 2 } ^ { + 0 . 1 }$ </td><td> $7 . 6 _ { - 0 . 2 } ^ { + 1 . 3 }$ </td><td> $6 . 6 _ { - 0 . 6 } ^ { + 0 . 4 }$ </td><td> $1 6 _ { - 2 } ^ { + 2 }$ </td><td> $3 7 _ { - 3 } ^ { + 0 . 1 }$ </td><td> $1 . 9 _ { - 0 . 1 } ^ { + 1 . 1 }$ </td><td> $\mathbf { 0 . 0 4 } _ { - 0 . 0 0 1 } ^ { + 0 . 0 1 }$ </td></tr><tr><td>0.95</td><td>0.999</td><td> $1 1 0 _ { - 1 5 } ^ { + 5 0 }$ </td><td> $4 . 1 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $9 . 3 _ { - 1 . 1 } ^ { + 0 . 0 2 }$ </td><td> $4 . 4 _ { - 0 . 0 1 } ^ { + 0 . 3 }$ </td><td> $2 1 _ { - 2 } ^ { + 2 }$ </td><td> $2 7 _ { - 1 } ^ { + 2 }$ </td><td> $0 . 6 _ { - 0 . 1 } ^ { + 0 . 8 }$ </td><td> $\mathbf { 0 . 0 3 _ { - 0 . 0 0 0 2 } ^ { + 0 . 0 0 3 } }$ </td></tr><tr><td>0.85</td><td>0.9999</td><td> $1 2 8 _ { - 2 4 } ^ { + 4 1 }$ </td><td> $9 . 2 _ { - 0 . 3 } ^ { + 0 . 5 }$ </td><td> $3 5 _ { - 5 } ^ { + 1 }$ </td><td> $1 1 _ { - 0 . 5 } ^ { + 2 }$ </td><td> $6 . 5 _ { - 0 . 3 } ^ { + 1 . 0 }$ </td><td> $3 0 _ { - 7 } ^ { + 3 }$ </td><td> $1 0 _ { - 2 } ^ { + 0 . 3 }$ </td><td> $\mathbf { 0 . 0 4 _ { - 0 . 0 0 3 } ^ { + 0 . 0 1 } }$ </td></tr><tr><td>0.9</td><td>0.9999</td><td> $1 1 0 _ { - 1 5 } ^ { + 5 0 }$ </td><td> $8 . 7 _ { - 0 . 4 } ^ { + 0 . 5 }$ </td><td> $3 6 _ { - 5 } ^ { + 0 . 4 }$ </td><td> $9 . 3 _ { - 0 . 5 } ^ { + 0 . 6 }$ </td><td> $6 . 5 _ { - 0 . 2 } ^ { + 0 . 7 }$ </td><td> $2 0 _ { - 7 } ^ { + 3 }$ </td><td> $3 . 1 _ { - 0 . 1 } ^ { + 1 . 2 }$ </td><td> $\mathbf { 0 . 0 4 } _ { - 0 . 0 0 1 } ^ { + 0 . 0 1 }$ </td></tr><tr><td>0.95</td><td>0.9999</td><td> $1 1 0 _ { - 1 5 } ^ { + 5 0 }$ </td><td> $1 0 _ { - 0 . 1 } ^ { + 1 }$ </td><td> $4 0 _ { - 2 } ^ { + 6 }$ </td><td> $9 . 7 _ { - 0 . 3 } ^ { + 0 . 5 }$ </td><td> $7 . 4 _ { - 0 . 4 } ^ { + 0 . 5 }$ </td><td> $7 . 9 _ { - 0 . 9 } ^ { + 1 0 . 2 }$ </td><td> $1 . 2 _ { - 0 . 1 } ^ { + 0 . 8 }$ </td><td> $\mathbf { 0 . 0 3 } _ { - 0 . 0 0 1 } ^ { + 0 . 0 0 4 }$ </td></tr><tr><td colspan="10"> $A ^ { ( m ) }$ </td></tr><tr><td>0.85</td><td>0.99</td><td> $6 . 4 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $5 . 6 _ { - 0 . 0 5 } ^ { + 0 . 1 }$ </td><td> $5 . 5 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $5 . 9 _ { - 0 . 0 4 } ^ { + 0 . 1 }$ </td><td> $6 . 6 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $6 . 0 _ { - 0 . 0 1 } ^ { + 0 . 0 0 5 }$ </td><td> $5 . 2 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $\mathbf { 3 . 1 } _ { - 0 . 0 3 } ^ { + 0 . 0 4 }$ </td></tr><tr><td>0.9</td><td>0.99</td><td> $6 . 4 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $5 . 5 _ { - 0 . 0 5 } ^ { + 0 . 0 4 }$ </td><td> $5 . 4 _ { - 0 . 1 } ^ { + 0 . 0 4 }$ </td><td> $5 . 7 _ { - 0 . 0 4 } ^ { + 0 . 1 }$ </td><td> $6 . 7 _ { - 0 . 1 } ^ { + 0 . 0 5 }$ </td><td> $6 . 0 _ { - 0 . 0 1 } ^ { + 0 . 0 1 }$ </td><td> $4 . 9 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $\mathbf { 3 . 1 _ { - 0 . 0 4 } ^ { + 0 . 0 0 3 } }$ </td></tr><tr><td>0.95</td><td>0.99</td><td> $6 . 4 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $5 . 4 _ { - 0 . 1 } ^ { + 0 . 0 4 }$ </td><td> $5 . 4 _ { - 0 . 1 } ^ { + 0 . 0 3 }$ </td><td> $5 . 4 _ { - 0 . 0 5 } ^ { + 0 . 0 4 }$ </td><td> $6 . 8 _ { - 0 . 1 } ^ { + 0 . 0 2 }$ </td><td> $5 . 9 _ { - 0 . 0 1 } ^ { + 0 . 0 1 }$ </td><td> $4 . 4 _ { - 0 . 0 3 } ^ { + 0 . 2 }$ </td><td> $\mathbf { 3 . 0 _ { - 0 . 0 2 } ^ { + 0 . 0 2 } }$ </td></tr><tr><td>0.85</td><td>0.999</td><td> $6 . 4 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $5 . 8 _ { - 0 . 0 1 } ^ { + 0 . 1 }$ </td><td> $5 . 8 _ { - 0 . 0 1 } ^ { + 0 . 1 }$ </td><td> $6 . 1 _ { - 0 . 0 2 } ^ { + 0 . 1 }$ </td><td> $6 . 0 _ { - 0 . 0 5 } ^ { + 0 . 0 5 }$ </td><td> $6 . 0 _ { - 0 . 0 1 } ^ { + 0 . 0 0 1 }$ </td><td> $5 . 4 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> ${ \bf 3 . 1 } _ { - 0 . 1 } ^ { + 0 . 0 3 }$ </td></tr><tr><td>0.9</td><td>0.999</td><td> $6 . 4 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $5 . 8 _ { - 0 . 0 2 } ^ { + 0 . 0 5 }$ </td><td> $5 . 8 _ { - 0 . 0 1 } ^ { + 0 . 1 }$ </td><td> $5 . 9 _ { - 0 . 0 2 } ^ { + 0 . 1 }$ </td><td> $6 . 1 _ { - 0 . 0 3 } ^ { + 0 . 1 }$ </td><td> $6 . 0 _ { - 0 . 0 1 } ^ { + 0 . 0 0 5 }$ </td><td> $5 . 1 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $\mathbf { 3 . 1 } _ { - 0 . 1 } ^ { + 0 . 0 1 }$ </td></tr><tr><td>0.95</td><td>0.999</td><td> $6 . 4 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $5 . 8 _ { - 0 . 0 3 } ^ { + 0 . 0 3 }$ </td><td> $5 . 9 _ { - 0 . 0 4 } ^ { + 0 . 0 2 }$ </td><td> $5 . 8 _ { - 0 . 0 3 } ^ { + 0 . 0 3 }$ </td><td> $6 . 2 _ { - 0 . 0 2 } ^ { + 0 . 1 }$ </td><td> $5 . 9 _ { - 0 . 0 1 } ^ { + 0 . 0 0 2 }$ </td><td> $4 . 6 _ { - 0 . 0 3 } ^ { + 0 . 2 }$ </td><td> $\mathbf { 3 . 1 } _ { - \mathbf { 0 . 0 3 } } ^ { + \mathbf { 0 . 0 1 } }$ </td></tr><tr><td>0.85</td><td>0.9999</td><td> $6 . 4 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $6 . 0 _ { - 0 . 0 2 } ^ { + 0 . 0 5 }$ </td><td> $6 . 2 _ { - 0 . 0 3 } ^ { + 0 . 1 }$ </td><td> $6 . 1 _ { - 0 . 0 1 } ^ { + 0 . 1 }$ </td><td> $5 . 8 _ { - 0 . 0 4 } ^ { + 0 . 0 5 }$ </td><td> $6 . 0 _ { - 0 . 0 0 4 } ^ { + 0 . 0 0 3 }$ </td><td> $5 . 5 _ { - 0 . 1 } ^ { + 0 . 0 4 }$ </td><td> ${ \bf 3 . 1 } _ { - \bf n \ t } ^ { + \bf 0 . 0 3 }$ </td></tr><tr><td>0.9</td><td>0.9999</td><td> $6 . 4 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $6 . 0 _ { - 0 . 0 1 } ^ { + 0 . 0 5 }$ </td><td> $6 . 2 _ { - 0 . 0 1 } ^ { + 0 . 1 }$ </td><td> $6 . 0 _ { - 0 . 0 1 } ^ { + 0 . 1 }$ </td><td> $5 . 8 _ { - 0 . 0 3 } ^ { + 0 . 0 4 }$ </td><td> $6 . 0 _ { - 0 . 0 1 } ^ { + 0 . 0 0 1 }$ </td><td> $5 . 2 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $\mathbf { 3 . 1 } _ { - 0 . 1 } ^ { + 0 . 0 1 }$ </td></tr><tr><td>0.95</td><td>0.9999</td><td> $6 . 4 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $6 . 2 _ { - 0 . 0 3 } ^ { + 0 . 0 3 }$ </td><td> $6 . 3 _ { - 0 . 0 4 } ^ { + 0 . 1 }$ </td><td> $6 . 1 _ { - 0 . 0 3 } ^ { + 0 . 0 2 }$ </td><td> $5 . 9 _ { - 0 . 0 1 } ^ { + 0 . 1 }$ </td><td> $5 . 9 _ { - 0 . 0 0 4 } ^ { + 0 . 0 0 3 }$ </td><td> $4 . 8 _ { - 0 . 0 5 } ^ { + 0 . 1 }$ </td><td> $\mathbf { 3 . 1 } _ { - 0 . 0 3 } ^ { + 0 . 0 3 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

e) Results $\mathbf { \beta } - \mathbf { \beta } \beta _ { 1 }$ and $\beta _ { 2 }$ sensitivity: Table V repeats the comparison over a $3 \times 3$ grid of the first- and secondmoment exponential-decay rates $\beta _ { 1 } \in \{ 0 . 8 5 , 0 . 9 , 0 . 9 5 \}$ and $\beta _ { 2 } \in \{ 0 . 9 9 , 0 . 9 9 9 , 0 . 9 9 9 9 \}$ , at the base step size α and the primary architecture, with 3 seeds per pair, and Figure 2 shows the corresponding training- and test-loss curves. The two rates enter Adam, Adam (variable LR), NadamW, CvAMSGrad, and the direction of Adam-AURA, whereas they reach Muon and Muon-AURA only through the Adam-type update of the biases, and RPROP does not use them. The RPROP column nonetheless differs between pairs, for example 110 and 128 in $\mathcal { L } _ { \operatorname* { m i n } } ^ { ( m ) }$ , because each pair is a separate run and we did not set the XLA compiler used by JAX [57] to be fully deterministic on the GPU. Muon-AURA attains the lowest $\dot { \mathcal { L } } _ { \operatorname* { m i n } } ^ { ( m ) }$ and $A ^ { ( m ) }$ at every pair and is insensitive to both rates, with $A ^ { ( m ) }$ between 3.0 and 3.1, whereas the $\mathcal { L } _ { \operatorname* { m i n } } ^ { ( m ) }$ of Muon, exposed to them in the same way, varies by more than a factor of five. Adam-AURA has the second lowest $A ^ { ( m ) }$ at every pair and benefits most from a larger $\beta _ { 1 } \mathrm { { : } }$ : raising $\beta _ { 1 }$ from 0.85 to 0.95 lowers its $\mathcal { L } _ { \operatorname* { m i n } } ^ { ( m ) }$ by a factor of about 8 to 12 at each $\beta _ { 2 } ,$ against at most 1.7 for Adam. This is consistent with the mechanism of AURA, since a longer averaging window makes consecutive Adam directions agree more often, which increases the multiplier; the default value $\beta _ { 1 } = 0 . 9$ of Table II therefore leaves this gain

![](images/c3784e37e918258024a3c99f0e2e3fb8cc1518e301456e9793c8be39809bc7a9.jpg)

![](images/90923aee296937c3da90cc9cd10d7ff10c2fd4b0113a5b1d3513a16381abeeaf.jpg)  
(a) β<sub>1</sub> = 0.85, β<sub>2</sub> = 0.99

![](images/b294fb42ebe1715c8e68eab3151a22bfe528e231c399741997b28b32cddece3d.jpg)

![](images/2844bea7856b5379eb7f1b1e13508644af88b0c26f63c08ec6e15a317e2a9ca8.jpg)

![](images/63d3ea4b1fa18e133afdde6d8091266bbcd2af3130e1fa7bf0786a840babab7e.jpg)  
(b) β<sub>1</sub> = 0.9, β<sub>2</sub> = 0.99

![](images/90d0e321eef39648832688f1aaad9b402df4bd66fd122beb6756104fdb9de4e8.jpg)  
(c) $\beta _ { 1 } = 0 . 9 5 , \ \beta _ { 2 } = 0 . 9 9$

![](images/1d622963cad6e08a2bd3783f157c0e9eefdb9acd83656c1925a0df44d095b559.jpg)  
(d) β<sub>1</sub> = 0.85, β<sub>2</sub> = 0.999

![](images/820eeb5996933f6c76889f2d4251d4fd1967cef4b70b02e4a3f6018aa1fdda36.jpg)

![](images/1e04a46b351f44f9d9710d3d944c5ced2d93a5fb6e7a1c8aa87bb579e9d2d78c.jpg)

![](images/937a0e91ec9ae84e3e0d02726952e5e2ee06f1faf8761aaaac49a9c933dc0045.jpg)  
(e) β<sub>1</sub> = 0.9, β<sub>2</sub> = 0.999

![](images/76eebafbe40b05d6aae709d3b4e7402686fc3b26e20754c01f5aa23454c0849d.jpg)  
(f) β<sub>1</sub> = 0.95, β<sub>2</sub> = 0.999

![](images/2ede33805f5278f5c6499badd4aca7d708cfcc0426f73e177df27da1b49e9163.jpg)

![](images/ca8bc706c62c1212e91a469eab3ae71b31ba3a4c2db359afde5bd181f31b0991.jpg)  
(g) β = 0.85, β = 0.9999

![](images/9373d3ef114e2b6da24d1b5bc83d22c01e9e9a59b4f35290c34c667da9f7fd50.jpg)

![](images/7e9aa4536b29259dfe32baad724c35d4af683d6ae8661aa04331f586acb0c72e.jpg)  
(h) β<sub>1</sub> = 0.9, β<sub>2</sub> = 0.9999

![](images/b0427851306a1604201a606572774285bd971f2f2a73bc4515819e74f64913cf.jpg)

![](images/62875f42528aa0ed1b1b6cfd5a692556828b6bae5ba6ab1c3c1428b8c988f7fb.jpg)  
(i) $\beta _ { 1 } = 0 . 9 5 , \ \beta _ { 2 } = 0 . 9 9 9 9$  
Fig. 2. Training-set and test-set mean squared errors for the Non-holomorphic benchmark over the $3 \times 3 ~ \beta _ { 1 } / \beta _ { 2 }$ grid. Rows fix $\beta _ { 2 } ~ ( 0 . 9 9 , 0 . 9 9 9 , 0 . 9 9 9 9 ) ;$ columns fix $\beta _ { 1 } \ \mathrm { { ( 0 . 8 5 , 0 . 9 , 0 . 9 5 ) } }$ . The step size is fixed to its baseline, and each panel shows the training error above the test error. Solid curves show the pointwise median over the 3 seeds; shaded regions span the interquartile range (25th–75th percentile) over seeds.

## D. TEST 2: univariate scalar complex holomorphic function

a) Rationale: This relatively inexpensive test case allows us to perform multiple training runs. We consider an important class of complex-valued functions: holomorphic functions, which have applications in, $\mathrm { e . g . }$ , solid mechanics [58], [59]. We investigate the effect of the mini-batch size and show that excessively small mini-batches can constitute a failure mode for AURA.

b) Learning task: Domain sampling and loss construction follow Section IV-C. The activation and target function are specified below.

The domain is the square $\Omega _ { \zeta } ~ = ~ \{ \zeta ~ \in ~ \mathbb { C } ~ : ~ | \mathbf { R e } \zeta | ~ \leq$ $1 , \ | \mathrm { I m } \zeta | \leq 1 \}$ . The target is the entire, multiscale function

$$
\begin{array} { r l } { f _ { \mathrm { h o l } } ( \zeta ) = \exp ( ( 0 . 3 9 - 0 . 2 2 \mathrm { i } ) \zeta ) } & { } \\ { \quad } & { + 0 . 1 2 \exp ( 3 . 5 \zeta ) + 0 . 0 3 \exp ( ( - 2 . 8 + 3 . 3 \mathrm { i } ) \zeta ) } \\ { \quad } & { + 0 . 0 0 2 \exp ( ( - 5 . 3 - 4 . 6 \mathrm { i } ) \zeta ) } \\ { \quad } & { + 0 . 0 0 1 5 \exp ( ( 5 . 5 + 5 . 0 \mathrm { i } ) \zeta ) } \\ { \quad } & { + 0 . 0 1 5 \sin ( 5 . 5 \zeta ) + 0 . 0 5 9 \zeta ^ { 4 } - 0 . 0 4 0 \mathrm { i } \zeta ^ { 5 } , \quad ( 1 } \end{array}\tag{3}
$$

whose oblique complex exponentials produce steep variations near three corners of $\Omega _ { \zeta } ,$ while the sine and polynomial terms contribute oscillatory and algebraic structure.

c) Network architectures: The networks follow the affine– activation composition defined in Section IV-C, with $n _ { 0 } =$ $n _ { L } = 1$ and the componentwise entire activation $\varsigma ( \zeta ) = \exp ( \zeta )$ Since complex affine maps and the exponential function are entire, the networks are entire, and hence holomorphic in $\zeta ,$ by construction.

The complete benchmark configuration, including architectures, activation, dataset sizes, and optimizer hyperparameters, is given in Table VI.

TABLE VI  
CONFIGURATION OF THE HOLOMORPHIC OPTIMIZER BENCHMARK.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Primary architecture</td><td>(1, 32, 32, 32, 32, 1), 3,265 trainable param- eters</td></tr><tr><td>Secondary</td><td>(1, 128, 128, 128, 128, 128, 1), 66,433 train-</td></tr><tr><td>architecture</td><td>able parameters</td></tr><tr><td>Hidden activation</td><td>exp(ζ)</td></tr><tr><td>Training / test set</td><td>2,500 / 500 points (test: 20%)</td></tr><tr><td>Mini-batch size</td><td>256 points</td></tr><tr><td>Updates / seeds</td><td>12,000 /  $\{ 0 , \ldots , 4 \}$ </td></tr><tr><td>Arithmetic</td><td>complex64 (32-bit real components) beta-scaled Gaussian-fallback rule for entire</td></tr><tr><td>Initialization</td><td>activations [58],  $\beta _ { \mathrm { i n i t } } = 0 . 5 ;$  zero biases</td></tr><tr><td>Base learning rate α (primary)</td><td> $5 \times 1 0 ^ { - 5 }$   $5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Secondary- architecture learning rate</td><td></td></tr><tr><td>Optimizer hyperparameters</td><td>default values; see Table II</td></tr></table>

d) Results – mini-batch size sensitivity: Table VII reports $\mathcal { L } _ { \operatorname* { m i n } } ^ { ( m ) }$ and $A ^ { ( m ) }$ for the mini-batch sizes 32, 128, and 256 at both architectures, together with the number of seeds whose training loss becomes non-finite $( \mathrm { N a N } ) ;$ the time ratio is reported only at size 256, the mini-batch size at which the SGD baseline is measured. Where every seed diverges, $A ^ { ( m ) }$ is omitted (–), since an area computed over the steps preceding the divergence is not comparable with that of a complete run. Figure 3 shows the corresponding training- and test-loss curves. The update budget and the step size of each architecture are held fixed, so that a smaller mini-batch means fewer passes over the training set, but the same number of updates. At the largest mini-batch, Muon-AURA and Adam-AURA rank first and second in both $\mathcal { L } _ { \operatorname* { m i n } } ^ { ( m ) }$ and $A ^ { ( m ) }$ at both architectures, with the minimum loss of Muon-AURA about 22 to 27 times lower than that of the best baseline; Muon-AURA remains the best method at size 128, where Adam-AURA already diverges in one or two seeds, and Muon-AURA and RPROP in one and two seeds at the secondary architecture. At size 32 the ranking is reversed: both AURA variants diverge in all five seeds at both architectures. RPROP, whose step size also follows the agreement between consecutive steps, is the only baseline that diverges, which suggests that the gradient noise of small minibatches corrupts this agreement signal; Adam, Adam (variable LR), NadamW, CvAMSGrad, and Muon converge in every run. AURA adds at most 0.13 to the time ratio of its base optimizer.

<sub>-SIZE</sub> <sub>SENSITIVITY</sub> <sub>FOR</sub> <sub>THEHO</sub>L<sup>OMORPHIC</sup> <sup>B</sup>
<table><tr><td></td><td></td><td colspan="3">L(m)</td><td colspan="3">A(m)</td><td colspan="2">Training time (×SGD)</td><td colspan="2">NaN</td></tr><tr><td>Arch.</td><td>Optimizer</td><td>32</td><td>128</td><td>256</td><td>32</td><td>128</td><td>256</td><td></td><td>256</td><td>128</td><td>256</td></tr><tr><td rowspan="9">Primary</td><td>RPROP</td><td>(3.3+10.1) × 10−2 -3.1</td><td>(1.5+7.5) × 10−3 +0:0</td><td>(5.5+5.6) × 10−4 -1.2</td><td>10.5+0.2</td><td>8.3+0.6</td><td>7.81+0.09 -0.1</td><td>-0.04</td><td>1.65+0.02</td><td>0</td><td>0</td></tr><tr><td>Adam</td><td>(3.3+5.6) × 10 -4 -0.4</td><td>(2.2+2.0 × 10−4 -0.04</td><td>(1.3+1.4 × 10−4 -0.03</td><td></td><td>8.14+0.14 -0.04</td><td>7.8+0.2 -0.02</td><td></td><td></td><td>0</td><td>0</td></tr><tr><td>Adam (variable LR)</td><td>1 × 10−3</td><td>. × 10−4</td><td>× 10−4</td><td></td><td>8.1+0.2 8.0 +0.0</td><td>7.8+0.3</td><td>1.58+0.083</td><td></td><td>0</td><td>0</td></tr><tr><td>NadamW</td><td>× 10 -4 1.4</td><td>× 10−4</td><td>× 10−4 -0.1</td><td>8.80+03</td><td>-0.1</td><td>7.7+0.2 -0.01</td><td></td><td>-0.004</td><td>0</td><td>0</td></tr><tr><td>CvAMSGrad</td><td>(6.9+5.9) × 10−4 1.3</td><td>(6.8+0.) × 10−4</td><td>1+1.3) (6.1 2 × 10−4</td><td></td><td>8.1+0.2</td><td>7.9+0.2 -0.1</td><td>1.63+0.04</td><td></td><td>0</td><td>0</td></tr><tr><td>Muon</td><td>× 10−4</td><td>-3.3 (3.0+0.1) × 10−4</td><td>× 10−4</td><td>8.76+0.14</td><td>-0.001 8.06+0.11</td><td>7.80+0.09</td><td>1.88+0.03</td><td>-0.06</td><td></td><td>0</td></tr><tr><td>Adam-AURA</td><td>× 10−2 2.3</td><td>× 10−5 -0.8</td><td>× 10-5 0.8</td><td></td><td>-0.001 7.7</td><td>60.03 6.75+0.05</td><td>-0.03</td><td></td><td>0 2</td><td>0</td></tr><tr><td>Muon-AURA</td><td>× 10−1</td><td>(3.3 +0.9 × 10-6 0.3</td><td>(5.3+2.3) × 10-6 -0.4)</td><td></td><td>5.9+0.62</td><td>5.72 -0.003</td><td>+0.06 -0.02</td><td>-0.09 1.90+0.01</td><td>0</td><td>0</td></tr><tr><td>RPROP</td><td>(1.3+0.3)</td><td>× 10−5</td><td>× 10−5</td><td></td><td>7+4</td><td>6.97+0.02</td><td></td><td>-0.11</td><td></td><td></td></tr><tr><td rowspan="8">Secondary</td><td></td><td>-0.2 × 100 </td><td>(1.8+9.3)</td><td>(7.9+1.5) (1.2+2</td><td>1</td><td>-0.1</td><td></td><td>1.56+0.10</td><td>-0.03 +0.04</td><td>2</td><td>0</td></tr><tr><td>Adam</td><td>(9.0+29.4) × 10−4 -4.5</td><td>× 10−4</td><td>+0.2) × 10−4</td><td>9.4+0.02 0.4</td><td>8.5+0.3 -0.1</td><td>8.1+0. 8.1+0.2</td><td>1.50</td><td>-0.03</td><td>0</td><td>0</td></tr><tr><td>Adam (variable LR)</td><td>(2.0+2.0) × 10−2 (4.2+37.8)</td><td>×10-3</td><td>(2.3+42.5 × 10−4 -0.02</td><td>9.6+0.4</td><td>8.7+0.3 8.3+0.3</td><td></td><td></td><td>1.52+0.03</td><td>0</td><td>0</td></tr><tr><td>NadamW</td><td>× 10−4</td><td>(0. ) × 10−5</td><td>× 10−4</td><td>9.4+0.02 −+0.41</td><td></td><td></td><td></td><td></td><td>0</td><td>0</td></tr><tr><td>CvAMSGrad</td><td>(1.1+3.4 ) × 10−3 -0.02</td><td>× 10−4 -3.8</td><td>(2.6+16.6) × 10−4 -0.1</td><td>9.2</td><td></td><td></td><td>2.09+0.021</td><td>-0.01</td><td>0</td><td>0</td></tr><tr><td>Muon</td><td>(8.5+0.5 × 10-5 -0.8</td><td>(6.7+19)) ×10-5</td><td>(4.2+ 2+0.) × 10-5</td><td></td><td>8.4+0. -0.04</td><td>F0.2</td><td></td><td>-0.0004</td><td>0</td><td>0</td></tr><tr><td>Adam-AURA</td><td>(4.2+1 × 10−2 00</td><td>(3.5+43)) × 10-5</td><td>(1.9+0.6) × 10−5</td><td></td><td>7.5+0.5</td><td>6.67+0.06</td><td></td><td>1.60+0.06 -0.06</td><td>1</td><td>0</td></tr><tr><td>Muon-AURA</td><td>(1.8+0.4) × 10−1 -1.1</td><td>(7.4+20.3) × 10−6 -4.8</td><td>×10-6 -0.1)</td><td></td><td>6.7</td><td>5+0.3 5.20 -0.5</td><td>+0.0002 -0.05</td><td>2.03+0.02 -0.02</td><td>1</td><td>0</td></tr></table>

Loss: TEST 2 – holomorphic  
![](images/6f712c342b95aabad8bc0eaea5f4758eb17f5a0028f5964521dbd1cb02bf722b.jpg)  
Fig. 3. Training-set (left panel of each row) and test-set (right panel) mean squared errors for the Holomorphic optimizer benchmark. The two architecture blocks show the same feedforward network trained independently at two architectures (Table VI): the primary architecture (1, 32, 32, 32, 32, 1) (3,265 parameters) at the top, and the wider and deeper secondary architecture (1, 128, 128, 128, 128, 128, 1) (66,433 parameters), trained at a ten-times-smaller step size, at the bottom. Within each architecture block, the rows sweep the mini-batch size across 32, 128 and 256 points from top to bottom at a fixed 12,000-update budget so a smaller mini-batch means fewer epochs over the training set, not fewer optimizer updates. Solid curves show the pointwise median over all 5 seeds; shaded regions span the interquartile range (25th–75th percentile) over seeds, whose edges are also drawn as dotted curves.

## E. TEST 3: multivariate complex $\mathbb { C } ^ { 4 }$ function

a) Rationale: This test uses a classical benchmark for complex-valued networks with four complex inputs [36], [52], [60]–[63]. We investigate the sensitivity of the optimizers to the floating-point precision, training a single architecture in single and double precision, and repeat the step-size sweep of Section IV-C.

b) Learning task: The target is the multivariate benchmark function considered in Section 5.1 of [36]:

$$
f _ { \mathrm { m c 4 } } ( \zeta _ { 1 } , \zeta _ { 2 } , \zeta _ { 3 } , \zeta _ { 4 } ) = \frac { 1 } { 1 . 5 } \left( \frac { \zeta _ { 2 } ^ { 2 } } { \zeta _ { 1 } } + \zeta _ { 3 } + 1 0 \zeta _ { 1 } \zeta _ { 4 } \right) , \qquad \zeta \in \Omega _ { \zeta } .\tag{14}
$$

The training and test sets are drawn as in (10)–(11), with

$$
\Omega _ { \zeta } = \bigl \{ \zeta \in \mathbb { C } ^ { 4 } :  { \mathrm { R e } } \zeta _ { k } ,  { \mathrm { I m } } \zeta _ { k } \in [ 0 . 5 , 1 ] , \ k = 1 , \dots , 4 \bigr \} .\tag{15}
$$

This domain keeps $\zeta _ { 1 }$ away from zero, ensuring that the target is well defined and holomorphic on a neighborhood of $\Omega _ { \zeta }$ . The loss is the mean squared error (12), evaluated at the vectorvalued inputs $\zeta .$

c) Network architecture: A single network is used, following the affine–activation composition defined in Section IV-C with $n _ { 0 } = 4$ inputs, the components of $\boldsymbol { \zeta } = ( \zeta _ { 1 } , \zeta _ { 2 } , \zeta _ { 3 } , \zeta _ { 4 } ) \in$ $\mathbb { C } ^ { 4 }$ , and $n _ { L } = 1$ output. The bounded activation

$$
\begin{array} { r } { \varsigma ( \zeta ) = \operatorname { t a n h } ( \operatorname { R e } \zeta ) + \mathrm { i } \operatorname { t a n h } ( \operatorname { I m } \zeta ) , } \end{array}
$$

is applied componentwise in the hidden layers, so the network is generally non-holomorphic, although the target is holomorphic. Here, unlike in Section IV-C, we did not use the SiLU function, in order to increase the variability of the test cases.

The complete benchmark configuration, including the architecture, activation, dataset sizes, and the swept step sizes and precisions, is given in Table VIII.

TABLE VIII  
CONFIGURATION OF THE MULTIVARIATE C4 OPTIMIZER BENCHMARK.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Network architecture</td><td>(4, 128, 128, 128, 128, 128, 1), 66,817 train- able parameters</td></tr><tr><td>Hidden activation</td><td>tanh(Reζ) + i tanh(Imζ) (componentwise across the four inputs)</td></tr><tr><td>Training / test set</td><td>1,296 / 259 points (test: 20%)</td></tr><tr><td>Mini-batch size Updates / seeds</td><td>128 points</td></tr><tr><td>Arithmetic</td><td> $1 2 , 0 0 0 / \left\{ 0 , \ldots , 4 \right\}$  complex64 and complex128 (single-</td></tr><tr><td>Initialization</td><td>and double-precision real components; both swept)</td></tr><tr><td></td><td>Glorot/Xavier normal rule [56], applied sep- arately to the real and imaginary parts; zero biases</td></tr><tr><td>Base learning rate α Optimizer hyperparameters</td><td> $1 \times 1 0 ^ { - 4 }$  , swept over  $\alpha / 1 0 ,$  α and 10α default values; see Table II</td></tr></table>

d) Results – α and precision sensitivity: Table IX reports the three metrics for the step sizes $\alpha / 1 0 , \alpha ,$ and 10α in single (complex64) and double (complex128) precision, and Figure 4 shows the corresponding training- and test-loss curves. The precision has little influence: double precision changes $A ^ { ( m ) }$ by at most about 0.1 for every method and step size other than Muon-AURA, and only Muon-AURA improves by more than 0.1 at every step size, by 0.11 to 0.28. The step size matters far more: over its 100-fold range, $\mathcal { L } _ { \operatorname* { m i n } } ^ { ( m ) }$ varies by more than an order of magnitude for Adam, its variants, and Muon, but by at most about a factor of two for Adam-AURA; RPROP is similarly insensitive, but its minimum loss is about thirty times that of Adam-AURA. Muon-AURA attains the lowest $\mathcal { L } _ { \operatorname* { m i n } } ^ { ( m ) }$ and $A ^ { ( m ) }$ in all six settings, with a margin of about 2 in $\bar { A } ^ { ( m ) }$ over the best baseline, that is, a geometricmean training loss about two orders of magnitude lower, and Adam-AURA ranks second in $A ^ { ( m ) }$ in all six settings. AURA adds at most 0.15 to the time ratio of its base optimizer.

TABLE IX  
STEP-SIZE AND PRECISION SENSITIVITY FOR THE MULTIVARIATE C4 BENCHMARK.
<table><tr><td></td><td></td><td colspan="3"> $\mathcal { L } _ { \mathrm { m i n } } ^ { ( m ) } ~ ( \times 1 0 ^ { - 4 } )$ </td><td colspan="3"> $A ^ { ( m ) }$ </td><td>Training time (×SGD)</td></tr><tr><td>Prec.</td><td>Optimizer</td><td> $\alpha / 1 0$ </td><td>α</td><td>10α</td><td> $\alpha / 1 0$ </td><td>α</td><td>10α</td><td>α</td></tr><tr><td rowspan="9">Single</td><td>RPROP</td><td> $3 8 _ { - 4 } ^ { + 3 }$ </td><td> $3 6 _ { - 1 } ^ { + 2 }$ </td><td> $3 1 _ { - 2 } ^ { + 5 }$ </td><td> $8 . 0 4 _ { - 0 ~ 0 7 } ^ { + 0 . 0 1 }$ </td><td> $7 . 9 7 _ { - 0 . 0 1 } ^ { + 0 . 1 5 }$ </td><td> $7 . 9 2 7 _ { - 0 . 0 0 7 } ^ { + 0 . 0 0 9 }$ </td><td> $1 . 5 5 _ { - 0 } ^ { + 0 . 0 5 }$ </td></tr><tr><td>Adam</td><td> $3 6 1 _ { - 1 4 } ^ { + 1 9 }$ </td><td> $4 . 1 _ { - 0 . 2 } ^ { + 0 . 2 } $ </td><td> $2 . 2 _ { - 0 . 1 } ^ { + 0 . 0 4 }$ </td><td> $9 . 5 0 _ { - 0 . 0 5 } ^ { + 0 . 0 0 2 }$ </td><td> $8 . 0 2 _ { - 0 . 0 5 } ^ { + \ddot { 0 } . \ddot { 0 } \bar { 2 } }$ </td><td> $7 . 8 5 _ { - 0 . 0 1 } ^ { + 0 . 0 2 }$ </td><td> $1 . 5 5 _ { - 0 . 1 1 } ^ { + 0 . 0 0 0 4 }$ </td></tr><tr><td>Adam (variable LR)</td><td> $9 9 7 _ { - 1 4 3 } ^ { + 4 9 }$ </td><td> $1 2 _ { - 0 . 5 } ^ { + 0 . 5 }$ </td><td> $1 . 7 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $9 . 6 8 _ { - 0 . 0 7 } ^ { + 0 . 0 1 }$ </td><td> $8 . 1 5 _ { - 0 . 1 0 } ^ { + 0 . 0 0 5 }$ </td><td> $7 . 4 0 _ { - 0 . 0 0 2 } ^ { + 0 . 0 4 }$ </td><td> $1 . 5 3 _ { - 0 . 0 1 } ^ { + 0 . 0 3 }$ </td></tr><tr><td>NadamW</td><td> $3 2 1 _ { - 7 } ^ { + 3 5 }$ </td><td> $7 . 4 _ { - 0 . 4 } ^ { + 0 . 3 }$ </td><td> $8 . 9 _ { - 3 . 3 } ^ { + 0 . 6 }$ </td><td> $9 . 5 0 _ { - 0 . 0 5 } ^ { + 0 . 0 0 1 }$ </td><td> $8 . 2 3 _ { - 0 . 0 4 } ^ { + 0 . 0 1 }$ </td><td> $8 . 4 3 _ { - 0 . 0 3 } ^ { + 0 . 0 1 }$ </td><td> $\mathbf { 1 . 5 1 _ { - 0 . 0 1 } ^ { + 0 . 0 3 } }$ </td></tr><tr><td>CvAMSGrad</td><td> $1 8 2 1 _ { - 5 0 8 } ^ { + 3 }$ </td><td> $1 8 _ { - 5 } ^ { + 0 . 1 }$ </td><td> $2 . 2 _ { - 0 . 0 1 } ^ { + 0 . 2 }$ </td><td> $9 . 8 4 _ { - 0 . 0 6 } ^ { + 0 . 0 2 }$ </td><td> $8 . 2 5 _ { - 0 . 1 3 } ^ { + 0 . 0 3 }$ </td><td> $7 . 3 4 _ { - 0 . 0 3 } ^ { + 0 . 0 3 }$ </td><td> $1 . 6 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td></tr><tr><td>Muon</td><td> $0 . 8 _ { - 0 . 1 } ^ { + 0 . 0 1 }$ </td><td> $2 7 _ { - 1 } ^ { + 0 . 5 }$ </td><td> $1 2 _ { - 0 . 4 } ^ { + 0 . 1 }$ </td><td> $8 . 3 5 _ { - 0 . 0 9 } ^ { + 0 . 0 3 }$ </td><td> $7 . 7 5 _ { - 0 . 0 2 } ^ { + 0 . 0 1 }$ </td><td> $7 . 8 1 8 _ { - 0 . 0 0 3 } ^ { + 0 . 0 1 7 }$ </td><td> $2 . 1 2 8 _ { - 0 . 0 0 6 } ^ { + 0 . 0 1 3 }$ </td></tr><tr><td>Adam-AURA</td><td> $1 . 2 _ { \phantom { + 0 . 0 3 } 0 \phantom { + 0 . 0 3 } } ^ { + 0 . 0 3 }$ </td><td> $0 . 8 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $0 . 9 _ { - 0 . 0 1 } ^ { + 0 . 2 }$ </td><td> $7 . 1 0 _ { - 0 . 0 1 } ^ { + 0 . 0 6 }$ </td><td> $6 . 7 4 _ { - 0 . 0 0 4 } ^ { + 0 . 0 6 }$ </td><td> $6 . 6 8 _ { - 0 . 0 2 } ^ { + 0 . 0 9 }$ </td><td> $1 . 6 2 _ { - 0 . 0 0 2 } ^ { + 0 . 0 7 }$ </td></tr><tr><td>Muon-AURA</td><td> $\mathbf { 0 . 1 } _ { - 0 . 0 0 3 } ^ { + 0 . 0 1 }$ </td><td> $\mathbf { 0 . 1 } _ { - \mathbf { 0 . 0 0 1 } } ^ { + \mathbf { 0 . 0 1 } }$ </td><td> $\mathbf { 0 . 1 } _ { - 0 . 0 0 2 } ^ { + 0 . 0 0 5 }$ </td><td> ${ \bf 5 . 7 5 _ { - 0 . 0 3 } ^ { + 0 . 0 1 } }$ </td><td> ${ \bf 5 . 6 4 } _ { - 0 . 0 6 } ^ { + { \bf U . U 1 } }$ </td><td> ${ \bf 5 . 3 4 } _ { - 0 . 0 3 } ^ { + 0 . 0 2 }$ </td><td> $2 . 0 7 _ { - 0 . 0 2 } ^ { + 0 . 0 2 }$ </td></tr><tr><td>RPROP</td><td> $4 1 _ { - 1 0 } ^ { + 2 3 }$ </td><td> $2 7 _ { - 1 } ^ { + 1 }$ </td><td> $3 2 _ { - 5 } ^ { + 1 }$ </td><td> $8 . 1 _ { - 0 . 2 } ^ { + 0 . 2 }$ </td><td> $7 . 8 5 _ { - 0 . 0 1 } ^ { + 0 . 0 4 }$ </td><td></td><td></td></tr><tr><td rowspan="8">Double</td><td></td><td></td><td></td><td></td><td></td><td></td><td> $7 . 9 0 3 _ { - 0 . 0 0 5 } ^ { + 0 . 0 0 4 }$ </td><td> $\mathbf { 1 . 5 5 _ { - n \ n 1 } ^ { + 0 . 0 6 } }$ </td></tr><tr><td>Adam Adam (variable LR)</td><td> $3 3 2 _ { - 1 0 } ^ { + 4 2 }$ </td><td> $4 . 1 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $2 . 0 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $9 . 4 7 _ { - 0 . 0 1 } ^ { + 0 . 0 4 }$ </td><td> $8 . 0 1 _ { - 0 . 0 5 } ^ { + 0 . 0 1 }$ </td><td> $7 . 8 6 0 _ { - 0 . 0 1 0 } ^ { + 0 . 0 0 6 }$ </td><td> $1 . 6 2 _ { - 0 . 0 2 } ^ { + 0 . 0 2 }$ </td></tr><tr><td>NadamW</td><td> $1 0 1 0 _ { - 1 1 1 } ^ { + 6 }$   $2 9 1 _ { - 1 1 } ^ { + \hat { 4 } \hat { 6 } }$ </td><td> $1 0 _ { - 0 . 5 } ^ { + 1 }$ </td><td> $1 . 7 _ { - 0 . 1 } ^ { + 0 . 2 }$   $7 . 7 _ { - 1 . 4 } ^ { + 1 . 8 }$ </td><td> $9 . 6 5 _ { - 0 . 0 2 } ^ { + 0 . 0 5 }$ </td><td> $8 . 0 9 _ { - 0 . 0 6 } ^ { + 0 . 0 4 }$ </td><td> $7 . 4 1 1 _ { - 0 . 0 1 7 } ^ { + 0 . 0 0 2 }$ </td><td> $1 . 6 1 _ { - 0 . 0 7 } ^ { + 0 . 0 7 }$ </td></tr><tr><td>CvAMSGrad</td><td> $1 5 4 0 _ { - 1 7 0 } ^ { + 1 7 9 }$ </td><td> $7 . 0 _ { - 0 . 3 } ^ { + 1 . 1 }$ </td><td>0:4</td><td> $9 . 4 6 _ { - 0 . 0 1 } ^ { + 0 . { \overset { . } { 0 } } { 5 } }$ </td><td> $8 . 2 2 _ { - 0 . 0 3 } ^ { + 0 . 0 0 5 }$ </td><td> $8 . 4 2 6 _ { - 0 . 0 1 0 } ^ { + 0 . 0 0 3 }$ </td><td> $1 . 6 4 4 _ { - 0 . 0 0 7 } ^ { + 0 . 0 0 6 }$ </td></tr><tr><td>Muon</td><td> $1 . 2 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td> $1 6 _ { - 0 . 4 } ^ { + 4 }$   $3 0 _ { - 1 } ^ { + \tilde { 0 . 4 } }$ </td><td> $2 . 4 _ { - 0 . 1 } ^ { + 0 . 3 }$   $1 4 _ { - 1 } ^ { + 0 . \bar { 4 } }$ </td><td> $9 . 7 9 _ { - 0 . 0 0 2 } ^ { + 0 . 0 5 }$ </td><td> $8 . 1 6 _ { - 0 . 0 5 _ { - } } ^ { + 0 . 0 8 }$ </td><td> $7 . 3 5 _ { - 0 . 0 2 } ^ { + 0 . 0 4 }$ </td><td> $1 . 5 5 _ { - 0 . 0 5 } ^ { + 0 . 1 0 }$ </td></tr><tr><td>Adam-AURA</td><td> $1 . 2 _ { - 0 . 0 2 } ^ { + 0 . 2 }$ </td><td> $0 . 9 _ { - 0 . 1 } ^ { + \hat { 0 } . 1 }$ </td><td> $0 . 9 _ { - 0 . 1 } ^ { + \dot { 0 } . 2 }$ </td><td> $8 . 4 _ { - 0 . 1 } ^ { + 0 . 2 }$   $7 . 1 4 _ { - 0 . 0 2 } ^ { + 0 . 0 2 }$ </td><td> $7 . 7 8 1 _ { - 0 . 0 0 0 1 } ^ { + 0 . 0 1 6 }$   $6 . 7 8 _ { - 0 . 0 2 } ^ { + 0 . 0 3 }$ </td><td> $7 . 8 6 _ { - 0 ~ 0 5 } ^ { + 0 . 0 2 }$   $6 6 \mathbf { 0 } ^ { + 0 . 0 5 }$ </td><td> $2 . 4 7 _ { - 0 . 0 2 } ^ { + 0 . 0 4 }$ </td></tr><tr><td>Muon-AURA</td><td></td><td> $\mathbf { 0 . 0 3 _ { - 0 . 0 0 2 } ^ { + 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 0 3 _ { - 0 . 0 0 1 } ^ { + 0 . 0 0 2 } }$ </td><td></td><td></td><td> $6 . 6 9 _ { - 0 . 0 7 } ^ { + 0 . 0 5 }$ </td><td> $1 . 7 6 _ { - 0 . 0 8 } ^ { + 0 . 0 6 }$ </td></tr><tr><td></td><td> $\mathbf { 0 . 1 _ { - 0 . 0 0 2 } ^ { + 0 . 0 1 } }$ </td><td></td><td></td><td> ${ \bf 5 . 6 4 } _ { - 0 . 0 7 } ^ { + 0 . 0 5 }$ </td><td> ${ \bf 5 . 3 6 _ { - 0 . 0 2 } ^ { + 0 . 0 3 } }$ </td><td> ${ \bf 5 . 1 3 _ { - 0 . 0 2 } ^ { + 0 . 0 2 } }$ </td><td> $2 . 4 4 _ { - 0 . 0 4 } ^ { + 0 . 0 7 }$ </td></tr></table>

![](images/880b617be713ac44039f2e7bd3a093fb5f0b37ece4cd5bff7f0a9c67b23cf4f3.jpg)  
Fig. 4. Training-set and test-set mean squared errors for the Multivariate C4 optimizer benchmark, on the single benchmark architecture (4, 128, 128, 128, 128, 128, 1) (66,817 parameters; Table VIII). The two precision blocks show the same network trained in single-precision (complex64, top) and double-precision (complex128, bottom) arithmetic. Within each precision block, the rows sweep the base step size $\alpha = 1 \stackrel { - } { \times } 1 0 ^ { - 4 }$ across α/10, α and 10α from top to bottom. Solid curves show the pointwise median over all 5 seeds; shaded regions span the interquartile range (25th–75th percentile) over seeds, whose edges are also drawn as dotted curves.

a) Rationale: This test evaluates the optimizers on a physics-informed learning task while retaining a fully connected holomorphic architecture. In addition, the hyperparameters of every optimizer are tuned under a limited budget, as recommended in [14], so that the methods are compared at, or close to, their best performance and no baseline is penalized by an unfavorable default.

b) Learning task: The ultimate goal is to solve the two-dimensional linear elasticity problem for a homogeneous material. We consider the geometry and boundary conditions of the plate-with-a-circular-hole benchmark presented in [58]. The computational domain is the upper-left quadrant

$$
\begin{array} { r } { \Omega _ { \zeta } = \{ \zeta = x + \mathrm { i } y : - a < x < 0 , 0 < y < a , x ^ { 2 } + y ^ { 2 } > r ^ { 2 } \} , } \end{array}
$$

with $a = 2 . 5$ m and $r = 1$ m. Traction conditions are imposed on the outer edges and hole arc, while symmetry conditions are imposed on the cut edges; see Section 4.1.2 of [58] for details.

Two holomorphic neural networks, $\varphi _ { \mathcal { W } } : \mathbb { C } \to \mathbb { C }$ and $\psi _ { \mathcal { W } } :$ $\mathbb { C } \to \mathbb { C } .$ , take the spatial coordinate $\zeta \in \Omega _ { \zeta }$ as input and approximate the Kolosov–Muskhelishvili potentials $\varphi$ and $\psi ,$ respectively. Under plane-strain conditions with $\lambda = \mu =$ 1 MPa, the Kolosov constant is $\kappa = ( \lambda + 3 \mu ) / ( \lambda + \mu ) = 2 .$ and stresses and displacements follow from

$$
\begin{array} { r l } & { \sigma _ { x x } = \mathrm { R e } ( 2 \varphi ^ { \prime } - \overline { { \zeta } } \varphi ^ { \prime \prime } - \psi ^ { \prime } ) , } \\ & { \sigma _ { y y } = \mathrm { R e } ( 2 \varphi ^ { \prime } + \overline { { \zeta } } \varphi ^ { \prime \prime } + \psi ^ { \prime } ) , } \\ & { \sigma _ { x y } = \mathrm { I m } ( \overline { { \zeta } } \varphi ^ { \prime \prime } + \psi ^ { \prime } ) , } \end{array}\tag{16}
$$

$$
u _ { x } + \mathrm { i } u _ { y } = \frac { 1 } { 2 \mu } \big ( \kappa \varphi - \zeta \overline { { \varphi ^ { \prime } } } - \overline { { \psi } } \big ) .\tag{17}
$$

Primes denote differentiation with respect to $\zeta ,$ evaluated by holomorphic automatic differentiation. This representation

satisfies the governing elasticity equations by construction, so training minimizes only the boundary-condition residuals:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { P I H N N } } ( \mathcal { W } ) = \displaystyle \frac { \alpha _ { N } } { | \mathcal { Z } _ { N } | } \sum _ { \zeta \in \mathcal { Z } _ { N } } \| \pmb { \sigma } _ { \mathcal { W } } ( \zeta ) \mathbf { n } - \mathbf { t } _ { 0 } ( \zeta ) \| ^ { 2 } } \\ & { \qquad + \displaystyle \frac { \alpha _ { S } } { | \mathcal { Z } _ { S } | } \sum _ { \zeta \in \mathcal { Z } _ { S } } \left( \sigma _ { x y , \mathcal { W } } ( \zeta ) ^ { 2 } + | \pmb { u } _ { \mathcal { W } } ( \zeta ) \cdot \mathbf { n } | ^ { 2 } \right) . } \end{array}\tag{18}
$$

Here, ${ \mathcal { Z } } _ { N }$ and ${ \mathcal { Z } } _ { S }$ contain the points on the traction and symmetry boundaries, respectively, n is the outward unit normal, $\mathbf { t } _ { 0 }$ is the prescribed traction, and $\alpha _ { N } , \alpha _ { S }$ are the corresponding boundary-length fractions defined in [58]. The boundary points remain fixed throughout training, and every update uses the full training set.

c) Network architecture: We use two fully connected networks, $\varphi _ { \mathcal { W } }$ and $\psi _ { \mathcal { W } }$ , each with

$$
( n _ { 0 } , n _ { 1 } , n _ { 2 } , n _ { 3 } , n _ { 4 } ) = ( 1 , 6 4 , 6 4 , 6 4 , 1 ) .
$$

The three hidden layers use the componentwise activation $\varsigma ( \zeta ) = \exp ( \zeta )$ , and the output layer is affine. Both potentials are entire functions of $\zeta .$ The networks are wider than in [58], which uses 10 units per hidden layer. A larger network makes the benchmark more meaningful for the optimizers. The wider networks require a smaller learning rate, $1 0 ^ { - 3 }$ instead of $1 0 ^ { - 2 }$ Adam (variable LR) uses the learning-rate schedule of [58].

The complete benchmark configuration is given in Table X.

TABLE X  
CONFIGURATION OF THE PINN OPTIMIZER BENCHMARK.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Architecture (each of 4,ψ)</td><td>(1, 64, 64, 64, 1), 17,026 trainable parame- ters in total</td></tr><tr><td>Hidden activation</td><td>exp(ζ)</td></tr><tr><td>Training / test set</td><td>200 / 20 boundary points (full batch, no</td></tr><tr><td>Updates / seeds</td><td>mini-batching) 6,000  $/ \{ 0 , \ldots , 4 \}$  (weight initialization</td></tr><tr><td>Arithmetic</td><td>only, boundary sampling fixed) complex64 (32-bit real components)</td></tr><tr><td>Initialization</td><td>initialization rule defined in [58] (their  $\mathbf { A l - }$ </td></tr><tr><td>Base learning rate α</td><td>gorithm 1),  $\beta _ { \mathrm { i n i t } } = 0 . 5 ;$  zero biases  $\bar { 1 } \times 1 0 ^ { - 3 }$  , starting point of the search; tuned</td></tr><tr><td>Optimizer</td><td>values in Table XI tuned per method (Table XI), starting from</td></tr><tr><td>hyperparameters</td><td>the default values of Table II</td></tr></table>

d) Hyperparameter optimization: The learning rate and the principal hyperparameters of every optimizer are searched jointly with the multivariate tree-structured Parzen estimator (TPE) [64] of Optuna [65]; Table XI lists the search space and the selected values, and every other hyperparameter keeps the value of Table II. Every searched quantity is sampled on a logarithmic scale except $\chi _ { \mathrm { a } } ; \chi _ { \mathrm { o } }$ and $\Psi _ { \mathrm { a } }$ are obtained from the sampled gap $\chi _ { \mathrm { a } } - \chi _ { \mathrm { o } }$ and ratio $\Psi _ { \mathrm { a } } / \Psi _ { \mathrm { o } } ,$ , which preserve the orderings of Table I. Each method receives the same budget of 50 trials: the first evaluates the starting point, namely the values of Table II with $\alpha = 1 0 ^ { - 3 }$ , sampling is random until ten trials are complete, and TPE proposes the remaining ones. A trial trains the networks from three weight-initialization seeds, 100, 101, 102 , disjoint from the evaluation seeds of Table X, and is scored by the mean of $A ^ { ( m , s ) }$ (7) over them; a trial whose loss becomes non-finite on any seed receives a score of at least $C = 1 0 .$ , the area of a loss held at 1, and is excluded from the selection. The admissible trial with the lowest score is selected and evaluated on the five seeds of Table X as in the other cases, without an intermediate confirmation stage; the search costs $8 \times 5 0 \times 3 = 1 . 2 0 0$ training runs. Adam-AURA is searched after Adam, with $\beta _ { 1 }$ and $\beta _ { 2 }$ fixed at the values selected for Adam, so that only α and the five gates $\beta _ { \zeta } , \chi _ { \mathrm { a } } , \chi _ { \mathrm { o } } , \Psi _ { \mathrm { a } } .$ , and $\Psi _ { \mathrm { o } }$ are searched. This exploits the plug-in property of AURA and isolates the effect of its gates on a tuned base optimizer; counted as a whole, Adam-AURA thus receives twice the budget of the other methods, but the coupling between the Adam and the AURA hyperparameters is left unexplored. Muon-AURA is instead searched independently.

TABLE XI  
HYPERPARAMETER SEARCH OF THE PINN BENCHMARK.
<table><tr><td>Optimizer</td><td>Search space</td><td>Selected values</td></tr><tr><td>RPROP</td><td> $\alpha \in [ 1 0 ^ { - 4 } , 1 0 ^ { - 1 } ] , 1 - \eta _ { - } \in [ 0 . 0 1 , 0 . 6 ] ,$   $\eta _ { + } - 1 \in [ 0 . 0 0 5 , 0 . 5 ]$ </td><td> $\alpha = 9 . 7 2 7 \times 1 0 ^ { - 3 } , \eta _ { - } = 0 . 7 1 3 1 , \eta _ { + } = 1 . 1 7 9$ </td></tr><tr><td>Adam</td><td> $\stackrel { \cdot \cdot } { \alpha } \in [ 1 0 ^ { - 4 } , 1 0 _ { . } ^ { - 1 } ] , 1 _ { - } ^ { - } \beta _ { 1 } \in [ 1 0 ^ { - 3 } , 0 . 3 ] ,$   $1 - \dot { \beta } _ { 2 } \in [ 1 0 ^ { - 4 } , \dot { 0 } . 1 ]$ </td><td> $\alpha = 1 . 8 8 8 \times 1 0 ^ { - 3 } , \beta _ { 1 } = 0 . 9 8 4 3 , \beta _ { 2 } = 0 . 9 9 9 8$ </td></tr><tr><td>Adam (variable LR)</td><td>as Adam</td><td> $\alpha = 3 . 9 6 8 \times 1 0 ^ { - 3 } , \beta _ { 1 } = 0 . 9 8 5 1 , \beta _ { 2 } = 0 . 9 2 2 9$ </td></tr><tr><td>NadamW</td><td>as Adam, and weight decay  $\in [ 1 0 ^ { - 5 } , 0 . 1 ]$ </td><td> $\alpha = 1 . 2 4 9 \times 1 0 ^ { - 3 } , \beta _ { 1 } = 0 . 9 9 6 9 , \beta _ { 2 } = 0 . 9 7 9 5 ,$  weight  $\mathrm { d e c a y ~ 3 . 4 3 7 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>CvAMSGrad</td><td>as Adam</td><td> $\alpha = \overset { \cdot } { 3 . 4 0 3 } \times 1 0 ^ { - 3 } , \beta _ { 1 } = 0 . 9 8 1 8 , \beta _ { 2 } = 0 . 9 9 8$ </td></tr><tr><td>Muon</td><td> $\alpha \in [ 1 0 ^ { - 4 } , 1 0 ^ { - 1 } ] , 1 - \beta \in [ 1 0 ^ { - 3 } , 0 . 3 ]$ </td><td> $\alpha = 2 . 8 9 8 \times 1 0 ^ { - 4 } , \beta = 0 . 9 9 6 2$ </td></tr><tr><td>Adam-AURA</td><td> $\alpha \in \lbrack 1 0 ^ { - 4 } , 1 0 ^ { - 1 } \rbrack , 1 - \beta _ { \zeta } \in \mathsf { \bar { \Gamma } } [ 0 . 0 1 , 0 . 6 \bigr ] , \chi _ { \mathrm { a } } \in [ 0 . 5 , 0 . 9 8 ] ,$   $\chi _ { \mathrm { a } } - \dot { \chi } _ { \mathrm { o } } \in [ 0 . 0 2 , 1 . 2 ] , \Psi _ { \mathrm { o } } \in [ 0 . 0 1 , 0 . \dot { 9 } ] ,$ </td><td> $\alpha = 1 . 1 7 1 \times 1 0 ^ { - 3 } , \beta _ { \zeta } = 0 . 9 7 3 , \chi _ { \mathrm { a } } = 0 . 7 3 8 3 ,$   $\chi _ { \mathrm { o } } = - 0 . 4 2 1 7 , \Psi _ { \mathrm { a } } = 0 . 0 2 4 2 2 , \Psi _ { \mathrm { o } } = 0 . 0 3 0 5 1$ </td></tr><tr><td>Muon-AURA</td><td> $\Psi _ { \mathrm { a } } / \Psi _ { \mathrm { o } } \in [ 0 . 0 0 5 , 0 . 9 ] ; \beta _ { 1 } , \beta _ { 2 }$  as selected for Adam as Muon and as Adam-AURA</td><td> $\alpha = 1 . 8 1 6 \times 1 0 ^ { - 3 } , \beta = 0 . 9 7 3 2 , \beta _ { \zeta } = 0 . 8 0 1 6 ,$   $\chi _ { \mathrm { a } } = 0 . 6 7 7 , \chi _ { \mathrm { o } } = - 0 . 0 4 1 6 , \Psi _ { \mathrm { a } } = 0 . 0 1 6 7 ,$   $\Psi _ { \mathrm { o } } = 0 . 1 1 9 1$ </td></tr></table>

e) Results: Table XII reports the three metrics at the tuned hyperparameters, together with the number of seeds whose training loss becomes non-finite (NaN), and Figure 5 shows the training- and test-loss curves at the tuned (bottom) hyperparameters. For completeness, the same figure also shows the results at the non-tuned hyperparameters, that is, those in Table II, to better illustrate the effect of the hyperparameter optimization. Tuning benefits every method, the baselines more than the AURA variants: it lowers $A ^ { ( m ) }$ by 0.2 to 1.8 for the baselines and by 0.5 and 0.8 for Adam-AURA and Muon-AURA, respectively. The overall conclusion is unchanged: Muon-AURA and Adam-AURA attain the two lowest values of both $\mathcal { L } _ { \operatorname* { m i n } } ^ { ( m ) }$ and $A ^ { ( m ) }$ ; the minimum loss of Muon-AURA is about 3.5 times below that of the best baseline, NadamW, and its margin of 1.3 in $A ^ { ( m ) }$ corresponds to a geometricmean training loss about 20 times lower; the minimum loss of Adam-AURA is about 1.7 times below that of NadamW and its margin of 0.8 in $A ^ { ( m ) }$ to a loss about six times lower. The largest effect of AURA is on Muon: alone, it reaches the highest minimum loss of all methods, and its combination with AURA lowers it by more than two orders of magnitude. At their tuned values, which include a step size four to ten times larger than the starting value of the search, RPROP and Adam (variable LR) diverge on one of the five evaluation seeds, although neither diverged on the three search seeds; every other method converges on every seed. In this full-batch setting, the methods whose step sizes follow the agreement between consecutive steps, and which diverged at small minibatches in Section IV-D, rank first and second in $A ^ { ( m ) } \ { \mathrm { ( A U R A ) } }$ or within 0.2 of the best baseline (RPROP), consistent with the interpretation given there.

TABLE XII  
TRAINING LOSS AND TRAINING TIME FOR THE PINN BENCHMARK AT THE TUNED HYPERPARAMETERS OF TABLE XI.
<table><tr><td>Optimizer</td><td> $\mathcal { L } _ { \operatorname* { m i n } } ^ { ( m ) }$ </td><td> $A ^ { ( m ) }$ </td><td>Training time (×SGD) NaN</td><td></td></tr><tr><td>RPROP</td><td> $( 4 . 3 _ { - 2 . 0 } ^ { + 2 . 5 } ) \times 1 0 ^ { - 5 }$ </td><td> $5 . 9 _ { - 0 . 2 } ^ { + 0 . 1 }$ </td><td> $1 . 2 _ { - 0 . 1 } ^ { + 0 . 1 }$ </td><td>1</td></tr><tr><td>Adam</td><td> $( 1 . 0 _ { - 0 . 1 } ^ { + 0 . 0 2 } ) \times 1 0 ^ { - 5 }$ </td><td> $6 . 2 6 _ { - 0 . 0 1 } ^ { + 0 . 0 1 }$ </td><td> $\mathbf { 1 . 2 1 } _ { - \mathbf { n } \ \mathbf { \hat { n } } \mathbf { 0 } \mathbf { 3 } } ^ { + \mathbf { 0 . 0 3 } }$ </td><td>0</td></tr><tr><td>Adam (variable LR)</td><td> $( 1 . 8 _ { - 0 . 4 } ^ { + 0 . 5 } ) \times 1 0 ^ { - 5 }$ </td><td> $6 . 3 _ { - 0 \mathrm { ~ 2 ~ } } ^ { + 0 . 1 }$ </td><td> $1 . 2 5 4 _ { - 0 . 0 0 9 } ^ { + 0 . 0 0 3 }$ </td><td>1</td></tr><tr><td>NadamW</td><td> $( 2 . 2 _ { - 0 . 0 3 } ^ { + 0 . 2 } ) \times 1 0 ^ { - 6 }$ </td><td> $5 . 7 6 _ { - 0 . 0 2 } ^ { + \ddot { 0 } . \mp 0 2 }$ </td><td> $1 . 2 4 9 _ { - 0 . 0 0 6 } ^ { + 0 . 0 0 1 }$ </td><td>0</td></tr><tr><td>CvAMSGrad</td><td> $( 5 . 7 _ { - 1 . 2 } ^ { + 1 . 5 } ) \times 1 0 ^ { - 6 }$ </td><td> $6 . 0 _ { - 0 . 1 } ^ { + 0 . 2 }$ </td><td> $1 . 3 7 _ { - 0 . 0 9 } ^ { + 0 . { \overset { \cdot } { 0 } } { 2 } }$ </td><td>0</td></tr><tr><td>Muon</td><td> $( 8 . 0 _ { - 1 . 1 } ^ { + \bar { 0 } . \bar { 9 } } ) \times 1 0 ^ { - 5 }$ </td><td> $6 . 6 0 _ { - 0 . 0 0 5 } ^ { + 0 . 0 5 }$ </td><td> $1 . 4 7 _ { - \Omega \ \cap 5 } ^ { + 0 . 7 \bar { 0 } \bar { 0 } \bar { 0 } 4 }$ </td><td>0</td></tr><tr><td>Adam-AURA</td><td> $( 1 . 3 _ { - 0 . 1 } ^ { + \bar { 0 } . \bar { 4 } } ) \times 1 0 ^ { - 6 }$ </td><td> $4 . 9 6 _ { - 0 . 0 5 } ^ { + 0 . 0 5 }$ </td><td> $1 . 3 9 _ { - 0 . 0 3 } ^ { + 0 . 0 4 }$ </td><td>0</td></tr><tr><td>Muon-AURA</td><td> $( { \bf 6 . 3 _ { - 0 . 7 } ^ { + 1 . 6 } } ) \times { \bf 1 0 ^ { - 7 } }$ </td><td> $\mathbf { 4 . 4 6 _ { - 0 . 1 2 } ^ { + 0 . 0 1 } }$ </td><td> $2 . 0 9 _ { - 0 . 0 1 } ^ { + 0 . 0 1 }$ </td><td>0</td></tr></table>

![](images/7badf439ec28d3c4d7321fcf1d1a99952c2fe7b7a1fc14b168ab19039d50c21a.jpg)  
Fig. 5. Training-set and test-set mean squared errors (boundary loss (18)) for the PINN optimizer benchmark. The two rows show the same network trained at the default hyperparameters of Table II with $\alpha = 1 0 ^ { - 3 }$ (top) and at the tuned hyperparameters of Table XI (bottom). Solid curves show the pointwise median over all 5 seeds; shaded regions span the interquartile range (25th–75th percentile) over seeds, whose edges are also drawn as dotted curves.

## V. CONCLUSIONS

We have proposed AURA, a per-parameter step-size multiplier for training CVNNs. AURA acts on top of any firstorder optimizer and leaves its update direction unchanged. It compares consecutive update directions of each complex parameter through a Dice-like measure, whose real part quantifies their alignment and whose imaginary part their signed rotation. The step is enlarged when the directions agree in length, alignment, and sense of rotation, and reduced when they do not. No additional gradient evaluation is required. We do not claim that the results achieved are optimal. Rather, the presented findings indicate that AURA can be adopted to improve the results of its base optimizer.

We combined AURA with Adam and Muon and compared the two variants with RPROP, Adam, NadamW, CvAMSGrad, and Muon on four test cases with fully connected CVNNs, from the approximation of scalar complex functions to a physicsinformed problem. The hyperparameters of AURA were held fixed across cases; in Section IV-F, every optimizer was tuned under the same budget. Muon-AURA attained the lowest minimum loss and the lowest area under the log learning curve in every setting with mini-batches of at least 128 points and in the full-batch case, at both fixed and tuned hyperparameters.

Adam-AURA ranked second in the area metric in most settings. AURA also reduced the sensitivity to the step size: over a 100- fold range of α, the minimum loss of Adam-AURA varied by less than an order of magnitude, against more than two for Adam. Tuning benefited the baselines more than the AURA variants, which nonetheless remained the two best methods. In the mini-batch cases, AURA added at most 0.15 to the time ratio of its base optimizer.

From the obtained results, we can conclude that AURA fails when the gradient is noisy. At the smallest mini-batch of Section IV-D, both variants diverged in every seed, together with RPROP. The cause lies in the agreement signal itself: AURA inherits no estimate of the gradient variance and measures the agreement between noisy directions. The fullbatch case of Section IV-F, where the two variants rank first and second, is consistent with this interpretation.

AURA has three further limitations. It stores five real numbers per parameter in addition to the state of the base optimizer, against the three of Adam. It adds a few elementwise operations per step. It introduces ten hyperparameters, of which the decay rate $\beta _ { \zeta }$ and the four thresholds are the least intuitive to set. On the other hand, the relatively high number of hyperparameters can be an advantage as it represents a

margin for adaptation.

Several developments follow from these results. The failure mode calls for a signal-to-noise estimate in the gate, so that the multiplier is not increased on the agreement of noisy directions. The current real-valued step size can be naturally extended to a complex step size, which rotates the update. For real parameters, $\zeta _ { j , t }$ is real and $\Psi _ { j , t }$ vanishes, so AURA reduces to a Dice-like test of sign and magnitude; whether the gains carry over to real-valued neural networks is an open question. Convolutional networks were excluded to delimit the scope of this work and are the natural next test. Finally, for clear computational cost reasons, we tested the proposed method on relatively small networks; whether the results extend to networks with billions of parameters, where the memory overhead matters most, remains to be assessed.

## DATA AND SOFTWARE AVAILABILITY

The code that reproduces all numerical experiments of Section IV, including the scripts that regenerate the tables and figures, is available at https://github.com/enricoballini/aura optimizer tests.git. The setup has been tested on Ubuntu 24.04.

AURA is distributed as the open-source Python package aura-optax (MIT license), which can be installed with pip install aura-optax; it requires Python 3.10 or later and depends only on JAX and Optax.

The package provides adam\_aura and muon\_aura, which implement Adam-AURA and Muon-AURA, and scale\_by\_aura, which implements the AURA step alone and can be chained after any other update rule.

For complex-valued parameters, the gradients returned by jax.grad must be conjugated before being passed to update, as for every Optax optimizer.

## ACKNOWLEDGMENT

This work was supported by the Danish Research Council for Independent Research through the grant no. 2035-00142B “Network-inspired models to predict the strength of heterogeneous materials”. The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## REFERENCES

[1] L. Bottou, F. E. Curtis, and J. Nocedal, “Optimization methods for largescale machine learning,” SIAM Review, vol. 60, no. 2, pp. 223–311.

[2] J. Zhuang, T. Tang, Y. Ding, S. C. Tatikonda, N. Dvornek, X. Papademetris, and J. Duncan, “AdaBelief optimizer: Adapting stepsizes by the belief in observed gradients,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 18 795– 18 806.

[3] S. K. Roy, M. E. Paoletti, J. M. Haut, S. R. Dubey, P. Kar, A. Plaza, and B. B. Chaudhuri, “AngularGrad: A new optimization technique for angular convergence of convolutional neural networks,” arXiv:2105.10190, 2021.

[4] S. R. Dubey, S. Chakraborty, S. K. Roy, S. Mukherjee, S. K. Singh, and B. B. Chaudhuri, “diffGrad: An optimization method for convolutional neural networks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 11, pp. 4500–4511, 2020.

[5] J. Lu, AdaSmooth: An Adaptive Learning Rate Method Based on Effective Ratio. Springer Nature Singapore, 2023, pp. 273–293.

[6] A. G. Baydin, R. Cornish, D. M. Rubio, M. Schmidt, and F. Wood, “Online learning rate adaptation with hypergradient descent,” in International Conference on Learning Representations, 2018. [Online]. Available: https://openreview.net/forum?id=BkrsAzWAb

[7] S. L. Goh and D. Mandic, “A class of gradient-adaptive step size algorithms for complex-valued nonlinear neural adaptive filters,” in Proceedings. (ICASSP ’05). IEEE International Conference on Acoustics, Speech, and Signal Processing, 2005., vol. 5. IEEE, 2005, pp. 253–256.

[8] S. L. Goh and D. P. Mandic, “Stochastic gradient-adaptive complexvalued nonlinear neural adaptive filters with a gradient-adaptive step size,” IEEE Transactions on Neural Networks, vol. 18, no. 5, pp. 1511–1516, 2007.

[9] J. Duchi, E. Hazan, and Y. Singer, “Adaptive subgradient methods for online learning and stochastic optimization,” Journal of Machine Learning Research, vol. 12, no. 61, pp. 2121–2159, 2011. [Online]. Available: http://jmlr.org/papers/v12/duchi11a.html

[10] M. D. Zeiler, “ADADELTA: An adaptive learning rate method,” arXiv:1212.5701, 2012.

[11] M. Eckhoff and M. Reiher, “Lifelong machine learning potentials,” Journal of Chemical Theory and Computation, vol. 19, no. 12, pp. 3509– 3525, 2023.

[12] ——, “CoRe optimizer: an all-in-one solution for machine learning,” Machine Learning: Science and Technology, vol. 5, no. 1, p. 015018, 2024.

[13] F. Schneider, L. Balles, and P. Hennig, “DeepOBS: A deep learning optimizer benchmark suite,” in International Conference on Learning Representations, 2019. [Online]. Available: https: //openreview.net/forum?id=rJg6ssC5Y7

[14] R. M. Schmidt, F. Schneider, and P. Hennig, “Descending through a crowded valley — benchmarking deep learning optimizers,” 2021. [Online]. Available: https://openreview.net/forum?id=k2Om84I9JuX

[15] D. Choi, C. J. Shallue, Z. Nado, J. Lee, C. J. Maddison, and G. E. Dahl, “On empirical comparisons of optimizers for deep learning,” 2020. [Online]. Available: https://openreview.net/forum?id=HygrAR4tPS

[16] E. Kiyani, K. Shukla, J. F. Urban, J. Darbon, and G. E. Karniadakis,´ “Optimizing the optimizer for physics-informed neural networks and kolmogorov-arnold networks,” Computer Methods in Applied Mechanics and Engineering, vol. 446, p. 118308, 2025.

[17] K. S. Mayer, J. A. Soares, A. A. Cruz, and D. S. Arantes, “Adaptive learning rate methods for complex-valued neural networks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 12, pp. 20 157–20 170, 2025.

[18] M. Rutkowski and P. A. Kowalski, “The impact of optimiser choice on the training dynamics of complex-valued neural networks: A comparative study with real-valued counterparts,” International Journal of Approximate Reasoning, vol. 197, p. 109722, 2026.

[19] C. Trabelsi, O. Bilaniuk, Y. Zhang, D. Serdyuk, S. Subramanian, J. F. Santos, S. Mehri, N. Rostamzadeh, Y. Bengio, and C. J. Pal, “Deep complex networks,” in International Conference on Learning Representations, 2018. [Online]. Available: https://openreview.net/forum? id=H1T2hmZAb

[20] C. Lee, H. Hasegawa, and S. Gao, “Complex-valued neural networks: A comprehensive survey,” IEEE/CAA Journal of Automatica Sinica, vol. 9, no. 8, pp. 1406–1426, 2022.

[21] E. Cole, J. Cheng, J. Pauly, and S. Vasanawala, “Analysis of deep complex-valued convolutional neural networks for mri reconstruction and phase-focused applications,” Magnetic Resonance in Medicine, vol. 86, no. 2, pp. 1093–1109, 2021.

[22] M. K. Almansoori and M. Telek, “Performance evaluation of complexvalued neural networks on real and complex-valued classification and reconstruction tasks,” Machine Learning with Applications, vol. 22, p. 100742, 2025.

[23] R. Savitha, S. Suresh, and N. Sundararajan, “Metacognitive learning in a fully complex-valued radial basis function neural network,” Neural Computation, vol. 24, no. 5, pp. 1297–1328, 2012.

[24] B. Zhang, Y. Liu, J. Cao, S. Wu, and J. Wang, “Fully complex conjugate gradient-based neural networks using wirtinger calculus framework: Deterministic convergence and its application,” Neural Networks, vol. 115, pp. 50–64, Jul. 2019.

[25] Z. Dong and H. Huang, “A training algorithm with selectable search direction for complex-valued feedforward neural networks,” Neural Networks, vol. 137, pp. 75–84, 2021.

[26] Y. Zhang, Q. Hua, D. Xu, H. Li, and H. Mu, “A complex-valued convolutional neural network with different activation functions in polarimetric sar image classification,” in 2019 International Radar Conference (RADAR). IEEE, 2019, pp. 1–4.

[27] G. Hinton, N. Srivastava, and K. Swersky, “Neural networks for machine learning, lecture 6e: rmsprop: Divide the gradient by a running average of its recent magnitude,” 2012, lecture slides. [Online]. Available: https: //www.cs.toronto.edu/∼tijmen/csc321/slides/lecture slides lec6.pdf

[28] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv:1412.6980, 2014.

[29] S. J. Reddi, S. Kale, and S. Kumar, “On the convergence of Adam and beyond,” in International Conference on Learning Representations, 2018.

[30] Q. Tong, G. Liang, and J. Bi, “Calibrating the adaptive learning rate to improve convergence of adam,” Neurocomputing, vol. 481, pp. 333–356, 2022.

[31] T. Dozat, “Incorporating Nesterov momentum into Adam,” in International Conference on Learning Representations (ICLR), Workshop Track, 2016. [Online]. Available: https://openreview.net/pdf? id=OM0jvwB8jIp57ZJjtNEZ

[32] H. Zhang and D. P. Mandic, “Is a complex-valued stepsize advantageous in complex-valued gradient learning algorithms?” IEEE Transactions on Neural Networks and Learning Systems, vol. 27, no. 12, pp. 2730–2735, 2016.

[33] Y. Zhang and H. Huang, “Adaptive complex-valued stepsize based fast learning of complex-valued neural networks,” Neural Networks, vol. 124, pp. 233–242, 2020.

[34] W. Zhao and H. Huang, “Adaptive orthogonal gradient descent algorithm for fully complex-valued neural networks,” Neurocomputing, vol. 546, p. 126358, Aug. 2023.

[35] X. Yin and H. Huang, “Flatness guided adaptive gradient descent algorithm for fully complex-valued neural networks,” in 2026 38th Chinese Control and Decision Conference (CCDC). IEEE, 2026, pp. 5944–5949.

[36] K. Zhang, H. Zhang, and X. Wang, “A hybrid complex spectral conjugate gradient learning algorithm for complex-valued data processing,” Engineering Applications of Artificial Intelligence, vol. 133, p. 108352, Jul. 2024.

[37] K. Zhang and H. Zhang, “An adaptive complex-valued stepsize training scheme for complex-valued neural networks,” Journal of Applied Mathematics and Computing, vol. 72, no. 1, 2025.

[38] W. Zhao and H. Huang, “Adaptive stepsize estimation based accelerated gradient descent algorithm for fully complex-valued neural networks,” Expert Systems with Applications, vol. 236, p. 121166, 2024.

[39] Y. Zhang, H. Huang, and G. Shen, “Adaptive cl-bfgs algorithms for complex-valued neural networks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 9, pp. 6313–6327, 2023.

[40] Y. Wang, Z. Wang, and H. Huang, “Stochastic adaptive cl-bfgs algorithms for fully complex-valued dendritic neuron model,” Knowledge-Based Systems, vol. 277, p. 110788, 2023.

[41] K. Jordan, Y. Jin, V. Boza, Y. Jiacheng, F. Cesista, L. Newhouse, and J. Bernstein, “Muon: An optimizer for hidden layers in neural networks,” 2024. [Online]. Available: https://kellerjordan.github.io/posts/muon/

[42] L. Egghe and L. Leydesdorff, “The relation between pearson’s correlation coefficient r and salton’s cosine measure,” Journal of the American Society for Information Science and Technology, vol. 60, no. 5, pp. 1027–1036, 2009.

[43] R. A. Jacobs, “Increased rates of convergence through learning rate adaptation,” Neural Networks, vol. 1, no. 4, pp. 295–307, 1988.

[44] M. Riedmiller and H. Braun, “A direct adaptive method for faster backpropagation learning: the rprop algorithm,” in IEEE International Conference on Neural Networks. IEEE, pp. 586–591.

[45] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7

[46] L. Balles and P. Hennig, “Dissecting adam: The sign, magnitude and variance of stochastic gradients,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, J. Dy and A. Krause, Eds., vol. 80. PMLR, 10–15 Jul 2018, pp. 404–413. [Online]. Available: https://proceedings.mlr.press/v80/balles18a.html

[47] H. Kesten, “Accelerated stochastic approximation,” The Annals of Mathematical Statistics, vol. 29, no. 1, pp. 41–59, 1958.

[48] G. Saridis, “Learning applied to successive approximation algorithms,” IEEE Transactions on Systems Science and Cybernetics, vol. 6, no. 2, pp. 97–103, 1970.

[49] A. G. Barto and R. S. Sutton, “Goal seeking components for adaptive intelligence: An initial assessment,” Air Force Wright Aeronautical Laboratories, Avionics Laboratory, Wright-Patterson Air Force Base, Ohio, Tech. Rep. AFWAL-TR-81-1070, Apr. 1981, technical Report.

[50] M. Riedmiller, “Advanced supervised learning in multi-layer perceptrons — from backpropagation to adaptive learning algorithms,” Computer Standards & Interfaces, vol. 16, no. 3, pp. 265–278, 1994.

[51] A. Kantsila, M. Lehtokangas, and J. Saarinen, “Complex rprop-algorithm for neural network equalization of gsm data bursts,” Neurocomputing, vol. 61, pp. 339–360, 2004.

[52] C.-A. Popa, “Enhanced gradient descent algorithms for complex-valued neural networks,” in 2014 16th International Symposium on Symbolic and Numeric Algorithms for Scientific Computing. IEEE, 2014, pp. 272–279.

[53] J. Liu, C. Lin, C. Li, L. Sheng, M. Sun, J. Yan, and W. Ouyang, “Adaptive gradient method with resilience and momentum,” arXiv:2010.11041, 2020.

[54] P. Malviya, G. Mordido, A. Baratin, R. B. Harikandeh, G. K. Dziugaite, R. Pascanu, and S. Chandar, “Torque-aware momentum,” arXiv:2412.18790, 2024.

[55] N. McGreivy and A. Hakim, “Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations,” Nature Machine Intelligence, vol. 6, no. 10, pp. 1256–1269, Sep. 2024.

[56] X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,” Journal of Machine Learning Research, vol. 9, pp. 249–256, 2010.

[57] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, Y. Katariya, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang, “JAX: composable transformations of Python+NumPy programs,” 2018, open-source software.

[58] M. Calafa, E. Hovad, A. P. Engsig-Karup, and T. Andriollo, “Physics-\` informed holomorphic neural networks (PIHNNs): Solving 2D linear elasticity problems,” Computer Methods in Applied Mechanics and Engineering, vol. 432, p. 117406, Dec. 2024.

[59] E. Ballini, A. P. Engsig-Karup, and T. Andriollo, “A holomorphic neural network framework for 3d boundary value problems governed by harmonic potentials,” Computer Methods in Applied Mechanics and Engineering, Accepted, arXiv:2605.31231, 2026.

[60] R. Savitha, S. Suresh, N. Sundararajan, and P. Saratchandran, “A new learning algorithm with logarithmic performance index for complexvalued neural networks,” Neurocomputing, vol. 72, no. 16-18, pp. 3771– 3781, 2009.

[61] R. Savitha, S. Suresh, and N. Sundararajan, “A self-regulated learning in fully complex-valued radial basis function networks,” in The 2010 International Joint Conference on Neural Networks (IJCNN). IEEE, 2010, pp. 1–8.

[62] M. F. Amin, R. Savitha, M. I. Amin, and K. Murase, “Complex-valued functional link network design by orthogonal least squares method for function approximation problems,” in The 2011 International Joint Conference on Neural Networks. IEEE, 2011, pp. 1489–1496.

[63] R. Savitha, S. Suresh, and N. Sundararajan, “A meta-cognitive learning algorithm for a fully complex-valued relaxation network,” Neural Networks, vol. 32, pp. 209–218, 2012.

[64] J. Bergstra, R. Bardenet, Y. Bengio, and B. Kegl, “Algorithms for´ hyper-parameter optimization,” in Advances in Neural Information Processing Systems, J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K. Weinberger, Eds., vol. 24. Curran Associates, Inc., 2011. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf

[65] Y. Ozaki, S. Watanabe, and T. Yanase, “Optunahub: A platform for black-box optimization,” Journal of Machine Learning Research, vol. 27, no. 203, pp. 1–10, 2026. [Online]. Available: http: //jmlr.org/papers/v27/25-2424.html

## APPENDIX

## SALTON-LIKE VERSUS DICE-LIKE MEASURE

The numerator of $\zeta _ { j , t }$ in (2b) may be normalized either by the geometric mean of $| \bar { d } _ { j , t } | ^ { 2 }$ and $| d _ { j , t - 1 } | ^ { 2 }$ , that is, $| d _ { j , t } | | d _ { j , t - 1 } |$ which gives the Salton-like measure $\zeta _ { j , t } ^ { \mathrm { S a l } }$ , or by their arithmetic mean, which gives the Dice-like measure used here. Their real parts are Salton’s cosine and Dice’s measure of the associated $\bar { \mathbb { R } } ^ { 2 }$ vectors; their imaginary parts have no counterpart in those measures, hence the suffix $\ddot { \mathbf { \omega } } _ { - } \vert \mathrm { i k e } ^ { \mathbf { \gamma } }$ . Writing $r _ { j , t } = | d _ { j , t } | / | d _ { j , t - 1 } |$ for the ratio of consecutive step lengths, the two differ by

$$
\frac { | \zeta _ { j , t } | } { | \zeta _ { j , t } ^ { \mathrm { S a l } } | } = \frac { 2 | d _ { j , t } | | d _ { j , t - 1 } | } { | d _ { j , t } | ^ { 2 } + | d _ { j , t - 1 } | ^ { 2 } } = \frac { 2 r _ { j , t } } { 1 + r _ { j , t } ^ { 2 } } \ \in ( 0 , 1 ] ,\tag{19}
$$

which equals 1 only for $r _ { j , t } = 1$ and decays as the two lengths separate, see Figure 6 and [42]. The Dice-like measure therefore damps $\zeta _ { j , t }$ whenever consecutive directions differ markedly in magnitude, which is the desired behavior, as it highlights updates that differ in magnitude, and is used for this reason.

![](images/cc2427c8abf89a1c5777fe140147946669283a427c5124bf9695822694dc3fe1.jpg)  
Fig. 6. Dice-like against Salton-like measure as a function of the ratio $r \bar { = } | d _ { j , t } | / | d _ { j , t - 1 } |$ of consecutive step lengths, see (19). The Salton-like measure has unit modulus for every $^ { r , }$ whereas the Dice-like one decays as the two lengths separate.