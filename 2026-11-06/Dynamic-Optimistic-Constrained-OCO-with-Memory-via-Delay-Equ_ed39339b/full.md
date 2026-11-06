# Dynamic Optimistic Constrained OCO with Memory via Delay Equivalence

Mohammed Abdullah L2S, CentraleSupélec Université Paris-Saclay, CNRS France mohammed.abdullah@centralesupelec.fr

Salah Eddine Elayoub L2S, CentraleSupélec Université Paris-Saclay, CNRS France salaheddine.elayoubi@centralesupelec.fr

George Iosifidis Delft University of Technology The Netherlands G.Iosifidis@tudelft.nl

Tijani Chahed Télécom SudParis Institut Polytechnique de Paris France tijani.chahed@telecom-sudparis.eu

## Abstract

Online Convex Optimization with memory and time-varying constraints (COCO-M) is a recently-introduced framework that captures the dynamics of stateful online learning and non-stochastic control with budget constraints. Despite its expressive power, COCO-M remains largely understudied. In this work, we tackle this problem in its full scale, providing an optimistic algorithm that incorporates untrusted predictions to tame universal dynamic regret and cumulative constraint violation. Our analysis leverages a potent reduction from memory to delay and treats state-dependent gradients as delayed feedback. We establish the first bounds that adapt to both environment non-stationarity (path length $P _ { T } )$ and prediction accuracy, while lifting previous restrictive assumptions such as functional separability. En route to these results, we derive findings of independent interest for the memory-to-delay reduction, and for delayed OCO with time-varying constraints. The proposed framework ensures sublinear worst-case guarantees that improve with prediction accuracy, reaching O(1) regret under perfect predictions, with cumulative constraint violation (CCV) $\mathcal { O } ( \sqrt { T } )$ for unknown $P _ { T }$ and ${ \mathcal { O } } ( \log T )$ for known or online observable $P _ { T }$ , while recovering the optimal COCO rates when memory disappears.

## 1 Introduction

Online Convex Optimization (OCO) is a powerful framework for sequential decision making: at each round $t = 1 , \dots , T$ , a learner selects $x _ { t } \in \mathcal { X }$ , an adversary reveals a convex loss function $f _ { t } : \mathcal { X } \to \mathbb { R }$ , and the learner incurs cost $f _ { t } ( x _ { t } )$ . Performance is measured via regret, $\mathcal { R } _ { T }$ , which compares the learner’s cumulative loss against a benchmark sequence $\{ u _ { t } \} _ { t }$ . Depending on the choice of benchmark, one obtains static regret against a fixed decision $\boldsymbol { u } _ { t } = \boldsymbol { x } ^ { \star } , \forall t ;$ worst-case dynamic regret against per-round minimizers, $u _ { t } = x _ { t } ^ { \star } = \arg \operatorname* { m i n } _ { x \in \mathcal { X } } f _ { t } ( x )$ , ∀t; or the more general universal dynamic regret, which allows comparison to any sequence $\{ u _ { t } \} _ { t }$ . Over the years, OCO has been extended to a wide range of settings with many practical applications [1, 2].

A key extension is OCO with memory (OCO-M), where the loss at time t depends on a window of past decisions $x _ { t - m } , \ldots , x _ { t }$ . This formulation arises naturally in applications such as recommender systems, portfolio optimization, and continual learning [3]. Importantly, OCO-M underlies the framework of non-stochastic control (NSC), which optimizes stateful systems under adversarial disturbances [4]. In many such problems, the learner must also satisfy time-average constraints of adversarially-varying functions $g _ { t } ( x _ { t } ) \leq 0$ . In this constrained OCO (COCO) setting, the goal is to control both regret and cumulative constraint violation $\begin{array} { r } { ( \mathrm { C C V } ) , Q _ { T } \doteq \sum _ { t } \operatorname* { m a x } \bar { \langle } g _ { t } \big ( x _ { t } \bar { \big ) , 0 \rangle } } \end{array}$ In practice, these constraints often exhibit memory as well, depending jointly on past decisions. Following [5], we refer to this setting as COCO-M.

Despite its many applications in stateful online decision problems and NSC with time-average constraints, COCO-M remains largely unexplored. This gap is not due to a lack of attention, but reflects inherent technical barriers. Indeed, even simplified versions of this problem challenge existing OCO methods: memory induces delay between decisions and their fully-observed feedback, time-varying constraints interact with long-term feasibility requirements, and dynamic benchmarks introduce additional non-stationarity. As a result, current techniques are insufficient to address COCO-M in its full generality, motivating the development of new methodological approaches.

Prior Work. We review briefly here the main lines of pertinent work and refer the reader to Appendix A for a comprehensive discussion. While COCO attracts growing attention, see [6] and references therein, only few works consider dynamic regret. Under this benchmark, the learner must additionally adapt to the variability of comparator, measured by its path length $\scriptstyle P _ { T } = \sum _ { t } \| u _ { t } - u _ { t - 1 } \|$ [7] established worst-case dynamic regret $\mathcal { O } ( \sqrt { ( 1 + P _ { T } ) T } )$ and $Q _ { T } = \mathcal { O } ( \sqrt { T } )$ for time-invariant constraints. For time-varying adversarial constraints, [8] obtained $\mathcal { O } ( \sqrt { ( 1 + P _ { T } ) T } )$ ) universal dynamic regret and $Q _ { T } = \mathcal { O } ( T ^ { 3 / 4 } )$ . Subsequent works [9, 10] refined these guarantees under different assumptions, and [11] provided the first bounds for universal dynamic regret and CCV. However, none of these important works considers memory. The few studies of dynamic regret in memorydependent problems are confined to the special case of switching costs (one-round memory) [12], [13], or to NSC without time-average constraints [14], [15], [16]. Thus, dynamic regret for COCO-M, and more broadly for constrained NSC, remains essentially open.

From a methodological standpoint, this work brings together memory, delay, and optimism. Existing static optimistic memory-to-delay reductions [5, 17] require separability across memory variables, which in COCO-M yields a restrictive benchmark and prevents a direct application to constrained NSC; we drop this assumption by introducing a more general memory-to-delay reduction. Learning with delay is typically treated with multiple interleaved learners, e.g., [18], while [19] unified delay and optimism by interpreting the delayed gradients as ”bad hints”. This line of work focuses on static regret, with few exceptions such as [20] which does not consider constraints. Finally, optimistic learning, i.e., incorporating potentially inaccurate predictions in OCO, has been studied with dynamic regret in [21] and [22], but without delay or memory effects. In summary, current methods do not provide the tools required to address even intermediate variants of the problem considered here.

Contributions. Motivated by this gap, we study COCO-M under universal dynamic regret while leveraging (possibly inaccurate) predictions of the losses and constraints. We propose a meta-learning framework that combines an adaptive hedge-based meta-learner with a collection of optimistic delayed Online Gradient Descent (OGD) base learners, each tuned to a different value of the (unknown) path length $P _ { T }$ . Both the meta-learner and the base learners are bespoke. Moreover, a key idea in our work is that learning with memory can be effectively mapped to learning with delay; we formalize this connection and employ it to study the targeted COCO-M problem. To the best of our knowledge, this is the first work to address this problem, and its contributions can be summarized as follows:

• Develops an optimistic delayed OCO algorithm and derives prediction-dependent universal dynamic regret guarantees (Theorem 1). This is the algorithm engine of the framework.

• Extends this framework to constrained OCO with time-varying constraints, obtaining regret and cumulative constraint violation bounds for unknown and known $P _ { T }$ (Theorems 2 and 3).

• Establishes a reduction from memory-dependent OCO to delayed OCO via partial gradient aggregation, without imposing function separability (Lemma 2). Building on this, it employs delayed OCO to deliver universal dynamic regret and CCV bounds for COCO-M (Corollary 1).

In all cases, our optimistic COCO-M bounds subsume the best-known results of simpler problem versions, e.g., without memory (or delay) and/or without predictions; and, when dropping the optimism, we obtain fully-adaptive COCO-M bounds, covering an important gap in literature. The details of proofs and intermediate results (some of independent interest) are included in Appendices.

## 2 Background & Preliminaries

Notation. Let $\mathcal { X } \subset \mathbb { R } ^ { d }$ be a nonempty, closed, bounded and convex set, with $\ell _ { 2 }$ -norm diameter $\begin{array} { r } { D \doteq \operatorname* { s u p } _ { x , y \in \mathcal { X } } \| x { - } y \| } \end{array}$ . We write $[ T ] \stackrel { - } { : = } \{ 1 , \ldots , T \}$ and, when needed, $[ T _ { m } ] : = \{ m , \dots , T \}$ , where m is the memory window, and use the shorthand notation:

$$
x _ { t - m } ^ { t } \ \doteq \left( x _ { t - m } , \ldots , x _ { t } \right) \in \boldsymbol { \mathcal { X } } ^ { m + 1 } , \quad x _ { a : b } = \sum _ { i = a } ^ { b } x _ { i } , \quad \mathrm { a n d } \quad z ^ { + } \doteq \operatorname* { m a x } \{ z , 0 \} , z \in \mathbb { R } .
$$

We denote with $\tilde { f } _ { t }$ the prediction of the function $f _ { t }$ and with $\nabla _ { x _ { s } } f _ { t } ( \cdot )$ the partial gradient (a subgradient) of $f _ { t }$ with respect to $x _ { s }$ . Also, $\Delta _ { N }$ is the N-dimensional unit simplex.

Metrics & Benchmarks. At each round $t \in [ T _ { m } ]$ , the learner decides $\boldsymbol { x } _ { t } \in \mathcal { X }$ and incurs loss $f _ { t } ( x _ { t - m } ^ { t } )$ and constraint violation $g _ { t } ( x _ { t - m } ^ { t } )$ , where $f _ { t } , g _ { t } : \mathcal { X } ^ { m + 1 } \mapsto \mathbb { R }$ . We compare the learner against a comparator sequence $u _ { 1 } ^ { T } \doteq ( u _ { 1 } , u _ { 2 } , \ldots , u _ { T } )$ that belongs to the feasible benchmark set:

$$
\mathcal { U } _ { T } : = \left\{ u _ { 1 } ^ { T } \in \mathcal { X } ^ { T } : g _ { t } \big ( u _ { t - m } ^ { t } \big ) \leq 0 , \forall t \in [ T _ { m } ] \right\} ,
$$

where we use the convention $x _ { 0 } = u _ { 0 } = 0$ , to streamline presentation. Observe that $\boldsymbol { \mathcal { U } } _ { T }$ is confined to satisfy the constraints on a per-round basis. This is necessary so as to avoid the impossibility result of [23] and is in line with prior COCO works (see Appendix Sec. $\mathbf { A } )$ . Still, $\boldsymbol { \mathcal { U } } _ { T }$ expands the static-regret COCO-M benchmark set in [5]; an advantage of our new memory-to-delay reduction in Sec. 5.

The learner is assessed using the metrics of dynamic regret and cumulative constraint violation<sup>1</sup>:

$$
\mathcal { R } _ { T } ( f ) \doteq \sum _ { t = m } ^ { T } f _ { t } ( x _ { t - m } ^ { t } ) - \sum _ { t = m } ^ { T } f _ { t } ( u _ { t - m } ^ { t } ) , \qquad \mathrm { Q _ { T } \doteq \sum _ { t = m } ^ { T } } g _ { t } ^ { + } ( x _ { t - m } ^ { t } ) .\tag{1}
$$

Our goal is to ensure both $\mathcal { R } _ { T } ( f )$ and $Q _ { T }$ are sublinear in $T$ and the comparator’s path length $\begin{array} { r } { P _ { T } \doteq \sum _ { t = 1 } ^ { T - 1 } \| u _ { t + 1 } - u _ { t } \| } \end{array}$ . Here we study universal dynamic regret, which is measured against all feasible comparator sequences [24]; this is a more general (and more challenging) metric [11].

Meta-learning with Hedge. Optimizing for dynamic regret requires knowledge of $P _ { T }$ or adaptation at runtime. A key idea here is to use a meta-learner that hedges against a set of $\begin{array} { r } { \bar { N } { = } \left\lceil \frac { 1 } { 2 } \bar { \log } _ { 2 } ( 1 + \bar { D T } ) \right\rceil { + } } \end{array}$ 1 base learners, that are implemented with different learning rates in a way that ensures at least one of them is tuned within a constant factor of $\sqrt { 1 + P _ { T } } \ [ 2 5 ]$ . Unfortunately, in our problem the hedging is compounded by the memory effects that introduce delay in discerning the best base learner.

Memory-to-Delay. In memory-less OCO, the loss gradient is available immediately; however, in OCO-M the impact of $x _ { t }$ is dispersed across the next m rounds; thus, memory induces feedback delay. To state this connection in a form that also applies to constrained problems, let $L _ { t } : \mathcal { X } ^ { m + 1 } \mapsto \mathbb { R }$ be a memory-dependent loss. In the unconstrained case, $L _ { t } = f _ { t }$ , while in the constrained case $L _ { t }$ is the penalized surrogate loss. We define the credit gradient $h _ { s }$ as the sum of partial gradients of all functions influenced by $\begin{array} { r } { x _ { s } , \mathrm { i . e . , } h _ { s } = \sum _ { t \in I _ { \ast } } \nabla _ { x _ { s } } \breve { L _ { t } } ( x _ { t - m } ^ { t } ) ^ { 2 } } \end{array}$ , where $I _ { s } = \{ t \in [ T _ { m } ] , \bar { t } - m \leq s \leq t \}$ Clearly, $h _ { s }$ cannot be computed until round $s + m$ , which reveals a feedback delay of m rounds in OCO-M. Using this observation (formalized in Sec. 5), we shift our focus from the per-round loss $L _ { t }$ to the per-decision credit $h _ { s }$ , i.e., from memory to a delayed feedback problem. A similar transformation, at the function level, was used by [5] and [17]; we discuss the differences in Sec. 5.

Optimism. Another key idea in this work is optimism, i.e., incorporating in learning (possibly inaccurate) predictions of future gradient vectors and/or actions [26]. Optimistic learning turns non-optimistic regret bounds of the form ${ \mathcal { O } } ( T ^ { k } ) .$ , for some $k \in [ 0 , 1 ]$ , to $\hat { \mathcal { O } } ( E _ { T } ^ { k } )$ , where $E _ { T }$ is the accumulated prediction error. These bounds hence range from $\mathcal { O } ( 1 )$ for accurate predictions, to the non-optimistic bound $\mathcal { O } ( T ^ { k } )$ in the worse case. This improvement happens, typically, through prediction errors-adaptive regularization, e.g., see [27, 28], and do not require prior information of the prediction accuracy. Finally, an important connection between optimism and feedback delay in OCO, which we also leverage here, was recently introduced in [19].

We rely on the following standard OCO assumptions throughout the paper.

Assumption 1. The decision set $\mathcal { X }$ is nonempty, closed, bounded, and convex, with $\mathbf { 0 } \in \mathcal { X }$ and diameter D.

Assumption 2. For every $t \in [ T _ { m } ]$ , thefunctions $f _ { t } , g _ { t } : \mathcal { X } ^ { m + 1 } \xrightarrow { } \mathbb { R }$ are convex in their arguments and uniformly bounded in absolute value by G.

Assumption 3. For every $t \in [ T _ { m } ]$ , the gradients of $f _ { t }$ and $g _ { t } ,$ , as well as their predicted counterparts, are uniformly bounded in norm by G.

Finally, we consider an oblivious adversary, in line with [3, 29, 30] and others. Hence, the loss and constraint functions are determined at $t = 0$ , but are not revealed before each round t is realized.

Remark 1. Unlike prior optimistic OCO/COCO-M works [5, 17], Assumptions 1-3 do not impose separability ofthe memory-dependentfunctions. Further, we do not impose Slater-type conditions

Roadmap. Given the identified memory-to-delay equivalence, we start in Section 3 by studying universal dynamic regret for the optimistic delayed OCO problem. In Section 4, we extend this analysis to the case of time-varying adversarial constraints (COCO with delay). Finally, Section 5 formalizes the memory-to-delay connection and shows how these results can be transferred to the targeted optimistic COCO-M problem.

## 3 Dynamic Optimistic OCO with Delay

We start with the problem of optimistic OCO with feedback delay. We use a decision-time convention that makes the delay explicit and facilitates making later the connection with memory. The decision $x _ { t }$ is played at time t, while its associated feedback vector $h _ { t }$ is fully revealed only after m rounds. Equivalently, at round $t ,$ the learner receives the delayed feedback $h _ { t - m }$ . Abusing slightly the notation (in order to facilitate presentation in Section 5), we redefine $L _ { t + m } : \mathcal { X } \mapsto \mathbb { R }$ to denote the convex loss at round $t + m$ which is associated (due to delay) with the earlier decision $x _ { t }$ . We also redefine $h _ { t } \in \nabla L _ { t + m } ( x _ { t } )$ . This notation aims to emphasize the delayed feedback. By convexity:

$$
L _ { t + m } ( x _ { t } ) - L _ { t + m } ( u _ { t } ) \leq \langle h _ { t } , x _ { t } - u _ { t } \rangle , \quad \forall u _ { 1 } ^ { T } \in \mathcal { X } ^ { T }
$$

Thus, we control the regret $\begin{array} { r } { \mathcal { R } _ { T } ( \ell ) : = \sum _ { t = 1 } ^ { T } \langle h _ { t } , x _ { t } - u _ { t } \rangle } \end{array}$ of the delayed linear losses $\ell _ { t } ( x ) : = \left. h _ { t } , x \right.$ We study the challenging case where the comparator path length $P _ { T }$ is unknown, and employ a meta-learning construction with $N$ base learners, similarly in spirit to [25]. At each round t, every base learner $i \in [ N ]$ proposes an action $\boldsymbol { x } _ { t } ^ { ( i ) } \in \mathcal { X }$ , and the meta-learner plays afused action:

$$
x _ { t } = X _ { t } w _ { t } , \qquad { \mathrm { w h e r e ~ } } X _ { t } : = [ x _ { t } ^ { ( 1 ) } , \dots , x _ { t } ^ { ( N ) } ] \qquad { \mathrm { a n d } } \quad w _ { t } \in \Delta _ { N } .
$$

For any fixed base learner $i \in [ N ]$ , the delayed linearized regret decomposes as:

$$
\mathcal { R } _ { T } ( \ell ) = \underbrace { \sum _ { t = 1 } ^ { T } \langle h _ { t } , x _ { t } - x _ { t } ^ { ( i ) } \rangle } _ { \mathrm { T e r m ~ ( I ) } } + \underbrace { \sum _ { t = 1 } ^ { T } \langle h _ { t } , x _ { t } ^ { ( i ) } - u _ { t } \rangle } _ { \mathrm { T e r m ~ ( I I ) } } .
$$

Since $x _ { t } = X _ { t } w _ { t }$ and $\boldsymbol { x } _ { t } ^ { ( i ) } = \boldsymbol { X } _ { t } \boldsymbol { e } _ { i }$ , where $e _ { i }$ is the i-th vertex of $\Delta _ { N }$ , Term (I) becomes:

$$
\mathrm { T e r m \left( I \right) } = \sum _ { t = 1 } ^ { T } \langle M _ { t } , w _ { t } - e _ { i } \rangle , \qquad \mathrm { w i t h } \qquad M _ { t } : = X _ { t } ^ { \top } h _ { t } .
$$

This way, we obtain a meta-level problem with losses $M _ { t }$ and a base-level problem with losses $h _ { t } ;$ and we treat both problems with delayed learning and optimism.

Unfortunately, standard dynamic-regret meta-learning schemes such as [25] do not handle delayed expert losses, while delayed/adaptive Hedge methods such as [2, 31] do not provide the optimistic prediction-dependent form needed here. Therefore, we build a bespoke meta-learning construction, leveraging and extending the delay-as-optimism framework of [19]: the missing delayed gradients are handled through predictions (or, hints) rather than worst-case gradient bounds; and we use a delayed-upper-bound (DUB) tuning to set the regularization from revealed quantities, thus avoiding over-regularization and preserving the benefit of predictions.

Let us clarify first the prediction structure which is essentially common for both problems. At each round t, after observing $h _ { t - m } .$ , the block $h _ { t - m + 1 } , \ldots , h _ { t }$ remains unavailable to the learner. To overcome this issue, the algorithm uses the prediction block $\widetilde { h } _ { t - m + 1 } ^ { t + 1 }$ , which includes both this missing information and the next hint $\widetilde { h } _ { t + 1 }$ . That is, at each round $t ,$ the learner needs to predict the gradient of $L _ { t + 1 } , \ldots , L _ { t + m + 1 }$ , and also the $x _ { t + 1 }$ , as it appears in the function of round $( t + m + 1 )$ . We emphasize that this is not the only possible way to construct the hint. As in prior optimistic works, predicted gradients may be evaluated instead at current decisions (which are known); see [19, 21, 26]. More generally, the algorithm does not require these quantities to be predicted separately: it only needs a bounded vector $\widetilde { h } _ { s }$ that serves as a prediction of $h _ { s }$

Now, focusing on Term (I), the meta-learner leverages delayed optimistic FTRL/AdaHedge to $M _ { t }$ and updates its fusing weights by executing:

$$
w _ { t + 1 } = \arg \operatorname* { m i n } _ { w \in \Delta _ { N } } \left\{ \left. w , M _ { 1 : t - m } + \widetilde M _ { t - m + 1 : t + 1 } \right. + \eta _ { t + 1 } ^ { \mathrm { M } } R _ { \Delta } ( w ) \right\} .\tag{2}
$$

where we defined $\widetilde { M } _ { t } = X _ { t } ^ { \top } \widetilde { h } _ { t }$ . Here we used an entropic regularizer $\begin{array} { r } { R _ { \Delta } ( w ) = \sum _ { i = 1 } ^ { N } } \end{array}$ w<sub>i</sub> log $w _ { i }$ , and the DUB regularization $\eta _ { t + 1 } ^ { \mathrm { M } } = \eta _ { t + 1 } ( \alpha _ { \mathrm { M } } )$ from [19], which employs the formula:

$$
\eta _ { t + 1 } ( \alpha ) = \frac { 2 } { \alpha } \operatorname* { m a x } _ { j \leq t - m - 1 } a _ { j - m + 1 : j } + \frac { 1 } { \alpha } \sqrt { \sum _ { r = 1 } ^ { t - m } ( a _ { r } ^ { 2 } + 2 \alpha b _ { r } ) } .\tag{3}
$$

Parameters $a _ { t } , b _ { t } \geq 0$ and $\alpha > 0$ are problem-specific, and for our meta-learning are defined as:

$$
\begin{array} { r l } & { a _ { t } ^ { \mathrm { M } } = 2 \Vert \bar { e } _ { t } ( M ) \Vert _ { \infty } , \quad b _ { t } ^ { \mathrm { M } } = \mathrm { h u b e r } ( \Vert \bar { e } _ { t } ( M ) \Vert _ { \infty } , \Vert \bar { n } _ { t } ( M ) \Vert _ { \infty } ) , \quad \alpha _ { \mathrm { M } } = \log N , \quad \mathrm { w h e r e } } \\ & { \bar { e } _ { t } ( M ) : = M _ { t - m : t } - \widetilde { M } _ { t - m : t } , \quad \bar { n } _ { t } ( M ) : = M _ { t - m } + \widetilde { M } _ { t - m + 1 : t + 1 } - \widetilde { M } _ { t - m : t } , \quad \mathrm { a n d } } \\ & { \mathrm { h u b e r } ( x , y ) : = \frac { 1 } { 2 } x ^ { 2 } - \frac { 1 } { 2 } ( x - y ) _ { + } ^ { 2 } \leq \operatorname* { m i n } \left\{ \frac { 1 } { 2 } x ^ { 2 } , x y \right\} . } \end{array}\tag{4}
$$

There are few interesting remarks here. First, the meta-learner operates on linear losses, so $w _ { t + 1 }$ itself does not need to be predicted. Second, before choosing $w _ { t + 1 } ,$ , the meta-learner already knows $X _ { t + 1 }$ and hence the only unknown quantity is the future feedback $h _ { t + 1 }$ . Finally, observe that we need to predict only the gradient vectors, not matrices $X _ { t }$ (see Lemma 3 in Appendix).

Turning our attention to Term (II), each base learner i is tuned to a path-length guess $\rho _ { i }$ , using the dyadic grid of possible values, as in [25]. However, here the base learners need also to handle the feedback delay and include the predictions. To enable this, we rely again on the toolbox of [19] which we modify for dynamic regret. This extension involves a non-trivial analysis due to the interaction of the prediction errors with the comparator variation (see Theorem 4 and Corol. 2 in Appendix); yet, interestingly, its implementation requires minimal changes. Namely, each base learner runs:

$$
x _ { t + 1 } ^ { ( i ) } = \arg \operatorname* { m i n } _ { x \in \mathcal { X } } \left\{ \left. h _ { t - m } + \widetilde { h } _ { t - m + 1 : t + 1 } - \widetilde { h } _ { t - m : t } , x \right. + \frac { \eta _ { t + 1 } ^ { i } } { 2 } \| x - x _ { t } ^ { ( i ) } \| _ { 2 } ^ { 2 } \right\} ,\tag{5}
$$

$$
\begin{array} { r } { \eta _ { t + 1 } ^ { i } = \eta _ { t + 1 } ( \alpha _ { i } ) \ \mathrm { w i t h } \ a _ { t } ^ { i } = D \| \bar { e } _ { t } ( h ) \| _ { 2 } , b _ { t } ^ { i } = \mathrm { h u b e r } ( \| \bar { e } _ { t } ( h ) \| _ { 2 } , \| \bar { n } _ { t } ( h ) \| _ { 2 } ) , \ \alpha _ { i } = D ( D + 2 \rho _ { i } ) . } \end{array}\tag{6}
$$

Alg. 1 summarizes the execution of this meta-learning framework, and the next theorem its regret.

Theorem 1 (Delayed optimistic OGD). Let $h _ { t }$ be the delayed gradientfeedback and let $\tilde { h } _ { t }$ be its prediction. For any comparator sequence $u _ { 1 } ^ { T } \in \mathcal { U } _ { T }$ , Algorithm 1 ensures universal dynamic regret:

$$
\mathcal { R } _ { T } ( L ) = \mathcal { O } \left( ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } + \sqrt { 1 + P _ { T } } \right) \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \tilde { h } _ { t } \| ^ { 2 } } \right) .
$$

Proofoutline. By convexity, the delayed loss regret is upper bounded by the delayed linearized regret, $\mathcal { R } _ { T } ( L ) \leq \mathcal { R } _ { T } ( \bar { \ell } ) = \mathrm { T e r m \bar { ( I ) } + T e r m \bar { ( I I ) } }$ . Term (I) is the meta-regret of Alg. 1, and is controlled by the delayed optimistic FTRL/AdaHedge update of [19, Theorem 12] with entropic regularizer, see Lemma 3. Term (II) is the delayed dynamic regret of the best-tuned base learner, Alg. 2, and is controlled by Theorem 4 and Corollary 2. Finally, the path-length grid, following the meta-learning idea of [25], ensures one base learner is tuned to $P _ { T }$ up to constants.

Algorithm 1 Universal Meta-Algorithm for Delayed Optimistic OCO   
Require: Grid of path-length scales $\{ \rho _ { i } \} _ { i = 1 } ^ { N }$ , initial weights $w _ { m } = \mathbf { 1 } / N .$   
Require: Constraint parameters $V , \Phi ,$ , and λ.   
1: Activate N base learners by running Algorithm 2 with scale $\rho _ { i } ,$ for each $i \in [ N ] .$   
2: for $t = m , \ldots , T$ do   
3: Receive $\boldsymbol { x } _ { t } ^ { ( i ) }$ from each base learner and set $X _ { t } = [ x _ { t } ^ { ( 1 ) } , \dots , x _ { t } ^ { ( N ) } ] .$   
4: Play the convex combination $x _ { t } = X _ { t } w _ { t }$   
5: Observe the delayed feedback $h _ { t - m }$ and the hints $\stackrel { \sim } { h } _ { t - m + 1 } ^ { t + 1 }$   
6: Send $h _ { t - m }$ and $\widetilde { h } _ { t - m + 1 : t + 1 }$ to all base learners.   
7: Receive the next predictions $\boldsymbol { x } _ { t + 1 } ^ { ( i ) }$ and set $X _ { t + 1 } = [ x _ { t + 1 } ^ { ( 1 ) } , \dots , x _ { t + 1 } ^ { ( N ) } ]$   
8: Define the delayed meta-loss and optimistic hints   
$M _ { t - m } = X _ { t - m } ^ { \top } h _ { t - m } , \qquad \widetilde { M } _ { s } = X _ { s } ^ { \top } \widetilde { h } _ { s } , \quad s = t - m + 1 , \ldots , t + 1 .$   
9: Compute $\eta _ { t + 1 } ^ { \mathrm { M } }$ using (3) with (4).   
10: Update $w _ { t + 1 }$ using (2).   
11: end for

Algorithm 2 Delayed Optimistic OGD Base Learner   
Require: Path-length scale $\rho ,$ initial points $x _ { 0 } , \ldots , x _ { m } .$   
Require: Constraint parameters $V , { \hat { \Phi } } ,$ and $\lambda .$   
1: for $t = m , \ldots , T$ do   
2: Submit $\boldsymbol { x } _ { t } ^ { ( i ) }$ to the meta-algorithm.   
3: Receive the delayed feedback $h _ { t - m }$ and the hints $\widetilde { h } _ { t - m + 1 : t + 1 } .$   
4: Compute $\eta _ { t + 1 } ^ { \rho }$ using (3) with (6).   
5: Update $\boldsymbol { x } _ { t + 1 } ^ { ( i ) }$ using (5).   
6: end for

Discussion. The factor $( m + 1 ) ^ { 3 / 2 }$ should be interpreted as the cost of maintaining a predictionadaptive bound under delayed feedback. If predictions are ignored and a uniform gradient bound is available, one can instead use fixed worst-case tuning and obtain a delayed dynamic-regret bound with linear dependence on m+1. We refer to Appendix B.1.2 for a more detailed discussion of this dependence. However, such a worst-case tuning would lose the benefit of accurate predictions and is less suitable for the constrained setting, where the surrogate gradients may not admit a useful uniform Lipschitz bound. Under worst-case predictions accuracy, Theorem 1 guarantees regret $\mathcal { O } \big ( m ^ { 3 / 2 } \sqrt { P _ { T } } \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } \| ^ { 2 } } , \big )$ , which matches the lower bound of [25] for $m = 1$ , and the delay dependency (equivalently, memory length) of [3] (static regret; no predictions). Interestingly, under perfect predictions we obtain regret $\mathcal { R } _ { T } ( L ) = \mathcal { O } ( 1 )$ . The bound holds for any value of m and recovers (up to constants) the respective no-memory bound $\mathcal { O } ( \sqrt { ( 1 + P _ { T } ) E _ { T } ( f ) } )$ of [22] in the known $P _ { T }$ case, where $E _ { T } ( f )$ is the accumulated prediction error.

Technically, the main difference from [19] is that, under a dynamic benchmark, prediction errors interact with the movement of comparator. We control this interaction using an auxiliary Young-type argument and show it does not change the desired prediction-adaptive order, except through the path-length dependence. We kindly refer the reader to Appendix B for the derivations and discussion. From a computational perspective, we use multiple base learners to adapt to unknown $P _ { T }$ . Still, unlike previous delayed OCO frameworks $\mathrm { e . g . }$ ., [18], we use only one copy of the algorithm for each $P _ { T }$ value, instead of $m + 1$ parallel copies that would have been trained only on a fraction of data.

## 4 Handling Time-average Adversarial Constraints

This section extends the above analysis to handle time-average adversarially-varying constraints, i.e., to constrained optimistic OCO with delayed feedback. To facilitate presentation, we use the shifted notation from Section 3: the feedback received at round t corresponds to decision $x _ { t - m } .$ COCO methods perform OCO over a time-varying Lagrangian-type surrogate using either a primal–dual method, e.g., [32], or a penalty method where the accumulated constraint violation, $Q _ { t } ,$ , is penalized directly in a scalarized surrogate loss $L _ { t } ( x )$ [6, 9, 33, 34]. We follow the latter approach and define:

$$
L _ { t } ( x _ { t - m } ) = V f _ { t } ( x _ { t - m } ) + \Phi ^ { \prime } \bigl ( Q _ { t - m - 1 } \bigr ) g _ { t } ^ { + } ( x _ { t - m } ) , \qquad Q _ { t } = Q _ { t - 1 } + g _ { t } ^ { + } ( x _ { t - m } ) ,\tag{7}
$$

where $V > 0$ is a trade-off parameter, and Φ an increasing convex penalty function. Equivalently, in the decision-time notation of Sec. 3, one sets $s = t - m$ , so that the feedback associated with $x _ { s }$ is the gradient of the surrogate loss revealed at $s + m$ . The penalty coefficient is evaluated at $Q _ { t - m - 1 }$ which is available when $x _ { t - m }$ is decided. This avoids the need to predict the future queue evolution when forming the gradient prediction. As a result, the prediction error of the surrogate gradient separates naturally into the prediction error of the objective and that of the constraint violation.

We define the prediction errors at the level of the delayed gradients. For every $s \in [ T ]$ , set

$$
\Delta _ { s } ^ { f } : = \nabla f _ { s + m } ( x _ { s } ) - \nabla \tilde { f } _ { s + m } ( \boldsymbol { \tilde { x } } _ { s } ) , \qquad E _ { T } ( f ) : = \sum _ { s = 1 } ^ { T } \lVert \Delta _ { s } ^ { f } \rVert _ { 2 } ^ { 2 } ,
$$

The term $E _ { T } ( g ^ { + } )$ is defined analogously. At t, all past decisions are known, thus suffices to predict only $\widetilde { \boldsymbol { x } } _ { t + 1 }$ . We now recall the decomposition connecting the surrogate regret to $\mathcal { R } _ { T } ( f )$ and $Q _ { T }$

Lemma 1 (adapted from [5, 11]). For any differentiable, convex and increasing penalty Φ, it is:

$$
\Phi ( Q _ { T } ) - \Phi ( Q _ { m - 1 } ) + V \mathcal { R } _ { T } ( f ) \leq \mathcal { R } _ { T } ( L ) + G ( m + 1 ) \Phi ^ { \prime } ( Q _ { T } ) ,\tag{8}
$$

where the dynamic regret is computed with respect to comparator sequences $u _ { 1 } ^ { T } \in \mathcal { U } _ { T }$

Remark 2. The queue is maintained globally using the actually played trajectory. After the learner plays $x _ { t } = X _ { t } w _ { t }$ , the queue is updated from the realized constraint violation, and the corresponding surrogate feedback $h _ { t }$ is constructed and fed to all base learners and to the meta-learner. Thus, the base learners do not maintain separate counterfactual queues.

## 4.1 Unknown Path Length: Regret and CCV Bounds

We first consider the case where $P _ { T }$ is unknown. We apply the meta-base delayed optimistic algorithm developed in Section 3 to the surrogate loss defined in (7).

Theorem 2 (Unknown-path-length bounds with CCV-oriented tuning). Let $u _ { 1 } ^ { T } \in \mathcal { U } _ { T }$ be any feasible comparator sequence with path length $P _ { T }$ . Using a quadratic penalty $\Phi ( Q ) \bar { { } = } Q ^ { 2 }$ , and

$$
V : = 4 ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } \right) ^ { 2 } \sqrt { E _ { T } ( g ^ { + } ) } + 2 G ^ { 2 } ( m + 1 ) + Q _ { m - 1 } ^ { 2 } .
$$

Then, under the delayed optimistic meta-base algorithm 1 the following bounds hold

$$
\begin{array} { r l r } & { } & { \mathcal { R } _ { T } ( f ) \leq \mathcal { O } \Big ( A \sqrt { E _ { T } ( f ) } + ( m + 1 ) ^ { 3 / 2 } ( 1 + P _ { T } ) \sqrt { E _ { T } ( g ^ { + } ) } + ( m + 1 ) \Big ) , \qquad a n d } \\ & { } & { Q _ { T } \leq \mathcal { O } \Big ( A \sqrt { T } E _ { T } ( g ^ { + } ) ^ { 1 / 4 } + ( m + 1 ) ^ { \frac { 5 } { 4 } } \sqrt { T } \Big ) \ w i t h \ A = \mathcal { O } \Big ( ( m + 1 ) ^ { \frac { 3 } { 2 } } \Big ( 1 + \sqrt { 1 + \log N } + \sqrt { 1 + P _ { T } } \Big ) \Big ) } \end{array}
$$

We note that $P _ { T }$ appearing in the bound on $Q _ { T }$ can be taken as the smallest feasible path length. In particular, it can be zero whenever there exists afixed decision that satisfies all constraints.

Discussion. Theorem 2 gives an optimistic delayed-COCO guarantee under a universal dynamic benchmark. Under worst-case predictions, we obtain an $\mathcal { O } ( ( 1 + P _ { T } ) \sqrt { T } )$ regret bound and an $\mathcal { O } \Big ( \operatorname* { m a x } \{ T ^ { 3 / 4 } , \sqrt { ( 1 + P _ { T } ) T } \} \Big )$ CCV bound, recovering the best known rates of [11] when $m = 0$ Under perfect predictions, our bounds reduce to $\mathcal { O } ( 1 )$ regret and $\mathcal { O } ( \sqrt { T } )$ CCV. We note that [7] achieves $\mathcal { O } ( \sqrt { ( 1 + P _ { T } ) T } )$ regret and $\mathcal { O } ( \sqrt { T } )$ CCV for time-invariant constraints. Our perfectprediction regime recovers the same CCV order while handling time-varying constraints. In the known $P _ { T }$ setting, we get better bounds as shown later. As discussed in Appendix B.1.2, the extra $\sqrt { m + 1 }$ factor in the simplified delayed-regret bound can be absorbed by the meta-complexity or path-length terms when log N or $P _ { T }$ dominates $m$ , leading to linear dependence on the delay. Finally, since $V$ depends on $E _ { T } ( \bar { g } ^ { + } )$ , we use a doubling trick to learn this quantity online, introducing at most an $\mathcal { O } ( \log T )$ overhead, as is standard in optimistic COCO, see, e.g., [5, 9].

Remark 3 (Penalty choice and path-length dependence). In COCO, the surrogate loss must do two things: track the comparator for regret and penalize constraint violations to control the queue. This coupling between tracking and feasibility is what leads to a higher dependency on $P _ { T }$ . If $P _ { T }$ is known, the penalty parameter can be tuned to the true scale of comparator motion, allowing balancing regret and queue control. When $P _ { T }$ is unknown, this tuning is not available. Moreover, the usual meta-learning approach cannot be applied directly to the penalty parameter, since different penalty choices generate different surrogate loss sequences. Thus, one must use a single surrogate construction that works uniformly; this leads to a more conservative trade-off.

Also, the $T ^ { 3 / 4 }$ -type term in the CCV bound comes from the same limitation. We use the quadratic penalty $\Phi ( Q ) = \bar { Q } ^ { 2 }$ , which is the smallest polynomial penalty that still yields sublinear queue growth. Stronger penalties, such as $\Phi ( Q ) = Q ^ { \beta }$ with $\beta > 2$ , or exponential penalties, may improve the CCV exponent, but they also amplify the path-length contribution in the surrogate regret bound. Thus, improving CCV in this way comes at the cost of worse dependence on $P _ { T }$ . We study next the known-path-length regime, where this trade-off can be tuned more carefully.

## 4.2 Known Path Length

When $P _ { T }$ is known we can tune the surrogate loss directly to the scale of the comparator motion via the penalty Φ. Hence, we can direcly run only one base learner, Algorithm 2, with $\rho = P _ { T }$

Theorem 3 (Known-path-length case with exponential penalty). For any $u _ { 1 } ^ { T } \in \mathcal { U } _ { T }$ with path length $P _ { T }$ to be known. Choose $\rho = P _ { T } , V = 1$ , and $\Phi ( Q ) = e ^ { \lambda Q } - 1$ , with:

$$
\lambda = { \frac { 1 } { 2 \left( 4 ( m + 1 ) ^ { 3 / 2 } \left( D + \sqrt { D ( D + P _ { T } ) } \right) \sqrt { E _ { T } ( g ^ { + } ) } + G ( m + 1 ) \right) } } .
$$

Feeding these parameters into Algorithm 2, thefollowing bounds hold:

$$
\mathcal { R } _ { T } ( f ) \leq \mathcal { O } \Big ( ( m + 1 ) ^ { 3 / 2 } \sqrt { ( 1 + P _ { T } ) E _ { T } ( f ) } \Big ) , \qquad a n d\tag{9}
$$

$$
Q _ { T } \leq \mathcal { O } \Big ( \Big ( ( m + 1 ) ^ { 3 / 2 } \sqrt { ( 1 + P _ { T } ) E _ { T } ( g ^ { + } ) } + G ( m + 1 ) \Big ) \log T \Big ) .\tag{10}
$$

Discussion. Theorem 3 gives prediction-dependent bounds for the known $P _ { T }$ case, without using meta-learning. We first compare with dynamic optimistic COCO works with predictions. When $m =$ 0, we recover the no-memory optimistic COCO bounds of [9]: $\mathcal { R } _ { T } ( f ) = \mathcal { O } \big ( \sqrt { P _ { T } E _ { T } ( f ) } \big )$ $Q _ { T } =$ $\mathcal { O } ( ( \sqrt { P _ { T } E _ { T } ( g ^ { + } ) } + G ) \log T )$ . Thus, our result extends their prediction-dependent guarantees to delayed and memory-dependent settings. The closest static optimistic memory-dependent works are [5, 17]. These works rely on separability of the memory terms, which facilitates isolating the contribution of each decision, but leads to a more restrictive benchmark in constrained problems and restricted class of loss functions. Our approach avoids this assumption and recovers the same type of prediction-dependent bounds while preserving the original joint feasibility benchmark. Finally, compared to [35], which studies the static memory-constrained problem without predictions, our bounds improve over their $\tilde { \mathcal { O } } ( T ^ { 2 / 3 } )$ regret and CCV rates.

Remark 4. If $P _ { T }$ is unknown but measurable online, a standard doubling trick recovers the bound of this subsection. The same applies to tuning λ, since $E _ { T } ( g ^ { + } )$ ) is observable online.

## 5 Memory as Delayed Feedback

We now transfer the delayed optimistic dynamic regret guarantees to the memory problems. In this setting, the loss and constraint functions depend on a block of past decisions, $f _ { t } ( x _ { t - m } ^ { t } )$ and $g _ { t } ( x _ { t - m } ^ { t } )$ Thus, $x _ { s }$ affects all losses whose memory window contains it, i.e., all t such that $t - m \leq s \leq t$ Hence, its full first-order contribution is revealed only after these losses are observed, which induces a feedback delay of at most m. Note that we redefine here the learning functions, $L _ { t }$ , and the horizon $( T _ { m }$ instead of T) (further explanations about notation conventions, please see Appendix Sec. D).

The reduction is performed at the regret level. The memory losses are not assumed to be linear. Rather, we use the standard OCO linearization argument: applying the first-order convexity inequality to each memory surrogate loss $L _ { t }$ around the played block $x _ { t - m } ^ { t }$ gives an upper bound in terms of partial gradients. More precisely, when $L _ { t }$ is nondifferentiable, we select a joint subgradient

$$
G _ { t } = ( G _ { t , t - m } , \ldots , G _ { t , t } ) \in \partial L _ { t } ( x _ { t - m } ^ { t } ) ,
$$

and interpret $\nabla _ { x _ { s } } L _ { t } ( x _ { t - m } ^ { t } )$ below as the coordinate block $G _ { t , s } .$ . Thus, all coordinate blocks associated with a given $L _ { t }$ are taken from the same joint subgradient, the same convention applies to predicted functions. For each decision index s, define

$$
I _ { s } : = \{ t \in [ T _ { m } ] : t - m \leq s \leq t \} , \qquad h _ { s } : = \sum _ { t \in I _ { s } } \nabla _ { x _ { s } } L _ { t } ( x _ { t - m } ^ { t } ) .\tag{11}
$$

Here, $I _ { s }$ is the set of losses whose memory window contains $x _ { s } ,$ and $h _ { s }$ is the corresponding credit gradient. It induces the auxiliary linearized loss $\ell _ { s } ( z ) : = \langle h _ { s } , z \rangle$ . Since $I _ { s } \subseteq \{ s , \dotsc , s + \bar { m } \}$ , the vector $h _ { s }$ is fully revealed no later than round $s + m$

Lemma 2 (Memory-to-delay linearization). Let $L _ { t } : \mathcal { X } ^ { m + 1 } \mapsto \mathbb { R }$ be jointly convex $\forall t \in [ T _ { m } ]$ and define $I _ { s } , \ h _ { s }$ as in (11), using the joint-subgradient convention above whenever $L _ { t }$ is nondifferentiable. Then,for any comparator sequence $\mathbf { \bar { \Psi } } u _ { 1 } ^ { T } \in \mathcal { U } _ { T }$ , it holds that

$$
\sum _ { t = m } ^ { T } L _ { t } ( x _ { t - m } ^ { t } ) - L _ { t } ( u _ { t - m } ^ { t } ) \leq \sum _ { s = 1 } ^ { T } \langle h _ { s } , x _ { s } - u _ { s } \rangle .
$$

Moreover, $h _ { s }$ is revealed with delay at most m.

The following corollary transfers the delayed guarantees to the memory setting.

Corollary 1 (Transfer from delayed OCO to memory). Any regret guaranteefor delayed OCO with delay at most m, applied to the auxiliary linearized losses $\bar { \ell } _ { s } ( z ) = \bar { \langle } h _ { s } , z \rangle$ , yields the corresponding guarantee for the memory-dependent regret. In particular, Theorems $I , \ 2 ,$ , and 3 apply to the memory-dependent problem after replacing the delayed gradients by the credit gradients $h _ { s } .$ . The corresponding objective and constraint prediction errors induced by the credit gradients are specified below.

In the delayed case without memory, the hint predicts one delayed gradient, e.g., $\nabla f _ { s + m } ( x _ { s } )$ , and constructing the next hint only requires a one-step proxy decision. In the memory case, the hint must predict the whole credit gradient $h _ { s }$ . Thus, one may set

$$
\widetilde { h } _ { s } : = \sum _ { t \in I _ { s } } \nabla _ { x _ { s } } \widetilde { L } _ { t } ( \widetilde { x } _ { t - m } ^ { t } ) .\tag{12}
$$

Consequently, constructing $\widetilde { h } _ { s + 1 }$ may require predicting the future memory block up to $\widetilde { x } _ { s + m + 1 }$ , not only the next decision $\widetilde { x } _ { s + 1 }$ . Thus, the hints used by Algorithm 1 when $P _ { T }$ is unknown, or directly by Algorithm 2 when $P _ { T }$ is known, are

$$
\widetilde { h } _ { s - m + 1 : s + 1 } = \sum _ { \tau = s - m + 1 } ^ { s + 1 } \sum _ { t \in I _ { \tau } } \nabla _ { x _ { \tau } } \widetilde { L } _ { t } ( \widetilde { x } _ { t - m } ^ { t } ) .
$$

Remark 5. In the constrained case, we use the surrogate loss

$$
L _ { t } ( x _ { t - m } ^ { t } ) = V f _ { t } ( x _ { t - m } ^ { t } ) + \Phi ^ { \prime } ( Q _ { t - m - 1 } ) g _ { t } ^ { + } ( x _ { t - m } ^ { t } ) , \qquad Q _ { t } = Q _ { t - 1 } + g _ { t } ^ { + } ( x _ { t - m } ^ { t } ) .\tag{13}
$$

The regret decomposition Lemma of [5] applies to this surrogate too for $u _ { 1 } ^ { T } \in \mathcal { U } _ { T }$ . Thus, once the dynamic regret of $L _ { t }$ is controlled via Corollary 1, the objective-regret and CCV bounds follow as in the delayed case. Equivalently, the prediction errors for the final COCO-M bounds are defined at the level of the objective and constraint credit gradients, $E _ { T } ^ { \mathrm { m e m } } ( f )$ and $E _ { T } ^ { \mathrm { m e m } } ( g ^ { + } )$ , where

$$
E _ { T } ^ { \mathrm { m e m } } ( f ) : = \sum _ { s = 1 } ^ { T } \left\| \sum _ { t \in I _ { s } } \left[ \nabla _ { x _ { s } } f _ { t } ( x _ { t - m } ^ { t } ) - \nabla _ { x _ { s } } \widetilde { f } _ { t } ( \widetilde { x } _ { t - m } ^ { t } ) \right] \right\| ^ { 2 } ,\tag{14}
$$

$$
E _ { T } ^ { \mathrm { m e m } } ( g ^ { + } ) : = \sum _ { s = 1 } ^ { T } \operatorname* { m a x } _ { r \in I _ { s } } \left\| \sum _ { \substack { t \in I _ { s } } } \left[ \nabla _ { x _ { s } } g _ { t } ^ { + } ( x _ { t - m } ^ { t } ) - \nabla _ { x _ { s } } \widetilde { g } _ { t } ^ { + } ( \widetilde { x } _ { t - m } ^ { t } ) \right] \right\| ^ { 2 } .\tag{15}
$$

The latter quantity provides a penalty-independent upper envelope for the normalized constraint credit-gradient prediction error, as shown in Appendix D.

Discussion. Prior optimistic approaches to OCO with memory [5, 17], isolate the contribution of each decision at the level of loss functions, which requires separability across the memory variables. Our reduction keeps the memory-dependent loss in its original jointly convex form and linearizes only at the regret level. The price is that optimistic hints must predict credit gradients rather than single gradients. This may require predicting future memory blocks, but avoids imposing componentwise feasibility on separable parts of the constraint, and preserves the original joint feasibility benchmark. For the constraints, different memory contributions carry different queue-dependent penalty coefficients. We therefore use a penalty-independent envelope of the aggregated prediction errors, which parallels the separable setting [5] while controlling the corresponding weighted error. It vanishes under perfect predictions and preserves an $O ( \sqrt { T } )$ worst-case dependence, up to memorydependent factors.

## 6 Conclusions and Limitations

The proposed framework contributes in closing a gap in COCO with memory, and NSC with timeaverage budget constraints, and can be used to tackle important problems in ML, operations research, control of cyber-physical systems, and so on. Aiming for universal dynamic regret ensures the learning is against the most demanding comparators; while the inclusion of optimism enables the bounds to improve, reaching path-length-oblivious regret $\mathcal { R } _ { T } = \mathcal { O } ( 1 )$ , together with $Q _ { T } = \mathcal { O } ( \sqrt { T } )$ when $P _ { T }$ is unknown, and $\stackrel { \bar { Q } } { T } = \mathcal { O } ( \log \bar { T } )$ when known or observable at runtime.

There are few limitations which call for follow up work. First, the framework requires predicting, apart from future functions, also some future actions. This cyclic dependency is implicit in almost all optimistic learning works (unless the losses are linear); still, it is worth investigating whether it can be avoided, see discussion in [21]. Secondly, the constraint violation does not scale down to $\mathcal { O } ( 1 )$ under perfect predictions. While this matches the best-known optimistic COCO bounds (without memory), it is certainly exciting to explore further improvements. Finally, expanding the set of eligible benchmarks in COCO is a key challenge; our work contributes in this direction but the question of approaching closer to horizon-long feasibility remains open.

## Acknowledgments

This work was supported in part by the French government through the France 2030 program under the Celtic RAI-6Green project, and in part by the European Commission under Grant No. 101192462 (FLECON-6G).

## References

[1] Elad Hazan. Introduction to online convex optimization. 2016.

[2] Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213, 2019.

[3] Oren Anava, Elad Hazan, and Shie Mannor. Online learning for adversaries with memory: price of past mistakes. Advances in Neural Information Processing Systems, 28, 2015.

[4] Elad Hazan and Karan Singh. Introduction to online control, 2026. URL https://arxiv. org/abs/2211.09619.

[5] Mohammed Abdullah, George Iosifidis, Salah Eddine Elayoubi, and Tijani Chahed. Constrained online convex optimization with memory and predictions. Proceedings ofthe AAAI Conference on Artificial Intelligence, 40(24):19524–19532, Mar. 2026. doi: 10.1609/aaai.v40i24.39031.

[6] Abhishek Sinha and Rahul Vaze. Beyond $\tilde { o } ( \backslash \sqrt { t } )$ constraint violation for online convex optimization with adversarial constraints. arXiv preprint arXiv:2505.06709, 2025.

[7] Xinlei Yi, Xiuxian Li, Tao Yang, Lihua Xie, Tianyou Chai, and Karl H Johansson. Regret and cumulative constraint violation analysis for online convex optimization with long term constraints. In International Conference on Machine Learning, pages 11998–12008, 2021.

[8] Hengquan Guo, Xin Liu, Honghao Wei, and Lei Ying. Online convex optimization with hard constraints: Towards the best of two worlds and beyond. Advances in Neural Information Processing Systems, 35:36426–36439, 2022.

[9] Jordan Lekeufack and Michael I Jordan. An optimistic algorithm for online convex optimization with adversarial constraints. arXiv preprint arXiv:2412.08060, 2024.

[10] Qingsong Liu, Wenfei Wu, Longbo Huang, and Zhixuan Fang. Simultaneously achieving sublinear regret and constraint violations for online convex optimization with time-varying constraints. Performance Evaluation, 152:102240, 2021.

[11] Subhamon Supantha and Abhishek Sinha. Universal dynamic regret and constraint violation bounds for constrained online convex optimization. arXiv preprint arXiv:2510.01867, 2025.

[12] Lijun Zhang, Wei Jiang, Shiyin Lu, and Tianbao Yang. Revisiting smoothed online learning. In Advances in Neural Information Processing Systems 34, pages 13599–13612, 2021.

[13] Sifan Yang, Wenhao Yang, Wei Jiang, Yuanyu Wan, and Lijun Zhang. Smoothed online convex optimization with delayed feedback. In James Kwok, editor, Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25, pages 6812–6820. International Joint Conferences on Artificial Intelligence Organization, 8 2025. doi: 10.24963/ ijcai.2025/758. URL https://doi.org/10.24963/ijcai.2025/758. Main Track.

[14] Peng Zhao, Yu-Hu Yan, Yu-Xiang Wang, and Zhi-Hua Zhou. Non-stationary online learning with memory and non-stochastic control. Journal of Machine Learning Research, 24(206):1–70, 2023.

[15] Dheeraj Baby and Yu-Xiang Wang. Optimal dynamic regret in lqr control. Advances in Neural Information Processing Systems, 35:24879–24892, 2022.

[16] Hongyu Zhou, Zirui Xu, and Vasileios Tzoumas. Efficient online learning with memory via frank-wolfe optimization: Algorithms with bounded dynamic regret and applications to control. In 2023 62nd IEEE Conference on Decision and Control (CDC), pages 8266–8273, 2023. doi: 10.1109/CDC49753.2023.10384153.

[17] Naram Mhaisen and George Iosifidis. Optimistic online non-stochastic control via ftrl. In 2024 IEEE 63rd Conference on Decision and Control (CDC), pages 7116–7123. IEEE, 2024.

[18] Pooria Joulani, Andras Gyorgy, and Csaba Szepesvári. Online learning under delayed feedback. In International conference on machine learning, pages 1453–1461. PMLR, 2013.

[19] Genevieve E Flaspohler, Francesco Orabona, Judah Cohen, Soukayna Mouatadid, Miruna Oprescu, Paulo Orenstein, and Lester Mackey. Online learning with optimism and delay. In International Conference on Machine Learning, pages 3363–3373. PMLR, 2021.

[20] Yuanyu Wan, Chang Yao, Mingli Song, and Lijun Zhang. Non-stationary online convex optimization with arbitrary delays. In Proceedings of the 41st International Conference on Machine Learning, pages 49991–50011, 2024.

[21] Pedro Zattoni Scroccaro, Arman Sharifi Kolarijani, and Peyman Mohajerin Esfahani. Adaptive composite online optimization: Predictions in static and dynamic environments. IEEE Transactions on Automatic Control, 68(5):2906–2921, 2023.

[22] Naram Mhaisen and George Iosifidis. On the dynamic regret of following the regularized leader: Optimism with history pruning. In International Conference on Machine Learning, pages 43990–44016. PMLR, 2025.

[23] Shie Mannor, John N Tsitsiklis, and Jia Yuan Yu. Online learning with sample path constraints. Journal ofMachine Learning Research, 10(3), 2009.

[24] Lijun Zhang, Tianbao Yang, Rong Jin, and Zhi-Hua Zhou. Dynamic regret of strongly adaptive methods. In Proceedings of the 35th International Conference on Machine Learning (ICML), pages 5882–5891, 2018.

[25] Lijun Zhang, Shiyin Lu, and Zhi-Hua Zhou. Adaptive online learning in dynamic environments. Advances in neural information processing systems, 31, 2018.

[26] Alexander Rakhlin and Karthik Sridharan. Online learning with predictable sequences. In Conference on Learning Theory, pages 993–1019, 2013.

[27] Mehryar Mohri and Scott Yang. Accelerating online convex optimization via adaptive prediction. In Artificial Intelligence and Statistics, pages 848–856. PMLR, 2016.

[28] Pooria Joulani, András György, and Csaba Szepesvári. A modular analysis of adaptive (non-)convex optimization: Optimism, composite objectives, variance reduction, and variational bounds. Theor. Comput. Sci., 808:108–138, February 2020.

[29] Neri Merhav, Erik Ordentlich, Gadiel Seroussi, and Marcelo J Weinberger. On sequential strategies for loss functions with memory. IEEE Transactions on Information Theory, 48(7): 1947–1958, 2002.

[30] Andras Gyorgy and Gergely Neu. Near-optimal rates for limited-delay universal lossy source coding. IEEE Transactions on Information Theory, 60(5):2823–2834, 2014.

[31] Alexander Korotin, Vladimir V’yugin, and Evgeny Burnaev. Adaptive hedging under delayed feedback. Neurocomputing, 397:356–368, 2020.

[32] Jianjun Yuan and Andrew Lamperski. Online convex optimization for cumulative constraints. In Advances in Neural Information Processing Systems (NeurIPS), pages 6140–6149, 2018.

[33] Abhishek Sinha and Rahul Vaze. Optimal algorithms for online convex optimization with adversarial constraints. Advances in Neural Information Processing Systems, 37:41274–41302, 2024.

[34] Nikolaos Liakopoulos, Apostolos Destounis, Georgios Paschos, Thrasyvoulos Spyropoulos, and Panayotis Mertikopoulos. Cautious regret minimization: Online optimization with long-term budget constraints. Proceedings of Machine Learning Research, pages 3944–3952. PMLR, 09–15 Jun 2019.

[35] Xin Liu, Zixian Yang, and Lei Ying. Online nonstochastic control with adversarial and static constraints. In International Conference on Machine Learning, pages 22277–22288. PMLR, 2023.

[36] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th international conference on machine learning (icml-03), pages 928–936, 2003.

[37] Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games. Cambridge university press, 2006.

[38] Dheeraj Baby and Yu-Xiang Wang. Optimal dynamic regret in exp-concave online learning. In Proceedings of the 34th Annual Conference On Learning Theory, pages 359–409, 2021.

[39] Peng Zhao, Yu-Jie Zhang, Lijun Zhang, and Zhi-Hua Zhou. Dynamic regret of convex and smooth functions. In Advances in Neural Information Processing Systems, volume 33, pages 12510–12520, 2020.

[40] Peng Zhao, Yu-Jie Zhang, Lijun Zhang, and Zhi-Hua Zhou. Adaptivity and non-stationarity: Problem dependent dynamic regret for online convex optimization. Journal of Machine Learning Research, 25(98):1–52, 2024.

[41] Lijun Zhang, Shiyin Lu, and Tianbao Yang. Minimizing dynamic regret and adaptive regret simultaneously. In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), pages 309–319, 2020.

[42] Tianbao Yang, Lijun Zhang, Rong Jin, and Jinfeng Yi. Tracking slowly moving clairvoyant: Optimal dynamic regret of online learning with true and noisy gradient. In International Conference on Machine Learning, pages 449–457. PMLR, 2016.

[43] Lijun Zhang, Tianbao Yang, Jinfeng Yi, Rong Jin, and Zhi-Hua Zhou. Improved dynamic regret for non-degenerate functions. In Advances in Neural Information Processing Systems 30, pages 732–741, 2017.

[44] Lijun Zhang, Guanghui Wang, Jinfeng Yi, and Tianbao Yang. A simple yet universal strategy for online convex optimization. In International Conference on Machine Learning, pages 26605–26623. PMLR, 2022.

[45] Xi Chen, Yining Wang, and Yu-Xiang Wang. Non-stationary stochastic optimization under $l _ { p , q }$ -variation measures. arXiv preprint arXiv:1708.03020, 2017.

[46] Rahul Vaze and Abhishek Sinha. $o ( \sqrt { \{ T \} } )$ static regret and instance dependent constraint violation for constrained online convex optimization. arXiv preprint arXiv:2502.05019, 2025.

[47] Rahul Vaze and Abhishek Sinha. $o ( \sqrt { T } )$ static regret and instance dependent constraint violation for constrained online convex optimization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[48] Rodolphe Jenatton, Jim Huang, and Cédric Archambeau. Adaptive algorithms for online convex optimization with long-term constraints. In International Conference on Machine Learning, pages 402–411. PMLR, 2016.

[49] Daron Anderson, George Iosifidis, and Douglas J Leith. Lazy lagrangians for optimistic learning with budget constraints. IEEE/ACM Transactions on Networking, 31(5):1935–1949, 2023.

[50] Mehrdad Mahdavi, Rong Jin, and Tianbao Yang. Trading regret for efficiency: online convex optimization with long term constraints. The Journal of Machine Learning Research, 13(1): 2503–2528, 2012.

[51] Heyan Huang, Ping Wu, Haolin Lu, and Zhengyang Liu. On the constrained online convex optimization with feedback delay. Expert Systems with Applications, 287:127871, 2025. ISSN 0957-4174. doi: https://doi.org/10.1016/j.eswa.2025.127871. URL https: //www.sciencedirect.com/science/article/pii/S0957417425014939.

[52] Yingying Li, Subhro Das, and Na Li. Online optimal control with affine constraints. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 8527–8537, 2021.

[53] N. Merhav, E. Ordentlich, G. Seroussi, and M.J. Weinberger. On sequential strategies for loss functions with memory. IEEE Transactions on Information Theory, 48(7):1947–1958, 2002. doi: 10.1109/TIT.2002.1013135.

[54] Raman Arora, Ofer Dekel, and Ambuj Tewari. Online bandit learning against an adaptive adversary: from regret to policy regret. ICML’12, 2012.

[55] Chang-Heng Wang, Jaime Llorca, Antonia M. Tulino, and Tara Javidi. Dynamic cloud network control under reconfiguration delay and cost. IEEE/ACM Transactions on Networking, 27(2): 491–504, 2019.

[56] Samrat Mukhopadhyay and Abhishek Sinha. Online caching with optimal switching regret. In 2021 IEEE International Symposium on Information Theory (ISIT), pages 1546–1551, 2021.

[57] Tareq Si Salem, Giovanni Neglia, and Stratis Ioannidis. No-regret caching via online mirror descent. ACM Trans. Model. Perform. Eval. Comput. Syst., 8(4), 2023.

[58] Kush Bhatia and Karthik Sridharan. Online learning with dynamics: A minimax perspective. In Advances in Neural Information Processing Systems, volume 33, pages 15020–15030, 2020.

[59] Raunak Kumar, Sarah Dean, and Robert Kleinberg. Online convex optimization with unbounded memory. In Advances in Neural Information Processing Systems, volume 36, pages 26229– 26270, 2023.

[60] Naman Agarwal, Brian Bullins, Elad Hazan, Sham Kakade, and Karan Singh. Online control with adversarial disturbances. In International Conference on Machine Learning, pages 111–119. PMLR, 2019.

[61] Nanfei Jiang, Spencer Hutchinson, and Mahnoosh Alizadeh. Online nonstochastic control with convex safety constraints. arXiv preprint arXiv:2501.18039, 2025.

[62] Marko Nonhoff and Matthias A Müller. An online convex optimization algorithm for controlling linear systems with state and input constraints. In 2021 American Control Conference (ACC), pages 2523–2528. IEEE, 2021.

[63] Yu-Hu Yan, Peng Zhao, and Zhi-Hua Zhou. Online non-stochastic control with partial feedback. Journal ofMachine Learning Research, 24(273):1–50, 2023.

[64] Hongyu Zhou and Vasileios Tzoumas. Safe non-stochastic control of linear dynamical systems. In 2023 62nd IEEE Conference on Decision and Control (CDC), pages 5033–5038. IEEE, 2023.

[65] Marcelo J Weinberger and Erik Ordentlich. On delayed prediction of individual sequences. IEEE Transactions on Information Theory, 48(7):1959–1976, 2002.

[66] Kent Quanrud and Daniel Khashabi. Online learning with adversarial delays. In Advances in Neural Information Processing Systems 28, pages 1270–1278, 2015.

[67] Pooria Joulani, Andras Gyorgy, and Csaba Szepesvári. Delay-tolerant online convex optimization: Unified analysis and adaptive-gradient algorithms. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 30, 2016.

[68] Yu-Guan Hsieh, Franck Iutzeler, Jérôme Malick, and Panayotis Mertikopoulos. Multi-agent online optimization with delays: Asynchronicity, adaptivity, and optimism. Journal of Machine Learning Research, 23(78):1–49, 2022.

[69] Yuanyu Wan. Black-box reductions for decentralized online convex optimization in changing environments. In Proceedings of Thirty Eighth Conference on Learning Theory, pages 5605– 5631, 2025.

[70] Juncheng Wang, Min Dong, Ben Liang, Gary Boudreau, and Hatem Abou-Zeid. Delay-tolerant oco with long-term constraints: Algorithm and its application to network resource allocation. IEEE/ACM Transactions on Networking, 31(1):147–163, 2023. doi: 10.1109/TNET.2022. 3188285.

[71] Ali Jadbabaie, Alexander Rakhlin, Shahin Shahrampour, and Karthik Sridharan. Online optimization: Competing with dynamic comparators. In Proceedings ofthe 18th International Conference on Artificial Intelligence and Statistics, pages 398–406, 2015.

The structure of this appendix is as follows.

1. Section A provides a comprehensive discussion of related work.

2. Section B provides the proof of Theorem 1. This requires deriving a dynamic-regret bound for our delayed optimistic OCO algorithm; the section includes both the auxiliary results and their proofs.

3. Section C provides the proofs of Theorem 2 and Theorem 3, respectively in Sections C.1 and C.2.

4. Section D provides the proofs of the memory-to-delay equivalence results stated in the main paper, namely Lemma 2 and Corollary 1.

## A Discussion of Related Work

This section provides a detailed discussion of related works. Table 1 presents a concise high-level comparison with selected prior results. The proposed method is at the nexus of different OCO flavors, namely OCO with time-average adversarially-changing constraints, OCO with memory, OCO with delay and OCO with predictions. Therefore, we review below most relevant works.

## A.1 Dynamic Regret in OCO and COCO

Regarding the standard OCO model, the seminal work [36] proved the universal dynamic regret bound $\mathcal { O } ( ( 1 + \mathcal { P } _ { T } ) \sqrt { T } )$ for Online Gradient Descent (OGD), where $\mathcal { P } _ { T }$ is the norm-based path-length of the benchmark. More recently, [25] showed this problem has a lower bound $\Omega ( \sqrt { ( 1 + \mathcal { P } _ { T } ) T } )$ and designed a tight algorithm (i.e., meeting this bound) using Hedge [37]. Namely, they deploy a set of OGD experts (or, base learners) with learning rates that cover the entire range of path-length values, and a meta-learning algorithm that discerns at runtime which base learner is aligned with the realized functions. Refinements of this result include curvature-specific bounds [38], variation-based guarantees [39, 40], and methods which achieve both adaptive and dynamic regret [41]. On the other hand, [42] showed that OGD achieves $\mathcal { O } ( \sqrt { ( 1 + \mathcal { P } _ { T } ) T } )$ worst-case dynamic regret, where $\mathcal { P } _ { T }$ here is the path-length of the per-round minimizers. Under additional assumptions, tighter worst-case dynamic regret bounds were established by follow-up works, e.g., see [43–45].

There is a recent flurry of research in COCO, with important works improving the bounds of the most general problem (time-varying adversarial constraints) while dropping assumptions (such as the Slater condition); see for instance [33], [46], [47], [6] which follow a penalty-based approach like the one considered in this work. A different class of COCO techniques rely on primaldual algorithms with adaptive regularization, e.g., see [32, 48–50]. However, dynamic regret in COCO is largely underexplored. The few exceptions include [7] that achieved $\mathcal { O } ( \sqrt { ( 1 + \mathcal { P } _ { T } ) T } )$ worst-case dynamic regret and $\mathcal { O } ( \sqrt { T } )$ CCV for time-invariant functions, that was subsequently improved to $\mathcal { O } ( \sqrt { ( 1 + \mathcal { P } _ { T } ) T } )$ regret and ${ \mathcal { O } } ( \log T )$ CCV by [8]. In case of time-varying adversarial constraints, [8] achieved $\mathcal { O } ( \sqrt { ( 1 + \mathcal { P } _ { T } ) T } )$ universal dynamic regret and $\mathcal { O } ( T ^ { 3 / 4 } )$ CCV. More recently, [9] proposed an algorithm with worst-case dynamic regret $\mathcal { O } ( \sqrt { ( 1 + \mathcal { P } _ { T } ) T } )$ and $\tilde { \mathcal { O } } ( { \sqrt { T } } )$ CCV. [10] introduced an algorithm with regret $\mathcal { O } ( \sqrt { ( 1 + \mathcal { P } _ { T } ) T } )$ and CCV $\mathcal { O } ( \operatorname* { m a x } ( \sqrt { T } , V _ { g } ) )$ ), where $\begin{array} { r } { V _ { g } = \sum _ { t } \operatorname* { s u p } _ { x \in \mathcal { X } } | | g _ { t } ( x ) - \bar { g _ { t - 1 } } ( x ) | | _ { 2 } } \end{array}$ . Other recent works include [51] which considers time-invariant constraints, and [11] which established universal dynamic regret and CCV guarantees under adversarial constraints using a penalty-based method. None of these works considers memory dependency; a property that changes fundamentally the required methodological approach.

## A.2 Memory Effects: COCO-M and NSC

The first study on OCO-M appeared in [53], followed later by [54]. A seminal work in this area is [3] which proved that the Follow-The-Regularized-Leader algorithm can be applied to a proxy memory-less function and guarantee sublinear regret for the memory-based function. A special case of OCO-M is when the learner is penalized for decision switches, i.e., the loss function has memory of just $m = 1$ . This setting captures several practical problems such as workload management in datacenters [55], caching with fetching delays, etc., [56], [57]. On the other hand, there are studies for the case of infinite (but discounted) memory effects, e.g., see [58], [59].

<table><tr><td rowspan=2 colspan=1>Reference</td><td rowspan=2 colspan=1>Constraints</td><td rowspan=2 colspan=1>Loss</td><td rowspan=1 colspan=1>Memory</td><td rowspan=2 colspan=1>Predictions</td><td rowspan=2 colspan=1>Regret</td><td rowspan=2 colspan=1>Assumptions</td></tr><tr><td rowspan=1 colspan=1>Obj. Constr.</td></tr><tr><td rowspan=1 colspan=1>[35]</td><td rowspan=1 colspan=1>adversarial</td><td rowspan=1 colspan=1>convex</td><td rowspan=1 colspan=1>√     √</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>Static</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[52]</td><td rowspan=1 colspan=1>static affine</td><td rowspan=1 colspan=1>convex</td><td rowspan=1 colspan=1>√     √</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>Static</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[3]</td><td rowspan=1 colspan=1>none</td><td rowspan=1 colspan=1>convex</td><td rowspan=1 colspan=1>√     x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>Static</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[17]</td><td rowspan=1 colspan=1>none</td><td rowspan=1 colspan=1>linear</td><td rowspan=1 colspan=1>√     x</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>Static</td><td rowspan=1 colspan=1>separability</td></tr><tr><td rowspan=1 colspan=1>[33]</td><td rowspan=1 colspan=1>adversarial</td><td rowspan=1 colspan=1>convex</td><td rowspan=1 colspan=1>x     x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>Static</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[49]</td><td rowspan=1 colspan=1>adversarial</td><td rowspan=1 colspan=1>convex</td><td rowspan=1 colspan=1>x     x</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>Static</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[5]</td><td rowspan=1 colspan=1>adversarial</td><td rowspan=1 colspan=1>convex</td><td rowspan=1 colspan=1>√     √</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>Static</td><td rowspan=1 colspan=1>separability</td></tr><tr><td rowspan=1 colspan=1>[11]</td><td rowspan=1 colspan=1>adversarial</td><td rowspan=1 colspan=1>convex</td><td rowspan=1 colspan=1>X     x</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>Dynamic</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[9]</td><td rowspan=1 colspan=1>adversarial</td><td rowspan=1 colspan=1>convex</td><td rowspan=1 colspan=1>x      $x$ </td><td rowspan=1 colspan=1> $\overline { { \checkmark } }$ </td><td rowspan=1 colspan=1>Dynamic</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>This work</td><td rowspan=1 colspan=1>adversarial</td><td rowspan=1 colspan=1>convex</td><td rowspan=1 colspan=1> $\checkmark$       $\checkmark$ </td><td rowspan=1 colspan=1> $\checkmark$ </td><td rowspan=1 colspan=1>Dynamic</td><td rowspan=1 colspan=1></td></tr></table>

Table 1: Summary of the most relevant prior works. The current study is the first to consider loss and constraint functions with memory dependency, under universal dynamic regret while using (imperfect) predictions.

The advent of non-stochastic control [60] renewed the interest in OCO-M. In NSC, the action at each round is a weighted sum of the past disturbances, and these weights are learned through OCO-M. Such formulations have a vast range of practical applications, see [4]. However, most NSC works impose deterministic constraints on the state and input [52, 61–63] or assume the constraint at t+1 is revealed before that slot [64]; thus, they do not focus on time-average constraints nor on unknown constraints. Notable exceptions include [35] which analyzes an adversarial COCO-M setting and proves $\mathcal { O } ( T ^ { 2 / 3 } \log ^ { 2 } T )$ static regret and CCV; and [5] that improves these bounds to $\mathcal { O } ( m ^ { 3 / 2 } \sqrt { T \log T } )$ static regret and $\mathcal { O } ( m ^ { 3 / 2 } \sqrt { T \log T } \vee T ^ { 3 / 4 } )$ CCV.

Dynamic regret is significantly less explored for this class of problems, with exceptions confined to the special case of switching costs, cf.[12], [13], or to NSC without budget constraints, e.g., [14], [15], [16]. Therefore, the problem of dynamic regret for COCO-M, and time-average constrained NSC, remains open. Our work aspires to fill this gap.

## A.3 Delay and Predictions in OCO

Two key methods we employ in this work are OCO with delay, which we use to handle memory effects, unearthing an equivalence between these two OCO versions; and optimism, which expedites the learning convergence. Starting with the former, [65] was the first to study the setting where the effect of decision $x _ { t }$ is observed after d rounds, and extended vanilla OCO algorithms to cope with this delay. Significant follow-up works include [18] which handls arbitrary feedback delays, and [66] that improved the regret bounds to $O ( \sqrt { \sum _ { t } d _ { t } } )$ ). These works share the main elegant idea, namely running multiple delay-free copies of a base learning algorithm over the delay window. However, this replication is computationally costly and each copy is trained only on partial observations, as it does not utilize the feedback from every round. Few other works avoid replication and incorporate delays directly into the analysis. For example, [67] derived adaptive FTRL/Mirror-Descent bounds through a single-instance reduction based on prediction drift; and [68] developed a delayed dual averaging with data- and delay-dependent guarantees. Finally, [31] studied the delayed Hedge problem without using replication. All these studies focus on static regret.

More recently, [20] proved a universal dynamic regret bound of $\mathcal { O } ( \sqrt { d T } ( P _ { T } + 1 ) )$ by leveraging the delayed Hedge of [31] to meta-learn over a set of delayed-OGD base learners; and [69] extended this to a decentralized setting. Finally, [70] studied worse-case dynamic regret in COCO with fixed delay. Unlike these important works, our method is adaptive and does not require an a priori Lipschitz bound on the gradients. Besides, our approach differs conceptually as we handle delay through optimism.

Optimistic learning builds on the idea that predictions of forthcoming loss gradients can expedite the learning by reducing the regret bounds commensurately to the accuracy of these predictions, [26]. This framework has been extended in various ways, e.g., by achieving data-adaptive optimistic bounds [27, 28]. Our approach here is motivated by the insightful proposal of [19], that interpreted delayed feedback as a ”bad hint” or, equivalently, used predictions to recover missing gradients in OCO with delay. Here, we extend this idea, for the first time, to the universal dynamic regret setting.

Such demanding benchmarks are indeed rare in optimistic learning, even for the simpler problem without feedback delay. The few exceptions include [21, 22, 71]. However, these works do not consider delayed feedback or memory, nor their extension to these settings is straightforward (see main paper). Hence, our optimistic delayed OCO algorithm is of independent interest.

Summary. To the best of our knowledge, this is the first work to tackle memory and optimism, both in the loss and constraint functions, under universal dynamic regret for adversarial COCO. This result extends the recently-introduced COCO-M problem of [5] to the universal dynamic setting, while being able to drop key assumptions such as the function separability and also expand the benchmark set. Furthermore, our analysis yields several byproducts of independent interest: (i) we enable the study of Non-Stochastic Control with adversarial time-varying constraints under dynamic regret guarantees, whereas prior works focus on static regret [5, 35]; (ii) we establish dynamic regret bounds for optimistic delayed OCO (and delayed COCO), while [20] study dynamic regret for delayed OCO but cannot benefit from predictions; (iii) we extend the “delay-as-optimism” framework of [19] to the dynamic setting by handling the interaction between prediction errors and comparator variation, while other dynamic optimistic OCO algorithms such as [22] do not handle memory nor delay.

## B Proofs of Section 3

## B.1 Proof of Theorem 1

We provide the proof of Theorem 1, which we restate here for the readers convenience.

Let $h _ { t }$ be the delayed gradient feedback and let $\tilde { h } _ { t }$ be its prediction. For any comparator sequence $u _ { 1 } ^ { T } \in \mathcal { U } _ { T }$ , Algorithm 1 ensures dynamic regret:

$$
\mathcal { R } _ { T } ( L ) = \mathcal { O } \left( ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } + \sqrt { 1 + P _ { T } } \right) \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \tilde { h } _ { t } \| ^ { 2 } } \right) .
$$

The proof of this result comprises two stages. First, we bound Term (II) and then Term (I). For the former, we introduce an optimistic dynamic regret variant of OGD that uses adaptive regularization and can handle delays. For Term (I), we adapt the static regret delayed optimistic algorithm of [19].

## B.1.1 Bounding Term(II)

We first analyze the undelayed optimistic problem. This step is useful because the delayed update will later be recovered through the delay-as-optimism reduction. In the dynamic setting, the main challenge is the movement of the comparator. We therefore use the projected one-step geometry, which yields a clean telescoping argument and exposes the additional interaction between the prediction error and the comparator variation. We introduce the following algorithm and respective bound:

Theorem 4 (Optimistic dynamic OGD). Let $( \eta _ { t } ) _ { t \geq 1 }$ be a nondecreasing positive sequence. Let $L _ { t } : \mathcal { X }  \mathbb { R }$ be convex losses, and let $h _ { t } = \nabla L _ { t } \overline { { ( x _ { t } } } )$ . In this theorem, feedback is non-delayed ( after playing $x _ { t }$ the learner observes $h _ { t } )$ . consider the projected optimistic one-step update

$$
x _ { t + 1 } = \Pi _ { X } \left( x _ { t } - { \frac { 1 } { \eta _ { t + 1 } } } { \big ( } h _ { t } + { \tilde { h } } _ { t + 1 } - { \tilde { h } } _ { t } { \big ) } \right) , \qquad x _ { 1 } = \Pi _ { X } \left( x _ { 0 } - { \frac { 1 } { \eta _ { 1 } } } { \tilde { h } } _ { 1 } \right) ,
$$

we have that for any comparator sequence $u _ { 1 } , \dotsc , u _ { T } \in \mathcal { U } _ { T }$ , the regret satisfies:

$$
\mathcal { R } _ { T } ( L ) \leq \eta _ { T + 1 } \Big ( D ^ { 2 } + { \textstyle \frac { 3 } { 2 } } D P _ { T } \Big ) + \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \Bigg ( D \| e _ { t } \| _ { 2 } , \ \frac { 1 } { \eta _ { t + 1 } } \operatorname { h u b e r } \bigl ( \| e _ { t } \| _ { 2 } , \| n _ { t } \| _ { 2 } \bigr ) \Bigg ) - \sum _ { t = 1 } ^ { T } \langle e _ { t } , \delta _ { t } \rangle ,\tag{16}
$$

where $e _ { t } : = h _ { t } - \tilde { h } _ { t } , n _ { t } : = h _ { t } + \tilde { h } _ { t + 1 } - \tilde { h } _ { t } , a n d \delta _ { t } : = u _ { t } - u _ { t + 1 } .$

Theorem 4 will serve as a technical bridge with the delayed analysis. Related optimistic dynamicregret guarantees have been studied before; see, for instance, [71] and, more recently [21] and [22]. However, these results are developed for the standard non-delayed feedback model and cannot be used here. The purpose of Theorem 4 is to derive a non-delayed optimistic dynamic-regret bound in a form compatible with the delayed-upper-bound (DUB) argument, both in the m-step delayed setting of [19] and in the one-step delayed adaptive-tuning argument of [2]. After applying the delay-as-optimism reduction, this structure will allow us to choose the regularization adaptively from observed quantities, avoiding over-regularization and preserving the benefit of accurate predictions. The additional term compared with the static-benchmark case of [19] is $\textstyle - \sum _ { t = 1 } ^ { T } \langle e _ { t } , \delta _ { t } \rangle$ , which couples the prediction error with the movement of the comparator.

We now pass from the undelayed optimistic bound to the delayed setting. By Lemma 2 of [19], delayed feedback can be viewed as optimism with a particular choice of “bad” hint. In our notation, this hint is

$$
\tilde { h } _ { t + 1 } ^ { \mathrm { b a d } } : = \tilde { h } _ { t - m + 1 : t + 1 } - h _ { t - m + 1 : t } .
$$

Applying this representation to the bound in (16), we obtain the same form of guarantee after replacing $e _ { t }$ and $n _ { t }$ by

$$
\bar { e } _ { t } : = h _ { t - m : t } - \tilde { h } _ { t - m : t } , \qquad \bar { n } _ { t } : = h _ { t - m } + \tilde { h } _ { t - m + 1 : t + 1 } - \tilde { h } _ { t - m : t } .
$$

The algebraic verification of this substitution is deferred to Appendix B.2.2, using Corollary 3 presented there.

The genuinely new term in the dynamic optimistic analysis is

$$
- \sum _ { t = 1 } ^ { T } \langle \bar { e } _ { t } , \delta _ { t } \rangle , \qquad \delta _ { t } : = u _ { t } - u _ { t + 1 } .
$$

It appears because the comparator used here changes from $u _ { t }$ to $u _ { t + 1 }$ . In a standard non-optimistic analysis, this change only contributes through the path length. In the optimistic analysis, however, the update direction depends on predictions, so the movement of the comparator interacts with the prediction error $\bar { e } _ { t }$ . The term $- \langle \bar { e } _ { t } , \delta _ { t } \rangle$ therefore measures whether the prediction error helps or hurts the tracking of the moving comparator. However this term is yet controlled and upper bounded by the desired bound that is of order $\sqrt { T }$ in the worst case and shrinks with prefect prediction and/or with static benchmark to be zero.

Corollary 2 (DUB-tuned bound for Term (II)). Let

$$
a _ { t } : = D \| \bar { e } _ { t } \| _ { 2 } , \qquad b _ { t } : = \mathrm { h u b e r } ( \| \bar { e } _ { t } \| _ { 2 } , \| \bar { n } _ { t } \| _ { 2 } ) ,
$$

Assume that the base learner uses the DUB-tuned regularization

$$
\eta _ { t + 1 } = \frac { 2 } { \alpha } \operatorname* { m a x } _ { j \leq t - m - 1 } a _ { j - m + 1 : j } + \frac { 1 } { \alpha } \sqrt { \sum _ { i = 1 } ^ { t - m } \bigl ( a _ { i } ^ { 2 } + 2 \alpha b _ { i } \bigr ) } , \quad w i t h \alpha : = D \bigl ( D + 2 P _ { T } \bigr ) .\tag{17}
$$

Then the delayed dynamic regret ofthe best-tuned base learner satisfies

$$
\mathrm { T e r m } \left( \mathrm { I I } \right) \le 4 ( m + 1 ) ^ { 3 / 2 } \left( D + \sqrt { D ( D + P _ { T } ) } \right) \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \tilde { h } _ { t } \| ^ { 2 } } .
$$

Compared with [20], our method is adaptive and does not require an a priori Lipschitz bound on the gradients. The standard black-box reduction of [18] handles delays by running multiple delay-free copies of a base algorithm, but this replication can be computationally costly and each copy is trained only on a partial subsequence of the observations. Other works avoid replication and incorporate delays directly into the analysis. For example, [67] derives adaptive FTRL/Mirror-Descent bounds through a single-instance reduction based on prediction drift. More recently, [68] develops delayed dual averaging with data- and delay-dependent guarantees, including an optimistic variant. These works, however, focus on static regret. Our contribution is complementary. Building on the optimistic delayed-OCO perspective of [19], we develop delayed optimistic guarantees for a dynamic comparator sequence. This distinction is important because static-regret techniques do not directly control the path length of a moving comparator. We then use these dynamic delayed-OCO guarantees as a bridge to constrained OCO with memory through the credit-gradient reduction.

## B.1.2 Bounding Term(I)

We now turn our focus to Term(I) and bound it using an optimistic delayed AdaHedge algorithm. Using $x _ { t } = X _ { t } w _ { t }$ and $\boldsymbol { x _ { t } ^ { i } } = \boldsymbol { X _ { t } } \boldsymbol { e _ { i } }$ , it follows that $x _ { t } - x _ { t } ^ { i } = X _ { t } ( w _ { t } - e _ { i } )$ , thus we can write:

$$
\mathrm { T e r m } \left( \mathrm { I } \right) \leq \sum _ { t = 1 } ^ { T } \langle M _ { t } , \ w _ { t } - e _ { i } \rangle , \quad \mathrm { w h e r e } \ M _ { t } : = X _ { t } ^ { \top } h _ { t } .
$$

This is the meta-regret of the delayed expert-combination scheme. This part follows from the delayed optimistic OCO framework of [19] applied to the linear meta-losses $\dot { M _ { t } } = X _ { t } ^ { \top } h _ { t }$

Lemma 3 (Bound on Term (I)). Under the assumptions above, and using the static delayed optimistic FTRL meta-learner of[19, Theorem $1 2 ] , f o r$ every expert $i \in [ N ]$ , we have:

$$
T e r m \left( I \right) \leq 8 D ( m + 1 ) ^ { \frac { 3 } { 2 } } \left( 1 + \sqrt { 1 + \log N } \right) \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \tilde { h } _ { t } \| ^ { 2 } } .\tag{18}
$$

• Finally, the proof of Theorem 1 is obtained by recalling that:

$$
\mathcal { R } _ { T } ( L ) \leq \mathcal { R } _ { T } ( \ell ) = \mathrm { T e r m { } ( I ) } + \mathrm { T e r m { } ( I I ) } ,
$$

and hence the results follows by combining Lemma 3 with Corollary 2.

Discussion The factor $( m + 1 ) ^ { 3 / 2 }$ in Theorem 1 should be understood as the price of obtaining a prediction-adaptive bound under delayed feedback. The factor m + 1 comes from the block prediction error $\bar { e } _ { t } = h _ { t - m : t } - \tilde { h } _ { t - m : t } .$ , whereas the additional $\sqrt { m }$ comes from the delayed-upper-bound (DUB) argument. More precisely, it appears when controlling the unrevealed window term

$$
\operatorname* { m a x } _ { t \in [ T ] } a _ { t - m : t - 1 } \leq D \sqrt { m } \sqrt { \sum _ { t = 1 } ^ { T } \| \bar { e } _ { t } \| ^ { 2 } } .
$$

This term is the cost of choosing the regularization adaptively from delayed observations, which avoids over-regularization and preserves the benefit of accurate predictions.

If predictions are ignored and a uniform gradient bound is available, one can instead use a fixed worst-case tuning and recover a delayed dynamic-regret bound with linear dependence on $m + 1$ Thus, the $( m + 1 { \bar { ) } } ^ { 3 / 2 }$ dependence is not intrinsic to delayed OCO itself, but to the adaptive optimistic tuning under delay.

We also note that the sharper bounds in (35) and (33) retain more information than the simplified $( m + 1 ) ^ { 3 / 2 }$ statement. In particular, within the prediction-adaptive bound itself, the additional $\sqrt { m }$ factor in Term (I) can be dominated by the meta-complexity term when $m = O ( \log N )$ , while the corresponding term in Term (II) can be dominated by the path-length term when $P _ { T } \gtrsim D m$

Remark 6. One may alternatively implement the adaptive DUB tuning of[19, Theorem 12], which yields the same theoretical order, but yields for better performance in practice.

## B.2 Auxiliary Results and Proofs for Theorem 1 and Section 3

This subsection provides the intermediate results that are required for the proof of the main result of Section 3, namely of Theorem 1 that we proved in the previous subsection.

## B.2.1 Proof of Theorem 4

We remind the reader that this theorem assumes no delays, and thus after committing $x _ { t }$ at round $t ,$ the gradient $h _ { t } = \nabla _ { x _ { t } } L _ { t } ( x _ { t } )$ is revealed directly and there is no delay of m steps.

Recall that:

$$
\begin{array} { r } { e _ { t } : = h _ { t } - \tilde { h } _ { t } , \qquad n _ { t } : = h _ { t } + \tilde { h } _ { t + 1 } - \tilde { h } _ { t } , \qquad v _ { t } : = x _ { t } - x _ { t + 1 } , \qquad \delta _ { t } : = u _ { t } - u _ { t + 1 } , } \end{array}
$$

Proof. Since each $L _ { t }$ is convex and $h _ { t } = \nabla _ { x _ { t } } L _ { t } ( x _ { t } )$ , we have

$$
L _ { t } ( x _ { t } ) - L _ { t } ( u _ { t } ) \leq \langle h _ { t } , x _ { t } - u _ { t } \rangle , \qquad \mathrm { a n d ~ t h u s } \qquad \mathcal { R } _ { T } ( L ) \leq \sum _ { t = 1 } ^ { T } \langle h _ { t } , x _ { t } - u _ { t } \rangle .
$$

Step 1: Exact decomposition. For each $t \in \{ 1 , \ldots , T \}$ , we have

$$
\langle h _ { t } , x _ { t } - u _ { t } \rangle = \langle e _ { t } , v _ { t } \rangle + \langle n _ { t } , x _ { t + 1 } - u _ { t } \rangle + \langle \tilde { h } _ { t } , x _ { t } - u _ { t } \rangle - \langle \tilde { h } _ { t + 1 } , x _ { t + 1 } - u _ { t } \rangle .\tag{19}
$$

Summing (19) over $t = 1 , \dots , T$ yields

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \langle h _ { t } , x _ { t } - u _ { t } \rangle = \displaystyle \sum _ { t = 1 } ^ { T } \langle n _ { t } , x _ { t + 1 } - u _ { t } \rangle + \displaystyle \sum _ { t = 1 } ^ { T } \langle e _ { t } , v _ { t } \rangle } \\ { \displaystyle \qquad + \displaystyle \sum _ { t = 1 } ^ { T } \Big ( \langle \tilde { h } _ { t } , x _ { t } - u _ { t } \rangle - \langle \tilde { h } _ { t + 1 } , x _ { t + 1 } - u _ { t } \rangle \Big ) . } \end{array}\tag{20}
$$

Now the last sum telescopes as follows:

$$
\sum _ { t = 1 } ^ { T } \Big ( \langle \tilde { h } _ { t } , x _ { t } - u _ { t } \rangle - \langle \tilde { h } _ { t + 1 } , x _ { t + 1 } - u _ { t } \rangle \Big ) = \langle \tilde { h } _ { 1 } , x _ { 1 } - u _ { 1 } \rangle + \sum _ { t = 1 } ^ { T } \langle \tilde { h } _ { t + 1 } , \delta _ { t } \rangle ,\tag{21}
$$

where the last equality uses $\delta _ { T } = 0$ and $\tilde { h } _ { T + 1 } = 0$ . Substituting (21) into (20), we obtain:

$$
\sum _ { t = 1 } ^ { T } \langle h _ { t } , x _ { t } - u _ { t } \rangle = \sum _ { t = 1 } ^ { T } \langle n _ { t } , x _ { t + 1 } - u _ { t } \rangle + \sum _ { t = 1 } ^ { T } \bigl ( \langle e _ { t } , v _ { t } \rangle + \langle \tilde { h } _ { t + 1 } , \delta _ { t } \rangle \bigr ) + \langle \tilde { h } _ { 1 } , x _ { 1 } - u _ { 1 } \rangle .\tag{22}
$$

Since $n _ { t } = e _ { t } + \tilde { h } _ { t + 1 }$ , we have:

$$
\langle e _ { t } , v _ { t } \rangle + \langle \tilde { h } _ { t + 1 } , \delta _ { t } \rangle = \langle e _ { t } , v _ { t } - \delta _ { t } \rangle + \langle n _ { t } , \delta _ { t } \rangle .
$$

Hence, it is:

$$
\sum _ { t = 1 } ^ { T } \langle h _ { t } , x _ { t } - u _ { t } \rangle = \sum _ { t = 1 } ^ { T } \langle n _ { t } , x _ { t + 1 } - u _ { t + 1 } \rangle + \sum _ { t = 1 } ^ { T } \langle e _ { t } , v _ { t } - \delta _ { t } \rangle + \langle \widetilde { h } _ { 1 } , x _ { 1 } - u _ { 1 } \rangle .\tag{23}
$$

Step 2: One-step projected inequality. By the Euclidean projection optimality condition, it is:

$$
\left. x _ { t } - { \frac { 1 } { \eta _ { t + 1 } } } n _ { t } - x _ { t + 1 } , u _ { t + 1 } - x _ { t + 1 } \right. \leq 0 .
$$

Rearranging, gives:

$$
\langle n _ { t } , x _ { t + 1 } - u _ { t + 1 } \rangle \leq \frac { \eta _ { t + 1 } } { 2 } \Big ( \| x _ { t } - u _ { t + 1 } \| ^ { 2 } - \| x _ { t + 1 } - u _ { t + 1 } \| ^ { 2 } - \| v _ { t } \| ^ { 2 } \Big ) .\tag{2}
$$

Substituting this into (23), we obtain:

$$
\sum _ { t = 1 } ^ { T } \langle h _ { t } , x _ { t } - u _ { t } \rangle \leq A _ { T } + \sum _ { t = 1 } ^ { T } \left( \langle e _ { t } , v _ { t } - \delta _ { t } \rangle - \frac { \eta _ { t + 1 } } { 2 } \| v _ { t } \| ^ { 2 } \right) + \langle \tilde { h } _ { 1 } , x _ { 1 } - u _ { 1 } \rangle ,\tag{24}
$$

where

$$
A _ { T } : = \sum _ { t = 1 } ^ { T } \frac { \eta _ { t + 1 } } { 2 } \Big ( \| x _ { t } - u _ { t + 1 } \| ^ { 2 } - \| x _ { t + 1 } - u _ { t + 1 } \| ^ { 2 } \Big ) .
$$

Step 3: Local bound preserving the Huber structure. Rewrite the local term in (24) as:

$$
\langle e _ { t } , v _ { t } - \delta _ { t } \rangle - \frac { \eta _ { t + 1 } } { 2 } \| v _ { t } \| ^ { 2 } = \Big ( \langle e _ { t } , v _ { t } \rangle - \frac { \eta _ { t + 1 } } { 2 } \| v _ { t } \| ^ { 2 } \Big ) - \langle e _ { t } , \delta _ { t } \rangle .\tag{25}
$$

We first bound the bracketed term as in the proof of Theorem 16 of [19]. Since:

$$
x _ { t + 1 } = \Pi _ { { \mathcal { X } } } \left( x _ { t } - \frac { 1 } { \eta _ { t + 1 } } n _ { t } \right) \qquad \mathrm { a n d } \qquad x _ { t } = \Pi _ { { \mathcal { X } } } ( x _ { t } ) ,
$$

the nonexpansiveness of Euclidean projection implies that:

$$
\| v _ { t } \| = \| x _ { t } - x _ { t + 1 } \| \leq \frac { 1 } { \eta _ { t + 1 } } \| n _ { t } \| .
$$

Also, since $x _ { t } , x _ { t + 1 } \in \mathcal { X }$ and diam $. ( \mathcal { X } ) = D$ , it is $\| v _ { t } \| \leq D$ . Hence, we can write:

$$
\left\| v _ { t } \right\| \leq \operatorname* { m i n } \left( D , \frac { \| n _ { t } \| } { \eta _ { t + 1 } } \right) .
$$

Applying Lemmas 17–18 of [19] with the Euclidean norm, we obtain:

$$
\langle e _ { t } , v _ { t } \rangle - \frac { \eta _ { t + 1 } } { 2 } \| v _ { t } \| ^ { 2 } \leq \operatorname* { m i n } \biggl ( D \| e _ { t } \| , ~ \frac { 1 } { \eta _ { t + 1 } } \mathrm { h u b e r } \bigl ( \| e _ { t } \| , \| n _ { t } \| \bigr ) \biggr ) .\tag{26}
$$

Therefore, we can write:

$$
\langle e _ { t } , v _ { t } - \delta _ { t } \rangle - \frac { \eta _ { t + 1 } } { 2 } \| v _ { t } \| ^ { 2 } \leq \operatorname* { m i n } \biggl ( D \| e _ { t } \| , \ \frac { 1 } { \eta _ { t + 1 } } \operatorname { h u b e r } \bigl ( \| e _ { t } \| , \| n _ { t } \| \bigr ) \biggr ) - \langle e _ { t } , \delta _ { t } \rangle .\tag{27}
$$

Step 4: Bound the weighted telescope $A _ { T }$ . We rewrite $A _ { T }$ as

$$
A _ { T } = \frac { \eta _ { 2 } } { 2 } \left. x _ { 1 } - u _ { 2 } \right. ^ { 2 } - \frac { \eta _ { T + 1 } } { 2 } \left. x _ { T + 1 } - u _ { T + 1 } \right. ^ { 2 } + \sum _ { t = 2 } ^ { T } \biggl [ \frac { \eta _ { t + 1 } - \eta _ { t } } { 2 } \left. x _ { t } - u _ { t + 1 } \right. ^ { 2 } + \frac { \eta _ { t } } { 2 } \left( \left. x _ { t } - u _ { t + 1 } \right. ^ { 2 } - \left. x _ { t } - u _ { t } \right. ^ { 2 } \right) \biggr ]
$$

Since $\{ \eta _ { t } \}$ is nondecreasing and $\| x _ { t } - u _ { t + 1 } \| \leq D$ , the first and third terms yield:

$$
\frac { \eta _ { 2 } } { 2 } \| x _ { 1 } - u _ { 2 } \| ^ { 2 } + \sum _ { t = 2 } ^ { T } \frac { \eta _ { t + 1 } - \eta _ { t } } { 2 } \| x _ { t } - u _ { t + 1 } \| ^ { 2 } \leq \frac { \eta _ { T + 1 } D ^ { 2 } } { 2 } .
$$

Moreover, we can write:

$$
\begin{array} { r } { \| x _ { t } - u _ { t + 1 } \| ^ { 2 } - \| x _ { t } - u _ { t } \| ^ { 2 } = 2 \langle x _ { t } - u _ { t } , u _ { t } - u _ { t + 1 } \rangle + \| \delta _ { t } \| ^ { 2 } \leq 2 D \| \delta _ { t } \| + \| \delta _ { t } \| ^ { 2 } . } \end{array}
$$

Thus, we get:

$$
A _ { T } \leq \frac { \eta _ { T + 1 } D ^ { 2 } } { 2 } + \eta _ { T + 1 } \sum _ { t = 1 } ^ { T - 1 } \left( D \| \delta _ { t } \| + \frac 1 2 \| \delta _ { t } \| ^ { 2 } \right) .\tag{28}
$$

Step 5: Final bound. Substituting (27), (28), and the initialization estimate

$$
\langle \tilde { h } _ { 1 } , x _ { 1 } - u _ { 1 } \rangle \leq \frac { \eta _ { T + 1 } D ^ { 2 } } { 2 } , \quad \mathrm { i n t o } \quad ( 2 4 ) \quad \mathrm { w e ~ o b t a i n } \mathrm { . }
$$

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \langle h _ { t } , x _ { t } - u _ { t } \rangle \leq \frac { \eta _ { T + 1 } D ^ { 2 } } { 2 } + \eta _ { T + 1 } \sum _ { t = 1 } ^ { T - 1 } \Bigg ( D \| \delta _ { t } \| + \frac { 1 } { 2 } \| \delta _ { t } \| ^ { 2 } \Bigg ) } \\ { \displaystyle \qquad + \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \Bigg ( D \| e _ { t } \| , \ \frac { 1 } { \eta _ { t + 1 } } \operatorname { h u b e r } ( \| e _ { t } \| , \| n _ { t } \| ) \Bigg ) - \sum _ { t = 1 } ^ { T } \langle e _ { t } , \delta _ { t } \rangle + \frac { \eta _ { T + 1 } D ^ { 2 } } { 2 } . } \end{array}
$$

Therefore, it holds:

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \langle h _ { t } , x _ { t } - u _ { t } \rangle \leq \eta _ { T + 1 } \Big ( D ^ { 2 } + \frac { 3 } { 2 } D P _ { T } \Big ) + \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \Biggl ( D \| e _ { t } \| , ~ \frac { 1 } { \eta _ { t + 1 } } \operatorname { h u b e r } \bigl ( \| e _ { t } \| , \| n _ { t } \| \bigr ) \Biggr ) } \\ { \displaystyle \qquad - \sum _ { t = 1 } ^ { T } \langle e _ { t } , \delta _ { t } \rangle . } \end{array}
$$

And this proves the claim.

## B.2.2 Delay as optimism

We now transfer the previous undelayed optimistic bound to the delayed-feedback setting. The key observation, following Lemma 2 of [19], is that a delayed optimistic update can be written as an undelayed optimistic one-step update with a particular choice of “bad” hint. This allows us to reuse Theorem 4 without reproving the whole dynamic regret argument. In the delayed setting, the gradient $h _ { t } \in \nabla L _ { t } ( x _ { t } )$ is revealed only at time $t + m$ . Throughout the delayed analysis, we use the convention

$$
h _ { r } = \widetilde { h } _ { r } = 0 , \qquad r \leq 0 ,
$$

and empty sums are set to zero. Thus, the first m rounds are warm-start rounds: the corresponding decisions are included in the regret, but no genuine delayed feedback has yet been revealed. Define the bad hint

$$
\tilde { h } _ { t + 1 } ^ { \mathrm { b a d } } : = \tilde { h } _ { t - m + 1 : t + 1 } - h _ { t - m + 1 : t } .
$$

Then the optimistic one-step update

$$
x _ { t + 1 } = \Pi _ { \mathcal { X } } \bigg ( x _ { t } - \frac { 1 } { \eta _ { t + 1 } } \big ( h _ { t } + \tilde { h } _ { t + 1 } ^ { \mathrm { b a d } } - \tilde { h } _ { t } ^ { \mathrm { b a d } } \big ) \bigg )
$$

is equivalent to the delayed optimistic update

$$
x _ { t + 1 } = \Pi _ { \mathcal { X } } \Bigg ( x _ { t } - \frac { 1 } { \eta _ { t + 1 } } \big ( h _ { t - m } + \widetilde { h } _ { t - m + 1 : t + 1 } - \widetilde { h } _ { t - m : t } \big ) \Bigg ) .
$$

Indeed, the update uses the most recent revealed gradient $h _ { t - m }$ , together with predictions for the missing gradients.

Corollary 3 (Dynamic delayed optimistic OGD). As a direct application of Lemma $2 \ [ I 9 ] ,$ let $( \eta _ { t } ) _ { t \geq 1 }$ be a nondecreasing positive sequence, and consider the delayed optimistic update

$$
x _ { t + 1 } = \Pi _ { \mathcal { X } } \left( x _ { t } - \frac { 1 } { \eta _ { t + 1 } } \big ( h _ { t - m } + \tilde { h } _ { t - m + 1 : t + 1 } - \tilde { h } _ { t - m : t } \big ) \right) , \qquad t = 1 , \ldots , T .
$$

For any comparator sequence $u _ { 1 } ^ { T } \in \mathcal { X } ^ { T }$ , define:

$$
P _ { T } : = \sum _ { t = 1 } ^ { T - 1 } \| u _ { t + 1 } - u _ { t } \| , \qquad \delta _ { t } : = u _ { t } - u _ { t + 1 } ,
$$

with the convention $u _ { T + 1 } : = u _ { T }$ . Also define:

$$
\bar { e } _ { t } : = h _ { t - m : t } - \tilde { h } _ { t - m : t } , \quad a n d \quad \bar { n } _ { t } : = h _ { t - m } + \tilde { h } _ { t - m + 1 : t + 1 } - \tilde { h } _ { t - m : t } .
$$

Then the dynamic regret

$$
\mathcal { R } _ { T } ( L ) : = \sum _ { t = 1 } ^ { T } L _ { t } ( x _ { t } ) - \sum _ { t = 1 } ^ { T } L _ { t } ( u _ { t } )
$$

satisfies

$$
\begin{array} { l } { \displaystyle \mathcal { R } _ { T } ( L ) \leq \eta _ { T + 1 } \Big ( D ^ { 2 } + \frac { 3 } { 2 } D P _ { T } \Big ) } \\ { \displaystyle \qquad + \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \Biggl \{ D \| \bar { e } _ { t } \| , \ \frac { 1 } { \eta _ { t + 1 } } \mathrm { h u b e r } \big ( \| \bar { e } _ { t } \| , \| \bar { n } _ { t } \| \big ) \Biggr \} - \sum _ { t = 1 } ^ { T } \langle \bar { e } _ { t } , \delta _ { t } \rangle . } \end{array}\tag{29}
$$

## B.2.3 Proof of Corollary 2

From the delayed dynamic bound, obtained by combining Theorem 4 with the delay to optimism reduction (with bad hints), we have

$$
\mathrm { T e r m } \left( \Pi \right) \le \eta _ { T + 1 } D \left( D + 2 P _ { T } \right) + \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \left\{ D \| \bar { e } _ { t } \| , \frac { 1 } { \eta _ { t + 1 } } \mathrm { h u b e r } ( \| \bar { e } _ { t } \| , \| \bar { n } _ { t } \| ) \right\} - \sum _ { t = 1 } ^ { T } \langle \bar { e } _ { t } , \delta _ { t } \rangle .\tag{30}
$$

The first two terms have the form required by the delayed-upper-bound Lemma 22 of [19], with the regularization scale now depending on the path length. The last term is specific to the dynamic comparator and will be controlled separately by introducing an axillary variable $\beta .$

For any $\beta > 0$ , Young’s inequality gives

$$
- \langle \bar { e } _ { t } , \delta _ { t } \rangle \leq \frac { 1 } { 2 \beta } \| \bar { e } _ { t } \| ^ { 2 } + \frac { \beta } { 2 } \| \delta _ { t } \| ^ { 2 } .
$$

Summing over t, and taking $\beta = \sqrt { \frac { \sum _ { t = 1 } ^ { T } | | \bar { e } _ { t } | | ^ { 2 } } { \sum _ { t = 1 } ^ { T } | | \delta _ { t } | | ^ { 2 } } }$ , we can write:

$$
- \sum _ { t = 1 } ^ { T } \langle \bar { e } _ { t } , \delta _ { t } \rangle \leq \sqrt { \left( \sum _ { t = 1 } ^ { T } \| \bar { e } _ { t } \| ^ { 2 } \right) \left( \sum _ { t = 1 } ^ { T } \| \delta _ { t } \| ^ { 2 } \right) } .
$$

Therefore we can update the bound of the term as follows:

$$
\mathrm { T e r m } \left( \Pi \right) \le \eta _ { T + 1 } D \left( D + 2 P _ { T } \right) + \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \left\{ a _ { t } , \frac { b _ { t } } { \eta _ { t + 1 } } \right\} + \sqrt { \left( \sum _ { t = 1 } ^ { T } \| \bar { e } _ { t } \| ^ { 2 } \right) \left( \sum _ { t = 1 } ^ { T } \| \delta _ { t } \| ^ { 2 } \right) } ,\tag{31}
$$

where

$$
a _ { t } : = D \| \bar { e } _ { t } \| , \qquad b _ { t } : = \mathrm { h u b e r } ( \| \bar { e } _ { t } \| , \| \bar { n } _ { t } \| ) .
$$

We now apply the DUB tuning Lemma with $\alpha = D \big ( D + 2 P _ { T } \big )$ , and step:

$$
\eta _ { t + 1 } = \frac { 2 } { \alpha } \operatorname* { m a x } _ { j \leq t - m - 1 } a _ { j - m + 1 : j } + \frac { 1 } { \alpha } \sqrt { \sum _ { i = 1 } ^ { t - m } a _ { i } ^ { 2 } + 2 \alpha b _ { i } } ,
$$

where quantities involving negative indices are interpreted according to the zero-padding convention stated above.

We allow $\eta _ { t }$ to be dependent on $P _ { T }$ here because this bound corresponds to the base learner tuned to the correct path-length scale. This gives:

$$
\eta _ { T + 1 } D \big ( D + 2 P _ { T } \big ) + \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \Bigg \{ a _ { t } , \frac { b _ { t } } { \eta _ { t + 1 } } \Bigg \} \leq 2 \left( 2 \operatorname* { m a x } _ { t \in [ T ] } a _ { t - m : t - 1 } + \sqrt { \sum _ { t = 1 } ^ { T } ( a _ { t } ^ { 2 } + 2 \alpha b _ { t } ) } \right) ,\tag{32}
$$

because $D ( D + 2 P _ { T } ) / \alpha = 1$ . We next bound the two terms in (32). First, using:

$$
\mathrm { h u b e r } ( x , y ) \leq \frac { 1 } { 2 } x ^ { 2 } , \mathrm { w e \ o b t a i n } \colon b _ { t } \leq \frac { 1 } { 2 } \| { \bar { e } } _ { t } \| ^ { 2 } .
$$

Hence, we can write:

$$
a _ { t } ^ { 2 } + 2 \alpha b _ { t } \leq D ^ { 2 } \lVert \bar { e } _ { t } \rVert ^ { 2 } + \alpha \lVert \bar { e } _ { t } \rVert ^ { 2 } = ( D ^ { 2 } + \alpha ) \lVert \bar { e } _ { t } \rVert ^ { 2 } ,
$$

and therefore,

$$
\sum _ { t = 1 } ^ { T } ( a _ { t } ^ { 2 } + 2 \alpha b _ { t } ) \leq ( D ^ { 2 } + \alpha ) \sum _ { t = 1 } ^ { T } \| { \bar { e } } _ { t } \| ^ { 2 } .
$$

Second, by Cauchy–Schwarz,

$$
a _ { t - m : t - 1 } = \sum _ { j = \operatorname* { m a x } \{ 1 , t - m \} } ^ { t - 1 } a _ { j } \leq \sqrt { m } \left( \sum _ { j = \operatorname* { m a x } \{ 1 , t - m \} } ^ { t - 1 } a _ { j } ^ { 2 } \right) ^ { 1 / 2 } .
$$

Since $a _ { j } = D \| \bar { e } _ { j } \|$ , we get

$$
\operatorname* { m a x } _ { t \in [ T ] } a _ { t - m : t - 1 } \leq D \sqrt { m } \sqrt { \sum _ { t = 1 } ^ { T } \| \bar { e } _ { t } \| ^ { 2 } } .
$$

Combining the previous estimates yields:

$$
\eta _ { T + 1 } D ( D + 2 P _ { T } ) + \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \left\{ a _ { t } , \frac { b _ { t } } { \eta _ { t + 1 } } \right\} \leq \left( 4 D \sqrt { m } + 2 \sqrt { D ^ { 2 } + \alpha } \right) \sqrt { \sum _ { t = 1 } ^ { T } \| \bar { e } _ { t } \| ^ { 2 } } .
$$

Since $\alpha = D ( D + 2 P _ { T } )$ , we have $D ^ { 2 } + \alpha = 2 D ( D + P _ { T } )$ . Thus:

$$
2 \sqrt { D ^ { 2 } + \alpha } = 2 \sqrt { 2 D ( D + P _ { T } ) } \leq 3 \sqrt { D ( D + P _ { T } ) } .
$$

It remains to bound the interaction term. Since it is $\| \delta _ { t } \| \leq D$ , we can write:

$$
\sum _ { t = 1 } ^ { T } \| \delta _ { t } \| ^ { 2 } \leq D \sum _ { t = 1 } ^ { T } \| \delta _ { t } \| = D P _ { T } .
$$

Therefore,

$$
\sqrt { \left( \sum _ { t = 1 } ^ { T } \lVert \bar { e } _ { t } \rVert ^ { 2 } \right) \left( \sum _ { t = 1 } ^ { T } \lVert \delta _ { t } \rVert ^ { 2 } \right) } \leq \sqrt { D P _ { T } } \sqrt { \sum _ { t = 1 } ^ { T } \lVert \bar { e } _ { t } \rVert ^ { 2 } } .
$$

Since it is:

$$
\sqrt { D P _ { T } } \leq \sqrt { D ( D + P _ { T } ) } ,
$$

we obtain the looser, but cleaner, bound:

$$
\mathrm { T e r m } \left( \mathrm { I I } \right) \leq 4 \left( D \sqrt { m } + \sqrt { D ( D + P _ { T } ) } \right) \sqrt { \sum _ { t = 1 } ^ { T } \| \bar { e } _ { t } \| ^ { 2 } } .
$$

Finally, given that it holds:

$$
\| \bar { e } _ { t } \| ^ { 2 } \leq \left( m + 1 \right) \sum _ { s = t - m } ^ { t } \| h _ { s } - \tilde { h } _ { s } \| ^ { 2 } ,
$$

we get,

$$
\mathrm { T e r m \left( I I \right) } \le 4 ( m + 1 ) \left( D \sqrt { m } + \sqrt { D ( D + P _ { T } ) } \right) \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \tilde { h } _ { t } \| ^ { 2 } } .\tag{33}
$$

a cleaner term:

$$
\mathrm { T e r m } \left( \mathrm { I I } \right) \le 4 ( m + 1 ) ^ { 3 / 2 } \left( D + \sqrt { D ( D + P _ { T } ) } \right) \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \tilde { h } _ { t } \| ^ { 2 } } .
$$

This proves the corollary.

## B.2.4 Proof of Lemma 3

Lemma 3 bounds Term (I) using the optimistic delayed AdaHedge meta-learner of [19]. At round $t ,$ the meta-loss is the linear function

$$
\langle M _ { t } , w \rangle , \qquad M _ { t } : = X _ { t } ^ { \top } h _ { t } .
$$

Thus, its actual and predicted gradients are

$$
M _ { t } = X _ { t } ^ { \top } h _ { t } , \qquad \widetilde { M } _ { t } : = X _ { t } ^ { \top } \widetilde { h } _ { t } .
$$

In the delayed setting, the meta-learner needs a prediction of the unavailable block of gradient meta-losses. Thus, at the time when $w _ { t + 1 }$ is computed, we use the block hint:

$$
\tilde { M } _ { t - m + 1 : t + 1 } : = \sum _ { s = t - m + 1 } ^ { t + 1 } \tilde { M } _ { s } = \sum _ { s = t - m + 1 } ^ { t + 1 } X _ { s } ^ { \top } \tilde { h } _ { s } ,
$$

which predicts

$$
M _ { t - m + 1 : t + 1 } : = \sum _ { s = t - m + 1 } ^ { t + 1 } M _ { s } .
$$

Importantly, only the gradient-related quantities $h _ { s }$ need to be predicted. The matrices $X _ { s }$ are known when the corresponding meta-update is performed. In particular, after processing the feedback available at round t, each base learner outputs its candidate decision for round $t + 1$ , so:

$$
X _ { t + 1 } = [ x _ { t + 1 } ^ { 1 } , \dots , x _ { t + 1 } ^ { N } ]
$$

is available before the meta-learner computes Wt+ $w _ { t + 1 }$ . Therefore, there is no need to predict $X _ { t + 1 }$ . We now define the meta-loss prediction error

$$
\varepsilon _ { t } : = \left\| \sum _ { s = t - m } ^ { t } ( \tilde { M } _ { s } - M _ { s } ) \right\| _ { \infty } .
$$

As before, we use the convention $M _ { r } = \widetilde { M } _ { r } = 0 { \mathrm { f o r } } r \leq 0$ . We first bound Term (I) in terms of $\varepsilon _ { t } .$ and later relate $\varepsilon _ { t }$ to prediction errors $E _ { T } ( f )$ and $E _ { T } ( g ^ { + } )$ ). Specifically, we want to bound:

$$
\mathrm { T e r m \left( I \right) } = \sum _ { t = 1 } ^ { T } \langle M _ { t } , \ w _ { t } - e _ { i } \rangle .
$$

In the considered delayed setting, at round $t + 1$ only the losses $M _ { 1 : t - m } : = ( M _ { 1 } , \dots , M _ { t - m } )$ are revealed, while the missing block $M _ { t - m + 1 : t + 1 }$ may be predicted by a hint. For each $t \in [ T ]$ , define

$$
b _ { t } : = \mathrm { h u b e r } \left( \left. \sum _ { s = t - m } ^ { t } ( \tilde { M } _ { s } - M _ { s } ) \right. _ { \infty } , \left. M _ { t } \right. _ { \infty } \right) \ \mathrm { ~ a n d ~ } a _ { t } : = 2 \operatorname* { m i n } \left\{ \left. \sum _ { s = t - m } ^ { t } ( \tilde { M } _ { s } - M _ { s } ) \right. _ { \infty } , \left. M _ { t } \right. _ { \infty } \right\}
$$

where we use the $\ell _ { 1 } { \mathrm { - g e o m e t r y } }$ on $\mathcal { W } = \Delta _ { N }$ , so that $\mathrm { d i a m } ( \mathcal { W } ) = 2$

Then, by Theorem 12 of [19], we can write:

$$
\mathrm { T e r m \left( I \right) } \leq \left( \frac { \psi ( e _ { i } ) } { \alpha } + 1 \right) \left( 2 \operatorname* { m a x } _ { t \in \left[ T \right] } a _ { t - m : t - 1 } + \sqrt { \sum _ { t = 1 } ^ { T } \Bigl ( a _ { t } ^ { 2 } + 2 \alpha b _ { t } \Bigr ) } \right) .\tag{34}
$$

We use the shifted negative-entropy regularizer:

$$
\psi ( w ) = \sum _ { j = 1 } ^ { N } w _ { j } \log ( w _ { j } ) + \log N , \qquad w \in \Delta _ { N } ,
$$

and choose $\begin{array} { r } { \alpha = \operatorname* { s u p } _ { w \in \Delta _ { N } } \psi ( w ) = \log N } \end{array}$ . Since $\psi ( e _ { i } ) = \log N$ , we have that:

$$
{ \frac { \psi ( e _ { i } ) } { \alpha } } + 1 = 2 .
$$

Then we can write

$$
a _ { t , F } = 2 \operatorname* { m i n } \{ \varepsilon _ { t } , \| M _ { t } \| _ { \infty } \} \leq 2 \varepsilon _ { t } . \quad \mathrm { a n d } \quad b _ { t } = \mathrm { h u b e r } ( \varepsilon _ { t } , \| M _ { t } \| _ { \infty } ) \leq \frac { 1 } { 2 } \varepsilon _ { t } ^ { 2 } .
$$

Therefore,

$$
a _ { t , F } ^ { 2 } + 2 \alpha b _ { t } \leq 4 \varepsilon _ { t } ^ { 2 } + \alpha \varepsilon _ { t } ^ { 2 } = ( 4 + \log N ) \varepsilon _ { t } ^ { 2 } , \quad \mathrm { a n d ~ t h u s : } \quad \sum _ { t = 1 } ^ { T } \Bigl ( a _ { t , F } ^ { 2 } + 2 \alpha b _ { t } \Bigr ) \leq ( 4 + \log N ) \sum _ { t = 1 } ^ { T } \varepsilon _ { t } ^ { 2 } .
$$

Next,

$$
a _ { t - m : t - 1 } = \sum _ { j = \operatorname* { m a x } \{ 1 , t - m \} } ^ { t - 1 } a _ { j , F } \leq 2 \sum _ { j = \operatorname* { m a x } \{ 1 , t - m \} } ^ { t - 1 } \varepsilon _ { j } .
$$

By Cauchy–Schwarz,

$$
a _ { t - m : t - 1 } \leq 2 \sqrt { m } \left( \sum _ { j = \operatorname* { m a x } \{ 1 , t - m \} } ^ { t - 1 } \varepsilon _ { j } ^ { 2 } \right) ^ { 1 / 2 } \leq 2 \sqrt { m } \left( \sum _ { j = 1 } ^ { T } \varepsilon _ { j } ^ { 2 } \right) ^ { 1 / 2 } .
$$

Thus,

$$
\operatorname* { m a x } _ { t \in \left[ T \right] } a _ { t - m : t - 1 } \leq 2 \sqrt { m } \sqrt { \sum _ { t = 1 } ^ { T } \varepsilon _ { t } ^ { 2 } } .
$$

We now relate $\varepsilon _ { t }$ to the prediction error of the credit gradients. Since it is $\tilde { M } _ { s } ~ = ~ X _ { s } ^ { \top } \tilde { h } _ { s }$ <sub>s</sub> and $M _ { s } = X _ { s } ^ { \top } h _ { s }$ , we have $\tilde { M } _ { s } - M _ { s } = X _ { s } ^ { \top } ( \tilde { h } _ { s } - h _ { s } )$ . Therefore, we can write:

$$
\varepsilon _ { t } = \left\| \sum _ { s = t - m } ^ { t } X _ { s } ^ { \top } ( \tilde { h } _ { s } - h _ { s } ) \right\| _ { \infty } \leq \sum _ { s = t - m } ^ { t } \| X _ { s } ^ { \top } ( \tilde { h } _ { s } - h _ { s } ) \| _ { \infty } .
$$

Now, for any vector v it holds:

$$
\| X _ { s } ^ { \top } v \| _ { \infty } = \operatorname* { m a x } _ { i \in [ N ] } | \langle x _ { s } ^ { i } , v \rangle | \leq \operatorname* { m a x } _ { i \in [ N ] } \| x _ { s } ^ { i } \| \| v \| \leq D \| v \| ,
$$

since $\| x _ { s } ^ { i } \| \leq D$ for all $i \in [ N ]$ . Therefore, $\begin{array} { r } { \varepsilon _ { t } \leq D \sum _ { s = t - m } ^ { t } \| h _ { s } - \tilde { h } _ { s } \| } \end{array}$ , and applying Cauchy– Schwarz we can write:

$$
\varepsilon _ { t } ^ { 2 } \leq D ^ { 2 } ( m + 1 ) \sum _ { s = t - m } ^ { t } \| h _ { s } - \tilde { h } _ { s } \| ^ { 2 } .
$$

Summing over t and using that each term $\| h _ { s } - \tilde { h } _ { s } \| ^ { 2 }$ appears in at most $m + 1$ windows yields

$$
\sum _ { t = 1 } ^ { T } \varepsilon _ { t } ^ { 2 } \leq D ^ { 2 } ( m + 1 ) ^ { 2 } \sum _ { t = 1 } ^ { T } \| h _ { t } - \tilde { h } _ { t } \| ^ { 2 } \Rightarrow \sqrt { \sum _ { t = 1 } ^ { T } \varepsilon _ { t } ^ { 2 } } \leq D ( m + 1 ) \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \tilde { h } _ { t } \| ^ { 2 } } .
$$

Returning to (34) and using the bounds derived above for each term, we obtain

$$
\mathrm { T e r m \left( I \right) } \le 2 D ( m + 1 ) \left( 4 \sqrt { m } + \sqrt { 4 + \log N } \right) \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \tilde { h } _ { t } \| ^ { 2 } } .\tag{35}
$$

A cleaner upper bound is

$$
\mathrm { T e r m \left( I \right) } \le 8 D ( m + 1 ) ^ { \frac { 3 } { 2 } } \left( 1 + \sqrt { 1 + \log N } \right) \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \tilde { h } _ { t } \| ^ { 2 } } .\tag{36}
$$

## C Proofs of Section 4

## C.1 Proof of Theorem 2

Proof. Define

$$
A _ { N } : = 8 D ( m + 1 ) ^ { 3 / 2 } \left( 1 + { \sqrt { 1 + \log N } } \right) ,
$$

and

$$
A _ { P } : = 4 ( m + 1 ) ^ { 3 / 2 } \left( D + \sqrt { D ( D + P _ { T } ) } \right) .
$$

For compactness, let

$$
A : = A _ { N } + A _ { P } .
$$

From the delayed optimistic OCO analysis, the surrogate dynamic regret satisfies

$$
\mathcal { R } _ { T } ( L ) \leq A \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \widetilde { h } _ { t } \| _ { 2 } ^ { 2 } } .
$$

We now relate the surrogate-gradient prediction error to the prediction errors of the objective and constraint functions. In reveal-time notation, the surrogate loss is

$$
L _ { t } ( x _ { t - m } ) = V f _ { t } ( x _ { t - m } ) + \Phi ^ { \prime } ( Q _ { t - m - 1 } ) g _ { t } ^ { + } ( x _ { t - m } ) .
$$

Equivalently, setting $s = t - m$ , the same surrogate can be written in decision-time notation as

$$
L _ { s + m } ( x _ { s } ) = V f _ { s + m } ( x _ { s } ) + \Phi ^ { \prime } ( Q _ { s - 1 } ) g _ { s + m } ^ { + } ( x _ { s } ) .
$$

Thus, the feedback associated with decision $x _ { s }$ is

$$
h _ { s } = \nabla L _ { s + m } ( x _ { s } ) .
$$

Since $\Phi ^ { \prime }$ is nondecreasing and $Q _ { s - 1 } \leq Q _ { T }$ , we have

$$
\| h _ { s } - \widetilde { h } _ { s } \| _ { 2 } \leq V \| \Delta _ { s } ^ { f } \| _ { 2 } + \Phi ^ { \prime } ( Q _ { T } ) \| \Delta _ { s } ^ { g ^ { + } } \| _ { 2 } .
$$

Relabeling s as $t ,$ and using the quadratic penalty $\Phi ( Q ) = Q ^ { 2 }$ , so that $\Phi ^ { \prime } ( Q _ { T } ) = 2 Q _ { T }$ , Minkowski’s inequality gives

$$
\sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \widetilde { h } _ { t } \| _ { 2 } ^ { 2 } } \leq V \sqrt { E _ { T } ( f ) } + 2 Q _ { T } \sqrt { E _ { T } ( g ^ { + } ) } .
$$

Therefore,

$$
\mathcal { R } _ { T } ( L ) \leq A V \sqrt { E _ { T } ( f ) } + 2 A Q _ { T } \sqrt { E _ { T } ( g ^ { + } ) } .
$$

Combining this with the delayed regret decomposition lemma gives

$$
Q _ { T } ^ { 2 } - Q _ { m - 1 } ^ { 2 } + V \mathcal { R } _ { T } ( f ) \leq A V \sqrt { E _ { T } ( f ) } + 2 A Q _ { T } \sqrt { E _ { T } ( g ^ { + } ) } + 2 G ( m + 1 ) Q _ { T } .
$$

Define

$$
B _ { T } : = A \sqrt { E _ { T } ( g ^ { + } ) } + G ( m + 1 ) .
$$

Then

$$
Q _ { T } ^ { 2 } - Q _ { m - 1 } ^ { 2 } + V \mathcal { R } _ { T } ( f ) \leq A V \sqrt { E _ { T } ( f ) } + 2 B _ { T } Q _ { T } .
$$

For the regret bound, rearranging gives

$$
V \mathcal { R } _ { T } ( f ) \leq A V \sqrt { E _ { T } ( f ) } + 2 B _ { T } Q _ { T } - Q _ { T } ^ { 2 } + Q _ { m - 1 } ^ { 2 } .
$$

Using $2 a b - b ^ { 2 } \leq a ^ { 2 }$ , with $a = B _ { T }$ and $b = Q _ { T }$ , we obtain

$$
V \mathcal { R } _ { T } ( f ) \leq A V \sqrt { E _ { T } ( f ) } + B _ { T } ^ { 2 } + Q _ { m - 1 } ^ { 2 } .
$$

Thus,

$$
\mathcal { R } _ { T } ( f ) \leq A \sqrt { E _ { T } ( f ) } + \frac { B _ { T } ^ { 2 } + Q _ { m - 1 } ^ { 2 } } { V } .
$$

We now choose

$$
V : = 4 ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } \right) ^ { 2 } \sqrt { E _ { T } ( g ^ { + } ) } + 2 G ^ { 2 } ( m + 1 ) + Q _ { m - 1 } ^ { 2 } .
$$

Since

$$
B _ { T } ^ { 2 } + Q _ { m - 1 } ^ { 2 } = \Bigl ( A \sqrt { E _ { T } ( g ^ { + } ) } + G ( m + 1 ) \Bigr ) ^ { 2 } + Q _ { m - 1 } ^ { 2 } ,
$$

we have

$$
B _ { T } ^ { 2 } + Q _ { m - 1 } ^ { 2 } \leq 2 A ^ { 2 } E _ { T } ( g ^ { + } ) + 2 G ^ { 2 } ( m + 1 ) ^ { 2 } + Q _ { m - 1 } ^ { 2 } .
$$

Therefore,

$$
\begin{array} { c } { { \displaystyle \frac { B _ { T } ^ { 2 } + Q _ { m - 1 } ^ { 2 } } { V } \le \frac { 2 A ^ { 2 } E _ { T } ( g ^ { + } ) } { 4 ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } \right) ^ { 2 } \sqrt { E _ { T } ( g ^ { + } ) } } } } \\ { { + \frac { 2 G ^ { 2 } ( m + 1 ) ^ { 2 } } { 2 G ^ { 2 } ( m + 1 ) } + \frac { Q _ { m - 1 } ^ { 2 } } { Q _ { m - 1 } ^ { 2 } } } } \\ { { \le \displaystyle \frac { A ^ { 2 } } { 2 ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } \right) ^ { 2 } } \sqrt { E _ { T } ( g ^ { + } ) } + ( m + 1 ) + 1 . } } \end{array}
$$

Hence,

$$
\mathcal { R } _ { T } ( f ) \leq A \sqrt { E _ { T } ( f ) } + \frac { A ^ { 2 } } { 2 ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } \right) ^ { 2 } } \sqrt { E _ { T } ( g ^ { + } ) } + ( m + 1 ) + 1 .
$$

Substituting the definitions of $A _ { N }$ and $A _ { P } ,$ , and hiding problem-dependent constants, gives

$$
A = \mathcal { O } \Big ( ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } + \sqrt { 1 + P _ { T } } \right) \Big ) .
$$

Moreover, since

$$
\frac { \left( 1 + \sqrt { 1 + \log N } + \sqrt { 1 + P _ { T } } \right) ^ { 2 } } { \left( 1 + \sqrt { 1 + \log N } \right) ^ { 2 } } = \mathcal { O } ( 1 + P _ { T } ) ,
$$

we obtain

$$
\begin{array} { r l } & { \Biggl | \mathcal { R } _ { T } ( f ) \leq \mathcal { O } \Biggl ( ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } + \sqrt { 1 + P _ { T } } \right) \sqrt { E _ { T } ( f ) } \Biggr . } \\ & { \qquad \left. + ( m + 1 ) ^ { 3 / 2 } ( 1 + P _ { T } ) \sqrt { E _ { T } ( g ^ { + } ) } + ( m + 1 ) \right) . } \end{array}
$$

We now prove the CCV bound. Using the lower bound

$$
\mathcal { R } _ { T } ( f ) \geq - G D T
$$

in the coupled inequality gives

$$
Q _ { T } ^ { 2 } - 2 B _ { T } Q _ { T } - \left( A V \sqrt { E _ { T } ( f ) } + V G D T + Q _ { m - 1 } ^ { 2 } \right) \leq 0 .
$$

Since $Q _ { T } \geq 0$ , the implication

$$
x ^ { 2 } - 2 a x - b \leq 0 \quad \Longrightarrow \quad x \leq a + { \sqrt { a ^ { 2 } + b } }
$$

yields

$$
Q _ { T } \leq B _ { T } + \sqrt { B _ { T } ^ { 2 } + A V \sqrt { E _ { T } ( f ) } + V G D T + Q _ { m - 1 } ^ { 2 } } .
$$

Using

$$
\sqrt { x + y + z + w } \leq \sqrt { x } + \sqrt { y } + \sqrt { z } + \sqrt { w } ,
$$

we get

$$
Q _ { T } \leq 2 B _ { T } + \sqrt { A V \sqrt { E _ { T } ( f ) } } + \sqrt { V G D T } + Q _ { m - 1 } .
$$

Using the crude bound $E _ { T } ( f ) \leq T$ , we have

$$
{ \sqrt { A V { \sqrt { E _ { T } ( f ) } } } } \leq { \sqrt { A V { \sqrt { T } } } } = { \frac { \sqrt { A } } { T ^ { 1 / 4 } } } { \sqrt { V T } } .
$$

Also,

$$
{ \sqrt { V G D T } } = { \mathcal { O } } ( { \sqrt { V T } } ) ,
$$

where problem-dependent constants are hidden. Therefore,

$$
Q _ { T } \leq 2 B _ { T } + \mathcal { O } \left( \left( 1 + \frac { \sqrt { A } } { T ^ { 1 / 4 } } \right) \sqrt { V T } \right) + Q _ { m - 1 } .
$$

It remains to expand $\sqrt { V T }$ . By the choice of $V _ { ; }$

$$
V = 4 ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } \right) ^ { 2 } \sqrt { E _ { T } ( g ^ { + } ) } + 2 G ^ { 2 } ( m + 1 ) + Q _ { m - 1 } ^ { 2 } .
$$

Hence,

$$
\begin{array} { r } { \sqrt { V T } \leq 2 ( m + 1 ) ^ { 3 / 4 } \left( 1 + \sqrt { 1 + \log N } \right) \sqrt { T } E _ { T } ( g ^ { + } ) ^ { 1 / 4 } + \sqrt { 2 } G \sqrt { m + 1 } \sqrt { T } + Q _ { m - 1 } \sqrt { T } . } \end{array}
$$

Substituting this into the previous inequality gives

$$
\begin{array} { r l } & { Q _ { T } \leq \mathcal { O } \Bigg ( A \sqrt { E _ { T } ( g ^ { + } ) } + G ( m + 1 ) + Q _ { m - 1 } } \\ & { \qquad + \left( 1 + \frac { \sqrt { A } } { T ^ { 1 / 4 } } \right) \left[ ( m + 1 ) ^ { 3 / 4 } \left( 1 + \sqrt { 1 + \log N } \right) \sqrt { T } E _ { T } ( g ^ { + } ) ^ { 1 / 4 } \right. } \\ & { \qquad \left. + \sqrt { m + 1 } \sqrt { T } + Q _ { m - 1 } \sqrt { T } \right] \Bigg ) . } \end{array}
$$

Finally, substituting

$$
A = { \mathcal { O } } \Bigl ( ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } + \sqrt { 1 + P _ { T } } \right) \Bigr ) ,
$$

we get

$$
\sqrt { A } = \mathcal { O } \bigg ( ( m + 1 ) ^ { 3 / 4 } \sqrt { 1 + \sqrt { 1 + \log N } + \sqrt { 1 + P _ { T } } } \bigg ) .
$$

Therefore,

$$
\begin{array} { r l } & { Q _ { T } \leq \mathcal { O } \Bigg ( ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } + \sqrt { 1 + P _ { T } } \right) \sqrt { E _ { T } ( g ^ { + } ) } + ( m + 1 ) + Q _ { m - 1 } } \\ & { \qquad + \left[ 1 + \frac { ( m + 1 ) ^ { 3 / 4 } \sqrt { 1 + \sqrt { 1 + \log N } + \sqrt { 1 + P _ { T } } } } { T ^ { 1 / 4 } } \right] \mathrm { t h i s ~ t e r m ~ i s ~ o f ~ o r d e r ~ } ( m + 1 ) ^ { 3 / 4 } } \\ & { \qquad \times \left[ ( m + 1 ) ^ { 3 / 4 } \left( 1 + \sqrt { 1 + \log N } \right) \sqrt { T } E _ { T } ( g ^ { + } ) ^ { 1 / 4 } + \sqrt { m + 1 } \sqrt { T } + Q _ { m - 1 } \sqrt { T } \right] \Bigg ) . } \end{array}
$$

Thus

$$
\begin{array} { r } { \boxed { Q _ { T } \leq \mathcal { O } \bigg ( ( m + 1 ) ^ { 3 / 2 } \left( 1 + \sqrt { 1 + \log N } + \sqrt { 1 + P _ { T } } \right) \sqrt { T } E _ { T } ( g ^ { + } ) ^ { \frac { 1 } { 4 } } + ( m + 1 ) ^ { \frac { 5 } { 4 } } \sqrt { T } \bigg ) . } } \end{array}
$$

This proves the stated regret and CCV bounds.

## C.2 Proof of Theorem 3

Proof. Since $P _ { T }$ is known, we do not need the meta-learner. The base learner is tuned directly to the path length $P _ { T }$ . Define

$$
A _ { P } : = 4 ( m + 1 ) ^ { 3 / 2 } \left( D + \sqrt { D ( D + P _ { T } ) } \right) .
$$

Then the delayed optimistic dynamic OCO bound for the base learner gives

$$
\mathcal { R } _ { T } ( L ) \leq A _ { P } \sqrt { \sum _ { t = 1 } ^ { T } \| h _ { t } - \tilde { h } _ { t } \| _ { 2 } ^ { 2 } } .
$$

Thus

$$
\Phi ( Q _ { T } ) - \Phi ( Q _ { m - 1 } ) + \mathcal { R } _ { T } ( f ) \leq A _ { P } \sqrt { E _ { T } ( f ) } + \Big ( A _ { P } \sqrt { E _ { T } ( g ^ { + } ) } + G ( m + 1 ) \Big ) \Phi ^ { \prime } ( Q _ { T } ) .
$$

Now choose

$$
\Phi ( Q ) = e ^ { \lambda Q } - 1 , \qquad \Phi ^ { \prime } ( Q ) = \lambda e ^ { \lambda Q } .
$$

Then

$$
e ^ { \lambda Q _ { T } } - e ^ { \lambda Q _ { m - 1 } } + \mathcal { R } _ { T } ( f ) \leq A _ { P } \sqrt { E _ { T } ( f ) } + \lambda \Big ( A _ { P } \sqrt { E _ { T } ( g ^ { + } ) } + G ( m + 1 ) \Big ) e ^ { \lambda Q _ { T } } .
$$

Let

$$
B _ { T } : = A _ { P } \sqrt { E _ { T } ( g ^ { + } ) } + G ( m + 1 ) .
$$

By the choice of $\lambda ,$

$$
\lambda = \frac { 1 } { 2 B _ { T } } ,
$$

and therefore $\lambda B _ { T } = 1 / 2$ . Hence

$$
\frac { 1 } { 2 } e ^ { \lambda Q _ { T } } + \mathcal { R } _ { T } ( f ) \leq e ^ { \lambda Q _ { m - 1 } } + A _ { P } \sqrt { E _ { T } ( f ) } .
$$

We first bound the regret. From the previous inequality,

$$
\mathcal { R } _ { T } ( f ) \leq A _ { P } \sqrt { E _ { T } ( f ) } + e ^ { \lambda Q _ { m - 1 } } - \frac { 1 } { 2 } e ^ { \lambda Q _ { T } } .
$$

Since the last term is non-positive, we get

$$
\begin{array} { r } { \mathcal { R } _ { T } ( f ) \leq A _ { P } \sqrt { E _ { T } ( f ) } + e ^ { \lambda Q _ { m - 1 } } . } \end{array}
$$

The term $Q _ { m - 1 }$ only depends on the initial delayed rounds and is absorbed into the constants. Therefore,

$$
\mathcal { R } _ { T } ( f ) = \mathcal { O } \Big ( A _ { P } \sqrt { E _ { T } ( f ) } \Big ) .
$$

Using the definition of $A _ { P }$ , this yields

$$
\mathscr { R } _ { T } ( f ) = \mathscr { O } \Big ( ( m + 1 ) ^ { 3 / 2 } \sqrt { ( 1 + P _ { T } ) E _ { T } ( f ) } \Big ) .
$$

We now bound the CCV. Using the lower bound

$$
\mathcal { R } _ { T } ( f ) \geq - G D T ,
$$

we obtain

$$
\frac { 1 } { 2 } e ^ { \lambda Q _ { T } } \leq e ^ { \lambda Q _ { m - 1 } } + A _ { P } \sqrt { E _ { T } ( f ) } + G D T .
$$

Thus,

$$
e ^ { \lambda Q _ { T } } \leq 2 \left( e ^ { \lambda Q _ { m - 1 } } + A _ { P } \sqrt { E _ { T } ( f ) } + G D T \right) .
$$

Taking logarithms gives

$$
Q _ { T } \leq \frac { 1 } { \lambda } \log \Bigl ( 2 \Bigl ( e ^ { \lambda Q _ { m - 1 } } + A _ { P } \sqrt { E _ { T } ( f ) } + G D T \Bigr ) \Bigr ) .
$$

Since $1 / \lambda = 2 B _ { T }$ , we get

$$
Q _ { T } \leq 2 \Big ( A _ { P } \sqrt { E _ { T } ( g ^ { + } ) } + G ( m + 1 ) \Big ) \log \Big ( 2 \Big ( e ^ { \lambda Q _ { m - 1 } } + A _ { P } \sqrt { E _ { T } ( f ) } + G D T \Big ) \Big ) .
$$

Under the standard boundedness assumptions, the logarithmic factor is of order ${ \mathcal { O } } ( \log T )$ . Therefore,

$$
\begin{array} { r } { Q _ { T } = \mathcal { O } \Big ( \Big ( A _ { P } \sqrt { E _ { T } ( g ^ { + } ) } + G ( m + 1 ) \Big ) \log T \Big ) . } \end{array}
$$

Substituting the definition of $A _ { P } ,$ , we obtain

$$
Q _ { T } = \mathcal { O } \Big ( \Big ( ( m + 1 ) ^ { 3 / 2 } \sqrt { ( 1 + P _ { T } ) E _ { T } ( g ^ { + } ) } + G ( m + 1 ) \Big ) \log T \Big ) .
$$

This proves the theorem.

## D Proof of Section 5

## D.1 Proof of Lemma 2

We start with the proof of the main result of this section, namely Lemma 2.

Proof. The proof follows from two remarkably simple observations: first, convexity allows us to linearize each memory loss around the played memory window; second, after re-indexing the resulting terms, the contribution of each decision $x _ { s }$ can be grouped into a single gradient quantity that is revealed only after, at most, m future rounds.

In detail, recall that the memory regret is:

$$
\mathcal { R } _ { T } ( L ) : = \sum _ { t = m } ^ { T } L _ { t } ( x _ { t - m } , \ldots , x _ { t } ) - \sum _ { t = m } ^ { T } L _ { t } ( u _ { t - m } , \ldots , u _ { t } ) .
$$

Since $L _ { t }$ is convex in its joint argument $( z _ { 0 } , \ldots , z _ { m } ) \in \mathcal { X } ^ { m + 1 }$ , the first-order convexity inequality gives, for every $t \in [ T _ { m } ] : = \{ m , \ldots , T \}$

$$
L _ { t } ( x _ { t - m } ^ { t } ) - L _ { t } ( u _ { t - m : t } ) \leq \sum _ { j = 0 } ^ { m } \Big \langle \nabla _ { x _ { t - j } } L _ { t } ( x _ { t - m } ^ { t } ) , x _ { t - j } - u _ { t - j } \Big \rangle .
$$

Here $\nabla _ { x _ { t - j } } L _ { t } ( x _ { t - m } ^ { t } )$ denotes the partial gradient of $L _ { t }$ with respect to its $( m - j )$ -th coordinate, evaluated at the played decision $x _ { t - m : t } .$ . Summing the above inequality over t, we obtain:

$$
\mathcal { R } _ { T } ( L ) \leq \sum _ { t = m } ^ { T } \sum _ { j = 0 } ^ { m } \Big \langle \nabla _ { x _ { t - j } } L _ { t } ( x _ { t - m } ^ { t } ) , x _ { t - j } - u _ { t - j } \Big \rangle .
$$

At this stage, the right-hand side is still indexed by the loss time t. However, for the online-learning interpretation, it is more natural to regroup the terms according to the decision time. To do so, we set:

$$
\begin{array} { r } { s : = t - j , \mathrm { o r , e q u i v a l e n t l y : } t = s + k , k : = j . } \end{array}
$$

Then, for a fixed s, the index k must satisfy

$$
0 \leq k \leq m \qquad { \mathrm { a n d } } \qquad m \leq s + k \leq T .
$$

Therefore the admissible range of k is

$$
k \in \left[ \operatorname* { m a x } \{ 0 , m - s \} , \operatorname* { m i n } \{ m , T - s \} \right] .
$$

Using this change of variables and swapping the order of summation, we can write:

$$
\mathcal { R } _ { T } ( L ) \leq \sum _ { t = 1 } ^ { T } \left. \sum _ { k = \operatorname* { m a x } \{ 0 , m - t \} } ^ { \operatorname* { m i n } \{ m , T - t \} } \nabla _ { x _ { t } } L _ { t + k } ( x _ { t + k - m } ^ { t + k } ) , x _ { t } - u _ { t } \right. ,
$$

where we just used t instead of $s .$ This motivates the definition of the credit gradient

$$
h _ { t } : = \sum _ { k = \operatorname* { m a x } \{ 0 , m - t \} } ^ { \operatorname* { m i n } \{ m , T - t \} } \nabla _ { x _ { t } } L _ { t + k } ( x _ { t + k - m } ^ { t + k } ) .
$$

With this notation, we can write:

$$
\mathcal { R } _ { T } ( L ) \leq \sum _ { t = 1 } ^ { T } \langle h _ { t } , \ x _ { t } - u _ { t } \rangle .
$$

Indexing convention and delayed interpretation. The original memory-dependent losses are indexed by $t \in [ T _ { m } ] : = \{ m , \ldots , T \}$ , since a full memory window is only available from round m onward. After linearization, however, the auxiliary delayed problem is naturally indexed by the decision times $s \in [ T ]$ . Indeed, even the early decisions $x _ { 1 } , \ldots , x _ { m }$ may appear in later memory windows, and the corresponding partial gradients must be assigned to the credit gradient of the decision that generated them. Algorithmically, feedback starts at round m: whenever a memory loss $L _ { t }$ is revealed, it contributes partial gradients to all decisions in its memory window $x _ { t - m : t }$

This also explains why the resulting auxiliary problem is a delayed linear problem. By construction, $h _ { t }$ aggregates the effect of the single decision $x _ { t }$ on the losses

$$
L _ { t } , L _ { t + 1 } , . . . , L _ { t + m } .
$$

Hence $h _ { t }$ is not entirely known when $x _ { t }$ is played; it is only fully revealed once all these losses have been observed, that is, by round $t + m$ . Therefore, $\{ h _ { t } \} _ { t = 1 } ^ { T }$ is a sequence of linear losses revealed with delay at most $m .$ . Consequently, the original memory-dependent regret is upper bounded by the regret of this delayed linear problem. Thus, any applicable delayed- $. \mathrm { o c o }$ regret guarantee for the auxiliary linear losses can be transferred to the memory-dependent problem through the credit gradients $h _ { s }$ . The resulting guarantee has the same structure as in the delayed setting; the only difference is the effective gradients are now the credit gradients $h _ { t } ,$ , whose size introduces constants depending on the memory length m.

At a conceptual level, the key idea is that we postpone the construction of the auxiliary delayed loss until after linearizing the memory-dependent regret. The contribution of each decision is then captured through the corresponding coordinate block of the joint gradient and aggregated into its credit gradient. This is also why constructing optimistic hints may require predicting the future block $x _ { t + 1 : t + m } .$ , rather than only $x _ { t + 1 }$ . In the separable case, this recovers the usual forward-gradient aggregation, while our construction does not require separability and therefore preserves the original joint feasibility benchmark. In contrast, [5] applies the penalty to an individual component of the separable constraint, $g _ { t + i } ^ { + , i } ( x _ { t } )$ , whereas here the penalty is applied to the joint constraint function.

## D.2 Proof of Remark 5

Proof. To transfer the delayed-feedback guarantees of Sections $_ { 3 - 4 }$ to the memory setting, it remains to relate the prediction error of the credit gradients to the memory prediction errors defined in Remark 5.

In the unconstrained case, $L _ { t } = f _ { t }$ , and therefore

$$
\boldsymbol { h _ { s } } - \boldsymbol { \widetilde { h } _ { s } } = \sum _ { t \in I _ { s } } \left[ \boldsymbol { \nabla _ { \boldsymbol { x } _ { s } } } f _ { t } ( \boldsymbol { x } _ { t - m } ^ { t } ) - \boldsymbol { \nabla _ { \boldsymbol { x } _ { s } } } \boldsymbol { \widetilde { f _ { t } } } ( \boldsymbol { \widetilde { x } _ { t - m } ^ { t } } ) \right] .
$$

Hence, directly from (14),

$$
\left( \sum _ { s = 1 } ^ { T } \| h _ { s } - \widetilde { h } _ { s } \| ^ { 2 } \right) ^ { 1 / 2 } = \sqrt { E _ { T } ^ { \mathrm { m e m } } ( f ) } .
$$

Therefore, Theorem 1 applies directly to the memory problem through Corollary 1.

We next consider the constrained case. To recover the prediction-error structure used in Section 4, it suffices to show that

$$
\left( \sum _ { s = 1 } ^ { T } \| h _ { s } - \widetilde { h } _ { s } \| ^ { 2 } \right) ^ { 1 / 2 } \leq V \sqrt { E _ { T } ^ { \mathrm { m e m } } ( f ) } + \Phi ^ { \prime } ( Q _ { T } ) \sqrt { E _ { T } ^ { \mathrm { m e m } } ( g ^ { + } ) } .\tag{37}
$$

Recall that

$$
L _ { t } ( x _ { t - m } ^ { t } ) = V f _ { t } ( x _ { t - m } ^ { t } ) + \Phi ^ { \prime } ( Q _ { t - m - 1 } ) g _ { t } ^ { + } ( x _ { t - m } ^ { t } ) ,
$$

and

$$
h _ { s } = \sum _ { t \in I _ { s } } \nabla _ { x _ { s } } L _ { t } ( x _ { t - m } ^ { t } ) .
$$

For the predicted credit gradient, we use the same queue coefficient, namely

$$
\widetilde { L } _ { t } ( \widetilde { x } _ { t - m } ^ { t } ) = V \widetilde { f } _ { t } ( \widetilde { x } _ { t - m } ^ { t } ) + \Phi ^ { \prime } ( Q _ { t - m - 1 } ) \widetilde { g } _ { t } ^ { + } ( \widetilde { x } _ { t - m } ^ { t } ) .
$$

Indeed, $t \in { \cal I } _ { s }$ implies $t \leq s + m$ , and therefore $t - m - 1 \leq s - 1$ . Hence, $\Phi ^ { \prime } ( Q _ { t - m - 1 } )$ is already available when the hint associated with $x _ { s }$ is formed.

Using the definitions of $h _ { s }$ and $\widetilde { h } _ { s } .$ , we obtain

$$
\begin{array} { r l } & { \displaystyle h _ { s } - \widetilde { h } _ { s } = V \sum _ { t \in I _ { s } } \Big [ \nabla _ { x _ { s } } f _ { t } ( x _ { t - m } ^ { t } ) - \nabla _ { x _ { s } } \widetilde { f } _ { t } ( \widetilde { x } _ { t - m } ^ { t } ) \Big ] } \\ & { \quad \quad \quad \quad + \sum _ { t \in I _ { s } } \Phi ^ { \prime } ( Q _ { t - m - 1 } ) \left[ \nabla _ { x _ { s } } g _ { t } ^ { + } ( x _ { t - m } ^ { t } ) - \nabla _ { x _ { s } } \widetilde { g } _ { t } ^ { + } ( \widetilde { x } _ { t - m } ^ { t } ) \right] . } \end{array}\tag{38}
$$

It remains to control the second term. Fix s and order $I _ { s } = \{ t _ { 1 } < \cdots < t _ { | I _ { s } | } \}$ . Since both $Q _ { t }$ and $\Phi ^ { \prime }$ are nondecreasing,

$$
0 \leq \frac { \Phi ^ { \prime } ( Q _ { t _ { 1 } - m - 1 } ) } { \Phi ^ { \prime } ( Q _ { s - 1 } ) } \leq \cdots \leq \frac { \Phi ^ { \prime } ( Q _ { t _ { | I _ { s } | } - m - 1 } ) } { \Phi ^ { \prime } ( Q _ { s - 1 } ) } \leq 1 ,
$$

whenever $\Phi ^ { \prime } ( Q _ { s - 1 } ) > 0 .$

A discrete summation-by-parts identity then gives

$$
\begin{array} { r l } & { \quad \displaystyle \sum _ { j = 1 } ^ { \lfloor I _ { j } \rfloor } \frac { \Phi ^ { \prime } ( Q _ { t _ { j } - m - 1 } ) } { \Phi ^ { \prime } ( Q _ { s - 1 } ) } \left[ \nabla _ { x _ { s } } g _ { t _ { j } } ^ { + } ( x _ { t _ { j } - m } ^ { t _ { j } } ) - \nabla _ { x _ { s } } \tilde { g } _ { t _ { j } } ^ { + } ( \tilde { x } _ { t _ { j } - m } ^ { t _ { j } } ) \right] } \\ & { = \displaystyle \frac { \Phi ^ { \prime } ( Q _ { t _ { 1 } - m - 1 } ) } { \Phi ^ { \prime } ( Q _ { s - 1 } ) } \sum _ { j = 1 } ^ { \lfloor x _ { s } \rfloor } \left[ \nabla _ { x _ { s } } g _ { t _ { j } } ^ { + } ( x _ { t _ { j } - m } ^ { t _ { j } } ) - \nabla _ { x _ { s } } \tilde { g } _ { t _ { j } } ^ { + } ( \tilde { x } _ { t _ { j } - m } ^ { t _ { j } } ) \right] } \\ & { \quad \displaystyle + \sum _ { j = 2 } ^ { \lfloor I _ { s } \rfloor } \left( \frac { \Phi ^ { \prime } ( Q _ { t _ { j } - m - 1 } ) - \Phi ^ { \prime } ( Q _ { t _ { j - 1 } - m - 1 } ) } { \Phi ^ { \prime } ( Q _ { s - 1 } ) } \right) \sum _ { r = j } ^ { \lfloor I _ { s } \rfloor } \left[ \nabla _ { x _ { s } } g _ { t _ { r } } ^ { + } ( x _ { t _ { r } - m } ^ { t _ { r } } ) - \nabla _ { x _ { s } } \tilde { g } _ { t _ { r } } ^ { + } ( \tilde { x } _ { t _ { r } - m } ^ { t _ { r } } ) \right] . } \end{array}
$$

All coefficients on the right-hand side are nonnegative, and their sum is

$$
\frac { \Phi ^ { \prime } ( Q _ { t _ { | I _ { s } | } - m - 1 } ) } { \Phi ^ { \prime } ( Q _ { s - 1 } ) } \leq 1 .
$$

Therefore, by the triangle inequality,

$$
\begin{array} { r l r } {  { \| \sum _ { t \in I _ { s } } \frac { \Phi ^ { \prime } ( Q _ { t - m - 1 } ) } { \Phi ^ { \prime } ( Q _ { s - 1 } ) } [ \nabla _ { x _ { s } } g _ { t } ^ { + } ( x _ { t - m } ^ { t } ) - \nabla _ { x _ { s } } \widetilde g _ { t } ^ { + } ( \widetilde x _ { t - m } ^ { t } ) ] \| } }  \\ & { } & { \leq \operatorname* { m a x } _ { r \in I _ { s } } \| \sum _ { t \in I _ { s } } [ \nabla _ { x _ { s } } g _ { t } ^ { + } ( x _ { t - m } ^ { t } ) - \nabla _ { x _ { s } } \widetilde g _ { t } ^ { + } ( \widetilde x _ { t - m } ^ { t } ) ] \| . } \\ & { } & { \quad \quad \quad } \end{array}\tag{39}
$$

Multiplying (39) by $\Phi ^ { \prime } ( Q _ { s - 1 } )$ and using $\Phi ^ { \prime } ( Q _ { s - 1 } ) \leq \Phi ^ { \prime } ( Q _ { T } )$ yields

$$
\begin{array} { r l } {  { \bigg \| \sum _ { t \in I _ { s } } \Phi ^ { \prime } ( Q _ { t - m - 1 } ) [ \nabla _ { x _ { s } } g _ { t } ^ { + } ( x _ { t - m } ^ { t } ) - \nabla _ { x _ { s } } \tilde { g } _ { t } ^ { + } ( \tilde { x } _ { t - m } ^ { t } ) ] \bigg \| } \quad } & { } \\ & { \leq \Phi ^ { \prime } ( Q _ { T } ) \operatorname* { m a x } _ { r \in I _ { s } } \| \sum _ { t \in I _ { s } } [ \nabla _ { x _ { s } } g _ { t } ^ { + } ( x _ { t - m } ^ { t } ) - \nabla _ { x _ { s } } \tilde { g } _ { t } ^ { + } ( \tilde { x } _ { t - m } ^ { t } ) ] \| . } \\ & { \geq r } \end{array}\tag{40}
$$

If $\Phi ^ { \prime } ( Q _ { s - 1 } ) = 0$ , monotonicity implies $\Phi ^ { \prime } ( Q _ { t - m - 1 } ) = 0$ for every $t \in I _ { s } ,$ , so (40) holds trivially.

Combining (38) and (40), and then applying Minkowski’s inequality across s, gives

$$
\begin{array} { r l } { \displaystyle \left( \sum _ { s = 1 } ^ { T } \| h _ { s } - \widetilde { h } _ { s } \| ^ { 2 } \right) ^ { 1 / 2 } \leq { V } \left( \displaystyle \sum _ { s = 1 } ^ { T } \left\| \displaystyle \sum _ { t \in I _ { s } } \left[ \nabla _ { x _ { s } } f _ { t } ( x _ { t - m } ^ { t } ) - \nabla _ { x _ { s } } \widetilde { f } _ { t } ( \widetilde { x } _ { t - m } ^ { t } ) \right] \right\| ^ { 2 } \right) ^ { 1 / 2 } } & { } \\ { \displaystyle } & { + \Phi ^ { \prime } ( Q _ { T } ) \left( \displaystyle \sum _ { s = 1 } ^ { T } \displaystyle \operatorname* { m a x } _ { r \in I _ { s } } \left\| \displaystyle \sum _ { t \in I _ { s } } \left[ \nabla _ { x _ { s } } g _ { t } ^ { + } ( x _ { t - m } ^ { t } ) - \nabla _ { x _ { s } } \widetilde { g } _ { t } ^ { + } ( \widetilde { x } _ { t - m } ^ { t } ) \right] \right\| ^ { 2 } \right) ^ { 1 / 2 } . } \end{array}
$$

By (14)–(15), this is exactly

$$
\left( \sum _ { s = 1 } ^ { T } \| h _ { s } - \widetilde { h } _ { s } \| ^ { 2 } \right) ^ { 1 / 2 } \leq V \sqrt { E _ { T } ^ { \mathrm { m e m } } ( f ) } + \Phi ^ { \prime } ( Q _ { T } ) \sqrt { E _ { T } ^ { \mathrm { m e m } } ( g ^ { + } ) } .
$$

Thus, in the unconstrained case, the delayed prediction-error term reduces exactly to $\sqrt { E _ { T } ^ { \mathrm { m e m } } ( f ) }$ , so Theorem 1 transfers directly. In the constrained case, (37) has the same form as the prediction-error decomposition used in Section 4. Hence, after replacing $E _ { T } ( f )$ and $E _ { T } ( g ^ { + } )$ by $E _ { T } ^ { \mathrm { m e m } } ( f )$ and $E _ { T } ^ { \mathrm { m e m } } ( g ^ { + } )$ , respectively, the proofs of Theorems 2 and 3 carry over through Corollary 1. □