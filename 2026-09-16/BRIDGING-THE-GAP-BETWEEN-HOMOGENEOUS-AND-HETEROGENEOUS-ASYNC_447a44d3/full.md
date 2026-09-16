# BRIDGING THE GAP BETWEEN HOMOGENEOUS AND HETEROGENEOUS ASYNCHRONOUS OPTIMIZATION IS SURPRISINGLY DIFFICULT

Alexander Tyurin

AXXX, Moscow, Russia

Applied AI Institute, Moscow, Russia

## ABSTRACT

Modern large-scale machine learning tasks often require multiple workers, devices, CPUs, or GPUs to compute stochastic gradients in parallel and asynchronously to train model weights. Theoretical results typically distinguish between two settings: (i) the homogeneous setting, where all workers have access to the same data distribution, and (ii) the heterogeneous setting, where each worker operates on different data distributions. Known optimal time complexities in these settings reveal a significant gap, with far more pessimistic guarantees in the heterogeneous case. In this work, we investigate whether these pessimistic optimal time complexities can be overcome under different assumptions. Surprisingly, we show that improvement is provably impossible under widely used first- and second-order similarity assumptions for any randomized algorithm. We then turn to the interpolation regime and demonstrate that the weak interpolation assumption alone is also insufficient. Finally, we introduce a minimal combination of irreducible assumptions, strong interpolation and the local Polyak-Łojasiewicz condition, to derive a new time complexity bound that matches the dependence on worker computation times in the best-known result in the homogeneous setting, without requiring identical data distributions.

## 1 INTRODUCTION

We consider optimization problems described by

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } \Big \{ f ( x ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { \xi _ { i } \sim \mathcal { D } _ { i } } \left[ f _ { i } ( x ; \xi _ { i } ) \right] \Big \} ,\tag{1}
$$

where $f _ { i } : \mathbb { R } ^ { d } \times \mathbb { S } _ { \xi _ { i } }  \mathbb { R }$ and $\xi _ { i }$ is a random variable with distribution $\mathcal { D } _ { i }$ on $\mathbb { S } _ { \xi _ { i } }$ for all $i \in [ n ]$ Let us denote $f _ { i } ( x ) \overset { \cdot } { : = } \mathbb { E } _ { \xi _ { i } \sim \mathcal { D } _ { i } } \left[ f _ { i } ( x ; \xi _ { i } ) \right]$ . In our setup, we have n workers/clients/CPUs/GPUs working in parallel and asynchronously, and each worker i has access only to the stochastic gradient $\nabla f _ { i } ( x ; { \bar { \xi } } _ { i } )$ of the function $f _ { i }$ for all $x \in \mathbb { R } ^ { d }$ . We concentrate on the standard convergence metric and want to find a (possibly random) point x¯ such that $\mathbb { E } [ \left. \bar { x } - x _ { * } \right. ^ { 2 } ] \leq \varepsilon$ , where $x _ { * }$ is a solution of (1). Such a problem arises in many machine learning (ML), deep learning, federated learning (FL), and data science problems (Konecný et al.ˇ , 2016; McMahan et al., 2017; Goodfellow et al., 2016). In general, we use the following standard assumptions from convex stochastic optimization, but each result states which assumptions it requires.

Assumption 1.1 (Global smoothness). The function f is differentiable and L–smooth, i.e., $\| \nabla f ( x ) - \nabla f ( y ) \| \leq L \left\| x - y \right\|$ for all $x , y \in \mathbb { R } ^ { d }$

Assumption 1.2 (Local smoothness). The functions $f _ { i }$ are differentiable and $\scriptstyle L _ { i } - \operatorname { s m o o t h }$ . We also define $\bar { L } _ { \mathrm { m a x } } : = \operatorname* { m a x } _ { i \in [ n ] } L _ { i }$ . Note that $L \leq L _ { \operatorname* { m a x } }$ (This is why we distinguish Assum. 1.1 and 1.2). Assumption 1.3 (Convexity). The functions $f _ { i }$ are convex for all $i \in [ n ]$ . The function $f$ attains a minimum at a (non-unique) point $\boldsymbol { x } _ { * } \in \mathbb { R } ^ { d }$

Assumption 1.4 (Unbiased and $\sigma ^ { 2 }$ -variance-bounded noise). For all $x ~ \in ~ \mathbb { R } ^ { d }$ , stochastic gradients $\bar { \nabla } f _ { i } ( x ; \xi )$ are unbiased and σ<sup>2</sup>-variance-bounded, $\mathrm { i . e . , ~ } \mathbb { E } _ { \xi _ { i } } [ \nabla f _ { i } ( x ; \xi _ { i } ) ] ~ = ~ \nabla f _ { i } ( x )$ and $\mathbb { E } _ { \xi _ { i } } [ \| \nabla f _ { i } ( x ; \xi _ { i } ) - \nabla f _ { i } ( x ) \| ^ { 2 } ] \le \sigma ^ { 2 }$ for all $i \in [ n ]$ , where $\sigma ^ { 2 } \geq 0$

We also consider the case where $f$ satisfies the PŁ-condition, which is a much weaker assumption than µ–strong convexity (Karimi et al., 2016):

Assumption 1.5 (Global Polyak-Łojasiewicz condition). There exists $\mu > 0$ such that $\left\| \nabla f ( x ) \right\| ^ { 2 } \geq$ $2 \mu \left( f ( { \bar { x } } ) - f ^ { * } \right)$ for all $x \in \mathbb { R } ^ { d }$ , where $f ^ { * }$ is the finite optimal function value of $f .$

We focus on the modern setup where many workers work together in a distributed environment, where the workers can have arbitrarily computation behaviors due to hardware delays or network connectivity problems. Most previous works typically assume that the workers have the same performance that does not change over time. In contrast, our focus is on the setting where the computation times are heterogeneous and non-constant.

In the literature, the optimization problem (1) in the asynchronous environment is considered in two regimes: i) heterogeneous setting, where the functions ${ \dot { f } } _ { i }$ can be arbitrarily different; in the context of ML and FL, it means the workers have access to different datasets. ii) homogeneous setting, where the functions $f _ { i }$ are equal; in the context of ML and FL, it means the workers have access to the same dataset (Koloskova et al., 2022; Mishchenko et al., 2022; Feyzmahdavian & Johansson, 2023).

Notations. $[ n ] : = \{ 1 , \ldots , n \} ; \mathbb { N } _ { 0 } : = \{ 0 , 1 , 2 , \ldots \} ; \| \cdot \|$ is the standard Euclidean norm; $\langle \cdot , \cdot \rangle$ is the standard dot product; $g = \mathcal { O } ( f )$ : exist $C > 0$ such that $g ( z ) \leq C \times f ( z )$ for all $z \in \mathcal { Z } ; g = \Omega ( f )$ exist $C > 0$ such that $g ( z ) \geq C \times f ( z )$ for all $z \in \mathcal { Z } ; g = \Theta ( f ) : g = \mathcal { O } ( f )$ and $g = \Omega ( f )$ ; $g = \widetilde { \Theta } ( f )$ : the same as $g = \Theta ( f )$ but up to logarithmic factors.

## 1.1 PREVIOUS WORK

Oracle complexity. In the classical optimization theory (Nemirovskij & Yudin, 1983), algorithms are compared in terms of oracle calls. Assume that the number of workers is one and we work with nonconvex functions and Assumptions 1.1 and 1.4. It is well known (Arjevani et al., 2022; Carmon et al., 2020) that the optimal oracle complexity is $\mathcal { O } \left( { L \Delta } / { \varepsilon } + { \sigma ^ { 2 } L \Delta } / { \bar { \varepsilon } ^ { 2 } } \right)$ to find $\bar { x } \in \mathbb { R } ^ { d }$ such that $\mathbb { E } [ \| \nabla f ( \bar { x } ) \| ^ { 2 } ] \le \varepsilon$ . It is attained by the vanilla SGD method: $\boldsymbol { x } ^ { k + 1 } = \boldsymbol { x } ^ { k } - \gamma \nabla f ( \boldsymbol { x } ^ { k } ; \xi ^ { k } )$ where $\xi ^ { k }$ are i.i.d. random samples, $\Delta : \dot { = } \operatorname { \mathcal { f } } ( x ^ { 0 } ) - { f ^ { * } } , x ^ { 0 } \in { \mathbb R } ^ { d }$ is a starting point, and $\gamma =$ $\Theta \left( \operatorname* { m i n } \{ { 1 } / { L } , \varepsilon / { L \sigma ^ { 2 } } \} \right)$ is a step size. In the convex setting (Assumption 1.3), the optimal oracle complexity is Θ $\left( \bar { \sqrt { L } } R / \sqrt { \varepsilon } + \bar { \sigma } ^ { 2 } R ^ { 2 } / \varepsilon ^ { 2 } \right)$ (Lan, 2020; Nemirovskij & Yudin, 1983) to find $\bar { x } \in \mathbb { R } ^ { d }$ such that $\mathbb { E } [ f ( \bar { x } ) ] - f ( x _ { * } ) \leq \varepsilon$ , where $R : = \| x ^ { 0 } - x _ { * } \|$ . In the µ–strongly convex setting, the optimal complexity $\widetilde { \Theta } \left( \sqrt { L } / \sqrt { \mu } + \sigma ^ { 2 } / \mu ^ { 2 } \varepsilon \right)$ is to find $\bar { x } \in \mathbb { R } ^ { d }$ such that $\mathbb { E } [ \left. \bar { x } - x _ { * } \right. ^ { 2 } ] \leq \varepsilon \left( \mathbf { u p } \right.$ to logarithmic factors).

Oracle complexity with many workers. Many works discovered oracle complexities with multiple workers. Arjevani & Shamir (2015); Scaman et al. (2017) analyze the heterogeneous convex setting and provide lower bounds when the workers are synchronized. Lu & De Sa (2021) consider the similar setup but in the nonconvex setting. Arjevani et al. (2020) analyze settings where methods receive delayed stochastic gradients. Woodworth et al. (2018) provide lower bounds for parallel setups with intermittent communications and delayed updates. The primary limitation of these results is the assumption that all workers have consistent computational performance, without accounting for individual delays, random lags, or variations in performance over time.

Time complexity. To address the problem of analyzing methods with workers having different computation capabilities and performances, Mishchenko et al. (2022) proposed to consider the fixed computation model. In this model, it is assumed that

worker i requires at most $\tau _ { i }$ seconds to calculate one stochastic gradient.

Without loss of generality, we assume that the times are sorted: $\tau _ { 1 } \leq \cdots \leq \tau _ { n }$ . One of the most popular methods is Asynchronous SGD (Lian et al., 2015; Zhang et al., 2015; Feyzmahdavian et al., 2016; Sra et al., 2016; Dutta et al., 2018; Stich & Karimireddy, 2020; Wu et al., 2022; Islamov et al., 2024; Maranjyan et al., 2025; Maranjyan & Richtárik, 2026). In the homogeneous setting, Mishchenko et al. (2022); Koloskova et al. (2022); Cohen et al. (2021) showed that Asynchronous SGD and Picky SGD can provably improve the performance of the synchronized Minibatch SGD method that does the steps $\begin{array} { r } { x ^ { k + 1 } = x ^ { k } - \gamma \dot { / } \ i \sum _ { i = 1 } ^ { n } \dot { \nabla } f ( x ^ { k } ; \xi _ { i } ^ { k } ) } \end{array}$ , where $\gamma$ is a stepsize, $\xi _ { i } ^ { k }$ are i.i.d. samples, and $\nabla f ( x ^ { k } ; \xi _ { i } ^ { k } )$ are calculated in parallel in n workers. Minibatch SGD requires $\mathcal { O } \left( L \Delta / \varepsilon + \sigma ^ { 2 } L \Delta / n \varepsilon ^ { 2 } \right)$ iterations (Cotter et al., 2011; Goyal et al., 2017; Gower et al., 2019) in the nonconvex setting.

Algorithm 1 Malenia SGD or Rennala SGD when $w _ { i } ^ { k } = { 1 } / { B _ { i } ^ { k } } \mathrm { { ~ o r ~ } } w _ { i } ^ { k } = { { \mathrm { ~ } } ^ { n } } / { \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } }$ , respectively   
1: Input: point $x ^ { 0 }$ , stepsize γ, parameter $S ,$ weights $\{ w _ { i } ^ { k } \}$   
2: for $k = \mathbf { \bar { 0 } } , 1 , \dots , \bar { K } - 1$ do   
3: Ask all workers to calculate stochastic gradients at $x ^ { k } ;$ init $g _ { i } ^ { k } = 0$ and $B _ { i } ^ { k } = 0 \forall i \in [ n ]$   
4: while $\begin{array} { r } { \big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( w _ { i } ^ { k } ) ^ { 2 } B _ { i } ^ { k } \big ) ^ { - 1 } \le \frac { S } { n } } \end{array}$ do   
5: Wait for the next worker $j$   
6: Update $B _ { j } ^ { k } = B _ { j } ^ { k } + 1$   
7: Receive stochastic gradient $\nabla f _ { j } \big ( x ^ { k } ; \xi _ { j , B _ { j } ^ { k } } ^ { k } \big )$ and update $g _ { j } ^ { k } = g _ { j } ^ { k } + \nabla f _ { j } ( x ^ { k } ; \xi _ { j , B _ { j } ^ { k } } ^ { k } )$   
8: Ask this worker to calculate a stochastic gradient at $x ^ { k }$   
9: end while   
10: $\begin{array} { r } { g _ { w } ^ { k } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } g _ { i } ^ { k } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \nabla f _ { i } ( x ^ { k } ; \xi _ { i j } ^ { k } ) } \end{array}$   
11: $\boldsymbol { x } ^ { k + 1 } = \boldsymbol { x } ^ { k } - \gamma g _ { w } ^ { k }$   
12: Stop all the workers’ calculations (or ignore the unfinished calculations in the next iterations)   
13: end for

Moreover, Minibatch SGD converges after $\mathcal { O } \left( \operatorname* { m a x } _ { i \in [ n ] } \tau _ { i } \times \left( L \Delta / \varepsilon + \sigma ^ { 2 } L \Delta / n \varepsilon ^ { 2 } \right) \right)$ seconds because it waits for the slowest worker with max $\mathrm { { \dot { \iota } } } ( \in [ n ] ^ { \tau _ { i } }$ in every iteration. Asynchronous SGD, methods with the step $\begin{array} { r } { x ^ { k + 1 } = x ^ { k } - \gamma ^ { k } / n \sum _ { i = 1 } ^ { n } \nabla f ( x ^ { k - \delta _ { k } } ; \xi _ { i } ^ { k - \delta _ { k } } ) } \end{array}$ and $\delta _ { k } { \mathrm { - d e l a y } }$ yed stochastic gradients, improve this time complexity to $\mathcal { O } ( \left( 1 / n \sum _ { i = 1 } ^ { n } 1 / \tau _ { i } \right) ^ { - 1 } \left( L \Delta / \varepsilon + \sigma ^ { 2 } L \Delta / n \varepsilon ^ { 2 } \right) )$ .

Optimal time complexities in the heterogeneous and homogeneous settings. Surprisingly, the time complexity can be further improved. In the nonconvex setup (under Assumptions 1.1, and 1.4), Tyurin & Richtárik (2023) formalized the notion of time complexities and showed that the optimal time complexity is

$$
\begin{array} { r } { T _ { \mathrm { h o m o g } } : = \Theta \left( \underset { m \in [ n ] } { \operatorname* { m i n } } \left[ \left( \frac { 1 } { m } \underset { i = 1 } { \overset { m } { \sum } } \frac { 1 } { \tau _ { i } } \right) ^ { - 1 } \left( \frac { L \Delta } { \varepsilon } + \frac { \sigma ^ { 2 } L \Delta } { m \varepsilon ^ { 2 } } \right) \right] \right) } \end{array}\tag{2}
$$

seconds in the homogeneous setup to find an ε–stationary point, achieved by the Rennala SGD method, where, without loss of generality, the times are sorted: $\tau _ { 1 } \leq \cdots \leq \tau _ { n }$ . In the heterogeneous setup, the optimal time complexity is

$$
T _ { \mathrm { h e t e r } } : = \Theta \left( \tau _ { n } \frac { L \Delta } { \varepsilon } + \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { \sigma ^ { 2 } L \Delta } { n \varepsilon ^ { 2 } } \right) ,\tag{3}
$$

achieved by the Malenia SGD method.

A unifying perspective on Rennala SGD and Malenia SGD. Let us look closer to the Rennala SGD and Malenia SGD methods (see Algorithm 1) that achieve the optimal time complexities (2) and (3) in the homogeneous and heterogeneous setting, accordingly. We now recall how they work. In every iteration, Rennala SGD and Malenia SGD ask all workers to calculate stochastic gradients asynchronously at the same iterate $x ^ { k }$ . Assume that worker i has calculated $B _ { i } ^ { k }$ stochastic gradients for all $i \in [ n ]$ at the iteration $k .$ Then the methods do the steps

$$
\begin{array} { r } { x ^ { k + 1 } = x ^ { k } - \gamma g _ { \mathsf { R } } ^ { k } , \quad g _ { \mathsf { R } } ^ { k } : = \frac { 1 } { \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } } \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \nabla f _ { i } ( x ^ { k } ; \xi _ { i j } ^ { k } ) } \end{array}\tag{Rennala SGD}
$$

and

$$
\begin{array} { r } { x ^ { k + 1 } = x ^ { k } - \gamma g _ { \mathsf { M } } ^ { k } , \quad g _ { \mathsf { M } } ^ { k } : = \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \frac { 1 } { B _ { i } ^ { k } } \displaystyle \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \nabla f _ { i } ( x ^ { k } ; \xi _ { i j } ^ { k } ) , } \end{array}\tag{Malenia SGD}
$$

accordingly. Rennala SGD and Malenia SGD ask all workers calculating stochastic gradients until $\textstyle { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } > S / n$ and $\begin{array} { r } { \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { 1 } / { B _ { i } ^ { k } } \right) ^ { - 1 } > S / n } \end{array}$ correspondingly, where $S$ is a parameter. Hence, both methods asynchronously collect and aggregate stochastic gradients to compute $g _ { \mathsf { R } } ^ { k }$ and $g _ { \mathsf { M } } ^ { k }$ , and then perform a descent step. However, the way the methods aggregate is both different and important. It turns out the variance of the Rennala SGD’s update is smaller. Indeed, one can easily show that

$$
\mathbb { E } \left[ \left. g _ { \mathsf { R } } ^ { k } - \mathbb { E } \left[ g _ { \mathsf { R } } ^ { k } \right] \right. ^ { 2 } \right] \leq \frac { \sigma ^ { 2 } } { n } \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } \right) ^ { - 1 } \mathrm { ~ a n d ~ } \mathbb { E } \left[ \left. g _ { \mathsf { M } } ^ { k } - \mathbb { E } \left[ g _ { \mathsf { M } } ^ { k } \right] \right. ^ { 2 } \right] \leq \frac { \sigma ^ { 2 } } { n } \left( \frac { n } { \sum _ { i = 1 } ^ { n } \frac { 1 } { B _ { i } ^ { k } } } \right) ^ { - 1 } .
$$

Thus, the variance of Rennala SGD improves with the arithmetic mean of $B _ { i } ^ { k }$ , while the variance of Malenia SGD improves with the harmonic mean of $B _ { i } ^ { k }$ , which can be much smaller. Why wouldn’t we use Rennala SGD in all scenarios if it is better? Because $g _ { \mathsf { R } } ^ { k }$ is biased if $\{ f _ { i } \}$ are non-homogeneous. In general, $\mathbb { E } \left[ g _ { \mathsf { R } } ^ { k } \right] \neq \nabla f ( x ^ { k } )$ ), while it is always true that $\dot { \mathbb { E } } \left[ g _ { \mathsf { M } } ^ { k } \right] = \dot { \nabla f } ( \dot { x } ^ { k } )$ . Both methods can be generalized into $\begin{array} { r } { x ^ { k + 1 } = x ^ { k } - \gamma g _ { w } ^ { k } , g _ { w } ^ { k } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \nabla f _ { i } ( x ^ { k } ; \xi _ { i j } ^ { k } ) } \end{array}$ , where the weights $\{ w _ { i } ^ { k } \}$ are free parameters. If we take $\begin{array} { r } { \boldsymbol { w } _ { i } ^ { k } = \boldsymbol { n } / \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } } \end{array}$ for all $i \in [ n ]$ , we get Rennala SGD with small variance. If we take $w _ { i } ^ { k } = { ^ { 1 } } / { B _ { i } ^ { k } }$ , we get Malenia SGD with high variance but with an unbiased estimator. The weights enable interpolation between the methods.

Difference between the two settings. Using the inequality of arithmetic and harmonic means, one can easily show that $T _ { \mathrm { h o m o g } } \leq T _ { \mathrm { h e t e r } }$ (ignoring constant factors). At the same time, the gap between the complexities can be arbitrarily huge. Indeed, when the performance $\tau _ { 1 }$ of the fastest worker tends to 0, one can easily show that $\bar { T _ { \mathrm { h o m o g } } }  0$ and $T _ { \mathrm { h e t e r } }  \dot { \Theta } ( \tau _ { n } L \Delta / \varepsilon + ( \frac { \bar { 1 } } { n } \sum _ { i = 2 } ^ { n } \tau _ { i } ) \sigma ^ { 2 } L \Delta / n \varepsilon ^ { 2 } )$ , and $T _ { \mathrm { h e t e r } }$ improves by at most $\textstyle \sum _ { i = 1 } ^ { n } \tau _ { i } / \sum _ { i = 2 } ^ { n } \tau _ { i } \leq 2$ . While the improvement in the homogeneous setup is ∞. Consider another example when the performance $\tau _ { n }$ of the slowest worker (straggler) tends to ∞. Then $T _ { \mathrm { h e t e r } }  \infty$ and $T _ { \mathrm { h o m o g } }  \Theta ( \mathrm { m i n } _ { m \in [ n - 1 ] } [ ( 1 / m \sum _ { i = 1 } ^ { m } 1 / \tau _ { i } ) ^ { - 1 } ( L \Delta / _ { \varepsilon } + \sigma ^ { 2 } L \Delta / _ { m \varepsilon ^ { 2 } } ) ] )$ , so the complexity $T _ { \mathrm { h o m o g } }$ is robust to stragglers unlike $T _ { \mathrm { h e t e r } }$

Arbitrarily computation dynamics. The previous discussion explain that a significant gap appears between homogeneous and heterogeneous problems under the fixed computation model. This “arithmetic mean vs harmonic mean $\mathrm { g a p ' }$ was also observed in (Tyurin, 2025), where the author generalizes the fixed computation model to the universal computation model, accounting for potential disruptions caused by hardware or network delays, and any variations in computation speeds. For simplicity, in this work, we will continue working with the fixed computation model, but we also show how our final results translate to the universal computation model in Section A.

Convex world. When we want to find a point x¯ such that E $[ f ( { \bar { x } } ) ] - f ^ { * } \leq \varepsilon$ in the convex setup, the gap is similar. The optimal time complexity in the homogeneous setup is

$$
\Theta \left( \operatorname* { m i n } _ { m \in \left[ n \right] } \left[ \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 1 } { \tau _ { i } } \right) ^ { - 1 } \left( \frac { \sqrt { L } R } { \sqrt { \varepsilon } } + \frac { \sigma ^ { 2 } R ^ { 2 } } { m \varepsilon ^ { 2 } } \right) \right] \right)\tag{4}
$$

seconds (Tyurin & Richtárik, 2023). While the optimal time complexity in the heterogeneous setup is

$$
\begin{array} { r } { \Theta \left( \tau _ { n } \frac { \sqrt { L } R } { \sqrt { \varepsilon } } + \left( \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { \sigma ^ { 2 } R ^ { 2 } } { n \varepsilon ^ { 2 } } \right) } \end{array}\tag{5}
$$

seconds under Assumptions 1.1, 1.3, and 1.4 (our new contribution, Theorem E.4; the final puzzle piece needed to reveal the systematic gap between the two settings). Both complexities are achieved by the accelerated versions of Rennala SGD and Malenia SGD accordingly.

Strongly convex world. Assume additionally that the function f is µ–strongly convex. Using reduction (Woodworth & Srebro, 2016), up to logarithmic factors, we can obtain the optimal time complexity

$$
\widetilde \Theta \left( \operatorname* { m i n } _ { m \in \left[ n \right] } \left[ \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 1 } { \tau _ { i } } \right) ^ { - 1 } \left( \sqrt { \frac { L } { \mu } } + \frac { \sigma ^ { 2 } } { m \varepsilon \mu } \right) \right] \right)\tag{6}
$$

in the homogeneous setting and the optimal time complexity

$$
\begin{array} { r } { \widetilde { \Theta } \left( \tau _ { n } \sqrt { \frac { L } { \mu } } + \left( \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { \sigma ^ { 2 } } { n \varepsilon \mu } \right) } \end{array}\tag{7}
$$

in the heterogeneous setting when we want to find a point x¯ such that $\mathbb { E } \left[ f ( \bar { x } ) \right] - f ^ { * } \leq \varepsilon$ . Here we also observe a large gap between the settings. Note that the complexities (3), (5), and $( 7 )$ can only be improved under additional assumptions because they are optimal.

Main question: Having the systematic gap between the homogeneous and heterogeneous setups, the goal of this work is to identify theoretical assumptions that are as weak as possible to improve the results of asynchronous methods in heterogeneous scenarios. Under which assumptions can we improve the dependence on the arithmetic mean of $\{ \tau _ { i } \}$ (see (3), (5), and (7)) to the dependence on the harmonic mean of $\{ \tau _ { i } \}$ (see (2), (4), and $( 6 ) ) ?$ Right now, the only possible way is to assume that the functions $\{ f _ { i } \}$ are equal—an assumption we clearly want to avoid in the heterogeneous setting. Is there any chance to relax this assumption?

## 1.2 CONTRIBUTIONS

To the best of our knowledge, this is the first work to address the main question in any setting; our analysis considers the convex setting under standard Assumptions 1.1, 1.2, 1.3, 1.4, and 1.5.

Analysis of first- and second-order similarity. First, we consider the celebratedfirst- and secondorder similarity and, surprisingly, prove that even under these assumptions—no matter how close the functions $\{ f _ { i } \}$ are—any randomized algorithm cannot converge before $\Omega \Bigl ( \Bigl ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tau _ { i } \Bigr ) \frac { \sigma ^ { 2 } } { n \mu ^ { 2 } \varepsilon } \Bigr )$ seconds for small ε (Theorem 2.3). Thus, it is infeasible to break the dependence on the arithmetic mean of $\{ \tau _ { i } \}$ under these assumptions.

Investigation of the interpolation assumption. Inspired by Theorem 2.3, which provides a construction with local functions having different minimizers, we decided to go in another direction and consider the interpolation assumption. Thus, we introduce two additional assumptions, strong interpolation and the local Polyak-Łojasiewicz condition, and prove that it is impossible to drop either of these assumptions for improvement (Theorems 3.6 and 3.7).

Bridging the gap. By identifying this minimal set of assumptions, we derive a new time complexity result that matches the dependence on worker computation times in the best-known bound in the homogeneous setting (Theorem 3.8), but without requiring the functions $f _ { i }$ to be identical. Our theoretical results are validated numerically in Section H.

To bridge the gap in Section 3.2, we need to introduce Assumptions 3.3 and 3.4. However, our primary goal was to illustrate and prove that these assumptions are indeed necessary. Merely stating the assumptions might not be convincing; this is why the central part ofour paper investigates different assumptions and shows that most ofthem do not allow us to bridge the gap. While previous work noted the existence ofthe gap, our contribution goesfurther by systematically investigating which assumptions are sufficient and which are insufficient to eliminate it.

## 2 FIRST-ORDER AND SECOND-ORDER SIMILARITY DON’T HELP

The main problem with the arithmetic mean dependence in the heterogeneous setting is that this setting considers a worst-case scenario with arbitrarily heterogeneous functions. Due to the fact that Malenia SGD is optimal, we have to introduce assumptions to obtain faster convergence. One of the most popular assumptions in the literature isfirst-order and second-order similarity ofthefunctions (Arjevani & Shamir, 2015; Szlendak et al., 2021; Mishchenko et al., 2022):

Assumption 2.1 (First-Order Similarity). The functions f<sub>i</sub> satisfy $\begin{array} { r l r } { \operatorname* { m a x } _ { i , j \in [ n ] } \| \nabla f _ { i } ( x ) - \nabla f _ { j } ( x ) \| ^ { 2 } } & { \leq } & { \delta _ { 1 } } \end{array}$ for all $ { \boldsymbol { { x } } } ^ { \mathrm { ~ ~ } } \in  { \mathbb ~ { R } } ^ { d }$ for some $\delta _ { 1 } \geq 0 .$ It implies $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| \nabla f _ { i } ( x ) - \nabla f ( x ) \| ^ { 2 } \leq \delta _ { 1 } } \end{array}$ for all $x \in \mathbb { R } ^ { d }$

Assumption 2.2 (Second-Order Similarity). The functions f<sub>i</sub> satisfy $\begin{array} { r } { \operatorname* { m a x } _ { i , j \in [ n ] } \left\| \nabla ^ { 2 } f _ { i } ( x ) - \nabla ^ { 2 } f _ { j } ( x ) \right\| ^ { 2 } ~ \leq ~ \delta _ { 2 } } \end{array}$ for all $ { \boldsymbol { { x } } } ^ { \mathrm { ~ ~ } } \in ~ \mathbb { R } ^ { d }$ for some $\delta _ { 2 } \geq 0$ . It implies $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left\| \nabla ^ { 2 } f _ { i } ( x ) - \nabla ^ { 2 } f ( x ) \right\| ^ { 2 } \leq \delta _ { 2 } } \end{array}$ for all $x \in \mathbb { R } ^ { d }$

One might expect that when both $\delta _ { 1 }$ and $\delta _ { 2 }$ are small, it would be possible to exploit the similarity and design a method with smaller variance and better dependence on $\{ \tau _ { i } \}$ . Surprisingly, this is not the case: for any $\delta _ { 1 } > 0$ and $\delta _ { 2 } \geq 0$ , one can construct a problem for which the convergence speed of Malenia SGD (Theorem F.2) cannot be improved, up to logarithmic factors, in small-ε regimes:

Theorem 2.3 (Lower Bound). Consider stochastic gradients $\nabla f _ { i } ( x ; \xi _ { i } ) = \nabla f _ { i } ( x ) + \xi _ { i } e _ { i }$ with $\xi _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ for all $i \in [ n ]$ , and $x \in \mathbb { R } ^ { n }$ . Consider any randomized algorithm that has access only to the stochastic gradients (randomized method), which starts at $x ^ { 0 } = 0 ,$ , under thefixed computation model and any $R , \mu , \beta , \sigma , \varepsilon > 0$ such that $\begin{array} { r } { 0 < \varepsilon \le { \frac { c \beta ^ { 2 } } { \mu ^ { 2 } n ^ { 2 } } } } \end{array}$ and $\begin{array} { r } { R ^ { 2 } \ge \frac { \beta ^ { 2 } } { \mu ^ { 2 } n } } \end{array}$ , where $c > 0$ is a universal constant. For any time budget $t \leq c _ { 0 } \left( { \textstyle { \frac { 1 } { n } } } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) { \frac { \sigma ^ { 2 } } { n \mu ^ { 2 } \varepsilon } }$ , where $c _ { 0 }$ is a universal constant, there exist $f _ { i } ( x ) : \mathbb { R } ^ { n }  { \mathrm { ~ } }$ R such that $\begin{array} { r } { f _ { i } ( x ) = \frac { \mu } { 2 } \left. x \right. ^ { 2 } - \beta \varphi _ { i } \left. x , e _ { i } \right. } \end{array}$ and $\varphi _ { i } \in [ - 1 , 1 ]$ . Assumptions $I . I , \ I . 2 ,$ $I . 3 , I . 4 ,$ and 1.5 hold. Moreover, Assumption 2.1 (thefirst-order similarity) is satisfied with $\delta _ { 1 } = 2 \beta ^ { 2 }$ and Assumption 2.2 (the second-order similarity) is satisfied with $\delta _ { 2 } = 0 , \left\| x ^ { 0 } - x ^ { * } \right\| ^ { 2 } \leq R ^ { 2 }$ , and the method cannot produce a point x¯ such that E $[  { \bar { x } } - x ^ { * }   ^ { 2 } ] \leq \varepsilon$ within t seconds, where $x ^ { * }$ is the minimizer off.

Hence, for any small $\delta _ { 1 } > 0$ and $\delta _ { 2 } \geq 0$ , the convergence speed cannot be improved over that of Malenia SGD up to logarithmic factors in small-ε regimes (compare to Theorem F.2). Due to the construction in Theorem 2.3, we can choose any $\beta > 0$ , and hence any $\delta _ { 1 } > 0$ . No matter how close the functions are to each other, the lower bound does not allow us to break the pessimistic time complexity. In view of this, additional assumptions about the first- and second-order similarity will not help to improve the time complexity of Malenia SGD.

Remark 2.4. For the construction in Theorem 2.3, we can also show that $\left\| \nabla f _ { i } ( x ) \right\| ^ { 2 } \leq 2 \left\| \nabla f ( x ) \right\| ^ { 2 } +$ $2 \beta ^ { 2 } d$ for all $i \in [ n ]$ , which corresponds to the $\rho { - } s t r o n g$ growth condition when $\beta = 0$ and $\rho = 2$ (Schmidt & Roux, 2013). Since Theorem 2.3 holds for all $\beta > 0$ , we have proved the result for $\mathrm { { a } ^ { * } { \tilde { s l i g h t l y } } ^ { * } }$ broader class of problems and have “almost” established that, even under the strong growth condition, the convergence speed of Malenia SGD cannot be improved up to logarithmic factors. Whether a similar result holds for the class of problems satisfying max $i \in [ n ] \left\| \nabla f _ { i } ( x ) \right\| ^ { 2 } \leq$ $2 \parallel \nabla f ( x ) \parallel ^ { 2 }$ for all $x \in \mathbb { R } ^ { d }$ remains an important open research question.

Takeaway 1: Even with first-order and second-order similarity, for any randomized algorithm, there is still no hope of improving upon the convergence speed of Malenia SGD, up to logarithmic factors, in small-ε regimes.

## 3 UNDERSTANDING THE GAP VIA INTERPOLATION ASSUMPTIONS

Looking at Takeaway 1, we see that a different similarity assumption is required to close the gap between the heterogeneous and homogeneous results. Recall Theorem 2.3. The local minima of the functions $f _ { i }$ are not the same. This motivates us to explore an alternative assumption known as the interpolation assumption (Vaswani et al., 2019). This assumption provides another way to capture the similarity among the functions $f _ { i }$ by requiring that they share the same set of minimizers as the function $f .$

Assumption 3.1 (Weak Interpolation). $\operatorname { I f } x ^ { * }$ is a minimizer of $f ,$ that is, $\nabla f ( x ^ { * } ) = 0$ , then $x ^ { * }$ is also a minimizer of each $f _ { i }$ for all $i \in [ n ]$

Interpolation is a property of the solutions of $f _ { i }$ , whereas the heterogeneity assumptions, Assumptions 2.1 and 2.2, concern the gradients and Hessians. These are different characteristics of $f _ { i }$ (see Remark 3.5). Assumption 3.1 is considered practical in modern optimization literature, as there is evidence that it holds for large deep learning models (Zou & Gu, 2019; Zhang et al., 2021). However, as we show next, this assumption alone is not sufficient to achieve improved time complexity, leading to yet another pessimistic result:

Theorem 3.2 (Lower Bound). Consider any randomized method under thefixed computation model and assume that $n \geq 2$ . Let us fix any $\varepsilon , L _ { \mathrm { m a x } } , R , \mu , \sigma ^ { 2 } > 0$ such that $\mu < L _ { \operatorname* { m a x } } / ( 2 n ) , \varepsilon < 0 . 0 1$ and $R > 1 0$ . For any time budget $\begin{array} { r } { t \le c _ { 0 } \left( \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { \sigma ^ { 2 } } { \varepsilon n \mu ^ { 2 } } } \end{array}$ , where $c _ { 0 }$ is a universal constant, there existfunctions $\{ f _ { i } \}$ and stochastic gradients $\{ \nabla f _ { i } ( \cdot ; \cdot ) \}$ such that $\{ f _ { i } \}$ satisfy Assumptions 1.2, 1.3, and 3.1, f satisfies Assumptions 1.1 and 1.5 with $L = L _ { \mathrm { m a x } } , \{ \nabla f _ { i } ( \cdot ; \cdot ) \}$ satisfy Assumption 1.4 such that the method cannotfind ε–solution in terms ofdistances to the solution set after t seconds, when the method starts at a point in a distance less or equal to $R$ to the closest solution.

Table 1: The summary of our results and the time complexities (up to logarithmic factors) to get a point x¯ such that $\mathbb { E } [ \left. \bar { x } - x _ { * } \right. ^ { 2 } ] \leq \varepsilon$ under the fixed computation model (worker i requires at most $\tau _ { i }$ seconds to calculate one stochastic gradient; $\tau _ { 1 } \leq \dots \leq \tau _ { n } )$ and Assumptions 1.1, 1.2, 1.3, 1.4, and 1.5, where $\bar { x } _ { * }$ is the closest solution to x. ¯ The table compares methods in the fully heterogeneous setting and lists the extra assumptions the methods require to work.
<table><tr><td>Method</td><td>Time Complexity Guarantees</td><td>Additional Assumptions</td></tr><tr><td>Minibatch SGD</td><td> $\begin{array} { r } { \tau _ { n } \left( \frac { L } { \mu } + \frac { \sigma ^ { 2 } } { n \varepsilon \mu ^ { 2 } } \right) } \end{array}$ </td><td></td></tr><tr><td>Asynchronous SGD (Mishchenko et al., 2022)</td><td> $\left( { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } { \frac { 1 } { \tau _ { i } } } \right) ^ { - 1 } \left( { \frac { L } { \mu } } + { \frac { \sigma ^ { 2 } } { n \varepsilon \mu ^ { 2 } } } \right)$ </td><td>{fi} are equal µ-strong convexity</td></tr><tr><td>Malenia SGD (Tyurin &amp; Richtárik, 2023) (Theorem F.2)</td><td> $\begin{array} { r } { \tau _ { n } \frac { L } { \mu } + \left( \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { \sigma ^ { 2 } } { n \varepsilon \mu ^ { 2 } } } \end{array}$ </td><td></td></tr><tr><td>Rennala SGD (Tyurin &amp; Richtárik, 2023) min (Theorem F.1) m∈[n]</td><td> $\Biggl [ \Biggl ( { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } { \frac { 1 } { \tau _ { i } } } \Biggr ) ^ { - 1 } \Biggl ( { \frac { L } { \mu } } + { \frac { \sigma ^ { 2 } } { m \varepsilon \mu ^ { 2 } } } \Biggr ) \Biggr ]$ </td><td>{fi} are equal</td></tr><tr><td colspan="3">Lower Bounds (new results)</td></tr><tr><td></td><td>Under the first-order and second-order similarity, the following results state that it is infeasible to improve Malenia SGD in small-ε regimes:</td><td></td></tr><tr><td colspan="3">Any randomized method  $\geq \left( { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) { \frac { \sigma ^ { 2 } } { n \varepsilon \mu ^ { 2 } } }$  Assumptions 2.1 and 2.2</td></tr><tr><td>(Theorem 2.3)</td><td></td><td>(first-order and second-order similarity don&#x27;t help) Under weak interpolation, the following result states that it is infeasible to improve Malenia SGD in small-ε regimes:</td></tr><tr><td colspan="3"></td></tr><tr><td>Any randomized method (Theorem 3.2)</td><td> $\geq \left( { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) { \frac { \sigma ^ { 2 } } { n \varepsilon \mu ^ { 2 } } }$ </td><td>Assumption 3.1</td></tr><tr><td colspan="3">The following results state that any randomized method can not improve Malenia SGD for small ε if we discard Assumption 3.3 or 3.4:</td></tr><tr><td>Any randomized method (Theorem 3.6)</td><td> $\geq \left( { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) { \frac { \sigma ^ { 2 } } { n \varepsilon \mu ^ { 2 } } }$ </td><td>Assumptions 3.1 and 3.4 (weak interpolation is not enough)</td></tr><tr><td>Any randomized method (Theorem 3.7)</td><td> $\geq \left( { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) { \frac { \sigma ^ { 2 } } { n \varepsilon \mu ^ { 2 } } }$ </td><td>Assumption 3.3</td></tr><tr><td colspan="3">Upper Bound (new result)</td></tr><tr><td colspan="3">The following results state that under Assumption 3.3 and 3.4 it is possible to improve Malenia SGD:</td></tr><tr><td>Rennala SGD min (Theorem 3.8) m∈[n]</td><td> $\Biggl [ \Biggl ( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 1 } { \tau _ { i } } \Biggr ) ^ { - 1 } \Biggl ( \frac { L _ { \mathrm { m a x } } } { \mu } + \frac { \sigma ^ { 2 } } { m \varepsilon \mu ^ { 2 } } \Biggr ) \Biggr ]$ </td><td>Assumptions 3.3 and  $3 . 4$  (weaker than the equality of functions {fi})</td></tr></table>

Thus, even under Assumption 3.1, we can not improve the arithmetic mean dependence on $\{ \tau _ { i } \}$  
Takeaway 2: Using the weak interpolation assumption, which captures the similarity of the functions in a different way compared to first-order and second-order similarity, it is still infeasible to improve the pessimistic dependence on $\{ \tau _ { i } \}$ achieved by Malenia SGD using any randomized algorithm.

## 3.1 STRONG INTERPOLATION AND LOCAL PŁ CONDITION ARE BOTH REQUIRED

Once again, we need to go deeper and introduce additional assumptions to break the lower bound from Theorem 3.2. To further investigate the problem, we now turn to two related assumptions.

Assumption 3.3 (Strong Interpolation). For all $i \in [ n ]$ , a point $x ^ { * }$ is a minimizer of $f ,$ , that is, $\nabla f ( x ^ { * } ) = 0 .$ , ifand only if it is also a minimizer of $f _ { i }$

This assumption is clearly stronger than the weak interpolation assumption since it requires all the functions to share the set of minimizers (see Remark 3.9).

Assumption 3.4 (Local Polyak-Łojasiewicz condition). There exists $\mu$ such that $\left\| \nabla f _ { i } ( x ) \right\| ^ { 2 } \geq$ $2 \mu \left( f _ { i } ( x ) - f _ { i } ^ { * } \right)$ for all $x \in \bar { \mathbb { R } ^ { d } }$ and for all $i \in [ n ]$ , where $f _ { i } ^ { * }$ is the finite optimal function value of $f _ { i }$

This assumption, unlike Assumption 1.5, requires each function to satisfy PŁ condition.

Remark 3.5. The similarity and interpolation assumptions are neither disjoint nor does one imply the other. For example, consider $\begin{array} { r } { f _ { i } ( \hat { x } ) = \frac { 1 } { 2 } x ^ { 2 } + c _ { i } ( \hat { 1 } - \cos x ) , 0 \le c _ { i } < \bar { 1 } } \end{array}$ . These functions have the same unique minimizer $x ^ { \star } = 0$ and are strongly convex and smooth, and thus satisfy strong interpolation and the local PŁ condition. At the same time, $| f _ { i } ^ { \prime } ( x ) - f _ { j } ^ { \prime } ( x ) | = | c _ { i } - c _ { j } | |$ sin x|, $| f _ { i } ^ { \prime \prime } ( x ) - f _ { j } ^ { \prime \prime } ( x ) | = | c _ { i } - c _ { j } | |$ cos x|, so the first- and second-order similarity assumptions also hold. Conversely, the construction in Theorem 2.3 satisfies the similarity assumptions while interpolation fails. In the other direction, the functions $f _ { i } ( x ) = a _ { i } x ^ { 2 } / 2 , a _ { i } > 0$ , have the same minimizer and satisfy the local PŁ condition, whereas for $a _ { i } \neq a _ { j }$ their gradient difference $| ( a _ { i } - a _ { j } ) x |$ is unbounded. Thus, interpolation-type and similarity assumptions describe different, overlapping forms of heterogeneity.

It turns out again that if we do not assume both Assumption 3.3 and Assumption 3.4, then it is infeasible for any randomized method to get a time complexity faster than in Malenia SGD (Theorem F.2) for ε small enough. This statement is formalized in the following two theorems.

Theorem 3.6 (Lower Bound). Consider any randomized method under thefixed computation model and assume that $n \geq 2$ . Let us fix any $\varepsilon , L _ { \mathrm { m a x } } , R , \mu , \sigma ^ { 2 } > 0$ such that $\mu < L _ { \operatorname* { m a x } } / ( 2 n ) , \varepsilon < 0 . 0 1$ and $R > 1 0$ . For any time budget $t \leq c _ { 0 } \left( { \textstyle { \frac { 1 } { n } } } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) { \frac { \sigma ^ { 2 } } { \varepsilon n \mu ^ { 2 } } }$ , where $c _ { 0 }$ is a universal constant, there existfunctions $\{ f _ { i } \}$ and stochastic gradients $\{ \nabla f _ { i } ( \cdot ; \cdot ) \}$ such that $\{ f _ { i } \}$ satisfy Assumptions 1.2, 1.3, $3 . I ,$ , and 3.4 (Assumption 3.3 is not imposed and may or may not hold), f satisfies Assumptions 1.1 and 1.5 with $L = \hat { L } _ { \mathrm { m a x } } , \{ \nabla f _ { i } ( \cdot ; \cdot ) \}$ satisfy Assumption 1.4 such that the method cannot find $\varepsilon -$ solution in terms ofdistances to the solution set after t seconds, when the method starts at a point in a distance less or equal to R to the closest solution.

Theorem 3.7 (Lower Bound). Consider any randomized method under thefixed computation model and assume that $n \geq 2$ . Let us fix any $\varepsilon , L _ { \mathrm { m a x } } , R , \mu , \sigma ^ { 2 } > 0$ such that $\mu < L _ { \operatorname* { m a x } } / ( 2 n ) , \varepsilon < 0 . 0 1$ and $R > 1 0$ . For any time budget $t \leq c _ { 0 } \left( { \textstyle { \frac { 1 } { n } } } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) { \frac { \sigma ^ { 2 } } { \varepsilon n \mu ^ { 2 } } }$ , where $c _ { 0 }$ is a universal constant, there existfunctions $\{ f _ { i } \}$ and stochastic gradients $\{ \nabla f _ { i } ( \cdot ; \cdot ) \}$ such that $\{ f _ { i } \}$ satisfy Assumptions 1.2, 1.3, $3 . I ,$ and 3.3 (Assumption 3.4 is not imposed and may or may not hold with parameter µ), f satisfy Assumptions 1.1 and 1.5 with $L = L _ { \mathrm { m a x } } , \{ \nabla f _ { i } ( \cdot ; \cdot ) \}$ satisfy Assumption $1 . 4 ,$ such that the method cannotfind ε–solution in terms ofdistances to the solution set after t seconds, when the method starts at a point in a distance less or equal to R to the closest solution.

Takeaway 3: Even when the weak interpolation assumption is combined with only one of Assumptions 3.3 and 3.4, we still obtain only the arithmetic mean dependence on $\{ \tau _ { i } \}$

Once we drop either Assumption 3.3 or Assumption 3.4, it becomes possible to construct a “bad” function (see the proof of theorems) that provides no room for any randomized method to improve.

## 3.2 FINALLY BRIDGING THE GAP

However, if assume that both Assumption 3.3 and Assumption 3.4 hold, then, finally, we can proof the convergence with harmonic-like dependence on $\{ \tau _ { i } \}$ :

Theorem 3.8 (Upper Bound). Let Assumptions 1.2, 1.3, 1.4, 3.3, 3.4 hold<sup>1</sup>. We choose $w _ { i } ^ { k } = $ n/P<sup>n</sup> B<sup>k</sup> for all $k \geq 0 , i \in [ n ]$ in Algorithm 1 (reduces to Rennala SGD). We take $\gamma = 1 / { _ { L \mathrm { m a x } } } , S =$ $4 \sigma ^ { 2 } / \mu L _ { \mathrm { m a x } } \varepsilon .$ , and run Rennala SGD $\begin{array} { r } { f o r k \ge \Omega \left( \frac { L _ { \mathrm { m a x } } } { \mu } \log \frac { R ^ { 2 } } { \varepsilon } \right) } \end{array}$ iterations, then E $\left[ { \left\| { x ^ { k + 1 } - x _ { * } ^ { k + 1 } } \right\| } ^ { 2 } \right] \leq$ $\varepsilon ,$ where $x _ { * } ^ { k + 1 }$ is the closest solution to $x ^ { k + \mathrm { { i } } }$ . Moreover, under the fixed computation model, the method requires $\begin{array} { r } { \mathcal { O } \left( \underset { m \in [ n ] } { \operatorname* { m i n } } \left[ \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 1 } { \tau _ { i } } \right) ^ { - 1 } \left( \frac { L _ { \operatorname* { m a x } } } { \mu } + \frac { \sigma ^ { 2 } } { m \varepsilon \mu ^ { 2 } } \right) \right] \log { \frac { R ^ { 2 } } { \varepsilon } } \right) } \end{array}$ seconds.

Under weaker assumptions, without requiring the equality of the functions $\{ f _ { i } \}$ , this theorem yields time complexity guarantees with a “harmonic”-like dependence on the times $\{ \tau _ { i } \}$ for the Rennala SGD method, improving upon the previous theoretical results in Theorem F.1 and (Tyurin & Richtárik, 2023). Notice that the method in Theorem 3.8 is still biased because E $\begin{array} { r l } {  { [ \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \nabla f _ { i } ( x ; \xi _ { i j } ^ { k } ) / \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } ] \neq \nabla f ( x ) } } \end{array}$ in general. That said, we can successfully prove the theorem under this constraint. One of the primary reasons for this is the right choice of convergence metric. Initially, we aimed to analyze the biased gradient estimator in terms of function values and gradient norms, trying to prove that the method returns a point x¯ such that $\mathbb { E } [ f ( \bar { x } ) ] - f ^ { * } \leq \varepsilon$ or $\mathbb { E } [ \left. \nabla f ( \bar { x } ) \right. ^ { 2 } ] \leq \varepsilon$ . However, the more appropriate approach is to show $\mathbb { E } [ \left. \bar { x } - x _ { * } \right. ^ { 2 } ] \leq \varepsilon$ . Using this convergence metric allows us to analyze the biased gradient estimator. This observation can be important on its own. Note that we can get convergence in terms of $\mathbb { E } [ f ( \bar { x } ) ] - f ^ { * } \leq \varepsilon$ using L–smoothness, but the result would be loose. One interesting observation is that we do not observe a regime where any other method or strategy improves upon both Malenia SGD and Rennala SGD.

Takeaway 4: Improving the pessimistic dependence in Malenia SGD is possible with Rennala SGD and the additional assumptions, Assumption 3.3 and Assumption 3.4, in convex optimization.

Remark 3.9. Theorem 3.8 is proved under the strong interpolation assumption. This assumption is essential for our result: even relaxing strong interpolation to weak interpolation is insufficient to improve the pessimistic time complexity achieved by Malenia SGD. At the same time, strong interpolation does not require the local functions $f _ { i }$ to be identical. For example, consider the least-squares problems $\begin{array} { r } { f _ { i } ( x ) \stackrel { * } { = } \frac { 1 } { 2 } \| \mathbf { A } _ { i } ( x - x ^ { \star } ) \| ^ { 2 } } \end{array}$ , where the matrices $\mathbf { A } _ { i }$ can be different but have a common kernel N. Then argmin $f _ { i } = x ^ { \star } + N$ for every worker i, so the strong interpolation assumption holds. Thus, this setting provides a concrete machine-learning example with heterogeneous, non-identical local functions covered by Theorem 3.8.

## 3.3 EXTENSION TO NONCONVEX AND GENERAL CONVEX OPTIMIZATION

Although in this paper we focus on the convergence metric $\mathbb { E } [ \| x ^ { k } - x _ { * } \| ^ { 2 } ]$ under Assumptions 1.1, 1.2, 1.3, 1.4, and 1.5, we briefly sketch how the constructions behind Theorems 2.3, 3.2, 3.6, and 3.7 may extend to general convex optimization in terms of function values and to nonconvex optimization in terms of gradient norms. Indeed, in the general convex optimization setting, we typically consider Assumptions 1.1, 1.3, and 1.4. The same constructions can be considered with ${ \bar { f ( x ) } } - { \bar { f } } ^ { * } = { \textstyle \frac { \mu } { 2 } } \left\| x - x ^ { * } \right\| ^ { 2 }$ , where $\mu = L$ . For sufficiently small ε, applying the construction in the proof of Theorem 3.7 with squared-distance accuracy $2 \varepsilon / \mu$ suggests a lower bound of order $\begin{array} { r } { \Omega \big ( \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { \sigma ^ { 2 } } { \varepsilon n L } \big ) } \end{array}$ seconds for finding x¯ such that $\mathbb { E } \left[ f ( \bar { x } ) \right] - f ^ { * } \leq \varepsilon$ . Similarly, applying the construction with squared-distance accuracy $\varepsilon / L ^ { 2 }$ suggests a lower bound of order $\begin{array} { r } { \Omega \big ( \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { \sigma ^ { 2 } } { \varepsilon n } \big ) } \end{array}$ seconds for finding x¯ such that $\mathbb { E } [ \left. \nabla f ( \bar { x } ) \right. ^ { 2 } ] \leq \varepsilon$ . In nonconvex optimization, we consider Assumptions 1.1 and 1.4, and may use the same (convex) construction to prove the lower bounds. These observations suggest that the importance of Assumptions 3.3 and 3.4 for improving the arithmeticmean dependence may extend to other optimization settings. However, obtaining tight dependence on other parameters such as L and ε would require a different construction and is left for future work.

## 4 CONCLUSIONS

In this work, we investigated various assumptions and setups in an effort to break the pessimistic dependence on $\{ \tau _ { i } \}$ achieved by Malenia SGD. We considered the first- and second-order similarity, strong growth, and interpolation assumptions. We proved that under the first- and second-order similarity assumptions, it is infeasible to improve the dependence on the arithmetic mean of {τ } with any randomized algorithm. We also showed that under weak interpolation (Assumption 3.1), it is likewise not possible for any randomized algorithm to improve upon the result of Malenia SGD. Subsequently, we presented new theoretical results that provide improved time complexity guarantees in the heterogeneous setting, without assuming that the functions $f _ { i }$ are identical (Theorem 3.8). These results are obtained under the standard assumptions of convex optimization, together with Assumptions 3.3 and 3.4. Importantly, we have not merely introduced these assumptions to close the gap, but have shown that neither Assumption 3.3 nor Assumption 3.4 can be dropped in general within our setting, highlighting the fundamental limits of heterogeneous stochastic optimization. At the same time, we acknowledge that strong interpolation is a restrictive assumption in heterogeneous settings, as it requires the local functions to share the same set of minimizers. Identifying this fundamental difficulty is one of the main goals of our work.

There are many unexplored directions that can build on our initial results and observations. While we focused on the most common assumptions in federated and distributed learning, our findings may inspire the development of new assumptions and settings where it is possible to improve upon Malenia

SGD. Moreover, our upper bounds and lower bounds were investigated in terms of $\mathbb { E } [ \| x ^ { k } - x _ { * } \| ^ { 2 } ]$ convergence, and the lower bounds are only tight up to logarithmic factors and in small-ε regimes. It would be interesting to see whether tight lower bounds can be obtained in terms of $\mathbb { E } [ \| \nabla f ( \mathbf { \check { x } } ^ { k } ) \| ^ { 2 } ]$ in the non-convex setting, and in terms of $\mathbb { E } [ f ( x ^ { k } ) ] - f ^ { * }$ for convex functions.

## REFERENCES

Yossi Arjevani and Ohad Shamir. Communication complexity of distributed convex learning and optimization. Advances in Neural Information Processing Systems, 28, 2015.

Yossi Arjevani, Ohad Shamir, and Nathan Srebro. A tight convergence analysis for stochastic gradient descent with delayed updates. In Algorithmic Learning Theory, pp. 111–132. PMLR, 2020.

Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Nathan Srebro, and Blake Woodworth. Lower bounds for non-convex stochastic optimization. Mathematical Programming, pp. 1–50, 2022.

Yair Carmon, John C Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points i. Mathematical Programming, 184(1):71–120, 2020.

Alon Cohen, Amit Daniely, Yoel Drori, Tomer Koren, and Mariano Schain. Asynchronous stochastic optimization robust to arbitrary delays. Advances in Neural Information Processing Systems, 34: 9024–9035, 2021.

Andrew Cotter, Ohad Shamir, Nati Srebro, and Karthik Sridharan. Better mini-batch algorithms via accelerated gradient methods. Advances in Neural Information Processing Systems, 24, 2011.

Sanghamitra Dutta, Gauri Joshi, Soumyadip Ghosh, Parijat Dube, and Priya Nagpurkar. Slow and stale gradients can win the race: Error-runtime trade-offs in distributed SGD. In International Conference on Artificial Intelligence and Statistics, pp. 803–812. PMLR, 2018.

Hamid Reza Feyzmahdavian and Mikael Johansson. Asynchronous iterations in optimization: New sequence results and sharper algorithmic guarantees. Journal ofMachine Learning Research, 24 (158):1–75, 2023.

Hamid Reza Feyzmahdavian, Arda Aytekin, and Mikael Johansson. An asynchronous mini-batch algorithm for regularized stochastic optimization. IEEE Transactions on Automatic Control, 61 (12):3740–3754, 2016.

Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, volume 1. MIT Press, 2016.

Robert Mansel Gower, Nicolas Loizou, Xun Qian, Alibek Sailanbayev, Egor Shulgin, and Peter Richtárik. SGD: General analysis and improved rates. In International Conference on Machine Learning, pp. 5200–5209. PMLR, 2019.

Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, 2016.

Xinmeng Huang, Yiming Chen, Wotao Yin, and Kun Yuan. Lower bounds and nearly optimal algorithms in distributed learning with communication compression. Advances in Neural Information Processing Systems (NeurIPS), 2022.

Rustem Islamov, Mher Safaryan, and Dan Alistarh. AsGrad: A sharp unified analysis of asynchronous-SGD algorithms. In International Conference on Artificial Intelligence and Statistics, pp. 649–657. PMLR, 2024.

Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximalgradient methods under the polyak-łojasiewicz condition. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2016, Riva del Garda, Italy, September 19-23, 2016, Proceedings, Part I 16, pp. 795–811. Springer, 2016.

Anastasia Koloskova, Sebastian U Stich, and Martin Jaggi. Sharper convergence guarantees for asynchronous SGD for distributed and federated learning. Advances in Neural Information Processing Systems (NeurIPS), 2022.

Jakub Konecný, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, andˇ Dave Bacon. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492, 2016.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical report, University of Toronto, Toronto, 2009.

Guanghui Lan. First-order and stochastic optimization methods for machine learning. Springer, 2020.

Xiangru Lian, Yijun Huang, Yuncheng Li, and Ji Liu. Asynchronous parallel stochastic gradient for nonconvex optimization. Advances in Neural Information Processing Systems, 28, 2015.

Yucheng Lu and Christopher De Sa. Optimal complexity in decentralized training. In International Conference on Machine Learning, pp. 7111–7123. PMLR, 2021.

Artavazd Maranjyan and Peter Richtárik. Ringleader asgd: The first asynchronous sgd with optimal time complexity under data heterogeneity. In International Conference on Learning Representations, volume 2026, pp. 4738–4768, 2026.

Artavazd Maranjyan, Alexander Tyurin, and Peter Richtárik. Ringmaster ASGD: The first asynchronous SGD with optimal time complexity. In International Conference on Machine Learning, 2025.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial Intelligence and Statistics, pp. 1273–1282. PMLR, 2017.

Konstantin Mishchenko, Francis Bach, Mathieu Even, and Blake Woodworth. Asynchronous SGD beats minibatch SGD under arbitrary delays. Advances in Neural Information Processing Systems (NeurIPS), 2022.

Arkadij Semenovic Nemirovskij and David Borisovich Yudin. Problem complexity and method ˇ efficiency in optimization. 1983.

Yurii Nesterov. Lectures on convex optimization, volume 137. Springer, 2018.

Kevin Scaman, Francis Bach, Sébastien Bubeck, Yin Tat Lee, and Laurent Massoulié. Optimal algorithms for smooth and strongly convex distributed optimization in networks. In International Conference on Machine Learning, pp. 3027–3036. PMLR, 2017.

Mark Schmidt and Nicolas Le Roux. Fast convergence of stochastic gradient descent under a strong growth condition. arXiv preprint arXiv:1308.6370, 2013.

Suvrit Sra, Adams Wei Yu, Mu Li, and Alex Smola. Adadelay: Delay adaptive distributed stochastic optimization. In Artificial Intelligence and Statistics, pp. 957–965. PMLR, 2016.

Sebastian U Stich and Sai Praneeth Karimireddy. The error-feedback framework: SGD with delayed gradients. Journal of Machine Learning Research, 21(237):1–36, 2020.

Rafał Szlendak, Alexander Tyurin, and Peter Richtárik. Permutation compressors for provably faster distributed nonconvex optimization. In International Conference on Learning Representations, 2021.

Alexander Tyurin. Tight time complexities in parallel stochastic optimization with arbitrary computation dynamics. In International Conference on Learning Representations (ICLR), 2025.

Alexander Tyurin and Peter Richtárik. Optimal time complexities of parallel stochastic optimization methods under a fixed computation model. Advances in Neural Information Processing Systems (NeurIPS), 2023.

Alexander Tyurin, Kaja Gruntkowska, and Peter Richtárik. Freya PAGE: First optimal time complexity for large-scale nonconvex finite-sum optimization with heterogeneous asynchronous computations. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Sharan Vaswani, Aaron Mishkin, Issam Laradji, Mark Schmidt, Gauthier Gidel, and Simon Lacoste-Julien. Painless stochastic gradient: Interpolation, line-search, and convergence rates. Advances in neural information processing systems, 32, 2019.

Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge university press, 2019.

Blake E Woodworth and Nati Srebro. Tight complexity bounds for optimizing composite objectives. Advances in Neural Information Processing Systems, 29, 2016.

Blake E Woodworth, Jialei Wang, Adam Smith, Brendan McMahan, and Nati Srebro. Graph oracle models, lower bounds, and gaps for parallel stochastic optimization. Advances in Neural Information Processing Systems, 31, 2018.

Xuyang Wu, Sindri Magnusson, Hamid Reza Feyzmahdavian, and Mikael Johansson. Delay-adaptive step-sizes for asynchronous learning. arXiv preprint arXiv:2202.08550, 2022.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM, 64(3):107–115, 2021.

Wei Zhang, Suyog Gupta, Xiangru Lian, and Ji Liu. Staleness-aware async-sgd for distributed deep learning. arXiv preprint arXiv:1511.05950, 2015.

Difan Zou and Quanquan Gu. An improved analysis of training over-parameterized deep neural networks. Advances in Neural Information Processing Systems, 32, 2019.

## CONTENTS

1 Introduction 1   
1.1 Previous work 2   
1.2 Contributions 5   
2 First-Order and Second-Order Similarity Don’t Help 5   
Understanding the Gap via Interpolation Assumptions 6   
3.1 Strong interpolation and local PŁ condition are both required . 7   
3.2 Finally bridging the gap . 8   
3.3 Extension to nonconvex and general convex optimization 9   
4 Conclusions 9   
A Arbitrarily computation dynamics 14   
B Proof of the Main Results 14   
C Proof of Lower Bounds 17   
D Auxiliary Results 20   
E Lower Bound in the Heterogeneous Convex Setting 20   
F Proof of Theorems F.1 and F.2 23   
G Assumptions 1.3, 3.3 and 3.4 imply Assumption 1.5 26   
H Experiments 27   
H.1 Without interpolation 27   
H.2 With interpolation 27   
H.3 ResNet-18 and CIFAR-10 . 28   
I Experiments Details 29   
I.1 Quadratic optimization task generation procedure 29   
I.2 Experiments with ResNet and CIFAR-10 . 29

## A ARBITRARILY COMPUTATION DYNAMICS

Our new result can be readily extended to the universal computation model. To encompass virtually all computation scenarios, assume that each worker i performs computations based on a computation power function $v _ { i } : \mathbb { R } _ { + } \to \mathbb { R } _ { + }$ . Then the number of stochastic gradients that worker i can calculate from a time $t _ { 0 }$ to a time $t _ { 1 }$ is an integral of the computation power $v _ { i }$ followed by the floor operation:

“# of stoch. grad. in

$$
[ t _ { 0 } , t _ { 1 } ] ^ { \flat } = \left\lfloor \int _ { t _ { 0 } } ^ { t _ { 1 } } v _ { i } ( \tau ) d \tau \right\rfloor .\tag{8}
$$

For instance, if worker i is inactive for the first t seconds and then active again, it would mean $v _ { i } ( \tau ) = 0$ for all $\tau \leq t$ and $v _ { i } ( \tau ) > 0$ for all $\tau > t$ . Using the universal computation model, we can prove the theorem:

Theorem A.1. Consider the assumptions, algorithm, and parameters from Theorem 3.8. Then, Rennala SGD converges after at most $\begin{array} { r } { \bar { t } _ { \left\lceil c \times \frac { L _ { \operatorname* { m a x } } } { \mu } \log \frac { R ^ { 2 } } { \varepsilon } \right\rceil } } \end{array}$ seconds, where the sequence $\left\{ \bar { t } _ { k } \right\}$ is defined recursively as $\bar { t } _ { k } : =$

$$
\operatorname* { m i n } \left\{ t \geq 0 : \sum _ { i = 1 } ^ { n } \left\lfloor \int _ { \bar { t } _ { k - 1 } } ^ { t } v _ { i } ( \tau ) d \tau \right\rfloor \geq \operatorname* { m a x } \left\{ \left\lceil 2 S \right\rceil , 1 \right\} \right\}\tag{9}
$$

for all $k \geq 1 ( \bar { t } _ { 0 } \equiv 0 )$ , and c is a universal constant.

A similar result was obtained in (Tyurin, 2025). However, Tyurin (2025) requires the equality of the functions $\{ f _ { i } \}$

## B PROOF OF THE MAIN RESULTS

Theorem 3.8 (Upper Bound). Let Assumptions 1.2, 1.3, 1.4, 3.3, 3.4 hold<sup>2</sup>. We choose $w _ { i } ^ { k } \ =$ $n { \Big / } { \sum } _ { i = 1 } ^ { n } B _ { i } ^ { k }$ for all $\bar { k } \ge 0 , i \in [ n ]$ in Algorithm 1 (reduces to Rennala SGD). We take $\gamma = 1 / { _ { L _ { \mathrm { m a x } } } } , \dot { S } =$ $4 \sigma ^ { 2 } / \mu L _ { \mathrm { m a x } } \varepsilon .$ , and run Rennala SGD for $\begin{array} { r } { k \geq \Omega \left( \frac { L _ { \mathrm { m a x } } } { \mu } \log \frac { R ^ { 2 } } { \varepsilon } \right) } \end{array}$ iterations, then $\mathbb { E } \left[ \left. x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right. ^ { 2 } \right] \leq$ $\varepsilon ,$ where $x _ { * } ^ { k + 1 }$ is the closest solution to $x ^ { k + \mathrm { { i } } }$ . Moreover, under the fixed computation model, the method requires $\begin{array} { r } { \mathcal { O } \left( \underset { m \in [ n ] } { \operatorname* { m i n } } \left[ \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 1 } { \tau _ { i } } \right) ^ { - 1 } \left( \frac { L _ { \operatorname* { m a x } } } { \mu } + \frac { \sigma ^ { 2 } } { m \varepsilon \mu ^ { 2 } } \right) \right] \log { \frac { R ^ { 2 } } { \varepsilon } } \right) } \end{array}$ seconds.

Proof. Let us define $x _ { * } ^ { k }$ as an euclidean projection of the point $x ^ { k + 1 }$ on to the solution set of the main problem $( 1 )$ , and take the condition expectation $\mathbb { E } _ { k } \left[ \cdot \right]$ w.r.t. the randomness from the iteration $k$ only. Then we have

$$
\mathbb { E } _ { k } \left[ \left. x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right. ^ { 2 } \right] \leq \mathbb { E } _ { k } \left[ \left. x ^ { k + 1 } - x _ { * } ^ { k } \right. ^ { 2 } \right]
$$

due to the projection’s properties. Then

$$
\begin{array} { r l } & { \mathbb { E } _ { k } \left[ \left\| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right\| ^ { 2 } \right] } \\ & { \leq \mathbb { E } _ { k } \left[ \Bigg \| x ^ { k } - \gamma \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \nabla f _ { i } ( x ^ { k } ; \xi _ { i j } ^ { k } ) - x _ { * } ^ { k } \Bigg \| ^ { 2 } \right] } \\ & { = \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } - 2 \gamma \mathbb { E } _ { k } \left[ \left. x ^ { k } - x _ { * } ^ { k } , \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \nabla f _ { i } ( x ^ { k } ; \xi _ { i j } ^ { k } ) \right. \right] + \gamma ^ { 2 } \mathbb { E } _ { k } \left[ \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \nabla f _ { i } ( x ^ { k } ; \xi _ { i j } ^ { k } ) \right\| ^ { 2 } \right] . } \end{array}
$$

Using unbiasedness (Assumption 1.4) and the variance decomposition equality, we get

$$
\mathbb { E } _ { k } \left[ \left. x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right. ^ { 2 } \right]
$$

$$
\begin{array} { r l } & { \leq \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } - 2 \gamma \left. x ^ { k } - x _ { * } ^ { k } , \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \nabla f _ { i } ( x ^ { k } ) \right. } \\ & { \quad + \gamma ^ { 2 } \left\| \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \nabla f _ { i } ( x ^ { k } ) \right\| ^ { 2 } + \gamma ^ { 2 } \mathbb { E } _ { k } \left[ \left\| \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } \displaystyle \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \big ( \nabla f _ { i } ( x ^ { k } ; \xi _ { i j } ^ { k } ) - \nabla f _ { i } ( x ^ { k } ) \big ) \right\| ^ { 2 } \right] . } \end{array}
$$

Consider the last term, due to the independence of stochastic gradients and Assumption 1.4, we ensure that

$$
\begin{array} { r l } & { \mathbb { E } _ { k } \left[ \left\| \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \big ( \nabla f _ { i } ( x ^ { k } ; \xi _ { i j } ^ { k } ) - \nabla f _ { i } ( x ^ { k } ) \big ) \right\| ^ { 2 } \right] } \\ & { = \displaystyle \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } ( w _ { i } ^ { k } ) ^ { 2 } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \mathbb { E } _ { k } \left[ \big \| \nabla f _ { i } ( x ^ { k } ; \xi _ { i j } ^ { k } ) - \nabla f _ { i } ( x ^ { k } ) \big \| ^ { 2 } \right] \leq \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } ( w _ { i } ^ { k } ) ^ { 2 } B _ { i } ^ { k } \sigma ^ { 2 } . } \end{array}
$$

Thus

$$
\begin{array} { l l } { \displaystyle \mathbb { E } _ { k } \left[ \left\| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right\| ^ { 2 } \right] } & { ( 1 0 ) } \\ { \displaystyle \leq \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } - \frac { 2 \gamma } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \left. x ^ { k } - x _ { * } ^ { k } , \nabla f _ { i } ( x ^ { k } ) \right. + \gamma ^ { 2 } \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \nabla f _ { i } ( x ^ { k } ) \right\| ^ { 2 } + \frac { \gamma ^ { 2 } } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } ( w _ { i } ^ { k } ) ^ { 2 } B _ { i } ^ { k } \sigma ^ { 2 } . } \end{array}
$$

We now consider the second and the third term. Since $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } = 1 } \end{array}$ , using Jensen’s inequality, we get

$$
\begin{array} { r l } & { - 2 \gamma \left. x ^ { k } - x _ { * } ^ { k } , \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \nabla f _ { i } ( x ^ { k } ) \right. + \gamma ^ { 2 } \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \nabla f _ { i } ( x ^ { k } ) \right\| ^ { 2 } } \\ & { \leq - 2 \gamma \left. x ^ { k } - x _ { * } ^ { k } , \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \nabla f _ { i } ( x ^ { k } ) \right. + \gamma ^ { 2 } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \left\| \nabla f _ { i } ( x ^ { k } ) \right\| ^ { 2 } . } \end{array}
$$

Due to Assumption 3.3, we get

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \left\| \nabla f _ { i } ( x ^ { k } ) \right\| ^ { 2 } = \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \left\| \nabla f _ { i } ( x ^ { k } ) - \nabla f _ { i } ( x _ { * } ^ { k } ) \right\| ^ { 2 } } \\ & { \displaystyle \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } L _ { i } \left. x ^ { k } - x _ { * } ^ { k } , \nabla f _ { i } ( x ^ { k } ) - \nabla f _ { i } ( x _ { * } ^ { k } ) \right. } \\ & { \displaystyle \leq L _ { \operatorname* { m a x } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \left. x ^ { k } - x _ { * } ^ { k } , \nabla f _ { i } ( x ^ { k } ) - \nabla f _ { i } ( x _ { * } ^ { k } ) \right. } \\ & { \displaystyle = L _ { \operatorname* { m a x } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \left. x ^ { k } - x _ { * } ^ { k } , \nabla f _ { i } ( x ^ { k } ) \right. . } \end{array}
$$

In the first inequality, we use Lemma D.1 under Assumption 1.2 and convexity (Assumption 1.3). In the second inequality, we use the bound $L _ { i } \ \leq \ L _ { \operatorname* { m a x } }$ for all $i \in [ n ]$ . Taking $\gamma \leq 1 / L _ { \operatorname* { m a x } }$ and substituting the last inequality to (10), we obtain

$$
\begin{array} { r l } & { \mathbb { E } _ { k } \left[ \left\| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right\| ^ { 2 } \right] } \\ & { \leq \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } - ( 2 \gamma - L _ { \operatorname* { m a x } } \gamma ^ { 2 } ) \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \left. x ^ { k } - x _ { * } ^ { k } , \nabla f _ { i } ( x ^ { k } ) \right. + \frac { \gamma ^ { 2 } } { n ^ { 2 } } \displaystyle \sum _ { i = 1 } ^ { n } ( w _ { i } ^ { k } ) ^ { 2 } B _ { i } ^ { k } \sigma ^ { 2 } } \\ & { \leq \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } - \gamma \displaystyle \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \left. x ^ { k } - x _ { * } ^ { k } , \nabla f _ { i } ( x ^ { k } ) \right. + \frac { \gamma ^ { 2 } } { n ^ { 2 } } \displaystyle \sum _ { i = 1 } ^ { n } ( w _ { i } ^ { k } ) ^ { 2 } B _ { i } ^ { k } \sigma ^ { 2 } . } \end{array}
$$

Using the convexity, Assumption 3.4, and Lemma D.2, we get

$$
\begin{array} { r l } & { \mathbb { E } _ { k } \left[ \left\| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right\| ^ { 2 } \right] } \\ & { \leq \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } - \frac { \gamma \mu } { 2 } \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } ^ { k } B _ { i } ^ { k } \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } + \frac { \gamma ^ { 2 } } { n ^ { 2 } } \displaystyle \sum _ { i = 1 } ^ { n } ( w _ { i } ^ { k } ) ^ { 2 } B _ { i } ^ { k } \sigma ^ { 2 } . } \end{array}
$$

We take $\begin{array} { r } { w _ { i } ^ { k } = n / \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } } \end{array}$ in the theorem for all $i \in [ n ]$ . Thus

$$
\mathbb { E } _ { k } \left[ \left. x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right. ^ { 2 } \right] \leq \left( 1 - \frac { \gamma \mu } { 2 } \right) \left. x ^ { k } - x _ { * } ^ { k } \right. ^ { 2 } + \frac { \gamma ^ { 2 } \sigma ^ { 2 } } { \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } } .
$$

In Algorithm 1, with the chosen weights $\{ w _ { i } ^ { k } \}$ , we wait for the moment when $\textstyle \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } > S$ . Thus

$$
\mathbb { E } _ { k } \left[ \left\| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right\| ^ { 2 } \right] \leq \left( 1 - \frac { \gamma \mu } { 2 } \right) \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } + \frac { \gamma ^ { 2 } \sigma ^ { 2 } } { S } .
$$

Unrolling the recursion and taking the full expectation, we obtain

$$
\begin{array} { r l r } {  { \mathbb { E } [ \| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \| ^ { 2 } ] \leq ( 1 - \frac { \gamma \mu } { 2 } ) ^ { k + 1 } \| x ^ { 0 } - x _ { * } ^ { 0 } \| ^ { 2 } + \sum _ { j = 0 } ^ { k } ( 1 - \frac { \gamma \mu } { 2 } ) ^ { j } \frac { \gamma ^ { 2 } \sigma ^ { 2 } } { S } } } \\ & { } & { \leq ( 1 - \frac { \gamma \mu } { 2 } ) ^ { k + 1 } \| x ^ { 0 } - x _ { * } ^ { 0 } \| ^ { 2 } + \frac { 2 \gamma \sigma ^ { 2 } } { \mu S } . } \end{array}
$$

Due the choice of $\gamma , S ,$ and the condition on $k ,$ we have E $\left\lceil \left\| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right\| ^ { 2 } \right\rceil \leq \varepsilon$

It is sufficient to run the method for

$$
\mathcal { O } \left( \frac { L _ { \mathrm { m a x } } } { \mu } \log \frac { R ^ { 2 } } { \varepsilon } \right)
$$

iterations. In each iteration, the method has to ensure that $\textstyle \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } > S$ . A sufficient time for that is

$$
2 \operatorname* { m i n } _ { m \in \left[ n \right] } \left[ \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 1 } { \tau _ { i } } \right) ^ { - 1 } \left( 1 + \frac { 4 \sigma ^ { 2 } } { m L _ { \operatorname* { m a x } } \varepsilon \mu } \right) \right] .
$$

under the fixed computation model (see Theorem 11 in (Tyurin et al., 2024)).

Theorem A.1. Consider the assumptions, algorithm, and parameters from Theorem 3.8. Then, Rennala SGD converges after at most $\begin{array} { r } { \bar { t } _ { \left\lceil c \times \frac { L _ { \operatorname* { m a x } } } { \mu } \log \frac { R ^ { 2 } } { \varepsilon } \right\rceil } } \end{array}$ seconds, where the sequence $\left\{ \bar { t } _ { k } \right\}$ is defined recursively as $\bar { t } _ { k } : =$

$$
\operatorname* { m i n } \left\{ t \geq 0 : \sum _ { i = 1 } ^ { n } \left\lfloor \int _ { \bar { t } _ { k - 1 } } ^ { t } v _ { i } ( \tau ) d \tau \right\rfloor \geq \operatorname* { m a x } \left\{ \left\lceil 2 S \right\rceil , 1 \right\} \right\}\tag{9}
$$

for all $k \geq 1 ( \bar { t } _ { 0 } \equiv 0 )$ , and c is a universal constant.

Proof. From the proof of Theorem 3.8, we know that it is sufficient to run the method for

$$
c \times { \frac { L _ { \operatorname* { m a x } } } { \mu } } \log { \frac { R ^ { 2 } } { \varepsilon } }
$$

iterations, where c is a universal constant. The method waits the moment when $\textstyle \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } > S$ in each iteration. The workers work in parallel, and for all $i \in [ n ]$ ], will calculate

$$
\left\lfloor \int _ { 0 } ^ { t } v _ { i } ( \tau ) d \tau \right\rfloor
$$

stochastic gradients after t seconds. In total, all workers will calculate $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \left\lfloor \int _ { 0 } ^ { t } v _ { i } ( \tau ) d \tau \right\rfloor } \end{array}$ stochastic gradients. Hence, the first iteration will end by

$$
\bar { t } _ { 1 } : = \operatorname* { m i n } \left\{ t \geq 0 : \sum _ { i = 1 } ^ { n } \left\lfloor \int _ { 0 } ^ { t } v _ { i } ( \tau ) d \tau \right\rfloor \geq \operatorname* { m a x } \left\{ \left\lceil 2 S \right\rceil , 1 \right\} \right\} ,
$$

seconds. After that, the second iteration starts before time $\bar { t } _ { 1 }$ and ends no later than time

$$
\bar { t } _ { 2 } : = \operatorname* { m i n } \left\{ t \geq 0 : \sum _ { i = 1 } ^ { n } \left\lfloor \int _ { \bar { t } _ { 1 } } ^ { t } v _ { i } ( \tau ) d \tau \right\rfloor \geq \operatorname* { m a x } \left\{ \left\lceil 2 S \right\rceil , 1 \right\} \right\} ,
$$

because worker i can calculate at least

$$
\left\lfloor \int _ { \bar { t } _ { 1 } } ^ { t } v _ { i } ( \tau ) d \tau \right\rfloor
$$

stochastic gradients between the end of the first iteration and a time t. Using the same reasoning, we can recursively define

$$
\begin{array} { r } { \bar { t } _ { 3 } , \ldots , \bar { t } _ { \left[ c \times \frac { L _ { \mathrm { m a x } } } { \mu } \log \frac { R ^ { 2 } } { \varepsilon } \right] } . } \end{array}
$$

The algorithm will converge by $\begin{array} { r } { \bar { t } _ { \left\lceil c \times \frac { L _ { \operatorname* { m a x } } } { \mu } \log \frac { R ^ { 2 } } { \varepsilon } \right\rceil } } \end{array}$ seconds due to the discussion at the beginning of the theorem. □

## C PROOF OF LOWER BOUNDS

Theorem 2.3 (Lower Bound). Consider stochastic gradients $\nabla f _ { i } ( x ; \xi _ { i } ) = \nabla f _ { i } ( x ) + \xi _ { i } e _ { i }$ with $\xi _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ for all $i \in [ n ]$ , and $x \in \mathbb { R } ^ { n }$ . Consider any randomized algorithm that has access only to the stochastic gradients (randomized method), which starts at $x ^ { 0 } = 0 \colon$ , under thefixed computation model and any $R , \mu , \beta , \sigma , \varepsilon > 0$ such that $\begin{array} { r } { 0 < \varepsilon \le { \frac { c \beta ^ { 2 } } { \mu ^ { 2 } n ^ { 2 } } } } \end{array}$ and $\begin{array} { r } { R ^ { 2 } \ge \frac { \beta ^ { 2 } } { \mu ^ { 2 } n } } \end{array}$ , where $c > 0$ is a universal constant. For any time budget $t \leq c _ { 0 } \left( { \textstyle { \frac { 1 } { n } } } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) { \frac { \sigma ^ { 2 } } { n \mu ^ { 2 } \varepsilon } }$ , where $c _ { 0 }$ is a universal constant, there exist $f _ { i } ( x ) : \mathbb { R } ^ { n } $ R such that $\begin{array} { r } { f _ { i } ( x ) = \frac { \mu } { 2 } \left. x \right. ^ { 2 } - \beta \varphi _ { i } \left. x , e _ { i } \right. } \end{array}$ and $\varphi _ { i } \in [ - 1 , 1 ]$ . Assumptions 1.1, 1.2, $I . 3 , I . 4 ,$ and 1.5 hold. Moreover, Assumption 2.1 (thefirst-order similarity) is satisfied with $\delta _ { 1 } = 2 \beta ^ { 2 }$ and Assumption 2.2 (the second-order similarity) is satisfied with $\delta _ { 2 } = 0 , \left\| x ^ { 0 } - x ^ { * } \right\| ^ { 2 } \leq R ^ { 2 }$ , and the method cannot produce a point x¯ such that E $[  { \bar { x } } - x ^ { * }   ^ { 2 } ] \leq \varepsilon$ within t seconds, where $x ^ { * }$ is the minimizer off.

Proof. We assume that $\varphi _ { i } \in [ - 1 , 1 ]$ for all $i \in [ n ]$ and define them later. The first-order similarity of these functions is

$$
\operatorname* { m a x } _ { x \in \mathbb { R } ^ { n } } \operatorname* { m a x } _ { i , j \in [ n ] } \| \nabla f _ { i } ( x ) - \nabla f _ { j } ( x ) \| ^ { 2 } \leq 2 \beta ^ { 2 } .
$$

Thus, the parameter $\beta$ from the construction controls this similarity. Taking $\beta$ small, we increase similarity between the functions. Notice that the second-order similarity between the functions is zero since $\nabla ^ { 2 } f _ { i } ( x ) = \mu \mathbf { I }$ for all $i \in [ n ]$ . The optimal point is $x ^ { * }$ such that $\begin{array} { r } { x _ { i } ^ { * } ~ = ~ \frac { \beta \varphi _ { i } } { \mu n } } \end{array}$ and $\begin{array} { r } { \left\| x ^ { 0 } - x ^ { * } \right\| ^ { 2 } \leq \frac { \beta ^ { 2 } } { \mu ^ { 2 } n } \leq R ^ { 2 } } \end{array}$

Let $\begin{array} { r } { N _ { i } ( t ) : = \left| \frac { t } { \tau _ { i } } \right| } \end{array}$ denote an upper bound on the number of stochastic gradients returned by worker i by time t. For every stochastic gradient returned by worker i, the method gets

$$
\nabla f _ { i } ( { \bar { x } } ; \xi ) = \mu { \bar { x } } - \beta \varphi _ { i } e _ { i } + \xi e _ { i } = \mu { \bar { x } } - \beta \left( \varphi _ { i } - { \frac { \xi } { \beta } } \right) e _ { i }\tag{11}
$$

where x¯ is a query point and $\xi \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . Therefore, all the information obtained from worker i about $\varphi _ { i }$ consists of at most $N _ { i } ( t )$ independent observations from $\begin{array} { r } { \mathcal { N } ( \varphi _ { i } , \frac { \sigma ^ { 2 } } { \beta ^ { 2 } } ) } \end{array}$ .

Let $\bar { x } _ { t }$ be the point returned by the algorithm at time t for the objective parametrized by $\varphi .$ . Then,

$$
\left\| { \bar { x } } _ { t } - x ^ { * } \right\| ^ { 2 } = \sum _ { i = 1 } ^ { n } \left( ( { \bar { x } } _ { t } ) _ { i } - { \frac { \beta \varphi _ { i } } { \mu n } } \right) ^ { 2 } = { \frac { \beta ^ { 2 } } { \mu ^ { 2 } n ^ { 2 } } } \sum _ { i = 1 } ^ { n } \left( { \frac { \mu n } { \beta } } ( { \bar { x } } _ { t } ) _ { i } - \varphi _ { i } \right) ^ { 2 } .
$$

Thus, we reduced the problem to estimating $\varphi _ { i }$ using the observations $\begin{array} { r } { \mathcal { N } ( \varphi _ { i } , \frac { \sigma ^ { 2 } } { \beta ^ { 2 } } ) } \end{array}$ . Using the classical statistical result $( \mathrm { e . g . }$ ., Example 15.4 from (Wainwright, 2019)), for a universal constant $c _ { 2 } > 0$ ，

$$
\operatorname* { s u p } _ { \varphi \in [ - 1 , 1 ] ^ { n } } \mathbb { E } \left[ \left. \bar { x } _ { t } - x ^ { * } \right. ^ { 2 } \right] \geq c _ { 2 } \frac { \beta ^ { 2 } } { \mu ^ { 2 } n ^ { 2 } } \sum _ { i = 1 } ^ { n } \operatorname* { m i n } \left\{ 1 , \frac { \sigma ^ { 2 } } { \beta ^ { 2 } N _ { i } ( t ) } \right\} ,
$$

where $\bar { x } _ { t }$ depends on $\varphi .$ Thus,

$$
\operatorname* { s u p } _ { \varphi \in [ - 1 , 1 ] ^ { n } } \mathbb { E } \left[ \left. \bar { x } _ { t } - x ^ { * } \right. ^ { 2 } \right] \geq c _ { 2 } \frac { \beta ^ { 2 } } { \mu ^ { 2 } n ^ { 2 } } \sum _ { i = 1 } ^ { n } \operatorname* { m i n } \left\{ 1 , \frac { \sigma ^ { 2 } \tau _ { i } } { \beta ^ { 2 } t } \right\} .
$$

We take

$$
T = c _ { 1 } \frac { \sigma ^ { 2 } } { \mu ^ { 2 } n ^ { 2 } \varepsilon } \sum _ { i = 1 } ^ { n } { \tau _ { i } } ,
$$

where $c _ { 1 }$ is a universal constant. Then,

$$
\operatorname* { s u p } _ { \varphi \in [ - 1 , 1 ] ^ { n } } \mathbb { E } \left[ \left. \bar { x } _ { t } - x ^ { * } \right. ^ { 2 } \right] \geq c _ { 2 } \frac { \beta ^ { 2 } } { \mu ^ { 2 } n ^ { 2 } } \sum _ { i = 1 } ^ { n } \operatorname* { m i n } \left\{ 1 , \frac { \sigma ^ { 2 } \tau _ { i } } { \beta ^ { 2 } T } \right\}
$$

due to the bound on t. The condition on ε ensures that $\begin{array} { r } { \frac { \sigma ^ { 2 } \tau _ { i } } { \beta ^ { 2 } T } \leq 1 } \end{array}$ for all $i \in [ n ]$ . Therefore,

$$
\operatorname* { s u p } _ { \varphi \in [ - 1 , 1 ] ^ { n } } \mathbb { E } \left[ \left\| { \bar { x } } _ { t } - x ^ { * } \right\| ^ { 2 } \right] \geq 2 \varepsilon .\tag{12}
$$

Finally, E $\left[ \left. \bar { x } _ { t } - x ^ { * } \right. ^ { 2 } \right] > .$ ε after t seconds for some $\varphi$ due to (12). The adversary can choose this $\varphi .$ □

Theorem 3.6 (Lower Bound). Consider any randomized method under thefixed computation model and assume that $n \geq 2$ . Let us fix any $\varepsilon , L _ { \mathrm { m a x } } , R , \mu , \sigma ^ { 2 } > 0$ such that $\mu < L _ { \operatorname* { m a x } } / ( 2 n ) , \varepsilon < 0 . 0 1$ and $R > 1 0$ . For any time budget $t \leq c _ { 0 } \left( { \textstyle { \frac { 1 } { n } } } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) { \frac { \sigma ^ { 2 } } { \varepsilon n \mu ^ { 2 } } }$ , where $c _ { 0 }$ is a universal constant, there existfunctions $\{ f _ { i } \}$ and stochastic gradients $\{ \nabla f _ { i } ( \cdot ; \cdot ) \}$ such that $\{ f _ { i } \}$ satisfy Assumptions 1.2, 1.3, $3 . I ,$ , and 3.4 (Assumption 3.3 is not imposed and may or may not hold), f satisfies Assumptions 1.1 and 1.5 with $L = \hat { L } _ { \mathrm { m a x } } , \{ \nabla f _ { i } ( \cdot ; \cdot ) \}$ satisfy Assumption 1.4 such that the method cannot find $\varepsilon -$ solution in terms ofdistances to the solution set after t seconds, when the method starts at a point in a distance less or equal to R to the closest solution.

Proof. Without loss of generality, assume that the method starts at 0. Let $\begin{array} { r } { S _ { \tau } ~ = ~ \sum _ { i = 1 } ^ { n } \tau _ { i } } \end{array}$ and $a _ { i } = R \sqrt { \tau _ { i } / S _ { \tau } }$ . For all $i \in [ n ] , \varphi _ { i } \in [ - a _ { i } , a _ { i } ]$ is defined later, and

$$
f _ { i } ( x ) = \frac { n \mu } { 2 } \left. { x - \varphi , e _ { i } } \right. ^ { 2 } , \qquad \nabla f _ { i } ( x ; \xi _ { i } ) = n \mu \left. { x - \varphi , e _ { i } } \right. e _ { i } + \xi _ { i } e _ { i } ,
$$

where $\xi _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . The average function is

$$
f ( x ) = { \frac { \mu } { 2 } } \left\| x - \varphi \right\| ^ { 2 } ,
$$

and its unique minimizer is $x ^ { * } = \varphi$ . Moreover, $\| \varphi \| \leq R$

Each $f _ { i }$ is convex and smooth with parameter $n \mu \leq L _ { \mathrm { m a x } }$ , and

$$
\begin{array} { r } { \left\| \nabla f _ { i } ( x ) \right\| ^ { 2 } = 2 n \mu f _ { i } ( x ) \geq 2 \mu f _ { i } ( x ) . } \end{array}
$$

Hence Assumptions 1.3, 1.2, and 3.4 hold. The function $f$ is smooth with parameter $\mu \leq L _ { \mathrm { m a x } }$ and satisfies the global PŁ condition with constant $\mu .$ . Assumption 3.1 holds because $\varphi$ minimizes every $f _ { i }$ , while Assumption 3.3 does not hold in general because

$$
\operatorname { a r g m i n } f _ { i } = \{ x \in \mathbb { R } ^ { n } : x _ { i } = \varphi _ { i } \} \neq \{ \varphi \} = \operatorname { a r g m i n } f .
$$

Finally, the stochastic gradients are unbiased and have variance $\sigma ^ { 2 }$

The rest of the proof is the same as the proof of Theorem 2.3. Every stochastic gradient returned by worker i gives an observation from $\mathcal { N } ( \varphi _ { i } , \sigma ^ { 2 } / ( n ^ { 2 } \mu ^ { 2 } ) )$ . Let $N _ { i } ( t ) : = \lfloor t / \tau _ { i } \rfloor$ be an upper bound on the number of stochastic gradients calculated by worker i by time t. The same Gaussian mean-estimation lower bound gives

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { \varphi _ { i } \in [ - a _ { i } , a _ { i } ] } \mathbb { E } [ \| \bar { x } _ { t } - \varphi \| ^ { 2 } ] \geq c _ { 1 } \sum _ { i = 1 } ^ { n } \operatorname* { m i n } \{ a _ { i } ^ { 2 } , \frac { \sigma ^ { 2 } } { n ^ { 2 } \mu ^ { 2 } N _ { i } ( t ) } \} } } \\ & { } & { \ ~ i \in [ n ] } \\ & { } & { \geq c _ { 1 } \operatorname* { m i n } \{ R ^ { 2 } , \frac { \sigma ^ { 2 } S _ { \tau } } { n ^ { 2 } \mu ^ { 2 } t } \} \geq 2 \varepsilon } \end{array}
$$

for a universal constant $c _ { 1 } > 0$ , where $\bar { x } _ { t }$ any possible query point by time t for the objective with $\varphi .$ The second inequality follows from $N _ { i } ( t ) \leq \bar { t } / \tau _ { i }$ and $\dot { a } _ { i } ^ { 2 } = \dot { R } ^ { 2 } \tau _ { i } / \dot { S } _ { \tau }$ , which imply

$$
\operatorname* { m i n } \left\{ a _ { i } ^ { 2 } , \frac { \sigma ^ { 2 } } { n ^ { 2 } \mu ^ { 2 } N _ { i } ( t ) } \right\} \geq \frac { \tau _ { i } } { S _ { \tau } } \operatorname* { m i n } \left\{ R ^ { 2 } , \frac { \sigma ^ { 2 } S _ { \tau } } { n ^ { 2 } \mu ^ { 2 } t } \right\} ,
$$

where remains to sum over i and use $\scriptstyle \sum _ { i = 1 } ^ { n } \tau _ { i } / S _ { \tau } = 1$ . The third inequality follows from $\varepsilon < 0 . 0 1$ ， $R > 1 0$ , and the condition on t. Finally,

$$
\mathbb { E } \left[ \left. \bar { x } _ { t } - \varphi \right. ^ { 2 } \right] > \varepsilon
$$

for some $\varphi _ { i } \in [ - a _ { i } , a _ { i } ]$ , which the adversary can choose.

Theorem 3.2 (Lower Bound). Consider any randomized method under thefixed computation model and assume that $n \geq 2$ . Let us fix any $\varepsilon , \dot { L _ { \mathrm { m a x } } } , R , \mu , \sigma ^ { 2 } > 0$ such that $\mu < L _ { \operatorname* { m a x } } / ( 2 n ) , \varepsilon < 0 . 0 1$ and $R > 1 0$ . For any time budget $\scriptstyle t \leq c _ { 0 } \left( { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) { \frac { \sigma ^ { 2 } } { \varepsilon n \mu ^ { 2 } } }$ , where $c _ { 0 }$ is a universal constant, there exist functions $\{ f _ { i } \}$ and stochastic gradients $\{ \nabla f _ { i } ( \cdot ; \cdot ) \}$ such that $\{ f _ { i } \}$ satisfy Assumptions 1.2, 1.3, and $3 . l , f$ satisfies Assumptions 1.1 and 1.5 with $L = \dot { L } _ { \mathrm { m a x } } , \{ \nabla f _ { i } ( \dot { \cdot } ; \cdot ) \}$ satisfy Assumption 1.4 such that the method cannotfind ε–solution in terms ofdistances to the solution set after t seconds, when the method starts at a point in a distance less or equal to R to the closest solution.

Proof. The theorem is a simple corollary of Theorem 3.6 because Theorem 3.6 is stated under more strict assumptions on the class of the functions and stochastic gradients. The result of Theorem 3.6 holds even under additional Assumption 3.4. □

Theorem 3.7 (Lower Bound). Consider any randomized method under thefixed computation model and assume that $n \geq 2$ . Let us fix any $\varepsilon , L _ { \mathrm { m a x } } , R , \mu , \sigma ^ { 2 } > 0$ such that $\mu < L _ { \operatorname* { m a x } } / ( 2 n ) , \varepsilon < 0 . 0 1$ and $R > 1 0$ . For any time budget $\begin{array} { r } { t \le c _ { 0 } \left( \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { \sigma ^ { 2 } } { \varepsilon n \mu ^ { 2 } } } \end{array}$ , where $c _ { 0 }$ is a universal constant, there existfunctions $\{ f _ { i } \}$ and stochastic gradients $\{ \nabla f _ { i } ( \cdot ; \cdot ) \}$ such that $\{ f _ { i } \}$ satisfy Assumptions 1.2, 1.3, 3.1, and 3.3 (Assumption 3.4 is not imposed and may or may not hold with parameter µ), f satisfy Assumptions 1.1 and 1.5 with $L = L _ { \mathrm { m a x } } ^ { \setminus } , \{ \nabla f _ { i } ( \cdot ; \cdot ) \}$ satisfy Assumption 1.4, such that the method cannotfind ε–solution in terms ofdistances to the solution set after t seconds, when the method starts at a point in a distance less or equal to R to the closest solution.

Proof. Without loss of generalization, assume that the method starts at 0. Let $\begin{array} { r } { S _ { \tau } = \sum _ { i = 1 } ^ { n } \tau _ { i } } \end{array}$ , take

$$
a _ { i } = \frac { n \mu \tau _ { i } } { S _ { \tau } } , \qquad f _ { i } ( x ) = \frac { a _ { i } } 2 ( x - \varphi ) ^ { 2 } , \qquad \nabla f _ { i } ( x ; \xi _ { i } ) = a _ { i } ( x - \varphi ) + \xi _ { i } ,
$$

where $\xi _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ and $\varphi \in [ - R , R ]$ is defined later. Since $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } a _ { i } = \mu } \end{array}$

$$
f ( x ) = { \frac { \mu } { 2 } } { ( x - \varphi ) ^ { 2 } } .
$$

Thus, $f$ satisfies the global PŁ condition with constant $\mu ,$ and every $f _ { i }$ is convex and a<sub>i</sub>–smooth with $a _ { i } \leq n \mu \leq L _ { \operatorname* { m a x } }$ . Moreover, all the functions have the same unique minimizer $\varphi$ , so Assumptions 3.1 and 3.3 hold. Notice that $\left\| \nabla f _ { i } ( x ) \right\| ^ { 2 } = 2 a _ { i } ( f _ { i } ( x ) - f _ { i } ^ { * } )$ . Finally, the stochastic gradients are unbiased and have variance $\sigma ^ { 2 }$

Every stochastic gradient returned by worker i gives an observation from ${ \mathcal N } ( \varphi , \sigma ^ { 2 } / a _ { i } ^ { 2 } )$ . Let $N _ { i } ( t ) : =$ $\lfloor t / \tau _ { i } \rfloor$ be an upper bound on the number of stochastic gradients computed by worker i by time t. Let $P _ { \varphi } ^ { t }$ be the joint distribution of these observations by time t. For Gaussians with the same variance, the KL divergence $D _ { \mathrm { K L } } ( { \mathcal { N } } ( m _ { 1 } , v ) \| { \mathcal { N } } ( m _ { 0 } , v ) ) = ( m _ { 1 } - m _ { 0 } ) ^ { 2 } / ( 2 v )$ . By the chain rule for KL divergence,

$$
D _ { \mathrm { K L } } \bigl ( P _ { \varphi } ^ { t } \bigr \| P _ { - \varphi } ^ { t } \bigr ) \leq \sum _ { i = 1 } ^ { n } \frac { 2 \varphi ^ { 2 } a _ { i } ^ { 2 } } { \sigma ^ { 2 } } \times \frac { t } { \tau _ { i } } = \frac { 2 t \varphi ^ { 2 } n ^ { 2 } \mu ^ { 2 } } { \sigma ^ { 2 } S _ { \tau } }
$$

because $\begin{array} { r } { N _ { i } ( t ) \leq \frac { t } { \tau _ { i } } } \end{array}$ . Since $\begin{array} { r } { t \leq \frac { 1 } { 1 0 0 } \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { \sigma ^ { 2 } } { \varepsilon n \mu ^ { 2 } } } \end{array}$ , we get

$$
D _ { \mathrm { K L } } \bigl ( P _ { \varphi } ^ { t } \bigr | \bigr | P _ { - \varphi } ^ { t } \bigr ) \leq \frac { \varphi ^ { 2 } } { 3 2 \varepsilon } .
$$

Choosing any

$$
\varphi \in \{ - 4 \sqrt { \varepsilon } , 4 \sqrt { \varepsilon } \} ,
$$

we ensure that $- \varphi , \varphi \in [ - R , R ]$ (since $\varepsilon < 0 . 0 1$ and $R > 1 0 )$ and $D _ { \mathrm { K L } } \big ( P _ { \varphi } ^ { t } \big \| P _ { - \varphi } ^ { t } \big ) \leq \frac { 1 } { 2 }$ . Pinsker’s inequality gives $\left\| P _ { \varphi } ^ { t } - P _ { - \varphi } ^ { t } \right\| _ { \mathrm { T V } } \leq 1 / 2$ . Hence, the two-point Le Cam bound (Wainwright, 2019) gives

$$
\operatorname* { m a x } _ { \varphi \in \{ - 4 \sqrt { \varepsilon } , 4 \sqrt { \varepsilon } \} } \mathbb { E } \left[ \left| \bar { x } _ { t } - \varphi \right| ^ { 2 } \right] \geq 4 \varepsilon ,
$$

where $\bar { x } _ { t }$ is the random variable that the random algorithm can produce based on all available information up to time t for the objective with parameter $\varphi$ . Thus, the adversary can take $\varphi =$ arg $\begin{array} { r } { \operatorname* { m a x } _ { \varphi \in \{ - 4 \sqrt { \varepsilon } , 4 \sqrt { \varepsilon } \} } \mathbb { E } \left[ \left| \bar { x } _ { t } - \varphi \right| ^ { 2 } \right] } \end{array}$ to get $\mathbb { E } \left[ \left| \bar { x } _ { t } - \varphi \right| ^ { 2 } \right] > \varepsilon .$ □

## D AUXILIARY RESULTS

In this section, we present well-known results from optimization.

Lemma D.1 (Nesterov (2018)). Let $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ be afunction, which L–smooth and convex. Then for all $x , y \in \mathbb { R } ^ { d }$ we have:

$$
\begin{array} { r } { \left\| \nabla f ( x ) - \nabla f ( y ) \right\| ^ { 2 } \leq L \left. \nabla f ( x ) - \nabla f ( y ) , x - y \right. . } \end{array}\tag{13}
$$

Lemma D.2 (Karimi et al. (2016)). Let $f : \mathbb { R } ^ { d }  \mathbb { F }$ R be a convexfunction, which satisfies PŁ condition with a parameter $\mu$ (Assumption 1.5). Then,for all $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , we have

$$
\langle \nabla f ( x ) , x - { \bar { x } } _ { \ast } \rangle \geq { \frac { \mu } { 2 } } \left. { \bar { x } } _ { \ast } - x \right. ^ { 2 }\tag{14}
$$

where x¯<sub>∗</sub> is the projection of x onto the solution set $O f \operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } f ( x )$

## E LOWER BOUND IN THE HETEROGENEOUS CONVEX SETTING

This section complements the results from (Tyurin & Richtárik, 2023), where the authors only prove the optimal time complexities in the homogeneous nonconvex, heterogeneous nonconvex, and homogeneous convex settings. Here, we resolve the last piece, the heterogeneous convex setting.

Protocol 2 Time Multiple Oracles Protocol   
1: Input: function(s) $f \in { \mathcal { F } } ,$ , oracles and distributions $( ( O _ { 1 } , . . . , O _ { n } ) , ( \mathcal { D } _ { 1 } , . . . , \mathcal { D } _ { n } ) ) \in \mathcal { O } ( f )$   
algorithm $A \in { \mathcal { A } }$   
2: $s _ { i } ^ { 0 } = 0$ for all $i \in [ n ]$   
3: for $k = 0 , \ldots , \infty { \dot { \mathbf { d o } } }$   
4: $( t ^ { k + 1 } , i ^ { k + 1 } , x ^ { k } ) = A ^ { k } ( g ^ { 1 } , \dots , g ^ { k } ) .$ $\triangleright t ^ { k + 1 } \geq t ^ { k }$   
5: $( s _ { i ^ { k + 1 } } ^ { k + 1 } , g ^ { k + 1 } ) = O _ { i ^ { k + 1 } } ( \overline { { t } } ^ { k + 1 } , x ^ { \mathring { k } } , s _ { i ^ { k + 1 } } ^ { k } , \xi ^ { k + 1 } ) , \quad \xi ^ { k + 1 } \sim \mathcal { D } _ { i ^ { k + 1 } } \quad \triangleright s _ { j } ^ { k + 1 } = s _ { j } ^ { k } \quad \forall j \ne i ^ { k + 1 }$   
6: end for

Following Tyurin & Richtárik (2023), we have to formalize and introduce the following protocol and classes.

We investigate the optimization problem (1) when the function $f$ is convex. For the convex case, using Protocol 2, we use the complexity measure

$$
\operatorname* { m } _ { \operatorname* { m e } } \left( \mathcal { A } , \mathcal { F } \right) : = \operatorname* { i n f } _ { A \in \mathcal { A } } \operatorname* { i n f } \left\{ t \geq 0 \left| \operatorname* { s u p } _ { f \in \mathcal { F } \left( O , \mathcal { D } \right) \in \mathcal { O } \left( f \right) } \left( \mathbb { E } \left[ f ( x ^ { k \left( t \right) } ) \right] - \operatorname* { i n f } _ { x \in Q } f ( x ) \right) \leq \varepsilon \right\} , \right.\tag{15}
$$

where $x ^ { k }$ is generated by Protocol $2 , k ( t )$ is the largest index such that $t ^ { k ( t ) } \leq t ,$ and $Q$ is a convex set. Let us take any set $\dot { Q }$ , and consider the following class of convex functions.

Definition E.1 (Function Class $\mathcal { F } _ { Q , L } ^ { \mathrm { c o n v } } )$ .

We assume that a function $f : \mathbb { R } ^ { d }  \mathbb { R }$ is convex, differentiable, L-smooth on the set $Q .$ , i.e.,

$$
\| \nabla f ( x ) - \nabla f ( y ) \| \leq L \left\| x - y \right\| \quad \forall x , y \in Q .
$$

A set of all functions with such properties we define as $\mathcal { F } _ { Q , L } ^ { \mathrm { c o n v } } .$

Definition E.2 (Algorithm Class $\boldsymbol { A } _ { \mathrm { z r } } )$

An algorithm $A = \{ A ^ { k } \} _ { k = 0 } ^ { \infty }$ is a sequence such that

$$
A ^ { k } : \underset { k \mathrm { t i m e s } } { \underbrace { \mathbb { R } ^ { d } \times \dots \times \mathbb { R } ^ { d } } } \to \mathbb { R } _ { \ge 0 } \times \mathbb { R } ^ { d } \quad \forall k \ge 1 , A ^ { 0 } \in \mathbb { R } _ { \ge 0 } \times \mathbb { R } ^ { d } ,
$$

and, for all $k \geq 1$ and $g ^ { 1 } , \ldots , g ^ { k } \in \mathbb { R } ^ { d } , t ^ { k + 1 } \geq t ^ { k }$ , where $t ^ { k + 1 }$ and $t ^ { k }$ are defined as $( t ^ { k + 1 } , \cdot ) =$ $A ^ { k } ( g ^ { 1 } , \ldots , g ^ { k } )$ and $( t ^ { k } , \cdot ) = \bar { A } ^ { k - 1 } ( g ^ { 1 } , \cdot \cdot \cdot , g ^ { k - 1 } )$ . Moreover, $x ^ { k } \in Q$ for all $k \geq 0$

The following oracle helps to formalize the fixed computation model.

$$
\begin{array} { r l } & { \ O _ { \tau } ^ { \nabla f } : \underbrace { \mathbb { B } _ { \geq 0 } } _ { \mathrm { t i m e } } \times \underbrace { \mathbb { E } _ { \sum } ^ { d } } _ { \mathrm { p o i n t } } \times \underbrace { ( \mathbb { B } _ { \geq 0 } \times \mathbb { R } ^ { d } \times \{ 0 , 1 \} ) } _ { \mathrm { i n p u t s t a t e } } \times \mathbb { S } _ { \xi } \to \underbrace { ( \mathbb { R } _ { \geq 0 } \times \mathbb { R } ^ { d } \times \{ 0 , 1 \} ) } _ { \mathrm { o u p u t s t a t e } } \times \mathbb { R } ^ { d } } \\ & { \mathrm { s u c h ~ t h a t } \ O _ { \tau } ^ { \nabla f } ( t , x , ( s _ { t } , s _ { x } , s _ { q } ) , \xi ) = \left\{ \begin{array} { l l } { ( ( s , x , 1 ) , } & { 0 ) , \qquad s _ { q } = 0 , } \\ { ( ( s _ { t } , s _ { x } , 1 ) , } & { 0 ) , \qquad s _ { q } = 1 \mathrm { ~ a n d ~ } t < s _ { t } + \tau , } \\ { ( ( 0 , 0 , 0 ) , \qquad \nabla f ( s _ { x } ; \xi ) ) , } & { s _ { q } = 1 \mathrm { ~ a n d ~ } t \geq s _ { t } + \tau , } \end{array} \right. } \end{array}\tag{16}
$$

and $\nabla f ( \cdot ; \cdot )$ is a stochastic mapping.

## Definition E.3 (Oracle Class $\mathcal { O } _ { \tau _ { 1 } , . . . , \tau _ { n } } ^ { \mathrm { c o n v } , \sigma ^ { 2 } } ) .$

Let us consider an oracle class such that, for any $f \in \mathcal { F } _ { Q , L } ^ { \mathrm { c o n v } }$ , it returns oracles $O _ { i } = O _ { \tau _ { i } } ^ { \triangledown f _ { i } }$ and distributions $\mathcal { D } _ { i }$ for all $i \in [ n ]$ , where $\nabla f _ { i } ( \cdot ; \cdot )$ is an unbiased $\sigma ^ { 2 }$ -variance-bounded mapping on the set $Q$ of the gradient of the local function in worker i. The oracles $O _ { \tau _ { i } } ^ { \nabla f _ { i } }$ are defined in (16). We define such oracle class as $\mathcal { O } _ { \tau _ { 1 } , . . . , \tau _ { n } } ^ { \mathrm { c o n v } , \sigma ^ { 2 } }$ . Without loss of generality, we assume that $0 < \tau _ { 1 } \leq \cdot \cdot \cdot \leq \tau _ { n }$

Notice that this oracle class differs from the oracle class for convex functions in (Tyurin & Richtárik, 2023) because we consider the heterogeneous setting where the oracles return unbiased stochastic gradients of the local functions $f _ { i } ,$ which can be different. We refer the reader to (Tyurin & Richtárik, 2023) for additional details about the time complexities formalization. We are now ready to state the theorem.

Theorem E.4 (Informal theorem (see the formal Theorem E.5)). Let Assumptions 1.3, 1.1, and 1.4 hold. It is impossible to converge faster than

$$
\Theta \left( \tau _ { n } \sqrt { L } R / \sqrt { \varepsilon } + \left( 1 / n \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \sigma ^ { 2 } R ^ { 2 } / n \varepsilon ^ { 2 } \right)
$$

seconds under the fixed computation model.

Theorem E.5. Let us consider the oracle class $\mathcal { O } _ { \tau _ { 1 } , . . . , \tau _ { n } } ^ { \mathrm { c o n v } , \sigma ^ { 2 } }$ for some $\sigma ^ { 2 } > 0$ and $0 < \tau _ { 1 } \leq \cdot \cdot \cdot \leq \tau _ { n }$ We fix any $R , L , \varepsilon > 0$ such that $\sqrt { L } R > c _ { 1 } \sqrt { \varepsilon } > 0$ . For any

$$
t \leq c \times \left[ \tau _ { n } \frac { \sqrt { L } R } { \sqrt { \varepsilon } } + \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { \sigma ^ { 2 } R ^ { 2 } } { n \varepsilon ^ { 2 } } \right] ,
$$

in the view Protocol 2, for any algorithm $A \in { \mathcal { A } } _ { \mathrm { z r } } .$ , there exists a set $Q ,$ afunction $f \in \mathcal { F } _ { Q , L } ^ { \mathrm { c o n v } }$ and oracles and distributions $( ( O _ { 1 } , \dots , O _ { n } ) , ( \mathcal { D } _ { 1 } , \dots , \mathcal { D } _ { n } ) ) \in \mathcal { O } _ { \tau _ { 1 } , \dots , \tau _ { n } } ^ { \mathrm { c o n v } , \sigma ^ { 2 } } ( f )$ such that

$$
\mathbb { E } \left[ f ( x ^ { k ( t ) } ) \right] - \operatorname* { i n f } _ { x \in Q } f ( x ) > \varepsilon ,
$$

where $k ( t )$ is the largest index such that $t ^ { k ( t ) } \leq t .$ and R is the euclidean distance between 0 (starting point) and the closest solution $x _ { * } \in Q$ . The quantities $c _ { 1 }$ , and c are universal constants.

Proof. First term. It is easy to prove the dependence on the first term $\tau _ { n } \frac { \sqrt { L } R } { \sqrt { \varepsilon } }$ using the same idea as in (Lu & De Sa, 2021; Tyurin $\&$ Richtárik, 2023; Huang et al., 2022). It is sufficient to put a $\mathbf { \ddot { h a r d } } ^ { \prime }$ convex function (Nesterov, 2018; Woodworth et al., 2018) to the slowest worker corresponding with the time $\tau _ { n } = \operatorname* { m a x } _ { i \in [ n ] } \tau _ { i }$ . In particular, we can consider the “hard” quadratic function $\bar { f }$ from (Nesterov, 2018)[Section 2.1.2] and take the functions

$$
f _ { i } ( x ) = \left\{ \begin{array} { l l } { n \times \bar { f } ( x ) , } & { i = n } \\ { 0 , } & { i < n . } \end{array} \right.
$$

for all $x \in \mathbb { R } ^ { d }$ . The function $\begin{array} { r } { f = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } f _ { i } = \bar { f } } \end{array}$ belongs to the class $\mathcal { F } _ { Q , L } ^ { \mathrm { c o n v } }$ . We take the stochastic gradients without noise, i.e., $\nabla f _ { i } ( x ; \xi _ { i } ) = \nabla f _ { i } ( x )$ deterministically for all $x \in \mathbb { R } ^ { d } , \xi _ { i } \in \mathbb { S } _ { \xi _ { i } }$ , and $i \in [ n ]$ . It is clear that the only worker that can solve the problem is worker $n ,$ and it takes $\tau _ { n }$ seconds to find one gradient by the oracle construction. Thus, the required time complexity is Θ $\left( \tau _ { n } \frac { \sqrt { L } R } { \sqrt { \varepsilon } } \right)$ since the required oracle complexity is $\begin{array} { r } { \Theta \left( \frac { \sqrt { L } R } { \sqrt { \varepsilon } } \right) } \end{array}$ (Nesterov, 2018).

Second term. The proof of the second term is slightly trickier and uses the construction from (Woodworth et al., 2018). Let us fix any algorithm. We use the proof of Lemma 10 from (Woodworth et al., 2018) that has the following result. For any $\sigma ^ { 2 } , B > \bar { 0 }$ and any algorithm, it is possible to construct a one dimensional linear function $g : { \dot { \mathbb { R } } }  \mathbb { R }$ on the domain $\{ x \in \mathbb { R } : | { \dot { x } } | \leq B \}$ , a stochastic gradient mapping $\nabla g : \mathbb { R } \times \mathbb { S } _ { \xi }  \mathbf { \bar { \mathbb { R } } }$ , and a distribution $\mathcal { D }$ such that

$$
\mathbb { E } \left[ g ( x ^ { N } ) \right] - \operatorname* { m i n } _ { | x | \leq B } g ( x ) \geq { \frac { \sigma B } { 8 { \sqrt { N } } } }\tag{17}
$$

after $N$ queries of the oracle, where $\nabla g$ is unbiased and $\sigma ^ { 2 } .$ -variance-bounded.

The idea is to put a function $g _ { i }$ to each worker but with different domain sizes. In particular, for all $i \in [ n ]$ , we take the function ${ \bar { f } } _ { i } : \mathbb { R } ^ { n } \to \mathbb { R }$ such that

$$
f _ { i } ( x ) = g _ { i } ( x _ { i } ) ,\tag{18}
$$

where $g _ { i }$ is the function from Lemma 10 of (Woodworth et al., 2018) applied independently with $B = R _ { i }$ and $N = N _ { i } ( \bar { t } )$ , and $x _ { i }$ is the $i ^ { \mathrm { { t h } } }$ coordinate of a vector x. For all $i \in [ n ]$ , we consider the function $f _ { i }$ on the domain $\{ x _ { i } \in \mathbb { R } \mid \lvert x _ { i } \rvert \leq R _ { i } \}$ , where $\begin{array} { r } { R _ { i } : = R \times \frac { \sqrt { \tau _ { i } } } { \sqrt { \sum _ { i = 1 } ^ { n } \tau _ { i } } } } \end{array}$ . One can see that $f$

is convex, 0–smooth (because $g _ { i }$ is linear). The distance between 0 and the optimal point is less or equal to R because

$$
\sum _ { i = 1 } ^ { n } R _ { i } ^ { 2 } = \sum _ { i = 1 } ^ { n } R ^ { 2 } { \frac { \tau _ { i } } { \sum _ { i = 1 } ^ { n } \tau _ { i } } } = R ^ { 2 }
$$

and the optimal point for the problem $q _ { i } ( x _ { i } )  \mathrm { m i n } _ { | x _ { i } | \leq R _ { i } }$ is either $R _ { i } \ \mathrm { o r } - R _ { i }$ . We take

$$
Q = \{ x \in \mathbb { R } ^ { n } : | x _ { i } | \leq R _ { i } \quad \forall i \in [ n ] \} .
$$

Let us define the time

$$
\bar { t } : = \frac { \sigma ^ { 2 } R ^ { 2 } } { 2 5 6 n \varepsilon ^ { 2 } } \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) .\tag{19}
$$

By the time $\bar { t } ,$ worker i can calculate at most

$$
N _ { i } ( \bar { t } ) : = \left\lfloor \frac { \bar { t } } { \tau _ { i } } \right\rfloor\tag{20}
$$

stochastic gradients. Therefore,

$$
\begin{array} { r l } & { \mathbb { E } \left[ f ( \bar { x } ) \right] - \underset { x \in Q } { \operatorname* { m i n } } f ( x ) = \cfrac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \left( \mathbb { E } \left[ g _ { i } ( \bar { x } _ { i } ) \right] - \underset { | x _ { i } | \leq R _ { i } } { \operatorname* { m i n } } g _ { i } ( x _ { i } ) \right) \overset { ( 1 7 ) } { \geq } \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \frac { \sigma R _ { i } } { 8 \sqrt { N _ { i } ( \bar { t } ) } } } \\ & { \quad \quad \quad \quad = \cfrac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \cfrac { \sigma R \sqrt { \tau _ { i } } } { 8 \sqrt { N _ { i } ( \bar { t } ) } \sqrt { \sum _ { i = 1 } ^ { n } \tau _ { i } } } \quad \overset { ( 1 9 ) , ( 2 0 ) } { \geq } \sum _ { i = 1 } ^ { n } \frac { 2 \varepsilon \tau _ { i } } { \sum _ { i = 1 } ^ { n } \tau _ { i } } = 2 \varepsilon . } \end{array}
$$

where x¯ is any possible output of the algorithm before the time t.<sup>¯</sup>

## F PROOF OF THEOREMS F.1 AND F.2

Theorem F.1. Let Assumptions 1.1, 1.3, 1.4, and 1.5 hold, and thefunctions $\{ f _ { i } \}$ are equal. Let us take $\gamma = 1 / L$ and $S = 4 \hat { \sigma } ^ { 2 } / \mu L \varepsilon$ , then Rennala SGD (Algorithm 1 with $\begin{array} { r } { w _ { i } ^ { k } = { \tilde { n } } \Big / { \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } } \Big ) ⨏ { \hbar n d s } ~ x ^ { k + 1 } } \end{array}$ such that E $\left[ \left\| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right\| ^ { 2 } \right] \leq \varepsilon$ after

$$
\mathcal { O } \left( \operatorname* { m i n } _ { m \in \left[ n \right] } \left[ \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { 1 } { \tau _ { i } } \right) ^ { - 1 } \left( \frac { L } { \mu } + \frac { \sigma ^ { 2 } } { m \varepsilon \mu ^ { 2 } } \right) \right] \log \frac { R ^ { 2 } } { \varepsilon } \right)\tag{21}
$$

seconds, where $x _ { * } ^ { k + 1 }$ is the closest solution to $x ^ { k + 1 }$

Proof. Since the functions are equal, Rennala SGD is equivalent to

$$
\begin{array} { r l } & { x ^ { k + 1 } = x ^ { k } - \gamma g _ { \mathsf { R } } ^ { k } , } \\ & { g _ { \mathsf { R } } ^ { k } : = \frac { 1 } { \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } } \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \nabla f ( x ^ { k } ; \xi _ { i j } ^ { k } ) . } \end{array}
$$

Clearly, $g _ { \mathsf { R } } ^ { k }$ is unbiased and

$$
\mathbb { E } _ { k } \left[ \left\| g _ { \mathsf { R } } ^ { k } - \nabla f ( x ^ { k } ) \right\| ^ { 2 } \right] = \left( \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } \right) ^ { - 2 } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \mathbb { E } _ { k } \left[ \left\| \nabla f ( x ^ { k } ; \xi _ { i j } ^ { k } ) - \nabla f ( x ^ { k } ) \right\| \right] ^ { 2 } \leq \sigma ^ { 2 } \left( \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } \right) ^ { - 1 } .
$$

Rennala SGD waits for the moment when $\textstyle \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } > S$ (see Alg. 1 with $\begin{array} { r } { w _ { i } ^ { k } = { { n } } / { \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } } ) } \end{array}$ . Thus

$$
\mathbb { E } _ { k } \left[ \left\| g _ { \mathsf { R } } ^ { k } - \nabla f ( x ^ { k } ) \right\| ^ { 2 } \right] \leq \frac { \sigma ^ { 2 } } { S } \leq \frac { \mu L \varepsilon } { 4 }
$$

We can use Theorem F.3 to get

$$
\mathbb { E } \left[ \left. x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right. ^ { 2 } \right] \leq \left( 1 - \frac { \gamma \mu } { 2 } \right) ^ { k + 1 } \left. x ^ { 0 } - x _ { * } ^ { k } \right. ^ { 2 } + \frac { \gamma L \varepsilon } { 2 } .
$$

Since $\begin{array} { r } { \gamma = \frac { 1 } { L } } \end{array}$ , we obtain

$$
\mathbb { E } \left[ \left. x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right. ^ { 2 } \right] \leq \left( 1 - \frac { \mu } { 2 L } \right) ^ { k + 1 } \left. x ^ { 0 } - x _ { * } ^ { k } \right. ^ { 2 } + \frac { \varepsilon } { 2 } .
$$

The last inequality ensure that the method finds an ε–solution after

$$
\mathcal { O } \left( \frac { L } { \mu } \log \frac { R ^ { 2 } } { \varepsilon } \right)
$$

iterations. In each iteration, the method has to ensure that $\textstyle \sum _ { i = 1 } ^ { n } B _ { i } ^ { k } > S$ . A sufficient time for that is

$$
2 \operatorname* { m i n } _ { m \in [ n ] } \left[ \left( { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } { \frac { 1 } { \tau _ { i } } } \right) ^ { - 1 } ( 1 + S ) \right] .
$$

under the fixed computation model (see Theorem 11 in (Tyurin et al., 2024)). It is left to multiply this time by $\begin{array} { r } { \mathcal { O } \left( \frac { L } { \mu } \log \frac { R ^ { 2 } } { \varepsilon } \right) } \end{array}$ □

Theorem F.2. Let Assumptions 1.1, 1.3, 1.4, and 1.5 hold. Let us take $\gamma = { ^ { 1 } \mathord { \left/ { \vphantom { ^ { 1 } \sum } } \right. \kern - delimiterspace } L }$ and $S = { 4 \sigma ^ { 2 } } / { \mu L \varepsilon }$ 2 then Malenia SGD (Algorithm 1 with $w _ { i } ^ { k } = 1 / B _ { i } ^ { k } )$ finds $x ^ { k + 1 }$ such that E $\left[ \left\| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right\| ^ { 2 } \right] \leq \varepsilon$ after

$$
\mathcal { O } \left( \left[ \tau _ { n } \frac { L } { \mu } + \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { \sigma ^ { 2 } } { n \varepsilon \mu ^ { 2 } } \right] \log \frac { R ^ { 2 } } { \varepsilon } \right)\tag{22}
$$

seconds, where $x _ { * } ^ { k + 1 }$ is the closest solution to $x ^ { k + 1 }$

Proof. The proof of this theorem almost repeats the proof of Theorem F.1. The variance of Malenia SGD is

$$
\mathbb { E } _ { k } \left[ \left\| g _ { \mathsf { M } } ^ { k } - \nabla f ( x ^ { k } ) \right\| ^ { 2 } \right] = \mathbb { E } _ { k } \left[ \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { 1 } { B _ { i } ^ { k } } \sum _ { j = 1 } ^ { B _ { i } ^ { k } } \nabla f _ { i } ( x ^ { k } ; \xi _ { i j } ^ { k } ) - \nabla f ( x ^ { k } ) \right\| ^ { 2 } \right] \leq \frac { \sigma ^ { 2 } } { n } \left( \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \frac { 1 } { B _ { i } ^ { k } } \right) .
$$

The method waits for the moment when $\textstyle \left( { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } 1 / B _ { i } ^ { k } \right) ^ { - 1 } > { \frac { S } { n } }$ . Therefore

$$
\mathbb { E } _ { k } \left[ \left\| g _ { \mathsf { M } } ^ { k } - \nabla f ( x ^ { k } ) \right\| ^ { 2 } \right] \leq \frac { \sigma ^ { 2 } } { S } .
$$

Using the same reasoning, the method finds an ε–solution after

$$
\mathcal { O } \left( \frac { L } { \mu } \log \frac { R ^ { 2 } } { \varepsilon } \right)
$$

iterations. In each iteration, the method has to ensure that $\begin{array} { r } { \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { 1 } / { B _ { i } ^ { k } } \right) ^ { - 1 } > \frac { S } { n } . \mathrm { A } } \end{array}$ sufficient time for that is

$$
\bar { t } = 2 \left( \tau _ { n } + \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tau _ { i } \right) \frac { S } { n } \right)
$$

under the fixed computation model because the number of computed stochastic gradients $\begin{array} { r } { B _ { i } ^ { k } \geq \left\lfloor \frac { \bar { t } } { \tau _ { i } } \right\rfloor } \end{array}$ and

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { 1 } { B _ { i } ^ { k } } \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { 1 } { \left\lfloor \frac { \bar { t } } { \tau _ { i } } \right\rfloor } \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { 2 \tau _ { i } } { \bar { t } } < \frac { n } { S } ,
$$

where we use $\lfloor x \rfloor \geq { \frac { x } { 2 } }$ for all $x \geq 1$ . Multiplying t<sup>¯</sup>by $\begin{array} { r } { \mathcal { O } \left( \frac { L } { \mu } \log \frac { R ^ { 2 } } { \varepsilon } \right) } \end{array}$ , we get the result. □

Theorem F.3. Consider the method

$$
\boldsymbol { x } ^ { k + 1 } = \boldsymbol { x } ^ { k } - \gamma \nabla f ( \boldsymbol { x } ^ { k } ; \xi ^ { k } ) ,\tag{23}
$$

where $\mathbb { E } _ { k } \left[ \nabla f ( x ^ { k } ; \xi ^ { k } ) \right] = \nabla f ( x ^ { k } ) , \mathbb { E } _ { k } \left[ \left\| \nabla f ( x ^ { k } ; \xi ^ { k } ) - \nabla f ( x ^ { k } ) \right\| ^ { 2 } \right] \leq \sigma ^ { 2 }$ , and $\sigma ^ { 2 } > 0$ . Let Assumptions $I . 3 , I . 5 ,$ , and $I . I$ hold. Let us take $\gamma = 1 / L$ and $S = { 2 \sigma ^ { 2 } } / { \mu L \varepsilon }$ , then the methodfinds $x ^ { k + 1 }$ such that

$$
\mathbb { E } \left[ \left. x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right. ^ { 2 } \right] \leq \left( 1 - \frac { \gamma \mu } { 2 } \right) ^ { k + 1 } \left. x ^ { 0 } - x _ { * } ^ { k } \right. ^ { 2 } + \frac { 2 \gamma \sigma ^ { 2 } } { \mu } ,
$$

where $x _ { * } ^ { k + 1 }$ is the closest solution o $f \operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } f ( x ) \ t o \ x ^ { k + 1 }$

Proof. Using the properties of the projection and (23), we have

$$
\begin{array} { r l } & { \mathbb { E } _ { k } \left[ \left\| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right\| ^ { 2 } \right] \leq \mathbb { E } _ { k } \left[ \left\| x ^ { k + 1 } - x _ { * } ^ { k } \right\| ^ { 2 } \right] } \\ & { \qquad = \mathbb { E } _ { k } \left[ \left\| x ^ { k } - \gamma \nabla f ( x ^ { k } ; \xi ^ { k } ) - x _ { * } ^ { k } \right\| ^ { 2 } \right] } \\ & { \qquad = \mathbb { E } _ { k } \left[ \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } \right] - 2 \gamma \mathbb { E } _ { k } \left[ \left. \nabla f ( x ^ { k } ; \xi ^ { k } ) , x ^ { k } - x _ { * } ^ { k } \right. \right] + \gamma ^ { 2 } \mathbb { E } _ { k } \left[ \left\| \nabla f ( x ^ { k } ; \xi ^ { k } ) \right\| ^ { 2 } \right] } \\ & { \qquad = \mathbb { E } _ { k } \left[ \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } \right] - 2 \gamma \left. \nabla f ( x ^ { k } ) , x ^ { k } - x _ { * } ^ { k } \right. + \gamma ^ { 2 } \mathbb { E } _ { k } \left[ \left\| \nabla f ( x ^ { k } ; \xi ^ { k } ) \right\| ^ { 2 } \right] . } \end{array}
$$

In the last equality, we use the unbiasedness. Due the variance decomposition equality, we get

$$
\mathbb { E } _ { k } \left[ \left. x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right. ^ { 2 } \right] \leq \mathbb { E } _ { k } \left[ \left. x ^ { k } - x _ { * } ^ { k } \right. ^ { 2 } \right] - 2 \gamma \left. \nabla f ( x ^ { k } ) , x ^ { k } - x _ { * } ^ { k } \right. + \gamma ^ { 2 } \left. \nabla f ( x ^ { k } ) \right. ^ { 2 } + \gamma ^ { 2 } \mathbb { E } _ { k } \left[ \left. \nabla f ( x ^ { k } ; \xi ^ { k } ) - \nabla f ( x ^ { k } ) \right. ^ { 2 } \right]
$$

Since the function f is L–smooth and $\nabla f ( x _ { * } ^ { k } ) = 0 .$ , we obtain

$$
\begin{array} { r l } & { - 2 \gamma \left. \nabla f ( { x } ^ { k } ) , { x } ^ { k } - { x } _ { * } ^ { k } \right. + \gamma ^ { 2 } \left\| \nabla f ( { x } ^ { k } ) \right\| ^ { 2 } } \\ & { = - 2 \gamma \left. \nabla f ( { x } ^ { k } ) - \nabla f ( { x } _ { * } ) , { x } ^ { k } - { x } _ { * } ^ { k } \right. + \gamma ^ { 2 } \left\| \nabla f ( { x } ^ { k } ) - \nabla f ( { x } _ { * } ) \right\| ^ { 2 } } \\ & { \leq - 2 \gamma \left. \nabla f ( { x } ^ { k } ) - \nabla f ( { x } _ { * } ) , { x } ^ { k } - { x } _ { * } ^ { k } \right. + L \gamma ^ { 2 } \left. \nabla f ( { x } ^ { k } ) - \nabla f ( { x } _ { * } ) , { x } ^ { k } - { x } _ { * } ^ { k } \right. } \\ & { = \gamma \left( L \gamma - 2 \right) \left. \nabla f ( { x } ^ { k } ) - \nabla f ( { x } _ { * } ) , { x } ^ { k } - { x } _ { * } ^ { k } \right. . } \end{array}
$$

Taking $\begin{array} { r } { \gamma \le \frac { 1 } { L } } \end{array}$ and substituting the inequality to (24), we get

$$
\begin{array} { r } { \mathbb { E } _ { k } \left[ \left\| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right\| ^ { 2 } \right] \leq \mathbb { E } _ { k } \left[ \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } \right] - \gamma \left. \nabla f ( x ^ { k } ) , x ^ { k } - x _ { * } ^ { k } \right. + \gamma ^ { 2 } \mathbb { E } _ { k } \left[ \left\| \nabla f ( x ^ { k } ; \xi ^ { k } ) - \nabla f ( x ^ { k } ) \right\| ^ { 2 } \right] . } \end{array}
$$

The $\sigma ^ { 2 } .$ –variance bounded ensures that

$$
\begin{array} { r } { \mathbb { E } _ { k } \left[ \left\| x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right\| ^ { 2 } \right] \leq \mathbb { E } _ { k } \left[ \left\| x ^ { k } - x _ { * } ^ { k } \right\| ^ { 2 } \right] - \gamma \left. \nabla f ( x ^ { k } ) , x ^ { k } - x _ { * } ^ { k } \right. + \gamma ^ { 2 } \sigma ^ { 2 } . } \end{array}
$$

Due to convexity and Assumption 1.5, we can use Lemma D.2, which yields

$$
\begin{array} { r l } & { \mathbb { E } _ { k } \left[ \left. x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right. ^ { 2 } \right] \leq \mathbb { E } _ { k } \left[ \left. x ^ { k } - x _ { * } ^ { k } \right. ^ { 2 } \right] - \frac { \gamma \mu } { 2 } \left. x ^ { k } - x _ { * } ^ { k } \right. + \gamma ^ { 2 } \sigma ^ { 2 } } \\ & { \qquad = \left( 1 - \frac { \gamma \mu } { 2 } \right) \mathbb { E } _ { k } \left[ \left. x ^ { k } - x _ { * } ^ { k } \right. ^ { 2 } \right] + \gamma ^ { 2 } \sigma ^ { 2 } . } \end{array}
$$

Unrolling the recursion and taking the full expectation, we obtain

$$
\mathbb { E } \left[ \left. x ^ { k + 1 } - x _ { * } ^ { k + 1 } \right. ^ { 2 } \right] \leq \left( 1 - \frac { \gamma \mu } { 2 } \right) ^ { k + 1 } \left. x ^ { 0 } - x _ { * } ^ { 0 } \right. ^ { 2 } + \frac { 2 \gamma \sigma ^ { 2 } } { \mu }
$$

## G ASSUMPTIONS 1.3, 3.3 AND 3.4 IMPLY ASSUMPTION 1.5

Theorem G.1. Let {f } satisfy Assumption 1.3, 3.3, and Assumption 3.4 with constant $\mu ,$ then $f$ satisfies Assumption 1.5 with constant $\frac { \bar { \mu } } { 4 }$

Proof. We fix $x \in \mathbb { R } ^ { d }$ . Since Assumption 3.3 hold, then the functions share the closest solution $x _ { * }$ to x. Assumption 3.4 ensures that

$$
f _ { i } ( x ) - f _ { i } ( x _ { * } ) \geq { \frac { \mu } { 2 } } \left. x - x _ { * } \right. ^ { 2 } .
$$

for all $i \in [ n ]$ (Karimi et al., 2016). Thus

$$
f ( x ) - f ( x _ { * } ) \geq { \frac { \mu } { 2 } } \left. x - x _ { * } \right. ^ { 2 } .
$$

Due to convexity, we get

$$
f ( x _ { * } ) \geq f ( x ) + \langle \nabla f ( x ) , x _ { * } - x \rangle .
$$

Therefore

$$
f ( x ) - f ( x _ { * } ) \leq \langle \nabla f ( x ) , x - x _ { * } \rangle \leq \| \nabla f ( x ) \| \| x - x _ { * } \| \leq \| \nabla f ( x ) \| { \sqrt { \frac { 2 } { \mu } } } { \sqrt { f ( x ) - f ( x _ { * } ) } }
$$

and

$$
\frac { \mu } { 4 } \left( f ( x ) - f ( x _ { * } ) \right) \leq \frac { 1 } { 2 } \left. \nabla f ( x ) \right. ^ { 2 } ,
$$

which is Assumption 1.5 with constant $\frac { \mu } { 4 }$

## H EXPERIMENTS

We conduct a comparison between Rennala SGD and Malenia SGD on both stochastic quadratic optimization tasks and real-world machine learning problems. These are standard quadratic optimization and computer vision problems, the design of which we explain in Section I. We developed a library that simulates the behavior of $n = 1 0 0$ workers. Both methods have two hyperparameters: step size γ and parameter S. We do a grid search for both methods and find the best pairs in all setups. We start with synthetic quadratic optimization problems, which are generated without and with the interpolation regime. The procedure is described in Section I.1.

## H.1 WITHOUT INTERPOLATION

![](images/f14abf9cb82248d6666f29c05417c4a2d2d2a8e6d1b2b5dbc6511591dbf16105.jpg)  
Figure 1: Comparison of the methods on a quadratic optimization problem without interpolation. We take the computation time $\tau _ { i } = i ^ { 2 }$ for all $i \in [ n ]$

In Figure 1, we present results without interpolation. The plots concur with the theory from Section 3, where we explain that it is essential to have interpolation to break the time complexity of Malenia SGD. Rennala SGD has biased gradient estimators and does not converge to a minimum of the quadratic optimization problem in Figure 1.

## H.2 WITH INTERPOLATION

![](images/d3649e0d28f39e1a90ad51feb25e6a977ce0b444dc531d5684389700f528be26.jpg)

![](images/2eedfbd51f8efcd809b9ba6c37f73ec021a2190ac2e8d05ebffa860518158bbc.jpg)  
Figure 2: Comparison of the methods on quadratic optimization problems with interpolation. Times $\{ \tau _ { i } \}$ less diverse: Left plot: $\tau _ { i } = \sqrt { i }$ for all $i \in [ n ]$ . Right plot: $\tau _ { 1 } = 0 . 0 1 , \tau _ { 2 } = 1 , \dots , \tau _ { n } = 1$

In Figures 2 and 3, we consider the methods in the interpolation regime. As expected, according to Section 3.2, Rennala SGD outperforms Malenia SGD in all experiments. We compare the methods with different $\{ \tau _ { i } \}$ . In Figures 2, the times $\{ \tau _ { i } \}$ are less diverse, so the difference between the methods is less profound. In Figures $3 , \{ \tau _ { i } \}$ are more different; thus, we can see that Rennala SGD converges much faster to low function values because it has much less variance in the corresponding gradient estimator.

![](images/7352f736a086e14c74dd46d4fa8a57db0410df665905ce94b9bdedf6dba35ddf.jpg)

![](images/f2df0902bbd771969cd6aa2e6347384bf2bb421691fc6d1e757c4287b1df5675.jpg)  
Figure 3: Comparison of the methods on quadratic optimization problems with interpolation. Times $\{ \tau _ { i } \}$ more diverse: Left plot: $\tau _ { i } = i ^ { 2 }$ for all $i \in [ n ]$ . Right plot: $\tau _ { 1 } = 0 . 0 0 1$ $\tau _ { 2 } = 1 , \ldots , \tau _ { n } = 1$

## H.3 RESNET-18 AND CIFAR-10

We also verify how Rennala SGD and Malenia SGD work with ResNet-18 and the CIFAR-10 classification problem (Krizhevsky et al., 2009) (License: MIT). Both algorithms take step size $\gamma = 0 . 2 5 ,$ sample a batch of size 128, and the smallest S such that all workers calculate at least one batch. The dataset CIFAR-10 is split between the workers, so we consider the heterogeneous setting; all workers access different samples. The results of the experiments are presented in Figure 4. One can see that Rennala SGD converges faster in terms of accuracy, which might be explained by the fact that neural networks work in the interpolation regime. Note that this is an empirical observation in the nonconvex setup, and explaining it from the theoretical point of view is an important future work.

![](images/5ccbe2d6f3ea8f41d3434e2a3b9cd87777409ec0b8074876fe0a89e313cafecd.jpg)  
Figure 4: Comparison of the methods on the CIFAR-10 classification problem with ResNet-18. We take the computation time $\tau _ { i } = i ^ { 2 }$

## I EXPERIMENTS DETAILS

The experiments were run in Python 3 using an Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz.

## I.1 QUADRATIC OPTIMIZATION TASK GENERATION PROCEDURE

In Section H, we perform experiments using synthetic quadratic optimization problems

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \frac { 1 } { 2 } x ^ { \top } \mathbf { A } _ { i } x - x ^ { \top } b _ { i } \right) .
$$

Below, we present the algorithm, based on (Szlendak et al., 2021), that generates these problems. In all experiments, we take $s = 3$ to ensure that the generated matrices are diverse. We take $n = 1 0 0$ $d = 1 0 0$ , and $\lambda = 0 . 0 0 1$ . The stochastic gradients are equal to the true gradients plus standard Gaussian noise added to the coordinates to emulate stochasticity.

With these parameters and procedures, we run the experiments from Section H.1. To conduct the experiments from Section H.2 in the interpolation regime, we take the matrices $\mathbf { A } _ { 1 } , \cdots , \mathbf { A } _ { n } .$ , vectors $b _ { 1 } , \cdots , b _ { n }$ returned by Algorithm 3. Let $\textstyle { \bar { x } } ,$ be the solution of the quadratic optimization problem $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bar { \mathbf { A } } _ { i } \bar { x } _ { * } = \frac { 1 } { n } \bar { \sum _ { i = 1 } ^ { n } \bar { b } _ { i } } } \end{array}$ . Then, we redefine the vectors {b<sub>i</sub>} as $b _ { i } = \mathbf { A } _ { i } { \bar { x } } _ { : }$ to ensure that we are working in the interpolation regime. With this strategy, the matrices are still different, and the functions $\{ { \bar { f } } _ { i } \}$ are not equal.

Algorithm 3 Generate quadratic optimization tasks   
1: Parameters: number nodes n, dimension $d ,$ regularizer λ, and noise scale s.   
2: for $i = 1 , \ldots , n$ do   
3: Generate random noises $\eta _ { i } ^ { s } = 1 + s \zeta _ { i } ^ { s }$ and $\eta _ { i } ^ { b } = s \zeta _ { i } ^ { b } .$ , i.i.d. $\zeta _ { i } ^ { s } , \zeta _ { i } ^ { b } \sim \mathcal { N } ( 0 , 1 )$   
4: Take vector $b _ { i } = \frac { \eta _ { i } ^ { s } } { 4 } ( - 1 + \eta _ { i } ^ { b } , 0 , \cdot \cdot \cdot , 0 ) \in \mathbb { R } ^ { d }$   
5: Take the initial tridiagonal matrix   
$\mathbf { A } _ { i } = \frac { \eta _ { i } ^ { s } } { 4 } \left( \begin{array} { l l l l } { 2 } & { - 1 } & & { 0 } \\ { - 1 } & { \ddots } & { \ddots } \\ & { \ddots } & { \ddots } & \\ { 0 } & & { - 1 } & { 2 } \end{array} \right) \in \mathbb { R } ^ { d \times d }$   
6: end for   
7: Take the mean of matrices $\begin{array} { r } { \mathbf { A } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { A } _ { i } } \end{array}$   
8: Find the minimum eigenvalue $\lambda _ { \mathrm { m i n } } ^ { \mathrm { ~ ~ } } ( \mathbf { A } )$   
9: for $i = 1 , \ldots , n$ do   
10: Update matrix $\mathbf { A } _ { i } = \mathbf { A } _ { i } + ( \lambda - \lambda _ { \operatorname* { m i n } } ( \mathbf { A } ) ) \mathbf { I }$   
11: end for   
12: Take starting point $x ^ { 0 } = ( { \sqrt { d } } , 0 , \cdots , 0 )$   
13: Output: matrices $\mathbf { A } _ { 1 } , \cdots , \mathbf { A } _ { n } ,$ vectors $b _ { 1 } , \cdots , b _ { n } ,$ , starting point $x ^ { 0 }$

## I.2 EXPERIMENTS WITH RESNET AND CIFAR-10

In Section H.3, we consider the standard computer vision classification problem with ResNet-18 (He et al., 2016) and CIFAR-10 (Krizhevsky et al., 2009). We conduct the experiments using PyTorch and implement both Rennala SGD and Malenia SGD optimizers. For reproducibility, we use the default ResNet-18 architecture provided in PyTorch and split randomly and evenly the CIFAR-10 dataset across multiple workers to create a heterogeneous data distribution scenario. We use standard preprocessing techniques for CIFAR-10, including normalization and random cropping, and train the network for a fixed number of epochs. The performance metrics include top-1 accuracy. In total, we solve the optimization problem

$$
\begin{array} { r } { \underset { x \in \mathbb { R } ^ { d } } { \operatorname* { m i n } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathrm { l o s s } ( \mathrm { R e s N e t } ( a _ { i j } ; x ) , y _ { i j } ) \right) , } \end{array}
$$

where $" 1 0 5 5 \ '$ is the standard cross-entropy loss, $\{ a _ { i j } , y _ { i j } \}$ are samples from CIFAR-10 splitted between the workers.