# Non-Interfering Asymmetric AdamW Updates for Constrained Exploration of Equifinality in Hybrid Models

Ekaterina Gikalo<sup>1,2</sup> egikalo@bgc-jena.mpg.de

Christian Reimers<sup>1,3</sup> creimers@bgc-jena.mpg.de

Zavud Baghirov<sup>1,2</sup> zbaghirov@bgc-jena.mpg.de

Martin Jung<sup>1</sup> mjung@bgc-jena.mpg.de

Markus Reichstein<sup>1,3</sup> mreichstein@bgc-jena.mpg.de

Marco Körner<sup>4,2,3</sup> marco.koerner@dlr.de

<sup>1</sup>Max Planck Institute for Biogeochemistry, Jena, Germany <sup>2</sup>TUM School of Engineering and Design, Technical University of Munich (TUM), Munich, Germany <sup>3</sup>ELLIS Unit Jena, Jena, Germany

<sup>4</sup>German Remote Sensing Data Center (DFD), German Aerospace Center (DLR), Weßling, Germany

## Abstract

Hybrid models let neural networks supply the unresolved parameters of a processbased simulator, and are fitted to observations that constrain aggregate fluxes and storages but not the internal partitioning that produces them. Such a fit is equifinal: configurations indistinguishable in the observed aggregates can disagree substantially about the unobserved internals, and therefore about the mechanism they attribute to the same data. We study evapotranspiration, whose total flux is observed while its split into transpiration and evaporation is not. A model ensemble built from random seeds alone spans only a narrow band of the partitionings the data admit. We produce the spread deliberately, training members under auxiliary repulsion on transpiration, subject to the fit to observations not degrading. Noninterference is usually arranged by making the auxiliary gradient orthogonal to the main one, but under AdamW the applied step is a preconditioned momentum direction, so orthogonality between raw gradients no longer implies it. We introduce Non-interfering Asymmetric Network Optimisation (NANO), an update whose firstorder effect on the main loss is exactly that of the main objective alone, whose auxiliary step is bounded, and which concentrates repulsion in the coordinates the main gradient leaves idle.

## 1 Problem setting

Hybrid models embed neural networks in a process-based simulator and train them end-to-end through differentiable physics, so that known structure is retained and the unresolved parameters are fitted to data [1]. A model is equifinal when distinct configurations of it reproduce the available observations equally well, so that the data select no single mechanism among them [2]. In a hybrid model, equifinality can arise from several sources at once: different neural network outputs can produce the same simulated targets; the process model may be too simplified to describe the full system; its components may be interchangeable, a deficit in one being compensated by an excess in another; and the observations could constrain only a few aggregate dimensions of a much higher-dimensional internal state.

![](images/358c4b4b4d2bdb5cf05b0352c574536cf2b8b7551078d90691627b019e3c10ef.jpg)  
Figure 1: Equifinality in the partitioning of evapotranspiration. Left: ET is the sum of transpiration, soil evaporation and interception evaporation. Right: network is trained against ET alone, so different partitionings can reproduce the same observed target.

Our study focuses on evapotranspiration, $\mathrm { E T } = T + E _ { s } + E _ { i } ,$ , whereby water is transported from the Earth’s surface into the atmosphere through evaporation— $- i . e . ,$ , soil evaporation $\dot { E } _ { s }$ from the soil surface and interception evaporation $E _ { i }$ from canopies before it reaches the ground—, as well as through stomatal transpiration T from vegetation stocks (cf. Figure 1). While the transpiration fraction $\mathrm { \bar { \cal R } } = T / \mathrm { E T }$ is key to understanding the water-carbon cycle, the coupling between terrestrial water fluxes and carbon uptake, and the response to drought and irrigation, only the total ET is constrained through observations. Published global estimates of R vary widely, between 0.80–0.90 [3] and $0 . 6 4 \pm 0 . 1 3$ [4]. Making this equifinality explicit therefore allows a model to report honestly what the observations do and do not determine.

The Hybrid Hydrological Carbon Cycle Model (H2CM) [5] is a hybrid model of the coupled water and carbon cycles: recurrent and feed-forward networks predict its process parameters, the physics propagates them into states and fluxes, and these are compared against spatio-temporal observations. While ET is among those targets, listed in Appendix A, its partitioning is not.

## 2 Methodology

Training a standard random-seed ensemble underestimates the equifinality in H2CM, as no term in the main objective distinguishes members that agree on ET but disagree on its partitioning. Instead, we produce the spread deliberately: M members are trained under an auxiliary repulsion loss $\mathcal { L } _ { \mathrm { a u x } }$ The difficulty lies in adding the repulsion without increasing the main loss ${ \mathcal { L } } _ { \mathrm { m a i n } } .$

Multi-task methods rebalance or reshape task gradients [6–11], while asymmetric variants protect a designated main task by gating or decomposing the auxiliary gradient [12, 13]. Closest to our setting are Projecting Conflicting Gradients (PCGrad) [6], which introduces orthogonal projection between two gradients, and BiLevel Optimisation with Orthogonal Projection (Bloop) [9], which projects the auxiliary gradient orthogonally to an exponential moving average of the main gradient. All of them, however, are formulated for raw gradients under plain descent.

Under Adam and AdamW [14, 15] the applied step is a diagonally preconditioned momentum direction, so orthogonality between raw gradients no longer implies non-interference, and moments accumulated from a combined direction carry the auxiliary signal into the main task’s steps. Methods that adapt the optimiser itself keep per-task or per-term moments [16, 17], or, in continual learning, project a new task’s Adam update into the null space of previous tasks’ features [18], protecting tasks that are no longer trained. We instead protect the main task trained concurrently with the auxiliary one, and keep both moments free of the auxiliary signal, which requires an update that

D1 leaves the descent on the main objective unchanged to first order, equal to what the training on the main task would have produced alone;

D2 bounds the loss the auxiliary term adds to the main objective to second order;

D3 biases the auxiliary term toward coordinates the main gradient does not persistently use.

The contributions of this paper are: i) we formulate the search for equifinal diversity as a constrained problem and derive a cheap local surrogate for it, ii) we introduce NANO, an optimiser-level construction that realises $\mathrm { D } \bar { 1 } \mathrm { - } \mathrm { D } 3$ within AdamW, and iii) we show on H2CM that it produces spread in the unobserved transpiration fraction $R = T / \mathrm { E T }$ , without degrading the fit to the observations.

## 2.1 Repulsion loss

We train an ensemble of M models on identical batch sequences. Let $\pmb { \theta } ^ { ( m ) } \in \mathbb { R } ^ { n }$ be the parameters of one member $m , { \mathcal { L } } _ { \mathrm { m a i n } }$ its loss to the observed targets, and $\mu ( \pmb \theta ^ { ( m ) } ) \in \mathbb { R }$ an internal quantity of the hybrid model, not constrained directly by those targets. In our experiments, $\mu ( \pmb \theta ^ { ( m ) } )$ is the spatio-temporal mean transpiration T. With $\bar { N ( m ) } = \{ \nu ^ { = } , \nu ^ { + } \}$ we denote the closest diagnostic values among the other members below and above $\mu ( \pmb \theta ^ { ( m ) } )$ . The repulsion loss is

$$
\mathcal { L } _ { \mathrm { a u x } } \left( \pmb { \theta } ^ { ( m ) } \right) = \tau \operatorname* { m e a n } _ { \nu \in N ( m ) } \exp \left( - \frac { r _ { \nu } } { \tau } \right) , \qquad r _ { \nu } = \sqrt { ( \mu ( \pmb { \theta } ^ { ( m ) } ) - \nu ) ^ { 2 } + \varepsilon _ { \mu } ^ { 2 } } ,\tag{1}
$$

where $\varepsilon _ { \mu } > 0$ is a small constant for numerical stability and $\tau > 0$ sets the range of the repulsion between members. Since $\nu ^ { \pm }$ move as the other members train, $\mathcal { L } _ { \mathrm { a u x } }$ is non-stationary even at fixed $\pmb \theta ^ { ( m ) }$ and the scheme is a simultaneous update rather than gradient descent on a joint objective.

What we would like to find is

$$
\operatorname* { m i n } _ { \theta ^ { ( 1 ) } , \ldots , \theta ^ { ( M ) } } \sum _ { m } \mathcal { L } _ { \mathrm { a u x } } \big ( \theta ^ { ( m ) } \big ) \qquad \mathrm { s . t . } \qquad \mathcal { L } _ { \mathrm { m a i n } } \big ( \theta ^ { ( m ) } \big ) \leq \mathcal { L } _ { \mathrm { m a i n } } ^ { \star } + \delta \quad \forall 1 \leq m \leq M ,\tag{2}
$$

where $\mathcal { L } _ { \mathrm { m a i n } } ^ { \star }$ is the loss reachable without the auxiliary term and $\delta$ is a small constant. Note that we do not solve Equation (2) directly, we only use it to motivate the local surrogate derived next.

## 2.2 Non-interfering updates under AdamW

Why a weighted sum fails. At step t with learning rate $\alpha ,$ , we define both gradients as $g _ { \mathrm { m a i n } } =$ $\nabla _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { m a i n } } ( \pmb { \theta } _ { t } ^ { ( m ) } )$ and $g _ { \mathrm { a u x } } = \nabla _ { \theta } \mathcal { L } _ { \mathrm { a u x } } ( \theta _ { t } ^ { ( m ) } )$ for a model $m .$ . The default multi-objective construction with a weight w is $\mathcal { L } = \mathcal { L } _ { \mathrm { m a i n } } + w \mathcal { L } _ { \mathrm { a u x } }$ and gives the direction $d _ { \mathrm { s u m } } = { \displaystyle g _ { \mathrm { m a i n } } + w g _ { \mathrm { a u x } } } .$ This violates D1, as the auxiliary term contributes $- \alpha w \langle \pmb { g } _ { \mathrm { m a i n } } , \pmb { g } _ { \mathrm { a u x } } \rangle$ to $\Delta \mathcal { L } _ { \mathrm { m a i n } } , i . e .$ , an increase in the main loss whenever $\langle \pmb { g } _ { \mathrm { m a i n } } , \pmb { g } _ { \mathrm { a u x } } \rangle < 0$ . It also offers no stable scale for w, as $\| \pmb { g } _ { \mathrm { a u x } } \|$ is set by the inter-member distances and decays as members separate. The one-sided adaptation of PCGrad to our asymmetric setting projects the auxiliary gradient only and leaves $\pmb { g } _ { \mathrm { m a i n } }$ untouched. With the projection $\begin{array} { r } { \pi ( \pmb { x } ; \pmb { y } ) = \pmb { x } - \frac { \langle \pmb { x } , \pmb { y } \rangle } { \| \pmb { y } \| ^ { 2 } } \pmb { y } } \end{array}$ , the update direction becomes $d _ { \mathrm { P C } } = g _ { \mathrm { m a i n } } + \pi ( g _ { \mathrm { a u x } } ; g _ { \mathrm { m a i n } } )$ for which $\langle { \pmb g } _ { \mathrm { m a i n } } , { d } _ { \mathrm { P C } } \rangle = \| { \pmb g } _ { \mathrm { m a i n } } \| ^ { 2 }$ holds. Hence, to first order, the auxiliary term moves θ only along directions in which ${ \mathcal { L } } _ { \mathrm { m a i n } }$ is constant. This realises D1 for plain gradient descent, but it does not carry over to Adam optimiser’s preconditioning.

Keeping the moments free of the auxiliary signal. Adam optimisation applies a step $- \alpha P \hat { m }$ where mˆ is a bias-corrected first moment and $P = \mathrm { d i a g } \left( ( \sqrt { \hat { v } } ) ^ { - 1 } \right)$ is constructed from the second moment. Running Adam on the summed objective accumulates m $ \beta _ { 1 } m + ( 1 - \beta _ { 1 } ) ( g _ { \mathrm { m a i n } } + g _ { \mathrm { a u x } } )$ and $v  \beta _ { 2 } v + ( 1 - \beta _ { 2 } ) ( g _ { \mathrm { m a i n } } + g _ { \mathrm { a u x } } ) ^ { 2 }$ . Since P depends on $\mathbf { \pmb { g } } _ { \mathrm { a u x } } \mathrm { : }$ , the main task’s per-coordinate step sizes are affected by the auxiliary signal and inherit its non-stationarity. We therefore accumulate both moments from the main gradient alone, i.e.,

$$
m _ { \mathrm { m a i n } } \gets \beta _ { 1 } m _ { \mathrm { m a i n } } + ( 1 - \beta _ { 1 } ) g _ { \mathrm { m a i n } } , \qquad v _ { \mathrm { m a i n } } \gets \beta _ { 2 } v _ { \mathrm { m a i n } } + ( 1 - \beta _ { 2 } ) g _ { \mathrm { m a i n } } ^ { 2 } ,\tag{3}
$$

Table 1: Test-set main loss and spread of R across 10 members on H2CM, as mean and standard deviation over three seeds. A good method keeps the fit unchanged while revealing disagreement in the unobserved partitioning.
<table><tr><td>Method</td><td>median  $\mathcal { L } _ { \mathrm { m a i n } } ^ { \mathrm { t e s t } } \downarrow$ </td><td> $\operatorname* { m a x } \mathcal { L } _ { \mathrm { m a i n } } ^ { \mathrm { t e s t } } \downarrow$ </td><td>R std ↑</td><td>R spread ↑</td></tr><tr><td>Baseline</td><td> $1 . 9 1 9 \pm 0 . 0 1 7$ </td><td> $2 . 1 3 0 \pm 0 . 0 1 0$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 3$ </td><td> $0 . 0 3 1 \pm 0 . 0 1 3$ </td></tr><tr><td>PCGrad</td><td> $1 . 8 8 7 \pm 0 . 0 2 5$ </td><td> $2 . 1 6 3 \pm 0 . 0 2 6$ </td><td> $0 . 1 7 7 \pm 0 . 0 0 7$ </td><td> $0 . 6 2 1 \pm 0 . 0 0 9$ </td></tr><tr><td>Bloop</td><td> $2 . 1 3 5 \pm 0 . 0 6 4$ </td><td> $2 . 4 2 9 \pm 0 . 1 5 2$ </td><td> $0 . 1 0 3 \pm 0 . 0 1 5$ </td><td> $0 . 3 4 0 \pm 0 . 0 4 4$ </td></tr><tr><td>NANO (ours)</td><td> $1 . 8 7 6 \pm 0 . 0 1 6$ </td><td> $2 . 0 3 4 \pm 0 . 0 7 6$ </td><td> $0 . 1 4 3 \pm 0 . 0 2 6$ </td><td> $0 . 4 9 1 \pm 0 . 0 9 3$ </td></tr></table>

with the usual bias corrections, and precondition main and auxiliary tasks with $\hat { \pmb v } _ { \mathrm { m a i n } }$ . This is how D3 is realised: the auxiliary term is scaled by the coordinate-wise history of the main gradient, damped where the main objective is persistently active, and amplified where it is idle. That amplification is unbounded, as $\sqrt { \hat { v } _ { \operatorname* { m a i n } } ^ { ( i ) } \to 0 }$ , so for a floor fraction $c \geq 0$ we define

$$
\begin{array} { r l r l } & { P _ { c } = \operatorname { d i a g } \Big ( \big ( \sqrt { \hat { v } _ { \operatorname* { m a i n } } } + \varepsilon _ { c } \big ) ^ { - 1 } \Big ) , } & & { \varepsilon _ { c } ^ { ( i ) } = \operatorname* { m a x } \Big ( c \cdot \underset { j \in B ( i ) } { \operatorname { m e d i a n } } \sqrt { \hat { v } _ { \operatorname* { m a i n } } ^ { ( j ) } } , \ \varepsilon _ { 0 } \Big ) , } \end{array}\tag{4}
$$

where $\varepsilon _ { \mathrm { 0 } }$ is a small constant for numerical stability and $B ( i )$ is the parameter tensor containing coordinate i, so the floor is relative to that tensor’s own scale. We use $P _ { \mathrm { m a i n } } ~ = ~ P _ { c _ { \mathrm { m a i n } } }$ and $P _ { \mathrm { a u x } } = P _ { c _ { \mathrm { a u z } } }$ with $c _ { \mathrm { a u x } } \geq c _ { \mathrm { m a i n } }$ , so the floor bounds the auxiliary amplification where the main gradient is idle, while $c _ { \operatorname* { m a i n } } = 0$ leaves the main task optimisation exactly AdamW’s.

Projecting after preconditioning. An exponential moving average estimates a quantity assumed fixed on the averaging timescale, whereas the auxiliary signal moves with the other members, so we use the instantaneous gradient $\mathbf { \pmb { g } } _ { \mathrm { a u x } } .$ . Prior to any correction, the step direction is $P _ { \mathrm { m a i n } } \hat { m } _ { \mathrm { m a i n } } +$ $P _ { \mathrm { a u x } } g _ { \mathrm { a u x } }$ , for which

$$
\Delta \mathcal { L } _ { \mathrm { m a i n } } \approx - \alpha \langle g _ { \mathrm { m a i n } } , P _ { \mathrm { m a i n } } \hat { m } _ { \mathrm { m a i n } } \rangle - \alpha \langle g _ { \mathrm { m a i n } } , P _ { \mathrm { a u x } } g _ { \mathrm { a u x } } \rangle ,\tag{5}
$$

so D1 requires $\langle g _ { \mathrm { m a i n } } , P _ { \mathrm { a u x } } g _ { \mathrm { a u x } } \rangle = 0 \ :$ . The preconditioned auxiliary direction, not the raw gradient, must be orthogonal to the main gradient. We also remove a second direction $\hat { m } _ { \operatorname* { m a i n } }$ to bound the second-order term and satisfy D2, as described subsequently.

The projection $\Pi ( { \pmb x } ; { \pmb y } , { \pmb z } ) = \pi \big ( \pi ( { \pmb x } ; { \pmb y } ) ; \pi ( { \pmb z } ; { \pmb y } ) \big )$ is constructed by applying the projection π twice, first against y and then against the part of z that is orthogonal to y, resulting in a projection

$$
\tilde { g } _ { \mathrm { a u x } } = \Pi \bigl ( P _ { \mathrm { a u x } } g _ { \mathrm { a u x } } ; g _ { \mathrm { m a i n } } , \hat { m } _ { \mathrm { m a i n } } \bigr )\tag{6}
$$

against the span of $g _ { \mathrm { m a i n } }$ and $\hat { m } _ { \mathrm { m a i n } }$ and hence $\langle g _ { \mathrm { m a i n } } , \tilde { g } _ { \mathrm { a u x } } \rangle = \langle \hat { m } _ { \mathrm { m a i n } } , \tilde { g } _ { \mathrm { a u x } } \rangle = 0$ . Since $\langle g _ { \mathrm { m a i n } } , \tilde { g } _ { \mathrm { a u x } } \rangle = 0 .$ , Equation (5) simplifies to $\Delta \mathcal { L } _ { \mathrm { m a i n } } \approx - \alpha \langle \pmb { g } _ { \mathrm { m a i n } } , \pmb { P } _ { \mathrm { m a i n } } \hat { \pmb { m } } _ { \mathrm { m a i n } } \rangle$ , exactly the first-order change the main task alone would produce on this batch.

Bounding the magnitude. Orthogonality fixes the direction of the auxiliary term but not its size, while the second-order term grows with $\| \tilde { \pmb g } _ { \mathrm { a u x } } \|$ . Two rescalings bound it and realise D2. With $\pmb { u } = P _ { \mathrm { m a i n } } \hat { m } _ { \mathrm { m a i n } }$ being the main step direction, the first bound is in $\ell _ { 2 }$ norm, i.e.,

$$
\tilde { g } _ { \mathrm { a u x } }  \mathrm { m i n } \biggl ( 1 , \frac { \lvert \lvert u \rvert } { \lvert \lvert \tilde { g } _ { \mathrm { a u x } } \rvert \rvert } \biggr ) \tilde { g } _ { \mathrm { a u x } } ,\tag{7}
$$

so the auxiliary term never dominates the step in raw length. The second bound is measured in the geometry the preconditioner induces, $i . e . , P _ { \mathrm { m a i n } } = { H } ^ { - 1 }$ with $\pmb { H } = \mathrm { d i a g } ( \sqrt { \hat { \pmb { v } } _ { \mathrm { m a i n } } } )$ , the diagonal curvature proxy Adam’s step implicitly inverts. Substituting H for the Hessian, a step −αd along $\pmb { d } = \pmb { u } + \lambda \tilde { \pmb { g } } _ { \mathrm { a u x } }$ contributes the second-order term $\bar { \frac { 1 } { 2 } } \alpha ^ { 2 } d ^ { \top } H d$ . Writing $q _ { \operatorname* { m a i n } } = \pmb { u } ^ { \top } \pmb { H } \pmb { u }$ and $q _ { \mathrm { a u x } } = \tilde { g } _ { \mathrm { a u x } } ^ { \top } H \tilde { g } _ { \mathrm { a u x } } ,$ we require the auxiliary contribution not to exceed the main one, $i . e .$ $\lambda ^ { 2 } q _ { \mathrm { a u x } } \leq q _ { \mathrm { m a i n } }$ . The largest $\lambda \in \ [ 0 , 1 ]$ satisfying this is obtained by

$$
\lambda = \operatorname* { m i n } \Bigl ( 1 , \ : \sqrt { q _ { \operatorname* { m a i n } } / q _ { \operatorname* { a u x } } } \Bigr ) , \qquad \mathrm { r e s u l t i n g ~ i n } \qquad d ^ { \top } H d \leq 2 q _ { \operatorname* { m a i n } } .\tag{8}
$$

The cross-term $\pmb { u } ^ { \top } \pmb { H } ( \lambda \tilde { \pmb { g } } _ { \mathrm { a u x } } )$ vanishes, since $c _ { \mathrm { m a i n } } = 0$ gives $\pmb { H } \pmb { u } = \hat { { \pmb m } } _ { \mathrm { m a i n } }$ and $\tilde { g } _ { \mathrm { a u x } }$ is orthogonal to $\hat { m } _ { \mathrm { m a i n } }$ . The second-order term of the step is therefore at most twice the one the main branch would have produced alone. Both rescalings are non-negative scalars, so $\langle \pmb { g } _ { \mathrm { m a i n } } , \lambda \tilde { \pmb { g } } _ { \mathrm { a u x } } \rangle =$ $\langle \hat { m } _ { \mathrm { m a i n } } , \lambda \tilde { g } _ { \mathrm { a u x } } \rangle = 0$ continues to hold exactly. With decoupled weight decay $\gamma ,$ the step is $\pmb { \theta } _ { t + 1 } = ( 1 - \alpha \gamma ) \pmb { \theta } _ { t } - \alpha \pmb { d }$ . We call the resulting optimiser NANO, summarised in Algorithm 1. The decay term is identical to the one present in AdamW [15] and does not interact with the projection.

![](images/f778c3a22875f0291645ea904c9d4d5b4c3050ee06da9208adcc3e6e75ac6656.jpg)  
Figure 2: Main test loss of 10 members per method over 3 seeds, relative to the baseline median. A method preserves the fit if none of its members exceeds the baseline’s worst test loss.

## 3 Experiments

We train 10 H2CM members concurrently on identical batch sequences at $1 ^ { \circ }$ spatial and daily temporal resolution, repelled along the spatio-temporal mean transpiration $T _ { \ast }$ , and repeat each ensemble over three seeds. We report the main loss $\mathcal { L } _ { \mathrm { m a i n } } ^ { \mathrm { t e s t } }$ on the test set, and the diversity through the standard deviation and spread of the transpiration fraction $R = T / \mathrm { E T }$ over members. As a baseline, we train a random-seed ensemble without $\mathcal { L } _ { \mathrm { a u x } }$ . NANO, PCGrad [6] and Bloop [9] are all trained with the same repulsion loss. Table 1 and Figure 2 report the spread in R together with the main test loss relative to the baseline median. Spread counts as equifinality only if the members producing it fit the observations as well as the baseline does. NANO raises the standard deviation of R from 0.009 to 0.143 and its spread from 0.031 to 0.491, with all members below the baseline’s worst test loss. Bloop yields a smaller spread at a worse fit. PCGrad attains the largest spread at a comparable median, but its worst members lie at or above the baseline’s worst.

## 4 Limitations

First, neutrality is approximate. Orthogonality and the second-order bound hold only per step, so residuals can accumulate into a drift in ${ \mathcal { L } } _ { \mathrm { m a i n } }$ . Moreover, $\pmb { H } = \mathrm { d i a g } ( \sqrt { \hat { v } _ { \mathrm { m a i n } } } )$ is a running gradient magnitude, not a curvature, so the cross term need not vanish under the true Hessian. Second, the repulsion acts only between nearest neighbours, so the spread we obtain is a lower bound on the equifinality present. Third, the scheme costs one extra backward pass per member per step, and since $\mathcal { L } _ { \mathrm { a u x } }$ depends on the neighbours’ current $\mu ,$ the M members advance in lockstep, paced by the slowest. Fourth, we neither ablate NANO’s components nor sweep the repulsion strength per method. Finally, we test one model and one diagnostic. The update uses only the two gradients and AdamW’s moments, and assumes nothing about the model, the loss, $\mathrm { o r } \ \mu$ beyond differentiability, so we expect it to transfer. Only $\tau ,$ which depends on $\mu ,$ , must be set anew for each diagnostic.

## 5 Conclusion

We elicit the equifinality in H2CM by repelling ensemble members along the unobserved transpiration, subject to the fit to observations not degrading. The constraint has to be imposed inside the optimiser: NANO accumulates both AdamW moments from the main gradient alone, projects the preconditioned auxiliary direction, dynamically bounds its length and steers the repulsion towards the coordinates the main gradient leaves idle. The main objective then changes to first order exactly as under AdamW alone, with a second-order term at most twice that of AdamW in the preconditioner’s geometry. On H2CM, this raises the spread in R from 0.031 to 0.491, with all members below the baseline’s worst test loss.

## References

[1] Markus Reichstein, Gustau Camps-Valls, Bjorn Stevens, Martin Jung, Joachim Denzler, Nuno Carvalhais, and Prabhat. “Deep learning and process understanding for data-driven Earth system science”. In: Nature 566.7743 (2019), pp. 195–204. DOI: 10.1038/s41586-019- 0912-1.

[2] Keith Beven. “A manifesto for the equifinality thesis”. In: Journal of Hydrology 320.1-2 (2006), pp. 18–36. DOI: 10.1016/j.jhydrol.2005.07.007.

[3] Scott Jasechko, Zachary D. Sharp, John J. Gibson, S. Jean Birks, Yi Yi, and Peter J. Fawcett. “Terrestrial water fluxes dominated by transpiration”. In: Nature 496.7445 (2013), pp. 347–350. DOI: 10.1038/nature11983.

[4] Stephen P. Good, David Noone, and Gabriel Bowen. “Hydrologic connectivity constrains partitioning of global terrestrial water fluxes”. In: Science 349.6244 (2015), pp. 175–177. DOI: 10.1126/science.aaa5931.

[5] Zavud Baghirov, Markus Reichstein, Basil Kraft, Bernhard Ahrens, Marco Körner, and Martin Jung. “H2CM (v1.0): Hybrid modeling of global water-carbon cycles constrained by atmospheric and land observations”. In: Geoscientific Model Development 19.10 (2026), pp. 4467–4496. DOI: 10.5194/gmd-19-4467-2026.

[6] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. “Gradient Surgery for Multi-Task Learning (NeurIPS)”. In: Advances in Neural Information Processing Systems. Curran Associates Inc., 2020, pp. 5824–5836. arXiv: 2001 . 06782 [cs.LG, cs.CV, cs.RO, stat.ML].

[7] Ozan Sener and Vladlen Koltun. “Multi-Task Learning as Multi-Objective Optimization”. In: Advances in Neural Information Processing Systems (NeurIPS). Vol. 31. Curran Associates Inc., 2018, pp. 525–536.

[8] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. “Conflict-Averse Gradient Descent for Multi-task learning”. In: Advances in Neural Information Processing Systems (NeurIPS). Vol. 34. Curran Associates Inc., 2021. arXiv: 2110.14048 [cs.LG, cs.AI].

[9] Yu-Guan Hsieh, James Thornton, Eugene Ndiaye, Michal Klein, Marco Cuturi, and Pierre Ablin. “Careful with that Scalpel: Improving Gradient Surgery with an EMA”. In: International Conference on Machine Learning (ICML). JMLR, 2024. arXiv: 2402.02998 [cs.LG, stat.ML].

[10] Lukas Liebel and Marco Körner. Auxiliary Tasks in Multi-task Learning. 2018. arXiv: 1805. 06334 [cs.CV, cs.LG].

[11] Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. “Orthogonal Gradient Descent for Continual Learning”. In: International Conference on Artificial Intelligence and Statistics (AISTATS). Proceedings of Machine Learning Research. 2020, pp. 3762–3773. arXiv: 1910. 07104 [cs.LG].

[12] Yunshu Du, Wojciech M. Czarnecki, Siddhant M. Jayakumar, Mehrdad Farajtabar, Razvan Pascanu, and Balaji Lakshminarayanan. Adapting Auxiliary Losses Using Gradient Similarity. 2018. arXiv: 1812.02224 [stat.ML].

[13] Lucio M. Dery, Yann Dauphin, and David Grangier. “Auxiliary Task Update Decomposition: The Good, The Bad and The Neutral”. In: International Conference on Learning Representations (ICLR). 2021. arXiv: 2108.11346 [cs.LG].

[14] Diederik P. Kingma and Jimmy Ba. “Adam: A Method for Stochastic Optimization”. In: International Conference on Learning Representations (ICLR). 2015. arXiv: 1412 . 6980 [cs.LG].

[15] Ilya Loshchilov and Frank Hutter. “Decoupled Weight Decay Regularization”. In: International Conference on Learning Representations (ICLR). 2017. arXiv: 1711.05101 [cs.LG].

[16] Enneng Yang, Junwei Pan, Ximei Wang, Haibin Yu, Li Shen, Xihua Chen, Lei Xiao, Jie Jiang, and Guibing Guo. “AdaTask: A Task-Aware Adaptive Learning Rate Approach to Multi-Task Learning”. In: AAAI Conference on Artificial Intelligence. Vol. 37. 9. AAAI Press, 2023, pp. 10745–10753. DOI: 10.1609/aaai.v37i9.26275.

[17] Itzik Malkiel and Lior Wolf. “MTAdam: Automatic Balancing of Multiple Training Loss Terms”. In: Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2021, pp. 10713–10729. DOI: 10.18653/v1/ 2021.emnlp-main.837.

[18] Shipeng Wang, Xiaorong Li, Jian Sun, and Zongben Xu. “Training Networks in Null Space of Feature Covariance for Continual Learning”. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2021, pp. 184–193. DOI: 10.1109/cvpr46437.2021. 00025.

## A Main loss

We use the main loss of H2CM [5],

$$
\mathcal { L } _ { \operatorname* { m a i n } } = \sum _ { k = 1 } ^ { K } \frac { 1 } { N _ { k } } \sum _ { \ell = 1 } ^ { N _ { k } } \bigl ( y _ { k , \ell } - \hat { y } _ { k , \ell } \bigr ) ^ { 2 } ,\tag{9}
$$

where $y _ { k , \ell }$ and $\hat { y } _ { k , \ell }$ are the observed and predicted values of data constraint $k ,$ both z-standardised per constraint, and $N _ { k }$ is the number of data points of constraint k. The constraints are monthly terrestrial water storage anomalies, snow water equivalent, evapotranspiration, runoff, fraction of absorbed photosynthetically active radiation, gross primary productivity and net ecosystem exchange, together with the mean annual carbon use efficiency.

## B Algorithm

Algorithm 1 One optimisation step for member m   
1: $g _ { \mathrm { m a i n } } \gets \nabla _ { \theta } \mathcal { L } _ { \mathrm { m a i n } } ; \quad g _ { \mathrm { a u x } } \gets \nabla _ { \theta } \mathcal { L } _ { \mathrm { a u x } }$   
2: update $m _ { \mathrm { m a i n } } , v _ { \mathrm { m a i n } }$ by Equation (3), bias-correct to $\hat { m } _ { \operatorname* { m a i n } } , \hat { v } _ { \operatorname* { m a i n } }$   
3: build $P _ { \mathrm { m a i n } } , P _ { \mathrm { a u x } }$ by Equation $( 4 )$ with floors $c _ { \mathrm { m a i n } } \leq c _ { \mathrm { a u x } }$ ▷ D3   
4: $\pmb { u }  \pmb { P } _ { \operatorname* { m a i n } } \hat { \pmb { m } } _ { \mathrm { m a i n } } ; \quad \pmb { H }  \mathrm { d i a g } ( \sqrt { \hat { \pmb { v } } _ { \mathrm { m a i n } } } )$   
5: $\tilde { g } _ { \mathrm { a u x } } \gets \Pi \big ( P _ { \mathrm { a u x } } g _ { \mathrm { a u x } } ; g _ { \mathrm { m a i n } } , \hat { m } _ { \mathrm { m a i n } } \big )$ by Equation (6) ▷ D1   
6: $\tilde { g } _ { \mathrm { a u x } } \gets \operatorname* { m i n } \bigl ( 1 , \| u \| / \| \tilde { g } _ { \mathrm { a u x } } \| \bigr ) \tilde { g } _ { \mathrm { a u x } }$ ▷ D2   
7: $q _ { \operatorname* { m a i n } }  u ^ { \top } H \underline { { u } } ; \quad q _ { \operatorname { a u x } }  \tilde { g } _ { \operatorname { a u x } } ^ { \top } H \tilde { g } _ { \operatorname { a u x } }$   
8: $\lambda  \operatorname* { m i n } ( 1 , \sqrt { q _ { \operatorname* { m a i n } } / q _ { \mathrm { a u x } } } )$ ▷ D2   
9: $\pmb { \theta } \gets ( 1 - \alpha \gamma ) \pmb { \theta } - \alpha ( \pmb { u } + \lambda \tilde { \pmb { g } } _ { \mathrm { a u x } } )$

## C Hyperparameters

All training settings are inherited from the H2CM baseline or are optimiser defaults: AdamW with base learning rate $5 \cdot 1 0 ^ { - 3 }$ , linear warmup over 25 epochs from a tenth of that value, cosine annealing to $5 \cdot 1 0 ^ { - 5 }$ over the remaining 275, batch size 512, gradient clipping at value 0.5, and moment decay rates $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$

The repulsion Equation (1) uses $\varepsilon _ { \mu } = 1 0 ^ { - 6 }$ and $\tau = 0 . 1$ . Since τ depends on $\mu ,$ it sets the distance in mean transpiration below which neighbouring members repel. $\tau = 0 . 1$ was chosen as a separation sufficient for the members to be distinct. The preconditioner floors of Equation (4) are $c _ { \operatorname* { m a i n } } = 0$ leaving the main branch exactly $\mathrm { \bf A d a m W } ^ { \prime } \mathrm { \bf s } ,$ and $c _ { \mathrm { a u x } } = 0 . 1$ , chosen empirically, with $\varepsilon _ { 0 } = 1 0 ^ { - 8 }$ The second projection direction in Equation (6) is dropped when the component of $\hat { m } _ { \mathrm { m a i n } }$ orthogonal to $\pmb { g } _ { \mathrm { m a i n } }$ falls below $1 0 ^ { - 3 } \Vert \hat { m } _ { \operatorname* { m a i n } } \Vert$ , where it is numerically unreliable.

For the comparison experiments, all three methods share the same $\mathcal { L } _ { \mathrm { a u x } } .$ . PCGrad is the one-sided variant described in Section 2.2, whose combined gradient ${ \pmb g } _ { \mathrm { m a i n } } + \pi ( { \pmb g } _ { \mathrm { a u x } } ; { \pmb g } _ { \mathrm { m a i n } } )$ is passed to AdamW. Bloop uses $\rho = 0 . 0 1$ for its gradient EMA, the value used throughout the original work, and the auxiliary weight $w = 1 . 0$

## D Computational resources

All runs were performed on a cluster node with NVIDIA A100 GPUs, one per ensemble member, so an ensemble of $M = 1 0$ members occupies 10 GPUs and 160 CPU cores. Training one ensemble for 300 epochs takes roughly 12 h with our update, against 9 h for the random-seed baseline, the difference being the extra backward pass per member per step.

## E Per-member ET fit

Table 1 and Figure 2 report the aggregate main loss of Equation (9), which could hide a shift in ET compensated by the other constraints. Figure 3 therefore shows the mean ET of each member of the

![](images/fd3959edcc23250a58241ce9215593d2011b5a9e8dcbad8a6b886e89d107d565.jpg)  
Figure 3: Per-member mean ET against the main test loss, for the baseline and NANO ensembles over three seeds.  
baseline and NANO ensembles over three seeds. The NANO members span a range of ET similar to the baseline’s, so the repulsion redistributes ET between its components rather than changing total ET.