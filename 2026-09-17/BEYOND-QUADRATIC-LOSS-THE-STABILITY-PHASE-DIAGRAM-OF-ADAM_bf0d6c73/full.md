# BEYOND QUADRATIC LOSS: THE STABILITY PHASE DIAGRAM OF ADAM

PREPRINT

Gaoxiang Tang IIIS, Tsinghua University

Huanran Chen College AI, Tsinghua University

Ziming Liu College AI, Tsinghua University Shanghai Qizhi Institute MetaCircle

## ABSTRACT

Loss spikes are recurrent instabilities in neural-network training and can arise from multiple mechanisms. For Adam in particular, macroscopic loss spikes have been linked to optimizer dynamics, yet how its two momentum timescales govern them remains unclear. We investigate this dependence by mapping training dynamics across the $( \beta _ { 1 } , \beta _ { 2 } )$ plane. Across a range of model-task settings, an approximately linear boundary, $1 - \beta _ { 2 } = C ( 1 - \beta _ { 1 } )$ , separates spiky from non-spiky dynamics, whereas a one-dimensional quadratic loss produces approximately cubic slope. A one-dimensional superquadratic loss $L ( x ) \propto { \bar { | x | } } ^ { n }$ recovers the near-linear scaling and links the boundary coefficient to the effective loss exponent n. We further show that confident cross-entropy losses develop a core-wall landscape comprising a narrow quadratic core followed by a steep wall, which produces effective superquadratic behavior at the scale of an optimizer update. Together, these results connect Adam loss spikes to both the mismatch between momentum timescales and finite-scale superquadratic loss geometry beyond the Hessian.

## 1 Introduction

![](images/e695e8dd0e432a0357bcfe63f7efeba03d771298e104ee0f8b4e16297b7d5a28.jpg)

![](images/ba619206e6865bdf8b3f4abe15b5b76412dca24f449d6702abb4aef403c63d83.jpg)

![](images/6996f33fee94913f7cf3b9700062614c7b096f519a6175a631974a0742f112b8.jpg)  
Figure 1: Overview of the empirical boundary scaling and finite-scale loss geometry. (a) Adam stability phase diagram in the $( \beta _ { 1 } , \beta _ { 2 } )$ plane for $L ( x ) { \overset { \cdot } { = } } x ^ { 2 } / 2$ . A power-law fit gives $1 - \beta _ { 2 } \simeq 1 \bar { 5 . } 6 ( 1 - \bar { \beta } _ { 1 } ) ^ { 3 . 0 7 }$ , showing the approximately cubic scaling of the quadratic model. (b) Phase diagram for the modular-division Transformer, whose detected boundary follows the near-linear relation $1 - \dot { \beta _ { 2 } } \simeq 2 . 2 1 ( 1 ^ { - } \beta _ { 1 } ) ^ { 0 . 9 3 }$ . In (a,b), color denotes the measured oscillation period. (c) Directional loss slice $L ( s ) = L ( \theta + s \hat { d } _ { \mathrm { p r e c o n d } } )$ at a pre-spike point of the modular-division Transformer, where $d _ { \mathrm { p r e c o n d } }$ is the Adam-preconditioned gradient direction. The green dashed line shows the magnitude of a typical update, which is comparable to the flat-core width,

Loss spikes are abrupt, macroscopic loss excursions that recur across neural-network training settings. Their causes depend on the training regime. In Adam training, such spikes have been widely reported Chowdhery et al. (2023); Molybog et al. (2023); Thilak et al. (2022). Under a local quadratic approximation, gradient descent is stable when $\dot { \lambda _ { \mathrm { m a x } } ( H _ { t } ) } < 2 / \eta$ . Crossing this threshold can induce Edge-of-Stability dynamics characterized by non-monotonic loss excursions and spikes Xing et al. (2018); Jastrzebski et al. (2020); Cohen et al. (2021). For adaptive optimizers, the analogous criterion replaces the raw Hessian with a preconditioned Hessian, leading to the Adaptive Edge of Stability for Adam and RMSProp Cohen et al. (2023, 2025). Sustained violations of this effective stability threshold have also been linked to macroscopic Adam loss spikes Bai et al. (2026b). Other mechanisms apply in more specific settings. Numerical Feature Inflation explains spikes in low-precision models trained with cross-entropy loss Liu et al. (2026), while weight-norm criticality applies to models with scale-invariant layers Li et al. (2026).

Adam's first and second gradient moments have memory timescales set by $\beta _ { 1 }$ and $\beta _ { 2 }$ , respectively Kingma and Ba (2015). We investigate how the mismatch between $\bar { \beta } _ { 1 }$ and $\beta _ { 2 }$ governs loss spikes. We focus on macroscopic loss spikes triggered by violations of Adam's EoS condition, rather than the microscopic oscillations associated with short-period attractors Bock and Weiß (2019); Fong and Yang (2026). Across diverse models and tasks, we identify the near-linear stability boundary $1 - \beta _ { 2 } \propto \left( 1 - \beta _ { 1 } \right)$ shown in Figure 1(b). To isolate the underlying dynamics, we study Adam on the one-dimensional quadratic loss $L ( x ) = k x ^ { 2 } / 2$ . This model instead produces the approximately cubic boundary $1 - \beta _ { 2 } \propto ( 1 - \beta _ { 1 } ) ^ { 3 }$ in Figure 1(a). The discrepancy motivates an analysis beyond the conventional quadratic approximation.

We then analyze the Adam stability phases for a superquadratic one-dimensional loss $L ( x ) = | x | ^ { n }$ with $n > 2$ . We find a wedge-shaped spiky phase bounded approximately by

$$
C _ { L } ( 1 - \beta _ { 1 } ) \geq 1 - \beta _ { 2 } \geq C _ { R } ( 1 - \beta _ { 1 } ) .\tag{1}
$$

The right boundary follows from the competition between the effective learning rate $\eta _ { \mathrm { e f f } }$ and the critical learning rate ηcrit, giving $C _ { R } = 2 ( n - 2 ) / n$ . The left boundary involves more complex oscillatory dynamics and is described empirically by $C _ { L } \simeq 1 2 ( n - 2 ) / ( 3 n - 4 )$ . Recent work obtained the same right-boundary condition as the local-stability boundary of full Adam on even-degree degenerate polynomials Bai et al. (2026a). We recover it for real $n > 2$ under the quiet-phase approximation and use it to delimit recurrent EoS-triggered spikes.

Local loss landscapes are often analyzed quadratically, although their Hessian spectra typically contain a broad nearzero bulk and a few isolated outliers Sagun et al. (2018). This structure has motivated river-valley descriptions of neural-network optimization Xing et al. (2018); Wen et al. (2024). In confident cross-entropy models, the Hessian can vanish altogether, which can be understood using the generalized Gauss-Newton decomposition Schraudolph (2002).

We find that directional slices along the preconditioned gradient exhibit a core-wall landscape composed of a flat quadratic core, a steep superquadratic wall, and an outer rollover, as illustrated in Figure 1(c). We show that this structure arises naturally from confident cross-entropy and derive the scale of its quadratic core. During training. the landscape undergoes finite-scale progressive sharpening, in which the preconditioned sharpness increases while the flat core contracts. Once a typical update becomes comparable to the core width, Adam probes the wall and encounters a large effective exponent n. We estimate this exponent at pre-spike points and substitute it into the empirical relation for $\bar { C } _ { L }$ . The resulting estimates capture the scale of the observed boundary coefficients.

This work makes three contributions. First, we identify a near-linear phase boundary separating spiky and non-spiky Adam dynamics across diverse models and tasks. Second, we connect the transition from cubic to linear boundary scaling to superquadratic loss geometry and show how the memory timescales of the first and second moments control the phase boundaries. Third, we show how confident cross-entropy losses develop a core-wall landscape consisting of a quadratic core, a superquadratic wall, and an outer rollover, and we relate the effective exponent at the optimizer's update scale to the empirical boundary coefficient.

## 2 Beta boundaries and slopes for six models

We study six model-task settings: one-layer Transformers Vaswani et al. (2017) for modular division and addition modulo 53, a character-level Transformer for next-character prediction on Tiny Shakespeare, an MLP and a VGG11- style CNN Simonyan and Zisserman (2015) for CIFAR-10 classification Krizhevsky (2009), and a convolutional autoencoder for binarized MNIST reconstruction LeCun et al. (1998). All models use AdamW Loshchilov and Hutter (2019) with cross-entropy for classification and language modeling or binary cross-entropy for reconstruction. Within each setting, only $\beta _ { 1 }$ and $\beta _ { 2 }$ vary, while the architecture, data, random seed, and other optimizer hyperparameters remain fixed. Appendix B.1 gives the full configurations.

We scan logarithmically spaced memory gaps $\delta = 1 - \beta _ { 1 }$ and $\gamma = 1 - \beta _ { 2 } .$ All six diagrams cover $\delta \in [ 1 0 ^ { - 3 } , 0 . 2 ]$ and $\gamma \in [ 5 \times \bar { 1 0 } ^ { - 4 } , 0 . 3 ]$ . After the initial transient, we extend trajectories with unresolved periods and identify macroscopic oscillations from low-frequency spectral peaks that meet the prominence criterion in Appendix B.2.

For each fixed $\gamma _ { \mathrm { : } }$ the boundary is the immediate right neighbor of the largest sampled $\delta$ with no accepted period. We fit these points over the selected ranges using log $\overset { \cdot } { \gamma } = \log \overset { \cdot } { C } + p \log \overset { \cdot } { \delta }$ to estimate $C$ and $p .$ All six phase diagrams

(c) Tiny Shakespeare

(a) Modular Division

(e) CIFAR CNN  
(d) CIFAR MLP  
![](images/d01a3f4b9e7c21d217a8d35d423bdc32315584befb833ebb77592ce903970ec3.jpg)

$$
\overline { { C _ { L } = 1 2 ( n - 2 ) / ( 3 n - 4 ) } }
$$

![](images/bf19782cc9790d420225fac749a12861151bfba9f52172b682fd01c639516e93.jpg)

![](images/205e72664ea6acb80c958453872cb6cc6869510cbec4fa0f85a3ec5095fa8ad7.jpg)

![](images/ad12d8761baaa3b78aed09b5786e5bbe26c26d5d734d87be9118cb634e2975ca.jpg)

![](images/88a7bda1a9938982b7e0acd0b0e16bce8d8d9fcf1b9ff2e80b74262007ea1eaf.jpg)

![](images/bf52c8a274d9e198f22c4b2d1615e554d9aaa43f108a4ba821d8ed402abd6c38.jpg)  
Figure 2: Phase boundaries for six models. (a-f) Modular division, modular addition, Tiny Shakespeare, a CIFAR-10 MLP, a CIFAR-10 CNN, and an MNIST autoencoder, respectively. The axes are the memory gaps $\delta = 1 - \beta _ { 1 }$ and $\gamma = 1 - \beta _ { 2 }$ , and color indicates the detected macroscopic oscillation period in optimizer steps. Red points mark boundary locations, white dashed curves show power-law fits $\gamma = C \delta ^ { p }$ , and solid cyan lines show estimates from the effective loss exponents in Table 1 and the empirical relation $C _ { L } = 1 2 ( n - 2 ) / ( 3 n - 4 )$ in Section 3. Gray cells have no accepted period. Unsampled regions are uncolored.

approach the linear scaling $\gamma \propto \delta ,$ with fitted exponents $p = 0 . 9 3 , 0 . 8 5 , 0 . 7 5 , 0 . 9 5 , 1 . 0 6 , 0 . 9 5$ for (a-f), respectively. The region above the boundary, where $( 1 - \beta _ { 2 } ) / ( \bar { 1 } - \beta _ { 1 } )$ is larger, generally contains no detected macroscopic spikes. Within the spiky region, periods tend to increase as β2 approaches one and depend more weakly on $\beta _ { 1 }$

## 3 One-dimensional toy model of Adam stability

To isolate the boundary mechanism, we first state the relevant local stability condition.

Lemma 1 (Local stability condition for Adam). Consider a network in a locally quadratic region with Hessian $H _ { t }$ and preconditioner $D _ { t } = \mathrm { d i a g } [ ( \sqrt { \hat { v } _ { t } } + \epsilon ) ^ { - 1 } ]$ . With both held fixed locally and late-time bias corrections neglected, Adam is unstable if the following condition holds Cohen et al. (2023); Bai et al. (2026b):

$$
\frac { 1 - \beta _ { 1 } } { 1 + \beta _ { 1 } } \lambda _ { \operatorname* { m a x } } ( D _ { t } H _ { t } ) > \frac { 2 } { \eta } .\tag{2}
$$

Appendix A.1 gives the derivation. Its one-dimensional specialization follows immediately.

Corollary 2 (Stability condition for Adam in 1D). For one-dimensional curvature $\lambda _ { t } = L ^ { \prime \prime } ( x _ { t } ) > 0$ deine

$$
\eta _ { \mathrm { e f f , t } } = \frac { \eta } { \sqrt { \hat { v } _ { t } } + \epsilon } , \qquad \eta _ { \mathrm { c r i t , t } } = \frac { 2 ( 1 + \beta _ { 1 } ) } { ( 1 - \beta _ { 1 } ) \lambda _ { t } } .\tag{3}
$$

Under the same local approximation, Adam is unstable when $\eta _ { \mathrm { e f f , t } } > \eta _ { \mathrm { c r i t , t } }$

We first study the quadratic loss $L ( x ) = k x ^ { 2 } / 2$ using Adam with $k = 1$ , learning rate $\eta = 0 . 1$ , numerical stabilizer $\epsilon = 1 0 ^ { - 3 0 }$ , no weight decay, $x _ { 0 } = 1$ , and $m _ { 0 } = v _ { 0 } = 0$ . Each beta pair is trained for 250,000 updates, of which the first 125,000 are discarded. We estimate the period from upward crossings of the local stability threshold and set $T = 1 2 5 , 0 0 0 / N _ { \mathrm { c r o s s } }$ when at least three crossings are detected. Figure 3(a) shows the approximately cubic boundary $\gamma \simeq 1 5 . 6 \delta ^ { 3 . 0 \dot { 7 } }$ , consistent with Figure 1(a). Repeated crossings occur mainly below this boundary. Appendix C.2 examines the effect of weight decay on 1-dimensional models.

![](images/6cab7844a243bbf37e7ac0a66e466f759bad0961ecb75c9358c2f4345f849965.jpg)  
Figure 3: Phase diagrams for the one-dimensional losses $L ( x ) = k | x | ^ { n } / n$ with $k = 1$ . (a-g) Results for $n =$ 2, 2.1, 2.2, 2.5, 3, 5, 8, respectively, where $\delta = 1 - \beta _ { 1 } , \gamma = 1 - \beta _ { 2 }$ , and color indicates the period. Solid cyan lines show the empirical left boundary $\gamma = 1 2 ( n - 2 ) \delta / ( 3 n - 4 )$ , and dashed pink lines show the predicted right boundary $\gamma = 2 ( n - 2 ) \delta / n$ . Both coefficients vanish at $n = 2 ,$ ,where the boundary is approximately cubic. The blue, orange, and gray points in (b) identify the trajectories in Figure 4. (h) Measured boundary coefficients as functions of n.

The boundary remains approximately cubic across the tested k and $\eta ,$ and also across € while $\epsilon \ll \sqrt { \hat { v } _ { t } }$ Appendix C.1 reports these controls, which support an explanation based on the relative memory timescales of the two moments. Because the quadratic exponent differs substantially from the near-unit neural-network exponents, we next consider $L ( x ) = k | x | ^ { \bar { n } } / n$ . For $1 < n < 2 $ divergent curvature near zero produces rapid oscillations rather than separated macroscopic spikes. We therefore focus on the superquadratic case $n > 2 .$

For $n > 2 ,$ Figure 3(b-g) shows a wedge-shaped spiky region between two lines of approximately unit log-log slope,

$$
C _ { L } ( 1 - \beta _ { 1 } ) \geq 1 - \beta _ { 2 } \geq C _ { R } ( 1 - \beta _ { 1 } ) .\tag{4}
$$

We derive the right boundary from Adam's local stability condition and characterize the left boundary empirically. The boundaries observed in the six neural-network phase diagrams correspond to the left boundary $C _ { L }$ . The right boundary $C _ { R }$ requires the loss to remain superquadratic asymptotically as the displacement approaches zero. As Section 4 shows, the measured neural-network landscapes instead enter a quadratic core in this limit, so the right-boundary mechanism does not apply to them

During a quiet interval $( 1 - \beta _ { 2 } ) g _ { t + 1 } ^ { 2 } \ll \beta _ { 2 } v _ { t }$ makes self-decay dominate the second moment, giving $v _ { t } \propto \beta _ { 2 } ^ { t }$ . While $\sqrt { \hat { v } _ { t } } \gg \epsilon ,$ this decay raises $\eta _ { \mathrm { e f f , t } }$ . The approach to zero on a superquadratic loss simultaneously lowers the curvature and raises $\eta _ { \mathrm { c r i t , t } }$ . Figure $4 ( \mathrm { d - f } )$ illustrates three resulting regimes. If $\eta _ { \mathrm { c r i t , t } }$ grows faster, $\eta _ { \mathrm { e f f , t } }$ cannot catch it and sustained macroscopic spikes are suppressed, as in panel (f). If $\eta _ { \mathrm { e f f } , 1 }$ repeatedly reaches the threshold, each crossing triggers a spike whose large gradients replenish $v _ { t }$ and lower $\eta _ { \mathrm { e f f , t } }$ , as in panel (e). Alternatively, the trajectory can be captured by a short-period microscopic attractor, where $\eta _ { \mathrm { e f f , t } }$ saturates below $\eta _ { \mathrm { c r i t , t } }$ and rapid oscillations persist without macroscopic spikes, as in panel (d). Theorem 3 formalizes the first regime for trajectories attracted to the rescaled fixed points. We absorb $1 / n$ into k and write $L ( x ) = k | x | ^ { n }$ , which leaves the boundary coefficients unchanged.

Theorem 3 (Right boundary for $L ( x ) = k | x | ^ { n }$ models). Consider Adam applied to the loss landscape $L ( x ) = k | x | ^ { n }$ where $\frac { 1 } { \gamma _ { \cdot } } < \beta _ { 1 } < 1 , 0 < \beta _ { 2 } < 1$ , and $n > 2 .$ Assume $v _ { 0 } > 0 , v _ { t + 1 } = \beta _ { 2 } v _ { t }$ , and $\epsilon = 0 .$ Then trajectories attracted to either nonzero fxed point of the rescaled dynamics eventually cease to exhibit EoS-triggered spikes when

$$
\beta _ { 1 } \beta _ { 2 } ^ { - n / [ 2 ( n - 2 ) ] } < 1 .\tag{5}
$$

(f) non-spiky regime effective and critical LR  
![](images/6e2ddeedaa2fb6e97b442197200e5b453901b3d327053cfd39d658ac40f5aaa3.jpg)

![](images/1477e7f90ffedef5320ef3c75671b42e2e3526033d5e16cd54177514cbc003f8.jpg)

![](images/cc7d843cfcd410842dc923dd8db97919a684dcd7d71f097bb500f0b6e7b96af0.jpg)

![](images/b6ea97dd8fe426a68931c3a9bb4ca157eb0c458d4184c3a82f210912ec6665e9.jpg)

![](images/7ae2390f4a2e94c33c12801e8a082ddff4c8ca63394f6792bfcc9939d7ad63bf.jpg)

![](images/a5722832ae3f1335911a8a8bd3cd7e71aa364f434253439837f5ebf85c3729d1.jpg)  
Figure 4: Loss and learning-rate dynamics for $n = 2 . 1$ . The columns correspond to $( \beta _ { 1 } , \beta _ { 2 } ) \ : = \ : ( 0 . 9 9 9 , 0 . 9 9 5 )$ (0.99, 0.995), and (0.9, 0.995), matching the blue, orange, and gray points in Figure 3(b). (a–c) Loss curves. (d–f) Effective learning rate $\eta _ { \mathrm { e f f } }$ in green and critical rate $\eta _ { \mathrm { c r i t } }$ in pink. In (d), rapid oscillations persist while $\eta _ { \mathrm { e f f } }$ remains below $\eta _ { \mathrm { c r i t } }$ and eventually saturates. In (e), $\eta _ { \mathrm { e f f } }$ repeatedly reaches the threshold, producing spikes that reduce it. In (f), faster growth of $\eta _ { \mathrm { c r i t } }$ suppresses sustained macroscopic spikes.

Proof sketch. An equivalent condition was derived by Bai et al. (2026a) from the local stability of the full normalized Adam dynamics on even-degree polynomials. Here the approximation $v _ { t + 1 } = \beta _ { 2 } v _ { t }$ decouples the second moment and yields the shorter proof below for real $n > 2$ . Define the stability ratio $S _ { t } = \eta _ { \mathrm { e f f , t } } / \eta _ { \mathrm { c r i t , t } }$ . If $S _ { t }$ eventually remains below one, the local stability condition precludes further EoS-triggered spikes. During a scale-free quiet interval, set $q = \beta _ { 2 } ^ { 1 / [ 2 ( n - 2 ) ] }$ and $x _ { t } = q ^ { t } y _ { t }$ . This gives $\eta _ { \mathrm { e f f , t } } | x _ { t } | ^ { n - 2 } = \eta _ { \mathrm { e f f , 0 } } | y _ { t } | ^ { n - 2 }$ and yields the autonomous recurrence

$$
y _ { t + 1 } = a y _ { t } - b y _ { t - 1 } - c | y _ { t } | ^ { n - 2 } y _ { t } , \qquad b = \beta _ { 1 } \beta _ { 2 } ^ { - n / [ 2 ( n - 2 ) ] } ,\tag{6}
$$

where $a , b ,$ and c are time independent. Whenever the nonzero fixed points exist within the theorem's parameter range, both satisfy $S _ { * } < 1$ . Linearization and the Jury conditions show that these fixed points are stable exactly when $b < 1$ so attracted trajectories eventually cease to spike. The boundary b = 1 gives $\beta _ { 2 } = \beta _ { 1 } ^ { 2 ( n - 2 ) / n }$ . Appendix A.2 provides the fixed-point bounds and full stability calculation. □

Along these trajectories, $x _ { t } \sim q ^ { t } y _ { * }$ gives $g _ { t } ^ { 2 } / v _ { t } \propto q ^ { 2 t }  0 .$ , consistent with the self-decay approximation. In the phase diagram coordinates, the right boundary is

$$
\gamma _ { R } ( \delta , n ) = 1 - ( 1 - \delta ) ^ { 2 ( n - 2 ) / n } = \frac { 2 ( n - 2 ) } { n } \delta + O ( \delta ^ { 2 } ) .\tag{7}
$$

The plotted line uses the leading small-δ coefficient $C _ { R } = 2 ( n - 2 ) / n .$ The $n = 2$ boundary is not linear and instead exhibits the approximately cubic scaling in Figure 3(a).

These regimes show that the left boundary involves more complex oscillatory dynamics and finite-step effects, and a complete analytical explanation remains open. We summarize its dependence on n by the empirical relation $C _ { L } \simeq \bar { 1 } 2 ( n - 2 ) / ( 3 n - \bar { 4 } )$ , which captures the trend in Figure 3(h).

## 4 Core-wall landscape and its exponential-sum mechanism

The one-dimensional results suggest that the near-linear β-boundary reflects a superquadratic loss profile at the scale explored by Adam. Although the conventional quadratic approximation remains valid sufficiently close to a low-loss minimum, its domain of validity can be much smaller than an optimizer update Ma et al. (2022a). The optimizer then probes the steep finite-scale growth outside this neighborhood, motivating our core-wall landscape description. It refines the usual river-valley picture by resolving the basin around a low-loss minimum into an inner quadratic core, a superquadratic wall, and an outer rollover. Related notions of flatness have been connected to generalization Hochreiter and Schmidhuber (1997); Keskar et al. (2017), while basin-like parameter regions have been used to study behavioral preservation and fine-tuning robustness Peng et al. (2024); Chen et al. (2026). We quantify this directional geometry using an effective logarithmic exponent. Experimentally, we observe the core-wall landscape at pre-spike points in all six model-task settings. Appendix D.1 details the pre-spike selection procedure and illustrates how contraction of the quadratic core toward the optimizer's update scale produces progressive sharpening at finite scales.

Definition 4 (Core-wall landscape). For a frozen direction $\hat { d } ,$ define the directional slice ${ \phi ( s ) = L ( \theta + s \hat { d } ) }$ and let S\* be one of its local minima. For $r > 0$ , define the two excess-loss branches and their local exponents by

$$
\Delta L _ { \pm } ( r ) = \phi ( s _ { * } \pm r ) - \phi ( s _ { * } ) , \qquad n _ { \mathrm { e f f , \pm } } ( r ) = \frac { \mathrm { d } \log \Delta L _ { \pm } ( r ) } { \mathrm { d } \log r } .\tag{8}
$$

As r increases, the branches pass successively through a quadratic core with $n _ { \mathrm { e f f , \pm } } \simeq 2$ , a superquadratic wall beginning at the first crossing of the operational threshold $n _ { \mathrm { e f f , \pm } } = 2 . 5$ , and an outer rollover in which $n _ { \mathrm { e f f , \pm } }$ decreases from its wall value. We call this ordered basin structure the core-wall landscape.

To quantify this observation across the six tasks in Figure 2, we evaluate low-loss checkpoints along the Adampreconditioned gradient direction

$$
\phi _ { t } ( s ) = L ( \theta _ { t } + s \hat { d } _ { t } ) , \qquad \hat { d } _ { t } = \frac { D _ { t } g _ { t } } { \lVert D _ { t } ^ { 1 / 2 } g _ { t } \rVert _ { 2 } } .\tag{9}
$$

During descent and near the bottoms of temporal loss valleys, these slices exhibit a core-wall landscape. Conventional progressive sharpening tracks the growth of Hessian or preconditioned-Hessian sharpness Cohen et al. (2021, 2023). Our slices reveal a finite-scale counterpart in which the quadratic core also contracts toward the optimizer's update scale, as illustrated in Appendix Figure 9. We next show that this geometry arises naturally from confident cross-entropy losses. The flat core follows from Hessian collapse as the predicted probabilities approach their hard targets.

Lemma 5 (Hessian collapse for confident cross-entropy). On a fixed dataset of N samples with one-hot targets $y _ { i } ,$ let $\begin{array} { r } { L ( \theta ) = N ^ { - 1 } \sum _ { i } \dot { \ell _ { i } } ( \theta ) } \end{array}$ be the unregularized softmax cross-entropy, with twice differentiable $l o g i t s ~ z _ { i } ( \theta )$ and probabilities $p _ { i } = \mathrm { s o f t m a x } ( z _ { i } )$ . Along any parameter sequence such that $p _ { i } \to y _ { i }$ for every sample, assume that $\bar { J } _ { i } = \partial z _ { i } / \partial \theta$ and $\nabla _ { { \theta } } ^ { 2 } z _ { i k }$ remain uniformly bounded. Then $\| \mathrm { \dot { V } } _ { \theta } ^ { 2 } L \| _ { 2 } \to 0$

Proof sketch. The generalized Gauss-Newton decomposition writes each sample Hessian as $\begin{array} { r } { J _ { i } ^ { \top } C _ { i } J _ { i } + \sum _ { k } ( p _ { i k } - } \end{array}$ $y _ { i k } ) \nabla _ { \theta } ^ { 2 } z _ { i k }$ , where $C _ { i } = \mathrm { d i a g } ( p _ { i } ) - p _ { i } p _ { i } ^ { \intercal }$ . If $\varepsilon _ { i } = 1 - p _ { i , c _ { i } }$ , then $\| C _ { i } \| _ { 2 } \leq 2 \varepsilon _ { i }$ and $\| p _ { i } - y _ { i } \| _ { 1 } = 2 \varepsilon _ { i }$ . The assumed derivative bounds therefore make both terms $O ( \varepsilon _ { i } )$ , and the finite average vanishes as every $\varepsilon _ { i } \to 0 .$ Appendix ${ \mathrm { A } } . 3$ gives the full norm bounds. □

Lemma 5 identifies the origin of the flat core but does not determine the size of the neighborhood in which the quadratic approximation is accurate. We derive this scale from the exponential-sum structure of confident cross-entropy.

Definition 6 (Confident cross-entropy loss slice). Let $i \in \{ 1 , \ldots , N \}$ index samples or tokens, let yi be the correct class for sample ¿, and let $j \neq y _ { i }$ index its incorrect classes. Along the frozen slice $\boldsymbol { \theta } ( s ) = \boldsymbol { \theta } _ { * } + s \boldsymbol { \hat { d } } ,$ define the correct-versus-incorrect logit margin as $m _ { i j } ( s ) = z _ { i , y _ { i } } ( s ) - z _ { i j } ( s )$ . The exact cross-entropy slice is

$$
L _ { \mathrm { C E } } ( s ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \left( 1 + \sum _ { j \neq y _ { i } } e ^ { - m _ { i j } ( s ) } \right) .\tag{10}
$$

In the confident regime, the relevant margins are large and positive, so $\log ( 1 + u ) \simeq u$ . If they are locally linearized as $m _ { i j } ( s ) \simeq m _ { i j } - a _ { i j } s .$ where $m _ { i j } = m _ { i j } ( 0 )$ and $a _ { i j } = - \mathrm { d } m _ { i j } ( s ) / \mathrm { d } s | _ { s = 0 }$ , we call

$$
L _ { \mathrm { c o n f } } ( s ) = \frac { 1 } { N } \sum _ { i } \sum _ { j \neq y _ { i } } e ^ { - m _ { i j } + a _ { i j } s } .\tag{11}
$$

the condent cross-entropy loss slice.

In confident cross-entropy, the large margins $m \gg 1$ suppress the baseline loss, $L _ { \mathrm { c o n f } } ( 0 ) \propto \sum e ^ { - m } \simeq 0$ , whereas the directional slopes a can have much larger magnitudes. At the local loss minimum preceding the second spike of the modular-division Transformer, for example, $\langle \overline { { { m } } } \rangle _ { w } \pm \mathrm { s t d } _ { w } ( m ) = 1 7 . 0 8 \pm 1 . 2 9 $ , whereas $\langle a \rangle _ { w } \simeq 1 . 3 3 \times \mathrm { i } 0 ^ { 2 }$ and $\sigma _ { a } \simeq 2 . 3 0 \times 1 0 ^ { 5 }$ . Such large slope magnitudes make factors of the form $e ^ { - m + a s }$ grow exponentially under very small displacements, leaving only an $O ( 1 / \sigma _ { a } )$ neighborhood in which the quadratic approximation remains accurate. The following theorem makes this scale precise under a Gaussian model for a.

![](images/9bd90484c428cce23d27f8719cd148d0075ad022e67744d7c3fba841785a5d17.jpg)

![](images/9102a0acaa08e5d0b66da81559d3adc24f5638337dcb5855e37721275c02485b.jpg)  
Figure 5: Directional loss slices shown as $\Delta L = L - L$ \* versus $| s - s _ { * } |$ on log-log axes. (a) The selected modulardivision pre-spike point at step 37,210 for $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ . Solid curves are the measured left and right branches, dashed curves are piecewise core-wall landscape fits, circles and crosses mark the fitted knots, and the vertical dashed line marks the typical projected update. Shading identifies the core, wall, and rollover fit intervals. (b) A synthetic confident cross-entropy slice. Solid curves show the exact loss, and dashed curves show its confident approximation. Purple dotted lines mark $1 / \sigma _ { a }$ . In both examples, the wall emerges at the exponential-sum scale.

Theorem 7 (Quadratic-core scale of confident cross-entropy). Approximate the CE-weighted distribution of directional margin slopes by a Gaussian, $a \stackrel { w } { \sim } \mathcal N ( \mu _ { a } , \sigma _ { a } ^ { 2 } )$ . Within this Gaussian model, with the origin chosen at the minimum of the confident surrogate and $\sigma _ { a } > 0$ , the slice has a quadratic-core scale $s _ { \mathrm { c o r e } } = O ( 1 / \sigma _ { a } )$

Proof sketch. The normalized confident loss is the moment-generating function of the weighted slope distribution. Gaussianity and stationarity at $s = 0$ give

$$
\frac { \Delta L _ { \mathrm { c o n f } } ( s ) } { L _ { \mathrm { c o n f } } ( 0 ) } = e ^ { ( \sigma _ { a } s ) ^ { 2 } / 2 } - 1 , \qquad n _ { \mathrm { e f f } } = \frac { ( \sigma _ { a } s ) ^ { 2 } e ^ { ( \sigma _ { a } s ) ^ { 2 } / 2 } } { e ^ { ( \sigma _ { a } s ) ^ { 2 } / 2 } - 1 } .\tag{12}
$$

Thus $n _ { \mathrm { e f f } } = 2 + ( \sigma _ { a } s ) ^ { 2 } / 2 + O ( ( \sigma _ { a } s ) ^ { 4 } )$ , so departure from the quadratic core occurs at $| s | = O ( 1 / \sigma _ { a } )$ ; the threshold $n _ { \mathrm { e f f } } = 2 . 5$ gives $\vert s \vert \simeq 0 . 9 6 4 / \sigma _ { a }$ Appendix $_ { \mathrm { A } . 4 }$ contains the full derivation. □

To resolve the local shape, we refine $s _ { * }$ for each frozen slice and plot $\Delta L = \phi _ { t } ( s ) - \phi _ { t } ( s _ { * } )$ against $| s - s _ { * } |$ on logarithmic axes. Figure 5 compares a measured modular-division slice with a synthetic confident cross-entropy example. We construct the latter as $\begin{array} { r } { L _ { \mathrm { C E } } ( s ) = \log ( 1 + \sum _ { i = 1 } ^ { 1 6 } e ^ { - M _ { i } + a _ { i } s } ) } \end{array}$ , with $M _ { i } \sim \mathrm { U n i f o r m } [ 1 0 , 2 0 ]$ and $a _ { i } \sim \mathcal { N } ( 0 , 3 0 0 0 ^ { 2 } )$ , then shift the slopes so that $\begin{array} { r } { \sum _ { i } e ^ { - M _ { i } } a _ { i } = 0 } \end{array}$ and $s _ { * } = 0$ In both cases, the logarithmic slope rises from an approximately quadratic core to a superquadratic wall and then decreases in the outer rollover.

For the measured slice in Figure ${ \mathfrak { I } } ( { \mathfrak { a } } )$ , averaging the two branches gives $n _ { \mathrm { e f f } } \simeq 2 . 1 8$ in the core and $n _ { \mathrm { e f f } } \simeq 1 1 . 0$ along the wall. The core-wall transition occurs at $\bar { O ( 1 0 ^ { - 6 } ) }$ , comparable to the typical update scale $| s _ { \mathrm { u p d } } | \simeq 6 . 3 \times 1 0 ^ { - 6 }$ while the rollover begins at $O ( 1 0 ^ { - 5 } )$ . Consistently, $1 / \sigma _ { a } \simeq 4 . 3 5 \times 1 0 ^ { - 6 }$ for the measured slice. The synthetic slice in Figure 5(b) exhibits the same three regimes with $\mathrm { 1 } / \dot { \sigma } _ { a } \simeq 2 . 9 7 \times 1 0 ^ { - 4 }$

Beyond the core, exponential reweighting favors terms with increasingly extreme directional slopes. $\mathrm { I f } ~ \langle a \rangle _ { s }$ denotes their loss-weighted mean at displacement s, then

$$
\frac { \mathrm { d } \log L _ { \mathrm { c o n f } } } { \mathrm { d } \log | s | } = s \langle a \rangle _ { s } .\tag{13}
$$

The growth of $| \langle a \rangle _ { s } |$ produces a superquadratic wall that admits a finite-interval power-law approximation, yielding the effective exponent used in the one-dimensional model. At larger displacements, the logarithm in exact cross-entropy changes a dominant contribution from exponential to asymptotically linear growth, reducing the effective exponent and producing the rollover.

Finally, we connect these directional measurements to the $\beta \mathrm { . }$ -phase diagrams. We use the fitted wall exponent over the resolved intermediate regime as an effective input to the empirical left-boundary relation from Section 3,

$$
{ C _ { L } } ( n _ { \mathrm { w a l l } } ) \simeq { \frac { 1 2 ( n _ { \mathrm { w a l l } } - 2 ) } { 3 n _ { \mathrm { w a l l } } - 4 } } , \qquad 1 - \beta _ { 2 } = { C _ { L } } ( n _ { \mathrm { w a l l } } ) ( 1 - \beta _ { 1 } ) .\tag{14}
$$

Table 1: Effective wall exponents and the resulting empirical left-boundary coefficients for the six model-task settings.
<table><tr><td>Model-task setting</td><td> $n _ { \mathrm { w a l l } }$ </td><td>Estimated  $C _ { L }$ </td></tr><tr><td>Modular division</td><td>10.90</td><td>3.72</td></tr><tr><td>Modular addition</td><td>10.58</td><td>3.71</td></tr><tr><td>Tiny Shakespeare</td><td>9.41</td><td>3.67</td></tr><tr><td>CIFAR-10 MLP</td><td>10.48</td><td>3.71</td></tr><tr><td>CIFAR-10 CNN</td><td>6.52</td><td>3.49</td></tr><tr><td>MNIST autoencoder</td><td>10.22</td><td>3.70</td></tr></table>

The non-spiky region below the right boundary of the pure power-law model arises asymptotically as $s \to 0$ while the loss remains superquadratic. In the measured neural-network landscapes, however, this limit enters the quadratic core, cutting off the pure-superquadratic right-boundary mechanism before it can be observed. We therefore compare the measured wall exponent only with the empirical left boundary.

For a controlled comparison, all six model-task settings use the same six reference beta pairs and apply the same core-wall-rollover fitting procedure at the pre-spike points. We first obtain one jointly fitted $n _ { \mathrm { w a l l } }$ for each beta pair and then report the unweighted mean over the six pairs in Table 1. We evaluate $C _ { L }$ from this mean exponent. Appendix D.1 describes how the pre-spike points are located, and Appendix D.2 gives the directional-slice fit and aggregation procedure.

## 5 Conclusion and outlook

Taken together, our results make three contributions. First, they establish a near-linear phase boundary between spiky and non-spiky Adam dynamics across diverse models and tasks. Second, our analysis attributes the shift from cubic to linear boundary scaling to superquadratic loss geometry and shows how the first- and second-moment memory timescales set the phase boundaries. Third, we show that confident cross-entropy forms a core-wall landscape with a quadratic core, a superquadratic wall, and an outer rollover, linking the effective exponent at the optimizer's update scale to the empirical boundary coefficient.

Looking ahead, three questions appear especially promising.

1. Basin- and valley-like loss landscapes arise in broader underparameterized neural-network settings Bosman et al. (2020); Ruiz-Garcia et al. (2021). Can the core-wall exponential-sum mechanism also explain these landscapes?

2. More generally, non-quadratic geometry with an exponent $n _ { \mathrm { e f f } }$ that changes across direction, scale, and training time is likely to be the norm. Can an optimizer estimate $n _ { \mathrm { e f f } }$ at its update scale and adapt its learning rate or moment timescales to the evolving phase boundary?

3. What mechanism drives progressive sharpening at finite scales? How do the margins $m _ { i j }$ and their directional derivatives $a _ { i j } = - \mathrm { d } m _ { i j } / \mathrm { d } s$ evolve during training?

## 6 Related work

Edge of Stability (EoS). Trajectory studies linked learning rates to sharp directions and showed that progressive sharpening reaches the quadratic threshold $2 / \eta .$ producing EoS dynamics Xing et al. (2018); Jastrzębski et al. (2018); Jastrzebski et al. (2020); Cohen et al. (2021). Adaptive EoS uses the preconditioned Hessian, central-flow theory describes averaged oscillations, and decoupling between $g _ { t } ^ { 2 }$ and $v _ { t }$ can trigger Adam spikes Cohen et al. (2023, 2025); Bai et al. (2026b). Low-dimensional analyses cover quadratic two-cycles and frozen-EoS restoration Bock and Weiß (2019); Fong and Yang (2026). On even-degree degenerate polynomials, Bai et al. (2026a) obtain the same principal stability boundary as Theorem 3 from full Adam dynamics, together with fixed-point existence and a linear convergence rate. Our quiet-phase reduction gives a shorter proof for real $n > 2$ and interprets the condition as the right edge of a recurrent-spike wedge. Complementary work studies near-zero Hessian bulk, loss slices, and river-valley dynamics Sagun et al. (2018); Li et al. (2018); Wen et al. (2024). We instead ask how finite-scale geometry along the Adam-preconditioned gradient and two moment-memory gaps organize the spike phase diagram.

Loss spikes. Loss spikes in large-language-model training may require checkpoint-and-data interventions Chowdhery et al. (2023), while unusually large Adam updates can become weakly aligned with descent Molybog et al. (2023).

Momentum-dependent oscillations, spikes, and divergence have also been observed Ma et al. (2022b). Proposed mechanisms include lower-loss-as-sharper geometry and lagging second moments that amplify preconditioned curvature even on quadratics Li et al. (2023); Bai et al. (2026b). Slingshot connects cyclic adaptive-optimizer instability to grokking, although neither implies the other Power et al. (2022); Thilak et al. (2022). Numerical Feature Inflation and weight-norm criticality explain spikes in low precision or scale-invariant networks with weight decay Liu et al. (2026); Li et al. (2026). Our one-dimensional examples require none of these ingredients and attribute beta-boundary scaling to competing moment timescales and superquadratic finite-scale geometry.

Core-wall landscapes. Flat minima have been associated with generalization Hochreiter and Schmidhuber (1997); Keskar et al. (2017), although parameter-space sharpness is not invariant to function-preserving reparameterizations Dinh et al. (2017). Volume flatness, worst-case loss increases, and Hessian sharpness address different questions. Recent LLM studies define basins by preserving alignment or task performance under parameter perturbations Peng et al. (2024); Chen et al. (2026). These characterize fine-tuning robustness, whereas our core-wall landscape uses local training loss along a dynamically selected direction to explain spike onset.

Optimization with non-quadratic curvature. Beyond the quadratic threshold, prior work studies unstable convergence, multiscale subquadratic landscapes, and cubic self-stabilization Ahn et al. (2022); Ma et al. (2022a); Damian et al. (2023). Generalized self-concordance bounds finite-displacement departures for logistic loss, while the generalized Gauss-Newton decomposition shows why pointwise curvature need not determine finite-step dynamics Bach (2010); Schraudolph (2002). Neither derives the directional scale $O ( 1 / \sigma _ { a } )$ of a confident cross-entropy exponential sum. Our quadratic core is followed by a superquadratic wall whose finite-interval exponent predicts the beta-boundary coefficient.

## References

Kwangjun Ahn, Jingzhao Zhang, and Suvrit Sra. Understanding the unstable convergence of gradient descent. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 247–257, Baltimore, Maryland, USA, 2022. PMLR. URL https://proceedings . mlr . press/ v162/ahn22a.html.

Francis Bach. Self-concordant analysis for logistic regression. Electronic Journal of Statistics, 4:384–414, 2010. doi: 10.1214/09-EJS521. URL https://doi.org/10.1214/09-EJS521.

Zhiwei Bai, Jiajie Zhao, Zhangchen Zhou, Zhi-Qin John Xu, and Yaoyu Zhang. Towards understanding Adam convergence on highly degenerate polynomials, 2026a. URL https://arxiv.org/abs/2603.09581. Accepted at ICML 2026.

Zhiwei Bai, Zhangchen Zhou, Jiajie Zhao, Xiaolong Li, Zhiyu Li, Feiyu Xiong, Hongkang Yang, Yaoyu Zhang, and Zhi-Qin John Xu. Adaptive preconditioners trigger loss spikes in Adam, 2026b. URL https://arxiv. org/abs/ 2506.04805. Accepted at ICML 2026.

Sebastian Bock and Martin Georg Weiß. Non-convergence and limit cycles in the Adam optimizer. In Igor V. Tetko, Věra Kůrková, Pavel Karpov, and Fabian Theis, editors, Artificial Neural Networks and Machine Learning— ICANN 2019: Deep Learning, volume 11728 of Lecture Notes in Computer Science, pages 232–243, Cham, 2019. Springer International Publishing. doi: 10.1007/978-3-030-30484-3\_20. URL https://doi.org/10.1007/ 978-3-030-30484-3\_20.

Anna Sergeevna Bosman, Andries P. Engelbrecht, and Mardé Helbig. Visualising basins of attraction for the crossentropy and the squared error neural network loss functions. Neurocomputing, 400:113–136, 2020. doi: 10.1016/j. neucom.2020.02.113. URL https://doi.org/10.1016/j.neucom.2020.02.113.

Tony Cai, Jianqing Fan, and Tiefeng Jiang. Distributions of angles in random packing on spheres. Journal of Machine Learning Research, 14(57):1837–1864, 2013. URL https://jmlr.org/papers/v14/cai13a.html.

Huanran Chen, Yinpeng Dong, Zeming Wei, Yao Huang, Yichi Zhang, Hang Su, and Jun Zhu. Unveiling the basin-like loss landscape in large language models, 2026. URL https: //arxiv. org/abs/2505.17646. Published at ICLR 2026.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke. Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. PaLM: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023. URL https://www. jmlr.org/papers/v24/22-1144.html.

Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. In Proceedings of the Ninth International Conference on Learning Representations, Virtual Conference, 2021. OpenReview.net. URL https://openreview.net/forum?id=jh-rTtvkGeM.

Jeremy M. Cohen, Behrooz Ghorbani, Shankar Krishnan, Naman Agarwal, Sourabh Medapati, Michal Badura, Daniel Suo, David Cardoze, Zachary Nado, George E. Dahl, and Justin Gilmer. Adaptive gradient methods at the edge of stability, 2023. URL https://arxiv.org/abs/2207.14484. NeurIPS 2023 Workshop on Heavy Tails in Machine Learning.

Jeremy M. Cohen, Alex Damian, Ameet Talwalkar, J. Zico Kolter, and Jason D. Lee. Understanding optimization in deep learning with central flows, 2025. URL https://arxiv. org/abs/2410.24206. Published at ICLR 2025.

Alex Damian, Eshaan Nichani, and Jason D. Lee. Self-stabilization: The implicit bias of gradient descent at the edge of stability. In Proceedings of the Eleventh International Conference on Learning Representations, Kigali, Rwanda, 2023.OpenReview.net. URL https://openreview.net/forum?id=nhKHA59gXz.

Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep nets. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1019–1028, Sydney, Australia, 2017. PMLR. URL https://proceedings.mlr.press/v70/dinh17b.html.

Yiman Fong and Heng Yang. Provable edge-of-stability for Adam on a one-dimensional quadratic, 2026. URL https://arxiv.org/abs/2608.20638.

Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural Computation, 9(1):1–42, 1997. doi: 10.1162/neco. 1997.9.1.1. URL https://doi.org/10.1162/neco.1997.9.1.1.

Stanisław Jastrzębski, Zachary Kenton, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey. On the relation between the sharpest directions of DNN loss and the SGD step length, 2018. URL https://arxiv. org/ abs/1807.05031.

Stanislaw Jastrzebski, Maciej Szymczak, Stanislav Fort, Devansh Arpit, Jacek Tabor, Kyunghyun Cho, and Krzysztof Geras. The break-even point on optimization trajectories of deep neural networks, 2020. URL https : //arxiv. org/abs/2002.09572.

Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima, 2017. URL https://arxiv. org/ abs/1609.04836. Published at ICLR 2017.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the Third International Conference on Learning Representations, San Diego, California, USA, 2015. International Conference on Learning Representations. doi: 10.48550/arXiv.1412.6980. URL https://arxiv.org/abs/1412.6980.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, Toronto, Ontario, Canada, 2009. URL https://www.cs.toronto.edu/\~kriz/learning-features-2009-TR.pdf.

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. doi: 10.1109/5.726791. URL https://doi . org/ 10.1109/5.726791.

Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets, 2018. URL https://arxiv.org/abs/1712.09913. Advances in Neural Information Processing Systems 31.

Xiaolong Li, Zhi-Qin John Xu, and Zhongwang Zhang. Loss spike in training neural networks, 2023. URL https: //arxiv.org/abs/2305.12133.

Xiaolong Li, Zhangchen Zhou, and Zhi-Qin John Xu. Weight-norm criticality: A mechanism for loss spikes induced by the normalization and weight decay, 2026. URL https://arxiv.org/abs/2607.21005.

Hanqing Liu, Jianjun Cao, Yuanze Li, and Zijian Zhou. Grokking or glitching? how low-precision drives slingshot loss spikes, 2026. URL https://arxiv.org/abs/2605.06152.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proceedings of the Seventh International Conference on Learning Representations, New Orleans, Louisiana, USA, 2019. OpenReview.net. URL https: //openreview.net/forum?id=Bkg6RiCqY7.

Chao Ma, Daniel Kunin, Lei Wu, and Lexing Ying. Beyond the quadratic approximation: The multiscale structure of neural network loss landscapes. Journal of Machine Learning, 1(3):247–267, 2022a. doi: 10.4208/jml.220404. URL https://www.global-sci.com/jml/article/view/13179.

Chao Ma, Lei Wu, and Weinan E. A qualitative study of the dynamic behavior for adaptive gradient algorithms. In Joan Bruna, Jan Hesthaven, and Lenka Zdeborova, editors, Proceedings of the 2nd Mathematical and Scientific Machine Learning Conference, volume 145 of Proceedings of Machine Learning Research, pages 671–692, Virtual Conference, 2022b. PMLR. URL https://proceedings.mlr.press/v145/ma22a.html.

Igor Molybog, Peter Albert, Moya Chen, Zachary DeVito, David Esiobu, Naman Goyal, Punit Singh Koura, Sharan Narang, Andrew Poulton, Ruan Silva, Binh Tang, Diana Liskovich, Puxin Xu, Yuchen Zhang, Melanie Kambadur, Stephen Roller, and Susan Zhang. A theory on Adam instability in large-scale machine learning, 2023. URL https://arxiv.org/abs/2304.09871.

Sheng Yun Peng, Pin-Yu Chen, Matthew Hull, and Duen Horng Chau. Navigating the safety landscape: Measuring risks in finetuning large language models, 2024. URL https://arxiv. org/abs/2405.17374.

Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets, 2022. URL https://arxiv.org/abs/2201.02177.

Miguel Ruiz-Garcia, Ge Zhang, Samuel S. Schoenholz, and Andrea J. Liu. Tilting the playing field: Dynamical loss functions for machine learning. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 9157–9167, Virtual, 2021. PMLR. URL https://proceedings.mlr.press/v139/ruiz-garcia21a.html.

Levent Sagun, Utku Evci, V. Ugur Güney, Yann Dauphin, and Leon Bottou. Empirical analysis of the hessian of over-parametrized neural networks, 2018. URL https://arxiv.org/abs/1706.04454. ICLR 2018 Workshop Track.

Nicol N. Schraudolph. Fast curvature matrix-vector products for second-order gradient descent. Neural Computation, 14(7):1723–1738, 2002. doi: 10.1162/08997660260028683. URL https://doi.org/10.1162/ 08997660260028683.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Proceedings of the Third International Conference on Learning Representations, San Diego, California, USA, 2015. International Conference on Learning Representations. doi: 10.48550/arXiv.1409.1556. URL https: //arxiv.org/abs/1409.1556.

Vimal Thilak, Etai Littwin, Shuangfei Zhai, Omid Saremi, Roni Paiss, and Joshua Susskind. The slingshot mechanism: An empirical study of adaptive optimizers and the grokking phenomenon, 2022. URL https://arxiv. org/abs/ 2206.04817.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems, volume 30, pages 5998–6008, Long Beach, California, USA, 2017. Curran Associates, Inc. URL https://proceedings.neurips.cc/paper\_files/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, and Tengyu Ma. Understanding warmup-stable-decay learning rates: A river valley loss landscape perspective, 2024. URL https://arxiv.org/abs/2410.05192.

Chen Xing, Devansh Arpit, Christos Tsirigotis, and Yoshua Bengio. A walk with SGD, 2018. URL https : //arxiv. org/abs/1802.08770.

## A Proofs

## A.1 Proof of Lemma 1

Proof. For gradient descent, the local perturbation vector evolves as

$$
\delta _ { t + 1 } = ( I - \eta H _ { t } ) \delta _ { t } .\tag{15}
$$

Along an eigenvector of $H _ { t }$ with eigenvalue λ, this becomes

$$
\delta _ { t + 1 } = ( 1 - \eta \lambda ) \delta _ { t } .\tag{16}
$$

Stability requires $| 1 - \eta \lambda | < 1$ , which gives

$$
\eta \lambda < 2 \qquad \Longrightarrow \qquad \lambda < \frac { 2 } { \eta } .\tag{17}
$$

With first-moment momentum,

$$
m _ { t } = \beta _ { 1 } m _ { t - 1 } + ( 1 - \beta _ { 1 } ) g _ { t } , \qquad \theta _ { t + 1 } = \theta _ { t } - \eta m _ { t } .\tag{18}
$$

Linearizing $g _ { t } \approx H _ { t } \delta _ { t }$ and eliminating $m _ { t }$ gives the vector recurrence

$$
\delta _ { t + 1 } = \left[ ( 1 + \beta _ { 1 } ) I - \eta ( 1 - \beta _ { 1 } ) H _ { t } \right] \delta _ { t } - \beta _ { 1 } \delta _ { t - 1 } .\tag{19}
$$

Along a Hessian eigen-direction, the characteristic equation is

$$
r ^ { 2 } - \alpha r + \beta _ { 1 } = 0 , \qquad \alpha = ( 1 + \beta _ { 1 } ) - \eta ( 1 - \beta _ { 1 } ) \lambda .\tag{20}
$$

The Jury stability condition gives

$$
\eta ( 1 - \beta _ { 1 } ) \lambda < 2 ( 1 + \beta _ { 1 } ) ,\tag{21}
$$

or equivalently

$$
\frac { 1 - \beta _ { 1 } } { 1 + \beta _ { 1 } } \lambda < \frac { 2 } { \eta } .\tag{22}
$$

Adam replaces the raw gradient by a coordinate-wise normalized gradient:

$$
v _ { t } = \beta _ { 2 } v _ { t - 1 } + ( 1 - \beta _ { 2 } ) g _ { t } ^ { 2 } , \qquad \theta _ { t + 1 } = \theta _ { t } - \eta { \frac { \hat { m } _ { t } } { \sqrt { \hat { v } _ { t } } + \epsilon } } .\tag{23}
$$

Because $v _ { t }$ tracks squared gradients, its RMS scale is $\sqrt { \hat { v } _ { t } } ,$ giving the preconditioner

$$
D _ { t } = \mathrm { d i a g } \left( { \frac { 1 } { \sqrt { \hat { v } _ { t } } + \epsilon } } \right) .\tag{24}
$$

Freezing $D _ { t }$ and $H _ { t }$ replaces the local operator $H _ { t }$ by $D _ { t } H _ { t }$ . In the absence of first-moment momentum, the corresponding perturbation dynamics are

$$
\delta _ { t + 1 } \approx ( I - \eta D _ { t } H _ { t } ) \delta _ { t } .\tag{25}
$$

The matrix $D _ { t } H _ { t }$ is similar to the symmetric matrix $D _ { t } ^ { 1 / 2 } H _ { t } D _ { t } ^ { 1 / 2 }$ , so its eigenvalues are real. Including first-moment momentum and applying the preceding condition to the largest eigenvalue yields the instability criterion

$$
\frac { 1 - \beta _ { 1 } } { 1 + \beta _ { 1 } } \lambda _ { \operatorname* { m a x } } ( D _ { t } H _ { t } ) > \frac { 2 } { \eta } ,\tag{26}
$$

which is the Adam EoS instability condition.

## A.2 Proof of Theorem 3

Proof. Under the scale-free, bias-uncorrected dynamics, $\eta _ { \mathrm { e f f , t } } = \eta / \sqrt { v _ { t } }$ . Define the stability ratio

$$
S _ { t } = \frac { \eta _ { \mathrm { e f f , t } } } { \eta _ { \mathrm { c r i t , t } } } = \frac { ( 1 - \beta _ { 1 } ) k n ( n - 1 ) } { 2 ( 1 + \beta _ { 1 } ) } \eta _ { \mathrm { e f f , t } } | x _ { t } | ^ { n - 2 } .\tag{27}
$$

A trajectory that eventually remains at $S _ { t } < 1$ has no further threshold crossings and hence no further spikes triggered by this mechanism. Because $v _ { t } = \beta _ { 2 } ^ { t } v _ { 0 }$ , the effective learning rate satisfies $\eta _ { \mathrm { e f f , t } } = \eta _ { \mathrm { e f f , 0 } } \beta _ { 2 } ^ { - t / 2 }$ . Absorb this time dependence into the position scale by setting

$$
q = \beta _ { 2 } ^ { 1 / [ 2 ( n - 2 ) ] } , \qquad x _ { t } = q ^ { t } y _ { t } .\tag{28}
$$

Then $\eta _ { \mathrm { e f f , t } } | x _ { t } | ^ { n - 2 } = \eta _ { \mathrm { e f f , 0 } } | y _ { t } | ^ { n - 2 }$ . Eliminating the first moment from the discrete updates gives

$$
\begin{array} { r } { x _ { t + 1 } = \left( 1 + \beta _ { 1 } \beta _ { 2 } ^ { - 1 / 2 } \right) x _ { t } - \beta _ { 1 } \beta _ { 2 } ^ { - 1 / 2 } x _ { t - 1 } } \\ { - ( 1 - \beta _ { 1 } ) k n \eta _ { \mathrm { e f f } , 0 } \beta _ { 2 } ^ { - t / 2 } | x _ { t } | ^ { n - 2 } x _ { t } . } \end{array}\tag{29}
$$

Under the rescaling above, this becomes the autonomous recurrence

$$
y _ { t + 1 } = a y _ { t } - b y _ { t - 1 } - c | y _ { t } | ^ { n - 2 } y _ { t } ,\tag{30}
$$

where

$$
a = \frac { 1 + \beta _ { 1 } \beta _ { 2 } ^ { - 1 / 2 } } { q } , \qquad b = \frac { \beta _ { 1 } } { q ^ { n } } = \beta _ { 1 } \beta _ { 2 } ^ { - n / [ 2 ( n - 2 ) ] } , \qquad c = \frac { ( 1 - \beta _ { 1 } ) k n \eta _ { \mathrm { e f f , 0 } } } { q } .\tag{31}
$$

The two nonzero fixed points exist when $a - 1 - b > 0$ , equivalently $b q < 1$ , and satisfy

$$
y _ { * } = \pm \left( \frac { a - 1 - b } { c } \right) ^ { 1 / ( n - 2 ) } .\tag{32}
$$

Both have the same stability ratio because S depends only on $| y |$

$$
S _ { * } = \frac { ( n - 1 ) q ( a - 1 - b ) } { 2 ( 1 + \beta _ { 1 } ) } = \frac { ( n - 1 ) ( 1 - q ) ( 1 - b q ) } { 2 ( 1 + \beta _ { 1 } ) } .\tag{33}
$$

Existence implies $q > \beta _ { 1 } ^ { 1 / ( n - 1 ) }$ , while $q < 1$ implies b $\gamma > \beta _ { 1 }$ . Therefore,

$$
S _ { * } < \frac { ( n - 1 ) ( 1 - \beta _ { 1 } ^ { 1 / ( n - 1 ) } ) ( 1 - \beta _ { 1 } ) } { 2 ( 1 + \beta _ { 1 } ) } < \frac { ( - \ln \beta _ { 1 } ) ( 1 - \beta _ { 1 } ) } { 2 ( 1 + \beta _ { 1 } ) } < \frac { \ln 2 } { 6 } < 1 .\tag{34}
$$

To establish stability, write $y _ { t } = y _ { * } + u _ { t }$ . Expansion around either fixed point yields

$$
u _ { t + 1 } = \left[ ( n - 1 ) ( 1 + b ) - ( n - 2 ) a \right] u _ { t } - b u _ { t - 1 } + O ( u _ { t } ^ { 2 } ) ,\tag{35}
$$

with characteristic equation

$$
\lambda ^ { 2 } - \left[ ( n - 1 ) ( 1 + b ) - ( n - 2 ) a \right] \lambda + b = 0 .\tag{36}
$$

The roots satisfy $| \lambda _ { \pm } | < 1$ if and only if

$$
b < 1 , \qquad 0 < ( n - 2 ) ( a - 1 - b ) < 2 ( 1 + b ) .\tag{37}
$$

Within the stated parameter range, the second condition follows automatically from $b < 1$ . Indeed,

$$
q > { \frac { 1 } { \sqrt { 2 } } } , \qquad 1 - q < { \frac { \ln 2 } { n } } , \qquad 0 < 1 - b q < { \frac { 1 } { 2 } } ,\tag{38}
$$

SO

$$
0 < ( n - 2 ) ( a - 1 - b ) = ( n - 2 ) \frac { ( 1 - q ) ( 1 - b q ) } { q } < \frac { n - 2 } { n } \frac { \ln 2 } { \sqrt { 2 } } < 2 ( 1 + b ) .\tag{39}
$$

Conversely, $| \lambda _ { \pm } | < 1$ requires $b = \lambda _ { + } \lambda _ { - } < 1$ $\mathbf { A } { \boldsymbol { \mathrm { t } } } \ b \ = \ 1$ , the fixed points still exist and the same bound gives $0 < ( n - 2 ) ( a - 1 - b ) < 4$ , placing the conjugate roots on the unit circle. Immediately beyond this boundary, at least one root has modulus greater than one.

Thus, for $b < 1$ , trajectories attracted to either fixed point satisfy $S _ { t }  S _ { * } < 1$ and eventually cease crossing the EoS threshold. The loss of strict linear stability occurs at $b = 1$ , giving $\beta _ { 2 } = \beta _ { 1 } ^ { 2 ( n - 2 ) / n }$ □

## A.3 Proof of Lemma 5

Proof. The generalized Gauss-Newton decomposition Schraudolph (2002) gives

$$
\nabla _ { \theta } ^ { 2 } \ell _ { i } = \underbrace { J _ { i } ^ { \top } C _ { i } J _ { i } } _ { G _ { i } } + \underbrace { \sum _ { k } ( p _ { i k } - y _ { i k } ) \nabla _ { \theta } ^ { 2 } z _ { i k } } _ { B . } , \qquad C _ { i } = \mathrm { d i a g } ( p _ { i } ) - p _ { i } p _ { i } ^ { \top } .\tag{40}
$$

Let $c _ { i }$ be the correct class and $\varepsilon _ { i } = 1 - p _ { i , c _ { i } }$ . Since $C _ { i } \succeq 0$

$$
\| C _ { i } \| _ { 2 } \leq \mathrm { t r } C _ { i } = 1 - \| p _ { i } \| _ { 2 } ^ { 2 } \leq 1 - ( 1 - \varepsilon _ { i } ) ^ { 2 } \leq 2 \varepsilon _ { i } , \qquad \| p _ { i } - y _ { i } \| _ { 1 } = 2 \varepsilon _ { i } .\tag{41}
$$

Writing $B _ { 1 , i } = \| J _ { i } \| _ { 2 }$ and $B _ { 2 , i } = \operatorname* { m a x } _ { k } \| \nabla _ { \theta } ^ { 2 } z _ { i k } \| _ { 2 }$ , submultiplicativity and the triangle inequality yield

$$
\| G _ { i } \| _ { 2 } \leq 2 \varepsilon _ { i } B _ { 1 , i } ^ { 2 } , \qquad \| R _ { i } \| _ { 2 } \leq 2 \varepsilon _ { i } B _ { 2 , i } , \qquad \| \nabla _ { \theta } ^ { 2 } L \| _ { 2 } \leq \frac { 2 } { N } \sum _ { i } \varepsilon _ { i } ( B _ { 1 , i } ^ { 2 } + B _ { 2 , i } ) \longrightarrow 0 .\tag{42}
$$

## A.4 Proof of Theorem 7

Proof. The ratio $L _ { \mathrm { c o n f } } ( s ) / L _ { \mathrm { c o n f } } ( 0 )$ is the moment-generating function of the CE-weighted slope distribution. Replacing this distribution by the Gaussian model gives

$$
\frac { L _ { \mathrm { c o n f } } ( s ) } { L _ { \mathrm { c o n f } } ( 0 ) } = \mathbb { E } _ { w } [ e ^ { a s } ] = \exp \left( \mu _ { a } s + \frac { \sigma _ { a } ^ { 2 } s ^ { 2 } } { 2 } \right) .\tag{43}
$$

Because $s = 0$ is the local minimum of the slice,

$$
0 = \frac { L _ { \mathrm { c o n f } } ^ { \prime } ( 0 ) } { L _ { \mathrm { c o n f } } ( 0 ) } = \mathbb { E } _ { w } [ a ] = \mu _ { a } .\tag{44}
$$

Therefore, with $u = \sigma _ { a } s ,$

$$
\frac { \Delta L _ { \mathrm { c o n f } } ( s ) } { L _ { \mathrm { c o n f } } ( 0 ) } = e ^ { u ^ { 2 } / 2 } - 1 .\tag{45}
$$

Define the effective log-log exponent by

$$
n _ { \mathrm { e f f } } ( s ) = \frac { \mathrm { d } \log \Delta L _ { \mathrm { c o n f } } ( s ) } { \mathrm { d } \log | s | } .\tag{46}
$$

The Gaussian model gives

$$
n _ { \mathrm { e f f } } ( u ) = \frac { u ^ { 2 } e ^ { u ^ { 2 } / 2 } } { e ^ { u ^ { 2 } / 2 } - 1 } .\tag{47}
$$

For $| u | \ll 1$

$$
n _ { \mathrm { e f f } } ( u ) = 2 + \frac { u ^ { 2 } } { 2 } + O ( u ^ { 4 } ) .\tag{48}
$$

Hence the loss is quadratic for $| u | \ll 1$ and departs from the quadratic form when $| u | = O ( 1 )$ . Since $u = \sigma _ { a } s ,$ the quadratic-core scale is $s _ { \mathrm { c o r e } } = \stackrel { } { O } ( \stackrel { . } { 1 } / \sigma _ { a } )$ . Defining the core boundary by $n _ { \mathrm { e f f } } = 2 . 5$ gives $| u _ { \mathrm { c o r e } } | \simeq 0 . 9 6 4$ , or

$$
| s _ { \mathrm { c o r e } } | \simeq \frac { 0 . 9 6 4 } { \sigma _ { a } } .\tag{49}
$$

## B Full-model experiments

## B.1 Experimental setup for the six models

All six beta-plane scans use a fixed random seed of 42 and AdamW. The beta values are varied across the scan and are therefore omitted from the setup below. The training horizon is adapted for each task: each run is inspected at successive stages, and unresolved points are resumed at a longer horizon when necessary. The maximum horizon reported here is the largest extension used or permitted for the corresponding final panel, rather than a fixed number of steps applied to every beta setting. Parameter counts refer to trainable model parameters and exclude optimizer state.

## B.1.1 Modular division

Model. The modular-division model is a one-layer causal Transformer with vocabulary size 54, model dimension $d _ { \mathrm { m o d e l } } = 1 2 8$ , four attention heads, head dimension $d _ { \mathrm { h e a d } } = 3 2$ , and an MLP dimension of 512. The sequence length is three, corresponding to the prompt $[ x , y , = ]$ The model uses ReLU activations and no LayerNorm. The input embedding, positional embedding, attention, MLP, and output unembedding together contain 211,456 trainable parameters.

Dataset. The task is $x y ^ { - 1 }$ mod 53, with $x \in \{ 0 , \ldots , 5 2 \}$ and $y \in \{ 1 , \ldots , 5 2 \}$ . The complete dataset contains 2,756 examples, which are randomly split into 1,378 training and $^ { 1 , 3 7 8 }$ test examples using a 50% training fraction.

Optimizer. Training uses cross-entropy loss, AdamW with learning rate $1 \dot { 0 } ^ { - 3 }$ , weight decay 0.5, and Adam epsilon $1 0 ^ { - 8 }$ . The training loader uses batch size 512, and the learning rate is linearly warmed up during the first 100 optimizer steps.

Training steps. The final scan starts from 8,000 optimizer steps and adaptively extends unresolved boundary candidates, with a maximum horizon of 30,000 steps.

## B.1.2 Modular addition

Model. The modular-addition model uses the same one-layer Transformer architecture as modular division: vocabulary size 54, $d _ { \mathrm { m o d e l } } = 1 2 8$ , four heads with $d _ { \mathrm { h e a d } } = 3 2$ , MLP dimension 512, sequence length three, ReLU activations, no LayerNorm, and 211,456 trainable parameters.

Dataset. The task is $( x + y )$ mod 53, with $x , y \in \{ 0 , \ldots , 5 2 \}$ . The complete dataset contains 2,809 examples, randomly split into 1,404 training and 1,405 test examples using a 50% training fraction.

Optimizer. Training uses cross-entropy loss, AdamW with learning rate $1 0 ^ { - 3 }$ , weight decay 0.5, and Adam epsilon $1 0 ^ { - 8 }$ . The batch size is 512, with the same linear 100-step learning-rate warmup as modular division.

Training steps. The final scan starts from 8,000 optimizer steps and adaptively extends unresolved boundary candidates up to a maximum of 30,000 steps.

## B.1.3 Tiny Shakespeare

Model. The Tiny Shakespeare model is a compact causal character Transformer. It has a 65-character vocabulary, model dimension 48, four attention heads, two Transformer blocks, and an MLP dimension of 192 in each block. The character and positional embeddings are 48-dimensional, the positional context length is 64, and the output classifier has no bias. The model uses GELU activations, zero attention dropout, and no LayerNorm. It has $^ { 6 5 , 4 7 2 }$ trainable parameters.

Dataset. The data are drawn from the first 90% of data/tinyshakespeare/input . txt, giving a training corpus of 1,003,854 characters. Each run uses the same fixed random set of 448 character windows, each with context length 64. Only the final eight target positions in each window contribute to the loss, while the preceding 56 target positions are masked. The loss is cross-entropy evaluated in float64.

Optimizer. Training uses full-batch AdamW with learning rate $1 0 ^ { - 3 }$ , weight decay 0.1, and Adam epsilon $1 0 ^ { - 8 }$ , with no learning-rate scheduler.

Training steps. The base trajectory is 12,000 optimizer steps. Unresolved points are adaptively extended through 30,000 and 50,000 steps, with selected points extended to a maximum of 100,000 steps.

## B.1.4 CIFAR-10 MLP

Model. The CIFAR-10 MLP receives a $3 \times 3 2 \times 3 2$ image, flattened to 3,072 input features. It contains five hidden linear layers of width 512, with a ReLU after each hidden layer, followed by a linear 512-to-10 classifier. The model has 2,629,130 trainable parameters and does not use dropout or normalization layers.

Dataset. For each run, 200 images are randomly selected from the CIFAR-10 training set and 200 images from the CIFAR-10 test set, using seed 42. The images are converted with ToTensor() only, with no data augmentation or additional normalization.

Optimizer. Optimization uses the 200 training images in full batch with cross-entropy loss and AdamW, using learning rate $1 0 ^ { - 3 }$ , weight decay 0.5, and Adam epsilon $1 0 ^ { - 8 }$

Training steps. The final scan starts from 4,000 optimizer updates per run and adaptively extends unresolved points to a maximum of 16,000 updates.

## B.1.5 CIFAR-10 CNN

Model. The CIFAR-10 CNN is a VGG11-style convolutional network adapted to $3 2 \times 3 2$ inputs. Its eight convolutional layers have channel widths $3  6 4  1 2 8  2 5 6  2 5 6  5 1 2  5 1 2  5 1 2  5 1 2$ , with ReLU activations and five $2 \times 2$ max-pooling operations. The classifier maps the final 512 features through two 512-unit linear layers to 10 classes. Each of the two intermediate classifier layers is followed by ReLU and dropout with probability 0.5. The model has 9,750,922 trainable parameters.

Dataset. The data protocol is the same as for the CIFAR-10 MLP: 200 randomly selected CIFAR-10 training images and 200 randomly selected test images, ToTensor() only, and no augmentation or additional normalization.

Optimizer. Optimization is full-batch cross-entropy training with AdamW, learning rate $1 0 ^ { - 3 }$ , weight decay 0.5, and Adam epsilon $1 0 ^ { - 8 }$

Training steps. The base scan uses 8,000 optimizer updates per run. Unresolved points are adaptively extended to 16,000 or 30,000 updates, with a maximum horizon of 30,000 updates.

## B.1.6 MNIST autoencoder

Model. The MNIST model is a compact binary convolutional autoencoder. The encoder consists of a 1 → 8 convolution and an $8  1 6$ convolution, both with kernel size $^ { 3 , }$ stride 2, and ReLU activations, reducing a $2 8 \times 2 8$ image to a $1 6 \times 7 \times 7$ representation. A linear layer maps this representation to a 32-dimensional latent vector, and a second linear layer maps it back to $1 6 \times 7 \times 7$ The decoder uses a 16 → 8 transposed convolution followed by ReLU and an $8  1$ transposed convolution that produces the output logits. Both transposed convolutions use kernel size 4 and stride 2. The autoencoder has 54,425 trainable parameters.

Dataset. Each run uses 64 randomly selected images from the MNIST training split, with seed 42. The grayscale inputs are binarized at threshold 0.5, and the reconstruction target is the binarized input itself.

Optimizer. Training uses binary cross-entropy with logits and full-batch AdamW with learning rate $2 \times 1 0 ^ { - 3 }$ , weight decay 0.2, and Adam epsilon $1 \dot { 0 } ^ { - 1 2 }$ . No learning-rate scheduler is used.

Training steps. The initial phase scan checks successive horizons, and unresolved points are adaptively resumed through a maximum of 200,000 optimizer steps.

## B.2 Period-detection algorithm

We estimate the macroscopic period of loss oscillations from loss traces sampled once per optimizer step. We use the training loss for modular arithmetic, Tiny Shakespeare, and MNIST reconstruction, and the test loss for the two CIFAR-10 models. For a trajectory of S steps, we discard the first 20% to reduce the influence of the initial transient and analyze the remaining segment. Segments containing fewer than 100 samples are not assigned a period.

To suppress rapid fluctuations while accommodating the large dynamic range of the loss, we first transform the signal as $u _ { t } \overset { \cdot \ b { \cdot } } { = } \log _ { 1 0 } \bigl ( \dot { L } _ { t } + 1 0 ^ { - 1 2 } \bigr )$ and smooth it with a third-order Butterworth low-pass filter applied forward and backward. The cutoff frequency is 0.05 cycles per step for all models. We then restore the linear loss scale, $\widetilde { L } _ { t } = 1 0 ^ { \widetilde { u } _ { t } }$ , and subtract its mean. We estimate the power spectrum $P ( f )$ of this signal using Welch's method.

We identify local spectral peaks and retain only those satisfying both

$$
\operatorname { p r o m i n e n c e } ( f _ { k } ) \geq 0 . 0 2 \operatorname* { m a x } _ { f } P ( f ) , \qquad f _ { k } > \frac { 1 } { T _ { \operatorname* { m a x } } } , \qquad T _ { \operatorname* { m a x } } = S .\tag{50}
$$

Here, prominence measures how strongly a spectral peak stands out from its surrounding spectral background, and $T _ { \mathrm { m a x } }$ is the length of the complete trajectory before transient removal. Among the accepted peaks, we select the one with the largest spectral power and define the detected period as

$$
f _ { * } = \underset { f _ { k } \in \mathcal C } { \arg \operatorname* { m a x } } P ( f _ { k } ) , \qquad T = \frac { 1 } { f _ { * } } ,\tag{51}
$$

where C denotes the set of accepted peaks. The resulting $T$ is measured in optimizer steps. If no peak satisfies both criteria, the trajectory is marked as having no detected period within the observation window. Unresolved scan points are adaptively extended as described in Appendix B.1, and the period is re-estimated from the longer trace. For all

models except the CIFAR-10 MLP, we additionally require the minimum training loss in the final 10% of the complete trajectory to be at most 0.05. We exempt the CIFAR-10 MLP because dead ReLU units can prevent its training loss from reaching this threshold even when the trajectory remains relevant to the phase-boundary analysis.

## C One-dimensional toy-model controls

## C.1 Sensitivity of the Adam boundary for $L ( x ) = k x ^ { 2 } / 2$

To test the sensitivity of the quadratic toy-model boundary to non-beta Adam parameters, we use a common grid spanning $\delta = 1 - \dot { \beta _ { 1 } } \in [ 1 0 ^ { - \bar { 2 } } , 1 0 ^ { - 1 } ]$ and $\gamma = 1 - \beta _ { 2 } \in [ \bar { 5 } \times 1 0 ^ { - 4 } , 1 0 ^ { - 1 } ]$ . Each control changes one of $k , \eta ,$ or € relative to the baseline in Figure $6 ( \mathrm { a } ) .$ , while holding the other two fixed. As in the main quadratic experiment, each beta pair is trained for 250,000 updates, the first 125,000 updates are discarded, and a period is assigned when at least three upward crossings of the local stability threshold are detected. We fit the resulting boundary to $\gamma = C \delta ^ { p }$ . Figure 6 shows the phase diagrams, and Table 2 reports the fits.

![](images/e37ddbdf62e4c01d10782467b4ce8e3612a97822441b037b663b810d332c6562.jpg)  
Figure 6: Sensitivity of the quadratic-loss boundary to k, $\eta ,$ and €. The phase diagrams show $L ( x ) = k x ^ { 2 } / 2$ in the $( 1 - \beta _ { 1 } , 1 - \beta _ { 2 } )$ plane, with dashed fits $\gamma = C \delta ^ { p }$

Table 2: Power-law fits for the quadratic-loss parameter controls in Figure 6.
<table><tr><td>Panel</td><td>Parameters</td><td>Valid points</td><td> $p$ </td><td>C</td></tr><tr><td>(a)</td><td> $k = 1 , \eta = 0 . 1 , \epsilon = 1 0 ^ { - 8 }$ </td><td>351</td><td>3.095</td><td>17.70</td></tr><tr><td>(b)</td><td> $k = 1 0 , \eta = 0 . 1 , \epsilon = 1 0 ^ { - 8 }$ </td><td>353</td><td>3.064</td><td>16.30</td></tr><tr><td>(c)</td><td> $k = 1 , \eta = 0 . 0 1 , \epsilon = 1 0 ^ { - 8 }$ </td><td>351</td><td>3.210</td><td>25.00</td></tr><tr><td>(d)</td><td> $k = 1 , \eta = 0 . 1 , \epsilon = 1 0 ^ { - 4 }$ </td><td>365</td><td>3.009</td><td>15.02</td></tr></table>

Changing k by one order of magnitude leaves the fitted boundary nearly unchanged, and changing η preserves the approximately cubic scaling while producing only a moderate shift in its prefactor. The robustness to € is conditional: € must remain small compared with $\sqrt { \hat { v } _ { t } }$ over the part of the trajectory that determines the boundary. If ε is too large, it dominates the denominator $\sqrt { \hat { v } _ { t } } + \epsilon$ and suppresses the late-time decay of the effective second-moment scale, so the expected boundary scaling is no longer obtained.

## C.2 Effect of weight decay in the one-dimensional toy model

Weight decay contributes the parameter update $\Delta \theta _ { \mathrm { w d } } = - \eta \lambda _ { \mathrm { w d } } \theta$ . In a neural network, the parameter origin $\theta = 0$ and a local minimum of the task loss are generally distinct. Moreover, the weight-decay update and the Adam-preconditioned gradient direction tend to have small overlaps in high dimensions Cai et al. (2013). At the modular-division pre-spike point used in Figures 1(c) and ${ \mathfrak { I } } ( { \mathfrak { a } } )$ , their signed Euclidean projection is

$$
\left. \Delta \theta _ { \mathrm { w d } } , \frac { \hat { d } _ { t } } { \| \hat { d } _ { t } \| _ { 2 } } \right. = + 9 . 4 5 0 \times 1 0 ^ { - 3 } .\tag{52}
$$

This geometry cannot be represented by the centered one-dimensional loss $L ( x ) = k | x | ^ { n } / n \colon$ in one dimension, weight decay and the loss gradient can only be parallel or antiparallel, and decay toward the origin also points toward the loss minimum at $x = 0$ . We therefore shift the minimum and use

$$
L ( x ) = { \frac { k } { n } } | x - x _ { * } | ^ { n } , \qquad x _ { * } = 1 .\tag{53}
$$

We choose the small value $\lambda _ { \mathrm { w d } } = 1 0 ^ { - 3 }$ so that the projected decay term perturbs rather than dominates the Adam dynamics. We set $x _ { 0 } = 2$ , preserving the unit initial displacement $x _ { 0 } - x _ { * } = 1$ , and otherwise retain the main toy-model settings: $k = 1 , \eta = 0 . 1 , \bar { \epsilon } = 1 0 ^ { - 3 0 }$ , 250,000 updates, and a 125,000-update burn-in.

![](images/c1644c06255a7f78e2ed3332c30ecf38d1da5bf47b235e8cd9eb666571b81c95.jpg)

![](images/fa4e23cdba74ec9a719892091cc9e066d641d98f8aa41a1ed60eeb3b9df4340a.jpg)

![](images/3f809dcbc3f1fa0bc77d23076980096acca87165e1113ec9b9e0e074df83386e.jpg)

![](images/8344ab28273efd455627b2044b847bbcfd964759f38abd3a0618b969a02b515c.jpg)

![](images/67026fd600726e2f372b8eafd5602506a43261130ddaa5ccca851aeed9727ca9.jpg)

![](images/217a13cfd6cf3cae604a36b4a121d1dfbdbceed7f3d18185b5331b13dfe5f1a5.jpg)

![](images/a29f23db1689f9b9031c10341d719bc001cefe81a6beee2d900f45fab98afee6.jpg)  
Figure 7: Effect of weak decoupled weight decay on the shifted one-dimensional losses $L ( x ) = | x - 1 | ^ { n } / n .$ AdamW uses $\eta = 0 . 1 , \lambda _ { \mathrm { w d } } = 1 0 ^ { - 3 } , \epsilon = 1 0 ^ { - 3 0 }$ , and $x _ { 0 } = 2$ . Each beta pair is run for 250,000 updates, with the first 125,000 discarded. Color indicates the period estimated from repeated upward crossings of the local stability threshold. The solid cyan and dashed pink lines show $\gamma = 1 2 ( n - 2 ) \delta / ( 3 n - 4 )$ and $\gamma = 2 ( n - 2 ) \delta / n ,$ , respectively. Relative to Figure 3, the left boundary shifts slightly upward, periodic trajectories extend below the reference right boundary, and the right edge of the wedge is unresolved.

The weak shifted decay therefore preserves the approximately unit-slope left boundary while modestly enlarging the periodic region on that side. However, the lower/right boundary that closes the no-decay wedge disappears over the scanned range. Thus, the near-linear left-boundary scaling is robust to this weak projected weight-decay force, whereas the two-sided wedge is not.

## D Finite-scale landscape analysis

## D.1 Locating the pre-spike points and progressive sharpening at finite scales

We define a temporal valley as a low-loss segment between macroscopic high-loss excursions. After discarding the initial training transient, the loss trace is partitioned into successive valleys using the task's loss scale: a valley must enter the low-loss regime and must subsequently be closed by a new high-loss excursion. The last valley in a finite trace is marked as right-censored if no closing excursion has yet been observed and is not used in the exponent fit.

Within each completed valley $V _ { j }$ , the pre-spike point is the single checkpoint

$$
t _ { \mathrm { p r e } , j } = \arg \operatorname* { m i n } _ { t \in V _ { j } } L _ { t } .\tag{54}
$$

Thus each completed valley contributes at most one event, preventing long valleys or densely saved portions of a trace from receiving disproportionate weight. The pre-spike point is selected solely from the temporal loss trace. No directional-slope or landscape-shape criterion is used to move the anchor to a nearby checkpoint. When the exact pre-spike point was not retained as a checkpoint during the original run, deterministic replay is used to reconstruct the model and optimizer state at $t _ { \mathrm { p r e } , j }$

Across the examined models and β settings, the selected points occur immediately before a spike. Figure 8 illustrates the selection for three modular-division trajectories.

(c) peak  
local minimum of loss curve— preconditioned gradient norm $\| P ^ { 1 / 2 } g \|$  
![](images/4b830c8352e3af26d258a623471039c029c14d984575b5964bc7e262185719ab.jpg)

![](images/5a16440df041b318b5948d3dddb708b503d135406c6f05180362cbcb38876974.jpg)

![](images/e3d016dc3e50af71bee2411a3d6edc1e3599346e9e156e50349f7ecad7065f87.jpg)  
Figure 8: Training-loss traces for the modular-division Transformer at $( \beta _ { 1 } , \beta _ { 2 } ) \ = \ ( 0 . 9 , 0 . 9 9 9 ) , \ ( 0 . 9 , 0 . 9 9 )$ , and (0.9, 0.95), from left to right. Diamonds mark the local minima of the loss curve, which are used directly as the pre-spike points.

The pre-spike point $t _ { \mathrm { p r e } }$ and the spatial minimum s play different roles. The former selects the pre-spike model state. After freezing that state and the Adam preconditioner, we define the full-model directional slice

$$
\phi _ { t } ( s ) = L ( \theta _ { t } + s \hat { d } _ { t } ) , \qquad \hat { d } _ { t } = \frac { D _ { t } g _ { t } } { \lVert D _ { t } ^ { 1 / 2 } g _ { t } \rVert _ { 2 } } ,\tag{55}
$$

and refine its one-dimensional minimum $s _ { * } .$ The direction combines the current gradient with the second-moment preconditioner and captures the geometry relevant to the Adam update. The exponent fit is centered at $s _ { * } .$ so the procedure does not assume that the pre-spike point itself lies exactly at a stationary point of the frozen directional slice. To compare the slice with the dynamics, we freeze $\hat { d } _ { t }$ at step t, project the ten updates ending at t onto $\hat { d } _ { t } .$ , and define the typical update scale $\left| s _ { \mathrm { u p d } } \right|$ as the median of their absolute projections.

We classify each recentered slice according to Definition 4. We use the modular-division Transformer at $( \beta _ { 1 } , \beta _ { 2 } ) =$ (0.9, 0.999) as a representative example.

![](images/a80ae1cf33ead6e7b20ff3b61e6094a1a9cfcdb0ca4cbd0206c415a14374d884.jpg)

![](images/c7558eccfa148a3236bf280f2dca2d2490a75ae02a5bdf44f34c6bb08e1faa77.jpg)

![](images/f2167c6e2df00e007e3d403a54b894e885906170951e688e8b13d6442852259d.jpg)  
Figure 9: Evolution of the directional loss slice through a modular-division spike at $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ . The three panels show a quiet point (step 28,600, $L _ { 0 } \simeq 6 . 8 \times \mathrm { \bar { 1 0 ^ { - 6 } } } ) .$ , the pre-spike point (step $3 7 , 2 1 0 , \dot { L _ { 0 } } \simeq 8 . 9 \times \mathrm { 1 0 ^ { - 8 } } )$ , and the spike peak (step $\bar { 3 7 } , 2 2 0 , \bar { L _ { 0 } } \simeq 2 . 0 )$ , respectively. The green band and dashed edges indicate the magnitude of a typical Adam-preconditioned update: $\dot { \vert s _ { \mathrm { u p d } } \vert } \simeq 5 . 1 \dot { \times } 1 0 ^ { - 7 } , 6 . 3 \times 1 0 ^ { - 6 }$ , and $1 . 4 \times \bar { 1 0 } ^ { - 6 }$ in the three panels. As the system approaches the pre-spike state, the flat core contracts to the update scale. This phenomenon can be viewed as progressive sharpening at finite scales. At the peak, the directional wall is encountered and the slice becomes strongly nonquadratic.

We define the core-wall transition as the first displacement at which

$$
n _ { \mathrm { e f f } } ( s ) = \frac { \mathrm { d } \log \Delta L } { \mathrm { d } \log | s - s _ { * } | }\tag{56}
$$

reaches 2.5. During the quiet phase in Figure 9, the left and right transition distances are approximately $7 . 0 \times 1 0 ^ { - 4 }$ and $8 . 0 \times 1 0 ^ { - 4 }$ , more than three orders of magnitude larger than $| s _ { \mathrm { u p d } } | \simeq 5 . 1 \times 1 0 ^ { - 7 }$ . At the pre-spike point, these distances

contract to $2 . 1 6 \times 1 0 ^ { - 6 }$ and $3 . 5 5 \times 1 0 ^ { - 6 }$ , while $\lvert s _ { \mathrm { u p d } } \rvert$ grows to $6 . 3 \times 1 0 ^ { - 6 }$ . A typical update therefore crosses the quadratic core and reaches the superquadratic wall. At the spike peak, the reference loss is approximately 2.0, and the strongly asymmetric slice no longer exhibits a core-wall landscape.

## D.2 Fitting and aggregating the wall exponent

Every model–task setting is evaluated at the same six reference beta pairs,

$$
\begin{array} { r } { ( \beta _ { 1 } , \beta _ { 2 } ) \in \{ ( 0 . 9 , 0 . 9 ) , ( 0 . 9 , 0 . 9 5 ) , ( 0 . 9 , 0 . 9 9 ) , ( 0 . 9 , 0 . 9 9 9 ) , ( 0 . 9 9 , 0 . 9 9 ) , ( 0 . 9 9 , 0 . 9 9 ) , ( 0 . 9 9 , 0 . 9 9 9 ) \} . } \end{array}\tag{57}
$$

For each usable pre-spike slice, define the displacement and excess loss on its left and right branches by

$$
r = | s - s _ { * } | , \quad \quad \Delta L = \phi _ { t } ( s ) - \phi _ { t } ( s _ { * } ) ,\tag{58}
$$

and transform the positive, resolved samples to $u = \log r$ and $y = \log \Delta L$ The two branches are retained separately so that basin asymmetry is absorbed by branch-specific parameters rather than by averaging the profiles before fitting.

Each event-side branch is first fit with a continuous three-segment line,

$$
y ( u ) = c + n _ { \mathrm { c o r e } } ( u - b _ { 1 } ) + ( n _ { \mathrm { w a l l } } - n _ { \mathrm { c o r e } } ) [ u - b _ { 1 } ] _ { + } + ( n _ { \mathrm { r o l l } } - n _ { \mathrm { w a l l } } ) [ u - b _ { 2 } ] _ { + } , \qquad [ z ] _ { + } = \operatorname* { m a x } ( z , 0 ) .\tag{59}
$$

The breakpoints $b _ { 1 } ~ < ~ b _ { 2 }$ separate the quadratic core, steep wall, and outer rollover. Candidate breakpoints are enumerated, with at least five samples required in each regime, and the pair with the smallest total squared residual in log space is selected subject to

$$
n _ { \mathrm { w a l l } } > n _ { \mathrm { c o r e } } , \qquad n _ { \mathrm { w a l l } } > n _ { \mathrm { r o l l } } .\tag{60}
$$

These inequalities operationally identify the wall as the middle steepening region. They also prevent the flatter outer rollover from being folded into the phase-relevant wall exponent.

After determining $b _ { 1 }$ and $b _ { 2 }$ separately for every event and side, all branches within one beta setting are refit jointly using only samples through $b _ { 2 } \mathrm { : }$

$$
y _ { e , \pm } ( u ) = c _ { e , \pm } + n _ { \mathrm { c o r e } } \operatorname* { m i n } ( u - b _ { 1 , e , \pm } , 0 ) + n _ { \mathrm { w a l l } } \operatorname* { m a x } ( u - b _ { 1 , e , \pm } , 0 ) , \qquad u \leq b _ { 2 , e , \pm } .\tag{61}
$$

The intercept and breakpoints remain specific to each event-side branch, while $n _ { \mathrm { c o r e } }$ and $n _ { \mathrm { w a l l } }$ are shared. This produces one beta-level exponent $n _ { \mathrm { w a l l } , k }$ from all usable pre-spike events at the k-th beta pair. Uncertainty is estimated with 2,000 event-level bootstrap resamples, with the left and right branches of each event always resampled together.

Finally, the exponent shown in the main-text table is the unweighted arithmetic mean of the six beta-level estimates,

$$
n _ { \mathrm { w a l l } } ^ { ( \mathrm { t a s k } ) } = \frac { 1 } { 6 } \sum _ { k = 1 } ^ { 6 } n _ { \mathrm { w a l l } , k } .\tag{62}
$$

Each beta pair therefore receives the same weight even when the number of usable completed valleys differs. The boundary coefficient is computed only after this averaging step,

$$
C _ { L } = \frac { 1 2 \big ( n _ { \mathrm { w a l l } } ^ { ( \mathrm { t a s k } ) } - 2 \big ) } { 3 n _ { \mathrm { w a l l } } ^ { ( \mathrm { t a s k } ) } - 4 } ,\tag{63}
$$

rather than by averaging six separately transformed coefficients.