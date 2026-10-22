# Internal Guidance as a Sampling Method for Tilted Difusion Models

Curtis McDonald<sup>1</sup> and Peter Bartlett<sup>1,2</sup>

<sup>1</sup>Simons Institute, University of California Berkeley <sup>2</sup>Google DeepMind

Oct 7, 2026

## Abstract

Reward guidance methods sample an exponential tilt of a base density using a difusion model trained with only samples from the base distribution. Existing methods use what we call external guidance: additive terms in the drift or multiplicative terms on the transition probabilities. We propose internal guidance which leaves the output of the base denoiser unmodified and instead changes the point at which it is evaluated. For discrete distributions embedded in the hypercube $\{ - 1 , 1 \} ^ { d }$ , the denoiser of any linearly tilted density is exactly the base denoiser evaluated at a shifted and rescaled input. In practice the base denoiser is a neural network trained on noised samples of the base distribution. The required evaluation points for sampling the tilted density would be out-of-distribution for basic noising of the base distribution. We modify the training algorithm with random diagonal rescalings of the data, so that the denoiser network learns both in- and out-of-distribution evaluations of the denoiser relative to the base distribution. A single denoiser network samples binary reward tilts, class labels, and infilling with no fine-tuning and no external guidance term.<sup>1</sup>

## 1 Introduction

Difusion models designed via Gaussian noising have become the dominant generative algorithm for continuous distributions [29, 13, 32], and continue to show promise for discrete targets as well [7, 16, 35]. The original theory of difusion models was established for unconditional generation [31, 32]: given access to an initial pool of samples from a distribution π, we train a model to sample from π. As the field has matured, interest has turned towards conditional generation and tilting. Given access to only a pool of samples from π, and a reward function r, we want to train a sampling algorithm to sample from an exponential tilt by the reward function for some strength τ with density $\tilde { \pi } ( x ) \propto \pi ( x ) e ^ { \tau r ( x ) }$

Existing methods utilize what the authors refer to as external guidance, implementing an additive vector field called a guidance term to modify SDE paths at sampling time [10, 28, 11], or a density ratio to reweight Markov chain conditional probabilities [33, 24, 17, 12]. Methods such as classier-fee guidance [14] implement ad hoc and closed-form guidance terms, but are known to structurally sample the wrong invariant measure and introduce sampling artifacts such as corrupted samples and mode collapse [8, 4]. Exact guidance schemes derive the analytically correct guidance terms, but result in an intractable object that must either be learned in an expensive neural network training run, or approximated by a tractable analog such as an empirical average that introduces compounding errors at sample time.

A recent line of work [15, 21, 20, 22] has opened up a new method for reward-tilted sampling we call internal guidance. Rather than modify the output of the denoiser of the base distribution with additive or multiplicative terms outside of the object, we realize we can instead shift our evaluation point inside the existing base denoiser. Recent work of Moitra et al. [21, Lemma 3.3] has shown for an internal mapping function H we call a symmetry map, the evaluation of the denoiser of a linear tilted density is an evaluation of the base denoiser at a modified input

$$
G _ { \mathrm { t i l t } } ( u , a ) = G _ { \mathrm { b a s e } } ( \tilde { u } = H _ { 1 } ( u , a ) , \tilde { a } = H _ { 2 } ( u , a ) ) .
$$

These symmetry mappings are closed-form objects involving shifting and rescaling inputs, derived from the interactions of linear tilts with the quadratic log-density terms introduced via Gaussian convolution.

Internal guidance has clear benefits. There is no intractable external term to approximate, and the symmetry maps H are all explicit. A single denoising difusion model, trained on unconditional sampling of its input distribution, is able to sample an entire family of tilted densities at inference time with no additional reinforcement learning, fine tuning, or external guidance required.

The catch with this method is one of coverage. If we had access to the oracle denoiser G we could implement such a method with no additional error over base sampling. However, we instead have a network $g _ { \theta }$ trained to approximate G. This network is only accurate in an $L ^ { 2 }$ sense along the sequence of marginal densities produced by noising the base distribution, which we call the training manifold. The equivalent points $( \tilde { u } , \tilde { a } )$ induced by the symmetry map represent an out-ofdistribution query from the perspective of the base model denoiser, and we have no control on the accuracy of our trained network at this point.

The relationship of the symmetry map can be captured by an equivalence class: for a tilted density π˜ and an evaluation point $( u , a )$ , the equivalence class is all points $( \tilde { u } , \tilde { a } )$ where the base denoiser evaluated at $( \tilde { u } , \tilde { a } )$ is equal to the tilted denoiser at $( u , a )$ . The work of Moitra et al. [21] can be interpreted as showing this equivalence class exists and is non-empty, and established one symmetry map which creates the class. However, this symmetry alone is not enough to force overlap between the equivalence class of points produced by tilted sampling and the high probability region of the base distribution training manifold the denoiser network was trained on.

To have equivalent points that live near the training manifold, we need to introduce additional symmetry relationships that expand the equivalence class. We restrict our study to discrete distributions embedded in the hypercube $\{ - 1 , 1 \} ^ { d }$ , noised under continuous Gaussian convolution, and a stochastic localization algorithm to implement the reverse sampling and produce discrete outputs. The benefit of embedding in the hypercube is all discrete points are constant norm $\| x \| ^ { 2 } = d .$ The conditional densities which define the denoiser are quadratic tilts to the base density which introduce linear and quadratic terms to the log-density. By the constant norm property, all quadratic terms normalize out and the denoiser is a pure linear tilt to the base density. This opens up additional symmetries that expand the equivalence class and allow modification to the training manifold. With these additional symmetries, we train a denoiser network that can be evaluated at points which are out-of-distribution from the pure noising of the base distribution, but lie in the equivalence class of points seen at sampling time. We visualize a sketch of this idea in Figure 1. Our contributions are threefold:

1. Conditional generation tasks such as infilling and reward tilting by binary rewards can be cast as linear tilts in an extended state space by promoting the reward coordinate to an additional random variable noised and denoised when sampling.

![](images/7b34f6da8e54b840226bf946b35e8875433335afe98044d6cbf72c7514b8e8cc.jpg)  
Figure 1: Sketch of denoiser symmetry. The training manifold is where the denoiser network has learned the base denoiser. The sampling manifold is the typical vectors that will be encountered when sampling a tilted density. The equivalence relationships can find a point with equal evaluation of the base denoiser near the training manifold, not all equivalence relationships map back to the region of the training manifold.

2. We establish the theory of denoiser symmetry, the evaluation of the denoiser of a linear tilt to a base density can be recast as an evaluation of the base denoiser at a shifted evaluation point. We establish additional symmetries specific to hypercube embedded discrete distributions, by the removal of quadratic terms in the conditional log-density.

3. We introduce diagonal matrix rescalings to our hypercube embedding during training called Λ focusing, turning an out-of-distribution learning problem into an in-distribution learning problem, and allowing the base denoiser to be evaluated at locations of the typical noising path.

We demonstrate performance on small discrete sampling targets such as a non-linear reward tilt to a small 12 dimension Ising model with ground truth comparison denoiser; generating and solving 4x4, 6x6, and 9x9 Sudoku puzzles; and conditional class generation of MNIST digit images. These problems are small enough to act as a test bed where the quality of our sampling can be compared against exact ground truth values as a diagnostic of efective sampling, with future work to focus on scaling and more complex target distributions for internal guidance.

## 2 Setup

We consider discrete distributions embedded in the hypercube $\mathcal { X }  \{ - 1 , 1 \} ^ { d }$ , noised under Gaussian convolution, and sampled via a reverse process. This form of sampling was considered in Chen et al. [7], referred to as “analog bits”, although the structural results of linear tilts and internal guidance we derive here were not known at the time. Any discrete distribution can be embedded in the hypercube by assigning each element x of the original discrete state space X of size less than $2 ^ { d }$ , a unique embedding vector $e ( x ) \in \{ - 1 , 1 \} ^ { d }$ . This is natural for token generation tasks with a finite library such as language models; constraint satisfaction tasks such as generating and solving Sudoku puzzles; image generation with discrete pixel values such as grayscale MNIST digits; and distributions natively in the hypercube such as Ising models.

Our target density $\pi$ is now restricted to a distribution on the hypercube $\{ - 1 , 1 \} ^ { d }$ . We are interested in sampling two forms of tilts to π. For any vector $\boldsymbol { v } \in \mathbb { R } ^ { d }$ , a linear tilt to $\pi , \pi _ { v } ( x ) \propto$ $\pi ( x ) e ^ { \langle v , x \rangle }$ . For a binary reward function r with outputs $\{ - 1 , 1 \}$ tilts to π by $\tau > 0$ times $r ,$ $q _ { \tau } ( x ) \propto \pi ( x ) e ^ { \tau r ( x ) }$

To unify these two objectives, we promote the reward value $r ( x )$ to a new random variable $x _ { * } = r ( x )$ in an extended $d + 1$ dimensional state space and consider linear tilts in the extended state space. Our original coordinates $x _ { i }$ are already $\{ - 1 , 1 \}$ valued, so from the perspective of our density $x _ { * }$ is simply another $\{ - 1 , 1 \}$ coordinate. It has a deterministic conditional distribution $\pi ( x ^ { * } | x ) = 1 \{ x ^ { * } = r ( x ) \}$ , but there is no issue with having a probability 1 event in a discrete support set. We then define the extended base density with $d + 1$ coordinates where the rewardtilted version of $\pi$ is equivalent to a linear tilt in the extended state space by the one-hot vector $e _ { * } = ( 0 , \cdot \cdot \cdot , 0 , 1 ) \in \mathbb { R } ^ { d + 1 }$

$$
\begin{array} { c } { { \pi ( x , x ^ { * } ) = \pi ( x ) 1 \{ r ( x ) = x ^ { * } \} } } \\ { { \pi ( x ) e ^ { \tau r ( x ) } = \pi ( x , x ^ { * } ) e ^ { \tau \langle e _ { * } , ( x , x ^ { * } ) \rangle } . } } \end{array}
$$

As such, linear tilts by vectors v and tilts by the reward function can both be unified as pure linear tilts in the extended state space. We thus consider π as a distribution on the d+1 dimensional hypercube, with the final coordinate being the reward coordinate.

For values $a , \sigma > 0$ consider a random variable defined by a linear combination of a sample from π with an independent normal random variable

$$
U = a X + \sigma Z , X \sim \pi , Z \sim N ( 0 , I ) .
$$

The denoiser of $\pi$ is the conditional expectation of X given $U .$

$$
F ^ { \mathrm { b a s e } } ( u , a , \sigma ) = E _ { \pi } [ X | u , a , \sigma ] = \sum _ { x } x \frac { \pi ( x ) e ^ { - \frac { 1 } { 2 \sigma ^ { 2 } } \| u - a x \| ^ { 2 } } } { E _ { \pi } [ e ^ { - \frac { 1 } { 2 \sigma ^ { 2 } } \| u - a X \| ^ { 2 } } ] } .
$$

The denoiser defines the drift which implements the reverse SDE sampling algorithm. A special case is the variance preserving (VP) combination, when $a ^ { 2 } + \sigma ^ { 2 } = 1$ , or $\sigma = { \sqrt { 1 - a ^ { 2 } } }$ . In that case, we simply indicate $F ^ { \mathrm { b a s e } } ( u , a )$ and note it is a $\mathrm { V P }$ combination.

In the extended state space, we are really considering the joint noising of the token bit string and the reward coordinate at the same time, $u = ( u ^ { \mathrm { t o k e n } } , u ^ { * } )$

$$
\begin{array} { r l r } & { } & { \binom { U ^ { \mathrm { t o k e n } } } { U _ { * } } = a \left( { x } _ { * } ^ { \mathrm { t o k e n } } \right) + \sqrt { 1 - a ^ { 2 } } \left( \begin{array} { c } { Z } \\ { Z _ { * } } \end{array} \right) } \\ & { } & { a \in [ 0 , 1 ] , X , X * \sim \pi , Z \sim N ( 0 , I ) , Z _ { * } \sim N ( 0 , 1 ) . } \end{array}
$$

Thus our sampler carries a “noisy reward” value in its $d + 1$ coordinate, and does joint denoising of the reward coordinate and the token string at the same time through the reverse difusion.

Denoisers are expectations of quadratic tilts to the base density. By restricting our base density to be supported on the hypercube, all quadratic terms can be dropped from the exponent of the conditional densities that define the denoiser expectation. As such, all denoiser evaluations of π are

expectations of linear tilts to $\pi ,$ and all denoisers of linear tilts to $\pi$ are themselves further linear tilts to this object. One operator defines the denoiser of π and every linear tilt to π: the general mean-tilt operator.

Definition 1. For any vector $v \in \mathbb { R } ^ { d + 1 }$ and any value $a \in [ 0 , 1 ]$ , define VP tilt strength $\textstyle \beta ( a ) = { \frac { a } { 1 - a ^ { 2 } } }$ and define the general mean-tilt operator,

$$
G ( v , a ) = \sum _ { x } x \frac { \pi ( x ) e ^ { \beta ( a ) \langle v , x \rangle } } { \sum _ { \bar { x } } \pi ( \bar { x } ) e ^ { \beta ( a ) \langle v , \bar { x } \rangle } } .
$$

The base denoiser $F ^ { \mathrm { b a s e } }$ is equal to the operator $G .$

## 2.1 Sampling Via Stochastic Localization

Our sampling algorithm will be the stochastic localization path, which is equivalent to a time shifting and rescaling of the marginal densities of score-based difusion. See Montanari [23] and our discussion in Appendix B.2 for further connection. The benefit of stochastic localization is the drift term is exactly the denoiser, rather than being related to the score via Tweedie’s formula [25]. This makes it simpler to construct denoiser-to-denoiser relationships rather than score-to-score relationships.

Definition 2 (Stochastic Localization). For a target density $\pi ,$ initialize a random variable at the origin $X _ { 0 } = 0$ . Then follow the SDE using the denoiser as a drift at a specific time scaling,

$$
\begin{array} { r } { d X _ { t } = F ^ { \pi } ( X _ { t } , t , \sqrt { t } ) d t + d B _ { t } , } \end{array}
$$

then the marginals at time t are equivalent to

$$
X _ { t } = t X + { \sqrt { t } } Z , X \sim \pi , Z \sim N ( 0 , I ) .
$$

For large $T _ { i }$ , the normalized variable $X _ { T } / T$ is equal to the target density with a small normal perturbation of scaling $1 / \sqrt { T }$ . The discrete implementation of this SDE as an algorithm is shown in Algorithm 2.

## 3 Denoiser Symmetries

First, we consider properties of the training manifold distribution. Typical points produced under VP noising of the base distribution are norm $\sqrt { d + 1 }$ with high probability at any signal level, and all roughly the same magnitude in each coordinate.

Lemma 1 (Constant Norm Shell). Let π be any distribution on $\{ - 1 , 1 \} ^ { d + 1 }$ and $U = a X + { \sqrt { 1 - a ^ { 2 } } } Z$ with $X \sim \pi , Z \sim N ( 0 , I ) , a \in [ 0 , 1 ]$ . Then for any a,

1. $E \| \boldsymbol { U } \| ^ { 2 } = d + 1$ and $\operatorname { V a r } \| U \| ^ { 2 } = 2 ( 1 - a ^ { 4 } ) ( d + 1 )$ ; hence w. $. h . p .$ . the norm of U is near $\sqrt { d + 1 }$ for any $\begin{array} { r } { \epsilon > 0 , P \big ( | \frac { \| U \| ^ { 2 } } { d + 1 } - 1 | > \epsilon \big ) \le \frac { 2 ( 1 - a ^ { 4 } ) } { \epsilon ^ { 2 } ( d + 1 ) } } \end{array}$

2. The magnitude vector $( | U _ { 1 } | , \dots , | U _ { d + 1 } | )$ is i.i.d. with common law $| N ( a , 1 - a ^ { 2 } ) |$ . Its distribution does not depend on π.

We then consider the points produced under a stochastic localization path of a tilted target density. The non-VP scaling of localization paired with a shift term from the linear tilt results in evaluation points that can be of various norm and have one large dominant coordinate in magnitude.

Lemma 2 (Stochastic Localization Tilt). Let $X _ { t }$ be a point observed at time t in a stochastic localization for a target density which is a linear tilt to π by a vector τv. Let $\textstyle \varphi = { \frac { - 1 + { \sqrt { 5 } } } { 2 } } \approx 0 . 6 1 8$ be the number such that $\varphi = 1 - \varphi ^ { 2 } \implies \varphi = \beta ^ { - 1 } ( 1 )$ . Then the denoiser of the stochastic localization is the base denoiser at signal level $\varphi$ with a shift by τv in the evaluation point,

$$
\begin{array} { l } { { \displaystyle E _ { \pi _ { v } } [ X | X _ { t } = t X + \sqrt { t } Z ] = G ( \tau v + X _ { t } , \varphi ) } } \\ { { \displaystyle \qquad = \sum _ { x } x \frac { \pi ( x ) e ^ { \langle \tau v + X _ { t } , x \rangle } } { E _ { \pi } [ e ^ { \langle \tau v + X _ { t } , x \rangle } ] } . } } \end{array}
$$

In particular, for a reward tilt $v = e _ { * }$ , this amounts to large additive term in the $d + 1$ coordinate.

Based on these two lemmas alone, we see the typical evaluation points for G seen in VP noising of the base distribution and those required by stochastic localization of a tilted density look quite diferent. We introduce the following two lemmas to allow for overlap of these two kinds of evaluations. Both of these lemmas are only possible for hypercube embedded distributions where the quadratic tilt of the denoiser simplifies to a pure linear tilt.

First, any G evaluation can be renormalized to a norm $\sqrt { d + 1 }$ vector and an equivalent VP signal level a˜.

Lemma 3 (Renormalization). For any vector v and signal level a, the evaluation of the general mean-tilt operator is equal to an evaluation at a norm $\sqrt { d + 1 }$ scaling of v and an equivalent signal level a˜,

$$
\begin{array} { l } { { G ( v , a ) = G ( \sqrt { d + 1 } \frac { v } { \| v \| } , \tilde { a } ) } } \\ { { \ } } \\ { { C = \beta ( a ) \frac { \| v \| } { \sqrt { d + 1 } } , \tilde { a } = \frac { 1 } { 2 } ( \sqrt { \frac { 1 } { C ^ { 2 } } + 4 } - \frac { 1 } { C } ) . } } \end{array}
$$

Second, for the base density we can multiply our vectors by any diagonal matrix Λ and maintain the property that each conditional expectation is a pure linear tilt. We call this method Λ focusing.

Lemma 4 (Λ focusing). Let Λ be a diagonal matrix, and U a vector produced by first putting $X \sim \pi$ through the matrix and then performing VP noising,

$$
U = a \Lambda X + \sqrt { 1 - a ^ { 2 } } Z , X \sim \pi , Z \sim N ( 0 , I ) , a \in [ 0 , 1 ] .
$$

Then the denoiser is an evaluation of G at the same signal level, but a Λ mapping of the observed u

$$
E [ X | U = a \Lambda X + { \sqrt { 1 - a ^ { 2 } } } Z ] = G ( \Lambda u , a ) .
$$

We construct our diagonal matrix Λ to maintain t $\operatorname { r } ( \Lambda ^ { 2 } ) = d + 1$ by taking a norm $\sqrt { d }$ vector $q \in \mathbb { R } ^ { d }$ , and for some $\lambda \in [ 1 , \sqrt { d + 1 } ]$ ，

$$
\Lambda _ { j , j } = \sqrt { 1 + \frac { 1 } { d } - \frac { \lambda ^ { 2 } } { d } } q _ { j } , j \neq d + 1 ,
$$

$$
\Lambda _ { d + 1 , d + 1 } = \lambda .
$$

This efectively moves our embedding points x from the symmetric hypercube $\{ - 1 , 1 \} ^ { d + 1 }$ to a nonsymmetric hypercube more focused near the one-hot reward vector $e _ { d + 1 }$ we are likely to encounter

![](images/c73efc591c907f902025a4660ed68b200b37bf8d460d008be7dc5da96c1b41ab.jpg)  
Figure 2: Reward based Λ focusing Mode 0 maintains norm of hypersphere points while placing more mass in reward coordinate and increasing cosine similarity with the one-hot vector of the reward coordinate at the pole of the hypersphere.

at sample time, visualized in Figure 2. We consider two choices of $q$ which we call Mode 0 and Mode 1.

Mode 0: Reward Focusing Here we set $q = \vec { 1 } _ { d }$ so all coordinates are present in the Λ matrix and we trade mass from the token component of the vector to the reward coordinate, forcing the hypercube to reside near the pole of the hypersphere.

Mode 1: Infill Training Let $A _ { + }$ be a set of indexes which we would like to fix as positive, and A<sub>−</sub> a set of indexes we would like to fix as negative. Then $\pi$ conditioned on these indexes is the limit of an infinite strength linear tilt by the positive and negative vector

$$
\pi ( x | x _ { A _ { + } } = 1 , x _ { A _ { - } } = - 1 ) = \operatorname* { l i m } _ { \tau \to \infty } \pi ( x ) e ^ { \tau ( \langle e _ { A _ { + } } , x \rangle - \langle e _ { A _ { - } } , x \rangle ) } .
$$

As such, pick a set of indexes $A$ of size $| A | = k .$ , and set q as the norm d indicator set of $\mathrm { A } ,$ shutting of coordinates outside the set, $q _ { j } = 1 \{ j \in A \} \sqrt { d / k }$ . With the indexes $A ^ { C }$ set to 0, this is forcing the trained network denoiser to learn conditionals with only partial information supplied by the indexes present in the set,

$$
E [ X | U _ { A } , U = a X + { \sqrt { 1 - a ^ { 2 } } } Z ] = G ( \operatorname { D i a g } ( 1 _ { A } ) u , a ) .
$$

We refer to this concept as “soft masking”, similar in spirit to masked difusion models [2, 26, 27] where certain tokens are set to a ‘MASK’ token and do not influence the conditional expectation. Here, we can get a similar efect by shutting of certain indexes with a $q _ { j } = 0$ , forcing the denoiser to learn conditional distributions with full information of the U vector, and partial information with various random subsets $U _ { A }$

## 3.1 Combined Results

Combining these symmetries of shifting, normalizing, and Λ focusing, we show how points generated via Λ focusing the training set and points generated via the stochastic localization path of a tilted

density can both be recast as evaluations of the base denoiser at a norm $\sqrt { d + 1 }$ vector and an equivalent VP signal level.

Corollary 1. Let U be a point generated at training time by $\Lambda$ focusing a training point and then applying $V P$ noising

$$
U = a \Lambda X + \sqrt { 1 - a ^ { 2 } } Z , X \sim \pi , Z \sim N ( 0 , I ) , a \in [ 0 , 1 ] ,
$$

then the denoiser at this observation point is equal to an evaluation of the base denoiser at a norm $\sqrt { d + 1 }$ vector and an equivalent signal level $\tilde { a }$

$$
E _ { \pi } [ X | U = a \Lambda X + \sqrt { 1 - a ^ { 2 } } Z ] = G ( \sqrt { d + 1 } \frac { \Lambda u } { \| \Lambda u \| } , \tilde { a } )
$$

$$
C = \frac { a } { 1 - a ^ { 2 } } \frac { | | \Lambda u | | } { \sqrt { d + 1 } } , \tilde { a } = \frac { 1 } { 2 } ( \sqrt { \frac { 1 } { C ^ { 2 } } + 4 } - \frac { 1 } { C } ) .
$$

Corollary 2. Let $X _ { t }$ be a point observed at time t in a stochastic localization for a target density which is a linear tilt to $\pi$ by a vector τv. Then the evaluation of the denoiser can be cast as an evaluation of the base denoiser at a norm $\sqrt { d + 1 }$ vector and an equivalent signal level a˜,

$$
E _ { \pi _ { v } } [ X | X _ { t } = t X + { \sqrt { t } } Z ] = G ( { \sqrt { d + 1 } } { \frac { \tau v + X _ { t } } { \| \tau v + X _ { t } \| } } , { \tilde { a } } )
$$

$$
C = \frac { \| \tau v + X _ { t } \| } { \sqrt { d + 1 } } , \tilde { a } = \frac { 1 } { 2 } \big ( \sqrt { \frac { 1 } { C ^ { 2 } } + 4 } - \frac { 1 } { C } \big ) .
$$

At sampling time, we encounter evaluations of the G operator at various a˜ levels and norm $\sqrt { d + 1 }$ vectors with most of their magnitude contained in the $d + 1$ coordinate. Via Λ focusing, points in our training set which do achieve reward create vectors with similar properties the network can train on. Both types of vectors inhabit the portion of the hypersphere near the one-hot pole $e _ { * }$ which forces overlap of the training distribution and the typical points witnessed at sample time. A further visualization of these symmetries can be found in Appendix Figure 5.

## 4 Training and Sampling Algorithms

In training we may noise the data under any joint distribution of signal level a and Λ focusing matrix. After renormalizing the noisy vector to norm $\sqrt { d + 1 }$ with equivalent signal level ${ \tilde { a } } ,$ the mean-tilt operator G is the population minimizer of the inverse problem regardless of that distribution. Standard VP noising is the special case $\Lambda = I$ with no renormalization.

Lemma 5 (General regression minimizer). Let $\mu$ be any noise distribution over pairs $( a , \Lambda )$ with $a \in [ 0 , 1 ]$ and $\Lambda  { a }$ diagonal matrix. Draw $( a , \Lambda ) \sim \mu , X \sim \pi , Z \sim N ( 0 , I )$ independently, and set $U = a \Lambda X + { \sqrt { 1 - a ^ { 2 } } } Z$ . Let $H : \mathbb { R } ^ { d + 1 } \times [ 0 , 1 ] \to { \mathcal { S } } \times [ 0 , 1 ]$ be the renormalization map of Lemma ${ \mathcal { B } } ,$

$$
\begin{array} { r } { H ( v , a ) = \left( \sqrt { d + 1 } \frac { v } { \| v \| } , \ \tilde { a } \right) } \end{array}
$$

$$
\begin{array} { l l } { \displaystyle { C = \beta ( a ) \frac { \| v \| } { \sqrt { d + 1 } } , \quad \tilde { a } = \frac { 1 } { 2 } ( \sqrt { \frac { 1 } { C ^ { 2 } } + 4 } - \frac { 1 } { C } ) . } } \end{array}
$$

Then over all measurable functions $g ,$ the tilt mean operator G is the population minimizer,

$$
G = \underset { g } { \mathrm { a r g m i n } } \ E \big | \big | X - g \big ( H ( \Lambda U , a ) \big ) \big | \big | ^ { 2 } .
$$

We minimize the empirical loss by SGD in Algorithm 1, drawing a fresh $( a , \Lambda )$ per training point. We sample $\beta$ log-uniformly on $[ \beta _ { \mathrm { m i n } } , \beta _ { \mathrm { m a x } } ] = [ 0 . 0 2$ , 50] and set $a = a ( \beta )$ , the inverse of $\beta ( a ) = a / ( 1 - a ^ { 2 } )$ . With probability p<sub>0</sub> we take $q = 1 _ { d } ,$ else a random k-subset is active. With probability $p _ { \lambda }$ we take $\lambda = 1$ , else $\lambda$ is log-uniform on $[ 1 , \sqrt { d + 1 } ]$ . When $q = 1 _ { d }$ and $\lambda = 1$ this is plain VP noising and no renormalization is applied. We use $p _ { 0 } = 3 / 4 , p _ { \lambda } = 1 / 4 .$ , so $3 / 1 6$ of draws are plain VP noising.

Algorithm 1 Batch SGD step for the tilt–mean network $g _ { \theta }$   
Require: Data $\{ X _ { i } \} \subset \{ - 1 , 1 \} ^ { d + 1 }$ (reward in coordinate $d + 1 )$ ; batch size $N ;$ step size $\gamma$   
1: $\mathcal { L }  0$   
2: for $n = 1 , \ldots , N$ do   
3: Draw $x \sim \{ X _ { i } \} , \ Z \sim { \mathcal { N } } ( 0 , I _ { d + 1 } ) , \ \beta \sim \operatorname { L o g U n i f } [ \beta _ { \operatorname* { m i n } } , \beta _ { \operatorname* { m a x } } ] , \ a  a ( \beta )$   
4: $\operatorname { W . p . } p _ { 0 } \colon q  { \mathbf { 1 } } _ { d } ;$ else draw $k \sim \mathrm { U n i f } \{ 2 , \dots , d \}$ and set $q  \sqrt { d / k } 1 _ { S }$ for a random k-subset   
S   
5: W.p. p<sub>λ</sub>: λ ← 1; else $\lambda \sim \mathrm { L o g U n i f } [ 1 , \sqrt { d + 1 } ]$   
6: $\begin{array} { r } { \Lambda _ { j , j }  \sqrt { 1 + \frac { 1 } { d } - \frac { \lambda ^ { 2 } } { d } q _ { j } } , j \leq d ; \quad \Lambda _ { d + 1 , d + 1 }  \lambda } \end{array}$   
7: $\eta \gets ( a \Lambda ^ { 2 } x + \sqrt { ( 1 - a ^ { 2 } ) } \Lambda Z )$ ▷ tilt vector   
8: if ${ \boldsymbol { q } } = \mathbf { 1 } _ { d }$ and $\lambda = 1$ then   
9: $( v ^ { * } , s ^ { * } )  ( \eta , ~ a )$ ▷ basic VP noising   
10: else   
11: $\begin{array} { r } { C \gets \beta ( a ) \frac { | | \eta | | } { \sqrt { d + 1 } } , \quad \tilde { a } \gets \frac { 1 } { 2 } ( \sqrt { \frac { 1 } { C ^ { 2 } } + 4 } - \frac { 1 } { C } ) ; } \end{array}$   
12: $v ^ { * }  \sqrt { d + 1 } \eta / \| \eta \|$   
13: $\mathcal { L }  \mathcal { L } + \| x - g _ { \boldsymbol { \theta } } ( v ^ { * } , s ^ { * } ) \| ^ { 2 }$   
14: $\theta  \theta - \gamma \nabla _ { \theta } \mathcal { L }$

We note our sampler also makes use of self-conditioning [7] which is a common practice in the field. We discuss the form of self-conditioning in Appendix B.4 to avoid over complication of the discussion in the main body.

Sampling a linear tilt follows Corollary 2: at each step the state is shifted by the tilt vector and renormalized before querying the network, shown in Algorithm 2. The terminal $X _ { T }$ is continuous, and is decoded with one final call to the denoiser network.

```latex
Algorithm 2 Stochastic-localization sampling from the tilted density $\mu \propto \pi e ^ { \left. \nu , \cdot \right. }$ using symmetries
Require: Trained network $g _ { \boldsymbol { \theta } } ;$ tilt vector $\nu = \tau _ { \mathrm { i n f l l } } ( e _ { A ^ { + } } - e _ { A ^ { - } } ) + \tau _ { \mathrm { r e w a r d } } e _ { \ast } ;$ time grid $0 = t _ { 0 } < t _ { 1 } <$
$\bullet \cdot \cdot < t _ { K } = T$
1: $X _ { t _ { 0 } } \gets 0$
2: for $k = 0 , \ldots , K - 1$ do
3: $\delta \gets t _ { k + 1 } - t _ { k }$
4: $\eta  \nu + X _ { t _ { k } }$
5: $\begin{array} { r } { C \gets \frac { \| \eta \| } { \sqrt { d + 1 } } , \quad s ^ { * } \gets \frac { 1 } { 2 } ( \sqrt { \frac { 1 } { C ^ { 2 } } + 4 } - \frac { 1 } { C } ) } \end{array}$
6: $v ^ { * }  \sqrt { d + 1 } \eta / \| \eta \|$
7: Draw $\xi \sim \mathcal { N } ( 0 , I _ { d + 1 } )$
8: $X _ { t _ { k + 1 } } \gets X _ { t _ { k } } + \delta g _ { \theta } ( v ^ { * } , s ^ { * } ) + \sqrt { \delta } \xi$
9: $\eta  \nu + X _ { T } ,$ compute $( \boldsymbol { v } ^ { * } , \boldsymbol { s } ^ { * } )$ from η as above
10: return $\hat { x } \gets \mathrm { s i g n } \big ( g _ { \theta } ( v ^ { * } , s ^ { * } ) \big )$
```

## 5 Experiments

We demonstrate the performance of the method in three experimental settings: a 12-dimensional Ising model with a non-linear reward tilt, 4x4, 6x6, and 9x9 Sudoku unconditional grid generation and puzzle solving (infilling), and MNIST conditional digit generation. Full technical descriptions of the target density, training set, model architecture, and additional details for each experiment can be found in Appendix C. All experiments are sampled up to a stochastic localization time $T = 2 0$ , with the discrete grid $\{ t _ { k } \} _ { k = 0 } ^ { K }$ is constructed with $K = 1 0 0 0$ and a scheduling algorithm defined in Appendix B.5. We note that at each τ tilt strength, we know the theoretical probability of producing a reward-achieving solution using the log-odds curve as discussed in Appendix B.3.

## 5.1 12-Dimensional Ising Model

The 12-dimensional Ising model has a reward tilt which gives reward when the number of positive bits is exactly 10. The state space of the Ising model is small enough to enumerate giving exact ground truth for the denoiser to compare our network against. In Appendix C Figure 8 we see the network denoiser has learned the ground truth denoiser well with low $L ^ { 2 }$ error across tilt strengths, and produces the correct proportion of reward-achieving solutions as predicted by the log-odds curve in Appendix C Figure 9. The Ising model is a small scale experiment with ground truth denoiser to calibrate the method before proceeding to larger state spaces.

## 5.2 4x4 Sudoku

The 4x4 Sudoku problem has a small enough state space to enumerate the ground truth denoiser. We compare the network denoiser to the ground truth in $L ^ { 2 }$ error in Appendix C Figure 11, the network learns the denoiser well. For the log-odds curve in Appendix C Figure 12 we can consider the stochastic localization algorithm implemented with either the ground truth denoiser or the network estimation. Both methods track the log-odds up to 98% reward-achieving solutions at $\tau \approx 3 . 5$ , but the network sampler does start to lag at higher tilt strengths while the ground truth denoiser does continue to track at high tilts up to $\tau = 5$

With 288 reward-achieving puzzles, we can track if the network is uniformly sampling these puzzles. In 3,000 generated reward-achieving puzzles, the exact-denoiser sampler visits all 288 solutions, each between 4 and 21 times, and passes a $\chi ^ { 2 }$ test of uniformity $( \chi _ { 2 8 7 } ^ { 2 } = 2 8 0 . 1 , p = 0 . 6 0 )$ The network visits 287 of the 288 solutions, each between 0 and 25 times, and fails the test $( \chi _ { 2 8 7 } ^ { 2 } = 3 9 1 . 9 , p < 1 0 ^ { - 4 } )$ . The network therefore covers nearly the whole solution set, but it does not sample it uniformly: under the reward tilt it has learned to favor some solutions over others. Results are visualized in Appendix C Figure 15.

An added benefit of embedding in the hypercube is we can do early bit-by-bit decoding of the continuous vector $X _ { t }$ at a time t based on the coordinate-wise entries of the decoder $| E [ X _ { i } | X _ { t } ] | \geq$ 0.75. One of the main complaints against continuous difusion samplers for discrete targets relative to autoregressive models and masked difusion is that because they operate in continuous space there is no discernible discrete token string until the final decoding step. This technique solves that problem and produces a partially complete token string before terminal decoding, visualized in Appendix C Figure 13.

## 5.3 6x6 Sudoku

For 6x6 Sudoku, the state space is large and there are vastly more reward-achieving puzzles than the 4x4 problem. In the 4x4 problem, the state space was size $2 ^ { 3 2 }$ with only 288 reward-achieving puzzles. The training set of size 150,000 contained 5% reward-achieving puzzles indicated by a 1 in their reward coordinate. Thus each reward-achieving puzzle was seen in the training set and a memorization check was performed with $\mathrm { ~ a ~ } \chi ^ { 2 }$ test of uniform sampling.

For 6x6 Sudoku the state space is $2 ^ { 1 0 8 }$ with 28 million reward-achieving puzzles. The training set of size 5 million contained 5% reward-achieving puzzles, which is 250,000 of the 28 million possible valid grids, 0.88%.

We cannot fully enumerate the denoiser for reward tilts in the 6x6 setting, but we can show the $L ^ { 2 }$ error for the untilted density $\tau = 0$ is controlled for the learned network in Appendix C Figure 17. The network also properly tracks the log-odds across tilt strengths in Appendix C Figure 18.

We check memorization by the rate the network reproduces a reward-achieving puzzle provided in the training set. With the training set containing 0.88% of the reward-achieving puzzles, in 1,000 reward-achieving generations proper uniform sampling would produce exact matches with the training set by chance 3–14 with a 95% interval. We observe 10 matches in 1,000 samples, consistent with no memorization.

## 5.4 9x9 Sudoku

For 9x9 Sudoku the state space is too large to enumerate a ground truth denoiser. The number of reward-achieving puzzles is so large there is a near 0 probability of recreating a training set puzzle under proper sampling. The training set is 10 million Sudoku grids with 5% being reward-achieving puzzles.

The 9x9 Sudoku target density is the first example where the efect of self-conditioning is noticeable, self-conditioning is discussed in more detail in Appendix B.4. In summary, we consider two modes of self-conditioning: mode “two-pass” reuses the network’s denoiser evaluation at a given time and maintains a memory one Markov structure. Mode “carry” reuses a previous time point’s denoiser value and breaks this Markov structure but is more commonly used in modern methods.

The two methods give diferent results when tracking the log-odds in Appendix C Figures 20a and 20b. The reward coordinate is denoised jointly with the tokens, so the sampler reports its own prediction of reward in the d + 1 coordinate, and a correctly sampling model should have this prediction agree with the graded reward. The two-pass self-conditioning mode tracks the logodds in its own reward prediction in its d + 1 coordinate, but not in the graded validity, which plateaus near 60%. The model’s d + 1 coordinate reports it is producing reward-achieving grids far more often than it actually is. The carry self-conditioning mode plateaus near 90% graded validity with a 8–12% gap between predicted and graded reward. The reward function is very sharp, mis-sampling a single bit gives 0 reward. Sampled non-reward grids are typically close to reward ones: 90% of two-pass samples lie within 10 out of 324 bits of the nearest valid grid in Hamming distance (Appendix C Figure 21), consistent with a few bits mis-sampled late in the trajectory being penalized by the sharp validity reward.

The model performs both unconditional grid generation and infilling, which some may call “solving” a puzzle, by adding a cell-fixing tilt to the reward tilt (Figure 3). This type of sampling is natural for the difusion model here which operates under linear tilts and decodes in any order, but would be dificult for a left-to-right autoregressive model which would need to backtrack against later given values. We use a reward tilt strength and infill tilt strength of $\tau = 3$ and carry selfcondition mode across various numbers of given cell values in Appendix C Table 1 and Figure 4. Note the sampler may not maintain the initial given cell values as they are a tilt vector not a hard constraint. Given 71/81 cells the model maintains the initial cells and achieved reward in 100% of 2000 draws. Given 11/81 cells, the initial cell values are maintained 92% of the time, a reward achieved 87% of the time, and both objectives satisfied 80% of the time.

![](images/82c03505b7d871a0eba24ca975296f05ed8c74257ba403a55669eafc9462d4b7.jpg)  
(a) Unconditional generation from a blank grid.

![](images/eccebe7e89ce9c92c8d6512a77ca3d3fb540833eece5e118e68ed1d49583af19.jpg)  
(b) Conditional generation from a partially given puzzle.

Figure 3: Sudoku 9x9 completions sampled from the tilt-operator network.  
![](images/f1c3ec073a78de060f4a394c29fd114fff8524a04c1acb7aca859400601d567b.jpg)  
Figure 4: Probability of correct grid infilling across input cell numbers. Given no initial cells, should return a valid grid 95% of the time, we measure saturation at 87% without any infilling.

Finally, the model has no indication of memorizing the training set. Of 1000 unconditional draws at τ = 3 none match the training set. Splitting the training set into two halves, the cell-wise Hamming distance from each reward-achieving sample to its nearest reference grid in one half has the same distribution as that of sample from the split training set half in Appendix Figure 23. Model produced reward-achieving puzzles are novel and no more similar than a two part split of training points to themselves.

## 5.5 MNIST Digit Images

The MNIST data set consists of 60,000 images of hand drawn digits 0-9 and 10,000 hold out examples. We encode reward as a 10-bit one-hot class block, and generating class tilts along the whole block with 1 class being positively tilted and the other 9 negatively tilted. Generated images are classified by an independent CNN classifier (98.74% held-out accuracy). Since the scorer has a ceiling to its classifier accuracy, the observable success rate saturates at the classifier’s per-class sensitivity; points beyond τ ≈ 1.69 are uninformative.

In Appendix C Figure 29 we see generated pools of images show similar variety and quality as hold out examples not used for training indicating no mode collapse or degradation of digit quality under tilting.

In Appendix C Figures 24-27 self-conditioning mode two-pass and carry are also compared for their tracking of the log-odds. In this experiment, carry mode seems to over-sample a class at medium tilts. Two-pass tracks the classifier-corrected exact rate within its confidence interval through τ ≈ 1, in both the classifier score and the network’s own reward block. Carry overshoots the tilt in every class: at τ = 0.51 it produces the requested class 57.8% of the time against an exact 45.9%, with the 95% interval excluding the exact value at every τ from 0.26 to 1.03. Both methods produce 95% class correct images at large tilts, and further tilt strengths go beyond the judge CNN’s sensitivity.

## Acknowledgments

We gratefully acknowledge the support of the NSF for FODSI through grant DMS-2023505, of the NSF and the Simons Foundation for the Collaboration on the Theoretical Foundations of Deep Learning through awards DMS-2031883 and 814639, and of the ONR through MURI award N000142112431, and a research grant from Bridgewater to the Simons Institute.

## AI Use Statement

In this work, we used generative AI tools to generate synthetic data sets, help develop theoretical models or conceptual frameworks, formulate mathematical claims, assist in the writing of proofs, design or provide feedback on research methodology or experiments, implement methods, clean and reformat datasets, support qualitative and thematic data analysis, interpret results. AI assisted coding was used in the project to generate synthetic training sets, write training code, and write evaluation code. AI was used to suggest hyper-parameter choices and present analysis of outputs. All code and outputs also human verified. All theorems and proofs are original constructs of the authors. A first draft written with purely human analysis. This draft is then presented to the AI for suggested edits, which may be incorporated by the authors into a refined proof. All proofs are human verified as correct and began with the authors own steps of logic as the foundation.

We have not used generative AI tools to provide critical ingredients for proving mathematical claims, propose or refine hypotheses, or assist with translation. “Critical ingredients” for proofs is subjective. The authors do not consider AI as presenting any of the critical ideas that make the proofs work, but rather suggesting useful notation and suggesting more succinct summaries of steps. All logical steps in the proofs were human designed initially.

Additionally, we used generative AI tools to suggest experimental parameters, create or edit software code, summarize or analyse existing literature, discover research topics or identify gaps, brainstorming, sourcing/searching for information, edit a research paper to improve readability, identify relevant literature.

We have reviewed all AI-assisted work. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## References

[1] Anderson, B. D. O. (1982). Reverse-time difusion equation models. Stochastic Processes and their Applications, 12(3):313–326.

[2] Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and van den Berg, R. (2021). Structured denoising difusion models in discrete state-spaces. In Advances in Neural Information Processing Systems (NeurIPS).

[3] Benton, J., De Bortoli, V., Doucet, A., and Deligiannidis, G. (2024). Nearly d-linear convergence bounds for difusion models via stochastic localization. In International Conference on Learning Representations.

[4] Bradley, A. and Nakkiran, P. (2025). Classifier-free guidance is a predictor-corrector. Transactions on Machine Learning Research.

[5] Chen, H., Lee, H., and Lu, J. (2023a). Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 4735–4763. PMLR.

[6] Chen, S., Chewi, S., Li, J., Li, Y., Salim, A., and Zhang, A. R. (2023b). Sampling is as easy as learning the score: theory for difusion models with minimal data assumptions. In International Conference on Learning Representations.

[7] Chen, T., Zhang, R., and Hinton, G. (2023c). Analog bits: Generating discrete data using difusion models with self-conditioning. In International Conference on Learning Representations (ICLR).

[8] Chidambaram, M., Gatmiry, K., Chen, S., Lee, H., and Lu, J. (2024). What does guidance do? a fine-grained analysis in a simple setting. In Advances in Neural Information Processing Systems (NeurIPS).

[9] Conforti, G., Durmus, A., and Silveri, M. G. (2025). KL convergence guarantees for score difusion models under minimal data assumptions. SIAM Journal on Mathematics of Data Science, 7(1):86–109.

[10] Dandapanthula, S. and Bofi, N. M. (2026). Are we really tilting? The mechanics of reward guidance in flow and difusion models. arXiv preprint arXiv:2606.02884.

[11] Denker, A., Vargas, F., Padhy, S., Didi, K., Mathis, S., Dutordoir, V., Barbano, R., Mathieu, E., Komorowska, U. J., and Liò, P. (2024). DEFT: Eficient fine-tuning of difusion models by learning the generalised h-transform. In Advances in Neural Information Processing Systems (NeurIPS), volume 37.

[12] He, J., Hernández-Lobato, J. M., Du, Y., and Vargas, F. (2026). RNE: Plug-and-play difusion inference-time control and energy-based training. In International Conference on Learning Representations (ICLR).

[13] Ho, J., Jain, A., and Abbeel, P. (2020). Denoising difusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), volume 33.

[14] Ho, J. and Salimans, T. (2022). Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598.

[15] Holderrieth, P., Singer, U., Jaakkola, T., Chen, R. T., Lipman, Y., and Karrer, B. (2025). Glass flows: Transition sampling for alignment of flow and difusion models. arXiv preprint arXiv:2509.25170.

[16] Hu, K., Qiu, L., Lu, Y., Zhao, H., Li, T., Kim, Y., Andreas, J., and He, K. (2026). Elf: Embedded language flows. arXiv preprint arXiv:2605.10938.

[17] Lee, C. K., Jeha, P., Frellsen, J., Liò, P., Albergo, M. S., and Vargas, F. (2025). Debiasing guidance for discrete difusion with sequential Monte Carlo. arXiv preprint arXiv:2502.06079.

[18] Lee, H., Lu, J., and Tan, Y. (2022). Convergence for score-based generative modeling with polynomial complexity. In Advances in Neural Information Processing Systems, volume 35, pages 22870–22882.

[19] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. (2023). Flow matching for generative modeling. In International Conference on Learning Representations (ICLR).

[20] McDonald, C. (2026). Technical note on relating scores of tilted distributions. arXiv preprint arXiv:2604.27196.

[21] Moitra, A., Risteski, A., and Rohatgi, D. (2026a). Steering difusion models with quadratic rewards: a fine-grained analysis. arXiv preprint arXiv:2602.16570.

[22] Moitra, A., Risteski, A., and Rohatgi, D. (2026b). The tractability landscape of difusion alignment: regularization, rewards, and computational primitives. arXiv preprint arXiv:2605.11361.

[23] Montanari, A. (2023). Sampling, difusions, and stochastic localization. arXiv preprint arXiv:2305.10690.

[24] Nisonof, H., Xiong, J., Allenspach, S., and Listgarten, J. (2025). Unlocking guidance for discrete state-space difusion and flow models. In International Conference on Learning Representations (ICLR).

[25] Robbins, H. E. (1992). An empirical Bayes approach to statistics. In Kotz, S. and Johnson, N. L., editors, Breakthroughs in Statistics: Foundations and Basic Theory, pages 388–394. Springer, New York.

[26] Sahoo, S. S., Arriola, M., Schif, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., and Kuleshov, V. (2024). Simple and efective masked difusion language models. In Advances in Neural Information Processing Systems (NeurIPS).

[27] Shi, J., Han, K., Wang, Z., Doucet, A., and Titsias, M. K. (2024). Simplified and generalized masked difusion for discrete data. In Advances in Neural Information Processing Systems (NeurIPS).

[28] Skreta, M., Akhound-Sadegh, T., Ohanesian, V., Bondesan, R., Aspuru-Guzik, A., Doucet, A., Brekelmans, R., Tong, A., and Neklyudov, K. (2025). Feynman-Kac correctors in difusion: Annealing, guidance, and product of experts. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), volume 267 of Proceedings of Machine Learning Research, pages 55906–55949.

[29] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning (ICML), pages 2256–2265.

[30] Song, Y., Durkan, C., Murray, I., and Ermon, S. (2021a). Maximum likelihood training of score-based difusion models. In Advances in Neural Information Processing Systems, volume 34, pages 1415–1428.

[31] Song, Y. and Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems (NeurIPS), volume 32.

[32] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2021b). Score-based generative modeling through stochastic diferential equations. In International Conference on Learning Representations (ICLR).

[33] Wan, Z., Ouyang, Y., Xie, L., Fang, F., Zha, H., and Cheng, G. (2026). Discrete guidance matching: Exact guidance for discrete flow matching. In International Conference on Learning Representations (ICLR).

[34] Wang, Y., He, Y., and Tao, M. (2024). Evaluating the design space of difusion-based generative models. Advances in Neural Information Processing Systems, 37:19307–19352.

[35] Yang, Z., Guo, W., Zhang, S., Sahoo, S. S., Chen, Y., Vahdat, A., Mardani, M., and Thickstun, J. (2026). Continuous difusion scales competitively with discrete difusion for language. arXiv preprint arXiv:2605.18530.

## A PROOFS

Proof of Lemma 1

Proof. Throughout the proof X and Z are independent and $X _ { i } ^ { 2 } = 1$ for every i.

Condition (1):

Note the moments $E [ Z _ { i } ] = E [ Z _ { i } ^ { 3 } ] = 0 , E [ Z _ { i } ^ { 2 } ] = 1 , E [ Z _ { i } ^ { 4 } ] = 3$ for a standard normal random variable.

The random variable U is the linear combination of X and Z, and $Z$ is mean 0. The norm of the sum decomposes into the sum of the norms with a cross term that is 0,

$$
\begin{array} { r l } & { E [ \| U \| ^ { 2 } ] = a ^ { 2 } E [ \| X \| ^ { 2 } ] + 2 a \sqrt { 1 - a ^ { 2 } } \langle E [ X ] , E [ Z ] \rangle + ( 1 - a ^ { 2 } ) E [ \| Z \| ^ { 2 } ] } \\ & { \qquad = a ^ { 2 } E [ \| X \| ^ { 2 } ] + ( 1 - a ^ { 2 } ) E [ \| Z \| ^ { 2 } ] . } \end{array}
$$

Since the state space of X is restricted to the hypercube $\{ - 1 , 1 \} ^ { d + 1 } , \| X \| ^ { 2 } = d + 1$ w.p. 1. Since Z is standard normal. $E [ \| Z \| ^ { 2 } ] = d + 1$ . The expectation is then $d + 1$ for any $\mathrm { V P }$ combination a. The variance decomposes as

$$
E [ ( \| U \| ^ { 2 } - ( d + 1 ) ) ^ { 2 } ] = \sum _ { i , j = 1 } ^ { d + 1 } E [ ( U _ { i } ^ { 2 } - 1 ) ( U _ { j } ^ { 2 } - 1 ) ] .
$$

Note that $X _ { i } ^ { 2 } = 1$ constantly, so we expand

$$
U _ { i } ^ { 2 } - 1 = 2 a \sqrt { 1 - a ^ { 2 } } X _ { i } Z _ { i } + ( 1 - a ^ { 2 } ) ( Z _ { i } ^ { 2 } - 1 ) .
$$

Consider first the diagonal terms $i = j$

$$
\begin{array} { r l } & { E [ ( U _ { i } ^ { 2 } - 1 ) ^ { 2 } ] = 4 a ^ { 2 } ( 1 - a ^ { 2 } ) E [ Z _ { i } ^ { 2 } ] + ( 1 - a ^ { 2 } ) ^ { 2 } E [ ( Z _ { i } ^ { 2 } - 1 ) ^ { 2 } ] + 4 a ( 1 - a ^ { 2 } ) ^ { 3 / 2 } E [ X _ { i } ] E [ Z _ { i } ^ { 3 } - Z _ { i } ] } \\ & { \phantom { a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a } } \\ & { \phantom { a a a a a a a a a a a a a a a } = 4 a ^ { 2 } ( 1 - a ^ { 2 } ) + 2 ( 1 - a ^ { 2 } ) ^ { 2 } } \\ & { \phantom { a a a a a a a a a a a a a a a a a a a } = 2 ( 1 - a ^ { 4 } ) . } \end{array}
$$

Then the cross terms $i \neq j$ are mean 0. Expand $E [ ( U _ { i } ^ { 2 } - 1 ) ( U _ { j } ^ { 2 } - 1 ) ]$ , every term vanishes since $Z _ { i } \perp Z _ { j }$ and both are independent of X:

$$
\begin{array} { c } { { E [ X _ { i } X _ { j } Z _ { i } Z _ { j } ] = E [ X _ { i } X _ { j } ] E [ Z _ { i } ] E [ Z _ { j } ] = 0 , } } \\ { { E [ X _ { i } Z _ { i } ( Z _ { j } ^ { 2 } - 1 ) ] = E [ X _ { i } ] E [ Z _ { i } ] E [ Z _ { j } ^ { 2 } - 1 ] = 0 , } } \\ { { E [ ( Z _ { i } ^ { 2 } - 1 ) ( Z _ { j } ^ { 2 } - 1 ) ] = E [ Z _ { i } ^ { 2 } - 1 ] E [ Z _ { j } ^ { 2 } - 1 ] = 0 . } } \end{array}
$$

Therefore Var $\begin{array} { r } { \| U \| ^ { 2 } = \sum _ { i } \operatorname { V a r } U _ { i } ^ { 2 } = 2 ( 1 - a ^ { 4 } ) ( d + 1 ) } \end{array}$ . Since the variance is on the same scale as the expectation $d + 1$ , a Chebyshev’s inequality shows the norm is highly concentrated around its expected value in ratio,

$$
P \Big ( \Big | \frac { \| U \| ^ { 2 } } { d + 1 } - 1 \Big | > \epsilon \Big ) = P \big ( | \| U \| ^ { 2 } - ( d + 1 ) | > \epsilon ( d + 1 ) \big ) \leq \frac { 2 ( 1 - a ^ { 4 } ) ( d + 1 ) } { \epsilon ^ { 2 } ( d + 1 ) ^ { 2 } } = \frac { 2 ( 1 - a ^ { 4 } ) } { \epsilon ^ { 2 } ( d + 1 ) } .
$$

Condition (2):

Condition on $X = x \in \{ - 1 , 1 \} ^ { d + 1 }$ . Since $| x _ { i } | = 1$

$$
| U _ { i } | = | a x _ { i } + { \sqrt { 1 - a ^ { 2 } } } Z _ { i } | = | a + { \sqrt { 1 - a ^ { 2 } } } x _ { i } Z _ { i } | .
$$

Coordinate-wise sign flips preserve $N ( 0 , I )$ , so $( x _ { 1 } Z _ { 1 } , \dots , x _ { d + 1 } Z _ { d + 1 } )$ has the same law as $Z .$ Hence, conditionally on $X = x , ( | U _ { 1 } | , \dots , | U _ { d + 1 } | )$ has the law of $( | a + { \sqrt { 1 - a ^ { 2 } } } Z _ { 1 } | , \ldots , | a + { \sqrt { 1 - a ^ { 2 } } } Z _ { d + 1 } | )$ This vector has i.i.d. coordinates with law $| N ( a , 1 - a ^ { 2 } ) |$ . The conditional law does not depend on $x ,$ so it is also the unconditional law, whatever π is. □

## Proof of Lemma 2

Proof. Let $q ( x ) \propto \pi ( x ) e ^ { \tau \langle x , v \rangle }$ . It is a property of the stochastic localization path the marginal density of $X _ { t }$ is equivalent to a specific linear combination with a normal random variable,

$$
X _ { t } = t X + { \sqrt { t } } Z , p ( x _ { t } | x ) \sim N ( t x , t I ) .
$$

The conditional density for $p ( \boldsymbol { x } | \boldsymbol { x } _ { t } )$ is then

$$
\begin{array} { r l } & { p ( x | x _ { t } ) \propto q ( x ) e ^ { - \frac { 1 } { 2 t } \| x _ { t } - t x \| ^ { 2 } } } \\ & { \qquad \propto \pi ( x ) e ^ { \tau \langle x , v \rangle } e ^ { - \frac { 1 } { 2 t } \| x _ { t } \| ^ { 2 } + \frac { t } { t } \langle x , x _ { t } \rangle - \frac { t } { 2 } \| x \| ^ { 2 } } , } \end{array}
$$

with $x$ as the random variable and $x _ { t }$ as the conditioning point, $\| x _ { t } \| ^ { 2 }$ can be dropped as a normalizing constant scaling all $x$ locations equally. Furthermore, by the assumption $x$ lives in the hypercube, it is constant norm and $\| { \boldsymbol { x } } \| ^ { 2 }$ can also be dropped. Thus $p ( \boldsymbol { x } | \boldsymbol { x } _ { t } )$ is proportional to a pure linear tilt to the base density $\pi .$

$$
p ( x | x _ { t } ) \propto \pi ( x ) e ^ { \left. \tau v + x _ { t } , x \right. } .
$$

Recall the definition of the tilt strength function $\textstyle \beta ( a ) = { \frac { a } { 1 - a ^ { 2 } } } , a \in [ 0 , 1 ]$ . The golden ratio $\theta =$ $\textstyle { \frac { 1 + { \sqrt { 5 } } } { 2 } } \approx 1$ .618 is precisely the number such that $\theta + 1 = \theta ^ { 2 }$ . We define $\varphi = \theta - 1 \approx 0 . 6 1 8$ as the number such that

$$
1 - \varphi ^ { 2 } = 1 - ( \theta - 1 ) ^ { 2 } = - \theta ^ { 2 } + 2 \theta = - 1 + \theta = \varphi .
$$

Thus it is the case that $\beta ( \varphi ) = 1$ and we can express the stochastic localization’s denoiser in terms of $G$ using $\varphi$ as our mixture level and $\tau v + X _ { t }$ as the evaluation location,

$$
E _ { \pi _ { v } } [ X | X _ { t } = t X + { \sqrt { t } } Z ] = \sum _ { x } x { \frac { \pi ( x ) e ^ { \beta ( \varphi ) \langle \tau \nu + X _ { t } , x \rangle } } { E _ { \pi } [ e ^ { \beta ( \varphi ) \langle \tau \nu + X _ { t } , x \rangle } ] } } = G ( \tau v + X _ { t } , \varphi ) .
$$

## Proof of Lemma 3

Proof. For the mean-tilt operator $G ( v , a )$ , the VP tilt strength operator $\begin{array} { r } { \beta ( a ) = \frac { a } { 1 - a ^ { 2 } } } \end{array}$ captures the term that scales the inner product $\langle v , x \rangle$

$$
G ( v , a ) = \sum _ { x } x \frac { \pi ( x ) e ^ { \beta ( a ) \langle v , x \rangle } } { \sum _ { \bar { x } } \pi ( \bar { x } ) e ^ { \beta ( a ) \langle v , \bar { x } \rangle } } .
$$

The operator $\beta$ maps $[ 0 , 1 ]  [ 0 , \infty )$ and is increasing, thus any tilt strength scaling b can be achieved by some signal level that is the $\beta$ mapping at that value $a = \beta ^ { - 1 } ( b )$ . Inverting $\beta$ results in finding the roots of a quadratic solved by quadratic formula,

$$
\begin{array} { c } { { \beta ( a ) = C } } \\ { { a = ( 1 - a ^ { 2 } ) C } } \\ { { C a ^ { 2 } + a - C = 0 \implies a = \frac { - 1 \pm \sqrt { 1 + 4 C ^ { 2 } } } { 2 C } , } } \end{array}
$$

only the positive root results in $a \in [ 0 , 1 ]$ , so we have inverse mapping for any $C \in [ 0 , \infty )$

$$
\beta ^ { - 1 } ( C ) = \frac { 1 } { 2 } \left( \sqrt { \frac { 1 } { C ^ { 2 } } + 4 } - \frac { 1 } { C } \right) .
$$

For any inputs $v , a$ to the base denoiser, we can multiply and divide the v vector by any scaling λ in the exponent and collect the remaining term in front of the inner product as some new tilt strength,

$$
\sum _ { x } x { \frac { \pi ( x ) e ^ { \beta ( a ) \langle v , x \rangle } } { \sum _ { \bar { x } } \pi ( { \bar { x } } ) e ^ { \beta ( a ) \langle v , { \bar { x } } \rangle } } } = \sum _ { x } x { \frac { \pi ( x ) e ^ { ( \beta ( a ) \lambda ) \langle { \frac { v } { \lambda } } , x \rangle } } { \sum _ { \bar { x } } \pi ( { \bar { x } } ) e ^ { ( \beta ( a ) \lambda ) \langle { \frac { v } { \lambda } } , { \bar { x } } \rangle } } } .
$$

Setting $\begin{array} { r } { \lambda = \frac { \| v \| } { \sqrt { d + 1 } } } \end{array}$ results in $\begin{array} { r } { \tilde { v } = \frac { v } { \lambda } } \end{array}$ as being a norm $\sqrt { d + 1 }$ vector, and we solve for the equivalent signal level a˜ via the inverse map $\beta ^ { - 1 } ( \beta ( a ) \lambda )$

## Proof of Lemma 4

Proof. In this proof we take advantage of each coordinate of x itself being $\{ - 1 , 1 \}$ valued such that $x _ { i } ^ { 2 } = 1$ for all coordinates i. This is a property of the hypercube $\{ - 1 , 1 \} ^ { d }$ , while the hypersphere only has the property that $\| x \| ^ { 2 } = d .$ By construction the conditional density of $U | X$ is $N ( a \Lambda X , ( 1 -$ $a ^ { 2 } ) I )$ . Therefore, the reverse conditional density via Bayes rule is a quadratic tilt to the base density

$$
\begin{array} { r l } & { p ( x | u , a , \Lambda ) \propto \pi ( x ) p ( u | x , a , \Lambda ) } \\ & { \qquad \propto \pi ( x ) e ^ { - \frac { 1 } { 2 ( 1 - a ^ { 2 } ) } \| u - a \Lambda x \| ^ { 2 } } } \\ & { \qquad \propto \pi ( x ) e ^ { \frac { a } { 1 - a ^ { 2 } } \langle \Lambda u , x \rangle } e ^ { - \frac { 1 } { 2 ( 1 - a ^ { 2 } ) } \| u \| ^ { 2 } - \frac { a ^ { 2 } } { 2 ( 1 - a ^ { 2 } ) } \sum _ { i = 1 } ^ { d + 1 } \Lambda _ { i , i } ^ { 2 } x _ { i } ^ { 2 } } . } \end{array}
$$

X is the random variable while u is the conditioning value, thus $\| u \| ^ { 2 }$ can be ignored as a normalizing constant. Since each $x _ { i } ^ { 2 } = 1$ for any element of the hypercube, no matter the values of $\Lambda _ { i , i }$ the x dependent quadratic term is also constant and is removed in normalization. Thus the conditional expectation is a pure linear tilt to the base density. □

## Proof of Corollary 1

Proof. In this proof we collect the previous denoiser symmetry results of Lemma 3 and 4 the specify the form of the base denoiser under Λ focusing as a $\sqrt { d + 1 }$ vector with an equivalent signal level a˜. Via Lemma 4, the conditional expectation after Λ focusing and noising is $G$ at ΛU and a,

$$
E _ { \pi } [ X | U = a \Lambda X + \sqrt { 1 - a ^ { 2 } } Z ] = G ( \Lambda u , a ) ,
$$

we can then renormalize to $\begin{array} { r } { \tilde { u } = \sqrt { d + 1 } \frac { \Lambda u } { \Vert \Lambda u \Vert } } \end{array}$ and solve for a˜ using Lemma 3.

## Proof of Corollary 2

Proof. In this proof we collect the previous denoiser symmetry results of Lemma 2 and 3 to specify the form of the tilted denoiser under a stochastic localization path as a $\sqrt { d + 1 }$ vector with an equivalent signal level a˜. By Lemma 2, the denoiser of the stochastic localization path for a tilt defined by a vector τv is a linear shift to the evaluation point of the mean-tilt operator at the signal level $\varphi \approx 0 . 6 1 8$

$$
E _ { \pi _ { v } } [ X | X _ { t } = t X + { \sqrt { t } } Z ] = G ( \tau v + X _ { t } , \varphi ) ,
$$

we can then renormalize to $\begin{array} { r } { \tilde { u } = \sqrt { d + 1 } \frac { \tau v + X _ { t } } { \Vert \tau v + X _ { t } \Vert } } \end{array}$ and solve for a˜ using Lemma 3.

## Proof of Lemma 5

Proof. The loss function here is set up as an inverse problem, where for any two random variables with a joint distribution, the minimizer of the expected squared loss of one variable minus a function of the other is the conditional mean. The previous Lemmas have established that various conditional means can be expressed as evaluations of G at norm $\sqrt { d + 1 }$ vectors, which shows this loss function with renormalization is an equivalent expression of an inverse problem.

Throughout the proof let $( a , \Lambda ) \sim \mu , X \sim \pi$ and $Z \sim N ( 0 , I )$ be independent random variables and $U = a \Lambda X + { \sqrt { 1 - a ^ { 2 } } } Z$ . For a measurable g write

$$
\begin{array} { r } { L ( g ) = E _ { ( a , \Lambda ) \sim \mu , X \sim \pi , Z } \big \| X - g ( U , a , \Lambda ) \big \| ^ { 2 } . } \end{array}
$$

The proof proceeds in three steps: (i) for fixed $( a , \Lambda )$ the minimizer of the squared loss is the conditional mean of X given U; (ii) for random $( a , \Lambda ) \sim \mu ,$ the tower property reduces the problem to (i) because $( a , \Lambda )$ is independent of $X ; ( \mathrm { i i i } )$ the resulting minimizer is $G ( \Lambda u , a )$ , which Lemma 3 rewrites as $G ( H ( \Lambda u , a ) )$ , a function of the renormalized input alone.

Step 1: fixed $( a , \Lambda )$ . Fix $a \in [ 0 , 1 ]$ and a diagonal Λ, and let $U _ { a , \Lambda } = a \Lambda X + \sqrt { 1 - a ^ { 2 } } Z$ with $X \sim \pi , Z \sim N ( 0 , I )$ independent. For any measurable $f ,$ , the tower property gives the orthogonal decomposition

$$
E _ { X , Z } \left\| X - f ( U _ { a , \Lambda } ) \right\| ^ { 2 } = E _ { X , Z } \left\| X - E [ X \mid U _ { a , \Lambda } ] \right\| ^ { 2 } + E _ { X , Z } \left\| E [ X \mid U _ { a , \Lambda } ] - f ( U _ { a , \Lambda } ) \right\| ^ { 2 } ,
$$

It is then a standard fact that the minimizing function is the conditional mean

$$
g _ { a , \Lambda } ( u ) : = E _ { X \sim \pi } [ X \mid U _ { a , \Lambda } = u ] \quad \mathrm { s a t i s f i e s } \quad g _ { a , \Lambda } = \underset { f } { \operatorname { a r g m i n } } E _ { X , Z } \Vert X - f ( U _ { a , \Lambda } ) \Vert ^ { 2 } .
$$

By Lemma 4, the posterior of X given $U _ { a , \Lambda } = u$ is the linear tilt $\pi ( x ) e ^ { \beta ( a ) \left. \Lambda u , x \right. }$ , hence $g _ { a , \Lambda } ( u ) =$ $G ( \Lambda u , a )$

Step 2: random $( a , \Lambda ) \sim \mu$ . Condition on $( a , \Lambda )$ and apply the tower property with the $( a , \Lambda )$ expectation outermost,

$$
L ( g ) = E _ { ( a , \Lambda ) \sim \mu } \Big [ E \big [ \| X - g ( U , a , \Lambda ) \| ^ { 2 } \big | a , \Lambda \big ] \Big ] .
$$

Because $( a , \Lambda )$ is independent of $( X , Z )$ , the conditional law of $( X , U )$ given $( a , \Lambda )$ is the law of $( X , U _ { a , \Lambda } )$ from Step 1; in density form,

$$
p ( x , u \mid a , \Lambda ) = p ( u \mid x , a , \Lambda ) p ( x \mid a , \Lambda ) = p ( u \mid x , a , \Lambda ) \pi ( x ) ,
$$

i.e. X is conditionally independent of $( a , \Lambda )$ , and $( a , \Lambda )$ enters only through the noising kernel $p ( u \mid x , a , \Lambda )$ . Therefore the inner conditional expectation is

$$
\begin{array} { r } { E \big [ \| X - g ( U , a , \Lambda ) \| ^ { 2 } \big | a , \Lambda \big ] = E _ { X \sim \pi , Z } \big | \big | X - g ( U _ { a , \Lambda } , a , \Lambda ) \big | \big | ^ { 2 } , } \end{array}
$$

which, for each fixed $( a , \Lambda )$ , is exactly the objective of Step 1 with $f = g ( \cdot , a , \Lambda )$ . Since $g$ is allowed to depend on $( a , \Lambda )$ , the choice of $g ( \cdot , a , \Lambda )$ for one value of $( a , \Lambda )$ does not constrain its choice for another, so the outer expectation is minimized by minimizing the integrand pointwise in $( a , \Lambda )$

$$
\operatorname* { m i n } _ { g } L ( g ) = E _ { ( a , \Lambda ) \sim \mu } \Big [ \operatorname* { m i n } _ { f } E _ { X , Z } \big \| X - f ( U _ { a , \Lambda } ) \big \| ^ { 2 } \Big ] , \qquad g ^ { \star } ( u , a , \Lambda ) = g _ { a , \Lambda } ( u ) = G ( \Lambda u , a ) .
$$

Step 3: renormalization. By Lemma 3, for any $v \in \mathbb { R } ^ { d + 1 } \ \backslash$ {0} and $a \in [ 0 , 1 ]$

$$
\begin{array} { r } { G ( v , a ) = G \Big ( \sqrt { d + 1 } \frac { v } { \| v \| } , \tilde { a } \Big ) = G \big ( H ( v , a ) \big ) , \qquad \tilde { a } = \beta ^ { - 1 } \Big ( \beta ( a ) \frac { \| v \| } { \sqrt { d + 1 } } \Big ) , } \end{array}
$$

so $g ^ { \star } ( u , a , \Lambda ) = G ( H ( \Lambda u , a ) )$ . In particular the unconstrained minimizer depends on $( U , a , \Lambda )$ only through $H ( \Lambda U , a )$ . The class $\{ g \circ H ( \Lambda \cdot , \cdot ) : g$ measurable on $S \times [ 0 , 1 ] \}$ is a subset of the measurable functions of $( U , a , \Lambda )$ , and the minimizer over the larger class lies in ${ \mathrm { i t } } ,$ so it is also the minimizer over the smaller class:

$$
G = \underset { g } { \mathrm { a r g m i n } } E \big | \big | X - g \big ( H ( \Lambda U , a ) \big ) \big | \big | ^ { 2 } ,
$$

uniquely up to sets of measure zero under the law of $H ( \Lambda U , a )$

## B Additional Discussion

## B.1 Visualization of Denoiser Symmetry

![](images/460330f01678a0701e7b0942ad801c3736a34ebf2019b0eebb7a7056d7598ffa.jpg)  
Figure 5: Sketch of overlap between renormalized training set via Λ focusing and points shifted by tilt along stochastic localization path.

## B.2 Stochastic Localization

Stochastic localization can be viewed as a diferent time scaling and spatial scaling than the usual score-based difusion derived from the forwards flow Ornstein-Uhlenbeck process as outlined in Montanari [23]. We re-derive the equivalence here for an audience more familiar with the scorebased construction. The score-based difusion is variance preserving, initialized at a standard normal draw, and uses the score as its drift. The stochastic localization process is not variance preserving (it is growing in magnitude over time), is initialized at a Dirac mass at the origin, and uses the denoiser as its drift term. Both can be derived from the same reverse difusion process theory.

Time reversal. Both constructions are reverse-time implementations of a forward difusion producing a sequence of marginal densities that are equivalent to convolutions with Gaussian random variables of various mixture amounts. By the time-reversal result of Anderson [1], popularized in modern literature by Song et al. [32], Lipman et al. [19], consider any target density π and initialize $X _ { 0 } \sim \pi$ . Consider any forward process

$$
d X _ { t } = f ( X _ { t } , t ) d t + g ( t ) d B _ { t } , \qquad t \in [ 0 , T ] , X _ { 0 } \sim \pi ,
$$

and let $p _ { t }$ denote the marginal density of $X _ { t }$ with score $s _ { t } = \nabla \log p _ { t }$ . The reverse process started from $Y _ { 0 } \sim p _ { T }$

$$
d Y _ { t } = - \big [ f ( Y _ { t } , T - t ) - g ( T - t ) ^ { 2 } s _ { T - t } ( Y _ { t } ) \big ] d t + g ( T - t ) d B _ { t } ,\tag{1}
$$

satisfies $Y _ { t } ~ \sim ~ p _ { T - t }$ , i.e. it traverses the forward marginals in reverse order. The only quantity required to implement (1) is the score, hence the name score-based difusion. For general f and $g$ the marginal $p _ { t }$ , and hence $s _ { t } .$ , has no specific closed form.

Linear forward processes. When the forward drift $f ( X _ { t } , t )$ is linear in $X _ { t }$ the marginals are Gaussian convolutions and the score is available through Tweedie’s formula [25]. Let $\alpha , \sigma : [ 0 , T ] \to$ $\mathbb { R } _ { + }$ be diferentiable with $\sigma / \alpha$ non decreasing, and define

$$
\kappa ( t ) : = \frac { d } { d t } \log \frac { \sigma ( t ) } { \alpha ( t ) } = \frac { \dot { \sigma } ( t ) } { \sigma ( t ) } - \frac { \dot { \alpha } ( t ) } { \alpha ( t ) } \ \geq \ 0 .
$$

Consider the linear forward process

$$
d X _ { t } = \frac { \dot { \alpha } ( t ) } { \alpha ( t ) } X _ { t } d t + \sigma ( t ) \sqrt { 2 \kappa ( t ) } d B _ { t } , \qquad X _ { 0 } = \alpha ( 0 ) X + \sigma ( 0 ) Z ,\tag{2}
$$

with $X \sim \pi$ and $Z \sim N ( 0 , I )$ independent. A direct computation shows that its marginals are equivalent to convolution with a normal random variable at a specific scaling,

$$
\begin{array} { r } { X _ { t } \stackrel { d } { = } \alpha ( t ) X + \sigma ( t ) Z , \qquad \mathrm { i . e . } \qquad p _ { t } ( x _ { t } \mid X = x _ { 0 } ) = N ( x _ { t } ; \alpha ( t ) x _ { 0 } , \sigma ( t ) ^ { 2 } I ) . } \end{array}
$$

Writing $m _ { t } ( x ) : = E [ X \mid X _ { t } = x ]$ for the denoiser, Tweedie’s formula [25] gives the score as a posterior mean under a quadratic tilt of π and is related to the denoiser,

$$
s _ { t } ( x _ { t } ) = \int - \frac { x _ { t } - \alpha ( t ) x _ { 0 } } { \sigma ( t ) ^ { 2 } } \frac { \pi ( x _ { 0 } ) e ^ { - \frac { 1 } { 2 \sigma ( t ) ^ { 2 } } \| x _ { t } - \alpha ( t ) x _ { 0 } \| ^ { 2 } } } { \int \pi ( x _ { 0 } ^ { \prime } ) e ^ { - \frac { 1 } { 2 \sigma ( t ) ^ { 2 } } \| x _ { t } - \alpha ( t ) x _ { 0 } ^ { \prime } \| ^ { 2 } } d x _ { 0 } ^ { \prime } } d x _ { 0 } = \frac { \alpha ( t ) m _ { t } ( x _ { t } ) - x _ { t } } { \sigma ( t ) ^ { 2 } } .
$$

Substituting into (1), and abbreviating $\alpha = \alpha ( T - t ) , \sigma = \sigma ( T - t ) , \kappa = \kappa ( T - t )$ , the reverse process is

$$
d Y _ { t } = \Big [ - { \frac { \dot { \alpha } } { \alpha } } Y _ { t } + 2 \sigma ^ { 2 } \kappa s _ { T - t } ( Y _ { t } ) \Big ] d t + \sigma \sqrt { 2 \kappa } d B _ { t } .\tag{3}
$$

Equivalently stated in terms of the denoiser instead of the score,

$$
d Y _ { t } = \Big [ - \Big ( { \frac { \dot { \alpha } } { \alpha } } + 2 \kappa \Big ) Y _ { t } + 2 \kappa \alpha m _ { T - t } ( Y _ { t } ) \Big ] d t + \sigma \sqrt { 2 \kappa } d B _ { t } .\tag{4}
$$

Signal level and growth. A linear combination $\alpha ( t ) X + \sigma ( t ) Z$ is called variance preserving (VP) when $\alpha ( t ) ^ { 2 } + \sigma ( t ) ^ { 2 } = 1$ for all t. Any general combination can be factored into a scalar growth term and a variance-preserving (VP) combination. Define the signal-to-noise fraction λ and growth term τ

$$
\lambda ( t ) ^ { 2 } : = \frac { \alpha ( t ) ^ { 2 } } { \alpha ( t ) ^ { 2 } + \sigma ( t ) ^ { 2 } } , \qquad \tau ( t ) : = \sqrt { \alpha ( t ) ^ { 2 } + \sigma ( t ) ^ { 2 } } ,
$$

so that

$$
\alpha ( t ) X + \sigma ( t ) Z = \tau ( t ) \Bigl ( \lambda ( t ) X + \sqrt { 1 - \lambda ( t ) ^ { 2 } } Z \Bigr ) .
$$

The VP case is $\tau \equiv 1$ . The VP scaling uniquely identifies the probability distribution of the random variable induced by a convolution with a normal, any scaling does not change the “shape” of the density but simply the magnitude of the random variable.

Score-based difusion and stochastic localization are defined by two diferent choices of α and σ.

## B.2.1 Score-Based Difusion from the Ornstein–Uhlenbeck Process

Taking $\alpha ( t ) = e ^ { - t } , \sigma ( t ) = \sqrt { 1 - e ^ { - 2 t } }$ gives $\dot { \alpha } / \alpha = - 1$ and $2 \sigma ^ { 2 } \kappa = 2$ , so (2) is the Ornstein– Uhlenbeck (OU) process with drift linear in $X _ { t }$ and constant in time, and (3) reads

$$
\begin{array} { l } { { d X _ { t } = - X _ { t } d t + \sqrt { 2 } d B _ { t } , \qquad X _ { 0 } \sim \pi , } } \\ { { d Y _ { t } = \left( Y _ { t } + 2 s _ { T - t } ( Y _ { t } ) \right) d t + \sqrt { 2 } d B _ { t } . } } \end{array}
$$

This is a VP combination, $\tau ( t ) \equiv 1$ , with signal level $\lambda ( t ) ^ { 2 } = e ^ { - 2 t }$ . The reverse drift is implemented by the score alone.

## B.2.2 The Stochastic Localization Time Scaling

Fix $T > 0$ and take $\alpha ( t ) = T - t , \sigma ( t ) = \sqrt { T - t } \mathrm { o n } t \in [ 0 , T ]$ . Then

$$
\frac { \dot { \alpha } ( t ) } { \alpha ( t ) } = - \frac { 1 } { T - t } , \qquad \frac { \dot { \sigma } ( t ) } { \sigma ( t ) } = - \frac { 1 } { 2 ( T - t ) } , \qquad \kappa ( t ) = \frac { 1 } { 2 ( T - t ) } , \qquad \sigma ( t ) ^ { 2 } 2 \kappa ( t ) = 1 ,
$$

so the forward process (2) is

$$
d X _ { t } = - \frac { 1 } { T - t } X _ { t } d t + d B _ { t } , \qquad X _ { 0 } = T X + \sqrt { T } Z ,
$$

which is driven to $X _ { T } = 0$ . In the reverse process (4) the coeficient of $Y _ { t }$ vanishes, $\dot { \alpha } / \alpha + 2 \kappa = 0$ and $2 \kappa \alpha = 1$ , so the reverse drift is exactly the denoiser:

$$
d Y _ { t } = E \big [ X \big | t X + \sqrt { t } Z = Y _ { t } \big ] d t + d B _ { t } , \qquad Y _ { 0 } = 0 .
$$

This is the stochastic localization process of Montanari [23], with $Y _ { t } \overset { d } { = } t X + \sqrt { t } Z$ and $Y _ { t } / t \to X$ as $t \to \infty$

Relation to the OU difusion. Both processes are naturally indexed by their reverse-time clocks, along which the signal level increases. In stochastic localization time s, the signal level and growth are

$$
\lambda _ { \mathrm { S L } } ( s ) ^ { 2 } = \frac { s } { 1 + s } , \qquad \tau _ { \mathrm { S L } } ( s ) = \sqrt { s ( 1 + s ) } .
$$

For the OU difusion on a horizon $[ 0 , T ]$ , write $Y _ { r } ^ { \mathrm { O U } } \sim X _ { T - r }$ for the reverse process at reverse time $r \in [ 0 , T ]$ , so that

$$
\lambda _ { \mathrm { O U } } ( r ) ^ { 2 } = e ^ { - 2 ( T - r ) } , \qquad \tau _ { \mathrm { O U } } = 1 .
$$

Matching signal levels, $e ^ { - 2 ( T - r ) } = s / ( 1 + s )$ , gives the time change

$$
\begin{array} { r } { r ( s ) = T - \frac { 1 } { 2 } \log { \left( 1 + 1 / s \right) } , } \end{array}
$$

which is increasing in s and maps $s \in [ ( e ^ { 2 T } - 1 ) ^ { - 1 } , \infty )$ onto $[ 0 , T )$ ; as $T \to \infty$ every $s > 0$ is covered. Therefore

$$
Y _ { s } ^ { \mathrm { S L } } \ \stackrel { d } { = } \ \sqrt { s ( 1 + s ) } \ Y _ { r ( s ) } ^ { \mathrm { O U } } .
$$

Stochastic localization is thus the OU reverse difusion run under the time change $r ( s )$ and rescaled in space by $\tau _ { \mathrm { S L } } ( s )$ : the marginal density produced at reverse time s of stochastic localization is the marginal produced at reverse time $r ( s )$ of the score-based reverse difusion, scaled by $\sqrt { s ( 1 + s ) }$ An example of the equivalent marginals produced for a simple mixture of normals is shown in Figure 6 to visualize the equivalence.

$$
Y _ { r } = e ^ { - ( T - r ) } X + { \sqrt { 1 - e ^ { - 2 ( T - r ) } } } Z , T { = } 2
$$

![](images/7f1f3fee7e63a688dfc0af6cb6fa1d4d19e18aaf75203189c24e25261cdb09a8.jpg)  
Figure 6: Visualization of score-based difusion and stochastic localization reverse flow path for an asymmetric mixture of normals target density in one dimension. Marginal densities produced at each stage of the processes are equivalent under a time shift and rescaling.

## B.3 Log-Odds Calculation

All tilts used in the experiments are linear tilts of the extended base density by a vector supported on the reward coordinates. A linear tilt reweights π between reward classes without altering the conditional distribution within a class, and the resulting class probabilities have a closed form in τ . This provides an exact reference value at every tilt specifying the probability of producing a reward-achieving solution at that tilt strength.

Single reward bit. Let $r : \{ - 1 , 1 \} ^ { d }  \{ - 1 , 1 \} , A = \{ x : r ( x ) = 1 \}$ , and $p = \pi ( A )$ . The reward tilt is the linear tilt by $\tau e _ { * }$ in the extended state space, and

$$
q _ { \tau } ( x ) = { \frac { \pi ( x ) e ^ { \tau r ( x ) } } { E _ { \pi } [ e ^ { \tau r ( X ) } ] } } = \pi ( x \mid A ) { \frac { p e ^ { \tau } } { p e ^ { \tau } + ( 1 - p ) e ^ { - \tau } } } + \pi ( x \mid A ^ { c } ) { \frac { ( 1 - p ) e ^ { - \tau } } { p e ^ { \tau } + ( 1 - p ) e ^ { - \tau } } } .
$$

The conditional laws $\pi ( \cdot \mid A )$ and $\pi ( \cdot \mid A ^ { c } )$ are unchanged. With logit $\begin{array} { r } { p = \log { \frac { p } { 1 - p } } } \end{array}$ and $p _ { \tau } = q _ { \tau } ( A )$ ),

$$
\mathrm { l o g i t } p _ { \tau } = \mathrm { l o g i t } p + 2 \tau .\tag{5}
$$

The slope 2 is the diference $r ( x ) - r ( x ^ { \prime } ) = 2$ between the two reward values. In the Ising and Sudoku experiments p is the fraction of reward-achieving examples in the training set.

One-hot class block. For K classes the reward is a block of K coordinates and class c is encoded by $r _ { c } \in \{ - 1 , 1 \} ^ { K } , + 1$ in position c and −1 elsewhere. The tilt $h = \tau r _ { c }$ assigns class k the weight $\exp ( \tau \langle r _ { c } , r _ { k } \rangle )$ , with $\langle r _ { c } , r _ { c } \rangle = K$ and $\langle r _ { c } , r _ { k } \rangle = K - 4$ for $k \neq c ,$ since the two patterns difer in exactly two coordinates. Canceling the common factor $e ^ { ( K - 4 ) \tau }$

$$
\pi _ { \tau } ( c ) = \frac { \pi ( c ) e ^ { 4 \tau } } { \pi ( c ) e ^ { 4 \tau } + 1 - \pi ( c ) } , \qquad \pi _ { \tau } ( k ) = \frac { \pi ( k ) } { \pi ( c ) e ^ { 4 \tau } + 1 - \pi ( c ) } , \quad k \neq c ,
$$

so that

$$
\mathrm { l o g i t } \pi _ { \tau } ( c ) = \mathrm { l o g i t } \pi ( c ) + 4 \tau ,\tag{6}
$$

and $\pi _ { \tau } ( k ) / \pi _ { \tau } ( k ^ { \prime } ) = \pi ( k ) / \pi ( k ^ { \prime } )$ for $k , k ^ { \prime } \neq c .$ The slope 4 is the gap $\left. r _ { c } , r _ { c } \right. - \left. r _ { c } , r _ { k } \right.$ . Tilting along the single axis $e _ { c }$ yields slope 2 with the same proportionality among non-target classes. In general, a tilt h supported on the reward block shifts the log-odds of class c against class k by $\langle h , r _ { c } - r _ { k } \rangle$ (5) and (6) are the two cases used.

Tilt grid. Inverting (5), the tilt strength at which reward is achieved with probability $p ^ { \star }$ is

$$
\begin{array} { r } { \tau ( p ^ { \star } ) = \frac { 1 } { 2 } \big ( \log \mathrm { i t } p ^ { \star } - \log \mathrm { i t } p \big ) , } \end{array}
$$

with $\textstyle { \frac { 1 } { 4 } }$ in place of $\frac { 1 } { 2 }$ under (6). Experiments fix $\tau _ { 9 8 } = \tau ( 0 . 9 8 ) ~ ( \tau _ { 9 5 }$ for MNIST, where the classifier ceiling limits resolution) and evaluate five tilt strengths below $\tau _ { 9 8 }$ and five above, up to the maximum tilt strength $\sqrt { d + 1 }$

Estimation. At each $\tau , N$ samples are drawn $( N = 2 0 0 0$ unless stated), decoded, and graded by the reward function, or by the scoring classifier for MNIST. The observed fraction $\hat { p } _ { \tau }$ is reported on the logit scale against (5) or (6), with 95% Clopper–Pearson binomial intervals mapped through the logit. Near $p _ { \tau } = 1$ the logit is steep: intervals narrow in probability are wide in log-odds and often include $\hat { p } = 1$ as an upper endpoint, and the largest log-odds resolvable at sample size N is of order $\log \mathrm { i t } ( 1 - 1 / N )$ , approximately 7.6 for $N = 2 0 0 0$ Points whose exact value exceeds this are uninformative regarding under-sampling. Conversely, a discrepancy in log-odds at high τ may correspond to a diference of less than one percentage point in reward rate. For MNIST the exact rate is passed through the classifier’s per-class sensitivity and false-positive rate prior to comparison, as described in Appendix C.

Clopper–Pearson Confidence Intervals. Let $K \sim \mathrm { B i n o m i a l } ( N , p _ { \tau } )$ be the number of rewarded samples at a given τ , with observed value k and $\hat { p } _ { \tau } = k / N$ . The Clopper–Pearson interval is obtained by inverting two one-sided exact binomial tests. Its endpoints are the values of $p$ at which the observed count would just be rejected at level $\alpha / 2$ in each tail:

$$
\sum _ { j = k } ^ { N } { \binom { N } { j } } p _ { L } ^ { j } ( 1 - p _ { L } ) ^ { N - j } = { \frac { \alpha } { 2 } } , \qquad \sum _ { j = 0 } ^ { k } { \binom { N } { j } } p _ { U } ^ { j } ( 1 - p _ { U } ) ^ { N - j } = { \frac { \alpha } { 2 } } .
$$

By the identity linking binomial tails to the regularized incomplete beta function, these endpoints have the closed form

$$
p _ { L } = B ^ { - 1 } \bigl ( { \textstyle { \frac { \alpha } { 2 } } } ; k , N - k + 1 \bigr ) , \qquad p _ { U } = B ^ { - 1 } \bigl ( 1 - { \textstyle { \frac { \alpha } { 2 } } } ; k + 1 , N - k \bigr ) ,
$$

where $B ^ { - 1 } ( q ; a , b )$ is the q-quantile of a $\mathrm { B e t a } ( a , b )$ distribution, with the conventions $p _ { L } = 0$ when $k = 0$ and $p _ { U } = 1$ when $k = N$ . Because the interval comes from exact tail probabilities and not from a normal approximation, its coverage is at least $1 - \alpha$ for every $p _ { \tau }$ and every N. This property is useful here, since many of our estimates lie close to the boundary $p _ { \tau } ~ = ~ 1$ , where Wald-type intervals are badly anti-conservative or collapse to zero width. Coverage is in general strictly greater than $1 - \alpha$ , so the intervals are conservative. The logit is strictly increasing, so [logit $( p _ { L } ) , \log \mathrm { i t } ( p _ { U } ) ]$ is a valid $1 - \alpha$ interval for the log-odds. When every sample is rewarded $( k = N )$ , the interval becomes $[ ( \alpha / 2 ) ^ { 1 / N } , 1 ]$ . Its upper endpoint in log-odds is then $+ \infty$ , and its lower endpoint is logit $( ( \alpha / 2 ) ^ { 1 / N } ) \approx 6 . 3$ for $N = 2 0 0 0$ and $\alpha = 0 . 0 5$

Model-reported reward. Since the reward coordinate is denoised jointly with the token coordinates, the sampler also produces an internal prediction of reward: the sign of the terminal $d + 1$ coordinate, or the argmax of the class block. For a correctly sampling model this prediction and the externally graded reward agree in expectation. Comparing the two curves against the exact line separates two failure modes: a gap between the internal prediction and the exact line indicates the sampler does not produce the tilted marginal on the reward coordinate; a gap between the internal prediction and the graded reward indicates the token coordinates are inconsistent with the reward coordinate produced. The 9x9 Sudoku two-pass and carry results in Appendix C are distinguished by this comparison.

## B.4 Self-Conditioning

Self-conditioning as originally introduced in Chen et al. [7] is a method to improve the sample quality of a denoising difusion model. It was originally derived as a mostly ad hoc engineering decision, rather than a method arising from sound theory. Its interaction with standard theory for error control for difusion models has not yet been fully understood [30, 18, 6, 5, 3, 9, 34]. In this paper we consider two forms of self-conditioning: one method we call mode “two-pass” is a Markovian form of self-conditioning, it uses only information available at a given time point of a discrete difusion to estimate the denoiser as in Algorithm 6 in Chen et al. [7]. This method can be considered an architecture choice for the denoiser estimator network and is compatible with existing difusion model error control theory which views the reverse process as a Euler-Maruyama discretization of the continuous SDE with pointwise denoiser estimation.

The second method of self-conditioning we call mode “carry” is when the denoiser reuses its previous estimate at time $t - 1$ to make its denoiser estimate at time t. This is the more common method implemented in modern denoising difusion models, however it breaks the memory-one Markov structure under which difusion model error control theory is currently understood.

The denoiser is a conditional expectation on the origin point of a noisy observation given partial information provided by the noised observation,

$$
F ( u , a ) = E _ { \pi } [ X | U = a X + \sqrt { 1 - a ^ { 2 } } Z ] .
$$

If we had some additional side information W from some other random variable, this could produce a more precise conditional estimate of the origin point X

$$
F ( u , w , a ) = E _ { \pi } [ X | U = a X + \sqrt { 1 - a ^ { 2 } } Z , W ] .
$$

A self-conditioned network takes the form $g _ { \theta } ( u , w , a )$ , where $( u , a )$ are the noisy input and signal level and $w \in [ - 1 , 1 ] ^ { d + 1 }$ is a prior estimate of x. The input $w = 0$ is equidistant from all points of $\{ - 1 , 1 \} ^ { d + 1 }$ and encodes the absence of side information, so that $g _ { \theta } ( u , 0 , a )$ is an estimate of $G ( u , a )$ In Algorithms 1 and $2 , ( u , a )$ is the renormalized pair $( \boldsymbol { v } ^ { * } , \boldsymbol { s } ^ { * } )$

In the training algorithm, a self-conditioned network is trained by some portion of the time being fed $w = 0$ as its second argument, and some portion of the time begin fed its own no side information estimate as its second argument $w = g _ { \boldsymbol { \theta } } ( u , 0 , a )$ . This modifies the single step of batch SGD training algorithm in lines 13 and 14 in Algorithm 3.

Both self-conditioning modes “two-pass” and “carry” are trained in the same way, and difer in their implementation as a sampling algorithm.

Two-pass is the simpler method, as in sampling it reuses its own estimate with no side information as a second input to a second pass evaluation. This can be viewed as an architecture choice for a larger network that is ultimately still a function of a single pair $( u , a )$ alone,

$$
\begin{array} { r } { \tilde { g } _ { \theta } ( u , a ) = g _ { \theta } ( u , g _ { \theta } ( u , 0 , a ) , a ) , } \end{array}
$$

Using $\tilde { g }$ in place of $g$ in line 8 of Algorithm 2 is the sampling algorithm implemented by two-pass.

The mode carry reuses the previous iteration’s denoiser estimate as its second argument in selfconditioning. Initialize $\hat { G } _ { - 1 } = 0$ , and let $\hat { G } _ { k }$ be the denoiser estimate at iteration k. The denoiser estimates are defined recursively as

$$
\begin{array} { c } { { \hat { G } _ { k } = g _ { \theta } \big ( v _ { k } ^ { * } , \hat { G } _ { k - 1 } , s _ { k } ^ { * } \big ) , } } \\ { { { } } } \\ { { X _ { t _ { k + 1 } } = X _ { t _ { k } } + \delta \hat { G } _ { k } + \sqrt { \delta } \xi _ { k } , } } \end{array}
$$

This modifies the stochastic localization discrete sampling algorithm as shown in Algorithm 4.

Algorithm 3 Batch SGD step for the self-conditioned tilt–mean network $g _ { \theta } ( u , w , a )$   
Require: Data $\{ X _ { i } \} \subset \{ - 1 , 1 \} ^ { d + 1 }$ (reward in coordinate $d + 1 )$ ; batch size $N ;$ step size $\gamma ;$ self  
conditioning probability $p _ { \mathrm { s c } }$   
1: $\mathcal { L }  0$   
2: for $n = 1 , \ldots , N$ do   
3: Draw $x \sim \{ X _ { i } \} , \ Z \sim { \mathcal { N } } ( 0 , I _ { d + 1 } ) , \ \beta \sim \operatorname { L o g U n i f } [ \beta _ { \operatorname* { m i n } } , \beta _ { \operatorname* { m a x } } ] , \ a  a ( \beta )$   
4: W.p. $p _ { 0 } \colon q \gets \mathbf { 1 } _ { d } \colon$ else draw $k \sim \mathrm { U n i f } \{ 2 , \dots , d \}$ and set $q  \sqrt { d / k } 1 _ { S }$ for a random k-subset   
S   
5: $\mathrm { W . p . } p _ { \lambda } \colon \lambda  1 ;$ <sub>else</sub> <sub>λ</sub> <sub>∼</sub> <sub>LogUnif[1,</sub> √<sub>d</sub> <sub>+</sub> <sub>1]</sub>   
6: $\begin{array} { r } { \Lambda _ { j , j }  \sqrt { 1 + \frac { 1 } { d } - \frac { \lambda ^ { 2 } } { d } q _ { j } } , \ j \leq d ; \quad \Lambda _ { d + 1 , d + 1 }  \lambda } \end{array}$   
7: $\eta \gets a \Lambda ^ { 2 } x + \sqrt { 1 - a ^ { 2 } } \Lambda Z$ ▷ tilt vector   
8: if ${ \boldsymbol { q } } = \mathbf { 1 } _ { d }$ and $\lambda = 1$ then   
9: $( v ^ { * } , s ^ { * } )  ( \eta , ~ a )$ ▷ basic VP noising   
10: else   
11: $\begin{array} { r } { C \gets \beta ( a ) \frac { \| \eta \| } { \sqrt { d + 1 } } , \quad s ^ { * } \gets \frac { 1 } { 2 } \Big ( \sqrt { \frac { 1 } { C ^ { 2 } } + 4 } - \frac { 1 } { C } \Big ) , \quad v ^ { * } \gets \sqrt { d + 1 } \eta / \| \eta \| } \end{array}$   
12: $\mathrm { W . p . ~ } 1 - p _ { \mathrm { s c } } \colon w  0$ ▷ no side information   
else $w  \mathrm { s g } [ g _ { \boldsymbol { \theta } } ( v ^ { * } , 0 , s ^ { * } ) ]$ ▷ own first-pass estimate, no gradient   
13: $\mathcal { L }  \mathcal { L } + \| x - g _ { \boldsymbol { \theta } } ( v ^ { * } , w , s ^ { * } ) \| ^ { 2 }$   
14: $\theta  \theta - \gamma \nabla _ { \theta } \mathcal { L }$

## Implementation

Ising Model and 4x4 Sudoku Target distributions for the Ising Model and 4x4 Sudoku mode have state spaces of size $2 ^ { 1 3 }$ and $2 ^ { 3 3 }$ , small relative to the other target distributions. Self-conditioning was not implemented for these problems and the denoiser was trained using Algorithm 1 and stochastic localization implemented using Algorithm 2.

6x6 Sudoku For the 6x6 Sudoku self-conditioning mode “two-pass” was implemented and the sampler tracked performance well across log-odds levels considering the confidence intervals in Figure 18.

9x9 Sudoku The 9x9 Sudoku problem, both modes of self-conditioning were implemented and saw difering results.

Mode two-pass saturated its probability of producing proper puzzles at 60% and did not increase at higher tilt strengths. The model also saw significant diferences between the model’s own prediction of if it was producing a reward in its d + 1 coordinate and actual reward given to the token string produced by the reward function, seen in Figure 20a.

Mode carry tracked the log-odds better and saturated at 90% reward-achieving solution, showing less mismatch between its prediction of reward and the actual reward of the token string produced, see Figure 20b.

MNIST Digits For the MNIST digit image generation problem, both modes of self-conditioning were implemented. In this problem mode two-pass follows the log-odds curve better than the carry mode. Mode carry over-samples the reward-achieving solutions at medium strength tilts as seen in Figure 27. As a conjecture of what may be the cause, carry self-conditioning may cause the model to become too confident in its early predictions, continuing to refine upon estimates made early in the process while two-pass is a more faithful implementation of a proper Euler-Maruyama discretization of the continuous-time SDE which defines stochastic localization.

Algorithm 4 Stochastic-localization sampling from the tilted density $\mu \propto \pi e ^ { \left. \nu , \cdot \right. }$ with carry-mode   
self-conditioning   
Require: Trained self-conditioned network $g _ { \theta } ( u , w , a )$ ; tilt vector $\nu = \tau _ { \mathrm { i n f l l } } ( e _ { A ^ { + } } - e _ { A ^ { - } } ) + \tau _ { \mathrm { r e w a r d } } e _ { * } ;$   
time grid $0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { K } = T$   
1: $X _ { t _ { 0 } } \gets 0 , \quad \hat { G } _ { - 1 } \gets 0$   
2: for $k = 0 , \ldots , K - 1$ do   
3: $\delta \gets t _ { k + 1 } - t _ { k }$   
4: $\eta  \nu + X _ { t _ { k } }$   
5: $\begin{array} { r } { C \gets \frac { \lVert \boldsymbol \eta \rVert } { \sqrt { d + 1 } } , \quad s ^ { * } \gets \frac { 1 } { 2 } \Big ( \sqrt { \frac { 1 } { C ^ { 2 } } + 4 } - \frac { 1 } { C } \Big ) , \quad v ^ { * } \gets \sqrt { d + 1 } \eta / \lVert \eta \rVert } \end{array}$   
6: $\hat { G } _ { k } \gets g _ { \theta } ( v ^ { * } , \hat { G } _ { k - 1 } , s ^ { * } )$ ▷ previous estimate as side information   
7: Draw $\xi \sim \mathcal { N } ( 0 , I _ { d + 1 } )$   
8: $X _ { t _ { k + 1 } } \gets X _ { t _ { k } } + \delta \hat { G } _ { k } + \sqrt { \delta } \xi$   
9: $\eta  \nu + X _ { T }$ compute $( \boldsymbol { v } ^ { * } , \boldsymbol { s } ^ { * } )$ from η as above   
10: return $\hat { x } \gets \mathrm { s i g n } \Big ( g _ { \theta } \big ( v ^ { * } , g _ { \theta } \big ( v ^ { * } , 0 , s ^ { * } \big ) , s ^ { * } \big ) \Big )$ ▷ two-pass final decode

## B.5 Discrete Step Size Scheduling

The stochastic localization discretization in Algorithm 2 requires the sequence of discrete time steps $\{ t _ { k } \} _ { k = 0 } ^ { K }$ to be specified as an input. Here we describe for an input time frame $[ 0 , T ]$ , discretization level K, target tilt vector ν, and dimension d how we construct the $t _ { k }$ schedule.

The learned denoiser $g _ { \boldsymbol \theta } ( \boldsymbol { v } ^ { * } , \boldsymbol { s } ^ { * } )$ depends on the sampling time t only through the equivalent signal level $s ^ { * }$ which is determined by the norm of the $\eta = X _ { t } + \nu$ vector with the applied shift. Thus there is a non-linear relationship between how the $t _ { k }$ scaling interacts with the $s ^ { * }$ values the denoiser will be evaluated at.

Recall that at time t Algorithm 2 queries the network at $\eta = \nu + X _ { t }$ with

$$
C = { \frac { \| \eta \| } { \sqrt { d + 1 } } } , \qquad s ^ { * } = \beta ^ { - 1 } ( C ) = { \frac { 1 } { 2 } } { \Big ( } { \sqrt { { \frac { 1 } { C ^ { 2 } } } + 4 } } - { \frac { 1 } { C } } { \Big ) } , \qquad \beta ( s ) = { \frac { s } { 1 - s ^ { 2 } } } .
$$

Since $\| \eta \|$ is random, we replace it by a deterministic proxy. Under the theoretical proper sampling marginal, $X _ { t } = t X + { \sqrt { t } } Z$ with $\| X \| ^ { 2 } = d + 1$ , we have $E \| X _ { t } \| ^ { 2 } = ( d + 1 ) ( t ^ { 2 } + t )$ . Considering $E \| X _ { t } + \nu \| ^ { 2 }$ and expanding the norm, the cross term $\sqrt { t } \langle \nu , Z \rangle$ has mean zero, and we evaluate the remaining cross term at its aligned value $\langle \nu , X \rangle = \| \nu \|$ , which corresponds to the tilt succeeding. This gives

$$
\bar { C } ( t ) = \sqrt { \frac { \| \nu \| ^ { 2 } + 2 t \| \nu \| + ( d + 1 ) ( t ^ { 2 } + t ) } { d + 1 } } , \qquad \bar { s } ( t ) = \beta ^ { - 1 } ( \bar { C } ( t ) ) ,
$$

this is an approximation but gives a deterministic function of t for the anticipated ratio of the norm of η to $\sqrt { d + 1 }$ for the purposes of designing a time schedule. s¯ is strictly increasing in t. Setting $\begin{array} { r } { s _ { k } = \bar { s } ( 0 ) + \frac { k } { K } ( \bar { s } ( T ) - \bar { s } ( 0 ) ) } \end{array}$ for $k = 0 , \ldots , K$ , the grid is $t _ { k } = \bar { s } ^ { - 1 } ( s _ { k } )$ . The inverse is available in closed form, since $\bar { C } ( t ) ^ { 2 }$ is quadratic in t: with $C _ { k } = \beta ( s _ { k } )$

$$
t _ { k } = \frac { - ( d + 1 + 2 \| \nu \| ) + \sqrt { ( d + 1 + 2 \| \nu \| ) ^ { 2 } - 4 ( d + 1 ) ( \| \nu \| ^ { 2 } - ( d + 1 ) C _ { k } ^ { 2 } ) } } { 2 ( d + 1 ) } ,
$$

with the endpoints fixed at $t _ { 0 } = 0$ and $t _ { K } = T$ . The grid is deterministic, is computed once before sampling, and depends on the tilt only through $\| \nu \|$ ; for $\nu = 0$ it starts from $\bar { s } ( 0 ) = 0$ . The resulting

steps are finest near t = 0 and grow monotonically towards T. All experiments use this schedule with $T = 2 0$ and $K = 1 0 0 0$

Algorithm 5 Time grid equally spaced in the queried signal level $\overline { { s ^ { * } } }$   
Require: Horizon $T ;$ number of steps $K ;$ dimension $d ;$ tilt norm $n = \| \nu \| ( n = 0$ if untilted)   
1: $D \gets d + 1$   
2: Define $\bar { C } ( t ) = \sqrt { \big ( n ^ { 2 } + 2 t n + D ( t ^ { 2 } + t ) \big ) / D }$ ▷ proxy for $\| \nu + X _ { t } \| / \sqrt { d + 1 }$   
3: Define $\begin{array} { r } { \beta ^ { - 1 } ( C ) = \frac 1 2 \left( \sqrt { \frac { 1 } { C ^ { 2 } } + 4 } - \frac { 1 } { C } \right) } \end{array}$ for $C > 0 .$ , and $\beta ^ { - 1 } ( 0 ) = 0$   
4: $s _ { \mathrm { l o } } \gets \beta ^ { - 1 } ( \bar { C } ( 0 ) ) , ~ s _ { \mathrm { h i } } \gets \beta ^ { - 1 } ( \bar { C } ( T ) )$   
5: for $k = 0 , \ldots , K$ do   
6: $s _ { k }  s _ { \mathrm { l o } } + \frac { k } { K } ( s _ { \mathrm { h i } } - s _ { \mathrm { l o } } )$ ▷ uniform in $s ^ { * }$   
7: $C _ { k } \gets s _ { k } / ( 1 - s _ { k } ^ { 2 } )$ ▷ $C _ { k } = \beta ( s _ { k } )$   
$- ( D + 2 n ) + \sqrt { ( D + 2 n ) ^ { 2 } - 4 D \left( n ^ { 2 } - D C _ { k } ^ { 2 } \right) }$   
8: t<sub>k</sub> ← ▷ solves $\bar { C } ( t _ { k } ) = C _ { k }$   
2D   
9: $t _ { 0 } \gets 0 , \quad t _ { K } \gets T$ ▷ fix endpoints against rounding   
10: return $( t _ { 0 } , t _ { 1 } , \dots , t _ { K } )$

## C ADDITIONAL EXPERIMENTS INFORMATION

Code for all experiments, including data generation, training, and evaluation is available at the the GitHub link:

https://github.com/CMcDonald-1/internal\_guidance\_diffusion\_models The datasets and trained checkpoints can be downloaded from the link: https://osf.io/kewg9/

## C.1 12-Dimensional Ising Model

Target Density:

The target density is on $\{ - 1 , 1 \} ^ { d }$ with $d = 1 2$ , with pmf proportional to

$$
\pi ( x ) \propto e ^ { { \frac { 1 } { 2 } } x ^ { T } J x + \langle s , x \rangle } .
$$

J is a Wigner (GOE-like) coupling matrix, generated by drawing a full $d \times d$ matrix of i.i.d. standard normals scaled by ${ J _ { 0 } } / { \sqrt { d } }$ with $J _ { 0 } = 1 . 5$ , retaining the strict upper triangle, and mirroring it onto the lower triangle so that $J = J ^ { T }$ The diagonal is set to zero. The s vector has i.i.d. Gaussian entries with standard deviation $\sigma = 0 . 3$

The reward is the non-linear indicator function

$$
r ( x ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { e x a c t l y ~ 1 0 ~ o f ~ t h e ~ 1 2 ~ c o r d i n a t e s ~ p o s i t i v e } } } \\ { - 1 } & { { \mathrm { e l s e } } } \end{array} \right. }
$$

equivalently $r ( x ) = + 1$ if $\langle \mathbf { 1 } , x \rangle = 8$ . We append the reward as a 13th coordinate in the extended state space $\{ - 1 , 1 \} ^ { 1 3 }$ , which turns the non-linear reward tilt $q _ { \tau } \propto \pi e ^ { \tau r ( x ) }$ into the linear tilt $q _ { \tau } \propto \pi ( x , x ^ { * } ) e ^ { \tau x ^ { * } }$

Training Set:

The training set is 100,000 samples from the base distribution with reward information. 4.41% achieve reward, which is the base probability of reward under the target distribution.

## Model Architecture:

The operator network is a residual MLP with a mean-field skip connection,

$$
G _ { \theta } ( u , a ) = \mathrm { t a n h } \big ( \beta ( a ) u + g _ { \theta } ( a ) \odot r _ { \theta } ( u , a ) \big ) ,
$$

where $\beta ( a ) u$ is the exact operator for a uniform base distribution, so the residual stack only has to learn the correlation correction, and the tanh enforces the $[ - 1 , 1 ] ^ { 1 3 }$ box by construction. The signal level enters only through a 10-dimensional level embedding $e ( a ) = [ a _ { : }$ , log(1 + $\beta ( a ) )$ , sin(2πaf), cos $\left( 2 \pi a f \right) ]$ with Fourier frequencies $f \in \{ 1 , 2 , 4 , 8 \}$ . The input $u \in \mathbb { R } ^ { 1 3 }$ is scaled by $1 / { \sqrt { d } } ,$ concatenated with $e ( a )$ , and mapped by a linear layer to width 96. This is followed by 4 pre-activation residual blocks, each consisting of LayerNorm followed by FiLM modulation $( 1 + g _ { k } ) \mathrm { L N } ( h ) + b _ { k }$ with $( g _ { k } , b _ { k } )$ predicted from $e ( a )$ , then two width-96 linear layers with SiLU activations. A final LayerNorm and linear layer project to the 13 output coordinates, and a separate gain head $( 1 0  3 2  1 3$ , SiLU) predicts the per-coordinate, level-dependent gain $g _ { \theta } ( a )$ . The output layer is zero-initialized, so training begins exactly at the mean-field prediction. The model has 88,254 parameters. Full implementation details are in the supplementary code.

## Training Run:

The model was trained for 30,000 Adam steps with batch size 1024, learning rate $1 0 ^ { - 3 }$ , and a cosine-annealing learning rate schedule over the full 30,000 steps. Each batch re-noises freshly drawn training samples rather than using a fixed noise realization: $\beta$ is drawn log-uniformly on [0.02, 50], and the Λ-augmentation is applied with $\lambda \sim \mathrm { U n i f } ( 1 , 3 . 5 )$ . The Λ profile is applied to every training row, with λ drawn independently per row; no infill-focused (k-sparse) profiles were used in this run. The objective is the unweighted per-coordinate squared error $\mathbb { E } \Vert X - G _ { \theta } ( V , a ) \Vert ^ { 2 }$ summed over the 13 coordinates.

Training ran on a single NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell, 5888 CUDA cores, 12.8 GB, CUDA 12.9) and took 10.7 minutes of wall-clock time (0.18 GPU-hours). Processing $3 0 , 0 0 0 \times 1 0 2 4 \approx 3 . 0 7 \times 1 0 ^ { 7 }$ samples through an 88,254-parameter network corresponds to approximately $1 . 6 \times 1 0 ^ { 1 3 }$ training FLOPs under the standard 6ND estimate.

## Extended Results

For all models in question, τ tilt strengths tested are capped at $\sqrt { d + 1 }$ , as this is the maximum amount of Λ focusing seen in the training runs. Note this tilt strength is still very strong producing near 100% probability of reward for the problems analyzed. In Figure 7, we note the untilted model matches the mean and covariance of the target Ising model, efectively recreating the first and second moments.

For various signal levels a and various tilt strengths τ, 100 samples are generated for a pair of $( u , a )$ and average squared Euclidean distance between network denoiser and ground truth denoiser is computed in Figure 8. Error does increase with tilt strength, as evaluations of tilted densities are out-of-distribution evaluations of the base denoiser only learned via Λ focusing. However, there is still a uniform bound across tilt strengths and signal levels of 0.12 which can be converted into a bound on terminal KL error at the final sample point across all τ strengths in the tested range of 0 to 3.6.

The probability of reward and log-odds plot is seen in Figure 9 across a range of $\tau$ from 0 to $\sqrt { d + 1 }$ . At 2000 samples per $\tau$ the model tracks the log-odds perfectly.

The 12-dimensional Ising model is a small toy model with exact enumeration of the ground truth denoiser to demonstrate highly accurate sampling across tilts and denoiser learning when a ground truth is available.

Ising d=13 untilted covariance, n=10000 (SSR=0.0134, Frob=0.1159)  
![](images/88e2a11e3d83fa168dc7689f77b5bdcc0f08e4848fdc55faf70a191f15da51bc.jpg)  
Figure 7: Ising model covariance matrix ground truth and network sampled.

Ising d=13 denoiser error vs. a, by reward-tilt tau (n=100/point)  
![](images/a73a3945232b47a703a6a53b78dad53ac8aebe749aa86ec6a34d9507f55ac406.jpg)  
Figure 8: Comparison of $L ^ { 2 }$ error of learned denoiser for Ising model across tilt strengths.

![](images/01f36089942cabccadb430cf6c78b719045c8ea2c83510d9bc67b5b9f896cae0.jpg)

![](images/ffac6ec17f007f48f62c15e58d33c7c36cbef7d23a8c4d1bed7921e7575d3dd0.jpg)  
Figure 9: Ising model log-odds across tilt strengths.

## C.2 4x4 Sudoku

## Target Density

4×4 Sudoku has a state space of size $2 ^ { 3 2 }$ , with only 288 valid puzzles which satisfy row, column and box constraints. Each cell is a 2-bit encoding of the numbers 1–4 with no dead code words, so the 16 cells give exactly 32 bits and every bit string decodes to a grid.

<table><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td></tr></table>

Figure 10: Example valid 4x4 grid.

The base distribution is a weak generator sampling each row independently. With probability $1 - \rho$ it makes the row a uniformly random permutation of the numbers 1–4, and thus a valid row; with probability $\rho$ it instead draws the row’s 8 bits uniformly at random, giving any of the 256 possible rows. We use $\rho = 0 . 0 2$

A single row is therefore valid with probability $\begin{array} { r } { ( 1 - \rho ) + \rho \cdot \frac { 2 4 } { 2 5 6 } = 9 8 . 1 9 \% } \end{array}$ , and all four rows are simultaneously valid 92.94% of the time. The generator has no knowledge of column or box constraints, however, so it produces a proper puzzle only 0.081% of the time by chance.

The reward is the indicator of a valid puzzle appended as a 33rd coordinate, giving the extended state space $\{ - 1 , 1 \} ^ { 3 3 }$

Training Set

Because a proper puzzle arises by chance only 0.081% of the time, the base distribution is mixed with an injection component: each sample is drawn from the uniform distribution over the 288 valid solutions with probability $p _ { \mathrm { i n j e c t } } = 0 . 0 5$ , and from the weak generator otherwise. The training set consists of 150,000 samples drawn from this mixture, of which in expectation 142,500 come from the weak generator and 7,500 are injected proper Sudoku puzzles. The resulting exact reward probability is 5.08%; the realized training file contains 7,590 genuinely valid grids (5.06%).

## Model Architecture

The operator network is the same residual MLP with mean-field skip connection used for the Ising experiment,

$$
G _ { \theta } ( u , a ) = \mathrm { t a n h } \big ( \beta ( a ) u + g _ { \theta } ( a ) \odot r _ { \theta } ( u , a ) \big ) ,
$$

scaled up for the larger state space. The signal level enters through a 10-dimensional level embedding $e ( a ) = [ a , \log ( 1 + \beta ( a ) )$ , sin $( 2 \pi a f )$ ， $\cos ( 2 \pi a f ) ]$ with Fourier frequencies $f \in \{ 1 , 2 , 4 , 8 \}$ . The input $u \in \mathbb { R } ^ { 3 3 }$ is scaled by $1 / { \sqrt { d } } .$ , concatenated with $e ( a )$ , and mapped by a linear layer to width 1024, followed by 8 pre-activation residual blocks, each consisting of LayerNorm, FiLM modulation $( 1 + g _ { k } ) \mathrm { L N } ( h ) + b _ { k }$ with $( g _ { k } , b _ { k } )$ predicted from $e ( a )$ , and two width-1024 linear layers with SiLU activations. A final LayerNorm and linear layer project to the 33 output coordinates, and a separate gain head $( 1 0  3 2  3 3$ , SiLU) predicts the per-coordinate gain $g _ { \theta } ( a )$ . The output layer is zero-initialized so training begins exactly at the mean-field prediction. The model has 17,072,582 parameters. Full implementation details are in the supplementary code.

## Training Run

The model was trained for 30,000 Adam steps with batch size 8192, learning rate $1 0 ^ { - 3 }$ , and a cosine-annealing learning rate schedule over the full 30,000 steps. Each batch re-noises freshly drawn training samples, with $\beta$ drawn log-uniformly on [0.02, 50]. Two focusing profiles are mixed per row: with probability 0.25 the row uses the sparse infill profile, in which k is drawn uniformly on $\{ 2 , \ldots , 3 2 \}$ and a random k-subset of the core coordinates is set to $\sqrt { 3 2 / k }$ with the rest zeroed, and otherwise it uses the reward-focusing profile. Independently, the reward-axis stretch λ is set to 1 with probability 0.25 and otherwise drawn log-uniformly on [1, 5]. Both profiles preserve the budget identity $\begin{array} { r } { \sum _ { i } \Lambda _ { i i } ^ { 2 } = d , } \end{array}$ , so the operator being learned is unchanged. The objective is the unweighted per-coordinate squared error $\mathbb { E } \| X - G _ { \theta } ( V , a ) \| ^ { 2 }$ summed over the 33 coordinates.

Training ran on a single NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell, 5888 CUDA cores, 12.8 GB, CUDA 12.9) and took 67.3 minutes of wall-clock time (1.12 GPU-hours). Processing $3 0 , 0 0 0 \times 8 1 9 2 \approx 2 . 4 6 \times 1 0 ^ { 8 }$ samples through a 17,072,582-parameter network corresponds to approximately $2 . 5 \times 1 0 ^ { 1 6 }$ training FLOPs under the standard 6ND estimate.

## Extended Results

For the 4x4 Sudoku problem, the reward set is only 288 valid grids, and the weak generator decomposes into a product measure across rows. As such, the ground truth denoiser can be computed via enumeration at any reward tilt strength. The estimated $L ^ { 2 }$ error across tilt strength is shown in Figure 11. The denoiser error does increase with τ tilt strength, but does have a uniform bound across tilt and signal levels of 0.5.

We show the actual 33 values of the base denoiser $E [ X _ { i } | u , a ]$ and learned network for 5 sampled u values from $\mathrm { ~ a ~ } \tau = 0$ draw and $\mathrm { ~ a ~ } \tau = 4$ draw in Figure 14. Expected values are all between −1 and 1. Note the $\tau = 0$ draws are all noisy versions of non-reward-achieving solutions, and the 33rd coordinate (the reward coordinate) accurately assigns them all a -1. The $5 \tau = 4$ draws are noisy reward-achieving solutions and the 33rd coordinate assigns them all $\mathrm { ~ a ~ } { + 1 }$ value.

The untilted $\tau = 0$ network achieves near 0 error on these examples and is visually indistinguishable. The $\tau = 4$ network is still very good, but is noticeably a bit of on a few coordinates, which agrees with the average increase in squared error seen in Figure 11.

Sudoku 4x4 d=33 reward-tilt vs. tau(p0=5.08%, n=2000/tau, K=1000)

![](images/989e859c91ce81d8d551aa993ed6dbc994cf7ed54301e9d182e15a6adf242c52.jpg)  
Figure 11: Comparison of $L ^ { 2 }$ error of learned denoiser of 4x4 Sudoku problem across tilt strengths.

![](images/865c10be06ca8a51785d535c08c3830ba022afbfae1818667efb55657164f9af.jpg)  
Figure 12: 4x4 Sudoku log-odds across tilt strengths.

For the 4x4 Sudoku problem we have access to the ground truth denoiser and the net estimated denoiser. Thus we can run stochastic localization with exact denoiser evaluations and with network approximated denoiser evaluation and track the error introduced by incorrect denoiser evaluations isolated from discretization error.

To track the log-odds in Figure 12, we compute the τ that would achieve 98% reward if sampled correctly. We then look at 5 τ levels before this value and 5 levels after up to maximum tilt value strength ${ \sqrt { d + 1 } } = 5 . 7 4$ . Note all levels past the dashed line should be achieving reward more than 98% of the time. Confidence intervals are computed using a Clopper-Pearson method which often includes 100% reward as an upper bound.

The localization samples using the ground truth denoiser tracks the log-odds across all levels, even past 98% when considering the confidence intervals from 2000 samples. The network based sampler tracks well up to the 98% level, but then does strictly under-sample past this high bar with the confidence intervals not overlapping the theoretical log-odds. Note when the network based sampler fails to exactly track the log-odds at this high tilt it is still generating over 98% reward-achieving puzzles, but still under-sampling its theoretical value at this tilt.

As an added bonus to the stochastic localization path, the denoiser $E [ X _ { i } | X _ { t } ]$ acts as both the drift term and a prediction of the terminal sign of a given coordinate. As such, if a coordinate ever passes a threshold $| E [ X _ { i } | X _ { t } ] | > \ell ,$ , we can consider that bit “confident” and fix that bit to the sign of the denoiser at that moment. This allows for “early decoding” seeing which bits the network fixes in which order and at what point of the trajectory. Note a bit is allowed to flip signs, all that is decoded is the terminal sign value. Fixed bits determine what tokens (cell values) are still valid among the unfixed bits.

One of the arguments against continuous difusion for discrete targets is that the trajectory remains a continuous value that cannot be interpreted in discrete token space until the terminal decoding step, unlike masked difusion or autoregressive models which build partial discrete token strings along their trajectory.

By using the confidence based bit decoding, we can achieve a similar efect to see what discrete set of tokens the puzzle was considering at each moment in the trajectory and soft decode at any time.

When both bits in a cell are fixed we record the cell value, when one bit is fixed such as $( * , - 1 )$ this means the cell will get either a (1, −1) which is token 2 or $( - 1 - 1 )$ ) which is token 1. We record this with Sudoku “pencil marks” showing what is considered still possible in the cell.

We note the network is not “solving” the puzzle “logically” as a human would by applying row, column, and box constraints in sequence. For example, in the second frozen grid in Figure 13, there is a 1 fixed in the bottom left box. This would rule out all 1’s in that column, but the upper box cells in the same column have not yet ruled out a 1 in that column since the bits for those two cells have not yet reached their own confidence level. This is therefore a sequence of most likely puzzles tracking the most confident bits at each moment in the trajectory.

We call this path a “fuzzy trajectory” tracking the denoiser coordinate values across the signal levels, noting when a coordinate achieves a confidence level more than 0.75 in magnitude, and recording events when a cell gets fixed in Figure 13. Note some cells do become confident but ultimately flip their token value by the end of the run. Note the row 2 column 4 cell flips from a 2 to a 4 by the end of sampling. We run our stochastic localization up to a T value of 20 but the network has mostly decided on its final value much earlier by time $T = 3$ it would seem.

Note there are 288 valid 4x4 Sudoku grids, and the training set contained 7,500 examples so each valid puzzle was seen many times in the training run. therefore, the model should be judged on its ability to generate each puzzle with equal probability in a large collection of generated outputs.

We generate 3000 valid puzzles with the stochastic localization path using the network denoiser and the ground truth and perform a chi-squared test for uniformity. The network based sampler visits all puzzles except for 1, and its maximum visit to one particular puzzle is 25. The ground truth denoiser visits every solution, with minimum visits being 4 and maximum being 21. Visually, they are not that distinct in their visit histogram in Figure 15. However, the ground truth sampler passes a chi-squared test for uniformity, while the network based sampler fails the test. As such, it has learned some preference for certain solutions, although it is not a strong preference.

![](images/c7b9ac7047eb4d1173e0dff837aa3388ff9dda153cb867b8752c246b6080a3c0.jpg)

![](images/00132992ce4adc81942c2287b15a186faefef36f7d3e65d3b858999109438470.jpg)  
Figure 13: Sudoku 4x4 denoiser trajectory and early decoding. Partially complete discrete token sequences fixed when denoiser coordinates pass magnitude threshold.

![](images/8e5284478d37620600b7b82c45aa700cfc1c0632ee14999e854dcad8762618c3.jpg)  
(a)

![](images/33aba6db5dc674dd5b468c2508956f327bae10c6def057cfffff967a5a3183b4.jpg)  
(b)  
Figure 14: Example ground truth denoiser and network denoiser in 33 coordinates at sampled points. Samples in a) are untilted sampled and b) tilted samples.

Distribution of net-sampled solutions over the 288 valid sudoku grids (n=3000, 2928 valid, tau=4)  
![](images/5face055894e47e475628bbc8ae9dc6e6492673a949737afe2be26df6f6382ee.jpg)

![](images/733e7c1dbaa1aef5e7c7b3df481c4f6450dc69bbbfe76d6a164df055833755b3.jpg)  
(a) 4x4 Sudoku chi-squared test network samples.

Net vs exact-sampler distribution over 288 solutions (n=3000, tau=4)  
![](images/8f25c4791c724565a6e1a5e0fbb6665fc6641a337ff2ce966ab83d799b273de6.jpg)

Sorted counts per solution: net vs exact sampler  
![](images/e31d3a49c6a207be1996f8cddc3574919917f56eaa3f70fec312a56316d8b23e.jpg)  
(b) 4x4 Sudoku chi-squared test ground truth samples.  
Figure 15

## C.3 6x6 Sudoku

## Target Density

$6 \times 6$ Sudoku has 6 rows, 6 columns, and six non-overlapping $2 \times 3$ rectangular boxes, each of which must be a permutation of 1–6 to be valid. Each cell is a 3-bit encoding of the numbers 1–6, giving $3 6 \times 3 = 1 0 8$ core bits and a state space of $2 ^ { 1 0 8 }$ . Unlike the $4 \times 4$ case, this encoding has dead code words: three bits admit eight values, of which only six are legal, so two codes per cell decode to the invalid values 7 and 8. The generator never emits them, but the model operates on the full $2 ^ { 1 0 8 }$ space and can.

<table><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>3</td></tr></table>

Figure 16: Example valid 6x6 grid.

There are 28,200,960 proper puzzles, vastly more than the 4x4 puzzle. The training set does not contain all proper puzzles, thus the network’s generations should be novel making new proper puzzles not seen in the training set.

The base distribution is again a weak generator sampling each row independently. With probability $1 - \rho$ the row is a uniformly random permutation of 1–6, and thus valid; with probability ρ it is instead six i.i.d. uniform digits drawn from 1–6, which may repeat. We use $\rho = 0 . 0 2$ . A row is therefore valid with probability $\begin{array} { r } { ( 1 - \rho ) + \rho \cdot \frac { 7 2 0 } { 6 ^ { 6 } } = 9 8 . 0 3 \% } \end{array}$ , and all six rows are simultaneously valid 88.75% of the time.

Because the generator enforces no column or box constraints, and the grid is much larger than in the 4 × 4 case, it produces a proper puzzle with probability only $1 . 8 \times 1 0 ^ { - 1 0 }$ — efectively never. The reward is again the validity indicator.

## Training Set

Since the weak generator essentially never produces a proper puzzle on its own, reward-achieving examples must be injected: each sample is drawn uniformly from the set of valid solutions with probability $p _ { \mathrm { i n j e c t } } = 0 . 0 5$ , and from the weak generator otherwise. The training is scaled up to 5,000,000 samples from this mixture, roughly 33 times larger than the $4 \times 4$ training set, which is necessary because the injected solutions are drawn from a pool of 28.2 million rather than 288. The realized set contains 249,903 valid grids (5.00%), of which 248,778 are distinct.

## Model Architecture

The target distributions and models were trained in order from smallest to largest state space starting with the Ising Model, then 4x4, 6x6, 9x9 Sudoku, then the MNIST problem. As the state space scaled up in size, larger data sets were needed to learn the more complicated density and tilt, and more expressive model architectures.

The previous Ising and 4x4 Sudoku problems used MLP models with no attention mechanism. For 6x6, 9x9 Sudoku and MNIST we use a transformer architecture with attention.

This problem uses a transformer featurizer rather than the residual MLP, since the row, column and box constraints are joint combinatorial structure that a flat trunk must discover entirely from data, whereas attention can relate cells directly. The 108 core bits are split into 6 tokens of 18 bits (one per row), and the reward coordinate is given its own token rather than folded into the level conditioning, so attention can relate $x _ { * }$ to every row and vice versa; attention is non-causal over the resulting 7 slots. The model has 4 layers with $d _ { \mathrm { m o d e l } } = 2 5 6$ , 8 attention heads, and feed forward dimension 1024, and noise-level conditioning enters by AdaLN from the same Fourier leve embedding $e ( a )$ used elsewhere.

The output head is unchanged, preserving the mean-field skip

$$
G _ { \theta } ( u , a ) = \mathrm { t a n h } \big ( \beta ( a ) u + g _ { \theta } ( a ) \odot r _ { \theta } ( u , a ) \big ) ,
$$

with every residual branch zero-initialized so that at initialization the network is exactly the meanfield denoiser. The network is additionally self-conditioned, as discussed in Appendix B.4, the previous estimate of $x _ { 0 }$ is concatenated to the input along the feature axis of each token, doubling the per-slot input channels from 18 to 36. The model has 3,249,028 parameters, substantially fewer than the $4 \times 4 ~ \mathrm { M L P }$ despite the much larger state space. Full implementation details are in the supplementary code.

## Training Run

The model was trained for 180,000 Adam steps with batch size 8192, learning rate $1 0 ^ { - 3 }$ , and a cosine-annealing schedule over the full 180,000 steps. During training the self-conditioning input is supplied with probability 0.5, so the network also learns the zero-input fallback needed at the first sampling step. Each batch re-noises freshly drawn samples, with $\beta$ drawn log-uniformly on [0.02, 50]; with probability 0.3 the row instead draws $\beta$ log-uniformly from the narrower band [0.5, 2.5], which over-samples the regime empirically hardest to fit. With probability 0.25 a row uses the sparse infill profile (k uniform on $\{ 2 , \ldots , 1 0 8 \}$ , a random k-subset set to $\sqrt { 1 0 8 / k } )$ , and otherwise the reward-focusing profile; independently, the reward-axis stretch λ is 1 with probability 0.25 and otherwise log-uniform on [1, 10]. The objective is the unweighted per-coordinate squared error summed over the 109 coordinates; the final training loss was 53.92.

Training ran on a single NVIDIA H100 80GB HBM3 GPU (Hopper, CUDA 12.6) on the PSC Bridges-2 cluster and took 3.70 hours. Processing $1 8 0 , 0 0 0 \times 8 1 9 2 \approx 1 . 4 7 \times 1 0 ^ { 9 }$ samples through a 3,249,028-parameter network corresponds to approximately $2 . 9 \times 1 0 ^ { 1 6 }$ training FLOPs under the standard 6ND estimate.

## Extended Results

For the 6x6 Sudoku problem, the weak generator samples each row independently thus the ground truth denoiser expectation can be enumerated for the untilted density. With a much larger set of valid puzzles we do not enumerate the denoiser with reward tilting applied. The $L ^ { 2 }$ error for $\tau = 0$ with the base denoiser is shown in Figure 17.

For the log-odds ratio, the τ achieving 98% reward is computed, and 5 values below and 5 values up to $\sqrt { d }$ are computed. With only 2000 samples, we do not have the specificity to detect very high probabilities, hence the confidence intervals on the log-odds start to become uninformative for very high tilts. Note these tilts are already producing over 98% correct puzzles.

At $\tau _ { 9 8 } = 3 . 4 2$ we drew 1000 samples, of which 917 decoded to valid grids, and checked each for an exact match against all 5,000,000 training rows. The training set covers 248,778 of the 28,200,960 valid $6 \times 6$ grids (0.88%), so a sampler drawing uniformly from valid grids would produce 8.1 exact matches by chance (null 95% interval 3–14); we observe 10 (1.09% of valid samples, 95% CI 0.52–2.00%, $p = 0 . 2 9$ for $\geq 1 0$ under the null), consistent with no memorization.

![](images/5ef31accb08dc49bdee797344699ce8b699a376b3b1790fdaceb45bf87c6db3a.jpg)  
Figure 17: $L ^ { 2 }$ error with enumerated ground truth denoiser untilted.

![](images/dfb54afc6d1029dd5add59b9225af5edec62d316e5aa9a59dc14ff51a46593ad.jpg)  
Figure 18: 6x6 Sudoku log-odds across tilt strengths.

## C.4 9x9 Sudoku

## Target Density

Full-size $9 \times 9$ Sudoku has 9 rows, 9 columns, and nine non-overlapping $3 \times 3$ boxes, each required to be a permutation of $\{ 1 , \ldots , 9 \}$ . There are $\approx 6 . 6 7 \times 1 0 ^ { 2 1 }$ proper puzzles.

<table><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>3</td></tr></table>

Figure 19: Example valid 9x9 grid.

Each cell is a 4-bit encoding of the numbers 1–9, giving $8 1 \times 4 = 3 2 4$ core bits and a state space of $2 ^ { 3 2 4 }$ . The dead-code word problem is considerably worse than at $6 \times 6 \cdot$ : four bits admit sixteen values of which only nine are legal.

The base distribution is again a weak generator sampling each row independently: with probability $1 - \rho \mathrm { ~ a ~ }$ uniformly random permutation of 1–9, and with probability ρ nine i.i.d. uniform digits from 1–9, which may repeat. We use $\rho = 0 . 0 2 { \mathrm { . } }$ giving a row validity of 98.00% and all nine rows simultaneously valid 83.39% of the time. With no column or box constraints enforced, the generator produces a proper puzzle with probability $5 . 1 \times 1 0 ^ { - 2 9 }$

The reward is the validity indicator, appended as a 325th coordinate to give the extended state space $\{ - 1 , 1 \} ^ { 3 2 5 }$

Training Set

The training set consists of 10,000,000 samples, drawn from the weak generator with probability $1 - p _ { \mathrm { i n j e c t } }$ and from a valid-solution sampler with probability $p _ { \mathrm { i n j e c t } } = 0 . 0 5$ . Unlike $6 \times 6$ no eficient exactly uniform sampler over valid grids exists at this scale: the $6 \times 6$ construction enumerated the 39,168 grids with first row $1 , \ldots , 6$ and composed them with relabellings, whereas the corresponding $9 \times 9$ quotient has $1 . 8 4 \times 1 0 ^ { 1 6 }$ reduced grids and is not enumerable. Injection instead uses randomized backtracking composed with a uniform draw from the full symmetry group $( | G | = 1 , 2 1 8 , 9 9 8 , 1 0 8 , 1 6 0 )$ , which is exactly uniform within each symmetry orbit and biased only in the orbit label.

The realized training set contains 500,883 valid grids (5.01%), all of them distinct, so each is seen roughly 147 times over the training run. Proper sampling should essentially have 0 probability of recreating a training example, thus the model should generate 100% novel valid puzzles and never recreate a valid puzzle seen in the training set.

## Model Architecture

The architecture is the same self-conditioned transformer used at $6 \times 6 .$ , scaled up. The 324 core bits are split into 9 tokens of 36 bits (one per row), the reward coordinate is given its own token, and non-causal attention runs over the resulting 10 slots. The model has 8 layers with $d _ { \mathrm { m o d e l } } = 5 1 2$ , 8 heads, and feed-forward dimension 2048, with AdaLN level conditioning and the same zero-initialized mean-field skip

$$
G _ { \theta } ( u , a ) = \mathrm { t a n h } \big ( \beta ( a ) u + g _ { \theta } ( a ) \odot r _ { \theta } ( u , a ) \big ) ,
$$

so that at initialization the network is exactly the mean-field denoiser. Self-conditioning concatenates the previous $x _ { 0 }$ estimate along each token’s feature axis, doubling per-slot input channels from 36 to 72. The model has 25,558,382 parameters. Full implementation details are in the supplementary code.

## Training Run

The model was trained for 180,000 Adam steps with batch size 8192, learning rate $1 0 ^ { - 3 } .$ , and a cosine-annealing schedule, with self-conditioning supplied with probability 0.5. Signal levels are drawn log-uniformly on [0.02, 50], with probability 0.3 instead drawn from the harder band [0.5, 2.5]. With probability 0.25 a row uses the sparse infill profile (k uniform on $\{ 2 , \ldots , 3 2 4 \}$ , a random ksubset set to $\sqrt { 3 2 4 / k } )$ , and otherwise the reward-focusing profile; the reward-axis stretch λ is 1 with probability 0.25 and otherwise log-uniform on [1, 17].

Training ran on a single NVIDIA H100 80GB HBM3 GPU (Hopper, CUDA 12.6) on the PSC Bridges-2 cluster and took 22.5 hours. Processing $1 8 0 , 0 0 0 \times 8 1 9 2 \approx 1 . 4 7 \times 1 0 ^ { 9 }$ samples through a 25,558,382-parameter network corresponds to approximately $2 . 3 \times 1 0 ^ { 1 7 }$ training FLOPs under the standard 6ND estimate.

## Extended Results

For the 9x9 Sudoku, there is no ground truth denoiser to compare against since the row space is too large to enumerate. We consider the log-odds ratio.

For the 9x9 model, we consider the two forms of self-conditioning. Mode “two-pass” is a Markovian form of self-conditioning which may be interpreted as an architecture choice where the denoiser is fed its own prediction

$$
\tilde { G } _ { t } = G ( X _ { t } , G ( X _ { t } , 0 , a ) , a ) .
$$

Mode “carry” uses the previous time point’s estimate as input, which breaks the Markov structure and the sampling path is no longer a pure Euler-Maruyama discretization of the continuous SDE with pointwise denoiser estimation,

$$
\tilde { G } _ { t } = G ( X _ { t } , \tilde { G } _ { t - 1 } , a ) .
$$

For the 9x9 model we see noticeable diference in performance between the two-pass selfconditioning and the carry self-conditioning.

Two-pass fails to properly track the log-odds. It plateaus around 60% reward-achieving solutions, and is already lagging behind at lower tilt strengths. Note the model has its own prediction of if it thinks it is producing a reward-achieving solution in its d+1 coordinate, and we can determine if the produced token string is reward-achieving as well. For a properly sampling model, these two should agree. For the two-pass mode, we see the model’s own predictions of reward (orange) do trace the log-odds, but the actual proportion of reward-achieving solutions does not. Therefore, the model thinks it is producing 90% plus reward-achieving, but in reality only 60% are.

![](images/10fabf564e7fe875a1c98d6fd582653928b9cfce69c376aee31d76cba762e5bd.jpg)

![](images/5b31d0f0dc7161eefe882f295b3162e53cb2e77156918bb297c3a9557e22cdbe.jpg)  
(a) Self-conditioning mode two-pass.

Sudoku 9x9 d=325 reward-tilt vs. tau(p0=5.00%, n=1000/tau, K=1000, self\_cond\_mode='carry')  
![](images/a50f8ef4e9003b5d552e26b9ef81202fd2f3c5a4f181e12b5927b0fbb4c3614e.jpg)

![](images/44bc81ed0388c2ccc4ab08ad4bc882e6d14b548290c314e3f6d034cffa5ada77.jpg)  
(b) Self-conditioning mode carry.  
Figure 20: 9x9 Sudoku log-odds across tilt strengths for both self-conditioning modes.

The carry mode does much better at tracking the log-odds, plateauing at 90% reward-achieving. At higher tilt strengths, it also experiences some discrepancy between the solutions it thinks are reward-achieving via its d + 1 bit being positive, but are graded as invalid puzzles by the Sudoku conditions. This discrepancy is more on the order of 5-10% for the carry mode, rather than the 30% of the two-pass mode.

To consider why this may be the case, it is possible the few missing percent of samples that were “supposed” to terminate at a valid puzzle mis-sampled a few bits due to errors in the learned denoiser accumulating over the sample path, which would get a -1 reward from the sharp correct/incorrect puzzle reward function for any cell violations.

Using a CP-SAT optimization problem we can find for each generated puzzle the nearest valid solution in Hamming distance. We plot the CDF of how many produced puzzles in 300 samples are valid (distance 0), 1 bit away from correct, 2, and so on from a valid puzzle in Figure 21. We see an example puzzle in Figure 22 that was generated by two-pass self-conditioning at a tilt strength of 3 and is 8 bit flips away from a valid puzzle. We see in Figure 21 for the two-pass mode, while only 60% of puzzles are valid, 90% are within 10 bits out of 324 from a valid puzzle. Two-pass and carry have comparable proportions of puzzles produced within 10 bits of a valid puzzle.

![](images/be37aaec6443de37605f70f6ea76496796550b907894bccddfa4a1ced7175fe5.jpg)  
Figure 21: Cumulative distribution function of Hamming distance to nearest valid puzzle for 300 samples.

Here we summarize the infilling performance of the model, using an infill strength and a reward tilt strength of $\tau = 3$ across all tests. Given less initial cell values, the sampler does produce less valid puzzles and does not always retain the initial cell values given to it. Probability of returning the given initial cells and generating a valid puzzle ranges from 80% to 100% across the amount of initial cells given.

Table 1: Sudoku $9 \times 9$ infilling as a function of the number of given cells. Each row is based on 100 uniformly random valid solutions × 20 draws per puzzle $( n = 2 0 0 0 )$ , with a fresh random mask per puzzle. Tilt strength $\tau = 3$ for both the reward and the cell-fixing tilts, 1000 Euler steps, self-conditioning mode carry. Brackets give 95% Clopper–Pearson intervals. “Joint” is the fraction of samples that are valid grids and also keep every given cell.
<table><tr><td>Given</td><td>Removed</td><td>Valid grid</td><td>Keeps given cells</td><td>Joint success</td></tr><tr><td>71</td><td>10</td><td>1.000 [0.998, 1.000]</td><td>1.000 [0.998, 1.000]</td><td>1.000 [0.998, 1.000]</td></tr><tr><td>61</td><td>20</td><td>1.000 [0.998, 1.000]</td><td>0.999 [0.997, 1.000]</td><td>0.999 [0.997, 1.000]</td></tr><tr><td>51</td><td>30</td><td>0.999 [0.997, 1.000]</td><td>0.998 [0.994, 0.999]</td><td>0.997 [0.994, 0.999]</td></tr><tr><td>41</td><td>40</td><td>0.996 [0.992, 0.998]</td><td>0.989 [0.983, 0.993]</td><td>0.985 [0.979, 0.990]</td></tr><tr><td>31</td><td>50</td><td>0.953 [0.942, 0.961]</td><td>0.944 [0.933, 0.954]</td><td>0.899 [0.885, 0.912]</td></tr><tr><td>21</td><td>60</td><td>0.925 [0.913, 0.936]</td><td>0.918 [0.905, 0.930]</td><td>0.851 [0.835, 0.866]</td></tr><tr><td>11</td><td>70</td><td>0.867 [0.851, 0.882]</td><td>0.924 [0.911, 0.935]</td><td>0.803 [0.784, 0.820]</td></tr></table>

A natural concern for a sampler that reaches high validity is that it has simply memorized its training set. We test this on Sudoku 9×9 with the self-conditioned transformer (carry, $\tau = 3 .$ , 1000 Euler steps). The 500,883 valid training grids, all distinct, are split at random into a reference half (250,000 grids) and a held-out half. Of N = 1000 generations, 869 are valid Sudoku solutions, and none of the 1000 matches any reference set exactly. Given that there are about $6 . 6 7 \times 1 0 ^ { 2 1 }$ valid grids, a single exact match would have been strong evidence of copying. We also compute, for each valid generation, the cell-wise Hamming distance to its nearest reference grid, and compare this with the same distance computed for 869 held-out training grids, which are true unseen solutions from the data distribution (Figure 23). The two distributions are almost identical. Generated grids are therefore no closer to the training data than new samples from the data distribution would be, and the closest generation (39 cells away) still difers from every reference grid in almost half its cells. The model has learned the Sudoku constraints rather than copying its training grids.

![](images/f43f8b3b19fc96252731911c8d2ea3aac06851e4b1582e06ddb38bc2f8f61cbd.jpg)  
Figure 22: Generated invalid puzzle (right) with constraint violations and nearest valid puzzle (left) in Hamming distance. Nearest valid grid would require 8 bit flips to the binary encoding. The bit flips are not all restricted to the invalid bits of the produced puzzle, but previously valid bits may be flipped to get to a valid puzzle.

![](images/8d0fb2c0ea373e5b17cd8a802c0c04cfdec31e2dcc58ae857ac4ff0291688132.jpg)

![](images/258ea49dcb838749c27dfd45c58e8eb97ae1c7373c587e6b62ffedc886514ac6.jpg)  
Figure 23: CDF and density of nearest distance reward-achieving gird in training set for network generated samples and held out set. Nearly identical distributions, so generated puzzles no more similar to training set than a held out portion. No puzzles are exact match, closet puzzle difer in 39 cells

## C.5 MNIST Conditional Image Generation

## Target Density

MNIST consists of $2 8 \times 2 8$ grayscale images of handwritten digits 0–9, with pixel intensities in $\{ 0 , \ldots , 2 5 5 \}$ . Each pixel is encoded with 8 bits, giving $7 8 4 \times 8 = 6 { , } 2 7 2$ core bits and a state space of $2 ^ { 6 2 7 2 }$ . Since the pixel values have a natural ordering, we use a Gray code encoding for each pixel, so that adjacent intensity levels difer in exactly one bit.

Since there are 10 digit classes, the reward is a 10-bit block rather than a single coordinate. The class label c is stored as a one-hot pattern $r _ { c } \in \{ - 1 , 1 \} ^ { 1 0 } , + 1$ in position c and −1 elsewhere, appended to the image bits to give the extended state space $\{ - 1 , 1 \} ^ { 6 2 8 2 }$ . To generate class c we tilt along the whole pattern, $h = \tau r _ { c }$ , rather than along the single axis $e _ { c }$ . Thus one class receives a positive tilt while the others all receive a negative one. This causes the log-odds of a class to have a slope of 4,

$$
\mathrm { l o g i t } \pi _ { \tau } ( c ) = \mathrm { l o g i t } \pi ( c ) + 4 \tau ,
$$

where $\pi ( c )$ is the training-set frequency of class c (0.090–0.112). The slope is 4 rather than the slope 2 of the single-bit reward tilt in the Sudoku and Ising problems.

## Training Set

The training set is the standard 60,000 MNIST training images and their labels, each encoded as above. The oficial 10,000-image test split is held out and never used for training. It is used for three things: (i) a fixed batch of 4,096 held-out images, noised with the training recipe, is scored every 1,000 steps to monitor overfitting; (ii) it provides the real-digit comparison in Figure 28; and (iii) it is used to measure the error rates of the classifier that scores generated samples (see Extended Results).

## Model Architecture

We use the same self-conditioned transformer as for $6 \times 6$ and $9 \times 9$ Sudoku, with tokens built from image patches. The image is divided into a $7 \times 7$ grid of non-overlapping $4 \times 4$ pixel patches.

Each patch becomes one token of $1 6 \times 8 = 1 2 8$ bits (pixels in row order within a patch, Gray bits most significant first), giving 49 image tokens. The 10-bit class block is given its own token, and non-causal attention runs over the resulting 50 slots. The model has 4 layers with $d _ { \mathrm { m o d e l } } = 2 5 6 , 8$ attention heads, and feed-forward dimension 1024, with AdaLN noise-level conditioning from the Fourier level embedding $e ( a )$ and the same zero-initialized mean-field skip

$$
G _ { \theta } ( u , a ) = \operatorname { t a n h } \left( \beta ( a ) u + g _ { \theta } ( a ) \odot r _ { \theta } ( u , a ) \right) .
$$

Self-conditioning concatenates the previous $x _ { 0 }$ estimate to each token’s features, doubling the perslot input channels from 128 to 256. The model has 3,555,256 parameters. Full implementation details are in the supplementary code.

## Training Run

The model was trained for 30,000 Adam steps with batch size 2048, learning rate $1 0 ^ { - 3 }$ , and a cosine-annealing schedule, with the self-conditioning input supplied with probability 0.5. Signal levels are drawn log-uniformly on $\beta \in [ 0 . 0 2 , 5 0 ]$ . Λ focusing is applied to the whole 10-bit reward block, maintaining the sign of those inputs but increasing their magnitude. The λ is 1 with probability 0.25 and otherwise log-uniform on [1, 24]. The upper limit difers from the Sudoku runs: because the whole 10-coordinate block is stretched at once, the admissible range is $\lambda \ <$ $\sqrt { d / 1 0 } = 2 5 . 0 6$ rather than ${ \sqrt { d } } .$ , and we train just below this limit.

$3 0 , 0 0 0 \times 2 0 4 8 \approx 6 . 1 \times 1 0 ^ { 7 }$ draws amount to roughly 1,000 passes over the 60,000 training images. The held-out loss decreased steadily from 720.6 at step 1,000 to 632.1 at step 30,000 and was still flat-to-decreasing at the end. This gives no sign of overfitting.

Training ran on a single NVIDIA H100 80GB GPU on the PSC Bridges-2 cluster for 1 hour and 15 minutes. Processing $6 . 1 \times 1 0 ^ { 7 }$ samples through a 3,555,256-parameter network corresponds to approximately $1 . 3 \times 1 0 ^ { 1 5 }$ training FLOPs under the standard 6ND estimate.

## Extended Results

To detect if a generated image is of a given class, we train an independent score CNN classifier. The scorer is a two-layer CNN (1.2M parameters) trained for 3 epochs with Adam on the 60,000 MNIST training images, reaching 98.74% accuracy on the held-out test split.

The classifier is an imperfect instrument and cannot detect very strong tilt accuracies beyond its own classification error. Its error is also not constant across classes. As such, our ability to detect tilt strength saturates at the classifier accuracy.

Near the top of the tilt range its error is the dominant source of uncertainty. Let $s _ { c }$ be the probability the classifier labels a true class-c image as $^ { c , }$ and $e _ { c }$ the probability it labels an image of another class as c. For a true class-c rate $p ,$ the rate we observe is

$$
m = p s _ { c } + ( 1 - p ) e _ { c } .
$$

We measure $s _ { c }$ and $e _ { c }$ on the held-out set $\left( s _ { c } \right.$ ranges from 0.962 for class 9 to 0.997 for class $1 ;$ $e _ { c } \le 0 . 0 0 2 3 )$ . We then compare the observed rate with the exact marginal passed through this equation, rather than correcting the observed rate. The observed rate therefore levels of at about $s _ { c }$ rather than 1 (log-odds 3.2–5.9 depending on the class). Once the corrected exact rate comes within two standard errors of this limit, which happens for $\tau \geq 1 . 6 9$ in every class, the sample size is not what limits resolution; the classifier cannot distinguish further improvements. We treat these points as uninformative about the classifier-scored rate.

We report two rates from the same samples: the classifier-scored rate, and the class the network itself writes in its 10-bit reward block (argmax of the block). The second is read directly from the sample and involves no classifier. The τ grid has 6 points on $[ 0 , \tau _ { 9 5 } ]$ , where $\tau _ { 9 5 } = 1 . 2 8 5$ is the strength at which the exact rate reaches 95% for the mean class frequency, and 3 points on $\left( \tau _ { 9 5 } , 2 . 5 \right]$ . We anchor at 95% rather than the 98% used for Sudoku because the classifier limit leaves less usable range. Each (class, τ ) pair uses 256 samples with 1000 Euler steps, giving 2,560 samples per τ pooled over classes. Tables 2 and 3 give the pooled results, and Figures 24–27 show each class separately.

Table 2: MNIST pooled class-conditional log-odds, two-pass self-conditioning, $n = 2 { , } 5 6 0$ samples per τ (256 per class). “Exact” is logit of the exact tilted class marginal; “Exact (scored)” is the same rate passed through each class’s measured classifier errors and averaged over classes, which is the right reference for the classifier-scored column. “Classifier” is the fraction of samples the CNN assigns to the requested class, with a 95% Clopper–Pearson interval. “Reward block” is the fraction whose reward block has its argmax at the requested class. <sup>†</sup>Beyond the classifier’s resolution limit.
<table><tr><td>T</td><td>Exact</td><td>Exact (scored)</td><td colspan="3">Classifier</td><td>Reward block</td></tr><tr><td>0.00</td><td>-2.20</td><td>-2.20</td><td></td><td>-2.11 [−2.24, −1.99]</td><td>-2.10</td></tr><tr><td>0.26</td><td>-1.17</td><td>-1.18</td><td></td><td>−1.19 [−1.29, −1.10]</td><td>-1.12</td></tr><tr><td>0.51</td><td>-0.14</td><td>-0.16</td><td>0.00</td><td>[−0.07, 0.08]</td><td>0.10</td></tr><tr><td>0.77</td><td>0.89</td><td>0.84</td><td>0.93</td><td>0.84, 1.02]</td><td>1.14</td></tr><tr><td>1.03</td><td>1.91</td><td>1.82</td><td>1.78</td><td>1.67, 1.89]</td><td>2.03</td></tr><tr><td>1.29</td><td>2.94</td><td>2.71</td><td>2.28</td><td>2.15, 2.42</td><td>2.91</td></tr><tr><td>1.69†</td><td>4.56</td><td>3.75</td><td>2.76</td><td>2.60, 2.93]</td><td>3.78</td></tr><tr><td>2.10†</td><td>6.18</td><td>4.20</td><td>3.13</td><td>2.94, 3.34]</td><td>4.34</td></tr><tr><td>2.50†</td><td>7.80</td><td>4.32</td><td>3.36</td><td>3.14, 3.59]</td><td>6.05</td></tr></table>

Table 3: As Table 2, for carry self-conditioning, with the same τ grid, sample sizes and seeds. <sup>†</sup>Beyond the classifier’s resolution limit.
<table><tr><td>T</td><td>Exact</td><td>Exact (scored)</td><td></td><td>Classifier</td><td></td><td>Reward block</td></tr><tr><td>0.00</td><td>-2.20</td><td>-2.20</td><td></td><td>−2.19 [−2.32, −2.06]</td><td></td><td>-2.19</td></tr><tr><td>0.26</td><td>-1.17</td><td>-1.18</td><td>-0.92</td><td>[−1.01, −0.84]</td><td></td><td>-0.92</td></tr><tr><td>0.51</td><td>-0.14</td><td>-0.16</td><td>0.31</td><td>0.23,</td><td>0.39]</td><td>0.33</td></tr><tr><td>0.77</td><td>0.89</td><td>0.84</td><td>1.33</td><td>1.24,</td><td>1.43]</td><td>1.42</td></tr><tr><td>1.03</td><td>1.91</td><td>1.82</td><td>2.12</td><td>2.00,</td><td>2.25]</td><td>2.20</td></tr><tr><td>1.29</td><td>2.94</td><td>2.71</td><td>2.83</td><td>2.66,</td><td>3.00]</td><td>3.19</td></tr><tr><td>1.69†</td><td>4.56</td><td>3.75</td><td>3.23</td><td>3.03,</td><td>3.45]</td><td>3.80</td></tr><tr><td>2.10†</td><td>6.18</td><td>4.20</td><td>3.73</td><td>3.47,</td><td>4.00]</td><td>5.28</td></tr><tr><td>2.50†</td><td>7.80</td><td>4.32</td><td>4.05</td><td>3.75,</td><td>4.37]</td><td>5.45</td></tr></table>

The self-conditioning modes two-pass (Markovian) and carry (non-Markovian) behave in the opposite way to $9 \times 9$ Sudoku. The carry model over-samples the reward class at a given tilt, producing more outputs of a given class than the log-odds would predict. The two-pass mode tracks the log-odds more truthfully.

With two-pass, the classifier-scored rate matches the corrected exact rate within its confidence interval, or nearly so, from τ = 0 through τ ≈ 1 (e.g. 1.78 vs. 1.82 at $\tau = 1 . 0 3 )$ , and the network’s reward block follows the same curve. With carry, the sampler overshoots over the whole rising region. At $\tau = 0 . 5 1$ it produces the requested class 57.8% of the time against an exact rate of 45.9% (log-odds 0.31 vs. −0.16), and the 95% interval excludes the exact value at every τ from 0.26 to 1.03. The untilted rate is correct under both modes, so this is an error in how carry responds to the tilt, not in the base distribution.

![](images/497db87cf175f1270bc48cac1045f0488fffc4c3dbf203584fe5a50d4c27f7ba.jpg)  
Figure 24: MNIST class-conditional log-odds against tilt strength τ for each class, two-pass selfconditioning. Exact marginal passed through each class’s classifier errors (line, with a band from the uncertainty in those errors), classifier-scored rate (points with 95% Clopper–Pearson intervals), and the network’s own reward-block rate.

![](images/d4662a63b202dd5f6a988171d9c22a89dafe7501f9f85465048369b8ab7ae750.jpg)  
Figure 25: MNIST log-odds pooled over all ten classes, two-pass self-conditioning (Table 2).

![](images/9b0885faf2b8d6f4e4f60cb06877e3485f7217f9642b0367fc5d1e6d40218326.jpg)  
Figure 26: As Figure 24, for carry self-conditioning. The overshoot over $0 . 2 6 \leq \tau \leq 1 . 0 3$ appears in every class.

![](images/b5de662acf76d6a59459248672f0a65dc7ff8259bdb34d4b5e6b8d79fda2e134.jpg)  
Figure 27: MNIST log-odds pooled over all ten classes, carry self-conditioning (Table 3).

The log-odds check does not consider the sample quality or variety of digits produced. We sample 25 class 0 images from the hold out set, 25 samples images at the carry mode, and 25 samples at two-pass mode at a tilt τ<sub>0.95</sub> with a projected 95% success rate. We compare the outputs visually for sample quality and variety to detect mode collapse or corruption of digit quality in Figure 28. Digits generated by the network show the same variety and quality as true hold out images. Note one of the generated samples was mis-sampled as a class 8 image, but this is to be expected with a 5% non-reward rate at this tilt strength. Shared seed and randomness was used in the two sampling runs which results in similarity in the images generated in pane b) and c).

REAL MNIST holdout, class 0 (n=25, seed=0) scorer agrees on 25/25; panel title = scorer's reading

![](images/6c15c8c113159e98ebd29672fed26b79803ee14f5905a6a0e60a5e4078f6f62d.jpg)  
(a) 25 class-0 images from the heldout set.

MNIST samples, class 0 (tau=1.289, K=1000, carry) scorer agrees on 25/25: panel title = scorer's reading  
![](images/bcc31e9a5611fe169e56625f42f71c86c39affe2c159dc12ed7e22537c5c580a.jpg)  
(b) 25 generated class-0 images, carry (non-Markovian) selfconditioning.  
MNIST samples, class 0 (tau=1.289, K=1000. two pass) scorer agrees on 24/25; panel title = scorer's readinc

![](images/d3a866d164e8639c34086b51090ba6d74f63f70c1f094c9f3550ec94bd3c5628.jpg)  
(c) 25 generated class-0 two-pass (Markovian) conditioning.  
Figure 28: 25 class-0 images from the held-out set (a), and network samples under both selfconditioning modes with the same random seed (b, c). Panel titles give the classifier’s label (green if it matches the requested class, red if not). The samples are similar in quality and variety to the held-out digits, with no sign of mode collapse. τ = 1.289 is set so that the exact class-0 rate is 95%, so about 1 in 25 samples is expected to be of another class.

![](images/d00380153fbc67e4953f44cd9bd6448b55ef57a8a165e497817de2a9bd73b8df.jpg)

Figure 29: Example digit generations tilted towards each class. Leftmost image per digit is a sample of 25 images from hold out set network was not trained on, second image is from sample mode carry and third image is sample mode two-pass. Middle and right grids share the same random seed in their sampling, resulting in often similar samples which is not a form of memorization. Tilt strength is calibrated to 95% reward-achieving in log-odds theory, thus it is reasonable to see a few mis-samples in repeated 25 size sets.