# Generative difusion in long-range correlated landscapes: Speciation cascade and the foam structure of the score

J´erˆome Garnier-Brun<sup>1</sup>, Giulio Biroli<sup>2</sup>, Marc M´ezard<sup>1,4</sup>, and Jean-Philippe Bouchaud<sup>3,4</sup>

<sup>1</sup>Department of Computing Sciences, Bocconi University, Milan, Italy <sup>2</sup>Laboratoire de Physique de l’Ecole Normale Sup´erieure, ENS, Paris, France <sup>´</sup> <sup>3</sup>Capital Fund Management, Paris, France <sup>4</sup>Acad´emie des Sciences, Paris, France

November 3, 2026

## Abstract

While state of the art for image and video generation, the fundamental reasons for the efectiveness of generative difusion remain unclear. In particular, the observed advantage of following an entire reverse process with respect to single-shot or autoregressive methods, and for what type of data it may hold, is not well understood. In this work, we aim to better understand the role of the reverse process by considering a complex yet analytically tractable data model, the Gibbs measure of a high-dimensional Gaussian random potential with long-range correlations. Focusing on the exact score, we show that the “speciation” threshold, in which a trajectory commits to one class of the data, becomes a speciation window, in which the reverse process progressively resolves the Parisi tree of pure states of the full replica-symmetry breaking measure, committing to ever finer branches down to a single pure state. Importantly, these commitments take place at noise levels where the noisy sample distinguishes the pure states only through an energy of order one, beyond the reach of standard replica saddle points. Constructing a Larkin argument suited to such O(1) energy gaps, we describe the score as a foam, with slab-like cells separated by shock-like boundaries. In the variance-exploding process we consider, this window is followed by a shrinkage of the samples towards the data support, and finally by reconstruction within the selected pure state. The denoising error of the exact score follows these stages, dropping during the speciation window with a profile set by the Parisi function alone, and then within the selected state. We believe that the generative foam we describe, and in particular the boundaries between cells, could play an important role in determining where the errors of trained difusion models redistribute the mass of the generated samples when the data are long-range correlated.

## Contents

## 1 Introduction

Our results in a nutshell: a multiscale speciation cascade

2 Setting 5   
2.1 Generative difusion 5   
2.1.1 Two measures . . . 6   
2.1.2 A sharp limit: score shocks 7   
2.2 The data: a long-range random potential and its Parisi tree 7   
2.3 Pure state decomposition of the noised measure . 9   
3 The onset of speciation 9   
3.1 The speciation time 9   
3.2 A spectral criterion . 10   
3.3 Data covariance of the fRSB measure 10   
4 Speciation cascade and generative foam 12   
4.1 The Larkin balance . . 12   
4.2 The speciation window: Descending the Parisi tree 15   
4.3 Geometry of the foam and shape of the cells 16   
4.4 Shocks and cusps in the score correlator 18   
5 The denoising error across scales 19   
5.1 Speciation regime: the foam seen by the denoiser 19   
5.2 Within the selected state: a replica-symmetric posterior 21   
6 Beyond the exact score: imperfect reverse dynamics 22   
7 Discussion 24   
A Replica theory of the data distribution 31   
B Spectral estimate of the speciation scale 33   
C Low-lying gaps and a real-space GREM 34   
D Zero temperature curvature of the foam 37   
E State attribution in the speciation regime 38   
F Replica computations at fixed κ 40   
G The source  of the perturbed reverse dynamics 45

## 1 Introduction

Difusion-based models [1–3] are state of the art in the artificial generation of images, videos and sounds. They operate by learning to reverse a noising process that maps a complex data distribution to an easy-to-sample counterpart, usually a Gaussian. Generation then occurs by following an entire stochastic trajectory through the high-dimensional data space starting from pure noise and guided by a learned force field referred to as the score. Taking its roots in stochastic thermodynamics [1], this approach is particularly well suited to statistical physics analyses, which have notably been used to identify the diferent phases occurring in the reverse process [4–7] and the transition of these models from generation to memorisation of training data [8–11].

Understanding the role of data structure in generation is certainly a key open problem. In fact, one possible reason why difusion models (DM) outperform earlier one-shot generative models, such as normalising flows [12], is that the time-dependent DM generation, in many practical applications, proceeds efectively scale by scale [13, 14]. This makes them particularly well suited to multi-scale data such as natural images, videos and audio. Data are also hierarchically organised into abstract features and sub-features, which appears key to beating the curse of dimensionality [15, 16], and how this hierarchy emerges during generation is an active research topic [17–20]. In these works, however, the hierarchy is a latent tree generating the elements of a single sample. Difusion naturally raises the question of the spatial structure between samples, i.e. how they are organised relative to one another in data space, and how this organisation is recovered along the reverse trajectory [21]. This question then takes on a distinctly geometrical flavour, since structure in the data must show up as structure of a scalar landscape in the very space where sampling takes place, which is the core object of our paper.

As difusion consists of progressively adding noise in “time” (written $t = 1 / ( 2 \kappa )$ in the variance-exploding process we consider below), this spatial recovery becomes a temporal one. In the simplest tractable model of structured data, a mixture of two Gaussians, Biroli, Bonnaire, De Bortoli, and M´ezard [6] (BBBM) notably identified successive regimes of the reverse process. By reading the two modes as data categories, they defined and located a speciation time at which a trajectory commits to one class. This criterion was since generalised by Achilli et al. [7] to arbitrary classes through Bayes classification, the speciation time being set by a free-entropy diference between classes, with successive speciation times when several classes are present. In a complementary, non-asymptotic approach, Li and Chen [22] characterised critical windows of the reverse process, i.e. narrow time intervals during which a given feature of the output is decided, and interpreted difusion as a hierarchical sampler that commits to features at a discrete sequence of times. In these models, however, structure is minimal to enable analytical descriptions, with a few classes or a finite hierarchy of mixtures specified by hand, so that speciation reduces to a finite sequence of commitments.

Real data are instead correlated at all scales, likely organising into clusters within clusters. What becomes of speciation when it is no longer a few precise commitments but an entire cascade, and on which timescales the full hierarchy is recovered, remains open.To address these questions, we focus on high-dimensional probability distributions borrowed from disordered systems theory, and which do show such full hierarchy. We study N-dimensional data drawn from a Gibbs measure $P _ { 0 } ( { \pmb x } ) \propto \mathrm { e } ^ { - V ( { \pmb x } ) }$ , where V is a Gaussian random potential of the M´ezard–Parisi type [23– 25], with N large. More precisely, we consider the long-range correlated case $\overline { { V ( { \pmb x } ) V ( { \pmb y } ) } } =$ $B ( 0 ) - \sigma ^ { 2 } | { \pmb x } - { \pmb y } | ^ { 2 H } , 0 < H < 1$ , for which the Gibbs measure displays, at strong enough disorder, full replica-symmetry breaking (fRSB), i.e. an ultrametric tree of pure states embedded in space [26]. Such a setting is “rough but not featureless”: rich enough to organise into a hierarchy of clusters, yet simple enough for closed-form results [23, 24, 27, 28]. In this work, we focus on the exact score, in the vein of [6]; finite training sets on the same data landscape will be the object of a forthcoming work. Note that we are also not concerned here with the algorithmic implementation of such a denoiser, which is expected to be hard, at least in some regimes [29–33].

The M´ezard–Parisi model lets us import decades of knowledge on disordered systems into the analysis of generative difusion. As we shall see, however, the scales relevant to generation lie beyond the reach of the standard approach of a replica theory leading to an extensive saddle point. Indeed, in $N \gg 1$ dimensions, the multi-modal structure emerges at difusion times where the signal-to-noise ratio is only of order 1/N per coordinate. As in the two-Gaussian analysis of BBBM, the sampling trajectory then commits to a branch of the hierarchy through the projection of the noisy point onto the few directions that separate competing modes, an O(1) quantity, while the N remaining directions only carry noise. The extensive saddle point, dominated by these N directions, is blind to it. Note that this is not specific to our data model: whenever $P _ { 0 }$ is dominated by a few macroscopically distinct states, the choice among them is made at these same times, where the noisy point exerts a total field of order one on the data. This is the difusion counterpart of the vanishing symmetry-breaking field that selects a phase in the thermodynamic limit, and the limits of large N and long difusion times do not commute.

In order to describe the generative process, we therefore combine diferent techniques, all of which build on our knowledge of the Parisi tree of pure states and the statistics of their weights. We start by computing the covariance matrix of the data, allowing us to identify the scaling of the average onset of speciation following the spectral criterion introduced by BBBM (Sec. 3). Placing ourselves at this relevant timescale, we propose an adapted Larkin balance argument [34, 35], which recovers the speciation scaling and opens the description of a generative cascade that

Increasing ∙ = 1=(2t)  
![](images/254203595ce2d720c9cd126bba0c527d2fd4ae5bc0b76637af73216f54026969.jpg)  
(a) Long-time Gaussian

![](images/b6270452fedbc22a489bbee812eee18f07906b7688374a86522ec56f0e30ecfd.jpg)

Support of data P<sub>0</sub>

<sup>∙</sup> » <sup>1=N</sup>

![](images/2b2f275cf6bfb1f6f2357ae5a60e258acb32ff0e24498dfa0b20c4c53209d9df.jpg)  
(b) Onset of speciation

![](images/143e1898be127a3579689a49bd218ba6c7aa219f6d22b31bcd51b7f94e070077.jpg)  
(c) Hierarchical resolution

![](images/dffceb551679648b55b63cef068914eed537545d77ead65fff9449cf876aecb3.jpg)

Typical support of smoothed $P _ { t }$

![](images/2729bf0549f146dc4aea5b77061575b8a319dffc00c9b8a461eb746070a54fdf.jpg)  
(d) End of cascade

![](images/cd32bb086a337f333e1e10a60b163250b7e6f09ed55e98f87ba691b4d5aff8ca.jpg)  
(e) Contraction towards P<sub>0</sub>

![](images/041de54189d476d14ca056adaef3a81265b05b316951ac98cc03f51e801d97e0.jpg)  
(f) Recovery of P<sub>0</sub>  
Figure 1: Schematic of hierarchical state resolution along the reverse difusion scheme. The opaque inner sphere is the support of the data distribution $P _ { 0 } ,$ of radius $L \sqrt { N }$ , shown with its fixed partition into pure states; the translucent outer sphere marks the typical support of the smoothed distribution $P _ { t } ,$ i.e. the shell of radius $\sqrt { N ( L ^ { 2 } + \kappa ^ { - 1 } ) }$ on which it concentrates. (a) At the start of the reverse process $P _ { t }$ is efectively Gaussian and the shell is homogeneous. (b) At the onset of speciation a single boundary divides it into two broad sectors, separated by a first shock. $^ \mathrm { ( c , d ) }$ Sectors are subdivided hierarchically until the shell carries the same cells as the data sphere, marking the end of the speciation cascade. (e) The shell then contracts onto the data sphere without further transition, keeping their global topology while modifying their local shapes, and (f) coincides with the data sphere as $t \to 0$ , recovering $P _ { 0 }$ . Colours label states and the branches that contain them, not computed probability densities. The figure is a schematic three-dimensional cartoon of an N-dimensional problem, which does not account for the very strong anisotropy of the cells in high dimension (see Sec. 4.3).

resolves the spatial structure level by level down the Parisi tree (Sec. 4). At intermediate times, this progressive refinement leads to a “foam” of cells separated by sharp walls—the shocks of an associated Burgers problem, that leads to the well known “cusp” singularity in the score correlator [28, 36, 37] (Sec. 4.4)—until the finest level is reached. We then translate this picture in terms of the Bayes-optimal denoising error, which follows the cascade down to the width of the selected state, before a planted replica computation shows that the recovery within that state at short times is a simple narrowing of the posterior with no further transition (Sec. 5). Finally, we argue that the walls of the foam are precisely where errors of an imperfect reverse dynamics feed excess probability (Sec. 6); as we will discuss in a forthcoming work, we conjecture that this abundance of well-organised shock surfaces is a mechanism that helps learned scores to generalise.

## Our results in a nutshell: a multiscale speciation cascade

Our model ofers a generalisation of the speciation scenario of BBBM [6] in the case of hierarchically structured data with long-range correlations. In such a case, speciation is not a single decision but a cascade of branchings down the Parisi tree of states, spread over a window of time scales all of order N (or $\kappa = O ( 1 / N ) )$ ) but separated by orders of magnitude when disorder is strong. We follow speciation through the evolution of the efective potential W, which is defined as minus the logarithm of the probability distribution at time t and whose derivative is minus the score function—the force which drives the generative dynamics. In the variance-exploding difusion process W reads

$$
W ( { \pmb y } , \kappa ) = - \log \int \mathrm { d } { \pmb x } \mathrm { e } ^ { - V ( { \pmb x } ) - \frac { \kappa } { 2 } | { \pmb y } - { \pmb x } | ^ { 2 } } - \frac { N } { 2 } \log \kappa , \qquad \kappa = \frac { 1 } { 2 t } ,\tag{1}
$$

which is the central object that we characterise geometrically in the following as a high dimensional, strongly anisotropic foam. Our results are summarised in Fig. 1, see also Fig. 2 below. As reverse difusion operates, the foam structure of the efective potential subdivides and drives the speciation cascade from long to short difusion time as

• Small $\kappa  0$ . The smoothed potential $W ( \pmb { y } )$ has a single dominant basin;

• Speciation cascade, $\tilde { \kappa } _ { s } / N \lesssim \kappa \lesssim \tilde { \kappa } _ { c } / N$ with $\tilde { \kappa } _ { s } \propto \tilde { \sigma }$ and $\tilde { \kappa } _ { c } \propto \tilde { \sigma } ^ { 2 / H } , 0 < H < 1$ , where $\tilde { \sigma } = { \cal O } ( 1 )$ is the disorder strength appropriately rescaled with $N \ ( \mathrm { S e c . \ 2 . 2 } )$ , such that the range of this regime diverges when $\tilde { \sigma }  \infty . \quad W ( \pmb { y } )$ is a foam of slab-like cells of characteristic width $\overline { { \ell } } _ { \mathrm { L } } \propto \kappa ^ { - 1 / ( \overline { { 2 } } - H ) }$ . The error in reconstructing a clean sample from its noised version decreases from its pure-noise value towards the width $\Delta _ { 1 } = { \cal O } ( 1 )$ (defined in Eq. (24)) of a pure state that is assigned to a given trajectory;

• Selected state, $\kappa \gg \tilde { \kappa } _ { c } / N$ but $\kappa \Delta _ { 1 } \ll 1$ , a window of width $\sim N$ in κ. The posterior distribution (or internal measure) $p ( { \pmb x } \mid { \pmb y } , { \boldsymbol \kappa } ) \propto \exp \left( - V ( { \pmb x } ) - \textstyle { \frac { \kappa } { 2 } } | { \pmb y } - { \pmb x } | ^ { 2 } \right)$ for typical y is a single pure state and the error stays close to $\Delta _ { 1 }$ , so that no further transition happens;

• Within the state, $\kappa \Delta _ { 1 } \gtrsim 1$ . The posterior distribution can be shown to be replica symmetric and stable as it is confined within a state, $W ( \pmb { y } )$ approaches $V ( \pmb { y } )$ , and the error decreases from $\Delta _ { 1 }$ to $1 / \kappa$

Note that the sharpness of the foam represented in Fig. 2, defined as the ratio Γ between the size of the cells and the thickness of the walls, is found to be given by

$$
\Gamma ( \kappa ) : = \kappa \ell _ { \mathrm { L } } ^ { 2 } \propto \kappa ^ { - H / ( 2 - H ) } .\tag{2}
$$

As a result, the cells resolved earlier in the cascade process are also the sharpest in that sense. Importantly, these walls are where excess probability is created when the reverse difusion process is generated with errors in the learnt score (Sec. 6).

## 2 Setting

## 2.1 Generative difusion

As mentioned in the introduction, difusion models rely on transporting dificult to sample data points to an easy to sample measure, and to learn the reverse of that process in order to readily generate new samples. Here, we focus on a variance-exploding (VE) schedule, which allows for the simplest computations, although we expect our results to hold qualitatively in a variancepreserving (VP) setting.

The forward process is free difusion from a data point,

$$
{ \dot { \pmb y } } ( t ) = \sqrt { 2 } { \pmb \eta } ( t ) , \qquad { \pmb y } ( 0 ) = { \pmb x } \sim P _ { 0 } ,\tag{3}
$$

i.e. $\pmb { y } = \pmb { x } + \sqrt { 2 t } \pmb { z }$ , with ${ \boldsymbol z } \sim \mathcal { N } ( { \bf 0 } , { \bf I } _ { N } )$ . At the distribution level, this straightforwardly leads to a convolution between the initial distribution and a Gaussian:

$$
{ \cal P } _ { t } = { \cal P } _ { 0 } \star { \cal N } ( 0 , 2 t { \bf I } _ { N } ) ,\tag{4}
$$

which is a solution to the heat equation $\partial _ { t } P _ { t } = \nabla ^ { 2 } P _ { t }$

Generation runs the reverse process, which starting from ${ \pmb y } ( T ) \sim \mathcal { N } ( { \bf 0 } , 2 T { \bf I } _ { N } )$ for $t = T \gg 1$ can be written as the Langevin equation

$$
\dot { \pmb y } ( t ) = - 2 \pmb s ( \pmb y , t ) + \sqrt { 2 } \pmb \eta ( t ) ,\tag{5}
$$

with the score

$$
\pmb { \mathscr { s } } ( \pmb { y } , t ) = \nabla \log P _ { t } ( \pmb { y } ) .\tag{6}
$$

Indeed, in reverse time $\tau = T - t$ , the density of y then obeys the Fokker–Planck equation

$$
\partial _ { \tau } P _ { \tau } = - \nabla \cdot \left[ 2 \pmb { s } P _ { \tau } \right] + \nabla ^ { 2 } P _ { \tau } ,\tag{7}
$$

which is solved by $P _ { T - \tau }$ , since $s P = \nabla P$

In the following, we will often work with the “precision” of the noise that we introduced above

$$
\kappa : = \frac { 1 } { 2 t } ,\tag{8}
$$

so that $\kappa  0$ is pure noise and $\kappa  \infty$ recovers the data. Note that in the Ornstein–Uhlenbeck $( \mathrm { V P } )$ scheme of [6] (which corresponds to keeping y on the sphere), $\begin{array} { r } { \pmb { x } _ { t } = \mathrm { e } ^ { - t } \pmb { x } _ { 0 } + \sqrt { 1 - \mathrm { e } ^ { - 2 t } } \pmb { z } , } \end{array}$ the same signal-to-noise ratio corresponds to $\kappa = 1 / ( \mathrm { e } ^ { 2 t _ { \mathrm { V P } } } - 1 )$ . In the large N limit, the regime of speciation identified by BBBM, $t _ { \mathrm { V P } } = \textstyle \frac { 1 } { 2 } \log ( N )$ , is therefore $\kappa = O ( 1 / N )$ ), suggesting the rescaling

$$
\tilde { \kappa } = \kappa N ,\tag{9}
$$

a regime which will be at the heart of Secs. 3–5.

## 2.1.1 Two measures

In this work, the data are drawn from a Gibbs measure at unit temperature, $P _ { 0 } ( { \pmb x } ) = \mathrm { e } ^ { - V ( { \pmb x } ) } / \mathcal { Z }$ Formally, this is not a restriction, since any distribution can be written in this form with the potential V =  log $P _ { 0 }$ up to a constant. In this language, we propose to distinguish two important objects. On one hand, we have the external measure describing the distribution of the noisy data

$$
P _ { t } ( \pmb { y } ) = \left( \frac { \kappa } { 2 \pi } \right) ^ { N / 2 } \frac { 1 } { \mathcal { Z } } \int \mathrm { d } \pmb { x } \mathrm { e } ^ { - V ( \pmb { x } ) - \frac { \kappa } { 2 } | \pmb { y } - \pmb { x } | ^ { 2 } } , \quad \mathcal { Z } = \int \mathrm { d } \pmb { x } \mathrm { e } ^ { - V ( \pmb { x } ) } .\tag{10}
$$

We will describe it with the fixed-y potential W, defined through

$$
Z ( \pmb { y } , \kappa ) : = \mathrm { e } ^ { - W ( \pmb { y } , \kappa ) } = \left( \frac { \kappa } { 2 \pi } \right) ^ { N / 2 } \int \mathrm { d } \pmb { x } \mathrm { e } ^ { - V ( \pmb { x } ) - \frac { \kappa } { 2 } | \pmb { y } - \pmb { x } | ^ { 2 } } .\tag{11}
$$

On the other hand, we define the internal measure,

$$
p ( { \pmb x } \mid { \pmb y } , { \boldsymbol \kappa } ) \propto \operatorname { e } ^ { - V ( { \pmb x } ) - \frac { \kappa } { 2 } | { \pmb y } - { \pmb x } | ^ { 2 } } \ .\tag{12}
$$

This is the original measure of x modified by the addition of a quadratic pinning field of precision κ around y. To understand the generation process of typical samples, we will be particularly interested in characterising the internal measure when y is a typical sample drawn from $P _ { t } ( y )$ We will denote its mean by $\hat { \pmb x } ( \pmb y , \kappa ) : = \langle \pmb x \rangle _ { \pmb y , \kappa }$ and its covariance by $\hat { \mathbf { C } } ( \pmb { y } , \kappa ) : = \langle \pmb { x } \pmb { x } ^ { \top } \rangle _ { \pmb { y } , \kappa } -$ $\hat { \pmb x } \hat { \pmb x } ^ { \top }$ .<sup>1</sup> This internal measure has a straightforward interpretation in the difusion setting, as the conditional mean ${ \hat { \mathbf { \mathbf { x } } } } ( \mathbf { \mathbf { \mathit { y } } } , \kappa )$ is the Bayes-optimal denoiser given a sample corrupted by the Gaussian process at precision κ. It is directly related to the score through the exact Tweedie identity [38]

$$
\nabla W ( { \pmb y } , \kappa ) = - { \pmb s } = \kappa \big ( { \pmb y } - { \hat { \pmb x } } ( { \pmb y } , \kappa ) \big ) .\tag{13}
$$

This identity will also be relevant to study the geometry of the efective potential W, through the equality on the Hessian

$$
\nabla \nabla W ( { \pmb y } , \kappa ) = \kappa { \bf I } _ { N } - \kappa ^ { 2 } \hat { \bf C } ( { \pmb y } , \kappa ) .\tag{14}
$$

Through the Tweedie identity, computing the score therefore amounts to solving a denoising problem. As such, it will be natural to characterise the reverse process through the denoising error of the Bayes-optimal xˆ, that is the minimum mean-squared error (MMSE),

$$
\Delta ( \kappa ) : = \frac { 1 } { N } \mathbb { E } \left| x _ { 0 } - \hat { x } ( \pmb { y } , \kappa ) \right| ^ { 2 } , \qquad \pmb { y } = \pmb { x } _ { 0 } + \frac { 1 } { \sqrt { \kappa } } z\tag{15}
$$

with E over $\mathbf { \boldsymbol { x } } _ { 0 } \sim P _ { 0 }$ , the Gaussian white noise $z ,$ and the disorder. Since y is generated from $\mathbf { \boldsymbol { x } } _ { 0 } \sim \mathbf { \boldsymbol { P } } _ { 0 }$ , Bayes’ rule implies that given ${ \mathbf { \psi } } _ { \mathbf { \psi } } \mathbf { , \psi } \mathbf { x } _ { \mathrm { 0 } }$ is itself a sample of $p ( { \pmb x } \mid { \pmb y } , { \kappa } )$ . This is the Nishimori identity, which allows the ground truth to be exchanged with a replica and will be essential to compute the MMSE analytically.

## 2.1.2 A sharp limit: score shocks

In this work, we choose to fix the temperature of the Gibbs measure to unity for convenience. However, in order to intuitively understand the geometry of W, it will be useful to think about a “hardened”, zero temperature limit of the problem. For $\beta > 0$ define

$$
Z _ { \beta } ( \pmb { y } , \kappa ) = \left( \frac { \beta \kappa } { 2 \pi } \right) ^ { N / 2 } \int \mathrm { d } \pmb { x } \exp \left( - \beta \left[ V ( \pmb { x } ) + \frac { \kappa } { 2 } | \pmb { y } - \pmb { x } | ^ { 2 } \right] \right) , \qquad W _ { \beta } = - \frac { 1 } { \beta } \log Z _ { \beta } ( \pmb { x } ) ,\tag{16}
$$

At $\beta = 1 , W _ { 1 } \equiv W$ is the smoothed potential of the difusion. As $Z _ { \beta }$ solves the heat equation with difusion constant $1 / \beta$ , a Cole-Hopf transform leads $W _ { \beta }$ to follow a deterministic KPZ equation (see e.g. [36]) with viscosity $1 / \beta$

$$
\partial _ { t } W _ { \beta } = \frac { 1 } { \beta } \nabla ^ { 2 } W _ { \beta } - | \nabla W _ { \beta } | ^ { 2 } , \qquad W _ { \beta } ( \pmb { y } , 0 ) = V ( \pmb { y } ) ,\tag{17}
$$

and thus $\pmb { v } = 2 \nabla W _ { \beta }$ is a Burgers velocity field [39, 40] (also called the Polchinski equation in the context of the renormalisation group, see e.g. [41]). We can thus identify $\begin{array} { r } { s = - \frac { 1 } { 2 } v } \end{array}$ , extending the mapping discussed in Sarkar [42] to N dimensions.

The inviscid limit $\beta \to \infty$ leads to the Hopf–Lax solution (also called the Moreau envelope of $V ( { \pmb x } ) )$ 0

$$
W _ { \infty } ( \pmb { y } , t ) = \operatorname* { m i n } _ { \pmb { x } } \left[ V ( \pmb { x } ) + \frac { | \pmb { y } - \pmb { x } | ^ { 2 } } { 4 t } \right] ,\tag{18}
$$

that is the zero-temperature M´ezard–Parisi problem [23, 37] and inviscid Burgers with random initial data [36]. It has genuine shocks where the minimiser jumps, which are rounded at finite $\beta .$ . Note that since $\beta V$ has the statistics of $V$ at a stronger disorder (Sec. 2.2), this sharp limit is also the one approached at strong disorder. We shall study this shock-like picture in Secs. 4.3 and 4.4.

## 2.2 The data: a long-range random potential and its Parisi tree

We now detail the data model we are considering in this study. We take $P _ { 0 } ( { \pmb x } ) = \mathrm { e } ^ { - V ( { \pmb x } ) } / \mathcal { Z }$ on the sphere $\Omega = S ^ { N - 1 } ( L \sqrt { N } )$ (a hard-wall ball, cube or torus of side $L \sqrt { N }$ can be used interchangeably), with V a centred Gaussian potential of covariance

$$
\overline { { V ( \pmb { x } _ { 1 } ) V ( \pmb { x } _ { 2 } ) } } = B ( | \pmb { x } _ { 1 } - \pmb { x } _ { 2 } | ) = B ( 0 ) - \sigma ^ { 2 } | \pmb { x } _ { 1 } - \pmb { x } _ { 2 } | ^ { 2 H } , \qquad 0 < H < 1 ,\tag{19}
$$

in the inertial range $\xi _ { \mathrm { S R } } \ll | { \pmb x } _ { 1 } - { \pmb x } _ { 2 } | \ll \xi _ { \mathrm { L R } }$ , with $\xi _ { \mathrm { S R } } = { \cal O } ( 1 )$ a short-range “ultraviolet” regulariser and $\xi _ { \mathrm { L R } }$ the long-range cutof giving the largest correlation length, that we will choose to be equal to the size of the ball. Overlines will henceforth indicate averages over the realisation of the potential, and the temperature is set to one.<sup>2</sup> The Hurst exponent $H \in ( 0 , 1 )$

controls the roughness: $H  0$ corresponds to log-correlated-like landscapes $[ 4 3 , 4 4 ] ; H  1$ approaches smooth quadratic covariance. The case $H = 1 / 2$ in dimension $N = 1$ is a Brownian potential which has received a lot of attention [25, 27, 45, 46].

Configurations on the sphere are typically $O ( { \sqrt { N } } )$ apart. For the large-N limit to be nontrivial, these distances must lie inside the inertial range, and the variance of the potential across them must be extensive, so that energy and entropy compete at the same order. We thus adopt the canonical large-N scaling

$$
\sigma = \tilde { \sigma } N ^ { \left( 1 - H \right) / 2 } ,\tag{20}
$$

with $\xi _ { \mathrm { L R } } = L \sqrt { N }$ and $L , \tilde { \sigma } = O ( 1 )$ , under which the large scale square excursion of the potential $\tilde { \sigma } ^ { 2 } L ^ { 2 H } N$ is extensive in $N$ . Note that all results shall depend on $\tilde { \sigma }$ and $L$ only through $\tilde { \sigma } L ^ { H }$ once lengths are measured in units of $L$

In the large-N limit, $P _ { 0 }$ can be studied by the replica method and is described by the overlaps $Q _ { a b } = \pmb { x } _ { a } \cdot \pmb { x } _ { b } / N$ between configurations. For geometric statements we use instead squared distances per coordinate,

$$
D _ { a b } ^ { 2 } = \frac { | { \pmb x } _ { a } - { \pmb x } _ { b } | ^ { 2 } } { N } = 2 ( L ^ { 2 } - Q _ { a b } ) ,\tag{21}
$$

together with the physical length $\ell = \sqrt { N } D$ and the width $\Delta = D ^ { 2 } / 2$ . In these variables the covariance reads $B ( \sqrt { N } D ) = N B ( D ^ { 2 } )$ , with $B ( z ) = B ( 0 ) / N - \tilde { \sigma } ^ { 2 } z ^ { H }$

Let us summarise the most salient features of the Gibbs measure $P _ { 0 }$ , obtained with the replica method [23, 24]; we refer unfamiliar readers to Appendix A, where we rederive these results in our notation following [47]. The first important result is that the replica-symmetric solution $Q _ { a b } = q$ for all $a \neq b$ is stable only for $\tilde { \sigma } < \tilde { \sigma } _ { \mathrm { A T } }$ , where

$$
\tilde { \sigma } _ { \mathrm { A T } } ^ { 2 } L ^ { 2 H } = \frac { 1 } { 2 ^ { H } H ( 1 - H ) [ ( 1 - H ) / ( 2 - H ) ] ^ { H } } .\tag{22}
$$

Since we want a highly structured distribution, we place ourselves beyond this de Almeida– Thouless line, $\tilde { \sigma } > \tilde { \sigma } _ { \mathrm { A T } }$ , where the measure undergoes full replica symmetry breaking (fRSB). We call this regime strong disorder, and will further typically assume $\tilde { \sigma } \gg 1$ for the asymptotic statements.

The Parisi function $u ( q )$ , i.e. the probability that two configurations drawn from $P _ { 0 }$ have an overlap smaller than $q ,$ is then continuous between the two plateau values $q _ { 0 }$ and $q _ { 1 }$ of the overlap,

$$
u ( q _ { 0 } < q < q _ { 1 } ) = \frac { 2 - H } { 2 \tilde { \sigma } \sqrt { H ( 1 - H ) } } D ^ { - H } , \quad \mathrm { w i t h } D ^ { 2 } = 2 ( L ^ { 2 } - q ) .\tag{23}
$$

The upper plateau sets the width of a pure state,

$$
\Delta _ { 1 } = L ^ { 2 } - q _ { 1 } = \left[ \frac { 1 } { 2 ^ { H } \tilde { \sigma } ^ { 2 } H ( 1 - H ) } \right] ^ { 1 / H } ,\tag{24}
$$

which shrinks as $\tilde { \sigma } ^ { - 2 / H }$ at strong disorder, while the lower plateau,

$$
q _ { 0 } = \frac { L ^ { 2 } } { 2 - H } ,\tag{25}
$$

i.e. $D _ { 0 } = L \sqrt { 2 ( 1 - H ) / ( 2 - H ) }$ , does not depend on the disorder strength. At the two ends, u takes the values $u _ { 0 } = u ( q _ { 0 } ^ { + } ) \propto 1 / \tilde { \sigma }$ and $u _ { 1 } = u ( q _ { 1 } ^ { - } ) = ( 2 - H ) / 2$ , the latter independent of $\tilde { \sigma } .$ . It will be convenient to label the levels of the hierarchy by the Parisi level $u ( q )$ , and to invert the relation:

$$
D ( u ) = \left[ \frac { 2 - H } { 2 \tilde { \sigma } \sqrt { H ( 1 - H ) } } \right] ^ { 1 / H } u ^ { - 1 / H } , \qquad u _ { 0 } < u < u _ { 1 } .\tag{26}
$$

Physically, $P _ { 0 }$ decomposes into pure states $\alpha$ , i.e. its ergodic components, with weights $w _ { \alpha } \propto \mathrm { e } ^ { - F _ { \alpha } }$ , centres $\mathbf { \Omega } _ { m } { } ^ { \alpha }$ and thermal width $\Delta _ { 1 }$ per coordinate. Two states drawn from $P _ { 0 }$ are at a distance of at least $D ( u )$ with probability u [48], so that the physical distance between them is distributed as

$$
\rho ( \ell ) \propto \ell ^ { - 1 - H } , \qquad \ell _ { 1 } < \ell < \ell _ { 0 } ,\tag{27}
$$

with atoms of weight $u _ { 0 }$ at $\ell _ { 0 } = \sqrt { N } D _ { 0 }$ and $H / 2$ at $\ell _ { 1 } = \sqrt { N } D _ { 1 }$ , the latter corresponding to two configurations in the same state. The exponent is fixed by the roughness of V alone. Since $\ell _ { 0 } = O ( \sqrt { N } )$ and $\ell _ { 1 } \propto \sqrt { N } \tilde { \sigma } ^ { - 1 / H }$ , the whole hierarchy lies inside the inertial range as $N \to \infty$ and strong disorder makes it broad, $D _ { 1 } / D _ { 0 } \propto \tilde { \sigma } ^ { - 1 / H }  0$

The states are organised in an ultrametric tree, the Parisi tree of states (M´ezard–Virasoro) [26, $4 9 ] \ ( \mathrm { A p p . \ A . 5 } )$ . Two states whose branches separate at level $q$ have overlap q. At each level, the free energies of the competing sub-clusters behave as the low-lying levels of Derrida’s random energy model [50]. They are independent, with a density that grows exponentially as $\mathrm { e } ^ { u ( q ) F } \ [ 2 6 , 4 9 , 5 1 ]$ , so that only a few of them carry a significant weight. At the finest level, the participation ratio of the pure states is [26, 52]

$$
\overline { { \sum _ { \alpha } w _ { \alpha } ^ { 2 } } } = 1 - u _ { 1 } = \frac { H } { 2 } .\tag{28}
$$

Although infinitely many states contribute as $N  \infty .$ , a few of them carry most of the weight, and the measure is condensed.

## 2.3 Pure state decomposition of the noised measure

The decomposition of $P _ { 0 }$ into pure states carries over to $P _ { t }$ . Writing $\begin{array} { r } { P _ { 0 } = \sum _ { \alpha } w _ { \alpha } P _ { 0 } ^ { ( \alpha ) } } \end{array}$ , with $P _ { 0 } ^ { ( \alpha ) }$ the measure restricted to the state α and $w _ { \alpha } = \mathrm { e } ^ { - F _ { \alpha } } / \mathcal { Z }$ , the convolution (4) gives

$$
P _ { t } ( \pmb { y } ) = \sum _ { \alpha } w _ { \alpha } \left( P _ { 0 } ^ { ( \alpha ) } \star \mathcal { N } ( 0 , 2 t \mathbf { I } _ { N } ) \right) ( \pmb { y } ) = \frac { 1 } { \mathcal { Z } } \sum _ { \alpha } \mathrm { e } ^ { - F _ { \alpha } ( \pmb { y } ) } ,\tag{29}
$$

where $F _ { \alpha } ( \pmb { y } ) = F _ { \alpha } - \log \big ( P _ { 0 } ^ { ( \alpha ) } \star \mathcal { N } ( 0 , 2 t \mathbf { I } _ { N } ) \big ) ( \pmb { y } )$ is the free energy of the state α pinned around $\mathbf { \nabla } _ { \mathbf { \pmb { y } } } .$ . Since $\mathrm { e } ^ { - W } = \mathcal { Z } P _ { t }$ , the smoothed potential is exactly a soft minimum over the states,

$$
W ( { \pmb y } , \kappa ) = - \log \sum _ { \alpha } \mathrm { e } ^ { - F _ { \alpha } ( { \pmb y } ) } .\tag{30}
$$

Even though the initial measure $P _ { 0 }$ is fragmented in pure states, by increasing t the noised versions of the pure states probability distributions merge leading to a cascade of transitions and eventually to a unique pure state for P —coinciding with a Gaussian when $t \to \infty$ . Understanding this cascade is the aim of the next section. We will do it focusing on the potential $W$ which is equal (up to a constant) to minus the log of $P _ { t }$

## 3 The onset of speciation

## 3.1 The speciation time

One of the key results of BBBM is the identification of the so-called speciation time $t _ { s }$ . Strictly speaking, there are in fact two possible definitions of $t _ { s }$

The first, which motivates its name, is one of attribution. Assuming a distribution is multimodal and each mode can be identified with a class, it is natural to define $t _ { s }$ as the time at which a given trajectory becomes committed to one of the classes. Achilli et al. [7] makes this precise through Bayes classification of the noised configuration, locating $t _ { s }$ where the free-entropy diference between two classes becomes comparable to its fluctuations.

The second is instead based on geometry. As the long-time distribution is a unimodal Gaussian, one can ask when non-trivial modes start to emerge in the potential W. This question itself has two answers: (i) the time at which the long-time basin of W loses convexity, a spinodal; and (ii) the time at which new stationary points of W first appear away from it. The two need not coincide, and in between may lie a binodal region in which the long-time basin is still convex while other basins already exist, as in a first-order transition. We come back to this distinction in Sec. 4.

## 3.2 A spectral criterion

In the mixture of two Gaussians of equal weights studied in BBBM, all of these coincide: the loss of convexity is a pitchfork bifurcation that creates the two new basins, and trajectories commit at the same time. The speciation time can then be located by the simplest of these criteria, the loss of convexity of the long-time basin, which takes the form of a spectral criterion.

While initially proposed in the variance-preserving context, the argument only uses loss of convexity and therefore transposes directly to our variance-exploding setting. Let $y ^ { \star }$ be the unique minimum of $W$ at long times. By Eq. (13), $\pmb { y } ^ { \star } = \hat { \pmb { x } } ( \pmb { y } ^ { \star } , \kappa )$ , and by Eq. (14) it destabilises when

$$
\begin{array} { r } { \kappa _ { s } \hat { c } _ { \mathrm { m a x } } ( \pmb { y } ^ { \star } , \kappa _ { s } ) = 1 , } \end{array}\tag{31}
$$

where $\hat { c } _ { \mathrm { m a x } }$ is the largest eigenvalue of the posterior covariance $\hat { \mathbf { C } } .$ . Now, assume that this occurs at small κ (which we can check a posteriori). It then follows that C<sup>ˆ</sup> is well approximated by the data covariance C of $P _ { 0 } .$ , giving the spectral criterion $t _ { s } \simeq c _ { \mathrm { m a x } } ( \mathbf { C } ) / 2$ . Therefore the long time behaviour of the reverse process can be predicted by the bare covariance matrix of the data.

## 3.3 Data covariance of the fRSB measure

We now set out to explicitly compute the speciation time for our data model. We start with a pure state decomposition,

$$
\mathbf { C } = \underbrace { \sum _ { \alpha } w _ { \alpha } \mathbf { C } ^ { \alpha } } _ { \mathbf { C } _ { \mathrm { { i n t r a } } } } + \underbrace { \sum _ { \alpha } w _ { \alpha } ( \pmb { m } ^ { \alpha } - \pmb { \bar { m } } ) ( \pmb { m } ^ { \alpha } - \pmb { \bar { m } } ) ^ { \top } } _ { \mathbf { C } _ { \mathrm { { i n t e r } } } } ,\tag{32}
$$

which we have itself decomposed in a intrastate and an interstate contribution. Here $m ^ { \alpha } = \langle \pmb { x } \rangle _ { \alpha }$ is the mean within a pure state, $\begin{array} { r } { \bar { m } = \sum _ { \alpha } w _ { \alpha } \pmb { m } ^ { \alpha } } \end{array}$ , and $C _ { i j } ^ { \alpha } = \langle x _ { i } x _ { j } \rangle _ { \alpha } - \langle x _ { i } \rangle _ { \alpha } \langle x _ { j } \rangle _ { \alpha }$ describes intrastate fluctuations, while $\mathbf { C } _ { \mathrm { { i n t e r } } }$ represents interstate contributions.

We start by computing the interstate covariance. For any two states, the study of the microstructure of ultrametricity of M´ezard and Virasoro [49], adapted to a Gaussian measure, gives

$$
\overline { { { m _ { i } ^ { \alpha } m _ { j } ^ { \gamma } \mid Q _ { \alpha \gamma } = q } } } = \delta _ { i j } q ,\tag{33}
$$

with $q = q _ { 1 }$ for $\gamma = \alpha$ . Therefore, when averaging over the disorder,

$$
\frac { 1 } { N } \overline { { \mathrm { T r ~ } \mathbf { C } _ { \mathrm { i n t e r } } } } = q _ { 1 } - \int _ { 0 } ^ { 1 } \mathrm { d } u q ( u ) = q _ { 1 } - \bar { q } ,\tag{34}
$$

with $\begin{array} { r } { \bar { q } = \int \mathrm { d } q P ( q ) q } \end{array}$ the mean overlap, and $P ( q ) = \mathrm { d } u / \mathrm { d } q$ the distribution of overlaps, which includes peaks of weight $u _ { 0 }$ at $q _ { 0 }$ and $1 - u _ { 1 }$ at $q _ { 1 }$ . Rotational invariance then implies

$$
\overline { { ( { \bf C } _ { \mathrm { i n t e r } } ) _ { i j } } } = \Delta _ { m } \delta _ { i j } ,\tag{35}
$$

with $\Delta _ { m } : = q _ { 1 } - \bar { q }$ . Integrating by parts and using the fRSB solution of Sec. 2.2,

$$
\Delta _ { m } = \int _ { q _ { 0 } } ^ { q _ { 1 } } u ( q ) \mathrm { d } q = \frac { u _ { 0 } D _ { 0 } ^ { 2 } - u _ { 1 } D _ { 1 } ^ { 2 } } { 2 - H } = \frac { D _ { 0 } ^ { 2 - H } - D _ { 1 } ^ { 2 - H } } { 2 \tilde { \sigma } \sqrt { H ( 1 - H ) } } \propto \frac { 1 } { \tilde { \sigma } } .\tag{36}
$$

The above diagonal expression is only true on average, as it suppresses the random orientations of possible large eigenvectors. Let M be the $N \times \mathcal { N } _ { s }$ matrix whose columns are the vectors $\mathbf { \Omega } _ { m } { } ^ { \alpha }$ , let w be the column vector of state weights, and set $\mathbf { A } _ { \pmb { w } } = \mathrm { d i a g } ( \pmb { w } ) - \pmb { w } \pmb { w } ^ { \top }$ . Then $\begin{array} { r } { { \bf { C } } _ { \mathrm { { i n t e r } } } = { \bf M } { \bf { A } } _ { w } { \bf M } ^ { \top } , \frac { 1 } { N } { \bf M } ^ { \top } { \bf M } = { \bf Q } } \end{array}$ , the latter being the full matrix of overlaps $Q _ { \alpha \gamma } = \pmb { m } ^ { \alpha } \cdot \pmb { m } ^ { \gamma } / N$ between state centres. Hence the nonzero spectra obey

$$
\mathrm { S p e c } _ { \neq 0 } ( \mathbf { C } _ { \mathrm { i n t e r } } ) = N \mathrm { S p e c } _ { \neq 0 } ( \mathbf { G } ) ,\tag{37}
$$

with $\mathbf { G } = \mathbf { A } _ { w } ^ { 1 / 2 } \mathbf { Q } \mathbf { A } _ { w } ^ { 1 / 2 }$ . This reduces the required spectral problem to the overlap matrix and the statistics of the weights.

In the simplest case of two states, G has rank one and Eq. (37) gives the unique nonzero eigenvalue $2 N w ( 1 - w ) ( q _ { 1 } - Q _ { \alpha \gamma } ) = N w ( 1 - w ) D _ { \alpha \gamma } ^ { 2 }$ . In the fRSB case, the efective number of states contributing to the measure is given by the participation ratio, which as mentioned above is O(1) [Eq. (28)]. This leads us to assume that the efective rank

$$
r _ { \mathrm { e f f } } ( \mathbf { G } ) = \frac { \mathrm { T r } \mathbf { G } } { \lambda _ { \mathrm { m a x } } ( \mathbf { G } ) } = \frac { \mathrm { T r } { \mathbf { C } } _ { \mathrm { i n t e r } } } { c _ { \mathrm { m a x } } ( \mathbf { C } _ { \mathrm { i n t e r } } ) }\tag{38}
$$

remains $O ( 1 )$ as $N  \infty$ , and carries no power-law dependence on $\tilde { \sigma } .$ . While the participation ratio is known to fluctuate from sample to sample, the scaling of the ratio for a typical sample is the same as the one of the disordered average (as we check below). Using Eq. (37) we thus get

$$
c _ { \mathrm { m a x } } ( { \bf C } _ { \mathrm { i n t e r } } ) = N \lambda _ { \mathrm { m a x } } ( { \bf G } ) \sim N g _ { H } \Delta _ { m } \propto \frac { N } { \tilde { \sigma } } ,\tag{39}
$$

where $g _ { H } = O ( 1 )$ incorporates the efective rank and the sample-to-sample fluctuations of $\operatorname { T r } \mathbf { G }$ We finally consider the connected fluctuations inside the pure states, $\begin{array} { r } { \mathbf { C } _ { \mathrm { i n t r a } } = \sum _ { \alpha } w _ { \alpha } \mathbf { C } ^ { \alpha } } \end{array}$ . For a typical state, we write ${ \bf K } = ( { \bf C } ^ { \alpha } ) ^ { - 1 }$ the stifness matrix. As the fRSB solution is marginally stable, we assume that this marginality appears in the physical intrastate susceptibility through a soft lower edge of the normalised stifness spectrum,

$$
\rho _ { K } ( \lambda ) = f \lambda ^ { \theta } + o ( \lambda ^ { \theta } ) , \qquad \lambda \to 0 , \qquad 0 < \theta < 1 .\tag{40}
$$

$\mathrm { A t }$ finite $N ,$ the typical smallest stifness is estimated by requiring an order-one number of eigenvalues below it, $\begin{array} { r } { N \int _ { 0 } ^ { \lambda _ { \operatorname* { m i n } } } \mathrm { d } \lambda \rho _ { K } ( \lambda ) \simeq 1 } \end{array}$ , which gives

$$
\lambda _ { \operatorname* { m i n } } \simeq \left( \frac { 1 + \theta } { N f } \right) ^ { 1 / ( 1 + \theta ) } ,\tag{41}
$$

with $c _ { \operatorname* { m a x } } ( \mathbf { C } ^ { \alpha } ) ~ = ~ \lambda _ { \operatorname* { m i n } } ^ { - 1 }$ . The density of soft modes $f$ is fixed by the width of the states, $N ^ { - 1 } \operatorname { T r } \mathbf { C } ^ { \alpha } = \Delta _ { 1 }$ , under a one-scale assumption on the stifness spectrum (App. B.2), so that

$$
c _ { \mathrm { m a x } } ( { \bf C } ^ { \alpha } ) \propto N ^ { 1 / ( 1 + \theta ) } \tilde { \sigma } ^ { - 2 / H } .\tag{42}
$$

The same scaling bounds the weighted intrastate contribution, since $\begin{array} { r } { c _ { \mathrm { m a x } } ( { \bf C } _ { \mathrm { i n t r a } } ) \leq \sum _ { \alpha } w _ { \alpha } c _ { \mathrm { m a x } } ( { \bf C } ^ { \alpha } ) } \end{array}$ For every $\theta > 0$ , this contribution is subextensive in N and is therefore subdominant to the $O ( N )$ interstate eigenvalue.

Combining the two contributions, the spectral criterion (31) gives

$$
\tilde { \kappa } _ { s } \propto \tilde { \sigma } \quad \longrightarrow \quad t _ { s } \propto \frac { N } { \tilde { \sigma } } .\tag{43}
$$

Since $\kappa _ { s } = O ( 1 / N )$ , using the $\kappa = 0$ covariance is consistent at the level of scaling.

Let us interpret the role of the disorder strength. Here, the dependence does not come from the width of the states, whose narrowing, $\Delta _ { 1 } \propto \tilde { \sigma } ^ { - 2 / H }$ , only afects the subdominant intrastate term. Instead, it comes from the spread of the state centres, $\Delta _ { m } = q _ { 1 } - \bar { q } ;$ which is half the mean squared distance per coordinate between the centres of two states drawn from $P _ { 0 }$ . At strong disorder this spread decreases as $1 / \tilde { \sigma }$ , because the Gibbs weight concentrates on few, closely related states. Two samples of the data then typically belong to the same deep branch, even though the hierarchy extends up to distances $D _ { 0 } = O ( 1 )$ . Resolving this smaller spread requires less noise, and speciation occurs at larger $\kappa ,$ i.e. later in the reverse process.

As mentioned above, however, this estimate is annealed due to the disorder-averaged covariance, through the assumption that the trace of $\mathbf { C } _ { \mathrm { { i n t e r } } }$ in a typical sample scales as its average. Since the weights of the pure states fluctuate strongly from one sample to the next, the measure being condensed (Sec. 2.2), it remains to check whether Eq. (43) holds for a given sample. We address this question in the following section.

## 4 Speciation cascade and generative foam

Section 3 located the onset of speciation at $\tilde { \kappa } _ { s } \propto \tilde { \sigma } .$ , i.e. at $\kappa = O ( 1 / N )$ , consistent with the work of BBBM and Achilli et al. [7]. The challenge with this scale in our disordered setting where the score cannot be written explicitly is that the part of the pinning that distinguishes two states α and $\gamma$ in W, κ $: y \cdot ( m ^ { \alpha } - m ^ { \gamma } )$ , is of order one, whereas the free energy, and with it the replicated action, is of order N. A standard replica computation, based on an extensive saddle point, cannot see the competition between states and is therefore not suitable to study this long-time regime.

Two questions remain open. The spectral estimate of Sec. 3 is annealed, and an independent, sample-wise derivation of the onset is needed. Moreover, unlike the two-component mixture [4, 6], the fRSB measure contains a hierarchy of competing pure states. It is thus interesting to follow how the reverse “cascade” process resolves it beyond the first split. Increasing $\kappa ,$ the reverse process eventually needs to resolve finer branches of the Parisi tree. At intermediate scales, it appears natural that $W ( \pmb { y } , \kappa )$ is locally dominated by diferent states in diferent regions of y-space. We call the resulting collection of basins, separated by shock-like walls, the generative foam, see Fig. 2 for a low-dimensional illustration.

In this section, we address both questions with a Larkin argument, which balances the $O ( 1 )$ pinning against the free-energy gaps between branches of the Parisi tree and holds sample by sample (Sec. 4.1). It gives an independent estimate of the onset, follows the resolution down the tree (Sec. 4.2), and fixes the geometry of the foam (Sec. 4.3).

## 4.1 The Larkin balance

The Larkin argument was first introduced to study the collective pinning of vortex lattices $[ 3 4$ 35]. Our problem is its analogue on a N-dimensional sphere [25], as one configuration x in the random potential V is confined around y by $\kappa | \pmb { y } - \pmb { x } | ^ { 2 } / 2$ . Beyond the Larkin scale several metastable configurations compete, and the preferred one jumps as the centre y of the well moves, producing a shock in the “zero temperature” Hopf–Lax limit [36].

The Larkin balance can therefore be applied, but its usual geometric interpretation must be adapted to the difusion setting in large dimensions. Indeed, at the speciation scale, $\kappa = \tilde { \kappa } / N$ , a typical noised configuration has $| \pmb { y } | = O ( N )$ , whereas the competing states remain on the sphere $| { \pmb x } | = L \sqrt { N }$ . The distance $| y - x |$ is consequently dominated by the noise, such that we need to be careful in comparing the pinned free energies (29) of two competing pure states α and $\gamma$ of $P _ { 0 }$ , with centres $\mathbf { \Omega } _ { m } { } ^ { \alpha }$ and $m ^ { \gamma }$ and free energies $F _ { \alpha }$ and $F _ { \gamma }$

In the following we will make a rigid-state approximation

$$
F _ { \alpha } ( { \pmb y } ) = F _ { \alpha } - \log \left. \mathrm { e } ^ { - \frac { \kappa } { 2 } | { \pmb y } - { \pmb x } | ^ { 2 } } \right. _ { \alpha } \simeq F _ { \alpha } + \frac { \kappa } { 2 } \left. | { \pmb y } - { \pmb x } | ^ { 2 } \right. _ { \alpha } = F _ { \alpha } + \frac { \kappa } { 2 } | { \pmb y } - { \pmb m } ^ { \alpha } | ^ { 2 } + \mathrm { c o n s t } ,\tag{44}
$$

and likewise for $F _ { \gamma } ( \pmb { y } )$ , with $\langle \cdot \rangle _ { \alpha }$ the average within the state $\alpha .$ . This is a good approximation because the x-dependent terms of the pinning field lead to non-extensive contributions of order one in the speciation regime. As such, they can be treated perturbatively, thus justifying the expansion above (see also [4]), while the intra-state fluctuations are, to leading order, the same for all states at a given y. Starting from this expression, we can compare the two contributions. Since the two states have the same norm $| m ^ { \alpha } | ^ { 2 } = N q _ { 1 }$ to leading order in N, their relative free energy is

![](images/d26047e6961a2be81a73e37cb54431791befc6b9470141732c4d2fe143a1baf2.jpg)  
Figure 2: Smoothing of a rough landscape with $H = 3 / 4$ under the variance-exploding difusion process in (a) $N = 1$ and (b) $N = 2$ dimensions. The smoothed potential W is shown in its zerotemperature (Hopf–Lax) form, Eq. (18), for easier visualisation. In (a), markers show shocks, where the minimiser ${ \pmb x } ^ { \star } ( { \pmb y } )$ of the Hopf–Lax solution jumps between two competing basins and W has a kink; dashed lines continue each branch a short distance beyond its shock. For (b), we show a hill-shaded W, with the colour scale normalised in each panel (dark: low W, i.e. high P ). Shocks appear as creases separating the cells of a “foam”. As t grows the cells merge hierarchically, from a dense foam $( t _ { 1 } )$ to a few large basins (t<sub>2</sub>) and a single remaining shock $\left( t _ { 3 } \right)$ , until one basin fills the box (t<sub>4</sub>). The times are chosen separately for the two rows to show comparable stages.

$$
F _ { \alpha } ( { \pmb y } ) - F _ { \gamma } ( { \pmb y } ) = F _ { \alpha } - F _ { \gamma } - \kappa { \pmb y } \cdot { \pmb d } ,\tag{45}
$$

with $\pmb { d } = \pmb { m } ^ { \alpha } - \pmb { m } ^ { \gamma } : = \ell \pmb { n }$ and $| { \boldsymbol n } | = 1$ . All terms involving $| y | ^ { 2 }$ , as well as the equal radial contributions from the two states, cancel from this diference.

Let us decompose y as ${ \pmb y } = \phi \ell { \pmb n } + { \pmb y } _ { \bot }$ , with ${ \pmb y } _ { \perp } \cdot { \pmb d } = 0$ , such that the latter contribution also disappears from Eq. (45). Here $\phi = O ( 1 )$ is the position of y along the axis joining the two centres, with—say— $\begin{array} { r } { \mathbf { \nabla } \cdot \phi = \frac { 1 } { 2 } } \end{array}$ when $y = m ^ { \alpha }$ and $\phi = - \textstyle { \frac { 1 } { 2 } }$ when $y = m ^ { \gamma }$ . Equation (45) then gives the crossing condition $\kappa \bar { \phi } \ell ^ { 2 } = E _ { \mathrm { g a p } } ( \ell )$ , where $E _ { \mathrm { g a p } } ( \ell )$ is the typical free-energy diference between the deepest state within distance ℓ and its closest competitor. Since $\phi = O ( 1 )$ , this defines an operative Larkin scale such that

$$
\begin{array} { r } { \kappa \ell _ { \mathrm { L } } ^ { 2 } \sim E _ { \mathrm { g a p } } ( \ell _ { \mathrm { L } } ) . } \end{array}\tag{46}
$$

The energy balance has the familiar Larkin form, but here it arises only after projecting the $O ( N )$ noised configuration onto the $O ( \sqrt { N } )$ direction separating two competing states.

The subtle point now is to estimate the relevant energy gap in high dimension. There, a shell at distance ℓ from a reference minimum can be divided into $\mathcal { M } \sim \mathrm { e } ^ { c N }$ macroscopically distinct cells, whose mutual separations are an $O ( 1 )$ fraction of ℓ. This exponential multiplicity is consistent with complexity calculations for high-dimensional random landscapes [47, 53]. This count is much larger than in low dimensions but much smaller than the naive $\ell ^ { \bar { N } }$ count that ignores correlations. Treating the macro-cells as efectively independent, extreme-value statistics of Gaussian variables of variance $\sigma ^ { 2 } \ell ^ { 2 H }$ then gives

![](images/25570da147e674b1772035b91cd187e1f87d75a5f9b4020de6cbf503de145677.jpg)  
Figure 3: Gap- versus depth-Larkin balance, sketched on a one-dimensional rough potential $V ( x )$ with $H = 3 / 4$ . The dashed vertical lines delimit the window of width L supporting the data (black), the grey curve being a hypothetical continuation of the disorder beyond it for a softly constrained potential. The data state (filled blue dot) and its closest competitor (open blue circle) are a distance $\ell _ { \mathrm { L } }$ apart and difer by $E _ { \mathrm { g a p } }$ . From a generic point of the landscape (open orange circle), the deepest state within reach (filled orange dot) lies $E _ { \mathrm { d e p t h } }$ lower, at a distance $\ell _ { \mathrm { D } }$ . The dashed parabolas illustrate the balances $\kappa \ell _ { \mathrm { L } } ^ { 2 } \sim E _ { \mathrm { g a p } }$ and $\kappa \ell _ { \mathrm { D } } ^ { 2 } \sim E _ { \mathrm { d e p t h } }$ (note that here their stifnesses are not equal in this one-dimensional sketch, which is only illustrative). As the depth greatly exceeds the gap, $E _ { \mathrm { d e p t h } } \sim N E _ { \mathrm { g a p } }$ in high dimension, the depth-Larkin length reaches beyond the data window.

$$
E _ { \mathrm { g a p } } ( \ell ) \sim \frac { \sigma \ell ^ { H } } { \sqrt { N } } = \tilde { \sigma } \left( \frac { \ell } { \sqrt { N } } \right) ^ { H } .\tag{47}
$$

Here, the $1 / \sqrt { N }$ suppression relative to the naive $\sigma \ell ^ { H }$ is the crucial correction: in many dimensions, low-lying states are exponentially numerous, and their gaps are correspondingly compressed by an extreme-value factor $1 / \sqrt { 2 }$ log $\overline { { \mathcal { M } } } \sim 1 / \sqrt { c N }$

Importantly, this scaling is consistent with the replica solution. First, at large scales, i.e. for the deeper states corresponding to $\ell \sim \sqrt { N }$ , we find an energy gap of order one as expected in fRSB. Second, and more precisely, at temperature $T .$ , the analogue of Eq. (23) reads [47]

$$
u _ { T } ( q ) = \left. \frac { T } { 2 } \frac { 2 - H } { \tilde { \sigma } \sqrt { H ( 1 - H ) } } D ^ { - H } \right| _ { D ^ { 2 } = 2 ( L ^ { 2 } - q ) } .\tag{48}
$$

At a given level of the Parisi hierarchy, $u _ { T } ( q )$ sets the characteristic free-energy gap (Sec. 2.2),

$$
E _ { \mathrm { g a p } } ( D ) \sim \frac { T } { u _ { T } ( D ) } \sim \tilde { \sigma } D ^ { H } .\tag{49}
$$

The gap is therefore independent of the temperature, and using $D = \ell / \sqrt { N }$ recovers Eq. (47). The same gap can also be reproduced without replicas, in a generalised random energy model built in real space from a dyadic decomposition of $V \ ( \mathrm { A p p . ~ C } )$

Note that one could also balance the confinement against the depth of the deepest state within distance $\ell , E _ { \mathrm { d e p t h } } ( \ell ) \sim \sigma \ell ^ { H } \sqrt { N } = N E _ { \mathrm { g a p } } ( \ell )$ , which is enhanced by the extreme-value factor $\sqrt { 2 \log \mathcal { M } } \sim \sqrt { N }$ instead of compressed by it. This leads to what we call the depth-Larkin balance, $\kappa \ell _ { \mathrm { D } } ^ { 2 } \sim E _ { \mathrm { d e p t h } } ( \ell _ { \mathrm { D } } )$ , which gives

$$
\ell _ { \mathrm { D } } \sim \left( \frac { \sigma \sqrt { N } } { \kappa } \right) ^ { \frac { 1 } { 2 - H } } ,\tag{50}
$$

i.e. $\ell _ { \mathrm { D } } ( \kappa ) = \ell _ { \mathrm { L } } ( \kappa / N )$ . In difusion, this scale is never realised, however. Indeed, the well is centred on a noised data point, which already sits in a low-lying state of a measure supported on the sphere, so that moving away from it gains only a gap, not a depth, as the posterior freeenergy diference (62) of Sec. 5 makes explicit. At $\kappa = \tilde { \kappa } / N$ the depth balance would moreover require $\ell _ { \mathrm { D } } \gg L \sqrt { N }$ , beyond the size of the data sphere, see Fig. 3 for an illustration. The depth-Larkin scale is instead the relevant one when κ is a genuine confining potential centred on a generic point, as for a particle or a manifold pinned in a random potential in a box of linear size $L \sqrt { N } \gg \ell _ { \mathrm { D } }$ , where it reproduces the replica results [25, 27] and underlies the cusp analysis of Le Doussal et al. [28] (Sec. 4.4). As we show in Sec. 4.3, the distinction between the two balances is essential in determining the shape of the cells.

## 4.2 The speciation window: Descending the Parisi tree

Replacing the expression for the energy gap in Eq. (46), we finally obtain

$$
\ell _ { \mathrm { L } } ( \kappa ) \sim \left( \frac { \sigma } { \kappa \sqrt { N } } \right) ^ { \frac { 1 } { 2 - H } } = \sqrt { N } \left( \frac { \tilde { \sigma } } { \tilde { \kappa } } \right) ^ { \frac { 1 } { 2 - H } } ,\tag{51}
$$

i.e. $D _ { \mathrm { L } } = \ell _ { \mathrm { L } } / \sqrt { N } \sim ( \tilde { \sigma } / \tilde { \kappa } ) ^ { 1 / ( 2 - H ) }$ . For a given $\kappa ,$ pure states at distances smaller than $\ell _ { \mathrm { L } } ( \kappa )$ haven’t speciated whereas the ones beyond it have, and the ones at scale $\ell _ { \mathrm { L } } ( \kappa )$ are speciating.

The cascade then starts when $\ell _ { \mathrm { L } } = \ell _ { 0 }$ and ends when $\ell _ { \mathrm { L } } = \ell _ { 1 }$ , leading to an attribution window

$$
\tilde { \kappa } _ { s } \sim \tilde { \sigma } D _ { 0 } ^ { - ( 2 - H ) } \mathrm { ~ \lesssim ~ } \tilde { \kappa } \mathrm { ~ \lesssim ~ } \tilde { \kappa } _ { c } \sim \tilde { \sigma } D _ { 1 } ^ { - ( 2 - H ) } ,\tag{52}
$$

so that we recover the spectral speciation result $\tilde { \kappa } _ { s } \propto \tilde { \sigma }$ , as $D _ { 0 }$ is independent of the disorder strength. This result also adds a time characterising the end of the cascade, $\tilde { \kappa } _ { c } \propto \tilde { \sigma } ^ { 2 / H } \propto D _ { 1 } ^ { - 2 }$ and therefore the speciation window span is given by

$$
\frac { \tilde { \kappa } _ { c } } { \tilde { \kappa } _ { s } } = \left( \frac { D _ { 0 } } { D _ { 1 } } \right) ^ { 2 - H } \propto \tilde { \sigma } ^ { ( 2 - H ) / H } ,\tag{53}
$$

which becomes infinitely broad when $\tilde { \sigma } \to \infty$ . We also remark that the window widens dramatically as $H  0$

Beyond the first speciation, the Larkin scale provides a natural way of locating the resolution scale of the smoothed landscape within the Parisi tree. As long as $\ell _ { \mathrm { L } }$ lies in the continuous fRSB range, we define the corresponding Parisi level $u ^ { \star } ( \kappa )$ through

$$
D \big ( u ^ { \star } ( \kappa ) \big ) = \frac { \ell _ { \mathrm { L } } ( \kappa ) } { \sqrt { N } } .\tag{54}
$$

Substituting the Larkin scale in the expression for $D ( u )$ gives, at the level of scaling, $u ^ { \star } \sim$ $( \tilde { \kappa } ^ { H } / \tilde { \sigma } ^ { 2 } ) ^ { 1 / ( 2 - H ) }$ (note that u has dimension of ˜σ) or, normalising by the onset where $u ^ { \star } = u _ { 0 }$

$$
\frac { u ^ { \star } ( \tilde { \kappa } ) } { u _ { 0 } } = \left( \frac { \tilde { \kappa } } { \tilde { \kappa } _ { s } } \right) ^ { \frac { H } { 2 - H } } , \qquad \tilde { \kappa } _ { s } \leq \tilde { \kappa } \leq \tilde { \kappa } _ { c } .\tag{55}
$$

At the first speciation, $\ell _ { \mathrm { { L } } } = \ell _ { 0 }$ and therefore $u ^ { \star } = u _ { 0 }$ . Note that $u _ { 0 } \propto 1 / \tilde { \sigma }$ is genuinely small at strong disorder, so that the cursor then spans a wide range, from $u _ { 0 } \ll 1$ for ${ \tilde { \kappa } } = { \tilde { \kappa } } _ { s }$ to $u _ { 1 } = O ( 1 )$ for $\tilde { \kappa } = \tilde { \kappa } _ { c }$ . As ˜κ increases, $\ell _ { \mathrm { L } }$ decreases and $u ^ { \star }$ moves monotonically through the continuous branch towards $u _ { 1 } = ( 2 - H ) / 2$

The smoothing thus progressively resolves finer levels of the Parisi tree, with $u ^ { \star } ( \kappa )$ providing a natural parametrisation of this hierarchical cascade. Once $u ^ { \star }$ reaches $u _ { 1 }$ , the scale-matching construction has exhausted the hierarchy of the equilibrium data measure. This evolution is shown in Fig. 4(a), while the quantitative evolution of $u ^ { \star }$ and the associated $u ( q )$ are shown in Fig. $^ { 4 ( \mathrm { b , c } ) }$

![](images/c9d8a0a3cb10f9f25190c691b9759505e77e35e754377d5fb43ecaa4f194c2e7.jpg)

![](images/7f2b341762b73a50e51ddd38fa29de797da552faae29df9f91833c889389ea8a.jpg)

![](images/f77cc013c339f08e3e614e54b6974e790cd16bab520fe9c4dde85f9914ad2de3.jpg)  
Figure 4: (a) Schematic ultrametric (Parisi) tree of the pure states of $P _ { 0 }$ , going down in the Parisi level u from the first speciation at $u _ { 0 }$ to $u _ { 1 } = ( 2 - H ) / 2$ , where the leaves are the pure states. For $\tilde { \kappa } < \tilde { \kappa } _ { s }$ the smoothed landscape is a single efective cell (grey trunk). At a given $\tilde { \kappa } .$ the cursor $u ^ { \star } ( \tilde { \kappa } )$ of Eq. (55) cuts the tree at the level resolved by the Larkin scale (dashed lines, with a dot on each resolved branch), and moves towards $u _ { 1 }$ as $\tilde { \kappa }$ increases. (b) Parisi cursor $u ^ { \star }$ versus ${ \tilde { \kappa } } / { \tilde { \kappa } _ { s } }$ for $H = 0 . 2 5 , 0 . 5 , 0 . 7 5$ at $\tilde { \sigma } = 1 0 ~ ( L = 1 )$ . Open circles mark $u _ { 0 }$ at the onset and filled circles $u _ { 1 }$ , reached at $\tilde { \kappa } _ { c } ,$ beyond which the cursor saturates (dashed). The shaded region is the single cell, $\tilde { \kappa } < \tilde { \kappa } _ { s } . \mathrm { ( c ) }$ The same cursor on the equilibrium Parisi function $u ( q )$ versus $D ^ { 2 } = 2 ( L ^ { 2 } - q )$ (inverted axis), with ticks at log-spaced $\tilde { \kappa }$ moving from $D _ { 0 }$ towards $D _ { 1 }$

This cursor also has a direct interpretation. Two states drawn from $P _ { 0 }$ lie in diferent cells of the foam when their distance exceeds the cell scale, $D > D _ { \mathrm { { L } } }$ , which happens with probability u evaluated at that scale. Averaged over the disorder, $u ^ { \star } ( \tilde { \kappa } )$ is therefore the probability that two states drawn from the data are already distinguished by the smoothed landscape at $\tilde { \kappa } .$ . It grows from $u _ { 0 }$ at the onset to $u _ { 1 }$ at the end of the cascade.

## 4.3 Geometry of the foam and shape of the cells

Let us now describe more precisely the geometry of the smoothed potential in this intermediate regime. Within the rigid-state picture of Sec. 4.1, we recall that each pure state contributes to the partition function $Z ( \boldsymbol { y } )$ through its cell potential $\begin{array} { r } { F _ { \alpha } ( { \pmb y } ) = F _ { \alpha } + \frac { \kappa } { 2 } | { \pmb y } - { \pmb m } ^ { \alpha } | ^ { 2 } } \end{array}$ 2

$$
W ( { \pmb y } ) \simeq - \log \sum _ { \alpha } \mathrm { e } ^ { - F _ { \alpha } ( { \pmb y } ) } .\tag{56}
$$

For a given $^ { y , }$ wherever one of the cell potentials lies below all the others by more than a few units, the sum is dominated by a single term $\alpha = \alpha ^ { \star } ( \pmb { y } )$ , and $W ( { \pmb y } ) \simeq F _ { \alpha } ( { \pmb y } )$ is simply a harmonic well of curvature κ centred on $\mathbf { \Omega } _ { m } { } ^ { \alpha }$ . We call the region of the $\textbf {  { y } }$ sphere where this is the case the cell of the state $\alpha$ . The boundary between two neighbouring cells is then a hyperplane perpendicular to $\pmb { d } = \pmb { m } ^ { \alpha } - \pmb { m } ^ { \gamma }$ , displaced from the bisector by $( F _ { \alpha } - F _ { \gamma } ) / ( \kappa \ell )$ towards the state of higher free energy, whose cell is accordingly smaller. The cells therefore tile y-space as a weighted Voronoi, or more precisely a power (Laguerre) diagram of the state centres as illustrated in Figs. 1 and 2.

At a given $\kappa ,$ neighbouring states are at a distance $| { \pmb m } ^ { \alpha } - { \pmb m } ^ { \gamma } | = \ell = \ell _ { \mathrm { L } }$ . The wall separating them is characterised by the equality of the energy $F _ { \alpha } - \kappa \pmb { y } \cdot \pmb { m } ^ { \alpha } = F _ { \gamma } - \kappa \pmb { y } \cdot \pmb { m } ^ { \gamma }$ . Moving at a distance r from this wall creates an imbalance of these energies of order $\kappa \ell r ;$ this is of order one when $r \sim ( \kappa \ell ) ^ { - 1 }$ which defines the typical thickness of the wall.

We define the sharpness of the foam’s walls as the ratio between the size ℓ of the cells to the thickness of the walls $( \kappa \ell ) ^ { - 1 }$ :

$$
\Gamma ( \kappa ) : = \kappa \ell _ { \mathrm { L } } ^ { 2 } = \tilde { \kappa } D _ { \mathrm { L } } ^ { 2 } \propto \tilde { \kappa } ^ { - H / ( 2 - H ) } .\tag{57}
$$

As a result, walls between coarser branches, resolved earlier, are the sharpest. In fact $\Gamma ( \kappa )$ characterises the least sharp walls among all previously created shocks. Indeed, as mentioned in Sec. 4.2, $u _ { 0 } \propto 1 / \tilde { \sigma }$ is genuinely small at strong disorder, so that the initial foam has $\Gamma ( \kappa _ { s } ) \sim$ $1 / u _ { 0 } \gg 1$ . The sharpness then decreases along the cascade, down to $\Gamma ( \kappa _ { c } ) \sim 1 / u _ { 1 } = O ( 1 )$ , where the walls become as thick as the cells and the foam picture ends. By the Larkin balance (46), Γ is also the free-energy gap $E _ { \mathrm { g a p } } ( \ell _ { \mathrm { L } } )$ at the cell scale, so that the foam is sharp precisely where the hierarchy is strongly condensed, i.e. where the gaps are large and u is small. For the difusion itself $( \beta = 1 )$ , the walls are not genuine discontinuities of $\nabla W$ , their width being set by the noise, but they appear as such on the scale of the cells whenever $\Gamma \gg 1$

Note that at the centre of a wall separating two states of equal weight, the curvature of W normal to the wall is reduced to

$$
\pmb { n } ^ { \top } \nabla \nabla W \pmb { n } \leq \kappa \left( 1 - \frac { \kappa \ell ^ { 2 } } { 4 } \right) = \kappa \left( 1 - \frac { \Gamma } { 4 } \right) ,\tag{58}
$$

with n the wall-normal unit vector. The above is an equality for rigid states, the fluctuations inside the states only lowering it further. Walls thus become concave, i.e. ridges of W separating two basins, as soon as $\Gamma > 4$ . While we focus on the walls separating two states here, the ultrametricity of the potential means that a cluster in fact splits into several equidistant subclusters. When K of them have comparable free energies the walls between them meet on junctions where K cell potentials coincide, i.e. on sets of codimension $K - 1$ . These junctions generalise the walls to higher codimension, that we shall come back to in Sec. 6.

The free-energy balance also fixes the shape of the cells, which in high dimension need not be spheroidal. By Eq. (45), the free-energy diference between two cells is proportional to the normal distance r to the wall between them, $| F _ { \gamma } ( { \pmb y } ) - F _ { \alpha } ( { \pmb y } ) | = \kappa \ell r$ . A noised point thus lies within normal distance r of a wall when the smallest free-energy diference to a competitor is below κℓr. Let $\rho _ { E } ( \Delta E )$ be the probability density of the smallest free-energy diference $\Delta E$ between the state of the noised point and its competitors at scale ℓ. Assuming this density is finite and smooth as $\Delta E  0$ , the probability to lie within a small normal distance r of a wall is $\rho _ { E } ( 0 ) \kappa \ell r$ . Since the diferences are spread over a range of order $E _ { \mathrm { g a p } } ( \ell ) , \rho _ { E } ( 0 ) \sim 1 / E _ { \mathrm { g a p } } ( \ell )$ and the density of walls per unit length along the normal, i.e. the surface-to-volume ratio of the cells weighted by the noised points, is

$$
S _ { V } \sim \frac { \kappa \ell } { E _ { \mathrm { g a p } } ( \ell ) } = \frac { 1 } { \ell } \frac { \kappa \ell ^ { 2 } } { E _ { \mathrm { g a p } } ( \ell ) } .\tag{59}
$$

At the gap-Larkin scale, $\kappa \ell _ { \mathrm { L } } ^ { 2 } \sim E _ { \mathrm { g a p } } ( \ell _ { \mathrm { L } } )$ , one thus gets a surface to volume ratio $S _ { V } \sim 1 / \ell _ { \mathrm { L } }$ . This result indicates that the cells are very anisotropic. Indeed, a ball of radius ℓ in N dimensions has $S _ { V } = N / \ell$ , and more generally a factor N is the signature of a cell with extension similar in all directions. For instance, splitting a simplex into pyramids with apex at an interior point, one per facet, gives exactly a volume $\textstyle { \frac { 1 } { N } } \sum _ { k } h _ { k } A _ { k }$ , with $A _ { k }$ the area of facet k and $h _ { k }$ its normal distance to the apex, so that $S _ { V } = \bar { N / h }$ , with h<sup>¯</sup> the area-weighted mean distance to the facets. In order to get a surface to volume ratio of order $1 / \ell$ , one needs that the cells be slabs, closed by O(1) walls within reach of a noised point and open in all other directions.

The slab geometry can equivalently be seen by ranking the competitors by their free-energy diference, which grows only logarithmically with the rank for extreme-value gaps.<sup>3</sup> The number of walls within normal distance r then grows as $\mathrm { e } ^ { r / \ell _ { \mathrm { L } } }$ , so that closing all directions (which requires at least N walls) would require $r \sim ( \log N ) \ell _ { \mathrm { L } }$ , i.e. beyond the data sphere.

Here, the depth balance $\kappa \ell _ { \mathrm { D } } ^ { 2 } \sim N E _ { \mathrm { g a p } } ( \ell _ { \mathrm { D } } )$ would instead give round cells, $S _ { V } \sim N / \ell _ { \mathrm { D } }$ which is the geometry of the pinned manifolds of Ref. [28]. Recall that the gap and depth balances involve competitors spaced by the same $E _ { \mathrm { g a p } } ,$ and difer only through the penalty at the Larkin scale, which is larger by a factor N for the depth balance. The same ranking then gives $\mathrm { e } ^ { N r / \ell _ { \mathrm { D } } }$ walls within normal distance $r ,$ so that the cell is closed in every direction already at $r \sim \ell _ { \mathrm { D } } \log N / N$ , and the exponentially many walls needed for it to be closed at comparable distances in all directions are reached at $r \sim \ell _ { \mathrm { D } }$ . The cell is then nearly round, and a step of random direction leaves it after a chord $\sim \ell _ { \mathrm { D } } / \sqrt { N } = O ( 1 )$

![](images/4312b854337d50b12e3f30d8780ea24233ffac5f7d4c97a89c01bf82faf1c5b1.jpg)  
Figure 5: Wall crossing and the cusp of the score correlator. A flat wall of unit normal n separates the cells of two states whose centres $\mathbf { \Omega } _ { m } { } ^ { \alpha }$ and $m ^ { \gamma }$ lie a distance ℓ apart along n. On either side, the score $\pmb { \mathscr { s } } = - \nabla W = \kappa ( \pmb { m } - \pmb { y } )$ points towards the centre of its own state, so that across the wall W jumps by $\kappa { \ell n }$ . A step v from the noised point $\textbf {  { y } }$ crosses the wall only if $\textbf {  { y } }$ lies within the normal distance $| \boldsymbol { v } \cdot \boldsymbol { n } |$ of it, which happens with probability $S _ { V } | \boldsymbol { v } \cdot \boldsymbol { n } |$ . For a random direction, $| \hat { \pmb v } \cdot { \pmb n } | \sim 1 / \sqrt { N }$ , which gives a crossing rate $\nu \sim S _ { V } / \sqrt { N }$ per unit length, each crossing adding $\kappa ^ { 2 } \ell ^ { 2 } / N$ to $\mathcal { C } ( v )$

A final geometric information can be obtained in the Hopf–Lax limit, where the curvatures of W away from the walls follow from those of V at the minimiser through an exact map that we derive and interpret in $\mathrm { A p p . ~ D }$

## 4.4 Shocks and cusps in the score correlator

As written in Sec. 2.1, the gradient of the smoothed potential W—that is the score—can be understood as the Burgers velocity field $2 \nabla W _ { \beta }$ in the arbitrary $\beta$ generalisation. Cell (or slab) boundaries therefore have a straightforward interpretation as shocks in the $\beta \to \infty$ Hopf–Lax limit. This mapping between the replica-symmetry breaking of Gaussian random fields and Burgers shocks is at the heart of Bouchaud et al. [36] and Balents et al. [37], where the authors consider short-range disorder, notably to understand the properties of “Burgulence” [39, 40]. It is also intimately related to Functional Renormalisation Group approaches to random manifolds [28, 54, 55]. In this context a natural object to consider is the velocity correlator, where shocks translate to a cusp non-analyticity at the Larkin scale. As noted at the start of this section, however, the pinning that distinguishes the states is of order one at $\kappa = \tilde { \kappa } / N ,$ so that the tworeplica computation of this object by Le Doussal, M¨uller, and Wiese [28] (LMW), carried out at $\kappa = O ( 1 )$ , does not apply directly. As we shall see, the foam nonetheless predicts its form.

Our geometric argument for the cusp singularity of the score correlation is the following. Consider a straight step v from a point y. It crosses a wall if y lies within normal distance $| \boldsymbol { v } \cdot \boldsymbol { n } |$ of it, with a probability $S _ { V } | \pmb { v } \cdot \pmb { n } | ,$ , and for a step of random direction $\hat { \pmb { v } } = \pmb { v } / | \pmb { v } |$ , the projection along the wall normal is typically $| \hat { \pmb v } \cdot { \pmb n } | \sim 1 / \sqrt { N }$ . These two independent conditions, proximity to a wall and alignment with its normal, give a crossing rate $\nu \sim S _ { V } / \sqrt { N }$ per unit length of the step. Each crossing changes the score by the jump $\kappa { \ell n }$ . For the per-component correlator of the zero-temperature potential, whose walls are genuine discontinuities, ${ \mathcal { C } } ( v ) =$ $N ^ { - 1 } \overline { { | \nabla W _ { \infty } ( \pmb y + \pmb v ) - \nabla W _ { \infty } ( \pmb y ) | ^ { 2 } } }$ , the foam therefore predicts a linear cusp, with a norm that scales as the score jump squared,

$$
\mathcal { C } ( \pmb { v } ) \sim \frac { S _ { V } } { \sqrt { N } } \frac { \kappa ^ { 2 } \ell ^ { 2 } } { N } | \pmb { v } | ,\tag{60}
$$

which holds as long as the step crosses at most one wall.

As mentioned above, in our generative difusion context where $\kappa = \tilde { \kappa } / N$ and the original data is on a sphere, we have cells that are in fact slabs with the surface to volume ratio $S _ { V } \sim 1 / \ell _ { \mathrm { L } }$ $\mathrm { A t }$ the operative Larkin gap scale, we thus have

$$
\mathcal { C } ( \pmb { v } ) \sim \frac { \kappa ^ { 2 } \ell _ { \mathrm { L } } } { N ^ { 3 / 2 } } \vert \pmb { v } \vert = \frac { \tilde { \kappa } ^ { ( 3 - 2 H ) / ( 2 - H ) } \tilde { \sigma } ^ { 1 / ( 2 - H ) } } { N ^ { 3 } } \vert \pmb { v } \vert .\tag{61}
$$

On the other hand, if we were instead in the standard unbounded space $\kappa = O ( 1 )$ setting of random manifolds and a generic centre, we would be in the case $S _ { V } \sim N / \ell _ { \mathrm { D } }$ , such that Eq. (60) gives $\mathcal { C } ( \pmb { v } ) \sim \kappa ^ { 2 } \ell _ { \mathrm { D } } | \pmb { v } | / \sqrt { \bar { N } } = \kappa ^ { ( 3 - 2 H ) / ( 2 - H ) } \tilde { \sigma } ^ { 1 / ( 2 - H ) } | \pmb { v } |$ , with no dependence on N—the factor N of round cells and the scale $\ell _ { \mathrm { D } } \propto \sqrt { N }$ compensating the factors $1 / \sqrt { N }$ and $1 / N$ of Eq. (60). This matches exactly the scaling of the cusp in the functional renormalisation group solution of LMW in their single-shock regime,<sup>4</sup> which supports the geometric picture outlined here. The exponents in ˜κ and $\tilde { \sigma }$ of Eq. (61) are thus identical to those of this depth-Larkin setting, the amplitude being suppressed by $N ^ { - 2 }$ from $\kappa = \tilde { \kappa } / N$ and by $N ^ { - 1 }$ from the slab geometry.

Turning the scaling argument presented here into a proper replica computation is not straightforward, for the reason given at the start of this subsection and encountered throughout this work. At $\kappa = \tilde { \kappa } / N$ the pinning that distinguishes the states is of order one and does not move the extensive saddle point, so that the cusp would have to be extracted from the $O ( 1 )$ reweighting of the tree by two planted noisy points, y and $\mathbf { \boldsymbol { y } } + \mathbf { \boldsymbol { v } } .$ , as for the posterior over states of Sec. 5. We therefore leave this computation for future work.

## 5 The denoising error across scales

Generative difusion can be seen as a succession of denoising problems. It is therefore useful to reformulate the geometric picture we have uncovered in terms of the Bayes-optimal denoiser, whose error $\Delta ( \kappa )$ , defined in Eq. (15) above, decreases from the prior width $L ^ { 2 } - \bar { q }$ to zero along the reverse process. Indeed, in the context of a trained model, the denoising error is essentially the loss that a deep network is trained against. We first study the behaviour of this denoising error across the speciation window described above, before finally computing it for $\kappa = O ( 1 )$ i.e. when one denoises within the selected state.

## 5.1 Speciation regime: the foam seen by the denoiser

In an inference context, the Larkin balance of Sec. 4.1 acquires a direct reinterpretation. Consider a noised point $\pmb { y } = \pmb { m } ^ { \alpha } + \pmb { z } / \sqrt { \kappa }$ generated from or “planted $\mathrm { i n } ^ { \dag }$ a pure state $\alpha .$ , treated as rigid. The denoiser aims at finding the state of the signal, $\alpha ,$ and weighs each state $\gamma$ by its posterior probability $\pi _ { \gamma } ( \pmb { y } , \kappa )$ . Recalling the pure state decomposition of the measure with $w _ { \alpha } \propto \mathrm { e } ^ { - F _ { \alpha } }$ , the posterior free-energy diference between a competitor and the planted state, i.e. their log-odds ratio, is (App. E.1)

$$
\log \frac { \pi _ { \gamma } } { \pi _ { \alpha } } = - ( F _ { \gamma } - F _ { \alpha } ) - \frac { \tilde { \kappa } D _ { \alpha \gamma } ^ { 2 } } { 2 } + \sqrt { \tilde { \kappa } } D _ { \alpha \gamma } \xi ,\tag{62}
$$

where $D _ { \alpha \gamma } ^ { 2 } = 2 ( q _ { 1 } - Q _ { \alpha \gamma } )$ is the squared distance between their centres and $\xi \sim \mathcal { N } ( 0 , 1 )$ . This is the Larkin balance (46) seen from the planted state, with three terms of diferent origins. The gap $F _ { \gamma } - F _ { \alpha }$ is the prior free-energy diference between the two states. The penalty $\tilde { \kappa } D _ { \alpha \gamma } ^ { 2 } / 2$ is the cost of moving away from state $\alpha .$ The tilt $\sqrt { \tilde { \kappa } } D _ { \alpha \gamma } \xi ,$ , of zero mean and independent of the disorder, is due to the displacement of $\textbf {  { y } }$ with respect to $\mathbf { \Omega } _ { m } { } ^ { \alpha }$ and is small compared with the penalty for $\tilde { \kappa } D _ { \alpha \gamma } ^ { 2 } \gg 1$ . For two branches of equal weight, $F _ { \gamma } = F _ { \alpha }$ , Eq. (62) reduces to the speciation criterion of Achilli et al. [7], the penalty being compared with its fluctuations.

In this reading, a wall of the foam is where the denoiser attributes equal probability to two states. Indeed, if the planted state α has lower free energy, then $\pi _ { \alpha } > \pi _ { \gamma }$ at all times (i.e. all $\tilde { \kappa } )$ . If on the contrary $F _ { \alpha } > F _ { \gamma } :$ , so that the planted configuration is in the “minority branch” of the split, the denoiser finds the right α provided that $\tilde { \kappa } > 2 ( F _ { \alpha } - F _ { \gamma } ) / D _ { \alpha \gamma } ^ { 2 }$ Therefore the attribution of the right label by the denoiser follows the Larkin cascade.

Within the rigid state approximation relevant for the speciation regime, the denoising error can be split as

$$
\Delta \simeq \Delta _ { 1 } + \Delta _ { c } ( \tilde { \kappa } ) ,\tag{63}
$$

where $\Delta _ { 1 }$ corresponds to the error due to intra-state fluctuations of the draw, while

$$
\Delta _ { c } = \frac { 1 } { N } \mathbb { E } \left| m ^ { \alpha } - \sum _ { \gamma } \pi _ { \gamma } \pmb { m } ^ { \gamma } \right| ^ { 2 } = q _ { 1 } - \mathbb { E } \sum _ { \gamma } \pi _ { \gamma } Q _ { \alpha \gamma } ,\tag{64}
$$

where $\mathbb { E }$ denotes the average over the random potential, over the state α drawn with its weight $w _ { \alpha }$ and over the noise z, i.e. over the draw of $\mathbf { \mathscr { y } } .$ The second form follows from expanding the square, using $| { m ^ { \gamma } } | ^ { 2 } / N = q _ { 1 }$ for all states, and exchanging α with a posterior sample i.e. enforcing the Nishimori identity of Sec. 2.1. As we increase ˜κ starting from $0 , \Delta _ { c }$ decreases from $\Delta _ { c } ( 0 ) = q _ { 1 } - \bar { q } = \Delta _ { m }$ , the interstate width of Eq. (36), to $\Delta _ { c } = 0$ at the end of the attribution.

We can go further and describe the drop in $\Delta _ { c }$ by assuming that the splits met along the branch of the true state in the tree (Fig. 4(a)) are independent. At a level $q$ of the tree, the log-odds given by Eq. (62) shift the free-energy diference $f = F _ { \gamma } - F _ { \alpha }$ between a competing branch and the branch of the planted state by $\delta f = c - \sqrt { 2 c } \xi$ in favour of the latter, with $c = \tilde { \kappa } ( q _ { 1 } - q )$ . For $u ( q ) \ll 1$ , the two branches are typically very unequal, their diference being set by the gap of Eq. (49), $f \sim E _ { \mathrm { g a p } } \sim 1 / u ( q ) \gg 1$ , and the denoiser picks the wrong branch only at the rare balanced splits. Averaging over these, the probability of a wrong guess is reduced by a factor $G ( c )$ compared with a guess drawn from the prior, which decreases from $G ( 0 ) = 1$ to $G ( c ) \propto c ^ { - 1 / 2 } \mathrm { e } ^ { - c / 4 } ~ \mathrm { ( A p p . ~ E . 2 ) }$ . A split is thus decided once $\tilde { \kappa } ( q _ { 1 } - q ) = O ( 1 )$ . Weighting each level by the probability $P ( q ) = \mathrm { d } u / \mathrm { d } q$ that a prior guess leaves the branch of the true state there, and by the cost $q _ { 1 } - q$ of such a guess, gives

$$
\Delta _ { c } ( \tilde { \kappa } ) \simeq \int \mathrm { d } q P ( q ) ( q _ { 1 } - q ) G \big ( \tilde { \kappa } ( q _ { 1 } - q ) \big ) .\tag{65}
$$

We plot the total error $\Delta$ in the regime $\kappa = \tilde { \kappa } / N$ in Fig. 6. $\mathrm { A t } ~ \tilde { \kappa } = 0 , ~ \Delta _ { c }$ reduces to $q _ { 1 } - { \bar { q } } .$ and as $\tilde { \kappa }$ grows the splits are decided one after the other, the coarsest first, consistent with our foam picture. The drop thus sets in when the coarsest split is decided, $\tilde { \kappa } ( q _ { 1 } - q _ { 0 } ) = O ( 1 )$ , i.e. at $\tilde { \kappa } \sim 1 / D _ { 0 } ^ { 2 }$ , independently of $\tilde { \sigma }$ . Inside the window, only the levels with $D ^ { 2 } \lesssim 1 / \tilde { \kappa }$ are still undecided, and they keep their prior cost, so that

$$
\Delta _ { c } ( \tilde { \kappa } ) \sim D ^ { 2 } u ( D ) \big | _ { D = 1 / \sqrt { \tilde { \kappa } } } \propto \tilde { \sigma } ^ { - 1 } \tilde { \kappa } ^ { - ( 2 - H ) / 2 } .\tag{66}
$$

As in the Larkin cascade, $\tilde { \sigma }$ and H enter only through u. This decay reaches the width of the states, $\Delta _ { c } \sim \Delta _ { 1 }$ , at $\tilde { \kappa } \sim 1 / D _ { 1 } ^ { 2 } \propto \tilde { \sigma } ^ { 2 / H }$ , the end $\tilde { \kappa } _ { c }$ of the speciation window, where $u = O ( 1 )$ Beyond it the attribution appears complete, still at $\kappa = o ( 1 )$ , which we check in Sec. 5.2.

This approximation is expected to hold when balanced splits are rare along the branch, $u \ll 1$ , i.e. at strong disorder, and to become less precise towards the end of the window, where $u  u _ { 1 } = 1 - H / 2$ . A direct Monte Carlo sampling of the posterior on random trees of states with the statistics of M´ezard and Virasoro [49] (App. E.3), which does not include such approximations, proves to follow Eq. (65) closely, with deviations that grow only towards the end of the window (Fig. 6), as expected.

![](images/5f6338f463a23d12d716dde58fbe0d91598fa6ba054c1fd132c0017ac6ea0b34.jpg)  
Figure 6: Per-coordinate denoising error $\Delta .$ averaged over the disorder, over the whole reverse schedule, on the two scales of κ joined at the plateau $\Delta = \Delta _ { 1 }$ (dotted across the cut); $H = 1 / 2$ $\tilde { \sigma } = 2 4$ . Left, $\kappa = O ( 1 / N )$ , against $\tilde { \kappa } = N \kappa \colon$ squares, Monte Carlo of the rigid-state posterior over states on a sampled Parisi tree (App. E.3); dashed, independent splits, Eq. (65). Right, $\kappa = O ( 1 )$ : planted branch, Eq. (70).

Note that, being an averaged quantity, $\Delta _ { c }$ need not follow the typical case described in Sec. 4.1, in which the planted state lies in the dominant branch of every split and is found from the outset. It is instead dominated by the rare draws in which one of these splits is balanced, which leads it to start dropping at $\tilde { \kappa } = O ( 1 )$ , independently of ${ \tilde { \sigma } } ,$ whereas the typical speciation begins later at $\tilde { \kappa } _ { s } \propto \tilde { \sigma }$ (see Fig. 6).

## 5.2 Within the selected state: a replica-symmetric posterior

Once the cascade has reached the finest level of the hierarchy, at $\tilde { \kappa } _ { c } ,$ what remains at $\kappa = O ( 1 )$ is denoising within the selected state, whose error $\Delta ( \kappa )$ decreases from the width $\Delta _ { 1 }$ of the state to 0 as $\kappa  \infty$ . The pinning term $\kappa \mathbf { \boldsymbol { y } } \cdot \mathbf { \boldsymbol { x } }$ is then extensive, and a replica computation applies. Since $\pmb { y } = \pmb { x } _ { 0 } + \pmb { z } / \sqrt { \kappa }$ is generated from the data, the relevant object is the quenched, trajectory-averaged log-partition function of the internal measure from a planted configuration,

$$
\mathcal { F } ( \kappa ) = \overline { { \frac { 1 } { \mathcal { Z } } \int \mathrm { d } \boldsymbol { x } _ { 0 } \mathrm { e } ^ { - V ( \boldsymbol { x } _ { 0 } ) } \mathbb { E } _ { z } \log Z ( \boldsymbol { y } , \kappa ) } } , \qquad \boldsymbol { y } = \boldsymbol { x } _ { 0 } + \frac { z } { \sqrt { \kappa } } ,\tag{67}
$$

with $\begin{array} { r } { \mathcal { Z } = \int \mathrm { d } \pmb { x } \mathrm { e } ^ { - V ( \pmb { x } ) } } \end{array}$ . The noise couples the planted configuration $\scriptstyle { \mathbf { \mathscr { x } } } _ { 0 }$ and the replicas of the internal measure symmetrically, $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ being itself a posterior sample. This is the replica-level expression of the Nishimori identity, which makes the computation tractable (App. F.2). We look for a saddle point in which the replicas of $P _ { 0 }$ keep their equilibrium fRSB structure, while those of the internal measure follow the branch of $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ with a single mutual overlap $q _ { \mathrm { p o s t } } \geq q _ { 1 }$ i.e. a replica-symmetric posterior of width $\Delta = L ^ { 2 } - q _ { \mathrm { p o s t } } \leq \Delta _ { 1 }$ . The lower plateau and the whole continuum of the Parisi solution then drop out, and the plant enters only through $\Delta _ { 1 }$ (App. F.3). With the within-state “stifness”

$$
\mu ( \Delta ) : = \frac { 1 } { \Delta } + 2 B ^ { \prime } ( 2 \Delta ) ,\tag{68}
$$

the single saddle-point equation reduces to

$$
\begin{array} { r } { \mu ( \Delta ) = \mu ( \Delta _ { 1 } ) + \kappa , } \end{array}\tag{69}
$$

i.e. pinning raises the within-state stifness by exactly κ. For our long-range family, with $\tilde { \Delta } = \Delta / \Delta _ { 1 }$

$$
\kappa \Delta _ { 1 } = \frac { 1 } { \tilde { \Delta } } - 1 - \frac { \tilde { \Delta } ^ { H - 1 } - 1 } { 1 - H } ,\tag{70}
$$

which for $\begin{array} { r } { H = \frac { 1 } { 2 } } \end{array}$ gives $\Delta ( \kappa ) = \Delta _ { 1 } / ( 1 + \sqrt { \kappa \Delta _ { 1 } } ) ^ { 2 }$ . The error thus depends on κ only through $\kappa \Delta _ { 1 }$ , and crosses over from $\Delta _ { 1 }$ for $\kappa \Delta _ { 1 } \ll 1$ to $\Delta \simeq 1 / \kappa$ for $\kappa \Delta _ { 1 } \gg 1$ . This error is shown on the right of Fig. 6, and for several values of H in App. F.4.

Beyond the denoising error itself, this computation provides an important consistency check of our picture. If a single state has been selected, the overlap between two samples of the posterior at fixed y takes a single value, i.e. the posterior is replica symmetric, whereas several competing states would break replica symmetry. Here, the replica-symmetric solution is stable as long as its replicon eigenvalue

$$
\lambda _ { R } ^ { \mathrm { p o s t } } = \frac { 1 } { \Delta ^ { 2 } } - 4 B ^ { \prime \prime } ( 2 \Delta ) = \frac { 1 } { \Delta ^ { 2 } } \left[ 1 - \left( \frac { \Delta } { \Delta _ { 1 } } \right) ^ { H } \right] ,\tag{71}
$$

is positive, where the second form uses Eq. (24). It is therefore positive for every $0 < \Delta < \Delta _ { 1 }$ and vanishes only at the equilibrium endpoint $\Delta = \Delta _ { 1 }$ , where the pure state is marginal, confirming that the posterior is indeed replica symmetric at any fixed $\kappa > 0$ , and a single state has been selected, consistently with the speciation picture. Note that in this check, planting y is essential. Taking a random y independent of the disorder, one instead finds a fRSB internal measure up to a finite κ (App. F.1), i.e. a competition between pure states persisting far beyond the speciation regime, which would be a contradiction.

$\mathrm { A s } \ \kappa  0 ^ { + }$ , the planted branch tends to the width $\Delta _ { 1 }$ of a single pure state, not to the prior error $L ^ { 2 } - { \bar { q } } .$ while its stability margin closes, $\lambda _ { R } ^ { \mathrm { p o s t } }  0$ . The limits $N  \infty$ and $\kappa  0$ thus do not commute, the posterior recovering the whole tree only in the speciation regime $\kappa = \tilde { \kappa } / N$ that is invisible to this replica computation and required the Larkin arguments developed above.

## 6 Beyond the exact score: imperfect reverse dynamics

The focus of this work has been the exact score, which one could hope to learn given a perfectly adapted architecture and enough data. In practice however, for any finite number of samples, the learned score is only an approximation of it. In the context of long-range correlated data, for which we have shown that the reverse process gradually uncovers a generative foam, it appears relevant to ask how errors in the learned score interact with this spatial structure. In this section we take a first step in this direction by asking where a generic, noisy error of the score feeds probability in the exact dynamics, and leave the study of trained denoisers on structured data to a subsequent work.

Recall that in terms of the smoothed score, $\pmb { s } = - \nabla W$ , the exact reverse dynamics (7) reads $\partial _ { \tau } P = \nabla \cdot [ 2 \nabla W P ] + \nabla ^ { 2 } P$ . To avoid precisely parametrising the learned score, we propose to assume that its error is centred, isotropic and decorrelates quickly along trajectories. On average over the errors, it then simply adds an efective white noise to the drift, i.e. an extra difusivity ϵ,

$$
\partial _ { \tau } P _ { \tau } ^ { \epsilon } = \nabla \cdot \left[ 2 \nabla W P _ { \tau } ^ { \epsilon } \right] + ( 1 + \epsilon ) \nabla ^ { 2 } P _ { \tau } ^ { \epsilon } .\tag{72}
$$

Assuming this noise is small, we expand to the first order with $P ^ { \epsilon } = P + \epsilon \delta P$ . The error part $\delta P$ then follows

$$
\partial _ { \tau } \delta P = \mathcal { L } _ { \kappa } \delta P + P \mathcal { I }\tag{73}
$$

![](images/b587b89516dc78146575bebfa5deef6fc3813a3598bdadfbf305271fff9a6251.jpg)  
Figure 7: Source /κ, Eq. (74), in the two-dimensional foam of Fig. 2 (same field, unit temperature, periodic box), at three stages of speciation, with κ increasing from (a), when only a few cells have formed, to (c), at the end of the cascade. The walls of $W$ at each time are drawn as lines. As the cells form, is negative at their cores (blue) and positive across the walls (red).

where $\mathcal { L } _ { \kappa }$ is the Fokker-Planck operator of the exact reverse dynamics $( \epsilon = 0 ) , \mathcal { L } _ { \kappa \varphi } = \nabla$ $[ 2 \nabla W \varphi ] + \nabla ^ { 2 } \varphi$ , and the amplifying “stray” source term is

$$
\mathcal { I } ( \pmb { y } , \kappa ) : = \frac { \nabla ^ { 2 } P } { P } = | \nabla W | ^ { 2 } - \nabla ^ { 2 } W .\tag{74}
$$

This source term redistributes probability without changing the total mass, $\begin{array} { r } { \int P \mathcal { I } = 0 \left( \mathrm { A p p . ~ G . 1 } \right) } \end{array}$

Equation (74) already suggests that the geometry of the foam is important, with notably the second term, $- \nabla ^ { 2 } W$ , positive where W is concave, i.e. across the walls between the cells of the foam, and negative close to the peaks of the exact probability distribution P, where W is locally minimal. Although the score error is isotropic, the probability it feeds is therefore not uniform, but set by the geometry of the foam, as probability is removed from peaks, and added along walls.

In our setting, this dependence on the geometry can be made explicit (App. G.2). In the sharp foam limit, $\Gamma = \kappa \ell _ { \mathrm { L } } ^ { 2 } \gg 1$ , the source is of order $\mathcal { I } \sim \kappa \Gamma \propto \kappa ^ { \bar { ( } 2 - 2 H ) / ( 2 - \bar { H } ) }$ on the walls between two cells of size $\ell _ { \mathrm { L } }$ , and reaches a maximum value of the same order on the junctions where $K \sim \sqrt { \Gamma }$ cells meet, i.e. on shocks of codimension K 1. Errors of the score thus feed probability not only onto the walls, but also onto regions where many states compete, and where the exact denoiser outputs a barycentre of several of them. Figure 7 shows this source in a twodimensional foam at three stages of speciation. As the cells form, $\mathcal { I }$ is negative at their cores and positive across the walls that separate them, so that the geometry of the foam directly sets where errors of the score feed probability.

Note that the source in itself only sets where excess probability is fed (compared to the exact reverse scheme). Where it ends up also depends on its transport by $\mathcal { L } _ { \kappa } ,$ and thus on the case at hand. In the illustrative two-dimensional foam of Fig. 7 for instance, it mostly broadens the occupied cells and shifts weight between states during speciation, a reweighting that is frozen once the walls are sharp, rather than accumulating on the walls. The late stage of the dynamics matters as well. While the states are unresolved, $\kappa \Delta _ { 1 } \ll 1$ , the exact drift $- 2 \nabla W \simeq - 2 \kappa ( \pmb { y } - \pmb { m } ^ { \alpha } )$ restores each cell towards its centre at a rate 2κ, and pulls back the excess fed into it. It is likely that a trained network does not reproduce such a strongly restoring late-time drift, so that in practice the probability fed during speciation may persist much more than in these exact dynamics that are only illustrative.

This is therefore where the foam may matter for imperfect generation. On the walls and junctions, the exact denoiser outputs barycentres of the competing states, $\begin{array} { r } { \hat { \pmb x } = \sum _ { \gamma } \pi _ { \gamma } \pmb { m } ^ { \gamma } } \end{array}$ , and in the foam these involve $K = O ( 1 )$ states because the tree is condensed and few states are therefore balanced at any given noised point. The extent to which such interpolations are good samples, i.e. whether the potential at a barycentre is predicted by its neighbours as in so-called “Kriging” [56, 57], and whether the mass fed onto the walls survives the subsequent dynamics of trained models, will be the subject of a forthcoming dedicated paper.

## 7 Discussion

In this paper, we have studied the exact process of generative difusion for Gaussian random fields with long-range correlations in high dimension. Such random fields lead to a hierarchical organisation of samples in space which appears to be plausibly present in real-world data such as images. What we have shown is that these correlations and hierarchy lead to a speciation cascade on the typical speciation timescale first identified in Biroli, Bonnaire, De Bortoli, and M´ezard [6]. There, the smooth potential associated to generative difusion develops as a $f o a m ,$ with cells associated to the pure states of the Gibbs measure separated by shock-like boundaries, the geometry of which we have described. At shorter times, the details of the potential within the pure states of the measure are finally resolved. The denoising error of the exact score follows these stages, its averaged drop during speciation being set by the Parisi function alone, while we showed that the walls of the foam are where generic errors of an imperfect score feed excess probability during the reverse process.

As highlighted by the combination of diferent methods that we resorted in, no single theory adequately describes the whole reverse process. At $\kappa = O ( 1 ) ( t = O ( 1 ) )$ the pinning term $\kappa \mathbfit { y } \cdot \mathbfit { x }$ is extensive, and the planted replica saddle point of Sec. 5.2 controls the posterior. In the speciation regime $\kappa = \tilde { \kappa } / N$ (meaning $t = O ( N ) )$ it is instead of order one, an $O ( 1 )$ perturbation of an extensive problem that a saddle point at $N  \infty$ does not see, leading us to develop alternative scaling arguments. This dificulty can be summarised in the clear importance of the order of limits for the mean posterior overlap $\bar { q } _ { \mathrm { p o s t } } ^ { ( N ) } ( \kappa ) : = N ^ { - 1 } \overline { { \mathbb { E } | \hat { \pmb x } ( \pmb y , \kappa ) | ^ { 2 } } }$ at finite $N$ , since

$$
\operatorname* { l i m } _ { \kappa \to 0 } \operatorname* { l i m } _ { N \to \infty } \bar { q } _ { \mathrm { p o s t } } ^ { ( N ) } ( \kappa ) = q _ { 1 } , \quad \mathrm { w h i l e } \quad \operatorname* { l i m } _ { N \to \infty } \operatorname* { l i m } _ { \kappa \to 0 } \bar { q } _ { \mathrm { p o s t } } ^ { ( N ) } ( \kappa ) = \bar { q } = \int _ { 0 } ^ { 1 } \mathrm { d } u q ( u ) < q _ { 1 } .\tag{75}
$$

The two limits probe opposite sides of the attribution regime $N \kappa = O ( 1 )$ . Selection among states happens at $\kappa = \tilde { \kappa } / N$ ; resolution of configurations within the selected state happens at fixed $\kappa > 0$

A second subtlety that we have encountered, as in almost every problem related to disordered systems, is the diference between typical and average behaviour. Sample by sample, the geometric and informational notions of speciation coincide. Indeed, a trajectory ending in the minority branch of a split commits to it once its planted bias $\tilde { \kappa } D ^ { 2 } / 2$ overcomes the free-energy diference f between the branches, which is also when this branch acquires a basin of its own in $P _ { t }$ . In a typical sample, f is of the order of the gap, $E _ { \mathrm { g a p } } ( D ) \sim 1 / u ( D ) \gg 1$ at the coarse levels, so that speciation occurs at the Larkin balance, from $\tilde { \kappa } _ { s } \propto \tilde { \sigma }$ on. Only a fraction $\sim u ( D )$ of the samples have a balanced split, $f = O ( 1 )$ , yet these dominate disorder averages such as the denoising error of Fig. 6, which therefore start to move at $\tilde { \kappa } \sim 1 / D _ { 0 } ^ { 2 }$ , independently of $\tilde { \sigma }$ . The spectral criterion of Sec. 3, a spinodal evaluated with the disorder-averaged covariance, describes neither exactly, but nonetheless still shares the typical scaling. The two descriptions merge at the end of the cascade, $\tilde { \kappa } _ { c } .$ where $u _ { 1 } = O ( 1 )$ and balanced splits are no longer rare.

Note that several features of this picture rely on the condensation of the Gibbs measure. With $M = \mathrm { e } ^ { { \cal O } ( N ) }$ comparable states, a trajectory commits only once its planted bias $\kappa \ell ^ { 2 } / 2$ overcomes the entropy log $M = O ( N )$ of its competitors, which pushes speciation to $\kappa = O ( 1 )$ as shown explicitly for Gaussian mixtures with exponentially many centroids [58]. This would be our situation for a truncated landscape in which exponentially many states, such as the local minima of high-dimensional random landscapes [47, 53, 59], carry comparable weights instead of being Gibbs-weighted. In such a scenario, we would expect our picture to mostly hold, albeit displaced entirely at $\kappa = O ( 1 )$ . However, for M growing subexponentially with $N$ , we expect the scalings of the present paper to hold even if such low-lying states are uniformly weighted instead of being Gibbs weighted.

The long-range correlations matter as well. For a short-range potential, whose Parisi function has a single step, the tree has a single level, and the cascade collapses into a single first-order speciation, with no window. In general, the walls of the foam are the shocks of Burgers turbulence with random initial conditions [40, 60–63], a connection we have only used at the level of scaling. We expect the structure of our arguments, namely the Larkin balance between pinning and freeenergy gaps and the parametrisation of the cascade by the Parisi level $u ^ { \star } ( \kappa )$ , to carry over to other hierarchical measures, whereas the exponents are specific to the long-range Gaussian model. Whether this is the case, in particular for non-Gaussian potentials, remains to be seen. Finally, as mentioned above, in a variance-preserving schedule the same results should hold with $\kappa = 1 / ( \mathrm { e } ^ { 2 t } - 1 )$ , albeit without the required contraction of the noise shell towards the data sphere.

Several points remain open. In particular, the rigid-state picture is not controlled near $\tilde { \kappa } _ { c } ,$ where $\Gamma = O ( 1 )$ , meaning that the walls are as thick as the cells, and the finite-N corrections to the separation of scales have not been estimated. More broadly, the Larkin balance and the foam structure (cell size, anisotropy) remain at the level of scaling arguments. A two-copy replica computation analogous to that of Le Doussal et al. [28], with two points y and $\textit { \textbf { y } } + v$ but at $\tilde { \kappa } = O ( 1 )$ , would be necessary to characterise the geometry of the foam more precisely.

Overall, our results give a concrete sense in which following an entire reverse process matters for hierarchically structured data, as the commitments to the successive levels of the hierarchy are spread over a window of noise levels that broadens with the disorder, each being decided at its own scale. Whether the trajectories of trained models feel the walls of the foam we describe, and whether they could benefit from the spatial structure by generating interpolations between states there, will be the subject of a forthcoming work on finite data.

## Acknowledgments

We want to thank Y. Fyodorov and P. Le Doussal for very insightful comments, as well as B. Achilli and E. Ventura for stimulating discussions. The work of J. G.-B. was supported by the European Union’s Horizon Europe program under the Marie Sk lodowska-Curie grant agreement No. 101210798. J. G.-B. also thanks Ecole Normale Sup´erieure’s Centre de Sciences des Donn´ees<sup>´</sup> for its hospitality during the completion of this project.

## Generative AI statement

We acknowledge the use of large language models, primarily Anthropic’s Claude Opus 4.7–5.5, which were mainly used to write numerical code and figure scripts, to check derivations and challenge scaling arguments, and to suggest edits to the manuscript. All results were checked by the authors, who take full responsibility for the content, and who wrote the final text.

## References

[1] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. pmlr, 2015.

[2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

[3] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic diferential equations. In International Conference on Learning Representations, 2021.

[4] Giulio Biroli and Marc M´ezard. Generative difusion in very large dimensions. Journal of Statistical Mechanics: Theory and Experiment, 2023(9):093402, 2023.

[5] Gabriel Raya and Luca Ambrogioni. Spontaneous symmetry breaking in generative difusion models. Advances in Neural Information Processing Systems, 36:66377–66389, 2023.

[6] Giulio Biroli, Tony Bonnaire, Valentin De Bortoli, and Marc M´ezard. Dynamical regimes of difusion models. Nature Communications, 15(1):9957, 2024.

[7] Beatrice Achilli, Marco Benedetti, Giulio Biroli, and Marc M´ezard. Theory of speciation transitions in difusion models with general class structure. Journal of Statistical Mechanics: Theory and Experiment, 2026(4):043304, 2026.

[8] Beatrice Achilli, Enrico Ventura, Gianluigi Silvestri, Bao Pham, Gabriel Raya, Dmitry Krotov, Carlo Lucibello, and Luca Ambrogioni. Losing dimensions: Geometric memorization in generative difusion. arXiv preprint arXiv:2410.08727, 2024.

[9] Enrico Ventura, Beatrice Achilli, Gianluigi Silvestri, Carlo Lucibello, and Luca Ambrogioni. Manifolds, random matrices and spectral gaps: The geometric phases of generative difusion. In International Conference on Learning Representations, 2025.

[10] Beatrice Achilli, Luca Ambrogioni, Carlo Lucibello, Marc M´ezard, and Enrico Ventura. Memorization and generalization in generative difusion under the manifold hypothesis. Journal of Statistical Mechanics: Theory and Experiment, 2025(7):073401, 2025.

[11] Tony Bonnaire, Rapha¨el Urfin, Giulio Biroli, and Marc M´ezard. Why difusion models don’t memorize: The role of implicit dynamical regularization in training. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[12] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 1530–1538. PMLR, 2015.

[13] Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon. Perception prioritized training of difusion models. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11462–11471. IEEE, 2022.

[14] Florentin Guth, Simon Coste, Valentin De Bortoli, and Stephane Mallat. Wavelet scorebased generative modeling. Advances in neural information processing systems, 35:478–491, 2022.

[15] Marc M´ezard. Spin glass theory and its new challenge: structured disorder. Indian Journal of Physics, 98(11):3757–3768, 2024.

[16] Francesco Cagnetta, Leonardo Petrini, Umberto M Tomasini, Alessandro Favero, and Matthieu Wyart. How deep neural networks learn compositional data: The random hierarchy model. Physical Review X, 14(3):031001, 2024.

[17] Antonio Sclocchi, Alessandro Favero, and Matthieu Wyart. A phase transition in difusion models reveals the hierarchical nature of data. Proceedings of the National Academy of Sciences, 122(1):e2408799121, 2025.

[18] Antonio Sclocchi, Alessandro Favero, Noam Itzhak Levi, and Matthieu Wyart. Probing the latent hierarchical structure of data via difusion models. In The Thirteenth International Conference on Learning Representations, 2025.

[19] Alessandro Favero, Antonio Sclocchi, Francesco Cagnetta, Pascal Frossard, and Matthieu Wyart. How compositional generalization and creativity improve as difusion models are trained. In International Conference on Machine Learning, pages 16286–16306. PMLR, 2025.

[20] J´erˆome Garnier-Brun, Luca Biggio, Davide Beltrame, Marc M´ezard, and Luca Saglietti. Biased generalization in difusion models. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, Seoul, South Korea, 2026. PMLR.

[21] J´er´emie Klinger, Rapha¨el Urfin, Giulio Biroli, and Marylou Gabri´e. Weighting schedules govern what and when score-based generative models learn from multimodal data. arXiv preprint arXiv:2609.35322, 2026.

[22] Marvin Li and Sitan Chen. Critical windows: non-asymptotic theory for feature emergence in difusion models. arXiv preprint arXiv:2403.01633, 2024.

[23] Marc M´ezard and Giorgio Parisi. Interfaces in a random medium and replica symmetry breaking. Journal of Physics A: Mathematical and General, 23(23):L1229, 1990.

[24] Marc M´ezard and Giorgio Parisi. Replica field theory for random manifolds. Journal de Physique I, 1(6):809–836, 1991.

[25] Marc M´ezard and Giorgio Parisi. Manifolds in random media: two extreme cases. Journal de Physique I, 2(12):2231–2242, 1992.

[26] M. M´ezard, G. Parisi, N. Sourlas, G. Toulouse, and M. Virasoro. Nature of the spin-glass phase. Phys. Rev. Lett., 52:1156–1159, Mar 1984.

[27] A. Engel. Replica symmetry breaking in zero dimension. Nuclear Physics B, 410(3):617–646, 1993. ISSN 0550-3213.

[28] Pierre Le Doussal, Markus M¨uller, and Kay J¨org Wiese. Cusps and shocks in the renormalized potential of glassy random manifolds: How functional renormalization group and replica symmetry breaking fit together. Physical Review B, 77(6):064203, 2008.

[29] Ahmed El Alaoui, Andrea Montanari, and Mark Sellke. Sampling from mean-field Gibbs measures via difusion processes. arXiv preprint arXiv:2310.08912, 2023.

[30] Andrea Montanari. Sampling, difusions, and stochastic localization. arXiv preprint arXiv:2305.10690, 2023.

[31] Brice Huang, Andrea Montanari, and Huy Tuan Pham. Sampling from spherical spin glasses in total variation via algorithmic stochastic localization. arXiv preprint arXiv:2404.15651, 2024.

[32] Davide Ghio, Yatin Dandi, Florent Krzakala, and Lenka Zdeborov´a. Sampling with flows, difusion, and autoregressive neural networks from a spin-glass perspective. Proceedings of the National Academy of Sciences, 121(27):e2311810121, 2024.

[33] Davide Straziota, Elizaveta Demyanenko, Carlo Baldassi, and Carlo Lucibello. Generative difusion for perceptron problems: Statistical physics analysis and eficient algorithms. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[34] Anatoly I Larkin. Efect of inhomogeneties on the structure of the mixed state of superconductors. Soviet Journal of Experimental and Theoretical Physics, 31:784, 1970.

[35] AI Larkin and Yu N Ovchinnikov. Pinning in type II superconductors. Journal of Low Temperature Physics, 34(3):409–428, 1979.

[36] Jean-Philippe Bouchaud, Marc M´ezard, and Giorgio Parisi. Scaling and intermittency in Burgers turbulence. Physical Review E, 52(4):3656, 1995.

[37] Leon Balents, Jean-Philippe Bouchaud, and Marc M´ezard. The large scale energy landscape of randomly pinned objects. Journal de Physique I, 6(8):1007–1020, 1996.

[38] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661–1674, 2011.

[39] Uriel Frisch and J´er´emie Bec. Burgulence. In New trends in turbulence; Turbulence: nouveaux aspects: 31 July–1 September 2000, pages 341–383. Springer, 2002.

[40] J´er´emie Bec and Konstantin Khanin. Burgers turbulence. Physics reports, 447(1-2):1–66, 2007.

[41] Roland Bauerschmidt, Thierry Bodineau, and Benoit Dagallier. Stochastic dynamics and the Polchinski equation: an introduction. Probability Surveys, 21:200–290, 2024.

[42] Krisanu Sarkar. Score shocks: The Burgers equation structure of difusion generative models. arXiv preprint arXiv:2604.07404, 2026.

[43] Yan V Fyodorov and Jean-Philippe Bouchaud. Freezing and extreme-value statistics in a random energy model with logarithmically correlated potential. Journal of Physics A: Mathematical and Theoretical, 41(37):372001, 2008.

[44] Yan V Fyodorov and Jean-Philippe Bouchaud. Statistical mechanics of a single particle in a multiscale random potential: Parisi landscapes in finite-dimensional Euclidean spaces. Journal of Physics A: Mathematical and Theoretical, 41(32):324009, 2008.

[45] Kurt Broderix and Reiner Kree. Thermal equilibrium with the Wiener potential: testing the replica variational approximation. EPL (Europhysics Letters), 32(4):343–348, 1995.

[46] Alain Comtet, C´ecile Monthus, and Marc Yor. Exponential functionals of Brownian motion and disordered systems. Journal of applied probability, 35(2):255–271, 1998.

[47] Yan V Fyodorov and H-J Sommers. Classical particle in a box with random potential: exploiting rotational symmetry of replicated Hamiltonian. Nuclear Physics B, 764(3):128– 167, 2007.

[48] Giorgio Parisi. Order parameter for spin-glasses. Physical Review Letters, 50(24):1946, 1983.

[49] Marc M´ezard and Miguel Angel Virasoro. The microstructure of ultrametricity. Journal de Physique, 46(8):1293–1307, 1985.

[50] Bernard Derrida. Random-energy model: An exactly solvable model of disordered systems. Physical Review B, 24(5):2613, 1981.

[51] M M´ezard, G Parisi, and MA Virasoro. Random free energies in spin glasses. Journal de Physique Lettres, 46(6):217–222, 1985.

[52] Bernard Derrida and G´erard Toulouse. Sample to sample fluctuations in the random energy model. Journal de Physique Lettres, 46(6):223–228, 1985.

[53] Alan J Bray and David S Dean. Statistics of critical points of Gaussian fields on largedimensional spaces. Physical review letters, 98(15):150201, 2007.

[54] Daniel S Fisher. Interface fluctuations in disordered systems: 5- ε expansion and failure of dimensional reduction. Physical review letters, 56(18):1964, 1986.

[55] Leon Balents and Daniel S Fisher. Large-n expansion of (4-ε)-dimensional oriented manifolds in random media. Physical Review B, 48(9):5949, 1993.

[56] Daniel G Krige. A statistical approach to some basic mine valuation problems on the Witwatersrand. Journal of the Southern African Institute of Mining and Metallurgy, 52(6): 119–139, 1951.

[57] Georges Matheron. Principles of geosatistics. Econ. Geol., 58:1246–1266, 1963.

[58] Enrico Ventura, Beatrice Achilli, Luca Ambrogioni, and Carlo Lucibello. Emergence of distortions in high-dimensional guided difusion models. arXiv preprint arXiv:2602.00716, 2026.

[59] Valentina Ros and Yan V Fyodorov. The high-dimensional landscape paradigm: Spinglasses, and beyond. In Spin Glass Theory and Far Beyond: Replica Symmetry Breaking After 40 Years, pages 95–114. World Scientific, 2023.

[60] Ya G Sinai. Statistics of shocks in solutions of inviscid Burgers equation. Communications in Mathematical Physics, 148(3):601–621, 1992.

[61] Shigeo Kida. Asymptotic properties of Burgers turbulence. Journal of Fluid mechanics, 93 (2):337–377, 1979.

[62] L Frachebourg and Ph A Martin. Exact statistical properties of the Burgers equation. Journal of Fluid Mechanics, 417:323–349, 2000.

[63] Jean Bertoin. The inviscid Burgers equation with Brownian initial velocity. Communications in mathematical physics, 193(2):397–406, 1998.

[64] Jairo RL de Almeida and David J Thouless. Stability of the Sherrington-Kirkpatrick solution of a spin glass model. Journal of Physics A: Mathematical and General, 11(5):983–990, 1978.

[65] Bernard Derrida and E Gardner. Solution of the generalised random energy model. Journal of Physics C: Solid State Physics, 19(13):2253–2274, 1986.

[66] David Ruelle. A mathematical reformulation of Derrida’s REM and GREM. Communications in Mathematical Physics, 108(2):225–239, 1987.

[67] Erwin Bolthausen and Alain-Sol Sznitman. On Ruelle’s probability cascades and an abstract cavity method. Communications in Mathematical Physics, 197(2):247–276, 1998.

[68] Marc M´ezard, Giorgio Parisi, and Miguel Angel Virasoro. Spin Glass Theory and Beyond, volume 9 of World Scientific Lecture Notes in Physics. World Scientific, Singapore, 1987.

[69] Dmitry Panchenko. The Sherrington–Kirkpatrick Model. Springer Monographs in Mathematics. Springer, New York, 2013.

[70] Bernard Derrida and Henrik Flyvbjerg. Statistical properties of randomly broken objects and of multivalley structures in disordered systems. Journal of Physics A: Mathematical and General, 20(15):5273–5288, 1987.

[71] Jean-Philippe Bouchaud and Marc M´ezard. Universality classes for extreme-value statistics. Journal of Physics A: Mathematical and General, 30(23):7997, 1997.

[72] Uriel Frisch, Pierre-Louis Sulem, and Mark Nelkin. A simple dynamical model of intermittent fully developed turbulence. Journal of Fluid Mechanics, 87(4):719–736, 1978.

[73] Roberto Benzi, Giovanni Paladin, Giorgio Parisi, and Angelo Vulpiani. On the multifractal nature of fully developed turbulence and chaotic systems. Journal of Physics A: Mathematical and General, 17(18):3521–3531, 1984.

[74] Jim Pitman and Marc Yor. The two-parameter Poisson–Dirichlet distribution derived from a stable subordinator. The Annals of Probability, 25(2):855–900, 1997.

[75] Silvio Franz and Giorgio Parisi. Recipes for metastable states in spin glasses. Journal de Physique I, 5(11):1401–1415, 1995.

[76] Dongning Guo, Shlomo Shamai, and Sergio Verd´u. Mutual information and minimum mean-square error in Gaussian channels. IEEE transactions on information theory, 51(4): 1261–1282, 2005.

[77] Miquel Payar´o and Daniel P. Palomar. Hessian and concavity of mutual information, diferential entropy, and entropy power in linear vector Gaussian channels. IEEE Transactions on Information Theory, 55(8):3613–3628, 2009.

## A Replica theory of the data distribution

We recall here the known results for the Gibbs measure $P _ { 0 } \propto \mathrm { e } ^ { - V }$ , which guide our geometric understanding of the generative difusion. We follow the replica analysis of M´ezard & Parisi [23, 24], see also [27], which was later refined by Fyodorov and Sommers [47]. We use the notations introduced in Sec. 2.2.

## A.1 Replicated partition function and overlap action

Consider the Gibbs measure of the unsmoothed random potential on $\Omega = S ^ { N - 1 } ( L \sqrt { N } )$ , with partition function

$$
\mathcal { Z } _ { V } = \int _ { \Omega } \mathrm { d } \pmb { x } \mathrm { e } ^ { - V ( \pmb { x } ) } .\tag{76}
$$

The replicated partition function can be directly averaged over the disorder due to the Gaussian nature of the randomness,

$$
\overline { { \mathcal { Z } _ { V } ^ { n } } } = \int _ { \Omega } \prod _ { a = 1 } ^ { n } \mathrm { d } \pmb { x } _ { a } \exp \left( \frac { 1 } { 2 } \sum _ { a , b = 1 } ^ { n } B ( | \pmb { x } _ { a } - \pmb { x } _ { b } | ) \right) .\tag{77}
$$

The integrand depends on the replicas only through their overlaps $Q _ { a b } ,$ with $Q _ { a a } = L ^ { 2 }$ , via the rescaled correlation $B ( z ) = B ( 0 ) / N - \tilde { \sigma } ^ { 2 } z ^ { H }$ of Sec. 2.2, with derivatives $B ^ { \prime } ( z ) = - \tilde { \sigma } ^ { 2 } H z ^ { H - 1 }$ 2 $\mathcal { B } ^ { \prime \prime } ( z ) = \tilde { \sigma } ^ { 2 } H ( 1 - H ) z ^ { H - 2 } , \mathcal { B } ^ { \prime \prime \prime } ( z ) = - \tilde { \sigma } ^ { 2 } H ( 1 - H ) ( 2 - H ) z ^ { H - 3 }$ , and where we have used the canonical scaling $\sigma ^ { 2 } = \tilde { \sigma } ^ { 2 } N ^ { 1 - H }$ , Eq. (20).

Taking the direct route detailed in Fyodorov and Sommers [47]—that is, avoiding the historical Gaussian ansatz and instead taking advantage of the rotationally invariant nature of the integrand—the replicated partition function can be computed purely in terms of the overlap matrix $Q .$ Up to factors subexponential in $N$

$$
\overline { { \mathcal { Z } _ { V } ^ { n } } } \asymp \int _ { \mathcal { Q } _ { n } } \mathrm { d } Q \exp \left[ N \mathcal { S } _ { n } ( Q ) \right] ,\tag{78}
$$

where $\mathcal { Q } _ { n }$ is the set of positive definite matrices with $Q _ { a a } = L ^ { 2 }$ , and

$$
\mathcal { S } _ { n } ( Q ) = \frac { 1 } { 2 } \log \operatorname* { d e t } Q + \frac { 1 } { 2 } \sum _ { a , b = 1 } ^ { n } \mathcal { B } \big ( 2 ( L ^ { 2 } - Q _ { a b } ) \big ) .\tag{79}
$$

The first term is the configurational entropy associated with realising a given overlap matrix (the volume of n-tuples of vectors with Gram matrix NQ scales as $( \operatorname * { d e t } Q ) ^ { ( N - n - 1 ) / \bar { 2 } } )$ , while the second results from the Gaussian disorder average. The large-N problem is consequently reduced to a saddle point over Q. Varying an of-diagonal element gives

$$
( Q ^ { - 1 } ) _ { a b } = 2 B ^ { \prime } ( D _ { a b } ^ { 2 } ) , \qquad a \neq b .\tag{80}
$$

This equation is the starting point for both the replica-symmetric and Parisi solutions.

## A.2 Replica-symmetric solution and AT line

For the replica-symmetric (RS) ansatz $Q _ { a \neq b } = q .$ , let $\Delta = L ^ { 2 } - q$ . In the replica limit, $( Q ^ { - 1 } ) _ { a \neq b } =$ $- q / \Delta ^ { 2 }$ . Using $B ^ { \prime } ( z ) = - \tilde { \sigma } ^ { 2 } H \acute { z } ^ { H - 1 }$ in (80) gives

$$
q = 2 ^ { H } \tilde { \sigma } ^ { 2 } H \Delta ^ { H + 1 } .\tag{81}
$$

The eigenvalue controlling fluctuations that break replica symmetry is the replicon eigenvalue

$$
\lambda _ { R } = \frac { 1 } { \Delta ^ { 2 } } - 4 B ^ { \prime \prime } ( 2 \Delta ) .\tag{82}
$$

The RS solution is stable when $\lambda _ { R } \geq 0$ . At the de Almeida–Thouless (AT) transition [64], $2 ^ { H } \tilde { \sigma } _ { \mathrm { A T } } ^ { 2 } H ( 1 - H ) \Delta _ { \mathrm { A T } } ^ { H } = 1$ . Combining this condition with (81) gives $q _ { \mathrm { A T } } = \Delta _ { \mathrm { A T } } / ( 1 - H )$ = $L ^ { 2 } / ( 2 - H ) , \Delta _ { \mathrm { A T } } = \stackrel { \_ } { L } { } ^ { 2 } ( 1 - H ) / ( 2 - H )$ , and thus the AT line (22). For $\tilde { \sigma } < \tilde { \sigma } _ { \mathrm { A T } }$ the spherical Gibbs measure is replica symmetric; for $\tilde { \sigma } > \tilde { \sigma } _ { \mathrm { A T } }$ the RS solution is unstable.

## A.3 Full-RSB solution

For $\tilde { \sigma } > \tilde { \sigma } _ { \mathrm { A T } }$ the solution breaks replica symmetry continuously. Henceforth we place ourselves in this strong disorder regime, where the overlap matrix must be parametrised by a nondecreasing Parisi function $u ( q )$ . Let its continuous part extend from $q _ { 0 }$ to $q _ { 1 }$ , and define

$$
\chi ( q ) = L ^ { 2 } - q _ { 1 } + \int _ { q } ^ { q _ { 1 } } \mathrm { d } p u ( p ) = \Delta _ { 1 } + \int _ { q } ^ { q _ { 1 } } \mathrm { d } p u ( p ) ,\tag{83}
$$

which is constant, $\chi ( q ) = \chi ( q _ { 0 } )$ , for $q \leq q _ { 0 }$

Evaluating the determinant term for a Parisi matrix, lim $n {  } 0 n ^ { - 1 }$ log det $Q = \log \Delta _ { 1 } + q _ { 0 } / \chi ( q _ { 0 } ) +$ $\int _ { q _ { 0 } } ^ { q _ { 1 } } \mathrm { d } q / \chi ( q )$ , and the disorder term with the $n  0$ rule $\begin{array} { r } { n ^ { - 1 } \sum _ { a \neq b } g ( Q _ { a b } )  - \int _ { 0 } ^ { 1 } \mathrm { d } u g ( q ( u ) ) } \end{array}$ , the action per replica is, with $D ^ { 2 } ( q ) = 2 ( L ^ { 2 } - q )$

$$
S [ u ] = \frac { 1 } { 2 } \Bigg [ \log \Delta _ { 1 } + \frac { q _ { 0 } } { \chi ( q _ { 0 } ) } + \int _ { q _ { 0 } } ^ { q _ { 1 } } \frac { \mathrm { d } q } { \chi ( q ) } + \mathcal { B } ( 0 ) - \mathcal { B } ( D _ { 1 } ^ { 2 } ) - 2 \int _ { q _ { 0 } } ^ { q _ { 1 } } \mathrm { d } q u ( q ) \mathcal { B } ^ { \prime } \big ( D ^ { 2 } ( q ) \big ) \Bigg ] .\tag{84}
$$

The physical solution is the stable extremum over nondecreasing $0 \leq u ( q ) \leq 1$

Stationarity with respect to $u ( q ) , q _ { 0 } < q < q _ { 1 }$ , using $\delta \chi ( p ) / \delta u ( q ) = \theta ( q - p )$ , gives

$$
- \frac { q _ { 0 } } { \chi ( q _ { 0 } ) ^ { 2 } } - \int _ { q _ { 0 } } ^ { q } \frac { \mathrm { d } p } { \chi ( p ) ^ { 2 } } = 2 \mathcal { B } ^ { \prime } \bigl ( D ^ { 2 } ( q ) \bigr ) ,\tag{85}
$$

the Parisi-matrix form of $\mathrm { E q . ~ } ( 8 0 )$ (the left-hand side is the Parisi function of $Q ^ { - 1 } )$ . Diferentiating once in $q$ gives $\chi ( q ) ^ { - 2 } = 4 B ^ { \prime \prime } ( D ^ { 2 } ( q ) )$ , i.e. the vanishing of the replicon (82) along the whole continuous branch (marginal stability); evaluating (85) at $q  q _ { 0 } ^ { + }$ fixes $q _ { 0 } ;$ continuity of $\chi$ at $q _ { 1 }$ fixes $q _ { 1 }$ . These steps give

$$
\chi ( q ) = \frac { 1 } { 2 \sqrt { \mathcal { B } ^ { \prime \prime } \bigl ( 2 ( L ^ { 2 } - q ) \bigr ) } } , \qquad q _ { 0 } < q < q _ { 1 } .\tag{86}
$$

Diferentiating with respect to $q$ (with $\chi ^ { \prime } = - u )$ gives the continuous Parisi function [cf. Eq. (23)]

$$
u ( q ) = - \frac { \mathcal { B } ^ { \prime \prime \prime } \big ( 2 ( L ^ { 2 } - q ) \big ) } { 2 \big [ \mathcal { B } ^ { \prime \prime } \big ( 2 ( L ^ { 2 } - q ) \big ) \big ] ^ { 3 / 2 } } = \frac { 1 } { 2 } \frac { 2 - H } { 2 ^ { H / 2 } \tilde { \sigma } \sqrt { H ( 1 - H ) } } ( L ^ { 2 } - q ) ^ { - H / 2 } .\tag{87}
$$

The endpoints [cf. Eqs. (24), (25)] follow from

$$
\Delta _ { 1 } = L ^ { 2 } - q _ { 1 } = \frac { 1 } { 2 \sqrt { \cal B ^ { \prime \prime } \big ( 2 \big ( L ^ { 2 } - q _ { 1 } \big ) \big ) } } = \bigg [ \frac { 1 } { 2 ^ { H } \widetilde { \sigma } ^ { 2 } H \big ( 1 - H \big ) } \bigg ] ^ { 1 / H } ,\tag{88}
$$

and

$$
q _ { 0 } = - \frac { B ^ { \prime } \big ( 2 ( L ^ { 2 } - q _ { 0 } ) \big ) } { 2 B ^ { \prime \prime } \big ( 2 ( L ^ { 2 } - q _ { 0 } ) \big ) } = \frac { L ^ { 2 } } { 2 - H } .\tag{89}
$$

Inverting Eq. (87) with $u \ : = \ : u ( q )$ gives Eq. (26). The upper breakpoint is universal, $u _ { 1 } =$ $u ( q _ { 1 } ^ { - } ) = ( 2 - H ) / 2$ , while the lower breakpoint is

$$
u _ { 0 } = u ( q _ { 0 } ^ { + } ) = \frac { 2 - H } { 2 \tilde { \sigma } \sqrt { H ( 1 - H ) } } D _ { 0 } ^ { - H } \propto \frac { 1 } { \tilde { \sigma } } .\tag{90}
$$

Compactly, $u ( D ) = u _ { 1 } ( D _ { 1 } / D ) ^ { H }$ on the continuous branch, so that $u _ { 0 } = u _ { 1 } ( D _ { 1 } / D _ { 0 } ) ^ { H }$ . The branch opens exactly at the AT threshold: $D _ { 1 } = D _ { 0 }$ (equivalently $q _ { 0 } = q _ { 1 } = q _ { \mathrm { A T } } , u _ { 0 } = u _ { 1 } )$ if and only if $\tilde { \sigma } = \tilde { \sigma } _ { \mathrm { A T } }$ , Eq. (22).

Rescaling $\pmb { x } = L \pmb { x } ^ { \prime }$ maps Ω onto $S ^ { N - 1 } ( \sqrt { N } )$ and $\tilde { \sigma } ^ { 2 } D ^ { 2 H }$ onto $( \tilde { \sigma } L ^ { H } ) ^ { 2 } D ^ { \prime 2 H } , D ^ { \prime } = D / L$ , and shifts log det Q by a constant. Hence $q / L ^ { 2 } , D / L$ and the Parisi function depend on $( \tilde { \sigma } , L )$ only through $\tilde { \sigma } L ^ { H }$ [as in Eqs. $( 2 2 ) , ( 8 9 ) – ( 9 0 ) ]$ , while the difusion enters only through $\kappa L ^ { 2 }$

## A.4 Pair-distance distribution

The variable $u = u ( q ) \in ( 0 , 1 )$ has the following probabilistic meaning [48]: if two pure states $\alpha , \gamma$ are drawn independently from the Gibbs measure, then $\operatorname* { P r } [ Q _ { \alpha \gamma } \leq q ( u ) ] = u$ . With $D _ { \alpha \gamma } ^ { 2 } =$ $2 ( q _ { 1 } - Q _ { \alpha \gamma } )$ the squared distance between their centres, configurations drawn from α and $\gamma$ are at squared distance $D _ { \alpha \gamma } ^ { 2 } + D _ { 1 } ^ { 2 }$ , so that $\mathrm { P r } [ D _ { \alpha \gamma } ^ { 2 } + D _ { 1 } ^ { 2 } \ge D ^ { 2 } ( u ) ] = u$ . The continuous part of Eq. (26) then gives the density $\tilde { P } ( D ) \propto D ^ { - 1 - H }$ on $( D _ { 1 } , D _ { 0 } )$ , i.e. Eq. (27), and the two plateaux give its atoms.

## A.5 Statistics of the pure-state weights

We collect standard facts on the weights $w _ { \alpha } \propto \mathrm { e } ^ { - F _ { \alpha } }$ of the pure states in the Parisi tree of states [26, 49, 51, 65] (mathematically, the Ruelle probability cascade [66–69]), used in Secs. 3.3, 4.1 and 5.

(i) Free-energy gaps. Within a given parent cluster, the free energies $f$ of its children at level $q ,$ measured from the lowest one, form near the bottom a Poisson point process of intensity

$$
\varrho ( f ) \mathrm { d } f \propto \mathrm { e } ^ { u ( q ) f } \mathrm { d } f .\tag{91}
$$

Hence the gap between the two dominant siblings is exponentially distributed with mean $1 / u ( q )$ , and the ratio of their weights is $U ^ { 1 / u }$ with $U$ uniform on (0, 1). This is the $E _ { \mathrm { g a p } } \sim 1 / u$ of Eq. (49) (at T = 1).

(ii) Leaf weights. Marginally, the weights of the terminal states are Poisson–Dirichlet $\mathrm { P D } ( u _ { 1 } )$ [26, $5 1 , 7 0 ] , u _ { 1 } = ( 2 - H ) / 2$ , whence

$$
\overline { { \sum _ { \alpha } w _ { \alpha } ^ { p } } } = \frac { \Gamma ( p - u _ { 1 } ) } { \Gamma ( p ) \Gamma ( 1 - u _ { 1 } ) } , \qquad p > u _ { 1 } .\tag{92}
$$

In particular $\overline { { \textstyle \sum _ { \alpha } w _ { \alpha } ^ { 2 } } } = 1 - u _ { 1 } = H / 2$ , which is $O ( 1 )$ , i.e. the measure is condensed on a finite number of states.

(iii) Condensation at coarse levels. The weights of the coarsest clusters (level $q _ { 0 } )$ are $\mathrm { P D } ( u _ { 0 } )$ with $u _ { 0 } \propto 1 / \tilde { \sigma } , \mathrm { E q . ~ } ( 9 0 )$ . By (i), a typical sample has a single dominant coarse branch, the second carrying a relative weight $\sim \mathrm { e } ^ { - O ( 1 ) / u _ { 0 } }$ , and only a fraction $O ( u _ { 0 } )$ of samples has a balanced coarse split. Sample-dependent quantities sensitive to the coarse levels—the sample overlap distribution $P _ { V } ( q )$ at low $q ,$ the interstate covariance $\mathbf { C } _ { \mathrm { { i n t e r } } }$ of Sec. 3.3— are non-self-averaging, and their disorder means are dominated by these rare balanced samples. The same holds at any level q with $u _ { 0 }$ replaced by $u ( q )$

## B Spectral estimate of the speciation scale

## B.1 Typical versus annealed spectral estimate

In Sec. 3 we assumed that the top eigenvalue $c _ { \mathrm { m a x } } ( \mathbf { C } _ { \mathrm { i n t e r } } )$ of a typical sample scales as its disorder average. This fails at strong disorder. The coarse clusters carry Poisson–Dirichlet weights $\mathrm { P D } ( u _ { 0 } )$ with $u _ { 0 } \propto 1 / \tilde { \sigma } \ \mathrm { ( A p p . \ A . 5 ) }$ , so that in a typical sample the coarse split is strongly unbalanced, the minority cluster carrying a relative weight $\sim \mathrm { e } ^ { - O ( 1 ) / u _ { 0 } }$ . Only the fraction $O ( u _ { 0 } )$ of samples with a balanced coarse split has $c _ { \mathrm { m a x } } \sim N D _ { 0 } ^ { 2 }$ , and becomes unstable at $\tilde { \kappa } \sim 1 / D _ { 0 } ^ { 2 }$ independently of $\tilde { \sigma } .$ . These rare samples carry the mean, ${ \cal O } ( u _ { 0 } ) N D _ { 0 } ^ { 2 } \propto N / \tilde { \sigma }$ . Equations (39) and (43) are therefore annealed statements, which nonetheless agree in scaling with the Larkin balance describing typical samples.

## B.2 Intrastate susceptibility and soft modes

We give the details behind the intrastate estimate of Sec. 3, Eqs. (40)–(42). With the soft edge (40), the first two moments of the covariance spectrum are

$$
\frac { 1 } { N } \mathrm { T r } { \bf C } ^ { p } = \int _ { 0 } ^ { \infty } \mathrm { d } \lambda \frac { \rho _ { K } ( \lambda ) } { \lambda ^ { p } } , \qquad p = 1 , 2 .\tag{93}
$$

The first integral converges at the lower edge for $\theta > 0 ,$ while the second diverges for $\theta < 1$ The interval $0 < \theta < 1$ therefore describes a finite thermal variance together with a divergent spin-glass susceptibility. The coeficient f controls the density of soft modes and has dimension $[ \lambda ] ^ { - 1 - \theta }$ . With the cutof (41),

$$
\frac { 1 } { N } \mathrm { T r } { \bf C } ^ { 2 } \simeq \int _ { \lambda _ { \mathrm { m i n } } } ^ { \infty } \mathrm { d } \lambda \frac { f \lambda ^ { \theta } } { \lambda ^ { 2 } } \simeq \frac { f } { 1 - \theta } \lambda _ { \mathrm { m i n } } ^ { \theta - 1 } = \frac { f ^ { 2 / ( 1 + \theta ) } } { 1 - \theta } \left( \frac { N } { 1 + \theta } \right) ^ { ( 1 - \theta ) / ( 1 + \theta ) } .\tag{94}
$$

It remains to determine the dependence of $f$ on the disorder strength. It follows from the intrastate width fixed by the fRSB solution, Eq. (88), $N ^ { - 1 } \mathrm { T r } { \bf C } = L ^ { 2 } - q _ { 1 } = \Delta _ { 1 } = D _ { 1 } ^ { 2 } / 2$ , so that the characteristic stifness scale is

$$
k _ { * } = \Delta _ { 1 } ^ { - 1 } = \frac { 2 } { D _ { 1 } ^ { 2 } } = 2 \big [ \tilde { \sigma } ^ { 2 } H ( 1 - H ) \big ] ^ { 1 / H } .\tag{95}
$$

Assuming that the stifness spectrum has a one-scale form, $\rho _ { K } ( \lambda ; \tilde { \sigma } ) = k _ { * } ^ { - 1 } \widehat { \rho } _ { H } ( \lambda / k _ { * } ) , \widehat { \rho } _ { H } ( v ) \simeq$ $f _ { H } v ^ { \theta }$ , we are left with

$$
f = f _ { H } k _ { * } ^ { - 1 - \theta } \propto \tilde { \sigma } ^ { - 2 ( 1 + \theta ) / H } ,\tag{96}
$$

where $f _ { H } = O ( 1 )$ is a dimensionless coeficient describing the shape of the rescaled edge. Substitution into Eq. (41) gives $c _ { \mathrm { m a x } } ( \mathbf { C } ^ { \alpha } ) \simeq [ N f _ { H } / ( 1 + \theta ) ] ^ { 1 / ( 1 + \theta ) } D _ { 1 } ^ { 2 } / 2 , \mathrm { i . e . ~ E q . ~ ( 4 2 ) }$

## C Low-lying gaps and a real-space GREM

This appendix provides an alternative derivation of the low-lying gap scaling

$$
E _ { \mathrm { g a p } } ( \ell ) \sim \frac { \sigma \ell ^ { H } } { \sqrt { N } } = \tilde { \sigma } \left( \frac { \ell } { \sqrt { N } } \right) ^ { H }\tag{97}
$$

that underlies the Larkin argument of $\mathrm { S e c . 4 . 1 \ [ E q . \ ( 4 7 ) ] }$ . The replica route, $E _ { \mathrm { g a p } } \sim T / u _ { T } ( q )$ , is also given in Sec. 4.1 [Eq. (49)]. Here instead we derive the gap without replicas, from a multiscale decomposition of V combined with extreme-value statistics. The decomposition defines an explicitly solvable real-space GREM, which also reproduces $\iota \propto D ^ { - H } / \tilde { \sigma } , \ell _ { 1 } \propto \sqrt { N } \tilde { \sigma } ^ { - 1 / H }$ and $\rho ( \ell ) \propto \ell ^ { - 1 - H }$ (see below).

Let $V ( { \pmb x } )$ be a Gaussian field in N dimensions with power-law increments

$$
\big ( V ( \pmb { x } ) - V ( \pmb { x } ^ { \prime } ) \big ) ^ { 2 } = 2 \sigma ^ { 2 } | \pmb { x } - \pmb { x } ^ { \prime } | ^ { 2 H } , \qquad 0 < H < 1 ,\tag{98}
$$

i.e. $B ( r ) = B ( 0 ) { - \sigma } ^ { 2 } r ^ { 2 H }$ , in the inertial range $\xi _ { \mathrm { S R } } \ll | { \pmb x } - { \pmb x } ^ { \prime } | \ll \xi _ { \mathrm { L R } }$ . We write $r _ { 0 } : = \xi _ { \mathrm { L R } } = L \sqrt { N }$ for the largest scale. The crucial property is self-similarity: over a region of size $r , V$ fluctuates by $\sigma r ^ { H }$ , and $V ( r \pmb { x } ) \overset { d } { = } r ^ { H } V ( \pmb { x } )$ . Hence the natural energy unit at scale r is $\sigma r ^ { H }$

Let ${ \boldsymbol { \mathbf { \mathit { x } } } } ^ { \star } =$ arg min $V .$ . We ask for the typical energy gap $E _ { \mathrm { g a p } } ( \ell )$ between $V ( x ^ { \star } )$ and $V ( x ^ { \star \star } )$ , where $\scriptstyle { \pmb x } ^ { \star \star }$ is a secondary low-lying minimum at Euclidean distance $\sim \ell$ from ${ \pmb x } ^ { \star }$

In N dimensions, one can place $\sim ( r _ { 0 } / \xi _ { \mathrm { S R } } ) ^ { N }$ points mutually separated by $\gtrsim \xi _ { \mathrm { S R } }$ , suggesting a REM count [50] of $\sim r _ { 0 } ^ { N }$ competing minima. This is wrong: long-range correlations mean that nearby points sit in the same basin. The correct object is not the number of cells but the number of independent scales of $V ,$ analogous to the law of the iterated logarithm. What enters the gap at scale r is an exponential number $\mathrm { e } ^ { c N }$ of candidates per scale (the siblings of a common parent), not per cell.

Write $V$ as a sum over dyadic scales:

$$
V ( { \pmb x } ) = \sum _ { k = 0 } ^ { K } V _ { k } ( { \pmb x } ) ,\tag{99}
$$

with $K = \log _ { 2 } ( r _ { 0 } / \xi _ { \mathrm { S R } } )$ , and $V _ { k }$ decorrelating on the length $r _ { k } = r _ { 0 } 2 ^ { - k }$ and having amplitude

$$
a _ { k } = \sigma r _ { k } ^ { H } = \sigma r _ { 0 } ^ { H } 2 ^ { - k H } .\tag{100}
$$

The variances $a _ { k } ^ { 2 }$ sum, for $H > 0$ , to a finite total $\sigma ^ { 2 } r _ { 0 } ^ { 2 H } / ( 1 - 2 ^ { - 2 H } )$ , reproducing $\overline { { V ^ { 2 } } }$ up to boundary corrections.

The decomposition naturally defines a tree. Partition Ω hierarchically into cells: generationk cells $( k = 0 , \ldots , K )$ have linear size $r _ { k } ;$ each generation-k cell is parent to $2 ^ { N }$ generation- $( k { + } 1 )$ children, and the $2 ^ { N }$ generation-0 cells are the children of the root Ω. This defines a $2 ^ { N } \mathrm { \mathrm { - a r y } }$ tree of depth K, whose leaves correspond to configurations resolved at the UV cutof. Attach to each generation-k cell an independent centred Gaussian increment of variance $a _ { k } ^ { 2 }$ . The value of V at leaf α is

$$
V _ { \alpha } = \sum _ { k = 0 } ^ { K } V _ { k } ^ { ( \alpha ) } ,\tag{101}
$$

with $V _ { k } ^ { ( \alpha ) }$ the generation-k increment on α’s ancestor chain. Two leaves $\alpha , \gamma$ whose last common ancestor sits in generation $k _ { \alpha \gamma } ^ { \star }$ (with $k ^ { \star } = - 1$ for the root) are at distance $\ell _ { \alpha \gamma } \sim r _ { k ^ { \star } + 1 }$ and have

$$
\overline { { ( V _ { \alpha } - V _ { \gamma } ) ^ { 2 } } } = 2 \sum _ { k > k _ { \alpha \gamma } ^ { \star } } a _ { k } ^ { 2 } = \frac { 2 \sigma ^ { 2 } r _ { k ^ { \star } + 1 } ^ { 2 H } } { 1 - 2 ^ { - 2 H } } \sim 2 \sigma ^ { 2 } \ell _ { \alpha \gamma } ^ { 2 H } ,\tag{102}
$$

reproducing the continuum structure function (98). The tree is thus a legitimate discretisation of V , equivalent to Derrida–Gardner’s Generalised Random Energy Model (GREM) [65] with real-space embedding—i.e. a real-space GREM in which leaves of the tree correspond to points of the physical domain. Ultrametricity is exact by construction, since leaves at tree-distance d have covariance depending only on $d ,$ not on their specific identities. This ultrametricity, which in the continuum field is emergent from the RSB solution, is built-in in the dyadic version.

On this tree, V is a branching random walk (BRW) with K levels and step variances $a _ { k } ^ { 2 }$ Greedy minimisation of the BRW $\mathrm { g i }$ ves

$$
V _ { \mathrm { m i n } } \approx - \sqrt { 2 N \log 2 } \sum _ { k = 0 } ^ { K } a _ { k } = - \sigma r _ { 0 } ^ { H } \sqrt { 2 N \log 2 } \sum _ { k = 0 } ^ { K } 2 ^ { - k H } .\tag{103}
$$

(This is exact for the GREM.) For $H > 0$ the sum converges and only the top $O ( 1 / H )$ scales contribute, $V _ { \mathrm { m i n } } \sim - \sigma r _ { 0 } ^ { H } \sqrt { N }$ with no log $r _ { 0 } ;$ for $H  0$ all scales contribute equally and one recovers the log-correlated freezing result $V _ { \mathrm { m i n } } \sim - \sigma \sqrt { N }$ log $r _ { 0 }$ [43].

Now, consider two low-lying minima at distance $r _ { k }$ . Their common coarse-scale contributions $V _ { j } , j < k _ { \mathrm { : } }$ , cancel from their energy diference. The competition at scale $r _ { k }$ is thus among the $2 ^ { N }$ sub-cells of scale $r _ { k }$ of a common parent cell of scale $r _ { k - 1 }$ , i.e. among

$$
\mathcal { M } = 2 ^ { N } = \mathrm { e } ^ { c N }\tag{104}
$$

siblings $( c = \log 2 )$ , whose level-k values are i.i.d. Gaussians of variance $a _ { k } ^ { 2 } .$ (For $k = 0$ the parent is the whole domain; on the sphere the $2 ^ { N }$ orthants provide a natural top-level partition into cells of size $\sim L \sqrt { N } . )$ Standard extreme-value statistics (EVT) [71] for  i.i.d. Gaussians of variance $a _ { k } ^ { 2 }$ predicts that the minimum sits at $- a _ { k } \sqrt { 2 \log \mathcal { M } }$ , while the second minimum sits at approximately the same location with spacing

$$
E _ { \mathrm { g a p } } ( r _ { k } ) \sim \frac { a _ { k } } { \sqrt { 2 \log \mathcal { M } } } = \frac { \sigma r _ { k } ^ { H } } { \sqrt { 2 N \log 2 } } .\tag{105}
$$

The relative gap between the top two is $1 / ( 2 \log { \mathcal { M } } ) = 1 / ( 2 N \log { 2 } )$ , small in $N { : }$ the two best cells are typically extremely close in energy compared to their absolute depth $\sim \sigma r _ { k } ^ { H } \sqrt { N }$

Fine-scale contributions $V _ { j } \ ( j > k )$ add, to each sibling, the (independent) minimum over its own sub-tree. These shifts have fluctuations $O ( \sigma r _ { k + 1 } ^ { H } / \sqrt { N } )$ , at the same order as the gap, meaning that they don’t change the scaling (nor the exponential intensity of the extreme-value process, see the freezing argument below). The result is Eq. (97), up to an $O ( 1 )$ prefactor. The prefactor $1 / \sqrt { 2 c }$ absorbs the H-dependent constants and the log-density c of candidates per scale.

Applying the canonical scaling $\sigma = \tilde { \sigma } N ^ { ( 1 - H ) / 2 } , r _ { 0 } = L \sqrt { N }$ with ˜σ, $L = O ( 1 ) \ [ \mathrm { E q . \ ( 2 0 ) } ]$ , the natural energy unit is

$$
\sigma r _ { 0 } ^ { H } = \tilde { \sigma } L ^ { H } \sqrt { N } ,\tag{106}
$$

so the ground-state energy is extensive, $V _ { \operatorname* { m i n } } / N = O ( \tilde { \sigma } L ^ { H } )$ . The extreme-value spacing becomes $E _ { \mathrm { g a p } } ( r _ { 0 } ) \sim \sigma r _ { 0 } ^ { H } / \sqrt { N } = \tilde { \sigma } L ^ { H } = O ( 1 )$ , i.e., the gap to a distant secondary minimum is $O ( 1 )$ independently of N and the canonical scaling sits precisely at the “generic degeneracy” scaling. The near-degeneracy probability that the gap falls within an $O ( 1 )$ window correspondingly saturates, $P ( r _ { 0 } , N ) \sim \sqrt { N } / ( \sigma r _ { 0 } ^ { H } ) = 1 / ( \tilde { \sigma } L ^ { H } ) = { \cal O } ( 1 )$ , meaning distant $O ( 1 )$ -degenerate secondary minima are generic rather than rare. This is the regime in which the problem is genuinely glassy and in which the replica analysis of Sec. 2.2 applies. Here, the probability $\propto 1 / \tilde { \sigma }$ is the counterpart of the top Parisi breakpoint $u _ { 0 } \propto 1 / \tilde { \sigma }$ (probability that two configurations lie in diferent top-level branches), see Eq. (111) below.

A single level is efectively a REM of $2 ^ { N }$ Gaussians of variance $a ^ { 2 }$ , and thus has near its minimum $a { \sqrt { 2 N \log 2 } }$ a density of levels growing as $\mathrm { e } ^ { f / T _ { c } }$ with $T _ { c } = a / \sqrt { 2 N \log { 2 } }$ . It freezes below $T _ { c } .$ , where its Boltzmann weights become Poisson–Dirichlet with parameter $u = T / T _ { c }$ . In the GREM the variance per unit entropy $a _ { k } ^ { 2 } / ( N \log 2 )$ decreases with depth, so each generation freezes separately [65], generation k at

$$
T _ { k } = \frac { a _ { k } } { \sqrt { 2 N \log 2 } } ,\tag{107}
$$

with $T _ { 0 } > T _ { 1 } > . . . ,$ meaning the coarsest level freezes first upon cooling. In the canonical scaling, $a _ { k } = \tilde { \sigma } L ^ { H } \sqrt { N } 2 ^ { - k H }$ and

$$
T _ { k } = \frac { \tilde { \sigma } } { \sqrt { 2 \log 2 } } \left( \frac { r _ { k } } { \sqrt { N } } \right) ^ { H } ,\tag{108}
$$

independent of N. In our $T = 1$ setting, generation k is frozen if $T _ { k } > 1$ , i.e.

$$
r _ { k } \ > \ \ell _ { 1 } ^ { \mathrm { G R E M } } : = \sqrt { N } ( 2 \log 2 ) ^ { 1 / ( 2 H ) } \tilde { \sigma } ^ { - 1 / H } ,\tag{109}
$$

which reproduces the scaling $\ell _ { 1 } = \sqrt { N } D _ { 1 } \propto \sqrt { N } \tilde { \sigma } ^ { - 1 / H }$ of the short-distance plateau [Eq. (24)]. The top level is frozen if $\tilde { \sigma } L ^ { H } > \sqrt { 2 \log { 2 } }$ , the GREM counterpart of the strong-disorder condition $\tilde { \sigma } > \tilde { \sigma } _ { \mathrm { A T } } \propto L ^ { - H }$ , otherwise no level is frozen and the measure is replica symmetric.

The Parisi weight of a frozen generation is $u ^ { ( k ) } = T / T _ { k }$ ; at $T = 1$ and with $D = r _ { k } / \sqrt { N }$

$$
u ( D ) = \frac { \sqrt { 2 \log 2 } } { \tilde { \sigma } } D ^ { - H } ,\tag{110}
$$

the same scaling $D ^ { - H } / \tilde { \sigma }$ as Eq. (23). In the cascade, $u ^ { ( k ) }$ is the probability that two configurations drawn from the Gibbs measure lie in diferent generation-k cells. Therefore, the last common ancestor has $\mathrm { P r } [ k _ { \alpha \gamma } ^ { \star } = k ] = u ^ { ( k + 1 ) } - u ^ { ( k ) } \propto \bar { 2 } ^ { k H }$ , and, exactly as in App. A.4, $\rho ( \ell ) \propto \ell ^ { - 1 - H }$ on $\ell _ { 1 } ^ { \mathrm { G R E M } } < \ell < L \dot { \sqrt { N } }$ . The weight of the coarsest generation,

$$
u ^ { ( 0 ) } = \frac { \sqrt { 2 \log 2 } } { \tilde { \sigma } L ^ { H } } \propto \frac { 1 } { \tilde { \sigma } } ,\tag{111}
$$

is the counterpart of the lower breakpoint $u _ { 0 } \propto 1 / ( \tilde { \sigma } D _ { 0 } ^ { H } )$ of Eq. (26), meaning that at strong disorder the measure is condensed already at the coarsest level.

We can finally make the sibling gap $E _ { \mathrm { g a p } } \sim T / u$ explicit. Within a frozen generation $k ,$ the free energies of the $2 ^ { N }$ sibling cells (each the level-k increment plus the free energy of its own sub-tree), measured from the lowest, form near the bottom a Poisson process of intensity $\propto \mathrm { e } ^ { f / T _ { k } } = \mathrm { e } ^ { u ^ { ( k ) } f / T }$ . There, the independent sub-tree shifts do not change the exponent of the intensity, because the deeper levels freeze at lower temperatures $( T _ { k + 1 } < T _ { k }$ , so the required exponential moments are finite). This is Eq. (91) with an explicit u, and the typical sibling gap is

$$
E _ { \mathrm { g a p } } ( r _ { k } ) \sim \frac { T } { u ^ { ( k ) } } = T _ { k } = \frac { \sigma r _ { k } ^ { H } } { \sqrt { 2 N \log 2 } } ,\tag{112}
$$

reproducing both the EVT estimate (105) and Eq. (47).

Note that this real-space GREM captures exponents and scales but not prefactors, and it does not reproduce the atom of weight $H / 2$ at $q _ { 1 }$ . Below $\ell _ { 1 } ^ { \mathrm { G R E M } }$ , the levels are liquid, and with a continuum of levels u reaches 1 continuously instead of saturating at $u _ { 1 } = ( 2 - H ) / 2$ Overall, this dyadic construction mirrors Sinai’s multifractal decomposition of Burgers flow with self-similar initial data [60] and is similar to the β-model of turbulence [72, 73].

## D Zero temperature curvature of the foam

In the Hopf–Lax limit, $\begin{array} { r } { W _ { \infty } ( \pmb { y } ) = \operatorname* { m i n } _ { \pmb { x } } U ( \pmb { x } ; \pmb { y } ) } \end{array}$ , the Hessian of $W _ { \infty }$ can be related directly to the Hessian of V at the minimiser. Let ${ \pmb x } ^ { \star } ( { \pmb y } )$ denote the minimiser of $U ( \pmb { x } ; \pmb { y } ) = V ( \pmb { x } ) + \kappa | \pmb { y } - \pmb { x } | ^ { 2 } / 2$ at fixed y, assumed unique; the stationarity condition reads $\nabla V ( \pmb { x } ^ { \star } ) = \kappa ( \pmb { y } - \pmb { x } ^ { \star } )$ . Diferentiating implicitly with respect to $\pmb { y } .$

$$
\big [ \nabla \nabla V ( { \pmb x } ^ { \star } ) + \kappa { \bf I } \big ] \frac { \partial { \pmb x } ^ { \star } } { \partial { \pmb y } } = \kappa { \bf I } ,\tag{113}
$$

and using $\nabla W _ { \infty } = \kappa ( \pmb { y } - \pmb { x } ^ { \star } )$ , so that $\nabla \nabla W _ { \infty } = \kappa \mathbf { I } - \kappa \partial \mathbf { \boldsymbol { x } } ^ { \star } / \partial \mathbf { \boldsymbol { y } }$ , one obtains

$$
\nabla \nabla W _ { \infty } ( \pmb { y } ) = \kappa \mathbf { I } - \kappa ^ { 2 } \big [ \nabla \nabla V ( \pmb { x } ^ { \star } ) + \kappa \mathbf { I } \big ] ^ { - 1 } ,\tag{114}
$$

which is nothing but the zero-temperature form of Eq. (14).

In the eigenbasis of $\nabla \nabla V ( \mathbf { x } ^ { \star } )$ , with eigenvalues $\omega _ { i }$ , the curvatures of $W _ { \infty }$ are given by the map

$$
\Lambda _ { i } = \frac { \kappa \omega _ { i } } { \omega _ { i } + \kappa } .\tag{115}
$$

Minima of $V$ are thus mapped onto curvatures $0 < \Lambda _ { i } < \kappa$ , so that $W _ { \infty }$ is always softer than the bare harmonic cost, and soft directions of V remain soft directions of $W _ { \infty }$ . Conversely, $W _ { \infty }$ becomes concave in a given direction only where the minimiser sits in a region of negative curvature of $V , - \kappa < \omega _ { i } < 0$ . The map diverges at $\omega _ { i } = - \kappa$ , where the Hessian of $V ( { \pmb x } ) + \kappa | { \pmb y } -$ $x | ^ { 2 } / 2$ at the minimiser acquires a zero mode. There, ${ \pmb x } ^ { \star } ( { \pmb y } )$ can no longer be followed smoothly as y moves, and generically the minimum merges with a neighbouring saddle and disappears—i.e. undergoes a saddle-node bifurcation. The walls of $W _ { \infty }$ are not located at these points, however. As y moves, the minimiser ${ \pmb x } ^ { \star } ( { \pmb y } )$ follows the lowest of the local minima of $V ( \pmb { x } ) + \kappa | \pmb { y } - \pmb { x } | ^ { 2 } / 2$ and a wall is crossed when another local minimum becomes lower. There ${ \pmb x } ^ { \star }$ jumps from one minimum to the other while both still exist, so that $W _ { \infty } .$ , the lowest value, stays continuous while its gradient jumps. The divergence of the map instead signals the disappearance of a local minimum by a saddle-node bifurcation, which only occurs once this minimum has ceased to be the lowest one, i.e. beyond the wall, on a metastable branch. The two coincide only where a wall is born, at a point where the lowest minimum itself flattens and splits into two. This is the zero-temperature counterpart of the distinction between spinodal and binodal made in Sec. 3.

At $\beta = 1$ , this picture must be slightly amended. Within a pure state, treated as Gaussian, the Hessian of V is to be replaced by the stifness ${ \bf K } ^ { \alpha } = ( { \bf C } ^ { \alpha } ) ^ { - 1 }$ of the state introduced in Sec. 3. $\mathrm { A s }$ it is positive definite, the curvatures $\Lambda _ { i } = \kappa \lambda _ { i } / ( \lambda _ { i } + \kappa )$ are positive and the map does not diverge. The many shocks generated by the roughness of V below the width $\ell _ { 1 }$ of the states lie inside them, and are washed out by the thermal fluctuations. $\mathrm { A t } ~ \beta = 1$ , the walls therefore arise only from the competition between pure states, which is precisely what the smoothed minimum (56) describes.

## E State attribution in the speciation regime

## E.1 Rigid-state posterior

Drawing $\mathbf { \boldsymbol { x } } _ { 0 } \sim P _ { 0 }$ amounts to choosing a pure state α with probability $w _ { \alpha } \propto \mathrm { e } ^ { - F _ { \alpha } }$ and sampling ${ \pmb x } _ { 0 } = { \pmb m } ^ { \alpha } + \zeta _ { \alpha }$ within it. As in Sec. 5.1, we treat the states as rigid, neglecting the intra-state fluctuations $\zeta _ { \alpha }$ , which are sub-dominant with respect to the inter-state ones (Sec. 3.3) and are only resolved at $\kappa = O ( 1 )$ (Sec. 5.2). For a noised point $\pmb { y } = \pmb { m } ^ { \alpha } + \pmb { z } / \sqrt { \kappa }$ planted in $\alpha _ { \mathrm { { i } } }$ , with ${ \boldsymbol z } \sim \mathcal { N } ( { \bf 0 } , { \bf I } _ { N } )$ , the posterior weight of a state $\gamma$ is

$$
\pi _ { \gamma } ( \pmb { y } , \kappa ) = \frac { w _ { \gamma } \langle \mathrm { e } ^ { \kappa \pmb { y } \cdot \pmb { x } } \rangle _ { \gamma } } { \sum _ { \gamma ^ { \prime } } w _ { \gamma ^ { \prime } } \langle \mathrm { e } ^ { \kappa \pmb { y } \cdot \pmb { x } } \rangle _ { \gamma ^ { \prime } } } \simeq \frac { w _ { \gamma } \mathrm { e } ^ { \tilde { \kappa } Q _ { \alpha \gamma } + \sqrt { \tilde { \kappa } } g _ { \gamma } } } { \sum _ { \gamma ^ { \prime } } w _ { \gamma ^ { \prime } } \mathrm { e } ^ { \tilde { \kappa } Q _ { \alpha \gamma ^ { \prime } } + \sqrt { \tilde { \kappa } } g _ { \gamma ^ { \prime } } } } ,\tag{116}
$$

with Gaussian fields $g _ { \gamma } = z \cdot \pmb { m } ^ { \gamma } / \sqrt { N } , \mathbb { E } g _ { \gamma } g _ { \gamma ^ { \prime } } = Q _ { \gamma \gamma ^ { \prime } }$ . The first equality is exact on the sphere, where $| { \pmb x } | ^ { 2 } = N L ^ { 2 }$ . The second replaces each state by its centre and uses $\kappa m ^ { \alpha } \cdot m ^ { \gamma } = \tilde { \kappa } Q _ { \alpha \gamma } .$ the terms $\kappa | { m } ^ { \gamma } | ^ { 2 } / 2 = \tilde { \kappa } q _ { 1 } / 2$ being the same for all states. The posterior thus depends on $N$ only through ˜κ. Taking the ratio of the weights of a competitor $\gamma$ and of the planted state $\alpha ,$ with $Q _ { \alpha \alpha } = q _ { 1 }$

$$
\log \frac { \pi _ { \gamma } } { \pi _ { \alpha } } = - ( F _ { \gamma } - F _ { \alpha } ) - \tilde { \kappa } ( q _ { 1 } - Q _ { \alpha \gamma } ) + \sqrt { \tilde { \kappa } } ( g _ { \gamma } - g _ { \alpha } ) .\tag{117}
$$

With $D _ { \alpha \gamma } ^ { 2 } = 2 ( q _ { 1 } - Q _ { \alpha \gamma } )$ , the second term is the penalty ${ \tilde { \kappa } D _ { \alpha \gamma } ^ { 2 } } / { 2 }$ , while $g _ { \gamma } - g _ { \alpha }$ is Gaussian with zero mean and variance $Q _ { \gamma \gamma } + Q _ { \alpha \alpha } - 2 Q _ { \alpha \gamma } = D _ { \alpha \gamma } ^ { 2 }$ , i.e. $g _ { \gamma } - g _ { \alpha } = D _ { \alpha \gamma } \xi$ with $\xi \sim \mathcal { N } ( 0 , 1 )$ This gives the log-odds ratio (62) of the main text.

## E.2 Denoising along independent splits

We detail here the independent-split approximation leading to Eq. (65). Consider a single split between the branch of the planted state α and a branch $\gamma$ whose centres have overlap q with it, with the two states at squared distance $D _ { \alpha \gamma } ^ { 2 } = 2 ( q _ { 1 } - q )$ . By Eq. (62), the observation shifts the free-energy diference $f = F _ { \gamma } - F _ { \alpha }$ between the two branches by $\delta f = \tilde { \kappa } D _ { \alpha \gamma } ^ { 2 } / 2 - \sqrt { \tilde { \kappa } } D _ { \alpha \gamma } \xi$ in favour of the branch which contains the planted state. Given a split with free-energy diference $f _ { i }$ , the probability of a wrong guess is the prior weight $1 / ( 1 + \mathrm { e } ^ { - f } )$ of the branch of the planted state times the posterior weight $1 / ( 1 + \mathrm { e } ^ { f + \delta f } )$ that the other branch keeps after the noisy observation. Placing ourselves among the balanced splits, f is spread uniformly, and integrating this product over f gives $\delta f / ( \mathrm { e } ^ { \delta f } - 1 )$ . Writing $c = \tilde { \kappa } D _ { \alpha \gamma } ^ { 2 } / 2 = \tilde { \kappa } ( q _ { 1 } - q )$ and averaging over the noise, the probability of a wrong guess at the level $q$ is thus reduced by a factor

$$
G ( c ) = \mathbb { E } _ { \xi } \left[ \frac { c - \sqrt { 2 c } \xi } { \mathrm { e } ^ { c - \sqrt { 2 c } \xi } - 1 } \right] ,\tag{118}
$$

compared with a guess drawn from the prior, for which $\delta f = 0$ . It decreases from $G ( 0 ) = 1$ to $G ( c ) \propto c ^ { - 1 / 2 } \mathrm { e } ^ { - c / 4 }$ at large c.

It remains to count the splits along the branch of the planted state. By definition of the Parisi function, a guess drawn from the prior has an overlap smaller than $q$ with the true state with probability $u ( q )$ , so it leaves the branch of the true state at the level q with probability $P ( q ) \mathrm { d } q .$ , with $P ( q ) = \mathrm { d } u / \mathrm { d } q$ the distribution of overlaps of Sec. 3.3, including its peak of weight u<sub>0</sub> at $q _ { 0 }$ . Such a guess costs $q _ { 1 } - q$ in the second form of Eq. (64), $\begin{array} { r } { \Delta _ { c } = \mathbb { E } \sum _ { \gamma } \pi _ { \gamma } ( q _ { 1 } - Q _ { \alpha \gamma } ) } \end{array}$ , and treating each level as an independent split gives Eq. (65).

## E.3 Monte Carlo sampling of the posterior over states

The Monte Carlo points of Fig. 6 evaluate the centre error (64) for rigid states, without the independent-split approximation leading to Eq. (65). Indeed, the posterior (116) only involves the weights of the states and their mutual overlaps, so that N never enters. Here, a sample therefore consists of a tree of states with its weights, a planted state drawn with its weight, and one realisation of the noisy Gaussian fields $g _ { \gamma }$ . We describe these steps in turn, and then the truncation that makes the tree finite.

(i) Discretised tree. As in a K-step replica-symmetry-breaking solution, the continuous Parisi function is replaced by a staircase with K levels, $u ( q ) = u ^ { ( k ) }$ for $q ^ { ( k - 1 ) } \leq q < q ^ { ( k ) }$ , with $q ^ { ( 0 ) } = q _ { 0 } , q ^ { ( K ) } = q _ { 1 } , u ^ { ( 1 ) } = u _ { 0 }$ and $u ^ { ( K ) } = u _ { 1 }$ , so that the atoms u<sub>0</sub> at $q _ { 0 }$ and 1  u<sub>1</sub> at $q _ { 1 }$ are kept. The $u ^ { ( k ) }$ are the values of u at K points log-spaced in D from $D _ { 0 }$ to $D _ { 1 }$ and the overlaps $q ^ { ( k ) } , 0 < k < K$ , sit at the geometric midpoints in D. K is chosen such that u changes little from one level to the next, $K = 1 0$ for Fig. 6. The tree then has K levels below its root, and two states whose branches separate at depth k have overlap $Q _ { \alpha \gamma } = q ^ { ( k ) }$

(ii) Weights. In the picture of Sec. 2.2, the children of a node at depth $k - 1$ are sub-clusters whose free energies are distributed like the lowest levels of a random energy model, with a density growing as $\mathrm { e } ^ { u ^ { ( k ) } F }$ , the weight of each child being its Boltzmann weight normalised within its parent. Rather than drawing these free energies, we generate the weights of the children one after the other, by breaking of random fractions of the mass of the parent. The first child receives a fraction $V _ { 1 }$ of this mass, the second a fraction $V _ { 2 }$ of what remains, and so on, with independent $V _ { j } \sim \operatorname { B e t a } ( 1 - u ^ { ( k ) } , j u ^ { ( k ) } - u ^ { ( k - 1 ) } )$ and $u ^ { ( 0 ) } = 0$ This stick-breaking construction reproduces exactly the law of the weights of the Parisi tree [67, 74], the children appearing in size-biased order, i.e. each new child is drawn among the remaining ones with a probability proportional to its weight. For $u ^ { ( k ) } \ll 1 , V _ { 1 }$ is close to one, so that a single child carries almost all the mass of its parent, which is the condensation discussed in the main text. The leaves at depth K are the pure states, and their masses are the weights $w _ { \gamma } .$ . (Note that this is diferent from the real-space GREM construction described above that goes all the way down to the UV cutof.)

(iii) Planted state. The planted state α must be drawn with probability $w _ { \alpha }$ . Equivalently, one can descend the tree from its root, choosing at each node the next branch with a probability proportional to its weight. This is exactly how the first child of the stickbreaking is chosen, so that the planted state is the leaf reached by always following the first child, its competitors at each level being the other children of the nodes along this path. Building the tree around the planted path allows us to keep this path exactly, however small its weight, while truncating its light competitors.

(iv) Fields and posterior. Each edge from depth $k - 1$ to depth k carries an independent Gaussian increment of variance $\stackrel { \cdot } { q ^ { ( k ) } } - q ^ { ( k - 1 ) }$ , and $g _ { \gamma }$ is the sum of the increments along the path of $\gamma ,$ so that $\mathbb { E } g _ { \gamma } g _ { \gamma ^ { \prime } } = Q _ { \gamma \gamma ^ { \prime } } - q _ { 0 }$ . The missing part $q _ { 0 }$ is common to all states and cancels in $\pi _ { \gamma } .$ . The posterior then follows from Eq. (116), i.e. from the log-weights log $w _ { \gamma } + \tilde { \kappa } Q _ { \alpha \gamma } + \sqrt { \tilde { \kappa } } g _ { \gamma }$ , and one draw of the fields serves for all values of ${ \tilde { \kappa } } .$ The estimator $\begin{array} { r } { q _ { 1 } - \sum _ { \gamma } \pi _ { \gamma } Q _ { \alpha \gamma } } \end{array}$ has mean $\Delta _ { c }$ by the second form of Eq. (64), and is averaged over trees and field draws.

(v) Truncation. Each node has infinitely many children, almost all of them extremely light, so the tree must be truncated. Along a branch that leaves the planted path at depth $j ,$ children lighter than a fraction $w _ { \mathrm { r e l } }$ of the mass of the planted path at depth $j$ are merged, together with the mass not yet broken of, into a single lump per parent, which keeps its exact total weight and its exact overlap $q ^ { ( j ) }$ with the planted state. The competitors of the planted state at the finest level, which carry the error at large ${ \tilde { \kappa } } ,$ are kept down to a much smaller fraction $w _ { \mathrm { l e a f } }$ of the mass of their parent. A lump attached to a node at depth $d \geq j$ carries the field of that node, while the increments below it are unresolved. They are replaced by the factor $\mathrm { e } ^ { \tilde { \kappa } ( q _ { 1 } - q ^ { ( d ) } ) / 2 }$ that they contribute on average. The truncation error should vanish as $w _ { \mathrm { r e l } } , w _ { \mathrm { l e a f } }  0$ . Monte Carlo points are shown where the result no longer changes when $w _ { \mathrm { l e a f } }$ is lowered by a factor 10, down to 10<sup>−7</sup>, while we fix $w _ { \mathrm { r e l } } = 1 0 ^ { - 5 }$

## F Replica computations at fixed κ

This appendix gives the replica computations behind Sec. 5.2 for the internal measure $p ( { \pmb x } \mid { \pmb y } , { \kappa } )$ at fixed $\kappa > 0$ , after $N  \infty$ . We first treat the naive case, in which $\textbf {  { y } }$ is independent of the disorder, and then the planted case, where y is a noisy version of an equilibrium configuration and the unknown is the posterior width $\Delta ( \kappa ) = L ^ { 2 } - q _ { \mathrm { p o s t } }$ , i.e. the per-coordinate MMSE.

## F.1 Naive independent pinning

A natural first attempt is to consider $Z ( { \pmb y } , \kappa ) \propto \int$ dx exp $\begin{array} { r } { \big ( - V ( \pmb { x } ) - \frac { \kappa } { 2 } | \pmb { y } - \pmb { x } | ^ { 2 } \big ) } \end{array}$ for a y fixed independently of the disorder, and to compute log $\overline { { Z ( { \pmb y } , { \boldsymbol \kappa } ) } }$ with replicas as for the equilibrium problem. Since the data lie on the sphere $| { \pmb x } | ^ { 2 } = N L ^ { 2 }$ , the internal measure is $\propto \exp [ - V ( { \pmb x } ) +$ $\kappa \boldsymbol { y } \cdot \boldsymbol { x } ]$ , corresponding to the original problem in an external field directed along $^ { y , }$ of strength $h ^ { \bar { 2 } } = \bar { | } \kappa y | ^ { 2 } / \bar { N } = \kappa ^ { 2 } \bar { r } ^ { 2 }$ , with $r ^ { 2 } = | \pmb { y } | ^ { 2 } / N$ . By isotropy, the result depends only on $h ,$ and averaging over the direction of y turns the field into a channel term acting on the inner replicas only,

$$
{ \mathcal S } _ { h } ( Q ) = \frac { 1 } { 2 } \log \operatorname* { d e t } Q + \frac { 1 } { 2 } \sum _ { a , b } { \mathcal B } ( D _ { a b } ^ { 2 } ) + \frac { h ^ { 2 } } { 2 } \sum _ { a , b } Q _ { a b } .\tag{119}
$$

For the replica-symmetric ansatz $Q _ { a \neq b } = q , \Delta = L ^ { 2 } - q$ , the saddle-point equation becomes

$$
\frac { q } { \Delta ^ { 2 } } = 2 ^ { H } \widetilde { \sigma } ^ { 2 } H \Delta ^ { H - 1 } + h ^ { 2 } .\tag{120}
$$

The field term being linear in $Q$ , the replicon is unchanged, $\lambda _ { R } = \Delta ^ { - 2 } - 4 B ^ { \prime \prime } ( 2 \Delta )$ , and the RS solution loses stability when

$$
1 = 2 ^ { H } \tilde { \sigma } ^ { 2 } H ( 1 - H ) \Delta ^ { H } ,\tag{121}
$$

i.e. at $\Delta = \Delta _ { 1 } ~ [ \mathrm { E q . ~ ( 2 4 ) } ]$ . Eliminating q between Eqs. (120) and (121) gives the AT field

$$
h _ { \mathrm { A T } } ^ { 2 } = \frac { L ^ { 2 } } { \Delta _ { 1 } ^ { 2 } } - \frac { 2 - H } { 1 - H } \frac { 1 } { \Delta _ { 1 } } .\tag{122}
$$

The internal measure is replica symmetric for $h ^ { 2 } > h _ { \mathrm { A T } } ^ { 2 }$ and fRSB for $h ^ { 2 } < h _ { \mathrm { A T } } ^ { 2 } \ ( h _ { \mathrm { A T } } = 0$ at $\tilde { \sigma } = \tilde { \sigma } _ { \mathrm { A T } }$ , where $\Delta _ { 1 } = L ^ { 2 } ( 1 - H ) / ( 2 - H ) )$

Along a typical noisy trajectory, we have $\pmb { y } = \pmb { x } _ { 0 } + \pmb { z } / \sqrt { \kappa }$ which by concentration sits on the sphere $r ^ { 2 } \to L ^ { 2 } + 1 / \kappa$ , hence

$$
h _ { \mathrm { t y p } } ^ { 2 } ( \kappa ) = \kappa ^ { 2 } L ^ { 2 } + \kappa .\tag{123}
$$

![](images/f0632a434fe4f8710dad02799e7c46eae6c391500750717006bd3421e55aacdd.jpg)

![](images/66cc23c8639b985d31d5ae50581340036cd7ac53a5402d025df053f9770b2516.jpg)  
Figure 8: Independent (non-planted) pinning: replica-symmetry breaking of the internal measure $\propto \mathrm { e } ^ { - V ( \pmb { x } ) + \kappa \pmb { y } \cdot \pmb { x } }$ for $\textbf {  { y } }$ independent of the disorder, at $H = 1 / 2$ and $L = 1$ . Left: phase diagram in the plane of disorder strength $\tilde { \sigma } ^ { 2 }$ and pinning field $h ^ { 2 } = | \kappa y | ^ { 2 } / N ;$ ; the internal measure is RS above the AT line $h ^ { 2 } = h _ { \mathrm { A T } } ^ { 2 } ( \tilde { \sigma } )$ of Eq. (122) and fRSB below it. Right: the same transition along a typical difusion trajectory, $h ^ { 2 } = h _ { \mathrm { t v p } } ^ { 2 } ( \kappa ) = \kappa ^ { 2 } L ^ { 2 } +$ κ with $\kappa = 1 / ( 2 t )$ , in the plane of difusion time t and $\tilde { \sigma } ^ { 2 } ;$ the region $t > t _ { \mathrm { A T } } ^ { \mathrm { i n t } } ( \tilde { \sigma } )$ [Eq. (124)] would be fRSB. Only the strong-disorder regime $\tilde { \sigma } ^ { 2 } \geq \tilde { \sigma } _ { \mathrm { A T } } ^ { 2 }$ is shown. This is not the conditioning ensemble relevant for difusion: for a planted $^ { y , }$ , the internal measure is RS for every fixed $\kappa > 0 \ ( \mathrm { A p p . \ F . 4 } )$

Substituting it into the AT criterion, $h _ { \mathrm { t y p } } ^ { 2 } ( \kappa _ { \mathrm { A T } } ^ { \mathrm { i n t } } ) = h _ { \mathrm { A T } } ^ { 2 } ,$ gives

$$
\kappa _ { \mathrm { A T } } ^ { \mathrm { i n t } } = \frac { \sqrt { 1 + 4 L ^ { 2 } h _ { \mathrm { A T } } ^ { 2 } } - 1 } { 2 L ^ { 2 } } ,\tag{124}
$$

i.e. $t _ { \mathrm { A T } } ^ { \mathrm { i n t } } = 1 / ( 2 \kappa _ { \mathrm { A T } } ^ { \mathrm { i n t } } )$ and the phase diagram of Fig. 8. At strong disorder $\kappa _ { \mathrm { A T } } ^ { \mathrm { i n t } } \simeq 1 / \Delta _ { 1 } \propto \tilde { \sigma } ^ { 2 / H }$ ， which is the disorder dependence of the end of the cascade, $\tilde { \kappa } _ { c } ,$ but at $\kappa = O ( 1 )$ instead of $O ( 1 / N )$ . Clearly this contradicts the main result of our paper which is that speciation occurs entirely on the ˜κ scale.

## F.2 The planted object and its action

In order to obtain a result consistent with the Larkin picture, we must instead consider the strong correlation between the noisy observation and the disorder in the $\kappa = O ( 1 )$ regime.

We set out to compute $\mathcal { F } ( \kappa ) , \mathrm { E q . } ( 6 7 )$ , with $\begin{array} { r } { Z ( \pmb { y } , \kappa ) = \int \mathrm { d } \pmb { x } \mathrm { e } ^ { - V ( \pmb { x } ) + \kappa \pmb { y } \cdot \pmb { x } } } \end{array}$ . Constants $\kappa | \boldsymbol { y } | ^ { 2 } / 2$ and $\kappa | \pmb { x } | ^ { 2 } / 2$ have been dropped $( | \pmb { x } | ^ { 2 } = N L ^ { 2 } )$ , so that $Z$ defines the same internal measure as Eq. (12). In this planted computation, two replica tricks are needed, first to compute the normalisation $\begin{array} { r } { \mathcal { Z } ^ { - 1 } = \operatorname* { l i m } _ { m  0 } \mathcal { Z } ^ { m - 1 } } \end{array}$ , and then to do the quenched average log $Z = \operatorname { l i m } _ { n  0 } ( Z ^ { n } -$ $1 ) / n$ for the inner free energy. Both are taken on

$$
\Xi ( m , n ) = { \overline { { \mathcal { Z } ^ { m - 1 } \int \mathrm { d } { \pmb x } _ { 0 } \mathrm { e } ^ { - V ( { \pmb x } _ { 0 } ) } \mathbb { E } _ { z } Z ( { \pmb y } , \kappa ) ^ { n } } } } ,\tag{125}
$$

which satisfies $\Xi ( m , 0 ) = \overline { { \mathcal { Z } ^ { m } } } \to 1$ , so that

$$
{ \frac { \mathcal { F } ( \kappa ) } { N } } = \operatorname* { l i m } _ { n  0 } \operatorname* { l i m } _ { m  0 } { \frac { 1 } { n N } } { \big [ } \log \Xi ( m , n ) - \log \Xi ( m , 0 ) { \big ] } .\tag{126}
$$

Writing the powers as products of integrals introduces two sets of replicas: the plant set $\mathsf { P } = \{ \pmb { x } _ { 0 } , \pmb { x } _ { 1 } ^ { \prime } , \ldots , \pmb { x } _ { m - 1 } ^ { \prime } \}$ , m 0 replicas of the equilibrium measure, and the inner set $\mid { \bf \mu } =$

$\{ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } \} , n  0$ replicas of the internal measure. Explicitly,

$$
\Xi ( m , n ) = \overline { { \int \prod _ { c = 1 } ^ { m - 1 } \mathrm { d } x _ { c } ^ { \prime } \mathrm { e } ^ { - V ( x _ { c } ^ { \prime } ) } \int \mathrm { d } x _ { 0 } \mathrm { e } ^ { - V ( x _ { 0 } ) } \mathbb { E } _ { z } \prod _ { a = 1 } ^ { n } \int \mathrm { d } x _ { a } \mathrm { e } ^ { - V ( x _ { a } ) + \kappa x _ { a } \cdot ( x _ { 0 } + z / \sqrt { \kappa } ) } } } ,\tag{127}
$$

We start by integrating over $_ { z , }$

$$
\mathbb { E } _ { z } \prod _ { a \in 1 } \mathrm { e } ^ { \kappa x _ { a } \cdot ( x _ { 0 } + z / \sqrt { \kappa } ) } = \exp \left[ \kappa \sum _ { a \in 1 } x _ { 0 } \cdot x _ { a } + \frac { \kappa } { 2 } \left| \sum _ { a \in 1 } x _ { a } \right| ^ { 2 } \right] = \exp \left[ \frac { \kappa } { 2 } \left| x _ { 0 } + \sum _ { a \in 1 } x _ { a } \right| ^ { 2 } - \frac { \kappa } { 2 } | x _ { 0 } | ^ { 2 } \right] .\tag{128}
$$

Since $| { \pmb x } _ { 0 } | ^ { 2 } = N L ^ { 2 }$ is a constant, the channel couples the planted replica $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ and the inner replicas symmetrically, meaning that $\scriptstyle { \mathbf { \mathscr { x } } } _ { 0 }$ is a posterior sample, exchangeable with $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n }$ . This is the replica-level expression of the Nishimori identity, and is what makes the computation tractable.

After the Gaussian disorder average over all $m + n$ replicas (as for the equilibrium problem, $\begin{array} { r } { \mathrm { A p p . ~ A } ) , \Xi ( m , n ) \asymp \int _ { \mathscr { Q } _ { m + n } } \mathrm { d } Q \operatorname { e } ^ { N S ( Q ) } } \end{array}$ with

$$
\mathcal { S } ( Q ) = \frac { 1 } { 2 } \log \operatorname* { d e t } Q + \frac { 1 } { 2 } \sum _ { i , j \in \mathsf { P } \cup \mathsf { I } } \mathsf { B } ( D _ { i j } ^ { 2 } ) + \frac { \kappa } { 2 } \sum _ { i , j \in \{ 0 \} \cup \mathsf { I } } Q _ { i j } - \frac { \kappa } { 2 } L ^ { 2 } ,\tag{129}
$$

where $Q$ is the $( m + n ) \times ( m + n )$ overlap matrix between all the configurations, both from the planted and the inner set, $D _ { i j } ^ { 2 } = 2 ( L ^ { 2 } - Q _ { i j } )$ , and $\mathcal { Q } _ { m + n }$ is the set of positive-definite matrices with diagonal $L ^ { 2 }$ . The indices $i , j$ therefore run over all replicas. In the following, indices $a , b$ will be reserved for inner replicas, and c for the plant ones other than $\scriptstyle { \mathbf { \mathscr { x } } } _ { 0 }$ . The first two terms are the equilibrium action for $m + n$ replicas, while the channel adds a term linear in $Q$ on the $n + 1$ exchangeable replicas $\{ 0 \} \cup \{$

Finding ourselves with these two sets of replicas, we decompose the overlaps with the following mixed replica-symmetry ansatz:

• The plant block $Q ^ { \mathsf { P } } \ ( m \times m )$ : this block inherits all of the properties of the equilibrium measure described in Sec. 2.2. As such, in the strong disorder regime it is natural to expect full replica-symmetry breaking, with diagonal overlaps $L ^ { 2 }$ and a continuous function $u ( q )$ on $[ q _ { 0 } , q _ { 1 } ]$ as we take $m \to 0$

• The inner block $Q ^ { \parallel } \left( n \times n \right)$ : following speciation, we expect it to be replica symmetric, with $Q _ { a b } = q _ { \mathrm { p o s t } }$ for $a \neq b$ , with $q _ { \mathrm { p o s t } } \geq q _ { 1 }$ (we will equivalently write $\Delta : = L ^ { 2 } - q _ { \mathrm { p o s t } } \leq \Delta _ { 1 } )$

• The cross block: within the posterior RS ansatz, the Nishimori identity gives $Q _ { a 0 } = q _ { \mathrm { p o s t } } .$ equal to the overlap of two distinct inner replicas. We further assume that the inner replicas follow the branch containing $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ , inheriting its overlaps with the other plant replicas: $Q _ { a c } = Q _ { 0 c }$ for $c \in \mathsf { P } \setminus \{ 0 \}$ and $a \in { \mathsf { I } }$ . These cross overlaps are bounded by $q _ { 1 }$

Ordering the replicas as (P, I) with $\scriptstyle { \mathbf { \mathscr { x } } } _ { 0 }$ first, and writing q for the column $( q _ { \mathrm { p o s t } } , Q _ { 0 1 } , \ldots , Q _ { 0 , m - 1 } ) ^ { \top }$

$$
Q = \left( \mathbf { { \cal { Q } } } _ { n } ^ { \mathrm { { P } } } \quad \Delta \mathbf { { \cal { I } } } _ { n } ^ { \top } \quad \right) ,\tag{130}
$$

with ${ \bf J } _ { n }$ an $n \times n$ matrix of all ones. At leading order in $n .$ , the plant block is evaluated on its equilibrium saddle. Within this ansatz, the only remaining unknown is thus $q _ { \mathrm { p o s t } } ( \kappa )$ , which we determine by stationarity of the action.

## F.3 Evaluation of the action

We evaluate the action (129) on the mixed ansatz (130). Given $q _ { \mathrm { p o s t } } > q _ { 1 } \geq Q _ { 0 c }$ , its cross block is also what ultrametricity of the triplet $( \pmb { x } _ { 0 } , \pmb { x } _ { a } , \pmb { x } _ { c } ^ { \prime } )$ forces, and the $O ( n )$ shift of the plant block away from its equilibrium saddle contributes only at $O ( n ^ { 2 } )$ , by stationarity.

Determinant term. By the Schur complement,

$$
\operatorname* { d e t } Q = \operatorname* { d e t } Q ^ { \mathrm { P } } \cdot \operatorname* { d e t } \big [ \Delta \mathbf { I } _ { n } + ( q _ { \mathrm { p o s t } } - \Upsilon ) \mathbf { J } _ { n } \big ] ,\tag{131}
$$

with $\Upsilon : = { \pmb q } ^ { \top } ( Q ^ { \mathsf { P } } ) ^ { - 1 } { \pmb q }$ . Since de $\left[ \Delta \mathbf { I } _ { n } + b \mathbf { J } _ { n } \right] = \Delta ^ { n - 1 } ( \Delta + n b )$

$$
\frac { 1 } { 2 } \log \operatorname * { d e t } Q = \frac { 1 } { 2 } \log \operatorname * { d e t } Q ^ { \mathrm { P } } + \frac { n } { 2 } \left[ \log \Delta + \frac { q _ { \mathrm { p o s t } } - \Upsilon } { \Delta } \right] + O ( n ^ { 2 } ) .\tag{132}
$$

The vector q is the 0-th column of $Q ^ { \mathsf { P } }$ with its diagonal entry $L ^ { 2 }$ replaced by $q _ { \mathrm { p o s t } }$ , i.e. $q =$ $Q ^ { \mathsf { P } } e _ { 0 } - \Delta e _ { 0 }$ . Hence $\Upsilon = L ^ { 2 } - 2 \Delta + \Delta ^ { 2 } [ ( Q ^ { \mathsf { P } } ) ^ { - 1 } ] _ { 0 0 }$ , and thus

$$
\frac { q _ { \mathrm { p o s t } } - \Upsilon } { \Delta } = 1 - \Delta \mathcal { P } ,\tag{133}
$$

where $\begin{array} { r } { \mathcal { P } : = \operatorname* { l i m } _ { m  0 } [ ( Q ^ { \mathsf { P } } ) ^ { - 1 } ] _ { 0 0 } } \end{array}$ is the diagonal element of the inverse equilibrium Parisi matrix in the replica limit.

Diagonal element of the inverse Parisi matrix. To evaluate $\mathcal { P } _ { \cdot }$ , recall from App. A the function $\begin{array} { r } { \chi ( \boldsymbol { q } ) = \Delta _ { 1 } + \int _ { \boldsymbol { a } } ^ { q _ { 1 } } \mathrm { d } p u ( \boldsymbol { p } ) \ [ \mathrm { E q . ~ ( 8 3 ) } ] } \end{array}$ , constant for $q \leq q _ { 0 }$ , and the equilibrium saddle relations $\chi ( q ) ^ { - 2 } = 4 B ^ { \prime \prime } ( \dot { 2 } ( L ^ { 2 } - q ) )$ for $q _ { 0 } < q < q _ { 1 } , \chi ( q _ { 1 } ) = \Delta _ { 1 }$ and $q _ { 0 } = - B ^ { \prime } ( D _ { 0 } ^ { 2 } ) / ( 2 B ^ { \prime \prime } ( D _ { 0 } ^ { 2 } ) )$ [Eqs. (86)–(89)]. In this fRSB setting it is a standard result that

$$
\operatorname * { l i m } _ { m \to 0 } { \frac { 1 } { m } } \log \operatorname * { d e t } Q ^ { \mathsf { P } } = \log \Delta _ { 1 } + \int _ { q _ { 0 } } ^ { q _ { 1 } } { \frac { \mathrm { d } q } { \chi ( q ) } } + { \frac { q _ { 0 } } { \chi ( q _ { 0 } ) } } .\tag{134}
$$

By Jacobi’s formula, a uniform shift of the diagonal $L ^ { 2 }$ at fixed of-diagonal entries gives $\partial _ { L ^ { 2 } }$ log det $Q ^ { \mathsf { P } } = { \mathrm { T r } } ( Q ^ { \mathsf { P } } ) ^ { - 1 } = m { \mathcal { P } }$ . Since $L ^ { 2 }$ enters (134) only through $\Delta _ { 1 }$ and $\chi ,$ with $\partial _ { L ^ { 2 } } \chi = 1$

$$
\mathcal { P } = \frac { 1 } { \Delta _ { 1 } } - \int _ { q _ { 0 } } ^ { q _ { 1 } } \frac { \mathrm { d } q } { \chi ( q ) ^ { 2 } } - \frac { q _ { 0 } } { \chi ( q _ { 0 } ) ^ { 2 } } .\tag{135}
$$

On the equilibrium solution, $\chi ^ { - 2 } = 4 B ^ { \prime \prime } ( D ^ { 2 } )$ on the continuum (with $\mathrm { d } q = - \mathrm { d } D ^ { 2 } / 2 )$ and, by continuity of $\chi , q _ { 0 } / \chi ( q _ { 0 } ) ^ { 2 } = - 2 B ^ { \prime } ( D _ { 0 } ^ { 2 } )$ , so that the $B ^ { \prime } ( D _ { 0 } ^ { 2 } )$ contributions cancel:

$$
\int _ { q _ { 0 } } ^ { q _ { 1 } } \frac { \mathrm { d } q } { \chi ( q ) ^ { 2 } } + \frac { q _ { 0 } } { \chi ( q _ { 0 } ) ^ { 2 } } = 2 B ^ { \prime } ( D _ { 0 } ^ { 2 } ) - 2 B ^ { \prime } ( D _ { 1 } ^ { 2 } ) - 2 B ^ { \prime } ( D _ { 0 } ^ { 2 } ) = - 2 B ^ { \prime } ( D _ { 1 } ^ { 2 } ) ,\tag{136}
$$

finally leading to

$$
\mathcal { P } = \frac { 1 } { \Delta _ { 1 } } + 2 B ^ { \prime } ( 2 \Delta _ { 1 } ) .\tag{137}
$$

The lower plateau and the entire continuum drop out, and the plant block enters the innerblock equations only through $\Delta _ { 1 }$ . In terms of the stifness $\mu$ of Sec. $5 . 2 , \mathcal { P } = \mu ( \Delta _ { 1 } )$ . Combining with Eqs. (132)–(133), the determinant term contributes $\begin{array} { r } { \frac { n } { 2 } [ \log \Delta + 1 - \Delta \mu ( \Delta _ { 1 } ) ] + O ( n ^ { 2 } ) } \end{array}$ , the n-independent $\frac { 1 } { 2 }$ log det $Q ^ { \mathsf { P } }$ cancelling in Eq. (126).

Disorder and channel terms. We split the disorder sum as $\sum _ { \mathsf { P } , \mathsf { P } } + 2 \sum _ { \mathsf { P } , \mathsf { I } } + \sum _ { \mathsf { I } , \mathsf { I } }$ . The first is n-independent, so it again drops from log $\Xi ( m , n ) { - } \log \Xi ( m , 0 )$ . The last is $n \dot { B } ( 0 ) + \dot { n } ( n - 1 ) B ( 2 \Delta )$ by replica symmetry. In the cross sum, each inner replica contributes $B ( 2 \Delta )$ from $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ and $\begin{array} { r l } { - \int _ { 0 } ^ { 1 } \mathrm { d } u B ( D ^ { 2 } ( u ) ) } \end{array}$ from the other plant replicas, using the standard $m \to 0$ rule $\begin{array} { r } { \sum _ { c \neq 0 } f ( Q _ { 0 c } )  } \end{array}$ $\begin{array} { r l } { \mathrm { ~ - ~ } \int _ { 0 } ^ { 1 } \mathrm { d } u f ( q ( u ) ) } \end{array}$ . Altogether,

$$
\frac { 1 } { 2 } \sum _ { i , j } \mathcal { B } ( D _ { i j } ^ { 2 } ) = \mathrm { c o n s t } + \frac { n } { 2 } \left[ \mathcal { B } ( 0 ) + \mathcal { B } ( 2 \Delta ) - 2 \int _ { 0 } ^ { 1 } \mathrm { d } u \mathcal { B } ( D ^ { 2 } ( u ) ) \right] + O ( n ^ { 2 } ) .\tag{138}
$$

Here $D ( u )$ includes the plateaus of the Parisi function, $D ( u ) = D _ { 0 }$ for $u < u _ { 0 }$ and $D ( u ) = D _ { 1 }$ for $u > u _ { 1 }$ , Eq. (26) giving it in between. The channel term involves the $n + 1$ replicas $\{ 0 \} \cup \mathsf { I } .$ all at mutual overlap $q _ { \mathrm { p o s t } }$

$$
\frac { \kappa } { 2 } \sum _ { i , j \in \{ 0 \} \cup \cup \atop i , j \in \{ 0 \} \cup \cup \atop i } Q _ { i j } - \frac { \kappa } { 2 } L ^ { 2 } = \frac { \kappa } { 2 } \big [ ( n + 1 ) L ^ { 2 } + ( n + 1 ) n q _ { \mathrm { p o s t } } \big ] - \frac { \kappa } { 2 } L ^ { 2 } = \frac { n \kappa } { 2 } \big ( L ^ { 2 } + q _ { \mathrm { p o s t } } \big ) + O ( n ^ { 2 } ) .\tag{139}
$$

Resulting planted free entropy. Collecting the $O ( n )$ coeficients, the planted log-partition function per site, $\mathcal { F } ( \kappa ) / N$ , is the extremum over $\Delta$ of

$$
\phi ( \kappa , \Delta ) = \frac { 1 } { 2 } \log \Delta + \frac { 1 } { 2 } - \frac { \Delta } { 2 } \left[ \frac { 1 } { \Delta _ { 1 } } + 2 \mathcal { B } ^ { \prime } ( 2 \Delta _ { 1 } ) \right] + \frac { 1 } { 2 } \mathcal { B } ( 2 \Delta ) + \frac { \kappa } { 2 } \big ( 2 L ^ { 2 } - \Delta \big ) + C _ { \mathrm { e q } } ,\tag{140}
$$

with $\begin{array} { r } { C _ { \mathrm { e q } } = \frac { 1 } { 2 } B ( 0 ) - \int _ { 0 } ^ { 1 } } \end{array}$ du $B ( D ^ { 2 } ( u ) )$ an equilibrium constant. The structure is efectively that of a Franz–Parisi-type potential [75] for the overlap with the reference configuration.

## F.4 Saddle point and stability

In the planted free entropy (140), the lower plateau and the whole continuum of the Parisi solution have dropped out, and the plant enters only through the width $\Delta _ { 1 }$ of its state. Its extremum over the single remaining direction $\Delta$ is conveniently written in terms of the withinstate “stifness”

$$
\mu ( \Delta ) = \frac { 1 } { \Delta } + 2 B ^ { \prime } ( 2 \Delta ) ,\tag{141}
$$

for which $\begin{array} { r } { \partial _ { \Delta } \phi = \frac { 1 } { 2 } \big [ \mu ( \Delta ) - \mu ( \Delta _ { 1 } ) - \kappa \big ] } \end{array}$ . Stationarity thus gives the self-consistent equation (69), $\mu ( \Delta ) = \mu ( \Delta _ { 1 } ) + \kappa , \mathrm { i . e }$ . pinning raises the within-state stifness by exactly $\kappa .$ . For our long-range family, $2 \bar { B ^ { \prime } } ( 2 \bar { \Delta } ) = - 2 ^ { \bar { H } } \tilde { \sigma } ^ { 2 } H \bar { \Delta ^ { H - 1 } }$ , and using Eq. (24) to substitute $2 ^ { \bar { H } } \tilde { \sigma } ^ { 2 } \Delta _ { 1 } ^ { H } H = 1 / ( \bar { 1 } - \bar { H ^ { } } )$ the saddle-point equation takes the form (70) in terms of the normalised width $\tilde { \Delta } = \Delta / \Delta _ { 1 }$ . For $\begin{array} { r } { H = \frac { 1 } { 2 } } \end{array}$ its right-hand side reduces to $( 1 / \sqrt { \tilde { \Delta } } - 1 ) ^ { 2 }$ , which gives the closed form

$$
\Delta ( \kappa ) = \frac { \Delta _ { 1 } } { ( 1 + \sqrt { \kappa \Delta _ { 1 } } ) ^ { 2 } } .\tag{142}
$$

In general, the solution saturates at $\Delta _ { 1 }$ as $\kappa  0$ , as expected by construction of the ansatz, while as $\kappa  \infty$ the term $1 / \tilde { \Delta }$ dominates the right-hand side for every H, so that $\Delta ( \kappa ) \simeq 1 / \kappa$

By the Nishimori identity, $\Delta ( \kappa )$ is the MMSE of the denoiser, shown in Fig. 9 for several values of H. For comparison, the dotted lines show a Gaussian state of the same width, for which the disorder term is absent, $\mu ( \Delta ) = 1 / \Delta$ , so that Eq. (69) gives $\Delta = \Delta _ { 1 } / ( 1 + \kappa \Delta _ { 1 } )$ , the MMSE of a Gaussian prior of variance $\Delta _ { 1 }$ . Both start at $\Delta _ { 1 }$ and decay as $1 / \kappa .$ , the true error lying below the Gaussian one at every κ and converging to it at large κ.

It remains to check that the replica-symmetric inner block is stable, i.e. that the posterior at fixed y is indeed dominated by a single state for every $\kappa > 0$ . Here, the relevant fluctuations are those that break the symmetry between the inner replicas while leaving the rest of the ansatz untouched, namely symmetric perturbations $\delta Q$ of the inner block with $\delta Q _ { a a } = 0$ and $\begin{array} { r } { \sum _ { b \in \mathsf { I } } \delta Q _ { a b } = 0 . } \end{array}$ , the replicon fluctuations. Their vanishing row sums remove the rank-one coupling to the plant block, so that $Q ^ { - 1 }$ acts on them simply as $1 / \Delta$ , while the channel term, being linear in $Q ,$ does not contribute at second order. Expanding the determinant and disorder terms of Eq. (129) then gives $\begin{array} { r } { \mathcal { S } ( Q + \delta Q ) - \mathcal { S } ( Q ) = - \frac { 1 } { 4 } \lambda _ { R } ^ { \mathrm { p o s t } } \operatorname { T r } ( \delta Q ) ^ { 2 } } \end{array}$ , with the replicon eigenvalue

$$
\lambda _ { R } ^ { \mathrm { p o s t } } = \frac { 1 } { \Delta ^ { 2 } } - 4 B ^ { \prime \prime } ( 2 \Delta ) = - \mu ^ { \prime } ( \Delta ) ,\tag{143}
$$

i.e. minus the slope of the within-state stifness. Using Eq. (24) to write $4 B ^ { \prime \prime } ( 2 \Delta ) = \Delta _ { 1 } ^ { - H } \Delta ^ { H - 2 }$ its long-range form is Eq. (71). It is positive for every $0 < \Delta < \Delta _ { 1 }$ , and vanishes only at the equilibrium endpoint $\Delta = \Delta _ { 1 }$ , where the pure state is marginal. The saddle point indeed remains in this interval. Writing $g ( \Delta ) : = \mu ( \Delta ) - \mu ( \Delta _ { 1 } )$ , Eq. (69) reads $g ( \Delta ) = \kappa$ , with $g ( \Delta _ { 1 } ) = 0$ $g ( 0 ^ { + } ) = + \infty$ and $g ^ { \prime } = - \lambda _ { R } ^ { \mathrm { p o s t } } < 0 \mathrm { o n } ( 0 , \Delta _ { 1 } )$ , so that there is a unique solution on this branch for every fixed $\kappa > 0$ , and it is stable. A root with $\Delta > \Delta _ { 1 }$ , where $\lambda _ { R } ^ { \mathrm { p o s t } } < 0$ , would instead be unstable and would violate the assumption $q _ { \mathrm { p o s t } } \geq q _ { 1 }$ of the ansatz. The scalar curvature is consistent with this, $\partial _ { \Delta } ^ { 2 } \phi = - \lambda _ { R } ^ { \mathrm { p o s t } } / 2 < 0$ , the free entropy being maximised over the order parameter on the Nishimori line. Unlike the independently pinned problem of App. F.1, the planted saddle therefore has no AT instability at any finite $\kappa > 0$ , perfectly consistent with our global picture.

![](images/d901bf330293782c54de3bf264715c887782454ead087ff29ff96e42a9225f68.jpg)  
Figure 9: Posterior width, or equivalently denoising MMSE $\Delta ( \kappa )$ on the planted replicasymmetric branch, Eq. (69), for $H \ : = \ : 0 . 2 5 , 0 . 5 , 0 . 7 5$ at $\tilde { \sigma } \ = \ 1 0$ . At small κ it saturates at the thermal width $\Delta _ { 1 }$ of the selected pure state, and at large κ, $\Delta \simeq 1 / \kappa$ . Dotted, Gaussian state of the same width, $\Delta _ { 1 } / ( 1 + \kappa \Delta _ { 1 } )$ . Inset: $1 - ( \Delta / \Delta _ { 1 } ) ^ { H }$ , equal to $\dot { \Delta } ^ { 2 } \lambda _ { R } ^ { \mathrm { p o s t } }$ on the planted branch, versus $\kappa \Delta _ { 1 }$ . The planted branch leaves $\Delta _ { 1 }$ as $\sqrt { \kappa } ,$ , inherited from the marginality of the pure states $\left( \mathrm { A p p . ~ F . 5 } \right)$ , whereas the Gaussian state leaves it linearly.

## F.5 Small-κ behaviour

The behaviour of the planted branch at small κ is also consistent with the speciation picture of Sec. 5.1. As $\kappa  0 ^ { + }$ the planted branch tends to the width $\Delta _ { 1 }$ of a single pure state, while its stability margin closes, $\lambda _ { R } ^ { \mathrm { p o s t } } \to 0$ . This marginality is to be expected, since the selection of a single state, strict at any fixed $\kappa > 0$ , must give way to the competition between pure states of the regime $\kappa = \tilde { \kappa } / N$ . Here it takes the form of a $\sqrt { \kappa }$ departure from $\Delta _ { 1 }$ . Expanding the saddle-point equation (70) around $\begin{array} { r } { \tilde { \Delta } = 1 , \kappa \Delta _ { 1 } = \frac { H } { 2 } ( 1 - \tilde { \Delta } ) ^ { 2 } + O \big ( ( 1 - \tilde { \Delta } ) ^ { 3 } \big ) } \end{array}$ , we indeed have

$$
\Delta ( \kappa ) = \Delta _ { 1 } - \sqrt { \frac { 2 \Delta _ { 1 } ^ { 3 } } { H } } \sqrt { \kappa } + { \cal O } ( \kappa ) , \qquad \lambda _ { R } ^ { \mathrm { p o s t } } = \sqrt { \frac { 2 H } { \Delta _ { 1 } ^ { 3 } } } \sqrt { \kappa } + { \cal O } ( \kappa ) .\tag{144}
$$

The $1 / 2$ exponent is due to the marginality of the pure states. It is thus a signature of the continuous replica-symmetry breaking brought by long-range correlations, H entering only the prefactor. For short-range correlations, whose measure breaks replica symmetry in one step and whose states are not marginal, the onset would instead be linear, as for the Gaussian state (inset of Fig. 9). This inheritance from marginality can also be seen through the so-called I-MMSE relations [76]. Their second-order form [77] gives $- \partial _ { \kappa } \Delta = N ^ { - 1 }$ E Tr $\hat { \mathbf { C } } ^ { 2 }$ for any prior, while diferentiating Eq. (69) with $\mu ^ { \prime } = - \lambda _ { R } ^ { \mathrm { p o s t } }$ gives $\partial _ { \kappa } \Delta = - 1 / \lambda _ { R } ^ { \mathrm { p o s t } }$ . Hence $\lambda _ { R } ^ { \mathrm { p o s t } } = \left[ N ^ { - 1 } \mathbb { E } \operatorname { T r } \hat { \mathbf { C } } ^ { 2 } \right] ^ { - 1 }$ and the diverging slope of the error as $\kappa  0$ inherits the diverging spin-glass susceptibility of the soft modes of the pure state (Sec. 3.3).

## G The source of the perturbed reverse dynamics

In this appendix, we provide further details supporting Sec. 6.

## G.1 Exact representations and transport of

(a) Posterior form. From the Tweedie and Hessian identities $( 1 3 ) , ( 1 4 ) , | \nabla W | ^ { 2 } = \kappa ^ { 2 } | \pmb { y } - \hat { \pmb { x } } | ^ { 2 }$ and $\nabla ^ { 2 } W = N \kappa - \kappa ^ { 2 } \mathrm { T r } \hat { \mathbf { C } }$ , so that, using $| \pmb { y } - \hat { \pmb { x } } | ^ { 2 } + \operatorname { T r } \hat { \mathbf { C } } = \langle | \pmb { y } - \pmb { x } | ^ { 2 } \rangle _ { \pmb { y } , \kappa }$

$$
\mathcal { I } ( \pmb { y } , \kappa ) = \kappa ^ { 2 } | \pmb { y } - \hat { \pmb { x } } | ^ { 2 } + \kappa ^ { 2 } \operatorname { T r } \hat { \mathbf { C } } - N \kappa = \kappa ^ { 2 } \langle | \pmb { y } - \pmb { x } | ^ { 2 } \rangle _ { \pmb { y } , \kappa } - N \kappa .\tag{145}
$$

The second form also follows directly from $\nabla _ { \boldsymbol { y } } ^ { 2 } \mathrm { e }$ −κ|y− $\mathbf { \nabla } \cdot \mathbf { x } ^ { | 2 } / 2 = \left( \kappa ^ { 2 } | \pmb { y } - \pmb { x } | ^ { 2 } - N \kappa \right) \mathrm { e } ^ { - \kappa | \pmb { y } - \pmb { x } | ^ { 2 } / 2 }$ It holds for any $P _ { 0 }$

(b) Conservation and fluctuations. $\begin{array} { r } { \int P \mathcal { I } \mathrm { d } \pmb { y } = \int \nabla ^ { 2 } P \mathrm { d } \pmb { y } = 0 } \end{array}$ . Equivalently, by the Nishimori identity, $\mathbb { E } | \pmb { y } - \hat { \pmb { x } } | ^ { 2 } + \mathbb { E } \operatorname { T r } \hat { \mathbf { C } } = \mathbb { E } | z | ^ { 2 } / \kappa = N / \kappa , \mathrm { i . e }$

$$
\mathcal { I } = \kappa ^ { 2 } \big ( | \pmb { y } - \pmb { \hat { x } } | ^ { 2 } - \mathbb { E } | \pmb { y } - \pmb { \hat { x } } | ^ { 2 } \big ) + \kappa ^ { 2 } \big ( \operatorname { T r } \hat { \mathbf { C } } - \mathbb { E } \operatorname { T r } \hat { \mathbf { C } } \big ) .\tag{146}
$$

Since $| \pmb { y } - \hat { \pmb { x } } | ^ { 2 } \simeq | \pmb { z } | ^ { 2 } / \kappa .$ , the unconditional standard deviation of is $\kappa \sqrt { 2 N }$ , much larger than the mixture excess $\kappa ^ { 2 } \mathrm { T r } \hat { \mathbf { C } } _ { \mathrm { i n t e r } } \ ( \mathrm { e . g . }$ . κΓ/4 for a wall at $\ell = \ell _ { \mathrm { L } }$ , with $\Gamma = O ( 1 )$ in $N )$ Walls and junctions are therefore visible in $\mathcal { I }$ only after conditioning on position as we do below. The quantity that detects them unconditionally is Tr $\hat { \mathbf { C } } = \kappa ^ { - 1 } ( N - \nabla ^ { 2 } W / \kappa )$ , i.e. $. \nabla ^ { 2 } W$ , not $\mathcal { I }$ directly.

(c) Transport. With $\partial _ { t } W = \nabla ^ { 2 } W - | \nabla W | ^ { 2 }$ (the KPZ equation (17) at $\beta = 1$ , forward time $t = 1 / ( 2 \kappa ) )$ one has $\partial _ { t } W = - \mathcal { I }$ , and diferentiating $\mathcal { I } = | \nabla W | ^ { 2 } - \nabla ^ { 2 } W$

$$
\begin{array} { r } { \partial _ { t } \mathcal { I } + 2 \nabla W \cdot \nabla \mathcal { I } = \nabla ^ { 2 } \mathcal { I } . } \end{array}\tag{147}
$$

is thus a passive scalar, advected by the Burgers velocity 2 W and difused, with no source term.

## G.2 Walls and junctions

To evaluate the source on the “skeleton” of the foam, we consider a point y where K cells meet, i.e. a wall between two cells $( K = 2 )$ or a junction of several. The centres of the K competing states span an afine subspace of dimension $r = K - 1$ , and the $N - r$ remaining directions are orthogonal to all of them. In these transverse directions the posterior does not distinguish the competing states, $\pmb { y } - \hat { \pmb { x } }$ is essentially the noise, $\kappa ^ { 2 } | \pmb { y } - \hat { \pmb { x } } | _ { \bot } ^ { 2 } \simeq ( N - r ) \kappa .$ and the extensive terms of Eq. (145) cancel on average, as they do for  as a whole since $\textstyle \int P { \mathcal { I } } = 0$ . In the r spanned directions, y sits where the competing states have equal posterior weights, so that xˆ is their barycentre, while C<sup>ˆ</sup> reduces to the inter-state covariance $\hat { \mathbf { C } } _ { \mathrm { i n t e r } }$ of the competing centres, with eigenvalues $c _ { 1 } , \ldots , c _ { r }$ . The component $\Pi ( { \pmb y } - \hat { { \pmb x } } )$ of $\pmb { y } - \hat { \pmb x }$ in this span vanishes only if the K states also have equal prior weights, y being otherwise displaced from the barycentre by their free-energy diferences. Averaging over the transverse noise,

$$
\mathbb { E } \big [ \mathcal { I } \mid \mathrm { j u n c t i o n } \big ] = \kappa ^ { 2 } | \Pi ( \pmb { y } - \hat { \pmb { x } } ) | ^ { 2 } + \kappa \sum _ { i = 1 } ^ { r } ( \kappa c _ { i } - 1 ) \ \geq \ - \sum _ { i = 1 } ^ { r } \Lambda _ { i } .\tag{148}
$$

where $\Lambda _ { i } = \kappa - \kappa ^ { 2 } c _ { i }$ are the curvatures of W along the span, Eq. (14). On the skeleton, $\mathcal { I }$ is thus minus the total curvature of $W$ across the junction, plus a non-negative term that vanishes for equal prior weights. It is positive whenever the total curvature of W along the span is negative. Concretely,

• Wall $( K = 2 , r = 1 , c = \ell ^ { 2 } / 4 )$ . For prior free energies difering by $\Delta F _ { \ l }$ , the wall is displaced from the bisector by $\Delta F / ( \kappa \ell )$ , so that $\mathbb { E } [ \mathcal { I } \ \lvert \mathrm { \ w a l l } ] = \kappa ( \kappa \ell ^ { 2 } / 4 - 1 ) + ( \Delta F ) ^ { 2 } / \ell ^ { 2 } =$ $\kappa \big [ \Gamma / 4 - 1 + ( \Delta F ) ^ { 2 } / \Gamma \big ]$ at $\ell = \ell _ { \mathrm { L } }$ . At the Larkin scale $\Delta F \sim E _ { \mathrm { g a p } } \sim \Gamma$ , so that both terms are of the same order.

• Ultrametric (simplex) junction. States branching at the same level of the ultrametric tree are mutually equidistant. K of them at squared distance $\ell ^ { 2 }$ thus sit at the vertices of a regular simplex $( K \le N + 1 )$ , which spans a $( K - 1 )$ )-dimensional afine subspace, and the junction where their cells meet has codimension $K - 1$ in y-space. For equal weights,

$$
\hat { \mathbf { C } } _ { \mathrm { i n t e r } } = \frac { 1 } { K } \sum _ { \alpha = 1 } ^ { K } ( \pmb { m } ^ { \alpha } - \bar { \pmb { m } } ) ( \pmb { m } ^ { \alpha } - \bar { \pmb { m } } ) ^ { \top } = \frac { \ell ^ { 2 } } { 2 K } \Pi _ { K - 1 } ,\tag{149}
$$

with $\Pi _ { K - 1 }$ the projector on their span. Up to the intra-state part, the curvature of $W$ is then $\kappa - \kappa ^ { 2 } \ell ^ { 2 } / ( 2 K )$ in the $K - 1$ in-simplex directions and $\kappa$ in the $N - K + 1$ transverse ones, so that for $\kappa \ell ^ { 2 } > 2 K$ the junction is a maximum of $W$ in $K - 1$ directions. The source reads

$$
\mathbb { E } \big [ \mathcal { I } \mid \mathrm { s i m p l e x } \big ] = \kappa ( K - 1 ) \left( \frac { \kappa \ell ^ { 2 } } { 2 K } - 1 \right) ,\tag{150}
$$

and is positive only for $K < \kappa \ell ^ { 2 } / 2$ . As K grows, more directions become concave, but each of them less so, since the $K - 1$ directions share the fixed spread of the centres, $\mathrm { E q . ~ ( 1 5 1 ) }$ The source is therefore largest for $K \simeq ( \kappa \ell ^ { 2 } / 2 ) ^ { 1 / 2 }$ , where it reaches $\kappa [ ( \kappa \ell ^ { 2 } / 2 ) ^ { 1 / 2 } - 1 ] ^ { 2 } \simeq$ $\kappa ^ { 2 } \ell ^ { 2 } / 2$ . At the cell scale, $\ell = \ell _ { \mathrm { L } }$ , this is $K \simeq \sqrt { \Gamma / 2 }$ , with a source $\simeq \kappa \Gamma / 2$ of the same order as on a wall, $\kappa ( \Gamma / 4 - 1 )$ , as stated in Sec. 6. The codimension of a junction thus controls the number of unstable directions of $W$ , but not the sign of $\mathcal { I }$

• General bound. For any posterior over states,

$$
\mathrm { T r } \hat { \mathbf { C } } _ { \mathrm { i n t e r } } = \frac { 1 } { 2 } \sum _ { \alpha , \gamma } \pi _ { \alpha } \pi _ { \gamma } | \boldsymbol m ^ { \alpha } - \boldsymbol m ^ { \gamma } | ^ { 2 } \leq \frac { 1 } { 2 } \operatorname* { m a x } _ { \alpha , \gamma } | \boldsymbol m ^ { \alpha } - \boldsymbol m ^ { \gamma } | ^ { 2 } ,\tag{151}
$$

half the squared diameter of the posterior support. By ultrametricity, this diameter does not grow with the number of competing states. A junction mixing several levels of the tree is a simplex of simplices, whose source is dominated by its coarsest active level. For states at a common scale $\ell \sim \ell _ { \mathrm { L } }$ , the excess is thus bounded, $\kappa ^ { 2 } \mathrm { T r } \hat { \mathbf { C } } _ { \mathrm { i n t e r } } \leq \kappa \Gamma / 2$ , whatever the number of competing states and the codimension of the junction. In a sharp foam, junctions are therefore fed at most about twice as strongly as walls, and far more strongly than the cores of the cells. Being of order $\kappa \Gamma$ rather than extensive in $N$ , this excess is visible in $\mathcal { I }$ only once the radial fluctuations are averaged out at fixed position, as pointed out in $\mathrm { A p p . ~ G . 1 }$