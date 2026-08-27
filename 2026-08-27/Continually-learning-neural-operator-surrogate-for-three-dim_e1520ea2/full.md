# Continually learning neural-operator surrogate for three-dimensional airborne electromagnetic Bayesian inversion

Jaehong Chung<sup>1</sup>, Andrew Lockwood<sup>2</sup>, and Jef Caers<sup>1</sup>

<sup>1</sup>Mineral-X, Department of Earth and Planetary Sciences, Stanford University, USA <sup>2</sup>Xcalibur Smart Mapping, Perth, Australia

## Abstract

Three-dimensional probabilistic inversion of time-domain airborne electromagnetic (AEM) data is limited by the cost of the forward solve. Even though one simulation takes only tens of seconds, a Bayesian inversion of a survey of millions of soundings requires of order $1 0 ^ { 1 \dot { 0 } }$ forward evaluations. To address this, we develop a continually learning neural-operator surrogate of the three-dimensional AEM forward operator that replaces the solver inside the Bayesian inversion. We start from the point of view that regardless of what geological prior is specified, Maxwell’s laws remain invariant. Secondly, we avoid the limitation of learning on a single prior by continual learning on consecutive priors, which means our surrogate becomes richer as it is applied in future case studies, either by the authors, or by the scientific community. We use a validity check built on ensemble disagreement to divert cases with measurements outside the training range to the solver. Driven by the surrogate, the identical Markov chain Monte Carlo sampler reproduces the full-solver posterior, and its credible intervals cover the truth within 2.6 percentage points. Applied to the 2013 Capricorn TEMPEST survey in Western Australia, the surrogate inverts over two million soundings in seconds, a computation infeasible for the solver. Testing the geological prior against the entire survey costs minutes. The framework delivers uncertaintyquantified conductivity imaging at survey scale, which we believe is essential to perform near real-time mineral-systems targeting with geophysics.

## Highlights

• One neural-operator surrogate accumulates geological priors, so the airborne electromagnetic surveys it can invert grow with every prior learned, and accuracy on the earlier priors improves.

• The surrogate reproduces the full-solver posterior with calibrated credible intervals.

• The surrogate inverts the two million soundings of the Capricorn TEMPEST survey in seconds, against about 26,300 years for the solver.

Keywords: airborne electromagnetics; Bayesian inversion; neural operator; continual learning; uncertainty quantification; mineral exploration

## 1 Introduction

Airborne electromagnetic (AEM) surveys sense the electrical conductivity of the subsurface and are a primary tool for critical-mineral exploration, groundwater assessment, and structural mapping (Nabighian, 1979; Auken et al., 2015; Yang and Oldenburg, 2012). The standard product of such a survey is a single conductivity model obtained by deterministic inversion. Many conductivity models, however, fit the same data within noise, because the electromagnetic inverse problem is non-unique (Tarantola and Valette, 1982; Tarantola, 2005). Exploration decisions based on a single model do not address the cost risks associated with follow-up surveys or drilling.

A Bayesian framework quantifies this non-uniqueness by sampling. It returns a posterior distribution of conductivity models consistent with the data and the prior, and the spread of that posterior is the uncertainty (Mosegaard and Tarantola, 1995; Sambridge and Mosegaard, 2002). Its computational cost in three dimensions has kept it from routine use, for two reasons. One is the dimensionality of a voxelized model space. This can be addressed by parameterizing the conductivity field as a Gaussian process, which encodes spatial continuity in a small set of hyperparameters (Kitanidis, 1996; Wang et al., 2022), or any other set of lower-dimensional projections. The other is the number of forward solves the sampler requires. A three-dimensional electromagnetic solve takes tens of seconds, a sampler requires thousands of solves per sounding, and an airborne survey may have hundreds of thousands to millions of soundings. Replacing the solver with a forward surrogate reduces this cost by orders of magnitude while keeping an explicit likelihood, so accepted samples remain conditioned on the observations (Moghadas et al., 2020; Yang et al., 2024). A network can instead be trained to carry the data to the model directly, which removes the sampler as well as the solver (Wu et al., 2023). The posterior it returns is subject to the specific prior it is trained on. We take the other route and surrogate only the forward map, so the sampler and the likelihood are unchanged. A forward surrogate has recently reached three-dimensional ground-loop data (Liu et al., 2026) and survey-long sequences of two-dimensional lines (Caers et al., 2026).

A limitation common to these forward surrogates is that the surrogate itself is static and priorspecific. It is trained once, on models drawn from a single geostatistical prior, and it is accurate only within that prior’s support. But surveys do not need to remain within one prior. Geology changes along the flight lines, and repeated surveys extend the data in time as well as in space. When a sounding falls outside the training support, the surrogate still returns a prediction. The failure in extending the single prior is most severe for compact, strong conductors of economic interest, which are the structures a smooth training ensemble contains least. Training a separate surrogate for each new region does not settle the problem. Each new surrogate requires its own solver-labeled data, and a growing collection of single-prior models is not the same as a single surrogate that accumulates multiple priors in a single neural model. This distribution-shift problem is identified as an open direction in Liu et al. (2026).

We address the shift by learning the forward operator continually. The physics governing the electromagnetic response, Maxwell’s equations, remains invariant regardless of the prior. In machine learning terms a new prior is a covariate shift. Continual learning methods built for conflicting tasks protect the network’s weights from change (Kirkpatrick et al., 2017). Between priors nothing conflicts, so it is enough to retain a small memory of models from the earlier priors and to require each update to reproduce the previous predictions on them, by experience replay and function-space distillation (Hinton et al., 2015; Li and Hoiem, 2017; Lopez-Paz and Ranzato, 2017). One operator then accumulates priors rather than exchanging one for another, and an accuracy test accepts an update only if no earlier prior regresses.

The key contributions of this study are the following: (1) a neural operator that learns the threedimensional airborne electromagnetic forward map across a sequence of geostatistical priors without forgetting, (2) a validity check that flags soundings outside the training support and routes them to the full solver, and (3) a demonstration that the surrogate reproduces the solver posterior on real data from Western Australia. The operator then inverts a full airborne survey, a computation infeasible for the traditional solver.

## 2 AEM survey and geological priors

The inversion targets an airborne electromagnetic survey in Western Australia. One geological prior is built for the survey area, and four more are built for regions with diferent conductivity structures. The idea is to test the continual learning by a sequential assimilation of these various priors.

## 2.1 The Capricorn survey, Western Australia

The field data are from the 2013 TEMPEST survey over the Turee Creek area, Western Australia, released as the Capricorn regional survey. It holds 2,155,272 soundings on 191 flight lines at 5 km line spacing (Figure 1(a)). The fixed-wing system transmits at 25 Hz and records the vertical magnetic field in fifteen time gates from 0.013 to 16.2 ms. The nominal transmitter clearance is 120 m, and the receiver trails 117 m behind and 41.5 m below the transmitter. Those three numbers define the survey geometry the surrogate reads at inference. The released channel is corrected to the nominal clearance rather than to the recorded per-sounding value.

The processing report provides the survey’s additive noise level per gate, from 0.0267 fT at the first gate to 0.0070 fT at gate 14, with a median of 0.0117 fT over the fifteen gates (Geoscience Australia, 2020). Line 1005801 carries the single-line results below and grounds the prior built for this area (Figure 1(b,c)).

The geological prior for this area is built from the regional geology. The stratigraphy is a threelayer sequence. Lateritic regolith overlies Turee Creek Group siliciclastics, which overlie folded, magnetite-bearing banded iron formation of the Hamersley Group, 2.6 to 2.4 Ga, deformed by the Ophthalmia Orogeny (Thorne and Tyler, 1996; Krapez, 1996). The regolith is deeply weathered and therefore conductive (Anand and Paine, 2002), while fresh banded iron formation is resistive (Dentith and Mudge, 2014). That contrast sets the vertical conductivity structure the prior encodes. Figure 2(a) shows the sequence as a conceptual section, and Table 1 gives the hyperparameter ranges the prior is sampled over.

## 2.2 Comparison regions and their geological priors

One region comes with one geological prior, so the Capricorn survey alone cannot test whether one operator works across geology. We therefore build four more geological priors. The Seward Peninsula, Alaska, is a resistive crystalline host carrying compact graphite and sulfide conductors (Emond et al., 2024). Denmark holds layered glacial sediments incised by buried tunnel valleys (Barfod et al., 2016; Møller et al., 2009). Zeeland, the Netherlands, is a coastal section with a sharp fresh over saline interface (Revil et al., 2017). Northeastern Wisconsin holds conductive glacial cover over resistive bedrock (Minsley et al., 2022). Figure 2(b-e) shows the conceptual geology of each, beside the Capricorn section on the same scale. The priors are built from this geological information alone, and no survey data from these regions are used. Table 1 lists the hyperparameter ranges the four priors are sampled over. Figure 3 shows two realizations of each prior.

## 3 Methodology

We introduce the methodology to build a continually learning surrogate of the three-dimensional forward solver and the Bayesian inversion that uses it. The surrogate learns the map from a conductivity model to its airborne decay, $\sigma \mapsto \mathbf { d }$ , over a sequence of geological priors. At inference a validity check decides whether a sounding lies inside the range the surrogate was trained on and passes it to the full solver when it does not. The surrogate then replaces the solver inside a Markov chain Monte Carlo sampler based on the probability perturbation method (PPM), which recovers the posterior distribution of conductivity models for each sounding. We use PPM with a Metropolis sampler because it is agnostic to the forward model, so the identical sampler runs with the solver or the surrogate. Figure 4 summarizes the framework and each subsection provides the details.

![](images/45a3d9a81f5ec91d40bd5c4726ac8fe4ee45e6a853dc389a5a7d45c83a1e8b9a.jpg)

![](images/611b015a1f635c3f7f7eef1f5c5af6f02422532bad7ab04a63be525e83bfacfc.jpg)

![](images/a0d18a3f7ea00565e0a31b929fb25077ee60a72f526eb4ecc3a808594f5586c2.jpg)  
Figure 1: The Capricorn TEMPEST survey. (a) The 191 flight lines, colored by the gate time at which each sounding’s decay reaches the noise level. Line 1005801 is marked in black, and the inset locates the survey in Australia. (b) Decay section of line 1005801, $\log _ { 1 0 } | B _ { z } |$ (fT) over the fifteen gates along the 178 km line. Dashed vertical lines mark the three soundings of panel (c). (c) Decays of those three soundings, spanning slow (conductive), median, and fast (resistive) responses. The dashed gray curve is the noise model, and gates below it are shaded and drawn open.

Table 1: Gaussian process hyperparameters of the five geological priors. Each entry is a uniform distribution $U [ a , b ]$ sampled per realization. σ is the unit conductivity in $\log _ { 1 0 } \mathrm { S / m }$ , h is a layer thickness, d is a top or interface depth, and w is a body width, all in meters. $\ell _ { h }$ and $\ell _ { v }$ are the horizontal and vertical correlation lengths in meters, α is the horizontal-to-vertical anisotropy factor, and ν is the Matérn smoothness, where a larger ν gives a smoother field. (The Zeeland source is a frequency-domain survey, and only its conductivity structure is used.)
<table><tr><td>Prior</td><td>Type</td><td>Variable</td><td> $U [ a , b ]$ </td><td> $\ell _ { h } \ \mathrm { ( m ) }$ </td><td> $\ell _ { v } \mathrm { ~ } ( \mathrm { m } )$ </td><td>α</td><td>ν</td><td>Source</td></tr><tr><td rowspan="4">Capricorn</td><td rowspan="2">Conductivity</td><td>σregolith</td><td>[−2.5, 0]</td><td rowspan="4"></td><td rowspan="4"></td><td rowspan="4"></td><td rowspan="4"></td><td rowspan="4">Thorne and Tyler (1996)</td></tr><tr><td> $\sigma _ { \mathrm { s e d i m e n t } }$ </td><td>[−3.0, −0.1]</td></tr><tr><td></td><td>σbasement</td><td>[−4.0, −1.5] [300,900] [120, 300] [1.0, 2.5] [2, 5]</td></tr><tr><td>Geometry</td><td> $h _ { \mathrm { r e g o l i t h } }$ </td><td>[0, 120]</td></tr><tr><td rowspan="4"></td><td rowspan="2">Conductivity</td><td> $d _ { \mathrm { b a s e m e n t } }$ </td><td>[40, 400]</td><td rowspan="4">[300, 900] [120, 300] [1.0, 3.0] [1.5, 4] Emond et al. (2024)</td><td rowspan="4"></td><td rowspan="4"></td><td rowspan="4"></td><td rowspan="4"></td></tr><tr><td> $\sigma _ { \mathrm { h o s t } }$ </td><td>[−3.3, −2.6] [−0.5, 0.5]</td></tr><tr><td rowspan="2">Geometry</td><td> $\sigma _ { \mathrm { b o d y } }$ </td><td></td></tr><tr><td> $w _ { \mathrm { b o d y } }$ </td><td>[160, 480] [80, 340]</td></tr><tr><td rowspan="4">Denmark</td><td rowspan="2"></td><td> $d _ { \mathrm { b o d y } }$ </td><td>[−1.7, -1.3]</td><td rowspan="4">[400,1200] [70,150] [2,5] [1.5,4]</td><td rowspan="4"></td><td rowspan="4"></td><td rowspan="4"></td><td rowspan="4">Barfod et al. (2016) Møller et al. (2009)</td></tr><tr><td> $\sigma _ { \mathrm { t o p } }$ </td><td>[−2.6, -2.0]</td></tr><tr><td rowspan="2">Conductivity</td><td> $\sigma _ { \mathrm { b a s e } }$ </td><td>[−1.5, −1.1]</td></tr><tr><td> $\sigma _ { \mathrm { v a l l e y } }$ </td><td></td></tr><tr><td rowspan="2">Zeeland</td><td rowspan="2">Geometry Conductivity</td><td> $d _ { \mathrm { v a l l e y } }$ </td><td>[90, 240]</td><td rowspan="2">[700, 2500] [150, 320] [1.0, 2.0] [2, 5] Revil et al. (2017)</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td> $\sigma _ { \mathrm { f r e s h } }$ </td><td>[−2.3, -1.5] [−0.6, 0.1]</td></tr><tr><td rowspan="2"></td><td rowspan="2">Geometry</td><td> $\sigma _ { \mathrm { s a l i n e } }$   $d _ { \mathrm { i n t e r f a c e } }$ </td><td></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td><td>[90, 340]</td></tr><tr><td rowspan="2">Wisconsin</td><td rowspan="2">Conductivity</td><td> $\sigma _ { \mathrm { c o v e r } }$   $\sigma _ { \mathrm { b e d r o c k } }$ </td><td>[−1.6, −1.1]</td><td rowspan="2"></td><td rowspan="2">[−3.0, −2.3] [400, 1500] [90, 190] [1.5, 3.5] [1.5, 4] Minsley et al. (2022)</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td> $d _ { \mathrm { b e d r o c k } }$ </td><td></td></tr><tr><td></td><td>Geometry</td><td></td><td>[150, 360]</td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/236cda53462b9c3577e25e9e45cee6f709d8d8ccb1b83ed1036145995d56fdc6.jpg)  
Figure 2: Conceptual geology of the five regions, on one shared $\log _ { 1 0 } \sigma$ scale. (a) Capricorn, (b) Seward Peninsula, (c) Denmark, (d) Zeeland, and (e) northeastern Wisconsin.

## 3.1 Governing equations for forward simulation

Airborne measurements record a difusive process. When the transmitter current is switched of, the collapsing primary field induces eddy currents in the subsurface, and the current system difuses downward and outward while it decays (Nabighian, 1979). The rate of decay provides the subsurface conductivity information. A conductive subsurface sustains the current system and the measured field decays slowly, whereas a resistive subsurface dissipates it within the earliest gates. The receiver records the vertical magnetic field of this decay at fifteen gate times.

The governing equations are Maxwell’s equations in the quasi-static limit. At the time scales and subsurface conductivities of airborne surveys, conduction currents dominate displacement currents, so Faraday’s law and Ampère’s law reduce to

$$
\nabla \times \mathbf { E } ( \mathbf { x } , t ) = - \frac { \partial \mathbf { B } ( \mathbf { x } , t ) } { \partial t } , \qquad \nabla \times \mathbf { B } ( \mathbf { x } , t ) = \mu _ { 0 } \big ( \sigma ( \mathbf { x } ) \mathbf { E } ( \mathbf { x } , t ) + \mathbf { J } _ { s } \big ) ,\tag{1}
$$

where E and B are the electric and magnetic fields, σ is the electrical conductivity, $\mu _ { 0 }$ is the vacuum permeability, and ${ \bf J } _ { s }$ is the source current of the transmitter loop. Taking the curl of Faraday’s law

3-D volume

![](images/b6c9134bc876bce5c76cc830878b7617a1ae7974d63e6c8bf9134d0eb20885d8.jpg)

Cut-away sections

![](images/b26e08476e5c04671e64f3e2d3e04ec8f69671e97fcb1d18755e11fd189aa12f.jpg)

3-D volume

![](images/aebe6b1216fbfb53c870aad034f6aac127e345107995e97d1df84b503b1672a5.jpg)

Cut-away sections

![](images/046857545da2606c33c46e0febcea748d417ba1d3a89df2b79d4e4cca7f19fe3.jpg)

![](images/41af63ff2c14dccde778ec8beb491438cc35a160c8d9d60f403048a27dceff91.jpg)

![](images/603188ab4ff239054cf2a6ebf482b2ab613e00f873ebf8229006248368a64e08.jpg)

![](images/aed60de7f73cedae889f4d4ebb3f06d018e5261eeaef1f0780143af333a2e2e1.jpg)

![](images/816a884488e627c421d69d75b8cadb203a7970967ad7a76df74c9fb039e7eb71.jpg)

![](images/a195501a36b555d3efe5dd39456fcb0f21ed5f74a33d73fb285412281e992803.jpg)

![](images/db53a91e837b157d9959a9996b035f0ca20426fe46e3175c65d19ee4516e7dfe.jpg)

![](images/c47c8ce9d8f9f3cadb7df20ec917055ee5ebf911f975e18bf713229cb31bdf5a.jpg)

(S/m) 0.0 -0.5 -1.0 -1.5 -2.0 -2.5 -3.0 -3.5 -4.0

![](images/a5c2355d2c4788554897565213aa910a5c40143389986738589b0197d4930bb7.jpg)

![](images/1f2995627fbfda32269762b676047c250e77e28db7f4461823ee9e30174d0ce8.jpg)

![](images/b820a7a5f6144558e9ef70a3f2a0ce68252ca524fdbf743b1aec909d0e7ecd4e.jpg)

![](images/b7010ca2cea1f3b2ef65164f3086e9ddb3efa32618ed5f8ce71dd995c768484e.jpg)

![](images/56fc9b3b3b47ca9d7c213214145c719bee979cd4f02080ec7cfecff0870588b6.jpg)

![](images/ad71e9ac80e00a026b3bebae1b07ed5e50ac760c4f1a93992456137e1b94f749.jpg)

![](images/85153e8f11acddfdc647232713958b7595ebc2d1ac1189cc29e52c3933d7464f.jpg)

![](images/ba0c91934a89f0657fe0e6f6ebb8ad12175d7878629a649c05b53d69b030ed1a.jpg)

![](images/2e8dcc15a37379193a1e2a402660109bcaf84f554a6249048d2f929df61fc19b.jpg)  
Figure 3: Two realizations of each geological prior, sampled from a Gaussian process. Each row is one prior, and each realization is shown as the three-dimensional volume and a cut-away along a horizontal and a vertical mid-plane.

and substituting Ampère’s law eliminates B and gives the difusion equation for the electric field,

$$
\nabla \times \nabla \times \mathbf { E } + \mu _ { 0 } \sigma ( \mathbf { x } ) { \frac { \partial \mathbf { E } } { \partial t } } = - \mu _ { 0 } { \frac { \partial \mathbf { J } _ { s } } { \partial t } } ,\tag{2}
$$

solved for a step-of source, with the pre-shutof steady state as the initial condition and the fields vanishing at infinity. The magnetic field follows from Faraday’s law,

$$
\mathbf { B } = - \int \nabla \times \mathbf { E } d t ,\tag{3}
$$

and is sampled at the receiver location and gate times.

The instrument does not record a step-of response. The TEMPEST system transmits a 50% duty-cycle square wave at 25 Hz, and the released data are transformed to the response of a 100% duty-cycle waveform at the same base frequency. Because the governing equation is linear in the fields, the duty-cycle response is the alternating-sign superposition of step-of responses across successive polarity reversals,

$$
d ( t ) = \sum _ { k = 0 } ^ { K } ( - 1 ) ^ { k } s \bigl ( t + k T _ { 1 / 2 } \bigr ) , \qquad T _ { 1 / 2 } = 2 0 \mathrm { m s } ,\tag{4}
$$

where $d ( t )$ is the recorded response, $s ( \cdot )$ is the step-of response, $T _ { 1 / 2 }$ is the half-period, and K is the number of half-cycles retained. The sum converges within a few half-cycles.

The response of one sounding is laterally local. The induced current system contributes measurably to the data only within a bounded lateral zone, the footprint of the sounding, of order one kilometer for this system and depth range. Each sounding is therefore modeled on a local crop of the survey volume, $3 2 \times 3 2 \times 2 4$ voxels at $4 0 \times 4 0 \times 2 5$ m. The simulation is a three-dimensional finite-volume solve on an OcTree mesh in SimPEG (Cockett et al., 2015).

## 3.2 Neural operator surrogate for forward simulation

The surrogate approximates the solution operator of Equations 2 and 3, which maps a conductivity field to the decay it produces,

$$
F : \sigma ( \mathbf { x } ) \mapsto \mathbf { d } = \big ( B _ { z } ( t _ { 1 } ) , \dots , B _ { z } ( t _ { 1 5 } ) \big ) .\tag{5}
$$

The operator is defined on functions rather than fixed vectors. The network inherits this structure, taking the conductivity as an input field and outputting the response at any gate time and geometry in the training range. The architecture below is therefore a neural operator rather than a regression network, trained on solver-labeled pairs $( \sigma , \mathbf { d } )$ . The network has three parts, each with a specific purpose. A model encoder compresses the conductivity field into coeficients. A prior embedding tells the network which geological prior the field was drawn from. A query network reads the gate time and the survey geometry and builds the basis the coeficients weight. We will see that this allows changing the survey geometry itself when applying the surrogate model. Figure $^ \mathrm { 4 ( a , b ) }$ shows the three parts on one forward pass, with the tensors each stage actually produces, and the following paragraphs describe each component in detail.

First, the model encoder is a three-dimensional Fourier Neural Operator (Li et al., 2020; Kovachki et al., 2023). The input is the conductivity field on the $3 2 \times 3 2 \times 2 4$ crop, concatenated with three normalized coordinate channels, and a pointwise linear layer lifts this four-channel field to a Cchannel feature field $v _ { 0 }$ . Each of the L layers then filters the field in the frequency domain, keeps the eight lowest modes per axis, and adds a pointwise map,

$$
v _ { l + 1 } = \phi \bigl ( W _ { l } v _ { l } + \mathcal { F } ^ { - 1 } ( R \odot \mathcal { F } v _ { l } ) \bigr ) , \qquad l = 0 , \ldots , L - 1 ,\tag{6}
$$

where $v _ { l }$ is the C-channel feature field at layer $l , \mathcal { F }$ is the Fourier transform over the three spatial axes, R holds the learnable weights that mix the channels of each retained low mode, ⊙ denotes the mode-wise multiplication, $W _ { l }$ is a pointwise linear map, and $\phi$ is a GELU nonlinearity. Averaging the final field $v _ { L }$ over the voxel grid Ω of the crop gives the encoder feature,

$$
h = \frac { 1 } { | \Omega | } \sum _ { { \bf x } \in \Omega } v _ { L } ( { \bf x } ) \ \in \mathbb { R } ^ { C } ,\tag{7}
$$

where $| \Omega | = 3 2 \times 3 2 \times 2 4$ is the number of voxels.

Second, the prior embedding is a compact code $z ,$ a fingerprint of the prior, computed from a set of example realizations by a permutation-invariant encoder (DeepSets). The code modulates the encoder feature by a rescaling and a shift (feature-wise linear modulation, FiLM), and a two-layer perceptron turns the modulated feature into P coeficients,

$$
z = \rho \Big ( \frac { 1 } { E } \sum _ { e } \psi ( \varphi _ { e } ) \Big ) , \qquad h ^ { \prime } = \gamma ( z ) \odot h + \beta ( z ) , \qquad \mathbf { b } = W _ { 2 } \phi \big ( W _ { 1 } h ^ { \prime } \big ) \ \in \mathbb { R } ^ { P } ,\tag{8}
$$

where $\varphi _ { e }$ is a six-number statistic computed from each of the $E$ example realizations, holding the mean, the standard deviation, the horizontal and vertical correlation lengths, the conductive fraction, and the vertical contrast of the prior, $\psi$ and $\rho$ are the encoder networks, $\gamma , \beta$ are the modulation scale and shift, and $W _ { 1 } , W _ { 2 }$ are the weights of the projection network. The coeficients b $\mathbf { \Psi } = \left( b _ { 1 } , \ldots , b _ { P } \right)$ depend on the conductivity field through h and on the prior through z, written $b _ { p } ( \sigma ; z )$ below. Because the governing physics is consistent regardless of the prior, the embedding is not meant to change the operator the network approximates. It lets a network of fixed size allocate its capacity to the part of model space each prior occupies.

Third, the query network is a small perceptron τ that reads the query $q = [ \log _ { 1 0 } t ,$ g] of gate time t and the three-component survey geometry $\mathbf { g } ,$ the transmitter clearance and the in-line and vertical receiver ofsets, and returns the basis,

$$
\pmb { \tau } ( q ) = \bigl ( \tau _ { 1 } ( q ) , \dots , \tau _ { P } ( q ) \bigr ) \ \in \mathbb { R } ^ { P } .\tag{9}
$$

The transmitter clearance changes from sounding to sounding as the aircraft follows the terrain, and each sounding is inverted at its own recorded geometry rather than at one nominal height.

The three parts assemble into the prediction, the DeepONet form of Lu et al. (2021), with the model encoder as its branch and the query network as its trunk, optionally on top of an analytic layered-earth baseline,

$$
\hat { y } ( \boldsymbol { q } ) = \log _ { 1 0 } \left| \hat { d } ( \boldsymbol { q } ) \right| = \log _ { 1 0 } \left| d _ { \mathrm { 1 D } } ( \boldsymbol { \bar { \sigma } } ; \boldsymbol { q } ) \right| + \sum _ { p = 1 } ^ { P } b _ { p } ( \boldsymbol { \sigma } ; \boldsymbol { z } ) \tau _ { p } ( \boldsymbol { q } ) + b _ { 0 } ,\tag{10}
$$

where $d _ { \mathrm { 1 D } }$ is a fast one-dimensional layered-earth forward of the footprint-averaged column $\bar { \sigma } ( z )$ evaluated at the query $q ,$ and $b _ { 0 }$ is a learned bias. The baseline carries positivity and the late-time behavior, and the branch-trunk sum supplies the three-dimensional correction. It is used in the constrained controlled-study checkpoint; the production and field checkpoints deployed here set it to zero and predict $\log _ { 1 0 } | \hat { d } |$ directly. The coeficients $b _ { p }$ are independent of the query, so they are computed once per model and reused across all gate times and geometries.

## 3.3 Loss function with continual learning and physical constraints

The inversion aims to recover the posterior $p ( \sigma \mid \mathbf { d } ) \propto p ( \mathbf { d } \mid \sigma ) p ( \sigma )$ . Across a sequence of geostatistical priors, the likelihood $p ( \mathbf { d } \mid \sigma )$ is fixed by the physics of Section 3.1 and only the prior $p ( \sigma )$ changes, so a new prior widens the conductivity fields the surrogate covers without changing the map it learns. The goal of training is one operator that stays accurate across geologically distinct priors.

First, the base loss term fits the surrogate to solver labels on the current prior $\pi _ { s } .$ , where s indexes the stage in the sequence,

$$
\mathcal { L } _ { \mathrm { d a t a } } ^ { \pi _ { s } } ( \theta ) = \frac { 1 } { N n _ { g } } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { n _ { g } } \left( \hat { y } _ { i , t } - y _ { i , t } \right) ^ { 2 } ,\tag{11}
$$

where $y _ { i , t }$ is the solver label of training model i at gate $t , \ \hat { y } _ { i , t }$ is the prediction of Equation 10 for the same model and gate, N is the number of training models, and $n _ { g } = 1 5$ is the number of gates.

Second, when the survey reaches a new geological region, training on the new prior alone overwrites what the surrogate learned on the earlier ones. Experience replay prevents the overwrite, a strategy inspired by the brain’s replay of past experience during learning (Rolnick et al., 2019). A memory R keeps a fixed number M of models per earlier prior, drawn without replacement, and the replay term evaluates the data loss of Equation 11 on the memory,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e p l a y } } ( \theta ) = \mathcal { L } _ { \mathrm { d a t a } } ^ { \mathcal { R } } ( \theta ) . } \end{array}\tag{12}
$$

The memory grows only with the number of priors seen, by M models per prior. A second term on the same memory requires the update to reproduce the previous checkpoint’s own predictions there, not only the solver labels, which is function-space distillation,

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } ( \theta , \theta _ { s - 1 } ) = \frac { 1 } { \lvert \mathcal { R } \rvert n _ { g } } \sum _ { i \in \mathcal { R } } \sum _ { t = 1 } ^ { n _ { g } } \big ( \hat { y } _ { i , t } ( \theta ) - \hat { y } _ { i , t } ( \theta _ { s - 1 } ) \big ) ^ { 2 } ,\tag{13}
$$

where $\theta _ { s - 1 }$ is the checkpoint the stage started from, held fixed.

Third, two physical consistency terms encode domain knowledge in the loss. A difusive step-of field decays monotonically and its decay curve is concave on a log-log plot. The first term ${ \mathcal { L } } _ { \mathrm { m o n o } }$ penalizes a predicted decay that rises with time, and the second term $\mathcal { L } _ { \mathrm { c v x } }$ penalizes a predicted decay that curves upward in log-log,

$$
\mathcal { L } _ { \mathrm { m o n o } } = \overline { { \left( \Delta _ { t } \hat { y } \right) _ { + } ^ { 2 } } } , \qquad \mathcal { L } _ { \mathrm { c v x } } = \overline { { \left( \Delta _ { u } ^ { 2 } \hat { y } \right) _ { + } ^ { 2 } } } ,\tag{14}
$$

where $\Delta _ { t } \hat { y }$ is the first diference of the predicted decay over gate time, $\Delta _ { u } ^ { 2 } \hat { y }$ is its second diference over $u = \log _ { 1 0 } t , ( x ) _ { + } = \operatorname* { m a x } ( 0 , x )$ is the positive part, and the overbar is the mean over gates and models. The penalty applies to the predicted decay of the training data the network is trained on. That training data is the duty-cycle one of Equation 4, whose labels need not be monotone, so the two terms act as a soft bias rather than as a property the labels already satisfy.

Thus, the loss at stage s assembles the terms,

$$
\begin{array} { r } { \mathcal { L } _ { s } = \mathcal { L } _ { \mathrm { d a t a } } ^ { \pi _ { s } } ( \theta ) + \lambda _ { r } \mathcal { L } _ { \mathrm { r e p l a y } } ( \theta ) + \lambda _ { d } \mathcal { L } _ { \mathrm { d i s t i l l } } ( \theta , \theta _ { s - 1 } ) + \lambda _ { c } \big ( \mathcal { L } _ { \mathrm { m o n o } } + \kappa \mathcal { L } _ { \mathrm { c v x } } \big ) , } \end{array}\tag{15}
$$

where $\lambda _ { r }$ and $\lambda _ { d }$ weight the two rehearsal terms and $\lambda _ { c }$ the physical constraints. The rehearsal terms carry unit weight, $\lambda _ { r } = \lambda _ { d } = 1$ , with one replay minibatch drawn per earlier prior at every step; the constraint terms are used in the single-stage controlled training, at $\lambda _ { c } = 0 . 1$ and $\kappa = 0 . 3$ and no one run combines all four terms. Each stage starts from the previous checkpoint $\theta _ { s - 1 }$ and minimizes ${ \mathcal { L } } _ { s } ,$ , so the operator is updated rather than retrained.

An accuracy gate closes each stage. Minimizing $\mathcal { L } _ { s }$ yields a candidate $\theta _ { s }$ , and the candidate is evaluated on the frozen test set $\mathcal { T } _ { s ^ { \prime } }$ of every earlier prior, data unseen in any training stage. With $\varepsilon ( \theta ; \mathcal { T } )$ the relative $L _ { 2 }$ error of checkpoint θ on test set $\tau .$ , the per-sounding $| | \hat { \mathbf { d } } - \mathbf { d } | | _ { 2 } / | | \mathbf { d } | | _ { 2 }$ of the reconstructed decay averaged over the set, the candidate is accepted only if no earlier prior has regressed by more than a tolerance $\delta ,$

$$
\mathrm { a c c e p t } \ \theta _ { s } \ \Longleftrightarrow \varepsilon \big ( \theta _ { s } ; \ \mathcal { T } _ { s ^ { \prime } } \big ) \leq \operatorname* { m a x } \big \{ \big ( 1 + \delta \big ) \varepsilon _ { s ^ { \prime } } ^ { \mathrm { m i n } } , \ \varepsilon _ { s ^ { \prime } } ^ { \mathrm { m i n } } + \varepsilon _ { \mathrm { a b s } } \big \} \quad \mathrm { f o r ~ e v e r y ~ } s ^ { \prime } < s ,\tag{16}
$$

where $\varepsilon _ { s ^ { \prime } } ^ { \mathrm { { m i n } } }$ is the lowest error any accepted checkpoint has reached on $\mathcal { T } _ { s ^ { \prime } } , ~ \delta = 0 . 2 5$ , and $\varepsilon _ { \mathrm { a b s } }$ is one percentage point, so that a regression is rejected only when it is both relatively and absolutely large; on the deployed baselines the relative branch binds. A rejected candidate rolls back to $\theta _ { s }$ −1 and is retried once with the memory M and the distillation weight $\lambda _ { d }$ both doubled. If the retry also regresses, the deployed surrogate is unchanged.

## 3.4 Validity check at inference

The surrogate returns a decay for any input, including a conductivity field far from every training prior. A validity check warns of the extrapolation, per sounding, at inference time.

The check is built on a deep ensemble (Lakshminarayanan et al., 2017), $M _ { e } = 5$ copies of the surrogate trained from diferent random initializations on the same data. Inside the training support the data pull all members to the same decay. Outside the support each member extrapolates on its own and the predictions spread. The disagreement score measures that spread, the standard deviation across members averaged over the gates,

$$
u ( \sigma ) = \frac { 1 } { n _ { g } } \sum _ { t = 1 } ^ { n _ { g } } \sqrt { \frac { 1 } { M _ { e } } \sum _ { m = 1 } ^ { M _ { e } } \left( \hat { y } _ { t } ^ { ( m ) } - \bar { y } _ { t } \right) ^ { 2 } } , \qquad \bar { y } _ { t } = \frac { 1 } { M _ { e } } \sum _ { m = 1 } ^ { M _ { e } } \hat { y } _ { t } ^ { ( m ) } ,\tag{17}
$$

where $\hat { y } _ { t } ^ { ( m ) }$ is the prediction of member m at gate t. The field check takes the unbiased $1 / ( M _ { e } - 1 )$ in place of $1 / M _ { e }$ and averages over the gates the signal-to-noise screen retains; each threshold is calibrated and applied under its own convention, so the two are not on a comparable scale. At survey scale, where a sounding is inverted by ranking a shared prior ensemble, its score is the median of u over the draws the ranking accepts.

A threshold turns the score into a decision. The threshold $u 9 5$ is the 95th percentile of u on validation data of the training priors. A sounding with $u \leq u _ { 9 5 }$ is inverted with the surrogate, and a sounding with $u > u _ { 9 5 }$ is routed to the full solver of Section 3.1.

## 3.5 Surrogate accelerated Bayesian inversion

Bayesian inversion treats the conductivity field as a random quantity and recovers the posterior, $p ( \sigma \mid { \bf d } _ { \mathrm { o b s } } ) \propto p ( { \bf d } _ { \mathrm { o b s } } \mid \sigma ) p ( \sigma )$ , given the observed decay $\mathbf { d } _ { \mathrm { { o b s } } }$ of a sounding. The prior $p ( \sigma )$ is the geostatistical prior, and the likelihood $p ( \mathbf { d } _ { \mathrm { o b s } } \mid \sigma )$ scores how probable the observed decay is when the subsurface is $\sigma ,$ by simulating the decay of σ and comparing it with the observation. The posterior has no closed form, so it is explored by Markov chain Monte Carlo (MCMC). The chain starts from a geological prior realization and repeats one step, propose a perturbed conductivity field, simulate its decay, and accept or reject the proposal by comparing its data fit with the current field, as laid out in Figure 4 (c). After enough iterations, the accepted fields are distributed by the posterior (Metropolis et al., 1953; Hastings, 1970; Mosegaard and Tarantola, 1995). One step needs three ingredients, a likelihood that scores the fit, a proposal that generates the perturbed field, and an acceptance rule. The following paragraphs define each in turn.

The likelihood is set by the measurement noise. Each observed data point is treated as the true decay plus a Gaussian error whose standard deviation is a relative term with an additive floor, the form in standard use for AEM (Green and Lane, 2003),

$$
\varsigma _ { t , n } = \rho \left| \boldsymbol d _ { t , n } ^ { \mathrm { o b s } } \right| + \varsigma _ { 0 } , \qquad \rho = 0 . 0 8 , \quad \varsigma _ { 0 } = 0 . 0 1 \ \mathrm { f T } ,\tag{18}
$$

where $d _ { t , n } ^ { \mathrm { o b s } }$ is the observed data at gate t of sounding $n , \ p$ is the relative error, and $\varsigma _ { 0 }$ is the additive floor. The same model applies to the controlled and the field inversions, and it defines the signal-to-noise screen that drops gates whose median amplitude falls below the noise.

Under this Gaussian noise, the negative log of the likelihood is, up to a constant, a sum of squared residuals scaled by $\varsigma _ { t , n }$ . The data misfit the chain evaluates is that sum with one additional dynamic-range weight,

$$
\Phi ( \sigma ) = \frac { 1 } { n _ { g } } \sum _ { t = 1 } ^ { n _ { g } } \omega _ { t } ^ { 2 } \frac { 1 } { N _ { d } } \sum _ { n = 1 } ^ { N _ { d } } \left( \frac { \hat { d } _ { t , n } - d _ { t , n } ^ { \mathrm { o b s } } } { \varsigma _ { t , n } } \right) ^ { 2 } , \qquad \omega _ { t } = \big ( \operatorname* { m a x } _ { n } d _ { t , n } ^ { \mathrm { o b s } } - \operatorname* { m i n } _ { n } d _ { t , n } ^ { \mathrm { o b s } } \big ) ^ { - 1 } ,\tag{19}
$$

where $\hat { d } _ { t , n }$ is the predicted data of Equation 10, from the surrogate in the accelerated inversions and from the solver in the reference runs, and $N _ { d }$ is the number of soundings inverted together. The weight $\omega _ { t }$ is the reciprocal dynamic range of gate t across those soundings, so it down-weights the gates whose amplitude varies most across the set. Every inversion reported here is single-sounding, $N _ { d } = 1$ , for which $\omega _ { t } \equiv 1$ and Φ reduces to the gate-averaged $\chi ^ { 2 }$ . A lower $\Phi ( \sigma )$ means a more probable field under the likelihood.

The proposal generates the perturbed field by the PPM (Caers, 2003; Caers and Hofman, 2006). A prior realization is generated from underlying uniform variables, and the sampler perturbs the uniforms rather than the field itself. Each uniform $w _ { 0 }$ is moved toward an independent draw $w _ { 1 }$ with radius r,

$$
w _ { r } = \mathrm { P P M } ( w _ { 0 } , r , w _ { 1 } ) = \left\{ \begin{array} { l l } { w _ { 0 } , } & { r w _ { 0 } < w _ { 1 } < 1 - r + r w _ { 0 } , } \\ { w _ { 1 } / r , } & { w _ { 1 } \leq r w _ { 0 } , } \\ { ( w _ { 1 } + r - 1 ) / r , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{20}
$$

where r controls the step size, a small r leaving most uniforms unchanged. The perturbed $w _ { r }$ stays marginally uniform for any r, so every proposal remains a prior realization, making the method prioragnostic. The prior $p ( \sigma )$ enters the chain through this construction, and the kernel is symmetric, with no proposal-ratio correction.

The acceptance rule decides between the proposed and the current field. A proposal is accepted with the tempered Metropolis probability α = min{1, exp(− $( \Phi _ { r } - \Phi _ { i } ) / T ) \}$ , where $\Phi _ { r }$ and $\Phi _ { i }$ are the proposed and current misfits and $T$ is the sampling temperature. A better fit is always accepted, and a worse fit is accepted with a probability that falls with the misfit increase, which lets the chain explore the posterior rather than descend to a single best model. A rejected large step of radius $r _ { 1 }$ triggers a smaller delayed-rejection step of radius $r _ { 2 }$ with the detailed-balance correction of Green and Mira (2001). The radius $r _ { 2 }$ is annealed over the iterations and the temperature is adapted by

Robbins-Monro toward the acceptance rate $\alpha ^ { * } = 0 . 2 3$ (Robbins and Monro, 1951; Roberts et al., 1997). The sampler is forward agnostic. The identical algorithm runs with the surrogate or with the solver, which is the basis of the equivalence test of Section 4.3.

![](images/a129e9b5d4697733b32905610229d626578857a9f47e343d403173a270e63434.jpg)

(b) Prior embedding, query network and output  
![](images/345feaf7c469164b607779076ed9a58d237c874fee685638b1670a78386c7456.jpg)

(c) Bayesian inversion by PPM-MCMC  
![](images/7bf27b4bf3283690c5ff7f88580497a59479bcbed0ea807531cbc2d6dfc4734c.jpg)  
Figure 4: The neural operator surrogate and the sampler it drives. (a) The model encoder, with the eight lowest modes boxed. (b) The prior embedding, the query network and the output. The layered-earth baseline is dashed because the deployed checkpoints set it to zero. (c) The sampler. The gray fill marks the same operator, trained in (a) and (b) and called in (c).

## 4 Results

## 4.1 Surrogate accuracy and validity

We evaluate the surrogate by comparison with the forward solver, since the surrogate aims to reproduce the solver up to its numerical accuracy. Figure 5(a) compares the surrogate and the solver on the Capricorn test data, the 125 soundings not used in training, with $R ^ { 2 } = 0 . 9 9 2$ on $\log _ { 1 0 } | B _ { z } |$ and a median gate error of 4.7% for the initialization shown. Over ten initializations the error is $5 . 0 \pm 0 . 3 \%$ . Figure 5(b) shows the median sounding, where the surrogate follows the solver across the full decay.

The accuracy is limited by the amount of training data rather than by the network capacity. Figure 6 shows the test error against the training data size on Capricorn, from 60 to 1,000 training soundings, ten runs at each size, scored on the same 125 test soundings. The error falls as a power law, $\varepsilon \propto n ^ { - 0 . 4 6 }$ in the number of training soundings n, and is still falling at the largest size we reach.

The surrogate error is small against the measurement noise the inversion already carries. Each gate is observed with the uncertainty of Equation 18, and a surrogate error below that level keeps the misfit unchanged. In those units the error has a median of 0.28 standard deviations, and the last gate, whose relative error is largest, sits at 0.04 of a standard deviation because the 0.01 fT floor supplies 98.8% of its uncertainty.

Once we have trained the surrogate on the Capricorn prior, exploration moves to geology that may be distinctly diferent, and nothing guarantees the accuracy carries over. We therefore score the trained operator, without further training, on the test data of the four other regions. Figure 7(a) orders the regions by how unlike the Capricorn training data they are. Denmark and Wisconsin are close in accuracy compared to Capricorn’s own test soundings, 6.6% and 5.4% versus 5.2% on Capricorn itself. Zeeland is well outside and the operator reaches 11.8%. Seward is the exception at 67.7%. The warning of this error comes from the disagreement score of Equation 17, the spread among the five ensemble members that share the training data and difer only in initialization. Inside the training range the data pull the members to the same decay, and outside it each member extrapolates on its own. Figure 7(b) gives the fraction of each region the check sends to the solver, with the threshold at the 95th percentile of the score on the Capricorn validation data. The check warns on 32.8% of Seward and 16.8% of Zeeland, against 0.8% of Denmark and Wisconsin and 4.0% of the Capricorn test data, so it only flags those regions with poor prediction, and does not waste compute on cases that are already accurate.

(a)  
![](images/193572c5c20059e9bc4c498f9418067780d2fe70bcddb3148c2f49d32d1a6d25.jpg)

(b)  
![](images/3714026ff7bca3dc487ac21b2b2461b7b8ec9edf4f272349f9960e9fbb13a618.jpg)  
Figure 5: Forward accuracy on the Capricorn test data. (a) Surrogate against forward solver at the fifteen gates, with the 1:1 line. (b) The median sounding, surrogate dashed and forward solver solid. The shaded strip marks gates whose median decay falls below the noise model.

![](images/123b191cc529ce9949391f19ef8873534664ca73b1d823e7d1c65a7feae351fd.jpg)  
Figure 6: Test error against training set size on Capricorn. Markers and bars are the mean and standard deviation of ten runs at each size, scored on the same 125 test soundings. The line is a power law fitted to all fifty runs.

(a)  
![](images/5deff0af6c5691aae726a86f56158fc01c549b8c3305e2c9b410fc33a0d4471c.jpg)

(b)  
![](images/3acf7e13d285d82a3c9b1b7544cd9fbae53305aacc14a903707b75a1fb1588b2.jpg)  
Figure 7: The validity check on the four regions the operator has not learned, trained on the Capricorn prior alone. (a) The disagreement score of Equation 17 against dissimilarity, the median distance from a test model to the nearest Capricorn training model in a standardized 64-number lateral profile. Markers are the median over soundings and bars run from the tenth to the ninetieth percentile. The dashed line is the threshold $u _ { 9 5 } ,$ the 95th percentile of the score on the Capricorn validation data. (b) The fraction of each region’s test soundings above the threshold, which are inverted with the forward solver.

## 4.2 Continual learning across priors

We train one surrogate on the five geological priors, sequentially, as would be the case when flying many surveys around the world. Here, we assume the Capricorn survey is flown first. The risk of a sequence is that the region already learned deteriorates. We therefore compare our surrogate with fine-tuning, both trained on the same data in the same order. Fine-tuning trains on each new region’s training data alone. Our surrogate trains on the new region together with replay of stored models from the earlier priors and a distillation term that keeps the update close to the previous checkpoint’s predictions. Every error below is the median gate error on the 125 test soundings of the region named, averaged over five initializations.

Figure 8 follows each region from the stage that taught it to the end of the sequence. Under fine-tuning the error on a region rises once the operator moves on, by 7.5 percentage points averaged over the four regions with a later stage. Capricorn goes from 5.2% to 9.6%, Seward from 12.1% to 29.1%, and Zeeland from 3.9% to 11.9%. Under our surrogate the same average is −0.6 percentage points, negative in every initialization. Each region ends the sequence more accurate than when the operator learned it. Capricorn has 5.2% error as the only region the operator knows and 4.0% once the other four are in, Seward 12.1% against 11.5%, and Denmark 3.5% against 2.8%. A new prior supplies conductivity fields the earlier ones did not contain, and the operator fits the same forward map over more of its domain.

Neither result depends on the order of the regions. We repeat the sequence in four arrival orders, the deployed one, its reverse, the hardest region first, and the easiest first, with three initializations for each new order, and Figure 8 gives all four. Our surrogate ends at 4.4, 4.9, 4.4 and 4.5%, and its forgetting is negative in every order. Fine-tuning ends at 11.6, 18.9, 15.4 and 25.9%, with forgetting from 7.5 to 23.6 percentage points. Retraining on all five regions at once reaches 4.6%, against 4.4% for the sequence.

## 4.3 Equivalence of the surrogate and solver posteriors

We test whether replacing the forward solver with the surrogate changes the posterior. The same Bayesian inversion runs twice on the same measured sounding, once with the solver computing every forward response and once with the surrogate, under the Capricorn prior and the operator of the continual sequence. The first comparison scores one pool of prior realizations twice. A pool of 400 realizations is forwarded at the recorded geometry, each realization is scored against the observed decay, and the best-fitting 10% form the posterior, so only the forward operator difers between the two runs. Over fifteen soundings of line 1005801 the two posteriors share 85.5% of the accepted realizations, at a data-fit residual of 11.3% under the surrogate against 12.2% under the solver, and per depth they difer by 0.45 of the distance two random halves of the same pool difer by. The surrogate moves the posterior less than the sampling does.

The second comparison involves the full PPM-MCMC sampler. The identical sampler runs with each operator at the same iteration budget and the same random seeds, thirty-two chains of 500 iterations on one sounding of line 1005801, 8,000 retained states apiece. Figure 9 shows the two posteriors. The uncertainty range the surrogate returns is 96% as wide as the solver’s, the two ranges share 85% of their combined extent at the median depth, and the final acceptance is 0.30 against 0.28. The surrogate-driven and solver-driven posteriors difer by 10.3 times what a random split of the pooled states gives, and two independent halves of the solver run difer from each other by 12.2. Neither engine is fully mixed at this budget, with a Gelman-Rubin statistic of 1.39 for the solver and 1.47 for the surrogate, so the comparison bounds the surrogate’s efect. The surrogate ran the experiment in 2,718 s against 1,138,000 s for the solver, forty-five minutes against thirteen

![](images/b067350ba1442c71b911e19861fb44998a018c0802dbd1ec1883cbd10038335d.jpg)  
Figure 8: Continual learning with and without replay of the earlier priors, across the five geological priors and four arrival orders. Each row is one order, the order our surrogate was trained in first, and each column is a position in the sequence, with the region named in its panel. Arrows run from the stage that taught the region to the end of the sequence.

days.

The surrogate posterior is calibrated, its uncertainty neither too narrow nor too wide. To test this, we take 400 realizations of the Capricorn prior the operator has not been trained on, treat each in turn as the truth, invert its stored solver response with the surrogate under the acceptance rule the survey inversion uses, and count how often the truth falls inside the posterior credible intervals. A 50% interval contains the truth 50.6% of the time, an 80% interval 79.3%, a 90% interval 88.7% and a 95% interval 92.4%, so the largest departure from nominal over the four levels is 2.6 percentage points and the smallest is 0.6.

![](images/d671cb1ec1febc5b478d0b13be073f46fd8719b7fe0f2b9a425f4440ff9dfa97.jpg)  
Figure 9: Sampler-level equivalence on the Capricorn prior. The identical PPM-MCMC sampler runs with the forward solver and with the surrogate, at the same budget and seeds, on one measured sounding of line 1005801. Bands are the 10–90% posterior range of the volume-mean resistivity, lines the medians, pooled over thirty-two chains per operator. The quoted 96% is the median over depth of the surrogate range divided by the solver range.

## 4.4 Survey-scale inversion and prior updating

We now scale from one sounding to the survey, the 2,155,272 soundings of the Capricorn TEMPEST data. Bayesian inversion at this scale is infeasible for the solver, whose forward evaluations alone would take tens of thousands of years. With the surrogate the inversion costs seconds, and testing the geological prior against the survey costs minutes. We first test the prior against every flown line, then invert the survey under it, and revise the prior where the test fails.

The falsification test asks whether the observed soundings look like members of the prior. The prior’s forward responses define a 95% envelope in data space, the 95th percentile of their robust Mahalanobis distance in the principal-component subspace carrying 95% of their variance, and a line is falsified when fewer than 90% of its soundings fall inside it. The Capricorn prior contains

92% of its grounding line 1005801 and 94% of a neighboring line it never saw. Figure 10(a) extends the test to the 190 unseen lines. The prior explains 127 of them at a median coverage of 97.9%, so a prior grounded on one line holds over most of a survey 500 km across, and the 63 lines it does not explain form a block in the west and north-west, named before any inversion is run.

The surrogate then inverts the survey under this prior. Each sounding scores a shared ensemble of 10,000 prior realizations forwarded at its own recorded geometry, and the 100 best-fitting realizations form its posterior. The 2,131,667 soundings whose transmitter clearance does not exceed the 180 m ceiling of the trained range are inverted in 24.9 s on one GPU, at 11.7 µs per sounding, against about 26,300 years for the solver at 38.8 s per forward solve. The validity check flags no sounding inside the trained 90 to 180 m clearance band, so the surrogate inverts the entire survey without a fall-back to the solver. Resistivity rises with depth at 84% of soundings, from a median of 63 Ω·m at 38 m to 340 Ω·m at 238 m, the conductive weathered regolith (Anand and Paine, 2002) over the resistive banded iron formation (Dentith and Mudge, 2014), and the north-western block is the most resistive at depth of the whole survey, 929 Ω·m against 264 Ω·m elsewhere.

Figure 11 maps the posterior-median resistivity and the width of the posterior. The resistivity is fixed to within a factor of 2.1 at 38 m and 5.0 at 238 m.

Line 1015101 shows the clearest failure, with the prior containing 59% of its soundings, and we run one turn of the prior revision on it. Coverage alone would not settle the comparison, because widening any prior raises coverage, so the control is the deployed prior widened by the same amount without moving its center. The control grows the model space from 7.90 to 10.53 decades and fixes 31 of the 63 falsified lines while losing 19 the deployed prior already explained. A two-component mixture that correlates the layers, on the same layer ranges, fixes 57 and loses none, and the survey increases from 127 to 184 of 190 lines in 8.65 decades (Table 2, Figure 10(b)). The gain comes from the layer correlation and not from width. Folding the new prior into the operator repeats the continual step on measured data. Its error on the new prior falls from 28.1% to 8.1%, the accuracy on the earlier region is unchanged, and the posterior on the original line does not move. One flight line and one retraining revise the prior for a survey 500 km across.

Table 2: The four priors compared. Model space is the width of the prior in decades of log σ summed over layers, for the mixture the weighted mean of its two components. Lines explained and lines lost are counted over the 190 unseen flight lines.
<table><tr><td>prior</td><td>model space (decades)</td><td>lines explained (of 190)</td><td>lines lost</td></tr><tr><td>deployed Capricorn</td><td>7.90</td><td>127</td><td></td></tr><tr><td>width control</td><td>10.53</td><td>139</td><td>19</td></tr><tr><td>regional, independent layers</td><td>11.75</td><td>176</td><td>4</td></tr><tr><td>two-component mixture</td><td>8.65</td><td>184</td><td>0</td></tr></table>

## 5 Discussion

The continual learning result follows from the invariance of the operator. The physics does not change with the prior, so a new prior shifts the inputs the surrogate sees. Replay with distillation already contains every region so far learned. In fact, we find the accuracy on earlier priors improves as continual learning progresses. Across four arrival orders the operator ends within half a percentage point of itself, where fine-tuning spreads over fifteen. This closes the distribution-shift limitation left open by static surrogate frameworks (Liu et al., 2026).

The framework changes the cost structure of probabilistic inversion. A Bayesian product over two million soundings, infeasible for the solver at any budget, costs seconds, and testing a prior against an entire survey costs minutes. The prior stops being a fixed input chosen before the inversion and becomes a quantity the survey itself revises, and the falsification test tells where the next prior is needed at the cost of one training data set and one retraining.

(a)  
![](images/f05990f45ac43b880210e45d6ecd9083a372620271e673c8177025b12e4376af.jpg)

(b)  
![](images/162607cf4e2b310e4097726d19d95c58f728b5dc9a2a065c2c45aa84e361c2e1.jpg)  
Figure 10: The falsification test over the whole survey, before and after one turn of the prior revision. Each flown line is colored by its coverage, the fraction of its soundings falling inside the 95% envelope of the prior’s forward responses, and a line is falsified below the 90% level marked on the bar. (a) The deployed Capricorn prior, with the line the updated prior was built from marked in amber. (b) The updated prior.

Two diagnostics carry beyond the Capricorn application. The disagreement score is a detector of leaving the training support. As ensemble members become individually more accurate they become more similar, and the partial correlation of score with error at fixed clearance falls from +0.485 under an earlier, less accurate ensemble to −0.014, so the score degrades as the ensemble improves. The second diagnostic is a control for prior updating. Coverage rises whenever a prior is widened, so a coverage gain is evidence of a better prior only against the same prior widened by the same amount without re-centering. Here the widened control occupies the largest model space of the four priors tried and still explains fewer lines than the mixture, and it falsifies nineteen lines the deployed prior had explained.

## 6 Conclusion

We have developed a continually learning neural-operator surrogate of the three-dimensional airborne electromagnetic forward operator, with a validity check that scores each sounding against the support the operator was trained on.

The operator learns a sequence of geological priors without forgetting, and replay of stored models from the earlier priors does more than that. Every prior ends the sequence more accurate than when the operator learned it, because a new prior provides conductivity models the earlier ones did not contain and the operator fits the same forward map over more of its domain. That

Posterior-median resistivity (Ω⋅m)  
![](images/b1db901f657e0ff85edbf917d27fc322235478dc6df06c24f69c95a07add7b79.jpg)

How tightly the data constrain the resistivity (10–90% of the posterior)  
![](images/d508f1d28c6b539eaaed91982701388bdedaf2f176ebfe4b21e98b81edab4209.jpg)  
Figure 11: Whole-survey inversion of Capricorn TEMPEST. The top row is the posterior-median resistivity and the bottom row the width of the posterior, at 38 m $\left( \mathrm { a } , \mathrm { c } \right)$ and 238 m $( \mathrm { b } , \mathrm { d } )$ . The width is the factor in resistivity between the 10th and 90th percentiles, so a value of 2 fixes the resistivity to within a factor of two. All 2,131,667 inverted soundings are shown, plotted one in four for legibility. Panel (a) marks line 1005801, which grounds the prior, and lines 1015101 and 1013902, which it does not explain.

gain survives every arrival order we tested.

The surrogate reproduces the solver posterior. Driven by the identical sampler at matched settings, it departs from the solver by less than two independent solver-driven chains depart from each other, and its credible intervals cover the truth within 2.6 percentage points of nominal. The validity check routes out-of-range soundings to the solver.

The prior becomes a testable and revisable input rather than a fixed choice. A cheap forward ensemble tests the prior against every line of a survey before any inversion is run. A second prior built from a single failing line, folded into the same operator, raises the lines the survey explains from 127 to 184 of 190 and falsifies none the first prior already explained, so what the data ask for is the structure of the prior and not its width.

After the one-time cost of the training data, a sounding costs $1 1 . 7 \mu \mathrm { s } .$ , and inverting the survey’s two million soundings takes seconds against about 26,300 years for the solver. Probabilistic inversion, with calibrated uncertainty, is available at the scale of an airborne survey. The framework is the foundation for a joint three-dimensional magnetics and electromagnetic inversion developed separately.

## Data and code availability

The Capricorn TEMPEST survey data are publicly available from Geoscience Australia at https: //pid.geoscience.gov.au/dataset/ga/81642 (eCat record 81642), including the acquisition and processing report (Geoscience Australia, 2020). The sources of the geological priors are cited in Table 1. The code will be public on Github upon acceptance for publication.

## References

Ravi R Anand and Marko Paine. Regolith geology of the Yilgarn Craton, Western Australia: implications for exploration. Australian Journal of Earth Sciences, 49(1):3–162, 2002.

Esben Auken, Anders Vest Christiansen, Casper Kirkegaard, Gianluca Fiandaca, Cyril Schamper, Ahmad Ali Behroozmand, Andrew Binley, Emil Nielsen, Flemming Efersø, Niels Bøie Christensen, et al. An overview of a highly versatile forward and stable inverse algorithm for airborne, ground-based and borehole electromagnetic and electric data. Exploration Geophysics, 46(3): 223–235, 2015.

Adrian AS Barfod, Ingelise Møller, and Anders V Christiansen. Compiling a national resistivity atlas of Denmark based on airborne and ground-based transient electromagnetic data. Journal of Applied Geophysics, 134:199–209, 2016.

Jef Caers. History matching under training-image-based geological model constraints. SPE Journal, 8(03):218–226, 2003.

Jef Caers and Todd Hofman. The probability perturbation method: A new look at Bayesian inverse modeling: Caers and Hofman. Mathematical Geology, 38(1):81–100, 2006.

Jef Caers, Peng Li, Jonas Kloeckner, Juan Pablo Daza, Zhen Yin, and Celine Scheidt. Stochastic inversion of geophysical data by sequential Bayesian updating under a non-stationary Gaussian process prior. EarthArXiv preprint, 2026.

Rowan Cockett, Seogi Kang, Lindsey J Heagy, Adam Pidlisecky, and Douglas W Oldenburg. Sim-PEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications. Computers & Geosciences, 85:142–154, 2015.

Michael Dentith and Stephen T. Mudge. Geophysics for the Mineral Exploration Geoscientist. Cambridge University Press, Cambridge, 2014. ISBN 9780521809511. doi: 10.1017/ CBO9781139024358.

A. M. Emond, L. A. Fusso, and SkyTEM Canada Incorporated. Seward Peninsula airborne electromagnetic survey, Kigluaik, Bendeleben, and Darby mountains, Alaska. Geophysical Report GPR 2024-2, Alaska Division of Geological & Geophysical Surveys, 2024. URL https: //doi.org/10.14509/31303.

Geoscience Australia. AusAEM 02 WA/NT, 2019-20 Airborne Electromagnetic Survey: TEM-PEST(R) airborne electromagnetic data and conductivity estimates. Geoscience Australia Data Release, 2020. URL https://ecat.ga.gov.au/geonetwork/srv/api/records/ 8e598964-b4f3-4500-86ba-4c36d762f14e.

Andy Green and Richard Lane. Estimating noise levels in AEM data. ASEG Extended Abstracts, 2003(2):1–5, 2003.

Peter J Green and Antonietta Mira. Delayed rejection in reversible jump Metropolis–Hastings. Biometrika, 88(4):1035–1053, 2001.

W Keith Hastings. Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 57(1):97–109, 1970.

Geofrey Hinton, Oriol Vinyals, and Jef Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526, 2017.

Peter K Kitanidis. On the geostatistical approach to the inverse problem. Advances in Water Resources, 19(6):333–342, 1996.

Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research, 24(89):1–97, 2023.

B Krapez. Sequence stratigraphic concepts applied to the identification of basin-filling rhythms in Precambrian successions. Australian Journal of Earth Sciences, 43(4):355–380, 1996.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems, 30, 2017.

Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(12):2935–2947, 2017.

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial diferential equations. arXiv preprint arXiv:2010.08895, 2020.

Zhuo Liu, Zhen Yin, Jonas Kloeckner, Jack Muir, and Jef Caers. Probability-perturbation McMC for 3D Bayesian TDEM inversion with unknown covariance parameters. EarthArXiv preprint, 2026.

David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in Neural Information Processing Systems, 30, 2017.

Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218–229, 2021.

Nicholas Metropolis, Arianna W Rosenbluth, Marshall N Rosenbluth, Augusta H Teller, and Edward Teller. Equation of state calculations by fast computing machines. The Journal of Chemical Physics, 21(6):1087–1092, 1953.

Burke J Minsley, Ben B Bloss, David J Hart, William Fitzpatrick, Maureen A Muldoon, Esther K Stewart, Randall J Hunt, Stephanie R James, N Leon Foks, and Matthew J Komiskey. Airborne electromagnetic and magnetic survey data, northeast Wisconsin (ver. 1.1, June 2022). US Geological Survey (USGS) Data Release, page 37, 2022.

Davood Moghadas, Ahmad A Behroozmand, and Anders Vest Christiansen. Soil electrical conductivity imaging using a neural network-based forward solver: Applied to large-scale Bayesian electromagnetic inversion. Journal of Applied Geophysics, 176:104012, 2020.

Ingelise Møller, Verner H Søndergaard, Flemming Jørgensen, Esben Auken, and Anders V Christiansen. Integrated management and utilization of hydrogeophysical data on a national scale. Near Surface Geophysics, 7(5-6):647–659, 2009.

Klaus Mosegaard and Albert Tarantola. Monte Carlo sampling of solutions to inverse problems. Journal of Geophysical Research: Solid Earth, 100(B7):12431–12447, 1995.

Misac N Nabighian. Quasi-static transient response of a conducting half-space—An approximate representation. Geophysics, 44(10):1700–1705, 1979.

André Revil, A Coperey, Z Shao, N Florsch, Ida Lykke Fabricius, Y Deng, JR Delsman, PS Pauw, M Karaoulis, PGB De Louw, et al. Complex conductivity of soils. Water Resources Research, 53 (8):7121–7147, 2017.

Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics, pages 400–407, 1951.

Gareth O Roberts, Andrew Gelman, and Walter R Gilks. Weak convergence and optimal scaling of random walk Metropolis algorithms. The Annals of Applied Probability, 7(1):110–120, 1997.

David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, and Gregory Wayne. Experience replay for continual learning. Advances in neural information processing systems, 32, 2019.

Malcolm Sambridge and Klaus Mosegaard. Monte Carlo methods in geophysical inverse problems. Reviews of Geophysics, 40(3):3–1, 2002.

Albert Tarantola. Inverse Problem Theory and Methods for Model Parameter Estimation. SIAM, 2005.

Albert Tarantola and Bernard Valette. Generalized nonlinear inverse problems solved using the least squares criterion. Reviews of Geophysics, 20(2):219–232, 1982.

AM Thorne and IM Tyler. Geology of the Rocklea 1: 100,000 sheet. Geological Survey of Western Australia, 1996.

Lijing Wang, Peter K Kitanidis, and Jef Caers. Hierarchical Bayesian inversion of global variables and large-scale spatial fields. Water Resources Research, 58(5):e2021WR031610, 2022.

Sihong Wu, Qinghua Huang, and Li Zhao. Fast Bayesian inversion of airborne electromagnetic data based on the invertible neural network. IEEE Transactions on Geoscience and Remote Sensing, 61:1–11, 2023.

Dikun Yang and Douglas W Oldenburg. Three-dimensional inversion of airborne time-domain electromagnetic data with applications to a porphyry deposit. Geophysics, 77(2):B23–B34, 2012.

Xu Yang, Yunhe Liu, Yang Su, Changchun Yin, Luyuan Wang, Yu Gao, Xiuyan Ren, and Bo Zhang. An eficient Bayesian inference for geo-electromagnetic data inversion based on surrogate modeling with adaptive sampling DNN. IEEE Transactions on Geoscience and Remote Sensing, 62:1–17, 2024.