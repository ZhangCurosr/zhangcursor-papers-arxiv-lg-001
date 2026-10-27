# Machine Learning Applications in the Solar Interior and Atmosphere

Vishal Upendran and Carlos J. D´ıaz Baso

Abstract The solar interior and atmosphere constitute a strongly coupled, multiscale dynamical system defined by continuous temporal evolution. Predicting the state of this system requires modeling physical interactions that span disparate spatial and temporal scales, constrained by heterogeneous and sparse data. This chapter formalizes the Sun within the mathematical framework of dynamical systems theory, defining a global state vector (<sup>⃗</sup>S) governed by thermodynamical, kinetic, and magnetic properties. Because solar observations represent projected, multi-modal, and partially observable measurements (O⃗ [⃗S]) of this hidden state, pure physicsbased deduction is fundamentally constrained. We present a unifying thesis: the primary role of machine learning in heliophysics is to act as a structured family of mathematical operators—specifically Inference (I), Emulation (E), Propagation (P), and Control (C)—that reconstruct, simulate, and couple this global state vector across disparate physical layers. By framing the vast, multi-disciplinary literature within this unified state-space taxonomy, we demonstrate how data-driven techniques bridge observational gaps to resolve hidden multiscale couplings and dramatically accelerate the simulation of non-linear processes, establishing a rigorous pathway to predict the global evolution of the Sun.

## 1.1 The Sun as a multiscale dynamical system

One of the nearest astrophysical objects of study available to mankind is our very own Sun. It is a star of spectral type G2V, with a radius of about $6 . 9 \times 1 0 ^ { 8 }$ meters and a mass of about $2 \times 1 0 ^ { 3 0 }$ kilograms, located at a distance of about $1 . 5 \times 1 0 ^ { 1 1 }$ meters from Earth as the source of almost all the energy that sustains life. It is a complex system with several layers, each with its own unique properties. Several phenomena occur all across the solar interior and atmosphere, and depending on the time scale of each phenomenon, they may be coupled to the global evolution of the Sun.

While we have a general understanding of the different layers of the Sun, the physical processes occurring in each layer are deeply intertwined. The solar interior consists of the core, the radiative and convective zones. Nuclear fusion in the core generates energy, which is then transported outwards through the other zones. Photons, our main source of information about the Sun, are generated in the core and then undergo a random walk through the radiative zone, taking about $1 0 ^ { 5 }$ years to reach the convective zone. In the convective zone, plasma motion becomes dominant, and energy is transported by convection. Depending on the density stratification, the photons become free streaming, and are able to escape the Sun, giving us a direct view of the solar surface, which we call the photosphere. The photosphere is the visible surface of the Sun, and it is where most of the solar radiation is emitted. Above the photosphere, the plasma density decreases, and we observe different layers of the solar atmosphere. Further higher up, the plasma becomes free streaming, and we observe the solar wind, which is a stream of charged particles that flows outwards from the Sun and fills the heliosphere.

The emission from the Sun is mostly a blackbody spectrum, but it exhibits a wide range of spectral lines corresponding to the transitions of ionization states of different elements. Each of these elements, and their ionization fractions depend on the specific layer of the solar atmosphere, and thus serve as diagnostics of the physical conditions in that layer. The solar atmosphere starts off from the photosphere, where the different elements are not quite fully ionized. With height, the plasma shows a mix of ions and neutrals in the chromosphere, transition region, and becomes almost fully ionized in the solar corona. Spectral observations of the different layers of the solar atmosphere suggest that the temperature in the solar atmosphere increases with height, reaching more than $1 0 ^ { 6 }$ K in the solar corona, which is a long-standing puzzle in solar physics called the Coronal Heating problem. However, this is not a simple local phenomenon – a complete understanding of the solar atmosphere requires understanding the coupling between the different layers. This coupling includes the solar interior setting the boundary conditions for the atmosphere, as well as the dynamic interactions between the atmospheric layers themselvesPriest (2014). A general graphic depicting the layers of the Sun is shown in Fig. 1.1.

The interior of the Sun is not directly observable through photons. While neutrino observations are potentially able to probe it, the more standard way of estimating interior properties is through helioseismology, which studies the propagation of acoustic waves. The acoustic waves are generated by turbulent convection in the convective zone, and they propagate through the solar interior, being refracted by the density stratification. By measuring the travel times of these waves at the solar surface, we can infer the properties of the solar interior, such as the sound speed, density, and flow velocities Charbonneau (2010).

![](images/461cd5c8f80b240dfac3a0e2c28d2e4fc0269ace2f7d9e6b67f23c8fffca4da8.jpg)  
Fig. 1.1: An illustration of different layers of the Sun. Art by Kelvinsong - Own work, CC BY-SA 3.0. Taken from Wikimedia commons.

Plasma motion from the interior also brings magnetic fields to the surface. The different structures seen in the solar atmosphere are strongly modulated by the magnetic field, which is thought to be generated internally through a dynamo process. The interaction between plasma, magnetic field, and radiation gives rise to the morphological structure and phenomenon observed in the solar atmosphere. These phenomena manifest across length, time, and energy scales spanning about 10 orders of magnitude. Crucially, they can be coupled across all of these different scales. Hence, efforts seek to understand the Sun in a multiscale manner, balancing detailed studies of specific phenomena with the global evolution of the star. Fig. 1.2 displays the set of different known phenomena in the solar interior and atmosphere, and their characteristic length and time scales. Each color scale corresponds to the kind of phenomena for which we mark the length and time scales. From Fig. 1.2, we see that the Sun exhibits phenomena ranging across ∼7 orders of magnitude in spatial extent, and ∼8 orders of magnitude in time. Not depicted in this figure is the range of energy scales measured, which can be upto ∼9 orders of magnitude Aschwanden (2005).

This multiscale nature motivates us to think of the Sun as a system of systems, where each subsystem is described by physical variables such as density, velocity, magnetic field, and thermodynamic state. We collect these variables into a state vector $( \vec { S } )$ , whose components are inferred from observations and serve as the foundation for our analytical operators $( \mathcal { S } , \mathcal { P } _ { t  t + A t } , \mathcal { P } _ { 1  2 } , \mathcal { E } ,$ $\mathcal { C } )$ . The coupling between subsystem state vectors gives rise to the phenomena observed across the solar interior and atmosphere.

Solar Phenomena Across Length and Time Scales  
![](images/780e0cea4b9394baddf1f714f88c95ec4c340b9c7bbaa3503bf2c092f7b5c837.jpg)  
Fig. 1.2: An illustration of the length and time scales of various phenomena observed in the Sun. The plotted ranges are compiled from the phenomenon–reference associations in the source table: granulation and supergranulation Rincon and Rieutord (2018); chromospheric network Priest (2014); sunspots Solanki (2003); light bridges Lagg et al. (2014); active regions Canfield (2000); Type I and Type II spicules Zaqarashvili and Erdelyi ´ (2009); nanojets, picoflares, and nanoflares Klimchuk (2015); Parker (1988); coronal jets Raouafi et al. (2016); campfires Berghmans et al. (2021); Ellerman bombs and IRIS bombs Ellerman (1917); Peter et al. (2014); microflares Hannah et al. (2011); coronal bright points Madjarska (2019); corona loops Reale (2014); solar flares Shibata and Magara (2011); CME source regions and global CMEs Chen (2011); prominences Parenti (2014); coronal holes and open flux regions Cranmer (2009); and the solar dynamo Charbonneau (2010).

The proximity of the Sun to Earth allows us to study it in much greater detail than any other star. Inherently, it is an excellent laboratory to study not just solar phenomena, but general astrophysical plasmas at great detail. As a consequence, we have a wealth of data available to study the Sun, ranging from remote sensing observations of the solar atmosphere and the heliosphere across the electromagnetic spectrum to in-situ measurements of the solar wind and energetic particles. This results in a very rich multi-messenger data set that allows us to probe the Sun at different scales, presenting strong observational constraints on the physical models, and hence allowing us to test unsolved problems at very high precision.

With the ever growing data volume in solar physics, it is becoming increasingly difficult to analyze and interpret the data using traditional methods. Thus, the field of solar physics is increasingly turning towards machine learning (ML) to analyze and interpret the data Camporeale et al. (2018); Bobra et al. (2021); Asensio Ramos et al. (2023); Illarionov (2025).

This chapter focuses on the remote-sensing of source phenomena in the solar interior and atmosphere, while the subsequent chapter addresses the heliosphere and in-situ processes.

## 1.2 State vectors and ML operators

The state-vector formalism gives a common language for organizing ML applications across the solar interior and atmosphere. The central object is not the algorithm, but the physical state on which the algorithm acts. A convolutional network, transformer, neural operator, diffusion model, or random forest has a different scientific meaning depending on its target. Its physical relevance changes whether it estimates subsurface flow, advances a magnetogram, reconstructs a coronal volume, or forecasts an eruptive event. We therefore organize the literature in a hierarchy whose first level is the state vector itself and whose second level is the ML operation applied to that state vector, defined below.

Let $\vec { S }$ denote the physical state vector of a solar subsystem. Typical variables may include Magneto hydrodynamic (MHD) variables, like density, velocity, temperature, magnetic field, or particle / species population in different levels, or even parameters like line-profile parameters, morphology, or reduced event variables.

Each state vector gives rise to an observable vector $\vec { O } \left( \mathrm { e . g . } \right.$ ., travel times, spectropolarimetric profiles, or intensity maps) measured by instruments, which is related to the physical state through a forward observation operator $\mathcal { O }$ such that ${ \vec { O } } = { \mathcal { O } } [ { \vec { S } } ]$ . Telescopes do not generally measure the precise state vector directly; instead, they measure projections, line-of-sight integrals, spectra, polarization states, image intensities, or helioseismic travel times. Hence, typically, the forward observation operator may be broken up into a combination of an observable synthesis $( \mathcal { O } _ { \mathrm { p h y s } } )$ and the telescope observation operator $( \mathcal { O } _ { \mathrm { i n s t r } } )$ , as $\mathcal { O } = \mathcal { O } _ { \mathrm { i n s t r } } [ \mathcal { O } _ { \mathrm { p h y s } } [ \cdot ] ]$

In this language, we define four operator classes $( \mathcal { I } , \mathcal { P } _ { t  t + \Delta t } , \mathcal { P } _ { 1  2 } , \mathcal { E } , \mathcal { C } )$ relating state vectors and observations across different systems:

$$
\mathcal { I } : \vec { O } = \mathcal { O } [ \vec { S } ]  \hat { \vec { S } } , \quad \mathcal { P } _ { a  b } : \vec { S } _ { a }  \vec { S } _ { b } ,\tag{1.1}
$$

$$
\boldsymbol { \mathcal { E } } : ( \theta , \vec { S } _ { 0 } )  \vec { S } _ { \mathrm { s i m } } , \quad \boldsymbol { \mathcal { C } } : ( \vec { S } , \boldsymbol { \mathcal { J } } )  a .\tag{1.2}
$$

The four operator classes cover the full scope of the reviewed work, and are defined as:

1. Inference $( { \mathcal { S } } ) { \mathrm { : } }$ Given observations ${ \vec { O } } = { \mathcal { O } } [ { \vec { S } } ]$ , recover an estimate of the physical state $( \hat { \vec { S } } )$ , by approximating the inverse of a known physical forward operator (such as radiative transfer, helioseismic kernels, or instrument point-spread functions). This is the machine-learning analogue to classical physical inversions. It includes point estimates, full posteriors via simulation-based inference, and discrete event labeling.

2. Propagation $( { \mathcal { P } } _ { a  b } ) \colon$ Map one state description to another, either with or without a physical forward observational operator connecting them. This class has two natural sub-cases that dominate the literature:

• Prediction $( \mathcal { P } _ { t  t + \Delta t } ) ;$ : Temporal propagation of a given state vector over time, $\vec { S } ( t )  \vec { S } ( t + \Delta t )$ , mapping directly to standard forecasting.

• Coupling $( { \mathcal { P } } _ { 1  2 } ) \cdot$ : Cross-domain coupling and translation, $\vec { S } _ { 1 } ( t )  \vec { S } _ { 2 } ( t +$ $\Delta t )$ , mapping states across different domains, such as translating EUV coronal images into photospheric magnetograms, or cross-calibrating different telescopes.

3. Emulation (E): Replace a slow physics-based solver (such as a multi-dimensional MHD or non-LTE radiative transfer code) with a learned surrogate model that takes model parameters or initial conditions as input and returns the simulation output in milliseconds. This includes deterministic surrogates and generative models that sample plausible solar states from a learned distribution $p ( \vec { S } )$

4. Control & Optimization $( \mathcal { C } ) \colon$ Optimal action selection given the current solar state and a scientific objective (e.g., adaptive optics corrections, automated telescope scheduling, or active-learning target selection).

We note that the key criterion separating $\mathcal { I }$ from $\mathcal { P } _ { 1  2 }$ is whether a physical forward operator $\mathcal { O }$ exists such that ${ \vec { O } } = { \mathcal { O } } ( { \vec { S } } )$ . If such an operator exists, the ML model approximates $\mathcal { O } ^ { - 1 }$ and the task is ${ \mathcal { I } } ;$ ; otherwise no, the learned mapping encodes inter-subsystem coupling learned from data and the task is $\mathcal { P } _ { 1  2 }$ . Even so, the application under study may consider several of the defined operators at once, and might not present one clear operator in the work. This overlap is not a failure of the taxonomy; rather, it reflects the reality that the Sun is a highly coupled dynamical system.

The literature surveyed here clusters into four physical families, each corresponding to a distinct region or problem type: the solar interior (§1.3); the solar atmospheric state spanning photosphere, chromosphere, transition region, and corona (§1.4); coherent structures and transient events (§1.5); observation-space operators covering image restoration, cross-domain translation, representation learning and observatory contro(§1.6). Each family is introduced in its physical context in the sections that follow. We finally present a discussion of the applications, and potential path forward in §1.7.

When discussing any ML application, we ask three questions:

1. What state vector is being inferred, propagated, emulated, or controlled?

2. What observations constrain the state vector under consideration?

3. What physical assumptions make the learned operator meaningful?

This framing keeps the review anchored in solar physics rather than the algorithms themselves. In the solar context, ML approximates operators that are otherwise expensive, ill-posed, or only partially specified. The most mature applications are those where the state vector and operator class are physically explicit. The least secure are those where the output is visually convincing but the implied state vector is poorly defined.

## 1.3 The solar interior

The solar interior is inaccessible to direct electromagnetic observation<sup>1</sup>. All information about its state must be deduced from surface signatures, which probe the convection zone of the Sun. These include acoustic travel times, mode frequencies, and large-scale photospheric magnetic patterns. This is done through forward models whose inversions are computationally expensive and non-unique. Machine learning enters this domain as a tool for constructing tractable approximations to these inverse $( \mathcal { I } )$ and coupling $( \mathcal { P } _ { 1  2 } )$ operators. The operational fidelity of these inference operators $( \mathcal { I } )$ remains constrained by the synthetic-to-real domain gap, because synthetic training data rarely captures the full stochastic noise of real telescope observations. It replaces costly pipelines with learned surrogates that can be evaluated rapidly at scale. We organize the interior literature into three interconnected families. The first is deep interior dynamics and physics emulation, targeting the hidden subsurface flow field (§. 1.3.1). The second covers latent magnetic states, targeting pre-emergence signatures and geometrically inaccessible far-side magnetic fields (§. 1.3.2). The third family focuses on global dynamo and solar cycle evolution to capture the long-term magnetic cycle of the internal solar dynamo (§. 1.3.3).

## 1.3.1 Deep interior dynamics andphysics emulation

The first family covers the fluid dynamics, kinematic flows, and thermodynamics of the solar interior convection zone. This framework aligns with major collaborative efforts, such as the COFFIES (Consequences of Fields and Flows in the Interior and Exterior of the Sun) framework Hoeksema et al. (2020) and the WHOLESUN project Brun et al. (2019). Both are large-scale, multi-institutional projects aimed at linking subsurface velocity fields with global magnetic structures across the entire Sun.

To model these processes, a representative state vector $\vec { S } _ { \mathrm { f l o w } }$ describes the subsurface velocity fields (either in global spherical components $\left( \nu _ { r } , \nu _ { \theta } , \nu _ { \phi } \right)$ or local Cartesian components $\left( \nu _ { x } , \nu _ { y } , \nu _ { z } \right) )$ , convective perturbations, and turbulent transport:

$$
\vec { S } _ { \mathrm { f l o w } } = \{ \nu _ { r } , \nu _ { \theta } , \nu _ { \phi } , \delta c , \ldots \} ,\tag{1.3}
$$

where $\nu _ { r } , \nu _ { \theta }$ , and $\nu _ { \phi }$ represent the radial, meridional, and zonal components of the large-scale velocity fields. Local convective and granular flows are often instead mapped to Cartesian velocities $\vec { \nu } = \left( \nu _ { x } , \nu _ { y } , \nu _ { z } \right)$ and horizontal components $\vec { \nu } _ { h } = \left( \nu _ { x } , \nu _ { y } \right)$ . The term $\delta c$ denotes perturbations in the local sound speed, acting as a proxy for temperature anomalies in the solar interior. Note that subgrid-scale quantities such as the Reynolds stress tensor $\tau _ { i j }$ and acoustic source events ε are not directly retrievable from surface observations; they are simulation-derived quantities used for training labels in the inference-based paradigm.

The state vector gives rise to different kinds of pressure and gravity waves, which leave periodic signatures in velocities / intensities measured on the solar photosphere. By modeling the photospheric signatures from $\vec { S } _ { \mathrm { f l o w } }$ , one is able to estimate the inverse mapping $( \mathcal { I } )$ to derive the state vector from the observations, where ML has played a significant role.

## 1.3.1.1 Inferring subsurface flows from acoustic diagnostics

Helioseismology is the primary observational pathway to the solar interior, converting surface Dopplergrams and line-of-sight velocities into wave-travel times and phase shifts, as reviewed in Kosovichev et al. (2025). Using these acoustic diagnostics to retrieve subsurface flows is historically a massive computational bottleneck. ML addresses this by learning direct approximations of the inverse operator:

$$
\begin{array} { r } { \mathcal { I } _ { \mathrm { f l o w } } : \vec { D } _ { \mathrm { a c o u s t i c } } = \mathcal { O } _ { \mathrm { a c o u s t i c } } [ \vec { S } _ { \mathrm { f l o w } } ] \to \hat { \vec { S } } _ { \mathrm { f l o w } } , } \end{array}\tag{1.4}
$$

where $\vec { D } _ { \mathrm { a c o u s t i c } }$ is the vector of observed acoustic diagnostics related to the flow state via the helioseismic observation operator $\mathcal { O } _ { \mathrm { a c o u s t i c } }$

Two distinct inference problems arise when extracting the solar interior state from these acoustic observations. The first concerns computational scalability. Classical ring-diagram inversions consume over 30 CPU hours per Carrington rotation, creating a bottleneck that prevents near-real-time analysis. Supervised networks trained on ring-diagram mode-fit flow parameters replace this step, yielding surface flow maps in milliseconds. Crucially, they faithfully preserve subtle large-scale flow features—such as equatorial Rossby waves—that aggressive spectral filtering would otherwise suppress Alshehhi et al. (2019).

The second problem concerns acoustic source localization. Weak source signatures are typically overwhelmed by the convective background and resonant p-mode power, making source detection difficult with standard filtering methods. By training a network on radiative-MHD simulations to recognize those patterns, one can recover acoustic emission sites in photospheric Doppler time series and the network’s learned filters can be reused directly as an interpretable operator Bahauddin and Rast (2021). A critical extension of this idea is the automated detection of sunquakes, capturing impulsive flare-related seismic signatures that are typically difficult to detect using conventional methods Mercea et al. (2023).

## 1.3.1.2 Simulation-backed kinematic reconstruction from intensity

While global models can resolve macro-scale structures, the true multi-scale velocity field within the convective boundary remains physically inaccessible to direct measurement. Here, ML functions strictly as an inference operator $( \mathcal { I } )$ to reconstruct these motions not from acoustic waves, but directly from surface intensity patterns:

$$
\mathcal { S } _ { \mathrm { k i n e } } : \vec { I } _ { \mathrm { s u r f } } = \mathcal { O } _ { \mathrm { r a d } } [ \vec { S } _ { \mathrm { f l o w } } ]  \hat { \vec { \nu } } ,\tag{1.5}
$$

where $\vec { I } _ { \mathrm { s u r f } }$ represents the emergent surface intensity map related to the flow state through the radiative transfer observation operator $\mathcal { O } _ { \mathrm { r a d } }$ , and $\hat { \vec { \nu } }$ represents either the two-dimensional horizontal velocity field $\vec { \nu } _ { h } ( x , y )$ at the photospheric surface boundary $( \tau _ { 5 0 0 0 \mathring \mathrm { A } } \approx 1 )$ , or its full three-dimensional depthdependent extension $\vec { \nu } ( x , y , z )$ into the upper convection zone.

Surface intensity patterns are a projection of this multi-scale velocity field. Granular and supergranular flows at different depths produce degenerate surface signatures that single-wavelength observations cannot disentangle. This inaccessibility is solved by the simulation-based inference paradigm: radiative-MHD simulations provide paired {surface observable, velocity field} training examples, making available the hidden-state labels that nature withholds.

The initial application of this paradigm focused on the surface boundary, where DeepVel Tremblay et al. (2018) demonstrated that Convolutional Neural Networks (CNNs)—which extract spatial features by sliding localized filters across an image, assuming that local patterns (like convective granules) look the same regardless of where they appear on the solar disk (translation invariance)—could recover transverse velocity fields ${ \vec { \nu } } _ { h }$ at granular scales from consecutive surface intensity maps. This successfully surpassed classical Local Correlation Tracking (LCT) and Fourier-based LCT (FLCT) methods by exploiting subtle intensity-morphology correlations that classical optical flow algorithms ignore. Subsequent work progressively extended this boundary mapping into the solar interior. By leveraging temporal and multi-height correlations, operators were trained to recover simultaneously granular and supergranular motions Tremblay and Attie (2020), flows within both the quiet Sun and magnetically complex active regions Tremblay et al. (2021); Lennard et al. (2025), and height-dependent kinematic signals across multiple photospheric levels Kostic et al.´ (2026). The same paradigm extends naturally below the surface: by pairing radiation-MHD simulations with synthetic surface observables, large-scale internal flows at 10 Mm depth can be inferred from just three snapshots separated by one hour, drastically reducing the required input data volume Masaki and Hotta (2024).

Taken together, these results show that the surface and subsurface kinematic state is recoverable from surface observables (up to a certian depth and resolution). The projection of the multi-scale velocity field onto surface observables is inherently degenerate, and this information loss cannot be recovered regardless of simulation quality. Closing the fidelity gap points naturally toward physics-informed approaches that enforce the governing equations directly as constraints, linking this inference framework to the atmospheric reconstruction methods discussed in Section 1.4.

## 1.3.2 Latent magnetic states: pre-emergence and the far-side

## 1.3.2.1 Subsurface signatures and pre-emergence flux detection

Flux tubes are plasma structures containing coherent and ordered magnetic field. The rise of magnetic flux tubes through the convection zone alters local thermodynamic properties and generates acoustic travel-time perturbations before the flux emerges at the surface. Detecting these subsurface structures is vital for operational space-weather forecasting, providing early warnings of active region (AR) births.

To formalize this, models target a reduced pre-emergence event state vector,

$$
\vec { z } _ { \mathrm { e m } } = \{ p _ { \mathrm { e m } } , x _ { \mathrm { e m } } , y _ { \mathrm { e m } } , t _ { \mathrm { e m } } , \phi _ { \mathrm { e m } } , \ldots \} ,\tag{1.6}
$$

where $p _ { \mathrm { e m } }$ denotes the probability of impending flux emergence, $( x _ { \mathrm { e m } } , y _ { \mathrm { e m } } )$ is the estimated emerging spatial coordinate, $t _ { \mathrm { e m } }$ is the lead time until detection, and $\Phi _ { \mathrm { e m } }$ represents the anticipated magnetic flux proxy.

Rather than executing computationally expensive 3D reconstructions of the magnetic field vector, ML approximates an inference operator. This operator maps photospheric surface observations $\vec { M } _ { \mathrm { s u r f } }$ (such as line-of-sight magnetograms or Dopplergrams) directly to estimated event parameters or continuous flux proxies in ${ \vec { z } } _ { \mathrm { e m } } ;$

$$
\mathcal { I } _ { \mathrm { e m } } : \vec { M } _ { \mathrm { s u r f } } = \mathcal { O } _ { \mathrm { p h o t o } } [ \vec { S } _ { \mathrm { m a g } } ]  \hat { \vec { z } } _ { \mathrm { e m } } ,\tag{1.7}
$$

where $\mathcal { O } _ { \mathrm { p h o t o } }$ is the photospheric observation operator mapping the subsurface magnetic/dynamic state $\vec { S } _ { \mathrm { m a g } }$ to the surface signatures. Depending on the specific architecture, these models retrieve either discrete components of $\vec { z } _ { \mathrm { e m } }$ (such as the binary probability of emergence $p _ { \mathrm { e m } }$ via patch classification) or spatial projections (such as 2D maps of predicted unsigned magnetic flux $\Phi _ { \mathrm { e m } } ( x , y ) ,$ ). Pre-emergence detection is fundamentally a weak-signal-in-stochastic-noise problem. A rising magnetic flux tube leaves coherent but faint imprints in surface observables hours before any flux breaches the surface. These subtle signatures are buried beneath the high-amplitude, chaotic convective background of granular and supergranular flows Cheung and Isobe (2014).

Instead of classical matched filters that require an analytically specified template, one can derive them directly from the data: by training on labeled preemergence and non-emergence line-of-sight magnetograms from SDO/HMI, the machine learning model learns discriminative patterns directly from the data rather than from theory Dhuri et al. (2020). The critical validation is not accuracy on a held-out test set but physical self-consistency. For instance, saliency maps of the trained classifier recover exactly the small-scale dipolar structures predicted by fluxtube rise theory, confirming that the network latches onto the physically causal signature and not onto a proxy correlation. A critical analysis shows that the model incorporates filters sensitive to small-scale, intense magnetic fields and pre-emergence bipolar structures consistent with flux-tube rise theory, confirming that the neural network responds to physically meaningful signatures.

Building on numerical simulations showing that subsurface structures alter acoustic power observed at the photosphere above them (Hartlep et al., 2011), modern sequence models—specifically Long Short-Term Memory (LSTM) networks, which are designed to selectively retain or discard information over long time sequences by routing data through gated memory cells—have been designed to predict active region emergence before any visible photospheric manifestation (Kasapis et al., 2025). Such models learn the mapping from multi-frequency acoustic power maps to changes in continuum intensity and unsigned magnetic flux. Hence, they are able to successfully forecast the pre-emergence phase-shift signatures and the subsequent localized decrease in continuum intensity that marks active region birth. A pre-emergence model that reports only a binary label makes a weaker scientific contribution than one whose learnedfeatures are consistent with physicalflux-tube signatures and whose outputs are calibrated probability-of-emergence estimates with uncertainty bounds.

By extension, the same approach can also probe physical dynamics beyond detection. Conditioning classifiers on supergranular flow configurations enables quantification of the impact of surface flows on emergence trajectories. These experiments find that quiet-Sun magnetic flux emergence is neither systematically influenced by nor affects the supergranular-scale flows in its vicinity Mani et al. (2024). This dual capability—prediction plus causal testing—underscores why model design is critical.

## 1.3.2.2 Far-side active region detection and reconstruction

Monitoring the Sun’s non-visible hemisphere is critical to build synchronic, full-Sun magnetic maps and to anticipate the rotation of active regions onto the Earthfacing disk Leka et al. (2026). In this sub-family, the primary state vector is $\vec { S } _ { \mathrm { f a r B } }$ representing the magnetic field configuration on the far-side:

$$
\vec { S } _ { \mathrm { f a r B } } = \{ B _ { r } , B _ { \theta } , B _ { \phi } , \varPhi _ { \mathrm { f a r } } , \ldots \} ,\tag{1.8}
$$

where $B _ { r } , B _ { \theta }$ , and $B _ { \phi }$ represent the radial, meridional, and zonal components of the magnetic field vector on the far-side, and $\Phi _ { \mathrm { f a r } }$ is a scalar activity proxy. In practice, due to the indirect nature of far-side diagnostics, models are often restricted to retrieving simplified projections such as binary active-region detection masks or the radial field component $B _ { r }$

Helioseismic holography can identify large far-side active regions by measuring acoustic phase shifts or travel-time perturbations from multiple-bounce waves (δ⃗τ). These observed seismic phase shifts are related to the far-side magnetic state via a forward helioseismic operator $ { \mathcal { O } } _ { \mathrm { s e i s m i c } }$ such that $\delta \vec { \tau } =$ $\mathcal { O } _ { \mathrm { s e i s m i c } } [ \vec { S } _ { \mathrm { f a r B } } ]$ . ML acts as an inference operator $( \mathcal { I } )$ to filter this noise and retrieve binary active-region states or magnetic proxies:

$$
\mathcal { S } _ { \mathrm { f a r B } } : \delta \vec { \tau } = \mathcal { O } _ { \mathrm { s e i s m i c } } [ \vec { S } _ { \mathrm { f a r B } } ]  \hat { \vec { S } } _ { \mathrm { f a r B } } .\tag{1.9}
$$

Far-side seismic maps suffer from extremely low signal-to-noise ratios because the acoustic waves must travel multiple times across the solar interior, making classical threshold-based detection insensitive to all but the largest active regions. Deep learning models substantially lower this detection floor by functioning as advanced spatial and temporal filters that isolate the weak coherent phase-shift signals from the stochastic acoustic background Felipe and Asensio Ramos (2019). The natural progression of this inference capability follows an ascending hierarchy of physical richness. First, a binary U-Net establishes the presence or absence of an active region, and is validated against simultaneous STEREO EUV observations Broock et al. (2021). Next, attention-enhanced convolutional LSTM (ConvLSTM) networks exploit temporal coherence across successive seismic maps to reduce false alarms Broock et al. (2022). Finally, regression networks replace the binary label with continuous synthetic magnetograms generated directly from helioseismic phase shifts Broock et al. (2024). This enriches the recovered state from a detection flag to a quantitative estimate of the far-side magnetic flux distribution.

Recent work has begun to close this loop by empirically calibrating the helioseismic phase shift to the unsigned magnetic field using far-side Solar Orbiter/PHI magnetograms Hamada et al. (2025). Enforcing Hale’s polarity law to a pair of Gaussian lobes converts the unsigned far-side signal into a smooth, polarity-resolved magnetogram. Such a scheme avoids unrealistically razor-thin current sheets, approximately conserves flux, and yields far-side magnetograms suitable for assimilation into full-Sun flux-transport and coronal models rather than serving purely as binary activity masks Hamada et al. (2026).

A complementary observation pathway opens when space missions such as STEREO provide EUV imaging of the non-visible hemisphere. The optically thin coronal emission is controlled by the photospheric magnetic flux in a non-trivial manner. This is exploited to translate EUV imaging observations into quantitative surface magnetograms as a data-driven image-to-image translation problem. Here, deep learning models use a learned surrogate rather than a physical forward operator, mapping the coronal observations $\vec { I } _ { \mathrm { E U V } }$ to the estimated far-side magnetic state via a cross-domain propagation operator ${ \mathcal { P } } _ { \mathrm { E U V  f a r B } } ;$

$$
\mathcal { P } _ { \mathrm { E U V  f a r B } } : \vec { I } _ { \mathrm { E U V } } = \mathcal { O } _ { \mathrm { E U V } } [ \vec { S } _ { \mathrm { a t m o s } } ]  \hat { \vec { S } } _ { \mathrm { f a r B } } ,\tag{1.10}
$$

where $\mathcal { O } _ { \mathrm { E U V } }$ is the coronal observation operator mapping the atmospheric state $\vec { S } _ { \mathrm { a t m o s } }$ to the observed EUV intensities. This propagation operator faces a supervision problem: far-side magnetograms are not directly available for supervised training, so the mapping $\mathcal { P } _ { \mathrm { E U V  f a r B } }$ cannot be trained directly from simultaneous farside observations. The methodological resolution is the proxy-pair construction — training on Earth-facing co-temporal SDO/AIA and SDO/HMI pairs, where the coronal morphology and photospheric field are simultaneously observable, then transferring the learned mapping to the far-side where only coronal morphology exists. The model is validated against withheld Earth-facing ground truth as the active region rotates into view Kim et al. (2019). Building on those ideas, new models use EUV 304 and 171A images together with SFT-predicted magnetic field data from one so-<sup>˚</sup> lar rotation earlier as inputs to reconstruct full-disk farside vector magnetograms. Its main advance is the extension to multispacecraft farside viewing from STEREO-A, STEREO-B, and Solar Orbiter, together with the first comparison of AI-generated farside results against SDO/HMI during near-conjunction intervals and the derivation of active-region magnetic parameters from those reconstructions Jeong et al. (2025b). Together, helioseismic inference and EUV-to-magnetogram propagation establish an operational chain from subsurface acoustic measurements and off-disk coronal images to the photospheric boundary conditions required by global MHD forecasting models. We note, however, that the transition from these models to realtime space-weather forecasting pipelines involves operational challenges (e.g., sensor degradation, telemetry drops, and live data assimilation) that are discussed in detail in the next chapter.

Through the inference $( \mathcal { I } _ { \mathrm { f a r B } } )$ and cross-domain propagation $( \mathcal { P } _ { \mathrm { E U V  f a r B } } )$ operators, machine learning successfully estimates the far-side magnetic state from noisy or indirect observations. However, because direct measurement is impossible until the active region rotates onto the Earth-facing disk, physical self-consistency becomes the primary metric for algorithmic success. Unconstrained generative models are prone to hallucinating visually convincing features that violate fundamental flux balance. Therefore, the critical next step for these architectures is not merely improving image-to-image accuracy, but embedding strict physical conservation laws, quantifying epistemic uncertainty, and leveraging multi-viewpoint data (e.g., Solar Orbiter) to guarantee the reliability of these synthetic boundary conditions for global forecasting.

## 1.3.3 Global dynamo and solar cycle evolution

The third physical family scales up from local convective and magnetic features to the macroscopic, long-term behavior of the internal solar dynamo. The solar cycle, which governs the rhythmic 11-year rise and fall of solar activity, is driven by the cyclic regeneration of toroidal and poloidal magnetic fields within the convection zone. Although predictions in this domain rely primarily on surface observations (such as synoptic magnetic maps, sunspot numbers, and $F _ { 1 0 . 7 }$ radio flux), they are fundamentally modeling the deep, global magnetohydrodynamic dynamo.

To formalize these macro-scale systems, the internal global dynamo state vector $\vec { S } _ { \mathrm { d y n } }$ can be defined as:

$$
\vec { S } _ { \mathrm { d y n } } = \{ B _ { \mathrm { t o r } } , B _ { \mathrm { p o l } } , u _ { p } , \eta , \ldots \} ,\tag{1.11}
$$

where $B _ { \mathrm { t o r } }$ and $B _ { \mathrm { p o l } }$ represent the internal toroidal and poloidal magnetic field strengths, $u _ { p }$ denotes the meridional circulation velocity, and η represents the turbulent magnetic diffusivity.

Because this deep state is visually inaccessible, machine learning operates in this macroscopic domain in two primary capacities. First, it models the spatial evolution of the photospheric boundary, providing both data-driven forecasting operators $( \mathcal { F } _ { \mathrm { S F T } } )$ for synoptic maps and physics-informed continuous solvers for the governing flux-transport equations. Second, it acts as a temporal propagator $( \mathcal { P } _ { \Delta t } )$ that forecasts highly compressed historical observations $( \vec { y } _ { \mathrm { c y c l e } } = \mathcal { O } _ { \mathrm { c y c l e } } [ \vec { S } _ { \mathrm { s u r f B } } ] \in$ $\{ \mathrm { S S N } , F _ { 1 0 . 7 } \} )$ ) over multi-year baselines. While full inverse data assimilation $( \mathcal { I } )$ remains the ultimate theoretical goal for directly inferring the internal $\vec { S } _ { \mathrm { d y n } }$ state, current architectures are driving advances toward forward-modeling and sequenceprediction pathways.

## 1.3.3.1 Machine learning for surface flux transport and solar cycle modeling

Classical solar cycle predictions rely heavily on Surface Flux Transport (SFT) and kinematic Babcock–Leighton dynamo models to propagate magnetic flux and forecast polar field reversals. Classical SFT describes the advection and diffusion of the radial magnetic field $( B _ { r } )$ by differential rotation, meridional circulation, and supergranular dispersal Yeates et al. (2023). However, traditional grid-based numerical solvers are computationally expensive and struggle with non-linear stability. ML is increasingly deployed to replace or augment these classical solvers, bifurcating into pure data-driven surrogates and physics-informed continuous solvers.

At the purely data-driven end of the spectrum, image-to-image translation networks based on conditional Generative Adversarial Networks (cGANs) act as direct forecasting operators $( \mathcal { F } _ { \mathrm { S F T } } )$ ). These models map current synoptic observations directly to future configurations:

$$
\mathcal { F } _ { \mathrm { S F T } } : \vec { B } _ { \mathrm { s y n } } ( t ) = \mathcal { O } _ { \mathrm { s y n } } [ \vec { S } _ { \mathrm { s u r f B } } ( t ) ] \to \hat { \vec { B } } _ { \mathrm { s y n } } ( t + \Delta t ) ,\tag{1.12}
$$

where $\vec { B } _ { \mathrm { s y n } } ( t )$ represents the observed 2D photospheric magnetic flux distribution for a given solar rotation (related to the surface state via the synoptic mapping operator $\mathcal { O } _ { \mathrm { s y n } } )$ , and $\hat { \vec { B } } _ { \mathrm { s y n } } ( t + \Delta t )$ is the predicted global magnetic configuration one full rotation (≈ 27 days) later. Rather than explicitly calculating the physical advectiondiffusion equations, the cGAN architecture learns the macroscopic transport rules— such as differential rotation and meridional flow—directly from historical data. By utilizing a generative network to synthesize the future map and a discriminator network to enforce pixel-wise structural fidelity, these image-to-image models successfully outperform conventional SFT solvers in global correlation for nextrotation forecasting. By learning the synoptic-map propagation directly from data, they efficiently capture the surface transport of existing magnetic elements Jeong et al. (2025a,c). However, their predictive horizon—like that of conventional SFT codes—naturally does not encompass the stochastic emergence of new active regions.

To integrate physical fidelity directly into the architectural learning process, Physics-Informed Neural Networks (PINNs) offer a powerful alternative. PINNs embed the governing physical differential equations (such as the advection-diffusion or MHD equations) directly into the network’s loss function, penalizing violations of physical laws (e.g., flux conservation) and forcing the model to respect the underlying physics without requiring a conventional grid. Rather than acting as inverse data-assimilation tools, baseline PINN architectures function as highly efficient forward surrogate solvers for the linear SFT equation. By embedding the governing PDE directly into the loss function, the network learns a continuous, mesh-free representation of the flux evolution without requiring explicit spatial discretization or numerical time-stepping, demonstrating better flux conservation and mesh independence compared to classical explicit schemes Athalathil et al. (2024).

The deeper advantage of this differentiable framework emerges when scaled to a non-linear Babcock–Leighton dynamo regime. Global cycle models depend heavily on the non-linear feedback mechanisms that govern the conversion of toroidal flux into poloidal flux. PINN-based forward solvers robustly capture these complex nonlinearities achieving superior numerical stability and flux conservation compared to traditional upwind schemes Athalathil et al. (2026). While current implementations operate with fixed transport parameters, establishing these continuous differentiable solvers lays the critical groundwork for future multiparameter data assimilation and the simultaneous inversion of the internal dynamo state.

## 1.3.3.2 Long-term time-series and solar cycle forecasting

Because the internal dynamo state remains inaccessible to direct observation, most machine-learning approaches forecast low-dimensional observables that encode information about the global magnetic cycle, such as sunspot number, $F _ { 1 0 . 7 }$ flux, polar field strength, or synoptic magnetic harmonics.

The most widespread application in this family is the direct forecasting of cycle-related observables using historical activity indices (including retrospective predictions for Cycle 24 and forward forecasts for Cycles 25 and 26). Here, machine learning acts as a temporal propagation operator, mapping a history of past observations to a future forecast:

$$
\mathcal { P } _ { \varDelta t } : \vec { y } _ { \mathrm { c y c l e } } ( t ) = \mathcal { O } _ { \mathrm { c y c l e } } [ \vec { S } _ { \mathrm { s u r f B } } ( t ) ]  \hat { \vec { y } } _ { \mathrm { c y c l e } } ( t + \varDelta t ) ,\tag{1.13}
$$

where $\vec { y } _ { \mathrm { c y c l e } }$ represents the vector of observed cycle activity indices (e.g., SSN, $F _ { 1 0 . 7 } )$ through the observation operator $\mathcal { O } _ { \mathrm { c y c l e } }$ and ∆t ranges from several months to a full solar cycle (approx. 11 yr). These models primarily predict cycle amplitude, peak timing, and overall temporal evolution.

A physical motivation for sequence-based architectures is that solar activity exhibits memory on multiple timescales: the Hale magnetic cycle (∼22 yr) modulates the 11-year sunspot envelope, meridional circulation imposes a multi-year transport delay on polar field buildup, and stochastic active region emergence introduces cycle-to-cycle scatter on shorter timescales Cheung and Isobe (2014). From a dynamo perspective, forecasting the sunspot number (SSN) time-series can be viewed as a highly compressed, scalar reduction of the global photospheric boundary state $( \vec { S } _ { \mathrm { s u r f B } }  \mathrm { S S N } )$ , serving as an accessible proxy to constrain and predict the unobservable internal dynamo state. Early baseline attempts using simple, static feed-forward networks Calvo et al. (1995); Fessant et al. (1996); Uwamahoro et al. (2009) demonstrated proof-of-concept forecasting, although their feed-forward architectures lacked explicit mechanisms for retaining information across long temporal sequences. In contrast, Long Short-Term Memory (LSTM) networks address this directly by selectively retaining or discarding information across sequences of arbitrary length. LSTM architectures trained on historical sunspot indices typically outperform classical regression, with benchmarks across non-deep, deep, and ensemble methods establishing the relative skill of each class Wang et al. (2021a); Li et al. (2021); Prasad et al. (2022, 2023); Dang et al. (2022).

The prediction target extends beyond scalar sunspot number. Machine-learning methods have been applied to daily $F _ { 1 0 . 7 }$ radio-flux indices Huang et al. (2009), to the planetary geomagnetic activity index $A p$ Okoh et al. (2018), to the sunspot butterfly diagram itself Covas et al. (2019), and to spherical-harmonic representations of synoptic magnetic maps Zhuo et al. (2024), extending the forecasting problem from scalar activity proxies to the evolution of the large-scale magnetic field itself. In the same vein, machine-learning analyses of extended historical records have identified 120-year oscillatory variations and forecast future extended solar minima Velasco Herrera et al. (2021); Rodr´ıguez et al. (2024a).

The methodological progression in this family reflects three distinct forecasting objectives. First, capturing the multi-scale temporal structure requires models capable of resolving both short- and long-term dependencies simultaneously Benson et al. (2020); Su et al. (2024). Second, resolving long-range dependencies to capture sub-cycle features over the full historical sequence requires transformer-based architectures or multivariate models Cao et al. (2024); Rodr´ıguez et al. (2024b). Third, moving beyond point forecasts toward probabilistic forecasting provides more actionable output; Bayesian forecasting architectures that constrain uncertainty bounds on cycle amplitude and timing Bizzarri et al. (2022) offer a more operationally relevant product than deterministic forecasts, because forecast uncertainty can be explicitly incorporated into planning and risk assessment.

Machine learning has established that the solar cycle amplitude, peak timing, and sub-cycle fine structure are all partially predictable from historical activity indices. However, true operational forecasting remains challenging; as noted in recent comprehensive reviews Karak (2026), many pure ML models struggled to accurately predict Solar Cycle 25’s strength ahead of time. A fundamental limitation of purely data-driven forecasting is the intrinsically small sample size available for supervised learning. Even with historical sunspot records spanning centuries, the availability of only approx. two dozen solar cycles severely limits statistical power and increases overfitting risk. Furthermore, current machine-learning models primarily propagate observable surface proxies forward in time, rather than addressing the central physical challenge: inferring the latent dynamo state $\vec { S } _ { \mathrm { d y n } }$ from incomplete observations. Therefore, the path toward operationally useful forecasting ultimately lies in the hybrid data-assimilation frameworks described in Section 1.3.3.1, combining the temporal sequence modeling of deep learning with the physical constraints of continuous differential solvers.

## 1.4 The solar atmosphere: from surface to corona

The solar atmosphere is a highly tenuous plasma where magnetic field dynamics increasingly dominate the thermodynamics with height. From the photosphere onwards, the atmosphere experiences a near-exponential drop in density across ≈ 10 orders of magnitude. Alongside this, we observe a dramatic rise in temperature from $\approx 5 5 0 0 \mathrm { K } \mathrm { t o } \geq 1 0 ^ { 6 } \mathrm { K }$ . The plasma also undergoes varying stages of ionization from the photosphere to corona, where it is fully ionized. Furthermore, the magnetic field starts playing an increasingly larger role in the solar atmosphere, adding to the complexity of processes in the system.

Structurally, the solar atmosphere is divided into four main layers: the photosphere, the chromosphere, the transition region, and the corona. Just above the photosphere lies the chromosphere, which typically consists of partially ionized plasma with temperatures rising from a minimum of $\approx 4 . 4 \times 1 0 ^ { 3 } \mathrm { ~ K ~ }$ up to $\approx 2 \times 1 0 ^ { 4 } ~ \mathrm { K }$

Above this, the temperature rapidly climbs up in a geometrically very thin layer to $\approx 1 0 ^ { 6 } 1$ K, which we call the transition region. Owing to the sharp reduction in gas pressure, the magnetic field starts to dominate the dynamics of this region. Above the transition region lies the solar corona, where temperature can go higher than $1 0 ^ { 6 }$ K. Typically, the dynamics in this part of the atmosphere is dictated by the magnetic field, including the morphology, strength, etc. In general, the entire atmosphere is quite tightly coupled, with mass and energy transfer from bottom to the top and vice-versa.

The predominant mode of observations in this region is through spectral lines, which form at different characteristic temperatures and pressures <sup>2</sup>. The observation mode is either through spectral band-passes covering several spectral lines, purely spectral, or spectropolarimetric. The observations can be either taken on ground or in space, with each having its own set of complications and issues. Since a wealth of spectral lines form across various conditions in the solar atmosphere, we are able to generate a sense of the typical plasma state in the atmosphere. In particular, this wealth of data means a strong contribution of ML techniques in understanding the plasma state in the atmosphere at a wide range of spatial, temporal, and energy scales.

The state vector is defined as:

$$
\vec { S } _ { \mathrm { a t m o s } } = \{ T , \chi _ { i } , n _ { e } , \vec { \nu } , \vec { B } , \xi , \ldots \} ( x , y , z , t ) ,\tag{1.14}
$$

where $T$ represents temperature, $\chi _ { i }$ the composition, $n _ { e }$ the electron number density, ⃗v the velocity vector, $\vec { B }$ the magnetic field vector, and $\xi$ the microturbulence, all as a function of spatial coordinates and time, respectively. Sometimes it is non-trivial to work in geometric height, and instead prefer to work in terms of optical depth (τ), which is defined as

$$
\begin{array} { r } { d \tau = \alpha _ { \nu } ( s ) d s . } \end{array}
$$

Here, $\alpha _ { \nu }$ would be the absorption coefficient of the plasma, which depends on the radiation properties, composition, density, etc. and s is the line of sight length, which would be some combination of path length from $x , y , z$ . The main observables from this state vector would be spectral & spectro polarimetric observations, narrow band filtergrams across different wavelengths, and radio signatures.

## 1.4.1 Inference: thermodynamic andplasma state of the atmosphere

The observations in this section are either spectral, spectropolarimetric or imaging observations of the atmosphere. Given the observations $( \mathcal { O } \bar { [ \vec { S } _ { \mathrm { a t m o s } } ] } )$ , we seek to obtain $\vec { S } _ { \mathrm { a t m o s } }$ . This is depicted pictorially in Fig. 1.3. We note that such an inference can be accomplished in various ways: through development of a direct inference (I) method, or through the development of surrogate models (O), or through the development of emulators (E) coupled with existing inversion methods.

![](images/937d8fa4bd038fd3c7553111ba61d3a1d7f5d64d73f83a69b382b1de7573f77e.jpg)  
Fig. 1.3: Several ways of using ML to perform inference of the solar state vector given observations. These include performing a direct inversion given the observations, or model the forward process in a data-driven manner.

In the lower solar atmosphere, the path length for photons is short when compared to the line of sight length, so effectively the light seen is formed in optically thick conditions. Effective inference $( \mathcal { I } )$ in optically thick regimes requires full radiative-plasma modeling, as simpler approximations often fail to capture vertical stratifications. However, higher up in the atmosphere, the path length for photons is long, rendering the radiation optically thin. In such conditions, $\breve { \vec { S } } _ { \mathrm { a t m o s } }$ can be effectively reduced to a few free parameters, since different parameters get decoupled from radiation.

## 1.4.1.1 Optically thick plasmas

A Stokes measurement encodes the full polarimetric state of light into four quantities: total intensity (I), linear polarization along two axes $( Q , U )$ , and circular polarization (V) — making it the primary observational diagnostic of magnetic field strength and orientation in the solar atmosphere.

In the photosphere and chromosphere, much of the information about the atmosphere is encoded in spectral-line profiles and the Stokes vectors. Recovering these physical parameters from spectra constitutes a fundamental inversion task (I). A key observational challenge here is that polarimetric signals are extremely weak. They are easily corrupted by instrumental noise, polarization crosstalk, and atmospheric turbulence (seeing). Consequently, advanced spatial and spectral denoising is required before physical parameters can be reliably recovered. These spectra lines form in optically thick conditions, making the line formation highly complex. A compact lower-atmosphere state vector may be written as:

$$
\vec { S } _ { \mathrm { a t m o s , t h i c k } } = \{ T , \rho , n _ { e } , \vec { \nu } , \vec { B } , \chi _ { i } , \ldots \} ( x , y , z , t ) ,\tag{1.15}
$$

where $T$ represents temperature, $\rho$ the mass density, $n _ { e }$ the electron density, ⃗v the velocity field, $\vec { B }$ the magnetic field vector, and $\chi _ { i }$ the chemical composition. The most straightforward estimation of $\vec { S } _ { \mathrm { a t m o s , t h i c k } }$ is to estimate the inversion operator such that:

$$
\mathcal { S } : \mathcal { O } [ \vec { S } _ { \mathrm { a t m o s , t h i c k } } ]  \vec { S } _ { \mathrm { a t m o s , t h i c k } } .\tag{1.16}
$$

Early neural-network inversions established the foundation for modern techniques by demonstrating learned mappings from Stokes profiles to thermodynamic parameters and depth-dependent stratifications Carroll and Staude (2001); Carroll and Kopf (2008). Systematic tests also showed improved stability and noise tolerance relative to classical iterative schemes Socas-Navarro (2005).

Because inversions rely heavily on repeated forward modeling, a critical application of ML has been accelerating the forward radiative-transfer calculations themselves. Learned radiative-transfer models act as fast emulation operators $( \mathcal { E } )$ to generate synthetic Stokes profiles directly from 1D atmospheric states, bypassing traditional synthesis codes Raygoza et al. (2024); Asensio Ramos and de la Cruz Rodr´ıguez (2025). Beyond end-to-end profile synthesis, a highly practical strategy is learning surrogates for expensive intermediate quantities, such as non-LTE departure coefficients. Utilizing 3D convolutional neural networks (CNNs) and Graph Neural Networks (GNNs)—which represent physical quantities as nodes on a graph and map their couplings along connections (edges) to model non-local interactions like multi-line radiative transport—these surrogates bypass the traditional statistical equilibrium solver, achieving order-10<sup>3</sup> acceleration while preserving chromospheric structure Chappell and Pereira (2022); Vicente Arevalo et al.´ (2022, 2026). Alternatively, the statistical equilibrium solver can be reframed as an active control task (C). Policy-Based Radiative Transfer utilizes model-free reinforcement learning policies to discover the depth-dependent source function dynamically, eliminating the need for approximate lambda operators Panos and Milic´ (2025).

Returning to the inverse problem $( \mathcal { I } )$ , some retrieval systems leverage these speedups through hybrid architectures by providing high-quality initializations for iterative inversions, accelerating convergence while retaining physics-based refinement Gafeira et al. (2021). When fully supervised mapping is employed, training sets are typically constructed in two ways depending on the specific task and the available realism: using synthetic data from numerical simulations degraded by instrumental effects, or using real observations paired with classical inversion results Asensio Ramos and D´ıaz Baso (2019); Milic and Gafeira´ (2020); Liu et al. (2020); Yang et al. (2024, 2025); Sainz Dalda et al. (2019, 2026a). Feed-forward architectures like CNNs and Transformers Campbell et al. (2026)—which use selfattention mechanisms to weigh the relationships between different parts of an input (such as different spectral channels or pixels) regardless of their physical distance— excel at these fast mappings. By drastically reducing inference time, these models enable continuous, large field-of-view studies that turn single-pixel physical conclusions into population statistics. Examples include recovering chromospheric properties of Ca II K fibrils across an entire field of view Kianfar et al. (2020), or mapping the spatio-temporal distribution of chromospheric heating across a plage region at high cadence Morosin et al. (2022). Such tasks were previously computationally intractable with classical pixel-by-pixel inversions. Beyond scaling up singleinstrument inversions, this distillation paradigm can even bridge disparate observational domains: Visual Transformers have been deployed to translate broadband imaging and magnetograms (SDO/AIA and HMI) directly into chromospheric thermodynamic states, using $\mathrm { I R I S } ^ { 2 }$ spectral inversions as the ground-truth target Sainz Dalda et al. (2026b). A fundamental caveat of these supervised approaches is that their physical fidelity is strictly bounded by the realism of the training set. Synthetic models often fail to capture the full complexity of real observations, such as stochastic noise and instrumental artifacts. Furthermore, a new model must typically be trained from scratch for each new instrument, which remains computationally expensive and time-consuming.

To circumvent the biases of simulation-based training sets, a separate class of architectures shifts away from feed-forward prediction toward per-observation optimization using PINNs and Neural Fields. In these representations, the state vector is defined such as:

$$
\vec { S } _ { \mathrm { a t m o s , t h i c k } } = \mathbf { N N } ( x , y , z , t ) ,
$$

where NN is a neural network representation of the state vector. The physics forward model is assumed to be a smooth, differentiable framework, and the problem is defined as:

$$
\mathbf { N N } : \{ x , y , z , t \}  \mathcal { O } [ \mathbf { N N } ( x , y , z , t ) ] .\tag{1.17}
$$

These methods embed the radiative-transfer equations directly into the optimization loss function Jarolim et al. (2025a); Li et al. (2025b). This idea of using a neural network to represent the atmosphere as a continuous field can be very powerful to enforce spatio-temporal consistency across the atmosphere providing results that are not only visually convincing but also physically consistent and robust against noise and ambiguities D´ıaz Baso et al. (2025a). Unlike fast surrogates, these methods require optimization at inference time, but they guarantee that the retrieved parameters strictly enforce physical and spatial constraints.

Finally, because spectropolarimetric inversion is mathematically non-identifiable under limited observables, deterministic point estimates are physically incomplete and the uncertainty of the inferred state vector becomes vital for scientific interpretation Mistryukova et al. (2023). Probabilistic models explicitly target this illposedness. In highly dynamic flare conditions, where standard assumptions fail, invertible neural networks (INNs) provide a probabilistic route by jointly learning forward and inverse mappings, successfully recovering thermodynamic structure from complex RADYN-simulated profiles like Hα and Ca II 8542 Osborne et al. (2019). Approximate Bayesian inference can also be framed as a Doppler imaging problem, where neural networks learn to map stellar line profiles to surface abundance maps while capturing posterior uncertainty Asensio Ramos et al. (2022). To obtain the full posterior distribution of $\vec { S } _ { \mathrm { a t m o s , t h i c k } }$ , Normalizing Flows —which learn complex probability distributions by transforming a simple distribution (like a Gaussian) through a sequence of invertible, differentiable steps—offer amortized posterior estimation, shifting the computational cost entirely to training while enabling rapid, mathematically calibrated Bayesian inference at test time D´ıaz Baso et al. (2022) for individual pixels or even performing Bayesian inference across an entire 2D field-of-view using flow-matching techniques (Asensio Ramos et al., 2026).

## 1.4.1.2 Optically thin plasmas

With increasing height and reducing density, radiation forms in optically thin conditions. In the transition region and the corona, the excitation of the upper atomic levels can be approximated as collision-dominated, while de-excitation is predominantly due to spontaneous emission Aschwanden (2005). This simplification decouples the radiation field from the local plasma state, allowing us to express the emission purely in terms of temperature, electron density, and ion properties. Consequently, our state vector reduces to $\vec { S } _ { \mathrm { a t m o s , t h i n } } = \{ T , \dot { \chi } _ { i } , n _ { e } , \vec { \nu } \} ( x , y , z , t )$ , where $T$ represents temperature, $\chi _ { i }$ the composition or ionization fraction, $n _ { e }$ the electron density, and ⃗v the velocity field.

For spectral line emission, this discrete state vector must be integrated along the line of sight. In optically thin conditions, the observed intensity I is classically written as:

$$
I \propto \int n _ { e } ^ { 2 } G ( T ) d s ,\tag{1.18}
$$

where $G ( T )$ is the contribution function of the spectral line of interest, encapsulating the relevant atomic physics (elemental abundance, ionization fraction, and excitation rates). In a more general setting that accounts for macroscopic plasma motions (Doppler shifts), this can be rewritten as Cheung et al. (2019):

$$
I \propto \int _ { \nu } \int _ { T } \mathcal { D } ( T , \nu ) \mathcal { G } ( T , \nu ) d T d \nu ,\tag{1.19}
$$

where $\mathcal { D } ( T , \nu )$ is the Differential Emission Measure (DEM), a macroscopic measure of the plasma density distribution across different temperatures and velocities along the line of sight. If the observations permit, these integrals can be further expanded to disambiguate specific chemical compositions. Because $\mathcal { D } ( T , \nu )$ is a property of the plasma rather than of any particular instrument passband, it provides a portable, instrument-agnostic description of the thermal state — making it the natural quantity for cross-mission comparison and multi-wavelength synthesis.

Because the line-of-sight integration destroys 3D spatial information, multiple different thermal distributions can produce the identical observed intensity. Classically, this is solved using linear solvers stabilized by regularization Hannah and

Kontar (2012); Cheung et al. (2015, 2019); Plowman and Caspi (2020); Youn et al. (2025). Machine learning transforms this by deploying learned inference operators to instantly map multi-wavelength intensities to DEM distributions. However, because of the inherent mathematical degeneracy of the integral, deterministic pointestimate networks are often physically insufficient. The most robust formulations explicitly solve for the full posterior distribution Kashyap and Drake (1998), quantifying the range of thermal structures consistent with the observed intensities.

Beyond standard DEM inversions, machine learning operators can disentangle overlapping physical regimes that violate the pure optically thin assumption. Very frequently, cool chromospheric plasma $( \mathrm { e . g . }$ , filaments) is suspended in the corona, absorbing underlying EUV radiation. Deep learning techniques act as inference operators $( \mathcal { I } )$ to mathematically separate the cool, optically thick absorbing component from the hotter, optically thin background plasma McMullan et al. (2026).

Finally, observing the off-limb solar corona (such as in white-light coronagraphy) presents a distinct geometrical inverse problem. Here, photons undergo Thomson scattering off free electrons, making the direct target the three-dimensional electron density. This 3D tomographic reconstruction can be achieved by using neural field representations and solving the forward scattering problem as a differentiable operator (Asensio Ramos, 2023) or as an image translation problem from synoptic disk observations to electron density maps Jang et al. (2021).

## 1.4.2 Three-dimensional magnetic-field extrapolation

A very specific subsection of the atmospheric problem is dedicated to estimating the three-dimensional magnetic field across the upper solar atmosphere. Significant dynamics in the solar atmosphere are controlled by the magnetic field, which couples mass and energy flows between the lower and upper atmosphere. Because direct 3D measurement of the coronal field remains observationally challenging, these volume fields must be derived from boundary estimates. We define this lower boundary state as:

$$
\vec { S } _ { \mathrm { s u r f B } } = \{ \vec { B } _ { \mathrm { p h o t } } \} ,\tag{1.20}
$$

where $\vec { B } _ { \mathrm { p h o t } }$ represents the vector magnetic field at the photosphere. The extraction and refinement of this state-of-the-art boundary condition from raw observations— including cross-instrument harmonization $( \mathrm { e . g . }$ , SDO/HMI to Hinode/SOT), spatial super-resolution, and far-side reconstruction—are structured as observation-space operators $( \mathcal { P } _ { 1  2 } )$ and global temporal propagators, which are discussed extensively in Sections 1.6 and 1.3, respectively.

Assuming a reliable surface boundary state $\vec { S } _ { \mathrm { s u r f B } }$ has been retrieved, one seeks the 3D coronal field:

$$
\begin{array} { r } { \vec { B } _ { \mathrm { c o r } } = \mathcal { P } _ { \mathrm { z = 0  z } } [ \vec { S } _ { \mathrm { s u r f B } } ] . } \end{array}\tag{1.21}
$$

This must satisfy physical constraints such as $\nabla \cdot { \vec { B } } = 0$ and, in force-free models, $\nabla \times \vec { B } = \alpha \vec { B }$ . Hence, this step is not an inference problem $( \mathcal { I } )$ from local emission, but a spatial propagation problem $( \mathcal { P } _ { 1  2 } )$ . Potential-field models assume zero currents $( \alpha = 0 ) ;$ linear force-free models allow a spatially uniform current helicity (α = const); and nonlinear force-free (NLFFF) models permit spatially varying $\alpha ,$ making them the most physically realistic but also the most computationally demanding.

ML addresses this extrapolation problem in two distinct ways. The first is solution fitting: a continuous neural representation is optimized for a particular active region or boundary condition while penalizing violations of the governing Maxwell equations. PINNs provide this grid-free formulation, successfully achieving forcefree coronal-field extrapolations Jarolim et al. (2023); Zhang et al. (2024c). Because the photospheric field is not itself fully force-free, applying multi-height constraints (incorporating chromospheric fields) significantly improves the physical validity of these PINN extrapolations Jarolim et al. (2024). Additionally, generative approaches provide a probabilistic route to mapping boundary information into a coronal volume; for instance, denoising diffusion models operating on spherical harmonics can synthesize physically realistic 3D coronal magnetic fields across the solar cycle da Silva et al. (2025).

The second route is operator learning. Instead of optimizing a new neural field from scratch for each active region, one learns the mapping from a boundary magnetogram directly to the 3D coronal field across a massive training set. We write this as an emulation problem:

$$
\vec { B } _ { \mathrm { c o r } } = \mathcal { E } _ { \mathrm { z = 0 } } \left[ \vec { S } _ { \mathrm { s u r f B } } \right] .\tag{1.22}
$$

Using architectures such as Fourier Neural Operators (FNOs) and Physics-Informed Neural Operators (PINOs) Li et al. (2020); Jeon et al. (2025); Yang et al. (2026), the network provides a direct, single-pass estimation of the full 3D field. FNOs learn mappings between continuous infinite-dimensional function spaces (such as 2D boundary fields to 3D volume fields) directly in the frequency domain, allowing them to remain independent of the grid resolution. While this functions as a highly accelerated emulator $( \mathcal { E } )$ of classical extrapolation codes at inference time, it comes with a fundamental limitation: the learned operator strictly inherits the assumptions, errors, and systematic biases of the specific numerical NLFFF solvers used to generate its training targets.

Finally, global coronal-field modeling introduces an additional layer of complexity because the spatial propagation $( \mathcal { P } _ { \mathrm { z = 0 \to z } } )$ requires a synchronic boundary condition that includes the unobserved hemisphere. As discussed in Section 1.3, AIgenerated far-side magnetograms Jeong et al. (2020) close this loop. This represents the ultimate coupled inference chain: an inference operator $( \mathcal { I } )$ reconstructs the hidden far-side boundary, a spatial propagation operator $( \mathcal { P } )$ extrapolates the global boundary into the 3D corona, and the resulting coronal volume serves as the physical input for downstream space-weather forecasting models.

## 1.4.3 Evolution of the atmospheric magnetic state

The atmosphere is not a sequence of independent images. At active-region scales, its state evolves dynamically under flux emergence, cancellation, shearing, rotation, and reconnection. ML models therefore act as temporal propagation operators (P), bridging the gap between static extrapolations and fully dynamic MHD:

$$
\vec { S } ( t + \Delta t ) = \mathcal { P } _ { t  t + \Delta t } [ \vec { S } ( t ) ] .\tag{1.23}
$$

At these localized scales, learned operators predict the short-term evolution of the photospheric magnetic field from sequences of earlier magnetograms Bai et al. (2021). Rather than simulating the slow advection of the global solar cycle (which is governed by the interior dynamo; see Section 1.3), these atmospheric propagation operators focus on the rapid topological changes that precede energetic events.

To support global atmospheric evolution, handling missing observational data is critical. Earth-view magnetograms do not observe the full Sun simultaneously, yet global coronal and heliospheric models require a synchronic boundary condition. One approach is to infer far-side magnetic information from multi-spacecraft EUV observations Jeong et al. (2025b). Another is to generate magnetogram sequences directly from EUV image sequences, treating the dynamic EUV corona as an indirect tracer of the evolving magnetic lower boundary Sun et al. (2022). These methods couple atmospheric layers in a practical sense: coronal emission is used to continually update a hidden photospheric magnetic state that drives subsequent coronal or heliospheric modeling.

## 1.5 Coherent magnetic structures and transient events

Solar observations reveal ordered magnetic and thermodynamic structures spanning multiple spatiotemporal scales, though their apparent morphology is heavily biased by instrumental sensitivity. These structures may be in an unstable equilibrium, and may become unstable to release mass and energy into the atmosphere. As illustrated in Fig. 1.4, distinct atmospheric layers exhibit different morphological features (e.g., coronal holes, filaments) driven by variations in their emission measure profiles. In the AIA 193 A image, we have marked several regions: coronal holes,<sup>˚</sup> active regions, filaments, and quiet Sun. The differences in the intensities may be understood through the formalism of DEM $( \mathcal { D } ( T , \nu )$ ; from Eq. 1.19). We have seen in Eq. 1.19 that for optically thin plasmas, the intensity is a function of both $\mathcal { D } ( T , \nu )$ and $\mathcal { G } ( T , \nu )$ . Coronal holes represent regions of open magnetic flux characterized by both lower plasma density and reduced temperatures compared to the quiet Sun. This dramatic drop in emission measure $( n _ { e } ^ { 2 } )$ across temperatures sampled by the AIA 193 A passband is why they appear as dark regions in the image<sup>˚</sup> O’Dwyer et al. (2010). On the other hand, active regions contain enough plasma in the AIA 193 A temperature sensitivity, and hence appear bright. These structures are visible<sup>˚</sup> above a rather persistent background, which we call the quiet Sun. The filaments we see in this image present an interesting physical case — visually, one would think they have properties similar to coronal holes. However, multi-wavelength observations tell us that filaments are cold plasma suspended above hotter plasma below. They appear dark because background EUV emission is absorbed by dense, cool plasma via photoionization continua, not because there is a lack of emitters. In the right panel of the same figure, we display an image of AIA 1700 A, where we have<sup>˚</sup> marked two important regions — sunspots and plage. In this image, the DEM formalism fails since the observation is in the optically thick regime. However, through the aid of modeling and multiwavelength observations, we can say that plage corresponds to concentrated magnetic field which gives rise to enhanced emission. In the same image, we see the darker sunspots. One might think the magnetic field here is weaker, but the exact opposite is true. The concentrated magnetic field in a sunspot suppresses local convection. This blockage prevents energy transport from the interior, lowering the surface temperature and the observed intensity.

![](images/32c124b7e5a0b3fa9b03e02c332c10c6c7600773ab1d79704dab3550f0e17eb1.jpg)  
Fig. 1.4: Solar observations in AIA at 193 A in the left panel, and AIA 1700<sup>˚</sup> A in the<sup>˚</sup> right panel. Several features in the images are marked in both the images. Credit: SDO/AIA team from LMSAL.

Apart from the features marked in this image, features and activity on the Sun occur at a wide range of scales, each of which couple across different parts of the atmosphere. Furthermore, some of these structures couple not just across the layers of the atmosphere, but have an impact through the whole heliosphere. In this section, we shall consider literature that work with morphological structures and features across different length, time and energy scales.

## 1.5.1 Structure identification

We first focus on algorithms developed for feature identification. The state vector for any particular feature is the same as those of the solar atmosphere, as enumerated in Eq. 1.14. For the state vector $\vec { S } _ { \mathrm { f e a t } }$ of feature, the feature itself is defined as:

$$
\mathcal { F } _ { \mathrm { f e a t } } = \{ ( x , y , z , t ) \in \mathbb { R } ^ { 4 } \mid \vec { S } _ { \mathrm { a t m o s } } ( x , y , z , t ) = \vec { S } _ { \mathrm { f e a t } } \} ,\tag{1.24}
$$

where we define a feature $( \mathcal { F } _ { \mathrm { f e a t } } )$ as a set of points in space and time where the state vector is consistent with the state vector defining the feature. We operate in the observation space, so feature identification translates to an inference operator $( \mathcal { I } )$ such that:

$$
\mathcal { I } _ { \mathrm { f e a t } } : \vec { O } = \mathcal { O } [ \vec { S } _ { \mathrm { a t m o s } } , \vec { S } _ { \mathrm { f e a t } } ]  \mathcal { F } _ { \mathrm { f e a t } } .\tag{1.25}
$$

## 1.5.1.1 Spatial Boundary Extraction

The oldest observations of the Sun correspond to white-light images of the photosphere, where sunspots are seen very clearly. Hence, it is one of the oldest datadriven problems in solar physics, dating back centuries to recorded sunspot diagrams Vaquero (2007). The foundational challenge is that the boundary between umbra, penumbra, and quiet Sun is not a sharp physical interface but a continuous gradient in $\vec { S } _ { \mathrm { a t m o s } }$ . Early work focused on automated sunspot group classification based on feedforward neural networks Colak and Qahwaji (2008). Better representations of these segmentation masks were developed by coupling morphological operations to generate a more complete $\mathcal { F } _ { \mathrm { s u n s p o t } }$ Ashamari et al. (2015).

In all these tasks the ground truth, on which the model is trained, is very critical. Hence, it is imperative to have schemes for a well-posed definition of the ground truth. In such a case, unsupervised learning through genetic algorithm with simulated annealing has shown major success in determining sunspot umbra and penumbra Yang et al. (2018). Similarly, self-organizing maps (SOMs) have been employed with great success on spectroscopic data Schilliro and Romano (2021). These unsupervised neural networks project high-dimensional data onto a low-dimensional grid while preserving the topological structure of the data, so that physically similar profiles map close to each other. Gaussian Mixture Models (GMMs) have shown great success in segmenting not only sunspots, but even the small-scale fine structure in sunspots Gong et al. (2023). They model the overall data distribution as a combination of multiple multi-dimensional Gaussian sub-distributions, allowing for soft probabilistic clustering rather than hard boundaries. Classical segmentation and contouring operators have also been tested for sunspot detection Chen et al. (2025a,b), and they are scaled up with deep learning models like U-Net.

While automated methods have been developed specifically for sunspots owing to their fine structure and global impact, such methods have also been developed for other features through straightforward application of ML inference $( \mathcal { I } )$ . Filaments, which we have also seen in Fig. 1.4, are important structures that can become unstable and launch as CMEs into space. Filament identification and study is hence very important to track evolution, understand trigger of instability, and model the evolution of the flux rope after instability. Studies have been dedicated to identifying filaments through spectroscopic data and k-means clustering Kuckein et al. (2020). This unsupervised algorithm partitions data into k distinct groups by iteratively minimizing the distance between data points (e.g., spectral profiles) and their group centroids. The same filaments, when viewed outside of disk, appear bright rather than dark due to no emission in the background. Such regions can also be identified using a U-net formalism Zhang et al. (2024a). Fibrils are elongated structures in the chromosphere tracing magnetic field. These can be accurately identified through a U-net formalism Jiang et al. (2021). The chromosphere also exhibits vortex motions of plasma, which have been named chromospheric swirls. Identifying chromospheric swirls requires mapping the geometrical symmetries of the plasma. By pre-processing curvature-based features, unsupervised clustering operators (I ) can successfully isolate these vortex motions Dakanalis et al. (2021). Such feature generation can however be circumvented to the application of symmetry-aware methods and large amount of labeled data.

In the upper atmosphere, two of the most prominent morphological structures are active regions and coronal holes, which we have seen in Fig. 1.4. Coronal hole segmentation is very important to quantify the amount of open flux in the solar atmosphere, which contributes to both the solar wind in the heliosphere and also acts to determine the strength of the next solar cycle. Some methods to estimate coronal holes include using shallow ML supervised models Reiss et al. (2015), contour based segmentation & c-means clustering Bandyopadhyay et al. (2020), unsupervised k-means clustering Inceoglu et al. (2022), and CNN-based supervised models Illarionov et al. (2020). A major failure mode of all of these aforementioned coronal hole identification methods is a confusion between coronal holes and filaments. Filament identification uses passbands sensitive to different temperatures when compared to coronal holes, so as to obtain the most contrast in the features intensity-wise. This confusion is mitigated by using multi-wavelength input data. Additionally, models are trained on clean datasets that explicitly eliminate the coronal hole—filament ambiguity Jarolim et al. (2021).

## 1.5.1.2 Semantic Classification and Feature Representation

However, more than the identification of sunspots, it is also important to identify the group ofthe sunspot. These are visually assigned labels to sunspots that capture the spatial complexity and structure of the sunspots. This is further related to the probability of instability in these sunspots (for e.g. through solar flares), and the activity of the Sun itself. To address this, cutouts of the identified sunspot are classified into different groups. Classification models include Deep Neural Networks (DNNs) Fang et al. (2019), CNNs Mkwanda et al. (2024), and XGBoost Tang et al. (2021). DNNs are multi-layer feedforward networks that learn abstract physical mappings by passing inputs through sequential layers of non-linear neurons. In contrast, XGBoost is an ensemble method that builds decision trees sequentially to correct the errors of previous trees via gradient boosting. However, efforts have also focused on a full-pipeline of sunspot identification, clustering and classification through the application of multiple neural networks Sayez et al. (2023). We however note that the full computation graph is not tied in together in such cases, and each task is handled independently by an individual network, which does not exchange information with other networks. Recent work on supervised classification of sunspots into groups, however, clearly shows that vision transformers outperform CNNs, and point towards magnetograms being a better source of data than continuum Legnaro et al. (2025).

Apart from single feature extraction methods, there has been significant efforts in a zero-shot identification (I) of multiple features using a trained model. For instance, coronal holes, active region core, and active region boundaries can be distinguished through SVD + Bayesian classifier De Wit (2006). The algorithm considers intensities in different passbands as a feature set, computes a decision boundary on SVD-projected space, and uses these labels for a supervised classification. A more straightforward algorithm is the application of Gaussian mixture model for segmenting our coronal holes and active regions Upendran et al. (2020). In the lower atmosphere, for instance, cutouts of images, consisting of predominantly one class, are successfully classified by VGG-style networks Armstrong and Fletcher (2019). Global segmentation of images based on multi-channel data, based on Faster R-CNN, provides bounding box for different features in a zero-shot manner after training Baek et al. (2021). Development of segmentation methods in absence of a ground truth is very tricky. To this end, it is imperative to also evaluate the results from ML models to quantify “ground truth”. For example, Seth et al. (2025) develop a global algorithm that develops bounding box for different chromospheric features based on Solar Ultraviolet Imaging Telescope (SUIT; Tripathi et al., 2017) onboard Aditya-L1 Tripathi et al. (2023). This work trained a YOLO (You Only Look Once) model on synthetic data, modeling the instrument properties of SUIT and IRIS. The model frames target classification and bounding box localization as a single regression task evaluated in a single forward pass. The main innovation is that statistical properties (like entropy and skew) were computed for all features. These were then compared across the synthetic ground truth and the real SUIT observations. This approach helps evaluate model performance when real ground truth is unavailable.

Discrete latent representations offer an alternative to continuous autoencoders for spectral compression. Autoencoders are neural networks that compress data into a low-dimensional latent space and reconstruct it. For example, vector quantization of IRIS spectra into a finite codebook produces state labels that are intrinsically interpretable as physical activity classes and computationally efficient for real-time classification Ivanov et al. (2021). Vision-language representation learning extends the state space from visual to semantic: mapping AIA spatial features to natural language captions enables text-query-based solar event discovery and opens the archive to large language model integration Baek et al. (2024).

## 1.5.1.3 Temporal Tracking

All the methods presented above consider sunspots data independent of the time coordinate. The underlying assumption is that the training set covers sunspots of almost every possible configuration and property, and hence the ML method is able to identify it correctly. However, an important aspect of feature extraction in solar physics is temporal consistency, where in sunspots actually rotate from east to west, along with an evolution with time. This motion and temporal consistency to track active regions is formulated as a temporal propagation $( \mathcal { P } _ { t  t + \Delta t } )$ , done for instance by Gong et al. (2024). In this work, the authors develop a feature detection module, a Kalman filter based position prediction, and a temporal association module based on flow graph. This ensures a strong temporal continuity across different frames in the active region.

## 1.5.1.4 Historical Data Rescue

Apart from current solar data, there is a wealth of archival data covering sunspot drawings across centuries. Digitization of this data via inference operators $( \mathcal { I } )$ extends the available dataset capability backward in time. Convolutional networks applied to century-long photographic sun-charts extract solar limb, disk center, and plage boundaries from hand-drawn records. This converts analogue archives into ML-ready labeled datasets for solar cycle reconstruction Mishra et al. (2026). Tabular historical records also present significant digitization challenges due to handwriting variability. For example, the handwritten catalogs from the Zurich Observatory contain coordinates of sunspots, prominences, and faculae that bridge Sporer’s early¨ observations with 20th-century records. These were digitized using a convolutional recurrent neural network (CRNN) with an adaptive learning approach. The model automatically retrained on each page to leverage internal dependencies among the coordinate table values Illarionov and Arlt (2022).

## 1.5.2 Transient activity

The coherent structures identified in Section 1.5.1 exist in unstable equilibria dictated by the interplay between energy injection $( \tau _ { \mathrm { i n } } )$ and dissipation $( \tau _ { \mathrm { d i s s } } )$ timescales. If the time scale of energy injection is shorter than cooling time scales, there are discrete jumps of energy in the system. For very slowly varying heating, discrete jumps in energy are not as prominent. One more complication is traveling waves — these are hydrodynamic or MHD waves, which result in oscillations across the whole of the solar atmosphere, and may or may not result in energy deposition. Studying these events is therefore imperative to understand the general stability and energetics of the solar atmosphere.

## 1.5.2.1 Solar flares

Solar flares provide a primary example of this transient activity. Solar flares are sudden release of energy in the solar atmosphere that manifest as intense, short-lived brightenings across the electromagnetic spectrum. This rapid energy release drives global atmospheric responses, including the launch of fast-mode waves across the solar disk. Furthermore, a solar flare may also result in mass flows, and launching of filaments across the interplanetary space, resulting in Coronal Mass Ejections (CMEs). The scale of impact of such events thus prompts detailed statistical studies of both flares, and regions that give rise to flares.

The ML applications in flare physics fall into several distinct operator classes. The first critical task is using a temporal propagation operator $( \mathcal { P } _ { t  t + \Delta t } )$ to forecast signatures of instabilities that would be potentially correlated with flares. Static point-in-time boundary conditions are physically insufficient for forecasting without dynamic context, such as flux emergence rates and helicity injection. Because flares are rare, standard loss functions suffer from an “all-clear” bias, so training set rebalancing may help learning but the test set must remain untouched to reflect physical reality.

A very well-prepared and standard dataset containing active region data is from features extracted from magnetic field measurements of the photosphere (SHARP; Bobra et al., 2014). The SHARP dataset has been extensively used to study the underlying physics of active region producing flares, and also to develop ML models to forecast flares. Typically, a given active region can become unstable and flare, but not all active regions flare. To this end, understanding parameters that may be correlated with flaring active regions is imperative. Using the SHARP data, this may be formulated as a classification of active regions into flaring or non-flaring. Such an exercise can be done even with simple models like Support Vector Machines (SVMs), and reveals that the total unsigned vertical current is the most important differentiating parameter between flaring and non-flaring regions Bobra and Couvidat (2015). Similarly, the mean free magnetic energy and the past flaring history of an active region are also important indicators of flare probability of active regions Yang (2025). Simple ML models, with extracted physics features, are also able to discover the flare productivity of active regions several days prior to the flare Dhuri et al. (2019).

Apart from the SHARP features, there has been progress in deriving topologybased features to study their impact on flare forecasting algorithms. Such effort shows that topological parameters based on connectedness of sets Deshmukh et al. (2023) or on the magnetic topology of the active region Williams et al. (2026) already contain enough information, and may not require highly complex ML models for such classification. Similarly, k-shape clustering has been used to great effectiveness to identify step-like changes in the horizontal magnetic field during a flare, which is strongly correlated with flare dynamics and activity Alanazi and Fletcher (2026).

Sometimes, the particular SHARP parameters may not be available, since the vector magnetic field data required to compute these parameters may not be at a good enough cadence to capture the time-scale of activity during a flare. Hence, there has been efforts to translate from high cadence AIA intensities Benson et al. (2021) and line-of-sight magnetic field Dhuri et al. (2022). These efforts act as crossdomain propagation operators $( \mathcal { P } )$ , translating across states, and correspond to

$$
\mathcal { P } _ { \mathrm { d a t a }  \mathrm { S H A R P } } : \vec { S } _ { \mathrm { d a t a } }  \vec { S } _ { \mathrm { S H A R P } } .\tag{1.26}
$$

While forecasting models attempt to predict flares, inference operators $( \mathcal { I } )$ diagnose them in real-time. While the photosphere shows indications of system stability, the atmosphere is known to already start responding to energy accumulation before the start of a flare. For instance, spectral lines in the chromosphere already show indications of flaring activity several minutes before a flare Panos et al. (2018); Woods et al. (2021); Ivanov et al. (2021); Zbinden et al. (2024); Zbinden and Kleint (2026). The key methodological insight is that clustering is not merely dimensionality reduction, but physical taxonomy. When hierarchical k-means is applied to Mg II h&k profiles during flaring conditions, the resulting clusters directly map to distinct atmospheric regimes — quiet-Sun double peaks, pre-flare single-peaked heated profiles, and broad blueshifted-reversal profiles at ribbon fronts — and the last cluster correlates directly with hard X-ray signatures of high-energy electron deposition Panos et al. (2018). This connection between spectral morphology and particle acceleration physics makes the clustering result a physical discovery, not just a compression.

Inherently, the shapes of spectral lines that form in optically thick conditions depend on the atmospheric column properties, and these shapes can be multi-peaked non-Gaussian profiles. Hence, a systematic study of different spectral lines and their inter-correlations during flares can be expressed through a formalism of mutual information. A very innovative way of performing this would be to transform continuous spectra to categorical data using k-means clustering, and then compute the mutual information across different spectral lines Panos et al. (2021). A very interesting paradigm is to rephrase the solution through a mutual information neural estimator to study such spectral associations. The complexity of spectral observations, reduced by ML, reveals the inherent coupling across spectral lines which occurs due to inherent coupling across the atmospheric column. Thus, application of both information theory and ML enables interpretation of potentially even new physics from complex spectral observations Panos et al. (2021).

Solar flares are typically inferred from soft X-ray observations. However, if the flare footpoint is behind the solar disc, a visual inspection is needed to decipher if the spectral response corresponds to a flare or not. However, parameterizing the problem as classification using spectrograms enables identification of occulted flares Ishikawa et al. (2021).

A single energetic flare can trigger a cascade of secondary atmospheric phenomena that ripple across the entire solar disk. Following solar flares, the whole atmosphere may respond to release of energy, through the propagation of fast-mode waves across the atmosphere, known as EUV / EIT waves. Detecting these waves enables us to determine the energetics and properties of the solar flare, and the overarching environment in which the active region rests. Furthermore, with CMEs, they are also able to provide early indicators of onset of CMEs, enabling space weather preparedness of humanity. The particular cutout containing the wavefront can be identified by models like Faster Region-based CNN (Faster R-CNN), and the outline of the wavefront drawn using a pix2pix GAN formalism Xu et al. (2020), enabling a very fast identification of wavefronts due to flares.

## 1.5.2.2 Other transient events

While solar flares are excellent examples of transient events in the solar atmosphere, they are the largest of such events, in terms of energy, length and time scales. The transient events occur at a wide variety of time scales, occurring across the solar atmosphere. Since the amount of energy in such events might be small, the atmospheric response may not be observed across all spectral lines / wavelength bands.

In the photosphere, concentration of magnetic flux elements are seen as bright points. Identifying, tracking and studying these bright points presents us with an understanding of magnetic field dynamics in the photosphere, which are used to understand the energy injection into the solar atmosphere. To isolate these transient bright points from the photospheric background, ML inference operators (I) estimate the exact separation thresholds directly from the observational data Bai et al. (2023).

In the chromosphere, Ellerman bombs are very compact (∼ 1000 km), dynamic (lasting ∼ 5 min) events seen in lower atmosphere. CNN and DNN models, trained on both high resolution spectra and low resolution imaging data are able to identify these events very accurately Soler Poquet et al. (2025). Such methods are also validated by studying the variation of performance through permutation of inputs, and show consistency with known physics. Apart from Ellerman bombs, the chromosphere also displays a myriad zoo of features, including rapid flows seen as elongated streaks in space, known as spicules — chromospheric plasma jets that are thought to be a key channel for mass and energy transport into the solar corona. k-means clustering has proven to be extremely successful in identifying spectral profiles exhibiting different kinds of spicules Bose et al. (2019).

Much higher in the transition region and corona, small-scale impulsive events are observed again. Parker’s nanoflare hypothesis proposes that the corona is sustained by a superposition of sub-resolution reconnection events — individually too faint to detect, but collectively sufficient to maintain coronal temperatures above 10<sup>6</sup> K. For instance, combining Zernike moments of images with an SVM classifier works effectively in scaling up identification of small-scale events with the SolO/EUI Alipour et al. (2022). Sometimes, the individual small-scale impulsive events may not be directly observable. Rather, what may be observed is the impact of the statistical distribution of such events. To estimate the distributional parameters of these events, one may perform a simulation based inference by training on a set of synthetic data, and performing inference on real data Bazarghan et al. (2008); Upendran and Tripathi (2021); Upendran et al. (2022). These methods push the limits the of the instruments, and enable large scale statistical studies of the smallest transients in the solar atmosphere. Such simulation based inference can also be performed through a realistic physics simulator. By training a model on features extracted from simulations, a Decision Tree or Random Forest can provide insights into the typical heating mechanism in the active regions Barnes et al. (2021). A Random Forest makes predictions by aggregating the consensus vote of many individual decision trees trained on different data subsets to avoid overfitting. Such weak impulsive events are also observed in radio imaging spectroscopy. They are identifiable using a combination of t-distributed Stochastic Neighbor Embedding (t-SNE) and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithms Bawaji et al. (2023). Here, t-SNE acts as a non-linear dimensionality reduction technique that maps high-dimensional embeddings to 2D or 3D spaces for visual diagnostics.

Radio burst observations present a disk-integrated spectral response of the solar atmosphere. These signatures arise due to deceleration of accelerated electrons, and have a peculiar radio spectrum. Identifying and classifying these bursts is traditionally manual, making it an ideal target for automated inference operators (I). Using You Only Look Once (YOLO)-based Zhang et al. (2024b) and GAN-based models Zhang et al. (2021), researchers have demonstrated that ML inference models (I) can successfully identify these signatures in a rather straightforward manner. Specifically, given a full spectral observation, Zhang et al. (2024b) demonstrate bounding boxes around the radio bursts, including the particular type of radio burst. Such automation processes enable a statistical study of radio and EUV signatures of events occurring in the heliosphere.

In a similar vein, reconnection events which possibly give rise to the non-thermal electrons causing radio bursts can be simulated using kinetic plasma simulations. Unsupervised methods like Bayesian Gaussian Mixture models model electron velocity distribution as a sum of thermal and non-thermal components, effectively identifying non-thermal electrons in simulations Yao et al. (2022). Such physicsinformed modeling methods make the model transparent, and present accurate constraints on the physics.

Apart from such drastic transient events, travelling waves are very common in the solar atmosphere. Recurrent networks applied to IRIS Doppler time-series characterize wave physics in the transition region by identifying 25–40 second damping lags between chromospheric and transition region bright point oscillations — demonstrating that learned temporal sequence analysis can extract physically meaningful propagation timescales Yuan et al. (2026). Similarly, CNN-LSTM models can differentiate between images with/without kink oscillations in the solar corona Belov et al. (2025). In such methods, a t-SNE visualization of the LSTM embeddings clearly differentiates between the two classes naturally, providing support in favor of model learning the physics in some sense.

While clustering strategies map standard atmospheric regimes into discrete physical states, they are fundamentally blind to extreme physical anomalies and rare transient phenomena—such as intense localized heating, high-velocity jets, or smallscale reconnection events—which are often lost when ML models optimize for average statistical trends. To isolate these highly informative physical extremes, unsupervised density estimators can be deployed. For example, Inspectorch D´ıaz Baso et al. (2026) operates across heterogeneous solar data archives (including Hinode/SP, IRIS, SST/CHROMIS, SDO/AIA, and Solar Orbiter/EUI) to assign probabilistic anomaly scores directly to observed profiles. Rather than relying on manual catalogs, this framework isolates physically significant outliers—such as chromospheric profiles with uncommon broadening, extreme Doppler shifts, or transient EUI coronal brightenings—offering a systematic way to filter massive datasets and focus computational analysis on the most energetic and rare physical processes.

## 1.6 Observation-space operators: instrument correction, cross-mission harmonization, and instrument design

This family of applications cuts across traditional physical layers. It operates instead as the mathematical link that harmonizes heterogeneous data archives, synthesizes virtual instruments, and learns unified representation spaces. Rather than modeling temporal evolution, these models construct learned mapping operators $( \mathcal { P } _ { 1  2 } )$ that couple disparate instruments, viewpoints, wavelengths, historical archives, or even semantic descriptions.

Mathematically, these applications are formulated through coupling operators that act on the solar state vector or translate between distinct observational representations:

$$
\begin{array} { r } { \mathcal { P } _ { a  b } : \vec { S } _ { a }  \vec { S } _ { b } \quad \mathrm { o r } \quad \mathcal { P } _ { 1  2 } : \vec { O } _ { 1 } = \mathcal { O } _ { 1 } [ \vec { S } ]  \hat { \vec { O } } _ { 2 } = \mathcal { O } _ { 2 } [ \vec { S } ] , } \end{array}\tag{1.27}
$$

where the source state $\vec { S } _ { a }$ and the target state $\vec { S } _ { b }$ may represent observational encodings (e.g., detector pixel arrays, spectral profiles) or latent physical parameters (e.g., localized vector magnetic fields, plasma temperatures).

To systematically review this diverse literature, we categorize these methodologies into three distinct, interconnected thematic families defined by the mathematical nature of their mapping operations, their role within our state-vector taxonomy, and the components of the solar state vector they act upon. The first family focuses on correcting local atmospheric and optical degradations within a single instrument setup. These methods solve classical inverse problems to reverse localized physical distortions—like atmospheric turbulence, point spread function broadening, or polarimetric noise—and recover the unperturbed physical state. Crucially, this family also includes active atmospheric correction via adaptive optics. This represents the first active deployment of the control (C ) operator in solar instrumentation, using reinforcement learning to calculate optimal deformable mirror commands in real time. The second family covers cross-mission harmonization for multi-generation studies.

Rather than inverting local physical degradations, these networks act as domain-todomain translation operators $( \mathcal { P } _ { 1  2 } )$ . They map observations across distinct hardware generations (e.g., SOHO/MDI to SDO/HMI) to correct for differences in pixel grids and detector aging, establishing consistent solar climate records. The final family exploits the strong vertical thermodynamic coupling and magnetic connectivity in the solar atmosphere. These models execute cross-wavelength image translation and virtual instrument synthesis. Serving as cross-domain coupling operators $( \mathcal { P } _ { 1  2 } )$ ), they translate observations across height-dependent passbands to synthesize missing channels or predict integrated EUV irradiances.

No single instrument can observe the full solar state, making cross-representation translation crucial for placing separately observed subsystems into a common, unified state-space framework. However, a major physical challenge in this domain is the risk of “hallucination”—generative models introducing non-physical details to satisfy mathematical loss profiles. To address this, state-of-the-art architectures increasingly incorporate physics-informed loss constraints (such as divergence-free magnetic bounds or radiative transfer consistency). They also employ rigorous cross-validation against held-out space missions. Finally, downstream scientific validations are performed to ensure that translated states represent physically plausible solar phenomena. Furthermore, standard convolutional operators applied to full-disk images inherently assume a flat grid, ignoring projection effects and spherical geometry unless explicitly corrected.

## 1.6.1 Correcting local atmospheric and optical degradations

Operating strictly within the domain of a single instrument setup, the first family of applications is formulated around reversing the localized optical distortions, atmospheric aberrations, or detector noise that degrade the measured solar state. This is depicted through modeling $\mathcal { O } _ { \mathrm { i n s t r } }$ or $\mathcal { O } _ { \mathrm { i n s t r } } ^ { - 1 }$ in Fig. 1.3.

The physical state vector $\vec { S } _ { \mathrm { d e c } }$ represents the unperturbed spatial or polarimetric scene, the wavefront phase distortions, or the mechanical deformable mirror states:

$$
\vec { S } _ { \mathrm { d e c } } : \mathcal { O } _ { p h y s } [ \vec { S } ] = \{ I , Q , U , V , \phi , \vec { a } _ { \mathrm { A O } } , . . . \} ( x , y , \lambda , t ) ,\tag{1.28}
$$

where $I , Q , U$ , and $V$ represent the four Stokes profiles containing the full polarimetric signature of the solar plasma; $\phi$ is the atmospheric wavefront phase aberration representing optical phase delays; and $\vec { a } _ { \mathrm { A O } }$ represents the mechanical actuator correction commands of an active adaptive optics system. Mathematically, the spatial degradation process can be formulated as a non-linear forward convolution of the true physical scene with a spatially varying Point Spread Function (PSF), added with polarimetric or detector noise:

$$
\vec { O } ( x , y ) = \iint \vec { S } _ { \mathrm { d e c } } ( x ^ { \prime } , y ^ { \prime } ) \cdot \mathrm { P S F } ( x - x ^ { \prime } , y - y ^ { \prime } ; x , y ) d x ^ { \prime } d y ^ { \prime } + \vec { \eta } ( x , y ) ,\tag{1.29}
$$

where $\vec { O }$ represents the degraded multi-channel observations and $\vec { \eta }$ represents stochastic noise. In this context, $M L$ acts as an inverse operator $( \mathcal { I } )$ to reconstruct the unperturbed physical state $f _ { \theta } ( \vec { O } ) \approx \vec { S } _ { \mathrm { d e c } } .$ , an emulator $( \mathcal { E } )$ to represent the convolving forward process, or a controller (C) to calculate deformable mirror actuator corrections in real time.

## 1.6.1.1 Denoising

Recovering sub-resolution solar structure from seeing-degraded observations is an ill-posed inverse problem. The convolving PSF is spatially variant, partially unknown, and entangled with stochastic noise, making analytical inversion completely unstable without strong regularization. To address detector and photon noise, selfsupervised and generative architectures have been deployed to bypass simulated ground-truth reference data. The first self-supervised approach—structurally analogous to noise-to-noise training in machine vision—exploits the spatial and spectral coherence latent in the observations themselves. By using neighboring pixels and wavelength channels as implicit mutual constraints, these models recover weak polarimetric signals at or near the detector noise floor from real observations without requiring simulated training pairs D´ıaz Baso et al. (2019). Alternatively, generative models trained on paired short- and long-exposure data replicate temporal averaging from a single noisy frame, effectively reconstructing the photon statistics of a much longer integration without sacrificing temporal cadence Park et al. (2020).

## 1.6.1.2 Spatial PSF deconvolution

To address spatial seeing aberrations, a deeper transition shifts machine learning from static supervised regression to physics-guided inversion of analytical image formation models. The key methodological insight is differentiable physics integration: by embedding classical optical diffraction theory directly into the neural loss function, the wavefront aberration becomes a latent variable optimized jointly with the scene, so the PSF need not be known in advance. This blind deconvolution approach parameterizes both the convolving wavefront and the unperturbed scene as latent variables optimized against bursts of observed frames, accelerating classical reconstructions by three orders of magnitude Asensio Ramos and Olspert (2021).

Anisoplanatism—the spatially varying turbulence that makes the PSF different at every point in a wide-field image—breaks the single-PSF assumption and produces stitching artifacts when handled patch-wise. The resolution is a continuous coordinate-based forward emulator (E) for wide-field PSF convolution that eliminates patch boundaries Asensio Ramos (2024). This has been extended to implicit neural representations. These models simultaneously optimize continuous photospheric scenes and anisoplanatic PSFs directly on ground-based data Schirninger et al. (2026) (see also Fig. 1.3). Furthermore, temporal coherence constraints enable attention-based fusion of sequential observations: explicit inter-frame alignment using deep convolutional architectures suppresses residual wavefront noise while preserving fine-scale photospheric and chromospheric structures Wang et al. (2021b); Zhang et al. (2023). For traditional supervised approaches, simulation-based inference remains a powerful tool: radiative-MHD simulations (e.g., MURaM) provide sub-resolution Stokes profiles as ground truth, enabling convolutional inverse operators $( \mathcal { I } )$ trained on synthetic pairs and validated against high-resolution Hinode observations D´ıaz Baso and Asensio Ramos (2018).

## 1.6.1.3 Spectral PSF deconvolution

Spectral and polarimetric degradations constitute a distinct class of instrumentspace distortions, separable from spatial PSF broadening and corrupting different components of the state vector. Line-profile smearing in spectrographs erases Doppler velocity and thermal line-width information, whereas non-linear saturation in filter-based magnetograph Stokes signals suppresses inferred field strengths. Spectral PSF deconvolution is the most direct solution: every spectrograph broadens line profiles through a known instrumental response, and learned convolutional surrogates trained on this forward model successfully recover the true physical line shape at science cadence, bypassing the computational cost of iterative deconvolution Molnar et al. (2020).

## 1.6.1.4 Wavefront sensing and adaptive optics control

Active compensation of atmospheric turbulence transitions ML from passive postfacto state estimation to real-time physical control. Classical adaptive optics (AO) reconstructors are linear: they invert a static interaction matrix measured during calibration to map wavefront sensor slopes to actuator commands. This linearity fails under the conditions most demanding for solar observations—extreme photon starvation on faint targets, non-stationary turbulence at daytime ground-based telescopes, and actuator cross-coupling not captured by the calibration matrix.

Transcending these limits requires a policy that can learn the non-linear actuatorto-wavefront dynamics online and adapt to changing conditions without re-calibration Nousiainen et al. (2022) established this with the Policy Optimization for Adaptive Optics (PO4AO) framework. This was the first deployment of the $\mathcal { C }$ operator in solar instrumentation. The model-based reinforcement learning policy $\pi _ { \theta } ( \vec { a } _ { \mathrm { A O } } | \vec { s } _ { \mathrm { W F S } } )$ maps pyramid wavefront sensor states directly to deformable mirror actuator commands. It learns this mapping by maximizing the Strehl ratio reward $R ( \vec { s } _ { \mathrm { W F S } } , \vec { a } _ { \mathrm { A O } } )$ Non-common-path aberrations (NCPA) represent a second class of residual that linear reconstructors cannot address by design. These are static and dynamic optical errors downstream of the wavefront sensor and are therefore invisible to it. Nousiainen et al. (2026) solved this with PO4NCPA, using sequential phase diversity to execute active focal-plane control without prior calibration. A third physical limit is phase sign ambiguity: a single intensity image is symmetric under phase conjugation, so sign-correct wavefront recovery from a single frame is underdetermined. Zhang et al. (2025) constrained this through multispectral phase-shift measurements, while Zhang et al. (2026) resolved it explicitly via the Physically Optimized Single-Image Sensorless (POSS) method, embedding physics-guided sign constraints into a lightweight Residual Network (ResNet-18) to achieve millisecond phase diversity accuracy.

## 1.6.1.5 Saturation artifact correction

During high-energy flaring events, extreme EUV and X-ray emission frequently saturates the CCD pixels of space-based imagers, leading to severe information loss in the flaring cores. Rather than relying on simple linear interpolation or discarding the data, deep learning models exploit spatial correlations from surrounding, non-saturated pixels to reconstruct peak flare intensities at full spatial and temporal resolution Yu et al. (2022). This enables accurate thermodynamic and magnetic diagnostics even within the most extreme, energetic regimes of solar active regions.

Deconvolution, denoising, and wavefront control act as structured inverse operators $( { \mathcal { I } } ) .$ , emulators (E) or controllers (C) that restore the unperturbed solar state vector under optical degradation. However, their physical accuracy remains bounded by polarimetric noise floors and calibration limits. No super-resolution network can transcend these hardware limits. Visual sharpness in super-resolved magnetograms is physically insufficient; the generative operator $( \mathcal { P } _ { 1  2 } )$ must strictly preserve magnetic flux conservation to avoid hallucinating unphysical fields. The deepest open problem is not spatial fidelity but physical uniqueness: guaranteeing stable, sign-correct wavefront solutions under anisoplanatic wide-field turbulence and photon starvation is a constraint that current reinforcement learning controllers do not yet satisfy.

## 1.6.2 Temporal consolidation and archive homogenization

While deconvolution and denoising operate within a single instrument to remove localized optical distortions, the second family maps observed solar states across distinct hardware platforms, space missions, or generational baselines. This domain is defined by the homogenized physical state vector $\vec { S } _ { \mathrm { c a l } } .$

$$
\vec { S } _ { \mathrm { c a l } } = \{ B _ { \mathrm { L O S } } , \vec { B } _ { \mathrm { p h o t } } , I _ { \mathrm { c o n t } } , \ldots \} _ { \mathrm { h o m } } ( x , y , t ) ,\tag{1.30}
$$

where $B _ { \mathrm { L O S } }$ represents the line-of-sight magnetic field, $\vec { B } _ { \mathrm { p h o t } }$ the vector magnetic field, $I _ { \mathrm { c o n t } }$ the continuum intensity, and the subscript “hom” denotes a standardized spatial scale, detector sensitivity, and calibration baseline. Because these networks map a source instrument’s observational space to a target instrument’s standard without a physical forward operator, they are formulated as cross-domain coupling

operators $\mathcal { P } _ { 1  2 } \mathrm { : }$

$$
\mathcal { P } _ { 1  2 } : \vec { O } _ { 1 } = \mathcal { O } _ { 1 } [ \vec { S } _ { \mathrm { c a l } } ]  \hat { \vec { O } } _ { 2 } = \mathcal { O } _ { 2 } [ \vec { S } _ { \mathrm { c a l } } ] ,\tag{1.31}
$$

correcting for generational differences in pixel grids, spectral responses, and sensor aging to establish consistent, multi-decade solar climate archives.

## 1.6.2.1 Multi-mission standardization and domain homogenization

Solar cycle studies, irradiance reconstructions, and space climate models require magnetic and intensity baselines that span multiple instrument generations—a requirement that no single mission satisfies. The observational gaps, spatial resolution jumps, and calibration discontinuities between successive missions introduce systematic offsets that can masquerade as real solar variability if left uncorrected. The core methodological difficulty is supervision: co-temporal image pairs from two instruments are often unavailable because the instruments operated in different eras or at incompatible cadences.

Unpaired domain translation resolves this by learning mappings from unpaired datasets rather than paired examples. It uses adversarial training with Cycle-Consistent Generative Adversarial Networks (CycleGANs). These networks learn to translate images between two domains without co-temporal training pairs. They enforce cycle consistency: translating an image from domain A to B and back to A must reconstruct the original image Jarolim et al. (2025b).

The SOHO/MDI–to–SDO/HMI transition is the most consequential instrument gap in the modern magnetogram record, separating 16 years of continuous monitoring from the high-cadence, high-resolution HMI era. Bridging it requires resolving not only spatial resolution differences but also systematic field-strength biases, detector calibration offsets, and non-stationary drifts caused by sensor aging and orbital variations over the mission lifetimes. The key advance over generic superresolution is embedding known physical degradation kernels into the learned inverse mapping. By including the measured PSF, pixel scale, and calibration curve, the network corrects for physically characterized distortions rather than inferring them from data alone. Validation against scientific benchmarks (such as flux distributions and active region complexity) rather than visual metrics then tests whether the recovered states support downstream physical analysis Munoz-Jaramillo et al.˜ (2024). Temporal gaps require a different strategy. Denoising Diffusion Probabilistic Models (diffusion models) generate data by learning to reverse a gradual noise-injection process, preserving sharp high-frequency details and statistical textures. These models are deployed to interpolate sequences across gaps, preserving dynamic flux evolution rather than blending adjacent frames Li et al. (2025a). Additionally, neural networks can estimate absolute magnetic flux directly from continuum intensity images, serving as a valuable tool for observational historical flux reconstructions Bai et al. (2021).

Spatial super-resolution is a critical sub-component of this homogenization process. Rather than simply applying traditional sharpening filters, deep learning models estimate unresolved magnetic structure by utilizing priors learned from higherresolution instruments or synthetic simulations. Following early baseline enhancements of SDO/HMI D´ıaz Baso and Asensio Ramos (2018), subsequent architectures have extended this capability to both line-of-sight and vector magnetograms across HMI, MDI, GST/NIRIS, and related data products Rahman et al. (2020); Xu et al. (2024); Qin et al. (2024); Xu et al. (2025). The practical scientific question governing these operators $( \mathcal { P } _ { 1  2 } )$ is whether the recovered small-scale flux, gradients, and polarity-inversion lines are physically reliable enough to support derived calculations, such as helicity proxies or flare-prediction features.

The data-rescue problem of handling centuries of heterogeneous historical conventions is fundamentally distinct from the high-throughput processing required for modern, highly cadenced observatories. Before the era of systematic magnetograph surveys, the only available proxies for surface magnetic field were chromospheric diagnostics—Hα plage morphology, Ca II K intensity—which encode magnetic information indirectly through the plasma response to concentrated flux. Plage regions are areas of enhanced chromospheric brightness co-spatial with concentrated photospheric magnetic flux; their historical brightness in Hα and Ca II K filtergrams thus serves as the primary magnetic field proxy for the pre-magnetograph era. Reconstructing a pseudo-magnetogram from these proxies is a physical wager: it is meaningful only if the magnetic-to-chromospheric coupling is sufficiently regular to invert statistically, an assumption that depends on activity level and spatial scale. Conditional GANs trained chronologically on co-temporal Hα and HMI pairs can synthesize pseudo-magnetograms while preserving magnetic polarity consistency, extending quantitative magnetic field estimates into the ground-based photographic archive Gao et al. (2023). The same chromospheric-to-magnetic inversion holds for Ca II K filtergrams, where it demonstrates that historical archives spanning decades before the HMI era can serve as quantitative proxies for magnetic field evolution across past solar cycles—provided the chromospheric coupling remains within the regime the training data sampled Shin et al. (2020).

Cross-instrument calibration and domain homogenization act as coupling operators $( \mathcal { P } _ { 1  2 } )$ whose primary contribution is the reconstruction of standardized multidecade solar climate records across sensor generations. Two fundamental challenges remain open: spatial super-resolution of magnetograms offers no guarantee of magnetic flux conservation, and unpaired cross-sensor domain mappings risk introducing polarity inconsistencies that propagate undetected into long-term solar cycle analyses.

## 1.6.3 Virtual instruments

The third family of applications exploits the vertical thermodynamic coupling and magnetic connectivity that structures the solar atmosphere from the photosphere to the corona. Rather than mapping the same physical state across different instruments, these models map the solar state across entirely different wavelengths, heights, or spectral profiles. The state vector $\vec { S } _ { \mathrm { w a v e } }$ represents this multi-wavelength, stratified atmospheric profile:

$$
\vec { S } _ { \mathrm { w a v e } } = \{ \vec { B } _ { \mathrm { p h o t } } , I _ { \mathrm { C a I I K } } , I _ { \mathrm { H } \alpha } , I _ { \mathrm { E U V } } ( \lambda _ { i } ) , F _ { \mathrm { E U V } } , \ldots \} ( x , y , t ) ,\tag{1.32}
$$

where $\vec { B } _ { \mathrm { p h o t } }$ represents the photospheric vector magnetic field, $I _ { \mathrm { C a I I K } }$ and $I _ { \mathrm { H } \alpha }$ the chromospheric Ca II K and Hα line intensities, $I _ { \mathrm { E U V } } ( \lambda _ { i } )$ the spatial extreme ultraviolet (EUV) coronal maps at wavelength $\lambda _ { i } ,$ and $F _ { \mathrm { E U V } }$ the integrated full-disk spectral irradiance. These applications are formulated as cross-domain coupling operators $\mathcal { P } _ { 1  2 }$ that synthesize unobserved channels, or as inference operators $\mathcal { I } _ { \mathrm { w a v e } }$ that map spatial features to integrated spectral irradiances:

$$
\mathcal { I } _ { \mathrm { w a v e } } : \vec { O } _ { \mathrm { A I A } } = \mathcal { O } _ { \mathrm { A I A } } [ \vec { S } _ { \mathrm { w a v e } } ] \to F _ { \mathrm { E U V } } ( t ) ,\tag{1.33}
$$

creating “virtual instruments” that substitute for absent, damaged, or saturated hardware.

## 1.6.3.1 Spectral channel proxy prediction

The foundational physical premise of this family is that the solar atmosphere is a vertically stratified, thermodynamically coupled system: the photospheric magnetic field governs the coronal plasma topology, and the radiative coupling between heights means that observations at one wavelength contain latent information about other atmospheric layers. By exploiting these physical correlations as learned cross-domain mappings $( \mathcal { P } _ { 1  2 } )$ , generative models can synthesize coronal or chromospheric diagnostics from photospheric inputs without physical instruments. To achieve this, the generic encoder-aggregator-forecast pattern provides a highly reusable architectural template. Individual encoders extract features from diverse, multi-source solar data. An aggregator then unifies them into a single latent state, and a forecast head maps them to the target observation. For instance, models can cross-calibrate and re-interpret SDO/HMI magnetic field data by mapping it to Hinode/SOT equivalents, lowering the noise floor and resolving disambiguation issues Higgins et al. (2022); Wang et al. (2024). However, as demonstrated by Fouhey and Leka (2025), these inter-instrument mappings are highly brittle; minute spatial co-registration or alignment discrepancies between the source instruments will severely corrupt the target vector field.

The first demonstration that this coupling is statistically learnable established that conditional GANs and L1-loss optimized CNNs can map line-of-sight HMI magnetic boundary conditions to multi-wavelength AIA UV and EUV images, with adversarial loss preserving the fine spatial contrast of coronal loop structures that pixel-wise losses blur away Park et al. (2019). A systematic quantification of the physical limits of this learned coupling reveals that spatial morphologies and thermal brightness covariances are faithfully recovered over three orders of magnitude under quiet and weakly active conditions.

The adversarial loss that makes GAN-synthesized images visually sharp is simultaneously the mechanism that can hallucinate physically inconsistent coronal structures: the discriminator penalizes lack of realism, not violation of radiative transfer or magnetic consistency. This architectural tension—between perceptual quality and physical validity—is the organizing design question of the field. Resolving it in favor of physical validity often means replacing adversarial discriminators with adaptive instance modulation architectures. These eliminate spurious high-frequency features at the cost of some spatial sharpness Sayez et al. (2025). Historical reconstructions push these mappings backward in time: pseudo-EUV (304 A) images synthesized from HMI magnetograms extend access to chromo-<sup>˚</sup> spheric dynamics into the pre-SDO archive Dash et al. (2021), while Hα-to-He I 10830 A translation extends chromospheric diagnostics across more than a century<sup>˚</sup> of ground-based records Marena et al. (2025).

## 1.6.3.2 Solar irradiance prediction

When hardware spectrometers are absent, damaged, or suffer severe degradation, predicting the integrated extreme ultraviolet (EUV) solar spectral irradiance from spatial imager observations becomes operationally critical. AIA narrowband coronal images contain sufficient information to reconstruct the full EUV solar spectral irradiance with relative errors below 4%, completely bypassing expensive Differential Emission Measure (DEM) inversions Szenicer et al. (2019). This principle has been successfully deployed to compensate for the SDO/EVE/MEGS-A hardware failure, with earlier historical precedents predicting SDO/EVE EUV irradiances using active region magnetic filling factors and radial basis function neural networks Colak and Qahwaji (2013). Extending irradiance reconstructions further into the pre-SDO era, Bayesian frameworks (I) applied to low-fidelity, coarse chromospheric proxies (such as Ca II K and SOHO/SEM) successfully reconstruct full-disk coronal EUV emissions with rigorous, physics-based uncertainty bounds Jiang et al. (2025).

Cross-wavelength translation and virtual instrument synthesis rest on the physical wager that the thermodynamic coupling between atmospheric layers is regular enough for a statistical model to invert—a wager that demonstrably fails during energetic flares, when the atmospheric state departs furthest from the training manifold. Whether spatial imagers, which project the full emission measure distribution onto a handful of broadband channels, carry the causal information needed to uniquely substitute for a physical spectrometer remains an unresolved foundational question.

## 1.6.4 Instrument design

A unifying physical insight underlies this group of applications: the solar atmospheric state, despite being sampled across dozens of spectral channels or hundreds of wavelength points, occupies a manifold of much lower intrinsic dimensionality than the raw observational space. Because the radiative transfer processes that populate each passband are strongly coupled through a shared set of thermodynamic and magnetic parameters—temperature, density, velocity, and field strength as functions of height—the multi-channel data cube is inherently redundant. The full observational vector $\vec { O }$ can often be well approximated by a projection onto a compact latent representation ⃗z with dim $( \vec { z } ) \ll \dim ( \vec { O } )$ . ML exploits this structure along three complementary axes: spectral compression, replacing costly full-profile diagnostics with compact latent encodings; virtual instrument synthesis, reconstructing missing or degraded channels from the remainder; and observational design, selecting non-redundant wavelengths that maximize physical information per collected photon.

This redundancy has been quantified empirically across the AIA channel set: training models to predict each passband from all possible subsets of the others reveals that three channels—131, 1600, and 304 A—span the thermal vertical structure<sup>˚</sup> well enough to recover the full set under quiet conditions. However, cross-passband translations systematically fail during massive flares because the plasma is driven far outside the thermodynamic manifold sampled during quiet-Sun training Lim et al. (2021); Salvatelli et al. (2022). The same compressibility holds in high-resolution line spectroscopy: a 4-dimensional autoencoder latent space faithfully represents the 110-dimensional Mg II k profile space sampled by IRIS, with a compression factor of $\sim 2 7$ and sub-noise residuals, where the latent axes correlate directly with physical diagnostics such as line widths and asymmetries Sadykov et al. (2021). Instrument compression models impose specific physical constraints on the latent manifold, distinguishing standard autoencoders from strictly invertible normalizing flows.

Beyond compressing existing data, understanding this intrinsic dimensionality provides a principled basis for optimizing future observational design. Quantitative analyses of how spectral resolution affects recoverable information content in solar Stokes spectra allow for the deliberate optimization of observing strategies and ML input grids D´ıaz Baso et al. (2025b). The most consequential application of this approach is the direct, non-uniform optimization of wavelength sampling. Rather than distributing exposure time uniformly across a spectral range, neural networks can select optimized wavelength grids for Fabry-Perot interferometers that maxi-´ mize Fisher information regarding physical atmospheric parameters, thereby significantly improving spectro-polarimetric inversions without requiring additional photons D´ıaz Baso et al. (2023).

## 1.7 Conclusion and future directions

In this review, we have redefined the Sun through the lens of a state vector characterized by partial observability. This is motivated by the presence of phenomena varying at a range of scales, while at the same time interacting with each other. Through the operator framework $( \mathcal { I } , \mathcal { P } , \mathcal { E } , \mathcal { C } )$ , we can compare methods across domains while preserving physical meaning. Such a definition emphasizes how ML serves not merely as an algorithmic optimization, but as a tool for formal state estimation, uncertainty quantification, and dynamical modeling. By mapping the literature to this taxonomy, several synthesis conclusions emerge regarding the current limits and successes of data-driven heliophysics:

The solar interior: Inference operators for helioseismic and simulation-backed inversions now recover subsurface flow and latent magnetic information at scales that were previously computationally prohibitive. However, the dominant limitation is no longer network capacity, but the domain mismatch between synthetic training environments and real observations. Furthermore, for solar-cycle and dynamo applications, pure sequence learning remains sample-limited. The most credible path is hybrid data-assimilation and physics-constrained forecasting rather than purely black-box extrapolation.

The solar atmosphere: Radiative transfer, especially in the optically thick regime, is a computationally expensive calculation. Hence, it experiences bottlenecks in both the forward and the inverse operation. Here, ML has proved important as a physics surrogate engine, and as an inverse approximation, including uncertainty estimation. Following from radiative transfer, ML has also been successfully used to generate 3-D representation of the magnetic field, approximating the forward operator, and generating accurate solutions at very high time cadence. We note that most of the applications of ML are generally tied to very specific data, and the application has been very neatly separated out into different radiation regimes, and between the magnetism - radiation domains. A comprehensive understanding necessitates development of methods which are learning a general operator, and not extremely tailored to specific data, which would need to be retrained for different data modality / instrument.

Coherent structures and transients: ML has been a powerful tool in detecting structures and transients across a range of length, time and energy scales. The identification strategies have moved past simple detection to physically structured characterization. This enables development of statistical studies of these structures, presenting power constraints on the mass and energy transport across the solar atmosphere. The physics understanding from these ML applications has resulted in the field moving from a rather discrete set of labels towards a more continuous paradigm. The transition toward unsupervised density estimators provides a systematic way to filter massive datasets without losing the highly informative, rare physical extremes that drive solar transients.

• The observation space: ML has been used extensively in standard data reduction and calibration methodologies, including removal mechanisms of instrumental artifacts and for instrument design. A significant application of ML has been as a cross-instrument calibrator and for development of virtual instruments in domains with either lost coverage or no coverage. However, such translation and restoration operators are scientifically useful only when validated in physics space, not just in image space. Better visual perceptual quality is insufficient if flux conservation, polarity consistency, thermodynamic plausibility, or downstream forecast skill are degraded. The field urgently needs harmonized validation frameworks and reproducible benchmark datasets that allow fair model comparison across missions and epochs.

These points lead to a practical agenda for the next phase of solar ML:

• Move from point estimates to probabilistic operators with calibrated posteriors and explicit failure modes: Given that the forward problem in solar physics is always some form of projection operation, probabilistic estimate of the full distribution of the estimated parameters is needed from future applications of ML.

• Prioritize hybrid physics-ML formulations that encode symmetries, conservation constraints, and observation geometry: Explicit encoding of known physical constraints is imperative in ML modeling of the Sun.

• Design models and datasets for non-stationarity: The instruments performing measurements show a general degradation in time. Even if the instruments would be perfect, the solar cycle is not the same every time. This renders the system potentially non-ergodic, and requires adaptation of model to changing times (Jha et al., 2024)

• Treat interpretability as a physical requirement: The learned features must be testable against known mechanisms and hypothesis, and not just visually plausible (Upendran et al., 2020).

• Close the research-to-operations loop with transparent reporting standards, robust all-clear evaluation, and stakeholder-specific metrics.

The goal is no longer to show that ML can emulate a task, but to establish which learned operators remain physically valid under distribution shift, sparse supervision, and incomplete observability. This brings us to multi-modal foundation models, which are trained on large sets of multimodal data, and can generate powerful representations of data by cross-correlating across space, time and wavelength. Pioneering efforts such as Solaris (Majid et al., 2024), and the recently developed Surya foundation model (Roy et al., 2025) leverage massive, multi-instrument archives to learn unified, physics-aware representations. Such models are then very quickly fine-tuned for any other downstream tasks, and demonstrate that this transition toward fully coupled, data-driven modeling of the Sun-to-Earth system is already successfully underway. Their their long-term value will depend on whether they improve causal understanding, uncertainty-aware forecasting, and coordinated Sun-to-Earth decision making. However, the end goal would be to have a unified embedding space, where different models would interact with each-other, working with different data modalities, and moving to an embedding space that would be a very strong approximation of the state vector that quantifies every aspect of the Sun.

Acknowledgements VU would like to acknowledge NASA for support under award number 80NSSC25K7956 in support for writing this chapter. The authors acknowledge the use of AI tools for data collection and compilation. The agentic AI pipeline is made available on Github.

## References

Alanazi, N. and Fletcher, L. (2026). Systematic time evolution of the horizontal magnetic field around the polarity inversion line in flares. MNRAS, 548(1):stag573.

Alipour, N., Safari, H., Verbeeck, C., Berghmans, D., Auchere, F., Chitta, L. P., An-\` tolin, P., Barczynski, K., Buchlin, E., Aznar Cuadrado, R., Dolla, L., Georgoulis,<sup>´</sup> M. K., Gissot, S., Harra, L., Katsiyannis, A. C., Long, D. M., Mandal, S., Parenti, S., Podladchikova, O., Petrova, E., Soubrie,´ E., Sch<sup>´</sup> uhle, U., Schwanitz, C.,¨ Teriaca, L., West, M. J., and Zhukov, A. N. (2022). Automatic detection of smallscale EUV brightenings observed by the Solar Orbiter/EUI. A&A, 663:A128.

Alshehhi, R., Hanson, C. S., Gizon, L., and Hanasoge, S. (2019). Supervised neural networks for helioseismic ring-diagram inversions. A&A, 622:A124.

Armstrong, J. A. and Fletcher, L. (2019). Fast solar image classification using deep learning and its importance for automation in solar physics. Sol. Phys., 294(6):80.

Aschwanden, M. J. (2005). Physics of the Solar Corona: An Introduction with Problems and Solutions. Springer Praxis Books. Springer Berlin, Heidelberg.

Asensio Ramos, A. (2023). Tomographic Reconstruction of the Solar K-Corona Using Neural Fields. Sol. Phys., 298(11):135.

Asensio Ramos, A. (2024). Solar multiobject multiframe blind deconvolution with a spatially variant convolution neural emulator. A&A, 688:A88.

Asensio Ramos, A., Cheung, M. C. M., Chifu, I., and Gafeira, R. (2023). Machine learning in solar physics. Living Reviews in Solar Physics, 20(1):4.

Asensio Ramos, A. and de la Cruz Rodr´ıguez, J. (2025). Neural translation for Stokes inversion and synthesis. A&A, 703:A55.

Asensio Ramos, A. and D´ıaz Baso, C. J. (2019). Stokes inversion based on convolutional neural networks. A&A, 626:A102.

Asensio Ramos, A., D´ıaz Baso, C. J., and Kochukhov, O. (2022). Approximate Bayesian neural Doppler imaging. A&A, 658:A162.

Asensio Ramos, A. and Olspert, N. (2021). Learning to do multiframe wavefront sensing unsupervised: Applications to blind deconvolution. A&A, 646:A100.

Asensio Ramos, A., Yang, K. E., Mart´ınez Gonzalez, M. J., Dodds, S. C., and Sun,´ X. (2026). 3DStokesFlow: simulation-based inference for 3D Stokes profiles using flow matching. arXiv e-prints, page arXiv:2606.13004.

Ashamari, O., Qahwaji, R., Ipson, S., Scholl, M., Nibouche, O., and Haberreiter,¨ M. (2015). Identification of photospheric activity features from SOHO/MDI data using the ASAP tool. Journal ofSpace Weather and Space Climate, 5:A15.

Athalathil, J. J., Talafha, M. H., and Vaidya, B. (2026). Investigating Nonlinear Quenching Effects on Polar Field Buildup in the Sun Using Physics-informed Neural Networks. ApJ, 1000(1):79.

Athalathil, J. J., Vaidya, B., Kundu, S., Upendran, V., and Cheung, M. C. M. (2024). Surface Flux Transport Modeling Using Physics-informed Neural Networks. ApJ, 975(2):258.

Baek, J.-H., Kim, S., Choi, S., Park, J., and Kim, D. (2024). Deep learning-based solar image captioning. Advances in Space Research, 73(6):3270–3281.

Baek, J.-H., Kim, S., Choi, S., Park, J., Kim, J., Jo, W., and Kim, D. (2021). Solar Event Detection Using Deep-Learning-Based Object Detection Methods. Sol. Phys., 296(11):160.

Bahauddin, S. M. and Rast, M. P. (2021). Identifying Acoustic Wave Sources on the Sun. I. Two-dimensional Waves in a Simulated Photosphere. ApJ, 915(1):36.

Bai, H., Yang, P., Zhao, L., Gong, X., Zhong, L., Yang, Y., and Rao, C. (2023). Hybrid Detection Algorithm and Study on the Quantity and Brightness Evolution Characteristics of Photospheric Bright Point Groups. ApJ, 956(1):62.

Bai, L., Bi, Y., Yang, B., Hong, J.-C., Xu, Z., Shang, Z.-H., Liu, H., Ji, H.-S., and Ji, K.-F. (2021). Predicting the evolution of photospheric magnetic field in solar active regions using deep learning. Research in Astronomy and Astrophysics, 21(5):113.

Bai, X., Liu, H., Deng, Y., Jiang, J., Guo, J., Bi, Y., Feng, T., Jin, Z., Cao, W., Su, J., and Ji, K. (2021). A deep learning method to estimate magnetic fields in solar active regions from photospheric continuum images. A&A, 652:A143.

Bandyopadhyay, S., Das, S., and Datta, A. (2020). Comparative study and development of two contour-based image segmentation techniques for coronal hole detection in solar images. Sol. Phys., 295(8):110.

Barnes, W. T., Bradshaw, S. J., and Viall, N. M. (2021). Understanding heating in active region cores through machine learning. ii. classifying observations. ApJ, 919(2):132.

Bawaji, S., Alam, U., Mondal, S., Oberoi, D., and Biswas, A. (2023). An unsupervised machine learning-based algorithm for detecting weak impulsive narrowband quiet sun emissions and characterizing their morphology. ApJ, 954(1):39.

Bazarghan, M., Safari, H., Innes, D. E., Karami, E., and Solanki, S. K. (2008). A nanoflare model for active region radiance: application of artificial neural networks. A&A, 492(1):L13–L16.

Belov, S. A., Zhong, Y., Kolotkov, D. Y., and Nakariakov, V. M. (2025). Detection of kink oscillations in solar coronal loops by a convolutional neural network-long short-term memory neural network. ApJS, 281(1):12.

Benson, B., Pan, W. D., Prasad, A., Gary, G. A., and Hu, Q. (2020). Forecasting Solar Cycle 25 Using Deep Neural Networks. Sol. Phys., 295(5):65.

Benson, B., Pan, W. D., Prasad, A., Gary, G. A., and Hu, Q. (2021). On the estimation of the sharp parameter meanalp from aia images using deep neural networks. Sol. Phys., 296(11):163.

Berghmans, D., Auchere, F., Long, D. M., Soubri\` e, E., Mierla, M., Zhukov,´ A. N., Schuhle, U., Antolin, P., Harra, L., Parenti, S., Podladchikova, O., Aznar¨

Cuadrado, R., Buchlin, E., Dolla, L., Verbeeck, C., Gissot, S., Teriaca, L., Haber-<sup>´</sup> reiter, M., Katsiyannis, A. C., Rodriguez, L., Kraaikamp, E., Smith, P. J., Stegen, K., Rochus, P., Halain, J. P., Jacques, L., Thompson, W. T., and Inhester, B. (2021). Extreme-UV quiet Sun brightenings observed by the Solar Orbiter/EUI. A&A, 656:L4.

Bizzarri, I., Barghini, D., Mancuso, S., Alessio, S., Rubinetti, S., and Taricco, C. (2022). Forecasting the solar cycle 25 using a multistep Bayesian neural network. MNRAS, 515(4):5062–5070.

Bobra, M., Mason, J., Argall, M. R., D´ıaz Baso, C. J., Bloch, T., Chen, Y., Cheung, M., Felipe, T., Fouhey, D., Galvez, R., Jin, M., Munoz-Jaramillo, A., Panos,˜ B., de Lima, R. P., Macneil, A. R., McInnes, L., Owens, M., Petrik, M., Ramos, A. A., Small, C., Szenicer, A., Thomas, R., Watt, C., and Wright, P. (2021). Helioml/helioml: Helioml 0.4.0 (2021-02-08). Zenodo, Version 0.4.0.

Bobra, M. G. and Couvidat, S. (2015). Solar Flare Prediction Using SDO/HMI Vector Magnetic Field Data with a Machine-learning Algorithm. ApJ, 798(2):135.

Bobra, M. G., Sun, X., Hoeksema, J. T., Turmon, M., Liu, Y., Hayashi, K., Barnes, G., and Leka, K. D. (2014). The Helioseismic and Magnetic Imager (HMI) Vector Magnetic Field Pipeline: SHARPs - Space-Weather HMI Active Region Patches. Sol. Phys., 289(9):3549–3578.

Bose, S., Henriques, V. M. J., Joshi, J., and Rouppe van der Voort, L. (2019). Characterization and formation of on-disk spicules in the Ca II K and Mg II k spectral lines. A&A, 631:L5.

Broock, E. G., Asensio Ramos, A., and Felipe, T. (2022). FarNet-II: An improved solar far-side active region detection method. A&A, 667:A132.

Broock, E. G., Asensio Ramos, A., and Felipe, T. (2024). The return of FarNet-II: Generation of solar far-side magnetograms from helioseismic data. A&A, 692:A182.

Broock, E. G., Felipe, T., and Asensio Ramos, A. (2021). Performance of solar far-side active region neural detection. A&A, 652:A132.

Brun, A. S., Gizon, L., Archontis, V., and Carlsson, M. (2019). WHOLESUN: Untangling the complex physical mechanisms behind our eruptive star and its twins. ERC Synergy Grant Project description, Grant Agreement No. 810218.

Calvo, R. A., Ceccato, H. A., and Piacentini, R. D. (1995). Neural Network Prediction of Solar Activity. ApJ, 444:916.

Campbell, R. J., Mathioudakis, M., and Quintero Noda, C. (2026). Transformers for stratified spectropolarimetric inversion: Proof of concept. ApJ, 996(1):63.

Camporeale, E., Wing, S., and Johnson, J. (2018). Machine learning techniques for space weather. Elsevier.

Canfield, R. (2000). Solar Active Regions. In Murdin, P., editor, Encyclopedia of Astronomy and Astrophysics, page 2023.

Cao, J., Xu, T., Deng, L., Zhou, X., Li, S., Liu, Y., Wang, W., and Zhou, W. (2024). An Improved Prediction of Solar Cycles 25 and 26 Using the Informer Model: Gnevyshev Peaks and North-South Asymmetry. ApJ, 969(2):120.

Carroll, T. A. and Kopf, M. (2008). Zeeman-tomography of the solar photosphere. Three-dimensional surface structures retrieved from Hinode observations. A&A, 481(1):L37–L40.

Carroll, T. A. and Staude, J. (2001). The inversion of Stokes profiles with artificia neural networks. A&A, 378:316–326.

Chappell, B. A. and Pereira, T. M. D. (2022). Sunnynet: A neural network approach to 3d non-lte radiative transfer. A&A, 658:A182.

Charbonneau, P. (2010). Dynamo Models of the Solar Cycle. Living Reviews in Solar Physics, 7(1):3.

Chen, J., Gyenge, N. G., Jiang, Y., Erdelyi, R., Liu, J., and Wang, Y. (2025a). A´ Bias-free Deep Learning Approach for Automated Sunspot Segmentation. ApJ, 980(2):261.

Chen, P. F. (2011). Coronal Mass Ejections: Models and Their Observational Basis. Living Reviews in Solar Physics, 8(1):1.

Chen, R., Lu, W., Hao, Q., Meng, Y., Chen, P. F., and Shi, C. (2025b). Statistical analyses of solar active regions in sdo/hmi magnetograms detected by the unsupervised machine learning method dsard. ApJS, 280(2):70.

Cheung, M. C. M., Boerner, P., Schrijver, C. J., Testa, P., Chen, F., Peter, H., and Malanushenko, A. (2015). Thermal Diagnostics with the Atmospheric Imaging Assembly on board the Solar Dynamics Observatory: A Validated Method for Differential Emission Measure Inversions. ApJ, 807(2):143.

Cheung, M. C. M., De Pontieu, B., Mart´ınez-Sykora, J., Testa, P., Winebarger, A. R., Daw, A., Hansteen, V., Antolin, P., Tarbell, T. D., Wuelser, J.-P., Young, P., and MUSE Team (2019). Multi-component Decomposition of Astronomical Spectra by Compressed Sensing. ApJ, 882(1):13.

Cheung, M. C. M. and Isobe, H. (2014). Flux Emergence (Theory). Living Reviews in Solar Physics, 11(1):3.

Colak, T. and Qahwaji, R. (2008). Automated mcintosh-based classification of sunspot groups using mdi images. Sol. Phys., 248(2):277–296.

Colak, T. and Qahwaji, R. (2013). Prediction of Extreme Ultraviolet Variability Experiment (EVE)/ Extreme Ultraviolet Spectro-Photometer (ESP) Irradiance from Solar Dynamics Observatory (SDO)/ Atmospheric Imaging Assembly (AIA) Images Using Fuzzy Image Processing and Machine Learning. Sol. Phys., 283(1):143–156.

Covas, E., Peixinho, N., and Fernandes, J. (2019). Neural Network Forecast of the Sunspot Butterfly Diagram. Sol. Phys., 294(3):24.

Cranmer, S. R. (2009). Coronal Holes. Living Reviews in Solar Physics, 6(1):3.

da Silva, D. E., Kirk, M. S. F., Mathews, N., and Munoz-Jaramillo, A. (2025). Dif-˜ fusion Modeling of the Three-Dimensional Magnetic Field in the Sun’s Corona. arXiv e-prints, page arXiv:2510.01441.

Dakanalis, I., Tsiropoula, G., Tziotziou, K., and Koutroumbas, K. (2021). Automated Detection of Chromospheric Swirls Based on Their Morphological Characteristics. Sol. Phys., 296(1):17.

Dang, Y., Chen, Z., Li, H., and Shu, H. (2022). A comparative study of non-deep learning, deep learning, and ensemble learning methods for sunspot number prediction. arXiv e-prints, page arXiv:2203.05757.

Dash, A., Ye, J., and Wang, G. (2021). High Resolution Solar Image Generation using Generative Adversarial Networks. arXiv e-prints, page arXiv:2106.03814.

De Wit, T. D. (2006). Fast segmentation of solar extreme ultraviolet images. Sol. Phys., 239(1-2):519–530.

Deshmukh, V., Baskar, S., Berger, T. E., Bradley, E., and Meiss, J. D. (2023). Comparing feature sets and machine-learning models for prediction of solar flares. Topology, physics, and model complexity. A&A, 674:A159.

Dhuri, D. B., Bhattacharjee, S., Hanasoge, S. M., and Kiran Mahapatra, S. (2022). Deep-learning Reconstruction of Sunspot Vector Magnetic Fields for Forecasting Solar Storms. ApJ, 939(2):64.

Dhuri, D. B., Hanasoge, S. M., Birch, A. C., and Schunker, H. (2020). Application and Interpretation of Deep Learning for Identifying Pre-emergence Magnetic Field Patterns. ApJ, 903(1):27.

Dhuri, D. B., Hanasoge, S. M., and Cheung, M. C. M. (2019). Machine learning reveals systematic accumulation of electric current in lead-up to solar flares. Proceedings ofthe National Academy ofScience, 116(23):11141–11146.

D´ıaz Baso, C. J. and Asensio Ramos, A. (2018). Enhancing SDO/HMI images using deep learning. A&A, 614:A5.

D´ıaz Baso, C. J., Asensio Ramos, A., and de la Cruz Rodr´ıguez, J. (2022). Bayesian Stokes inversion with normalizing flows. A&A, 659:A165.

D´ıaz Baso, C. J., Asensio Ramos, A., de la Cruz Rodr´ıguez, J., da Silva Santos, J. M., and Rouppe van der Voort, L. (2025a). Exploring spectropolarimetric inversions using neural fields: Solar chromospheric magnetic field under the weakfield approximation. A&A, 693:A170.

D´ıaz Baso, C. J., de la Cruz Rodr´ıguez, J., and Danilovic, S. (2019). Solar image denoising with convolutional neural networks. A&A, 629:A99.

D´ıaz Baso, C. J., Milic, I., Rouppe van der Voort, L., and Schlichenmaier, R.´ (2025b). Spectral resolution effects on the information content in solar spectra. A&A, 693:A272.

D´ıaz Baso, C. J., Rouppe van der Voort, L., de la Cruz Rodr´ıguez, J., and Leenaarts, J. (2023). Designing wavelength sampling for Fabry-Perot observa-´ tions. Information-based spectral sampling. A&A, 673:A35.

D´ıaz Baso, C. J., Soler Poquet, I. J., Kuckein, C., van Noort, M., and Poirier, N. (2026). Inspectorch: Efficient rare event exploration in solar observations. arXiv e-prints, page arXiv:2602.20316.

Ellerman, F. (1917). Solar Hydrogen “bombs”. ApJ, 46:298.

Fang, Y., Cui, Y., and Ao, X. (2019). Deep learning for automatic recognition of magnetic type in sunspot groups. Advances in Astronomy, 2019:9196234.

Felipe, T. and Asensio Ramos, A. (2019). Improved detection of far-side solar active regions using deep learning. A&A, 632:A82.

Fessant, F., Bengio, S., and Collobert, D. (1996). On the prediction of solar activity using different neural network models. Annales Geophysicae, 14(1):20–26.

Fouhey, D. and Leka, K. (2025). Learning to Estimate Photospheric Vector Fields from Line-of-sight Magnetograms. ApJ, 993(1):58.

Gafeira, R., Orozco Suarez, D., Mili´ c, I., Quintero Noda, C., Ruiz Cobo, B., and´ Uitenbroek, H. (2021). Machine learning initialization to accelerate Stokes profile inversions. A&A, 651:A31.

Gao, F., Liu, T., Sun, W., and Xu, L. (2023). Generating Space-based SDO/HMI-like Solar Magnetograms from Ground-based Hα Images by Deep Learning. ApJS, 266(2):19.

Gong, L., Yang, Y., Feng, S., Dai, W., Liang, B., and Xiong, J. (2024). Solar active regions detection and tracking based on deep learning. Sol. Phys., 299(8):121.

Gong, X., Zhong, L., and Rao, C. (2023). Automatic segmentation of the fine structures of sunspots in high-resolution solar images. A&A, 670:A132.

Hamada, A., Creelman, M., Jain, K., and Lindsey, C. (2025). FArSide Trained Active Region Recognition (FASTARR): A Machine Learning Approach. ApJS, 278(2):53.

Hamada, A., Jain, K., Strecker, H., Lindsey, C., and Orozco Suarez, D. (2026).´ Polarity-resolved far-side magnetograms based on helioseismic measurements. Scientific Reports, 16:13110.

Hannah, I. G., Hudson, H. S., Battaglia, M., Christe, S., Kasparovˇ a, J., Krucker, S.,´ Kundu, M. R., and Veronig, A. (2011). Microflares and the Statistics of X-ray Flares. Space Sci. Rev., 159(1-4):263–300.

Hannah, I. G. and Kontar, E. P. (2012). Differential emission measures from the regularized inversion of Hinode and SDO data. A&A, 539:A146.

Hartlep, T., Kosovichev, A. G., Zhao, J., and Mansour, N. N. (2011). Signatures of Emerging Subsurface Structures in Acoustic Power Maps of the Sun. Sol. Phys., 268(2):321–327.

Higgins, R. E. L., Fouhey, D. F., Antiochos, S. K., Barnes, G., Cheung, M. C. M., Hoeksema, J. T., Leka, K. D., Liu, Y., Schuck, P. W., and Gombosi, T. I. (2022). SynthIA: A Synthetic Inversion Approximation for the Stokes Vector Fusing SDO and Hinode into a Virtual Observatory. ApJS, 259(1):24.

Hoeksema, J. T., Brummell, N., Bush, R. I., Komm, R., Kosovichev, A. G., Mendez, B., Scherrer, P. H., Upton, L., Wray, A. A., and Zevin, D. (2020). COFFIES - Developing a Reliable Physical Model of the Solar Activity Cycle. AGU Fall Meeting Abstracts. Bibcode: 2020AGUFMSH0020007H.

Huang, C., Liu, D.-D., and Wang, J.-S. (2009). RESEARCH PAPER: Forecast daily indices of solar activity, F10.7, using support vector regression method. Research in Astronomy and Astrophysics, 9(6):694–702.

Illarionov, E., Kosovichev, A., and Tlatov, A. (2020). Machine-learning approach to identification of coronal holes in solar disk images and synoptic maps. ApJ, 903(2):115.

Illarionov, E. A. (2025). Machine learning methods in solar physics. Physics-Uspekhi, 68(4):374–392.

Illarionov, E. A. and Arlt, R. (2022). Reconstruction of the Solar Activity from the Catalogs of the Zurich Observatory. Sol. Phys., 297(7):79.

Inceoglu, F., Shprits, Y. Y., Heinemann, S. G., and Bianco, S. (2022). Identification of coronal holes on aia/sdo images using unsupervised machine learning. ApJ, 930(2):118.

Ishikawa, S.-n., Matsumura, H., Uchiyama, Y., and Glesener, L. (2021). Automatic detection of occulted hard x-ray flares using deep-learning methods. Sol. Phys., 296(2):39.

Ivanov, S., Tsizh, M., Ullmann, D., Panos, B., and Voloshynovskiy, S. (2021). Solar activity classification based on Mg II spectra: Towards classification on compressed data. Astronomy and Computing, 36:100473.

Jang, S., Kwon, R.-Y., Linker, J. A., Riley, P., Shin, G., Downs, C., and Kim, Y.- H. (2021). Development of a deep learning model for inversion of rotational coronagraphic images into 3d electron density. ApJ, 920(2):L30.

Jarolim, R., Molnar, M. E., Tremblay, B., Centeno, R., and Rempel, M. (2025a). Pinn me: A physics-informed neural network framework for accurate milneeddington inversions of solar magnetic fields. ApJ, 985:L7.

Jarolim, R., Thalmann, J. K., Veronig, A. M., and Podladchikova, T. (2023). Probing the solar coronal magnetic field with physics-informed neural networks. Nature Astronomy, 7:1171–1179.

Jarolim, R., Tremblay, B., Rempel, M., Molnar, M., Veronig, A. M., Thalmann, J. K., and Podladchikova, T. (2024). Advancing Solar Magnetic Field Extrapolations through Multiheight Magnetic Field Measurements. ApJ, 963(1):L21.

Jarolim, R., Veronig, A. M., Hofmeister, S., Heinemann, S. G., Temmer, M., Podladchikova, T., and Dissauer, K. (2021). Multi-channel coronal hole detection with convolutional neural networks. A&A, 652:A13.

Jarolim, R., Veronig, A. M., Potzi, W., and Podladchikova, T. (2025b). A deep¨ learning framework for instrument-to-instrument translation of solar observation data. Nature Communications, 16(1):3157.

Jeon, M., Jeong, H.-J., Moon, Y.-J., Kang, J., and Kusano, K. (2025). Real-time extrapolation of nonlinear force-free fields from photospheric vector magnetic fields using a physics-informed neural operator. ApJS, 277(2):54.

Jeong, H.-J., Jeon, M., Kim, D., Kim, Y., Baek, J.-H., Moon, Y.-J., and Choi, S. (2025a). Prediction of the Next Solar Rotation Synoptic Maps Using an Artificial Intelligence-based Surface Flux Transport Model. ApJS, 278(1):5.

Jeong, H.-J., Moon, Y.-J., Park, E., and Lee, H. (2020). Solar coronal magnetic field extrapolation from synchronic data with ai-generated farside. ApJ, 903(2):L25.

Jeong, H.-J., Park, E., Lee, H., Youn, J., Jeon, M., Moon, Y.-J., Poedts, S., Carella, F., Wang, H., Kim, D., Kim, Y., and Kang, J. (2025b). Artificial-intelligencebased reconstruction of solar farside vector magnetograms from multispacecraft extreme-ultraviolet data. ApJS, 281(2):63.

Jeong, H.-J., Poedts, S., Wang, H., Dineva, E., Gonidakis, P., Carella, F., Miloshevich, G., Linan, L., and Lee, H. (2025c). Prediction of time-evolving radial magnetic fields on the solar surface using deep learning. ApJ, 989(2):L31.

Jha, B. K., Pandey, C., Issan, O., Jackson, I., Heyns, M., Mukundan, R., Upendran, V., Ferdousi, B., Tigas, P., Lavin, A., and Saleem, H. (2024). Geo-CLoak: Operational Machine Learning Tool for Global Geomagnetic Field Perturbation

Forecasting. In AGU Fall Meeting Abstracts, volume 2024 of AGU Fall Meeting Abstracts, pages SM11C–06.

Jiang, H., Jing, J., Wang, J., Liu, C., Li, Q., Xu, Y., Wang, J. T. L., and Wang, H. (2021). Tracing hα fibrils through bayesian deep learning. ApJS, 256(1):20.

Jiang, H., Li, Q., Wang, J. T. L., Wang, H., and Criscuoli, S. (2025). Reconstruction of Solar Extreme-ultraviolet Irradiance Using Ca II K Images and SOHO/SEM Data with Bayesian Deep Learning and Uncertainty Quantification. ApJS, 280(2):50.

Karak, B. B. (2026). Solar cycle prediction: challenges, progress, and future perspectives. Reviews ofModern Plasma Physics, 10(1):11.

Kasapis, S., Kitiashvili, I. N., Kosovichev, A. G., and Stefan, J. T. (2025). Prediction of Intensity Variations Associated with Emerging Active Regions using Helioseismic Power Maps and Machine Learning. ApJS, 280(2):64.

Kashyap, V. and Drake, J. J. (1998). Markov-Chain Monte Carlo Reconstruction of Emission Measure Distributions: Application to Solar Extreme-Ultraviolet Spectra. ApJ, 503(1):450–466.

Kianfar, S., Leenaarts, J., Danilovic, S., de la Cruz Rodr´ıguez, J., and D´ıaz Baso, C. J. (2020). Physical properties of bright Ca II K fibrils in the solar chromosphere. A&A, 637:A1.

Kim, T., Park, E., Lee, H., Moon, Y.-J., Bae, S.-H., Lim, D., Jang, S., Kim, L., Cho, I.-H., Choi, M., and Cho, K.-S. (2019). Solar farside magnetograms from deep learning analysis of STEREO/EUVI data. Nature Astronomy, 3:397–400.

Klimchuk, J. A. (2015). Key aspects of coronal heating. Philosophical Transactions ofthe Royal Society ofLondon Series A, 373(2042):20140256–20140256.

Kosovichev, A., Basu, S., Bekki, Y., Buitrago-Casas, J. C., Chatzistergos, T., Chen, R., Christensen-Dalsgaard, J., Donea, A., Fleck, B., Fournier, D., Garc´ıa, R. A., Getling, A., Gizon, L., Gough, D. O., Hanasoge, S., Hanson, C. S., Hess Webber, S. A., Hoeksema, J. T., Howe, R., Jain, K., Kasapis, S., Kashyap, S. G., Kitiashvili, I. N., Komm, R., Korzennik, S. G., Krivova, N. A., Kuhn, J. R., Liang, Z.-C., Lindsey, C., Mahajan, S. S., Mandal, K., Mani, P., Martinez Oliveros, J. C., Mathur, S., Rabello Soares, M. C., Rajaguru, S. P., Reiter, J., Rhodes, E. J., Rozelot, J.-P., Scherrer, P. H., Solanki, S. K., Stefan, J. T., Toomre, J., Tripathy, C., Upton, L. A., and Zhao, J. (2025). Structure and Dynamics of the Sun’s Interior Revealed by the Helioseismic and Magnetic Imager. Sol. Phys., 300(5):70.

Kostic, T., Mili ´ c, I., Rempel, M., Welsch, B., Kazachenko, M., Asensio Ramos, A.,´ and Tremblay, B. (2026). Multi-height probing of horizontal flows in the solar photosphere. A&A, 705:A44.

Kuckein, C., Gonzalez Manrique, S. J., Kleint, L., and Asensio Ramos, A. (2020).´ Determining the dynamics and magnetic fields in He I 10830 A during a solar<sup>˚</sup> filament eruption. A&A, 640:A71.

Lagg, A., Solanki, S. K., van Noort, M., and Danilovic, S. (2014). Vigorous convection in a sunspot granular light bridge. A&A, 568:A60.

Legnaro, E., Guastavino, S., Piana, M., and Massone, A. M. (2025). Deep Learning for Active Region Classification: A Systematic Study from Convolutional Neura Networks to Vision Transformers. ApJ, 981(2):157.

Leka, K. D., Wagner, E. L., Upton, L., Jha, B. K., Jain, K., and Petty, S. (2026). Addressing Known Challenges in Solar Flare Forecasting I: Limb-Flare Prediction With a 4π Full-Heliosphere Framework. Space Weather, 24(2):e2025SW004582.

Lennard, M. G., Silva, S. S. A., Tremblay, B., Ramos, A. A., Verth, G., Ballai, I., Iijima, H., Hotta, H., Rempel, M., Park, S.-H., and Fedun, V. (2025). Recovering coherent flow structures in active regions using machine learning. MNRAS, 539(3):2498–2512.

Li, J., Yurchyshyn, V., Wang, J. T. L., Wang, H., Abduallah, Y., Alobaid, K. A., Xu, C., Chen, R., and Xu, Y. (2025a). Improving the Temporal Resolution of SOHO/MDI Magnetograms of Solar Active Regions Using a Deep Generative Model. ApJ, 980(2):228.

Li, Q., Shen, B., Jiang, H., Yurchyshyn, V. B., Baildon, T., Yi, K., Cao, W., and Wang, H. (2025b). Mvpinn: Integrating milne-eddington inversion with physics-informed neural networks for gst/niris observations. arXiv e-prints, page arXiv:2507.09430.

Li, Q., Wan, M., Zeng, S.-G., Zheng, S., and Deng, L.-H. (2021). Predicting the 25th solar cycle using deep learning methods based on sunspot area data. Research in Astronomy and Astrophysics, 21(7):184.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. (2020). Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895.

Lim, D., Moon, Y.-J., Park, E., and Lee, J.-Y. (2021). Selection of Three (Extreme)Ultraviolet Channels for Solar Satellite Missions by Deep Learning. ApJ, 915(2):L31.

Liu, H., Xu, Y., Wang, J., Jing, J., Liu, C., Wang, J. T. L., and Wang, H. (2020). Inferring vector magnetic fields from stokes profiles of gst/niris using a convolutional neural network. ApJ, 894(1):70.

Madjarska, M. S. (2019). Coronal bright points. Living Reviews in Solar Physics, 16(1):2.

Majid, H. A., Sittoni, P., and Tudisco, F. (2024). Solaris: A foundation model of the sun. arXiv preprint arXiv:2411.16339.

Mani, P., Hanson, C. S., Dhanpal, S., Hanasoge, S., Das, S. B., and Rempel, M. (2024). Magnetic Flux in the Sun Emerges Unaffected by Supergranular-scale Surface Flows. ApJ, 965(2):186.

Marena, M., Li, Q., Wang, H., and Shen, B. (2025). Reconstructing He I 10830 A<sup>˚</sup> Images Using Hα Images through Deep Learning. ApJ, 984(2):99.

Masaki, H. and Hotta, H. (2024). Detection of solar internal flows with numerical simulation and machine learning. PASJ, 76(6):L33–L38.

McMullan, L., Antolin, P., Kleint, L., and Panos, B. (2026). Deepfilter: A machine learning technique for removing the hot aia 304 a channel component for the˚ analysis of coronal rain. A&A, 707:A237.

Mercea, V., Paraschiv, A. R., Lacatus, D. A., Marginean, A., and Besliu-Ionescu, D. (2023). A Machine Learning Enhanced Approach for Automated Sunquake Detection in Acoustic Emission Maps. Sol. Phys., 298(1):4.

Milic, I. and Gafeira, R. (2020). Mimicking spectropolarimetric inversions using´ convolutional neural networks. A&A, 644:A129.

Mishra, D. K., Chatterjee, S., Jha, B. K., Raju, H., Priyadarshi, A., Hegde, M., Routh, S., Banerjee, D., and Khan, M. S. (2026). Machine Learning-based Identification of the Solar Disk and Plages in Kodaikanal Solar Observatory Historical Suncharts. ApJS, 283(1):19.

Mistryukova, L., Plotnikov, A., Khizhik, A., Knyazeva, I., Hushchyn, M., and Derkach, D. (2023). Stokes inversion techniques with neural networks: Analysis of uncertainty in parameter estimation. Sol. Phys., 298(8):98.

Mkwanda, N. M., Tian, W., and Li, J. (2024). Sunspot Group Detection and Classification by Dual Stream Convolutional Neural Network Method. Research in Astronomy and Astrophysics, 24(9):095022.

Molnar, M., Reardon, K. P., Osborne, C., and Milic, I. (2020). Spectral decon-´ volution with deep learning: removing the effects of spectral PSF broadening. Frontiers in Astronomy and Space Sciences, 7:29.

Morosin, R., de la Cruz Rodr´ıguez, J., D´ıaz Baso, C. J., and Leenaarts, J. (2022). Spatio-temporal analysis of chromospheric heating in a plage region. A&A, 664:A8.

Munoz-Jaramillo, A., Jungbluth, A., Gitiaux, X., Wright, P. J., Shneider, C., Mal-˜ oney, S. A., Baydin, A. G., Gal, Y., Deudon, M., and Kalaitzis, F. (2024). Physically Motivated Deep Learning to Superresolve and Cross Calibrate Solar Magnetograms. ApJS, 271(2):46.

Nousiainen, J., Rajani, C., Kasper, M., Helin, T., Haffert, S. Y., Verinaud, C., Males,´ J. R., Van Gorkom, K., Close, L. M., Long, J. D., Hedglen, A. D., Guyon, O., Schatz, L., Kautz, M., Lumbres, J., Rodack, A., Knight, J. M., and Miller, K. (2022). Toward on-sky adaptive optics control using reinforcement learning. Model-based policy optimization for adaptive optics. A&A, 664:A71.

Nousiainen, J., Taskin, I., Kasper, M., Orban De Xivry, G., and Absil, O. (2026). Focal plane wavefront control with model-based reinforcement learning. arXiv e-prints, page arXiv:2604.00993.

O’Dwyer, B., Del Zanna, G., Mason, H. E., Weber, M. A., and Tripathi, D. (2010). SDO/AIA response to coronal hole, quiet Sun, active region, and flare plasma. A&A, 521:A21.

Okoh, D. I., Seemala, G. K., Rabiu, A. B., Uwamahoro, J., Habarulema, J. B., and Aggarwal, M. (2018). A Hybrid Regression-Neural Network (HR-NN) Method for Forecasting the Solar Activity. Space Weather, 16(9):1424–1436.

Osborne, C. M. J., Armstrong, J. A., and Fletcher, L. (2019). Radynversion: Learning to invert a solar flare atmosphere with invertible neural networks. ApJ, 873(2):128.

Panos, B., Kleint, L., Huwyler, C., Krucker, S., Melchior, M., Ullmann, D., and Voloshynovskiy, S. (2018). Identifying Typical Mg II Flare Spectra Using Machine Learning. ApJ, 861(1):62.

Panos, B., Kleint, L., and Voloshynovskiy, S. (2021). Exploring Mutual Information between IRIS Spectral Lines. I. Correlations between Spectral Lines during Solar Flares and within the Quiet Sun. ApJ, 912(2):121.

Panos, B. and Milic, I. (2025). Policy-Based Radiative Transfer: Solving the 2-Level´ Atom Non-LTE Problem using Soft Actor-Critic Reinforcement Learning. arXiv e-prints, page arXiv:2504.15679.

Parenti, S. (2014). Solar Prominences: Observations. Living Reviews in Solar Physics, 11(1):1.

Park, E., Moon, Y.-J., Lee, J.-Y., Kim, R.-S., Lee, H., Lim, D., Shin, G., and Kim, T. (2019). Generation of Solar UV and EUV Images from SDO/HMI Magnetograms by Deep Learning. ApJ, 884(1):L23.

Park, E., Moon, Y.-J., Lim, D., and Lee, H. (2020). De-noising SDO/HMI Solar Magnetograms by Image Translation Method Based on Deep Learning. ApJ, 891(1):L4.

Parker, E. N. (1988). Nanoflares and the Solar X-Ray Corona. ApJ, 330:474.

Peter, H., Tian, H., Curdt, W., Schmit, D., Innes, D., De Pontieu, B., Lemen, J., Title, A., Boerner, P., Hurlburt, N., Tarbell, T. D., Wuelser, J. P., Mart´ınez-Sykora, J., Kleint, L., Golub, L., McKillop, S., Reeves, K. K., Saar, S., Testa, P., Kankelborg, C., Jaeggli, S., Carlsson, M., and Hansteen, V. (2014). Hot explosions in the cool atmosphere of the Sun. Science, 346(6207):1255726.

Plowman, J. and Caspi, A. (2020). A Fast, Simple, Robust Algorithm for Coronal Temperature Reconstruction. ApJ, 905(1):17.

Prasad, A., Roy, S., Sarkar, A., Chandra Panja, S., and Narayan Patra, S. (2022). Prediction of solar cycle 25 using deep learning based long short-term memory forecasting technique. Advances in Space Research, 69(1):798–813.

Prasad, A., Roy, S., Sarkar, A., Panja, S. C., and Patra, S. N. (2023). An Improved Prediction of Solar Cycle 25 Using Deep Learning Based Neural Network. Sol. Phys., 298(3):50.

Priest, E. (2014). Magnetohydrodynamics of the Sun.

Qin, Y., Ji, K.-F., Liu, H., and Yu, X.-G. (2024). Resolution Enhancement of SOHO/MDI Magnetograms. Research in Astronomy and Astrophysics, 24(6):065029.

Rahman, S., Moon, Y.-J., Park, E., Siddique, A., Cho, I.-H., and Lim, D. (2020). Super-resolution of SDO/HMI Magnetograms Using Novel Deep Learning Methods. ApJ, 897(2):L32.

Raouafi, N. E., Patsourakos, S., Pariat, E., Young, P. R., Sterling, A. C., Savcheva, A., Shimojo, M., Moreno-Insertis, F., DeVore, C. R., Archontis, V., Tor¨ ok, T.,¨ Mason, H., Curdt, W., Meyer, K., Dalmasse, K., and Matsui, Y. (2016). Solar Coronal Jets: Observations, Theory, and Modeling. Space Sci. Rev., 201(1-4):1– 53.

Raygoza, J. M., Lopez, I. H., and Ram´ırez Velez, J. C. (2024). Feed-forward neural´ networks to estimate stokes profiles. Rev. Mex. Astron. Astrofis., 60:343–354.

Reale, F. (2014). Coronal Loops: Observations and Modeling of Confined Plasma. Living Reviews in Solar Physics, 11(1):4.

Reiss, M. A., Hofmeister, S. J., De Visscher, R., Temmer, M., Veronig, A. M., Delouille, V., Mampaey, B., and Ahammer, H. (2015). Improvements on coronal hole detection in sdo/aia images using supervised classification. Journal of Space Weather and Space Climate, 5:A23.

Rincon, F. and Rieutord, M. (2018). The Sun’s supergranulation. Living Reviews in Solar Physics, 15(1):6.

Rodr´ıguez, J.-V., Sanchez Carrasco, V. M., Rodr´ ´ıguez-Rodr´ıguez, I., Perez Aparicio,´ A. J., and Vaquero, J. M. (2024a). Are we crossing a minimum of the Gleissberg centennial cycle? Multivariate machine learning-based prediction of the sunspot number using different proxies of solar activity and spectral analysis. Advances in Space Research, 74(8):4173–4188.

Rodr´ıguez, J.-V., Sanchez Carrasco, V. M., Rodr´ ´ıguez-Rodr´ıguez, I., Perez Aparicio,´ A. J., and Vaquero, J. M. (2024b). Predicting Solar Cycle 26 Using the Polar Flux as a Precursor, Spectral Analysis, and Machine Learning: Crossing a Gleissberg Minimum? Sol. Phys., 299(8):117.

Roy, S., Schmude, J., Lal, R., Gaur, V., Freitag, M., Kuehnert, J., van Kessel, T., Hegde, D. V., Munoz-Jaramillo, A., Jakubik, J., Vos, E., Mandal, K., Akbari˜ Asanjan, A., de Sousa Almeida, J. L., Lin, A., Singh, T., Yang, K., Pandey, C., Hong, J., Aydin, B., Kurth, T., McGranaghan, R., Kasapis, S., Upendran, V., Bahauddin, S., da Silva, D., Pogorelov, N. V., Spalding, A., Watson, C., Maskey, M., Guhathakurta, M., Bernabe-Moreno, J., and Ramachandran, R. (2025). Surya: Foundation Model for Heliophysics. arXiv e-prints, page arXiv:2508.14112.

Sadykov, V. M., Kitiashvili, I. N., Sainz Dalda, A., Oria, V., Kosovichev, A. G., and Illarionov, E. (2021). Compression of Solar Spectroscopic Observations: a Case Study of Mg II k Spectral Line Profiles Observed by NASA’s IRIS Satellite. arXiv e-prints, page arXiv:2103.07373.

Sainz Dalda, A., de la Cruz Rodr´ıguez, J., De Pontieu, B., and Gosiˇ c, M. (2019).´ Recovering thermodynamics from spectral profiles observed by iris: A machine and deep learning approach. ApJ, 875(2):L18.

Sainz Dalda, A., de la Cruz Rodr´ıguez, J., Hansteen, V., De Pontieu, B., and Gosiˇ c,´ M. (2026a). The IRIS<sup>2+</sup> Inversion Tool: Recovering the Radiative Losses and the Thermodynamics in the Lower Solar Atmosphere. ApJ, 997(2):229.

Sainz Dalda, A., Upendran, V., Kim, J., Cho, K., Killam, P. S., Hansteen, V., and De Pontieu, B. (2026b). Predicting the thermodynamics in the chromosphere from the translation of SDO data into the IRIS<sup>2</sup> inversion results using a visual transformer model. arXiv e-prints, page arXiv:2604.21184.

Salvatelli, V., dos Santos, L. F. G., Bose, S., Neuberg, B., Cheung, M. C. M., Janvier, M., Jin, M., Gal, Y., and Gunes¨ Baydin, A. (2022). Exploring the Limits of Synthetic Creation of Solar EUV Images via Image-to-image Translation. ApJ, 937(2):100.

Sayez, N., De Vleeschouwer, C., Delouille, V., Bechet, S., and Lefevre, L. (2023).\` SunSCC: Segmenting, Grouping and Classifying Sunspots From Ground-Based Observations Using Deep Learning. Journal of Geophysical Research (Space Physics), 128(12):e2023JA031548.

Sayez, N., De Vleeschouwer, C., Delouille, V., Bechet, S., and Lefevre, L. (2025).\` Mitigating hallucination with non-adversarial strategies for image-to-image translation in solar physics. A&A, 702:A83.

Schilliro, F. and Romano, P. (2021). Segmentation of spectroscopic images of the low solar atmosphere by the self-organizing map technique. MNRAS, 503(2):2676–2687.

Schirninger, C., Jarolim, R., Veronig, A. M., Rempel, M., and W”oger, F. (2026). Neural blind deconvolution to reconstruct high-resolution ground-based solar observations. A&A, 708:A170.

Seth, P., Upendran, V., Anand, M., Sarkar, J., Roy, S., Chaki, P., Chowdhury, P., Ghosh, B., and Tripathi, D. (2025). SPACE-SUIT: an Artificial Intelligence Based Chromospheric Feature Extractor and Classifier for SUIT. Sol. Phys., 300(7):89.

Shibata, K. and Magara, T. (2011). Solar Flares: Magnetohydrodynamic Processes. Living Reviews in Solar Physics, 8(1):6.

Shin, G., Moon, Y.-J., Park, E., Jeong, H., Lee, H., and Bae, S.-H. (2020). Generation of High-resolution Solar Pseudo-magnetograms from Ca II K Images by Deep Learning. ApJ, 895(1):L16.

Socas-Navarro, H. (2005). Strategies for Spectral Profile Inversion Using Artificial Neural Networks. ApJ, 621(1):545–553.

Solanki, S. K. (2003). Sunspots: An overview. A&A Rev., 11(2-3):153–286.

Soler Poquet, I. J., D´ıaz Baso, C. J., Rouppe van der Voort, L. H. M., and Vissers, G. J. M. (2025). Automatic detection of Ellerman bombs using deep learning. A&A, 699:A54.

Su, X., Liang, B., Feng, S., Cai, Y., Dai, W., and Yang, Y. (2024). Solar cycle prediction using a combinatorial deep learning model. MNRAS, 527(3):5675– 5682.

Sun, W., Xu, L., Ma, S., Yan, Y., Liu, T., and Zhang, W. (2022). A dynamic deeplearning model for generating a magnetogram sequence from an sdo/aia euv image sequence. ApJS, 262(2):45.

Super-Kamiokande Collaboration et al. (2025). Measurement of solar neutrino interaction rate below 3.49 MeV in Super-Kamiokande-IV. arXiv e-prints, page arXiv:2512.19887.

Szenicer, A., Fouhey, D. F., Munoz-Jaramillo, A., Wright, P. J., Thomas, R., Galvez,˜ R., Jin, M., and Cheung, M. C. M. (2019). A deep learning virtual instrument for monitoring extreme UV solar spectral irradiance. Science Advances, 5(10):eaaw6548.

Tang, R., Zeng, X., Chen, Z., Liao, W., Wang, J., Luo, B., Chen, Y., Cui, Y., Zhou, M., Deng, X., Li, H., Yuan, K., Hong, S., and Wu, Z. (2021). Multiple cnn variants and ensemble learning for sunspot group classification by magnetic type. ApJS, 257(2):38.

Tremblay, B. and Attie, R. (2020). Inferring Plasma Flows at Granular and Supergranular Scales with a New Architecture for the DeepVel Neural Network. Frontiers in Astronomy and Space Sciences, 7:25.

Tremblay, B., Cossette, J.-F., Kazachenko, M. D., Charbonneau, P., and Vincent, A. (2021). Inferring depth-dependent plasma motions from surface observations using the DeepVel neural network. Journal ofSpace Weather and Space Climate, 11:9.

Tremblay, B., Roudier, T., Rieutord, M., and Vincent, A. (2018). Reconstruction of Horizontal Plasma Motions at the Photosphere from Intensitygrams: A Comparison Between DeepVel, LCT, FLCT, and CST. Sol. Phys., 293(4):57.

Tripathi, D., Chakrabarty, D., Nandi, A., Prasad, B., Ramaprakash, A., Shaji, N., Sankarasubramanian, K., Thampi, R., and Yadav, V. (2023). The aditya-l1 mission of isro. Proceedings ofthe International Astronomical Union, 18:17–27.

Tripathi, D., Ramaprakash, A. N., Khan, A., Ghosh, A., Chatterjee, S., Banerjee, D., Chordia, P., Gandorfer, A., Krivova, N., Nandy, D., Rajarshi, C., and Solanki, S. K. (2017). The solar ultraviolet imaging telescope on-board aditya-l1. Current Science, 113(04):616.

Upendran, V., Cheung, M. C. M., Hanasoge, S., and Krishnamurthi, G. (2020). Solar Wind Prediction Using Deep Learning. Space Weather, 18(9):e02478.

Upendran, V. and Tripathi, D. (2021). On the Impulsive Heating of Quiet Solar Corona. ApJ, 916(1):59.

Upendran, V., Tripathi, D., Mithun, N. P. S., Vadawale, S., and Bhardwaj, A. (2022). Nanoflare Heating of the Solar Corona Observed in X-Rays. ApJ, 940(2):L38.

Uwamahoro, J., McKinnell, L.-A., and Cilliers, P. J. (2009). Forecasting solar cycle 24 using neural networks. Journal of Atmospheric and Solar-Terrestrial Physics, 71(5):569–574.

Vaquero, J. M. (2007). Historical sunspot observations: A review. Advances in Space Research, 40(7):929–941.

Velasco Herrera, V. M., Soon, W., and Legates, D. R. (2021). Does Machine Learning reconstruct missing sunspots and forecast a new solar minimum? Advances in Space Research, 68(3):1485–1501.

Vicente Arevalo, A., Asensio Ramos, A., and D´ ´ıaz Baso, C. J. (2026). Accelerating 3D Non-LTE Synthesis with Graph Neural Networks. arXiv e-prints, page arXiv:2605.09543.

Vicente Arevalo, A., Asensio Ramos, A., and Esteban Pozuelo, S. (2022). Acceler-´ ating Non-LTE Synthesis and Inversions with Graph Networks. ApJ, 928(2):101.

Wang, Q.-J., Li, J.-C., and Guo, L.-Q. (2021a). Solar cycle prediction using a long short-term memory deep learning model. Research in Astronomy and Astrophysics, 21(1):012.

Wang, R., Fouhey, D. F., Higgins, R. E. L., Antiochos, S. K., Barnes, G., Hoeksema, J. T., Leka, K. D., Liu, Y., Schuck, P. W., and Gombosi, T. I. (2024). SuperSynthIA: Physics-ready Full-disk Vector Magnetograms from HMI, Hinode, and Machine Learning. ApJ, 970(2):168.

Wang, S., Chen, Q., He, C., Zhang, C., Zhong, L., Bao, H., and Rao, C. (2021b). Blind restoration of solar images via the Channel Sharing Spatio-temporal Network. A&A, 652:A50.

Williams, T., Prior, C. B., MacTaggart, D., Aslam, O. P. M., and Bloomfield, D. S. (2026). Investigating the efficacy of topologically derived time series for flare forecasting. ii. xgboost model. ApJ, 999(1):87.

Woods, M. M., Sainz Dalda, A., and De Pontieu, B. (2021). Unsupervised Machine Learning for the Identification of Preflare Spectroscopic Signatures. ApJ, 922(2):137.

Xu, C., Wang, J. T. L., Wang, H., Jiang, H., Li, Q., Abduallah, Y., and Xu, Y. (2024). Super-Resolution of SOHO/MDI Magnetograms of Solar Active Regions Using SDO/HMI Data and an Attention-Aided Convolutional Neural Network. Sol. Phys., 299(3):36.

Xu, C., Xu, Y., Wang, J. T. L., Li, Q., and Wang, H. (2025). Improving the spatial resolution of SDO/HMI transverse and line-of-sight magnetograms using GST/NIRIS data with machine learning. A&A, 697:A110.

Xu, L., Liu, S., Yan, Y., and Zhang, W. (2020). Euv wave detection and characterization using deep learning. Sol. Phys., 295(3):44.

Yang, H., Zuo, P., Liu, Z., Zou, P., Jiang, C., Zou, Z., Huang, Y., Wang, Y., Zhang, K., and Zhang, R. (2026). Application of a Grid-constrained U-shaped Neural Operator in Real-time Solar Corona Magnetic Field Extrapolation. ApJS, 283(2):56.

Yang, K. E., Sun, X., Tarr, L. A., Liu, J., Sadowski, P., Dodds, S. C., Rempel, M., Jaeggli, S. A., Schad, T. A., Cunnyngham, I., Glaser, Y., and Wolniewicz, L. (2025). Spectropolarimetric inversion in four dimensions with deep learning (spin4d). ii. a physics-informed machine learning method for 3d solar photosphere reconstruction. ApJ, 995(2):146.

Yang, K. E., Tarr, L. A., Rempel, M., Dodds, S. C., Jaeggli, S. A., Sadowski, P., Schad, T. A., Cunnyngham, I., Liu, J., Glaser, Y., and Sun, X. (2024). Spectropolarimetric inversion in four dimensions with deep learning (spin4d). i. overview, magnetohydrodynamic modeling, and stokes profile synthesis. ApJ, 976(2):204.

Yang, Y. (2025). Feature importance analysis of solar flares and prediction research with ensemble machine learning models. Frontiers in Astronomy and Space Sciences, 11:1509061.

Yang, Y., Yang, H., Bai, X., Zhou, H., Feng, S., and Liang, B. (2018). Automatic Detection of Sunspots on Full-disk Solar Images using the Simulated Annealing Genetic Method. PASP, 130(992):104503.

Yao, X., Munoz, P. A., and B˜ uchner, J. (2022). Non-thermal electron velocity distri-¨ bution functions due to 3D kinetic magnetic reconnection for solar coronal plasma conditions. Physics ofPlasmas, 29(2):022104.

Yeates, A. R., Cheung, M. C. M., Jiang, J., Petrovay, K., and Wang, Y.-M. (2023). Surface Flux Transport on the Sun. Space Sci. Rev., 219(4):31.

Youn, J., Lee, H., Jeong, H.-J., Lee, J.-Y., Park, E., and Moon, Y.-J. (2025). Can we properly determine differential emission measures from solar orbiter/eui/fsi with deep learning? A&A, 695:A125.

Yu, X., Xu, L., Ren, Z., Zhao, D., and Sun, W. (2022). Image Desaturation for SDO/AIA Using Mixed Convolution Network. Research in Astronomy and Astrophysics, 22(6):065009.

Yuan, D., Sadeghi, R., and Tavabi, E. (2026). High-Frequency Doppler Shift Oscillations and Their Damping in Transition Region Bright Point. Solar System Research, 60(1):11.

Zaqarashvili, T. V. and Erdelyi, R. (2009). Oscillations and Waves in Solar Spicules.´ Space Sci. Rev., 149(1-4):355–388.

Zbinden, J. and Kleint, L. (2026). Anticipating Future Flare Locations with Machine Learning and the Mg II Spectral Lines. ApJ, 1001(1):104.

Zbinden, J., Kleint, L., and Panos, B. (2024). Investigating and comparing the IRIS spectral lines Mg II, Si IV, and C II for flare precursor diagnostics. A&A, 689(A72):A72.

Zhang, C., Wang, S., Zhong, L., Chen, Q., and Rao, C. (2023). Cascaded Temporal and Spatial Attention Network for solar adaptive optics image restoration. A&A, 674:A126.

Zhang, L., Zhong, L., Guo, Y., Gong, X., and Rao, C. (2025). Nondeterministic wavefront estimation based on deep learning for multi-band synchronous highresolution reconstruction technology. Optics Express, 33(5):9224.

Zhang, T., Hao, Q., and Chen, P. F. (2024a). Statistical analyses of solar prominences and active region features in 304 a filtergrams detected via deep learning.˚ ApJS, 272(1):5.

Zhang, W., Wang, B., Wu, Z., Chen, Y., and Yan, F. (2024b). Identification and extraction of type ii and iii radio bursts based on yolov7. A&A, 683:A90.

Zhang, W., Yan, F., Han, F., He, R., Li, E., Wu, Z., and Chen, Y. (2021). Auto recognition of solar radio bursts using the c-dcgan method. Frontiers in Physics, 9:646556.

Zhang, Y., Xu, L., and Yan, Y. (2024c). Physics-informed neural network for force-free magnetic field extrapolation. Research in Astronomy and Astrophysics, 24(10):105010.

Zhang, Y. C., Wang, Z., Li, H., Wang, J., Qiao, C., Huang, Y., Fan, C., and Wang, Y. (2026). Resolving phase ambiguity via physically optimized single-image sensorless wavefront sensing with neural networks. Optics Express, 34(4):7122

Zhuo, R., He, J., Duan, D., Lin, R., Wu, Z., Yan, L., and Wei, Y. (2024). Prediction of solar activities: Sunspot numbers and solar magnetic synoptic maps. Science China Earth Sciences, 67(8):2460–2477.