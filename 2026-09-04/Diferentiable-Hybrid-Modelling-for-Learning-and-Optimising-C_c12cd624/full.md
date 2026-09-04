# Diferentiable Hybrid Modelling for Learning and Optimising Chemical Transport Processes from Experimental Data

Arthur Jessop<sup>1</sup>, Mohammed Alsubeihi<sup>1</sup>, Ben Moseley<sup>2,3</sup>, Ashwin Kumar Rajagoapalan<sup>1\*</sup>

<sup>1</sup>Department of Chemical Engineering, The University of Manchester, Manchester, M13 9PL, United Kingdom. <sup>2</sup>Department of Earth Science and Engineering, Imperial College London, South Kensington, SW7 2BX, United Kingdom. <sup>3</sup>I-X Centre for AI in Science, Imperial College London, South Kensington, SW7 2BX, United Kingdom.

\*Corresponding author(s). E-mail(s): a.rajagopalan@manchester.ac.uk; Contributing authors: arthur.jessop@manchester.ac.uk; mohammed.alsubeihi@manchester.ac.uk; b.moseley@imperial.ac.uk;

## Abstract

Reliable transport models are essential when modelling and optimising many chemical engineering processes, yet, most models assume hand-picked constitutive laws which may not reflect reality, and often assume initial conditions are known exactly. Both restrictions can significantly bias model predictions and lead to systematic error when used in predictive and control settings. Black-box neural surrogate alternatives for modelling can better match real example data, but are confined to the task they were trained on and cannot be interrogated for physical consistency. Here we introduce a general-purpose diferentiable hybrid modelling framework for transport processes, specifically for the case of population balance equations. Our framework integrates a JAX finite volume population balance solver with learnable neural network components which are trained to both discover constitutive laws and fit initial conditions from real experimental data, allowing us to better model real experimental transport systems. Furthermore, we use our framework for process optimisation, using its diferentiability to allow us to direct optimising experimental settings for quantities of interest. This

work highlights the huge potential of such diferentiable hybrid modelling frameworks for learning and optimising any given chemical separation which involves mass, energy, and/or momentum transport.

Keywords: Diferentiable physics, Hybrid modelling, Population balance equations, Data assimilation

## 1 Introduction

Transport equations are the mathematical foundation for modelling dynamic systems across chemical engineering. They describe how mass, heat, and energy are transported within complex systems. Their applications are broad, ranging from predicting the rheology of consumer products [1], to quantifying adsorption processes for carbon capture [2], analysing population balances in pharmaceutical and agrochemical processes [3], and guiding the design of fluidized beds [4], bioreactors [5], and heat exchangers [6]. With suitably constructed process models built on these governing equations, engineers can explore design spaces, predict, optimise and troubleshoot process outcomes, and implement model predictive control. Consequently, the ability to construct expressive and accurate models is a key enabler for the digital and autonomous manufacturing goals of Industry 4.0 and 5.0 [7].

Transport equations themselves express the conservation of mass, heat, and momentum. To model specific systems, constitutive laws and equations must be fit within these transport equations to capture system dynamics. Traditionally these equations are hand-picked by experts and fitted via trial-and-error for every new problem. However, selecting a fixed functional form inherently limits the model to a physical interpretation that may not capture the true underlying phenomena. Even if a proposed equation fits the available data, there is no guarantee it correctly describes the physics. A second assumption that is often made and rarely stated is that the initial conditions from which the laws are learned are exactly known. For dynamical systems where one tracks a quantity of interest over time, solutions can be very sensitive to initial conditions [8] In practice these are experimentally measured and therefore sub ject to noise or inherent measurement device limitations but often treated as ground truth. Together these force trial-and-error workflows in both model development and in the experimental conditions that these models are fitted to, limiting experimental eficiency and slowing model development.

While these issues afect all transport equations, this work focuses on population balance equations (PBEs) [3, 9]. These are number balance equations frequently applied to crystallisation, a unit operation common in the pharmaceutical, agrochemical, and fine chemicals industries to manufacture solid-form products [3]. Formulated as partial diferential equations (PDEs), they track the number of particles of a given size and shape to model how the overall particle size and shape distribution (PSSD) evolves over time. We take crystallization as our demonstration system as the issues raised so far are acutely relevant. Process kinetics are unknown and system-specific [10], meaning a model must be built for every new system. Most commercial monitoring tools also reduce particle shapes down to a single dimension, where a multidimensional approach would be more descriptive and facilitate the recovery of multidimensional laws [11]. Additionally, a recent comparative study has shown that many of the most popular tools frequently disagree with one another and ground-truth measurements [12]. Ultimately, if measured data does not contain enough information to describe the behaviour we are trying to learn, no amount of model sophistication will be able to capture it [13]. This means that models for these systems are both hard to formulate, and often limited due to the measurement techniques available.

An alternative to mechanistic approaches is to use completely ‘black box’ methods, such as neural networks (NNs), which ofer high flexibility, but lack physical intuition and constraints. These models can violate physical laws and be neither interpretable nor explainable, both of which are essential in highly regulated industries [14], such as pharmaceuticals. A more basic limitation of these methods is that their task is often limited to the one proposed in training. Recent approaches in crystallisation have focused on training black-box or surrogate models [15–18], which show promise for their specific tasks but cannot estimate quantities that were not inputs during training, or immediately extend to diferent types of simulation or systems. Physicsinformed approaches [19, 20] which instead guide training with physical knowledge and equations have seen growing interest in crystallisation [21–25]. Hybrid approaches that instead embed black-box components in mechanistic models have a longer history in process engineering [26] and have recently been applied to crystallisation [27]. However, these methods rarely support forward simulation (predicting process behaviour), model inversion (discovering governing laws), optimising experimental conditions (essential for process design), and targetting specific process outcomes (critical for product quality) all within the same model - no unified framework currently exists [28]. Conventional approaches, such as purely mechanistic models, that do extend to all of these tasks are limited to using numerical gradient-based methods which require a new forward model evaluation for every optimised parameter at each iteration. This, combined with slow simulation times, limits training and optimisation to just a handful of parameters. Automatic diferentiation and the tools of diferentiable physics [29] remove this scaling entirely and open the door to exploring much more flexible mechanistic and hybrid methods.

Here, we introduce a diferentiable hybrid modelling framework that combines mechanistic transport equations with learnable components, built on an accelerated JAX-based HR-FVM PBE solver [30, 31]. Its diferentiable formulation also enables end-to-end optimisation of experimental design and operating conditions with respect to process trajectories and downstream performance metrics, unifying model building, simulation and process design into a single framework. Together with an experimental dataset [32], we demonstrate how this framework can address both sources of model uncertainty, unknown functional form and partially uncertain initial conditions, while extending to process optimisation and design tasks.

Specifically, this work makes the following contributions:

• Framework: a diferentiable hybrid modelling framework for joint discovery of physical laws and initial conditions in transport equations from experimental data, built on JAX and leveraging automatic diferentiation to remove parameter scaling limits of numerical-gradient based methods.

• Data-Driven Model Discovery: a workflow comparison against the conventional pipeline for the same system [32], in which a hand-picked rate law is fitted to measured initial conditions. Jointly learning the rate law and assimilating the initial conditions recovers physically consistent, non-linear kinetics and reduces the trajectory loss on the validation experiments sevenfold, without additional experiments.

• Process Optimisation: demonstration of trajectory and product specification optimisation enabled by end-to-end model diferentiability, targetting relevant downstream metrics such as aspect ratio and yield.

![](images/4fe55295e8da75ba8229a989d3d7736ef0daf15384b460719e5ab4436b1a3df5.jpg)  
Fig. 1 A diferentiable hybrid framework for discovering transport laws, calibrating initial conditions, and optimising process trajectories.

(a) Experimental system. Batch temperature driven dissolution of l-glutamic acid in water, monitored in situ by the DISCO [33] which characterises crystals in two characteristic lengths (length $L _ { 1 }$ width $L _ { 2 } )$ . The dataset contains 13 real experimental trajectories [32] with varying temperature profiles and initial conditions $( f _ { 0 } , \ c _ { 0 } )$ . Example seed $\left( f _ { 0 } \right)$ and product $\left( f _ { \mathrm { f i n a l } } \right)$ distributions are shown along observed time series data. (b) System modelling. The simplified batch dissolution population balance equation is shown with the learned dissolution law (D) parametrised by θ highlighted. Different options for formulating the law are shown. (c) Hybrid framework and tasks. All inputs pass through the diferentiable solver to give predicted outputs, and automatic diferentiation is used to propagate gradients back from the trajectory mean squared error (MSE) loss function L(ϕ) to any subset of these inputs. This supports two tasks: model building, where predicted trajectories are matched to observed trajectories by optimising $\Phi = [ \Theta , f _ { 0 } , c _ { 0 } ]$ , and process optimisation, where a specific trajectory or process objective is targeted by optimising $\Phi = T ( t )$ or $[ T ( t ) , f _ { 0 } , c _ { 0 } ]$

## 2 Results

## 2.1 System & Framework

We study the dissolution of a small molecule amino acid (L-glutamic acid) in a solvent (water), a system in which particles are characterised in multiple dimensions, their length and width $( L _ { 1 } , L _ { 2 } )$ , using the DISCO [32, 33]. The driving force is the supersaturation S [-], which is a measure of how far the system is from equilibrium, and the operating condition is the temperature $T \ [ { } ^ { \circ } \mathrm { C } ] . \mathrm { I f } \ S = 1$ , the system is at equilibrium, if $S < 1$ then crystals dissolve (shrink), and if $S > 1$ then crystals grow. Throughout, the average volume weighted particle length and widths $( \overline { { L } } _ { 1 } , \overline { { L } } _ { 2 } )$ and the solute concentration (c) are used as the main observed quantities and training targets. The solver and its physical structure remain fixed throughout, but the individual kinetic terms are swappable, allowing the model to be composed for a number of diferent phenomena (e.g. growth, nucleation, breakage) (see Methods, Section 4.2 for the full PBE formulation). In Section 2.2 we learn a dissolution rate and assimilate initial conditions from an experimental dataset [32], and in Section 2.3 we demonstrate how we can optimise through and over the entire solver to target specific experimental trajectories and product specifications.

## 2.2 Model Building & Data Assimilation

The learned function throughout is the size-independent dissolution rate $D ( S , T )$ which determines the rate at which each characteristic length decreases $( D _ { 1 } , D _ { 2 } )$ , and how the solute concentration (c) increases over time as mass is transferred from the solid (crystal) phase to the liquid phase during dissolution. We benchmark against M1, the empirical rate law of B¨otschi et al. [32], evaluated in our solver using their published coeficients rather than refitting. Their approach, proposing four candidate empirical forms (M1 - M4) and fitting them within a mechanistic solver to find the simplest model with the lowest error, is the conventional route. This is a particularly demanding instance of the task; resolving kinetics in two dimensions compounds the combinatorial cost of testing candidate empirical forms, a search rarely attempted because of limited data availability, the lack of suficiently fast solvers for full multidimensional PSSDs, and the time constraints of real-world process development.

We embed a neural network directly in the diferentiable solver to model this behaviour, replacing the hand-picked rate law with a flexible function to make a Hybrid Model. In doing so, the hybrid framework avoids the need to manually specify and test multiple candidate rate expressions. For example, for dissolution in two dimensions, four candidate equations for each dimension would lead to sixteen possible model pairings, with combinatorial complexity increasing further as additional crystallisation mechanisms are considered. In contrast, the neural network acts as a universal approximator [34], enabling exploration of a broader functional space directly from the data in a single training efort, predicting the rate in both dimensions $( D _ { 1 } , D _ { 2 } )$ irrespective of the underlying mechanism.

When trained traditionally with a train/validation data split and the measured initial conditions held fixed (Case A, Section 4.5), the hybrid model lowers the validation loss (0.812) versus the fixed empirical baseline (1.264), where validation loss is the mean squared error (MSE) between the model prediction and measured trajectories (see Methods, Section 4.5). However, in this case, with fixed measured initial conditions, the learned rate function is physically inconsistent. The embedded NN predicts a non-zero dissolution rate at equilibrium $( S = 1 )$ and demonstrates an inverted temperature dependence, where the rate decreases as temperature increases, opposite to the expected trend. This is demonstrated over 20 diferent models with random seeding and shown in Extended Figure 5. There is no structural element of the NN architecture we use that enforces rate to go through or approach zero at equilibrium, only that it should be clipped to zero for $S > 1$ , so any equilibrium behaviour must be learned from the data. We show that the hybrid model can improve predictive performance but encode the wrong physics, which is why predictive accuracy alone is not always a suficient test. Since the underlying PBE is sensitive to its initial conditions (a common trait) we look to our data and try to understand if the inconsistency is caused by the model itself or the data it is fit to.

Re-examination of the experimental data reveals that some initial conditions, particularly those of $\overline { { L } } _ { 1 }$ and ${ \overline { { L } } } _ { 2 } .$ , deviate from the expected values inferred from the remainder of the time series. This is visible where the experimental IC (yellow circles) depart from where the rest of the experimental data (grey dots) would extrapolate back to in Figure 2(c). We leverage the diferentiability of our solver and include the entire initial condition as uncertain free variables and assimilate (calibrate) these in parallel to training the hybrid model (Case B), testing whether the mismatch influences the learned kinetics. This separates measurement error in the initial condition from the underlying physics governing the experimental trajectories. Since this task uses the entire dataset (with only two experiments held out for later optimisation tasks), the original validation experiments are now included in the training data. For a fair comparison with the empirical baseline, we still report the trajectory loss on this same subset of experiments; however, this is no longer a true validation loss, as the hybrid model has already been fitted to these experiments. Both are the same trajectory MSE (Eq. 17), the only diference is whether those experiments were seen during training. This is a more general workflow comparison, rather than a specific rate model comparison. Over $4 . 8 \times 1 0 ^ { 5 }$ parameters are involved in this optimisation, which is only possible with automatic diferentiation (AD) which makes gradient cost efectively independent of parameter count. The parameter count in this case is mostly dominated by the 11 initial distributions which are each a matrix of $3 6 0 \times 1 2 0$ elements. Within the diferentiable framework, fitting the two-parameter M1 empirical law for a fixed 500 epochs takes 5 minutes. This refit is reported only for computational benchmarking, the predictive performance comparison throughout uses the published coeficients. When adding the NN parameters and assimilated initial condition parameters, this only increases to 6 minutes, a small addition in computational cost for such a large increase in parameters, as seen in the table in Figure 2(b). For comparison, a numerical-gradient based method takes in excess of 15 minutes to fit the same two-parameter model (Extended Figure 7) - over twice as long as the full hybrid model with initial condition assimilation, despite optimising five orders of magnitude fewer parameters.

Through this joint optimisation the recovered kinetics become physically consistent, Fig 2(a). The S dependence of the hybrid model now correctly passes through zero at S = 1, and the learned function is in the same order of magnitude to the empirical baseline whilst revealing a non-linear response. This is, again, consistent across 20 models trained from unique random seeds. The previously observed temperature artefact is also removed, with the models exhibiting negligible temperature dependence consistent with the empirical model. In particular, where the input density of S is high, the model spread is narrow and the learned function difers in shape to the empirical equation, indicating that the non linearity is expressed in the data itself and better describes the underlying physics. At low S density, the model spread is wider as this part of the function is seen less in training.

During the joint optimisation the calibrated initial conditions only shift marginally from their measured values. The data assimilation only adjusts the shape, not the mass of the distribution, which is renormalised during simulation which ensures that it remains physical. For E7, the case shown in Fig 2(b), the experimental and calibrated distributions are almost entirely overlapping as shown by both the full distribution itself and marginal distributions. E7 was selected as it exhibited the largest shift in ICs over the validation set. In fact, the corresponding mean length and width shift only 24.1µm and 1.4µm respectively, representing shifts of 4.6% and 2.0%.

The conventional pipeline (fixed empirical law, measured IC) reaches a trajectory loss of 1.264 on the validation experiments, whereas the full diferentiable pipeline (learned hybrid law, assimilated ICs) reaches 0.176. This is clear in Fig 2(c) where the proposed hybrid workflow gives a prediction much closer to the time series experimental data. Crucially this does not require any additional experiments which would consume both time and material. We can also see that the initial concentration shift is small, and much smaller than the distribution shift. When assimilating the initial conditions with the fixed empirical rate, the trajectory loss on the validation experiments is 0.195 against the hybrid’s 0.176. Over the range in the data the two rate laws give relatively similar predictive accuracy, however the hybrid only requires a single training efort, avoiding the combinatorial testing of multiple candidate empirical equations, and is only tractable with a diferentiable solver.

Taken together, these results indicate that for this dataset, over the experimental range sampled, the uncertainty in the initial conditions is the main constraint in model and predictive quality. Crucially, jointly assimilating this many IC parameters is only tractable due to AD, and, contrary to fully black-box approaches, every recovered term and IC can be individually inspected for physical consistency. Having established a framework for building a physically consistent model of the system dynamics, we next move on to use this model, and other model formulations, to directly optimise the operating conditions that drive the process within the same framework.

![](images/ed6c92fbd37c58b809280e7f3470b5e71e1433e5278f5cdee71657250a1254f5.jpg)

![](images/2fe0f15a9bb51effe549faf48509841e21a22510dc4e5f1346e1f85291f18ba2.jpg)

![](images/36350656f5e91f8662529640405fe1eb1c77f7511978d4b9da8efc113a426ac2.jpg)  
Fig. 2 Joint model training and initial condition assimilation (calibration) recovers a physically consistent, non-linear dissolution law. (a) Discovering a physical law. The learned dissolution rates $( D _ { 1 } , \ D _ { 2 } )$ as functions of supersaturation (at fixed ${ \bar { T } } = 2 { \bar { 8 } } ^ { \circ } \mathrm { C } )$ and temperature (at fixed $S = 0 . 9 9 5 )$ for the calibrated-IC hybrid model (green) and the empirical baseline (orange). The solid green line is the ensemble mean and the shaded band is a min/max range over 20 random initialisations. Input frequency density is shown above each axis. The shaded region on the left marks low-S input density where the ensemble spread widens. The hybrid recovers a non-linear S dependence that vanishes at equilibrium $( S = 1 )$ and negligible temperature dependence, in line with the empirical equation. (b) Calibrating initial conditions of E7: the measured seed distribution (orange) compared to the calibrated distribution (green), with marginal $L _ { 1 }$ and $L _ { 2 }$ distributions and their mean values highlighted with dashed lines. The calibrated distribution is renormalised to measured seed mass, so assimilation only adjusts the shape. The table reports parameter counts and time taken to optimise for 500 epochs for the empirical and the calibrated-IC hybrid case, demonstrating the scaling behaviour of AD in JAX. (c) Improving model performance. Predicted time series across all validation experiments for the three target metrics $( c , \overline { { L } } _ { 1 } , \overline { { L } } _ { 2 } )$ : empirical model with un-calibrated IC (orange, trajectory error = 1.264) compared to the hybrid model with calibrated IC (green, trajectory error = 0.176). Calibrated ICs shift only marginally from the measured values (e.g. E7: − $\cdot 0 . 0 7 \mathrm { g } \mathrm { k g } ^ { - 1 }$ , −24.1 µm, −1.4 µm for $c , \overline { { L } } _ { 1 } , \overline { { L } } _ { 2 } )$ yet drastically improve the fit.

## 2.3 End-to-End Process Optimisation

The same AD capability that enables training hybrid models and data assimilation can, at inference time, with the model frozen, be applied to the operating condition to perform process optimisation. This is simply a case of changing the free variables that are optimised.

We demonstrate two distinct optimisation tasks. The first is process trajectory optimisation, in Section 2.3.1, where we target existing trajectories using the frozen hybrid dissolution model that we built previously in Section 2.2. In the second task we optimise towards a specific product specification using fixed published size dependent kinetics in Section 2.3.2, specifically using the M4 dissolution law [32] (Eq. 11) combined with a size-dependent growth model from Ochsenbein et al. [35] (Eq. 13). For this task, we manipulate the crystal aspect ratio (the ratio of the crystal length and width, $L _ { 1 } / L _ { 2 } )$ , which requires kinetics that are size-dependent, hence the use of alternative published kinetics.

Both tasks are subject to the same set of constraints, which keep the operating conditions within the capabilities of the equipment and each kinetic model within the range over which it was fitted. The temperature is bounded to the range spanned by the experimental data $( 2 0 - 4 1 ^ { \circ } \mathrm { C } )$ and its rate of change to the limit of the jacketed vessel used in the experiments $( | \mathrm { d } T / \mathrm { d } t | \leq 0 . 0 5 ^ { \circ } \mathrm { C } \operatorname* { m i n } ^ { - 1 } )$ . The supersaturation is held above a dissolution floor, $S _ { \mathrm { m i n } } = 0 . 9 7 5 [ 3 2 ]$ , and, wherever a growth law is active, below a ceiling, $S _ { \mathrm { m a x } } = 1 . 2 5 ~ [ 3 5 ]$ . The temperature bound is imposed as a hard constraint, while the remaining are included as soft penalties in the optimisation loss (Eq. 20); full details are given in Methods (Section 4.6).

## 2.3.1 Trajectory Optimisation

Often the operating condition, $T ( t )$ , is an imposed quantity manually defined to achieve a desired process trajectory - mirroring the philosophy of empirical equation selection, where the definition is driven by testing, expertise and process knowledge. Here we replace this manual process with direct optimisation of $T ( t )$ toward a desired trajectory, using the hybrid dissolution model built in Section 2.2. Temperature profiles are parametrised by 300 linearly spaced temperatures - again this is only tractable with AD, and considerably finer than previous approaches that simplify profiles to a handful of points [15]. Full IC assimilation is also possible here.

We target crystal length and width trajectories $( \overline { { L } } _ { 1 }$ and $\overline { { L } } _ { 2 } )$ which are measured by the DISCO [33] and directly related to product specification; concentration is not a specification target here. Of the supersaturation bounds, only $S _ { \mathrm { m i n } }$ is active here, since the model describes pure dissolution. These targets carry experimental noise, and the optimiser recovers a $T ( t )$ that fits the underlying trajectories despite it.

We first validate the task on synthetic data, separating the optimiser’s ability to recover $T ( t )$ from any mismatch due to model or IC uncertainty before testing on experimental data. Target trajectories are generated by defining operating conditions (a stepped ramp and a sigmoid), taking the IC from $\mathrm { E 5 } ,$ , and performing a forward simulation. The original operating condition is reliably recovered from both targets across a range of initial $T ( t )$ guesses, (Extended Figure 6(a)).

We then target a real experimental trajectory using the held-out experiment E5. With the initial condition fixed at its measured value, the optimised temperature profile departs from the experimental one and the length and width trajectories are not accurately captured (Extended Figure $6 ( \mathrm { b } ) ,$ ). This is the same failure mode seen in Section 2.2, where the IC is treated as ground truth, this time relevant at inference rather than training. This is visible in Extended Figure $6 ( \mathrm { b } )$ where the optimisation without data assimilation (dashed blue line) deviates more from the measured $T ( t )$ to account for the IC mismatch. Freeing the initial condition for assimilation recovers a profile that closely matches the experimental one (Figure $\mathrm { 3 ( a ) ) }$ , and tracks the target trajectories more closely than the smoothed-guess initialisation. The fact that the optimised profile is similar to the experimental profile for an unseen experiment indicates that the model has learned to represent the underlying dynamics well, rather than fit noise in the training data. Relative to the smoothed initial guess, which is the standard approach to model inference, the optimised profile reduces the $T ( t )$ MAE from $0 . 1 0 ^ { \circ } \mathrm { C }$ to $0 . 0 5 ^ { \circ } \mathrm { C }$ whilst remaining within the $S _ { \mathrm { m i n } }$ bound (Figure 6).

## 2.3.2 Product Aspect Ratio Optimisation

The previous task focussed on optimising towards a known answer. Here we shift the focus to an unknown answer and optimise towards a specific process product specification using the combined growth-dissolution model. We define criteria that must be met by the final product, and let the solver turn this into an operating condition to implement and test in a future experiment - this is relevant to process exploration and design.

The product specification we aim to improve is the particle aspect ratio given by the ratio of the average crystal length and widths, $\overline { { L } } _ { 1 } / \overline { { L } } _ { 2 }$ . Downstream performance is impacted by this as the ratio afects packing, drying, filtration [36], bioavailability of the solid product obtained through crystallisation [37]. We again take the IC from E5 and aim to reduce the aspect ratio. Both $S _ { \mathrm { m i n } }$ and $S _ { \mathrm { m a x } }$ are active here, since growth and dissolution are both present, and we add the constraint that the final product volume should not decrease relative to the initial seed volume, $V _ { \mathrm { p r o d u c t } } / V _ { \mathrm { s e e d } } \geq 1$ This task therefore aims to design a process that optimises product quality whilst not losing material. Often temperature cycling, alternating between growth and dissolution regimes, is used for this kind of problem [38] so we initialise our operating condition as a linear ramp that starts in dissolution and ends in growth.

Through this optimisation, as shown in Figure 3 (b), we are able to design a $T ( t )$ for this system that reduces the product aspect ratio from 6.50 to 4.88, an improvement of 25%. The optimal profile in this case retains the single cycle, first dissolving at constant supersaturation, with the product volume reducing to around 35%, before entering a growth region where the volume increases back up to 100%. This stays within the supersaturation limits and satisfies the product volume specification, both shown in the figure.

To reduce the aspect ratio, the optimiser exploits a physically interpretable feature of the combined growth-dissolution model. The growth is size-dependent in length by a factor of $L _ { 1 } ^ { 1 . 2 }$ , but has no size-dependence in the width. By first dissolving the population to a smaller size before entering the growth region, length growth is suppressed relative to width, redistributing mass toward the width and reducing the aspect ratio, while also contributing to the volume increase. This reduces the aspect ratio, and contributes to the volume increase.

It must be noted that this is a purely simulated result, and not validated against experimental data as in Section 2.3.1. As is the case for process design and model inference in real world scenarios, this is a model-generated hypothesis for process improvement that would have to be validated through further experiments, which is beyond the scope of this work.

![](images/08085a3f4cff122a4faeed3e2e57c840829e374786781bf275c2f0bd9c794cae.jpg)  
Fig. 3 End-to-end optimisation of operating conditions recovers real process trajectories and proposes new conditions to meet a product specification. (a) Process trajectory optimisation. Using the frozen hybrid dissolution model (Figure 2), the temperature profile $T ( t )$ and initial condition $( f _ { 0 } , c _ { 0 } )$ are optimised (solid) from an initial guess (dotted) to match the observed $\overline { { L } } _ { 1 }$ (pink) and $\overline { { L } } _ { 2 }$ (orange) trajectories of the held-out experiment E5 (open squares). The optimised S(t) remains within the $S _ { \mathrm { m i n } }$ bound throughout, and the recovered $T ( t )$ closely tracks the true experimental profile despite the profile never being provided as an optimisation target. (b) Product specification optimisation. Using the size-dependent growth & dissolution model with the initial condition of E5, T(t) is optimised (solid, from a linear dissolution → growth initial guess, dotted) to reduce the final product aspect ratio subject to S , S<sub>max</sub>, and a volume-conservation constraint $( V _ { \mathrm { p r o d u c t } } / V _ { \mathrm { s e e d } } \geq 1 )$ . The optimal profile first dissolves the seed population $( S < 1$ , red, with volume falling to ≈35%) before switching to growth (S > 1, green, with volume recovering to 100%). The seed (grey) and product (blue) PSSDs are shown, with dotted lines marking the aspect ratio $( \overline { { L } } _ { 1 } / \overline { { L } } _ { 2 } )$ of each. The optimised profile reduces the aspect ratio from 6.50 to 4.88 (25% reduction) with the same final mass.

## 3 Discussion

This work demonstrates how end-to-end diferentiable modelling extends mechanistic PBE solvers beyond forward simulation to a broader class of modelling, inference, and optimisation tasks. By embedding neural networks directly within a mechanistic PBE solver, we combine the flexibility of modern machine learning with the structure of established physics. The same solver and governing equations support forward simulation, recovery of a physical law, assimilation of the initial conditions that law is learned from, and optimisation of the operating conditions that drive the process. Only the free variables ϕ change between these tasks, whether to diagnose the data $( \Phi = \left[ f _ { 0 } , c _ { 0 } \right] )$ ), to build a model $( \Phi = \Theta )$ , or to design an experiment $( \Phi = T ( t ) )$ . It is this generality, rather than any single result, that is a key contribution of this work.

Crucially, we show that improved predictive performance is not always suficient when one wants to uncover a transferable, generalizable physical law. This is where the most important finding of our work lies. The initial Case A results clearly demonstrate that hybrid models can achieve improved predictive performance while encoding physically inconsistent behaviour (loss reduces to 0.812 vs. 1.264). The learned rate does not approach zero at equilibrium and its temperature dependence is inverted, and both persist across 20 randomly initialised ensemble members. This is a direct consequence of coupling a flexible NN with a set of PDEs that is sensitive to its initial conditions, and treating uncertain experimental data as ground truth. The network is unable to distinguish noise in the initial condition or time series data from physical behaviour and minimises the loss function without any regard for established physics. Notably, this failure is only visible because the learned component is a small, inspectable function embedded within a mechanistic model - which would not be the case if we trained a black-box for the whole system. This kind of insight is often overlooked, particularly in the crystallisation and broader chemical separations communities which are rapidly adopting ML methods [14, 39–41].

A diferentiable framework ofers tools to address this. Rather than training a model and evaluating it on fixed, uncertain data points, we can use the same gradient-based methods to jointly optimise the initial conditions alongside the kinetic parameters. This enables diagnosis of the source of inconsistency and recovery of physically consistent behaviour. It also separates error in the rate law from error in the data that law is fitted to. Assimilating the initial conditions for the empirical law recovers a comparable trajectory loss to the hybrid (0.195 against 0.176), which identifies the measured initial conditions rather than the functional form as the limiting factor for this dataset. A study of this type is only possible using automatic diferentiation, as it involves optimising over $4 . 8 \times 1 0 ^ { 5 }$ NN parameters and initial conditions across the entire solver - a task we show is essentially impossible when using numerical gradient based approaches. The same scaling makes rapid screening of empirical expressions practical, with small models (2 - 4 parameters) trained between 6 and 20 times faster and forward simulations run 13 times faster than the reference workflow, though we note this compares complete workflows on their native hardware, GPU-enabled JAX against .mex compiled CPU code. Diferentiability therefore extends the role of mechanistic solvers beyond forward simulation and simple parameter estimation, enabling systematic data diagnosis, complex optimisations (tens to hundreds of thousands of parameters), sensitivity analysis, and physical consistency checks.

Because the model is diferentiable end-to-end, the same machinery can be applied at inference. Again, this is simply a case of changing the free variables that are optimised. The hybrid model is frozen here, and is asked to design an operating profile for a process, a task never posed during its training. This is the property that distinguishes the framework from the black-box and surrogate approaches discussed in Section 1, which are confined to the task and the quantities specified when they were trained. Optimising the temperature profile $T ( t )$ against the held-out experiment E5 recovers a temperature profile close to the one actually used, with an MAE of $0 . 0 5 ^ { \circ } \mathrm { C }$ . This profile was never supplied as an optimisation target, and the experiment was never seen during training, testing the learned dynamics in the inverse direction. Applied instead to a product specification, the framework returns both a candidate operating profile and an interpretable mechanism for it, exploiting the $L _ { 1 } ^ { 1 . 2 }$ size dependence of the growth law to preferentially redistribute mass to the crystal width. This replaces the manual specification of operating conditions with gradient-based design. It must be noted that this result is purely simulated, and would require experimental validation. More generally, our demonstration is limited to a single material system of thirteen experiments, and the learned rate is size-independent - extending the learnable component to size-dependent kinetics, which this task currently takes from published correlations, is the natural next step.

Importantly, this approach does not introduce many of the practical limitations associated with the ML methods that are commonly employed in traditional engineering disciplines. There is no additional data requirement compared to empirical modelling, the NNs used remain small and straightforward to train, and embedding the NN within the PBE solver ensures that the governing conservation laws of the system are always satisfied and that the predictive accuracy is relatively insensitive to the network size. Throughout this work the solver and the governing equation were unchanged, and only the convection term was swapped. The same framework can therefore extend directly to other kinetic expressions within the PBE, such as nucleation, breakage and agglomeration, and to other transport equations entirely. More broadly, this work shows that existing experimental datasets can contain more information than conventional fitting workflows extract from them, and that a diferentiable solver is what makes that information accessible - for diagnosis, for model building, and for design.

## 4 Methods

## 4.1 Experimental Data & Split

The experimental procedure used to generate the 13 experiments (E1 to E13) used in this work was obtained by the senior author of this work. This is detailed by B¨otschi et al. [32] and forms the benchmark for our work. Part of this dataset is itself adapted by work by the same authors from 2018 [42]. A key enabler for this data was the DISCO [33], which facilitates monitoring the size and shape of crystals in real time during the experiments, as shown in Figure 1(a).

The data split used was the same as that of B¨otschi et al. with two experiments from the validation set (E5, E12) left aside for the process optimisation task in Section 2.3. For the standard training approach, this is the split that was used. For the approach where the initial conditions were assimilated (Section 2.2), the entire dataset was used for training (again, without E5 or E12), since each initial condition needed to be individually calibrated. This let us diagnose whether the non-physical behaviour observed was a structural issue or one caused by the data split.

Key time series from the experimental data are shown in Extended Figure 4 with each experiment labelled with its experiment number, seed mass, and whether it is part of the training, validation, or held out set. The operating condition $T ( t )$ and the three experimental predictors $( c , L _ { 1 } , L _ { 2 } )$ are shown.

## 4.2 Modelling

We model a batch crystallisation system where there is no flow in or out of the crystallizer during each experiment. The PBE that we use to simulate this system is a partial diferential equation given by

$$
\frac { \partial f ( \mathbf { L } , t ) } { \partial t } + \nabla _ { \mathbf { L } } \cdot [ U ( \mathbf { L } , \mathbf { Y } , t ) f ( \mathbf { L } , t ) ] = 0\tag{1}
$$

where $f ( \mathbf { L } , t )$ is the number density of crystals with characteristic lengths $\mathbf { L } = \left( L _ { 1 } , L _ { 2 } \right)$ at time t, and Y is the system state, capturing information such as the temperature and the concentration. The rate at which crystals grow or dissolve (the convection rate) is given by ${ \cal U } ( { \bf L } , { \bf Y } , t )$

The PBE describes the solid phase $( { \mathrm { i . e . } }$ , the crystal population) and must therefore be coupled with the following mass balance to describe the liquid phase:

$$
\frac { \mathrm { d } c } { \mathrm { d } t } = - \rho _ { \mathrm { c } } k _ { \mathrm { v } } \frac { \mathrm { d } V _ { \mathrm { c } } } { \mathrm { d } t } .\tag{2}
$$

Here, $c \ [ \mathrm { g } \mathrm { k g } ^ { - 1 } ]$ is the solute concentration in the liquid phase, $\rho _ { \mathrm { c } } = 1 . 5 9 \times 1 0 ^ { - 1 2 }$ [g $\lbrace \mathrm { { \mu m } ^ { - 3 } } ]$ ] is the crystal density, $k _ { \mathrm { v } } = \pi / 4 \left[ - \right]$ is the shape factor accounting for noncuboidal crystal shapes, and $V _ { \mathrm { c } } \ [ \mathrm { \mu m ^ { 3 } \mathrm { k g } ^ { - 1 } } ]$ is the total crystal volume.

These coupled equations are subject to the following initial and boundary conditions

$$
f ( \mathbf { L } , t = 0 ) = f _ { 0 }\tag{3}
$$

$$
c ( t = 0 ) = c _ { 0 }\tag{4}
$$

$$
f ( \mathbf { L } = 0 , t ) = 0\tag{5}
$$

$$
f ( \mathbf { L } = \infty , t ) = 0\tag{6}
$$

where $f _ { 0 }$ is the initial distribution, commonly referred to as the seed distribution, and $c _ { 0 }$ is the initial concentration.

We solve a fully discrete system, using the finite volume method (FVM) [43] for the spatial coordinates and forward Euler with a CFL condition for the temporal coordinates. Further details on the solver implementation can be found in our previous work [31] which details the JAX solver.

## 4.3 Empirical Kinetic Models

The convection term ${ \cal U } ( { \bf L } , { \bf Y } , t )$ is the system specific part of the model. This is the term that changes throughout this work whilst the underlying solver and PDE is unchanged. We consider models that are functions of temperature $T ~ [ { } ^ { \circ } \mathrm { C } ]$ and the relative supersaturation $S \ [ - ]$ , given at a time t by

$$
S = \frac { c - c ^ { * } ( T ) } { c ^ { * } ( T ) } ,\tag{7}
$$

where $c ^ { * } ( T ) ~ [ \mathrm { g k g } ^ { - 1 } ]$ is the system specific solubility given by

$$
c ^ { * } ( T ) = 3 . 3 7 e ^ { 0 . 0 3 6 T } .\tag{8}
$$

B¨otschi et al. [32] proposed and fitted four diferent empirical dissolution rate expressions to their experimental data, and we use two of these retaining their original names. These are the two parameter size-independent linear model M1, which serves as our predictive performance benchmark, and the four parameter size-dependent non-linear model M4. Both are used at their published parameters alongside a size-dependent growth model from Ochsenbein et al. [35] which is combined with M4 and used in Section 2.3. This growth model was fitted for the same system of L-glutamic acid in water.

Each of these models is only used over the range in which it was originally fitted. These ranges define the supersaturation limits applied during the optimisations in Section 2.3. For M1 and M4, this is $S \ge S _ { \mathrm { m i n } } = 0 . 9 7 5 \ [ 3 2 ]$ , corresponding to the range of supersaturations observed in the experimental data. For the size-dependent growth model, this is $S \le S _ { \mathrm { m a x } } = 1 . 2 5$ , as reported in their work [35].

## 4.3.1 M1 - predictive performance baseline

The linear M1 model was the chosen model of B¨otschi et al. [32] and acts as our predictive performance baseline in Section 2.2. This is also used in Extended Figure 7 for computational benchmarking. The M1 model is given by

$$
\begin{array} { c } { { D _ { 1 } = - 2 . 3 2 ( 1 - S ) } } \\ { { D _ { 2 } = - 0 . 1 9 ( 1 - S ) . } } \end{array}\tag{9}
$$

(10)

## 4.3.2 Size-dependent growth and dissolution model

We use an extended combined growth-dissolution kinetic model for the aspect ratio optimisation task in Section 2.3. In this case two models are combined where the dissolution equation is used for $S < 1$ , and the growth equation is used for $S > 1$

Dissolution is the M4 model from B¨otschi et al. [32] given by

$$
D _ { 1 } = - 2 . 2 4 ( 1 - S ) \left( 1 + L _ { 1 } \right) ^ { - 5 . 0 5 \times 1 0 ^ { - 2 } }\tag{11}
$$

$$
D _ { 2 } = - 6 . 4 3 \times 1 0 ^ { - 4 } ( 1 - S ) \left( 1 + L _ { 2 } \right) ^ { 1 . 2 9 } .\tag{12}
$$

This is the four parameter model shown in Extended Figure 7.

The growth model is from [35] and is size-dependent in just the $G _ { 1 }$ growth rate. This model is given by

$$
G _ { 1 } = \exp \left( - \frac { 6 . 9 \times 1 0 ^ { 4 } } { T ^ { 2 } \ln ( S ) } \right) \exp \left( - \frac { 1 . 9 \times 1 0 ^ { 3 } } { T } \right) ( S - 1 ) ^ { 2 / 3 } ( \ln ( S ) ) ^ { 1 / 6 } L _ { 1 } ^ { 1 . 2 }\tag{13}
$$

$$
G _ { 2 } = ( S - 1 ) ^ { 2 . 7 } \exp \left( - \frac { 7 \times 1 0 ^ { 2 } } { T } \right) .\tag{14}
$$

## 4.4 Hybrid Dissolution Rate

In Section 2.2 we replace the convection term of the PBE with a NN to describe a dissolution rate. The rest of the FVM solver remains untouched, ensuring that the physical constraints of the mechanistic solver are enforced (e.g. the number and mass balance).

The dissolution rate is parametrised as a multilayer perceptron (MLP), mapping two state variables to two dissolution rates:

$$
D = \mathrm { N N } _ { \Theta } ( S , T )\tag{15}
$$

where θ is weights and biases of the network. This matches the size-independent form of M1 which forms the baseline for this approach. The network has 2 input neurons, 2 output neurons, tanh activated hidden layers, and is clipped to return zero for cases where $S \geq 1$ . This enforces no dissolution under supersaturated conditions. The main text refers to networks with a configuration of 3 hidden layers of 20 neurons each $\left( 3 \times 2 0 \right)$ , which was chosen after performing preliminary hyperparameter sweeps where it was concluded that increasing the size of the network did not dramatically impact the training. Results for smaller networks (including $1 \times 5$ and $2 \times 5 )$ are also reported in Extended Figure 5 yielding similar predictive performance, indicating minimal sensitivity to network size.

Each input S and $T$ value is standardised before being passed to the network, mapping a raw data value x to

$$
x ^ { * } = \frac { x - \mu _ { x } } { \sigma _ { x } }\tag{16}
$$

where $\mu _ { x }$ and $\sigma _ { x }$ are the mean and standard deviation of all observed values of $x$ in the experimental data set.

The output layer uses a negative softplus activation to ensure that the rates predicted are negative. Whilst the clip enforces zero dissolution for $S \geq 1$ , no structural element of the network forces the rate to approach zero as $S  1$ . Since dissolution rates are not directly measurable, this is scaled by a reference rate. The reference rates are estimated as the mean change in average particle length and width per unit time across all experiments, e.g. $\dot { L } _ { 1 , \mathrm { r e f } } = \Delta \bar { L } _ { 1 } / \Delta t$

## 4.5 Hybrid Training Procedure

Three time resolved predictors (process outputs) are used for training: the concentration of the molecules dissolving from the solid phase into the liquid phase (c), and the average particle length and width $( \overline { { L } } _ { 1 } , \overline { { L } } _ { 2 } )$ . Examples of these predictors from the experimental dataset are shown in $\mathrm { F i g 1 ( a ) }$ and the full dataset is shown in Extended Figure 4. These are used in the mean squared error (MSE) loss function

$$
\mathcal { L } ( \Phi ) = \frac { 1 } { 3 N } \sum _ { i = 1 } ^ { 3 } \sum _ { j = 1 } ^ { N } ( y _ { i , j } - \hat { y } _ { i , j } ( \Phi ) ) ^ { 2 } .\tag{17}
$$

where each predictor is equally weighted, y is an experimental data point, and ${ \hat { y } } ( \Phi )$ is a model prediction obtained from the FVM solver as a function of the free variables ϕ. Experimental data and predictions are standardised per predictor before calculating the loss using the same standardisation as $\operatorname { E q } .$ . 16, with $\mu$ and σ calculated over the full dataset ahead of training.

We consider two training cases, both minimising L over the free variables ϕ. In Case A, only the dissolution model parameters, $\Phi = \Theta$ , are optimised, while in Case $B ,$ the initial conditions $f _ { 0 }$ and $c _ { 0 }$ are assimilated and included as additional free variables to account for experimental uncertainty in the measured initial conditions, $\Phi = [ \Theta , f _ { 0 } , c _ { 0 } ]$

$$
\operatorname { C a s e \ A : } \quad \operatorname* { m i n } _ { \theta } ~ { \mathcal { L } }\tag{18}
$$

$$
\mathrm { C a s e \ B : } \quad \operatorname* { m i n } _ { \theta , f _ { 0 } , c _ { 0 } } \ C .\tag{19}
$$

In both cases an Adam (gradient based) optimizer from the optax [44] library is used to minimise $\mathcal { L }$ with a fixed learning rate of $1 0 ^ { - 3 }$ applied to $\theta _ { : }$ and in Case B this also applies to $f _ { 0 }$ and $c _ { 0 }$ . A ReLU is applied to $f _ { 0 }$ at each update step to enforce nonnegativity, since a negative value in the particle size distribution is physically infeasible. This, combined with the fact that the PSSD is rescaled to the correct mass during simulation, ensured that the $f _ { 0 }$ optimisation was physical and stable. For Case A, all models are trained for up to 500 epochs, with the final model parameters taken from the epoch yielding the smallest validation loss. Early stopping is implemented such that if the validation loss does not decrease over 100 epochs, training is terminated. In Case B, there is no data split so early stopping is based on the training loss rather than a validation loss.

## 4.6 Process Optimisation

## Parametrisation

As in Section 4.5, ϕ denotes the free variables optimised in a given task. In this task, this is always $T ( t )$ , and in some tasks $f _ { 0 }$ and $c _ { 0 }$ are optimised alongside it. $T ( t )$ is defined by 300 linearly spaced values in time across the length of each experiment.

## Bounds and penalties

Two classes of constraint are applied: the operating conditions are restricted to what the equipment can deliver, and the supersaturation is restricted to the range over which the kinetic model in use is valid (Section 4.3).

The temperature is constrained to the range spanned by the experimental data, $2 0 - 4 1 ^ { \circ } \mathrm { C } .$ , as a hard constraint applied by box projection (optax) at every optimisation iteration. The remaining constraints are imposed as soft penalties in the loss function, $\mathcal { P } _ { \dot { T } }$ and $\mathcal { P } _ { S } , \mathcal { P } _ { \dot { T } }$ limits the rate at which the temperature can change during an experiment and is set by the limits of the jacketed vessel: $| \mathrm { d } T / \mathrm { d } t | \leq 0 . 0 5 \ ^ { \circ } \mathrm { C } \operatorname* { m i n } ^ { - 1 }$ $\mathcal { P } _ { S }$ keeps the supersaturation above $S _ { \mathrm { m i n } }$ throughout, and below $S _ { \mathrm { m a x } }$ wherever a growth law is present.

## Optimisation procedure

$\Phi$ is optimised with Adam (optax) and is used to minimise the loss function

$$
\mathcal { L } ( \Phi ) = \mathcal { L } _ { \mathrm { t a s k } } ( \Phi ) + \mathcal { P } _ { \dot { T } } + \mathcal { P } _ { S } ,\tag{20}
$$

for 500–1000 iterations, where $\mathcal { L } _ { \mathrm { t a s k } }$ is defined per task, and $\mathcal { P } _ { \dot { T } } , \mathcal { P } _ { S }$ are the soft penalties. Their weights are ramped over the run rather than fixed so that the optimiser settles more smoothly towards the constraint boundaries. In the two trajectorytracking tasks, $\mathcal { L } _ { \mathrm { t a s k } }$ is the mean squared error between a target trajectory and the simulated one,

$$
\mathrm { M S E } \big ( \overline { { L } } _ { 1 } , \overline { { L } } _ { 2 } \big ) = \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { 2 } \sum _ { j = 1 } ^ { N } \big ( y _ { i , j } - \hat { y } _ { i , j } ( \Phi ) \big ) ^ { 2 } ,\tag{21}
$$

which takes the same form as the training loss (Eq. 17) but over the two size predictors only, with $y$ the target value at one of $N$ time points and ${ \hat { y } } ( \Phi )$ the prediction from the FVM solver. The target is defined per task below.

## 4.6.1 Synthetic Trajectory Optimisation

A target temperature profile, $T _ { \mathrm { r e f } } ( t )$ , is synthetically generated (as a stepped ramp or a smooth sigmoid) and simulated from the initial condition of E5 using the hybrid dissolution model (Section 2.2), giving synthetic $L _ { 1 }$ and $L _ { 2 }$ targets. The free variables are $\Phi = T ( t )$ , and $\mathcal { L } _ { \mathrm { t a s k } }$ is the MSE in Eq 21 evaluated against the synthetic target trajectory. $T ( t )$ is initialised as a straight line between the endpoints of the synthetic ‘truth’, $T _ { \mathrm { r e f } } ( t )$ , with the start and end temperatures randomly shifted to test whether the solution is robust to noise in the initial guess. This is repeated 20 times, and the optimised profiles along with their spread and simulation results are shown in Figure 6.

## 4.6.2 Experimental Trajectory Optimisation

Here, $\mathcal { L } _ { \mathrm { t a s k } }$ is the same MSE equation, Eq 21, and is evaluated against smoothed measured $L _ { 1 }$ and $L _ { 2 }$ trajectories from E5 rather than synthetic targets. Since the initial conditions of this experiment were not seen during training, $f _ { 0 }$ and $c _ { 0 }$ are included as free variables alongside $T ( t ) \colon \Phi = [ T ( t ) , f _ { 0 } , c _ { 0 } ]$ . T(t) is initialised to a smoothed version of the recorded temperature profile.

## 4.6.3 Product Aspect Ratio Optimisation

$T ( t )$ is optimised from the initial condition of E5 using the combined growthdissolution model from Section 4.3.2. $\Phi \ : = \ : T ( t )$ , and the task is to minimise the product aspect ratio subject to a soft volume constraint:

$$
\mathcal { L } _ { \mathrm { t a s k } } = \overline { { L } } _ { 1 , \mathrm { p r o d u c t } } / \overline { { L } } _ { 2 , \mathrm { p r o d u c t } } + \mathcal { P } _ { V } ,\tag{22}
$$

where ${ \mathcal { P } } _ { V }$ is a soft penalty enforcing $V _ { \mathrm { p r o d u c t } } / V _ { \mathrm { s e e d } } \geq 1$ , with $V$ the total crystal volume, so that the product volume cannot fall below that of the seed. This ensures

that the volume of product crystals must be greater than equal to that of the initial seed. T(t) is initialised as a linear ramp descending from dissolution (S < 1) into growth (S > 1).

## 4.7 Computational Benchmarking

We compare the computational performance of the original MATLAB solver from B¨otschi et al. [32] and our JAX based solver [31]. Between the solvers we compare the M1 and M4 kinetic models, representing 2 and 4 optimised parameters. We only report training times for M1 and M4 here for computational purposes - elsewhere in this work these models are used at their published coeficients [32]. Additionally we show JAX results for the hybrid dissolution model from Section 2.2, showing the case where we train only the NN and the case where we include the entire IC for assimilation.

## Forward simulation.

We report the mean time required to perform a single forward simulation, averaged over the thirteen experiments in the data set [32].

## Model Training.

MATLAB uses the fmincon optimiser with numerically estimated gradients at default settings, run to its own convergence criterion. JAX uses the Adam optimiser (optax) with automatic diferentiation with early stopping criteria (Section 4.5). For the JAX models we report both (1) wall-clock time to convergence under early stopping, and (2) wall-clock time for a fixed budget of 500 epochs. This isolates the convergence from how the computational cost scales with the number of parameters that are optimised and is not applicable to MATLAB which does not train in epochs. The parameter counts, and all times are reported in Extended Figure 7.

## Hardware.

Both the JAX solver and MATLAB solver are run on the same machine for computational performance evaluation. This computer is equipped with an AMD Ryzen 3900 12-core CPU and an NVIDIA RTX 4090 GPU. The JAX solver was able to use the GPU automatically, whereas the MATLAB code was .mex compiled and did not support GPU execution.

## References

[1] Edwards, D.A., Brenner, H., Wasan, D.T.: Interfacial Transport Processes and Rheology. Butterworth-Heinemann, Boston (1991)

[2] Rajagopalan, A.K., Avila, A.M., Rajendran, A.: Do adsorbent screening metrics predict process performance? A process optimisation based study for postcombustion capture of CO2. International Journal of Greenhouse Gas Control 46, 76–85 (2016) https://doi.org/10.1016/j.ijggc.2015.12.033 . Accessed 2025-06-20

[3] Ramkrishna, D.: Population Balances: Theory and Applications to Particulate Systems in Engineering. Academic Press, San Diego, CA (2000)

[4] Kunii, D., Levenspiel, O.: Fluidization Engineering, 2nd edn. Butterworth-Heinemann, Boston, MA (1991)

[5] McDufie, N.G.: Bioreactor Design Fundamentals. Butterworth-Heinemann, Boston (1991)

[6] Sekuli´c, D.P., Shah, R.K.: Fundamentals of Heat Exchanger Design, 2nd edn. John Wiley & Sons, Hoboken, NJ (2023)

[7] Akundi, A., Euresti, D., Luna, S., Ankobiah, W., Lopes, A., Edinbarough, I.: State of Industry 5.0—Analysis and Identification of Current Research Trends. Appl. Syst. Innov. 5(1), 27 (2022) https://doi.org/10.3390/asi5010027 . Accessed 2025-10-09

[8] Lorenz, E.N.: Deterministic Nonperiodic Flow. J. Atmospheric Sci. 20(2), 130– 141 (1963) https://doi.org/10.1175/1520-0469(1963)020⟨0130:DNF⟩2.0.CO;2 . Chap. Journal of the Atmospheric Sciences. Accessed 2026-09-02

[9] Ramkrishna, D., Singh, M.R.: Population balance modeling: Current status and future prospects. Annu. Rev. Chem. Biomol. Eng. 5, 123–146 (2014) https://doi. org/10.1146/annurev-chembioeng-060713-040241

[10] Cote, A., Erdemir, D., Girard, K.P., Green, D.A., Lovette, M.A., Sirota, E., Nere, N.K.: Perspectives on the Current State, Challenges, and Opportunities in Pharmaceutical Crystallization Process Development. Cryst. Growth Des. 20(12), 7568–7581 (2020) https://doi.org/10.1021/acs.cgd.0c00847

[11] de Albuquerque, I., Mazzotti, M., Ochsenbein, D.R., Morari, M.: Efect of needlelike crystal shape on measured particle size distributions. AIChE J. 62(9), 2974– 2985 (2016) https://doi.org/10.1002/aic.15270 . Accessed 2024-10-03

[12] Biri, D., Jaeggi, A., Prykhodko, O., Hicks, W., Perini, G., Mazzotti, M., Rajagopalan, A.K.: Is there a “right” particle size and shape measurement tool?: A comparative study of seven online and ofline devices on nine diferent particle

populations. Powder Technology 466, 121405 (2025) https://doi.org/10.1016/j. powtec.2025.121405 . Accessed 2026-09-03

[13] Wieland, F.-G., Hauber, A.L., Rosenblatt, M., T¨onsing, C., Timmer, J.: On structural and practical identifiability. Current Opinion in Systems Biology 25, 60–69 (2021) https://doi.org/10.1016/j.coisb.2021.03.005 arXiv:2102.05100 [stat]. Accessed 2026-03-31

[14] Chew, J.W., Andersson, R., Bierweiler, T., Ryttestal, P., Wik, T.: AI in chemical engineering: From promise to practice. AIChE J. 72(7), 70358 (2026) https:// doi.org/10.1002/aic.70358 . Accessed 2026-09-02

[15] Barhate, Y., Kang, Y.S., Nazemifard, N., Renner, C.B., Yang, Y., Papageorgiou, C., Nagy, Z.K.: Digital design of crystallization processes using statistical machine learning. Computers & Chemical Engineering 206, 109518 (2026) https://doi. org/10.1016/j.compchemeng.2025.109518 . Accessed 2026-01-12

[16] Raponi, A., Marchisio, D.: Multivariate optimization and inverse design of multiphase reactive systems via deep learning and 3D CFD–PBM simulations. Chemical Engineering Journal 531, 173908 (2026) https://doi.org/10.1016/j.cej. 2026.173908 . Accessed 2026-02-11

[17] Orosz, A., Neal, M., Rao, R., Roberts, C., Szil´agyi, B., Nagy, Z.K.: 2D population<sup>´</sup> balance modeling and ML-based multi-objective optimization for the crystallization process of resveratrol. AIChE J. 71(12), 70094 (2025) https://doi.org/10. 1002/aic.70094 . Accessed 2026-09-02

[18] Zheng, Y., Wang, X., Wu, Z.: Machine Learning Modeling and Predictive Control of the Batch Crystallization Process. Ind. Eng. Chem. Res. 61(16), 5578–5592 (2022) https://doi.org/10.1021/acs.iecr.2c00026 . Accessed 2026-03-26

[19] Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., Yang, L.: Physics-informed machine learning. Nat Rev Phys 3(6), 422–440 (2021) https: //doi.org/10.1038/s42254-021-00314-5 . Accessed 2024-10-10

[20] Rackauckas, C., Ma, Y., Martensen, J., Warner, C., Zubov, K., Supekar, R., Skinner, D., Ramadhan, A., Edelman, A.: Universal Diferential Equations for Scientific Machine Learning. arXiv (2021). https://doi.org/10.48550/arXiv.2001. 04385 . http://arxiv.org/abs/2001.04385 Accessed 2025-05-27

[21] Mukherjee, A., Zavala, V.M.: Physics-Constrained Machine Learning for Chemical Engineering. arXiv (2025). https://doi.org/10.48550/arXiv.2508.20649 . http: //arxiv.org/abs/2508.20649 Accessed 2025-08-29

[22] Sharma, N., Liu, Y.A.: A hybrid science-guided machine learning approach for modeling chemical processes: A review. AIChE J. 68(5), 17609 (2022) https: //doi.org/10.1002/aic.17609 . Accessed 2024-12-03

[23] Pessina, D., Tian, T., Watson, O., Heng, J.Y., Papathanasiou, M.M.: Transfer learning of data-driven crystallisation processes via constrained Neural Ordinary Diferential Equations. Digital Chemical Engineering 18, 100292 (2026) https: //doi.org/10.1016/j.dche.2026.100292 . Accessed 2026-07-20

[24] Wu, G., Yion, W.T.G., Dang, K.L.N.Q., Wu, Z.: Physics-informed machine learning for MPC: Application to a batch crystallization process. Chemical Engineering Research and Design 192, 556–569 (2023) https://doi.org/10.1016/j.cherd.2023. 02.048 . Accessed 2025-06-19

[25] Nai, D., Li, H., Grover, M.A., Medford, A.J.: Modeling Batch Crystallization under Uncertainty Using Physics-Informed Machine Learning. Ind. Eng. Chem. Res. 65(18), 9563–9577 (2026) https://doi.org/10.1021/acs.iecr.6c00599 . Accessed 2026-07-27

[26] von Stosch, M., Oliveira, R., Peres, J., Feyo de Azevedo, S.: Hybrid semiparametric modelling in process systems engineering: Past, present and future. Comput. Chem. Eng. 60, 86–101 (2014) https://doi.org/10.1016/j.compchemeng. 2013.08.008

[27] Lima, F.A.R.D., Rebello, C.M., Costa, E.A., Santana, V.V., Moares, M.G.F., Barreto, A.G., Secchi, A.R., Souza, M.B., Nogueira, I.B.R.: Improved modeling of crystallization processes by Universal Diferential Equations. Chemical Engineering Research and Design 200, 538–549 (2023) https://doi.org/10.1016/j.cherd. 2023.11.032 . Accessed 2025-05-07

[28] Pickles, T., Xiouras, C., Oner, M., Brown, C.J.: Bridging Data and The-<sup>¨</sup> ory: Hybrid Modelling for Pharmaceutical Crystallisation Process Development. Digital Discovery (2026) https://doi.org/10.1039/D6DD00342G . Accessed 2026- 08-04

[29] Thuerey, N., Holzschuh, B., Holl, P., Kohl, G., Lino, M., Liu, Q., Schnell, P., Trost, F.: Physics-Based Deep Learning. arXiv (2025). https://doi.org/10.48550/ arXiv.2109.05237 . http://arxiv.org/abs/2109.05237 Accessed 2026-09-02

[30] Bradbury, J., Frostig, R., Hawkins, P., Johnson, M.J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., Zhang, Q.: JAX: Composable Transformations of Python+NumPy Programs (2018). http: //github.com/google/jax

[31] Alsubeihi, M., Jessop, A., Moseley, B., Fonte, C.P., Rajagopalan, A.K.: Modern, Eficient, and Diferentiable Transport Equation Models Using JAX: Applications to Population Balance Equations. Ind. Eng. Chem. Res. (2025) https://doi.org/ 10.1021/acs.iecr.4c04208 . Accessed 2025-02-17

[32] B¨otschi, S., Rajagopalan, A.K., Morari, M., Mazzotti, M.: Feedback Control for the Size and Shape Evolution of Needle-like Crystals in Suspension. IV. Modeling

and Control of Dissolution. Cryst. Growth Des. 19(7), 4029–4043 (2019) https: //doi.org/10.1021/acs.cgd.9b00445

[33] Rajagopalan, A.K., Schneeberger, J., Salvatori, F., B¨otschi, S., Ochsenbein, D.R., Oswald, M.R., Pollefeys, M., Mazzotti, M.: A comprehensive shape analysis pipeline for stereoscopic measurements of particulate populations in suspension. Powder Technol. 321, 479–493 (2017) https://doi.org/10.1016/j.powtec.2017.08. 044

[34] Hornik, K., Stinchcombe, M., White, H.: Multilayer feedforward networks are universal approximators. Neural Networks 2(5), 359–366 (1989) https://doi.org/ 10.1016/0893-6080(89)90020-8 . Accessed 2025-06-24

[35] Ochsenbein, D.R., Schorsch, S., Vetter, T., Mazzotti, M., Morari, M.: Growth Rate Estimation of β l-Glutamic Acid from Online Measurements of Multidimensional Particle Size Distributions and Concentration. Ind. Eng. Chem. Res. 53(22), 9136–9148 (2014) https://doi.org/10.1021/ie4031852 . Accessed 2025-04-20

[36] Perini, G., Salvatori, F., Ochsenbein, D.R., Mazzotti, M., Vetter, T.: Filterability prediction of needle-like crystals based on particle size and shape distribution data. Separation and Purification Technology 211, 768–781 (2019) https://doi. org/10.1016/j.seppur.2018.10.042 . Accessed 2026-03-26

[37] Lovette, M.A., Browning, A.R., Grifin, D.W., Sizemore, J.P., Snyder, R.C., Doherty, M.F.: Crystal Shape Engineering. Ind. Eng. Chem. Res. 47(24), 9812–9833 (2008) https://doi.org/10.1021/ie800900f . Accessed 2026-08-10

[38] Snyder, R.C., Studener, S., Doherty, M.F.: Manipulation of crystal shape by cycles of growth and dissolution. AIChE J. 53(6), 1510–1517 (2007) https://doi.org/10. 1002/aic.11174 . Accessed 2026-08-10

[39] Ma, Y., Yuan, X., Yewale, A., Benyahia, B.: Recent advances in the digital transformation of crystallization process development and operation: Synergy between model- and AI-driven strategies. Particuology 110, 223–253 (2026) https://doi.org/10.1016/j.partic.2026.01.014 . Accessed 2026-03-26

[40] Lima, F.A.R.D., de Moraes, M.G.F., Barreto, A.G., Secchi, A.R., Grover, M.A., de Souza, M.B.: Applications of machine learning for modeling and advanced control of crystallization processes: Developments and perspectives. Digital Chemical Engineering 14, 100208 (2025) https://doi.org/10.1016/j.dche.2024.100208 . Accessed 2025-06-23

[41] Xiouras, C., Cameli, F., Quill´o, G.L., Kavousanakis, M.E., Vlachos, D.G., Stefanidis, G.D.: Applications of Artificial Intelligence and Machine Learning Algorithms to Crystallization. Chem. Rev. 122(15), 13006–13042 (2022) https://doi.org/10. 1021/acs.chemrev.2c00141 . Accessed 2024-10-03

[42] B¨otschi, S., Rajagopalan, A.K., Morari, M., Mazzotti, M.: An Alternative Approach to Estimate Solute Concentration: Exploiting the Information Embedded in the Solid Phase. J. Phys. Chem. Lett. 9(15), 4210–4214 (2018) https: //doi.org/10.1021/acs.jpclett.8b01998 . Accessed 2025-05-19

[43] LeVeque, R.J.: Finite Volume Methods for Hyperbolic Problems. Cambridge University Press, Cambridge, UK (2002). https://doi.org/10.1017/ CBO9780511791253

[44] DeepMind, Babuschkin, I., Baumli, K., Bell, A., Bhupatiraju, S., Bruce, J., Buchlovsky, P., Budden, D., Cai, T., Clark, A., Danihelka, I., Dedieu, A., Fantacci, C., Godwin, J., Jones, C., Hemsley, R., Hennigan, T., Hessel, M., Hou, S., Kapturowski, S., Keck, T., Kemaev, I., King, M., Kunesch, M., Martens, L., Merzic, H., Mikulik, V., Norman, T., Papamakarios, G., Quan, J., Ring, R., Ruiz, F., Sanchez, A., Sartran, L., Schneider, R., Sezener, E., Spencer, S., Srinivasan, S., Stanojevi´c, M., Stokowiec, W., Wang, L., Zhou, G., Viola, F.: The DeepMind JAX Ecosystem (2020). http://github.com/google-deepmind

## Extended Figures

![](images/b692cc65471a9b81ce1f90055dd2624cb3c7970b1edd0debd78bddb0a0142a7f.jpg)

![](images/5e0549fde2aba7abb00cb8270bf11444ca5f7d15b6223cc422025888a41e72fe.jpg)  
Fig. 4 Experimental dataset and initial condition assimilation. (a) All experimental trajectories in the dataset (E1–E13), labelled by training/validation/held-out split, for the operating condition $T ( t )$ and the three target metrics: solute concentration $^ { c , }$ average crystal length ${ \overline { { L } } } _ { 1 } ,$ and width ${ \overline { { L } } } _ { 2 } .$ . For each target trajectory the measured initial condition (dark blue) and the assimilated initial condition (orange cross) are marked. (b) Full initial and assimilated particle size and shape distributions (PSSDs) for each experiment, with the corresponding experimental seed mass. Assimilated PSSDs are similar in shape to the original measurement. The two held out experiments do not have assimilated $\mathrm { P S S D s } ,$ since they were not included in the hybrid model training.

![](images/e1f2751ed33ee91f6b917df62da5e4fb06b746b98d1d9069d294b54ed0f68c89.jpg)

![](images/eca0bb55b9ffeca79780e0e6f2dcf50816b68f4b0854b4f1bfd5a04bfe205cf8.jpg)  
Fig. 5 Learned functional forms and predictive performance. (a) Learned dissolution rate functional forms $D _ { 1 }$ and $D _ { 2 }$ vs. supersaturation S (fixed $T = 2 8 ^ { \circ } \mathrm { C }$ , left) and vs. temperature T (fixed $S = 0 . 9 9 5$ , right), for the M1 empirical baseline and hybrid models of varying architecture $( 1 \times 5 , 2 \times 5 , 3 \times 2 0$ hidden layers × nodes), with and without joint IC assimilation. Shaded bands give the spread across the 20-member ensembles where multiple models were trained. Importantly, this shows how the hybrid model trained on the measured initial condition can learn a non-physical form, with the rate never approaching zero at $S = 1 . \ ( \mathrm { b } )$ Predicted vs. measured trajectories for the four validation experiments (E6, E7, E8, E13) across all three target metrics $( c \overline { { L } } _ { 1 } , \overline { { L } } _ { 2 } )$ . Legend values give the combined trajectory loss across these experiments for each model. IC assimilation improves the fit for both the empirical (M1) and hybrid models, with the largest gain for the hybrid model, and the improvement is consistent across network architecture.

b  
![](images/891409d19a78a0a8588733a7650a101dfa05bb68e8b81c71b539040d53437ec8.jpg)

![](images/ff202cc5952c2326d012cd5c02d304fdc715a102c3296dc262c8e5e797bafca4.jpg)

![](images/27904433668b80d54ab110c57ba34ca05b3b6e1b7d55e720fee252e1aced685e.jpg)  
Fig. 6 Optimisation verification. (a) Synthetic recovery of the operating condition $T ( t )$ for two target profiles (stepped ramp, sigmoid), using the frozen hybrid dissolution model with the IC fixed at its measured value (E5). Grey lines show 20 random linear-ramp initial guesses. The recovered mean (blue) and min/max envelope (shaded) closely match the known ground truth (dotted) in $T ( t )$ and in the resulting $\overline { { L } } _ { 1 } .$ , L<sub>2</sub> trajectories, with MAE as reported. This isolates the optimiser’s ability to recover a known solution from any error due to model or IC uncertainty. (b) Experimental trajectory optimisation on the held-out experiment E5. Left: optimised $T ( t )$ and resulting S(t) with the IC fixed at its measured value (dashed) vs. with the IC freed for joint assimilation (solid), against the initial guess (dotted) and experimental measurement (squares). $S ( t )$ remains within the $S _ { \mathrm { m i n } }$ bound (grey line) in both cases. Right: corresponding $\overline { { L } } _ { 1 } \ : \mathrm { ( p i n k ) }$ and $\overline { { L } } _ { 2 }$ (orange) trajectories. Freeing the IC for assimilation reduces the $T ( t )$ MAE from $0 . 1 0 ^ { \circ } \mathrm { C }$ to $0 . 0 5 ^ { \circ } \mathrm { C }$ and tracks the experimental trajectories more closely than either the fixed-IC optimisation or the initial guess.

![](images/d37acb1647ce63ac2c2ec8b315b76f6edf8e46cdb95d65d724571f35d07b3023.jpg)  
Fig. 7 Computational performance. Wall-clock time (log scale) for forward simulation and optimisation, comparing the MATLAB workflow of B¨otschi et al. [32] against the JAX diferentiable workflow of this work, for the M1 and M4 empirical models and the neural-network hybrid model $( 3 \times 2 0 )$ , with and without IC assimilation. For JAX, the green segment gives time to convergence under early stopping, and the stacked orange segment (bold total above) gives the wall-clock time for a fixed 500 epochs. MATLAB does not train in epochs and has no equivalent segment. The table gives the corresponding model, IC, and total parameter counts for each case. Forward simulation is ∼13× faster in $\operatorname { J A X } \ ( 0 . 0 6 \mathrm { s } \ \mathrm { v s . } \ 0 . 7 6 \mathrm { - } 0 . 8 2 \mathrm { s } )$ . Adding $4 . 8 \times 1 0 ^ { 5 } \ \mathrm { I C }$ parameters for joint assimilation $( \mathrm { N N } \ 3 \times 2 0 ,$ , 942 → 476,153 total parameters) changes JAX convergence time only marginally (291 s $ 2 8 2 \mathrm { s } )$ and the fixed 500-epoch time by <15% $( 3 2 2 \mathrm { s }  3 6 5 \mathrm { s } )$ , despite the five-order-of-magnitude increase in parameter count. This is consistent with gradient cost under automatic diferentiation being efectively independent of parameter count. MATLAB optimisation of the far smaller $\mathrm { M 1 / M 4 }$ models takes 907 s and 3823 s respectively.