# Butterfly Effect and the Kinetic Energy Cascade in Probabilistic Machine Learning Weather Prediction Models

Jiakai Chen¹, Joel Oskarsson2, Simon Driscoll3, and Sebastian Schemm³

1Department of Physics, University of Cambridge, Cambridge, UK 2ETH AI Center, ETH Zurich, Zurich, Switzerland

3Department of Applied Mathematics and Theoretical Physics, University of Cambridge, Cambridge, UK

## Key Points:

• Purely machine-learning-based models produce realistic kinetic energy spectra magnitudes but do not exhibit expected upscale energy transfer

• The stochastic ensemble configuration of the hybrid NeuralGCM model reproduces upscale kinetic energy transfer but encoder-level noise injection underestimates mesoscale energy

• Machine-learning-based weather models exhibit upscale error growth but fail to reproduce the rapid initial ensemble-spread growth of small-scale errors

## Abstract

This study analyses kinetic energy (KE) spectra, difference kinetic energy (DKE) spectra, and signatures of KE transfer across spatial scales in four state-of-the-art probabilistic machine learning weather prediction (MLWP) models—NeuralGCM-ENS, Four-CastNet 3, AIFS-ENS, and GenCast. Results are compared with those from the physicsbased numerical weather prediction model IFS-ENS. While NeuralGCM-ENS successfully reproduces the expected upscale transfer of KE, noise injection at its encoder stage underestimates mesoscale KE. Conversely, AIFS-ENS, GenCast, and FourCastNet 3 produce realistic KE spectral magnitudes but do not capture the expected upscale transfer of KE. In particular, AIFS-ENS and GenCast, which employ spatially uncorrelated stochastic perturbations, exhibit enhanced accumulation of KE at high wavenumbers. All examined models exhibit upscale error growth, reflected by the progressive shift of the DKE spectral peak toward larger wavelengths over time. However, the MLWP models struggle to reproduce the rapid initial growth of ensemble spread at small spatial scales associated with the butterfly effect. The results show that MLWP models can misrepresent the known scale transfer of kinetic energy despite producing skilful weather forecasts.

## Plain Language Summary

This paper investigates whether cutting-edge AI weather models reproduce established principles of atmospheric fluid dynamics. We tested four recent AI weather models to see if they can reproduce the “butterfly effect"—where tiny errors grow over time to cause large changes in weather patterns—and how they transfer energy between different length scales. We found that while AI models correctly show errors growing to larger scales over time, they struggle to capture the rapid initial growth of very small errors. Additionally, while some purely data-driven models produce realistic amounts of total energy, they fail to show realistic transfer of energy between length scales. This suggests that future AI weather models require better physical constraints to reliably predict weather patterns and extreme events.

## 1 Introduction

Reliable weather forecasting supports early warnings and planning across sectors such as transport, energy, and agriculture. Traditional numerical weather prediction (NWP) solves the equations governing atmospheric evolution but remains computationally intensive despite advances in computing, data assimilation, and parameterisation (Lynch, 2008; Ritchie et al., 1995; Bauer et al., 2015a, 2015a, 2020).In recent years, data-driven methods for weather forecasting have emerged as a promising new direction (Weyn et al., 2019; Keisler, 2022). These models train deep neural networks on historical atmospheric datasets such as ERA5 (Hersbach et al., 2020) to predict atmospheric evolution from an initial condition. MLWP models' out-of-sample forecasts, usually measured by metrics such as root-mean-square error (RMSE) and Continuous Ranked Probability Score (CRPS), have been shown to outperform predictions from state-of-the-art NWP models for up to 10 days (Bi et al., 2023; Lam et al., 2023; Rasp et al., 2024; Ben Bouallègue et al., 2024; Bodnar et al., 2025; Han et al., 2024) also at high resolution. Crucially, once trained, these data-driven models operate up to 104 to 10⁵ times faster than modern NWP models (Bi et al., 2023; Lam et al., 2023).

Most early MLWP models and studies of their physical consistency focus on deterministic forecasting, in which a single prediction is produced from a given initial atmospheric state. These models are generally trained without explicit physical or dynamical constraints, and the physical consistency of their forecasts remains uncertain. For example, it was observed that many MLWP models like Pangu-Weather (Bi et al., 2023)

and Aurora (Bodnar et al., 2025) were not able to produce the rapid growth in ensemble variance expected from the butterfly effect (Lorenz, 1963) when their inputs are given tiny perturbations (Selz & Craig, 2026). It was also found that MLWP models struggle in representing dynamical balance relationships of geostrophic flows (Bonavita, 2024). Furthermore, evaluations of deterministic MLWP models showed that, despite accurately capturing the locations of weather systems, their forecasts were overly smooth and tended to underestimate extreme wind and precipitation intensities (Adamov et al., 2025).

Unlike deterministic models, recent probabilistic MLWP models such as GenCast (Price et al., 2023), FourCastNet 3 (Bonev et al., 2025), and stochastic NeuralGCM (hereafter NeuralGCM-ENS) (Kochkov et al., 2024) generate ensembles of atmospheric trajectories. Traditional ensemble forecasting perturbs both the initial conditions and the parameters governing subgrid-scale processes in NWP models (Kalnay, 2012; ECMWF, 2019; Palmer, 2019; Yamaguchi et al., 2018), whereas some probabilistic MLWP models learn distributions over possible trajectories and generate members from a common initial condition through internal stochastic sampling (Bonev et al., 2025). These probabilistic models often employ fundamentally different architectures from deterministic models. For example, GenCast implements a diffusion-based model, and has been shown to outperform earlier deterministic models in terms of producing higher-resolution features and more physically realistic power spectra (Price et al., 2023). This paper evaluates the physical consistency of recent probabilistic MLWP weather models, focusing on the butterfly effect and multi-scale kinetic energy transfer.

## 2 Characteristics of the Kinetic Energy and Difference Kinetic Energy Spectrum

## 2.1 Butterfly Effect and DKE

The butterfly effect describes the amplification of small initial perturbations into substantial forecast differences (Lorenz, 1963). Globally averaged difference kinetic energy (DKE), defined here as the sum of the ensemble variances of the zonal and meridional winds, measures this ensemble spread (for a more formal definition see Section 3.2). Weak perturbations initially grow rapidly through convection before spreading through gravity waves and geostrophic adjustment and subsequently growing upscale at synoptic scales (Rotunno & Snyder, 2008; Leung et al., 2020; Selz & Craig, 2023, 2015). Their initial growth rate increases as the perturbation magnitude decreases (Judt, 2018). However, deterministic MLWP models such as Pangu-Weather (Bi et al., 2023) have been found not to reproduce this rapid initial growth (Selz & Craig, 2023, 2026).

In the spherical harmonic representation of DKE, the butterfly effect appears as a shift of the spectral peak toward larger scales as small-scale errors saturate, a process termed “upscale error growth" (Baumgart et al., 2019; Selz et al., 2022; Sun & Zhang, 2016). Numerical experiments show that, when initialized from slightly perturbed initial states, the peak of the DKE spectrum produced by deterministic AI models such as Pangu-Weather (Bi et al., 2023) and Aurora (Bodnar et al., 2025) exhibits shifts inconsistent with the butterfly effect (Selz & Craig, 2026, 2023).

These studies primarily evaluate deterministic MLWP models by constructing ensembles initialized with prescribed perturbations. The choice of initial perturbations can affect ensemble spread and probabilistic forecast skill. For Pangu-Weather, random-field perturbations designed to preserve linear balances yielded lower CRPS than Gaussiannoise and IFS-derived perturbations, although all three approaches underestimate forecast uncertainty (Bülte et al., 2026).

This study focuses on recent probabilistic models, which are designed and trained to generate ensembles whose spread represents the range of plausible forecast outcomes and are therefore expected to produce more realistic ensemble spread than manually perturbed deterministic models. A recent paper evaluated the ability of GenCast, a probabilistic model, to reproduce the butterfly effect through analysis of its DKE spectra (Kim et al., 2026), suggesting that GenCast underestimates the growth rates of DKE spectrum towards the extremes of very large or very small length scales. Our study extends existing work by comparing multiple probabilistic MLWP models that differ in both their architectures—including transformers and graph neural networks—and their probabilistic formulations, such as diffusion-based generation and noise injection with CRPS-based training.

## 2.2 Upscale energy cascade and kinetic energy spectrum

Rapid rotation and stable stratification make free-atmospheric flow behave approximately as two-dimensional turbulence, except near the planetary boundary layer (Charney, 1971; Vallis, 2017; Stull, 2012). Two-dimensional turbulence exhibits a net upscale transfer of energy from smaller to larger length scales (Gkioulekas & Tung, 2007; Métais et al., 1996; Smith et al., 2002; Xia et al., 2011; Kolmogorov, 1995), a phenomenon referred to as upscale energy cascade. It was also observed experimentally (Maltrud & Vallis, 1991; Smith & Yakhot, 1994) and numerically (Xiao et al., 2009b; Chen et al., 2006b) that at long wavelengths, 2D turbulence follows the Kolmogorov-Kraichnan scaling (Kraichnan, 1967; Vallis, 2019) where $E ( n ) \sim n ^ { - 5 / 3 }$ for wavenumber $n ,$ while in the limit of short wavelengths $E ( n )$ follows a steeper scaling law of $E ( n ) \sim n ^ { - 3 }$ . In the atmosphere, the transition occurs near $n \sim 1 0 0$ , corresponding to approximately 400 km (Burgess et al., 2013). Recent numerical experiments observed that many deterministic MLWP models reproduce the expected KE spectra at large scales but underestimate small-scale KE (Li et al., 2025; Bonavita, 2024).

While most existing studies have focused on deterministic models, recent probabilistic models employ distinct architectures and stochastic formulations that may produce more realistic KE spectra. For example, NeuralGCM-ENS injects noise at every time step of the atmospheric state's evolution by seeding its machine-learned physics module with new random fields, which could shift the KE spectra of the predicted atmospheric states. To date, it remains open to what extend the new probabilistic models reproduce the cross-scale energy transfer and the butterfly effect.

## 3 Experimental design

## 3.1 Data

All ML weather models in this study were initialised from the ERA5 reanalysis dataset (Hersbach et al., 2020) at 00:00 UTC on 26 June 2021 and run forward for five days. This initialisation time was chosen because it coincided with a period of strong convective activity over the North American continent(Selz & Craig, 2026). For each MLWP model, we analysed a 50-member ensemble, matching the ensemble size of IFS-ENS. Before analysis, we conservatively remapped the forecast fields from all models onto a common N360 Gaussian grid with 360 longitudes and 180 latitudes (equivalent to a 1° global resolution) using the pyshtools library, thereby preserving the global means of the kinetic energy and difference kinetic energy fields.

## 3.2 Calculations of Kinetic Energy and Difference Kinetic Energy

We refer to the kinetic energy per unit mass at each grid point as simply kinetic energy (KE), and the sum of variances across ensemble members of the east-west wind (u) and north-south wind (v) at each grid point the difference kinetic energy (DKE), both in units of $\mathrm { m ^ { 2 } s ^ { - 2 } }$

$$
\mathrm { K E } ( { \bf r } , \tau ) = \frac { 1 } { 2 } \left( u ^ { 2 } + v ^ { 2 } \right)\tag{1}
$$

$$
\mathrm { D K E } ( \mathbf { r } , \tau ) = \mathrm { v a r } ( u ) + \mathrm { v a r } ( v )\tag{2}
$$

In all subsequent analyses, we examine velocity fields on the 2D surface at the 500 hPa pressure level. This level has been extensively studied in meteorological research because it lies in the mid-troposphere, where large-scale atmospheric flow strongly influences and steers surface weather systems (George & Gray, 1976; Torn et al., 2018; Chan, 1985; Ding et al., 2023).

We calculate the KE spectra $\mathrm { K E } ( n , \tau )$ for a given wavenumber n and lead time τ following the methodology of NCAR Technical Note NCAR/TN-388+STR (Jakob et al., 1993), which we outline briefly below.

Let $\delta ( { \bf r } , \tau )$ be the divergence and $\zeta ( { \bf r } , \tau )$ the magnitude of curl of the velocity field at each grid point. Using the Python library pyshtools (Wieczorek & Meschede, 2018), we then calculate the spherical harmonic coefficients $\zeta _ { n } ^ { m }$ and $\delta _ { n } ^ { m }$ for the decomposition into spherical harmonic modes $Y _ { n } ^ { m } ( \lambda , \theta )$ of degree n and order m. It can then be shown (Jakob et al., 1993) that the KE spectra at wavenumber $n , K E _ { n }$ , is given by:

$$
\mathrm { K E } ( n , \tau ) = \frac { a ^ { 2 } } { 4 n ( n + 1 ) } \left[ \zeta _ { n } ^ { 0 } ( \zeta _ { n } ^ { 0 } ) ^ { * } + \delta _ { n } ^ { 0 } ( \delta _ { n } ^ { 0 } ) ^ { * } + 2 \sum _ { m = 1 } ^ { n } \zeta _ { n } ^ { m } ( \zeta _ { n } ^ { m } ) ^ { * } + 2 \sum _ { m = 1 } ^ { n } \delta _ { n } ^ { m } ( \delta _ { n } ^ { m } ) ^ { * } \right]\tag{3}
$$

with a being the average radius of Earth, and asterisks denoting complex conjugation. The decomposition of $\mathrm { D K E } ,$ denoted by $\mathrm { D K E } ( n , \tau )$ , into spectral components is done in an identical manner. Following conservative remapping to the N360 grid, we compute the spectra over degrees $1 \leq n \leq 1 7 9$ , where $n = 1 7 9$ is the Nyquist limit imposed by the 360-point longitudinal grid

## 3.3 Models

## 3.3.1 IFS-ENS

IFS-ENS is ECMWF's 50-member NWP ensemble, initialized using the Ensemble of Data Assimilation (Owens & Hewson, 2018; Lang et al., 2019). We obtained the IFS-ENS forecasts from the WeatherBench 2 platform (Rasp et al., 2024), which sourced them from the TIGGE archive (Bougeault et al., 2010). The forecasts have a horizontal resolution of 0.25° and are provided at 6-hour intervals.

## 3.3.2 NeuralGCM-ENS

NeuralGCM is a hybrid model combining a physics-based dynamical core for largescale flow with learned representations of subgrid processes (Kochkov et al., 2024). NeuralGCM-ENS (the ensemble version of NeuralGCM) was trained using a combination of grid-point and spectral CRPS losses, and generates an ensemble of forecasts from a single, unperturbed initial condition. The spectral component was restricted to wavenumbers up to 80 because higher wavenumbers are filtered for stability in its dynamical core (Kochkov et al., 2024). Stochasticity is introduced through Gaussian random fields with learned spatial and temporal correlations, which are supplied as additional inputs to both the encoder and the learned physics module. We ran NeuralGCM-ENS to obtain a 50-member ensemble, matching the size of IFS-ENS ensemble. The highest resolution available for NeuralGCM is 1.4°, with outputs at 1-hour intervals.

## 3.3.3 FourCastNet 3

FourCastNet 3 (FCN3) is a purely data-driven probabilistic AI model (Bonev et al., 2025). Building on the spectral-loss approach previously used in NeuralGCM, FCN3 combines spatial CRPS with spectral CRPS computed from spherical harmonic coefficients. Unlike NeuralGCM, FCN3 applies the spectral loss across all resolved frequencies and variables, encouraging realistic spatial correlations and power spectra (Kochkov et al., 2024; Bonev et al., 2025). FCN3 introduces stochasticity through a latent field with spatial and temporal correlations generated by spherical diffusion. We ran FCN3 at a resolution of 0.25°, with forecast fields produced at 6-hour intervals.

## 3.3.4 GenCast

GenCast is a data-driven, transformer-based diffusion model that generates atmospheric states from spatially uncorrelated Gaussian noise, conditioned on the two preceding states (Price et al., 2023). GenCast was reported to produce more realistic power spectra than an ensemble generated by perturbing other MLWP models like GraphCast (Lam et al., 2023), addressing the common problem of excessive smoothing and blurring of atmospheric features. We downloaded GenCast forecast data from Google DeepMind's WeatherNext platform (DeepMind Technologies Limited, 2024). The forecasts have a horizontal resolution of $0 . 2 5 ^ { \circ }$ and are provided at 12-hour intervals.

## 3.3.5 AIFS-ENS

AIFS-ENS (Lang et al., 2026) is the probabilistic version of ECMWF's Artificial Intelligence Forecasting System, a transformer-based weather forecasting model trained to produce ensemble forecasts using a loss function based on the Continuous Ranked Probability Score (CRPS). Training with CRPS effectively addressed blurring as it removed the need to collapse uncertain outcomes into a single mean state (Lang et al., 2026). For each ensemble member, AIFS-ENS samples spatially uncorrelated Gaussian noise on the latent grid used by its transformer processor. We use version 1.0 of AIFS-ENS, which gives output at a resolution of 0.25° and 6-hour intervals.

## 4 Results

## 4.1 Difference Kinetic Energy

## 4.1.1 Spectral DKE growth

Fig. 1 shows the growth of DKE at spherical harmonic wavenumbers of n = 1 (solid lines), n = 10 (dashed lines) and n = 100 (dotted lines) separately. Studies using highresolution NWP models show that weak initial perturbations first undergo rapid amplification through convective processes at small scales (Lorenz, 1969; Selz et al., 2022; Selz & Craig, 2023). The resulting errors spread to larger scales through divergent motions and gravity-wave propagation, before being transferred to balanced flow through geostrophic adjustment and subsequently growing at synoptic scales (Selz & Craig, 2015; Zhang et al., 2007). Weakly perturbed ensembles therefore exhibit rapid initial error growth and eventually approach the saturation levels of strongly perturbed ensembles. By contrast, strong initial perturbations immediately introduce substantial synoptic-scale differences, which dominate their DKE from the beginning.

In Fig. 1, we see that for n = 1 (λ ≈ 40000 km) and n = 10 (λ ≈ 4000 km), NeuralGCM-ENS, AIFS-ENS and FCN3 which have lower DKE than IFS-ENS at the first forecasted time step showed a fast initial growth in DKE as expected, converging with IFS-ENS after around 12 hours, and grew at the same rate afterwards. GenCast has a lower initial DKE level comparable to IFS-ENS from the first time step, and showed similar growth rate as the other models. Hence ensemble spread growths realistically at large length scales for all three models.

![](images/8d35ef89d22732d4d175c75e143fb577f4076b84f884cf1d2d4306b5a8401dd4.jpg)  
Figure 1. Time evolution of spectral DKE at different spherical harmonic wavenumbers. Solid, dashed, and dotted lines represent n = 1 (λ ≈ 40000 km), n = 10 (λ ≈ 4000 km), and n = 100 (λ ≈ 400 km), respectively.

On the other hand, at higher wavenumbers the models show a different behaviour. At $n = 1 0 0 \ ( \lambda \approx 4 0 0 \ \mathrm { k m } )$ , the initial level of DKE provided by ECMWF's Ensemble Data Assimilation (EDA) (Lang et al., 2019) in IFS-ENS is already saturated hence we see the DKE of IFS-ENS staying constant in time, which matches results from previous studies using other high resolution NWP models initiated with EDA ensemble (Selz & Craig, 2023). AIFS-ENS, FCN3 and GenCast have a similar initial DKE level as IFS-ENS, hence we expect their spectrum to show saturation as well, which was observed in Fig. 1. However, while NeuralGCM-ENS has an initial DKE more than an order of magnitude weaker than IFS-ENS, it did not show the expected fast growth to saturation. Similar behaviour was observed in previous studies when the deterministic MLWP model Pangu-Weather was initialized with low DKE (Selz & Craig, 2026, 2023). This behaviour of NeuralGCM-ENS may be due to its transition of using numerical solver for large scale processes to machine-learning parameters for sub-grid processes at these high wavenumbers. The resolution of NeuralGCM-ENS is 1.4° (Kochkov et al., 2024), corresponding to a wavenumber of $n \sim 1 3 0$ , which may explain its loss in physical consistency and lack of DKE growth as we approach that length scale. Another possible explanation is the spectral treatment used in NeuralGCM-ENS: spherical harmonic modes above wavenumber 80 were excluded from its spectral CRPS loss because these modes are filtered for stability in the dynamical core (Kochkov et al., 2024). A plot of the total DKE integrated over all wavenumbers is given in Fig. S1.

## 4.1.2 Upscale error growth

Upscale error growth, which represents the growth of ensemble spread and forecast uncertainty, is evident for all four models in Fig. 2, where the DKE spectral peak gradually shifts towards lower wavenumbers over time. At small spatial scales, however, NeuralGCM-

![](images/29c2976f12d17e3de06c77f5548111d3b957d051a07c57e3c6c12a6bb68209cf.jpg)  
Figure 2. DKE spectra at different forecast lead times.

ENS produces neither the expected increase in high-wavenumber DKE nor the expected DKE spectral slope. Instead, DKE decreases during the first few hours before saturating at a lower level (red dotted line in Fig. 1). In particular, NeuralGCM-ENS's spectra at the first time step show an almost flat spectrum for $n > 1 0 0 .$ which is indicative of white noise. This suggests that the encoder-level noise injection, which provides the initial source of ensemble spread at $\tau \ = \ 0$ , introduces a white-noise-like spectrum at high wavenumbers rather than perturbations consistent with balanced atmospheric dynamics. In Fig. S2 we show that the divergent and rotational KE spectrum of NeuralGCM-ENS merge at around $n > 1 0 0$ , further supporting that white noise is produced at high wavenumbers.

## 4.2 Kinetic Energy Spectra

## 4.2.1 Transfer of KE

To investigate whether each model reproduces the upscale energy cascade, we plot $\Delta \mathrm { K E } ( n , \tau )$ , the change of spectral KE with respect to the spectra at $\tau = 0 .$ For example, wavenumbers with negative $\Delta \mathrm { K E } ( n , \tau )$ shows a loss in KE from that length scale with respect to initial conditions. Fig. 3 shows the plots for $\tau = 2 4$ hours and $\tau = 1 0 8$ hours. $\Delta \mathrm { K E } ( n , \tau )$ was calculated for integer values of n and subsequently smoothed using a moving average over ten nearest wavenumbers. The corresponding unsmoothed spectrum is provided in Fig. S5.

![](images/cfed6525fda49405cffb4137f81cd9618bdd763274cea70d807fc4c18d4f91bc.jpg)

![](images/a0d3579e6004c0897ae125adc6072128b4826661ad11b698594d0ac2fc8a6077.jpg)  
Figure 3. Change of spectral KE with respect to spectra at $\tau = 0$

Initial conditions used in ensemble prediction incorporates noise due to uncertainties in observations such as instrument errors (Lang et al., 2019), hence are largely uncorrelated between different points and corresponds to energy at short length scales. The subplot for IFS-ENS shows the expected upscale energy transfer, where for high wavenumbers we see a consistently negative ∆KE at $\tau = 1 0 8$ hours except at very large scales.

For all four MLWP models, the KE transport matches IFS-ENS closely at low wavenumbers below approximately $n = 5 0$ but constrast substantially at higher wavenumbers. At very large spatial scales all models display positive values (below wavenumber n \~

8at $\tau \ = \ 2 4 \ \mathrm { h o u r s } )$ , which increase in both height and width as lead time progresses to $\tau = 1 0 8$ hours, indicating the transfer of KE towards larger length scales.

At synoptic scales, there is a narrow plateau of positive values around $n \sim 3 0$ at $\tau = 2 4$ hours, which is observed in all models including IFS-ENS. It is later absent at forecast lead time of $\tau = 1 0 8$ hours and it reflects the growth of atmospheric features at that characteristic scale, such as synoptic weather systems, which typically have length scales of order 1000–4000 km (Uccellini & Johnson, 1979)

At the smallest spatial scales the behaviour differs between IFS-ENS and all tested models. For example, NeuralGCM-ENS reproduces the expected negative tail at high wavenumbers, which indicates KE transport towards larger scles, although with a lower magnitude at the extreme small scales and with a different slope. This is to be contrasted with the graphs of FCN3 and GenCast. FCN3 exhibits a KE spectrum at small scales (≈ at wavenumbers $\mathrm { ~ n ~ } \geq 1 0 0 )$ that fluctuates around zero, indicating that KE is not transported properly to larger scales and with an artificial peak at $\mathrm { n = 1 0 ^ { 2 } }$ . This suggest KE accumulation at the meso-scales. The positive tails of GenCast and AIFS-ENS suggest that the models may be generating unrealistic meso and macro-scale features and kinetic energy contained at these scales. This could be due to the use of spatially uncorrelated noise in the two models, in contrast to FCN3 and NeuralGCM-ENS. GenCast uses a new sample of isotropic Gaussian white noise to generate predictions at each forecast step through its diffusion model. AIFS-ENS samples an independent Gaussian white-noise field over each grid point, using the spatially uncorrelated noise field as latent variable inputs into its transformer. FCN3 incorporates stochasticity through a latent random field that captures spatio-temporal correlations. This is done using a spherical diffusion process, in which independent Gaussian random coefficients are combined with spherical harmonic basis functions to generate latent fields with fixed spatial and temporal correlation length scales. This approach appears to generate a more realistic KE slope in the spectrum compared to the other tested purely data-driven models. The hybrid NeuralGCM-ENS also constructs its latent Gaussian random fields in a spherical harmonic basis, with learned spatial and temporal correlation length scales and this approach generates no KE accumulation at small scales but still an unrealistic spectral KE slope. In Fig. S3 we integrated the KE spectrum for $n > 7 0$ to highlight the increasing mesoscale KE accumulated in GenCast and AIFS-ENS, and the decreasing trend in NeuralGCM-ENS and IFS-ENS.

This result suggests that NeuralGCM-ENS facilitates a somewhat more realistic interaction and upscale transfer of energy between different length scales but the spectral slope does not agree with expectations from basic fluid dynamics. This could be due to the hybrid approach taken by NeuralGCM-ENS, where the neural network is only used to parametrise sub-grid processes like cloud formation.

## 4.2.2 Magnitude and scaling of KE spectra

In Fig. 4, we plotted the globally integrated KE spectrum at each time for the four models. We see that the data driven models FCN3, AIFS-ENS and GenCast closely follow the expected $E ( n ) \sim n ^ { - 3 }$ scaling at high wavenumbers, which is also observed in IFS-ENS. For FCN3 and GenCast, this agreement is consistent with previous evaluations showing that they produce sharp forecasts with realistic power spectra, associated with their spectral-CRPS and diffusion-based formulations, respectively (Bonev et al., 2025; Price et al., 2023). Although AIFS-ENS does not use an explicit spectral loss, previous evaluations found that its spectra remained stable with lead time and did not exhibit the progressive loss of high-wavenumber energy observed in MSE-trained models(Lang et al., 2026).

By contrast, NeuralGCM-ENS follows the $n ^ { - 3 }$ scaling only up to approximately $n = 7 0$ and underestimates KE at higher wavenumbers. This behaviour is likely related to its spectral treatment during training: the spectral CRPS loss was restricted to wavenumbers up to 80 because higher-wavenumber modes are filtered for stability in its dynamical core (Kochkov et al., 2024). The close correspondence between the onset of KE underestimation and this cutoff suggests that the filtering may contribute to the observed loss of high-wavenumber energy. Figure S4 provides a side-by-side comparison of the KE spectra of all models at τ = 108 hours.

![](images/1e5530b8eb1fe6f504e737da1b45e89d1ea6e3898f5fd15c6cdfb0b534357dec.jpg)  
Figure 4. Globally integrated KE spectrum

Overall, our results suggest that although the data-driven models FCN3, AIFS-ENS and GenCast produced more realistic KE magnitudes, they were unable to reproduce realistic transfers of energy across spatial scales. In contrast, NeuralGCM-ENS exhibited upscale energy transfer, but underestimated the KE spectrum at high wavenumbers.

## 5 Conclusions

This study investigates the physical consistency of state-of-the-art probabilistic MLWP weather models through the analysis of (a) kinetic energy and (b) difference kinetic energy spectra with a focus on the upscale growth of energy, kinetic energy spectra shape and evolution and change in kinetic energy as function of wave number and forecast time.

Concerning the upscale energy growth, all MLWP models examined, namely NeuralGCM-ENS, FCN3, AIFS-ENS and GenCast, exhibit upscale error growth in their DKE spectra (Fig. 2). This suggests that the models are broadly able to reproduce large-scale atmospheric dynamics qualitatively consistent with the butterfly effect. However, at small spatial scales the hybrid NeuralGCM is a notable exception. Its DKE spectrum displays a nearly flat, white-noise-like structure at initial time steps, suggesting that the injection of noise at the encoder level produced perturbations inconsistent with balanced atmospheric dynamics. There is also an unnatural transition at the interface between the machine-learned and resolved scales. Indeed, the KE spectrum (Fig. 4) indicates that NeuralGCM-ENS underestimates the magnitude of small scale KE, likely due to the machinelearned parametrization of sub-grid processes, the noise injection, which generates a flat KE spectrum at very small scales, and the cut off in the spectral loss, while producing a realistic KE spectrum at the resolved scales.

Considering the transport of KE across scales, it is the NeuralGCM-ENS hybrid model that reproduces the expected upscale transfer of KE most closely, showing a reduction in KE at high wavenumbers over time that is similar to that seen in the IFS-ENS model, but not with expected slope in spectral space (Fig. 3). This suggests that incorporating a physics-based dynamical core improves the physical consistency of multiscale energy interactions but predominantly at the resolved scales. While the purely data-driven models FCN3, AIFS-ENS and GenCast produce DKE spectra with more realistic DKE at high wavenumbers (small scales), which is likely due to FCN3's spectral loss function and GenCast's diffusion-based generative architecture, they nevertheless all three fail to produce the expected upscale transfer of KE at small scale. In particular, AIFS-ENS and GenCast accumulate KE at high wavenumbers, which may be due to their use of spatially uncorrelated noise.

Overall, our results suggest that although MLWP models achieve competitive scores in metrics such as RMSE, these metrics do not fully capture basic KE behaviour expected from observations. The DKE spectrum appears realistic in data-driven models but the KE transport across scales is not, with the hybrid model being closest to a realistic crossscale KE behvior. These findings raise broader questions: To what extent and how much physical consistency should machine learning weather forecasting models adhere to, acknowledging that these are not built as general purpose fluid dynamics solvers but only for the purpose of weather forecasting? To what extent does an inaccurate representation of known energy transfer across spatial scales undermine the trustworthiness of these models, given that their forecasts may nevertheless be judged to be of high quality according to other metrics?

## Open Research Section

Data for the IFS-ENS forecasts were obtained from the TIGGE archive (Bougeault et al., 2010). GenCast ensemble predictions are available via the WeatherNext dataset provided by Google Earth Engine (DeepMind Technologies Limited, 2024). Forecasts were generated using the official code and model checkpoints for AIFS-ENS v1.0 (https:// huggingface.co/ecmwf/aifs-ens-1.0), FourCastNet 3 (https://huggingface.co/ nvidia/fourcastnet3), and NeuralGCM-ENS (https://github.com/neuralgcm/neuralgcm).

## Conflict of Interest declaration

The authors declare there are no conflicts of interest for this manuscript.

## Acknowledgments

JO was supported by the ETH AI Center through an ETH AI Center postdoctoral fellowship. SD would like to acknowledge support from both Schmidt Sciences, LLC, and the Institute of Computing for Climate Science, University of Cambridge. JC performed most of the analysis during a master thesis project at the University of Cambridge supervised by SS and SD.

## References

Adamov, S., et al. (2025). Building machine learning limited area models: Kilometer-scale weather forecasting in realistic settings. arXiv preprint arXiv:2504.09340. https://doi.org/10.48550/arXiv.2504.09340.

Bauer, P., Thorpe, A., & Brunet, G. (2015). The quiet revolution of numerical weather prediction. Nature, 525(7567), 47-55.

Bauer, P., et al. (2020). The ECMWF Scalability Programme: Progress and Plans. European Centre for Medium-Range Weather Forecasts.

Baumgart, M., et al. (2019). Quantitative view on the processes governing the upscale error growth up to the planetary scale using a stochastic convection scheme. Monthly Weather Review, 147(5), 1713-1731.

Ben Bouallègue, Z., et al. (2024). The rise of data-driven weather forecasting: A first statistical assessment of machine learning-based weather forecasts in an operational-like context. Bulletin of the American Meteorological Society, 105, E864-E883.

Bi, K., et al. (2023). Accurate medium-range global weather forecasting with 3D neural networks. Nature, 619(7970), 533-538.

Bodnar, C., et al. (2025). A foundation model for the Earth system. Nature, 641 (8065), 1180-1187.

Bonavita, M. (2024). On some limitations of current machine learning weather prediction models. Geophysical Research Letters, 51 (12), e2023GL107377.

Bonev, B., et al. (2025). Fourcastnet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale. arXiv preprint arXiv:2507.12144.

Bougeault, P., et al. (2010). The THORPEX Interactive Grand Global Ensemble. Bulletin of the American Meteorological Society, 91 (8), 1059-1072.

Burgess, B. H., Erler, A. R., & Shepherd, T. G. (2013). The troposphere-tostratosphere transition in kinetic energy spectra and nonlinear spectral fluxes as seen in ECMWF analyses. Journal of the Atmospheric Sciences, 70(2), 669-687.

Bülte, C., et al. (2026). Uncertainty quantification for data-driven weather models. Artificial Intelligence for the Earth Systems, 5(1), 240049.

Chan, J. C. L. (1985). Identification of the steering flow for tropical cyclone motion from objectively analyzed wind fields. Monthly Weather Review, 113(1), 106-116.

Charney, J. G. (1971). Geostrophic turbulence. Journal of the Atmospheric Sciences, 28(6), 1087-1095.

Chen, S., et al. (2006). Physical mechanism of the two-dimensional inverse energy cascade. Physical Review Letters, 96(8), 084502.

DeepMind Technologies Limited. (2024). WeatherNext Gen Forecasts. Google Earth Engine Dataset. https://developers.google.com/earth-engine/datasets/ catalog/projects-gcp-public-data-weathernext\_assets\_126478713\_1\_0

Ding, T., et al. (2023). Impact of convection-permitting and model resolution on the simulation of mesoscale convective system properties over East Asia. Journal of Geophysical Research: Atmospheres, 128(24), e2023JD039395.

ECMWF. (2019). IFS Documentation CY46R1 - Part V: Ensemble Prediction System.

George, J. E., & Gray, W. M. (1976). Tropical cyclone motion and surrounding parameter relationships. Journal of Applied Meteorology, 15(12), 1252-1264.

Gkioulekas, E., & Tung, K. K. (2007). A new proof on net upscale energy cascade in two-dimensional and quasi-geostrophic turbulence. Journal of Fluid Mechanics, 576, 173-189.

Han, T., et al. (2024). FengWu-GHR: Learning the kilometer-scale medium-range global weather forecasting. arXiv preprint arXiv:2402.00059.

Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999–2049.

Jakob, A. R., Hack, A. J., & Williamson, A. D. (1993). Solutions to the Shallow Water Test Set Using the Spectral Transform Method. University Corporation for Atmospheric Research.

Judt, F. (2018). Insights into atmospheric predictability through global convectionpermitting model simulations. Journal of the Atmospheric Sciences, 75(5), 1477-1497.

Kalnay, E. (2012). Atmospheric modeling, data assimilation and predictability. Cambridge University Press, Cambridge.

Keisler, R. (2022). Forecasting global weather with graph neural networks. arXiv preprint arXiv:2202.07575.

Kim, H., et al. (2026). A spectral test of the butterfly effect and physical consistency in the diffusion-based GenCast's ensembles. npj Climate and Atmospheric Science.

Kochkov, D., et al. (2024). Neural general circulation models for weather and climate. Nature, 632(8027), 1060-1066.

Kolmogorov, A. N. (1995). Turbulence: the legacy of A.N. Kolmogorov. Cambridge University Press.

Kraichnan, R. H. (1967). Inertial ranges in two-dimensional turbulence (No. RR11).

Lam, R., et al. (2023). Learning skillful medium-range global weather forecasting. Science, 382(6677), 1416-1421.

Lang, S., Hólm, E., Bonavita, M., & Tremolet, Y. (2019). A 50-member Ensemble of Data Assimilations. ECMWF Newsletter, 158, 27–29.

Lang, S., et al. (2026). AIFS-CRPS: Ensemble forecasting using a model trained with a loss function based on the continuous ranked probability score. npj Artificial Intelligence, 2(1).

Leung, T. Y., et al. (2020). Impact of the mesoscale range on error growth and the limits to atmospheric predictability. Journal of the Atmospheric Sciences, 77(11), 3769-3779.

Li, Z., et al. (2025). Exploring the differences in atmospheric mesoscale kinetic energy spectra between AI based and physics based models. Scientific Reports, 15(1), 15504.

Lorenz, E. N. (1963). Deterministic nonperiodic flow. Journal of the Atmospheric Sciences, 20(2), 130-141.

Lorenz, E. N. (1969). The predictability of a flow which possesses many scales of motion. Tellus, 21 (3), 289-307.

Lynch, P. (2008). The origins of computer weather prediction and climate modeling. Journal of Computational Physics, 227, 3431–3444.

Maltrud, M. E., & Vallis, G. K. (1991). Energy spectra and coherent structures in forced two-dimensional and beta-plane turbulence. Journal of Fluid Mechanics, 228, 321-342.

Métais, O., et al. (1996). Inverse cascade in stably stratified rotating turbulence. Dynamics of Atmospheres and Oceans, 23(1-4), 193-203.

Owens, R., & Hewson, T. (2018). ECMWF Forecast User Guide. ECMWF. URL https://www.ecmwf.int/422 node/16559.

Palmer, T. (2019). The ECMWF ensemble prediction system: Looking back (more than) 25 years and projecting forward 25 years. Quarterly Journal of the Royal Meteorological Society, 145, 12-24.

Pathak, J., et al. (2022). Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators. arXiv preprint arXiv:2202.11214.

Price, I., et al. (2023). GenCast: Diffusion-based ensemble forecasting for mediumrange weather. arXiv preprint arXiv:2312.15796.

Rasp, S., et al. (2024). WeatherBench 2: A benchmark for the next generation of data-driven global weather models. Journal of Advances in Modeling Earth Systems, 16, e2023MS004019.

Ritchie, H., et al. (1995). Implementation of the semi-Lagrangian method in a highresolution version of the ECMWF forecast model. Monthly Weather Review, 123(2), 489-514.

Rotunno, R., & Snyder, C. (2008). A generalization of Lorenz's model for the predictability of flows with many scales of motion. Journal of the Atmospheric Sciences, 65(3), 1063-1076.

Selz, T., & Craig, G. C. (2015). Upscale error growth in a high-resolution simulation of a summertime weather event over Europe. Monthly Weather Review, 143(3), 813-827.

Selz, T., & Craig, G. C. (2023). Can artificial intelligence-based weather prediction models simulate the butterfly effect? Geophysical Research Letters, 50(20), e2023GL105747.

Selz, T., & Craig, G. C. (2026). Can AI-based weather prediction models simulate the butterfly effect? The role of architecture and implementation. Journal of Geophysical Research: Machine Learning and Computation, 3(3), e2025JH001180.

Selz, T., Riemer, M., & Craig, G. C. (2022). The transition from practical to intrinsic predictability of midlatitude weather. Journal of the Atmospheric Sciences, 79(8), 2013-2030.

Smith, L. M., & Yakhot, V. (1994). Finite-size effects in forced two-dimensional turbulence. Journal of Fluid Mechanics, 274, 115-138.

Smith, K. S., et al. (2002). Turbulent diffusion in the geostrophic inverse cascade. Journal of Fluid Mechanics, 469, 13-48.

Stull, R. B. (2012). An introduction to boundary layer meteorology. Springer Science & Business Media.

Sun, Y. Q., & Zhang, F. (2016). Intrinsic versus practical limits of atmospheric predictability and the significance of the butterfly effect. Journal of the Atmospheric Sciences, 73(3), 1419-1438.

Torn, R. D., et al. (2018). Tropical cyclone track sensitivity in deformation steering flow. Monthly Weather Review, 146(10), 3183-3201.

Uccellini, L. W., & Johnson, D. R. (1979). The coupling of upper and lower tropospheric jet streaks and implications for the development of severe convective storms. Monthly Weather Review, 107(6), 682-703.

Vallis, G. K. (2017). Atmospheric and oceanic fluid dynamics. Cambridge University Press.

Vallis, G. K. (2019). Essentials of atmospheric and oceanic dynamics. Cambridge University Press.

Weyn, J. A., Durran, D. R., & Caruana, R. (2019). Can machines learn to predict weather? Using deep learning to predict gridded 500-hPa geopotential height from historical weather data. Journal of Advances in Modeling Earth Systems,

11, 2680–2693.

Wieczorek, M. A., & Meschede, M. (2018). SHTools: Tools for working with spherical harmonics. Geochemistry, Geophysics, Geosystems, 19(8), 2574-2592.

Xia, H., et al. (2011). Upscale energy transfer in thick turbulent fluid layers. Nature Physics, 7(4), 321-32.

Xiao, Z., et al. (2009). Physical mechanism of the inverse energy cascade of twodimensional turbulence: a numerical investigation. Journal of Fluid Mechanics, 619, 1-44.

Yamaguchi, H., et al. (2018). Introduction to JMA's new global ensemble prediction system. CAS/JSC WGNE, Research Activities in Atmospheric and Oceanic Modelling, 42, 6-13.

Zhang, F., Bei, N., Rotunno, R., Snyder, C., & Epifanio, C. C. (2007). Mesoscale predictability of moist baroclinic waves: Convection-permitting experiments and multistage error growth dynamics. Journal of the Atmospheric Sciences 64 (10), 3579–3594. https://doi.org/10.1175/JAS4028.1.