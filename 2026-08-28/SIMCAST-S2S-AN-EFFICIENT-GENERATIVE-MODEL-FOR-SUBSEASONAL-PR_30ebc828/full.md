# SIMCAST-S2S: AN EFFICIENT GENERATIVE MODEL FOR SUBSEASONAL PRECIPITATION FORECASTING VIA TRANSFER LEARNING FROM CLIMATE SIMULATIONS

Hiep V. Dang Department of Environmental Sciences, University of Virginia, Charlottesville, VA 22903, United States zgp2ps@virginia.edu

Antonios Mamalakis   
School of Data Science, Department of Environmental Sciences,   
University of Virginia, Charlottesville, VA 22903, United States npa4tg@virginia.edu

## ABSTRACT

Subseasonal-to-seasonal (S2S) precipitation forecasting has substantial financial and societal impact, yet remains challenging because of weak predictive signals, high associated uncertainty, and the computational cost of operational systems, which constrains simulation fidelity. We introduce SimCast-S2S, a generative latent-diffusion framework for probabilistic S2S precipitation forecasting that addresses three major bottlenecks in data-driven prediction. First, because S2S prediction requires uncertainty quantification rather than only deterministic point forecasts, SimCast-S2S is the first data-driven system that uses a diffusion-based generative pipeline for S2S prediction, enabling effective sampling from the underlying conditional distribution. Second, since generating large probabilistic ensembles is computationally costly in physical space, SimCast-S2S instead operates in a compact latent space learned by variational autoencoders, enabling efficient large-ensemble generation. Third, diffusion models typically require large training datasets; SimCast-S2S overcomes this via transfer learning with low-rank adaptation (LoRA), pretraining on large ensembles of climate simulations before fine-tuning on limited reanalysis data. On reanalysis data, SimCast-S2S outperforms deep learning baselines, including convolutional neural networks and U-Net architectures. Notably, despite using only a subset of atmospheric input variables and no post-processing, bias correction, or calibration, SimCast-S2S remains competitive with, and in many cases outperforms, state-of-the-art operational systems such as the ECMWF-S2S baseline. These results indicate that latent generative modeling combined with simulation-to-reanalysis transfer learning offers an efficient and scalable path toward data-driven probabilistic S2S precipitation forecasting.

## 1 Introduction

Subseasonal to seasonal (S2S) forecasting remains among the most important challenges in climate science, as it refers to the transition zone between weather and climate timescales [1, 2, 3, 4, 5]. At short (weather) lead times, forecasts depend mainly on the initial conditions of the atmosphere, and weather prediction models can track the evolution of storms, fronts, and circulation patterns with relatively high accuracy before errors in initial conditions accumulate significantly [6, 7, 8]. At seasonal and longer lead times, the focus shifts away from individual weather events and toward slowly varying boundary conditions enforced by climate variability in sea surface temperature, soil moisture, snow cover, and other land–ocean interactions that provide sources of predictability [9, 10, 11, 12]. S2S forecasting, usually about 3 to 6 weeks ahead, lies between these two regimes [1, 2, 3, 4, 5], where small errors in the initial atmospheric conditions have grown substantially, while at the same time, slower sources of variability do not yet yield stable predictive signals. Skill therefore depends on whether models can capture a mixture of fine-scale variability in atmospheric dynamics and slower Earth-system processes [13, 14, 10, 11, 12, 15].

Recent progress in physical modeling and computation has made it possible to produce S2S forecasts operationally on a global scale [2, 16]. A central development in modern S2S forecasting is the use of ensemble prediction systems (EPSs) [17, 18, 19, 20, 8, 21, 22, 23, 2]. In chaotic atmospheric systems, a single deterministic forecast is usually insufficient [6, 24, 8]. Thus, an EPS generates multiple forecast members by perturbing initial conditions and model physics [17, 18, 19, 20, 8, 23]. This approach is especially important at S2S lead times because forecast errors often grow exponentially as the atmosphere evolves before nonlinear saturation occurs [25, 6, 26, 8], and an EPS provides a probabilistic description of possible future states. Operational centers, such as the European Centre for Medium-Range Weather Forecasts (ECMWF), now issue S2S ensemble forecasts out to 46 days, with products often summarized as weekly anomalies relative to model climatology. The value of EPS forecasting is particularly clear for precipitation, which is one of the most difficult variables to model and predict. Precipitation is strongly affected by nonlinear processes, including convection, moisture transport, land–atmosphere feedbacks, and tropical variability [4]. It is widely observed that small errors in circulation, moisture transport, and humidity fields usually lead to large errors in the location and intensity of precipitation [27, 28, 29]. Forecasting centers partially address this problem with post-processing, calibration, multi-model combination, and statistical correction [30], but they still face major limitations regarding systematic model biases and high computational costs [7, 31, 32].

Statistical and machine-learning (ML) models have become an increasingly important alternative, and in many cases a complementary approach, to conventional physics-based forecasting systems [33, 34, 35, 36]. These data-driven models can exploit large archives of observations, reanalysis products, and dynamical model output to learn nonlinear relationships between atmospheric predictors and future target states. Instead of numerically solving the governing equations at every forecast step, ML models are trained to approximate the forecast mapping directly from data. This allows fast and highly parallelizable predictions during inference. Recent studies have shown that such approaches can improve S2S forecasting when they are trained on large reanalysis datasets and multi-model forecast archives [34, 36]. Furthermore, as will be shown later, ML models pretrained on large amounts of simulated climate data can be transferred to real-world forecasting tasks through fine-tuning on observations or reanalysis products; see also [37, 38, 39]. This pretraining strategy can help the model learn general atmospheric structures before adapting to the observed climate system, and it may improve forecast skill compared to models trained from scratch on limited observational data.

Traditional EPS outputs are very computationally expensive because they require repeated model integrations with perturbed initial conditions and model physics [8, 19]. A state-of-the-art subcategory of ML methods, Generative AI, provides a framework for probabilistic S2S forecasting. Generative models aim to approximate the entire conditional distribution of future atmospheric states conditioned on relevant large-scale current climate states, allowing multiple plausible forecast samples to be drawn efficiently after training. This makes them especially useful for variables such as precipitation, where forecast uncertainty is strongly shaped by nonlinear dynamics. Recent diffusion-based forecasting systems have shown that generative models can produce realistic probabilistic forecasts and emulate large ensemble distributions at substantially lower computational cost than traditional EPSs [40, 41].

Despite their promise, generative models also have important limitations. As noted earlier, purely datadriven models are data-thirsty, requiring large training datasets to infer atmospheric dynamics [33, 42], and such datasets are often unavailable or limited, particularly beyond weather timescales. In addition, not all generative frameworks are equally stable. Generative adversarial networks (GANs), for example, have been widely used for realistic sample generation [43], but their adversarial training procedure is known to suffer from instability, mode collapse, and sensitivity to optimization settings [44, 45]. Encoder–decoder probabilistic frameworks, including models related to variational autoencoders [46], provide a more stable alternative and have been used in recent machine-learning S2S forecasting systems. For example, FuXi-S2S combines an encoder–decoder architecture with a perturbation module in the learned latent space, which has been shown to outperform ECMWF-S2S for selected variables, including precipitation and outgoing longwave radiation [36], although the authors did not assess how well the model captured uncertainty. However, such models still require substantial input variables, which makes training expensive and limits their accessibility.

To address the above challenges, namely, i) the need for large forecast ensembles to characterize S2S uncertainty, ii) the large training datasets required by deep generative models, particularly under weak predictive signals and training stability issues, and iii) the high computational cost of ensemble generation, we propose SimCast-S2S, an efficient generative model based on the diffusion framework [47, 48]. SimCast-S2S uses multiple domain-specific variational autoencoders (VAEs) to compress high-dimensional physical input fields into low-dimensional Gaussian latent representations [46]. A compact diffusion network is then trained in this latent space to map standard Gaussian noise, conditioned on past and current atmospheric states, to the future target latent representation. In the final step, a separate decoder maps the generated latent representation back to the physical space to produce the final forecast (see Methods). This design follows the general idea of latent generative modeling, where operating in a compressed representation can substantially reduce computational cost compared to generating directly in the original high-dimensional space [49].

This computational advantage is substantial in practice: on a standard A100 GPU, generating one ensemble member takes approximately 12 seconds, and different ensemble members can be produced in parallel across multiple GPUs. As a result, forecasting speed scales almost linearly with the number of GPUs and generating a e.g., 100-member forecast ensemble to better characterize forecast uncertainty becomes inexpensive compared to repeatedly running a full physics-based numerical model. Most importantly, this high degree of efficiency also makes it feasible to pre-train SimCast-S2S on a large ensemble of climate simulation output, which, we will show, addresses the need for large training datasets and substantially improves accuracy. Specifically, we train the model on 28 ensemble members of the Community Earth System Model version 2 Large Ensemble (CESM2-LE), each initialized from perturbed initial conditions [50]. By pre-training SimCast-S2S on multiple simulation-based realizations, the model can learn a broader mapping between large-scale climate fields and future S2S precipitation variability rather than constraining training on a single and limited reanalysis dataset. The learned representation is then used for real-world forecasting by fine-tuning the model on ECMWF Reanalysis v5 (ERA5), a global atmospheric reanalysis widely used as a reference dataset for weather and climate applications [51]. Our results indicate that SimCast-S2S outperforms other machine-learning baselines and challenges the state-of-the-art ECMWF-S2S physics-based forecast system on the ERA5 reanalysis dataset. The improvements of SimCast-S2S are shown in terms of both deterministic forecast accuracy and probabilistic uncertainty quantification.

![](images/4bebdf943811c22460aca0c5293045de3a8fc111ecd4dab23791f6623262ed31.jpg)  
Figure 1: An ensemble of precipitation-anomaly forecasts generated by SimCast-S2S for 2-week period from February 12, 2021 to February 25, 2021 (ERA5 dataset), conditioned on the atmospheric inputs from January 01, 2021 to January 28, 2021. The ensemble mean and the corresponding ERA5 reanalysis are also provided. All fields are expressed in units of standard deviation. This example shows stronger ensemble agreement in the tropics, especially along the equatorial Pacific, and larger member-to-member variability over the extratropics. However, coherent extratropical signals remain visible over the North Pacific, the North Atlantic, and much of the Southern Hemisphere storm-track region. These areas of high consensus are also consistent with the ERA5 target.

## 2 Results

SimCast-S2S is first pretrained on 28 ensemble members from CESM2-LE using five atmospheric variables across three pressure levels 200 hPa, 500 hPa, and 850 hPa, along with five single-level variables (all considered during the last 28 days) to forecast precipitation during the 3rd and 4th week into the future. The model is trained from 1950 to 2000, validated from 2001 to 2010, and tested in years from 2011 to 2014. After pretraining, SimCast-S2S is transferred to ERA5 dataset where it is fine-tuned in the years from 1940 to 2010, validated from 2011 to 2020, and tested from 2021 to 2025 (see Section 4 for more details).

SimCast-S2S is computationally efficient, making it feasible to generate large forecast ensembles. A 100- member SimCast-S2S ensemble reproduces many of the large-scale precipitation anomaly structures while still retaining realistic member-to-member variability (see Fig 1). This section provides a comprehensive quantitative evaluation of the quality of the forecasts using all the test samples. Subsection 2.1 compares the deterministic skill of SimCast-S2S with other ML baselines (e.g., convolutional neural networks, UNets etc.) and with the operational ECMWF-S2S system. Subsection 2.2 evaluates probabilistic forecast skill across the 2021–2025 ERA5 test samples. Subsection 2.4 assesses the spatial realism of the generated precipitation fields. Subsection 2.3 assesses the uncertainty reproduced by SimCast-S2S and ECMWF-S2S, and evaluates how well the forecast ranges cover the true events. Finally, subsection 2.5 discusses the computational efficiency and scalability of SimCast-S2S under different hardware configurations.

Table 1: Deterministic forecast skill measured by the mean absolute error (MAE), expressed in units of standard deviation and scaled by $1 0 ^ { - 2 }$ . We report the average MAE across all test samples and the inter-sample standard deviation. The first result column evaluates models on held-out CESM2 test data, and the remaining three result columns evaluate models on ERA5 test data under different training settings: CESM2-only training, ERA5-only training, and CESM2 pretraining followed by ERA5 fine-tuning. SimCast-S2S and ECMWF-S2S are evaluated using the ensemble mean of 100 generated members. Bold values indicate the best performance on ERA5 among different models and settings.
<table><tr><td></td><td colspan="2">Trained on CESM2</td><td>Trained on ERA5</td><td>Pretrained on CESM2 &amp; Finetuned on ERA5</td></tr><tr><td>Model / Tested on</td><td>CESM2</td><td>ERA5</td><td>ERA5</td><td>ERA5</td></tr><tr><td colspan="5">Proposed generative model</td></tr><tr><td>SimCast-S2S  $( \eta = 0 . 0 )$ </td><td> $1 . 2 5 \pm 0 . 0 6$ </td><td> $1 . 3 2 \pm 0 . 0 6$ </td><td> $1 . 3 7 \pm 0 . 0 6$ </td><td> ${ \bf 1 . 3 0 \pm 0 . 0 6 }$ </td></tr><tr><td>SimCast-S2S  $( \eta = 0 . 5 )$ </td><td> $1 . 2 5 \pm 0 . 0 6$ </td><td> $1 . 3 2 \pm 0 . 0 6$ </td><td> $1 . 3 7 \pm 0 . 0 6$ </td><td> ${ \bf 1 . 3 0 \pm 0 . 0 6 }$ </td></tr><tr><td>SimCast-S2S  $( \eta = 1 . 0 )$ </td><td> $1 . 2 5 \pm 0 . 0 6$ </td><td> $1 . 3 2 \pm 0 . 0 7$ </td><td> $1 . 3 8 \pm 0 . 0 7$ </td><td> ${ \bf 1 . 3 0 \pm 0 . 0 6 }$ </td></tr><tr><td colspan="5">Neural-network baselines</td></tr><tr><td>CNN-Small</td><td> $1 . 4 3 \pm 0 . 0 8$ </td><td> $1 . 4 2 \pm 0 . 0 6$ </td><td> $1 . 3 4 \pm 0 . 0 7$ </td><td> $1 . 3 8 \pm 0 . 0 6$ </td></tr><tr><td>CNN-Medium</td><td> $1 . 3 1 \pm 0 . 0 7$ </td><td> $1 . 3 2 \pm 0 . 0 6$ </td><td> $1 . 4 6 \pm 0 . 0 6$ </td><td> $1 . 3 3 \pm 0 . 0 6$ </td></tr><tr><td>CNN-Large</td><td> $1 . 5 1 \pm 0 . 0 9$ </td><td> $1 . 4 7 \pm 0 . 0 8$ </td><td> $1 . 4 7 \pm 0 . 0 7$ </td><td> $1 . 5 0 \pm 0 . 0 8$ </td></tr><tr><td>UNet-Small</td><td> $1 . 3 5 \pm 0 . 0 8$ </td><td> $1 . 3 5 \pm 0 . 0 6$ </td><td> $1 . 3 6 \pm 0 . 0 7$ </td><td> $1 . 3 5 \pm 0 . 0 7$ </td></tr><tr><td>UNet-Medium</td><td> $1 . 3 2 \pm 0 . 0 8$ </td><td> $1 . 3 2 \pm 0 . 0 6$ </td><td> $1 . 3 6 \pm 0 . 0 8$ </td><td> $1 . 3 3 \pm 0 . 0 6$ </td></tr><tr><td>UNet-Large</td><td> $1 . 3 1 \pm 0 . 0 7$ </td><td> $1 . 3 2 \pm 0 . 0 6$ </td><td> $1 . 3 5 \pm 0 . 0 7$ </td><td> $1 . 3 3 \pm 0 . 0 6$ </td></tr><tr><td colspan="5">Operational baseline</td></tr><tr><td>ECMWF-S2S</td><td></td><td></td><td></td><td> $1 . 3 2 \pm 0 . 0 6$ </td></tr></table>

## 2.1 Deterministic Skill Evaluation

Although the atmosphere is highly chaotic and a deterministic forecast cannot represent the full range of possible future states, deterministic skill remains an essential first-order measure of forecast quality, as it assesses the conditional mean of the underlying distribution of future states. In an ensemble forecasting system, the ensemble mean summarizes the predictable component shared across members by averaging out member-specific variability. If the individual ensemble members are centered around physically meaningful future states, the ensemble mean should provide an accurate estimate of the expected precipitation anomalous field. Evaluating deterministic skill tests whether SimCast-S2S captures the dominant predictable signal and is also necessary for fair comparison with popular Deep Learning (DL) baselines, which often produce deterministic predictions. We have evaluated the deterministic skill of SimCast-S2S on the ERA5 test samples using its ensemble-mean prediction, and compared it against the ensemble-mean prediction of ECMWF-S2S and deterministic predictions from different variants of Convolutional Neural Networks (CNNs) [52] and UNets [53]. Each CNN and UNet variant corresponds to a specific model size—small, medium, or large—determined by the number of layers and hidden features. Architectural details are provided in Section 4.

On the CESM2 test set, SimCast-S2S achieves the lowest mean absolute error (MAE) among all evaluated models and training settings (see the first column in Table 1). All three sampling settings obtain (1.25 ±

![](images/0eb9770cd450afe9add933560ad9706f9fcd75f110860d5c0ee2e29b4e99b2ca.jpg)  
Figure 2: Validation anomaly correlation coefficient (ACC) for SimCast-S2S under different sampling parameters η and for ECMWF-S2S. ACC is calculated using the ensemble mean forecast. Dark bars show unweighted ACC, and light bars show latitude-weighted ACC. Red outlines indicate cases where SimCast-S2S has significantly higher ACC than ECMWF-S2S under a one-sided bootstrap test at the 2.5% significance level.

$0 . 0 6 ) \times 1 0 ^ { - 2 }$ std, while the best CNN and UNet baselines reach only $( 1 . 3 1 \pm 0 . 0 7 ) \times 1 0 ^ { - 2 }$ std. The baseline results also show that increasing model size does not guarantee better performance. CNN-Large performs worse than CNN-Medium, indicating possible overfitting. Similarly, UNet-Large has substantially more parameters than UNet-Medium, but it produces nearly the same error, which suggests that the UNet architecture gains little from further scaling. SimCast-S2S also exhibits competitive simulation-to-reanalysis transfer without ERA5 fine-tuning. When trained only on CESM2 and evaluated directly on ERA5, SimCast S2S achieves an MAE of $1 . 3 2 \times 1 0 ^ { - 2 } \ \mathrm { s t d }$ , lower than most CNN and UNet baselines and comparable to the best-performing UNet variants (see the second column in Table 1). This result indicates that CESM2 training alone learns precipitation features that partially transfer to ERA5. However, the remaining error also shows a clear simulation-to-reanalysis gap. This gap (although marginal) may be partially attributable to imperfect/biased representations of the S2S relevant processes in CESM2 simulations that have led to a slightly compromised learning that is revealed when models are tested on real-world data. Regarding different sizes of the baseline DL models, results mirror the behavior of when tested on the CESM2 test set. SimCast-S2S performs worst when trained only on ERA5 (see the third column in 1), with MAE increasing to $1 . 3 7 \mathrm { - 1 . 3 8 \times 1 0 ^ { - 2 } }$ std, a large degradation especially when compared to $1 . 3 2 \times 1 0 ^ { - 2 }$ std in CESM2 training. This result indicates that the diffusion-based SimCast-S2S requires more training data, and training on ERA5 alone was not sufficient. Large simulated ensembles are therefore important for training the generative mapping before adapting the model to reanalysis data [54, 55, 56, 57, 58]. Pretraining on CESM2 followed by ERA5 fine-tuning yields the best performance on the ERA5 test set (see the last column in Table 1). Under this setting, all three SimCast-S2S variants achieve $( 1 . 3 0 \pm 0 . 0 6 ) \times 1 0 ^ { - 2 }$ std, which is the lowest MAE on ERA5 in the table. SimCast-S2S outperforms all CNN and UNet baselines and also improves over the operational ECMWF-S2S benchmark, which obtains $( 1 . 3 2 \pm 0 . 0 6 ) \times 1 0 ^ { - 2 }$ std on the same ERA5 test samples. It is clear that CESM2 pretraining exposes the model to a larger set of physically plausible atmospheric evolutions and ERA5 fine-tuning adapts this learned representation to the observation-constrained atmosphere. Transfer learning is the key strategy that allows SimCast-S2S to overcome a central limitation of generative models, namely their dependence on large training datasets. The simulation-to-reanalysis transfer strategy, rather than the architecture alone, enables SimCast-S2S to outperform the operational baseline. More details on the transfer learning method are discussed in Section 4.4.

![](images/ea0567695a6ea9722c5bfc4ad455d3c508ff48cd156459e1cc0a6ec672e0c61a.jpg)  
Figure 3: Probabilistic forecast skill of ECMWF-S2S and SimCast-S2S for precipitation anomalies. Top row: Spatial distribution of tercile-based Ranked Probability Skill Score (RPSS). Bottom row: Continuous Ranked Probability Skill Score (CRPSS). For each metric, the left and middle columns show the skill scores of ECMWF-S2S and SimCast-S2S, respectively, while the right column shows the difference between SimCast-S2S and ECMWF-S2S. Positive values in the left and middle columns indicate skill above the climatological baseline, whereas negative values indicate performance below it. Positive values in the right column indicate higher skill for SimCast-S2S relative to ECMWF-S2S. Dots indicate regions where the corresponding positive skill score, or positive difference in skill score, is statistically significant at the 2.5% level based on 1,000 bootstrap resamples.

The deterministic results also show that the choice of sampling parameter $\eta ,$ which controls the trade-off between diversity and stability (see Subsection 4.3 in Methods), has little effect on the ensemble-mean MAE. Across all training settings, the three SimCast-S2S variants produce nearly identical deterministic errors. This is somewhat expected because MAE is evaluated using the ensemble mean, which averages out member-specific sampling variability. However, different η values result in different anomaly correlation coefficients (ACC). SimCast-S2S achieves substantially higher mean ACC than ECMWF-S2S for $\eta = 0$ and $\eta = 0 . 5$ , whereas its performance for $\eta = 1 . 0$ is closer to that of ECMWF-S2S over the 2021–2025 validation period (see Fig. 2). Given the similar MAE across sampling settings, the higher ACC for $\eta = 0$ and $\eta = 0 . 5$ indicates better spatial correspondence between the predicted and the true precipitation patterns rather than simply smaller point-wise error magnitudes. As described in Section 4, η controls the sampling behavior of SimCast-S2S and is not relevant to training. We therefore treat η as an untrainable hyperparameter and select it using validation ACC reported in Fig 2. Among the three experimented values, $\eta = 0$ achieves the highest ACC. We use $\eta = 0$ as the default sampling setting for SimCast-S2S, and all remaining analyses in this section report SimCast-S2S results under this choice.

## 2.2 Probabilistic Skill Evaluation

Deterministic skill evaluation is limiting for subseasonal forecasts because small perturbations in the initial state can lead to substantially different precipitation outcomes; thus, a point forecast can at best approximate the expected value of the conditional distribution and may differ substantially from any individual realization because of the system’s inherent internal variability. Since SimCast-S2S is inherently stochastic, it produces a distribution of possible forecasts rather than a point estimate. We next evaluate how well the forecast distribution produced by SimCast-S2S covers the observed ERA5 target.

![](images/6217e4c2bed618b7959371dd1d901730a9176d1ae5556b31faa28d0cc3349cfb.jpg)  
Figure 4: Seasonal CRPSS for SimCast-S2S and ECMWF-S2S precipitation forecasts during the 2021–2025 test period over the global, tropical, and extratropical domains. Seasons are defined as winter (December– February), spring (March–May), summer (June–August), and fall (September–November). Positive CRPSS indicates better probabilistic skill than the climatological baseline. Higher values indicate better performance.

Probabilistic forecast skill is strongest in the tropics for both ECMWF-S2S and SimCast-S2S, particularly along the equatorial Pacific, as measured by the ranked probability skill score (RPSS) for tercile forecasts (see Fig. 3). SimCast-S2S not only exhibits stronger tropical skill pattern but also shows more widespread increased skill relative to ECMWF-S2S across many extratropical regions, especially in the North Pacific, North Asia, North America, the North Atlantic, and the Southern Ocean. This advantage is more clearly visible in the difference map, where positive values indicate higher skill for SimCast-S2S (see red shading). The continuous ranked probability skill score (CRPSS) in Fig. 3, which evaluates the full predictive distribution rather than discrete tercile categories, shows a broadly consistent spatial pattern. SimCast-S2S exhibits higher CRPSS than ECMWF-S2S across most of the globe, indicating that its advantage extends beyond tercile-category prediction to the continuous forecast distribution. It should also be noted that some persistently dry regions, such as the Sahara, the southeastern Pacific off South America, and Antarctica, have precipitation values that are zero or near zero for much of the time, with low temporal variability. In these regions, the verifying precipitation anomalies are small, and both models’ forecasts remain close to the reference climatology.

The spatial patterns in Fig. 3 show where probabilistic skill is concentrated. Fig. 4 examines which season this skill occurs. Spring and summer appear to be the most difficult seasons to predict subseasonal precipitation variability. ECMWF-S2S shows negative CRPSS over the extratropics in these seasons; however, SimCast-S2S still manages to maintain positive CRPSS. In fact, SimCast-S2S shows higher CRPSS than ECMWF-S2S across all seasons and regions, with the largest differences occurring in the tropics. This agrees with the spatial maps in Fig. 3, where the strongest probabilistic skill is concentrated along the tropical regions. In the extratropics, both models exhibit weaker skill.

## 2.3 Uncertainty Quantification

Beyond aggregate probabilistic skill, it is important to examine how well each model represents forecast uncertainty. A useful ensemble forecast should not only produce accurate probabilistic predictions but also assign uncertainty in a way that is statistically consistent with the reanalysis data. In this section, we evaluate uncertainty quantification from two complementary perspectives. First, we examine the percentile histogram to assess the calibration of the ensemble distribution across global, tropical, and extratropical regions. Second, we analyze the relationship between ensemble spread and forecast error, together with interval scores for different nominal coverage levels.

![](images/a66d0303165962777691641addac164dfb95f7fcaa164e57bbc4e5face5c3522.jpg)  
Figure 5: Probability integral transform (PIT) histograms for SimCast-S2S and ECMWF-S2S precipitation forecasts. The red dashed line indicates the uniform distribution expected under perfect calibration. Both models show U-shaped histograms, indicating underdispersive ensemble forecasts. SimCast-S2S shows nearly constant $\chi ^ { 2 }$ distance across regions, whereas ECMWF-S2S shows better extratropical calibration but larger tropical deviation from uniform behavior.

A well-calibrated probabilistic forecast should produce an approximately uniform probability integral transform (PIT) distribution, indicating that the verifying observed/reanalysis estimates are evenly distributed across the forecast distribution. Both SimCast-S2S and ECMWF-S2S exhibit U-shaped PIT histograms, with elevated frequencies near the lowest and highest percentiles across the global, tropical, and extratropical regions (see Fig. 5). This pattern indicates underdispersion, meaning that the forecast distributions are narrow and assign insufficient probability to outcomes in the tails. In order words, both SimCast-S2S and ECMWF-S2S tend to underestimate forecast uncertainty.

The degree of underdispersion varies by region and model. Globally, ECMWF-S2S shows a smaller $\chi ^ { 2 }$ distance from uniform behavior than SimCast-S2S. This global difference is mainly driven by the extratropics, where ECMWF-S2S has a lower $\chi ^ { 2 }$ distance. In the tropics, however, ECMWF-S2S has a higher $\chi ^ { 2 }$ distance than SimCast-S2S. That being said, this comparison should be interpreted carefully. SimCast-S2S is evaluated directly from its raw ensemble output, without additional statistical calibration or post-processing. ECMWF-

![](images/4bec24099336afcdb5a9456e2b6b500900320c3d0a12623610d12af1b35ffbc7.jpg)

Figure 6: Spread–error and interval-score diagnostics for SimCast-S2S and ECMWF-S2S precipitation forecasts over global, tropical, and extratropical domains. The top row compares mean ensemble spread with the RMSE of the ensemble mean. Both models are above the ideal one-to-one dashed line, indicating forecast errors are larger than ensemble spread. Spread–error correlations are high. This suggests that larger spread is still associated with larger error and the ensemble spread truly reflects useful uncertainty information. The bottom row shows mean interval scores for different central interval coverages, where lower values indicate better probabilistic forecasts. SimCast-S2S consistently exhibits better score, particularly in tropical regions.

S2S, in contrast, is accompanied by post-processed forecasts that are commonly used to adjust systematic forecast biases and construct calibrated anomaly or probabilistic products. Therefore, the ECMWF-S2S ensemble evaluated here may not represent an equally raw forecast product in the same sense as in SimCast-S2S. The relatively stable $\chi ^ { 2 }$ distance across regions for SimCast-S2S suggests more spatially uniform ensemble dispersion, whereas the larger tropical–extratropical contrast for ECMWF-S2S indicates stronger regional dependence signaling regional overfitting in the post-processing calibration.

Another key diagnostic of ensemble reliability is the relationship between ensemble spread and forecast error. In a well-scaled ensemble, these two quantities should be close to the one-to-one line, so that larger forecast uncertainty corresponds to larger realized error. Both SimCast-S2S and ECMWF-S2S lie slightly above this line in all three regions of focus, indicating that the realized errors are larger than the ensemble spread (see the top row of Fig. 6). Although high spread–error correlations are observed in both models, the main difference is that SimCast-S2S lies closer to the one-to-one line in most cases over the tropics, and hence provides a better-scaled estimate of forecast uncertainty than ECMWF-S2S in these regions. Over the extratropics, SimCast-S2S slightly underestimates the uncertainty compared to ECMWF-S2S.

The ensemble prediction intervals are further evaluated using the mean interval score (see the bottom row of Fig. 6). The x-axis shows the nominal central interval coverage. For example, a 90% central interval spans the 5th to 95th percentiles of the ensemble distribution, corresponding to the central 90% of the forecast distribution around the median. The y-axis shows the mean interval score. This score penalizes two types of behavior: intervals that are unnecessarily wide and intervals that fail to contain the ground truth. Therefore, a lower interval score indicates both better sharpness and coverage. A formal description of mean interval score can be found in Section . As expected, the interval score increases as the nominal coverage becomes larger because higher coverage requires wider prediction intervals. Across the global, tropical, and extratropical regions, SimCast-S2S shows slightly lower error than ECMWF-S2S, particularly at the highest coverage levels over the tropical regions. This is consistent with the PIT histogram in Fig. 5.

![](images/cfe47dad9ce742cea848a61886b433a0b2f14e9c3ce8c6be87cca8a7e00c692b.jpg)  
Figure 7: Spatial autocorrelation of ERA5, SimCast-S2S, and ECMWF-S2S precipitation fields along zonal (west–east), meridional (south–north), main-diagonal (northwest–southeast), and anti-diagonal (southwest– northeast) directions. The x-axis shows spatial lag, and the y-axis shows autocorrelation. Smaller differences from the ERA5 distributions indicate better agreement with the ground truth regarding spatial structure of precipitation. SimCast-S2S better follows the true meridional and diagonal autocorrelation, especially at smaller spatial lags where fine-scale precipitation structure remains meaningful and most impactful.

## 2.4 Spatial Structure Realism

We next examine whether the forecasted precipitation fields have realistic spatial structure. This is important because a model can achieve good aggregate scores while still producing fields that are too smooth, too noisy, or spatially inconsistent with true precipitation. Spatial autocorrelation is evaluated in four directions: zonal, meridional, main diagonal, and anti-diagonal (see Fig. 7). The meridional direction is especially important because it provides a strong test of whether the models capture realistic transitions across latitude-dependent precipitation regimes.

Across all four directions, the ERA5 fields show decreasing autocorrelation with increasing spatial lag, and both SimCast-S2S and ECMWF-S2S are able to reproduce this basic decay. In the zonal direction, SimCast-S2S and ECMWF-S2S show nearly equal performance. SimCast-S2S and ECMWF-S2S both track the observed decay reasonably well although both models show slightly stronger west-east persistence than the reanalysis data. The meridional direction provides a more demanding test because precipitation regimes vary strongly with latitude, spanning e.g., deep convective precipitation tied to the intertropical convergence zone in the tropics, suppressed precipitation beneath the subsiding branch of the Hadley circulation in the subtropics, frontal and baroclinic precipitation along midlatitude storm tracks, and weaker orographically-driven precipitation in the high-latitudes. In this direction, SimCast-S2S follows the observed decay more closely than ECMWF-S2S across most lags whereas ECMWF-S2S unfavorably persists more spatial structures across latitude bands.

The main diagonal and anti-diagonal directions follow the northwest–southeast and southwest–northeast directions, respectively. In both directions, SimCast-S2S is closer to the true autocorrelation than ECMWF-S2S at smaller spatial lags, where local precipitation structure is still meaningfully present. As the lag increases, the autocorrelation quickly collapses to zero. This is the distance range where precipitation at one location has little statistical connection to precipitation farther away. Consequently, the advantage of SimCast-S2S diminishes (although still present), not because the models become equally realistic, but because there is little remaining spatial dependence for either model to reproduce.

## 2.5 Computational Efficiency and Scalability

A practical advantage of SimCast-S2S is that probabilistic forecasts can be generated rapidly once the model is deployed. Conventional subseasonal systems such as ECMWF-S2S generate probabilistic forecasts by repeatedly integrating a full numerical model forward in time for many ensemble members. SimCast-S2S instead produces ensemble members through stochastic sampling during the denoising process of the diffusion model (see Subsection 4.3), so each member can be generated independently and parallelized efficiently across GPUs.

SimCast-S2S can generate large precipitation ensembles efficiently at 192 × 288 resolution. On a single A100 GPU, it takes approximately 12.3 seconds per ensemble member, or 123 seconds for 10 members, 246 seconds for 20 members, 615 seconds for 50 members, and 1230 seconds, or 20.5 minutes, for 100 members (see Fig. 8). With three A100 GPUs, the same 100-member ensemble requires 438.21 seconds, or about 7.3 minutes. With four A100 GPUs, it decreases further to 342.19 seconds, or about 5.7 minutes. Thus, a 100-member ensemble can be generated in less than ten minutes using a small number of industry-grade A100 GPUs.

It should be noted that the scaling behavior is close to linear because ensemble generation is highly parallelizable. Moving from one to two A100 GPUs reduces the 100-member runtime from 1230.0 seconds to 634.19 seconds, corresponding to a parallel efficiency of $\begin{array} { r } { \frac { 1 2 3 0 } { 2 \times 6 3 4 . 1 9 } = 0 . 9 6 9 7 } \end{array}$ . With three GPUs, the efficiency is 0.9356, and with four GPUs it is 0.8986. The efficiency decreases gradually as more GPUs are added, reaching 0.7392 at eight GPUs (see panel (b) of Fig. 8). This departure from ideal linear scaling is small and expected [59, 60]. It reflects fixed overheads from model loading, data movement, GPU synchronization, and inter-device communication. These overheads are more visible when the ensemble size is small, but they are increasingly amortized as the number of ensemble members grows [61]. SimCast-S2S also benefits directly from newer accelerator hardware. For a 100-member ensemble on one GPU, the wall-clock time is 2213 seconds on V100, 1477 seconds on A6000, 1231 seconds on A100, and 683 seconds on H200 (see panel (a) of Fig. 8). With eight H200 GPUs, the 100-member runtime decreases to only 115.55 seconds, or less than two minutes. This is a strong indication that SimCast-S2S can exploit both per-GPU improvement and multi-GPU parallelism.

![](images/a8cfb4a61d48c27a441af659c5b4a1f857562ea3ab58f7db6ed85404fbbf1839.jpg)  
Figure 8: Computational efficiency and scalability of SimCast-S2S ensemble generation. Panel (a) shows wall-clock inference time as a function of number of GPUs, GPU types and ensemble sizes. Panel (b) compares measured A100 runtimes with ideal linear scaling; closer agreement with the dashed lines indicates better multi-GPU scalability.

To our knowledge, ECMWF does not publicly disclose the wall-clock integration time. Products correspond ing to the 00:00 UTC forecast base time are scheduled to become available at 20:00 UTC [62]. However, this 20-hour interval likely encompasses an end-to-end operational workflow, including initialization, numerical integration, post-processing, and dissemination, not just model runtime. An earlier ECMWF benchmark nevertheless provides some indication of the computational expense. Its previous 51-member, 15-day ensemble required an average wall-clock time of 82 minutes on 1,530 compute nodes [63]. Although this historical benchmark is not directly comparable with the runtime of SimCast-S2S, it illustrates the large-scale computational resources required by conventional dynamical ensemble forecasting. In contrast, generating a 100-member probabilistic ensemble with SimCast-S2S is just a matter of minutes on a single GPU node. This efficiency makes large-ensemble S2S prediction widely accessible to typical research groups and also allows uncertainty estimates and extreme-event probabilities to be produced with much lower latency.

## 3 Discussion

We tackle three key bottlenecks that have limited data-driven models for S2S forecasting. First, generative models are notoriously data-hungry; we address this through transfer learning via LoRA, pretraining on large ensembles of climate simulations before fine-tuning on limited reanalysis data. Second, generating large probabilistic ensembles at high resolution is computationally prohibitive; we address this by operating in a compact latent space rather than physical space, which makes large-ensemble generation efficient. Third, S2S forecasting fundamentally requires representing forecast uncertainty, not just a single deterministic outcome; we address this by using a generative (diffusion-based) framework that directly models the forecast distribution rather than a point estimate. We show that each of these design choices works: our model outperforms all deterministic AI baselines, and using only a subset of input variables and without any post-processing or bias correction, it challenges a leading operational forecasting system.

A central result of this study is that the higher probabilistic skill of SimCast-S2S also translates into stronger deterministic performance. SimCast-S2S exhibits higher probabilistic skill across much of the globe as shown in Figs. 3 and 4, consistent with the lower deterministic errors reported in Table 1. This result suggests that the generated forecast members are not merely adding stochastic variability around a weak deterministic forecast; rather, they contain a coherent predictive signal that survives ensemble averaging and is closer to the true conditional distribution.

Our uncertainty diagnostics in Subsection 2.3 indicate that both SimCast-S2S and ECMWF-S2S remain underdispersive, i.e., their ensemble spread is narrow relative to realized true variability. However, the degree and regional structure of this underdispersion differ. ECMWF-S2S shows stronger uncertainty behavior in the extratropics, while SimCast-S2S is more competitive in the tropics and it overall lies closer to the spread–RMSE 1:1 reference. This comparison is demanding for SimCast-S2S because ECMWF-S2S uses reforecasts to estimate model-specific systematic errors; bias correction then reduces persistent offsets, and probabilistic calibration adjusts ensemble probabilities toward observed event frequencies. In contrast, SimCast-S2S is evaluated directly from its raw ensemble output without comparable post-processing or calibration. Therefore, its competitive tropical performance and improved spread–error scaling suggest that the generative ensemble already contains meaningful uncertainty information before any statistical correction is applied.

Importantly, we show that SimCast-S2S better captures the spatial decay of precipitation dependence, especially in the meridional and diagonal directions. This is notable because precipitation regimes vary strongly with latitude, from tropical convection to subtropical suppression, midlatitude storm tracks and high-latitude stratiform precipitation. It is well known that deep convolutional forecast models trained with point-wise loss functions favor overly smooth fields and suppress fine-scale precipitation structure [64, 65, 66]. In contrast, SimCast-S2S learns the spatial organization of precipitation across dynamically distinct regimes. This is one of the strongest arguments for using a diffusion-based generative model in this setting. The model is trained to sample from a learned distribution of atmospheric states, and therefore can preserve spatial dependence in ways that cannot be done by point-wise loss functions alone.

SimCast-S2S also changes the practical cost structure of subseasonal ensemble prediction. Conventional dynamical systems generate ensemble forecasts by repeatedly integrating a full numerical model forward in time. SimCast-S2S directly models the one-to-one mapping, from the current atmospheric state to the target weeks. This is done through neural-network stochastic sampling, where individual members can be generated independently and parallelized efficiently across accelerators. As shown in Subsection 2.5, a 100-member ensemble can be generated in minutes on a small GPU node. The same ensemble size would require thousands of CPU nodes and several hours of wall-clock time in a conventional dynamical forecasting system [63]. This efficiency opens opportunities for research settings where operational-scale computing is unavailable, and for applications that require many ensemble members, such as uncertainty quantification, tail-risk assessment, scenario generation, and impact modeling [67, 68, 69].

A limitation is that the realism of SimCast-S2S is currently statistical rather than explicitly physical. The model can learn realistic distributions, spatial structures, and predictor–target relationships, but it is not guaranteed to satisfy conservation laws or dynamical balances. This limitation is closely tied to the same design choice that makes the model efficient: latent-space sampling. Working in a low-dimensional latent representation largely reduces computational cost, but physical laws such as mass conservation, moisture conservation, and momentum balance only make sense in physical space. There is no obvious way to impose these laws directly on abstract latent variables. As a result, SimCast-S2S may generate fields that are statistically plausible but not strictly dynamically consistent. A natural next step is therefore to combine latent-space generation with (occasional) physical-space correction. One possible direction is a hybrid sampling scheme in which the model generates most of the trajectory in latent space, periodically decodes the state into physical variables, evaluates physical residuals, applies corrections, and then maps the corrected state back into latent space. For precipitation forecasting, this could involve constraints derived from the moisture budget, physical bounds on humidity and precipitation, or consistency between circulation and moisture convergence. Such a scheme is feasible because neural networks are inherently differentiable, and physical residuals can be computed directly from their computation graph [70, 71, 72, 73, 74]. Incorporating physical constraints into the generative model may also help improve representation of extremes, which remain a key weakness of both SimCast-S2S and ECMWF-S2S.

Another important direction is interpretability. SimCast-S2S uses many atmospheric predictors, but the present analysis does not fully explain which fields, regions, or dynamical patterns the model relies on when generating forecasts. Explainable AI methods could help determine whether the model uses physically meaningful sources of predictability, such as tropical moisture anomalies, upper-level circulation, midlatitude wave activity, atmospheric rivers, or teleconnection patterns [75, 76, 77, 78, 79, 80, 81, 82]. Methods such as input perturbation, integrated gradients, latent-space sensitivity analysis, and counterfactual sampling could be adapted to our generative setting [83, 84, 85]. This would be especially valuable because generative models can be right for the wrong reasons, and such failures are difficult to diagnose from forecast scores alone. Therefore, connecting SimCast-S2S skill to interpretable physical mechanisms becomes crucial for making the model more scientifically useful and for identifying regimes in which it is likely to fail.

Our results demonstrate the potential of latent, diffusion-based, generative modeling combined with simulation-to-reanalysis transfer learning as an efficient and scalable framework for a new generation of probabilistic S2S precipitation forecasts. Further advances in physical consistency and interpretability could strengthen their reliability and scientific utility.

## 4 Methods

## 4.1 Data

SimCast-S2S is designed to maximize computational efficiency so that it can be trained on large ensembles of climate-model simulations and then transferred to observation-constrained reanalysis data. We first pretrain the model on 28 ensembles from the Community Earth System Model version 2 (CESM2), a fully coupled Earth system model that represents interactions among the atmosphere, ocean, land surface, sea ice, and other climate components [86, 50]. After pretraining on CESM2 simulations, SimCast-S2S is fine-tuned and evaluated on the fifth-generation European Centre for Medium-Range Weather Forecasts atmospheric reanalysis, ERA5 [51]. ERA5 combines a global numerical weather prediction model with a large collection of real-world observations through data assimilation, including information from satellites, radiosondes, aircraft, and surface stations. Therefore, ERA5 is not a strictly observational dataset, but rather an observation-constrained reconstruction of the atmosphere. As a result, it provides one of the closest available approximations to the observed atmospheric state and can be used to transfer the model from idealized climate-model simulations to realistic atmospheric conditions.

To make transfer learning between CESM2 and ERA5 physically meaningful, the input variables must be defined consistently across the two datasets. We therefore select variables that are available, or have close physical counterparts, in both CESM2 and ERA5. The selected variables are also chosen to describe the atmospheric column across three representative pressure levels: 850 hPa for the lower troposphere, 500 hPa for the middle troposphere, and 200 hPa for the upper troposphere. In addition, single-level variables are included to describe surface and column-integrated conditions that are directly linked to precipitation, including surface temperature, sea-level pressure, outgoing longwave radiation, total precipitable water, and precipitation itself.

The list of all CESM2 variables used for pretraining and their ERA5 counterparts used for fine-tuning are provided in Table 2. At each pressure level, we include wind, geopotential, temperature, and humidity. These variables jointly describe the dynamical and thermodynamical state of the atmosphere: winds capture horizontal and vertical transport, geopotential represents large-scale circulation structure, temperature describes thermal stratification, and humidity provides the moisture supply needed for precipitation. At the surface or column-integrated level, the selected variables provide additional constraints on boundary-layer conditions, radiative state, pressure patterns, and atmospheric moisture content.

Table 2: CESM2 input variables used for SimCast-S2S pretraining and their corresponding ERA5 variables used for fine-tuning.
<table><tr><td>Pressure level</td><td>Variable type</td><td>CESM2 variable</td><td>ERA5 variable</td></tr><tr><td rowspan="5">200 hPa</td><td>Zonal wind</td><td>U200</td><td>u_200</td></tr><tr><td>Meridional wind</td><td>V200</td><td>v_200</td></tr><tr><td>Temperature</td><td>T200</td><td>t_200</td></tr><tr><td>Specific humidity</td><td>Q200</td><td>q_200</td></tr><tr><td>Geopotential height</td><td>Z200</td><td>z_200</td></tr><tr><td rowspan="6">500 hPa</td><td>Zonal wind</td><td>U500</td><td>u_500</td></tr><tr><td>Meridional wind</td><td>V500</td><td>v_500</td></tr><tr><td>Vertical velocity</td><td>OMEGA500</td><td>w_500</td></tr><tr><td>Temperature</td><td>T500</td><td>t_500</td></tr><tr><td>Specific humidity</td><td>Q500</td><td>q_500</td></tr><tr><td>Geopotential height</td><td>Z500</td><td>z_500</td></tr><tr><td rowspan="5">850 hPa</td><td>Zonal wind</td><td>U850</td><td>u_850</td></tr><tr><td>Meridional wind</td><td>V850</td><td>v_850</td></tr><tr><td>Temperature</td><td>T850</td><td>t_850</td></tr><tr><td>Specific humidity</td><td>Q850</td><td>q_850</td></tr><tr><td>Geopotential height</td><td>Z850</td><td>z_850</td></tr><tr><td rowspan="5">Single level</td><td>Surface temperature</td><td>TS</td><td>t2m</td></tr><tr><td>Sea-level pressure</td><td>PSL</td><td>msl</td></tr><tr><td>Precipitation</td><td>PRECT</td><td>tp</td></tr><tr><td>Top longwave flux</td><td>FLUT</td><td>avg_tnlwrf</td></tr><tr><td>Total precipitable water</td><td>TMQ</td><td>tcwv</td></tr></table>

Some variables are physically related but not identical across the two datasets, so simple conversions are required. For example, ERA5 reports average top net longwave radiation flux (avg\_tnlwrf), following the ECMWF sign convention in which downward flux is positive. This has the opposite sign convention from the CESM2 upwelling longwave flux at the top of the model (FLUT), so the ERA5 field is multiplied by −1. Similarly, CESM2 reports geopotential height (Z), whereas ERA5 reports geopotential (z). To make the variables equivalent, ERA5 geopotential is divided by gravitational acceleration, $g \approx 9 . 8 0 6 6 5 \mathrm { ~ m ~ s ^ { - 2 } }$ to obtain geopotential height. Precipitation also requires unit adjustment. ERA5 reports accumulated precipitation over an hourly interval (tp), whereas CESM2 reports total precipitation as a per-second rate (PRECT). Therefore, ERA5 precipitation is divided by 3, 600. These conversions are made to ensure that SimCast-S2S is pretrained and fine-tuned on variables with comparable units.

All input and target variables were transformed into standardized anomalies before model training. From a meteorological standpoint, this step separates subseasonal variability from the much stronger background structure of the climate system, including the seasonal cycle, spatially varying climatology, and slow long-term trends. Raw atmospheric fields are dominated by predictable differences between seasons and regions; however, subseasonal forecasting is primarily concerned with departures from those expected states. Removing the local trend and day-of-year climatology therefore encourages the model to learn physically meaningful anomaly relationships rather than simply reproducing the mean annual cycle or the background climate state. The same transformation was applied to both input variables and the precipitation target to preserve a consistent meteorological reference frame. In this form, the model learns how anomalous large-scale atmospheric conditions, such as circulation, temperature, and moisture anomalies, are associated with anomalous precipitation. Specifically, for each variable, a grid-point-wise linear trend was first estimated from the daily fields and removed to reduce slow nonstationary changes unrelated to subseasonal variability. A smoothed day-of-year climatology was then obtained. After that, each field was standardized by subtracting the corresponding climatological mean and dividing by the climatological standard deviation. For validation and testing, the detrending coefficients and climatological statistics estimated from the training period were reused to avoid information leakage.

![](images/dd4d7f8583d7eb26e9a90fc440a17491d99d80c1199c4030484e3acd8ca1f2dc.jpg)  
Figure 9: Schematic illustration of the VAE mapping between the high-dimensional physical space and the low-dimensional latent space for variable group $g .$ . The encoder $\mathcal { E } _ { g }$ maps a anomaly field $\mathbf { x } _ { g }$ to a Gaussian latent distribution $\mathbf z _ { g } \sim \mathcal N ( \mu _ { g } , \sigma _ { g } ^ { 2 } )$ . Sampling from this distribution introduces stochasticity in latent space, and the decoder $\mathcal { D } _ { g }$ maps the sampled latent representation back to a reconstructed physical field $\hat { \mathbf { x } } _ { g }$ . The latent space has a much lower dimension than the physical space, and stochastic sampling introduces a small perturbation between $\hat { \mathbf { x } } _ { g }$ and $\mathbf { x } _ { g }$

## 4.2 Physical and Latent Space

SimCast-S2S represents the atmospheric state in two coupled spaces: the gridded physical space and a learned probabilistic latent space. In physical space, each sample is a collection of standardized anomaly fields defined on a latitude–longitude grid, with separate variable groups describing circulation, mass, thermodynamic, moisture, and precipitation processes. Although this representation preserves direct meteorological interpretability, it is high-dimensional and highly redundant because large-scale atmospheric fields exhibit strong spatial covariance and coherent dynamical structures. Directly modeling the conditional distribution of future precipitation in this space would therefore require learning stochastic relationships across a large number of mutually dependent grid points.

To reduce dimensionality, SimCast-S2S uses variational autoencoders (VAEs) [87] to map each physical variable group from its gridded physical space to a compact probabilistic latent space. For variable group g, let $\mathbf { x } _ { g } \in \mathsf { \overline { { X } } } _ { g } \subset \mathbb { R } ^ { C _ { g } \times H \times W }$ denote a physical-space field group, where $C _ { g }$ is the number of variables in the group and $\mathbf { \bar { \boldsymbol { H } } } \times \mathbf { \boldsymbol { W } }$ is the latitude–longitude grid. The corresponding latent representation is defined in a lower-dimensional space $\mathbf { z } _ { g } \in \mathcal { Z } _ { g } \subset \mathbb { R } ^ { d _ { g } }$ , where $d _ { g } \ll C _ { g } H W$ . Under a parameterization $\phi ,$ the encoder therefore learns a probabilistic mapping between the high-dimensional physical space $\chi _ { g }$ and the compact latent space $\mathcal { Z } _ { g } , \mathrm { i . e . } \ \mathcal { E } _ { \phi _ { g } } : \mathcal { X } _ { g } \to \mathcal { Z } _ { g }$

Although the encoder $\mathcal { E } _ { \phi _ { \underline { { c } } } }$ models a probabilistic latent representation, it is still a deterministic neural network. Given an input field group $\mathbf { x } _ { g }$ , the encoder first predicts the mean and standard deviation of the approximate posterior:

$$
\left( \mu _ { g } , \sigma _ { g } ^ { 2 } \right) = \mathcal { E } _ { \phi _ { g } } \left[ \mathbf { x } _ { g } \right] .\tag{1}
$$

These parameters define a Gaussian approximate posterior $q _ { \phi } \left( \mathbf { z } _ { g } \mid \mathbf { x } _ { g } \right) = \mathcal { N } \left( \pmb { \mu } _ { g } , \operatorname { d i a g } \left( \pmb { \sigma } _ { g } ^ { 2 } \right) \right)$ . The stochasticity is introduced by sampling $\mathbf { z } _ { g }$ from this approximate posterior. To allow gradients to pass through, the sampling is done using a continuous reparameterization trick:

$$
\begin{array} { r } { \mathbf { z } _ { g } = \pmb { \mu } _ { g } + \pmb { \sigma } _ { g } \odot \pmb { \epsilon } , } \end{array}\tag{2}
$$

where $\odot$ is the element-wise multiplication, and $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$

The decoder $\mathcal { D } _ { \theta _ { g } } : \mathcal { Z } _ { g }  \mathcal { X } _ { g }$ , parameterized by $\theta ,$ maps the sampled latent vector back to physical space:

$$
\hat { \mathbf { x } } _ { g } = { \mathcal { D } } _ { \theta _ { g } } \left( \mathbf { z } _ { g } \right) .\tag{3}
$$

where $\hat { \mathbf { x } } _ { g } \in \mathcal { X } _ { g }$ is the reconstructed standardized anomaly field.

In probabilistic form, the decoder parameterizes the conditional reconstruction distribution $p _ { \theta } \left( \mathbf { x } _ { g } \mid \mathbf { z } _ { g } \right)$ which describes the distribution of physical-space fields that can be reconstructed from a given latent state $\mathbf { z } _ { g }$

There is always some discrepancy between $\hat { \mathbf { x } } _ { g }$ and $\mathbf { x } _ { g }$ for two reasons. First, the latent vector $\mathbf { z } _ { g }$ ∼ $q _ { \phi } \left( \mathbf { z } _ { g } \mid \mathbf { x } _ { g } \right)$ is a random variable, different samples from the same approximate posterior can lead to slightly different decoded fields. Second, some information is inevitably lost when a high-dimensional physical field is compressed into a much lower-dimensional latent space. As a result, the same anomaly field $\mathbf { x } _ { g } ,$ when passed through the encoding and decoding processes multiple times, can produce multiple possible reconstructions $\hat { \mathbf { x } } _ { g }$ . This introduces a natural source of perturbation in the reconstructed physical fields, which supports the stochastic nature of SimCast-S2S. Fig. 9 illustrates this encoding–decoding process.

The learning objective of a VAE is to maximize the marginal likelihood of the observed data $p _ { \phi , \theta } ( \mathbf { x } )$ Although this marginal likelihood is intractable, it can be showed that this objective can be obtained equivalently by minimizing its negative evidence lower bound (ELBO), which leads to the following loss function:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { V A E } } ( \pmb { \theta } , \phi , \mathbf { x } ) : = - \mathbb { E } _ { q _ { \phi } ( \mathbf { z } | \mathbf { x } ) } [ \log p _ { \pmb { \theta } } ( \mathbf { x } \mid \mathbf { z } ) ] + D _ { \mathrm { K L } } [ q _ { \phi } ( \mathbf { z } \mid \mathbf { x } ) \parallel p ( \mathbf { z } ) ] . } \end{array}\tag{4}
$$

The first term in Eq. 4 is the negative log-likelihood of the observed dataset, which encourages the model to reconstruct the data accurately. The second term is the KL divergence between two distributions, which acts as a regularizer that encourages the approximate posterior $q _ { \phi } ( \mathbf { z } \mid \mathbf { x } )$ to remain close to the prior $p ( \mathbf { z } )$

In practice, it is common to further assume that the reconstruction error follows a Gaussian likelihood, i.e., $\mathbf { x } - \hat { \mathbf { x } } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ , where and $\sigma \in \mathbb { R }$ is a fixed constant. This is equivalent to modeling the likelihood as $p _ { \theta } ( \mathbf { x } \mid \mathbf { z } ) = \mathcal { N } ( \mathbf { x } ; \hat { \mathbf { x } } , \sigma ^ { 2 } \mathbf { I } )$ . Under this assumption, the negative log-likelihood simplifies to a scaled MSE loss, and minimizing the negative log-likelihood is equivalent to minimizing the MSE between the reconstruction and the input, up to a constant scaling factor.

$$
- \log p _ { \theta } ( \mathbf { x } \mid \mathbf { z } ) = \frac { 1 } { 2 \sigma ^ { 2 } } \| \mathbf { x } - \hat { \mathbf { x } } \| ^ { 2 } + \mathrm { c o n s t } .\tag{5}
$$

Also, with the prior $p ( \mathbf { z } )$ assumed to be a standard Gaussian, the KL divergence in Eq. 4 becomes a special case and can be written in a compact form:

$$
D _ { \mathrm { K L } } \left[ q _ { \phi } ( \mathbf { z } \mid \mathbf { x } ) \parallel p _ { \theta } ( \mathbf { z } ) \right] = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { d _ { g } } \left( \sigma _ { j } ^ { 2 } + \mu _ { j } ^ { 2 } - 1 - \log \sigma _ { j } ^ { 2 } \right) ,\tag{6}
$$

![](images/3b0e1d3b89ea594b08f3ce489c7b4a47edc4f0585ceae542a340555c5a3e5cd4.jpg)

Figure 10: VAE architecture used to map physical-space anomaly fields into a compact latent representation and reconstruct them back to physical space. The encoder E progressively compresses the input field x through ConvStack and DownSampling blocks to obtain the latent vector z. The decoder D mirrors this process using UpSampling and ConvStack blocks to reconstruct the field xˆ.  
Algorithm 1 VAE Training Procedure for variable group g   
Require: Dataset D, encoder $\mathcal { E } _ { \phi _ { g } }$ , decoder $\mathcal { D } _ { \theta _ { g } }$ , loss balancing term $\lambda \in \mathbb { R } ^ { + }$   
repeat   
Sample $\mathbf { x } \in \mathbb { D }$   
Encoder: compute $\left( \mu _ { g } , \sigma _ { g } ^ { 2 } \right) = \mathcal { E } _ { \phi _ { g } } \left[ \mathbf { x } _ { g } \right]$   
Sample $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$   
Compute $\mathbf { z } _ { g } = \pmb { \mu } _ { g } + \pmb { \sigma } _ { g } \odot \pmb { \epsilon }$   
Decoder: compute $\hat { \mathbf { x } } _ { g } = \mathcal { D } _ { \theta _ { g } } \left( \mathbf { z } _ { g } \right)$   
Take a gradient step on:   
$\begin{array} { r } { \nabla _ { \theta , \phi } \left[ \lambda \| \mathbf { x } - \hat { \mathbf { x } } \| ^ { 2 } + \frac { 1 } { 2 } \sum _ { j = 1 } ^ { d _ { g } } \left( \sigma _ { j } ^ { 2 } + \mu _ { j } ^ { 2 } - 1 - \log \sigma _ { j } ^ { 2 } \right) \right] } \end{array}$   
until converged

where $\mu _ { i }$ and $\sigma _ { i }$ are the components of $\mu _ { g }$ and $\sigma _ { g } ,$ respectively.

Combining Eq. 4, 6, 5, the training procedure of VAE is summarized in Algo. 1. The encoder parameters ϕ and decoder parameters θ are optimized jointly using the Adam optimizer [88]. Fig. 10 describe the structure of $\mathcal { E } _ { \phi _ { g } }$ and $\mathcal { D } _ { \theta _ { g } }$ , which together form the complete VAE architecture. Both the encoder and decoder use a standard convolutional architecture [89, 90]. The encoder is built from stacked convolutional layers, each followed by a downsampling block to progressively reduce the spatial resolution by a factor for four. The decoder mirrors this structure but uses upsampling blocks. Each upsampling block contains a transposed convolutional layer to expand the spatial resolution by a factor for four.

The number of ConvStack–DownScaling blocks directly controls the latent dimension. Let $C _ { g }$ denote the number of physical variables in group g, let $E _ { g }$ denote the number of latent channels per variable, and let $N _ { g }$ denote the number of ConvStack–DownScaling blocks, the latent dimension is $\begin{array} { r } { d _ { g } = \frac { C _ { g } E _ { g } H W } { 4 ^ { N _ { g } } } } \end{array}$ , where HW is the original physical-space grid. The corresponding compression ratio is $\frac { 4 ^ { N _ { g } } } { E _ { g } }$ . For example, for a group of $C _ { g } = 7$ variables, setting $N _ { g } = 3$ and $E _ { g } = 2 8$ gives a 16× compression. For a group of $C _ { g } = 1$ variable, setting $N _ { g } = 3$ and $E _ { g } = 1 6$ gives a 4× compression.

In this study, 21 variables listed in Table 2 are divided into five physically motivated groups, namely wind, mass, thermal, hydro, and precip. Table 3 details the variables included in each group for both CESM2 and ERA5. Each group has a physical-space dimension of $C _ { g } H W$ where $H W = 1 9 2 \times 2 8 8 = 5 5 { , } 2 9 6$

Table 3: Variable groups used in SimCast-S2S and their corresponding CESM2 and ERA5 variables. The 21 input variables are divided into five physically motivated groups: wind, mass, thermal, hydro, and precipitation. Each group has physical-space dimension $C _ { g } H W$ , latent-space dimension $d _ { g }$ , and compression ratio $C _ { g } H W / d _ { g }$ . Here, $H \times W = 1 9 2 \times 2 8 8$
<table><tr><td>Group</td><td>CESM2 variables</td><td>ERA5 variables</td><td> $C _ { g } H W$ </td><td> $d _ { g }$ </td><td>Compression</td></tr><tr><td>wind</td><td>U200, V200, U500, V500, OMEGA500, U850, V850</td><td>u200, v200, u500, v500, w500, u850, v850</td><td>387,072</td><td>24,192</td><td>16 times</td></tr><tr><td>mass</td><td>Z200, Z500, Z850, PSL</td><td>z200, z500, z850, msl</td><td>221,184</td><td>13,824</td><td>16 times</td></tr><tr><td>thermal</td><td>TS, T200, T500, T850, FLUT</td><td>skt, t200, t500, t850, avgtnlwrf</td><td>276,480</td><td>17,280</td><td>16 times</td></tr><tr><td>hydro</td><td>TMQ, Q200, Q500, Q850</td><td> $\mathrm { t c w v } , \mathrm { q } 2 0 0 , \mathrm { q } 5 0 0 , \mathrm { q } 8 5 0$ </td><td>221,184</td><td>13,824</td><td>16 times</td></tr><tr><td>precip</td><td>PRECT</td><td>tp</td><td>55,296</td><td>13,824</td><td>4 times</td></tr></table>

Choosing the compression level involves a trade-off between efficiency and reconstruction accuracy: a smaller latent dimension gives a more compact representation, while a larger latent dimension reduces reconstruction error. Empirically, we find reconstruction error decreases approximately exponentially as the latent dimension increases. Based on our analysis, we choose a 16× compression for the wind, mass, thermal, and hydro groups, and a 4× compression for precipitation. The lower compression ratio for precipitation is necessary because its distribution is highly intermittent, strongly skewed, and dominated by localized extremes. More aggressive compression would therefore oversmooth the latent distribution and degrade reconstruction fidelity.

## 4.3 SimCast-S2S model

SimCast-S2S is a latent diffusion model [91, 92, 93, 94, 95, 96]. The central idea is to model the stochastic evolution in the low-dimensional latent space. During the forward diffusion process, Gaussian noise is gradually added to the target latent vector until it is fully corrupted into a standard Gaussian. A neural network is then trained to reverse this process: starting from a noisy target latent state, it progressively removes noise by conditioning on the latent representations of the past atmospheric states. The objective is to recover the original target latent representation.

Consider a variance schedule $0 < \beta _ { 1 } , \beta _ { 2 } , \cdots , \beta _ { K } < 1$ . For a precipitation latent ${ \bf z } _ { 0 } \sim p ( { \bf z } )$ , a discrete Markov chain $\{ \mathbf { z } _ { 0 } , \mathbf { z } _ { 1 } , \cdots , \mathbf { z } _ { K } \}$ is defined by:

$$
\mathbf { z } _ { k } : = \sqrt { 1 - \beta _ { t } } \mathbf { z } _ { k - 1 } + \sqrt { \beta _ { k } } \varepsilon _ { k } , \quad \varepsilon _ { k } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) , \quad k : 1 \to K ,\tag{7}
$$

or equivalently:

$$
q ( \mathbf { z } _ { k } \mid \mathbf { z } _ { k - 1 } ) = { \mathcal { N } } \left( \mathbf { z } _ { k } ; { \sqrt { 1 - \beta _ { k } } } \mathbf { z } _ { k - 1 } , \beta _ { k } \mathbf { I } \right) .\tag{8}
$$

Eq. 7 defines a first-order Markov chain where the signal is gradually degraded by $\sqrt { 1 - \beta _ { k } }$ , and Gaussian noise with variance $\beta _ { k }$ is injected at each step. $\mathrm { A s } \ k \to K , \mathbf { z } _ { K } \sim { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } )$

Because each transition is Gaussian and linear, we can compute $q ( \mathbf { z } _ { k } \mid \mathbf { z } _ { 0 } )$ in closed form. Defining $\begin{array} { r } { \bar { \alpha } _ { k } : = \prod _ { i = 1 } ^ { k } ( 1 - \beta _ { j } ) } \end{array}$ where $\bar { \alpha } _ { 0 } = 1$ , which quantifies how much signal remains from $\mathbf { z } _ { 0 }$ at step k, Eq. 7 is now simplified as:

$$
{ \bf z } _ { k } = \sqrt { \bar { \alpha } _ { k } } { \bf z } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { k } } \epsilon ,\tag{9}
$$

in which $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ is a fresh standard Gaussian variable. This implies that $q ( \mathbf { z } _ { k } \mid \mathbf { z } _ { 0 } )$ is Gaussian with a closed form:

$$
q ( \mathbf { z } _ { k } \mid \mathbf { z } _ { 0 } ) = \mathcal { N } ( \mathbf { z } _ { k } ; \sqrt { \bar { \alpha } _ { k } } \mathbf { z } _ { 0 } , ( 1 - \bar { \alpha } _ { k } ) \mathbf { I } ) ,\tag{10}
$$

Eq. 10 is important because it enables direct sampling of $\mathbf { x } _ { k }$ from the clean data point $\mathbf { x } _ { \mathrm { 0 } }$ without iteratively going through all the diffusion steps.

We can train a neural network $f _ { \theta } ( \mathbf { Z } _ { \mathrm { c o n d } } , t , \mathbf { z } _ { k } , k )$ to approximate $\epsilon ,$ where $\mathbf { Z } _ { \mathrm { c o n d } }$ is the concatenated latent representation of the initial physical state and t is the day of year as detailed in section 4.2. The training objective reduces to a simple $\ell _ { 2 }$ loss function:

$$
\mathcal { L } _ { \mathrm { S i m C a s t - S 2 S } } ( \theta ) = \mathbb { E } _ { \mathbf { z } _ { 0 } , t } \left[ \| \epsilon - f _ { \theta } ( Z _ { \mathrm { c o n d } } , t , \mathbf { z } _ { k } , k ) \| _ { 2 } ^ { 2 } \right] ,\tag{11}
$$

Because the forward process $q ( \mathbf { z } _ { k } \mid \mathbf { z } _ { k - 1 } )$ is Markov and Gaussian, it can be shown that the reverse process $p ( \mathbf { z } _ { k - 1 } \mid \mathbf { z } _ { k } )$ is also Markov and Gaussian:

$$
\begin{array} { r } { p _ { \theta } ( \mathbf { z } _ { k - 1 } \mid \mathbf { z } _ { k } ) = \mathcal { N } \left( \mathbf { z } _ { k - 1 } ; \mu _ { \theta } ( Z _ { \mathrm { c o n d } } , t , \mathbf { z } _ { k } , k ) , \mathbf { \Sigma } _ { \mathrm { { L } } } \mathbf { I } \right) . } \end{array}\tag{12}
$$

The mean $\mu _ { \theta } ( Z _ { \mathrm { c o n d } } , t , \mathbf { z } _ { k } , k )$ is not a separate neural network by itself, but is analytically computed from the noise prediction $\hat { \epsilon } _ { k - 1 } = f _ { \theta } ( Z _ { \mathrm { c o n d } } , t , { \mathbf z } _ { k } , k )$ . In practice, the variance $\Sigma _ { k } = \mathrm { V a r } [ { \bf x } _ { k - 1 } \ | \ { \bf x } _ { k } ]$ is either fixed or learned. In this study, we make a common choice of setting $\begin{array} { r } { \Sigma _ { k } = \tilde { \beta } _ { k } = \frac { 1 - \bar { \alpha } _ { k - 1 } } { 1 - \bar { \alpha } _ { k } } \beta _ { k } } \end{array}$ . This choice ensures the reverse variance matches the true posterior variance $\tilde { \beta } _ { k } = \mathrm { V a r } [ { \bf z } _ { k - 1 } \mid { \bf z } _ { k } , { \bf z } _ { 0 } ]$

Recall that the generative process $p _ { \theta } ( \mathbf { z } _ { k - 1 } \mid \mathbf { z } _ { k } )$ is meant to approximate the true reverse process of the forward Markov chain. The ideal reverse transition is the posterior distribution $q ( \mathbf { z } _ { k - 1 } \mid \mathbf { z } _ { k } )$ , but this is intractable. However, we can exactly compute the posterior $q ( \mathbf { z } _ { k - 1 } \mid \mathbf { z } _ { k } , \mathbf { z } _ { 0 } )$ , which is the distribution over $\mathbf { z } _ { k - 1 }$ conditioned on both the noisy input $\mathbf { z } _ { k }$ and the original clean sample $\mathbf { z } _ { 0 }$ . Here, we use $q ( \mathbf { z } _ { k - 1 } \mid \mathbf { z } _ { k } , \mathbf { z } _ { 0 } )$ as a tractable surrogate for the intractable marginal posterior $q ( \mathbf { z } _ { k - 1 } \mid \mathbf { z } _ { k } )$ . Because affine transformations and marginalizations of Gaussians preserve Gaussianity, the triplet $( { \bf z } _ { 0 } , { \bf z } _ { k - 1 } , { \bf z } _ { k } )$ jointly forms a multivariate Gaussian distribution, and conditioning on any subset—including both $\mathbf { z } _ { 0 }$ and ${ \bf z } _ { k } - { \bf a } { \bf l } { \bf s } { \bf o }$ yields another Gaussian. This allows us to analytically compute the posterior $q ( \mathbf { z } _ { k - 1 } \mid \mathbf { z } _ { k } , \mathbf { z } _ { 0 } )$ in closed form using standard results from Gaussian conditioning:

$$
q ( \mathbf { z } _ { k - 1 } \mid \mathbf { z } _ { k } , \mathbf { z } _ { 0 } ) = \mathcal { N } \left( \mathbf { z } _ { k - 1 } ; \tilde { \pmb { \mu } } ( \mathbf { z } _ { k } , \mathbf { z } _ { 0 } ) , \boldsymbol { \Sigma } _ { k } \mathbf { I } \right) ,\tag{13}
$$

where

$$
{ \tilde { \mu } } ( \mathbf { z } _ { k } , \mathbf { z } _ { 0 } ) = \mathbb { E } [ \mathbf { z } _ { k - 1 } \mid \mathbf { z } _ { k } , \mathbf { z } _ { 0 } ] = { \frac { { \sqrt { { \bar { \alpha } } _ { k - 1 } } } \beta _ { k } } { 1 - { \bar { \alpha } } _ { k } } } \mathbf { z } _ { 0 } + { \frac { { \sqrt { { \boldsymbol { \alpha } } _ { k } } } { \bigl ( } 1 - { \bar { \alpha } } _ { k - 1 } { \bigr ) } } { 1 - { \bar { \alpha } } _ { k } } } \mathbf { z } _ { k } ,\tag{14}
$$

During the generative process, we do not have access to the true clean sample $\mathbf { z } _ { \mathrm { 0 } }$ . To approximate the reverse transition distribution $q ( \mathbf { z } _ { k - 1 } \mid \mathbf { z } _ { k } , \mathbf { z } _ { 0 } )$ , we leverage the reparameterized form of the forward process, which expresses $\mathbf { z } _ { k }$ as a linear combination of $\mathbf { z } _ { 0 }$ and Gaussian noise (see Eq. 9). Using the predicted noise $\hat { \mathbf { \epsilon } } = f _ { \theta } ( \mathbf { Z } _ { \mathrm { c o n d } } , t , \mathbf { z } _ { k } , k )$ , we can estimate $\mathbf { z } _ { 0 }$ as:

$$
\hat { \bf z } _ { 0 } : = \frac { 1 } { \sqrt { \bar { \alpha } _ { k } } } \left[ { \bf z } _ { k } - \sqrt { 1 - \bar { \alpha } _ { k } } \hat { \epsilon } \right] .\tag{15}
$$

![](images/fdc5b5db51a499f092c522d76abe59e604c41e32af303e42cb5103fd44168202.jpg)  
Figure 11: Schematic illustration of the SimCast-S2S latent diffusion framework. Historical atmospheric fields are first separated into five physical groups: wind, thermal, mass, hydro, and precipitation, as described in Table 3. Each group is encoded by its pretrained VAE encoder into a low-dimensional latent space. The resulting conditioning latents are concatenated to form $\mathbf { Z } _ { \mathrm { c o n d } }$ . Starting from a random Gaussian latent $\hat { \mathbf { z } } _ { K } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , the neural network iteratively denoises the latent state, conditioned on $\mathbf { Z } _ { \mathrm { c o n d } } .$ , the denoising step $k ,$ and seasonal information t, until it produces the target precipitation latent $\hat { \mathbf { z } } _ { 0 }$ . The pretrained precipitation decoder $\mathcal { D } _ { \mathrm { p r e c i p } }$ then maps $\hat { \mathbf { z } } _ { 0 }$ back to physical space to obtain the subseasonal precipitationanomaly forecast field $\hat { \mathbf { y } }$

This estimator effectively inverts the forward process at timestep k under the assumption that ϵˆ accurately predicts the noise component added to $\mathbf { z } _ { 0 }$ to obtain $\mathbf { z } _ { k }$ . We then substitute this estimate $\hat { \mathbf { z } } _ { 0 }$ into the analytic expression for the true posterior mean $\tilde { \pmb { \mu } } ( { \bf z } _ { k } , { \bf z } _ { 0 } )$ . With algebraic simplification, this expression becomes a compact and efficient form that is commonly used in practice:

$$
\tilde { \pmb { \mu } } ( { \bf z } _ { k } , \hat { \bf z } _ { 0 } ) = \frac { 1 } { \sqrt { 1 - \beta _ { k } } } \left( { \bf z } _ { k } - \frac { \beta _ { k } } { \sqrt { 1 - \bar { \alpha } _ { k } } } \hat { \bf \epsilon } \right) .\tag{16}
$$

With the mean in Eq. 13 now fully defined, the full stochastic reverse process at timestep k is obtained by sampling from a Gaussian centered at this predicted mean and with variance given by $\begin{array} { r } { \Sigma _ { k } = \check { \beta } _ { k } = \frac { 1 - \bar { \alpha } _ { k - 1 } } { 1 - \bar { \alpha } _ { k } } \beta _ { k } } \end{array}$ This yields the update rule used during sampling:

$$
\mathbf { z } _ { k - 1 } = \frac { 1 } { \sqrt { 1 - \beta _ { k } } } \left( \mathbf { z } _ { k } - \frac { \beta _ { k } } { \sqrt { 1 - \bar { \alpha } _ { k } } } \hat { \epsilon } \right) + \sqrt { \Sigma _ { k } } \epsilon , \quad \mathrm { w h e r e ~ } \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ,\tag{17}
$$

which defines one step of the generative chain, gradually transforming pure noise into a clean latent.

Eq. 17 is known as the Denoising Diffusion Probabilistic Models (DDPM) [93]. In this study, we also employ a method to further generalize this update rule by decomposing the stochasticity in Eq. 9 into a component along the data-aligned direction $\epsilon _ { \theta }$ , and a fresh isotropic noise $\boldsymbol { \zeta } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$

$$
\begin{array} { r } { \mathbf { z } _ { k - 1 } = \sqrt { \bar { \alpha } _ { k - 1 } } \hat { \mathbf { z } } _ { 0 } + c _ { k } \hat { \epsilon } + \sigma _ { k } \zeta , } \end{array}\tag{18}
$$

with the scalar coefficients $c _ { k } \geq 0$ and $\sigma _ { k } \ge 0$ to be determined. To ensure variance matching, the variance of $\mathbf { z } _ { k - 1 }$ conditioned on $\mathbf { z } _ { 0 }$ under Eq. 18 must satisfy the constraint:

$$
\mathrm { V a r } [ { \bf x } _ { k - 1 } \mid { \bf x } _ { 0 } ] = c _ { k } ^ { 2 } { \bf I } + \sigma _ { k } ^ { 2 } { \bf I } .\tag{19}
$$

or equivalently,

$$
c _ { k } ^ { 2 } \ + \ \sigma _ { k } ^ { 2 } = 1 - \bar { \alpha } _ { k - 1 }\tag{20}
$$

Among all pairs $\left( c _ { k } , \sigma _ { k } \right)$ that satisfy the constraint in Eq. 20, we introduce a parameter $\eta \in [ 0 , 1 ]$ to control the split between $c _ { k }$ and $\sigma _ { k }$ . This is done by defining

$$
\sigma _ { k } ^ { 2 } : = \eta ^ { 2 } \tilde { \beta } _ { k } ,\tag{21}
$$

The remaining variance is allocated to $c _ { k }$

$$
c _ { t } ^ { 2 } = 1 - \bar { \alpha } _ { t - 1 } - \sigma _ { t } ^ { 2 } .\tag{22}
$$

There are two important special cases. When $\eta = 1$ , we have $\sigma _ { k } ^ { 2 } = \tilde { \beta } _ { k }$ . It can be shown that this recovers the DDPM sampling rule described in Eq. 17. When η = 0 $\eta = 0$ , we have $\sigma _ { k } = 0$ and $c _ { k } = \sqrt { 1 - \bar { \alpha } _ { k - 1 } }$ . In this case, the reverse process becomes deterministic and corresponds to Denoising Diffusion Implicit Model (DDIM) sampling [97]. Eq. 18 then reduces to

$$
\mathbf { z } _ { k - 1 } = \sqrt { \bar { \alpha } _ { k - 1 } } \hat { \mathbf { z } } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { k - 1 } } \hat { \epsilon } .\tag{23}
$$

When $0 < \eta < 1$ , we obtain an interpolation between the deterministic path $( \eta = 0 )$ and the fully stochastic path $( \eta = 1 )$ :

$$
{ \bf z } _ { k - 1 } = \sqrt { \bar { \alpha } _ { k - 1 } } \hat { \bf z } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { k - 1 } - \sigma _ { k } ^ { 2 } } \underbrace { \frac { { \bf z } _ { k } - \sqrt { \bar { \alpha } _ { k } } \hat { \bf z } _ { 0 } } { \sqrt { 1 - \bar { \alpha } _ { k } } } } _ { \hat { \epsilon } } + \sigma _ { k } \zeta , \quad \zeta \sim \mathcal { N } ( { \bf 0 } , { \bf I } ) ,\tag{24}
$$

in which, $\sigma _ { t } ^ { 2 }$ takes the general form: $\begin{array} { r } { \sigma _ { t } ^ { 2 } = \eta ^ { 2 } \tilde { \beta } _ { t } = \eta ^ { 2 } \cdot \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \left( 1 - \frac { \bar { \alpha } _ { t } } { \bar { \alpha } _ { t - 1 } } \right) } \end{array}$

The interpolated form from Eq. 24 has its mean still depending on $\hat { \mathbf { z } } _ { 0 }$ and $\hat { \epsilon } ,$ but the variance injected is scaled down by $\overline { { \eta } } ^ { 2 }$ . Thus, η interpolates between two anchors: when $\eta = 0$ , the process is purely deterministic, when $\eta = 1$ , we recover the standard stochastic sampling; and for $0 < \eta < 1$ , the process offers a trade-off between diversity and stability.

So far, this formulation assumes that the neural network predicts the noise term ϵ directly. However, this is not the only possible parameterization of the denoising objective. Instead of predicting ϵ, SimCast-S2S is trained to predict a velocity variable v, which represents a rotated coordinate in the two-dimensional plane spanned by the clean latent $\mathbf { z } _ { 0 }$ and the noise ϵ. This v-prediction formulation provides an alternative way to describe the same diffusion trajectory. However, as will be shown later in this subsection, v-prediction yields a more stable training scheme than ϵ-prediction.

![](images/63fd933338b9ec74f70f9cba246b2eabc5102f58dcf6ae0c8846568203b9daac.jpg)  
Figure 12: Geometric illustration of the v-prediction formulation on the unit circle.

Recall that the forward process is defined as ${ \bf z } _ { k } = \sqrt { \bar { \alpha } _ { k } } { \bf z } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { k } } \epsilon ( \mathrm { E q . ~ } 9 )$ , where $\epsilon \sim \mathcal { N } ( 0 , I )$ is pure Gaussian and $\mathbf { z } _ { 0 }$ denotes the clean data sample. Because $\mathbf { z } _ { 0 }$ and ϵ are independent and orthogonal in expectation, i.e. $\mathbb { E } [ \mathbf { z } _ { 0 } ^ { \top } \epsilon ] = 0$ , their covariance vanishes:

$$
\operatorname { C o v } ( \mathbf { z } _ { 0 } , \pmb { \epsilon } ) = \mathbb { E } \big [ ( \mathbf { z } _ { 0 } - \mathbb { E } \mathbf { z } _ { 0 } ) ^ { \top } ( \pmb { \epsilon } - \mathbb { E } \pmb { \epsilon } ) \big ] = \mathbb { E } ( \mathbf { z } _ { 0 } - \mathbb { E } \mathbf { z } _ { 0 } ) ^ { \top } \mathbb { E } ( \pmb { \epsilon } - \mathbb { E } \pmb { \epsilon } ) = \mathbf { 0 } ,\tag{25}
$$

Because $\mathbf { z } _ { 0 }$ is a member in a Gaussian latent space, it follows that each $\mathbf { z } _ { k }$ is itself Gaussian for all k with mean 0 and constant unit variance I:

$$
\mathbb { E } [ \mathbf { z } _ { k } ] = \sqrt { \bar { \alpha } _ { k } } \mathbb { E } [ \mathbf { z } _ { 0 } ] + \sqrt { 1 - \bar { \alpha } _ { k } } \mathbb { E } [ \mathbf { \epsilon } ] = \sqrt { \bar { \alpha } _ { k } } \mathbf { 0 } + \sqrt { 1 - \bar { \alpha } _ { k } } \mathbf { 0 } = \mathbf { 0 } ,\tag{26}
$$

$$
\mathrm { V a r } ( \mathbf { z } _ { k } ) = \bar { \alpha } _ { k } \mathrm { V a r } ( \mathbf { z } _ { 0 } ) + \left( 1 - \bar { \alpha } _ { k } \right) \mathrm { V a r } ( \epsilon ) = \bar { \alpha } _ { k } \mathbf { I } + \left( 1 - \bar { \alpha } _ { k } \right) \mathbf { I } = \mathbf { I } .\tag{27}
$$

Hence, the diffusion process can be viewed as a rotation on the unit $( \mathbf { z } _ { 0 } , \epsilon )$ plane. By introducing an angular parameter $\omega _ { k }$ such that $\sqrt { \bar { \alpha } _ { k } } = \cos \omega _ { k }$ and ${ \sqrt { 1 - { \bar { \alpha } } _ { k } \ } } = \sin \omega _ { k }$ , the sample $\mathbf { z } _ { k }$ can be expressed as

$$
{ \bf z } _ { k } = \cos \omega _ { k } { \bf z } _ { 0 } + \sin \omega _ { k } \epsilon ,\tag{28}
$$

which parameterizes a circular trajectory from pure data $( \omega _ { 0 } = 0 )$ to pure noise $\begin{array} { r } { ( \omega _ { K } = \frac { \pi } { 2 } ) } \end{array}$ . Differentiating this expression with respect to $\omega _ { k }$ yields

$$
\frac { \partial \mathbf { z } _ { k } } { \partial \theta _ { k } } = - \sin \omega _ { k } \mathbf { z } _ { 0 } + \cos \omega _ { k } \epsilon = - \sqrt { 1 - \bar { \alpha } _ { k } } \mathbf { z } _ { 0 } + \sqrt { \bar { \alpha } _ { k } } \epsilon \equiv \mathbf { v } _ { k } .\tag{29}
$$

Because $\mathbf { z } _ { 0 }$ and ϵ are orthogonal and have unit variance, $\mathbf { z } _ { k }$ and $\mathbf { v } _ { k }$ also form an orthonormal pair. Geometrically, $\mathbf { z } _ { k }$ represents the position on the unit circle (or hypersphere) and the vector $\mathbf { v } _ { k }$ defines the tangent (velocity) direction of the probability-flow trajectory. The derivative with respect to k shows us that the magnitude of the diffusion motion is modulated by $\omega _ { k } ^ { \prime } ( k )$ and the direction is fully determined by $\mathbf { v } _ { k } .$

$$
\frac { d \mathbf { z } _ { k } } { d k } = \frac { d \omega _ { k } } { d k } \frac { \partial \mathbf { x } _ { k } } { \partial \omega _ { k } } = \omega _ { t } ^ { \prime } ( k ) \mathbf { v } _ { k } ,\tag{30}
$$

Algorithm 2 SimCast-S2S Training   
Require: Dataset D   
Require: Pretrained VAE encoders $\mathcal { E } _ { g }$ for $g \in$ {wind, mass, thermal, hydro, precip}   
Require: Neural denoiser $f _ { \theta }$   
Require: Diffusion schedule $\{ \bar { \alpha } _ { k } \} _ { k = 1 } ^ { K }$ , with $\bar { \alpha } _ { 0 } = 1$   
repeat   
Sample $\left( \mathbf { x } _ { \mathrm { w i n d } } , \mathbf { x } _ { \mathrm { m a s s } } , \mathbf { x } _ { \mathrm { t } } \right.$ <sub>hermal</sub>, x<sub>hydro</sub>, x<sub>precip</sub>, $t , \mathbf { y } _ { \mathrm { p r e c i p } } ) \in \mathbb { D }$   
for $g \in$ {wind, mass, thermal, hydro, precip} do   
Compute: $\left( \mu _ { g } , \sigma _ { g } ^ { 2 } \right) = \mathcal { E } _ { g } \left( \mathbf { x } _ { g } \right)$   
Sample: $\epsilon _ { g } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$   
Compute conditioning latent: $\mathbf { z } _ { g } = \pmb { \mu } _ { g } + \pmb { \sigma } _ { g } \odot \pmb { \epsilon } _ { g }$   
end for   
Concatenate: $\mathbf { Z } _ { \mathrm { c o n d } } = \mathbf { z } _ { \mathrm { w i n d } } \oplus \mathbf { z } _ { \mathrm { m a s s } } \oplus \mathbf { z } _ { \mathrm { t } }$ <sub>hermal</sub> ⊕ z<sub>hydro</sub> ⊕ z<sub>precip</sub>.   
Compute: $( \pmb { \mu } _ { y } , \pmb { \sigma } _ { y } ^ { 2 } ) = \mathcal { E } _ { \mathrm { p r e c i p } } \left( \mathbf { y } _ { \mathrm { p r e c i p } } \right)$   
Sample: $\epsilon _ { y } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$   
Compute clean latent: $\mathbf { z } _ { 0 } = \pmb { \mu } _ { y } + \pmb { \sigma } _ { y } \odot \pmb { \epsilon } _ { y }$   
Sample: $k \sim$ Uniform $( \{ 1 , \ldots , K \} )$   
Sample: $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$   
Compute noisy latent: $\begin{array} { r } { { \bf z } _ { k } = \sqrt { \bar { \alpha } _ { k } } { \bf z } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { k } } \epsilon . } \end{array}$   
Compute true velocity: $\mathbf { v } _ { k } = - \sqrt { 1 - \bar { \alpha } _ { k } } \mathbf { z } _ { 0 } + \sqrt { \bar { \alpha } _ { k } } \epsilon .$   
Predict velocity: $\hat { \mathbf { v } } _ { k } = f _ { \pmb { \theta } } \left( \mathbf { Z } _ { \mathrm { c o n d } } , t , \mathbf { z } _ { k } , k \right)$   
Take a gradient step on: $\nabla _ { \pmb { \theta } } \| \mathbf { v } _ { k } - \hat { \mathbf { v } } _ { k } \| _ { 2 } ^ { 2 }$   
until converged

Since $\mathbf { v } _ { k }$ is still a unit vector, i.e., $\mathrm { V a r } ( \mathbf { v } _ { k } ) = \mathbf { I }$ , the transition from the ϵ-prediction to the v-prediction formulation can be interpreted as a change of coordinate basis from the original $( \mathbf { z } _ { 0 } , \epsilon )$ system to the rotated $\left( \mathbf { z } _ { k } , \mathbf { v } _ { k } \right)$ system. As shown in Fig. 12, this transformation corresponds to a $9 0 ^ { \circ }$ counterclockwise rotation on the unit circle in the $( \mathbf { z } _ { 0 } , \epsilon )$ plane, preserving orthogonality and unit norm. Consequently, conversion between two basis becomes immediately straightforward:

$$
\{ \begin{array} { l l } { \mathbf { z } _ { 0 } = \sqrt { \bar { \alpha } _ { k } } \mathbf { z } _ { k } - \sqrt { 1 - \bar { \alpha } _ { k } } \mathbf { v } _ { k } , } & { \qquad \quad \mathbf { z } _ { k } = \sqrt { \bar { \alpha } _ { k } } \mathbf { z } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { k } } \epsilon , } \\ { \epsilon = \sqrt { 1 - \bar { \alpha } _ { k } } \mathbf { z } _ { k } + \sqrt { \bar { \alpha } _ { k } } \mathbf { v } _ { k } } & { \qquad ( 3 \mathrm { l a } ) } \end{array}  \qquad \begin{array} { l l } { \{ \mathbf { z } _ { k } = \sqrt { \bar { \alpha } _ { k } } \mathbf { z } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { k } } \epsilon ,  } \\ {  \mathbf { v } _ { k } = - \sqrt { 1 - \bar { \alpha } _ { k } } \mathbf { z } _ { 0 } + \sqrt { \bar { \alpha } _ { k } } \epsilon \} = 0 . } \end{array}\tag{31b}
$$

The signal-to-noise ratio (SNR) at timestep k is defined as the ratio between the signal variance and the noise variance:

$$
\mathrm { S N R } _ { k } = \frac { \mathrm { V a r } ( \sqrt { { \bar { \alpha } } _ { k } } \mathbf { z } _ { 0 } ) } { \mathrm { V a r } ( \sqrt { 1 - { \bar { \alpha } } _ { k } } \epsilon ) } = \frac { { \bar { \alpha } } _ { k } } { 1 - { \bar { \alpha } } _ { k } } .\tag{32}
$$

The standard ϵ-prediction objective minimizes the MSE between the true and predicted noise: $\mathcal { L } _ { \epsilon } =$ $\mathbb { E } _ { k } \| \boldsymbol { \epsilon } - \boldsymbol { \hat { \epsilon } } \| ^ { 2 }$ . To examine how this loss contributes to the construction loss $\mathcal { L } _ { \mathbf { z } _ { 0 } } = \mathbb { E } _ { k } ^ { - } \lVert \mathbf { z } _ { 0 } - \hat { \mathbf { z } } _ { 0 } \rVert ^ { 2 }$ at different denoising step $k _ { : }$ , we rewrite $\mathcal { L } _ { \mathbf { z } _ { 0 } }$ in terms of the true clean sample $\begin{array} { r } { { \bf z } _ { 0 } = \frac { 1 } { \sqrt { \bar { \alpha } _ { k } } } [ { \bf z } _ { k } - \sqrt { 1 - \bar { \alpha } _ { k } } \epsilon ] } \end{array}$ and the predicted clean sample $\begin{array} { r } { \hat { \mathbf { z } } _ { 0 } = \frac { 1 } { \sqrt { \bar { \alpha } _ { k } } } [ \mathbf { z } _ { k } - \sqrt { 1 - \bar { \alpha } _ { k } } \hat { \epsilon } ] } \end{array}$ . The reconstruction loss is then:

$$
\mathcal { L } _ { \mathbf { z } _ { 0 } } = \mathbb { E } _ { k } \Big \| \mathbf { z } _ { 0 } - \hat { \mathbf { z } } _ { 0 } \Big \| ^ { 2 } = \mathbb { E } _ { k } \left[ \frac { 1 - \bar { \alpha } _ { k } } { \bar { \alpha } _ { k } } \Big \| \epsilon - \hat { \epsilon } \Big \| ^ { 2 } \right] = \mathbb { E } _ { k } \left[ \frac { 1 } { { \mathrm { S N R } } _ { k } } \Big \| \epsilon - \hat { \epsilon } \Big \| ^ { 2 } \right] .\tag{33}
$$

We see that each term in $\mathcal { L } _ { \epsilon }$ contributes differently to the target reconstruction loss $\mathcal { L } _ { \mathbf { z } _ { 0 } }$ by $\frac { 1 } { \mathrm { S N R } _ { k } }$ . This weight blows up as $\bar { \alpha } _ { k }  0 ( \mathrm { i . e . } k  K )$ . Although ϵ-prediction still provides an unbiased estimator of the score function, its inverse-SNR imbalance implies poor numerical conditioning and motivates the v-prediction formulation, whose contribution remains stable over time.

```latex
Algorithm 3 SimCast-S2S Generating
Require: Input fields: x<sub>wind</sub>, x<sub>mass</sub>, $\mathbf { x } _ { \mathrm { t } }$ <sub>hermal</sub>, x<sub>hydro</sub>, x<sub>precip</sub>
Require: Day-of-year information t
Require: Pretrained VAE encoders $\mathcal { E } _ { g }$ for $g \in$ {wind, mass, thermal, hydro, precip}
Require: Pretrained precipitation decoder $\mathcal { D } _ { \mathrm { p r e c i p } }$
Require: Trained neural denoiser $f _ { \theta }$
Require: Diffusion schedule $\{ \bar { \alpha } _ { k } \} _ { k = 0 } ^ { K } ,$ with $\bar { \alpha } _ { 0 } = 1$
Require: Stochasticity parameter $\eta \in [ 0 , 1 ]$
for $g \in$ {wind, mass, thermal, hydro, precip} do
Compute: $\left( \mu _ { g } , \pmb { \sigma } _ { g } ^ { 2 } \right) = \mathcal { E } _ { g } \left( \mathbf { x } _ { g } \right)$
Sample: $\epsilon _ { g } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$
Compute conditioning latent: $\mathbf { z } _ { g } = \pmb { \mu } _ { g } + \pmb { \sigma } _ { g } \odot \pmb { \epsilon } _ { g }$
end for
Concatenate: $\begin{array} { r } { \mathbf { Z } _ { \mathrm { c o n d } } = \mathbf { z } _ { \mathrm { w i n d } } \oplus \mathbf { z } _ { \mathrm { m a s s } } \oplus \mathbf { z } _ { \mathrm { t h e r m a l } } \oplus \mathbf { z } _ { \mathrm { h y d r o } } \oplus \mathbf { z } _ { \mathrm { p r e c i p } } . } \end{array}$
Initialize: $\hat { \mathbf { z } } _ { K } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$
for $k = K , K - 1 , \ldots , 1$ do
Predict velocity: $\hat { \mathbf { v } } _ { k } = f _ { \pmb { \theta } } ( \mathbf { Z } _ { \mathrm { c o n d } } , t , \hat { \mathbf { z } } _ { k } , k ) .$
Compute clean latent estimate: $\hat { \bf z } _ { 0 } = \sqrt { \bar { \alpha } _ { k } } \hat { \bf z } _ { k } - \sqrt { 1 - \bar { \alpha } _ { k } } \hat { \bf v } _ { k }$
Compute noise estimate: $\hat { \pmb { \epsilon } } _ { k } = \sqrt { 1 - \bar { \alpha } _ { k } } \hat { \bf z } _ { k } + \sqrt { \bar { \alpha } _ { k } } \hat { \bf v } _ { k } .$
Compute: $\begin{array} { r } { \sigma _ { k } = \eta \sqrt { \frac { 1 - \bar { \alpha } _ { k - 1 } } { 1 - \bar { \alpha } _ { k } } } \sqrt { 1 - \frac { \bar { \alpha } _ { k } } { \bar { \alpha } _ { k - 1 } } } . } \end{array}$
Compute: $c _ { k } = \sqrt { 1 - \bar { \alpha } _ { k - 1 } - \sigma _ { k } ^ { 2 } } .$
Sample: $\zeta _ { k } \sim \dot { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } )$
Compute: $\hat { \mathbf { z } } _ { k - 1 } = \sqrt { \bar { \alpha } _ { k - 1 } } \hat { \mathbf { z } } _ { 0 } + c _ { k } \hat { \mathbf { \epsilon } } _ { k } + \sigma _ { k } \boldsymbol { \zeta } _ { k } .$
end for
return precipitation-anomaly forecast: $\hat { \mathbf { y } } = \mathcal { D } _ { \mathrm { p r e c i p } } \left( \hat { \mathbf { z } } _ { 0 } \right)$
```

Indeed, writing the reconstruction loss $\mathcal { L } _ { \mathbf { z } _ { 0 } }$ with $\mathbf { z } _ { 0 } = \sqrt { \bar { \alpha } _ { k } } \mathbf { z } _ { k } - \sqrt { 1 - \bar { \alpha } _ { k } } \mathbf { v } _ { k }$ and $\hat { \bf z } _ { 0 } = \sqrt { \bar { \alpha } _ { k } } { \bf z } _ { k } - \sqrt { 1 - \bar { \alpha } _ { k } } \hat { \bf v } _ { k }$ it now becomes:

$$
\mathcal { L } _ { \mathbf { z } _ { 0 } } = \mathbb { E } _ { k } \big \| \mathbf { z } _ { 0 } - \hat { \mathbf { z } } _ { 0 } \big \| ^ { 2 } = \mathbb { E } _ { k } \left[ ( 1 - \bar { \alpha } _ { k } ) \left\| \pmb { v } _ { k } - \hat { \pmb { v } } _ { k } \right\| ^ { 2 } \right] .\tag{34}
$$

Therefore, minimizing $\mathcal { L } _ { \mathbf { v } } = \mathbb { E } _ { k } \big \| \pmb { v } _ { k } - \hat { \pmb { v } } _ { \theta } ( \mathbf { z } _ { k } , k ) \big \| ^ { 2 }$ induces $\mathbf { z } _ { 0 }$ -errors that scale only by the bounded factors $1 - \bar { \alpha } _ { k } \in [ 0 , 1 ]$ . As a result, although v-prediction optimizes a step-stationary target (unit variance for all k) just like ϵ-prediction, it has no inverse-SNR amplification. The per-step gradient magnitudes remain commensurate across k, which implies much more stable optimization.

From the five pretrained VAEs corresponding to the five groups of physical fields listed in Table 3, together with the loss function in Eq. 11, the general sampling rule in Eq. 24, and the basis conversions in Eq. 31a–31b, we obtain the complete training and generating algorithms for SimCast-S2S, summarized in Algorithms 2–3, respectively.

In this study, $\mathbf { Z } _ { \mathrm { c o n d } }$ and t encode the physical atmospheric states over the previous 28 days and their associated seasonal context, respectively. Both $\mathbf { Z } _ { \mathrm { c o n d } }$ and t are step-independent and remain fixed throughout the denoising process. The neural network uses these inputs to guide the reverse process so that the generated latent $\hat { \mathbf { z } } _ { 0 }$ represents the precipitation anomaly over lead days 15–28, corresponding to the 14-day subseasonal forecast window. A conceptual illustration of the latent diffusion process is provided in Fig. 11. During generation, the neural network denoises the latent state sequentially from step $K$ to step 0. It should be noted that the generation process is initialized randomly and $\hat { \mathbf { z } } _ { K }$ can be far apart from $\mathbf { z } _ { K }$ . However, the neural network uses the conditioning information to guide the denoising trajectory so that $\hat { \mathbf { z } } _ { 0 }$ is expected to be close to $\mathbf { z } _ { 0 }$ . Additionally, if $\eta = 0$ , the denoising trajectory (yellow) is deterministic for a fixed initial latent $\hat { \mathbf { z } } _ { K }$ . If $0 < \eta \leq 1$ , additional Gaussian noise is injected at each reverse step, which makes the denoising trajectory stochastic.

![](images/49ebe10cfd0f62c8f72405b4e7d6b5b678468b1605ee5717cd9bb133866b678d.jpg)  
Figure 13: Architecture of the SimCast-S2S denoiser. At step k, the noisy target precipitation latent $\hat { \mathbf { z } } _ { k }$ is passed through a sequence of ConvStack and Transformer blocks to predict the velocity $\hat { \mathbf { v } } _ { k }$ . Although the denoiser is trained to predict $\hat { \mathbf { v } } _ { k }$ , the next denoised latent state $\hat { \mathbf { z } } _ { k - 1 }$ is obtained using Eqs. 31a–31b and Eq. 24. The conditioning information is formed by concatenating the latent representations of the five physical groups, $\mathbf { z } _ { \mathrm { w i n d } }$ , z<sub>mass</sub>, z<sub>thermal</sub>, $\mathbf { z } _ { \mathrm { h y d r o } }$ , and $\mathbf { z } _ { \mathrm { p r e c i p } }$ , into $\mathbf { Z } _ { \mathrm { c o n d } }$ , which is injected into the Transformer blocks through cross-attention. The diffusion step k and seasonal information t are encoded using separate sinusoidal embedding layers and injected into the ConvStack blocks. This denoising operation is repeated for K reverse steps, transforming an initial Gaussian latent sample $\hat { \mathbf { z } } _ { K } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ into the generated target precipitation latent $\hat { \mathbf { z } } _ { 0 }$

The neural network $f _ { \theta }$ is designed as a compact denoising engine for latent-space forecasting. It follows an encoder–decoder structure with multiple downsampling blocks, each combining convolutional and transformer layers, followed by a symmetric set of upsampling blocks. The convolutional layers refine local spatial structure within the latent fields and the transformer layers handle the temporal exchange of information between the conditioning history and the forecast target.

This temporal structure is explicit. The conditioning sequence contains 28 tokens, one for each input day, and the target sequence contains 14 tokens, one for each forecast day. Through cross-attention [98], the 14 noisy target tokens query the 28 conditioning tokens and extract the parts of the past atmospheric evolution most relevant for denoising the future precipitation latent. In this formulation, $\mathbf { Z } _ { \mathrm { c o n d } }$ provides the key–value pairs, and the intermediate representation of $\hat { \mathbf { z } } _ { k }$ provides the queries. The denoiser therefore selectively learns which parts of the 28-day atmospheric history should inform each forecast day.

The diffusion step k and day-of-year t are passed through separate sinusoidal embedding blocks [98], which convert scalar inputs into vector representations while preserving their relative proximity in Euclidean space. Given the noisy target latent $\hat { \mathbf { z } } k$ , the conditioning latent sequence $\mathbf { Z } _ { \mathrm { c o n d } }$ , the diffusion-step embedding, and the day-of-year embedding, the denoiser predicts the velocity $\hat { \mathbf { v } } _ { k }$ . This predicted velocity defines the direction used to move the latent state from $\hat { \mathbf { z } } _ { k }$ toward $\hat { \mathbf { z } } _ { k - 1 }$ at each reverse step. The detailed architecture of $f _ { \theta }$ is provided in Fig. 13

## 4.4 Transfer Learning: LoRA

Perhaps one of the most important features of SimCast-S2S is its computationally efficient latent-space framework. This efficiency allows SimCast-S2S to be pretrained on 28 CESM2 atmospheric simulations and then transferred to real-world atmospheric estimates from ERA5. Traditional transfer learning typically follows one of common strategies: fine-tuning the full pretrained network with a small learning rate [99, 100, 101], freezing early layers and retraining only task-specific layers [102, 103], or keeping the pretrained backbone mostly fixed while adding small trainable adaptation modules [104, 105]. SimCast-S2S follows the third strategy by using low-rank adaptation (LoRA) [106].

LoRA transfer-learning strategy is applied to $\mathcal { E } _ { \mathrm { p r e c i p } } , \mathcal { D } _ { \mathrm { p r e c i p } } .$ , and $f _ { \theta } .$ . We do not adapt the VAE encoders and decoders for group wind, mass, thermal, hydro because these VAEs show no substantial degradation when evaluated on ERA5 with no fine-tuning. In contrast, precipitation exhibits a stronger distributional shift between CESM2 and ERA5, so the precipitation encoder and decoder are fine-tuned together with the denoising network.

Each ConvStack layer in Figs. 10 and 13 contains multiple convolutional layers [89, 90], and each Transformer layer in Fig. 13 also contains multiple dense layers as their core components [98]. For any such layer, let $\mathbf { W } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times d _ { \mathrm { o u t } } }$ denote the pretrained weight matrix learned from CESM2, where $d _ { \mathrm { i n } }$ and $d _ { \mathrm { o u t } }$ are the input and output dimensions, respectively.

Instead of fine-tuning the full weight matrix W, which can be computationally expensive and may overwrite useful structure learned during CESM2 pretraining, LoRA freezes W and learns only a low-rank update $\Delta { \bf W }$ :

$$
\mathbf { W } ^ { \prime } = \mathbf { W } + \Delta \mathbf { W } .\tag{35}
$$

The matrix $\Delta \mathbf { W }$ can be written in a low-rank factorization:

$$
\Delta \mathbf { W } = \mathbf { B } \mathbf { A } ,\tag{36}
$$

where $\mathbf { B } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times r } , \mathbf { A } \in \mathbb { R } ^ { r \times d _ { \mathrm { o u t } } }$ , and most importantly, $r \ll \operatorname* { m i n } ( d _ { \mathrm { i n } } , d _ { \mathrm { o u t } } )$ is the rank of matrix $\Delta { \bf W }$ . For an input feature vector h $\mathbf { \Sigma } \in \mathbb { R } ^ { d _ { \mathrm { i r } } }$ , the adapted output becomes

$$
\mathbf { h } \mathbf { W } ^ { \prime } = \mathbf { h } \mathbf { W } + \mathbf { h } \mathbf { B } \mathbf { A } .\tag{37}
$$

The first term on the right-hand side preserves the pretrained CESM2 knowledge. The second term introduces a small trainable correction that adapts the model from CESM2 to ERA5. During fine-tuning, only A and B are optimized, W remains intact. As a result, we only need to optimize $O ( r ( d _ { \mathrm { i n } } + d _ { \mathrm { o u t } } ) )$ number of parameters, instead of $O ( d _ { \mathrm { i n } } d _ { \mathrm { o u t } } )$ . Because r is chosen to be much smaller than both $d _ { \mathrm { i n } }$ and $d _ { \mathrm { o u t } }$ $O ( r ( d _ { \mathrm { i n } } + d _ { \mathrm { o u t } } ) )$ is orders of magnitude smaller than $O ( d _ { \mathrm { i n } } d _ { \mathrm { o u t } } )$ . This makes LoRA more memory efficient and less prone to overfitting.

LoRA is particularly useful for SimCast-S2S because the model can exploit a large ensemble of CESM2 simulations. As shown in Section 2.3, CESM2 pretraining followed by ERA5 fine-tuning with LoRA performs substantially better than training on ERA5 alone, and it is one of the main reasons SimCast-S2S outperforms the process-based ECMWF-S2S. The only task assigned to SimCast-S2S during fine-tuning is to adjust for the domain shift from many simulated worlds to one single real world.

## 4.5 Deep Learning Baselines: CNN, UNet

SimCast-S2S is benchmarked against two widely used deep learning architectures for spatial prediction: CNN and UNet. CNN models are built on top of the convolution operator and provide a natural baseline for gridded prediction tasks and provide a natural baseline for gridded prediction tasks because they learn local spatial patterns through shared kernels [89, 90]. The UNet architecture extends this idea with an encoder–decoder structure and skip connections that help preserve multiscale spatial information [53]. In this study, we construct three CNN baselines and three U-Net baselines with different model sizes. Tables 4–5 summarize the key design choices.

Table 4: Architectural configurations of the CNN baselines. Each CNN uses a stack of convolutional layers (N) with a fixed embedding dimension (d). The three model sizes increase both the embedding dimension and the number of layers.
<table><tr><td>Model</td><td>d</td><td>N</td></tr><tr><td>cnn-small</td><td>256</td><td>4</td></tr><tr><td>cnn-medium</td><td>512</td><td>8</td></tr><tr><td>cnn-large</td><td>1024</td><td>12</td></tr></table>

Table 5: Architectural configurations of the UNet baselines. Each U-Net uses a symmetric encoder–decoder structure with four sampling blocks, with embedding dimensions $\{ d _ { i } \} _ { i = 1 } ^ { 4 }$ , respectively. In the encoder, each block downsamples the spatial resolution and doubles the embedding dimension. In the decoder, the corresponding upsampling blocks recover the spatial resolution and halve the embedding dimension.
<table><tr><td>Model</td><td> $d _ { 1 }$ </td><td> $d _ { 2 }$ </td><td> $d _ { 3 }$ </td><td> $d _ { 4 }$ </td></tr><tr><td>unet-small</td><td>128</td><td>256</td><td>512</td><td>1024</td></tr><tr><td>unet-medium</td><td>256</td><td>512</td><td>1024</td><td>2048</td></tr><tr><td>unet-large</td><td>512</td><td>1024</td><td>2048</td><td>4096</td></tr></table>

## 4.6 Physical Baseline: ECMWF-S2S

To place SimCast-S2S against a strong process-based reference, we use ECMWF-S2S as the operational physical baseline. ECMWF is widely regarded as one of the world-leading numerical weather prediction centers, and its subseasonal forecasts, ECMWF-S2S, represent a highly developed dynamical forecasting system [7, 2, 107]. Unlike neural-network baselines, ECMWF-S2S is produced by integrating a physically based Earth-system model that explicitly resolves atmospheric dynamics, parameterizes unresolved physical processes, and generates ensemble forecasts through perturbed initial conditions and model uncertainties [8, 2]. Outperforming ECMWF-S2S would indicate that SimCast-S2S is competitive with one of the most advanced operational forecasting systems currently available. At the same time, this comparison should be interpreted with care. SimCast-S2S is not intended to replace ECMWF-S2S. Instead, it provides an inexpensive alternative for subseasonal precipitation forecasting, accessible to research groups and computing centers with smaller-scale computational resources.

## 4.7 Evaluation Metrics

## 4.7.1 Deterministic Skill Metrics

We evaluate deterministic skill over all test samples using the ensemble-mean precipitation-anomaly forecast. Let $\hat { \mathbf { y } } ^ { ( m ) }$ and $\mathbf { y } ^ { ( m ) }$ denote the predicted and verifying precipitation-anomaly field for test sample m, respectively. Here, $m = 1 , \ldots , M$ indexes the test samples and $i = 1 , \ldots , N$ indexes the spatial grid points. The first deterministic metric is mean absolute error (MAE), which measures the average magnitude of the grid-level forecast error across samples and space:

$$
\mathrm { M A E } = \frac { 1 } { M N } \sum _ { m = 1 } ^ { M } \sum _ { i = 1 } ^ { N } \left| \hat { y } _ { i } ^ { ( m ) } - y _ { i } ^ { ( m ) } \right| \quad \in \mathbb { R } .\tag{38}
$$

Lower MAE indicates smaller absolute error and therefore better deterministic accuracy. Because all precipitation fields are converted to standardized anomalies, MAE measures error relative to the local climatology.

The second deterministic metric is anomaly correlation coefficient (ACC), which evaluates whether the forecast captures the spatial pattern of the observed precipitation anomaly. For each test sample, ACC is first computed over the spatial grid:

$$
\operatorname { A C C } ^ { ( m ) } = \frac { \sum _ { i = 1 } ^ { N } \hat { y } _ { i } ^ { ( m ) } y _ { i } ^ { ( m ) } } { \sqrt { \sum _ { i = 1 } ^ { N } \left( \hat { y } _ { i } ^ { ( m ) } \right) ^ { 2 } } \sqrt { \sum _ { i = 1 } ^ { N } \left( y _ { i } ^ { ( m ) } \right) ^ { 2 } } } \quad \in \mathbb { R } .\tag{39}
$$

The reported ACC is then averaged across all test samples:

$$
\mathrm { A C C } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathrm { A C C } ^ { ( m ) } \quad \in \mathbb { R } .\tag{40}
$$

Higher ACC indicates stronger agreement between the predicted and observed anomaly patterns. A skillful subseasonal precipitation forecast should achieve both low MAE and high ACC, meaning that it is close to the observed anomaly field in magnitude and also reproducing the spatial structure of wet and dry anomalies.

## 4.7.2 Probabilistic Skill Metrics

Probabilistic skill is evaluated using the full ensemble distribution the ensemble mean. For each test sample m, SimCast-S2S produces an ensemble of precipitation-anomaly forecasts $\{ \hat { \mathbf { y } } ^ { ( m , e ) } \} _ { e = 1 } ^ { E }$ , where E is the number of ensemble members. These ensemble members define an empirical predictive distribution at each grid point. The probabilistic metrics evaluate whether this distribution assigns high probability to the observed outcome and whether it improves over a reference forecast distribution.

The first probabilistic metric is the ranked probability skill score (RPSS), which is used for tercile-based categorical forecasts. For each grid point, the precipitation anomaly distribution is divided into three climatological categories: below-normal, near-normal, and above-normal. Let $\mathbf { p } ^ { ( m ) } = ( p _ { 1 } ^ { ( m ) } , p _ { 2 } ^ { ( m ) } , p _ { 3 } ^ { ( m ) } )$ denote the forecast probabilities assigned to the three categories for sample $m .$ , and let $\mathbf { o } ^ { ( m ) } = ( o _ { 1 } ^ { ( m ) } , o _ { 2 } ^ { ( m ) } , o _ { 3 } ^ { ( m ) } )$ denote the observed one-hot category vector. The ranked probability score (RPS) is

$$
\operatorname { R P S } ^ { ( m ) } = \sum _ { c = 1 } ^ { 3 } \left( \sum _ { j = 1 } ^ { c } p _ { j } ^ { ( m ) } - \sum _ { j = 1 } ^ { c } o _ { j } ^ { ( m ) } \right) ^ { 2 } \quad \in \mathbb { R } ^ { N } .\tag{41}
$$

The RPSS compares the forecast RPS against the climatological reference. The sample-level RPSS is defined as

$$
\mathrm { R P S S } ^ { ( m ) } = 1 - \frac { \overline { { \mathrm { R P S } } } _ { \mathrm { f o r e c a s t } } ^ { ( m ) } } { \overline { { \mathrm { R P S } } } _ { \mathrm { c l i m a t o l o g y } } ^ { ( m ) } } \quad \in \mathbb { R } .\tag{42}
$$

Here and throughout the rest of this section, the overbar denotes averaging over all N grid points. The reported RPSS is obtained by averaging the sample-level skill scores over all M test samples. Positive RPSS indicates that the probabilistic forecast improves upon climatology. Negative RPSS indicates worse performance than climatology.

A more general version of RPSS is the continuous ranked probability skill score (CRPSS), which evaluates the full continuous predictive distribution rather than discrete tercile categories. For a predictive cumulative distribution function $F ^ { ( m ) }$ and verifying observation $y ^ { ( m ) }$ of sample m, the continuous ranked probability score (CRPS) is

$$
\mathrm { C R P S } \left( F ^ { ( m ) } , y ^ { ( m ) } \right) = \int _ { - \infty } ^ { \infty } \left[ F ^ { ( m ) } ( x ) - \mathbf { 1 } _ { y ^ { ( m ) } \leq x } \right] ^ { 2 } d x \quad \in \mathbb { R } ^ { N } .\tag{43}
$$

For ensemble forecasts, $F ^ { ( m ) }$ is represented by the empirical distribution of ensemble members. The CRPSS of sample m is then defined as

$$
\mathrm { C R P S S } ^ { ( m ) } = 1 - \frac { \overline { { \mathrm { C R P S } } } _ { \mathrm { f o r e c a s t } } ^ { ( m ) } } { \overline { { \mathrm { C R P S } } } _ { \mathrm { c l i m a t o l o g y } } ^ { ( m ) } } \quad \in \mathbb { R } .\tag{44}
$$

Similar to RPSS, the reported RPSS is the average of all sample-level skill scores in the test dataset. Higher CRPSS is better, positive CRPSS indicates improvement from the climatology.

The third probabilistic metric is the Brier skill score (BSS), which is used to evaluate binary event probabilities. In this study, BSS is applied to extreme precipitation events exceeding the 90th percentile. Let $\bar { p } ^ { ( m ) }$ denote the forecast probability of the event estimated from the ensemble forecasts, and let the one-hot category $o ^ { ( m ) } \in \{ 0 , 1 \}$ } denote whether the extreme event occurred. The Brier score of sample m is

$$
\begin{array} { r l } { \mathrm { B S } ^ { ( m ) } = \left( p ^ { ( m ) } - o ^ { ( m ) } \right) ^ { 2 } } & { { } \in \mathbb { R } ^ { N } . } \end{array}\tag{45}
$$

The BSS also compares this score with the corresponding climatological event probability:

$$
\mathrm { B S S } ^ { ( m ) } = 1 - \frac { \overline { { \mathrm { B S } } } _ { \mathrm { f o r e c a s t } } ^ { ( m ) } } { \overline { { \mathrm { B S } } } _ { \mathrm { c l i m a t o l o g y } } ^ { ( m ) } } \in \mathbb { R } .\tag{46}
$$

The reported BSS is also the average score of all test samples. Positive BSS indicates that the ensemble forecasts capture extreme events better than the climatological reference.

Although RPSS, CRPSS, and BSS evaluate different aspects of probabilistic forecast quality, they share a common idea. A skillful subseasonal precipitation forecast should place probability mass near the observed precipitation anomaly. The three metrics differ mainly in how they break down the forecast probability density function (PDF). RPSS evaluates probability mass over discretized partitions of the PDF, CRPSS evaluates the continuous PDF as it is, and BSS evaluates the right tail of the PDF for extreme precipitation events.

## 4.7.3 Uncertainty Quantification

A powerful metric for quantifying the uncertainty of ensemble forecasts is the probability integral transform (PIT), which directly tests whether the predictive distribution is statistically consistent with the observations. The idea is simple. For an unbiased probabilistic forecast, the observation should behave like a random draw from the forecast distribution. Therefore, the PIT values should be uniformly distributed on [0, 1].

For test sample m and grid point i, let $\widehat { F } _ { m , i }$ denote the empirical cumulative distribution function (CDF) defined by the E–member ensemble forecast $\{ \hat { y } _ { i } ^ { ( m , e ) } \} _ { e = 1 } ^ { E }$ . The PIT value is computed as

$$
u _ { m , i } = { \widehat { F } } _ { m , i } \left( y _ { i } ^ { ( m ) } \right) = { \frac { 1 } { E } } \sum _ { e = 1 } ^ { E } \mathbf { 1 } \left\{ { \hat { y } } _ { i } ^ { ( m , e ) } \leq y _ { i } ^ { ( m ) } \right\} .\tag{47}
$$

where $y _ { i } ^ { ( m ) }$ is the true precipitation anomaly. An unbiased ensemble should produce PIT values that are approximately uniform. A ∪–shaped PIT histogram indicates underdispersion, meaning that the ensemble spread is too narrow and observations fall too often in the tails of the forecast distribution. A ∩–shaped PIT histogram indicates overdispersion, meaning that the ensemble spread is too wide.

To quantify the deviation of the PIT histogram from uniformity, we compute the $\chi ^ { 2 } .$ –distance to flatness. Let the interval [0, 1] be divided into B equally spaced bins, and let

$$
n _ { b } = \sum _ { m = 1 } ^ { M } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \left\{ u _ { m , i } \in I _ { b } \right\}\tag{48}
$$

denote the number of PIT values falling in bin $I _ { b }$ . The total number of PIT values is MN, so the expected count under a perfectly uniform PIT distribution is $\textstyle { \frac { M N } { B } }$ in each bin. The $\chi ^ { 2 }$ distance to flatness is then defined as

$$
\chi ^ { 2 } = \sum _ { b = 1 } ^ { B } \frac { \left( n _ { b } - \frac { M N } { B } \right) ^ { 2 } } { \frac { M N } { B } } .\tag{49}
$$

A lower value of $\chi ^ { 2 } .$ –distance indicates a PIT histogram closer to uniformity and therefore better capture the true uncertainty. A large $\chi ^ { 2 } .$ –distance value indicates that the ensemble distribution is systematically inconsistent with the observed precipitation anomalies, either because the ensemble is underdispersed, or overdispersed.

A completely different view on uncertainty quantification is provided by the relationship between ensemble spread and deterministic error. The spread–RMSE relationship evaluates whether the ensemble spread is consistent with the actual error of the ensemble-mean forecast. We want larger forecast uncertainty to correspond to larger ensemble-mean error. Consequently, the average ensemble spread should be comparable to the average RMSE.

For each test sample m and grid point i, the ensemble-mean forecast over all E members is

$$
\bar { y } _ { i } ^ { ( m ) } = \frac { 1 } { E } \sum _ { e = 1 } ^ { E } \hat { y } _ { i } ^ { ( m , e ) } ,\tag{50}
$$

and the ensemble spread is

$$
s _ { i } ^ { ( m ) } = \sqrt { \frac { 1 } { E - 1 } \sum _ { e = 1 } ^ { E } \left( \hat { y } _ { i } ^ { ( , e ) } - \bar { y } _ { i } ^ { ( m ) } \right) ^ { 2 } } .\tag{51}
$$

The spatially averaged ensemble spread for sample m is then defined as

$$
\mathrm { S p r e a d } ^ { ( m ) } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( s _ { i } ^ { ( m ) } \right) ^ { 2 } } ,\tag{52}
$$

and the corresponding RMSE of the ensemble mean is simply

$$
\mathrm { R M S E } ^ { ( m ) } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \bar { y } _ { i } ^ { ( m ) } - y _ { i } ^ { ( m ) } \right) ^ { 2 } } .\tag{53}
$$

For an ideally statistically consistent ensemble, it can be shown that the spread–RMSE diagram should approximately be the 1:1 line [108]. If the spread is much smaller than the RMSE, the ensemble is underdispersed and does not represent enough forecast uncertainty. If the spread is much larger than the RMSE, the ensemble is overdispersed and produces an unnecessarily broad forecast distribution.

We also evaluate uncertainty using the mean interval score (MIS). For a central $( 1 - \alpha )$ prediction interval, let $l _ { i } ^ { ( m ) }$ and $u _ { i } ^ { ( m ) }$ denote the lower and upper ensemble quantiles at grid point i for sample $m .$ The interval score is defined as

$$
\mathrm { I S } _ { i } ^ { ( m ) } ( \alpha ) = \left( u _ { i } ^ { ( m ) } - l _ { i } ^ { ( m ) } \right) + \frac { 2 } { \alpha } \left( l _ { i } ^ { ( m ) } - y _ { i } ^ { ( m ) } \right) \mathbf { 1 } \left\{ y _ { i } ^ { ( m ) } < l _ { i } ^ { ( m ) } \right\} + \frac { 2 } { \alpha } \left( y _ { i } ^ { ( m ) } - u _ { i } ^ { ( m ) } \right) \mathbf { 1 } \left\{ y _ { i } ^ { ( m ) } > u _ { i } ^ { ( m ) } \right\} .\tag{54}
$$

The MIS is obtained by averaging over all test samples and grid points:

$$
\mathrm { M I S } ( \alpha ) = \frac { 1 } { M N } \sum _ { m = 1 } ^ { M } \sum _ { i = 1 } ^ { N } \mathrm { I S } _ { i } ^ { ( m ) } ( \alpha ) .\tag{55}
$$

Lower MIS indicates better interval forecasts. The first term $( u _ { i } ^ { ( m ) } - l _ { i } ^ { ( m ) } )$ rewards sharp prediction intervals. The terms $\begin{array} { r } { \frac { 2 } { \alpha } ( l _ { i } ^ { ( m ) } - y _ { i } ^ { ( m ) } ) \mathbf { 1 } \{ y _ { i } ^ { ( m ) } < l _ { i } ^ { ( m ) } \} } \end{array}$ and $\mathbf { \Delta } _ { \alpha } ^ { 2 } ( y _ { i } ^ { ( m ) } - u _ { i } ^ { ( m ) } ) \mathbf { 1 } \{ y _ { i } ^ { ( m ) } > u _ { i } ^ { ( m ) } \}$ penalize observations that fall outside the predicted interval. Therefore, MIS jointly evaluates sharpness and statistical consistency. A good ensemble should produce intervals that are narrow but still contain the observed precipitation anomaly.

## 4.7.4 Realism Quantification: Spatial Auto-correlation

Beyond deterministic accuracy and probabilistic skill, we also care about whether the generated precipitationanomaly fields have realistic spatial structure. This is important because a forecast can obtain reasonable pointwise scores but smooths out the physical structures at the same time. A good way to quantify realism is comparing the spatial autocorrelation between the forecasts and the true observations. Spatial autocorrelation quantifies the internal spatial texture of a precipitation field by measuring how strongly the field correlates with a spatially shifted version of itself.

For a precipitation-anomaly field $\mathbf { y } ^ { ( m ) }$ , we compute directional autocorrelation by shifting the field by a spatial lag ℓ and correlating the original field with the shifted field over their overlapping grid points. Let $\mathcal { S } _ { d , \ell }$ denote the shift operator in direction d with lag ℓ, where $\textit { d } \in$ {zonal, meridional, main-diagonal, anti-diagonal}.

For example, a zonal shift compares each grid point with another grid point ℓ pixels away in the longitudinal direction, a meridional shift compares grid points ℓ pixels apart in the latitudinal direction. The two diagonal shifts compare grid points displaced simultaneously in both spatial dimensions.

Let $\Omega _ { d , \ell }$ denote the set of grid points for which both the original field and the shifted field are defined. The observed directional autocorrelation for test sample $m .$ , in direction $d ,$ and with lag ℓ is defined as

$$
\rho _ { \mathrm { o b s } } ^ { ( m ) } ( d , \ell ) = \frac { \sum _ { i \in \Omega _ { d , \ell } } \left( y _ { i } ^ { ( m ) } - \bar { y } _ { d , \ell } ^ { ( m ) } \right) \left( S _ { d , \ell } y _ { i } ^ { ( m ) } - \overline { { S _ { d , \ell } y } } _ { d , \ell } ^ { ( m ) } \right) } { \sqrt { \sum _ { i \in \Omega _ { d , \ell } } \left( y _ { i } ^ { ( m ) } - \bar { y } _ { d , \ell } ^ { ( m ) } \right) ^ { 2 } } \sqrt { \sum _ { i \in \Omega _ { d , \ell } } \left( S _ { d , \ell } y _ { i } ^ { ( m ) } - \overline { { S _ { d , \ell } y } } _ { d , \ell } ^ { ( m ) } \right) ^ { 2 } } } ,\tag{56}
$$

where $\bar { y } _ { d , \ell } ^ { ( m ) }$ and $\overline { { S _ { d , \ell } y } } _ { d , \ell } ^ { ( m ) }$ are the spatial means of the original and shifted fields over the overlapping domain $\Omega _ { d , \ell }$ . The same calculation is applied to each forecast ensemble member. For ensemble member $e ,$ the forecast spatial autocorrelation is $\rho _ { \mathrm { f o r e c a s t } } ^ { ( m , e ) } ( d , \ell )$

We compute this quantity for multiple lags ℓ and for the four directions shown in Fig. 7. The resulting distributions are summarized with boxplots across test samples and ensemble members.

A realistic precipitation generator should reproduce the observed decay of spatial autocorrelation with increasing lag. In other words, the forecast autocorrelation $\rho _ { \mathrm { f o r e c a s t } } ^ { ( m , e ) } ( d , \ell )$ should follow the same lagdependent structure as the observed autocorrelation $\rho _ { \mathrm { o b s } } ^ { ( m ) } ( d , \ell )$ across directions d and spatial lags ℓ. If $\rho _ { \mathrm { f o r e c a s t } } ^ { ( m , e ) } ( d , \ell )$ remains too high relative to $\rho _ { \mathrm { o b s } } ^ { ( m ) } ( d , \ell )$ as ℓ increases, the generated fields are overly smooth. If $\rho _ { \mathrm { f o r e c a s t } } ^ { ( m , e ) } ( d , \ell )$ decays too rapidly, the generated fields are too noisy.

## References

[1] National Academies of Sciences, Engineering, and Medicine. Next Generation Earth System Prediction: Strategiesfor Subseasonal to Seasonal Forecasts. The National Academies Press, Washington, DC, 2016.

[2] F. Vitart, C. Ardilouze, A. Bonet, A. Brookshaw, M. Chen, C. Codorean, M. Déqué, L. Ferranti, E. Fucile, M. Fuentes, H. Hendon, J. Hodgson, H.-S. Kang, A. Kumar, H. Lin, G. Liu, X. Liu, P. Malguzzi, I. Mallas, M. Manoussakis, D. Mastrangelo, C. MacLachlan, P. McLean, A. Minami, R. Mladek, T. Nakazawa, S. Najm, Y. Nie, M. Rixen, A. W. Robertson, P. Ruti, C. Sun, Y. Takaya, M. Tlostykh, F. Venuti, D. Waliser, S. Woolnough, T. Wu, D.-J. Won, H. Xiao, R. Zaripov, and L. Zhang. The subseasonal to seasonal (s2s) prediction project database. Bulletin of the American Meteorological Society, 98(1):163–173, 2017.

[3] Kathy Pegion, Ben P. Kirtman, Emily Becker, Dan C. Collins, Erik LaJoie, Robert Burgman, Ray Bell, Tim DelSole, Daehyun Min, Yan Zhu, Wei Li, Eric Sinsky, Hui Guan, Jon Gottschalck, Eric J. Metzger, Neil P. Barton, Deepthi Achuthavarier, Jelena Marshak, Randal Koster, Hai Lin, Norman Gagnon, Michael Bell, Michael K. Tippett, Andrew W. Robertson, Shuhua Sun, Steven G. Benjamin, Benjamin W. Green, Rainer Bleck, Hyemi Kim, Jia Wang, Sarah Henderson, and Adrian M. Tompkins. The subseasonal experiment (SubX): A multimodel subseasonal prediction experiment. Bulletin ofthe American Meteorological Society, 100(10):2043–2060, 2019.

[4] A. W. Robertson, F. Vitart, and S. J. Camargo. Subseasonal to seasonal prediction of weather to climate with application to tropical meteorology. Journal of Geophysical Research: Atmospheres, 125(7):e2018JD029375, 2020.

[5] Christopher J. White and Coauthors. Advances in the application and utility of subseasonal-to-seasonal predictions. Bulletin ofthe American Meteorological Society, 103(6):E1448–E1472, 2022.

[6] E. N. Lorenz. The predictability of a flow which possesses many scales of motion. Tellus, 21(3):289– 307, 1969.

[7] Peter Bauer, Alan Thorpe, and Gilbert Brunet. The quiet revolution of numerical weather prediction. Nature, 525:47–55, 2015.

[8] Martin Leutbecher and Tim N. Palmer. Ensemble forecasting. Journal of Computational Physics, 227(7):3515–3539, 2008.

[9] T. N. Palmer and D. L. T. Anderson. The prospects for seasonal forecasting—a review paper. Quarterly Journal ofthe Royal Meteorological Society, 120(518):755–793, 1994.

[10] Randal D. Koster, Sarith P. P. Mahanama, T. J. Yamada, Gianpaolo Balsamo, Alexis A. Berg, Marie Boisserie, Paul A. Dirmeyer, Francisco J. Doblas-Reyes, Guy Drewitt, Christopher T. Gordon,

Zhichang Guo, Jee-Hoon Jeong, David M. Lawrence, Wan-Shik Lee, Zhao Li, Lifeng Luo, Sergey Malyshev, William J. Merryfield, Sonia I. Seneviratne, Tanja Stanelle, Bart J. J. M. van den Hurk, Frédéric Vitart, and Eric F. Wood. Contribution of land surface initialization to subseasonal forecast skill: First results from a multi-model experiment. Geophysical Research Letters, 37(2):L02402, 2010.

[11] Annarita Mariotti and Coauthors. Windows of opportunity for skillful forecasts subseasonal to seasonal and beyond. Bulletin ofthe American Meteorological Society, 101(5):E608–E625, 2020.

[12] William J. Merryfield and Coauthors. Current and emerging developments in subseasonal to decadal prediction. Bulletin of the American Meteorological Society, 101(6):E869–E896, 2020.

[13] Chidong Zhang. Madden-julian oscillation. Reviews of Geophysics, 43(2):RG2003, 2005.

[14] Mark P. Baldwin and Timothy J. Dunkerton. Stratospheric harbingers of anomalous weather regimes. Science, 294(5542):581–584, 2001.

[15] Jadwiga H. Richter, Anne A. Glanville, Teagan King, Sanjiv Kumar, Stephen G. Yeager, Nicholas A. Davis, Yanan Duan, Megan D. Fowler, Abby Jaye, Jim Edwards, Julie M. Caron, Paul A. Dirmeyer, Gokhan Danabasoglu, and Keith Oleson. Quantifying sources of subseasonal prediction skill in CESM2. npj Climate and Atmospheric Science, 7(59), 2024.

[16] F. Vitart and A. W. Robertson. The sub-seasonal to seasonal prediction project (s2s) and the prediction of extreme events. npj Climate and Atmospheric Science, 1(1):3, 2018.

[17] Zoltan Toth and Eugenia Kalnay. Ensemble forecasting at NMC: The generation of perturbations. Bulletin ofthe American Meteorological Society, 74(12):2317–2330, 1993.

[18] Franco Molteni, Roberto Buizza, Tim N. Palmer, and Thomas Petroliagis. The ECMWF ensemble prediction system: Methodology and validation. Quarterly Journal of the Royal Meteorological Society, 122(529):73–119, 1996.

[19] R. Buizza, M. Miller, and T. N. Palmer. Stochastic representation of model uncertainties in the ecmwf ensemble prediction system. Quarterly Journal ofthe Royal Meteorological Society, 125(560):2887– 2908, 1999.

[20] Tilmann Gneiting and Adrian E. Raftery. Weather forecasting with ensemble methods. Science, 310(5746):248–249, 2005.

[21] Philippe Bougeault, Zoltan Toth, Craig Bishop, Barbara Brown, David Burridge, De Hui Chen, Beth Ebert, Manuel Fuentes, Thomas M. Hamill, Ken Mylne, Jennifer Nicolau, Tiziana Paccagnella, Young-Youn Park, David Parsons, Baudouin Raoult, Daniel Schuster, Pedro Silva Dias, Richard Swinbank, Yasushi Takeuchi, Warren Tennant, Laurence Wilson, and Steve Worley. The THORPEX interactive grand global ensemble. Bulletin ofthe American Meteorological Society, 91(8):1059–1072, 2010.

[22] Richard Swinbank, Masashi Kyouda, Peter Buchanan, Lizzie Froude, Thomas M. Hamill, Tim D. Hewson, Jeffrey H. Keller, Mio Matsueda, John Methven, Florian Pappenberger, Michael Scheuerer, Helen A. Titley, Laurence Wilson, and Munehiko Yamaguchi. The TIGGE project and its achievements. Bulletin ofthe American Meteorological Society, 97(1):49–67, 2016.

[23] Tim N. Palmer. The ECMWF ensemble prediction system: Looking back (more than) 25 years and projecting forward 25 years. Quarterly Journal of the Royal Meteorological Society, 145(S1):12–24, 2019.

[24] T. N. Palmer. Predicting uncertainty in forecasts of weather and climate. Reports on Progress in Physics, 63(2):71–116, 2000.

[25] Edward N. Lorenz. Deterministic nonperiodic flow. Journal of the Atmospheric Sciences, 20(2):130– 141, 1963.

[26] Cecil E. Leith. Theoretical skill of monte carlo forecasts. Monthly Weather Review, 102(6):409–418, 1974.

[27] J. M. Fritsch and R. E. Carbone. Improving quantitative precipitation forecasts in the warm season: A uswrp research and development strategy. Bulletin ofthe American Meteorological Society, 85(7):955– 965, 2004.

[28] E. E. Ebert and J. L. McBride. Verification of precipitation in weather systems: Determination of systematic errors. Journal ofHydrology, 239(1–4):179–202, 2000.

[29] A. Arakawa. The cumulus parameterization problem: Past, present, and future. Journal of Climate, 17(13):2493–2525, 2004.

[30] S. Vannitsem, D. S. Wilks, and J. W. Messner. Statistical postprocessing of ensemble forecasts. Elsevier, 2021.

[31] Oliver Watt-Meyer, Noah D. Brenowitz, Spencer K. Clark, Brian Henn, Anna Kwa, Jeremy McGibbon, W. Andre Perkins, and Christopher S. Bretherton. Correcting weather and climate models by machine learning nudged historical simulations. Geophysical Research Letters, 48(15):e2021GL092555, 2021.

[32] Zied Ben-Bouallègue, Mariana C. A. Clare, Linus Magnusson, Estibaliz Gascon, Michael Maier-Gerber, Martin Janousek, Mark Rodwell, Florian Pinault, Jesper S. Dramsch, Simon T. K. Lang, Baudouin Raoult, Florence Rabier, Matthieu Chevallier, Irina Sandu, Peter Dueben, Matthew Chantry, and Florian Pappenberger. The rise of data-driven weather forecasting. Bulletin of the American Meteorological Society, 105(6):E1057–E1078, 2024.

[33] S. Rasp, P. D. Dueben, S. Scher, J. A. Weyn, S. Mouatadid, and N. Thuerey. Weatherbench: A benchmark dataset for data-driven weather forecasting. Journal of Advances in Modeling Earth Systems, 12(11):e2020MS002203, 2020.

[34] J. Hwang, P. Orenstein, J. Cohen, K. Pfeiffer, and L. Mackey. Improving subseasonal forecasting in the western u.s. with machine learning. arXiv preprint arXiv:1809.07394, 2019.

[35] Hiep Vo Dang and Phong C. H. Nguyen. Deep operator learning for high-fidelity fluid flow field reconstruction from sparse sensor measurements. Journal of Computing and Information Science in Engineering, 26(1):011007, 2026.

[36] L. Chen, X. Zhong, H. Li, J. Wu, B. Lu, D. Chen, S.-P. Xie, L. Wu, Q. Chao, C. Lin, Z. Hu, and Y. Qi. A machine learning model that outperforms conventional global subseasonal forecast models. Nature Communications, 15(1):6425, 2024.

[37] J. A. Weyn, D. R. Durran, and R. Caruana. Data-driven medium-range weather prediction with a resnet pretrained on climate simulations: A new model for weatherbench. Journal ofAdvances in Modeling Earth Systems, 13(2):e2020MS002405, 2021.

[38] T. Nguyen, J. Brandstetter, A. Kapoor, J. K. Gupta, and A. Grover. Climax: A foundation model for weather and climate. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 25904–25938, 2023.

[39] C. Bodnar, W. P. Bruinsma, A. Lucic, M. Stanley, J. Brandstetter, P. Garvan, M. Riechert, J. Weyn, H. Dong, A. Vaughan, J. K. Gupta, K. Tambiratnam, A. Archibald, E. Heider, M. Welling, R. E. Turner, and P. Perdikaris. A foundation model for the earth system. Nature, 641:1180–1187, 2025.

[40] L. Li, R. Carver, I. Lopez-Gomez, F. Sha, and J. Anderson. Generative emulation of weather forecast ensembles with diffusion models. Science Advances, 10(13):eadk4489, 2024.

[41] I. Price, A. Sanchez-Gonzalez, F. Alet, T. R. Andersson, A. El-Kadi, D. Masters, T. Ewalds, J. Stott, S. Mohamed, P. Battaglia, R. Lam, and M. Willson. Probabilistic weather forecasting with machine learning. Nature, 637:84–90, 2025.

[42] P. D. Dueben and P. Bauer. Challenges and design choices for global weather and climate models based on machine learning. Geoscientific Model Development, 11(10):3999–4009, 2018.

[43] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, volume 27, 2014.

[44] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, volume 29, 2016.

[45] L. Mescheder, A. Geiger, and S. Nowozin. Which training methods for gans do actually converge? In Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 3481–3490, 2018.

[46] D. P. Kingma and M. Welling. Auto-encoding variational bayes. In International Conference on Learning Representations, 2014.

[47] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.

[48] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. International Conference on Learning Representations, 2021.

[49] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[50] Keith B. Rodgers, Sun-Seon Lee, Nan Rosenbloom, Axel Timmermann, Gokhan Danabasoglu, Clara Deser, Jim Edwards, Jong-Seong Kim, Isla R. Simpson, Keith Stein, Malte F. Stuecker, Ryohei Yamaguchi, Tamás Bódai, Eui-Seok Chung, Lei Huang, Who M. Kim, Jean-François Lamarque, Danica L. Lombardozzi, William R. Wieder, and Stephen G. Yeager. Ubiquity of human-induced changes in climate variability. Earth System Dynamics, 12(4):1393–1411, 2021.

[51] H. Hersbach, B. Bell, P. Berrisford, S. Hirahara, A. Horányi, J. Muñoz-Sabater, J. Nicolas, C. Peubey, R. Radu, D. Schepers, A. Simmons, C. Soci, S. Abdalla, X. Abellan, G. Balsamo, P. Bechtold, G. Biavati, J. Bidlot, M. Bonavita, G. De Chiara, P. Dahlgren, D. Dee, M. Diamantakis, R. Dragani, J. Flemming, R. Forbes, M. Fuentes, A. Geer, L. Haimberger, S. Healy, R. J. Hogan, E. Hólm, M. Janisková, S. Keeley, P. Laloyaux, P. Lopez, C. Lupu, G. Radnoti, P. de Rosnay, I. Rozum, F. Vamborg, S. Villaume, and J.-N. Thépaut. The era5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730):1999–2049, 2020.

[52] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[53] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, volume 9351 of Lecture Notes in Computer Science, pages 234–241. Springer, 2015.

[54] Stephan Rasp and Nils Thuerey. Data-driven medium-range weather prediction with a resnet pretrained on climate simulations: A new model for weatherbench. Journal ofAdvances in Modeling Earth Systems, 13(2):e2020MS002405, 2021.

[55] Yoo-Geun Ham, Jeong-Hwan Kim, and Jing-Jia Luo. Deep learning for multi-year ENSO forecasts. Nature, 573(7775):568–572, 2019.

[56] Jose González-Abad, Maialen Iturbide, Alfonso Hernanz, and José Manuel Gutiérrez. Pre-training for deep statistical climate downscaling: Enhancing consistency and robustness across regional datasets. Geoscientific Model Development, 19:5781–5804, 2026.

[57] Alper Unal, Busra Asan, Ismail Sezen, Bugra Yesilkaynak, Yusuf Aydin, Mehmet Ilicak, and Gozde Unal. Climate model-driven seasonal forecasting approach with deep learning. Environmental Data Science, 2:e29, 2023.

[58] Scott A. Martin, Noah Brenowitz, Dale Durran, and Michael Pritchard. Long-range distillation: Distilling 10,000 years of simulated climate into long timestep AI weather models. arXiv preprint arXiv:2512.22814, 2025.

[59] Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, and Soumith Chintala. PyTorch Distributed: Experiences on accelerating data parallel training. Proceedings ofthe VLDB Endowment, 13(12):3005–3018, 2020.

[60] Gene M. Amdahl. Validity of the single processor approach to achieving large scale computing capabilities. In Proceedings of the April 18–20, 1967, Spring Joint Computer Conference, AFIPS ’67 (Spring), pages 483–485, New York, NY, USA, 1967. Association for Computing Machinery.

[61] John L. Gustafson. Reevaluating Amdahl’s Law. Communications ofthe ACM, 31(5):532–533, 1988.

[62] European Centre for Medium-Range Weather Forecasts. Atmospheric model sub-seasonal forecast (set VI – sub-seasonal). https://www.ecmwf.int/en/forecasts/datasets/set-vi. Accessed: 28 July 2026.

[63] Peter Bauer, Tiago Quintino, Nils Wedi, Antonino Bonanni, Marcin Chrust, Willem Deconinck, Michail Diamantakis, Peter Düben, Stephen English, Johannes Flemming, Paddy Gillies, Ioan Hadade, James Hawkes, Mike Hawkins, Olivier Iffrig, Christian Kühnlein, Michael Lange, Peter Lean, Olivier Marsden, Andreas Müller, Sami Saarinen, Domokos Sarmany, Michael Sleigh, Simon Smart, Piotr Smolarkiewicz, Daniel Thiemert, Giovanni Tumolo, Christian Weihrauch, Cristiano Zanna, and Pedro Maciel. The ECMWF scalability programme: Progress and plans. ECMWF Technical Memorandum 857, European Centre for Medium-Range Weather Forecasts, February 2020.

[64] Michaël Mathieu, Camille Couprie, and Yann LeCun. Deep multi-scale video prediction beyond mean square error. In International Conference on Learning Representations, 2016.

[65] Suman Ravuri, Karel Lenc, Matthew Willson, Dmitry Kangin, Remi Lam, Piotr Mirowski, Megan Fitzsimons, Maria Athanassiadou, Sheleem Kashem, Sam Madge, Rachel Prudden, Amol Mandhane, Aidan Clark, Andrew Brock, Karen Simonyan, Raia Hadsell, Niall Robinson, Ellen Clancy, Alberto Arribas, and Shakir Mohamed. Skilful precipitation nowcasting using deep generative models of radar. Nature, 597:672–677, 2021.

[66] Lucy Harris, Andrew T. T. McRae, Matthew Chantry, Peter D. Dueben, and Tim N. Palmer. A generative deep learning approach to stochastic downscaling of precipitation forecasts. Journal of Advances in Modeling Earth Systems, 14(10):e2022MS003120, 2022.

[67] Tilmann Gneiting and Matthias Katzfuss. Probabilistic forecasting. Annual Review of Statistics and Its Application, 1:125–151, 2014.

[68] Thorsten Kurth, Shashank Subramanian, Peter Harrington, Jaideep Pathak, Morteza Mardani, David Hall, Andrea Miele, Karthik Kashinath, and Animashree Anandkumar. FourCastNet: Accelerating global high-resolution weather forecasting using adaptive fourier neural operators. In Proceedings of the Platform for Advanced Scientific Computing Conference, 2023.

[69] Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Timo Ewalds, Andrew El-Kadi, Jack Stott, Shakir Mohamed, Peter Battaglia, Remi Lam, Matthew Willson, et al. Probabilistic weather forecasting with machine learning. Nature, 637:84–90, 2025.

[70] Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. Automatic differentiation in machine learning: A survey. Journal of Machine Learning Research, 18(153):1–43, 2018.

[71] Maziar Raissi, Paris Perdikaris, and George Em Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal ofComputational Physics, 378:686–707, 2019.

[72] George Em Karniadakis, Ioannis G. Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3:422–440, 2021.

[73] Phong CH Nguyen, Yen-Thi Nguyen, Joseph B Choi, Pradeep K Seshadri, HS Udaykumar, and Stephen S Baek. Parc: Physics-aware recurrent convolutional neural networks to assimilate meso scale reactive mechanics of energetic materials. Science advances, 9(17):eadd6868, 2023.

[74] Phong CH Nguyen, Xinlun Cheng, Shahab Azarfar, Pradeep Seshadri, Yen T Nguyen, Munho Kim, Sanghun Choi, HS Udaykumar, and Stephen Baek. Parcv2: Physics-aware recurrent convolutional neural networks for spatiotemporal dynamics modeling. arXiv preprint arXiv:2402.12503, 2024.

[75] Antonios Mamalakis, Imme Ebert-Uphoff, and Elizabeth A. Barnes. Explainable artificial intelligence in meteorology and climate science: Model fine-tuning, calibrating trust and learning new science. In Andreas Holzinger, Randy Goebel, Ruth Fong, Taesup Moon, Klaus-Robert Müller, and Wojciech Samek, editors, xxAI – Beyond Explainable AI, volume 13200 of Lecture Notes in Computer Science, pages 315–339. Springer, Cham, 2022.

[76] Antonios Mamalakis, Elizabeth A. Barnes, and Imme Ebert-Uphoff. Investigating the fidelity of explainable artificial intelligence methods for applications of convolutional neural networks in geoscience. Artificial Intelligencefor the Earth Systems, 1(4), 2022.

[77] Timothy B. Higgins, Aneesh C. Subramanian, Andre Graubner, Lukas Kapp-Schwoerer, Peter A. G. Watson, Sarah Sparrow, Karthik Kashinath, Sol Kim, Luca Delle Monache, and William Chapman. Using deep learning for an analysis of atmospheric rivers in a high-resolution large ensemble climate data set. Journal ofAdvances in Modeling Earth Systems, 15(5):e2022MS003495, 2023.

[78] Timothy B. Higgins, Aneesh C. Subramanian, Will E. Chapman, David A. Lavers, and Andrew C. Winters. Subseasonal potential predictability of horizontal water vapor transport and precipitation extremes in the north pacific. Weather and Forecasting, 39(6):833–846, 2024.

[79] Shivam Singh, Manish Kumar Goyal, and Srinidhi Jha. Role of large-scale climate oscillations in precipitation extremes associated with atmospheric rivers: Nonstationary framework. Hydrological Sciences Journal, 68(3):395–411, 2023.

[80] Shivam Singh and Manish Kumar Goyal. An innovative approach to predict atmospheric rivers: Exploring convolutional autoencoder. Atmospheric Research, page 106754, 2023.

[81] Manish Kumar Goyal and Shivam Singh. Understanding Atmospheric Rivers Using Machine Learning. SpringerBriefs in Applied Sciences and Technology. Springer, 2024.

[82] Jeremy Malcolm Corner, Antonios Mamalakis, and Kathleen Schiro. Artificial intelligence identifies shortwave troughs as an important local feature of extreme precipitation. ESS Open Archive preprint, 2026.

[83] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "why should i trust you?": Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1135–1144, 2016.

[84] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 3319–3328. PMLR, 2017.

[85] Sandra Wachter, Brent Mittelstadt, and Chris Russell. Counterfactual explanations without opening the black box: Automated decisions and the gdpr. Harvard Journal ofLaw & Technology, 31(2):841–887, 2017.

[86] Gokhan Danabasoglu, Jean-François Lamarque, Julio Bacmeister, David A. Bailey, Alice K. DuVivier, Jim Edwards, Louisa K. Emmons, John Fasullo, Rolando Garcia, Andrew Gettelman, Cecile Hannay, Marika M. Holland, William G. Large, Peter H. Lauritzen, David M. Lawrence, Jan T. M. Lenaerts, Keith Lindsay, William H. Lipscomb, Michael J. Mills, Richard Neale, Keith W. Oleson, Bette

Otto-Bliesner, Adam S. Phillips, William Sacks, Simone Tilmes, Leo van Kampenhout, Mariana Vertenstein, Alessandro Bertini, John Dennis, Clara Deser, Charles Fischer, Baylor Fox-Kemper, Jennifer E. Kay, Douglas Kinnison, Paul J. Kushner, Vincent E. Larson, Matthew C. Long, Stephen Mickelson, J. Keith Moore, Eric Nienhouse, Lorenzo Polvani, Philip J. Rasch, and W. G. Strand. The community earth system model version 2 (CESM2). Journal of Advances in Modeling Earth Systems, 12(2):e2019MS001916, 2020.

[87] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In International Conference on Learning Representations, 2014.

[88] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

[89] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings ofthe IEEE, 86(11):2278–2324, 1998.

[90] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, volume 25, 2012.

[91] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings ofthe 32nd International Conference on Machine Learning, volume 37 of Proceedings ofMachine Learning Research, pages 2256–2265. PMLR, 2015.

[92] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, volume 32, 2019.

[93] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.

[94] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

[95] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[96] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Yingxia Shao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4):1–39, 2023.

[97] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.

[98] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

[99] Sinno Jialin Pan and Qiang Yang. A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering, 22(10):1345–1359, 2010.

[100] Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neural networks? In Advances in Neural Information Processing Systems, volume 27, 2014.

[101] Fuzhen Zhuang, Zhiyuan Qi, Keyu Duan, Dongbo Xi, Yongchun Zhu, Hengshu Zhu, Hui Xiong, and Qing He. A comprehensive survey on transfer learning. Proceedings ofthe IEEE, 109(1):43–76, 2021.

[102] Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, and Trevor Darrell. DeCAF: A deep convolutional activation feature for generic visual recognition. In Proceedings of the

31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 647–655. PMLR, 2014.

[103] Maxime Oquab, Léon Bottou, Ivan Laptev, and Josef Sivic. Learning and transferring mid-level image representations using convolutional neural networks. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 1717–1724, 2014.

[104] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning multiple visual domains with residual adapters. In Advances in Neural Information Processing Systems, volume 30, 2017.

[105] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP. In Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2790–2799. PMLR, 2019.

[106] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

[107] Frédéric Vitart. The next extended-range configuration for IFS cycle 48r1. ECMWF Newsletter, 173, 2022.

[108] Vincent Fortin, Mohamed Abaza, François Anctil, and Richard Turcotte. Why should ensemble spread match the RMSE of the ensemble mean? Journal of Hydrometeorology, 15(4):1708–1713, 2014.