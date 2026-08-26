# Causal Analysis for Time Series Foundation Models

Mathis Jander 1,2,\*, Wouter van Heeswijk 1, and Martijn Mes 1

1Faculty of Behavioral, Management and Social Sciences, University of Twente, Enschede, Netherlands 2European Central Bank, Frankfurt am Main, Germany Corresponding author: mathis.jander@utwente.nl

August 26, 2026

## Abstract

Transitioning from bespoke time series models towards time series foundation models changes the relationship of model and application from one-to-one to one-to-many. This shift introduces concentration risk as many, potentially high-risk, forecasting applications are exposed to the same biases and failure modes of a single time series foundation model. At the same time, this centralization allows for economies of scale in model development and validation. In this study we investigate how biases and failure modes of time series foundation models can be identified before deployment. We propose a causal analysis framework to investigate the ability of a time series foundation model to preserve time series patterns. To achieve this, we intervene on parameterized synthetic time series generators and measure the corresponding change in model output under ceteris paribus conditions. We apply our causal analysis framework to Chronos-2 and TimesFM-2 . 5 and test them across six distinct time series patterns. We find safe configurations for trend and harmonic oscillation patterns. The results also indicate a bias in both models towards overestimating persistence, sudden failures for both models against the regime switch pattern and failure for TimesFM-2 . 5 against the energy-release pattern. Our review of the original works for both models indicates that the findings might be explained by the data used for pretraining. We conclude our study with suggestions for further model development, recommendations for application-specific model selection, and a discussion of limitations and further research directions.

## 1 Introduction

Forecasts are widely used to inform decision-making in critical domains such as finance, energy and policy. At the same time, the time series models that we use to create these forecasts get less interpretable as the size of the datasets and number of trainable parameters increase. Traditionally, for each forecasting application a bespoke model was trained on an application-specific dataset, having model development and deployment within on team or organisation. With the creation of large, pretrained time series foundation models, such as Chronos-2 (Ansari et al., 2025) and TimesFM-2.5 (Das et al., 2024), the development and the use of time series models is separated further as the foundation models are developed by organizations, such as Amazon Science or Google Research, that then grant access to a larger developer community for many different downstream applications. While this shift from bespoke models for each application to foundation models is expected to create economies of scale in training and application (Ansari et al., 2025; Das et al., 2024; Woo et al., 2024), it also creates two problems. First, it reduces the insight that the deploying organization has into model development and therefore what biases and failure modes it might have. Second, as one foundation model is intended to be used for many downstream applications, this introduces a systemic risk as the models biases and failure modes could affect many applications instead of being contained to one application as in the case of bespoke models. Looking at the economies of scale and the resulting risk profile, we argue that this centralization towards a few, widely used foundation models presents a necessity to scrutinize time series foundation models in more depth than bespoke models. Therefore, we will concern ourselves with the question of how we can identify biases and failure modes in time series foundation models before deploying them to any particular forecasting application.

This study makes four contributions to the current body of knowledge on time series foundation models. First, we define and apply a framework for causal analysis of time series foundation models. Second, we show dose-response relationships for Chronos-2 and TimesFM-2.5 for six distinct time series patterns that suggest a bias towards overestimating persistence and identify several failure modes. Third, through our analysis we find evidence for a bias towards overestimating persistence in both models and show sudden failures for the regime switch and energy-release generators. Fourth, based on our empirical findings we are able to inform model selection for downstream applications as well as the development of future time series foundation models.

The remainder of this study is structured as follows. We begin by reviewing the existing literature for answers on our research question in Section 2. We then define a causal analysis framework for time series foundation models and operationalize it with our experimental set-up in Section 3. We analyze the results in Section 4 and compare our findings with prior studies in Section 5. In Section 6, we conclude this study by summarizing our findings, acknowledging limitations and outlining avenues for further research.

## 2 Literature Review

Before defining our causal analysis framework, we review the existing literature related to our research question. To understand whether there is a methodological gap in our ability to identify and potentially mitigate biases and failure modes of time series foundation models, we first review examples of how other engineering disciplines deal with technologies that possess a similar risk profile. Having these examples as a reference, we can compare literature on in time series foundation models against these standards. Hereby, we start with the original studies introducing the Chronos-2 and the TimesFM-2.5 models. We then widen our scope to other studies on causal analysis of time series foundation models. Finally, we will review adjacent work on explainability and robustness to further define the extent of the gap in literature.

While foundation models present novel challenges in machine learning, other engineering disciplines already developed practices to assess technologies that will be used in many, potentially high-risk, contexts. In drug development, new compounds are required to go through a three-stage process before being released to the market (Food and Drug Administration, 2020; European Medicines Agency, 2009) In the first stage, a new compound is tested in vitro under ceteris paribus conditions to assert high degree of causal control. The goal of this stage is to isolate causal effects of the compound. In the second stage, preclinical trials on animals allow for in vivo testing of the compounds toxicity and other effects on an organism. Stage three then proceeds to in vivo testing on humans to validate the expected treatment effect. After market release, a drug is continuously monitored for rare adverse effects that might have not been noticed during the previous stages. In automotive manufacturing, car development follows a similar process (International Organization for Standardization, 2018; Peter, 2022; Zollino et al. 2025). In a first stage components and subsystems are tested in dedicated test rigs that simulate temperature, abrasion and other causes compromising their integrity over time. These test rigs allow to isolate a factor such as temperature and observe its effect on a component under ceteris paribus conditions. In a second stage, the full vehicle is tested under controlled conditions to validate the proper integration of all subsystems. This entails testing aerodynamics, fuel efficiency and crash behavior. After a car is released to market periodic inspections and defect investigations, potentially leading to recalls, are common depending on the jurisdiction. In both examples a new technology is first tested under ceteris paribus conditions to isolate causal effects before then validating the technologies expected behavior under more realistic, although also more confounding, conditions.

In the respective studies introducing Chronos-2 (Ansari et al., 2025) and TimesFM−2 .5 (Das et al., 2024), both time series foundation models are compared against other models by testing their predictive accuracy on benchmark datasets. For example, TimesFM-2 . 5 outperforms every other model on the Monash benchmark, while 1lmt ime is the second to last place. On the Darts benchmark on the other hand, 1lmtime is placed first and TimesFM-2.5 third. We cannot discern with confidence whether these performance differences are a consequence of the the training data, model architecture or any other factor. We also cannot tell whether these findings will hold for future observations of the same data-generating process. As it is similar to in vivo drug testing or test-driving a car, benchmarking alone does not allow us to understand what caused the observed performance results.

To the best of our knowledge, there is currently no literature applying causal analysis to study time series foundation models. While previous research explores causal inference from time series data, such as Granger causality (Granger, 1969) or Pearlian structural causal models (Pearl, 2010) to uncover causal relationships, these studies aim to infer cause and effect between variables from time series data and do not use time series data for model evaluation. Consequently, interventions on data-generating processes to study time series foundation model behavior remains unexplored.

Besides the literature concerned with time series foundation models, there are several adjacent research streams within the larger machine learning field that pursue related aims. Research on explainable AI produced several techniques to quantify the relationship between inputs and outputs of machine learning models. Prominent techniques are SHAP (Lundberg and Lee, 2017), LIME (Ribeiro et al., 2016), and Partial Dependence Plots (Friedman, 2001) which were developed for tabular data. The first two assign an importance score to each input feature of a given sample, based on idiosyncratic axioms and assumptions. Partial Dependence Plots on the other hand change a feature's value and observe the effect it has on the model output. Repeating this for a set of samples allows to estimate the average treatment effect of the intervention. Sweeping across several intervention values create a dose-response curve mapping the expected interaction between the input feature and the model output. While the intervention on realized samples is intuitive for most applications with tabular data, for time series this is not the case. Instead, we would like to intervene on patterns of the data-generating process itself, such as trend, seasonality or autocorrelation.

The robustness literature concerns itself with understanding how modifications of inputs degrades the performance of a machine learning model. This research stream can be roughly distinguished in two different subcategories (Braiek and Khomh, 2025). First, robustness to corruption of input samples, either from noise (Hendrycks and Dietterich, 2019) or intentional adversarial manipulation (Goodfellow et al., 2015). Second, from shifts in the data distribution (Recht et al., 2019; Koh et al., 2021). When we consider robustness analysis for our research question, we face three issues. First, because these methods modify observed values rather than controlling the underlying data-generating process, we face the same problem of meaning of an intervention on a time series as in the case of the explainable AI literature. Second, robustness analysis reveals to what extent model performance degrades under corruption of its inputs, but it does not reveal biases and failure modes of time series foundation models if we intervene on properties of the underlying data-generating process. While allowing us to analyze failure modes due to input corruption, robustness analysis does not allow us to identify biases and failure modes on uncorrupted samples. Third, the notion of robustness to data distribution shifts becomes unclear when we apply it to time series foundation models. Defining what the training data distribution is and what constitutes a distributional shift from that is not obvious considering the large and diverse amount of data time series foundation models are trained on (Ansari et al., 2025; Das et al., 2024).

To conclude, other engineering disciplines established staged processes to test high-risk technologies such as pharmacology and cars. The process begins with in vitro causal analysis under ceteris paribus conditions, trading off realism for the ability to identify causal mechanisms. In a second stage, the technology is then tested under more realistic conditions in vivo. This two-stage process allows us to identify causal mechanisms and map dose-response relationships in isolation and then validate them under more realistic conditions. Reviewing the literature on time series foundation models suggests that current validation efforts are focused on the second stage through performance benchmarking experiments on realized observations of uncontrolled data-generating processes. We also find that related research efforts, namely explainable AI and robustness analysis, do not provide methodologies for causal analysis of time series foundation model's biases and failure modes. Together, these findings outline the methodological gap that we aim to address in the remainder of this study.

## 3 Methodology

In this section, we first propose our causal analysis framework for time series foundation models in Section 3.1. We then describe the experimental configuration for this study in Section 3.2.

## 3.1 Causal Analysis Framework

To formalize our causal analysis framework for time series foundation models, we define a mathematical notation for the data-generating process, the time series foundation model, and our logic for causal inference.

Let $t \in \mathbb { N }$ represent a discrete point in time. A generator $G$ is a stochastic or deterministic function parameterized by a d-dimensional vector $\theta \in \Theta \subseteq \mathbb { R } ^ { d }$ . A generator is a function that maps discrete points in time to time series values

$$
G : \mathbb { N }  \mathbb { R } .\tag{1}
$$

A trajectory $\mathbf { y }$ is a single realization of length $T$ sampled from the generator

$$
\mathbf { y } = G _ { \theta } ( t ) + \epsilon , \quad \mathbf { y } \in \mathbb { R } ^ { T } , \quad t \in \{ 1 , \dots , T \} .\tag{2}
$$

By explicitly separating the structural parameters θ from the realization-specific noise $\epsilon ,$ we can isolate the impact of parameter interventions on model output.

A time series foundation model M is a function that accepts a trajectory y as input and returns an output trajectory $M ( \mathbf { y } )$ of length $H$

$$
M : \mathbb { R } ^ { T }  \mathbb { R } ^ { H } .\tag{3}
$$

A parameter statistic δ is a function that compresses a trajectory into a scalar value

$$
\delta : \mathbb { R } ^ { l }  \mathbb { R } , \quad l \in \{ T , H \} .\tag{4}
$$

We apply $\delta$ to estimate a property related to a generator parameter $\theta _ { i }$ from a realized trajectory.

For causal analysis, we rely on Pearl's causal framework (Pearl, 2010). Under this framework we can define our components as a directed acyclic graph where $\theta \to \mathbf { y } \to M ( \mathbf { y } )$

A parameter intervention utilizes Pearl's do-operator to set our target parameter $\theta _ { i }$ to a chosen value α while keeping all other parameters and the added noise sequence € constant to create a ceteris paribus condition. With this setup we can ensure that any observed change in $\delta ( \mathbf { y } )$ and $\delta ( M ( \mathbf { y } ) )$ is strictly caused by the change in our parameter $\theta _ { i } ,$ eliminating potential confounding effects from other parameters or noise. With that, we then can establish a dose-response relationship by mapping the scalar responses $\delta ( \mathbf { y } )$ and $\delta ( M ( \mathbf { y } ) )$ across an interval of the intervened parameter $\theta _ { i } .$ Figure 1 provides a sketch of the proposed causal analysis framework.

![](images/09e07f16f1ec88b47fb761d8c27deb23e11311cffd4cd5438d8f0dd2f99d0084.jpg)  
Figure 1: The causal analysis framework. Parameter intervention on $\theta _ { i }$ serves as controlled dose and parameter statistics on $\mathbf { y }$ and $M ( \mathbf { y } )$ serve as observed responses.

## 3.2 Experimental Configuration

For the empirical study of biases and failure modes in time series foundation models, we apply our causal analysis framework to Chronos-2 (Ansari et al., 2025) and TimesFM-2.5 (Das et al., 2024), as they are frequently downloaded time series forecasting models on Hugging Face1. We test both models across six different generators, each producing a distinct time series pattern. For all experimental setups, we fix $T = 2 0 0 , H = 2 0 0 .$ , and responses across $n = 5 0$ independent noise realizations.

The first generator models a random walk with drift, where the trajectory evolves according to a standard stochastic drift process defined by

$$
y _ { t } = y _ { t - 1 } + \mu + \epsilon _ { t } , \quad \epsilon _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) ,\tag{5}
$$

with the initial value fixed to $\begin{array} { r l r } { y _ { 0 } } & { { } = } & { 0 . 0 } \end{array}$ and the noise scale $\begin{array} { r l r l r l } { { \mathrm { ~ \bf ~ { ~ t o ~ } ~ } } \sigma } & { { } = } & { 1 . 0 . } & { } & { { } \mathrm { ~ W e ~ } } \end{array}$ perform a parameter intervention on the drift term, sweeping $\mu \in$ $\{ - 0 . 0 5 , - 0 . 0 2 5 , - 0 . 0 0 5 , 0 . 0 0 5 , 0 . 0 2 5 , 0 . 0 5 \}$ . As parameter statistic $\hat { \mu }$ we calculate the empirical mean of the first differences via

$$
{ \hat { \mu } } = { \frac { 1 } { n - 1 } } \sum _ { t = 2 } ^ { n } ( y _ { t } - y _ { t - 1 } ) .\tag{6}
$$

To test how foundation models preserve temporal memory and persistence, we implement a first-order autoregressive process, AR(1), governed by

$$
y _ { t } = c + \beta y _ { t - 1 } + \epsilon _ { t } , \quad \epsilon _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) ,\tag{7}
$$

with the fixed parameters $y _ { 0 } ~ = ~ 0 . 0 , c = 0 . 0$ , and $\sigma =$ 1.0. We perform intervention on the autoregressive coefficient across the sweep $\beta \in \{ - 0 . 5 , 0 . 0 , 0 . 3 , 0 . 6 , 0 . 8 5 , 0 . 9 8 \}$

To measure responses, the parameter statistic $\hat { \beta }$ estimates the $\boldsymbol { \mathrm { A R } } ( 1 ) \ \beta$ coefficient. We implement the estimation of $\hat { \beta }$ with statsmodels.tsa.ar\_model.AutoReg (Seabold and Perktold, 2010)

For the third experiment, we produce a periodic time series via a harmonic oscillator , which generates a stationary sine wave and add noise such that

$$
y _ { t } = A \sin \left( \frac { 2 \pi } { \lambda } t + \psi \right) + \epsilon _ { t } , \quad \epsilon _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) .\tag{8}
$$

Here, the amplitude is held at $A = 1 . 0$ , the phase shift at $\psi =$ 0.0, and the noise scale at $\sigma = 0 . 0 5$ . We intervene directly on the wavelength parameter $\lambda \in \{ 5 , 1 0 , 2 5 , 5 0 , 7 5 , 1 0 0 \}$ . The parameter statistic λ applies a Fast Fourier Transform (FFT) to the trajectory and extracts the most dominant frequency by peak height and converts it to wavelength. We implement $\hat { \lambda }$ with scipy. signal.welch (Virtanen et al., 2020).

To test whether structural breaks are preserved by both time series foundation models, we use a regime switch generator with fixed dwell time that features a piecewise linear trend with changing signs:

$$
y _ { t } = y _ { t - 1 } + m _ { t } + \epsilon _ { t } , \quad \epsilon _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) .\tag{9}
$$

The drift alternates between $m _ { t } ~ \in ~ \{ 1 . 0 , - 1 . 0 \}$ , reversing its state exactly every $\tau _ { d w }$ steps under a noise scale of $\sigma = 0 . 0 5$ . We intervene on the dwell time parameter $\tau \in \{ 5 , 1 0 , 2 5 , 5 0 , 7 5 , 1 0 0 \}$ . The estimated dwell time $\hat { \tau }$ serves as the parameter statistic. We use ruptures.Pelt (Truong et al., 2020) to calculate î.

To capture nonlinear trigger events and critical thresholds, we use an energy-release generator. This generator is characterized by a build-up phase and a reset once a threshold κ is crossed.

$$
y _ { t } = y _ { t - 1 } + s _ { t } + | \epsilon _ { t } |\tag{10}
$$

where $s _ { t } = 0 . 2$ and $\epsilon _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ with $\sigma = 0 . 0 5$ . Whenever $y _ { t } > \kappa$ , we reset $y _ { t } = 0$ . We intervene on the threshold $\kappa \in \{ 5 , 1 0 , 2 5 , 5 0 , 7 5 , 1 0 0 \}$ . We calculate the parameter statistic ê by averaging peak values in our trajectory. We identify a peak if a time step is followed by a negative increment that is larger than half of the maximum value in the trajectory.

For the last experiment, we test how well the time series foundation models preserve long-range dependence. We model fractional Brownian motion, fBm, process. The persistence of a trend, so an upward movement followed by an upward movement or a downward movement followed by a downward movement, is determined by the Hurst exponent H. For $H = 0 . 5$ , the process equals a random walk. For $H > 0 . 5$ trends are persistent and for $H \ : < \ : 0 . 5$ antipersistent. We intervene on the Hurst exponent, sweeping $H \in \{ 0 . 1 5 , 0 . 3 , 0 . 4 5 , 0 . 5 5 , 0 . 7 , 0 . 8 5 \}$ . As parameter statistic, we use  as the estimated Hurst exponent. To implement the generator we use fbm . FBM²and to calculate $\hat { H }$ we use hurst.compute\_ ${ \mathrm { ~ H ~ C ~ } } ^ { 3 }$ . We summarize all six experiments with their respective generator, intervention sweep and parameter statistic in Table 1.

Table 1: Summary of experiments with generators, intervention sweeps and parameter statistics.
<table><tr><td>Experiment Generator</td><td></td><td>Intervention Sweep</td><td>δ</td></tr><tr><td>1</td><td>Random Walk</td><td> $\mu \in \{ - 0 . 0 5 , - 0 . 0 2 5 , \ldots , 0 . 0 5 \}$ </td><td>ρ</td></tr><tr><td>2</td><td>AR(1)</td><td> $\beta \in \{ - 0 . 5 , 0 . 0 , \ldots , 0 . 9 8 \}$ </td><td>β</td></tr><tr><td>3</td><td>Harmonic Oscillator</td><td> $\lambda \in \{ 5 , 1 0 , 2 5 , 5 0 , 7 5 , 1 0 0 \}$ </td><td>λ</td></tr><tr><td>4</td><td>Regime Switch</td><td> $\tau \in \{ 5 , 1 0 , 2 5 , 5 0 , 7 5 , 1 0 0 \}$ </td><td>τ</td></tr><tr><td>5</td><td>Energy-release</td><td> $\kappa \in \{ 5 , 1 0 , 2 5 , 5 0 , 7 5 , 1 0 0 \}$ </td><td>κ</td></tr><tr><td>6</td><td>fBM</td><td> $H \in \{ 0 . 1 5 , 0 . 3 , \ldots , 0 . 8 5 \}$ </td><td>H</td></tr></table>

## 4 Results

In this section, we review the results of applying our causal analysis framework to the experiments outlined in the previous section. For each experiment, we compare different intervention trajectories y with the same noise realization € against the model outputs $M ( \mathbf { y } )$ of $\mathtt { C h r o n o s } { - } 2$ and TimesFM-2 . 5. We further compare the distributions of the trajectory parameter statistic $\delta ( \mathbf { y } )$ and the model output parameter statistic $\delta ( M ( \mathbf { y } ) )$ for a given intervention do $( \theta _ { i } = \alpha )$ across all $n = 5 0$ noise realizations, as well as plot $\delta ( \mathbf { y } )$ against $\delta ( M ( \mathbf { y } ) )$ for a given intervention and noise realization. Taken together, we are able to characterize potential biases and failure modes of $\mathtt { C h r o n o s } { - } 2$ and TimesFM-2.5 from the data generated by our experiments.

For Experiment 1, we find that both time series foundation models seem to smooth out the noise of the random walk process in their outputs, as shown in Figure 2. Figure 3 and 4 reveal that Chronos- $\mathbf { \nabla } \cdot 2 \mathit { \Omega } ^ { \prime } \mathbf { s }$ parameter statistic distribution is more closely aligned with the distribution of $\delta ( \mathbf { y } )$ . Figure 3 also shows that TimesFM-2.5 underestimates the magnitude of the drift parameter consistently for positive and negative values compared to Chronos-2. For the random walk with drift generator and the given range of interventions and noise realizations, Chronos-2 appears better at preserving the drift of an input time series than TimesFM–2 . 5. The trajectory plot in Figure 5 shows that both models' output flatlines the AR(1)-process in Experiment 2 across all interventions of the displayed noise realization. For all interventions except $\mathrm { d o } ( \beta = 0 . 9 8 )$ , both models’ output is close to zero. Figure 6 shows that both models produce high $\hat { \beta }$ values, even for low $\beta$ values. $\operatorname { A s } \beta$ values increase, the differences between the distribution of $\delta ( \mathbf { y } )$ and the respective distributions of $\delta ( M ( \mathbf { y } ) )$ of Chronos-2 and TimesFM-2.5 decrease. Figure 6 and 7 indicate that both models are biased towards consistently overestimating autocorrelation of lag one within our intervention range.

In Experiment 3, both models preserve the wavelength of the harmonic oscillator across the full range of interventions as shown in Figures 8, 9 and 10. For our intervention range, we did not find a value for which any of two the models estimated wavelength parameter statistic $\delta ( M ( \mathbf { y } ) )$ does not align with the intervention parameter λ and the trajectory parameter statistic $\delta ( \mathbf { y } )$ , with δ being λ. Figure 8 also shows that some amplitude values are not exactly preserved, as they show small variations along interventions.

Experiment 4 indicates failure modes for both models in preserving the regime switch pattern for some interventions. Figure 11 shows that Chronos-2 and TimesFM-2.5 fail to preserve regimes for interventions where $\tau \geq 5 0$ and $\tau \geq$ 25, respectively. Figure 13 shows that while TimesFM-2 . 5 maintains some noise realizations on the ideal line for a given intervention, Chronos-2 seems to deviate less. Figure 12 also seems to confirm these insights.

![](images/8cac06e4db73ce83265a016f7ada9d14b72ef20243fb1818bc10755c08e1bfcf.jpg)  
Figure 2: Trajectories across intervention sweep for a single noise realization in Experiment 1. Both time series foundation models seem to preserve the drift while removing noise. Towards the end of the trajectory, TimesFM-2 . 5 seems to pull values towards zero.

![](images/2541e8e523dc87938bdf37852cf9a43f883d25068eadeef9741826f11f81ecad.jpg)  
Figure 3: Distributions of parameter statistics for trajectory and model outputs across noise realizations in Experiment 1 $( n = 5 0 )$ . TimesFM-2.5 displays lower magnitudes for drift than the trajectory.

![](images/1626ac3bb70fb4c482670bbae403bc0fc522419a3a79e076b7eed7bf7ef74df4.jpg)  
Figure 4: Comparison of model parameter statistic $\delta ( M ( \mathbf { y } ) )$ against trajectory parameter statistic $\delta ( \mathbf { y } )$ for Experiment 1 $( n = 6 \times 5 0 = 3 0 0$ per model). Both models center around the idea line, while TimesFM-2 . 5 shows bias towards zero.

![](images/f6f96ad0d5fe12b7f1ea6e6496a39337130866205b4474c00fb0cc5ba829dba2.jpg)  
Figure 5: Trajectories across intervention sweep for a single noise realization in Experiment 2. Trajectories collapse close to flat lines for both time series foundation models.

![](images/203fe14ac64f09bb43f4236ed279acfd740252a95f0ba12dca75a3a193b06d90.jpg)  
Figure 6: Distributions of parameter statistics for trajectory and model outputs across noise realizations in Experiment $2 \ ( n \ = \ 5 0 )$ . Both time series foundation models overestimate $\beta$ for lower values, except for $\mathrm { d o } ( \beta =$ $- 0 . 5 )$ , where $\mathtt { C h r o n o s } { - } 2$ seems to estimate $\beta$ better than TimesFM-2.5

![](images/9df4ae6c996c32c2512b4971dc0fc2b93185ef53a164318aceac69b5f859cb0a.jpg)  
Figure 7: Comparison of model parameter statistic $\delta ( M ( \mathbf { y } ) )$ against trajectory parameter statistic $\delta ( \mathbf { y } )$ for Experiment 2 $( n = 6 \times 5 0 = 3 0 0$ per model). Both time series foundation models seem to produce outputs with a higher $\hat { \beta }$ than in the input trajectory for $0 < \beta < 0 . 8 5$

![](images/fb60e1f1c9cd08fb02d824d37accb1b5c2e020d79bf13fce956125357f83670a.jpg)  
Figure 8: Trajectories across intervention sweep for a single noise realization in Experiment 3. Both time series foundation models seem to preserve wavelength across interventions.

![](images/ba4f68dbaa4aaa068042fe93f830fc57407d2805309e666e00de9ddf8b6d3af8.jpg)  
Figure 9: Distributions of parameter statistics for trajectory and model outputs across noise realizations in Experiment 3 $( n = 5 0 )$ . Both models' distributions align with the trajectory distribution and display no variance.

![](images/5c554abb2e8062816608ee79535f32af016a925e7662d59fd416014058d03812.jpg)  
Figure 10: Comparison of model parameter statistic $\delta ( M ( \mathbf { y } ) )$ against trajectory parameter statistic $\delta ( \mathbf { y } )$ for Experiment 3 $( n = 6 \times 5 0 = 3 0 0$ per model). Both models show ideal alignment with no visible deviations.

For the energy-release generator in Experiment 5, Figure 14shows that Chronos-2 preserves the threshold pattern across the full range of interventions while TimesFM-2 . 5 seems to start smoothing at $\kappa \geq 2 5$ and loses pattern completely for $\kappa ~ \geq ~ 5 0$ Figure 16 shows that for $\hat { \kappa } ~ \approx ~ 5$ and ê ≈ 10 both models' $\delta ( M ( \mathbf { y } ) )$ aligns with the trajectory $r _ { \textbf { S } } \delta ( \mathbf { y } )$ . For ê ≈ 25 dispersion begins, while at $\hat { \kappa } ~ \approx ~ 5 0$ both models begin to underestimate κ. With ê ≈ 75 both models seem less biased toward underestimation but express higher variance in $\delta ( M ( \mathbf { y } ) )$ values and with ê ≈ 100, TimesFM-2.5 underestimates κ noticably more than Chronos-2. Figure 15also suggests that Chronos-2 is better at preserving the threshold pattern across the range of interventions than TimesFM-2 .5.

Experiment 6 tests how well each model preservers the Hurst exponent of a fractional Brownian motion generator. The model output $M ( \mathbf { y } )$ displayed in Figure 17 suggests that both models struggle with preserving the jaggedness for interventions where $H < 0 . 5$ and both smooth out the input trajectory y across interventions. Figure 19 confirms this, as it shows that both models are biased towards overestimating $\hat { H }$ , therefore being biased towards persistence of trends. The figure also reveals that Chronos-2 underestimates $\hat { H }$ more for lower values of H than TimesFM-2 .5. TimesFM-2.5 only shows cases of underestimation for higher values of H. Figure 18 is also aligned with these findings, as distributions of both models overestimate H relative to the trajectory distribution, while Chronos-2 does so less than TimesFM-2.5.

![](images/94b7157404170fc1526917d927593e0dbf68f9ccd789fc4040d17ebf7315bd99.jpg)  
Figure 11: Trajectories across intervention sweep for a single noise realization in Experiment 4. Chronos-2 and TimesFM-2.5 fail to preserve regimes for interventions where $\tau \geq 5 0$ and $\tau \geq 2 5 ,$ , respectively.

![](images/7f70462e3e05a7b40585f2a7707f3c098b396836bb28fb577233d304cb2df178.jpg)  
Figure 12: Distributions of parameter statistics for trajectory and model outputs across noise realizations in Experiment 4 $( n = 5 0 )$ . TimesFM-2 . 5 exhibits stronger variance and deviation from trajectory parameter statistic than Chronos-2 for higher values of τ. Further TimesFM-2.5 seems to overestimate τ while Chronos-2 seems to underestimate it with increasing values of $\tau .$

![](images/bd7bb65c76c6a135607c45a74bf0d629ee20e326453d4506fb1ab034a3e09d00.jpg)  
Figure 13: Comparison of model parameter statistic $\delta ( M ( \mathbf { y } ) )$ against trajectory parameter statistic $\delta ( \mathbf { y } )$ for Experiment 4 $( n = 6 \times 5 0 = 3 0 0$ per model). TimesFM-2.5 seems to produce outputs with higher î than the trajectories for $\tau \geq 5 0$ , while Chronos-2 produces outputs with lower î for $\tau \geq 5 0$

![](images/5227deabe998062f1d5c0ca5488e94fe1a2407321bf4ce80b0e82729516c5e95.jpg)  
Figure 14: Trajectories across intervention sweep for a single noise realization in Experiment 5. Chronos-2 preserves the threshold pattern across the full range of interventions while TimesFM-2.5 seems to start smoothing at $\kappa \geq 2 5$ and loses pattern completely for $\kappa \geq 5 0$

![](images/e51fa4537b2b99e9a6004205619a61c43e29e8ec694da7a738b01427470567eb.jpg)  
Figure 15: Distributions of parameter statistics for trajectory and model outputs across noise realizations in Experiment $5 ~ ( n ~ = ~ 5 0 )$ $\mathtt { C h r o n o s } { - } 2$ is better at preserving the threshold pattern across the range of interventions than ${ \mathrm { T i m e s F M } } - 2 . 5$

![](images/4374b2ac9bf998d35599fe25dcfce26c046f6140f5bc831e5cb80deb0af34120.jpg)  
Figure 16: Comparison of model parameter statistic $\delta ( M ( \mathbf { y } ) )$ against trajectory parameter statistic $\delta ( \mathbf { y } )$ for Experiment 5 $( n = 6 \times 5 0 = 3 0 0$ per model). Chronos-2 displays better aligment than TimesFM-2.5.

![](images/3ba9e1d085e8880460015208eb8213af7c15d35556f76b95d4ad0ea4ca091a0a.jpg)  
Figure 17: Trajectories across intervention sweep for a single noise realization in Experiment 6. Both models struggle with preserving the jaggedness for interventions where $H < 0 . 5$ and both smooth out the input trajectory y across interventions.

![](images/be46a61fc8d86b240f0ff32ee2877e06cadac4e7867868e7fef8a7cf2264e7e9.jpg)  
Figure 18: Distributions of parameter statistics for trajectory and model outputs across noise realizations in Experiment $6 ( n = 5 0 )$ . Distributions of both models overestimate H relative to the trajectory distribution, while Chronos-2 does so less than TimesFM-2.5.

![](images/dc34dcf0d13e263e25bd8952d1a0652ecbddbf0a81a29339b6a61885cdd13786.jpg)  
Figure 19: Comparison of model parameter statistic $\delta ( M ( \mathbf { y } ) )$ against trajectory parameter statistic $\delta ( \mathbf { y } )$ for Experiment 6 $( n = 6 \times 5 0 = 3 0 0$ per model). Both models are biased towards overestimating H.

In summary, through the application of our causal analysis framework for time series foundation models, we were able to identify stable configurations, as well several biases and failure modes across our experiments. We find that Chronos-2 and TimesFM-2 . 5 preserved generator characteristics for a random walk with drift and a harmonic oscillator, within the experimental configurations. We further identify a potential bias towards overestimating persistence in time series in the form of autocorrelation or long-term dependence for both models. We also demonstrate that both models fail to preserve regime switch pattern within our range of interventions and that TimesFM-2.5 deviates strongly from the threshold pattern in Experiment 5 for higher threshold values while Chronos-2 only deviates weakly. In the next section, we compare our findings to previous work, consider explanations for our findings and discuss implications of our findings for model development and application.

## 5 Discussion

With the findings from the previous section, several questions arise. First, do we see our findings reflected in benchmark evaluations of Chronos-2 and TimesFM-2.5?Second, do we find evidence in the training and architecture that could explain our findings? And lastly, what are the implications of our findings for time series foundation model development and application?

To address whether our findings are reflected in previous benchmark comparisons, we review reported performance results in Ansari et al. (2025) and Das et al. (2024). Ansari et al. (2025), introducing Chronos-2, compares the model against competitor models on fevbench, GIFT-Eval and Chronos Benchmark II for univariate tasks. The study reports Average Win Rate and Skill Score per model as aggregate metrics for each benchmark without detailing performance on individual time series used in the respective benchmark. This limits our ability to draw direct comparisons between our findings and the benchmark results reported. Ansari et al. (2025) also apply Chronos-2 to two case studies. First, the Rossmann sales forecasting case study shows a strong smoothing of jagged patterns for the univariate forecasting, in line with our findings in Experiment 1, 2 and 6. Second, for the univariate energy price forecasting case study, the reported forecast aligns with the somewhat cyclical pattern of the time series. We find this in line with our results in Experiment 3. For both case studies, Ansari et al. (2025) only report one sample and no quantitative evaluation, therefore we should view our comparison as showing an absence of contradictory findings rather than a proof for generalizability of our in vitro findings to the in vivo case studies in Ansari et al. (2025).

Das et al. (2024) list performance for all datasets used in both the Darts and Monash benchmarks for TimesFM-2.5 and competitor models. For the Darts benchmark, TimesFM-2.5 shows no clear domination while it does so for Monash with mixed rankings across datasets in both benchmarks. We do not find a clear domination for the listed datasets where we might assume trend or cyclical patterns neither in the Darts nor the Monash benchmark. Even if we are confident that a dataset exhibits a certain time series pattern, the comparison along competitor models and not along datasets does not allow us to assess whether findings in our experiments transfer to real-world datasets. For example, we would expect that TimesFM-2 .5 shows lower scale-adjusted error on a dataset with cyclical pattern or drift than for one with regime switches or a threshold pattern. Our review of both studies therefore suggest that their findings do not seem to contradict our experimental results, yet we also find limited comparability. Preferably, we would select realworld datasets from data-generating processes with assumed patterns and validate our in vitro findings against these.

To answer our second question, whether we find evidence explaining our results, we inspect the training data used for both time series foundation models. Chronos-2is trained on 23 different datasets, some of them synthetically generated (Ansari et al., 2025). Five of the datasets are under the category of energy and six under transportation. If we assume that these datasets possess cyclical patterns due physical processes such as day-night cycles or workdays and weekend, then almost half of the datasets represent this time series pattern. This emphasis on cyclical patterns in the training data would align with our findings of preservation of wavelength in Experiment 3. To create the univariate synthetic dataset, Ansari et al. (2025) uses TSI to create combinations of trend, seasonality and irregularity. We find that this is in line with our results for Experiments 1 and 3 in which Chronos-2 demonstrated preservation of drift in a random walk and wavelength in a harmonic oscillator across the full range of interventions. For multivariate synthetic data, Ansari et al. (2025) report using autoregressive models, exponential smoothing models, TSI and KernelSynth. Taken together, the training data might be biased towards trend and cyclical patterns which would explain our findings in Experiment 1 and 3.

Das et al. (2024) list four out of the nineteen training datasets as trend. The categories eletricity, traffic and weather are represented by one dataset each and plausibly cyclical patterns. Das et al. (2024) also mention that majority of their datasets exhibit a periodic pattern. They further report that the synthetic data consists of piece-wise linear trends, ARMA processes and seasonal patterns. The provided illustrative examples also heavily lean towards cyclical time series. Similar to Ansari et al. (2025), we also find evidence in the training data that indicates a bias towards trend and cyclical patterns. The therefore lesser representation of other time series patterns could explain the observed failure modes in Experiment 4 and 5 as well as the bias towards overestimating persistence for lower persistence values of β and H, suggested by the findings of Experiment 2 and 6.

Having discussed our findings within the context of the original works, we now may concern ourselves with the implications of these findings. We can group them into (i) implications for future model development and (ii) implications for model selection for applications. For model development (i), our empirical findings and review of original works indicate that more diverse training data might be beneficial, specifically targeting the identified failure modes and biases. To what extent more diverse training data is able to remedy those might only be found out through experimentation, as model architecture, other factors, or a combination of these factors might be the actual cause. Ansari et al. (2025) also show that a version of Chronos-2 that was only trained on synthetic data performed close to the version trained with real and synthetic datasets and suggest that synthetic-only training might be a viable pretraining strategy for time series foundation models. Building on this evidence, we might replace real-world training datasets that have fixed empiricial distributions with parameterized synthetic generators that allow us to pretrain time series foundation models across a wide range of generator configurations, ensuring safe-use boundaries. For model selection (ii), our findings also indicate differences between Chronos-2 and TimesFM-2.5 for different applications. While the discussed benchmark evaluations give comparisons of competitor models across different datasets, our analysis demonstrates how a given model performs across a range of potential realizations of a given time series pattern. This additional insight might be particularly relevant for practitioners with domain knowledge of their application, allowing them to specify the kind of time series patterns they expect in their application. In this situation understanding how time series foundation models perform against a range of pattern realizations might be more insightful for model selection than how the models rank across a range of different datasets exhibiting a diverse range of, partially unknown, time series patterns. If we want to predict foot traffic and assume a cyclical day and night pattern as well as weekday and weekend pattern and need to choose between Chronos-2 and TimesFM-2.5, we might be more interested in which model preserves cyclical structure such as wavelength rather than understanding how both models perform across a wide array of datasets from different applications. We summarize our recommendations for model selection by time series pattern in Table 2.

Table 2: Comparison of model behaviors, failure modes, and model selection recommendations across patterns.
<table><tr><td>Pattern</td><td>Chronos-2 Behavior</td><td>Behavior</td><td>TimesFM-2.5 Recommendation</td></tr><tr><td>Random Walk</td><td>Preserves closer to trajec- (bias tory)</td><td>drift Underestimates magnitude (β drift magnitude toward zero)</td><td>Chronos-2</td></tr><tr><td>AR(1)</td><td>to line</td><td>Output collapses Output collapses to line</td><td>Avoid</td></tr><tr><td>Harmonic Oscil- Preserves lator</td><td>lengths</td><td>λ; Preserves λ; cuts amplitude cuts amplitude for some wave- for some wave-</td><td>Tie</td></tr><tr><td>Regime Switch</td><td>Preserves regimes dwell time up to at τ = 25 τ = 50</td><td>lengths Deviates earlier; and leaves ideal line</td><td>Chronos-2</td></tr><tr><td>Energy-release</td><td>Preserves pattern Smoothes across underestimates strong at κ = 100</td><td>at range; 25 &lt; κ &lt; 50; under- estimation at κ = 100</td><td>Chronos-2</td></tr><tr><td>fBm</td><td>Overestimates Overestimates H; strong Ê; H &lt; 0.5</td><td>strong smoothing for smoothing for H &lt; 0.5</td><td>Avoid</td></tr></table>

## 6 Conclusion

We conclude this study by summarizing our four contributions, highlighting limitations and suggesting further research directions. To the best of our knowledge this is the first study that formalizes a causal analysis framework for time series foundation models. We identified dose-response relationships for Chronos-2 and TimesFM-2.5 across six generators and five interventions each. We identify a potential biases towards overestimating persistence in Experiment 2 and 6 for both models and showcase idiosyncratic failure modes for each of them. Lastly we make concrete recommendations for application-specific model selection based on our experimental results.

Our findings are limited to the specific experimental configurations and models analyzed. We used a fixed set of generators and intervention parameter per generator, sweeped a limited range of values, and used only one noise scale per experiment. We further only evaluated the preservation of parameter statistics on a fixed window of 200 time steps and used nominal parameter statistic values and did not relate them to the window length. For example, it is not clear whether the failure mode in Experiment 4 is caused by the nominal dwell time or the number of observable regime switches within the input trajectory.

While the limitations mentioned outline the epistemic boundaries of this study, they also signal opportunities for further research. Future work could extend experiments to more generators and time series foundation models. We could create domain specific experiments to analyse model suitability for domains such as climate, finance or sales. We could also adapt the causal analysis framework towards analyzing relationships between structure and behavior, to understand how different components of a time series foundation model affect its behavior in inference. Future model development efforts could focus on mitigating the biases and failure modes we identified while other research efforts could be directed toward in vivo validation of our findings through real-world datasets. Together, the structured in vitro causal analysis of time series foundation models might improve their capabilities as wells as inform their application and regulation.

## Code and Data Availability

The codebase for this study can be found at https : / / gi thub.com/MSCA-DN-Digital-Finance/tsfm\_ causal\_analysis.

The dataset containing the experimental results can be found at https://zenodo.org/records/22082090

## Author Contributions

Mathis Jander: Conceptualization, Methodology, Software, Investigation, Formal Analysis, Data Curation, Writing – Original Draft, Visualization.

Wouter van Heeswijk: Conceptualization, Funding Acquisition, Project Administration, Supervision, Writing – Review & Editing.

Martijn Mes: Conceptualization, Supervision, Writing – Review & Editing.

## Acknowledgments

The views expressed in this work are those of the authors and do not necessarily reflect those of the European Central Bank or the Eurosystem.

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.

This project has received funding from the Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101119635

![](images/885b8b0d3aef3603ea8096561abd2fb21890a47dcf5b93d3c7c186fb4700e27b.jpg)

Funded by the European Union

## References

Ansari, A. F., Shchur, O., Küken, J., Auer, A., Han, B., Mercado, P., Rangapuram, S. S., Shen, H., Stella, L., Zhang, X., Goswami, M., Kapoor, S., Maddix, D. C., Guerron, P., Hu, T., Yin, J., Erickson, N., Desai, P. M., Wang, H., Rangwala, H., Karypis, G., Wang, Y., and Bohlke-Schneider, M. (2025). Chronos-2: From univariate to universal forecasting. (arXiv:2510.15821).

Braiek, H. B. and Khomh, F. (2025). Chapter 3 - machine

learning robustness: a primer. In Lorenzi, M. and Zuluaga, M. A., editors, Trustworthy AI in Medical Imaging, The MICCAI Society book Series, pages 37–71. Academic Press.

Das, A., Kong, W., Sen, R., and Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting. (arXiv:2310.10688).

European Medicines Agency (2009). Authorisation of medicines. https://www.ema.europa.eu/en/about-us/whatwe-do/authorisation-medicines.

Food and Drug Administration (2020). The drug development process. https://www.fda.gov/patients/learn-aboutdrug-and-device-approvals/drug-development-process.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5):1189–1232.

Goodfellow, I. J., Shlens, J., and Szegedy, C. (2015). Explaining and harnessing adversarial examples. (arXiv:1412.6572).

Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica, 37(3):424–438.

Hendrycks, D. and Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. (arXiv:1903.12261).

International Organization for Standardization (2018). ISO 26262: Road vehicles — functional safety. https://www.iso.org/publication/PUB200262.html.

Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., Hu, W., Yasunaga, M., Phillips, R. L., Gao, I., Lee, T., David, E., Stavness, I., Guo, W., Earnshaw, B., Haque, I., Beery, S. M., Leskovec, J., Kundaje, A., Pierson, E., Levine, S., Finn, C., and Liang, P. (2021). Wilds: A benchmark of in-the-wild distribution shifts. In Proceedings of the 38th International Conference on Machine Learning, pages 5637–5664. PMLR.

Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Pearl, J. (2010). An introduction to causal inference. The International Journal of Biostatistics, 6(2):7.

Peter, M. (2022). The essentials of the new car development process + free v-model. http://www.magna.com/insideautomotive/concept-creation/new-car-developmentprocess.

Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. (2019). Do imagenet classifiers generalize to imagenet? In Proceedings of the 36th International Conference on Machine Learning, pages 5389–5400. PMLR.

Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). "why should i trust you?": Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, pages 1135–1144. Association for Computing Machinery.

Seabold, S. and Perktold, J. (2010). statsmodels: Econometric and statistical modeling with python. In Proceedings of the 9th Python in Science Conference, pages 92–96.

Truong, C., Oudre, L., and Vayatis, N. (2020). Selective review of offline change point detection methods. Signal Processing, 167:107299.

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J. Jones, E., Kern, R., Larson, E., Carey, C. J., Polat, İ., Feng, Y., Moore, E. W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E. A., Harris, C. R., Archibald, A. M., Ribeiro, A. H., Pedregosa, F., van Mulbregt, P., and SciPy 1.0 Contributors (2020). SciPy 1.0: Fundamental algorithms for scientific computing in python. Nature Methods, 17:261–272.

Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., and Sahoo, D. (2024). Unified training of universal time series forecasting transformers. (arXiv:2402.02592).

Zollino, P., Ludewig, C., and Aschhoff, R. (2025). Test center complete: Volkswagen group now able to fully develop and validate products in China for China. Volkswagen Group.