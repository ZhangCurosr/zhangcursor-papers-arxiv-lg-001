# Odds-Structured Distributional Regression for Interpretable Time-to-Event Prediction

Jakob Versnjak<sup>a,b,f</sup>, Manuel Welte<sup>a,c,f</sup>, Titus Kuehne<sup>b,d,e,f</sup>, Marcus Kelm<sup>b,d,e,f</sup>, Grégoire Montavon<sup>a,c,f,∗</sup>

<sup>a</sup>BIFOLD – Berlin Institute for the Foundations of Learning and Data, Berlin, Germany <sup>b</sup>Deutsches Herzzentrum der Charité, Institute of Computer-assisted Cardiovascular Medicine, Augustenburger Platz 1, 13353 Berlin, Germany

<sup>c</sup>Deutsches Herzzentrum der Charité, Institute for AI in Medicine, Invalidenstraße 120/121, 10115 Berlin, Germany

<sup>d</sup>Department of Congenital Heart Disease – Pediatric Cardiology, Deutsches Herzzentrum der Charité, Augustenburger Platz 1, Berlin 13353, Germany

<sup>e</sup>DZHK (German Centre for Cardiovascular Research), Partner Site Berlin, Potsdamer Str. 58, Berlin 10785, Germany

<sup>f</sup>Charité – Universitätsmedizin Berlin, Corporate Member of Freie Universität Berlin and Humboldt-Universität zu Berlin, Charitéplatz 1, Berlin 10117, Germany

## Abstract

Time-to-event prediction is fundamentally challenged by partial observability and aleatoric uncertainty. In clinical medicine, for instance, the timing of disease progression or acute readmission is contingent on highly dynamic and largely latent health trajectories. Rather than forcing a deterministic point forecast, distributional regression ofers a principled means to address this complexity by predicting a probability distribution over time. However, explaining these distributional predictions presents a key obstacle: raw distribution parameters output by the model are abstract mathematical quantities that do not align well with domain concepts. To gain interpretability, we propose a novel paradigm where output parameters are designed to be additive in log-odds—a widely established measure in clinical risk modeling. Our approach, called ODDS-DR, moves beyond technical convenience to deliver clear domain alignment as well as the layer-wise computational structure required for robust, time-dependent feature attribution. Validated across two real-world clinical cohorts, our framework matches state-of-the-art predictive performance while resolving distributional outputs into interpretable risk markers across flexible prognostic horizons.

Keywords: Distributional regression, Explainable artificial intelligence, Time-to-event modeling, Cumulative log-odds, Clinical risk factors

## 1. Introduction

In real-world applications—and particularly within clinical medicine—predicting time-to-event outcomes is fundamentally challenged by latent physiological states, sparse monitoring schedules, and the inherent variability of downstream clinical actions and care pathways. Consequently, critical endpoints—such as acute readmission, disease progression, therapeutic response, or mortality— exhibit profound temporal stochasticity. Capturing the full complexity of temporal uncertainty requires more than summarizing it via error bars or rigid hazard ratios; it demands modeling the event’s full probabilistic distribution over time.

Distributional regression [1, 2, 3] provides the flexibility to capture this temporal complexity. Unlike classical Cox-based semi-parametric estimators [4, 5]—which enforce time-invariant proportional hazard ratios—distributional regression parameterizes the complete conditional density function via distinct location, scale, and shape parameters. In practice, however, outputting an accurate predictive distribution is rarely suficient. Domain experts value the factors driving a prediction as much as the forecast itself. This emphasis on interpretability is ubiquitous in clinical fields, where identifying actionable risk markers is often as critical as the risk estimate itself [6, 7, 8]. In time-to-event tasks, this means looking beyond event timing alone to disentangle which factors operate across diferent prognostic horizons—separating long-term baseline predispositions (e.g., genetics) from short-term metabolic or lifestyle disruptions.

However, a key methodological obstacle limits the practical utility of distributional regression models. The parameters produced at their output are inherently technical and specific to the chosen distribution family. Identifying features that drive these raw parameters risks severe misalignment with domain experts. Model-agnostic explainers [9, 6] can in principle access and attribute interpretable readouts from the predicted distributions post-hoc, but their sampling-based mechanisms are computationally demanding and brittle in the presence of nonlinearities.

In this paper, we address these limitations by redesigning distributional regression models to natively embed log-odds interpretability. Unlike bounded probabilities or family-specific distribution parameters, the log-odds maps evidence additively onto the real line, while unifying diverse distribution families under a single, comparable metric. For an event occurring at time T, the cumulative log-odds by time t is defined as:

$$
L ( t ) = \log \left( { \frac { P ( T \leq t ) } { P ( T > t ) } } \right) .\tag{1}
$$

We propose a novel reparameterization where the network outputs combine additively to reconstruct L(t) across arbitrary horizons t. We derive this additive structure for Gaussian, Logistic, and Log-Logistic families. In each case, it yields the layer-wise computational graph required for fast, robust propagation-based attribution methods such as Layer-wise Relevance Propagation (LRP) [10, 11], where explanations at any time horizon can be retrieved as a linear combination of only two explanation prototypes. Consequently, practitioners can query specific temporal horizons (e.g., “what factors drive this patient’s risk of surgical reintervention within six months?”) or track risk markers across time (e.g., “how does the TISS score influence mortality risk from the acute window into the long term?”) consistently and orders of magnitude faster than with model-agnostic explainers. Our framework, termed Odds-Structured Distributional Regression (ODDS-DR), is illustrated in Fig. 1.

We validate our approach across two real-world clinical cohorts: the public SUPPORT2 survival dataset and an institutional time-to-reintervention cohort of aortic coarctation (CoA). In both cases, our framework matches or exceeds the predictive performance of standard baselines (e.g., DeepSurv [5], DeepHit [12], Deep Survival Machines [13]) while uncovering risk factors that closely align with established clinical guidelines. Beyond achieving clinical alignment on conventional time ranges, we demonstrate our method’s capability to extend evidence-based risk assessment to arbitrary temporal horizons and individual patient profiles.

A observed time-to-event phenomena  
![](images/3be2682761b5d8d03c16622770629cd48474ff03a3af7a98972d362ba6fe5512.jpg)

C classical distributional regression (baseline)  
![](images/68616acec3a73ac27cb33fce7c1467541fa8fd6b69f0c4d0b57b7a1af040cf3f.jpg)

B modeled time-to-event phenomena  
![](images/804d5ee66923522a52b2ff0f873ac77b87e3ecfc4721dc43be2d579466a4af4e.jpg)

D ODDS-DR (ours)  
![](images/7d1c2b37873a36326a19439fb57798c6a3f691fc9546244499cc639d0a431eb0.jpg)  
Figure 1: Time-to-event modeling and proposed ODDS-DR framework. (A) Clinical data records both the occurrence of events and their temporal organization. (B) Distributional regression models this temporal dynamic by parameterizing continuous, patient-specific probability densities over time. (C) Classical baseline: Standard distributional regression predicts distribution-specific parameters θ(x) (e.g., shape and scale) that lack clinical interpretability and decomposability into input features. (D) ODDS-DR (ours): By parameterizing model representations to contribute linearly to event log-odds L(x, t)—a clinically established quantity—our framework enables exact linear attribution to intermediate representations, followed by propagation (e.g., via LRP) back to input clinical features.

## 2. Related work

Rigorously modeling, quantifying, and explaining time-to-event uncertainty intersects three active areas of literature. In Section 2.1, we review continuous distributional regression and heteroskedastic modeling. In Section 2.2, we contrast continuous parametric density formulations against conventional semi-parametric and discrete time-to-event paradigms. Finally, in Section 2.3, we examine existing feature attribution methods for uncertainty and time-to-event predictions. The advantages and limitations of the methods described below are summarized in Table 1.

## 2.1. Heteroskedastic Modeling and Distributional Regression

A first simple class of models that embeds uncertainty directly within the model structure is heteroskedastic regression, which adds to each model output a predictive spread that depends on the input [14]. Another class of models, distributional regression—formalized via Generalized Additive Models for Location, Scale, and Shape (GAMLSS) [2] and Mixture Density Networks [1]— generalizes this principle by simultaneously learning distinct functional dependencies for location, scale, and shape parameters to yield complete conditional densities. Modern semi-structured extensions integrate deep neural networks with structured additive predictors [15], enabling high-capacity representation learning while retaining transparent, parameter-specific additive components. In survival analysis, Deep Conditional Transformation Models (DCTMs) [16, 17] reparameterize distributions via Bernstein polynomials, which yields an additive decomposition of log-odds for the logistic distribution.

In contrast to these works, our ODDS-DR framework establishes the design of parameterizations that are log-odds-additive as its core principle, applicable to a broad range of distributions. To the best of our knowledge, ODDS-DR is also the first framework that exploits log-odds additive structures both for prediction and feature attribution.

## 2.2. Time-to-Event Frameworks

A significant body of work addresses uncertain outcomes specifically in time-to-event analysis, where Cox Proportional Hazards (PH) models [4] form the dominant paradigm. DeepSurv [5] provides a modern extension of this framework, parameterizing the log-hazard with a neural network to capture non-linear feature interactions, while retaining the original assumption of time-invariant relative hazards. Cox-Time [18] relaxes this constraint by providing time as an additional network input, allowing relative hazards to depend jointly on features and time. Because the network outputs an instantaneous hazard rate, obtaining survival curves at test time requires numerically integrating these rates over time. To keep this computation tractable, predictions are evaluated on a discretized time grid derived from a subset of the training data. Specifically designed for interpretability, architectures such as Cox-SE [19] and CoxNAM [20] compute sample-specific or additive feature contributions directly within the network while retaining the proportional hazards formulation. Departing from Cox-based formulations, another line of work partitions the continuous time axis into discrete intervals, using architectures such as DeepHit [12] or neural multi-task formulations [21, 22] to model non-linear, time-varying event probabilities across pre-specified bins.

More closely related to our ODDS-DR framework, prior work addresses time-to-event modeling through distributional regression [23, 24, 17]. Within this family of methods, Deep Survival Machines (DSM) [13] allows for the expression of fine-grained time-to-event profiles by letting neural networks output complex parametric forms (such as mixtures of Weibull or Log-Normal distributions). In contrast to Cox-based approaches, the distributional regression framework provides a variety of advantages: the likelihood factorizes into independent per-sample contributions, enabling direct applicability of fast mini-batch-based training. Furthermore, direct access to the predicted distribution at the output simplifies multi-horizon querying and attribution.

Our proposed framework, ODDS-DR, inherits the key advantages of distributional regression, particularly its expressivity and integration into neural network training pipelines. Crucially, what distinguishes ODDS-DR from existing distributional regression methods is its focus on native explainability. Specifically, by parameterizing predicted distributions to be additive in log-odds, our framework yields feature attributions that are both inherently interpretable and structured across entire time horizons.

## 2.3. Explainable AI Frameworks for Probabilistic Outcomes

While Explainable AI frameworks are well established for scalar point predictions [9, 10, 25], fewer works have addressed the question of how to attribute continuous probabilistic outcomes. Some recent eforts focus primarily on explaining dispersion statistics, such as ensemble variance [26] or heteroskedastic Gaussian spread [27]. A series of recent works have addressed the explainability of probability-linked quantities as they occur in time-to-event models. Specific to Cox-based models, Cox-SE [19] and CoxNAM [20] parameterize instance-level feature contributions within the loghazard ratio to facilitate the explanation of this quantity.

Leveraging post-hoc interpretability methods that can readily apply to any black-box model— including distributional regression—the authors of [28] adapt classical gradient-based explanation techniques to survival, enabling explanations resolved across both input features and time. Perturbation-based estimators have similarly been extended over time: SurvSHAP(t) [29] computes time-resolved Shapley values across the prediction horizon, SurvLIME(t) [30] fits local Cox surrogates, and SurvBeNIM [31] constructs non-parametric local survival approximations. More recently, SurvFD and SurvSHAP-IQ [32] decompose time-dependent and time-independent feature interactions.

<table><tr><td>Property</td><td>Classifier</td><td>[4]</td><td>[5]</td><td>[18]</td><td>[12]</td><td>[13]</td><td>[20]</td><td>Cox DeepSurv Cox-Time DeepHit DSM CoxNAM ODDS-DR (ours)</td></tr><tr><td>Modeling capability</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Continuous event time</td><td>X</td><td></td><td></td><td></td><td> $\pmb { \chi } ^ { ( \mathbf { b } ) }$ </td><td></td><td></td><td></td></tr><tr><td>Right-censoring support</td><td>X</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Time-varying effects</td><td>X</td><td>X</td><td>X</td><td></td><td></td><td></td><td>X</td><td></td></tr><tr><td>Non-linear representations</td><td></td><td>X</td><td></td><td></td><td></td><td></td><td>V</td><td></td></tr><tr><td>Interpretability</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Multi-horizon attribution</td><td>X</td><td>X</td><td>X</td><td></td><td></td><td></td><td>X</td><td></td></tr><tr><td>Time-parametric attribution(a)</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>Optimization &amp; inference</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mini-batch loss decomposition</td><td></td><td>X</td><td>X</td><td></td><td></td><td></td><td>X</td><td></td></tr><tr><td>Closed-form risk inference</td><td></td><td></td><td></td><td>x(c)</td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Systematic comparison of ODDS-DR with existing time-to-event frameworks. Symbols: ✓ supported natively; ✗ not supported. Superscripts denote methodological specifics: <sup>(a)</sup> Attributions are continuous, closed-form functions of time (see definition in Section 3); <sup>(b)</sup> Relies on a discretized time grid rather than a continuous-time formulation; <sup>(c)</sup> Requires numerical integration over a discrete grid rather than an analytical closed-form evaluation.

ODDS-DR departs from these existing frameworks by addressing explainability natively within the model architecture, specifically by outputting parametric forms that are additive in the logodds. This design enables the application of propagation-based attribution methods such as LRP [10, 11], which are significantly faster to compute and more robust than post-hoc methods reliant on repeated sampling of nonlinear functions. In particular, ODDS-DR’s structure allows generating explanations resolved across an entire time horizon from only two prototypical attributions.

## 3. Odds-Structured Distributional Regression (ODDS-DR)

Standard neural distributional regression predicts the parameters of a continuous conditional density $p ( T \mid \pmb { x } )$ —such as location and scale $( \mu , \sigma )$ for Gaussian or Logistic models, or $( \alpha , \beta )$ for Loglogistic distributions. However, aligning network outputs with these raw mathematical parameters is largely a technical convenience that lacks principled grounding in either modeling or interpretability. Our proposed method, ODDS-DR, introduces a novel paradigm where the model outputs additively predict cumulative log-odds across continuous prognostic horizons.

Let $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ denote an input feature vector, $T \in \mathbb { R } ^ { + }$ a continuous target variable measuring time to event, and t a prognostic horizon. Our framework designs the model outputs $z _ { j } ( \pmb { x } ) \in \mathbb { R }$ so that they decompose the cumulative log-odds as:

$$
L ( \mathbf { \boldsymbol { x } } , t ) = \log { \left( \frac { P ( T \leq t \mid \mathbf { \boldsymbol { x } } ) } { P ( T > t \mid \mathbf { \boldsymbol { x } } ) } \right) }\tag{2}
$$

$$
= \sum _ { j } \lambda _ { j } ( t ) z _ { j } ( { \pmb x } ) ,\tag{3}
$$

where the weighting coeficients $\lambda _ { j } ( t )$ depend solely on the horizon t. This linear parameterization of ODDS-DR ofers two key advantages over classical parameterizations and broader time-to-event methods.

Property 1: Epistemic Inductive Bias. The homogeneous linear relation between model outputs and log-odds ensures that uncertainty is naturally preserved in two fundamental modes: feature missingness and out-of-domain prediction. In both regimes, a properly regularized model tends to produce outputs close to zero (e.g., owing to the positive homogeneous tendency of neural networks or to attenuated responses outside the data support). These behaviors yield the natural implication chains:

$$
\begin{array} { r l } { x \approx \mathbf { 0 } \implies z \approx \mathbf { 0 } \implies L \approx 0 } & { { } ( \mathrm { f e a t u r e ~ m i s s i n g n e s s } ) , } \\ { p ( \mathbf { x } ) \approx 0 \implies z \approx \mathbf { 0 } \implies L \approx 0 } & { { } ( \mathrm { d a t a ~ o u t l i e r n e s s } ) . } \end{array}
$$

This conservative inductive bias is essential in high-stakes domains—such as clinical risk stratification on underrepresented patient subgroups—as it prevents the model from generating spurious, unsupported certainty.

Property 2: Fast Time-Parametric Attribution. Leveraging the common linearity axiom of explanations, attributing the log-odds to the input features can be reduced to explaining individual output neurons:

$$
\mathcal { E } \Big \{ \sum _ { j } \lambda _ { j } ( t ) z _ { j } ( \pmb { x } ) \Big \} = \sum _ { j } \lambda _ { j } ( t ) \mathcal { E } \big \{ z _ { j } ( \pmb { x } ) \big \} .\tag{4}
$$

Specifically, calculating $\mathcal { E } \{ z _ { j } ( \pmb { x } ) \}$ is a staple in Explainable AI, with methods such as LRP requiring only one backward pass. Consequently, attributing log-odds across an entire continuum of prognostic horizons demands only as many backward passes as there are output neurons—two for the two-parameter distributions considered in this work—which is orders of magnitude more eficient than model-agnostic approaches. Furthermore, Eq. (4) reveals an intrinsic low-rank structure in multi-horizon attribution, providing a compact basis to analyze how risk markers vary across prognostic horizons.

In the following subsections, we derive the explicit mappings relating network outputs, log-odds, and the standard parameters of common distribution families used to predict events under diverse uncertainty profiles.

## 3.1. ODDS-DR for Low-Uncertainty Events

Regularly recurring physiological events $( \mathrm { e . g . }$ , cardiac cycles, respiratory intervals, or sensory evoked responses), where deviations from a baseline rhythm indicate underlying pathology, can be categorized as low-uncertainty events. Tracking the log-odds of event occurrence relative to a reference duration holds clinical value in deciding whether an event is pathologically delayed or imminent. Bell-shaped distributions—including Gaussian, Laplace, or Logistic distributions— are a natural model of deviations around an overall predictable schedule. The log-odds of such distributions can be expressed as $L = g ( { ( t - \mu ) } / { \sigma } )$ , where g is typically a monotonically increasing odd function. For the Gaussian case, it is given by $g ( \xi ) ~ = ~ 2 \operatorname { t a n h } ^ { - 1 } ( \operatorname { e r f } ( \xi / \sqrt { 2 } ) )$ , whereas for the heavier-tailed Logistic distribution, it is simply the identity function $g ( \xi ) = \xi$ (derivations in Appendix B.1). This expression can be Taylor-expanded in the vicinity of some baseline $t _ { 0 }$ as:

![](images/3c8a30585550c6ec5aa82979420aae90f532a8115979afe503a1d3b6c37e5697.jpg)

![](images/97ae0d543a2c3bb338949d1ea8e970c1d369a647f5355608029bb1dc086d1139.jpg)  
Figure 2: Comparison of parameterization geometries for a Gaussian distribution. Left (Standard parameterization): Classical formulation with location $\mu = z _ { \mu }$ and scale $\sigma = \mathrm { s o f t p l u s } ( z _ { \sigma } )$ , where variance changes monotonically along the vertical axis independently of location. Right (ODDS-DR, log-odds-additive parameterization): Our reparameterization $\left( z _ { 1 } , z _ { 2 } \right)$ places maximum uncertainty (spread) at the origin (0, 0), with lateral displacement along $z _ { 1 }$ scaling proportionally with density spread. This geometry guarantees exact log-odds additivity and facilitates feature attribution.

$$
L ( { \pmb x } , t ) \approx \underbrace { g \bigg ( \frac { t _ { 0 } - \mu } { \sigma } \bigg ) } _ { z _ { 1 } ( { \pmb x } ) } \cdot \underbrace { 1 } _ { \lambda _ { 1 } ( t ) } + g ^ { \prime } \bigg ( \frac { t _ { 0 } - \mu } { \sigma } \bigg ) \cdot \underbrace { \bigg ( \frac { 1 } { \sigma } \bigg ) } _ { z _ { 2 } ( { \pmb x } ) } \cdot \underbrace { ( t - t _ { 0 } ) } _ { \lambda _ { 2 } ( t ) }\tag{5}
$$

where we recover a structure similar—or equivalent in the Logistic case $( g ^ { \prime } = 1 )$ —to that of Eq. (3), with separate input-dependent and horizon-dependent terms. The motivation for not absorbing $g ^ { \prime }$ in the output $z _ { 2 } ( \pmb { x } )$ is to ensure a closed-form relation between outputs and model parameters:

$$
\begin{array} { l } { { \sigma = 1 / z _ { 2 } , } } \\ { { \mu = t _ { 0 } - \frac { g ^ { - 1 } ( z _ { 1 } ) } { z _ { 2 } } , } } \end{array}
$$

which is convenient for model training. Lastly, to enforce the positivity of $\sigma ,$ we define $z _ { 2 } = \| z _ { 2 } \|$ 2 as the Euclidean norm over a vector of pre-outputs. While this serves a similar role to a standard softplus activation, it preserves positive homogeneity, which is desirable for fulfilling Property 1 as well as stable propagation-based attribution via LRP. The relation between the outputs and the probability density function is shown in Fig. 2.

## 3.2. ODDS-DR for High-Uncertainty Events

While symmetric, bell-shaped location-scale distributions are suited for deviations around a recurring rhythm, they may fall short for events with high temporal uncertainty, such as disease onset, mortality, and need for reintervention. Lifetime distributions—including log-normal, log-logistic, and Weibull forms—provide a highly difuse and flexible alternative by stretching probability mass over a much broader timeline, allowing it to both concentrate near $t = 0$ and extend out to very large time values. Focusing specifically on the Log-logistic distribution with parameters $( \alpha , \beta )$ and proceeding similarly as in Section 3.1, we can express the log-odds additively as a function of log(t):

$$
\begin{array} { r l } & { L ( \pmb { x } , t ) = \beta \cdot ( \log ( t ) - \log ( \alpha ) ) } \\ &  \qquad = \underbrace { \beta \cdot ( \log ( t _ { 0 } ) - \log ( \alpha ) ) } _ { z _ { 1 } ( \pmb { x } ) } \cdot \underbrace { \frac { 1 } { \lambda _ { 1 } ( t ) } + \underbrace { \beta } _ { z _ { 2 } ( \pmb { x } ) } \cdot \underbrace { ( \log ( t ) - \log ( t _ { 0 } ) ) } _ { \lambda _ { 2 } ( t ) } , } \end{array}\tag{6}
$$

(7)

which forms an exact linear decomposition centered at an operating horizon log $\left( t _ { 0 } \right)$ . The derivation is given in Appendix B.2. This expansion again leads to the identification of two additive terms, where outputs $z _ { 1 }$ and $z _ { 2 }$ can be defined to implement the generic form of $\mathrm { E q . ~ ( 3 ) }$ . Rearranging the equation above, we get the following expressions for the Log-Logistic parameters:

$$
\begin{array} { l } { \beta = z _ { 2 } , } \\ { \alpha = \exp ( - z _ { 1 } / z _ { 2 } ) \cdot t _ { 0 } . } \end{array}
$$

To enforce the positivity requirement of $z _ { 2 } .$ , we proceed as in Section 3.1 by defining it as the norm of a vector of pre-outputs. An analysis of the reparameterization above yields a similar interpretation as for the Bell-shaped cases: Output neurons with value close to zero produce a near-zero shape parameter $\beta ,$ corresponding to a maximum level of uncertainty (log-odds of zero).

## 3.3. Accounting for Event Non-Occurrence

In the presence of competing events or non-susceptibility, the primary event may never occur (e.g., due to acquired immunity or lifetime avoidance). Such event non-occurrence can be formalized via a mixture model where $\eta \in ( 0 , 1 )$ denotes the marginal probability of ever experiencing the event, and $1 - \eta$ denotes the non-occurrence probability. Under this mixture, the log-odds of the event occurring before time t can be expressed as:

$$
L ^ { ( \mathrm { n e w } ) } = \log \left( \frac { \eta \cdot P ( T \leq t ) } { \eta \cdot P ( T > t ) + ( 1 - \eta ) } \right)\tag{8}
$$

$$
= \sin \left( L + \log \eta , L _ { 0 } \right) ,\tag{9}
$$

where $L$ denotes the baseline log-odds conditional on event occurrence $\left( \mathrm { c f . \ E q . \ ( 2 ) } \right)$ , $L _ { 0 } = \log [ \eta / ( 1 -$ $\eta ) ]$ is the marginal log-odds of susceptibility, and smin $( a , b ) = - \log ( { e ^ { - a } + e ^ { - b } } )$ is the smooth minimum.

Equation (9) ofers an intuitive asymptotic interpretation. On a short-term horizon $( t  0$ $L \ll 0 )$ , the first term dominates, yielding the linear approximation $L ^ { \mathrm { ( n e w ) } } \approx L + \log \eta .$ Thus, short-term risk remains governed by the original outputs $z _ { 1 }$ and $z _ { 2 } ,$ shifted by the base rate log η. Conversely, on a long-term horizon $( t  \infty , L \gg 0 )$ , the second term dominates such that $L ^ { \mathrm { ( n e w ) } } $ $L _ { 0 }$ . The predicted log-odds naturally plateaus at the marginal log-odds of susceptibility, preventing runaway over-prediction.

## 4. Qualitative Experiments on Synthetic Data

We build a toy dataset that synthesizes some main aspects of timing clinical events. First, a phenotype representation x of the individual, here embedded in a one-dimensional input space. Second, the time T of event occurrence, which is also one-dimensional, with a limited 10-month observational period. The input and time-to-event targets form a joint distribution that can be depicted as a two-dimensional scatter plot, shown in Fig. 3 (top). The dataset is designed to exhibit the following joint input-target structure: (i) a general population that rarely experiences the clinical event within the 10-month observational time frame, (ii) an underrepresented subgroup that appears to share the statistics of the general population but that is distinctly undersampled, (iii) risk group A, which occasionally experiences the clinical event early after baseline, emulating short-term risks, and (iv) risk group B, experiencing the clinical event more frequently but over a longer time period, emulating longer-term risks.

![](images/ebb704843207c27f135939531e3ced7cb3469e8ae2f3cc3df01b1d5fac2b1377.jpg)

![](images/526a55df95758576a6001b3278e838b30841d4635c7f4455aea44f96f23c6219.jpg)

![](images/d8a08cc437ec2c331419efe90d7130e98039506346aec9a6b48457d3a3b9ffb3.jpg)  
Figure 3: ODDS-DR behavior on synthetic time-to-event data with heterogeneous risk profiles and representativeness gaps. Top: Observed and censored event times across four subpopulations: general population, under-represented, risk group A (events concentrated early at $t \leq 2 )$ , and risk group B (events distributed over an extended horizon $t \le 1 0 )$ . Bottom left: Predicted log-odds across subpopulations evaluated at horizons $t \in \{ 2 , 6 , 1 0 \}$ , decomposing total risk $( \lambda _ { 1 } z _ { 1 } + \lambda _ { 2 } z _ { 2 }$ , pink) into time-invariant baseline risk $\left( \lambda _ { 1 } z _ { 1 } \right.$ , dashed) and time-dependent adjustment. Bottom right: Continuous risk trajectories over time for risk groups A and B, demonstrating how ODDS-DR captures distinct temporal risk dynamics.

We train a log-logistic ODDS-DR model on this data. From the one-dimensional input, it produces the outputs $z _ { 1 } \in \mathbb { R }$ and $z _ { 2 } = \| z _ { 2 } \|$ with $z _ { 2 } \in \mathbb { R } ^ { 3 }$ . The mapping from input to outputs is achieved by a two-layer model, specifically, a static Gaussian feature expansion layer followed by a linear layer. The model is trained to maximize log-likelihood (full-batch, 2500 epochs, momentum of 0.85, learning rate: 0.01, weight decay: 0.01). The log-odds profile $\boldsymbol { L } ( \boldsymbol { x } , t )$ produced by our model after training can be navigated both in the input domain and across time, and further decomposed using Eq. (3) into two terms: a largely time-invariant baseline $\lambda _ { 1 } ( t ) z _ { 1 } ( \pmb { x } )$ , and another term $\lambda _ { 2 } ( t ) z _ { 2 } ( { \pmb x } )$ capturing the time-bound fluctuations. This analysis is depicted in Fig. 3 (bottom).

As expected, the general population is predicted to have significantly lower log-odds than the risk subgroups. Distinctly, our method demonstrates the capability to assign higher risk to the underrepresented subgroup in spite of an absence of observed clinical events in the data. This advantageous behavior derives directly from ODDS-DR’s Property 1 presented in Section 3, where low output activations arising from missing evidence or lack of support drive the parametric family to its mode of high uncertainty.

Looking more closely at the two risk groups, we observe that they have comparable timeindependent risks (captured by $\lambda _ { 1 } ( t ) z _ { 1 } ( \pmb { x } )$ and displayed as a dotted line), but difer significantly with respect to the horizon-dependent contributor $\lambda _ { 2 } ( t ) z _ { 2 } ( { \pmb x } )$ , causing overall risk to be similar on a short-term horizon $( t = 2 )$ while steadily growing for risk group B over the longer term $( t = 6$ and $t = 1 0 )$ . This ability of our method to represent risk profiles that are both subgroup-specific and time-dependent is also observed in longitudinal profiles (Fig. 3, bottom right), where both risk profiles follow a parametric structure (Property 2 presented in Section 3), with risk group B exhibiting a noticeably steeper progression.

## 5. Experiments on Real-world Clinical Cohorts

Broadening our analysis to real-world multivariate data, we evaluate our method on two clinical cohorts featuring varying population sizes, types of events, multi-dimensional panels of clinical measurements, as well as missing values and difering levels of right-censoring.

The first dataset, SUPPORT2 [33], derives from the ‘Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatment’—a prospective multi-centre observational study including 9,105 severely ill patients across five US clinical centres, recruited during 1989–1991 and 1992–1994. The primary outcome of interest was all-cause mortality. Data include 67 diferent features covering demographics, clinical and laboratory measurements, disease characteristics, as well as clinician- and model-based prognostic estimates. The median observed event time was 58 days (IQR: 14–252 days), with 32% right-censoring. Missingness varied across features, reaching up to 53% for laboratory measurements. With its large sample size, this dataset is ideal for testing the expressive power of time-to-event models. Although SUPPORT2 reflects clinical practice from more than 30 years ago, it remains a widely used benchmark in survival analysis and Explainable AI, enabling methodological comparison. We therefore use it primarily for methodological validation of ODDS-DR and acknowledge that the clinical relevance of individual risk markers may not fully transfer to contemporary practice.

The second dataset, CoA, is a prospective, well-characterized single-centre observational study conducted at the Deutsches Herzzentrum der Charité, comprising 160 patients with aortic coarctation (CoA) across 218 visits recruited between 2014 and 2022 [8]. The cohort provides a clinically detailed and disease-specific setting for evaluating the proposed framework, combining 37 structured clinical features extracted from routinely collected electronic health records, echocardiography and cardiovascular magnetic resonance (CMR) imaging. Missing values are present across imagingderived features, generally below 35%, except for two CMR parameters, which are missing in 51% of visits. The event of interest was re-coarctation requiring invasive reintervention, either catheterbased (stenting or balloon angioplasty) or surgical. The median observed event time was 21.5 months (IQR: 8.5–41.25 months). The cohort is inherently limited in size due to the low prevalence of the disease and further characterized by high right-censoring (75%), posing distinct regularization challenges.

Both datasets used a 70/30 training-validation split. SUPPORT2 was stratified by event status, whereas CoA used the same split as in our recent publication [8], stratified by sex, age, and event status. Overall, these two datasets provide broad coverage, spanning large-scale, low-noise conditions to intrinsically smaller-scale, high-noise settings. Their associated clinical task also comes in each case with clinical guidelines that we use as ground-truth signals to evaluate not only prediction accuracy, but also the plausibility of extracted risk markers.

## 5.1. Training and Hyperparameter Selection

In our experiments, outputs $z _ { 1 } ~ \in ~ \mathbb { R }$ and $z _ { 2 } ~ \in \ \mathbb { R } ^ { 3 }$ are predicted by a fully-connected feedforward neural network with 2–4 hidden layers and 16–256 neurons per layer, selected through hyperparameter tuning. For the CoA cohort, dropout was fixed at 0.5 to improve regularization and training stability, whereas for SUPPORT2 it was optimized over 0–0.5. Each hidden layer is followed by a ReLU activation. Features were standardized to zero mean and unit variance, with normalization statistics estimated exclusively from the observed (non-missing) values of each feature $x _ { i } .$ , such that $\mathbb { E } [ x _ { i } ] = 0$ and $\mathrm { s t d } ( x _ { i } ) = 1$ . To account for missing values in the input data, each standardized feature $x _ { i }$ was transformed into a new input representation via the map

$$
x _ { i } \mapsto \left( 1 - x _ { i } , 1 + x _ { i } \right) \cdot p _ { i }\tag{10}
$$

where $p _ { i } \in \{ 0 , 1 \}$ denotes feature presence, with $p _ { i } = 1$ if observed and $p _ { i } = 0$ if missing. This reserves the zero-input to feature missingness, distinguishing it from an actual observed zero value. This promotes the default mode of full feature missingness mapping to the zero output vector, enabling us to fulfill ODDS-DR’s Property 1 (cf. Section 3) in an end-to-end fashion. The model was trained using stochastic gradient descent (learning rate: $\mathrm { i } 0 ^ { - 4 } { - } 1 0 ^ { - 2 } ;$ momentum: 0–0.95; weight decay: $1 0 ^ { - 8 } – 1 0 ^ { - 2 } )$ while accounting for right-censored observations in the loss function.

All preprocessing, including feature scaling, was performed exclusively in the training cohort and subsequently applied to the validation cohort. Hyperparameters were optimized within the training cohort using cross-validation and Bayesian optimization.

## 5.2. Baselines

Our model was evaluated against a representative set of methods used for survival analysis, including classical distributional regression with classical parameterizations, the well-established Cox proportional hazard model [4] (Appendix A.1), deep extensions of the latter (DeepSurv [5] and Cox-Time [18]), as well as the non-parametric DeepHit model [12]. The same neural network architecture was used for all distributional models, difering only in the output layer: in the baseline models, the outputs directly represented the distributional parameters, whereas our parameterization made the outputs linearly related to the log-odds.

## 5.3. Results

Table 2 reports the discriminative performance of ODDS-DR alongside baseline methods and ablation variants, evaluated using Harrell’s C-index on the held-out validation sets. ODDS-DR matches or exceeds state-of-the-art survival models on both cohorts. On SUPPORT2, ODDS-DR paired with a log-logistic distribution achieves the highest overall discrimination, outperforming both deep survival baselines and linear approaches. On the CoA cohort—characterized by a lowsample regime and correspondingly wider confidence intervals—ODDS-DR’s performance aligns closely with that of the robust linear Cox model, while retaining a tangible edge over more complex neural survival formulations.

Table 2: Discriminative performance (Harrell’s C-index) on the SUPPORT2 and CoA validation sets across diferent distribution choices for ODDS-DR, an ablation study using standard parameterizations (no reparameterization), and competing baseline models. For C-index computation, distributional models and DeepHit were ranked by the restricted area under the predicted survival curve; DeepSurv, CoxNAM, and Cox by the log-risk scores, and Cox-Time by the cumulative hazard evaluated at the median observed event time of the training set. Values in parentheses report 95% confidence intervals obtained via non-parametric bootstrapping with 100 resamples. Bold denotes the best performance; underlined values indicate the second best.
<table><tr><td>Model</td><td>Distribution</td><td>SUPPORT2</td><td></td><td>CoA</td></tr><tr><td>ODDS-DR (ours)</td><td>Gaussian</td><td>0.80 (0.79–0.81)</td><td>0.66</td><td>(0.52–0.82)</td></tr><tr><td>ODDS-DR (ours)</td><td>Logistic</td><td>0.81 (0.80–0.82)</td><td></td><td>0.60 (0.44–0.77)</td></tr><tr><td>ODDS-DR (ours)</td><td>Log-logistic</td><td>0.83 (0.82–0.83)</td><td></td><td>0.64 (0.50–0.77)</td></tr><tr><td>Ablation study:</td><td></td><td></td><td></td><td></td></tr><tr><td>no reparameterization</td><td>Gaussian</td><td>0.81 (0.80–0.82)</td><td></td><td>0.61 (0.44–0.77)</td></tr><tr><td>no reparameterization</td><td>Logistic</td><td>0.80 (0.79–0.81)</td><td></td><td>0.61 (0.43–0.80)</td></tr><tr><td>no reparameterization</td><td>Log-logistic</td><td>0.82 (0.81–0.83)</td><td></td><td>0.57 (0.42–0.73)</td></tr><tr><td>DeepSurv [5]</td><td></td><td>0.81 (0.80–0.82)</td><td></td><td>0.59 (0.45–0.75)</td></tr><tr><td>Cox [4]</td><td></td><td>0.80 (0.79–0.81)</td><td></td><td>0.68 (0.54–0.81)</td></tr><tr><td>Cox-Time [18]</td><td></td><td>0.81 (0.80–0.83)</td><td></td><td>0.58 (0.43–0.72)</td></tr><tr><td>DeepHit [12]</td><td></td><td>0.81 (0.80–0.82)</td><td></td><td>0.65 (0.51–0.79)</td></tr><tr><td>DSM [13]</td><td></td><td>0.82 (0.81–0.83)</td><td></td><td>0.62 (0.48–0.75)</td></tr><tr><td>CoxNAM [20]</td><td></td><td>0.81 (0.80–0.82)</td><td></td><td>0.59 (0.37–0.81)</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Importantly, the ablation study demonstrates that the reparameterization introduced in ODDS-DR—designed to enforce linear decodability of log-odds for instance-level interpretability—does not compromise discriminative accuracy. Across distribution choices, the log-odds-reparameterized variants consistently match or slightly exceed the accuracy of baseline parameterizations. We attribute this resilience and slight performance edge to the beneficial inductive bias imposed by our parameterization, which acts as a regularizing prior in regions of low data density and under incomplete feature observations (cf. Property 1 introduced in Section 3).

## 5.4. Quantitative Evaluation of Extracted Risk Markers

Beyond its time-to-event predictive accuracy, the primary methodological advantage of ODDS-DR lies in its structured design—namely, the linear decodability of log-odds—which facilitates the direct attribution of predicted outcomes to input clinical features. A key application of this functionality is clinical marker discovery. Because absolute ground-truth markers are rarely known in clinical cohorts, preventing direct validation, we resort to clinical guidelines and established observational studies as a proxy ground truth.

For the CoA cohort, reference features identified as clinically relevant for reintervention include the ‘pressure gradient across the coarctation’, ‘body surface area (BSA)’, and ‘age’ [34, 35, 8]. For the SUPPORT2 cohort, relevant predictors include ‘age’, the ‘number of simultaneous diseases (comorbidities)’, and validated clinical scoring systems, specifically a score based on the Glasgow Coma Scale [36], the physiology score [37], and the APACHE III score [38]. For models that support a specific evaluation horizon (such as ODDS-DR and Cox-Time), horizons were set to 1 year for SUPPORT2 and 3 years for CoA, reflecting the time frames over which their respective clinical guidelines are most informative. We assess alignment with clinical guidelines via the Area

Table 3: Guideline alignment evaluated by Area Under the ROC Curve (AUROC), measuring how accurately feature attribution scores retrieve clinically established risk markers across the SUPPORT2 and CoA cohorts. Values in the right column denote 95% confidence intervals computed via non-parametric bootstrapping. Bold indicates the highest AUROC per dataset. The hourglass symbol $( ^ { \overline { { \mathbb { S } } } } )$ marks attribution methods incurring substantial computational overhead. Layer-wise Relevance Propagation (LRP) is employed where analytically tractable; DeepHit is paired with Integrated Gradients (IG) [39] and SHAP [6] due to softmax coupling; and Cox-Time’s prediction being an integral creates a natural pairing with IG. DeepSurv, CoxNAM, and Cox regression are inherently time-invariant and denoted $\mathrm { b y } \stackrel {  } { - } .$
<table><tr><td>Dataset</td><td>Method + Attribution</td><td>Horizon</td><td>Guideline alignment (AUROC)</td><td>95% confidence interval</td></tr><tr><td rowspan="7">SUPPORT2</td><td>ODDS-DR (Log-logistic) + LRP (ours)</td><td>1 year</td><td>0.89</td><td>0.88-0.90</td></tr><tr><td> $\mathrm { C o x - T i m e \ [ 1 8 ] + I G ^ { \Xi } }$ </td><td>1 year</td><td>0.87</td><td>0.87–0.88</td></tr><tr><td> $\mathrm { D e e p H i t } \ [ 1 2 ] + \mathrm { I G } ^ { \Xi }$ </td><td>1 year</td><td>0.81</td><td>0.80-0.82</td></tr><tr><td>DeepHit  $| 1 2 | + \mathrm { S H A P } ^ { \overline { { \mathbb { S } } } }$ </td><td>1 year</td><td>0.78</td><td>0.77-0.80</td></tr><tr><td>DSM  $\mathrm { [ 1 3 ] + I G } \overline { { \mathbb { Z } } }$ </td><td>1 year</td><td>0.84</td><td>0.83-0.85</td></tr><tr><td>DSM  $[ 1 3 ] + \mathrm { S H A P } ^ { \overline { { \mathbb { S } } } }$ </td><td>1 year</td><td>0.84</td><td>0.82-0.84</td></tr><tr><td> $\mathrm { D e e p S u r v } \ [ 5 ] + \mathrm { L R P }$ </td><td></td><td>0.86</td><td>0.84-0.88</td></tr><tr><td rowspan="7"></td><td> $\mathrm { C o x N A M \ [ 2 0 ] + c u s t o m }$  Cox + coefficient ranking</td><td></td><td>0.76 0.78</td><td>0.76-0.78</td></tr><tr><td>ODDS-DR (Log-logistic) + LRP (ours)</td><td>3 years</td><td>0.84</td><td>0.73-0.88</td></tr><tr><td> $\mathrm { C o x - T i m e \ [ 1 8 ] + I G ^ { \Xi } }$ </td><td>3 years</td><td>0.71</td><td>0.62-0.80</td></tr><tr><td> $\mathrm { D e e p H i t } \ [ 1 2 ] + \mathrm { I G } ^ { \Xi }$ </td><td>3 years</td><td>0.64</td><td>0.55-0.73</td></tr><tr><td> $\mathrm { { D e e p H i t } \bar { [ 1 2 ] } + S H A P \bar { \Sigma } }$ </td><td>3 years</td><td>0.65</td><td>0.53-0.71</td></tr><tr><td> $\mathrm { D S M } \ [ 1 3 ] + \mathrm { I G } ^ { \overline { { \mathbb { S } } } }$ </td><td>3 years</td><td>0.68</td><td>0.54-0.77</td></tr><tr><td>DSM  $[ 1 3 ] + \mathrm { S H A P } ^ { \overline { { \mathbb { Z } } } }$ </td><td>3 years</td><td>0.66</td><td>0.56-0.72</td></tr><tr><td></td><td> $\mathrm { D e e p S u r v } \ [ 5 ] + \mathrm { L R P }$ </td><td></td><td>0.78</td><td>0.68-0.84</td></tr><tr><td> $\mathrm { C o x N A M \ [ 2 0 ] + c u s t o m }$ </td><td></td><td></td><td>0.62</td><td>0.58-0.69</td></tr><tr><td>Cox + coefficient ranking</td><td></td><td></td><td>0.66</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Under the Receiver Operating Characteristic curve (AUROC), computed using absolute feature attributions averaged over the entire dataset. The results are summarized in Table 3.

Across both datasets, ODDS-DR achieves strong alignment with clinical guidelines, yielding the highest AUROC scores on both benchmarks (0.89 on SUPPORT2 and 0.84 on CoA), with a significant diference compared to simpler baselines such as Cox. While these results confirm the practical utility of our approach, we interpret them with appropriate nuance: clinical guidelines represent expert consensus rather than an absolute, exhaustive ground truth, and the wider confidence intervals on the smaller CoA cohort (0.73–0.88) reflect expected sampling variability.

## 5.5. Qualitative Analysis

While the quantitative evaluation above demonstrates competitive predictive performance and efective feature retrieval, the primary advantage of our approach over recent methods lies in its intrinsic interpretability, enabling the inference of clinical markers at the single-instance level robustly across multiple time horizons. We leverage this high analytic resolution to conduct an in-depth analysis of risk factors across both cohorts. In Fig. 4, we illustrate the dataset-wide importance of individual clinical features across several time horizons.

<table><tr><td rowspan=2 colspan=1>1.00</td><td rowspan=2 colspan=1>0.96</td><td rowspan=2 colspan=1>0.89</td><td rowspan=2 colspan=1>0.86</td><td></td></tr><tr><td rowspan=16 colspan=1>-1.0-0.8-0.6-0.4-0.2-0.0</td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.64</td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.69</td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.83</td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.80</td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.24</td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.60</td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.91</td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.68</td></tr><tr><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.65</td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.66</td></tr><tr><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.52</td></tr><tr><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=4>6   12Time horizon (months)</td><td rowspan=1 colspan=1>36</td></tr></table>

![](images/d1ef7cc8cdc68a4aa3df765a4373d2d46307a7c2b61244e5bdbc4c0933f10c87.jpg)  
Relative attribution over time

Absolute attribution  
A SUPPORT2 dataset
<table><tr><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>1.12</td><td rowspan=1 colspan=1>1.24</td><td rowspan=1 colspan=1>1.36</td></tr><tr><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>1.10</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>1.35</td></tr><tr><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.44</td></tr><tr><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.39</td></tr><tr><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.28</td><td rowspan=1 colspan=1>0.30</td></tr><tr><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.24</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.28</td></tr><tr><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>0.22</td></tr><tr><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>0.28</td></tr><tr><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.20</td></tr><tr><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.19</td></tr><tr><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.18</td></tr><tr><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.20</td></tr><tr><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.21</td></tr><tr><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.15</td></tr><tr><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>0.16</td></tr><tr><td rowspan=1 colspan=2>2Time horizo</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>12n (mont</td></tr></table>

B CoA dataset

![](images/283761628b0e53b72895c09c23c0992c4c346a12eb5d722ff95bfb1f0bfebc2c.jpg)  
Figure 4: Time-varying feature attribution across prediction horizons. Results are shown for (A) SUPPORT2 and (B) CoA cohorts across increasing follow-up times. Denoting by $R _ { i t } ^ { ( n ) }$ the attribution of log-odds L onto feature x<sub>i</sub> at horizon t for patient n, the left panels display mean absolute attribution scores, $R _ { i t } = \mathbb { E } _ { n } [ | R _ { i t } ^ { ( n ) } | ]$ ], reflecting overall feature importance. The right panels display feature-wise normalized scores, $R _ { i t } / \operatorname* { m a x } _ { t ^ { \prime } } R _ { i t ^ { \prime } }$ , to isolate and compare relative temporal trajectories across clinical markers.

In the SUPPORT2 cohort, several clinical features emerge as primary markers of mortality risk. Specifically, TISS scores and a do-not-resuscitate (DNR) order drive risk and its progression across the entire time horizon. Other variables, such as a cancer diagnosis, contribute substantially to mortality but exhibit a distinct temporal dynamic: their impact is secondary at baseline but increases multifold after 24 months. In contrast, the predominantly short-term relevance of the TISS and physiology scores is clinically sound, as these instruments are purposely designed to index short-term mortality risk.

In the CoA cohort, early risks are strongly attributable to specific clinical features, whereas longterm risk progression shows less concentrated attribution and remains partly unexplained. Among the primary drivers of early reintervention is ‘beta-blocker intake’—a marker that likely reflects underlying hypertension, which has been linked to an increased risk of early aortic re-coarctation [8]. An even more pronounced trend toward short-term reintervention is observed for ‘previous balloon angioplasty’, where its role as a clinical marker attenuates sharply over longer horizons. A similar temporal pattern has been reported in previous cohorts, where the hazard of reintervention after balloon angioplasty was highest during early follow-up [40]. This observation motivates the hypothesis that its prognostic value might be confined to the early period following the baseline visit, a pattern that remains to be confirmed in other cohorts.

The ability of our analysis to infer clinical variables that drive short- and long-term risk may prove useful as a stratification tool for clinical follow-up planning. For example, patients presenting with multiple early-risk factors could benefit from intensified monitoring and shorter surveillance intervals.

## 6. Conclusion and Discussion

We introduced ODDS-DR, an odds-informed parameterization of deep distributional regression that makes continuous probabilistic predictions directly amenable to feature attribution. By structuring network outputs to contribute linearly to event log-odds, the resulting explanations become continuous and parametric in time, cleanly disentangling short-term from long-term effects. Furthermore, rather than relying on costly model-agnostic explainers evaluated independently across discrete time horizons, ODDS-DR recovers entire feature-resolved risk trajectories in just two forward-backward passes, achieving orders-of-magnitude computational speedups.

Empirical results on real-world clinical time-to-event benchmarks corroborate the theoretical advantages of our method. Beyond matching the predictive performance of competitive proportional hazards and deep distributional baselines, ODDS-DR produces feature attributions that substantially align with clinical guidelines across their relevant time horizons, while resolving fine-grained temporal dynamics at the single-feature level.

Case studies on SUPPORT2 and the CoA cohort highlight distinct facets of clinical utility: on SUPPORT2, ODDS-DR scales computationally while delivering expressive, fully auditable nonlinear models; on the CoA cohort, it demonstrates high fidelity in a granular, single-center setting with dense multimodal monitoring—a regime representative of specialized clinical management for rare cardiovascular conditions, isolated from the cross-site batch efects that frequently confound multi-cohort evaluations.

## Limitations and Future Work

From a methodological standpoint, ODDS-DR currently relies on parametric families with closed-form log-odds expressions, confining the framework to unimodal density profiles. While unimodal formulations proved suficient for our benchmark evaluations, capturing more complex hazard structures—such as those induced by competing events or non-susceptible subgroups—will demand richer, e.g., mixture-based representations. Because standard probability-space mixture models do not yield additive log-odds decompositions, enhancing expressiveness by designing mixture mechanisms directly within the log-odds domain represents a natural next methodological step.

From an application perspective, our clinical analysis is observational and subject to sample size constraints, particularly in the CoA cohort, which reflects an intrinsically small target population. Furthermore, our framework estimates prognostic risk under prevailing care patterns rather than evaluating counterfactual treatment regimes. Rather than supporting immediate therapeutic decisions, our findings yield exploratory hypotheses on how specific markers predict patient vulnerability under routine care.

Finally, the computational eficiency of ODDS-DR and its native compatibility with propagation-based attribution methods (such as Layer-wise Relevance Propagation) naturally invite extension to high-dimensional data modalities. In particular, applying ODDS-DR to baseline medical imaging to map time-resolved risk trajectories of lesion progression or structural tissue degeneration ofers a promising avenue for future work.

## Data Availability Statement

The SUPPORT2 dataset [33] is publicly available and can be downloaded through the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/880/support2). The CoA dataset [8] is subject to patient privacy and institutional data protection regulations. Data access requests must be directed to the corresponding author of the original study, subject to formal confidentiality agreements and required ethical approvals.

## Ethics Statement

The prospective observational study of patients with CoA has a registered protocol (NCT02591940) and was approved by the institutional ethics committee (Charité–Universitätsmedizin Berlin: EA2/172/13) and conducted in accordance with the principles of the Declaration of Helsinki. All participants or their legal representatives provided written informed consent before enrolment.

## Acknowledgements

This work was funded by the German Ministry for Education and Research as BIFOLD – Berlin Institute for the Foundations of Learning and Data (ref. BIFOLD25B). G.M. is supported by the Einstein Center for Early Disease Interception (EC-EDI). M.K. has received funding from the Bundesministerium für Forschung, Technologie und Raumfahrt (BMFTR, Federal Ministry of Research, Technology and Space, Project: VADYS-ME, grant number: 01EJ2406A). M.K. and T.K. have received funding within the CHAIN project (Project ID: 101314833), supported by the European Union’s EU4Health programme.

## Declaration of Generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the author(s) used Gemini 3.8 and Claude Opus 5.5 in order to improve formulations. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the publication.

## References

[1] C. Bishop, Mixture density networks, Tech. rep., Aston University, Birmingham (1994).

[2] D. M. Stasinopoulos, R. A. Rigby, Generalized additive models for location scale and shape (GAMLSS) in R, Journal of Statistical Software 23 (7) (2007).

[3] S. Rasp, S. Lerch, Neural networks for postprocessing ensemble weather forecasts, Monthly Weather Review 146 (11) (2018) 3885–3900.

[4] D. R. Cox, Regression models and life-tables, Journal of the Royal Statistical Society Series B: Statistical Methodology 34 (2) (1972) 187–202.

[5] J. L. Katzman, U. Shaham, A. Cloninger, J. Bates, T. Jiang, Y. Kluger, Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network, BMC Medical Research Methodology 18 (1) (2018) 24.

[6] S. M. Lundberg, G. G. Erion, H. Chen, A. J. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, S. Lee, From local explanations to global understanding with explainable AI for trees, Nat. Mach. Intell. 2 (1) (2020) 56–67.

[7] J. Keyl, P. Keyl, G. Montavon, R. Hosch, A. Brehmer, et al., Decoding pan-cancer treatment outcomes using multimodal real-world data and explainable artificial intelligence, Nature Cancer 6 (2) (2025) 307–322.

[8] L. Fierley, J. Versnjak, G. Gabel, P. Kramer, L. Goubergrits, F. Berger, G. Montavon, T. Kuehne, M. Kelm, Prediction of hypertension and restenosis under guideline-directed management in aortic coarctation: development and validation of machine-learning models, eClinicalMedicine 97 (2026) 104041.

[9] E. Štrumbelj, I. Kononenko, Explaining prediction models and individual predictions with feature contributions, Knowledge and Information Systems 41 (3) (2013) 647–665.

[10] S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. Müller, W. Samek, On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation, PLOS ONE 10 (7) (2015) e0130140.

[11] G. Montavon, A. Binder, S. Lapuschkin, W. Samek, K.-R. Müller, Layer-wise relevance propagation: An overview, in: Explainable AI, Vol. 11700 of Lecture Notes in Computer Science, Springer, 2019, pp. 193–209.

[12] C. Lee, W. R. Zame, J. Yoon, M. van der Schaar, DeepHit: A deep learning approach to survival analysis with competing risks, in: AAAI, AAAI Press, 2018, pp. 2314–2321.

[13] C. Nagpal, X. Li, A. Dubrawski, Deep survival machines: Fully parametric survival regression and representation learning for censored data with competing risks, IEEE Journal of Biomedical and Health Informatics 25 (8) (2021) 3163–3175.

[14] D. Nix, A. Weigend, Estimating the mean and variance of the target probability distribution, in: Proceedings of 1994 IEEE International Conference on Neural Networks (ICNN’94), IEEE, 1994, p. 55–60 vol.1.

[15] D. Rügamer, C. Kolb, N. Klein, Semi-structured distributional regression, The American Statistician 78 (1) (2023) 88–99.

[16] B. Sick, T. Hathorn, O. Dürr, Deep transformation models: Tackling complex regression problems with neural network based transformation models, in: 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 2476–2481.

[17] G. Campanella, I. Häggström, L. Kook, T. Hothorn, T. J. Fuchs, A flexible deep learning framework for survival analysis with medical data, in: MICCAI (15), Vol. 15974 of Lecture Notes in Computer Science, Springer, 2025, pp. 3–13.

[18] H. Kvamme, Ø. Borgan, I. Scheel, Time-to-event prediction with neural networks and Cox regression, J. Mach. Learn. Res. 20 (2019) 129:1–129:30.

[19] A. Alabdallah, O. Hamed, M. Ohlsson, T. Rögnvaldsson, S. Pashami, CoxSE: Exploring the potential of self-explaining neural networks with Cox proportional hazards model for survival analysis, Knowledge-Based Systems 333 (2026) 114996.

[20] L. Xu, C. Guo, CoxNAM: An interpretable deep survival analysis model, Expert Systems with Applications 227 (2023) 120218.

[21] M. F. Gensheimer, B. Narasimhan, A scalable discrete-time survival model for neural networks, PeerJ 7 (2019) e6257.

[22] H. Kvamme, Ø. Borgan, Continuous and discrete-time survival prediction with neural networks, Lifetime Data Analysis 27 (4) (2021) 710–736.

[23] Z. Zhang, Parametric regression model for survival data: Weibull regression model as an example, Annals of Translational Medicine 4 (24) (2016).

[24] A. Bennis, S. Mouysset, M. Serrurier, Estimation of conditional mixture Weibull distribution with right censored data using neural network for time-to-event analysis, in: PAKDD (1), Vol. 12084 of Lecture Notes in Computer Science, Springer, 2020, pp. 687–698.

[25] S. M. Lundberg, S. Lee, A unified approach to interpreting model predictions, in: NIPS, 2017, pp. 4765–4774.

[26] F. Bley, S. Lapuschkin, W. Samek, G. Montavon, Explaining predictive uncertainty by exposing second-order efects, Pattern Recognition 160 (2025) 111171.

[27] P. Iversen, S. Witzke, K. Baum, B. Y. Renard, Identifying drivers of predictive aleatoric uncertainty, in: IJCAI, ijcai.org, 2025, pp. 5453–5462.

[28] S. H. Langbein, N. Koenen, M. N. Wright, Gradient-based explanations for deep learning survival models, in: ICML, Vol. 267 of Proceedings of Machine Learning Research, PMLR / OpenReview.net, 2025.

[29] M. Krzyziński, M. Spytek, H. Baniecki, P. Biecek, Survshap(t): Time-dependent explanations of machine learning survival models, Knowledge-Based Systems 262 (2023) 110234.

[30] M. S. Kovalev, L. V. Utkin, E. M. Kasimov, Survlime: A method for explaining machine learning survival models, Knowl. Based Syst. 203 (2020) 106164.

[31] L. V. Utkin, D. Y. Eremenko, A. V. Konstantinov, V. A. Muliukha, Survbenim: The beranbased neural importance model for explaining survival models, IEEE Access 13 (2025) 24137– 24157.

[32] S. H. Langbein, H. Baniecki, F. Fumagalli, N. Koenen, M. N. Wright, J. Herbinger, Functional decomposition and shapley interactions for interpreting survival models, in: Proceedings of the 43rd International Conference on Machine Learning, Vol. 306 of Proceedings of Machine Learning Research, PMLR, 2026, pp. 62603–62641.

[33] W. Knaus, J. Lynn, Study to understand prognoses and preferences for outcomes and risks of treatment (support) and hospitalized elderly longitudinal project (help), 1989–1997, version 2 (2020).

[34] H. Baumgartner, J. De Backer, S. V. Babu-Narayan, W. Budts, M. Chessa, et al., 2020 esc guidelines for the management of adult congenital heart disease: The task force for the management of adult congenital heart disease of the european society of cardiology (esc)., European Heart Journal 42 (6) (2021) 563–645.

[35] E. M. Isselbacher, O. Preventza, J. H. Black, J. G. Augoustides, A. W. Beck, et al., 2022 acc/aha guideline for the diagnosis and management of aortic disease: A report of the american heart association/american college of cardiology joint committee on clinical practice guidelines, Circulation 146 (24) (2022) e334–e482.

[36] V. Ihalainen, A. Pölkki, A. Moser, J. Takala, S. M. Jakob, S. Bendel, R. Raj, P. T. Pekkarinen, T. Koivisto, J. Starkopf, M. Reinikainen, The glasgow coma scale as a predictor of mortality in intensive care: relevance of underlying cause of impaired consciousness, Journal of Critical Care 94 (2026) 155605.

[37] W. Knaus, F. Harrell, J. Lynn, L. Goldman, R. Phillips, A. Connors, Jr, N. Dawson, W. Fulkerson, R. Calif, N. Desbiens, P. Layde, R. Oye, P. Bellamy, R. Hakim, D. Wagner, The support prognostic model. objective estimates of survival for seriously ill hospitalized adults. study to understand prognoses and preferences for outcomes and risks of treatments, Annals of internal medicine 122 (1995) 191–203.

[38] W. A. Knaus, D. P. Wagner, E. A. Draper, J. E. Zimmerman, M. Bergner, P. G. Bastos, C. A. Sirio, D. J. Murphy, T. Lotring, A. Damiano, F. E. Harrell, The apache iii prognostic system: Risk prediction of hospital mortality for critically iii hospitalized adults, Chest 100 (6) (1991) 1619–1636.

[39] M. Sundararajan, A. Taly, Q. Yan, Axiomatic attribution for deep networks, in: ICML, Vol. 70 of Proceedings of Machine Learning Research, PMLR, 2017, pp. 3319–3328.

[40] H.-H. Chiu, J.-K. Wang, Y.-S. Chen, I.-S. Chiu, C.-I. Chang, M.-T. Lin, C.-W. Lu, S.-N. Chiu, C.-A. Chen, M.-H. Wu, Long-term outcomes of native coarctation of the aorta after balloon angioplasty or surgical aortoplasty in newborns and young infants less than 3 months of age, Acta Cardiologica Sinica 29 (2) (2013) 168–174.

## Appendix A. Technical Background

Appendix A.1. Cox proportional hazards model

The Cox proportional hazards model describes the instantaneous event rate, or hazard, conditional on a set of covariates x. The hazard function is defined as

$$
h ( t \mid \pmb { x } ) = \operatorname* { l i m } _ { \Delta t  0 } \frac { P ( t \leq T < t + \Delta t \mid T \geq t , \pmb { x } ) } { \Delta t } ,\tag{A.1}
$$

where $T$ denotes the event time. Equivalently, the hazard can be expressed in terms of the survival function $S ( t \mid x )$ as

$$
h ( t \mid x ) = - { \frac { d } { d t } } \log S ( t \mid x ) = { \frac { f ( t \mid x ) } { S ( t \mid x ) } }\tag{A.2}
$$

where $f ( t \mid x )$ denotes the corresponding probability density function. The Cox model assumes that the hazard can be factorized into a time-dependent baseline hazard $h _ { 0 } ( t )$ and a covariate-dependent relative risk term:

$$
h ( t \mid \pmb { x } ) = h _ { 0 } ( t ) \exp \Big ( \beta ^ { \top } \pmb { x } \Big )\tag{A.3}
$$

where $\beta$ denotes the vector of regression coeficients. The term $h _ { 0 } ( t )$ describes the baseline hazard and can be left unspecified. The proportional-hazards property follows directly by considering two individuals with covariates $\mathbf { \mathcal { x } } _ { i }$ and $\mathbf { \boldsymbol { x } } _ { j }$ . Their hazard ratio is

$$
{ \frac { h ( t \mid \mathbf { x } _ { i } ) } { h ( t \mid \mathbf { x } _ { j } ) } } = { \frac { h _ { 0 } ( t ) \exp ( \boldsymbol { \beta } ^ { \top } \mathbf { x } _ { i } ) } { h _ { 0 } ( t ) \exp ( \boldsymbol { \beta } ^ { \top } \mathbf { x } _ { j } ) } } = \exp \left[ \boldsymbol { \beta } ^ { \top } ( \mathbf { x } _ { i } - \mathbf { x } _ { j } ) \right]\tag{A.4}
$$

Since the baseline hazard $h _ { 0 } ( t )$ cancels, the hazard ratio does not depend on time. Consequently, the classical Cox model assumes time-invariant relative efects of the covariates. A major advantage of the Cox model is that the regression coeficients can be estimated without explicitly specifying the baseline hazard. For an individual i experiencing an event at time $t _ { i } ,$ let

$$
\mathcal { R } ( t _ { i } ) = \{ j : \tilde { T } _ { j } \geq t _ { i } \}\tag{A.5}
$$

denote the risk set containing all individuals who are still under observation and event-free immediately before $t _ { i }$ . Conditional on one event occurring at $t _ { i }$ , the contribution of individual i to the partial likelihood is

$$
\frac { \exp ( \boldsymbol { \beta } ^ { \top } \mathbf { x } _ { i } ) } { \sum _ { j \in \mathcal { R } ( t _ { i } ) } \exp ( \boldsymbol { \beta } ^ { \top } { \boldsymbol { x } } _ { j } ) }\tag{A.6}
$$

The partial likelihood over all observed events is therefore

$$
\mathcal { L } _ { \mathrm { p a r t i a l } } = \prod _ { i : \delta _ { i } = 1 } \frac { \exp ( \boldsymbol { \beta } ^ { \top } \mathbf { x } _ { i } ) } { \sum _ { j \in \mathcal { R } ( t _ { i } ) } \exp ( \boldsymbol { \beta } ^ { \top } \mathbf { x } _ { j } ) }\tag{A.7}
$$

where $\delta _ { i } = 1$ indicates an observed event. Taking the logarithm gives

$$
\ell ( \boldsymbol { \beta } ) = \sum _ { i : \delta _ { i } = 1 } \left[ \boldsymbol { \beta } ^ { \top } \mathbf { x } _ { i } - \log \Big ( \sum _ { j \in \mathcal { R } ( t _ { i } ) } \exp ( \boldsymbol { \beta } ^ { \top } \mathbf { x } _ { j } ) \Big ) \right]\tag{A.8}
$$

The regression coeficients are obtained by maximizing the partial log-likelihood.

## Appendix A.2. DeepSurv

DeepSurv extends the Cox proportional hazards model by replacing the linear predictor $\beta ^ { \top }$ x with a nonlinear function $g _ { \pmb { \theta } } ( \pmb { x } )$ represented by a neural network. The hazard is consequently modeled as

$$
h ( t \mid x ) = h _ { 0 } ( t ) \exp \left( g _ { \pmb { \theta } } ( \pmb { x } ) \right)\tag{A.9}
$$

The neural network is trained by minimizing the negative Cox partial log-likelihood,

$$
\mathcal { I } _ { \mathrm { C o x } } = - \sum _ { i : \delta _ { i } = 1 } \left[ g _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { i } ) - \log \Big ( \sum _ { j \in \mathcal { R } ( t _ { i } ) } \exp \left( g _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { j } ) \right) \Big ) \right]\tag{A.10}
$$

DeepSurv therefore allows nonlinear relationships between the input variables and relative risk while retaining the proportional-hazards assumption. In particular, the relative efect of two individuals remains independent of time,

$$
\frac { h ( t \mid \pmb { x } _ { i } ) } { h ( t \mid \pmb { x } _ { j } ) } = \exp \left[ g _ { \pmb { \theta } } ( \pmb { x } _ { i } ) - g _ { \pmb { \theta } } ( \pmb { x } _ { j } ) \right]\tag{A.11}
$$

An important computational property of Cox partial likelihood is that the loss of an individual experiencing an event depends on all individuals belonging to its corresponding risk set. Consequently, observations are not independent within the loss function. The exact evaluation of the partial likelihood therefore requires access to the complete risk sets. Standard mini-batch training, in which subsets of observations are sampled independently, provides only an approximation to the full Cox objective. This contrasts with the distributional regression models considered in this work, for which the likelihood decomposes into independent per-sample contributions and can therefore be optimized accurately using mini-batches.

## Appendix A.3. Negative log-likelihood for right-censored distributional regression

Let $T$ denote the event time with probability density function $f ( t \mid \theta )$ , cumulative distribution function $F ( t \mid \theta )$ , and survival function

$$
S ( t \mid \pmb \theta ) = 1 - F ( t \mid \pmb \theta )\tag{A.12}
$$

where the distribution parameters $\pmb \theta$ are predicted by the neural network. For each individual, we have

$$
\tilde { T } _ { i } = \operatorname* { m i n } ( T _ { i } , C _ { i } )\tag{A.13}
$$

where $C _ { i }$ denotes the censoring time, together with the event indicator

$$
\delta _ { i } = \left\{ { \begin{array} { l l } { 1 , } & { T _ { i } \leq C _ { i } } \\ { 0 , } & { T _ { i } > C _ { i } } \end{array} } \right.\tag{A.14}
$$

For an observed event $( \delta _ { i } = 1 )$ , the likelihood contribution is given by the probability density evaluated at the observed event time,

$$
\mathcal { L } _ { i } = f ( \tilde { T } _ { i } \mid \pmb { \theta } _ { i } )\tag{A.15}
$$

For a right-censored observation $( \delta _ { i } \ = \ 0 )$ , the available information is that the event has not occurred up to the censoring time. Its likelihood contribution is therefore

$$
\mathcal { L } _ { i } = P ( T _ { i } > \tilde { T } _ { i } \mid \pmb \theta _ { i } ) = S ( \tilde { T } _ { i } \mid \pmb \theta _ { i } )\tag{A.16}
$$

Both cases can be written compactly as

$$
\mathcal { L } _ { i } = f ( \tilde { T } _ { i } \mid \pmb { \theta } _ { i } ) ^ { \delta _ { i } } S ( \tilde { T } _ { i } \mid \pmb { \theta } _ { i } ) ^ { 1 - \delta _ { i } }\tag{A.17}
$$

The corresponding negative log-likelihood is

$$
\mathcal { I } = - \sum _ { i = 1 } ^ { N } \Big [ \delta _ { i } \log f ( \tilde { T _ { i } } \mid \theta _ { i } ) + ( 1 - \delta _ { i } ) \log S ( \tilde { T _ { i } } \mid \theta _ { i } ) \Big ]\tag{A.18}
$$

Equivalently, using $S ( t \mid \pmb \theta ) = 1 - F ( t \mid \pmb \theta )$

$$
\mathcal { I } = - \sum _ { i = 1 } ^ { N } \Big [ \delta _ { i } \log f ( \tilde { T _ { i } } \mid \pmb { \theta } _ { i } ) + \left( 1 - \delta _ { i } \right) \log \Big ( 1 - F ( \tilde { T _ { i } } \mid \pmb { \theta } _ { i } ) \Big ) \Big ]\tag{A.19}
$$

This formulation applies to any continuous parametric distribution for which the probability density and survival function can be evaluated (Weibull, Gaussian, logistic, log-logistic).

## Appendix B. Derivations of ODDS-DR Parameterizations

Appendix B.1. Log-odds for bell-shaped distributions

In this subsection, we provide derivations for the log-odds expressions of the two bell-shaped distributions used in the main paper, namely, the logistic and Gaussian distributions. In particular, we demonstrate that their log-odds can be expressed as $g \big ( \frac { t - \mu } { \sigma } \big )$ , where $g$ is a monotonically increasing odd function, while $\mu$ and $\sigma$ are location and scale parameters of the distributions, respectively.

Log-odds for the Logistic distribution. Let $T$ be a random variable following a logistic distribution with parameters $\mu$ and $\sigma > 0$ . Its CDF is given by $\begin{array} { r } { F ( t ) = \frac { e ^ { \frac { t - \mu } { \sigma } } } { 1 + e ^ { \frac { t - \mu } { \sigma } } } } \end{array}$ . Hence, the log-odds are given by:

$$
L ( t ) = \log \left( { \frac { P ( T \leq t ) } { P ( T > t ) } } \right) = \log \left( { \frac { F ( t ) } { 1 - F ( t ) } } \right)\tag{B.1}
$$

$$
= \log \left( \frac { e ^ { \frac { t - \mu } { \sigma } } } { 1 + e ^ { \frac { t - \mu } { \sigma } } } \cdot \frac { 1 } { \frac { 1 } { 1 + e ^ { \frac { t - \mu } { \sigma } } } } \right) = \frac { t - \mu } { \sigma }\tag{B.2}
$$

We identify the desired structure by setting g to the identity function.

Log-odds for the Gaussian distribution. Let T follow a Gaussian distribution with mean $\mu$ and variance $\sigma ^ { 2 }$ . Its CDF is given by $\begin{array} { r } { F ( t ) = \frac { 1 } { 2 } \left( 1 + \mathrm { e r f } ( \frac { t - \mu } { \sigma \sqrt { 2 } } ) \right) } \end{array}$ , and the log-odds can be expressed as:

$$
L ( t ) = \log \left( { \frac { F ( t ) } { 1 - F ( t ) } } \right) = \log \left( { \frac { 1 + \operatorname { e r f } \left( { \frac { t - \mu } { \sigma { \sqrt { 2 } } } } \right) } { 1 - \operatorname { e r f } \left( { \frac { t - \mu } { \sigma { \sqrt { 2 } } } } \right) } } \right) = 2 \operatorname { t a n h } ^ { - 1 } \left( \operatorname { e r f } \left( { \frac { t - \mu } { \sigma { \sqrt { 2 } } } } \right) \right)\tag{B.3}
$$

For $g ( \xi ) = 2 \operatorname { t a n h } ^ { - 1 } ( \operatorname { e r f } ( \xi / \sqrt { 2 } ) )$ , we once again identify the desired structure.

Appendix B.2. Log-odds for the Log-Logistic distribution

In this subsection, we derive the log-odds expression for a random variable T following the log-logistic distribution. Its CDF is given by $\begin{array} { r } { F ( t ) = \frac { t ^ { \beta } } { \alpha ^ { \beta } + t ^ { \beta } } } \end{array}$ , where α and $\beta$ are the parameters of the distribution. The log-odds are given by:

$$
L ( t ) = \log { \left( \frac { t ^ { \beta } } { \alpha ^ { \beta } + t ^ { \beta } } \cdot \frac { \alpha ^ { \beta } + t ^ { \beta } } { \alpha ^ { \beta } } \right) }\tag{B.4}
$$

$$
= \beta \cdot ( \log ( t ) - \log ( \alpha ) )
$$

$$
= \underbrace { { \boldsymbol { \beta } } \cdot ( \log ( t _ { 0 } ) - \log ( \alpha ) ) } _ { z _ { 1 } } + \underbrace { { \boldsymbol { \beta } } } _ { z _ { 2 } } \cdot ( \log ( t ) - \log ( t _ { 0 } ) )\tag{B.5}
$$

(B.6)