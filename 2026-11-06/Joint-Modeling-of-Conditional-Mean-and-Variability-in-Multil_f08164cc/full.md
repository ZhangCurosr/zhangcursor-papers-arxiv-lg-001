# Joint Modeling of Conditional Mean and Variability in Multilevel Functional Data: Application to Postprandial Glucose Trajectories in the AEGIS Study

Yang Xiao<sup>1</sup>, Francisco Gude<sup>2,3</sup>, Marcos Matabuena<sup>1,∗</sup>

<sup>1</sup> Division of Biological and Life Sciences, Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, United Arab Emirates

2 Instituto de Investigaci´on Sanitaria de Santiago de Compostela (IDIS), Santiago de Compostela, Spain

3 Department of Medicine, University of Santiago de Compostela, Santiago de Compostela, Spain

Corresponding author: marcos.matabuena@mbzuai.ac.ae

## Abstract

Functional data analysis (FDA) regression models provide a natural analytical framework for quantifying time-varying associations between Euclidean predictors and high-frequency biological signals (e.g., minute-by minute physical activity measurements and glucose trajectories). In digital health applications, multiple functional trajectories are often collected from each participant. Existing multilevel functional regression methods primarily focus on modeling the conditional mean as a function of individual-level covariates (e.g., age, body mass index, and demographic characteristics), whereas conditional variability has received comparatively little attention, despite its importance for understanding biological processes and characterizing individual heterogeneity. To address this gap, we introduce, to the best of our knowledge, the first heteroskedastic functional mixed-efects regression framework that jointly models the conditional mean and variance of repeated functional observations while incorporating participant-specific random efects in both components. This framework enables the simultaneous characterization of time-varying biological responses and their variability, capturing dimensions of individual heterogeneity that conventional multilevel functional regression models cannot adequately represent. We further develop a participantlevel bootstrap procedure to quantify uncertainty in the estimated functional parameters. We apply the proposed framework to postprandial glucose trajectories from the AEGIS study, revealing clinically relevant patterns of time-varying variability that are overlooked by conventional analyses focusing primarily on mean glucose responses. In this general adult population, premeal glucose levels, carbohydrate intake, and meal type exhibit distinct time-dependent associations with postprandial glucose variability, whereas glycated hemoglobin (HbA1c) shows a comparatively weaker association with variability despite retaining its association with the conditional mean. We also identify substantial between-participant heterogeneity in residual variability, even among individuals with similar observed covariates, highlighting dimensions of biological heterogeneity that cannot be adequately characterized through conditional mean models alone. Overall, our findings establish heteroskedastic functional mixed-efects regression as a flexible framework for jointly characterizing the conditional mean, variability, and participant-specific heterogeneity of longitudinal biological signals, with important implications for personalized modeling in digital health and precision nutrition in the analyzed case study.

## 1 Introduction

The ongoing revolution in biological measurement technologies is enabling the collection of multiple high-frequency time series of physiological variables, such as glucose levels and heart rate, on diferent time scales, with the potentia to accelerate scientific discovery and improve clinical decision making Matabuena et al. (2023). A key feature of these modern datasets is that devices record observations every few seconds or minutes; these time-stamped measurements can be naturally represented as a mathematical function over time. Because individuals are typically monitored repeatedly (e.g., across days or other time periods), the resulting data structure collected is functional and multilevel (hierarchical) Matabuena and Crainiceanu (2026). To emphasize this structure, Figure 1 displays, for two participants in the AEGIS study, glucose trajectories after breakfast, lunch and dinner, recorded every five minutes on five days, together with the corresponding fitted mean curves and gaussian assympotic prediction bands. The plots of individual glucose trajectories show substantial heterogeneity both within and between individuals across meals and over days, underscoring the importance of multilevel random-efects models that capture individual variation specific to the time after a meal for personalized prediction.

In digital health applications, multilevel functional models are increasingly used to analyze high-frequency measurements of functional biomarkers. When individual-level covariate information is not available and only repeated functional trajectories are observed, the prevailing approach is multilevel functional principal component analysis (MFPCA) (D et al., 2009). MFPCA has been developed for a wide range of hierarchical designs and estimation paradigms (Crainiceanu et al., 2024), and recent work has introduced computationally eficient algorithms for both Gaussian and non-Gaussian functional outcomes (Cui et al., 2023; Leroux et al., 2023).

When individual-level covariate information is available (e.g. demographic and anthropometric measures), a range of frequentist and Bayesian functional regression methods have been proposed to model the conditional mean function (Morris and Carroll, 2006; Greven and Scheipl, 2017). Scalable implementations for large cohorts, such as NHANES, have also emerged (Cui et al., 2022; Sun and Kowal, 2025). These often rely on a “mass-univariate-thensmooth” pipeline: separate univariate models are fitted at each time point on the functional grid and the resulting coeficient estimates are then smoothed over time to enforce temporal regularity and improve statistical inference. Because each time point can be modeled independently, this strategy is naturally parallelizable and scales well in large epidemiological studies—for example, minute-by-minute accelerometer-derived physical activity signals collected from about 100,000 participants in the UK Biobank.

Despite substantial progress in multilevel functional modeling, most multilevel functional regression methods target only the conditional mean, leaving other distributional features, such as the conditional variance, largely unmodeled. Approaches that capture temporal structure in conditional variability are still scarce (Backenroth et al., 2018; St¨ocker et al., 2021), although variability is key for characterizing individual heterogeneity and for designing temporally targeted digital health interventions (e.g., chronobiologically personalized interventions). For scalar repeated measurements, mixed-efects location–scale models allow covariates and subject-specific random efects to influence both the conditional mean and within-subject variance (Hedeker et al., 2008; Hedeker and Nordgren, 2013). Subsequent developments have extended these models to nonlinear learning curves (Williams et al., 2019) and dyadic data (Rast and Ferrer, 2018), while recent computational advances have enabled more eficient estimation (Gill and Hedeker, 2023; Wu and Hedeker, 2026). However, these approaches are designed for scalar responses and do not directly accommodate the functional structure of repeated trajectories, where both the conditional mean and variability may evolve continuously over time. To address this limitation, we propose a joint multilevel functional location–scale model that simultaneously estimates covariate-dependent mean and variance functions, enabling the characterization of time-varying biological responses and their heterogeneity across individuals.

Why variability matters. Understanding biological variability is essential for elucidating physiological mechanisms, characterizing individual heterogeneity, and informing the development of personalized, time-dependent interventions. Across clinical domains, variability provides information beyond average responses. In oncology, variability patterns have been used to characterize tumor heterogeneity and identify clinically relevant subtypes (Chen et al., 2013); in diabetes, glucose variability complements mean glucose as an indicator of glycemic regulation and metabolic instability (Rodbard, 2018); and in physical activity, temporal variability has been associated with circadian organization and age (Xiao et al., 2015; Li et al., 2021). Despite its clinical relevance, biological variability is frequently summarized using scalar measures, such as daily standard deviations (Rodbard, 2018), which obscure how variability evolves throughout the day or in response to specific physiological stimuli. This limitation is particularly relevant for high-frequency biological signals, such as continuous glucose monitoring (CGM) trajectories, where both the magnitude and timing of variability may carry important physiological information. For example, two individuals may exhibit similar average postprandia glucose responses but difer substantially in the day-to-day variability of their glucose trajectories, with these diferences changing over the postprandial period (Figure 1). Characterizing such heterogeneity requires moving beyond conditional mean regression to explicitly model the conditional variance as a function of time and individual-level covariates, thereby capturing aspects of biological regulation that conventional summary measures may overlook.

AEGIS motivation. Our methodological development is motivated by the need to characterize time-dependent variability in postprandial glucose responses among adults without diabetes, using continuous glucose monitoring (CGM) data from the AEGIS study (Gude et al., 2017; Calvo-Malvar et al., 2025), while accounting for individual characteristics and meal composition. Postprandial glucose regulation reflects complex interactions among dietary composition, insulin sensitivity, pancreatic beta-cell function, and other physiological processes. Large-scale studies have documented substantial interindividual heterogeneity in glycemic responses to identical meals, providing a scientific basis for personalized nutrition approaches (Zeevi et al., 2015; Berry et al., 2020). More recently, detailed metabolic phenotyping has demonstrated that individual responses to diferent carbohydrate-rich foods are associated with underlying metabolic characteristics, including insulin resistance and beta-cell dysfunction (Wu et al., 2025). However, considerable variability may also arise within the same individual across repeated exposures to similar meals. Recent evidence has highlighted the limited reproducibility of postprandial glycemic responses to duplicate meals, raising questions about the reliability of personalized dietary recommendations derived from a small number of CGM observations (Hengist et al., 2025). Together, these findings motivate statistical approaches that distinguish expected postprandial glucose responses from their within-individual variability, while accounting for individual metabolic characteristics and diferences in meal composition.

![](images/47c74c834d87975b960e9a837568cd0b8137fdce26517b923536239cd3c2a79c.jpg)  
Figure 1: Postprandial glucose trajectories of two AEGIS participants after breakfast, lunch and dinner on five study days (D1–D5), on a five-minute grid from the start of each meal. Solid lines are the observed glucose and dashed lines the participant’s fitted mean for that meal. Shading is an approximate pointwise 95% prediction band with asymptotic approximation, the fitted mean ±1.96 times the fitted residual standard deviation; crosses mark observations outside it.

Although glycemic variability is increasingly recognized as an important dimension of glucose regulation, clinical assessments commonly rely on scalar summaries, such as the coeficient of variation and time-in-range metrics (Battelino et al., 2019). While these measures provide useful information about overall glycemic patterns, they do not directly characterize how the conditional variability of repeated postprandial trajectories evolves over the period following a meal. Functional data analysis (FDA) ofers a natural framework for studying high-resolution glucose trajectories (Gaynanova et al., 2022), and previous analyses of AEGIS postprandial responses have employed multilevel functional regression to investigate associations with individual and meal characteristics (Calvo-Malvar et al., 2025; Matabuena et al., 2026). However, existing approaches have primarily focused on conditional mean responses or summarized variability using scalar measures (Gude et al., 2017). Consequently, the time-dependent structure of within-individual glucose variability, and its association with metabolic and dietary characteristics, remains insuficiently characterized.

Goals and contributions. This paper addresses the lack of methods for estimating conditional variability in multilevel functional data. We propose a computationally eficient algorithm that estimates the conditional mean and then, in further steps, the conditional variability (heteroscedasticity) in multilevel functional models and provides uncertainty quantification for functional parameters. The model is a functional mixed-efects location–scale model, in which covariates and participant random functions act on both the mean and the log residual standard deviation of each trajectory (Section 2.1). The conditional mean and the conditional variability at a covariate profile are two parallel primary targets, each with its own time-varying covariate efects (Section 2.2). Our key idea is a three-step fitting strategy: (i) to estimate the conditional mean with a multilevel functional regression fitted score by score on a spline basis, (ii) to estimate the conditional variability by a regression of cross-fitted squared residuals on covariates, with participant random functions, and (iii) to correct this scale fit for the error in predicting each participant’s mean. Step (iii) addresses the main dificulty of variance estimation in multilevel functional data: when participants contribute few trajectories, the squared residuals also contain the error of the predicted participant mean, which pulls the estimated variance slopes towards zero and does not vanish as the number of participants grows (Section 2.4). The correction estimates this component, removes it through conditional refits of the scale model, and estimates the level of variability separately. To propagate uncertainty from all steps, we use a participant bootstrap that repeats every step and yields pointwise and simultaneous confidence bands for the conditional mean and conditional variance functional parameters.

We further demonstrate the practical relevance of our framework through an analysis of postprandial glucose trajectories from the AEGIS study, uncovering distinct time-varying associations between premeal glucose levels, carbohydrate intake, meal type, and conditional glucose variability. These associations would be systematically underestimated without the proposed bias correction. Finally, we introduce an individual-level functional $R ^ { 2 }$ that quantifies the model-based reduction in squared prediction error relative to a common population variance. Its substantial variation across participants reveals pronounced heterogeneity in the extent to which the model explains individual glucose variability.

Paper roadmap. Section 2 introduces the modeling framework and the associated estimation and inference procedures. Section 3 presents a detailed analysis of the AEGIS case study. Section 4 concludes with a discussion of the results, limitations, and future work. The Appendix gives estimation and inference details and implementation details of the case study.

## 2 Methodology

We model the conditional mean and variability of repeated functional trajectories with a multilevel location–scale model. Estimation fits the mean, estimates the scale from cross-fitted residuals and corrects it for error in the predicted participant mean efects; inference resamples participants and repeats every estimation step.

## 2.1 Data, notation and model

We observe N independent participants sampled from a common population. Participant $i \in [ N ] : = \{ 1 , . . . , N \}$ has available $n _ { i } \geq 2$ functional observations $Y _ { i j } ( \cdot ) \in \mathcal { H } , j = 1 , \ldots , n _ { i }$ , where H is a separable Hilbert space of real-valued functions defined on a common time domain T. A natural example that we assume along this paper is $\mathcal { H } = L ^ { 2 } ( \mathbb { T } )$ , the space of square-integrable functions, equipped with the inner product $\begin{array} { r } { \langle f , g \rangle _ { \mathcal { H } } = \int _ { \mathbb { T } } f ( t ) g ( t ) , } \end{array}$ dt and the induced norm $\| f \| _ { \mathcal H } = \{ \int _ { \mathbb { T } } f ^ { 2 } ( t ) d t \} ^ { 1 / 2 }$ . In our application, $\mathbb { T } = [ 0 , 1 8 0 ]$ represents the first 180 minutes following a meal. The trajectories are observed on a common discrete grid $\mathcal { T } = \{ t _ { 1 } , . . . , t _ { T } \} \subset \mathbb { T }$

Each functional observation is associated with a p-dimensional covariate vector $X _ { i j } = ( 1 , X _ { i j , - } ^ { \top } ) ^ { \top } \in \mathbb { R } ^ { p }$ , where the first component corresponds to the intercept. We write $X _ { i } = \left( X _ { i 1 } , \ldots , X _ { i n _ { i } } \right)$ for the collection of covariates associated with participant i. Covariates may vary across participants, across repeated trajectories within the same participant, or both.

Grid points and basis functions are indexed by $k = 1 , \dots , T$ and $\ell ,$ respectively, and $\begin{array} { r } { M = \sum _ { i = 1 } ^ { N } n _ { i } } \end{array}$ denotes the total number of observed trajectories. We use $Y _ { i j } ( \cdot )$ to denote the entire functional trajectory and $Y _ { i j } ( t )$ to denote its value at time t. For covariate vectors and coeficient functions, the subscript − indicates exclusion of the intercept component, whereas the subscript a identifies the ath component. The superscripts $\mu$ and ω distinguish quantities associated with the mean and scale models, respectively, including the spline bases $B _ { \ell } ^ { ( \mu ) } ( \cdot )$ and $B _ { \ell } ^ { \bar { ( \omega ) } } ( \cdot )$ and their respective dimensions $K _ { \mu }$ and $K _ { \omega }$ . Hats denote estimators, while the superscript (−f) indicates quantities computed excluding cross-fitting fold $f , \ ( r )$ denotes the rth scale-model refit $( r = 0$ for the initial fit), and ∗ indicates bootstrap quantities obtained by resampling participants. Finally, $I _ { n }$ denotes the $n \times n$ identity matrix, ${ \bf 1 } _ { n }$ the n-dimensional vector of ones, and ⊥⊥ statistical independence.

Suppose that the trajectories follow a function-on-scalar location–scale model:

$$
Y _ { i j } ( t ) = X _ { i j } ^ { \top } \beta ( t ) + u _ { i } ( t ) + \exp \{ X _ { i j } ^ { \top } \gamma ( t ) + v _ { i } ( t ) \} \epsilon _ { i j } ( t ) , \qquad i = 1 , \dots , N , \quad j = 1 , \dots , n _ { i } , \quad t \in { \mathcal { T } } ,\tag{1}
$$

where $\beta ( \cdot )$ and $\gamma ( \cdot )$ are coeficient functions governing the conditional mean and log residual standard deviation (SD), respectively. The participant-specific random efects $u _ { i } ( \cdot )$ and $v _ { i } ( \cdot )$ act on the mean and log residual SD and are shared across all trajectories of participant i. Both efects are centered, with $\mathbb { E } \{ u _ { i } ( t ) ^ { 2 } \} < \infty$ and $\mathbb { E } \{ \exp ( 2 v _ { i } ( t ) ) \} < \infty$ When the functional quantities are replaced by scalars, model (1) reduces to a mixed-efects location–scale model for repeated measurements (Hedeker et al., 2008; Hedeker and Nordgren, 2013; Lee and Nelder, 2006). In the absence of participant-specific random efects, it reduces to a location–scale regression model (Rigby and Stasinopoulos, 2005). Define

$$
\mu _ { i j } ( t ) = X _ { i j } ^ { \top } \beta ( t ) + u _ { i } ( t ) , \qquad \omega _ { i j } ( t ) = \exp \{ X _ { i j } ^ { \top } \gamma ( t ) + v _ { i } ( t ) \} ,
$$

as the participant-specific conditional mean and residual SD, respectively. Thus, $Y _ { i j } ( t ) = \mu _ { i j } ( t ) + \omega _ { i j } ( t ) \epsilon _ { i j } ( t )$ , where $\epsilon _ { i j } ( t )$ is a standardized error process. We impose the following assumptions:

(C1) $\mathbb { E } \{ \epsilon _ { i j } ( t ) \mid X _ { i } , u _ { i } , v _ { i } \} = 0$ and $\operatorname { V a r } \{ \epsilon _ { i j } ( t ) \mid X _ { i } , u _ { i } , v _ { i } \} = 1 ;$

(C2) $\epsilon _ { i 1 } ( \cdot ) , \ldots , \epsilon _ { i n _ { i } } ( \cdot )$ are conditionally independent given $( X _ { i } , u _ { i } , v _ { i } )$ ;

(C3) $\{ u _ { i } ( \cdot ) , v _ { i } ( \cdot ) \}$ is independent of $( n _ { i } , X _ { i } )$

The random efects capture participant-specific departures from the population-level mean and scale functions. For participants with the same covariates, a positive $u _ { i } ( t )$ indicates a higher expected glucose level at time t after a meal, whereas a positive $v _ { i } ( t )$ indicates greater residual variability around the participant-specific mean trajectory relative to the fixed-efect scale. Conditions (C1) and (C2) specify that, conditional on the covariates and participant efects, trajectories are independent, with conditional means $\mu _ { i j } ( \cdot )$ and residual SDs $\omega _ { i j } ( \cdot )$ . Thus, within-participant dependence across trajectories is induced by the shared random efects.

Condition (C3) assumes that the participant efects are independent of both the covariates and the number of observed trajectories, $n _ { i }$ (Seaman et al., 2014). This assumption allows the coeficient functions $\beta ( \cdot )$ and $\gamma ( \cdot )$ to characterize covariate associations with the conditional mean and log residual SD, as developed in Section 2.2. Independence from $n _ { i }$ is particularly relevant because participants contributing more trajectories receive greater weight in estimation; an association between cluster size and the random efects could therefore introduce bias. The mean and scale random efects may be correlated, as in scalar mixed-efects location–scale models (Hedeker et al., 2008). Importantly, these assumptions impose no restrictions on temporal dependence within individual trajectories.

## 2.2 Estimands

Our primary estimands characterize the conditional mean and within-participant variability of functional trajectories at a given covariate profile x. Specifically, we define the population-level mean function $m ( x , t ) = x ^ { \top } \beta ( t )$ and the participant-averaged residual variance

$$
\begin{array} { r l } & { \bar { q } ( x , t ) = \mathbb { E } \left[ \mathrm { V a r } \{ Y _ { i j } ( t ) \mid X _ { i j } , u _ { i } , v _ { i } \} \mid X _ { i j } = x \right] } \\ & { ~ = \mathbb { E } \{ \omega _ { i j } ^ { 2 } ( t ) \mid X _ { i j } = x \} } \\ & { ~ = \exp \{ 2 x ^ { \top } \gamma ( t ) \} \mathbb { E } \left[ \exp \{ 2 v _ { i } ( t ) \} \right] . } \end{array}\tag{2}
$$

The second equality follows from (C1), whereas the third uses (C3). The mean function $m ( x , t )$ represents the expected trajectory at covariate profile x, averaged over participant-specific efects. In contrast, $\bar { q } ( x , t )$ quantifies the residual

variability around participant-specific mean trajectories, averaged over participants with the same covariate profile.   
We cal $m ( x , \cdot )$ the reference mean, and $m ( x , \cdot )$ and $\boldsymbol { \bar { q } } ( \boldsymbol { x } , \cdot )$ together the reference moments at profile x.

Under (C1) and (C3), the total conditional variance admits the decomposition

$$
\operatorname { V a r } \{ Y _ { i j } ( t ) \mid X _ { i j } = x \} = \operatorname { V a r } \{ u _ { i } ( t ) \} + { \bar { q } } ( x , t ) ,
$$

separating between-participant heterogeneity from within-participant residual variability.

The non-intercept coeficient functions describe covariate associations with the mean and residual variability. Holding other covariates fixed, a one-unit increase in the ath covariate changes the conditional mean by $\beta _ { a } ( t )$ and multiplies the participant-specific residual SD by $\exp \{ \gamma _ { a } ( t ) \}$ . These efects describe conditional associations rather than causal relationships.

Let $\gamma _ { - } ( \cdot )$ denote the non-intercept coeficient functions in the scale model. Since

$$
\frac { 1 } { 2 } \log \bar { q } ( x , t ) = x ^ { \top } \gamma ( t ) + \frac { 1 } { 2 } \log \mathbb { E } \left[ \exp \{ 2 v _ { i } ( t ) \} \right] ,
$$

averaging over the participant-specific scale efects changes only the intercept, leaving the covariate slopes $\gamma _ { - } ( \cdot )$ unchanged. Accordingly, we report the overall level of within-participant variability through $\boldsymbol { \bar { q } } ( \boldsymbol { x } , \cdot )$ rather than through the intercept of the log-SD model. Appendix ?? further relates $\boldsymbol { \bar { q } } ( \boldsymbol { x } , \cdot )$ to the residual variance evaluated at $v _ { i } ( \cdot ) = 0$

## 2.3 Estimating the mean and the scale

## 2.3.1 Conditional mean

We represent the coeficient functions in a finite spline basis and estimate participant efects through score-wise linear mixed models, following functional mixed-model constructions (Guo, 2002; Greven and Scheipl, 2017) and basis-space functional mixed models that fit each basis coeficient separately (Morris and Carroll, 2006). Let $B _ { 1 } ^ { ( \mu ) } ( \cdot ) , \ldots , B _ { K _ { \mu } } ^ { ( \mu ) } ( \cdot )$ be a cubic B-spline basis for the mean model, made orthonormal on T by QR decomposition: with $B ^ { ( \mu ) } ( t ) =$ $\begin{array} { r } { \{ B _ { 1 } ^ { ( \mu ) } ( t ) , \dots , B _ { K _ { \mu } } ^ { ( \mu ) } ( t ) \} ^ { \top } , T ^ { - 1 } \sum _ { k = 1 } ^ { T } B ^ { ( \mu ) } ( t _ { k } ) B ^ { ( \mu ) } ( t _ { k } ) ^ { \top } = I _ { K _ { \mu } } } \end{array}$ . We approximate

$$
\beta ( t ) \approx \sum _ { \ell = 1 } ^ { K _ { \mu } } \theta _ { \ell } B _ { \ell } ^ { ( \mu ) } ( t ) , \qquad u _ { i } ( t ) \approx \sum _ { \ell = 1 } ^ { K _ { \mu } } a _ { i \ell } B _ { \ell } ^ { ( \mu ) } ( t ) ,
$$

with projection coeficients $\begin{array} { r } { \theta _ { \ell } = T ^ { - 1 } \sum _ { k = 1 } ^ { T } \beta ( t _ { k } ) B _ { \ell } ^ { ( \mu ) } ( t _ { k } ) \in \mathbb { R } ^ { p } } \end{array}$ and $\begin{array} { r } { a _ { i \ell } = T ^ { - 1 } \sum _ { k = 1 } ^ { T } u _ { i } ( t _ { k } ) B _ { \ell } ^ { ( \mu ) } ( t _ { k } ) } \end{array}$ . Projecting (1) in the same way gives exactly

$$
C _ { i j \ell } = X _ { i j } ^ { \top } \theta _ { \ell } + a _ { i \ell } + \xi _ { i j \ell } , \qquad i = 1 , \dots , N , \quad j = 1 , \dots , n _ { i } , \quad \ell = 1 , \dots , K _ { \mu } ,\tag{3}
$$

where $\begin{array} { r } { C _ { i j \ell } = T ^ { - 1 } \sum _ { k = 1 } ^ { T } Y _ { i j } ( t _ { k } ) B _ { \ell } ^ { ( \mu ) } ( t _ { k } ) } \end{array}$ is the projection score of trajectory j in direction ℓ and $\begin{array} { r } { \xi _ { i j \ell } = T ^ { - 1 } \sum _ { k = 1 } ^ { T } \omega _ { i j } ( t _ { k } ) \epsilon _ { i j } ( t _ { k } ) B _ { \ell } ^ { ( \mu ) } ( t _ { k } } \end{array}$ is the projected residual; basis approximation error enters only through the reconstruction in (4). For each ℓ, (3) is a random-intercept model, with $a _ { i \ell }$ shared by the trajectories of participant i and $\xi _ { i j \ell }$ of mean zero (Appendix ??). It is fitted by restricted maximum likelihood (REML) (Bates et al., 2015) as a working Gaussian mixed model with $a _ { i \ell } \sim N ( 0 , \tau _ { \ell } ^ { 2 } )$ and $\xi _ { i j \ell } \sim N ( 0 , \sigma _ { \ell } ^ { 2 } )$ ; cross-score covariance is not modeled in these fits. The fitted coeficients give

$$
\widehat { \beta } ( \cdot ) = \sum _ { \ell = 1 } ^ { K _ { \mu } } \widehat { \theta } _ { \ell } B _ { \ell } ^ { ( \mu ) } ( \cdot ) , \qquad \widehat { u } _ { i } ( \cdot ) = \sum _ { \ell = 1 } ^ { K _ { \mu } } \widehat { a } _ { i \ell } B _ { \ell } ^ { ( \mu ) } ( \cdot ) ,\tag{4}
$$

where $\widehat { a } _ { i \ell }$ is the shrinkage prediction of participant i’s efect from that participant’s trajectories (Harville, 1976), and $\widehat { \mu } _ { i j } ( t ) = X _ { i j } ^ { \top } \widehat { \beta } ( t ) + \widehat { u } _ { i } ( t )$ . Appendix ?? gives the basis construction, the derivation and properties of (3) and the REML fit.

## 2.3.2 Conditional scale

The residual $Y _ { i j } ( t ) - \widehat { \mu } _ { i j } ( t )$ of Section 2.3.1 is on average smaller in magnitude than the deviation $Y _ { i j } ( t ) - \mu _ { i j } ( t )$ because $Y _ { i j } ( \cdot )$ itself enters $\widehat { u } _ { i } ( \cdot )$ and pulls $\widehat { \mu } _ { i j } ( \cdot )$ towards $Y _ { i j } ( \cdot )$ . With three trajectories, for instance, each makes up a third of the average from which $\widehat { u } _ { i } ( \cdot )$ is predicted.

We therefore use cross-fitted residuals (Fan et al., 2012; Chernozhukov et al., 2018), which replace $\widehat { \mu } _ { i j } ( \cdot )$ by a prediction from a mean fit that does not use $Y _ { i j } ( \cdot )$ . The trajectories are split into folds $\mathcal { T } _ { 1 } , \dots , \mathcal { T } _ { F } , F = \operatorname* { m a x } _ { i } n _ { i }$ , each containing at most one trajectory per participant. For $( i , j ) \in \mathcal { T } _ { f }$ , refitting the mean model without fold f gives $\widehat { \mu } _ { i j } ^ { ( - f ) } ( t ) = X _ { i j } ^ { \top } \widehat { \beta } ^ { ( - f ) } ( t ) + \widehat { u } _ { i } ^ { ( - f ) } ( t )$ , where $\widehat { u } _ { i } ^ { ( - f ) } ( \cdot )$ is predicted from participant $i \mathrm { \ ' } _ { \mathrm { S } }$ other trajectories; the cross-fitted residual and its square are

$$
\begin{array} { r } { \hat { e } _ { i j } ^ { ( - f ) } ( t ) = Y _ { i j } ( t ) - \widehat { \mu } _ { i j } ^ { ( - f ) } ( t ) , \qquad Q _ { i j } ( t ) = \hat { e } _ { i j } ^ { ( - f ) } ( t ) ^ { 2 } . } \end{array}\tag{5}
$$

Each trajectory lies in exactly one fold, so $f$ is determined by $( i , j )$ ; we omit the superscript $( - f )$ from $Q _ { i j } ( t )$ and from later quantities computed from the cross-fitted fits.

We model the conditional mean of $Q _ { i j } ( \cdot )$ with a log-link Gamma regression, a standard approach to variance-function estimation from squared residuals (Davidian and Carroll, 1987):

$$
\eta _ { i j } ( t ) = \log \mathbb { E } \{ Q _ { i j } ( t ) \mid X _ { i j } , b _ { i } \} = X _ { i j } ^ { \top } \alpha ( t ) + b _ { i } ( t ) ,\tag{6}
$$

where $\alpha ( \cdot )$ contains the covariate efects on this log mean and $b _ { i } ( \cdot )$ is a participant random function. The Gamma distribution is a working assumption for estimating this mean. As in Section 2.3.1, both functions are expanded in an orthonormalized cubic B-spline basis $B _ { 1 } ^ { ( \omega ) } ( \cdot ) , \ldots , B _ { K _ { \omega } } ^ { ( \omega ) } ( \cdot )$

$$
\alpha ( t ) = \sum _ { \ell = 1 } ^ { K _ { \omega } } \zeta _ { \ell } B _ { \ell } ^ { ( \omega ) } ( t ) , \qquad b _ { i } ( t ) = \sum _ { \ell = 1 } ^ { K _ { \omega } } b _ { i \ell } B _ { \ell } ^ { ( \omega ) } ( t ) .
$$

The participant coeficients $b _ { i \ell }$ are penalized as random efects, with one penalty for each ℓ selected by fast REML, treating the grid points of a trajectory as independent in this working fit (Appendix ??). Like $\theta _ { \ell }$ , the coeficients $\zeta _ { \ell }$ are not penalized, so the smoothness of $\beta ( \cdot )$ and $\alpha ( \cdot )$ is set by $K _ { \mu }$ and $K _ { \omega } ,$ which are fixed before fitting.

If the mean-prediction error $\mu _ { i j } ( t ) - \widehat { \mu } _ { i j } ^ { ( - f ) } ( t )$ were zero, then $\mathbb { E } \{ Q _ { i j } ( t ) \mid X _ { i j } , v _ { i } \} = \omega _ { i j } ( t ) ^ { 2 }$ by (C1), so $\eta _ { i j } ( t )$ in (6) would equal log $\omega _ { i j } ( t ) ^ { 2 } = 2 \{ X _ { i j } ^ { \top } \gamma ( t ) + v _ { i } \dot { ( t ) } \}$ . Matching this with $X _ { i j } ^ { \top } \alpha ( t ) + b _ { i } ( t )$ gives $\alpha ( t ) = 2 \gamma ( t )$ and $b _ { i } ( t ) = 2 v _ { i } ( t )$ and hence the estimates

$$
\begin{array} { r } { \widehat { \gamma } ^ { ( 0 ) } ( t ) = \frac { 1 } { 2 } \widehat { \alpha } ^ { ( 0 ) } ( t ) , \qquad \widehat { v } _ { i } ^ { ( 0 ) } ( t ) = \frac { 1 } { 2 } \widehat { b } _ { i } ^ { ( 0 ) } ( t ) . } \end{array}\tag{7}
$$

However, the mean-prediction error is not zero, so these estimates are uncorrected; the superscript (0) marks them as the starting point of the correction in Section 2.4. This two-stage construction draws on separate dispersion modeling (Smyth, 1989) and functional and distributional regression (St¨ocker et al., 2021; Klein, 2024).

## 2.4 Scale correction for mean-prediction error

By (1) and (5), a cross-fitted residual is the error $\omega _ { i j } ( t ) \epsilon _ { i j } ( t )$ of the held-out trajectory plus the mean-prediction error $\Delta _ { i j } ( t ) = \mu _ { i j } ( t ) - \widehat { \mu } _ { i j } ^ { ( - f ) } ( t )$

$$
\begin{array} { r } { \widehat { e } _ { i j } ^ { ( - f ) } ( t ) = \omega _ { i j } ( t ) \epsilon _ { i j } ( t ) + \Delta _ { i j } ( t ) , \qquad \Delta _ { i j } ( t ) = \{ u _ { i } ( t ) - \widehat { u } _ { i } ^ { ( - f ) } ( t ) \} + X _ { i j } ^ { \top } \{ \beta ( t ) - \widehat { \beta } ^ { ( - f ) } ( t ) \} . } \end{array}
$$

As N grows, the second term of $\Delta _ { i j } ( \cdot )$ vanishes at rate $N ^ { - 1 / 2 }$ , apart from the spline approximation error of $\beta ( \cdot )$ , because ${ \widehat { \beta } } ^ { ( - f ) } ( \cdot )$ is estimated from all participants. The first term does not vanish when $n _ { i }$ stays bounded, because $\widehat { u } _ { i } ^ { ( - f ) } ( \cdot )$ is predicted from only the $n _ { i } - 1$ other trajectories of participant i. Under (C1) and (C2), given $( X _ { i } , u _ { i } , v _ { i } ) , \epsilon _ { i j } ( \cdot )$ has mean zero and is independent of the training trajectories that determine $\Delta _ { i j } ( \cdot )$ , so the cross term $2 \omega _ { i j } ( t ) \epsilon _ { i j } ( t ) \Delta _ { i j } ( t )$ has mean zero. Squaring and taking expectations gives (Appendix ??)

$$
\mathbb { E } \{ Q _ { i j } ( t ) \mid X _ { i } , v _ { i } \} = \omega _ { i j } ( t ) ^ { 2 } + \psi _ { i j } ( t ) + O ( N ^ { - 1 } ) , \qquad \psi _ { i j } ( t ) = \mathbb { E } [ \{ u _ { i } ( t ) - \widehat { u } _ { i } ^ { ( - f ) } ( t ) \} ^ { 2 } \mid X _ { i } , v _ { i } ] .\tag{8}
$$

Unlike the working model (6), which uses $X _ { i j }$ only, this expectation conditions on all of $X _ { i }$ , because $\widehat { u } _ { i } ^ { ( - f ) } ( \cdot )$ depends on the covariates of participant i’s other trajectories; it averages over $u _ { i } ( \cdot )$ and the training errors. On average, the squared residual therefore exceeds $\omega _ { i j } ( t ) ^ { 2 }$ by the prediction-error variance $\psi _ { i j } ( t )$

For a trajectory-level covariate, which changes $\omega _ { i j } ( t ) ^ { 2 }$ but leaves $\psi _ { i j } ( t )$ unchanged, the local log-variance slope is multiplied by $\dot { \omega _ { i j } } ( t ) ^ { 2 } / \{ \omega _ { i j } ( t ) ^ { 2 } + \psi _ { i j } ( t ) \} < 1$ , so the uncorrected slopes are pulled towards zero (Appendix ??). The distortion is largest for participants with few trajectories, whose efects are predicted least precisely. The correction estimates $\psi _ { i j } ( \cdot ) \ ( \mathrm { S t e p \ 1 } )$ , refits the scale model with this component removed to estimate the slopes (Step 2), and then estimates the level of $\boldsymbol { \bar { q } } ( \boldsymbol { x } , \cdot )$ (Step 3).

Step 1: estimate the prediction-error variance. Let $a _ { i } = ( a _ { i 1 } , \ldots , a _ { i K _ { \mu } } ) ^ { \top }$ and $\xi _ { i j } = ( \xi _ { i j 1 } , \ldots , \xi _ { i j K _ { \mu } } ) ^ { \top }$ collect the participant scores and projected residuals in (3). In the training set of fold f, participant i has $n _ { i } - 1$ trajectories, and the shrinkage factors of Appendix ?? give

$$
L _ { i } = \mathrm { d i a g } _ { \ell } \Big \{ \frac { \widehat { \tau } _ { \ell } ^ { 2 } } { \widehat { \tau } _ { \ell } ^ { 2 } + \widehat { \sigma } _ { \ell } ^ { 2 } / ( n _ { i } - 1 ) } \Big \} , \qquad u _ { i } ( t ) - \widehat { u } _ { i } ^ { ( - f ) } ( t ) \approx B ^ { ( \mu ) } ( t ) ^ { \top } \{ ( I - L _ { i } ) a _ { i } - L _ { i } \bar { \xi } _ { i } \} ,
$$

where $\bar { \xi } _ { i }$ is the average of the $\xi _ { i { j ^ { \prime } } }$ over these $n _ { i } - 1$ trajectories, and the approximation ignores the estimation error in ${ \widehat { \theta } } _ { \ell } ^ { ( - f ) }$ and the part of $u _ { i } ( \cdot )$ outside the mean basis (Appendix ??). The two terms are the error from shrinkage towards the population mean and the noise in the average of the training trajectories (Harville, 1976); by (C1) they are uncorrelated, so $\psi _ { i j } ( t )$ is approximately the sum of their variances. With $\widehat { \Sigma } _ { a }$ and $\widehat { \Sigma } _ { \xi }$ the moment estimates, from the training set, of Cov $\left( { { a } _ { i } } \right)$ and of the covariance of $\xi _ { i j }$ pooled over participants (Appendix ??), the working estimate of $\psi _ { i j } ( \cdot )$ is

$$
\begin{array} { r } { \widehat { \psi } _ { i j } ( t ) = \widehat { \psi } _ { i j } ^ { \mathrm { s h r } } ( t ) + \rho _ { i j } ( t ) \widehat { \psi } _ { i j } ^ { \mathrm { a v g } } ( t ) , } \end{array}\tag{9}
$$

where

$$
\widehat { \psi } _ { i j } ^ { \mathrm { s h r } } ( t ) = \pmb { B } ^ { ( \mu ) } ( t ) ^ { \top } ( I - L _ { i } ) \widehat { \Sigma } _ { a } ( I - L _ { i } ) \pmb { B } ^ { ( \mu ) } ( t ) , \qquad \widehat { \psi } _ { i j } ^ { \mathrm { a v g } } ( t ) = \frac { \pmb { B } ^ { ( \mu ) } ( t ) ^ { \top } L _ { i } \widehat { \Sigma } _ { \xi } L _ { i } \pmb { B } ^ { ( \mu ) } ( t ) } { n _ { i } - 1 }
$$

are computed once per fold. Because $\widehat { \Sigma } _ { \xi }$ is pooled, the factor $\rho _ { i j } ( t )$ rescales the averaging noise to participant i: it is the ratio of the fitted residual variance at t of participant i’s other trajectories to that of all trajectories, and it is the only part of (9) that changes with the scale fit. Appendix ?? defines $\rho _ { i j } ( \cdot )$ , and Appendix ?? states the approximation it involves.

Step 2: estimate the covariate slopes by conditional refitting. Subtracting $\widehat { \psi } _ { i j } ( t )$ from $Q _ { i j } ( t )$ could give negative responses, which the penalized Gamma fit with participant efects cannot use. We instead multiply $Q _ { i j } ( t )$ by an estimate of $\omega _ { i j } ( t ) ^ { 2 } / \{ \omega _ { i j } ( t ) ^ { 2 } + \psi _ { i j } ( t ) \}$ , which by (8) would turn the mean of $Q _ { i j } ( t )$ into $\omega _ { i j } ( t ) ^ { 2 }$ . Because this factor involves the unknown $\omega _ { i j } ( t ) ^ { 2 }$ , we compute it from the current scale fit, refit the scale model to the rescaled $Q _ { i j } ( \cdot )$ and repeat; the number of refits S is fixed in advance. Starting from the fitted predictor $\widehat { \eta } _ { i j } ^ { ( 0 ) } ( t ) = X _ { i j } ^ { \top } \widehat { \alpha } ^ { ( 0 ) } ( t ) + \widehat { b } _ { i } ^ { ( \bar { 0 } ) } ( t )$ of (6), for $r = 1 , \ldots , S$ , refit r fits the model of Appendix ??, with reselected penalties, to

$$
\widetilde { Q } _ { i j } ^ { ( r ) } ( t ) = Q _ { i j } ( t ) \frac { \exp \{ \widehat { \eta } _ { i j } ^ { ( r - 1 ) } ( t ) \} } { \exp \{ \widehat { \eta } _ { i j } ^ { ( r - 1 ) } ( t ) \} + \widehat { \psi } _ { i j } ^ { ( r - 1 ) } ( t ) } ,\tag{10}
$$

where exp $\{ \widehat { \eta } _ { i j } ^ { ( r - 1 ) } ( t ) \}$ is the fitted residual variance from fit $r - 1$ and $\hat { \psi } _ { i j } ^ { ( r - 1 ) } ( t )$ is (9) with $\rho _ { i j } ( t )$ computed from these fit ted variances; refit r gives $\widehat { \eta } _ { i j } ^ { ( r ) } ( t ) = X _ { i j } ^ { \top } \widehat { \alpha } ^ { ( r ) } ( t ) { + } \widehat { b } _ { i } ^ { ( r ) } ( t )$ . In an idealized version of the update (Appendix ??), exp $\{ \widehat { \eta } _ { i j } ^ { ( r ) } ( t ) \}$ decreases to $\omega _ { i j } ( t ) ^ { 2 }$ , and at each refit the error exp $\{ \widehat { \eta } _ { i j } ^ { ( r ) } ( t ) \} - \omega _ { i j } ( t ) ^ { 2 }$ is multiplied by at most $\psi _ { i j } ( t ) / \{ \omega _ { i j } ( t ) ^ { 2 } + \psi _ { i j } ( t ) \} < 1$ With $\alpha ( t ) = \{ \alpha _ { \mathrm { i n t } } ( t ) , \alpha _ { - } ( t ) ^ { \top } \} ^ { \top }$ partitioned like $X _ { i j }$ , the corrected slopes are $\begin{array} { r } { \widehat { \gamma } _ { - } ( t ) = \frac { 1 } { 2 } \widehat { \alpha } _ { - } ^ { ( S ) } ( t ) } \end{array}$

Step 3: estimate the participant-averaged variance level. The covariate part exp $\{ x ^ { \top } \widehat { \alpha } ^ { ( S ) } ( t ) \}$ of the refits estimates the residual variance of a participant with $v _ { i } ( \cdot ) = 0$ , and by $( 2 ) \ \bar { q } ( x , t )$ is this variance times E $\exp \{ 2 v _ { i } ( t ) \}$ , which does not depend on x. By (8) and (C3), $\mathbb { E } \{ Q _ { i j } ( t ) - \psi _ { i j } ( t ) \mid X _ { i } \} = \bar { q } ( X _ { i j } , t ) + O ( N ^ { - 1 } )$ . We therefore hold the slopes fixed and estimate from $Q _ { i j } ( t ) - \widehat { \psi } _ { i j } ^ { ( S ) } ( t )$ a separate intercept function $\bar { \alpha } _ { \mathrm { i n t } } ( \cdot )$ , the value of log $\bar { q } ( x , \cdot )$ at $x _ { - } = 0$ . In this level fit, with $\bar { \alpha } _ { \mathrm { i n t } } ( \cdot )$ expanded in the scale basis and $\bar { \eta } _ { i j } ( t ) = \bar { \alpha } _ { \mathrm { i n t } } ( t ) + X _ { i j , - } ^ { \top } \widehat { \alpha } _ { - } ^ { ( S ) } ( t )$ , its $K _ { \omega }$ coeficients solve

$$
\sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { n _ { i } } \sum _ { k = 1 } ^ { T } B _ { \ell } ^ { ( \omega ) } ( t _ { k } ) \left[ \frac { Q _ { i j } ( t _ { k } ) - \widehat { \psi } _ { i j } ^ { ( S ) } ( t _ { k } ) } { \exp \{ \bar { \eta } _ { i j } ( t _ { k } ) \} } - 1 \right] = 0 , \qquad \ell = 1 , \ldots , K _ { \omega } .\tag{11}
$$

This is the Gamma quasi-score (Wedderburn, 1974) used as an estimating equation, so individual negative values of $Q _ { i j } ( t _ { k } ) - \widehat { \psi } _ { i j } ^ { ( S ) } ( t _ { k } )$ are allowed. Appendix ?? gives the starting values of the refits and the solver for (11).

Corrected estimates. With $\widehat { \bar { \alpha } } _ { \mathrm { i n t } } ( \cdot )$ solving (11), the participant-averaged residual variance is estimated by

$$
\widehat { \bar { q } } ( x , t ) = \exp \{ \widehat { \bar { \alpha } } _ { \mathrm { i n t } } ( t ) + x _ { - } ^ { \top } \widehat { \alpha } _ { - } ^ { ( S ) } ( t ) \} .
$$

Participant-specific scale curves use the last refit, $\widehat { v } _ { i } ( t ) = \frac { 1 } { 2 } \widehat { b } _ { i } ^ { ( S ) } ( t )$ and $\begin{array} { r } { \widehat { \omega } _ { i j } ( t ) = \exp [ \frac { 1 } { 2 } \{ X _ { i j } ^ { \top } \widehat { \alpha } ^ { ( S ) } ( t ) + \widehat { b } _ { i } ^ { ( S ) } ( t ) \} ] } \end{array}$ ; the level-fit intercept, which already averages over the $v _ { i } ( \cdot )$ , is not substituted into them (Appendix ??).

## 2.5 Participant-bootstrap inference

Participants are independent, and every estimate in Sections 2.3 and 2.4 is built from participant-level contributions. Drawing N participants with replacement therefore mimics drawing a new cohort from the same population, while keeping each participant’s trajectories, their within-participant dependence and their number intact (Davison and Hinkley, 1997, Section 3.8). Field and Welsh (2007) and Field et al. (2010) study such cluster bootstraps for clustered and multilevel data. Each replicate keeps each trajectory’s original fold label and repeats all other lines of Algorithm1: the full-data mean fit, mean cross-fitting, Gamma penalty selection, the S refits and the level fit. The bootstrap distribution therefore includes the variability these steps add.

For an estimand curve $\phi ( \cdot )$ , such as a coeficient function or log $\boldsymbol { \bar { q } } ( \boldsymbol { x } , \cdot )$ , let $\widehat { \phi } ^ { * ( b ) } ( \cdot )$ be its estimate from replicate $b = 1 , \dots , B$ and $\mathrm { s e } ( t )$ the SD of $\widehat { \phi } ^ { * ( 1 ) } ( t ) , \ldots , \widehat { \phi } ^ { * ( B ) } ( t )$ , the bootstrap standard error at t. The 95% simultaneous band and the statistic that calibrates it are

$$
\widehat { \phi } ( t ) \pm c \mathrm { s e } ( t ) , \quad t \in \mathcal { T } , \qquad Z _ { b } ^ { * } = \operatorname* { m a x } _ { t \in \mathcal { T } } \frac { | \widehat { \phi } ^ { * ( b ) } ( t ) - \widehat { \phi } ( t ) | } { \mathrm { s e } ( t ) } ,
$$

where c is the empirical 0.95 quantile of $Z _ { 1 } ^ { * } , \ldots , Z _ { B } ^ { * }$ (Fan and Zhang, 2000; Chang et al., 2017; Montiel Olea and Plagborg-Møller, 2019; Cui et al., 2022). Because $Z _ { b } ^ { * }$ is the largest deviation of replicate b from the estimate, in standard errors, 95% of the replicate curves lie entirely inside the band. Each band is simultaneous over the grid of one curve only. A pointwise interval at t uses instead the 0.95 quantile of the standardized deviations at t alone. We report both: a pointwise interval concerns a single time, whereas a band supports statements about the whole curve, such as the times at which it excludes zero. For $\beta ( \cdot )$ , the bands cover its spline projection, the target that the score means identify under (C1) and (C3) (Appendix ??). For $\boldsymbol { \bar { q } } ( \boldsymbol { x } , \cdot )$ , work with log $\boldsymbol { \bar { q } } ( \boldsymbol { x } , \cdot )$ and exponentiate the limits. A whole-curve test of $\phi ( t ) = 0$ for all $t \in \mathcal T$ compares the largest distance of the estimate from zero with $Z _ { 1 } ^ { * } , \ldots , Z _ { B } ^ { * }$

$$
Z _ { \mathrm { o b s } } = \operatorname* { m a x } _ { t \in \mathcal { T } } \frac { | \widehat { \phi } ( t ) | } { \mathrm { s e } ( t ) } , \qquad p = \frac { 1 + \# \{ b : Z _ { b } ^ { * } \geq Z _ { \mathrm { o b s } } \} } { B + 1 } \geq \frac { 1 } { B + 1 } .
$$

When testing multiple coeficient functions, such as the non-intercept terms in a model, we adjust the resulting p-values using the Benjamini–Hochberg (BH) procedure (Benjamini and Hochberg, 1995). Appendix $? ?$ provides further details on the treatment of duplicated participants and failed fits in bootstrap replicates, the quantile definition, and the lower bound imposed on $\mathrm { s e } ( t )$

## 2.6 Software implementation

All analyses were performed in R version 4.6.1 (R Core Team, 2026). Cubic B-splines were generated with splines::bs and normalized by QR decomposition. The score-wise mean models and the penalized Gamma scale models were fitted with custom R functions that use the structure of the model: with the orthonormal basis, each curve mode separates into score-wise problems, and each participant efect involves only that participant’s trajectories. The mean fits minimize the profiled REML criterion of lme4::lmer (Bates et al., 2015). The Gamma fits use the penalized quasilikelihood iterations (Breslow and Clayton, 1993) and fREML criterion of mgcv::bam with method $\begin{array} { r l } { { } = } & { { } " \underline { { \sf { I R E M L } } } ^ { \prime \prime } } \end{array}$ (Wood et al., 2015; Wood, 2017), evaluated exactly rather than with the covariate discretization of discrete = TRUE (Wood et al., 2017).

The prediction-error covariances, conditional refits, level fit and participant bootstrap were implemented in custom R functions. Bootstrap refits were run in parallel, with a separate random-number seed for each attempt, so that results do not depend on the number of workers. In the analysis of Section 3, Algorithm1 fits the penalized Gamma model (6) four times: once initially and once in each of the three conditional refits of Step 2 in Section 2.4 $( S = 3 )$ . On these data, one such fit took on average 0.36 seconds (SD 0.02) with our functions and 178.62 seconds (SD 1.30) with mgcv::bam (version 1.9-4), even with its faster approximate option discrete = TRUE. Both were timed over 10 single-thread runs on a MacBook Pro with an Apple M5 chip and 16 GB of memory. The 500 bootstrap refits therefore need 2,000 such fits, which at these speeds take about 12 minutes with our functions and about 99 hours with mgcv::bam.

## 3 AEGIS case study

## 3.1 Data and analysis

The A Estrada Glycation and Inflammation Study (AEGIS) investigated blood glucose and its relation to inflammation and obesity in the general adult population (Matabuena et al., 2026). Its baseline examination took place in A Estrada, Galicia, northwestern Spain, between August 2012 and March 2015 (Calvo-Malvar et al., 2025). Adults aged 18 years or older were drawn as a stratified random sample from the Spanish National Health System registry (Gude et al., 2017; Matabuena et al., 2026). The study was approved by the Clinical Research Ethics Committee of Galicia (code 2012/025), and all participants gave informed consent. Of the 1,516 participants examined, 581 completed blinded CGM monitoring for up to seven days and, over the same days, recorded their meals in a food diary that a dietitian reviewed. Of these 581 participants, 516 did not have diabetes (Matabuena et al., 2026). Among them, we analyzed the $N = 3 6 3$ who had complete data and at least three recorded meals of each type (breakfast, lunch and dinner), 5,237 meals in total (9–15 per participant). The method itself needs fewer meals (Algorithm1); we required three of each type so that each participant’s summaries in Section 3.4 rest on at least three meals of each type.

The glucose response of a person to the same meal can difer substantially between diferent days and conditions, which limits how well one response predicts the next (Hengist et al., 2025). Because each participant recorded repeated meals in free-living conditions, the expected postmeal course of glucose can be separated from this meal-to-meal variation. We therefore study how meal composition and participant characteristics are associated with the expected course and with this variation. Specifically, we address five questions. (i) Does the association of carbohydrate with the mean and the meal-to-meal variability of glucose difer between breakfast, lunch and dinner? (ii) How is premeal glucose associated with the mean and the variability? (iii) Is HbA1c associated with both the mean and the variability? (iv) How does the variability change after a meal and between meal types? (v) Is the individual $R ^ { 2 }$ similar across participants?

In the notation of Section 2.1, j indexes the meals of participant i, and the fitted model (1) specifies

$$
\begin{array} { r } { \mathbb { E } \{ Y _ { i j } ( t ) \mid X _ { i j } , u _ { i } , v _ { i } \} = X _ { i j } ^ { \top } \beta ( t ) + u _ { i } ( t ) , \qquad \frac { 1 } { 2 } \log { \mathrm { V a r } \{ Y _ { i j } ( t ) \mid X _ { i j } , u _ { i } , v _ { i } \} } = X _ { i j } ^ { \top } \gamma ( t ) + v _ { i } ( t ) , } \end{array}
$$

with the same covariates in both parts,

$$
\begin{array} { r } { X _ { i j } = ( 1 , \mathrm { a g e } _ { i } , \mathrm { B M } _ { i } , \mathrm { H b A l c } _ { i } , \mathrm { s e x } _ { i } , \mathrm { c a l } _ { i j } , Y _ { i j } ( - 5 ) , I _ { i j , \mathrm { L } } , I _ { i j , \mathrm { D } } , \mathrm { c a r b } _ { i j } I _ { i j , \mathrm { B } } , \mathrm { c a r b } _ { i j } I _ { i j , \mathrm { L } } , \mathrm { c a r b } _ { i j } I _ { i j , \mathrm { D } } ) ^ { \top } , } \end{array}
$$

where $Y _ { i j } ( - 5 )$ is premeal glucose, $\mathrm { c a l } _ { i j }$ and $\mathrm { c a r b } _ { i j }$ are the calories (kcal) and carbohydrate (g) of meal j, and sex is 0 for female and 1 for male participants. Age, BMI, HbA1c and sex describe the participant, and the other covariates describe the meal. Table 1 summarizes the covariates.

Carbohydrate intake is a primary dietary determinant of postprandial glucose, whereas total caloric intake reflects the overall energy content of a meal, including contributions from fat and protein. HbA1c reflects average glycemic exposure over the preceding months and is widely used in the diagnosis of diabetes and prediabetes (American Diabetes Association, 2020). Premeal glucose represents the initial condition of the metabolic system from which the postprandial glucose response evolves, influencing the subsequent glucose trajectory (Matabuena et al., 2026). Age, sex, BMI, and meal timing have also been associated with postprandial glycemic responses in this population (Calvo-Malvar et al., 2025). Finally, meal-type indicators allow glucose responses and their associations with carbohydrate intake to difer across breakfast, lunch, and dinner.

The indicator $I _ { i j , \mathrm { B } }$ is 1 if meal j is a breakfast and 0 otherwise; $I _ { i j , \mathrm { L } }$ and $I _ { i j , \mathrm { D } }$ do the same for lunch and dinner. Breakfast is the reference meal, so only $I _ { i j , \mathrm { L } }$ and $I _ { i j , \mathrm { D } }$ appear on their own in $X _ { i j } .$ , and their coeficients are diferences from breakfast. The carbohydrate amount carb enters only through the last three entries of $X _ { i j }$ , its products with the three indicators, so each meal type has its own carbohydrate slope. The diference between two meal types therefore depends on the carbohydrate amount, and we report it at 50 g with the other covariates fixed.

The fixed efects are the coeficient functions $\beta ( \cdot )$ and $\gamma ( \cdot )$ , whose 11 non-intercept curves are shown in Figures 2 and 3. The random efects are the participant mean and scale efects $u _ { i } ( \cdot )$ and $v _ { i } ( \cdot )$ , shared by all meals of participant i. Given $X _ { i j } , u _ { i }$ and $v _ { i } ,$ the only random part of $Y _ { i j } ( t )$ is the standardized error $\epsilon _ { i j } ( t )$ , which varies from meal to meal. The variance in the second equation therefore describes variation between meals of the same participant, and $\mathrm { V a r } \{ u _ { i } ( t ) \}$ does not enter it. Variation between participants with the same covariates enters through $u _ { i } ( \cdot )$ and $v _ { i } ( \cdot )$ Their distribution is not specified, and by (C3) it does not depend on the covariates. Section 3.3 reports their estimated SDs. By (C1) and (C3), the reference mean $m ( x , t ) = \mathbb { E } \{ Y _ { i j } ( t ) \mid X _ { i j } = x \}$ is the expected glucose t minutes after a meal with covariates x, and the participant-averaged residual variance $\bar { q } ( \bar { x } , t ) = \mathbb { E } [ \{ Y _ { i j } ( t ) - \mu _ { i j } ( t ) \} ^ { 2 } \mid X _ { i j } = x ]$ is the expected squared deviation of such a meal from its participant’s mean curve.

Trajectories were analyzed on a grid $\tau$ of $T = 3 7$ points from 0 to 180 minutes after meal start, at five-minute intervals. We used $K _ { \mu } = 8$ mean and $K _ { \omega } = 6$ scale basis functions and applied the estimation procedure in Sections 2.3–2.4, with $F = 1 5$ meal-wise cross-fitting folds formed by randomly ordering each participant’s meals and $S = 3$ conditional refits. Efects are reported in the natural increments shown in the figures. Carbohydrate slopes describe an increase of 10 g at fixed calories.

Pointwise intervals, simultaneous bands and whole-curve tests were computed as in Section 2.5 from $B \ : = \ : 5 0 0$ participant-bootstrap refits, with BH adjustment within each model’s 11 non-intercept coeficient curves. Appendix ?? gives implementation and contrast details.

Table 1: Characteristics of the 363 participants and their 5,237 meals. Continuous variables are summarized as mean (SD); female participants are summarized as count (percentage). Participant characteristics are calculated once per participant, and meal covariates are calculated within each meal type.
<table><tr><td></td><td colspan="3">(a) Participant characteristics Characteristic</td></tr><tr><td>BMI</td><td> $\mathrm { A g e \ ( y e a r s ) }$   $( \mathrm { k g / m ^ { 2 } } )$ </td><td></td><td>47.2 (14.2)</td></tr><tr><td>HbA1c (%)</td><td></td><td colspan="2">28.1 (5.0)</td></tr><tr><td>Female, n (%)</td><td></td><td colspan="2">5.35 (0.33) 230 (63.4)</td></tr><tr><td>(b) Meals</td><td></td><td></td><td></td></tr><tr><td>Meal type</td><td>Meals</td><td>Calories (kcal)</td><td>Carbohydrate (g)</td></tr><tr><td>Breakfast</td><td>1727</td><td>318.1 (200.1)</td><td>44.8 (29.1)</td></tr><tr><td>Lunch</td><td>1755</td><td>879.6 (425.6)</td><td>74.0 (46.0)</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Dinner</td><td>1755</td><td>649.2 (389.9)</td><td>59.2 (40.3)</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

## 3.2 Covariate associations in conditional mean and conditional variability

Ten of the 11 mean coeficient curves and seven of the 11 log-SD curves had BH-adjusted $p < 0 . 0 5$ (Figures 2 and 3, Table 2).

Question (i) concerns carbohydrate and meal type. Carbohydrate was associated with higher mean glucose and greater residual SD at all three meals. The slopes difered between meals only in contrasts involving dinner, and only after 100 minutes. In the mean model, the carbohydrate bands excluded zero at 5–125 minutes for breakfast, 0–165 for lunch and 0–180 for dinner. In the log-SD model, they excluded zero at 0–95 minutes for breakfast, 0–60 and 75–95 for lunch, and 5–150 for dinner.

We also compared the three carbohydrate slopes directly, using paired contrasts within each bootstrap refit. The dinner-minus-breakfast mean slope was positive at 135–180 minutes and passed BH adjustment over the three meal-pair comparisons $( p _ { \mathrm { B H } } = 0 . 0 1 2 )$ . The dinner-minus-lunch slope was positive at 110–135 and 155–170 minutes in the mean model $( p _ { \mathrm { B H } } = 0 . 0 4 5 )$ and at 100–140 minutes in the log-SD model $( p _ { \mathrm { B H } } = 0 . 0 4 8 )$ . The other comparisons did not pass the adjusted 0.05 threshold (Appendix Table ??).

Higher premeal glucose (question ii) was associated with higher mean glucose over the whole 180 minutes and with greater residual SD at 0–95 minutes. Premeal glucose is likely related to the participant efects, contrary to (C3), so these associations may partly reflect diferences between participants.

For HbA1c (question iii), the evidence was weak. HbA1c was associated with the mean only at 170–180 minutes $( p _ { \mathrm { B H } } = 0 . 0 3 3 )$ . Its association with the residual SD did not pass BH adjustment $\left( p _ { \mathrm { B H } } = 0 . 3 2 9 \right)$ . Participants did not have diabetes, so HbA1c varied little (SD 0.33 percentage points; Table 1).

Among the other covariates, age had a positive mean association at 40–105 minutes, with an isolated finding at 0 minutes. BMI had a positive mean association at 90–180 minutes. With carbohydrate held fixed, calories had a negative association with the mean at 0–55 minutes and with residual SD at 0–20 minutes. Recorded sex passed BH adjustment in neither model, nor did age and BMI in the log-SD model. Section 3.3 describes the meal-type contrasts.

The directions of the mean associations agree with those reported by Calvo-Malvar et al. (2025), who applied multilevel functional regression to 514 participants without diabetes from the same study but only focus on conditional mean estimation. In both analyses, glucose increased with meal carbohydrate (glycemic load in their analysis), age, BMI and HbA1c, and the carbohydrate association lasted longer after lunch and dinner than after breakfast.

![](images/dd44c0ee41f89b39309095c781e1a3720be7dde8cd471a9828970d8e183593ca.jpg)  
Figure 2: Mean-glucose associations over postmeal time and meal type. Panels show 11 conditional-mean coeficient efects in mg/dL, with covariate increments in the labels. Dark shading shows pointwise nominal 95% intervals and light shading simultaneous bands from 500 participant-bootstrap refits. Meal contrasts compare lunch and dinner with breakfast at 50 g carbohydrate; female (sex coded 0) is the recorded-sex reference. Calories and carbohydrate adjust for each other. The three carbohydrate panels share a vertical scale, as do the two meal-contrast panels.

![](images/dd7ff85b8788e1aa13616acb39f13f760c5950e6e1440752e2d19899e9d4bc69.jpg)  
Figure 3: Residual-SD associations over postmeal time. Efects are displayed as $1 0 0 \{ \exp ( d _ { a } \gamma _ { a } ( t ) ) - 1 \}$ percent change in residual SD, with increments $d _ { a }$ in the panel labels; efects and band limits are calculated on the log-SD scale before transformation. Dark and light shading show pointwise nominal 95% intervals and simultaneous bands from 500 participant-bootstrap refits. Meal contrasts use breakfast at $\mathrm { 5 0 ~ g }$ carbohydrate, and female (sex coded 0) is the recorded-sex reference. The three carbohydrate panels share a vertical scale, as do the two meal contrasts; each simultaneous band covers one coeficient curve.

Table 2: AEGIS whole-curve tests and exclusion windows. BH adjustment is applied within each model’s 11 coeficient curves. Values of $p _ { \mathrm { B H } }$ below 0.05 are in bold. Windows are in minutes and use each curve’s nominal 95% simultaneous band, with signs in parentheses; they are not jointly simultaneous across coeficients. Covariate increments follow Figures 2–3, and meal contrasts are evaluated at 50 g carbohydrate.
<table><tr><td></td><td colspan="2">Conditional mean,  $\beta _ { a } ( t )$ </td><td colspan="2">Log residual SD,  $\gamma _ { a } ( t )$ </td></tr><tr><td>Coefficient</td><td>pBH</td><td>Windows</td><td>pBH</td><td>Windows</td></tr><tr><td>Age</td><td>0.003</td><td>0 (+); 40–105 (+)</td><td>0.074</td><td>none</td></tr><tr><td>BMI</td><td>0.017</td><td>90–180 (+)</td><td>0.117</td><td>none</td></tr><tr><td>HbA1c</td><td>0.033</td><td>170-180 (+)</td><td>0.329</td><td>none</td></tr><tr><td>Recorded sex</td><td>0.124</td><td>none</td><td>0.707</td><td>none</td></tr><tr><td>Calories</td><td></td><td>0.003 0–55 (−)</td><td></td><td>0.004 0–20 (−)</td></tr><tr><td>Premeal glucose</td><td>0.003</td><td>0-180 (+)</td><td></td><td>0.004 0–95 (+)</td></tr><tr><td>Lunch vs breakfast</td><td>0.003</td><td>5–25 (−); 60–180 (+)</td><td></td><td>0.038 90–140 (+)</td></tr><tr><td>Dinner vs breakfast</td><td>0.003</td><td>5-35 (−); 60-180 (+)</td><td></td><td>0.004 65–180 (+)</td></tr><tr><td>Carbohydrate: breakfast</td><td>0.003</td><td>5-125 (+)</td><td>0.004 0–95 (+)</td><td></td></tr><tr><td>Carbohydrate: lunch</td><td></td><td>0.003 0–165 (+)</td><td></td><td>0.015 0–60 (+); 75–95 (+)</td></tr><tr><td>Carbohydrate: dinner</td><td></td><td>0.003 0-180 (+)</td><td></td><td>0.004 5-150 (+)</td></tr></table>

(a) Conditional mean

## 3.3 Reference mean and variability

To compare the fitted trajectories on a common covariate profile, we fixed the continuous covariates at their sample means, sex at female and carbohydrate at 50 g. Figure 4 shows, at this profile x, the reference mean $m ( x , t )$ and the participant-averaged residual variance $\boldsymbol { \bar { q } } ( \boldsymbol { x } , t )$ defined in Section 2.2. The mean peaks occurred at 40, 45 and 60 minutes for breakfast, lunch and dinner, respectively, with peak values of 118.1–118.5 mg/dL. At this profile, the diferences between the lunch or dinner curves and the breakfast curve are the meal contrasts tested in Table 2: the mean contrast for $m ( x , \cdot )$ , and twice the log-SD contrast for log $\boldsymbol { \bar { q } } ( \boldsymbol { x } , \cdot )$ . In the mean, lunch and dinner were below breakfast at 5–25 and 5–35 minutes, respectively, and above breakfast at 60–180 minutes. Dinner fell most slowly after its peak.

Turning to question (iv), $\boldsymbol { \bar { q } } ( \boldsymbol { x } , t )$ rose from about 6 to 257–271 $\mathrm { ( m g / d L ) ^ { 2 } }$ by 45 minutes, then fell after breakfast and lunch but not after dinner. The value at meal start is low because premeal glucose is a covariate. By 180 minutes, $\bar { q } ( x , t )$ was about 40% below its peak after breakfast and lunch, while the dinner curve stayed between 253 and 276 $\mathrm { ( m g / d L ) ^ { 2 } }$ from 45 minutes on. In the log-SD meal contrasts, lunch had greater residual SD than breakfast at 90–140 minutes, and dinner at 65–180 minutes. No lunch-versus-dinner contrast was computed at this profile.

Participants difered substantially in residual variability: averaged over 5 to 180 minutes, the model-implied SD of $v _ { i } ( t )$ was 0.35, so at the same covariates, participants one SD apart difered in residual SD by a factor of about 1.4. Averaged over the grid, the participant mean efect $u _ { i } ( t )$ had SD 6.9 mg/dL and accounted for 12–22% of the variance of the projection scores $C _ { i j \ell }$ , so most of the conditional variance was meal-to-meal variation within a participant.

![](images/733e07c511ae64a15f4f43ce3939b7d69743fbc7c25bc3c84cdac0e652e5c4b4.jpg)

(b) Conditional variability  
![](images/d20bc2bbcccbc804639fe5f62b2e6bb292f369c5d9801e2ac2eb313c793706e7.jpg)  
Figure 4: Dinner has a broader late elevation in the reference mean and residual variance. Part (a) shows $m ( x , t )$ and part (b) shows $\boldsymbol { \bar { q } } ( \boldsymbol { x } , t )$ at the sample covariate means, female sex and $\mathrm { 5 0 ~ g }$ carbohydrate. The participant-averaged residual variance excludes between-participant mean variation. The three meal panels within each part share a vertical scale. Dark and light shading show pointwise nominal 95% intervals and simultaneous bands from 500 participant-bootstrap refits; variance limits are constructed on the log scale and exponentiated.

(a) Paired values per participant

## 3.4 Individual contribution of participant mean efects

An individual $R ^ { 2 }$ summarizes how much model-based squared error remains for participant i relative to a common population variance. A meal j of participant i can be predicted by the reference mean $n ( X _ { i j } , t ) = X _ { i j } ^ { \top } \beta ( t )$ , which uses only the covariates, or by the participant-specific mean $\mu _ { i j } ( t ) = m ( X _ { i j } , t ) + u _ { i } ( t )$ . Averaging their expected squared errors over 0–180 minutes gives

$$
\begin{array} { r l } & { \quad R _ { i , \mathrm { f i x e d } } ^ { 2 } ( X _ { i j } ) = 1 - \displaystyle \frac { 1 } { D } \frac { 1 } { 1 8 0 } \mathbb { E } \| Y _ { i j } ( \cdot ) - m ( X _ { i j } , \cdot ) \| _ { \mathcal { H } } ^ { 2 } , } \\ & { \quad R _ { i , \mathrm { p a r t i c i p a n t } } ^ { 2 } ( X _ { i j } ) = 1 - \displaystyle \frac { 1 } { D } \frac { 1 } { 1 8 0 } \mathbb { E } \| Y _ { i j } ( \cdot ) - m ( X _ { i j } , \cdot ) - u _ { i } ( \cdot ) \| _ { \mathcal { H } } ^ { 2 } , } \\ & { \quad \quad D = \displaystyle \frac { 1 } { 1 8 0 } \int _ { \mathbb { T } } \mathrm { V a r } \{ Y _ { i j } ( t ) \} d t . } \end{array}\tag{12}
$$

In the first two lines, the expectation is over the standardized error $\epsilon _ { i j } ( \cdot )$ , which varies from meal to meal, with $X _ { i j } , \ u _ { i } ( \cdot )$ and $\boldsymbol { v } _ { i } ( \cdot )$ fixed, so by (C1) the expected squared errors at time t are $\omega _ { i j } ( t ) ^ { 2 } + u _ { i } ( t ) ^ { 2 }$ and $\omega _ { i j } ( t ) ^ { 2 }$ . The variance in the third line is over participants and meals. We estimate D by the average of $\begin{array} { r } { \frac { 1 } { 1 8 0 } \| Y _ { i j } ( \cdot ) ^ { \circ } - \bar { Y } ( \cdot ) \| _ { \mathcal { H } } ^ { 2 } } \end{array}$ over all $M = 5 { , } 2 3 7$ curves, where $\bar { Y } ( t )$ is their pointwise mean, which gives $D = 3 6 1 . 5 ~ ( \mathrm { m g / d L } ) ^ { 2 }$ , an SD of 19.0 $\mathrm { m g / d L }$ ; integrals use normalized trapezoidal weights on T . We report the averages over the $n _ { i }$ meals of participant $i ,$ written $R _ { i , \mathrm { f i x e d } } ^ { 2 }$ and $R _ { i , \mathrm { p a r t i c i p a n t } } ^ { 2 }$ . Both use all covariates and $v _ { i } ( \cdot )$ and difer only by the paired contribution of $u _ { i } ( \cdot )$ $\Delta R _ { i } ^ { 2 } = R _ { i , \mathrm { p a r t i c i p a n t } } ^ { 2 } - \dot { R } _ { i , \mathrm { f i x e d } } ^ { 2 } \stackrel { . } { > } 0$ . They are estimated by substituting $\widehat { u } _ { i } ( \cdot )$ and $\widehat { \omega } _ { i j } ( \cdot )$ , and are structural model-based summaries rather than measures of observed prediction accuracy.

The individual $R ^ { 2 }$ varied widely between participants, so the answer to question (v) is no. The median $R _ { i } ^ { 2 }$ was 0.477 (IQR 0.173–0.639) for the reference mean and 0.548 (0.294–0.702) for the participant-specific mean (Figure 5a). The median paired contribution was 0.054 (IQR 0.024–0.117), and 109 of 363 participants had $\Delta R _ { i } ^ { 2 } > 0 . 1$ (panel b). This median, 5.4% of D, is not the diference between the two group medians. There were 57 negative values of $R _ { i , \mathrm { f i x e d } } ^ { 2 }$ and 38 of $R _ { i , \mathrm { p a r t i c i p a n t } } ^ { 2 } { \mathrm { : } }$ , where the model-based squared error exceeded the common baseline.

(b) Contribution of mean effects  
![](images/913aa3bf5a11245b86ed44b1b40317b1284f4488c7fa35468fb97386f152c291.jpg)

![](images/a64436eee48a07abe9d825d53baa802b024711fe5773c00cdf7bbd8bb81f9654.jpg)  
Figure 5: Participant mean efects reduce model-based squared error by a median of $5 . 4 \%$ of the common population variance $\mathrm { ( } D = 3 6 1 . 5 ~ \mathrm { ( m g / d L ) ^ { 2 } } )$ . (a) Each point pairs $R _ { i , \mathrm { f i x e d } } ^ { 2 }$ and $R _ { i , \mathrm { p a r t i c i p a n t } } ^ { \mathrm { 2 } }$ of one participant; axes have the same range and the solid line is the identity line. (b) Distribution of the paired contribution $\Delta R _ { i } ^ { 2 }$ , with its median marked. All 363 participants and negative values are retained. Both values retain the fitted participant scale efect and treat participant efects as known; the nonnegative contribution follows from the definition. No individual uncertainty intervals are shown.

## 4 Discussion

Previous work has used multilevel functional regression to study the conditional mean of postprandial glucose trajectories in the AEGIS cohort, leaving the analysis of conditional variability for future research (Matabuena et al., 2026). In this paper, we extend this approach by jointly modeling the conditional mean and variance of repeated functional observations. To our knowledge, this is the first multilevel functional location–scale model that allows covariates and participant-specific random efects to influence both the mean and residual variability of the trajectories.

The AEGIS results show the value of studying both the mean and variability of postprandial glucose trajectories. At the reference covariate profile, the participant-averaged residual variance reached $\mathrm { 2 5 7 - 2 7 6 ~ ( m g / d L ) ^ { 2 } }$ , corresponding to a standard deviation of about $1 6 { - } 1 7 \ \mathrm { m g / d L }$ . This is similar in magnitude to the average postmeal glucose increase of approximately 15 mg/dL. Higher carbohydrate intake was associated with greater glucose variability at breakfast, lunch, and dinner. We also found that residual variability was higher after dinner than after breakfast, particularly from 65 minutes after the meal onwards. Variability also difered considerably between participants. For the same covariate values, a one-standard-deviation diference in the participant-specific scale efect corresponded to an approximately 1.4-fold diference in residual standard deviation. These findings represent associations rather than causal efects. Finally, the contribution of participant-specific mean efects to individual $R ^ { 2 }$ varied across participants. These $R ^ { 2 }$ values describe how well the model explains individual trajectories relative to a common population variance and should not be interpreted as measures of predictive performance.

Several limitations should be acknowledged. Our interpretation relies on condition (C3), which may be restrictive because premeal glucose is likely associated with participant-specific efects in AEGIS. Although the model allows the mean and scale random efects to be correlated, as in conventional mixed-efects location–scale models (Hedeker et al., 2008), the proposed prediction-error correction does not explicitly account for this correlation. Moreover, the residual variability observed in AEGIS may reflect several sources beyond physiological variation, including sensor measurement error, unrecorded meal components, and inaccuracies in carbohydrate intake or meal timing. Some of these sources may increase with meal size and contribute to the observed associations between carbohydrate intake and residual variability. The proposed correction also relies on an approximation of the prediction-error variance through the rescaling factor $\rho _ { i j } ( \cdot )$ and does not account for participant-level variation outside the selected mean basis. The confidence bands do not correct for smoothing or projection bias (Hall and Horowitz, 2013).

Future work could extend the proposed correction to account for correlation between participant-specific mean and scale efects and explore other aspects of the conditional distribution beyond the mean and variance. More generally, the proposed framework provides a way to study both the expected response and its variability in repeated functional data, with potential applications to other wearable sensors and longitudinal physiological measurements.

## References

American Diabetes Association. 2. classification and diagnosis of diabetes: Standards of medical care in diabetes—2020. Diabetes Care, 43(Supplement 1):S14–S31, 2020. ISSN 0149-5992. doi: 10.2337/dc20-S002. URL https://care.diabetesjournals. org/content/43/Supplement\_1/S14.

Daniel Backenroth, Jef Goldsmith, Michelle D. Harran, Juan C. Cortes, John W. Krakauer, and Tomoko Kitago. Modeling motor learning using heteroscedastic functional principal components analysis. Journal of the American Statistical Association, 113(523):1003–1015, 2018. doi: 10.1080/01621459.2017.1379403. URL https://doi.org/10.1080/01621459.2017.1379403. PMID: 30416231.

Douglas Bates, Martin M¨achler, Ben Bolker, and Steve Walker. Fitting linear mixed-efects models using lme4. Journal of Statistical Software, 67(1):1–48, 2015. doi: 10.18637/jss.v067.i01. URL https://doi.org/10.18637/jss.v067.i01.

Tadej Battelino, Thomas Danne, Richard M. Bergenstal, et al. Clinical targets for continuous glucose monitoring data interpretation: Recommendations from the international consensus on time in range. Diabetes Care, 42(8):1593–1603, 2019. doi: 10.2337/dci19-0028.

Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological), 57(1):289–300, 1995. doi: 10.1111/j.2517-6161.1995.tb02031. x.

Sarah E. Berry, Ana M. Valdes, David A. Drew, et al. Human postprandial responses to food and potential for precision nutrition. Nature Medicine, 26(6):964–973, 2020. doi: 10.1038/s41591-020-0934-0.

N. E. Breslow and D. G. Clayton. Approximate inference in generalized linear mixed models. Journal of the American Statistical Association, 88(421):9–25, 1993. doi: 10.1080/01621459.1993.10594284.

Mar Calvo-Malvar, Oscar Lado-Baleato, Ana Cao R´ıos, Cristina Porca Fern´andez, Alfonso Ben´ıtez-Calvo, Carmen Fernandez-<sup>´</sup>

Merino, Juan S´anchez-Castro, Robert Wagner, Marcos Matabuena, and Francisco Gude. Age, sex, BMI, meal timing, and glycemic response to meal glycemic load. JAMA Network Open, 8(9):e2533193, 2025. doi: 10.1001/jamanetworkopen.2025.33193.

Chung Chang, Xuejing Lin, and R. Todd Ogden. Simultaneous confidence bands for functional regression models. Journal of Statistical Planning and Inference, 188:67–81, 2017. doi: 10.1016/j.jspi.2017.03.002. URL https://doi.org/10.1016/j.jspi. 2017.03.002.

Guanhua Chen, Patrick F. Sullivan, and Michael R. Kosorok. Biclustering with heterogeneous variance. Proceedings of the National Academy of Sciences, 110(30):12253–12258, 2013. doi: 10.1073/pnas.1304376110. URL https://www.pnas.org/doi/ abs/10.1073/pnas.1304376110.

Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1):C1–C68, 2018. doi: 10.1111/ectj.12097. URL https://doi.org/10.1111/ectj.12097.

Ciprian M. Crainiceanu, Jef Goldsmith, Andrew Leroux, and Erjia Cui. Functional Data Analysis with R. Chapman and Hall/CRC, Boca Raton, FL, 2024. doi: 10.1201/9781003278726.

Erjia Cui, Andrew Leroux, Ekaterina Smirnova, and Ciprian M. Crainiceanu. Fast univariate inference for longitudinal functional models. Journal of Computational and Graphical Statistics, 31(1):219–230, 2022. doi: 10.1080/10618600.2021.1950006.

Erjia Cui, Ruonan Li, Ciprian M. Crainiceanu, and Luo Xiao. Fast multilevel functional principal component analysis. Journal of Computational and Graphical Statistics, 32(2):366–377, 2023. doi: 10.1080/10618600.2022.2115500.

Marie Davidian and Raymond J. Carroll. Variance function estimation. Journal of the American Statistical Association, 82 (400):1079–1091, 1987. doi: 10.1080/01621459.1987.10478543.

Anthony C. Davison and David V. Hinkley. Bootstrap Methods and their Application. Cambridge University Press, Cambridge, 1997.

Chong-Zhi Di, Ciprian M Crainiceanu, Brian S Cafo, and Naresh M Punjabi. Multilevel functional principal component analysis. The Annals of Applied Statistics, 3(1):458–488, 2009. doi: 10.1214/08-AOAS206.

Jianqing Fan and Wenyang Zhang. Simultaneous confidence bands and hypothesis testing in varying-coeficient models. Scandinavian Journal of Statistics, 27(4):715–731, 2000. doi: 10.1111/1467-9469.00218. URL https://doi.org/10.1111/ 1467-9469.00218.

Jianqing Fan, Shaojun Guo, and Ning Hao. Variance estimation using refitted cross-validation in ultrahigh dimensional regression. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 74(1):37–65, 2012. doi: 10.1111/j.1467-9868.2011. 01005.x.

Chris A. Field and Alan H. Welsh. Bootstrapping clustered data. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(3):369–390, 2007. doi: 10.1111/j.1467-9868.2007.00593.x. URL https://doi.org/10.1111/j.1467-9868. 2007.00593.x.

Christopher A. Field, Zhen Pang, and Alan H. Welsh. Bootstrapping robust estimates for clustered data. Journal of the American Statistical Association, 105(492):1606–1616, 2010. doi: 10.1198/jasa.2010.tm09541.

Irina Gaynanova, Naresh Punjabi, and Ciprian Crainiceanu. Modeling continuous glucose monitoring (CGM) data during sleep. Biostatistics, 23(1):223–239, 2022. doi: 10.1093/biostatistics/kxaa023.

N. Gill and Donald Hedeker. Fast estimation of mixed-efects location-scale regression models. Statistics in Medicine, 42(9): 1430–1444, 2023. doi: 10.1002/sim.9679.

Sonja Greven and Fabian Scheipl. A general framework for functional regression modelling. Statistical Modelling, 17(1-2):1–35, 2017. doi: 10.1177/1471082X16681317.

Francisco Gude, Pablo D´ıaz-Vidal, Cintia R´ua-P´erez, Manuela Alonso-Sampedro, Carmen Fern´andez-Merino, Jes´us Rey-Garc´ıa, Carmen Cadarso-Su´arez, Marcos Pazos-Couselo, Jos´e M. Garc´ıa-L´opez, and Arturo Gonzalez-Quintela. Glycemic variability and its association with demographics and lifestyles in a general adult population. Journal of diabetes science and technology, 11(4):780–790, 2017. doi: 10.1177/1932296816682031.

Wensheng Guo. Functional mixed efects models. Biometrics, 58(1):121–128, 2002. doi: 10.1111/j.0006-341X.2002.00121.x.

Peter Hall and Joel Horowitz. A simple bootstrap method for constructing nonparametric confidence bands for functions. The Annals of Statistics, 41(4):1892–1921, 2013. doi: 10.1214/13-AOS1137.

David Harville. Extension of the Gauss-Markov theorem to include the estimation of random efects. The Annals of Statistics, 4 (2):384–395, 1976. doi: 10.1214/aos/1176343414.

Donald Hedeker and Rachel Nordgren. MIXREGLS: A program for mixed-efects location scale analysis. Journal of Statistical Software, 52(12):1–38, 2013. doi: 10.18637/jss.v052.i12.

Donald Hedeker, Robin J. Mermelstein, and Hakan Demirtas. An application of a mixed-efects location scale model for analysis of ecological momentary assessment (EMA) data. Biometrics, 64(2):627–634, 2008. doi: 10.1111/j.1541-0420.2007.00924.x.

Aaron Hengist, Jude Anthony Ong, Katherine McNeel, Juen Guo, and Kevin D. Hall. Imprecision nutrition? intraindividual variability of glucose responses to duplicate presented meals in adults without diabetes. The American Journal of Clinical Nutrition, 121(1):74–82, 2025. doi: 10.1016/j.ajcnut.2024.10.007.

Nadja Klein. Distributional regression for data analysis. Annual Review of Statistics and Its Application, 11:321–346, 2024. doi: 10.1146/annurev-statistics-040722-053607.

Youngjo Lee and John A. Nelder. Double hierarchical generalized linear models (with discussion). Journal of the Royal Statistical Society: Series C (Applied Statistics), 55(2):139–185, 2006. doi: 10.1111/j.1467-9876.2006.00538.x.

Andrew Leroux, Ciprian Crainiceanu, and Julia Wrobel. Fast Generalized Functional Principal Components Analysis, June 2023. URL http://arxiv.org/abs/2305.02389. arXiv:2305.02389 [stat].

Jingen Li, Virend K Somers, Francisco Lopez-Jimenez, Junrui Di, and Naima Covassin. Demographic characteristics associated with circadian rest-activity rhythm patterns: a cross-sectional study. International Journal of Behavioral Nutrition and Physical Activity, 18(1):107, 2021. doi: 10.1186/s12966-021-01174-z.

Marcos Matabuena and Ciprian M. Crainiceanu. Multilevel functional distributional models with applications to continuous glucose monitoring in diabetes clinical trials. The Annals of Applied Statistics, 20(1):476–495, March 2026. doi: 10.1214/ 26-AOAS2139. URL https://doi.org/10.1214/26-AOAS2139.

Marcos Matabuena, Marta Karas, Sherveen Riazati, Nick Caplan, and Philip R. Hayes. Estimating knee movement patterns of recreational runners across training sessions using multilevel functional regression models. The American Statistician, 77(2): 169–181, 2023. doi: 10.1080/00031305.2022.2105950. URL https://doi.org/10.1080/00031305.2022.2105950.

Marcos Matabuena, Joseph Sartini, and Francisco Gude. Beyond scalar metrics: Functional data analysis of postprandial continuous glucose monitoring in the AEGIS study. BMC Medical Research Methodology, 26:39, 2026. doi: 10.1186/ s12874-025-02748-2.

Jos´e Luis Montiel Olea and Mikkel Plagborg-Møller. Simultaneous confidence bands: Theory, implementation, and an application to SVARs. Journal of Applied Econometrics, 34(1):1–17, 2019. doi: 10.1002/jae.2656.

Jefrey S. Morris and Raymond J. Carroll. Wavelet-based functional mixed models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 68(2):179–199, 2006. doi: 10.1111/j.1467-9868.2006.00539.x.

R Core Team. R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria, 2026. URL https://www.R-project.org/. Version 4.6.1.

Philippe Rast and Emilio Ferrer. A mixed-efects location scale model for dyadic interactions. Multivariate Behavioral Research, 53(5):756–775, 2018. doi: 10.1080/00273171.2018.1477577.

Robert A. Rigby and D. Mikis Stasinopoulos. Generalized additive models for location, scale and shape. Journal of the Royal Statistical Society: Series C (Applied Statistics), 54(3):507–554, 2005. doi: 10.1111/j.1467-9876.2005.00510.x. URL https://doi.org/10.1111/j.1467-9876.2005.00510.x.

David Rodbard. Glucose variability: a review of clinical applications and research developments. Diabetes Technology & Therapeutics, 20(S2):S2–5–S2–15, 2018. doi: 10.1089/dia.2018.0092.

Shaun Seaman, Menelaos Pavlou, and Andrew Copas. Review of methods for handling confounding by cluster and informative cluster size in clustered data. Statistics in Medicine, 33(30):5371–5387, 2014. doi: 10.1002/sim.6277.

Gordon K Smyth. Generalized linear models with varying dispersion. Journal of the Royal Statistical Society: Series B (Methodological), 51(1):47–60, 1989. doi: 10.1111/j.2517-6161.1989.tb01747.x.

Almond St¨ocker, Sarah Brockhaus, Sophia Anna Schafer, Benedikt von Bronk, Madeleine Opitz, and Sonja Greven. Boosting functional response models for location, scale and shape with an application to bacterial competition. Statistical Modelling, 21 (5):385–404, 2021. doi: 10.1177/1471082X20917586.

Thomas Y Sun and Daniel R Kowal. Ultra-eficient mcmc for bayesian longitudinal functional data analysis. Journal of Computational and Graphical Statistics, 34(1):34–46, 2025. doi: 10.1080/10618600.2024.2362227.

R. W. M. Wedderburn. Quasi-likelihood functions, generalized linear models, and the Gauss–Newton method. Biometrika, 61 (3):439–447, 1974. doi: 10.1093/biomet/61.3.439.

Donald R. Williams, Daniel R. Zimprich, and Philippe Rast. A Bayesian nonlinear mixed-efects location scale model for learning. Behavior Research Methods, 51(5):1968–1986, 2019. doi: 10.3758/s13428-019-01255-9.

Simon N Wood. Generalized additive models: an introduction with R. Chapman and Hall/CRC, 2 edition, 2017. doi: 10.1201/9781315370279.

Simon N. Wood, Yannig Goude, and Simon Shaw. Generalized additive models for large data sets. Journal of the Royal Statistical Society: Series C (Applied Statistics), 64(1):139–155, 2015. doi: 10.1111/rssc.12068.

Simon N. Wood, Zheyuan Li, Gavin Shaddick, and Nicole H. Augustin. Generalized additive models for gigadata: Modeling the U.K. black smoke network daily data. Journal of the American Statistical Association, 112(519):1199–1210, 2017. doi: 10.1080/01621459.2016.1195744. URL https://doi.org/10.1080/01621459.2016.1195744.

Brian Ping-Huan Wu and Donald Hedeker. Fast estimation and valid statistical inference for mixed-efect location-scale models using variational inference. Statistics in Medicine, 45(13–14):e70640, 2026. doi: 10.1002/sim.70640.

Yue Wu, Ben Ehlert, Ahmed A. Metwally, et al. Individual variations in glycemic responses to carbohydrates and underlying metabolic physiology. Nature Medicine, 31(7):2232–2243, 2025. doi: 10.1038/s41591-025-03719-2.

Luo Xiao, Lei Huang, Jennifer A Schrack, Luigi Ferrucci, Vadim Zipunnikov, and Ciprian M Crainiceanu. Quantifying the

lifetime circadian rhythm of physical activity: a covariate-dependent functional approach. Biostatistics, 16(2):352–367, 2015. doi: 10.1093/biostatistics/kxu045.

David Zeevi, Tal Korem, Niv Zmora, et al. Personalized nutrition by prediction of glycemic responses. Cell, 163(5):1079–1094, 2015. doi: 10.1016/j.cell.2015.11.001.