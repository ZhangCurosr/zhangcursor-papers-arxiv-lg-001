# A Structural FHMM for Interpretable Disease Trajectories in T2DM

Alessandro Mari<sup>1</sup>, Ekaterina Krymova<sup>2</sup>, Guillaume Obozinski<sup>2</sup>, Maria Luisa Marques de Sa Faquetti<sup>3</sup>, Adrian Martinez de la Torre<sup>3</sup>, and Andrea Burden<sup>3</sup>

<sup>1</sup>Swiss Data Science Center (SDSC), Ecole Polytechnique F´ed´erale de Lausanne (EPFL), Switzerland <sup>2</sup>Swiss Data Science Center (SDSC), ETH Z¨urich, Switzerland

<sup>3</sup>Pharmacoepidemiology Group, Institute of Pharmaceutical Sciences, Zurich, Switzerland

August 24, 2026

## Abstract

In this work, we propose a structural variant of the Factorial Hidden Markov Model (FHMM) for the analysis of disease trajectories in patients with Type 2 diabetes mellitus (T2DM). The model represents a patient’s latent health state as a combination of multiple independent, simultaneously evolving components, associated with comorbidities and lab results. This structured latent representation facilitates the identification of clinically meaningful patient states and clustering of common disease trajectories. We evaluate the proposed approach using The IQVIA Medical Research Data incorporating data from THIN, a Cegedim database of anonymized electronic health records (EHR), identifying patients with a first-ever prescription for a non-insulin antidiabetic drug (NIAD) between January 2006 and December 2019. The model identifies multiple clinically coherent latent components corresponding to known patterns of diabetes-related complications and reveals heterogeneous progression pathways, including distinct microvascular-dominant and multi-organ trajectories associated with elevated comorbidity burden and mortality. These results demonstrate that the proposed framework captures meaningful longitudinal structure in EHR data and provides interpretable insights into the evolution of T2DM and its comorbidities.

## 1 Background

Type 2 diabetes mellitus (T2DM) is a chronic metabolic disorder characterized by elevated blood glucose levels resulting from insulin resistance and inadequate insulin secretion. It represents a major global health concern and is frequently accompanied by a range of comorbidities, which substantially worsen quality of life and increase the risk of morbidity and mortality [1]. The presence and progression of such comorbidities contribute to the heterogeneity of T2DM trajectories, making disease management and risk stratification particularly challenging. The inference of a patient’s underlying health state from longitudinal clinical observations, together with information on T2DM and its associated comorbidities, can support the identification of disease progression patterns and emerging health risks. Leveraging retrospective electronic health records (EHRs) to characterize these evolving health states ofers the potential to improve disease understanding, stratify patients into clinically meaningful subgroups, and inform timely interventions.

In this study, we analyze longitudinal observations from EHRs associated with T2DM and multiple comorbidities in order to infer patients’ latent health states and characterize common patterns of disease evolution. To this end, we propose a statistical model designed to capture the multi-dimensional and dynamic nature of patient health, with a particular focus on interpretability of the latent representation and its clinical relevance.

Specifically, we consider a structural variant of the Factorial Hidden Markov Model (FHMM) [2], which enables modeling the dynamics of a multi-dimensional patient health state through several independent latent components evolving simultaneously over time. Given longitudinal clinical observations, the proposed model infers the latent configuration of these components, providing a structured representation of disease progression and associated health risks. In contrast to a standard Hidden Markov Model (HMM), which represents patient health using a single latent state, the FHMM framework allows the patient’s overall condition to be approximated as a combination of a small number of evolving latent processes. This structure increases modeling flexibility and facilitates interpretation, as individual latent chains can be associated with distinct aspects of disease evolution or comorbidity burden.

By using retrospective information on patients’ health indicators and disease history, the proposed model enables the identification of clinically meaningful latent states and the analysis of transitions between them. The resulting latent trajectories can be used to cluster patients according to common progression patterns, characterize typical disease pathways, and assess the evolution of comorbidityrelated risks over time.

Related work. Tracking and modeling comorbidities from longitudinal health observations in patients with T2DM is crucial for understanding disease heterogeneity and supporting efective disease management. A wide range of statistical and machine learning approaches have been proposed to analyze large-scale health data and uncover temporal patterns associated with disease progression and risk. HMM and their generalizations are among the most widely used probabilistic approaches for modeling disease dynamics in healthcare applications [3, 4, 5, 6]. Extensions such as mixtures of Input–Output Hidden Markov Models have been proposed to capture more complex temporal dependencies [7]. Nonparametric latent feature models, including approaches based on the Indian Bufet Process, have also been explored for comorbidity analysis, particularly in the context of psychiatric disorders [8].

More recently, deep generative models have been applied to longitudinal clinical data. In [9], the authors propose a variational autoencoder-based framework to learn latent representations that jointly generate clinical measurements and medical concepts. While efective, their approach requires manual assignment of latent dimensions to specific medical concepts. In contrast, our model automatically learns a structured factorization of the latent space by construction. Moreover, the latent representation in our approach is discrete and subject to monotonicity constraints, which enhances interpretability and facilitates the analysis of disease trajectories.

Methods addressing the joint evolution of multiple diseases have also been proposed. For example, [10] introduces a bivariate copula-based hidden process to model the interaction between two diseases using population-level time series. In the context of T2DM, machine learning approaches have been applied to model disease progression [11], while clustering-based analyses have been used to study patterns of comorbidities [12]. Our work complements these approaches by providing a structured, interpretable latent-state model adapted to the analysis of complex, multi-comorbidity disease trajectories.

## 2 Methods

## 2.1 Data

The IQVIA Medical Research Database UK (IMRD-UK) incorporates data supplied by The Health Improvement Network (THIN), a Cegedim database of anonymized electronic health records generated from the daily records of General Practitioners (GPs). The study protocol and use of the data were reviewed and approved by the Scientific Review Committee (SRC, reference number 20SR062).

The dataset includes patients from participating primary care practices who have provided general consent for their data to be used for research purposes. Consent may be withdrawn at any time, in which case the individual’s data are removed from the dataset. Secondary care providers, such as ambulatory care or specialists, provide additional feedback on diagnoses to the GPs.

Generally, the database comprises routinely collected data from clinical practice, including patient diagnoses, symptoms, prescriptions, referrals, immunizations, laboratory test results (including glycated hemoglobin [HbA1c]), selected behavioral factors (body mass index [BMI], alcohol use, and smoking status), and demographic characteristics. As with most electronic healthcare databases, data are generated during routine care rather than for research purposes, resulting in unequally spaced observations and incomplete capture of certain variables. Lifestyle factors such as BMI, smoking status, and alcohol consumption are subject to missingness. In addition, diagnoses are recorded only when they are among the main reasons for a visit, laboratory tests are captured only when ordered; and while medication prescriptions are identified, prescription fills and adherence are not known.

## 2.2 Cohort description

We identified all adult patients (aged 18+) with a first-ever prescription of a non-insulin antidiabetic drug (NIAD) between January 1, 2006, and December 31, 2019. The index date was defined as the date of the first NIAD prescription. To ensure identification of new users, patients were required to have at least one year of valid data prior to the index date. We excluded patients without any laboratory measurement for variables of interest (see Table 1) within the 4-years prior to the index date after data aggregation (see Section 2.2.1).

## 2.2.1 Data preprocessing

![](images/a3115cb8d3368b16b9cffd0d4c5be1ad7293f6d82dfb3c78a7ee467ba66b0cfd.jpg)  
Figure 1: Graphical representation of the preprocessing steps.

To harmonize the dataset and prepare it for modeling, we applied the following data preparation steps. A summary of these steps is provided in Figure 1.

Comorbidities and Laboratory Measurements: The comorbidities and laboratory variables included in the analysis are listed in Table 1. Comorbidities were identified at or prior to the index date and during follow-up using read codes recorded at each patient’s clinical encounters. Only chronic conditions were considered. Accordingly, a comorbidity c was defined as present at time t if any corresponding read code appeared during the relevant six-month interval. Once identified, comorbidities were assumed to persist over time, reflecting their chronic or recurrent nature; thus, the indicator for comorbidity c remained set to 1 following its first occurrence. In the base model, drugs were only used in modeling the initial state. A detailed discussion of the efects of drugs used in the transition matrix is provided in Appendix C.

The laboratory values are standardized to consistent units (see Table 1), and outliers are removed using predefined thresholds (see Appendix A). To address irregular measurement timing and partial missingness inherent to routine clinical data, laboratory measurements were aggregated into six-month intervals, with the median value used when multiple measurements were available within a given interval. Patients were excluded if any required laboratory variable had no recorded measurements during the four years preceding NIAD initiation. This step was necessary to enable forward imputation of missing values during the post-NIAD period.

Additionally, to separate the timing of doctors’ decisions regarding diagnoses and medication prescriptions, we used the comorbidity status (presence or absence) and lab tests shifted forward in time relative to the prescriptions. This approach helps avoid issues related to six-month aggregation, where diagnoses might be influenced by medication prescriptions rather than the patient’s actual health state.

To further preserve the intended temporal ordering between patient health status and clinical decisionmaking, comorbidity indicators and laboratory measurements were shifted forward in time relative to medication prescriptions. This approach mitigates potential reverse-causation artifacts introduced by six-month aggregation, whereby diagnoses or laboratory results recorded shortly before any treatment initiation could otherwise appear contemporaneous with (or influenced by) drug exposure.

Clinical Stratification: Patient observations were stratified by age and sex to account for clinically meaningful heterogeneity. Age was determined at the index date and categorized into two groups: patients aged 64 years or younger and those aged 65 years or older. This stratification reflects diferences in diabetes progression and the burden of age-related comorbidities across the life course [13]. Addition ally, analyses were stratified by sex to capture known diferences in disease prevalence and comorbidity profiles, such as osteoporosis, between male and female patients [14].

<table><tr><td>Comorbidities</td><td>Lab Values</td><td>Drugs (with ATC code)</td></tr><tr><td>• Stroke</td><td>• Glycated hemoglobin (Hb A1c) in %</td><td>• Diabetes Drugs</td></tr><tr><td>• Myocardial infarction (MI)</td><td>• Low-density Lipopro- tein (LDL) cholesterol in</td><td>• Antithrombotic agents (B01)</td></tr><tr><td>• Congestive heart failure (CHF)</td><td>mmol/L • High-density Lipopro-</td><td>• Cardiac therapy (C01) • Antihypertensives (C02)</td></tr><tr><td>• Peripheral vascular dis- ease (PVD)</td><td>tein (HDL) cholesterol in mmol/L</td><td>• Diuretics (C03)</td></tr><tr><td>• Diabetic retinopathy (DR)</td><td>• Triglycerides (TG) in mmol/L</td><td>• Beta blocking agents (C07) • Calcium channels blockers</td></tr><tr><td>• Hypertension (HT)</td><td>• Blood pressure (Sys- tolic/Diastolic) in mmHg</td><td>(C08)</td></tr><tr><td>• Chronic kidney disease (CKD)</td><td>• Body Mass Index (BMI) in</td><td>• ACE inhibitors (C09) • Statins (C10)</td></tr><tr><td>• Osteoporosis</td><td>kg/m² • Glomerular filtration rate</td><td>• Corticosteroids (H02)</td></tr><tr><td>• Chronic obstructive pulmonary disease</td><td>(GFR) in mL/min/1.73m²</td><td>• Drugs for treatment of bone diseases (M05)</td></tr><tr><td>(COPD) • Chronic liver disease (CLD)</td><td></td><td></td></tr></table>

Table 1: List of variables that are included in the model.

## 2.3 Structural FHMM

Our modeling approach is based on Factorial Hidden Markov Model (FHMM) [2], which is an extension of the hidden Markov chain, where the output is conditioned not on one hidden Markov chain with M states, but on K independent hidden Markov chains. This allows to model the time series depending on several independent latent processes (see Figure 2). In terms of comorbidities and lab-values progression modeling, the FHMM approach translates into the assumption that there exist K hidden health state components, which are independent and that are jointly influencing the output. Although the independence assumption may appear unrealistic, since body systems rarely function in isolation, the FHMM can be understood as an independent basis decomposition of the dynamics, where several hidden processes evolve separately and jointly generate the observed comorbidities and lab values. This view highlights FHMM as a structured approximation: the latent chains act as abstract basis processes that capture key aspects of disease progression.

We assume that the evolution of comorbidity states and laboratory test measurements, denoted by

![](images/2ce3b29b06f8ad48bc8889130d358b8de5dd51c8ed5df2162ed48ecfe85b1802.jpg)  
Figure 2: Directed graphical model of a FHMM with $M = 3$ hidden chains. Arrows represent causal relationships and gray nodes are observed variables. White nodes are not observed. The variables $X _ { t }$ are the drugs taken at time t that influence the patient health state $H _ { t } \ { \stackrel { \mathrm { d e f } } { = } } \ ( H _ { t } ^ { ( 1 ) } , \ldots , H _ { t } ^ { ( M ) } )$ . The results of the lab tests $Y _ { t } ^ { \mathrm { L A B } }$ and the set of comorbidities $Y _ { t } ^ { \mathrm { C O M } }$ are direct consequences of the health state $H _ { t }$ at time t.

$Y _ { t } .$ , follows the generative process defined by

$$
H _ { 1 } ^ { ( m ) } \mid X _ { 1 } \sim \pi ^ { ( m ) } ( \cdot \mid X _ { 1 } ; \theta ^ { \pi } ) ,
$$

$$
m = 1 , \ldots , M ,\tag{1}
$$

$$
H _ { t } ^ { ( m ) } \mid H _ { t - 1 } ^ { ( m ) } , X _ { t } \sim P ^ { ( m ) } ( \cdot \mid H _ { t - 1 } ^ { ( m ) } , X _ { t } ; \theta ^ { P } ) ,
$$

$$
m = 1 , \ldots , M ,\tag{2}
$$

$$
Y _ { t } \mid H _ { t } \sim p ( \cdot \mid H _ { t } ; \eta ) ,\tag{3}
$$

where $H _ { t } \ { \stackrel { \mathrm { d e f } } { = } } \ ( H _ { t } ^ { ( 1 ) } , \ldots , H _ { t } ^ { ( M ) } )$ denotes the joint latent configuration of M parallel hidden Markov chains with K possible states each. At each time point, the combination of these M latent variables, each taking one of K-states, determines the distribution of the observable $Y _ { t }$

In the presence of the auxillary input variables, such as in our case medications, one can model the output assuming the dependence of the hidden states on the input, which leads to Input-output FHMM formulation.

Exact inference for (3) can, in principle, be carried out by representing the FHMM as a single HMM with $K ^ { M }$ states, where the transition matrix can be written as

$$
P = P ^ { ( 1 ) } \otimes P ^ { ( 2 ) } \otimes \cdots \otimes P ^ { ( M ) } \in \mathbb { R } ^ { K ^ { M } } \times \mathbb { R } ^ { K ^ { M } } ,
$$

where ⊗ is the Kronecker product. Nevertheless, the introduction of additional chains makes the estimation intractable, as a naive implementation of the Forward-Backward algorithm has complexity $O ( K ^ { 2 M } T )$ . The complexity can be reduced to $O ( M K ^ { M + 1 } T )$ by carefully computing the matrix-vector dot product [15], but still grows exponentially in the number of chains. One of the ways to make the estimation tractable is to use variational approximation, which we describe in Section 2.4.

We impose the following problem-specific structure on the FHMM to improve identifiability and interpretability.

## 2.3.1 Problem-specific structure

The transition matrix specifies the probabilities of moving between latent health states from one time step to the next. The general FHMM defined in Section 2.3 is not easily interpretable as the states of the chaines are not ordered and any permutation of the states labels $\{ 1 , . . . , K \}$ gives the same model. In order to make the identification of the states possible, we assume that states are ordered by worsening of the patient’s condition, and that lower state values represent a healthy patient. Additionally, we assume that the health of a patient can degrade only by d units (a jump of d states) and that the last state is an absorbing state, representing the impossibility of improving the patient’s health condition. These assumptions mean that the transition matrix $P _ { \theta ^ { P } }$ is a band matrix with bandwidth $d ,$ where the last row of the matrix $P _ { \theta ^ { P } }$ contains all zeros except in the last column, which is equal to one. Figure 3 shows all possible transitions when $d = 1$ . The transition matrix is fully parameterized without the inclusion of input variables, such as medications. A discussion on incorporating medications into the parameterization is provided in Appendix C.

$$
\textcircled { 1 } \overrightarrow { 2 } \overrightarrow { 2 } \overrightarrow { 5 } \overrightarrow { 4 } \overrightarrow { 5 }
$$

Figure 3: Possible transitions between states of a hidden chain with 5 states and jumps of size $d = 1$

The Emission probability $p ( \cdot | H _ { t } ; \eta )$ should capture that the higher the patients’ state value is, the worse their health conditions are. In particular, the conditional mean

$$
\mathbb { E } [ Y _ { t } | H _ { t } ] \stackrel { \mathrm { d e f } } { = } g \left( \eta _ { 0 } + \sum _ { m } \tilde { { H _ { t } ^ { ( m ) } } } ^ { \top } \eta ^ { ( m ) } \right) ,
$$

should be a monotone function of each state component $H _ { t } ^ { ( m ) }$ for all $m ,$ where $\tilde { H } _ { t } ^ { ( m ) } \in \{ 0 , 1 \} ^ { K }$ is a one-hot encoded version of $H _ { t } ^ { ( m ) } \in \{ 1 , \dots , K \}$ and $g$ is some link function. Note that $\eta _ { 1 } ^ { ( m ) } = 0$ for identifiability issues and that the intercept is captured by $\eta _ { 0 }$

As in [16], to guarantee monotonicity we parametrize the variable $\eta ^ { ( m ) }$ as the cumulative sum of $\eta _ { + } ^ { ( m ) } \geq 0$ and $\eta _ { - } ^ { ( m ) } \geq 0 , \mathrm { i . e . }$

$$
\eta ^ { ( m ) } \stackrel { \mathrm { d e f } } { = } D \eta _ { + } ^ { ( m ) } - D \eta _ { - } ^ { ( m ) } ,
$$

where D is a lower triangular matrix with all ones. This means that each diference between successive coeficients is either non-negative or non-positive: if $\eta _ { - } ^ { ( m ) } = 0 ;$ , then

$$
\eta _ { i } ^ { ( m ) } - \eta _ { i - 1 } ^ { ( m ) } = \eta _ { + , i } ^ { ( m ) } \geq 0 \iff \eta _ { i } ^ { ( m ) } \geq \eta _ { i - 1 } ^ { ( m ) } ,
$$

hence $\eta ^ { ( m ) }$ is non-decreasing. Similarly, if $\eta _ { + } ^ { ( m ) } = 0$ , then

$$
\eta _ { i } ^ { ( m ) } - \eta _ { i - 1 } ^ { ( m ) } = - \eta _ { - , i } ^ { ( m ) } \leq 0 \iff \eta _ { i } ^ { ( m ) } \leq \eta _ { i - 1 } ^ { ( m ) } ,
$$

hence $\eta ^ { ( m ) }$ is non-increasing. To decide whether the slope should be positive or negative, we add a group-lasso penalty [17]:

$$
p ( \eta ) = \lambda \times \sum _ { m } \left( \| \eta _ { - } ^ { ( m ) } \| _ { 2 } + \| \eta _ { - } ^ { ( m ) } \| _ { 2 } \right) .\tag{4}
$$

This penalty enforces sparsity at the group level, i.e. it encourages either $\eta _ { + } ^ { ( m ) } = 0$ (purely decreasing slope) or $\eta _ { - } ^ { ( m ) } = 0$ (purely increasing slope).

The initial state is modeled as a function $\pi _ { \boldsymbol { \theta } ^ { \pi } }$ of the list of drugs taken over a period of 4 years prior to NIAD, age, smoking states and alcohol status jointly denoted by $X _ { 1 }$ . The distribution is assumed to be unimodal and is parametrized by a gamma distribution, i.e.,

$$
\left[ \pi _ { \theta ^ { \pi } } ( X _ { 1 } ) \right] _ { i } = { \mathrm { S o f t m a x } } \left\{ ( \alpha ( X _ { 1 } ) - 1 ) \log i - \beta ^ { \mathrm { i n i t } } i \right\} \propto i ^ { \alpha ( X _ { 1 } ) - 1 } \exp ( - \beta ^ { \mathrm { i n i t } } i ) ,\tag{5}
$$

with $\alpha ( X _ { 1 } ) = W ^ { \mathrm { i n i t } } X _ { 1 } + \alpha ^ { \mathrm { i n i t } }$ and $\theta ^ { \pi } = \{ W ^ { \mathrm { i n i t } } , \alpha ^ { \mathrm { i n i t } } , \beta ^ { \mathrm { i n i t } } \}$ . All parameters are assumed to be nonnegative. If α is close to zero, the probability mass is concentrated mostly around the lowest state value, whereas larger values of α increase the probability of starting from a higher state value.

Assuming that a drug prescription is equivalent to a degradation of the patient’s health, we impose $W ^ { \mathrm { i n i t } } \geq 0$ and $\alpha ^ { \mathrm { i n i t } } \geq 0$ . In addition, $W ^ { \mathrm { i n i t } } , \ \alpha ^ { \mathrm { i n i t } }$ and $\beta ^ { \mathrm { i n i t } }$ follow a priori a (truncated) Gaussian distribution centered at 0, 1 and 1 respectively, which is equivalent to add the penalizations $\lVert W ^ { \mathrm { i n i t } } \rVert _ { 2 } ^ { 2 }$ $\lVert \alpha ^ { \mathrm { i n i t } } - 1 \rVert _ { 2 } ^ { 2 }$ and $\lVert \beta ^ { \mathrm { i n i t } } - 1 \rVert _ { 2 } ^ { 2 }$ to the objective function. The underlying assumptions are

1. the drug efect is a priori small $\mathrm { ( i . e . , } \ : W ^ { \mathrm { i n i t } }$ is likely to be small),

2. in the absence of drugs, the likelihood of starting from a higher state is small, that is, the prior distribution is $[ \pi _ { \theta ^ { \pi } } ( X _ { 1 } ) ] _ { i } \propto \exp ( - i )$

## 2.4 Model estimation

Parameters of the models are estimated via Expectation-Maximization (EM) following the work of Ghahramani [2], with the aim to maximize the Evidence Lower Bound (ELBO)

$$
\operatorname* { m a x } _ { \theta , q } F ( q , \theta ) = \operatorname* { m a x } _ { \theta , q } \mathbb { E } _ { h _ { 1 : T } \sim q } \left[ \log \frac { p _ { \theta } ( y _ { 1 : T } , h _ { 1 : T } \mid x _ { 1 : T } ) } { q ( h _ { 1 : T } \mid x _ { 1 : T } , y _ { 1 : T } ) } \right] ,\tag{6}
$$

over a class of approximative distributions $q \in \mathcal { D }$ (E step) and the parameters $\theta = ( \theta ^ { \pi } , \theta ^ { P } , \eta )$ (M step), where we used the notation $s _ { a : b }$ to denote the sequence s from a to b. In [2], the authors proposed to use a structured mean field variational approximation [18]

$$
q ( h _ { 1 : T } \mid x _ { 1 : T } ) \stackrel { \mathrm { d e f } } { = } \prod _ { m } q ( h _ { 1 : T } ^ { ( m ) } \mid x _ { 1 : T } , y _ { 1 : T } ) ,
$$

with

$$
\begin{array} { r } { q \left( \boldsymbol { H } _ { 1 } ^ { ( m ) } = i \mid \boldsymbol { x } _ { 1 } , \boldsymbol { y } _ { 1 : T } \right) \propto e _ { 1 , i } ^ { ( m ) } \times \pi _ { i } ^ { ( m ) } , } \\ { q \left( \boldsymbol { H } _ { t } ^ { ( m ) } = j \mid \boldsymbol { H } _ { t - 1 } ^ { ( m ) } = i , \boldsymbol { x } _ { t } , \boldsymbol { y } _ { 1 : T } \right) \propto e _ { t , j } ^ { ( m ) } \times \boldsymbol { P } _ { i , j } ^ { ( m ) } , } \end{array}\tag{7}
$$

as an approximation of the optimal distribution $q ( h _ { 1 : T } \mid x _ { 1 : T } , y _ { 1 : T } ) = p ( h _ { 1 : T } \mid y _ { 1 : T } , x _ { 1 : T } )$ , which reduces the inference complexity to $O ( K ^ { 2 } M T )$ in the case of Gaussian emissions. The parameters $\phi = \{ e _ { t , j } ^ { ( m ) } \} _ { t , j , m }$ are learned by optimizing the ELBO (6).

In the case of binary outputs, such as comorbidities, the complexity still grows exponentially in the number of chains $\bar { O } ( M \bar { K } ^ { M + 1 } T )$ [15]. However, inspired by the Gaussian case, a second order approximation of the emission log-likelihood log $p _ { \eta } ( y _ { t } \mid h _ { t } ^ { ( 1 : M ) } )$ is derived using the inequality [19]

$$
- \log ( 1 + \exp y ) \geq - \log \left\{ 1 + \exp ( - z ) \right\} - ( y + z ) / 2 - \frac { \lambda ( z ) } { 2 } ( y ^ { 2 } - z ^ { 2 } ) ,\tag{8}
$$

with $\begin{array} { r } { \lambda ( z ) ~ = ~ \frac { \operatorname { t a n h } ( z / 2 ) } { 2 z } } \end{array}$ and some additional variable $z ~ \in ~ \mathbb { R }$ . Exploiting this inequality reduces the complexity of the model training to $O ( K ^ { 2 } M T )$ (see Appendix B for more details). After training, parameters θ are kept fixed and the parameters ϕ are learned when presented with new data. We refer this step as ”fine-tuning”.

## 2.5 Trajectory analysis

The emission parameters $\eta ^ { ( m ) }$ model the risk of developing a comorbidity for a given state, but do not capture the progression of comorbidities over time. On the other hand, the sequence of hidden states $\{ H _ { t } ^ { ( m ) } \} _ { t > 0 }$ encodes the temporal dynamics of disease evolution, allowing us to infer how patients transition between latent health states and how these transitions influence the development and accumulation of comorbidities across time.

In particular, the variational posterior distribution $q _ { \phi } ( H _ { t } ^ { ( m ) } \mid x _ { 1 : T } , y _ { 1 : T } )$ is given by $( 7 )$ , after fitting the parameters ϕ for each patient. The latter is computed using the Baum–Welch algorithm for HMMs (see [2]). From the posterior distribution, we calculate the expected state of each latent chain at each time step as

$$
\mathbb { E } _ { q } [ h _ { t } ^ { ( m ) } ] = \sum _ { h _ { t } } h _ { t } q _ { \phi } ^ { ( m ) } ( h _ { t } \mid x _ { 1 : T } , y _ { 1 : T } ) ,
$$

yielding an M-dimensional representation per time point. We then apply K-means clustering to all time steps, selecting the number of clusters based on the Davies–Bouldin index and prediction strength [20, 21].

## 2.6 Experimental setup

Patient data were partitioned into three distinct subsets: training (80%), validation (10%), and test (10%). Stratification was applied to ensure a consistent distribution of comorbidities across these subsets (see Table 6 in the Appendix).

All models were trained for a maximum of 300 EM iterations, with early stopping applied if the objective function F showed no improvement. Model parameters were initialized to zero, except for $\alpha ^ { \mathrm { i n i t } } , \beta ^ { \mathrm { i n i t } }$ , which were initialized to one. Multiple FHMMs with varying numbers of chains and states were trained, and model selection was performed using the Akaike Information Criterion (AIC) [22] and the Bayesian Information Criterion (BIC) [23].

## 3 Results

## 3.1 Patient cohort and characteristics

After applying the preprocessing workflow described in Section 2.2.1, the final cohort comprised 88201 patients, including 47983 men and 40218 women. Figure 4 illustrates the filtering process of patients based on missing laboratory data during the four years preceding NIAD initiation. In total, 100325 patients were excluded due to missing data for key laboratory values.

![](images/162a8434f38ef6d34fc6cd53ac58732fe881bf89cb666965a828f38c7b7256a8.jpg)  
Figure 4: Filtering of patients based on missing laboratory data during the four years preceding NIAD initiation. A total of 100325 patients were excluded. Abbreviations are defined in Table 1.

The median follow-up duration after NIAD initiation was approximately 4.0 years across all groups. Baseline patient characteristics at the index date (first NIAD prescription) are summarized in Table 2. Continuous variables, including age, body mass index (BMI), and glycated hemoglobin (HbA1c), are reported as medians with interquartile ranges (IQRs), while categorical variables—such as sex, smoking status, and comorbidities—are presented as counts and percentages.

Smoking prevalence was higher among men than women. Hypertension and diabetic retinopathy were the most prevalent comorbidities at baseline, with diabetic retinopathy showing the greatest increase in prevalence following NIAD initiation. Comorbidities that developed after NIAD initiation are indicated in parentheses in Table 2.

## 3.2 Model selection

Table 3 presents the AIC and BIC scores for various models evaluated on the validation set. We use the notation $\mathrm { F H M M } ( M , K ) _ { d }$ to denote a factorial hidden Markov model (FHMM) with M chains, K states per chain, and a transition matrix bandwidth of d. Similarly, $\mathrm { H M M } _ { d }$ refers to a standard hidden Markov model (HMM) with K states and a transition matrix $P _ { \theta ^ { P } }$ of bandwidth d.

For comparison, an HMM with $K = 2 7 = 3 ^ { 3 }$ states is evaluated against an FHMM of equivalent size. In the FHMM, each chain has 2 possible transitions $( d = 1 )$ , which corresponds to an HMM bandwidth of $d = 8 = 2 ^ { 3 }$ . Because the relationship between HMM state values and worsening health conditions is less direct (where a single disease progression may be too restrictive), the HMM emission probabilities are parameterized without monotonicity constraints, and drug information is excluded from the transition matrix input.

Memory constraints limit our comparison to a state dimension of $\tilde { K } = K ^ { 3 } = 2 7$ , with $K = 3$ . In the HMM, the full matrix

$$
\xi _ { t } = \mathbb { E } \left[ \tilde { H } _ { t } \tilde { H } _ { t - 1 } ^ { \top } \right] \in \mathbb { R } ^ { K ^ { 3 } \times K ^ { 3 } }
$$

<table><tr><td>Sex</td><td colspan="2">Men</td><td colspan="2">Women</td></tr><tr><td>Age group</td><td>&lt; 65</td><td>65+</td><td>&lt; 65</td><td>65+</td></tr><tr><td>Number of patients Average age at NIAD (y) ± SD Median follow-up duration in years post-NIAD [IQR]</td><td>27191 54.0 ± 8.0  $4 . 5 \ [ 5 . 0 ]$ </td><td>21989  $7 3 . 7 \pm 6 . 3$   $4 . 0 ~ [ 4 . 5 ]$ </td><td>20792  $5 2 . 5 \pm 9 . 3$   $4 . 0 \ [ 5 . 0 ]$ </td><td>18229  $7 4 . 9 \pm 6 . 7$   $4 . 0 \ [ 4 . 5 ]$ </td></tr><tr><td>Average BMI at NIAD  $( \mathrm { k g } / \mathrm { m ^ { 2 } } ) \pm \mathrm { S D }$  Average HbA1c at NIAD  $( \% ) \pm \mathrm { S D }$  % Ever smoked Comorbidities at NIAD initiation (%)</td><td> $3 2 . 9 \pm 6 . 3$   $7 . 4 \pm 1 . 7$  45.2</td><td> $3 0 . 1 \pm 5 . 1$   $7 . 1 \pm 1 . 4$  34.4</td><td> $3 5 . 4 \pm 7 . 9$   $7 . 2 \pm 1 . 6$  39.4</td><td> $3 1 . 0 \pm 6 . 4$   $7 . 0 \pm 1 . 3$  28.8</td></tr><tr><td>Stroke</td><td>5.3</td><td>15.6</td><td>4.5</td><td>13.9</td></tr><tr><td>Myocardial infarction</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>8.0</td><td>15.8</td><td>2.6</td><td>7.1</td></tr><tr><td>Congestive heart failure</td><td>3.8</td><td>11.9</td><td>1.9</td><td>9.2</td></tr><tr><td>Peripheral vascular disease</td><td>3.3</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>10.1</td><td>1.4</td><td>5.0</td></tr><tr><td>Diabetic retinopathy</td><td>32.3</td><td>35.3</td><td>29.8</td><td>34.7</td></tr><tr><td>Hypertension</td><td>57.6</td><td>70.7</td><td>53.9</td><td>77.4</td></tr><tr><td>Chronic kidney disease</td><td>5.4</td><td>10.6</td><td>6.6</td><td>9.6</td></tr><tr><td>Osteoporosis</td><td>0.6</td><td>1.9</td><td>2.2</td><td>10.6</td></tr><tr><td>COPD</td><td>6.4</td><td>15.6</td><td>6.7</td><td>13.5</td></tr><tr><td>Chronic liver disease</td><td>5.1</td><td>3.0</td><td>6.2</td><td>3.7</td></tr><tr><td>Laboratory measurements 6 months before NIAD</td><td></td><td></td><td></td><td></td></tr><tr><td>Diastolic blood pressure (mmHg)</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>84.5 (31.7)</td><td>77.4 (22.3)</td><td>81.5 (38.0)</td><td>77.5 (22.0)</td></tr><tr><td>Systolic blood pressure (mmHg)</td><td>139.1 (31.7)</td><td>138.5 (22.3)</td><td>134.3 (38.0)</td><td>140.5 (22.0)</td></tr><tr><td>GFR (mL/min/1.73 m²)</td><td>76.8 (35.2)</td><td>67.6 (29.2)</td><td>75.3 (46.3)</td><td>64.1 (29.3)</td></tr><tr><td>HDL (mmol/L)</td><td>1.1 (31.0)</td><td>1.2 (31.5)</td><td>1.2 (48.0)</td><td>1.4 (33.7)</td></tr><tr><td>LDL (mmol/L)</td><td>3.1 (45.5)</td><td>2.5 (43.6)</td><td>3.3 (56.5)</td><td>2.8 (44.8)</td></tr><tr><td>Triglycerides (mmol/L)</td><td>2.8 (35.3)</td><td>2.1 (37.1)</td><td>2.3 (51.1)</td><td>2.1 (38.8)</td></tr></table>

Table 2: Patient characteristics at the index date (NIAD initiation) for the cohort after preprocessing. Comorbidities are reported as percentages of the cohort population. Laboratory measurements are summarized as mean values, with the proportion of missing observations reported in parentheses. Ab breviations are defined in Table 1. SD denotes standard deviation and IQR denotes interquartile range.

must be computed for all patients and timesteps, where $\tilde { H } _ { t }$ is the one-hot encoding of $H _ { t }$ . In contrast, the FHMM primarily requires computing

$$
\boldsymbol { \xi } _ { t } ^ { ( m ) } = \mathbb { E } [ \boldsymbol { \tilde { H } } _ { t } ^ { ( m ) } \boldsymbol { \tilde { H } } _ { t - 1 } ^ { ( m ) }  ^ { \top } ] \in \mathbb { R } ^ { K \times K }
$$

for all patients, timesteps t, and chains m. Since memory complexity scales quadratically with K, we kept K small. In particular, increasing the number of states from 4 to 6 did not necessarily improve the BIC on the validation set for larger models (see Table 3).

The top-performing model was $\mathrm { F H M M } ( 8 , 4 ) _ { 2 }$ across all sex and age groups, except among $M e n < 6 5$ for whom $\mathrm { F H M M } ( 6 , 4 ) _ { 2 }$ performed better according to both BIC and AIC. The model FHMM(8, 4)<sub>2</sub> corresponds to the largest latent space, capable of encoding up to $4 ^ { 8 } = 6 5 , 5 3 6$ possible states. In what follows, we will focus on the best performing model for each age and sex group. Generally, FHMM models consistently outperform HMM models in terms of BIC.

## 3.3 Model analysis

Next we provide the analysis of the results for Women < 65 subgroup. Other subgroups can be found in the Appendix. The model selected on validation was ”fine-tuned” on the testing set, meaning that the variational parameters ϕ were newly inferred based on the latter.

Emission parameters. The emission parameters for each latent state are shown in Figure 5. By construction (Section 2.3.1), higher states correspond to greater probabilities of specific comorbidities, revealing clusters of conditions that tend to co-occur:

<table><tr><td rowspan="2">Model</td><td colspan="2">Men &lt; 65</td><td colspan="2">Women &lt; 65</td><td colspan="2">Men 65+</td><td colspan="2">Women 65+</td></tr><tr><td>BIC</td><td>AIC</td><td>BIC</td><td>AIC</td><td>BIC</td><td>AIC</td><td>BIC</td><td>AIC</td></tr><tr><td>HMM(27)8</td><td>329077</td><td>314469</td><td>236219</td><td>221959</td><td>234704</td><td>220497</td><td>212736</td><td>198738</td></tr><tr><td>FHMM(3, 3)1</td><td>310663</td><td>307445</td><td>225526</td><td>222385</td><td>229629</td><td>226500</td><td>203633</td><td>200550</td></tr><tr><td>FHMM(4, 4)2</td><td>292039</td><td>286438</td><td>216236</td><td>210769</td><td>221682</td><td>216235</td><td>195206</td><td>189839</td></tr><tr><td>FHMM(4, 6)2</td><td>293949</td><td>285052</td><td>211923</td><td>203239</td><td>220663</td><td>212010</td><td>195312</td><td>186787</td></tr><tr><td>FHMM(6, 4)2</td><td>264231</td><td>256136</td><td>190847</td><td>182946</td><td>201848</td><td>193976</td><td>181320</td><td>173563</td></tr><tr><td>FHMM(6, 6)2</td><td>273908</td><td>260869</td><td>196096</td><td>183369</td><td>196499</td><td>183819</td><td>181122</td><td>168628</td></tr><tr><td>FHMM(8, 4)2</td><td>273337</td><td>262748</td><td>178959</td><td>168623</td><td>181179</td><td>170881</td><td>162739</td><td>152592</td></tr><tr><td>FHMM(10, 2)1</td><td>276474</td><td>270739</td><td>211039</td><td>192272</td><td>186698</td><td>188498</td><td>174079</td><td>168585</td></tr></table>

Table 3: Comparison between models with various number of chains. Metrics are evaluated on the validation set and drugs are not used as input.

![](images/5ff9b743e03dab225230bb5da55ba5fb273b8dcde37676fdb66bc8ef8c755238.jpg)  
Figure 5: Top: Emission parameters η modeling the risk of developing a comorbidity for a given state. Larger values indicate a higher probability of having the comorbidity. The values are reported on a logarithmic scale. Bottom: Mean value of lab tests with respect to the intercept (i.e., state 0). Red (resp. blue) indicates an increase (resp. decrease) in E[Y<sub>t</sub>|H<sub>t</sub>] relative to the intercept. The model includes 8 chains with 4 states per chain and is fitted on on Women < 65.

• Chain 0, termed ”Osteoporosis progression,” combines elevated risks of osteoporosis, chronic kidney disease and fatty liver disease with a high-HDL/LDL and low-triglyceride profile.

• Chain 1, ”Cardiovascular progression,” captures macrovascular outcomes (peripheral vascular disease, heart failure, death) and renal failure (low GFR), along with osteoporosis.

• Chain 2, ”Infarction progression,” is marked by myocardial infarction risk accompanied by BMI reduction.

• Chain 3, ”Metabolic syndrome progression,” features obesity, dyslipidemia, chronic obstructive pulmonary disease, liver disease and death risk.

• Chain 4, ”Hypertension progression,” reflects isolated elevation of systolic and diastolic blood pressure.

![](images/f856cb329232193015728614fa65c8dca0f397116917ed1f33014c0c9f3231df.jpg)  
Figure 6: Estimated contribution of drug prescriptions to initial latent state allocation $( W ^ { \mathrm { i n i t } } )$ , based on a model with 8 chains and 4 states per chain fitted to the $W o m e n < 6 5$ subgroup. Higher values indicate a greater probability of starting in a more severe latent state. These estimates capture baseline correlations, not causal efects, since they describe state assignment rather than counterfactual deviation.

• Chain 5, ”Retinopathy progression,” shows increased diabetic retinopathy probability alongside poor glycemic and lipid control (elevated HbA1c, triglycerides and BMI, reduced HDL).

• Chain 6, ”Vascular progression,” combines stroke, peripheral vascular disease and COPD with a comparatively favorable lipid profile.

• Chain 7, ”Cardiorenal progression,” integrates myocardial infarction, stroke, chronic heart failure, chronic kidney disease and dyslipidemia with obesity. Hypertension and diabetic retinopathy tend to develop independently of other comorbidities in our model.

Initial latent state. The relationship between the initial latent state and the input variables is shown in Figure 6. Positive values (in red) correspond to a higher probability of beginning in a more severe state. Smoking shows a marked positive value in chain 3, reflecting its association with increased COPD risk. The age parameter is positive across several chains, indicating that older patients tend to start in worse health states. This pattern is not observed in chain 5, where age has no significant efect on the initial state. Receiving treatment for bone diseases is associated with an increased likelihood of being in the osteoporosis state at the start.

State Clustering. Following Section 2.5, we identified 13 clusters in the latent representation according to the selection criterion (see Figure 12 in the Appendix). Table 4 details the baseline characteristics of each cluster at NIAD initiation, and Figure 7 presents their time-averaged clinical profiles.

Several clusters demonstrate similarities at NIAD initiation. Clusters 0, 1, and 11 share characteristics of younger patients (age < 50) with low comorbidity burden. However, their clinical trajectories difer significantly - particularly cluster 1, which despite its younger profile shows higher smoking prevalence, elevated BMI, and increased mortality risk (see Table 4).

The clusters can be divided based on their mortality risk. At one end of the spectrum, clusters 0, 2, 4, 5, and 11 form a lower-risk group characterized by relatively low death rates. An intermediate category includes clusters 1, 9, and 10, which show progressing disease with moderate mortality risk. The highest-risk group comprises clusters 3, 6, 7, 8, and 12, showing substantially elevated mortality rates that indicate advanced disease states. Within this high-risk category, cluster 7 is particularly notable: it is defined by the highest BMI (morbid obesity with BMI > 40) and carries an exceptionally high comorbidity burden, indicating particularly severe disease progression.

The healthiest subgroup, cluster 0 (”young low-risk”), is characterized by preserved metabolic and renal function and low cardiovascular risk. Cluster 3 (”pulmonary-metabolic”) combines metabolic dysfunction and pulmonary disease with high smoking prevalence and renal impairment. This cluster’s profile, with increased risk in nearly all comorbidities, positions it as a transition toward severe multimorbidity. On the other hand, cluster 10 (”bone-vascular”) features high osteoporosis prevalence and increased peripheral vascular disease, suggesting shared pathophysiological pathways between skeletal and vascular systems.

<table><tr><td>Cluster</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td></tr><tr><td># of patients</td><td>3373</td><td>1468</td><td>1549</td><td>792</td><td>2774</td><td>1491</td><td>1101</td><td>1431</td><td>1006</td><td>1421</td><td>594</td><td>1143</td><td>1362</td></tr><tr><td>Avg. age at NIAD (y) ± SD</td><td>47.01 ± 10.68</td><td>47.1 ± 10.3</td><td>54.01 ± 7.19</td><td>57.49 ± 6.09</td><td>54.27 ± 7.5</td><td>54.53 ± 8.24</td><td>54.61 ± 7.71</td><td>55.52 ± 7.38</td><td>56.12 ± 7.1</td><td>53.19 ± 7.93</td><td>57.12 ± 6.27</td><td>48.75 ± 9.98</td><td>55.23 ± 7.86</td></tr><tr><td>Avg. follow-up post-NIAD (y) ± SD</td><td>3.73 ± 2.68</td><td>4.52 ± 2.98</td><td>5.88 ± 3.15</td><td>4.87 ± 2.99</td><td>4.63 ± 3.13</td><td>5.23 ± 3.28</td><td>5.23 ± 3.19</td><td>6.21 ± 3.27</td><td>5.37 ± 3.24</td><td>5.81 ± 3.25</td><td>6.79 ± 3.23</td><td>5.08 ± 3.0</td><td>6.52 ± 3.35</td></tr><tr><td>Avg. BMI at NIAD (kg/m2) ± SD</td><td>33.15 ± 7.55</td><td>37.25 ± 7.79</td><td>34.07 ± 6.16</td><td>35.74 ± 7.81</td><td>34.54 ± 6.94</td><td>34.31 ± 6.73</td><td>35.04 ± 6.93</td><td>40.96 ± 9.87</td><td>36.64 ± 7.38</td><td>38.35 ± 7.72</td><td>31.23 ± 6.45</td><td>32.96 ± 6.62</td><td>38.04 ± 8.34</td></tr><tr><td>Avg. HbA1c at NIAD (%) ± SD</td><td>6.83 ± 1.56</td><td>7.26 ± 1.66</td><td>7.48 ± 1.61</td><td>7.24 ± 1.47</td><td>6.89 ± 1.37</td><td>7.15 ± 1.49</td><td>7.25 ± 1.48</td><td>7.51 ± 1.61</td><td>7.35 ± 1.5</td><td>7.05 ± 1.46</td><td>7.16 ± 1.59</td><td>7.94 ± 1.83</td><td>6.92 ± 1.37</td></tr><tr><td>% Ever smoked</td><td>23.04</td><td>79.56</td><td>20.46</td><td>78.41</td><td>20.01</td><td>29.51</td><td>43.23</td><td>56.81</td><td>55.07</td><td>59.25</td><td>30.13</td><td>24.41</td><td>50.07</td></tr><tr><td>% Ever alcohol</td><td>70.53</td><td>82.63</td><td>85.15</td><td>85.61</td><td>78.98</td><td>79.61</td><td>82.29</td><td>84.35</td><td>82.9</td><td>86.77</td><td>82.15</td><td>82.06</td><td>82.53</td></tr><tr><td>Comorbidities at NIAD [N,(%)]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stroke</td><td>&lt;1 %</td><td>&lt; 1%</td><td>&lt; 1%</td><td>&lt;1%</td><td>&lt; 1 %</td><td>294 (19.72)</td><td>&lt; 1%</td><td>178 (12.44)</td><td>&lt; 1 %</td><td>&lt; 1%</td><td>&lt;1%</td><td>&lt; 1 %</td><td>104 (7.64)</td></tr><tr><td>Myocardial infarction</td><td>&lt; 1%</td><td>&lt; 1 %</td><td>&lt;1%</td><td>16 (2.02)</td><td>&lt; 1%</td><td>&lt; 1 %</td><td>12 (1.09)</td><td>158 (11.04)</td><td>&lt; 1%</td><td>&lt; 1%</td><td>&lt;1 %</td><td>&lt; 1 %</td><td>171 (12.56)</td></tr><tr><td>Congestive heart failure</td><td>&lt;1%</td><td>&lt;1 %</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt;1 %</td><td>49 (3.29)</td><td>&lt; 1 %</td><td>89 (6.22)</td><td>&lt; 1%</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt;1%</td><td>50 (3.67)</td></tr><tr><td>Peripheral vascular disease</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt; 1%</td><td>43 (5.43)</td><td>&lt; 1 %</td><td>19 (1.27)</td><td>&lt; 1 %</td><td>25 (1.75)</td><td>&lt; 1 %</td><td>&lt;1%</td><td>65 (10.94)</td><td>&lt;1%</td><td>&lt;1 %</td></tr><tr><td>Diabetic retinopathy</td><td>&lt;1%</td><td>116 (7.90)</td><td>520 (33.57)</td><td>118 (14.90)</td><td>&lt;1 %</td><td>213 (14.29)</td><td>188 (17.08)</td><td>436 (30.47)</td><td>191 (18.99)</td><td>69 (4.86)</td><td>67 (11.28)</td><td>445 (38.93)</td><td>&lt;1%</td></tr><tr><td>Hypertension</td><td>79 (2.34)</td><td>23 (1.57)</td><td>1155 (74.56)</td><td>384 (48.48)</td><td>2230 (80.39)</td><td>821 (55.06)</td><td>593 (53.86)</td><td>938 (65.55)</td><td>769 (76.44)</td><td>1113 (78.33)</td><td>288 (48.48)</td><td>&lt;1 %</td><td>843 (61.89)</td></tr><tr><td>Chronic kidney disease</td><td>&lt;1 %</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>&lt;1 %</td><td>&lt; 1 %</td><td>399 (26.76)</td><td>18 (1.63)</td><td>191 (13.35)</td><td>&lt;1%</td><td>&lt;1 %</td><td>&lt;1 %</td><td>&lt; 1%</td><td>170 (12.48)</td></tr><tr><td>Osteoporosis</td><td>&lt; 1%</td><td>&lt; 1 %</td><td>&lt;1%</td><td>&lt; 1%</td><td>&lt; 1%</td><td>35 (2.35)</td><td>65 (5.90)</td><td>20 (1.40)</td><td>&lt;1%</td><td>&lt; 1%</td><td>150 (25.25)</td><td>&lt;1%</td><td>&lt; 1%</td></tr><tr><td>COPD</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt;1%</td><td>497 (62.75)</td><td>&lt;1%</td><td>&lt; 1%</td><td>78 (7.08)</td><td>149 (10.41)</td><td>&lt;1%</td><td>&lt;1%</td><td>6 (1.01)</td><td>&lt;1%</td><td>82 (6.02)</td></tr><tr><td>Chronic liver disease</td><td>&lt;1%</td><td>&lt; 1%</td><td>&lt; 1%</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt; 1%</td><td>531 (48.23)</td><td>65 (4.54)</td><td>&lt; 1%</td><td>&lt;1%</td><td>&lt;1 %</td><td>&lt;1 %</td><td>39 (2.86)</td></tr><tr><td>Comorbidities during follow-up [N,(%)]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stroke</td><td>&lt; 1 %</td><td>&lt; 1%</td><td>&lt; 1%</td><td>10 (1.26)</td><td>&lt; 1 %</td><td>102 (6.84)</td><td>14 (1.27)</td><td>59 (4.12)</td><td>&lt; 1%</td><td>&lt; 1 %</td><td>14 (2.36)</td><td>&lt; 1%</td><td>58 (4.26)</td></tr><tr><td>Myocardial infarction</td><td>&lt; 1%</td><td>&lt; 1 %</td><td>&lt; 1%</td><td>10 (1.26)</td><td>&lt; 1%</td><td>&lt; 1 %</td><td>11 (1.00)</td><td>27 (1.89)</td><td>&lt;1%</td><td>20 (1.41)</td><td>&lt; 1 %</td><td>&lt;1%</td><td>29 (2.13)</td></tr><tr><td>Congestive heart failure</td><td>&lt;1%</td><td>&lt; 1 %</td><td>&lt; 1%</td><td>&lt; 1%</td><td>&lt; 1%</td><td>31 (2.08)</td><td>&lt;1%</td><td>59 (4.12)</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>6 (1.01)</td><td>&lt; 1 %</td><td>55 (4.04)</td></tr><tr><td>Peripheral vascular disease</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt;1%</td><td>20 (2.53)</td><td>&lt;1 %</td><td>&lt; 1 %</td><td>&lt;1%</td><td>&lt; 1 %</td><td>&lt;1%</td><td>&lt;1%</td><td>22 (3.70)</td><td>&lt;1%</td><td>&lt; 1%</td></tr><tr><td>Diabetic retinopathy</td><td>163 (4.83)</td><td>224 (15.26)</td><td>610 (39.38)</td><td>166 (20.96)</td><td>202 (7.28)</td><td>283 (18.98)</td><td>222 (20.16)</td><td>462 (32.29)</td><td>219 (21.77)</td><td>247 (17.38)</td><td>153 (25.76)</td><td>486 (42.52)</td><td>176 (12.92)</td></tr><tr><td>Hypertension</td><td>126 (3.74)</td><td>36 (2.45)</td><td>175 (11.30)</td><td>66 (8.33)</td><td>309 (11.14)</td><td>138 (9.26)</td><td>70 (6.36)</td><td>76 (5.31)</td><td>76 (7.55)</td><td>148 (10.42)</td><td>72 (12.12)</td><td>40 (3.50)</td><td>89 (6.53)</td></tr><tr><td>Chronic kidney disease</td><td>&lt;1%</td><td>&lt; 1 %</td><td>&lt;1%</td><td>13 (1.64)</td><td>&lt;1 %</td><td>166 (11.13)</td><td>39 (3.54)</td><td>124 (8.67)</td><td>&lt; 1%</td><td>&lt;1%</td><td>21 (3.54)</td><td>&lt; 1%</td><td>88 (6.46)</td></tr><tr><td>Osteoporosis</td><td>&lt; 1%</td><td>&lt;1%</td><td>&lt; 1%</td><td>15 (1.89)</td><td>&lt; 1 %</td><td>19 (1.27)</td><td>21 (1.91)</td><td>&lt; 1 %</td><td>&lt;1%</td><td>&lt;1%</td><td>50 (8.42)</td><td>&lt; 1%</td><td>&lt; 1 %</td></tr><tr><td>COPD</td><td>&lt;1%</td><td>31 (2.11)</td><td>&lt;1%</td><td>136 (17.17)</td><td>&lt;1 %</td><td>15 (1.01)</td><td>49 (4.45)</td><td>94 (6.57)</td><td>29 (2.88)</td><td>46 (3.24)</td><td>11 (1.85)</td><td>&lt;1%</td><td>58 (4.26)</td></tr><tr><td>Chronic liver disease</td><td>&lt;1%</td><td>28 (1.91)</td><td>&lt;1%</td><td>8 (1.01)</td><td>&lt;1%</td><td>16 (1.07)</td><td>296 (26.88)</td><td>75 (5.24)</td><td>26 (2.58)</td><td>36 (2.53)</td><td>19 (3.20)</td><td>14 (1.22)</td><td></td></tr><tr><td>Deaths before 2020</td><td>&lt; 1 %</td><td>42 (2.86)</td><td>&lt; 1%</td><td>79 (9.97)</td><td>&lt; 1 %</td><td>21 (1.41)</td><td>60 (5.45)</td><td>132 (9.22)</td><td>148 (14.71)</td><td>30 (2.11)</td><td>12 (2.02)</td><td>&lt;1%</td><td>49 (3.60) 93 (6.83)</td></tr></table>

Table 4: Patient characteristics in each cluster for the Women < 65 cohort with average values and standard deviation where applicable. Comorbidities are presented as absolute counts at NIAD initiation, along with the number of new comorbidities developed post-NIAD. Proportions, expressed as percentages of the cohort population, are reported in parentheses.

Among the most severe phenotypes, cluster 8 (”advanced nephropathy”) shows severe GFR loss, persistent hypertension, and high mortality. Cluster 12 (”cardio-pulmonary multimorbidity”) is defined by COPD, CHF, and elevated stroke risk. Cluster 6 (”hepatorenal syndrome”) represents a very severe situation, linking chronic liver disease, renal dysfunction (low GFR), and poor glycemic control, consistent with fatty liver disease trends [24].

State progression. The temporal distribution of the clusters is shown in Figure 8. Clusters associated with advanced or progressing disease (e.g., cluster 7 and transitional clusters 2, 6, 10, 12) gain members over time, while healthier clusters (0, 1, 4) decline.

Figure 9 presents the cumulative incidence functions for time to first cluster transition, stratified by initial cluster assignment. The transition dynamics are further illustrated in the appendix (Figure 13). Patients in cluster 0 often progress into early microvascular states, particularly cluster 11 or cluster 4, defined by higher prevalence of diabetic retinopathy and hypertension, respectively. After approximately four years, these states may progress to early multimorbidity (cluster 5) or a combination of both diseases (cluster 2), as confirmed by Figure 9 showing most cluster 4 patients transitioning to cluster 2. In contrast, cluster 1, despite representing younger patients, exhibits higher mortality risk and demonstrates progression to advanced disease states. These patients typically evolve into clusters 6 or 7 in later stages, indicating more severe disease trajectories.

Cluster 10 (bone-vascular) represents an intermediate severity profile. Patients in this cluster tend to progress toward broader multimorbidity through two primary pathways: either via clusters 2 or 5, or occasionally involving hepatic complications (cluster 6). The advanced disease profiles represented by cluster 12 (cardio-pulmonary multimorbidity) typically transition to cluster 7, the most severe multimorbid state, that serves as a frequent absorbing state. Similarly, clusters 2, 5, and 11 demonstrate low transition probabilities, functioning as stable states where patients tend to remain over time (at least when starting in those clusters).

The transition graph in Figure 10 (showing only transitions with less than 1% probability) reveals that clusters 0, 2, 4, 5, 10, and 11 form a transient subgraph. This configuration indicates that once patients transition out of these clusters, they rarely return, suggesting a unidirectional progression toward more severe disease states. The observed pattern reflects the clinical reality that as patients develop serious comorbidities, their condition tends to evolve toward increasingly complex and severe multimorbidity profiles rather than reverting to less severe states.

![](images/e88249a164b86c2ab9dda794a11c893eb16409ad26bd93e4cf310680c739d7e3.jpg)  
Figure 7: Average cluster-specific probabilities and lab values for Women < 65. Values have been centered by subtracting the average at the time of NIAD initiation to improve readability. Red bars indicate increases, and blue bars indicate decreases relative to the average patient at NIAD. Comorbidity risks are shown as changes in probability (left axis), and lab values as percentage changes relative to the reference value (right axis). Annotated numbers indicate absolute changes in the original units (see Table 1). Error bars represent the standard error of the mean.

## 4 Discussion

We identified eight latent components that capture distinct patterns of comorbidity in type 2 diabetes in women under 65 years of age (with additional population analyses provided in the Appendix). At the interpretation level, the inferred latent chains correspond to recognizable diabetes-related complications, spanning microvascular, macrovascular, metabolic, skeletal, pulmonary, and cardiorenal involvement. Several components reflect well-established comorbidity patterns in T2DM; for example, chains dominated by macrovascular and renal outcomes align with the known coupling between cardiovascular disease and chronic kidney disease in diabetes, often described as a cardiorenal syndrome [26]. Similarly, components emphasizing microvascular complications, such as retinopathy and hypertension, appear to develop independently of other complications and follow expected clinical patterns, including associations with elevated blood pressure and poor glycemic and lipid control, respectively [27, 28]. The skeletal involvement in our osteoporosis-related component co-occurred with chronic kidney and liver disease, reflecting the clustering of multisystem complications in T2DM. This pattern is consistent with the understanding that diabetes afects multiple organ systems through interconnected metabolic and vascular pathways [29, 30].

![](images/1f243ef29e700c57713a63bc1eafff2cd9a1b9e5c6eeb814870d6dc073f2fcc6.jpg)  
Figure 8: Distribution of the clusters over time.

While many of these associations are individually well documented, our model learns them jointly from longitudinal data without prespecifying disease groupings or assuming independence between outcomes. This yields an interpretable decomposition of multimorbidity into parallel disease processes that may co-occur and progress at diferent rates within individuals.

At the patient level, clustering of latent-state trajectories highlights how these components combine into progression pathways. Patients often transition from relatively healthy or low-risk metabolic states (clusters 0, 2, 4, 5, 10, 11) toward advanced multimorbid states, with clusters 6 and 7 representing terminal disease profiles associated with elevated mortality. These trajectories broadly separate into at least two modes of progression: one dominated by microvascular complications (clusters 2, 4, 5, 10, 11), and another driven by pulmonary, hepatic, renal, and cardio-pulmonary involvement (clusters 3, 6, 8, 9, 12).

From a methodological perspective, our model reveals patterns that are dificult to extract with alternative approaches. While static clustering approaches summarize patient similarity but do not encode temporal dynamics [12], and deep generative models can learn flexible latent representations from longitudinal clinical data but often require additional assumptions or post hoc alignment to relate latent dimensions to clinical concepts [9], our approach provides a structured factorization that enables direct clinical interpretation. Classical HMM-based models capture temporal dynamics but often rely on a single latent process, potentially mixing distinct disease mechanisms [3, 4]. The factorial structure adopted here enables multimorbidity to be represented as the superposition of multiple latent chains, providing a natural framework for chronic diseases characterized by overlapping and asynchronous complications.

This study has several limitations. Real-world electronic health records are collected based on routine clinical care, resulting in unequally spaced measurements that often required forward imputation to maintain complete information at each time-point. Some variables, particularly BMI and other lifestyle variables, exhibit missingness that is likely not missing at random. While forward imputation avoids using future information, alternative imputation schemes might be more appropriate for handling missing values. The current model uses Gaussian emissions for scalability, though extensions to other distributions (e.g., count-based likelihoods) are possible. The data were grouped into binned time steps, imposing an assumption of regular observation intervals that does not reflect the irregular nature of real-world measurements. The analysis was restricted to a relatively small set of commonly occurring chronic comorbidities and laboratory values, excluding less frequent conditions that might contribute to multimorbidity patterns. Issues of data imbalance persist, as many comorbidities represent rare events, and treatment adherence was not explicitly modeled, adding uncertainty to treatment-related efects. Finally, the requirement for HbA1c measurements prior to the index date may introduce selection bias, as tested individuals are more likely to have suspected or established comorbidities. As a result, the study cohort may represent a systematically sicker subset of the broader population, potentially limiting generalizability.

![](images/ab932bba9f533952ae0ea591bba41319910672bdafd5d06cbdb2d1e0360f94ef.jpg)  
Figure 9: Cumulative Incidence Function (CIF) for the time to first cluster transition in Women < 65, stratified by initial cluster assignment at 6 months post-NIAD. The estimates account for censoring due to loss to follow-up and study termination, computed using the Aalen nonparametric estimator [25].

## 5 Conclusion

Our study demonstrates that comorbidities in T2DM can be efectively modeled as the interaction of multiple latent disease processes evolving over time. The structural FHMM approach identified eight coherent latent components that explain the diferent disease trajectories, providing insights into the complex progression patterns of T2DM and its complications.

![](images/2d5081ac16ab798d34df84d664fa601bdaae607c7187c4defd6b6ca1bc3b9a0e.jpg)  
Figure 10: Transition graph. Red arrows (resp. blue) means an increase (resp. decrease) of the comorbidity risk (Women < 65). Arrow thickness is proportional to the number of observed transitions. Transitions with less than 1% probability are not shown.

The advantages of our approach include its ability to learn meaningful patterns of comorbidities jointly from longitudinal data, without requiring pre-specified disease groupings. This makes it broadly applicable, particularly in contexts where prior knowledge is unavailable or cannot be encoded. The model captures temporal dynamics often missed by static clustering approaches and delivers more interpretable results compared to deep generative models. The factorial structure allows for the representation of comorbidities as a superposition of multiple concurrent disease processes, ofering a better understanding of disease progression than single-process HMMs.

Looking forward, several directions could improve the clinical utility of this approach. While our model efectively characterizes T2DM progression patterns, real-time clinical implementation would require incorporation of additional comorbidities beyond the core set analyzed. Although prescribed medications were included in our initial analysis, they did not significantly improve model performance, likely due to challenges in capturing complex treatment patterns from EHR data. Future work could explore more sophisticated feature engineering of medication data. Nevertheless the interpretability demonstrated by our approach suggests strong potential for integration into real-world clinical workflows, where it could provide valuable insights into disease progression patterns and inform personalized care strategies for T2DM patients.

## 6 Funding

This research was funded by a Swiss Data Science Centre Collaboration Grant (C19-09).

## References

[1] D. Tomic, J. E. Shaw, and D. J. Magliano, “The burden and risks of emerging complications of diabetes mellitus,” Nature Reviews Endocrinology, vol. 18, pp. 525–539, Sep 2022.

[2] Z. Ghahramani and M. Jordan, “Factorial hidden markov models,” Advances in Neural Information Processing Systems, vol. 8, 1995.

[3] A. M. Alaa, S. Hu, and M. Schaar, “Learning from clinical judgments: Semi-markov-modulated marked hawkes processes for risk prognosis,” in International Conference on Machine Learning, pp. 60–69, PMLR, 2017.

[4] X. Wang, D. Sontag, and F. Wang, “Unsupervised learning of disease progression models,” in Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 85–94, 2014.

[5] O. Ben-Assuli, T. Heart, J. R. Vest, R. Ramon-Gonen, N. Shlomo, and R. Klempfner, “Profiling readmissions using hidden markov model-the case of congestive heart failure,” Information Systems Management, vol. 38, no. 3, pp. 237–249, 2021.

[6] Y.-Y. Liu, S. Li, F. Li, L. Song, and J. M. Rehg, “Eficient learning of continuous-time hidden markov models for disease progression,” Advances in neural information processing systems, vol. 28, 2015.

[7] T. Ceritli, A. P. Creagh, and D. A. Clifton, “Mixture of input-output hidden markov models for heterogeneous disease progression modeling,” in Workshop on Healthcare AI and COVID-19, pp. 41– 53, PMLR, 2022.

[8] F. J. Ruiz, I. Valera, C. Blanco, and F. Perez-Cruz, “Bayesian nonparametric comorbidity analysis of psychiatric disorders,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 1215–1247, 2014.

[9] C. Trottet, M. Sch¨urch, A. Mollaysa, A. Allam, and M. Krauthammer, “Generative time series models with interpretable latent processes for complex disease trajectories,” in Deep Generative Models for Health Workshop NeurIPS 2023, 2023.

[10] Z. Oflaz, C. Yozgatligil, and A. S. Selcuk-Kestel, “Modeling comorbidity of chronic diseases using coupled hidden markov model with bivariate discrete copula,” Statistical Methods in Medical Research, p. 09622802231155100, 2023.

[11] L. Fregoso-Aparicio, J. Noguez, L. Montesinos, and J. A. Garc´ıa-Garc´ıa, “Machine learning and deep learning predictive models for type 2 diabetes: a systematic review,” Diabetology & Metabolic Syndrome, vol. 13, no. 1, pp. 1–22, 2021.

[12] A. Martinez-De la Torre, F. Perez-Cruz, S. Weiler, and A. M. Burden, “Comorbidity clusters associated with newly treated type 2 diabetes mellitus: a bayesian nonparametric analysis,” Scientific Reports, vol. 12, no. 1, p. 20653, 2022.

[13] R. R. Kalyani, S. H. Golden, and W. T. Cefalu, “Diabetes and aging: Unique considerations and goals of care,” Diabetes Care, vol. 40, no. 4, pp. 440–443, 2017.

[14] G. T. Russo, A. Giandalia, E. L. Romeo, M. Nunziata, M. Muscianisi, M. C. Rufo, A. Catalano, and D. Cucinotta, “Fracture risk in type 2 diabetes: Current perspectives and gender diferences,” International Journal of Endocrinology, vol. 2016, p. 1615735, 2016.

[15] R. Schweiger, Y. Erlich, and S. Carmi, “FactorialHMM: fast and exact inference in factorial hidden Markov models,” Bioinformatics, vol. 35, pp. 2162–2164, 11 2018.

[16] J. Chiquet, Y. Grandvalet, and C. Charbonnier, “Sparsity with sign-coherent groups of variables via the cooperative-lasso,” The Annals of Applied Statistics, vol. 6, no. 2, pp. 795–830, 2012.

[17] M. Yuan and Y. Lin, “Model selection and estimation in regression with grouped variables,” Journal of the Royal Statistical Society: Series B (Statistical Methodology), vol. 68, no. 1, pp. 49–67, 2006.

[18] M. J. Wainwright and M. I. Jordan, “Graphical Models, Exponential Families, and Variational Inference,” Foundations and Trends in Machine Learning, vol. 1, pp. 1–305, Jan. 2008.

[19] T. S. Jaakkola and M. I. Jordan, “A variational approach to Bayesian logistic regression models and their extensions,” in Proceedings of the Sixth International Workshop on Artificial Intelligence and Statistics (D. Madigan and P. Smyth, eds.), vol. R1 of Proceedings of Machine Learning Research, pp. 283–294, PMLR, 04–07 Jan 1997. Reissued by PMLR on 30 March 2021.

[20] D. L. Davies and D. W. Bouldin, “A cluster separation measure,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-1, no. 2, pp. 224–227, 1979.

[21] R. Tibshirani and G. Walther, “Cluster validation by prediction strength,” Journal of Computational and Graphical Statistics, vol. 14, no. 3, pp. 511–528, 2005.

[22] H. Akaike, “A new look at the statistical model identification,” IEEE Transactions on Automatic Control, vol. 19, no. 6, pp. 716–723, 1974.

[23] G. Schwarz, “Estimating the dimension of a model,” The Annals of Statistics, vol. 6, no. 2, pp. 461– 464, 1978.

[24] Z. M. Younossi, A. B. Koenig, D. Abdelatif, Y. Fazel, L. Henry, and M. Wymer, “Global epidemiology of nonalcoholic fatty liver disease—meta-analytic assessment of prevalence, incidence, and outcomes,” Hepatology, vol. 64, no. 1, pp. 73–84, 2016.

[25] O. Aalen, “Nonparametric estimation of partial transition probabilities in multiple decrement mod els,” The Annals of Statistics, vol. 6, no. 3, pp. 534–545, 1978.

[26] C. Ronco, P. McCullough, S. D. Anker, I. Anand, N. Aspromonte, S. M. Bagshaw, R. Bellomo, T. Berl, I. Bobek, D. Cruz, et al., “Cardiorenal syndrome,” Journal of the American College of Cardiology, vol. 52, no. 19, pp. 1527–1539, 2008.

[27] N. Cheung, P. Mitchell, and T. Y. Wong, “Diabetic retinopathy,” Lancet, vol. 376, pp. 124–136, July 2010. Epub 2010 Jun 26.

[28] R. Lee, T. Y. Wong, and C. Sabanayagam, “Epidemiology of diabetic retinopathy, diabetic macular edema and related vision loss,” Eye and Vision (London, England), vol. 2, p. 17, 2015.

[29] P. Vestergaard, “Discrepancies in bone mineral density and fracture risk in patients with type 1 and type 2 diabetes—a meta-analysis,” Osteoporosis International, vol. 18, pp. 427–444, Apr. 2007.

[30] N. Napoli, M. Chandran, D. D. Pierroz, B. Abrahamsen, A. V. Schwartz, and S. L. Ferrari, “Mechanisms of diabetes mellitus-induced bone fragility,” Nature Reviews Endocrinology, vol. 13, pp. 208– 219, Apr. 2017.

## A Data preparation

<table><tr><td>Lab Value</td><td>Low</td><td>High</td><td>Comment</td></tr><tr><td>Glomerular filtration rate</td><td>0</td><td>150</td><td></td></tr><tr><td>Hb A1C</td><td>0</td><td>16</td><td></td></tr><tr><td>High Density Lipoprotein</td><td>0</td><td>5</td><td></td></tr><tr><td>Low Density Lipoprotein</td><td>0</td><td>10</td><td></td></tr><tr><td>Triglycerides</td><td>0</td><td>15</td><td></td></tr><tr><td>BMI</td><td>0</td><td>80</td><td></td></tr><tr><td>Diastolic</td><td>30</td><td>150</td><td>Lower than systolic</td></tr><tr><td>Systolic</td><td>80</td><td>240</td><td>Higher than diastolic</td></tr></table>

Table 5: Predefined thresholds used to remove outliers. Values are removed if they are not included in the lower and upper limits.

<table><tr><td></td><td colspan="3">Men &lt; 65</td><td colspan="3">Men 65+</td><td colspan="3">Women  $< 6 5$ </td><td colspan="3">Women 65+</td></tr><tr><td></td><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>COPD</td><td>7.7</td><td>7.6</td><td>8.0</td><td>16.4</td><td>16.1</td><td>17.0</td><td>7.8</td><td>7.9</td><td>7.9</td><td>13.3</td><td>12.8</td><td>15.6</td></tr><tr><td>Diabetic retinopathy</td><td>32.8</td><td>33.4</td><td>32.9</td><td>34.8</td><td>36.3</td><td>36.4</td><td>30.4</td><td>31.5</td><td>31.1</td><td>34.6</td><td>33.8</td><td>35.7</td></tr><tr><td>Hypertension</td><td>59.5</td><td>59.7</td><td>61.3</td><td>71.2</td><td>71.0</td><td>71.9</td><td>56.9</td><td>57.0</td><td>57.1</td><td>78.6</td><td>79.9</td><td>79.2</td></tr><tr><td>Myocardial infarction</td><td>8.8</td><td>9.2</td><td>9.0</td><td>16.6</td><td>17.1</td><td>18.5</td><td>3.0</td><td>2.8</td><td>2.8</td><td>7.9</td><td>6.8</td><td>7.3</td></tr><tr><td>Peripheral vascular disease</td><td>4.2</td><td>3.8</td><td>4.5</td><td>11.0</td><td>11.1</td><td>10.5</td><td>1.7</td><td>1.8</td><td>1.6</td><td>5.5</td><td>6.2</td><td>5.7</td></tr><tr><td>Stroke</td><td>6.2</td><td>6.4</td><td>6.7</td><td>17.6</td><td>17.6</td><td>17.6</td><td>5.1</td><td>5.1</td><td>5.0</td><td>15.8</td><td>14.2</td><td>14.4</td></tr><tr><td>Osteoporosis</td><td>0.7</td><td>0.6</td><td>0.7</td><td>2.2</td><td>2.3</td><td>2.1</td><td>2.9</td><td>3.1</td><td>2.6</td><td>11.8</td><td>11.8</td><td>11.8</td></tr><tr><td>Chronic kidney disease</td><td>5.9</td><td>5.9</td><td>5.9</td><td>11.5</td><td>12.7</td><td>10.7</td><td>6.9</td><td>6.5</td><td>6.5</td><td>10.3</td><td>9.1</td><td>9.6</td></tr><tr><td>Chronic liver disease</td><td>4.8</td><td>4.8</td><td>4.8</td><td>2.7</td><td>2.4</td><td>3.3</td><td>6.2</td><td>6.2</td><td>6.2</td><td>2.9</td><td>3.5</td><td>3.3</td></tr><tr><td>Congestive heart failure</td><td>4.5</td><td>4.5</td><td>4.5</td><td>13.3</td><td>14.0</td><td>14.2</td><td>2.3</td><td>2.2</td><td>2.4</td><td>10.6</td><td>10.6</td><td>10.5</td></tr><tr><td>Death</td><td>5.2</td><td>5.4</td><td>5.2</td><td>22.3</td><td>22.2</td><td>22.2</td><td>4.1</td><td>4.1</td><td>3.9</td><td>19.4</td><td>19.1</td><td>20.1</td></tr></table>

Table 6: Number of patient diagnosed with the comorbidity for each data split. Numbers are given in percentage.

## B Variational approximation

This section provides details on the fitting algorithm used to train the FHMM. The fitting algorithm mainly follows [2], where the authors reduce the inference complexity to $O ( K ^ { 2 } M T )$ by exploiting a structured mean field variational approximation [18] for the posterior distribution $q _ { \phi }$ used in the E step of their EM procedure. In particular, due to their specific choice, (6) can be written as

$$
\boldsymbol { F } ( \boldsymbol { q } _ { \phi } , \boldsymbol { \theta } ) = \mathbb { E } _ { \boldsymbol { q } _ { \phi } } \left[ \sum _ { t = 1 } ^ { T } \log p _ { \theta } \left( \boldsymbol { y } _ { t } | \tilde { H } _ { t } ^ { ( 1 : M ) } \right) - \sum _ { t = 1 } ^ { T } \sum _ { m = 1 } ^ { M } ( \tilde { H } _ { t } ^ { ( m ) } ) ^ { \top } \log e _ { t } ^ { ( m ) } \right] - \boldsymbol { Z } + \boldsymbol { Z } _ { \boldsymbol { q } _ { \phi } } ,
$$

where $Z$ and $Z _ { q _ { \phi } }$ are respectively normalizing constants of $p _ { \theta } ( y _ { 1 : T } | h _ { 1 : T } , x _ { 1 : T } )$ and $q _ { \phi } ( h _ { 1 : T } \mid x _ { 1 : T } )$ and where $\tilde { H } _ { t } ^ { ( m ) }$ is a one-hot encoded version of $H _ { t } ^ { ( m ) }$

The E step requires the optimization of the parameters $\phi ,$ which can be computing by solving

$$
\left. \frac { \partial F } { \partial \log e _ { \tau } ^ { ( n ) } } \right| _ { q _ { \hat { \phi } } , \theta } = 0 ,
$$

where the gradient is

$$
\left. \frac { \partial F } { \partial \log e _ { \tau } ^ { ( n ) } } \right| _ { q _ { \phi } , \theta } = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { q _ { \phi } } \left[ \left\{ \log p _ { \theta } \left( y _ { t } | \tilde { H } _ { t } ^ { ( 1 : M ) } \right) - \sum _ { m = 1 } ^ { M } \left( \tilde { H } _ { t } ^ { ( m ) } \right) ^ { \top } \log e _ { t } ^ { ( m ) } \right\} \times \left\{ \tilde { H } _ { \tau } ^ { ( n ) } - \mathbb { E } _ { q _ { \phi } } \left[ \tilde { H } _ { \tau } ^ { ( n ) } \right] \right\} \right] .\tag{9}
$$

The M step requires the optimization of $\mathbb { E } _ { q _ { \phi } }$ [log $p _ { \theta } { \left( y _ { 1 : T } , h _ { 1 : T } \right) } \big ]$ with respect to $\theta ^ { \pi } , \theta ^ { P } , \eta .$ . Each parameter can be optimized individually by solving

$$
\begin{array} { r l } & { \displaystyle \hat { \theta } ^ { \pi } = \arg \operatorname* { m a x } _ { \theta ^ { \pi } } \sum _ { m } \gamma _ { 1 } ^ { ( m ) ^ { \top } } \log \pi _ { \theta ^ { \pi } } ^ { ( m ) } , } \\ & { \displaystyle \hat { \theta } ^ { P } = \arg \operatorname* { m a x } _ { \theta ^ { P } } \sum _ { m } \sum _ { t } \log P _ { \theta ^ { P } } ^ { ( m ) } \operatorname { T r } ( \xi _ { t } ^ { ( m ) } ) , } \\ & { \displaystyle \hat { \eta } = \arg \operatorname* { m a x } _ { \eta } \sum _ { t } \mathbb { E } _ { q _ { \phi } } \left[ \log p _ { \eta } ( y _ { t } \mid h _ { t } ^ { ( 1 : M ) } ) \right] , } \end{array}\tag{10}
$$

where

$$
\begin{array} { r l } & { \gamma _ { t } ^ { ( m ) } \ { \stackrel { \mathrm { d e f } } { = } } \ \mathbb { E } _ { q _ { \phi } } \left[ \tilde { H } _ { t } ^ { ( m ) } \right] \in \mathbb { R } ^ { K } , } \\ & { \zeta _ { t } ^ { ( m ) } \ { \stackrel { \mathrm { d e f } } { = } } \ \mathbb { E } _ { q _ { \phi } } \left[ \tilde { H } _ { t } ^ { ( m ) } \tilde { H } _ { t - 1 } ^ { ( m ) } \ ^ { \top } \right] \in \mathbb { R } ^ { K \times K } , } \end{array}
$$

are computed using the Baum–Welch algorithm for HMMs (see [2]), whose complexity is $O ( K ^ { 2 } T )$

Equations (9) and (10) involves expectations that are expensive to compute. In particular, it requires the computation of $q ( H _ { t _ { 1 } } ^ { ( m ) } , H _ { t _ { 2 } } ^ { ( m ) } \mid x _ { 1 : T } )$ for any $t _ { 1 } , t _ { 2 } .$ which has a memory and computational complexity of $O ( M T ^ { 2 } K ^ { 2 } )$ . Inspired by the Gaussian case, for which analytical expressions exists (see [2]), a second order approximation of the emission log-likelihood log $p _ { \eta } ( y _ { t } \mid h _ { t } ^ { ( 1 : M ) } )$ is derived using the inequality (8). In that case, the gradient is reduced to

$$
\begin{array} { r l } & { \frac { \partial F } { \partial \log e _ { \tau } ^ { ( n ) } } \Bigg | _ { q , \theta } \approx \sum _ { t } \mathrm { C o v } \left( \tilde { H } _ { \tau } ^ { ( n ) } , \tilde { H } _ { t } ^ { ( n ) } \right) \times \left\{ y _ { t } \eta ^ { ( n ) } - \eta ^ { ( n ) } \left( \frac { 1 } { 2 } + \lambda ( z ) \eta ^ { ( n ) } \eta ^ { \top } \gamma _ { t } \right) \right. } \\ & { \qquad \left. - \frac { \lambda ( z ) } { 2 } \left( \mathrm { D i a g } ( \eta ^ { ( n ) } \eta ^ { ( n ) } ) ^ { \top } ) - 2 \eta ^ { ( n ) } \eta ^ { ( n ) } \eta ^ { ( n ) } \right) - \log e _ { t } ^ { ( n ) } \right\} . } \end{array}\tag{11}
$$

Finding the zeros of (11) (E step) can then be easily done without computing the expensive terms Cov $\left( \tilde { H } _ { \tau } ^ { ( n ) } , \tilde { H } _ { t } ^ { ( n ) } \right)$ . On the other hand, the M step for binary outputs becomes

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \eta } \mathbb { E } _ { q _ { \varphi } } \left[ \log p _ { \eta } ( y _ { t } \mid h _ { t } ^ { ( 1 : M ) } ) \right] = \operatorname* { m a x } _ { \eta } \mathbb { E } _ { q _ { \varphi } } \left[ y _ { t } \tilde { H } _ { t } ^ { \top } \eta - \log ( 1 + \exp ( \tilde { H } _ { t } ^ { \top } \eta ) ) \right] \geq } & { } \\ { \displaystyle \operatorname* { m a x } _ { \eta } \left( y _ { t } - \frac { 1 } { 2 } \right) \gamma _ { t } ^ { \top } \eta - \log \left\{ 1 + \exp ( - z ) \right\} - z / 2 - \frac { \lambda ( z ) } { 2 } \left( \eta ^ { \top } \mathbb { E } \left[ \tilde { H } _ { t } \tilde { H } _ { t } ^ { \top } \right] \eta - z ^ { 2 } \right) , } & { } \end{array}\tag{12}
$$

where $\tilde { H } _ { t } = \left( \tilde { H } _ { t } ^ { ( 1 ) } , \dots , \tilde { H } _ { t } ^ { ( M ) } \right)$ and $\gamma _ { t } = \left( \gamma _ { t } ^ { ( 1 ) } , \dots , \gamma _ { t } ^ { ( M ) } \right)$ . Note that Equation 12 is a lower bound of the ELBO, which can be optimized with resepect to z, which admits a closed form solution

$$
z _ { t } = \sqrt { \eta ^ { \top } \mathbb { E } \left[ \tilde { H } _ { t } \tilde { H } _ { t } ^ { \top } \right] \eta } .
$$

## C Input medications

When drugs are used as input, the entries of the matrix $P _ { \theta ^ { P } }$ are given

$$
[ P _ { \theta ^ { P } } ] _ { i j } \propto \exp ( \alpha _ { i j } ^ { \mathrm { d r u g } } + f _ { l } ( X _ { t } , i ) \mathbb { 1 } _ { \{ i > j \} } + f _ { r } ( X _ { t } , i ) \mathbb { 1 } _ { \{ j > i \} } ) ,
$$

where $f _ { l } ( X _ { t } , i ) = W _ { l , i } ^ { \mathrm { d r u g } } \cdot X _ { t } \geq 0$ and $f _ { r } ( X _ { t } , i ) = W _ { r , i } ^ { \mathrm { d r u g } } \cdot X _ { t }$ . This parametrization captures the fact that drugs can either add probability mass on lower state values $( W _ { l , i } ^ { \mathrm { d r u g } } > 0 )$ or add probability mass on higher state values $( W _ { r , i } ^ { \mathrm { d r u g } } > 0 )$ . It is important to note that adding probability mass to lower state values will automatically reduce the probability mass on higher state values, and vice versa. In addition, $W _ { l , i } ^ { \mathrm { d r u g } } , W _ { r , i } ^ { \mathrm { d r u g } }$ , follow a priori a Gaussian distribution centered at 0, which is equivalent to add the penalizations $\| W _ { l , i } ^ { \mathrm { d r u g } } \| _ { 2 } ^ { 2 }$ to the objective function. The underlying assumptions is that the drug efect is a priori small $( \mathrm { i . e . , } W _ { l , i } ^ { \mathrm { d r u g } }$ is likely to be small).

Table 7 shows the performance of the model when drugs are used as input. The use of drugs in the model, is marginal and it does not improve the BIC score. The AIC is improved in few cases. Figure 11 show how the input variables afect the evolution of the latent state for the model FHMM(8, 4)<sub>2</sub> fitted on the male population younger than 65. Age generally has a negative efect on chain transitions, increasing the probability of moving to a worsening state. Smoking also accelerates progression along chain 4, which is associated with a higher risk of COPD. The use of calcium channel blockers is linked to chain 5, which models hypertension. Here we observe an apparent worsening efect, although this may simply reflect that the model associates higher states with greater use of calcium channel blockers. By contrast, the addition of a new antidiabetic drug typically produces a positive efect on the patient’s state.

<table><tr><td rowspan="2">Category</td><td rowspan="2">Metric Model</td><td colspan="2">BIC</td><td colspan="2">AIC</td></tr><tr><td>Drugs</td><td>No Drugs</td><td>Drugs</td><td>No Drugs</td></tr><tr><td rowspan="5">Men &lt; 65</td><td>FHMM(3, 3)1</td><td>315914</td><td>310663</td><td>308086</td><td>307445</td></tr><tr><td>FHMM(4, 4)2</td><td>308458</td><td>292039</td><td>294662</td><td>286438</td></tr><tr><td>FHMM(6, 4)2</td><td>291660</td><td>264231</td><td>271273</td><td>256136</td></tr><tr><td>FHMM(8, 4)2</td><td>280861</td><td>273337</td><td>253882</td><td>262748</td></tr><tr><td>FHMM(10, 2)1</td><td>286149</td><td>276474</td><td>270171</td><td>270739</td></tr><tr><td rowspan="5">Women &lt; 65</td><td>FHMM(3, 3)1</td><td>231096</td><td>225526</td><td>223455</td><td>222385</td></tr><tr><td>FHMM(4, 4)2</td><td>231223</td><td>216236</td><td>217757</td><td>210769</td></tr><tr><td>FHMM(6, 4)2</td><td>212362</td><td>190847</td><td>192462</td><td>182946</td></tr><tr><td>FHMM(8, 4)2</td><td>203653</td><td>178959</td><td>177320</td><td>168623</td></tr><tr><td>FHMM(10, 2)1</td><td>218704</td><td>211039</td><td>203108</td><td>205441</td></tr><tr><td rowspan="5">Men 65+</td><td>FHMM(3, 3)1</td><td>229811</td><td>229629</td><td>222198</td><td>226500</td></tr><tr><td>FHMM(4, 4)2</td><td>228468</td><td>221682</td><td>215051</td><td>216235</td></tr><tr><td>FHMM(6, 4)2</td><td>218800</td><td>201848</td><td>198973</td><td>193976</td></tr><tr><td>FHMM(8, 4)2</td><td>203550</td><td>181179</td><td>177312</td><td>170881</td></tr><tr><td>FHMM(10, 2)1</td><td>204037</td><td>192274</td><td>188498</td><td>186698</td></tr><tr><td rowspan="5">Women 65+</td><td>FHMM(3, 3)1</td><td>208526</td><td>203633</td><td>201026</td><td>200550</td></tr><tr><td>FHMM(4, 4)2</td><td>204033</td><td>195206</td><td>190813</td><td>189839</td></tr><tr><td>FHMM(6, 4)2</td><td>195974</td><td>181320</td><td>176439</td><td>173563</td></tr><tr><td>FHMM(8, 4)2</td><td>180633</td><td>162739</td><td>154781</td><td>152592</td></tr><tr><td>FHMM(10, 2)1</td><td>184247</td><td>174079</td><td>168937</td><td>168585</td></tr></table>

Table 7: Comparison between models with drugs $( X _ { t } )$ and without drugs. Metrics are evaluated on the validation set.

## D Additional Figures

The number of clusters is determined using the Davies-Bouldin index [20] and the prediction strength introduced by [21]. The prediction strength is estimated using a 3-fold cross-validation setup, where two-thirds of the data is used for training and one-third for testing.

![](images/f96ae3403fedd9c0d8b588da061ed209623da4303377a00912aab51c88cec263.jpg)

![](images/c82356437d70a7d8b9db697be386aee309dcfa2a3a2f1f6762d0ebb1bb1d5de7.jpg)  
Figure 11: Top: Average efect of drugs on the transition matrix, estimated as $1 / K \textstyle \sum _ { i = 1 } ^ { K } \left( W _ { r , i } ^ { \mathrm { d r u g } } - W _ { l , i } ^ { \mathrm { d r u g } } \right)$ Red (resp. blue) means an increase of the probability to transition to a higher (resp. lower) state value. Bottom: Emission parameters η modeling the risk of developing a comorbidity for a given state. Larger values indicate a higher probability of having the comorbidity. The model includes 8 chains with 4 states per chain and is fitted on $M e n < 6 5$

![](images/abacc3371568f3da3f15e933cdd88387685ed4f4ac9c5bc2ba11f4c7814c733b.jpg)

![](images/0df7eebf66976af9a1bc587c06823ea3624ec64928f99521a2156d7e393b209e.jpg)  
Figure 12: Cluster selection using Davies-Bouldin index and the prediction strength. The optimal number of cluster is indicated by the black dashed line. This is fitted on the population of Women < 65.

![](images/d8769f95c49fd18bfcdb836987d7cbaf99245caba53af55db6a0f6857015c16a.jpg)

![](images/f3ea164ffe71f23204d627f1767a22cd6854e1c0f4348a8d77dd78295ec96481.jpg)  
Cluster Evolution starting from 11

![](images/a9178299c2c98dc1bcf7155334302498b6555e2808250be24d3e396247ade08a.jpg)

![](images/effbc8be98a668508477e13560273eba70d2b7ff07ce13975ec28614b8862cf0.jpg)

![](images/dd1867b6a15c44aaabfb60cab02cdcc464ce3a67acea7915d0992e3c27f625aa.jpg)

![](images/56652a5011403e0a9070c1f387eea423ee049681400939135f9a320af7f78f82.jpg)

![](images/b35920f8e743bfa80787762a17178b22da7c10b6d9b72cb727d240b33d08b7de.jpg)

![](images/e1378859567579c718c7c543deb94db231ee8f9c1aad44d698d27c9d63905767.jpg)

![](images/8f0d7a0da46a338a8c6ea1b89c30b86e08dfca2db8437746aee879ae8ad404d9.jpg)

![](images/69006712dc237c6d0fc79921bd512eb2857a5b3a916292077875d7eb6d134463.jpg)  
Figure 13: Evolution of patient clusters 0 (healthy), 2 (pathway to multimorbidity), 3 (pulmonarymetabolic), 4 (isolated hypertension), 6 (hepatic), 8 (advanced nephropathy), 9 (obese hypertension), 10 (bone-vascular), 11 (isolated DR), and 12 (cardio-pulmonary multimorbidity) over time (Women < 65), conditional on the initial cluster assignment (6 months post-NIAD). The number of patients in each cluster is indicated alongside the corresponding bars. For clarity, flows representing less than 1% of the initial cluster size, as well as transitions to no available follow-up data, are not displayed. Clusters 1, 5, and 7 are excluded due to the low number of transitions observed from these clusters.

<table><tr><td>Cluster</td><td>0</td><td></td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td></td><td>14</td></tr><tr><td># of patients</td><td>6504</td><td>1788</td><td>1698</td><td>1166</td><td>3911</td><td>1522</td><td>360</td><td>1212</td><td>957</td><td>1534</td><td>1010</td><td>1511</td><td>1070</td><td>13 703</td><td>685</td></tr><tr><td>Avg. age at NIAD (y) ± SD</td><td>50.88 ± 8.68</td><td>53.66 ± 8.11</td><td>55.28 ± 6.95</td><td>57.41 ± 6.39</td><td>53.89 ± 7.32</td><td>56.18 ± 6.73</td><td>58.28 ± 5.63</td><td>53.21 ± 7.78</td><td>57.69 ± 5.92</td><td>57.42 ± 6.14</td><td>55.41 ± 7.21</td><td>51.53 ± 8.35</td><td>56.68 ± 6.42</td><td>55.89 ± 7.02</td><td>54.0 ± 7.97</td></tr><tr><td>Avg. follow-up post-NIAD (y) ± SD</td><td>4.28 ± 3.1</td><td>6.85 ± 3.25</td><td>5.78 ± 2.86</td><td>5.38 ± 3.11</td><td>4.72 ± 3.22</td><td>6.52 ± 3.27</td><td>4.49 ± 2.89</td><td>5.97 ± 3.06</td><td>5.46 ± 3.59</td><td>5.58 ± 3.57</td><td>3.63 ± 2.85</td><td>5.29 ± 2.75</td><td>6.25 ± 2.88</td><td>5.05 ± 3.08</td><td>6.29 ± 2.64</td></tr><tr><td>Avg. BMI at NIAD (kg/m2) ± SD</td><td>30.85 ± 5.52</td><td>34.96 ± 6.67</td><td>31.58 ± 4.79</td><td>34.31 ± 6.68</td><td>32.47 ± 5.32</td><td>30.86 ± 4.48</td><td>32.25 ± 6.67</td><td>43.79 ± 5.7</td><td>32.82 ± 4.93</td><td>33.54 ± 4.23</td><td>33.08 ± 6.39</td><td>30.38 ± 4.95</td><td>36.46 ± 6.85</td><td>32.01 ± 6.18</td><td>34.96 ± 6.77</td></tr><tr><td>Avg. HbA1c at NIAD (%) ± SD</td><td>7.45 ± 1.73</td><td>7.39 ± 1.61</td><td>7.47 ± 1.59</td><td>7.31 ± 1.64</td><td>7.23 ± 1.56</td><td>7.19 ± 1.6</td><td>7.31 ± 1.61</td><td>7.44 ± 1.63</td><td>7.36 ± 1.7</td><td>7.2 ± 1.45</td><td>7.4 ± 1.71</td><td>7.69 ± 1.7</td><td>7.48 ± 1.59</td><td>7.47 ± 1.63</td><td>7.59 ± 1.65</td></tr><tr><td>% Ever smoked</td><td>43.48</td><td>53.75</td><td>32.63</td><td>62.86</td><td>39.43</td><td>34.82</td><td>71.67</td><td>39.69</td><td>53.81</td><td>44.72</td><td>55.15</td><td>39.38</td><td>44.11</td><td>70.13</td><td>53.87</td></tr><tr><td>% Ever alcohol</td><td>84.62</td><td>88.59</td><td>92.34</td><td>92.02</td><td>91.2</td><td>89.49</td><td>93.06</td><td>90.68</td><td>89.97</td><td>92.57</td><td>90.79</td><td>89.61</td><td>93.74</td><td>87.77</td><td>91.82</td></tr><tr><td>Comorbidities at NIAD [N,(%)]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stroke</td><td>86 (1.32)</td><td>46 (2.57)</td><td>62 (3.65)</td><td>82 (7.03)</td><td>&lt;1%</td><td>34 (2.23)</td><td>37 (10.28)</td><td>&lt; 1 %</td><td>32 (3.34)</td><td>258 (16.82)</td><td>55 (5.45)</td><td>18 (1.19)</td><td>57 (5.33)</td><td>25 (3.56)</td><td>30 (4.38)</td></tr><tr><td>Myocardial infarction</td><td>&lt;1%</td><td>370 (20.69)</td><td>&lt;1%</td><td>99 (8.49)</td><td>&lt;1%</td><td>&lt; 1 %</td><td>63 (17.50)</td><td>76 (6.27)</td><td>517 (54.02)</td><td>30 (1.96)</td><td>61 (6.04)</td><td>&lt;1 %</td><td>201 (18.79)</td><td>33 (4.69)</td><td>162 (23.65)</td></tr><tr><td>Congestive heart failure</td><td>&lt; 1 %</td><td>&lt; 1%</td><td>&lt;1%</td><td>95 (8.15)</td><td>&lt;1%</td><td>&lt; 1%</td><td>35 (9.72)</td><td>46 (3.80)</td><td>231 (24.14)</td><td>&lt;1 %</td><td>24 (2.38)</td><td>&lt; 1%</td><td>103 (9.63)</td><td>&lt; 1 %</td><td>&lt; 1%</td></tr><tr><td>Peripheral vascular disease</td><td>&lt; 1 %</td><td>54 (3.02)</td><td>&lt; 1%</td><td>&lt;1%</td><td>&lt; 1%</td><td>&lt; 1%</td><td>108 (30.00)</td><td>&lt;1 %</td><td>&lt;1%</td><td>&lt; 1 %</td><td>230 (22.77)</td><td>&lt; 1 %</td><td>15 (1.40)</td><td>7 (1.00)</td><td>21 (3.07)</td></tr><tr><td>Diabetic retinopathy</td><td>159 (2.44)</td><td>&lt; 1%</td><td>931 (54.83)</td><td>188 (16.12)</td><td>97 (2.48)</td><td>&lt;1%</td><td>61 (16.94)</td><td>&lt;1%</td><td>22 (2.30)</td><td>50 (3.26)</td><td>144 (14.26)</td><td>714 (47.25)</td><td>556 (51.96)</td><td>87 (12.38)</td><td>352 (51.39)</td></tr><tr><td>Hypertension</td><td> $ { \mathrm { ~ 3 1 0 ~ } } _ { , 1 , 7 7 } )$ </td><td>19 (1.06)</td><td>1573 (92.64)</td><td>1012 (86.79)</td><td>3398 (86.88)</td><td>1428 (93.82)</td><td>207 (57.50)</td><td>1080 (89.11)</td><td>721 (75.34)</td><td>1363 (88.85)</td><td>700 (69.31)</td><td>29 (1.92)</td><td>945 (88.32)</td><td>27 (3.84)</td><td>&lt; 1 %</td></tr><tr><td>Chronic kidney disease</td><td>&lt; 1 %</td><td>&lt;1% &lt; 1 %</td><td>&lt; 1 % &lt;1%</td><td>434 (37.22) &lt; 1 %</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>57 (15.83)</td><td>&lt; 1 %</td><td>&lt; 1%</td><td>&lt; 1 %</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt;1%</td><td>198 (28.17)</td><td>&lt;1%</td></tr><tr><td>Osteoporosis</td><td>&lt;1%</td><td></td><td></td><td>500 (42.88)</td><td>&lt;1%</td><td>&lt; 1 % &lt; 1%</td><td>96 (26.67)</td><td>&lt; 1 %</td><td>&lt; 1%</td><td>&lt;1%</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>&lt; 1 %</td></tr><tr><td>COPD</td><td>&lt;1%</td><td>&lt; 1%</td><td>&lt; 1%</td><td></td><td>&lt;1%</td><td></td><td>158 (43.89)</td><td>&lt;1%</td><td>&lt; 1 %</td><td>&lt;1%</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>&lt; 1%</td><td>343 (48.79)</td><td>7 (1.02)</td></tr><tr><td>Chronic liver disease</td><td> $8 1 \ ( 1 . 2 5 )$ </td><td>38 (2.13)</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt; 1%</td><td>75 (20.83)</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt;1%</td><td>411 (40.69)</td><td>36 (2.38)</td><td>&lt;1%</td><td>15 (2.13)</td><td>10 (1.46)</td></tr><tr><td>Comorbidities during follow-up [N,(%)]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stroke</td><td>&lt;1%</td><td>47 (2.63)</td><td>34 (2.00)</td><td>43 (3.69)</td><td>42 (1.07)</td><td>35 (2.30)</td><td>10 (2.78)</td><td>&lt;1%</td><td>39 (4.08)</td><td>84 (5.48)</td><td>19 (1.88)</td><td>18 (1.19)</td><td>32 (2.99)</td><td>14 (1.99)</td><td>22 (3.21)</td></tr><tr><td>Myocardial infarction</td><td>&lt; 1 %</td><td>107 (5.98)</td><td>&lt; 1%</td><td>25 (2.14)</td><td>&lt;1%</td><td>&lt; 1 %</td><td>8 (2.22)</td><td>39 (3.22)</td><td>58 (6.06)</td><td>58 (3.78)</td><td>25 (2.48)</td><td>&lt;1%</td><td>58 (5.42)</td><td>11 (1.56)</td><td>33 (4.82)</td></tr><tr><td>Congestive heart failure</td><td>&lt;1%</td><td>58 (3.24)</td><td>&lt;1%</td><td>65 (5.57)</td><td>&lt;1%</td><td>&lt;1%</td><td>24 (6.67)</td><td>42 (3.47)</td><td>95 (9.93)</td><td>31 (2.02)</td><td>13 (1.29)</td><td>&lt;1%</td><td>79 (7.38)</td><td>11 (1.56)</td><td>16 (2.34)</td></tr><tr><td>Peripheral vascular disease</td><td>&lt; 1 %</td><td>47 (2.63)</td><td>20 (1.18)</td><td>32 (2.74)</td><td>&lt;1%</td><td>21 (1.38)</td><td>27 (7.50)</td><td>&lt;1%</td><td>30 (3.13)</td><td>22 (1.43)</td><td>72 (7.13)</td><td>&lt;1%</td><td>21 (1.96)</td><td>9 (1.28)</td><td>19 (2.77)</td></tr><tr><td>Diabetic retinopathy</td><td>700 (10.76)</td><td>263 (14.71)</td><td>767 (45.17)</td><td>256 (21.96)</td><td>457 (11.68)</td><td>224 (14.72)</td><td>72 (20.00)</td><td>171 (14.11)</td><td>116 (12.12)</td><td>192 (12.52)</td><td>169 (16.73)</td><td>797 (52.75)</td><td>514 (48.04)</td><td>130 (18.49)</td><td>333 (48.61)</td></tr><tr><td>Hypertension</td><td>585 (8.99)</td><td>202 (11.30)</td><td>113 (6.65)</td><td>75 (6.43)</td><td>395 (10.10)</td><td>90 (5.91)</td><td>24 (6.67)</td><td>88 (7.26)</td><td>38 (3.97)</td><td>91 (5.93)</td><td>75 (7.43)</td><td>160 (10.59)</td><td>54 (5.05)</td><td>58 (8.25)</td><td>54 (7.88)</td></tr><tr><td>Chronic kidney disease</td><td>66 (1.01) &lt; 1 %</td><td>35 (1.96) &lt; 1%</td><td>32 (1.88) &lt; 1 %</td><td>165 (14.15)</td><td>44 (1.13)</td><td>36 (2.37)</td><td>40 (11.11)</td><td>39 (3.22)</td><td>31 (3.24)</td><td>27 (1.76)</td><td>13 (1.29) &lt; 1 %</td><td>22 (1.46) &lt; 1%</td><td>47 (4.39) &lt; 1 %</td><td>102 (14.51) &lt; 1 %</td><td>21 (3.07) &lt; 1 %</td></tr><tr><td>Osteoporosis COPD</td><td>&lt; 1 %</td><td>62 (3.47)</td><td>21 (1.24) 37 (2.18)</td><td>&lt; 1 % 134 (11.49) 23 (1.97)</td><td>&lt; 1 % 48 (1.23) 50 (1.28)</td><td>&lt; 1 % 21 (1.38) 24 (1.58)</td><td>20 (5.56) 36 (10.00) 27 (7.50)</td><td>&lt; 1 % 26 (2.15) 27 (2.23)</td><td>&lt; 1% 16 (1.67) 33 (3.45)</td><td>&lt; 1% 39 (2.54)</td></table>

Table 8: Patient characteristics in each cluster for the $M e n < 6 5$ cohort with average values and standard deviation where applicable. Comorbidities are presented as absolute counts at NIAD initiation, along with the number of new comorbidities developed post-NIAD. Proportions, expressed as percentages of the cohort population, are reported in parentheses.

![](images/d5b9fd1d543076a9f0154a54a849821b95a8a001f34dbf3409c63694b6a4fc9a.jpg)  
Figure 14: Average cluster-specific probabilities and lab values for $M e n < 6 5$ . Values have been centered by subtracting the average at the time of NIAD initiation to improve readability. Red bars indicate increases, and blue bars indicate decreases relative to the average patient at NIAD. Comorbidity risks are shown as changes in probability (left axis), and lab values as percentage changes relative to the reference value (right axis). Annotated numbers indicate absolute changes in the original units (see Table 1). Error bars represent the standard error of the mean.

Cluster Evolution starting from 0  
![](images/3b25d766a463f73b02d48aadca95eb085da42c07c9f21e4cb56fbcfb2b7340b6.jpg)  
Cluster Evolution starting from 3

Cluster Evolution starting from  
Cluster Evolution starting from 2  
![](images/93c2874ec0951ff095933ee5dc3b4993d814c386ea325907237a96d2f1ba6cc9.jpg)

![](images/4e690798b23000bf8378e4797bbaba46648a399ade617d0bdd9fd600d8900493.jpg)

![](images/3dd0d70f06e1dd65e3901b63fda643ac58b81384a9325183a4b798910eae30cf.jpg)  
Cluster Evolution starting from 5

Cluster Evolution starting from 7  
Cluster Evolution starting from 4  
![](images/091c06c086ee2ce39204697d93bef67cf0314d666e0dea17600f4c3966c6570b.jpg)  
Cluster Evolution starting from 8

![](images/3aeb24cc72263688d531047843c6a29cadcd82f0cd69503621c810316150640c.jpg)

![](images/d3ae0dbc0841170d5c9e63acefc6dfb2611034af476809781fcd5892c169a670.jpg)  
Cluster Evolution starting from 10

![](images/9fb46628722b87653a4d8f8ad008c2004a8833f3efb40cea2072797712ef3c86.jpg)  
Cluster Evolution starting from 11

Cluster Evolution starting from 9  
![](images/d1627e9766b41547c56f5304f3d1cc8019209a184af6f363655674bfc5e42f78.jpg)  
Cluster Evolution starting from 12

![](images/f51b6f92574d598bf92d55b3ae12c04e75735866c7f170bacd71b0c6081fbbee.jpg)  
Cluster Evolution starting from 13

![](images/028edacc5080afc84229b4142e4bc159396ffd102877a9e1a809f7b85fcd0c80.jpg)  
Cluster Evolution starting from 14

![](images/ec530951b8507139811c58a899cd74b0ecf9bab657445ebdcf959242d86437d6.jpg)

![](images/784090c9aa7aab06845cedf12377d7347082224220cb06362c37ae41da59422c.jpg)

![](images/243a60f12137980c0a4f860ec1aa68ffaac618c7c13b2a077d44f354c56e4096.jpg)  
Figure 15: Evolution of patient clusters over time $( M e n < 6 5 )$ , conditional on the initial cluster assignment (6 months post-NIAD). The number of patients in each cluster is indicated alongside the corresponding bars. For clarity, only the most frequent transitions are shown (see Figure 16 for more details). Cluster 6 is excluded due to the low number of transitions observed from this cluster.

![](images/8404fef05b087eaa09783d2a161cebc48c0dcca4b616d3d7097d2e780b4cfab6.jpg)  
Figure 16: Cumulative Incidence Function (CIF) for the time to first cluster transition in $M e n < 6 5 .$ stratified by initial cluster assignment at 6 months post-NIAD. The estimates account for censoring due to loss to follow-up and study termination, computed using the Aalen nonparametric estimator [25].

![](images/a8bd5eeec2a8a7681e927711c4965fa63567db2aa3940889841658ae2fe0421f.jpg)  
Figure 17: Transition graph. Red arrows (resp. blue) means an increase (resp. decrease) of the comorbidity risk $( M e n < 6 5 )$ . Arrow thickness is proportional to the number of observed transitions. Transitions with less than 1% probability are not shown.

## F Male 65+

<table><tr><td>Cluster</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td></tr><tr><td># of patients</td><td>2732</td><td>1515</td><td>1813</td><td>838</td><td>1974</td><td>1430</td><td>1125</td><td>1214</td><td>1954</td><td>3028</td><td>769</td><td>762</td><td>1319</td></tr><tr><td>Avg. age at NIAD (y) ± SD</td><td>72.61 ± 6.01</td><td>72.22 ± 5.62</td><td>73.01 ± 5.74</td><td>75.48 ± 6.69</td><td>74.38 ± 6.09</td><td>74.57 ± 6.37</td><td>75.52 ± 6.54</td><td>73.3 ± 6.12</td><td>74.73 ± 6.57</td><td>72.0 ± 5.39</td><td>75.14 ± 6.76</td><td>73.46 ± 6.24</td><td>74.14 ± 6.3</td></tr><tr><td>Avg. follow-up post-NIAD (y) ± SD</td><td>4.4 ± 3.1</td><td>5.31 ± 2.86</td><td>5.08 ± 2.85</td><td>4.45 ± 2.8</td><td>4.9 ± 2.9</td><td>5.15 ± 3.28</td><td>4.37 ± 2.85</td><td>5.79 ± 3.08</td><td>4.04 ± 2.78</td><td>4.46 ± 3.1</td><td>4.89 ± 2.81</td><td>5.35 ± 2.73</td><td>4.96 ± 2.97</td></tr><tr><td>Avg. BMI at NIAD (kg/m2) ± SD</td><td>28.68 ± 4.35</td><td>30.01 ± 4.87</td><td>29.41 ± 3.84</td><td>29.9 ± 5.1</td><td>30.54 ± 5.51</td><td>29.78 ± 3.97</td><td>30.13 ± 5.43</td><td>34.35 ± 6.26</td><td>29.0 ± 3.88</td><td>29.71 ± 4.17</td><td>32.81 ± 5.74</td><td>33.89 ± 6.31</td><td>28.94 ± 5.19</td></tr><tr><td>Avg. HbA1c at NIAD (%) ± SD</td><td>6.96 ± 1.33</td><td>7.33 ± 1.41</td><td>7.16 ± 1.36</td><td>7.09 ± 1.39</td><td>6.94 ± 1.33</td><td>6.95 ± 1.24</td><td>7.12 ± 1.36</td><td>6.87 ± 1.3</td><td>7.11 ± 1.37</td><td>6.92 ± 1.28</td><td>7.01 ± 1.36</td><td>7.21 ± 1.38</td><td>7.1 ± 1.36</td></tr><tr><td>% Ever smoked</td><td>29.61</td><td>31.22</td><td>26.86</td><td>27.92</td><td>45.54</td><td>27.48</td><td>56.44</td><td>27.68</td><td>41.91</td><td>27.87</td><td>34.72</td><td>29.27</td><td>48.6</td></tr><tr><td>% Ever alcohol</td><td>89.09</td><td>90.83</td><td>94.65</td><td>91.53</td><td>93.31</td><td>88.67</td><td>90.58</td><td>90.86</td><td>90.69</td><td>90.62</td><td>91.94</td><td>93.31</td><td>91.51</td></tr><tr><td>Comorbidities at NIAD [N,(%)]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stroke</td><td>189 (6.92)</td><td>111 (7.33)</td><td>&lt; 1%</td><td>748 (89.26)</td><td>218 (11.04)</td><td>45 (3.15)</td><td>196 (17.42)</td><td>88 (7.25)</td><td>316 (16.17)</td><td>31 (1.02)</td><td>154 (20.03)</td><td>60 (7.87)</td><td>139 (10.54)</td></tr><tr><td>Myocardial infarction Congestive heart failure</td><td>201 (7.36)</td><td>170 (11.22)</td><td>&lt;1%</td><td>&lt; 1%</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>579 (51.47)</td><td>&lt; 1 %</td><td>1156 (59.16)</td><td>&lt; 1 % &lt; 1%</td><td>450 (58.52) 166 (21.59)</td><td>&lt; 1 % 52 (6.82)</td><td>209 (15.85)</td></tr><tr><td>Peripheral vascular disease</td><td>72 (2.64) &lt; 1 %</td><td>77 (5.08) &lt;1 %</td><td>35 (1.93) &lt; 1 %</td><td>45 (5.37)</td><td>144 (7.29)</td><td>175 (12.24)</td><td>233 (20.71)</td><td>59 (4.86)</td><td>258 (13.20)</td><td>&lt;1%</td><td>295 (38.36)</td><td>&lt; 1 %</td><td>133 (10.08) &lt; 1 %</td></tr><tr><td>Diabetic retinopathy</td><td>28 (1.02)</td><td>667 (44.03)</td><td>1179 (65.03)</td><td>&lt; 1% 207 (24.70)</td><td>&lt; 1 % 227 (11.50)</td><td>&lt;1 % 25 (1.75)</td><td>478 (42.49)</td><td>&lt; 1 % &lt; 1 %</td><td>731 (37.41) 393 (20.11)</td><td>55 (1.82)</td><td>168 (21.85)</td><td>465 (61.02)</td><td>222 (16.83)</td></tr><tr><td>Hypertension</td><td>118 (4.32)</td><td>30 (1.98)</td><td>1774 (97.85)</td><td>805 (96.06)</td><td>1911 (96.81)</td><td>1377 (96.29)</td><td>200 (17.78) 849 (75.47)</td><td>1159 (95.47)</td><td>1504 (76.97)</td><td>2893 (95.54)</td><td>542 (70.48)</td><td>730 (95.80)</td><td>27 (2.05)</td></tr><tr><td>Chronic kidney disease</td><td>53 (1.94)</td><td>52 (3.43)</td><td>&lt; 1%</td><td>&lt;1 %</td><td>68 (3.44)</td><td>&lt; 1%</td><td>66 (5.87)</td><td>345 (28.42)</td><td>&lt;1 %</td><td>&lt; 1 %</td><td>290 (37.71)</td><td>289 (37.93)</td><td>46 (3.49)</td></tr><tr><td>Osteoporosis</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>&lt; 1%</td><td>&lt; 1%</td><td>39 (1.98)</td><td>&lt; 1 %</td><td>28 (2.49)</td><td>&lt; 1%</td><td>&lt; 1%</td><td>&lt; 1%</td><td>13 (1.69)</td><td>14 (1.84)</td><td>37 (2.81)</td></tr><tr><td>COPD</td><td>&lt; 1 %</td><td>&lt;1%</td><td>&lt; 1%</td><td>&lt;1%</td><td>1063 (53.85)</td><td>&lt; 1 %</td><td>609 (54.13)</td><td>&lt; 1%</td><td>&lt;1%</td><td>&lt; 1%</td><td>&lt; 1%</td><td>29 (3.81)</td><td>669 (50.72)</td></tr><tr><td>Chronic liver disease</td><td>36 (1.32)</td><td>26 (1.72)</td><td>31 (1.71)</td><td>14 (1.67)</td><td>34 (1.72)</td><td>24 (1.68)</td><td>27 (2.40)</td><td>&lt;1%</td><td>28 (1.43)</td><td>54 (1.78)</td><td>16 (2.08)</td><td>37 (4.86)</td><td>21 (1.59)</td></tr><tr><td>Comorbidities during follow-up [N,(%)]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stroke</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Myocardial infarction</td><td>94 (3.44)</td><td>63 (4.16)</td><td>65 (3.59)</td><td>90 (10.74)</td><td>99 (5.02)</td><td>50 (3.50)</td><td>82 (7.29)</td><td>47 (3.87)</td><td>105 (5.37)</td><td>89 (2.94)</td><td>45 (5.85)</td><td>42 (5.51)</td><td>53 (4.02)</td></tr><tr><td></td><td>47 (1.72)</td><td>38 (2.51)</td><td>22 (1.21)</td><td>10 (1.19)</td><td>42 (2.13)</td><td>22 (1.54)</td><td>67 (5.96)</td><td>18 (1.48)</td><td>91 (4.66)</td><td>40 (1.32)</td><td>41 (5.33)</td><td>26 (3.41)</td><td>23 (1.74)</td></tr><tr><td>Congestive heart failure</td><td>68 (2.49)</td><td>68 (4.49)</td><td>59 (3.25)</td><td>30 (3.58)</td><td>118 (5.98)</td><td>104 (7.27)</td><td>113 (10.04)</td><td>72 (5.93)</td><td>120 (6.14)</td><td>51 (1.68)</td><td>66 (8.58)</td><td>63 (8.27)</td><td>62 (4.70)</td></tr><tr><td>Peripheral vascular disease</td><td>&lt; 1 %</td><td>22 (1.45)</td><td>19 (1.05)</td><td>19 (2.27)</td><td>51 (2.58)</td><td>15 (1.05)</td><td>115 (10.22)</td><td>13 (1.07)</td><td>150 (7.68)</td><td>&lt;1 %</td><td>52 (6.76)</td><td>12 (1.57)</td><td>25 (1.90)</td></tr><tr><td>Diabetic retinopathy</td><td>214 (7.83)</td><td>510 (33.66)</td><td>615 (33.92)</td><td>152 (18.14)</td><td>341 (17.27)</td><td>159 (11.12)</td><td>216 (19.20)</td><td>132 (10.87)</td><td>311 (15.92)</td><td>296 (9.78)</td><td>148 (19.25)</td><td>221 (29.00)</td><td>303 (22.97)</td></tr><tr><td>Hypertension Chronic kidney disease</td><td>245 (8.97)</td><td>140 (9.24) 62 (4.09)</td><td>32 (1.77) 47 (2.59)</td><td>15 (1.79)</td><td>41 (2.08) 89 (4.51)</td><td>21 (1.47) 46 (3.22)</td><td>47 (4.18) 49 (4.36)</td><td>33 (2.72) 166 (13.67)</td><td>54 (2.76) 49 (2.51)</td><td>88 (2.91) 71 (2.34)</td><td>19 (2.47) 120 (15.60)</td><td>21 (2.76) 110 (14.44)</td><td>98 (7.43) 70 (5.31)</td></tr><tr><td>Osteoporosis</td><td>84 (3.07)</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>26 (3.10) 10 (1.19)</td><td>&lt; 1%</td><td>&lt; 1 %</td><td>16 (1.42)</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>14 (1.84)</td><td>&lt; 1 %</td></tr><tr><td>COPD</td><td>&lt; 1 %</td><td>33 (2.18)</td><td>35 (1.93)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Chronic liver disease</td><td>42 (1.54)</td><td></td><td>20 (1.10)</td><td>13 (1.55)</td><td>193 (9.78)</td><td>31 (2.17)</td><td>142 (12.62)</td><td>27 (2.22)</td><td>38 (1.94)</td><td>41 (1.35)</td><td>22 (2.86)</td><td>26 (3.41)</td><td>120 (9.10)</td></tr><tr><td></td><td>28 (1.02) 344 (12.59)</td><td>30 (1.98) 205 (13.53)</td><td>200 (11.03)</td><td>13 (1.55) 195 (23.27)</td><td>26 (1.32) 472 (23.91)</td><td>&lt; 1 % 254 (17.76)</td><td>17 (1.51) 384 (34.13)</td><td>22 (1.81) 222 (18.29)</td><td>&lt;1 % 424 (21.70)</td><td>31 (1.02) 315 (10.40)</td><td>21 (2.73) 219 (28.48)</td><td>18 (2.36) 163 (21.39)</td><td>17 (1.29) 309 (23.43)</td></tr><tr><td>Deaths before 2020</td></table>

Table 9: Patient characteristics in each cluster for the Men 65+ cohort with average values and standard deviation where applicable. Comorbidities are presented as absolute counts at NIAD initiation, along with the number of new comorbidities developed post-NIAD. Proportions, expressed as percentages of the cohort population, are reported in parentheses.

![](images/128b7d562aa45d68df16f173762b33d0bc977a1250ed0894489507f504ad874a.jpg)  
Figure 18: Average cluster-specific probabilities and lab values for Men 65+. Values have been centered by subtracting the average at the time of NIAD initiation to improve readability. Red bars indicate increases, and blue bars indicate decreases relative to the average patient at NIAD. Comorbidity risks are shown as changes in probability (left axis), and lab values as percentage changes relative to the reference value (right axis). Annotated numbers indicate absolute changes in the original units (see Table 1). Error bars represent the standard error of the mean.

Cluster Evolution starting from 4

Cluster Evolution starting from  
Cluster Evolution starting from 0  
![](images/e289acf3b50a44d2bbe65415dceb1112f563f6a9c6618f190152dc3d12434f99.jpg)  
Cluster Evolution starting from 5

![](images/ed360659878c35badb80c51ccaacf57ce2093b09501796b0facc109ae80fe347.jpg)  
Cluster Evolution starting from

![](images/4248b6c40d10bb8b000fe1b2fadb803e16244938510ac26747b6b7b8f47ec663.jpg)  
Cluster Evolution starting from 9

![](images/318f21cdf8268554fa588e93a47a118a5c9916086213b3223d803e13646a97af.jpg)

![](images/6f4fa1a7182aa229d77eee6abd99b231a1a3fe45ab41f507ba5b4f9a9aff0ab4.jpg)

![](images/62a27ba327d010c730acb2d98f8463a4c7acd386b94b4674b08b80c35c3525bf.jpg)

![](images/8c126db3c1fe326fae3b39ffc96ac2a3fec056a5201cc9d69484a989c670ad39.jpg)  
Figure 19: Evolution of patient clusters over time (Men 65+), conditional on the initial cluster assignment (6 months post-NIAD). The number of patients in each cluster is indicated alongside the corresponding bars. For clarity, only the most frequent transitions are shown (see Figure 20 for more details). Clusters 2,3,6,8,10,11 are excluded due to the low number of transitions observed from these clusters.

![](images/93ce9e77ae246f486046f29fd3f0c8a6011f12a9dfa2d2d96563533890fa02e6.jpg)  
Figure 20: Cumulative Incidence Function (CIF) for the time to first cluster transition in Men 65+, stratified by initial cluster assignment at 6 months post-NIAD. The estimates account for censoring due to loss to follow-up and study termination, computed using the Aalen nonparametric estimator [25].

![](images/5c01b7b2bac9084fba02c4b77b1c20a9e683ac030546c447a955d985ff0698e1.jpg)  
Figure 21: Transition graph. Red arrows (resp. blue) means an increase (resp. decrease) of the comorbidity risk (Men 65+). Arrow thickness is proportional to the number of observed transitions. Transitions with less than 1% probability are not shown.

## G Female 65+

<table><tr><td>Cluster</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td></tr><tr><td># of patients</td><td>2604</td><td>857</td><td>1629</td><td>742</td><td>1626</td><td>1276</td><td>787</td><td>1258</td><td>681</td><td>3702</td><td></td><td></td><td>452</td></tr><tr><td>Avg. age at NIAD (y) ± SD</td><td>72.7 ± 5.8</td><td>72.98 ± 5.79</td><td>74.91 ± 6.32</td><td>74.08 ± 6.18</td><td>76.72 ± 6.69</td><td>79.47 ± 7.44</td><td>76.98 ± 7.06</td><td>73.32 ± 5.86</td><td>78.33 ± 7.13</td><td>73.24 ± 5.72</td><td>716 73.69 ± 6.0</td><td>644 77.85 ± 7.15</td><td>74.04 ± 6.35</td></tr><tr><td>Avg. follow-up post-NIAD (y) ± SD</td><td>4.7 ± 3.12</td><td>5.11 ± 2.68</td><td>5.07 ± 2.71</td><td>4.35 ± 2.77</td><td>5.52 ± 3.12</td><td>3.5 ± 2.66</td><td>3.79 ± 2.64</td><td>6.0 ± 3.23</td><td>4.04 ± 2.31</td><td>4.82 ± 3.24</td><td>5.7 ± 2.68</td><td>4.24 ± 2.78</td><td>4.91 ± 2.73</td></tr><tr><td>Avg. BMI at NIAD (kg/m2) ± SD</td><td>30.24 ± 6.15</td><td>29.99 ± 5.92</td><td>29.84 ± 4.83</td><td>32.63 ± 6.74</td><td>28.96 ± 5.31</td><td>27.99 ± 4.72</td><td>29.53 ± 6.46</td><td>36.76 ± 7.37</td><td>30.7 ± 6.34</td><td>30.74 ± 4.8</td><td>36.67 ± 7.09</td><td>34.36 ± 7.83</td><td>31.65 ± 6.33</td></tr><tr><td>Avg. HbA1c at NIAD (%) ± SD</td><td>6.96 ± 1.27</td><td>7.24 ± 1.32</td><td>7.09 ± 1.29</td><td>7.08 ± 1.39</td><td>6.83 ± 1.3</td><td>7.08 ± 1.35</td><td>7.15 ± 1.42</td><td>6.98 ± 1.26</td><td>7.37 ± 1.49</td><td>6.9 ± 1.25</td><td>7.05 ± 1.18</td><td>7.09 ± 1.41</td><td>7.32 ± 1.37</td></tr><tr><td>% Ever smoked</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>37.0</td><td>21.47</td><td>25.56</td><td></td><td>23.67</td></tr><tr><td>% Ever alcohol</td><td>30.72</td><td>32.79 86.23</td><td>24.86 86.68</td><td>64.56</td><td>18.02</td><td>31.03 77.59</td><td>43.96 74.84</td><td>24.24 80.21</td><td>85.32</td><td>81.5</td><td>84.36</td><td>33.7</td><td></td></tr><tr><td></td><td>79.26</td><td></td><td></td><td>83.42</td><td>83.95</td><td></td><td></td><td></td><td></td><td></td><td></td><td>77.02</td><td>84.07</td></tr><tr><td>Comorbidities at NIAD [N,(%)]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stroke</td><td>137 (5.26)</td><td>63 (7.35)</td><td>141 (8.66)</td><td>80 (10.78)</td><td>158 (9.72)</td><td>263 (20.61)</td><td>109 (13.85)</td><td>105 (8.35)</td><td>138 (20.26)</td><td>197 (5.32)</td><td>73 (10.20)</td><td>123 (19.10)</td><td>30 (6.64)</td></tr><tr><td>Myocardial infarction</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>&lt; 1%</td><td>&lt;1 %</td><td>322 (25.24)</td><td>259 (32.91)</td><td>&lt; 1 %</td><td>189 (27.75)</td><td>&lt; 1 %</td><td>&lt;1 %</td><td>162 (25.16)</td><td>&lt;1 %</td></tr><tr><td>Congestive heart failure</td><td>&lt; 1%</td><td>&lt; 1 %</td><td>&lt;1%</td><td>&lt; 1%</td><td>&lt; 1%</td><td>299 (23.43)</td><td>179 (22.74)</td><td>&lt; 1%</td><td>214 (31.42)</td><td>&lt;1 %</td><td>&lt; 1%</td><td>230 (35.71)</td><td>&lt; 1%</td></tr><tr><td>Peripheral vascular disease</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt;1%</td><td>&lt;1%</td><td>203 (15.91)</td><td>120 (15.25)</td><td>&lt; 1%</td><td>161 (23.64)</td><td>&lt;1 %</td><td>&lt;1%</td><td>132 (20.50)</td><td>&lt;1%</td></tr><tr><td>Diabetic retinopathy</td><td>61 (2.34)</td><td>485 (56.59) 9 (1.05)</td><td>1014 (62.25)</td><td>33 (4.45)</td><td>24 (1.48)</td><td>41 (3.21)</td><td>96 (12.20)</td><td>&lt; 1 %</td><td>436 (64.02)</td><td>73 (1.97)</td><td>465 (64.94)</td><td>23 (3.57)</td><td>252 (55.75)</td></tr><tr><td>Hypertension</td><td>110 (4.22)</td><td>44 (5.13)</td><td>1599 (98.16) &lt;1%</td><td>704 (94.88)</td><td>1582 (97.29)</td><td>1226 (96.08)</td><td>31 (3.94)</td><td>1233 (98.01) 326 (25.91)</td><td>663 (97.36)</td><td>3587 (96.89)</td><td>708 (98.88)</td><td>626 (97.20)</td><td>439 (97.12)</td></tr><tr><td>Chronic kidney disease</td><td>113 (4.34)</td><td>60 (7.00)</td><td>65 (3.99)</td><td>42 (5.66)</td><td>26 (1.60)</td><td>&lt; 1 %</td><td>55 (6.99)</td><td></td><td>56 (8.22) 88 (12.92)</td><td>&lt; 1 %</td><td>173 (24.16) 53 (7.40)</td><td>204 (31.68)</td><td>18 (3.98) 22 (4.87)</td></tr><tr><td>Osteoporosis COPD</td><td>177 (6.80)</td><td>95 (11.09)</td><td>91 (5.59)</td><td>78 (10.51)</td><td>483 (29.70)</td><td>80 (6.27)</td><td>101 (12.83)</td><td>&lt; 1 % &lt;1%</td><td>124 (18.21)</td><td>&lt; 1% &lt;1%</td><td>53 (7.40)</td><td>76 (11.80) 118 (18.32)</td><td></td></tr><tr><td>Chronic liver disease</td><td>246 (9.45)</td><td></td><td></td><td>647 (87.20)</td><td>&lt; 1 %</td><td>132 (10.34)</td><td>195 (24.78)</td><td></td><td></td><td></td><td></td><td>13 (2.02)</td><td>31 (6.86)</td></tr><tr><td></td><td>57 (2.19)</td><td>32 (3.73)</td><td>&lt;1%</td><td>18 (2.43)</td><td>22 (1.35)</td><td>16 (1.25)</td><td>15 (1.91)</td><td>19 (1.51)</td><td>12 (1.76)</td><td>46 (1.24)</td><td>&lt;1 %</td><td></td><td>157 (34.73)</td></tr><tr><td>Comorbidities during follow-up [N,(%)]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stroke</td><td>98 (3.76)</td><td>22 (2.57)</td><td>75 (4.60)</td><td>27 (3.64)</td><td>77 (4.74)</td><td>68 (5.33)</td><td>50 (6.35)</td><td>57 (4.53)</td><td>37 (5.43)</td><td>134 (3.62)</td><td>44 (6.15)</td><td>45 (6.99)</td><td>25 (5.53)</td></tr><tr><td>Myocardial infarction</td><td>&lt; 1 %</td><td>&lt; 1%</td><td>19 (1.17)</td><td>9 (1.21)</td><td>&lt; 1 %</td><td>43 (3.37)</td><td>14 (1.78)</td><td>26 (2.07)</td><td>36 (5.29)</td><td>39 (1.05)</td><td>8 (1.12)</td><td>30 (4.66)</td><td>5 (1.11)</td></tr><tr><td>Congestive heart failure</td><td>45 (1.73)</td><td>22 (2.57)</td><td>33 (2.03)</td><td>33 (4.45)</td><td>40 (2.46)</td><td>86 (6.74)</td><td>58 (7.37)</td><td>66 (5.25)</td><td>58 (8.52)</td><td>73 (1.97)</td><td>38 (5.31)</td><td>76 (11.80)</td><td>13 (2.88)</td></tr><tr><td>Peripheral vascular disease</td><td>&lt; 1 %</td><td>&lt;1%</td><td>&lt; 1 %</td><td>13 (1.75)</td><td>21 (1.29)</td><td>42 (3.29)</td><td>29 (3.68)</td><td>16 (1.27)</td><td>34 (4.99)</td><td>&lt; 1 %</td><td>&lt;1 %</td><td>28 (4.35)</td><td>6 (1.33)</td></tr><tr><td>Diabetic retinopathy</td><td>296 (11.37)</td><td>372 (43.41)</td><td>615 (37.75)</td><td>109 (14.69)</td><td>215 (13.22)</td><td>82 (6.43)</td><td>107 (13.60)</td><td>169 (13.43)</td><td>245 (35.98)</td><td>415 (11.21)</td><td>251 (35.06)</td><td>57 (8.85)</td><td>153 (33.85)</td></tr><tr><td>Hypertension</td><td>324 (12.44)</td><td>86 (10.04)</td><td>27 (1.66)</td><td>21 (2.83)</td><td>32 (1.97)</td><td>13 (1.02)</td><td>56 (7.12)</td><td>22 (1.75)</td><td>9 (1.32)</td><td>76 (2.05)</td><td>&lt; 1 %</td><td>&lt; 1 %</td><td>12 (2.65)</td></tr><tr><td>Chronic kidney disease</td><td>64 (2.46)</td><td>34 (3.97)</td><td>30 (1.84)</td><td>26 (3.50)</td><td>41 (2.52)</td><td>17 (1.33)</td><td>33 (4.19)</td><td>116 (9.22)</td><td>25 (3.67)</td><td>61 (1.65)</td><td>65 (9.08)</td><td>89 (13.82)</td><td>9 (1.99)</td></tr><tr><td>Osteoporosis</td><td>78 (3.00)</td><td>26 (3.03)</td><td>56 (3.44)</td><td>22 (2.96)</td><td>147 (9.04)</td><td>24 (1.88)</td><td>22 (2.80)</td><td>30 (2.38)</td><td>24 (3.52)</td><td>43 (1.16)</td><td>23 (3.21)</td><td>21 (3.26)</td><td>12 (2.65)</td></tr><tr><td>COPD</td><td>78 (3.00)</td><td>22 (2.57)</td><td>38 (2.33)</td><td>93 (12.53)</td><td>34 (2.09)</td><td>37 (2.90)</td><td>36 (4.57)</td><td>33 (2.62)</td><td>31 (4.55)</td><td>54 (1.46)</td><td>20 (2.79)</td><td>40 (6.21)</td><td>11 (2.43)</td></tr><tr><td>Chronic liver disease</td><td>40 (1.54) 191 (7.33)</td><td>9 (1.05) 86 (10.04)</td><td>&lt; 1% 142 (8.72)</td><td>10 (1.35) 165 (22.24)</td><td>23 (1.41) 203 (12.48)</td><td>&lt; 1 % 603 (47.26)</td><td>&lt; 1 % 352 (44.73)</td><td>14 (1.11) 134 (10.65)</td><td>10 (1.47) 268 (39.35)</td><td>37 (1.00) 211 (5.70)</td><td>&lt; 1 % 87 (12.15)</td><td>&lt; 1 % 279 (43.32)</td><td>43 (9.51) 34 (7.52)</td></tr><tr><td>Deaths before 2020</td></table>

Table 10: Patient characteristics in each cluster for the Women 65+ cohort with average values and standard deviation where applicable. Comorbidities are presented as absolute counts at NIAD initiation, along with the number of new comorbidities developed post-NIAD. Proportions, expressed as percentages of the cohort population, are reported in parentheses.

![](images/ef29afa049f3d5107e7be52665e71d3991ba3cfc2b97ba4dbffeb7c3256e6b8e.jpg)

![](images/9b05d0c225bf2de2a36c26f73ca004d08d1d9e8502c8269e3b195bfe137acc88.jpg)

![](images/a5d40d04fa2ca87e4609e3c4f6960093ac501bc3eaef3474341ec8f13b5f465b.jpg)

![](images/772456354b791e32e8021b76b2689966f1ba39826d96fc9adca575d65ed027c3.jpg)

![](images/1ae48331c533b235ade1dd22126154e3a75f46047035ce05a5bfee3d84c81280.jpg)

![](images/49bae4ca17233e21eee28786eea6af2aa8a76756929527c04bec6cbf205df4c0.jpg)

![](images/712521094b97ad0a5685a83e41e764354f76a657bb755250463b58208e2959d3.jpg)

![](images/8473a374a406cbf1a5feef855f139c277fc20ce198799910b5ab744bdce96117.jpg)

![](images/d7018dd521b60706275acde58a1baf041aba99cd9fba58ee7ef526cc488a28f0.jpg)

![](images/5aaa8669c6467134aa919e28936e2793beeb44f0a071301551181e6e122a093b.jpg)

![](images/89edb7c32f802768bafbbc6e9a1f86e627b012e22731363c0fc8428996614cd7.jpg)

![](images/de6fe70248513da4d2b61657eba6aee43a7b7695af17cb0a63c98255bcc209ab.jpg)

![](images/758eb00107c61ce075a385e3e7b635a1bd480e0d2611a5b70bd8fe107d9d045d.jpg)

![](images/343f48384fd5d745117ff7120ccba1e0cf1cae5777b6ef509fc2661d39cf4ac2.jpg)  
Figure 22: Average cluster-specific probabilities and lab values for Women 65+. Values have been centered by subtracting the average at the time of NIAD initiation to improve readability. Red bars indicate increases, and blue bars indicate decreases relative to the average patient at NIAD. Comorbidity risks are shown as changes in probability (left axis), and lab values as percentage changes relative to the reference value (right axis). Annotated numbers indicate absolute changes in the original units (see Table 1). Error bars represent the standard error of the mean.

Cluster Evolution starting from 9

Cluster Evolution starting from 7

Cluster Evolution starting from 10

Cluster Evolution starting from 5

Cluster Evolution starting from 0

![](images/e4f2babe67187dbf7cfda42aee6a51b7b5b2ec5fa6d093a3374b9d95b73b2e8a.jpg)  
Cluster Evolution starting from 4  
Cluster Evolution starting from 2

![](images/c868ebaa5b15fba904b98a95274335d91aa6e05847163098aa34c5412759b40e.jpg)  
Cluster Evolution starting from 3

![](images/e00fb1d9460486ed2da6b07b263937956c0519ce55d92cb9d2e8d193244d0fdf.jpg)

![](images/9a193c7b48a02629de1b1d9c33a6e62bcd5cd2f9b6242d726e1de44d865d86fe.jpg)  
Cluster Evolution starting from 6

![](images/545fe06c972f8aecb699811070d4f1fdb6b32824c5df0c6749dcbd67984932c4.jpg)

![](images/a893123b60be9b846c2eca1515ee477f8386699f2ab68fd17b6fec7472f60556.jpg)

![](images/4bfa3b7a52816f11266d9b709cf670f6ef8f3aa3d1c9bfb55ab490ace3f92b7e.jpg)

![](images/c168142b8b666ce2a46137cdf7bfe30b97d7b8be0439910b3c42c56c222a1b3c.jpg)

![](images/51974d3fc2bf3d883aca65af94794c072a8f99d476928f6012fd2bf9d4565ea2.jpg)

![](images/163a1587ae5187163d6428ec522b6f05319d6327b9247275ccca9c42599524e6.jpg)

![](images/eca3be51e0281596da323753f7cc7371e3f0f3ad86689346e79e2276bd0e0eab.jpg)  
Figure 23: Evolution of patient clusters over time (Women 65+), conditional on the initial cluster assignment (6 months post-NIAD). The number of patients in each cluster is indicated alongside the corresponding bars. For clarity, only the most frequent transitions are shown (see Figure 24 for more details). Clusters 1,8 are excluded due to the low number of transitions observed from these clusters.

![](images/16504492419403ecb5b549daff110d52fc5f7c7afbf159de95fb50cf27ce2b1c.jpg)  
Figure 24: Cumulative Incidence Function (CIF) for the time to first cluster transition in Women 65+, stratified by initial cluster assignment at 6 months post-NIAD. The estimates account for censoring due to loss to follow-up and study termination, computed using the Aalen nonparametric estimator [25].

![](images/019373d5f47cb00a61d372dd6730910a01aff2de594d2a0943c77a724f714925.jpg)  
Figure 25: Transition graph. Red arrows (resp. blue) means an increase (resp. decrease) of the comorbidity risk (Women 65+). Arrow thickness is proportional to the number of observed transitions. Transitions with less than 1% probability are not shown.