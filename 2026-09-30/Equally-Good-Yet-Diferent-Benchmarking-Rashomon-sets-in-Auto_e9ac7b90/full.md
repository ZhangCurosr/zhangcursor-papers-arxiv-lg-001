# Equally Good, Yet Diferent: Benchmarking Rashomon sets in AutoML packages

Katarzyna Woźnica<sup>1,2</sup> Katarzyna Rogalska<sup>1</sup> Zuzanna Sieńko<sup>1</sup> Mustafa Cavus<sup>3</sup>

<sup>1</sup>Warsaw University of Technology

<sup>2</sup>Systems Research Institute, Polish Academy of Sciences

<sup>3</sup>Eskisehir Technical University

Abstract The Rashomon efect describes the existence of multiple near-optimal models that achieve comparable performance while ofering fundamentally diferent explanations. This creates a critical vulnerability in AutoML: x-hacking, the selective post-hoc choice of a model based on its explanation rather than predictive merit. No existing AutoML framework exposes this risk. We introduce ARSA ML, an open-source Python framework that quantifies Rashomon set structure and predictive multiplicity within AutoML pipelines. Using ARSA ML, we benchmark AutoGluon and H2O across 28 binary classification datasets, and conduct a post-hoc x-hacking analysis revealing a consistent structural asymmetry: AutoGluon produces larger, diverse sets with stable explanations, while H2O generates compact sets with markedly higher prediction divergence and explanation instability — making H2O users considerably more exposed to x-hacking. This gap persists across all evaluated metrics and epsilon thresholds, pointing to a fundamental diference in each framework’s model-building strategy. ARSA ML is available at https://pypi.org/project/arsa-ml/.

## 1 Introduction

The Rashomon efect (Breiman, 2001) describes multiple predictive models that achieve comparable performance while explaining the same phenomenon in fundamentally diferent ways. This occurs because data provide only an imperfect representation of reality (Ganesh et al., 2025)—an efect amplified by variations in training sets (Renard et al., 2024), hyperparameters (Cavus et al., 2025b), and preprocessing (Cavus and Biecek, 2025). Consequently, this leads to model multiplicity at the dataset level and predictive multiplicity at the individual level (D’Amour et al., 2022; Marx et al., 2020), where near-optimal models may issue conflicting predictions in high-stakes domains such as credit scoring or medical diagnosis. Characterizing the Rashomon set (Fisher et al., 2019)—the collection of models within a specified performance tolerance—is therefore essential to understanding these implications.

Because near-optimal models may assign diferent importance to diferent features (Rudin et al., 2024), the Rashomon set induces a corresponding multiplicity of explanations. For instance, a model attributing loan rejection primarily to income can coexist with one attributing it to credit history, both achieving identical predictive performance. This explanation multiplicity enables x-hacking (Sharma et al., 2024): the selective, post-hoc choice of a model from the Rashomon set based on its explanation properties rather than predictive merit. The susceptibility of a model selection process to x-hacking—which we call x-hackability—therefore depends on how large and how explanation-diverse its Rashomon set is.

The Rashomon perspective is largely absent from Automatic Machine Learning (AutoML) systems (Hutter et al., 2019; Erickson et al., 2020; Feurer et al., 2022; LeDell et al., 2020). By optimizing a single performance criterion and returning one best model, AutoML creates the illusion of a uniquely correct solution—particularly misleading for users with limited ML expertise who are least equipped to recognize the existence of equally valid alternatives. While AutoML does engage with model diversity in ensemble construction, this serves error decorrelation rather than acknowledging that multiple models explain the phenomenon diferently. Sharma et al. (2024) has demonstrated x-hacking’s feasibility in auto-sklearn (Feurer et al., 2022) pipelines, but a more fundamental question remains: do AutoML systems structurally vary in the x-hacking opportunities they provide, and do they ofer users any means to recognize this risk? To our knowledge, no existing AutoML framework exposes Rashomon set metrics, quantifies explanation disparity across near-optimal models, or alerts users to output x-hackability.

![](images/f56db2792ecc3682b78cf57eb3b03d43372c60436511a1a59ff58f9b2b81722b.jpg)  
Figure 1: General schema of using the ARSA ML package. It summarizes the Rashomon sets for two packages: AutoGluon and H2O, but there is a possibility to provide results from any AutoML package.

To address this gap, we introduce ARSA ML (AutoML Rashomon set Analysis), a framework for analyzing the Rashomon efect within AutoML pipelines (see Figure 1). ARSA ML provides unified implementations of Rashomon set metrics (Marx et al., 2020; Semenova et al., 2023, 2019; Watson-Daniels et al., 2023; Hsu and Calmon, 2022) spanning three levels: set-level metrics characterizing size and behavioral diversity, dataset-level metrics quantifying predictive multiplicity, and instance-level metrics identifying observations most exposed to conflicting predictions. ARSA ML also introduces the Rashomon Intersection—the overlap of Rashomon sets constructed under multiple evaluation criteria—together with a principled multi-criterion reference model selection methodology. The framework ofers a common interface for AutoGluon (Erickson et al., 2020) and H2O (LeDell et al., 2020), with a converter for other frameworks.

We apply the ARSA ML package to benchmark AutoGluon and H2O across Rashomon diversity, and investigate the following research questions: RQ1: How do their distinct model-building strategies afect Rashomon set size and diversity? RQ2: To what extent do the frameworks exhibit ambiguity, discrepancy, viable prediction range, and Rashomon capacity for identical observations? RQ3: Do their near-optimal models converge on the same influential features, or does explanation disparity make one framework more susceptible to x-hacking?

## 2 Related Work

AutoML and Diversity. AutoML frameworks utilize diversity primarily as a mechanism for error decorrelation in ensemble models. However, this performance-centric perspective overlooks the broader implications of model multiplicity. Recent studies have introduced the concept of xAutoML, aiming to make these pipelines interpretable. Zhai et al. (2024) proposed a domain-specific xAutoML framework using knowledge-informed feature extraction and model-agnostic selection methods. Similarly, Karthikeyan et al. (2025) combined the H2O AutoML system with LIME and SHAP methods, demonstrating the utility of hybrid systems by emphasizing that accuracy alone is insuficient for clinical trust. Despite these advances, most xAutoML tools still focus on explaining a single best model instead of accounting for the entire landscape of viable solutions.

The Rashomon Efect. The Rashomon efect Breiman (2001) describes the existence of multiple models that achieve similar predictive performance while ofering diferent explanations. This phenomenon is closely related to model multiplicity and model under-specification. Ewald et al.

(2026) extended this definition to CASHomon sets, considering multiple model classes and hyperparameters simultaneously. Black et al. (2022) further refined this notion by distinguishing between procedural multiplicity, where models difer in their internal decision-making mechanisms, and predictive multiplicity, where they produce diferent predictions for the same observations. As noted by Anders et al. (2020), procedural multiplicity is critical for detecting fairwashing scenarios, where model internals are manipulated without changing outputs.

Rashomon sets are characterized by global metrics such as the Rashomon Ratio (Semenova et al., 2019), Pattern Rashomon Ratio (Semenova et al., 2023), Ambiguity (Marx et al., 2020), and Discrepancy (Marx et al., 2020), which define the overall size and level ofdisagreement within the set. Local metrics, including Viable Prediction Range (VPR) and Rashomon Capacity, provide instancelevel insights into prediction uncertainty. To visualize these diferences, variable importance clouds (Dong and Rudin, 2020) have been proposed to illustrate the range of feature importance across all models in a Rashomon set.

Consequences in Rashomon Efect, Explainability, and AutoML. The omission of the Rashomon perspective in AutoML processes leads to various consequences, both negative and positive. On the negative side, selecting a single model leads to model selection bias and the risk of x-hacking (Sharma et al., 2024), where users may selectively choose a model that supports a desired narrative. Rawal et al. (2026) demonstrated that this multiplicity can be exploited for adversarial fairwashing, where explanatory methods are misled to hide discriminatory features.

On the positive side, exploring the Rashomon set enables uncertainty estimation (Cavus et al., 2025a, 2026) and ensures that selected models align with ethical constraints. Müller et al. (2023) provided an empirical evaluation showing that models within Rashomon sets exhibit high solution diversity, which can be measured through feature importance disagreement. Furthermore, Bifarin and Fernández (2024) argued that relying on a single best model in complex domains such as metabolomics can lead to misleading biological conclusions, and analyzing the entire solution landscape is essential. To formalize these trade-ofs, the MIMOSA framework (Guidotti et al., 2025) integrates fairness, privacy, and causality into the generation of interpretable models, providing a theoretical foundation for the trustworthy AI analysis that ARSA ML aims to automate.

## 3 ARSA ML Package

ARSA ML is an open source package publicly available at https://pypi.org/project/arsa-ml/. ARSA ML is built upon AutoGluon and H2O frameworks to generate diverse sets of trained models. Moreover, the package remains framework-agnostic, supporting the analysis of any externally trained models supplied in a compatible format (see Appendix 3 for a full description of the package architecture). Figure 1 illustrates the general workflow where, as a result, users get an interactive Streamlit application, described in Appendix B.

```python
Code Example 1: ARSA ML code example
from arsa_ml . pipelines . builder_abstract import *
from arsa_ml . pipelines . pipelines_user_input import *
# create pipeline from H2O saved models
builder = BuildRashomonH2O ( models_directory = example_models_path , test_data = test_h2o , target_column = target_column ,
df_name = ’heart ’, base_metric =’accuracy ’, feature_imp_needed = True )
# preview Rashomon set properties
builder . preview_rashomon ()
# set epsilon value
builder . set_epsilon (0.03)
# launch pipeline
rashomon_set , visualizer = builder . build ()
# close dashboard
builder . dashboard_close ()
```

## 3.1 Rashomon Analysis Metrics

In ARSA ML we adopt and unify existing Rashomon analysis metrics for systematic analysis of AutoML pipelines.

Let H denote the hypothesis space of candidate models and let $M : \mathcal H \to$ R denote an evaluation metric, where higher values indicate better performance. For a given metric �, we denote by $h _ { 0 } ^ { M } \in \mathcal { H }$ the reference model, taken throughout to be the best-performing model returned by the AutoML system under metric �. All Rashomon set constructions and derived metrics are defined relative to this reference. Where a single metric is used and unambiguous, we write $h _ { 0 } ^ { M }$ consistently; where two metrics $M _ { 1 }$ and $M _ { 2 }$ are considered simultaneously, the corresponding reference models $h _ { 0 } ^ { M _ { 1 } }$ and $h _ { 0 } ^ { M _ { 2 } }$ may difer. We denote by � the number of observations in the dataset, by $x _ { i }$ the �-th observation.

We consider the notion of the Rashomon set, defined as the collection of models whose performance is within a tolerance level $\epsilon > 0$ of a reference model $h _ { 0 } ^ { M }$ :

$$
R _ { \epsilon } ( h _ { 0 } ^ { M } , M ) = \{ h \in \mathcal { H } : M ( h ) \geq M ( h _ { 0 } ^ { M } ) - \epsilon \} .\tag{1}
$$

To characterize the Rashomon set along complementary dimensions, we employ a hierarchy of metrics spanning set-level size, population-level predictive multiplicity, and instance-level diversity.

Set-level metrics. The Rashomon Ratio (Semenova et al., 2023) measures the proportion of near-optimal models:

$$
\hat { R } _ { r a t i o } ( \mathcal { H } , \epsilon ) = \left| R _ { \epsilon } ( h _ { 0 } ^ { M } , M ) \right| / \left| \mathcal { H } \right| .\tag{2}
$$

A high Rashomon Ratio indicates that near-optimal performance is widespread across the hypothesis space, suggesting that architectural or algorithmic choices have limited impact on predictive quality alone.

However, two models may be architecturally distinct yet produce identical predictions on the observed data. To capture behaviorally meaningful diversity, rather than model count, we use the Pattern Rashomon Ratio (Semenova et al., 2019, 2023):

$$
\hat { R } _ { r a t i o } ^ { p a t } ( \mathcal { H } , \epsilon ) = | \pi ( \mathcal { H } , \epsilon ) | / | \psi ( \mathcal { H } ) | ,\tag{3}
$$

where $\pi ( \mathcal { H } , \epsilon )$ and $\psi ( \mathcal { H } )$ denote unique prediction vectors for Rashomon-set models and all models in $\mathcal { H } ,$ respectively. These metrics quantify the number and diversity of near-optimal models.

Population-level metrics. We quantify disagreement among near-optimal models using ambiguity and discrepancy (Marx et al., 2020), defined with respect to a fixed reference model $\bar { h } _ { 0 } ^ { M }$ . For class predictions, ambiguity is defined as

$$
\alpha _ { \epsilon } ( h _ { 0 } ^ { M } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \operatorname* { m a x } _ { h \in { \cal R } _ { \epsilon } ( h _ { 0 } ^ { M } , M ) } \mathbb { 1 } [ h ( x _ { i } ) \neq h _ { 0 } ^ { M } ( x _ { i } ) ] ,\tag{4}
$$

and measures the fraction of observations for which at least one near-optimal model disagrees with the reference model. Discrepancy is defined as

$$
D _ { \epsilon } ( h _ { 0 } ^ { M } ) = \operatorname* { m a x } _ { h \in { \cal R } _ { \epsilon } ( h _ { 0 } ^ { M } , M ) } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { 1 } [ h ( x _ { i } ) \neq h _ { 0 } ^ { M } ( x _ { i } ) ] ,\tag{5}
$$

and captures the maximum proportion of predictions that may change when switching models. For probabilistic predictions, these metrics are extended using a threshold $\delta > 0$ (Watson-Daniels et al., 2023) to count probability diferences exceeding �. All definitions naturally extend to multiclass classification via arg max (Hsu and Calmon, 2022).

Instance-level metrics. We further consider instance-level metrics, defined for each observation $x _ { i } .$ . The Viable Prediction Range (VPR) (Watson-Daniels et al., 2023) captures the range of predicted probabilities across Rashomon models:

$$
V _ { \epsilon } ( x _ { i } ) = \left[ \operatorname* { m i n } _ { h \in R _ { \epsilon } } h ( x _ { i } ) _ { 1 } , \operatorname* { m a x } _ { h \in R _ { \epsilon } } h ( x _ { i } ) _ { 1 } \right] ,\tag{6}
$$

quantifying uncertainty in risk estimates for a given instance.

To measure predictive multiplicity at the instance level, we use Rashomon Capacity (Hsu and Calmon, 2022). Let $\mathcal { M } _ { \epsilon } ( x _ { i } ) = \bar { \{ h ( x _ { i } ) : h \in R _ { \epsilon } ( h _ { 0 } ^ { M } , M ) \} }$ be the set of probability predictions generated by all models in the Rashomon set for observation $x _ { i }$ . The Rashomon Capacity is defined as

$$
m _ { C } ( x _ { i } ) = 2 ^ { C ( \mathcal { M } _ { \epsilon } ( x _ { i } ) ) } , \quad C ( \mathcal { M } _ { \epsilon } ( x _ { i } ) ) = \operatorname* { s u p } _ { P _ { Z } } \operatorname* { i n f } _ { q \in \Delta _ { c } } \mathbb { E } _ { h \sim P _ { Z } } \left[ D _ { K L } ( h ( x _ { i } ) \parallel q ) \right] ,\tag{7}
$$

where $P _ { Z }$ ranges over all probability distributions on $R _ { \epsilon } ( h _ { 0 } ^ { M } , M ) , \Delta _ { c }$ denotes the probability simplex over � classes, $q \in \Delta _ { c }$ is a reference distribution, and $D _ { K L }$ is the Kullback–Leibler divergence.

Rashomon Intersection. To account for multiple evaluation criteria, we introduce the Rashomon Intersection, defined as the intersection of Rashomon sets constructed for diferent metrics:

$$
R I _ { \epsilon } ( M _ { 1 } , M _ { 2 } ) = R _ { \epsilon } ( h _ { 0 } ^ { M _ { 1 } } , M _ { 1 } ) \cap R _ { \epsilon } ( h _ { 0 } ^ { M _ { 2 } } , M _ { 2 } ) .\tag{8}
$$

It captures models that are simultaneously near-optimal with respect to both $M _ { 1 }$ and $M _ { 2 }$ . Since the reference model may difer across metrics, any downstream analysis requires a single joint reference model. We define it as the solution to the weighted optimization problem over the intersection:

$$
h _ { 0 } = \arg \operatorname* { m a x } _ { h \in R I _ { \epsilon } ( M _ { 1 } , M _ { 2 } ) } \big ( w _ { 1 } M _ { 1 } ( h ) + w _ { 2 } M _ { 2 } ( h ) \big ) , \quad w _ { 1 } , w _ { 2 } \in [ 0 , 1 ] , \ w _ { 1 } + w _ { 2 } = 1 .\tag{9}
$$

We propose three methods for determining $w _ { 1 }$ and �<sub>2</sub>: (1) Custom weights The user specifies $w _ { 1 }$ and $w _ { 2 }$ directly, reflecting domain-specific priorities. (2) Entropy method (Wang et al., 2023). Weights are derived from the entropy of each metric’s distribution across models within Rashomon Intersection. (3)CRITIC method (Wang et al., 2023). This is an objective weighting technique used in multi-criteria decision-making. Details of these methods can be found in Appendix C.

## 4 Explanation Hackability

The metrics introduced in Section 3.1 characterize the Rashomon efect along predictive dimensions: how many near-optimal models exist, how much their predictions difer across the population, and how uncertain the predicted probability of a single observation is. However, they do not address a complementary question of practical importance: do near-optimal models agree on which features drive the prediction?

If the Rashomon set contains near-optimal models that assign substantially diferent importance to the input features, then the choice of which model to report determines which explanation is delivered — without any sacrifice in predictive performance. We formalise this risk as explanation hackability (XHack): the degree to which a framework’s Rashomon set permits the selection of models that support conflicting feature-importance narratives at no performance cost.

Feature importance metric. Let $\mathcal { F } ~ = ~ \{ f _ { 1 } , \ldots , f _ { p } \}$ denote the set of input features. Each model $h \in \mathcal H$ is equipped with a feature importance ranking $\varphi ( h ) : \mathcal { F }  \{ 1 , . . . , p \}$ , where $\varphi ( h ) ( f ) = 1$ denotes the most important feature according to ℎ. For a feature $f \in { \mathcal { F } }$ and a threshold $k \in \{ 1 , . . . , p \}$ , define the top-� importance metric:

$$
\phi ^ { f , k } ( h ) = \mathbb { 1 } \left[ \varphi ( h ) ( f ) \leq k \right] \in \{ 0 , 1 \} ,\tag{10}
$$

which equals 1 if model ℎ considers $f$ among its � most important features, and 0 otherwise.

Rashomon Intersection for feature importance. We instantiate the Rashomon Intersection from Section 3.1 with $M _ { 1 } = M$ (a predictive performance metric) and $M _ { 2 } = \phi ^ { f , k }$ (the top-� importance indicator for feature $f )$ and since $\phi ^ { f , k }$ is binary, its Rashomon set at any $\epsilon < 1$ reduces exactly to models that rank $f$ in their top-� :

$$
\begin{array} { r } { R I _ { \epsilon } ( M , \phi ^ { f , k } ) = R _ { \epsilon } ( h _ { 0 } ^ { M } , M ) \cap R _ { \epsilon } ( h _ { 0 } ^ { \phi ^ { f , k } } , \phi ^ { f , k } ) = \left\{ h \in R _ { \epsilon } ( h _ { 0 } ^ { M } , M ) \big | \varphi ( h ) ( f ) \le k \right\} . } \end{array}\tag{11}
$$

This is the set of near-optimal models that also consider $f$ an important feature. A large $| R I _ { \epsilon } ( M , \phi ^ { f , k } ) |$ relative to $| R _ { \epsilon } ( h _ { 0 } ^ { M } , M ) |$ means that performance-competitive models consistently agree on the importance of $f ;$ a small intersection means that $f { \boldsymbol { s } }$ importance is specific to only a minority of near-optimal models.

XHackability. The intersection $R I _ { \epsilon } ( M , \phi ^ { f , k } )$ characterizes the explanation stability of a specific feature $f .$ . To obtain a single, comparable scalar that captures the overall hackability ofa framework’s explanation landscape, we ask: across all features, what is the largest fraction of near-optimal models that agree on any single feature’s importance? XHackbility is the complement of this maximum:

Definition (Explanation Hackability). The explanation hackability for given dataset with feature set $F ,$ performance metric �, tolerance $\epsilon ,$ and top-� threshold is:

$$
\mathrm { X H a c k } ( k , M , \epsilon ) = 1 - \operatorname* { m a x } _ { f \in \mathcal { F } } \frac { \big | R I _ { \epsilon } ( M , \phi ^ { f , k } ) \big | } { \big | R _ { \epsilon } ( h _ { 0 } ^ { M } , M ) \big | } .\tag{12}
$$

where the term max $: f \in \mathcal { F } \left| R I _ { \epsilon } ( M , \phi ^ { f , k } ) \right| / | R _ { \epsilon } ( h _ { 0 } ^ { M } , M ) |$ identifies the feature whose importance is most consistently shared across near-optimal models; XHack measures how far even this best-case feature falls from universal agreement. XHack $\in [ 0 , 1 ]$ : at ${ \mathrm { X H a c k } } = 0 $ , every near-optimal model agrees on at least one feature’s top-� membership, so no narrative manipulation is possible within $R _ { \epsilon }$ without sacrificing performance; at XHack = 1, no feature is consistently ranked in the top-�, meaning any desired feature-importance narrative can be supported by some near-optimal model. Intermediate values scale accordingly — for instance, XHack = 0.6 implies that even the most consistently important feature appears in the top-� of only 40% of near-optimal models.

## 5 Benchmark of Rashomon set Diversity in AutoGluon and H2O

## 5.1 Methodology

Datasets. Experiments are conducted on 30 binary classification datasets retrieved from OpenML (Bischl et al., 2017); these datasets are widely used in benchmarks like TabArena (Erickson et al., 2025). Finally, only 28 are reported here since technical dificulties with training of AutoML frameworks for two of 30. Only binary target datasets are included. The default target column defined by OpenML is used as the response variable in each case. Before training, the target variable is integer-encoded. For evaluation, we use stratified 4-fold cross-validation.

Model Training. Since ARSA ML package, AutoML frameworks are used for model training: AutoGluon and H2O AutoML. AutoGluon trains models with the good\_quality preset and a time budget of 2h per fold. H2O AutoML is configured with an equivalent time limit and capped at 20 models. This limitation is introduced due to out-of-memory errors encountered during training on the server machine, which caused instability when a larger number of models was allowed. Both frameworks handle internal preprocessing automatically, including missing value imputation, categorical encoding, and feature normalization.

Feature Importance. We extract feature importance rankings via native APIs, applying targeted adjustments to ensure cross-framework consistency at the original-feature level. In AutoGluon, we compute permutation importance uniformly across all models using the framework’s training set evaluation. While we utilize H2O’s default permutation importance for most models, we compute it post-hoc for stacked ensembles and utilize weight-based importance for Deep Learning models. To reconcile $\mathrm { H } 2 \mathrm { O } ^ { \mathrm { \prime } }$ s internal feature discretization with our required granularity, we assign each original feature the minimum rank of its constituent bins and recompute the final rankings across the original feature set.

Rashomon set Construction. All computations are performed using ARSA ML and experimental code is available on GitHub<sup>1</sup>. Six evaluation metrics are considered: accuracy, balanced accuracy, F1, precision, recall, and ROC AUC. For each metric, seven epsilon thresholds are applied, defined as fixed percentages of the best observed score: 1%, 2.5%, 5%, 10%, 15%, 20%, and 25%. This yielded 42 Rashomon set configurations per fold (6 metrics × 7 epsilon levels). Results are stored in json files available in Zendo<sup>2</sup>.

## 5.2 Results

Table 1 reports the reference model performance across 28 benchmark datasets under six evaluation metrics. The results reveal a clear split between frameworks: AutoGluon achieves higher accuracy, precision, and ROC AUC, winning the majority of datasets under these metrics, while H2O dominates on balanced accuracy, F1, and recall, where it wins 21, 22, and 25 datasets, respectively.

Table 1: Reference model performance across 28 benchmark datasets.
<table><tr><td>Metric</td><td>AG</td><td>H2O</td><td> $\mathrm { W } _ { \mathrm { A G } }$ </td><td> $\mathrm { W _ { H 2 0 } }$ </td><td>p</td></tr><tr><td>Accuracy</td><td> $\mathbf { 0 . 8 8 2 7 \pm 0 . 0 8 5 7 }$ </td><td> $0 . 8 7 5 1 \pm 0 . 0 8 8 7$ </td><td>24</td><td>4</td><td>&lt;0.001</td></tr><tr><td>Balanced Accuracy</td><td> $0 . 7 1 8 9 \pm 0 . 1 3 8 2$ </td><td> $\mathbf { 0 . 7 6 4 1 \pm 0 . 1 0 2 1 }$ </td><td>7</td><td>21</td><td>&lt;0.001</td></tr><tr><td>F1</td><td> $0 . 5 9 4 3 \pm 0 . 2 9 2 8$ </td><td> $\mathbf { 0 . 6 3 7 8 \pm 0 . 2 3 3 8 }$ </td><td>5</td><td>22</td><td>&lt;0.001</td></tr><tr><td>Precision</td><td> $\mathbf { 0 . 7 9 2 6 \pm 0 . 1 4 4 1 }$ </td><td> $0 . 6 5 8 1 \pm 0 . 2 3 5 9$ </td><td>22</td><td>5</td><td>&lt;0.001</td></tr><tr><td>Recall</td><td> $0 . 5 5 9 3 \pm 0 . 3 1 8 5$ </td><td> $\mathbf { 0 . 7 4 0 7 \pm 0 . 1 8 6 4 }$ </td><td>2</td><td>25</td><td>&lt;0.001</td></tr><tr><td>ROC AUC</td><td> $\mathbf { 0 . 8 5 5 7 \pm 0 . 0 9 0 8 }$ </td><td> $0 . 8 5 2 3 \pm 0 . 0 9 0 2$ </td><td>18</td><td>10</td><td>0.009</td></tr></table>

Mean ± std computed across 28 datasets (scores averaged over folds per dataset). W = number of datasets won. $p { : }$ paired Wilcoxon signed-rank test (two-sided). Bold indicates the better mean.

RQ1: Rashomon set size and composition. Figure 2a shows how the Rashomon set evolves with increasing � under the ROC AUC metric. As � increases, all metrics grow and then stabilize, indicating that a larger tolerance expands the set but with diminishing gains in diversity. AutoGluon consistently exhibits a higher Rashomon Ratio, suggesting a larger set of near-optimal models. However, the Pattern Rashomon Ratio is similar across frameworks, implying that many of these models produce similar predictions. In contrast, H2O achieves comparable behavioral diversity with fewer models.

Figure 3 presents the distribution of model types within the Rashomon Sets at three � levels under the ROC AUC metric. The two frameworks exhibit markedly diferent compositional profiles. AutoGluon’s Rashomon Sets are dominated by Tree-Based Models and Gradient Boosting models, with this distribution remaining stable across all � levels, suggesting uniform representation of algorithm families across performance tiers. Notably, AutoGluon’s Gradient Boosting category encompasses multiple distinct algorithms – LightGBM, XGBoost, and CatBoost – and similarly, its Tree-Based Models comprise both Random Forest and Extra Trees variants, making AutoGluon’s efective algorithmic diversity even broader than the aggregated bars suggest. H2O, by contrast, is heavily concentrated on Gradient Boosting, represented primarily by GBM, while the share of Neural Networks grows substantially with � – from 12.4% at $\epsilon = 0 . 0 1$ to 29.9% at $\epsilon = 0 . 1 5 \textrm { -- }$ indicating that H2O’s Deep Learning models enter the set only as the tolerance widens.

![](images/5a4cc5fd49a4f9d58b9de2ae68e2ccbd82c2d8e462b7fd0f3a37f4e89b124b88.jpg)

![](images/cf83ef87ae0181bf9f8f103be6f78184afd47ee0e456afabc80ff0280cb9e4dc.jpg)

![](images/9a12cf7b8b4188b842ac9c7e2df05b86f5483232dba1f16d4f1515bf0d15654f.jpg)

![](images/583b0cfd94613e5c45965019c41cf89b32b5440922d35d816f0bfb14238ecbc3.jpg)  
(a) Set-level metrics for range of epsilons

![](images/e84c210c75168a829879f172f3398969b452cf8c20428e64381f6866f2b05cfa.jpg)  
(b) Population-level metrics for range of epsilons

![](images/28d61d6a9f290efbbbb8e67b0a25ec75f6344d4aec618d9cf247a9363ce4e8d4.jpg)  
(c) Instance-level metrics for range of epsilons

![](images/1db6ab7b04a28180a6a6b84bd8278fad76d6549b6fa72af27d0f68632456d735.jpg)  
Model Type Distribution Across All Rashomon Sets

![](images/facde385bb77b6669a4dd2c81b7540456b513952c12d177be4d7f1a6ad622c09.jpg)

![](images/b10d730990d72dea6e1a309ec4ea93efcc1e9ba08715d47781be38e555024d74.jpg)  
Figure 3: Diferences in algorithm types within Rashomon sets between frameworks.

RQ2: Predictive multiplicity at population and instance level. Figure 2b H2O shows higher ambiguity and discrepancy across all � levels, indicating stronger predictive multiplicity and less consistent predictions. This pattern is reinforced by higher VPR and slightly higher Rashomon Capacity shown in Figure 2c, suggesting greater instance-level uncertainty. Overall, AutoGluon produces larger but more homogeneous Rashomon sets, while H2O yields smaller yet more diverse and uncertain prediction behaviors

RQ3: Explanation hackability. Figure 4 shows the distribution of XHack scores across 28 datasets at three � levels under ROC AUC. AutoGluon explanations are substantially more consistent: for most datasets, the XHack is close to zero at small �, meaning that near-optimal AutoGluon models largely agree on which features matter most. H2O, by contrast, exhibits markedly higher and more variable XHack values across the same datasets, indicating that its Rashomon sets routinely contain models supporting conflicting feature-importance narratives at no performance cost. In both frameworks XHack increases with $\epsilon ,$ as expected: a wider tolerance admits more models and therefore more explanation diversity. However, the gap between frameworks persists across all � levels, suggesting a structural diference rather than an artefact of threshold choice. This finding is consistent with the set-composition results in Figure 3: H2O’s Rashomon sets are dominated by GBM variants whose feature rankings can diverge substantially, while AutoGluon’s broader mix of tree ensembles converges on more stable importance orderings. Together, these results imply that H2O users face a considerably higher risk of x-hacking.

![](images/d52a43da38283dbef88b21b2812095d30e94ffce11fe4c6127e507a2202f1253.jpg)  
Figure 4: Distribution of XHackability scores across 28 datasets at increasing � levels (ROC AUC metric). Each box shows the interquartile range across datasets; the line is the median. AutoGluon (left) shows medians close to zero that grow slowly with �; H2O (right) exhibits higher and more dispersed values throughout, indicating greater susceptibility to explanation manipulation.

## 6 Conclusions

AutoML systems have made high-performing predictive models widely accessible, yet they typically return a single "best" model while ignoring many equally valid alternatives. ARSA ML addresses this gap by making these alternatives visible and measurable. By converting a broad set of Rashomon metrics into an automated, framework-agnostic pipeline, it allows practitioners to quantify the predictive and explanatory flexibility within their model set before committing to a final deployment.

Our benchmark of AutoGluon and H2O reveals a consistent asymmetry in how this flexibility manifests: the two frameworks occupy opposite ends of a size–diversity–hackability trade-of. AutoGluon generates larger, algorithmically diverse sets that remain prediction-homogeneous with stable explanations. In contrast, H2O generates compact sets concentrated in a single model family, yet these sets exhibit greater prediction divergence and explanation instability. Because neither framework currently exposes these dynamics, ARSA ML is essential for making this underlying behavior actionable.

These technical trade-ofs connect directly to an emerging regulatory crisis. As Frohnapfel et al. (2026) argues, predictive multiplicity conflicts with the EU AI Act’s requirement that high-risk systems report accuracy not just at the dataset level, but for specific individuals. If two near-optimal models disagree on a single person’s outcome, the choice of one model over another becomes an arbitrary decision with significant ethical consequences. By operationalizing metrics like ambiguity, discrepancy, Viable Prediction Range, and Rashomon Capacity, ARSA ML integrates compliancerelevant data directly into the AutoML pipeline, making legal and statistical accountability accessible to every practitioner.

Acknowledgements. This research was carried out with the support of the High Performance Computing Center at Faculty of Mathematics and Information Science Warsaw University of Technology.

## References

Anders, C., Pasliev, P., Dombrowski, A.-K., Müller, K.-R., and Kessel, P. (2020). Fairwashing explanations with of-manifold detergent. In Proceedings of the International Conference on Machine Learning.

Bifarin, O. O. and Fernández, F. M. (2024). Automated Machine Learning and Explainable AI (AutoML-XAI) for Metabolomics: Improving Cancer Diagnostics. Journal of the American Society for Mass Spectrometry, 35(6).

Bischl, B., Casalicchio, G., Feurer, M., Hutter, F., Lang, M., Mantovani, R. G., van Rijn, J. N., and Vanschoren, J. (2017). OpenML benchmarking suites and the OpenML100. stat, 1050.

Black, E., Raghavan, M., and Barocas, S. (2022). Model Multiplicity: Opportunities, Concerns, and Solutions. In Proceedings ofthe ACM Conference on Fairness Accountability and Transparency.

Breiman, L. (2001). Statistical Modeling: The Two Cultures (with comments and a rejoinder by the author). Statistical Science, 16(3).

Cavus, M. and Biecek, P. (2025). Investigating the impact of balancing, filtering, and complexity on predictive multiplicity: A data-centric perspective. Information Fusion, 123.

Cavus, M., Rijn, J. N. v., and Biecek, P. (2026). Quantifying Model Uncertainty with AutoML and Rashomon Partial Dependence Profiles: Enabling Trustworthy and Human-centered XAI. Information Systems Frontiers.

Cavus, M., van Rijn, J. N., and Biecek, P. (2025a). Beyond the single-best model: Rashomon partial dependence profile for trustworthy explanations in automl. In Proceedings of the International Conference on Discovery Science.

Cavus, M., Woźnica, K., and Biecek, P. (2025b). The role of hyperparameters in predictive multiplicity. arXiv preprint arXiv:2503.13506.

D’Amour, A., Heller, K., Moldovan, D., Adlam, B., Alipanahi, B., Beutel, A., Chen, C., Deaton, J., Eisenstein, J., Hofman, M. D., et al. (2022). Underspecification presents challenges for credibility in modern machine learning. Journal of Machine Learning Research, 23(226).

Dong, J. and Rudin, C. (2020). Exploring the Cloud of Variable Importance for the Set of All Good Models. Nature Machine Intelligence, 2(12).

Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., and Smola, A. (2020). AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data. arXiv preprint arXiv:2003.06505.

Erickson, N., Purucker, L., Tschalzev, A., Holzmüller, D., Desai, P., Salinas, D., and Hutter, F. (2025). Tabarena: A living benchmark for machine learning on tabular data. Advances in Neural Information Processing Systems.

Ewald, F. K., Binder, M., Feurer, M., Bischl, B., and Casalicchio, G. (2026). CASHomon Sets: Eficient Rashomon Sets Across Multiple Model Classes and their Hyperparameters. arXiv preprint arXiv:2603.15321.

Feurer, M., Eggensperger, K., Falkner, S., Lindauer, M., and Hutter, F. (2022). Auto-sklearn 2.0: Hands-free automl via meta-learning. Journal of Machine Learning Research, 23(261).

Fisher, A., Rudin, C., and Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable’s importance by studying an entire class of prediction models simultaneously. Journal of Machine Learning Research, 20.

Frohnapfel, K., Seyfert, M., Bordt, S., von Luxburg, U., and Meding, K. (2026). Using predictive multiplicity to measure individual performance within the AI Act. In Proceedings of the ACM Conference on Fairness, Accountability, and Transparency.

Ganesh, P., Taik, A., and Farnadi, G. (2025). Systemizing multiplicity: The curious case of arbitrariness in machine learning. In Proceedings ofthe AAAI/ACM Conference on AI, Ethics, and Society.

Guidotti, R., Cinquini, M., Manerba, M. M., Setzu, M., and Spinnato, F. (2025). Towards the Formalization of a Trustworthy AI for Mining Interpretable Models exploiting Sophisticated Algorithms. arXiv preprint arXiv:2510.20621.

Hsu, H. and Calmon, F. P. (2022). Rashomon capacity: a metric for predictive multiplicity in classification. In Advances in Neural Information Processing Systems.

Hutter, F., Kotthof, L., and Vanschoren, J., editors (2019). Automated Machine Learning: Methods, Systems, Challenges. The Springer Series on Challenges in Machine Learning. Springer Cham.

Karthikeyan, P., Malaserene, I., and Deepakraj, E. (2025). Explainable AI-based cervical cancer prediction using FSAE feature engineering and H2O AutoML. Scientific Reports, 15(1).

LeDell, E., Poirier, S., et al. (2020). H2O Automl: Scalable automatic machine learning. In Proceedings of the AutoML Workshop at the International Conference on Machine Learning.

Marx, C. T., Du Pin Calmon, F., and Ustun, B. (2020). Predictive multiplicity in classification. In Proceedings of the International Conference on Machine Learning.

Müller, S., Toborek, V., Beckh, K., Jakobs, M., Bauckhage, C., and Welke, P. (2023). An empirical evaluation of the Rashomon efect in explainable machine learning. In Proceedings ofthe Joint European Conference on Machine Learning and Knowledge Discovery in Databases.

Rawal, K., Delaney, E., Fu, Z., Wachter, S., and Russell, C. (2026). Evaluating the Ability of Explanations to Disambiguate Models in a Rashomon Set. Workshop on Navigating Model Uncertainty and the Rashomon Efect: From Theory and Tools to Applications and Impact.

Renard, X., Laugel, T., and Detyniecki, M. (2024). Understanding prediction discrepancies in classification. Machine Learning, 113(10).

Rudin, C., Zhong, C., Semenova, L., Seltzer, M., Parr, R., Liu, J., Katta, S., Donnelly, J., Chen, H., and Boner, Z. (2024). Amazing things come from having many good models. arXiv preprint arXiv:2407.04846.

Semenova, L., Chen, H., Parr, R., and Rudin, C. (2023). A path to simpler models starts with noise. In Advances in Neural Information Processing Systems, volume 36.

Semenova, L., Rudin, C., and Parr, R. (2019). On the Existence of Simpler Machine Learning Models (arXiv v3; later published at ACM Conference on Fairness, Accountability, and Transparency 2022). arXiv preprint arXiv:1908.01755v3.

Sharma, R., Redyuk, S., Mukherjee, S., Šipka, A., Hüllermeier, E., Vollmer, S., and Selby, D. (2024). X hacking: The threat of misguided AutoML. In Proceedings of the International Conference on Machine Learning.

Wang, Z., Nabavi, S. R., and Rangaiah, G. P. (2023). Selected Multi-criteria Decision-Making Methods and Their Applications to Product and System Design. Springer Nature Singapore.

Watson-Daniels, J., Parkes, D. C., and Ustun, B. (2023). Predictive multiplicity in probabilistic classification. In Proceedings of Association for the Advancement of Artificial Intelligence.

Zhai, W., Han, Q., Chen, L., and Shi, X. (2024). Explainable AutoML (xAutoML) with Adaptive Modeling for Yield Enhancement in Semiconductor Smart Manufacturing. In Proceedings of the International Conference on Artificial Intelligence and Automation Control.

![](images/8ac60db43cd3cec2887252263ee665d7c33143eedc58e66fb5257341a6889fa3.jpg)  
Figure 5: Technical schema of ARSA ML modules

Converters. The Converters module normalizes the heterogeneous outputs of diferent AutoML frameworks into a unified internal representation: a model leaderboard, class-prediction vectors, probability-prediction matrices, and feature-importance rankings. Three concrete implementations are provided: PredictorConverter for AutoGluon, H2OConverter for H2O, and a base Converter interface for custom integrations. Each converter exposes a convert() method whose output is passed directly to the Rashomon Analysis module, and a save\_results() method for caching converted artefacts to disk.

Rashomon Analysis. The core module implements two classes. RashomonSet accepts converter output together with a base metric � and a tolerance �, constructs the set $R _ { \epsilon } ( h _ { 0 } ^ { M } , M )$ , and exposes methods for every metric describe in Section 3.1: Rashomon Ratio, Pattern Rashomon Ratio, Ambiguity, Discrepancy (binary, probabilistic, and multiclass variants), Viable Prediction Range, Rashomon Capacity, Percent Agreemen Agreement Rate, and Cohen’s Kappa. A summarize\_rashomon() met returns a consolidated view of all metrics for rapid inspection. RashomonIntersection extends RashomonSet to the multi-metric setting: it computes two independent Rashomon sets under metrics $M _ { 1 }$ and $M _ { 2 }$ , takes their intersection, and selects a joint reference model via the weighted-sum approach described in Section 3.1 (custom weights, Entropy, or CRITIC). All single-metric metrics are inherited and recomputed on the intersection.

Visualizers. The Visualizer module provides interactive Plotly-based plots covering the full set of Rashomon metrics:

• gauge and scatter plots for Rashomon Ratio and Pattern Rashomon Ratio across � values;

• lollipop charts for Ambiguity and Discrepancy;

• histograms and box plots for Rashomon Capacity and VPR widths;

• agreement bar charts and Cohen’s Kappa heatmaps;

• a feature-importance heatmap across all models in the set.

IntersectionVisualizer adds a Venn diagram of the two constituent Rashomon sets and a scatter plot of model scores with the Pareto front highlighted, enabling comparison with standard multi-objective optimization. All plots include hover tooltips for precise numerical inspection.

Pipelines. The Pipelines module integrates the three preceding modules into end-to-end workflows that require minimal user efort. Seven concrete Pipeline classes cover every combination of input type (raw AutoGluon predictor, saved H2O models, pre-converted results) and analysis target (RashomonSet or RashomonIntersection). Each exposes three methods: preview\_rashomon(), which plots Rashomon set size against � to guide threshold selection; set\_epsilon(); and build(), which constructs the analysis objects and launches a local Streamlit dashboard. This design allows practitioners to obtain a full Rashomon analysis from a trained AutoML predictor in three lines of code, while retaining access to individual modules for custom workflows.

## B Streamlit application

An additional product of this work is a Streamlit web application published on the Streamlit Cloud, allowing users to experiment with the package functionalities in a no-code manner. It consists of four pages - Home page, Datasets page, Rashomon page, and the Intersection page. The initial view appearing after visiting the website provides a brief overview of the application’s main purpose and key features, along with the intuitive explanations of the predictive multiplicity problem, the Rashomon Efect, and the introduced approach of the Rashomon Intersection. On the Datasets page, users can find detailed descriptions of the eight pre-saved datasets available for analysis. Finally, on the Rashomon page and on the Intersection page, users can construct the corresponding objects by selecting a dataset and specifying the related parameters using interactive widgets in order to view the dashboard for analysis.

## Home page

The home page contains the application overview and an intuitive explanation of the predictive multiplicity problem and the Rashomon Efect. On this page, users can gain an understanding of the key definitions and metrics illustrated on the dashboards, as well as access articles from the bibliography. Figure 6 presents the application’s Home screen that appears after visiting the website, while Figure 7 illustrates an example section with the Rashomon set description. This page contains similar, intuitive explanations of the newly introduced concept of the Rashomon Intersection, along with all metrics related to the predictive multiplicity problem. We provided some real-life examples, along with diagrams, to allow easier understanding of these concepts without the wide knowledge of this topic. We decided not to include formal definitions, which can be found in the literature, in the application, as it would not be helpful for most of the users and would reduce the readability of the page. Instead, we provided links to the key articles used in this project.

![](images/108564069697ed75da0fb85fca264f964412b9ae0569d3ee4d4ad09dc1009d30.jpg)  
Figure 6: Initial screen of the ARSA ML web application. It contains a top navigation bar that allows<sup>classification</sup> <sup>problem.</sup> <sup>It</sup> <sup>was</sup> <sup>developed</sup> <sup>to</sup> <sup>enable</sup> <sup>experimentation</sup> <sup>with</sup> <sup>the</sup> <sup>library's</sup> <sup>features</sup> <sup>in</sup> <sup>a</sup> <sup>no-code</sup> <sup>environment</sup> <sup>he</sup> <sup>Rashomon</sup> <sup>Set</sup> <sup>concept</sup>switching between pages, a package logo, and a brief overview of the application’s function-Datasets page. The application consists of two analytical dashboards : the Rashomon page and the Intersection page. alities. Additional Home page elements are visible upon scrolling.<sup>Detailed</sup> <sup>explanations</sup> <sup>of</sup> <sup>the</sup> <sup>key</sup> <sup>concepts,</sup> <sup>such</sup> <sup>as</sup> <sup>the</sup> <sup>Rashomon</sup> <sup>Set,</sup> <sup>the</sup> <sup>Rashomon</sup>

![](images/0ac50863cfe43831f771edb37a70bd484b65f5abaf072b1658e8d351f3b096e1.jpg)  
<sup>All</sup> <sup>of</sup> <sup>these</sup> <sup>considerations</sup> <sup>should</sup> <sup>be</sup> <sup>carefully</sup> <sup>analyzed</sup> <sup>before</sup> <sup>making</sup> <sup>any</sup> <sup>final</sup> <sup>decisions,</sup> <sup>especially</sup> <sup>when</sup> <sup>these</sup> <sup>decisions</sup> <sup>have</sup> <sup>a</sup> <sup>direct</sup> <sup>impact</sup> <sup>on</sup><sub>Regarding</sub> <sub>the</sub> <sub>illustration,</sub> <sub>the</sub> <sub>best</sub> <sub>model</sub> <sub>predicted</sub> <sub>the</sub> <sub>negative</sub> <sub>class</sub> <sub>for</sub> <sub>the</sub> <sub>client</sub> <sub>number</sub> <sub>10.</sub> <sub>Depending</sub> <sub>on</sub> <sub>the</sub> <sub>classification</sub> <sub>problem,</sub> <sub>this</sub> <sub>could</sub>Figure 7: One of the Home page elements - an intuitive explanation of the Rashomon set concept <sup>mean</sup> <sup>that</sup> <sup>the</sup> <sup>client</sup> <sup>will</sup> <sup>not</sup> <sup>receive</sup> <sup>a</sup> <sup>loan</sup> <sup>or</sup> <sup>insurance,</sup> <sup>or</sup> <sup>that</sup> <sup>they</sup> <sup>are</sup> <sup>not</sup> <sup>a</sup> <sup>carrier</sup> <sup>of</sup> <sup>a</sup> <sup>disease.</sup> <sup>Upon</sup> <sup>closer</sup> <sup>examination</sup> <sup>of</sup> <sup>this</sup> <sup>situation,</sup> <sup>we</sup> <sup>notice</sup>based on a sample scenario presented on a graphic. An expandable panel contains additional <sub>gives</sub> <sub>a</sub> <sub>different</sub> <sub>prediction</sub> <sub>for</sub> <sub>the</sub> <sub>client</sub> <sub>number</sub> <sub>10.</sub> <sub>This</sub> <sub>problem</sub> <sub>is</sub> <sub>reffered</sub> <sub>to</sub> <sub>as</sub> <sub>predictive</sub> <sub>multiplicity</sub> <sub>problem.</sub>information about the importance of the detailed analysis of the predictive multiplicity problem before the decision-making process.

## Datasets page

This page is dedicated to a detailed description of the pre-saved datasets available for experimentation. It contains information about the source, license, classification task type, and dataset characteristics supported by additional charts. For the purpose of this project, we selected eight datasets and divided them into three categories: ’business application’, ’predictive multiplicity problem’, and ’datasets challenging for ML algorithms’. The first category represents datasets that are related to employment, finance, and banking, such as a credit score classification dataset. The second category consists of datasets for which we find the predictive multiplicity to be especially problematic. Those datasets are related to recidivism, bias, or disease prediction. The last group contains datasets with class imbalance, feature correlation, or multiple classes, which are often causes of incorrect predictions produced by ML models. Table 2 presents information about the selected datasets.

Table 2: A short overview of the eight datasets selected as pre-saved datasets for analysis in the Streamlit application. Column ’Classes’ contains the number of unique labels in the target column, ’Features’ and ’Observations’ inform about the number of features and samples in the dataset, and ’Category’ reflects one of the 3 categories described previously.
<table><tr><td>Dataset name</td><td>Classes</td><td>Features</td><td>Observations</td><td>Category</td></tr><tr><td>HR Job Change</td><td>2</td><td>13</td><td>19158</td><td>Business Application</td></tr><tr><td>Credit Score</td><td>3</td><td>7</td><td>164</td><td>Business Application</td></tr><tr><td>Breast Cancer</td><td>2</td><td>32</td><td>569</td><td>Predictive Multiplicity Problem</td></tr><tr><td>Heart Failure</td><td>2</td><td>11</td><td>918</td><td>Predictive Multiplicity Problem</td></tr><tr><td>COMPAS</td><td>2</td><td>11</td><td>6172</td><td>Predictive Multiplicity Problem</td></tr><tr><td>Glass Types</td><td>6</td><td>9</td><td>214</td><td>Challenging For ML Algorithms</td></tr><tr><td>Letter Recognition</td><td>26</td><td>16</td><td>20000</td><td>Challenging For ML Algorithms</td></tr><tr><td>Yeast</td><td>10</td><td>9</td><td>1484</td><td>Challenging For ML Algorithms</td></tr></table>

All datasets presented in Table 2 are described on the Datasets page. Figure 8 illustrates the view of the Datasets page before selecting a particular dataset for analysis. After choosing a dataset, users can expand a corresponding panel and view all information about the chosen dataset, such as the source link, license, feature description, and the dataset’s characteristics. Figure 9 presents a part of the panel’s content for the COMPAS dataset containing its characteristics and the supporting charts.

![](images/d3cfcc3cfda82b720f85b3697cc78e353428ae1ba283ebcf32fe32f88cf734d2.jpg)  
Figure 8: The initial screen of the Datasets page with expandable panels allowing the exploration of every pre-saved dataset.

![](images/5e60dee7c89960909786498bfe51c5d554e88c7f247a39dce3b2237b2701eaa3.jpg)  
Figure 9: A part of the expanded panel for the COMPAS dataset. It contains a short description of the dataset characteristics, the percentage of missing values found in a dataset, along with class distribution and numerical features correlation charts.

## Rashomon page

The primary page of the application is the Rashomon page, where users can select custom parameters and analyze the properties of the created Rashomon set. Firstly, one of the pre-saved datasets should be selected for analysis. Then, widgets allowing the specification of other parameters, such as the base metric, epsilon, and delta (only for binary classification tasks), appear on the screen. The available widgets with sample parameter selection are illustrated in Figure 10.

![](images/bbe80096494f62b6e7b75a13fd331a0236ce6fa61025f27e5f1fef5a85bd4c8f.jpg)  
Figure 10: The initial view of the Rashomon page providing parameter selection using interactive widgets. After specifying each parameter, corresponding information appears below the luonwidget.

After all parameters have been selected, an interactive dashboard with multiple charts is displayed, allowing a detailed analysis of the attributes and metrics related to the constructed Rashomon set. Additionally, users can switch between pages illustrating the Rashomon sets created using AutoGluon and H2O models with the same parameter configuration in order to explore the diferences between these two frameworks. Figure 11 illustrates initial plots appearing on the dashboard after successfully constructing the Rashomon set. They contain primary characteristics of the constructed Rashomon set, such as the reference model, the base metric, and the number of included models. Additionally, Figure 12 presents sample plots available for analysis of the Viable Prediction Range metric calculated for each sample using the models from the Rashomon set. Similarly, all plots returned by the Visualizer class are included in the dashboard and available upon scrolling.

![](images/2a9e16e4edc6f8f33eca926f769138530ef690bb5aee94211b6e2b0112cb2585.jpg)  
Figure 11: Initial screen ofthe Rashomon set analysis dashboard. On the left-hand side ofthe dashboard, users may find a table containing all models that are included in the Rashomon set for specified parameters, as well as a brief description of the table properties. On the right-hand side, there is a gauge plot illustrating the fraction of models from the leaderboard that are a part of the Rashomon set, along with the selected parameters and the name of the reference model.

![](images/18d2dac652f520c2384b158825548398576b3ce832f3544bf1b51dcea7235676.jpg)  
1 <sup>This</sup> <sup>violin</sup> <sup>plot</sup> <sup>shows</sup> <sup>how</sup> <sup>the</sup> <sup>base</sup> <sup>model's</sup>Figure 12: Sample plots available on the Rashomon page of the application. The chart on the left is a Rashomon set. For each observation, thehistogram presenting the distribution of the VPR widths across all samples in the analyzed [min risk prediction, max risk prediction] acrossdataset, while the box plot on the right-hand side illustrates the widths with respect to the true labels.

Figure 13 presents plots from the dashboard, related to feature importance obtained from the models in the Rashomon set. For some complex models, such as stacked ensembles, feature importance is not provided by the H2O framework. In such cases, a corresponding heatmap row remains empty, which is illustrated on the following charts. Figure 13a illustrates the top three most important features obtained from the reference model in comparison with the three features selected as the most important ones by the majority of the near-optimal models. Figure 13b illustrates a heatmap enabling a more detailed analysis of these features for each model in the Rashomon set.

Top 3 Feature Importance in Rashomon Set  
![](images/10adb0415b98633384d5b4672fa8712e2f3f7016533bffed95efd3ba706709e5.jpg)  
This table compares the top 3 most important features of the base model -  
StackedEnsemble\_BestOfFamily\_1\_AutoML\_1\_20251028\_140001, with the features that appear most frequently in the Rashomon Set at positions 1, 2, and 3.  
If the base model does not provide feature importance values, a ‘–’ will be displayed to indicate that this information is unavailable.

## (a) Table illustrating a scenario, when feature importance cannot be obtained from the reference model. In such cases, the first column of the table remains empty.

Top 3 Feature Importance Across Models  
![](images/dded2f094a77d1c51f11ae82457664ba081b09e56de01f6419eb787606456854.jpg)  
(b) A heatmap presenting the top 3 most important features obtained from every model in the Rashomon set. Regarding StackedEnsembleBestOfFamily and StackedEnsembleAllModels models, feature importances could not be obtained. Therefore, the corresponding matrix rows are empty.  
Figure 13: Feature importance plots available on the dashboard.

## Intersection page

The last page of the application is dedicated to the analysis of the newly introduced concept of the Rashomon Intersection. Similar to the Rashomon page, users are asked to select a dataset and specify the necessary parameters, such as two evaluation metrics, epsilon and a weighted sum method (’custom weights’, ’entropy’, or ’CRITIC’). Figure 14 presents the initial view providing widgets for parameter selection. Similarly, after specifying certain parameters, the corresponding information is displayed below the widget.

![](images/112eb83bdcbf707a2aae2ad2c41f6794dcc836483bfbb92d5b5fa65ff144aa76.jpg)

![](images/f1b582ef8e86044ea273128a0649ed47e08da27d46a48df7e1a992c19bd52754.jpg)  
<sub></sub>Figure 14: The initial view of the Intersection page providing parameter selection using interactive widgets. Please note that after selecting the ’custom weights’ weighted sum method, onadditional widgets for weights specification appear on the screen, allowing the selection of a weight for the first evaluation metric, while the second weight is calculated automatically as 1 − �<sub>1</sub>.

Then, a dashboard, resembling the one on the Rashomon page, is displayed. Since the metrics for the Rashomon set also apply to the Rashomon Intersection, many plots from the Rashomon page are reused on the Intersection page. Additionally, we enable the comparison between the properties of the Rashomon Intersection and a widely used concept of the Pareto Front. Figure 15 presents the first part of the dashboard appearing on the Intersection page after the specification of all necessary parameters. Additionally, Figure 16 presents sample plots available on the Intersection page that enable a brief analysis of the separate Rashomon sets (for both specified evaluation metrics) in comparison to their intersection. Please note that all plots visualizing metrics related to the Rashomon set, which are available on the Rashomon page, are also available for analysis of the Rashomon Intersection, as the predictive multiplicity metrics apply for both objects.

![](images/f7ad617a97cfb423a83b3c8a24711a9c459649f4a1a5aa8e5053c05fa7066ffa.jpg)  
accuracy, precision and epsilon value : 0.050. The values of the base metrics are rounded to <sup>models,</sup> <sup>there</sup> <sup>is</sup> <sup>no</sup> <sup>other</sup> <sup>model</sup> <sup>that</sup> <sup>simultaneously</sup> <sup>achieves</sup> <sup>better</sup> <sup>scores</sup> <sup>in</sup> <sup>al</sup> <sup>evaluation</sup>Figure 15: The first part of the dashboard appearing on the Intersection page after the successful<sub>metrics.</sub> <sub>At</sub> <sub>least</sub> <sub>one</sub> <sub>metric</sub> <sub>is</sub> <sub>superior</sub> <sub>compared</sub> <sub>to</sub> <sub>other</sub> <sub>models.</sub> parameter specification and construction of the Rashomon Intersection. The table on the left-hand side presents models included in the Intersection (here for AutoGluon framework), along with their scores in both evaluation metrics. The right side of the dashboard focuses on the multi-objective optimization approach that computes the Pareto Front.

![](images/966da1cd97d1d9a62b705c26fea542a5d70dad7be3047846a00f72eec73a3d3b.jpg)  
Figure 16: Sample plots available on the Intersection page allowing the analysis of the separate Rashomon sets, as well as their intersection. Here we present an example scenario using the H2O framework. The Venn diagram and gauge plots illustrate which models did not achieve high enough precision score to be included in the Rashomon Intersection.

## C Rashomon Intersection - selection of reference models

To choose the weight values for selection of the reference model, we propose three methods: (1) the Custom weights method, (2) the Entropy method, and (3) the Criteria Importance Through Intercriteria Correlation (CRITIC) method. Methods (2) and (3) are based on the distributions of the metric values $M _ { 1 }$ and $M _ { 2 }$ for each of the � models in the Rashomon Intersection. Let $\hat { M } _ { 1 } =$ $[ \hat { M } _ { 1 1 } , . . . \hat { M _ { m 1 } } ]$ denote a vector of $M _ { 1 }$ values across all � models. Similarly, let $\hat { M _ { 2 } } = [ \hat { M _ { 1 2 } } , . . . \hat { M _ { m 2 } } ]$ denote a vector of $M _ { 2 }$ values across all � models.

1. Custom weights - weight values are selected based on subjective preferences; when constructing the Rashomon Intersection, we can specify the importance of each metric for a given classification problem. For example, we can assign $w _ { 1 } = 0 . 3$ to metric $M _ { 1 }$ and $w _ { 2 } = 0 . 7$ to metric $M _ { 2 }$

2. Entropy method - weights are calculated based on the theory of informational uncertainty (Wang et al. (2023)). Intuitively, a given objective (e.g., an evaluation metric �) will be assigned a higher weight if the diferences in its values are more significant compared to those of other objectives (e.g, other metrics). For a better understanding of the Entropy method, let’s consider the Rashomon Intersection consisting of � models and two evaluation metrics $M _ { 1 }$ and $M _ { 2 }$ (i.e., objectives). The process of calculating weights $w _ { 1 }$ and $w _ { 2 }$ is as follows.

(a) Normalize the values of $\hat { M } _ { 1 }$ and $\hat { M } _ { 2 }$ achieved by models using the sum normalization. For $j \in \{ 1 , 2 \}$

$$
F _ { i j } = \frac { \hat { M _ { i j } } } { \sum _ { k = 1 } ^ { m } \hat { M _ { k j } } } ,\tag{13}
$$

where $i \in [ m ]$ and $\hat { M _ { i j } }$ is the value of the evaluation metric $M _ { j }$ (here $M _ { 1 }$ or $M _ { 2 } )$ achieved by the model �.

(b) For each metric, calculate its entropy as

$$
E _ { j } = - \frac { 1 } { l n ( m ) } \sum _ { i = 1 } ^ { m } ( F _ { i j } \ l n ( F _ { i j } ) ) ,\tag{14}
$$

where $j \in \{ 1 , 2 \}$ , � is the number of models in the Rashomon Intersection, and $F _ { i j }$ is the normalized value of the evaluation metric $M _ { j }$ for the model �.

(c) Calculate weights for each metric as

$$
w _ { j } = \frac { 1 - E _ { j } } { \left( 1 - E _ { 1 } \right) + \left( 1 - E _ { 2 } \right) } .\tag{15}
$$

3. CRITIC method - by definition, weight values are computed based on the correlation matrix between objectives and the standard deviation of each objective (Wang et al. (2023)).

To understand how weights are assigned for two criteria (here, two diferent evaluation metrics), we present the CRITIC formula for the case of two objectives.

Assume we have the Rashomon Intersection consisting of� models and two evaluation metrics $M _ { 1 }$ and $M _ { 2 }$ (i.e., objectives). The $\hat { M } _ { 1 }$ and $\hat { M } _ { 2 }$ are normalized the same as in the Entropy method. Let $\hat { \rho }$ denote the empirical correlation matrix between random variables $\hat { M } _ { 1 }$ and $\hat { M } _ { 2 }$ . Formally,

$$
\hat { \rho } = \left[ \begin{array} { l l } { \hat { \rho } _ { 1 1 } } & { \hat { \rho } _ { 1 2 } } \\ { \hat { \rho } _ { 2 1 } } & { \hat { \rho } _ { 2 2 } } \end{array} \right] ,
$$

where $\hat { \rho } _ { 1 1 } = \hat { \rho } _ { 2 2 } = 1$ (correlation of a random variable with itself) and $\hat { \rho } _ { 1 2 } = \hat { \rho } _ { 2 1 }$ is the empirical correlation between $\hat { M } _ { 1 }$ and $\hat { M } _ { 2 }$

Additionally, let $\hat { \sigma } _ { 1 } , \hat { \sigma } _ { 2 }$ denote empirical standard deviations for $\hat { M } _ { 1 }$ and $\hat { M } _ { 2 } .$ , respectively.

Formally, to compute weights using the CRITIC method, we define:

$$
\begin{array} { r } { c _ { 1 } = \hat { \sigma } _ { 1 } \big [ \big ( 1 - \hat { \rho } _ { 1 1 } \big ) + \big ( 1 - \hat { \rho } _ { 1 2 } \big ) \big ] , } \end{array}\tag{16}
$$

$$
c _ { 2 } = \hat { \sigma } _ { 2 } \big [ \big ( 1 - \hat { \rho } _ { 2 1 } \big ) + \big ( 1 - \hat { \rho } _ { 2 2 } \big ) \big ] .\tag{17}
$$

If $\hat { \sigma } _ { 1 } = 0$ or $\hat { \sigma } _ { 2 } = 0$ , the correlation matrix $\hat { \rho }$ cannot be computed and the method fails. If $\hat { \sigma } _ { 1 } , \hat { \sigma } _ { 2 } >$ 0, then $\hat { \rho } _ { 1 1 } = \hat { \rho } _ { 2 2 } = 1$ and the above formulas can be simplified to:

$$
\begin{array} { r } { c _ { 1 } = \hat { \sigma } _ { 1 } \big ( 1 - \hat { \rho } _ { \hat { M } 1 , \hat { M } 2 } \big ) , \quad c _ { 2 } = \hat { \sigma } _ { 2 } \big ( 1 - \hat { \rho } _ { \hat { M } 1 , \hat { M } 2 } \big ) , } \end{array}\tag{18}
$$

where $\hat { \rho } _ { M 1 , M 2 } = \hat { \rho } _ { 1 2 } = \hat { \rho } _ { 2 1 }$ denotes empirical correlation between $\hat { M } _ { 1 }$ and $\hat { M } _ { 2 }$ .

Weights using the CRITIC method are defined as:

$$
w _ { 1 } = \frac { c _ { 1 } } { c _ { 1 } + c _ { 2 } } = \frac { \hat { \sigma } _ { 1 } \bigl ( 1 - \hat { \rho } _ { \hat { M } 1 , \hat { M } 2 } \bigr ) } { \bigl ( \hat { \sigma } _ { 1 } + \hat { \sigma } _ { 2 } \bigr ) \bigl ( 1 - \hat { \rho } _ { \hat { M } 1 , \hat { M } 2 } \bigr ) } = \frac { \hat { \sigma } _ { 1 } } { \hat { \sigma } _ { 1 } + \hat { \sigma } _ { 2 } } ,\tag{19}
$$

$$
w _ { 2 } = \frac { c _ { 2 } } { c _ { 1 } + c _ { 2 } } = \frac { \hat { \sigma } _ { 2 } \bigl ( 1 - \hat { \rho } _ { \hat { M } 1 , \hat { M } 2 } \bigr ) } { \bigl ( \hat { \sigma } _ { 1 } + \hat { \sigma } _ { 2 } \bigr ) \bigl ( 1 - \hat { \rho } _ { \hat { M } 1 , \hat { M } 2 } \bigr ) } = \frac { \hat { \sigma } _ { 2 } } { \hat { \sigma } _ { 1 } + \hat { \sigma } _ { 2 } } ,\tag{20}
$$

In this case, the weights depend only on the standard deviations of the metric values.

As mentioned above, this method is not always feasible. For instance, when either of the objectives has zero variance, the method fails. This may occur when all models present in the Rashomon Intersection produce the same values for one of the selected evaluation metrics.

## D Results across datasets

Table 3: Reference model performance per dataset.
<table><tr><td rowspan="2">#</td><td rowspan="2">Dataset Name</td><td rowspan="2">OpenML No. ID</td><td rowspan="2">Feat.</td><td colspan="2">Accuracy</td><td colspan="2">AUC</td></tr><tr><td>AG</td><td>H2O</td><td>AG</td><td>H2O</td></tr><tr><td>1</td><td>APSFailure</td><td>46908</td><td>171</td><td>0.9956</td><td>0.9945</td><td>0.9937</td><td>0.9917</td></tr><tr><td>2</td><td>Amazon_employee_access</td><td>46905</td><td>10</td><td>0.9548</td><td>0.9498</td><td>0.8991</td><td>0.8635</td></tr><tr><td>3</td><td>Bank_Customer_Churn</td><td>46911</td><td>11</td><td>0.8672</td><td>0.8604</td><td>0.8775</td><td>0.8751</td></tr><tr><td>4</td><td>Diabetes130US</td><td>46922</td><td>48</td><td>0.9127</td><td>0.8862</td><td>0.6735</td><td>0.6709</td></tr><tr><td>5</td><td>E-CommereShippingData</td><td>46924</td><td>11</td><td>0.6898</td><td>0.6658</td><td>0.7533</td><td>0.7549</td></tr><tr><td>6</td><td>Fitness_Club</td><td>46927</td><td>7</td><td>0.7973</td><td>0.7893</td><td>0.8377</td><td>0.8309</td></tr><tr><td>7</td><td>GiveMeSomeCredit</td><td>46929</td><td>11</td><td>0.9389</td><td>0.9303</td><td>0.8703</td><td>0.8543</td></tr><tr><td>8</td><td>HR_Analytics_Job_Change_of</td><td>46935</td><td>13</td><td>0.8006</td><td>0.7972</td><td>0.8103</td><td>0.8049</td></tr><tr><td>9</td><td>_Data_Scientists Is-this-a-good-customer</td><td>46938</td><td>14</td><td>0.8886</td><td>0.8886</td><td>0.8015</td><td>0.8033</td></tr><tr><td>10</td><td>Marketing_Campaign</td><td>46940</td><td>26</td><td>0.9107</td><td>0.9018</td><td>0.9252</td><td>0.9211</td></tr><tr><td>11</td><td>NATICUSdroid</td><td>46969</td><td>87</td><td>0.9557</td><td>0.9536</td><td>0.9882</td><td>0.9884</td></tr><tr><td>12</td><td>bank-marketing</td><td>46910</td><td>14</td><td>0.8944</td><td>0.8777</td><td>0.7720</td><td>0.7761</td></tr><tr><td>13</td><td>blood-transfusion-service-center</td><td>46913</td><td>5</td><td>0.8503</td><td>0.8503</td><td>0.8377</td><td>0.8330</td></tr><tr><td>14</td><td>churn</td><td>46915</td><td>20</td><td>0.9672</td><td>0.9728</td><td>0.9406</td><td>0.9409</td></tr><tr><td>15</td><td>coil2000_insurance_policies</td><td>46916</td><td>86</td><td>0.9409</td><td>0.9267</td><td>0.7813</td><td>0.7759</td></tr><tr><td>16</td><td>credit-g</td><td>46918</td><td>21</td><td>0.7840</td><td>0.7800</td><td>0.8034</td><td>0.8140</td></tr><tr><td>17</td><td>credit_card_clients_default</td><td>46919</td><td>24</td><td>0.8240</td><td>0.8071</td><td>0.7957</td><td>0.7921</td></tr><tr><td>18</td><td>customer_satisfaction_in_airline</td><td>46920</td><td>22</td><td>0.9621</td><td>0.9606</td><td>0.9950</td><td>0.9948</td></tr><tr><td>19</td><td>diabetes</td><td>46921</td><td>9</td><td>0.8281</td><td>0.8125</td><td>0.8836</td><td>0.8715</td></tr><tr><td>20</td><td>hazelnut-spread-contaminant-</td><td>46930</td><td>31</td><td>0.9500</td><td>0.9517</td><td>0.9862</td><td>0.9877</td></tr><tr><td></td><td>detection</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>21 22</td><td>heloc in_vehicle_coupon_recommendation</td><td>46932 46937</td><td>24</td><td>0.7358</td><td>0.7285</td><td>0.8086</td><td>0.8104</td></tr><tr><td></td><td></td><td></td><td>25</td><td>0.7846</td><td>0.7789</td><td>0.8505</td><td>0.8564</td></tr><tr><td>23</td><td>jm1</td><td>46979</td><td>22</td><td>0.8254</td><td>0.8012</td><td>0.7690</td><td>0.7691</td></tr><tr><td>24</td><td>online_shoppers_intention</td><td>46947</td><td>18</td><td>0.9089</td><td>0.9056</td><td>0.9371</td><td>0.9372</td></tr><tr><td>25</td><td>polish_companies_bankruptcy</td><td>46950</td><td>65</td><td>0.9689</td><td>0.9668</td><td>0.9674</td><td>0.9540</td></tr><tr><td>26</td><td>qsar-biodeg</td><td>46952</td><td>42</td><td>0.9011</td><td>0.9053</td><td>0.9483</td><td>0.9579</td></tr><tr><td>27</td><td>seismic-bumps</td><td>46956</td><td>16</td><td>0.9350</td><td>0.9350</td><td>0.8231</td><td>0.8024</td></tr><tr><td>28</td><td>taiwanese_bankruptcy_prediction</td><td>46962</td><td>95</td><td>0.9765</td><td>0.9754</td><td>0.9662</td><td>0.9663</td></tr></table>

## E Algorithm mapping

Table 4: Mapping of model name keywords to model type categories.
<table><tr><td>Keyword match</td><td>Example model names</td><td>Category</td><td>Framework</td></tr><tr><td>lightgbm</td><td>LightGBM, LightGBMLarge, LightGBMXT</td><td>Gradient Boosting</td><td>AG</td></tr><tr><td>xgboost</td><td>XGBoost, XGBoost BAG L1</td><td>Gradient Boosting</td><td>AG</td></tr><tr><td>catboost</td><td>CatBoost, CatBoost BAG L2</td><td>Gradient Boosting</td><td>AG</td></tr><tr><td>gbm</td><td>GBM, GBM_1</td><td>Gradient Boosting</td><td>H2O</td></tr><tr><td>drf</td><td>DRF, DRF_1</td><td>Tree Based Models</td><td>H2O</td></tr><tr><td>extratrees</td><td>ExtraTreesEntr, ExtraTreesGini</td><td>Tree Based Models</td><td>AG</td></tr><tr><td>forest</td><td>RandomForest, RandomForestEntr</td><td>Tree Based Models</td><td>AG</td></tr><tr><td>xrt</td><td>XRT, XRT_1</td><td>Tree Based Models</td><td>H2O</td></tr><tr><td>deeplearning</td><td>DeepLearning, DeepLearning_1</td><td>Neural Networks</td><td>H2O</td></tr><tr><td>neural</td><td>NeuralNetFastAI, NeuralNetTorch</td><td>Neural Networks</td><td>AG</td></tr><tr><td>glm</td><td>GLM, GLM_1</td><td>Linear Models</td><td>H2O</td></tr><tr><td>linear</td><td>LinearModel, LinearModel_BAG_L1</td><td>Linear Models</td><td>AG</td></tr><tr><td>ensemble</td><td>WeightedEnsemble_L3, StackedEnsemble_AllModels</td><td>Model Ensemble</td><td>AG, H2O</td></tr><tr><td>(no match)</td><td>KNeighbors, NaiveBayes</td><td>Other</td><td>AG, H2O</td></tr></table>

## F Limitations

Several limitations of the present study should be acknowledged.

Scope restricted to binary classification. All experiments are conducted exclusively on binary classification tasks retrieved from OpenML. Whether the observed diferences in Rashomon set structure between AutoGluon and H2O generalise to multiclass classification, regression, or other prediction tasks remains an open question. The metrics employed—ambiguity, discrepancy, VPR, and Rashomon Capacity—admit natural multiclass extensions (Hsu and Calmon, 2022), but their empirical behaviour in those settings has not been examined here.

Framework configuration asymmetry. AutoGluon was configured with the good\_quality preset, which employs bagging and stacking and can produce a large pool of reference models, while H2O AutoML was capped at 20 models. Although both frameworks were given an equivalent time budget of 2h per fold, the ceiling on H2O’s model count is a structural confound: observed diferences in Rashomon Ratio and set size may partly reflect the number of models available rather than the intrinsic diversity of each framework’s hypothesis space. Future work should control for model count or report results stratified by pool size.

Inconsistency in feature importance estimation. For AutoGluon, feature importance is obtained directly from the predictor, whereas for H2O stacked ensembles—which do not natively expose compatible importance scores—permutation importance is computed post-hoc using scikit-learn, scored by accuracy. Non-ensemble H2O models use framework-internal estimates. Because the XHack analysis and RQ3 conclusions rest on feature importance rankings, this methodological inconsistency introduces a potential confound: measured diferences in explanation hackability between frameworks may partially reflect diferences in how importance is computed rather than genuine diferences in model behaviour.

Rashomon sets bounded by AutoML output. The Rashomon sets analyzed here are restricted to models that the AutoML framework chose to train and retain. Both AutoGluon and H2O return only a subset of all models explored during the search process, biased towards high-performing configurations. This means the empirical Rashomon set is a subset of the theoretical one, and conclusions about set size and diversity are conditional on each framework’s internal model selection and pruning strategy.

Single time budget. All experiments were conducted under a fixed time limit of 2h per fold for both frameworks. This choice reflects a single operating point in the trade-of between computational cost and model pool diversity. It is plausible that the observed diferences in Rashomon set size and structure between AutoGluon and H2O are sensitive to this budget.

ARSA ML framework coverage. The current implementation of ARSA ML provides native support for AutoGluon and H2O only. Although a custom converter interface exists for other frameworks, the efort required for integration may limit adoption. Results and tooling cannot be directly applied to other widely-used AutoML systems, such as auto-sklearn or FLAML, without additional engineering.

## G Negative societal impact

By quantifying and publicly reporting x-hackability scores across frameworks and datasets, this work provides information of where explanation manipulation is structurally easiest. Our findings demonstrate that H2O’s Rashomon sets consistently exhibit higher and more variable x-hackability than those of AutoGluon. While this result is intended to alert practitioners and auditors to framework-level risk, it could equally guide a malicious actor in selecting a framework and tolerance threshold that maximizes the availability of models supporting a desired narrative.

## H Compute Resources

Experiments described in Section 5 were computed on a cluster consisting of 2× AMD EPYC 7413 CPUs (48 cores), and 3TB of RAM. Each pipeline execution was constrained to a maximum memory usage of 32 GB and 8 cores.