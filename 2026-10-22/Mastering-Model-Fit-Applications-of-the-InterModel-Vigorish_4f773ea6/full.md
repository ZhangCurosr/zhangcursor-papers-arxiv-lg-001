# Mastering Model Fit: Applications of the InterModel Vigorish for Interpretable Machine Learning

Jiani Yan<sup>∗†,1,2</sup>, Pengyun Wang<sup>∗,3</sup>, Klint Kanopka<sup>4</sup>, Ben Domingue<sup>5</sup>, and Charles Rahal<sup>†,1,2</sup>

<sup>1</sup>Leverhulme Centre for Demographic Science, University of Oxford

<sup>2</sup>Nufield College, University of Oxford

<sup>3</sup>Data Science Institute, University of Chicago

<sup>4</sup>NYU Steinhardt, New York University

<sup>5</sup>Stanford Graduate School of Education, Stanford University

## Abstract

Accuracy alone gives little information about which features or model components improve probabilistic fit, which outcomes benefit, or how fit changes during training. We introduce the InterModel Vigorish (IMV) into the Machine Intelligence space in order to express diferences in fit as an expected betting advantage over a reference model. SHAP-IMV attributes gains to features through Shapley-weighted subset comparisons, Multi-IMV evaluates one-vs-rest and pairwise class discrimination, and Ablate-IMV assesses architectural components and feature groups by ablation. We evaluate these applications over ten random seeds using tabular data, images, sensor sequences, and graphs. Feature rankings vary across estimators. High one-vs-rest accuracy can accompany negligible improvement over a null, while ablation can preserve or improve fit under a fixed training budget. On six graph-classification benchmarks, we time-profile IMV over 200 training epochs, reporting seed means and ranges. The open-source imvpy library and accompanying notebooks provide code to replicate our work and to use the IMV more widely.

## 1 Introduction

Two classifiers can make identical class predictions yet assign diferent probabilities to the observed outcomes. In applications of machine learning across disciplines [1–4], choosing between them requires a measure suited to the comparison and an appropriate evaluation of generalization [5, 6]. Measures such as F1 [7], R<sup>2</sup> [8], cross-entropy [9], and AIC and BIC [10, 11] answer diferent questions. Threshold-based scores summarize classification decisions, probabilistic scores assess the probabilities assigned to outcomes, and information criteria account for complexity given a specified likelihood. Each can be used to compare models, but a diference in scores need not have the same interpretation relative to a baseline [12].

We adapt the InterModel Vigorish (IMV) [12, 13] — a model-agnostic measure for comparing binary probabilistic predictions — to three tasks. SHAP-IMV uses Shapley values [14, 15] to attribute gains in fit to features, conditional on the estimator, baseline, and feature set. Multi-IMV evaluates multiclass predictions through one-vs-all and one-vs-one comparisons [16], showing how fit difers across individual classes and class pairs. In Ablate-IMV, we remove components, refit the resulting models, and compare their probabilistic fit with that of the full model.

We examine logistic regression [17], XGBoost [18], Light-GBM [19], convolutional and bidirectional recurrent networks, multilayer perceptrons, and graph convolutional networks [20]. The experiments cover binary and multiclass tabular prediction, image parity, human activity recognition, and graph classification. Across these settings, IMV expresses gains in probabilistic fit relative to a reference model. Computing the score adds little cost once predictions are available; exact feature attribution is more demanding because it requires repeated model fitting.

## 2 Related Work

## 2.1 Feature Importance and Local Explanations

Model evaluation concerns both overall predictive performance and the contribution of individual predictors [21–23]. Feature-importance methods address the latter either globally, or for individual predictions [15]. Information Gain measures information gained at tree splits [24, 25], while Gini Importance measures reductions in impurity, commonly in random forests [26]. Mutual Information measures bivariate dependence [24]. These quantities describe particular algorithms or relationships between variables; they do not directly measure gains in held-out probabilistic fit. For individual predictions, SHAP assigns contributions to features [15, 27], LIME approximates the fitted model with a local linear surrogate [28], and Anchors constructs high-precision prediction rules [29, 30]. The local scope of these explanations can leave global behavior unexamined. SAGE uses a loss-based value function to allocate predictive power through global Shapley attribution [31]. Our SHAP-IMV also uses global Shapley attribution, but refits models for each feature subset and defines the value as held-out IMV relative to a baseline, so allocations are expressed in IMV units and sum to the full model’s gain.

## 2.2 Global Metrics for Machine Learning

Brier score [32], accuracy [33], precision, recall, F1 [34], and ROC area [35] summarize overall predictive performance [36]. IMV gives comparisons of probabilistic fit an explicit reference and a betting-scale interpretation [12]. Likelihood-ratio tests and information criteria provide other approaches to model comparison [10, 11]. Whichever measure is used, claims about generalization depend on the evaluation design. Here we use IMV to attribute gains in probabilistic fit to features, to evaluate multiclass predictions, and to assess the efects of model ablation.

## 3 Preliminaries

## 3.1 Problem Definition

For binary classification [12], let dataset � contain � observations with outcomes $Y _ { i } \in \{ 0 , 1 \}$ and � features $\mathbf { X } _ { i } \mathbf { \Psi } =$ $( X _ { i 1 } , \ldots , X _ { i d } )$ . Model $f$ assigns probability $p _ { i } = f ( \mathbf { X } _ { i } ) \in$ $( 0 , 1 )$ to $Y _ { i } = 1$ . The observed outcome contributes likelihood $L _ { i } = p _ { i } ^ { Y _ { i } } ( 1 - p _ { i } ) ^ { 1 - Y _ { i } }$ and log-likelihood $\ell _ { i } = Y _ { i }$ log $p _ { i } + ( 1 -$ $Y _ { i } )$ log $( 1 - p _ { i } )$ . We summarize fit by the geometric mean likelihood:

$$
L ( p ; D ) \equiv \left( \prod _ { i = 1 } ^ { N } p _ { i } ^ { Y _ { i } } ( 1 - p _ { i } ) ^ { 1 - Y _ { i } } \right) ^ { \frac { 1 } { N } } .\tag{1}
$$

## 3.2 The InterModel Vigorish (IMV)

IMV interprets a diference in probabilistic fit through an equivalent betting problem [12, 13, 37]. For $L ( p ; D ) \ge 0 . 5 .$ the equivalent coin weight $w \in [ 0 . 5 , 1 ]$ matches the coin’s expected log-likelihood to the model’s mean log-likelihood:

$$
\begin{array} { r } { \log L ( p ; D ) = w \log w + ( 1 - w ) \log ( 1 - w ) . } \end{array}\tag{2}
$$

Here log $\begin{array} { r } { L = N ^ { - 1 } \sum _ { i } \ell _ { i } } \end{array}$ . We extend � log � continuously to $x = 0$ by defining 0 log $0 : = 0$ . The right-hand side ranges from − log 2 to zero; consequently, no exact coin weight exists when $L < 0 . 5$

We use the default settings of imvpy 1.2.0 [38]. Probabilities are clipped to $[ 1 0 ^ { - 9 } , 1 - 1 0 ^ { - 9 } ]$ , and below-chance likelihoods receive the boundary weight $w = 0 . 5 \mathrm { i f } - \log ( 2 L ) \leq 0 . 5$ nats. Larger deficits produce a warning and an undefined score. The boundary rule is an implementation convention rather than an exact solution of Equation (2), so distinct poor predictions can receive the same weight.

Consider models $f _ { 0 }$ and $f _ { 1 }$ with coin weights � and $w _ { 1 }$ Bettor A stakes one dollar on heads and B stakes $1 / O$ dollars on tails. Fair odds for the baseline coin require

$$
w _ { 0 } \times \frac { 1 } { O } = \left( 1 - w _ { 0 } \right) \times 1 .\tag{3}
$$

Thus $1 / O = ( 1 - w _ { 0 } ) / w _ { 0 }$ . If the coin is replaced by one with weight $w _ { 1 }$ and the odds are held fixed, A’s expected profit becomes

$$
w _ { 1 } \times \frac { 1 - w _ { 0 } } { w _ { 0 } } - \left( 1 - w _ { 1 } \right) \times 1 = \frac { w _ { 1 } - w _ { 0 } } { w _ { 0 } } .\tag{4}
$$

This betting edge, or ‘vigorish’, defines IMV [12] (Figure 1). Positive values favor $f _ { 1 }$ and negative values favor baseline $f _ { 0 } .$ The comparison requires predictions for the same outcomes, in the same order. With the baseline and evaluation set held fixed,

IMV orders predictions identically to mean log-likelihood, and inversely to log loss, within the exact coin domain. Its upper bound is $1 / w _ { 0 } - 1$ , so the scale of possible improvement depends on the baseline and outcome prevalence.

## 3.3 Calculating IMV in Python

Box 1 illustrates an IMV calculation with our imvpy library. We simulate 1,000 binary outcomes from one predictor, fit logistic regression on 70% of the observations, and evaluate its predictions on the remaining 30%. For comparison, the baseline assigns the training-set prevalence to every test observation. We pass the baseline probabilities, enhanced probabilities, and observed outcomes to vanilla\_imv, in that order.

```python
Box 1: Vanilla IMV
# Install once: python -m pip install imvpy
import numpy as np
from imvpy import vanilla_imv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 1. Simulate a binary outcome from one predictor.
rng = np.random.default_rng(42)
x = rng.normal(size=(1000, 1))
p = 1 / (1 + np.exp(-(2 * x[:, 0] - 1)))
y = rng.binomial(1, p)
# 2. Fit on training data and predict test outcomes.
x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.3, random_state=42)
model = LogisticRegression().fit(x_train, y_train)
p_enhanced = model.predict_proba(x_test)[:, 1]
# 3. Compare against the training prevalence.
p_baseline = y_train.mean()
imv = vanilla_imv(p_baseline, p_enhanced, y_test)
print(f'IMV: {imv:.3f}')
```

The resulting IMV is approximately 0.282. This places the improvement in probabilistic fit on an immediately interpretable scale: if the betting odds are set so that the prevalence baseline would break even, replacing the baseline with the fitted model yields an expected profit of approximately \$0.282 for each \$1 staked. Thus, rather than expressing the improvement only as an abstract diference in likelihood or log-likelihood, IMV translates it into the more intuitive quantity of an expected advantage. This provides a simple way to communicate the magnitude of the improvement and to compare models evaluated against the same reference model and evaluation data.

## 4 Methodology

## 4.1 Overview

We use SHAP-IMV to attribute predictive value to features through Shapley weighting [15]. Multi-IMV compares classes one-vs-all and one-vs-one [39], while Ablate-IMV compares architectural components and feature groups. Each comparison measures improvement relative to an explicit reference model. The examples below use simulated data and the imvpy library to demonstrate these three applications.

## 4.2 SHAP-IMV for Feature Attribution

Let $\mathcal { F } = \{ 1 , \ldots , d \}$ index the features, and let $f _ { S }$ use subset $S \subseteq { \mathcal { F } }$ plus a constant predictor. The reference $f ( )$ is the same estimator fitted on the constant alone. We define $\nu ( S )$ as the mean held-out $\mathrm { I M V } ( f _ { S } , f ( ) )$ over cross-validation folds shared by all subsets, refitting both models on each fold’s training observations. The empty-subset and reference models coincide, so $\nu ( \emptyset ) = 0 .$ . The contribution of feature � is its Shapley-weighted average marginal gain [14, 15]:

![](images/3cb1d3876dbe128437a77ab7eb34a4a56568207e9d20eff71f8818196c87eaa5.jpg)  
Figure 1: IMV calculation and applications. Steps 1–3 start with the probabilities that reference model � and candidate model � assign to the same observed outcomes. We aggregate these as geometric mean likelihoods. When both are at least 0.5, Equation (2) gives equivalent coin weights �<sub>0</sub> and �<sub>1</sub>. The directional score IM $\mathsf { V } ( f _ { 1 } , f _ { 0 } ) = ( w _ { 1 } - \overset { \sim } { w _ { 0 } } ) \big / w _ { 0 }$ measures the relative change in coin weight; positive values indicate higher mean log-likelihood for �<sub>1</sub>. (A) SHAP-IMV attributes gains in fit to features through Shapley-weighted marginal contributions from refitted subset models, using a constant-only reference. (B) Multi-IMV reuses multiclass predictions for one-vs-all and pairwise comparisons, renormalizing probabilities within each class pair. (C) Ablate-IMV compares full and refitted ablated models on the same held-out observations to assess architectural components or feature groups. Scores depend on the reference and evaluation data. All data, probabilities, scores, and networks shown are schematic.

$$
\phi _ { j } = \sum _ { S \subseteq { \mathcal { F } } \backslash \{ j \} } { \frac { | S | ! ( d - | S | - 1 ) ! } { d ! } } { \big [ } \nu ( S \cup \{ j \} ) - \nu ( S ) { \big ] } .\tag{5}
$$

Including the empty subset ensures that contributions are averaged across all feature-entry orders. Before rounding, allocations satisfy Shapley eficiency, $\begin{array} { r } { \sum _ { j } \phi _ { j } = \nu ( \mathcal { F } ) - \nu ( \infty ) } \end{array}$ as well as symmetry, dummy-feature, and additivity properties. These properties do not guarantee unbiased estimation, causal identification, or social fairness. The values depend on the estimator, representation, reference, and evaluation data. Evaluating every subset requires $2 ^ { d }$ fits per fold, which motivates our eight-predictor specifications. Model fitting can therefore dominate the cost of IMV scoring. The library rounds contributions to three decimals, so displayed sums agree with �(F ) only within rounding precision. Section 4.2.1 gives a simulated example.

4.2.1 Simulated SHAP-IMV example. Box 2 generates 1,000 binary outcomes from two predictors. It uses BinaryIMV to evaluate every feature subset with logistic regression over three stratified folds, comparing each subset with a constant-only model fitted on the training observations.

The global SHAP-IMV values attribute the predictive gain to the two features. This example returns 0.209 for x1 and 0.056 for x2.

## 4.3 Multi-IMV for Multiclass Evaluation

4.3.1 One-vs-All Comparisons. Multi-IMV evaluates multiclass predictions through binary contrasts [39, 40]. Within each fold, we fit one enhanced multiclass estimator $f _ { 1 }$ and one constant-only reference $f _ { 0 }$ . Let $p _ { r i k }$ denote the probability assigned by model $r \in \{ 0 , 1 \}$ to class � for held-out instance �, with probability columns aligned by class label. For each class $k \in \{ 1 , \ldots , K \}$ , define

![](images/cccd7382dce5feb4be41d018606dbc57543dcd786a11bdf48cf6e3e3c91f3c1e.jpg)

$$
Y _ { i } ^ { ( k ) } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f } } Y _ { i } = k , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{6}
$$

We compare $p _ { 0 i k }$ and $p _ { 1 i k }$ with these binary outcomes on all held-out instances. This uses the original probability columns, without renormalization or separate binary classifiers. If � <sub>�</sub> and $w _ { 1 k }$ are the corresponding coin weights, then

$$
\mathrm { I M V } _ { k } ( f _ { 1 } , f _ { 0 } ) = \frac { w _ { 1 k } - w _ { 0 k } } { w _ { 0 k } } .\tag{7}
$$

Each class has its own reference weight, reflecting its binary prevalence and the reference probabilities on the evaluation fold.

4.3.2 The One-vs-One Multi-IMV Confusion Matrix. Pairwise evaluation focuses on discrimination between two classes.

It supplements one-vs-all evaluation but does not eliminate class imbalance. For classes $k \neq l ,$ retain only instances with $Y _ { i } \in \{ k , l \}$ and define

$$
Y _ { i } ^ { ( k , l ) } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f } } Y _ { i } = k , } \\ { 0 } & { { \mathrm { i f } } Y _ { i } = l . } \end{array} \right. }\tag{8}
$$

For both models, renormalize the retained class probabilities as $q _ { r i k } ^ { ( k , l ) } = p _ { r i k } / ( p _ { r i k } + p _ { r i l } )$ . We reuse the fitted multiclass estimators without training a model for each class pair. Let $w _ { r , k l }$ be the coin weight obtained from these probabilities and outcomes, and define $\lambda _ { k l } = \left( w _ { 1 , k l } - w _ { 0 , k l } \right) / w _ { 0 , k l }$ . For $K = 3 ,$ the (symmetric) Multi-IMV matrix is

$$
\Lambda = \left( \begin{array} { c c c } { { 0 } } & { { \lambda _ { 1 2 } } } & { { \lambda _ { 1 3 } } } \\ { { \lambda _ { 1 2 } } } & { { 0 } } & { { \lambda _ { 2 3 } } } \\ { { \lambda _ { 1 3 } } } & { { \lambda _ { 2 3 } } } & { { 0 } } \end{array} \right) .\tag{9}
$$

Swapping the two class labels complements both the binary outcomes and probabilities, leaving their likelihoods unchanged. Thus; $\lambda _ { k l } = \lambda _ { l k }$ , and we set the diagonal to zero by convention. Despite its name, this matrix contains baseline-relative probabilistic scores rather than confusion counts. We average both one-vs-all and pairwise scores across folds within each seed, then across seeds, and recommend reporting both alongside conventional metrics. Section 4.3.4 gives a simulated example.

4.3.3 Reading One-vs-All and Pairwise Scores. The two views answer diferent questions. $\mathrm { I M V } _ { k }$ is a class-specific diagnostic: it measures how much the features improve the probability of membership in class �, with all other classes pooled. $\lambda _ { k l }$ is a pairwise diagnostic: among instances from classes � and � only, it measures how much the features help separate the two. The views use diferent populations and reference weights, and neither is an average of the other.

Both scores are bounded by class balance. If the reference assigns the binary prevalence � to every instance, then log $L =$ � log $\pi + \left( 1 - \pi \right) \log ( 1 - \pi )$ , and Equation (2) gives $w _ { 0 } =$ max(�, 1 − �). The bound $1 / w _ { 0 } - 1$ is then the minorityto-majority ratio of the contrast: $\pi _ { k } / ( 1 - \pi _ { k } )$ for one-vs-all with $\pi _ { k } < 0 . 5 .$ , and min(�<sub>�</sub>, �<sub>�</sub>)/max(�<sub>�</sub>, �<sub>�</sub>) for a pair with class counts �<sub>�</sub> and $n _ { l }$ . A frequent class therefore has a high one-vs-all bound but low pairwise bounds against rarer classes, whereas a class of intermediate frequency can have high pairwise bounds throughout.

We therefore compare the same class or pair across models directly, and compare diferent classes or pairs relative to their bounds. The ratio $( w _ { 1 } - w _ { 0 } ) / ( 1 - w _ { 0 } )$ gives the share of the attainable improvement achieved. Values near one indicate near-complete separation, while pairs well below one show where classes remain confused.

4.3.4 Simulated Multi-IMVexample. Box 3 generates 900 observations with three outcome classes. It uses MulticlassIMV to compare logistic regression predictions with a constant-only model over three stratified folds.

The first output is the symmetric pairwise Multi-IMV matrix, with classes indexing its rows and columns and zeros on the diagonal. The second compares each class with all remaining classes. Both report mean IMV over the held-out folds.

```python
Box 3: Multi-IMV
import pandas as pd
from imvpy import MulticlassIMV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
X, y = make_classification(
n_samples=900, n_features=3, n_informative=3,
n_redundant=0, n_classes=3, n_clusters_per_class=1,
class_sep=0.5, flip_y=0, random_state=42)
features = ['x1', 'x2', 'x3']
data = pd.DataFrame(X, columns=features).assign(y=y)
multi = MulticlassIMV(data=data, outcome_variable= $" y " ,$
optional_explanatory_variables=features,
model_creator=lambda: LogisticRegression(max_iter=2000),
n_splits=3, random_state=42, stratified=True)
_, pairwise = multi.k_fold_imv_matrix()
_, one_vs_rest = multi.k_fold_one_vs_all()
print('Pairwise IMV:')
print(pairwise.round(3))
print('One-vs-rest IMV:')
print(pd.Series(one_vs_rest, index=pairwise.index).round(3))
```

## 4.4 Ablate-IMV for Model Component Comparisons

Ablation studies typically compare full models with variants using accuracy and other performance metrics [41, 42]. Ablate-IMV measures each variant’s probabilistic fit against a reference and compares variants directly. Consider a full model $F _ { 0 }$ , ablations $F _ { 1 }$ and $F _ { 2 }$ , and an uninformative reference $F ( )$ predicting 50% for every binary outcome $( w ( ) = 0 . 5 )$ Equation (4) gives

$$
\mathbf { I M V } ( F _ { i } , F ( \ O ) ) = \frac { w _ { i } - w ( \ O ) } { w ( \ O ) } , \quad \mathrm { w h e r e ~ } i \in \{ 0 , 1 , 2 \} .\tag{10}
$$

These scores measure fit relative to the 0.5-probability reference, not the original model. Comparing two ablations directly gives

$$
\mathbf { I M V } ( F _ { i } , F _ { j } ) = \frac { w _ { i } - w _ { j } } { w _ { j } } , \quad \mathrm { w h e r e } i , j \in \{ 1 , 2 \} , i \neq j .\tag{11}
$$

We can also compare the original model with each ablation:

$$
\mathrm { I M V } ( F _ { 0 } , F _ { i } ) = \frac { w _ { 0 } - w _ { i } } { w _ { i } } , \quad \mathrm { w h e r e } i \in \{ 1 , 2 \} .\tag{12}
$$

In the directional M, rows represent enhanced models and columns represent basic models:

$$
\mathbf { M } = \left( \begin{array} { c c c } { 0 . 0 } & { \mathbf { I M V } ( F _ { 0 } , F _ { 1 } ) } & { \mathbf { I M V } ( F _ { 0 } , F _ { 2 } ) } \\ { \mathbf { I M V } ( F _ { 1 } , F _ { 0 } ) } & { 0 . 0 } & { \mathbf { I M V } ( F _ { 1 } , F _ { 2 } ) } \\ { \mathbf { I M V } ( F _ { 2 } , F _ { 0 } ) } & { \mathbf { I M V } ( F _ { 2 } , F _ { 1 } ) } & { 0 . 0 } \end{array} \right) .\tag{13}
$$

Each column’s reference weight supplies the denominator, so M is generally asymmetric. Positive entries favor the row model, and reversing a comparison need not negate its IMV. Reporting both $\mathrm { I M V } ( F _ { i } , F ( ) )$ and M separates fit against an uninformative null from directional comparisons among models.

4.4.1 Simulated Ablate-IMV example. Box 4 simulates a binary outcome and fits two neural networks: one with two hidden layers and one with the second hidden layer removed. Both are fitted from scratch on the same training observations. We then use AblationIMV to compare their probabilities on the same test observations.

```python
B 4 Abl IMV
import numpy as np
import pandas as pd
from imvpy import AblationIMV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
rng = np.random.default_rng(42)
X = rng.normal(size=(2000, 3))
p = 1 / (1 + np.exp(-(2 * X[:, 0] + X[:, 1])))
y = rng.binomial(1, p)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, stratify=y, random_state=42)
predictions = {}
variants = {'Full': (8, 8), 'Ablated': (8,)}
for name, layers in variants.items():
model = MLPClassifier(hidden_layer_sizes=layers,
solver='lbfgs', alpha=1, max_iter=2000,
random_state=42).fit(X_train, y_train)
predictions[name] = pd.DataFrame({'True Label': y_test,
'Positive Probability': model.predict_proba(
X_test)[:, 1]})
matrix = AblationIMV.calculate_imv_matrix(predictions)
print(matrix.round(3))
```

Each matrix entry compares the row model with the column model. The negative Full–Ablated entry in this example indicates that the additional layer does not improve held-out fit.

## 5 Experiments

## 5.1 Design and Implementation

5.1.1 Datasets. We examine SHAP-IMV on OpenML Titanic [43], Wisconsin Diagnostic Breast Cancer [44], and Adult Income [45], restricting each specification to eight predictors so that exact enumeration remains tractable. The multiclass experiments use Nursery [46], Car Evaluation [47], and Dry Bean [48] from the UCI Machine Learning Repository [49]. For ablation, the tasks are MNIST odd/even digit classification [50], UCI HAR upstairs/downstairs activity recognition [51], and MAGIC Gamma gamma/hadron classification [52]. The GNN case study uses PROTEINS, NCI1, NCI109, Mutagenicity, AIDS, and DD [53]. Sections 5.2 and 6.1 describe the samples and preprocessing.

5.1.2 Comparison Methods. We compare SHAP-IMV rankings with SHAP [15], LIME [28], and Anchors [29], and report Multi-IMV alongside per-class Brier loss [32], accuracy, precision, and recall [23]. Section 5.3 defines the methods and metrics.

5.1.3 Models and Training. For SHAP-IMV, logistic regression allows up to 5,000 iterations [54]; XGBoost [18] and LightGBM [19] each use 60 boosting iterations, maximum tree depth three, and learning rate 0.2. Exact attribution uses five-fold stratified cross-validation for each of ten seeds (42– 51); we report the mean and sample standard deviation of $\phi _ { j }$ across seeds, with seed-level contributions rounded to three decimals before averaging. We then evaluate the selected topfive subsets using ten-fold stratified cross-validation with the same ten seeds. Standardization uses the full dataset, and feature selection precedes the refit evaluation folds. Multi-IMV evaluates logistic regression, XGBoost, and LightGBM with five-fold stratified cross-validation across ten seeds (42–51).

For Ablate-IMV, the MNIST model is a two-convolution CNN trained for one epoch with Adadelta (learning rate 1.0, batch size 256). UCI HAR uses a two-layer bidirectional GRU with attention, trained for 15 epochs with Adam (learning rate $1 0 ^ { - 3 }$ , batch size 128); MAGIC Gamma uses an MLP with hidden widths 64 and 32, trained for 12 epochs with AdamW (learning rate $1 0 ^ { - 3 } .$ , batch size 256). Each full and ablated model is evaluated for ten seeds (42–51). MNIST and HAR retain their oficial partitions, while MAGIC Gamma uses seed-specific stratified 75/25 splits. The GNN case study uses a three-layer GCN with hidden width 64, dropout 0.5, learning rate 0.001, and batch size 512. Training lasts 200 epochs for each of ten seeds, with seed-specific stratified 70/15/15 training/validation/test splits. We report held-out metrics at every epoch, including the untrained epoch 0.

5.1.4 Software and Hardware. The experiments use imvpy 1.2.0, available on PyPI [38]. The saved GNN manifests record CPU execution with PyTorch 2.9.0 [55], PyTorch Geometric 2.7.0 [56], NumPy 1.26.4, and Scikit-learn 1.7.2 [54]. The runtime benchmark used Python 3.12.7 and an Intel Core i9-14900KF CPU. The notebooks record other dependencies and available details of the runs, as described in our Code and data availability statement.

## 5.2 Datasets and Preprocessing

5.2.1 Datasets for SHAP-IMV. Each binary classification task uses eight specified predictors: passenger characteristics for Titanic, cellular measurements for Breast Cancer, and socioeconomic characteristics for Adult Income.

Titanic The OpenML Titanic dataset (ID 40945, version 1) contains 1,309 passengers, including 500 survivors [43]. We code survival as the positive target outcome and use sex, title extracted from the passenger’s name, ticket class, age multiplied by ticket class, fare, embarkation port, an indicator for travelling without siblings, spouses, parents, or children aboard, and age itself. Titles are grouped into five categories, with a separate code for unmatched names. We impute the 263 missing ages and one missing fare with their medians, assign the two missing embarkation ports to Southampton, and retain all passengers.

Breast Cancer Wisconsin (Diagnostic) The diagnostic dataset contains 569 breast-mass target samples: 212 malignant and 357 benign [44]. Its 30 continuous measurements describe cell nuclei in digitized images of fine-needle aspirates. We select mean radius, texture, smoothness, compactness, symmetry, fractal dimension, concavity, and concave points, and code malignancy as one. All eight inputs are complete, so no observations are removed or imputed.

Adult Income This dataset contains 48,842 records extracted from the 1994 Census database, of which 11,687 have annual income above \$50,000 [45]. The selected inputs are age, educational attainment (education-num), weekly working hours, capital gains, capital losses, and indicators for female sex, married status, and white race. The positive outcome is income above the threshold. We harmonize labels and pool the original training and test files for cross-validation rather than using the oficial test partition. Missing values occur in other source fields, but the eight selected inputs are complete, so all records are retained.

5.2.2 Datasets for Multi-IMV. The multiclass benchmarks difer in their class frequencies. We retain these frequencies within the selected classes for both one-versus-rest and pairwise comparisons.

Nursery This dataset comprises 12,960 cases generated from a hierarchical decision model for ranking nursery-school applications [46] as the target feature. The eight categorical predictors describe parental circumstances, childcare arrangements, family structure, housing, finances, and social and health conditions. We exclude the recommend and very\_recom classes, which contain two and 328 cases, respectively. The retained sample has 12,630 observations: 4,320 not recommended, 4,266 priority, and 4,044 special priority.

Car Evaluation This dataset contains 1,728 configurations generated by a hierarchical acceptability model [47]. Its six categorical predictors describe purchase and maintenance costs, doors, passenger capacity, luggage space, and safety. We retain all four outcome classes: unacceptable (1,210), acceptable (384), good (69), and very good (65).

Dry Bean This dataset contains 13,611 photographed grains represented by 16 size and shape measurements [48]. We retain all seven varieties as classes: Dermason (3,546), Sira (2,636), Seker (2,027), Horoz (1,928), Cali (1,630), Barbunya (1,322), and Bombay (522).

None of the three multiclass datasets requires imputation. Nursery and Car inputs are coded as integer categories, and Dry Bean measurements are standardized. Comparisons between estimators depend on these choices, including the use of ordinal codes for nominal predictors.

The SHAP-IMV and Multi-IMV analyses use five stratified folds across ten seeds. Titanic imputation and SHAP-IMV and Dry Bean standardization are performed before constructing these folds.

5.2.3 Datasetsfor Ablate-IMV. We ablate models for images, sensor sequences, and tabular data, evaluating every variant on the same test observations within each dataset and seed.

MNIST This canonical dataset contains 70,000 grayscale images of handwritten digits, each represented by 28 × 28 pixels [50]. We use OpenML mnist\_784, version 1, preserving the canonical 60,000-image training and 10,000-image test partitions. Odd digits are coded as positive classes, and even ones as negative. Pixels are divided by 255 and normalized using the fixed mean 0.1307 and standard deviation 0.3081. We use this parity task to compare a convolutional network with its architectural ablations.

Human Activity Recognition Using Smartphones UCI HAR records six activities performed by 30 participants carrying waist-mounted smartphones [51]. We retain walking upstairs (one) and downstairs (zero). The supplied participantdisjoint partition yields 2,059 training windows from 21 participants and 891 test windows from nine participants. Each window spans 2.56 seconds and contains 128 readings from nine channels: three axes each of body acceleration, total acceleration, and angular velocity. Channel normalization uses training windows alone. These sequences are used for recurrent-network ablations.

MAGIC Gamma Telescope This dataset contains 19,020 simulated telescope events with ten continuous image measurements and no missing predictors [52]. Gamma events are coded one (12,332 cases) and hadron backgrounds zero (6,688). Stratified 75/25 splits provide 14,265 training and 4,755 test observations per seed, with standardization fitted on the training observations. We hold the multilayer perceptron’s architecture fixed and remove groups of measurements describing geometry, size and concentration, moments, or orientation and distance. The simulated class proportions do not represent gamma-event prevalence in observational telescope data.

## 5.3 Comparison Methods and Metrics

5.3.1 Feature Explanation Methods. For each local explanation method, the estimator is fitted to all eight standardized predictors on the full analysis sample. We sample rows to explain without replacement for each seed, using all rows when the sample is smaller than the explanation budget. Importance scores are averaged over seeds to select five predictors. These methods explain a fixed fitted model, whereas SHAP-IMV refits subsets within cross-validation; the explanation scales and computational budgets also difer.

SHAP SHAP allocates a prediction among features relative to a background distribution [15, 27]. We average absolute contributions over at most 2,000 rows per seed. For logistic regression, LinearExplainer uses interventional feature masking: we supply 200 background rows, which SHAP’s default independent masker subsamples to 100. XGBoost uses its native exact TreeSHAP pred\_contribs interface, excluding the bias column. LightGBM uses TreeExplainer without a supplied background, giving tree-path-dependent perturbation. The resulting contributions are in the models raw-output units rather than IMV units. Their interpretation depends on assumptions about correlated predictors and does not establish causal efects.

LIME LIME fits a distance-weighted local surrogate to model predictions on perturbed inputs [28, 57]. We explain at most 200 rows per seed, using 2,000 perturbations per row and requesting all eight features and the probability of class one. Continuous predictors are discretized into quartiles; binary predictors are declared categorical. We rank features by their mean absolute surrogate coeficient across explained rows. This score measures coeficient magnitude rather than a signed contribution to predictive fit.

Anchors Anchors searches for feature rules that preserve a predicted class under its perturbation distribution [29]. We use Alibi’s AnchorTabular on at most 100 rows per seed, with a target precision of 0.95 and quartile cut points at 25, 50, and 75 percent for all inputs. We retain returned rules even when the search misses its precision target. Failed searches contribute zero counts but remain in the denominator. Because achieved precision and failure counts were not saved, we cannot verify that the returned rules met the target. For feature �, let $c _ { j }$ count its rule inclusions and let $r _ { j }$ sum its reciprocal positions within returned rules (first position one). With � attempted explanations and $c _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { j } c _ { j } .$ , the seed-level ranking score is

$$
s _ { j } = \frac { c _ { j } } { n } + \frac { r _ { j } } { 2 n \operatorname* { m a x } ( c _ { \operatorname* { m a x } } , 1 ) } .
$$

The second term breaks count ties within a seed. Averaging over seeds therefore gives a frequency score adjusted for rule position, rather than a ranking based on frequency alone. The comparisons are descriptive and use unequal explanation budgets, so they cannot establish that one method is uniformly superior. Assessing how well the selection procedures generalize would require nested feature selection and preprocessing fitted within each fold.

## 5.3.2 Multiclass Evaluation Metrics.

Brier Score Loss Brier loss measures squared probability error [32]. For binary outcomes,

$$
{ \mathrm { B r i e r ~ S c o r e } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } ( y _ { i } - { \hat { p } } _ { i } ) ^ { 2 } ,
$$

where � counts instances, $y _ { i } \in \{ 0 , 1 \}$ is the observed label, and $\hat { p _ { i } }$ is its predicted positive-class probability.

Accuracy, Precision, and Recall Accuracy, precision, and recall summarize classification decisions [33, 58]. Let TP and TN count correct positive and negative predictions, and FP and FN count false positives and negatives. Accuracy is the fraction correctly classified:

$$
{ \mathrm { A c c u r a c y } } = { \frac { T P + T N } { T P + T N + F P + F N } } .
$$

Precision is the fraction of positive predictions that are correct:

$$
{ \mathrm { P r e c i s i o n } } = { \frac { T P } { T P + F P } } .
$$

Recall is the fraction of actual positives recovered:

$$
{ \mathrm { R e c a l l } } = { \frac { T P } { T P + F N } } .
$$

## 5.4 SHAP-IMV Results

We compare SHAP-IMV with rankings obtained from mean absolute SHAP values [15], mean absolute LIME weights [28], and anchor inclusion frequencies adjusted for rule position [29]. After averaging importance scores over ten seeds, each method selects five of the eight predictors. We refit the same estimator on each fixed subset using ten-fold stratified cross-validation per seed and report mean accuracy and positive-class precision in Table 1. Since preprocessing and selection use the full dataset before refit evaluation, this comparison describes the selected subsets; it does not provide a nested estimate of how the selection procedures generalize.

The four methods largely agree. SHAP-IMV selects the same five predictors as at least one alternative in five of the nine dataset+estimator settings, and shares at least four of five predictors with every alternative in all but one. Its accuracy is never more than 0.5 percentage points below the best alternative. For the boosting models on Adult Income, SHAP-IMV’s subsets give the highest precision. With equal weight given to each dataset, SHAP-IMV selects the subsets with the highest mean accuracy for XGBoost (86.96%) and LightGBM (86.99%), while LIME leads for logistic regression (86.00%). On Breast Cancer, however, SHAP-IMV subsets are slightly less accurate than SHAP or LIME subsets for every estimator. Methods that select identical sets share the same evaluation results.

The full rankings in Table 2 and seed variability in Figure 2 show where the estimators difer. For Titanic, sex ranks first under logistic regression (0.147), while title leads under both boosting models (0.113 each). All nevertheless select sex, title, class, age multiplied by class, and fare. Radius and concave points occupy the top two positions for Breast Cancer under every estimator. Married status leads for Adult Income, where the fifth selected predictor is weekly working hours for logistic regression and capital loss for boosting. These rankings describe predictive contributions within the specified models and feature sets, and do not imply causal efects.

## 5.5 Multi-IMV Results

Table 3 reports Multi-IMV, Brier loss, and one-vs-rest accuracy, precision, and recall for Car Evaluation. The reference for Multi-IMV is the same estimator fitted to a constant input. Entries are means and sample standard deviations across ten seeds, each evaluated with five stratified folds; global metrics summarize the four-class problem. For the rare Good class, logistic regression achieves 96.01% one-vs-rest accuracy while never predicting the class, giving zero precision and recall. Its Brier loss is 0.04, yet mean Multi-IMV rounds to 0.00. High accuracy and low Brier loss can therefore give a misleading impression when read without the class frequencies or a reference model. XGBoost and LightGBM recover Good with recalls of 89.13% and 95.22%, and their Multi-IMV scores of 0.03 and 0.04 are small in absolute terms, compared with 0.40 and 0.42 for Unacceptable. Relative to the bounds set by class frequency (0.04 for Good, 4.0% of cars; 0.43 for Unacceptable, 70.0%), however, both boosting models attain most of the available improvement for each class. Global support-weighted F1 is 63.84%, 96.69%, and 97.89% for logistic regression, XGBoost, and LightGBM, respectively.

Figure 3 places pairwise heatmaps alongside one-vs-rest bars for all three datasets and estimators. In Dry Bean, the pairwise IMV for Horoz/Seker exceeds that for Bombay/Dermason under every estimator. This contrast largely reflects class balance: Horoz and Seker have similar counts (bound 0.95), whereas Dermason outnumbers Bombay nearly seven to one (bound 0.15), and every estimator comes within 0.003 of both bounds. For the same reason, Dermason, the most frequent class, has the largest one-vs-rest IMV but comparatively low pairwise scores. Dermason/Sira lies furthest below its bound (0.67 of 0.74), identifying the pair the classifiers separate least completely. These scores compare probabilistic fit between class pairs; they are not counts of classification errors and provide a diferent view from the one-vs-rest scores. Heatmaps use a common colour scale across estimators within each dataset. Further per-class metrics appear in Section 5.5.1.

5.5.1 Nursery and Dry Bean Comparisons. The Dry Bean, Car Evaluation, and Nursery results occupy panels a.–f., g.– l., and m.–r. of Figure 3, respectively. Each dataset has a row of pairwise heatmaps and a row of one-vs-rest bars for logistic regression, XGBoost, and LightGBM. All four Car Evaluation classes are retained. For Nursery, the analysis retains not\_recom, priority, and spec\_prior, excluding recommend and very\_recom. Tables 4 and 5 report the additional per-class metrics for Nursery and Dry Bean.

![](images/8e9323842ac7eb322d5fb962dc11c0541b22d96c85eee20da123a48297a79187.jpg)

b.  
![](images/6cf2d66da3db78ef21476af3881db28659305237027fe06b747beecd691cdc26.jpg)

c.  
![](images/70b187ef568a8d0fb3d346a1044ad7192804aaf2f404bc082bfd78dd327266df.jpg)

d.  
![](images/e87bac8603704068c02cab10fbdc48a81484a13f551f08ed18f1b4742fd77d19.jpg)

e.  
![](images/12cdee8d8ab8e3dce3f7ead4ba63afaf1a4fdc5a5ae69f1ca90fcc4f527705f0.jpg)

f.  
![](images/44ce90c18393a12103bd8b391b0530500cae4e5cc5af30519de660e03cae189d.jpg)

g.  
![](images/5907496b199dc0b60e32076090d0994a7c8ef0a8f19462659d403869da2982cc.jpg)

h.  
![](images/8e03b541d680fac414ef098a3ce3869a982e4cbdb82da76b191b0e31c645cbdd.jpg)

![](images/794a1c0eaabcf1d7892eff17cb953091b2c186f15242ee4268c4680a16c4fdfb.jpg)  
Figure 2: SHAP-IMV feature contributions for Titanic (a.–c.), Wisconsin Diagnostic Breast Cancer (d.–f.), and Adult Income (g.–i.). Columns show logistic regression, XGBoost, and LightGBM; each panel ranks eight predictors by mean signed SHAP-IMV. All 256 subsets are evaluated by five-fold stratified cross-validation against the same estimator fitted to a constant input. Shapley contributions use the mean IMV across folds. Bars and labels give means across ten seeds (42–51), with whiskers spanning one sample standard deviation rather than confidence intervals. Standardization uses the full dataset before cross-validation. Y-scales are shared within rows, and the contributions describe predictive fit rather than causal efects.

<table><tr><td rowspan="2" colspan="2">Methods</td><td colspan="2">Titanic</td><td colspan="2">Breast Cancer</td><td colspan="2">Adult Income</td><td colspan="2">Average</td></tr><tr><td>Accuracy</td><td>Precision</td><td>Accuracy</td><td>Precision</td><td>Accuracy</td><td>Precision</td><td>Accuracy</td><td>Precision</td></tr><tr><td rowspan="4">LR</td><td>SHAP-IMV</td><td>79.50</td><td>74.36</td><td>93.92</td><td>94.27</td><td>83.78</td><td>70.94</td><td>85.73</td><td>79.85</td></tr><tr><td>SHAP</td><td>79.74</td><td>73.95</td><td>94.10</td><td>94.46</td><td>83.78</td><td>70.94</td><td>85.87</td><td>79.78</td></tr><tr><td>LIME</td><td>79.74</td><td>73.95</td><td>94.10</td><td>94.46</td><td>84.16</td><td>71.35</td><td>86.00</td><td>79.92</td></tr><tr><td>Anchors</td><td>79.50</td><td>74.36</td><td>93.92</td><td>94.27</td><td>83.78</td><td>70.94</td><td>85.73</td><td>79.85</td></tr><tr><td rowspan="4">XGB</td><td>SHAP-IMV</td><td>81.16</td><td>79.33</td><td>93.76</td><td>93.11</td><td>85.97</td><td>79.42</td><td>86.96</td><td>83.95</td></tr><tr><td>SHAP</td><td>81.16</td><td>79.33</td><td>94.22</td><td>93.34</td><td>85.29</td><td>75.77</td><td>86.89</td><td>82.81</td></tr><tr><td>LIME</td><td>80.44</td><td>80.47</td><td>94.22</td><td>93.34</td><td>85.98</td><td>78.96</td><td>86.88</td><td>84.25</td></tr><tr><td>Anchors</td><td>81.16</td><td>79.33</td><td>94.22</td><td>93.34</td><td>85.29</td><td>75.77</td><td>86.89</td><td>82.81</td></tr><tr><td rowspan="4">LGBM</td><td>SHAP-IMV</td><td>81.00</td><td>79.48</td><td>93.97</td><td>93.41</td><td>86.00</td><td>79.45</td><td>86.99</td><td>84.11</td></tr><tr><td>SHAP</td><td>81.00</td><td>79.48</td><td>94.39</td><td>93.87</td><td>85.30</td><td>75.91</td><td>86.90</td><td>83.09</td></tr><tr><td>LIME</td><td>80.02</td><td>79.34</td><td>94.39</td><td>93.87</td><td>86.00</td><td>79.03</td><td>86.80</td><td>84.08</td></tr><tr><td>Anchors</td><td>81.00</td><td>79.48</td><td>94.39</td><td>93.87</td><td>85.30</td><td>75.91</td><td>86.90</td><td>83.09</td></tr></table>

Table 1: Predictive performance after top-5 feature selection by SHAP-IMV, SHAP, LIME, and Anchors. LR, XGB, and LGBM denote logistic regression, XGBoost, and LightGBM. Each dataset uses a fixed set of 8 variables drawn from the source data; Breast Cancer refers to the Wisconsin Diagnostic dataset. Rankings are averaged over seeds to select a fixed subset for refitting the same estimator. Accuracy and positive-class precision are percentages averaged over 10 stratified folds for each of 10 seeds; Average gives equal weight to the 3 datasets. Preprocessing and feature selection precede refit cross-validation, so the results describe these fixed subsets rather than a nested evaluation of the selection procedures. Within each estimator and column, bold and underlining mark the best and second-best distinct values at the displayed precision; ties share formatting, and columns in which all methods agree are unmarked.

## 5.6 Ablate-IMV Results

Figure 4 compares architectural ablations on MNIST and UCI HAR and feature-group ablations on MAGIC Gamma. In each heatmap, the row is the enhanced model and the column is its reference; the bars instead compare every model with constant 0.5-probability predictions. Variants are evaluated on the same test observations within each seed. Means are taken over ten runs (42–51), and whiskers describe variation across seeds rather than confidence intervals.

5.6.1 MNIST: Architectural Ablation of a CNN. The CNN in panels a. and d. classifies odd/even digits using two convolutions, max pooling, dropout, and a hidden layer. Variants remove the second convolution (NoConv2), hidden layer (NoHidden), or dropout (NoDropout), or replace the convolutional and hidden stack with a linear classifier (Linear). After one epoch, the linear model has a mean IMV of 0.751 against the 0.5 baseline, compared with 0.975 for the full CNN. The other ablations are closer to the full model; removing dropout raises the score slightly to 0.980. Under this training budget, removing a component need not reduce held-out fit.

e.  
![](images/d3b0e5b2d4a27a9b65cfd3061f0662b4247240f1c2eab550cc89cc1a04677221.jpg)

![](images/90febaa6901a30e463be6056062b222b65eb0ffdb07dc2b57d79d5eb001321c2.jpg)

![](images/15594403a4632b2be96e615ca0d2a88a031f26ae8f1723d5a0710bdba26df456.jpg)

d.  
![](images/f916ec1206fc0a4377e6ea353a610eebba26f508f2b386eac14da69ba24f2c4d.jpg)

![](images/1bf7ab9b45098ea75188774394e031d57c43285c561b55440d1068b969dfd4e7.jpg)

![](images/5529ea8a9e76458f30dbc497061d0a3f5b0e6728050603a3861e90810a0087b3.jpg)

![](images/08e409446806c8492c958653b03d003ac551fe06e392f965d4115f0bfd7a097a.jpg)

i.  
![](images/9ebfe2ae5db9d642da09714e4ca62c4eacb7d6de7da837e230b2b39535eedf52.jpg)

![](images/2ad7796b2a26403b062de197bbf0a39060f25a7dc60883173ff21b5d509f7327.jpg)

![](images/68e155bf9756ab211378949da30743ae624ab7d7288c0e2b2998ebae1e011e61.jpg)

![](images/92bf635091fbd2d2c2442858568ac1f5cabe53e224e6dbea5dec07b06a4706d5.jpg)

![](images/adb9069e53b1feb62ca8a5054da58f56ff2b8c03ed272e29e22b1879075d7adf.jpg)

![](images/01978f7495914a09fc9c2632b80cff5c190f5bbc5fa3ca6f4bcb3543c51e4846.jpg)

![](images/6289c8444863a828868ac6d4072d82108d3f29bdf8a86076f46fa00da26bb185.jpg)

![](images/bc8d3d7f969cb30ecb7c4dc12521ef50b79c5ce15e5f3be14c6c3ffad0b35ede.jpg)

p.  
![](images/56da456e3db762625624e84e338c30543ab420c7ac635431b237e6658d622c94.jpg)

q.  
![](images/31238ee1d8c6b3b6279cdaa604148fa5e5d78f8db356af63b403410da194f707.jpg)

![](images/e719709376090e26115a77add48f4561ddaac7e01226db04dd0adec0ec18b0ef.jpg)  
Figure 3: Multi-IMV on Dry Bean (a.–f.; seven classes, 16 features), Car Evaluation (g.–l.; four classes, six features), and Nursery (m.–r.; eight features, retaining not\_recom, priority, and spec\_prior). Columns show logistic regression, XGBoost, and LightGBM, each compared with its constant-input counterpart. Heatmaps restrict held-out observations to class pairs and renormalize both models’ probabilities without binary refitting. Matrices are symmetric with conventional zero diagonals and one colour scale per dataset. Bars show one-vs-rest IMV from unrenormalized class probabilities on all held-out observations; the two views have diferent populations and reference coin weights. Five-fold stratified scores are averaged within each of ten seeds (42–51). Cells, bars, and labels show seed means; whiskers show one sample standard deviation, not confidence intervals. Dashed lines show the ceiling of the One-vs-rest IMV for the targeted category.

<table><tr><td rowspan="2">Dataset Model</td><td rowspan="2"></td><td colspan="8">Rank by mean SHAP-IMV</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td rowspan="3">Titanic</td><td>LR</td><td>Sex (0.147)</td><td>Title (0.065)</td><td>Class (0.041)</td><td>Age × Class (0.035)</td><td>Fare (0.018)</td><td>Embarked (0.013)</td><td>Alone (0.011)</td><td>Age (0.004)</td></tr><tr><td>XGB</td><td>Title (0.113)</td><td>Sex (0.105)</td><td>Fare (0.060)</td><td>Class (0.045)</td><td>Age × Class (0.022)</td><td>Embarked (0.013)</td><td>Alone (0.009)</td><td>Age (-0.000)</td></tr><tr><td>LGBM</td><td>Title (0.113)</td><td>Sex (0.104)</td><td>Fare (0.053)</td><td>Class (0.046)</td><td>Age × Class (0.025)</td><td>Embarked (0.012)</td><td>Alone (0.009)</td><td>Age (0.001)</td></tr><tr><td rowspan="3">Breast Cancer</td><td>LR</td><td>Concave points (0.129)</td><td>Radius (0.124)</td><td>Concavity (0.096)</td><td>Compactness (0.072)</td><td>Texture (0.050)</td><td>Smoothness (0.031)</td><td>Symmetry (0.021)</td><td>Fractal dim. (0.016)</td></tr><tr><td>XGB</td><td>Concave points (0.129)</td><td>Radius (0.128)</td><td>Concavity (0.115)</td><td>Compactness (0.071)</td><td>Texture (0.051)</td><td>Smoothness (0.027)</td><td>Symmetry (0.013)</td><td>Fractal dim. (0.003)</td></tr><tr><td>LGBM</td><td>Radius (0.127)</td><td>Concave points (0.126)</td><td>Concavity (0.113)</td><td>Compactness (0.068)</td><td>Texture (0.050)</td><td>Smoothness (0.028)</td><td>Symmetry (0.014)</td><td>Fractal dim. (0.010)</td></tr><tr><td rowspan="3">Adult Income</td><td>LR</td><td>Married (0.059)</td><td>Education (0.041)</td><td>Capital gain (0.032)</td><td>Age (0.013)</td><td>Hours/week (0.012)</td><td>Sex (female) (0.010)</td><td>Capital loss (0.006)</td><td>Race (white) (0.002)</td></tr><tr><td>XGB</td><td>Married (0.050)</td><td>Capital gain (0.045)</td><td>Education (0.034)</td><td>Age (0.028)</td><td>Capital loss (0.018)</td><td>Hours/week (0.015)</td><td>Sex (female) (0.009)</td><td>Race (white) (0.001)</td></tr><tr><td>LGBM</td><td>Married (0.050)</td><td>Capital gain (0.045)</td><td>Education (0.034)</td><td>Age (0.028)</td><td>Capital loss (0.018)</td><td>Hours/week (0.015)</td><td>Sex (female) (0.009)</td><td>Race (white) (0.001)</td></tr></table>

Table 2: Complete SHAP-IMV feature rankings for the three binary classification examples. LR, XGB, and LGBM denote logistic regression, XGBoost, and LightGBM. Each dataset uses a fixed set of 8 variables drawn from the source data; Breast Cancer refers to the Wisconsin Diagnostic dataset. Predictors are ordered by decreasing mean exact Shapley contribution to model IMV; parentheses give means to three decimals. Attribution uses 5-fold stratified cross-validation for each of 10 seeds. The 10-fold refit evaluation in Table 1 is separate and evaluates the fixed subset in ranks 1–5 for its SHAP-IMV rows. Figure 2 plots the same means with one sample standard deviation across seeds. Rankings describe predictive contributions within each specification, not causal efects.

<table><tr><td>Methods</td><td>Unacc</td><td>Acc</td><td>Good</td><td>Very good</td></tr><tr><td>Multi-IMV (LR)</td><td> ${ \bf 0 . 1 5 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 5 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 2 \pm 0 . 0 0 }$ </td></tr><tr><td>Brier Loss (LR)</td><td> $0 . 1 7 \pm 0 . 0 0$ </td><td> $0 . 1 6 \pm 0 . 0 0$ </td><td> $0 . 0 4 \pm 0 . 0 0$ </td><td> $0 . 0 3 \pm 0 . 0 0$ </td></tr><tr><td>Accuracy (LR)</td><td> $7 2 . 1 6 \pm 0 . 3 1$ </td><td> $7 4 . 3 0 \pm 0 . 3 4$ </td><td> $9 6 . 0 1 \pm 0 . 0 0$ </td><td> $9 5 . 4 8 \pm 0 . 0 8$ </td></tr><tr><td>Precision (LR)</td><td> $7 4 . 5 1 \pm 0 . 1 9$ </td><td> $3 4 . 9 9 \pm 1 . 3 4$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $3 3 . 9 1 \pm 1 . 7 4$ </td></tr><tr><td>Recall (LR)</td><td> $9 1 . 5 8 \pm 0 . 3 3 $ </td><td> $1 8 . 2 3 \pm 0 . 7 8$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $2 1 . 2 3 \pm 1 . 2 1$ </td></tr><tr><td>Multi-IMV (XGB)</td><td> ${ \bf 0 . 4 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 2 5 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 4 \pm 0 . 0 0 }$ </td></tr><tr><td>Brier Loss (XGB)</td><td> $0 . 0 2 \pm 0 . 0 0$ </td><td> $0 . 0 3 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td></tr><tr><td>Accuracy (XGB)</td><td> $9 7 . 8 4 \pm 0 . 2 0 $ </td><td> $9 7 . 0 6 \pm 0 . 2 4 $ </td><td> $9 8 . 8 7 \pm 0 . 2 0 $ </td><td> $9 9 . 5 4 \pm 0 . 1 4$ </td></tr><tr><td>Precision (XGB)</td><td> $9 9 . 3 2 \pm 0 . 2 0 $ </td><td> $9 1 . 9 3 \pm 0 . 8 9$ </td><td> $8 3 . 6 7 \pm 1 . 7 7$ </td><td> $9 1 . 8 5 \pm 2 . 5 0 $ </td></tr><tr><td>Recall (XGB)</td><td> $9 7 . 5 8 \pm 0 . 2 1 $ </td><td> $9 5 . 1 3 \pm 0 . 5 5$ </td><td> $8 9 . 1 3 \pm 4 . 3 3 $ </td><td> $9 6 . 4 6 \pm 1 . 9 3$ </td></tr><tr><td>Multi-IMV (LGBM)</td><td> ${ \bf 0 . 4 2 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 2 7 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 4 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 4 \pm 0 . 0 0 }$ </td></tr><tr><td>Brier Loss (LGBM)</td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 2 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Accuracy (LGBM)</td><td> $9 8 . 6 7 \pm 0 . 2 0 $ </td><td> $9 8 . 1 3 \pm 0 . 1 4$ </td><td> $9 9 . 1 8 \pm 0 . 1 9$ </td><td> $9 9 . 7 3 \pm 0 . 1 0$ </td></tr><tr><td>Precision (LGBM)</td><td> $9 9 . 8 4 \pm 0 . 1 7 $ </td><td> $9 4 . 6 7 \pm 0 . 6 4 $ </td><td> $8 5 . 9 0 \pm 2 . 1 4$ </td><td> $9 5 . 1 0 \pm 1 . 6 6$ </td></tr><tr><td>Recall (LGBM)</td><td> $9 8 . 2 6 \pm 0 . 2 1 $ </td><td> $9 7 . 0 6 \pm 0 . 6 1 $ </td><td> $9 5 . 2 2 \pm 3 . 1 3$ </td><td> $9 8 . 0 0 \pm 1 . 6 3 $ </td></tr></table>

Table 3: One-vs-rest Multi-IMV and classification performance on Car Evaluation. LR, XGB, and LGBM denote logistic regression, XGBoost, and LightGBM. Values are means ± one sample standard deviation across 10 seeds, each using 5-fold stratified cross-validation; these are not confidence intervals. IMV compares the model using the predictors with the same estimator fitted to a constant input only. We average IMV over folds within each seed and calculate the other metrics from pooled out-of-fold predictions. Accuracy, precision, and recall are percentages for each class versus the rest, based on the multiclass argmax prediction; undefined precision or recall is set to zero. Brier Loss is the Brier score loss on class probabilities. Rounding to two decimals can produce a displayed zero standard deviation even when runs difer. Unacc and Acc denote Unacceptable and Acceptable. Global metrics (LR): F1 Score: $6 3 . 8 4 \pm 0 . 3 2 ;$ AUC-ROC: $7 7 . 2 9 \pm 0 . 1 2 ;$ Log Loss: $0 . 6 9 \pm 0 . 0 0$ ; Hamming Loss: 0.31 ± 0.00; Cohen’s Kappa: $1 6 . 3 1 \pm 0 . 7 8 ;$ Jaccard: $5 2 . 4 2 \pm 0 . 3 1$ . Global metrics (XGB): F1 Score: 96.69 ± 0.29; AUC-ROC: 99.71 ± 0.05; Log Loss: $0 . 1 4 \pm 0 . 0 0 $ Hamming Loss: 0.03 ± 0.00; Cohen’s Kappa: 92.79 ± 0.63; Jaccard: $9 3 . 7 6 \pm 0 . 5 1 .$ Global metrics (LGBM): F1 Score: 97.89 ± 0.18; AUC-ROC: 99.90 ± 0.04; Log Loss: $0 . 0 8 \pm 0 . 0 0 ;$ Hamming Loss: $0 . 0 2 \pm 0 . 0 0 ; \mathrm { C }$ ohen’s Kappa: $9 5 . 3 8 \pm 0 . 4 0 ;$ Jaccard: 95.94 ± 0.34. F1, one-vs-rest AUC-ROC, and Jaccard use support-weighted class averages. These three metrics and Cohen’s kappa are multiplied by 100; log loss and Hamming loss are unscaled.

5.6.2 UCI HAR: Recurrent and Temporal Components. Panels b. and e. distinguish upstairs/downstairs activity from smartphone inertial sequences, using the oficial participantdisjoint partition. The full model has two bidirectional GRU layers and temporal attention. Variants use one recurrent direction (UniGRU), one layer (OneLayer), mean pooling instead of attention (NoAttention), or an MLP using only channel means (MeanPoolMLP). The mean-only MLP scores 0.411 against the 0.5 baseline, compared with 0.919 for the full model. Mean pooling over recurrent states scores slightly higher than the full model, at 0.930. The full recurrent model has higher mean fit than the channel-mean MLP, but its attention mechanism gives no gain over recurrent mean pooling under this training budget.

5.6.3 MAGIC Gamma: Feature-Group Ablation. Panels c. and f. classify gamma/hadron events with a fixed MLP hidden architecture. Ablations remove geometry (NoGeometry), size and concentration (NoConcentration), moments (NoM oments), or orientation and distance (NoOrientation). All variants share seed-specific stratified 75/25 splits and standardization fitted on the training data. Removing orientation and distance gives the largest decrease in mean IMV against the 0.5 baseline (0.701 versus 0.815 for all features); removing moments leaves the mean almost unchanged (0.815). The heatmap allows direct comparisons between variants. The apparent usefulness of these feature groups remains conditional on the specified architecture, training budget, and dataset.

<table><tr><td>Methods</td><td>Not recommend</td><td>Priority acceptance</td><td>Special acceptance</td></tr><tr><td>Multi-IMV (LR)</td><td> ${ \bf 0 . 5 2 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 3 1 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 2 8 \pm 0 . 0 0 }$ </td></tr><tr><td>Brier Loss (LR)</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 1 3 \pm 0 . 0 0$ </td><td> $0 . 1 3 \pm 0 . 0 0$ </td></tr><tr><td>Accuracy (LR)</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $7 8 . 7 5 \pm 0 . 0 5$ </td><td> $7 8 . 7 5 \pm 0 . 0 5$ </td></tr><tr><td>Precision (LR)</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $6 8 . 2 0 \pm 0 . 0 7$ </td><td> $6 7 . 1 6 \pm 0 . 1 0$ </td></tr><tr><td>Recall (LR)</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $6 9 . 4 8 \pm 0 . 1 2$ </td><td> $6 5 . 8 2 \pm 0 . 1 0$ </td></tr><tr><td>Multi-IMV (XGB)</td><td> ${ \bf 0 . 5 2 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 4 9 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 4 5 \pm 0 . 0 0 }$ </td></tr><tr><td>Brier Loss (XGB)</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 2 \pm 0 . 0 0$ </td><td> $0 . 0 2 \pm 0 . 0 0$ </td></tr><tr><td>Accuracy (XGB)</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $9 9 . 4 0 \pm 0 . 0 5$ </td><td> $9 9 . 4 0 \pm 0 . 0 5$ </td></tr><tr><td>Precision (XGB)</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $9 9 . 9 2 \pm 0 . 0 3$ </td><td> $9 8 . 2 4 \pm 0 . 1 6$ </td></tr><tr><td>Recall (XGB)</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $9 8 . 3 1 \pm 0 . 1 5$ </td><td> $9 9 . 9 2 \pm 0 . 0 3 $ </td></tr><tr><td>Multi-IMV (LGBM)</td><td> ${ \bf 0 . 5 2 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 4 9 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 4 6 \pm 0 . 0 0 }$ </td></tr><tr><td>Brier Loss (LGBM)</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td></tr><tr><td>Accuracy (LGBM)</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $9 9 . 7 6 \pm 0 . 0 6$ </td><td> $9 9 . 7 6 \pm 0 . 0 6$ </td></tr><tr><td>Precision (LGBM)</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $9 9 . 9 8 \pm 0 . 0 3$ </td><td> $9 9 . 2 9 \pm 0 . 1 7 $ </td></tr><tr><td>Recall (LGBM)</td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $9 9 . 3 2 \pm 0 . 1 7 $ </td><td> $9 9 . 9 8 \pm 0 . 0 3 $ </td></tr></table>

Table 4: Multi-IMV results for Nursery using logistic regression, XGBoost, and LightGBM. Means and sample standard deviations are calculated across ten seeds, each evaluated with five stratified folds. Metric definitions and units follow Table 3. Global metrics (LR): F1 Score: 78.74 ± 0.05; AUC-ROC: $9 1 . 7 7 \pm 0 . 0 1 ;$ Log Loss: $0 . 3 9 \pm 0 . 0 0 ;$ Hamming Loss: 0.21 ± 0.00; Cohen’s Kappa: 68.11 ± 0.08; Jaccard: 67 $. 8 7 \pm 0 . 0 6 .$ . Global metrics (XGB): F1 Score: 99.40 ± 0.05; AUC-ROC: 99.99 ± 0.00; Log Loss: 0.08 ± 0.00; Hamming Loss: 0.01 ± 0.00; Cohen’s Kappa: $9 9 . 1 0 \pm 0 . 0 8 ;$ Jaccard: 98.82 ± 0.10. Global metrics (LGBM): F1 Score: 99.76 ± 0.06; AUC-ROC: 100.00 ± 0.00; Log Loss: 0.06 ± 0.00; Hamming Loss: 0.00 ± 0.00; Cohen’s Kappa: 99.65 ± 0.09; Jaccard: 99.53 ± 0.12.

<table><tr><td>Methods</td><td>Seker</td><td>Barbunya</td><td>Bombay</td><td>Cali</td><td>Dermason</td><td>Horoz</td><td>Sira</td></tr><tr><td>Multi-IMV (LR)</td><td> ${ \bf 0 . 1 7 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 1 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 4 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 1 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 3 2 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 1 6 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 2 1 \pm 0 . 0 0 }$ </td></tr><tr><td>Brier Loss (LR)</td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 3 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 4 \pm 0 . 0 0$ </td></tr><tr><td>Accuracy (LR)</td><td> $9 8 . 4 5 \pm 0 . 0 2 $ </td><td> $9 8 . 5 5 \pm 0 . 0 3 $ </td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $9 8 . 5 3 \pm 0 . 0 1 $ </td><td> $9 5 . 8 3 \pm 0 . 0 3$ </td><td> $9 8 . 7 1 \pm 0 . 0 1$ </td><td> $9 4 . 7 6 \pm 0 . 0 4$ </td></tr><tr><td>Precision (LR)</td><td> $9 4 . 9 3 \pm 0 . 1 0$ </td><td> $9 4 . 2 2 \pm 0 . 1 3 $ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $9 3 . 4 1 \pm 0 . 1 0$ </td><td> $9 2 . 2 8 \pm 0 . 0 6$ </td><td> $9 5 . 4 7 \pm 0 . 1 0$ </td><td> $8 5 . 6 1 \pm 0 . 1 2$ </td></tr><tr><td>Recall (LR)</td><td> $9 4 . 6 8 \pm 0 . 0 8 $ </td><td> $9 0 . 6 3 \pm 0 . 2 2 $ </td><td> $9 9 . 9 6 \pm 0 . 0 8 $ </td><td> $9 4 . 3 6 \pm 0 . 0 6$ </td><td> $9 1 . 6 6 \pm 0 . 0 6$ </td><td> $9 5 . 4 1 \pm 0 . 0 6$ </td><td> $8 7 . 6 9 \pm 0 . 1 2$ </td></tr><tr><td>Multi-IMV (XGB)</td><td> ${ \bf 0 . 1 7 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 1 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 4 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 1 3 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 3 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 1 6 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 2 1 \pm 0 . 0 0 }$ </td></tr><tr><td>Brier Loss (XGB)</td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 3 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 3 \pm 0 . 0 0$ </td></tr><tr><td>Accuracy (XGB)</td><td> $9 8 . 5 6 \pm 0 . 0 3 $ </td><td> $9 8 . 7 3 \pm 0 . 0 6$ </td><td> $9 9 . 9 8 \pm 0 . 0 1 $ </td><td> $9 8 . 6 5 \pm 0 . 0 6$ </td><td> $9 5 . 7 8 \pm 0 . 0 8$ </td><td> $9 8 . 7 4 \pm 0 . 0 3$ </td><td> $9 5 . 0 7 \pm 0 . 0 7$ </td></tr><tr><td>Precision (XGB)</td><td> $9 5 . 5 0 \pm 0 . 1 4$ </td><td> $9 4 . 6 0 \pm 0 . 5 0 $ </td><td> $9 9 . 6 2 \pm 0 . 0 9$ </td><td> $9 4 . 1 2 \pm 0 . 3 1$ </td><td> $9 0 . 9 9 \pm 0 . 1 6$ </td><td> $9 5 . 8 3 \pm 0 . 1 0$ </td><td> $8 7 . 6 7 \pm 0 . 3 1 $ </td></tr><tr><td>Recall (XGB)</td><td> $9 4 . 8 1 \pm 0 . 2 4 $ </td><td> $9 2 . 2 0 \pm 0 . 2 7 $ </td><td> $9 9 . 8 9 \pm 0 . 1 3 $ </td><td> $9 4 . 6 1 \pm 0 . 2 9 $ </td><td> $9 2 . 9 9 \pm 0 . 2 3 $ </td><td> $9 5 . 2 8 \pm 0 . 1 9$ </td><td> $8 6 . 7 1 \pm 0 . 1 5$ </td></tr><tr><td>Multi-IMV (LGBM)</td><td> ${ \bf 0 . 1 7 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 1 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 4 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 1 3 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 3 3 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 1 6 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 2 1 \pm 0 . 0 0 }$ </td></tr><tr><td>Brier Loss (LGBM)</td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 3 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td><td> $0 . 0 4 \pm 0 . 0 0$ </td></tr><tr><td>Accuracy (LGBM)</td><td> $9 8 . 5 5 \pm 0 . 0 3 $ </td><td> $9 8 . 7 5 \pm 0 . 0 4$ </td><td> $9 9 . 9 8 \pm 0 . 0 0$ </td><td> $9 8 . 6 7 \pm 0 . 0 6$ </td><td> $9 5 . 7 6 \pm 0 . 0 7$ </td><td> $9 8 . 7 6 \pm 0 . 0 3$ </td><td> $9 5 . 0 5 \pm 0 . 0 9$ </td></tr><tr><td>Precision (LGBM)</td><td> $9 5 . 3 5 \pm 0 . 2 1 $ </td><td> $9 4 . 6 0 \pm 0 . 2 1 $ </td><td> $9 9 . 7 3 \pm 0 . 1 0$ </td><td> $9 4 . 3 6 \pm 0 . 3 1$ </td><td> $9 1 . 0 2 \pm 0 . 1 9$ </td><td> $9 5 . 9 2 \pm 0 . 2 8 $ </td><td> $8 7 . 5 2 \pm 0 . 3 1 $ </td></tr><tr><td>Recall (LGBM)</td><td> $9 4 . 8 7 \pm 0 . 2 2 $ </td><td> $9 2 . 3 8 \pm 0 . 3 4$ </td><td> $9 9 . 7 7 \pm 0 . 0 8 $ </td><td> $9 4 . 5 3 \pm 0 . 2 6 $ </td><td> $9 2 . 9 0 \pm 0 . 1 6$ </td><td> $9 5 . 2 7 \pm 0 . 1 0$ </td><td> $8 6 . 7 9 \pm 0 . 2 3$ </td></tr></table>

Table 5: Multi-IMV results for Dry Bean using logistic regression, XGBoost, and LightGBM. Means and sample standard deviations are calculated across ten seeds, each evaluated with five stratified folds. Metric definitions and units follow Table 3. Global metrics (LR): F1 Score: 92.43 ± 0.04; AUC-ROC: 99.34 ± 0.00; Log Loss: $0 . 2 1 \pm 0 . 0 0 ;$ Hamming Loss: 0.08 ± 0.00; Cohen’s Kappa: 90.83 ± 0.05; Jaccard: 86.10 ± 0.06. Global metrics (XGB): F1 Score: 92.75 ± 0.11; AUC-ROC: 99.42 ± 0.01; Log Loss: $0 . 1 9 \pm 0 . 0 0 ;$ Hamming Loss: $0 . 0 7 \pm 0 . 0 0 ;$ Cohen’s Kappa: $9 1 . 2 3 \pm 0 . 1 4 ;$ Jaccard: $8 6 . 6 5 \pm 0 . 1 9 .$ Global metrics (LGBM): F1 Score: 92.75 ± 0.11; AUC-ROC: 99.41 ± 0.01; Log Loss: 0.20 ± 0.00; Hamming Loss: 0.07 ± 0.00; Cohen’s Kappa: 91.23 ± 0.13; Jaccard: $8 6 . 6 6 \pm 0 . 1 8 .$

<table><tr><td>Dataset</td><td>Graphs</td><td>Mean nodes</td><td>Mean edges</td></tr><tr><td>PROTEINS</td><td>1,113</td><td>39.06</td><td>72.82</td></tr><tr><td>NCI1</td><td>4,110</td><td>29.87</td><td>32.30</td></tr><tr><td>NCI109</td><td>4,127</td><td>29.68</td><td>32.13</td></tr><tr><td>Mutagenicity</td><td>4,337</td><td>30.32</td><td>30.77</td></tr><tr><td>AIDS</td><td>2,000</td><td>15.69</td><td>16.20</td></tr><tr><td>DD</td><td>1,178</td><td>284.32</td><td>715.66</td></tr></table>

Table 6: Datasets used in the GNN experiments. Mean edge counts refer to undirected edges before GCN self-loops are added.

## 6 IMV and Model Training: A Case Study

## 6.1 Graph Datasets and Preprocessing

We evaluate six binary graph-classification benchmarks drawn from TUDataset [53]; Table 6 reports graph counts and sizes. PROTEINS links secondary-structure elements; DD links nearby amino acids in larger graphs. Both distinguish enzymes from non-enzymes [59, 60]. NCI1 and NCI109 contain molecules screened for activity against non-small-cell lung and ovarian cancer cell lines, respectively [60, 61]. Mutagenicity comprises 2,401 Ames-test mutagens and 1,936 nonmutagens [62]; AIDS comprises 400 HIV-active and 1,600 inactive compounds [63]. Molecular graphs represent atoms as nodes and bonds as edges. Following TU label ordering, class one denotes nonmutagenicity in Mutagenicity and inactivity in AIDS.

PyTorch Geometric loads the original TU releases, retaining all graphs and categorical node labels [56]. Numeric node attributes (PROTEINS: one; AIDS: four) are standardized using training nodes only; bond labels are unused. Each seed defines an approximately 70%/15%/15% stratified random training/validation/test split, without isomorphism cleaning or molecular scafold-based splitting.

## 6.2 Architecture and Training

The GCN comprises three graph-convolution layers with symmetric normalization, self-loops, and ReLU, followed by global mean pooling, dropout, and a linear binary-logit output. Adam minimizes binary cross-entropy without weight decay, learning-rate scheduling, or early stopping. Each optimizer step covers up to 512 graphs through node-count-limited microbatches with sample-weighted gradient accumulation. Validation metrics are recorded but never used for tuning or checkpoint selection; test curves follow the full fixed training run.

a.  
![](images/a3033fc55da6974a31402ffb4dcb04a4b66be84b08bdd31de148bdacfa842698.jpg)

b.  
![](images/f5a417fd4db857c15ac4e0a454a83c5087ee7dad4b3485d9e8ab8b6ebe12156f.jpg)

c.  
![](images/e5955a19aa312033a1890835b747a0afed4ee1e6302aef9b1e61b3405ce1456a.jpg)

d.  
![](images/fb0067f2f3790cc9d030f3752697c7e774509d285f20441325a208ac708f4932.jpg)

e.  
![](images/d497afba4ac7dd0a44f346ef34b045c5c96cadfd14a565a83990191ac0e13a1e.jpg)

![](images/3c52142926ad3391f163bd550eb5907cbf2b1cbf9318ea6fc9aabdb410799a30.jpg)  
Figure 4: Ablate-IMV for MNIST odd/even digits (CNN; a., d.), UCI HAR upstairs/downstairs activity (bidirectional GRU; b., e.), and MAGIC Gamma gamma/hadron events (MLP; c., f.). Heatmaps compare enhanced rows with basic columns; positive values favor rows, and reverse comparisons need not be negatives. Diagonals are zero; heatmaps share a zero-centred colour scale. Bars compare models with constant 0.5 predictions, not the full model shown first. Cells, bars, and labels show ten-seed means (42–51); whiskers show one sample standard deviation, not confidence intervals. MNIST/HAR ablate architecture; MAGIC removes feature groups with hidden architecture fixed. Training lasts one, 15, and 12 epochs, respectively. Oficial MNIST/HAR test sets and stratified 75/25 MAGIC splits align observations across variants within seeds.

![](images/ae57cf0102af081c74221b37df7a08b18b0b89d2ed449c3e1ed66ea0e1ab8a39.jpg)

![](images/bbb30a66c7a2441b7525a8523e2708cac90620cfa8aeaaff70a8e735081c11ff.jpg)  
Figure 5: Test performance during training of a three-layer GCN on a. PROTEINS (1,113 graphs), b. NCI1 (4,110), c. NCI109 (4,127), d. Mutagenicity (4,337), e. AIDS (2,000), and f. DD (1,178). Counts refer to full datasets. Lines show means across ten seeds (42–51): solid for IMV, dashed for accuracy, dash-dotted for precision, and dotted for recall. Shading spans the minimum and maximum across seeds at each epoch, not confidence intervals. Each seed defines a stratified 70/15/15 training/validation/test split; the same test graphs are evaluated from untrained epoch 0 through epoch 200. IMV compares probabilities with the training-prevalence null; classification uses a 0.5 threshold with class 1 positive. Curves are not smoothed, and negative IMV values are retained.

## 6.3 Epoch-Level Results

Figure 5 tracks each seed’s fixed test graphs at every epoch from untrained epoch 0 through 200, using ten seeds (42–51) per dataset. Lines show seed means; shading spans the epoch-wise minimum and maximum. These ranges reflect variation across splits and training runs, not confidence intervals. All epochs and negative IMV values are retained, without smoothing or path simplification; a steady mean can mask variation between runs. IMV compares probabilities with the trainingprevalence null. Accuracy, binary precision, and binary recall use threshold 0.5 with class 1 positive, so IMV can vary without changes in predicted labels or these metrics. Section 7 reports computational costs.

## 7 Computational Complexity Analysis

For the vanilla\_imv call in Box 1, validating � test observations and computing two geometric mean Bernoulli likelihoods take $O ( n )$ time. The default Brent root finder then solves for each likelihood’s equivalent coin weight. If the two scalar searches require $k _ { 0 }$ and $k _ { 1 }$ iterations, total time is $O ( n + k _ { 0 } + k _ { 1 } )$ . At fixed numerical precision, these searches do not operate on the observation vectors, so computation scales linearly with $n .$ . The current vectorised implementation requires $O ( n )$ auxiliary memory for the broadcast baseline and temporary arrays. We extend the simulation in Box 1 to total sample sizes of 1,000, 10,000, 100,000, and 1,000,000. The data-generating process, random seed, logistic regression, and 70/30 training/test split stay the same, with training-set prevalence as the baseline. Timing begins after data generation, model fitting, and test prediction. It covers the complete call vanilla\_imv(p\_baseline, p\_enhanced, y\_test) with default settings, including input validation, baseline broadcasting, likelihood computation, and root finding. Runs are sequential and use one numerical-library thread. After a warm-up call, Python’s timeit calibrates a batch lasting at least 0.2 seconds. We run nine batches of that size on identical inputs, dividing durations by call counts. Table 7 reports the median and interquartile range of per-call times and the median time per test observation. The interquartile range describes timing variation, not IMV uncertainty. The notebook src/simulations/complexity.ipynb generates the table and saves individual batches and software and hardware metadata. Fixed overhead, including the two scalar root searches, raises the cost per observation at small sample sizes. At larger sample sizes, the cost per observation varies little, consistent with the linear dependence on �. These timings cover IMV calculation alone, excluding fitting and prediction, and are not total training times per epoch.

<table><tr><td>n</td><td>IMV</td><td>Median (ms)</td><td>IQR (ms)</td><td>Time/obs. (µs)</td></tr><tr><td>300</td><td>0.282</td><td>0.078</td><td>0.002</td><td>0.260</td></tr><tr><td>3,000</td><td>0.279</td><td>0.124</td><td>0.000</td><td>0.041</td></tr><tr><td>30,000</td><td>0.303</td><td>0.719</td><td>0.010</td><td>0.024</td></tr><tr><td>300,000</td><td>0.303</td><td>6.987</td><td>0.319</td><td>0.023</td></tr></table>

Table 7: Runtime of vanilla\_imv for the simulation in Box 1. The median and interquartile range (IQR) summarize 9 batches of repeated calls; � is the number of test observations. Timing excludes fitting and prediction. imvpy 1.2.0; Python 3.12.7; Intel(R) Core(TM) i9-14900KF; one numerical-library thread.

## 8 Conclusion

SHAP-IMV selected predictor subsets that performed comparably to those chosen by SHAP, LIME, and Anchors, while expressing contributions as held-out gains in fit; its rankings changed with the estimator. Multi-IMV gains difered across classes, and Ablate-IMV showed that removing model components need not reduce fit under our training budgets. The GNN results show how IMV and threshold-based metrics can change diferently during training. These findings depend on the baseline, evaluation design, and coin-mapping domain; they neither guarantee better prediction nor establish causal efects. Exact attribution requires substantial refitting. The imvpy package [38] implements these three comparisons of probabilistic fit against an explicit baseline.

Code and data availability. The IMV\_ML repository provides the notebooks and plotting code under GNU GPL v3. The notebooks retrieve the cited datasets, generate local results, and display validation summaries inline; run\_all.sh runs the pipeline. imvpy is published on PyPI [38], and the analysis pins a Git commit.

## References

[1] Michael I Jordan and Tom M Mitchell. Machine learning: Trends, perspectives, and prospects. Science, 349(6245):255–260, 2015.

[2] Giuseppe Carleo, Ignacio Cirac, Kyle Cranmer, Laurent Daudet, Maria Schuld, Naftali Tishby, Leslie Vogt-Maranto, and Lenka Zdeborová. Machine learning and the physical sciences. Reviews ofModern Physics, 91(4):045002, 2019.

[3] Charles Rahal, Mark Verhagen, and David Kirk. The rise of machine learning in the academic social sciences. AI & SOCIETY, 39(2):799–801, 2024.

[4] Xiaoyun Lin, Xiaowei Du, Shican Wu, Shiyu Zhen, Wei Liu, Chunlei Pei, Peng Zhang, Zhi-Jian Zhao, and Jinlong Gong. Machine learning-assisted dual-atom sites design with interpretable descriptors unifying electrocatalytic reactions. Nature Communications, 15 (1):8169, 2024.

[5] Gavin C Cawley and Nicola LC Talbot. On over-fitting in model selection and subsequent selection bias in performance evaluation. The Journal ofMachine Learning Research, 11:2079–2107, 2010.

[6] Yaoqing Yang, Ryan Theisen, Liam Hodgkinson, Joseph E Gonzalez, Kannan Ramchandran, Charles H Martin, and Michael W Mahoney. Test accuracy vs. generalization gap: Model selection in nlp without accessing training or testing data. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3011–3021, 2023.

[7] C Van Rijsbergen. Information retrieval: theory and practice. In Proceedings ofthe joint IBM/University of Newcastle upon tyne seminar on data base systems, volume 79, pages 1–14, 1979.

[8] Karl Pearson. Vii. mathematical contributions to the theory of evolution.—iii. regression, heredity, and panmixia. Philosophical Transactions of the Royal Society of London. Series A, containing papers of a mathematical or physical character, 187:253–318, 1896.

[9] David E Rumelhart, Geofrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. nature, 323(6088):533–536, 1986.

[10] Hirotugu Akaike. A new look at the statistical model identification. IEEE transactions on automatic control, 19(6):716–723, 1974.

[11] Gideon Schwarz. Estimating the dimension of a model. The annals ofstatistics, pages 461–464, 1978.

[12] Benjamin W. Domingue, Charles Rahal, Jessica Faul, Jeremy Freese, Klint Kanopka, Alexandros Rigos, Ben Stenhaug, and Ajay Shanker Tripathi. The intermodel vigorish (imv) as a flexible and portable approach for quantifying predictive accuracy with binary outcomes. PLOS ONE, 20(3):e0316491, 2025. doi: 10.1371/ journal.pone.0316491.

[13] Benjamin W Domingue, Klint Kanopka, Radhika Kapoor, Stefi Pohl, R Philip Chalmers, Charles Rahal, and Mijke Rhemtulla. The intermodel vigorish as a lens for understanding (and quantifying) the value of item response models for dichotomously coded items. Psychometrika, 89(3):1034–1054, 2024. doi: 10.1007/s11336-024-09977-2.

[14] Lloyd S Shapley. A value for n-person games. Contribution to the Theory ofGames, 2, 1953.

[15] Scott M. Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems, volume 30, pages 4765–4774. Curran Associates, Inc, 2017.

[16] Stefan Knerr, Léon Personnaz, and Gérard Dreyfus. Single-layer learning revisited: a stepwise procedure for building and training a neural network. In Neurocomputing: algorithms, architectures and applications, pages 41–50. Springer, 1990.

[17] David W Hosmer Jr, Stanley Lemeshow, and Rodney X Sturdivant. Applied logistic regression. John Wiley & Sons, 2013.

[18] Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings ofthe 22nd acm sigkdd international conference on knowledge discovery and data mining, pages 785–794, 2016.

[19] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. Lightgbm: A highly eficient gradient boosting decision tree. Advances in neural information processing systems, 30, 2017.

[20] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.

[21] Marina Sokolova and Guy Lapalme. A systematic analysis of performance measures for classification tasks. Information processing & management, 45(4): 427–437, 2009.

[22] Jin Huang and Charles X Ling. Using auc and accuracy in evaluating learning algorithms. IEEE Transactions on knowledge and Data Engineering, 17(3):299–310, 2005.

[23] David MW Powers. Evaluation: from precision, recall and f-measure to roc, informedness, markedness and correlation. arXiv preprint arXiv:2010.16061, 2020.

[24] Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27 (3):379–423, 1948.

[25] J. Ross Quinlan. Induction of decision trees. Machine Learning, 1:81–106, 1986.

[26] Leo Breiman. Random forests. Machine learning, 45: 5–32, 2001.

[27] Erik Štrumbelj and Igor Kononenko. An eficient explanation of individual classifications using game theory. The Journal of Machine Learning Research, 11: 1–18, 2010.

[28] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. " why should i trust you?" explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pages 1135–1144, 2016.

[29] Marco Tulio Ribeiro, Sameer Singh, and Carlos

Guestrin. Anchors: High-precision model-agnostic explanations. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018. doi: 10.1609/aaai.v32i1.11491.

[30] Peter Hase and Mohit Bansal. Evaluating explainable ai: Which algorithmic explanations help users predict model behavior? arXiv preprint arXiv:2005.01831, 2020.

[31] Ian Covert, Scott M. Lundberg, and Su-In Lee. Understanding global feature contributions with additive importance measures. In Advances in Neural Information Processing Systems, volume 33, pages 17212–17223. Curran Associates, Inc., 2020.

[32] Glenn W Brier. Verification of forecasts expressed in terms of probability. Monthly weather review, 78(1): 1–3, 1950.

[33] Evelyn Fix. Discriminatory analysis: nonparametric discrimination, consistency properties, volume 1. USAF school of Aviation Medicine, 1985.

[34] Peter Flach and Meelis Kull. Precision-recall-gain curves: Pr analysis done right. Advances in neural information processing systems, 28, 2015.

[35] James A Hanley and Barbara J McNeil. The meaning and use of the area under a receiver operating characteristic (roc) curve. Radiology, 143(1):29–36, 1982.

[36] Mitchell L Gordon, Kaitlyn Zhou, Kayur Patel, Tatsunori Hashimoto, and Michael S Bernstein. The disagreement deconvolution: Bringing machine learning performance metrics in line with reality. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pages 1–14, 2021.

[37] Lijin Zhang, Charles Rahal, Klint Kanopka, Esther Ulitzsch, Zhiyong Zhang, and Benjamin W. Domingue. Evaluating model predictive performance in confirmatory factor analysis with binary outcomes using the InterModel Vigorish. Multivariate Behavioral Research, pages 1–20, 2026. doi: 10.1080/00273171.2026.2645212. Advance online publication.

[38] imvpy. imvpy: Intermodel vigorish for probabilistic model comparison and attribution. Python package, version 1.2.0, 2026. URL https://pypi.org/project/ imvpy/1.2.0/.

[39] Mikel Galar, Alberto Fernández, Edurne Barrenechea, Humberto Bustince, and Francisco Herrera. An overview of ensemble methods for binary classifiers in multi-class problems: Experimental study on one-vsone and one-vs-all schemes. Pattern Recognition, 44(8): 1761–1776, 2011.

[40] Christopher M. Bishop. Pattern recognition and machine learning. Springer, 2006.

[41] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017.

[42] Kaiming He, Ross Girshick, and Piotr Dollár. Rethinking imagenet pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4918–4927, 2019.

[43] Frank E. Harrell, Jr. and Thomas Cason. Titanic Passenger Data. OpenML, dataset 40945, version 1, 2017. URL https://www.

openml.org/d/40945. Source documentation: https://hbiostat.org/data/repo/titanic.html.

[44] William Wolberg, Olvi Mangasarian, Nick Street, and W. Street. Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository, 1993.

[45] Barry Becker and Ronny Kohavi. Adult. UCI Machine Learning Repository, 1996. DOI: https://doi.org/10.24432/C5XW20.

[46] Vladislav Rajkovic. Nursery. UCI Machine Learning Repository, 1989. DOI: https://doi.org/10.24432/C5P88W.

[47] Marko Bohanec. Car Evaluation. UCI Machine Learning Repository, 1988. DOI: https://doi.org/10.24432/C5JP48.

[48] Dry Bean. UCI Machine Learning Repository, 2020. DOI: https://doi.org/10.24432/C50S4B.

[49] Arthur Asuncion, David Newman, et al. UCI Machine Learning Repository, 2007.

[50] Yann LeCun, Corinna Cortes, and Christopher J. C. Burges. MNIST: Handwritten digit images. OpenML, dataset 554, version 1, 2014. URL https://www. openml.org/d/554.

[51] Jorge Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto, and Xavier Parra. Human Activity Recognition Using Smartphones. UCI Machine Learning Repository, 2013.

[52] R. Bock. MAGIC Gamma Telescope. UCI Machine Learning Repository, 2004.

[53] Christopher Morris, Nils M. Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, and Marion Neumann. TUDataset: A collection of benchmark datasets for learning with graphs. In ICML Workshop on Graph Representation Learning and Beyond, 2020. URL https://arxiv.org/abs/2007.08663.

[54] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in python. the Journal ofmachine Learning research, 12: 2825–2830, 2011.

[55] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

[56] Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

[57] Damien Garreau and Ulrike Luxburg. Explaining the explainer: A first theoretical analysis of lime. In International conference on artificial intelligence and statistics, pages 1287–1296. PMLR, 2020.

[58] Kevin P Murphy. Machine learning: a probabilistic perspective. MIT press, 2012.

[59] Karsten M Borgwardt, Cheng Soon Ong, Stefan Schö- nauer, SVN Vishwanathan, Alex J Smola, and Hans-Peter Kriegel. Protein function prediction via graph kernels. Bioinformatics, 21(suppl\_1):i47–i56, 2005.

[60] Nino Shervashidze, Pascal Schweitzer, Erik Jan van

Leeuwen, Kurt Mehlhorn, and Karsten M. Borgwardt. Weisfeiler–Lehman graph kernels. Journal ofMachine Learning Research, 12(77):2539–2561, 2011. URL https://jmlr.org/papers/v12/shervashidze11a. html.

[61] Nikil Wale, Ian A Watson, and George Karypis. Comparison of descriptor spaces for chemical compound retrieval and classification. Knowledge and Information Systems, 14:347–375, 2008.

[62] Jeroen Kazius, Ross McGuire, and Roberta Bursi. Derivation and validation of toxicophores for mutagenicity prediction. Journal of Medicinal Chemistry, 48 (1):312–320, 2005. doi: 10.1021/jm040835a.

[63] Kaspar Riesen and Horst Bunke. IAM graph database repository for graph based pattern recognition and machine learning. In Structural, Syntactic, and Statistical Pattern Recognition, pages 287–297. Springer, 2008. doi: 10.1007/978-3-540-89689-0\_33.