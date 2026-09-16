# Splitting the Diference: Interpretable Causal Forests for Treatment Efect Heterogeneity and Bias

Nicolas Alexander Ihlo<sup>∗</sup>, Merle Behr<sup>†</sup> Faculty of Informatics and Data Science University of Regensburg, Germany

September 15, 2026

In various fields, such as medicine and marketing, accurately predicting individual treatment efects holds significant promise. However, achieving reliable predictions alone is often insuficient for making informed decisions; it is equally important to understand why the treatment efect is higher for some individuals than for others. To address this two-fold challenge of prediction and interpretation, we introduce an algorithm based on decision trees and random forests for estimating individual treatment efects. Our algorithm is simple: it operates exactly like a standard random forest, but with a diferent splitting criterion, and requires no additional workarounds such as double machine learning or orthogonalization as used in Generalized random forests. It handles observational studies with varying treatment propensities without requiring separate estimation of the full propensity function. This is achieved by combining two splitting criteria—one targeting heterogeneity in the treatment efect, the other targeting bias correction for the average treatment efect—which together improve split point selection and automatically distinguish confounders from features responsible for heterogeneity. As a result, interpretation follows directly from the fitted tree structure itself, that is, from which features the trees split on and with which split statistics, without requiring separate post-hoc analysis. For the theoretical analysis of this algorithm, we consider a change point model with step functions for potential outcomes and treatment propensity and provide insights into the theoretical underpinnings of our approach. Simulation studies show that our

simple algorithm achieves comparable, and often better, prediction accuracy than existing methods, while substantially improving interpretability.

Keywords: causal inference, statistical machine learning, interpretable machine learning, heterogeneous treatment efect estimation

## 1. Introduction

Estimating treatment efects, i.e., changes in an outcome due to some intervention, is of interest across many fields [Manski, 1993, Imbens and Rubin, 2015, Angrist and Pischke, 2015], including medicine, marketing, public policy, and economics. Beyond average efects, many applications benefit from individual-level estimates: for example, estimating the efectiveness of a drug for a specific patient enables individualized medicine. For such estimates to be useful in practice, however, accurate predictions are often not enough by themselves; understanding why the treatment efect difers across individuals, i.e., interpreting the estimation method, is equally important. In this paper, we focus on the estimation of individual treatment efects with a particular emphasis on interpretability.

In many settings, treatment efects cannot be estimated from randomized experiments and one has to rely on observational data instead. This introduces the problem of confounding: features that influence both the outcome and the treatment propensity, and which can bias treatment efect estimates even when they do not afect the treatment efect itself [Hernan and Robins, 2020]. Crucially, this means a feature can appear relevant to treatment efect estimation for two entirely diferent reasons: either because it is a confounder that must be corrected for, or because it genuinely drives treatment efect heterogeneity. Distinguishing between these two cases is precisely the interpretation we are interested in. Throughout this paper, we assume that all confounders are observed, i.e., we rule out hidden confounding, and focus on the estimation of individual treatment efects under observed confounding.

Machine learning (ML) methods are well suited to this task, as they can learn flexible structures directly from data with little manual adjustment. General supervised learning methods, originally developed for classification or regression, can be adapted for treatment efect estimation, for example via metalearners [K¨unzel et al., 2019] and double/debiased machine learning (DML) [Chernozhukov et al., 2018]; other methods are purpose-built for causal inference, building on neural networks, for example TarNet [Shalit et al., 2017], DragonNet [Shi et al., 2019], and RA-Net [Curth and van der Schaar, 2021]. However, many of the most powerful ML methods act as black boxes and ofer little insight into their prediction mechanism, even though such insight is often required for interpretation and downstream decision-making.

Random forest (RF) [Breiman, 2001], originally developed for supervised learning but since extended to causal inference as well, ofers a compromise between these two goals: it achieves state-of-the-art prediction accuracy, particularly on tabular data [Shmuel et al., 2025], while remaining, to a certain extent, interpretable through its underlying tree structure. This structure has, for instance, been used to gain insight into the mechanism of the fitted forest, e.g., by feature importance statistics such as mean decrease in impurity (MDI) [Breiman, 2001], as well as several extensions and other approaches, e.g., MDI+ [Agarwal et al., 2025], iRF [Basu et al., 2018], LSSFind [Behr et al., 2022], and TreeSHAP [Lundberg et al., 2020].

Several RF-type algorithms have been proposed for estimating treatment efects, and it is this line of work that we build on and extend in this paper; we review these approaches in the following subsection. Our main motivation for building on RF, rather than a black-box method such as a neural network, is its interpretability, and accordingly, our goal is to preserve as much of this interpretability as possible in the RF-type algorithm we propose.

## 1.1. Previous Tree-based Methods for Causal Inference

A popular adaptation of RF is causal forest [Wager and Athey, 2018], which itself uses a simplified version of the causal-inference-specific adaptation of decision trees, causal tree, introduced by Athey and Imbens [2016]. While causal forest proposes two possible procedures, we focus here on procedure 1 (double sample trees), as only it incorporates outcomes in the construction of the trees and is therefore able to model treatment efect heterogeneity via the tree structure. Both causal tree and causal forest build trees by selecting splits that maximize the variance of treatment efect predictions, a criterion derived from an analogy to mean squared error (MSE) minimization in regression (see Section 3.1 for details); however, this analogy relies on properties of the regression setting that do not hold in general for causal inference, and the resulting susceptibility to confounding has been noted previously, e.g., by Athey et al. [2019]. Notably, among the tree-based methods discussed below, causal forest is the only one that preserves the simple structure of the original RF: the individual treatment efect is estimated directly as the average prediction of a single ensemble of decision trees, each of which remains interpretable on its own.

Other adaptations of tree-based methods for causal inference typically sacrifice this simplicity. Metalearners [K¨unzel et al., 2019], for instance, do not estimate the treatment efect directly with a single forest, but instead combine separate forests fit to the outcomes under treatment and under control, so that the treatment efect estimate is no longer the direct output of one interpretable ensemble. Generalized random forest (GRF) [Athey et al., 2019], used as an alternative implementation of causal forest, instead modifies the splitting criterion via gradient-based approximations to an estimating equation and a correspondingly changed prediction mechanism, while additionally incorporating an orthogonalization step to reduce the influence of confounders. This orthogonalization step is similar in spirit to DML, reflected in the name “CausalForestDML” used for its implementation in the popular package EconML [Battocchi et al., 2019]. While efective at reducing confounding bias, this two-step procedure again departs from the direct, single-ensemble structure that makes RF interpretable in the first place. A method to improve feature importance for treatment efect heterogeneity in GRF was recently developed by B´enard and Josse [2025], but requires additional post-hoc steps which are computationally costly, and, more importantly, does not address our goal of interpreting the fitted tree structure itself. Further adaptations include orthogonal random forest [Oprescu et al., 2019]; see Jiang et al. [2021] for an overview and comparison of these methods.

<table><tr><td>Sample</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>Treatment  $T$ </td><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td><td>1</td></tr><tr><td>Treated outcome  $Y ^ { T = 1 }$ </td><td>0.5</td><td>0.5</td><td>0.5</td><td>1.5</td><td>1.5</td><td>1.5</td></tr><tr><td>Control outcome  $Y ^ { T = 0 }$ </td><td>0.5</td><td>0.5</td><td>0.5</td><td>1.5</td><td>1.5</td><td>1.5</td></tr><tr><td>Feature  $X _ { 1 }$ </td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Feature  $X _ { 2 }$ </td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td></tr></table>

Table 1: Example for data in a potential outcomes model, with binary treatment $T _ { \ast }$ potential outcomes $Y ^ { T = 1 }$ and $Y ^ { T = 0 }$ , and two binary features $X _ { 1 } , X _ { 2 }$ . Note that for each sample only the outcomes corresponding to the treatment assignment can be observed (shown in bold). The outcomes in italics are unobserved and cannot be used for estimating the treatment efect.

Causal forest is therefore the only method that retains the interpretable, single-ensemble structure of RF for individual treatment efect estimation. However, as noted above, it does not explicitly account for confounders during split selection, so splits driven purely by confounding cannot be distinguished from splits that reflect genuine treatment efect heterogeneity, undermining the very interpretability its tree structure would otherwise ofer. In this paper, we propose a modification of RF for causal inference, denoted as IntCF (interpretable causal forest), that follows the general idea of causal forest but uses a diferent splitting criterion, one that explicitly accounts for confounders at the split selection step. This direct inclusion of both heterogeneity and confounder detection in the tree structure opens the possibility for improved interpretation, especially for distinguishing features based on how they afect the treatment efect prediction, while preserving the simple, single-ensemble structure that makes RF interpretable.

## 1.2. Confounding versus Heterogeneity: an Illustrative Example

In the following, we provide an illustrative example, which demonstrates why the original splitting criterion of causal forest [Wager and Athey, 2018] fails to result in interpretable tree structures under confounding, and how our new splitting criterion in IntCF improves on this. Consider the data in Table 1, with binary treatment $T ,$ potential outcomes $Y ^ { T = 1 }$ and $Y ^ { T = 0 }$ , and two binary features $X _ { 1 } , X _ { 2 }$ . For each sample, only the upright, bold outcome can be observed; the outcome in italics is unobservable. The treatment efect $Y ^ { T = 1 } - Y ^ { T = 0 }$ is zero for every sample. However, directly estimating the average treatment efect from the observed outcomes gives a biased estimate: the diference between the mean observed treated outcome $\textstyle { \frac { \ - 0 . 5 + 1 . 5 + 1 . 5 } { 3 } } = { \frac { 7 } { 6 } }$ and the mean observed control outcome $\textstyle { \frac { 0 . 5 + 0 . 5 + 1 . 5 } { 3 } } = { \frac { 5 } { 6 } } { \mathrm { ~ i s ~ } } { \frac { 1 } { 3 } } \neq 0$ , because of the higher prevalence of treatment among samples with higher potential outcomes.

Splitting the data into two subsets based on feature $X _ { 1 }$ would eliminate this bias, leading to conditional average treatment efect estimates of 0 in both subsets. However, the splitting criterion used in causal forest [Wager and Athey, 2018], which is based on heterogeneity in outcomes, will not select this split, as the estimated outcome is the same on both sides. Even worse, the split on feature $X _ { 2 } ,$ , which results in an estimate of $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ for the treatment efect of samples 1–4, an apparent efect that is itself an artifact of confounding rather than true heterogeneity, would seem preferable.

![](images/8618cdd8210dbde4a5e03ece6170a4edf5b897e211e2600729dbbb23eab00d64.jpg)  
Figure 1: Decision stumps after the first split for the data in Table 1, using the combined splitting criterion of our proposed method, IntCF (left), and the criterion of the original causal forest algorithm [Wager and Athey, 2018] (right). $\hat { \tau }$ denotes the estimated treatment efect in each node, computed as the diference between the mean observed outcome under treatment and under control. IntCF selects the split on $X _ { 1 }$ , correctly identifying it as bias correction rather than heterogeneity, and yields unbiased estimates in both child nodes; causal forest instead selects the split on $X _ { 2 }$ , which appears to indicate heterogeneity but remains biased due to the unaddressed confounding from $X _ { 1 }$

This is in contrast to the modification we propose in IntCF below. There, we explicitly combine two diferent splitting criteria, one for heterogeneity in the treatment efect and one for bias correction. With this approach, the change in predicted average treatment efect from splitting on $X _ { 1 }$ is recognized as bias correction rather than heterogeneity, so the split is still used when building the tree. Interpretability is preserved, since the split is explicitly labeled as reducing bias rather than as improving the estimated treatment efect heterogeneity.

Applying IntCF to this example yields the decision stump shown in Figure 1 (left) after the first split. While IntCF estimates the correct treatment efect in this example, causal forest produces a biased estimate, as the confounding introduced by feature $X _ { 1 }$ is not removed (Figure 1, right). Our combined splitting criterion in IntCF correctly indicates that the selected split corrects bias rather than reflecting treatment efect heterogeneity.

## 1.3. Outline of Paper

The remainder of this paper is organized as follows. In Section 2, we introduce the considered setting, notation, and data model. In Section 3, we introduce the IntCF algorithm, with a particular focus on our new splitting criteria, as well as an additional step for honest predictions and validation of splits. In Section 4, we derive theoretical properties of our new splitting criteria under a change point model. In Section 5, an extensive simulation study, including a semi-synthetic benchmark as well as a real data example, demonstrates that IntCF achieves competitive prediction accuracy while substantially improving interpretability. Finally, in Section 6, we discuss our results and outline open questions.

## 2. Preliminaries

## 2.1. Data and Potential Outcomes Model

We consider a potential outcomes framework [Rubin, 1974], in which the data generating process is described by the joint distribution of

$$
( T , X , Y ^ { T = 0 } , Y ^ { T = 1 } ) ,
$$

with binary treatment $T \in \{ 0 , 1 \}$ , a feature vector $X \in \mathbb { R } ^ { d }$ , and potential outcomes $Y ^ { \mathit { T } = 0 } \mathrm { ~ a n d ~ } \bar { Y } ^ { \mathit { T } = 1 }$ . For each unit, only the potential outcome corresponding to the realized treatment is observed, i.e., we observe $Y \sim Y ^ { T }$ , while the other potential outcome remains unobserved. We aim to predict the conditional average treatment efect (CATE)

$$
\tau ( \mathbf { x } ) : = \mathbb { E } \left[ Y ^ { T = 1 } - Y ^ { T = 0 } \mid \mathbf { X } = \mathbf { x } \right] ,\tag{1}
$$

for a given feature vector $\mathbf { x } \in \mathbb { R } ^ { d } ;$ ; throughout this paper, we refer to $\tau ( \mathbf { x } )$ as the individual treatment efect, since it is the treatment efect estimate available given the observed covariates x. Beyond predicting $\tau ( \mathbf { x } )$ , we are interested in interpreting treatment efect heterogeneity, i.e., in identifying which of the d features actually influence $\tau ( \mathbf { x } )$ . We observe N independent and identically distributed copies of $( T , \mathbf { X } , Y )$ ,

$$
( t _ { i } , \mathbf { x } _ { i } = ( x _ { i , j } ) _ { j = 1 } ^ { d } , y _ { i } ) , \quad i = 1 , \ldots , N .
$$

The main dificulty in estimating $\tau ( \mathbf { x } )$ is that, for each observation i, only one of $y _ { i } ^ { { T = 1 } }$ and $y _ { i } ^ { { T = 0 } }$ is observed; the individual treatment efect

$$
\tau _ { i } : = y _ { i } ^ { T = 1 } - y _ { i } ^ { T = 0 }
$$

can therefore never be observed, so standard machine learning methods for regression, which require direct observations of the target variable, cannot be applied to estimate τ directly.

We further define

$$
p ( \mathbf { x } ) : = P ( T = 1 \mid \mathbf { X } = \mathbf { x } )\tag{2}
$$

for the treatment propensity, and

$$
\mu ( \mathbf { x } ) : = \frac { 1 } { 2 } \mathbb { E } \left[ Y ^ { T = 0 } + Y ^ { T = 1 } \mid \mathbf { X } = \mathbf { x } \right]\tag{3}
$$

for the main outcome.

## 2.2. Decision Trees and RFs

Decision trees for classification and regression tasks were proposed by Breiman et al. [1984] and later extended to RFs [Breiman, 2001]. RFs use a collection of decision trees, whose predictions are combined (for regression, usually by taking the average) to obtain an ensemble prediction. Each tree divides the feature space into subsets by performing binary splits based on a threshold applied to one of the features at each inner node. For deciding which split to use, a splitting criterion is applied, most commonly the impurity decrease according to an impurity measure, for example mean squared error (MSE) for regression.

Specifically, when selecting a split for a node $e _ { \mathrm { p a r e n t } }$ , which contains samples with indices in $J _ { \mathrm { p a r e n t } }$ , a collection of potential splits is considered; this might be all possible splits or a subset of these. Each potential split creates two child nodes, $e _ { L }$ and $e _ { R } .$ containing samples with indices $J _ { L }$ and $J _ { R } ,$ respectively, such that $J _ { \mathrm { p a r e n t } } = J _ { L } \sqcup J _ { R }$ is a disjoint union. We denote the number of samples in the left and right child nodes by $n _ { L } = \left| J _ { L } \right|$ and $n _ { R } = | J _ { R } |$ , respectively. In a regression setting, each training sample has an attached response value $y _ { i }$ . For each of these three nodes, a prediction and an impurity value are computed; comparing these then yields the impurity decrease for the candidate split. Finally, the potential split with the highest impurity decrease is selected as the actual split, and the procedure is repeated recursively at the two child nodes until a stopping criterion is reached.

## 3. Interpretable Causal Tree and Forest Algorithm

## 3.1. Correcting Heterogeneity Splitting Rule for Confounding

As outlined in Section 2.2, a central element for the construction of decision trees from training data is the splitting rule. In this section, we will derive a new splitting rule, which can be applied to data from a potential outcomes model in a causal inference setting. Recall that the main dificulty compared to a regression setting is that for each observation i, only one of $y _ { i } ^ { { T = 1 } }$ and $y _ { i } ^ { { T = } 0 }$ is observed; the individual treatment efect $\tau _ { i } = y _ { i } ^ { T = 1 } - y _ { i } ^ { T = 0 }$ can never be observed directly, hence, standard splitting rules as used for decision trees in regression, cannot be applied directly. Previous work has therefore considered diferent splitting criteria which overcome this problem [see, e.g., Athey and Imbens, 2016, Wager and Athey, 2018, Athey et al., 2019].

To motivate our new splitting criteria, as well as previously considered splitting criteria for causal forest [Wager and Athey, 2018], we start with a recap of the impurity decrease via MSE in the regression setting, the standard split statistic of RF implementations. There, the splitting criterion is obtained by maximizing among possible splits $J _ { \mathrm { p a r e n t } } = J _ { L } \sqcup J _ { R }$ the decrease in MSE, that is,

$$
\Delta ^ { \mathrm { R F } } ( L , R ) : = \frac { 1 } { N } \sum _ { i \in J _ { \mathrm { p a r e n t } } } ( \hat { y } _ { \mathrm { p a r e n t } } - y _ { i } ) ^ { 2 } - \frac { 1 } { N } \left( \sum _ { i \in J _ { L } } ( \hat { y } _ { L } - y _ { i } ) ^ { 2 } + \sum _ { i \in J _ { R } } ( \hat { y } _ { R } - y _ { i } ) ^ { 2 } \right)\tag{4}
$$

with node predictions obtained as averages, that is, $\begin{array} { r } { \widehat { y } _ { \mathrm { p a r e n t } } : = \frac { 1 } { n _ { L } + n _ { R } } \sum _ { i \in J _ { \mathrm { p a r e n t } } } } \end{array}$ y and similar

$$
\hat { y } _ { L } : = \frac { 1 } { n _ { L } } \sum _ { i \in J _ { L } } y _ { i } \quad \mathrm { a n d } \quad \hat { y } _ { R } : = \frac { 1 } { n _ { R } } \sum _ { i \in J _ { R } } y _ { i } .\tag{5}
$$

Note that for such averages we have

$$
{ \hat { y } } _ { \mathrm { p a r e n t } } = { \frac { n _ { L } \cdot { \hat { y } } _ { L } + n _ { R } \cdot { \hat { y } } _ { R } } { n _ { L } + n _ { R } } } .\tag{6}
$$

Using (5), it is easy to see that (4) is equal to

$$
\frac { 1 } { N } \left( n _ { L } ( \hat { y } _ { \mathrm { p a r e n t } } - \hat { y } _ { L } ) ^ { 2 } + n _ { R } ( \hat { y } _ { \mathrm { p a r e n t } } - \hat { y } _ { R } ) ^ { 2 } \right) .\tag{7}
$$

Moreover, using (6), it is easy to see that (7) is equal to

$$
\frac { n _ { L } \cdot n _ { R } } { N ( n _ { L } + n _ { R } ) } \left( \hat { y } _ { L } - \hat { y } _ { R } \right) ^ { 2 } .\tag{8}
$$

Note that (8) equals, up to a factor of $\frac { n _ { L } + n _ { R } } { N }$ (which is constant within each node), the variance among the predictions at the two child nodes.

For the causal inference setting with outcome of interest being the (unobserved) treatment efect $\tau _ { i }$ , the direct analog to the splitting criterion from (4) is to consider

$$
\frac { 1 } { N } \sum _ { i \in J _ { \mathrm { p a r e n t } } } ( \hat { \tau } _ { \mathrm { p a r e n t } } - \tau _ { i } ) ^ { 2 } - \frac { 1 } { N } \left( \sum _ { i \in J _ { L } } ( \hat { \tau } _ { L } - \tau _ { i } ) ^ { 2 } + \sum _ { i \in J _ { R } } ( \hat { \tau } _ { R } - \tau _ { i } ) ^ { 2 } \right) .\tag{9}
$$

For the node predictions $\hat { \tau } _ { \mathrm { p a r e n t } } , \hat { \tau } _ { L }$ and $\scriptstyle { \hat { \tau } } _ { R }$ one can obtain natural estimates with the diference-in-means estimator at a specific node $e ,$ that is,

$$
\hat { \tau } _ { e } : = \frac { 1 } { | \{ i \in J _ { e } : t _ { i } = 1 \} | } \sum _ { \stackrel { i \in J _ { e } } { t _ { i } = 1 } } y _ { i } - \frac { 1 } { | \{ i \in J _ { e } : t _ { i } = 0 \} | } \sum _ { \stackrel { i \in J _ { e } } { t _ { i } = 0 } } y _ { i } ,\tag{10}
$$

$\mathrm { e . g . }$ , for the parent node e = parent, for the left child node $e = L$ , or for right child node $e = R$

The problem with the decrease in MSE as in (9) is, however, that the individual treatment efects $\tau _ { i }$ are not observed and hence, one cannot use (9) for a splitting criterion. Therefore, the proposal of Wager and Athey [2018] for causal forest was to consider the analog of (8) instead. That means, they propose to select splits $J _ { \mathrm { p a r e n t } } = J _ { L } \sqcup J _ { R }$ such that the variance among the predicted treatment efects at the child nodes is maximized, i.e., they maximize at each node

$$
\Delta ^ { \mathrm { C F } } ( L , R ) : = \widehat { \mathrm { H e t } } : = \frac { n _ { L } \cdot n _ { R } } { N ( n _ { L } + n _ { R } ) } ( \widehat { \tau } _ { L } - \widehat { \tau } _ { R } ) ^ { 2 } .\tag{11}
$$

We will call this the heterogeneity criterion. The major problem with using the heterogeneity criterion (11) as a replacement for the decrease in MSE in (9), is that the corresponding analog equations to (5) and (6), which were needed to derive the equivalence between (4) and (8) in the regression setting, do not hold in the causal inference setting. More precisely, the analog to (5) is given by the condition

$$
\hat { \tau } _ { L } = \frac { 1 } { n _ { L } } \sum _ { i \in J _ { L } } \tau _ { i } \quad \mathrm { a n d } \quad \hat { \tau } _ { R } = \frac { 1 } { n _ { R } } \sum _ { i \in J _ { R } } \tau _ { i }\tag{12}
$$

and the analog to (6) is given by the condition

$$
\hat { \tau } _ { \mathrm { p a r e n t } } = \frac { n _ { L } \hat { \tau } _ { L } + n _ { R } \hat { \tau } _ { R } } { n _ { L } + n _ { R } } .\tag{13}
$$

In contrast to the regression setting, (12) and (13) are not guaranteed to hold in the causal inference setting. However, if these conditions do hold, one can derive equivalence between (11) and (9), as the following two lemmas show.

Lemma 1. If (12) holds, then the decrease in MSE (9) equals the mean squared change in prediction

$$
\frac { 1 } { N } \left( n _ { L } ( \hat { \tau } _ { \mathrm { p a r e n t } } - \hat { \tau } _ { L } ) ^ { 2 } + n _ { R } ( \hat { \tau } _ { \mathrm { p a r e n t } } - \hat { \tau } _ { R } ) ^ { 2 } \right) .\tag{14}
$$

The proof is given in the appendix. Note that (14) is the analog to (7) from the regression setting.

Lemma 2. If (13) holds, then the mean squared change in prediction (14) equals the heterogeneity criterion Het d in (11).

The proof is given in the appendix.

Both (12) and (13) may fail to hold under confounding, and either violation breaks the equivalence between the heterogeneity criterion Het d in (11) and the decrease in MSE in (9). However, the two assumptions difer fundamentally in nature. A violation of (13) is a property of the candidate split itself: for any given split, $\hat { \tau } _ { \mathrm { p a r e n t } } , ~ \hat { \tau } _ { L }$ , and $\hat { \tau } _ { R }$ can be computed directly from the observed data, and one can check whether the parent estimate coincides with the sample-size-weighted average of the child estimates. Note that even in a randomized controlled trial, i.e., in the absence of any confounding, (12) and (13) will typically not hold exactly, simply due to finite-sample noise. However, without confounding both conditions hold in expectation. Under confounding, in contrast, they need not hold even in expectation. A strong violation of (13), beyond what would be expected from sampling noise alone, therefore indicates a systematic efect rather than a chance fluctuation and, as illustrated in our example above, arises precisely when treatment propensity difers between the two child nodes, i.e., it is a direct symptom of confounding along the considered split. This makes it possible to explicitly measure and correct for such violations as part of the splitting criterion.

In contrast, (12) is not a property of any particular split, but a standing identification assumption: it requires that, conditional on already being in a given node, the diferencein-means estimator (10) is unbiased for the true average treatment efect within that node, i.e., that no unobserved confounding remains once conditioned on the node. This within-node unconfoundedness assumption underlies the use of the diference-in-means estimator at any node regardless of the tree structure, and is required by essentially every tree-based method for treatment efect estimation, including our own leaf-level estimates. Since it is not tied to any single candidate split, it cannot be assessed or corrected by comparing ${ \hat { \tau } } _ { \mathrm { p a r e n t } } , \ { \hat { \tau } } _ { L }$ , and $\hat { \tau } _ { R }$ at a given step, but only by conditioning on the relevant confounders while growing the tree as a whole. We therefore focus our new splitting criterion on correcting for violations of (13), while retaining (12) as a standing assumption, consistent with its role throughout the tree-based causal inference literature.

As Lemma 2 shows, whenever the mean squared change in prediction (14) and the heterogeneity criterion $\widehat { \mathrm { H e t } }$ in (11) are not equal, this can be directly attributed to a violation of (13) and hence, a bias correction. We therefore treat the diference between (14) and (11) as one component of our splitting criterion, quantifying the extent to which a split corrects for confounding, as opposed to detecting heterogeneity in the treatment efect. The following Lemma gives an explicit expression how this diference corresponds to the violation of (13).

Lemma 3. The diference of (14) and (11) is given by

$$
\widehat { \mathrm { B i a s } } : = \frac { n _ { L } + n _ { R } } { N } \left( \widehat { \tau } _ { \mathrm { p a r e n t } } - \frac { n _ { L } \widehat { \tau } _ { L } + n _ { R } \widehat { \tau } _ { R } } { n _ { L } + n _ { R } } \right) ^ { 2 } .\tag{15}
$$

We call (15) the bias criterion. The proof is given in the appendix.

In total, we have shown the following decomposition

$$
\mathrm { M S E ~ d e c r e a s e } \approx \frac { 1 } { N } \left( n _ { L } ( \hat { \tau } _ { \mathrm { p a r e n t } } - \hat { \tau } _ { L } ) ^ { 2 } + n _ { R } ( \hat { \tau } _ { \mathrm { p a r e n t } } - \hat { \tau } _ { R } ) ^ { 2 } \right) = \widehat { \mathrm { H e t } } + \widehat { \mathrm { B i a s } } .\tag{16}
$$

Note that the decomposition in (16) is analog to a classical bias-variance decomposition. The two criteria, $\widehat { \mathrm { H e t } }$ and Bias d , capture diferent reasons for improved predictions: identified heterogeneity (which we measure by the variance of predictions) and bias correction (which we estimate by the square of change of average prediction).

To improve interpretability of the final trees, we want to separate these two reasons. Therefore, at each node, we select the split that has the strongest efect from either of these two sources. As both the heterogeneity criterion Het d in (11) and the bias criterion Bias d in (15) are directly derived from the mean squared change (14), their values are comparable. Hence, with IntCF we propose to use the maximum of both criteria to select the split. That is, our splitting criterion for IntCF is obtained by maximizing among possible splits $J _ { \mathrm { p a r e n t } } = J _ { L } \sqcup J _ { R }$ the maximum of heterogeneity and bias improvement,

that is,

$$
\Delta ^ { \mathrm { I n t C F } } ( L , R ) : = \operatorname* { m a x } \left( \widehat { \mathrm { H e t } } , \widehat { \mathrm { B i a s } } \right) .\tag{17}
$$

For our theoretical analysis, we also consider signed versions of these criteria, which we refer to as signed splitting criteria, defined by

$$
\widehat { \mathrm { H e t } } ^ { \pm } : = \frac { \sqrt { n _ { L } \cdot n _ { R } } } { n _ { L } + n _ { R } } ( \widehat { \tau } _ { L } - \widehat { \tau } _ { R } ) ,\tag{18}
$$

the signed heterogeneity, and

$$
\widehat { \mathrm { B i a s } } ^ { \pm } : = \widehat { \tau } _ { \mathrm { p a r e n t } } - \frac { n _ { L } \widehat { \tau } _ { L } + n _ { R } \widehat { \tau } _ { R } } { n _ { L } + n _ { R } } ,\tag{19}
$$

the signed bias. By squaring and then multiplying by $\frac { n _ { L } + n _ { R } } { N }$ (a constant factor at each node), one recovers the heterogeneity criterion Het d from the signed heterogeneity $\widehat { \mathrm { H e t } } ^ { \pm }$ ， and the bias criterion Bias from the signed biasd $\widehat { \mathrm { B i a s } } ^ { \pm }$

## 3.2. Honest Validation of Splits

In this section, we introduce a second modification to standard RF that we propose for IntCF, complementing the new splitting criterion (17) introduced in Section 3.1. Importantly, this modification does not afect how the trees themselves are grown: trees are constructed exactly as before, simply using the combined splitting criterion (17) in place of the standard impurity decrease. Instead, this modification concerns how we later summarize, at each node, the heterogeneity and bias contribution identified during tree construction, for example when computing a mean-decrease-in-impurity-type measure of feature importance.

While building the tree, especially for nodes close to the root, the criteria used to select splits might not yet be informative: as discussed in Section 3.1, the within-node unconfoundedness assumption (12) need not hold until the tree has conditioned on the relevant confounders, so for nodes near the root the estimates entering the heterogeneity criterion Het (11) d and the bias criterion Bias (15) d may still be afected by confounding that is only corrected at later splits.

To address this problem, we use out-of-bag (or hold-out) samples to summarize the heterogeneity and bias contribution of each split. This overall idea is not new and has been used in various forms before, both for prediction, where it is known as honest prediction, and for feature importance, e.g., in debiased or out-of-bag variants of MDI [Li et al., 2019, Zhou and Hooker, 2021]. We build on both of these ideas. First, we also use out-of-bag or hold-out samples for our final predictions; this is simply the standard honest-prediction procedure of double-sample trees, as used in causal forest [Wager and Athey, 2018], and not itself a new contribution. Second, and in addition, we use these samples as validation set to re-evaluate the split statistics at every node, similar to out-of-bag MDI approaches for RF. However, simply recomputing the heterogeneity criterion Het (11) d and the bias criterion Bias (15) on out-of-bag samples, using the same diference-in-means estimator d at each node, is not suficient in the causal inference setting: while using held-out data addresses the classical overfitting bias that motivates honest estimation and out-of-bag MDI, it does not address confounding. In particular, the within-node unconfoundedness assumption (12) need not hold until the tree has conditioned on the relevant confounders, regardless of whether the underlying estimate is computed on training or on out-of-bag data. We therefore develop a modified validation procedure in the remainder of this section, which draws on the fully grown tree (or forest) to correct for this remaining confounding.

To use the validation set to also validate split attribution at all nodes, we extend the double-sample-trees procedure to compute predictions from this data set not only at leaf nodes, but also at inner nodes. To obtain both an estimate of corrected bias and of identified heterogeneity, we compute two separate estimates for each node, using the following procedure.

For the first estimate, we feed the validation samples through the considered tree and, at each node, compute an estimate as before using the diference-in-means estimator (10), but now based on validation samples instead of training samples. Denote the indices of validation samples in node e by $I _ { e } ;$ throughout this section, $n _ { e } , n _ { L } , n _ { R } .$ , and N refer to the corresponding counts of validation samples, analogous to their earlier use for training samples in Section 3.1. At node $e ,$ we call this estimate

$$
\hat { \tau } _ { e } ^ { \mathrm { f } } = \frac { 1 } { | \{ i \in I _ { e } : t _ { i } = 1 \} | } \sum _ { \stackrel { i \in I _ { e } } { t _ { i } = 1 } } y _ { i } - \frac { 1 } { | \{ i \in I _ { e } : t _ { i } = 0 \} | } \sum _ { \stackrel { i \in I _ { e } } { t _ { i } = 0 } } y _ { i }\tag{20}
$$

the forward prediction. It might still be influenced by bias that is only corrected at later splits. As with double-sample trees, at leaf nodes these estimates are used as predictions for samples falling in the respective leaf.

For the second estimate, we use the final predictions $\hat { \tau } _ { i } ^ { \mathrm { v a l } }$ for the validation samples (these can be either the predictions from the single tree or from an entire RF). At each node e, a new estimate $\hat { \tau } _ { e } ^ { \mathrm { b } }$ for the CATE can then be computed by averaging the final predictions of validation samples contained in this node, i.e.,

$$
\hat { \tau } _ { e } ^ { \mathrm { b } } = \frac { 1 } { | I _ { e } | } \sum _ { i \in I _ { e } } \hat { \tau } _ { i } ^ { \mathrm { v a l } } .\tag{21}
$$

For inner nodes, this estimate can also be computed as $\begin{array} { r } { \hat { \tau } _ { \mathrm { p a r e n t } } ^ { \mathrm { b } } = \frac { n _ { L } \hat { \tau } _ { L } ^ { \mathrm { b } } + n _ { R } \hat { \tau } _ { R } ^ { \mathrm { b } } } { n _ { L } + n _ { R } } } \end{array}$ . Since these estimates can therefore be computed iteratively from the leaves up to the root, we call them backward predictions. If the final predictions are unbiased, so are the backward predictions.

One task in validating split attribution is to estimate the heterogeneity between treatment efects in the child nodes at each split. For this, we use a formula analogous to the heterogeneity splitting criterion (11), but now using the backward predictions, i.e.,

$$
\widehat { \mathrm { H e t } } ^ { v } : = \frac { n _ { L } \cdot n _ { R } } { N ( n _ { L } + n _ { R } ) } \left( \widehat { \tau } _ { L } ^ { \mathrm { b } } - \widehat { \tau } _ { R } ^ { \mathrm { b } } \right) ^ { 2 } .\tag{22}
$$

We refer to (22) as the heterogeneity validation criterion.

For the bias, such a direct transfer is not possible: unlike the heterogeneity criterion, the backward predictions already incorporate bias corrections made throughout the tree, not only the correction attributable to the split under consideration. Instead, we can estimate the squared bias corrected by the remainder of the tree by comparing forward and backward estimates, using $\begin{array} { r } { \frac { n _ { L } + n _ { R } } { N } \left( \hat { \tau } _ { \mathrm { p a r e n t } } ^ { \mathrm { f } } - \hat { \tau } _ { \mathrm { p a r e n t } } ^ { \mathrm { b } } \right) ^ { 2 } } \end{array}$ . But this also includes bias that was only corrected in descendant nodes. For that reason, we instead use the decrease in squared bias,

$$
\widehat { \mathrm { B i a s } } ^ { v } : = \frac { n _ { L } + n _ { R } } { N } \left( \widehat { \tau } _ { \mathrm { p a r e n t } } ^ { \mathrm { f } } - \widehat { \tau } _ { \mathrm { p a r e n t } } ^ { \mathrm { b } } \right) ^ { 2 } - \frac { n _ { L } } { N } \left( \widehat { \tau } _ { L } ^ { \mathrm { f } } - \widehat { \tau } _ { L } ^ { \mathrm { b } } \right) ^ { 2 } - \frac { n _ { R } } { N } \left( \widehat { \tau } _ { R } ^ { \mathrm { f } } - \widehat { \tau } _ { R } ^ { \mathrm { b } } \right) ^ { 2 } ,\tag{23}
$$

which we refer to as the bias validation criterion.

In each node, one can show that sum of the heterogeneity validation criterion and the bias validation criterion equals the decrease in mean squared error obtained by replacing the forward estimate of the average treatment efect with the final individual predictions, as the following lemma shows.

Lemma 4. For the validation criteria (22) and (23) we have

$$
\widehat { \mathrm { H e t } } ^ { v } + \widehat { \mathrm { B i a s } } ^ { v } = \frac { 1 } { N } \left( \sum _ { i \in I _ { \mathrm { p a r e n t } } } \left( \widehat { \tau } _ { \mathrm { p a r e n t } } ^ { \mathrm { f } } - \widehat { \tau } _ { i } ^ { \mathrm { v a l } } \right) ^ { 2 } - \sum _ { i \in I _ { L } } \left( \widehat { \tau } _ { L } ^ { \mathrm { f } } - \widehat { \tau } _ { i } ^ { \mathrm { v a l } } \right) ^ { 2 } - \sum _ { i \in I _ { R } } \left( \widehat { \tau } _ { R } ^ { \mathrm { f } } - \widehat { \tau } _ { i } ^ { \mathrm { v a l } } \right) ^ { 2 } \right) .
$$

The proof is given in the appendix. Again, note that the decomposition in Lemma 4 is analog to a classical bias-variance decomposition, similar to (16) for the splitting criteria.

## 3.3. Summary of Algorithm

We now combine the two modifications introduced above—the amended splitting criterion $\Delta ^ { \mathrm { I n t C F } }$ (17) from Section 3.1 and the honest validation procedure from Section 3.2—into a complete algorithm for growing decision trees and RFs. Analogous to how a RF is built from individual decision trees (Section 2.2), we first describe how to grow a single tree, which we call the interpretable causal tree (IntCT); the corresponding forest, IntCF, is then obtained by combining many such trees, as described below.

## 3.3.1. Interpretable Causal Tree (IntCT)

1. Split the samples into a training set J and a validation set I.

2. Build the tree based on the training samples J:

a) At the current node, consider all possible splits, determine the resulting sample sets for the two child nodes, and compute the treatment-efect estimates $\hat { \tau } _ { L } , \hat { \tau } _ { R }$ for these child nodes using the diference-in-means estimator (10).

b) For each possible split, compute the heterogeneity criterion Het (11)d and the bias criterion Bias (15).d

c) Select the split that maximizes $\Delta ^ { \mathrm { I n t C F } } = \mathrm { m a x } \left( \widehat { \mathrm { H e t } } , \widehat { \mathrm { B i a s } } \right)$ (17), i.e., the split with the largest value of either criterion.

d) Create the new child nodes and assign them their respective training samples.

e) If no stopping criterion $( \mathrm { e . g . } ,$ , maximum depth or minimum leaf size) has been reached, repeat the above steps at each child node.

3. Validate the splits and compute predictions using the validation samples (see Section 3.2); this step retraces the tree built in Step 2, but now operates on I instead of J:

a) Compute the forward predictions $\hat { \tau } _ { e } ^ { \mathrm { f } } ~ ( 2 0 )$ for all nodes of the tree.

b) At the leaves, use the forward prediction as the tree’s prediction for samples that fall into the respective leaf; this is the standard “honest prediction” procedure, as proposed, e.g., in [Wager and Athey, 2018].

c) Use the tree (or the forest) to compute estimated treatment efects $\hat { \tau } _ { i } ^ { \mathrm { v a l } }$ for all validation samples.

d) Use these estimated treatment efects to compute the backward predictions $\hat { \tau } _ { e } ^ { \mathrm { b } }$ according to (21) for all nodes of the tree.

e) Compute the heterogeneity validation criterion Het d (22) and the bias validation criterion Bias d (23) from the forward and backward predictions, and store their values at each node.

4. The final result is the tree together with its leaf-level predictions and the validation criterion values Hetd , Biasd stored at each split.

## 3.3.2. Interpretable Causal Forest (IntCF)

In a forest, the same algorithm is used to grow each tree, with the training/validation split of Step 1 performed independently for each tree, so that every sample is used, either in the training or in the validation role, in every tree of the forest. This plays a role analogous to the bootstrap sampling used in standard RFs, where the samples not drawn in the bootstrap, the out-of-bag samples, serve as the validation set. As with standard RFs, only a random subset of features is considered at each split when growing each tree.

## 3.3.3. Feature Importance

Following the original mean-decrease-in-impurity approach to feature importance [Breiman et al., 1984], we sum, over all nodes that split on a given feature, a measure of that split’s contribution to obtain the feature’s importance. Instead of the impurity decrease, we use a validation criterion for this purpose: by default, only the heterogeneity validation criterion Hetd (22) is used, but the bias validation criterion Biasd (23), or the sum of both, can also be used if desired.

In many implementations, for example scikit-learn [Pedregosa et al., 2011], it is customary to scale the importance measure so that it sums to 1. In contrast to this standard MDI procedure, the bias validation criterion can take negative values, so feature importance scores that include it may also be negative. We therefore scale our feature importance scores so that the sum of the non-negative scores is 1.

## 4. Population-level Analysis of the Splitting Criteria under a Change Point Model

In this section, we provide theoretical support for the central claim of this paper: that the combined splitting criterion introduced in Section 3.1 correctly distinguishes splits driven by confounding-induced bias from splits that reflect genuine heterogeneity in the treatment efect. Rather than finite-sample guarantees, we study this question at the population level, i.e., in the limit as the sample size $N \to \infty$ , under a stylized change point model in which the treatment efect, the propensity, and the main outcome are step functions of a single covariate.

The remainder of this section is organized as follows. Section 4.1 derives the population limits of the signed heterogeneity criterion $\widehat { \mathrm { H e t } } ^ { \pm }$ and the signed bias criterion Biasd . ± Section 4.2 explains why, due to the marginalization inherent to decision-tree splits, it sufices to study a univariate change point model, which we introduce formally. Section 4.3 then shows that any such change point model decomposes into a component with a constant treatment efect and a component for which the diference-in-means estimator is unbiased. Finally, Section 4.4 shows that our two splitting criteria correctly recover this decomposition: the population bias criterion attains an extremum at the true change point exactly when bias is present, and the population heterogeneity criterion attains an extremum there exactly when the treatment efect is genuinely heterogeneous. This provides theoretical justification for the algorithm’s ability to separate these two sources of improvement, which underlies its interpretability.

## 4.1. Population-level Analysis

Rather than deriving finite-sample results, we focus on the corresponding population analogs of (18) and (19) in order to elucidate general structural properties of these splitting criteria. To this end, consider a decision tree with an inner node e and the associated hyper-rectangle $R ( e )$ . At node e, we examine a candidate split along a given variable, say $x _ { 1 }$ , at threshold k. Throughout the analysis, we impose the following simplifying assumptions:

A1 The feature vector X is uniformly distributed on the unit hypercube, that is, $X \sim U ( [ 0 , 1 ] ^ { d } )$

A2 The outcomes $Y ^ { T = 1 } , Y ^ { T = 0 }$ are bounded.

A3 The positivity assumption holds, meaning that there exists some $\epsilon > 0$ such that, for all $x \in [ 0 , 1 ] ^ { d }$

$$
\epsilon \le P ( T = 1 \mid X = x ) \le 1 - \epsilon .
$$

A4 The hyper-rectangle $R ( e )$ , with positive volume $\mu ( R ( e ) ) > 0$ , the choice of the splitting variable (without loss of generality assumed to be $x _ { 1 } )$ , as well as the considered threshold k are independent of the training data $D = \left\{ ( y _ { 1 } , \mathbf { x } _ { 1 } , t _ { 1 } ) , \dotsc , ( y _ { N } , \mathbf { x } _ { N } , t _ { N } ) \right\}$

Note that, since decision trees are invariant under monotone transformations of the covariates, the first assumption is essentially equivalent to assuming independence among the diferent features. Assumptions A1 and $\mathrm { A 2 }$ are standard in the analysis of decisiontree-based methods [see, for example, Behr et al., 2022, Wager and Athey, 2018]. Assumption A3 is a requirement for causal inference in general [compare, for example, Rosenbaum and Rubin, 1983]. The final assumption is clearly violated for our intCT algorithm, since all split points in the tree are selected in a data-dependent manner, implying that the resulting hyper-rectangle $R ( e )$ likewise depends on the training data $D$ . Nevertheless, extending the subsequent results to this setting appears to be relatively straightforward and mainly of a technical nature; see Remark 1 for details.

Under Assumptions A1–A4, it follows directly from the law of large numbers and the continuous mapping theorem that, when the training data $D$ are generated i.i.d. according to the data-generating process $P ( Y ^ { T = 0 } , Y ^ { T = 1 } , X , T )$ , for any fixed threshold k

$$
\begin{array} { r } { \widehat { \mathrm { H e t } } ^ { \pm } \stackrel { \mathrm { p } } {  } \mathrm { H e t } ^ { \pm } ( k ) , \qquad \widehat { \mathrm { B i a s } } ^ { \pm } \stackrel { \mathrm { p } } {  } \mathrm { B i a s } ^ { \pm } ( k ) , \qquad \mathrm { a s } \ N  \infty . } \end{array}\tag{24}
$$

The corresponding population quantities are given by

$$
\mathrm { H e t } ^ { \pm } ( k ) : = \sqrt { k ( 1 - k ) } \cdot \left[ \left( \bar { Y } _ { L } ^ { T = 1 } - \bar { Y } _ { L } ^ { T = 0 } \right) - \left( \bar { Y } _ { R } ^ { T = 1 } - \bar { Y } _ { R } ^ { T = 0 } \right) \right] ,\tag{25}
$$

and

$$
\mathrm { B i a s } ^ { \pm } ( k ) : = \left( \bar { Y } ^ { T = 1 } - \bar { Y } ^ { T = 0 } \right) - k \cdot \left( \bar { Y } _ { L } ^ { T = 1 } - \bar { Y } _ { L } ^ { T = 0 } \right) - \left( 1 - k \right) \cdot \left( \bar { Y } _ { R } ^ { T = 1 } - \bar { Y } _ { R } ^ { T = 0 } \right)\tag{26}
$$

where, for $a \in \{ 0 , 1 \}$

$$
\begin{array} { r l } & { \bar { Y } ^ { T = a } : = \mathbb { E } \left[ Y ^ { T = a } \mid X \in R ( e ) , T = a \right] , } \\ & { \bar { Y } _ { L } ^ { T = a } : = \mathbb { E } \left[ Y ^ { T = a } \mid X _ { 1 } \leq k , X \in R ( e ) , T = a \right] , } \\ & { \bar { Y } _ { R } ^ { T = a } : = \mathbb { E } \left[ Y ^ { T = a } \mid X _ { 1 } > k , X \in R ( e ) , T = a \right] } \end{array}
$$

are expectations of observed outcomes in the whole node as well as to left and right of the considered split, respectively. In the following, we study the properties of the population counterparts $\operatorname { H e t } ^ { \pm } ( k )$ and $\mathrm { B i a s } ^ { \pm } ( k )$ associated with our proposed splitting criteria.

Remark 1. To dispense with Assumption $^ { 4 , }$ it would sufice to establish the convergence result in (24) uniformly over all hyper-rectangles $R ( e )$ with a fixed minimal volume. More precisely, one would require a result of the form that, for any $\epsilon > 0$

$$
\mathrm { P } [ \operatorname* { s u p } _ { \substack { R , k , j \in \{ 1 , \dots , d \} , \mu ( R ) > \delta } } | \widehat { \mathrm { H e t } } ^ { \pm } - \mathrm { H e t } ^ { \pm } ( k ) | > \epsilon ] \  \ 0 \quad a s \ N  \infty ,
$$

and an analogous statement for the bias term. The assumption of a minimal volume can be enforced in practice by imposing a maximal tree depth and by requiring splitting thresholds k to be bounded away from zero and one; both restrictions are standard in the analysis of RFs [see, $e . g .$ , Scornet and Hooker, 2026]. Establishing such uniform convergence results should be feasible using standard tools from empirical process theory, including uniform convergence arguments and concentration inequalities [see, e.g., Behr et al., 2022]. Since this extension is largely technical and does not afect the qualitative insights of our analysis, we do not pursue it further here and instead focus on the population quantities $\operatorname { H e t } ^ { \pm } ( k )$ and $\mathrm { B i a s } ^ { \pm } ( k )$ in (25) and (26) directly.

## 4.2. Marginalization Efects

By construction, when a decision tree considers a split along a given variable $X _ { j }$ , say $X _ { 1 }$ the influence of the remaining variables $X _ { 2 } , \ldots , X _ { d }$ is marginalized out through averaging, as reflected in the conditional expectations in (25) and (26). Consequently, once a specific variable $X _ { 1 }$ is selected, decision trees can capture only the marginal efect of this variable. This is an intrinsic property of decision trees and gives rise to well-known limitations of such methods.

This marginalization is visible directly in the population criteria themselves: ${ \mathrm { H e t } } ^ { \pm } ( k )$ and $\mathrm { B i a s } ^ { \pm } ( k )$ in (25) and (26) depend on the joint distribution of $( { \bf X } , T , Y ^ { T = 0 } , Y ^ { T = 1 } )$ only through $X _ { 1 }$ and the conditioning event $\mathbf { X } \in R ( t )$ , which is fixed across both candidate children. It therefore sufices, for the purpose of studying the behavior of a single candidate split, to consider a data-generating process in which only one covariate, say $x _ { 1 }$ carries any signal. Here, we study a univariate change point model for CATE, propensity, and main outcome, in analogy to the general definitions of $\tau , p , \mu$ in (1), (2) and (3):

$$
\begin{array} { r l } & { \tau ( x _ { 1 } ) : = \beta _ { 0 } + \beta _ { 1 } \mathbf { 1 } _ { x _ { 1 } \leq \gamma } , } \\ & { p ( x _ { 1 } ) : = q _ { 0 } + q _ { 1 } \mathbf { 1 } _ { x _ { 1 } \leq \gamma } , } \\ & { \mu ( x _ { 1 } ) : = \alpha _ { 0 } + \alpha _ { 1 } \mathbf { 1 } _ { x _ { 1 } \leq \gamma } . } \end{array}\tag{27}
$$

Each coeficient governs a distinct aspect of the model. The slope $\beta _ { 1 }$ governs treatment efect heterogeneity: $\beta _ { 1 } = 0$ corresponds to a constant treatment efect $\tau ( x _ { 1 } ) = \beta _ { 0 }$ , i.e., a model without treatment efect heterogeneity. Analogously, $q _ { 1 }$ governs the degree of confounding through x<sub>1</sub>: $q _ { 1 } = 0$ corresponds to a constant propensity $p ( x _ { 1 } ) = q _ { 0 } , \mathrm { i . e . }$ a setting in which treatment assignment does not depend on $x _ { 1 }$ , as in a randomized trial. Finally, $\alpha _ { 1 }$ governs whether $x _ { 1 }$ has a direct, prognostic efect on the main outcome: $\alpha _ { 1 } = 0$ corresponds to a constant main outcome $\mu ( x _ { 1 } ) = \alpha _ { 0 }$ . Intuitively, confounding bias for the diference-in-means estimator arises when $x _ { 1 }$ afects both treatment assignment $( q _ { 1 } \neq 0 )$ and the potential outcomes; Theorem $5$ below makes this precise.

By construction, this model has a unique natural split point at $\gamma \colon$ splitting exactly there separates $\tau , p ,$ and $\mu$ into constant pieces on either side, which is what allows the resulting heterogeneity and bias to be exactly quantified.

With $\mu$ and $\tau$ specified by the change point model above—now understood as depending on x only through its first coordinate $x _ { 1 } -$ the potential outcomes are defined as

$$
Y ^ { T = 0 } ( \mathbf { x } ) = { \boldsymbol { \mu } } ( \mathbf { x } ) - { \frac { 1 } { 2 } } { \boldsymbol { \tau } } ( \mathbf { x } ) + \varepsilon _ { 0 } { \mathrm { ~ a n d ~ } } Y ^ { T = 1 } ( \mathbf { x } ) = { \boldsymbol { \mu } } ( \mathbf { x } ) + { \frac { 1 } { 2 } } { \boldsymbol { \tau } } ( \mathbf { x } ) + \varepsilon _ { 1 }\tag{28}
$$

with mean-zero noise terms $\varepsilon _ { 0 } , \varepsilon _ { 1 }$ that are assumed to be independent of one another and of X.

## 4.3. Bias–Heterogeneity Decomposition of the Change Point Model

In the following, we provide theoretical insight into why the population splitting criteria He ${ \bf \cdot } ^ { \pm } ( k )$ and $\mathrm { B i a s } ^ { \pm } ( k )$ in (25) and (26) are able to distinguish between heterogeneity in the treatment efect and bias induced by confounding. To this end, we first observe that any change point model of the form (27) can be decomposed into the sum of two components: one corresponding to a model with a constant treatment efect, and another for which the diference-in-means estimator is unbiased. This decomposition is formalized in the following lemma. More precisely, consider a potential outcome model $( Y ^ { T = 0 } , Y ^ { T = 1 } , X , T )$ with treatment efect $\tau ( x )$ , main outcome $\mu ( x )$ , and propensity $p ( x )$ as defined in Section 2.1. We say that the diference-in-means estimator is unbiased if

$$
\mathbb { E } \left[ \tau ( X ) \right] = \mathbb { E } \left[ Y ^ { T = 1 } \mid T = 1 \right] - \mathbb { E } \left[ Y ^ { T = 0 } \mid T = 0 \right] ,\tag{29}
$$

or, equivalently (recall Equation 28), if

$$
\begin{array} { r } { \mathbb { E } \left[ \tau ( X ) \right] = \mathbb { E } \left[ \mu ( X ) + \frac { 1 } { 2 } \tau ( X ) \ | \ T = 1 \right] - \mathbb { E } \left[ \mu ( X ) - \frac { 1 } { 2 } \tau ( X ) \ | \ T = 0 \right] . } \end{array}
$$

Theorem 5. Consider a potential outcome model $( Y ^ { T = 0 } , Y ^ { T = 1 } , X , T )$ with a single uniformly distributed covariate $X \sim U ( [ a , b ] ) , a < b \in \mathbb { R }$ , potential outcomes as in $( 2 8 )$ 2 and main outcome, treatment $e f f e c t ,$ and propensity following a change point model as in (27), i.e.,

$$
\begin{array} { r } { \mu ( x ) = \alpha _ { 0 } + \alpha _ { 1 } \cdot \mathbf { 1 } _ { x \leq \gamma } , \quad \tau ( x ) = \beta _ { 0 } + \beta _ { 1 } \cdot \mathbf { 1 } _ { x \leq \gamma } , \quad p ( x ) = q _ { 0 } + q _ { 1 } \cdot \mathbf { 1 } _ { x \leq \gamma } . } \end{array}
$$

Assume that Assumption A3 holds. Then $\mu ( x )$ and $\tau ( x )$ can be decomposed as

$$
\begin{array} { c c } { { \mu ( x ) = \mu ^ { B } ( x ) + \mu ^ { H } ( x ) } } & { { a n d \quad \tau ( x ) = \tau ^ { B } ( x ) + \tau ^ { H } ( x ) , } } \end{array}
$$

such that, for suitable constants $\alpha _ { 0 } ^ { \prime } , \alpha _ { 1 } ^ { \prime } , \alpha _ { 0 } ^ { \prime \prime } , \alpha _ { 1 } ^ { \prime \prime } , \beta _ { 0 } ^ { \prime } , \beta _ { 0 } ^ { \prime \prime } , \beta _ { 1 } ^ { \prime \prime }$ , the following holds:

1. (Change point model with constant treatment efect)

$$
\mu ^ { B } ( x ) = \alpha _ { 0 } ^ { \prime } + \alpha _ { 1 } ^ { \prime } \cdot { \bf 1 } _ { x \leq \gamma } \ a n d \ \tau ^ { B } ( x ) = \beta _ { 0 } ^ { \prime } .
$$

2. (Change point model with unbiased diference-in-means)

$\mu ^ { H } ( x ) = \alpha _ { 0 } ^ { \prime \prime } + \alpha _ { 1 } ^ { \prime \prime } \cdot { \bf 1 } _ { x \leq \gamma }$ and $\tau ^ { H } ( x ) = \beta _ { 0 } ^ { \prime \prime } + \beta _ { 1 } ^ { \prime \prime } \cdot { \bf 1 } _ { x \leq \gamma }$ , where

$$
\begin{array} { r } { \mathbb { E } \left[ \tau ^ { H } ( X ) \right] = \mathbb { E } \left[ \mu ^ { H } ( X ) + \frac { 1 } { 2 } \tau ^ { H } ( X ) \ | \ T = 1 \right] - \mathbb { E } \left[ \mu ^ { H } ( X ) - \frac { 1 } { 2 } \tau ^ { H } ( X ) \ | \ T = 0 \right] } \end{array}\tag{30}
$$

The proof is given in the appendix.

Remark 2. Note that this decomposition is not unique. The condition (30) does not restrict $\alpha _ { 0 } ^ { \prime }$ and $\alpha _ { 0 } ^ { \prime \prime }$ as well as $\beta _ { 0 } ^ { \prime }$ and $\beta _ { 0 } ^ { \prime \prime }$ , so that $\alpha _ { 0 }$ and $\beta _ { 0 }$ can be freely distributed onto the respective pair of parameters. If the treatment assignment is randomized, $i . e . , p ( x )$ constant in [0, 1], any parameters will fulfill (30), so the same freedom of choice also extends to $\alpha _ { 1 }$

![](images/23fd8e30c78ba39c10d1d0686f88586bfade661d08adfeae1a597fab4896102d.jpg)  
Figure 2: Illustration of the decomposition of a change point model into a component with constant treatment efect and a component with unbiased diference-in-means, as in Theorem 5. The original model was defined by $\mu ( x ) = 0 + 2 \cdot { \bf 1 } _ { x \leq 0 . 6 } , \tau ( x ) =$ $1 + 3 \cdot 1 _ { x \leq 0 . 6 }$ and $p ( x ) = 0 . 2 5 + 0 . 5 \cdot \mathbf { 1 } _ { x \leq 0 . 6 }$

Intuitively, this decomposition isolates exactly the two reasons a split can improve the estimated treatment efect (cf. Section 4.2): the bias component $( \mu ^ { B } , \tau ^ { B } )$ has, by construction, a constant treatment efect, so any change in the estimated treatment efect attributable to this component must be due to bias correction rather than genuine heterogeneity. Conversely, the heterogeneity component $( \mu ^ { H } , \tau ^ { H } )$ satisfies the unbiasedness condition (30) by construction, so any change in its estimated treatment efect must reflect genuine heterogeneity rather than bias correction. We make this precise in Section 4.4 below. Figure 2 illustrates such a decomposition as described in Theorem 5.

## 4.4. Identifying Heterogeneity and Bias with Signed Splitting Criteria

By Theorem 5 and the linearity of expectation, it follows directly that the population splitting criteria $\operatorname { H e t } ^ { \pm } ( k )$ and $\mathrm { B i a s } ^ { \pm } ( k )$ in (25) and (26) can be decomposed accordingly.

To make this explicit, we first rewrite $\operatorname { H e t } ^ { \pm } ( k )$ and $\mathrm { B i a s } ^ { \pm } ( k )$ in terms of the treatment efect function $\tau ( x )$ and the main outcome function $\mu ( x )$ as follows:

$$
\begin{array} { r l } & { \mathrm { H e t } ^ { \pm } ( k ) = \sqrt { k ( 1 - k ) } \cdot \Big [ \big ( \mathbb { E } \left[ \mu ( X ) + \frac { 1 } { 2 } \tau ( X ) \mid X _ { 1 } \leq k , X \in R ( e ) , T = 1 \right] } \\ & { \qquad - \mathbb { E } \left[ \mu ( X ) - \frac { 1 } { 2 } \tau ( X ) \mid X _ { 1 } \leq k , X \in R ( e ) , T = 0 \right] \big ) } \\ & { \qquad - \big ( \mathbb { E } \left[ \mu ( X ) + \frac { 1 } { 2 } \tau ( X ) \mid X _ { 1 } > k , X \in R ( e ) , T = 1 \right] } \\ & { \qquad - \mathbb { E } \left[ \mu ( X ) - \frac { 1 } { 2 } \tau ( X ) \mid X _ { 1 } > k , X \in R ( e ) , T = 0 \right] \big ) \Big ] . } \end{array}\tag{31}
$$

and

$$
\begin{array} { r l } & { \mathrm { B i a s } ^ { \pm } ( k ) = \mathbb { E } \left[ \mu ( X ) + \frac { 1 } { 2 } \tau ( X ) \mid X \in { \cal R } ( e ) , T = 1 \right] } \\ & { \qquad - \mathbb { E } \left[ \mu ( X ) - \frac { 1 } { 2 } \tau ( X ) \mid X \in { \cal R } ( e ) , T = 0 \right] } \\ & { \qquad - k \cdot \left[ \mathbb { E } \left[ \mu ( X ) + \frac { 1 } { 2 } \tau ( X ) \mid X _ { 1 } \leq k , X \in { \cal R } ( e ) , T = 1 \right] \right. } \\ & { \qquad \left. - \mathbb { E } \left[ \mu ( X ) - \frac { 1 } { 2 } \tau ( X ) \mid X _ { 1 } \leq k , X \in { \cal R } ( e ) , T = 0 \right] \right] } \\ & { \qquad - \left( 1 - k \right) \cdot \left[ \mathbb { E } \left[ \mu ( X ) + \frac { 1 } { 2 } \tau ( X ) \mid X _ { 1 } > k , X \in { \cal R } ( e ) , T = 1 \right] \right. } \\ & { \qquad \quad \left. - \mathbb { E } \left[ \mu ( X ) - \frac { 1 } { 2 } \tau ( X ) \mid X _ { 1 } > k , X \in { \cal R } ( e ) , T = 0 \right] \right] . } \end{array}\tag{32}
$$

Consequently, under the assumptions of Theorem $5 ,$ we obtain the decomposition

$$
\mathrm { H e t } ^ { \pm } ( k ) = \mathrm { H e t } _ { B } ^ { \pm } ( k ) + \mathrm { H e t } _ { H } ^ { \pm } ( k ) \quad \mathrm { a n d } \quad \mathrm { B i a s } ^ { \pm } ( k ) = \mathrm { B i a s } _ { B } ^ { \pm } ( k ) + \mathrm { B i a s } _ { H } ^ { \pm } ( k ) ,\tag{33}
$$

where $\operatorname { H e t } _ { B } ^ { \pm } ( k )$ is defined analogously to (31), with $\mu$ and τ replaced by $\mu ^ { B }$ and $\tau ^ { B }$ as in Theorem $5 ,$ and where $\operatorname { H e t } _ { H } ^ { \pm } ( k )$ , $\mathrm { B i a s } _ { B } ^ { \pm } ( k )$ , and $\mathrm { B i a s } _ { H } ^ { \pm } ( k )$ are defined in complete analogy. Note that, while there is some freedom in the choice of decomposition in Theorem $5 ,$ any such decomposition results in the same values for $\operatorname { H e t } _ { B } ^ { \pm } ( k )$ , $\operatorname { H e t } _ { H } ^ { \pm } ( k )$ $\mathrm { B i a s } _ { B } ^ { \pm } ( k )$ , and $\mathrm { B i a s } _ { H } ^ { \pm } ( k )$ . An illustrative example of this decomposition for both the bias and heterogeneity splitting criteria is shown in Figure 3.

Theorem 6. Consider a potential outcome model $( Y ^ { T = 0 } , Y ^ { T = 1 } , X , T )$ with a single uniformly distributed covariate $X \sim U ( [ a , b ] ) , a < b \in \mathbb { R }$ , potential outcomes as in $( 2 8 )$ ， and main outcome, treatment $e f f e c t ,$ , and propensity following a change point model as in (27), i.e.,

$$
\mu ( x ) = \alpha _ { 0 } + \alpha _ { 1 } \cdot { \bf 1 } _ { x \leq \gamma } , \quad \tau ( x ) = \beta _ { 0 } + \beta _ { 1 } \cdot { \bf 1 } _ { x \leq \gamma } , \quad p ( x ) = q _ { 0 } + q _ { 1 } \cdot { \bf 1 } _ { x \leq \gamma } .
$$

Fix a decomposition as in Theorem $^ { 5 , }$ and define Het $\displaystyle { ; _ { B } ^ { \pm } ( k ) }$ $\operatorname { H e t } _ { H } ^ { \pm } ( k )$ , Bias $\mathinner { ! } _ { B } ^ { \pm } ( k )$ , and Bias $\mathfrak { s } _ { H } ^ { \pm } ( k )$ as in (33). Then, under Assumption A3, the following statements hold:

1. If the diference-in-means estimate is biased as in (29), i.e.,

$$
\mathbb { E } \left[ \tau ( X ) \right] \neq \mathbb { E } \left[ Y ^ { T = 1 } \mid T = 1 \right] - \mathbb { E } \left[ Y ^ { T = 0 } \mid T = 0 \right] ,
$$

then $\mathrm { B i a s } _ { B } ^ { \pm } ( k )$ attains a global extremum at $k = \gamma$ and $\mathrm { H e t } _ { B } ^ { \pm } ( \gamma ) = 0$

![](images/f218f209b9073bfbce6df0bf4333b87d20d9e6e57ba5c318344446a9fe1a92d6.jpg)  
Figure 3: The signed heterogeneity and bias criteria, $\operatorname { H e t } ^ { \pm } ( k )$ and $\mathrm { B i a s } ^ { \pm } ( k )$ , together with their decomposition into Het $\begin{array} { r } { \frac { \pm } { B } ( k ) + \mathrm { H e t } _ { H } ^ { \pm } ( k ) } \end{array}$ and $\mathrm { B i a s } _ { B } ^ { \pm } ( k ) + \mathrm { B i a s } _ { H } ^ { \pm } ( k )$ as in (33), for the model shown in Figure 2.

2. If the treatment efect $\tau ( x )$ is non-constant, then $\operatorname { H e t } _ { H } ^ { \pm } ( k )$ attains a local extremum at $k = \gamma$ and $\mathrm { B i a s } _ { H } ^ { \pm } ( \gamma ) = 0$

The proof is given in the appendix.

Remark 3. By the relationship between signed and unsigned splitting criteria, as well as the details of the proof of Theorem 6, the extrema in Theorem 6 of the signed splitting criteria correspond to maxima of their unsigned counterparts while zeros are preserved. As the splitting criteria are non-negative, the zeros correspond to minima of the splitting criteria.

The interpretation of Theorem 6 is as follows. Theorem 5 shows that, locally at any fixed node, the potential outcome model marginalized with respect to a given splitting variable can be decomposed into two distinct components: one component that captures all heterogeneity in the treatment efect, and another component that captures all bias of the local diference-in-means estimator. This decomposition is intrinsic to the marginalization mechanism of decision trees and holds at the population level for each candidate split. As a consequence, a split at a given node may improve the local estimation of heterogeneous treatment efects for two fundamentally diferent reasons: either because it captures genuine heterogeneity in the treatment efect function, or because it reduces bias in the local average treatment efect induced by confounding. Standard decision tree–based approaches do not distinguish between these two sources of improvement, as they rely only on the heterogeneity criterion when selecting splits.

Theorem 6 shows that this separation can be achieved at the population level by the proposed splitting criteria. More precisely, the heterogeneity component of the signed heterogeneity criterion, He $; _ { H } ^ { \pm } ( k )$ , attains an extremum at the true jump location γ whenever the treatment efect is non-constant, while the bias component of the signed bias criterion, $\mathrm { B i a s } _ { B } ^ { \pm } ( k )$ is maximized at the same location whenever bias is present. The behavior predicted by Theorem 6 can also be observed empirically in the illustrative example shown in Figure 3. Thus, the population versions of our splitting criteria $\operatorname { H e t } ^ { \pm } ( k )$ and $\mathrm { B i a s } ^ { \pm } ( k )$ in (25) and (26) recover the split positions associated with treatment efect heterogeneity and bias, respectively, within the appropriate model components, with the former guaranteed up to a local optimum. Overall, this result provides a theoretical explanation for the empirical improvements observed for the proposed splitting criteria in our simulation studies, presented in the next section.

## 5. Simulations and Application

In this section, we apply IntCF on semi-synthetic and synthetic data sets, which allow for evaluation of prediction accuracy by having known ground-truth causal efects, as well as on a real data example. For synthetic data sets we are further able to compare importance measures to the relation of features to outcomes defined by the data model. Throughout, we pay particular attention to the trade-of between prediction accuracy, model simplicity, and interpretability, since this trade-of is the central motivation for our approach.

## 5.1. Prediction Accuracy on Real-data Benchmarks

First, we compare prediction accuracy of our method with other established tree-based treatment efect estimation methods on four established data sets using the framework of Balazadeh et al. [2025]. As tree-based competitors we consider generalized random forests (GRF) [Athey et al., 2019], CausalForestDML as implemented in the EconML package [Battocchi et al., 2019], and causal forest [Wager and Athey, 2018]. These models were employed to gain predictions for average treatment efect (ATE) and CATE on four data sets with ground-truth causal efects: IHDP [Ramey et al., 1992, Hill, 2011], ACIC 2016 [Dorie et al., 2019], Lalonde CPS and PSID cohorts [LaLonde, 1986] with causal efects from RealCause [Neal et al., 2021]. The predictions were evaluated against ground-truth ATE and CATE using relative ATE error and precision in estimation of heterogeneous treatment efects (PEHE) [see Balazadeh et al., 2025].

<table><tr><td rowspan="2">Method</td><td colspan="4">Mean PEHE ± Standard Error (↓ better)</td><td colspan="4">Mean ATE Relative Error ± Standard Error (↓ better)</td></tr><tr><td>IHDP</td><td>ACIC 2016 Lalonde cps</td><td>(×103)</td><td>Lalonde PSID (×103)</td><td>IHDP</td><td>ACIC 2016</td><td>Lalonde cPs</td><td>Lalonde PSID</td></tr><tr><td>IntCF</td><td>2.99±0.49</td><td>1.57±0.10</td><td>9.99±0.06</td><td>15.93±0.27</td><td>0.21±0.04</td><td>0.15±0.04</td><td>0.39±0.02</td><td>0.18±0.02</td></tr><tr><td>causal forest</td><td>3.72±0.61</td><td>2.19±0.12</td><td>9.58±0.03</td><td>16.02±0.17</td><td>0.22±0.04</td><td>0.28±0.06</td><td>0.24±0.01</td><td>0.09±0.01</td></tr><tr><td>GRF</td><td>3.67±0.61</td><td>1.32±0.30</td><td>12.33±0.06</td><td>22.91±0.17</td><td>0.18±0.03</td><td>0.07±0.02</td><td>0.82±0.02</td><td>0.85±0.02</td></tr><tr><td>Forest DML</td><td>4.53±0.73</td><td>1.48±0.31</td><td>12.95±0.04</td><td>22.99±0.15</td><td>0.08±0.01</td><td>0.05±0.01</td><td>1.03±0.01</td><td>1.05±0.01</td></tr></table>

Table 2: CATE & ATE results. Columns correspond to benchmark suites: IHDP, ACIC 2016, Lalonde CPS/PSID. (left half) mean PEHE, (right half) mean ATE relative error. Lalonde PEHE is in thousands. The best and second best entries in each column are highlighted. This table is modified from Balazadeh et al. [2025], with results for IntCF in the first line, results for causal forest [Wager and Athey, 2018] in the second line and the results for GRF and Forest DML of Balazadeh et al. [2025] as comparison in the remaining two lines.

Results are shown in Table 2. Overall, IntCF is mostly comparable with causal forest, with some improvement on data sets IHDP and ACIC 2016, while causal forest produces better ATE estimates on the Lalonde data sets. The results for the other two methods follow a diferent pattern: they perform better in estimating ATE for data sets IHDP and ACIC 2016, but their results for PEHE and ATE on the Lalonde data sets are significantly worse than those of IntCF. Overall, IntCF performs best in two settings, is second best in three settings, and third out of four in the remaining three settings, so that we conclude that IntCF is very competitive in terms of prediction accuracy compared to other established tree-based competitors.

Beyond the tree-based competitors considered here, [Balazadeh et al., 2025] also benchmark several additional, non-tree-based methods on the same four data sets, using the identical evaluation setup we adopt here; we refer to Balazadeh et al. [2025] for the full comparison. Some of these methods, most notably CausalPFN, achieve higher prediction accuracy than any of the tree-based methods considered in Table 2, including IntCF, in most settings. However, CausalPFN and related methods are complete black boxes: unlike tree-based approaches, they ofer no direct way to inspect which features drive a given prediction, or to what extent, and hence cannot answer the type of interpretability questions that motivate this paper. Tree-based methods therefore occupy a useful middle ground between prediction accuracy and interpretability. Within this class of methods, IntCF is particularly attractive: as shown above, it is competitive in prediction accuracy with the best tree-based alternatives, while, as we show in the remainder of this section, achieving substantially better interpretability than all of them.

## 5.2. Interpretation Accuracy on Synthetic Data

While the data sets in Section 5.1 allow for evaluation of prediction accuracy by providing ground-truth causal efects, they do not lend themselves to evaluation of interpretability of models, as the mechanisms behind these efects are not known. For this reason, we generated synthetic data to compare importance measures to the structure of treatment

efects as defined by the data model.

Four data models were used to generate the synthetic data sets for this section, each with a complexity parameter s. This parameter scaled the dimension of the data, so that features for all models are generated uniformly from $[ 0 , 1 ] ^ { d }$ with $d = 5 s$ , and the parameter also afected the functions $\mu , \tau , p$ defining the models. These models were defined as follows:

1. Interaction model: An adaption of ${ \mathrm { a } } ,$ so-called, locally-spiky-sparse (LSS) model, based on [Basu et al., 2018, Behr et al., 2022], with interactions for the treatment efect, but constant baseline and randomized treatment assignment:

$$
\begin{array} { l } { p ( X ) = 0 . 5 , } \\ { \displaystyle \mu ( X ) = 0 , } \\ { \displaystyle \tau ( X ) = \sum _ { i = 1 } ^ { s } \mathbf { 1 } _ { X _ { 2 i - 1 } \leq \gamma } \cdot \mathbf { 1 } _ { X _ { 2 i } \leq \gamma } , } \end{array}
$$

where $\gamma = 0 . 7$ . The features responsible for treatment efect heterogeneity are therefore $X _ { 1 } , \ldots , X _ { 2 s }$ , appearing in interacting pairs $\left( X _ { 2 i - 1 } , X _ { 2 i } \right)$ ; since $p$ and $\mu$ do not depend on X at all, this model contains no confounders. Note that the marginal efect (see Section 4.2) of any single feature in this LSS model again corresponds to a change point model, directly connecting this simulation setting to our theoretical analysis in Section 4.

2. Change point model: An additive change point model with both confounding features and features responsible for treatment efect heterogeneity:

$$
\begin{array} { l } { \displaystyle p ( X ) = 0 . 2 + \frac { 0 . 6 } { s } \sum _ { i = s + 1 } ^ { 2 s } \mathbf { 1 } _ { X _ { i } \le \gamma , } } \\ { \displaystyle \mu ( X ) = \sum _ { i = s + 1 } ^ { 2 s } \mathbf { 1 } _ { X _ { i } \le \gamma , } } \\ { \displaystyle \tau ( X ) = \sum _ { i = 1 } ^ { s } \mathbf { 1 } _ { X _ { i } \le \gamma , } } \end{array}
$$

this time with $\gamma = 0 . 5$ . The features responsible for treatment efect heterogeneity are therefore $X _ { 1 } , \ldots , X _ { s }$ , while $X _ { s + 1 } , \ldots , X _ { 2 s }$ act as pure confounders, afecting both the propensity and the main outcome but not the treatment efect itself.

3. Linear model: A model with linear functions for propensity, main efect and

treatment efect:

$$
p ( X ) = 0 . 2 5 + \frac { 0 . 5 } { s } \sum _ { i = s + 1 } ^ { 2 s } \mathbf { X } _ { i } ,
$$

$$
\mu ( X ) = \sum _ { i = s + 1 } ^ { 2 s } \mathbf { X } _ { i } ,
$$

$$
\tau ( X ) = \sum _ { i = 1 } ^ { s } \mathbf { X } _ { i } .
$$

As in the change point model, the features responsible for treatment efect heterogeneity are $X _ { 1 } , \ldots , X _ { s }$ , while $X _ { s + 1 } , \ldots , X _ { 2 s }$ act as pure confounders.

4. Mixed model: Combination of linear and LSS model:

$$
p ( X ) = 0 . 2 + \frac { 0 . 6 } { s } \sum _ { i = 2 s + 1 } ^ { 3 s } \mathbf { 1 } _ { \mathbf { X } _ { i } \leq \gamma } ,
$$

$$
\mu ( X ) = \sum _ { i = 2 s + 1 } ^ { 3 s } \mathbf { X } _ { i } ,
$$

$$
\sigma ( X ) = \sum _ { i = 1 } ^ { s } \mathbf { 1 } _ { \mathbf { X } _ { 2 i - 1 } \leq \gamma } \cdot \mathbf { 1 } _ { \mathbf { X } _ { 2 i } \leq \gamma } + \frac { 1 } { s } \sum _ { i = s + 1 } ^ { 2 s } \mathbf { X } _ { i } ,
$$

with $\gamma = 0 . 7$ . The features responsible for treatment efect heterogeneity are therefore $X _ { 1 } , \ldots , X _ { 2 s }$ , with $X _ { s + 1 } , \ldots , X _ { 2 s }$ entering both the interaction and the linear part of $\tau ; X _ { 2 s + 1 } , . . . , X _ { 3 s }$ act as pure confounders.

In all four models, any remaining features beyond those listed above, up to the total dimension $d = 5 s$ , are pure noise, unrelated to $p , \mu .$ , or τ. The ROC-AUC statistic which we introduced below treats a feature as a positive only if it appears in the formula for $\tau ;$ confounders and noise features are therefore both treated as negatives, even though confounders do afect $p$ and $\mu .$ . Data was generated with normal distributed noise terms where the variance was equal to the variance of treatment efects.

Again, we compare our method IntCF to the three other tree-based causal machine learning methods, causal forest [Wager and Athey, 2018], GRF [Athey et al., 2019] and CausalForestDML from EconML [Battocchi et al., 2019]. For the evaluation, both predictions and feature importance generated by these models were considered. Both GRF and CausalForestDML use a maximum depth and decay exponent for their feature importance calculation. We set maximum depth to infinity (or a very high value if not possible) and decay exponent to zero to ensure that the importance measures are comparable with those of the other estimation methods. For predictions, the precision in estimation of heterogeneous treatment efects (PEHE) was calculated and normalized by dividing it by the PEHE of a dummy predictor, which predicts a constant treatment efect by the diference-in-means from the training samples. For the feature importance, we considered a ROC-AUC statistic which measures how good the feature importance can distinguish between features which actually appear in the respective formula for treatment efect τ and those which do not.

![](images/2c206480d2f6e771af4b795d9dc64574a647a3f8df36a30b3f8c421de3c7956e.jpg)  
Figure 4: Comparison of precision in estimation of heterogeneous efects (PEHE) of IntCF (blue), Forest DML (yellow), causal forest (green) and GRF (red), normalized to PEHE of a constant diference-in-means estimator. For each method, a band of one standard error is around the respective mean. In this figure, lower values are better. Top left shows interaction model, top right change point model, bottom left linear model, and bottom right mixed model.

Each of the models with parameter s from 1 to 10 were used for simulations. For each model and parameter, 500 · s training samples and 100 000 separate evaluation samples were generated. The training samples were then used to train the machine learning models, which were subsequently evaluated on the evaluation samples. For each combination of data model, parameter s, and machine learning method, these steps were repeated 20 times.

The results of these simulations for precision in estimation are shown in Figure 4. In most considered data models IntCF achieves the best prediction results, especially for the Interaction and Mixed model. For the Change point model, Forest DML and GRF, both of which include a double machine learning/orthogonalization step, are the best performing methods. Between the two methods that do not use this additional step,

![](images/7e51e7d40abdcf21ddd88be146143792ec8390013df7fbe4cecce568764ff5de.jpg)  
Figure 5: ROC-AUC for feature importance as signal feature classifier for IntCF (blue), Forest DML (yellow), causal forest (green) and GRF (red). For each method, a band of one standard error is around the respective mean. In this figure, higher values are better. Top left shows interaction model, top right change point model, bottom left linear model, and bottom right mixed model.

IntCF has a better precision than causal forest in all cases except one (Interaction model with s = 1).

Simulation results regarding feature importance are shown in Figure 5. In all considered combinations of data model and complexity parameter s, the feature importance of IntCF results in a ROC-AUC of almost 1, while each other method achieves smaller values in most situations. This reflects a substantial improvement in interpretability, as IntCF is best at identifying the features actually relevant for treatment efect heterogeneity. This improvement is particularly notable given the relative simplicity of IntCF. As discussed in Section 3, IntCF retains the same structure as the original RF: predictions are obtained directly as the average of individual decision trees, with no additional correction step required, and feature importance can be read of directly from the tree structure. Causal forest shares this simple structure, but does not distinguish heterogeneity from bias when selecting splits, and consequently performs worse in our feature importance comparison. GRF and CausalForestDML, in contrast, rely on a substantially more involved procedure: both incorporate an additional orthogonalization (double machine learning) step, which alters the prediction mechanism and means that predictions can no longer be read of directly from the individual trees, unlike for IntCF and causal forest. Despite this added complexity, GRF and CausalForestDML do not achieve better feature importance scores than IntCF in our simulations. IntCF thus achieves the strongest interpretability results of all four methods while remaining, by construction, the structurally simplest.

![](images/19a5fe5729782ef84569e03ccaadd670e4ed682f25245de8166a17313b95bf5f.jpg)  
Figure 6: Importance scores for heterogeneity (blue) and bias (orange) produced by IntCF for NHEFS data set. Features are sorted by heterogeneity score. For the bias scores, negative values where truncated at 0.

## 5.3. Application on NHEFS Data set

Finally, we demonstrate the application of our algorithm on a real-world data set: the National Health and Nutrition Examination Survey I Epidemiologic Followup Study (NHEFS)<sup>2</sup> is a longitudinal study of a cohort first examined in 1971–75. This data set was also used previously to exemplify the handling of confounders and treatment efect heterogeneity [Hernan and Robins, 2020]. Like Hernan and Robins [2020], we consider data from 1566 smokers to estimate the efect of quitting smoking on weight change up to the follow-up examination in 1982–84, using the same 9 baseline covariates as Hernan and Robins [2020], including, e.g., the weight at the initial examination (wt71), the number of years of smoking (smokeyrs), and age (age). Notably, our method does not require these covariates to be labeled a priori as confounders or efect modifiers; instead, it attributes each split to bias correction or heterogeneity directly from the fitted tree.

For the average treatment efect, we obtain an estimated weight gain of 3.2 kg due to quitting smoking, slightly lower than the 3.4 kg reported by Hernan and Robins [2020].

Additionally, we evaluate feature importance scores for the contribution to heterogeneity and bias of the predictions, shown in Figure 6. As can be seen in Figure 6, the feature with the highest heterogeneity importance is wt71, the weight at the initial examination, consistent with other methods that identify this feature as having the strongest influence on heterogeneity [see, e.g., B´enard and Josse, 2025]. In addition, our bias importance score identifies age as the most relevant confounder.

## 6. Discussion

## 6.1. Summary

In this paper, we introduced the interpretable causal forest (IntCF), a modification of RF for estimating individual treatment efects. The only change relative to standard RF is the splitting criterion: by combining a heterogeneity criterion with a novel bias criterion, each split can be attributed to either genuine heterogeneity in the treatment efect or a correction for confounding bias. Together with a validation step that honestly re-evaluates these contributions at each node (Section 3.2), this enables direct interpretation of the resulting tree structure, for example through standard feature importance measures. We supported this splitting criterion theoretically by analyzing its population limit under a change point model tailored to causal inference, showing that it correctly attributes a split to bias correction precisely when the naive diference-in-means estimator is biased, and to heterogeneity precisely when the treatment efect is genuinely non-constant. In simulations, IntCF achieved prediction accuracy competitive with established treebased methods, while substantially outperforming all of them in recovering the features responsible for treatment efect heterogeneity, especially in the presence of confounders.

## 6.2. Simplicity as a Feature, not a Limitation

A recurring theme of this paper is that IntCF requires no machinery beyond a modified splitting criterion: predictions are still obtained directly as the average of individual decision trees, exactly as in the original RF. This stands in contrast to generalized random forest and CausalForestDML, both of which rely on an additional orthogonalization (double machine learning) step that changes the prediction mechanism and severs the direct correspondence between the fitted trees and the resulting predictions. Causal forest shares IntCF’s simple structure, but does not distinguish heterogeneity from bias at the split level, and correspondingly falls short of IntCF’s interpretability in our simulations. That IntCF exceeds the interpretability of substantially more complex competitors, while remaining competitive with them in prediction accuracy, is in our view the central practical contribution of this paper: interpretability need not come at the cost of added model complexity.

## 6.3. Limitations and Future Work

Several open questions remain. On the theoretical side, our analysis of the splitting criteria is currently limited to the population level; extending these results to finite samples, to more general data-generating processes beyond the change point model, and to the validation step from Section 3.2 are natural next steps. Likewise, our analysis of marginalization efects in Section 4.2 considered only a single confounder at a time; understanding how these efects interact in the presence of multiple, simultaneously confounding features warrants further investigation.

Another open question is the consistency of the resulting IntCF estimator, which we did not establish in this paper. We expect that an adaptation of the consistency proof for causal forest from Wager and Athey [2018] should be feasible. Such a proof typically requires the conditional mean function E $[ Y \mid \mathbf { X } = \mathbf { x } ]$ to be Lipschitz continuous, an assumption violated by the discontinuous change point model used in our theoretical analysis. However, as tree depth increases, the fraction of samples afected by any such discontinuity should vanish, suggesting that this is a technical rather than a fundamental obstacle.

## Code Availability

An implementation of IntCF, together with code to reproduce all analyses and figures in this paper, is publicly available at https://github.com/behr-group/intcf.

## Acknowledgments

The project was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), project number 509149993, TRR 374.

## References

Abhineet Agarwal, Ana M. Kenney, Yan Shuo Tan, Tifany M. Tang, and Bin Yu. Integrating random forests and generalized linear models for improved accuracy and interpretability, 2025. URL https://arxiv.org/abs/2307.01932.

Joshua D. Angrist and J¨orn-Stefen Pischke. Mastering ’metrics: the path from cause to efect. Princeton University Press, 2015.

Susan Athey and Guido Imbens. Recursive partitioning for heterogeneous causal efects. Proceedings of the National Academy of Sciences, 113(27):7353–7360, 2016. doi: 10.1073/pnas.1510489113.

Susan Athey, Julie Tibshirani, and Stefan Wager. Generalized random forests. The Annals of Statistics, 47(2):1148–1178, 2019. doi: 10.1214/18-AOS1709.

Vahid Balazadeh, Hamidreza Kamkari, Valentin Thomas, Benson Li, Junwei Ma, Jesse C. Cresswell, and Rahul G. Krishnan. CausalPFN: Amortized causal efect estimation via in-context learning. In Advances in Neural Information Processing Systems, volume 38, 2025.

Sumanta Basu, Karl Kumbier, James B. Brown, and Bin Yu. Iterative random forests to discover predictive and stable high-order interactions. Proceedings of the National Academy of Sciences, 115(8):1943–1948, 2018. doi: 10.1073/pnas.1711236115.

Keith Battocchi, Eleanor Dillon, Maggie Hei, Greg Lewis, Paul Oka, Miruna Oprescu, and Vasilis Syrgkanis. EconML: A Python Package for ML-Based Heterogeneous Treatment Efects Estimation. https://github.com/py-why/EconML, 2019. Version 0.x.

Merle Behr, Yu Wang, Xiao Li, and Bin Yu. Provable boolean interaction recovery from tree ensemble obtained via random forests. Proceedings of the National Academy of Sciences, 119(22):e2118636119, 2022. doi: 10.1073/pnas.2118636119.

Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001. doi: 10.1023/A: 1010933404324.

Leo Breiman, Jerome H. Friedman, Richard A. Olshen, and Charles J. Stone. Classification and Regression Trees. Chapman and Hall/CRC, 1984. doi: 10.1201/9781315139470.

Cl´ement B´enard and Julie Josse. Variable importance for causal forests: breaking down the heterogeneity of treatment efects. Journal of Causal Inference, 13(1):20230062, 2025. doi: 10.1515/jci-2023-0062.

Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1):C1–C68, 2018. doi: 10.1111/ectj.12097.

Alicia Curth and Mihaela van der Schaar. Nonparametric estimation of heterogeneous treatment efects: From theory to learning algorithms. In Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS), volume 130, pages 1810–1818, 2021.

Vincent Dorie, Jennifer Hill, Uri Shalit, Marc Scott, and Dan Cervone. Automated versus do-it-yourself methods for causal inference: Lessons learned from a data analysis competition. Statistical Science, 34(1):43–68, 2019. doi: 10.1214/18-STS667.

Miguel A. Hernan and James M. Robins. Causal Inference: What If. Chapman & Hall/CRC, 2020.

Jennifer L. Hill. Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics, 20(1):217–240, 2011. doi: 10.1198/jcgs.2010. 08162.

Guido W. Imbens and Donald B. Rubin. Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction. Cambridge University Press, 2015. doi: 10. 1017/CBO9781139025751.

Hao Jiang, Peng Qi, Jingying Zhou, Jack Zhou, and Sharath Rao. A short survey on forest based heterogeneous treatment efect estimation methods: Meta-learners and specific models. In 2021 IEEE International Conference on Big Data (Big Data), pages 3006–3012, 2021. doi: 10.1109/BigData52589.2021.9671439.

S¨oren R. K¨unzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners for estimating heterogeneous treatment efects using machine learning. Proceedings of the National Academy of Sciences, 116(10):4156–4165, 2019. doi: 10.1073/pnas.1804597116.

Robert J. LaLonde. Evaluating the econometric evaluations of training programs with experimental data. The American Economic Review, 76(4):604–620, 1986.

Xiao Li, Yu Wang, Sumanta Basu, Karl Kumbier, and Bin Yu. A debiased MDI feature importance measure for random forests. In Advances in Neural Information Processing Systems, volume 32. 2019.

Scott M. Lundberg, Gabriel Erion, Hugh Chen, Alex DeGrave, Jordan M. Prutkin, Bala Nair, Ronit Katz, Jonathan Himmelfarb, Nisha Bansal, and Su-In Lee. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2(1):56–67, 2020. doi: 10.1038/s42256-019-0138-9.

Charles F. Manski. Identification problems in the social sciences. Sociological Methodology, 23:1–56, 1993. doi: 10.2307/271005.

Brady Neal, Chin-Wei Huang, and Sunand Raghupathi. RealCause: Realistic causal inference benchmarking, 2021. URL https://arxiv.org/abs/2011.15007.

Miruna Oprescu, Vasilis Syrgkanis, and Zhiwei Steven Wu. Orthogonal random forest for causal inference. In Proceedings of the 36th International Conference on Machine Learning, volume 97, pages 4932–4941, 2019.

Fabian Pedregosa, Ga¨el Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Edouard Duchesnay. Scikit-learn: Machine learning in Python.<sup>´</sup> Journal of Machine Learning Research, 12(85):2825–2830, 2011.

Craig T. Ramey, Donna M. Bryant, Barbara H. Wasik, Joseph J. Sparling, Kaye H. Fendt, and Lisa M. La Vange. Infant health and development program for low birth weight, premature infants: program elements, family participation, and child intelligence. Pediatrics, 89(3):454–465, 1992. doi: 10.1542/peds.89.3.454.

Paul R. Rosenbaum and Donald B. Rubin. The central role of the propensity score in observational studies for causal efects. Biometrika, 70(1):41–55, 1983. doi: 10.1093/ biomet/70.1.41.

Donald B. Rubin. Estimating causal efects of treatments in randomized and nonrandomized studies. Journal of Educational Psychology, 66(5):688–701, 1974. doi: 10.1037/h0037350.

Erwan Scornet and Giles Hooker. Theory of random forests. Annual Review of Statistics and Its Application, 13:99–121, 2026. doi: 10.1146/annurev-statistics-112723-034707.

Uri Shalit, Fredrik D. Johansson, and David Sontag. Estimating individual treatment efect: generalization bounds and algorithms. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pages 3076–3085, 2017.

Claudia Shi, David Blei, and Victor Veitch. Adapting neural networks for the estimation of treatment efects. In Advances in Neural Information Processing Systems, volume 32. 2019.

Assaf Shmuel, Oren Glickman, and Teddy Lazebnik. A comprehensive benchmark of machine and deep learning models on structured data for regression and classification. Neurocomputing, 655:131337, 2025. doi: 10.1016/j.neucom.2025.131337.

Stefan Wager and Susan Athey. Estimation and inference of heterogeneous treatment efects using random forests. Journal of the American Statistical Association, 113(523): 1228–1242, 2018. doi: 10.1080/01621459.2017.1319839.

Zhengze Zhou and Giles Hooker. Unbiased measurement of feature importance in treebased methods. ACM Transactions on Knowledge Discovery from Data (TKDD), 15 (2):26:1–26:21, 2021. doi: 10.1145/3429445.

## A. Proofs

## A.1. Proofs of Section 3

Proof of Lemma 1. In the following computation, the property $J _ { \mathrm { p a r e n t } } = J _ { L } \sqcup J _ { R }$ is used in the second equation and Assumption (12) is used in the third equation:

$$
\begin{array} { r l } { \frac { 1 } { 6 } \sum _ { j = 1 } ^ { n } \sum _ { i = 1 } ^ { n } - \frac { 1 } { 6 } ( \frac { 1 } { 2 4 } \sum _ { j = 1 } ^ { n } - 6 \frac { j - 1 } { 2 4 } - \frac { 1 } { 2 4 } \sum _ { i = 1 } ^ { n } - 6 \frac { j - 1 } { 2 4 } ) } & { { } } \\ { = \frac { 1 } { 6 } ( \sum _ { j = 1 } ^ { n } \frac { 1 } { 6 } \sum _ { i = 1 } ^ { n } - \frac { 1 } { 6 } \sum _ { j = 2 } ^ { n } - \frac { 1 } { 6 } \sum _ { i = 1 } ^ { n } - \frac { 1 } { 6 } ) } & { { } } \\ { \frac { 1 } { 6 } ( \frac { 1 } { 6 } \sum _ { j = 1 } ^ { n } - \frac { 1 } { 6 } \sum _ { i = 1 } ^ { n } - \frac { 1 } { 6 } ) ( \frac { 1 } { 6 } \sum _ { j = 1 } ^ { n } \frac { 1 } { 6 } \sum _ { i = 1 } ^ { n } \frac { 1 } { 6 } ) ( \sum _ { i = 1 } ^ { n } \frac { 1 } { 6 } \sum _ { j = 1 } ^ { n } \frac { 1 } { 6 } ) ( \frac { 1 } { 6 } \sum _ { i = 1 } ^ { n } \frac { 1 } { 6 } ) ( \frac { 1 } { 6 } ) } & { { } } \\ { \frac { 1 } { 6 } ( \frac { 1 } { 6 } \sum _ { j = 1 } ^ { n } - \frac { 1 } { 6 } \sum _ { i = 1 } ^ { n } - \frac { 1 } { 6 } \sum _ { j = 1 } ^ { n } ) ( \frac { 1 } { 6 } \sum _ { i = 1 } ^ { n } - \frac { 1 } { 6 } \sum _ { j = 1 } ^ { n } \frac { 1 } { 6 } ) } & { { } } \\  \frac { 1 } { 6 } ( \frac { 1 } { 6 } \sum _ { i = 1 } ^ { n } \frac { 1 } { 6 } \sum _ { j = 1 } ^  \end{array}
$$

Proof of Lemma 2. By using assumption (13) direct computation yields:

$$
\begin{array} { r l } & { \frac { 1 } { N } ( \pi _ { L } ( \hat { \tau } _ { \mathrm { p x r e n t } } - \hat { \tau } _ { L } ) ^ { 2 } + n \pi ( \hat { \tau } _ { \mathrm { p a r e n t } } - \hat { \tau } _ { R } ) ^ { 2 } ) } \\ & { = \frac { 1 } { N } ( n _ { L } ( \frac { n _ { L } \hat { \tau } _ { L } + n _ { R } \hat { \tau } _ { R } } { n _ { L } + n _ { R } } - \hat { \tau } _ { L } ) ^ { 2 } + n \pi ( \frac { n _ { L } \hat { \tau } _ { L } + n _ { R } \hat { \tau } _ { R } } { n _ { L } + n _ { R } } - \hat { \tau } _ { R } ) ^ { 2 } ) } \\ & { = \frac { 1 } { N } ( n _ { L } ( \frac { - n _ { R } \hat { \tau } _ { L } + n _ { R } \hat { \tau } _ { L } } { n _ { L } + n _ { R } } ) ^ { 2 } + n _ { R } ( \frac { n _ { L } \hat { \tau } _ { L } - n _ { L } \hat { \tau } _ { R } } { n _ { L } + n _ { R } } ) ^ { 2 } ) } \\ & { = \frac { 1 } { N } ( \frac { n _ { L } \cdot n _ { R } ^ { 2 } } { ( n _ { L } + n _ { R } ) ^ { 2 } } ( - \hat { \tau } _ { L } + \hat { \tau } _ { R } ) ^ { 2 } + \frac { n _ { L } ^ { 2 } \cdot n _ { R } } { ( n _ { L } + n _ { R } ) ^ { 2 } } ( \hat { \tau } _ { L } - \hat { \tau } _ { R } ) ^ { 2 } ) } \\ &  = \frac { n _ { L } \cdot n _ { R } } { N } ( \frac { n _ { R } } { ( n _ { L } + n _ { R } ) ^ { 2 } } ( \frac { n _ { R } } { n _ { L } + n _ { R } } ( \hat { \tau } _ { L } - \hat { \tau } _ { R } ) ^ { 2 } + \frac  \end{array}
$$

Proof of Lemma 3. By direct computation:

$$
\begin{array} { r l } & { \frac { 1 } { N } ( \eta _ { \mathrm { e f f } } ( \eta _ { \mathrm { e f f } } , \eta _ { \mathrm { e f f } } ) - \hat { \eta } _ { \mathrm { e f f } } ^ { 2 } - \kappa _ { \mathrm { e f f } } ^ { 3 } / \frac { \eta _ { \mathrm { e f f } } } { N } ) ^ { 2 } - \frac { \eta _ { \mathrm { e f f } } - \hat { \eta } _ { \mathrm { e f f } } } { N ^ { 2 } ( \eta _ { \mathrm { e f f } } ) ( \eta _ { \mathrm { e f f } } ) ( \hat { \eta } _ { \mathrm { e f f } } - \hat { \eta } _ { \mathrm { e f f } } ^ { 2 } ) } } \\ & { - \frac { \eta _ { \mathrm { e f f } } } { N } ( \eta _ { \mathrm { e f f } } ( \eta _ { \mathrm { e f f } } , \eta _ { \mathrm { e f f } } ) - \hat { \eta } _ { \mathrm { e f f } } ^ { 3 } + m _ { \mathrm { e } } ^ { 2 } ( \eta _ { \mathrm { e f f } } ^ { 3 } - \eta _ { \mathrm { e f f } } ^ { 3 } - \eta _ { \mathrm { e f f } } ^ { 3 } - \eta _ { \mathrm { e f f } } ^ { 3 } - \eta _ { \mathrm { e f f } } ^ { 3 } - \eta _ { \mathrm { e f f } } ^ { 3 } ) ) } \\ &  = \frac { 1 } { N } ( \eta _ { \mathrm { e f f } } ( \hat { \eta } _ { \mathrm { e f f } } ^ { 2 } , 0 ) - \frac { \eta _ { \mathrm { e f f } } } { N ^ { 2 } ( \eta _ { \mathrm { e f f } } ) ( \hat { \eta } _ { \mathrm { e f f } } + \eta _ { \mathrm { e f f } } ^ { 3 } + \eta _ { \mathrm { e f f } } ^ { 3 } ) } + \eta _ { \mathrm { e f f } } ( \hat { \eta } _ { \mathrm { e f f } } ^ { 2 } , 0 ) - \frac  \eta _ { \mathrm { e f f } } - \eta _ { \mathrm { e f f } } ( \eta _ { \mathrm { e f f } } - \eta _ { \mathrm { e f f } } ^  \end{array}
$$

Proof of Lemma 4. At each node e we have $\begin{array} { r } { \hat { \tau } _ { e } ^ { \mathrm { b } } = \frac { 1 } { n } \sum _ { i \in I _ { e } } \hat { \tau } _ { i } ^ { \mathrm { v a l } } } \end{array}$ , so

$$
\begin{array} { l } { \displaystyle \sum _ { \zeta \in \mathcal { L } _ { c } } \left( \frac { \bar { \eta } ^ { \kappa } } { c } - \frac { \bar { \eta } ^ { \kappa } } { c _ { \kappa } ^ { 3 } } \right) ^ { 2 } - \sum _ { i \in \mathcal { L } _ { c } } \left( \frac { \bar { \eta } ^ { \kappa } } { c } - \frac { \bar { \eta } ^ { \kappa } } { c _ { \kappa } ^ { 3 } } \right) ^ { 2 } } \\ { = \displaystyle \sum _ { \zeta = L _ { c } } \left( ( \frac { \bar { \eta } ^ { \kappa } } { c } ) ^ { 2 } - 2 \frac { \bar { \eta } ^ { \kappa } } { c _ { \kappa } ^ { 3 } c _ { \kappa } ^ { 3 } } + ( \frac { \bar { \eta } ^ { \kappa } } { c _ { \kappa } ^ { 3 } } ) ^ { 2 } \right) - \sum _ { i \in L _ { c } } \left( ( \frac { \bar { \eta } ^ { \kappa } } { c } ) ^ { 2 } - 2 \frac { \bar { \eta } ^ { \kappa } } { c _ { \kappa } ^ { 3 } c _ { \kappa } ^ { 3 } } + ( \frac { \bar { \eta } ^ { \kappa } } { c _ { \kappa } ^ { 3 } } ) ^ { 2 } \right) } \\ { = \displaystyle \frac { \big ( ( \bar { \eta } _ { c } ^ { \kappa } ) ^ { 2 } - ( \frac { \bar { \kappa } } { c } ) c _ { \kappa } ^ { 2 } \big ) ^ { 2 } } { \big ( \frac { \bar { \eta } ^ { \kappa } } { c } - \frac { \bar { \eta } ^ { \kappa } } { c _ { \kappa } ^ { 3 } } \big ) ^ { 2 } } + 2 \left( \frac { \bar { \eta } ^ { \kappa } } { c } - \frac { \bar { \eta } ^ { \kappa } } { c _ { \kappa } ^ { 4 } } \right) \sum _ { \kappa \in \mathcal { L } _ { c } } \frac { \bar { \eta } ^ { \kappa \kappa } } { c _ { \kappa } ^ { 3 } } } \\  = \displaystyle \frac  \big ( ( \bar { \eta } _ { c } ^ { \kappa } ) ^ { 2 } - ( \end{array}
$$

and, using $\begin{array} { r } { \hat { \tau } _ { \mathrm { p a r e n t } } ^ { \mathrm { b } } = \frac { n _ { L } \hat { \tau } _ { L } ^ { \mathrm { b } } + n _ { R } \hat { \tau } _ { R } ^ { \mathrm { b } } } { n _ { L } + n _ { R } } } \end{array}$ , we get

$$
\begin{array} { r l } & { \underset { \leq t \leq m } { \sum } ( \frac { \hat { \gamma } _ { \mathbf { b } \theta } } { \mathbf { b } t } \alpha u - \hat { \gamma } _ { \mathbf { b } } ^ { - \mathbf { a } } \mathbf { b } ^ { \theta } ) ^ { t } - \underset { \leq t \leq m } { \sum } ( \frac { \hat { \gamma } _ { \mathbf { b } } ^ { - \mathbf { b } } } { 2 } - \frac { \hat { \gamma } _ { \mathbf { b } } ^ { - \mathbf { b } } } { t } ) ^ { t } - \underset { \leq t \leq m } { \sum } ( \frac { \hat { \gamma } _ { \mathbf { b } } ^ { - \mathbf { b } } } { 2 } - \frac { \hat { \gamma } _ { \mathbf { b } } ^ { - \mathbf { b } } } { t } ) ^ { t } } \\ & { = \underset { \leq t \leq m } { \sum } ( ( \frac { \hat { \gamma } _ { \mathbf { b } \theta } } { \mathbf { b } t } ) ^ { t } - 2 \frac { \hat { \gamma } _ { \mathbf { b } } ^ { t } } { t } ) \alpha \alpha \frac { \hat { \gamma } _ { \mathbf { b } \theta } ^ { t } } { t } + \binom { 2 \hat { \gamma } _ { \mathbf { b } \theta } ^ { t } } { t } ) ^ { t } } \\ &  \quad - \underset { \leq t \leq m } { \sum } ( ( \frac { \hat { \gamma } _ { \mathbf { b } \theta } } { \mathbf { b } t } ) ^ { t } - 2 \frac { \hat { \gamma } _ { \mathbf { b } } ^ { t } } { t } ) ^ { t } + \underset { \leq t \leq m } { \sum } ( \frac { \hat { \gamma } _ { \mathbf { b } \theta } ^ { t } } { t } ) ^ { t } - \underset { \leq t \leq m } { \sum } ( \frac { \hat { \gamma } _ { \mathbf { b } \theta } ^ { t } } { t } ) ^ { t } - \underset { \leq t \leq m } { \sum } \alpha \frac { \hat { \gamma } _ { \mathbf { b } \theta } ^ { t } } { t } + \langle \nabla _ { \mathbf { b } \theta } ^ { t }  \end{array}
$$

Combining these results, we have

$$
\begin{array} { r l } & { \displaystyle \sum _ { i \in I _ { \mathrm { p a r e n t } } } \left( \hat { \tau } _ { \mathrm { p a r e n t } } ^ { \mathrm { f } } - \hat { \tau } _ { i } ^ { \mathrm { v a l } } \right) ^ { 2 } - \sum _ { i \in I _ { L } } \left( \hat { \tau } _ { L } ^ { \mathrm { f } } - \hat { \tau } _ { i } ^ { \mathrm { v a l } } \right) ^ { 2 } - \sum _ { i \in I _ { R } } \left( \hat { \tau } _ { R } ^ { \mathrm { f } } - \hat { \tau } _ { i } ^ { \mathrm { v a l } } \right) ^ { 2 } = } \\ & { \displaystyle \frac { n _ { L } \cdot n _ { R } } { n _ { L } + n _ { R } } \left( \hat { \tau } _ { L } ^ { \mathrm { b } } - \hat { \tau } _ { R } ^ { \mathrm { b } } \right) ^ { 2 } + \frac { n _ { L } + n _ { R } } { N } \left( \hat { \tau } _ { \mathrm { p a r e n t } } ^ { \mathrm { f } } - \hat { \tau } _ { \mathrm { p a r e n t } } ^ { \mathrm { b } } \right) ^ { 2 } - \frac { n _ { L } } { N } \left( \hat { \tau } _ { L } ^ { \mathrm { f } } - \hat { \tau } _ { L } ^ { \mathrm { b } } \right) ^ { 2 } - \frac { n _ { R } } { N } \left( \hat { \tau } _ { R } ^ { \mathrm { f } } - \hat { \tau } _ { R } ^ { \mathrm { b } } \right) ^ { 2 } . } \end{array}
$$

## A.2. Proof of Theorem 5

Remark 4. In the following statements and proofs, we will assume w.l.o.g. that the covariate X is uniformly distributed on [0, 1], also implying $R ( e ) = [ 0 , 1 ]$ . If its domain would be a diferent interval [a, b], replacing it by ${ \frac { X - a } { b - a } } - a \cdot$ s well as performing the same transformation for γ and k—would create an equivalent model with the same values for treatment efect estimates and (signed) splitting criteria.

Lemma 7. For the univariate change-point model (27) with potential outcomes as in (28), assuming Assumption A3 holds, the bias of the diference-in-means estimate as defined in (29) is given by

$$
\begin{array} { r l } & { \left( \mathbb { E } \left[ Y ^ { T = 1 } \mid T = 1 \right] - \mathbb { E } \left[ Y ^ { T = 0 } \mid T = 0 \right] \right) - \mathbb { E } \left[ \tau ( X ) \right] } \\ & { \qquad = \gamma ( 1 - \gamma ) q _ { 1 } \frac { \alpha _ { 1 } + ( 1 / 2 - q _ { 0 } - \gamma q _ { 1 } ) \beta _ { 1 } } { ( q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) } . } \end{array}\tag{34}
$$

Proof. Because of Assumption A3, the conditional expectations are well defined and we have

$$
\begin{array} { l } { \mathbb { E } \left[ Y ^ { T = 0 } \mid T = 0 \right] = \alpha _ { 0 } - \beta _ { 0 } / 2 + \frac { \left( \alpha _ { 1 } - \beta _ { 1 } / 2 \right) \gamma \left( 1 - q _ { 0 } - q _ { 1 } \right) } { 1 - q _ { 0 } - \gamma q _ { 1 } } , } \\ { \mathbb { E } \left[ Y ^ { T = 1 } \mid T = 1 \right] = \alpha _ { 0 } + \beta _ { 0 } / 2 + \frac { \left( \alpha _ { 1 } + \beta _ { 1 } / 2 \right) \gamma \left( q _ { 0 } + q _ { 1 } \right) } { q _ { 0 } + \gamma q _ { 1 } } } \end{array}
$$

and hence,

$$
\begin{array} { r l } & { \quad \mathbb { E } \left[ Y ^ { T = 1 } \mid T = 1 \right] - \mathbb { E } \left[ Y ^ { T = 0 } \mid T = 0 \right] } \\ & { = \beta _ { 0 } + \frac { \left( \alpha _ { 1 } + \beta _ { 1 } / 2 \right) \gamma \left( q _ { 0 } + q _ { 1 } \right) } { q _ { 0 } + \gamma q _ { 1 } } - \frac { \left( \alpha _ { 1 } - \beta _ { 1 } / 2 \right) \gamma \left( 1 - q _ { 0 } - q _ { 1 } \right) } { 1 - q _ { 0 } - \gamma q _ { 1 } } } \\ & { = \beta _ { 0 } + \gamma ( \alpha _ { 1 } - \beta _ { 1 } / 2 ) \frac { \left( 1 - \gamma \right) q _ { 1 } } { \left( q _ { 0 } + \gamma q _ { 1 } \right) \left( 1 - q _ { 0 } - \gamma q _ { 1 } \right) } + \gamma \beta _ { 1 } \left( 1 + \frac { \left( 1 - \gamma \right) q _ { 1 } } { q _ { 0 } + \gamma q _ { 1 } } \right) . } \end{array}
$$

Moreover, we have $\mathbb { E } \left[ \tau ( X ) \right] = \beta _ { 0 } + \gamma \beta _ { 1 }$ and hence,

$$
\begin{array} { r l } & { \quad \quad \left( \mathbb { E } \left[ Y ^ { T = 1 } \mid T = 1 \right] - \mathbb { E } \left[ Y ^ { T = 0 } \mid T = 0 \right] \right) - \mathbb { E } \left[ \tau ( X ) \right] } \\ & { = ( \alpha _ { 1 } - \beta _ { 1 } / 2 ) \frac { \gamma \left( 1 - \gamma \right) q _ { 1 } } { \left( q _ { 0 } + \gamma q _ { 1 } \right) \left( 1 - q _ { 0 } - \gamma q _ { 1 } \right) } + \beta _ { 1 } \frac { \gamma \left( 1 - \gamma \right) q _ { 1 } } { q _ { 0 } + \gamma q _ { 1 } } } \\ & { = \gamma ( 1 - \gamma ) q _ { 1 } \frac { \alpha _ { 1 } + \left( 1 / 2 - q _ { 0 } - \gamma q _ { 1 } \right) \beta _ { 1 } } { \left( q _ { 0 } + \gamma q _ { 1 } \right) \left( 1 - q _ { 0 } - \gamma q _ { 1 } \right) } . } \end{array}
$$

Proof for Theorem 5. Define

$$
\begin{array} { r l r l r l r l r l } & { \alpha _ { 0 } ^ { \prime } : = \alpha _ { 0 } , } & & { \alpha _ { 1 } ^ { \prime } : = \alpha _ { 1 } - \alpha _ { 1 } ^ { \prime \prime } , } & & { \beta _ { 0 } ^ { \prime } : = \beta _ { 0 } , } & & { } & \\ & { \alpha _ { 0 } ^ { \prime \prime } : = 0 , } & & { \alpha _ { 1 } ^ { \prime \prime } : = - ( 1 / 2 - q _ { 0 } - \gamma q _ { 1 } ) \beta _ { 1 } , } & & { \beta _ { 0 } ^ { \prime \prime } : = 0 , } & & { \beta _ { 1 } ^ { \prime \prime } : = \beta _ { 1 } . } & & { } & \end{array}
$$

With these definitions, $\mu ( x ) = \mu ^ { B } ( x ) + \mu ^ { H } ( x )$ and $\tau ( x ) = \tau ^ { B } ( x ) + \tau ^ { H } ( x )$ follows directly.

For the second model, we consider the diference between the diference-in-means prediction $\begin{array} { r } { \mathbb { E } \left[ \tau ^ { H } ( X ) \right] = \mathbb { E } \left[ \mu ^ { H } ( X ) + \frac { 1 } { 2 } \tau ^ { H } ( X ) \ | \ T = 1 \right] - \mathbb { E } \left[ \mu ^ { H } ( X ) - \frac { 1 } { 2 } \tau ^ { H } ( X ) \ | \ T = 0 \right] } \end{array}$ and the expectation of the treatment efect E $\left[ \tau ^ { H } ( X ) \right]$ . By Lemma 7 this is given by

$$
\gamma ( 1 - \gamma ) q _ { 1 } \frac { \alpha _ { 1 } ^ { \prime \prime } + ( 1 / 2 - q _ { 0 } - \gamma q _ { 1 } ) \beta _ { 1 } ^ { \prime \prime } } { ( q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) } = 0 ,
$$

where for the evaluation we used $\alpha _ { 1 } ^ { \prime \prime } = - ( 1 / 2 - q _ { 0 } - \gamma q _ { 1 } ) \beta _ { 1 }$ and $\beta _ { 1 } ^ { \prime \prime } = \beta _ { 1 }$ . So the diference-in-means estimator is in expectation equal to the treatment efect, i.e., is unbiased. □

## A.3. Proof of Theorem 6

Remark 5. We will continue assuming $X \sim U ( [ 0 , 1 ] )$ , see Remark 4.

Further we will assume that $\gamma \in ( 0 , 1 )$ in (27). For values outside (0, 1), there is a model with $\beta _ { 1 } = q _ { 1 } = \alpha _ { 1 } = 0$ and $\gamma \in ( 0 , 1 )$ which has almost surely the same propensities and expected outcomes, which therefore also result in the same expected predictions and (signed) splitting criteria.

Lemma 8. For the univariate change-point model (27), assuming Assumption A3 holds, with potential outcomes as in (28) and a split point $k \in ( 0 , 1 )$ define

$$
\begin{array} { r l } & { \hat { \tau } _ { L } ^ { \infty } : = \mathbb { E } \left[ Y ^ { T = 1 } \mid X \le k , T = 1 \right] - \mathbb { E } \left[ Y ^ { T = 0 } \mid X \le k , T = 0 \right] , } \\ & { \hat { \tau } _ { R } ^ { \infty } : = \mathbb { E } \left[ Y ^ { T = 1 } \mid X > k , T = 1 \right] - \mathbb { E } \left[ Y ^ { T = 0 } \mid X > k , T = 0 \right] } \end{array}\tag{35}
$$

Then we have

$$
\hat { \tau } _ { L } ^ { \infty } = \left\{ \begin{array} { l l } { \beta _ { 0 } + \beta _ { 1 } } & { k \le \gamma , } \\ { \beta _ { 0 } + \left( \alpha _ { 1 } - \beta _ { 1 } / 2 \right) \frac { \gamma \left( k - \gamma \right) q _ { 1 } } { \left( k q _ { 0 } + \gamma q _ { 1 } \right) \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) } + \beta _ { 1 } \left( 1 - \frac { ( k - \gamma ) q _ { 0 } } { k q _ { 0 } + \gamma q _ { 1 } } \right) } & { k > \gamma } \end{array} \right.
$$

and

$$
\hat { \tau } _ { R } ^ { \infty } = \left\{ \begin{array} { l l } { \beta _ { 0 } + ( \alpha _ { 1 } - \beta _ { 1 } / 2 ) \frac { ( \gamma - k ) ( 1 - \gamma ) q _ { 1 } } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) } } & { } \\ { \quad + \beta _ { 1 } \left( 1 - \frac { ( 1 - \gamma ) q _ { 0 } } { ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } } \right) } & { k \le \gamma , } \\ { \beta _ { 0 } } & { k > \gamma . } \end{array} \right.
$$

Proof. For a splitting point k, we have

$$
\begin{array}{c} \begin{array} { r l } & { \mathbb { E } \left[ Y ^ { T = 0 } \mid X \leq k , T = 0 \right] = \left\{ \begin{array} { l l } { \alpha _ { 0 } - \beta _ { 0 } / 2 + \alpha _ { 1 } - \beta _ { 1 } / 2 } & { k \leq \gamma , } \\ { \alpha _ { 0 } - \beta _ { 0 } / 2 + \frac { ( \alpha _ { 1 } - \beta _ { 1 } / 2 ) \gamma ( 1 - q _ { 0 } - q _ { 1 } ) } { k - k q _ { 0 } - \gamma q _ { 1 } } } & { k > \gamma , } \end{array} \right. } \\ & { \mathbb { E } \left[ Y ^ { T = 1 } \mid X \leq k , T = 1 \right] = \left\{ \alpha _ { 0 } + \beta _ { 0 } / 2 + \alpha _ { 1 } + \beta _ { 1 } / 2 \right. } & { k \leq \gamma , } \\ & { \mathbb { E } \left[ \alpha _ { 0 } + \beta _ { 0 } / 2 + \frac { ( \alpha _ { 1 } + \beta _ { 1 } / 2 ) \gamma ( q _ { 0 } + q _ { 1 } ) } { k q _ { 0 } + \gamma q _ { 1 } } \right.} & { k > \gamma , } \end{array}   \end{array}
$$

and hence,

$$
\hat { \tau } _ { L } ^ { \infty } = \left\{ \begin{array} { l l } { \beta _ { 0 } + \beta _ { 1 } } & { k \le \gamma , } \\ { \beta _ { 0 } + ( \alpha _ { 1 } - \beta _ { 1 } / 2 ) \frac { \gamma ( k - \gamma ) q _ { 1 } } { ( k q _ { 0 } + \gamma q _ { 1 } ) ( k - k q _ { 0 } - \gamma q _ { 1 } ) } + \beta _ { 1 } \left( 1 - \frac { ( k - \gamma ) q _ { 0 } } { k q _ { 0 } + \gamma q _ { 1 } } \right) } & { k > \gamma . } \end{array} \right.
$$

Moreover, we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ Y ^ { T = 0 } \mid X > k , T = 0 \right] = \left\{ \begin{array} { l l } { \alpha _ { 0 } - \beta _ { 0 } / 2 + \frac { ( \alpha _ { 1 } - \beta _ { 1 } / 2 ) ( \gamma - k ) ( 1 - q _ { 0 } - q _ { 1 } ) } { ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } } } & { k \leq \gamma , } \\ { \alpha _ { 0 } - \beta _ { 0 } / 2 } & { k > \gamma , } \end{array} \right. } \\ & { \mathbb { E } \left[ Y ^ { T = 1 } \mid X > k , T = 1 \right] = \left\{ \begin{array} { l l } { \alpha _ { 0 } + \beta _ { 0 } / 2 + \frac { ( \alpha _ { 1 } + \beta _ { 1 } / 2 ) ( \gamma - k ) ( q _ { 0 } + q _ { 1 } ) } { ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } } } & { k \leq \gamma , } \\ { \alpha _ { 0 } + \beta _ { 0 } / 2 } & { k > \gamma , } \end{array} \right. } \end{array}
$$

and hence,

$$
\hat { \tau } _ { R } ^ { \infty } = \left\{ \begin{array} { l l } { \beta _ { 0 } + ( \alpha _ { 1 } - \beta _ { 1 } / 2 ) \frac { ( \gamma - k ) ( 1 - \gamma ) q _ { 1 } } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) } } & { } \\ { + \beta _ { 1 } \left( 1 - \frac { ( 1 - \gamma ) q _ { 0 } } { ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } } \right) } & { k \le \gamma , } \\ { \beta _ { 0 } } & { k > \gamma . } \end{array} \right.
$$

Note that all terms in the denominators are either treatment propensities for some subset of samples, scaled versions of these, or a product of multiple such propensities. By Assumption A3, these are positive, so all terms are defined. □

Proof of Theorem 6. Use the notation from (35) and

$$
\begin{array} { r l } & { \hat { \tau } _ { \mathrm { p a r e n t } } ^ { \infty } : = \mathbb { E } \left[ Y ^ { T = 1 } \mid T = 1 \right] - \mathbb { E } \left[ Y ^ { T = 0 } \mid T = 0 \right] } \\ & { \quad \quad \quad = \beta _ { 0 } + \gamma ( \alpha _ { 1 } - \beta _ { 1 } / 2 ) \frac { ( 1 - \gamma ) q _ { 1 } } { ( q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) } + \gamma \beta _ { 1 } \left( 1 + \frac { ( 1 - \gamma ) q _ { 1 } } { q _ { 0 } + \gamma q _ { 1 } } \right) , } \end{array}
$$

as calculated in Lemma 8. Note that we can write

$$
\begin{array} { r l } & { \mathrm { H e t } ^ { \pm } ( k ) = \sqrt { k ( 1 - k ) } \cdot ( \hat { \tau } _ { L } ^ { \infty } - \hat { \tau } _ { R } ^ { \infty } ) , } \\ & { \mathrm { B i a s } ^ { \pm } ( k ) = \hat { \tau } _ { \mathrm { p a r e n t } } ^ { \infty } - k \hat { \tau } _ { L } ^ { \infty } - ( 1 - k ) \hat { \tau } _ { R } ^ { \infty } . } \end{array}
$$

Further note, that $\alpha _ { 0 }$ does not appear in any of $\hat { \tau } _ { \mathrm { p a r e n t } } ^ { \infty } , \hat { \tau } _ { L } ^ { \infty } , \hat { \tau } _ { R } ^ { \infty }$ , so it does not have any efect on $\operatorname { H e t } ^ { \pm } ( k )$ or $\mathrm { B i a s } ^ { \pm } ( k )$ . Additionally, $\beta _ { 0 }$ appears in all three treatment efect estimates, but always as simple summand. So changing it does not have any efect on the diferences $\hat { \tau } _ { L } ^ { \infty } - \hat { \tau } _ { R } ^ { \infty }$ and $\hat { \tau } _ { \mathrm { p a r e n t } } ^ { \infty } - k \hat { \tau } _ { L } ^ { \infty } - ( 1 - k ) \hat { \tau } _ { R } ^ { \infty }$ , which define ${ \mathrm { H e t } } ^ { \pm } ( k )$ and $\operatorname { B i a s } ^ { \pm } ( k )$ . To simplify notation, we will therefore, without loss of generality, set all of $\alpha _ { 0 } , \alpha _ { 0 } ^ { \prime } , \alpha _ { 0 } ^ { \prime \prime } , \beta _ { 0 } , \beta _ { 0 } ^ { \prime } , \beta _ { 0 } ^ { \prime \prime }$ zero for the following analysis of the signed splitting criteria.

We proof the two statements (for bias and for heterogeneity) of the theorem separately.

Bias After setting $\alpha _ { 0 } ^ { \prime } = \beta _ { 0 } ^ { \prime } = 0$ , as explained above, the bias model is given by

$$
\begin{array} { c } { { \mu ^ { B } ( x ) = \alpha _ { 1 } ^ { \prime } \cdot { \bf 1 } _ { x \leq \gamma } , } } \\ { { \tau ^ { B } ( x ) = 0 , } } \\ { { p ( x ) = q _ { 0 } + q _ { 1 } \cdot { \bf 1 } _ { x \leq \gamma } . } } \end{array}
$$

In this model, the estimated treatment efect tends to

$$
\hat { \tau } _ { \mathrm { p a r e n t } } ^ { \infty } = \gamma \alpha _ { 1 } ^ { \prime } \frac { ( 1 - \gamma ) q _ { 1 } } { ( q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) } ,
$$

while the true treatment efect is constant zero: $\tau ^ { B } = 0$ In the child nodes for a split at $k$ , the estimates are

$$
\begin{array} { r l } & { \hat { \tau } _ { L } ^ { \infty } = \left\{ \begin{array} { l l } { 0 } & { k \le \gamma , } \\ { \alpha _ { 1 } ^ { \prime } \frac { \gamma \left( k - \gamma \right) q _ { 1 } } { \left( k q _ { 0 } + \gamma q _ { 1 } \right) \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) } } & { k > \gamma , } \end{array} \right. } \\ & { \hat { \tau } _ { R } ^ { \infty } = \left\{ \begin{array} { l l } { \alpha _ { 1 } ^ { \prime } \frac { \left( \gamma - k \right) \left( 1 - \gamma \right) q _ { 1 } } { \left( \left( 1 - k \right) q _ { 0 } + \left( \gamma - k \right) q _ { 1 } \right) \left( \left( 1 - k \right) \left( 1 - q _ { 0 } \right) - \left( \gamma - k \right) q _ { 1 } \right) } } & { k \le \gamma , } \\ { 0 } & { k > \gamma . } \end{array} \right. } \end{array}
$$

Therefore, the population signed splitting criteria are given by

$$
\begin{array} { r l } & { \mathrm { H e t } _ { B } ^ { \pm } ( k ) = \sqrt { k ( 1 - k ) } ( \hat { \tau } _ { L } ^ { \infty } - \hat { \tau } _ { R } ^ { \infty } ) , } \\ & { \quad \quad \quad = \{ \sqrt { k ( 1 - k ) } ( 0 - \alpha _ { 1 } ^ { \prime } \frac { ( \gamma - k ) ( 1 - \gamma ) q _ { 1 } } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) } ) \quad k \le \gamma , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad k > \gamma , } \\ & { \quad \quad = \sqrt { - \sqrt { k ( 1 - k ) } } \alpha _ { 1 } ^ { \prime } \frac { \gamma ( k - \gamma ) q _ { 1 } } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) } \quad k \le \gamma , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad k > \gamma , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad k > \gamma , } \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { B i a s } _ { B } ^ { \pm } ( k ) = \hat { \tau } _ { \mathrm { p a r e n t } } ^ { \infty } - k \hat { \tau } _ { L } ^ { \infty } - ( 1 - k ) \hat { \tau } _ { R } ^ { \infty } } \\ & { \quad \quad \quad = \{ \gamma \alpha _ { 1 } ^ { \prime } \frac { ( 1 - \gamma ) q _ { 1 } } { ( q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) } - k \cdot 0 - ( 1 - k ) \alpha _ { 1 } ^ { \prime } \frac { ( \gamma - k ) ( 1 - \gamma ) q _ { 1 } } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) } \quad k \le \gamma , } \\ & { \quad \quad \quad \quad \gamma \alpha _ { 1 } ^ { \prime } \frac { ( 1 - \gamma ) q _ { 1 } } { ( q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) } - k \alpha _ { 1 } ^ { \prime } \frac { \gamma ( k - \gamma ) q _ { 1 } } { ( k q _ { 0 } + \gamma q _ { 1 } ) ( k - k q _ { 0 } - \gamma q _ { 1 } ) } - ( 1 - k ) \cdot 0 \quad k > \gamma , } \\ & { \quad \quad \quad = \{ \alpha _ { 1 } ^ { \prime } ( \frac { \gamma ( 1 - \gamma ) q _ { 1 } } { ( q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) } - ( 1 - k ) \frac { ( \gamma - k ) ( 1 - \gamma ) q _ { 1 } } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) } ) \quad k \le \gamma ,  } \\ &  \quad \quad \quad \quad \quad \quad \quad  \quad \quad \quad \alpha _ { 1 } ^ { \prime } ( \frac { \gamma ( 1 - \gamma ) q _ { 1 } }  \end{array}
$$

Note that Het $\displaystyle { \frac { \pm } { B } ( k ) }$ and $\mathrm { B i a s } _ { B } ^ { \pm } ( k )$ are both continuous and at $k = \gamma$ we have

$$
\begin{array} { r l } & { \mathrm { H e t } _ { B } ^ { \pm } ( \gamma ) = 0 } \\ & { \mathrm { B i a s } _ { B } ^ { \pm } ( \gamma ) = \alpha _ { 1 } ^ { \prime } \cfrac { \gamma ( 1 - \gamma ) q _ { 1 } } { ( q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) } . } \end{array}
$$

Recall from Theorem 5 that $\mu ( x ) = \mu ^ { B } ( x ) + \mu ^ { H } ( x )$ and $\tau ( x ) = \tau ^ { B } ( x ) + \tau ^ { H } ( x )$ and hence,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \tau ( X ) \right] = \mathbb { E } \left[ \tau ^ { B } ( X ) \right] + \mathbb { E } \left[ \tau ^ { H } ( X ) \right] , } \\ & { \mathbb { E } \left[ \mu ( X ) + \frac { 1 } { 2 } \tau ( X ) \mid T = 1 \right] = \mathbb { E } \left[ \mu ^ { B } ( X ) + \frac { 1 } { 2 } \tau ^ { B } ( X ) \mid T = 1 \right] + \mathbb { E } \left[ \mu ^ { H } ( X ) + \frac { 1 } { 2 } \tau ^ { H } ( X ) \mid T = 1 \right] , } \\ & { \mathbb { E } \left[ \mu ( X ) - \frac { 1 } { 2 } \tau ( X ) \mid T = 0 \right] = \mathbb { E } \left[ \mu ^ { B } ( X ) - \frac { 1 } { 2 } \tau ^ { B } ( X ) \mid T = 0 \right] + \mathbb { E } \left[ \mu ^ { H } ( X ) - \frac { 1 } { 2 } \tau ^ { H } ( X ) \mid T = 0 \right] . } \end{array}
$$

From (30) it follows that

$$
\begin{array} { r l } & { \mathbb { E } \left[ \mu ( X ) + \displaystyle \frac { 1 } { 2 } \tau ( X ) \mid T = 1 \right] - \mathbb { E } \left[ \mu ( X ) - \displaystyle \frac { 1 } { 2 } \tau ( X ) \mid T = 0 \right] - \mathbb { E } \left[ \tau ( X ) \right] } \\ & { \quad = \mathbb { E } \left[ \mu ^ { B } ( X ) + \displaystyle \frac { 1 } { 2 } \tau ^ { B } ( X ) \mid T = 1 \right] - \mathbb { E } \left[ \mu ^ { B } ( X ) - \displaystyle \frac { 1 } { 2 } \tau ^ { B } ( X ) \mid T = 0 \right] - \mathbb { E } \left[ \tau ^ { B } ( X ) \right] . } \end{array}
$$

By Lemma 7 the right-hand side of this equation is equal to $\mathrm { B i a s } _ { B } ^ { \pm } ( \gamma )$ . Because we assume that the diference-in-means estimator of the original model is biased, i.e., E $\textstyle { \left[ { \mu ( X ) + { \frac { 1 } { 2 } } \tau ( X ) \mid T = 1 } \right] - \operatorname { \mathbb { E } } \left[ \mu ( X ) - { \frac { 1 } { 2 } } \tau ( X ) \mid T = 0 \right] - \operatorname { \mathbb { E } } \left[ \tau ( X ) \right] \neq 0 }$ , it follows that $\mathrm { B i a s } _ { B } ^ { \pm } ( \gamma ) \neq 0$ along with $\alpha _ { 1 } ^ { \prime } \neq 0$ and $q _ { 1 } \neq 0$ . Moreover, as $\mathrm { B i a s } _ { B } ^ { \pm } ( 0 ) = \mathrm { B i a s } _ { B } ^ { \pm } ( 1 ) = 0$ , it follows that $\mathrm { B i a s } _ { B } ^ { \pm }$ is non-constant and we will show that it has a global extremum at $k = \gamma$

Considering the derivative of $\mathrm { B i a s } _ { B } ^ { \pm }$ for $k > \gamma$ yields

$$
\begin{array} { r l } & { \frac { \mathrm { d } } { \mathrm { d } k } \alpha _ { 1 } ^ { \prime } \left( \frac { \gamma \left( 1 - \gamma \right) q _ { 1 } } { \left( q _ { 0 } + \gamma q _ { 1 } \right) \left( 1 - q _ { 0 } - \gamma q _ { 1 } \right) } - k \frac { \gamma \left( k - \gamma \right) q _ { 1 } } { \left( k q _ { 0 } + \gamma q _ { 1 } \right) \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) } \right) } \\ & { = - \alpha _ { 1 } ^ { \prime } \left( \frac { \left( \gamma \left( k - \gamma \right) q _ { 1 } + k \gamma q _ { 1 } \right) \left( k q _ { 0 } + \gamma q _ { 1 } \right) \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) } { \left( k q _ { 0 } + \gamma q _ { 1 } \right) ^ { 2 } \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) ^ { 2 } } \right. } \\ & { \quad \left. - \frac { k \gamma \left( k - \gamma \right) q _ { 1 } \left( q _ { 0 } \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) + \left( k q _ { 0 } + \gamma q _ { 1 } \right) \left( 1 - q _ { 0 } \right) \right) } { \left( k q _ { 0 } + \gamma q _ { 1 } \right) ^ { 2 } \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) ^ { 2 } } \right) } \\ & { = - \alpha _ { 1 } ^ { \prime } \gamma ^ { 2 } q _ { 1 } \frac { \left( q _ { 0 } ^ { 2 } + 2 q _ { 0 } q _ { 1 } - q _ { 0 } - q _ { 1 } \right) k ^ { 2 } + 2 \gamma q _ { 1 } ^ { 2 } k - \gamma ^ { 2 } q _ { 1 } ^ { 2 } } { \left( k q _ { 0 } + \gamma q _ { 1 } \right) ^ { 2 } \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) ^ { 2 } } . } \end{array}
$$

Note that $\mathrm { B i a s } _ { B } ^ { \pm }$ is contentiously diferentiable in $( \gamma , 1 )$ . Hence, for a potential extremum k of $\mathrm { B i a s } _ { B } ^ { \pm }$ in $( \gamma , 1 )$ , we would need

$$
\begin{array} { r } { 0 = - \alpha _ { 1 } ^ { \prime } \gamma ^ { 2 } q _ { 1 } \frac { ( q _ { 0 } ^ { 2 } + 2 q _ { 0 } q _ { 1 } - q _ { 0 } - q _ { 1 } ) k ^ { 2 } + 2 \gamma q _ { 1 } ^ { 2 } k - \gamma ^ { 2 } q _ { 1 } ^ { 2 } } { ( k q _ { 0 } + \gamma q _ { 1 } ) ^ { 2 } ( k - k q _ { 0 } - \gamma q _ { 1 } ) ^ { 2 } } } \\ { \iff 0 = ( q _ { 0 } ^ { 2 } + 2 q _ { 0 } q _ { 1 } - q _ { 0 } - q _ { 1 } ) k ^ { 2 } + 2 \gamma q _ { 1 } ^ { 2 } k - \gamma ^ { 2 } q _ { 1 } ^ { 2 } } \\ { \iff k = \frac { - 2 \gamma q _ { 1 } ^ { 2 } \pm \sqrt { ( 2 \gamma q _ { 1 } ^ { 2 } ) ^ { 2 } - 4 ( q _ { 0 } ^ { 2 } + 2 q _ { 0 } q _ { 1 } - q _ { 0 } - q _ { 1 } ) ( - \gamma ^ { 2 } q _ { 1 } ^ { 2 } ) } } { 2 ( q _ { 0 } ^ { 2 } + 2 q _ { 0 } q _ { 1 } - q _ { 0 } - q _ { 1 } ) } , } \end{array}
$$

but the discriminant is

$$
\begin{array} { r l } & { ( 2 \gamma q _ { 1 } ^ { 2 } ) ^ { 2 } - 4 ( q _ { 0 } ^ { 2 } + 2 q _ { 0 } q _ { 1 } - q _ { 0 } - q _ { 1 } ) ( - \gamma ^ { 2 } q _ { 1 } ^ { 2 } ) } \\ & { = 4 \gamma ^ { 2 } q _ { 1 } ^ { 4 } + 4 \gamma ^ { 2 } q _ { 0 } ^ { 2 } q _ { 1 } ^ { 2 } + 8 \gamma ^ { 2 } q _ { 0 } q _ { 1 } ^ { 3 } - 4 \gamma ^ { 2 } q _ { 0 } q _ { 1 } ^ { 2 } - 4 \gamma ^ { 2 } q _ { 1 } ^ { 3 } } \\ & { = - 4 \gamma ^ { 2 } q _ { 1 } ^ { 2 } ( q _ { 0 } + q _ { 1 } ) ( 1 - q _ { 0 } - q _ { 1 } ) < 0 . } \end{array}
$$

Therefore, there are no local extrema for $k \in ( \gamma , 1 )$ . Similarly, there are also no local extrema within $( 0 , \gamma )$ . As $\mathrm { B i a s } _ { B } ^ { \pm }$ is continuous and non-constant with $\mathrm { B i a s } _ { B } ^ { \pm } ( k ) = 0$ at $k = 0$ and $k = 1 , \mathrm { B i a s } _ { B } ^ { \pm } ( \gamma )$ must be a global extremum.

Heterogeneity After setting $\alpha _ { 0 } ^ { \prime \prime } = \beta _ { 0 } ^ { \prime \prime } = 0$ , as explained above, the heterogeneity model is given by

$$
\begin{array} { c } { { \mu ^ { H } ( x ) = \alpha _ { 1 } ^ { \prime \prime } \cdot { \bf 1 } _ { x \leq \gamma } , } } \\ { { \tau ^ { H } ( x ) = \beta _ { 1 } ^ { \prime \prime } \cdot { \bf 1 } _ { x \leq \gamma } , } } \\ { { p ( x ) = q _ { 0 } + q _ { 1 } \cdot { \bf 1 } _ { x \leq \gamma } , } } \end{array}
$$

with $\alpha _ { 1 } ^ { \prime \prime } = - ( 1 / 2 - q _ { 0 } - \gamma q _ { 1 } ) \beta _ { 1 } ^ { \prime \prime }$ , such that the bias calculated in Lemma 7 is 0.

For this model, the treatment efect obtained from the diference-in-means estimator tends to the correct average treatment efect, i.e., $\hat { \tau } _ { \mathrm { p a r e n t } } ^ { \infty } = \mathbb { E } \left[ \tau ( X ) \right] = \gamma \beta _ { 1 } ^ { \prime \prime }$ and the estimates in the child nodes converge to

$$
\begin{array} { r l } & { \hat { \tau } _ { L } ^ { \infty } = \left\{ \begin{array} { l l } { \beta _ { 1 } ^ { \prime \prime } } & { k \le \gamma , } \\ { \beta _ { 1 } ^ { \prime \prime } \left( 1 - ( k - \gamma ) \frac { ( k q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } ) - ( q _ { 0 } + \gamma q _ { 1 } ) \gamma q _ { 1 } } { ( k q _ { 0 } + \gamma q _ { 1 } ) ( k - k q _ { 0 } - \gamma q _ { 1 } ) } \right) } & { k > \gamma , } \end{array} \right. } \\ & { \hat { \tau } _ { R } ^ { \infty } = \left\{ \begin{array} { l l } { \beta _ { 1 } ^ { \prime \prime } ( \gamma - k ) \frac { ( q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) - k ( q _ { 0 } + q _ { 1 } ) ( 1 - q _ { 0 } - q _ { 1 } ) } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) } } & { k \le \gamma , } \\ { 0 } & { k > \gamma , } \end{array} \right. } \end{array}
$$

which can be calculated by using $\alpha _ { 1 } ^ { \prime \prime } = - ( 1 / 2 - q _ { 0 } - \gamma q _ { 1 } ) \beta _ { 1 } ^ { \prime \prime }$ in the formulas given by Lemma 8. The population signed splitting criteria evaluate to

$$
\begin{array} { r l } & { \mathrm { H e t } _ { H } ^ { \pm } ( k ) = \sqrt { k ( 1 - k ) } ( \hat { \tau } _ { L } ^ { \infty } - \hat { \tau } _ { R } ^ { \infty } ) , } \\ & { \qquad = \{ \sqrt { k ( 1 - k ) } ( \beta _ { 1 } ^ { \eta ^ { \alpha } } - \beta _ { 1 } ^ { \eta ^ { \prime } } ( \gamma - k ) \frac { ( \rho _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) - k ( q _ { 0 } + q _ { 1 } ) ( 1 - q _ { 0 } - q _ { 1 } ) } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) } ) \quad k \leq \gamma , } \\ & { \qquad = \{ \sqrt { k ( 1 - k ) } ( \beta _ { 1 } ^ { \eta ^ { \prime } } ( 1 - ( k - \gamma ) \frac { ( k q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } ) - ( q _ { 0 } + \gamma q _ { 1 } ) \gamma q _ { 1 } } { ( k q _ { 0 } + \gamma q _ { 1 } ) ( k - k q _ { 0 } - \gamma q _ { 1 } ) } )  \qquad k > \gamma , } \\ & { \qquad = \{ \beta _ { 1 } ^ { \eta ^ { \prime } } \sqrt { k ( 1 - k ) } ( 1 - ( \gamma - k ) \frac { ( q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) - k ( q _ { 0 } + q _ { 1 } ) ( 1 - q _ { 0 } - q _ { 1 } ) } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) } )  \quad k \leq \gamma , } \\ &  \qquad = \{ \beta _ { 1 } ^ { \eta ^ { \prime } } \sqrt { k ( 1 - k ) } ( 1 - ( k - \gamma ) \frac  ( k q _ { 0 } + \gamma \end{array}
$$

$$
\begin{array} { r l } & { \mathrm { B i a s } _ { H } ^ { \pm } ( k ) = \hat { \tau } _ { \mathrm { p a r e n t } } ^ { \infty } - k \hat { \tau } _ { L } ^ { \infty } - ( 1 - k ) \hat { \tau } _ { R } ^ { \infty } } \\ & { \qquad = \{ \begin{array} { l l } { \gamma \beta _ { 1 } ^ { \prime \prime } - k \beta _ { 1 } ^ { \prime \prime } - ( 1 - k ) \beta _ { 1 } ^ { \prime \prime } ( \gamma - k ) \frac { ( q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) - k ( q _ { 0 } + q _ { 1 } ) ( 1 - q _ { 0 } - q _ { 1 } ) } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) q _ { 0 } - ( \gamma - k ) q _ { 1 } ) } } & { k \leq \gamma , } \\ { \gamma \beta _ { 1 } ^ { \prime \prime } - k \beta _ { 1 } ^ { \prime \prime } ( 1 - ( k - \gamma ) \frac { ( k q _ { 0 } + \gamma q _ { 1 } ) ( 1 - q _ { 0 } ) - ( q _ { 0 } + \gamma q _ { 1 } ) \gamma q _ { 1 } } { ( k q _ { 0 } + \gamma q _ { 1 } ) ( k - k q _ { 0 } - \gamma q _ { 1 } ) } - ( 1 - k ) \cdot 0 } & { k > \gamma , } \end{array}  } \\ &  \qquad = \{ \begin{array} { l l } { - \beta _ { 1 } ^ { \prime \prime } ( \gamma - k ) \frac { k ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ^ { 2 } } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) } } & { k \leq \gamma , } \\  \beta _ { 1 } ^ { \prime \prime } ( k - \gamma ) \frac { ( 1 - k ) \gamma ^ { 2 } q _ { 1 } ^ { 2 } }  ( k q _ { 0 } + \gamma q _ { 1 } ) ( k - k q _ { 0 } - \end{array} \end{array}
$$

Note that He $; _ { H } ^ { \pm } ( k )$ and $\mathrm { B i a s } _ { H } ^ { \pm } ( k )$ are both continuous and at $k = \gamma$ we have

$$
\begin{array} { r l } & { \mathrm { H e t } _ { H } ^ { \pm } ( \gamma ) = \beta _ { 1 } ^ { \prime \prime } \sqrt { \gamma ( 1 - \gamma ) } , } \\ & { \mathrm { B i a s } _ { H } ^ { \pm } ( \gamma ) = 0 . } \end{array}
$$

If the treatment efect of the original model $\tau ( x )$ is non-constant, it follows that $0 < \gamma < 1$ ， $\beta _ { 1 } \neq 0$ , and $\beta _ { 1 } ^ { \prime \prime } \neq 0$ . This implies $\mathrm { H e t } _ { H } ^ { \pm } ( \gamma ) \neq 0$ and, as $\mathrm { H e t } _ { H } ^ { \pm } ( 0 ) = \mathrm { H e t } _ { H } ^ { \pm } ( 1 ) = 0$ , that Het $: \frac { \pm } { H }$ non-constant. We will now show that $\operatorname { H e t } _ { H } ^ { \pm } ( \gamma )$ is a local extremum.

Considering the derivative of $\mathrm { H e t } _ { H } ^ { \pm }$ for $k < \gamma$ yields

$$
\begin{array} { r l } { \frac { \mathrm { d } } { \mathrm { d } k } \mathrm { t e t } \frac { \mathrm { d } } { \mathrm { d } k } ( k ) = \beta _ { 1 } ^ { n } \sqrt { k ( 1 - k ) } } & { } \\ & { \cdot ( \frac { 1 - 2 k } { 2 k ( 1 - k ) } + \frac { ( ( \mathsf { f o n } + \gamma q ) ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) - k ( q _ { 0 } + q _ { 1 } ) ( 1 - q _ { 0 } - q _ { 1 } ) } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) } ) } \\ & { - \beta _ { 1 } ^ { n } \sqrt { k ( 1 - k ) } ( \gamma - k ) } \\ & { \cdot ( - \frac { ( \mathsf { f o n } + \gamma q ) } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( ( 1 - k ) ( 1 - q _ { 0 } - q _ { 1 } ) ) }  } \\ & { \cdot  ( ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ( \{ ( 1 - k ) ( 1 - q _ { 0 } - q _ { 1 } ) } - ( \gamma - k ) q _ { 1 } )    \\ & { \quad   + ( \mathsf { g n } + q _ { 1 } ) \frac { ( ( \mathsf { f o n } + \gamma q _ { 1 } ) ) - k ( q _ { 0 } - \gamma q _ { 1 } ) - k ( q _ { 0 } + q _ { 1 } ) ( 1 - q _ { 0 } - q _ { 1 } ) } { ( ( 1 - k ) q _ { 0 } + ( \gamma - k ) q _ { 1 } ) ^ { 2 } ( ( 1 - k ) ( 1 - q _ { 0 } ) - ( \gamma - k ) q _ { 1 } ) }   } \\ &  \quad   + ( 1 - q _ { 0 } - q _ { 1 } ) \frac  ( ( \mathsf { f o n } + \gamma q _ { 1 } ) ) ( 1 - q _ { 0 } - \gamma q _ { 1 } ) - k ( q _ { 0 } + q _ { 1 } ) (  \end{array}
$$

and for $k > \gamma$

$$
\begin{array} { r l } & { \frac { \mathrm { d } } { \mathrm { d } k } \mathrm { H e t } _ { H } ^ { \pm } ( k ) = \beta _ { 1 } ^ { \prime \prime } \sqrt { k ( 1 - k ) } \left( \frac { 1 - 2 k } { 2 k ( 1 - k ) } - \frac { \left( k q _ { 0 } + \gamma q _ { 1 } \right) ( 1 - q _ { 0 } ) - \left( q _ { 0 } + \gamma q _ { 1 } \right) \gamma q _ { 1 } } { \left( k q _ { 0 } + \gamma q _ { 1 } \right) \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) } \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad + \beta _ { 1 } ^ { \prime \prime } \sqrt { k ( 1 - k ) } ( k - \gamma ) \Big ( - \frac { q _ { 0 } ( 1 - q _ { 0 } ) } { \left( k q _ { 0 } + \gamma q _ { 1 } \right) \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad + q _ { 0 } \frac { \left( k q _ { 0 } + \gamma q _ { 1 } \right) ( 1 - q _ { 0 } ) - \left( q _ { 0 } + \gamma q _ { 1 } \right) \gamma q _ { 1 } } { \left( k q _ { 0 } + \gamma q _ { 1 } \right) ^ { 2 } \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad + \left( 1 - q _ { 0 } \right) \frac { \left( k q _ { 0 } + \gamma q _ { 1 } \right) ( 1 - q _ { 0 } ) - \left( q _ { 0 } + \gamma q _ { 1 } \right) \gamma q _ { 1 } } { \left( k q _ { 0 } + \gamma q _ { 1 } \right) \left( k - k q _ { 0 } - \gamma q _ { 1 } \right) ^ { 2 } } } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad  \end{array}
$$

Note that $\operatorname { H e t } _ { H } ^ { \pm } ( k )$ is continuously diferentiable in $( 0 , \gamma )$ and in $( \gamma , 1 )$ . The limits of these derivatives at γ are

$$
\begin{array} { l } { { \displaystyle \operatorname* { l i m } _ { k \to \gamma ^ { - } } \frac { \mathrm { d } } { \mathrm { d } k } \mathrm { H e t } _ { H } ^ { \pm } ( k ) = \beta _ { 1 } ^ { \prime \prime } \sqrt { \gamma ( 1 - \gamma ) } \frac { q _ { 0 } ( 1 - q _ { 0 } ) + 2 \gamma ^ { 2 } q _ { 1 } ^ { 2 } } { 2 \gamma ( 1 - \gamma ) q _ { 0 } ( 1 - q _ { 0 } ) } } } \\ { { \displaystyle \operatorname* { l i m } _ { k \to \gamma ^ { + } } \frac { \mathrm { d } } { \mathrm { d } k } \mathrm { H e t } _ { H } ^ { \pm } ( k ) = - \beta _ { 1 } ^ { \prime \prime } \sqrt { \gamma ( 1 - \gamma ) } \frac { ( 1 - q _ { 0 } - q _ { 1 } ) ( q _ { 0 } + q _ { 1 } ) + 2 ( 1 - \gamma ) ^ { 2 } q _ { 1 } ^ { 2 } } { 2 \gamma ( 1 - \gamma ) ( q _ { 0 } + q _ { 1 } ) ( 1 - q _ { 0 } - q _ { 1 } ) } } } \end{array}
$$

All of $\gamma , ( 1 - \gamma ) , q _ { 0 } , ( 1 - q _ { 0 } ) , ( q _ { 0 } + q _ { 1 } )$ and $( 1 - q _ { 0 } - q _ { 1 } )$ are assumed to be positive by Assumptions A3 and $\gamma ~ \in ~ ( 0 , 1 )$ , as well as $q _ { 1 } ^ { 2 } \geq 0$ . Therefore, we have sgn $\begin{array} { r } { ( \operatorname* { l i m } _ { k \to \gamma ^ { - } } \frac { \mathrm { d } } { \mathrm { d } k } \mathrm { H e t } _ { H } ^ { \pm } ( k ) ) = \mathrm { s g n } ( \beta _ { 1 } ^ { \prime \prime } ) } \end{array}$ and sgn $\begin{array} { r } { ( \operatorname* { l i m } _ { k \to \gamma ^ { + } } \frac { \mathrm { d } } { \mathrm { d } k } \mathrm { H e t } _ { H } ^ { \pm } ( k ) ) = - \mathrm { s g n } ( \beta _ { 1 } ^ { \prime \prime } ) } \end{array}$ . Also note that sgn $\widetilde { ( } \mathrm { H e t } _ { H } ^ { \pm } ( \gamma ) ) = \mathrm { s g n } ( \beta _ { 1 } ^ { \prime \prime } )$ . Together, it follows that $\operatorname { H e t } _ { H } ^ { \pm } ( \gamma )$ has a local extremum at $k = \gamma$ □