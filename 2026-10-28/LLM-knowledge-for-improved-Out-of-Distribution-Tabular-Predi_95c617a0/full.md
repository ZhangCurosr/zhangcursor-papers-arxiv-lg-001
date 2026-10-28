# LLM knowledge for improved Out-of-Distribution Tabular Prediction – A Cross-Validated Regularisation Approach

Sølve Langseth<sup>1</sup>, Kristofer Herland Hellton<sup>1</sup>, and Hugo L. Hammer<sup>∗1,2,3</sup>

<sup>1</sup>Department of Computer Science, Oslo Metropolitan University, Oslo, Norway <sup>2</sup>Department of Holistic Systems, SimulaMet, Oslo, Norway

<sup>3</sup>Department of Plastic and Reconstructive Surgery, Oslo University Hospital, Oslo, Norway

## Abstract

Prediction models degrade when the deployment population difers from the one used to train them, and training data cannot say which feature–response relationships will transfer. Expert elicitation is slow and unscalable. We ask whether coeficient priors elicited from large language models (LLMs) improve out-of-distribution (OOD) prediction as the shrinkage target of a regularised regression instead of zero, across five tabular datasets split on real covariates, selecting providers and prior structures by source-domain cross-validation alone. The priors help on four of five datasets, by up to 7.9% OOD mean squared error, and ten of twenty-four prior-informed results resolve the mean improvement at conventional confidence; single-deployment intervals are much wider. On classification the gain is in probabilistic accuracy, not ranking, and control tests with corrupted priors attribute the benefit to featurespecific content rather than scale. Two findings temper this. Variation across candidate priors exceeds the average benefit, yet cross-validation margins fall below their own noise, so selection avoids the worst prior rather than finding the best. On three datasets the prior improves in-domain accuracy as much as out-of-domain, making it better regularisation than shift correction. Only where the training data are large enough that the in-domain gain disappears does elicited knowledge demonstrably improve transfer. Code, priors and results: https://github.com/hugohammer/llm-priors-ood.

Keywords: prior elicitation, large language models, distribution shift, out-of-distribution prediction, regularised regression, model selection

## 1 Introduction

Prediction models are usually developed under the assumption that training and evaluation data are drawn from the same underlying distribution. Under this assumption, held-out test performance is a reliable proxy for the data a model will meet after deployment. In applied settings the assumption is often unrealistic, and performance can deteriorate when, for instance, the population encountered at prediction time difers from the one used for development [1]. As an example, a clinical prediction model trained at one hospital may perform worse at another because patient demographics, measurement routines, disease prevalence or treatment practices might difer.

Such changes are described as dataset or domain shift. We use domain shift for settings where the deployment population difers from the source population used for model development, the case most relevant here; it is closely related to covariate shift, where the feature distribution changes while the conditional outcome mechanism is assumed stable [2, 3, 1].

Despite extensive work, there is no generally reliable remedy. Standard validation can fail precisely because the validation set is drawn from the source distribution. The dominant practical response is ensembling: deep ensembles [4] train several networks from diferent initialisations and average their predictive distributions, producing better-calibrated uncertainty than more involved Bayesian methods, with ensemble variance rising on inputs far from the training distribution. Ensembles have limitations, however. They require fitting many models, are expensive, and when built from complex learners are hard to interpret. More fundamentally, ensemble disagreement diagnoses predictive uncertainty, but does not reduce it: it tells the user how confident the model is, not how to make a better prediction.

One of the few principled ways to improve predictions when training data are limited or unrepresentative is to bring in information from outside the training set: external knowledge about which features should matter, in what direction, and with what plausible magnitude. Recent evidence that tabular foundation models excel on small-to-medium IID data but are still beaten by conventional models on non-IID problems [5] suggests that scale alone does not solve domain shift, and that simple, inspectable models guided by external knowledge remain worth studying.

Useful external information often exists outside the training data, in previous studies, biological pathways, clinical expertise or established domain theory [6, 7]. Appropriately encoded, it can improve estimation and prediction where observed data are limited, noisy or unrepresentative [8]. The established framework is Bayesian modelling, where prior beliefs are expressed as a distribution over parameters and updated with data [9, 8]. A closely related approach is regularisation, where the estimator is penalised so that estimates shrink toward preferred values [10]. Ridge regression shrinks coeficients toward zero, which is exactly a generic prior belief that efects are small unless the data say otherwise [11, 9].

This equivalence makes prior-informed regularisation a natural way to combine external knowledge with predictive modelling: instead of shrinking every coeficient toward the same uninformative target of zero, one may shrink toward values suggested by domain knowledge. Under domain shift this may be helpful, because source-domain data can overemphasise relationships that do not transfer, while domain knowledge may encode more stable information about plausible efects.

Obtaining that knowledge, however, is the bottleneck. O’Hagan [7] describes expert elicitation as a structured and demanding process of translating substantive knowledge into probability distributions. Domain experts are usually not statisticians, such that the elicitor typically must guide them through uncertainty, probability and numerical efect sizes. Elicitation is therefore slow, hard to scale, and ofers no guarantee of a useful prior.

These constraints motivate interest in Large Language Models (LLMs) as an alternative. LLMs are trained on large text corpora and adapted to respond to natural-language prompts [12, 13], encoding broad statistical associations across many domains [13, 14]. They can be queried repeatedly and cheaply, applied across domains and feature sets, and constrained to emit structured numerical output. In a randomised trial of diagnostic reasoning the language model alone scored higher than physicians using conventional resources, even though giving physicians access to it did not improve their own reasoning [15], and LLM-derived priors have already been reported to be preferable to expert elicitation in an applied Bayesian survival analysis [16]. This does not make LLMs substitutes for human experts, but it does suggest they contain domain information worth eliciting.

The caveats are real. LLMs hallucinate, are sensitive to prompt wording, and reflect biases in their training data [17, 18, 19, 20]. These matter especially when the output is a numerical coeficient, since errors in sign, scale or uncertainty propagate directly into the fitted model.

We address three questions. First, can prior means estimated from repeated LLM coeficient samples improve OOD performance for regularised linear and logistic regression, relative to standard shrinkage toward zero? Second, does estimating prior uncertainty from the empirical distribution of LLM samples (as feature-specific variances or full covariance) help beyond the prior mean alone? Third, under what conditions do LLM priors help, do nothing, or harm? Our contributions are:

1. An evaluation of LLM-elicited coeficient priors under domain splits defined by real covariates across five datasets, with a cross-provider, cross-family selection pipeline driven only by source-domain data.

2. Placebo priors: priors re-indexed onto permuted features, and priors with randomly flipped signs, both preserving scale and correlation structure, establishing that the benefit derives from the priors’ feature-specific content and not from shrinking toward an arbitrary nonzero target.

3. A methodological negative result: Comparing classifiers using fixed decision thresholds (e.g., computing $F _ { 1 }$ at 0.5) substantially overstates performance gains whenever the domain split alters class balance. The reliable benefit of prior-informed regularisation lies in probabilistic calibration and proper scoring rules rather than threshold-dependent metrics [21].

4. A separation of two aspects of the benefit: on three of five datasets the prior improves indomain accuracy as much as out-of-domain accuracy and is therefore better regularisation, while on the one dataset with enough training data for the in-domain gain to disappear it improves transfer specifically.

5. An assessment of the selection step itself: depending on the LLM variant and prior family, an elicited prior can either improve or degrade out-of-domain predictive performance relative to the zero-centred baseline. Yet the cross-validation margins separating candidates are smaller than their fold-level noise, so the selection procedure is better at avoiding poor candidates than at identifying the best one.

## 2 Related work

The use of LLMs as a source of domain information for predictive modelling is recent but developing quickly. Existing work uses LLMs directly as predictors, to construct or modify features, or to supply prior information for statistical models.

This study builds on Riegler et al. [22], who asked several LLMs to suggest informative Gaussian priors for coeficients in linear and logistic regression. They found that LLMs often identify the correct direction of a feature efect but calibrate prior width poorly, exhibiting both over- and under-confidence depending on the variable and the provider. That finding motivates our estimator design directly, and Riegler et al. [22] explicitly name out-of-distribution (OOD) prediction as a future use case, which is the focus of this paper.

Capstick et al. [23] propose AutoElicit, which asks an LLM for its best guess of the mean and standard deviation of each coeficient under K = 100 automatically rephrased task descriptions, combining the results into a mixture prior $\textstyle \sum _ { k } \pi _ { k } { \mathcal { N } } ( \theta \mid \mu _ { k } , \sigma _ { k } ^ { 2 } )$ , for Bayesian linear and logistic regression, showing that LLM priors can beat uninformative priors and reduce labelling efort substantially. Its evaluation is in-distribution, and asks how few labels a well-specified prior saves rather than how well the fitted model transfers. Huang [24] generalises the elicitation target, converting unstructured context into tractable distributions via a Gaussian mixture and aggregating priors across agents by logarithmic opinion pooling. Zhang et al. [25] take a diferent encoding approach with LLM-Lasso, asking the LLM for feature-specific penalty scores that guide Lasso regularisation rather than for coeficient values.

Other work bypasses coeficients entirely: Requeima et al. [26] treat the LLM as an in-context probabilistic regressor competitive with Gaussian processes, and Zeng et al. [27] train on LLM embeddings of serialised rows, finding inconsistent robustness under $( Y \mid X )$ shift unless a few labelled target examples are available. Amin et al. [28] find that an ML ensemble substantially outperforms LLM prediction alone, with hybrid fusion adding only a small gain, supporting the broader principle that LLMs are more useful as structured inputs to classical models than as final predictors.

AutoElicit is the method closest to our own. While we detail their exact relationship below, their primary focus is in-distribution prediction, whereas ours is OOD generalisation. By collapsing AutoElicit’s mixture to a single Gaussian through moment matching, we obtain a diagonal Gaussian prior over the coeficients. This is the same object as our suggested Independent-Prior of Table 1. Three characteristics then separate the two methodologies. The first is where the prior’s width originates: AutoElicit asks the model to state a standard deviation, whereas we take the empirical spread of M sampled coeficient vectors, so its widths are self-reported and ours are behavioural. The second is the source of diversity, K rephrasing of the task description against M samples at a fixed prompt, which capture cross-prompt and within-prompt variability respectively and are complementary rather than competing. The third is the one our results make central: AutoElicit uses the elicited prior at the strength the model states, with no free parameter governing how far to trust it, whereas we introduce λ and let source-domain cross-validation rescale the entire precision.

Closest to this work is LoID [29], which extracts priors for Bayesian logistic regression from token-level predictions rather than generated coeficients, comparing the model’s preference for semantically opposed tokens across prompt phrasings and using the variation across phrasings as uncertainty. LoID is evaluated on fifteen tabular datasets, fourteen public and one private, under OOD covariate shift, and outperforms LLM Processes on thirteen of them and AutoElicit on eleven. We difer in the elicitation mechanism (repeated coeficient sampling rather than token log-probabilities), in the encoding of prior uncertainty (full covariance across coeficients, not only independent variances), and in how absolute prior strength is set (source-domain crossvalidation rather than fixed hyperparameters). We also difer in how the shift itself is constructed. LoID restricts its training set to a region of a real covariate (excluding respondents under 28 on Adult, or keeping only normoglycaemic patients on Diabetes) and evaluates on a held-out sample of the undivided population, so the target contains the training region as well as the part withheld from it. We train on one domain and test on a disjoint other, and on two datasets the division is not ours to choose at all: Heart Disease splits by recruiting cohort and Cancer by geographic region.

## 3 Methods

## 3.1 Notation and problem setting

Let $x \in \mathbb { R } ^ { p }$ denote a column vector of $p$ features and $y$ the associated response, real-valued for regression and binary for classification. A domain is a joint distribution over $( x , y )$ . We write $P _ { S }$ for the source domain, from which the available data are drawn, and $P _ { T }$ for the target or deployment domain, on which the fitted model will actually be used. The training sample is $\mathcal { D } _ { n } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , drawn independently from $P _ { S }$

All models in this study are linear in the features, with linear predictor

$$
\eta _ { \beta } ( x ) \ : = \ : \beta _ { 0 } + x ^ { \top } \beta , \qquad \beta \in \mathbb { R } ^ { p } , \ \beta _ { 0 } \in \mathbb { R } ,\tag{1}
$$

where $\beta$ is the coeficient vector and $\beta _ { 0 }$ an intercept. For regression we assume $y = \eta _ { \beta } ( x ) + \varepsilon$ with $\varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ ; for classification, $\operatorname* { P r } ( y = 1 \mid x ) = \log \mathrm { i t } ^ { - 1 } ( \eta _ { \beta } ( x ) )$ . In both cases we write $\mathcal { L } ( \beta ; \mathcal { D } _ { n } )$ for the negative log-likelihood of the sample, which reduces to the residual sum of squares in the regression case and to the Bernoulli negative log-likelihood in the classification case.

The quantity we would like to make small is the risk on the deployment domain,

$$
R _ { T } ( \beta ) \ = \ \mathbb { E } _ { ( x , y ) \sim P _ { T } } \big [ \ell \big ( y , \eta _ { \beta } ( x ) \big ) \big ] ,\tag{2}
$$

for a loss ℓ. Standard estimation instead minimises an empirical version of the source risk $R _ { S }$ and the two coincide only when $P _ { S } = P _ { T }$

When they difer we speak of domain shift, and it is worth distinguishing three kinds [2, 3]. Under covariate shift the feature distribution changes, $P _ { S } ( x ) \ne P _ { T } ( x )$ , while the conditional mechanism $P ( y \mid x )$ is unchanged. Under label shift the marginal response distribution changes, $P s ( y ) \neq P r ( y )$ , for instance when the target population contains a larger proportion of cases than the source. Under concept shift the association itself changes, $P s ( y \mid x ) \neq P _ { T } ( y \mid x )$ , as when the risk conferred by a given exposure difers between populations.

The distinction matters for what may be assumed rather than for what we do. The splits used here are defined by real covariates: housing stock age, census region, clinical cohort, patient age, employment sector, and such splits generally induce more than one kind of shift at once. In particular we have no way to verify that $P ( y \mid x )$ is preserved across, say, American and European cardiology cohorts, so concept shift is a live possibility on several of our datasets and we do not assume it away. This is a reason for the approach rather than an obstacle to it: if the conditional mechanism itself difers between domains, no reweighting of the source sample can recover the target relationship, and external knowledge about which efects are stable becomes the only available lever. Label shift matters separately and concretely, because it is what makes a fixed decision threshold inappropriate on the target domain, a point we return to in the Results.

The dificulty is one of identification rather than of estimation. Write $\beta _ { S } ^ { \star }$ and $\beta _ { T } ^ { \star }$ for the coeficient vectors minimising the source and target risks. Maximum likelihood on $\mathcal { D } _ { n }$ is consistent for $\beta _ { S } ^ { \star }$ , and no procedure that sees only $\mathcal { D } _ { n }$ can separate the components of $\beta _ { S } ^ { \star }$ that also hold under $P _ { T }$ from those that are artefacts of the source population, because $P _ { T }$ is simply not represented in the sample. Cross-validation does not help: its folds are themselves draws from $P _ { S }$ . Improving $R _ { T }$ therefore requires information from outside the training sample.

We refer to such information as global knowledge: statements about the relationship between features and response that are not derived from either sample, and that are expected to hold across populations rather than within one. Established domain theory, mechanistic understand ing, results from previous studies on diferent populations, and the judgement of experienced practitioners are all of this kind.

Such knowledge enters the model as a distribution over the coeficients. Write $f ( \beta )$ for a density on $\mathbb { R } ^ { p }$ expressing what is believed about $\beta$ before the sample is seen; in the Bayesian literature this is the prior distribution. The sample enters through the likelihood $f ( \mathcal { D } _ { n } \mid \beta ) =$ ex $\operatorname { p } \{ - \mathcal { L } ( \beta ; \mathcal { D } _ { n } ) \}$ , and the two combine into the posterior distribution

$$
f ( \beta \mid \mathcal { D } _ { n } ) \propto f ( \mathcal { D } _ { n } \mid \beta ) f ( \beta ) .\tag{3}
$$

The two factors pull in diferent directions, and that is the point. The likelihood is a statement about the source sample alone and is maximised at $\beta _ { S } ^ { \star }$ , whereas $f ( \beta )$ does not depend on that sample at all. Multiplying them moves the posterior of the source optimum and toward wherever the prior places its mass, by an amount governed by how concentrated the prior is. If the prior encodes global knowledge, this is exactly the correction the source sample cannot supply on its own. The notation $f ( \beta ) , f ( \mathcal { D } _ { n } \mid \beta )$ and $f ( \beta \mid D _ { n } )$ is used throughout the rest of this section.

## 3.2 Penalised regression as maximum a posteriori estimation

The standard device for controlling variance in a linear model is an $L _ { 2 }$ penalty,

$$
{ \widehat { \beta } } _ { \mathrm { r i d g e } } = \begin{array} { r } { \arg \operatorname* { m i n } _ { \beta } \Big \{ { \mathcal L } ( \beta ; { \mathcal D } _ { n } ) + \frac { 1 } { 2 } \lambda \| \beta \| _ { 2 } ^ { 2 } \Big \} , \qquad \lambda \geq 0 , } \end{array}\tag{4}
$$

which gives ridge regression in the Gaussian case and $L _ { \mathrm { 2 ^ { - r e g u l a r i s e d } } }$ logistic regression in the binary case [11, 10]. Penalisation trades a small increase in bias for a larger reduction in variance, and is especially useful when $p$ is large relative to n or the features are collinear.

Equation Eq. 4 has an exact Bayesian reading. Placing the prior $\beta \sim \mathcal { N } ( 0 , \lambda ^ { - 1 } I _ { p } )$ on the coeficients and combining it with the likelihood $\mathrm { g i }$ ves a posterior whose mode is precisely ${ \widehat { \beta } } _ { \mathrm { r i d g e } }$ since the negative log-prior equals $\begin{array} { r } { \frac { 1 } { 2 } \lambda \| \beta \| _ { 2 } ^ { 2 } } \end{array}$ up to an additive constant [10, §3.4], [9]. Ridge regression is therefore maximum a posteriori (MAP) estimation under a particular prior, and λ is the prior precision: large λ expresses a tight prior concentrated at the origin, and $\lambda  0$ recovers the unpenalised estimator.

This reading makes the content of the ridge penalty explicit, and it is worth stating plainly: ridge asserts that every coeficient is a priori centred at zero. That is a statement of ignorance, not of knowledge. It is a reasonable default and it does reduce variance, which is valuable under domain shift because a model that has fitted source-specific structure will generalise poorly. But shrinking toward zero provides no information about which efects transfer between populations, and it is exactly that information which the source sample lacks.

More generally, penalisation pulls the estimate toward the target, which need not mean making it smaller: if the target lies further from the origin than the unpenalised estimate does, the penalty moves the coeficients outward and $\| \beta \|$ grows with $\lambda .$ “Shrinkage” is then a misnomer for what the penalty does to the norm, although it remains accurate about what it does to the variance, which falls either way because the estimate is held close to a fixed point instead of varying freely with the sample. It reduces bias with respect to $R _ { T }$ only if the target lies closer to $\beta _ { T } ^ { \star }$ than the unshrunk estimate does. The origin is one possible target; global knowledge suggests others.

## 3.3 Priors that encode external knowledge

Replacing the zero-mean isotropic prior with a general Gaussian $\beta \sim \mathcal { N } ( \mu , \lambda ^ { - 1 } P ^ { - 1 } )$ , where $\boldsymbol { \mu } \in \mathbb { R } ^ { p }$ and $P$ is positive definite, and taking the posterior mode gives

$$
\widehat { \beta } ( \mu , P , \lambda ) \ = \ \arg \operatorname* { m i n } _ { \beta } \Big \{ \mathcal { L } ( \beta ; \mathcal { D } _ { n } ) + \frac { 1 } { 2 } \lambda \big ( \beta - \mu \big ) ^ { \top } P \left( \beta - \mu \right) \Big \} .\tag{5}
$$

Here $\mu$ is the shrinkage target and P the shrinkage structure; λ scales the whole penalty. Ridge is the special case $( \mu , P ) = ( 0 , I _ { p } )$ . The intercept is never penalised, as it is determined by the response base rate and the centring of the features rather than by any belief about feature– response relationships.

Three choices of $P$ correspond to three strengths of statement that a source of knowledge might make:

1. $P = I _ { p }$ uses only a point belief: every coeficient is pulled toward its target $\mu _ { j }$ with the same strength.

2. $P = \mathrm { d i a g } ( 1 / \sigma _ { 1 } ^ { 2 } , . . . , 1 / \sigma _ { p } ^ { 2 } )$ adds per-coeficient confidence. Coeficients the source is sure about have small $\sigma _ { j } ^ { 2 }$ and are held near $\mu _ { j } { \mathrm { : } }$ ; coeficients it is unsure about are left to the data.

3. $P = \Sigma ^ { - 1 }$ for a full covariance Σ additionally encodes dependence between beliefs, allowing the prior to express that two coeficients are believed to move together even when neither is determined precisely on its own.

These quantities have a natural interpretation when knowledge is elicited from several sources. Suppose M experts are each asked for a complete coeficient vector, yielding $\beta ^ { ( 1 ) } , \dots , \beta ^ { ( M ) } \in \mathbb { R } ^ { p }$ . The pooled belief and the disagreement among experts are then estimated by

$$
\widehat { \mu } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \beta ^ { ( m ) } , \qquad \widehat { \sigma } _ { j } ^ { 2 } = \frac { 1 } { M - 1 } \sum _ { m = 1 } ^ { M } \big ( \beta _ { j } ^ { ( m ) } - \widehat { \mu } _ { j } \big ) ^ { 2 } ,
$$

$$
\widehat { \boldsymbol { \Sigma } } = \frac { 1 } { M - 1 } \sum _ { m = 1 } ^ { M } \big ( \beta ^ { ( m ) } - \widehat { \mu } \big ) \big ( \beta ^ { ( m ) } - \widehat { \mu } \big ) ^ { \intercal } .\tag{6}
$$

Disagreement between experts becomes prior uncertainty, which is the classical aggregation view of structured elicitation $[ 7 , 8 ]$ . Nothing in Eq. 6 depends on what the experts are.

Moving the target away from the origin has a cost that is easy to overlook. With $\mu = 0$ increasing λ shrinks $\| \beta \|$ and therefore reduces variance monotonically. With $\mu \neq 0$ , increasing $\lambda$ drives $\widehat { \beta }$ toward $\mu$ rather than toward zero, so the implicit norm control that gives ridge its stability is given up: if $\mu$ is wrong, a large λ now moves the estimate away from the target optimum. The parameter λ thus governs a balance between two distinct things: how much the data are trusted relative to the prior, and how much variance reduction is retained, and the balance is favourable only to the extent that $\mu$ is informative.

One way to hedge against a misspecified target without abandoning it is to combine the two extremes at the level of predictions rather than coeficients. Writing g for the inverse link (the identity for regression and $\mathrm { l o g i t } ^ { - 1 }$ for classification) the blend is taken on the response scale,

$$
\widehat { y } _ { \gamma } ( x ) = ( 1 - \gamma ) g \Big ( \eta _ { \widehat { \beta } ( 0 , I , \lambda _ { 0 } ) } ( x ) \Big ) + \gamma g \Big ( \eta _ { \widehat { \beta } ( \mu , I , \lambda _ { 1 } ) } ( x ) \Big ) , \qquad \gamma \in [ 0 , 1 ] ,\tag{7}
$$

so that for classification it is the two predicted probabilities that are averaged, not the two sets of log-odds. The distinction is immaterial for regression, where $g$ is the identity and the two coincide, but it defines a diferent model in the logistic case, and averaging probabilities is what the fitted results below use.

which recovers the uninformative baseline at $\gamma = 0$ and the fully informed model at $\gamma = 1$

## 3.4 Selecting prior hyperparameters using LLMs

The prior in Eq. 5 is itself parameterised, by the quantities $( \mu , P , \lambda )$ ; in the Bayesian literature these are the prior hyperparameters, and they must be given values before the model can be fitted. Placing further distributions on them only defers the choice, since those distributions have parameters of their own, so at some level values have to be set directly [8]. We set them at this first level, which keeps the estimator comparable to standard penalised regression and lets one implementation cover both the uninformative and the informative case. What remains is how each of them is obtained, and the three are not on the same footing: it is useful to separate the shape of the prior, $( \mu , P )$ , from its overall scale, λ. The distinction matters because elicitation is generally far more reliable about relative statements: which efects are positive, which are larger than others, which are more uncertain, than about the absolute magnitude of a coeficient or the width of a prior interval [7]. The same asymmetry has been reported specifically for language-model elicitation, where the direction of an efect is usually right while the stated width is poorly calibrated in both directions [22]. Committing to an elicited scale would therefore import the least reliable part of the judgement.

We consequently take the shape from elicitation and estimate the scale from data, choosing λ by cross-validation on the source sample. The division of labour is deliberate: the prior supplies the information it is reliable about, and the data decide how far to trust it. It also means the procedure degrades gracefully, since an uninformative or actively misleading prior can be switched of by driving λ toward zero.

Everything in this section is independent of where the vectors $\beta ^ { ( 1 ) } , \dots , \beta ^ { ( M ) }$ come from. Obtaining them from human experts is the classical approach, and it is slow, costly and hard to scale. The central question of this paper is whether LLMs can supply them instead, cheaply and at scale, and whether the resulting priors improve prediction on a shifted deployment domain. The remainder of this section describes how we elicit priors from LLMs and how we evaluate the resulting predictive performance on domain splits defined by real covariates.

## 3.5 Pipeline overview

Figure 1 shows how the steps fit together. Three branches meet at the modelling step. The data branch fixes an OOD partition up front and fits all preprocessing on the source domain alone.

The elicitation branch queries several language models repeatedly for coeficient vectors and converts the resulting samples into the prior quantities of Eq. 6. The modelling branch solves Eq. 5 for each choice of $( \mu , P )$ , tunes λ, and chooses among providers and prior structures.

Each dataset is divided into a source partition and a held-out partition drawn from a systematically diferent population. The held-out partition plays the role of the target domain $P _ { T }$ of Section 3.1: it is the sample on which $R _ { T }$ is estimated. Following the convention in this literature we call it the out-of-distribution (OOD) partition and the source side in-domain (ID), but the two names denote exactly the two domains already introduced.

One principle governs the whole design and is worth stating before the detail: no quantity derived from the OOD partition enters any fitting, tuning or selection step. Preprocessing statistics, the prior strength λ, the winning language-model variant and the prior family are all determined on source-domain data only, which is the situation a practitioner is actually in. The OOD set is touched once, at evaluation.

![](images/2dba668793b7a38c954934cc29861d625e3b7ade889b934e49f991b1d634db03.jpg)  
Figure 1: Pipeline overview. The data branch (top) fixes the ID/OOD split and fits preprocessing on source data only. The elicitation branch (middle) feeds dataset context into prompt construction, draws repeated samples per provider, validates them against a JSON schema, and estimates $\widehat { \mu } , \widehat { \sigma } ^ { 2 }$ and $\widehat { \Sigma }$ as in Eq. 6. The modelling branch (bottom) fits each family, tunes λ by 5-fold CV on the source training pool, selects the winning variant per family by mean CV score, and evaluates on the held-out ID test and OOD sets. The OOD set never enters any selection step.

The model families used in the experiments are instances of Eq. 5 and difer only in the pair $( \mu , P )$ ; Table 1 collects them. Reading down the table, the prior is given progressively more structure to express: nothing at all beyond a preference for small coeficients, then a target, then a target with per-coeficient confidence, then a target with a full covariance. The number of elicited quantities entering the penalty therefore grows from none, through p, to $p + p$ , to $p { + } p ( p { + } 1 ) / 2$ , which is also the order in which the estimators become harder to estimate reliably from a finite number of elicitations. The last two rows are not points on that scale: Mixed hedges between the first two by blending their predictions, and the unregularised fit is the $\lambda = 0$ limit in which the prior drops out entirely and is reported as a reference.

Two numerical details matter in practice. The elicited variances enter the Independent-Prior precision as $1 / \widehat { \sigma } _ { j } ^ { 2 }$ , which is unbounded if every elicitation returns the same value for feature $j .$ For the Correlated-Prior, the corresponding $\widehat { \Sigma }$ is then singular and $\widehat { \Sigma } ^ { - 1 }$ undefined. Exact agreement is not hypothetical, since a confident elicitor asked repeatedly for the same coeficient may well repeat itself. We therefore add a small constant to the elicited variances before inverting them, $\widehat { \sigma } _ { j } ^ { 2 } + \varepsilon \mathrm { ~ w i t h ~ } \varepsilon = 1 0 ^ { - 8 }$ , and a matching jitter of $1 0 ^ { - 6 }$ to the diagonal of $\widehat { \Sigma }$ before inversion. These changes are not innocuous. When a unanimous elicitation gives $\widehat { \sigma } _ { j } ^ { 2 } = 0$ exactly, the Independent-Prior penalty weight becomes $\lambda / \varepsilon = \lambda \times 1 0 ^ { 8 }$ , which fixates that coeficient at $\widehat { \mu } _ { j }$ rather than merely shrinking toward it. Far from capping the confidence a unanimous elicitation may express, ε converts it into an equality constraint, and it is a free parameter of the estimator rather than a numerical detail. This matters because one deployed variant does return bit-identical draws, for four of eleven features on Boston Housing and two of seven on Diabetes, so we treat ε as something to be checked rather than asserted and report a sensitivity analysis over it in Appendix C.

Table 1: The fitted families as choices of $( \mu , P )$ in Eq. 5. The intercept is never penalised. $\widehat { \mu } , \widehat { \sigma } _ { j } ^ { 2 }$ and $\widehat { \Sigma }$ are the elicited quantities of Eq. $6 ;$ “elicited quantities” counts how many numbers from the elicitation enter the penalty.
<table><tr><td>Family</td><td></td><td>Elicited quantities</td></tr><tr><td>Baseline  $( \mathrm { r i d g e } , L _ { 2 } { \mathrm { - l o g i s t i c } } )$ </td><td> $I _ { p }$ </td><td></td></tr><tr><td>Prior-Mean</td><td> $I _ { p }$ </td><td>p</td></tr><tr><td>Independent-Prior</td><td> $\widehat { \mu }$   $\mathrm { d i a g } ( 1 / \widehat { \sigma } _ { j } ^ { 2 } )$   $\widehat { \Sigma } ^ { - 1 }$ </td><td> $2 p$ </td></tr><tr><td>Correlated-Prior</td><td> $\widehat { \mu }$ </td><td> $p + p ( p { + } 1 ) / 2$ </td></tr><tr><td>Mixed</td><td>blend of the two rows above, Eq. Eq. 7 p</td><td></td></tr><tr><td>Unregularised ref.</td><td> $\lambda = 0 ;$  the prior drops out</td><td></td></tr></table>

## 3.6 Datasets and domain splits

We use five publicly available tabular datasets spanning housing economics, cancer epidemiology, cardiology, diabetes screening and census income: Boston Housing [30], a county-level cancer mortality compilation [31], the multi-cohort UCI Heart Disease database [32], the Pima Indians Diabetes Database [33] and UCI Adult [34]. Two are regression tasks and three are binary classification. They were chosen to vary in the two properties we expect to govern whether a prior helps: training-set size, which ranges from 242 to about 27,000 rows, and the magnitude of the domain shift. Table 2 summarises them.

Crucially, the domain shift in each case is defined by a real covariate rather than by a synthetic perturbation: housing stock age, US census region, clinical cohort and continent, patient age, and employment sector. Each split therefore reproduces a shift a practitioner could plausibly meet on deployment, at the cost of giving one shift per dataset rather than a controllable family of them.

One feature is removed on ethical grounds rather than statistical ones. Boston Housing’s $B = 1 0 0 0 ( \mathrm { B k } - 0 . 6 3 ) ^ { 2 }$ , with Bk the proportion of Black residents, encodes a racial-segregation hypothesis in its functional form and is non-invertible; it is the reason the dataset was removed from major software distributions [35]. We drop the column from the design matrix and the corresponding coordinate from $\widehat { \mu }$ and ${ \widehat { \Sigma } } ,$ leaving eleven features. Taking the principal submatrix of $\widehat { \Sigma }$ is the marginal prior over the features actually fitted. No other dataset is afected. Dropping it costs about a third of the efect on this dataset: the best resplit average falls from +11.5% to +7.9%, and the Prior-Mean advantage disappears altogether, which is itself informative about how much of the original lift depended on that one covariate. The prompts still described the twelve-feature problem, so the elicited coeficients are for a specification we no longer fit; this is a mild omitted-variable mismatch that we prefer to the alternative.

Every dataset follows the same protocol. A domain-defining variable is chosen, the data are divided into in-domain (ID) source and OOD sets on that variable, and the splitting variable is excluded from the feature matrix, such that domain identity cannot leak into the model. The ID set is split 80/20 into a training pool and a held-out ID test set; only the training pool is used for tuning and fitting. Features are dropped when they are unusable in either domain and the remainder are median-imputed from the source domain, every fill value coming from the source half. The only dataset where features are dropped is Heart Disease, where ca, thal and slope are discarded because they have 98%, 76% and 50% missing values in the Hungarian and Swiss cohorts, though being near-complete in Cleveland. The criterion is therefore the recording practices of the target cohorts rather than source-domain missingness alone, and we state it that way. A fourth feature, fbs, is missing in about 20% of the target rows and was retained after an imputation audit; we report this for transparency, and repeating the whole selection without it changes the Heart Disease log-loss lift from +8.03% to +8.00%, so nothing in the result depends on it. Numeric features are standardised to zero mean and unit variance, and categorical features are one-hot expanded with levels aligned across source and OOD. The imputer and scaler are fitted on the training portion only (inside each cross-validation fold during tuning, and on the full training pool at deployment) and then applied to held-out data.

Table 2: Datasets, domain splits and resulting partitions. “Shifted” counts features with |Cohen’s $d | > 0 . 5$ between source and OOD.
<table><tr><td>Dataset</td><td>Domain split (source → OOD)</td><td>p</td><td> $n _ { \mathrm { t r } }$ </td><td>nooD</td><td>Shifted</td></tr><tr><td colspan="6">Regression</td></tr><tr><td>Boston Housing</td><td>housing stock: newer → older</td><td>11</td><td>284</td><td>151</td><td>9/11</td></tr><tr><td>Cancer</td><td>US region: South → non-South</td><td>13</td><td>1,106</td><td>1,664</td><td>5/13</td></tr><tr><td>Classification Heart Disease</td><td>cohort: Cleveland → Hungary/Switz.</td><td>18</td><td>242</td><td>417</td><td>2/18</td></tr><tr><td>Diabetes</td><td>patient age:  $\leq 2 9  > 2 9$ </td><td>7</td><td>316</td><td>372</td><td>2/7</td></tr><tr><td>Adult</td><td>sector: private → non-private</td><td>20</td><td>27,124</td><td>14,936</td><td>2/20</td></tr></table>

A sixth dataset, a small in vitro ruminant fermentation experiment with $n _ { \mathrm { t r a i n } } = 1 2 4 [ 3 6 ]$ , was run through the same pipeline but is not reported: its per-fold validation curve was essentially flat across the whole λ grid, the selection rule landed at $\lambda \approx 0$ for every family, and the deployed estimators either tied the unregularised fit or were marginally worse. This is a practical lower bound on when the methodology can deploy a meaningful prior: cross-validated prior-informed estimation needs a validation surface with enough signal to distinguish λ values.

## 3.7 Eliciting priors from LLMs

Each prompt has a system part containing stable elicitation instructions and a user part containing dataset-specific content. The system prompt assigns a domain-appropriate role (for example epidemiologist, cardiologist, demographic specialist), requires coeficients in raw feature units, excludes the intercept (which is determined by the outcome base rate and feature centering rather than by domain expertise), and requires conformance to a fixed JSON schema. For repeated sampling the system prompt additionally asks for one plausible draw from the model’s belief distribution rather than a collapsed average, and to favour efects likely to persist under domain shift.

The user prompt supplies five things: the modelling task, domain context describing the source population, feature meanings with explicit descriptions, a short block of domain hints written by us, and quantitative feature-scale summaries (median, interquartile range and central 95% range) computed from the source training pool only. These let the model reason about plausible magnitudes rather than guessing from feature names. The hint block states the expected sign of the dominant coeficients and, on some datasets, their relative size: on Diabetes, for instance, that glucose is the strongest single predictor and that BMI should be positive but smaller. It is reproduced in full for one dataset in Appendix D. This is a real constraint on the interpretation of what follows: the directional content of the prior is partly ours, while the magnitudes, the per-feature variances and the covariance structure are the model’s. Section 4.4 and the Limitations return to what can and cannot be attributed to the elicitation as a result. Deliberately, prompts describe the source domain, but never reveal the OOD target population:

the Cancer prompt states that the source consists of US South-region counties without disclosing that evaluation uses non-Southern counties. Any OOD improvement therefore cannot reflect targeted optimisation for a known shift. Because models are fitted on standardised features, raw priors are converted to standardised space by multiplying each coeficient by the corresponding feature standard deviation, recomputed per fold.

JSON output is constrained both by prompt instruction and by an API-level schema (OpenAI json schema strict mode, Anthropic tool input schema with forced tool choice, Google response schema). The exception is Anthropic extended-thinking mode, where the API forbids combining extended thinking with forced tool use; there the schema is enforced by prompt only and the parser recovers JSON from the text.

The twelve variants are six models from three providers, each queried in a non-reasoning and a reasoning mode. Table 3 lists them with the identifiers under which their elicited samples are stored in the repository, which is how a reader locates the 50 draws behind any dataset and variant. They are not twelve independent sources: two of the six are smaller siblings of others from the same provider, all three providers train on overlapping web corpora, and every variant received identical prompt text, so agreement across them is weaker evidence than twelve independent elicitations would be.

Table 3: The twelve elicitation variants. “Mode” distinguishes a plain completion from one in which the model was asked to reason before emitting the coeficient block; the two modes use diferent JSON schemas, described in Appendix D. “Stored $\mathrm { a s } ^ { \prime \prime }$ is the directory under priors/<dataset>/ in the repository. Dated API snapshot identifiers were not recorded at elicitation time; see the note below.
<table><tr><td>Label</td><td>Provider</td><td>Model</td><td>Mode</td><td>Stored as</td></tr><tr><td>GPT-5.1</td><td>OpenAI</td><td>GPT-5.1</td><td>plain</td><td>openai_-gpt-5-1</td></tr><tr><td>GPT-5.1 (R)</td><td>OpenAI</td><td>GPT-5.1</td><td>reasoning</td><td>openai_-gpt-5-1__reasoning</td></tr><tr><td>GPT-5-mini</td><td>OpenAI</td><td>GPT-5-mini</td><td>plain</td><td>openai--gpt-5-mini</td></tr><tr><td>GPT-5-mini (R)</td><td>OpenAI</td><td>GPT-5-mini</td><td>reasoning</td><td>openai_-gpt-5-mini__reasoning</td></tr><tr><td>Opus 4.5</td><td>Anthropic</td><td>Claude Opus 4.5</td><td>plain</td><td>anthropic__opus-4-5</td></tr><tr><td>Opus 4.5 (R)</td><td>Anthropic</td><td>Claude Opus 4.5</td><td>reasoning</td><td>anthropic__opus-4-5__reasoning</td></tr><tr><td>Haiku 4.5</td><td>Anthropic</td><td>Claude Haiku 4.5</td><td>plain</td><td>anthropic_haiku-4-5</td></tr><tr><td>Haiku 4.5 (R)</td><td>Anthropic</td><td>Claude Haiku 4.5</td><td>reasoning</td><td>anthropic_haiku-4-5__reasoning</td></tr><tr><td>Gemini Pro</td><td>Google</td><td>Gemini 2.5 Pro</td><td>plain</td><td>google_-gemini-2-5-pro</td></tr><tr><td>Gemini Pro (R)</td><td>Google</td><td>Gemini 2.5 Pro</td><td>reasoning</td><td>google--gemini-2-5-pro__reasoning</td></tr><tr><td>Gemini Flash</td><td>Google</td><td>Gemini 2.5 Flash</td><td>plain</td><td>google_-gemini-2-5-flash</td></tr><tr><td>Gemini Flash (R)</td><td>Google</td><td>Gemini 2.5 Flash</td><td>reasoning</td><td>google--gemini-2-5-flash__reasoning</td></tr></table>

One reproducibility caveat belongs with this table. The elicitation recorded the model family but not the dated snapshot identifier the provider served at the time, so the exact weights behind each bundle cannot be determined and re-querying the same family later will not reproduce the stored draws. This is the reason the samples themselves are archived rather than a script for regenerating them: the 60 stored bundles, not the model names, are the reproducible artefact. Future work of this kind should log the full versioned identifier returned by each API call.

For each (dataset, variant) combination, M = 50 samples were drawn at temperature 0.8: high enough for meaningful diversity, low enough to avoid incoherence. Outputs with missing keys, extra keys or invalid numeric values were rejected and replaced by additional calls until 50 valid samples accumulated, subject to a cap on total attempts. A single prompt variant is used for every LLM and dataset, to avoid overfitting the elicitation pipeline to these datasets and splits. Appendix D reproduces a complete system and user prompt verbatim.

## 3.8 Prior quantities and model fitting

The elicited vectors are the $\beta ^ { ( m ) }$ of Eq. 6, treated as draws from the model’s implicit belief distribution over the coeficients, and $\widehat { \mu } , \widehat { \sigma } _ { j } ^ { 2 }$ and $\widehat { \Sigma }$ are computed from them exactly as defined there. The precision $\widehat { P } = \widehat { \Sigma } ^ { - 1 }$ is formed with the diagonal jitter described there.

Regression solutions to Eq. 5 are closed-form; logistic solutions use L-BFGS, with Correlated-Prior started at zero and the other families at a seeded $0 . 0 1 \mathcal { N } ( 0 , 1 )$ draw. We note that the Correlated-Prior logistic fits are sensitive to the optimiser path: Σ estimated fromb $M = 5 0$ draws is ill-conditioned, and its inverse yields a penalty under which L-BFGS terminates at slightly diferent points depending on the starting value and on whether the fit is warm-started from the λ grid walk. The resulting diferences are in the third decimal of the reported metrics and do not afect any conclusion, but they are a further argument for shrinking Σ before inversion.b

## 3.9 Tuning, selection and evaluation

The λ grid is given as 51 log-spaced values from $1 0 ^ { - 4 }$ to $1 0 ^ { 6 }$ ; the γ grid is 101 equally spaced values on [0, 1]. For classification, the baseline is tuned on $\lambda _ { \mathrm { e q u i v } } = 1 / C$ such that its regularisation axis runs in the same direction as the one of prior families. Hyperparameters are chosen by 5-fold cross-validation on the source training pool under a strict-best-mean rule: for each grid value, per-fold validation scores are averaged and the best average selected (lowest MSE for regression, highest AUC for classification). Folds are built once per dataset and reused by every variant, so all variants are scored on identical splits. For each family, the variant whose family model achieves the best mean CV score is deployed; this within-family rule produces the four models reported per dataset in Tables 4 and 5. Where a single model is required, as in the selection analysis of Section 4.4, the same criterion is applied across the whole grid of 48 candidates, selecting the one (variant, family) combination with the best mean CV score. Both rules use source data only. We also record the fold-level standard error of the CV score at the selected candidate, because the margins between competing candidates turn out to be small relative to it. Models are then refit on the full training pool and evaluated on the held-out ID test and OOD sets. The OOD set never enters any tuning or selection step.

Regression reports MSE. Classification reports AUC, log loss and Brier score as threshold-free metrics, and accuracy, precision, sensitivity, specificity and $F _ { 1 }$ at a threshold.

One bootstrap procedure is used in two places and it is worth separating them. Within a single fitted deployment, uncertainty is assessed by 2,000 resamples of the OOD test set, stratified by class for classification, with percentile intervals from the 2.5th and 97.5th percentiles. The same resample indices are applied to every model within a call, so per-resample draws are paired; improvement-over-baseline claims use the paired diference $\Delta = \mathrm { m o d e l - b a s e l i n e }$ , which accounts for the correlation between a prior-informed model and its baseline on the same OOD sample. Taken alone this quantifies sensitivity to which OOD rows happened to be observed and, by construction, not variability in training, tuning or elicitation. It is not the interval reported in the main results: those nest this resampling inside repeated source splits, as described under “Uncertainty intervals” below. The single-deployment version is what Appendix B uses for the threshold analysis, where the quantity of interest is a comparison at one fitted operating point.

## 3.10 Follow-up analyses

Threshold recalibration. A fixed 0.5 cut-of is not a neutral basis for comparison when the domain split moves outcome prevalence, because it rewards whichever model happens to place more probability mass above it rather than the model that ranks or calibrates better. To compare models fairly we therefore give every one of them, the baseline included, the same freedom to choose an operating point, and we choose it on source data only. For each classification dataset and each deployed model including the baseline, a threshold is selected from 91 values on [0.05, 0.95] by maximising mean 5-fold CV $F _ { 1 }$ on the source training pool. Models are then re-scored on the OOD set at their own selected threshold. Paired bootstrap intervals on $\Delta F _ { 1 }$ are computed with each model at its own threshold. The OOD-maximising threshold is also recorded as a diagnostic upper bound.

Placebo priors. The placebos test whether the elicited numbers contain usable information about these particular features, or whether any shrinkage target of roughly the same shape and magnitude would serve as well. Each one corrupts the prior’s content while preserving its structure. Permuted placebos draw a random permutation π of the p features and re-index the whole prior: $\widehat { \mu } \mapsto \widehat { \mu } _ { \pi } , \widehat { \sigma } ^ { 2 } \mapsto \widehat { \sigma } _ { \pi } ^ { 2 } , \widehat { P } \mapsto \widehat { P } _ { \pi \pi }$ . Sign-flip placebos draw $s \in \{ - 1 , + 1 \} ^ { p }$ uniformly and set $\widehat { \mu } \mapsto \widehat { \mu } \odot s _ { : }$ , leaving variances unchanged and mapping $\hat { P } \mapsto D \hat { P } \hat { D }$ with $D = \deg ( s )$ which is the correct transformation since $D ^ { - 1 } = D$ and $\Sigma \mapsto D \Sigma D$ . Both preserve the prior’s scale and its correlation magnitudes. For every draw, λ is re-tuned by the same CV rule, so placebos receive the same tuning advantage as the real prior. We use 200 placebo draws per (dataset, family, placebo type) combination for regression and 100 for classification, and report the empirical p-value (#{placebo $\geq \mathrm { r e a l } \} + 1 ) / ( n + 1 )$ . The Adult dataset is excluded: crossvalidation drives λ to the grid floor there and the prior families collapse onto the baseline, so a placebo comparison is meaningless by construction.

Uncertainty intervals. Every interval in the Results propagates three sources of variability together. The ID 80/20 split is redrawn S times $( S = 5 0$ for Boston Housing and Cancer, 30 for Heart Disease and Diabetes, 10 for Adult); inside each redraw λ is re-selected by the cross-validation rule above and the model refitted; and the resulting model is then scored on 200 bootstrap resamples of the OOD partition, stratified by class for classification. The $S \times 2 0 0$ resulting values of the lift over the baseline are pooled.

Two intervals are read of this construction and they answer diferent questions. The 2.5th to 97.5th percentile range of the pooled draws is a prediction interval: it describes the lift a single future run of the whole pipeline would realise, and the accompanying fraction is the share of those draws below zero. A confidence interval for the mean lift is narrower, because the source-split component is averaged over the S repetitions while the target component is not, all S splits are scored on the same realised OOD partition, so there is only one draw of the target sample. Writing $\sigma _ { \mathrm { s p l i t } } ^ { 2 }$ for the variance of the S per-split means and $\sigma _ { \mathrm { t g t } } ^ { 2 }$ for the mean within-split bootstrap variance, we report

$$
\mathrm { S E } ( \bar { \Delta } ) = \sqrt { \sigma _ { \mathrm { s p l i t } } ^ { 2 } / S + \sigma _ { \mathrm { t g t } } ^ { 2 } } ,\tag{8}
$$

and a normal interval $\bar { \Delta } \pm 1 . 9 6 \mathrm { S E } ( \bar { \Delta } )$ . The distinction matters: read as a confidence interval, a prediction interval understates the evidence substantially, because $\sigma _ { \mathrm { s p l i t } } ^ { 2 }$ enters at full width rather than divided by S. Resampling only the OOD partition would capture target-sample variability alone, and redrawing only the ID split would capture training and tuning variability alone; the two are disjoint, so both are needed. Elicitation variability is not covered, as the deployed variant and its $M = 5 0$ draws are held fixed.

Resplits. The ID 80/20 split is redrawn under 50 seeds (Boston Housing, Cancer), 30 (Heart Disease, Diabetes) or 10 (Adult), and the full deployment step is repeated with λ re-tuned from scratch each time. The OOD partition is fixed by the domain split, so only the training sample varies. The selected variant per family is held at the deployed choice, so this measures stability of the lift given that variant rather than stability of variant selection. Adult uses every fourth λ grid point for tractability, since each logistic fit on n ≈ 27,000 costs about two seconds and its λ is driven to the lowest grid value regardless.

## 4 Results

We report predictive performance first on the two regression datasets and then on the three classification datasets, followed by a series of diagnostic controls. These controls are necessary because raw predictive metrics alone cannot explain why a model improves, nor whether the approach is dependable in practice. An observed lift could simply be an artifact of shrinking toward an arbitrary non-zero target rather than genuine domain understanding, while reporting only the final selected model conceals both the risk of harmful priors and the noise inherent in source-domain model selection. We therefore test whether the benefit remains under systematic corruption of the priors, assess how much provider and prior-structure choices matter, and examine what the elicited priors actually agree on. The regression subsection also sets out how every interval in the paper is computed, and the later subsections refer back to it.

## 4.1 OOD prediction on regression tasks

Table 4 reports the deployed model per family on the two regression datasets, with two intervals that answer diferent questions. The confidence interval is for the mean lift: if the whole procedure were repeated many times, how precisely do we know the average benefit? The prediction interval is for one run: if a practitioner deploys this pipeline once, how much could the realised lift vary? The two difer because the source-split component of variance is averaged over the S repetitions in the first and enters at full width in the second, and the distinction matters here: the prediction interval covers zero almost everywhere, the confidence interval does not.

On Boston Housing the Independent-Prior and Correlated-Prior are the strongest: Correlated-Prior with Opus 4.5 averages a 7.9% reduction in OOD MSE and Independent-Prior 7.4%, while Prior-Mean averages essentially nothing (−0.2%). All three shrink toward the same elicited mean $\widehat { \mu }$ and difer only in the prior covariance, so what separates them is the treatment of spread. Prior-Mean assigns every coeficient the common prior variance $\lambda ^ { - 1 }$ and discards the per-feature spread entirely; Independent-Prior takes each coeficient’s variance from the observed spread of the M draws; Correlated-Prior additionally uses the cross-coeficient covariance. On Boston Housing it is therefore not the elicited location that produces the efect (Prior-Mean has that and gains nothing from it), but the elicited relative confidence.

In fact, this elicited confidence is far more rigid than a standard variance metric suggests: rather than exhibiting a narrow spread, the LLM returned bit-identical values across all fifty draws for four of the eleven features. Consequently, the elicited standard deviation is exactly zero and the penalty weight $\lambda / ( \widehat { \sigma } _ { j } ^ { 2 } + \varepsilon )$ is set by the floor ε rather than by the elicitation. Those four coeficients are efectively held at $\widehat { \mu } _ { j }$ . This is a strong modelling commitment and we test it rather than assert it: Table S2 re-tunes and refits across four orders of magnitude of ε and against a Ledoit–Wolf shrinkage estimator of $\widehat { \Sigma }$ in place of the jitter. The Boston Housing lift is +11.15% at the deployed setting and +11.10% under Ledoit–Wolf, and degrades only as ε grows large enough to erase the elicited spread altogether. The efect is a property of the elicitation, not of the floor.

The reverse holds on Cancer, where Prior-Mean averages +2.4% and Independent-Prior and Correlated-Prior around +1%. This is not a case of the elicited spread being uninformative for lack of variation: the deployed variants there have no zero-variance coeficients at all, and their standard deviations range over a factor of two to four around the median. It is worth noting when the families would become identical: if the elicited standard deviations were equal across features, diag $( 1 / \widehat { \sigma } _ { j } ^ { 2 } )$ would reduce to a multiple of the identity and Independent-Prior would be Prior-Mean with λ rescaled, which the tuning grid absorbs. They separate only to the extent that the elicitation expresses diferential confidence, and on Cancer the diferential confidence it expressed was actively unhelpful, the coeficients the models were most certain about were not the ones that transferred best.

Taken together the two datasets separate two things the elicitation supplies. The location $\widehat { \mu }$ is useful on both. The relative confidence attached to it is useful on Boston Housing and misleading on Cancer, which is why no single prior structure is best on both datasets and why the choice between them has to be made per dataset rather than settled once.

Five of the six prior-informed results have positive means, and under the confidence interval five of six exclude zero: Independent-Prior and Correlated-Prior on Boston Housing and all three families on Cancer. Only Boston Housing’s Prior-Mean, which gains nothing on that dataset, does not.

Table 4: Regression datasets. The selected variant and λ are those chosen by mean 5-fold CV on the source training pool of the reference split, and the OOD MSE column is that deployed model’s error on the OOD partition. The MSE and the lift columns come from two diferent computations: the MSE is a single number from the reference split, whereas the lift and both intervals are averaged over repeated source splits, so the lift is not recoverable by comparing the two MSE entries. Lift is the percentage reduction in OOD MSE relative to the ridge baseline. “CI” is a 95% confidence interval for the mean lift; “PI” is a 95% prediction interval for the lift of a single deployment. Both propagate source-split, λ-selection and target-sample variability; they difer in whether the source component is averaged over the S splits. “Below” is the fraction of prediction draws in which the prior-informed model is worse than baseline. The variant named is the CV-best one the pipeline deploys, not the OOD-best one. A dagger (<sup>†</sup>) marks a confidence interval excluding zero.
<table><tr><td>Family</td><td>Selected variant</td><td></td><td>λ OOD MSE</td><td>Lift 95% CI (mean)</td><td></td><td>95% PI (one run) Below</td><td></td></tr><tr><td colspan="8">Boston Housing  $( n _ { \mathrm { t r a i n } } = 2 8 4 , n _ { \mathrm { O O D } } = 1 5 1 )$ </td></tr><tr><td>Unregularised ref.</td><td></td><td></td><td>44.99</td><td></td><td></td><td></td><td></td></tr><tr><td>Baseline ridge</td><td></td><td>25.1</td><td>44.10</td><td></td><td></td><td></td><td></td></tr><tr><td>Prior-Mean</td><td>Gemini Pro</td><td>251</td><td></td><td></td><td> $4 2 . 9 2 \ - 0 . 1 8 \ \left[ - 8 . 2 9 , + 7 . 9 2 \right]$ </td><td> $\left[ - 1 2 . 9 7 , + 1 0 . 5 4 \right]$ </td><td>48%</td></tr><tr><td>Independent-Prior Opus 4.5</td><td></td><td>10.0</td><td></td><td>39.56 +7.40</td><td> $[ + 0 . 5 7 , + 1 4 . 2 4 ] ^ { \dagger }$ </td><td> $[ - 1 . 0 4 , + 1 8 . 2 3 ]$ </td><td>4%</td></tr><tr><td>Correlated-Prior</td><td>Opus 4.5</td><td>15.8</td><td></td><td> $3 9 . 1 8 ~ + 7 . 8 5$ </td><td> $[ + 1 . 1 8 , + 1 4 . 5 1 ] ^ { \dagger }$ </td><td> $[ - 0 . 6 1 , + 1 8 . 8 7 ]$ </td><td>3%</td></tr><tr><td>Mixed</td><td>Gemini Pro</td><td>γ=1.00</td><td></td><td></td><td>42.92 −0.18 [−8.29, +7.92]</td><td> $\left[ - 1 2 . 9 7 , + 1 0 . 5 4 \right]$ </td><td>48%</td></tr><tr><td colspan="8">Cancer  $( n _ { \mathrm { t r a i n } } = 1 , 1 0 6 ,$  nooD = 1,664)</td></tr><tr><td>Unregularised ref. –</td><td></td><td></td><td>524.17</td><td></td><td></td><td></td><td></td></tr><tr><td>Baseline ridge</td><td></td><td>100</td><td>501.22</td><td></td><td></td><td></td><td></td></tr><tr><td>Prior-Mean</td><td>Opus 4.5</td><td>158</td><td> $4 8 1 . 6 7 ~ + 2 . 3 7 ~ $ </td><td></td><td> $[ + 1 . 7 1 , + 3 . 0 4 ] _ { . } ^ { \dagger }$ </td><td> $[ - 0 . 3 8 , + 7 . 5 3 ]$ </td><td>12%</td></tr><tr><td>Independent-Prior GPT-5.1</td><td></td><td>158</td><td> $4 8 7 . 2 7 \ + 1 . 1 3$ </td><td></td><td> $\cdot + 0 . 6 0 , + 1 . 6 6 ] ^ { \dagger }$ </td><td> $[ - 1 . 2 4 , + 3 . 4 4 ]$ </td><td>11%</td></tr><tr><td>Correlated-Prior</td><td>Gemini Pro (R)</td><td>158</td><td> $4 9 0 . 6 0 \ + 0 . 9 0 $ </td><td></td><td> $[ + 0 . 4 3 , + 1 . 3 8 ] ^ { \dagger }$ </td><td> $[ - 1 . 1 1 , + 3 . 3 8 ]$ </td><td>17%</td></tr><tr><td>Mixed</td><td>Opus 4.5</td><td>γ=0.68</td><td> $4 8 7 . 6 9 \ + 1 . 6 5$ </td><td></td><td> $[ + 1 . 1 9 , + 2 . 1 1 ] ^ { \dagger }$ </td><td> $[ - 0 . 2 6 , + 5 . 3 0 ]$ </td><td>11%</td></tr></table>

The same models were also scored on the held-out in-domain test split, and the comparison bears directly on what the prior is doing. On Boston Housing the in-domain gain is larger than the out-of-domain gain: +13.2% against +11.2% for Correlated-Prior on the reference split. The same holds on Heart Disease (+1.7% in-domain against +1.0%) and Diabetes (+3.7% against +1.0%). On those three datasets the elicited prior is a better regularisation rather than a correction for domain shift specifically. Cancer is the exception: in-domain the prior is worth +0.1% and out-of-domain +3.9%, so there the benefit is specific to the shift. Section 5.1 returns to why Cancer difers.

The construction comes with two caveats. First, bootstrapping the OOD partition treats it as a sample from $P _ { T }$ , which is the right way to view the risk $R _ { T }$ of Eq. 2, but each OOD partition here is a complete subpopulation: every non-Southern county, every Hungarian and Swiss patient in the database, rather than a draw from a larger pool. For the narrower question “did the prior help on this particular deployment population”, the target is fully observed and the $\sigma _ { \mathrm { t g t } } ^ { 2 }$ term of Eq. 8 should be dropped. Doing so narrows every interval and adds six further results that exclude zero, among them all four Diabetes log-loss and Prior-Mean results. We report the generalising version throughout because it is the question a practitioner faces before deploying; Appendix A shows how the estimated performance gain varies across source-data splits. For each split, we refit the models but evaluate them on the same OOD sample. Averaging over splits therefore reduces uncertainty due to the source split, but it does not provide additional target observations or reduce uncertainty due to the finite OOD sample. This distinction matters most for Boston Housing: with only 151 OOD observations, target-sample uncertainty accounts for roughly half of the variance used to construct its confidence intervals. Cancer has 1,664 OOD observations, so this component accounts for less than 5% of its interval variance. Its confidence intervals are consequently much narrower, even though its estimated gain is smaller.

## 4.2 OOD prediction on classification tasks

Table 5 shows the predictive performance of the diferent models on the three classification datasets. Two metrics are reported for each: AUC, which measures only how well the model ranks observations, and log loss, which measures the quality of the predicted probabilities themselves. Separating the two turns out to be important, because the priors act on one and not the other.

Ranking is barely afected on Heart Disease, where all three AUC confidence intervals include zero. Proper scoring rules tell a diferent story: log loss improves by more than 10% for every family on Heart Disease (0.4709 → 0.4327 for Prior-Mean), and all three confidence intervals exclude zero (+12.23% for Prior-Mean and +12.43% for Correlated-Prior). Diabetes improves on both metrics, with AUC confidence intervals excluding zero for Independent-Prior and Correlated-Prior, alongside positive log-loss gains of 3–4%. Brier scores move consistently in the same direction, and reliability curves show that the prior-informed models track calibration more closely than the baseline (Supplementary Fig. S2). Adult, by contrast, is the exception to these gains: with $n _ { \mathrm { t r a i n } } \approx 2 7 { , } 0 0 0$ , the baseline model is already close to its best attainable accuracy, cross-validation drives λ to the grid floor, and the prior families collapse onto the baseline. While two Adult results technically exclude zero in the direction of harm (−0.01% AUC and −0.05% log loss), they are detectable only because $n _ { \mathrm { O O D } } = 1 4 { , } 9 3 6$ renders them statistically precise, and have no practical consequence.

Threshold-dependent metrics such as $F _ { 1 }$ , sensitivity, and specificity are deliberately omitted from Table 5. All such metrics require an operating cut-of, but the domain splits here induce substantial label shift: outcome prevalence shifts from 0.212 to 0.495 on Diabetes and from 0.459 to 0.530 on Heart Disease. Under such shifts, evaluating models at a fixed 0.5 threshold artificially rewards whichever model produces higher probabilities, regardless of calibration or ranking quality. Appendix B demonstrates how this prevalence shift manufactures an apparent $F _ { 1 }$ gain that disappears once baselines are granted the same threshold freedom, confirming that proper scoring rules provide the only defensible basis for classification evaluation here.

Adult deserves a more careful reading than “the prior does not help when data are plentiful”. There are two distinct reasons a prior might be silent at large n, and they have diferent implications. The first is statistical: a prior of fixed precision contributes a constant to the log-posterior while the likelihood contributes a term growing with $n ,$ so its influence decays as $1 / n$ and the posterior concentrates on the maximum-likelihood estimate. Under domain shift that estimate converges to the source optimum $\beta _ { S } ^ { \star }$ , whose bias with respect to the target does not shrink with n; a large source sample therefore gives an increasingly precise estimate of the wrong quantity while at the same time removing the one source of information that could correct it. On that argument a prior ought to remain useful on Adult, since the bias it would correct is a property of the domain rather than of the sample size.

The second reason is specific to how we set the prior weight. Because λ is tuned rather than fixed, the asymptotic argument does not bind automatically, λ could in principle grow with $n ,$ but it is selected to minimise source-domain cross-validation error, and a prior that trades in-domain fit for out-of-domain transfer is strictly bad by that criterion. On Adult, the crossvalidation drives the prior weight low enough that every family collapses onto the baseline, and at that point the two reasons are observationally equivalent: a prior that contains nothing useful for this dataset and a prior whose usefulness the selection rule cannot see produce the same null. Separating them would require a selection objective aimed at the target domain rather than the source, which is a change to the method rather than a further analysis of ${ \mathrm { i t } } ,$ and we take it up in the Discussion. That the prior does no harm here is still the property the design was meant to guarantee: two of Adult’s intervals formally exclude zero on the side of harm, but at −0.01%

Table 5: Classification datasets. “Value” is the deployed model’s metric on the reference split; “lift” is the percentage improvement over the L -logistic baseline. “CI” is a 95% confidence interval for the mean lift and “PI” a 95% prediction interval for a single deployment; both propagate source-split, λ-selection and target-sample variability and difer only in whether the source component is averaged over the S splits. “Below” is the fraction of prediction draws in which the prior-informed model is worse than baseline. The Mixed family is omitted and tracks Prior-Mean closely (γ = 1.00 and 0.97). A dagger (<sup>†</sup>) marks a confidence interval excluding zero; on Adult the two such entries point in the direction of harm, at magnitudes of no practical consequence.
<table><tr><td>Family</td><td>Variant</td><td>Metric</td><td>Value</td><td>Lift 95% CI (mean)</td><td></td><td>95% PI (one run) Below</td><td></td></tr><tr><td colspan="8">Heart Disease (ntrain = 242, nooD = 417)</td></tr><tr><td>Baseline</td><td></td><td>AUC</td><td>0.8779</td><td></td><td></td><td></td><td></td></tr><tr><td>Prior-Mean</td><td>Opus 4.5</td><td>AUC</td><td>0.8860</td><td></td><td>+0.08 [−0.75, +0.91]</td><td>[−1.13, +2.15]</td><td>51%</td></tr><tr><td>Independent-Prior Gemini Pro (R)</td><td></td><td>AUC</td><td>0.8845</td><td></td><td>-0.07 [−1.13, +1.00]</td><td>[−1.50, +1.90]</td><td>57%</td></tr><tr><td>Correlated-Prior</td><td>Opus 4.5</td><td>AUC</td><td>0.8862</td><td></td><td>+0.18 [−0.68, +1.04]</td><td>[−1.03, +2.47]</td><td>47%</td></tr><tr><td>Baseline</td><td></td><td>LogLoss 0.4709</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prior-Mean</td><td>Opus 4.5</td><td></td><td></td><td></td><td>LogLoss 0.4327 +12.23 [+4.20, +20.25]† [−1.68, +39.30]</td><td></td><td>8%</td></tr><tr><td>Independent-Prior Gemini Pro (R)</td><td></td><td></td><td></td><td></td><td>LogLoss 0.4407 +10.78 [+3.20, +18.36]† [−3.45, +37.97]</td><td></td><td>19%</td></tr><tr><td>Correlated-Prior</td><td>Opus 4.5</td><td>LogLoss 0.4331 +12.43</td><td></td><td></td><td>[+4.85, +20.01]† [−1.45, +39.43]</td><td></td><td>7%</td></tr><tr><td colspan="8">Diabetes (ntrain = 316, nooD = 372)</td></tr><tr><td>Baseline</td><td></td><td>AUC</td><td>0.7730</td><td></td><td></td><td></td><td></td></tr><tr><td>Prior-Mean</td><td>Opus 4.5</td><td>AUC</td><td>0.7796</td><td></td><td>+1.41 [−0.11, +2.93]</td><td>[−0.28, +3.54]</td><td>5%</td></tr><tr><td>Independent-Prior Opus 4.5</td><td></td><td>AUC</td><td>0.7815</td><td>+1.78</td><td>[+0.12, +3.44]†</td><td>[−0.24, +4.10]</td><td>4%</td></tr><tr><td>Correlated-Prior</td><td>Opus 4.5</td><td>AUC</td><td>0.7805</td><td></td><td>+1.81 [+0.16, +3.47]†</td><td>[−0.24, +4.14]</td><td>4%</td></tr><tr><td>Baseline</td><td></td><td>LogLoss 0.6327</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prior-Mean</td><td>Opus 4.5</td><td>LogLoss 0.6025</td><td></td><td></td><td>+3.44 [−0.38, +7.25]</td><td>[−1.57, +23.44]</td><td>17%</td></tr><tr><td>Independent-Prior Opus 4.5</td><td></td><td>LogLoss 0.6047</td><td></td><td>+3.50</td><td>[−0.52, +7.51]</td><td>[-1.79, +23.39]</td><td>15%</td></tr><tr><td>Correlated-Prior</td><td>Opus 4.5</td><td>LogLoss 0.6035</td><td></td><td>+3.28</td><td>[−0.74, +7.30]</td><td>[-2.01, +23.39]</td><td>18%</td></tr><tr><td colspan="8">Adult (ntrain = 27,124, nooD = 14,936)</td></tr><tr><td>Baseline</td><td></td><td>AUC</td><td>0.8770</td><td></td><td></td><td></td><td></td></tr><tr><td>Prior-Mean</td><td>GPT-5-mini</td><td>AUC</td><td>0.8771</td><td></td><td>-0.00 [−0.00, +0.00]</td><td>[−0.01, +0.00]</td><td>64%</td></tr><tr><td>Independent-Prior Gemini Flash (R)</td><td></td><td>AUC</td><td>0.8768</td><td></td><td>−0.01 [−0.02, −0.01]†</td><td>[−0.04, −0.01]†</td><td>100%</td></tr><tr><td>Correlated-Prior</td><td>Opus 4.5 (R)</td><td>AUC</td><td>0.8769</td><td></td><td>-0.01 [−0.03, +0.00]</td><td>[−0.07, +0.00]</td><td>96%</td></tr><tr><td>Baseline</td><td></td><td>LogLoss 0.3940</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prior-Mean</td><td>GPT-5-mini</td><td>LogLoss 0.3940</td><td></td><td></td><td>+0.00 [−0.01, +0.02]</td><td>[−0.02, +0.06]</td><td>56%</td></tr><tr><td>Independent-Prior Gemini Flash (R)</td><td></td><td>LogLoss 0.3942</td><td></td><td></td><td>−0.05 [−0.08, −0.02]†</td><td>[−0.10, −0.02]†</td><td>100%</td></tr><tr><td>Correlated-Prior</td><td>Opus 4.5 (R)</td><td>LogLoss 0.3941</td><td></td><td></td><td>-0.04 [−0.09, +0.01]</td><td>[−0.19, +0.01]</td><td>91%</td></tr></table>

AUC and −0.05% log loss, detectable only because 14,936 OOD rows make them precise.

## 4.3 Placebo priors: the priors’ content, not their scale

A prior-informed model difers from ridge in two ways at once: it shrinks toward a non-zero target, and that target makes specific claims about which features matter and in which direction. The permuted and sign-flip placebos defined in the Methods separate these two properties: each preserves the prior’s overall scale and its correlation magnitudes while attaching them to the wrong features or the wrong directions. A model that gains merely from shrinking toward something non-zero of about the right size should therefore do as well under a placebo as under the real prior.

![](images/23d2cac68dd9551d292599cd7cc9d7f82d7a8119a49cf0c8afab5b24f04c1013.jpg)  
OOD lift over baseline (%), symlog scale: MSE reduction / log-loss improvement

Figure 2: Placebo priors. Each row is one (dataset, prior family, placebo type) combination. Vermillion points are individual placebo draws (200 for regression, 100 for classification) with the grey bar marking their 2.5–97.5 percentile range and the tick their median; the bluish-green marker is the real elicited prior. p is the empirical fraction of placebo draws reaching at least the real prior’s lift. Lift is OOD MSE reduction for Boston Housing and Cancer, and OOD log-loss improvement for Heart Disease and Diabetes, the metrics the priors actually move, as established in Tables 4 and 5. The horizontal axis is symmetric-log beyond ±10%, because corrupted priors on the classification datasets can be worse than baseline by several hundred per cent.

Figure 2 reports the outcome for all 24 (dataset, prior family, placebo) combinations. The real prior beats its placebo distribution in 20 of them at $p < 0 . 0 5$ , and the two corruptions are equally damaging: permutation, which keeps every elicited number but attaches it to the wrong feature, is rejected in 10 of 12 comparisons, and sign-flipping, which keeps each magnitude and discards only the direction, in 10 of 12 as well. The efect therefore depends on both the identity of the features the prior refers to and the directions it assigns to them.

The four exceptions do not group by placebo type but by family: three of them are Independent-Prior comparisons (Heart Disease under both corruptions, $\begin{array} { r l r } { p } & { { } = } & { 0 . 0 6 9 } \end{array}$ and $p = 0 . 5 3 5$ , and Diabetes under permutation, $p = 0 . 0 5 9 )$ , with Correlated-Prior sign-flip on Cancer $( p = 0 . 1 0 9 )$ the fourth. By dataset the counts are six of six on Boston Housing, five of six on Cancer and Diabetes, and four of six on Heart Disease. On the regression datasets every placebo median is below the ridge baseline, and on Boston Housing at most two of 200 draws match the real lift in any comparison, so a corrupted prior there is not merely less useful than the real one but actively worse than no prior at all.

The choice of metric matters here, and it is worth being explicit about it. The classification rows are scored on log loss rather than AUC, for the same reason Table 5 leads with log loss: AUC is the metric the priors leave essentially flat, so testing the priors’ content against it asks whether a corruption damages something the elicitation was never moving. It does not, and the result is correspondingly weak: on AUC only 7 of 12 classification comparisons reject, and sign-flipping is not rejected in four of six. That contrast is itself a caution for this literature. A placebo or ablation evaluated on a threshold-free ranking metric can understate an efect that is real in the predicted probabilities, and the mean real lift on these datasets is +0.91% on AUC against +6.08% on log loss.

The placebo tests clarify why the elicited priors help. Permuting the prior values across features or changing their signs generally reduces their predictive benefit, even though the values’ overall scale is preserved. The gains therefore depend on assigning plausible efects to the appropriate features and in the appropriate directions—not simply on shrinking coeficients toward non-zero values.

## 4.4 What the selection step can and cannot do

Every result above reports on a selected model, thus the properties of the selection rule matter as much as the properties of the priors. Per dataset there are 48 candidate combinations (12 language model × 4 prior model families), each at its own CV-tuned λ. Figure 3 shows the full range of OOD lifts over that combinations, the four models the within-family rule deploys, and the single model a cross-family rule selects.

The spread is wide: the candidate pool ranges from −2.3% to +11.2% on Boston Housing and from −2.9% to +6.1% on Cancer, and the worst candidate is below baseline on every dataset. An LLM-elicited prior is therefore not automatically safe because it is domain-informed. A study using a single provider, prompt and prior structure could have concluded that the method is clearly beneficial, neutral or actively harmful depending only on which combination it happened to pick, and that observation is the strongest practical argument in this paper for reporting the whole candidate distribution rather than a deployed configuration.

Source-domain selection uses that spread only partially. A cross-family rule picks Gemini Pro with Prior-Mean on Boston Housing, worth +2.7% compared with the best achievable +11.2%, and Gemini Pro (R) with Correlated-Prior on Cancer, worth +2.1% against +6.1%. On the classification datasets it reaches +0.9% of +1.7% and +1.0% of +2.1%. In every case the selected candidate is in the middle of the candidate range rather than near the best.

Source-domain cross-validation does not clearly distinguish the leading candidates. We assess the gap between the winner and the next distinct candidate using the standard error of their paired score diference. All candidates are evaluated on the same five folds, so a dificult fold tends to afect both scores, and much of that variation cancels when they are compared. The gap is only 0.06 standard errors on Boston Housing, Cancer and Adult, and 0.25 on Heart Disease; it reaches 1.63 on Diabetes. Thus, except perhaps on Diabetes, the cross-validation scores provide little basis for treating the selected candidate as clearly better than its closest alternative.

On Boston Housing, the two highest-scoring entries are actually the same fitted model:

![](images/090d4408683729a36406f6adf94afad311b87e2976c4cecdd92e8d39e84cb20b.jpg)  
Figure 3: Per-dataset OOD lift over baseline across all 48 (12 elicitation variants × 4 prior model families) candidates at their CV-tuned λ. Grey bar: the range from worst to best candidate. Blue circles: the four models deployed by the within-family rule, which are the models reported in Tables 4 and 5. Green diamond: the single model selected by a cross-family rule, the best of all 48 candidates by source-domain cross-validation. Every quantity is computed from source data alone; the distance from each marker to the top of its bar is the benefit that source-domain selection does not reach.

the Mixed family selects γ = 1, making it equivalent to Prior-Mean. We therefore compare the winner with the next distinct candidate. Although selection does not reliably identify the best-performing prior, the selected models avoid the worst-performing candidates in these experiments. Tables 4 and 5 report the variants selected within each prior family, as prescribed by the deployment procedure; their reported gains should not be interpreted as the result of reliably finding the best candidate.

What the selection step contributes is therefore protection rather than optimisation. Together with the tuned λ it keeps the deployed model away from the worst-performing candidates on every dataset, and on Adult, where the prior contains nothing useful, cross-validation drives the weight low enough that every family collapses onto the baseline and the cost of a useless prior is −0.03%. That is a safety property, not an optimality property.

## 4.5 What the elicited priors agree on

Figure 4 plots the prior-mean coeficient from all twelve variants per feature on the four datasets where priors help. The dominant feature in each panel (cancer incidence rate, property tax rate, exercise-induced ST depression, and plasma glucose) receives the same sign from every variant, with across-variant means clustering tightly. Sign disagreement is concentrated on lower-magnitude, more ambiguous features. Twelve independently prompted variants agreeing on the direction of the dominant predictor of county cancer mortality or of diabetes onset is unlikely to be chance, and is consistent with the priors containing genuine domain structure for the features that matter most.

Agreement on dominant signs does not make provider choice unimportant, as Figure 3 shows: smaller magnitude diferences, occasional sign disagreements, variance and covariance structure, and individual outlying samples all still matter once a prior enters a fitted model. For Gemini Flash on Boston Housing, 49 of 50 samples place the property-tax coeficient near the cluster but a single draw returns roughly −50 in raw units (a one-unit tax-rate increase lowering median home value by \$50,000) dragging the $M = 5 0$ mean to about −148 in standardised space. The CV-tuned λ and the selection step contain the damage, and this variant was not deployed on that dataset, but the episode argues for robust prior estimation.

![](images/f06aec70cfea9ad7db2016daa140706f359f9932346f5baa08a19b2a2decdfac.jpg)

![](images/b612f1e9a1b0c1ded07c281ed2834bd70bea76a97f27d6da9a5be5ac5c2030e0.jpg)

![](images/fa680f1546aae8dfcc07be18d0712eb98835a887a01139ba6d97011a93fe16c8.jpg)

![](images/423e9fc5d52bb1c3a80a045e8c6e36d799a3e3406bf5b8b1175b56dde16564f2.jpg)  
Figure 4: Cross-variant coeficient agreement on the four datasets where priors help. Each panel shows the prior-mean coeficient for all twelve variants per feature in source-standardised space. Filled circles: non-reasoning. Open circles: reasoning. Grey tick: across-variant mean. Heart Disease shows numeric features only. Of-panel points are annotated at the boundary.

## 5 Discussion

## 5.1 When LLM priors help

Replacing shrinkage toward zero with shrinkage toward an LLM-elicited target improved outof-distribution prediction on four of five datasets, and on most of those the improvement is resolved: ten of the twenty-four prior-informed results have a confidence interval for the mean lift that excludes zero in the beneficial direction. Two further results exclude zero on Adult, in the direction of harm and at magnitudes of no practical consequence. The gains are largest in the regression tasks, while in the classification tasks they reside in the probabilistic accuracy rather than ranking. The corresponding prediction intervals in Tables 4 and 5 are substantially wider and cover zero almost everywhere: the average benefit is measurable, while the benefit of any single deployment is not guaranteed.

The more interesting question is what the benefit is, and here the in-domain comparison is decisive. On Boston Housing, Heart Disease and Diabetes the elicited prior improves held-out in-domain accuracy by at least as much as it improves OOD accuracy. On those datasets the prior is simply a better regularisation target than zero: a useful result, but not one about domain shift, and not one that needs an OOD framing to be worth reporting. Cancer is the exception: in-domain the prior is worth +0.1% and out-of-domain +3.9%. It is the only dataset in the study where the elicited knowledge demonstrably improves transfer rather than fit.

The asymmetry between in-domain improvement for Boston Housing, Heart Disease and Diabetes and OOD improvement for Cancer deserves further attention. The datasets difer in what the source sample can identify. Boston Housing, Heart Disease and Diabetes have 242 to 316 training rows against 7 to 18 features, such that any sensible shrinkage target helps, both in-domain and out. Cancer has 1,106 rows and 13 features, thus the source fit is stable in-domain and the prior adds nothing there. Yet it adds several per cent out-of-domain. This is the behaviour of a prior correcting source-specific structure that does not transfer, which is the mechanism the approach aims for. Adult, with $n _ { \mathrm { t r a i n } } \approx 2 7 { , } 0 0 0$ , shows no efect in either direction: with such a large training sample, the posterior is dominated by the likelihood and the prior is efectively ignored, which from an OOD perspective is not necessarily optimal.

Two consequences follow for how the method should be positioned. It is most clearly useful where source data alone cannot identify stable coeficients, and in that regime its value is mostly better regularisation rather than shift correction. The stronger claim, that elicited knowledge improves transfer beyond what can be achieved by better in-domain regularisation, is supported here by Cancer alone. Testing whether this benefit generalises requires more datasets with enough source data for in-domain gains to be negligible, but with suficient domain shift for an out-of-domain gain to remain.

## 5.2 What selection does, and what it does not

Depending on the elicitation variant and prior model family, an LLM-elicited prior can either improve or degrade out-of-distribution predictive performance relative to the baseline (Figure 3). The worst candidate performs below baseline on every dataset. An LLM-elicited prior therefore has to be treated as a candidate source of information rather than an authority, and singleprovider, single-prompt, single-structure studies are capable of reaching qualitatively diferent conclusions from the same underlying method. We would encourage reporting the full candidate distribution, not only the deployed configuration.

What source-domain cross-validation does less well is pick between candidates. Measured against the standard error of the paired diference, which is the relevant scale when every candidate is scored on the same folds, the margin between the winner and the best genuinely diferent runner-up is 0.06 standard errors on three of the five datasets, 0.25 on Heart Disease and 1.63 on Diabetes; on every dataset the selected model is in the middle of the candidate range rather than near the best. The claim the evidence supports is therefore narrower than identification of the best prior: a tuned λ together with a selection rule that is at worst uninformative keeps the deployed model away from the worst candidates, and on Adult drives the prior’s influence to nothing when it has nothing to ofer. The guarantee is safety rather than optimality.

This limitation concerns the source-only selection rule used here, rather than model selection in general. Cross-validation on source data estimates source-domain performance, which need not rank candidates by their target-domain performance. Without information about the target domain, more extensive tuning on the same source data cannot resolve this mismatch. If unlabelled target features are available, importance-weighted cross-validation may provide a more target-relevant criterion under covariate shift, as discussed in Section 5.4. It does not, however, resolve changes in the relationship between features and outcomes.

## 5.3 Evaluating classifiers under label shift

Our threshold analysis, reported in full in Appendix B, illustrates a risk in comparing classifiers using $F _ { 1 }$ at a fixed decision threshold. In our Heart Disease and Diabetes splits, the positive-class prevalence is higher in the OOD population than in the source population. At a threshold of 0.5, diferences in how many observations each model classifies as positive can therefore create an apparent $F _ { 1 }$ advantage that disappears when the baseline is also allowed to select its threshold on source data. In addition to the threshold-free metrics of Table 5, we evaluated every classifier with $F _ { 1 }$ , and for threshold-dependent metrics of that kind the efect is not subtle: on Diabetes it accounts for an apparent $F _ { 1 }$ improvement of +0.089 with a confidence interval comfortably excluding zero, which disappears entirely when the baseline is allowed the same operating-point freedom. AUC is invariant to label shift and was correctly reported as unchanged; the thresholddependent metrics were not.

Three recommendations follow: report proper scoring rules, which are threshold-free and reward exactly the probability quality that prior-informed shrinkage delivers; if thresholddependent metrics are reported, tune the threshold for every model including baselines, on source data only; and document prevalence shift as a property of the design rather than an incidental detail.

## 5.4 Limitations and future work

A major threat to validity is possible overlap between the LLMs’ training corpora and the public datasets used here. Boston Housing, the UCI Heart Disease cohorts, Pima Diabetes and Adult are long-standing benchmarks, but we cannot determine whether, or in what form, any queried model encountered them during training. Prior work documents memorisation and contamination risks in tabular benchmarks [37, 38, 39]. This matters for interpreting the elicited coeficients: when an LLM gives nearly identical answers across samples, we cannot tell whether it is expressing stable domain knowledge or recalling a familiar dataset. Evaluation on datasets unlikely to have appeared in model training would provide a stronger test.

A separate concern applies to Boston Housing. That dataset has been withdrawn from major software distributions because one of its features encodes a racial-segregation hypothesis in its functional form. We remove the feature in question and fit the remaining eleven, as described in the Methods. We keep the dataset itself because it is a long-standing benchmark in this literature and including it makes our results comparable with earlier work, but we give its coeficients no causal interpretation. The conclusions do not rest on it. The claim that elicited knowledge supplies information the source domain cannot is supported by Cancer, and it would stand unchanged if Boston Housing were excluded.

Our comparisons also have a deliberately narrow scope. The matched baselines shrink coeficients toward zero, using ridge regression for the regression tasks and $L _ { 2 ^ { - } }$ regularised logistic regression for classification. They isolate the question of whether the elicited prior improves on zero-centred shrinkage within the same modelling framework; they do not establish that this approach outperforms other ways of using LLM knowledge. A direct comparison with AutoElicit or LoID would require new elicitation queries, because their prompting and extraction procedures cannot be reconstructed from our stored coeficient draws. We did not undertake that comparison here. Our stored draws allow the present analysis to be reproduced ofline, whereas new queries to commercial models could introduce diferences due to model updates unless their versions were recorded.

Each dataset contributes only one source–target split. Consequently, a gain on its OOD partition need not generalise to other plausible shifts within the same domain. The uncertainty intervals account for repeated source-data splits, re-tuning of λ, and resampling of the observed target partition, but hold the selected LLM variant and its elicited draws fixed. Repeating variant selection within each source split would assess the stability of that selection step; independently repeating elicitation would be needed to assess variability in the priors themselves. The small OOD partitions on several datasets also leave considerable uncertainty about the benefit of a single deployment. Larger target samples and multiple distinct domain splits would strengthen these conclusions.

A related interpretive limitation is that Boston Housing, Heart Disease and Diabetes show in-domain gains at least as large as their OOD gains. For those datasets, the evidence points primarily to better regularisation, not to an efect specific to domain shift. Cancer is the only dataset here with a substantial OOD gain and negligible in-domain gain. More datasets with suficient source data to make ordinary in-domain regularisation gains small, but with meaningful domain shift, would help test whether this pattern generalises.

The elicitation prompts also include author-written hints about the expected signs of dominant features (Appendix D). The placebo tests show that feature assignments and signs matter for predictive performance, but they cannot establish that the useful directions originated with the LLM rather than with these hints. The elicited magnitudes and uncertainty structure remain relevant, particularly on Boston Housing, where the prior families that use elicited spread outperform Prior-Mean. An otherwise identical elicitation without the hint block would clarif what the LLM contributes independently.

A further limitation concerns model selection under domain shift. In Eq. 5, the model is fitted to source-domain observations, and λ is selected by source-domain cross-validation. That procedure estimates source performance, which need not rank candidates by their target performance. Adult illustrates the ambiguity: cross-validation assigns the prior very little weight, but this alone cannot tell us whether a stronger prior would have transferred better. With no information about the target domain, more extensive tuning on the same source data cannot resolve that uncertainty.

If unlabelled target features are available and the conditional relationship between features and outcomes is suficiently stable, importance-weighted validation could ofer a more targetrelevant selection criterion [40, 41]. This is a diferent setting from the source-only selection studied here: it requires target covariates, adequate overlap between domains, and assumptions that may fail under concept shift. Likelihood tempering [42, 43] and alternative scaling of prior influence [44] address a related but distinct question: how strongly should source observations constrain a model relative to external information? Neither, on its own, identifies the weight that minimises target risk without target information or additional assumptions. These approaches are therefore directions for future study, rather than remedies established by the present results.

Finally, several extensions concern the priors themselves. Combining repeated coeficient sampling with AutoElicit-style prompt rephrasing could capture both within-prompt and cross prompt variability. Robust estimation of ${ \widehat { \mu } } ,$ or pooling samples across providers, could reduce sensitivity to outlying coeficient draws. Covariance shrinkage could also stabilise Correlated-

Prior, whose covariance matrix is estimated from 50 samples. A comparison of these approaches and other elicitation methods on common datasets and domain splits would help separate the value of the prior information from the way it is extracted and encoded.

## 5.5 Conclusion

LLM-elicited coeficient priors used as structured shrinkage in regularised regression improve OOD prediction on four of five real domain splits. For ten of the twenty-four prior-informed evaluations, the mean improvement is statistically significant at the standard 95% level. A placebo control attributes the performance lift over baseline to the priors’ feature-specific content rather than their scale. The efect is conditional, and largest when training data are limited relative to the number of features and absent where the training data are already suficient. However, prediction intervals for a single deployment remain wide and frequently cross zero due to variability in training splits and test-sample evaluation. Consequently, while the average benefit across repeated trials is statistically established, an improvement for any single run cannot be guaranteed.

In addition to predictive performance, two broader methodological findings emerge. First, the choice of language-model variant and prior structure matters: some combinations improve out-of-distribution predictive performance, while others degrade it relative to the baseline. Because source-domain cross-validation cannot reliably rank these candidates, the practical guarantee is protection against the worst-performing configurations rather than the identification of the optimal one; transparently reporting the full candidate distribution should therefore become standard practice. Second, evaluating classifiers at a fixed decision threshold—such as the common default of $F _ { 1 }$ at a 0.5 cut-of—substantially inflates apparent performance gains whenever the domain split alters the class distribution. In such settings, threshold-free proper scoring rules provide a much more reliable basis for comparison.

Finally, the central open question concerns whether language model priors act as true domain shift-adaptation mechanisms or merely as variance regularisers. Across most evaluated datasets, the elicited prior improves in-domain accuracy just as much as out-of-domain accuracy, indicating that it functions primarily as an efective regularisation target to counter small-sample noise rather than as a targeted remedy for domain shift. Only on Cancer, where the training sample is large enough to render in-domain regularisation gains negligible, does the prior provide an improvement specific to out-of-distribution prediction.

## Data availability

All five datasets, in the exact form used here, are in the repository below. Four are available from their original distributors: Boston Housing [30], the UCI Heart Disease database [32], the Pima Indians Diabetes Database [33] and the UCI Adult/Census Income dataset [34], the latter two under CC BY 4.0. The county-level cancer mortality compilation [31] is a special case: it was published openly on data.world, but that platform has since retired its open data community and the original page no longer resolves, so our copy is the only one we can point to. Its underlying sources: the US Census American Community Survey, Cancer.gov and ClinicalTrials.gov, are US federal government outputs and are not subject to copyright; we redistribute the compilation to keep the analysis reproducible, with attribution to its compiler, and will remove it on request.

## Code availability

The elicitation and modelling pipeline, the 50 validated coeficient samples for every (dataset, provider, mode) combination, the scripts for all follow-up analyses, and the result files and run logs behind every table and figure are available at https://github.com/hugohammer/ llm-priors-ood. Because the elicited samples are stored, every result in this paper can be reproduced without any further LLM API calls.

## Author contributions

H.L.H. conceived the project. S.L. and H.L.H. implemented the study and drafted the manuscript. K.H.H. and H.L.H. supervised S.L.’s work on the project. K.H.H. reviewed and critically revised the manuscript at several stages during its preparation. All authors read and approved the final manuscript.

## Competing interests

The authors declare no competing interests.

## References

[1] Liu, J. et al. Towards out-of-distribution generalization: A survey (2021). arXiv:2108.13624.

[2] Qui˜nonero-Candela, J., Sugiyama, M., Schwaighofer, A. & Lawrence, N. D. (eds) Dataset Shift in Machine Learning (MIT Press, Cambridge, MA, 2009).

[3] Moreno-Torres, J. G., Raeder, T., Alaiz-Rodr´ıguez, R., Chawla, N. V. & Herrera, F. A unifying view on dataset shift in classification. Pattern Recognition 45, 521–530 (2012).

[4] Lakshminarayanan, B., Pritzel, A. & Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems 30 (NIPS 2017) (2017).

[5] Purucker, L. et al. Beyond iid: How general are tabular foundation models, really? (2026). ArXiv:2606.30410.

[6] Sokolov, A., Carlin, D. E., Paull, E. O., Baertsch, R. & Stuart, J. M. Pathway-based genomics prediction using generalized elastic net. PLOS Computational Biology 12, 1–23 (2016). URL https://doi.org/10.1371/journal.pcbi.1004790.

[7] O’Hagan, A. Expert knowledge elicitation: Subjective but scientific. The American Statistician 73, 69–81 (2019).

[8] van de Schoot, R. et al. Bayesian statistics and modelling. Nature Reviews Methods Primers 1 (2021).

[9] Bishop, C. M. Pattern Recognition and Machine Learning (Springer, 2006).

[10] Hastie, T., Tibshirani, R. & Friedman, J. The Elements of Statistical Learning: Data Mining, Inference, and Prediction 2nd edn (Springer, 2009).

[11] Hoerl, A. E. & Kennard, R. W. Ridge regression: Biased estimation for nonorthogonal problems. Technometrics 12, 55–67 (1970).

[12] Brown, T. B. et al. Language models are few-shot learners. Advances in Neural Information Processing Systems 33 (2020).

[13] Bommasani, R. et al. On the opportunities and risks of foundation models (2021). arXiv:2108.07258.

[14] Minaee, S. et al. Large language models: A survey (2024). arXiv:2402.06196.

[15] Goh, E., Gallo, R., Hom, J. et al. Large language model influence on diagnostic reasoning: A randomized clinical trial. JAMA Network Open 7, e2440969 (2024).

[16] Evans, R., Felland, M., Evans, S. & Sloan, L. Large language model-derived priors can improve bayesian survival analyses: A glioblastoma application (2025). ArXiv:2511.01778.

[17] Bender, E. M., Gebru, T., McMillan-Major, A. & Shmitchell, S. On the dangers of stochastic parrots: Can language models be too big? Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency 610–623 (2021).

[18] Liu, P. et al. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys 55, Article 195 (2023).

[19] Gallegos, I. O. et al. Bias and fairness in large language models: A survey. Computational Linguistics 50, 1097–1179 (2024).

[20] Huang, L. et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems 43, Article 73 (2025).

[21] Gneiting, T. & Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102, 359–378 (2007).

[22] Riegler, M. A., Hellton, K. H., Thambawita, V. & Hammer, H. L. Using large language models to suggest informative prior distributions in Bayesian regression analysis. Scientific Reports 15, 33386 (2025).

[23] Capstick, A., Krishnan, R. G. & Barnaghi, P. AutoElicit: Using large language models for expert prior elicitation in predictive modelling. Proceedings of the 42nd International Conference on Machine Learning 267, 6746–6777 (2025).

[24] Huang, Y. LLM-Prior: A framework for knowledge-driven prior elicitation and aggregation (2025). ArXiv:2508.03766.

[25] Zhang, E. et al. LLM-Lasso: A robust framework for domain-informed feature selection and regularization (2025). arXiv:2502.10648.

[26] Requeima, J., Bronskill, J., Choi, D., Turner, R. E. & Duvenaud, D. Llm processes: Numerical predictive distributions conditioned on natural language (2024). URL https: //arxiv.org/abs/2405.12856. arXiv:2405.12856.

[27] Zeng, Y., Liu, J., Lam, H. & Namkoong, H. Llm embeddings improve test-time adaptation to tabular y|x-shifts (2024). URL https://arxiv.org/abs/2410.07395. arXiv:2410.07395.

[28] Amin, M. T. et al. Integrating machine learning ensembles and large language models for heart disease prediction using voting fusion (2026). URL https://arxiv.org/abs/2602. 22280. arXiv:2602.22280.

[29] Rezaeimanesh, S., Thind, K., Siddiqui, F. & Ghassemi, M. M. What language models know but don’t say: Non-generative prior extraction for generalization (2026). URL https: //arxiv.org/abs/2601.17609. ArXiv:2601.17609v3, arXiv:2601.17609.

[30] Harrison, D. & Rubinfeld, D. L. Hedonic housing prices and the demand for clean air. Journal of Environmental Economics and Management 5, 81–102 (1978).

[31] Rippner, N. OLS regression challenge: County-level cancer mortality. data.world (platform retired); archived copy at https://github.com/hugohammer/llm-priors-ood (2017). County-level compilation (cancer reg.csv, 3,047 counties) aggregated from the US Census American Community Survey, ClinicalTrials.gov and Cancer.gov. The original page, data.world/nrippner/ols-regression-challenge, no longer resolves following the retirement of the data.world open data community.

[32] Detrano, R. et al. International application of a new probability algorithm for the diagnosis of coronary artery disease. The American Journal of Cardiology 64, 304–310 (1989).

[33] Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C. & Johannes, R. S. Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. Proceedings of the Annual Symposium on Computer Application in Medical Care 261–265 (1988).

[34] Kohavi, R. Scaling up the accuracy of naive-Bayes classifiers: A decision-tree hybrid. Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD) 202–207 (1996).

[35] Pedregosa, F. et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12, 2825–2830 (2011). The load boston loader was deprecated in version 1.0 and removed in 1.2 over the ethical problem with the B variable.

[36] Wang, S., Kreuzer, M., Braun, U. & Schwarm, A. Efect of unconventional oilseeds (safflower, poppy, hemp, camelina) on in vitro ruminal methane production and fermentation. Journal of the Science of Food and Agriculture 97, 3864–3870 (2017).

[37] Bordt, S., Nori, H., Rodrigues, V., Nushi, B. & Caruana, R. Elephants never forget: Memorization and learning of tabular data in large language models. Conference on Language Modeling (COLM) (2024). ArXiv:2404.06209.

[38] Silvestri, M., Veglianti, F., Giorgi, F., Silvestri, F. & Tolomei, G. When large language models know the table: A framework for assessing data contamination in tabular datasets (2025). ArXiv:2510.20351.

[39] Gorla, A. & Puduppully, R. The illusion of generalization in tabular language models (2026). ArXiv:2602.04031.

[40] Shimodaira, H. Improving predictive inference under covariate shift by weighting the loglikelihood function. Journal of Statistical Planning and Inference 90, 227–244 (2000).

[41] Sugiyama, M., Krauledat, M. & M¨uller, K.-R. Covariate shift adaptation by importance weighted cross validation. Journal of Machine Learning Research 8, 985–1005 (2007).

[42] Gr¨unwald, P. & van Ommen, T. Inconsistency of Bayesian inference for misspecified linear models, and a proposal for repairing it. Bayesian Analysis 12, 1069–1103 (2017).

[43] de Heide, R., Kirichenko, A., Gr¨unwald, P. & Mehta, N. A. Safe-Bayesian generalized linear regression. Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 108, 2623–2633 (2020).

[44] Ibrahim, J. G. & Chen, M.-H. Power prior distributions for regression models. Statistical Science 15, 46–60 (2000).

[45] Saerens, M., Latinne, P. & Decaestecker, C. Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. Neural Computation 14, 21–41 (2002).

## A Supplementary figures

![](images/4d512337fda0715316b0e4b44484a548634418779f79ca311520061f571cac58.jpg)

![](images/6867408e1d03761dcaee00c0a75ab9cd1f5b3be71282ac5061636d86410135d3.jpg)

![](images/caf6c09bfe39ec1a01fb7b942274e110a8b89932dc1bf94402fd58adc213ce55.jpg)

![](images/c426882e9cf5e78a704479502d5e393fbf3f3ca2b65d6f87929cc73d087d8aa2.jpg)

![](images/0c24548698e119ff303d85e09f627ee1072e945973afd71c9d727b2d69b59588.jpg)  
Figure S1: OOD lift over baseline across ID resplits, with λ re-tuned on every resplit. Boxes show the interquartile range and median; points are individual resplits. Lift is MSE reduction for Boston Housing and Cancer and AUC gain for the classification datasets. The single-split values reported in Table 4 are in the upper part of these distributions.

![](images/75c7d147663b1d07f1dbfb01799b71eeffc8a2e0f3e6043d52f26eb9f3e9d50b.jpg)

![](images/25c288f3e21d7e3dcf65db60b5249dbf5aeaff733ba9a59316330c76b964901f.jpg)  
Figure S2: OOD calibration on Heart Disease (left) and Diabetes (right). Top panels: 10- quantile reliability curves for the deployed baseline and the deployed best-prior family, with the dotted diagonal marking perfect calibration. Bottom panels: histograms of predicted probabilities. AUC is essentially unchanged in both cases while Brier and log loss improve.

Table S1: Supplementary. OOD lift over baseline across ID resplits $( n = 5 0$ for Boston Housing and Cancer, 30 for Heart Disease and Diabetes, 10 for Adult). “Single split” is the value from Tables 4 and 5. “Below” is the fraction of resplits in which the prior-informed model is worse than baseline.
<table><tr><td>Dataset</td><td>Family</td><td>Single split</td><td>Mean</td><td>SD</td><td>[2.5, 97.5]</td><td>Below</td></tr><tr><td>Boston Housing</td><td>Prior-Mean</td><td>+2.7%</td><td>-0.02%</td><td>4.02</td><td>[−7.22, +6.55]</td><td>42%</td></tr><tr><td>Boston Housing</td><td>Independent-Prior</td><td>+10.3%</td><td>+7.52%</td><td>3.20</td><td>[+2.66, +14.43]</td><td>0%</td></tr><tr><td>Boston Housing</td><td>Correlated-Prior</td><td>+11.2%</td><td>+7.94%</td><td>3.48</td><td>[+2.09, +15.27]</td><td>0%</td></tr><tr><td>Boston Housing</td><td>Mixed</td><td>+2.7%</td><td>-0.02%</td><td>4.02</td><td>[−7.22, +6.55]</td><td>42%</td></tr><tr><td>Cancer</td><td>Prior-Mean</td><td>+3.9%</td><td>+2.37%</td><td>2.05</td><td>[−0.30, +7.25]</td><td>12%</td></tr><tr><td>Cancer</td><td>Independent-Prior</td><td>+2.8%</td><td>+1.13%</td><td>1.08</td><td>[−1.11, +3.33]</td><td>10%</td></tr><tr><td>Cancer</td><td>Correlated-Prior</td><td>+2.1%</td><td>+0.91%</td><td>1.09</td><td>[−0.89, +3.17]</td><td>16%</td></tr><tr><td>Cancer</td><td>Mixed</td><td>+2.7%</td><td>+1.65%</td><td>1.43</td><td>[−0.20, +5.10]</td><td>12%</td></tr><tr><td>Heart Disease</td><td>Prior-Mean</td><td>+0.9%</td><td>+0.08%</td><td>0.68</td><td>[−0.76, +1.50]</td><td>57%</td></tr><tr><td>Heart Disease</td><td>Independent-Prior</td><td>+0.8%</td><td>-0.07%</td><td>0.64</td><td>[−0.79, +1.20]</td><td>63%</td></tr><tr><td>Heart Disease</td><td>Correlated-Prior</td><td>+0.9%</td><td>+0.18%</td><td>0.74</td><td>[−0.61, +1.81]</td><td>50%</td></tr><tr><td>Diabetes</td><td>Prior-Mean</td><td>+0.8%</td><td>+1.40%</td><td>0.66</td><td>[+0.26, +2.62]</td><td>0%</td></tr><tr><td>Diabetes</td><td>Independent-Prior</td><td>+1.1%</td><td>+1.78%</td><td>0.77</td><td>[+0.59, +3.15]</td><td>0%</td></tr><tr><td>Diabetes</td><td>Correlated-Prior</td><td>+1.0%</td><td>+1.81%</td><td>0.78</td><td>[+0.59, +3.20]</td><td>0%</td></tr><tr><td>Adult</td><td>all families</td><td>≈0</td><td>-0.01%</td><td>≤ 0.02</td><td>[−0.05, +0.00]</td><td></td></tr></table>

## B Threshold-dependent metrics under label shift

We also computed the threshold-dependent metrics, and report them here rather than in the main results because they turn out to measure the decision threshold more than the model. Both classification datasets show large and apparently significant $F _ { 1 }$ improvements at the conventional 0.5 threshold: $+ 0 . 0 4 8 \ [ + 0 . 0 1 7 , + 0 . 0 8 0 ]$ for Prior-Mean on Heart Disease and +0.089 $\left[ + 0 . 0 3 9 , + 0 . 1 4 3 \right]$ on Diabetes, significant in all eight results (four prior families on two datasets). These gains are an artefact of the decision threshold, as we now show.

The domain splits induce label shift as well as covariate shift: outcome prevalence moves from 0.212 in the Diabetes source training pool to 0.495 in the OOD set, and from 0.459 to 0.530 on Heart Disease. A threshold calibrated for one base rate is not appropriate for another [45], so 0.5 is not a neutral comparison point: any model whose predicted probabilities are shifted upward will appear better on $F _ { 1 }$ without being a better model. Since the prior-informed models are shrunk toward a non-zero target, they produce more extreme probabilities than the baseline and are closer to the appropriate operating point by accident.

Giving every model, baseline included, a threshold chosen by 5-fold CV on source data only removes the efect entirely. Cross-validation selects 0.33 on Heart Disease, 0.25 on Diabetes and 0.33 on Adult: for every model, prior-informed or not. At those thresholds all eight previously significant $F _ { 1 }$ gains lose significance. On Diabetes the recalibrated baseline reaches $F _ { 1 } = 0 . 7 2 3 6$ above the 0.6472 that the prior-informed model attains at 0.5; recalibration alone gains the baseline +0.166, nearly twice the +0.089 credited to the prior. Independent-Prior ends up behind the recalibrated baseline. Figure S3 shows why: both $F _ { 1 }$ curves peak well below 0.5, and at 0.5 the baseline has already fallen of its peak while the prior-informed model has not.

![](images/8974ab3a027a4ca4d4d5c78429f59b7b2537ce5ba2d74f4005cf26b1040e6130.jpg)

![](images/206422fccb266a7022555c81273bdc01d0ca454a34cf733fb4f18e4f7c26566f.jpg)  
Figure S3: $F _ { 1 }$ on the OOD set as a function of the classification threshold, for the deployed $L _ { 2 }$ baseline (blue) and the deployed Prior-Mean model (orange). Crosses mark $F _ { 1 }$ at the fixed 0.5 threshold; filled circles mark $F _ { 1 }$ at the threshold chosen by 5-fold cross-validation on source data only, which is the operating point a practitioner would actually deploy. Panel titles give the prevalence shift from the source training pool to the OOD set. Both curves peak well below 0.5 because prevalence rises between the domains, so at 0.5 the baseline has already fallen of its peak while the prior-informed model has not. The prior-informed model’s apparent $F _ { 1 }$ advantage at 0.5 is therefore an operating-point efect, and it disappears once the baseline is given the same threshold freedom.

We report this as a methodological result rather than a footnote. Threshold-dependent metrics at a fixed cut-of are common in this literature, and under label shift they can manufacture an efect where none exists. The threshold-free improvements in log loss and Brier score reported above are unafected by this critique, and are the defensible form of the classification claim.

## C Sensitivity to the variance floor

The Independent-Prior penalty weight on coeficient $j$ is $\lambda / ( \widehat { \sigma } _ { j } ^ { 2 } + \varepsilon )$ , so when an elicitation is unanimous and $\widehat { \sigma } _ { j } ^ { 2 } = 0$ the floor $\varepsilon ,$ not the elicitation, sets how hard that coeficient is held at $\widehat { \mu } _ { j }$ . One deployed variant returns bit-identical draws on four of five datasets, so the question is whether the reported lifts are a property of the elicited spread or of the floor. Table S2 re-tunes λ and refits at each of five values of ε spanning seven orders of magnitude, and for Correlated-Prior also replaces the jittered inverse with a Ledoit–Wolf shrinkage estimator of $ { \widehat { \Sigma } } ,$ which is the principled alternative.

The headline results are not artefacts. Boston Housing’s Correlated-Prior lift is +11.15% at the deployed setting and +11.10% under Ledoit–Wolf, and Cancer is flat to two decimal places across the whole range. Lifts decay only once ε reaches $1 0 ^ { - 2 } \ \mathrm { o r } \ 1 0 ^ { - 1 }$ , which is where ε becomes larger than the elicited variances themselves and the Independent-Prior collapses toward Prior-Mean, so the decay is evidence that the elicited spread contains information, not evidence against it. Heart Disease’s Independent-Prior is the most floor-sensitive case, falling from +0.75% to +0.09%, and that is the one place where the reported number should be read with the floor in mind.

Table S2: OOD lift (%) for the two families whose precision depends on the elicited spread, as the variance floor ε is varied and λ re-tuned at each setting. “Zero-spread” counts features for which the deployed variant returned bit-identical values across all $M = 5 0$ draws, so that $\widehat { \sigma } _ { j } ^ { 2 } = 0$ exactly. The Ledoit–Wolf column replaces the jittered pseudo-inverse of $\widehat { \Sigma }$ with a shrinkage estimator and applies only to Correlated-Prior. The deployed configuration is $\varepsilon = 1 0 ^ { - 8 }$ . Lift is MSE reduction for the regression datasets and AUC gain for the classification ones.
<table><tr><td>Dataset</td><td>Family</td><td>Zero-spread</td><td>10⁻8</td><td> $1 0 ^ { - 6 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 2 }$ </td><td>10⁻¹</td><td>Ledoit-Wolf</td></tr><tr><td></td><td>Boston Housing Independent-Prior</td><td>4/11</td><td></td><td></td><td>+10.29 +10.29 +10.30 +9.27 +7.53</td><td></td><td></td><td></td></tr><tr><td></td><td>Boston Housing Correlated-Prior</td><td>4/11</td><td>+11.15</td><td>+11.15</td><td>+11.15 +9.70 +7.69</td><td></td><td></td><td>+11.10</td></tr><tr><td>Cancer</td><td>Independent-Prior</td><td>0/13</td><td>+2.78</td><td>+2.78</td><td></td><td>+2.78 +2.75 +2.44</td><td></td><td></td></tr><tr><td>Cancer</td><td>Correlated-Prior</td><td>0/13</td><td>+2.12</td><td>+2.12</td><td></td><td>+2.12 +2.12 +2.08</td><td></td><td>+2.20</td></tr><tr><td>Heart Disease</td><td>Independent-Prior</td><td>0/18</td><td>+0.75</td><td>+0.75</td><td></td><td></td><td>+0.73 +0.41 +0.09</td><td></td></tr><tr><td>Heart Disease</td><td>Correlated-Prior</td><td>2/18</td><td>+0.94</td><td>+0.95</td><td></td><td></td><td>+0.90 +0.93 +0.93</td><td>+0.93</td></tr><tr><td>Diabetes</td><td>Independent-Prior</td><td>2/7</td><td>+1.00</td><td>+1.00</td><td></td><td></td><td>+0.90 +0.86 +0.85</td><td></td></tr><tr><td>Diabetes</td><td>Correlated-Prior</td><td>2/7</td><td>+0.97</td><td>+0.97</td><td></td><td></td><td>+0.97 +0.86 +0.84</td><td>+0.87</td></tr><tr><td>Adult</td><td>Independent-Prior</td><td>0/20</td><td>-0.03</td><td>-0.01</td><td></td><td>-0.00 +0.00 +0.00</td><td></td><td></td></tr><tr><td>Adult</td><td>Correlated-Prior</td><td>0/20</td><td>-0.01</td><td>-0.01</td><td></td><td></td><td>-0.03 +0.00 -0.00</td><td>-0.01</td></tr></table>

## D Example elicitation prompt

The prompts are assembled from a fixed system part and a dataset-specific user part, as described in Section 3.7. Both are reproduced verbatim below for the Diabetes dataset in non-reasoning sampling mode; the other datasets difer only in the user part. Three non-ASCII characters in the original (µ, an em dash and ≈) are rendered here as u, -- and \~=, and long lines are wrapped. Note that the feature-scale summaries are computed on the source training pool only, and that the domain context describes the source population without revealing the out-of-distribution target.

## System prompt

You are helping construct informative priors for a regression model. Adopt the perspective of an endocrinologist and applied statistician with experience modelling type 2 diabetes risk in screening data.

Use the feature definitions and feature-scale summaries to reason about BOTH sign and magnitude.

All coefficients must be given in the original raw-feature units used by the model matrix.

Do not standardise, rescale, or reinterpret the coefficients as standardised effects.   
Do not include an intercept.

Output must be only a JSON object matching the provided schema.

Do not include explanations, markdown, code fences, or extra keys.

Return one plausible coefficient vector draw from your belief distribution.

Do not collapse uncertainty into a single average if multiple plausible settings

Keep the draw realistic and domain-grounded rather than intentionally noisy.

Prefer interpretable, conservative, domain-plausible coefficients.

Avoid extreme magnitudes unless they are clearly justified by the variable scale and domain mechanism.

## User prompt

## Task context

\- Binary logistic regression for the positive class in the Pima Indians Diabetes outcome (Outcome=1 = patient developed diabetes within 5 years of the visit): logit(P(y=1)) = intercept + sum\_j beta\_j \* x\_j

## Domain context:

\- Source domain is younger patients (Age <= dataset median, computed on the source-domain subset).

\- Priors should encode stable physiology associations likely to transfer beyond the source age range.

\- Cohort is Pima Indians women aged 21+; the data was collected to study type 2 diabetes onset.

## Dataset notes:

\- Age is used only for the OOD split definition and is excluded from predictors.

\- Several features (Glucose, BloodPressure, SkinThickness, Insulin, BMI) have observed zeros that are likely missing values; the pipeline imputes them with the source-domain median.

\- DiabetesPedigreeFunction is a derived risk score capturing family history of diabetes; it is unitless on a roughly 0-3 scale.

## Guidance:

\- Use signs and magnitudes consistent with established type 2 diabetes risk-factor evidence.

\- Prefer conservative values when uncertainty is high.

\- Avoid extreme magnitudes unless there is a strong physiological rationale.

## Feature meanings:

\- Pregnancies: number of times pregnant

\- Glucose: plasma glucose concentration after 2-hour oral glucose tolerance test (mg/dL)

\- BloodPressure: diastolic blood pressure (mm Hg)

\- SkinThickness: triceps skin fold thickness (mm)

\- Insulin: 2-hour serum insulin (uU/mL)

\- BMI: body mass index (weight in kg / height in m^2)

\- DiabetesPedigreeFunction: diabetes pedigree function -- a score reflecting family history of diabetes

Feature scale summaries from the actual design matrix:

\- Pregnancies: continuous/raw numeric; median=2, IQR=[1, 3], central95%\~=[0, 5]

\- Glucose: continuous/raw numeric; median=110, IQR=[95, 130], central95%\~=[78.7, 174], missing=0.006

\- BloodPressure: continuous/raw numeric; median=68, IQR=[60, 76], central95%\~=[50, 88.1], missing=0.051

\- SkinThickness: continuous/raw numeric; median=27, IQR=[19, 35], central95%\~=[12.6, 45], missing=0.203

\- Insulin: continuous/raw numeric; median=105, IQR=[66, 168], central95%\~=[36, 326], missing=0.383

\- BMI: continuous/raw numeric; median=31.2, IQR=[25.9, 36.8], central95%\~=[21, 44.6], missing=0.019

\- DiabetesPedigreeFunction: continuous/raw numeric; median=0.369, IQR=[0.241, 0.591], central95%\~=[0.143, 0.964]

## Relevant domain patterns to weigh alongside the summaries:

\- Higher Glucose is the strongest single predictor; expect a clear positive coefficient.

\- Higher BMI raises diabetes risk; the coefficient should be positive but smaller than Glucose’s.

\- Higher Pregnancies and DiabetesPedigreeFunction tend to increase risk modestly.

\- BloodPressure, SkinThickness, and Insulin effects are typically smaller and noisier; coefficients can be near-zero.

## Important modeling reminders:

\- Binary indicators use 0/1 coding, so their coefficients represent the change associated with turning the indicator on.

\- Continuous-feature coefficients should be scaled to the feature summaries shown above; do not assume a one-unit change is small unless the summary suggests it is.

\- When multiple variables capture overlapping structure, distribute signal conservatively rather than giving each one a large effect.

\- The output schema already fixes the keys. Choose only the numeric values.