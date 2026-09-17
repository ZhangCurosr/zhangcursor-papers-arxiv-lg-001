# Regional Explanations via Causal Sufficiency and Necessity

Xuexin Chen, Peng Liang, Zijian Li, Zhiyong Lin, and Ruichu Cai<sup>⋆</sup>

Abstract—Model explainability is essential for understanding and trusting machine learning models. Existing explainable AI methods often explain predictions through feature importance, counterfactual explanations, or rules. However, a region-level characterization of when and only when a prediction behavior arises remains less explored. This paper proposes Causal Sufficient and Necessary Regional Explanations (SNRE), a framework that learns an input region A and output region B such that membership in A is both sufficient and necessary for the model output to fall in B. Motivated by the classical Probability of Necessity and Sufficiency (PNS), we formulate a region-level PNS measure through stochastic interventions and derive a differentiable finite-sample estimator for optimization. SNRE parameterizes the input-output region pair with explicit and interpretable algebraic region families, together with a learnable feature mask, balancing expressiveness and interpretability. Experiments demonstrate that SNRE learns region pairs with strong sufficiency-necessity performance, robust explanation behavior, and practical utility for model analysis.

Keywords: Causal inference, causal explanation, model explainability, probability of necessity and sufficiency, stochastic intervention.

## I. INTRODUCTION

Deep neural networks (DNNs) have achieved remarkable success across a wide range of domains, yet their highly nonlinear and complex architectures make it difficult to understand why specific predictions are made. This lack of transparency has motivated the rapid development of Explainable AI (XAI) [1]–[3], which seeks to make model behavior and predictions interpretable to human users [4]–[6]. Broadly, the XAI literature explains model behavior from multiple perspectives, including which input factors contribute to a prediction [7], [8], how the prediction can be changed by minimally modifying a given input [9], and what interpretable rules are associated with a target prediction behavior [10].

![](images/a76a3124830a55e6eca5d3595bcb3e1e71ed49bc303b8553397f2d39c66b6998.jpg)  
Fig. 1. Conceptual comparison between conventional rule/region explanations and SNRE (Sufficient and Necessary Regional Explanations). Conventional rule/region explanations identify an input-side region A associated with a target prediction, yet samples outside A may yield the same output. SNRE learns an input-output region pair (A, B) by contrasting in-region and outof-region interventions, capturing both A → B and A<sup>¯</sup> → B<sup>¯</sup>.

Beyond these established efforts, a fundamental but largely unexplored question in XAI is: can we identify an input-output region pair that exhibits a rigorous causal relationship? As illustrated in Fig. 1, rule-based explanations [11], [12] are the closest existing form to such a regional view, since a rule antecedent naturally defines an input-side condition. However, such regions typically specify where a target prediction may hold, but not where the same prediction becomes unlikely. This leaves the sufficiency-necessity structure of the prediction behavior unexplained. Formally, given a trained model f, we seek to discover a region A in the input space and a target region B in the output space: intervening to place the input within A induces outputs in B, while forcing the input outside A prevents outputs from falling in B. Such sufficient-andnecessary region pairs reveal when and only when a prediction behavior arises, and capture stable bidirectional regularities in the model’s input-output mapping. Such a region-level structure provides a principled basis for downstream model analysis, such as examining changes in learned prediction regions after fine-tuning and guiding informative data selection.

To formalize such region-level causal relationships, the classical measure of causal sufficiency and necessity in causal inference, the Probability of Necessity and Sufficiency (PNS) [13], provides a natural theoretical foundation. However, extending this perspective to explanations in the form of continuous regions for DNNs face two fundamental challenges. First, classical PNS is defined for binary causes and outcomes, while regional explanations concern events X∈A and $f ( \mathbf { X } ) { \in } B$ . This raises a basic question: how to define and estimate sufficiencynecessity strength for continuous input-output regions. Existing PNS-based XAI methods [14], [15] largely bypass this difficulty by evaluating selected features or masks, without directly learning continuous causal regions in both input and output spaces. Second, how to design an expressive but interpretable boundary representation for non-linear input/output region that reduces the geometric complexity while preserving the attractive coverage property of necessity and sufficiency.

We propose Sufficient and Necessary Regional Explanations (SNRE), a framework that learns input-output region pairs with high sufficiency-necessity strength by optimizing a region-level PNS objective motivated by classical causal sufficiency and necessity. We first formulate a region-level PNS measure for input and output regions, motivated by the core semantics of classical PNS. The key requirement is that the intervention distributions cover both the region and its complement, rather than collapsing to a narrow subset. This is naturally realized by two complementary stochastic interventions [13], which draw in-region and out-of-region inputs from the data distribution. Based on this formulation, we derive a finite-sample estimator of region-level PNS and further construct a differentiable relaxation for optimization. SNRE then parameterizes the input-output region pair with explicit and interpretable algebraic region families, together with a learnable soft feature mask for sparse feature selection. By jointly optimizing the region parameters and the mask, SNRE learns both the regional boundaries and the sparse features that characterize the sufficiency-necessity relation. The learned region pairs can further serve as explicit structural summaries of model behavior, enabling downstream analyses such as tracking regional changes after model fine-tuning and guiding informative data selection. Finally, we conduct extensive experiments to evaluate the sufficiency-necessity performance, structural rationality, robustness, and practicality of SNRE.

The main contributions of our work are as follows: 1) We formulate a region-level PNS measure for input-output regions, motivated by classical causal sufficiency and necessity, and instantiate it through complementary stochastic interventions with a finite-sample estimator and differentiable relaxation. 2) We propose an end-to-end SNRE framework that learns sufficient-and-necessary input-output region pairs by optimizing the region-level PNS objective under explicit and interpretable algebraic region parameterizations with a learnable soft feature mask. 3) We conduct comprehensive experiments to validate the explanation quality, structural rationality, robustness, and practical utility of SNRE.

## II. RELATED WORK

## A. Regional Explanations

Regional explanations describe model behavior over inputspace regions. A common realization is rule-based explanation, where rule antecedents naturally define an input region A. Early work by Cohen [16] develops efficient rule induction for classification, and Wang and Rudin [17] introduce Bayesian rule sets to represent predictions with compact logical conditions. Lakkaraju et al. [18] learn interpretable decision sets optimized for accuracy, conciseness, and low overlap. Yang et al. [19] learn scalable Bayesian rule lists from premined candidate rules, while Wang and Rudin [20] impose a monotonic falling probability structure along ordered rules. Angelino et al. [21] learn certifiably optimal rule lists, and Proenc¸a and van Leeuwen [22] select compact probabilistic rule lists under the minimum description length principle. Pellegrina et al. [23] further improve the scalability of rule-list learning on large datasets. Tree-based methods also induce regional explanations by recursively partitioning the input space. Loh [24] reviews decision-tree methods whose leaves naturally define input regions, and Good et al. [25] combine sparse feature transformation learning with differentiable decisiontree fitting to obtain compact trees. Recent differentiable and logic-based learners further improve rule learning through neural optimization or logical constraints. Wang et al. [26] learn non-fuzzy logical rules via differentiable relaxation, Qiao et al. [27] incorporate logical rule components into neural architectures, and Ghosh et al. [28] learn sparse propositional rules through an incremental framework.

However, these methods mainly define input-side regions associated with target predictions, rather than directly learning continuous input-output region pairs. In contrast, SNRE learns such region pairs based on sufficiency-necessity principles.

## B. PNS-based Explanations

Probability of Necessity and Sufficiency (PNS) [29], [30] provides a causal perspective for post-hoc explanations by evaluating whether an explanatory factor is sufficient to preserve a prediction and necessary for producing it. Existing studies instantiate this principle over different explanatory units. Watson et al. [31] identify minimal feature subsets with necessity and sufficiency guarantees. Balkir et al. [32] assess token-level text explanations through necessity and sufficiency. Tan et al. [33] evaluate graph subgraphs via factual and counterfactual reasoning. Galhotra et al. [34] compute attribute-level probabilistic contrastive counterfactual scores for local, global, and contextual explanations. Cai et al. [15] optimize PNS lower bounds for graph explanations, while Chen et al. [14] estimate feature-level necessity and sufficiency probabilities through intervention-based attribution.

However, these methods mainly evaluate sufficiency and necessity over features, tokens, subgraphs, attributes, or masks, rather than directly learning continuous regions in both the input and output spaces. In contrast, SNRE directly learns interpretable input-output region pairs under a region-level sufficiency-necessity formulation.

## III. BACKGROUND AND PROBLEM FORMULATION

We start with some basic notation. We use an uppercase letter, such as X, to denote a random variable, and a lowercase letter, such as x, to denote its realization when the context is clear. A bold uppercase letter, such as X, denotes a random vector, while a bold lowercase letter, such as x, denotes a vector in the corresponding value space. For a random vector $\mathbf { X } ,$ we write $\mathrm { V a l } _ { \mathbf { X } }$ for its value space, so that $\mathbf { x } \in \mathrm { V a l } _ { \mathbf { X } } . \ \mathbf { A }$ fixed trained predictor is denoted by $f : { \mathrm { V a l } } _ { \mathbf { X } } \to { \mathrm { V a l } } _ { \mathbf { Y } }$ . We use $A \subseteq \mathrm { V a l } _ { \mathbf { X } }$ and $B \subseteq \operatorname { V a l } _ { \mathbf { Y } }$ to denote input and output regions, respectively.

## A. Causal Sufficiency and Necessity

In the context of model interpretability, rather than modeling underlying real-world causal mechanisms, we adopt existing model-based causal perspective [35], where the input variables X are treated as causes that without parents, while the model output Y as the effect. Interventions correspond to actively setting the input to specific values or regions and observing the resulting model output. Accordingly, sufficiency and necessity are defined with respect to the behavior of a fixed trained predictor under such input interventions.

We consider explanations in terms of regions in the input and output spaces. Let $A \subset \operatorname { V a l } _ { \mathbf { X } }$ denote an input region and $B \subset \operatorname { V a l } _ { \mathbf { Y } }$ denote an output region of interest.

At the region level, we distinguish three different causal relationships between the events $\mathbf { X } \in \ A$ and $\mathbf { Y } \in B , \ 1 )$ Sufficient but unnecessary cause A: the input region A is said to be a sufficient but unnecessary cause of the output region B if intervening to place the input within A is enough to induce an output in $B ,$ but $B$ may still be obtained even when the input is outside $A . 2 )$ necessary but insufficient cause A: A is a necessary but insufficient cause of B if, without placing the input within $A ,$ the output would not fall in B, but intervening to place the input within A alone does not guarantee an output in B. 3) sufficient and necessary cause A: A is a sufficient and necessary cause of B, if placing the input within A induces an output in $B ,$ and without placing the input within $A ,$ an output in B would not be obtained.

These notions capture a causal regularity at the region level: the occurrence of the event $\mathbf { X } \in A$ explains the occurrence of the event $\mathbf { Y } \in B$ by being both required and adequate.

## B. Problem Formulation

The region level notions of sufficiency and necessity above are conceptually appealing, but turning them into a practical objective requires a quantitative measure that can be computed and optimized for predictors.

In this work, we consider a trained deterministic predictor $f : { \mathrm { V a l } } _ { \mathbf { X } } { \xrightarrow { } } { \mathrm { V a l } } _ { \mathbf { Y } }$ and aim to estimate an input region $A \subset$ $\mathrm { V a l } _ { \mathbf { X } }$ and an output region $B \subset \operatorname { V a l } _ { \mathbf { Y } }$ such that membership in A exhibits a strong sufficiency-necessity relation with membership in B, as quantified by a PNS-based measure. We focus on the common setting where f is fixed after training and differentiable with respect to its input. Our goal is to develop a computable and optimizable formulation that enables learning such sufficient-and-necessary region pairs from finite data. Here, “deterministic” means that a fixed input x uniquely determines $f ( \mathbf { x } )$ . This includes standard sigmoid or softmax classifiers, whose probability or score outputs are deterministic functions of the input. Models with stochastic prediction, such as Bayesian neural networks [36] with sampled parameters, are beyond the scope of this work.

Remark 1 (Relation to decision regions): For a fixed predictor and output event B $( \mathbf { e } . \mathbf { g } . , B = [ 0 . 5 , 1 ] )$ , the exact decision region can be written as $S ^ { B } = \{ \mathbf { x } \in \operatorname { V a l } _ { \mathbf { X } } \mid f ( \mathbf { x } ) \in B \}$ , which satisfies $\mathbf { x } \in { \mathcal { S } } ^ { B } \Leftrightarrow f ( \mathbf { x } ) \in B .$ . However, for neural predictors, $\boldsymbol { \mathcal { S } } ^ { B }$ is usually implicit, highly non-linear, and difficult to interpret [37]. Therefore, our goal is not to recover the full decision region, but to extract an explicit and geometrically interpretable region pair $( A , B )$ that satisfies the sufficiencynecessity relation with high probability.

## IV. REGION-LEVEL PNS VIA STOCHASTIC INTERVENTIONS

We begin by revisiting the classical Probability of Necessity and Sufficiency (PNS) in causal inference, and use its core semantics to motivate a region-level sufficiency-necessity measure for input-output regions through stochastic interventions.

## A. Motivating Region-level PNS from Classical PNS

In causal inference, the PNS [13], [29], [30], introduced by Pearl, provides a principled way to quantify whether a cause is both sufficient and necessary for an outcome. Classical PNS is defined for binary causes and binary outcomes, where events correspond to whether a variable takes value 1 or 0. Let $X \in$ $\{ 0 , 1 \}$ be a binary cause and $Y \in \{ 0 , 1 \}$ be a binary outcome. Using the potential outcomes notation [38], let $Y _ { x }$ denote the outcome that would be observed under the hard intervention $d o ( X = x )$ . The classical PNS is defined as

$$
{ \mathrm { P N S } } : = P \left( Y _ { 1 } = 1 \ \wedge \ Y _ { 0 } = 0 \right) .\tag{1}
$$

This quantity measures the probability that setting $X ~ = ~ 1$ would make the outcome occur, while setting $X = 0$ would prevent the outcome from occurring.

A key observation is that PNS compares the model’s output behavior under two complementary interventions: one that forces the input to satisfy a specified condition, and another that forces it to violate that condition. In the binary setting, “satisfying the condition” corresponds to setting $X { = } 1$ , while “violating it” corresponds to setting $X { = } 0$ , i.e., $d o ( X { = } 1 )$ versus do $( X { = } 0 )$

However, this mechanism does not directly extend beyond binary variables. When the input space is not binary, causal statements are naturally expressed as events of set membership, e.g., the event $\mathbf { X } { \in } A$ versus $\mathbf { X } \notin A .$ , rather than equality to a single value. An event-level analogue of PNS thus requires contrasting interventions that enforce $\mathbf { X } { \in } A$ and $\mathbf { X } \notin A$ . The standard hard intervention form cannot directly realize such event-level interventions. A hard intervention $d o ( \mathbf { X } = \mathbf { x } )$ fixes the input to a single value, which degenerates the induced interventional distribution to a point. As a result, it cannot represent the typical model behavior over inputs inside versus outside a set (or region) A, as illustrated in Fig. 2.

![](images/3a7952d6d4652519612d135500842281df664ed72138b4b92a15faf433f3920c.jpg)  
Fig. 2. Illustration of why hard interventions fail for region-level events. In the binary case, hard interventions exactly match the two opposite target events. In the region case, hard interventions collapse the value space to singleton events and therefore cannot realize the contrast between $X \in A$ and ${ \bar { X ( \in { \bar { A } } . } }$

## B. Stochastic Interventions for Regional Events

To preserve sufficiency-necessity semantics at the event level, interventions must induce non-degenerate coverage over both the region of interest and its complement within a common domain. We therefore adopt two complementary stochastic intervention [13] regimes defined with respect to the input distribution. Specifically, let $P ( \mathbf { X } )$ denote the distribution over the input space, and let $\mathcal { D } \subseteq \mathrm { V a l } _ { \mathbf { X } }$ be a reference input domain on which input events are defined. A canonical choice is the support of the input distribution,

$$
{ \mathcal { D } } : = \operatorname { s u p p } ( P ( \mathbf { X } ) ) .\tag{2}
$$

The domain D is not restricted to purely continuous spaces and may include discrete or mixed components. In practice, $\mathcal { D }$ is instantiated implicitly by the observed samples $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { n } .$

For a measurable set $A \subseteq { \mathcal { D } }$ (interpreted as a region when $\mathcal { D }$ is continuous), define its complement as $\bar { A } { : = } { \mathcal { D } } \backslash A$ . Conditioning $P ( \mathbf { X } )$ on A and A<sup>¯</sup> yields two restricted-support distributions:

$$
P ^ { A } ( \mathbf { X } ) : = P ( \mathbf { X } \mid \mathbf { X } \in A ) , P ^ { { \bar { A } } } ( \mathbf { X } ) : = P ( \mathbf { X } \mid \mathbf { X } \in { \bar { A } } ) .\tag{3}
$$

These distributions define two complementary stochastic intervention schemes, denoted by $\pi ( \mathbf { X } ; P ^ { A } )$ and $\pi ( \mathbf { X } ; P ^ { \bar { A } } )$ Each scheme independently samples a value x from the corresponding conditional distribution and applies the hard intervention $d o ( \mathbf { X } \mathbf { = x } )$ , reflecting two separate counterfactual worlds as in the classical PNS formulation. By construction, the supports of $P ^ { A }$ and $P ^ { \bar { A } }$ form a partition of the domain D, and provide non-degenerate coverage of both A and A<sup>¯</sup>.

## C. Region-Level PNS and Estimation

We now define region-level PNS under the complementary stochastic intervention policies $\pi ( \mathbf { X } ; P ^ { A } )$ and $\pi ( \mathbf { X } ; P ^ { \bar { A } } )$ . Let ${ \bf Y } _ { \pi ( { \bf X } ; P ^ { A } ) }$ denote the model output under policy $\pi ( \mathbf { X } ; P ^ { A } )$ , and define ${ \bf Y } _ { \pi ( { \bf X } ; P ^ { \bar { A } } ) }$ analogously. The output space Val<sub>Y</sub> is given by the value space of the trained model, $\mathrm { e . g . , \mathbb { R } ^ { | \mathbf { Y } | } }$ or $( 0 , 1 )$ . For any measurable output set $B { \subset } \mathrm { V a l } _ { \mathbf { Y } }$ , with complement ${ \bar { B } } : = \operatorname { V a l } _ { \mathbf { Y } } \setminus B$ , we consider the output events $\mathbf { Y } \in B$ and $\mathbf { Y } \in \bar { B }$ . Recall that the input domain is ${ \mathcal { D } } : = \operatorname { s u p p } ( P ( \mathbf { X } ) )$ , where input events and their complements are defined.

![](images/fecac8d74586a727ff073af5bb60908ad53a6c730062f97200773e0ce09e956d.jpg)  
Fig. 3. Monte Carlo estimates of PNS(A, B<sup>⋆</sup>) as the continuous input region A is shifted away from an optimal reference region $A ^ { \star }$ under different sample sizes m.

Definition 1 (Region-level PNS): For a measurable input set $A \subset \mathcal { D }$ and a measurable output set $B \subset \operatorname { V a l } _ { \mathbf { Y } }$ , the region-level probability of sufficiency and necessity is defined as

$$
\operatorname { P N S } ( A , B ) : = P { \big ( } \mathbf { Y } _ { \pi ( \mathbf { X } ; P ^ { A } ) } \in B \ \land \ \mathbf { Y } _ { \pi ( \mathbf { X } ; P ^ { \bar { A } } ) } \in { \bar { B } } { \big ) } .\tag{4}
$$

This quantity measures the sufficiency-necessity strength between the input event $\mathbf { X } \in { \cal A }$ and the output event $\mathbf { Y } \in B$ under complementary stochastic interventions.

Since the predictor f is deterministic and the input variables X are treated as exogenous under the model-based causal perspective [35], the output is a deterministic function of the intervened input. The two intervention policies are sampled independently, so the joint event in (4) can be estimated by separately sampling from the in-region and outof-region intervention distributions. Given m i.i.d. samples $\{ \mathbf { x } _ { i } ^ { \mathrm { i n } } \} _ { i = 1 } ^ { m } { \sim } P ( \mathbf { X } | \mathbf { X } { \in } A )$ and $\{ \mathbf { x } _ { i } ^ { \mathrm { o u t } } \} _ { i = 1 } ^ { m } { \sim } P ( \mathbf { X } \mid \mathbf { X } \in \bar { A } )$ , we estimate $\mathrm { P N S } ( A , B )$ via

$$
\widehat { \mathrm { P N S } } ( A , B ) = \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathbb { I } \big [ f ( \mathbf { x } _ { i } ^ { \mathrm { i n } } ) \in B \big ] \right) \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathbb { I } \big [ f ( \mathbf { x } _ { i } ^ { \mathrm { o u t } } ) \in \bar { B } \big ] \right) .\tag{5}
$$

As $m  \infty , { \widehat { \mathrm { P N S } } } ( A , B )$ converges to $\mathrm { P N S } ( A , B )$ whenever both conditional distributions are well defined, i.e., both A and A<sup>¯</sup> have nonzero probability mass under $P ( \mathbf { X } )$ . A formal justification is provided in Appendix A. In implementation, we require both A and A<sup>¯</sup> to contain empirical samples to avoid trivial degeneration.

Fig. 3 visualizes $\widehat { \mathrm { P N S } } ( A , B ^ { \star } )$ when the continuous input region A is progressively shifted away from a reference region $A ^ { \star }$ , while the output region is fixed to $B ^ { \star }$ . Here $( A ^ { \star } , B ^ { \star } )$ is a constructed region pair that attains the highest PNS in this controlled setting. Each curve corresponds to a different sample size m used in (5). When $m = 1$ , the estimator uses only one sampled intervention per side, and even incorrect regions can achieve spuriously high PNS values, reflecting that such sparse intervention samples cannot represent the typical model behavior over a region and therefore cannot reliably distinguish $A ^ { \star }$ from incorrect alternatives. As m increases, the estimator increasingly captures the model’s typical output behavior over the conditional distributions on A and A<sup>¯</sup>: $\widehat { \mathrm { P N S } } ( A , B ^ { \star } )$ peaks sharply near $A ^ { \star }$ and decays monotonically as A moves away from it. This demonstrates that region-level PNS can meaningfully quantify sufficiency-necessity strength over continuous input-output regions.

![](images/65aef102ce86b1a8eb5716fe47045f6272e963ffbf284d1ee485b8e501ef711b.jpg)  
Fig. 4. SNRE framework for learning sufficient-and-necessary input-output region pairs. SNRE first parameterizes the input region $A _ { \boldsymbol { \theta } , \mathbf { m } }$ and output region $\bar { B _ { \phi } } .$ , and then estimates PNS through outer marginalization and two parallel weighted aggregation modules. The A-weighted module evaluates how strongly inputs inside $A _ { \boldsymbol { \theta } , \mathbf { m } }$ induce predictions inside $\bar { B _ { \phi } } .$ , while the A<sup>¯</sup>-weighted module evaluates how strongly inputs outside $A _ { \boldsymbol { \theta } , \mathbf { m } }$ induce predictions outside $\bar { B _ { \phi } } .$ The final PNS objective is obtained by averaging the product of the two aggregation terms over the training samples used for marginalization.

Remark 2 (Model-based intervention and identification): Unlike classical causal identification, SNRE does not aim to recover unobserved potential outcomes of an unknown realworld structural causal model from observational data. Instead, it treats the fixed trained predictor as the causal system [35], whose responses under input interventions can be directly evaluated. The remaining problem is therefore finite-sample estimation of the stochastic intervention distributions, rather than identification of unknown counterfactual responses.

## V. SNRE: SUFFICIENT AND NECESSARY REGIONAL EXPLANATIONS

Building on the region-level PNS formulation in Section IV, we now present Sufficient and Necessary Regional Explanations (SNRE), an end-to-end framework for learning inputoutput region pairs $( A , B )$ with high sufficiency-necessity strength. As illustrated in Fig. 4, SNRE takes a fixed trained predictor $f$ and training samples as inputs, and learns an input region $A _ { \boldsymbol { \theta } , \mathbf { m } }$ and an output region $B _ { \phi }$ by optimizing a differentiable relaxation of region-level PNS. SNRE parameterizes the input and output regions with explicit learnable region functions, and introduces a soft feature mask m to select the input dimensions involved in the sufficiency-necessity relation. It then estimates region-level PNS through samplebased marginalization over the dimensions not emphasized by the mask and two parallel weighted aggregation modules: the A-weighted module estimates the in-region-to- $B _ { \phi }$ term, while the A<sup>¯</sup>-weighted module estimates the out-of-region-to-$\bar { B } _ { \phi }$ term. The product of these two aggregation scores yields a differentiable objective for jointly optimizing the region parameters and the feature mask.

## A. Region Parameterization and Feature Mask

Given a trained predictor $f : { \mathrm { V a l } } _ { \mathbf { X } }  { \mathrm { V a l } } _ { \mathbf { Y } }$ and input data, our goal is to estimate an input region $A \subset { \mathcal { D } }$ and an output region $B \subset \operatorname { V a l } _ { \mathbf { Y } }$ that maximize the region-level $\mathrm { P N S } ( A , B )$ , where D denotes the empirical input domain instantiated by the available dataset. Since direct optimization over all measurable sets is intractable, we parameterize the regions A and B via differentiable functions.

1) Input region parameterization: We parameterize the boundary of the input region A by a quadratic hypersurface. Let $g _ { \theta } : \mathrm { V a l } _ { \mathbf { X } }  \mathbb { R }$ be defined as

$$
g _ { \theta } ( \mathbf { x } ) = \mathbf { x } ^ { \top } \mathbf { w } \mathbf { x } + \mathbf { v } ^ { \top } \mathbf { x } + b ,
$$

where $\boldsymbol { \theta } = ( \mathbf { w } , \mathbf { v } , b )$ are learnable parameters, $\mathbf { w } \in \mathbb { R } ^ { d \times d }$ is a symmetric matrix, $\mathbf { v } \in \mathbb { R } ^ { d } , b \in \mathbb { R }$ , and $d = \dim ( \operatorname { V a l } _ { \mathbf { X } } )$ . The corresponding input region is

$$
A _ { \theta } = \{ \mathbf { x } \in \mathcal { D } \mid g _ { \theta } ( \mathbf { x } ) \leq 0 \} .\tag{6}
$$

The boundary $g _ { \theta } ( \mathbf { x } ) = 0$ forms a quadratic hypersurface, which offers a favorable balance between expressivity and interpretability: it can represent a rich family of shapes, such as ellipsoids, hyperboloids, and paraboloids, while its algebraic form permits analytic reasoning about geometric properties such as centers, principal axes, and curvature. We empirically validate the effectiveness of this input-region parameterization in Section VI-C.

2) Output region parameterization: In general, the output region B can be parameterized by any explicit and interpretable region family that matches the structure of the output space. For higher-dimensional outputs, B may also be represented by a quadratic hypersurface, analogously to the input region. In this work, we focus on classification settings, where the model output is typically low-dimensional, such as a class probability or a vector of class scores. The target behavior is therefore naturally associated with a compact confidence region, so a lower-complexity parameterization is sufficient.

When the output region needs to be learned, we use the following hyperspherical form:

$$
h _ { \phi } ( \mathbf { y } ) = \left( \mathbf { y } - { \pmb \mu } \right) ^ { \top } ( \mathbf { y } - { \pmb \mu } ) - r ^ { 2 } ,\tag{7}
$$

where $\mathbf { \boldsymbol { \phi } } = \mathbf { \boldsymbol { \mu } } ( \mu , r )$ are learnable parameters, and $r \ > \ 0$ is enforced by $r ~ = ~ \mathrm { s o f t p l u s } ( \rho )$ for a free parameter $\rho .$ The induced output region is

$$
B _ { \phi } = \{ \mathbf { y } \in \operatorname { V a l } _ { \mathbf { Y } } \mid h _ { \phi } ( \mathbf { y } ) \leq 0 \} .\tag{8}
$$

Note that the target output event can also be specified in advance, in which case B can be fixed or only partially learned.

3) Feature selection via a learnable mask: The quadratic form of $g _ { \boldsymbol { \theta } }$ naturally admits a feature-selection mechanism. The linear coefficients v in $g _ { \theta }$ capture the main effect of each input dimension, while the quadratic term w encodes pairwise interactions. To identify the subset of input features that are most relevant for the sufficiency-necessity relation, we introduce a continuous, learnable mask $\textbf { m } \in \mathbf { \Theta } ( 0 , 1 ) ^ { d }$ Specifically, we parameterize m via a sigmoid function σ over free parameters m˜ $\in \mathbb { R } ^ { d } , \mathrm { i . e . , } \mathbf { m } = \sigma ( \tilde { \mathbf { m } } )$ . The mask is then incorporated into the quadratic region definition by gating each input dimension:

$$
\begin{array} { r l } & { g _ { \boldsymbol { \theta } , \mathbf { m } } ( \mathbf { x } ) = \mathbf { x } ^ { \top } \tilde { \mathbf { w } } \mathbf { x } + \tilde { \mathbf { v } } ^ { \top } \mathbf { x } + b , } \\ & { \tilde { \mathbf { w } } = \mathbf { u } \mathbf { w } \mathbf { u } , \tilde { \mathbf { v } } = \mathbf { u } \mathbf { v } , \mathbf { u } = \mathrm { d i a g } ( \sqrt { m _ { 1 } } , \dots , \sqrt { m _ { d } } ) . } \end{array}\tag{9}
$$

The use of $\sqrt { m _ { j } }$ rather than $m _ { j }$ itself avoids overly suppressing the quadratic term and leads to more stable gradients. When $m _ { j } \ \approx \ 0 .$ , the j-th feature contributes neither to the linear nor to the quadratic part of $g _ { \boldsymbol { \theta } , \mathbf { m } }$ , effectively removing it from the region definition. During training m is kept continuous to allow gradient-based optimization; at inference we threshold m (e.g., with a cutoff at 0.5) to obtain a binary mask m˜ $\in \{ 0 , 1 \} ^ { d }$ , which defines the selected feature subset. Thus, the input region is now jointly parameterized by (θ, m), and we write

$$
A _ { { \boldsymbol \theta } , \mathbf { m } } = \{ \mathbf { x } \in { \mathcal { D } } \mid g _ { { \boldsymbol \theta } , \mathbf { m } } ( \mathbf { x } ) \leq 0 \} .\tag{10}
$$

Here, the mask m specifies the input dimensions used to define the regional event A. In the explicit region function in (9), this selection is implemented by gating the corresponding dimensions. As described later in (14)–(15), the remaining dimensions are treated as background variables and marginalized when evaluating the intervention effect through the fixed predictor.

## B. Differentiable PNS Objective and Optimization

Maximizing the exact $\mathrm { P N S } ( A _ { \theta , { \bf m } } , B _ { \phi } )$ is challenging because Definition 1 involves hard event judgments and conditional input distributions that are not available in closed form. We therefore construct a differentiable Monte Carlo estimator via a sigmoid relaxation.

Given a dataset $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { n } ,$ we introduce a soft membership score for the input event $\mathbf { X } \in A _ { \theta , \mathbf { m } } ;$

$$
s _ { \theta , \mathbf { m } } ^ { A } ( \mathbf { x } ) = \sigma \big ( - g _ { \theta , \mathbf { m } } ( \mathbf { x } ) / \tau \big ) ,\tag{11}
$$

and a corresponding soft membership score for the output event $\mathbf { Y } \in B _ { \phi } \colon$

$$
s _ { \phi } ^ { B } ( \mathbf { y } ) = \sigma \big ( - h _ { \phi } ( \mathbf { y } ) / \tau \big ) ,\tag{12}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function and $\tau > 0$ is a temperature parameter. These quantities provide smooth approximations to the indicator functions $\mathbb { I } ( \mathbf { x } \in A _ { \theta , \mathbf { m } } )$ and $\mathbb { I } ( \mathbf { y } \ \in$ $B _ { \phi } )$ , respectively. Accordingly, for the complementary events $\mathbf { X } { \in } \bar { A } _ { \theta , \mathbf { m } }$ and $\mathbf { Y } \in \bar { B } _ { \phi }$ , we use

$$
s _ { \theta , \mathbf { m } } ^ { \bar { A } } ( \mathbf { x } ) : = 1 - s _ { \theta , \mathbf { m } } ^ { A } ( \mathbf { x } ) , \quad s _ { \phi } ^ { \bar { B } } ( \mathbf { y } ) : = 1 - s _ { \phi } ^ { B } ( \mathbf { y } ) .\tag{13}
$$

Substituting the above relaxations into the Monte Carlo estimator of $\mathrm { P N S } ( A _ { \theta , { \bf m } } , B _ { \phi } )$ in (5), yields the following differentiable objective:

$$
\begin{array} { r l } & { \widehat { \mathrm { P N S } } ( A _ { \boldsymbol { \theta } , \mathbf { m } } , B _ { \boldsymbol { \phi } } ) = } \\ & { \frac { 1 } { N } \displaystyle \sum _ { k = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \omega _ { \boldsymbol { \theta } , \mathbf { m } } ^ { A } ( \mathbf { x } _ { j } ) s _ { \boldsymbol { \phi } } ^ { B } \big ( f ( \tilde { \mathbf { x } } _ { j k } ) \big ) \sum _ { j = 1 } ^ { N } \omega _ { \boldsymbol { \theta } , \mathbf { m } } ^ { \bar { A } } ( \mathbf { x } _ { j } ) s _ { \boldsymbol { \phi } } ^ { \bar { B } } \big ( f ( \tilde { \mathbf { x } } _ { j k } ) \big ) , } \end{array}\tag{14}
$$

where $\tilde { \mathbf { x } } _ { j k }$ denotes the mixed intervention input constructed from two training samples $\mathbf { x } _ { j }$ and $\mathbf { x } _ { k }$ under the soft mask m:

$$
\tilde { \mathbf { x } } _ { j k } = \mathbf { m } \odot \mathbf { x } _ { j } + \left( 1 - \mathbf { m } \right) \odot \mathbf { x } _ { k } .\tag{15}
$$

Intuitively, dimensions emphasized by the mask are mainly taken from $\mathbf { x } _ { j }$ to enforce the input event, while dimensions not emphasized by the mask are supplied by $\mathbf { x } _ { k }$ and averaged over the dataset for sample-based marginalization. The weights $\omega _ { \phi , \mathbf { m } } ^ { A } ( \cdot )$ and $\omega _ { \theta , \mathbf { m } } ^ { \bar { A } } ( \cdot )$ are normalized coefficients used to approximate the conditional expectations under the stochastic intervention policies $\pi ( \mathbf { X } ; P ^ { \hat { A } } )$ and $\pi ( { \bf { X } } ; P ^ { \bar { A } } )$ by empirical averages over the dataset. They are defined as

$$
\begin{array} { r } { \omega _ { \boldsymbol { \theta } , \mathbf { m } } ^ { A } ( \mathbf { x } _ { j } ) = \frac { s _ { \boldsymbol { \theta } , \mathbf { m } } ^ { A } ( \mathbf { x } _ { j } ) } { \sum _ { \ell = 1 } ^ { N } s _ { \boldsymbol { \theta } , \mathbf { m } } ^ { A } ( \mathbf { x } _ { \ell } ) } , } \\ { \omega _ { \boldsymbol { \theta } , \mathbf { m } } ^ { \bar { A } } ( \mathbf { x } _ { j } ) = \frac { s _ { \boldsymbol { \theta } , \mathbf { m } } ^ { \bar { A } } ( \mathbf { x } _ { j } ) } { \sum _ { \ell = 1 } ^ { N } s _ { \boldsymbol { \theta } , \mathbf { m } } ^ { \bar { A } } ( \mathbf { x } _ { \ell } ) } . } \end{array}\tag{16}
$$

A detailed derivation of (14) is provided in Appendix B.

## C. Extracting Sufficient-and-Necessary Regional Explanations

After optimization converges, we binarize the learned mask to obtain $\mathbf { \bar { m } ^ { * } } \in \{ 0 , 1 \} ^ { d }$ by thresholding: $m _ { j } ^ { * } = 1 { \mathrm { ~ i f ~ } } m _ { j } > 0 . 5$ and $m _ { i } ^ { * } = 0$ otherwise. This yields the selected feature subset $\mathcal { T } ^ { * } = \stackrel { \triangledown } { \{ j : m _ { j } ^ { * } = 1 \} }$ . The resulting input and output regions are

$$
A ^ { * } : = A _ { \theta ^ { * } , \mathbf { m } ^ { * } } = \{ \mathbf { x } \in \mathcal { D } | g _ { \theta ^ { * } , \mathbf { m } ^ { * } } ( \mathbf { x } ) \leq 0 \} ,
$$

$$
B ^ { \ast } : = B _ { \phi ^ { \ast } } = \{ \mathbf { y } \in \operatorname { V a l } _ { \mathbf { Y } } | h _ { \phi ^ { \ast } } ( \mathbf { y } ) \leq 0 \} .\tag{17}
$$

TABLE I  
SUMMARY OF DATASETS USED IN THE EXPERIMENTS. ALL TASKS ARE FORMULATED AS BINARY CLASSIFICATION.
<table><tr><td>Dataset</td><td>Task</td><td>Usage in Experiments</td></tr><tr><td>German Credit [39]</td><td>Predict good/bad credit risk</td><td>Region quality, active learning</td></tr><tr><td>GMSC [40]</td><td>Predict serious delinquency within two years</td><td>Region quality, robustness</td></tr><tr><td>Mushroom [41]</td><td>Predict edible versus poisonous mushrooms</td><td>Region quality, mask robustness</td></tr><tr><td>Phishing Websites [42]</td><td>Predict phishing versus legitimate websites</td><td>Region quality, geometry ablation, efficiency</td></tr><tr><td>Synthetic Circles</td><td>Two-dimensional concentric-circle classification</td><td>Toy visualization of SNRE optimization</td></tr><tr><td>Synthetic Blobs</td><td>Two-dimensional synthetic classification</td><td>Fine-tuning-induced explanation drift</td></tr></table>

The pair $( A ^ { * } , B ^ { * } )$ constitutes a sufficient-and-necessary regional explanation for the model $f$ with respect to the selected features $\mathcal { T } ^ { * }$ . It summarizes a regional sufficiency-necessity relation: inputs placed within $A ^ { * }$ are mapped to $B ^ { * }$ with high probability, while inputs placed outside $A ^ { * }$ are mapped outside $B ^ { * }$ with high probability.

## VI. EXPERIMENTS

Building on the proposed region-level PNS formulation and the SNRE optimization framework, we evaluate SNRE on four real-world tabular datasets and two synthetic datasets with a summary given in Table I. The experiments are organized around five questions. 1) Can a toy example provide intuition about the joint optimization of the input region A and output region $B$ under the proposed region-level PNS objective? 2) On real datasets, do the learned region pairs exhibit strong sufficiency and necessity on unseen test samples, and outperform existing PNS-based and rule-based explanation methods? 3) Are the proposed design choices structurally rational, particularly the quadratic input-region parameterization and the learnable feature mask? 4) How robust and efficient is SNRE? 5) How practical are the region-pair explanations generated by SNRE?

For all experiments, we use the same post-hoc explanation setup. We first train a predictor $f$ on the training split and keep it fixed during explanation learning. SNRE and all baselines are trained on the same train/validation/test split. SNRE learns a sufficient-and-necessary region pair $( A , B )$ with respect to the prediction behavior of $f ,$ where the input region A and output region $B$ are parameterized separately and jointly optimized by maximizing the proposed sufficiency-necessity objective until convergence. Each experiment is repeated five times, and we report the mean result.

We compare SNRE with 17 baselines, including 11 rulebased explanation methods and 6 PNS-based explanation methods. For rule-based baselines, we train each method using the model output as the target rather than the ground-truth label. For PNS-based methods, since they typically return feature scores or feature subsets instead of continuous inputoutput region pairs, we convert their selected features into an input event A using the same box post-processing protocol. Specifically, for a selected feature subset S, we define $A ^ { S } = \{ \mathbf { x } : l _ { j } \leq x _ { j } \leq u _ { j } , \forall j \in { \mathcal { S } } \}$ . After comparing several box construction strategies, we use a mean-std box, where the bounds are estimated from training samples satisfying the fixed output event B: $l _ { j } = \mu _ { j } - \rho \sigma _ { j } , u _ { j } = \mu _ { j } + \rho \sigma _ { j }$ , where $\mu _ { j }$ and $\sigma _ { j }$ are the empirical mean and standard deviation of feature $j ,$ and $\rho$ is selected on the validation split.

![](images/fd4c1a1f96a21ec8026108ac1f9565c5f15702eb373195f74569c8a8cb4f4ea7.jpg)

![](images/205e6f742270843227f1194e7de45054928d4505bb27b0139fb3cbb9aa35909c.jpg)  
Fig. 5. Toy example for joint learning of an input-output region pair $( A , B )$ with SNRE. Left: snapshots of the learned input region $A ,$ represented by the quadratic boundary $g ( \mathbf { x } ) = 0 ,$ across optimization iterations. Colors indicate snapshot iterations, with dashed contours for intermediate snapshots and a solid contour for the final one. Right: optimization trajectories of the regionlevel $\mathrm { P N S } ( A , B )$ and the output region parameter c defining $B = [ c , 1 ]$ over optimization iterations.

For more details on the baselines, metrics, model configurations and hyperparameters, please refer to Appendix C.

## A. Toy Example: Visualizing Joint Learning of Input-Output Region Pairs

We construct a simple binary-classification toy example to make the optimization behavior of SNRE visually tangible, focusing on learning a sufficient-and-necessary region pair $( A , B )$ associated with positive predictions. The data consist of two concentric circles in a 2D continuous input space. A neural network classifier is trained to separate the inner disk (positive class, shown in green) from the outer ring (negative class, shown in blue). SNRE is then run for 1,000 optimization steps to jointly learn an input region A, represented by a quadratic boundary, together with an output region B. Because the output is one-dimensional, the parameterization in (7) reduces to an interval on the real line, and since this experiment targets positive outputs, we use the restricted form $B = [ c , 1 ]$ , where c is a learnable lower endpoint.

In this toy setting, the sufficient-and-necessary region pair associated with the positive class is expected to align A with the inner disk and to place B near the upper end of the model’s output distribution. Fig. 5 (left) visualizes training snapshots of the learned input-region boundary. At early iterations, the boundary (light dashed contours) is poorly aligned with the data geometry and often traverses the transition band between the two classes. As optimization proceeds, the boundary progressively tightens around the inner disk, eventually (dark red and solid contour) forming a close fit that cleanly separates the high-confidence positive samples from the negative ones. The coupled dynamics on the output side are shown in Fig. 5 (right). The lower bound c of B (orange solid line) rises steadily from about 0.5 to a value near 1.0, shifting B toward the high-confidence regime. This shift is essential: to maintain high sufficiency, A must avoid the ambiguous region where predictions are uncertain, and forcing B to high probabilities naturally drives A to contract onto the core of the positive class. Throughout this joint optimization of (A, B), the PNS(A, B) objective (purple line) increases steadily and approaches 1, indicating that the learned region pair becomes progressively more sufficient and necessary.

TABLE II  
COMPARISON OF PNS, PN, AND PS ON FOUR DATASETS. RESULTS ARE REPORTED AS MEAN±STD OVER FIVE FOLDS AND ROUNDED TO THREE DECIMALS. THE BEST AND SECOND-BEST RESULTS IN EACH COLUMN ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td colspan="3">Mushroom</td><td colspan="3">GMSC</td><td colspan="3">Phishing Websites</td><td colspan="3">German Credit</td></tr><tr><td>PNS</td><td>PN</td><td>PS</td><td>PNS</td><td>PN</td><td>PS</td><td>PNS</td><td>PN</td><td>PS</td><td>PNS</td><td>PN</td><td>PS</td></tr><tr><td>IDS</td><td>0.255±0.000</td><td>0.527±0.000</td><td>0.483±0.000</td><td>0.559±0.000</td><td>0.993±0.000</td><td>0.563±0.000</td><td>0.328±0.000</td><td>0.591±0.000</td><td>0.555±0.000</td><td>0.464±0.000</td><td>0.801±0.000</td><td>0.580±0.000</td></tr><tr><td>SBRL</td><td>0.254±0.000</td><td>0.526±0.000</td><td>0.483±0.000</td><td>0.573±0.000</td><td>0.996±0.000</td><td>0.575±0.000</td><td>0.442±0.000</td><td>0.759±0.000</td><td>0.582±0.000</td><td>0.447±0.000</td><td>0.718±0.000</td><td>0.622±0.000</td></tr><tr><td>FRL</td><td>0.256±0.000</td><td>0.529±0.004</td><td>0.483±0.000</td><td>0.573±0.000</td><td>0.996±0.000</td><td>0.575±0.000</td><td>0.328±0.000</td><td>0.591±0.000</td><td>0.555±0.000</td><td>0.383±0.000</td><td>0.758±0.000</td><td>0.506±0.000</td></tr><tr><td>DT</td><td>0.293±0.008</td><td>0.597±0.016</td><td>0.490±0.000</td><td>0.826±0.000</td><td>0.996±0.000</td><td>0.829±0.000</td><td>0.442±0.000</td><td>0.759±0.000</td><td>0.582±0.000</td><td>0.418±0.000</td><td>0.815±0.000</td><td>0.513±0.000</td></tr><tr><td>FLDT</td><td>0.547±0.156</td><td>0.813±0.125</td><td>0.662±0.096</td><td>0.881±0.006</td><td>0.998±0.000</td><td>0.883±0.006</td><td>0.725±0.029</td><td>0.897±0.034</td><td>0.809±0.041</td><td>0.720±0.034</td><td>0.844±0.038</td><td>0.853±0.039</td></tr><tr><td>RRL</td><td>0.966±0.030</td><td>0.992±0.006</td><td>0.974±0.025</td><td>0.677±0.057</td><td>0.994±0.001</td><td>0.681±0.058</td><td>0.893±0.017</td><td>0.949±0.016</td><td>0.941±0.007</td><td>0.665±0.022</td><td>0.822±0.023</td><td>0.810±0.033</td></tr><tr><td>CORELS</td><td>0.268±0.003</td><td>0.553±0.004</td><td>0.486±0.000</td><td>0.559±0.000</td><td>0.993±0.000</td><td>0.563±0.000</td><td>0.442±0.000</td><td>0.759±0.000</td><td>0.582±0.000</td><td>0.296±0.000</td><td>0.549±0.000</td><td>0.540±0.000</td></tr><tr><td>SamRuLe</td><td>0.268±0.000</td><td>0.553±0.000</td><td>0.486±0.000</td><td>0.559±0.000</td><td>0.993±0.000</td><td>0.563±0.000</td><td>0.442±0.000</td><td>0.759±0.000</td><td>0.582±0.000</td><td>0.441±0.006</td><td>0.724±0.008</td><td>0.610±0.006</td></tr><tr><td>MDL-RL</td><td>0.274±0.004</td><td>0.562±0.005</td><td>0.488±0.000</td><td>0.573±0.000</td><td>0.996±0.000</td><td>0.575±0.000</td><td>0.487±0.000</td><td>0.792±0.000</td><td>0.614±0.000</td><td>0.415±0.000</td><td>0.718±0.000</td><td>0.578±0.000</td></tr><tr><td>DRNet</td><td>0.509±0.230</td><td>0.761±0.158</td><td>0.644±0.158</td><td>0.598±0.348</td><td>0.978±0.002</td><td>0.611±0.356</td><td>0.536±0.032</td><td>0.540±0.035</td><td>0.993±0.006</td><td>0.582±0.002</td><td>0.590±0.002</td><td>0.985±0.001</td></tr><tr><td>IMLI</td><td>0.264±0.000</td><td>0.545±0.001</td><td>0.484±0.000</td><td>0.541±0.000</td><td>0.979±0.000</td><td>0.553±0.000</td><td>0.645±0.000</td><td>0.899±0.000</td><td>0.718±0.001</td><td>0.527±0.009</td><td>0.869±0.012</td><td>0.606±0.009</td></tr><tr><td>FANS</td><td>0.380±0.132</td><td>0.648±0.092</td><td>0.573±0.116</td><td>0.538±0.067</td><td>0.998±0.001</td><td>0.539±0.068</td><td>0.679±0.062</td><td>0.887±0.035</td><td>0.766±0.051</td><td>0.546±0.017</td><td>0.692±0.065</td><td>0.797±0.101</td></tr><tr><td>LEWIS</td><td>0.503±0.000</td><td>0.674±0.000</td><td>0.746±0.000</td><td>0.347±0.000</td><td>0.999±0.000</td><td>0.347±0.000</td><td>0.675±0.000</td><td>0.874±0.000</td><td>0.773±0.000</td><td>0.458±0.000</td><td>0.557±0.000</td><td>0.823±0.000</td></tr><tr><td>LENS</td><td>0.250±0.000</td><td>0.482±0.000</td><td>0.519±0.000</td><td>0.454±0.000</td><td>0.998±0.000</td><td>0.455±0.000</td><td>0.467±0.087</td><td>0.784±0.082</td><td>0.591±0.059</td><td>0.507±0.042</td><td>0.710±0.064</td><td>0.722±0.121</td></tr><tr><td>NSEG</td><td>0.268±0.010</td><td>0.488±0.002</td><td>0.550±0.021</td><td>0.316±0.083</td><td>0.998±0.000</td><td>0.317±0.084</td><td>0.597±0.001</td><td>0.854±0.002</td><td>0.699±0.002</td><td>0.467±0.004</td><td>0.724±0.023</td><td>0.645±0.022</td></tr><tr><td>CF2</td><td>0.261±0.000 0.836±0.031</td><td>0.487±0.000 0.907±0.011</td><td>0.536±0.000</td><td>0.247±0.110</td><td>0.998±0.000</td><td>0.247±0.110 0.455±0.023</td><td>0.730±0.005</td><td>0.882±0.000</td><td>0.827±0.006</td><td>0.470±0.013</td><td>0.753±0.008</td><td>0.624±0.011</td></tr><tr><td>NSE4T</td><td></td><td></td><td>0.921±0.023</td><td>0.454±0.030</td><td>0.998±0.011</td><td></td><td>0.718±0.034</td><td>0.884±0.022</td><td>0.812±0.020</td><td>0.481±0.040</td><td>0.752±0.012</td><td>0.639±0.045</td></tr><tr><td>SNRE</td><td>0.988±0.004</td><td>0.999±0.001</td><td>0.990±0.004</td><td>0.982±0.002</td><td>0.982±0.002</td><td>1.000±0.000</td><td>0.950±0.006</td><td>0.984±0.002</td><td>0.966±0.005</td><td>0.750±0.020</td><td>0.861±0.059</td><td>0.882±0.063</td></tr></table>

## B. Quantitative Evaluation of Learned Sufficient-andnecessary Region Pairs

We evaluate whether the learned explanations preserve strong sufficiency and necessity on unseen test data. For each method, we derive an input region A and measure its relation to a fixed, high-confidence output interval B (B = [0.8, 1] for Mushroom, German Credit, and Phishing Websites, and B = [0.5, 1] for GMSC). For SNRE, (A, B)is its learned region pair. For rule-based baselines (upper block of Table II), A is the rule antecedent. For PNS-based explanation methods (middle block of Table II), which usually output feature subsets or feature scores rather than regions, we post-process their results into a comparable input region A. Specifically, we use a mean-std bounding box, which was found to be the most effective post-processing strategy after extensive testing.

As shown in the table II, SNRE achieves the highest PNS on all four datasets, indicating the best overall generalization performance of the learned explanations on the test set. On Mushroom, GMSC, and Phishing Websites, SNRE not only leads in PNS, but also attains the best or near-best PS and PN. On German Credit, although SNRE does not achieve the highest value in either PS or PN individually, the higher scores of other methods are more one-sided. For example, IMLI yields a higher PN but a lower PS, while DRNet attains a high PS but suffers from a lower PN. In contrast, SNRE achieves a better balance between the two, and therefore still obtains the highest PNS overall. Therefore, the advantage of SNRE does not lie in maximizing a single metric in isolation, but in learning a more generalizable and better coordinated sufficient-and-necessary explanation.

## C. Structural Rationality Analysis of Region Parameterization

1) Input region parameterization: To validate the structural rationality of quadratic hypersurfaces, we compare five inputregion families: linear, box, sphere, quadratic, and MLPbased implicit boundaries. As shown in Fig. 6, the quadratic parameterization achieves the best or near-best PNS on both datasets. On Phishing Websites, it delivers the highest PS, PN, and PNS simultaneously, indicating that its geometric flexibility is sufficient to capture the target sufficient-andnecessary region. On German Credit, although it is not uniformly best on each individual term, it provides the most balanced trade-off between PS and PN, and thus remains optimal overall in PNS. In contrast, linear, box, and spherical regions are geometrically constrained and therefore struggle to jointly preserve coverage and exclusivity, leading to a clearer sufficiency-necessity mismatch.

Fig. 7 further explains this effect geometrically. On the rotated elliptical structure, linear boundaries are too rigid to align with the true region, while box and sphere parameterizations cannot capture rotation and anisotropy. By contrast, the quadratic boundary naturally matches the underlying elliptical geometry, which enables high sufficiency and necessity simultaneously. Although the MLP parameterization achieves a comparable fit, its boundary is implicit and lacks the explicit geometric semantics of a quadratic form. These results show that quadratic hypersurfaces provide a more favorable structural trade-off between expressive power and interpretability for sufficient-and-necessary regional explanations.

![](images/de21f6b636c6ce798504e137de91eb6422fad3af15a29d26eccc22e6f2835ea1.jpg)

![](images/887108bb183e74bdf6ec171e3708285319a2ae737effee49023550f595a6ecca.jpg)  
(a)

![](images/eaeaa36cf87259c44f94ad362519c9c5dd5137be92a3b65e603c45fe8724667a.jpg)  
(b)

![](images/0ff07ec0af3fee2868f6e4ef1d615cbe3360d64430f85373c8781e153adce74d.jpg)  
(c)

Fig. 6. Ablation on input-region parameterization. We compare five region families, including linear, box-shaped, spherical, quadratic, and MLP-based implicit boundaries, on Phishing Websites and German Credit. The results are reported in terms of PN, PS and PNS. Quadratic parameterization achieves a favorable balance between expressive power and explanation quality across datasets.  
![](images/affa3e77d7567b93c5721fc3654de5b1a144d97f3e5d7504c7f5c1eda2920105.jpg)

![](images/bc5de758b6d5ffc8e3cebd11ebb786371882c20a5ee3319cb1299e9972452390.jpg)

![](images/161423f84b592e93e54b6ef2178c427651a9e789654311b1ccdd5b4113dd4fb8.jpg)

![](images/d1d33f1fbef2e0375a79e1813507bf2e0f3779d3d1031a3b5fa453637da65562.jpg)

![](images/dc46fd0d1ee341453e6e1f9a2bf040ba02997379c0d1212dbc8fae7e5ce01052.jpg)  
Fig. 7. Comparison of input region shapes from different parameterizations. A 2D example shows that linear, box, and spherical shapes are too rigid to fit the data, while an MLP learns a complex, irregular boundary. The quadratic hypersurface forms a regular ellipse, best balancing geometric expressiveness and interpretability.  
Fig. 8. PNS under different feature-mask strategies with increasing noisefeature ratios.

2) Feature Mask: We next investigates the rationality of learning a feature mask within the SNRE optimization process. Specifically, we design a comparative experiment with three settings: Learned mask (the proposed SNRE method that jointly learns feature importance), No-mask (standard training without masking), and Random-k (randomly selecting kfeatures as a baseline). We evaluate the stability of the PNS metric under varying ratios of noise features on two datasets.

As shown in Fig. 8, the Learned mask strategy consistently maintains high PNS scores under both low and high noise ratios in both datasets, demonstrating strong robustness. In contrast, the No-mask and Random-k strategies exhibit significant fluctuations and sharp declines in PNS as the noise ratio increases. This indicates that learning a mask allows the model to ignore irrelevant noisy features during optimization, preventing the degradation of the decision boundary for the core mechanism, while the other two methods show much poorer PNS stability as the noise ratio increases.

![](images/7348c74812c6d26ff711a81be3ac4e769e672da64059e51681eb8abe615c5f5a.jpg)  
Fig. 9. Robustness analysis of SNRE explanations under increasing training input noise across four datasets, measured by Input IoU, Output IoU, and Mask Hamming.

## D. Robustness analysis of SNRE explanations

This section investigates the robustness of the explanations produced by SNRE under input perturbations. Specifically, we fix the same black-box model and data split, take the explainer trained on the clean training set as the reference explainer, and then retrain the explainer after injecting Gaussian noise with increasing magnitude into the training inputs only. The resulting explanations are compared against the reference one using three consistency measures: Input IoU, Output IoU, and Mask Hamming. Let $\mathbf { a } ^ { \mathrm { r e f } }$ and $\mathbf { a } ^ { \mathrm { c u r } }$ denote the binary membership vectors of the input region produced by the reference and perturbed explainers, respectively, and let $\mathbf { b } ^ { \mathrm { r e f } }$ and $\mathbf { b } ^ { \mathrm { c u r } }$ denote the corresponding binary membership vectors of the output region. Both input-region and output-region consistency are measured by the intersection-over-union:

$$
\operatorname { I o U } ( \mathbf { u } , \mathbf { v } ) = { \frac { | \mathbf { u } \cap \mathbf { v } | } { | \mathbf { u } \cup \mathbf { v } | } } .
$$

Accordingly, Input IoU is computed from $( \mathbf { a } ^ { \mathrm { { r e f } } } , \mathbf { a } ^ { \mathrm { { c u r } } } )$ , and Output IoU is computed from $( { \bf b } ^ { \mathrm { r e f } } , { \bf b } ^ { \mathrm { c u r } } )$ . In addition, let $\mathbf { m } ^ { \mathrm { r e f } } , \mathbf { m } ^ { \mathrm { c u r } } \in \{ 0 , 1 \} ^ { \bar { d } }$ denote the hard feature masks of the two explainers. Their discrepancy is measured by the normalized Hamming distance:

$$
\mathrm { H a m m i n g } ( { \mathbf { m } ^ { \mathrm { r e f } } } , { \mathbf { m } ^ { \mathrm { c u r } } } ) = \frac { 1 } { d } \sum _ { j = 1 } ^ { d } { \bf 1 } \big ( m _ { j } ^ { \mathrm { r e f } } \ne m _ { j } ^ { \mathrm { c u r } } \big ) .
$$

Hence, higher Input/Output IoU and lower Mask Hamming indicate better explanation stability under noise.

As shown in Fig. 9, SNRE exhibits strong stability across all four datasets. On Mushroom, both Input IoU and Output IoU remain almost at 1.0 across different noise levels, while Mask Hamming stays close to 0, indicating nearly unchanged explanations under perturbation. On German Credit, Phishing Websites, and GMSC, although moderate fluctuations can be observed, both IoU scores remain consistently high overall and the mask discrepancy stays small. These results suggest that the input region, output region, and selected explanatory features learned by SNRE do not drift substantially under noisy training inputs, demonstrating good robustness and consistency of the resulting explanations.

## E. Model Size and Efficiency

TABLE III  
MODEL SIZE AND EFFICIENCY ANALYSIS ON THE PHISHINGWEBSITES DATASET.
<table><tr><td>Method</td><td># Params</td><td>Training (min)</td><td>Inference (ms)</td><td>PNS</td></tr><tr><td>IDS</td><td></td><td>0.4169</td><td>0.0023</td><td>0.326</td></tr><tr><td>SBRL</td><td></td><td>0.5078</td><td>0.0055</td><td>0.444</td></tr><tr><td>FRL</td><td></td><td>0.4121</td><td>0.0045</td><td>0.326</td></tr><tr><td>DT</td><td></td><td>2.6360</td><td>0.0242</td><td>0.444</td></tr><tr><td>FLDT</td><td></td><td>55.3380</td><td>0.0252</td><td>0.76</td></tr><tr><td>RRL</td><td>87,509</td><td>19.2264</td><td>0.3256</td><td>0.895</td></tr><tr><td>CORELS</td><td></td><td>0.4196</td><td>0.0052</td><td>0.444</td></tr><tr><td>SamRule</td><td>一</td><td>2.3039</td><td>0.0046</td><td>0.444</td></tr><tr><td>MDL-RL</td><td></td><td>0.5028</td><td>0.0147</td><td>0.490</td></tr><tr><td>DRNet</td><td>6,240</td><td>3.9330</td><td>0.1180</td><td>0.</td></tr><tr><td>IMLI</td><td></td><td>6.9016</td><td>0.0029</td><td>0.352</td></tr><tr><td>SNRE</td><td>9,452</td><td>11.5569</td><td>0.0582</td><td>0.948</td></tr></table>

This subsection evaluate the model size and efficiency of different methods, in order to investigate whether the performance advantage of SNRE is obtained at the cost of substantially higher complexity. Specifically, on the Phishing-Websites dataset, we compare trainable parameters, training time, and single-sample inference time across methods, while also reporting the corresponding PNS, so as to jointly evaluate the trade-off between explanation quality and efficiency. Here, the training time reflects the overall cost of obtaining an explanation, the inference time reflects the efficiency of applying a learned explanation to an individual sample, and the model size characterizes the complexity of the method.

![](images/402c27cbdaef5d2d32a5e2feabf2ac17ee954968ca6d3b21e24e43e132a0888a.jpg)  
Fig. 10. Visualization of the retention of the dominant sufficiency–necessity explanation structure under fine-tuning on a synthetic dataset.

As shown in Table III, SNRE achieves the best PNS of 0.948 while maintaining a favorable efficiency–performance trade-off. Compared with high-performing baselines, SNRE uses only 9,452 parameters, which is substantially smaller than RRL with 87,509 parameters, and although slightly larger than DRNet with 6,240 parameters, it delivers a much higher PNS. In terms of inference efficiency, SNRE requires only 0.0582 ms per sample, making it faster than both RRL and DRNet. For training cost, SNRE is slower than lightweight rule-based methods such as IDS, FRL, and CORELS, but remains much faster than FLDT, while these faster methods generally lag far behind in PNS. Overall, SNRE is neither the smallest nor the fastest method in absolute terms, but it achieves the strongest sufficient-and-necessary explanation quality under a still moderate complexity budget, demonstrating a better overall cost-effectiveness.

F. Case Study: Diagnosing Fine-Tuning-Induced Regional Prediction Drift

This case study investigates whether SNRE can quantify how the learned sufficient-and-necessary prediction regions change after fine-tuning. We define the retention score as

$$
\mathrm { R e t e n t i o n } = \frac { 1 } { 3 } \left( \mathrm { I o U } _ { A } + \mathrm { I o U } _ { \mathrm { m a s k } } + \mathrm { I o U } _ { B } \right) ,\tag{18}
$$

where the three terms measure the overlap of the input region, feature mask and output region, before and after fine-tuning. A larger value indicates stronger preservation of the learned sufficient-and-necessary regional pattern.

![](images/fa2633c9fccc8da70215dd728716599ed1676544abbf388a016423108c8a77da.jpg)

![](images/509a74865c43412466ad9e1013976feacb146647496f8741ca615b058919fd88.jpg)  
Fig. 11. Comparison of test AUC and accuracy on the German Credit dataset across active learning rounds for random sampling, uncertainty sampling, and the proposed SNRE-guided sampling.

![](images/a747c40cc693aa90626c6896c50618e1c7907d3209f09b5323e2b8777e7dbabb.jpg)  
Fig. 12. Comparison of feature distributions on the German Credit dataset between the unlabeled samples and the samples acquired by SNRE-guided sampling.

To validate this metric, we visualize the learned input-region boundaries on a two-dimensional synthetic dataset. In Fig. 10, the gray dashed curve denotes the base-model boundary, and the red solid curve denotes the fine-tuned one. The Conservative fine-tuning case achieves the highest retention score (0.733) and remains visually closest to the base boundary, with only a mild local adjustment. By contrast, Aggressive finetuning and Medium fine-tuning obtain lower retention scores of 0.586 and 0.573, respectively, and exhibit more evident geometric deviations. This shows that the proposed retention score aligns well with visual intuition and provides a practical measure of fine-tuning-induced regional drift.

## G. Case Study: Active Learning via SNRE-Guided Sampling

This case study investigates whether SNRE-guided sampling can select more informative unlabeled samples on the German Credit dataset [43], [44]. The intuition is that, among unlabeled samples, those whose predictions already fall in the target output region while their inputs remain outside but near the learned sufficient-and-necessary region may be particularly informative for improving the model. Such samples suggest that the current prediction mechanism, especially its sufficientand-necessary component captured by region $A ,$ may not yet fully account for a meaningful part of the input space. Labeling these samples may therefore help the model adjust this part of its prediction mechanism and improve subsequent predictive performance.

Based on this intuition, we heuristically assign each pool sample x the acquisition score

$$
s ( x ) = \sigma \left( \frac { p ( x ) - c _ { \mathrm { l o w } } } { \tau _ { f } } \right) \exp \left( - \frac { g _ { + } ( x ) } { \tau _ { A } } \right) ,\tag{19}
$$

where $p ( x )$ is the predicted positive probability, $c _ { \mathrm { l o w } }$ is the lower threshold of the target output region, and $g _ { + } ( x ) \ =$ $\operatorname* { m a x } ( g ( x ) , 0 )$ measures how far $x$ lies outside the learned sufficient-and-necessary region $A .$ Thus, larger scores correspond to samples whose predictions already fall in the target region and whose inputs lie just outside, but close to, the current sufficient-and-necessary component of the model.

Results show that SNRE-guided sampling consistently achieves higher and more stable test AUC and accuracy, with faster convergence (Fig. 11). In contrast, random and uncertainty sampling exhibit larger performance fluctuations. This suggests that adding high-confidence samples near but outside the learned SNRE input region is more beneficial for subsequent model improvement than adding arbitrary or merely uncertain instances.

To characterize the samples selected by SNRE-guided sampling, we compare their feature distributions with those of the overall unlabeled pool, as shown in Fig. 12. The most distinctive attributes identified by distribution shift indicate that SNRE-guided sampling tends to select applicants who are older, have longer loan durations, have lived longer at their current residence, and are more often single males. These samples are not scattered outliers but form a coherent subgroup with consistent characteristics. In other words, the model has begun to assign high confidence to this subgroup, yet the learned SNRE input region A does not stably cover it. Labeling such samples encourages the model to recalibrate its predictions on this subgroup, improving overall performance and expanding the explanatory region to capture the newly learned regularity.

## VII. CONCLUSION

In this paper, we proposed Sufficient and Necessary Regional Explanations (SNRE), a framework for explaining a trained predictor through input-output region pairs. By formulating a region-level version of the Probability of Necessity and Sufficiency, SNRE provides a principled way to characterize when and only when a prediction behavior arises. We introduced stochastic interventions to make regionlevel sufficiency and necessity well-defined and computable from finite data, and developed a differentiable objective for learning such regions end to end. To balance expressiveness and interpretability, SNRE parameterizes regions with quadratic hypersurfaces and incorporates a learnable feature mask. Experiments on real-world and synthetic datasets show that SNRE achieves strong sufficiency-necessity quality and robust explanation behavior. Case studies further illustrate the practicality of the learned region pairs.

## REFERENCES

[1] W. Zhao, Y. Rao, J. Zhou, and J. Lu, “Diml: Deep interpretable metric learning via structural matching,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 4, pp. 2518–2532, 2023.

[2] P. Chormai, J. Herrmann, K.-R. Muller, and G. Montavon, “Disen-¨ tangled explanations of neural network predictions by finding relevant subspaces,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 11, pp. 7283–7299, 2024.

[3] S. Kim and D.-K. Chae, “What does a model really look at?: Extracting model-oriented concepts for explaining deep neural networks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 7, pp. 4612–4624, 2024.

[4] C. Rudin, “Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead,” Nature machine intelligence, vol. 1, no. 5, pp. 206–215, 2019.

[5] R. Guidotti, A. Monreale, S. Ruggieri, F. Turini, F. Giannotti, and D. Pedreschi, “A survey of methods for explaining black box models,” ACM computing surveys (CSUR), vol. 51, no. 5, pp. 1–42, 2018.

[6] Y.-M. Shin, S.-W. Kim, and W.-Y. Shin, “Page: Prototype-based modellevel explanations for graph neural networks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 10, pp. 6559– 6576, 2024.

[7] Y. Liu and Y. Ge, “When feature attribution methods meet robust models: A survey and outlook,” IEEE Communications Magazine, 2026.

[8] M. A. Nazir, E. Evangelista, S. M. S. Bukhari, and R. Sharma, “A survey of feature attribution techniques in explainable ai: Taxonomy, analysis and comparison,” Annals ofMathematics and Computer Science, vol. 28, pp. 115–126, 2025.

[9] S. Verma, V. Boonsanong, M. Hoang, K. Hines, J. Dickerson, and C. Shah, “Counterfactual explanations and algorithmic recourses for machine learning: A review,” ACM Computing Surveys, vol. 56, no. 12, pp. 1–42, 2024.

[10] J. Van Der Waa, E. Nieuwburg, A. Cremers, and M. Neerincx, “Evaluating xai: A comparison of rule-based and example-based explanations,” Artificial intelligence, vol. 291, p. 103404, 2021.

[11] O. M. Mastromichalakis, E. Dervakos, A. Chortaras, and G. Stamou, “Rule-based explanations of machine learning classifiers using knowledge graphs,” in Proceedings of the AAAI Symposium Series, vol. 3, no. 1, 2024, pp. 193–202.

[12] V. F. Souza, F. Cicalese, E. Laber, and M. Molinaro, “Decision trees with short explainable rules,” Advances in neural information processing systems, vol. 35, pp. 12 365–12 379, 2022.

[13] J. Pearl, Causality: Models, Reasoning, and Inference. Cambridge University Press, 2009.

[14] X. Chen, R. Cai, Z. Huang, Y. Zhu, J. Horwood, Z. Hao, Z. Li, and J. M. Hernandez-Lobato, “Feature attribution with necessity and sufficiency´ via dual-stage perturbation test for causal explanation,” in Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 6486–6502.

[15] R. Cai, Y. Zhu, X. Chen, Y. Fang, M. Wu, J. Qiao, and Z. Hao, “On the probability of necessity and sufficiency of explaining graph neural networks: A lower bound optimization approach,” Neural Networks, vol. 184, p. 107065, 2025.

[16] W. W. Cohen, “Fast effective rule induction,” in Machine learning proceedings 1995. Elsevier, 1995, pp. 115–123.

[17] T. Wang, C. Rudin, F. Doshi-Velez, Y. Liu, E. Klampfl, and P. Mac-Neille, “A bayesian framework for learning rule sets for interpretable classification,” Journal of Machine Learning Research, vol. 18, no. 70, pp. 1–37, 2017.

[18] H. Lakkaraju, S. H. Bach, and J. Leskovec, “Interpretable decision sets: A joint framework for description and prediction,” in Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, 2016, pp. 1675–1684.

[19] H. Yang, C. Rudin, and M. Seltzer, “Scalable Bayesian rule lists,” in Proceedings of the 34th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, D. Precup and Y. W. Teh, Eds., vol. 70. PMLR, 06–11 Aug 2017, pp. 3921–3930. [Online]. Available: https://proceedings.mlr.press/v70/yang17h.html

[20] F. Wang and C. Rudin, “Falling Rule Lists,” in Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, G. Lebanon and S. V. N. Vishwanathan, Eds., vol. 38. San Diego, California, USA: PMLR, 09–12 May 2015, pp. 1013–1022. [Online]. Available: https://proceedings.mlr.press/v38/wang15a.html

[21] E. Angelino, N. Larus-Stone, D. Alabi, M. Seltzer, and C. Rudin, “Learning certifiably optimal rule lists for categorical data,” Journal of Machine Learning Research, vol. 18, no. 234, pp. 1–78, 2018.

[22] H. M. Proenc¸a and M. van Leeuwen, “Interpretable multiclass classification by mdl-based rule lists,” Information Sciences, vol. 512, pp. 1372–1393, 2020.

[23] L. Pellegrina and F. Vandin, “Scalable rule lists learning with sampling,” in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 2352–2363.

[24] W.-Y. Loh, “Classification and regression trees,” Wiley interdisciplinary reviews: data mining and knowledge discovery, vol. 1, no. 1, pp. 14–23, 2011.

[25] J. Good, T. Kovach, K. Miller, and A. Dubrawski, “Feature learning for interpretable, performant decision trees,” Advances in Neural Information Processing Systems, vol. 36, pp. 66 571–66 582, 2023.

[26] Z. Wang, W. Zhang, N. Liu, and J. Wang, “Scalable rule-based representation learning for interpretable classification,” Advances in Neural Information Processing Systems, vol. 34, pp. 30 479–30 491, 2021.

[27] L. Qiao, W. Wang, and B. Lin, “Learning accurate and interpretable decision rule sets from neural networks,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 5, 2021, pp. 4303–4311.

[28] B. Ghosh, D. Malioutov, and K. S. Meel, “Efficient learning of interpretable classification rules,” Journal of Artificial Intelligence Research, vol. 74, pp. 1823–1863, 2022.

[29] J. Pearl, “Probabilities of causation: three counterfactual interpretations and their identification,” in Probabilistic and causal inference: the works of Judea Pearl, 2022, pp. 317–372.

[30] J. Tian and J. Pearl, “Probabilities of causation: Bounds and identification,” Annals of Mathematics and Artificial Intelligence, vol. 28, no. 1, pp. 287–313, 2000.

[31] D. S. Watson, L. Gultchin, A. Taly, and L. Floridi, “Local explanations via necessity and sufficiency: Unifying theory and practice,” in Uncertainty in Artificial Intelligence. PMLR, 2021, pp. 1382–1392.

[32] E. Balkir, I. Nejadgholi, K. C. Fraser, and S. Kiritchenko, “Necessity and sufficiency for explaining text classifiers: A case study in hate speech detection,” in Proceedings of the 2022 conference of the North American chapter of the association for computational linguistics: human language technologies, 2022, pp. 2672–2686.

[33] J. Tan, S. Geng, Z. Fu, Y. Ge, S. Xu, Y. Li, and Y. Zhang, “Learning and evaluating graph neural network explanations based on counterfactual and factual reasoning,” in Proceedings ofthe ACM web conference 2022, 2022, pp. 1018–1027.

[34] S. Galhotra, R. Pradhan, and B. Salimi, “Explaining black-box algorithms using probabilistic contrastive counterfactuals,” in Proceedings of the 2021 international conference on management of data, 2021, pp. 577–590.

[35] A. Geiger, D. Ibeling, A. Zur, M. Chaudhary, S. Chauhan, J. Huang, A. Arora, Z. Wu, N. Goodman, C. Potts et al., “Causal abstraction: A theoretical foundation for mechanistic interpretability,” Journal of Machine Learning Research, vol. 26, no. 83, pp. 1–64, 2025.

[36] C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra, “Weight uncertainty in neural network,” in International conference on machine learning. PMLR, 2015, pp. 1613–1622.

[37] G. Montufar, R. Pascanu, K. Cho, and Y. Bengio, “On the number of´ linear regions of deep neural networks,” Advances in neural information processing systems, vol. 27, 2014.

[38] D. B. Rubin, “Estimating causal effects of treatments in randomized and nonrandomized studies.” Journal of educational Psychology, vol. 66, no. 5, p. 688, 1974.

[39] P. Hoffman and K. Bach, “Statlog (german credit data); uci machine learning repository: Irvine, ca, usa, 1994.”

[40] C. Fusion and W. Cukierski, “Give me some credit,” https://kaggle.com/ competitions/GiveMeSomeCredit, 2011, kaggle.

[41] “Mushroom,” UCI Machine Learning Repository, 1981, DOI: https://doi.org/10.24432/C5959T.

[42] R. M. Mohammad, F. Thabtah, and L. McCluskey, “Intelligent rulebased phishing websites classification,” IET Information Security, vol. 8, no. 3, pp. 153–160, 2014.

[43] B. Settles, “Active learning literature survey,” 2009.

[44] Y. Gal, R. Islam, and Z. Ghahramani, “Deep bayesian active learning with image data,” in International conference on machine learning. PMLR, 2017, pp. 1183–1192.

## APPENDIX A

## MONTE CARLO ESTIMATION OF CONTINUOUS PNS

In this appendix, we provide a justification for the Monte Carlo estimator in (5). Recall that, for an input region $A \subseteq { \mathcal { D } }$ and an output region $B \subseteq \operatorname { V a l } _ { \mathbf { Y } }$ , the definition of PNS over regions in continuous domains (Definition 1) is

$$
\operatorname { P N S } ( A , B ) = P { \Bigl ( } \mathbf { Y } _ { \pi ( \mathbf { X } ; P ^ { A } ) } \in B \ \land \ \mathbf { Y } _ { \pi ( \mathbf { X } ; P ^ { { \bar { A } } } ) } \in { \bar { B } } { \Bigr ) } ,\tag{20}
$$

where

$$
P ^ { A } ( \mathbf { X } ) = P ^ { * } ( \mathbf { X } \mid \mathbf { X } \in A ) , \quad P ^ { { \bar { A } } } ( \mathbf { X } ) = P ^ { * } ( \mathbf { X } \mid \mathbf { X } \in { \bar { A } } ) .\tag{21}
$$

Since the predictor $f$ is deterministic and the input variables X have no parents, each intervention fully determines the model output. Moreover, the two stochastic intervention policies $\pi ( \mathbf { X } ; P ^ { A } )$ and $\pi ( \mathbf { X } ; P ^ { \bar { A } } )$ are independently sampled. As a result, the joint probability factorizes as

$$
\operatorname { P N S } ( A , B ) = P { \bigl ( } \mathbf { Y } _ { \pi ( \mathbf { X } ; P ^ { A } ) } \in B { \bigr ) } P { \bigl ( } \mathbf { Y } _ { \pi ( \mathbf { X } ; P ^ { { \bar { A } } } ) } \in { \bar { B } } { \bigr ) } .\tag{22}
$$

Under a hard intervention $d o ( \mathbf { X } = \mathbf { x } )$ , the model output is given by

$$
{ \mathbf Y } _ { d o ( { \mathbf X } = { \mathbf x } ) } = f ( { \mathbf x } ) .\tag{23}
$$

Therefore, under the stochastic intervention $\pi ( \mathbf { X } ; P ^ { A } )$ , we have

$$
P \big ( \mathbf { Y } _ { \pi ( \mathbf { X } ; P ^ { A } ) } \in B \big ) = \operatorname { \mathbb { E } } _ { \mathbf { X } \sim P ^ { * } ( \cdot | \mathbf { X } \in A ) } \big [ \mathbb { I } \big ( f ( \mathbf { X } ) \in B \big ) \big ] ,\tag{24}
$$

and analogously,

$$
P \big ( \mathbf { Y } _ { \pi ( \mathbf { X } ; P ^ { \tilde { A } } ) } \in \bar { B } \big ) = \mathbb { E } _ { \mathbf { X } \sim P ^ { * } ( \cdot | \mathbf { X } \in \bar { A } ) } \big [ \mathbb { I } \big ( f ( \mathbf { X } ) \in \bar { B } \big ) \big ] .\tag{25}
$$

Let $\{ \mathbf { x } _ { i } ^ { \mathrm { i n } } \} _ { i = 1 } ^ { m } \stackrel { \mathrm { i . i . d . } } { \sim } P ^ { * } ( \mathbf { X } \mid \mathbf { X } \in \cal { A } )$ and $\{ \mathbf { x } _ { i } ^ { \mathrm { o u t } } \} _ { i = 1 } ^ { m } \stackrel { \mathrm { i . i . d . } } { \sim }$ $P ^ { * } ( \mathbf { X } \mid \mathbf { X } \in \bar { A } )$ . We define the Monte Carlo estimator of PNS(A, B) as

$$
{ \widehat { \mathrm { P N S } } } ( A , B ) = \left( { \frac { 1 } { m } } { \sum _ { i = 1 } ^ { m } } \mathbb { I } { \big [ } f ( \mathbf { x } _ { i } ^ { \mathrm { i n } } ) \in B { \big ] } \right) \left( { \frac { 1 } { m } } { \sum _ { i = 1 } ^ { m } } \mathbb { I } { \big [ } f ( \mathbf { x } _ { i } ^ { \mathrm { o u t } } ) \in { \bar { B } } { \big ] } \right)\tag{26}
$$

By the law of large numbers, each empirical average converges almost surely to its corresponding conditional probability in (21). Consequently,

$$
{ \widehat { \mathrm { P N S } } } ( A , B ) \ { \xrightarrow { \mathrm { a . s . } } } \ \operatorname { P N S } ( A , B ) .\tag{27}
$$

## APPENDIX B DERIVATION OF THE EMPIRICAL AND DIFFERENTIABLE PNS OBJECTIVE

This appendix derives the empirical objective used in (14). Let the input variables be decomposed as $\mathbf { X } = ( \mathbf { I } , \mathbf { Z } )$ , where I denotes the intervened variables and Z is treated as the background variables to be marginalized. For an input region $A \subseteq \mathrm { V a l } _ { \mathbf { I } }$ and an output region $B \subseteq \operatorname { V a l } _ { \mathbf { Y } }$ , the stochasticintervention-based PNS can be written as

$$
\begin{array} { r l } & { \mathrm { P N S } ( \mathbf { I } , { A } , { B } ) = \mathbb { E } _ { \mathbf { z } \sim P ( \mathbf { Z } ) } \Big [ P ( \mathbf { Y } _ { \pi ( \mathbf { I } ; P ^ { A } ) } \in B \mid \mathbf { z } ) } \\ & { \qquad \quad \cdot P ( \mathbf { Y } _ { \pi ( \mathbf { I } ; P ^ { \bar { A } } ) } \in \bar { B } \mid \mathbf { z } ) \Big ] , } \end{array}\tag{28}
$$

where $P ^ { A } ( \mathbf { I } ) = P ( \mathbf { I } \mid \mathbf { I } \in A )$ and $P ^ { \bar { A } } ( \mathbf { I } ) = P ( \mathbf { I } \mid \mathbf { I } \in \bar { A } )$ Since f is deterministic,

$$
\begin{array} { r } { P ( \mathbf { Y } _ { \pi ( \mathbf { I } ; P ^ { A } ) } \in B \mid \mathbf { z } ) = \mathbb { E } _ { \mathbf { i } \sim P ^ { A } } \left[ \mathbb { I } ( f ( \mathbf { i } , \mathbf { z } ) \in B ) \right] , } \end{array}\tag{29}
$$

and the complement term is defined analogously.

Given a dataset $\mathcal { D } = \{ \mathbf { x } _ { n } \} _ { n = 1 } ^ { N }$ , each sample is decomposed accordingly as $\{ \mathbf { x } _ { n } ~ = ~ ( \mathbf { i } _ { n } , \mathbf { z } _ { n } )$ . The empirical intervention weights are

$$
\omega ^ { A } ( \mathbf { x } _ { j } ) = \frac { \mathbb { I } ( \mathbf { i } _ { j } \in A ) } { \sum _ { \ell = 1 } ^ { N } \mathbb { I } ( \mathbf { i } _ { \ell } \in A ) } , \qquad \omega ^ { \bar { A } } ( \mathbf { x } _ { j } ) = \frac { \mathbb { I } ( \mathbf { i } _ { j } \in \bar { A } ) } { \sum _ { \ell = 1 } ^ { N } \mathbb { I } ( \mathbf { i } _ { \ell } \in \bar { A } ) } .\tag{30}
$$

Then the empirical estimator is

$$
\begin{array} { r } { \widehat { \mathrm { P N S } } ( \mathbf { I } , A , B ) = \displaystyle \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \Big [ \displaystyle \sum _ { j = 1 } ^ { N } \omega ^ { A } ( \mathbf { x } _ { j } ) \mathbb { I } \big ( f ( \mathbf { i } _ { j } , \mathbf { z } _ { k } ) \in B \big ) \Big ] } \\ { \cdot \Big [ \displaystyle \sum _ { j = 1 } ^ { N } \omega ^ { \bar { A } } ( \mathbf { x } _ { j } ) \mathbb { I } \big ( f ( \mathbf { i } _ { j } , \mathbf { z } _ { k } ) \in \bar { B } \big ) \Big ] . } \end{array}\tag{31}
$$

This estimator averages over the background part $\mathbf { z } _ { k }$ while drawing the intervened part $\mathbf { i } _ { j }$ from the empirical in-region and out-of-region distributions.

Substituting the softening relaxations in (11)–(13) and the mixing operation in (15) into the hard empirical estimator (31) yields the differentiable objective

$$
\begin{array} { r l } & { \widehat { \mathrm { P N S } } ( A _ { \boldsymbol { \theta } , \mathbf { m } } , B _ { \boldsymbol { \phi } } ) = } \\ & { \frac { 1 } { N } \displaystyle \sum _ { k = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \omega _ { \boldsymbol { \theta } , \mathbf { m } } ^ { A } ( \mathbf { x } _ { j } ) s _ { \boldsymbol { \phi } } ^ { B } \big ( f ( \tilde { \mathbf { x } } _ { j k } ) \big ) \displaystyle \sum _ { j = 1 } ^ { N } \omega _ { \boldsymbol { \theta } , \mathbf { m } } ^ { \bar { A } } ( \mathbf { x } _ { j } ) s _ { \boldsymbol { \phi } } ^ { \bar { B } } \big ( f ( \tilde { \mathbf { x } } _ { j k } ) \big ) , } \end{array}\tag{32}
$$

## APPENDIX C DETAILS ON EXPERIMENTS

## A. Datasets

We evaluate SNRE on four real-world tabular datasets and two synthetic datasets. German Credit [39] contains personal and financial attributes of credit applicants. The label indicates whether the applicant is classified as good or bad credit. Phishing Websites [42] consists of website and URL-related attributes. The label indicates whether a website is phishing or legitimate. Mushroom [41] contains categorical attributes describing physical properties of mushrooms, such as cap shape, cap color, odor, gill size, stalk surface, veil type, and habitat. The label indicates whether a mushroom is edible or poisonous. Give Me Some Credit (GMSC) [40] contains financial and delinquency-related attributes. The label indicates whether an individual experiences serious delinquency within two years. Synthetic Circles is a two-dimensional synthetic dataset constructed from concentric circular structures. The label is determined by the constructed geometric region. Synthetic Blobs is a two-dimensional synthetic dataset constructed from Gaussian clusters with controlled geometric structure. The label is generated according to the predefined cluster/region configuration.

## B. Baselines

1) Rule-based explanation baselines: IDS [18] learns interpretable decision sets, i.e., unordered sets of if–then rules optimized for accuracy, conciseness, and low rule overlap. SBRL [19] learns scalable Bayesian rule lists by optimizing a posterior over ordered if–then rules constructed. FRL [20] learns falling rule lists, where the predicted probability is constrained to decrease monotonically along the ordered list. DT [24] denotes a standard decision tree classifier trained by recursive feature splitting. The input event A is obtained by taking the union of leaves that predict the target behavior. FLDT [25] denotes a feature-learning decision tree, which alternates between learning a sparse feature transformation and fitting a differentiable decision tree on the learned representations. RRL [26] denotes Rule-based Representation Learner, which learns non-fuzzy logical rules through a differentiable relaxation and gradient-based optimization. CORELS [21] learns certifiably optimal rule lists for categorical data by optimizing regularized empirical risk with an optimality certificate. SamRuLe [23] is a sampling-based scalable rule-list learner that approximates optimal rule lists on large datasets with statistical guarantees. MDL-RL [22] denotes an MDLbased rule-list learner that selects compact probabilistic rule lists according to the minimum description length principle. DRNet [27] denotes a decision-rule network that learns interpretable decision rule sets using a neural architecture with logical rule components. IMLI [28] learns sparse propositionallogic classification rules through an iterative learning process that jointly optimizes accuracy and rule size.

2) PNS-based explanation baselines: FANS [14] learns feature subsets by maximizing necessity-and-sufficiency effects under prediction perturbations. LEWIS [34] estimates global necessity and sufficiency scores through counterfactual interventions. LENS [31] identifies local minimal sufficient factors and aggregates them into global feature scores. NSE4T [32] estimates local necessity and sufficiency scores through feature perturbations. CF2 [33] combines factual and counterfactual reasoning to obtain necessary and sufficient explanations. NSEG [15] optimizes a lower bound of PNS to learn necessary and sufficient masks.

## C. Evaluation Metrics

Different experiments use different evaluation metrics. We summarize them below.

1) Sufficiency and necessity scores: Let $\mathcal { D } _ { \mathrm { e v a l } } = \{ { \bf x } _ { k } \} _ { k = 1 } ^ { n }$ be the evaluation set. Given the learned hard feature mask m $\in \{ 0 , 1 \} ^ { d }$ , the input and output events are defined as

$$
A = \{ \mathbf { x } | g ( \mathbf { x } ; \mathbf { m } ) \leq 0 \} , \qquad B = \{ \mathbf { y } | h ( \mathbf { y } ) \leq 0 \} .\tag{33}
$$

We denote the samples inside and outside the input region by

$$
\mathcal { D } _ { A } = \big \{ \mathbf { x } _ { j } \in \mathcal { D } _ { \mathrm { e v a l } } \vert \mathbf { x } _ { j } \in A \big \} , \qquad \mathcal { D } _ { \bar { A } } = \mathcal { D } _ { \mathrm { e v a l } } \setminus \mathcal { D } _ { A } .\tag{34}
$$

The empirical sufficiency score is computed as

$$
\widehat { \mathrm { P S } } = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } \frac { 1 } { | \mathscr { D } _ { A } | } \sum _ { \mathbf { x } _ { j } \in \mathscr { D } _ { A } } \mathbb { I } [ f ( \mathbf { m } \odot \mathbf { x } _ { j } + ( 1 - \mathbf { m } ) \odot \mathbf { x } _ { k } ) \in B ] .\tag{35}
$$

Here, m $\odot \mathbf { x } _ { j }$ enforces the feature dimensions selected by m to take values from an in-region sample, while $( 1 - \mathbf { m } ) \odot \mathbf { x } _ { k }$ together with the empirical averaging over different samples ${ \bf x } _ { k } .$ , marginalizes the feature dimensions not selected by m.

Similarly, the empirical necessity score is computed as

$$
\widehat { \mathrm { P N } } = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } \frac { 1 } { | { \cal D } _ { \bar { A } } | } \sum _ { { \bf x } _ { j } \in { \mathcal { D } } _ { \bar { A } } } \mathbb { I } [ f ( { \bf m } \odot { \bf x } _ { j } + ( 1 - { \bf m } ) \odot { \bf x } _ { k } ) \not \in B ] .\tag{36}
$$

It measures whether forcing the selected features to lie outside A makes the model output fall outside B, while again marginalizing the unselected features through the evaluation samples.

## D. Implementation

All predictors to be explained are three-layer MLPs with two hidden layers and one output layer. Adam is used as the optimizer. Each dataset is split into training, validation, and test sets, and predictor training is stopped early according to the validation loss, with a maximum of 200 epochs and a patience of 20 epochs.

For the regional-explanation experiments, the hidden dimension is 64, the batch size is 1024, and the data split is 60%/20%/20%. SNRE uses a learning rate of 0.008 and an early-stopping patience of 25, with hyperparameters selected on the validation set.