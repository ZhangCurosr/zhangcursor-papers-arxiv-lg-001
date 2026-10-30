# Bayes-Consistent Losses for Adaptive Inference in Classification

Shirley Xiaoqi Liu<sup>∗†</sup> Florence Regol<sup>∗‡</sup> Antonios Valkanas<sup>∗‡</sup>

Patrick Rebeschini<sup>†</sup> Mark Coates<sup>‡</sup>

## Abstract

Adaptive inference systems balance accuracy and inference cost by routing inputs to models with varying capabilities. Although jointly training models and routers is empirically more sampleeficient than training them separately, principled losses for joint training remain underexplored. We address this gap in a classification setting by proposing cost-sensitive hinge losses for jointly training classifiers with single-stage or cascade routers. Single-stage routers directly select a model per input, whereas cascade routers sequentially decide at each stage whether to predict with the current model or defer to the next stage, which uses a more accurate but more costly model. For both settings, we prove linear calibration bounds establishing Bayes consistency: minimizing the hinge risk recovers the Bayes-optimal risk under the target cost-sensitive zero-one loss. Moreover, since routers can require more labeled data to train than classifiers, we propose a novel post-training prediction-powered inference (PPI) procedure that uses unlabeled data to refine a trained single-stage router. Experiments on synthetic and real datasets demonstrate that the proposed hinge losses are efective for joint training and that PPI can correct routing errors when labeled data is scarce.

## 1 Introduction

Adaptive inference systems allocate computation on a per-sample basis to balance accuracy and inference cost. For each sample, they decide which one of several predictive models with diferent cost-accuracy profiles should be used for prediction. This framework covers a broad range of methods, including learning to defer [Mozannar and Sontag, 2020], model cascades [Chen et al., 2024a, Zhang et al., 2024, Valkanas et al., 2025b] and dynamic neural networks (DNNs) [Han et al., 2022a], including early-exit systems [Xin et al., 2021]. We study adaptive inference in a classification setting that involves two components: an array of classifiers with varying cost-accuracy profiles, and a router module that decides, for each sample, which classifier makes the prediction.

A central challenge is how to train such systems efectively. Many approaches rely on separate training, where the classifiers are trained independently and routing is determined by a rule chosen separately, such as confidence-score thresholding [Schuster et al., 2022, Jitkrittum et al., 2023]. This approach is often sample-ineficient: each classifier is trained on the full sample space, even though the trained router may send it samples from only a small subset of the space. This mismatch may lead to suboptimal results [Jitkrittum et al., 2023] and hinder the training process [Kubaty et al., 2025]. In contrast, joint training aligns classifier and router objectives, encouraging each classifier to specialize in samples routed to it. Empirically, joint training substantially improves prediction accuracy at a fixed sample size, including with losses designed to align training with test-time routing [Han et al., 2022b, Yu et al., 2023, Regol et al., 2024, Kubaty et al., 2025]. These results motivate the design of joint-training losses with consistency guarantees, which are currently scarce.

![](images/5f2e8700ca5d274deae5ac88b498276b783654085b13770c96ad8cd6ef4d4a5a.jpg)  
Figure 1: Illustration of flat (left) and cascade (right) routing models. A flat router selects a single classifier $f _ { t }$ based on X, while a cascade router sequentially invokes classifiers $f _ { 1 } , \ldots , f _ { T }$ until the first exit decision is made based on $( X , Z _ { 1 : t } )$ , or until the last classifier $f _ { T }$ is reached.

We address this gap in adaptive inference for classification under two common routing models, as illustrated in Fig. 1, considering T classifiers. Each sample has a base feature X, available from the start, and additional features $Z _ { 1 } , \dots , Z _ { T }$ , where $Z _ { t }$ is revealed once the t-th classifier is reached. In the single-stage (flat) model, the router directly selects a classifier t based on X alone, and the classifier then predicts from $( X , Z _ { t } )$ This model covers certain configurations of elastic networks [Devvrit et al., 2023], learning to defer to multiple experts [Verma et al., 2023], and width-adaptive DNNs [Han et al., 2022a]. In the multi-stage (cascade) model, classifiers are invoked sequentially: at each stage t, the router uses the accumulated features $( X , Z _ { 1 : t } )$ to either exit with the t-th classifier’s prediction or defer to the next stage, which uses a more powerful classifier. We assume that each classifier t remains lightweight by predicting from $( X , Z _ { t } )$ alone. This model covers cascades [Chen et al., 2024a, Zhang et al., 2024, Valkanas et al., 2025b], early-exit systems [Xin et al., 2021], and, more generally, depth-adaptive DNNs [Han et al., 2022a].

In our LLM experiments, for example, $Z _ { 1 } , \dots , Z _ { T }$ correspond to embeddings of the answers produced by pretrained LLMs of increasing size, which serve as inputs to the classifiers (see Appendix H for further justification of this modeling choice). Accessing $Z _ { t }$ incurs a known cost $\kappa _ { t } ,$ e.g., the cost of running the t-th LLM. Such systems are typically designed so that $\kappa _ { t }$ varies with t and is non-decreasing in t. Due to the heterogeneous costs, jointly training the classifiers and router requires a cost-sensitive loss that balances the classification errors, measured by the zero-one loss, against the costs. We propose two such losses, one for each routing model.

## 1.1 Main Contributions

• We propose hinge-based surrogates of the cost-sensitive zero-one loss for jointly training classifiers and routers in both flat and cascade models. In both cases, we prove linear calibration bounds (Theorems 3.1 and 3.2) that control the excess target risk by the excess hinge risk, thereby establishing Bayes consistency of the surrogate losses. This linear rate is sharp [Frongillo and Waggoner, 2021]. These results place joint training on a principled theoretical foundation. We validate our theory and demonstrate its practical utility through experiments on synthetic and real datasets.

• A key technical contribution is the calibration bound for the cascade hinge loss. It is challenging to establish because the exit stage is a stopping time induced by a sequence of adaptive deferral decisions. We address this via a dynamic programming approach [Bertsekas, 2017, Shiryaev, 2008], applying backward induction to recursively establish stage-wise calibration through a novel risk decomposition that separates excess risks into contributions from suboptimal deferral rules and suboptimal classifiers. A matching lower bound (Proposition C.8) shows that the resulting calibration constant is tight.

• Since routers can require more labeled data to train than classifiers, we also propose a novel post-training prediction-powered inference (PPI) procedure that leverages unlabeled data to refine a trained flat router. At each input, PPI recovers the oracle routing decision with high probability, whenever the width of its confidence interval on risk improvement is narrower than the gap between the oracle and the second-best routing decisions (Theorem 4.1). This constitutes the first integration of PPI into adaptive inference.

## 1.2 Related Work

Learning to defer allows a model to either predict with a learned classifier or defer to one or more experts with heterogeneous capabilities and costs [Madras et al., 2018, Keswani et al., 2021, Wilder et al., 2021]. Here, only the classifier and the deferral rule are learned; the experts are fixed. Building on foundational works on Bayes consistency [Bartlett et al., 2006, Tewari and Bartlett, 2007, Steinwart, 2007, Zhang, 2004], a line of works develops Bayes-consistent surrogate losses with a single expert [Mozannar and Sontag, 2020, Verma and Nalisnick, 2022] or with multiple experts [Verma et al., 2023]. For the single-expert setting, later works establish realizable H-consistency, a guarantee relative to a restricted hypothesis class [Mozannar et al., 2023, Mao et al., 2024a], where Mao et al. [2024a] further identify surrogates that are simultaneously realizable H-consistent, Bayes-consistent, and admit H-consistency bounds [Awasthi et al., 2022]. Closer to our multi-model setting, Mao et al. [2024b] and Mao et al. [2023] prove H-consistency bounds in the multiple-expert setting. Mao et al. [2024b] consider the standard setting in which the classifier and the deferral rule are learned jointly, while Mao et al. [2023] consider a two-stage setting where the classifier is first trained then fixed, and the deferral rule is learned in a second stage. Finally, Mao et al. [2025] establish realizable H-consistency, H-consistency bounds, and Bayes consistency for learning to defer to multiple experts, in both the standard and two-stage settings.

In contrast to learning to defer, which learns a single classifier, Regol et al. [2025] and Pugnana et al. [2025] recently studied a setting where all classifiers involved are learned: they study joint training of two classifiers with a deferral rule (or router), and their analysis is restricted to this setting. Since adaptive inference systems often involve more than two models [Chen et al., 2024a, Zhang et al., 2024, Valkanas et al., 2025b], we address this limitation by developing principled losses for jointly training multiple classifiers with the router under both flat and cascade models. As our work is primarily theoretical, we defer discussion of the rich literature on practical architectures and empirical performance in adaptive inference to Appendix A.

## 1.3 Notation

We use $x _ { i } = [ x ] _ { i }$ to denote the i-th entry of vector x. We write $[ n ] : = \{ 1 , 2 , . . . , n \}$ , and for $i , j \in  { \mathbb { N } }$ with $i \leq j$ , we use the shorthand $a _ { i : j } : = \{ a _ { i } , a _ { i + 1 } , \ldots , a _ { j } \}$ for sequences of variables or functions. We denote the n-dimensional probability simplex by $\Delta ^ { n - 1 }$ . We use uppercase letters for random variables, lowercase for their realizations (e.g., X and x), and calligraphic letters to denote spaces of realizations $( \mathrm { e . g . } , x \in { \mathcal { X } } )$

## 2 Multi-Model Classification

We study a multi-model classification problem, where the goal is to learn a tuple $\pi = ( f _ { 1 : T } , r )$ ， consisting of $T \geq 2$ classifiers $f _ { 1 : T }$ and a router r. Each classifier $f _ { t }$ takes the form:

$$
f _ { t } : \mathcal { X } \times \mathcal { Z } _ { t } \to [ K ] , \quad \mathrm { f o r } \quad t \in [ T ] ,\tag{1}
$$

where $\mathcal { X } \subset \mathbb { R } ^ { p }$ denotes the space of the base features $( \mathrm { e . g . }$ , a multiple-choice question prompt), [K] is the label space for K classes $( \mathrm { e . g . }$ , the answer choices), and $\mathcal { Z } _ { t } \subset \mathbb { R } ^ { p _ { t } }$ denotes the space of the additional feature (e.g., LLM embeddings) available at $f _ { t } .$ . Accessing $Z _ { t }$ incurs a known nonnegative cost $\kappa _ { t } .$ , with $\kappa _ { t } \geq \kappa _ { t - 1 }$ , reflecting that later classifiers are more accurate but more costly.<sup>1</sup> The routing rule takes the form

$$
r : \mathcal { H }  [ T ] ,\tag{2}
$$

which maps the history information $\begin{array} { r } { \mathcal { H } \subseteq \mathcal { X } \times \prod _ { t = 1 } ^ { T } \mathcal { Z } _ { t } } \end{array}$ available to the router to a classifier index.<sup>2</sup> The space H varies with the router type, as we specify in Section 2.1. We denote by Π the class of all multi-model classifiers $\pi = \left( f _ { 1 : T } , r \right)$ satisfying (1) and (2).

For a given $h \in \mathcal H$ , the prediction yˆ of $\pi = \left( f _ { 1 : T } , r \right)$ is the output of the classifier selected by the router:

$$
\hat { y } = f _ { r ( h ) } ( x , z _ { r ( h ) } ) .\tag{3}
$$

Training data. To train $\pi \in \Pi$ , we use n training samples generated i.i.d. according to

$$
X ^ { ( i ) } , Z _ { 1 : T } ^ { ( i ) } , Y ^ { ( i ) } \stackrel { \mathrm { i . i . d . } } { \sim } P \quad \mathrm { f o r } \quad i \in [ n ] ,\tag{4}
$$

where $X ^ { ( i ) } \in \mathcal { X }$ and $Z _ { t } ^ { ( i ) } \in \mathcal { Z } _ { t }$ for $t \in [ T ]$ are features, $Y ^ { ( i ) } \in [ K ]$ is the label, and P is the joint distribution over $( X ^ { ( i ) } , Z _ { 1 : T } ^ { ( i ) } , Y ^ { ( i ) } )$ . We drop the superscript <sup>(i)</sup> for a generic sample. In some cascade applications, it is natural to assume the conditional independence

$$
Y \perp Z _ { 1 : t - 1 } \mid ( X , Z _ { t } ) ,\tag{5}
$$

where $( X , Z _ { t } )$ subsumes all information in $Z _ { 1 : t - 1 }$ relevant to predicting Y. We now formalize two routing models and their corresponding multi-model classifiers.

## 2.1 Single-Stage and Cascade Routing Models

The single-stage router directly selects one of the $T$ classifiers using only the base feature, formalized as:

Definition 2.1 (Single-stage (flat) router). With history information $\mathcal { H } = \mathcal { X }$ , a flat router takes the form

$$
r : \mathcal { X }  [ T ] .\tag{6}
$$

The multi-stage cascade router sequentially decides, at each stage t, whether to exit with classifier $f _ { t }$ or defer to the more powerful classifier $f _ { t + 1 }$ after collecting additional features, as formalized below:

Definition 2.2 (Cascade router). With history information $\begin{array} { r } { \mathcal { H } _ { t } = \mathcal { X } \times \prod _ { \tau = 1 } ^ { t } \mathcal { Z } _ { \tau } } \end{array}$ available at stage $t \in [ T ]$ , a cascade router consists of stage-wise deferral rules

$$
d _ { t } : \mathcal { H } _ { t } \to \{ 0 , 1 \} \quad \mathrm { f o r } \quad t \in [ T - 1 ] ,\tag{7}
$$

where $d _ { t } ( h _ { t } ) = 0$ indicates exit at stage t and $d _ { t } ( h _ { t } ) = 1$ indicates deferral to the next stage. We fix $d _ { T } \equiv 0 .$ , so the cascade always terminates at the final stage T. Writing $\mathcal { H } = \mathcal { H } _ { T }$ for the full history, the stopping time induced by the deferral rules is the cascade routing rule

$$
r ( h ) \in \operatorname* { m i n } \{ t \in [ T ] : d _ { t } ( h _ { t } ) = 0 \} .\tag{8}
$$

Restricting the class Π of multi-model classifiers to flat or cascade routers (Definitions 2.1 or 2.2) yields the hypothesis classes $\Pi _ { \mathrm { H a t } } \subset \Pi$ and $\Pi _ { \mathrm { c a s c } } \subset \Pi$ , respectively, which are the focus of this work.

## 2.2 The Target Cost-Sensitive Zero-One Loss

To capture the tradeof between prediction error and cost, we follow prior work on cost-sensitive classification and learning to defer [Cao et al., 2022, Mao et al., 2024b] and augment the standard zeroone loss with the costs. This yields the $\jmath _ { - 1 - c }$ loss, denoted by $\begin{array} { r } { \ell : \mathcal { X } \times \dot { \prod } _ { t = 1 } ^ { T } \mathcal { Z } _ { t } \times [ K ] \times \boldsymbol { \Pi } \times \mathbb { R } _ { > 0 } ^ { T }  \mathbb { R } _ { \ge 0 } } \end{array}$ For a multi-model classifier $\pi = ( f _ { 1 : T } , r ) \in \Pi$ and a cost vector $c \in \mathbb { R } _ { \geq 0 } ^ { T }$ associated with $f _ { 1 : T }$ , we define

$$
\ell ( x , z _ { 1 : T } , y ; \pi , c ) = \mathbb { 1 } \{ f _ { r ( h ) } ( x , z _ { r ( h ) } ) \neq y \} + c _ { r ( h ) } .\tag{9}
$$

That is, an incorrect prediction incurs a unit loss, while exiting at the t-th classifier additionally incurs a cost $c _ { t }$ . In (9), the semicolon separates the random data arguments from the fixed multi-model classifier π and cost vector c. The corresponding 0-1-c population risk is then

$$
\mathcal { R } ( \pi ) \equiv \mathcal { R } ( \pi ; c ) : = \mathbb { E } \big [ \ell ( X , Z _ { 1 : T } , Y ; \pi , c ) \big ] .\tag{10}
$$

The form of c difers between the two routing models. For the single-stage router, $\boldsymbol { c } = [ \kappa _ { 1 } , \kappa _ { 2 } , \ldots , \kappa _ { T } ] ^ { \top }$ where recall $\kappa _ { t }$ denotes the cost of executing only the t-th classifier. In a cascade, since the classifiers are invoked sequentially, costs accumulate along the execution path, yielding $c = [ \kappa _ { 1 } , \kappa _ { 1 } +$ $\kappa _ { 2 } , . . . , \sum _ { t = 1 } ^ { T } \kappa _ { t } \big ] ^ { \top }$

## 3 Main Results

This section develops surrogate losses for the target 0-1-c loss (9), one for the flat model and one for the cascade model. Throughout, we use a $\mathrm { t i l d e } \widetilde { \cdot }$ to denote soft (score-based or probabilistic) predictors, reserving untilded notation for their hard, discrete counterparts. We introduce two soft multi-model-classifier classes $\widetilde { \Pi } _ { \mathrm { { f l a t } } }$ and $\widetilde { \Pi } _ { \mathrm { c a s c } }$ in parallel to $\Pi _ { \mathrm { { f l a t } } }$ and $\Pi _ { \mathrm { c a s c } }$ . Each element $\tilde { \pi }$ of these classes is composed of a sequence of score functions $\tilde { f } _ { 1 : T }$ and a soft router $\tilde { r }$ that outputs a probability distribution over the T classifiers. Formally, we define the class $\widetilde { \Pi } _ { \mathrm { { f l a t } } }$ as:

$$
\begin{array} { r l } & { \widetilde { \Pi } _ { \mathrm { f l a t } } = \Big \{ \tilde { \pi } = ( \tilde { f } _ { 1 : T } , \tilde { r } ) : \tilde { f } _ { t } : \mathcal { X } \times \mathcal { Z } _ { t } \to \mathbb { R } ^ { K } \mathrm { w i t h } } \\ & { \displaystyle \sum _ { k \in [ K ] } [ \tilde { f } _ { t } ( x , z _ { t } ) ] _ { k } = 0 \forall ( x , z _ { t } ) , \mathrm { a n d } \tilde { r } : \mathcal { X } \to \Delta ^ { T - 1 } \Big \} . } \end{array}\tag{11}
$$

Each $\widetilde { \pi } = ( \widetilde { f } _ { 1 : T } , \widetilde { r } ) \in \widetilde { \Pi } _ { \mathrm { f l a t } }$ induces a hard multi-model classifier $\pi = ( f _ { \mathrm { 1 : } T } , r ) = \varphi _ { \mathrm { f l a t } } ( \tilde { \pi } ) \in \Pi _ { \mathrm { f l a t } }$ via a deterministic argmax map φ<sub>flat</sub> $\colon \widetilde { \Pi } _ { \mathrm { f l a t } }  \Pi _ { \mathrm { f l a t } }$ , defined by the componentwise relations:

$$
f _ { t } ( x , z _ { t } ) \in \mathop { \mathrm { a r g } } _ { k \in [ K ] } \times [ \tilde { f } _ { t } ( x , z _ { t } ) ] _ { k } , \quad \forall t \in [ T ] ,\tag{12}
$$

$$
r ( x ) \in \underset { t \in [ T ] } { \arg \operatorname* { m a x } } \tilde { r } _ { t } ( x ) , \mathrm { w i t h } \tilde { r } _ { t } ( x ) = [ \tilde { r } ( x ) ] _ { t } .\tag{13}
$$

For the cascade, we define the class $\widetilde { \Pi } _ { \mathrm { c a s c } }$ and the map $\varphi _ { \mathrm { c a s c } } : \widetilde { \Pi } _ { \mathrm { c a s c } }  \Pi _ { \mathrm { c a s c } }$ analogously. The soft router $\tilde { r }$ consists of soft deferral rules $\tilde { d } _ { t } : \mathcal { H } _ { t } \to [ 0 , 1 ]$ for $t \in [ T - 1 ]$ with $d _ { T } \equiv 0$ , and the map φ<sub>casc</sub> induces the hard, thresholding router r which replaces (13) and mirrors the structure of (8), taking the form

$$
r ( h ) \in \operatorname* { m i n } \{ t \in [ T ] : \tilde { d } _ { t } ( h _ { t } ) < 0 . 5 \} .\tag{14}
$$

Our goal is to design, for each model $\in \{ \mathrm { { f l a t } , \mathrm { { c a s c } } } \}$ , a surrogate loss defined over the soft class $\widetilde { \Pi } _ { \mathrm { m o d e l } }$ such that driving the surrogate risk of $\tilde { \pi } \in \tilde { \Pi } _ { \mathrm { m o d e l } }$ to its minimum drives the target 0-1-c risk (10) of the induced hard multi-model classifier to its minimum over the hard class $\Pi _ { \mathrm { m o d e l } }$ . In particular, any minimizer $\tilde { \pi } _ { \mathrm { m o d e l } } ^ { * }$ of the surrogate risk induces a minimizer $\pi _ { \mathrm { m o d e l } } ^ { * } { = } \varphi _ { \mathrm { m o d e l } } ( \tilde { \pi } _ { \mathrm { m o d e l } } ^ { * } )$ of the target risk. This desired property is Bayes consistency. In the sequel, we drop subscript from $\varphi _ { \mathrm { m o d e l } }$ when the model is clear from context.

Cost-sensitive hinge loss. Let $[ x ] _ { + } : = \operatorname* { m a x } ( x , 0 )$ . We propose a cost-sensitive hinge loss ℓ<sup>hinge</sup> : $\mathbb { R } ^ { K } \times [ K ] \times \mathbb { R } _ { \geq 0 }  \mathbb { R } _ { \geq 0 }$ , which acts on the score vector v output by a given soft classifier $( \mathrm { e . g . } , \tilde { f } _ { t } )$ , the true label $y ,$ and the cost c of the classifier as follows:

$$
\ell ^ { \mathrm { h i n g e } } ( v , y ; c ) = \sum _ { k \in [ K ] \setminus y } \left[ v _ { k } + \frac { 1 } { K - 1 } \right] _ { + } + \frac { K } { K - 1 } c .\tag{15}
$$

This loss is a cost-sensitive extension of the multiclass hinge loss from [Tarigan and van de Geer, 2008] and is also used for building the surrogate loss in [Regol et al., 2025] for jointly training two classifiers. We use it in Sections 3.1 and 3.2 to construct surrogates of the target loss (9) for joint training in $\widetilde { \Pi } _ { \mathrm { { f l a t } } }$ and $\widetilde { \Pi } _ { \mathrm { c a s c } }$

## 3.1 Bayes-Consistent Loss for the Flat Model

For training $\tilde { \pi } ~ = ~ ( \tilde { f } _ { 1 : T } , \tilde { r } ) ~ \in ~ \widetilde { \Pi } _ { \mathrm { f l a t } }$ in the flat setting, we construct the surrogate loss $\ell _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } }$ $\begin{array} { r } { \mathcal { X } \times \prod _ { t = 1 } ^ { T } \mathcal { Z } _ { t } \times [ K ] \times \widetilde { \Pi } _ { \mathrm { f a t } } \times \mathbb { R } _ { \ge 0 } ^ { T } \to \mathbb { R } _ { \ge 0 } } \end{array}$ as a weighted sum of the hinge loss $\ell ^ { \mathrm { h i n g e } }$ across all classifiers:

$$
\ell _ { \mathrm { H a t } } ^ { \mathrm { h i n g e } } \left( x , z _ { 1 : T } , y ; \tilde { \pi } , c \right) = \sum _ { t = 1 } ^ { T } { \tilde { r } } _ { t } ( x ) \ell ^ { \mathrm { h i n g e } } \left( { \tilde { f } } _ { t } ( x , z _ { t } ) , y ; c _ { t } \right) ,\tag{16}
$$

with the associated risk $\mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) \equiv \mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } \left( \tilde { \pi } ; c \right) : = \mathbb { E } \left[ \ell _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } \left( X , Z _ { 1 : T } , Y ; \tilde { \pi } , c \right) \right]$ . The calibration bound below shows that the excess 0-1-c risk is linearly bounded by the excess hinge risk in the flat setting.

Theorem 3.1 (Linear calibration of $\ell _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } \mathrm { ~ w . r . t . ~ } \ell )$ . Define $\begin{array} { r } { \mathcal { R } _ { \mathcal { \hat { H } } a t } ^ { * } { : = } \operatorname* { m i n } _ { \pi \in \Pi _ { \mathcal { H } a t } } \mathcal { R } ( \pi ) } \end{array}$ and $\mathcal { R } _ { f l a t } ^ { h i n g e * } : =$ min $\cdot \tilde { \pi } \in \widetilde { \Pi } _ { f l a t }  \mathcal { R } _ { f l a t } ^ { h i n g e } ( \tilde { \pi } )$ . Then, for any $\tilde { \pi } \in \widetilde { \Pi } _ { f l a t }$

$$
\begin{array} { r } { \mathcal { R } \big ( \varphi ( \tilde { \pi } ) \big ) - \mathcal { R } _ { f a t } ^ { * } \leq T ( K - 1 ) \Big ( \mathcal { R } _ { f a t } ^ { h i n g e } ( \tilde { \pi } ) - \mathcal { R } _ { f a t } ^ { h i n g e * } \Big ) . } \end{array}\tag{17}
$$

This implies $\mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } )  \mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e * } } \Rightarrow \mathcal { R } \big ( \varphi ( \tilde { \pi } ) \big )  \mathcal { R } _ { \mathrm { f l a t } } ^ { * }$ , establishing Bayes consistency of the surrogate loss $\ell _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } }$ . We detail the proof in Appendix B.

## 3.2 Bayes-Consistent Loss for the Cascade

For training $\tilde { \pi } = ( \tilde { f } _ { 1 : T } , \tilde { r } ) \in \widetilde { \Pi } _ { \mathrm { c a s c } }$ , we build the cascade surrogate hinge loss $\begin{array} { r } { \ell _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } : \mathcal { X } \times \prod _ { t = 1 } ^ { T } \mathcal { Z } _ { t } \times } \end{array}$ $[ K ] \times \widetilde { \Pi } _ { \mathrm { c a s c } } \times \mathbb { R } _ { > 0 } ^ { T }  \mathbb { R } _ { \ge 0 }$ as follows:

$$
\ell _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( x , z _ { 1 : T } , y ; \tilde { \pi } , c ) = \sum _ { t = 1 } ^ { T } { \tilde { r } _ { t } ( x , z _ { 1 : t } ) \ell ^ { \mathrm { h i n g e } } \big ( \tilde { f } _ { t } ( x , z _ { t } ) , y ; c _ { t } \big ) } ,\tag{18}
$$

where

$$
\tilde { r } _ { t } ( x , z _ { 1 : t } ) = \left( \prod _ { \tau = 1 } ^ { t - 1 } \tilde { d } _ { \tau } ( x , z _ { 1 : \tau } ) \right) \big ( 1 - \tilde { d } _ { t } ( x , z _ { 1 : t } ) \big ) ,\tag{19}
$$

and $\tilde { d } _ { T } \equiv 0$ . The outer sum in (18) runs over exit stages $t \in [ T ]$ . The $\tilde { r } _ { t } ( x , z _ { 1 : t } )$ term represents the probability of exiting at stage t, and the $\ell ^ { \mathrm { h i n g e } }$ term is the hinge loss incurred upon exiting at that stage. We define the risk associated with the loss function $\ell _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } }$ as $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) \equiv \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ; c ) : =$ $\mathbb { E } \lceil \ell _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( X , Z _ { 1 : T } , Y ; \tilde { \pi } , c ) \rceil$

The cascade setting is technically more challenging to analyze than the flat setting because adaptive deferral decisions induce the stopping time in (14), which indexes the classifier ultimately invoked. As a result, the optimal deferral decision at each stage must account for both the immediate loss and the expected future loss. To prove that $\ell _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } }$ is calibrated in an analogous sense to Theorem 3.1 of the flat setting, we cast the cascade as a finite-horizon dynamic program and use backward induction with stage-wise risk decomposition to recursively propagate single-stage calibration through the cascade, ultimately yielding Theorem 3.2 below.

Theorem 3.2 (Linear calibration of $\ell _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } \ \mathrm { ~ w . r . t . ~ } \ell )$ . Assume $Y \bot Z _ { 1 : t - 1 } | \left( X , Z _ { t } \right)$ as in (5). Define $\begin{array} { r } { \mathcal { R } _ { c a s c } ^ { * } { : = } \operatorname* { m i n } _ { \pi \in \Pi _ { c a s c } } \mathcal { R } ( \pi ) } \end{array}$ and $\begin{array} { r } { \mathcal { R } _ { c a s c } ^ { h i n g e * } { : = } \operatorname* { m i n } _ { \tilde { \pi } \in \widetilde { \Pi } _ { c a s c } } \mathcal { R } _ { c a s c } ^ { h i n g e } ( \tilde { \pi } ) } \end{array}$ . Then, for any $\tilde { \pi } \in \tilde { \Pi } _ { c a s c }$ 2

$$
\mathcal { R } \big ( \varphi ( \tilde { \pi } ) \big ) - \mathcal { R } _ { c a s c } ^ { * } \leq 2 ^ { T - 1 } ( K - 1 ) \Big ( \mathcal { R } _ { c a s c } ^ { h i n g e } ( \tilde { \pi } ) - \mathcal { R } _ { c a s c } ^ { h i n g e * } \Big ) .\tag{20}
$$

This result establishes the Bayes consistency of $\ell _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } }$ . We detail its proof in Appendix C. Importantly, the calibration constant $2 ^ { T - 1 } ( K - 1 )$ cannot be improved: Proposition C.8 gives a matching lower bound by constructing bad instances for which the smallest valid calibration constant approaches $2 ^ { T - 1 } ( K - 1 )$ . In these instances, every soft deferral decision is a tie between exit and defer, $\mathrm { i . e . , } \tilde { d } _ { t } \equiv 0 . 5$ for all $t \in [ T - 1 ]$ . When the soft deferral decisions are more decisive, the constant can be substantially smaller: Proposition C.10 shows that for $m \in [ 0 , 0 . 5 ]$ , if $| \tilde { d } _ { t } ( H _ { t } ) - 0 . 5 | \geq m$ almost surely for all $t \in [ T - 1 ]$ , the constant reduces to $\begin{array} { r } { \big ( \frac { 2 } { 1 + 2 m } \big ) ^ { T - 1 } ( K - \overline { { 1 } } ) } \end{array}$

Sections 5.1 and 5.3 present experiments that validate Theorems 3.1–3.2, and demonstrate their practical utility.

## 4 Improving Routing with Unlabeled Data and PPI

While Section 3 establishes the Bayes consistency of the proposed hinge losses as the sample size $n \to \infty$ , practical training operates in the finite-sample regime. Empirical evidence from prior work [Shazeer et al., 2017, Kubaty et al., 2025], as well as our own experiments (see Fig. 7 in Appendix D.3), shows that the classifiers $f _ { 1 : T }$ converge faster than the router r. Consequently, r can be suboptimal if labeled data is limited, while $f _ { 1 : T }$ may be more reliable. This motivates post-training methods that improve the routing decisions eficiently without retraining, by leveraging unlabeled data.

In this regime, the prediction-powered inference (PPI) framework [Angelopoulos et al., 2023] provides a natural tool. PPI uses predictions from trained models on abundant unlabeled data to construct confidence intervals (CI) for quantities otherwise hard to estimate reliably from labeled data alone. In our setting, the trained classifiers can provide these predictions, which we use to refine our trained router.

Accordingly, we consider a setting with access to a large unlabeled dataset $\{ X ^ { ( i ) } , Z _ { 1 : T } ^ { ( i ) } \} _ { i = 1 } ^ { N } \stackrel { \mathrm { i . i . d . } } { \sim } P$ in addition to a held-out labeled dataset $\{ X ^ { ( i ) } , Z _ { 1 : T } ^ { ( i ) } , Y ^ { ( i ) } \} _ { i = 1 } ^ { l }$ where $N \gg l ,$ both separate from the data used to train the multi-model classifier. We show how PPI can be used post-training to improve a flat router, and note that the proposed method may be extended to the cascade setting via the recent e-values framework [Ramdas and Wang, 2025, Grünwald et al., 2020, Vovk and Wang, 2021, Csillag et al., 2025, Kilian et al., 2025]. For notational simplicity, we present our construction and high-probability guarantees pointwise for a fixed input feature $x \in \mathcal { X }$ . In practice, however, test inputs $( \mathrm { e . g . }$ , prompts or images) rarely coincide with training samples. We therefore implement the method using binning, which groups similar inputs to ensure coverage of previously unseen samples. The pointwise guarantees in the sequel should be interpreted at the bin level.

Oracle router $r _ { \mathbf { f l a t } } ^ { \dagger }$ . Let $\pi ^ { \dagger } = ( f _ { 1 : T } ^ { \dagger } , r _ { \mathrm { f l a t } } )$ denote the multi-model classifier after training. Given $f _ { 1 : T } ^ { \dagger }$ , the Bayes-optimal routing decision for feature x is the one that minimizes the conditional 0-1-c risk given $X = x .$ , taking the form

$$
r _ { \mathrm { f l a t } } ^ { \dag } ( x ) \in \arg \operatorname* { m i n } _ { t \in [ T ] } \mathcal { R } _ { t } ( x ; f _ { t } ^ { \dag } ) ,\tag{21}
$$

where, for any classifier $f _ { t } : \mathcal { X } \times \mathcal { Z } _ { t }  [ K ]$

$$
\mathscr { R } _ { t } ( x ; f _ { t } ) : = \mathbb { E } \left[ \mathbb { 1 } \left\{ f _ { t } ( x , Z _ { t } ) \neq Y \right\} \ | \ X = x \right] + c _ { t } .\tag{22}
$$

Since $\mathcal { R } _ { t } ( x ; f _ { t } ^ { \dagger } )$ is an oracle population quantity (an expectation over the true data distribution), the routing decision $r _ { \mathrm { f l a t } } ^ { \dagger } ( x )$ is not computable, but provides a natural benchmark. Our goal is not to estimate the oracle directly, but rather to certify whether the trained routing decision can be safely improved using unlabeled data to approach the oracle.

PPI-corrected router $r _ { \mathbf { f l a t } } ^ { \mathbf { P P } }$ . To this end, we compare each alternative routing decision to the trained baseline by comparing the diference in conditional risks. Let $\delta _ { t } ( x ) : = \mathcal { R } _ { r _ { \mathrm { H a t } } ( x ) } ( x ; f _ { r _ { \mathrm { H a t } } ( x ) } ^ { \dagger } ) -$ $\mathcal { R } _ { t } ( x ; f _ { t } ^ { \dagger } )$ for $t \in [ T ] \setminus \{ r _ { \mathrm { f i a t } } ( x ) \}$ . We leverage PPI to construct the 95% CI around $\delta _ { t } ( x )$ in the form:

$$
\begin{array} { r } { \mathbb { P } \left( \left| \delta _ { t } ( x ) - \delta _ { t } ^ { \mathrm { P P } } ( x ) \right| \geq w _ { t , N _ { b } , l _ { b } } ( x ) \right) \leq \frac { 0 . 0 5 } { T - 1 } , } \end{array}\tag{23}
$$

where $\delta _ { t } ^ { \mathrm { P P } } ( x )$ and $w _ { t , N _ { b } , l _ { b } } ( x )$ have explicit expressions which we detail in Appendix E.1, $N _ { b }$ and $l _ { b }$ denote the numbers of unlabeled and labeled points in the bin, and the probability is over the randomness of both the labeled and unlabeled data. Based on these CIs, we identify a set of routing destinations $s ( x )$ that are certified, with probability $1 - { \frac { 0 . 0 5 } { T - 1 } }$ , to be no worse than the baseline decision $r _ { \mathrm { { f l a t } } } ( x )$ . If $s ( x )$ is nonempty, we propose the PPI-corrected routing decision $r _ { \mathrm { f l a t } } ^ { \mathrm { P P } } ( x )$ by selecting the alternative with the largest certified improvement; otherwise, we retain the baseline.

Formally, we define

$$
\begin{array} { r l } & { \mathcal { S } ( x ) : = \big \{ t \in [ T ] \setminus \{ r _ { \mathrm { f l a t } } ( x ) \} : \delta _ { t } ^ { \mathrm { P P } } ( x ) - w _ { t , N _ { b } , l _ { b } } ( x ) \geq 0 \big \} , } \\ & { r _ { \mathrm { f l a t } } ^ { \mathrm { P P } } ( x ) : = \left\{ \begin{array} { l l } { \underset { t \in \mathcal { S } ( x ) } { \arg \operatorname* { m a x } } \delta _ { t } ^ { \mathrm { P P } } ( x ) - w _ { t , N _ { b } , l _ { b } } ( x ) } & { \mathrm { i f } \mathcal { S } ( x ) \neq \emptyset , } \\ { \hfill } \\ { r _ { \mathrm { f l a t } } ( x ) } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{24}
$$

Then $r _ { \mathrm { f l a t } } ^ { \mathrm { P P } }$ is guaranteed to be never worse than the baseline $r _ { \mathrm { { f l a t } } }$ in the sense of (26) below in Theorem 4.1. Under an additional mild separation assumption, where the oracle decision is unique while the gap between the oracle and the next runner-up is strictly positive, i.e.,

$$
\gamma ( x ) : = \delta _ { r _ { \mathrm { f l a t } } ^ { \dagger } ( x ) } ( x ) - \operatorname* { m a x } _ { t ^ { \prime } \in [ T ] \backslash \{ r _ { \mathrm { f l a t } } ^ { \dagger } ( x ) \} } \delta _ { t ^ { \prime } } ( x ) > 0 ,\tag{25}
$$

we can obtain the stronger guarantee in (27). The CI half-width $w _ { t , N _ { b } , l _ { b } } ( x )$ decreases with the accuracy of the classifier used to predict the labels of the N unlabeled samples, motivating our use of the most powerful classifier $f _ { T } ^ { \dagger }$ among $f _ { 1 : T } ^ { \dagger }$ in our experiments in Section 5.2. If the imputing classifier $f _ { T } ^ { \dagger }$ is accurate, $w _ { t , N _ { b } , l _ { b } } ( x )$ decreases with N roughly as $1 / \sqrt { N }$ , so for large enough N, the CI for $\delta _ { r _ { \mathrm { f l a t } } ^ { \dagger } ( x ) } ( x )$ becomes narrow enough to separate it from the runner-up, guaranteeing the recovery of the oracle decision: $r _ { \mathrm { f l a t } } ^ { \mathrm { P P } } ( x ) = r _ { \mathrm { f l a t } } ^ { \dagger } ( x )$

Theorem 4.1 (Certification for PPI-corrected router). For a given $x \in \mathcal { X }$ and a fixed trained multi-model classifier $\pi ^ { \dagger } = ( f _ { 1 : T } ^ { \dagger } , r _ { \mathcal { \hbar } a t } )$ , let $r _ { f l a t } ^ { \dagger } ( x )$ and $r _ { f l a t } ^ { P P } ( x )$ be as defined in (21) and (24), then

$$
\mathbb { P } \Big ( \mathcal { R } _ { r _ { \mathit { f l a t } } ^ { P P } ( x ) } \big ( x ; f _ { r _ { \mathit { f l a t } } ^ { P } ( x ) } ^ { \dagger } \big ) > \mathcal { R } _ { r _ { \mathit { f l a t } } ( x ) } \big ( x ; f _ { r _ { \mathit { f l a t } } ( x ) } ^ { \dagger } \big ) \cap \mathcal { S } ( x ) \ne \emptyset \Big ) \le 0 . 0 5 .\tag{26}
$$

Moreover, with $\gamma ( x )$ as defined in (25), then

$$
\mathbb { P } \Big ( r _ { \# a t } ^ { P P } ( x ) \neq r _ { \# a t } ^ { \dagger } ( x ) \cap S ( x ) \neq \emptyset \cap w _ { r _ { \# a t } ^ { \dagger } ( x ) , N _ { b } , l _ { b } } ( x ) < \gamma ( x ) / 2 \Big ) \leq 0 . 0 5 .\tag{27}
$$

The probabilities in (26) and (27) are over the randomness of both the labeled and unlabeled data. The proof is a combination of (23) with a union bound. For suficiently accurate classifiers $f _ { 1 : T } ^ { \dagger }$ and suficiently large N, the CIs become suficiently tight that $\boldsymbol { S } ( \boldsymbol { x } ) \neq \boldsymbol { \emptyset }$ with high probability, in which case (26) and (27) efectively reduce to stronger marginal guarantees. We experimentally illustrate the PPI corrections in Section 5.2.

## 5 Experiments

## 5.1 Validating the Bayes Consistency of Surrogate Losses on Synthetic Data

To validate Theorems 3.1 and 3.2 in a controlled setting, we study a synthetic classification task with known ground-truth conditional distributions. We compute the Bayes-optimal solutions (as characterized in Lemmas B.1 and C.4) under both the flat and cascade routing models and verify that minimizing our proposed surrogate losses recovers these solutions.

Under both models, we consider $T { = } 3$ classifiers for a $K { = } 3$ class problem. The data consists of a base feature $X \in \mathbb { R }$ and two additional features $Z _ { 2 } , Z _ { 3 } \in \mathbb { R }$ all sampled independently from Uniform[−1, 1], and a label $Y \in [ K ]$ generated according to:

$$
Y \mid ( X = x , Z _ { t } = z _ { t } ) \sim \mathrm { C a t e g o r i c a l } \big ( \theta ( x , z _ { t } ) \big ) ,\tag{28}
$$

![](images/616e5ca35ed486cdf8504a6b2028299c141e2c3312ee6d0e3a1c3090edf4a74e.jpg)  
Figure 2: (Top) Monte-Carlo estimate of the Bayes-optimal conditional risk and Bayes-optimal routing decisions for the flat router $r ^ { * }$ (left) and for the first stage of the cascade router $d _ { 1 } ^ { * } ~ \mathrm { ( r i g h t ) }$ (Bottom) Learned routing decisions on the test set, where points are colored according to the routing probabilities learned by the model.

where the full specification of $\pmb \theta$ is provided in Appendix D.1. We set the cost of $f _ { 1 }$ to $\kappa _ { 1 } = 0$ and assign the same cost $\kappa _ { 2 } = \kappa _ { 3 } = 0 . 0 5$ to $f _ { 2 }$ and $f _ { 3 } ,$ yielding cost vector $c = [ 0 , 0 . 0 5 , 0 . 0 5 ] ^ { \top }$ in the flat setting and $c = [ 0 , 0 . 0 5 , 0 . 1 ] ^ { \top }$ in the cascade setting.

The Bayes-optimal routing decisions in both settings are governed by the conditional risk $\mathcal { R } _ { t } ( x ; f _ { t } ^ { * } )$ in (22): see Eq. (31) for the flat and (82) for the cascade setting. We choose $\pmb \theta$ so that this conditional risk is minimized at diferent exit stage t over diferent regions of $x ;$ the solid lines in the top panels of Fig. 2 show its Monte-Carlo estimate denoted by $\hat { \mathcal { R } } _ { t } ( x ; f _ { t } ^ { * } )$ . This induces nontrivial Bayes-optimal routing boundaries, shown by the shaded regions in the top panels. In the flat setting (top left panel), the Bayes-optimal router $r ^ { * }$ selects the classifier with the lowest conditional risk, following (31). In the cascade setting (top right panel), the stage-1 Bayes-optimal deferral rule $d _ { 1 } ^ { * }$ does not align with the lower envelope of the risk curves, since it must compare the stage-1 conditional risk $\hat { \mathcal { R } } _ { 1 } ( x ; f _ { 1 } ^ { * } )$ against the expected cost of optimally continuing the cascade, following (82). Consequently, the left orange-shaded region in the cascade figure is narrower than the green-shaded region in the flat figure, indicating fewer samples are routed to $f _ { 3 }$ in the cascade, due to the accumulated cost of deferral.

Having illustrated the Bayes-optimal solutions, we now train flat and cascade multi-model classifiers, and compare the learned functions with their Bayes-optimal counterparts. Each classifier $f _ { 1 } ( x ) , f _ { 2 } ( x , z _ { 2 } ) , f _ { 3 } ( x , z _ { 3 } )$ is implemented as a three-layer multilayer perceptron. For simplicity, the synthetic setup takes $\mathcal { Z } _ { 1 } = \emptyset$ , so $f _ { 1 }$ and $d _ { 1 }$ depend on x alone. Routers are implemented as three-layer networks: a multiclass routing network $r : \mathcal { X }  \{ 1 , 2 , 3 \}$ in the flat setting, and two sequential binary routing networks $d _ { 1 } : \mathcal { X } \to \{ 0 , 1 \}$ and $d _ { 2 } : \mathcal { X } \times \mathcal { Z } _ { 2 } \to \{ 0 , 1 \}$ in the cascade setting. Models are trained end-to-end with Adam using our proposed surrogate losses, i.e., (16) for flat routers and (18) for cascades. Appendix D.1 provides additional architectural and optimization details.

The bottom panels of Fig. 2 visualize the learned routing boundaries for both the flat and cascade settings (via the first deferral function $\tilde { d } _ { 1 } )$ , where each test point is colored according to the routing probabilities learned by $\tilde { r }$ or $\tilde { d } _ { 1 }$ . In both cases, the learned regions closely match the Bayes-optimal regions in the top panels, confirming our Bayes consistency results from Theorems 3.1 and 3.2.

Complementing the routing visualizations in Fig. 2, Fig. 6 in Appendix D.2 directly compares the empirical 0-1-c risk over minibatches against the empirical Bayes-optimal 0-1-c risk for both the flat and cascade settings, under two cost configurations. In all cases, minimizing the proposed hinge losses attains the Bayes-optimal risk on the test set, further validating our theory.

![](images/fa4e3f0d396b6caa343f7ca7ac742351b112784a1c8f94b0156ae6dcf1e5706f.jpg)  
Figure 3: Comparison of the oracle router $r _ { \mathrm { f l a t } } ^ { \dagger } .$ , the router $r _ { \mathrm { { f l a t } } }$ trained on labeled data, and the PPI-corrected router $r _ { \mathrm { f l a t } } ^ { \mathrm { P P } }$ leveraging unlabeled data. The fixes and errors introduced by PPI are computed with respect to $r _ { \mathrm { f l a t } } ^ { \dagger } .$

## 5.2 Demonstrating Post-Training PPI-Based Routing Improvement on Synthetic Data

Fig. 7 in Appendix D.3 compares the training dynamics of the classifiers and router by plotting their training accuracies over minibatches. The classifiers converge quickly and plateau early, while the router requires much more labeled data to converge. When labeled data is scarce, we now demonstrate how the post-training PPI procedure of Section 4 leverages unlabeled data to improve routing decisions at test time.

We use the same synthetic data generating process from Section 5.1 with 2000 labeled samples $( n = 1 0 0 0$ training samples and l = 1000 PPI labeled samples) and N = 20, 000 unlabeled samples, and partition the input space X using K-means clustering (binning) with 40 bins to account for unseen data, as described in Section 4. We consider three classifiers and a flat router, set up identically to those in Section 5.1. The classifiers and router are trained on the training labeled samples only. Post-training, routing decisions at test time are corrected via the PPI procedure of Section 4.

Fig. 3 compares three routers: the oracle router $r _ { \mathrm { f l a t } } ^ { \dagger }$ , the trained router r , and the PPI- $r _ { \mathrm { { f l a t } } }$ corrected router $r _ { \mathrm { f l a t } } ^ { \mathrm { P P } }$ . While the trained router deviates noticeably from the oracle due to limited labeled data, PPI correction substantially improves it, approaching $r _ { \mathrm { f l a t } } ^ { \dagger }$ and fixing 10.65% routing errors at the cost of 1.45% mistakes. Note that Theorem 4.1 guarantees improvement towards the oracle router $r _ { \mathrm { f l a t } } ^ { \dagger } ,$ not the Bayes-optimal router $r _ { \mathrm { f l a t } } ^ { \ast }$ . However, even if the trained router is poor, the trained classifiers can be closer to their Bayes-optimal counterparts, in which case the oracle router would be near-optimal $( r _ { \mathrm { f l a t } } ^ { \dagger } \approx r _ { \mathrm { f l a t } } ^ { \ast } )$ . We illustrate this in Appendix E.2, where we additionally include the Bayes-optimal router and vary the amount of labeled data to show how this afects the behavior of the oracle, the trained router and the PPI correction. Finally, following the theory, the experiment above keeps the training data and the labeled PPI data separate. However, setting labeled data aside from training can be wasteful, so in Appendix E.2 we also show that reusing the training data as the labeled PPI data can still ofer improvements, even if it violates the independence assumption of Theorem 4.1.

![](images/f4fab436ac9760ed1d05aa85e44977b24bcb167e9c1705477a65fb25226f5b1c.jpg)  
Figure 4: Classifier accuracy and routing fractions on GSM8K across three cost configurations (costs increasing from left panel to right panel).

## 5.3 Large Language Models Experiment

To showcase the practical efectiveness of our surrogate losses, we conduct experiments in the $T = 3$ setting with LLMs from two model families LLaMA 3 1B, 70B, and 405B [Dubey et al., 2024] and Qwen 2.5 1.5B, 32B, 72B [Yang et al., 2025]. In this section we show flat router results for LLaMA models. For cascade router results, and Qwen model experiments with flat router and cascade router setups see Appendix G.

Datasets. We use two standard reasoning benchmarks from the literature: CommonsenseQA [Talmor et al., 2019] (commonsense logical reasoning) and GSM8K [Cobbe et al., 2021] (grade-school math reasoning). See Appendix F.1 for more details on the datasets. For all experiments we use a 80-10-10 train-validation-test split.

Setup. We form each input feature x by embedding the natural language prompt into a vector. Additional features $z _ { t }$ are obtained by embedding the LLM outputs in a similar fashion and learn light classifier modules $f _ { t }$ on top of them. We provide experimental and implementation details in Appendix F.2. Appendix G reports cascade-router results with the LLaMA models and flat-router results with the Qwen models.

Results. Fig. 4 presents a flat router experiment with the three LLaMA models. We present three panels, one per cost configuration (in increasing order of costs from left to right with $c = [ 0 , 0 . 0 1 , 0 . 0 2 ]$ [0, 0.05, 0.07], [0, 0.1, 0.2]). For each configuration, we display statistics through box-and-whisker plots computed via bootstrap sampling over 10 trials and the median (horizontal line).

The orange box represents the performance of each model on the entire dataset (independent of the routing decision), the green box represents the performance of each model on the samples that are routed to it, and the blue box (bottom row) represents the fraction of samples that are routed to each model. In configuration 1 (left panel), the gap between the orange and green boxes shows that the router successfully identifies easier samples on which LLaMA 1B is more accurate than average. As costs increase across the three configurations, a larger fraction of questions is routed to the cheapest model, causing this model’s accuracy on the routed subset (green) to decrease as the model receives a larger fraction of harder questions. Meanwhile, LLaMA 70B and 405B achieve accuracies in the nineties on routed samples. Thus, joint classifier-router training under our proposed loss enables cost savings by finding samples where the 1B is accurate.

In Appendices G.1–G.4, we provide additional experiments to demonstrate the utility of our proposed training losses in three aspects: (i) they induce consistent cost-accuracy tradeof behavior, similar to that in Fig. 4, across flat and cascade settings, as well as other datasets and model families (G.1); (ii) they attain the lowest empirical 0-1-c risk against popular separate-training and joint-training baselines (G.2 and G.3); and (iii) they scale stably with the number of classifiers T (G.4).

## 6 Conclusion

We present a principled framework for jointly training classifiers and routers in cost-sensitive multimodel classification. We propose hinge-based surrogate losses and establish their Bayes consistency with respect to the cost-sensitive zero-one loss in both single-stage and cascade routing models. To address label scarcity, we design a post-training PPI procedure that leverages unlabeled data to refine a trained single-stage router. Experiments on synthetic data and LLM benchmarks validate our theory and the utility of the hinge losses.

Our framework has some limitations. In practice, full joint training can be computationally demanding, and the hinge loss can be harder to optimize than a common log-based version. Moreover, our analysis assumes that the distribution of the additional features $Z _ { t }$ is fixed during training, which excludes settings where they are produced by models that are fine-tuned jointly with the classifiers and router. Appendix I discusses these, along with further limitations in more detail.

## Acknowledgements and Disclosure of Funding

S. X. Liu and P. Rebeschini were supported by UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number EP/Y028333/1]. F. Regol, A. Valkanas and M. Coates were supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) [reference number 260250], and the Fonds de recherche du Québec (FRQNT) [reference number 324302]. Cette recherche a été partiellement financée par le Conseil de recherches en sciences naturelles et en génie du Canada (CRSNG) [numéro de référence 260250] et le Fonds de recherche du Québec (FRQNT) [numéro de référence 324302].

## References

Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnic. Prediction-Powered Inference. Science, 382(6671):669–674, 2023.

Pranjal Awasthi, Anqi Mao, Mehryar Mohri, and Yutao Zhong. H-Consistency Bounds for Surrogate Loss Minimizers. In Proc. Int. Conf. Mach. Learn. (ICML), 2022.

Peter L Bartlett, Michael I Jordan, and Jon D McAulife. Convexity, Classification, and Risk Bounds. J. Am. Stat. Assoc., 101(473):138–156, 2006.

Dimitri P. Bertsekas. Dynamic Programming and Optimal Control. 4th edition, 2017.

Yuzhou Cao, Tianchi Cai, Lei Feng, Lihong Gu, Jinjie Gu, Bo An, Gang Niu, and Masashi Sugiyama. Generalizing Consistent Multi-Class Classification with Rejection to Be Compatible with Arbitrary Losses. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2022.

Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. Trans. Mach. Learn. Res. (TMLR), 2024a.

Shuhao Chen, Weisen Jiang, Baijiong Lin, James Kwok, and Yu Zhang. RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2024b.

Yanxi Chen, Xuchen Pan, Yaliang Li, Bolin Ding, and Jingren Zhou. EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models with 3D Parallelism. In Proc. Int. Conf. Mach. Learn. (ICML), 2024c.

C. Chow. On Optimum Recognition Error and Reject Tradeof. IEEE Trans. Inf. Theory, 16(1): 41–46, 1970.

C. K. Chow. An Optimum Character Recognition System Using Decision Functions. IRE Trans. Electron. Comput., EC-6(4):247–254, 1957.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training Verifiers to Solve Math Word Problems, 2021. arXiv:2110.14168.

Corinna Cortes, Giulia DeSalvo, and Mehryar Mohri. Learning with Rejection. In Proc. Int. Conf. Algorithmic Learn. Theory, 2016.

Daniel Csillag, Claudio Jose Struchiner, and Guilherme Tegoni Goedert. Prediction-Powered E-Values. In Proc. Int. Conf. Mach. Learn. (ICML), 2025.

Fnu Devvrit, Sneha Kudugunta, Aditya Kusupati, Tim Dettmers, Kaifeng Chen, Inderjit Dhillon, Yulia Tsvetkov, Hannaneh Hajishirzi, Sham Kakade, Ali Farhadi, and Prateek Jain. MatFormer: Nested Transformer for Elastic Inference. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2023.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, and et al. The Llama 3 Herd of Models, 2024. arXiv:2407.21783.

Ran El-Yaniv and Yair Wiener. On the Foundations of Noise-Free Selective Classification. J. Mach. Learn. Res., 11(53):1605–1641, 2010.

Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael Auli. Depth-Adaptive Transformer. In Proc. Int. Conf. Learn. Representations (ICLR), 2020.

Jessie Finocchiaro, Rafael M. Frongillo, and Bo Waggoner. An Embedding Framework for the Design and Analysis of Consistent Polyhedral Surrogates. J. Mach. Learn. Res., 25(63):1–60, 2024.

Marc Finzi, Shikai Qiu, Yiding Jiang, Pavel Izmailov, J. Zico Kolter, and Andrew Gordon Wilson. From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence, 2026. arXiv:2601.03220.

Rafael Frongillo and Bo Waggoner. Surrogate Regret Bounds for Polyhedral Losses. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2021.

Peter Grünwald, Rianne de Heide, and Wouter M Koolen. Safe Testing. In Proc. Inf. Theory Appl. Workshop (ITA), 2020.

Y. Han, G. Huang, S. Song, L. Yang, H. Wang, and Y. Wang. Dynamic Neural Networks: A Survey. IEEE Trans. Pattern Anal. Mach. Intell., 44(11):7436–7456, 2022a.

Yizeng Han, Yifan Pu, Zihang Lai, Chaofei Wang, Shiji Song, Junfen Cao, Wenhui Huang, Chao Deng, and Gao Huang. Learning to Weight Samples for Dynamic Early-Exiting Networks. In Proc. Eur. Conf. Comput. Vis. (ECCV), 2022b.

Kilian Hendrickx, Lorenzo Perini, Dries Van der Plas, Wannes Meert, and Jesse Davis. Machine Learning with a Reject Option: A Survey. Mach. Learn., 113(5):3073–3110, 2024.

Radu Herbei and Marten H. Wegkamp. Classification with Reject Option. Can. J. Stat., 34(4): 709–721, 2006.

Wittawat Jitkrittum, Neha Gupta, Aditya K Menon, Harikrishna Narasimhan, Ankit Rawat, and Sanjiv Kumar. When Does Confidence-Based Cascade Deferral Sufice? In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2023.

Vijay Keswani, Matthew Lease, and Krishnaram Kenthapadi. Towards Unbiased and Accurate Deferral to Multiple Experts. In Proc. AAAI/ACM Conf. AI Ethics Soc. (AIES), 2021.

Valentin Kilian, Stefano Cortinovis, and Francois Caron. Anytime-Valid, Bayes-Assisted, Prediction-Powered Inference. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2025.

Piotr Kubaty, Bartosz Wójcik, Bartłomiej Tomasz Krzepkowski, Monika Michaluk, Tomasz Trzcinski, Jary Pomponi, and Kamil Adamczewski. How to Train Your Multi-Exit Model? Analyzing the Impact of Training Strategies. In Proc. Int. Conf. Mach. Learn. (ICML), 2025.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast Inference from Transformers via Speculative Decoding. In Proc. Int. Conf. Mach. Learn. (ICML), 2023.

Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Ré, and Beidi Chen. Deja Vu: Contextual Sparsity for Eficient LLMs at Inference-Time. In Proc. Int. Conf. Mach. Learn. (ICML), 2023.

Phil Long and Rocco Servedio. Consistency versus Realizable H-Consistency for Multiclass Classification. In Proc. Int. Conf. Mach. Learn. (ICML), 2013.

David Madras, Toniann Pitassi, and Richard Zemel. Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2018.

Anqi Mao, Christopher Mohri, Mehryar Mohri, and Yutao Zhong. Two-Stage Learning to Defer with Multiple Experts. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2023.

Anqi Mao, Mehryar Mohri, and Yutao Zhong. Realizable H-Consistent and Bayes-Consistent Loss Functions for Learning to Defer. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2024a.

Anqi Mao, Mehryar Mohri, and Yutao Zhong. Principled Approaches for Learning to Defer with Multiple Experts. In Proc. Artif. Intell. and Image Analysis, 2024b.

Anqi Mao, Mehryar Mohri, and Yutao Zhong. Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees for Learning to Defer. In Proc. Int. Conf. Mach. Learn. (ICML), 2025.

Hussein Mozannar and David Sontag. Consistent Estimators for Learning to Defer to an Expert. In Proc. Int. Conf. Mach. Learn. (ICML), 2020.

Hussein Mozannar, Hunter Lang, Dennis Wei, Prasanna Sattigeri, Subhro Das, and David Sontag. Who Should Predict? Exact Algorithms for Learning to Defer to Humans. In Proc. Int. Conf. Artif. Intell. Stat. (AISTATS), 2023.

Niels Mündler, Jingxuan He, Slobodan Jenko, and Martin Vechev. Self-Contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation. In Proc. Int. Conf. Learn. Representations (ICLR), 2024.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. RouteLLM: Learning to Route LLMs from Preference Data. In Proc. Int. Conf. Learn. Representations (ICLR), 2025.

Andrea Pugnana, Giovanni De Toni, Cesare Barbera, Roberto Pellungrini, Bruno Lepri, and Andrea Passerini. To Ask or Not to Ask: Learning to Require Human Feedback, 2025. arXiv:2510.08314.

Aaditya Ramdas and Ruodu Wang. Hypothesis Testing with E-Values. Found. Trends Stat., 1(1-2): 1–390, 2025.

Florence Regol, Joud Chataoui, and Mark Coates. Jointly-Learned Exit and Inference for a Dynamic Neural Network. In Proc. Int. Conf. Learn. Representations (ICLR), 2024.

Florence Regol, Joseph Cotnareanu, Theodore Glavas, and Mark Coates. Is the Acquisition Worth the Cost? Surrogate Losses for Consistent Two-Stage Classifiers. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2025.

Pavel Rumiantsev, Soumyasundar Pal, Yingxue Zhang, and Mark Coates. FEval-TTC: Fair Evaluation Protocol for Test-Time Compute. In NeurIPS Workshop on Evaluating the Evolving LLM Lifecycle, 2025.

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Q. Tran, Yi Tay, and Donald Metzler. Confident Adaptive Language Modeling. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2022.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In Proc. Int. Conf. Learn. Representations (ICLR), 2017.

Albert N Shiryaev. Optimal Stopping Rules. Springer, 2008.

Ingo Steinwart. How to Compare Diferent Loss Functions and Their Risks. Constr. Approx., 26(2): 225–287, Aug 2007.

Manxi Sun, Wei Liu, Jian Luan, Pengzhi Gao, and Bin Wang. Mixture of Diverse Size Experts. In Proc. Conf. Empir. Methods Nat. Lang. Process. (EMNLP), Industry Track, 2024.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge. In Proc. Conf. North Amer. Chapter Assoc. Comput. Linguist. (NAACL), 2019.

Bernadetta Tarigan and Sara A. van de Geer. A Moment Bound for Multi-Hinge Classifiers. J. Mach. Learn. Res., 9(71):2171–2185, 2008.

Ambuj Tewari and Peter L. Bartlett. On the Consistency of Multiclass Classification Methods. J. Mach. Learn. Res., 8(36):1007–1025, 2007.

Antonios Valkanas, Boris N. Oreshkin, and Mark Coates. MODL: Multilearner Online Deep Learning. In Proc. Int. Conf. Artif. Intell. Stat. (AISTATS), 2025a.

Antonios Valkanas, Soumyasundar Pal, Pavel Rumiantsev, Yingxue Zhang, and Mark Coates. C3PO: Optimized Large Language Model Cascades with Probabilistic Cost Constraints for Reasoning. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2025b.

Rajeev Verma and Eric T. Nalisnick. Calibrated Learning to Defer with One-vs-All Classifiers. In Proc. Int. Conf. Mach. Learn. (ICML), 2022.

Rajeev Verma, Daniel Barrejón, and Eric Nalisnick. Learning to Defer to Multiple Experts: Consistent Surrogate Losses, Confidence Calibration, and Conformal Ensembles. In Proc. Int. Conf. Artif. Intell. Stat. (AISTATS), 2023.

Vladimir Vovk and Ruodu Wang. E-Values: Calibration, Combination and Applications. Ann. Stat., 49(3):1736–1754, 2021.

An Wang, Xingwu Sun, Ruobing Xie, Shuaipeng Li, Jiaqi Zhu, Zhen Yang, Pinxue Zhao, Weidong Han, Zhanhui Kang, Di Wang, Naoaki Okazaki, and Cheng-zhong Xu. HMoE: Heterogeneous Mixture of Experts for Language Modeling. In Proc. Conf. Empir. Methods Nat. Lang. Process. (EMNLP), 2025.

Yair Wiener and Ran El-Yaniv. Agnostic Selective Classification. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2011.

Bryan Wilder, Eric Horvitz, and Ece Kamar. Learning to Complement Humans. In Proc. Int. Joint Conf. Artif. Intell. (IJCAI), 2021.

Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. Unlocking Eficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding. In Proc. Findings Assoc. Comput. Linguist. (ACL), 2024.

Ji Xin, Raphael Tang, Yaoliang Yu, and Jimmy Lin. BERxiT: Early Exiting for BERT with Better Fine-Tuning and Extension to Regression. In Proc. Conf. Eur. Chapter Assoc. Comput. Linguist. (EACL), 2021.

Yilun Xu, Shengjia Zhao, Jiaming Song, Russell Stewart, and Stefano Ermon. A Theory of Usable Information Under Computational Constraints. In Proc. Int. Conf. Learn. Representations (ICLR), 2020.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Junyang Lin, Jingren Zhou, et al. Qwen2.5 technical report. Technical report, Alibaba Group, 2025.

Haichao Yu, Haoxiang Li, Gang Hua, Gao Huang, and Humphrey Shi. Boosted Dynamic Neural Networks. In Proc. AAAI Conf. Artif. Intell. (AAAI), 2023.

Ziqian Zeng, Yihuai Hong, Hongliang Dai, Huiping Zhuang, and Cen Chen. ConsistentEE: A Consistent and Hardness-Guided Early Exiting Method for Accelerating Language Model Inference. In Proc. AAAI Conf. Artif. Intell. (AAAI), 2024.

Tong Zhang. Statistical behavior and consistency of classification methods based on convex risk minimization. Ann. Stat., 32(1):56–85, 2004.

Xuechen Zhang, Zijian Huang, Ege Onur Taga, Carlee Joe-Wong, Samet Oymak, and Jiasi Chen. Eficient Contextual LLM Cascades through Budget-Constrained Policy Learning. In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2024.

## Appendix Contents

A Extended Literature Review 20   
A.1 Empirical Works 20   
A.2 Theoretical Works 22   
B Proofs for the Flat Router Model 24   
B.1 Bayes-Optimal Solutions . 24   
B.2 Relationship Between 0-1-c and Hinge Conditional Risks 26   
B.3 Proof of Theorem 3.1 . 29   
C Proofs for the Cascade Model 33   
C.1 Connection of Risks to Expected Value Functions 33   
C.2 Bayes-Optimal Solutions 36   
C.3 Connections Between 0-1-c and Hinge Quantities 40   
C.4 Proof of Theorem 3.2 . 42   
C.5 Lower Bound 46   
C.6 An Upper Bound in Favorable Instances 50   
D Synthetic Experiments 53   
D.1 Details for the Synthetic Experiments 53   
D.2 Empirical Risk Minimization in Synthetic Dataset for Various Costs 54   
D.3 Accuracy of Classifiers and Router in Synthetic Dataset 54   
E Post-Training PPI Correction 55   
E.1 Constructing Confidence Intervals Using Unlabeled Data with PPI 55   
E.2 Additional PPI Experiments: Impact of Labeled Data Budget on Performance . 55   
F Details for LLM Experiments 59   
F.1 Datasets . 59   
F.2 Model and Training Details 59   
G Additional LLM Experiments 62   
G.1 LLM Experiments with Additional Models and Datasets . 62   
G.2 Comparison to Supervised LLM Cascade Baselines 64   
G.3 Comparison to Adaptive Inference Baselines 65   
G.4 Scaling with the Number of Classifiers 67   
H Additional Feature Variables to Represent Varying Model Capacity 68   
I Limitations and Future Work 69

## A Extended Literature Review

We study multi-model adaptive inference for classification by casting it as a cost-sensitive classification problem with the classifiers carrying varying costs. We propose losses for jointly training the classifiers and their associated routing model. In this section, we survey related work on multi-model classification, both from an empirical and a theoretical perspective.

Section A.1 reviews empirical approaches, which prioritize inference cost savings through the design of architectures and loss functions that tend to lack statistical guarantees. These works can be further categorized into general approaches applicable to any multi-classifier architecture, and domain-specific ones. Since our work is mainly motivated by LLM-related applications, we review LLM domain-specific approaches. Section A.2 covers the theoretical works, which we organize into three main streams: reject-option classification, learning to defer, and joint classifier-router training. Figure 5 summarizes both empirical and theoretical works discussed here.

## A.1 Empirical Works

Domain-Agnostic Approaches. Adaptive computation and dynamic networks reduce average test-time inference cost by conditionally allocating computation, e.g., by exiting early in a deep network and escalating to more expensive components only when needed [Han et al., 2022a]. One common approach here is to train classifiers separately from the router and make test-time routing decisions using confidence scores compared against a calibrated threshold [Schuster et al., 2022, Jitkrittum et al., 2023].

This is attractive in practice due to its simplicity, as it allows classifiers to be trained independently from the router using standard supervised learning losses.

In contrast to separate training, a growing empirical literature addresses the train-test mismatch caused by this approach, namely that a classifier trained on all samples is applied at test time only to the samples routed to it [Yu et al., 2023, Han et al., 2022b, Regol et al., 2024]. In early-exit networks, where an input leaves at the first exit whose confidence passes a threshold, Yu et al. [2023] train all exits with one summed loss in which each exit is boosted on top of the earlier ones, so that deeper exits concentrate on the samples that earlier exits find hard. Han et al. [2022b] keep this exit rule and meta-learn per-sample loss weights for each exit, which encourage each exit to train on the samples it would receive at test time. Kubaty et al. [2025] show that the schedule for training the backbone and the exit heads also changes the cost-accuracy trade-of, and they propose training the backbone before the whole multi-exit network. Other works also learn the routing rule, as Regol et al. [2024] do by training exit gates together with the classifiers. Mixture-of-experts language models with experts of diferent sizes train a learned gate together with the experts [Sun et al., 2024, Wang et al., 2025], and Wang et al. [2025] add a loss term that penalizes the use of larger experts.

Two of these procedures alternate between deciding which samples each classifier should handle and retraining the classifiers. Regol et al. [2024] alternate between fitting the gates to the lowest-cost exit of each sample and retraining the classifiers with each loss weighted by the predicted probability of exiting at that classifier, while Han et al. [2022b] alternate between assigning samples to exits by simulating confidence-based exiting and updating the classifiers with the resulting weights. Han et al. [2022b] also report that training each exit only on the samples allocated to it, without weights, causes a drastic drop in accuracy. However, these works do not provide theoretical guarantees that their joint training procedures yield models achieving the desired cost-accuracy trade-of.

LLM Domain-Specific Approaches. Recent interest in eficient LLM inference has accelerated work on test-time compute allocation, both within a single model and via multi-model orchestration.

![](images/8baf2c2eb9a02ba76354db0d73811b2d7de24201a8ce8fb17b05000e478947bf.jpg)  
Figure 5: Key related works on multi-model classification.

Within a single model, works on inference eficiency include early-exit networks and depth-adaptive mechanisms [Elbayad et al., 2020, Zeng et al., 2024, Liu et al., 2023, Chen et al., 2024c]. Works on multi-model orchestration include draft-verify strategies (speculative decoding) [Xia et al., 2024, Leviathan et al., 2023], one-shot routers that select a model per input [Chen et al., 2024b, Ong et al., 2025], and cascades that escalate through models of increasing cost [Chen et al., 2024a, Zhang et al., 2024, Valkanas et al., 2025b]. The closest works to the setting we consider are the latter two categories: LLM routers and cascades.

An LLM router takes as input a query embedding produced by a small, low-cost language model, and uses it to select which LLM to invoke [Chen et al., 2024b, Ong et al., 2025]. Cascade approaches [Chen et al., 2024a, Zhang et al., 2024, Valkanas et al., 2025b] prompt LLMs sequentially from cheapest to most expensive, deciding at each step whether to return the response of the current LLM or escalate to a more expensive one. Both the routers and cascades are generally trained on top of fixed, pretrained LLMs. Unlike domain-agnostic approaches, LLM routers and cascades are tailored to LLM-specific signals such as self-consistency scores [Mündler et al., 2024] or human feedback [Ong et al., 2025], which serve as input to these components.

## A.2 Theoretical Works

While empirical approaches achieve inference cost savings in practice, they often lack rigorous analysis. Alongside the empirical work, a theoretical literature has developed surrogate losses for various cost-sensitive classification problems, providing statistical guarantees such as excess-risk bounds and diferent notions of surrogate consistency. This literature can be separated into three main threads, each corresponding to a diferent problem formulation. The first two, reject-option classification and learning to defer, are closely related to the multi-model classification problem and are the most intensively studied. The third thread, more recently developed, addresses the multi-model classification problem itself, so far only with T = 2 classifiers. Since our work sits in the third thread and the main text develops it in full, we focus on the first two here.

Reject-Option Classification. A reject-option classifier, i.e., a classifier with a reject option, may abstain from predicting at a cost lower than that of a misclassification [Chow, 1957, 1970]. The Bayes-optimal reject decision was characterized by Chow [1957, 1970]. Subsequent work developed the consistency theory for it, establishing excess-risk convergence rates for plug-in and empirical risk minimization (ERM) estimators in the binary classification case [Herbei and Wegkamp, 2006] and estimation error bounds that accommodate arbitrary classification-calibrated losses [Cao et al., 2022]. The same abstention mechanism is also studied under the name selective classification [El-Yaniv and Wiener, 2010, Wiener and El-Yaniv, 2011] and rejection learning [Cortes et al., 2016, Hendrickx et al., 2024].

Learning to Defer (L2D). Learning to defer generalizes abstention: instead of declining prediction outright, the classifier may defer to one or more experts of varying capabilities and costs [Madras et al., 2018, Keswani et al., 2021, Wilder et al., 2021]. This literature is close in spirit to multi-model classification problem, since a deferral decision routes the input to an alternative predictor. Mozannar and Sontag [2020] proposed a consistent surrogate for multiclass L2D with a single fixed expert, obtained by a reduction to cost-sensitive learning, which yields a softmax cross-entropy loss over the label space augmented with a deferral option. Verma and Nalisnick [2022] proposed an alternative one-vs-all surrogate for the same single-expert setting and proved that it is a consistent surrogate for the deferral loss.Verma et al. [2023] extended both the softmax and the one-vs-all surrogates to multiple experts.

All of these works established Bayes consistency, a guarantee about minimizers over the class of all measurable functions. This guarantee can be less informative when, in practice, the surrogate is minimized over a restricted hypothesis class H (for instance, linear functions). Long and Servedio [2013] illustrated this with linearly separable distributions, on which a Bayes-consistent surrogate minimized over linear functions keeps a positive classification error, while a surrogate that is not Bayes-consistent drives the error to zero. This motivated them to study consistency relative to a hypothesis class: H-consistency and realizable H-consistency. A surrogate is H-consistent if, for every distribution, minimizing the surrogate risk over H yields an error approaching the smallest error attainable within H. This requirement also includes settings in which no function in H attains zero error, either because the underlying distribution is noisy or because H is too restricted. Long and Servedio [2013] therefore introduced the weaker notion of realizable H-consistency, which imposes the same conditions, but only on distributions for which some function in H attains zero error, so that minimizing the surrogate over H must drive the classification error to zero. Awasthi et al. [2022] later introduced H-consistency bounds, which upper-bound the target estimation error (i.e., the gap between the target risk and the smallest target risk attainable within H) by a function of the surrogate estimation error for a given hypothesis set, and derived explicit bounds for binary classification. These guarantees were stated for classification with the 0-1 loss; the following works on learning to defer adapt them by replacing the classification error with a deferral loss.

In learning to defer with a single expert, Mozannar et al. [2023] showed that prior Bayes-consistent surrogates can fail to find a good linear classifier and deferral function even when a pair with zero error exists, and they proposed a surrogate that is realizable H-consistent for model classes closed under scaling. Mao et al. [2024a] showed that a family of surrogates that includes the one of Mozannar et al. [2023] up to a constant is realizable H-consistent under mild conditions and that suitable members admit H-consistency bounds. This also shows that the surrogate ofMozannar et al. [2023] is Bayes-consistent when the expert cost is its classification error, a question that was left open. For multiple pre-defined experts, Mao et al. [2024b] proved H-consistency bounds when the predictor and the deferral function are learned simultaneously, and Mao et al. [2023] proved such bounds for a two-stage variant that keeps a pre-trained predictor fixed while the deferral function is learned. In both works the cost of deferring to an expert can add an inference cost to that expert’s error. Mao et al. [2024b] also instantiated their surrogate with a constrained hinge loss whose scores sum to zero and obtained an H-consistency bound that is linear in the surrogate estimation error for distributions on which the best surrogate risk within H equals the Bayes surrogate risk. Finally, Mao et al. [2025] proposed surrogates for learning to defer to multiple experts that are realizable H-consistent, admit H-consistency bounds, and are Bayes-consistent, in both the standard and the two-stage settings. To our knowledge, this is the most general consistency analysis of learning to defer.

Joint classifier-router training. L2D assumes the expert(s) are fixed and learns a single classifier together with a deferral rule. Closest to our setting, Regol et al. [2025] and Pugnana et al. [2025] have studied joint training of two classifiers with a cost-sensitive routing rule, but neither extends to the general multi-model setting.Regol et al. [2025] prove Bayes consistency, while Pugnana et al. [2025] prove realizable H-consistency. In contrast, our framework allows an arbitrary number of classifiers with both single-stage and cascade routing, and like Regol et al. [2025], we establish Bayes consistency.

## B Proofs for the Flat Router Model

In this section, we prove the calibration result in Theorem 3.1 for the flat-router hinge loss $\ell _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } }$ For this, we first prove some auxiliary results in Sections B.1–B.2, and based on these, we prove calibration in Section B.3.

Notation. We use LHS (resp. RHS) as short for left (resp. right) hand side. We write $p ( \cdot \mid \cdot )$ for the conditional probability mass function of $^ { \prime } , ~ { \mathrm { e . g . } } , ~ p ( y ~ \mid ~ x ) \equiv \operatorname* { P r } ( Y ~ = ~ y ~ \mid X ~ = ~ x )$ and $p ( y ~ \vert ~ x , z _ { t } ) \equiv \mathrm { P r } ( Y = y ~ \vert ~ X = x , Z _ { t } = z _ { t } )$

## B.1 Bayes-Optimal Solutions

In this section, we characterize the Bayes-optimal solution $\pi _ { \mathrm { f l a t } } ^ { * } ~ \in ~ \Pi _ { \mathrm { f l a t } }$ under the 0-1-c loss (Lemma B.1) and the Bayes-optimal solution $\tilde { \pi } _ { \mathrm { f l a t } } ^ { * } \in \widetilde { \Pi } _ { \mathrm { f l a t } }$ under the hinge loss (Lemma B.2). We first characterize the Bayes-optimal solution within the hypothesis class $\Pi _ { \mathrm { { f l a t } } }$ under the 0-1-c loss associated with the population risk $\mathcal { R } ( \pi ) = \mathbb { E } \big [ \ell ( X , Z _ { 1 : T } , Y ; \pi , c ) \big ]$

Lemma B.1 (Bayes-optimal solution in $\Pi _ { \mathrm { { f l a t } } }$ for 0-1-c loss). For the 0-1-c risk R defined in (10) with cost vector $^ { c , }$ there exists a Bayes-optimal solution

$$
\pi _ { \hat { H } a t } ^ { * } = ( f _ { 1 : T } ^ { * } , r _ { \hat { H } a t } ^ { * } ) \in \underset { \pi \in \Pi _ { \hat { H } a t } } { \arg \operatorname* { m i n } } \mathcal { R } ( \pi )\tag{29}
$$

whose inducing components $( f _ { 1 : T } ^ { * } , r _ { \ : \ : \mathcal { H } a t } ^ { * } )$ satisfy:

$$
f _ { t } ^ { * } ( x , z _ { t } ) \in \underset { y \in [ K ] } { \arg \operatorname* { m a x } } p ( y \mid x , z _ { t } ) \quad \forall t \in [ T ] ,\tag{30}
$$

$$
\boldsymbol { r } _ { \mathcal { H } a t } ^ { * } ( \boldsymbol { x } ) \in \underset { t \in [ T ] } { \arg \operatorname* { m i n } } \mathcal { R } _ { t } ( \boldsymbol { x } ; \boldsymbol { f } _ { t } ^ { * } ) ,\tag{31}
$$

where $\mathcal { R } _ { t } ( x ; f _ { t } ^ { * } )$ denotes the flat stage-wise Bayes risk of $f _ { t } ^ { * }$ conditioned on $X = x .$

$$
\mathscr { R } _ { t } ( x ; f _ { t } ^ { * } ) = 1 - \mathbb { E } \left[ \operatorname* { m a x } _ { y \in [ K ] } p ( y \mid x , Z _ { t } ) \ | \ X = x \right] + c _ { t } \quad \forall t \in [ T ] .\tag{32}
$$

Proof. We first prove the Bayes-optimal solution for the classifiers $f _ { 1 : T } ^ { * }$ then the routing function $r _ { \mathrm { f l a t } } ^ { \ast }$

Part 1: Proof for the Bayes-optimal $f _ { 1 : T } ^ { * } .$ . We first minimize $\mathcal { R } ( \pi ) \equiv \mathcal { R } ( \pi ; c )$ with respect to $f _ { 1 : T }$ for an arbitrary routing rule $r _ { \mathrm { f l a t } } : \mathcal { X }  [ T ]$ . The 0-1-c risk can be written as

$$
\begin{array} { r l } & { \mathcal { R } ( \pi ) = \mathbb { E } \Big [ \ell ( X , Z _ { 1 : T } , Y ; \pi , c ) \Big ] } \\ & { \qquad = \mathbb { E } \Big [ 1 \Big \{ f _ { \mathrm { r l a t } } ( X ) \left( X , Z _ { \mathrm { r l a t } } ( X ) \right) \neq Y \Big \} + c _ { \mathrm { r l a t } } ( X ) \Big ] } \\ & { \qquad = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \Big [ 1 \big \{ r _ { \mathrm { f l a t } } ( X ) = t \} \cdot \Big ( 1 \big \{ f _ { t } ( X , Z _ { t } ) \neq Y \big \} + c _ { t } \Big ) \Big ] } \\ & { \qquad = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \Bigg [ 1 \big \{ r _ { \mathrm { f l a t } } ( X ) = t \} \cdot \underbrace { \Big ( \mathbb { E } \Big [ 1 \big \{ f _ { t } \left( X , Z _ { t } \right) \neq Y \big \} \mid X \Big ] + c _ { t } \Big ) } _ { = : \mathcal { R } _ { t } ( X ; f _ { t } ) } \Bigg ] , } \end{array}\tag{33}
$$

where the last step applies the tower property of conditional expectation, and $\mathcal { R } _ { t } ( X ; f _ { t } )$ denotes the conditional risk of the t-th classifier given $X ;$ the semicolon here emphasizes that X is a variable and $f _ { t }$ is treated as a parameter.

Since $c _ { t }$ is a constant that does not depend on $f _ { 1 : T }$ , minimizing $\mathcal { R } ( \pi )$ with respect to $f _ { t }$ reduces to minimizing $\mathcal { R } _ { t } ( X ; f _ { t } )$ pointwise in X. Fixing $x \in \mathcal { X }$ , by the tower property, we have

$$
\begin{array} { r l } & { { \mathcal R } _ { t } ( x ; f _ { t } ) : = { \mathbb E } \Big [ \mathbb { 1 } \big \{ f _ { t } ( X , Z _ { t } ) \neq Y \big \} \mid X = x \Big ] + c _ { t } } \\ & { \qquad = { \mathbb E } \Big [ { \mathbb E } \big [ \mathbb { 1 } \big \{ f _ { t } ( X , Z _ { t } ) \neq Y \big \} \mid X = x , Z _ { t } \big ] \mid X = x \Big ] + c _ { t } } \\ & { \qquad = { \mathbb E } \Big [ { \mathbb P } \big ( Y \neq f _ { t } ( x , Z _ { t } ) \mid X = x , Z _ { t } \big ) \mid X = x \Big ] + c _ { t } } \\ & { \qquad = 1 - { \mathbb E } \Big [ p ( f _ { t } ( x , Z _ { t } ) \mid x , Z _ { t } ) \mid X = x \Big ] + c _ { t } , } \end{array}\tag{34}
$$

where $p ( \cdot \mid x , Z _ { t } )$ denotes the conditional probability of $Y$ given $X = x$ and $Z _ { t }$ .

Therefore, minimizing $\textstyle { \mathcal { R } } _ { t } ( x ; f _ { t } )$ with respect to $f _ { t }$ reduces to maximizing $p ( f _ { t } ( x , z _ { t } ) \mid x , z _ { t } )$ for each $( x , z _ { t } ) \in \mathcal { X } \times \mathcal { Z } _ { t } ,$ so pointwise in $( x , z _ { t } ) \in \mathcal { X } \times \mathcal { Z } _ { t } .$ we have

$$
f _ { t } ^ { * } ( x , z _ { t } ) \in \underset { y \in [ K ] } { \arg \operatorname* { m a x } } p ( y \mid x , z _ { t } ) ,\tag{35}
$$

which proves (30).

Part 2: Proof for the Bayes-optimal $r _ { \mathbf { f l a t } } ^ { * }$ . We now minimize $\mathcal { R } ( \pi ) \equiv \mathcal { R } ( \pi ; c )$ with respect to $r _ { \mathrm { { f l a t } } }$ holding $f _ { 1 : T } = f _ { 1 : T } ^ { * }$ fixed. Substituting (35) into (34), the conditional Bayes risk of the t-th classifier is

$$
\begin{array} { r l } & { \mathcal { R } _ { t } ( x ; f _ { t } ^ { * } ) : = \mathbb { E } \Big [ \mathbb { 1 } \big \lbrace f _ { t } ^ { * } ( X , Z _ { t } ) \neq Y \big \rbrace | X = x \Big ] + c _ { t } } \\ & { \qquad = 1 - \mathbb { E } \left[ \underset { y \in [ K ] } { \operatorname* { m a x } } p ( y \mid x , Z _ { t } ) \mid X = x \right] + c _ { t } , } \end{array}\tag{36}
$$

yielding (32). Since $\pi = ( f _ { 1 : T } ^ { * } , r _ { \mathrm { f l a t } } )$ here, we write $\mathcal { R } ( f _ { 1 : T } ^ { * } , r _ { \mathrm { f l a t } } )$ to mean $\mathcal { R } ( \pi )$ . From (33), the unconditional risk is $\begin{array} { r } { \mathcal { R } ( f _ { 1 : T } ^ { * } , r _ { \mathrm { f l a t } } ) = \sum _ { t = 1 } ^ { T } \mathbb { E } [ \mathbb { 1 } \left\{ r _ { \mathrm { f l a t } } ( X ) = t \right\} \cdot \mathcal { R } _ { t } ( X ; f _ { t } ^ { * } ) ] } \end{array}$ , so minimizing $\mathcal { R } ( f _ { 1 : T } ^ { * } , r _ { \mathrm { f l a t } } )$ with respect to $r _ { \mathrm { { f l a t } } }$ reduces to a pointwise problem in X. For each $x \in \mathcal { X }$ , the optimal routing function selects the classifier with the minimum conditional risk, that is,

$$
r _ { \mathrm { f l a t } } ^ { * } ( x ) \in \underset { t \in [ T ] } { \arg \operatorname* { m i n } } \mathcal { R } _ { t } ( x ; f _ { t } ^ { * } ) ,\tag{37}
$$

which proves (31).

We now characterize the Bayes-optimal solution within the hypothesis class $\widetilde { \Pi } _ { \mathrm { { f l a t } } }$ under the surrogate hinge loss $\ell _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } }$ associated with the population risk $\mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) = \mathbb { E } \big [ \ell _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } ( X , Z _ { 1 : T } , Y ; \tilde { \pi } , c ) \big ]$

Lemma B.2 (Bayes-optimal solution in $\widetilde { \Pi } _ { \mathrm { { f l a t } } }$ for hinge loss). There exists a Bayes-optimal solution

$$
\tilde { \pi } _ { f l a t } ^ { * } = ( \tilde { f } _ { 1 : T } ^ { * } , \tilde { r } _ { f a t } ^ { * } ) \in \arg \operatorname* { m i n } _ { \tilde { \pi } \in \widetilde { \Pi } _ { f a t } } \mathcal { R } _ { f l a t } ^ { h i n g e } ( \tilde { \pi } ) ,
$$

where, with $f _ { t } ^ { * }$ defined as in (30),

$$
\left[ \tilde { f } _ { t } ^ { * } ( x , z _ { t } ) \right] _ { k } = \left\{ \begin{array} { l l } { 1 } & { f o r k = f _ { t } ^ { * } ( x , z _ { t } ) } \\ { - \frac { 1 } { K - 1 } } & { o t h e r w i s e } \end{array} \right. \quad \forall t \in [ T ] ,\tag{38}
$$

and

$$
\tilde { r } _ { f l a t } ^ { * } ( x ) \in \left\{ v \in \Delta ^ { T - 1 } : v _ { k } = 0 \quad \forall k \notin \underset { t \in [ T ] } { \arg \operatorname* { m i n } } \mathcal { R } _ { t } ( x ; f _ { t } ^ { * } ) \right\} .\tag{39}
$$

Proof. We prove (38) and (39) one at a time.

Part 1: Proof for the Bayes-optimal $\tilde { f } _ { 1 : T } ^ { * }$ . The result in (38) directly follows from [Tarigan and van de Geer, 2008, Lemma 1].

Part 2: Proof for the Bayes-optimal $\tilde { r } _ { \mathbf { f l a t } } ^ { * } .$ . We now minimize $\mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } )$ w.r.t. $\tilde { r } _ { \mathrm { { f l a t } } }$ holding $\tilde { f } _ { 1 : T } = \tilde { f } _ { 1 : T } ^ { * }$ fixed. By definition the overall flat-router hinge risk is

$$
\mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } [ \tilde { r } _ { \mathrm { f l a t } } ( X ) ] _ { t } \cdot \ell ^ { \mathrm { h i n g e } } ( \tilde { f } _ { t } ( X , Z _ { t } ) , Y ; c _ { t } ) \right]
$$

Using $\tilde { f } _ { 1 : T } = \tilde { f } _ { 1 : T } ^ { * }$ and writing $\tilde { \pi } = ( \tilde { f } _ { 1 : T } ^ { * } , \tilde { r } _ { \mathrm { f l a t } } )$ , we have via Lemma B.3 that

$$
\begin{array} { r l } & { \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( x , z _ { t } ; \tilde { f } _ { t } ^ { * } ) : = \mathbb { E } \Big [ \ell ^ { \mathrm { h i n g e } } ( \tilde { f } _ { t } ^ { * } ( X , Z _ { t } ) , Y ; c _ { t } ) \mid X = x , Z _ { t } = z _ { t } \Big ] } \\ & { \quad \quad \quad = \frac { K } { K - 1 } \mathcal { R } _ { t } ( x , z _ { t } ; f _ { t } ^ { * } ) , } \end{array}\tag{40}
$$

which implies that the overall hinge risk conditioned on $X = x$ can be written as

$$
\begin{array} { r l } & { { \mathcal R } ^ { \mathrm { h i n g e } } ( x ; \tilde { \pi } ) : = \displaystyle \sum _ { t = 1 } ^ { T } [ \tilde { r } _ { \mathrm { f a t } } ( x ) ] _ { t } \cdot { \mathbb E } \left[ \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( x , Z _ { t } ; \hat { f } _ { t } ^ { * } ) \mid X = x \right] } \\ & { \quad \quad \quad \quad = \displaystyle \frac { K } { K - 1 } \sum _ { t = 1 } ^ { T } [ \tilde { r } _ { \mathrm { f a t } } ( x ) ] _ { t } \cdot { \mathbb E } \left[ \mathcal { R } _ { t } ( x , Z _ { t } ; f _ { t } ^ { * } ) \mid X = x \right] } \\ & { \quad \quad \quad = \displaystyle \frac { K } { K - 1 } \sum _ { t = 1 } ^ { T } [ \tilde { r } _ { \mathrm { f a t } } ( x ) ] _ { t } \cdot { \mathcal R } _ { t } ( x ; f _ { t } ^ { * } ) , } \end{array}\tag{41}
$$

where the last step holds by definition of $\mathcal { R } _ { t } ( x ; f _ { t } ^ { * } )$ in (32). Therefore any minimizer $\tilde { r } _ { \mathrm { f l a t } } ^ { * } ( x ) \in \Delta ^ { T - }$ 1 of $\mathcal { R } ^ { \mathrm { h i n g e } } ( x ; \tilde { \pi } )$ must be supported on the set arg mi $\mathsf { 1 } _ { t \in [ T ] } \mathscr { R } _ { t } ( x ; f _ { t } ^ { * } )$ , establishing (39). In particular, any one-hot vector concentrated on any element of this set is a valid choice. □

## B.2 Relationship Between 0-1-c and Hinge Conditional Risks

Using the characterization of the Bayes-optimal solutions from Lemmas B.1–B.2, the next two lemmas connect the hinge and 0-1-c conditional Bayes risks, and the hinge and 0-1-c excess risks, respectively, which will be handy for our calibration proof in the next section.

Lemma B.3 (Hinge and 0-1-c conditional Bayes risks difer by a factor of $\frac { K } { K - 1 } )$ . For $t \in [ T ]$ , let $f _ { t } ^ { * }$ and $\tilde { f } _ { t } ^ { * }$ denote the Bayes-optimal classifiers from Lemmas $B . 1 { - } B . 2$ under the 0-1-c loss and the hinge loss, respectively. Define the conditional Bayes risks

$$
\mathscr { R } _ { t } ( x , z _ { t } ; f _ { t } ^ { * } ) : = \mathbb { E } \Big [ \mathbb { 1 } \big \{ f _ { t } ^ { * } ( X , Z _ { t } ) \neq Y \big \} + c _ { t } \ | \ X = x , Z _ { t } = z _ { t } \Big ] ,\tag{42}
$$

$$
\mathcal { R } _ { t } ^ { h i n g e } ( x , z _ { t } ; \tilde { f } _ { t } ^ { * } ) : = \mathbb { E } \Big [ \ell ^ { h i n g e } ( \tilde { f } _ { t } ^ { * } ( X , Z _ { t } ) , Y ; c _ { t } ) \mid X = x , Z _ { t } = z _ { t } \Big ] ,\tag{43}
$$

then

$$
\begin{array} { r } { \mathcal { R } _ { t } ^ { h i n g e } ( x , z _ { t } ; \tilde { f } _ { t } ^ { * } ) = \frac { K } { K - 1 } \mathcal { R } _ { t } ( x , z _ { t } ; f _ { t } ^ { * } ) . } \end{array}\tag{44}
$$

Proof. We can write

$$
\begin{array} { r l } & { { \mathcal R } _ { t } ^ { \mathrm { h i n g e } } ( x , z _ { t } ; \tilde { f } _ { t } ^ { * } ) : = { \mathbb E } \left[ \ell ^ { \mathrm { h i n g e } } ( \tilde { f } _ { t } ^ { * } ( X , Z _ { t } ) , Y ; c _ { t } ) \mid X = x , Z _ { t } = z _ { t } \right] } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad  \end{array}\tag{45}
$$

Lemma B.4 (Calibration of the stage-wise cost-sensitive hinge loss $\ell ^ { \mathrm { h i n g e } } )$ . Let $\tilde { f } _ { t }$ be a soft classifier as defined in (11) and $f _ { t }$ its induced hard classifier in (12). Define the conditional risks $\mathcal { R } _ { t } ( x , z _ { t } ; f _ { t } )$ and $\mathcal { \bar { R } } _ { t } ^ { h i n g e } ( x , \dot { z } _ { t } ; \dot { f } _ { t } )$ analogously to $( 4 2 ) - ( 4 3 )$ . Then, for all $t \in [ T ]$ and $( x , z _ { t } ) \in \mathcal { X } \times \mathcal { Z } _ { t }$

$$
\begin{array} { r } { \mathcal { R } _ { t } ( x , z _ { t } ; f _ { t } ) - \mathcal { R } _ { t } ( x , z _ { t } ; f _ { t } ^ { * } ) \leq ( K - 1 ) \left( \mathcal { R } _ { t } ^ { h i n g e } ( x , z _ { t } ; \tilde { f } _ { t } ) - \mathcal { R } _ { t } ^ { h i n g e } ( x , z _ { t } ; \tilde { f } _ { t } ^ { * } ) \right) . } \end{array}\tag{46}
$$

As an immediate remark, since (46) holds for every $z _ { t } ,$ we can trivially obtain the same bound for the risk conditioned on $X = x { \mathrm { ~ o n l y } }$ . Likewise, in our proofs for the cascade model in Appendix $\mathrm { C } ,$ (46) also holds with $( x , z _ { t } )$ replaced by $h _ { t }$ , since $p ( y \mid h _ { t } ) = p ( y \mid x , z _ { t } )$ under the conditiona independence assumption (5) and both $f _ { t }$ and $\tilde { f } _ { t }$ depend on $h _ { t }$ only through $( x , z _ { t } )$

Proof. We begin by setting up some notation. Fix $( x , z _ { t } )$ and let $p _ { y } : = p ( y \mid x , z _ { t } )$ . Let $v : =$ $\tilde { f } _ { t } ( x , z _ { t } ) \in \mathbb { R } ^ { K }$ be the score vector that $\tilde { f } _ { t }$ outputs and $j : = f _ { t } ( x , z _ { t } ) \in [ K ]$ the class index that $f _ { t }$ outputs. By the argmax map in (12), we have $j \in \arg \operatorname* { m a x } _ { k \in [ K ] } v _ { k }$ . Using the definition of the hinge loss $\ell ^ { \mathrm { h i n g e } }$ in (15), we define its cost-free variant $\bar { \ell } ^ { \mathrm { h i n g e } }$ as

$$
\bar { \ell } ^ { \mathrm { h i n g e } } ( v , y ) : = \ell ^ { \mathrm { h i n g e } } ( v , y ; 0 ) = \sum _ { k \in [ K ] \setminus y } \left[ v _ { k } + \frac { 1 } { K - 1 } \right] _ { + } .\tag{47}
$$

We further define two useful shorthand: $L ( v )$ as the associated conditional risk, and $U$ as the total penalty:

$$
L ( v ) : = \mathbb { E } \left[ \bar { \ell } ^ { \mathrm { h i n g e } } ( v , Y ) \mid X = x , Z _ { t } = z _ { t } \right] = \sum _ { y \in [ K ] } p _ { y } \bar { \ell } ^ { \mathrm { h i n g e } } ( v , y ) ,\tag{48}
$$

$$
U : = \sum _ { k \in [ K ] } \left[ v _ { k } + \frac 1 { K - 1 } \right] _ { + } .\tag{49}
$$

We first express both excess risks in (46) in terms of $p _ { y }$ and $L ( v )$ , then derive a lower bound for the conditional expectation $L ( v )$ to obtain the final inequality.

Part 1. Rewriting the stage-wise excess risks: By definition, we can write

$$
\begin{array} { r l } & { \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( x , z _ { t } ; \widetilde { f } _ { t } ) : = \mathbb { E } \left[ \ell ^ { \mathrm { h i n g e } } ( \widetilde { f } _ { t } ( X , Z _ { t } ) , Y ; c _ { t } ) \mathbf { \Phi } | \mathbf { \Phi } X = x , Z _ { t } = z _ { t } \right] } \\ & { \quad \quad \quad = \displaystyle \sum _ { y \in [ K ] } { p _ { y } \ell ^ { \mathrm { h i n g e } } ( v , y ; c _ { t } ) } = L ( v ) + \frac { K } { K - 1 } c _ { t } . } \end{array}\tag{50}
$$

Combined with $\begin{array} { r } { \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( x , z _ { t } ; \tilde { f } _ { t } ^ { * } ) = \frac { K } { K - 1 } \left( 1 - \operatorname* { m a x } _ { y \in [ K ] } p _ { y } \right) + \frac { K } { K - 1 } c _ { t } } \end{array}$ from (45), the cost terms cancel and the excess stage-wise hinge risk can be written into

$$
\mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( x , z _ { t } ; \tilde { f } _ { t } ) - \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( x , z _ { t } ; \tilde { f } _ { t } ^ { * } ) = L ( v ) - \frac { K } { K - 1 } \Bigl ( 1 - \operatorname* { m a x } _ { y \in [ K ] } p _ { y } \Bigr ) .\tag{51}
$$

Similarly, for the stage-wise 0-1-c risk, since $f _ { t }$ predicts label $j ,$ we can write

$$
\mathcal { R } _ { t } ( x , z _ { t } ; f _ { t } ) = \mathbb { E } \left[ \mathbb { 1 } \left\{ f _ { t } ( X , Z _ { t } ) \neq Y \right\} + c _ { t } \ \middle \vert \ X = x , Z _ { t } = z _ { t } \right] = 1 - p _ { j } + c _ { t } .\tag{52}
$$

Combined with $\begin{array} { r } { \mathcal { R } _ { t } ( x , z _ { t } ; f _ { t } ^ { * } ) = 1 - \operatorname* { m a x } _ { y \in [ K ] } p _ { y } + c _ { t } } \end{array}$ since $f _ { t } ^ { * }$ predicts the most likely label, we can write the excess 0-1-c stage-wise risk into

$$
\mathcal { R } _ { t } ( x , z _ { t } ; f _ { t } ) - \mathcal { R } _ { t } ( x , z _ { t } ; f _ { t } ^ { * } ) = \operatorname* { m a x } _ { y \in [ K ] } p _ { y } - p _ { j } .\tag{53}
$$

In the rest of the proof, we provide a lower bound for $L ( v )$ , which will be integrated into (51) and combined with (53) to prove the final calibration inequality (46).

Part 2. Lower bounding $L ( v )$ : Using the definition of $L ( v )$ in (48), we can rewrite it as follows

$$
L ( v ) = \sum _ { y \in [ K ] } p _ { y } \sum _ { \substack { k \in [ K ] \setminus y } } \left[ v _ { k } + \frac { 1 } { K - 1 } \right] _ { + } \stackrel { \mathrm { ( i ) } } { = } \sum _ { k \in [ K ] } \left( 1 - p _ { k } \right) \left[ v _ { k } + \frac { 1 } { K - 1 } \right] _ { + } \stackrel { \mathrm { ( i i ) } } { = } U - \sum _ { k \in [ K ] } p _ { k } \left[ v _ { k } + \frac { 1 } { K - 1 } \right] _ { + }
$$

$$
\stackrel { \mathrm { ( i i i ) } } { = } U - p _ { j } \left[ v _ { j } + \frac { 1 } { K - 1 } \right] _ { + } - \sum _ { k \in [ K ] \backslash j } p _ { k } \left[ v _ { k } + \frac { 1 } { K - 1 } \right] _ { + } ,\tag{54}
$$

where step (i) follows from exchanging the two summations, step (ii) applies the definition of U from (49), and step (iii) simply splits the $j \mathrm { - t h }$ term from the summation. Using this, we now lower bound L(v):

$$
\begin{array} { l } { { \displaystyle { \cal L } ( v ) = U - p _ { j } \left[ v _ { j } + \frac 1 { K - 1 } \right] _ { + } - \sum _ { k \in [ K ] \backslash j } p _ { k } \left[ v _ { k } + \frac 1 { K - 1 } \right] _ { + } } } \\ { { \displaystyle \quad \geq U - p _ { j } \left[ v _ { j } + \frac 1 { K - 1 } \right] _ { + } - \operatorname* { m a x } _ { y \in [ K ] } p _ { y } \left( U - \left[ v _ { j } + \frac 1 { K - 1 } \right] _ { + } \right) } } \\ { { \displaystyle \quad = U \left( 1 - \operatorname* { m a x } _ { y \in [ K ] } p _ { y } \right) + \left[ v _ { j } + \frac 1 { K - 1 } \right] _ { + } \left( \operatorname* { m a x } _ { y \in [ K ] } p _ { y } - p _ { j } \right) \ : , } } \end{array}\tag{55}
$$

where the inequality holds by applying $p _ { k } \le \operatorname* { m a x } _ { y \in [ K ] } p _ { y }$ and the definition of U from (49).

Part 3. Combining all components: To combine all components of the proof, we begin with two auxiliary inequalities about U and the hinge terms. First, U is lower bounded by $\frac { K } { K - 1 }$ because

$$
\begin{array} { r } { U = \displaystyle \sum _ { k \in \left[ K \right] } \left[ v _ { k } + \frac { 1 } { K - 1 } \right] _ { + } \stackrel { \mathrm { ( i ) } } { \geq } \displaystyle \sum _ { k \in \left[ K \right] } \left( v _ { k } + \frac { 1 } { K - 1 } \right) \stackrel { \mathrm { ( i i ) } } { = } \frac { K } { K - 1 } , } \end{array}\tag{56}
$$

where step (i) uses $[ x ] _ { + } \geq x ,$ and step (ii) uses $\begin{array} { r } { \sum _ { k \in [ K ] } v _ { k } = 0 } \end{array}$ . Second, since the predicted label $j \in \arg \operatorname* { m a x } _ { k \in [ K ] } v _ { k }$ , we have $v _ { j } \geq v _ { k }$ for all $k \in [ K ]$ , and so

$$
\left[ v _ { j } + \frac { 1 } { K - 1 } \right] _ { + } = \frac { 1 } { K } \sum _ { k \in [ K ] } \left[ v _ { j } + \frac { 1 } { K - 1 } \right] _ { + } \ge \frac { 1 } { K } \sum _ { k \in [ K ] } \left[ v _ { k } + \frac { 1 } { K - 1 } \right] _ { + } = \frac { 1 } { K } \cdot U \ge \frac { 1 } { K - 1 } ,\tag{57}
$$

where the last inequality applies the previous inequality (56).

Equipped with the lower bound in (55) for $L ( v )$ and the auxiliary inequalities (56)–(57), we can now lower bound the excess stage-wise hinge risk from (51) as follows:

$$
\begin{array} { r l } & { \mathbb { R } _ { t } ^ { \mathrm { H o m } } ( x , z , \hat { \mu } _ { t } ^ { \mathrm { A } } ) = - \ R _ { t } ^ { \mathrm { H o m } } ( x , z , \hat { \mu } _ { t } ^ { \mathrm { B } } ) } \\ & { = L ( v ) - \frac { R } { R _ { 1 } } ( 1 - \frac { \operatorname* { m a x } } { v ( \hat { \kappa } ( \hat { \kappa } ) ) \sigma _ { t } } ) } \\ & { \overset { ( a ) } { \geq } \mathcal { L } ( 1 - \frac { \operatorname* { m a x } _ { \hat { \mu } } \hat { \nu } _ { t } } { \operatorname* { s g n } ( \mathcal { R } ) } ) + \Big [ x _ { 3 } + \frac { 1 } { R _ { 1 } } \Big ] _ { + } ( \underset { y \in [ \hat { \kappa } ] } { \operatorname* { m a x } } \mathcal { P } _ { t } - p _ { 2 } ) - \frac { K } { R _ { 1 } } ( 1 - \underset { y \in [ \hat { \kappa } ] } { \operatorname* { m a x } } \mathcal { P } _ { t } ) } \\ & { = \underbrace { ( U - \frac { \kappa _ { \hat { \kappa } } } { \kappa ( \hat { \kappa } ) } ) } _ { \geq 0 ( b ) + 1 } ( 1 - \underset { y \in [ \hat { \kappa } ] } { \operatorname* { m a x } } \mathcal { P } _ { y } ) + \Big [ v _ { j } + \frac { 1 } { R _ { 1 } - 1 } \Big ] _ { + } ( \underset { y \in [ \hat { \kappa } ] } { \operatorname* { m a x } } \mathcal { P } _ { y } - p _ { j } ) } \\ & { \overset { ( a ) } { \geq } [ x _ { j } + \frac { 1 } { \kappa ( \hat { \kappa } ) } ] _ { + } ( \underset { y \in [ \hat { \kappa } ] } { \operatorname* { m a x } } \mathcal { P } _ { y } - p _ { j } ) } \\ &  \overset { ( b ) } { \leq } \frac { 1 } { \kappa ( \hat { \kappa } ) } { \operatorname* { m a x } } ( \underset  y \in [ \hat  \end{array}\tag{58}
$$

where (i) applies (55), (ii) holds by dropping the non-negative term $\left( U - { \frac { K } { K - 1 } } \right) \left( 1 - \operatorname* { m a x } _ { y \in [ K ] } p _ { y } \right)$ (iii) uses (57), and the last step uses the expression of the excess 0-1-c risk from (53). Multiplying both sides by $K - 1$ proves our final calibration inequality (46). □

## B.3 Proof of Theorem 3.1

Using the results in Sections B.1–B.2, we prove our main calibration result. We first restate the theorem.

Theorem 3.1 (Linear calibration of $\ell _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } }$ w.r.t. ℓ). Define $\begin{array} { r } { \mathcal { R } _ { \mathcal { \hat { H } } a t } ^ { * } { : = } \operatorname* { m i n } _ { \pi \in \Pi _ { \mathcal { \boldsymbol { H } } a t } } \mathcal { R } ( \pi ) } \end{array}$ and $\mathcal { R } _ { f l a t } ^ { h i n g e * } : =$ min $\mathcal { \tilde { \pi } } \mathrm { \in } \widetilde { \Pi } _ { \mathrm { \it f l a t } }  \mathcal { R } _ { \mathrm { \it f l a t } } ^ { h i n g e } ( \tilde { \pi } )$ . Then, for any $\tilde { \pi } \in \widetilde { \Pi } _ { f l a t }$

$$
\begin{array} { r } { \mathcal { R } \big ( \varphi ( \tilde { \pi } ) \big ) - \mathcal { R } _ { f a t } ^ { * } \leq T ( K - 1 ) \Big ( \mathcal { R } _ { f a t } ^ { h i n g e } ( \tilde { \pi } ) - \mathcal { R } _ { f a t } ^ { h i n g e * } \Big ) . } \end{array}\tag{17}
$$

Proof. First we set up some notation. Let

$$
\begin{array} { r } { \widetilde { \pi } _ { \mathrm { f a t } } = \left( \widetilde { f } _ { 1 : T } , \widetilde { r } _ { \mathrm { f l a t } } \right) \in \widetilde { \Pi } _ { \mathrm { f l a t } } \qquad \mathrm { a n d } \qquad \pi _ { \mathrm { f a t } } = \left( f _ { 1 : T } , r _ { \mathrm { f l a t } } \right) = \varphi ( \widetilde { \pi } _ { \mathrm { f l a t } } ) \in \Pi _ { \mathrm { f l a t } } } \end{array}
$$

denote an arbitrary soft predictor and its hard counterpart. Let

$$
\tilde { \pi } _ { \mathrm { f a t } } ^ { * } = ( \tilde { f } _ { \mathrm { 1 : } T } ^ { * } , \tilde { r } _ { \mathrm { f l a t } } ^ { * } ) \in \underset { \tilde { \pi } \in \tilde { \Pi } _ { \mathrm { f l a t } } } { \arg \operatorname* { m i n } } \mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) \qquad \mathrm { a n d } \qquad \pi _ { \mathrm { f l a t } } ^ { * } = ( f _ { \mathrm { 1 : } T } ^ { * } , r _ { \mathrm { f l a t } } ^ { * } ) \in \underset { \pi \in \Pi _ { \mathrm { f l a t } } } { \arg \operatorname* { m i n } } \mathcal { R } ( \pi )
$$

be the hinge risk minimizer of Lemma B.2, and the 0-1-c risk minimizer of Lemma B.1.

For a given input $x \in { \mathcal { X } } ,$ , we use

$$
\begin{array} { r } { t : = r _ { \mathrm { f l a t } } ( x ) \in \underset { \tau \in [ T ] } { \arg \operatorname* { m a x } } [ \tilde { r } _ { \mathrm { f l a t } } ( x ) ] _ { \tau } \qquad \mathrm { a n d } \qquad t ^ { * } : = r _ { \mathrm { f l a t } } ^ { * } ( x ) \in \underset { \tau \in [ T ] } { \arg \operatorname* { m i n } } \mathcal { R } _ { \tau } ( x ; f _ { \tau } ^ { * } ) } \end{array}\tag{59}
$$

to denote the index of the classifier selected by the hard predictor $\pi _ { \mathrm { f l a t } }$ and the Bayes-optimal predictor $\pi _ { \mathrm { f l a t } } ^ { * }$ , respectively.

Hence, to prove (17), it sufices to prove pointwise in $x \in \mathcal { X }$ that

$$
\begin{array} { r } { \mathcal { R } _ { t } ( x ; f _ { t } ) - \mathcal { R } _ { t ^ { * } } ( x ; f _ { t ^ { * } } ^ { * } ) \leq T ( K - 1 ) \left( \mathcal { R } _ { \mathrm { f a t } } ^ { \mathrm { h i n g e } } ( x ; \tilde { \pi } ) - \mathcal { R } _ { \mathrm { f a t } } ^ { \mathrm { h i n g e } } ( x ; \tilde { \pi } ^ { * } ) \right) , } \end{array}\tag{60}
$$

where each term here denotes conditional risks, conditioned on $X = x ,$ since taking total expectation on both sides yields the desired result. We prove (60) in the rest of this proof.

Part 1: Decomposition of the conditional excess 0-1-c risk. We now provide general definitions that we used to define the terms on the LHS of (60). For $\tau \in [ T ]$ , define in line with (22),

$$
\mathcal { R } _ { \tau } ( x ; f _ { \tau } ) : = \mathbb { E } \left[ \mathbb { 1 } \{ f _ { \tau } ( x , Z _ { \tau } ) \neq Y \} \ | \ X = x ] + \kappa _ { \tau } , \right.\tag{61}
$$

$$
\mathcal { R } _ { \tau } ( x ; f _ { \tau } ^ { * } ) : = \mathbb { E } \left[ \mathbb { 1 } \{ f _ { \tau } ^ { * } ( x , Z _ { \tau } ) \neq Y \} \ | \ X = x \right] + \kappa _ { \tau }
$$

$$
\stackrel { \mathrm { ( i ) } } { = } 1 - \mathbb { E } \left[ \operatorname* { m a x } _ { y \in \left[ K \right] } p ( y \mid x , Z _ { \tau } ) \mid X = x \right] + \kappa _ { \tau } ,\tag{62}
$$

where (i) applies the expression for $f _ { \tau } ^ { * }$ from (30). Combining these, we can write

$$
\mathcal { R } _ { t } ( x ; f _ { t } ) - \mathcal { R } _ { t ^ { * } } ( x ; f _ { t ^ { * } } ^ { * } ) = \underbrace { \mathcal { R } _ { t } ( x ; f _ { t } ) - \mathcal { R } _ { t } ( x ; f _ { t } ^ { * } ) } _ { = : \Delta _ { \mathrm { i n t e r m a l } } ( x ) } + \underbrace { \mathcal { R } _ { t } ( x ; f _ { t } ^ { * } ) - \mathcal { R } _ { t ^ { * } } ( x ; f _ { t ^ { * } } ^ { * } ) } _ { = : \Delta _ { \mathrm { s l e l e c t } } ( x ) } = \Delta _ { \mathrm { i n t e r m a l } } ( x ) + \Delta _ { \mathrm { s e l e c t } } ( x ) .\tag{63}
$$

Intuitively, $\Delta _ { \mathrm { i n t e r n a l } } ( x )$ denotes the excess risk caused by using a suboptimal classifier as the t-th classifier, so it is an excess risk internal to the t-th stage. Note $\Delta _ { \mathrm { i n t e r n a l } } ( x ) \geq 0$ because $f _ { t } ^ { * }$ is the Bayes-optimal t-th classifier; see the explanation around (34)–(35). The $\Delta _ { \mathrm { s e l e c t } } ( x )$ term is the excess risk caused by selecting a suboptimal exit stage $t ,$ and $\Delta _ { \mathrm { s e l e c t } } ( x ) \geq 0$ by the definition of $t ^ { * }$ in (59).

Part 2: Decomposition of the conditional excess hinge risk. We can similarly decompose the conditional excess hinge risk. To begin, define the stage-τ conditional hinge risk given $X = x { \mathrm { ~ a s } }$

$$
\mathscr { R } _ { \tau } ^ { \mathrm { h i n g e } } ( x ; \tilde { f } _ { \tau } ) : = \mathbb { E } \left[ \ell ^ { \mathrm { h i n g e } } ( \tilde { f } _ { \tau } ( x , Z _ { \tau } ) , Y ; \kappa _ { \tau } ) ~ | ~ X = x \right] \quad \mathrm { f o r } \quad \tau \in [ T ] .\tag{64}
$$

Evaluated at the Bayes-optimal solution, this is linked to the stage-τ conditional 0-1-c risk via

$$
\begin{array} { r l } & { \mathcal { R } _ { \tau } ^ { \mathrm { h i n g e } } ( x ; \tilde { f } _ { \tau } ^ { * } ) \overset { \mathrm { ( i ) } } { = } \mathbb { E } \left[ \mathcal { R } _ { \tau } ^ { \mathrm { h i n g e } } ( x , Z _ { \tau } ; f _ { \tau } ^ { * } ) \mid X = x \right] } \\ & { \qquad \overset { \mathrm { ( i i ) } } { = } \frac { K } { K - 1 } \mathbb { E } \left[ \mathcal { R } _ { \tau } ( x , Z _ { \tau } ; f _ { \tau } ^ { * } ) \mid X = x \right] \overset { \mathrm { ( i i i ) } } { = } \frac { K } { K - 1 } \mathcal { R } _ { \tau } ( x ; f _ { \tau } ^ { * } ) \quad \forall \tau \in [ T ] , } \end{array}\tag{65}
$$

where (i) and (iii) apply the tower property with respect to $P _ { Z _ { \tau } | X } ( \cdot \mid x )$ , and (ii) applies (44). This result will become useful shortly.

Note that Lemma B.2 proves that the support of any Bayes-optimal $\tilde { r } _ { \mathrm { { f l a t } } } ^ { * } ( x )$ is contained in arg $\begin{array} { r l } { \operatorname* { m i n } _ { t \in [ T ] } \mathcal { R } _ { t } ( x ; f _ { t } ^ { * } ) } & { { } } \end{array}$ , so the discrete router $r _ { \mathrm { { f l a t } } } ^ { * } ( x )$ from Lemma B.1 satisfies

$$
t ^ { * } : = r _ { \mathrm { f l a t } } ^ { * } ( x ) \in \underset { t \in [ T ] } { \arg \operatorname* { m i n } } \mathcal { R } _ { t } ( x ; f _ { t } ^ { * } ) \supseteq \underset { t \in [ T ] } { \arg \operatorname* { m a x } } \left[ \tilde { r } _ { \mathrm { f l a t } } ^ { * } ( x ) \right] _ { t } .\tag{66}
$$

Hence, using (64) and (66) and letting $\mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } ( x ; \tilde { \pi } ^ { * } )$ denote the conditional flat-router hinge risk, conditional on $X = x$ , we can write

$$
\begin{array} { l } { \mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } ( x ; \tilde { \pi } ^ { * } ) : = \displaystyle \sum _ { \tau = 1 } ^ { T } \left[ \tilde { r } _ { \mathrm { f l a t } } ^ { * } ( x ) \right] _ { \tau } \mathcal { R } _ { \tau } ^ { \mathrm { h i n g e } } ( x ; \tilde { f } _ { \tau } ^ { * } ) } \\ { \displaystyle \overset { ( \mathrm { i } ) } { = } \sum _ { \tau = 1 } ^ { T } \left[ \tilde { r } _ { \mathrm { f l a t } } ^ { * } ( x ) \right] _ { \tau } \frac { K } { K - 1 } \mathcal { R } _ { \tau } ( x ; f _ { \tau } ^ { * } ) \overset { ( \mathrm { i i } ) } { = } \frac { K } { K - 1 } \mathcal { R } _ { t ^ { * } } ( x ; f _ { t ^ { * } } ^ { * } ) \overset { ( \mathrm { i i i } ) } { = } \mathcal { R } _ { t ^ { * } } ^ { \mathrm { h i n g e } } ( x ; \tilde { f } _ { t ^ { * } } ^ { * } ) , } \end{array}\tag{67}
$$

where steps (i) and (iii) hold by (65), and step (ii) uses the fact that $\mathscr { R } _ { \tau } ( x ; f _ { \tau } ^ { * } ) = \mathscr { R } _ { t ^ { * } } ( x ; f _ { t ^ { * } } ^ { * } )$ for all $\tau \in \mathrm { a r g } \operatorname* { m i n } _ { t \in [ T ] } \mathscr { R } _ { t } ( x ; f _ { t } ^ { * } )$ . Using (64) and (67), we can therefore write

$$
\begin{array} { l } { \mathcal { R } _ { \Psi _ { \Psi } } ^ { \mathrm { h o r g } } ( x ; \tilde { \tau } ) - \mathcal { R } _ { \Psi _ { \Psi } } ^ { \mathrm { h i n g } } ( x ; \tilde { \tau } ; \tilde { \tau } ^ { * } ) } \\ { = \displaystyle \sum _ { \tau = 1 } [ \tilde { r } _ { \tilde { t } \mathrm { d a t } } ( x ) ] \tau \mathcal { R } _ { \tau } ^ { \mathrm { h i n g } } ( x ; \tilde { f } _ { \tau } ) - \mathcal { R } _ { \Psi _ { \tau } } ^ { \mathrm { h i n g } } ( x ; \tilde { f } _ { \tau } ^ { * } ) } \\ { = \displaystyle \sum _ { \tau = 1 } [ \tilde { r } _ { \tilde { t } \mathrm { d a t } } ( x ) ] \tau ( \mathcal { R } _ { \tau } ^ { \mathrm { h i n g } } ( x ; \tilde { f } _ { \tau } ) - \mathcal { R } _ { \tau } ^ { \mathrm { h i n g } } ( x ; \tilde { f } _ { \tau } ^ { * } ) + \mathcal { R } _ { \tau } ^ { \mathrm { h i n g } } ( x ; \tilde { f } _ { \tau } ^ { * } ) - \mathcal { R } _ { \tau } ^ { \mathrm { h i n g } \tau } ( x ; \tilde { f } _ { \tau } ^ { * } ) ) } \\ { = \displaystyle \sum _ { \tau = 1 } [ \tilde { r } _ { \tilde { t } \mathrm { d a t } } ( x ) ] _ { \tau } ( \mathcal { R } _ { \tau } ^ { \mathrm { h i n g } } ( x ; \tilde { f } _ { \tau } ) - \mathcal { R } _ { \tau } ^ { \mathrm { h i n g } } ( x ; \tilde { f } _ { \tau } ^ { * } ) + \mathcal { R } _ { \tau } ^ { \mathrm { h i n g } } ( x ; \tilde { f } _ { \tau } ^ { * } ) - \mathcal { R } _ { \tau } ^ { \mathrm { h i n g } \tau } ( x ; \tilde { f } _ { \tau } ^ { * } ) ) } \\  = \displaystyle \sum _ { \tau = 1 } [ \tilde { r } _ { \tilde { t } \mathrm { d a t } } ( x ) ] _ { \tau } ( \end{array}\tag{8}
$$

where we have $\Delta _ { \mathrm { i n t e r n a l } } ^ { \mathrm { h i n g e } } ( x , \tau ) \geq 0$ by applying [Tarigan and van de Geer, 2008, Lemma 1] to the Bayes-optimal soft classifier $\tilde { f } _ { \tau } ^ { * }$ , and $\begin{array} { r } { \Delta _ { \mathrm { s e l e c t } } ^ { \mathrm { h i n g e } } ( x , \tau ) = \frac { K } { K - 1 } \left( \mathcal { R } _ { \tau } ( x , f _ { \tau } ^ { * } ) - \mathcal { R } _ { t ^ { * } } ( x , f _ { t ^ { * } } ^ { * } ) \right) \geq 0 } \end{array}$ by (65) and the definition of $t ^ { * }$ in (59).

Part 3: Upper bounding $\Delta _ { \mathrm { i n t e r n a l } } ( x )$ and $\scriptstyle \Delta _ { \mathrm { s e l e c t } } ( x )$ by $\Delta _ { \mathrm { i n t e r n a l } } ^ { \mathrm { h i n g e } } ( x , \tau )$ and $\Delta _ { \mathbf { s e l e c t } } ^ { \mathbf { h i n g e } } ( x , \tau )$ . By Lemma B.4 (given only $X = x )$ , which links $\Delta _ { \mathrm { i n t e r n a l } }$ and $\Delta _ { \mathrm { i n t e r n a l } } ^ { \mathrm { h i n g e } }$ up to a factor of $K - 1$ , we can write

$$
\begin{array} { r l } & { \Delta _ { \mathrm { i n t e r n a l } } ( x ) \leq ( K - 1 ) \Delta _ { \mathrm { i n t e r n a l } } ^ { \mathrm { h i n g e } } ( x , t ) } \\ & { \qquad \leq ( K - 1 ) \frac { 1 } { \left[ \tilde { r } _ { \mathrm { f l a t } } ( x ) \right] _ { t } } \displaystyle \sum _ { \tau = 1 } ^ { T } [ \tilde { r } _ { \mathrm { f l a t } } ( x ) ] _ { \tau } \Delta _ { \mathrm { i n t e r n a l } } ^ { \mathrm { h i n g e } } ( x , \tau ) } \\ & { \qquad \leq T ( K - 1 ) \displaystyle \sum _ { \tau = 1 } ^ { T } [ \tilde { r } _ { \mathrm { f l a t } } ( x ) ] _ { \tau } \Delta _ { \mathrm { i n t e r n a l } } ^ { \mathrm { h i n g e } } ( x , \tau ) , } \end{array}\tag{69}
$$

where the last inequality holds because $[ \tilde { r } _ { \mathrm { f l a t } } ( x ) ] _ { \tau } \geq 0 , \Delta _ { \mathrm { i n t e r n a l } } ^ { \mathrm { h i n g e } } ( x , \tau ) \geq 0$ for all $\tau \in [ T ]$ and, by the definition of t in (59), $\begin{array} { r } { [ \tilde { r } _ { \mathrm { f l a t } } ( x ) ] _ { t } \ge \frac { 1 } { T } } \end{array}$ at the chosen classifier index t. Similarly, we have

$$
\begin{array} { r l } { \displaystyle \sum _ { r = 1 } ^ { T } \big | \tilde { r } _ { \mathrm { f i a t } } ( x ) \big | _ { T } \Delta _ { \mathrm { s e l e t } } ^ { \mathrm { h i g o r } } ( x , \tau ) = \displaystyle \sum _ { r = 1 } ^ { T } \big [ \tilde { r } _ { \mathrm { f i a t } } ( x ) \big ] _ { \tau } \left( \mathcal { R } _ { T } ^ { \mathrm { h i g o r } } ( x ; \hat { f } _ { \tau } ^ { * } ) - \mathcal { R } _ { t ^ { * } } ^ { \mathrm { h i g o r } } ( x ; \hat { f } _ { \tau } ^ { * } ) \right) } & { } \\ { \displaystyle \qquad \overset { \mathrm { ( i ) } } { = } \displaystyle \sum _ { r = 1 } ^ { T } \big [ \tilde { r } _ { \mathrm { f i a t } } ( x ) \big ] _ { \tau } \cdot \frac { K } { K - 1 } \left( \mathcal { R } _ { T } ( x ; f _ { \tau } ^ { * } ) - \mathcal { R } _ { t ^ { * } } ( x ; f _ { t ^ { * } } ^ { * } ) \right) } & { } \\ { \displaystyle \qquad \overset { \mathrm { ( i i ) } } { \geq } \big [ \tilde { r } _ { \mathrm { f i a t } } ( x ) \big ] _ { t } \cdot \frac { K } { K - 1 } \left( \mathcal { R } _ { t } ( x ; f _ { t } ^ { * } ) - \mathcal { R } _ { t ^ { * } } ( x ; f _ { t ^ { * } } ^ { * } ) \right) } & { } \\ { \displaystyle \qquad \overset { \mathrm { ( i i i ) } } { \geq } \frac { K } { T ( K - 1 ) } \Delta _ { \mathrm { s e l e t } } ( x ) , } & { } \\ { \displaystyle \qquad \overset { \mathrm { ( i i i ) } } { \geq } \frac { T } { T ( K - 1 ) } \sum _ { r = 1 } ^ { K } \big [ \tilde { r } _ { \mathrm { f i a t } } ( x ) \big ] _ { \tau } \Delta _ { \mathrm { s e l e t } } ^ { \mathrm { h i g o r } } ( x , \tau ) . } \end{array}\tag{70}
$$

where step (i) applies (65), step (ii) holds because $[ \tilde { r } _ { \mathrm { f l a t } } ( x ) ] _ { \tau } \geq 0$ for all $\tau \in [ T ]$ and, by the definition of $t ^ { * }$ in (59), $\mathcal { R } _ { \tau } ( x ; f _ { \tau } ^ { * } ) - \mathcal { R } _ { t ^ { * } } ( x ; f _ { t ^ { * } } ^ { * } ) \geq 0$ for all $\tau \in [ T ]$ , and step (iii) holds because $\begin{array} { r } { [ \tilde { r } _ { \mathrm { f l a t } } ( x ) ] _ { t } \ge \frac { 1 } { T } } \end{array}$ and $\Delta _ { \mathrm { s e l e c t } } ( x ) = \mathcal { R } _ { t } ( x ; f _ { t } ^ { * } ) - \mathcal { R } _ { t ^ { * } } ( x ; f _ { t ^ { * } } ^ { * } ) \geq 0$

Part 4: Combining all components. We have so far bounded the selection and internal excess 0-1-c risks by their excess hinge risk counterparts, respectively. The final step is to combine these components (Eq. (69) and (70)) to obtain

$$
\begin{array} { r l } & { \mathcal { R } _ { t } ( x ; f _ { t } ) - \mathcal { R } _ { t ^ { * } } ( x ; f _ { t ^ { * } } ^ { * } ) = \Delta _ { \mathrm { i n t e r n a l } } ( x ) + \Delta _ { \mathrm { s e l e c t } } ( x ) } \\ & { \qquad \leq T ( K - 1 ) \displaystyle \sum _ { \tau = 1 } ^ { T } [ \tilde { r } _ { \mathrm { f l a t } } ( x ) ] _ { \tau } \left( \Delta _ { \mathrm { i n t e r n a l } } ^ { \mathrm { h i n g e } } ( x , \tau ) + \Delta _ { \mathrm { s e l e c t } } ^ { \mathrm { h i n g e } } ( x , \tau ) \right) } \\ & { \qquad = T ( K - 1 ) \left( \mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } ( x ; \tilde { \pi } ) - \mathcal { R } _ { \mathrm { f l a t } } ^ { \mathrm { h i n g e } } ( x ; \tilde { \pi } ^ { * } ) \right) . } \end{array}\tag{71}
$$

This concludes the proof of (60).

## C Proofs for the Cascade Model

In this section, we prove the calibration result in Theorem 3.2 for the cascade hinge loss $\ell _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } }$ Similar to (42)–(43) of the flat case, we first define conditional 0-1-c and hinge risks as

$$
\begin{array} { r } { \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) : = \mathbb { E } \Big [ \mathbb { 1 } \big \{ f _ { t } ( X , Z _ { t } ) \neq Y \big \} + c _ { t } \ | \ H _ { t } = h _ { t } \Big ] , } \end{array}\tag{72}
$$

$$
\mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ) : = \mathbb { E } \left[ \ell ^ { \mathrm { h i n g e } } ( \tilde { f } _ { t } ( X , Z _ { t } ) , Y ; c _ { t } ) \mid H _ { t } = h _ { t } \right] .\tag{73}
$$

As elsewhere in the paper, we suppress the $^ { \circ } 0 - 1 - c ^ { \prime }$ superscript and use the ‘hinge’ superscript only to distinguish hinge quantities. Thus, $\mathcal { R } _ { t }$ means $\mathcal { R } _ { t } ^ { 0 - 1 - c }$ , and the same naming convention applies to other quantities throughout. The calibration proof relies on value functions, a concept standard in the dynamic programming literature which we restate below for completeness.

Definition C.1 (Value function). For a cascade classifier $\pi = ( f _ { 1 : T } , d _ { 1 : T } )$ , a loss $\in \{ 0 { \cdot } 1 { \cdot } c , \mathrm { h i n g e } \}$ and the stage index $t \in [ T ]$ , we define the stage-t value function $V _ { t } ^ { \pi , \mathrm { l o s s } } ( h _ { t } )$ as the expected cost-to-go from stage t onward, given the current history $h _ { t }$ . Formally, define

$$
\begin{array} { r } { V _ { t } ^ { \pi , \mathrm { l o s s } } ( h _ { t } ) : = ( 1 - d _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ^ { \mathrm { l o s s } } ( h _ { t } ; f _ { t } ) + d _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \pi , \mathrm { l o s s } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] , \quad \mathrm { f o r } ~ t < T } \end{array}\tag{74}
$$

and

$$
V _ { T } ^ { \pi , \mathrm { l o s s } } ( h _ { T } ) : = \mathcal { R } _ { T } ^ { \mathrm { l o s s } } ( h _ { T } ; f _ { T } ) .\tag{75}
$$

The recursion in (74) reflects the two possible actions at stage t given current history $h _ { t } \colon$ if the deferral function exits $( d _ { t } ( h _ { t } ) = 0 )$ , it incurs the immediate conditional risk $\mathcal { R } _ { t } ^ { \mathrm { l o s s } } ( h _ { t } ; f _ { t } )$ ; if it defers $( d _ { t } ( h _ { t } ) = 1 )$ , it incurs the expected future cost starting from stage $t + 1$ . At the terminal stage, there is no further stage to defer to, and the value function reduces to the immediate risk, yielding (75). Thus, $V _ { t } ^ { \pi , \mathrm { l o s s } } ( h _ { t } )$ represents the total expected risk incurred by following $\pi = ( f _ { 1 : T } , d _ { 1 : T } )$ from stage t onward, conditional on $h _ { t }$

Notation: Since the 0-1-c loss is only used with hard policies $( \mathrm { e . g . , \pi }$ and $\pi _ { \mathrm { c a s c } } ^ { * } )$ and the hinge loss only with soft policies $( \mathrm { e . g . , \tilde { \pi } a n d ~ } \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } )$ , we drop the superscript ‘loss’ from $V _ { t } ^ { \cdots , \mathrm { l o s s } } ( h _ { t } )$ for brevity, as the associated loss will be clear from context. As in Appendix B, we use LHS (resp. RHS) as short for left (resp. right) hand side; we write $p ( \cdot \mid \cdot )$ for the conditional probability mass function of $Y , \mathrm { e . g . } , p ( y \mid x ) \equiv \mathrm { P r } ( Y = y \mid X = x )$ and $p ( y ~ \vert ~ x , z _ { t } ) \equiv \mathrm { P r } ( Y = y ~ \vert ~ X = x , Z _ { t } = z _ { t } )$

Proof sketch: This section is organized as follows. We first show that the 0-1-c and hinge risks for cascades can be expressed in terms of the stage-1 value functions (Section C.1). Building on this connection, we characterize the Bayes-optimal solutions under both the 0-1-c loss and cascade hinge loss (Section C.2). Next, we establish two useful results linking hinge and 0-1-c quantities (Section C.3). Using this, we complete the proof of the calibration result in Theorem 3.2 (Section C.4).

## C.1 Connection of Risks to Expected Value Functions

In this section, we establish an essential connection between the global risk of a cascade classifier $\pi = ( f _ { 1 : T } , d _ { 1 : T } )$ and its associated value functions. Specifically, Lemma C.2 shows that both the 0-1-c risk $\mathcal { R } ( \pi )$ and hinge risk $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } )$ can be expressed as expected value functions at the initial stage for arbitrary $\pi \in \Pi _ { \mathrm { c a s c } }$ and $\tilde { \pi } \in \bar { \Pi } _ { \mathrm { c a s c } }$ , and Corollary C.3 specializes this result to their Bayes-optimal counterparts $\pi _ { \mathrm { c a s c } } ^ { * } \in \Pi _ { \mathrm { c a s c } }$ and $\tilde { \pi } _ { \mathrm { c a s c } } ^ { * } \in \widetilde { \Pi } _ { \mathrm { c a s c } }$ . These connections allow us to analyze the risks through the recursive structure of the cascade, and serves as the foundation for characterizing Bayes-optimal solutions in the next section and, ultimately, for proving calibration.

Lemma C.2 (Connection of risks to value functions for arbitrary π and π˜). For any hard cascade classifier $\pi \in \Pi _ { c a s c } ,$ we have

$$
\mathcal { R } ( \pi ) = \mathbb { E } \left[ V _ { 1 } ^ { \pi } ( H _ { 1 } ) \right] ,\tag{76}
$$

and similarly, for any soft cascade classifier $\tilde { \pi } \in \widetilde { \Pi } _ { c a s c }$ , we have

$$
\mathcal { R } _ { c a s c } ^ { h i n g e } ( \tilde { \pi } ) = \mathbb { E } \left[ V _ { 1 } ^ { \tilde { \pi } } ( H _ { 1 } ) \right] .\tag{77}
$$

Proof. Recall we write $H _ { t } = ( X , Z _ { 1 : t } )$ and $H \equiv H _ { T } = ( X , Z _ { 1 : T } )$ . Let $\pi = ( f _ { 1 : T } , r _ { \mathrm { c a s c } } )$ , where the stopping rule is defined as $r _ { \mathrm { c a s c } } ( h ) = \operatorname* { m i n } \{ t \in [ T ] : d _ { t } ( h _ { t } ) = 0 \}$ . By the definition of the 0-1-c risk $\mathcal { R } ( \pi )$ in (9)–(10) and the tower property, we can write

$$
\begin{array} { r l } { \mathcal { R } ( x ) = } & { \mathbb { E } \bigg [ \mathbb { 1 } \{ f _ { \mathrm { r o s s } } ( H ) \big ( X , Z _ { \mathrm { t o s s } } ( \theta ) \big ) \neq Y \} + c _ { \mathrm { v o s s } } ( \theta ) \bigg ] } \\ & { = \displaystyle \sum _ { \mathrm { t = 1 } } ^ { T } \mathbb { E } \bigg [ \mathbb { 1 } \{ r _ { \mathrm { r o s s } } ( H ) = t \} \cdot \big ( \mathbb { 1 } \{ f ( H ( X , Z _ { \mathrm { t o s s } } / t ) ) + \alpha \} \bigg ) \bigg ] } \\ & { = \displaystyle \sum _ { \mathrm { t = 1 } } ^ { T } \mathbb { E } \bigg [ \mathbb { E } \Big [ \mathbb { 1 } \{ r _ { \mathrm { r o s s } } ( H ) = t \} \cdot \big ( \mathbb { 1 } \{ f _ { \mathrm { t o s s } } ( X , Z _ { \mathrm { t o s s } } / t ) + \alpha \} \big ) H _ { \mathrm { t o s s } } \Big ] } \\ & { = \displaystyle \sum _ { \mathrm { t = 1 } } ^ { T } \mathbb { E } \bigg [ \bigg ( \prod _ { \mathrm { t = 1 } } ^ { T } \delta ( H _ { \mathrm { r o s s } } ) \bigg ) \big ( 1 - d _ { \mathrm { t o t } } H _ { \mathrm { t o s s } } \big ) \cdot \mathbb { E } \big [ \mathbb { 1 } \{ f _ { \mathrm { t o s s } } ( X , Z _ { \mathrm { t o s s } } / t ) + \alpha \} \big ] H _ { \mathrm { t o } s } \bigg ] } \\ & { = \displaystyle \sum _ { \mathrm { t = 1 } } ^ { T } \mathbb { E } \bigg [ \bigg ( \prod _ { \mathrm { t = 1 } } ^ { T } d _ { \mathrm { t o s s } } \bigg ) \Big ( 1 - d _ { \mathrm { t o t } } ( H _ { \mathrm { t o s s } } ) \Big ) \cdot \mathbb { E } _ { \mathrm { t } } \big ( \mathbb { 1 } \{ f _ { \mathrm { t o s s } } ( H _ { \mathrm { t o s s } } / t ) \Big ) } \\ &  = \displaystyle \sum _ { \mathrm { t = 1 } } ^ { T } \mathbb { E } \bigg [ \bigg ( \prod _ { \mathrm { t = 1 } } ^ { T } \end{array}
$$

Therefore, to prove (76), it sufices to show that

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \left( \prod _ { \tau = 1 } ^ { t - 1 } d _ { \tau } ( H _ { \tau } ) \right) \left( 1 - d _ { t } ( H _ { t } ) \right) \cdot { \mathcal R } _ { t } ( H _ { t } ; f _ { t } ) \right] = \mathbb { E } \left[ V _ { 1 } ^ { \pi } ( H _ { 1 } ) \right] ,\tag{78}
$$

which in turn implies that, it sufices to show that for all $i \in [ T ]$ 2

$$
\mathbb { E } \left[ \sum _ { t = i } ^ { T } \left( \prod _ { \tau = i } ^ { t - 1 } d _ { \tau } ( H _ { \tau } ) \right) \left( 1 - d _ { t } ( H _ { t } ) \right) \cdot { \mathcal R } _ { t } ( H _ { t } ; f _ { t } ) \Bigm | H _ { i } \right] = V _ { i } ^ { \pi } ( H _ { i } ) ,\tag{79}
$$

We prove (79) using backward induction.

Base case $i = T \colon$ To show (79) holds for $i = T$ , note that by the definition of the stage-T value function in (75),

$$
V _ { T } ^ { \pi } ( H _ { T } ) = \mathcal { R } _ { T } ( H _ { T } ; f _ { T } ) = \mathbb { E } \left[ \sum _ { t = T } ^ { T } \left( \prod _ { \tau = T } ^ { t - 1 } d _ { \tau } ( H _ { \tau } ) \right) \left( 1 - d _ { t } ( H _ { t } ) \right) \cdot \mathcal { R } _ { t } ( H _ { t } ; f _ { t } ) \right] ,
$$

where we used the fact that for $\begin{array} { r } { a > b , \prod _ { \tau = a } ^ { b } d _ { \tau } ( H _ { \tau } ) = 1 } \end{array}$

Backward induction: Now, suppose (79) holds for $i = s + 1$ , we next prove it holds for $i = s$ Using the recursive Bellman definition of the value function (74), we can write

$$
\begin{array} { r l } & { V _ { s } ^ { \pi } ( H _ { s } ) = ( 1 - { d _ { s } } ( H _ { s } ) ) \mathcal { R } _ { s } ( H _ { s } ; f _ { s } ) + { d _ { s } } ( H _ { s } ) \mathbb { E } \left[ V _ { s + 1 } ^ { \pi } ( H _ { s + 1 } ) \mid H _ { s } \right] } \\ & { \quad = ( 1 - { d _ { s } } ( H _ { s } ) ) \mathcal { R } _ { s } ( H _ { s } ; f _ { s } ) + } \\ & { \quad \quad \quad \quad d _ { s } ( H _ { s } ) \mathbb { E } \left[ \mathbb { E } \Bigg [ \displaystyle \sum _ { t = s + 1 } ^ { T } \left( \displaystyle \prod _ { \tau = s + 1 } ^ { t - 1 } { d _ { \tau } ( H _ { \tau } ) } \right) \left( 1 - { d _ { i } } ( H _ { t } ) \right) \cdot \mathcal { R } _ { t } ( H _ { t } ; f _ { t } ) \mid H _ { s + 1 } \Bigg ] \mid H _ { s } \right] } \\ & { \quad \quad = ( 1 - { d _ { s } } ( H _ { s } ) ) \mathcal { R } _ { \sigma } ( H _ { s } ; f _ { s } ) + } \\ & { \quad \quad \quad \quad d _ { s } ( H _ { s } ) \mathbb { E } \left[ \displaystyle \sum _ { t = s + 1 } ^ { T } \left( \displaystyle \prod _ { \tau = s + 1 } ^ { t - 1 } { d _ { \tau } ( H _ { \tau } ) } \right) \left( 1 - { d _ { i } } ( H _ { t } ) \right) \cdot \mathcal { R } _ { t } ( H _ { t } ; f _ { t } ) \mid H _ { s } \right] } \\ & { \quad \quad \quad = \mathbb { E } \left[ \displaystyle \sum _ { t = s } ^ { T } \left( \displaystyle \prod _ { \tau = s } ^ { t - 1 } { d _ { \tau } ( H _ { \tau } ) } \right) \left( 1 - { d _ { \tau } ( H _ { t } ) } \right) \cdot \mathcal { R } _ { t } ( H _ { t } ; f _ { t } ) \mid H _ { s } \right] , } \end{array}
$$

where the last step absorbs the $t = s$ term into the summation, yielding (79) for $i = s .$

Conclusion: Applying this backward induction argument down from $i = T$ −1 to $i = 1$ yields (79) for all $i \in [ T ]$ . Setting $i = 1$ in (79) and taking total expectation over $H _ { 1 }$ proves (78), which gives (76).

In the remainder of the proof, we establish (77). Let $\tilde { \pi } = ( \tilde { f } _ { 1 : T } , \tilde { d } _ { 1 : T } )$ . By definition of $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } )$ in (18) and tower property,

$$
\begin{array} { l } { \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) = \mathbb { E } \left[ \ell _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( X , Z _ { 1 : T } , Y ; \tilde { \pi } , c ) \right] } \\ { = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \left( \prod _ { \tau = 1 } ^ { t - 1 } \tilde { d } _ { \tau } ( H _ { \tau } ) \right) \left( 1 - \tilde { d } _ { t } ( H _ { t } ) \right) \mathbb { E } \left[ \ell ^ { \mathrm { h i n g e } } ( \tilde { f } _ { t } ( X , Z _ { t } ) , Y ; c _ { t } ) \mid H _ { t } \right] \right] } \\ { = \displaystyle \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \left( \prod _ { \tau = 1 } ^ { t - 1 } \tilde { d } _ { \tau } ( H _ { \tau } ) \right) \left( 1 - \tilde { d } _ { t } ( H _ { t } ) \right) \cdot \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( H _ { t } ; \tilde { f } _ { t } ) \right] , } \end{array}
$$

so to prove (77), it sufices to prove

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \left( \prod _ { \tau = 1 } ^ { t - 1 } \tilde { d } _ { \tau } ( H _ { \tau } ) \right) \left( 1 - \tilde { d } _ { t } ( H _ { t } ) \right) \cdot \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( H _ { t } ; \tilde { f } _ { t } ) \right] = \mathbb { E } \left[ V _ { 1 } ^ { \tilde { \pi } } ( H _ { 1 } ) \right] ,
$$

which in turn implies it sufices to prove for all $i \in [ T ]$ ，

$$
\mathbb { E } \left[ \sum _ { t = i } ^ { T } \left( \prod _ { \tau = i } ^ { t - 1 } \tilde { d } _ { \tau } ( H _ { \tau } ) \right) \left( 1 - \tilde { d } _ { t } ( H _ { t } ) \right) \cdot \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( H _ { t } ; \tilde { f } _ { t } ) \mid H _ { i } \right] = V _ { i } ^ { \tilde { \pi } } ( H _ { i } ) .\tag{80}
$$

This is analogous to (79). The proof works verbatim because the Bellman recursion and tower rule do not depend on whether $\tilde { d } _ { t }$ lies in {0, 1} or [0, 1]. □

Lemma C.2 connects the risks to the value functions for arbitrary cascade classifiers $\pi$ and π˜. Specializing this result to the optimal cascade classifiers $\pi _ { \mathrm { c a s c } } ^ { * }$ and $\tilde { \pi } _ { \mathrm { c a s c } } ^ { * }$ yields a corresponding connection between the Bayes risks and the value functions of the optimal cascade classifiers. This is stated in Corollary C.3.

Corollary C.3 (Connection of risks to value functions for Bayes-optimal $\pi _ { \mathrm { c a s c } } ^ { * }$ and $\tilde { \pi } _ { \mathrm { c a s c } } ^ { * } )$ . Let $\begin{array} { r } { \pi _ { c a s c } ^ { * } = \left( f _ { 1 : T } ^ { * } , d _ { 1 : T } ^ { * } \right) \in \arg \operatorname* { m i n } _ { \pi \in \Pi _ { c a s c } } \mathcal { R } ( \pi ) } \end{array}$ be as defined in Lemma $C . 4 ,$ then

$$
\mathcal { R } _ { c a s c } ^ { * } = \mathbb { E } \left[ V _ { 1 } ^ { \pi _ { c a s c } ^ { * } } ( H _ { 1 } ) \right] .
$$

Similarly, let $\begin{array} { r } { \tilde { \pi } _ { c a s c } ^ { * } = ( \tilde { f } _ { 1 : T } ^ { * } , \tilde { d } _ { 1 : T } ^ { * } ) \in \arg \operatorname* { m i n } _ { \tilde { \pi } \in \widetilde { \Pi } _ { c a s c } } \mathcal { R } _ { c a s c } ^ { h i n g e } ( \tilde { \pi } ) } \end{array}$ be as defined in Lemma C.5, then

$$
\mathcal { R } _ { c a s c } ^ { h i n g e * } = \mathbb { E } \left[ V _ { 1 } ^ { \tilde { \pi } _ { c a s c } ^ { * } } ( H _ { 1 } ) \right] .
$$

Proof. By (76), which relates R to $V _ { 1 } ^ { \pi }$ for an arbitrary hard cascade classifier π, we can write

$$
\mathcal { R } _ { \mathrm { c a s c } } ^ { * } : = \operatorname* { m i n } _ { \pi \in \Pi _ { \mathrm { c a s c } } } \mathcal { R } ( \pi ) = \mathcal { R } ( \pi _ { \mathrm { c a s c } } ^ { * } ) = \mathbb { E } \left[ V _ { 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { 1 } ) \right] .
$$

By (77), which relates $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } }$ to $V _ { 1 } ^ { \tilde { \pi } }$ for an arbitrary soft cascade classifier $\tilde { \pi } .$ , we can write

$$
\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e * } } : = \operatorname* { m i n } _ { \tilde { \pi } \in \tilde { \Pi } _ { \mathrm { c a s c } } } \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) = \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } ) = \mathbb { E } \left[ V _ { 1 } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } } ( H _ { 1 } ) \right] .
$$

## C.2 Bayes-Optimal Solutions

Using the connections of risks to expected value functions proven in the previous section, we now characterize the Bayes-optimal solutions under the 0-1-c loss and cascade hinge loss, respectively.

Lemma C.4 (Bayes-optimal solution in $\Pi _ { \mathrm { c a s c } }$ for 0-1-c loss). Assume $Y \perp Z _ { 1 : t - 1 } \mid ( X , Z _ { t } )$ as in (5). For the 0-1-c risk R with cost vector $\boldsymbol { c } = [ \kappa _ { 1 } , \kappa _ { 1 } + \kappa _ { 2 } , . . . , \sum _ { t = 1 } ^ { T } \kappa _ { t } ] ^ { \top }$ , there exists a Bayes-optimal cascade classifier

$$
\pi _ { c a s c } ^ { * } = ( f _ { 1 : T } ^ { * } , r _ { c a s c } ^ { * } ) \in \underset { \pi \in \Pi _ { c a s c } } { \arg \operatorname* { m i n } } \mathcal { R } ( \pi ) ,\tag{81}
$$

where $f _ { 1 : T } ^ { * }$ satisfies (30), and $r _ { c a s c } ^ { * } ( h ) : = \operatorname* { m i n } \{ t \in [ T ] : d _ { t } ^ { * } ( h _ { t } ) = 0 \}$ is induced by the deferral rules $d _ { 1 : T } ^ { * }$ given by

$$
\begin{array} { r } { d _ { t } ^ { * } ( h _ { t } ) = \mathbb { 1 } \Big \{ \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) > \mathbb { E } \Big [ V _ { t + 1 } ^ { \pi _ { \omega s c } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \Big ] \Big \} \quad \forall t < T , \quad a n d \quad d _ { T } ^ { * } ( h _ { T } ) = 0 , } \end{array}\tag{82}
$$

where

$$
\mathscr { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) = 1 - \operatorname* { m a x } _ { y \in [ K ] } p ( y \mid x , z _ { t } ) + c _ { t } , \quad \forall t \in [ T ]\tag{83}
$$

is the Bayes risk of exiting at stage t conditional on $H _ { t } = h _ { t }$

Proof. Throughout, we write $H _ { t } = ( X , Z _ { 1 : t } )$ and $H \equiv H _ { T } = ( X , Z _ { 1 : T } )$ . We first optimize $f _ { 1 : T }$ with $d _ { 1 : T }$ fixed and then optimize $d _ { 1 : T }$ with $f _ { 1 : T }$ fixed.

Part 1: Proof for the Bayes-optimal $f _ { 1 : T } ^ { * }$ . We first minimize $\mathcal { R } ( \pi )$ with respect to $f _ { 1 : T }$ for an arbitrary routing function $r _ { \mathrm { { c a s c } } } .$ or equivalently, an arbitrary $d _ { 1 : T }$ . Fix $d _ { 1 : T }$ , we can expand the 0-1-c risk as follows:

$$
\begin{array} { r l } & { \mathcal { R } ( \pi ) = \mathbb { E } [ \ell ( X , Z _ { 1 : T } , Y ; \pi , c ) ] } \\ & { \qquad = \mathbb { E } \bigg [ \mathbb { 1 } \left\{ f _ { \mathrm { r a s c } ( H ) } ( X , Z _ { \mathrm { r e a s c } ( H ) } ) \neq Y \right\} + c _ { \mathrm { r e a s c } ( H ) } \bigg ] } \\ & { \qquad = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \Big [ \mathbb { 1 } \left\{ r _ { \mathrm { c a s c } } ( H ) = t \right\} \cdot \Big ( \mathbb { 1 } \left\{ f _ { t } ( X , Z _ { t } ) \neq Y \right\} + c _ { t } \Big ) \Big ] } \\ & { \qquad \stackrel { \mathrm { ( i ) } } { = } \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \Bigg [ \mathbb { 1 } \left\{ r _ { \mathrm { c a s c } } ( H ) = t \right\} \cdot \underbrace { \Big ( \mathbb { E } \Big [ \mathbb { 1 } \left\{ f _ { t } ( X , Z _ { t } ) \neq Y \right\} | H _ { t } \Big ] + c _ { t } \Big ) } _ { = \mathcal { R } _ { t } ( H _ { t } ; f _ { t } ) } \Bigg ] , } \end{array}
$$

where $\mathcal { R } _ { t } ( H _ { t } ; f _ { t } )$ is the conditional risk of exiting at stage t given $H _ { t } ,$ as defined in (72). Note step (i) holds because $\mathbb { 1 } \{ r _ { \mathrm { c a s c } } ( H ) = t \}$ is measurable with respect to $H _ { t }$ . Minimizing $\mathcal { R } ( \pi )$ with respect to $f _ { t }$ reduces to minimizing $\mathcal { R } _ { t } ( H _ { t } ; f _ { t } )$ pointwise in $H _ { t }$ . Fix $h _ { t } \in \mathcal { H } _ { t }$ , using $Y \perp Z _ { 1 : t - 1 } \mid ( X , Z _ { t } )$ from (5), we have

$$
\begin{array} { r l } & { { \mathcal R } _ { t } ( h _ { t } ; f _ { t } ) = { \mathbb E } \Big [ \mathbb { 1 } \big \{ f _ { t } ( X , Z _ { t } ) \neq Y \big \} \mid H _ { t } = h _ { t } \Big ] + c _ { t } } \\ & { \quad \quad \quad \stackrel { \mathrm { ( i ) } } { = } { \mathbb E } \Big [ \mathbb { 1 } \big \{ Y \neq f _ { t } ( x , z _ { t } ) \big \} \mid X = x , Z _ { t } = z _ { t } \Big ] + c _ { t } } \\ & { \quad \quad \quad = 1 - p \big ( f _ { t } ( x , z _ { t } ) \mid x , z _ { t } \big ) + c _ { t } } \end{array}\tag{84}
$$

where recall $p ( \cdot \mid x , z _ { t } )$ denotes the conditional probability of Y given $( X , Z _ { t } ) = ( x , z _ { t } )$ , and step (i) here holds because $Y \perp Z _ { 1 : t - 1 } \mid ( X , Z _ { t } )$ by the assumption in (5). Hence, minimizing $\textstyle { \mathcal { R } } _ { t } ( h _ { t } ; f _ { t } )$ pointwise in $h _ { t }$ reduces to minimizing (84) pointwise in $( x , z _ { t } )$ , that is,

$$
f _ { t } ^ { * } ( x , z _ { t } ) \in \underset { y \in [ K ] } { \arg \operatorname* { m i n } } 1 - p \big ( y \mid x , z _ { t } \big ) + c _ { t } = \underset { y \in [ K ] } { \arg \operatorname* { m a x } } p \big ( y \mid x , z _ { t } \big ) .\tag{85}
$$

This proves that $f _ { 1 : T } ^ { * }$ satisfy (30). Substituting (85) into (84) yields

$$
\mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) = 1 - \operatorname* { m a x } _ { y \in [ K ] } p ( y \mid x , z _ { t } ) + c _ { t } ,
$$

as stated in (83).

Note the result in (85) holds uniformly for any routing policy $r _ { \mathrm { { c a s c } } } .$ , or equivalently, any $d _ { 1 : T }$ . In other words, it characterizes the Bayes-optimal classifiers $f _ { 1 : T } ^ { * }$ independently of how $d _ { 1 : T }$ is chosen. Therefore, it remains only to characterize the Bayes-optimal deferral rules $d _ { 1 : T } ^ { * }$ with the classifiers fixed at $f _ { 1 : T } = f _ { 1 : T } ^ { * }$

Part 2: Proof for the Bayes-optimal $d _ { 1 : T } ^ { * }$ . We now minimize $\mathcal { R } ( \pi )$ with respect to $d _ { 1 : T }$ with $f _ { 1 : T } = f _ { 1 : T } ^ { * }$ held fixed. Since $\pi = ( f _ { 1 : T } , d _ { 1 : T } )$ , we write $\mathcal { R } ( f _ { 1 : T } , d _ { 1 : T } )$ to mean $\mathcal { R } ( \pi )$ and $V _ { t } ^ { f _ { 1 : T } , d _ { 1 : T } } ( H _ { t } )$ to mean $V _ { t } ^ { \pi } ( H _ { t } )$ . By (76) which links R to $V _ { 1 }$ , we have

$$
\begin{array} { r } { d _ { 1 : T } ^ { * } \in \underset { d _ { 1 : T } } { \arg \operatorname* { m i n } } \mathcal { R } ( f _ { 1 : T } ^ { * } , d _ { 1 : T } ) = \underset { d _ { 1 : T } } { \arg \operatorname* { m i n } } \mathbb { E } \left[ V _ { 1 } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : T } } ( H _ { 1 } ) \right] , } \end{array}\tag{86}
$$

where the minimum is over $d _ { t } : \mathcal { H } _ { t } \to \{ 0 , 1 \}$ for $t \in [ T ]$ . By the recursive definition (74) of the value function, we can unroll (86) to optimize $d _ { t }$ one at a time, treating the earlier functions $d _ { 1 : t - 1 }$ as arbitrary. Consequently, the optimal $d _ { t } ^ { * }$ is pointwise optimal for every trajectory $d _ { 1 : t - 1 }$ , not just for the optimal trajectory $d _ { 1 : t - 1 } ^ { * }$ , and composing the pointwise minimizers $d _ { t } ^ { * }$ across $t \in [ T ]$ yields $d _ { 1 : T } ^ { * }$

In the rest of the proof, we characterize $d _ { t } ^ { * }$ using backward induction starting with $t = T$ Base case $t = T$ : Since $T$ is the terminal stage, $d _ { T } ^ { * } ( H _ { T } )$ is fixed to 0 (exit), as stated in (82). By definition (75) of the terminal-stage value function, $V _ { T } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : T - 1 } , d _ { T } ^ { * } } ( H _ { T } ) = \mathcal { R } _ { T } ( H _ { T } ; f _ { T } ^ { * } )$ for any arbitrary $d _ { 1 : T - 1 }$ , so we can write

$$
V _ { T } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : T - 1 } , d _ { T } ^ { * } } ( H _ { T } ) = V _ { T } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { T } ) \quad \mathrm { w h e r e } \quad \pi _ { \mathrm { c a s c } } ^ { * } = ( f _ { 1 : T } ^ { * } , d _ { 1 : T } ^ { * } ) .\tag{87}
$$

Backward induction: Using (87), we now prove the induction step. Suppose that for some $t \in \{ 1 , \dots T - 1 \}$ ,

$$
V _ { t + 1 } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : t } , d _ { t + 1 : T } ^ { * } } ( H _ { t + 1 } ) = V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } )\tag{88}
$$

for any arbitrary $d _ { 1 : t }$ , then

$$
\begin{array} { r } { d _ { t } ^ { * } ( h _ { t } ) = \mathbb { 1 } \left\{ \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) > \mathbb { E } \Big [ V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \ | \ H _ { t } = h _ { t } \Big ] \right\} . } \end{array}\tag{89}
$$

We fix $d _ { t + 1 : T } = d _ { t + 1 : T } ^ { \ast }$ and unroll the value function $V _ { 1 }$ in the objective (86) to optimize over $d _ { t }$ This gives

$$
\begin{array} { r l } & { \mathbb { E } \Big [ V _ { 1 } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : t } , d _ { t + 1 : T } ^ { * } } ( H _ { 1 } ) \Big ] = \underbrace { \mathbb { E } \Bigg [ ( 1 - d _ { 1 } ( H _ { 1 } ) ) \mathcal { R } _ { 1 } ( H _ { 1 } ; f _ { 1 } ^ { * } ) + d _ { 1 } ( H _ { 1 } ) \mathbb { E } \Big [ \cdots + \mathbb { E } \Big [ \cdots + d _ { t - 1 } ( H _ { t - 1 } ) \Big ] } _ { \mathrm { T e r m ~ I } } } \\ & { \qquad \mathbb { E } \Big [ V _ { t } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : t } , d _ { t + 1 : T } ^ { * } } ( H _ { t } ) \underbrace { \vert ~ H _ { t - 1 } \vert ~ \vert ~ H _ { t - 2 } \vert ~ \vert \cdots \vert ~ H _ { 1 } \Big ] } _ { \mathrm { T e r m ~ I I } } \Bigg ] . } \end{array}\tag{90}
$$

To clarify, Term I collects the nested expectations taken successively over $H _ { t - 1 } | H _ { t - 2 } , H _ { t - 2 } | H _ { t - 3 } , . . .$ down to $H _ { 2 } | H _ { 1 }$ , and finally over $H _ { 1 }$ through the outermost expectation, along with terms depending on $d _ { 1 } , \dotsc , d _ { t - 1 }$ . Term II collects the corresponding chain of conditionings from $H _ { t - 1 }$ down to $H _ { 1 }$ Hence, the part of (90) that depends on $d _ { t }$ is

$$
\begin{array} { r l } & { \mathbb { E } \left[ d _ { 1 } ( H _ { 1 } ) \mathbb { E } \left[ d _ { 2 } ( H _ { 2 } ) \dots \mathbb { E } \left[ d _ { t - 1 } ( H _ { t - 1 } ) \mathbb { E } \left[ V _ { t } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : t } , d _ { t + 1 : T } ^ { * } } ( H _ { t } ) \mid H _ { t - 1 } \right] \mid H _ { t - 2 } \right] \mid \cdots \mid H _ { 1 } \right] \right] } \\ & { = \mathbb { E } \left[ \displaystyle \prod _ { \tau = 1 } ^ { t - 1 } d _ { \tau } ( H _ { \tau } ) V _ { t } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : t } , d _ { t + 1 : T } ^ { * } } ( H _ { t } ) \right] , } \end{array}\tag{91}
$$

where the equality uses the tower rule of conditional expectation, and $V _ { t } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : t } , d _ { t + 1 : T } ^ { * } } ( H _ { t } )$ is the innermost term of (90), which by definition, is

$$
\begin{array} { r l } & { V _ { t } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : t } , d _ { t + 1 : T } ^ { * } } ( H _ { t } ) = \big ( 1 - d _ { t } ( H _ { t } ) \big ) \mathcal { R } _ { t } ( H _ { t } ; f _ { t } ^ { * } ) + d _ { t } ( H _ { t } ) \mathbb { E } \Big [ V _ { t + 1 } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : t } , d _ { t + 1 : T } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } \Big ] } \\ & { \qquad \overset { \mathrm { ( i ) } } { = } \big ( 1 - d _ { t } ( H _ { t } ) \big ) \mathcal { R } _ { t } ( H _ { t } ; f _ { t } ^ { * } ) + d _ { t } ( H _ { t } ) \mathbb { E } \Big [ V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } \Big ] , } \end{array}\tag{92}
$$

where (i) holds by the induction hypothesis (88). Hence, (90)–(91) imply that the Bayes-optimal deferral function $d _ { t } ^ { * }$ should satisfy

$$
d _ { t } ^ { * } \in \arg \operatorname* { m i n } _ { d _ { t } } \mathbb { E } \left[ V _ { 1 } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : t } , d _ { t + 1 : T } ^ { * } } ( H _ { 1 } ) \right] = \arg \operatorname* { m i n } _ { d _ { t } } \mathbb { E } \left[ \prod _ { \tau = 1 } ^ { t - 1 } d _ { \tau } ( H _ { \tau } ) V _ { t } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : t } , d _ { t + 1 : T } ^ { * } } ( H _ { t } ) \right] .\tag{93}
$$

Since the $\textstyle \prod _ { \tau = 1 } ^ { t - 1 } d _ { \tau } ( H _ { \tau } )$ term is independent of $d _ { t }$ and always non-negative, (93) reduces pointwise in $h _ { t } \in \mathcal { H } _ { t }$ to

$$
\begin{array} { r l } & { d _ { t } ^ { * } ( h _ { t } ) \in \underset { d \in \{ 0 , 1 \} } { \arg \operatorname* { m i n } } V _ { t } ^ { f _ { 1 : T } ^ { * } , d _ { 1 : t } , d _ { t + 1 : T } ^ { * } } ( h _ { t } ) } \\ & { \qquad = \underset { d \in \{ 0 , 1 \} } { \arg \operatorname* { m i n } } \left( 1 - d \right) \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) + d \mathbb { E } \Big [ V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \Big ] } \\ & { \qquad = \mathbb { 1 } \left\{ \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) > \mathbb { E } \Big [ V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \Big ] \right\} . } \end{array}\tag{94}
$$

This proves (89), that is, (82) for stage t. Substituting this Bayes-optimal $d _ { t } ^ { * }$ back into the recursion (92) yields:

$$
\begin{array} { r } { V _ { t } ^ { f _ { 1 : T } ^ { \star } , d _ { 1 : t - 1 } , d _ { t : T } ^ { \star } } ( H _ { t } ) = \big ( 1 - d _ { t } ^ { \star } ( H _ { t } ) \big ) \mathcal { R } _ { t } ( H _ { t } ; f _ { t } ^ { \star } ) + d _ { t } ^ { \star } ( H _ { t } ) \mathbb { E } \Big [ V _ { t + 1 } ^ { \pi _ { \operatorname { t a x } } ^ { \star } } ( H _ { t + 1 } ) \mid H _ { t } \Big ] \stackrel { ( \mathrm { i } ) } { = } V _ { t } ^ { \pi _ { \operatorname { t a x } } ^ { \star } } ( H _ { t } ) , } \end{array}\tag{95}
$$

for any arbitrary $d _ { 1 : t - 1 }$ , where (i) holds by definition of $V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } }$ . Thus, this establishes (88) for stage $t ,$ completing the induction step.

Conclusion: By backward induction, we obtain (82) for all $t \in [ T ]$

Connection of $d _ { 1 : T } ^ { * }$ to optimal stopping: Identify the filtration $\sigma ( H _ { t } )$ , the adapted reward process $\{ G _ { t } \} _ { t = 1 } ^ { T }$ with $G _ { t } = - \mathcal { R } _ { t } ( H _ { t } ; f _ { t } ^ { * } )$ , and let $\mathrm { T } = \{ \tau : \tau \in [ T ] , \{ \tau = t \} \in \sigma ( H _ { t } ) \}$ denote the set of stopping times. Then the deferral rule $d _ { 1 : T } ^ { * }$ in (82), which minimizes the risk $\mathcal { R } ( f _ { 1 : T } ^ { * } , d _ { 1 : T } )$ coincides with the optimal stopping rule for the finite-horizon problem $\operatorname { s u p } _ { \tau \in \mathrm { T } } \mathbb { E } [ G _ { \tau } ]$ [Shiryaev, 2008], which maximizes the reward, and our value function is its dynamic programming recursion.

While Lemma C.4 characterizes the Bayes-optimal solution under the target 0-1-c loss, the next lemma is in parallel and characterizes the Bayes-optimal solution under the cascade hinge loss.

Lemma C.5 (Bayes-optimal solution in $\widetilde { \Pi } _ { \mathrm { c a s c } }$ for hinge loss). Assume $Y \perp Z _ { 1 : t - 1 } \mid ( X , Z _ { t } )$ as in (5). There exists a Bayes-optimal cascade classifier

$$
\tilde { \pi } _ { c a s c } ^ { * } = ( \tilde { f } _ { 1 : T } ^ { * } , \tilde { d } _ { 1 : T } ^ { * } ) \in \arg \operatorname* { m i n } _ { \tilde { \pi } \in \tilde { \Pi } _ { c a s c } } \mathcal { R } _ { c a s c } ^ { h i n g e } ( \tilde { \pi } ) ,
$$

where, with $f _ { t } ^ { * }$ defined as in (30),

$$
\left[ \tilde { f } _ { t } ^ { * } ( x , z _ { t } ) \right] _ { k } = \left\{ \begin{array} { l l } { 1 } & { f o r k = f _ { t } ^ { * } ( x , z _ { t } ) } \\ { - \frac { 1 } { K - 1 } } & { o t h e r w i s e } \end{array} \right. \quad \forall t \in [ T ] ,\tag{96}
$$

and, with $\mathcal { R } _ { t } ^ { h i n g e } ( h _ { t } ; \tilde { f } _ { t } )$ defined as in (73),

$$
\tilde { d } _ { t } ^ { * } ( h _ { t } ) = \mathbb { 1 } \left\{ \mathcal { R } _ { t } ^ { h i n g e } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) > \mathbb { E } \Big [ V _ { t + 1 } ^ { \tilde { \pi } _ { c a s c } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \Big ] \right\} \quad \forall t < T \quad a n d \quad \tilde { d } _ { T } ^ { * } \equiv 0 .\tag{97}
$$

Proof. We minimize $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } )$ in two stages: first over $\tilde { f } _ { 1 : T }$ with $\tilde { d } _ { 1 : T }$ held fixed, then over $\tilde { d } _ { 1 : T }$ with $\tilde { f } _ { 1 : T } = \tilde { f } _ { 1 : T } ^ { * }$ fixed.

Part 1: Proof for the Bayes-optimal $\tilde { f } _ { 1 : T } ^ { * }$ . Since $p ( y \mid x , z _ { 1 : t } ) = p ( y \mid x , z _ { t } )$ , the result in (96) follows from [Tarigan and van de Geer, 2008, Lemma 1].

Part 2: Proof for the Bayes-optimal $\tilde { d } _ { 1 : T } ^ { * } .$ . We now prove (97). Since $\tilde { \pi } = ( \tilde { f } _ { 1 : T } , \tilde { d } _ { 1 : T } )$ , we write $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { f } _ { 1 : T } , \tilde { d } _ { 1 : T } )$ to mean $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } )$ , and $V _ { t } ^ { \tilde { f } _ { 1 : T } , \tilde { d } _ { 1 : T } } ( H _ { t } )$ to mean $V _ { t } ^ { \tilde { \pi } } ( H _ { t } )$ . By definition of $\tilde { \pi } _ { \mathrm { c a s c } } ^ { * }$ and using (77) which links $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } }$ and $V _ { 1 } ^ { \tilde { \pi } }$ , we can write

$$
\tilde { d } _ { 1 : T } ^ { * } \in \mathop { \mathrm { a r g } } \mathop { \operatorname* { m i n } } _ { \tilde { d } _ { 1 : T } } \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { f } _ { 1 : T } ^ { * } , \tilde { d } _ { 1 : T } ) = \mathop { \mathrm { a r g } } _ { \tilde { d } _ { 1 : T } } \mathop { \operatorname* { m i n } } _ { \mathbb { X } } \left[ V _ { 1 } ^ { \tilde { f } _ { 1 : T } ^ { * } , \tilde { d } _ { 1 : T } } ( H _ { 1 } ) \right] ,\tag{98}
$$

where each minimum is taken over $\tilde { d } _ { t } : \mathcal { H } _ { t } \to [ 0 , 1 ]$ for $t \in [ T ]$ . The proof proceeds by backward induction, following the same steps as those in Part 2 of the proof of Lemma C.4, see (86)–(95). For all $t < T$ , this yields

$$
\begin{array} { r l } & { \tilde { d } _ { t } ^ { * } ( h _ { t } ) \in \underset { \tilde { d } \in [ 0 , 1 ] } { \mathrm { a r g } } \operatorname* { m i n } \left( 1 - \tilde { d } \right) \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) + \tilde { d } \mathbb { E } \left[ V _ { t + 1 } ^ { \tilde { f } _ { 1 : T } ^ { * } \tilde { d } _ { 1 : t } , \tilde { d } _ { t + 1 : T } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] } \\ & { \qquad = \underset { \tilde { d } \in [ 0 , 1 ] } { \mathrm { a r g } } \operatorname* { m i n } \left( 1 - \tilde { d } \right) \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) + \tilde { d } \mathbb { E } \left[ V _ { t + 1 } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] . } \end{array}
$$

Since this objective is linear in $\tilde { d } \in [ 0 , 1 ]$ , the minimum is attained at an endpoint, that is,

$$
\tilde { d } _ { t } ^ { * } ( h _ { t } ) = \mathbb { 1 } \left\{ \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) > \mathbb { E } \Big [ V _ { t + 1 } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \Big ] \right\} ,
$$

yielding (97).

## C.3 Connections Between 0-1-c and Hinge Quantities

In this section we establish two useful results linking hinge and 0-1-c quantities, which will help prove calibration in the next section.

Lemma C.6 (Hinge and 0-1-c Bayes quantities difer by a factor of $\frac { K } { K - 1 } )$ . Assume $Y \perp Z _ { 1 : t - 1 } \mid$ $( X , Z _ { t } )$ as in (5). Let $\pi _ { c a s c } ^ { * } = ( f _ { 1 : T } ^ { * } , d _ { 1 : T } ^ { * } )$ and $\tilde { \pi } _ { c a s c } ^ { * } = ( \tilde { f } _ { 1 : T } ^ { * } , \tilde { d } _ { 1 : T } ^ { * } )$ denote the Bayes-optimal cascade classifiers from Lemmas $C . 4 ^ { - C . 5 }$ under the 0-1-c loss and the hinge loss, respectively. Define the conditional Bayes risks

$$
\begin{array} { r } { \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) : = \mathbb { E } \Big [ \mathbb { 1 } \big \{ f _ { t } ^ { * } ( X , Z _ { t } ) \neq Y \big \} + c _ { t } \mid H _ { t } = h _ { t } \Big ] , } \end{array}\tag{99}
$$

$$
\mathcal { R } _ { t } ^ { h i n g e } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) : = \mathbb { E } \left[ \ell ^ { h i n g e } \big ( \tilde { f } _ { t } ^ { * } ( X , Z _ { t } ) , Y ; c _ { t } \big ) \mid H _ { t } = h _ { t } \right] ,\tag{100}
$$

then

$$
\begin{array} { r } { \mathcal { R } _ { t } ^ { h i n g e } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) = \frac { K } { K - 1 } \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) \qquad \forall t \in [ T ] . } \end{array}\tag{101}
$$

Moreover, the stage-t value functions of $\tilde { \pi } _ { c a s c } ^ { * }$ and $\pi _ { c a s c } ^ { * }$ satisfy

$$
\begin{array} { r } { V _ { t } ^ { \tilde { \pi } _ { c a s c } ^ { * } } ( h _ { t } ) = \frac { K } { K - 1 } V _ { t } ^ { \pi _ { c a s c } ^ { * } } ( h _ { t } ) \qquad \forall t \in [ T ] . } \end{array}\tag{102}
$$

Proof. By Lemmas C.4–C.5 together with Lemma B.3, we have

$$
\begin{array} { r l } & { \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) = 1 - \underset { y \in [ K ] } { \operatorname* { m a x } } p ( y \mid x , z _ { t } ) + c _ { t } , } \\ & { } \\ & { \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) = \frac { K } { K - 1 } \left( 1 - \underset { y \in [ K ] } { \operatorname* { m a x } } p ( y \mid x , z _ { t } ) + c _ { t } \right) , } \end{array}
$$

which proves (101). To prove (102), we use backward induction.

Base case $t = T \colon$ we have

$$
V _ { T } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } } ( h _ { T } ) = \mathcal { R } _ { T } ^ { \mathrm { h i n g e } } ( h _ { T } ; \tilde { f } _ { T } ^ { * } ) \qquad \mathrm { a n d } \qquad V _ { T } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { T } ) = \mathcal { R } _ { T } ( h _ { T } ; f _ { T } ^ { * } ) ,
$$

so applying (101) gives (102) for $t = T$

Backward induction: Assume (102) holds for $t + 1$ , we prove it for t. First, by Lemmas C.4–C.5, since both Bayes-optimal deferral rules $d _ { 1 : T } ^ { * }$ and $\tilde { d } _ { 1 : T } ^ { * }$ $\mathrm { a r e } \in \{ 0 , 1 \}$ -valued (see (82) and (97)), we have

$$
\begin{array} { r } { V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) = \operatorname* { m i n } \left\{ \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) , \mathbb { E } \left[ V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] \right\} , } \end{array}\tag{103}
$$

$$
V _ { t } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) = \operatorname* { m i n } \left\{ \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) , \mathbb { E } \left[ V _ { t + 1 } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] \right\} .\tag{104}
$$

Note by (104), (101) and (103), we can write

$$
\begin{array} { r l } & { V _ { t } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) \overset { \mathrm { ( i ) } } { = } \operatorname* { m i n } \Big \{ \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) , \mathbb { E } \left[ V _ { t + 1 } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] \Big \} } \\ & { \qquad \overset { \mathrm { ( i i ) } } { = } \operatorname* { m i n } \Big \{ \frac { K } { K - 1 } \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) , \mathbb { E } \left[ \frac { K } { K - 1 } V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] \Big \} } \\ & { \qquad = \frac { K } { K - 1 } \operatorname* { m i n } \Big \{ \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) , \mathbb { E } \left[ V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] \Big \} } \\ & { \qquad \overset { \mathrm { ( i i i ) } } { = } \frac { K } { K - 1 } V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) , } \end{array}
$$

where step (i) follows from (104), step (ii) follows from (101) and the induction hypothesis, and step (iii) applies (103). This completes the induction.

Conclusion: By backward induction, we establish (102) for all $t \in [ T ]$

The next lemma is an algebraic inequality linking the hard and soft deferral decisions, which will be useful in the proof of the main result (Theorem 3.2) in Section C.4 to show that the suboptimality gap under a hard deferral rule is at most twice that under its soft counterpart.

Lemma C.7 (Inequality linking soft and hard deferral decisions). For $\tilde { d } \in [ 0 , 1 ]$ , let $d : = \mathbb { 1 } \{ \tilde { d } \geq 0 . 5 \}$ For all $\mathcal { R } , \bar { V } \in \mathbb { R }$ , it holds that

$$
( 1 - d ) \mathcal { R } + d \bar { V } - \operatorname* { m i n } ( \mathcal { R } , \bar { V } ) \leq 2 \Big ( ( 1 - \tilde { d } ) \mathcal { R } + \tilde { d } \bar { V } - \operatorname* { m i n } ( \mathcal { R } , \bar { V } ) \Big ) ,\tag{105}
$$

where the 2 on the RHS is the smallest valid constant.

Proof. We first establish two inequalities linking d and ${ \tilde { d } } .$ When $\tilde { d } \geq 0 . 5$ , we have $d = 1$ and $2 \tilde { d } \geq 1$ so $d \leq 2 \tilde { d }  { : }$ ; when $\tilde { d } < 0 . 5 , d = 0$ so $d \leq 2 \tilde { d }$ is immediate. The same argument applied to $1 - { \tilde { d } } .$ , using $1 - d = \mathbb { 1 } \{ 1 - \tilde { d } > 0 . 5 \}$ , gives $1 - d \leq 2 ( 1 - \tilde { d } )$ . Now we prove (105) by considering two cases. Case 1. When $\mathcal { R } \leq \bar { V } \colon$ we have

$$
\begin{array} { r l } & { ( 1 - d ) \mathcal { R } + d \bar { V } - \operatorname* { m i n } ( \mathcal { R } , \bar { V } ) = ( 1 - d ) \mathcal { R } + d \bar { V } - \mathcal { R } = d ( \bar { V } - \mathcal { R } ) } \\ & { \qquad \overset { \mathrm { ( i ) } } { \leq } 2 \tilde { d } ( \bar { V } - \mathcal { R } ) = 2 \left( \tilde { d } \bar { V } + ( 1 - \tilde { d } ) \mathcal { R } - \mathcal { R } \right) } \\ & { \qquad = 2 \left( ( 1 - \tilde { d } ) \mathcal { R } + \tilde { d } \bar { V } - \operatorname* { m i n } ( \mathcal { R } , \bar { V } ) \right) } \end{array}
$$

where step (i) applies $d \leq 2 { \tilde { d } } .$

Case 2. When $\mathcal { R } > \bar { V } :$ we have

$$
\begin{array} { r l } & { ( 1 - d ) \mathcal { R } + d \bar { V } - \operatorname* { m i n } ( \mathcal { R } , \bar { V } ) = ( 1 - d ) \mathcal { R } + d \bar { V } - \bar { V } = ( 1 - d ) ( \mathcal { R } - \bar { V } ) } \\ &  \qquad \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \end{array}
$$

where step (i) applies $1 - d \leq 2 ( 1 - \tilde { d } )$

Tightness of (105): Finally, the factor of 2 in (105) is tight and cannot be improved: setting $\mathcal { R } = 0$ , $\bar { V } = 1$ , and $\tilde { d } = 0 . 5$ (so d = 1) gives $\mathrm { L H S } = 1$ and $\mathrm { R H S } = 1$ □

## C.4 Proof of Theorem 3.2

In this section, we prove our main calibration result of Theorem 3.2. Like our convention in the main text, we will consistently denote 0-1-c risk and its related quantities without superscripts and hinge risks and their related quantities with ‘hinge’ superscripts, $\mathrm { e . g . , } \mathcal { R } _ { \mathrm { c a s c } } , \Delta _ { t }$ and $\bar { \mathcal { R } } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } , \Delta _ { t } ^ { \mathrm { h i n g e } }$ respectively.

Theorem 3.2 (Linear calibration of $\ell _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } }$ w.r.t. ℓ). Assume $Y \bot Z _ { 1 : t - 1 } | \left( X , Z _ { t } \right)$ as in (5). Define $\begin{array} { r } { \mathcal { R } _ { c a s c } ^ { * } { : = } \operatorname* { m i n } _ { \pi \in \Pi _ { c a s c } } \mathcal { R } ( \pi ) } \end{array}$ and $\begin{array} { r } { \mathcal { R } _ { c a s c } ^ { h i n g e * } { : = } \operatorname* { m i n } _ { \tilde { \pi } \in \widetilde { \Pi } _ { c a s c } } \mathcal { R } _ { c a s c } ^ { h i n g e } ( \tilde { \pi } ) } \end{array}$ . Then, for any $\tilde { \pi } \in \bar { \Pi } _ { c a s c }$ ，

$$
\mathcal { R } \big ( \varphi ( \tilde { \pi } ) \big ) - \mathcal { R } _ { c a s c } ^ { * } \leq 2 ^ { T - 1 } ( K - 1 ) \Big ( \mathcal { R } _ { c a s c } ^ { h i n g e } ( \tilde { \pi } ) - \mathcal { R } _ { c a s c } ^ { h i n g e * } \Big ) .\tag{20}
$$

Proof. In this proof, we write $\varphi ( \tilde { \pi } ) = \pi = ( f _ { 1 : T } , d _ { 1 : T } )$ . Let $\pi _ { \mathrm { c a s c } } ^ { * }$ and $\tilde { \pi } _ { \mathrm { c a s c } } ^ { * }$ denote the Bayes solutions under the target 0-1-c loss and the cascade hinge loss, respectively. That is,

$$
\begin{array} { r l } & { \pi _ { \mathrm { c a s c } } ^ { * } = ( f _ { 1 : T } ^ { * } , d _ { 1 : T } ^ { * } ) \in \arg \operatorname* { m i n } _ { \pi \mathrm { c a s c } } \mathcal { R } ( \pi ) , } \\ & { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } = ( \tilde { f } _ { 1 : T } ^ { * } , \tilde { d } _ { 1 : T } ^ { * } ) \in \arg \operatorname* { m i n } _ { \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) . } } \\ & { \qquad \tilde { \pi } { \in } \widetilde { \Pi } _ { \mathrm { c a s c } } } \end{array}
$$

Then our end result (20) can be rewritten into

$$
\begin{array} { r } { \mathcal { R } ( \pi ) - \mathcal { R } ( \pi _ { \mathrm { c a s c } } ^ { * } ) \leq C _ { 1 } \Big ( \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) - \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } ) \Big ) \quad \mathrm { w i t h } \quad C _ { 1 } = 2 ^ { T - 1 } ( K - 1 ) . } \end{array}\tag{106}
$$

By Lemma C.2 and Corollary C.3, the risks on both sides of (106) equal their expected stage-1 value functions, further reducing (106) to

$$
\mathbb { E } \left[ V _ { 1 } ^ { \pi } ( H _ { 1 } ) \right] - \mathbb { E } \left[ V _ { 1 } ^ { \pi _ { \operatorname { c a s c } } ^ { * } } ( H _ { 1 } ) \right] \leq C _ { 1 } \Big ( \mathbb { E } \left[ V _ { 1 } ^ { \tilde { \pi } } ( H _ { 1 } ) \right] - \mathbb { E } \left[ V _ { 1 } ^ { \tilde { \pi } _ { \operatorname { c a s c } } ^ { * } } ( H _ { 1 } ) \right] \Big ) .\tag{107}
$$

To prove (107), it sufices in turn to prove for every $t \in [ T ]$ and pointwise in $h _ { t } \in \mathcal { H } _ { t }$

$$
\underbrace { V _ { t } ^ { \pi } ( h _ { t } ) - V _ { t } ^ { \pi ^ { * } \operatorname { s c a s c } } ( h _ { t } ) } _ { = : \Delta _ { t } ( h _ { t } ) } \le C _ { t } \underbrace { \left( V _ { t } ^ { \tilde { \pi } } ( h _ { t } ) - V _ { t } ^ { \tilde { \pi } _ { \operatorname { c a s c } } ^ { * } } ( h _ { t } ) \right) } _ { = : \Delta _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ) } \quad \mathrm { w i t h } \quad C _ { t } = 2 ^ { T - t } ( K - 1 ) .\tag{108}
$$

Here, $\Delta _ { t } ( h _ { t } )$ and $\Delta _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } )$ denote the suboptimality gaps in the stage-t value functions associated with the 0-1-c risk and hinge risk, respectively. The form of $C _ { t }$ arises from unrolling the recursion backward from the terminal stage, where each step of the induction introduces an additional factor of 2, giving $C _ { T } = K - 1$ up to $C _ { 1 } = 2 ^ { T - 1 } ( K - 1 )$ . In the remainder of the proof, we prove (108) using backward induction.

Step 1. Base case where $t = T$ : By the definition (75) of stage-T value functions, (108) specialized to $t = T$ reduces to

(109)

$$
\begin{array} { r l } & { \underbrace { \mathcal { R } _ { T } ( h _ { T } ; f _ { T } ) - \mathcal { R } _ { T } ( h _ { T } ; f _ { T } ^ { * } ) } _ { = : \Delta _ { T } ( h _ { T } ) } \le C _ { T } \underbrace { \left( \mathcal { R } _ { T } ^ { \mathrm { h i n g e } } ( h _ { T } ; \tilde { f } _ { T } ) - \mathcal { R } _ { T } ^ { \mathrm { h i n g e } } ( h _ { T } ; \tilde { f } _ { T } ^ { * } ) \right) } _ { = : \Delta _ { T } ^ { \mathrm { h i n g e } } ( h _ { T } ) } \quad \mathrm { w i t h } \quad C _ { T } = K - 1 , } \\ & { \mathrm { w h e r e } \quad \Delta _ { T } ( h _ { T } ) \ge 0 , ~ \Delta _ { T } ^ { \mathrm { h i n g e } } ( h _ { T } ) \ge 0 \quad \mathrm { f o r ~ a l l } \quad h _ { T } \in \mathcal { H } _ { T } . } \end{array}\tag{110}
$$

Eq. (109) holds by the single-stage calibration bound of Lemma B.4 with $( x , z _ { t } )$ replaced by $h _ { t }$ . In (110), $\Delta _ { T } ( h _ { T } ) \geq 0$ follows from the Bayes optimality of $f _ { t } ^ { * }$ for the 0-1-c loss (see explanation above (84)), and $\Delta _ { T } ^ { \mathrm { h i n g e } } ( h _ { T } ) \geq 0$ follows from the Bayes optimality of $\tilde { f } _ { T } ^ { * }$ for the cascade hinge loss. More precisely, $\Delta _ { T } ^ { \mathrm { h i n g e } } ( h _ { T } ) \geq 0$ because of the proof of [Tarigan and van de Geer, 2008, Lemma 1] and the conditional independence assumption (5).

Step 2. Backward induction: Next, for $t \leq T - 1$ , we show that if the suboptimality gaps at stage t + 1 satisfy:

$$
\underbrace { V _ { t + 1 } ^ { \pi } ( h _ { t + 1 } ) - V _ { t + 1 } ^ { \pi _ { \alpha \leq c } ^ { * } } ( h _ { t + 1 } ) } _ { = : \Delta _ { t + 1 } ( h _ { t + 1 } ) } \leq C _ { t + 1 } \underbrace { \left( V _ { t + 1 } ^ { \pi } ( h _ { t + 1 } ) - V _ { t + 1 } ^ { \pi _ { \alpha \leq c } ^ { * } } ( h _ { t + 1 } ) \right) } _ { = : \Delta _ { t + 1 } ^ { \mathrm { h i n g e } } ( h _ { t + 1 } ) } \quad \mathrm { w i t h } \quad C _ { t + 1 } = 2 ^ { T - t - 1 } ( K - 1 ) ,\tag{111}
$$

$$
\begin{array} { r } { \mathrm { w h e r e } \quad \Delta _ { t + 1 } ( h _ { t + 1 } ) \geq 0 , \ \Delta _ { t + 1 } ^ { \mathrm { h i n g e } } ( h _ { t + 1 } ) \geq 0 \quad \mathrm { f o r ~ a l l } \quad h _ { t + 1 } \in \mathcal { H } _ { t + 1 } , } \end{array}\tag{112}
$$

then the corresponding suboptimality gaps at stage t satisfy

$$
\underbrace { V _ { t } ^ { \pi } ( h _ { t } ) - V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) } _ { = : \Delta _ { t } ( h _ { t } ) } \le C _ { t } \underbrace { \left( V _ { t } ^ { \tilde { \pi } } ( h _ { t } ) - V _ { t } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) \right) } _ { = : \Delta _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ) } \quad \mathrm { w i t h } \quad C _ { t } = 2 C _ { t + 1 } ,\tag{113}
$$

$$
\begin{array} { r } { \mathrm { w h e r e } \quad \Delta _ { t } ( h _ { t } ) \geq 0 , \ \Delta _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ) \geq 0 \quad \mathrm { f o r ~ a l l } \quad h _ { t } \in \mathcal { H } _ { t } . } \end{array}\tag{114}
$$

Step 2.1. Decomposing the suboptimality gap under the 0-1-c loss: Let m be the cost achieved by an oracle deferral rule that picks the cheaper routing option at stage t, that is,

$$
\mathsf { m } : = \operatorname* { m i n } \Big \{ \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) , \mathbb { E } \left[ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] \Big \} .\tag{115}
$$

For the induction step, we first expand the $\Delta _ { t } ( h _ { t } )$ term on the LHS of (113) by adding and subtracting m to obtain

$$
\Delta _ { t } ( h _ { t } ) = V _ { t } ^ { \pi } ( h _ { t } ) - V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) = \underbrace { V _ { t } ^ { \pi } ( h _ { t } ) - \mathsf { m } } _ { = : \Delta _ { t , \mathrm { s e l e c t } } ( h _ { t } ) } + \underbrace { \mathsf { m } - V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) } _ { = : \Delta _ { t , \mathrm { i n t e r n a l } } ( h _ { t } ) } .\tag{116}
$$

Using the definition of m in (115) and the recursive definition of the value function in $( 7 4 )$ , we can rewrite the two terms on the RHS of (116) as follows, making their non-negativity explicit:

(117)

$$
\begin{array} { r l } & { \begin{array} { r l } & { \sum _ { t , \mathrm { s u l e r t } } ( h _ { t } ) = \underbrace { ( 1 - d _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) + d _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] } _ { V _ { t } ^ { \pi } ( h _ { t } ) } } \\ & { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \qquad \displaystyle \cdot \underbrace { \operatorname* { m i n } \left\{ { \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) , \mathbb { E } \left[ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] } \right\} } _ { \displaystyle m } \ge 0 , } \end{array} } \\ & { \begin{array} { r } & { \qquad \mathrm { a n d } ( h _ { t } ) = \underbrace { \operatorname* { m i n } \left\{ { \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) , \mathbb { E } \left[ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] } \right\} } _ { \displaystyle m } } \\ & { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \qquad \displaystyle \quad \quad - \operatorname* { m i n } \left\{ { \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) , \mathbb { E } \left[ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] } \right\} \ge 0 . } \end{array} } \end{array}\tag{118}
$$

Here, $\Delta _ { t } ( h _ { t } )$ is decomposed into two terms: $\Delta _ { t , \mathrm { s e l e c t } } ( h _ { t } )$ measures the suboptimality gap of $d _ { t } ( h _ { t } )$ relative to the oracle deferral rule, while $\Delta _ { t , \mathrm { i n t e r n a l } } ( h _ { t } )$ measures the suboptimality gap of using π rather than $\pi _ { \mathrm { c a s c } } ^ { * }$ for the downstream stages, holding the deferral choice at t as the oracle. We have $\Delta _ { t , \mathrm { s e l e c t } } ( h _ { t } ) \geq 0$ because min $\{ a , b \} \leq ( 1 - \lambda ) a +$ λb for every $\lambda \in [ 0 , 1 ]$ . Moreover, $\Delta _ { t , \mathrm { i n t e r n a l } } ( h _ { t } ) \geq 0$ follows from the Bayes optimality of $f _ { t } ^ { * }$ , which gives $\mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) \ge \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } )$ , and the induction hypothesis (112), which assumes $\Delta _ { t + 1 } ( h _ { t + 1 } ) = V _ { t + 1 } ^ { \pi } ( h _ { t + 1 } ) - V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t + 1 } ) \geq 0$ for all $h _ { t + 1 } \in \mathcal { H } _ { t + 1 }$ Since the sum of two non-negative terms is non-negative, we obtain

$$
\Delta _ { t } ( h _ { t } ) \geq 0 ,\tag{119}
$$

which is the first part of the conclusion (114) of the induction hypothesis.

Step 2.2. Replacing hard with soft deferral rules: Using (115)–(118) and applying Lemma C.7, we can replace the hard deferral decision $d _ { t } ( h _ { t } )$ in $\Delta _ { t , \mathrm { s e l e c t } } ( h _ { t } )$ with its soft counterpart $\tilde { d } _ { t } ( h _ { t } )$ at the cost of a factor of 2, yielding

$$
\begin{array} { r l r } { \Delta _ { t } ( h _ { t } ) = \Delta _ { t , \mathrm { s e l e c t } } ( h _ { t } ) + \Delta _ { t , \mathrm { i n t e r n a l } } ( h _ { t } ) } & { } & \\ { \leq 2 \Big ( ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) + \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] - \mathfrak { m } \Big ) + \Big ( \mathfrak { m } - V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) \Big ) } \\ & { } & { ( 1 2 0 ) } \\ & { = 2 \Big ( ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) + \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] \Big ) - \mathfrak { m } - V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) } \\ & { } & { = 2 \Big ( ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) + \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] \Big ) - \Delta _ { t , \mathrm { i n t e r n a l } } ( h _ { t } ) - 2 V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) } \\ & { } & { \qquad \mathrm ( 1 2 1 ) } \end{array}
$$

where the equalities follow by rearranging terms and the final inequality follows from $\Delta _ { t , \mathrm { i n t e r n a l } } ( h _ { t } ) \geq 0$ Step 2.3. Connecting $\Delta _ { t } ( h _ { t } )$ to $\Delta _ { t } ^ { \mathbf { h i n g e } } ( h _ { t } )$ : To help connect the 0-1-c suboptimality gap $\Delta _ { t } ( h _ { t } )$ to its hinge counterpart $\Delta _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } )$ , we rewrite the RHS of (121) without the factor of 2, by adding

and subtracting the same terms to obtain

$$
\begin{array} { r l } & { \frac { 1 } { 2 } \Delta _ { t } ( h _ { t } ) \leq ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \mathcal { K } _ { t } ( h _ { t } ; f _ { t } ) + \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t } ^ { \pi } ( 1 ( H _ { t + 1 } ) \ \big \vert \ H _ { t } = h _ { t } \right] - V _ { t } ^ { \pi _ { \mathrm { s e c } } ^ { \mathrm { s e c } } } ( h _ { t } ) } \\ & { \qquad = \underbrace { ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \big ( \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) - \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) \big ) } _ { = : \Delta _ { 0 } ^ { \infty } } + ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) } \\ & { \qquad + \underbrace { \tilde { d } _ { t } ( h _ { t } ) \bigg ( \mathbb { E } \left[ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) \ \big \vert \ H _ { t } = h _ { t } \right] - \mathbb { E } \left[ V _ { t + 1 } ^ { \pi _ { \mathrm { s e c } } ^ { \mathrm { s e c } } } ( H _ { t + 1 } ) \ \big \vert \ H _ { t } = h _ { t } \right] \bigg ) } _ { = : \Delta _ { 0 } ^ { \infty } } } \\ & { \qquad + \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \pi _ { \mathrm { s e c } } ^ { \mathrm { s e c } } } ( H _ { t + 1 } ) \ \big \vert \ H _ { t } = h _ { t } \right] - V _ { t } ^ { \pi _ { \mathrm { s e c } } ^ { \mathrm { s e c } } } ( h _ { t } ) } \\ &  \qquad = \mathsf { A } + \mathbb { B } + \underbrace  ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ( h _  \end{array}\tag{122}
$$

Here, $\mathsf { A } \geq 0$ because $f _ { t } ^ { * }$ is Bayes-optimal. Term $\textsf { B } \geq 0$ because the induction hypothesis (112) assumes $\Delta _ { t + 1 } ( h _ { t + 1 } ) = V _ { t + 1 } ^ { \pi } ( h _ { t + 1 } ) - V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t + 1 } ) \geq 0$ for all $h _ { t + 1 } \in \mathcal { H } _ { t + 1 }$ . Term $\mathsf C \geq 0$ because, using (103) and the fact that min $\{ a , b \} \leq ( 1 - \lambda ) a + \lambda b$ for every $\lambda \in [ 0 , 1 ]$ , we have

$$
\begin{array} { r l } & { V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) = \operatorname* { m i n } \left\{ \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) , \mathbb { E } \left[ V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] \right\} } \\ & { \qquad \leq ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) + \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] . } \end{array}\tag{123}
$$

We upper bound terms A, B, C separately with their hinge counterparts. By single-stage calibration (Lemma B.4, with $h _ { t }$ in place of $x , z _ { t } )$ , similar to (109), we have

$$
\begin{array} { r } { \mathsf { A } \le ( K - 1 ) \underbrace { ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \Big ( \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ) - \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) \Big ) } _ { = : \tilde { \mathsf { A } } } = ( K - 1 ) \tilde { \mathsf { A } } . } \end{array}\tag{124}
$$

Since $\tilde { f } _ { t } ^ { * }$ is Bayes-optimal for the cascade hinge loss, $\mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ) - \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) \ge 0$ , and so $\tilde { \mathsf { A } } \geq 0$ By the induction hypothesis (111), we can write

$$
\begin{array} { r l } & { \mathsf { B } = \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) - V _ { t + 1 } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] } \\ & { \quad \le \tilde { d } _ { t } ( h _ { t } ) C _ { t + 1 } \mathbb { E } \left[ V _ { t + 1 } ^ { \tilde { \pi } } ( H _ { t + 1 } ) - V _ { t + 1 } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] } \\ & { \quad = C _ { t + 1 } \underbrace { \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \tilde { \pi } } ( H _ { t + 1 } ) - V _ { t + 1 } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] } _ { = : \tilde { \mathsf { B } } } = C _ { t + 1 } \tilde { \mathsf { B } } , } \end{array}\tag{125}
$$

where $\tilde { \mathsf { B } } \geq 0$ because the induction hypothesis (112) assumes $\Delta _ { t + 1 } ^ { \mathrm { h i n g e } } ( h _ { t + 1 } ) = V _ { t + 1 } ^ { \tilde { \pi } } ( h _ { t + 1 } )  – V _ { t + 1 } ^ { \tilde { \pi } _ { \mathrm { c a s c } } ^ { \ast } } ( h _ { t + 1 } ) \ge$ 0 for all $h _ { t + 1 } \in \mathcal { H } _ { t + 1 }$

Finally, since $\begin{array} { r } { \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ^ { * } ) = \frac { K - 1 } { K } \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) } \end{array}$ and $\begin{array} { r } { V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { \ast } } ( h _ { t } ) = \frac { K - 1 } { K } V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { \ast } } ( h _ { t } ) } \end{array}$ by Lemma C.6 for all $t \in [ T ]$ , we can write

$$
\begin{array} { r l } & { \mathsf { C } = ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \frac { K - 1 } { K } \mathcal { R } _ { t } ^ { \mathrm { h a s e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) + \tilde { d } _ { t } ( h _ { t } ) \frac { K - 1 } { K } \mathbb { E } \left[ V _ { t + 1 } ^ { \tilde { \pi } _ { \mathrm { t a s e } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] - \frac { K - 1 } { K } V _ { t } ^ { \tilde { \pi } _ { \mathrm { t a s e } } ^ { * } } ( h _ { t } ) } \\ & { \quad = \frac { K - 1 } { K } \underbrace { \left( ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) + \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \tilde { \pi } _ { \mathrm { t a s e } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] - V _ { t } ^ { \tilde { \pi } _ { \mathrm { t a s e } } ^ { * } } ( h _ { t } ) \right) } _ { = : \tilde { C } } = \frac { K - 1 } { K } \tilde { \mathsf { C } } , } \end{array}\tag{126}
$$

where $\tilde { \mathsf C } \ge 0$ following the same argument as that in (123) for $\mathsf C \geq 0$ , using (104) instead of (103). Step 2.4. Assembling all terms: Finally, recall from (111) that $C _ { t + 1 } = 2 ^ { T - t - 1 } ( K - 1 )$ . Since $t \leq T - 1$ implies $2 ^ { T - t - 1 } \geq 1$ , the pre-multiplicative factors $K - 1 , C _ { t + 1 }$ , and $\textstyle { \frac { K - 1 } { K } }$ in (124), (125), and (126), respectively, are all dominated by $C _ { t + 1 }$ . Together with $\tilde { { \mathsf { A } } } , \tilde { { \mathsf { B } } } , \tilde { { \mathsf { C } } } \geq 0$ , we have

$$
\mathsf { A } \leq C _ { t + 1 } \tilde { \mathsf { A } } , \quad \mathsf { B } \leq C _ { t + 1 } \tilde { \mathsf { B } } , \quad \mathrm { a n d } \quad \mathsf { C } \leq C _ { t + 1 } \tilde { \mathsf { C } } .
$$

Substituting this into (122) and rearranging yields

$$
\Delta _ { t } ( h _ { t } ) \le 2 \big ( \mathsf { A } + \mathsf { B } + \mathsf { C } \big ) \le 2 \cdot C _ { t + 1 } \big ( \tilde { \mathsf { A } } + \tilde { \mathsf { B } } + \tilde { \mathsf { C } } \big ) = C _ { t } \big ( \tilde { \mathsf { A } } + \tilde { \mathsf { B } } + \tilde { \mathsf { C } } \big ) .\tag{127}
$$

By the definition of $\tilde { \mathsf { A } } , \tilde { \mathsf { B } }$ , and $\tilde { \mathsf { C } }$ from (124), (125), and (126), we can write

$$
\begin{array} { r l } & { \tilde { \mathsf { A } } + \tilde { \mathsf { B } } + \tilde { \mathsf { C } } = ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \Big ( \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ) - \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) \Big ) } \\ & { \qquad + \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \tilde { \pi } } ( H _ { t + 1 } ) - V _ { t + 1 } ^ { \tilde { \pi } _ { \mathrm { e x t o c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] } \\ & { \qquad + \left( ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ^ { * } ) + \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \tilde { \pi } _ { \mathrm { e x t o c } } ^ { * } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] - V _ { t } ^ { \tilde { \pi } _ { \mathrm { e x t o c } } ^ { * } } ( h _ { t } ) \right) } \\ & { \qquad = ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ; \tilde { f } _ { t } ) + \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \left[ V _ { t + 1 } ^ { \tilde { \pi } } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] - V _ { t } ^ { \tilde { \pi } _ { \mathrm { e x t o c } } ^ { * } } ( h _ { t } ) } \\ &  \qquad \overset { \mathrm { ( i ) } } { = } V _ { t } ^ { \tilde { \pi } } ( h _ { t } ) - V _ { t } ^  \tilde { \pi } _  \mathrm  e x \end{array}\tag{128}
$$

where step (i) applies the definition of $V _ { t } ^ { \tilde { \pi } } ( h _ { t } )$ (see (74)), and step (ii) uses the definition of $\Delta _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } )$ in (113). Combining (127) and (128) proves (113). Together with the result $\Delta _ { t } ( h _ { t } ) \geq 0$ established in (119), this implies $\Delta _ { t } ^ { \mathrm { h i n g e } } ( h _ { t } ) \geq 0$ , proving the second part of (114). Taken together, we have completed the backward induction step.

Step 3. Conclusion: Applying this argument inductively down to $t = 1$ yields (108) for all $t \in [ T ]$ . Setting $t = 1$ in (108) and taking total expectation over $H _ { 1 }$ proves (106), which in turn proves (20). □

## C.5 Lower Bound

We prove the following lower bound to show that $2 ^ { T - 1 } ( K - 1 )$ is the smallest constant for which the upper bound in Theorem 3.2 holds uniformly over all data distributions $P$ in (4) satisfying (5), cost vectors $\boldsymbol { c } = \left[ \kappa _ { 1 } , \kappa _ { 1 } + \kappa _ { 2 } , \ldots , \sum _ { t = 1 } ^ { T } \kappa _ { t } \right] ^ { \top }$ with $\kappa _ { 1 } = 0$ and $\kappa _ { t } \geq \kappa _ { t - 1 }$ , and multi-model classifiers $\tilde { \pi } \in \widetilde { \Pi } _ { \mathrm { c a s c } }$

Proposition C.8 (Lower bound). Let $K \geq 2$ and $T \geq 2$ . For every $C < 2 ^ { T - 1 } ( K - 1 )$ , there exists a distribution P satisfying (5), an admissible cost vector $^ { c , }$ and $\tilde { \pi } \in \tilde { \Pi } _ { c a s c }$ such that

$$
\mathcal { R } \big ( \varphi ( \tilde { \pi } ) \big ) - \mathcal { R } _ { c a s c } ^ { * } > C \Big ( \mathcal { R } _ { c a s c } ^ { h i n g e } ( \tilde { \pi } ) - \mathcal { R } _ { c a s c } ^ { h i n g e * } \Big ) .\tag{129}
$$

Proof. We construct a bad instance of $P , c$ and $\tilde { \pi } _ { ; }$ , and show that the excess risks of the bad instance have a ratio that approaches $2 ^ { T - 1 } ( K - 1 )$

Step 1. Choose P and c for the bad instance: Set all incremental costs to zero, i.e., $\kappa _ { t } = 0$ for all $t \in [ T ]$ , giving the all-zero cost vector $c = \mathbf { 0 } _ { T }$ . Let X have any distribution on $x ,$ and let $Z _ { 1 } , Z _ { 2 } , \ldots , Z _ { T }$ be constant. Let $Y$ be independent of X with

$$
p _ { y } : = \operatorname* { P r } ( Y = y ) = \left\{ \begin{array} { l l l } { \frac { 1 } { K - 1 } } & { \mathrm { f o r } } & { y \in \left[ K - 1 \right] , } \\ { 0 } & { \mathrm { f o r } } & { y = K . } \end{array} \right.\tag{130}
$$

This ensures Y to be independent of the full history $H _ { T } = ( X , Z _ { 1 : T } )$ . In particular, (5) holds and $p ( y \mid x , z _ { 1 : t } ) = p ( y \mid x , z _ { t } ) = p _ { y }$ for all $t \in \left\lceil T \right\rceil$

Using the definition of $\ell ^ { \mathrm { h i n g e } }$ from (15), for a score vector $v \in \mathbb { R } ^ { K }$ with $\begin{array} { r } { \sum _ { k \in [ K ] } v _ { k } = 0 } \end{array}$ , we can write that

$$
\ell ^ { \mathrm { h i n g e } } ( v , y ; 0 ) = \sum _ { k \in [ K ] \setminus y } \underbrace { \left[ v _ { k } + \frac { 1 } { K - 1 } \right] _ { + } } _ { = : a _ { k } } + \frac { K } { K - 1 } \cdot 0 = \sum _ { k \in [ K ] \setminus y } a _ { k } = : \bar { \ell } ^ { \mathrm { h i n g e } } ( v , y ) .\tag{131}
$$

Define $L ( v )$ as follows, and expanding and rearranging gives

$$
\begin{array} { r l r } & { } & { L ( v ) : = \displaystyle \sum _ { y \in [ K ] } p _ { y } \bar { \ell } ^ { \mathrm { h i n g e } } ( v , y ) = \displaystyle \sum _ { y \in [ K ] } p _ { y } \sum _ { k \in [ K ] \backslash y } a _ { k } \stackrel { \mathrm { ( i ) } } { = } \sum _ { k \in [ K ] } a _ { k } \sum _ { y \in [ K ] \backslash k } p _ { y } } \\ & { } & { \quad \quad \quad = \displaystyle \sum _ { k \in [ K ] } a _ { k } ( 1 - p _ { k } ) \stackrel { \mathrm { ( i i ) } } { = } \sum _ { k = 1 } ^ { K - 1 } a _ { k } \cdot \frac { K - 2 } { K - 1 } + a _ { K } , } \end{array}\tag{132}
$$

where step (i) exchanges summations, and step (ii) applies (130). Moreover, we can lower bound $L ( v )$ by

$$
\begin{array} { r } { L ( v ) \stackrel { \mathrm { ( i ) } } { \geq } \frac { K - 2 } { K - 1 } \displaystyle \sum _ { k \in [ K ] } a _ { k } \stackrel { \mathrm { ( i i ) } } { \geq } \frac { K - 2 } { K - 1 } \displaystyle \sum _ { k \in [ K ] } \left[ v _ { k } + \frac { 1 } { K - 1 } \right] \stackrel { \mathrm { ( i i i ) } } { = } \frac { K - 2 } { K - 1 } \cdot K \cdot \frac { 1 } { K - 1 } = \frac { K ( K - 2 ) } { ( K - 1 ) ^ { 2 } } = : L ^ { * } , } \end{array}\tag{133}
$$

where step (i) uses $a _ { k } \geq 0$ for all $k \in [ K ]$ , step (ii) uses $[ x ] _ { + } \geq x$ for any x $\in \mathbb { R }$ , and step (iii) applies $\begin{array} { r } { \sum _ { k \in \left[ K \right] } v _ { k } = 0 } \end{array}$ . Define a score vector $v ^ { * } \in \mathbb { R } ^ { K } \mathrm { \ b y \ } v _ { 1 } ^ { * } : = \bar { 1 }$ and $v _ { k } ^ { * } : = - \frac { 1 } { K - 1 }$ for all $2 \leq k \leq K$ , which satisfies $\begin{array} { r } { \sum _ { k \in [ K ] } v _ { k } ^ { * } = 0 } \end{array}$ . Plugging $v = v ^ { * }$ into (132) and applying $\begin{array} { r } { p _ { 1 } = \frac { 1 } { K - 1 } } \end{array}$ from (130) obtains

$$
\begin{array} { r } { L ( v ^ { * } ) = \frac { K } { K - 1 } \frac { K - 2 } { K - 1 } + 0 = \frac { K ( K - 2 ) } { ( K - 1 ) ^ { 2 } } = L ^ { * } . } \end{array}\tag{134}
$$

Step 2. Compute Bayes risks $\mathcal { R } ^ { * }$ and $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e * } }$ : We now show that in the bad instance that we have set up, the Bayes 0-1-c risk and the Bayes cascade hinge risk are $\begin{array} { r } { \mathcal { R } ^ { * } = \frac { K - 2 } { K - 1 } } \end{array}$ and $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e * } } = L ^ { * }$ respectively.

First, using the definition of the 0-1-c risk $\mathcal { R } ( \pi )$ in $( 9 ) \substack { - } ( 1 0 )$ , we can express $\mathcal { R } ( \pi )$ in the bad instance as

$$
\begin{array} { r l } & { { \mathcal R } ( \pi ) = { \mathbb E } \Big [ \mathbb { 1 } \{ f _ { r ( H ) } ( X , Z _ { r ( H ) } ) \neq Y \} + c _ { r ( H ) } \Big ] } \\ & { \qquad = \operatorname* { P r } \Big ( f _ { r ( H ) } ( X , Z _ { r ( H ) } ) \neq Y \Big ) = 1 - \operatorname* { P r } \Big ( Y = f _ { r ( H ) } ( X , Z _ { r ( H ) } ) \Big ) , } \end{array}\tag{135}
$$

and, using (130), we can lower bound it by

$$
\begin{array} { r } { \mathcal { R } ( \pi ) \ge 1 - \displaystyle \operatorname* { m a x } _ { y \in [ K ] } p _ { y } = 1 - \frac { 1 } { K - 1 } = \frac { K - 2 } { K - 1 } . } \end{array}\tag{136}
$$

Note that equality holds for $f _ { t } \equiv 1$ for all $t \in [ T ]$ and any router $r ,$ attaining the lower bound $\frac { K - 2 } { K - 1 }$ Hence, the Bayes 0-1-c risk is

$$
\begin{array} { r } { \mathcal { R } _ { \mathrm { c a s c } } ^ { * } = \frac { K - 2 } { K - 1 } . } \end{array}\tag{137}
$$

Second, using the definition of the hinge risk $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } )$ below (18)–(19), we can express $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } )$ in the bad instance as

$$
\begin{array} { r l } { \left. { \mathcal { R } _ { \mathrm { s o s e } } ^ { \mathrm { h i n s e s } } ( \tilde { \varepsilon } ) = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \tilde { \gamma } _ { t } ( H _ { t } ) \mathrm { s i n s s } ^ { \varepsilon } ( \tilde { f } _ { t } ( X , Z _ { t } ) , Y ; \varepsilon _ { \infty } ) \right] } \qquad } & { } \\ & { \stackrel { \mathrm { ( i i ) } } { = } \mathbf { E } \left[ \sum _ { t = 1 } ^ { T } \tilde { \gamma } _ { t } ( H _ { t } ) \tilde { \varepsilon } ^ { \mathrm { i n s s } } ( \tilde { f } _ { t } ( X , Z _ { t } ) , Y ) \right] } \\ & { - \sum _ { t = 1 } ^ { T } \left[ \tilde { \gamma } _ { t } ( H _ { t } ) \mathbb { E } \left[ \tilde { \varepsilon } ^ { \mathrm { i n s s } t } ( \tilde { f } _ { t } ( X , Z _ { t } ) , Y ) \right] H _ { t } \right] \right] } \\ & { \stackrel { \mathrm { ( i i i ) } } { = } \underset { t = 1 } { \overset { \mathrm { ( i i i ) } } { \sum } } \mathbf { E } \left[ \tilde { \gamma } _ { t } ( H _ { t } ) \sum _ { t = 1 } ^ { T } p _ { t } \tilde { \varepsilon } ^ { \mathrm { i n s s } } ( \tilde { f } _ { t } ( X , Z _ { t } ) , y ) \right] } \\ & { \stackrel { \mathrm { ( i i i ) } } { = } \underset { t = 1 } { \overset { \mathrm { ( i i i ) } } { \sum } } \mathbf { E } \left[ \tilde { \gamma } _ { t } ( H _ { t } ) \sum _ { t = 1 } ^ { T } | \tilde { \varepsilon } ^ { \mathrm { i n s } } ( \tilde { f } _ { t } ( X , Z _ { t } ) , y ) \right] } \\ & { \stackrel { \mathrm { ( i i i ) } } { = } \underset { t = 1 } { \overset { \mathrm { ( i i i ) } } { \sum } } \mathbf { E } \left[ \tilde { \gamma } _ { t } ( H _ { t } ) L \left( \tilde { f } _ { t } ( X , Z _ { t } ) \right) \right] . } \end \end{array}\tag{138}
$$

where step (i) applies the definition of $\bar { \ell } ^ { \mathrm { h i n g e } }$ in (131), step (ii) uses $\operatorname* { P r } ( Y = y \mid H _ { t } ) = p _ { y }$ , and step (iii) uses the definition of $L$ in (132). Using (133) and $\textstyle \sum _ { t = 1 } ^ { T } { \tilde { r } } _ { t } ( H _ { t } ) = 1$ , we can further lower bound $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } )$ by

$$
\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) \geq \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \tilde { r } _ { t } ( H _ { t } ) L ^ { * } \right] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \tilde { r } _ { t } ( H _ { t } ) \right] \cdot L ^ { * } = L ^ { * } .\tag{139}
$$

Note that (134) implies that equality holds for $\tilde { f } _ { t } \equiv v ^ { * }$ for all $t \in [ T ]$ and for any soft router $\tilde { r } _ { 1 } , \ldots , \tilde { r } _ { T }$ Hence, the Bayes cascade hinge risk is

$$
\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e * } } = L ^ { * } .\tag{140}
$$

Step 3. Choose $\tilde { \pi } \in \widetilde { \Pi } _ { \mathbf { c a s c } }$ for the bad instance: Recall that $v ^ { * } \in \mathbb { R } ^ { K }$ is a score vector with   
$v _ { 1 } ^ { * } = 1$ and $v _ { k } ^ { * } = - \frac { 1 } { K - 1 }$ for all $2 \leq k \leq K$ . Fix $\epsilon \in [ 0 , \frac { 1 } { 2 } ]$ and $s \in ( 0 , \frac { 1 } { K - 1 } ]$ . Define a score vector   
$v ^ { ( s ) } \in \mathbb { R } ^ { K } \mathrm { \ b y \ } v _ { k } ^ { ( s ) } : = - s$ for $k \in [ K - 1 ]$ and $v _ { K } ^ { ( s ) } \mathrel { \mathop : } = ( K - 1 ) s$ , satisfying $\begin{array} { r } { \sum _ { k = 1 } ^ { K } v _ { k } ^ { ( s ) } = 0 } \end{array}$ . Then, define k   
$\tilde { \pi } = ( \tilde { f } _ { 1 : T } , \tilde { d } _ { 1 : T } )$ as

$$
\tilde { f } _ { t } \equiv v ^ { * } \quad \mathrm { f o r } t \le T - 1 , \quad \mathrm { a n d } \quad \tilde { f } _ { T } \equiv v ^ { ( s ) } ,\tag{141}
$$

and

$$
\begin{array} { r } { \tilde { d } _ { t } \equiv \frac { 1 } { 2 } + \epsilon \quad \mathrm { f o r } t \leq T - 1 , \quad \mathrm { a n d } \quad \tilde { d } _ { T } \equiv 0 . } \end{array}\tag{142}
$$

This constructed π˜ belongs to $\widetilde { \Pi } _ { \mathrm { c a s c } }$ by definition of $\widetilde { \Pi } _ { \mathrm { c a s c } }$ . The probability of exiting at stage $T$ is thus

$$
\tilde { r } _ { T } \equiv ( \frac { 1 } { 2 } + \epsilon ) ^ { T - 1 }\tag{143}
$$

and, by the definition of $L ( v )$ in (132), we have

$$
\begin{array} { l } { { { \cal L } ( v ^ { ( s ) } ) = \displaystyle \sum _ { k = 1 } ^ { K - 1 } \left[ v _ { k } ^ { ( s ) } + \frac { 1 } { K - 1 } \right] _ { + } \cdot \frac { K - 2 } { K - 1 } + \left[ v _ { K } ^ { ( s ) } + \frac { 1 } { K - 1 } \right] _ { + } } } \\ { { \mathrm { } = \displaystyle \sum _ { k = 1 } ^ { K - 1 } \left[ - s + \frac { 1 } { K - 1 } \right] _ { + } \cdot \frac { K - 2 } { K - 1 } + \left[ ( K - 1 ) s + \frac { 1 } { K - 1 } \right] _ { + } } } \\ { { \displaystyle \stackrel { ( i ) } { = } \sum _ { k = 1 } ^ { K - 1 } \left[ - s + \frac { 1 } { K - 1 } \right] \cdot \frac { K - 2 } { K - 1 } + \left[ ( K - 1 ) s + \frac { 1 } { K - 1 } \right] } } \\ { { \mathrm { } = \displaystyle \frac { K - 2 } { K - 1 } - s ( K - 2 ) + s ( K - 1 ) + \frac { 1 } { K - 1 } = s + 1 , } } \end{array}\tag{144}
$$

where step (i) holds because $s \in ( 0 , \frac { 1 } { K - 1 } ]$ by construction.

Step 4. Compute the risks $\mathcal { R } ( \varphi ( \tilde { \pi } ) )$ and $\mathcal { R } _ { \mathbf { c a s c } } ^ { \mathbf { h i n g e } } ( \tilde { \pi } )$ : We are now ready to derive the cascade hinge risk of the constructed π˜ and the 0-1-c risk of its hard prediction $\varphi ( \tilde { \pi } )$ . First, together with the expression for $\mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } )$ in (138), we can write

$$
\begin{array} { l } { \mathcal { \mathcal { R } } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \tilde { r } _ { t } ( H _ { t } ) L \big ( \tilde { f } _ { t } ( X , Z _ { t } ) \big ) \right] = \sum _ { t = 1 } ^ { T - 1 } \mathbb { E } \left[ \tilde { r } _ { t } ( H _ { t } ) L \big ( v ^ { * } \big ) \right] + \mathbb { E } \left[ \tilde { r } _ { T } ( H _ { T } ) L \big ( v ^ { ( s ) } \big ) \right] } \\ { \quad \quad \stackrel { \mathrm { ( i ) } } { = } \Big ( 1 - \mathbb { E } \big [ \tilde { r } _ { T } ( H _ { T } ) \big ] \Big ) L ^ { * } + \mathbb { E } \big [ \tilde { r } _ { T } ( H _ { T } ) \big ] \big ( s + 1 \big ) } \\ { \quad \quad = L ^ { * } + \mathbb { E } \big [ \tilde { r } _ { T } ( H _ { T } ) \big ] \big ( s + 1 - L ^ { * } \big ) \stackrel { \mathrm { ( i i ) } } { = } L ^ { * } + \big ( \frac { 1 } { 2 } + \epsilon \big ) ^ { T - 1 } \big ( s + 1 - L ^ { * } \big ) , } \end{array}\tag{145}
$$

where step (i) uses $L ( v ^ { * } ) = L ^ { * }$ from (134) and $L ( v ^ { ( s ) } ) = s + 1$ from (144), and step (ii) applies (143). Second, under the soft-to-hard map φ, as defined in (12) and (14), the hard classifiers are $f _ { t } \equiv 1$ for $t \leq T - 1$ and $f _ { T } \equiv K$ and, induced by the deferral rules, the hard routing rule is $r ( h ) = T$ for all $h \in \mathcal H$ . This implies that

$$
\begin{array} { r } { \mathcal { R } ( \varphi ( \tilde { \pi } ) ) = \operatorname* { P r } \Big ( f _ { r ( H ) } ( X , Z _ { r ( H ) } ) \neq Y \Big ) = \operatorname* { P r } \Big ( f _ { T } ( X , Z _ { T } ) \neq Y \Big ) = \operatorname* { P r } \big ( Y \neq K \big ) = 1 , } \end{array}\tag{146}
$$

where the last step uses (130).

Step 5. Compute excess risks and prove lower bound: The final step computes and links the excess risks. Combining (137) and (146) gives the excess 0-1-c risk:

$$
\begin{array} { r } { \mathcal { R } ( \varphi ( \tilde { \pi } ) ) - \mathcal { R } _ { \mathrm { c a s c } } ^ { * } = 1 - \frac { K - 2 } { K - 1 } = \frac { 1 } { K - 1 } . } \end{array}\tag{147}
$$

Combining (140) and (145) gives the excess cascade hinge risk:

$$
\begin{array} { r l } & { \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) - \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e * } } = L ^ { * } + ( \frac { 1 } { 2 } + \epsilon ) ^ { T - 1 } ( s + 1 - L ^ { * } ) - L ^ { * } = ( \frac { 1 } { 2 } + \epsilon ) ^ { T - 1 } ( s + 1 - L ^ { * } ) } \\ & { \qquad = ( \frac { 1 } { 2 } + \epsilon ) ^ { T - 1 } \Big [ s + 1 - \frac { K ( K - 2 ) } { ( K - 1 ) ^ { 2 } } \Big ] = ( \frac { 1 } { 2 } + \epsilon ) ^ { T - 1 } \Big [ s + \frac { 1 } { ( K - 1 ) ^ { 2 } } \Big ] , } \end{array}\tag{148}
$$

where the last step applies (133). Finally, taking ratio of (147) and (148) gives

$$
\frac { \mathcal { R } ( \varphi ( \tilde { \pi } ) ) - \mathcal { R } _ { \mathrm { c a s c } } ^ { * } } { \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e } } ( \tilde { \pi } ) - \mathcal { R } _ { \mathrm { c a s c } } ^ { \mathrm { h i n g e * } } } = \frac { \frac { 1 } { K - 1 } } { ( \frac { 1 } { 2 } + \epsilon ) ^ { T - 1 } \Big [ s + \frac { 1 } { ( K - 1 ) ^ { 2 } } \Big ] } = \frac { 2 ^ { T - 1 } ( K - 1 ) } { ( 1 + 2 \epsilon ) ^ { T - 1 } \Big [ s ( K - 1 ) ^ { 2 } + 1 \Big ] } = : \mathsf { R } .\tag{149}
$$

$\mathrm { A s } \ \epsilon , s  0$ , the denominator tends to 1, and so $\mathsf { R } \to 2 ^ { T - 1 } ( K - 1 )$ , implying that $2 ^ { T - 1 } ( K - 1 )$ is the constant in the lower bound. More concretely, for every $C < 2 ^ { { \cal T } - 1 } ( K - 1 )$ , we can choose ϵ and s small enough that $\mathsf { R } > C$ . For example, setting

$$
\epsilon = 0 \quad \mathrm { a n d } \quad s = \operatorname* { m i n } \left\{ { \frac { 1 } { K - 1 } } , { \frac { 2 ^ { T - 1 } ( K - 1 ) / C - 1 } { 2 ( K - 1 ) ^ { 2 } } } \right\} > 0 ,
$$

and using the shorthand $\mathsf { r } : = 2 ^ { T - 1 } ( K - 1 ) / C > 1$ gives

$$
s ( K - 1 ) ^ { 2 } + 1 \leq { \frac { \mathsf { r } - 1 } { 2 } } + 1 = { \frac { \mathsf { r } + 1 } { 2 } } < { \frac { \mathsf { r } + \mathsf { r } } { 2 } } = \mathsf { r } .
$$

This implies that

$$
\mathsf { R } = \frac { 2 ^ { T - 1 } ( K - 1 ) } { 1 \cdot \Big [ s ( K - 1 ) ^ { 2 } + 1 \Big ] } > \frac { 2 ^ { T - 1 } ( K - 1 ) } { \mathsf { r } } = C .\tag{150}
$$

This proves the proposition.

## C.6 An Upper Bound in Favorable Instances

Proposition C.8 shows that the constant in Theorem 3.2 cannot be improved in bad problem instances, where every soft deferral decision is maximally indecisive, that is, $\tilde { d } _ { t } \equiv 0 . 5$ . We now show that the constant in Theorem 3.2 is much smaller in more favorable problem instances, where the soft deferral decisions stay away from 0.5. To begin, we prove a refined, margin-dependent version of Lemma C.7 to connect soft and hard deferral decisions.

Lemma C.9 (Margin-dependent inequality linking soft and hard deferral decisions). For $\tilde { d } \in [ 0 , 1 ]$ let $d : = \mathbb { 1 } \{ \tilde { d } \geq 0 . 5 \}$ and $m : = | \tilde { d } - 0 . 5 |$ . For all $\mathcal { R } , \bar { V } \in \mathbb { R }$

$$
\begin{array} { r } { ( 1 - d ) \mathcal { R } + d \bar { V } - \operatorname* { m i n } ( \mathcal { R } , \bar { V } ) \ \leq \ \frac { 2 } { 1 + 2 m } \Big [ ( 1 - \tilde { d } ) \mathcal { R } + \tilde { d } \bar { V } - \operatorname* { m i n } ( \mathcal { R } , \bar { V } ) \Big ] , } \end{array}\tag{151}
$$

where ${ \frac { 2 } { 1 + 2 m } } \in [ 1 , 2 ]$ is the smallest constant valid at the given m.

Proof. Let $[ x ] _ { + } : = \operatorname* { m a x } ( x , 0 )$ . Using $m : = | \tilde { d } - 0 . 5 |$ , we can define and write

$$
\begin{array} { r } { q : = \operatorname* { m a x } ( \tilde { d } , 1 - \tilde { d } ) = \frac { 1 } { 2 } + m , \quad \frac { 1 } { q } = \frac { 2 } { 1 + 2 m } . } \end{array}\tag{152}
$$

For brevity, we use shorthand $H : = ( 1 - d ) \mathcal { R } + d \bar { V } - \operatorname* { m i n } ( \mathcal { R } , \bar { V } )$ and $S : = ( 1 - \tilde { d } ) \mathcal { R } + \tilde { d } \bar { V } - \operatorname* { m i n } ( \mathcal { R } , \bar { V } )$ so the desired bound in (151) can be equivalently written as

$$
H \leq { \textstyle \frac { 1 } { q } } S ,\tag{153}
$$

which we prove below. To establish tightness of (151), it therefore sufices to construct a case in which (153) holds with equality.

Step 1: We first rewrite H and S as follows, replacing the hard-to-handle min $( { \mathcal { R } } , { \bar { V } } )$ terms with terms involving $[ \cdot ] _ { + }$ . To rewrite $H _ { ; }$ , we consider two cases. If $\mathcal { R } \leq \bar { V }$ , then min $( \mathcal { R } , \bar { V } ) = \mathcal { R }$ , which implies that

$$
\begin{array} { r l } & { ( 1 - d ) ( \mathcal { R } - \bar { V } ) _ { + } + d ( \bar { V } - \mathcal { R } ) _ { + } = ( 1 - d ) \cdot 0 + d ( \bar { V } - \mathcal { R } ) } \\ & { \qquad = ( 1 - d ) \mathcal { R } + d \bar { V } - \operatorname* { m i n } ( \mathcal { R } , \bar { V } ) = H . } \end{array}\tag{154}
$$

Similarly, if $\mathcal { R } > \bar { V }$ , then $\operatorname* { m i n } ( \mathcal { R } , \bar { V } ) = \bar { V }$ , which implies that

$$
\begin{array} { r l } & { ( 1 - d ) ( \mathcal { R } - \bar { V } ) _ { + } + d ( \bar { V } - \mathcal { R } ) _ { + } = ( 1 - d ) ( \mathcal { R } - \bar { V } ) + d \cdot 0 } \\ & { \qquad = ( 1 - d ) \mathcal { R } + d \bar { V } - \operatorname* { m i n } ( \mathcal { R } , \bar { V } ) = H . } \end{array}\tag{155}
$$

Together, we have shown

$$
( 1 - d ) ( \mathcal { R } - \bar { V } ) _ { + } + d ( \bar { V } - \mathcal { R } ) _ { + } = H .\tag{156}
$$

Applying the same argument to $S$ obtains

$$
( 1 - \tilde { d } ) ( \mathcal { R } - \bar { V } ) _ { + } + \tilde { d } ( \bar { V } - \mathcal { R } ) _ { + } = S .\tag{157}
$$

Step 2: Using (156) and (157), we can now prove (153). Again, consider two cases: If $\tilde { d } < 0 . 5$ then $d = 0$ and $q = \operatorname* { m a x } ( \tilde { d } , 1 - \tilde { d } ) = 1 - \tilde { d } .$ . Hence, substituting this into (157) and using the fact that $[ \cdot ] _ { + } \geq 0$ gives

$$
\begin{array} { r l } & { S = ( 1 - \tilde { d } ) ( \mathcal { R } - \bar { V } ) _ { + } + \tilde { d } ( \bar { V } - \mathcal { R } ) _ { + } = q ( \mathcal { R } - \bar { V } ) _ { + } + \tilde { d } ( \bar { V } - \mathcal { R } ) _ { + } } \\ & { \qquad \geq q ( \mathcal { R } - \bar { V } ) _ { + } } \\ & { \qquad = q \Big ( ( 1 - d ) ( \mathcal { R } - \bar { V } ) _ { + } + d ( \bar { V } - \mathcal { R } ) _ { + } \Big ) = q H , } \end{array}\tag{158}
$$

where the last step applies (156). When $\mathcal { R } > \bar { V }$ , (158) holds with equality.

Analogously, if $\tilde { d } \geq 0 . 5$ , then $d = 1$ and $q = { \dot { d } } .$ , so substituting this into (157) and using $[ \cdot ] _ { + } \geq 0$ gives

$$
\begin{array} { r l } & { S = ( 1 - \tilde { d } ) ( \mathcal { R } - \bar { V } ) _ { + } + \tilde { d } ( \bar { V } - \mathcal { R } ) _ { + } = ( 1 - \tilde { d } ) ( \mathcal { R } - \bar { V } ) _ { + } + q ( \bar { V } - \mathcal { R } ) _ { + } } \\ & { \qquad \geq q ( \bar { V } - \mathcal { R } ) _ { + } } \\ & { \qquad = q \Big ( d ( \bar { V } - \mathcal { R } ) _ { + } + ( 1 - d ) ( \mathcal { R } - \bar { V } ) _ { + } \Big ) = q H , } \end{array}\tag{159}
$$

where the last step applies (156). When $\mathcal { R } < \bar { V }$ , (159) holds with equality.

Taken together, (158) and (159) establish (153) and show that the inequality is tight. This concludes the proof.

Note that setting $m = 0$ in (151) recovers (105) in Lemma C.7.

Proposition C.10 (Calibration for decisive soft deferral rules). Assume $Y \perp Z _ { 1 : t - 1 } \mid ( X , Z _ { t } )$ as in (5), and let $\mathcal { R } _ { c a s c } ^ { * }$ and $\mathcal { R } _ { c a s c } ^ { h i n g e * }$ be as in Theorem 3.2. Let $m \in [ 0 , 0 . 5 ]$ . Then, for any $\tilde { \pi } \in \bar { \Pi } _ { c a s c }$ whose deferral rules satisfy $| \tilde { d } _ { t } ( H _ { t } ) - 0 . 5 | \geq$ m almost surely for all $t \in [ T - 1 ]$ 」，

$$
\begin{array} { r } { \mathcal { R } \big ( \varphi ( \tilde { \pi } ) \big ) - \mathcal { R } _ { c a s c } ^ { * } \leq \big ( \frac { 2 } { 1 + 2 m } \big ) ^ { T - 1 } ( K - 1 ) \Big ( \mathcal { R } _ { c a s c } ^ { h i n g e } ( \tilde { \pi } ) - \mathcal { R } _ { c a s c } ^ { h i n g e * } \Big ) . } \end{array}\tag{160}
$$

Proof. Let $\textstyle \rho : = { \frac { 2 } { 1 + 2 m } } \in [ 1 , 2 ]$ . The proof follows that of Theorem 3.2 in Appendix C.4, with the constants $C _ { t + 1 }$ and $C _ { t }$ in (111) and (113) changed into $C _ { t + 1 } = \rho ^ { T - t - 1 } ( K - 1 )$ and $C _ { t } = \rho C _ { t + 1 }$ , and one subsequent change in Step 2.2. The rest of the proof stays unchanged.

In Step 2.2, we apply Lemma C.9 in place of Lemma C.7. By Lemma C.9, the factor of 2 in (120) becomes $2 / ( 1 + 2 | \tilde { d } _ { t } ( h _ { t } ) - 0 . 5 | )$ , giving the following counterpart of (120):

$$
\begin{array} { r l } & { \Delta _ { t } ( h _ { t } ) \leq \frac { 2 } { 1 + 2 \left| \tilde { d } _ { t } ( h _ { t } ) - 0 . 5 \right| } \underbrace { \left( ( 1 - \tilde { d } _ { t } ( h _ { t } ) ) \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) + \tilde { d } _ { t } ( h _ { t } ) \mathbb { E } \big [ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \big ] - \mathsf { m } \right) } _ { = : S _ { t } ( h _ { t } ) } } \\ & { \qquad + \left( \mathsf { m } - V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) \right) , } \end{array}\tag{161}
$$

where recall from (115) that $\mathsf { m } : = \operatorname* { m i n } \left\{ \mathcal { R } _ { t } ( h _ { t } ; f _ { t } ) , \mathbb { E } \left[ V _ { t + 1 } ^ { \pi } ( H _ { t + 1 } ) \mid H _ { t } = h _ { t } \right] \right\}$ and from (116) and (118) that $\Delta _ { t , \mathrm { i n t e r n a l } } ( h _ { t } ) = \mathsf { m } - V _ { t } ^ { \pi _ { \mathrm { c a s c } } ^ { * } } ( h _ { t } ) \geq 0$ . Note that $S _ { t } ( h _ { t } ) \geq 0$ since min $\{ a , b \} \leq ( 1 - \lambda ) a + \lambda b$ for every $\lambda \in [ 0 , 1 ]$ . Thus, together with $2 / ( 1 + 2 | \tilde { d } _ { t } ( h _ { t } ) - 0 . 5 | ) \leq \rho$ for almost every $h _ { t }$ by assumption, we can write

$$
\begin{array} { r l } & { \Delta _ { \mathrm { t } } ( h _ { k } ) \leq \rho \Big ( ( 1 - \tilde { d } _ { i } ( h _ { k } ) ) \mathcal { R } _ { 1 } ( h _ { k } ; f _ { k } ) + \tilde { d } _ { i } ( h _ { k } ) \mathbb { E } \big [ V _ { k + 1 } ^ { \pi } ( H _ { i + 1 } ) \mid H _ { i } - h _ { k } \big ] - \boldsymbol { m } \Big ) + \big ( m - V _ { k } ^ { \pi _ { \alpha \alpha \alpha } } ( h _ { k } ) \big ) } \\ & { \quad = \rho \Big ( ( 1 - \tilde { d } _ { i } ( h _ { k } ) ) \mathcal { R } _ { 1 } ( h _ { k } ; f _ { k } ) + \tilde { d } _ { i } ( h _ { k } ) \mathbb { E } \big [ V _ { k + 1 } ^ { \pi } ( H _ { i + 1 } ) \mid H _ { i } - h _ { k } \big ] \Big ) - ( \rho - 1 ) \boldsymbol { m } - V _ { i } ^ { \pi _ { \alpha \alpha \alpha } } ( h _ { k } ) } \\ & { \quad = \rho \Big ( ( 1 - \tilde { d } _ { i } ( h _ { k } ) ) \mathcal { R } _ { 1 } ( h _ { k } ; f _ { k } ) + \tilde { d } _ { i } ( h _ { k } ) \mathbb { E } \big [ V _ { k + 1 } ^ { \pi } ( H _ { i + 1 } ) \mid H _ { i } - h _ { k } \big ] \Big ) } \\ & { \quad \quad - ( \rho - 1 ) \Big ( \Delta _ { \mathrm { t } , \mathrm { i n d e m a l ~ ( h _ { k } ) } } ^ { \pi } + V _ { k } ^ { \pi } \pi _ { \alpha \alpha } ^ { 2 } ( h _ { k } ) \Big ) - V _ { k } ^ { \pi _ { \alpha \alpha \alpha } } ( h _ { k } ) } \\ &  \quad = \rho \Big ( ( 1 - \tilde { d } _ { i } ( h _ { k } ) ) \mathcal { R } _ { 1 } ( h _ { k } ; f _ { k } ) + \tilde { d } _ { i } ( h _ { k } ) \mathbb { E } \big [ V _ { k + 1 } ^ { \pi } ( H _ { i + 1 } ) \mid \end{array}\tag{162}
$$

where the inequality (i) follows from $\rho - 1 \geq 0$ and $\Delta _ { t , \mathrm { i n t e r n a l } } ( h _ { t } ) \geq 0$ . The RHS of (162) matches the RHS of (121) with $\rho$ in place of 2. Step 2.3 and Step 2.4 then go through with $\rho$ in place of $2 .$ The domination argument in Step 2.4 still holds: since $\rho \geq 1$ implies $\rho ^ { T - t - 1 } \geq 1$ for $t \leq T - 1$ the constants $K - 1 , C _ { t + 1 } = \rho ^ { T - t - 1 } ( K - 1 )$ and $\textstyle { \frac { K - 1 } { K } }$ are all dominated by $C _ { t + 1 }$ . This completes the backward induction in Step 2 with stage-wise calibration constants $C _ { t + 1 } = \rho ^ { T - t - 1 } ( K - 1 )$ and $C _ { t } = \rho C _ { t + 1 }$

The rest of the proof follows, yielding $C _ { 1 } = \rho ^ { T - 1 } ( K - 1 )$ which proves (160).

## D Synthetic Experiments

## D.1 Details for the Synthetic Experiments

The class-probability functions $\theta ( x , z _ { t } )$ are defined piecewise over the input space $\mathcal { X } .$ The goal is to induce regions where deferring to later classifiers improves accuracy enough to ofset the routing cost. The parameters of the posterior distribution $P _ { Y | X , Z _ { 2 } , Z _ { 3 } } = \mathrm { C a t e g o r i c a l } { \big ( } \theta ( x , z _ { 2 } , z _ { 3 } ) { \big ) }$ are given by;

$$
\theta _ { y } ( x , z _ { 2 } , z _ { 3 } ) = \frac { \exp ( \ell _ { y } ( x , z _ { 2 } , z _ { 3 } ) ) } { \sum _ { k = 1 } ^ { 3 } \exp ( \ell _ { k } ( x , z _ { 2 } , z _ { 3 } ) ) } ,\tag{163}
$$

$$
\ell _ { 1 } ( x , z _ { 2 } , z _ { 3 } ) = 1 . 2 - 4 | x | ,\tag{164}
$$

$$
\ell _ { 2 } ( x , z _ { 2 } , z _ { 3 } ) = - 0 . 2 + 2 . 2 x + 3 g ( x ) z _ { 2 } ,\tag{165}
$$

$$
\ell _ { 3 } ( x , z _ { 2 } , z _ { 3 } ) = - 0 . 2 - 2 . 2 x + 3 g ( x ) z _ { 3 } ,\tag{166}
$$

$$
g ( x ) = \sigma ( 1 0 ( | x | - 0 . 1 5 ) ) ,\tag{167}
$$

$$
\sigma ( u ) = \frac { 1 } { 1 + e ^ { - u } } .\tag{168}
$$

The posterior distributions $P _ { Y | X } , P _ { Y | X , Z _ { 2 } }$ and $P _ { Y | X , Z _ { 3 } }$ needed to compute the Bayes-optimal risks are estimated using 5000 Monte Carlo samples.

Model architecture. All models $( f _ { 1 } ( x ) , f _ { 2 } ( x , z _ { 2 } ) , f _ { 3 } ( x , z _ { 3 } ) , r ( x ) , d _ { 1 } ( x ) , d _ { 2 } ( x , z _ { 2 } ) )$ are multilayer perceptrons with ReLU activations, 3 hidden layers of width 128, and batch normalization. We observe that including batch normalization speeds up training but does not change performance, so its use is optional. Each classifier outputs $K - 1$ scores, with the final class score determined by the constraint that the scores sum to zero, as required by the multiclass hinge formulation.

Training. We use 100, 000 training, 1000 validation, and 2000 test samples. Models are trained using the AdamW optimizer with a learning rate of $1 0 ^ { - 3 }$ , a batch size of 512, and a maximum of 50 epochs, with early stopping based on validation performance with a patience of two epochs.

## D.2 Empirical Risk Minimization in Synthetic Dataset for Various Costs

We report the empirical risk of the learned models alongside the optimal empirical risk throughout training in Fig. 6, for the synthetic experiment presented in Section 5.1. We present both the cascade and the flat setting, for costs $\kappa _ { 1 } = 0 , \kappa _ { 2 } = \kappa _ { 3 } = \kappa = 0 . 0 3 \mathrm { ~ a n d ~ } \kappa _ { 1 } = 0 , \kappa _ { 2 } = \kappa _ { 3 } = \kappa = 0 . 0 6$ . We observe that training with the surrogate losses in both cases attains the optimal risk on the test set.

![](images/af3c9ae9096bbdab94d432feae9ceeb0213fba6645d41226ab3961515b070e73.jpg)  
Figure 6: Empirical risk of the learned model trained with surrogate loss in the flat case (two left figures) and the cascade case (two right figures), with varying costs per classifier. In all cases, the empirical risk evaluated on the test set of the learned model converges to the empirical risk of the Bayes-optimal solution.

## D.3 Accuracy of Classifiers and Router in Synthetic Dataset

In this experiment, we examine how the accuracy of the router and classifiers evolves during training in the synthetic experiment with flat routers presented in Section 5.1. As shown in Fig. 7, the accuracy of the classifiers quickly rises and plateaus, while the routing accuracy takes longer to converge.

![](images/051de594cb6ca47c24f8c37bc412d7ff9cb756047d084659a0f3add6ddf2ba0d.jpg)  
Figure 7: Classifier and routing-function accuracy during training in the flat router setting with $\kappa = 0 . 0 0 5$ . The router is slower to learn compared to the classifiers.

## E Post-Training PPI Correction

## E.1 Constructing Confidence Intervals Using Unlabeled Data with PPI

In this section, we detail the CI construction in (23) and discuss the result in (27). For a fixed $x \in \mathcal { X }$ , the CI for $\delta _ { t } ( x )$ in (23) is centered around $\delta _ { t } ^ { \mathrm { P P } } ( x )$ with half-width $w _ { t , N _ { b } , l _ { b } } ( x )$ , where $N _ { b }$ and $l _ { b }$ refer to the number of unlabeled and labeled samples w.r.t. the considered bin, respectively. We construct $\delta _ { t } ^ { \mathrm { P P } } ( x )$ and $w _ { t , N _ { b } , l _ { b } } ( x )$ as follows. Let

$$
\Delta _ { t } ( x , z _ { t } , z _ { r _ { \mathrm { f a t } } ( x ) } , y ) = \mathbb { 1 } \big \{ f _ { r _ { \mathrm { f a t } } ( x ) } ^ { \dagger } ( x , z _ { r _ { \mathrm { f a t } } ( x ) } ) \neq y \big \} - \mathbb { 1 } \big \{ f _ { t } ^ { \dagger } ( x , z _ { t } ) \neq y \big \} + c _ { r _ { \mathrm { f a t } } ( x ) } - c _ { t } ,\tag{169}
$$

then we define

$$
\begin{array} { r l } & { \delta _ { t } ^ { \mathrm { P P } } ( x ) : = \displaystyle \frac { 1 } { N _ { b } } \sum _ { i = 1 } ^ { N _ { b } } \underbrace { \Delta _ { t } \big ( x , z _ { t } ^ { ( i ) } , z _ { r _ { \mathrm { f a t } } ( x ) } ^ { ( i ) } , f _ { T } ^ { \dagger } ( x , z _ { T } ^ { ( i ) } ) \big ) } _ { \mathrm { T e r m ~ 1 } } } \\ & { \qquad - \displaystyle \frac { 1 } { l _ { b } } \sum _ { j = 1 } ^ { l _ { b } } \underbrace { \left[ \Delta _ { t } \big ( x , z _ { t } ^ { ( j ) } , z _ { r _ { \mathrm { f a t } } ( x ) } ^ { ( j ) } , f _ { T } ^ { \dagger } ( x , z _ { T } ^ { ( j ) } ) \big ) - \Delta _ { t } \big ( x , z _ { t } ^ { ( j ) } , z _ { r _ { \mathrm { f a t } } ( x ) } ^ { ( j ) } , y ^ { ( j ) } \big ) \right] } _ { \mathrm { T e r m ~ 2 } } , } \end{array}\tag{170}
$$

where the sum over Term 1 is evaluated using unlabeled data, while the sum over Term 2, called the rectifier [Angelopoulos et al., 2023], is evaluated using labeled data. We use the most expensive classifier $f _ { T } ^ { \dagger }$ to generate predictions on the unlabeled samples, since by design it is generally the most powerful classifier among $f _ { 1 : T } ^ { \dagger }$

The resulting CI has half-width $w _ { t , N _ { b } , l _ { b } } ( x ) : = 1 . 9 6 \sqrt { \hat { \sigma } _ { f } ^ { 2 } / N _ { b } + \hat { \sigma } _ { f - Y } ^ { 2 } / l _ { b } }$ , where $\hat { \sigma } _ { f } ^ { 2 }$ and $\hat { \sigma } _ { f - Y } ^ { 2 }$ are the empirical variances of the summands of Term 1 and Term 2, respectively. When the imputing classifier $f _ { T } ^ { \dagger }$ is accurate, the $\hat { \sigma } _ { f - Y } ^ { 2 } / l _ { b }$ term is small. The unlabeled-data term $\hat { \sigma } _ { f } ^ { 2 } / N _ { b }$ decreases as the number of unlabeled samples $\dot { N } _ { b }$ grows. In this case, taken together, the CIs are typically narrow and do not straddle zero, yielding the guarantee in (26) which says $r _ { \mathrm { f l a t } } ^ { \mathrm { P P } } ( x )$ yields a lower 0-1-c risk than the baseline $r _ { \mathrm { { f l a t } } } ( x )$ does. When, in addition, the CI for $\delta _ { r _ { \mathrm { f l a t } } ^ { \dagger } ( x ) } ( x )$ is suficiently small to separate it from the next runner-up, we obtain the guarantee in (27) which says $r _ { \mathrm { f l a t } } ^ { \mathrm { P P } } ( x ) = r _ { \mathrm { f l a t } } ^ { \dagger } ( x )$

## E.2 Additional PPI Experiments: Impact of Labeled Data Budget on Performance

We provide additional figures illustrating the behavior of the PPI algorithm for the setting presented in Section 5.2 as the amount of training and PPI labeled data varies, ranging from $n , l = 1 0 0$ to $n , l = 3 0 0 0$ samples (Fig. 8 from (a) to (d)), while the amount of unlabeled data remains fixed. As expected, with fewer labeled samples (100 and 200 shown in Fig. 8 (a) and (b), respectively), neither the trained routing model nor the classifiers can reach Bayes-optimality. The trained routing model $r _ { \mathrm { { f l a t } } }$ exhibits large errors w.r.t $r _ { \mathrm { f l a t } } ^ { \dagger } .$ , as indicated by the larger gray shaded regions. The error of the trained classifiers is reflected in the associated oracle router $r _ { \mathrm { f l a t } } ^ { \dagger } .$ , which is also far from the Bayes-optimal router $r _ { \mathrm { f l a t } } ^ { \ast } ,$ as indicated by the blue shaded regions (20.6% and 12.5% disagreement). In this regime, although the PPI correction yields larger relative improvements $\left( \mathrm { w . r . t . } \ r _ { \mathrm { f l a t } } ^ { \dagger } \right)$ , the overall routing remains farther from being Bayes-optimal due to the poor quality of the initial classifiers.

As the number of labeled samples increases (1000 and 3000 shown in Fig. 8 (c) and (d), respectively), the oracle router more closely approximates the Bayes-optimal router, and the trained routing model also more closely approximates the oracle. At $n = 1 0 0 0$ , the trained router still fails to capture the region around $x = - 0 . 6 0$ , where the oracle routes samples to the first classifier. With the PPI correction applied, part of this region is now correctly routed to $f _ { 1 }$ . At $n = 3 0 0 0$ , there is suficient data to learn both the classifiers and the router; the oracle router is very close to the Bayes-optimal router, the trained router is also very close to the oracle, and the PPI correction leaves the routing unchanged.

We also repeat the experiment but reuse the training data as labeled PPI data, instead of separating the labeled set into two independent partitions (Fig. 9). In that setting, the theoretical guarantee of Theorem 4.1 no longer holds as the classifiers are trained on the same data used for the correction, and are therefore not independent anymore. In practice, however, this violation has little impact, and performance is only slightly lower than with separate labeled sets (Fig. 8).

In summary, the PPI correction improves routing when the trained router is suboptimal, and its ability to recover the Bayes-optimal routing model depends on the quality of the underlying trained classifiers. The same trend holds when the training data is reused for the PPI correction. When suficient labeled data is available to train near-optimal classifiers, the correction more closely approaches the Bayes-optimal routing behavior.

![](images/53aff1b1101dd86cd220a96d6ceb6dd1667c3249d2d297f3904822633a818ad0.jpg)

![](images/c9c01893ee1a9ab4ae980e8bd64364eda0bae43bb97eaca8db9b2b152dfa5311.jpg)  
(a) 100 training samples  
(b) 200 training samples

![](images/b7e3b8aa623bfc39f28511d02ebd2f09b35b1320802e15f0a9cb2e2f7b11232d.jpg)  
(c) 1000 training samples

![](images/0153708fe0ab8058784e5d6b21de07be4fef0f24d1a1588bfe8db012cd85e66d.jpg)  
(d) 3000 training samples

Figure 8: Comparison of the Bayes-optimal routing function $r _ { \mathrm { f l a t } } ^ { \ast }$ , the router learned on labeled data $r _ { \mathrm { { f l a t } } }$ , and the PPI-corrected router $r _ { \mathrm { f l a t } } ^ { \mathrm { P P } }$ , for increasing numbers $n = l$ of labeled training and PPI samples. The error of the oracle w.r.t. the Bayes-optimal $r _ { \mathrm { f l a t } } ^ { \ast }$ is shown in blue shaded regions, the error of the trained router w.r.t. the oracle router is shown in gray shaded regions, and the fixes and errors introduced by PPI are computed w.r.t. the oracle router and shown by the green and red shaded regions. Note that these are computed pointwise for every input $x ,$ whereas the PPI correction is applied at the bin level, so the errors and fixes also reflect the efect of binning.

![](images/b3b3ba8064f79df0a57360d30bd67c066de15c28db332e70af08c3c2026865ab.jpg)

![](images/8a19d4d52e3f9fe9771788d54d27c6bf75fc6946c2e97e6f2ba39d927184a3db.jpg)

![](images/f646a82afea88042e2e9e75315c3c14751af3cecbfdbd903f47c7d8963d7988e.jpg)

![](images/7bbe49bc3302f77826ad13f7b777fb824a5b61213afe0f4af2540eaba54d8b6a.jpg)

![](images/b19542fd3e12bc4c3d452e6cca5429c80c85f359f0c7e0daf00f5ba9a0c7994e.jpg)

![](images/75b59f7642ca2a6b2bc149ab3ab5f44fb2f8b16c3f73bd05eae7a6704c581e9b.jpg)

![](images/1e79335b2a7c3ff178eecb02c61d03bddd11b1e8ce2df91854748720aad4a69a.jpg)  
(a) 100 training samples

![](images/2c6ea4a07a1bffcf726f2cd2d98b54fe9d0d825f6c5c3fcba98fe6ebf8245b11.jpg)  
(b) 200 training samples

![](images/71531ee22e3b65c220dca91d7003550383fda4768d4888b8d052bf8e809cfd68.jpg)  
(c) 1000 training samples

![](images/7124eedd6252a70b613ebaf0ccdc9b5fe0ed25666d0d259c9b86994ad5614d85.jpg)  
(d) 3000 training samples  
Figure 9: Comparison of the Bayes-optimal routing function $r _ { \mathrm { f l a t } } ^ { \ast }$ , the router learned on labeled data r , and the PPI-corrected router $r _ { \mathrm { f l a t } } ^ { \mathrm { P P } }$ , for increasing numbers n of labeled samples, when the same labeled data is used both for training and for the PPI correction.

## F Details for LLM Experiments

## F.1 Datasets

We evaluate the utility of our proposed training losses on two multiple-choice question-answering benchmarks that span commonsense reasoning and mathematical problem solving.

CommonsenseQA. CommonsenseQA [Talmor et al., 2019] is a dataset designed to test common sense knowledge and reasoning. Each question is a 5-way multiple choice problem (K = 5), where the goal is to identify semantically related concepts. The dataset contains 12,102 questions covering diverse common sense relations such as causes, efects, and properties of everyday concepts. Questions require reasoning beyond surface-level pattern matching, as distractors are carefully designed to be semantically similar to the correct answer.

GSM8K. GSM8K [Cobbe et al., 2021] is a dataset of 8,792 grade school math word problems designed to evaluate multi-step mathematical reasoning with final answers being numerical values.

Since GSM8K is originally a free-form numerical answer dataset, we convert it to a 4-way multiple choice format (K = 4) to match our classification framework. For each question, we generate distractors using the following procedure:

1. Distractor generation from weak model: We sample 20 chain-of-thought responses from a weak model (Qwen-1.5B) for each question. We collect all incorrect numerical answers produced by the model and normalize them (handling commas, decimal formatting, currency symbols).

2. Frequency-based distractor selection: We select the 3 most frequently occurring wrong answers as distractors, ensuring each distractor difers from the correct answer and other distractors to avoid duplicates.

3. Synthetic fallback: If the weak model does not produce enough distinct wrong answers, we supplement the answer pool with synthetic distractors based on common error patterns (e.g., of-by-one errors, order-of-magnitude mistakes, and arithmetic sign errors).

4. Random ordering: The correct answer position is randomized across the four labels A to D with a deterministic seed per question to ensure reproducibility.

This conversion procedure yields realistic distractors that reflect actual model errors rather than arbitrary numerical alternatives, making the resulting multiple-choice problems more representative of model failure modes.

Data splitting. For both datasets, we partition the data into training, validation, and test sets using an 80-10-10 split with a fixed random seed for reproducibility. The validation set is used for hyperparameter selection and early stopping, while the test set is held out for final evaluation.

## F.2 Model and Training Details

Setup and feature extraction. We mine LLM input-output pairs from natural language prompts and their natural language responses from a community repository [Rumiantsev et al., 2025] covering our multiple choice datasets (this repository also contains ground truth answers). To convert the natural language prompts and responses to fixed size vectors we use a sentence embedding language model.<sup>3</sup> Specifically, each sample is represented by a prompt embedding $x \in \mathbb { R } ^ { 7 6 8 }$ , obtained from the sentence embedding of the prompt text and answer embeddings $z _ { t } \in \mathbb { R } ^ { 7 6 8 }$ for $t \in [ T ]$ . Note that this means that the LLMs as well as the sentence embedding model are used to generate features $( x , z _ { t } )$ for our classifiers $f _ { t }$ which we describe in the next paragraph (the LLMs are not the classifiers themselves). For $t = 1 , z _ { 1 }$ is the embedding of the answer produced by the first LLM, so $z _ { 1 }$ carries information in these experiments. The prompt embedding x is shared by all classifiers and each $z _ { t }$ is specific to the t-th LLM, so $( x , z _ { t } )$ for $t > 1$ does not contain $z _ { 1 }$ . The flat router observes only x.

Architecture of classifier $\tilde { f } _ { t }$ . We describe the architecture of the classifiers used in the LLM experiments in Eq. (171). The classifier is a feedforward neural network that takes as inputs $x , z _ { t }$ and outputs K scores that sum to zero:

$$
\begin{array} { r } { \tilde { f } _ { t } ( x , z _ { t } ) = \Theta _ { 3 } ^ { ( t ) } \mathrm { R e L U } \left( \mathrm { B N } \left( \Theta _ { 2 } ^ { ( t ) } \mathrm { R e L U } \left( \mathrm { B N } \left( \Theta _ { 1 } ^ { ( t ) } [ x ; z _ { t } ] \right) \right) \right) \right) . } \end{array}\tag{171}
$$

Here, BN denotes batch normalization, and $[ \cdot ; \cdot ]$ denotes concatenation. The superscript (t) denotes that parameter matrices $\Theta _ { k } ^ { ( t ) }$ belong to the classifier network $\tilde { f } _ { t }$ . The classifier parameters have dimensions $\Theta _ { 1 } ^ { ( t ) } \in \mathbb { R } ^ { 1 2 8 \times 1 5 3 6 }$ $\Theta _ { 2 } ^ { ( t ) } \in \mathbb { R } ^ { 1 2 8 \times 1 2 8 }$ , and $\Theta _ { 3 } ^ { ( t ) } \in \mathbb { R } ^ { ( K - 1 ) \times 1 2 8 }$ . The classifier outputs $K - 1$ scores, with the final class score determined by the constraint that the scores sum to zero. During training, batch normalization normalizes each hidden layer with the mean and variance of the current mini-batch, so the output of a network for one sample also depends on the other samples in its mini-batch (during training only, not inference). While this slightly deviates from $\widetilde { \Pi } _ { \mathrm { { f l a t } } }$ and $\widetilde { \Pi } _ { \mathrm { c a s c } }$ definition as additional components such as normalization or regularization were not included in the optimization that our theory considers, it is important to verify that our proposed approach works in a standard neural network setting with the usual components such as batch normalization.

Architecture of flat router. The soft router $\tilde { r } ( x )$ outputs a probability distribution over the classifiers. It is implemented as a feedforward neural network:

$$
\tilde { r } ( x ) = \mathrm { S o f t m a x } \left( \mathrm { B N } \left( \Theta _ { 2 } ^ { ( r ) } \mathrm { R e L U } \left( \mathrm { B N } \left( \Theta _ { 1 } ^ { ( r ) } x \right) \right) \right) \right) .\tag{172}
$$

The superscript $( r )$ in $\Theta _ { k } ^ { ( r ) }$ denotes that we refer to the parameters of the router. The router parameters have dimensions $\Theta _ { 1 } ^ { ( r ) } \in \mathbb { R } ^ { 1 2 8 \times 7 6 8 }$ and $\Theta _ { 2 } ^ { ( r ) } \in \mathbb { R } ^ { T \times 1 2 8 }$

Architecture of cascade router. The cascade consists of similar stage-specific classifiers but with deferral routing modules implemented as feedforward neural networks.

The soft deferral rule implemented in the deferral routing modules $\tilde { d } _ { t } ( x , z _ { 1 : t } )$ takes the full history available up to stage t, namely the question embedding and all answer embeddings observed so far, and outputs a deferral probability. A deferral module is defined as

$$
\tilde { d } _ { t } ( \boldsymbol { x } , z _ { 1 : t } ) = \sigma \left( \mathrm { B N } \left( \Phi _ { 2 } ^ { ( t ) } \mathrm { R e L U } \left( \mathrm { B N } \left( \Phi _ { 1 } ^ { ( t ) } [ \boldsymbol { x } ; z _ { 1 : t } ] \right) \right) \right) \right) ,\tag{173}
$$

where σ denotes the sigmoid function. The deferral module at stage t has input dimension $7 6 8 + 7 6 8 t =$ $7 6 8 ( 1 + t )$ , so it is parameterized by $\Phi _ { 1 } ^ { ( t ) } \in \mathbb { R } ^ { 1 2 8 \times 7 6 8 ( 1 + t ) }$ and $\Phi _ { 2 } ^ { ( t ) } \in \mathbb { R } ^ { 1 \times 1 2 8 }$ . There are only $T - 1$ deferral modules, since the final stage always exits if reached.

Batch Normalization. Batch normalization is optional in our setup. As in the synthetic experiments, we observe that it only speeds up training and does not change performance significantly.

Training. Both the flat and the cascade architectures are trained using the AdamW optimizer with learning rate $1 0 ^ { - 3 }$ and mini-batch size 32. Training runs for a maximum of 1000 epochs for the cascade model and 100 epochs for the router model, with early stopping using a patience of 20 epochs based on validation loss. We employ the proposed hinge-based surrogate loss that balances classification accuracy with routing costs. Training takes approximately 3 hours per dataset with the majority of the time spent on generating embeddings from model answers using an Nvidia GeForce RTX 3060 12GB card.

Evaluation protocol. We report performance using 10 bootstrap trials on the held-out test set. For each trial, we resample the test set with replacement and compute the accuracy of each classifier on all test samples and on the samples routed to it, which for the cascade are the samples that exit at it. We also compute the fraction of samples routed to each classifier, or deferred past each stage of the cascade. This bootstrap procedure provides confidence intervals that account for finite sample variability.

Algorithm implementation. We implement our method in PyTorch and make our code available: https://github.com/networkslab/Consistent-Classifier-Routing.

## G Additional LLM Experiments

## G.1 LLM Experiments with Additional Models and Datasets

In this appendix we extend the experiments of Sec. 5.3 in two directions. We first train a cascade router with the LLaMA classifiers on GSM8K, and we then repeat the flat-router experiment of Sec. 5.3 with Qwen classifiers on CommonsenseQA, which changes both the model family and the dataset.

Cascade router with LLaMA classifiers. We use the three cost configurations of Sec. 5.3, so the per-classifier costs range from [0, 0.01, 0.02] in Configuration 1 to [0, 0.1, 0.2] in Configuration 3. A prompt pays for every classifier it passes through, so an answer from LLaMA 405B costs $\kappa _ { 1 } + \kappa _ { 2 } + \kappa _ { 3 }$ , from 0.03 in Configuration 1 to 0.3 in Configuration 3. Fig. 10 shows how the trained cascade spends this budget. In its bottom row, the fraction of prompts deferred past LLaMA 1B falls from about 66% in Configuration 1 to about 36% in Configuration 3, so the share answered by LLaMA 1B grows from about one third to about two thirds. The fraction deferred past LLaMA 70B, which is the share answered by LLaMA 405B, falls from about 60% to about 10%. The top row shows that LLaMA 70B and LLaMA 405B answer at least 90% of the prompts they receive correctly in every configuration, close to their accuracy on all prompts. For LLaMA 1B, the median accuracy on the prompts it answers stays within 4 percentage points of its accuracy on all prompts, so the first deferral module does not single out prompts that LLaMA 1B answers better than average. In contrast, the flat router of Fig. 4 sends only about 9% of the prompts to LLaMA 1B in Configuration 1, and LLaMA 1B answers about 66% of them correctly against about 49% over all prompts.

![](images/74265ddc966874df87e2be80fc37448f781df22830364aebab271ab3e48dee8c.jpg)  
Figure 10: Classifier accuracy and deferral fractions of the cascade router with LLaMA classifiers on GSM8K across three cost configurations (costs increasing from left panel to right panel). The top row shows the accuracy of each classifier on all test prompts and on the prompts it answers, and the bottom row shows the fraction of prompts deferred at each classifier.

Flat router with Qwen classifiers. We repeat the flat-router experiment of Sec. 5.3 with Qwen classifiers on CommonsenseQA and the same three cost configurations, which changes both the model family and the dataset. Fig. 11 shows the results. In the bottom row, the share of prompts routed to Qwen 1.5B stays near 30% in Configurations 1 and 2 and rises to 74% in Configuration 3, while the share routed to Qwen 32B falls from 38% to 9%. In the top row, Qwen 32B is the most accurate classifier on all prompts at 91%, while Qwen 72B reaches 83%. In Configuration 1, the prompts routed to Qwen 1.5B are answered correctly 83% of the time against 76% over all prompts, so the router keeps easier prompts for the cheapest classifier, as the flat router of Fig. 4 does for LLaMA 1B. In Configuration 3, Qwen 1.5B answers most prompts, and its accuracy on them falls to about its accuracy on all prompts. These results suggest that the behavior of our training loss is not specific to one model family.

![](images/85450fdfff56f2c9c71bc6e7d4e56bcb90938183c687489cbc7123e62534593c.jpg)  
Figure 11: Classifier accuracy and routing fractions of the flat router with Qwen classifiers on CommonsenseQA across three cost configurations (costs increasing from left panel to right panel). The top row shows the accuracy of each classifier on all test prompts and on the prompts routed to it, and the bottom row shows the fraction of prompts routed to each classifier.

## G.2 Comparison to Supervised LLM Cascade Baselines

Our work is primarily theoretical, but in LLM multiple-choice question answering it can be compared directly with applied cascade methods. In the cascade router setting, we compare it with FrugalGPT [Chen et al., 2024a] and TREACLE [Zhang et al., 2024] in one configuration, with Qwen classifiers on CommonsenseQA and per-classifier costs κ = [0, 0.05, 0.06] in order of model size. We choose these two methods because both learn when to escalate along a cascade of fixed LLMs under a cost budget, which makes them the separate-training counterpart of our setting. FrugalGPT uses a cascade of LLM APIs within a budget constraint [Chen et al., 2024a], and TREACLE casts the choice of LLM and prompt as budget-constrained policy optimization [Zhang et al., 2024]. Other LLM cascade routing methods do not fit this setup at all for structural reasons. C3PO [Valkanas et al., 2025b] does not admit direct comparison as it is an unsupervised technique, whereas the methods we compare to all require direct supervision. RouteLLM [Ong et al., 2025] learns from human preference data to route between exactly two LLMs, so it is a two-model router rather than a cascade of classifiers. RouterDC [Chen et al., 2024b] trains a query-only flat router to select one LLM per query (so it does not support a cascade router setup). It keeps the LLMs fixed and does not train classifiers on their outputs or optimize a 0-1-c objective, so it addresses a diferent problem. Our method and Joint CE use the Qwen cascade router, which is replicated for FrugalGPT and TREACLE cascades that query the Qwen models from the cheapest to the most expensive using the same costs. Both FrugalGPT and TREACLE keep the classifiers fixed and train only the routing rule, while our method trains the classifiers and the router jointly. Note that the asymmetry in optimization objectives is deliberate, as this indicates the value added by joint training versus frozen classifiers. Joint classifier - router training methods are expected to perform better and our results confirm this.

Table 1 reports three quantities on the test set for each method. The error rate is the fraction of test prompts answered incorrectly, and the average cost is the mean over the test set of the cost term $c _ { r ( h ) }$ in (9). The test empirical risk is the sum of the error rate and average cost, which is the empirical 0-1-c risk of Section 2.2, the risk our guarantees concern. Joint Cross Entropy (CE) trains the classifiers and the router jointly with the ordinary cross-entropy loss, as described in Appendix G.3. All methods use the same train-test split. Our method attains the lowest test empirical risk.

Table 1: Error rate, average cost and test empirical risk of each method with Qwen classifiers on the test set of CommonsenseQA. FrugalGPT and TREACLE are run with a target cost budget of 0.03, which acts as a soft constraint, so their average cost can exceed it slightly. Our method attains the lowest test empirical risk. Bold indicates statistically significant diference between the two best models via pairwise Wilcoxon test with 10 trials (our method achieves statistical significance at $p < 0 . 0 5$ level).
<table><tr><td>Method</td><td>Error Rate</td><td>Avg Cost</td><td>Test Empirical Risk</td></tr><tr><td>Hinge (ours)</td><td>0.1626</td><td>0.0344</td><td>0.1970</td></tr><tr><td>Joint CE</td><td>0.1870</td><td>0.0334</td><td>0.2204</td></tr><tr><td>FrugalGPT [Chen et al., 2024a]</td><td></td><td>0.2195 0.0351</td><td>0.2546</td></tr><tr><td>TREACLE [Zhang et al., 2024]</td><td>0.1951</td><td>0.0328</td><td>0.2279</td></tr></table>

The results show that joint model-router training under our proposed hinge-based loss provides meaningful improvements in empirical risk, while incurring similar cost to the other methods.

## G.3 Comparison to Adaptive Inference Baselines

Besides the empirical works which we compared to in Appendix G.2, one might wonder how some naive baselines (e.g. joint training with cross entropy) or theoretical techniques from the broader literature would perform in the cascade router setting. In this appendix, we set up a simple experiment and show that our proposed loss leads to better convergence and lower overall empirical risk.

We compare our proposed loss against adaptive-inference baselines on CommonsenseQA using the Qwen family. Similarly to the other experiments, the LLM weights are frozen and we learn a classifier based on prompt embeddings $x _ { i }$ and answer embeddings $z _ { i }$ . The learned classifier associated with stage t is $f _ { t } ( x , z _ { t } )$ . For baselines that use probabilistic predictions, we write

$$
\hat { p } _ { t } ( k \mid x , z _ { t } ) , \qquad k \in [ K ] ,\tag{174}
$$

for the class probabilities produced by the corresponding cross-entropy-trained classifier at stage t.   
The baselines are grouped into two categories: separate-training and joint-training baselines.

Separate-training baselines. In these baselines, the classifiers $f _ { 1 } , f _ { 2 } , f _ { 3 }$ are trained independently from the routing or deferral rule.

• Confidence Threshold (CT): For each stage t, a threshold $\tau _ { t }$ is tuned on a validation set to minimize empirical 0-1-c risk. The method exits at stage t when

$$
\operatorname* { m a x } _ { k \in [ K ] } \hat { p } _ { t } ( k \mid x , z _ { t } ) \geq \tau _ { t } ,\tag{175}
$$

and otherwise defers to the next stage.

• Confidence Threshold with cost adjustment (CT-c): This method uses the same thresholding rule as CT, but sets a cost-adjusted threshold using the estimated validation accuracy of each classifier and its cost. We follow the construction of Regol et al. (2025) [Regol et al., 2025].

• Soft Deferral: A probabilistic routing baseline that defers at stage t with probability

$$
1 - \operatorname* { m a x } _ { k \in [ K ] } \hat { p } _ { t } ( k \mid x , z _ { t } ) ,\tag{176}
$$

so lower-confidence predictions are more likely to be deferred.

• [Mozannar and Sontag, 2020]: The standard learning to defer baseline, which learns a classifier and a deferral rule that decides whether to predict or defer to a fixed expert. Its classifier and deferral rule are trained together, while the expert, here the classifier of the most expensive LLM, is trained separately.

Joint-training baselines. In these baselines, classifiers and routing decisions are trained jointly.

• Joint cross-entropy training (Joint CE): Joint training of the classifiers and the deferral modules with the loss $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \tilde { r } _ { t } ( x , z _ { 1 : t } ) \big ( \mathrm { C E } ( \tilde { f } _ { t } ( x , z _ { t } ) , y ) + \sum _ { t ^ { \prime } < t } \kappa _ { t } ^ { \prime } \big ) } \end{array}$ , where $\tilde { r } _ { t } ( x , z _ { 1 : t } )$ is the probability of exiting at stage t and $\sum { _ { t ^ { \prime } \leq t } \kappa _ { t } ^ { \prime } }$ is the cumulative cost of reaching stage $t ,$ as in (18). Here CE is the cross-entropy of the softmax of the scores $\tilde { f _ { t } } ( \boldsymbol { x } , \boldsymbol { z _ { t } } )$ . This loss replaces each hinge term of (18) by the cross-entropy and keeps the cost term, so the deferral modules are penalized for deferring to expensive classifiers.

• [Regol et al., 2025]: Joint training using the two-classifier hinge surrogate proposed by Regol et al. (2025), which corresponds to the two-model version of our proposed hinge loss.

Since Mozannar and Sontag [2020] and Regol et al. [2025] are formulated for two classifiers, we instantiate them using the cheapest and most expensive LLMs. This diference in the number of classifiers reflects the setting our contribution targets as neither of these two methods extends to more than two classifiers, which is the gap our proposed loss fills. Running our method with $T = 2$ would not give a more informative comparison, since for two classifiers the loss of Regol et al. (2025) is a special case of our hinge loss and the two would produce identical results. A diference can therefore only appear for $T > 2$ , and Appendix G.4 shows that our loss continues to reduce the risk as T grows to 5. All other baselines use three LLMs, same as ours. The router is a cascade router, so a sample that exits at stage t pays the cumulative cost $\kappa _ { 1 } + \cdots + \kappa _ { t }$ . The per-classifier costs are $\kappa = [ 0 , 0 . 0 5 , 0 . 0 6 ]$ in order of model size, and every method is evaluated by its test 0-1-c risk under these costs. Fig. 12 shows that training with our proposed loss achieves the lowest test-set risk among all baselines. A pairwise Wilcoxon test with 10 trials of our method with the best baseline (at convergence) shows that the diference is statistically significant at the $p < 0 . 0 5$ level. Additionally, the joint training methods (second column of legend of Fig. 12) outperform the separate-training methods, showing the value of training the classifiers together with the router.

![](images/91d439419bddbd935d7d59321e059a98e70150ca094f023c9c0a8e4cb122b60b.jpg)  
Figure 12: Test 0-1-c risk during training. Our method is compared with the separate-training baselines (CT, CT-c, Soft Deferral and L2D from Mozannar and Sontag 2020) and the joint-training baselines (Joint CE and Regol et al. 2025), and it reaches the lowest risk at the end of training.

## G.4 Scaling with the Number of Classifiers

In our experiments, the number of classifiers has so far been fixed at $T = 3 .$ . To evaluate whether our approach can scale beyond this setting, we consider up to $T = 5$ classifiers in the CommonsenseQA experiment (see Fig. 13). The experiment starts from a single-model setting $( T = 1 )$ , where no routing rule is required, and progressively increases the number of models to five. We use a cascade router, and the per-classifier costs increase with model size, with κ equal to 0 for Qwen-1.5B, 0.02 fo LLaMA-8B, 0.05 for Qwen-32B, 0.06 for Qwen-72B and 0.15 for LLaMA-405B. The results show a consistent reduction in risk as additional classifiers are added, with diminishing returns beyond three classifiers. This aligns with the intuition that adding more classifiers provides additional operating points along the cost-accuracy trade-of curve.

![](images/8312729b73617ff235e3f5720a2dcd7faa6e521b17db8c6f33f6b0b88d4fd7f1.jpg)  
Figure 13: Test 0-1-c empirical risk on CommonsenseQA as a function of the number of classifiers T, jointly-trained using our surrogate hinge loss. Starting from a naive single-model setting with the Qwen-1.5B model $( T = 1 )$ , where no routing rule needs to be learned, we incrementally add more models: Qwen-32B $\left( T = 2 \right)$ , Qwen-72B $( T = 3 )$ ， $\mathrm { L L a M A { - } 8 B }$ $( T = 4 )$ , and LLaMA-405B $( T = 5 )$ alongside a trainable routing rule. Risk decreases monotonically from $T = 1$ to $T = 5$ (with diminishing returns beyond $T = 3 )$ , indicating that the proposed framework scales well with the number of models and consistently benefits from additional routing options.

## H Additional Feature Variables to Represent Varying Model Capacity

In the main text, we note that the additional feature variables $Z _ { t }$ can represent hidden representations extracted by diferent architectures with varying modeling capacity. In general, hidden representations produced by larger models are more computationally expensive to obtain, but they are also often more informative. We therefore view these representations as extra “predictive signal” available to the classifier $f _ { t }$ and to the subsequent cascade routing rules.

This framing can be related to the recently introduced concepts of computationally constrained information [Xu et al., 2020] and epiplexity [Finzi et al., 2026]. These formulations provide a statistical abstraction of computational constraints, in which increased model capacity enables access to additional structured information in the data.

The standard approach to model classifiers of varying performance is to consider diferent hypothesis classes $\mathcal { F } _ { t } , \mathcal { F } _ { t + 1 }$ , where a smaller model is constrained to $\mathcal { F } _ { t }$ and a larger model to $\mathcal { F } _ { t + 1 }$ This formulation typically relies on the implicit assumption that the optimal function in $\mathcal { F } _ { t }$ is worse than that in $\mathcal { F } _ { t + 1 }$ , w.r.t. some expected loss $\mathcal { L } \mathrm { : }$

$$
\begin{array} { r } { \mathcal { L } ( f _ { t } ^ { * } ) \geq \mathcal { L } ( f _ { t + 1 } ^ { * } ) , } \end{array}\tag{177}
$$

$$
\mathrm { w h e r e } \quad f _ { t } ^ { * } \in \underset { f \in \mathcal { F } _ { t } } { \arg \operatorname* { m i n } } \mathcal { L } ( f ) \quad \mathrm { a n d } \quad f _ { t + 1 } ^ { * } \in \underset { f \in \mathcal { F } _ { t + 1 } } { \arg \operatorname* { m i n } } \mathcal { L } ( f ) .\tag{178}
$$

Instead, we adopt a computationally constrained information-based perspective: we introduce random variables X and $Z _ { 1 : T }$ , where X represents the base information accessible to every model, and $( X , Z _ { t } )$ represents the richer information available to more t-th model, which is richer for more powerful ones. As such, we model the random variables $Z _ { 1 : T }$ to be dependent. This modeling choice allows us to abstract away from the particular implementation of the architectures generating these hidden representations. However, for $Z _ { 1 : T }$ to be interpreted as the hidden representations, the generating architectures must remain fixed during training, since the distribution of $Z _ { 1 : T }$ is assumed to remain unchanged.

## I Limitations and Future Work

In this section we discuss the limitations of our work in more depth. First, while our empirical results suggest that the proposed losses are efective on real datasets, the main contribution of this work remains theoretical. Our primary goal is to provide a theoretical formulation of joint training and to propose a practical loss that is Bayes-consistent with respect to the targeted cost-sensitive zero-one loss. In practice, computational constraints can make full joint training infeasible (for instance, because it requires fitting all classifiers into memory), and simpler training schemes could still be preferable. Nevertheless even in these settings, our analysis can serve as a principled starting point for developing more scalable methods.

A second limitation of our approach is its reliance on the hinge loss. While this loss can be optimized using standard gradient-based techniques, it is less commonly used than smooth surrogate losses such as the logistic loss or the exponential loss. The hinge loss exhibits diferent optimization properties, such as non-diferentiability at zero and flat regions, which can make optimization more challenging in practice. We use the hinge loss because it yields a sharp linear transfer rate from the excess surrogate risk to the excess target risk, as our calibration bounds establish with Theorems 3.1 and 3.2. A linear transfer rate is the strongest transfer rate one can hope for [Frongillo and Waggoner, 2021]. The most natural family of functions that would preserve this behavior is the class of polyhedral losses, which are piecewise-linear convex losses, of which the multiclass hinge loss is one instance. These losses are known to admit linear transfer rates in standard classification settings [Frongillo and Waggoner, 2021, Finocchiaro et al., 2024]. We conjecture that our results extend to this broader family.

Smooth surrogates could also be used and might make optimization easier in practice. The tradeof is that they generally yield weaker guarantees. It is known that for suficiently non-polyhedral surrogates the square-root transfer rate is unavoidable [Frongillo and Waggoner, 2021], so replacing our hinge version with a smooth loss would in general forfeit the linear rate our analysis provides. Dificulties in developing a Bayes-consistent cross-entropy surrogate loss for our setting have been discussed by Regol et al. [2025], but further work in this direction could improve the practical applicability of the proposed surrogate losses.

Finally, our formulation relies on several assumptions that can be restrictive in some settings. First, we assume that the costs can be known before evaluating the classifiers. This is incompatible with autoregressive applications where the output length can vary, which would generally result in a varying cost that we cannot know in advance. Second, our analysis relies on defining a sequence of random variables $Z _ { t }$ that are progressively made accessible to the classifiers, which explains why costlier classifiers can achieve higher predictive performance. In the main text, we note that the $Z _ { t }$ can represent hidden representations computed by progressively larger-capacity models. This abstraction enables our statistical analysis, but it assumes that the distribution of those $Z _ { t }$ remains fixed during training. As a result, our framework directly applies to adaptive inference settings in which backbone networks generating the hidden representations are fixed during optimization, and only the intermediate classifiers are updated. It does not directly extend to settings where the backbone networks are also fine-tuned, since in that case the $Z _ { t }$ could no longer correspond to the hidden representations, as their distribution would evolve during training. This is a limitation of our current setup, as it does not cover the general adaptive inference setting. Extending the framework to this more general setting is an interesting direction for future work.