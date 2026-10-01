# Certification-Based Differentially Private Learning

Mihnea Ghitu

Department of Computing

Imperial College London

London, United Kingdom

mihnea.ghitu20@imperial.ac.uk

Matthew Wicker

Department of Computing

Imperial College London

London, United Kingdom

m.wicker@imperial.ac.uk

Abstract—Differential privacy (DP) in machine learning is typically achieved by adding noise to model parameters (private learning) or to model outputs (private prediction). Recent work uses formal methods, namely abstract interpretation, to provide tighter privacy guarantees, but only for private prediction in classification settings. In this work, we investigate the use of formal methods as a general tool for tighter privacy analysis. First, we generalize the abstract gradient training (AGT) framework to private prediction in continuous, unbounded regression. Second, by reducing learning in parameterized models to a regression problem over the parameter space, we introduce Abstract Gradient Sampling (AGS), an algorithm that enables reachability-based analysis to provide guarantees for private learning. In both private prediction and private learning, we provide tightened privacy accounting for the AGT framework and a theoretical analysis demonstrating when our smooth sensitivity upper-bounds yield favourable privacyutility trade-off. In practice, we validate that our regression bounds are tighter than global-sensitivity baselines on regression benchmarks, and, notably, yield the first finite privacy guarantees in settings where global prediction sensitivity is a priori unbounded. We also find that under matched conditions, our private learning algorithm can outperform standard private learners.

Index Terms—Differential Privacy, Formal Verification, Abstract Gradient Training, Private Learning, Private Prediction.

## I. INTRODUCTION

Differential privacy (DP) provides a principled framework for limiting the information revealed about individual data points through the outputs of a computation [1], [2] and serves as a necessary prerequisite to responsible deployment of learning algorithms in many domains [3]. Privacy is largely achieved through private learning, where privacy is enforced during model training by perturbing gradients [4], objectives [5], or parameters [6] despite the tendency of private learning algorithms to substantially reduce model utility [7]. An alternative that allows performant, non-private models to be used with privacy guarantees is private prediction [8]. In private prediction, privacy guarantees are enforced only at inference time by perturbing the model’s outputs according to their sensitivity. This paradigm is particularly appealing in modern machine learning pipelines, where models are often pre-trained, fine-tuned, or deployed in settings where retraining under DP constraints is impractical or undesirable [9].

While private prediction leaves the model parameters untouched, the noise added to model outputs often erodes the utility of model predictions [10]. Thus, a central challenge in private prediction is reducing the magnitude of outputprivatizing noise by obtaining tighter sensitivity bounds for model outputs [8], [11]. The use of naive global sensitivity bounds is typically overly conservative, leading to excessive noise and worse utility compared with private learning [10]. Recent promising work has connected certification-based techniques to smooth sensitivity, yielding improved sensitivity bounds [12], [13]. At its core, this offers a way to adapt the noise magnitude to the local geometry of the data and the model, and presents a compelling direction for not only tightening privacy analysis but also leveraging formal methods to inform algorithm design [14].

Unfortunately, the current works that connect formal methods to tightened privacy analysis do so only for private prediction and are restricted to analyzing classification tasks [11], [13]. To fully appreciate and unlock the potential of advances in formal methods for improved privacy analysis, we begin by extending prior works to provide tighter privacy guarantees in unbounded and/or continuous regression tasks. To achieve this, we develop a novel certification-based upper-bound on the notion of smooth sensitivity [12]. We then observe that learning algorithms in parametric models (i.e., algorithms that return parameter vectors) can themselves be viewed as unbounded, continuous regressors. We capitalize on this connection by developing Abstract Gradient Sampling (AGS). Prior work such as [15] uses Lipschitz bounds to calibrate privacy noise. AGS instead places formal methods at the core of private learning by employing abstract interpretation to soundly bound the sensitivity of each update.

By generalizing prior work that establishes this connection, we conduct a more complete empirical study of where formal methods can improve state-of-the-art privacy guarantees. Theoretically, we provide conditions under which the noise distribution from certification-based algorithms dominates those based on global sensitivity. Notably, our algorithms are, to the best of our knowledge, the first that provide finite pure differential privacy (ε-DP) private prediction guarantees when there is not a known, a priori upper bound on the global sensitivity. Further, we demonstrate that while pure-DP algorithms are unable to use advanced composition, techniques from formal methods are able to provide composition-like bounds tightening without resorting to approximate DP.

Empirically, we validate our approach across a wide range of benchmarks. We begin by studying the effect of our private prediction bounds across a toy linear regression setup and the California housing dataset. Across these benchmarks we find that our novel private regression technique, AGT-R, bests global sensitivity private prediction Private Aggregation of Teacher Ensembles (PATE) [16] by several orders of magnitude and additionally opens up a novel paradigm of tightening privacy analysis by improving formal methods techniques. To benchmark the effectiveness of the AGS algorithm, we study MNIST and the IMDB movie sentiment classification datasets. Similarly, we find that for small values of ϵ AGS significantly outperforms DP-SGD under matched algorithm conditions (same batch size, learning rate, etc). While the models for which our bounds apply are limited compared to the models analyzable by other private learning algorithms, we hope that the novel connection between formal methods and privacy in conjunction with the initial results demonstrating dominance in a subset of cases will spur future works in this direction. We summarize our paper’s contributions as follows:

• We extend the Abstract Gradient Training (AGT) framework to regression settings by deriving a novel over-approximation of smooth sensitivity for continuous-valued outputs.

• By connecting learning in parameterized models to unbounded, continuous regression, we present a novel algorithm termed Abstract Gradient Sampling (AGS) which is the first algorithm that uses formal methods as the basis for a differentially private learning algorithm.

• We theoretically show when the AGT upper bound on smooth sensitivity yields tighter DP compared to both the Laplace and Gaussian mechanisms in regression, and discuss where the AGS algorithm can enhance and improve differentially private stochastic gradient descent (DP-SGD).

• We empirically validate our approach across a hierarchy of model and dataset complexities, from linear regression and deep neural networks to foundation models. In all settings, our method can yield tighter privacy guarantees than existing DP baselines. Our work furthers the emerging connection between formal methods and privacy to the mutual benefit of both sub-fields.

## II. RELATED WORK

Private Prediction for Regression Differential privacy guarantees are typically achieved by adding noise — scaled to an algorithm’s sensitivity — to the algorithm’s output [1]. When the algorithm is a non-privately trained machine learning model, this can be done by perturbing the model’s predictions, a setting termed private prediction [8]. Global sensitivity, the largest possible change in output between neighbouring datasets [2], often yields loose bounds; tighter analyses can be obtained by reducing the global sensitivity of a given mechanism [17], [18], by exploiting properties of the specific dataset [12], or by relaxing to approximate differential privacy [19], [20]. In this work, we focus on differentially private prediction in the regression setting. When the regressor’s output is quantised, the PATE framework applies [16], [18]; when it is continuous, one can fall back on standard mechanisms based on global sensitivity, shard-and-aggregate, or data-dependent analysis [2].

To the best of our knowledge, no existing method provides finite privacy guarantees for a non-privately trained regression model whose output is neither quantised nor has an a priori known global sensitivity bound.

Private Learning Mechanisms Although private prediction offers conceptual advantages over private learning [8], its empirical performance lags behind state-of-the-art private learning mechanisms [10], [17]. Early work in private learning, like early work in private prediction, added noise to the outputs (and objective functions) of ERM algorithms [6]. Injecting noise during training instead yielded substantially better utility–privacy trade-offs, most notably via DP-SGD [4]. Subsequent work has steadily tightened the analysis of private learning algorithms [21]–[23], with these advances made widely accessible through programming interfaces such as Opacus [24]. In contrast to these probabilistic refinements, we tighten the analysis of DP-SGD using formal methods.

Combining Formal Methods and Privacy Formally proving that a machine learning model meets a given specification was largely popularized in response to adversarial robustness [25], [26]. However, it has recently been extended to other important notions of trustworthiness [27]–[30] including differential privacy [13]. The current use of formal methods to tighten privacy analysis of machine learning models is restricted to private prediction in classification settings. By adopting and extending the general smooth sensitivity-based framework of [13] we push certification-based privacy analysis to private learners.

## III. PRELIMINARIES

Notation. We denote a general machine learning model as a function f parameterized by $\theta \in \mathbb { R } ^ { p }$ which maps from an input space to an output space $f ^ { \theta } : \mathcal { X }  \mathcal { Y }$ (often with $\mathcal { X } = \mathbb { R } ^ { m }$ and $\mathcal { V } = \mathbb { R } ^ { n } )$ ). We consider supervised learning in the regression setting with a labeled dataset $D = \{ x ^ { ( i ) } , \bar { y ^ { ( i ) } } \} _ { i = 1 } ^ { N }$ and further assume that Y is a continuous metric space with distance metric $d i s t ( \cdot , \cdot )$

## A. Privacy

Differential privacy (DP) is a widely employed formal privacy guarantee, which is deeply rooted in statistical databases [1] and has been adopted as the standard for privacy guarantees in machine learning [2], [4]. DP ensures that the output of an algorithm applied to two adjacent sets of data is statistically indistinguishable. We formally define DP as follows:

Definition III.1 $( ( \epsilon , \delta ) { \tt - D P }$ [2]). A randomized mechanism M is $( \epsilon , \delta )$ -differentially private if, for all pairs of adjacent datasets $D , D ^ { \prime } \in \mathcal { D }$ and any subset $S \subseteq \operatorname { s u p p } ( { \mathcal { M } } )$

$$
\mathbb { P } \left( \mathcal { M } ( D ) \in S \right) \le e ^ { \epsilon } \mathbb { P } \left( M ( D ^ { \prime } ) \in S \right) + \delta .
$$

(ϵ, δ)-DP is also known as approximate DP, with pure or $\mathbf { \Lambda } _ { \epsilon - \mathrm { D P } }$ arising when $\delta = 0$ . Within machine learning, there are two distinct approaches to privacy in private prediction and private learning.

1) Private Learning: A private learning algorithm is one that returns a learned model that itself is differentially private. In machine learning, this is achieved by applying Definition III.1 taking $D$ to be the training dataset, M to be the learning algorithm, and $\operatorname { S u p p } ( { \mathcal { M } } )$ to be the space of all possible parameters. To ensure the final parameters, θ, satisfy the definition, Differentially Private Stochastic Gradient Descent (DP-SGD) [4] modifies standard SGD through two core operations at each training step: firstly, it uses a gradient clipping operation to bound the contribution of each training point and adds noise proportional to the clipping parameter to ensure the training algorithms output distribution satisfies Definition III.1 [4]. Alternative approaches to private learning include DP-ERM [6], where output and objective perturbation at training time are shown to satisfy DP, and PATE [18], which partitions data into disjoint subsets (thus reducing the model’s sensitivity through sharding) and employs a teacher-student framework, where the student model distills the teacher’s noisy aggregate vote into a publicly trained model, converting private predictions into a private learning procedure. Moreover, substantial advancements have built on top of DP-SGD to enhance privacy analysis including the use of advanced composition theorems [31] and the moments accountant [4], [22], among others [32], [33].

2) Private Prediction: Private learning is not without drawbacks; chiefly, private learning can cause substantial utility degradation which often requires practitioners to retrain their algorithm at different privacy levels potentially compromising their privacy analysis and incurring considerable computational cost [34], [35]. Private prediction represents an alternative to achieving differential privacy in machine learning. The central idea is to first learn a model $f ^ { \theta } ( x )$ without any privacy protection and to subsequently add noise η to the results of each prediction such that Definition III.1 is satisfied [8].

To ensure that the released prediction satisfies $( \epsilon , \delta ) \ / { - \mathrm { D P } } ,$ one typically calibrates the added noise, $\eta ,$ to the prediction’s sensitivity. The most common and general notion of sensitivity is the global $\ell _ { p }$ sensitivity with respect to the cardinality of the symmetric difference $\dot { d } ( D , D ^ { \prime } ) \stackrel { - } { = } | D \triangle D ^ { \prime } |$

Definition III.2 (Global $\ell _ { p }$ Sensitivity [2]). A function $f :$ $\mathcal { D } \to \mathbb { R } ^ { n }$ has global $\ell _ { p }$ sensitivity

$$
\Delta _ { p } f = \operatorname* { m a x } _ { \substack { D , D ^ { \prime } \in \mathcal { D } , d ( D , D ^ { \prime } ) = 1 } } \Vert f ( D ) - f ( D ^ { \prime } ) \Vert _ { p } .
$$

Throughout this work we shall refer to the $\ell _ { 1 }$ sensitivity as $\Delta f$ . With access to the global sensitivity of a prediction, both pure and approximate DP can be satisfied by the following well-known mechanisms:

Definition III.3 (Laplace $( \epsilon , 0 ) \mathopen { } \mathclose \bgroup \left. \mathrm { D P } \right.$ Mechanism [2]). For a function $f : \mathcal { D }  \mathbb { R } ^ { n }$ , the mechanism $\mathcal { M } ( D ) = f ( D ) + \eta$ where $\eta \sim \mathrm { L a p } ( \Delta f / \epsilon )$ satisfies pure $( \epsilon , 0 )$ -differential privacy.

Definition III.4 (Gaussian Mechanism [2]). For a function $f : \mathcal { D }  \mathbb { R } ^ { n }$ , the mechanism $\mathcal { M } ( D ) = f ( D ) + \eta$ where $\eta \sim \mathcal { N } ( 0 , \sigma ^ { 2 } \mathbb { I } )$ and $\begin{array} { r l } { \sigma \ge \frac { \Delta _ { 2 } f } { \epsilon } \sqrt { 2 \ln ( 1 . 2 5 / \delta ) } } \end{array}$ satisfies $( \epsilon , \delta )$ differential privacy for $\delta \in ( \bar { 0 } , 1 )$ .

Any mechanism employing the global sensitivity $( \mathrm { e . g . }$ Definitions III.3 and III.4) is data-independent by nature of $\Delta f$ being defined over all possible neighboring datasets D and $D ^ { \prime }$ In general, private prediction calibrated to the global sensitivity incurs utility cost exceeding that of private learning [10]. Though one must be careful to ensure releases remain private, a tighter privacy analysis can be achieved by considering the local sensitivity:

Definition III.5 (Local $\ell _ { 1 }$ Sensitivity [2]). The local $\ell _ { 1 }$ sensitivity of a function $f : \mathcal { D }  \mathbb { R } ^ { n }$ at point $x \in \mathcal { D }$ is defined as $\begin{array} { r } { \mathrm { L S } ( f , x ) = \operatorname* { m a x } _ { y : d ( x , y ) = 1 } \| f ( x ) - f ( y ) \| _ { 1 } } \end{array}$

As the local sensitivity itself is data-dependent, one cannot directly calibrate noise to the local sensitivity. However, to rectify this, Nissim et. al. [12], proposed a smooth upper bound on this sensitivity with respect to datasets a distance k apart termed the smooth sensitivity:

Definition III.6 (β-Smooth Sensitivity [12]). The $\beta \mathrm { . }$ -smooth sensitivity of a function $f : \mathcal { D }  \mathbb { R } ^ { n }$ at a point $x \in \mathcal { D }$ can be defined as $\begin{array} { r } { \mathbf { S } \mathbf { S } ^ { \beta } ( f , x ) = \operatorname* { m a x } _ { k \in \mathbb { N } ^ { + } } e ^ { - \beta k } \hat { A ^ { k } } ( f , x ) } \end{array}$ where $\begin{array} { r } { A ^ { k } ( f , x ) : = \operatorname* { m a x } _ { y : d ( x , y ) \leq k } \operatorname { L S } ( f , y ) } \end{array}$

The major advancement of the smooth sensitivity is the ability to derive associated mechanisms (similar to Definitions III.3 and III.4) that obtain pure- and approximate-DP. For example, pure-DP can be achieved by adding Cauchy noise calibrated to the smooth sensitivity:

Definition III.7 (Cauchy Mechanism [12]). For a function $f : \mathcal { D }  \mathbb { R } ^ { n }$ , the mechanism $\mathcal { M } ( D ) = f ( D ) + \eta$ where $\eta \sim \mathrm { C a u c h y } ( 6 \mathrm { S S } ^ { \beta } ( f , x ) / \epsilon )$ and $\forall \beta \ \leq \ ( \epsilon / 6 )$ satisfies $( \epsilon , 0 ) \cdot$ differential privacy.

Similarly, to satisfy approximate-DP, one can use Laplace noise calibrated to the smooth sensitivity:

Definition III.8 (Laplace $( \epsilon , \delta ) \ / – \mathrm { D P }$ Mechanism [12]). For a function $f : \mathcal { D }  \mathbb { R } ^ { n }$ , the mechanism $\mathcal { M } ( D ) = f ( D ) + \eta ,$ where $\eta \sim \mathrm { L a p } ( 2 \mathrm { S S } ^ { \beta } ( f , x ) / \epsilon )$ and $\forall \beta \le \epsilon / ( 2 \ln ( 2 / \delta ) )$ with $\delta \in ( 0 , 1 )$ satisfies $( \epsilon , \delta )$ -differential privacy.

While Nissim et al. [12] provide a systematic treatment of admissible noise distributions that yield privacy guarantees based on smooth sensitivity, we focus our attention only on the mechanisms outlined in this section, leaving analysis of our techniques in combination with other mechanisms to future works.

## B. Valid Parameter Space Bounds

Leveraging formal methods to compute the above smooth sensitivity mechanisms, Sosnin et al. [11], [13] cast the local sensitivity of a learning algorithm as a reachability specification that can be verified using abstract interpretation. Their framework computes the parameter envelopes that bound the effect of data modifications during training, thus bypassing the need to optimize over the intractable space of all possible datasets. Let $\mathcal { M } ( f , \theta _ { \mathrm { i n i t } } , D )$ denote a gradient-based algorithm trained on dataset $D$ that returns the final parameters of a model $f ,$ starting from a fixed initialization $\theta _ { \mathrm { i n i t } }$ . The parameter envelopes computed by Sosnin et al. [11], [13] are termed valid parameter space bounds:

Definition III.9 (Valid Parameter-Space Bound [11]). For a nominal dataset D and a distance threshold k, a parameter envelope $T _ { k } = [ \theta _ { L } ^ { k } , \theta _ { U } ^ { k } ]$ is a valid parameter-space bound if it contains all possible model parameters that result from training on any dataset $\tilde { D }$ formed by up to k additions, removals or substitutions from D:

$$
\mathcal { M } ( f , \theta _ { \mathrm { i n i t } } , \tilde { D } ) = \tilde { \theta } \in T _ { k } , \quad \forall \tilde { D } \ \mathrm { s . t . } \ d ( D , \tilde { D } ) \leq k .
$$

To compute these intervals using abstract interpretation, Sosnin et al., present Abstract Gradient Training $( A G T )$ which leverages bound propagation to construct the envelopes. The algorithm considers batchdx of size b and a per-element gradient clipping threshold $\gamma .$ We refer to quantities pertaining to training on the original, unmodified dataset D as nominal (i.e., $k = 0 )$ . The updates are computed element-wise: AGT aggregates the $b - k$ nominal points’ gradients and accounts for the $k$ differing points by assuming they produce gradients equal to $\gamma$ in the most adversarial direction. In this way, it can isolate all reachable parameters to produce final, axis-aligned parameter envelopes $T _ { k } = [ \theta _ { L } ^ { k } , \theta _ { U } ^ { k } ]$ . Importantly, this allows the propagation of valid bounds from parameter to output space, acting as a proxy for computing local sensitivity in prediction space.

Shortcomings of Prior Approaches: The AGT algorithm comes with some downsides. In particular, the bounds provided only apply to private prediction in classification settings. Beyond this restrictive setting, the utility of formal-methods-based privacy guarantees remains an open question. Moreover, we highlight that the privacy analysis in [13] does not explore the compatibility of AGT and amplification theorems, thus limiting the scale of problems that can be studied with their mechanism. In what follows, we extend the AGT algorithm to regression, discuss the use of amplification and AGT, and provide the first algorithms for using bound propagation for private learning.

## IV. METHODOLOGY

In this section, we will first introduce the theoretical underpinnings of our novel method, AGT-R, and analyze when is has superiority over global sensitivity.We will then introduce a novel private learning method, Abstract Gradient Sampling (AGS), which we will similarly prove can be superior to existing approaches.

## A. AGT-R

We begin by recalling the mechanism presented in Sosnin et al. [11], [13] which first uses AGT to compute valid parameterspace bounds via bound propagation for distances (adjacency quantifiers) $k \in \mathcal { K } \subset \mathbb { N } ^ { + }$ . The AGT mechanism subsequently employs parameter envelopes $\{ T _ { k } \} _ { k \in \mathcal { K } }$ associated with the adjacency quantifier specified by the subscript. Using this notation, we restate their proposed smooth sensitivity bound:

Theorem IV.1 (Sosnin et al. [11], [13]). Let $\mathcal { M } ( f , \theta _ { \mathrm { i n i t } } , D ) = \theta$ $T _ { k } = [ \theta _ { L } ^ { k } , \theta _ { U } ^ { k } ]$ satisfying Def. III.9, and let $f ^ { \theta } ( x )$ denote the prediction of a binary classifier and $\mathbb { 1 } ( \cdot )$ represent the indicator function. Additionally let $f ^ { T _ { k } } ( x )$ be 1 when after propagating the envelope through the model $f _ { i }$ , the lower bound of the output envelope satisfies $y _ { L } ^ { k } \ge 0 . 5$ , and 0 otherwise. Then the following is an upper-bound on the β-smooth sensitivity:

$$
\overline { { \mathrm { S S } _ { c } ^ { \beta } } } ( f _ { x } , D ) = \operatorname* { m a x } _ { k \in [ N ] } \Big [ \mathbb { 1 } \left( f ^ { T _ { k } } ( x ) \neq f ^ { \mathcal { M } ( f , \theta _ { \mathrm { i n i t } } , D ) } ( x ) \right) e ^ { - \beta k } \Big ] .
$$

Albeit sound, a critical observation about this theorem is that it only works in classification settings, where $\forall y , y ^ { \prime } \in \mathcal { V } , y \neq$ $y ^ { \prime } \colon d ( y , y ^ { \prime } ) = 1$ , and fails otherwise, i.e. in regression settings. We notice that the constraint exists because the indicator function acts as a sound-bounding function for the output sensitivity at distance $k \ ( A ^ { k }$ in Definition III.6). To extend this bound to cases where the output space $\mathcal { V }$ is continuous and unbounded $( \boldsymbol { \mathrm { e } } . \boldsymbol { \mathrm { g } } . , \mathcal { V } = \mathbb { R } )$ , we proceed to systematically generalize the sound-bounding function used.

We first define what it means for a function $\mathfrak { d } ( x , k )$ to be sound-bounding for local sensitivity $A ^ { k } ( f , x )$ by introducing the properties it needs to satisfy:

$$
\begin{array} { r l } { ( i ) } & { { } A ^ { k } ( f , x ) \leq \mathfrak { d } ( x , k ) , \forall x \in \mathcal { X } , k \in \mathcal { K } } \\ { ( i i ) } & { { } \mathfrak { d } ( x , k ) \leq \Delta _ { p } f , \forall p \in \mathbb { N } , x \in \mathcal { X } , k \in \mathcal { K } . } \end{array}
$$

We note that the first property enforces soundness, while the second avoids triviality (i.e., ensures it is lower than the global sensitivity). For $\mathcal { V } = \mathbb { R }$ one can construct such a soundbounding function using only an arbitrary, finite number of envelopes and the mechanism proposed by AGT (further discussed in $\ S \ \mathrm { I V - A } 2 )$ . For regression, we begin by assuming that we have computed valid parameter-space bounds for all values of $k \in [ N ]$ , and we construct the sound bounding function. Firstly, let $B _ { k } ( D )$ be the set of all datasets within radius k of $D$ and allow $\begin{array} { r } { y _ { L } ^ { k } : = \operatorname* { m i n } _ { \tilde { D } \in B _ { k } ( D ) } f ^ { \tilde { \theta } } ( x ) \leq \operatorname* { m i n } _ { \theta ^ { \prime } \in T _ { k } } f ^ { \theta ^ { \prime } } ( x ) } \end{array}$ and let $y _ { U } ^ { k } : = \operatorname* { m a x } _ { { \tilde { D } } \in B _ { k } ( D ) } f ^ { { \tilde { \theta } } } ( x ) \leq \operatorname* { m a x } _ { \theta ^ { \prime } \in T _ { k } } f ^ { \theta ^ { \prime } } ( x )$ . We define the following sound bounding function for regression: $\mathfrak { d } ( x , k ) \ : = \ \operatorname* { m a x } ( y _ { U } ^ { \overline { { k } } } - y _ { L } ^ { k + 1 } , y _ { U } ^ { k + 1 } - y _ { L } ^ { k } )$ . For this $\mathfrak { d } ( x , k )$ property (i) is satisfied by observing:

$$
\begin{array} { r l } {  { A ^ { ( k ) } ( D ) : = \operatorname* { m a x } _ { \tilde { D } \in B _ { k } ( D ) } L S ( f , \tilde { D } ) } } \\ & { \le \operatorname* { m a x } _ { \tilde { D } \in B _ { k } ( D ) , D ^ { \bullet } \in B _ { k + 1 } ( D ) } \| f ^ { \theta ^ { \star } } ( x ) - f ^ { \tilde { \theta } } ( x ) \| _ { 1 } } \\ & { \le \operatorname* { m a x } _ { \tilde { y } \in [ y _ { L } ^ { k } , y _ { U } ^ { k } ] , y ^ { \star } \in [ y _ { L } ^ { k + 1 } , y _ { U } ^ { k + 1 } ] } \| y ^ { \star } - \tilde { y } \| _ { 1 } } \\ & { = \sum _ { i } \operatorname* { m a x } ( y _ { U , i } ^ { k + 1 } - y _ { L , i } ^ { k } , y _ { U , i } ^ { k } - y _ { L , i } ^ { k + 1 } ) . } \end{array}
$$

The first inequality follows because $B _ { 1 } ( \tilde { D } ) \subseteq B _ { k + 1 } ( D )$ and the second by the soundness of the certified intervals. Property (ii) is satisfied by propagating $k = N$ to get the universe of reachable parameters for which:

$$
L S _ { N } ( f , x ) \leq \operatorname* { m a x } _ { \theta ^ { \prime } \in T _ { N } } \| f ^ { \theta } ( x ) - f ^ { \theta ^ { \prime } } ( x ) \| _ { 1 } \leq G S ( f )
$$

Once a sound-bounding function is established a β-smooth upper-bound on the smooth sensitivity is given by using the construction of Nissim et al. [12]. Formally:

Theorem IV.2. Given a model $f ,$ data point $x \in \mathcal { X }$ and $k \in \mathbb N$ let $\mathfrak { d } ( x , k )$ be an AGT-constructed (as above) sound-bounding function of the local sensitivity $A ^ { k } ( f , x )$ . Then the following is a β-smooth upper bound on the β-smooth sensitivity of [12]:

$$
\overline { { \mathrm { S S } ^ { \beta } } } ( f _ { x } , D ) = \operatorname* { m a x } _ { k \in [ N ] } \left[ \mathfrak { d } ( x , k ) e ^ { - \beta k } \right]
$$

Proof. The first property of the smooth bound proposed by [12] follows directly by applying property (i) of a soundbounding function: $\mathfrak { d } ( x , k ) \geq A ^ { k } ( f , x )$ . The second property, namely β-smoothness, is a direct consequence of the from the exponential discount factor $e ^ { - \beta k }$ and the fact that ${ \mathfrak { d } } ( D , k ) \leq$ $\mathfrak { d } ( \tilde { D } , k + 1 )$ for $d ( D , \tilde { D } ) = 1$ which is true both when using exact valid parameter-space bounds and any sound orthotope overapproximation. □

One can observe that by letting d be the indicator function, this recovers exactly Theorem IV.1 of Sosnin et al. [13]:

Proposition IV.1. Let $\begin{array} { r l r } { \mathcal { V } } & { { } = } & { \{ 0 , 1 \} } \end{array}$ and instantiate the sound bounding function with $\begin{array} { r l r l } { \mathfrak { d } ( x , k ) } & { { } } & { = } & { { } } \end{array}$ $\mathbb { 1 } \left( f ^ { T _ { k } } ( x ) \neq f ^ { \mathcal { M } ( f , \theta _ { \mathrm { i n i t } } , \boldsymbol { D } ) } ( x ) \right)$ . Then, Theorem IV.1 can be recovered and holds: $\overline { { \mathrm { S S } ^ { \beta } } } = \overline { { \mathrm { S S } _ { c } ^ { \beta } } }$

Thus, sound bounding functions allow us to strictly general ize the analysis provided in prior works by following through with the smooth sensitivity analysis.Using Theorem IV.2 we can achieve (ϵ, 0) and $( \epsilon , \delta ) \ / – \mathrm { D P }$ guarantees are given through mechanisms in Definition III.7 and III.8, respectively.

Lastly, our formulation and privatization mechanism provide a finite bound on smooth sensitivity and noise for unbounded output domains.

Remark IV.1 $( \overline { { \mathrm { S S } ^ { \beta } } }$ boundedness for unbounded Y). Theorem IV.2 can provide finite sensitivity bounds even for algorithms with a priori infinite global sensitivity $( \mathrm { i . e . , } d ( \mathrm { s u p } ) \mathrm { - }$ inf $\mathcal { V } ) = \infty )$ when $\begin{array} { r } { \mathfrak { d } ( x , N ) = \operatorname* { m a x } _ { \theta ^ { \prime } \in T _ { N } } \| f ^ { \theta } ( x ) - f \underline { { \theta ^ { \prime } } } ( { x } ) \| _ { 1 } \le } \end{array}$ ∞ in turn yields a finite smooth sensitivity bound $\mathrm { S S } ^ { \beta }$

1) Amplification of Generalized AGT Mechanisms: Before turning to practical computations of generalized sound bounding functions, we remark on the use of amplification with AGT-based mechanisms.

Let $\ b { \cal D } ~ = ~ ( z _ { 1 } , \dots , z _ { N } ) ~ \in ~ \mathcal { X } ^ { N }$ be a fixed-size dataset equipped with substitution adjacency, and let

$$
I \sim \operatorname { U n i f } \big ( \{ I \subseteq [ N ] : | I | = m \} \big ) , \qquad q : = \frac { m } { N } ,
$$

where $m \leq N$ is public. Denote the resulting size-m subsample by $D _ { I }$ . For a fixed public query $x ,$ let $A _ { m } ( \cdot ; x )$ denote the AGT mechanism of Definition III.7, III.4 instantiated on the randomized dataset of size $m .$ , then we have the following privacy amplification:

Lemma IV.1 (Amplification of AGT mechanisms by sampling without replacement [36]). Suppose that $A _ { m }$ is $( \epsilon _ { \mathrm { b } } , \delta _ { \mathrm { b } } )$ -DP under substitution adjacency on ${ \mathcal { X } } ^ { m }$ . Then the subsampled mechanism $\begin{array} { r l r } { \hat { \mathcal A } _ { N , m } ^ { \mathrm { W O R } } ( D ; x ) } & { { } : = } & { \mathcal A _ { m } ( D _ { I } ; x ) } \end{array}$ is $( \log ( 1 + q \left( e ^ { \epsilon _ { \mathrm { b } } } - 1 \right) ) , q \delta _ { \mathrm { b } } ) \ – \mathbf { D P }$ under substitution adjacency on $\dot { \mathcal { X } } ^ { N }$

Equivalently, to obtain a target outer guarantee $( \epsilon , \delta )$ , it suffices to instantiate the base AGT-R mechanism with

$$
\epsilon _ { \mathrm { b } } = \log \left( 1 + \frac { e ^ { \epsilon } - 1 } { q } \right) , \qquad \delta _ { \mathrm { b } } = \frac \delta q ,\tag{1}
$$

where the latter equality assumes $\delta \leq q .$ . In the pure-DP case, $\delta = \delta _ { \mathrm { b } } = 0$

Lemma IV.1 which is a direct application of [36, Theorem 9] allows AGT-R to operate on a single persistent subsample of m records while providing a privacy guarantee with respect to the original dataset of size N.

Algorithm 1 AGT-R (Pure ϵ-DP)   
1: Input: Model $f ,$ initialization $\theta _ { \mathrm { i n i t } }$ , train dataset $D _ { t } \left( \left| D _ { t } \right| = N \right) ,$ public   
test input x, AGT procedure ${ \mathcal { M } } _ { \mathrm { A G T } } ,$ and privacy constants: $\epsilon , \delta , \beta ,$   
global sensitivity $\Delta { \dot { f } } ,$ and the set of distance radii: ${ \mathrm { \widehat { \kappa } } } \subset \mathbb { N } ^ { + } .$   
2: Output: An $( \epsilon , \mathsf { 0 } ) { \ - } \dot { \mathrm { D P } }$ prediction on x.   
3: $\theta , \ [ \theta _ { L } ^ { k } , \theta _ { U } ^ { k } ] _ { k \in \mathcal { K } }  \forall k _ { i } \in \mathcal { K } , \ M _ { \mathrm { A G T } } ( f , \theta _ { \mathrm { i n i t } } , D _ { t } , k _ { i } )$   
4: $\begin{array} { r } { \forall i \in [ N ] , \quad \overline { { A } } [ i ] \gets \operatorname* { m i n } \left( \Delta f , \ \operatorname* { m a x } _ { \theta ^ { \prime } \in [ \theta _ { L } ^ { N } , \theta _ { U } ^ { N } ] } \| f ^ { \theta ^ { \prime } } ( x _ { i } ) - f ^ { \theta } ( x _ { i } ) \| _ { 1 } \right) } \end{array}$   
5: for k in sort(K) do   
6: $\begin{array} { r } { \overline { { A ^ { k } } } ( f , x )  \operatorname* { m a x } _ { \theta ^ { \prime } \in [ \theta _ { I , } ^ { k } , \theta _ { I J } ^ { k } ] } \| f ^ { \theta ^ { \prime } } ( x _ { i } ) - f ^ { \theta } ( x _ { i } ) \| _ { 1 } } \end{array}$   
7: for all $j \in [ N ] s . t . , j < k$ do   
8: A[i] ← min $\left( \overline { { A } } [ i ] , \quad \overline { { A ^ { k } } } ( f , x ) \right)$   
9: end for   
10: end for   
11: $\forall i \in [ N ] , \quad \underline { { B } } [ i ]  \underline { { \exp } } ( - \beta i )$   
12: $\forall i \in \dot { [ } N \dot { ] } , \quad \overline { { \mathrm { S } } } [ \dot { i } ] ^ { \cdot } \gets \overline { { A } } [ \dot { i } ] \cdot B [ i ]$   
13: $\overline { { \mathrm { S S } ^ { \beta } } } \dot {  }$ max S[k]   
14: $\begin{array} { r } { \hat { y }  f ^ { \theta } ( x ) + \eta , \quad \eta \sim \mathrm { C a u c h y } \Big ( \frac { 6 \overline { { \mathrm { S S } ^ { \beta } } } } { \epsilon } \Big ) } \end{array}$   
15: return yˆ

2) Tractable Computation via AGT: Algorithm 1 first trains the model via AGT, yielding nominal parameters θ and parameter envelopes $[ \theta _ { L } ^ { k } , \theta _ { U } ^ { k } ] _ { k \in \mathcal { K } }$ , where k denotes the cardinality of the symmetric difference between neighbouring datasets that parameterizes AGT.

We observe that in Theorem IV.2, $\kappa : = [ N ]$ implying that one must compute the maximum over all possible symmetric difference radii (with N-many AGT runs, for example). In fact, we show that arbitrary $\kappa \subset \mathbb { N }$ bounds Theorem IV.2 as long as $N \in \kappa$ . Consider a set K that has at least one $ { \mathrm { \tilde { \Delta } g a p \mathrm { \Delta } } } ^ { \smash { \prime } \mathclose { \Delta } }$ at some value k, i.e., $k \in [ N ] \land k \notin \mathcal { K }$ . A sound upper-bound in Theorem IV.2 must somehow bound the local sensitivity at radius k to achieve a valid smooth-sensitivity bound. We first observe that $\forall k ^ { \prime } > k , A ^ { k ^ { \prime } } ( f , x ) > A ^ { k } ( \dot { f , } x )$ . Thus, if we observe a value $A ^ { k ^ { \prime } } ( f , x )$ it is sound for every $k < k ^ { \prime }$ This naturally gives rise to a sound “back-filling” procedure: $\forall k ^ { \prime } \notin { \cal K }$ define $k ^ { \star } : = \operatorname* { m i n } _ { k \in \mathcal { K } } k > k ^ { \prime }$ and then let $\overline { { A ^ { k ^ { \prime } } } } =$ $\overline { { A ^ { k ^ { \star } } } }$ . Finally, ensuring that $N \in { \mathcal { K } }$ allows us to soundly fill all gaps $k ^ { \prime } \notin \mathcal { K }$ . By the same reasoning in Theorem IV.2, taking the maximum over all k after applying the back-filling procedure obtains a upper-bound on smooth sensitivity (line 13 of Algorithm 1).

The choice of $\kappa$ naturally induces a trade-off: small cardinality sets reduce the number of training runs and forward passes (which grow proportionally with k) but may yield looser bounds. Denser sets of values in K progressively tighten the smooth sensitivity at a greater computational cost. To conclude this analysis, we note that Algorithm 1 covers only the pure $( \epsilon , 0 ) \mathopen { } \mathclose \bgroup \left. \mathrm { D P } \right.$ case, but few changes are required to achieve approximate $( \epsilon , \delta ) \ / { - } \mathrm { D P } \colon \eta _ { i }$ on line 14 must be sampled from $\mathrm { L a p } ( { \textstyle { \frac { 2 \overline { { S } } S ^ { \beta } } } { \epsilon } } )$ with $\beta \leq \frac { \epsilon } { 2 \ln ( 2 / \delta ) }$

3) Improving Utility of Private Prediction: In this section, we establish the conditions under which AGT-R can be (probabilistically) guaranteed to outperform global sensitivity based privatization in regression settings. We analyze equivalent approaches in pure- and approximate-DP, providing closed-form privacy-utility trade-offs. Proofs of theorems are deferred to Appendix B-A and B-B.

Pure $( \epsilon , 0 ) \ / – D P \mathrm { : }$ The noise distributions that guarantee $( \epsilon , 0 ) \mathopen { } \mathclose \bgroup \left. \mathrm { D P } \right.$ are $\mathrm { L a p } ( 0 ; \Delta f / \epsilon )$ and Cauchy $( 0 ; 6 \mathbf { S S } ^ { \beta } ( f , x ) / \epsilon )$ for global and smooth sensitivity, respectively. We analyze and prove the critical conditions that need to be satisfied for smooth sensitivity privatization to outperform global sensitivity privatization at a fixed privacy level $\epsilon .$ Importantly, we make this systematic by defining a flexible cost fraction $c ,$ which represents the exact utility improvement multiplier our method offers, provided the condition below is satisfied.

Theorem IV.3. Consider a regressor $f$ trained on dataset D performing private prediction at point x and a real constant $c \in ( 1 , \infty )$ . At a failure probability $\alpha$ (or, equivalently, at confidence level $1 - \alpha )$ , Cauchy noise with parameter $\gamma =$ $6 5 { \mathrm { S } } ^ { \beta } ( f , x ) / \epsilon$ and $\beta < \epsilon / 6$ provides c-tighter error bounds than Laplace noise with parameter $b = \Delta f / \epsilon$ whenever

$$
\mathsf { S S } ^ { \beta } ( f , x ) < \frac { \Delta f } { 6 c } \left[ \frac { \ln \left( \frac { 1 } { \alpha } \right) } { \tan \left( \frac { \pi ( 1 - \alpha ) } { 2 } \right) } \right] .
$$

Analysis of Utility Trade-offs: The condition in Thm. IV.3 is a probabilistic certificate of utility improvement, and we make two observations about it. Firstly, since an exact expression for $\mathsf { S } \mathsf { S } ^ { \beta }$ is not available, the left-hand side is in practice replaced by the upper bound $\overline { { \mathsf { S } } } \overline { { \mathsf { S } } } ^ { \beta }$ computed by AGT, which makes the condition sufficient but conservative. Secondly, as $\alpha  0$ we have tan $( \pi ( 1 - \alpha ) / 2 ) \sim 2 / ( \pi \alpha )$ , so the bracketed term decays as ${ \scriptstyle \frac { \pi } { 2 } } \alpha \ln ( 1 / \alpha ) \ \to \ 0$ : the critical sensitivity vanishes faster than the $\ln ( 1 / \alpha )$ grows and thus no deterministic guarantee is attainable. The condition therefore fails when the AGT bounds are unacceptably large, when the enforced improvement factor $c$ is unrealistic, or when the desired guarantee approaches determinism. Although we cannot control the smooth sensitivity directly, it is possible to simulate over α and c and use AGT-R only when the ratio between $\overline { { \mathrm { S S } ^ { \beta } } }$ and $\Delta f$ falls below the resulting critical value. We present this simulation in the left column of Fig. 1 and analyze it in §VI.

Approximate $( \epsilon , \delta ) \ / – D P \mathrm { : }$ We now turn to the Gaussian mechanism, which, although it offers lighter tails via (ϵ, δ)-DP, requires $\delta \ \ll \ 1 / N$ , making its noise scale prohibitive in limited-sample regimes. Similarly to before, we derive and prove a sufficient condition on the smooth sensitivity under which our smooth sensitivity-based Laplace mechanism (approximate $( \epsilon , \delta ) \mathrm { - D P ) }$ yields tighter error bounds than the Gaussian mechanism.

Theorem IV.4. Consider the regressor $f ,$ dataset $D ,$ point $x ,$ confidence level $1 - \alpha$ , and real constant $c \in \mathsf { \Gamma } ( 1 , \infty )$ Additionally, consider a fixed failure probability $\delta \in ( 0 , 1 )$ of approximate-DP. Laplace noise with scale parameter $b ~ = ~ 2 { \bf S } { \bf S } ^ { \beta } ( f , x ) / \epsilon$ and $\beta ~ < ~ ( \epsilon / ( 2 \ln ( 2 / \delta ) ) )$ provides $c -$ tighter error bounds than Gaussian noise with parameter $\begin{array} { r } { \sigma = \frac { \Delta _ { 2 } f \sqrt { 2 \cdot \ln ( 1 . 2 5 / \delta ) } } { \epsilon } } \end{array}$ whenever:

$$
\mathbf { S } \mathbf { S } ^ { \beta } ( f , x ) < \frac { \Delta _ { 2 } f } { 2 c } \left[ \frac { \sqrt { 2 \ln \left( \frac { 1 . 2 5 } { \delta } \right) } \cdot \Phi ^ { - 1 } \left( 1 - \frac { \alpha } { 2 } \right) } { \ln \left( \frac { 1 } { \alpha } \right) } \right] ,
$$

where $\Phi ( \cdot )$ denotes the standard normal CDF and $\delta$ is the allowed privacy leakage in approximate DP.

Analysis of Utility Trade-offs: The condition in Thm. IV.4 is again a probabilistic certificate, $\Delta _ { 2 } f , c ,$ and $\overline { { \mathsf { S } \mathsf { S } ^ { \beta } } }$ playing the same role as before. The confidence level, however, behaves differently. Firstly, both quantiles grow as $\alpha \  \ 0 .$ , since $\Phi ^ { - 1 } ( 1 - \stackrel { \cdot } { \alpha } / 2 ) \sim \stackrel { \cdot } { \sqrt { 2 \ln ( 1 / \alpha ) } }$ , so the bracketed factor decays only as $\sqrt { 2 / \ln ( 1 / \alpha ) }$ rather than collapsing polynomially. This reflects the tails involved, as Laplace and Gaussian noise are both light-tailed, whereas Cauchy noise is not. Secondly, the factor $\sqrt { 2 \ln ( 1 . 2 5 / \delta ) }$ acts in our favour, since a smaller leakage $\delta$ forces the Gaussian mechanism to inject more noise. In summary, this condition is markedly more permissive, and remains satisfiable at confidence levels for which Thm. IV.3 already fails. We report the critical ratio $\mathrm { S S } ^ { \beta } / \Delta _ { 2 } f$ for various values of α and c at fixed δ in the right column of Figure 1.

## V. ABSTRACT GRADIENT SAMPLING

The previous section established AGT-R as a mechanism for private prediction: given a trained model, we add noise calibrated to smooth sensitivity in the output space to privatize individual predictions. We now turn to what is arguably the more fundamental problem in the ML privacy literature: private learning—i.e., releasing a model whose parameters themselves satisfy differential privacy. The gold standard for private learning is widely considered to be DP-SGD [4], with improvements such as Rényi DP [21] and the moments accountant [22], [37]. We show that the smooth sensitivity framework developed for AGT-R extends naturally to the parameter space, yielding a novel approach to private learning. The central observation is that where the learning algorithm is viewed as a multi-dimensional regression over parameters, $\theta ,$ the valid parameter space bounds yield exactly the local sensitivities used to calibrate noise in AGT-R. In this section, we begin by directly applying AGT-R to the output of the learning algorithm $( \ S \mathrm { V }  – \mathbf { A } ) ;$ ; we then systematically generalize this approach into an algorithm we call Abstract Gradient Sampling (§V-B).

## A. Post-Training Privatization

Where the learning algorithm returns a real-valued vector, $\theta = \mathcal { M } ( f , \theta _ { \mathrm { i n i t } } , D )$ , the $\ell _ { 1 }$ sensitivity of that vector is given by the valid parameter space bound with $k = 1 \ \mathrm { i } . \mathrm { e } . , T _ { 1 }$ i.e., Definition III.5. Moreover, the parameter envelopes produced by AGT for any $k \geq 1$ maintain the guarantee for that for any neighboring dataset $\tilde { D }$ with $| \tilde { D } \triangle D | \le k$ , the parameters obtained by training on $\tilde { D }$ reside within the bounds. Formally, we have that if $\tilde { \theta } = \mathcal { M } ( f , \theta _ { \mathrm { i n i t } } , \tilde { D } )$ , then by construction, we have that $\tilde { \theta } \in [ \theta _ { L } ^ { k } , \theta _ { U } ^ { k } ]$ . As in the (private prediction) regression case, the local sensitivity at $\tilde { D }$ compares <sup>˜</sup>θ with the parameters of a neighbor of $\tilde { D } .$ , which lie in $[ \theta _ { L } ^ { k + 1 } , \theta _ { U } ^ { k + 1 } ]$ , so the soundbounding function construction of §IV-A gives:

![](images/7280c79117528f2b070e97ac18f19009af0510dc42fc4ba43c256bb2cea0050c.jpg)  
Figure 1: Feasibility of smooth-sensitivity privatization. Each curve gives the largest sensitivity ratio $\rho = \mathrm { S S } ^ { \beta } / \Delta f$ for which our smooth-sensitivity mechanism attains a c-times tighter error bound than the global-sensitivity baseline at confidence $1 - \alpha ;$ the region below each curve is feasible. (a) Pure-DP (Thm. IV.3), Cauchy vs. Laplace; black dots mark the peak ratio. (b) Approx-DP (Thm. IV.4), Laplace vs. Gaussian, for $1 - \alpha > 0 . 8$ and measured against $\Delta _ { 2 } f$

$$
\overline { { A ^ { k } } } ( f _ { \theta } , D ) = \sum _ { i } \operatorname* { m a x } \left( \theta _ { U , i } ^ { k + 1 } - \theta _ { L , i } ^ { k } , \theta _ { U , i } ^ { k } - \theta _ { L , i } ^ { k + 1 } \right) .
$$

It is now clear that ${ \overline { { A ^ { k } } } } ( f _ { \theta } , D ) \geq A ^ { k } ( f _ { \theta } , D )$ . Thus, applying the same construction as in $\mathrm { { A l g . } \ 1 }$ , yields a $\beta \mathrm { . }$ -smooth upper bound on the smooth sensitivity:

$$
\overline { { \mathrm { S S } ^ { \beta } } } ( f _ { \theta } , D ) = \operatorname* { m a x } _ { k } e ^ { - \beta k } \cdot \overline { { A ^ { k } } } ( f _ { \theta } , D ) \geq \mathrm { S S } ^ { \beta } ( f _ { \theta } , D ) .
$$

Finally, the privatization mechanism follows naturally: indeed, calibrating Cauchy noise to $\overline { { \mathrm { S S } ^ { \beta } } } ( f _ { \theta } , D )$ ensures the released parameters themselves are private, which exactly matches the guarantee of DP-SGD:

Corollary V.1 (Single-Release AGS). Let $f$ be a model with parameters $\theta \in \Theta \subseteq \mathbb { R } ^ { p }$ , let $\theta _ { \mathrm { i n i t } }$ be a random initialization, and let $\theta ^ { n _ { s } } = \mathcal { M } ( f , \theta _ { \mathrm { i n i t } } , D )$ be the parameters obtained after $n _ { s }$ training steps. Let $u ( \beta ) : = \mathrm { S S } ^ { \beta } ( f _ { \theta ^ { n _ { s } } } , D )$ denote the $\beta \mathrm { \cdot }$ smooth sensitivity. The release $\begin{array} { r c l c r } { { \theta _ { \mathrm { p r i v a t e } } ^ { n _ { s } } } } & { { = } } & { { \theta ^ { n _ { s } } } } & { { + } } & { { \eta _ { u ( \beta ) } , } } \end{array}$ where $\eta _ { u ( \beta ) } \in \mathbb { R } ^ { p }$ has independent coordinates, each with density:

(C) η<sub>u(β),j</sub> ∼ Cauchy(6 u/ϵ) , β ≤ ϵ/(6p), (L) $\eta _ { u ( \beta ) , j } \sim \mathrm { L a p } ( 2 u / \epsilon ) , \qquad \beta \le \epsilon / \bigl ( 2 p ( 1 + 2 \ln ( 2 p / \delta ) ) \bigr )$ satisfies pure-DP (for C) and approx. DP (for L).

Proof. Both noise densities are admissible in the sense of [12, Def. 2.4], so the claim is [12, Lemma 2.5] with $S = u ( \beta )$

and $\alpha = \epsilon / 6 , \mathrm { r e s p . } \ \alpha = \epsilon / 2 ;$ the admissibility is proved in Appendix B-C. □

Advanced Composition for Repeated Release: Where we release the parameters n times along one trajectory, we can apply advanced composition in order to get tighter privacy guarantees than basic composition. Formally:

Corollary V.2 (Advanced Composition for Repeated Release). Suppose that each of the n releases along the trajectory of Algorithm 2 is obtained by Corollary V.1 at budget $( \epsilon _ { 0 } , \delta _ { 0 } )$ Then, for any $\delta _ { \mathrm { a c } } \in ( 0 , 1 )$ , their joint release is

$$
\left( \epsilon _ { 0 } \sqrt { 2 n \ln ( 1 / \delta _ { \mathrm { a c } } ) } + n \epsilon _ { 0 } ( e ^ { \epsilon _ { 0 } } - 1 ) , n \delta _ { 0 } + \delta _ { \mathrm { a c } } \right) \cdot \mathrm { D P } .
$$

Proof. Each release is $( \epsilon _ { 0 } , \delta _ { 0 } ) { \scriptstyle - \mathrm { D P } }$ by Corollary V.1 and the releases are adaptively composed along one trajectory, so the bound is the advanced composition theorem of [31, Thm. III.3]. □

## B. Abstract Gradient Sampling (AGS)

Using AGT-R for post-training privatization bears substantial similarities with existing private training algorithms: gradients are clipped to enforce a bounded sensitivity and carefully calibrated noise is added to privatize the model parameter. In popular private learning algorithms, however, noise addition is done at each step of the learning process. In this section, we generalize post-training privatization to allow the application of AGT-R privatization at intermediate steps of the learning algorithm. We term this generalization Abstract Gradient Sampling.

To make this generalization we define the notion of intermediate privatization through “sampling” at step $i \in \{ 1 , \ldots , n _ { s } \}$ exactly as in Corollary V.1 but with $u ( \beta , i )$ defining an upperbound on the $\beta \mathrm { . }$ -smooth sensitivity at step i we can use the single release mechanism: $\theta _ { \mathrm { p r i v a t e } } ^ { i } = \theta ^ { i } + \eta _ { u ( \beta , i ) } \mathbf { W h i l e } \ \theta _ { \mathrm { p r i v a t e } } ^ { i }$ can be released with an accompanying privacy guarantee, we are often only interested in the final learning parameter, therefore, we must establish how to carry out our private learning algorithm from an intermediate point. In what follows we first demonstrate how multiple releases affects the AGT semantics and how to leverage tight privacy accounting for multiple releases.

1) Multiple Release and AGT Semantics: The AGT algorithm accounts for dataset sensitivity through the use of abstract interpretation and formally verified parameter envelopes. Once a parameter has been released with DP guarantees, however, the DP mechanism accounts for the dataset sensitivity, and intuitively the parameter envelopes are no longer necessary. We make this formal with Lemma V.1:

Lemma V.1 (Envelope Collapse). Sampling collapses certified envelopes ∀k onto the released point: $[ \theta _ { L } ^ { \dot { k } , i } , \bar { \theta } _ { U } ^ { k , i } ] \ = \ \{ \theta _ { \mathrm { p r i v a t e } } ^ { i } \}$

Proof. To safely collapse envelopes one must prove two properties (i) the privacy guarantee is accounted for and (ii) the AGT semantics are preserved. Satisfaction of (i) follows directly from Thm. V.1 which gives the DP mechanism. Satisfaction of (ii) is done by observing that if ∀k, $\theta _ { L } ^ { k } = \theta _ { U } ^ { k }$ then $\forall k \operatorname* { m a x } _ { k } \delta ( x , k ) = 0$ and finally $\forall \beta , \overline { { \mathrm { S S } ^ { \beta } } } = 0$ implying that the AGT release deterministically reports $\theta _ { \mathrm { p r i v a t e } } ^ { i } .$ Thus collapsing all envelopes preserves the privacy guarantee and maintains valid AGT semantics. □

2) Tighter Privacy Accounting for AGS: The final privacy guarantee for AGS must cover all of its parameter releases, which can be done with composition:

Theorem V.1 (AGS Sequential Composition). Let $S = \{ i _ { 1 } <$ $\dots < i _ { m } \} \subseteq \left\{ 1 , \dots , n _ { s } \right\}$ with $i _ { m } = n _ { s }$ be the sampling steps, partitioning training into windows $( i _ { j - 1 } , i _ { j } ]$ (with $i _ { 0 } = 0 )$ , each certified by $\mathcal { M } _ { \mathrm { A G T } }$ from the release preceding it. If the release at step $i _ { j } \mathrm { i s } ( \epsilon _ { j } , \delta _ { j } )$ -DP, then $\theta _ { \mathrm { p r i v a t e } } ^ { n _ { s } }$ is released with guarantee $\begin{array} { r l } { \left( \sum _ { j = 1 } ^ { m } \epsilon _ { j } , } & { { } \sum _ { j = 1 } ^ { m } \delta _ { j } \right) \mathrm { - D P } . } \end{array}$

As with prior sections, we highlight that sequential composition is substantially looser the state-of-the-art numerical composition [38]. We can leverage these results by observing that for every AGS mechanism (Corollary V.1), M, we can let

$$
\delta _ { \mathcal { M } } ( \epsilon ) : = \operatorname* { i n f } \left\{ \delta : \mathcal { M } \mathrm { ~ i s ~ } ( \epsilon , \delta ) \mathbf { - D P } \right\}
$$

denote its privacy curve which can be subsequently optimized by numerical accounting techniques which can substantially improve the privacy guarantee [38].

3) Statement of the AGS Algorithm: The complete algorithm we propose, Abstract Gradient Sampling (AGS), can be found in Algorithm 2. AGS receives a user-defined set of steps at which sampling should occur, along with the desired privacy budget for the windows. Then, for each window, AGT is performed to obtain parameter envelopes and compute the upper bound on smooth sensitivity, which is subsequently leveraged to perform the noise addition and privatize the window’s parameters, as per Theorem V.1. Since the privacy cost has been paid by sampling, the parameter bounds are reset (Lemma V.1) and the budget is accounted for through composition (Theorem V.1). Finally, the algorithm returns a private set of parameters, along with a total privacy guarantee.

## C. Analysis of AGS

The power of AGS lies in the ability to control the moment of sampling (i.e., privatization). This is a key advantage: indeed, due to this particularity of our algorithm, it is possible to design a strategy that only releases parameters when the smoothsensitivity-calibrated noise has a lower magnitude than the noise DP-SGD would accumulate over the same steps. We investigate this strategy theoretically and provide closed-form expressions for the conditions that need to be satisfied for AGS to be preferable to DP-SGD.

Algorithm 2 AGS   
1: Input: Dataset D, function $f ,$ initialization $\theta _ { \mathrm { i n i t } }$ , sampling steps   
set $S \subseteq \{ 1 , . . . , n _ { s } - 1 \} \cup \{ \bar { n } _ { s } \}$ , step privacy budgets $\epsilon _ { i } , \delta _ { i } , \forall i \in$   
S.   
2: $\epsilon  0 , \delta  0$   
3: $\theta _ { L } ^ { 0 } = \dot { \theta } _ { \mathrm { i n i t } } = \theta _ { U } ^ { 0 }$   
4: $\theta _ { \mathrm { p r i v a t e } } ^ { \mathrm { c u r r } } = \theta _ { \mathrm { i n i t } }$ {Window’s private parameter start}   
5: for i in S do   
6: $[ \theta _ { L } ^ { i } , \theta _ { U } ^ { i } ] , \theta ^ { i } \gets \mathcal { M } _ { \mathrm { A G T } } ( f , \theta _ { \mathrm { p r i v a t e } } ^ { \mathrm { c u r r } } , D , \cdot )$   
7: $\theta _ { \mathrm { p r i v a t e } } ^ { \mathrm { c u r r } }  \theta ^ { i } + \mathrm { A G S } ( \theta _ { L } ^ { i } , \theta _ { U } ^ { i } )$   
8: {Paid privacy cost - reset interval}   
9: $\theta _ { U } ^ { i } = \theta _ { L } ^ { i } = \theta _ { \mathrm { p r i v a t e } } ^ { \mathrm { c u r r } }$   
10: $\epsilon  \epsilon + \epsilon _ { i } , \delta  \delta + \delta _ { i }$   
11: end for   
12: return $\theta _ { \mathrm { p r i v a t e } } ^ { \mathrm { c u r r } } , \epsilon , \delta$

Utility Analysis - Pure DP: The analysis of the incurred privacy cost is straightforward via the scale parameter and standard composition for both Laplace DP-SGD and AGS. However, utility is considerably difficult to characterize: noise addition in parameter space affects the per-step quality parameter estimate non-linearly and additionally, gradient descent counteracts this effect. Thus, the utility analyses of Theorems IV.3 and IV.4 are not applicable. While our prior bounds introduced no additional assumptions on top of those made by Sosnin et. al. [13], in what follows we will need to reason about training dynamics, which require introducing additional assumptions. For (ϵ, 0)-DP, we present below a condition for AGS to outperform Laplace DP-SGD. The theorem’s proof is deferred to Appendix B-D.

Theorem V.2. Let the loss function $\mathcal { L } ( \pmb \theta , D )$ be 1-dimensional, µ-strongly convex, and L-smooth with a locally constant Hessian across the $s _ { p } .$ -step window. Consider the DP-SGD parameter update over $s _ { p }$ steps with a learning rate $\eta \leq 1 / L$ , where the gradient is perturbed by i.i.d. noise $Z _ { s } \sim \mathrm { L a p } ( ( s _ { p } \Delta f ) / \epsilon )$ with mean zero (so that the final per-window privacy guarantee is $( \epsilon , 0 ) { \tt - D P }$ by basic composition). For a user-defined confidence level $\alpha \in ( 0 , 1 )$ , a single AGS update using Cauchy noise with scale $\gamma = 6 \mathrm { S S } ^ { \beta } ( f ) / \epsilon$ and $\beta < \epsilon / 6$ yields a strictly tighter parameter space error bound than Laplace DP-SGD with probability 1 − α whenever:

$$
\begin{array} { r l } & { \mathrm { S S } ^ { \beta } ( f _ { \theta } , D ) < \frac { s _ { p } \Delta f } { 3 \tan \big ( \frac { \pi } { 2 } ( 1 - \alpha ) \big ) } \times } \\ & { \quad \quad \times \operatorname* { m a x } \left( \sqrt { \frac { \eta \ln \frac { \alpha } { 2 } } { c \mu ( \eta \mu - 2 ) } } , \frac { - \eta \ln \frac { \alpha } { 2 } } { c } \right) , } \end{array}
$$

where $c > 0$ is an absolute constant.

Utility Analysis - Approximate DP: We now take the exact same approach in the approximate DP case, namely Gaussian DP-SGD versus AGS with Laplace noise calibrated to the smooth sensitivity. Proof is deferred to Appendix B-E.

Theorem V.3. Consider the regressor $f ,$ dataset D, confidence level $1 - \alpha .$ , an absolute constant $c > 0$ from the subgaussian Hoeffding inequality, and a fixed failure probability $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ of approximate-DP. Assume the loss $\mathcal { L }$ is $\mu -$ strongly convex and L-smooth with a locally constant Hessian across the $s _ { p } \mathrm { - s t e p }$ window, and that the step size satisfies $\eta ~ \leq ~ 1 / L$ . Then a single addition of Laplace noise with scale $b = 2 \mathrm { S S } ^ { \beta } ( f ) / \epsilon$ and $\beta < \epsilon / ( 2 \ln ( 2 / \delta ) )$ (AGS) provides tighter error bounds than Gaussian DP-SGD with parameter $\sigma = s _ { p } \Delta _ { 2 } f \sqrt { 2 \ln ( 1 . 2 5 s _ { p } / \delta ) } / \epsilon$ (composing to $( \epsilon , \delta ) \ / – \mathrm { D P }$ across the window) in the sense that $E _ { L } < E _ { G }$ , whenever:

$$
\mathrm { S S } ^ { \beta } ( f _ { \theta } , D ) < \frac { 1 } { 2 \ln \left( \frac { 1 } { \alpha } \right) } \sqrt { \frac { 1 6 \eta s _ { p } ^ { 2 } ( \Delta _ { 2 } f ) ^ { 2 } \ln \left( \frac { 1 . 2 5 s _ { p } } { \delta } \right) \ln \left( \frac { 2 } { \alpha } \right) } { 3 c \mu ( 2 - \eta \mu ) } } ,
$$

where $\operatorname { S S } ^ { \beta } ( f , x )$ is the β-smooth sensitivity and $\delta$ is the allowed privacy leakage in approximate DP.

These theorem and subsequent proofs begin to demonstrate the core of the interplay between AGS and classical DP-SGD dynamics. Once we have access to the per-parameter error bounds, one can bound the total error for the whole set of parameters, albeit rather loosely, using the triangle inequality, or extend our approach using the Matrix Bernstein inequality in Thm. 5.4.1. of [39]. Even having access to only the perparameter utility error bounds, the performance degradation can be easily empirically measured by simply employing bound propagation techniques, such as IBP [40] or even AGT [13] itself.

## D. Limitations and Future Work

AGS is one of the few algorithms to approach private learning from the lens of formal methods, and the first to employ parameter envelopes and frame learning as regression to satisfy differential privacy. Although performant in multiple scenarios, as we will demonstrate in §VI, it comes with some limitations, which we discuss below.

Computationally, it is more expensive to obtain guarantees due to the fact that AGT costs roughly 4× as much compared to a nominal training run, and the smooth sensitivity release requires one such pass per radius of the sensitivity staircase. Although in practice we require < 10 values for the staircase, this amounts to an order of magnitude increase in computational complexity. For tighter parameter envelopes, large batches significantly help, and so does more exact optimization (LP, MILP, MIQCP), which additionally increase the cost. Scopewise, the noise of a single release grows with the number of released parameters, since the sensitivities of the individual coordinates add up in the certificate, so AGS is best suited to compact models, or to the trainable part of a larger pre-trained one, rather than to end-to-end training of deep networks. The parameter envelopes that the certificate is built on also loosen as they are propagated through non-linear layers, which makes deep models harder to certify tightly. Regarding the privacy analysis, the smooth sensitivity mechanism draws less benefit from the failure probability $\delta$ than mechanisms with dataindependent noise such as the Gaussian mechanism, whose guarantees improve under composition; narrowing this gap would strengthen AGS under approximate differential privacy. Finally, each release along the training trajectory is currently accounted for as a separate mechanism and paid for from the same budget. A line of recent work shows that when only the final model is published, and the intermediate models are never observed, the privacy loss of an iterative algorithm can be far smaller than this composition suggests; bringing that view to the certified trajectory of AGS is the direction we find most promising.

## VI. EXPERIMENTS

In this section, we systematically validate the strength of AGT-R and AGS in a variety of toy and real-world settings.

## A. AGT-R

1) Condition Simulations: We start by presenting a brief analysis of AGT-R from the lens of the conditions derived in Theorems IV.3 and IV.4. We run simulations to assess at which ratios between the smooth and global sensitivity (i.e., $\rho = { \mathrm { S S } ^ { \beta } } / { \Delta _ { p } f } , p \in \{ 1 , 2 \} )$ , for increasing confidence levels $1 - \alpha$ , and (user-chosen) utility improvement constant c, the conditions hold. Figure 1 shows the simulation: pure $( \epsilon , 0 ) { \tt - D P }$ in the left column and approximate $( \epsilon , \delta ) \ / – \mathrm { D P }$ in the right.

The graph on the left hand side reveals that, in the best case, private predictions with Cauchy noise calibrated to the smooth sensitivity will yield smaller errors than Laplace-calibrated private predictions at confidence $1 - \alpha \approx 0 . 4$ and sensitivity ratio $\rho \ < \ 0 . 1 1 7$ . This essentially means that, in order to preserve utility, our AGT-based upper bounds on $\mathsf { S } \mathsf { S } ^ { \beta }$ must be at least ten times smaller than the global sensitivity. However, this always holds for unbounded output spaces.

Turning to the approximate $( \epsilon , \delta ) \ / – \mathrm { D P }$ setting, we now restrict the plot to the high-confidence regime $1 - \alpha > 0 . 8$ . The feasibility region is markedly more relaxed: since the reference is $\Delta _ { 2 } f \le \Delta _ { 1 } f$ and both mechanisms are light-tailed, the admissible ratios exceed 1, peaking at $\rho \approx 1 . 9$ for $c = 1$ . A ratio $\rho > 1$ means our bound need not even improve on the global sensitivity to win, so at $c = 1$ the Laplace-calibrated smooth sensitivity is preferable across essentially the whole range, dropping below unity only past $1 - \alpha \approx 0 . 9 9 5$ . A detailed analysis, including the constraints on $\beta$ that make these curves optimistic ceilings, is deferred to Appendix C-A.

![](images/21fa2c275a31ca3cc28b92e446287d2ef059c197dbff88984bb6d9f5d6747c89.jpg)

(a) PATE comparison. Error above the non-private one for PATE-R and PATE-AGT-R at their best T, AGT-R on the full data, and AGT-R on one shard with the amplified budget.  
![](images/f90cbe5d1af5989db0dbbebf5610d1e36aae59156e934bf42db8d0ac1484393b.jpg)  
(b) AGT-R ablations. Linear: the concretisation frequency F of AGT against IBP; California: the batch size. Pure DP with the tighter Cauchy release at $\beta = \epsilon / 2$ (MAE), approximate DP with Laplace at $\delta = 1 0 ^ { - 5 }$ (test MSE), global sensitivity in black. Mean over 200 noise trials.  
Figure 2: AGT-R on Linear and California under pure (left of each pair) and approximate (right) DP.

2) Real-World Datasets: For AGT-R, we consider two regression settings. The first is a synthetic linear regression task $( y = 2 x + 1 + \xi , 4 0 , 0 0 0$ training points) learned by a linear model. The second is the California Housing dataset [41]– [43], comprising 20,640 records with 8 features that predict median house value, learned by an 8-64-1 ReLU network. We compare our novel method in the pure and approximate DP settings with data-independent private prediction, namely the Laplace and Gaussian mechanisms with global sensitivity. We additionally use as a baseline PATE [18] with mean aggregation for regression (which we term PATE-R) and lastly equip each PATE shard with AGT-R capabilities and coin this technique PATE-AGT-R. For the two latter private prediction algorithms, we denote as T the number of shards and report each at its best T per budget. Lastly, we run AGT-R on a single shard with the budget amplified by sampling without replacement. Figure 2 reports, for both datasets, the mean absolute error (MAE) under pure DP and the MSE under approximate DP $( \delta = 1 0 ^ { - 5 } )$ : the top row compares AGT-R with the baselines above, the bottom row ablates AGT-R itself. Dataset, mechanism and training details are given in Appendix E-A.

a) Private prediction baselines (Figure 2a).: Figure 2a reports the excess error over the non-private prediction, which keeps arms that reach the non-private floor apart on the log axis;

the raw scale with global sensitivity is Figure 8 (Appendix C-D). There, across both datasets and settings, private prediction with global sensitivity ranks last in MAE (pure DP) and MSE (approximate DP): at ϵ = 1, the best arm’s error is 57 times (linear) and 6 times (California) lower in pure DP, and five and three orders of magnitude lower in approximate DP. Because of the task’s simplicity, in the linear dataset case, AGT-R’s and PATE’s variants are essentially indistinguishable from a performance point of view, with one exception. Indeed, at a low budget (ϵ = 0.1) in approximate DP, AGT-R on the full data is two orders of magnitude worse in terms of error; however, through dataset subsampling and amplification, this gap is quickly closed.

On the California dataset, the differences are more noticeable. While AGT-R on the full data is the least performant of the four arms due to full-data training, it still consistently outperforms data-independent private prediction (Appendix C-D). Notably, at ϵ = 1 its error is 4.5 times lower in pure DP and over 30 times lower in approximate DP. Interestingly, the most accurate technique in pure DP is PATE-R, while the winner in approximate DP is PATE-AGT-R. The explanation for this is simple: because the Cauchy distribution is heavy-tailed and each point is the mean over 200 noise trials, the AGT-based arms tend to sample far-from-mean points more often, which means the utility degradation is more acute. In contrast to that, the addition of Laplace noise (which has finite second moments) and the 1/T reduction in sensitivity from averaging over shards essentially make PATE-AGT-R state-of-the-art, with test MSE within 0.06 of the non-private one (0.94) at $\epsilon \geq 2 $

![](images/e7df03c54a37e375ca1220d34880dd790d07b8bc329924d001c9f0a0ff505d66.jpg)  
Figure 3: Privacy–utility trade-offs of AGS for 5-class MNIST (left pair) and IMDB (right pair). AGS releases the trained parameters once (Cauchy mechanism under pure DP, Laplace under approximate DP, $\delta = 1 0 ^ { - 5 } )$ . Matched DP-SGD is accounted by basic composition or, under approximate DP, by exact composition of the Gaussian mechanism; optimized DP-SGD is tuned on a public selection set. On IMDB, the hidden-layer arms replace the logistic head with one hidden layer.

b) Batch and concretization frequency (Figure 2b).: For the linear regression ablation we train a linear model with SGD for 16 steps (batch size 5), and for California Housing a ReLU MLP (input–64–1) for 330 SGD steps at every batch size, both with gradient clipping at $\gamma = 0 . 1 ;$ ; the global sensitivity is the clipping ceiling of the trajectory (0.64) for linear regression and the output range (10 standard deviations) for California Housing.

The reachability problem solved by AGT [13] is solved over windows (termed concretization frequencies). A key advantage of our approach to privacy is that it exposes formal method techniques as explicit tuning knobs. We show that increasing the concretization frequency from interval bound propagation to a MILP every F = 16 steps leads to substantially tighter smooth sensitivity estimates, from about 0.25 to 0.015, giving a four- to six-fold improvement in utility.

For the California Housing dataset, we ablate the batch size BS, which is known to influence the AGT algorithm, up to the full batch (BS = 16,512). As increasing the batch size tightens the bounds from AGT it should in turn tighten the smooth sensitivity upper bounds, as well as downstream utility. Indeed we see that smooth sensitivity and batch size appear roughly inversely proportional: BS=250 peaks near 6, BS=4000 near 0.4, while the full batch is nearly flat close to zero (Appendix C-B).

The critical point we would like to emphasize is that when compared with global sensitivity, our method is dominant in approximate DP at every privacy budget, frequency and batch size, with up to two orders of magnitude lower MSE. In pure DP, it dominates once the certificate is tight enough $( F \ge 8$ from ϵ = 0.2 on linear regression, $\mathtt { B S } \ge 1 0 0 0$ from ϵ = 1 on California Housing).

## B. AGS

We evaluate AGS’s performance with three distinct datasets: blobs [44] (two Gaussian clusters in 8 dimensions), MNIST [45] and the IMDB binary movie sentiment analysis task [46]. We further augment this evaluation, in Appendix D-A, with an investigation of the behaviour of AGS upon varying the number of releases (i.e., “sampling” steps) at a fixed privacy budget. Finally, we benchmark a hybrid DP-SGD+AGS approach on the SST2 dataset [47] setup of [48] and [49] against DP-LoRA finetuning.

1) Utility: For the results on the blobs dataset (200 points, linear hinge-loss classifier), which can be seen in Figure 4, the parameter envelopes are computed by solving an exact optimization problem, at different concretization frequencies F (the number of training steps after which the MILP is solved). We compare AGS with PATE [18] and matched (i.e., with the exact same parameters) DP-SGD. Looking at the approximate DP arm of the figure, there is strong evidence that AGS is the stronger mechanism: from ϵ = 5 it leads both baselines, by more than 0.2 in accuracy at ϵ = 20, and approaches the non-private accuracy of 0.90 at ϵ = 100 whatever the concretization frequency. In pure DP, the frequency matters: solving the MILP every 10 steps instead of every step raises the accuracy at ϵ = 30 from 0.59 to 0.86.

Figure 3 shows how our approach fares in more difficult tasks from a performance perspective (i.e. task accuracy) compared to the same baselines as in blobs, with the addition of optimized DP-SGD, where we use amplification and composition to the greatest extent in trying to optimize for accuracy. We restrict MNIST to its first five digits (28,596 private training images), a multi-class problem on which the parameter envelope has to bound 45 parameters instead of the 9 of blobs, and on IMDB we train on 20,000 private reviews embedded by the frozen all-mpnet-base-v2 sentence encoder [50], [51], the family of pretrained text encoders that DP text pipelines build on [52], showing that AGS scales to foundation-model backbones by certifying only the model trained on top of them. In both cases this model is a linear head on 8 PCA components of the inputs (of the 768-dimensional sentence embeddings on IMDB). Without fail, this version wins across all our setups, showing performance on-par with the non-private benchmark even at ϵ = 1. Matched DP-SGD with accounting does so as well, in the approximate DP cases. As expected, due to the heavy-tailedness of the Cauchy distribution, AGS saturates later in the case of pure DP, reaching accuracies on-par with the non-private benchmark at ϵ = 15 on 5-class MNIST and $\epsilon = 5$ on IMDB. With the exception of pure DP on MNIST, AGS consistently outperforms PATE from $\epsilon = 2 .$ , notably showing $> 0 . 1$ gains in accuracy for $1 . 5 \le \epsilon \le 7$ in approximate DP on IMDB. A similar behaviour is observed when comparing AGS with basic matched DP-SGD, whereby our method also wins in pure DP on MNIST from ϵ = 5.

![](images/2ef59393e58a1ad60db7446cf45b54624f24873a99754282d7a7896ed0b41b2c.jpg)  
Figure 4: Privacy–utility trade-offs of AGS on blobs. AGS certifies the parameter envelope with an exact MILP solved every F training steps and releases the trained parameters once (Cauchy mechanism under pure DP, Laplace under approximate DP, $\delta \ : = \ : 1 0 ^ { - 5 } )$ . DP-SGD (matched) uses the same hyperparameters with basic composition. The dotted line marks the non-private accuracy. Medians over 100 noise draws.

## C. Hybrid Approach

We employ the SST-2 dataset [47] to show that AGS can work well in conjunction with DP-SGD and reaches performance within 4 points of non-private fine-tuning (94.3%), which is also on-par with the best full DP-SGD method. As in §VI-B1, we use the all-mpnet-base-v2 sentence encoder on the binary sentiment analysis task induced by our dataset, whose 67,349 training sentences we split into 5,000 held-out test sentences and 62,349 private training instances, with the 872-sentence validation split as the public set.

The encoder is fine-tuned with DP-LoRA [48] (rank 16) using DP-SGD in Opacus [24] for 3 epochs at batch size 1,024 to $( \epsilon _ { 1 } , 5 \cdot 1 0 ^ { - 6 } ) – \mathrm { D P }$ , and the same model scored with its own head is the end-to-end DP-LoRA arm. The AGS head is a logistic head on the PCA-8 projection of the adapted features $( p = 1 8 )$ , released once at $( \epsilon _ { 2 } , 5 \cdot 1 0 ^ { - 6 } ) – \mathrm { D P } ,$ so that the pipeline is $( \epsilon _ { 1 } + \epsilon _ { 2 } , 1 0 ^ { - 5 } ) – \mathbf { D P } ;$ ; the split and the decision threshold are chosen on the public set (full details in Appendix E-B). While most settings tested we find that DP-LoRA is the best approach, in Figure 5 we do observe that hybrid DP-LoRA+AGS can outperform both DP-LoRA and AGS alone when the number of sentences is at 10k.

![](images/0a1b2571312937433af61a5569d4474aaeef73a74037971378de428075b40e5d.jpg)  
Figure 5: Hybrid approach on SST-2: a DP-LoRA fine-tuned encoder with an AGS head against end-to-end DP-LoRA finetuning and a frozen encoder with an AGS or a DP-SGD head. Test accuracy against the total budget $\epsilon = \epsilon _ { 1 } + \epsilon _ { 2 }$ under $( \epsilon , 1 0 ^ { - 5 } ) – \mathrm { D P }$ (left), a pure-DP head (middle), and the private set size N at ϵ = 3 effect on accuracy (right).

## VII. CONCLUSION

In this work, we proposed two novel differentially private algorithms — one for private prediction and one for private learning — grounded in formal verification via Abstract Gradient Training. Firstly, we formally introduced AGT-R, a novel private prediction method that guarantees differential privacy in arbitrarily complex regression settings. We characterized AGT-R both theoretically and empirically and provided closed-form conditions for when our formal verification-based upper-bound on smooth sensitivity yields better utility-privacy trade-offs. Notably, we tackle private regression without making additional assumptions on the output space, namely unboundedness or discreteness. We further leveraged our insights gathered from AGT-R to pose learning as a private regression problem, and use the same bounds originating from AGT to devise a novel private learning algorithm, AGS. We demonstrate theoretically how AGS satisfies (pure or approximate) differential privacy by sampling at predefined user-chosen steps from an appropriate noise distribution. Once again, we provide a closed form expression for the case when AGS provides better utility than standard DP-SGD. Lastly, we verify all our claims through carefully-curated experiments that show both the inner mechanisms workings of our proposed approaches, as well as how they compare to prior established DP-ensuring techniques.

## REFERENCES

[1] C. Dwork, F. McSherry, K. Nissim, and A. Smith, “Calibrating noise to sensitivity in private data analysis,” in Theory of cryptography conference. Springer, 2006, pp. 265–284.

[2] C. Dwork, A. Roth et al., “The algorithmic foundations of differential privacy,” Foundations and trends® in theoretical computer science, vol. 9, no. 3–4, pp. 211–407, 2014.

[3] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill et al., “On the opportunities and risks of foundation models,” arXiv preprint arXiv:2108.07258, 2021.

[4] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, “Deep learning with differential privacy,” in Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, 2016, pp. 308–318.

[5] J. Zhang, Z. Zhang, X. Xiao, Y. Yang, and M. Winslett, “Functional mechanism: regression analysis under differential privacy,” Proceedings of the VLDB Endowment, vol. 5, no. 11, pp. 1364–1375, 2012.

[6] K. Chaudhuri, C. Monteleoni, and A. D. Sarwate, “Differentially private empirical risk minimization.” Journal of Machine Learning Research, vol. 12, no. 3, 2011.

[7] F. Tramer and D. Boneh, “Differentially private learning needs better features (or much more data),” in International Conference on Learning Representations, 2021.

[8] C. Dwork and V. Feldman, “Privacy-preserving prediction,” in Conference On Learning Theory. PMLR, 2018, pp. 1693–1702.

[9] C. A. Choquette-Choo, N. Dullerud, A. Dziedzic, Y. Zhang, S. Jha, N. Papernot, and X. Wang, “Capc learning: Confidential and private collaborative learning,” arXiv preprint arXiv:2102.05188, 2021.

[10] L. van der Maaten and A. Hannun, “The trade-offs of private prediction,” arXiv preprint arXiv:2007.05089, 2020.

[11] M. R. Wicker, P. Sosnin, I. Shilov, A. Janik, M. N. Mueller, Y.-A. de Montjoye, A. Weller, and C. Tsay, “Certification for differentially private prediction in gradient-based training,” in Forty-second International Conference on Machine Learning, 2025.

[12] K. Nissim, S. Raskhodnikova, and A. Smith, “Smooth sensitivity and sampling in private data analysis,” in Proceedings of the thirty-ninth annual ACM symposium on Theory of computing, 2007, pp. 75–84.

[13] P. Sosnin, M. Wicker, J. Collyer, and C. Tsay, “Abstract gradient training: A unified certification framework for data poisoning, unlearning, and differential privacy,” arXiv preprint arXiv:2511.09400, 2025.

[14] C. Tsay, “Relaxation-informed training of neural network surrogate models,” arXiv preprint arXiv:2604.22746, 2026.

[15] L. Béthune, T. Massena, T. Boissin, A. Bellet, F. Mamalet, Y. Prudent, C. Friedrich, M. Serrurier, and D. Vigouroux, “Dp-sgd without clipping: The lipschitz neural network way,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 10 749–10 794.

[16] N. Papernot, S. Song, I. Mironov, A. Raghunathan, K. Talwar, and Ú. Erlingsson, “Scalable private learning with pate,” arXiv preprint arXiv:1802.08908, 2018.

[17] R. Bassily, O. Thakkar, and A. Guha Thakurta, “Model-agnostic private learning,” Advances in neural information processing systems, vol. 31, 2018.

[18] N. Papernot, M. Abadi, U. Erlingsson, I. Goodfellow, and K. Talwar, “Semi-supervised knowledge transfer for deep learning from private training data,” arXiv preprint arXiv:1610.05755, 2016.

[19] C. Dwork, K. Kenthapadi, F. McSherry, I. Mironov, and M. Naor, “Our data, ourselves: Privacy via distributed noise generation,” in Annual international conference on the theory and applications of cryptographic techniques. Springer, 2006, pp. 486–503.

[20] T. Steinke and J. Ullman, “Between pure and approximate differential privacy,” arXiv preprint arXiv:1501.06095, 2015.

[21] I. Mironov, “Rényi differential privacy,” in 2017 IEEE 30th computer security foundations symposium (CSF). IEEE, 2017, pp. 263–275.

[22] Y.-X. Wang, B. Balle, and S. P. Kasiviswanathan, “Subsampled rényi differential privacy and analytical moments accountant,” in The 22nd international conference on artificial intelligence and statistics. PMLR, 2019, pp. 1226–1235.

[23] A. Koskela, M. Tobaben, and A. Honkela, “Individual privacy accounting with gaussian differential privacy,” arXiv preprint arXiv:2209.15596, 2022.

[24] A. Yousefpour, I. Shilov, A. Sablayrolles, D. Testuggine, K. Prasad, M. Malek, J. Nguyen, S. Ghosh, A. Bharadwaj, J. Zhao et al., “Opacus: User-friendly differential privacy library in pytorch,” arXiv preprint arXiv:2109.12298, 2021.

[25] G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer, “Reluplex: An efficient smt solver for verifying deep neural networks,” in International conference on computer aided verification. Springer, 2017, pp. 97–117.

[26] X. Huang, M. Kwiatkowska, S. Wang, and M. Wu, “Safety verification of deep neural networks,” in International conference on computer aided verification. Springer, 2017, pp. 3–29.

[27] M. Wicker, L. Laurenti, A. Patane, and M. Kwiatkowska, “Probabilistic safety for bayesian neural networks,” in Conference on uncertainty in artificial intelligence. PMLR, 2020, pp. 1198–1207.

[28] E. Benussi, A. Patane, M. Wicker, L. Laurenti, and M. Kwiatkowska, “Individual fairness guarantees for neural networks,” arXiv preprint arXiv:2205.05763, 2022.

[29] M. Wicker, J. Heo, L. Costabello, and A. Weller, “Robust explanation constraints for neural networks,” arXiv preprint arXiv:2212.08507, 2022.

[30] H. Dang, M. R. Wicker, G. Botterweck, and A. Patane, “Certifiably quantisation-robust training and inference of neural networks,” in The 28th International Conference on Artificial Intelligence and Statistics, 2025.

[31] C. Dwork, G. N. Rothblum, and S. Vadhan, “Boosting and differential privacy,” in 2010 IEEE 51st annual symposium on foundations of computer science. IEEE, 2010, pp. 51–60.

[32] M. Bun and T. Steinke, “Concentrated differential privacy: Simplifications, extensions, and lower bounds,” in Theory of cryptography conference. Springer, 2016, pp. 635–658.

[33] K. Pan, Y.-S. Ong, M. Gong, H. Li, A. K. Qin, and Y. Gao, “Differential privacy in deep learning: A literature survey,” Neurocomputing, vol. 589, p. 127663, 2024.

[34] N. Papernot and T. Steinke, “Hyperparameter tuning with renyi differential privacy,” arXiv preprint arXiv:2110.03620, 2021.

[35] A. Koskela and T. D. Kulkarni, “Practical differentially private hyperparameter tuning with subsampling,” Advances in Neural Information Processing Systems, vol. 36, pp. 28 201–28 225, 2023.

[36] B. Balle, G. Barthe, and M. Gaboardi, “Privacy amplification by subsampling: Tight analyses via couplings and divergences,” in Advances in Neural Information Processing Systems, 2018.

[37] I. Mironov, K. Talwar, and L. Zhang, “R\’enyi differential privacy of the sampled gaussian mechanism,” arXiv preprint arXiv:1908.10530, 2019.

[39] R. Vershynin, “High-dimensional probability,” 2025.

[38] S. Gopi, Y. T. Lee, and L. Wutschitz, “Numerical composition of differential privacy,” Advances in Neural Information Processing Systems, vol. 34, pp. 11 631–11 642, 2021.

[40] S. Gowal, K. Dvijotham, R. Stanforth, R. Bunel, C. Qin, J. Uesato, R. Arandjelovic, T. Mann, and P. Kohli, “On the effectiveness of interval bound propagation for training verifiably robust models,” arXiv preprint arXiv:1810.12715, 2018.

[41] R. K. Pace and R. Barry, “Sparse spatial autoregressions,” Statistics & Probability Letters, vol. 33, no. 3, pp. 291–297, 1997.

[42] Y. Li, “The asymmetric house price dynamics: Evidence from the california market,” Regional Science and Urban Economics, vol. 52, pp. 1–12, 2015.

[43] R. K. Pace and R. Barry, “California housing dataset,” 1997, available via scikit-learn: https://scikit-learn.org/stable/datasets/real\_world.html# california-housing-dataset.

[44] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, no. 85, pp. 2825–2830, 2011.

[45] Y. Lecun, “The mnist database of handwritten digits,” http://yann. lecun. com/exdb/mnist/, 1998.

[46] A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts, “Learning word vectors for sentiment analysis,” in Proceedings ofthe 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, 2011, pp. 142–150.

[47] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Y. Ng, and C. Potts, “Recursive deep models for semantic compositionality over a sentiment treebank,” in Proceedings of the 2013 conference on empirical methods in natural language processing, 2013, pp. 1631–1642.

[48] D. Yu, S. Naik, A. Backurs, S. Gopi, H. A. Inan, G. Kamath, J. Kulkarni, Y. T. Lee, A. Manoel, L. Wutschitz et al., “Differentially private finetuning of language models,” arXiv preprint arXiv:2110.06500, 2021.

[49] X. Li, F. Tramer, P. Liang, and T. Hashimoto, “Large language models can be strong differentially private learners,” arXiv preprint arXiv:2110.05679, 2021.

[50] K. Song, X. Tan, T. Qin, J. Lu, and T.-Y. Liu, “Mpnet: Masked and permuted pre-training for language understanding,” Advances in neural information processing systems, vol. 33, pp. 16 857–16 867, 2020.

[51] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), 2019, pp. 3982–3992.

[52] C. Xie, Z. Lin, A. Backurs, S. Gopi, D. Yu, H. A. Inan, H. Nori, H. Jiang, H. Zhang, Y. T. Lee et al., “Differentially private synthetic data via foundation model apis 2: Text,” arXiv preprint arXiv:2403.01749, 2024.

[53] B. Becker and R. Kohavi, “Adult,” UCI Machine Learning Repository, 1996, extracted from the 1994 US Census database; 48,842 instances, 14 attributes.

[54] R. Iyengar, J. P. Near, D. Song, O. Thakkar, A. Thakurta, and L. Wang, “Towards practical differentially private convex optimization,” in 2019 IEEE symposium on security and privacy (SP). IEEE, 2019, pp. 299–316.

[55] R. Redberg, A. Koskela, and Y.-X. Wang, “Improving the privacy and practicality of objective perturbation for differentially private linear learners,” Advances in neural information processing systems, vol. 36, pp. 13 819–13 853, 2023.

[56] P. Kairouz, S. Oh, and P. Viswanath, “The composition theorem for differential privacy,” in International conference on machine learning. PMLR, 2015, pp. 1376–1385.

## APPENDIX A THEORETICAL DERIVATION OF PARAMETER-SPACE BOUNDS

To construct the parameter envelope $T _ { k } = [ \theta _ { L } ^ { k } , \theta _ { U } ^ { k } ]$ required by AGT-R, we must bound the gradient updates at each optimization step. Consider a regression model trained with a loss function $\mathcal L ( \cdot , \cdot )$ and a gradient clipping threshold $\gamma .$ Let $\mathcal { M }$ represent our gradient-based training algorithm and $\theta _ { \mathrm { i n i t } }$ be a fixed initialization. The valid parameter-space bound $T _ { k }$ must satisfy:

$$
\mathcal { M } ( f , \theta _ { \mathrm { i n i t } } , D ^ { \prime } ) \in T _ { k } \quad \forall D ^ { \prime } \mathrm { ~ s . t . ~ } d ( D , D ^ { \prime } ) \leq k .
$$

At each training iteration on a nominal batch B of size $b ,$ the set of possible descent directions under k arbitrary dataset additions or removals is bounded using either bound propagation (IBP [40]) or encoding the problem as an optimiza tion (LP/QP/MILP/MIQCP). We bound the perturbed update $\Delta \theta \in [ \theta _ { L } ^ { k } , \theta _ { U } ^ { k } ]$ element-wise as follows:

$$
\Delta \theta _ { L } = \frac { 1 } { b } \left( \mathrm { S E M i n } _ { b - k } \{ \delta _ { L } ^ { ( i ) } \} - k \gamma \mathbf { 1 } _ { d } \right) ,\tag{2}
$$

$$
\Delta \theta _ { U } = \frac { 1 } { b } \left( \mathrm { S E M a x } _ { b - k } \{ \delta _ { U } ^ { ( i ) } \} + k \gamma \mathbf { 1 } _ { d } \right) ,\tag{3}
$$

where SEMin and SEMax represent the sum of the elementwise bottom and top $b - k$ gradient bounds. The terms $\delta _ { L } ^ { ( i ) }$ and $\delta _ { U } ^ { ( i ) }$ are bound propagation- or optimization-derived bounds on the gradient of the loss $\mathcal { L }$ for the i-th sample with respect to the reachable parameters $\tilde { \theta } \in T _ { k - 1 }$ . The $k \gamma \mathbf { 1 } _ { d }$ term accounts for the worst-case scenario where the k differing points produce gradients exactly at the clipping threshold $\gamma$ in the adversarial direction.

By iteratively applying these worst-case updates across the entire optimization trajectory, we obtain the final envelope $T _ { k }$ To compute the upper bound on the smooth sensitivity for a regression output, we propagate the test query x through the network over $T _ { k }$ . The maximum output variation provides our certified bound:

$$
A _ { k } ( f , x ) = \operatorname* { m a x } _ { \tilde { \theta } \in { \cal T } _ { k } } \left\| f _ { \tilde { \theta } } ( x ) - f _ { \theta } ( x ) \right\| _ { 1 } \geq A _ { k } ( f , x ) .
$$

By substituting this upper bound into the AGT-R noise mechanism, we maintain $( \epsilon , \delta )$ -differential privacy while accounting for worst-case local sensitivity.

## APPENDIX B PROOFS

## A. Proof of Theorem IV.3

Proof. Let $Z _ { L } \ \sim \ \mathsf { L a p } ( b )$ and $Z _ { C } \ \sim \ \mathrm { C a u c h y } ( \gamma )$ with $\gamma , b$ defined as above. The worst case error $E _ { L }$ at an α quantile for the Laplace noise is $\mathbb { P } ( | Z _ { L } | \ > \ E _ { L } ) \ \le \ \alpha$ . Expanding using the CDF and remembering that the Laplace noise is symmetric about $\mu = 0 .$ , then $\mathbb { P } ( | Z _ { L } | > E _ { L } ) = 2 \cdot \mathbb { P } ( Z _ { L } >$ $E _ { L } ) = 2 [ 1 - ( 1 - ( 1 / 2 ) \exp ( - E _ { L } / b ) ) ] = \exp ( - E _ { L } / b ) \leq \alpha$ Rearranging, this yields a worst-case $\begin{array} { r } { E _ { L } = \frac { \Delta f } { \epsilon } \ln ( \frac { 1 } { \alpha } ) } \end{array}$ . We proceed analogously in the case of Cauchy noise: $\mathbb { P } ( | Z _ { C } | >$

$E _ { C } ) = 2 \cdot \mathbb { P } ( Z _ { C } > E _ { C } ) = 1 - ( 2 / \pi )$ arctan $\left( E _ { C } / \gamma \right) \leq \alpha$ We thus find an upper bound $\begin{array} { r } { E _ { C } \ = \ \frac { 6 \mathrm { S S } ^ { \beta } } { \epsilon } } \end{array}$ tan $\textstyle { \left( { \frac { \pi } { 2 } } \left( 1 - \alpha \right) \right) }$ Enforcing that $E _ { C } < \frac { 1 } { c } E _ { L }$ gives the desired result. □

## B. Proof of Theorem IV.4

Proof. Let $Z _ { G } \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ with σ defined as above, and $Z _ { L } \sim$ $\mathrm { L a p } ( 2 \mathrm { S S } ^ { \beta } / \epsilon )$ . The worst case error $E _ { G }$ at α is $\mathbb { P } ( | Z _ { G } | >$ $E _ { G } ) \leq \alpha$ . Since the Gaussian is symmetric around the mean and by standardizing $Z _ { G } ~ ( \mathrm { i . e . , } ~ Z _ { G } \dot { / } \sigma \sim \mathcal { N } ( 0 , 1 ) )$ , we have that $2 \cdot \mathbb { P } \left( \left( Z _ { G } / \sigma \right) > \left( E _ { G } / \sigma \right) \right) \leq \alpha$ and thus $1 - \Phi ( E _ { G } / \sigma ) \le \alpha / 2$ Rearranging, we obtain a worst case $\begin{array} { r } { E _ { G } = \sigma \Phi ^ { - 1 } ( 1 - \frac { \alpha } { 2 } ) } \end{array}$ For Laplace noise, we proceed analogously to Thm. (IV.3): $\mathbb { P } ( | Z _ { L } | > E _ { L } ) \le \alpha$ and setting our condition to be $E _ { L } < E _ { G }$ By symmetry, we have that $\mathbb { P } ( | Z _ { L } | > E _ { L } ) = 2 \cdot \mathbb { P } ( Z _ { L } >$ $E _ { L } ) = 2 [ 1 - ( 1 - ( 1 / 2 ) \exp ( - E _ { L } / b ) ) ] = \exp ( - E _ { L } / b ) \leq$ α. Once again, rearranging, we obtain a worst case $E _ { L } =$ $\begin{array} { r } { b \ln ( \frac { 1 } { \alpha } ) = \frac { \mathbf { \breve { 2 } S S } ^ { \beta } } { \epsilon } \ln ( \frac { 1 } { \alpha } ) } \end{array}$ . We enforce $\begin{array} { r } { E _ { L } < \frac { 1 } { c } E _ { G } } \end{array}$ to obtain the final result. □

## C. Proof of Corollary V.1

Recall [12, Def. 2.4]: a density h on $\mathbb { R } ^ { p }$ is $( \alpha , \beta )$ -admissible if, for all $\Delta$ with $\| \Delta \| _ { 1 } \le \alpha$ (the norm in which [12] define local sensitivity), all $| \lambda | \le \beta$ and all measurable $S , \operatorname* { P r } [ Z \in$ $S ] \le e ^ { \epsilon / 2 } \operatorname* { P r } [ \bar { Z } \in S + \dot { \Delta } ] + \delta / 2$ (sliding) and $\operatorname* { P r } [ Z \in S ] \leq$ $e ^ { \bar { \epsilon } / 2 } \operatorname* { P r } [ Z \in \bar { e } ^ { \lambda } S ] + \delta / 2$ (dilation), where $Z \sim h$

By [12, Lemma 2.5], $\theta ^ { n _ { s } } \ + \ ( u / \alpha ) Z$ is then $( \epsilon , \delta ) \cdot$ indistinguishable whenever u is a β-smooth upper bound on the $\ell _ { 1 }$ local sensitivity of $\theta ^ { n _ { s } }$ , which $u ( \beta )$ is by construction. It therefore suffices to show that $\begin{array} { r } { h ( z ) \ = \ \dot { \prod } _ { i = 1 } ^ { p } h _ { 1 } ( z _ { j } ) } \end{array}$ is $( \epsilon / 6 , \epsilon / ( 6 p ) )$ )-admissible with $\delta = 0$ for $h _ { 1 } ( z ) = \breve { 1 } / ( \pi ( 1 { + } z ^ { 2 } ) )$ , and $( \epsilon / 2 , \epsilon / ( 2 p ( 1 + 2 \ln ( 2 p / \delta ) ) ) ;$ -admissible for $h _ { 1 } ( z ) \ =$ $\begin{array} { r } { \frac { 1 } { 2 } e ^ { - | z | } ; } \end{array}$ the scales $u / \alpha$ are then $6 u / \epsilon$ and $2 u / \epsilon$ . Both properties follow from a bound on the log-density ratio, which is a sum over coordinates because h is a product: a ratio at most $e ^ { \epsilon / 2 }$ on a set B with $\operatorname* { P r } [ Z \notin B ] \le \bar { \delta / 2 }$ integrates to the required inequality.

Sliding. $\mid \frac { d } { d z }$ ln $h _ { 1 } | \leq 1$ for both densities $( \frac { 2 | z | } { 1 + z ^ { 2 } } \leq 1$ , resp. 1), so ln $\begin{array} { r } { h ( \tilde { z ) } - \ln h ( z + \Delta ) \le \sum _ { i } | \Delta _ { j } | = \| \hat { \Delta } \| _ { 1 } ^ { - } \le \alpha \le \epsilon / 2 } \end{array}$ for every z.

Dilation. The density of $e ^ { - \lambda } Z$ is $e ^ { p \lambda } h ( e ^ { \lambda } z )$ , so the log ratio is $\begin{array} { r } { \sum _ { i = 1 } ^ { p } \left[ \ln h _ { 1 } ( z _ { j } ) - \ln h _ { 1 } ( e ^ { \lambda } z _ { j } ) - \lambda \right] } \end{array}$ . (C) The j-th term is $g _ { j } ( \lambda ) - g _ { j } ( 0 ) - \lambda$ with $g _ { j } ( \lambda ) = \ln ( 1 + z _ { i } ^ { 2 } e ^ { 2 \lambda } )$ and $g _ { j } ^ { \prime } \in [ 0 , 2 )$ , hence at most $3 | \lambda | \le 3 \beta$ in absolute value; the sum is at most $3 p \beta = \epsilon / 2$ for every z, so $\delta = 0 . \left( \mathrm { L } \right)$ The j-th term is $| z _ { j } | ( e ^ { \lambda } - 1 ) - \overset { \cdot } { \lambda } \leq \beta ( 1 + 2 | z _ { j } | )$ , using $e ^ { \beta } - 1 \leq 2 \beta$ for $\beta \leq 1$ (which holds for every $\epsilon \leq 2 p )$ . Under h the $| z _ { j } |$ are i.i.d. Exp(1), so $B = \{ \operatorname* { m a x } _ { j } | z _ { j } | \leq \ln ( 2 p / \delta ) \}$ has $\operatorname* { P r } [ Z \notin B ] \leq p \cdot \delta / ( 2 p ) = \delta / 2$ by a union bound, and on B the sum is at most $p \beta \left( 1 + 2 \ln ( 2 p / \delta ) \right) = \epsilon / 2$ 口

## D. Proof of Theorem V.2

We start by computing a bound on the error term in the case of Laplace DP-SGD (which we will refer to, alternatively, as

noisy SGD). The clean and noisy updates are:

$$
\begin{array} { r l } & { \theta _ { t } = \theta _ { t - 1 } - \eta \partial _ { \theta _ { t - 1 } } \mathcal { L } ( \pmb { \theta } , D ) } \\ & { \tilde { \theta } _ { t } = \tilde { \theta } _ { t - 1 } - \eta \left( \partial _ { \tilde { \theta } _ { t - 1 } } \mathcal { L } ( \tilde { \pmb { \theta } } , D ) + Z _ { t } \right) , \quad Z _ { t } \sim \mathrm { L a p } \left( \frac { s _ { p } \Delta f } { \epsilon } \right) , } \end{array}
$$

yielding the error term $e _ { t } = \tilde { { \theta } } _ { t } - { \theta } _ { t } \mathrm { : }$

$$
e _ { t } = ( \tilde { \theta } _ { t - 1 } - \theta _ { t - 1 } ) - \eta \left( \partial _ { \tilde { \theta } _ { t - 1 } } \mathcal { L } ( \tilde { \theta } , D ) - \partial _ { \theta _ { t - 1 } } \mathcal { L } ( \theta , D ) \right) - \eta Z _ { t } .
$$

Noting that $e _ { t - 1 } = \tilde { \theta } _ { t - 1 } - \theta _ { t - 1 }$ , we expand the gradient of the loss function with respect to $\theta _ { t - 1 }$ at $\tilde { \theta } _ { t - 1 }$ using Taylor’s theorem with the Lagrange remainder to obtain:

$$
\begin{array} { r l } & { \partial _ { \theta _ { t - 1 } } \mathcal { L } ( \theta , D ) = \partial _ { \tilde { \theta } _ { t - 1 } } \mathcal { L } ( \tilde { \theta } , D ) } \\ & { \qquad + \partial _ { \tilde { \theta } _ { t - 1 } } ^ { 2 } \mathcal { L } ( \tilde { \theta } , D ) ( \theta _ { t - 1 } - \tilde { \theta } _ { t - 1 } ) } \\ & { \qquad + \mathcal { O } \left( \Vert \theta _ { t - 1 } - \tilde { \theta } _ { t - 1 } \Vert ^ { 2 } \right) . } \end{array}
$$

Rearranging, dropping the higher-order terms and noting that $\tilde { \theta } _ { t - 1 } - \theta _ { t - 1 } = e _ { t - 1 }$ , we obtain

$$
\partial _ { \tilde { \theta } _ { t - 1 } } \mathcal { L } ( \tilde { \theta } , D ) - \partial _ { \theta _ { t - 1 } } \mathcal { L } ( \theta , D ) = \partial _ { \tilde { \theta } _ { t - 1 } } ^ { 2 } \mathcal { L } ( \tilde { \theta } , D ) e _ { t - 1 } .
$$

Substituting this back, the recursive error term is

$$
\begin{array} { r l r } & { } & { e _ { t } \approx e _ { t - 1 } - \eta \partial _ { \tilde { \theta } _ { t - 1 } } ^ { 2 } \mathcal { L } ( \tilde { \theta } , D ) e _ { t - 1 } - \eta Z _ { t } } \\ & { } & { = \left( 1 - \eta \partial _ { \tilde { \theta } _ { t - 1 } } ^ { 2 } \mathcal { L } ( \tilde { \theta } , D ) \right) e _ { t - 1 } - \eta Z _ { t } . } \end{array}
$$

Iteratively expanding the provided expression, we obtain the $s _ { p }$ -window error term

$$
e _ { s _ { p } } = - \eta \left( \sum _ { s = 1 } ^ { s _ { p } } Z _ { s } \cdot \prod _ { r = s + 1 } ^ { s _ { p } } ( 1 - \eta h _ { r } ) \right) ,
$$

whereby we denoted by $h _ { r }$ (in the vein of a ‘’Hessian’‘) the second derivative of the loss function $\partial _ { \tilde { \theta } _ { r - 1 } } ^ { 2 } \mathcal { L } ( \tilde { \theta } , D )$ with respect to the noisy parameter. We take $e _ { 0 } { \stackrel { \cdot } { = } } 0$ , so that the first injected noise is $Z _ { 1 }$ . Taking the (Euclidean) 1-norm of the error term (i.e. absolute value, since we are in 1 dimension) we have that

$$
| e _ { s _ { p } } | = \left| \sum _ { s = 1 } ^ { s _ { p } } Z _ { s } \cdot ( - \eta ) \prod _ { r = s + 1 } ^ { s _ { p } } ( 1 - \eta h _ { r } ) \right| .
$$

Let us, for now, denote $\begin{array} { r c l } { a _ { s } } & { = } & { ( - \eta ) \prod _ { r = s + 1 } ^ { s _ { p } } \left( 1 - \eta h _ { r } \right) } \end{array}$ $a = ( a _ { 1 } , \ldots , a _ { s _ { p } } )$ and analyze $\textstyle \sum _ { s = 1 } ^ { s _ { p } } a _ { s } Z _ { s } |$ in isolation. In particular, notice that the Laplace distribution is a subexponential distribution and note the subexponential version of the Bernstein inequality in Theorem 2.9.1. of [39]. We assume the curvature is (locally) constant across the window, i.e. $h _ { r } \equiv h$ for $r ~ \in ~ \{ 1 , \ldots , s _ { p } \}$ , consistent with the first-order Taylor approximation employed above. Consequently the coefficients $a _ { s }$ are deterministic and independent of the noise variables $\{ Z _ { s } \}$ , which is precisely the condition required to apply the weighted subexponential Bernstein inequality (Corollary 2.9.2 of [39]) to $\sum _ { s } a _ { s } Z _ { s }$ . Using Corollary 2.9.2. of the same source [39], namely the weighted version of the subexponential Bernstein inequality, we obtain:

$$
\begin{array} { r l r } {  { \mathbb { P } \{ | e _ { s _ { p } } | \geq E _ { L } \} = \mathbb { P } \{ | \displaystyle \sum _ { s = 1 } ^ { s _ { p } } a _ { s } Z _ { s } | \geq E _ { L } \} } } \\ & { } & { \leq 2 \exp [ - c \mathrm { m i n } ( \frac { E _ { L } ^ { 2 } } { K ^ { 2 } \| a \| _ { 2 } ^ { 2 } } , \frac { E _ { L } } { K \| a \| _ { \infty } } ) ] , } \end{array}
$$

where $c > 0$ is an absolute constant and $K = \operatorname* { m a x } _ { s } \| Z _ { s } \| _ { \psi _ { 1 } }$ with $\| \cdot \| _ { \psi _ { 1 } }$ the subexponential norm (i.e. $\| Z \| _ { \psi _ { 1 } } = \operatorname* { i n f } \{ K >$ $0 : \mathbb { E } [ \exp ( | Z | / K ) ] \leq 2 \} \rangle$ ). Since $Z _ { s }$ are all i.i.d. Laplace with mean 0 and scale parameter $b = ( s _ { p } \Delta f ) / \epsilon$ , then $\forall s : K =$ $\| Z _ { s } \| _ { \psi _ { 1 } }$ , which can easily be computed as follows:

$$
{ \begin{array} { r l } & { \mathbb { E } [ \exp ( | Z _ { s } | / K ) ] = \displaystyle \int _ { \mathbb { R } } \exp \left( { \frac { | z | } { k } } \right) { \frac { 1 } { 2 b } } \exp \left( { \frac { - | z | } { b } } \right) d z } \\ & { \quad \quad \quad = { \frac { 1 } { b } } \displaystyle \int _ { 0 } ^ { \infty } \exp \left( - z { \frac { k - b } { b k } } \right) d z } \\ & { \quad \quad \quad = { \frac { k } { b - k } } \exp \left( - z { \frac { k - b } { b k } } \right) { \Bigg | } _ { 0 } ^ { \infty } } \\ & { \quad \quad \quad = { \frac { k } { k - b } } , } \end{array} }
$$

where we have enforced in the second equality that $k > b$ and flipped the sign of $z ,$ otherwise the integral would not converge. The expression is bounded by 2 when $\| Z _ { s } \| _ { \psi _ { 1 } } =$ $2 b = 2 ( s _ { p } \Delta f ) / \epsilon$

We now compute $\| a \| _ { 2 } ^ { 2 }$ and $\| a \| _ { \infty }$ . Firstly, we remind the reader that we have assumed $\mu \cdot$ -strong convexity and $L _ { - }$ smoothness, thus $\mu \leq h _ { r } \leq L ,$ , and we take the step size $\eta \leq 1 / L$ so that $0 \leq 1 - \eta h _ { r } \leq 1 - \eta \mu$ . Therefore, $| a _ { s } | \leq$ $\eta ( 1 - \eta \mu ) ^ { s _ { p } - s }$ and $\| a \| _ { \infty } =$ max<sub>s</sub> $| a _ { s } | \leq \eta ( 1 - \eta \mu ) ^ { 0 } = \eta$ (i.e. maximization happens when $s = s _ { p } )$

We bound the 2-norm in a similar way: $\begin{array} { r } { \| a \| _ { 2 } ^ { 2 } = \sum _ { s } a _ { s } ^ { 2 } \le } \end{array}$ $\textstyle \eta ^ { 2 } \sum _ { s = 1 } ^ { s _ { p } } ( 1 - \eta \mu ) ^ { 2 ( s _ { p } - s ) }$ . The latter is a geometric progression with ratio $r = ( 1 - \eta \mu ) ^ { 2 }$ , thus $\begin{array} { r } { \| a \| _ { 2 } ^ { 2 } \ \leq \ \eta ^ { 2 } \frac { 1 - ( \bar { 1 - \eta \mu } ) ^ { 2 s _ { p } } } { 1 - ( 1 - \eta \mu ) ^ { 2 } } \ \leq \ } \end{array}$ $\begin{array} { r } { \frac { \eta ^ { 2 } } { 1 - ( 1 - \eta \mu ) ^ { 2 } } = \frac { \eta } { \mu ( 2 - \eta \mu ) } } \end{array}$ . The last inequality follows since $( 1 -$ $\eta \mu \big ) ^ { 2 s _ { p } } \geq 0$ , so dropping it only increases the numerator, and the final equality uses $1 - ( 1 - \eta \mu ) ^ { 2 } = \eta \mu ( 2 - \eta \mu )$

Substituting $K = 2 ( s _ { p } \Delta f ) / \epsilon$ alongside our computed norms into the subexponential Bernstein inequality

$$
\begin{array} { r l } & { \mathbb { P } \big \{ | e _ { s _ { p } } | \ge E _ { L } \big \} \le 2 \exp \Big [ - c \cdot } \\ & { \qquad \cdot \operatorname* { m i n } \Biggl ( \frac { \epsilon ^ { 2 } E _ { L } ^ { 2 } \mu ( 2 - \eta \mu ) } { 4 \eta ( s _ { p } \Delta f ) ^ { 2 } } , \frac { \epsilon E _ { L } } { 2 \eta s _ { p } \Delta f } \Big ) \Big ] . } \end{array}
$$

To follow the same analysis as in Thm. IV.3, we express the error condition $E _ { L }$ in terms of a user-defined confidence $\alpha .$ Let us name the first argument in the minimum as A and the second as $B ;$ then the worst case error at an α quantile is: $2 \exp ( - c \operatorname* { m i n } ( A , B ) ) \leq \alpha$ . This results in min $( A , B ) \geq$ $\begin{array} { r } { \frac { - \ln ( \alpha / \hat { 2 } ) } { c } , \ \mathrm { s o } \ A \ge \frac { - \ln ( \overset { \cdot } { \alpha } / 2 ) } { c } } \end{array}$ and $\begin{array} { r } { B \geq \frac { - \ln ( \alpha / 2 ) } { c } } \end{array}$ simultaneously. We now solve for the smallest $E _ { L }$ satisfying each argument of the minimizer, naming $E _ { L , 1 }$ the solution w.r.t. $A$ and $E _ { L , 2 }$ the solution w.r.t. $B ,$ and obtain:

$$
E _ { L , 1 } = \sqrt { \frac { 4 ( s _ { p } \Delta f ) ^ { 2 } \eta } { \epsilon ^ { 2 } c \mu ( \eta \mu - 2 ) } \ln \frac { \alpha } { 2 } } \mathrm { a n d } E _ { L , 2 } = \frac { - 2 \eta s _ { p } \Delta f \ln \frac { \alpha } { 2 } } { \epsilon c } .
$$

Since both constraints must hold simultaneously, the worst-case error is the larger of the two, i.e. $E _ { L } = \operatorname* { m a x } ( E _ { L , 1 } , E _ { L , 2 } )$ . The error term in the Cauchy case is exactly as in the proof of Thm IV.3:

$$
E _ { C } = \frac { 6 \mathrm { S S } ^ { \beta } } { \epsilon } \tan \left( \frac { \pi } { 2 } ( 1 - \alpha ) \right) .
$$

Enforcing $E _ { C } ~ < ~ E _ { L }$ yields the required conditions under which AGS dominates Laplace DP-SGD. □

## E. Proof of Theorem V.3

We proceed exactly as in the proof of Thm. V.2, the only difference being that we now perturb the clean update with Gaussian noise and compare against a one-step addition of Laplace noise calibrated to smooth sensitivity $( \mathrm { i . e . }$ , we now find ourselves in the Approximate DP case). The clean and noisy updates are:

$$
\begin{array} { r l } & { \theta _ { t } = \theta _ { t - 1 } - \eta \partial _ { \theta _ { t - 1 } } \mathcal { L } ( \pmb { \theta } , D ) } \\ & { \tilde { \theta } _ { t } = \tilde { \theta } _ { t - 1 } - \eta \left( \partial _ { \tilde { \theta } _ { t - 1 } } \mathcal { L } ( \tilde { \pmb { \theta } } , D ) + Z _ { t } \right) , \quad Z _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , } \end{array}
$$

where $\begin{array} { r } { \sigma = \frac { s _ { p } \Delta _ { 2 } f } { \epsilon } \sqrt { 2 \ln ( 1 . 2 5 s _ { p } / \delta ) } } \end{array}$ is the noise scale of the Gaussian mechanism run at the per-step budget $( \epsilon / s _ { p } , \delta / s _ { p } )$ so that by basic composition the $s _ { p } \mathrm { - s t e p }$ window costs $( \epsilon , \delta )$ in total, matching the single AGS release. Since the Taylor expansion of the gradient and the subsequent unrolling of the recursion are identical to the Laplace case, we omit them here and pick up directly at the $s _ { p } .$ -window error term:

$$
e _ { s _ { p } } = - \eta \left( \sum _ { s = 1 } ^ { s _ { p } } Z _ { s } \cdot \prod _ { r = s + 1 } ^ { s _ { p } } ( 1 - \eta h _ { r } ) \right) ,
$$

where, as before, $h _ { r }$ denotes the ‘’Hessian’‘ $\partial _ { \tilde { \theta } _ { r - 1 } } ^ { 2 } \mathcal { L } ( \tilde { \theta } , D )$ Reusing the notation $\begin{array} { r } { a _ { s } = ( - \eta ) \prod _ { r = s + 1 } ^ { s _ { p } } ( 1 - \eta h _ { r } ) } \end{array}$ and $a =$ $( a _ { 1 } , \ldots , a _ { s _ { p } } )$ , we once again analyze $\textstyle \left| \sum _ { s = 1 } ^ { s _ { p } } a _ { s } Z _ { s } \right|$ in isolation. As in the Laplace case, we assume the curvature is locally constant across the window $( \mathrm { i } . \mathrm { e } . \ h _ { r } \equiv h )$ , so that the weights $a _ { s }$ are deterministic and independent of the noise $\{ Z _ { s } \}$ , as required by the concentration inequality we are about to invoke.

In contrast to the Laplace case, however, the Gaussian distribution is subgaussian, so we appeal to the general Hoeffding inequality for subgaussian random variables in Theorem 2.7.3 of [39]. Before doing so, we remind the reader that a constant a multiplied with a Gaussian is again Gaussian, namely $a \cdot { \mathcal { N } } ( m , \sigma ^ { 2 } ) \sim { \mathcal { N } } ( a m , a ^ { 2 } \sigma ^ { 2 } )$ (this follows immediately by inspecting the moment generating function). Hence each weighted term is itself a centered Gaussian,

$$
\boldsymbol { Z _ { s } ^ { \prime } } : = a _ { s } \boldsymbol { Z _ { s } } \sim \mathcal { N } \left( 0 , \eta ^ { 2 } \prod _ { r = s + 1 } ^ { s _ { p } } ( 1 - \eta h _ { r } ) ^ { 2 } \sigma ^ { 2 } \right) ,
$$

and the $Z _ { s } ^ { \prime }$ are independent, centered and subgaussian, exactly the regime in which the inequality applies. We obtain:

$$
\begin{array} { r l r } {  { \mathbb { P } \{ | e _ { s _ { p } } | \geq E _ { G } \} = \mathbb { P } \{ | \sum _ { s = 1 } ^ { s _ { p } } a _ { s } Z _ { s } | \geq E _ { G } \} } } \\ & { } & { \leq 2 \exp ( - \frac { c E _ { G } ^ { 2 } } { \sum _ { s = 1 } ^ { s _ { p } } \| Z _ { s } ^ { \prime } \| _ { \psi _ { 2 } } ^ { 2 } } ) , } \end{array}
$$

where $c > 0$ is an absolute constant and $\lVert \cdot \rVert _ { \psi _ { 2 } }$ denotes the subgaussian norm (i.e. $\| X \| _ { \psi _ { 2 } } =$ inf $\{ t > \stackrel { \cdot } { 0 } : \mathbb { E } [ \exp ( X ^ { 2 } / t ^ { 2 } ) ] \leq$ 2}). The subgaussian norm $\| \cdot \| _ { \psi _ { 2 } }$ of a centered Gaussian can be computed exactly as in the previous proof (mirroring the $\psi _ { 1 }$ -norm derivation for Laplace); by [39], for $X \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ it equals $\| X \| _ { \psi _ { 2 } } = \sigma { \sqrt { 8 / 3 } }$ . By the absolute homogeneity of the norm, it follows that $\lVert Z _ { s } ^ { \prime } \rVert _ { \psi _ { 2 } } = | a _ { s } | \sigma \sqrt { 8 / 3 }$ , and therefore $\begin{array} { r } { \sum _ { s = 1 } ^ { s _ { p } } \| Z _ { s } ^ { \prime } \| _ { \psi _ { 2 } } ^ { 2 } = \frac { 8 } { 3 } \sigma ^ { 2 } \| a \| _ { 2 } ^ { 2 } } \end{array}$

We now note that ∥a∥<sup>2</sup> is precisely the quantity we bounded in the proof of Thm. V.2, namely $\begin{array} { r } { \| a \| _ { 2 } ^ { 2 } \le \frac { \eta } { \mu ( 2 - \eta \mu ) } } \end{array}$ (which holds under µ-strong convexity, L-smoothness and the step-size condition $\eta \leq 1 / L )$ . Substituting our computed norm alongside $\sigma ^ { 2 } = \left[ 2 s _ { p } ^ { 2 } ( \Delta _ { 2 } \dot { f } ) ^ { 2 } \ln ( 1 . 2 5 s _ { p } / \delta ) \right] / \epsilon ^ { 2 }$ into the inequality yields:

$$
{ \mathbb { P } } \left\{ \left. e _ { s _ { p } } \right. \geq E _ { G } \right\} \leq 2 \exp \left( - \frac { 3 c E _ { G } ^ { 2 } \mu ( 2 - \eta \mu ) \epsilon ^ { 2 } } { 1 6 \eta s _ { p } ^ { 2 } ( \Delta _ { 2 } f ) ^ { 2 } \ln ( 1 . 2 5 s _ { p } / \delta ) } \right) .
$$

To follow the same analysis as before, we express the error $E _ { G }$ in terms of a user-defined confidence α. Setting the probability bound to α and solving for $E _ { G }$ gives the (single-regime) worstcase error

$$
E _ { G } = \sqrt { \frac { 1 6 \eta s _ { p } ^ { 2 } ( \Delta _ { 2 } f ) ^ { 2 } \ln \left( \frac { 1 . 2 5 s _ { p } } { \delta } \right) \ln \left( \frac { 2 } { \alpha } \right) } { 3 c \mu ( 2 - \eta \mu ) \epsilon ^ { 2 } } } .
$$

The error of AGS in this setting stems from a single addition of Laplace noise calibrated to the β-smooth sensitivity (with $\beta < \epsilon / ( 2 \ln ( 2 / \delta ) )$ ensuring approximate $( \epsilon , \delta ) \mathrm { - D P ) }$ , exactly as in the proof of Thm. IV.4:

$$
E _ { L } = \frac { 2 \mathrm { S S } ^ { \beta } } { \epsilon } \ln \left( \frac { 1 } { \alpha } \right) .
$$

Enforcing $E _ { L } < E _ { G }$ and solving for the smooth sensitivity yields the required conditions under which Laplace AGS dominates Gaussian DP-SGD:

$$
\mathrm { S S } ^ { \beta } ( f _ { \theta } , D ) < \frac { 1 } { 2 \ln \left( \frac { 1 } { \alpha } \right) } \sqrt { \frac { 1 6 \eta s _ { p } ^ { 2 } ( \Delta _ { 2 } f ) ^ { 2 } \ln \left( \frac { 1 . 2 5 s _ { p } } { \delta } \right) \ln \left( \frac { 2 } { \alpha } \right) } { 3 c \mu ( 2 - \eta \mu ) } } ,
$$

## APPENDIX C AGT-R ABLATIONS

In this section we validate the AGT-R certificate on its own. We first extend the analysis of the feasibility conditions of Figure 1, then show the certificates behind the two ablations of the main text, the concretization frequency on linear regression and the batch size on California Housing. We then present a study that ablates the number of shards in PATE-R and PATE-AGT-R and reveals its effect on performance, and lastly repeat the comparison of Figure 2a on the raw error scale together with the global-sensitivity baselines, thus tying directly into our main experimental results.

![](images/55d73a8a2b28b74e821663fb546813a7f4aef8d560aa6d4db08af58259b9ab7c.jpg)

![](images/ee185783b73500f87961922dd24e7af0129c2af230c4eae6bf3d630df4da70bc.jpg)

![](images/a87f199bb5c1f572d59291246782ceddf26cb6f3a92b44de76832cd016247d7f.jpg)

![](images/ccb71a38e665763abf0772bf880b1317a607fdfd84a32e10d2ea6dad1e2e6999.jpg)  
-- ε = 0.1 −- ε = 0.2 −−= ε = 0.5 −= ε = 1 −- ε = 2 −= ε = 5 −0− ∈ = 10  noise-free (PATE mean)  
Figure 6: Shard ablation on the linear regression task with $4 \times 1 0 ^ { 3 }$ training points, under pure DP. MAE against the number of shards T at fixed dataset size, one line per budget $\epsilon \in [ 0 . 1 , 1 0 ]$ , for PATE-R (Laplace noise on global sensitivity), PATE-AGT-R (Cauchy noise on the shard-wise smooth sensitivity, $\beta = \epsilon / 2 )$ and AGT-R on a single shard of $N / T$ points with the amplified budget $\epsilon _ { b } ;$ dashed: the noise-free PATE mean.

## A. Condition Simulations: Further Analysis

We complement the discussion of Figure 1 in the main text with a finer-grained analysis of the simulated conditions, starting with the pure $( \epsilon , 0 ) { \tt - D P }$ column. Additionally, we can observe that with high confidence $( \mathrm { i . e . } > 9 0 \% )$ ), having a ratio $\rho \approx 0 . 0 1$ , we can consistently achieve utility multipliers greater than 1. A trend that additionally arises is that as c increases, the gaps between consecutive c decrease, meaning a smaller ratio consistently achieves $c \gg 1$ better utility.

We now turn to the finer details of the approximate $( \epsilon , \delta ) \cdot$ DP column, where the admissible ratios peak at $\rho \approx 1 . 9$ for $c = 1$ . These largest values, however, are attained at 1 − $\alpha \approx 0 . 8$ , an operating point of little practical interest; at the more desirable $1 - \alpha = 0 . 9 8$ the admissible ratio falls to 1.12–1.44, still favourable but far less generous. A further important observation is that our analysis imposes conditions on $\beta$ and this constraint is stricter than in the pure-DP case: whereas Cauchy requires only $\beta < \epsilon / 6$ , the Laplace mechanism enforces $\beta \le \frac { \epsilon } { 2 \ln ( 2 / \delta ) }$ , roughly four times tighter at $\delta = 1 0 ^ { - 5 }$ Additionally, as detailed in §IV-A, AGT-R produces strict upperbounds on local sensitivity and is strictly over-approximate. Both these details force a practically larger ${ \mathrm { \dot { S } S } } ^ { \beta }$ , or equivalently a larger ϵ to remain valid, so the displayed curves are optimistic ceilings only attainable theoretically. Lastly, the diminishing returns in c reappear: $c = 2$ and $c = 3$ lie below unity across the entire window, peaking at $\rho \approx 0 . 9 6$ and 0.64.

## B. Certificates of the AGT-R Ablations

Figure 7 visualizes the $S S ^ { \beta }$ bounding algorithm behind the ablations of Figure 2b, averaged over test inputs. The dashed line (left y-axis) represents the certified bound on the local sensitivity ${ \bar { A } } ( x , k )$ as the number of substitutions k increases, while the solid line (right y-axis) tracks the induced upper bound on the smooth sensitivity, $\bar { A } ( x , k ) e ^ { - \beta k } \mathrm { a t } \beta = 1 / 2$ (the $\beta$ of the pure-DP release at $\epsilon = 1 )$ , with stars marking its maximum $S S ^ { \beta }$

Figure 7: Cross-radius certificates of the AGT-R ablations (left: concretization frequency on linear regression; right: batch size on California Housing). Dashed, left axis: the certified local-sensitivity bound ${ \bar { A } } ( x , k )$ averaged over test inputs; solid, right axis: the smooth-sensitivity bound at $\beta = 1 / 2 .$ , with its maximum starred; dotted: the global-sensitivity ceiling.

Beyond the largest computed k the certificate falls back to the global sensitivity (dotted), which is the clipping ceiling of the trajectory (0.64) for linear regression and the output range (10 standard deviations) for California Housing; the released prediction is clipped to this range. The staircases show what the trade-off panels only show indirectly: a looser certificate (IBP, small $F ;$ small BS) saturates at the ceiling within a few substitutions, so the exponential discount has little to act on and $S S ^ { \beta }$ peaks at $k = 1$ close to the ceiling, whereas a tight certificate (large $F ;$ full batch) stays well below the ceiling for all $k \leq 2 0$ and $S S ^ { \beta }$ peaks at $k = 2$ an order of magnitude lower. This is the mechanism behind the $0 . 2 5  0 . 0 1 5$ and the $6  0 . 0 8$ reductions in $S S ^ { \beta }$ reported in the main text.

## C. Number of shards (T)

This ablation uses the Linear dataset and exactly the same model and hyperparameters as in §VI-A, with the exception of the number of training points, which is now $4 \times 1 0 ^ { 3 }$ instead of $4 \times 1 0 ^ { 4 }$ . Figure 6 shows the MAE of PATE-R, PATE-AGT-R and AGT-R, the latter being restricted to one shard for ϵ ranging from 0.1 to 10, across an increasing number of shards, all while the dataset size is fixed.

![](images/22d9f0b070af346771fa40c94e497e0043095dda26b70214b63018e50ad16407.jpg)  
Figure 8: The comparison of Figure 2a on the raw scale with global sensitivity (grey, dotted). MAE under pure DP and test MSE under approximate DP $( \delta = 1 0 ^ { - 5 } )$ , each point the mean over 200 noise draws.

## D. Private Prediction Baselines with Global Sensitivity

Figure 8 repeats Figure 2a on the raw error scale and with the global-sensitivity arm (Laplace in pure DP, Gaussian in approximate DP, both on the full data). On the excess scale of the main text this arm would sit one to five orders of magnitude above every other, which is why it is shown here. Accordingly, the y-axes report the full error of the private release, MAE under pure DP and MSE under approximate DP, rather than the excess over the non-private prediction; the non-private floor (0.079 MAE and 0.0098 MSE on linear regression, 0.72 MAE and 0.94 MSE on California Housing) is where the AGT-based arms flatten out at large ϵ, and subtracting it recovers Figure 2a.

## APPENDIX D AGS ABLATIONS

## A. Sampling Ablation

Intuitively, in standard PATE, MAE decreases as long as a single shard holds enough data for significant overlap without vote dilution, which also improves privacy by amplification. This is confirmed experimentally when looking at the leftmost subfigure and happens at $T = 5 1 2 ,$ , i.e. ≈ 8 points per shard, after which point the MAE starts increasing. The story changes somewhat in the case of PATE-AGT-R: the error starts to get larger earlier and the threshold at which utility starts decreasing is highly dependant on ϵ. For example, MAE peaks at $T = 8$ for $\epsilon = 0 . 2 ;$ in contrast the same peak is observed at $T = 6 4$ when $\epsilon \ : = \ : 2$ . This indicates that under a certain specified privacy budget, noise calibrated to the parameter envelopes starts significantly affecting predictions. Whereas this budget is quickly reached as a function of the number of shards when the total privacy budget is already low, when this budget increases, the behaviour is observed at larger T. In the case of the first, the utility degradation is significant, while in the case of the latter it is not as visible due to the effect of mean aggregation. The rightmost panel of the figure can easily be interpreted: at 128 shards, meaning ≈ 31 datapoints per shard, AGT-R doesn’t have enough data to compute tight parameter envelopes, which is why the test MAE skyrockets, regardless of total privacy budgets. Lower total privacy budgets start, expectedly, at a higher MAE due to noise causing utility degradation.

![](images/bb093a89542746731f7b614ab2ee22fd2d6b9a72e9b6017bceda43ee67c519aa.jpg)

![](images/3a4f52fa4716f6b1573f131d1772171c0904c854361d1f23311230d3c07ff971.jpg)

![](images/10319f169afe2c0b87af9b7e8a9e43e2cf29545e8c3c7ccfe8cfd88854430f3c.jpg)  
Figure 9: Ablation of the number of releases (i.e., sampling steps) at different, fixed total privacy budgets on the UCI Census Income Dataset [53]. Left panel: test accuracy against the number of releases n under $( \epsilon , 1 0 ^ { - 5 } )$ -DP, with the perrelease budget set by optimal composition (filled) or basic composition (hollow), against the non-private accuracy (band) and the majority-class rate (dashed). Middle: the budget ϵ<sub>0</sub> each release receives under optimal composition (solid) against the basic split $\epsilon / n$ (dashed). Right: the ratio of the two, i.e. the gain from accounting.

As previously mentioned in the beginning of §VI-B, we perform an ablation on the number of releases or ‘’sampling steps’‘ (i.e., the addition of noise and collapse of the parameter envelope) to observe the effect of composition on utility and privacy. We employ the UCI Census Income Dataset [53], which has been used extensively to validate algorithmic techniques in DP literature [6], [54], [55].

The dataset consists of 30,222 private records. We train a logistic head on the first d = 16 PCA components with 300 steps of full-batch gradient descent split evenly into $n \in$ $\{ 2 ^ { 0 } , \ldots , 2 ^ { 8 } \}$ releases, each certified from the previous noisy release. We perform each ablation for a fixed total (approximate DP) privacy budget $\epsilon \in \{ 5 , 1 0 , 2 0 , 5 0 , 1 0 0 , 1 5 0 \}$ $\delta = 1 0 ^ { - 5 }$ split across releases either evenly $( \epsilon / n$ each) or by optimal composition [56].

Figure 9 shows the results. In the left panel, as the privacy budget increases, it takes more sampling steps for the test accuracy to decrease from ≈ 82% to below ≈ 78%, which happens at 4 releases for ϵ = 10, at 16 releases for $\epsilon = 5 0$ and at 64 releases for $\epsilon = 1 5 0 $ , where 32 releases still reach 80.9%. A larger total budget thus buys more free releases: the number of releases within one point of the non-private accuracy doubles from 8 at ϵ = 50 to 16 at ϵ = 150. By optimal composition, we only start gaining a larger ϵ per step (i.e., less noise injected) at the same cost δ when $n \geq 3 2$ (number of releases) for $\epsilon \leq 2 0$ and when $n \geq 2 ^ { 7 }$ for ϵ ≥ 100, in which case the gains (up to 3.5× at $\epsilon = 5 , n = 2 ^ { 8 } )$ can be observed in the rightmost panel and the corresponding optimal ϵ in the middle panel. $\mathrm { A t ~ } \epsilon \in \{ 1 0 0 , 1 5 0 \}$ , this first shows in the accuracy (75.0% vs. 74.8% and 75.3% vs. 74.8% at $n = 2 ^ { 7 } )$ , although both remain near the majority-class rate (74.6%).

## A. AGT-R

<table><tr><td></td><td>Linear</td><td>California</td></tr><tr><td>Data</td><td></td><td></td></tr><tr><td>Train / test points</td><td>40,000 / 2,000</td><td> $1 6 , 5 1 2 \mid 4 , 1 2 8$ </td></tr><tr><td>Output range (GS)</td><td>[−6, 6]</td><td>10 std. units</td></tr><tr><td>Baselines (Fig. 2a)</td><td></td><td></td></tr><tr><td>Model</td><td> $y = w x + b$ </td><td> $8 – 6 4 – 1 \ \mathrm { R e L U }$ </td></tr><tr><td>Steps / lr / γ</td><td> $2 0 / 0 . 3 / 1$ </td><td> $3 3 0 \mathrm { ~ / ~ } 0 . 0 1 \mathrm { ~ / ~ } 0 . 1$ </td></tr><tr><td>Shards T</td><td> ${ 2 ^ { 0 } , \dots , 2 ^ { 1 0 } }$ </td><td> ${ 2 ^ { 0 } , \ldots , 2 ^ { 6 } }$ </td></tr><tr><td>Certificate</td><td>exact per step</td><td> $\mathrm { A G T } \ [ 1 3 ]$ </td></tr><tr><td>Substitutions k</td><td>≤ min(1024, |shard|)</td><td>≤ 28</td></tr><tr><td>Ablations  $( F i g . ~ 2 b )$ </td><td></td><td></td></tr><tr><td>Steps  $/ \mathrm { ~ l r ~ } / \gamma$ </td><td> $1 6 \ ( \mathrm { b a t c h \ 5 } ) \ / \ 0 . 1 \ / \ 0 . 1$ </td><td> $3 3 0 \mathrm { ~ / ~ } 0 . 0 1 \mathrm { ~ / ~ } 0 . 1$ </td></tr><tr><td>Ablated</td><td> $F \in \{ \mathrm { I B P } , 1 , \dotsc , 1 6 \}$ </td><td> $\mathtt { B S } \in \{ 2 5 0 , \dots , 1 6 , 5 1 2 \}$ </td></tr><tr><td>Release (both datasets) Pure DP:</td><td></td><td></td></tr><tr><td> $\operatorname { C a u c h y } ( \operatorname { S S } ^ { \beta } / ( \epsilon - \beta ) ) , \beta = \epsilon / 2 ; \operatorname { M A E } \mathrm { ~ , ~ }$  Approx. DP: Laplace(2 SSβ  $\epsilon \in \{ 0 . 1 , 0 . 2 , 0 . 5 , 1 , 2 , 5 , 1 0 \} ; 2 0 0$  noise draws per point</td><td> $/ \epsilon ) , \beta = \epsilon / ( 2 \ln ( 2 / \delta ) )$ </td><td>of the clipped release , δ = 10−5; MSE</td></tr></table>

Table I: AGT-R settings for Figure 2. Training is fullbatch (full-shard) clipped gradient descent. PATE-R adds Laplace(GS/(T ϵ)) (Gaussian in approximate DP) to the shard mean; PATE-AGT-R calibrates to $\textstyle \frac { 1 } { T } \operatorname* { m a x } _ { i } \operatorname { S S } _ { i } ^ { \beta }$

## B. AGS

APPENDIX E  
HYPERPARAMETERS
<table><tr><td></td><td>Blobs</td><td>MNIST 0-4</td><td>IMDB</td></tr><tr><td>Data</td><td></td><td></td><td></td></tr><tr><td>Private / public / test Features</td><td> $2 0 0 \mid - \mid 4 \mathrm { k }$  R8, 2 Gaussians</td><td>28.6k / 2k / 5.1k PCA-8 (pixels)</td><td>20k / 5k /  25k PCA-8 (mpnet)</td></tr><tr><td>Model and training</td><td></td><td></td><td></td></tr><tr><td>Head (P)</td><td>linear (9)</td><td>logistic (45)</td><td>logistic (18)</td></tr><tr><td>Training</td><td>clipped SGD</td><td>contractive clipped GD, full batch</td><td></td></tr><tr><td>Steps / lr</td><td> $2 0 0 \ ( \mathrm { b } \ 2 0 ) \ / \ 0 . 5$ </td><td>1,000 /  2</td><td>1,000 /  2</td></tr><tr><td>Clip γ / contr. λ</td><td>0.05 / –</td><td>0.02 / 0.6</td><td>0.01 / 0.6</td></tr><tr><td>Certificate</td><td></td><td></td><td></td></tr><tr><td>Bound propagation</td><td>MILP,</td><td>CROWN</td><td>CROWN</td></tr><tr><td></td><td> $F \in \{ 1 , 2 , 5 , 1 0 \}$ </td><td></td><td></td></tr><tr><td>Substitutions k</td><td>{1, 2, 4, 8}</td><td> $1 , \ldots , 4 0 9 6 , N$ </td><td> $1 , \ldots , 5 1 2 , N$ </td></tr><tr><td>Release (Corollary V.1)</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Pure DP: independent</td><td> $\mathrm { C a u c h y } ( 2 u / \epsilon ) , \beta = \epsilon / ( 2 P )$ </td><td></td><td></td></tr><tr><td>Approx. DP: independent Laplace</td><td> $( 2 u / \epsilon )$ </td><td>, Gamma-tail  $\beta , \delta = 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Budgets € / draws Allocation</td><td>1-100 / 101 fixed  $( \ell _ { 1 } )$ </td><td>1-100 / 41  $\ell _ { 1 }$ </td><td>1-100 / 41 or anisotropic, chosen on public set</td></tr></table>

Table II: AGS settings for Figures 4 and 3. P is the number of released parameters and u the smooth sensitivity of the $\ell _ { 1 }$ cross-radius certificate. Public sets: MNIST 1,000 PATE pool and 1,000 selection; IMDB 2,500 and 2,500. Baselines are tuned on the same public sets.

<table><tr><td colspan="2">SST-2 (hybrid)</td></tr><tr><td>Data</td><td></td></tr><tr><td>Private / public / test</td><td>62,349 / 872 (GLUE validation) / 5,000</td></tr><tr><td>Features Encoder (DP-LoRA,</td><td>PCA-8 of the DP-LoRA all-mpnet-base-v2 embedding</td></tr><tr><td colspan="2"> $( \epsilon _ { 1 } , 5 \cdot 1 0 ^ { - 6 } ) – D P )$ </td></tr><tr><td>Adapter</td><td>rank 16 on the q, v projections of every layer</td></tr><tr><td>DP-SGD (Opacus)</td><td>batch 1,024, 3 epochs, clip  $1 , \mathrm { l r ~ } 1 0 ^ { - 3 } ,$  PRV accountant</td></tr><tr><td colspan="2">Head (AGS,  $( \epsilon _ { 2 } , 5 \cdot 1 0 ^ { - 6 } ) – D P )$ </td></tr><tr><td>Head (P) / training</td><td>logistic (18) / contractive clipped GD, full batch</td></tr><tr><td>Steps  $/ \operatorname { l r } / \gamma / \lambda$ </td><td>150 / 2 / 0.05 / 0.3</td></tr><tr><td>Certificate</td><td> $\mathrm { C R O W N } ; k = 1 , \dots , 5 1 2 , N$ </td></tr><tr><td>Release</td><td>as Table II; 41 draws</td></tr><tr><td colspan="2">Pipeline</td></tr><tr><td>Total  $\epsilon = \epsilon _ { 1 } + \epsilon _ { 2 }$ </td><td> $0 . 5 , 1 , 2 , 3 , 5 , 8 \mathrm { ~ a t ~ } \delta = 1 0 ^ { - 5 }$ </td></tr><tr><td>Chosen on public set</td><td>the split  $( \epsilon _ { 1 } , \epsilon _ { 2 } ) ,$  the allocation, the threshold</td></tr></table>

Table III: Settings of the SST-2 hybrid (Figure 5). The endto-end arm is the DP-LoRA model scored with its own head; the frozen-encoder arms use the same head on the un-adapted embedding.