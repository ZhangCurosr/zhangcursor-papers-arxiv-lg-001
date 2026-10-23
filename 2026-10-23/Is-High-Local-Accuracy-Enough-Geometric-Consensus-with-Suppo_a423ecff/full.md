# Is High Local Accuracy Enough? Geometric Consensus with Support-Adaptive Statistical Personalization for Federated Learning

Junyu Lin , Bosong Qi , Pengcheng Shi , and Jiayuan Li

Abstract—Real-world deployment of personalized federated learning (PFL) requires adapting to heterogeneous clients while preserving recognition performance beyond local distributions. However, high local accuracy can conceal failures on locally rare or absent classes. Effective personalization should therefore strengthen global recognition while raising the personalization floor. We propose an evaluation protocol that complements local accuracy with global, rare-class, and unseen-class evaluations to reveal generalization obscured by local scores, and assesses personalization floors among the worst-performing clients to expose weaknesses hidden by aggregate performance. To meet these deployment challenges, we propose FedGRC, a unified PFL framework integrating generalization and personalization through (i) Consensus-Guided Representation Learning (CGRL), which learns discriminative representations through shared directional and magnitude guidance; (ii) Multilayer Metric Synthesis (MMS), which constructs a shared covariance metric from these multilayer representations for statistical classification; and (iii) Support-Adaptive Shrinkage Personalization (SASP), which adapts class means within this metric by shrinking local estimates toward shared means according to local sample support, enabling adaptation while retaining shared knowledge. Extensive experiments, including benchmarking against 21 methods across 5 heterogeneity settings on 3 datasets and comprehensive ablation studies, validate the effectiveness of FedGRC. On CIFAR-10 and CIFAR-100, FedGRC surpasses state-of-the-art (SOTA) baselines in global recognition and personalization floors by at least 2.17% and 2.13%, respectively. On the real-world heterogeneous FedRS dataset, it outperforms the respective best baselines by 2.15%, 3.30%, and 4.61% in local adaptation, global recognition, and the personalization floor, demonstrating its practical deployment value.

Index Terms—Personalized federated learning; Statistical heterogeneity; Real-world generalization; Representation learning.

## 1 INTRODUCTION

P <sup>ERSONALIZED</sup> <sup>federated</sup> <sup>learning</sup> <sup>(PFL)</sup> <sup>adapts</sup> <sup>client</sup> models to local data using knowledge shared across the federation, without centralizing raw training data [1], [2], [3]. This combination of knowledge sharing and local adaptation is relevant to healthcare [4], autonomous driving [5], and remote sensing, where data distributions and acquisition conditions vary across clients. Remote sensing provides a representative example: differences in sensors and geographic coverage produce heterogeneous image appearances and skewed local label distributions, leaving some classes well represented at a client and others rare or absent [6], [7]. This poses a core deployment challenge for heterogeneous PFL: combining all-class recognition with effective local adaptation.

Several PFL methods, however, prioritize local personalization without adequately preserving generalization beyond the local distribution [8]. DBE [9] and GPFL [10] identify a loss of general feature information when representations are trained to serve personalized classifiers, while other studies examine the tension between personalization and generalization under changing test distributions [11], [12]. Motley [13] provides a concrete example: additional local fine-tuning can improve accuracy on local data while reducing accuracy on data sampled from other clients. These observations raise the question: Is high local accuracy sufficient to assess a personalized model’s recognition ability?

The results in Fig. 1 reveal a counterintuitive pattern for pFedFDA [14], FedLog [15], and pFedGM [16]: as label heterogeneity increases from Dir(0.3) to Dir(0.1), their local accuracy rises while their global accuracy falls. Here, global evaluation assesses the same client predictors on a classbalanced test set covering all federation-known classes. Local evaluation assigns most weight to frequent classes, while locally rare or absent classes receive little or none. A high local score can therefore conceal weaknesses in recognition across the full label space [17].

An informative evaluation of PFL under label heterogeneity should therefore extend beyond local accuracy. It should assess (i) adaptation to each client’s data distribution; (ii) recognition across the full label space; (iii) recognition of locally rare or absent classes; and (iv) weaknesses in individual clients that aggregate results may conceal [18]. HtFLlib [19] likewise evaluates client models on a global test set to assess generalization beyond local data. Our objective is thus to improve recognition across all federation-known classes and strengthen poorly performing clients’ weakest assessed abilities while retaining effective local adaptation.

Meeting this objective requires addressing three interdependent design challenges. First, representation learning must preserve shared discriminative structure under heterogeneous local updates; the feature anchors in FedFA [20] and common class geometry in FedETF [21] offer ways to guide this process. Second, useful representations alone do not eliminate classifier bias, as demonstrated by CCVR [22]; the classifier must draw on distributed class information without relying excessively on local label frequencies. Statistical personalization in pFedFDA [14] and pFedGM [16] provides a basis for combining shared and local feature-distribution information. Third, adaptation must account for local class support: well-observed classes support stronger local adaptation, whereas locally rare or absent classes require greater reliance on shared knowledge. These challenges are interdependent: the representation shapes the class statistics used by the classifier, while local sample support governs how strongly those statistics should be adapted.

![](images/39d0c6f3e9bd9ba7caf6d9fde53eb9518247262167631727d249f1701fe8c9f8.jpg)  
Fig. 1: Local personalization and global generalization on CIFAR-10 and CIFAR-100. Colors identify methods: • pFedFDA [14], • FedLog [15], • pFedGM [16], and • our FedGRC. Dir(0.3) and Dir(0.1) denote Dirichlet label partitions with strong and extreme heterogeneity, respectively. Local denotes accuracy on local test data; global denotes accuracy on a class-balanced test set covering all classes.

We propose FedGRC, a unified PFL framework built on Geometric Representation Consensus. FedGRC keeps the learned representation and covariance metric shared during personalization and adapts class means by balancing local and shared estimates according to local sample counts. The counts determine the influence of local estimates, while equal class priors avoid adding a prediction preference for locally frequent classes. Our main contributions are:

Comprehensive PFL Evaluation beyond Local Accuracy. We introduce a protocol that jointly assesses local adaptation, generalization across the full label space, and recognition of locally rare and absent classes. It also measures a recognition floor by taking each client’s lowest accuracy across local adaptation and rare- and unseen-class recognition, then summarizing the lowest-performing clients. This reveals weaknesses hidden by aggregate scores.

Unified Personalization and Generalization. We develop FedGRC to connect geometric representation learning with statistical personalization for local adaptation and recognition across the full label space. Shared directional references and class- and layer-specific magnitude references guide representation learning. Multilayer class statistics then yield a shared covariance metric, and support-dependent shrinkage adapts class means within this geometry.

Systematic Benchmarking and Validation. We reproduce and benchmark 21 external methods: 15 complete federated learning (FL) methods and 6 statistical classifiers, with the latter evaluated on 2 backbone types. The evaluation spans 5 heterogeneity settings across 3 datasets with different labelspace sizes, client counts, and degrees of heterogeneity, including real-world remote sensing data from FedRS [7]. Comparative experiments and ablations establish the contributions of the framework and its components. On FedRS, FedGRC gains 3.30% in global accuracy and 4.61% in the worst-performing clients’ recognition floor over the respective best baselines, with higher local personalization accuracy.

Open-Source Resources. The benchmark and method implementations, design details, complete evaluation protocol, and all experimental configurations and parameter settings are publicly available at https://github.com/JL181/ FedGRC-Benchmark.

## 2 RELATED WORK

## 2.1 Balancing Personalization and Generalization in Federated Learning

PFL methods combine shared knowledge with local adaptation through parameter regularization, representation sharing, or predictor combination [3], [6]. Ditto [23] regularizes personalized models toward a shared model to improve robustness and fairness, but requires additional model updates for personalization. FedRep [1] separates shared representations from local heads; however, these heads still guide representation learning, allowing local label imbalance to influence the shared features. FedROD [2] combines a classbalanced generic classifier with a personalized residual classifier on a shared feature extractor. This additive combination supports both generic recognition and local adaptation, but does not adjust the classifiers’ relative contributions as the test distribution changes.

Test-time weighting and predictor selection offer alternatives to fixed predictor combinations. FedTHE [11] optimizes a mixture weight using entropy and feature alignment informed by test history. BTFL [12] instead infers weights from current and historical observations through Bayesian modeling, reducing iterative adaptation while relying on approximate feature likelihoods. HPFL [8] uses markers to select among shared personalized plug-ins, expanding the available predictor pool at the cost of storage and matching operations. These mechanisms improve deployment flexibility, but their decisions depend on cues extracted from test observations. When data sources or class distributions change, confidence and feature similarity may not reliably identify the most suitable predictor or combination. Fed-GRC completes client adaptation before deployment and therefore requires neither test-time predictor selection nor mixture-weight estimation.

Training or selecting personalized predictors does not by itself establish whether each deployed model retains recognition across all federation-known classes. Motley [13] shows that local fine-tuning can improve local accuracy while degrading performance on other clients’ data. Other studies evaluate personalized models on both local and global test sets [19], [24], [25], [26], while broader benchmarks and fairness studies examine generalization and disparities across clients [18], [27], [28], [29]. Aggregate performance can nevertheless conceal poor recognition of locally rare or absent classes, as well as weaknesses concentrated in particular clients. Our evaluation protocol assesses the same deployed client models on both local and all-class test distributions, explicitly evaluating locally rare and unseen classes. Our protocol further measures each client’s weakest assessed ability and summarizes the lowest of these clientlevel scores, so that stronger abilities or performance at other clients do not mask these weaknesses.

## 2.2 Learning and Preserving Shared Representations under Data Heterogeneity

Shared training signals differ in both the information they convey and the aspects of learning they constrain. At the logit level, FedLC [30] uses class frequencies to calibrate predictions under label imbalance, but this correction does not by itself ensure that local updates preserve knowledge of locally absent classes. Methods targeting locally absent classes address this retention problem directly [31], [32]. FedVLS combines global-model distillation with logit suppression against minority-to-majority errors; distillation uses only teacher predictions on locally available inputs.

At the representation level, methods differ in the granularity of their shared references and the feature properties they constrain. DBE [9] corrects client-specific bias through local bias memory and global mean regularization, but a single client-wide mean cannot fully characterize class-specific differences across clients. GPFL [10] combines angular guidance and Euclidean alignment to class references through additional feature transformations and auxiliary objectives along a personalized training path. FedProto [33] exchanges class prototypes for feature regularization, while FedFA [20] uses Euclidean anchors for feature alignment and classifier calibration. Class-reference methods align feature centers or directions across clients [34], [35], [36]. Feature magnitudes have also been shaped through learned transformations [37], class-average norm regularization [38], and layerwise squared activation-norm penalties [39]. These representation constraints do not by themselves determine how a classifier should use within-class variation and feature correlations. FedGRC uses final-layer directional guidance and shared magnitude references for each class and selected layer. Its multilayer statistical classifier uses common layer scales that preserve within-layer magnitude differences.

Fixed classifiers provide another way to establish shared feature geometry. FedETF [21] uses an equiangular tightframe (ETF) classifier to impose common class geometry and prevent classifier drift. Within this line of work, FedDr+ [40] identifies missing-class forgetting under ordinary dot-regression and adds global feature distillation to improve retention, at the cost of computing teacher features. Separately, FedYoYo [41] combines self-distillation across weakly and strongly augmented views with distributionaware logit adjustment to strengthen minority-class learning and generalization. These advances improve shared representations, but preserving their recognition ability during local adaptation remains a separate design problem. In our personalization experiments, local fine-tuning of FedETF improves local accuracy but substantially reduces all-class and locally unseen-class recognition (Section 5.3). FedGRC links representation learning to personalization through statistical classification based on shared multilayer features. For FedGRC, the remaining design choices concern which classifier statistics to keep shared and how strongly each local class estimate should influence personalization.

## 2.3 Federated Statistical Classifier Construction and Local Adaptation

Statistical classifiers construct predictors from distributed feature summaries and have been used in one-shot and communication-efficient FL. Their designs differ in how they construct decision rules, represent within-class variation, and estimate feature correlations. CCVR [22] calibrates a classifier using sampled virtual features and server-side optimization, while leaving the underlying representation fixed. FedNCM [42] provides a simple mean-based readout, but does not account for class spread or feature correlations when comparing samples with class centers. Fed3R [43] uses second-order feature statistics to construct a closedform ridge classifier whose solution is independent of how fixed features are partitioned across clients. Exchanging second-order matrices increases communication cost relative to mean-only summaries and adds matrix-processing overhead. FedCGS [44] constructs a Gaussian classifier from a full covariance centered at the global feature mean, without synthetic sampling or iterative head training. GH-LDA [45] instead estimates shared within-class covariance by pooling residuals around class means. Covariance estimation introduces a further distinction: FedCOF [46] reduces communication by estimating covariance from client means, but class-conditional shifts can affect the estimate and the resulting decision boundaries. These methods differ in both the decision rule derived from shared statistics and the variation represented by those statistics. In continual learning, LayUP [47] constructs a Gram-matrix-based ridge readout, whereas FedGRC separates shared within-class covariance estimation from local class-mean adaptation. A related design choice is how to scale multilayer features. For OOD detection, MM++ [48] normalizes each sample’s feature vector at each layer to unit length before estimating a shared within-class covariance from the concatenated features. FedGRC instead applies a common scale to all samples at each layer, preserving within-layer magnitude differences for its statistical classifier.

Beyond constructing a shared classifier, statistical personalization must determine which quantities to adapt and how local sample support should influence that adaptation. FedCGS [44] additionally supports personalization through prototype-regularized local fine-tuning. FedLog [15] personalizes local feature extractors and learns a shared head from sufficient statistics through iterative server-side maximum a posteriori (MAP) optimization. Its adaptation therefore operates on local representations, whereas FedGRC keeps the representation shared and adapts class means.

pFedFDA [14] guides representations with a global generative classifier and interpolates local and shared means and covariances. Its interpolation coefficients are selected per client using local validation, rather than explicitly determined for each class from its sample count. Its local class priors favor locally frequent classes, whereas FedGRC uses equal class priors to avoid adding this frequency preference to the prediction rule. pFedGM [16] combines inter-class and intra-class representation objectives with count-informed distribution fusion, followed by local optimization of class means, scales, and biases. This flexibility requires estimating several quantities from potentially sparse class observations and can increase the risk of local overfitting. TOFA and SPEAR also use class counts to regulate mean shrinkage. TOFA [49] uses local and global class counts to shrink local means toward global estimates while estimating a clientspecific covariance. SPEAR [50] pools local means shrunk toward other clients’ averages into a shared classifier.

FedGRC combines a shared multilayer covariance metric with client-specific class means adapted according to local class counts, using a reference strength fixed across clients and classes. By keeping the representation and covariance metric shared, FedGRC limits personalization to class means and avoids estimating a separate local metric. FedGRC retains shared means for locally absent classes, keeps estimates for sparsely observed classes close to the shared means, and permits stronger adaptation for wellobserved classes. Local counts govern mean adaptation; equal priors support all-class recognition without favoring locally frequent classes.

## 3 PRELIMINARIES

## 3.1 Personalized Federated Learning

Consider K clients and a common label space $\begin{array} { r l } { ) ^ { } } & { { } = } \end{array}$ $\{ 1 , \ldots , C \}$ . Client k holds $\mathcal { D } ^ { ( k ) } = \{ ( x _ { i } ^ { ( k ) } , y _ { i } ^ { ( k ) } ) \} _ { i = 1 } ^ { N ^ { ( k ) } }$ , with $\begin{array} { r } { \dot { N } ~ = ~ \sum _ { k } N ^ { ( k ) } } \end{array}$ . Parenthesized superscripts (k) identify clients; $i , \ c , \ \ell ,$ and t index samples, classes, layers, and communication rounds, respectively. A notation reference is provided in Appendix C. For a model $g ( \cdot ; \pmb \theta )$ and classification loss $\ell ,$ the local empirical loss is defined as

$$
\mathcal { L } ^ { ( k ) } ( \pmb { \theta } ) = \frac { 1 } { N ^ { ( k ) } } \sum _ { i = 1 } ^ { N ^ { ( k ) } } \ell \Big ( g ( x _ { i } ^ { ( k ) } ; \pmb { \theta } ) , y _ { i } ^ { ( k ) } \Big ) .\tag{1}
$$

Standard FL coordinates local updates to learn a shared model without pooling raw data, commonly by minimizing the sample-weighted objective

$$
\operatorname* { m i n } _ { \pmb { \theta } } \sum _ { k = 1 } ^ { K } \omega ^ { ( k ) } \mathcal { L } ^ { ( k ) } ( \pmb { \theta } ) , \omega ^ { ( k ) } = \frac { N ^ { ( k ) } } { N } .\tag{2}
$$

When client distributions differ, a single shared predictor may not fit every client well. PFL therefore allows client-specific models while retaining collaboration through shared parameters, representations, or regularization [3]. A common representation-based formulation shares a feature extractor $h _ { \phi }$ and assigns each client a head $q _ { \psi ^ { ( k ) } } \left[ 1 \right]$ . With $\pmb \theta = ( \phi , \psi )$ and $g ( x ; \dot { \phi } , \psi ) = q _ { \psi } ( h _ { \phi } ( x ) )$ , the objective is

$$
\operatorname* { m i n } _ { \phi , \{ \psi ^ { ( k ) } \} _ { k = 1 } ^ { K } } \sum _ { k = 1 } ^ { K } \omega ^ { ( k ) } \mathcal { L } ^ { ( k ) } ( \phi , \psi ^ { ( k ) } ) ,\tag{3}
$$

where $\mathcal { L } ^ { ( k ) } ( \phi , \psi )$ is the loss in Eq. (1), expressed in terms of the feature-extractor and head parameters.

We denote the deployed label predictor by $f ^ { ( k ) } ( x ) \in \mathcal { V }$ and assess both local adaptation and broader recognition using this same predictor. Shared representations provide common knowledge, but the reliability of a locally fitted head depends on the sample support of each class. Constructing a head from class-level statistics allows shared and local evidence to be combined separately for each class.

## 3.2 Statistical Classification

Statistical classifiers construct decision rules from featuredistribution summaries [44], [45]. Let $\textbf { z } \in \mathbb { R } ^ { d }$ be a feature vector, $\pmb { \mu } _ { c }$ the mean of class $c , \Sigma \succ 0$ a covariance matrix shared across classes, and $\pi _ { c } > 0$ the prior probability of class c. Under a shared-covariance Gaussian model, Bayes’ rule gives the posterior probability of class c given z:

$$
p ( y = c \mid \mathbf { z } ) = \frac { \pi _ { c } \mathcal { N } ( \mathbf { z } ; \pmb { \mu } _ { c } , \pmb { \Sigma } ) } { \sum _ { r = 1 } ^ { C } \pi _ { r } \mathcal { N } ( \mathbf { z } ; \pmb { \mu } _ { r } , \pmb { \Sigma } ) } .\tag{4}
$$

Here $\mathcal { N } ( \mathbf { z } ; \mu _ { c } , \Sigma )$ is the multivariate Gaussian density evaluated at z. Taking logarithms and discarding terms common to all classes yields the linear discriminant [51]

$$
\begin{array} { r } { s _ { c } ( \mathbf { z } ) = { \pmb \mu } _ { c } ^ { \top } \pmb \Sigma ^ { - 1 } \mathbf { z } - \frac { 1 } { 2 } { \pmb \mu } _ { c } ^ { \top } \pmb \Sigma ^ { - 1 } { \pmb \mu } _ { c } + \log \pi _ { c } . } \end{array}\tag{5}
$$

The mean sets the class center, the inverse covariance defines the distance metric, and the log prior contributes an additive class-specific offset. For $\mathbf { M } \bar { \mathbf { \Omega } } \succ \mathbf { \Omega } 0 .$ , write $\| \mathbf { v } \| _ { \mathbf { M } } ^ { 2 } =$ $\mathbf { v } ^ { \top } \mathbf { M } \mathbf { v }$ . With equal priors, maximizing Eq. (5) is equivalent to minimizing $\| \mathbf { z } - \mu _ { c } \| _ { \Sigma ^ { - 1 } } ^ { 2 }$

In a federation, shared statistics support classes that are rare or absent locally, while local statistics can capture clientspecific structure. The reliability of these estimates depends on sample support. The central choices in Eq. (5) are which means, covariance, and priors to share or adapt, and how strongly local evidence should influence them.

## 3.3 Problem Formulation

For each client and class, define the training subset and sample support as

$$
\begin{array} { l } { \displaystyle \mathcal { D } _ { c } ^ { ( k ) } = \{ ( x , y ) \in \mathcal { D } ^ { ( k ) } : y = c \} , } \\ { \displaystyle n _ { c } ^ { ( k ) } = \lvert \mathcal { D } _ { c } ^ { ( k ) } \rvert , ~ N _ { c } = \displaystyle \sum _ { k = 1 } ^ { K } n _ { c } ^ { ( k ) } . } \end{array}\tag{6}
$$

The label space $\mathcal { V }$ comprises all federation-known classes: $N _ { c } > 0$ for every $c ,$ although a client may have $n _ { c } ^ { ( k ) } = 0$ The locally unseen set is ${ \mathcal { U } } ^ { ( k ) } = \{ c : n _ { c } ^ { ( k ) } { \} } = 0 \}$ . Both label frequencies and class-conditional input distributions may differ across clients. Every client predicts over all C classes.

Let $P ^ { ( k ) }$ be client $k ^ { \prime } \mathrm { s }$ local target distribution and $Q _ { c }$ the distribution of inputs from class c in the shared allclass evaluation population. The class-balanced distribution $Q$ assigns probability $1 / C$ to each class, with $X \ \parallel$ $\begin{array} { r } { Y \ = \ c \ \sim \ Q _ { c } . } \end{array}$ We use $\operatorname { P r } , \ \mathbb { E } ,$ and Var for probability, expectation, and variance, respectively, and $\hat { \mathbf { 1 } \{ \cdot \} }$ for the indicator function. For any labeled target distribution $R ,$ write $\mathcal { E } _ { R } ( f ) \ : = \ : \mathrm { P r } _ { ( x , y ) \sim R } [ f ( x ) \ : \neq \ : y ]$ . The class-wise error of the deployed predictor $f ^ { ( k ) }$ is

$$
e _ { c } ^ { ( k ) } = \mathbb { E } _ { x \sim Q _ { c } } \big [ \mathbf { 1 } \{ f ^ { ( k ) } ( x ) \neq c \} \big ] ,\tag{7}
$$

and its local risk and global all-class risk are

$$
{ \mathcal E } _ { \mathrm { l o c a l } } ^ { ( k ) } = \mathbb { E } _ { ( x , y ) \sim P ^ { ( k ) } } \left[ { \bf 1 } \{ f ^ { ( k ) } ( x ) \neq y \} \right] ,\tag{8a}
$$

$$
\mathcal { E } _ { \mathrm { g l o b a l } } ^ { ( k ) } = \mathcal { E } _ { Q } ( f ^ { ( k ) } ) = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } e _ { c } ^ { ( k ) } .\tag{8b}
$$

These risks assess the same client predictor under two distributions: the local risk is evaluated under $P ^ { ( k ) }$ , while the global all-class risk gives equal weight to every federationknown class. A low $\mathcal { E } _ { \mathrm { l o c a l } } ^ { ( k ) }$ therefore need not imply a low $\mathcal { E } _ { \mathrm { g l o b a l } } ^ { ( k ) }$ . Even a low all-class risk can conceal large ${ \bf \check { \Psi } } _ { e _ { c } } ( k )$ for locally rare or unseen classes, and averaging over clients can hide particularly weak client predictors. Jointly assessing local adaptation, class coverage, and client-level weaknesses better characterizes each predictor’s recognition ability.

Our research goal is to improve recognition across the full label space and strengthen the weakest assessed abilities of poorly performing clients while retaining effective local adaptation. We assess these requirements jointly through local, all-class, rare-class, unseen-class, and client-floor metrics in Section 5.1.2. FedGRC pursues this goal through shared representation learning, a common statistical metric, and support-dependent mean adaptation.

## 4 METHOD

We present FedGRC, a unified PFL framework that integrates geometric representation consensus with statistical personalization for local adaptation and full-labelspace recognition. The framework comprises three successive stages: (i) Consensus-Guided Representation Learning (CGRL) regularizes feature directions and magnitudes through shared class references; (ii) Multilayer Metric Synthesis (MMS) constructs a shared statistical geometry from the frozen multilayer representations; and (iii) Support-Adaptive Shrinkage Personalization (SASP) adapts class means according to local sample support within this geometry. Together, these stages aim to retain recognition of locally rare or absent classes during personalization and strengthen recognition floors across clients. Fig. 2 provides an overview of the framework; the complete procedure is summarized in Algorithm 1 and detailed in the following subsections.

## 4.1 Consensus-Guided Representation Learning

CGRL combines cross-entropy supervision with shared directional and magnitude guidance to learn representations for statistical classification. At each round, clients receive the shared model and class references and jointly optimize a backbone and a training-only classifier, parameterized by $\theta = \left( \phi , \psi \right)$ . Label-preserving weak and strong augmentations, $\mathcal { T } _ { \mathrm { w } }$ and $\mathcal { T } _ { \mathrm { s } } ,$ produce 2B views from a mini-batch of B examples. The semicolon between augmented batches in Algorithm 1 denotes batch concatenation. Let $B _ { 2 }$ index these views and $\check { p } _ { i , c }$ denote the predicted probability of class c for view i. The cross-entropy loss over both augmented views:

Algorithm 1: FedGRC: Overall Procedure   
Input: Client data $\{ \mathcal { D } ^ { ( k ) } \} _ { k = 1 } ^ { K } ;$ layers $\Lambda ;$ Communication   
rounds T; local steps $\mathbf { \bar { \mathbf { \mathit { E } } } ^ { ( k ) } } ;$ learning rate $\eta ;$   
Output: $\phi _ { T } , \{ s _ { \ell } \} , \{ \mathbf { W } ^ { ( k ) } , \mathbf { \bar { b } } ^ { ( k ) } \} _ { k = 1 } ^ { K } ;$   
▶ (i) CGRL: Representation Learning   
1 Initialize client model $\theta _ { 0 } = ( \phi _ { 0 } , \psi _ { 0 } ) ;$   
2 Initialize shared references;   
3 for $t = 0 , \ldots , T - 1$ do // Communication round   
4 Sample participating clients S<sub>t</sub>;   
5 foreach $\mathbf { \bar { \boldsymbol { k } } } \in \boldsymbol { S } _ { t } ^ { \ }$ in parallel do // Client local training   
6 Set local model $\pmb \theta ^ { ( k ) } \gets \pmb \theta _ { t } ;$   
7 Reset log-norm totals $H _ { \ell , c } ^ { ( k ) } \gets 0 , q _ { c } ^ { ( k ) } \gets 0 ;$   
8 for $e = \bar { 1 } , \ldots , E ^ { ( k ) }$ do // Local updates   
9 Sample a mini-batch $\begin{array} { r } { B \sim \mathcal { D } ^ { ( \boldsymbol { k } ) } ; } \end{array}$   
10 Form paired views $B _ { 2 }  [ \mathcal { T } _ { \mathrm { w } } ( \dot { B } ) ; \mathcal { T } _ { \mathrm { s } } ( B ) ] ;$   
11 Compute CGRL total loss:   
$\begin{array} { r } { \mathcal { L } _ { \mathrm { C G R L } } ^ { \mathrm { ~ ~ } }  \mathcal { L } _ { \mathrm { C E } } + w _ { t } ( \lambda _ { \mathrm { d i r } } \mathcal { L } _ { \mathrm { d i r } } + \lambda _ { \mathrm { r a d } } \mathcal { L } _ { \mathrm { r a d } } ) ; } \end{array}$   
12 Collect log-norm sums/counts $H _ { \ell , c } ^ { ( k ) } , q _ { c } ^ { ( k ) } ;$   
13 Update $\pmb { \theta } ^ { ( k ) }  \pmb { \theta } ^ { ( k ) } - \eta \nabla _ { \pmb { \theta } ^ { ( k ) } } \mathcal { L } _ { \mathrm { C G R I } }$ <sub>L</sub>;   
14 Estimate class directions $\bar { \mathbf { u } } _ { c } ^ { ( k ) }$ via Eq. (S34);   
15 Upload $\{ \pmb { \theta } ^ { ( k ) } , \bar { \mathbf { u } } _ { c } ^ { ( k ) } , n _ { c } ^ { ( k ) } , H _ { \ell , c } ^ { ( k ) } , q _ { c } ^ { ( k ) } \}$ to Server;   
16 The Server aggregates local models into $\pmb { \theta } _ { t + 1 }$ via   
Eq. (S33), updates references via Eqs. (S35), (S38),   
then broadcasts both to Clients;   
▶ (ii) MMS: Shared Metric Synthesis   
17 The Clients freeze backbone $\phi _ { T }$ from trained model $\pmb { \theta } _ { T } ;$   
18 foreach $\overline { { k = 1 } } , \ldots , K$ in parallel do // Local statistics   
19 Extract and concatenate layer features ${ \bf h } _ { i } ^ { ( k ) } ;$   
20 Count class samples $n _ { c } ^ { ( k ) } \gets | \mathcal { D } _ { c } ^ { ( k ) } | ;$   
21 Form class feature sums and outer-product sum:   
$\begin{array} { r } { \mathbf { t } _ { c } ^ { ( k ) } \gets \sum _ { i : y _ { i } ^ { ( k ) } = c } \mathbf { h } _ { i } ^ { ( k ) } , \mathbf { G } ^ { ( k ) } \gets \sum _ { i } ^ { \bullet } \mathbf { h } _ { i } ^ { ( k ) } ( \mathbf { h } _ { i } ^ { ( k ) } ) ^ { \top } } \end{array}$   
22 Upload $\{ \{ n _ { c } ^ { ( k ) } , \mathbf { t } _ { c } ^ { ( k ) } \} _ { c } , \mathbf { G } ^ { ( k ) } \}$ to Server;   
23 The Server aggregates client statistics:   
$\begin{array} { r } { N _ { c } \gets \sum _ { k } \tilde { n _ { c } ^ { ( k ) } } , \mathbf { t } _ { c } \gets \sum _ { k } \mathbf { t } _ { c } ^ { ( k ) } , \mathbf { G } \gets \sum _ { k } \mathbf { G } ^ { ( k ) } ; } \end{array}$   
24 Compute shared layer scales s via Eqs. (S19), (22);   
25 Derive shared statistics $\mu _ { c } , \widehat { \Sigma } _ { \mathrm { w } }$ via Eq. (S20);   
26 Construct regularized shared metric:   
$\lambda _ { A } \gets \gamma \mathrm { t r } ( \widehat { \Sigma } _ { \mathrm { w } } ) / d , ~ \mathbf { A } \gets ( \widehat { \Sigma } _ { \mathrm { w } } + \lambda _ { A } \mathbf { I } _ { d } ) ^ { - 1 } ;$   
27 Broadcast $\{ \{ s _ { \ell } \} _ { \ell \in \Lambda } , \{ \pmb { \mu _ { c } } \} _ { c } , \mathbf { \bar { A } } \}$ to Clients;   
▶ (iii) SASP: Support-Adaptive Shrinkage   
28 foreach $k = 1 , \ldots , K$ do // Client personalization   
29 Compute observed-class means $\mu _ { c } ^ { ( k ) }$ via Eq. (S20);   
30 foreach $c = 1 , \ldots , C$ do // Class adaptation   
31 Compute mean shift $\Delta _ { c } ^ { ( k ) }$ via Eq. (32);   
32 Personalize class mean using local support:   
$\tilde { \pmb { \mu } } _ { c } ^ { ( k ) }  \pmb { \mu } _ { c } + [ n _ { c } ^ { ( k ) } / ( n _ { c } ^ { ( k ) } + m _ { 0 } ) ] \Delta _ { c } ^ { ( k ) } ;$   
33 Set classifier weights $\mathbf { w } _ { c } ^ { ( k ) } \gets \mathbf { A } \widetilde { \mu } _ { c } ^ { ( k ) } ;$   
34 Set classifier bias $\begin{array} { r } { \dot { b } _ { c } ^ { ( k ) } \gets - 1 / 2 ( \dot { \widetilde { \mu } } _ { c } ^ { ( k ) } ) ^ { \top } { \mathbf w } _ { c } ^ { ( k ) } ; } \end{array}$   
35 return $\phi _ { T } , \{ s _ { \ell } \} , \{ \mathbf { W } ^ { ( k ) } , \mathbf { b } ^ { ( k ) } \} _ { k = 1 } ^ { K } ;$

$$
\mathcal { L } _ { \mathrm { C E } } = - \frac { 1 } { 2 B } \sum _ { i \in \mathcal { B } _ { 2 } } \log \check { p } _ { i , y _ { i } } .\tag{9}
$$

Let Λ be the set of selected layers and ${ \bf h } _ { \ell } ( x ; \phi ) \in \mathbb { R } ^ { d _ { \ell } }$ the pooled representation at layer $\ell \in \Lambda$ , where $d _ { \ell }$ is its feature dimension and $L \ \in \ \Lambda$ denotes the final backbone layer.

![](images/c5b00922c1093dcca6fbb23cce9d6111d79d88255d05d5022f27b9042d69c87c.jpg)  
Fig. 2: Overview of FedGRC. CGRL learns shared representations over $T$ communication rounds; MMS establishes a shared statistical basis in a single additional round, enabling SASP to construct personalized classifiers locally.

Statistical classification depends on feature magnitudes as well as directions, since both determine class means and within-class scatter. CGRL therefore combines directional supervision at the final layer with magnitude regularization across all layers used by MMS.

For each of two features, let $R _ { i } = \| \mathbf h _ { i } \| _ { 2 } > 0$ and $\mathbf { u } _ { i } =$ $\mathbf { h } _ { i } / R _ { i }$ . Their squared distance decomposes as

$$
\| \mathbf { h } _ { i } - \mathbf { h } _ { j } \| _ { 2 } ^ { 2 } = ( R _ { i } - R _ { j } ) ^ { 2 } + R _ { i } R _ { j } \| \mathbf { u } _ { i } - \mathbf { u } _ { j } \| _ { 2 } ^ { 2 } .\tag{10}
$$

This decomposition motivates regulating both magnitude and direction: differences in either contribute to feature dispersion. Within a local update, client and round indices are omitted when unambiguous.

We define the two geometric objectives and then describe their joint optimization and federated reference updates.

(1) Shared Directional Guidance. Directional guidance combines batch-level discrimination with shared class references. Let ${ \bf u } _ { i } = \mathrm { u n i t } ( { \bf h } _ { L } ( x _ { i } ; \phi ) )$ , with $\mathrm { u n i t } ( \mathbf { v } ) = \mathbf { v } / \| \mathbf { v } \| _ { 2 }$ for $\mathbf { v } \neq \mathbf { 0 }$ and unit ${ \bf \Pi } ( { \bf 0 } ) = { \bf 0 } .$ . For $\mathcal { P } _ { i } = \{ j \in B _ { 2 } : j \neq i , y _ { j } = y _ { i } \}$ we use the supervised contrastive loss $\mathcal { L } _ { \mathrm { C L } } \left[ 5 2 \right]$ to promote within-class alignment and between-class separation:

$$
\mathcal { L } _ { \mathrm { C L } } = - \frac { 1 } { 2 B } \sum _ { i } \frac { 1 } { | \mathcal { P } _ { i } | } \sum _ { j \in \mathcal { P } _ { i } } \log \frac { \exp ( \mathbf { u } _ { i } ^ { \top } \mathbf { u } _ { j } / \tau ) } { Z _ { i } } ,\tag{11}
$$

where $\begin{array} { r } { Z _ { i } = \sum _ { j \in { \mathcal { B } _ { 2 } } : j \neq i } \exp ( \mathbf { u } _ { i } ^ { \top } \mathbf { u } _ { j } / \tau ) } \end{array}$ and $\tau > 0$ is the temperature. The second view of each example ensures $| \mathcal { P } _ { i } | > 0$ Local batches may omit classes, and their comparisons do not provide common directional references across clients.

We therefore introduce a prototype cross-entropy loss using shared class directions ${ \bf p } _ { c } ( t )$ :

$$
\mathcal { L } _ { \mathrm { p r o t o } } = - \frac { 1 } { 2 B } \sum _ { i \in \mathcal { B } _ { 2 } } \log \frac { \exp ( \mathbf { u } _ { i } ^ { \top } \mathbf { p } _ { y _ { i } } ( t ) / \tau ) } { \sum _ { c = 1 } ^ { C } \exp ( \mathbf { u } _ { i } ^ { \top } \mathbf { p } _ { c } ( t ) / \tau ) } .\tag{12}
$$

This loss anchors local features to a common class geometry and includes references for classes absent from the current batch. The directional objective combines batch-level discrimination with shared class alignment weighted by $\beta \colon$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d i r } } = \mathcal { L } _ { \mathrm { C L } } + \beta \mathcal { L } _ { \mathrm { p r o t o } } . } \end{array}\tag{13}
$$

(2) Magnitude Consensus through Shared References. Directional alignment does not directly constrain feature magnitudes, which also contribute to the statistics used by MMS. We introduce a shared log-norm reference $\boldsymbol { a } _ { \ell , c } ( t )$ for each class and layer, allowing their characteristic scales to differ. Unlike the common radius of Ring loss [53], separate references allow class- and layer-specific scales. For $\bar { \ell } \in \Lambda$ and a numerical stability constant $\varepsilon > 0 ,$ , define the stabilized norm and its logarithm as

$$
r _ { \ell } ( x ) = \sqrt { \| \mathbf { h } _ { \ell } ( x ; \phi ) \| _ { 2 } ^ { 2 } + \varepsilon ^ { 2 } } , \ \rho _ { \ell } ( x ) = \log r _ { \ell } ( x ) .\tag{14}
$$

The identity $\rho _ { \ell } - a _ { \ell , c } = \log ( r _ { \ell } / \exp ( a _ { \ell , c } ) )$ shows that the penalty measures relative magnitude deviations. Magnitude alignment is disabled until a shared reference is established, and the magnitude penalty is omitted for features with very small norms. To enforce these conditions, let $v _ { c } ( t ) \in \{ 0 , 1 \}$ indicate whether the shared magnitude references for class c have been initialized in an earlier round, and define the binary indicator $\chi _ { \ell i } = v _ { y _ { i } } ( t ) \mathbf { 1 } \{ r _ { \ell } ( x _ { i } ) \geq r _ { \operatorname* { m i n } } \} .$ , with safenorm threshold $r _ { \operatorname* { m i n } } > 0$ . The magnitude loss is

$$
\mathcal { L } _ { \mathrm { r a d } } = \frac { 1 } { 2 B | \Lambda | } \sum _ { \ell \in \Lambda } \sum _ { i } \chi _ { \ell i } \big [ \rho _ { \ell } \big ( x _ { i } \big ) - a _ { \ell , y _ { i } } ( t ) \big ] ^ { 2 } .\tag{15}
$$

The indicator $\chi _ { \ell i }$ is held fixed during differentiation and affects only the magnitude penalty. Lemma 4.1 characterizes the penalty by separating within-class log-norm dispersion from the mean offset relative to the shared reference.

Lemma 4.1 (Magnitude Consensus). Fix a class c, layer ℓ, and round $t ,$ and treat $a = a _ { \ell , c } ( t )$ as constant. Let $\rho$ and $\chi$ denote the corresponding log norm and binary indicator for samples of class c. Let $p _ { \chi } = \mathrm { P r } ( \chi = 1 ) > 0$ and $\bar { \rho } _ { \chi } = \mathbb { E } [ \rho \ | \ \chi = 1 ]$ , with $\mathbb { E } [ \rho ^ { 2 } \mid \chi = \bar { 1 } ] < \infty$ . Then

$$
\mathbb { E } [ \chi ( \rho - a ) ^ { 2 } ] = p _ { \chi } \Big [ \underbrace { \mathrm { V a r } ( \rho \mid \chi = 1 ) } _ { \mathrm { d i s p e r s i o n } } + \underbrace { ( \bar { \rho } _ { \chi } - a ) ^ { 2 } } _ { \mathrm { r e f e r e n c e ~ o f f s e t } } \Big ] .\tag{16}
$$

Proof. The proof is given in Appendix A.1.

Thus, among samples with $\chi = 1$ , the penalty regularizes log-norm dispersion and mean deviation from shared references, whose scales may vary across classes and layers.

(3) Local Objective and Federated Reference Updates. The local objective augments cross-entropy with directional and magnitude regularization:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C G R L } } = \mathcal { L } _ { \mathrm { C E } } + w _ { t } \Big ( \underbrace { \lambda _ { \mathrm { d i r } } \mathcal { L } _ { \mathrm { d i r } } } _ { \mathrm { d i r e c t i o n a l } } + \underbrace { \lambda _ { \mathrm { r a d } } \mathcal { L } _ { \mathrm { r a d } } } _ { \mathrm { m a g n i t u d e } } \Big ) . } \end{array}\tag{17}
$$

The weights $\lambda _ { \mathrm { d i r } }$ and $\lambda _ { \mathrm { r a d } }$ control the two regularizers. The schedule $w _ { t }$ gradually introduces geometric guidance as the shared references are established. Local models are aggregated by client sample size; after training, the classifier is replaced by the statistical head.

Shared references remain fixed during local optimization and are updated between communication rounds. Directional references aggregate post-update class directions with class-count weights, whereas magnitude references aggregate online log-norm observations. Temporal averaging moderates reference fluctuations across rounds. $\mathsf { A p - }$ pendix A.9 details the reference estimators, update conventions, and warm-up schedule.

Complementary Geometry for Statistical Classification. We first analyze how the two objectives regulate feature directions and magnitudes, then connect these effects to the within-class statistics used by MMS.

Fix the references in one local step and write $\begin{array} { r l } { \mathbf { h } _ { i } } & { { } = } \end{array}$ $\mathbf { h } _ { L } ( x _ { i } ; \phi ) \neq \mathbf { 0 }$ . Let $q _ { c } ( \mathbf { u } _ { i } )$ be the softmax probability of class c from logits $\{ \mathbf { u } _ { i } ^ { \top } \mathbf { p } _ { r } ( t ) / \tau \} _ { r = 1 } ^ { C } .$ , and let $\mathring { \ell _ { \mathrm { p } , i } } = - \log q _ { y _ { i } } ( \mathbf { u } _ { i } )$ Thus, $\begin{array} { r } { \mathcal { L } _ { \mathrm { p r o t o } } \stackrel {  } { = } ( 2 B ) ^ { - 1 } \sum _ { i \in \mathcal { B } _ { 2 } } \ell _ { \mathrm { p } , i } } \end{array}$ . Differentiation through unit normalization gives

$$
\frac { \partial \mathbf { u } _ { i } } { \partial \mathbf { h } _ { i } } = \frac { \mathbf { I } - \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } } { \Vert \mathbf { h } _ { i } \Vert _ { 2 } } ,\tag{18a}
$$

$$
\nabla _ { \mathbf { h } _ { i } } \ell _ { \mathbf { p } , i } = \frac { \mathbf { I } - \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } } { \tau \| \mathbf { h } _ { i } \| _ { 2 } } \Big ( \sum _ { c = 1 } ^ { C } q _ { c } ( \mathbf { u } _ { i } ) \mathbf { p } _ { c } ( t ) - \mathbf { p } _ { y _ { i } } ( t ) \Big ) .\tag{18b}
$$

The projection ${ { \mathbf { I } } { - } { \mathbf { u } } _ { i } { \mathbf { u } } _ { i } ^ { \top } }$ removes the radial component, leaving a gradient tangent to the sphere at $\mathbf { h } _ { i } .$ . Both directional

terms depend on raw features through unit normalization and share this tangency property.

Abbreviate $\mathbf { h } = \mathbf { h } _ { \ell } ( x _ { i } ; \phi )$ and $\ell _ { \mathrm { r } } = \chi ( \rho - a ) ^ { 2 }$ . With the reference and indicator fixed, the magnitude penalty has the feature gradient

$$
\nabla _ { \mathbf { h } } \ell _ { \mathrm { r } } = { \frac { 2 \chi ( \rho - a ) } { \| \mathbf { h } \| _ { 2 } ^ { 2 } + \varepsilon ^ { 2 } } } \mathbf { h } .\tag{19}
$$

When $\chi = 1 .$ , the gradient is radial, inducing contraction above the reference and expansion below it under a sufficiently small feature-space descent step. At the final layer, write $\mathbf { g } _ { i } = \nabla _ { \mathbf { h } _ { i } } \mathcal { L } _ { \mathrm { d i n } }$ and $\mathbf { v } _ { i } = \nabla _ { \mathbf { h } _ { i } } \ell _ { \mathrm { r } , i } .$ . The tangential and radial gradients satisfy

$$
\mathbf { h } _ { i } ^ { \top } \mathbf { g } _ { i } = 0 , \mathbf { g } _ { i } ^ { \top } \mathbf { v } _ { i } = 0 .\tag{20}
$$

Directional supervision therefore preserves magnitude to first order in feature space, while the radial term directly regulates it. This complementarity concerns feature-space gradients; both objectives act through shared backbone parameters, whose gradients need not remain orthogonal.

To relate these feature-space mechanisms to the statistics used by MMS, fix a layer ℓ and class c and pool its $N _ { c }$ features across clients under a common backbone. Abbreviate these features as $\mathbf { h } _ { i } \ = \ R _ { i } \mathbf { u } _ { i }$ , with mean $\bar { \mathbf { h } } _ { \ell , c }$ and covariance $\widehat { \Sigma } _ { \ell , c }$ defined using denominator $N _ { c }$ . Writing tr for the matrix trace, we expand squared distances about the mean and apply Eq. (10) to obtain

$$
\begin{array} { r l } { \mathrm { { r r } } ( \widehat { \Sigma } _ { \ell , e } ) = \displaystyle \frac { 1 } { N c } \displaystyle \sum _ { i = 1 } ^ { N } | | \mathbf { h } _ { i } - | | _ { 2 } ^ { 2 } } & { } \\ { = \displaystyle \frac { 1 } { 2 N c } \displaystyle \sum _ { i = 1 / 2 = 1 } ^ { N } | | \mathbf { h } _ { i } - | | _ { 2 } ^ { 2 } } & { } \\ { = \displaystyle \frac { 1 } { 2 N c } \displaystyle \sum _ { i = 1 / 2 = 1 } ^ { N } | | \mathbf { h } _ { i } - | | _ { 2 } ^ { 2 } } & { } \\ { \displaystyle \frac { 1 } { \underset { m = 1 / 2 = 1 } { \overset { N } { \sum } } ( R _ { i } - R _ { j } ) ^ { 2 } } } & { } \\ { + \displaystyle \frac { 1 } { 2 N c } \displaystyle \sum _ { i = 1 / 2 = 1 } ^ { N } R _ { i } R _ { j } | | \mathbf { u } _ { i } - \mathbf { u } _ { j } | | _ { 2 } ^ { 2 } , } & { } \\ { \displaystyle \frac { 1 } { \underset { m = 1 / 2 = 1 } { \overset { N } { \sum } } ( R _ { i } - | | \mathbf { u } _ { i } - | | \mathbf { u } _ { j } | | _ { 1 } - \mathbf { u } _ { j } | | _ { 2 } ^ { 2 } ) } . } & { } \end{array}\tag{21}
$$

Radial scatter can persist even when directions are aligned, and feature magnitudes also weight the angular contribution. This identity links the geometry regulated by final-layer directional supervision and layerwise magnitude guidance to within-class covariance estimation. It explains the statistical relevance of both forms of guidance; CGRL optimizes their respective objectives rather than directly minimizing the covariance trace.

Appendix A.2 details the derivation and numerical normalization conventions. We then freeze the final aggregated backbone to build the statistical classifier.

## 4.2 Multilayer Metric Synthesis

MMS constructs a shared statistical metric from the frozen multilayer representations. Integrating multiple depths adds information beyond the final layer; pooling class statistics across clients avoids separate covariance estimation from limited local observations. MMS recomputes all statistics from features extracted by the final aggregated backbone $\phi _ { T }$ , ensuring that statistics from different clients describe the same feature space.

(1) Layer Scaling and Feature Composition. A single scalar per layer balances average coordinate variance across layers while preserving within-layer norm ratios. With $\bar { \mathbf { h } } _ { \ell } =$ $\begin{array} { r } { \dot { N ^ { - 1 } } \sum _ { k , i } \mathbf { h } _ { \ell } \dot { ( x _ { i } ^ { ( k ) } ; \phi _ { T } ) } } \end{array}$ , define the shared layer scale as

$$
s _ { \ell } ^ { 2 } = \frac { 1 } { N d _ { \ell } } \sum _ { k , i } \big \| \mathbf { h } _ { \ell } ( x _ { i } ^ { ( k ) } ; \phi _ { T } ) - \bar { \mathbf { h } } _ { \ell } \big \| _ { 2 } ^ { 2 } .\tag{22}
$$

For a fixed ordering $( \ell _ { 1 } , \ldots , \ell _ { m } )$ of $\Lambda ,$ with $m = | \Lambda | .$ , we concatenate the scaled features as follows:

$$
\mathbf { z } ( \boldsymbol { x } ) = \left[ \begin{array} { c } { \mathbf { h } _ { \ell _ { 1 } } ( \boldsymbol { x } ; \boldsymbol { \phi } _ { T } ) / s _ { \ell _ { 1 } } } \\ { \vdots } \\ { \mathbf { h } _ { \ell _ { m } } ( \boldsymbol { x } ; \boldsymbol { \phi } _ { T } ) / s _ { \ell _ { m } } } \end{array} \right] \in \mathbb { R } ^ { d } , \quad d = \sum _ { \ell \in \Lambda } d _ { \ell } .\tag{23}
$$

The scales $s _ { \ell }$ are estimated from globally centered features, while the concatenated features are scaled without centering. For $s _ { \ell } > 0 ,$ , each scaled block has unit average coordinate variance, establishing comparable numerical scales for covariance regularization. Uniform scaling within each layer preserves norm ratios across samples, retaining the magnitude structure shaped by CGRL.

(2) Federated Estimation of Class Statistics. Let $\begin{array} { r l } { \mathcal { I } _ { c } ^ { ( k ) } = } & { { } } \end{array}$ $\{ i : y _ { i } ^ { ( k ) } = c \}$ . Shared class means provide references when local observations are sparse or absent, while local means capture client-specific deviations for personalization:

$$
{ \pmb \mu } _ { c } ^ { ( k ) } = \frac { 1 } { n _ { c } ^ { ( k ) } } \sum _ { i \in \mathcal { I } _ { c } ^ { ( k ) } } { \bf z } ( x _ { i } ^ { ( k ) } ) , n _ { c } ^ { ( k ) } > 0 ,\tag{24a}
$$

$$
\pmb { \mu _ { c } = \frac { 1 } { N _ { c } } \sum _ { k : n _ { c } ^ { ( k ) } > 0 } n _ { c } ^ { ( k ) } \pmb { \mu } _ { c } ^ { ( k ) } . }\tag{24b}
$$

To compare deviations from class centers, MMS uses a covariance-based metric that accounts for within-class variability and feature correlations. MMS estimates within-class covariance relative to shared class means, capturing both within-client variability and cross-client mean shifts. With residuals $\mathbf { e } _ { i } ^ { ( k ) } = \mathbf { z } ( x _ { i } ^ { ( k ) } ) - \mu _ { y _ { i } ^ { ( k ) } }$ , the pooled covariance is

$$
\widehat { \boldsymbol { \Sigma } } _ { \mathrm { w } } = \frac { 1 } { N } \sum _ { k , i } \mathbf { e } _ { i } ^ { ( k ) } ( \mathbf { e } _ { i } ^ { ( k ) } ) ^ { \top } .\tag{25}
$$

Proposition 4.1 decomposes this covariance into withinclient variation and between-client mean shifts.

Proposition 4.1 (Within- and Between-Client Covariance Decomposition). For every nonempty client–class group, let $\widehat { \Sigma } _ { c } ^ { ( k ) }$ be its covariance about $\mu _ { c } ^ { ( k ) }$ , computed with denominator $n _ { c } ^ { ( k ) }$ , and let $\pmb { \Delta } _ { c } ^ { ( k ) } = \pmb { \mu } _ { c } ^ { ( k ) } - \dot { \pmb { \mu } } _ { c }$ . For $\mathcal { T } = \{ ( k , c ) : n _ { c } ^ { ( k ) } > 0 \}$

$$
\widehat { \pmb { \Sigma } } _ { \mathrm { w } } = \sum _ { ( k , c ) \in \mathcal { I } } \frac { n _ { c } ^ { ( k ) } } { N } \Big ( \underbrace { \widehat { \pmb { \Sigma } } _ { c } ^ { ( k ) } } _ { \mathrm { w i t h i n - c l i e n t } } + \underbrace { \pmb { \Delta } _ { c } ^ { ( k ) } ( \pmb { \Delta } _ { c } ^ { ( k ) } ) ^ { \top } } _ { \mathrm { b e t w e e n - c l i e n t } } \Big ) .\tag{26}
$$

Proof. The proof is given in Appendix A.3.

□

Pooling covariances computed about client-specific class means would omit the between-client mean-shift term. MMS retains this variation in the shared covariance, while

SASP adapts class means along the same displacements according to local support.

The concatenated representation also allows MMS to retain within-class cross-layer covariance. Partition ${ \bf e } _ { i } ^ { ( k ) }$ into layer blocks $\mathbf { e } _ { \ell i } ^ { ( k ) } \in \mathbb { R } ^ { d _ { \ell } }$ . The (ℓ, ℓ<sup>′</sup>) block is

$$
[ \widehat { \boldsymbol { \Sigma } } _ { \mathrm { w } } ] _ { \ell \ell ^ { \prime } } = \frac { 1 } { N } \sum _ { \boldsymbol { k } , i } \mathbf { e } _ { \ell i } ^ { ( \boldsymbol { k } ) } ( \mathbf { e } _ { \ell ^ { \prime } i } ^ { ( \boldsymbol { k } ) } ) ^ { \top } , \ \ell , \ell ^ { \prime } \in \Lambda .\tag{27}
$$

The off-diagonal blocks retain cross-covariances between layer-wise residuals, enabling the shared metric to account for their joint variation.

The total within-class variation connects directly to the feature geometry analyzed in CGRL. With $\widehat { \Sigma } _ { \ell , c }$ in Eq. (21) recomputed under $\phi _ { T }$ , shared layer scaling gives

$$
\operatorname { t r } ( \widehat { \Sigma } _ { \mathrm { w } } ) = \sum _ { c = 1 } ^ { C } \frac { N _ { c } } { N } \sum _ { \ell \in \Lambda } \frac { \operatorname { t r } ( \widehat { \Sigma } _ { \ell , c } ) } { s _ { \ell } ^ { 2 } } .\tag{28}
$$

The radial–angular decomposition links MMS’s total withinclass variation to radial and magnitude-weighted angular scatter, each scaled by $s _ { \ell } ^ { - 2 }$ within layer ℓ.

Appendix A.5 derives the layer scales, class means, and pooled covariance exactly from additive client counts and first- and second-order feature moments.

(3) Regularized Shared Metric Construction. The estimated within-class covariance may be singular or illconditioned, preventing direct inversion or making class comparisons sensitive to estimation errors. Since all personalized heads use the resulting metric, we add an isotropic ridge proportional to the estimated covariance’s average variance before inversion [54]:

$$
\lambda _ { A } = \gamma \frac { \mathrm { t r } ( \widehat { \Sigma } _ { \mathrm { w } } ) } { d } , ~ \mathbf { A } = ( \widehat { \Sigma } _ { \mathrm { w } } + \lambda _ { A } \mathbf { I } _ { d } ) ^ { - 1 } .\tag{29}
$$

Here $\mathrm { t r } ( \hat { \Sigma } _ { \mathrm { w } } ) / d$ equals the average coordinate variance, or equivalently the mean covariance eigenvalue. Thus, $\gamma > 0$ controls the relative regularization strength. For positive within-class trace, $\lambda _ { A } > 0$ makes the regularized covariance positive definite, ensuring a well-defined positive-definite metric A. The shared metric A replaces $\pmb { \Sigma } ^ { - 1 }$ in Eq. (5).

Spectral Interpretation and Conditioning. For the eigendecomposition $\widehat { \pmb { \Sigma } } _ { \mathrm { w } } = \mathbf { Q } \mathrm { d i a g } ( \sigma _ { 1 } , \ldots , \sigma _ { d } ) \mathbf { Q } ^ { \bar { \top } }$ , where $\mathbf { Q }$ contains orthonormal eigenvectors and $\sigma _ { j }$ are the eigenvalues, the squared distance to a class center $\pmb { \mu }$ is

$$
\| \mathbf { z } - \pmb { \mu } \| _ { \mathbf { A } } ^ { 2 } = \sum _ { j = 1 } ^ { d } \frac { [ \mathbf { Q } ^ { \top } ( \mathbf { z } - \pmb { \mu } ) ] _ { j } ^ { 2 } } { \sigma _ { j } + \lambda _ { A } } .\tag{30}
$$

The metric downweights deviations along directions with large within-class variance, while the ridge bounds sensitivity along directions with small variance.

For extreme covariance eigenvalues $\sigma _ { \mathrm { m i n } }$ and $\sigma _ { \mathrm { m a x } } ,$ the metric’s operator norm and spectral condition number quantify its maximum amplification and conditioning:

$$
\| \mathbf { A } \| _ { 2 } = { \frac { 1 } { \sigma _ { \operatorname* { m i n } } + \lambda _ { A } } } \leq { \frac { 1 } { \lambda _ { A } } } ,\tag{31a}
$$

$$
\kappa _ { 2 } ( \mathbf { A } ) = \frac { \sigma _ { \operatorname* { m a x } } + \lambda _ { A } } { \sigma _ { \operatorname* { m i n } } + \lambda _ { A } } \leq 1 + \frac { d } { \gamma } .\tag{31b}
$$

The ridge bounds the maximum amplification by $1 / \lambda _ { A }$ and the condition number by $1 + d / \gamma$ , including when the

estimated within-class covariance is singular. Appendix A.4 gives the derivation.

## 4.3 Support-Adaptive Shrinkage Personalization

Using the representation learned by CGRL and the class statistics and metric constructed by MMS, SASP builds a personalized classifier for each client. It first adapts class means according to local support, then combines these means with the shared metric to construct a classifier over all federation-known classes. This design targets clientspecific adaptation while retaining shared references for classes with limited or no local support. The backbone, layer scales, and covariance metric remain shared, avoiding covariance estimation from sparse local observations.

(1) Support-Dependent Mean Adaptation. Local class means capture client-specific shifts, but estimates based on few samples can be unreliable. SASP therefore shrinks each local mean toward its shared counterpart according to class support. We define the local displacement as follows, setting it to zero for locally absent classes:

$$
\pmb { \Delta } _ { c } ^ { ( k ) } = \left\{ \pmb { \mu } _ { c } ^ { ( k ) } - \pmb { \mu } _ { c } , \pmb { n } _ { c } ^ { ( k ) } > 0 , \right.\tag{32}
$$

For a reference strength $m _ { 0 } > 0$ shared across clients and classes, the personalized mean is defined by

$$
\widetilde { \pmb { \mu } } _ { c } ^ { ( k ) } = \underbrace { { \pmb { \mu } } _ { c } } _ { \mathrm { s h a r e d ~ c e n t e r } } + \underbrace { \overbrace { \frac { n _ { c } ^ { ( k ) } } { n _ { c } ^ { ( k ) } + m _ { 0 } } } ^ { \alpha _ { c } ^ { ( k ) } } { \pmb { \Delta } } _ { c } ^ { ( k ) } } _ { \mathrm { s u p p o r t - w e i g h t e d ~ l o c a l ~ s h i f t } } .\tag{33}
$$

The adaptation weight is defined as $\alpha ( n ) = n / ( n + m _ { 0 } )$ where n denotes local class support. The corresponding shared weight is $1 - \alpha ( n )$ . The local weight increases with class support, while a larger $m _ { 0 }$ increases the shared weight.

Regularized Estimation and Geometric Interpretation. We justify the adaptation rule as a regularized mean estimator that balances local observations against the shared reference. Proposition 4.2 shows that this balance yields exactly the support-dependent update in Eq. (33).

Proposition 4.2 (Support-Adaptive Mean Estimation). For fixed $\mathbf A \succ 0 , \mu _ { c } ,$ and $m _ { 0 } > 0 ,$ , the unique minimizer over the candidate class mean m $\in \mathbb { R } ^ { d } o f$

$$
J _ { c } ^ { ( k ) } ( \mathbf { m } ) = \sum _ { i \in \mathbb { Z } _ { c } ^ { ( k ) } } \underbrace { \| \mathbf { z } ( x _ { i } ^ { ( k ) } ) - \mathbf { m } \| _ { \mathbf { A } } ^ { 2 } } _ { \mathrm { l o c a l e v i d e n c e } } + \underbrace { m _ { 0 } \| \mathbf { m } - \pmb { \mu } _ { c } \| _ { \mathbf { A } } ^ { 2 } } _ { \mathrm { s h a r e d r e f e r e n c e } }\tag{34}
$$

is $\widetilde { \mu } _ { c } ^ { ( k ) }$ in Eq. (33), including $n _ { c } ^ { ( k ) } = 0$ with an empty data sum.

Proof. The proof is given in Appendix A.6.

□

The data term scales with class support, while m<sub>0</sub> gives the shared mean a fixed effective sample weight [55]. The common positive-definite metric A cancels from the optimality condition; it determines classification geometry without altering the estimated mean.

The following identities quantify how closely the adapted mean fits the local center and how far it departs from the shared reference. For an observed class, its distance to the local center satisfies

$$
\| \widetilde { \pmb { \mu } } _ { c } ^ { ( k ) } - \pmb { \mu } _ { c } ^ { ( k ) } \| _ { \mathbf { A } } = ( 1 - \alpha _ { c } ^ { ( k ) } ) \| \pmb { \mu } _ { c } - \pmb { \mu } _ { c } ^ { ( k ) } \| _ { \mathbf { A } } .\tag{35}
$$

The departure from the shared center is $\| \widetilde { \pmb { \mu } } _ { c } ^ { ( k ) } - \pmb { \mu } _ { c } \| _ { \mathbf { A } } =$ $\alpha _ { c } ^ { ( k ) } \| \Delta _ { c } ^ { \bullet } \| _ { \mathbf { A } }$ . For fixed local and shared means, greater support reduces the residual to the local center while permitting a larger displacement from the shared center. This displacement determines the score perturbations examined in the stability analysis below.

(2) Personalized Classifier Construction. The adapted means and the shared MMS metric define a client-specific classifier over all federation-known classes. SASP uses class counts to control mean adaptation and equal priors to avoid an additional preference for locally frequent classes. This choice is consistent with the equal class weighting of $\mathcal { E } _ { \mathrm { g l o b a l } } ^ { ( k ) } ;$ an unsmoothed local frequency prior would also exclude locally absent classes. With $\pi _ { c } = 1 / C ,$ the log-prior term is constant across classes. Substituting the personalized means and shared metric into Eq. (5) yields

$$
\begin{array} { r } { { \bf w } _ { c } ^ { ( k ) } = { \bf A } \widetilde { \pmb { \mu } } _ { c } ^ { ( k ) } , b _ { c } ^ { ( k ) } = - \frac { 1 } { 2 } ( \widetilde { \pmb { \mu } } _ { c } ^ { ( k ) } ) ^ { \top } { \bf w } _ { c } ^ { ( k ) } , } \end{array}
$$

$$
s _ { c } ^ { ( k ) } ( \mathbf { z } ) = ( \mathbf { w } _ { c } ^ { ( k ) } ) ^ { \top } \mathbf { z } + b _ { c } ^ { ( k ) } ,\tag{36a}
$$

(36b)

$$
f ^ { ( k ) } ( \boldsymbol { x } ) = \underset { \boldsymbol { c } \in \mathcal { V } } { \arg \operatorname* { m a x } } s _ { c } ^ { ( k ) } ( \mathbf { z } ( \boldsymbol { x } ) ) .\tag{36c}
$$

The rows $( \mathbf { w } _ { c } ^ { ( k ) } ) ^ { \top }$ and entries $b _ { c } ^ { ( k ) }$ form the client head $( \mathbf { W } ^ { ( k ) } , \mathbf { b } ^ { ( k ) } )$ . This rule assigns each sample to the nearest personalized class mean under the shared metric A. Under the Gaussian model with personalized means and common covariance ${ { \bf A } ^ { - 1 } }$ , it is also the equal-prior Bayes rule. $\mathsf { A p - }$ pendix $\mathrm { A . 7 }$ gives the distance and likelihood derivations.

Prediction Stability under Mean Adaptation. We compare the personalized classifier with its shared counterpart, obtained by setting all adaptation weights to zero. Although absent-class scores remain unchanged, adapting observedclass means can alter their decision regions. Let $s _ { c } ^ { 0 } ( \mathbf { z } ) \ =$ $\begin{array} { r } { \pmb { \mu } _ { c } ^ { \top } \mathbf { A } \mathbf { z } - \frac { 1 } { 2 } \pmb { \mu } _ { c } ^ { \top } \mathbf { A } \pmb { \mu } _ { c } } \end{array}$ be the shared score and $\delta _ { c } ^ { ( k ) } = \alpha _ { c } ^ { ( \bar { k } ) } \Delta _ { c } ^ { ( k ) }$ the adapted displacement. Substituting $\widetilde { \mu } _ { c } ^ { ( k ) } = \mu _ { c } + \delta _ { c } ^ { ( k ) }$ into Eq. (36b) yields the exact change

$$
\begin{array} { r l } & { s _ { c } ^ { ( k ) } ( { \mathbf { z } } ) - s _ { c } ^ { 0 } ( { \mathbf { z } } ) = \alpha _ { c } ^ { ( k ) } ( \Delta _ { c } ^ { ( k ) } ) ^ { \top } \mathbf { A } ( { \mathbf { z } } - \pmb { \mu } _ { c } ) } \\ & { \qquad - \frac { 1 } { 2 } ( \alpha _ { c } ^ { ( k ) } ) ^ { 2 } \| \Delta _ { c } ^ { ( k ) } \| _ { \mathbf { A } } ^ { 2 } . } \end{array}\tag{37}
$$

By the Cauchy–Schwarz inequality,

$$
\begin{array} { r l } & { | s _ { c } ^ { ( k ) } ( \mathbf { z } ) - s _ { c } ^ { 0 } ( \mathbf { z } ) | \leq \epsilon _ { c } ^ { ( k ) } ( \mathbf { z } ) , } \\ & { \qquad \epsilon _ { c } ^ { ( k ) } ( \mathbf { z } ) = \| \pmb { \delta } _ { c } ^ { ( k ) } \| _ { \mathbf { A } } \| \mathbf { z } - \pmb { \mu } _ { c } \| _ { \mathbf { A } } } \\ & { \qquad + \frac { 1 } { 2 } \| \pmb { \delta } _ { c } ^ { ( k ) } \| _ { \mathbf { A } } ^ { 2 } . } \end{array}\tag{38}
$$

Let $f ^ { 0 } ( x ) = \arg \operatorname* { m a x } _ { c } s _ { c } ^ { 0 } ( \mathbf { z } ( x ) )$ be the shared predictor and $c ^ { \star } = f ^ { 0 } ( x )$ . Define its winning margin and the perturbation budget as

$$
M ^ { 0 } ( x ) = s _ { c ^ { \star } } ^ { 0 } ( \mathbf { z } ( x ) ) - \operatorname* { m a x } _ { c \neq c ^ { \star } } s _ { c } ^ { 0 } ( \mathbf { z } ( x ) ) ,\tag{39a}
$$

$$
B ^ { ( k ) } ( x ) = \epsilon _ { c ^ { \star } } ^ { ( k ) } ( { \bf z } ( x ) ) + \operatorname* { m a x } _ { c \neq c ^ { \star } } \epsilon _ { c } ^ { ( k ) } ( { \bf z } ( x ) ) .\tag{39b}
$$

Corollary 4.1 (Prediction Stability). For fixed fitted statistics and $\mathbf A \succ 0$

$$
M ^ { 0 } ( x ) > B ^ { ( k ) } ( x ) \implies f ^ { ( k ) } ( x ) = f ^ { 0 } ( x ) .\tag{40}
$$

Proof. The proof and associated target-risk bounds are given in Appendix A.8. □

Direct local-mean adaptation uses local means for observed classes and shared means for absent classes. For fixed class means, metric, and query, SASP yields no larger a perturbation budget $B ^ { ( k ) } ( { \dot { x } } )$ , making the predictionpreservation condition easier to satisfy.

Inference requires only the frozen backbone, shared layer scaling, and the client-specific linear head; personalization is completed before deployment.

## 5 EXPERIMENTS

We evaluate local adaptation, generalization across the full label space, and client recognition floors through four complementary analyses. After describing the experimental setup and evaluation protocol, we (i) benchmark FedGRC against external baselines in Section 5.2; (ii) compare recognition before and after personalization to examine the effects of local adaptation in Section 5.3; (iii) assess the contributions of the framework’s components and their internal designs through ablations in Section 5.4; and (iv) examine backbone–classifier synergy and gains over strong baseline combinations in Section 5.5.

## 5.1 Experimental Setup and Evaluation Protocol

## 5.1.1 Baselines and Datasets

We benchmark 21 external methods, comprising 15 complete FL methods and 6 statistical classifiers evaluated on 2 backbone types. The resulting 27 baseline configurations fall into three categories:

Complete FL Methods. FedAvg [56], FedRep [1], FedROD [2], FedLC [30], FedTHE [11], FedETF [21], FedFA [20], pFedFDA [14], FedVLS [31], FedDr+ [40], BTFL [12], HPFL [8], FedYoYo [41], FedLog [15], and pFedGM [16] are evaluated using their own training procedures and designated classifiers to compare complete learning and prediction systems.

Statistical Classifiers on FedAvg. CCVR [22], Fed-NCM [42], Fed3R [43], FedCGS [44], FedCOF [46], and GH-LDA [45] are applied to the FedAvg backbone. These combinations assess the effects of statistical readouts on recognition with a standard federated representation.

Statistical Classifiers on FedYoYo. CCVR, FedNCM, Fed3R, FedCGS, FedCOF, and GH-LDA are also applied to the FedYoYo backbone. These combinations provide stronger references by pairing an advanced representation-learning method with existing statistical classifiers.

Evaluation spans 5 heterogeneity settings, 3 label-space sizes, and full or dynamic partial client participation:

CIFAR-10 [57]. We partition data from 10 classes among 20 clients using Dirichlet label partitions [58], with Dir(0.3) and Dir(0.1) representing strong and extreme label heterogeneity, respectively. All clients participate in each round.

CIFAR-100 [57]. We distribute data from 100 classes among 20 clients using Dir(0.3) and Dir(0.1) label partitions, with full participation each round.

FedRS [7]. We use the NIID-1 partition, comprising 15 unified classes and 135 clients from 8 remotesensing datasets. This setting combines label and sample-size imbalance with sensor- and regiondependent feature shifts. Dynamic participation is simulated by sampling 10 of the 135 clients per round to represent clients joining and leaving training.

All experiments use 3 random seeds. Results are reported as mean ± sample standard deviation, with means alone shown in the ablation, personalization, and crosscomparison tables. Each metric is computed within each run and then averaged across seeds. The comparison tables group the three baseline categories into separate blocks: (+) denotes an attached classifier, (†) and (∗) mark the best and second-best results, and (∆) gives FedGRC’s gain over the best baseline for each metric. Accuracies are reported in (%), and gains are absolute differences in percentage points.

## 5.1.2 Evaluation Protocol

Each deployed client model predicts over all C federationknown classes. The same predictor is evaluated on its local test set $\mathcal { T } _ { \mathrm { l o c a l } } ^ { ( k ) }$ and a shared global test set $\tau _ { \mathrm { g l o b a l } }$ covering every class, including those rare or absent in its local training data. The local test sets provide local accuracy and L@25; the shared global test set provides global, unseen, and rare accuracy, together with U@25 and R@25.

Let $L ^ { ( k ) }$ denote client k’s local accuracy and $m _ { L } ^ { ( k ) } =$ $| \mathcal { T } _ { \mathrm { l o c a l } } ^ { ( k ) }$ | its local test-set size. For class $c , a _ { c } ^ { ( k ) }$ is the number of correct predictions made by client k on $\tau _ { \mathrm { g l o b a l } }$ , and $t _ { c } > 0$ is the number of test samples in that class.

Local Adaptation and Global All-Class Generalization. The overall accuracies are

$$
l o c a l = \frac { \sum _ { k } m _ { L } ^ { ( k ) } L ^ { ( k ) } } { \sum _ { k } m _ { L } ^ { ( k ) } } , g l o b a l = \frac { 1 } { K C } \sum _ { k , c } \frac { a _ { c } ^ { ( k ) } } { t _ { c } } .\tag{41}
$$

The local metric weights clients by their local test-set sizes. The global metric gives equal weight to classes within each client and then to clients, measuring the deployed client models’ recognition on a common class-balanced all-class distribution.

Locally Unseen and Rare-Class Performance. For client k, locally unseen classes form ${ \mathcal U } ^ { ( k ) } ~ = ~ \{ c ~ :$ $n _ { c } ^ { ( k ) } = 0 \}$ , with $\begin{array} { r } { m _ { U } ^ { ( k ) } = \sum _ { c \in \mathcal { U } ^ { ( k ) } } t _ { c } } \end{array}$ corresponding test samples. For clients with locally unseen classes, define client-level and aggregate unseen accuracies as

$$
U ^ { ( k ) } = \frac { \sum _ { c \in \mathcal { U } ^ { ( k ) } } a _ { c } ^ { ( k ) } } { m _ { U } ^ { ( k ) } } , u n s e e n = \frac { \sum _ { k } m _ { U } ^ { ( k ) } U ^ { ( k ) } } { \sum _ { k } m _ { U } ^ { ( k ) } } .\tag{42}
$$

Clients with no locally unseen classes are excluded from these sums and the U@25 ranking, while contributing to all other metrics.

The locally rare set $\mathcal { R } ^ { ( k ) }$ contains the 2 classes with the smallest positive local training counts, with ties broken by class ID; if the client has 1 observed class, that class forms the set. With $\begin{array} { r } { m _ { R } ^ { ( k ) } \ = \ \sum _ { c \in \mathcal { R } ^ { ( k ) } } t _ { c } } \end{array}$ corresponding test samples, the client-level accuracy and aggregate rare metric are

$$
R ^ { ( k ) } = \frac { \sum _ { c \in \mathcal { R } ^ { ( k ) } } a _ { c } ^ { ( k ) } } { m _ { R } ^ { ( k ) } } , \quad r a r e = \frac { \sum _ { k } m _ { R } ^ { ( k ) } R ^ { ( k ) } } { \sum _ { k } m _ { R } ^ { ( k ) } } .\tag{43}
$$

Both aggregate metrics weight each client’s accuracy by its corresponding class-subset test count.

Ability-Specific Personalization Floors. The L@25, U@25, and R@25 metrics average the accuracies of the worst-performing 25% of clients in local adaptation, unseen-class recognition, and rare-class recognition, respectively. Each metric summarizes one ability’s lower tail using its own client ranking.

For a metric with $m _ { v }$ eligible clients, let v collect their accuracies, sorted as $v _ { ( 1 ) } \leq \cdots \leq v _ { ( m _ { v } ) } .$ . The target tail size is $q = \lceil K / 4 \rceil$ , based on all K clients, with $q _ { v } = \operatorname* { m i n } ( q , m _ { v } )$ eligible clients included in the average. The common averaging rule is

$$
\mathrm { T a i l _ { 2 5 } } ( \mathbf { v } ) = \frac { 1 } { q _ { v } } \sum _ { j = 1 } ^ { q _ { v } } v _ { ( j ) } .\tag{44}
$$

Writing the client accuracy vectors as $\mathbf { L } , \ \mathbf { U } ,$ and R, respectively, gives $L @ 2 { \dot { 5 } } \ = \ \operatorname { T a i l } _ { 2 5 } ( \mathbf { L } )$ , U@25 = $\mathrm { T a i l _ { 2 5 } ( \mathbf { U } ) }$ , and $R { \widehat { \boldsymbol { \omega } } } 2 5 = \operatorname { T a i l } _ { 2 5 } ( \mathbf { R } )$ . The target is 5 of 20 clients for CIFAR and 34 of 135 for FedRS.

Overall Personalization Floor. The F@25 metric takes the minimum of each client’s local, unseenclass, and rare-class accuracies before averaging the lowest 25% of the resulting client scores. For clients with no locally unseen classes, the minimum is taken over local and rare-class accuracies. With ${ \bf F } =$ $( F ^ { ( 1 ) } , \ldots , F ^ { ( K ) } ) ^ { \top }$

$$
F ^ { ( k ) } = \operatorname* { m i n } \{ L ^ { ( k ) } , U ^ { ( k ) } , R ^ { ( k ) } \} ,
$$

$$
F @ 2 5 = \operatorname { T a i l } _ { 2 5 } ( \mathbf { F } ) .\tag{45}
$$

(46)

Taking the within-client minimum retains each client’s weakest ability; averaging the lowest 25% then captures the clients with the weakest such scores. The resulting personalization floor summarizes weaknesses across both abilities and clients, even when the weakest ability differs from one client to another. Appendix B explains how this aggregation and min{L@25, U@25, R@25} differ and complement each other.

## 5.1.3 Implementation

Appendix A.9 details FedGRC’s implementation, and Table 10 consolidates its parameter settings and configuration. All experiments were conducted on an Ubuntu 22.04.5 LTS server with two Intel Xeon Platinum 8481C CPUs and four NVIDIA GeForce RTX 4090 GPUs, using Python 3.11.15 and PyTorch 2.0.1 with CUDA 11.8.

## 5.2 Comprehensive Evaluation of Personalization, Generalization, and Recognition Floors

Tables 1–3 compare local adaptation, generalization across the full label space, and client recognition floors. FedGRC achieves the highest mean global, unseen, rare, and F@25 scores in all 5 heterogeneity settings. Its gains over the respective best baselines are 2.17%–4.15% in global accuracy and 2.13%–6.46% in F@25. It also leads in local accuracy on both CIFAR datasets under Dir(0.3) and on FedRS; the two Dir(0.1) settings expose a trade-off between local adaptation and broader recognition.

Local Adaptation and Class Coverage. As heterogeneity increases from Dir(0.3) to Dir(0.1), pFedFDA [14], Fed-Log [15], and pFedGM [16] achieve higher local but lower global accuracy on both CIFAR datasets (Fig. 1). The same pattern occurs for FedRep [1], FedROD [2], FedTHE [11], BTFL [12], and HPFL [8]. For example, the local accuracy of FedLog [15] on CIFAR-10 rises from 76.80% to 88.90%, while its global accuracy falls from 29.32% to 19.50%. As labels become more concentrated, frequent classes receive more weight in local evaluation, so higher local accuracy can coexist with weaker recognition across the full label space. Indeed, under Dir(0.1), pFedFDA [14], FedLog [15], and pFedGM [16] attain 87.37%–88.90% local accuracy on CIFAR-10, yet their F@25 scores round to 0.00%. Higher local scores therefore conceal severe weaknesses in individual clients’ least reliable recognition abilities.

Benefits of Statistical Classifiers. The gains from statistical classification depend on the underlying representation. On CIFAR-100 Dir(0.3), the six statistical classifiers raise FedYoYo’s [41] global accuracy from 34.59% to 48.51%– 51.56%, whereas the same classifiers on FedAvg [56] yield 33.05%–35.47%, compared with 34.80% for its original head. This contrast underscores the importance of designing the representation and statistical classifier together. FedGRC further exceeds the strongest baseline combinations in global accuracy and F@25 across all five settings.

Comprehensive Performance on FedRS. On FedRS (Table 3), FedGRC reaches 89.11% local and 87.08% global accuracy, exceeding the respective best baselines by 2.15% and 3.30%. It also improves unseen and rare accuracy by 3.46% and 1.44%, respectively. These gains span local adaptation and all-class generalization, including recognition of classes with little or no local training support. This combination is especially relevant to remote-sensing clients with limited class coverage and sensor and geographic heterogeneity.

Trade-Offs under Extreme Label Heterogeneity. Under Dir(0.1), FedGRC trails the best local baseline by 15.64% on CIFAR-10 and 3.72% on CIFAR-100, despite leading in global, unseen, rare, and F@25. On CIFAR-10, BTFL [12] attains 89.45% local accuracy, compared with FedGRC’s 73.81%. BTFL’s global accuracy and F@25 are 29.64% and 0.84%, compared with 73.05% and 63.85% for FedGRC. Under extreme label heterogeneity, these broader recognition and client-floor gains come at a measurable local-accuracy cost.

Recognition Floors across Clients. On FedRS, Table 3 shows gains of 4.24%, 4.83%, and 2.09% in L@25, U@25, and R@25, respectively, over the best baseline for each metric. These gains extend the aggregate improvements to the weakest client groups in each recognition ability. FedGRC also raises F@25 by 4.61% over its best baseline. This gain captures a higher overall personalization floor even when the weakest ability differs across clients. The CIFAR results in Appendix B, Tables 11 and 12, likewise show the highest U@25, R@25, and F@25 in all 4 CIFAR heterogeneity settings. Under Dir(0.1), however, L@25 trails the best baseline by 13.57% on CIFAR-10 and 3.59% on CIFAR-100, showing that the local-adaptation trade-off also affects the lower tail. We next examine how personalization changes this balance.

TABLE 1: Comprehensive comparison on CIFAR-10 [57] under Dirichlet label heterogeneity. All methods are trained using ResNet-18. Symbols: (†) denotes the best, (∗) the second-best, and (∆) the gain over the best baseline.
<table><tr><td rowspan=1 colspan=12>Venue                   CIFAR-10, Dir(0.3), 20 clients                          CIFAR-10, Dir(0.1), 20 clientsMethod</td></tr><tr><td rowspan=1 colspan=7>local   global   unseen      rare    F@25</td><td rowspan=1 colspan=5>local   global  unseen      rare    F@25</td></tr><tr><td rowspan=1 colspan=3>Complete FL methods (method-specific training an</td><td rowspan=1 colspan=2>d classifiers)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FedAvg [56]</td><td rowspan=1 colspan=1>AISTATS&#x27;17</td><td rowspan=1 colspan=1> $6 0 . 7 7 { \scriptstyle \pm 0 . 3 9 } ^ { - }$ </td><td rowspan=1 colspan=1>59.05±0.98</td><td rowspan=1 colspan=1>58.16±0.59</td><td rowspan=1 colspan=1> $5 4 . 4 5 { \scriptstyle \pm 1 . 2 9 }$ </td><td rowspan=1 colspan=1> $4 0 . 4 0 { \scriptstyle \pm 1 . 9 4 }$ </td><td rowspan=1 colspan=1>44.84±0.52</td><td rowspan=1 colspan=1> $4 5 . 0 1 { \scriptstyle \pm 0 . 7 5 }$ </td><td rowspan=1 colspan=1> $4 2 . 7 2 { \scriptstyle \pm 0 . 9 5 }$ </td><td rowspan=1 colspan=1> $5 0 . 3 2 { \scriptstyle \pm 0 . 2 2 }$ </td><td rowspan=1 colspan=1> $2 1 . 8 4 { \scriptstyle \pm 0 . 9 2 }$ </td></tr><tr><td rowspan=1 colspan=1>FedRep [1]</td><td rowspan=1 colspan=2>ICML&#x27;21     $7 9 . 2 5 { \scriptstyle \pm 0 . 2 5 }$ </td><td rowspan=1 colspan=1> $3 0 . 7 7 { \scriptstyle \pm 0 . 0 9 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $7 . 4 4 { \scriptstyle \pm 0 . 2 0 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $8 8 . 2 9 { \scriptstyle \pm 0 . 0 7 }$ </td><td rowspan=1 colspan=1> $1 8 . 3 5 { \scriptstyle \pm 0 . 0 6 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $1 3 . 2 0 { \scriptstyle \pm 0 . 5 3 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td rowspan=1 colspan=3>FedROD [2]     ICLR&#x27;22     $8 1 . 5 1 { \scriptstyle \pm 0 . 0 9 }$ </td><td rowspan=1 colspan=1> $5 9 . 9 4 { \scriptstyle \pm 0 . 9 9 }$ </td><td rowspan=1 colspan=1> $6 9 . 6 2 { \scriptstyle \pm 1 . 0 3 }$ </td><td rowspan=1 colspan=1> $5 2 . 6 0 { \scriptstyle \pm 0 . 7 9 }$ </td><td rowspan=1 colspan=1> $4 1 . 8 1 { \pm } 0 . 5 8$ </td><td rowspan=1 colspan=1>89.26±0.15</td><td rowspan=1 colspan=1>48.35±0.62</td><td rowspan=1 colspan=1> $5 0 . 4 7 { \scriptstyle \pm 0 . 5 9 }$ </td><td rowspan=1 colspan=1> $4 3 . 3 9 { \scriptstyle \pm 0 . 8 1 }$ </td><td rowspan=1 colspan=1> $2 2 . 7 3 { \scriptstyle \pm 1 . 2 1 }$ </td></tr><tr><td rowspan=1 colspan=2>FedLC [30]]      ICML&#x27;22</td><td rowspan=1 colspan=1> $6 1 . 1 6 { \pm } 0 . 2 8$ </td><td rowspan=1 colspan=1> $5 9 . 7 6 { \scriptstyle \pm 0 . 5 3 }$ </td><td rowspan=1 colspan=1> $7 0 . 1 9 { \scriptstyle \pm 0 . 5 8 }$ </td><td rowspan=1 colspan=1> $5 2 . 0 5 { \scriptstyle \pm 0 . 2 5 }$ </td><td rowspan=1 colspan=1> $3 9 . 1 9 { \scriptstyle \pm 0 . 3 2 }$ </td><td rowspan=1 colspan=1> $4 4 . 8 8 { \scriptstyle \pm 0 . 1 1 }$ </td><td rowspan=1 colspan=1>45.18±0.46</td><td rowspan=1 colspan=1> $4 7 . 8 0 { \scriptstyle \pm 0 . 3 9 }$ </td><td rowspan=1 colspan=1> $3 8 . 4 4 { \scriptstyle \pm 0 . 8 7 }$ </td><td rowspan=1 colspan=1> $1 0 . 4 8 { \scriptstyle \pm 1 . 0 9 }$ </td></tr><tr><td rowspan=1 colspan=2>FedTHE [11]    ICLR&#x27;23</td><td rowspan=1 colspan=1> $7 4 . 9 6 \pm 1 . 0 1$ </td><td rowspan=1 colspan=1> $5 5 . 2 0 { \scriptstyle \pm 0 . 9 7 }$ </td><td rowspan=1 colspan=1> $6 1 . 3 1 { \pm } 0 . 8 8$ </td><td rowspan=1 colspan=1> $4 8 . 9 7 { \scriptstyle \pm 1 . 0 4 }$ </td><td rowspan=1 colspan=1> $3 8 . 8 1 \pm 1 . 5 5$ </td><td rowspan=1 colspan=1> $8 2 . 8 9 { \scriptstyle \pm 0 . 5 5 }$ </td><td rowspan=1 colspan=1> $4 3 . 1 2 \pm 0 . 6 0$ </td><td rowspan=1 colspan=1> $4 3 . 3 4 { \scriptstyle \pm 0 . 6 9 }$ </td><td rowspan=1 colspan=1> $4 2 . 7 3 { \scriptstyle \pm 0 . 4 0 }$ </td><td rowspan=1 colspan=1> $2 7 . 6 8 { \scriptstyle \pm 0 . 5 2 }$ </td></tr><tr><td rowspan=1 colspan=2>FedETF [21]     ICCV′23</td><td rowspan=1 colspan=1> $7 0 . 6 6 \pm 0 . 3 8$ </td><td rowspan=1 colspan=1> $6 9 . 2 7 { \scriptstyle \pm 0 . 8 6 }$ </td><td rowspan=1 colspan=1> $7 4 . 6 2 { \scriptstyle \pm 1 . 4 0 }$ </td><td rowspan=1 colspan=1> $6 3 . 5 4 { \scriptstyle \pm 0 . 5 1 }$ </td><td rowspan=1 colspan=1> $5 2 . 8 8 { \scriptstyle \pm 0 . 2 4 }$ </td><td rowspan=1 colspan=1> $5 7 . 3 5 { \scriptstyle \pm 0 . 1 4 }$ </td><td rowspan=1 colspan=1> $5 6 . 8 6 \pm 0 . 4 3$ </td><td rowspan=1 colspan=1> $5 5 . 7 9 2 0 . 6 5$ </td><td rowspan=1 colspan=1> $6 0 . 0 3 { \scriptstyle \pm 0 . 1 6 }$ </td><td rowspan=1 colspan=1> $3 3 . 8 4 \pm 4 . 6 1$ </td></tr><tr><td rowspan=1 colspan=1>FedFA [20]</td><td rowspan=1 colspan=1>TMC&#x27;24</td><td rowspan=1 colspan=1> $3 9 . 0 5 { \scriptstyle \pm 3 . 7 9 }$ </td><td rowspan=1 colspan=1> $3 8 . 2 7 \pm 3 . 0 7$ </td><td rowspan=1 colspan=1> $3 5 . 9 3 { \scriptstyle \pm 2 . 6 8 }$ </td><td rowspan=1 colspan=1> $3 5 . 6 1 \pm 3 . 5 2$ </td><td rowspan=1 colspan=1> $1 4 . 4 7 { \scriptstyle \pm 6 . 9 2 }$ </td><td rowspan=1 colspan=1> $1 8 . 4 0 { \scriptstyle \pm 1 . 4 8 }$ </td><td rowspan=1 colspan=1> $1 8 . 4 5 { \scriptstyle \pm 1 . 5 4 }$ </td><td rowspan=1 colspan=1> $1 7 . 5 5 { \scriptstyle \pm 3 . 9 0 }$ </td><td rowspan=1 colspan=1> $2 1 . 7 4 { \scriptstyle \pm 8 . 1 6 }$ </td><td rowspan=1 colspan=1> $0 . 3 5 { \scriptstyle \pm 0 . 3 1 }$ </td></tr><tr><td rowspan=1 colspan=1>pFedFDA [14]</td><td rowspan=1 colspan=1>NeurIPS&#x27;24</td><td rowspan=1 colspan=1> $8 0 . 8 1 { \scriptstyle \pm 0 . 2 8 }$ </td><td rowspan=1 colspan=1> $3 7 . 9 9 \pm 0 . 2 1$ </td><td rowspan=1 colspan=1> $6 . 1 0 { \scriptstyle \pm 1 . 4 0 }$ </td><td rowspan=1 colspan=1> $1 5 . 9 6 { \pm } 0 . 0 9$ </td><td rowspan=1 colspan=1> $1 . 7 0 { \scriptstyle \pm 0 . 5 5 }$ </td><td rowspan=1 colspan=1> $8 8 . 2 9 { \scriptstyle \pm 0 . 1 8 }$ </td><td rowspan=1 colspan=1> $1 9 . 3 8 { \scriptstyle \pm 0 . 2 7 }$ </td><td rowspan=1 colspan=1> $0 . 2 8 { \scriptstyle \pm 0 . 0 5 }$ </td><td rowspan=1 colspan=1> $1 5 . 4 4 { \scriptstyle \pm 0 . 7 0 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td rowspan=1 colspan=1>FedVLS [31]</td><td rowspan=1 colspan=1>AAAI&#x27;25</td><td rowspan=1 colspan=1> $6 0 . 7 6 { \scriptstyle \pm 0 . 1 6 }$ </td><td rowspan=1 colspan=1> $5 9 . 2 8 { \pm } 0 . 7 8 $ </td><td rowspan=1 colspan=1> $6 9 . 3 2 { \scriptstyle \pm 0 . 8 1 }$ </td><td rowspan=1 colspan=1> $5 1 . 3 9 { \scriptstyle \pm 1 . 0 0 }$ </td><td rowspan=1 colspan=1> $3 7 . 4 3 { \scriptstyle \pm 1 . 0 4 }$ </td><td rowspan=1 colspan=1> $4 6 . 3 2 { \scriptstyle \pm 0 . 2 6 }$ </td><td rowspan=1 colspan=1> $4 6 . 2 5 { \scriptstyle \pm 0 . 3 7 }$ </td><td rowspan=1 colspan=1> $4 8 . 5 1 \pm 0 . 2 9$ </td><td rowspan=1 colspan=1> $4 0 . 5 6 { \scriptstyle \pm 0 . 5 2 }$ </td><td rowspan=1 colspan=1> $1 4 . 1 0 { \scriptstyle \pm 0 . 4 7 }$ </td></tr><tr><td rowspan=1 colspan=1>FedDr+ [40]</td><td rowspan=1 colspan=1>TMLR&#x27;25</td><td rowspan=1 colspan=1> $6 9 . 1 6 { \pm } 1 . 0 9$ </td><td rowspan=1 colspan=1> $6 8 . 3 1 \pm 1 . 3 4$ </td><td rowspan=1 colspan=1> $6 7 . 2 6 \pm 2 . 3 1$ </td><td rowspan=1 colspan=1> $6 4 . 3 6 \pm 1 . 3 1$ </td><td rowspan=1 colspan=1> $5 1 . 2 2 { \scriptstyle \pm 2 . 0 5 }$ </td><td rowspan=1 colspan=1> $3 2 . 3 0 { \scriptstyle \pm 1 . 9 6 }$ </td><td rowspan=1 colspan=1> $3 2 . 4 9 \pm 2 . 1 0$ </td><td rowspan=1 colspan=1> $2 8 . 8 8 { \scriptstyle \pm 1 . 5 2 }$ </td><td rowspan=1 colspan=1> $3 7 . 2 1 \pm 4 . 4 1$ </td><td rowspan=1 colspan=1> $1 . 6 6 \pm 1 . 5 6$ </td></tr><tr><td rowspan=1 colspan=1>BTFL [12]</td><td rowspan=1 colspan=1>KDD&#x27;25</td><td rowspan=1 colspan=1> $8 0 . 3 0 { \scriptstyle \pm 0 . 2 7 }$ </td><td rowspan=1 colspan=1>53.91±0.40</td><td rowspan=1 colspan=1> $3 0 . 0 0 { \scriptstyle \pm 0 . 7 0 }$ </td><td rowspan=1 colspan=1> $4 6 . 8 7 { \scriptstyle \pm 0 . 8 9 }$ </td><td rowspan=1 colspan=1> $1 8 . 5 0 { \scriptstyle \pm 1 . 6 7 }$ </td><td rowspan=1 colspan=1> $8 9 . 4 5 { \scriptstyle \pm 0 . 2 5 } ^ { \dagg</td><td rowspan=1 colspan=1>er } 2 9 . 6 4 { \scriptstyle \pm 0 . 5 0 }$ </td><td rowspan=1 colspan=1> $7 . 4 5 { \scriptstyle \pm 0 . 5 9 }$ </td><td rowspan=1 colspan=1> $3 8 . 9 2 { \scriptstyle \pm 0 . 8 0 }$ </td><td rowspan=1 colspan=1> $0 . 8 4 \pm 0 . 4 4$ </td></tr><tr><td rowspan=1 colspan=1>HPFL [8]</td><td rowspan=1 colspan=1>ICLR&#x27;25</td><td rowspan=1 colspan=1> ${ \bf 8 1 . 7 2 } _ { \pm 0 . 4 1 } \mathrm { ~ ^ { * } }$ </td><td rowspan=1 colspan=1>50.06±0.35</td><td rowspan=1 colspan=1> $5 0 . 5 5 { \scriptstyle \pm 0 . 2 4 }$ </td><td rowspan=1 colspan=1> $4 4 . 2 4 { \scriptstyle \pm 0 . 7 9 }$ </td><td rowspan=1 colspan=1> $1 7 . 6 8 { \pm } 1 . 6 4$ </td><td rowspan=1 colspan=1> $8 9 . 4 1 { \scriptstyle \pm 0 . 1 6 ^ { * } }</td><td rowspan=1 colspan=1>2 6 . 0 3 { \scriptstyle \pm 0 . 4 6 }$ </td><td rowspan=1 colspan=1> $2 4 . 8 3 { \scriptstyle \pm 0 . 7 8 }$ </td><td rowspan=1 colspan=1> $2 5 . 3 9 { \scriptstyle \pm 0 . 9 3 }$ </td><td rowspan=1 colspan=1> $1 0 . 8 6 \pm 2 . 3 1$ </td></tr><tr><td rowspan=1 colspan=1>FedYoYo [41]</td><td rowspan=1 colspan=1>ICCV&#x27;25</td><td rowspan=1 colspan=1> $7 6 . 9 6 \pm 0 . 0 6$ </td><td rowspan=1 colspan=1> $7 6 . 6 4 \pm 0 . 4 8$ </td><td rowspan=1 colspan=1> $7 7 . 1 7 { \pm } 1 . 5 0$ </td><td rowspan=1 colspan=1> $7 4 . 5 2 { \scriptstyle \pm 0 . 5 7 }$ </td><td rowspan=1 colspan=1> $6 8 . 1 8 { \pm } 1 . 0 5 $ </td><td rowspan=1 colspan=1> $5 9 . 1 5 { \scriptstyle \pm 0 . 9 7 }$ </td><td rowspan=1 colspan=1> $5 8 . 8 0 { \scriptstyle \pm 0 . 8 1 }$ </td><td rowspan=1 colspan=1> $5 6 . 2 1 \pm 1 . 4 4$ </td><td rowspan=1 colspan=1> $6 6 . 1 2 \pm 1 . 4 3$ </td><td rowspan=1 colspan=1> $2 8 . 2 3 { \scriptstyle \pm 3 . 7 5 }$ </td></tr><tr><td rowspan=1 colspan=1>FedLog [15]</td><td rowspan=1 colspan=1>TMLR&#x27;26</td><td rowspan=1 colspan=1> $7 6 . 8 0 { \scriptstyle \pm 0 . 2 1 }$ </td><td rowspan=1 colspan=1> $2 9 . 3 2 { \scriptstyle \pm 0 . 0 6 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $6 . 1 7 { \scriptstyle \pm 0 . 2 9 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $8 8 . 9 0 { \scriptstyle \pm 0 . 3 5 }$ </td><td rowspan=1 colspan=1> $1 9 . 5 0 { \scriptstyle \pm 0 . 2 1 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $1 5 . 1 1 { \scriptstyle \pm 0 . 2 5 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td rowspan=1 colspan=1>pFedGM [16]</td><td rowspan=1 colspan=1>arXiv′26</td><td rowspan=1 colspan=1> $8 1 . 5 1 { \scriptstyle \pm 0 . 1 1 }$ </td><td rowspan=1 colspan=1>36.16±0.4</td><td rowspan=1 colspan=1>40.14±0.15</td><td rowspan=1 colspan=1> $1 5 . 3 6 { \pm } 0 . 3 8$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td rowspan=1 colspan=1> $8 7 . 3 7 { \scriptstyle \pm 0 . 1 3 }$ </td><td rowspan=1 colspan=1> $1 7 . 3 3 { \pm } 0 . 5 9$ </td><td rowspan=1 colspan=1>0.00±0.00</td><td rowspan=1 colspan=1> $1 1 . 7 4 { \scriptstyle \pm 1 . 2 2 }$ </td><td rowspan=1 colspan=1> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td rowspan=1 colspan=1>Statistical classifi</td><td rowspan=1 colspan=1>ers assembled o</td><td rowspan=1 colspan=1>n a vanilla b</td><td rowspan=1 colspan=1>ackbone (Fed</td><td rowspan=1 colspan=1>Avg +)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>+ CCVR [22]</td><td rowspan=1 colspan=1>NeurIPS&#x27;21</td><td rowspan=1 colspan=1>63.05±0.02</td><td rowspan=1 colspan=1> $6 1 . 4 6 \pm 1 . 0 5$ </td><td rowspan=1 colspan=1> $6 5 . 6 4 \pm 1 . 1 2$ </td><td rowspan=1 colspan=1> $5 6 . 9 4 { \scriptstyle \pm 1 . 2 4 }$ </td><td rowspan=1 colspan=1> $4 7 . 9 9 \pm 1 . 9 0$ </td><td rowspan=1 colspan=1>51.98±0.52</td><td rowspan=1 colspan=1> $5 1 . 3 2 { \scriptstyle \pm 0 . 6 3 }$ </td><td rowspan=1 colspan=1> $5 0 . 9 1 { \scriptstyle \pm 0 . 6 0 }$ </td><td rowspan=1 colspan=1> $5 2 . 8 2 { \scriptstyle \pm 0 . 8 4 }$ </td><td rowspan=1 colspan=1> $3 9 . 1 7 { \scriptstyle \pm 0 . 8 8 }$ </td></tr><tr><td rowspan=1 colspan=1>+ FedNCM [42]</td><td rowspan=1 colspan=1>NeurIPS&#x27;23</td><td rowspan=1 colspan=1> $6 3 . 1 7 { \scriptstyle \pm 0 . 2 4 }$ </td><td rowspan=1 colspan=1> $6 1 . 8 2 { \scriptstyle \pm 0 . 8 8 }$ </td><td rowspan=1 colspan=1> $6 6 . 6 9 { \scriptstyle \pm 0 . 7 1 }$ </td><td rowspan=1 colspan=1> $5 7 . 1 5 { \pm } 1 . 2 3 $ </td><td rowspan=1 colspan=1> $4 8 . 5 1 \pm 1 . 6 1$ </td><td rowspan=1 colspan=1> $5 2 . 0 3 { \scriptstyle \pm 0 . 5 3 }$ </td><td rowspan=1 colspan=1> $5 1 . 7 2 { \scriptstyle \pm 0 . 1 1 }$ </td><td rowspan=1 colspan=1> $5 1 . 2 3 { \scriptstyle \pm 0 . 2 7 }$ </td><td rowspan=1 colspan=1> $5 3 . 3 9 { \scriptstyle \pm 0 . 1 3 }$ </td><td rowspan=1 colspan=1> $4 0 . 6 0 { \scriptstyle \pm 1 . 0 8 }$ </td></tr><tr><td rowspan=1 colspan=1>+ Fed3R [43]</td><td rowspan=1 colspan=1>ICML&#x27;24</td><td rowspan=1 colspan=1> $6 3 . 4 9 { \scriptstyle \pm 0 . 2 4 }$ </td><td rowspan=1 colspan=1> $6 2 . 4 2 { \scriptstyle \pm 1 . 2 4 } \qua</td><td rowspan=1 colspan=1>d 6 6 . 1 7 { \scriptstyle \pm 0 . 9 2 }$ </td><td rowspan=1 colspan=1> $5 7 . 0 6 \pm 1 . 6 7$ </td><td rowspan=1 colspan=1> $4 4 . 8 4 { \scriptstyle \pm 2 . 2 7 }$ </td><td rowspan=1 colspan=1> $5 5 . 5 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td rowspan=1 colspan=1> $5 4 . 7 6 { \scriptstyle \pm 0 . 2 3 }$ </td><td rowspan=1 colspan=1> $5 3 . 9 3 { \scriptstyle \pm 0 . 2 7 }$ </td><td rowspan=1 colspan=1> $5 7 . 6 0 { \scriptstyle \pm 0 . 1 1 }$ </td><td rowspan=1 colspan=1> $4 1 . 3 5 { \scriptstyle \pm 0 . 5 7 }$ </td></tr><tr><td rowspan=1 colspan=1>+ FedCGS [44]</td><td rowspan=1 colspan=1>AAAI&#x27;25</td><td rowspan=1 colspan=1> $6 3 . 1 5 { \scriptstyle \pm 0 . 3 8 }$ </td><td rowspan=1 colspan=1>61.77±0.71</td><td rowspan=1 colspan=1>65.78±0.73</td><td rowspan=1 colspan=1> $5 7 . 9 8 { \scriptstyle \pm 0 . 8 5 }$ </td><td rowspan=1 colspan=1> $5 0 . 8 0 { \scriptstyle \pm 1 . 6 4 }$ </td><td rowspan=1 colspan=1> $5 5 . 0 8 { \scriptstyle \pm 0 . 4 0 }$ </td><td rowspan=1 colspan=1> $5 4 . 3 6 { \scriptstyle \pm 0 . 2 4 }$ </td><td rowspan=1 colspan=1> $5 4 . 5 0 { \scriptstyle \pm 0 . 1 1 }$ </td><td rowspan=1 colspan=1> $5 4 . 3 3 { \scriptstyle \pm 0 . 5 5 }$ </td><td rowspan=1 colspan=1> $4 4 . 8 9 { \scriptstyle \pm 0 . 3 4 }$ </td></tr><tr><td rowspan=1 colspan=1> $+ \mathrm { F e d C O F } \ [ 4 6 ]$ </td><td rowspan=1 colspan=1> $\mathrm { N e u r I P S } ^ { \prime } 2 5$ </td><td rowspan=1 colspan=1> $5 5 . 7 8 { \scriptstyle \pm 0 . 6 9 }$ </td><td rowspan=1 colspan=1> $5 4 . 1 0 { \scriptstyle \pm 0 . 4 4 }$ </td><td rowspan=1 colspan=1> $5 5 . 7 4 { \scriptstyle \pm 0 . 4 2 }$ </td><td rowspan=1 colspan=1> $5 0 . 1 8 { \scriptstyle \pm 0 . 2 3 }$ </td><td rowspan=1 colspan=1> $3 9 . 5 3 { \scriptstyle \pm 0 . 2 2 }$ </td><td rowspan=1 colspan=1> $4 6 . 7 7 { \scriptstyle \pm 0 . 6 0 }$ </td><td rowspan=1 colspan=1> $4 6 . 6 8 { \scriptstyle \pm 0 . 3 6 }$ </td><td rowspan=1 colspan=1> $4 6 . 0 0 { \scriptstyle \pm 0 . 2 5 }$ </td><td rowspan=1 colspan=1> $4 8 . 9 5 { \scriptstyle \pm 1 . 1 3 }$ </td><td rowspan=1 colspan=1> $3 4 . 6 6 \pm 1 . 3 2$ </td></tr><tr><td rowspan=1 colspan=1> $+ \mathrm { G H - L D A } \left[ 4 5 \right]$ </td><td rowspan=1 colspan=1>ICLR&#x27;26</td><td rowspan=1 colspan=1> $6 2 . 8 3 { \scriptstyle \pm 0 . 3 9 }$ </td><td rowspan=1 colspan=1>61.36±0.92</td><td rowspan=1 colspan=1>65.11±1.00</td><td rowspan=1 colspan=1> $5 7 . 9 5 { \scriptstyle \pm 0 . 9 6 }$ </td><td rowspan=1 colspan=1> $5 0 . 6 3 { \scriptstyle \pm 1 . 8 3 }$ </td><td rowspan=1 colspan=1> $5 5 . 2 2 { \scriptstyle \pm 0 . 2 7 }$ </td><td rowspan=1 colspan=1> $5 4 . 8 9 { \scriptstyle \pm 0 . 4 0 }$ </td><td rowspan=1 colspan=1> $5 4 . 9 1 { \scriptstyle \pm 0 . 4 6 }$ </td><td rowspan=1 colspan=1> $5 5 . 2 3 { \scriptstyle \pm 0 . 3 9 }$ </td><td rowspan=1 colspan=1> $4 5 . 2 4 \pm 0 . 4 1$ </td></tr><tr><td rowspan=1 colspan=1>Statistical classifie</td><td rowspan=1 colspan=1>rs assembled on</td><td rowspan=1 colspan=1> $t h e S O T A \ b a</td><td rowspan=1 colspan=1>c k b o n e \ ( F e d Y</td><td rowspan=1 colspan=1>o Y o + )$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1> $+ \ C C \mathrm { V R } \ [ 2 2 ]$ </td><td rowspan=1 colspan=1>NeurIPS&#x27;21</td><td rowspan=1 colspan=1> $7 8 . 1 8 { \pm } 0 . 1 9$ </td><td rowspan=1 colspan=1>77.53±0.06</td><td rowspan=1 colspan=1> $8 0 . 7 1 { \scriptstyle \pm 0 . 2 8 }$ </td><td rowspan=1 colspan=1> $7 3 . 3 8 { \pm } 0 . 1 7$ </td><td rowspan=1 colspan=1> $6 5 . 8 9 { \scriptstyle \pm 0 . 9 3 }$ </td><td rowspan=1 colspan=1> $6 7 . 8 0 { \scriptstyle \pm 0 . 1 0 }$ </td><td rowspan=1 colspan=1> $6 6 . 7 8 { \scriptstyle \pm 0 . 4 8 }$ </td><td rowspan=1 colspan=1>66.34±0.59</td><td rowspan=1 colspan=1> $6 8 . 1 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td rowspan=1 colspan=1> $5 3 . 2 5 { \scriptstyle \pm 1 . 3 2 }$ </td></tr><tr><td rowspan=1 colspan=1> $+ \mathrm { F e d N C M } \left[ 4 2 \right]$ </td><td rowspan=1 colspan=1>NeurIPS&#x27;23</td><td rowspan=1 colspan=1> $7 7 . 6 9 { \scriptstyle \pm 0 . 1 4 }$ </td><td rowspan=1 colspan=1> $7 7 . 3 0 { \scriptstyle \pm 0 . 4 2 }$ </td><td rowspan=1 colspan=1> $8 0 . 6 9 { \scriptstyle \pm 0 . 4 7 }$ </td><td rowspan=1 colspan=1> $7 2 . 9 4 { \scriptstyle \pm 0 . 5 1 }$ </td><td rowspan=1 colspan=1> $6 4 . 7 2 { \scriptstyle \pm 1 . 1 1 }$ </td><td rowspan=1 colspan=1> $6 3 . 5 6 { \scriptstyle \pm 0 . 6 0 }$ </td><td rowspan=1 colspan=1> $6 1 . 9 3 { \scriptstyle \pm 0 . 1 2 }$ </td><td rowspan=1 colspan=1> $6 1 . 1 2 { \scriptstyle \pm 0 . 0 9 }$ </td><td rowspan=1 colspan=1> $6 4 . 7 4 { \scriptstyle \pm 0 . 7 0 }$ </td><td rowspan=1 colspan=1> $5 0 . 4 3 { \scriptstyle \pm 0 . 7 2 }$ </td></tr><tr><td rowspan=1 colspan=1>+ Fed3R [43]</td><td rowspan=1 colspan=1> $\mathrm { I C M L } ^ { \prime } 2 4$ </td><td rowspan=1 colspan=1> $7 9 . 1 9 { \scriptstyle \pm 0 . 2 3 }$ </td><td rowspan=1 colspan=1> $7 8 . 3 9 { \scriptstyle \pm 0 . 2 0 }$ </td><td rowspan=1 colspan=1> $8 2 . 1 8 { \scriptstyle \pm 0 . 3 3 } ^ { \ast }$ </td><td rowspan=1 colspan=1> $7 3 . 8 0 { \scriptstyle \pm 0 . 1 0 }$ </td><td rowspan=1 colspan=1> $6 5 . 1 0 { \scriptstyle \pm 0 . 3 8 }$ </td><td rowspan=1 colspan=1> $7 0 . 7 1 { \scriptstyle \pm 0 . 1 7 }$ </td><td rowspan=1 colspan=1> $6 9 . 5 5 { \scriptstyle \pm 0 . 3 7 } ^ { * }$ </td><td rowspan=1 colspan=1> $6 8 . 8 7 { \scriptstyle \pm 0 . 4 6 }$ </td><td rowspan=1 colspan=1> $\mathbf { 7 1 . 6 1 { \scriptstyle \pm 0 . 4 7 } ^ { * } }$ </td><td rowspan=1 colspan=1> $5 2 . 4 9 { \scriptstyle \pm 2 . 0 1 }$ </td></tr><tr><td rowspan=1 colspan=1> $+ \mathrm { F e d C G S } \ [ 4 4 ]$ </td><td rowspan=1 colspan=1>AAAI&#x27;25</td><td rowspan=1 colspan=1> $7 9 . 0 8 { \scriptstyle \pm 0 . 2 8 }$ </td><td rowspan=1 colspan=1> $7 8 . 5 9 { \scriptstyle \pm 0 . 0 9 } ^ { * }$ </td><td rowspan=1 colspan=1> $8 1 . 5 3 { \scriptstyle \pm 0 . 3 6 }$ </td><td rowspan=1 colspan=1> $7 5 . 3 4 { \scriptstyle \pm 0 . 2 3 } ^ { \ast }$ </td><td rowspan=1 colspan=1> $6 9 . 7 2 { \scriptstyle \pm 0 . 6 9 } ^ { * }$ </td><td rowspan=1 colspan=1> $7 0 . 4 0 { \scriptstyle \pm 0 . 3 5 }$ </td><td rowspan=1 colspan=1> $6 9 . 0 3 { \scriptstyle \pm 0 . 3 6 }$ </td><td rowspan=1 colspan=1> ${ \bf 6 8 . 9 1 { \scriptstyle \pm 0 . 4 1 } } ^ { * }$ </td><td rowspan=1 colspan=1> $6 9 . 3 3 { \scriptstyle \pm 0 . 2 3 }$ </td><td rowspan=1 colspan=1> ${ \bf 6 1 . 0 3 } { \scriptstyle \pm 0 . 1 4 } ^ { * }$ </td></tr><tr><td rowspan=1 colspan=1>+ FedCOF [46]</td><td rowspan=1 colspan=1>NeurIPS&#x27;25</td><td rowspan=1 colspan=1> $7 6 . 7 0 { \scriptstyle \pm 0 . 1 0 }$ </td><td rowspan=1 colspan=1> $7 6 . 2 3 { \scriptstyle \pm 0 . 1 9 }$ </td><td rowspan=1 colspan=1> $7 9 . 6 7 { \scriptstyle \pm 0 . 2 9 }$ </td><td rowspan=1 colspan=1> $7 2 . 1 6 \pm 0 . 4 2$ </td><td rowspan=1 colspan=1> $6 2 . 4 7 { \scriptstyle \pm 0 . 9 2 }$ </td><td rowspan=1 colspan=1> $6 3 . 9 9 { \scriptstyle \pm 0 . 4 7 }$ </td><td rowspan=1 colspan=1> $6 2 . 7 6 { \scriptstyle \pm 0 . 1 6 }$ </td><td rowspan=1 colspan=1> $6 2 . 0 6 { \pm } 0 . 1 9$ </td><td rowspan=1 colspan=1> $6 4 . 9 8 { \scriptstyle \pm 0 . 3 2 }$ </td><td rowspan=1 colspan=1> $5 0 . 8 8 \pm 1 . 2 2$ </td></tr><tr><td rowspan=1 colspan=1> $+ \mathrm { G H - L D A } \left[ 4 5 \right]$ </td><td rowspan=1 colspan=1>ICLR&#x27;26</td><td rowspan=1 colspan=1> $7 8 . 8 9 { \scriptstyle \pm 0 . 2 3 }$ </td><td rowspan=1 colspan=1> $7 8 . 5 3 { \scriptstyle \pm 0 . 0 8 }$ </td><td rowspan=1 colspan=1> $8 1 . 4 0 { \scriptstyle \pm 0 . 3 4 }$ </td><td rowspan=1 colspan=1> $7 5 . 2 8 { \scriptstyle \pm 0 . 1 0 }$ </td><td rowspan=1 colspan=1> $6 9 . 6 3 { \scriptstyle \pm 0 . 5 3 }$ </td><td rowspan=1 colspan=1> $6 9 . 8 2 { \scriptstyle \pm 0 . 4 1 }$ </td><td rowspan=1 colspan=1> $6 8 . 6 8 { \scriptstyle \pm 0 . 3 5 }$ </td><td rowspan=1 colspan=1> $6 8 . 4 2 { \scriptstyle \pm 0 . 4 0 }$ </td><td rowspan=1 colspan=1> $6 9 . 4 2 { \scriptstyle \pm 0 . 5 1 }$ </td><td rowspan=1 colspan=1> $5 9 . 4 0 { \scriptstyle \pm 0 . 5 8 }$ </td></tr><tr><td rowspan=1 colspan=3>FedGRC (Ours)               $8 2 . 1 6 { \scriptstyle \pm 0 . 2 7 } ^ { \dagger }$ </td><td rowspan=1 colspan=1>80.76±0.31†</td><td rowspan=1 colspan=1> $8 4 . 2 1 { \scriptstyle \pm 0 . 4 6 } ^ { \dagger }$ </td><td rowspan=1 colspan=1> $7 7 . 5 1 { \scriptstyle \pm 0 . 4 1 } ^ { \dagger }$ </td><td rowspan=1 colspan=1> ${ \mathbf { 7 1 . 8 5 \pm 0 . 6 6 } } ^ { \dagger }$ </td><td rowspan=1 colspan=1> $7 3 . 8 1 { \scriptstyle \pm 0 . 4 6 }$ </td><td rowspan=1 colspan=1> $7 3 . 0 5 { \scriptstyle \pm 0 . 3 3 } ^ { \dagger }$ </td><td rowspan=1 colspan=1> $7 3 . 8 8 { \scriptstyle \pm 0 . 4 0 } ^ { \dagger }$ </td><td rowspan=1 colspan=1> $7 { \bf 1 . 9 5 } \pm 0 . 4 1 ^ { \dagger }$ </td><td rowspan=1 colspan=1> $6 3 . 8 5 { \scriptstyle \pm 0 . 1 3 } ^ { \dagger }$ </td></tr><tr><td rowspan=1 colspan=3>∆ vs. Best                        +0.44</td><td rowspan=1 colspan=1>+2.17</td><td rowspan=1 colspan=1>+2.03</td><td rowspan=1 colspan=1>+2.17</td><td rowspan=1 colspan=1>+2.13</td><td rowspan=1 colspan=1>-15.64</td><td rowspan=1 colspan=1> $+ 3 . 5 0$ </td><td rowspan=1 colspan=1>+4.97</td><td rowspan=1 colspan=1>+0.34</td><td rowspan=1 colspan=1>+2.82</td></tr></table>

## 5.3 Recognition Before and After Personalization

## 5.4 Assessing Contributions of the Proposed Designs

Table 4 compares predictors immediately before and after terminal personalization for 3 methods with explicit local adaptation stages. FedETF [21] adapts both the representation and classifier, whereas pFedGM [16] freezes its shared generator and adapts the statistical classifier. FedGRC keeps the shared CGRL backbone, MMS layer scales, and covariance metric fixed and replaces shared class means with SASP means, isolating the effect of its final adaptation stage.

55.53% to 55.61% and F@25 from 46.93% to 47.56%, with a 0.27% decrease in global accuracy. On FedRS, all five metrics improve, including gains of 1.41% in local accuracy and 1.61% in F@25. These results support class-mean adaptation within a shared representation and metric as a way to preserve broad recognition during personalization. Next, ablations isolate the contributions of representation learning, statistical readout, and support-adaptive mean estimation.

## 5.4.1 Ablation of FedGRC: Component Contributions to Generalization and Personalized Recognition

Both reference methods improve local accuracy while substantially reducing generalization and client recognition floors. On CIFAR-100 Dir(0.3), their local gains of 7.12%– 13.40% accompany losses of 6.99%–13.19% in global accuracy and 17.54%–22.69% in F@25. The same pattern occurs on FedRS: FedETF [21] gains 15.75% locally but loses 25.25% in global accuracy and 31.78% in F@25. These changes show that fitting the local distribution can substantially weaken recognition beyond it, even when adaptation is confined to the classifier. They motivate controlling the influence of local evidence during personalization.

Table 5 traces the construction of FedGRC on CIFAR-100 Dir(0.3) and FedRS by adding CGRL, MMS, and SASP sequentially. The initial system is trained with supervised cross-entropy averaged over paired weakly and strongly augmented views of each example and evaluated with the aggregated linear head. CGRL adds geometric guidance during representation learning while retaining the linear head; MMS replaces that head with a shared multilayer statistical readout; SASP then adapts the class means. Reported changes are relative to the initial system, while comparisons between adjacent stages reveal the effect of each addition.

SASP improves local adaptation and recognition floors while largely retaining the shared classifier’s all-class accuracy. On CIFAR-100 Dir(0.3), it raises local accuracy from

The complete configuration improves all 5 metrics over the initial system in both settings, demonstrating gains in local adaptation, broader recognition, and client floors from the combined design.

TABLE 2: Comprehensive comparison on CIFAR-100 [57] under Dirichlet label heterogeneity. All methods are trained using ResNet-18. Symbols: (†) denotes the best, (∗) the second-best, and (∆) the gain over the best baseline.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Venue</td><td colspan="5">CIFAR-100, Dir(0.3), 20 clients</td><td colspan="5">CIFAR-100, Dir(0.1), 20 clients</td></tr><tr><td>local</td><td colspan="2"></td><td>unseen rare</td><td>F@25</td><td>local</td><td>global</td><td>unseen</td><td>rare</td><td>F@25</td></tr><tr><td></td><td>global</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Complete FL methods (method-specific training and classifiers)</td><td></td><td></td></tr><tr><td>FedAvg [56]</td><td>AISTATS&#x27;17</td><td> $3 5 . 3 7 { \scriptstyle \pm 0 . 4 5 }$ </td><td>34.80±0.82 33.33±0.97</td><td></td><td> $3 6 . 5 7 { \scriptstyle \pm 1 . 2 8 }$ </td><td> $2 3 . 0 0 { \scriptstyle \pm 4 . 1 3 }$ </td><td> $3 2 . 8 5 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $3 2 . 7 3 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $3 2 . 3 9 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $2 8 . 8 5 { \scriptstyle \pm 1 . 2 5 }$ </td><td> $1 7 . 1 1 \pm 1 . 8 3$ </td></tr><tr><td>FedRep [1]</td><td>ICML&#x27;21</td><td> $3 7 . 7 4 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $1 1 . 8 8 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 3 7 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 9 . 8 1 \pm 0 . 3 1$ </td><td> $9 . 3 3 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 2 3 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>FedROD [2]</td><td>ICLR&#x27;22</td><td> $4 8 . 6 6 \pm 0 . 1 2$ </td><td> $3 4 . 7 2 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $3 4 . 9 9 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $3 5 . 1 0 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $2 3 . 6 7 { \scriptstyle \pm 1 . 7 7 }$ </td><td> ${ \bf 5 6 . 9 7 { \scriptstyle \pm 0 . 7 5 } } ^ { \dagger }$ </td><td>32.93±0.69</td><td> $3 3 . 3 9 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $2 8 . 8 2 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $1 8 . 2 2 { \scriptstyle \pm 1 . 8 0 }$ </td></tr><tr><td>FedLC [30]</td><td>ICML&#x27;22</td><td> $3 2 . 7 8 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $3 2 . 0 4 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $3 3 . 0 3 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $3 1 . 5 3 { \pm } 1 . 1 5$ </td><td> $2 0 . 5 0 { \scriptstyle \pm 1 . 5 3 }$ </td><td> $3 0 . 7 8 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $3 0 . 9 1 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $3 1 . 7 3 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $2 6 . 8 6 \pm 0 . 3 6$ </td><td> $1 6 . 7 2 { \scriptstyle \pm 0 . 8 1 }$ </td></tr><tr><td>FedTHE [11]</td><td>ICLR&#x27;23</td><td> $3 7 . 0 3 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $2 8 . 7 6 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $2 8 . 6 4 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $2 9 . 4 4 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $1 8 . 9 4 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $4 4 . 6 9 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $2 7 . 2 9 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $2 7 . 5 6 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $2 4 . 6 7 { \scriptstyle \pm 1 . 0 5 }$ </td><td> $1 4 . 7 8 { \scriptstyle \pm 1 . 5 5 }$ </td></tr><tr><td>FedETF [21]</td><td>ICCV&#x27;23</td><td> $3 9 . 0 8 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $3 8 . 7 7 \pm 0 . 1 4$ </td><td>36.94±0.63</td><td> $4 0 . 5 7 { \scriptstyle \pm 1 . 1 1 }$ </td><td> $2 4 . 0 2 { \scriptstyle \pm 3 . 0 5 }$ </td><td> $3 8 . 0 0 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $3 7 . 8 0 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $3 7 . 2 0 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $3 4 . 5 2 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $2 2 . 1 7 { \scriptstyle \pm 1 . 4 3 }$ </td></tr><tr><td>FedFA [20]</td><td>TMC&#x27;24</td><td> $7 . 5 3 { \scriptstyle \pm 2 . 6 2 }$ </td><td> $7 . 6 6 \pm 2 . 7 8$ </td><td> $6 . 2 9 { \scriptstyle \pm 1 . 9 8 }$ </td><td> $1 2 . 0 1 { \scriptstyle \pm 5 . 2 1 }$ </td><td> $2 . 4 1 \pm 2 . 5 4$ </td><td> $3 . 2 0 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $3 . 0 6 { \pm } 0 . 4 8$ </td><td>2.93±0.51</td><td> $5 . 3 9 \pm 4 . 1 8$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>pFedFDA [14]</td><td>NeurIPS&#x27;24</td><td> $4 6 . 0 4 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $2 5 . 6 4 \pm 0 . 2 8$ </td><td> $1 4 . 2 7 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $2 1 . 6 5 { \scriptstyle \pm 1 . 1 5 }$ </td><td> $9 . 6 2 \pm 1 . 8 8$ </td><td> $5 4 . 7 4 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $2 1 . 2 0 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $1 2 . 3 5 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $1 5 . 5 4 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $7 . 1 7 { \scriptstyle \pm 0 . 4 4 }$ </td></tr><tr><td>FedVLS [31]</td><td>AAAI&#x27;25</td><td> $3 1 . 2 0 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $3 0 . 7 9 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $3 3 . 4 1 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $2 9 . 5 1 { \pm } 1 . 6 0$ </td><td> $1 9 . 0 6 \pm 1 . 0 9$ </td><td> $2 9 . 8 7 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $2 9 . 3 8 { \pm } 1 . 0 1$ </td><td> $3 0 . 7 0 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $2 4 . 8 6 \pm 1 . 1 0$ </td><td> $1 4 . 7 6 { \scriptstyle \pm 0 . 3 7 }$ </td></tr><tr><td>FedDr+ [40]</td><td>TMLR&#x27;25</td><td> $3 9 . 3 6 \pm 0 . 4 1$ </td><td> $3 9 . 1 3 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $3 6 . 4 2 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $4 2 . 5 7 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $2 5 . 2 1 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $3 4 . 2 6 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $3 4 . 5 9 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $3 3 . 1 1 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $3 1 . 9 6 \pm 0 . 9 5$ </td><td> $2 0 . 6 7 { \scriptstyle \pm 0 . 7 3 }$ </td></tr><tr><td>BTFL [12]</td><td>KDD&#x27;25</td><td> $4 0 . 7 4 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $3 2 . 5 0 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $2 7 . 2 6 \pm 1 . 0 1$ </td><td> $3 4 . 0 8 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $2 0 . 8 5 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $5 0 . 1 7 { \scriptstyle \pm 0 . 5 4 }$ </td><td>29.64±0.25</td><td> $2 5 . 1 3 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $2 9 . 1 7 { \pm } 1 . 0 9$ </td><td> $1 7 . 9 1 { \scriptstyle \pm 0 . 7 0 }$ </td></tr><tr><td>HPFL [8]</td><td>ICLR&#x27;25</td><td> $4 3 . 8 8 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $3 2 . 9 2 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $3 1 . 6 6 \pm 0 . 6 0$ </td><td> $3 4 . 0 1 { \scriptstyle \pm 0 . 9 7 }$ </td><td> $2 0 . 4 7 { \scriptstyle \pm 3 . 2 0 }$ </td><td>53.13±0.82</td><td> $2 7 . 2 4 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $2 6 . 8 9 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $2 4 . 2 6 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $1 4 . 7 8 { \scriptstyle \pm 1 . 1 3 }$ </td></tr><tr><td>FedYoYo [41]</td><td>ICCV&#x27;25</td><td> $3 5 . 7 1 \pm 1 . 6 5$ </td><td> $3 4 . 5 9 { \scriptstyle \pm 1 . 7 2 }$ </td><td> $3 5 . 9 0 { \scriptstyle \pm 2 . 0 8 }$ </td><td> $3 1 . 7 5 { \scriptstyle \pm 1 . 4 1 }$ </td><td>13.17±3.08</td><td>39.97±1.35</td><td> $3 9 . 2 5 { \scriptstyle \pm 1 . 6 9 }$ </td><td> $3 9 . 0 1 { \scriptstyle \pm 1 . 5 9 }$ </td><td> $4 0 . 9 0 { \scriptstyle \pm 1 . 7 8 }$ </td><td> $2 8 . 3 9 { \scriptstyle \pm 5 . 4 8 }$ </td></tr><tr><td>FedLog [15]</td><td>TMLR&#x27;26</td><td> $3 5 . 2 6 { \pm } 0 . 1 0$ </td><td> $1 0 . 1 9 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $0 . 0 1 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 2 8 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>47.81±0.17</td><td> $8 . 3 8 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 1 5 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>pFedGM [16]</td><td>arXiv&#x27;26</td><td> $4 5 . 5 3 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $2 5 . 4 0 { \scriptstyle \pm 0 . 2 0 }$ </td><td>6.14±0.51</td><td> $2 2 . 5 7 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $4 . 2 1 { \pm } 0 . 3 8$ </td><td> $5 4 . 7 5 { \scriptstyle \pm 0 . 2 3 } ^ { * }$ </td><td>14.83±0.22</td><td>1.46±0.20</td><td> $1 3 . 3 1 { \pm } 1 . 3 7$ </td><td> $0 . 7 0 { \scriptstyle \pm 0 . 1 0 }$ </td></tr><tr><td colspan="10">Statistical classifiers assembled on a vanilla backbone (FedAvg +)</td><td></td></tr><tr><td> $+ \ C C \mathrm { V R } \ [ 2 2 ]$ </td><td>NeurIPS&#x27;21</td><td>35.09±0.40 34.56±0.74</td><td></td><td> $3 4 . 8 0 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $3 4 . 6 4 { \scriptstyle \pm 1 . 3 7 }$ </td><td> $2 2 . 3 3 { \scriptstyle \pm 2 . 9 7 }$ </td><td>33.56±0.41</td><td> $3 3 . 4 8 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $3 3 . 8 4 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $3 0 . 9 8 \pm 1 . 0 5$ </td><td> $1 8 . 8 3 { \scriptstyle \pm 0 . 7 3 }$ </td></tr><tr><td>+ FedNCM [42]</td><td>NeurIPS&#x27;23</td><td> $3 6 . 1 4 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $3 5 . 4 7 { \scriptstyle \pm 0 . 6 6 }$ </td><td>35.52±0.67</td><td> $3 6 . 1 1 { \scriptstyle \pm 1 . 4 0 }$ </td><td> $2 4 . 2 7 { \scriptstyle \pm 3 . 1 0 }$ </td><td> $3 4 . 2 4 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $3 4 . 2 2 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $3 4 . 5 3 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $3 0 . 2 7 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $1 8 . 7 8 { \scriptstyle \pm 1 . 0 0 }$ </td></tr><tr><td>+ Fed3R [43]</td><td>ICML&#x27;24</td><td> $3 4 . 2 7 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $3 3 . 6 7 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $3 2 . 9 1 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $3 5 . 0 7 { \scriptstyle \pm 1 . 8 3 }$ </td><td> $2 1 . 7 2 { \scriptstyle \pm 3 . 3 8 }$ </td><td> $3 3 . 1 0 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $3 2 . 8 7 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $3 2 . 9 7 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $2 8 . 7 8 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $1 6 . 7 8 { \scriptstyle \pm 1 . 3 9 }$ </td></tr><tr><td>+ FedCGS [44]</td><td> $\mathrm { \ A A A I ^ { \prime } } 2 5$ </td><td> $3 3 . 5 4 { \scriptstyle \pm 0 . 1 0 }$ </td><td>33.05±0.59</td><td> $3 3 . 6 7 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $3 3 . 2 1 \pm 1 . 0 7$ </td><td> $2 1 . 6 7 { \scriptstyle \pm 3 . 3 8 }$ </td><td> $3 2 . 6 0 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $3 2 . 2 4 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $3 2 . 7 8 { \pm } 0 . 5 8$ </td><td> $2 8 . 8 5 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $1 6 . 5 0 { \scriptstyle \pm 1 . 5 9 }$ </td></tr><tr><td> $+ \mathrm { F e d C O F } \ [ 4 6 ]$ </td><td> $\mathrm { N e u r I P S } ^ { \prime } 2 5$ </td><td> $3 5 . 1 9 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $3 4 . 5 4 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $3 4 . 2 0 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $3 5 . 7 2 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $2 3 . 8 9 { \scriptstyle \pm 3 . 7 3 }$ </td><td> $3 3 . 4 3 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $3 2 . 9 7 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $3 3 . 1 0 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $2 9 . 2 9 { \scriptstyle \pm 1 . 1 9 }$ </td><td> $1 7 . 8 9 { \scriptstyle \pm 0 . 7 5 }$ </td></tr><tr><td> $+ \mathrm { G H - L D A } \left[ 4 5 \right]$ </td><td>ICLR&#x27;26</td><td> $3 4 . 5 6 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $3 4 . 1 0 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $3 4 . 8 0 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $3 3 . 9 3 { \scriptstyle \pm 0 . 9 8 }$ </td><td> $2 2 . 7 8 { \scriptstyle \pm 3 . 2 6 }$ </td><td> $3 3 . 5 0 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $3 3 . 0 8 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $3 3 . 6 0 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $3 0 . 3 2 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $1 8 . 2 8 { \scriptstyle \pm 0 . 2 5 }$ </td></tr><tr><td colspan="10">Statistical classifiers assembled on the SOTA backbone (FedYoYo +)</td></tr><tr><td></td><td> $\mathrm { N e u r I P S ^ { \prime } } 2 1$ </td><td></td><td></td><td>50.88±0.55*</td><td> $5 5 . 5 8 { \scriptstyle \pm 1 . 1 3 }$ </td><td> $\mathbf { 4 1 . 1 0  { \pm 1 . 3 6 } } ^ { \mathbf { * } }$ </td><td> $4 9 . 9 5 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $4 9 . 7 1 \pm 0 . 3 1$ </td><td> $4 9 . 5 7 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $4 7 . 4 4 { \scriptstyle \pm 0 . 3 5 }$ </td><td></td></tr><tr><td> $+ \ C C \mathrm { V R } \ [ 2 2 ]$   $+ \mathrm { F e d N C M } \left[ 4 2 \right]$ </td><td>NeurIPS&#x27;23</td><td> $5 2 . 3 8 { \pm } 0 . 2 0 $   $4 9 . 3 6 \pm 0 . 3 3$ </td><td> ${ \bf 5 1 . 5 6 } { \scriptstyle \pm 0 . 3 3 } ^ { \ast }$   $4 8 . 5 1 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $4 7 . 1 1 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $5 3 . 7 3 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $3 6 . 4 3 { \scriptstyle \pm 1 . 1 8 }$ </td><td> $4 5 . 6 1 \pm 0 . 2 4$ </td><td> $4 4 . 9 5 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $4 4 . 5 0 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $4 2 . 4 8 \pm 0 . 9 7$ </td><td> $3 6 . 2 2 { \scriptstyle \pm 0 . 6 7 }$ </td></tr><tr><td> $+ \operatorname { F e d } 3 \mathrm { R } \left[ 4 3 \right]$ </td><td> $\mathrm { I C M L } ^ { \prime } 2 4$ </td><td> $5 2 . 4 3 { \scriptstyle \pm 0 . 0 7 } ^ { * }$ </td><td> $5 1 . 2 6 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $4 9 . 6 9 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $5 6 . 4 6 { \scriptstyle \pm 1 . 4 3 } ^ { * }$ </td><td> $3 6 . 3 9 \pm 1 . 0 8$ </td><td> $5 0 . 0 0 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $\mathbf { 4 9 . 7 8 \pm 0 . 3 9 } ^ { \ast }$ </td><td> $4 9 . 4 0 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $4 6 . 4 0 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $2 8 . 1 1 { \scriptstyle \pm 0 . 4 2 }$   $3 1 . 7 2 { \scriptstyle \pm 1 . 0 8 }$ </td></tr><tr><td> $+ \mathrm { F e d C G S } \ [ 4 4 ]$ </td><td> $\mathrm { \ A A A I ^ { \prime } } 2 5$ </td><td> $5 1 . 7 7 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $5 0 . 5 1 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $5 0 . 5 2 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $5 2 . 2 9 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $4 0 . 9 2 \pm 1 . 3 2$ </td><td> $4 9 . 9 1 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $4 9 . 7 2 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $\mathbf { 4 9 . 6 8 \pm 0 . 7 7 } ^ { \ast }$ </td><td> $4 8 . 3 6 { \scriptstyle \pm 0 . 6 0 } ^ { * }$ </td><td> $3 8 . 4 9 { \scriptstyle \pm 1 . 4 4 } ^ { * }$ </td></tr><tr><td>+ FedCOF [46]</td><td>NeurIPS&#x27;25</td><td> $5 0 . 9 2 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $5 0 . 1 0 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $4 8 . 8 8 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $5 5 . 1 1 \pm 0 . 4 8$ </td><td> $3 9 . 6 8 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $4 7 . 5 9 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $4 6 . 8 4 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $4 6 . 3 6 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $4 5 . 3 0 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $3 3 . 6 1 { \scriptstyle \pm 0 . 9 8 }$ </td></tr><tr><td> $+ \mathrm { G H - L D A } \left[ 4 5 \right]$ </td><td> $\mathrm { I C L R } ^ { \prime } 2 6$ </td><td> $5 1 . 3 9 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $5 0 . 4 1 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $4 9 . 9 3 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $5 2 . 6 1 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $3 9 . 9 7 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $4 9 . 5 8 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $4 9 . 0 2 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $4 8 . 9 2 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $4 7 . 0 3 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $3 6 . 8 2 { \scriptstyle \pm 1 . 1 1 }$ </td></tr><tr><td></td><td></td><td> $\mathbf { 5 5 . 6 1 \pm 0 . 3 2 } ^ { \dagger }$ </td><td></td><td></td><td> ${ \phantom { - } 5 8 . 0 0 \pm 0 . 5 2 } ^ { \dagger }$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">FedGRC (Ours)</td><td>+3.18 +4.15</td><td>55.71±0.65†</td><td> $5 5 . 7 5 { \scriptstyle \pm 0 . 7 8 } ^ { \dagger }$  +4.87</td><td> $+ 1 . 5 4$ </td><td> $4 7 . 5 6 \pm 1 . 8 3 ^ { \dagger }$   $+ 6 . 4 6$ </td><td> $5 3 . 2 5 { \scriptstyle \pm 0 . 3 5 }$  -3.72</td><td> $5 3 . 2 2 { \scriptstyle \pm 0 . 2 6 } ^ { \dagger }$  +3.44</td><td> ${ \mathbf { 5 3 . 6 1 } } \pm \mathrm { 0 . 3 2 } ^ { \dagger }$  +3.93</td><td> $5 2 . 9 2 { \scriptstyle \pm 0 . 7 1 } ^ { \dagger }$  +4.56</td><td> $4 1 . 8 3 \pm 1 . 5 3 ^ { \dagger }$  +3.34</td></tr></table>

With the aggregated linear head retained, CGRL improves all five metrics on FedRS. On CIFAR-100 Dir(0.3), adding CGRL while retaining the aggregated linear head reduces F@25 from 37.77% to 29.17%; replacing that head with MMS raises it to 46.93%. MMS also improves all five metrics on FedRS, including a 15.32% gain in F@25 over the CGRL-only stage. These comparisons support matching the statistical readout to geometric representation learning.

tudes. Used separately, directional guidance produces larger global accuracy gains than radial guidance on both datasets. On CIFAR-100 Dir(0.3), radial guidance alone improves global accuracy by 0.90% while leaving F@25 nearly unchanged. Adding radial guidance improves all five metrics over directional guidance alone, including rare accuracy from 57.13% to 58.00% and F@25 from 46.44% to 47.56%. These gains support complementary magnitude regulation and directional discrimination.

SASP further improves local adaptation and recognition floors through support-dependent mean updates, completing the transition from shared representations to personalized classification. We next examine each module’s design with the other configurations unchanged.

## 5.4.2 Ablation of CGRL: Directional Alignment and Magnitude Regulation for Statistical Classification

On FedRS, adding radial guidance to directional guidance further improves global accuracy by 0.38% and F@25 by 2.23%, while reducing local and rare accuracy by 0.12% and 0.17%. The magnitude penalty regularizes within-class lognorm dispersion and the mean log norm’s deviation from the shared reference (Lemma 4.1). It therefore complements directional guidance by regulating the magnitude structure retained in the class means and covariance used for statistical classification.

Table 6 separates the effects of directional and radial guidance, with cross-entropy training and the MMS+SASP readout retained in every configuration. Within each dataset block, the rows correspond to (i) neither guidance component; (ii) radial guidance only; (iii) directional guidance only; and (iv) both components. Directional guidance combines supervised contrastive learning with shared class directions, while radial guidance regulates feature magni-

Using the same full MMS+SASP readout, CGRL improves global accuracy and F@25 over cross-entropy training by 6.15% and 7.37% on CIFAR-100 Dir(0.3), and by 4.68% and 5.12% on FedRS. These gains under the same MMS+SASP configuration demonstrate the contribution of geometric consensus to the deployed statistical classifier.

TABLE 3: Comprehensive comparison on FedRS [7] with 8 remote-sensing sources, 15 classes, and 135 clients. All methods are trained using ResNet-18. Symbols: (†) denotes the best, (∗) the second-best, and (∆) the gain over the best baseline.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="4">Overall performance</td><td colspan="4">Worst 25% of clients</td></tr><tr><td>local</td><td>global</td><td>unseen</td><td>rare</td><td>L@25</td><td>U@25</td><td>R@25</td><td>F@25</td></tr><tr><td colspan="10">Complete FL methods (method-specific training and classifiers)</td></tr><tr><td>FedAvg [56]</td><td>AISTATS&#x27;17</td><td> $5 9 . 9 \breve { 7 } \pm 0 . 2 8$ </td><td> $6 \dot { 1 } . 2 7 \pm 0 . 4 9$ </td><td> $5 9 . 1 5 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $6 4 . 3 2 { \scriptstyle \pm 2 . 0 3 }$ </td><td> $4 1 . 4 9 { \scriptstyle \pm 3 . 5 9 }$ </td><td> $4 0 . 5 6 { \scriptstyle \pm 4 . 7 4 }$ </td><td> $4 7 . 8 7 { \scriptstyle \pm 2 . 9 8 }$ </td><td> $3 7 . 1 0 { \scriptstyle \pm 3 . 2 8 }$ </td></tr><tr><td>FedRep [1]</td><td>ICML&#x27;21</td><td> $7 1 . 3 0 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $2 5 . 1 9 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 8 . 9 3 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $6 2 . 4 8 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 . 5 2 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>FedROD [2]</td><td>ICLR&#x27;22</td><td> $7 1 . 6 2 \pm 0 . 1 1$ </td><td> $6 2 . 2 6 { \pm } 2 . 1 9$ </td><td> $6 3 . 1 3 { \scriptstyle \pm 3 . 7 1 }$ </td><td> $6 2 . 6 1 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $6 1 . 6 3 { \pm } 1 . 0 3$ </td><td> $5 0 . 8 0 { \scriptstyle \pm 4 . 4 8 }$ </td><td> $4 9 . 3 7 { \scriptstyle \pm 2 . 5 3 }$ </td><td> $4 6 . 6 3 { \scriptstyle \pm 3 . 2 6 }$ </td></tr><tr><td>FedLC [30]</td><td>ICML&#x27;22</td><td> $5 7 . 1 0 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $6 1 . 7 7 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $6 3 . 2 1 \pm 0 . 2 5$ </td><td> $6 1 . 5 6 { \pm } 1 . 5 5$ </td><td> $3 9 . 7 4 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $4 8 . 5 9 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $4 7 . 9 5 { \scriptstyle \pm 2 . 0 4 }$ </td><td> $3 7 . 8 8 \pm 0 . 4 9$ </td></tr><tr><td>FedTHE [11]</td><td>ICLR&#x27;23</td><td> $6 7 . 0 3 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $5 8 . 2 1 { \scriptstyle \pm 1 . 7 6 }$ </td><td> $5 7 . 2 0 { \scriptstyle \pm 2 . 2 7 }$ </td><td> $6 0 . 5 0 { \scriptstyle \pm 1 . 5 1 }$ </td><td> $5 4 . 5 3 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $4 0 . 7 6 \pm 3 . 2 7$ </td><td> $4 5 . 5 8 { \pm } 1 . 6 9$ </td><td> $3 9 . 2 6 \pm 3 . 1 5$ </td></tr><tr><td>FedETF [21]</td><td>ICCV&#x27;23</td><td> $5 9 . 9 9 { \scriptstyle \pm 1 . 4 3 }$ </td><td> $6 0 . 6 0 { \scriptstyle \pm 1 . 8 9 }$ </td><td> $5 9 . 2 8 { \pm } 1 . 1 7$ </td><td> $6 2 . 4 4 { \scriptstyle \pm 3 . 6 9 }$ </td><td> $3 9 . 0 1 { \scriptstyle \pm 7 . 6 0 }$ </td><td> $4 0 . 5 5 { \scriptstyle \pm 8 . 1 2 }$ </td><td> $4 5 . 6 3 { \scriptstyle \pm 3 . 6 8 }$ </td><td> $3 5 . 7 4 { \scriptstyle \pm 6 . 7 9 }$ </td></tr><tr><td>FedFA [20]</td><td>TMC&#x27;24</td><td> $5 2 . 1 5 { \pm } 4 . 6 9$ </td><td> $5 2 . 3 0 { \scriptstyle \pm 5 . 8 8 }$ </td><td> $4 7 . 2 7 { \pm } 7 . 3 9$ </td><td> $5 7 . 8 9 { \scriptstyle \pm 4 . 3 4 }$ </td><td> $3 3 . 0 0 { \scriptstyle \pm 4 . 3 6 }$ </td><td> $3 4 . 7 8 { \scriptstyle \pm 1 1 . 1 4 }$ </td><td> $4 2 . 3 0 { \scriptstyle \pm 6 . 3 3 }$ </td><td> $2 9 . 7 4 { \scriptstyle \pm 7 . 9 4 }$ </td></tr><tr><td>pFedFDA [14]</td><td>NeurIPS&#x27;24</td><td> $7 0 . 2 7 { \scriptstyle \pm 1 . 1 4 }$ </td><td> $4 7 . 5 9 \pm 1 . 0 3$ </td><td> $2 4 . 9 6 \pm 2 . 5 5$ </td><td> $4 9 . 5 7 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $5 9 . 4 5 { \scriptstyle \pm 1 . 3 7 }$ </td><td> $1 5 . 6 5 { \scriptstyle \pm 3 . 3 2 }$ </td><td> $3 3 . 5 3 { \pm } 1 . 1 0$ </td><td> $1 5 . 6 0 { \scriptstyle \pm 3 . 2 9 }$ </td></tr><tr><td>FedVLS [31]</td><td> $\mathrm { \ A A A I ^ { \prime } } 2 5$ </td><td> $5 3 . 8 1 \pm 0 . 7 5$ </td><td> $5 7 . 8 9 { \scriptstyle \pm 0 . 9 9 }$ </td><td> $5 7 . 6 3 { \pm } 1 . 8 1 $ </td><td> $5 8 . 5 9 { \scriptstyle \pm 2 . 1 4 }$ </td><td> $3 3 . 6 3 { \pm } 1 . 7 2$ </td><td> $4 5 . 6 7 { \scriptstyle \pm 1 . 8 3 }$ </td><td> $4 5 . 2 5 { \scriptstyle \pm 2 . 1 6 }$ </td><td> $3 2 . 6 0 { \scriptstyle \pm 1 . 6 6 }$ </td></tr><tr><td>FedDr+ [40]</td><td>TMLR&#x27;25</td><td> $6 2 . 1 4 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $6 2 . 1 9 { \scriptstyle \pm 1 . 2 7 }$ </td><td> $5 9 . 7 6 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $6 5 . 9 8 { \scriptstyle \pm 2 . 1 9 }$ </td><td> $4 2 . 8 6 \pm 2 . 2 3$ </td><td> $3 9 . 5 9 { \scriptstyle \pm 5 . 0 3 }$ </td><td> $4 8 . 9 0 { \scriptstyle \pm 2 . 2 8 }$ </td><td> $3 7 . 3 1 \pm 3 . 8 6$ </td></tr><tr><td>BTFL [12]</td><td>KDD&#x27;25</td><td> $6 7 . 4 0 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $5 7 . 9 9 \pm 2 . 2 5$ </td><td> $4 7 . 9 8 \pm 3 . 4 2$ </td><td> $6 3 . 7 5 { \scriptstyle \pm 1 . 9 5 }$ </td><td> $5 4 . 8 2 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $2 9 . 8 9 { \scriptstyle \pm 5 . 3 5 }$ </td><td> $4 7 . 3 3 \pm 2 . 5 2$ </td><td> $2 9 . 7 4 { \scriptstyle \pm 5 . 3 1 }$ </td></tr><tr><td>HPFL [8]</td><td>ICLR&#x27;25</td><td> $6 5 . 9 4 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $6 1 . 7 0 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $5 9 . 5 7 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $6 4 . 6 5 { \scriptstyle \pm 1 . 7 2 }$ </td><td> $5 3 . 5 1 \pm 1 . 0 5$ </td><td> $4 3 . 0 3 { \scriptstyle \pm 3 . 3 1 }$ </td><td> $4 9 . 8 7 { \scriptstyle \pm 1 . 3 3 }$ </td><td> $4 1 . 8 4 \pm 2 . 3 8$ </td></tr><tr><td>FedYoYo [41]</td><td>ICCV&#x27;25</td><td> $6 9 . 3 6 { \pm } 1 . 1 6 $ </td><td> $6 5 . 0 1 \pm 1 . 4 7$ </td><td> $5 9 . 3 8 { \pm } 2 . 5 0 $ </td><td> $7 1 . 5 5 { \scriptstyle \pm 1 . 2 9 }$ </td><td> $5 4 . 1 2 { \scriptstyle \pm 3 . 2 0 }$ </td><td> $5 0 . 1 7 { \scriptstyle \pm 9 . 8 0 }$ </td><td> $5 5 . 2 6 \pm 2 . 2 6$ </td><td> $4 5 . 6 6 \pm 8 . 3 7$ </td></tr><tr><td>FedLog [15]</td><td>TMLR&#x27;26</td><td> $5 2 . 3 1 \pm 1 . 2 7$ </td><td> $1 6 . 8 8 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $0 . 0 2 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 1 . 9 2 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $4 1 . 1 1 \pm 1 . 1 1$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 7 2 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>pFedGM [16]</td><td> $\mathrm { \ a r { X i v ^ { \prime } 2 6 } }$ </td><td> $6 7 . 3 4 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $4 2 . 4 2 \pm 1 . 3 9$ </td><td> $1 0 . 0 9 { \scriptstyle \pm 1 . 5 0 }$ </td><td> $5 3 . 2 5 { \scriptstyle \pm 1 . 5 5 }$ </td><td> $5 4 . 7 1 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $4 . 8 8 \pm 1 . 1 7$ </td><td> $3 8 . 2 8 { \scriptstyle \pm 2 . 7 1 }$ </td><td> $4 . 8 8 { \scriptstyle \pm 1 . 1 7 }$ </td></tr><tr><td colspan="10">Statistical classifiers assembled on a vanilla backbone  $( F e d A v g + )$ </td></tr><tr><td>+ CCVR [22]</td><td> $\mathrm { N e u r I P S ^ { \prime } } 2 1$ </td><td> $6 0 . 7 3 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $6 1 . 0 \bar { 8 } { \pm } 0 . 8 1$ </td><td> $5 8 . 6 2 { \scriptstyle \pm 1 . 2 9 }$ </td><td> $6 4 . 6 0 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $4 8 . 5 1 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $5 0 . 8 7 { \scriptstyle \pm 3 . 8 9 }$ </td><td> $5 3 . 6 2 { \scriptstyle \pm 1 . 3 1 }$ </td><td> $4 5 . 8 1 \pm 1 . 7 5$ </td></tr><tr><td>+ FedNCM [42]</td><td>NeurIPS&#x27;23</td><td> $6 0 . 4 9 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $6 0 . 8 7 { \scriptstyle \pm 1 . 0 1 }$ </td><td> $5 9 . 0 7 { \scriptstyle \pm 1 . 3 5 }$ </td><td> $6 4 . 0 5 { \scriptstyle \pm 0 . 9 8 }$ </td><td> $4 6 . 1 5 { \scriptstyle \pm 1 . 1 5 }$ </td><td> $5 0 . 8 2 { \scriptstyle \pm 4 . 7 7 }$ </td><td> $5 1 . 3 2 { \pm } 1 . 4 0$ </td><td> $4 3 . 5 7 { \scriptstyle \pm 1 . 9 1 }$ </td></tr><tr><td>+ Fed3R [43]</td><td>ICML&#x27;24</td><td> $6 0 . 6 9 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $6 0 . 7 5 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $5 7 . 9 1 \pm 0 . 7 3$ </td><td> $6 4 . 4 8 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $4 7 . 5 7 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $4 6 . 3 9 { \scriptstyle \pm 1 . 1 8 }$ </td><td> $5 1 . 5 7 { \pm } 1 . 9 9$ </td><td> $4 3 . 5 9 { \scriptstyle \pm 0 . 3 1 }$ </td></tr><tr><td>+ FedCGS [44]</td><td> $\mathrm { \ A A A I ^ { \prime } } 2 5$ </td><td> $6 0 . 1 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $5 7 . 5 1 \pm 1 . 1 2$ </td><td> $5 2 . 6 6 \pm 1 . 5 1$ </td><td> $6 2 . 9 0 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $4 7 . 4 8 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $4 6 . 3 3 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $5 2 . 2 2 { \scriptstyle \pm 1 . 3 5 }$ </td><td> $4 3 . 9 4 { \scriptstyle \pm 0 . 3 8 }$ </td></tr><tr><td> $+ \mathrm { F e d C O F } [ 4 6 ]$ </td><td> $\mathrm { N e u r I P S } ^ { \prime } 2 5$ </td><td> $5 9 . 6 5 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $6 0 . 3 1 { \scriptstyle \pm 1 . 1 2 }$ </td><td> $5 8 . 7 0 { \scriptstyle \pm 1 . 3 1 }$ </td><td> $6 2 . 8 1 \pm 0 . 6 7$ </td><td> $4 7 . 4 2 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $4 5 . 3 5 { \scriptstyle \pm 2 . 2 7 }$ </td><td> $5 0 . 9 2 { \scriptstyle \pm 1 . 1 3 }$ </td><td> $4 2 . 9 7 { \scriptstyle \pm 1 . 4 2 }$ </td></tr><tr><td> $+ \mathrm { G H - L D A } \left[ 4 5 \right]$ </td><td>ICLR&#x27;26</td><td> $6 0 . 7 0 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $5 9 . 8 1 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $5 5 . 9 8 \pm 0 . 2 7$ </td><td> $6 4 . 2 4 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $4 9 . 0 1 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $5 0 . 9 3 { \scriptstyle \pm 2 . 2 4 }$ </td><td> $5 5 . 1 5 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $4 7 . 1 5 { \scriptstyle \pm 0 . 6 8 }$ </td></tr><tr><td colspan="10">Statistical classifiers assembled on the SOTA backbone  $( F e d Y o Y o + )$ </td></tr><tr><td>+ CCVR [22]</td><td>NeurIPS&#x27;21</td><td> $8 4 . 7 4 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $8 2 . 8 5 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $8 2 . 1 1 \pm 1 . 0 7$ </td><td> $8 4 . 3 4 { \scriptstyle \pm 1 . 2 4 }$ </td><td> $7 6 . 3 8 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $\mathbf { 8 0 . 7 0 { \scriptstyle \pm 0 . 6 7 } ^ { * } }$ </td><td> $7 8 . 2 4 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $7 4 . 6 6 \pm 0 . 0 2$ </td></tr><tr><td>+ FedNCM [42]</td><td>NeurIPS&#x27;23</td><td> $8 1 . 7 1 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $8 1 . 4 8 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $8 1 . 6 5 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $8 2 . 4 4 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $6 8 . 1 1 \pm 1 . 9 4$ </td><td> $7 9 . 2 0 { \scriptstyle \pm 2 . 1 5 }$ </td><td> $7 4 . 4 9 { \scriptstyle \pm 1 . 6 3 }$ </td><td> $6 7 . 1 9 { \scriptstyle \pm 1 . 9 6 }$ </td></tr><tr><td>+ Fed3R [43]</td><td> $\mathrm { I C M L } ^ { \prime } 2 4$ </td><td> $8 6 . 9 6 { \scriptstyle \pm 0 . 0 4 } ^ { * }$ </td><td> $8 3 . 7 8 { \scriptstyle \pm 0 . 6 8 } ^ { * }$ </td><td> $8 3 . 8 8 { \scriptstyle \pm 0 . 6 7 } ^ { * }$ </td><td> $8 4 . 7 8 { \scriptstyle \pm 1 . 2 0 }$ </td><td> $7 9 . 6 8 { \pm } 1 . 1 5$ </td><td> $7 6 . 7 5 { \scriptstyle \pm 3 . 3 4 }$ </td><td> $7 7 . 5 7 { \pm } 1 . 4 4$ </td><td> $7 4 . 1 0 { \scriptstyle \pm 1 . 8 4 }$ </td></tr><tr><td>+ FedCGS [44]</td><td> $\mathrm { \ A A A I ^ { \prime } } 2 5$ </td><td> $8 6 . 5 6 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $8 3 . 7 5 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $8 2 . 0 5 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $8 5 . 9 0 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $7 9 . 7 2 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $7 9 . 8 6 \pm 1 . 9 3 $ </td><td> $8 0 . 8 2 { \scriptstyle \pm 0 . 9 3 } ^ { * }$ </td><td> $7 7 . 0 4 \pm 1 . 4 8 ^ { * }$ </td></tr><tr><td>+ FedCOF [46]</td><td>NeurIPS&#x27;25</td><td> $8 4 . 4 1 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $8 3 . 5 4 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $8 3 . 6 8 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $8 4 . 4 0 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $7 6 . 3 6 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $7 9 . 9 7 { \scriptstyle \pm 4 . 0 9 }$ </td><td> $7 8 . 1 4 \pm 1 . 4 2$ </td><td> $7 4 . 2 2 { \scriptstyle \pm 1 . 1 6 }$ </td></tr><tr><td>+ GH-LDA [45]</td><td> $\mathrm { I C L R } ^ { \prime } 2 6$ </td><td> $8 6 . 5 9 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $8 3 . 7 8 { \scriptstyle \pm 0 . 7 2 } ^ { * }$ </td><td> $8 2 . 1 2 { \scriptstyle \pm 0 . 9 7 }$ </td><td> $\mathbf { 8 6 . 1 0 { \scriptstyle \pm 0 . 8 2 } ^ { \ast } }$ </td><td> $\mathbf { 8 0 . 0 4 } \pm 0 . 5 3 ^ { \ast }$ </td><td> $7 9 . 5 1 \pm 2 . 6 6$ </td><td> $8 0 . 5 2 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $7 6 . 8 6 \pm 1 . 3 3$ </td></tr><tr><td>FedGRC (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">∆ vs. Best</td><td> $\mathbf { 8 9 . 1 1 \pm 0 . 0 5 } ^ { \dagger }$   $+ 2 . 1 5$ </td><td> ${ 8 7 . 0 8 \pm 0 . 4 9 } ^ { \dagger }$   $+ 3 . 3 0$ </td><td> $8 7 . 3 4 { \scriptstyle \pm 0 . 4 5 } ^ { \dagger }$   $+ 3 . 4 6$ </td><td> $8 7 . 5 4 { \scriptstyle \pm 0 . 2 4 } ^ { \dagger }$   $+ 1 . 4 4$ </td><td> $8 4 . 2 8 { \scriptstyle \pm 0 . 8 1 } ^ { \dagger }$   $+ 4 . 2 4$ </td><td> $8 5 . 5 3 { \scriptstyle \pm 0 . 7 9 } ^ { \dagger }$   $+ 4 . 8 3$ </td><td> $8 2 . 9 1 { \scriptstyle \pm 0 . 6 6 } ^ { \dagger }$   $+ 2 . 0 9$ </td><td> $\mathbf { 8 1 . 6 5 \pm 1 . 5 4 } ^ { \dagger }$   $+ 4 . 6 1$ </td></tr></table>

## 5.4.3 Ablation ofMMS: Contributions ofStatistical Geometry and Multilayer Feature Integration

TABLE 4: Recognition before and after local personalization. Before: shared predictors. After: FedETF [21], full-model adaptation; pFedGM [16], classifier adaptation with the generator unchanged; FedGRC, SASP with CGRL+MMS unchanged. The signed values in parentheses indicate changes (After minus Before).
<table><tr><td>Method</td><td>State</td><td>local</td><td>global</td><td>unseen</td><td>rare</td><td>F@25</td></tr><tr><td colspan="7">(a) CIFAR-100, Dir(0.3)</td></tr><tr><td>FedETF [21]</td><td>Before After</td><td>39.08 46.20 (+7.12)</td><td>38.77 25.58 (-13.19)</td><td>36.94 14.04 (-22.90)</td><td>40.57 18.72 (-21.85)</td><td>24.02 1.33 (-22.69)</td></tr><tr><td>pFedGM [16]</td><td>Before After</td><td>32.13 45.53 (+13.40)</td><td>32.39 25.40 (-6.99)</td><td>30.97 6.14 (-24.83)</td><td>34.03 22.57 (-11.46)</td><td>21.75 4.21 (-17.54)</td></tr><tr><td>FedGRC (Ours)</td><td>Before After</td><td>55.53 55.61 (+0.08)</td><td>55.98 55.71 (-0.27)</td><td>55.67 55.75 (+0.08)</td><td>57.83 58.00 (+0.17)</td><td>46.93 47.56 (+0.63)</td></tr><tr><td>(b) FedRS</td><td>Before</td><td>59.99</td><td>60.60</td><td>59.28</td><td></td><td></td></tr><tr><td>FedETF [21]</td><td>After</td><td>75.74 (+15.75)</td><td>35.35 (-25.25)</td><td>16.95 (-42.33)</td><td>62.44 29.81 (-32.63)</td><td>35.74 3.96 (-31.78)</td></tr><tr><td>pFedGM [16]</td><td>Before After</td><td>57.75 67.34 (+9.59)</td><td>56.38 42.42 (-13.96)</td><td>52.51 10.09 (-42.42)</td><td>60.51 53.25 (-7.26)</td><td>33.29 4.88 (-28.41)</td></tr><tr><td>FedGRC (Ours)</td><td>Before After</td><td>87.70 89.11 (+1.41)</td><td>86.89 87.08 (+0.19)</td><td>86.87 87.34 (+0.47)</td><td>87.11 87.54 (+0.43)</td><td>80.04 81.65 (+1.61)</td></tr></table>

Table 7 evaluates the statistical metric and multilayer integration with the same CGRL backbone and SASP procedure. Within each setting, the rows progress from (i) Euclidean classification with raw last-layer features to (ii) statistical classification with last-layer features and (iii) statistical classification with multilayer features, the complete MMS design. The statistical metric increases F@25 by 2.52% on CIFAR-100 Dir(0.3) and 6.25% on FedRS. Multilayer integration adds a further 1.62% and 2.22%, respectively, while also improving global and unseen accuracy in both settings. These gains reflect the complementary roles of the two MMS components: the covariance-based metric accounts for within-class variability, while multilayer integration supplies information from several representation depths. On FedRS, the two components together raise global accuracy by 2.79% and F@25 by 8.47%, improving all-class recognition alongside client recognition floors.

## 5.4.4 Ablation of SASP: Contributions of Support-Adaptive Mean Estimation and Local Class Priors

The configurations in Table 8 vary mean adaptation and class priors while using the same CGRL backbone, MMS layer scales, and shared covariance metric. Within each dataset block, the rows correspond to (i) shared means; (ii)

TABLE 5: FedGRC module construction on CIFAR-100 Dir(0.3) and FedRS: CGRL, MMS, and SASP are added sequentially. (✓) denotes inclusion. The signed values in parentheses indicate changes relative to the initial system.
<table><tr><td>SASP</td><td>MMS</td><td>CGRL</td><td></td><td>local</td><td></td><td>global</td><td>unseen</td><td></td><td>rare</td><td></td><td>F@25</td></tr><tr><td colspan="10">(a) CIFAR-100 [57], Dir(0.3), 20 clients</td><td colspan="3"></td></tr><tr><td>一</td><td></td><td></td><td>47.19</td><td></td><td>45.67</td><td></td><td>42.24</td><td></td><td>50.46</td><td></td><td>37.77</td><td></td></tr><tr><td>一</td><td>一</td><td>√</td><td>46.46</td><td>(−0.73)</td><td></td><td>46.53 (+0.86)</td><td>42.01 (-0.23)</td><td></td><td>45.83</td><td>(−4.63)</td><td>29.17</td><td>(-8.60)</td></tr><tr><td>一</td><td>√</td><td>√</td><td>55.53</td><td>(+8.34)</td><td></td><td>55.98 (+10.31)</td><td>55.67 (+13.43)</td><td></td><td>57.83</td><td>(+7.37)</td><td>46.93</td><td>(+9.16)</td></tr><tr><td>√</td><td>√</td><td>√</td><td>55.61</td><td>(+8.42)</td><td></td><td>55.71 (+10.04)</td><td>55.75 (+13.51)</td><td></td><td>58.00</td><td>(+7.54)</td><td>47.56</td><td>(+9.79)</td></tr><tr><td colspan="10">(b) FedRS [7], 8 sources, 15 classes, 135 clients (10 per round)</td><td colspan="3"></td></tr><tr><td>一</td><td></td><td></td><td>78.09</td><td></td><td>78.70</td><td></td><td>76.45</td><td></td><td>81.84</td><td></td><td>64.09</td><td></td></tr><tr><td>一</td><td></td><td>√</td><td>81.07</td><td>(+2.98)</td><td>81.03</td><td>(+2.33)</td><td>79.01 (+2.56)</td><td></td><td>84.03</td><td>(+2.19)</td><td>64.72 (+0.63)</td><td></td></tr><tr><td>一</td><td>√</td><td>√</td><td>87.70</td><td>(+9.61)</td><td>86.89</td><td>(+8.19)</td><td>86.87 (+10.42)</td><td></td><td>87.11</td><td>(+5.27)</td><td>80.04 (+15.95)</td><td></td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>89.11 (+11.02)</td><td>87.08</td><td>(+8.38)</td><td>87.34 (+10.89)</td><td></td><td>87.54</td><td>(+5.70)</td><td></td><td>81.65 (+17.56)</td></tr></table>

TABLE 6: Ablation of directional and radial guidance in CGRL. (✓) denotes inclusion. All configurations retain cross-entropy training and the same MMS+SASP readout. The signed values in parentheses indicate changes relative to training without either guidance component.
<table><tr><td>Radial Direction</td><td>local</td><td>global</td><td>unseen</td><td>rare</td><td>F@25</td></tr><tr><td>(a) CIFAR-100, Dir(0.3), 20 clients</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>一</td><td>一</td><td>49.60 49.97</td><td>49.56 50.46</td><td>50.29 51.42</td><td>40.19 40.17</td></tr><tr><td>√</td><td>(+0.37)</td><td>(+0.90)</td><td>51.59 (+1.30)</td><td>51.21 (−0.21)</td><td>(−0.02)</td></tr><tr><td>一 √</td><td>54.99 (+5.39)</td><td>55.06 (+5.50)</td><td>55.42 (+5.13)</td><td>57.13 (+5.71)</td><td>46.44 (+6.25)</td></tr><tr><td>√ √</td><td>55.61 (+6.01)</td><td>55.71 (+6.15)</td><td>55.75 (+5.46)</td><td>58.00 (+6.58)</td><td>47.56 (+7.37)</td></tr><tr><td>(b) FedRS, 135 clients</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>一</td><td>89.21</td><td>82.40</td><td>81.46</td><td>86.37</td><td>76.53</td></tr><tr><td>√</td><td>89.03 一 (−0.18)</td><td>83.53 (+1.13)</td><td>84.34 (+2.88)</td><td>85.93 (−0.44)</td><td>78.68 (+2.15)</td></tr><tr><td>一</td><td>89.23 √ (+0.02)</td><td>86.70 (+4.30)</td><td>87.29 (+5.83)</td><td>87.71 (+1.34)</td><td>79.42 (+2.89)</td></tr><tr><td>√</td><td>√</td><td>89.11 87.08 (−0.10) (+4.68)</td><td>87.34 (+5.88)</td><td>87.54 (+1.17)</td><td>81.65 (+5.12)</td></tr></table>

direct local means; (iii) direct local means with a local class prior, as used in pFedFDA [14]; (iv) count-based shrinkage of local means with a local class prior; and (v) count-based shrinkage of local means with equal class priors, constituting SASP. Configurations (i) and (ii) also use equal class priors; all configurations retain shared means for classes absent from local training.

Shrinkage for Rare-Class Recognition. Using local means directly improves local accuracy but severely degrades recognition of sparsely observed classes. On CIFAR-100 Dir(0.3), replacing shared means with local estimates without shrinkage or a local prior reduces rare accuracy from 57.83% to 1.17% and F@25 from 46.93% to 0.00%. On FedRS, rare accuracy falls from 87.11% to 44.44% and F@25 from 80.04% to 9.53%. The corresponding R@25 scores of 0.00% and 9.53% identify rare-class recognition as the principal lower-tail weakness in these configurations. Retaining shared means for absent classes leaves this rare-class weakness unresolved: sparsely observed classes are still represented by means estimated from limited local samples.

With equal priors, SASP restores rare accuracy to 58.00% on CIFAR-100 Dir(0.3) and 87.54% on FedRS, with corresponding F@25 scores of 47.56% and 81.65%. Proposition 4.2 explains the role of support in the regularized mean-fitting objective: the shared reference has greater relative influence when local sample support is small. Shrinkage therefore restrains updates based on sparse observations, addressing the rare-class degradation seen with direct local estimation.

TABLE 7: Ablation of the statistical metric and multilayer features in MMS. (✓) denotes inclusion. All configurations use the same CGRL backbone and SASP procedure. The signed values in parentheses indicate changes relative to Euclidean classification using only the last-layer features.
<table><tr><td>Multi- layer</td><td>Stat. metric</td><td>local</td><td>global</td><td>unseen</td><td>rare</td><td>F@25</td></tr><tr><td colspan="7">(a) CIFAR-100, Dir(0.3), 20 clients</td></tr><tr><td>一</td><td>一</td><td>55.00</td><td>54.25</td><td>53.24</td><td>57.38</td><td>43.42</td></tr><tr><td>一</td><td>√</td><td>55.62 (+0.62)</td><td>55.31 (+1.06)</td><td>55.07 (+1.83)</td><td>57.25 (−0.13)</td><td>45.94 (+2.52)</td></tr><tr><td>√</td><td>√</td><td>55.61 (+0.61)</td><td>55.71 (+1.46)</td><td>55.75 (+2.51)</td><td>58.00 (+0.62)</td><td>47.56 (+4.14)</td></tr><tr><td colspan="7">(b) FedRS, 135 clients</td></tr><tr><td>一</td><td></td><td>85.12</td><td>84.29</td><td>84.87</td><td>85.26</td><td>73.18</td></tr><tr><td>一</td><td>√</td><td>87.99 (+2.87)</td><td>85.99 (+1.70)</td><td>86.39 (+1.52)</td><td>86.41 (+1.15)</td><td>79.43 (+6.25)</td></tr><tr><td>√</td><td>√</td><td>89.11 (+3.99)</td><td>87.08 (+2.79)</td><td>87.34 (+2.47)</td><td>87.54 (+2.28)</td><td>81.65 (+8.47)</td></tr></table>

Mean Adaptation and Class Priors. With shrinkage retained, replacing local class priors with equal priors on CIFAR-100 Dir(0.3) improves unseen and rare accuracy by 1.33% and 0.96%, respectively, and raises F@25 from 44.33% to 47.56%, at a cost of 1.60% in local accuracy. On FedRS, equal priors improve global, unseen, and rare accuracy and raise F@25 by 0.24%, while reducing local accuracy by 0.49%. The improvements extend to unseen-class recognition among weaker clients, with U@25 increasing by 1.59% and 2.09% on CIFAR-100 and FedRS, respectively. These results support separating the roles of class counts and priors: counts govern mean adaptation, while equal priors support broader recognition and stronger client floors.

In both settings, direct local means achieve higher unseen accuracy, whereas SASP yields substantially higher rare accuracy and F@25. Unseen-class predictions can change because updates to observed-class means alter competing scores, even when absent-class means remain shared. For the same fitted statistics, shrinkage reduces the perturbation budget in Corollary 4.1.

TABLE 8: Ablation of local means, count-based shrinkage, and local class priors in SASP. (✓) denotes inclusion. All configurations use the same CGRL backbone and MMS readout. Locally absent classes retain shared means; without local priors, class priors are equal. The signed values in parentheses indicate changes relative to the shared-mean classifier.
<table><tr><td>Count shrinkage</td><td>Local prior</td><td>Local mean</td><td>local</td><td></td><td>global</td><td></td><td>unseen</td><td>rare</td><td></td><td>L@25</td><td></td><td>U@25</td><td></td><td>R@25</td><td>F@25</td></tr><tr><td colspan="10">(a) CIFAR-100, Dir(0.3), 20 clients</td><td colspan="7"></td></tr><tr><td></td><td></td><td></td><td>55.53</td><td></td><td>55.98</td><td></td><td>55.67</td><td></td><td>57.83</td><td>52.75</td><td></td><td>52.71</td><td></td><td>47.67</td><td></td><td>46.93</td></tr><tr><td></td><td></td><td> $\checkmark$ </td><td>57.30</td><td>(+1.77)</td><td></td><td>45.17 (–10.81)</td><td>61.88</td><td>(+6.21)</td><td>1.17 (−56.66)</td><td>54.47</td><td>(+1.72)</td><td>59.38</td><td>(+6.67)</td><td></td><td>0.00 (-47.67)</td><td>0.00 (-46.93)</td></tr><tr><td></td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>58.91</td><td>(+3.38)</td><td></td><td>45.29 (-10.69)</td><td>60.66</td><td>(+4.99)</td><td>1.21 (−56.62)</td><td>56.19</td><td>(+3.44)</td><td>58.24</td><td>(+5.53)</td><td>0.00 (-47.67)</td><td></td><td>0.00 (−46.93)</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>57.21</td><td>(+1.68)</td><td>55.63</td><td>(−0.35)</td><td>54.42</td><td>(−1.25)</td><td>57.04 (−0.79)</td><td>54.30</td><td>(+1.55)</td><td>51.31</td><td>(−1.40)</td><td>45.67</td><td>(−2.00)</td><td>44.33 (−2.60)</td></tr><tr><td> $\checkmark$ </td><td></td><td>√</td><td>55.61</td><td>(+0.08)</td><td>55.71</td><td>(−0.27)</td><td>55.75</td><td>(+0.08)</td><td>58.00 (+0.17)</td><td>52.68</td><td>(-0.07)</td><td>52.90</td><td>(+0.19)</td><td>48.33</td><td>(+0.66)</td><td>47.56 (+0.63)</td></tr><tr><td colspan="10">(b) FedRS, 135 clients</td><td colspan="7"></td></tr><tr><td></td><td></td><td></td><td>87.70</td><td></td><td>86.89</td><td></td><td>86.87</td><td></td><td>87.11</td><td>81.73</td><td></td><td>84.35</td><td></td><td>82.59</td><td></td><td>80.04</td></tr><tr><td></td><td></td><td> $\checkmark$ </td><td>89.98</td><td>(+2.28)</td><td>80.71</td><td>(−6.18)</td><td>89.69 (+2.82)</td><td></td><td>44.44 (−42.67)</td><td>84.33</td><td>(+2.60)</td><td>88.36</td><td>(+4.01)</td><td>9.53 (–73.06)</td><td></td><td>9.53 (−70.51)</td></tr><tr><td></td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>90.43</td><td>(+2.73)</td><td>80.57</td><td>(−6.32)</td><td>89.08</td><td>(+2.21)</td><td>43.98 (−43.13)</td><td>85.29</td><td>(+3.56)</td><td>87.32</td><td>(+2.97)</td><td>8.93 (–73.66)</td><td></td><td>8.93 (−71.11)</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>89.60</td><td>(+1.90)</td><td>86.89</td><td>(+0.00)</td><td>86.50</td><td>(−0.37)</td><td>87.36 (+0.25)</td><td>84.76</td><td>(+3.03)</td><td>83.44</td><td>(−0.91)</td><td>82.74</td><td>(+0.15)</td><td>81.41 (+1.37)</td></tr><tr><td>√</td><td></td><td> $\checkmark$ </td><td>89.11</td><td>(+1.41)</td><td>87.08</td><td>(+0.19)</td><td>87.34</td><td>(+0.47)</td><td>87.54 (+0.43)</td><td>84.28</td><td>(+2.55)</td><td>85.53</td><td>(+1.18)</td><td>82.91</td><td>(+0.32)</td><td>81.65 (+1.61)</td></tr></table>

TABLE 9: Backbone–classifier cross-comparison against strong baseline combinations across all heterogeneity settings. $B _ { 0 } { : }$ FedYoYo [41]; $B _ { 1 } \colon$ : the CGRL-trained FedGRC backbone. $H _ { 0 } \colon \mathrm { C C V R }$ [22] on CIFAR-100 Dir(0.3), and FedCGS [44] elsewhere; $H _ { 1 } \colon$ MMS+SASP. The signed values in parentheses indicate gains over $B _ { 0 } H _ { 0 } ;$ bold entries identify full FedGRC. For either metric, $M _ { i j } = M ( B _ { i } , \tilde { H _ { j } } )$ , and the shaded row reports the excess gain ${ \cal I } = M _ { 1 1 } - M _ { 1 0 } - M _ { 0 1 } + { \bf \dot { M } } _ { 0 0 }$
<table><tr><td rowspan="2">Training Classifier backbone</td><td rowspan="2">head</td><td colspan="2">CIFAR-10, Dir(0.3)</td><td colspan="2">CIFAR-10, Dir(0.1)</td><td colspan="2">CIFAR-100, Dir(0.3)</td><td colspan="2">CIFAR-100, Dir(0.1)</td><td colspan="2">FedRS</td></tr><tr><td>global</td><td>F@25</td><td>global</td><td>F@25</td><td>global</td><td>F@25</td><td>global</td><td>F@25</td><td>global</td><td>F@25</td></tr><tr><td> $B _ { 0 }$ </td><td> $H _ { 0 }$ </td><td>78.59</td><td>69.72</td><td>69.03</td><td>61.03</td><td>51.56</td><td>41.10</td><td>49.72</td><td>38.49</td><td>83.75</td><td>77.04</td></tr><tr><td> $B _ { 0 }$ </td><td> $H _ { 1 }$ </td><td>78.55 (−0.04) 69.32 (−0.40)</td><td></td><td>69.84 (+0.81)</td><td>60.38 (−0.65)</td><td>51.05 (−0.51)</td><td>40.98 (−0.12)</td><td>50.00 (+0.28)</td><td>37.61 (-0.88)</td><td>84.65 (+0.90)</td><td>79.57 (+2.53)</td></tr><tr><td> $B _ { 1 }$ </td><td> $H _ { 0 }$ </td><td>81.17 (+2.58) 71.39 (+1.67)</td><td></td><td>72.09 (+3.06)</td><td>63.47 (+2.44)</td><td>54.45 (+2.89)</td><td>45.07 (+3.97)</td><td>52.08 (+2.36)</td><td>41.07 (+2.58)</td><td>85.69 (+1.94)</td><td>76.88 (−0.16)</td></tr><tr><td> $\scriptstyle { \mathbf { } } _ { B _ { 1 } }$ </td><td> $\pmb { H _ { 1 } }$ </td><td>80.76 (+2.17) 71.85 (+2.13)</td><td></td><td>73.05 (+4.02)</td><td>63.85 (+2.82)</td><td>55.71 (+4.15)</td><td>47.56 (+6.46)</td><td>53.22 (+3.50)</td><td>41.83 (+3.34)</td><td>87.08 (+3.33)</td><td>81.65 (+4.61)</td></tr><tr><td>Excess gain I</td><td></td><td>-0.37</td><td>+0.86</td><td>+0.15</td><td>+1.03</td><td>+1.77</td><td>+2.61</td><td>+0.86</td><td>+1.64</td><td>+0.49</td><td>+2.24</td></tr></table>

## 5.5 Performance Gains over Strong Baseline Combinations and Backbone–Classifier Synergy

Table 9 uses the baseline combinations with the highest F@25 in Tables 1–3 as references: the FedYoYo [41] backbone $( B _ { 0 } )$ paired with CCVR [22] on CIFAR-100 Dir(0.3) or FedCGS [44] elsewhere $( H _ { 0 } )$ . The corresponding Fed-GRC components are the CGRL-trained backbone $( { \dot { B _ { 1 } } } )$ and MMS+SASP classifier $( H _ { 1 } )$ . Replacing the backbone, the classifier, or both separates their individual effects from the gains achieved through their combination.

The complete combination achieves higher F@25 than either single-component replacement in all five settings. On CIFAR-100 Dir(0.3), replacing FedYoYo with the CGRL backbone raises F@25 by 3.97% under the same CCVR classifier. MMS+SASP then adds 2.49% on the CGRL backbone, compared with a 0.12% decrease on FedYoYo. On FedRS, replacing only the backbone changes F@25 by −0.16%, whereas replacing both components yields a 4.61% gain. These comparisons demonstrate that the gains from each component depend on its pairing with the other.

The excess gain I measures how much the joint improvement exceeds the sum of the gains from the two individual replacements. It is positive for F@25 in all five settings, ranging from 0.86% to 2.61%, and for global accuracy in four settings. Together, these results support jointly designing geometric representation learning and statistical personalization to strengthen client recognition floors.

## 6 CONCLUSION

We revisited personalized federated learning from the perspective of global recognition and personalization floors, showing that high local accuracy alone can conceal weaknesses across classes and clients. Our evaluation protocol exposes these weaknesses through complementary assessments of local adaptation, recognition across federationknown classes, and the weakest assessed abilities of poorly performing clients. Guided by this perspective, FedGRC couples geometric representation consensus with multilayer statistical classification and support-adaptive class-mean shrinkage. Extensive comparisons and ablations demonstrate improved global recognition and personalization floors across heterogeneous settings, with simultaneous gains in local adaptation on FedRS. These findings support the joint design of representations and personalized classifiers for practical deployment.

Under extreme label heterogeneity, broader recognition gains accompany substantially lower local accuracy than the strongest local baselines, highlighting the remaining tradeoff between generalization and local adaptation. FedGRC retains a shared representation and covariance metric during personalization; its shrinkage weights are determined by local class sample counts and do not explicitly account for the magnitude of client-specific domain shifts. We will explore personalization jointly guided by class support and distribution shift, alongside representation and metric learning robust to domain shift, to improve local adaptation while preserving broad recognition and personalization floors.

## REFERENCES

[1] L. Collins, H. Hassani, A. Mokhtari et al., “Exploiting shared representations for personalized federated learning,” in Proc. Int. Conf. Mach. Learn. (ICML), ser. PMLR, vol. 139, 2021, pp. 2089– 2099.

[2] H.-Y. Chen and W.-L. Chao, “On bridging generic and personalized federated learning for image classification,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[3] A. Z. Tan, H. Yu, L. Cui et al., “Towards personalized federated learning,” IEEE Trans. Neural Netw. Learn. Syst., vol. 34, no. 12, pp. 9587–9603, Dec. 2023.

[4] I. Dayan, H. R. Roth, A. Zhong et al., “Federated learning for predicting clinical outcomes in patients with COVID-19,” Nat. Med., vol. 27, no. 10, pp. 1735–1743, 2021.

[5] G. Rizzoli, M. Caligiuri, D. Shenaj et al., “When cars meet drones: Hyperbolic federated learning for source-free domain adaptation in adverse weather,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2025, pp. 1587–1596.

[6] M. Ye, X. Fang, B. Du et al., “Heterogeneous federated learning: State-of-the-art and research challenges,” ACM Comput. Surv., vol. 56, no. 3, pp. 1–44, Mar. 2024, art. no. 79.

[7] H. Zhao, P. Peng, C. Chen et al., “FedRS-Bench: Realistic federated learning datasets and benchmarks in remote sensing,” in Proc. ACM SIGKDD Int. Joint Workshop Federated Learning for Data Mining and Graph Analytics (FedKDD), Toronto, ON, Canada, 2025, arXiv:2505.08325.

[8] L. Shen, Z. Tang, L. Wu et al., “Hot-pluggable federated learning: Bridging general and personalized FL via dynamic selection,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025, pp. 102 648–102 684.

[9] J. Zhang, Y. Hua, J. Cao et al., “Eliminating domain bias for federated learning in representation space,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, 2023, pp. 14 204–14 227.

[10] J. Zhang, Y. Hua, H. Wang et al., “GPFL: Simultaneously learning global and personalized feature information for personalized federated learning,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 5018–5028.

[11] L. Jiang and T. Lin, “Test-time robust personalization for federated learning,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[12] Y. Zhou and B. Liu, “BTFL: A Bayesian-based test-time generalization method for internal and external data distributions in federated learning,” in Proc. ACM SIGKDD Conf. Knowl. Discov. Data Min. (KDD), V.1, 2025, pp. 2147–2158.

[13] S. Wu, T. Li, Z. Charles et al., “Motley: Benchmarking heterogeneity and personalization in federated learning,” arXiv preprint arXiv:2206.09262, 2022.

[14] C. J. McLaughlin and L. Su, “Personalized federated learning via feature distribution adaptation,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 37, 2024, pp. 77 038–77 059.

[15] H. Yu, G. Zhang, H. Li et al., “FedLog: Personalized federated classification with less communication and more flexibility,” Trans. Mach. Learn. Res. (TMLR), 2026.

[16] P. Hu and J. Ma, “Personalized federated learning via Gaussian generative modeling,” arXiv preprint arXiv:2603.11620, 2026.

[17] Y. Li, X. Wang, R. Zeng et al., “Federated domain generalization: A survey,” Proc. IEEE, vol. 113, no. 4, pp. 370–410, Apr. 2025.

[18] W. Huang, M. Ye, Z. Shi et al., “Federated learning for generalization, robustness, fairness: A survey and benchmark,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 9387–9406, Dec. 2024.

[19] J. Zhang, X. Wu, Y. Zhou et al., “HtFLlib: A comprehensive heterogeneous federated learning library and benchmark,” in Proc. 31st ACM SIGKDD Conf. Knowl. Discovery Data Mining (KDD), V.2, 2025, pp. 5900–5911.

[20] T. Zhou, J. Zhang, and D. H. K. Tsang, “FedFA: Federated learning with feature anchors to align features and classifiers for heterogeneous data,” IEEE Trans. Mobile Comput., vol. 23, no. 6, pp. 6731– 6742, 2024.

[21] Z. Li, X. Shang, R. He et al., “No fear of classifier biases: Neural collapse inspired federated learning with synthetic and fixed classifier,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 5296–5306.

[22] M. Luo, F. Chen, D. Hu et al., “No fear of heterogeneity: Classifier calibration for federated learning with non-IID data,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 34, 2021, pp. 5972–5984.

[23] T. Li, S. Hu, A. Beirami et al., “Ditto: Fair and robust federated learning through personalization,” in Proc. Int. Conf. Mach. Learn. (ICML), ser. PMLR, vol. 139, 2021, pp. 6357–6368.

[24] Z. Pan, C. Li, F. Yu et al., “Balancing the trade-off between global and personalized performance in federated learning,” Inf. Sci., vol. 712, Sep. 2025, art. no. 122154.

[25] J. Dogani, M. Khastkhodaei, F. Khunjush et al., “SemanticDFL: Similarity-aware pull-based personalized decentralized federated learning,” Proc. ACM Meas. Anal. Comput. Syst., vol. 10, no. 2, Jun. 2026, art. no. 51.

[26] M. Hussaini, J. Theiß, and A. Stein, “Local performance vs. out-ofdistribution generalization: An empirical analysis of personalized federated learning in heterogeneous data environments,” arXiv preprint arXiv:2510.24503, 2025.

[27] D. Chen, D. Gao, W. Kuang et al., “pFL-Bench: A comprehensive benchmark for personalized federated learning,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 35, 2022, pp. 9344–9360.

[28] G. Zhang, S. Malekmohammadi, X. Chen et al., “Proportional fairness in federated learning,” Trans. Mach. Learn. Res. (TMLR), Jan. 2023.

[29] K. Pillutla, Y. Laguel, J. Malick et al., “Federated learning with superquantile aggregation for heterogeneous data,” Mach. Learn., vol. 113, no. 5, pp. 2955–3022, May 2024.

[30] J. Zhang, Z. Li, B. Li et al., “Federated learning with label distribution skew via logits calibration,” in Proc. Int. Conf. Mach. Learn. (ICML), ser. PMLR, vol. 162, 2022, pp. 26 311–26 329.

[31] K. Guo, Y. Ding, J. Liang et al., “Exploring vacant classes in label-skewed federated learning,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 39, no. 16, 2025, pp. 16 960–16 968.

[32] Y. Zhou, J. Wang, H. Yu et al., “Rethinking sparse supervision on federated long-tailed learning,” Knowl.-Based Syst., vol. 342, Jun. 2026, art. no. 115819.

[33] Y. Tan, G. Long, L. Liu et al., “FedProto: Federated prototype learning across heterogeneous clients,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 36, no. 8, 2022, pp. 8432–8440.

[34] Y. Dai, Z. Chen, J. Li et al., “Tackling data heterogeneity in federated learning with class prototypes,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 37, no. 6, 2023, pp. 7314–7322.

[35] X. Mu, Y. Shen, K. Cheng et al., “FedProc: Prototypical contrastive federated learning on non-IID data,” Future Gener. Comput. Syst., vol. 143, pp. 93–104, Jun. 2023.

[36] A. Li, J. Song, Z. Zheng et al., “From local bias to global consensus: Group-wise prototype federated learning under heterogeneous and cross-domain settings,” Inf. Sci., vol. 741, Jun. 2026, art. no. 123284.

[37] P. Zhang, M. Wen, Z. Zhu et al., “Taming the long tail in federated learning: A unified global and personalized model framework,” Inf. Sci., vol. 735, Apr. 2026, art. no. 123031.

[38] K. Hu, L. Xiang, P. Tang et al., “Feature norm regularized federated learning: Utilizing data disparities for model performance gains,” in Proc. Int. Joint Conf. Artif. Intell. (IJCAI), 2024, pp. 4136–4146.

[39] M. Yashwanth, G. K. Nayak, H. Rangwani et al., “Minimizing layerwise activation norm improves generalization in federated learning,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2024, pp. 2276–2285.

[40] S. Kim, M. Jeong, S. Kim et al., “FedDr+: Stabilizing dot-regression with global feature distillation for federated learning,” Trans. Mach. Learn. Res. (TMLR), 2025.

[41] S. Yan, Z. Li, C. Wu et al., “You are your own best teacher: Achieving centralized-level performance in federated learning under heterogeneous and long-tailed data,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 2750–2759.

[42] G. Legate, N. Bernier, L. Caccia et al., “Guiding the last layer in federated learning with pre-trained models,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, 2023, pp. 69 832–69 848.

[43] E. Fan\`ı, R. Camoriano, B. Caputo et al., “Accelerating heterogeneous federated learning with closed-form classifiers,” in Proc. Int. Conf. Mach. Learn. (ICML), ser. PMLR, vol. 235, 2024, pp. 13 029– 13 048.

[44] Z. Guan, Y. Zhou, and X. Gu, “Capture global feature statistics for one-shot federated learning,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 39, no. 16, 2025, pp. 16 942–16 950.

[45] F. Turazza, M. Picone, and M. Mamei, “The Gaussian-head OFL family: One-shot federated learning from client global statistics,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026, pp. 8041–8064.

[46] D. Goswami, S. Magistri, K. Wang et al., “Covariances for free: Exploiting mean distributions for training-free federated learning,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 38, 2025, pp. 65 081–65 115.

[47] K. Ahrens, H. H. Lehmann, J. H. Lee et al., “Read between the layers: Leveraging multi-layer representations for rehearsal-free continual learning with pre-trained models,” Trans. Mach. Learn. Res. (TMLR), 2024.

[48] R. Hossain, M. T. I. Bhuian, M. F. Shadiq et al., “MM++: Post-hoc scale-invariant multilayer OOD detection via top-K gated feature fusion,” arXiv preprint arXiv:2606.17352, 2026.

[49] L. Zhang, Z. Han, X. Feng et al., “TOFA: Training-free one-shot federated adaptation for vision-language models,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 40, no. 33, 2026, pp. 28 301–28 309.

[50] S. Zheng and S. Chen, “Rethinking one-shot federated graph learning: Training-free statistical estimation,” arXiv preprint arXiv:2609.06154, 2026.

[51] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd ed. New York, NY, USA: Springer, 2009.

[52] P. Khosla, P. Teterwak, C. Wang et al., “Supervised contrastive learning,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 18 661–18 673.

[53] Y. Zheng, D. K. Pal, and M. Savvides, “Ring loss: Convex feature normalization for face recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 5089–5097.

[54] J. H. Friedman, “Regularized discriminant analysis,” J. Amer. Statist. Assoc., vol. 84, no. 405, pp. 165–175, Mar. 1989.

[55] K. P. Murphy, “Conjugate Bayesian analysis of the Gaussian distribution,” Univ. British Columbia, Vancouver, BC, Canada, Tech. Rep., Oct. 2007.

[56] H. B. McMahan, E. Moore, D. Ramage et al., “Communicationefficient learning of deep networks from decentralized data,” in Proc. Int. Conf. Artif. Intell. Statist. (AISTATS), ser. PMLR, vol. 54, 2017, pp. 1273–1282.

[57] A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, Toronto, ON, Canada, Tech. Rep., 2009.

[58] T.-M. H. Hsu, H. Qi, and M. Brown, “Measuring the effects of non-identical data distribution for federated visual classification,” arXiv preprint arXiv:1909.06335, 2019.

# Supplementary Material

# Is High Local Accuracy Enough? Geometric Consensus with Support-Adaptive Statistical Personalization for Federated Learning

## APPENDIX A

THEORETICAL ANALYSIS AND IMPLEMENTATION DETAILS

This supplement uses the notation introduced in Section 3. Equations and theoretical results from the main article retain their original numbering; equations in this supplement use the prefix S.

## A.1 Proof of Lemma 4.1

Fix the class, layer, and round and use the notation of Lemma 4.1. Assume $p _ { \chi } \ = \ \mathrm { P r } ( \chi \ = \ 1 ) \ > \ 0$ and a finite conditional second moment. With $\bar { \rho } _ { \chi } = \mathbb { E } [ \rho \mid \chi = 1 ]$

$$
\begin{array} { r l } & { \mathbb { E } [ \chi ( \rho - a ) ^ { 2 } ] = p _ { \chi } \mathbb { E } [ ( \rho - a ) ^ { 2 } \mid \chi = 1 ] } \\ & { \qquad = p _ { \chi } \mathbb { E } [ ( \rho - \bar { \rho } _ { \chi } + \bar { \rho } _ { \chi } - a ) ^ { 2 } \mid \chi = 1 ] } \\ & { \qquad = p _ { \chi } \operatorname { V a r } ( \rho \mid \chi = 1 ) + p _ { \chi } ( \bar { \rho } _ { \chi } - a ) ^ { 2 } . } \end{array}\tag{S1}
$$

The cross term vanishes by conditional centering. This proves Eq. (16). If $\dot { p } _ { \chi } = 0$ , the contribution is zero. The same identity holds under the empirical distribution of a fixed batch.

## A.2 Complementary Feature Geometry

Fix one local step and treat all broadcast references as constants. For a nonzero feature h, put $r = \| \mathbf { h } \| _ { 2 }$ and ${ \bf u } = { \bf h } / r$ The differential of normalization is

$$
\mathrm { d } \mathbf { u } = { \frac { \mathrm { d } \mathbf { h } } { r } } - { \frac { \mathbf { h } \mathbf { h } ^ { \top } \mathrm { d } \mathbf { h } } { r ^ { 3 } } } = { \frac { \mathbf { I } - \mathbf { u } \mathbf { u } ^ { \top } } { r } } \mathrm { d } \mathbf { h } .\tag{S2}
$$

For $q _ { c } ( \mathbf { u } ) \ = \ \exp ( \mathbf { u } ^ { \top } \mathbf { p } _ { c } / \tau ) / \sum _ { r } \exp ( \mathbf { u } ^ { \top } \mathbf { p } _ { r } / \tau )$ and $\ell _ { \mathrm { p } } \ =$ $- \log q _ { y } ( \mathbf { u } )$ , differentiating the log-sum-exp gives

$$
\nabla _ { \mathbf { u } } \ell _ { \mathrm { p } } = \frac { 1 } { \tau } \Bigl ( \sum _ { c } q _ { c } ( \mathbf { u } ) \mathbf { p } _ { c } - \mathbf { p } _ { y } \Bigr ) .\tag{S3}
$$

The chain rule yields Eq. (18b). The full directional loss depends on raw features only through their normalized versions. Applying the chain rule to every occurrence of $\mathbf { u } _ { i } ,$ including contrastive comparisons, gives

$$
{ \bf g } _ { i } = \frac { \mathbf { I } - \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } } { \Vert \mathbf { h } _ { i } \Vert _ { 2 } } \nabla _ { \mathbf { u } _ { i } } \mathcal { L } _ { \mathrm { d i r } } .\tag{S4}
$$

Since $\mathbf { h } _ { i } ^ { \top } ( \mathbf { I } - \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } ) = \mathbf { 0 } ^ { \top }$ , the directional gradient is tangent to the sphere at $\mathbf { h } _ { i } .$ . For an isolated feature-space step of size $s ,$ this also gives

$$
\lVert \mathbf { h } _ { i } - s \mathbf { g } _ { i } \rVert _ { 2 } ^ { 2 } = \lVert \mathbf { h } _ { i } \rVert _ { 2 } ^ { 2 } + s ^ { 2 } \lVert \mathbf { g } _ { i } \rVert _ { 2 } ^ { 2 } .\tag{S5}
$$

Thus, the change in squared magnitude is second order in the step size.

For the radial term, $\begin{array} { r } { \rho = \frac 1 2 \log ( \| \mathbf h \| _ { 2 } ^ { 2 } + \varepsilon ^ { 2 } ) } \end{array}$ , hence

$$
\nabla _ { \mathbf { h } } \rho = \frac { \mathbf { h } } { \| \mathbf { h } \| _ { 2 } ^ { 2 } + \varepsilon ^ { 2 } } , ~ \nabla _ { \mathbf { h } } [ \chi ( \rho - a ) ^ { 2 } ] = 2 \chi ( \rho - a ) \nabla _ { \mathbf { h } } \rho .\tag{S6}
$$

The indicator χ and reference a are held fixed, matching the implementation’s stop-gradient convention. This yields Eq. (19). The radial gradient parallels h and is orthogonal to the same final-layer feature’s directional gradient.

The implementation clamps the normalization denominator from below; the unit-vector identities apply above this floor. When the radial gate is active, the configured safe-norm threshold places the feature well above the directional normalization floor. Through the backbone Jacobian $\mathbf { J } _ { i } = \partial \mathbf { h } _ { i } / \partial \phi$ , the corresponding parameter-gradient inner product becomes $\mathbf { g } _ { i } ^ { \top } \mathbf { J } _ { i } \mathbf { J } _ { i } ^ { \top } \mathbf { v } _ { i } ;$ the orthogonality above describes the feature-space gradients.

To relate this geometry to raw-feature scatter, let $\mathbf { h } _ { i } \ =$ $R _ { i } { \bf u } _ { i } , ~ { \bf h } _ { j } \ = \ R _ { j } { \bf u } _ { j } ,$ , with ${ R _ { i } } , { R _ { j } } ~ > ~ 0$ and unit directions. Expanding the squared distance gives

$$
\begin{array} { r l } & { \| \mathbf { h } _ { i } - \mathbf { h } _ { j } \| _ { 2 } ^ { 2 } = R _ { i } ^ { 2 } + R _ { j } ^ { 2 } - 2 R _ { i } R _ { j } \mathbf { u } _ { i } ^ { \top } \mathbf { u } _ { j } } \\ & { \qquad = ( R _ { i } - R _ { j } ) ^ { 2 } + 2 R _ { i } R _ { j } ( 1 - \mathbf { u } _ { i } ^ { \top } \mathbf { u } _ { j } ) } \\ & { \qquad = ( R _ { i } - R _ { j } ) ^ { 2 } + R _ { i } R _ { j } \| \mathbf { u } _ { i } - \mathbf { u } _ { j } \| _ { 2 } ^ { 2 } . } \end{array}\tag{S7}
$$

This proves Eq. (10).

To derive Eq. (21), let $\{ \mathbf { h } _ { i } \} _ { i = 1 } ^ { N _ { c } }$ denote the class-c features from all clients at layer ℓ under a common backbone. Their shared class mean and covariance are

$$
\begin{array} { l } { { \displaystyle { { \bar { \mathbf { h } } } _ { \ell , c } } = \frac { 1 } { N _ { c } } \sum _ { i = 1 } ^ { N _ { c } } { \mathbf { h } _ { i } } } , } \\ { { \displaystyle { \widehat { \mathbf { \Sigma } } } { } _ { { \widehat { \mathbf { L } } } \ell , c } } = \frac { 1 } { N _ { c } } \sum _ { i = 1 } ^ { N _ { c } } ( \mathbf { h } _ { i } - { \bar { \mathbf { h } } _ { \ell , c } } ) ( \mathbf { h } _ { i } - { \bar { \mathbf { h } } _ { \ell , c } } ) ^ { \top } . } \end{array}\tag{S8}
$$

Using $\begin{array} { r } { \sum _ { i } \mathbf { h } _ { i } = N _ { c } \bar { \mathbf { h } } _ { \ell , c } , } \end{array}$ expansion about the mean gives

$$
\begin{array} { l } { \displaystyle \mathrm { t r } ( \widehat { \Sigma } _ { \ell , c } ) = \frac { 1 } { N _ { c } } \sum _ { i } \| \mathbf { h } _ { i } - \bar { \mathbf { h } } _ { \ell , c } \| _ { 2 } ^ { 2 } } \\ { \displaystyle = \frac { 1 } { N _ { c } } \sum _ { i } \| \mathbf { h } _ { i } \| _ { 2 } ^ { 2 } - \| \bar { \mathbf { h } } _ { \ell , c } \| _ { 2 } ^ { 2 } . } \end{array}\tag{S9}
$$

The pairwise sum has the same centered form:

$$
\begin{array} { l } { { \displaystyle \sum _ { i , j } \| { \bf h } _ { i } - { \bf h } _ { j } \| _ { 2 } ^ { 2 } = 2 N _ { c } \sum _ { i } \| { \bf h } _ { i } \| _ { 2 } ^ { 2 } - 2 \Big \| \sum _ { i } { \bf h } _ { i } \Big \| _ { 2 } ^ { 2 } } } \\ { { \displaystyle ~ = 2 N _ { c } ^ { 2 } \mathrm { t r } \big ( \widehat { \Sigma } _ { \ell , c } \big ) } . } \end{array}\tag{S10}
$$

Dividing by $2 N _ { c } ^ { 2 }$ and substituting the angular–radial distance decomposition yields

$$
\begin{array} { l } { \displaystyle \mathrm { t r } ( \widehat { \Sigma } _ { \ell , c } ) = \frac { 1 } { 2 N _ { c } ^ { 2 } } \sum _ { i , j } ( R _ { i } - R _ { j } ) ^ { 2 } } \\ { \displaystyle + \frac { 1 } { 2 N _ { c } ^ { 2 } } \sum _ { i , j } R _ { i } R _ { j } \| \mathbf { u } _ { i } - \mathbf { u } _ { j } \| _ { 2 } ^ { 2 } . } \end{array}\tag{S11}
$$

This proves Eq. (21). The sum includes pairs from different clients and therefore captures both local dispersion and cross-client differences within the same class. The radial contribution can remain nonzero even when all directions agree; the angular contribution also depends on the feature magnitudes through $R _ { i } R _ { j }$

For MMS, all features above are recomputed under the frozen backbone $\phi _ { T }$ . For a fixed layer scale $s _ { \ell } \ > \ 0 ,$ , the scaled distance is $s _ { \ell } ^ { - 2 } \| \mathbf { h } _ { i } - \mathbf { h } _ { j } \| _ { 2 } ^ { 2 } ,$ and each nonzero raw feature’s log norm shifts by $\mathbf { \partial } \cdot \mathbf { \partial } - \log { s _ { \ell } } .$ . Norm ratios, withinclass raw log-norm variance, and differences between class means of raw log norms are therefore unchanged. The layerℓ residual block for a sample of class c is

$$
{ \bf e } _ { \ell i } ^ { ( k ) } = \frac { { \bf h } _ { \ell } ( x _ { i } ^ { ( k ) } ; \phi _ { T } ) - \bar { \bf h } _ { \ell , c } } { s _ { \ell } } .\tag{S12}
$$

Taking the trace in Eq. (25) and summing the squared norms of these blocks yields

$$
\begin{array} { l } { \displaystyle \mathrm { t r } ( \widehat { \Sigma } _ { \mathrm { w } } ) = \frac { 1 } { N } \sum _ { k , i } \sum _ { \ell \in \Lambda } \| \mathbf { e } _ { \ell i } ^ { ( k ) } \| _ { 2 } ^ { 2 } } \\ { = \displaystyle \sum _ { c = 1 } ^ { C } \frac { N _ { c } } { N } \sum _ { \ell \in \Lambda } \frac { \mathrm { t r } ( \widehat { \Sigma } _ { \ell , c } ) } { s _ { \ell } ^ { 2 } } . } \end{array}\tag{S13}
$$

This proves Eq. (28), linking class- and layer-specific scatter to the pooled within-class covariance used by MMS. The trace characterizes total within-class variation; the full covariance additionally retains directional structure and crosslayer correlations. These identities provide a geometric interpretation of CGRL: the magnitude loss regulates the log norms of views with $\chi _ { \ell i } = 1$ , while directional supervision also promotes between-class separation. CGRL optimizes these directional and magnitude objectives, which differ from covariance-trace minimization.

## A.3 Proof of Proposition 4.1

For $n _ { c } ^ { ( k ) } > 0 .$ , define the local class covariance as

$$
\widehat { \pmb { \Sigma } } _ { c } ^ { ( k ) } = \frac { 1 } { n _ { c } ^ { ( k ) } } \sum _ { i \in \mathcal { I } _ { c } ^ { ( k ) } } \left( \mathbf { z } ( x _ { i } ^ { ( k ) } ) - { \pmb \mu } _ { c } ^ { ( k ) } \right) \left( \mathbf { z } ( x _ { i } ^ { ( k ) } ) - { \pmb \mu } _ { c } ^ { ( k ) } \right) ^ { \top } .\tag{S14}
$$

Let $\mathbf { r } _ { i } ^ { ( k ) } = \mathbf { z } ( x _ { i } ^ { ( k ) } ) - \pmb { \mu } _ { c } ^ { ( k ) }$ within this class and $\Delta _ { c } ^ { ( k ) } = \mu _ { c } ^ { ( k ) } -$ $\mu _ { c }$ . Since $\begin{array} { r } { \sum _ { i \in \mathcal { T } _ { c } ^ { ( k ) } } \mathbf { r } _ { i } ^ { ( k ) } = \mathbf { 0 } } \end{array}$ , the cross terms vanish in

$$
\begin{array} { r l } & { \displaystyle \sum _ { i \in \mathbb { Z } _ { c } ^ { ( k ) } } ( \mathbf { z } ( x _ { i } ^ { ( k ) } ) - \boldsymbol { \mu } _ { c } ) ( \mathbf { z } ( x _ { i } ^ { ( k ) } ) - \boldsymbol { \mu } _ { c } ) ^ { \top } } \\ & { \quad = \displaystyle \sum _ { i \in \mathbb { Z } _ { c } ^ { ( k ) } } ( \mathbf { r } _ { i } ^ { ( k ) } + \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } ^ { ( k ) } ) ( \mathbf { r } _ { i } ^ { ( k ) } + \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } ^ { ( k ) } ) ^ { \top } } \\ & { \quad = n _ { c } ^ { ( k ) } \widehat { \mathbf { \Delta } } \mathbf { \Delta } \mathbf { \widehat { \mathbf { \Sigma } } } _ { c } ^ { ( k ) } + n _ { c } ^ { ( k ) } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } ^ { ( k ) } ( \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } ^ { ( k ) } ) ^ { \top } . } \end{array}\tag{S15}
$$

Summing over all nonempty groups and dividing by N proves the stated decomposition.

## A.4 Regularized Metric Bounds

Write $\widehat { \pmb { \Sigma } } _ { \mathrm { w } } = \mathbf { Q }$ diag $( \sigma _ { 1 } , \ldots , \sigma _ { d } ) \mathbf { Q } ^ { \top }$ with $\sigma _ { j } \ \geq \ 0$ . Positive trace and $\gamma > 0$ imply $\begin{array} { r } { \lambda _ { A } = \gamma \sum _ { j } \sigma _ { j } / d > 0 . } \end{array}$ . Consequently,

$$
\mathbf { A } = \mathbf { Q } \mathrm { d i a g } \left( { \frac { 1 } { \sigma _ { 1 } + \lambda _ { A } } } , \dots , { \frac { 1 } { \sigma _ { d } + \lambda _ { A } } } \right) \mathbf { Q } ^ { \top }\tag{S16}
$$

is positive definite. Its largest eigenvalue is $1 / ( \sigma _ { \operatorname* { m i n } } + \lambda _ { A } )$ and its smallest is $1 / ( \sigma _ { \mathrm { m a x } } { + \lambda _ { A } } )$ . These eigenvalues give the

operator-norm and condition-number identities in Eq. (31). Using $\sigma _ { \operatorname* { m i n } } \geq 0$ and $\sigma _ { \mathrm { m a x } } \le \textstyle \sum _ { j } \sigma _ { j }$ gives

$$
\kappa _ { 2 } ( \mathbf { A } ) \leq 1 + \frac { \sigma _ { \operatorname* { m a x } } } { \lambda _ { A } } \leq 1 + \frac { \sum _ { j } \sigma _ { j } } { \gamma \sum _ { j } \sigma _ { j } / d } = 1 + \frac { d } { \gamma } .\tag{S17}
$$

This bound covers singular empirical covariances and depends only on dimension and relative ridge strength.

## A.5 Construction from Additive Moments

Let ${ \bf h } _ { i } ^ { ( k ) }$ be the concatenation of unscaled layer features extracted with the same frozen backbone. For each client, retain the counts and moments

$$
\mathbf { t } _ { c } ^ { ( k ) } = \sum _ { i \in \mathcal { T } _ { c } ^ { ( k ) } } \mathbf { h } _ { i } ^ { ( k ) } , \mathbf { G } ^ { ( k ) } = \sum _ { i } \mathbf { h } _ { i } ^ { ( k ) } ( \mathbf { h } _ { i } ^ { ( k ) } ) ^ { \top } .\tag{S18}
$$

The additive quantities $\begin{array} { r } { \mathbf { t } _ { c } = \sum _ { k } \mathbf { t } _ { c } ^ { ( k ) } , \mathbf { G } = \sum _ { k } \mathbf { G } ^ { ( k ) } } \end{array}$ , and $\begin{array} { r } { N _ { c } = \sum _ { k } n _ { c } ^ { ( k ) } } \end{array}$ give

$$
\bar { \mathbf { h } } = \frac { 1 } { N } \sum _ { c } \mathbf { t } _ { c } , \widehat { \mathbf { \xi } } _ { \mathrm { \bf { X } } } = \frac { \mathbf { G } } { N } - \bar { \mathbf { h } } \bar { \mathbf { h } } ^ { \top } .\tag{S19}
$$

Its diagonal layer blocks give the shared scales $s _ { \ell }$ via Eq. (22). Set $\mathbf { D } _ { s } \mathbf { \bar { \Pi } } = \mathrm { b l o c k d i a g } _ { \ell \in \Lambda } \bigl ( s _ { \ell } ^ { - 1 } \mathbf { I } _ { d _ { \ell } } \bigr )$ . Then

$$
\begin{array} { c } { { \displaystyle { \pmb \mu } _ { c } = { \bf D } _ { s } { \bf t } _ { c } / N _ { c } , } } \\ { { \displaystyle { \pmb \mu } _ { c } ^ { ( k ) } = { \bf D } _ { s } { \bf t } _ { c } ^ { ( k ) } / n _ { c } ^ { ( k ) } \quad ( n _ { c } ^ { ( k ) } > 0 ) } , } \\ { { \displaystyle \hat { \bf \Sigma } _ { \bf w } = \frac { 1 } { N } { \bf D } _ { s } { \bf G } { \bf D } _ { s } - \sum _ { c } \frac { N _ { c } } { N } { \pmb \mu } _ { c } { \pmb \mu } _ { c } ^ { \top } . } } \end{array}\tag{S20}
$$

The last equality follows by expanding the residuals about their shared class means and using

$$
\sum _ { k , i : y _ { i } ^ { ( k ) } = c } \mathbf { z } ( x _ { i } ^ { ( k ) } ) = N _ { c } \pmb { \mu } _ { c } .
$$

This construction retains cross-layer covariance blocks and equals the pooled computation in exact arithmetic.

## A.6 Proof of Proposition 4.2

Fix $\mathbf A \succ 0$ and $\pmb { \mu } _ { c }$ in the objective $J _ { c } ^ { ( k ) }$ . For brevity, write $n = n _ { c } ^ { ( k ) }$ and $\begin{array} { r } { \mathbf { t } = \sum _ { i \in \mathcal { T } _ { c } ^ { ( k ) } } \mathbf { z } ( \dot { x _ { i } } ^ { ( k ) } ) } \end{array}$ . Differentiation gives

$$
\begin{array} { l } { { \nabla J _ { c } ^ { ( k ) } ( { \bf m } ) = 2 { \bf A } \big [ ( n + m _ { 0 } ) { \bf m } - { \bf t } - m _ { 0 } \pmb { \mu } _ { c } \big ] , } } \\ { { \nabla ^ { 2 } J _ { c } ^ { ( k ) } ( { \bf m } ) = 2 ( n + m _ { 0 } ) { \bf A } \succ 0 . } } \end{array}\tag{S21}
$$

Strict convexity gives a unique minimizer. Setting the gradient to zero yields

$$
{ \bf m } ^ { \star } = \frac { { \bf t } + m _ { 0 } \mu _ { c } } { n + m _ { 0 } } .\tag{S22}
$$

For $n > 0 ,$ , use $\mathbf { t } = n \mu _ { c } ^ { ( k ) }$ to obtain

$$
\mathbf { m } ^ { \star } = \frac { m _ { 0 } } { n + m _ { 0 } } \pmb { \mu } _ { c } + \frac { n } { n + m _ { 0 } } \pmb { \mu } _ { c } ^ { ( k ) } = \widetilde { \pmb { \mu } } _ { c } ^ { ( k ) } .\tag{S23}
$$

For $n = 0 .$ , the data sum is empty, $\mathbf { t } = \mathbf { 0 } ,$ and $\mathbf { m } ^ { \star } = \pmb { \mu _ { c } }$ . For $n > 0 ,$ , subtracting the local center gives

$$
\tilde { \pmb { \mu } } _ { c } ^ { ( k ) } - \pmb { \mu } _ { c } ^ { ( k ) } = ( 1 - \alpha _ { c } ^ { ( k ) } ) ( \pmb { \mu } _ { c } - \pmb { \mu } _ { c } ^ { ( k ) } ) .\tag{S24}
$$

Positive homogeneity of the A-norm yields Eq. (35). For fixed centers, $1 - \alpha ( n ) \ = \ m _ { 0 } / ( n + m _ { 0 } )$ decreases with support, reducing the local-center residual.

## A.7 Statistical Head Construction

All argmax operations use the same fixed tie-breaking rule. Expanding the personalized squared distance gives

$$
\begin{array} { r l } & { - \frac { 1 } { 2 } \| { \mathbf z } - \widetilde { \pmb \mu } _ { c } ^ { ( k ) } \| _ { \mathbf A } ^ { 2 } = - \frac { 1 } { 2 } { \mathbf z } ^ { \top } { \mathbf A } { \mathbf z } + ( \widetilde { \pmb \mu } _ { c } ^ { ( k ) } ) ^ { \top } { \mathbf A } { \mathbf z } } \\ & { \qquad - \frac { 1 } { 2 } ( \widetilde { \pmb \mu } _ { c } ^ { ( k ) } ) ^ { \top } { \mathbf A } \widetilde { \pmb \mu } _ { c } ^ { ( k ) } . } \end{array}\tag{S25}
$$

Dropping the class-independent first term yields the statistical head in $\mathrm { E q . } \ ( 3 6 )$ . Under z $\mid y = c \sim \mathcal { N } ( \widetilde { \pmb { \mu } } _ { c } ^ { ( k ) } , \mathbf { A } ^ { - 1 } )$ and equal priors $\pi _ { c } = 1 / C$ , the Bayes rule is

$$
\arg \operatorname* { m a x } _ { c } \frac { 1 } { C } \mathcal { N } ( \mathbf { z } ; \widetilde { \pmb { \mu } } _ { c } ^ { ( k ) } , \mathbf { A } ^ { - 1 } ) = \arg \operatorname* { m a x } _ { c } s _ { c } ^ { ( k ) } ( \mathbf { z } ) .\tag{S26}
$$

## A.8 Proof of Corollary 4.1 and Risk Bounds

Let $\begin{array} { r } { s _ { c } ^ { 0 } ( \mathbf { z } ) = \pmb { \mu } _ { c } ^ { \top } \mathbf { A } \mathbf { z } - \frac { 1 } { 2 } \pmb { \mu } _ { c } ^ { \top } \mathbf { A } \pmb { \mu } _ { c } } \end{array}$ denote the shared score. The personalized displacement is $\delta _ { c } ^ { ( k ) } = \widetilde { \mu } _ { c } ^ { ( k ) } - \mu _ { c }$ . Substitution into Eq. (36b) and subtraction of $s _ { c } ^ { 0 } ( \mathbf { z } )$ yield

$$
\begin{array} { r l } & { s _ { c } ^ { ( k ) } ( \mathbf { z } ) - s _ { c } ^ { 0 } ( \mathbf { z } ) = ( \pmb { \delta } _ { c } ^ { ( k ) } ) ^ { \top } \mathbf { A } ( \mathbf { z } - \pmb { \mu } _ { c } ) } \\ & { ~ - ~ \frac { 1 } { 2 } \| \pmb { \delta } _ { c } ^ { ( k ) } \| _ { \mathbf { A } } ^ { 2 } . } \end{array}\tag{S27}
$$

Cauchy–Schwarz in the A-inner product implies

$$
\begin{array} { r } { | s _ { c } ^ { ( k ) } ( \mathbf { z } ) - s _ { c } ^ { 0 } ( \mathbf { z } ) | \leq \epsilon _ { c } ^ { ( k ) } ( \mathbf { z } ) , \qquad } \\ { \epsilon _ { c } ^ { ( k ) } ( \mathbf { z } ) = \| \pmb { \delta } _ { c } ^ { ( k ) } \| _ { \mathbf { A } } \| \mathbf { z } - \pmb { \mu } _ { c } \| _ { \mathbf { A } } + \frac { 1 } { 2 } \| \pmb { \delta } _ { c } ^ { ( k ) } \| _ { \mathbf { A } } ^ { 2 } . } \end{array}\tag{S28}
$$

Substituting $\delta _ { c } ^ { ( k ) } = \alpha _ { c } ^ { ( k ) } \Delta _ { c } ^ { ( k ) }$ into Eq. (S27) recovers Eq. (37). For a locally absent class, both the displacement and the score perturbation are zero. For $c ^ { \star } = f ^ { 0 } ( x ) , \mathbf { z } = \mathbf { z } ( x )$ , and any $c \neq c ^ { \star }$ , the adapted margin obeys

$$
\begin{array} { r l } & { s _ { c ^ { \star } } ^ { ( k ) } ( \mathbf { z } ) - s _ { c } ^ { ( k ) } ( \mathbf { z } ) } \\ & { ~ \geq s _ { c ^ { \star } } ^ { 0 } ( \mathbf { z } ) - s _ { c } ^ { 0 } ( \mathbf { z } ) - \epsilon _ { c ^ { \star } } ^ { ( k ) } ( \mathbf { z } ) - \epsilon _ { c } ^ { ( k ) } ( \mathbf { z } ) } \\ & { ~ \geq M ^ { 0 } ( x ) - B ^ { ( k ) } ( x ) . } \end{array}\tag{S29}
$$

If $M ^ { 0 } ( x ) > B ^ { ( k ) } ( x )$ , every adapted margin is positive, so $c ^ { \star }$ remains the unique maximizer, proving Eq. (40). Ties among the largest shared scores have $\dot { M } ^ { 0 } ( x ) \stackrel { < } { = } 0$ and fall within the complementary event.

Risk Bounds. If $f ^ { ( k ) } ( x ) \ \ne \ y ,$ either $f ^ { 0 } ( x ) ~ \neq ~ y$ or adaptation changes a correct shared prediction. Combining this observation with the contrapositive of Eq. (40) yields

$$
\begin{array} { r l } & { \mathbf { 1 } \{ f ^ { ( k ) } ( x ) \neq y \} } \\ & { \quad \leq \mathbf { 1 } \{ f ^ { 0 } ( x ) \neq y \} + \mathbf { 1 } \{ f ^ { ( k ) } ( x ) \neq f ^ { 0 } ( x ) \} } \\ & { \quad \leq \mathbf { 1 } \{ f ^ { 0 } ( x ) \neq y \} + \mathbf { 1 } \{ M ^ { 0 } ( x ) \leq B ^ { ( k ) } ( x ) \} } \end{array}\tag{S30}
$$

for every labeled example. Taking expectations over any labeled target distribution R with input marginal $R _ { X }$ gives

$$
\mathcal { E } _ { R } ( f ^ { ( k ) } ) \leq \mathcal { E } _ { R } ( f ^ { 0 } ) + \operatorname* { P r } _ { x \sim R _ { X } } [ M ^ { 0 } ( x ) \leq B ^ { ( k ) } ( x ) ] .\tag{S31}
$$

Setting $R = Q$ yields the bound for the class-balanced allclass target in Section 3.3. For class $c ,$

$$
e _ { c } ^ { ( k ) } \leq e _ { c } ^ { 0 } + \Pr _ { x \sim Q _ { c } } [ M ^ { 0 } ( x ) \leq B ^ { ( k ) } ( x ) ] ,\tag{S32}
$$

where $e _ { c } ^ { 0 }$ is the shared predictor’s class-wise error. Mixtures over locally rare or unseen classes yield the corresponding subset risk bounds.

TABLE 10: Key parameters and implementation details of FedGRC.  
Configuration Description   
A. Representation   
ResNet-18 Trained from scratch; no pretrained weights.   
$\Lambda = \{ 2 , 3 , 4 \}$ Pooled dimensions: 128, 256, 512;   
concatenated dimension d = 896.   
B. Consensus-Guided Representation Learning (CGRL)   
$\lambda _ { \mathrm { d i r } } = \lambda _ { \mathrm { r a d } } = 1$ Directional and magnitude guidance weights.   
$\beta = 1$ Shared-prototype weight in the directional loss.   
τ = 0.1 Shared temperature for supervised   
contrastive and prototype guidance.   
$T _ { w } = 5 0$ Warm-up rounds for directional   
and magnitude guidance losses.   
$\xi = \zeta = 0 . 5$ New-observation weights for directional   
(normalized) and radial EMAs.   
$\varepsilon = 1 0 ^ { - 6 }$ Stability constant for the safe feature norm.   
$r _ { \operatorname* { m i n } } = 1 0 ^ { - 3 }$ Safe-norm threshold for magnitude guidance.   
C. Multilayer Metric Synthesis (MMS)   
$\gamma = 0 . 1$ Shared ridge coefficient: $\lambda _ { A } = \gamma \mathrm { t r } ( \widehat { \pmb { \Sigma } } _ { \mathrm { w } } ) / d .$   
D. Support-Adaptive Shrinkage Personalization $( S A S P )$   
m = 64 Shared-reference strength,   
fixed across clients and classes.   
$\pi _ { c } = 1 / C$ Equal priors; no local class-frequency bias.

## A.9 Reference Updates and Implementation Conventions

Table 10 summarizes FedGRC’s parameters and modeling choices. Reference initialization, statistics accumulation, and updates are detailed below.

The schedule $w _ { t } = \operatorname* { m i n } \{ ( t + 1 ) / T _ { w } , 1 \}$ gradually introduces the auxiliary terms over $T _ { w }$ warm-up rounds. Each participating client performs $E ^ { ( k ) }$ mini-batch updates of $\bar { \pmb { \theta } } = ( \bar { \phi } , \psi )$ , and the server aggregates

$$
\theta _ { t + 1 } = \sum _ { k \in S _ { t } } \omega _ { t } ^ { ( k ) } \pmb \theta ^ { ( k ) } , \omega _ { t } ^ { ( k ) } = \frac { N ^ { ( k ) } } { \sum _ { j \in S _ { t } } N ^ { ( j ) } } .\tag{S33}
$$

After local training, a pass with the updated backbone $\phi ^ { ( k ) }$ in evaluation mode and without random augmentation yields a direction for each locally observed class ${ \bf \bar { \Phi } } ( n _ { c } ^ { ( k ) } > 0 ) $

$$
\bar { \mathbf { u } } _ { c } ^ { ( k ) } = \mathrm { u n i t } \left( \frac { 1 } { n _ { c } ^ { ( k ) } } \sum _ { ( x , c ) \in \mathcal { D } _ { c } ^ { ( k ) } } \mathrm { u n i t } \Big ( \mathbf { h } _ { L } ( x ; \phi ^ { ( k ) } ) \Big ) \right) .\tag{S34}
$$

For a class observed in round $t ,$ count-weighted aggregation and normalized averaging give

$$
\widehat { \bf p } _ { c } ( t ) = \mathrm { u n i t } \left( \sum _ { k \in S _ { t } : n _ { c } ^ { ( k ) } > 0 } n _ { c } ^ { ( k ) } \bar { \bf u } _ { c } ^ { ( k ) } \right) ,\tag{S35a}
$$

$$
{ \bf p } _ { c } ( t + 1 ) = \mathrm { u n i t } ( ( 1 - \xi ) { \bf p } _ { c } ( t ) + \xi \widehat { { \bf p } } _ { c } ( t ) ) .\tag{S35b}
$$

Here $\xi$ is the weight assigned to the current round’s direction estimate; this update retains a common orientation while moderating changes between rounds.

Each class direction stays zero until the class is first observed; prototype loss starts only after the first aggregation. Once enabled, prototype cross-entropy uses all class references, assigning zero logits to zero references and including samples with zero target references. Classes absent from a round retain their previous direction.

Radial statistics are accumulated during local training. Let $\mathcal { Z } _ { t } ^ { ( k ) }$ be the multiset of all training occurrences processed by client k in round $t ,$ counting repeated occurrences separately. For each occurrence $i , \bar { \rho } _ { \ell } ( \bar { x _ { i } } )$ is evaluated before its associated parameter update. Define

$$
\begin{array} { l } { { \displaystyle H _ { \ell , c } ^ { ( k ) } ( t ) = \sum _ { i \in \mathcal { Z } _ { t } ^ { ( k ) } : y _ { i } = c } \rho _ { \ell } ( x _ { i } ) } , \ ~ } \\ { { \displaystyle q _ { c } ^ { ( k ) } ( t ) = \sum _ { i \in \mathcal { Z } _ { t } ^ { ( k ) } } \mathbf { 1 } \{ y _ { i } = c \} } . \ ~ } \end{array}\tag{S36}
$$

These sums and counts are accumulated locally during training and used to update the shared magnitude references. For $\begin{array} { r } { q _ { c } ( t ) = \sum _ { k \in \mathcal { S } _ { t } } q _ { c } ^ { ( k ) } ( t ) > 0 , } \end{array}$ , the pooled log-norm mean is

$$
\widehat { a } _ { \ell , c } ( t ) = \frac { \sum _ { k \in { \mathcal { S } _ { t } } } H _ { \ell , c } ^ { ( k ) } ( t ) } { q _ { c } ( t ) } .\tag{S37}
$$

All training occurrences contribute to the reference statistics, whether or not the radial penalty is applied. For a class observed both previously and in the current round, the reference update assigns weight ζ to the current estimate. With observation flag $v _ { c } ( t )$ , the complete update is

$$
\begin{array} { r l } & { a _ { \ell , c } ( t + 1 ) } \\ & { \quad = \left\{ \begin{array} { l l } { a _ { \ell , c } ( t ) , } & { q _ { c } ( t ) = 0 , } \\ { \widehat { a } _ { \ell , c } ( t ) , } & { q _ { c } ( t ) > 0 , v _ { c } ( t ) = 0 , } \\ { ( 1 - \zeta ) a _ { \ell , c } ( t ) + \zeta \widehat { a } _ { \ell , c } ( t ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \\ & { \quad v _ { c } ( t + 1 ) = v _ { c } ( t ) \vee \mathbf { 1 } \{ q _ { c } ( t ) > 0 \} . } \end{array}\tag{S38}
$$

Both references and flags are broadcast for the next round.

The occurrence counts $q _ { c } ^ { ( k ) } ( t )$ are distinct from the training-set counts $n _ { c } ^ { ( k ) }$ used by SASP. Terminal statistics are collected from all registered clients using the final postaggregation backbone, with augmentation disabled and normalization statistics fixed.

## APPENDIX B

## SUPPLEMENTARY EXPERIMENTS

This appendix details the dataset partitions and client-level data distributions, then presents further analysis and results on personalization floors and resource overhead.

## B.1 Datasets and Client Distributions

CIFAR-10 and CIFAR-100. CIFAR-10 and CIFAR-100 [1] each contain 60,000 images spanning 10 and 100 classes, respectively. We pool the original training and test sets and reserve 6,000 class-balanced images as a shared global test set, yielding 600 images per class for CIFAR-10 and 60 for CIFAR-100. The remaining 54,000 images are distributed among 20 clients using Dirichlet label partitions [2] with concentration parameters 0.3 and 0.1; each client’s data are then randomly split into approximately 75% training and 25% local test samples. Figs. 3 and 4 show the resulting client–class distributions. Both datasets exhibit unequal class counts and incomplete class coverage across clients, with Dir(0.1) producing more concentrated distributions and more missing classes than Dir(0.3).

FedRS. FedRS [3] integrates NaSC-TG2 [4], WHU-RS19 [5], [6], EuroSAT [7], AID [8], NWPU-RESISC45 [9], UCM [10], OPTIMAL-31 [11], and RSD46-WHU [12], [13] into a unified label space of 15 remote-sensing scene classes. We use its NIID-1 partition, comprising 135 clients, each drawing images from a single source dataset. The 83,548 images held by clients are split by class within each client into approximately 75% training and 25% local test samples, retaining training examples for every observed class. The shared global test set contains 940 images from the benchmark’s $T _ { B }$ set and covers all 15 classes. Fig. 5 shows differences in class coverage and sample counts across clients, with colors identifying the eight source datasets. These sources span different sensors, spatial resolutions, and geographic regions, introducing domain heterogeneity alongside label and sample-size imbalance.

## B.2 Personalization Floors: Analysis and Results

Client-Wise Minima before Aggregation. The order of aggregation determines which client weaknesses a recognition-floor metric captures. Using the client-level minima $\mathbf { \check { \textit { F } } } ^ { ( k ) }$ from Section 5.1.2, consider the two summaries

$$
\begin{array} { r } { F @ 2 5 = \mathrm { T a i l } _ { 2 5 } \big ( ( F ^ { ( k ) } ) _ { k = 1 } ^ { K } \big ) , } \\ { \operatorname* { m i n } \{ L @ 2 5 , U @ 2 5 , R @ 2 5 \} . } \end{array}
$$

The first ranks clients by their lowest accuracy across the assessed abilities and averages the lowest scores; the second takes the minimum of three ability-specific tail averages. They need not coincide because the weakest ability can differ across clients. For example, some clients may struggle with unseen classes while others struggle with rare classes. The client group used for F@25 can include both types according to their weakest scores, whereas the minimum of the three tail averages summarizes a single ability. Thus, F@25 captures a personalization floor across abilities and clients, while L@25, U@25, and R@25 characterize weaknesses in individual abilities. These views are complementary. Clients without unseen classes contribute to F@25 through their local and rare-class accuracies; an ability with no participating clients is reported as n/a.

Additional Results on CIFAR. Tables 11 and 12 extend the lower-tail evaluation to CIFAR-10 and CIFAR-100, using the same comparison categories and protocol as Section 5.1.2. FedGRC achieves the highest U@25, R@25, and F@25 in all 4 CIFAR heterogeneity settings. Its gains over the best F@25 baseline are 2.13% and 2.82% on CIFAR-10, and 6.46% and 3.34% on CIFAR-100, for Dir(0.3) and Dir(0.1), respectively. Under Dir(0.3), it also leads in L@25; under Dir(0.1), its L@25 trails the best baseline by 13.57% on CIFAR-10 and 3.59% on CIFAR-100. These results show stronger recognition floors alongside a trade-off in local adaptation under extreme label heterogeneity.

![](images/a9af5d830d77f080a4e209492ea325ee8f30d452f11c965f9684a26831fc7259.jpg)

![](images/a4b80e8eab2a549603a946fe518cf354b1638ce063ae62724fd802647b3b91d5.jpg)  
Fig. 3: Heterogeneity in client–class distributions of training and local test samples on CIFAR-10 [1] under Dir(0.3) and Dir(0.1). Blank entries indicate zero samples.

![](images/fda810e205b95c139cc70ca2c04d001db47fd6a5ea23a941f27e720e700e59e0.jpg)

![](images/e2ee725f59cf1b552e4797c5e18f48c19fde63c781b1da200e66c27f44d52915.jpg)  
Fig. 4: Heterogeneity in client–class distributions of training and local test samples on CIFAR-100 [1] under Dir(0.3) and Dir(0.1). Blank entries indicate zero samples.

![](images/bcd48f509790ec02830c5141f3df28f9683499954a0565c01e18e61505b952f0.jpg)  
Fig. 5: Heterogeneity in client–class distributions of training and local test samples on FedRS [3] under NIID-1. Blank entries indicate zero samples.

TABLE 11: Client recognition floors on CIFAR-10 [1]. L@25, U@25, R@25, and F@25 average the lowest 25% (5 of 20 clients), with clients ranked separately for each metric.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="4">CIFAR-10, Dir(0.3), 20 clients</td><td colspan="4">CIFAR-10, Dir(0.1), 20 clients</td></tr><tr><td>L@25</td><td>U@25</td><td>R@25</td><td>F@25</td><td>L@25</td><td>U@25</td><td>R@25</td><td>F@25</td></tr><tr><td colspan="10">Complete FL methods (method-specific training and classifiers)</td></tr><tr><td>FedAvg [14]</td><td>AISTATS&#x27;17</td><td>52.25±0.25</td><td> $4 7 . 3 2 \pm 1 . 0 8$ </td><td> $4 0 . 8 1 \pm 2 . 3 9$ </td><td> $4 0 . 4 0 { \scriptstyle \pm 1 . 9 4 }$ </td><td> $2 1 . 8 7 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $3 5 . 8 8 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $3 4 . 5 1 { \scriptstyle \pm 0 . 0 8 }$ </td><td>21.84±0.92</td></tr><tr><td>FedRep [15]</td><td>ICML&#x27;21</td><td> $6 9 . 5 4 \pm 0 . 5 1$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 2 4 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $7 8 . 6 5 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>0.05±0.04</td><td>0.00±0.00</td></tr><tr><td>FedROD [16]</td><td>ICLR&#x27;22</td><td> $7 2 . 4 3 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $5 3 . 7 6 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $4 3 . 2 0 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $4 1 . 8 1 { \pm } 0 . 5 8$ </td><td> $7 9 . 8 7 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $3 8 . 1 9 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $2 3 . 8 7 { \scriptstyle \pm 1 . 1 4 }$ </td><td> $2 2 . 7 3 { \scriptstyle \pm 1 . 2 1 }$ </td></tr><tr><td>FedLC [17]</td><td>ICML&#x27;22</td><td> $4 7 . 9 1 \pm 0 . 6 2$ </td><td> $5 4 . 3 3 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $4 1 . 3 7 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $3 9 . 1 9 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $1 5 . 9 5 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $3 3 . 6 9 { \scriptstyle \pm 0 . 5 4 }$ </td><td>15.79±1.04</td><td> $1 0 . 4 8 { \scriptstyle \pm 1 . 0 9 }$ </td></tr><tr><td>FedTHE [18]</td><td>ICLR&#x27;23</td><td> $6 4 . 2 1 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $5 0 . 1 2 { \scriptstyle \pm 1 . 2 7 }$ </td><td> $3 8 . 9 6 \pm 1 . 6 0 $ </td><td> $3 8 . 8 1 \pm 1 . 5 5$ </td><td> $7 2 . 4 4 \pm 1 . 8 2$ </td><td> $3 5 . 6 0 { \scriptstyle \pm 0 . 6 5 }$ </td><td>28.31±0.48</td><td>27.68±0.52</td></tr><tr><td>FedETF [19]</td><td>ICCV&#x27;23</td><td> $6 1 . 7 6 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $6 4 . 7 6 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $5 4 . 4 8 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $5 2 . 8 8 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $3 9 . 2 6 { \scriptstyle \pm 2 . 7 6 }$ </td><td> $4 7 . 0 1 \pm 1 . 1 2$ </td><td> $4 2 . 4 0 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $3 3 . 8 4 \pm 4 . 6 1$ </td></tr><tr><td>FedFA [20]</td><td>TMC&#x27;24</td><td> $2 4 . 2 9 { \scriptstyle \pm 5 . 9 0 }$ </td><td> $2 0 . 4 3 { \scriptstyle \pm 6 . 5 2 }$ </td><td> $2 0 . 9 4 \pm 1 0 . 7 7$ </td><td> $1 4 . 4 7 { \scriptstyle \pm 6 . 9 2 }$ </td><td> $1 . 1 8 { \pm } 1 . 1 4$ </td><td> $5 . 6 1 \pm 1 . 4 6$ </td><td>2.53±2.03</td><td>0.35±0.31</td></tr><tr><td>pFedFDA [21]</td><td>NeurIPS&#x27;24</td><td> $7 1 . 8 9 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $2 . 0 7 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $4 . 6 0 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $1 . 7 0 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $7 9 . 3 9 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td>0.42±0.14</td><td>0.00±0.01</td></tr><tr><td>FedVLS [22]</td><td>AAAI&#x27;25</td><td> $4 7 . 1 7 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $5 5 . 6 6 \pm 1 . 2 6$ </td><td> $4 0 . 0 7 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $3 7 . 4 3 { \scriptstyle \pm 1 . 0 4 }$ </td><td> $2 1 . 2 6 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $3 5 . 7 6 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $1 8 . 9 6 \pm 0 . 2 6$ </td><td>14.10±0.47</td></tr><tr><td>FedDr+ [23]</td><td>TMLR&#x27;25</td><td> $6 1 . 8 6 \pm 1 . 3 5$ </td><td> $5 7 . 9 5 { \pm } 2 . 5 6$ </td><td> $5 1 . 7 5 { \scriptstyle \pm 2 . 2 1 }$ </td><td> $5 1 . 2 2 { \scriptstyle \pm 2 . 0 5 }$ </td><td> $3 . 6 6 \pm 1 . 8 3$ </td><td> $1 1 . 1 2 \pm 3 . 8 3$ </td><td>6.27±6.37</td><td>1.66±1.56</td></tr><tr><td>BTFL [24]</td><td>KDD&#x27;25</td><td> $7 2 . 0 0 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $1 8 . 5 0 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $3 1 . 7 8 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $1 8 . 5 0 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $8 0 . 8 2 { \scriptstyle \pm 1 . 0 9 } ^ { * }$ </td><td> $0 . 8 4 \pm 0 . 4 4$ </td><td> $1 2 . 9 4 { \scriptstyle \pm 1 . 2 0 }$ </td><td>0.84±0.44</td></tr><tr><td>HPFL [25]</td><td>ICLR&#x27;25</td><td> $7 2 . 6 4 \pm 0 . 6 9$ </td><td> $3 0 . 2 9 { \scriptstyle \pm 2 . 1 8 }$ </td><td> $1 9 . 9 7 { \scriptstyle \pm 2 . 2 7 }$ </td><td> $1 7 . 6 8 { \scriptstyle \pm 1 . 6 4 }$ </td><td> ${ \mathbf 8 0 . 9 5 \pm 0 . 8 5 } ^ { \dagger }$ </td><td> $1 7 . 5 9 { \pm 2 . 3 3 }$ </td><td> $1 1 . 6 3 { \pm } 2 . 3 2$ </td><td>10.86±2.31</td></tr><tr><td>FedYoYo [26]</td><td>ICCV&#x27;25</td><td> $7 2 . 3 5 { \pm } 0 . 3 9$ </td><td> $7 3 . 1 5 { \pm } 1 . 5 2$ </td><td> $6 8 . 5 2 { \scriptstyle \pm 1 . 1 3 }$ </td><td> $6 8 . 1 8 { \pm } 1 . 0 5$ </td><td> $3 0 . 6 2 \pm 3 . 2 2$ </td><td> $4 4 . 4 9 { \scriptstyle \pm 2 . 1 7 }$ </td><td> $4 7 . 4 6 \pm 1 . 0 1$ </td><td>28.23±3.75</td></tr><tr><td>FedLog [27]</td><td>TMLR&#x27;26</td><td> $6 6 . 3 0 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 1 9 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $7 8 . 4 6 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>0.09±0.04</td><td>0.00±0.00</td></tr><tr><td>pFedGM [28]</td><td>arXiv′26</td><td>73.31±0.59</td><td>0.00±0.00</td><td> $1 . 9 2 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $7 7 . 8 0 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 8 { \scriptstyle \pm 0 . 0 6 }$ </td><td>0.00±0.00</td></tr><tr><td colspan="10">Statistical classifiers assembled on a vanilla backbone (FedAvg +)</td></tr><tr><td>+ CCVR [29]</td><td>NeurIPS&#x27;21</td><td> $5 5 . 5 8 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 5 . 3 7 { \scriptstyle \pm 1 . 5 7 }$ </td><td> $4 9 . 9 4 { \scriptstyle \pm 1 . 7 3 }$ </td><td> $4 7 . 9 9 \pm 1 . 9 0$ </td><td> $4 0 . 8 9 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $4 6 . 7 6 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $4 4 . 7 9 2 0 . 8 4$ </td><td> $3 9 . 1 7 { \scriptstyle \pm 0 . 8 8 }$ </td></tr><tr><td>+ FedNCM [30]</td><td>NeurIPS&#x27;23</td><td> $5 5 . 2 5 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $5 6 . 9 8 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $4 9 . 3 1 \pm 1 . 8 1$ </td><td> $4 8 . 5 1 \pm 1 . 6 1$ </td><td> $4 2 . 2 8 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $4 6 . 9 8 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $4 5 . 0 8 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $4 0 . 6 0 { \scriptstyle \pm 1 . 0 8 }$ </td></tr><tr><td>+ Fed3R [31]</td><td>ICML&#x27;24</td><td> $5 4 . 9 6 \pm 1 . 0 3$ </td><td> $5 4 . 4 4 { \scriptstyle \pm 1 . 5 4 }$ </td><td> $4 8 . 3 2 { \scriptstyle \pm 2 . 5 3 }$ </td><td> $4 4 . 8 4 \pm 2 . 2 7$ </td><td> $4 3 . 4 1 \pm 0 . 6 4$ </td><td> $4 8 . 1 2 \pm 0 . 8 7$ </td><td> $4 7 . 2 0 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $4 1 . 3 5 { \scriptstyle \pm 0 . 5 7 }$ </td></tr><tr><td>+ FedCGS [32]</td><td>AAAI&#x27;25</td><td> $5 6 . 5 2 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $5 7 . 1 2 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $5 2 . 0 3 { \pm } 1 . 4 9$ </td><td> $5 0 . 8 0 { \scriptstyle \pm 1 . 6 4 }$ </td><td> $4 6 . 3 7 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $5 1 . 2 3 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $4 7 . 9 8 \pm 1 . 0 0$ </td><td> $4 4 . 8 9 { \scriptstyle \pm 0 . 3 4 }$ </td></tr><tr><td>+ FedCOF [33]</td><td>NeurIPS&#x27;25</td><td> $4 8 . 0 1 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $4 5 . 6 3 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $4 1 . 2 2 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $3 9 . 5 3 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $3 6 . 6 0 { \scriptstyle \pm 1 . 0 7 }$ </td><td> $4 1 . 6 1 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $4 0 . 0 5 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $3 4 . 6 6 \pm 1 . 3 2$ </td></tr><tr><td> $+ \mathrm { G H - L D A } \left[ 3 4 \right]$ </td><td>ICLR&#x27;26</td><td> $5 6 . 7 4 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $5 6 . 3 9 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $5 2 . 1 3 { \scriptstyle \pm 1 . 7 9 }$ </td><td> $5 0 . 6 3 { \scriptstyle \pm 1 . 8 3 }$ </td><td> $4 6 . 4 8 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $5 1 . 3 4 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $4 8 . 7 4 \pm 1 . 0 8$ </td><td> $4 5 . 2 4 \pm 0 . 4 1$ </td></tr><tr><td colspan="10">Statistical classifiers assembled on the SOTA backbone (FedYoYo +)</td></tr><tr><td>+ CCVR [29]</td><td>NeurIPS&#x27;21</td><td> $7 1 . 8 7 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $7 3 . 6 8 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $6 7 . 0 6 \pm 0 . 4 4$ </td><td> $6 5 . 8 9 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $5 6 . 4 0 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $6 1 . 5 6 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $5 9 . 1 7 { \scriptstyle \pm 0 . 7 7 }$ </td><td></td></tr><tr><td>+ FedNCM [30]</td><td>NeurIPS&#x27;23</td><td> $7 0 . 8 3 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $7 2 . 1 7 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $6 6 . 8 9 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $6 4 . 7 2 { \scriptstyle \pm 1 . 1 1 }$ </td><td> $5 3 . 0 1 { \scriptstyle \pm 0 . 9 1 }$ </td><td> $5 6 . 1 8 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $5 5 . 6 2 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $5 3 . 2 5 { \scriptstyle \pm 1 . 3 2 }$ </td></tr><tr><td>+ Fed3R [31]</td><td>ICML&#x27;24</td><td> $7 2 . 7 0 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $7 4 . 2 7 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $6 6 . 9 8 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $6 5 . 1 0 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $5 6 . 5 6 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $6 2 . 7 4 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $6 0 . 9 3 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $5 0 . 4 3 { \scriptstyle \pm 0 . 7 2 }$   $5 2 . 4 9 { \scriptstyle \pm 2 . 0 1 }$ </td></tr><tr><td>+ FedCGS [32]</td><td>AAAI&#x27;25</td><td> $7 3 . 5 7 { \scriptstyle \pm 0 . 6 6 } ^ { * }$ </td><td> $7 5 . 3 6 \pm 0 . 1 7$ </td><td> $7 0 . 7 5 { \scriptstyle \pm 0 . 3 3 } ^ { \ast }$ </td><td> $6 9 . 7 2 { \scriptstyle \pm 0 . 6 9 } ^ { * }$ </td><td> $6 2 . 7 6 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $6 5 . 9 7 { \scriptstyle \pm 0 . 4 8 } ^ { * }$ </td><td> $6 3 . 6 3 { \scriptstyle \pm 0 . 5 4 } ^ { * }$ </td><td> ${ \bf 6 1 . 0 3 } { \scriptstyle \pm 0 . 1 4 } ^ { * }$ </td></tr><tr><td>+ FedCOF [33]</td><td>NeurIPS&#x27;25</td><td> $6 9 . 8 6 \pm 0 . 5 5$ </td><td> $6 9 . 9 6 \pm 1 . 0 6 $ </td><td> $6 5 . 2 0 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $6 2 . 4 7 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $5 3 . 2 3 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $5 7 . 9 0 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $5 7 . 1 5 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $5 0 . 8 8 \pm 1 . 2 2$ </td></tr><tr><td>+ GH-LDA [34]</td><td>ICLR&#x27;26</td><td> $7 3 . 5 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $7 5 . 4 5 { \scriptstyle \pm 0 . 2 8 } ^ { * }$ </td><td> $7 0 . 7 1 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $6 9 . 6 3 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $6 1 . 2 0 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $6 4 . 9 2 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $6 2 . 8 3 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $5 9 . 4 0 { \scriptstyle \pm 0 . 5 8 }$ </td></tr><tr><td>FedGRC (Ours)</td><td></td><td> $7 6 . 9 7 { \scriptstyle \pm 0 . 5 6 } ^ { \dagger }$ </td><td></td><td> $7 2 . 7 8 { \scriptstyle \pm 0 . 6 0 } ^ { \dagger }$ </td><td> ${ \bf 7 1 . 8 5 { \scriptstyle \pm 0 . 6 6 } } ^ { \dagger }$ </td><td> $6 7 . 3 8 { \scriptstyle \pm 0 . 3 7 }$ </td><td> ${ \bf 7 1 . 2 0 { \scriptstyle \pm 0 . 3 2 } } ^ { \dagger }$ </td><td> $6 5 . 9 5 { \scriptstyle \pm 0 . 3 5 } ^ { \dagger }$ </td><td></td></tr><tr><td colspan="2">∆ vs. Best</td><td>+3.40</td><td> $7 9 . 2 2 \pm 0 . 8 0 ^ { \dagger }$  +3.77</td><td>+2.03</td><td>+2.13</td><td>-13.57</td><td>+5.23</td><td>+2.32</td><td> $6 3 . 8 5 { \scriptstyle \pm 0 . 1 3 } ^ { \dagger }$  +2.82</td></tr></table>

TABLE 12: Client recognition floors on CIFAR-100 [1]. L@25, U@25, R@25, and F@25 average the lowest 25% (5 of 20 clients), with clients ranked separately for each metric.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="4">CIFAR-100, Dir(0.3), 20 clients</td><td colspan="4">CIFAR-100, Dir(0.1), 20 clients</td></tr><tr><td>L@25</td><td>U@25</td><td>R@25</td><td>F@25</td><td>L@25</td><td>U@25</td><td>R@25</td><td>F@25</td></tr><tr><td colspan="10">Complete FL methods (method-specific training and classifiers)</td></tr><tr><td>FedAvg [14]</td><td>AISTATS&#x27;17</td><td>32.40±0.99</td><td> $3 0 . 0 5 { \scriptstyle \pm 1 . 2 3 }$ </td><td> $2 3 . 2 8 \pm 4 . 2 5$ </td><td> $2 3 . 0 0 { \scriptstyle \pm 4 . 1 3 }$ </td><td> $2 7 . 0 8 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $3 0 . 6 5 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 7 . 1 1 \pm 1 . 8 3$ </td><td> $1 7 . 1 1 \pm 1 . 8 3$ </td></tr><tr><td>FedRep [15]</td><td>ICML&#x27;21</td><td> $3 3 . 7 5 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 4 . 2 2 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>FedROD [16]</td><td>ICLR&#x27;22</td><td> $4 4 . 8 8 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $3 1 . 9 5 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $2 3 . 6 7 { \scriptstyle \pm 1 . 7 7 }$ </td><td> $2 3 . 6 7 { \scriptstyle \pm 1 . 7 7 }$ </td><td> $5 2 . 5 9 { \scriptstyle \pm 1 . 2 6 } ^ { \dagger }$ </td><td> $3 2 . 0 1 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $1 8 . 2 2 { \scriptstyle \pm 1 . 8 0 }$ </td><td> $1 8 . 2 2 { \scriptstyle \pm 1 . 8 0 }$ </td></tr><tr><td>FedLC [17]</td><td>ICML&#x27;22</td><td> $2 9 . 5 5 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $2 9 . 9 4 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $2 0 . 5 0 { \scriptstyle \pm 1 . 5 3 }$ </td><td> $2 0 . 5 0 { \scriptstyle \pm 1 . 5 3 }$ </td><td> $2 3 . 5 5 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $3 0 . 2 9 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $1 6 . 8 9 { \scriptstyle \pm 1 . 1 1 }$ </td><td> $1 6 . 7 2 { \scriptstyle \pm 0 . 8 1 }$ </td></tr><tr><td>FedTHE [18]</td><td>ICLR&#x27;23</td><td> $3 3 . 1 0 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $2 5 . 6 1 \pm 0 . 4 7$ </td><td> $1 9 . 0 6 { \pm } 0 . 5 9$ </td><td> $1 8 . 9 4 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $3 9 . 1 1 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $2 6 . 0 7 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $1 4 . 7 8 { \scriptstyle \pm 1 . 5 5 }$ </td><td> $1 4 . 7 8 { \scriptstyle \pm 1 . 5 5 }$ </td></tr><tr><td>FedETF [19]</td><td>ICCV&#x27;23</td><td> $3 5 . 2 6 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $3 2 . 9 9 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $2 4 . 7 8 { \scriptstyle \pm 2 . 8 7 }$ </td><td> $2 4 . 0 2 { \scriptstyle \pm 3 . 0 5 }$ </td><td> $3 1 . 8 2 { \pm } 1 . 6 1 $ </td><td> $3 5 . 5 3 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $2 2 . 1 7 { \scriptstyle \pm 1 . 4 3 }$ </td><td> $2 2 . 1 7 { \scriptstyle \pm 1 . 4 3 }$ </td></tr><tr><td>FedFA [20]</td><td>TMC&#x27;24</td><td> $4 . 6 7 \pm 2 . 8 9$ </td><td> $4 . 4 3 { \scriptstyle \pm 1 . 7 4 }$ </td><td> $3 . 7 8 \pm 4 . 5 5$ </td><td> $2 . 4 1 \pm 2 . 5 4$ </td><td> $0 . 5 8 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $1 . 7 1 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>pFedFDA [21]</td><td>NeurIPS&#x27;24</td><td> $4 2 . 3 9 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $1 2 . 0 0 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 0 . 8 3 { \pm } 2 . 5 8 $ </td><td> $9 . 6 2 \pm 1 . 8 8$ </td><td> $4 9 . 5 7 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $1 0 . 9 3 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $7 . 1 7 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $7 . 1 7 { \scriptstyle \pm 0 . 4 4 }$ </td></tr><tr><td>FedVLS [22]</td><td>AAAI&#x27;25</td><td> $2 6 . 9 8 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $2 9 . 5 9 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $1 9 . 0 6 \pm 1 . 0 9$ </td><td> $1 9 . 0 6 \pm 1 . 0 9$ </td><td> $2 3 . 2 0 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $2 9 . 2 1 \pm 0 . 8 8$ </td><td> $1 4 . 8 4 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $1 4 . 7 6 { \scriptstyle \pm 0 . 3 7 }$ </td></tr><tr><td>FedDr+ [23]</td><td>TMLR&#x27;25</td><td> $3 4 . 9 2 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $3 1 . 9 7 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $2 7 . 5 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $2 5 . 2 1 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $2 8 . 1 1 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $3 0 . 7 9 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $2 0 . 6 7 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $2 0 . 6 7 { \scriptstyle \pm 0 . 7 3 }$ </td></tr><tr><td>BTFL [24]</td><td>KDD&#x27;25</td><td> $3 6 . 2 8 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $2 3 . 7 6 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $2 2 . 1 1 { \pm } 0 . 3 5$ </td><td> $2 0 . 8 5 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $4 4 . 2 4 { \scriptstyle \pm 0 . 9 8 }$ </td><td> $2 2 . 8 7 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $1 7 . 9 5 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $1 7 . 9 1 { \scriptstyle \pm 0 . 7 0 }$ </td></tr><tr><td>HPFL [25]</td><td>ICLR&#x27;25</td><td> $4 0 . 1 4 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $2 8 . 0 8 { \scriptstyle \pm 1 . 1 8 }$ </td><td> $2 0 . 5 6 { \scriptstyle \pm 3 . 1 3 }$ </td><td> $2 0 . 4 7 { \scriptstyle \pm 3 . 2 0 }$ </td><td> $4 7 . 9 0 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $2 5 . 1 5 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $1 4 . 7 8 { \scriptstyle \pm 1 . 1 3 }$ </td><td> $1 4 . 7 8 { \scriptstyle \pm 1 . 1 3 }$ </td></tr><tr><td>FedYoYo [26]</td><td>ICCV&#x27;25</td><td> $3 2 . 5 3 { \pm } 2 . 0 2$ </td><td> $3 3 . 4 4 \pm 2 . 2 4$ </td><td> $1 3 . 1 7 \pm 3 . 0 8$ </td><td> $1 3 . 1 7 { \scriptstyle \pm 3 . 0 8 }$ </td><td> $3 5 . 4 6 { \pm } 1 . 7 8$ </td><td> $3 7 . 7 2 { \scriptstyle \pm 2 . 1 0 }$ </td><td> $2 8 . 6 6 \pm 5 . 6 2$ </td><td> $2 8 . 3 9 { \scriptstyle \pm 5 . 4 8 }$ </td></tr><tr><td>FedLog [27]</td><td>TMLR&#x27;26</td><td> $3 0 . 5 3 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $4 2 . 0 6 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>pFedGM [28]</td><td>arXiv′26</td><td> $4 1 . 5 4 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $4 . 2 4 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $1 0 . 6 1 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $4 . 2 1 { \pm } 0 . 3 8$ </td><td> $\mathbf { 5 0 . 1 5 { \scriptstyle \pm 0 . 3 5 } ^ { \ast } }$ </td><td> $0 . 7 0 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $4 . 7 8 \pm 1 . 2 7$ </td><td> $0 . 7 0 { \scriptstyle \pm 0 . 1 0 }$ </td></tr><tr><td colspan="10">Statistical classifiers assembled on a vanilla backbone (FedAvg +)</td></tr><tr><td>+ CCVR [29]</td><td>NeurIPS&#x27;21</td><td> $3 1 . 9 9 { \scriptstyle \pm 0 . 9 9 }$ </td><td> $3 1 . 7 8 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $2 2 . 3 3 { \scriptstyle \pm 2 . 9 7 }$ </td><td> $2 2 . 3 3 { \scriptstyle \pm 2 . 9 7 }$ </td><td> $2 7 . 9 2 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $3 2 . 4 2 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $1 8 . 8 3 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $1 8 . 8 3 { \scriptstyle \pm 0 . 7 3 }$ </td></tr><tr><td>+ FedNCM [30]</td><td>NeurIPS&#x27;23</td><td> $3 3 . 0 5 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $3 2 . 5 5 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $2 4 . 3 3 { \scriptstyle \pm 3 . 0 9 }$ </td><td> $2 4 . 2 7 { \scriptstyle \pm 3 . 1 0 }$ </td><td> $2 8 . 4 1 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $3 3 . 2 1 \pm 0 . 0 9$ </td><td> $1 8 . 7 8 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $1 8 . 7 8 { \scriptstyle \pm 1 . 0 0 }$ </td></tr><tr><td>+ Fed3R [31]</td><td>ICML&#x27;24</td><td> $3 1 . 2 5 { \scriptstyle \pm 1 . 1 6 }$ </td><td> $2 9 . 6 9 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $2 1 . 8 3 { \scriptstyle \pm 3 . 4 6 }$ </td><td> $2 1 . 7 2 { \scriptstyle \pm 3 . 3 8 }$ </td><td> $2 6 . 9 8 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $3 1 . 5 1 { \pm } 0 . 1 6$ </td><td> $1 6 . 7 8 { \scriptstyle \pm 1 . 3 9 }$ </td><td> $1 6 . 7 8 { \scriptstyle \pm 1 . 3 9 }$ </td></tr><tr><td>+ FedCGS [32]</td><td> $\mathrm { \ A A A I ^ { \prime } } 2 5$ </td><td> $3 0 . 5 6 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $3 0 . 9 4 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $2 1 . 6 7 { \scriptstyle \pm 3 . 3 8 }$ </td><td> $2 1 . 6 7 { \scriptstyle \pm 3 . 3 8 }$ </td><td> $2 7 . 3 5 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $3 1 . 5 6 { \pm } 0 . 5 9$ </td><td> $1 6 . 5 0 { \scriptstyle \pm 1 . 5 9 }$ </td><td> $1 6 . 5 0 { \scriptstyle \pm 1 . 5 9 }$ </td></tr><tr><td>+ FedCOF [33]</td><td> $\mathrm { N e u r I P S ^ { \prime } } 2 5$ </td><td> $3 2 . 1 7 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $3 1 . 2 2 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $2 3 . 8 9 \pm 3 . 7 3$ </td><td> $2 3 . 8 9 { \scriptstyle \pm 3 . 7 3 }$ </td><td> $2 7 . 6 6 \pm 0 . 3 9$ </td><td> $3 1 . 8 8 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $1 7 . 8 9 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $1 7 . 8 9 { \scriptstyle \pm 0 . 7 5 }$ </td></tr><tr><td> $+ \mathrm { G H - L D A } \left[ 3 4 \right]$ </td><td>ICLR&#x27;26</td><td> $3 1 . 4 5 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $3 2 . 0 4 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $2 2 . 7 8 { \scriptstyle \pm 3 . 2 6 }$ </td><td> $2 2 . 7 8 { \scriptstyle \pm 3 . 2 6 }$ </td><td> $2 8 . 1 2 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $3 2 . 3 7 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $1 8 . 2 8 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $1 8 . 2 8 { \scriptstyle \pm 0 . 2 5 }$ </td></tr><tr><td colspan="10">Statistical classifiers assembled on the SOTA backbone  $( F e d Y o Y o + )$ </td></tr><tr><td>+ CCVR [29]</td><td> $\mathrm { N e u r I P S ^ { \prime } } 2 1$ </td><td> $4 9 . 0 9 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $4 7 . 5 2 { \scriptstyle \pm 0 . 3 5 } ^ { * }$ </td><td> $4 2 . 2 8 { \scriptstyle \pm 2 . 0 4 } ^ { * }$ </td><td> $\mathbf { 4 1 . 1 0 { \scriptstyle \pm 1 . 3 6 } } ^ { \ast }$ </td><td> $4 4 . 6 1 \pm 0 . 3 7$ </td><td> $4 8 . 1 1 \pm 0 . 2 2$ </td><td> $3 6 . 2 2 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $3 6 . 2 2 { \scriptstyle \pm 0 . 6 7 }$ </td></tr><tr><td> $+ \mathrm { F e d N C M } \bar { [ 3 0 ] }$ </td><td>NeurIPS&#x27;23</td><td> $4 6 . 1 9 2 0 . 6 3$ </td><td> $4 3 . 3 6 \pm 0 . 2 1$ </td><td> $3 8 . 0 0 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $3 6 . 4 3 { \scriptstyle \pm 1 . 1 8 }$ </td><td> $3 9 . 6 8 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $4 2 . 9 4 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $2 8 . 1 1 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $2 8 . 1 1 { \scriptstyle \pm 0 . 4 2 }$ </td></tr><tr><td>+ Fed3R [31]</td><td> $\mathrm { I C M L } ^ { \prime } 2 4$ </td><td> $4 8 . 5 7 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $4 5 . 9 1 \pm 0 . 1 0$ </td><td> $3 8 . 2 8 { \scriptstyle \pm 2 . 1 7 }$ </td><td> $3 6 . 3 9 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $4 4 . 0 4 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $4 7 . 8 0 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $3 1 . 7 2 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $3 1 . 7 2 { \scriptstyle \pm 1 . 0 8 }$ </td></tr><tr><td>+ FedCGS [32]</td><td>AAAI&#x27;25</td><td> $\mathbf { 4 9 . 3 7 { \scriptstyle \pm 0 . 2 1 } ^ { * } }$ </td><td> $4 7 . 5 0 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $4 1 . 3 9 { \scriptstyle \pm 1 . 5 5 }$ </td><td> $4 0 . 9 2 \pm 1 . 3 2$ </td><td> $4 4 . 8 6 \pm 0 . 7 2$ </td><td> $4 8 . 3 8 { \scriptstyle \pm 0 . 7 2 } ^ { * }$ </td><td> $3 8 . 5 5 { \scriptstyle \pm 1 . 4 9 } ^ { * }$ </td><td> $3 8 . 4 9 { \scriptstyle \pm 1 . 4 4 } ^ { * }$ </td></tr><tr><td>+ FedCOF [33]</td><td>NeurIPS&#x27;25</td><td> $4 7 . 6 8 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $4 5 . 2 4 \pm 0 . 1 7$ </td><td> $4 1 . 2 2 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $3 9 . 6 8 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $4 2 . 9 8 { \pm } 1 . 1 0$ </td><td> $4 4 . 7 2 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $3 3 . 6 1 { \scriptstyle \pm 0 . 9 8 }$ </td><td> $3 3 . 6 1 { \scriptstyle \pm 0 . 9 8 }$ </td></tr><tr><td> $+ \mathrm { G H - L D A } \left[ 3 4 \right]$ </td><td>ICLR&#x27;26</td><td> $4 8 . 6 0 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $4 6 . 7 7 \pm 0 . 1 7$ </td><td> $4 0 . 6 7 \pm 1 . 3 4$ </td><td> $3 9 . 9 7 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $4 4 . 5 4 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $4 7 . 6 3 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $3 6 . 8 9 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $3 6 . 8 2 { \scriptstyle \pm 1 . 1 1 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">FedGRC (Ours) ∆ vs. Best</td><td> $5 2 . 6 8 { \scriptstyle \pm 0 . 5 6 } ^ { \dagger }$   $+ 3 . 3 1$ </td><td> $5 2 . 9 0 { \scriptstyle \pm 0 . 8 3 } ^ { \dagger }$ </td><td> $4 8 . 3 3 { \scriptstyle \pm 1 . 8 4 } ^ { \dagger }$ </td><td> $4 7 . 5 6 \pm 1 . 8 3 ^ { \dagger }$   $+ 6 . 4 6$ </td><td> $4 9 . 0 0 { \scriptstyle \pm 0 . 8 5 }$   $- 3 . 5 9$ </td><td> $5 2 . 5 8 { \scriptstyle \pm 0 . 4 3 } ^ { \dagger }$   $+ 4 . 2 0$ </td><td> $4 1 . 8 3 \pm 1 . 5 3 ^ { \dagger }$   $+ 3 . 2 8$ </td><td> $\mathbf { 4 1 . 8 3 \pm } 1 . 5 3 ^ { \dagger }$   $+ 3 . 3 4$ </td></tr></table>

## B.3 Communication and Computational Overhead

## B.3.1 Communication Cost

## B.3.2 Computational Cost

As in FedAvg [14], the server broadcasts the current global model parameters to participating clients, which upload their updated parameters after local training. FedGRC adds $O ( C d _ { L } ^ { - } + C | \bar { \Lambda | } )$ scalar entries in each direction per participating client per round: clients upload class directions, log-norm sums, and counts, while the server broadcasts updated directional and magnitude references alongside the model parameters.

CGRL processes two augmented views per example, adding one view’s forward and backward computation over singleview FedAvg for the same examples and updates. The feature-level regularizers cost $O ( \dot { B } ^ { 2 } d _ { L } \ : + \ : B \dot { C } d _ { L } \ : + \ : B d )$ per update for contrastive, class-direction, and magnitude terms, respectively. After local training in each participating round, one additional forward-only pass over the local training set extracts class directions; magnitude statistics are accumulated during training. Server reference updates cost $O ( | S _ { t } | C ( d _ { L } + | \Lambda | ) )$ per round.

After training, all K clients participate once in the additive-moment construction of Appendix A.5. Each uploads class counts, class-wise first moments, and a single $d \times d$ second-moment matrix summed over all local samples, totaling $O ( C d + d ^ { 2 } )$ entries. From the uploaded moments, the server derives the shared layer scales, class means, and covariance metric. It broadcasts these quantities in $O ( d ^ { 2 } + C d + | \Lambda | )$ entries per client, after which each client constructs its personalized head locally. For fixed feature dimensions and class count, this terminal payload is independent of local sample size. Thus, MMS and SASP together require only one additional communication round to construct the personalized statistical heads. In our experiments, FedGRC’s total communication volume is 0.08%, 0.41%, and 0.25% higher than that of FedAvg on CIFAR-10, CIFAR-100, and FedRS, respectively.

To construct the personalized statistical heads with MMS and SASP, each client makes one pass over its training set using the frozen backbone; moment accumulation adds $O ( N ^ { ( k ) } \big )$ operations. Aggregation, covariance construction, and construction of all personalized heads together cost $O ( d ^ { 3 } + K C d ^ { 2 } )$ across the server and clients with dense arithmetic. The server inverts the shared covariance once; each client constructs its head in $O ( C d ^ { 2 } )$ time.

At inference, multilayer features come from a single backbone pass, followed by layer scaling and an $O ( \breve { C } d )$ linear head, compared with Fed $\mathsf { \bar { A } v g ^ { \prime } s } O ( C \mathsf { \bar { d } } _ { L } )$ head. The covariance metric is incorporated into the precomputed head parameters. Thus, FedGRC incurs negligible computational overhead at inference relative to FedAvg.

TABLE 13: Notation for the FedGRC framework, theoretical analysis, and experimental evaluation.
<table><tr><td>Symbol(s)</td><td>Definition</td></tr><tr><td colspan="2">A. Federated Learning Problem Formulation</td></tr><tr><td> $K , C , y$ </td><td>Client count, class count, and common federation-known label space.</td></tr><tr><td> $\mathcal { D } ^ { ( k ) } , \mathcal { D } _ { c } ^ { ( k ) } , \mathcal { T } _ { c } ^ { ( k ) }$ </td><td>Client  $k ^ { \prime } \mathrm { s }$  training set, class-c subset, and indices of that subset.</td></tr><tr><td> $N ^ { ( k ) } , N ; n _ { c } ^ { ( k ) } , N _ { c }$ </td><td>Training-sample counts for each client and the federation, both overall and for class c.</td></tr><tr><td> $\pmb \theta = ( \phi , \psi )$ </td><td>Model parameters: backbone  $\phi$  and classifier head  $\psi .$ </td></tr><tr><td> $g , h _ { \phi } , q _ { \psi }$ </td><td>Full model, feature extractor, and classifier head;  $g = q _ { \psi } \circ h _ { \phi } .$ </td></tr><tr><td> $P ^ { ( k ) } , Q _ { c } , Q$ </td><td>Local target distribution, class-c evaluation distribution, and class-balanced all-class distribution.</td></tr><tr><td> $\mathcal { E } _ { R } ( f ) , e _ { c } ^ { ( k ) }$ </td><td>Error probability of  $f$  under labeled distribution  $R ;$  client  $k ^ { \prime } \mathrm { s }$  class-c error under  $Q _ { c } ;$  local and</td></tr><tr><td> $\mathcal { E } _ { \mathrm { l o c a l } } ^ { ( k ) } , \mathcal { E } _ { \mathrm { g l o b a l } } ^ { ( k ) }$ </td><td>global risks under  $P ^ { ( k ) }$  and  $Q ,$  respectively.</td></tr><tr><td>B. Consensus-Guided Representation Learning</td><td></td></tr><tr><td colspan="2"></td></tr><tr><td> $\Lambda , L , d _ { \ell } , d$ </td><td>Selected layers, final layer, pooled feature dimension at  $\ell ,$  and total dimension  $\begin{array} { r } { d = \sum _ { \ell \in \Lambda } d _ { \ell } . } \end{array}$  Pooled layer features and unit directions; directional supervision uses the final layer.</td></tr><tr><td> ${ \bf h } _ { \ell } ( x ; \phi ) , { \bf u } _ { i }$ </td><td>Raw norm  $R _ { i } = \| \mathbf { h } _ { i } \| _ { 2 }$  , safe norm  $r _ { \ell } ( x ) = \sqrt { \| \mathbf { h } _ { \ell } ( x ; \phi ) \| _ { 2 } ^ { 2 } + \varepsilon ^ { 2 } }$  , and log norm  $\rho _ { \ell } ( x ) = \log r _ { \ell } ( x )$ </td></tr><tr><td> $R _ { i } , r _ { \ell } ( x ) , \rho _ { \ell } ( x )$ </td><td>magnitude guidance.</td></tr><tr><td> $\mathbf { p } _ { c } ( t ) , a _ { \ell , c } ( t )$ </td><td>Shared class direction and class- and layer-specific log-norm reference at round t.</td></tr><tr><td> $v _ { c } ( t ) , \chi _ { \ell i }$ </td><td>Magnitude reference initialization flag for class  $c ;$  penalty indicator for layer l and input i.</td></tr><tr><td> $B , B _ { 2 } , \mathcal { P } _ { i }$ </td><td>Batch size, index set of 2B paired views, and same-class positive views excluding i.</td></tr><tr><td> $\mathcal { T } _ { \mathrm { w } } , \mathcal { T } _ { \mathrm { s } }$   $\mathcal { L } _ { \mathrm { C E } } , \mathcal { L } _ { \mathrm { C L } } , \mathcal { L } _ { \mathrm { p r o t o } }$ </td><td>Label-preserving weak and strong augmentation operators. Training-head cross-entropy, SupCon loss, and shared-prototype cross-entropy.</td></tr><tr><td> $\mathcal { L } _ { \mathrm { d i r } } , \mathcal { L } _ { \mathrm { r a d } } , \mathcal { L } _ { \mathrm { C G R L } }$ </td><td>Directional loss (contrastive and prototype), log-norm loss, and full local objective.</td></tr><tr><td></td><td></td></tr><tr><td colspan="2">C. Statistical Modeling for Personalized Classification</td></tr><tr><td> $s _ { \ell } , \mathbf { z } ( x )$ </td><td>Shared layer scale and concatenated scaled features  $\mathbf { z } ( x ) = { \mathrm { c o l } } _ { \ell \in \Lambda } ( \mathbf { h } _ { \ell } / s _ { \ell } ) .$ </td></tr><tr><td> $\mu _ { c } , \tilde { \mu _ { c } } ^ { ( \acute { k } ) } , \widetilde { \mu } _ { c } ^ { ( \acute { k } ) }$ </td><td>Shared, local, and support-dependent shrinkage centers; locally absent classes retain</td></tr><tr><td> $\Delta _ { c } ^ { ( k ) } , \delta _ { c } ^ { ( k ) }$ </td><td>Local-minus-shared mean displacement (zero for absent classes) and applied shift</td></tr><tr><td> $m _ { 0 } , \alpha _ { c } ^ { ( k ) }$ </td><td> $\delta _ { c } ^ { ( k ) } = \alpha _ { c } ^ { ( k ) } \Delta _ { c } ^ { ( k ) }$   $\alpha _ { c } ^ { ( k ) } = n _ { c } ^ { ( k ) } / ( n _ { c } ^ { ( k ) } + m _ { 0 } ) .$ </td></tr><tr><td> $\boldsymbol { \Sigma } , \widehat { \boldsymbol { \Sigma } } _ { \mathrm { w } }$ </td><td>Shared-reference strength and local mean weight Common Gaussian covariance and empirical within-class covariance pooled about shared means.</td></tr><tr><td> $\widehat { \pmb { \Sigma } } _ { \ell , c } , \widehat { \pmb { \Sigma } } _ { c } ^ { ( k ) }$ </td><td>Class covariances: pooled raw layer-l features (divisor  $N _ { c } ) ;$  local scaled features (divisor  $n _ { c } ^ { ( k ) } )$ </td></tr><tr><td> $\lambda _ { A } , \mathbf { A }$ </td><td>Trace-scaled ridge  $\bar { \lambda } _ { A } = \gamma \mathrm { t r } ( \widehat { \pmb { \Sigma } } _ { \mathrm { w } } ) / d$  and shared metric  $\mathbf { A } = ( \widehat { \pmb { \Sigma } } _ { \mathrm { w } } + \lambda _ { A } \mathbf { I } _ { d } ) ^ { - 1 } .$ </td></tr><tr><td> $\pi _ { c }$ </td><td>Class prior; FedGRC uses equal priors  $\pi _ { c } = 1 / C .$ </td></tr><tr><td> $J _ { c } ^ { ( k ) } ( \mathbf { m } )$ </td><td>Mean-fitting objective balancing local fit and shared-center regularization under</td></tr><tr><td> $\mathbf { W } ^ { ( k ) } , \dot { \mathbf { b } } ^ { ( k ) }$ </td><td>Client classifier from personalized means and shared  $\mathbf { A } ;$   $\bar { c } : ( \mathbf { w } _ { c } ^ { ( k ) } ) ^ { \top }$  ; bias c:</td></tr><tr><td>D. Prediction Stability under Personalization</td><td>row</td></tr><tr><td colspan="2"></td></tr><tr><td> $s _ { c } ^ { 0 } , s _ { c } ^ { ( k ) } ; \ f ^ { 0 } , f ^ { ( k ) }$ </td><td>Shared and personalized class scores and corresponding label predictors.</td></tr><tr><td> $\epsilon _ { c } ^ { ( k ) } ( { \bf z } )$ </td><td>Bound on absolute score changes from mean adaptation; distinct from safe-norm constant ε.</td></tr><tr><td> $M ^ { 0 } \dot { ( x ) } , B ^ { ( k ) } ( x )$ </td><td>Shared winning margin and perturbation budget (winner&#x27;s score bound plus largest competing  $c ^ { \star } = f ^ { 0 } ( x )$ </td></tr><tr><td></td><td>bound); is the shared prediction.</td></tr><tr><td>E. Theoretical Analysis  $\mathrm { P r } , \mathbb { E } , \mathrm { V a r } ; ~ \mathrm { t r } ( \cdot )$ </td><td>Probability, expectation, variance; matrix trace (sum of diagonal entries).</td></tr><tr><td> $\mathbf { 1 } \{ \cdot \} , \parallel \cdot \parallel _ { 2 }$ </td><td>Indicator (1 if true, 0 otherwise); Euclidean norm for vectors or spectral norm for matrices.</td></tr><tr><td> $\mathbf { M } \succ 0 , \ \| \mathbf { v } \| _ { \mathbf { M } }$ </td><td>Positive-definite matrix and induced norm:  $\| \mathbf { v } \| _ { \mathbf { M } } ^ { 2 } = \mathbf { v } ^ { \top } \mathbf { M } \mathbf { v } .$ </td></tr><tr><td>F. Experimental Setup and Evaluation</td><td></td></tr><tr><td> $\mathcal { T } _ { \mathrm { l o c a l } } ^ { ( k ) } , \mathcal { T } _ { \mathrm { g l o b a l } }$   $\mathcal { U } ^ { ( k ) } , \mathcal { R } ^ { ( k ) }$ </td><td>Client  $k ^ { \prime } \mathrm { s }$  local test set and shared test set covering all federation-known classes. Locally absent classes and up to two observed classes with the fewest local training samples; ties</td></tr><tr><td></td><td>are broken by class ID.</td></tr><tr><td> $a _ { c } ^ { ( k ) } , t _ { c }$ </td><td>Client  $k ^ { \prime } \mathrm { s }$  correct class-c prediction count and class-c sample count in the shared test set.</td></tr><tr><td> $m _ { L } ^ { ( k ) } , m _ { U } ^ { ( k ) } , m _ { R } ^ { ( k ) }$ </td><td>Client  $k ^ { \prime } \mathrm { s }$  test-sample counts for local, unseen-class, and rare-class evaluation. </td></tr><tr><td> $L ^ { ( \breve { k } ) } , U ^ { ( \breve { k } ) } , R ^ { ( \breve { k } ) }$ </td><td>Client accuracies on the local test set and shared-test subsets for unseen and rare classes.</td></tr><tr><td> $F ^ { ( k ) }$ </td><td>Client minimum min  $\{ L ^ { ( k ) } , U ^ { ( k ) } , R ^ { ( k ) } \}$  , omitting  $U ^ { ( k ) }$  when  $\mathcal { U } ^ { ( k ) }$  is empty.</td></tr><tr><td> $\mathrm { T a i l } _ { 2 5 } , q , q _ { v } , m _ { v }$ </td><td>Mean of the lowest  $\hat { q _ { v } } = \operatorname* { m i n } ( q , m _ { v } )$  client scores;  $q = \lceil K / 4 \rceil ; m _ { v } \colon$  eligible-client count.</td></tr><tr><td>local, global</td><td>Client weights:  $m _ { L } ^ { ( k ) }$  for local accuracy; equal class weights within clients, then equal client weights for global accuracy.</td></tr><tr><td>unseen, rare</td><td>Subset accuracies with client weights  $m _ { I J } ^ { ( k ) }$  and  $m _ { R } ^ { ( k ) } ;$  unseen excludes clients with  ${ \mathcal { U } } ^ { ( k ) } = \emptyset$ </td></tr><tr><td>L@25, U@25, R@25; F@25</td><td> $\mathrm { T a i l _ { 2 5 } }$  applied separately to  ${ \cal L } ^ { ( k ) } , \breve { U } ^ { ( k ) } , \breve { R } ^ { ( k ) }$  , and  ${ \tilde { F } } ^ { ( k ) } ;$  F@25 takes client minima before tail</td></tr></table>

## REFERENCES

[1] A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, Toronto, ON, Canada, Tech. Rep., 2009.

[2] T.-M. H. Hsu, H. Qi, and M. Brown, “Measuring the effects of non-identical data distribution for federated visual classification,” arXiv preprint arXiv:1909.06335, 2019.

[3] H. Zhao, P. Peng, C. Chen et al., “FedRS-Bench: Realistic federated learning datasets and benchmarks in remote sensing,” in Proc. ACM SIGKDD Int. Joint Workshop Federated Learning for Data Mining and Graph Analytics (FedKDD), Toronto, ON, Canada, 2025, arXiv:2505.08325.

[4] Z. Zhou, S. Li, W. Wu et al., “NaSC-TG2: Natural scene classification with Tiangong-2 remotely sensed imagery,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 14, pp. 3228–3242, 2021.

[5] G.-S. Xia, W. Yang, J. Delon et al., “Structural high-resolution satellite image indexing,” in Proc. ISPRS TC VII Symp., vol. XXXVIII-7A, Vienna, Austria, 2010, pp. 298–303.

[6] D. Dai and W. Yang, “Satellite image classification via two-layer sparse coding with biased image representation,” IEEE Geosci. Remote Sens. Lett., vol. 8, no. 1, pp. 173–176, 2011.

[7] P. Helber, B. Bischke, A. Dengel et al., “EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 12, no. 7, pp. 2217–2226, 2019.

[8] G.-S. Xia, J. Hu, F. Hu et al., “AID: A benchmark data set for performance evaluation of aerial scene classification,” IEEE Trans. Geosci. Remote Sens., vol. 55, no. 7, pp. 3965–3981, 2017.

[9] G. Cheng, J. Han, and X. Lu, “Remote sensing image scene classification: Benchmark and state of the art,” Proc. IEEE, vol. 105, no. 10, pp. 1865–1883, 2017.

[10] Y. Yang and S. Newsam, “Bag-of-visual-words and spatial extensions for land-use classification,” in Proc. ACM SIGSPATIAL Int. Conf. Adv. Geographic Inf. Syst., 2010, pp. 270–279.

[11] Q. Wang, S. Liu, J. Chanussot et al., “Scene classification with recurrent attention of VHR remote sensing images,” IEEE Trans. Geosci. Remote Sens., vol. 57, no. 2, pp. 1155–1167, 2019.

[12] Y. Long, Y. Gong, Z. Xiao et al., “Accurate object localization in remote sensing images based on convolutional neural networks,” IEEE Trans. Geosci. Remote Sens., vol. 55, no. 5, pp. 2486–2498, 2017.

[13] Z. Xiao, Y. Long, D. Li et al., “High-resolution remote sensing image retrieval based on CNNs from a dimensional perspective,” Remote Sens., vol. 9, no. 7, 2017, art. no. 725.

[14] H. B. McMahan, E. Moore, D. Ramage et al., “Communicationefficient learning of deep networks from decentralized data,” in Proc. Int. Conf. Artif. Intell. Statist. (AISTATS), ser. PMLR, vol. 54, 2017, pp. 1273–1282.

[15] L. Collins, H. Hassani, A. Mokhtari et al., “Exploiting shared representations for personalized federated learning,” in Proc. Int. Conf. Mach. Learn. (ICML), ser. PMLR, vol. 139, 2021, pp. 2089– 2099.

[16] H.-Y. Chen and W.-L. Chao, “On bridging generic and personalized federated learning for image classification,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[17] J. Zhang, Z. Li, B. Li et al., “Federated learning with label distribution skew via logits calibration,” in Proc. Int. Conf. Mach. Learn. (ICML), ser. PMLR, vol. 162, 2022, pp. 26 311–26 329.

[18] L. Jiang and T. Lin, “Test-time robust personalization for federated learning,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[19] Z. Li, X. Shang, R. He et al., “No fear of classifier biases: Neural collapse inspired federated learning with synthetic and fixed classifier,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 5296–5306.

[20] T. Zhou, J. Zhang, and D. H. K. Tsang, “FedFA: Federated learning with feature anchors to align features and classifiers for heterogeneous data,” IEEE Trans. Mobile Comput., vol. 23, no. 6, pp. 6731– 6742, 2024.

[21] C. J. McLaughlin and L. Su, “Personalized federated learning via feature distribution adaptation,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 37, 2024, pp. 77 038–77 059.

[22] K. Guo, Y. Ding, J. Liang et al., “Exploring vacant classes in label-skewed federated learning,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 39, no. 16, 2025, pp. 16 960–16 968.

[23] S. Kim, M. Jeong, S. Kim et al., “FedDr+: Stabilizing dot-regression with global feature distillation for federated learning,” Trans. Mach. Learn. Res. (TMLR), 2025.

[24] Y. Zhou and B. Liu, “BTFL: A Bayesian-based test-time generalization method for internal and external data distributions in federated learning,” in Proc. ACM SIGKDD Conf. Knowl. Discov. Data Min. (KDD), V.1, 2025, pp. 2147–2158.

[25] L. Shen, Z. Tang, L. Wu et al., “Hot-pluggable federated learning: Bridging general and personalized FL via dynamic selection,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025, pp. 102 648–102 684.

[26] S. Yan, Z. Li, C. Wu et al., “You are your own best teacher: Achieving centralized-level performance in federated learning under heterogeneous and long-tailed data,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 2750–2759.

[27] H. Yu, G. Zhang, H. Li et al., “FedLog: Personalized federated classification with less communication and more flexibility,” Trans. Mach. Learn. Res. (TMLR), 2026.

[28] P. Hu and J. Ma, “Personalized federated learning via Gaussian generative modeling,” arXiv preprint arXiv:2603.11620, 2026.

[29] M. Luo, F. Chen, D. Hu et al., “No fear of heterogeneity: Classifier calibration for federated learning with non-IID data,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 34, 2021, pp. 5972–5984.

[30] G. Legate, N. Bernier, L. Caccia et al., “Guiding the last layer in federated learning with pre-trained models,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, 2023, pp. 69 832–69 848.

[31] E. Fan\`ı, R. Camoriano, B. Caputo et al., “Accelerating heterogeneous federated learning with closed-form classifiers,” in Proc. Int. Conf. Mach. Learn. (ICML), ser. PMLR, vol. 235, 2024, pp. 13 029– 13 048.

[32] Z. Guan, Y. Zhou, and X. Gu, “Capture global feature statistics for one-shot federated learning,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 39, no. 16, 2025, pp. 16 942–16 950.

[33] D. Goswami, S. Magistri, K. Wang et al., “Covariances for free: Exploiting mean distributions for training-free federated learning,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 38, 2025, pp. 65 081–65 115.

[34] F. Turazza, M. Picone, and M. Mamei, “The Gaussian-head OFL family: One-shot federated learning from client global statistics,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026, pp. 8041–8064.