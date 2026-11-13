# Uncertainty-aware quantum machine learning based on Dempster-Shafer theory

Marie Kempkes,<sup>1,</sup> <sup>2,</sup> <sup>∗</sup> Jakob Spiegelberg,<sup>2</sup> Evert van Nieuwenburg,<sup>1</sup> and Vedran Dunjko<sup>1</sup>

<sup>1</sup>Leiden University, Niels Bohrweg 1, 2333 CA Leiden, Netherlands

<sup>2</sup>Volkswagen Group Innovation, Berliner Ring 2, 38440 Wolfsburg, Germany

A large body of work in quantum machine learning has centered on characterizing the expressivity and performance of quantum models. In contrast, little attention has been paid to the reliability of predictions beyond average performance measures such as accuracy. In this work, we introduce an uncertainty-aware quantum classification framework based on Dempster-Shafer theory. This theory allows to express uncertainty through weights assigned not only to individual classification labels but to sets of labels. We introduce a training strategy that enables parameterized quantum circuits to learn such assignments directly from data. This provides an efficient way to prepare the corresponding quantum states and enables the application of different post-processing strategies. In particular, it offers a practical setting for quantum Dempster-Shafer fusion procedures with polynomial resource requirements, even when the corresponding mass functions are defined over exponentially many subsets. Overall, our work represents a step toward enabling quantum models to express uncertainty about their predictions, thereby extending the evaluation of quantum models beyond predictive performance alone.

## I. INTRODUCTION

Quantum machine learning (QML) has emerged as a promising paradigm that leverages quantum mechanical principles to learn from data [1–6]. Despite well-known challenges, including issues of trainability such as barren plateaus [7] and the classical simulability of certain quantum models [8], our understanding of quantum machine learning has improved significantly in recent years [9–11]. A growing body of work has identified learning tasks for which QML offers provable advantages over classical approaches [12–18].

However, as in classical machine learning, performance gains alone are insufficient to enable the deployment of quantum machine learning models in practice, especially, e.g., in high-risk applications. Standard evaluation metrics such as accuracy capture average performance, but do not quantify the uncertainty associated with individual predictions. For example, in autonomous driving, a classifier that achieves high accuracy on average may still fail to recognize rare but critical situations, such as unusual obstacles or ambiguous sensor inputs.

This limitation highlights the need for dedicated uncertainty quantification (UQ) methods. UQ aims to estimate a model’s uncertainty, i.e., what the model does not know for a given input. A common distinction is between epistemic uncertainty, which arises from incomplete knowledge of the underlying data-generating process and may be reduced through additional data or improved models, and aleatoric uncertainty, which reflects ambiguity inherent in the observed data and may persist even for an optimal predictor [19]. Access to such uncertainty estimates enables downstream safety mechanisms, e.g., abstention from high-risk decisions if the prediction for a given input is considered highly uncertain.

In classical machine learning, a variety of approaches to uncertainty quantification have been developed. Prominent examples include Bayesian neural networks [20], ensemblebased approaches [21], and Monte Carlo dropout [22]. Methods based on Dempster-Shafer theory (DST), a generalized probability framework, offer a principled way to represent uncertainty through assignments to sets of outcomes [23–25]. Interestingly, DST aligns well with the structure of quantum models, making it a strong candidate for uncertainty-aware QML [26].

![](images/9e810c40a8f93c3d9a9e92c6b6c1764e0814e12d8c933cb6e465832b259983f7.jpg)  
Figure 1. Conceptual overview of the proposed method. A parameterized quantum circuit is trained using the loss in Eq. (7). The resulting basis state probabilities encode masses from Dempster–Shafer theory (see Section III A), which are subsequently used in different post-processing steps (Section III B).

In this work, we introduce a framework that integrates Dempster-Shafer-based uncertainty quantification techniques into parameterized quantum circuits for classification tasks. A conceptual overview is shown in Fig. I. We propose a classification model in which a parameterized quantum circuit is trained to produce generalized probability assignments, referred to as masses, by encoding them into probabilities of basis states. We employ a training strategy from classical machine learning [25], adapt the corresponding loss function to parameterized quantum circuits and show that its evaluation is efficient in the number of qubits. Additionally, we show that the resulting mass function, whose dimension grows exponentially with the number of qubits, is sparse in an idealized zero-loss setting and investigate numerically whether this behavior persists under more realistic training conditions. We further propose different post-processing procedures for efficient extraction of information about the uncertainty associated with individual predictions. We evaluate the resulting uncertainty measures numerically on the downstream task of out-of-distribution detection using real-world datasets. Finally, we study settings in which multiple classifiers provide information about the same input. Dempster-Shafer theory provides dedicated rules for combining the resulting mass assignments, a procedure referred to as mass fusion. Recent work has shown that DST fusion operations can be implemented directly on quantum-encoded mass functions using polynomial quantum resources [27], avoiding explicit manipulation of exponentially large mass vectors. Our framework complements these results by providing a training procedure through which such quantum representations arise naturally from the learning process.

## II. PRELIMINARIES

In this section, we review the foundations of Dempster-Shafer theory and its application to classification in classical machine learning. Related work is discussed in Appendix A.

## A. Dempster-Shafer theory

Dempster-Shafer theory (DST), also known as evidence theory, is a mathematical framework for reasoning under incomplete, imprecise, or contradictory information. Instead of assigning probabilities to individual outcomes, DST assigns generalized probability assignments, so-called masses, to all subsets of the set of possible outcomes [28]. This formulation allows uncertainty to be represented at the level of subsets and supports distinctions between uncertainty arising from lack of information and uncertainty arising from randomness, which are commonly associated with epistemic and aleatoric uncertainty, respectively.

To illustrate this, consider tossing a coin that is known to be fair: both heads and tails have a probability of a half. If nothing is known about the fairness of the coin, DST represents this state of ignorance by assigning a mass of 1 to the set {heads, tails} and a mass of 0 to all other subsets. In this way, it refrains from making unjustified assumptions about the fairness of the coin.

We now introduce the basic objects of Dempster–Shafer theory. Let $\mathcal { V } = \{ { 1 , . . . , n } \}$ denote a frame of discernment, $\mathrm { i . e . }$ , the set of mutually exclusive outcomes of the system. A mass function m assigns masses to all the subsets of $\mathcal { V } ,$ m $: 2 ^ { y } \overset { } {  } [ 0 , 1 ]$ , such that $\begin{array} { r } { m ( \emptyset ) = 0 \mathrm { a n d } \sum _ { S \subseteq \mathcal { V } } m ( S ) = 1 } \end{array}$ The value $m ( S )$ represents the mass committed exactly to the set $S ,$ without attributing mass to any of its subsets. The cardinality of the subset receiving mass reflects the specificity of the available information. Mass assigned to singleton sets supports individual outcomes, whereas mass assigned to larger subsets indicates uncertainty regarding which element of the subset is correct. Consequently, a shift of mass toward larger subsets corresponds to a less specific representation of the available information. In this sense, mass assigned to larger subsets is often associated with epistemic uncertainty arising from a lack of information, while uncertainty distributed across several singleton sets is typically interpreted as reflecting aleatoric uncertainty. In particular, assigning mass to the full set $\mathcal { V }$ represents complete ignorance.

From the mass function, belief and plausibility functions are derived as [29]

$$
b ( S ) = \sum _ { C \subseteq S } m ( C ) ,\tag{1}
$$

$$
p ( S ) = \sum _ { C \cap S \neq \emptyset } m ( C ) , \qquad \forall S \subseteq \mathcal { Y } .\tag{2}
$$

These functions satisfy $b ( S ) ~ \leq ~ p ( S )$ . A central operation in Dempster-Shafer theory is massfusion, which provides a mechanism for combining multiple independent sources of evidence. Such situations arise naturally when information about the same input is available from multiple sources, through classifiers operating on complementary features or measurements collected by different sensors. For example, in autonomous driving one may want to combine information obtained from different camera and radar data. In these settings, each source is associated with a mass function, and Dempster-Shafer theory provides dedicated rules for aggregating them.

Two common fusion rules are the conjunctive and disjunctive combination rules, which correspond to different assumptions about the reliability and compatibility of the evidence sources [30]. Given $P$ independent mass functions $\{ m _ { p } \} _ { p = 1 } ^ { P }$ defined on the same frame of discernment $\mathcal { V } ,$ , the conjunctive rule is defined as

$$
m _ { \cap } ( S ) = \sum _ { C _ { 1 } \cap \cdots \cap C _ { P } = S } \prod _ { p = 1 } ^ { P } m _ { p } ( C _ { p } ) \qquad \forall S \subseteq \mathcal { V } ,\tag{3}
$$

where the summation runs over subsets $C _ { 1 } , \ldots , C _ { P } \subseteq { \mathcal { V } }$ Here, incompatibilities between evidence sources manifest as mass assigned to the empty set, $m _ { \cap } ( \emptyset )$ , which quantifies conflict between different sources.

The disjunctive rule represents a more cautious fusion strategy, appropriate when at least one evidence source is assumed to be reliable but their mutual consistency is uncertain. It is defined as

$$
m _ { \cup } ( S ) = \sum _ { C _ { 1 } \cup \cdots \cup C _ { P } = S } \prod _ { p = 1 } ^ { P } m _ { p } ( C _ { p } ) \qquad \forall S \subseteq \mathcal { V } .\tag{4}
$$

By construction, this rule avoids explicit conflict and shifts mass toward larger subsets.

## B. Dempster-Shafer classifier

Dempster-Shafer theory has recently been applied in classical machine learning through multi-class classification models that output Dempster-Shafer mass functions over the power set of labels [25]. Unlike conventional probabilistic classifiers, these models can assign support not only to individual labels but also to sets of labels. In Ref. [25], this is achieved by learning a full Dempster-Shafer mass function over $2 ^ { y }$

While this formulation enables a rich representation of uncertainty, it incurs exponential scaling in the number of classes. To address this challenge, an efficient alternative classifier is proposed which avoids the explicit representation of all $2 ^ { n }$ masses. Instead, it operates on lower-dimensional quantities derived from the mass function. Specifically, the model outputs the singleton plausibilities, whose number scales linearly with the number of classes. Although the full mass function is not explicitly represented, this approach still supports classification and uncertainty quantification through suitable post-processing procedures.

Let ${ \hat { p } } , p \in [ 0 , 1 ] ^ { n }$ denote the vectors of predicted and target singleton plausibilities derived from the ground-truth mass function, with components

$$
\hat { p } _ { k } : = \hat { p } ( \{ k \} ) , \qquad p _ { k } : = p ( \{ k \} ) , \qquad k \in [ n ] .\tag{5}
$$

The ground-truth mass function m encodes the information provided by the class labels. For a training example with true label $y \in \mathcal { V }$ , it assigns unit mass to the corresponding singleton and zero mass to all other subsets:

$$
m ( \{ y \} ) = 1 , \qquad m ( S ) = 0 ,\tag{6}
$$

for all $S \subseteq { \mathcal { V } } , \ S \neq \{ y \}$ . The associated target singleton plausibilities therefore satisfy $p _ { k } = \delta _ { k , y }$

The loss function below encourages the predicted singleton plausibilities to approximate this target while still permitting uncertainty over multiple labels

$$
\mathcal { L } ( \hat { p } , p ) = \lambda \| \hat { p } - p \| _ { 2 } ^ { 2 } - \left( 1 - \lambda \right) p \cdot \log ( \hat { p } ) ,\tag{7}
$$

where log(ˆp) is applied component-wise.

The squared-error term encourages the predicted singleton plausibilities to match the target plausibilities and, in particular, penalizes plausibility assigned to incorrect classes. In contrast, the cross-entropy term depends only on the plausibility assigned to the correct class. Since singleton plausibilities are not required to sum to one, the cross-entropy term alone does not penalize assigning high plausibility to multiple labels simultaneously. Consequently, the trade-off parameter λ controls the extent to which the model is encouraged to produce specific predictions rather than uncertainty distributed across multiple labels. The effect of λ is analyzed in more detail in Section III C, with an illustrative example shown in Fig. 2.

## III. RESULTS

## A. Dempster-Shafer classifier with parameterized quantum circuits

Consider a multi-class classification task with label set $\mathcal { V } =$ $\{ 1 , \ldots , n \}$ . Given a training set $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ consisting of inputs $x _ { i } \in \mathbb { R } ^ { d }$ and outputs $y _ { i } ~ \in ~ \mathcal { V }$ drawn i.i.d. from some fixed but unknown distribution, the goal is to learn a hypothesis which maps inputs to outputs.

To realize such a hypothesis, we employ an n-qubit parameterized quantum circuit (PQC), where each class is associated with one qubit. For input x and trainable parameters $\vartheta \in \mathbb { R } ^ { p }$ the PQC prepares the quantum state

$$
\rho ( x , \vartheta ) = U ( x , \vartheta ) \rho _ { 0 } U ^ { \dag } ( x , \vartheta ) ,\tag{8}
$$

where $\rho _ { 0 }$ denotes the initial state and $U ( x , \vartheta )$ the unitary implemented by the circuit. In the following, we drop the dependence of $\rho$ on x and ϑ for simplicity. For each input x and parameters $\vartheta ,$ the basis state probabilities of $\rho$ define a Dempster-Shafer mass function $\begin{array} { r } { \dot { m _ { \rho } } : 2 ^ { y }  [ 0 , 1 ] . } \end{array}$ . Specifically, let $i = ( i _ { 1 } , \dots , i _ { n } ) \in \{ 0 , \overset { . } { 1 } \} ^ { n }$ denote the binary representation of basis state |i⟩. We associate to |i⟩ the subset $S _ { i } = \{ k \in \mathcal { V } \mid i _ { k } = 1 \}$ . The mass assigned to $S _ { i }$ is then given by

$$
m _ { \rho } ( S _ { i } ) = \mathrm { T r } \{ \rho ( x , \vartheta ) | i \rangle \langle i | \} .\tag{9}
$$

Remark The encoding in Eq. (9) does not enforce the constraint $m _ { \rho } ( \emptyset ) = 0$ . Empirically, however, we observe that the learned empty-set mass is small and decreases with the number of qubits (Appendix C, Fig. A1).

We identify subsets $S \subseteq \mathcal { V }$ with their corresponding bit strings $i \in \{ 0 , 1 \} ^ { n }$ whenever convenient. Further, for notational simplicity, we write m instead of $m _ { \rho }$ whenever the underlying state is clear from context.

In the following proposition, we describe how two other relevant quantities from DST, namely singleton beliefs and plausibilities, are defined in this setting.

Proposition 1 (Singleton belief and plausibility). Let $\rho$ be an n-qubit quantum state as in $E q . ~ ( 8 )$ and let $m _ { \rho }$ be the mass function induced by Eq. (9). For each $k \in { \dot { \mathcal { V } } } ,$ , define the singleton belief $b _ { k } : = b ( \{ k \} )$ and singleton plausibility $p _ { k } : = p ( \{ k \} )$ . Then, these quantities are given by

$$
b _ { k } = \mathrm { T r } \{ \rho \left| k \right. \left. k \right| \} ,\tag{10}
$$

$$
p _ { k } = \mathrm { T r } \{ \rho _ { k } ( | 1 \rangle \langle 1 | ) _ { k } \} ,\tag{11}
$$

where $\rho _ { k }$ denotes the reduced density matrix of qubit k, $( | 1 \rangle \langle 1 | ) _ { k }$ is the projector onto the state |1⟩ acting on qubit k, and |k⟩ denotes the computational basis state with Hamming weight one corresponding to class k.

The proof of Proposition 1 can be found in Appendix B. We define the k-th output of the PQC-based hypothesis function as $[ f _ { \vartheta } ( x ) ] _ { k } = \bar { \mathrm { T r } } \{ \rho _ { k } ( | 1 \rangle \langle 1 | ) _ { k } \}$ , where $k \in \ \lceil n \rceil$ The predicted class is then given by arg ma $\mathfrak { c } _ { k \in [ n ] } [ f _ { \vartheta } ( x ) ] _ { k } .$

Given the loss function in Eq. (7), our goal in multi-class classification is to minimize the expected loss over the problem distribution. In practice, the problem distribution is unknown and we explicitly optimize the empirical risk, which is defined as the average loss over the set D.

By Proposition 1, the quantities in the loss function can be expressed as expectation values of single-qubit projectors. Therefore, these quantities can be estimated directly from repeated local measurements on the quantum state prepared by the parameterized circuit. Standard concentration bounds, such as Hoeffding’s inequality, imply that all singleton plausibilities can be efficiently estimated on a quantum circuit. In particular, achieving accuracy ϵ and confidence $1 - \delta$ requires $\begin{array} { r } { \mathcal { O } \big ( \frac { 1 } { \epsilon ^ { 2 } } \log \frac { 2 n } { \delta } \big ) } \end{array}$ samples.

Remark For $\lambda = 1$ , the loss in Eq. (7) reduces to a meansquared error on these scores, making this readout a natural choice for multi-class training with parameterized quantum circuits.

After training, one may want to approximate the full mass vector. In general, explicitly sampling the complete $2 ^ { n } \cdot$ dimensional mass vector is computationally inefficient when the mass function is dense. We therefore analyze when the mass functions exhibit sparsity, as this would allow for efficient approximation from measurement data.

In particular, we consider an idealized setting in which the singleton plausibilities are binary. Then, the corresponding mass vector is uniquely determined, as stated in the following proposition.

Proposition 2 (Unique mass for binary singleton plausibility). Consider the n-dimensional vector of singleton plausibilities $( p _ { k } ) _ { k = 1 } ^ { n } . \ : I f p _ { k } \in \{ 0 , 1 \}$ } for all k, then the corresponding mass function m over $2 ^ { \overset { \cdot } { y } }$ is unique and given by

$$
m ( S ) = \left\{ { 1 , } \quad i f S = S ^ { * } , \right.\tag{12}
$$

where $S ^ { \ast } = \{ k \in \mathcal { V } | p _ { k } = 1 \}$

The proof is deferred to Appendix B. Proposition 2 formalizes the observation that binary singleton plausibilities uniquely determine a 1-sparse mass function concentrated on a single set $S ^ { * }$ . Consequently, in the idealized limit of zero loss of the singleton plausibility objective (Eq. (7)) with binary targets, the learned mass functions are expected to approach this sparse regime.

For non-binary singleton plausibilities, however, the corresponding mass function is generally not unique. In this regime, the learned mass allocation is shaped jointly by the data through the singleton plausibility objective and by the inductive bias of the chosen PQC ansatz and optimization procedure. Proposition 2 indicates that this ambiguity progressively diminishes as the predicted plausibilities approach their binary targets.

In practical machine learning scenarios, however, the loss will not be exactly zero and hence predicted singleton plausibilities are generally not binary. We therefore investigate empirically how the sparsity of the learned mass functions depends on the quality of the learned plausibility estimates in Section III C, as illustrated in Fig. 3.

## B. Post-processing of model outputs

Once the model has been trained, the learned mass function can be used in different post-processing procedures. Dempster-Shafer theory provides several ways to quantify uncertainty, which require access to different quantities derived from the mass function in Eq. (9) and therefore incur different computational costs. In this section, we introduce several post-processing procedures that can be implemented efficiently within the proposed framework. All quantities considered in the following can be estimated from measurement outcomes of the PQC. Using standard sampling, an additive precision ϵ can be achieved with sample complexity $\mathcal { O } ( \epsilon ^ { - 2 } )$ . Alternatively, quantum amplitude estimation yields a quadratic improvement to $\mathcal { O } ( \epsilon ^ { - 1 } )$ at the expense of increased circuit depth.

a. Summed singleton plausibilities. We introduce a simple uncertainty measure obtained by summing the singleton plausibilities associated with the n output classes,

$$
\sigma _ { p } = \sum _ { k = 1 } ^ { n } p _ { k } .\tag{13}
$$

The intuition is that larger values of $\sigma _ { p }$ indicate that substantial plausibility is assigned to multiple labels simultaneously, reflecting higher overall uncertainty. Since $\sigma _ { p }$ depends only on the singleton plausibilities, it can be estimated efficiently from the quantities in Proposition 1 without reconstructing the full mass function.

b. Volume of singleton credal set. Another measure of uncertainty is the volume of the so-called singleton credal set denoted $\mathcal { C } ( m )$ . This set consists of all probability distributions $q \in \dot { \Delta } ^ { n - 1 }$ compatible with the mass function m, where $\Delta ^ { n - 1 }$ denotes the $( n - 1 )$ -dimensional probability simplex. Formally,

$$
\mathcal { C } ( m ) = \left\{ q \in \Delta ^ { n - 1 } \ \middle | \ b _ { k } \leq q _ { k } \leq p _ { k } , \forall k \in \mathcal { V } \right\} ,\tag{14}
$$

where $b _ { k } = b ( \{ k \} )$ and $p _ { k } = p ( \{ k \} )$ ) denote the singleton belief and plausibility values, respectively, and $\mathcal { Y } = \{ 1 , \ldots , n \}$ Note that the singleton credal set is a superset of the full credal set obtained by considering belief and plausibility constraints for all subsets. In our setting, singleton beliefs and plausibilities can be estimated efficiently from measurement data, thereby fully specifying the singleton credal set.

c. Averaged class-wise $( p - b ) \ – g a p .$ As an alternative uncertainty measure, we introduce the averaged class-wise gap between singleton plausibility and belief, as

$$
g _ { p - b } = { \frac { 1 } { n } } \sum _ { k = 1 } ^ { n } \bigl ( p _ { k } - b _ { k } \bigr ) .\tag{15}
$$

For each class, this quantity measures the width of the probability interval induced by the mass function and averages it across all classes. Larger values therefore indicate greater uncertainty. Since $g _ { p - b }$ depends only on singleton beliefs and plausibilities, it can be estimated efficiently from quantum measurements.

(a)

![](images/66ea1811fa252d23abbefb6b517dc7336faa5ce9a75876d4a5a4a3630819253d.jpg)

![](images/6dc5b6d67fcd2f1bef131d8c45af59af2bf36c4f200a45d4083c158d45a79f59.jpg)  
Figure 2. Illustration of the influence of the loss function parameter λ. (a) Exemplary mass vector for a representative MNIST image of the digit 3 for different values of λ. Smaller values of λ increase the mass assigned to larger subsets, resulting in more imprecise hypotheses. (b) Average mass assigned to subsets of different cardinalities |S| as a function of λ, quantitatively confirming the trend observed in panel (a).

d. Mass vector fusion. Dempster-Shafer theory provides a principled framework for combining evidence from multiple sources. In a machine learning setting, such sources may correspond to independently trained classifiers, models with different architectures, or models trained on different datasets. We therefore consider $P \in \mathbb { N }$ independently trained parameterized quantum circuits, each producing a mass function $m _ { \rho _ { p } }$ for $p \in \mathsf { \Gamma } [ P ]$ for a given input. The resulting mass functions can then be combined using the fusion rules introduced in Section II A.

Ref. [27] shows that DST fusion operations can be performed directly on quantum-encoded mass functions using polynomial quantum resources. This contrasts with naive fusion procedures based on explicitly represented mass vectors, whose dimension scales as 2<sup>n</sup> with the number of classes. Consequently, the quantum fusion procedure may offer benefits in settings where such explicit representations are manipulated.<sup>1</sup>

A key assumption of Ref. [27] is the availability of quantum states encoding the mass functions to be fused. Preparing such states is not efficient in general and may itself become a computational bottleneck. In our setting, this assumption is naturally satisfied: the mass functions arise directly as measurement statistics of trained PQCs. Moreover, quantities of interest after fusion, such as singleton beliefs and plausibilities, can be estimated directly from the fused quantum state without reconstructing the fused, potentially dense, mass vector. Our framework therefore provides a practical setting in which quantum DST fusion procedures of Ref. [27] can be applied to learned mass functions.

## C. Numerical results

In this section, we present numerical experiments that illustrate and complement the theoretical results of the previous section. We first investigate how the loss function parameter λ controls the uncertainty behavior of the proposed

Dempster-Shafer quantum classifier on an exemplary classification problem. We then empirically examine to what extent the sparsity of the learned mass function, shown in Proposition 2 under idealized assumptions, persists in practical settings. Finally, we evaluate different uncertainty measures from Section III B on a realistic image-classification task and demonstrate how Dempster-Shafer mass fusion can be used to aggregate evidence from multiple classifiers to improve predictive performance. Further details on the numerical experiments are provided in Appendix C.

a. Influence of the loss function parameter λ. To illustrate the influence of the loss function parameter λ, we train a quantum classifier on four MNIST digits for varying values of λ. Figure 2 (a) shows an exemplary mass function for an image of the digit 3. As λ decreases, mass is shifted from the correct singleton {3} toward larger subsets containing the true label, resulting in increasingly uncertain predictions. At the same time, subsets that do not contain the true label receive negligible mass. This indicates that the classifier has correctly identified implausible labels while expressing increasing uncertainty about the true class. Figure 2 (b) quantifies this behavior by showing the average mass assigned to subsets of different cardinalities |S| as a function of λ. The mass assigned to larger subsets decreases as λ increases, confirming the trend observed in panel (a). We therefore interpret λ as a hyperparameter that controls the caution of the resulting predictions. Smaller values lead to more imprecise but uncertainty-aware predictions, while larger values favor concentrated mass assignments. The parameter λ can easily be optimized through standard hyperparameter search procedures.

b. Empirical sparsity oflearned massfunctions. Proposition 2 establishes that binary singleton plausibilities uniquely determine a sparse mass function. Since the loss in Eq. (7) encourages singleton plausibilities to approach binary target values, it is natural to ask whether the resulting mass function is also sparse if the loss is not exactly zero. We therefore investigate empirically how the sparsity of the learned mass functions depends on the deviation of the singleton plausibilities from their binary targets.

For this purpose, we train PQCs with varying numbers of qubits n on MNIST classification tasks. Training is terminated once the mean absolute singleton plausibility error falls below a prescribed threshold ε, defined as $\varepsilon \quad =$ $\begin{array} { r } { \frac { 1 } { N n } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { n } | \hat { p } _ { k } - p _ { k } | } \end{array}$ , where N is the number of train samples. Hence, ε measures the average deviation from the ideal binary prediction for which Proposition 2 guarantees a sparse mass function.

(a)  
![](images/ee9c65f15787251b7c9c81b3fc1eefd6e23676c4e4043a864839c92c594b104d.jpg)  
(c)

![](images/3f2fcfe786a1af6df436404fc18b3b6dab04ddbebdd81fdb871e29bbaef31ce0.jpg)

![](images/3c9bf0d2e8a3aac21798690beb640e5c00c4095a017beacb24c0eb5fda20cd77.jpg)

(d)  
![](images/1823d6e19da53bcdaf11a7a98ff62dd1722e04b98f153591c778d65be783b527.jpg)  
Figure 3. Empirical sparsity of the learned mass vector beyond the ideal setting of Proposition 2. The quantity $k _ { \tau }$ denotes the minimum number of subsets required to account for a fraction τ of the total mass. Panels (a) and (c) show $k _ { \tau }$ as a function of the error threshold ε for different numbers of qubits, while panels (b) and (d) show the dependence on the number of qubits for fixed values of ε. As the training error increases, the mass functions become less sparse. Moreover, for fixed error thresholds, $k _ { \tau }$ grows rapidly with the number of qubits.

To quantify sparsity, we consider the quantity $k _ { \tau }$ , defined $k _ { \tau } : = \operatorname* { m i n } \Bigl \{ | C | : C \subseteq 2 ^ { \mathcal { V } } , \sum _ { S \in C } m ( S ) \geq \tau \Bigr \}$ , where |C| denotes the number of subsets contained in the collection $C .$ Thus, $k _ { \tau }$ is the minimum number of sets whose combined mass is at least $\tau$ and smaller values of $k _ { \tau }$ indicate sparser mass functions.

Several trends can be observed. From Figure 3 (a) and (c) we see that $k _ { \tau }$ increases with the error threshold ε, indicating that the mass functions become less sparse as the singleton plausibilities deviate further from their binary targets. From panels (b) and (d) one can see that the growth of $k _ { \tau }$ becomes more pronounced as the number of qubits increases. For sufficiently small systems, a large fraction of the mass remains concentrated on only a few subsets. However, for larger values of $n ,$ increasingly many subsets are required to capture the same fraction of the total mass. This effect is particularly visible in Figures 3(b) and (d), where $k _ { \tau }$ grows rapidly with the number of qubits.

These experiments suggest that effective sparsity can be expected only when the classifier is trained to high precision and when a moderate fraction of the total mass suffices for the intended application. Consequently, the exact reconstruction of a complete mass vector may become impractical for larger classification tasks. We therefore turn to the analysis of postprocessing procedures that can be applied without explicitly reconstructing the full mass vector.

c. Out-of-distribution detection with uncertainty metrics. To evaluate the proposed uncertainty measures, we consider an out-of-distribution (o.o.d.) detection task. The underlying assumption is that predictions on o.o.d. samples should have higher uncertainty than predictions on in-distribution (i.d.) samples. Consequently, a useful uncertainty measure should be able to distinguish between i.d. and o.o.d. data. We quantify this ability using the area under the receiver operating characteristic curve (AUROC), where larger values indicate better separation between the two distributions.

In the first experiment, a parameterized quantum circuit with 10 qubits is trained on MNIST using the loss function in Eq. (7). The trained model is then evaluated on MNIST test samples (i.d.) and FashionMNIST [31] samples (o.o.d.). The resulting AUROC values are reported in Table I. The summed singleton plausibility and the average plausibility-belief gap achieve AUROC values of 0.83 and 0.82, respectively, indicating good discrimination between i.d. and o.o.d. samples. In contrast, the singleton credal set volume performs close to random guessing with an AUROC of 0.54. A second experiment is conducted using CIFAR10 as the in-distribution dataset and FashionMNIST as the out-of-distribution dataset, where similar qualitative behavior is observed.

d. Massfusionfrom multiple classifiers. Beyond assessing the uncertainty of individual predictions, Dempster-Shafer theory also provides an approach for evidence aggregation from multiple models. Figure 4 demonstrates how mass fusion can be used to aggregate evidence from multiple quantum classifiers trained on complementary data representations.

<table><tr><td>Uncertainty metric</td><td>MNIST CIFAR10</td><td></td></tr><tr><td>Summed singleton plausibility</td><td>0.83</td><td>0.707</td></tr><tr><td>Average plausibility-belief gap</td><td>0.82</td><td>0.684</td></tr><tr><td>Credal set volume</td><td>0.54</td><td>0.625</td></tr></table>

Table I. AUROC values for out-of-distribution detection on two datasets using different uncertainty metrics derived from the Dempster-Shafer quantum classifier. All metrics are defined in Section III B.

In this experiment, each PQC classifier with 10 qubits is trained on a different patch of the MNIST input images, obtained by partitioning each image into four disjoint quadrants. While each individual model achieves only moderate accuracy when operating on partial information, combining their learned mass functions using Dempster’s disjunctive rule (Eq. (4)) leads to a clear improvement in overall classification performance.

As shown in Fig. 4, the accuracy of the fused classifier, computed on the singleton plausibilities, consistently exceeds that of each individual model as well as their mean performance. This demonstrates how Dempster-Shafer–based fusion can enable aggregation of complementary evidence sources, resulting in potentially more precise predictions than those obtained from any single classifier alone.

In the present experiments, the fusion step is performed classically, which is feasible for small problem sizes considered here. For larger problems, the exponentially growing number of possible subsets may pose computational challenges for classical fusion. Ref. [27] showed that DST fusion can be performed directly on quantum-encoded mass functions using only polynomial quantum resources. This approach assumes access to quantum states encoding the mass functions to be fused. As discussed in Section III B, our framework naturally provides such quantum representations through the learned parameterized quantum circuits. Moreover, our framework and the proposed post-processing procedures enable efficient readout for both classification and uncertainty quantification.

## IV. DISCUSSION

In this work, we address the problem of uncertainty quantification in quantum machine learning. Quantifying predictive uncertainty is an important step towards practical quantum machine learning, particularly in applications where decisions must be accompanied by confidence assessments. To this end, we introduce a classification framework based on Dempster-Shafer theory (DST) for parameterized quantum circuits (PQCs).

The proposed framework establishes a correspondence between quantities from DST and measurement outcomes of PQCs. In particular, so-called mass functions are represented through basis state probabilities of the quantum state. Building on a training strategy from classical machine learning [25], we show that the resulting loss function can be evaluated efficiently from measurement data. We further show analytically that, in the ideal setting of zero loss, the learned quantum state is associated with a sparse mass function. In numerical experiments, we investigate to what extent this sparsity persists under more realistic training conditions. In addition, we introduce several post-processing procedures that can be estimated efficiently from measurement data and provide uncertainty information for individual predictions.

![](images/ea0773cd9df50788cc43de34e1f7c8acd188ebbf621903ed910f2632d81772b4.jpg)

![](images/2292bfd3d64dae172b6af5a2d1389875d396b6272166858fb72e1dd71e9b627c.jpg)  
Figure 4. Mass fusion improves classification performance. Four individual classifiers are trained on disjoint patches of MNIST images, obtained by splitting each image into four parts, and evaluated separately. The fused classifier combines learned mass functions using Dempster’s rule as described in Section III B. Bars show test accuracy of each individual classifier and of the fused model; the dashed line indicates the mean accuracy of the individual classifiers.

We illustrate the versatility of the framework through several numerical experiments, in settings such as out-ofdistribution detection and fusion of mass functions from multiple sources. In the case of mass fusion, our framework may provide an option for efficient state preparation and readout for the quantum fusion procedures introduced in Ref. [27]. The learned PQCs directly generate the required quantumencoded mass representations, and relevant post-processing quantities can be estimated from the fused quantum state without reconstructing the full mass vector. This combination provides a direct interface between uncertainty-aware quantum learning models and quantum fusion procedures while avoiding explicit reconstruction of the full mass vector.

Several limitations remain. The numerical experiments are intended as proof-of-concept demonstrations and are restricted to simulated quantum systems. A more extensive empirical evaluation, including comparisons with established uncertainty quantification methods, is left for future work. Moreover, it remains open to what extent similar uncertainty representations and associated post-processing procedures can be realized efficiently in classical or quantuminspired models. While Ref. [25] shows that singleton plausibilities admit efficient classical representations, it is currently unclear whether an analogous result holds for singleton beliefs. A further limitation concerns the identifiability of the learned mass functions. The singleton plausibility objective in Eq. (7) directly constrains only the singleton plausibilities, and non-binary plausibility vectors generally do not uniquely determine the corresponding mass function. Consequently, the learned distribution of mass across subsets of Y is influenced by both the training data and the inductive bias of the

PQC ansatz and optimization procedure. While Proposition 2 formalizes the observation that this ambiguity vanishes in the binary plausibility limit, investigating additional regularization principles or refinement procedures that further constrain the mass function constitutes an interesting direction for future work.

Taken together, our results show that Dempster-Shafer theory provides a natural foundation for uncertainty-aware quantum machine learning and represents a step toward more reliable quantum models, as well as a new perspective for analyzing quantum systems beyond standard performance metrics.

## V. ACKNOWLEDGMENTS

V.D. acknowledges support from the European Union’s Horizon Europe program through the ERC CoG BeMAI-

[1] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, Quantum machine learning, Nature 549, 195 (2017).

[2] V. Dunjko and H. J. Briegel, Machine learning & artificial intelligence in the quantum domain: A review of recent progress, Reports on Progress in Physics 81, 074001 (2018).

[3] G. Carleo, I. Cirac, K. Cranmer, L. Daudet, M. Schuld, N. Tishby, L. Vogt-Maranto, and L. Zdeborova, Machine learn-´ ing and the physical sciences, Reviews of Modern Physics 91, 045002 (2019).

[4] M. Schuld and F. Petruccione, Machine Learning with Quantum Computers, Vol. 676 (Springer, 2021).

[5] M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio, et al., Variational quantum algorithms, Nature Reviews Physics 3, 625 (2021).

[6] K. Bharti, A. Cervera-Lierta, T. H. Kyaw, T. Haug, S. Alperin-Lea, A. Anand, M. Degroote, H. Heimonen, J. S. Kottmann, T. Menke, et al., Noisy intermediate-scale quantum algorithms, Reviews of Modern Physics 94, 015004 (2022).

[7] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, Barren plateaus in quantum neural network training landscapes, Nature communications 9, 4812 (2018).

[8] M. Cerezo, M. Larocca, D. Garc´ıa-Mart´ın, N. L. Diaz, P. Braccia, E. Fontana, M. S. Rudolph, P. Bermejo, A. Ijaz, S. Thanasilp, et al., Does provable absence of barren plateaus imply classical simulability?, Nature Communications 16, 7907 (2025).

[9] E. Anschuetz, A unified theory of quantum neural network loss landscapes, in International Conference on Learning Representations, Vol. 2025 (2025) pp. 97859–97918.

[10] E. Gil-Fuster, J. Eisert, and C. Bravo-Prieto, Understanding quantum machine learning also requires rethinking generalization, Nature Communications 15, 2277 (2024).

[11] E. Gil-Fuster, C. Gyurik, A. Perez-Salinas, and V. Dunjko, On´ the relation between trainability and dequantization of variational quantum learning models, in International Conference on Learning Representations, Vol. 2025 (2025) pp. 24069–24093.

[12] Y. Liu, S. Arunachalam, and K. Temme, A rigorous and robust quantum speed-up in supervised machine learning, Nature Physics 17, 1013 (2021).

Quantum (Grant No. 101124342). V.D. and E.vN acknowledge the Dutch National Growth Fund (NGF) as part of the Quantum Delta NL programme. GPT-5.6 was used to assist with code development, writing and mathematical consistency checks during the preparation of this manuscript. All scientific ideas, analyses, and conclusions are those of the authors, and all AI-assisted code and text were carefully reviewed, manually checked, and validated prior to inclusion.

## VI. DISCLAIMER

The results, opinions, and conclusions expressed in this publication are not necessarily those of Volkswagen Aktiengesellschaft or those of the European Union or the European Research Council. None of the parties involved can be held responsible for them.

[13] R. Sweke, J.-P. Seifert, D. Hangleiter, and J. Eisert, On the quantum versus classical learnability of discrete distributions, Quantum 5, 417 (2021).

[14] H.-Y. Huang, R. Kueng, and J. Preskill, Information-theoretic bounds on quantum advantage in machine learning, Physical Review Letters 126, 190505 (2021).

[15] C. Gyurik and V. Dunjko, Exponential separations between classical and quantum learners, arXiv:2306.16028 (2023).

[16] N. Pirnay, R. Sweke, J. Eisert, and J.-P. Seifert, Superpolynomial quantum-classical separation for density modeling, Physical Review A 107, 042416 (2023).

[17] R. Molteni, C. Gyurik, and V. Dunjko, Exponential quantum advantages in learning quantum observables from classical data, npj Quantum Information (2026).

[18] A. Barthe, M. Y. Rad, M. Grossi, and V. Dunjko, Quantum advantage in learning quantum dynamics via fourier coefficient extraction (2025), arXiv:2506.17089 [quant-ph].

[19] E. Hullermeier and W. Waegeman, Aleatoric and Epistemic Un-¨ certainty in Machine Learning: An Introduction to Concepts and Methods, Machine Learning 110, 457 (2021).

[20] D. J. C. MacKay, Practical bayesian framework for backpropagation networks, Neural Computation 4, 448 (1992).

[21] B. Lakshminarayanan, A. Pritzel, and C. Blundell, Simple and scalable predictive uncertainty estimation using deep ensembles, in Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17 (Curran Associates Inc., Red Hook, NY, USA, 2017) p. 6405–6416.

[22] Y. Gal and Z. Ghahramani, Dropout as a bayesian approximation: Representing model uncertainty in deep learning, in Proceedings ofthe 33rd International Conference on International Conference on Machine Learning - Volume 48 (JMLR.org, 2016) p. 1050–1059.

[23] S. K. Manchingal and F. Cuzzolin, Epistemic deep learning, arXiv preprint arXiv:2206.07609 (2022).

[24] S. K. Manchingal, M. Mubashar, K. Wang, K. Shariatmadar, and F. Cuzzolin, Random-set convolutional neural network (rs-cnn) for epistemic deep learning, arXiv preprint arXiv:2307.05772 (2023).

[25] M. C. Kempkes, V. Dunjko, E. van Nieuwenburg, and J. Spiegelberg, Reliable classifications with guaranteed confidence using the dempster-shafer theory of evidence, in Joint Eu-

ropean Conference on Machine Learning and Knowledge Discovery in Databases (Springer, 2024) pp. 89–105.

[26] Q. Zhou, G. Tian, and Y. Deng, Bf-qc: belief functions on quantum circuits, Expert Systems with Applications 223, 119885 (2023).

[27] H. Luo, Q. Zhou, L. Pan, Z. Li, and Y. Deng, Attribute fusionbased evidential classifier on quantum circuits, Quantum Machine Intelligence 8, 51 (2026).

[28] A. P. Dempster, Upper and lower probabilities induced by a multivalued mapping, The Annals of Mathematical Statistics 38, 325 (1967).

[29] G. Shafer, Perspectives on the theory and practice of belief functions, International Journal of Approximate Reasoning 4, 323 (1990).

[30] T. Denœux, Conjunctive and disjunctive combination of belief functions induced by nondistinct bodies of evidence, Artificial Intelligence 172, 234 (2008).

[31] H. Xiao, K. Rasul, and R. Vollgraf, Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, CoRR abs/1708.07747 (2017), arXiv:1708.07747.

[32] M. Sensoy, L. Kaplan, and M. Kandemir, Evidential deep learning to quantify classification uncertainty, Advances in Neural Information Processing Systems 31 (2018).

[33] W. Bao, Q. Yu, and Y. Kong, Evidential deep learning for open set action recognition, in 2021 IEEE/CVF International Conference on Computer Vision (ICCV) (2021) pp. 13329–13338.

[34] B. Charpentier, D. Zugner, and S. G¨ unnemann, Posterior net-¨ work: Uncertainty estimation without ood samples via densitybased pseudo-counts, Advances in Neural Information Processing Systems 33, 1356 (2020).

[35] V. Bengs, E. Hullermeier, and W. Waegeman, Pitfalls of epis-¨ temic uncertainty quantification through loss minimisation, in Advances in Neural Information Processing Systems, Vol. 35, edited by S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Curran Associates, Inc., 2022) pp. 29205– 29216.

[36] Y. Sale, M. Caprio, and E. Hullermeier, Is the volume of a¨ credal set a good measure for epistemic uncertainty?, in Proceedings ofthe Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, Proceedings of Machine Learning Research, Vol. 216, edited by R. J. Evans and I. Shpitser (PMLR, 2023) pp. 1795–1804.

[37] M. Wendlinger, K. Tscharke, and P. Debus, Old rules in a new game: Mapping uncertainty quantification to quantum machine learning, in 2025 IEEE International Conference on Quantum

Computing and Engineering (QCE), Vol. 01 (2025) pp. 1803– 1814.

[38] S. Park and O. Simeone, Quantum conformal prediction for reliable uncertainty quantification in quantum machine learning, IEEE Transactions on Quantum Engineering 5, 1 (2024).

[39] M. Kempkes, J. Spiegelberg, E. van Nieuwenburg, and V. Dunjko, Detecting underdetermination in parameterized quantum circuits, Machine Learning: Science and Technology 7, 015010 (2026).

[40] P. N. M. U. Hewage, M. Chakkravarthy, and R. K. Abeysekara, Quantum measurement statistics as bayesian uncertainty estimators for physics-constrained learning (2026), arXiv:2604.10896 [quant-ph].

[41] A. Vourdas, Quantum probabilities as dempster-shafer probabilities in the lattice of subspaces, Journal of Mathematical Physics 55 (2014).

[42] H. Luo, Q. Zhou, Z. Li, and Y. Deng, Variational quantum linear solver-based combination rules in dempster–shafer theory, Information Fusion 102, 102070 (2024).

[43] F. Xiao, Y. Zhou, and W. Pedrycz, An adaptive quantum circuit of dempster's rule of combination for uncertain pattern classification, Advances in Neural Information Processing Systems 38, 17848 (2025).

[44] V. Bergholm, J. Izaac, M. Schuld, C. Gogolin, S. Ahmed, V. Ajith, M. S. Alam, G. Alonso-Linaje, B. AkashNarayanan, A. Asadi, J. M. Arrazola, U. Azad, S. Banning, C. Blank, T. R. Bromley, B. A. Cordier, J. Ceroni, A. Delgado, O. D. Matteo, A. Dusko, T. Garg, D. Guala, A. Hayes, R. Hill, A. Ijaz, T. Isacsson, D. Ittah, S. Jahangiri, P. Jain, E. Jiang, A. Khandelwal, K. Kottmann, R. A. Lang, C. Lee, T. Loke, A. Lowe, K. McKiernan, J. J. Meyer, J. A. Montanez-Barrera, R. Moyard, Z. Niu,˜ L. J. O’Riordan, S. Oud, A. Panigrahi, C.-Y. Park, D. Polatajko, N. Quesada, C. Roberts, N. Sa, I. Schoch, B. Shi, S. Shu,´ S. Sim, A. Singh, I. Strandberg, J. Soni, A. Szava, S. Tha-´ bet, R. A. Vargas-Hernandez, T. Vincent, N. Vitucci, M. Weber,´ D. Wierichs, R. Wiersema, M. Willmann, V. Wong, S. Zhang, and N. Killoran, Pennylane: Automatic differentiation of hybrid quantum-classical computations (2022), arXiv:1811.04968 [quant-ph].

[45] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, Gradient-based learning applied to document recognition, Proceedings of the IEEE 86, 2278 (1998).

[46] J. B. Lasserre, Volume of slices and sections of the simplex in closed form, Optimization Letters 9, 1263 (2015).

# Appendix for “Uncertainty-aware quantum machine learning based on Dempster-Shafer theory”

## Appendix A: Related work

a. Uncertainty quantification in classical machine learning. Uncertainty quantification has been extensively studied in classical machine learning, with a wide range of approaches proposed to estimate uncertainty. Bayesian neural networks provide a principled probabilistic framework by placing distributions over model parameters, but typically rely on techniques that are computationally expensive [20]. Ensemble methods approximate predictive uncertainty by analyzing the outputs of multiple independently trained models. While empirically effective, this approach incurs a substantial computational overhead during both training and inference [21]. Monte Carlo dropout offers a lightweight alternative by interpreting dropout as approximate Bayesian inference [22], though the resulting uncertainty estimates can be sensitive to architectural and training choices [21]. Evidential deep learning approaches have gained increasing attention by modeling uncertainty through higher-order distributions over predictive class probabilities [32–34]. These methods learn such higher-order representations, for instance in the form of Dirichlet distributions or evidential mass assignments [23–25]. Recent work has highlighted that uncertainty measures learned through empirical risk minimization in evidential or second-order predictive models should be interpreted with care when relating them to epistemic uncertainty [35]. More recently, Ref. [36] showed that the volume of a credal set is not, in general, a reliable measure of epistemic uncertainty for multiclass problems.

b. Uncertainty in quantum machine learning. In contrast to the classical setting, uncertainty quantification in quantum machine learning has received comparatively limited attention. Several approaches adapt classical uncertainty quantification techniques to the quantum setting. Ref. [37] provides a systematic mapping of a range of classical uncertainty quantification methods to quantum machine learning models and investigates their applicability in practice. Dempster-Shafer-theoretic approaches, which form the basis of our work, are not considered in that analysis. Ref. [38] extends conformal prediction to quantum models by adapting this classical framework to account for the stochasticity of quantum measurements and hardware noise. Along similar lines, Ref. [39] studies model reliability by transferring the classical notion of underdetermination to parameterized quantum circuits, allowing for the identification of inputs for which predictions are inherently ambiguous. Ref. [40] connects quantum measurement statistics to Bayesian uncertainty, showing that repeated circuit evaluations can be interpreted in terms of calibrated posterior uncertainty without requiring explicit probabilistic models.

Connections between Dempster-Shafer theory and quantum mechanics have recently been explored, highlighting conceptual and mathematical similarities between quantum probabilities and mass functions [41]. More recently, quantum-inspired and hybrid approaches have studied how mass function representations can be encoded into quantum states and manipulated using quantum operations [26, 27, 42, 43]. These works enable implementations of Dempster–Shafer operations at the level of quantum states, avoiding the explicit manipulation of exponentially large mass vectors required by standard classical implementations. However, they generally assume access to quantum representations of masses and do not address how they can be efficiently constructed.

## Appendix B: Proof of Proposition 1 and Proposition 2

In this section, we provide the proofs of Proposition 1 and Proposition 2.

Proposition 1 (Singleton belief and plausibility). Let ρ be an n-qubit quantum state as in Eq. (8) and let $m _ { \rho }$ be the massfunction induced by Eq. (9). For each $k \in \mathcal { V } ,$ , define the singleton belief $b _ { k } : = b ( \{ k \} )$ and singleton plausibility $p _ { k } : = p ( \{ k \} )$ . Then, these quantities are given by

$$
b _ { k } = \mathrm { T r } \{ \rho \left| k \right. \left. k \right| \} ,\tag{B1}
$$

$$
p _ { k } = \mathrm { T r } \{ \rho _ { k } ( | 1 \rangle \langle 1 | ) _ { k } \} ,\tag{B2}
$$

where $\rho _ { k }$ denotes the reduced density matrix of qubit $k , ( | 1 \rangle \langle 1 | ) _ { k }$ is the projector onto the state |1⟩ acting on qubit k, and |k denotes the computational basis state with Hamming weight one corresponding to class k.

ProofofProposition 1. We start from the definition of belief in Dempster-Shafer theory. For a singleton set $\{ k \}$ , the belief reduces to the mass assigned exactly to that singleton, which yields

$$
b _ { k } = \sum _ { S \subseteq \{ k \} } m ( S ) = m ( \{ k \} ) = \operatorname { T r } \big ( \rho | k \rangle \langle k | \big ) ,\tag{B3}
$$

establishing Eq. (B1). Similarly, by definition, the plausibility of the singleton {k} is given by the total mass of all subsets that contain k,

$$
p _ { k } = p ( \{ k \} ) = \sum _ { S \cap \{ k \} \neq \emptyset } m ( S ) .\tag{B4}
$$

Using the correspondence between subsets and computational basis states, this sum ranges over all bit strings $i \in \{ 0 , 1 \} ^ { n }$ such that $i _ { k } = 1$ , yielding

$$
p _ { k } = \sum _ { \stackrel { i \in \{ 0 , 1 \} ^ { n } } { i _ { k } = 1 } } \left. i \right| \rho \left| i \right. = \operatorname { T r } \left( \rho _ { k } \left| 1 \right. \left. 1 \right| _ { k } \right) ,\tag{B5}
$$

which proves Eq. (B2).

Proposition 2 (Unique mass for binary singleton plausibility). Consider the n-dimensional vector of singleton plausibilities $( p _ { k } ) _ { k = 1 } ^ { n } . \ I f p _ { k } \in \{ 0 , 1 \}$ for all k, then the corresponding massfunction m over $2 ^ { y }$ is unique and given by

$$
m ( S ) = \left\{ { 1 , } \quad i f S = S ^ { * } , \right.\tag{B6}
$$

where

$$
S ^ { * } = \{ k \in \mathcal { V } | p _ { k } = 1 \} .\tag{B7}
$$

Proof. Let $S ^ { \ast } = \{ k \in \mathcal { V } | p _ { k } = 1 \}$ . Since $p _ { k } \in \{ 0 , 1 \}$ for all $k ,$ , it follows that $p _ { k } = 0$ whenever $k \notin S ^ { * }$

Consider any $k \not \in S ^ { * }$ . Since $\begin{array} { r } { p _ { k } = \sum _ { S \subset \mathcal { V } : k \in S } m ( S ) = 0 } \end{array}$ , non-negativity of the masses implies $m ( S ) = 0$ for every subset S containing k. Hence, every subset S satisfying $\overline { { S \cap { ( y \setminus S ^ { * } ) } } } \neq \emptyset$ has zero mass.

Now consider any $k \in S ^ { * }$ for which we have $\begin{array} { r } { p _ { k } = \sum _ { S \subset \mathcal { V } : k \in S } m ( S ) = 1 } \end{array}$ . Since the masses satisfy $\begin{array} { r } { \sum _ { S \subseteq \mathcal { Y } } m ( S ) = 1 } \end{array}$ and $\begin{array} { r } { \sum _ { S \subseteq y } m ( S ) = \sum _ { S \subseteq y : k \in S } m ( S ) + \sum _ { S \subseteq y : k \notin S } m ( S ) } \end{array}$ , it follows that $\Sigma _ { S \subseteq { \mathcal { V } } : k \not \in S } m ( S ) = 0$ . By non-negativity of masses, we can conclude that $m ( S ) = 0$ for every subset $\dot { S }$ not containing $k .$

Since the above arguments hold for every $k \in \mathcal { V }$ , any subset $S$ with nonzero mass must contain every element of $S ^ { * }$ and no element of $\boldsymbol { \ y } \cdot \boldsymbol { S } ^ { * }$ . Therefore, the only subset that can have nonzero mass is $S ^ { * }$ . By normalization of masses, $m ( S ^ { * } ) = 1$ Consequently,

$$
m ( S ) = \left\{ { 1 , \atop 0 } \right.\tag{B8}
$$

which proves the statement.

## Appendix C: Experimental details and mass of the empty-set

In this section, we provide information on the details of the numerical experiments from Section III C. We further analyze numerically the mass assigned to the empty set and how it scales in the number of qubits.

## 1. Experimental details

All experiments are implemented using the PennyLane framework [44]. In the following, we provide details on the data generation, parameterized quantum circuit (PQC) architecture, and training procedure used in our experiments.

a. Influence of the loss function parameter λ. To analyze the impact of the loss parameter λ, we consider a multi-class classification task on four digits of the MNIST dataset [45]. The input data is first reduced to 4 dimensions using principal component analysis (PCA), fitted on the training data and then escaled to the interval $[ - \pi , \pi ]$ . The model is implemented as a PQC of 4 qubits and 10 layers. Classical input data is encoded into the PQC using the AngleEmbedding routine provided by PennyLane. Each layer k of the circuit consists of parameterized single-qubit rotations followed by entangling operations. Concretely, for each qubit i and layer k, we apply

$$
\mathrm { R X } ( \vartheta _ { k i 1 } ) \mathrm { R Y } ( \vartheta _ { k i 2 } ) \mathrm { R Z } ( \vartheta _ { k i 3 } ) ,\tag{C1}
$$

followed by controlled-NOT (CNOT) gates between neighboring qubits and subsequent re-uploading using the same encoding routine as before.

Training is performed using mini-batch stochastic optimization with batch size 1 over 300 epochs on 5 training samples. We use the Adam optimizer with learning rate 0.01 to minimize the loss defined in Eq. (7).

b. Empirical sparsity oflearned massfunctions. To investigate the sparsity properties of the learned mass functions under a more realistic setting, we consider a multi-class classification task based on the MNIST dataset. The model operates on n qubits, where n is varied from 2 to 7, with each qubit associated to one class. The input data is reduced to n dimensions using principal component analysis (PCA) and subsequently rescaled to the interval $[ - \pi , \pi ]$ to match the range required for quantum feature encoding.

Classical input data is encoded using the AngleEmbedding routine provided by PennyLane. Each layer k of the circuit consists of parameterized single-qubit rotations followed by entangling operations. Concretely, for each qubit i and layer k, we apply

$$
\mathrm { R X } ( \vartheta _ { k i 1 } ) \mathrm { R Y } ( \vartheta _ { k i 2 } ) \mathrm { R Z } ( \vartheta _ { k i 3 } ) ,\tag{C2}
$$

followed by CNOT gates between neighboring qubits and subsequent re-uploading using the same encoding routine as before.   
The PQC consists of 20 such layers.

Training is performed using mini-batch stochastic optimization with batch size 1 on 5 training samples over at most 5000 epochs or until the required error threshold is reached. We use the Adam optimizer with learning rate 0.01 and set the loss hyperparameter to $\lambda = 0 . 9$

c. Out-of-distribution detection with uncertainty metrics. We evaluate the proposed uncertainty measures in an out-ofdistribution $( 0 . 0 . 0 . )$ detection setting using the MNIST dataset with $n = 1 0$ classes. The input data is first reduced to 10 dimensions using principal component analysis (PCA), normalized to the interval $[ - \pi , \pi ]$ , and subsequently encoded into the quantum circuit. The training set consists of $n _ { \mathrm { t r a i n } } = 3 0 0$ samples, while evaluation is performed on $n _ { \mathrm { t e s t } } = 3 0 0$ in-distribution samples and $n _ { \mathrm { o o d } } = 3 0 0$ out-of-distribution samples.

The model is implemented as a parameterized quantum circuit with $n = 1 0$ qubits and 10 layers. Classical input data is encoded using the AngleEmbedding routine provided by PennyLane. Each layer k of the circuit consists of parameterized single-qubit rotations followed by entangling operations. Concretely, for each qubit i and layer k, we apply

$$
\mathrm { R X } ( \vartheta _ { k i 1 } ) \mathrm { R Y } ( \vartheta _ { k i 2 } ) \mathrm { R Z } ( \vartheta _ { k i 3 } ) ,\tag{C3}
$$

followed by CNOT gates between neighboring qubits.

Training is performed using mini-batch stochastic optimization with batch size 16 over 50 epochs. We employ the Adam optimizer with learning rate 0.1 and set the loss parameter to $\lambda = 0 . 8$

Experiments on CIFAR data are conducted with n = 5 qubits and 5 classes. Images are first average-pooled and subsequently projected to a 5-dimensional PCA representation before normalization to the interval $[ - \pi , \pi ]$ . For the FashionMNIST o.o.d. dataset, the same preprocessing pipeline is applied. All remaining hyperparameters and architectural choices are identical to the MNIST setting.

For each prediction, the volume of the singleton credal set was computed classically from the estimated singleton beliefs and plausibilities using an exact inclusion-exclusion formula for the volume of a simplex [46].

The MNIST and CIFAR models achieve in-distribution test accuracies of 60.3% and 34.3%, respectively. The goal of the experiments is not to optimize predictive performance, but rather to evaluate the behavior of the proposed DST-based uncertainty measures in a proof-of-concept setting.

d. Mass fusion from multiple classifiers. To evaluate the effect of Dempster-Shafer mass fusion, we consider a multiview classification setup based on the MNIST dataset. Each input image is partitioned into four disjoint patches of equal size, and a separate model is trained on each patch. For each patch, the input is reduced to match the number of qubits using principal component analysis $( \mathrm { P C A } )$ and subsequently rescaled to the interval $[ - \pi , \pi ]$ ]. We use $n _ { \mathrm { t r a i n } } = 3 0 0$ training samples and $n _ { \mathrm { t e s t } } = 3 0 0$ test samples.

Each classifier is implemented as a parameterized quantum circuit with the same architecture as described in the previous paragraphs, using 5 layers with AngleEmbedding for data encoding and nearest-neighbor entangling gates.

Training is performed using mini-batch stochastic optimization with batch size 16 over 50 epochs. We use the Adam optimizer with learning rate 0.1 and set the loss parameter to $\lambda = 0 . 8$ . The individual mass functions obtained from the trained classifiers are fused using Dempster-Shafer combination rules during evaluation.

(a)  
![](images/0d7f6296d0947eeffb47b928b78ea0d7949f1ffe8ec4bd136b1e79776b5cf698.jpg)

(b)  
![](images/3927dede28e3a39bde32c5ce8d65619306ce29e614dd4707bdc10961e85a1202.jpg)  
Figure A1. Average empty-set mass as a function of the number of classes. (a) MNIST and (b) CIFAR10. Error bars denote the standard deviation across test samples. Although the proposed encoding does not explicitly enforce the Dempster-Shafer constraint $m ( \emptyset ) = 0 .$ , the learned mass assigned to the empty set decreases with the number of classes and is negligible for the larger-scale settings considered in the main experiments.

## 2. Empty-set mass

In contrast to what is required by standard Dempster-Shafer theory, the encoding in Eq. (9) does not explicitly enforce $m ( \emptyset ) =$ 0, since the empty set corresponds to the all-zero computational basis state. Figure A1 shows the average mass assigned to the empty set as a function of the number of classes. Although nonzero empty-set masses can occur, the learned mass assigned to the empty set decreases rapidly with problem size and becomes negligible in the regimes considered.

The details of the experiment are as follows: We consider a multi-class classification task based on the MNIST (CIFAR10) dataset with n classes, where n is varied from 2 to 10. The input data is reduced to n dimensions using principal component analysis $( \mathrm { P C A } )$ and subsequently rescaled to the interval $[ - \pi , \pi ]$ to match the range required for quantum feature encoding. We use $n _ { \mathrm { t r a i n } } = 3 0 0$ training samples and $n _ { \mathrm { t e s t } } = 3 0 0$ test samples.

The model is implemented as a PQC of n qubits consisting of 5 layers. Classical input data is encoded using the AngleEmbedding routine provided by PennyLane. Each layer k of the circuit consists of parameterized single-qubit rotations followed by entangling operations. Concretely, for each qubit i and layer k, we apply

$$
\mathrm { R X } ( \vartheta _ { k i 1 } ) \mathrm { R Y } ( \vartheta _ { k i 2 } ) \mathrm { R Z } ( \vartheta _ { k i 3 } ) ,\tag{C4}
$$

followed by controlled-NOT (CNOT) gates between neighboring qubits. Training is performed using mini-batch stochastic optimization with batch size 16 over 50 epochs. We use the Adam optimizer with learning rate 0.1 to minimize the loss defined in Eq. (7). The hyperparameter is set to $\lambda = 0 . 8$