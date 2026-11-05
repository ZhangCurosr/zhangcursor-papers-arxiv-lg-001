# DO UNLEARNING ALGORITHMS LEARNTHE RIGHT FEATURES?

Polina Dolgova<sup>1,2</sup> Amartya Sanyal<sup>3</sup> Sebastian U. Stich<sup>1</sup>

<sup>1</sup>CISPA Helmholtz Center for Information Security <sup>2</sup>Saarland University

<sup>3</sup>Department of Computer Science, University of Copenhagen

{polina.dolgova, stich}@cispa.de amsa@di.ku.dk

## ABSTRACT

Machine unlearning is commonly evaluated with standard post-deletion metrics that compare an unlearned model to one retrained on the remaining data. Such snapshot evaluations can conceal consequential differences in the features the two models encode. We prove that if an unlearned and a retrained model agree on their outputs but encode different features, a single gradient step on their prediction heads can break this agreement. We further show that deleting an arbitrarily small fraction of the data can cause retraining to select qualitatively different features, so matching retraining can require learning features that the full model never had. We then introduce a measure of feature recovery that separates missing from extraneous features. Across eight unlearning methods, models that perform well on standard metrics often fail to recover the retrained features. This mismatch predicts deviations from retraining under subsequent relearning and downstream adaptation, and directly correcting it systematically reduces these deviations.

## 1 INTRODUCTION

Training data may need to be removed after a model has been trained, for example, in response to privacy requests, licensing issues, or discovered errors. Deleting these examples from the dataset, however, does not remove their influence from the trained model. Retraining from scratch on the remaining data therefore provides the natural reference for successful deletion, but it can be expensive. Empirical machine unlearning instead aims to approximate retraining at substantially lower cost.

In image classification, most evaluations assess unlearning immediately after the model has been updated. These evaluations range from standard utility and forgetting metrics (Fan et al., 2024; Ebrahimpour-Boroojeny et al., 2025a) to more targeted comparisons of predictive behavior (Ebrahimpour-Boroojeny et al., 2025b; Dolgova & Stich, 2026), but largely evaluate the model at a single point in time. An unlearned model may later be trained on new data, adapted to downstream tasks, updated in response to further deletion requests, or exposed to relearning attacks (Hu et al., 2025; Xu et al., 2026; Siddiqui et al., 2025). Differences that are invisible immediately after unlearning may become consequential during such further training. This leads to our central question:

When an unlearned model matches retraining in its predictions, what else must it recover for that agreement to persist under further training; and do common unlearning algorithms achieve this?

We argue that a key part of the answer is the feature solution selected by retraining: which predictive features a model learns and how strongly it represents them. Two models can make similar predictions using different features and therefore respond differently when training continues.

We first ask when differences in learned features can remain harmless. Different representations can produce the same predictions under a fixed head, but this agreement may disappear once the head is updated. We show that preserving agreement across all one-step head updates requires matching model-defined feature directions and strengths. Feature recovery therefore affects whether agreement with retraining persists during adaptation.

Table 1: Missing retraining features predict downstream accuracy loss. ViT-Tiny/CIFAR-100 with 20% of the training data removed; a fresh linear head is trained on Flowers-102 on top of each frozen representation. Feature discrepancy measures how much of the retraining feature solution is missing on Flowers-102 inputs (Section 5). Retraining transfers as well as the full-data model, while unlearning methods lose accuracy as their discrepancy grows.
<table><tr><td></td><td>Retrain</td><td>Initial model</td><td>IU</td><td>ReprUnl</td><td>LTD</td><td>SalUn</td><td>FT</td><td>NegGrad+</td><td>AMUN</td><td>RL</td></tr><tr><td>Feature Discrepancy</td><td>0.000</td><td>0.049</td><td>0.050</td><td>0.079</td><td>0.099</td><td>0.134</td><td>0.139</td><td>0.144</td><td>0.151</td><td>0.160</td></tr><tr><td>Mean Accuracy (%)</td><td>90.29</td><td>89.71</td><td>89.69</td><td>85.61</td><td>85.49</td><td>80.40</td><td>79.62</td><td>84.44</td><td>78.00</td><td>75.46</td></tr></table>

Next, we show that unlearning can require learning new features. One might hope that retraining after deletion largely preserves the features learned on the full data, so that successful unlearning only needs to remove or weaken existing information. We show otherwise: in a simple theoretical construction, the full model relies on an easy shortcut, while deleting an arbitrarily small fraction of the data causes retraining to suppress it and learn a predictive feature that the full model never had. Matching retraining can therefore require learning a new feature, rather than merely erasing old ones. We also show that several approximate-unlearning dynamics either fail to produce this transition or can require optimization time approaching that of retraining.

Empirically, we distinguish feature change from feature recovery. Using the model-defined feature framework of Jiang et al. (2024), we introduce metrics for comparing unlearned and retrained feature solutions. These separately capture missing features that appear during retraining and features that should disappear after unlearning. Together with comparison to the full-data solution, this distinguishes feature persistence from off-target change. On ViT-Tiny/CIFAR-100, we evaluate eight approximate unlearning methods for image classification and find that methods with strong static unlearning performance can nevertheless produce very different future solutions: some remain close to the full-data model, while others move far from it without approaching retraining. We observe the same qualitative behavior with ResNet-56, SVHN, and smaller deletion fractions.

Finally, we connect feature mismatch to behavior under adaptation. Feature mismatch on the forget distribution tracks how far the model deviates from retraining after partial relearning. On downstream tasks, missing retraining features on target inputs similarly track losses in transfer performance (Table 1). We then test these relationships more directly by intervening on the feature components identified in our comparison: we restore missing retraining features and remove unmatched unlearned features. Reducing the measured mismatch through these interventions generally brings the model closer to retraining. Together, these results motivate evaluating unlearning not only by immediate behavior, but also by whether it recovers the feature solution needed to remain aligned with retraining under subsequent adaptation.

## 1.1 RELATED WORK

Representation-level unlearning. Recent work goes beyond output-level evaluation. Almudevar &´ Ortega (2026) and Sepahvand et al. (2025) aim to suppress forget-set information in learned representations. Others show that output-level forgetting can leave representations close to the full model (Kim et al., 2026), retain forgotten-class information (Gao et al., 2026), or differ from retraining despite output agreement (Yong et al., 2026). We distinguish missing retraining features from unmatched features in the unlearning model. In particular, we ask which features retraining selects, whether unlearning algorithms recover them, and whether missing them affects downstream adaptation. Our analysis builds on the model-defined feature framework of Jiang et al. (2024); we discuss connections to representation-similarity measures in Appendix F.

Post-unlearning adaptation and relearning. Prior work studies whether forgotten behavior can re-emerge under subsequent optimization (Hu et al., 2025; Xu et al., 2026; Siddiqui et al., 2025)or whether apparently forgotten class information remains recoverable from learned representations (Dehghani et al., 2026; Jang et al., 2026). Siddiqui et al. (2025), in particular, relates relearning after retain-set fine-tuning to weight-space displacement from the full-data model. We instead ask whether unlearning recovers the features selected by retraining and whether failure to do so leads to divergence from retraining under subsequent adaptation, theoretically under head adaptation and empirically under relearning and downstream transfer.

Additional work on unlearning guarantees, methods, and evaluation is discussed in Appendix A.

## 2 PRELIMINARIES

Machine Unlearning Setup. Let $D = D _ { r } \sqcup D _ { f }$ be the training set, partitioned into a retain set $D _ { r }$ and a forget set $D _ { f }$ . For a possibly randomized training algorithm ${ \mathcal { A } } ,$ let

$$
\theta _ { \mathrm { f u l l } } \sim \mathcal { A } ( D ) , \qquad \theta _ { \mathrm { r e t r } } \sim \mathcal { A } ( D _ { r } ) , \qquad \theta _ { \mathrm { u n l } } \sim \mathcal { U } ( \theta _ { \mathrm { f u l l } } , D _ { f } , D _ { r } ) ,
$$

where U is an unlearning method. The ideal unlearning target is to match the distribution of $\theta _ { \mathrm { r e t r } }$ . In practice, approximate unlearning is often evaluated by the weaker criterion of output agreement with retraining. We study whether this agreement extends to the feature solution selected by retraining, and whether feature-level differences become observable under subsequent model use.

Model Representations and Features. We follow the model-defined feature framework of Jiang et al. (2024). For a model $f _ { \theta } ( x ) = W _ { \theta } ^ { \top } \phi _ { \theta } ( x )$ and a comparison set $\mathcal { X } = \{ x _ { 1 } , \ldots , x _ { N } \}$ , let $\Phi _ { \theta } =$ $[ \phi _ { \theta } ( x _ { 1 } ) , \ldots , \phi _ { \theta } ( x _ { N } ) ] ^ { \top }$ be the representation matrix, and let $\begin{array} { r } { \widetilde { \Phi } _ { \theta } = ( I _ { N } - \frac { 1 } { N } \mathbf { 1 } \mathbf { 1 } ^ { \top } ) \Phi _ { \theta } } \end{array}$ be its columncentered version. Write its Singular Value Decomposition as

$$
\begin{array} { r } { \widetilde { \Phi } _ { \theta } = U _ { \theta } \Sigma _ { \theta } V _ { \theta } ^ { \top } , \qquad U _ { \theta } = [ u _ { \theta , 1 } , u _ { \theta , 2 } , \ldots ] , \qquad \lambda _ { \theta , i } = ( \Sigma _ { \theta } ) _ { i i } ^ { 2 } , } \end{array}
$$

The left singular vector $u _ { \theta , \ast }$ gives the normalized response pattern of the ith PCA direction across $x ,$ with strength $\lambda _ { \theta , i }$

Let R be a set of independently trained models obtained under the same training setup. From each model $\theta \in \mathcal R$ , we take the top-K PCA response patterns and cluster them jointly by similarity, yielding clusters $C _ { 1 } , \ldots , C _ { J _ { \mathcal { R } } }$ . Each cluster $C _ { t }$ defines a feature, represented by its sign-aligned response centroid $c _ { t }$ and average strength $w _ { t } = | C _ { t } | ^ { - 1 } \sum _ { ( \theta , i ) \in C _ { t } } \lambda _ { \theta , i } .$ . We collect these strengths in $w _ { \mathscr R } = ( w _ { 1 } , \dots , w _ { J _ { \mathscr R } } ) ^ { \top }$ . Details of the clustering and centroid construction are given in Appendix B.

## 3 WHY FEATURE MISMATCH MATTERS UNDER ADAPTATION

Empirical unlearning evaluations are typically static: they compare the unlearned and retrained mod els immediately after unlearning. In practice, however, an unlearned model may later be fine-tuned, trained on additional data, or adapted to a downstream task. We ask when predictive agreement between unlearned and retrained models remains stable under such updates.

We begin with a lightweight adaptation regime in which the learned representation is frozen and only the prediction head is updated. Such updates arise naturally in relearning attacks and in downstream adaptation with a frozen pretrained backbone. Proofs of all results in this section are in Appendix C.

Consider retrained and unlearned models with $f _ { \mathrm { r e t r } } ( x ) , f _ { \mathrm { u n l } } ( x ) \in \mathbb { R } ^ { C }$

$$
f _ { \mathrm { r e t r } } ( x ) = W _ { \mathrm { r e t r } } ^ { \top } \phi _ { \mathrm { r e t r } } ( x ) , \qquad f _ { \mathrm { u n l } } ( x ) = W _ { \mathrm { u n l } } ^ { \top } \phi _ { \mathrm { u n l } } ( x ) .
$$

Fix a labeled dataset $\{ ( x _ { i } , y _ { i } ) \} _ { i = \cdot } ^ { N }$ with input set $\mathcal { X } = \{ x _ { 1 } , \ldots , x _ { N } \}$ , and suppose the models are initially predictively equivalent on X ,

$$
W _ { \mathrm { r e t r } } ^ { \top } \phi _ { \mathrm { r e t r } } ( x ) = W _ { \mathrm { u n l } } ^ { \top } \phi _ { \mathrm { u n l } } ( x ) \qquad \forall x \in \mathcal { X } .\tag{1}
$$

For a differentiable loss $\ell ( z , y )$ applied to the prediction $z = f ( x )$ , define for each $( x _ { i } , y _ { i } )$

$$
\begin{array} { r } { g _ { i } : = \nabla _ { z } \ell ( z , y _ { i } ) \big | _ { z = f _ { \mathrm { r e t r } } ( x _ { i } ) } = \nabla _ { z } \ell ( z , y _ { i } ) \big | _ { z = f _ { \mathrm { u n l } } ( x _ { i } ) } \in \mathbb { R } ^ { C } , } \end{array}
$$

where the equality follows from predictive equivalence. Updating only the head on $( x _ { i } , y _ { i } )$ with step size η gives

$$
W _ { m , i } ^ { + } = W _ { m } - \eta \phi _ { m } ( x _ { i } ) g _ { i } ^ { \top } , \qquad m \in \{ \mathrm { r e t r , u n l } \} .
$$

Let $F _ { m , i } ^ { + } \in \mathbb { R } ^ { N \times C }$ collect the predictions on X after this update, and let $\Phi _ { m }$ and $\widetilde { \Phi } _ { m }$ denote the uncentered and centered representation matrices on $\mathcal { X } _ { : }$ , respectively.

The result below shows that predictive equivalence can be fragile under even a single head update.

Proposition 3.1. Suppose that $g _ { i } \neq 0$ for all i and that $f _ { \mathrm { r e t r } } ( x ) = f _ { \mathrm { u n l } } ( x ) f o r a l l x \in \mathcal { X } .$ . Then

$$
\sum _ { i = 1 } ^ { N } \frac { \lVert F _ { \mathrm { r e t r } , i } ^ { + } - F _ { \mathrm { u n l } , i } ^ { + } \rVert _ { F } ^ { 2 } } { \eta ^ { 2 } \lVert g _ { i } \rVert _ { 2 } ^ { 2 } } = \left. \Phi _ { \mathrm { r e t r } } \Phi _ { \mathrm { r e t r } } ^ { \top } - \Phi _ { \mathrm { u n l } } \Phi _ { \mathrm { u n l } } ^ { \top } \right. _ { F } ^ { 2 } \geq \left. \widetilde { \Phi } _ { \mathrm { r e t r } } \widetilde { \Phi } _ { \mathrm { r e t r } } ^ { \top } - \widetilde { \Phi } _ { \mathrm { u n l } } \widetilde { \Phi } _ { \mathrm { u n l } } ^ { \top } \right. _ { F } ^ { 2 } .
$$

The proposition quantifies how representation mismatch can be exposed by even a single head update. The centered Gram matrix decomposes as $\begin{array} { r } { \tilde { \Phi } \tilde { \Phi } ^ { \top } = \sum _ { j } \lambda _ { j } u _ { j } \bar { u } _ { j } ^ { \top } } \end{array}$ , so discrepancies in it reflect differences in both the response directions and strengths of model-defined features. As a special case, if output equivalence on $\mathcal { X }$ is preserved after updating on each $( x _ { i } , y _ { i } ) , i \in [ N ]$ , the centered Gram matrices must coincide, and the two models therefore have the same feature structure on $x ,$ up to the usual eigenspace ambiguities.

More generally, larger centered Gram mismatch implies larger normalized prediction disagreement after adaptation. The bound is one-sided: small centered mismatch does not rule out additional disagreement from components removed by centering. When the adaptation gradient is non-negligible, this lower bound translates centered Gram mismatch into a non-negligible prediction gap, which is particularly relevant for relearning on the forget set.

A fixed-head perspective. The analysis above considers output equivalence between complete retrained and unlearned models. Alternatively, if unlearning is viewed as removing forget-set information from the model, the representation and prediction head may be judged separately rather than required to jointly match the same retrained model. Under this view, a retrained representation ϕ<sub>retr</sub> is indistinguishable from $\phi _ { \mathrm { u n l } }$ under the current unlearned head if

$$
W _ { \mathrm { u n l } } ^ { \top } \big ( \phi _ { \mathrm { r e t r } } ( x ) - \phi _ { \mathrm { u n l } } ( x ) \big ) = 0 \qquad \forall x \in \mathcal { X } .
$$

Such a difference is “harmless” in the sense of Grant et al. (2026). In Appendix C.2, we show that this harmlessness is generally unstable: once the head is adapted, previously invisible representation differences can affect the predictions.

## 4 DATA DELETION CAN REQUIRE LEARNING NEW FEATURES

The previous section shows why agreement with the retrained features matters under subsequent model use. We now ask how unlearning changes that solution relative to the full-data model. A common view of unlearning is that it is a subtractive operation: forgetting data should eliminate or weaken features that depended on those data, so recovering retraining would mainly require suppressing features already present in the full model. We show that deletion can instead induce a genuinely different feature solution. Even an arbitrarily small deletion can cause retraining to learn features that are absent in the full-data model. Thus, matching retraining may require more than simply erasing existing features: an unlearning method may need to learn new ones.

We study this phenomenon in a minimal model with a linear shortcut and a classical XOR feature. The construction captures a setting where the shortcut is predictive across the data, while the XOR signal differs across subpopulations. Removing even a small group can therefore change which feature is favored by training. Exact trajectories and proofs are deferred to Appendix D.

Setting. Let $\varepsilon \in ( 0 , 1 / 2 )$ be the forget-set fraction and $c > 0$ control the strength of the shortcut feature. With a slight abuse of notation, we write the input as the triplet $x = ( \overline { { s } } , \widetilde { z } ) \in \mathbb { R } ^ { 3 }$ , where $s \in \mathbb R$ and $\tilde { z } \in \mathbb { R } ^ { 2 }$ , with label $y \in \{ - 1 , 1 \}$ . The data distribution is a mixture of two distributions $D _ { r }$ and $D _ { f }$ with weights $1 - \varepsilon$ and ε, respectively:

$$
\begin{array} { r } { \frac { \ d b } { \ d t } \frac { \ b y } { \ d t } \left| \begin{array} { c c c } { \ b y } & { \mid } & { \ b z } \\ { \chi ( \ b z ) } & { \mid } & { \sqrt { \varepsilon } z } \\ { \ b D _ { f } } \end{array} \right. \qquad z \sim \mathrm { U n i f } ( \{ - 1 , 1 \} ^ { 2 } ) , \quad \chi ( z ) = z _ { 1 } z _ { 2 } . } \end{array}
$$

To capture the mechanisms of both the shortcut and XOR features while keeping the training dynamics of the unlearning algorithms analytically tractable, we use a quadratic network,

$$
f _ { \theta } ( s , \tilde { z } ) = a s + \frac 1 2 ( u ^ { \top } \tilde { z } ) ^ { 2 } - \frac 1 2 ( v ^ { \top } \tilde { z } ) ^ { 2 } ,
$$

trained with logistic loss under full-batch gradient flow from $a ( 0 ) = 0 , u ( 0 ) = v ( 0 ) = \delta e _ { 1 } , \delta > 0$ The linear term captures the shortcut, while the quadratic terms can represent the XOR feature, since $z _ { 1 } z _ { 2 }$ is a difference of two squared linear forms.

The shortcut $s = c y$ is predictive in both groups $D _ { r }$ and $D _ { f }$ . The tuple $\tilde { z }$ carries a classical XOR feature: $\tilde { z } _ { 1 } \tilde { z } _ { 2 }$ agrees with $y$ on $D _ { r }$ and is reversed on $D _ { f }$ . In the full distribution, z˜ is scaled so that these contributions cancel, leaving the shortcut as the predictive signal, whereas after deleting $D _ { f }$ the XOR feature becomes predictive on $D _ { r }$ . We note that despite this rescaling, the groups have comparable input norms for large $c ,$ differing by at most a factor $\sqrt { 1 + 2 c ^ { - 2 } }$ . With $D _ { f }$ an ε-fraction of the data, the construction covers arbitrarily small deletions.

For feature analysis, we use the corresponding three-dimensional feature representation

$$
\phi _ { \theta } ( \boldsymbol { x } ) = \left( a s , ( u ^ { \top } \tilde { z } ) ^ { 2 } , ( v ^ { \top } \tilde { z } ) ^ { 2 } \right) ^ { \top }
$$

Below, we analyze full-data training, retraining, and three unlearning dynamics: Random Labels (RL), which optimizes on randomly relabeled forget examples; RL combined with fine-tuning on the retain set (RL+FT) (Warnecke et al., 2023); and Gradient Ascent (GA) (Thudi et al., 2022), which maximizes the original loss on the forget data. Along these trajectories, the model-defined features have the same identities and are associated with the same PCA directions,

$$
\boldsymbol { q } _ { \mathrm { s } } = ( 1 , 0 , 0 ) ^ { \top } , \qquad \boldsymbol { q } _ { \mathrm { + } } = \frac { 1 } { \sqrt { 2 } } ( 0 , 1 , 1 ) ^ { \top } , \qquad \boldsymbol { q } _ { \mathrm { - } } = \frac { 1 } { \sqrt { 2 } } ( 0 , 1 , - 1 ) ^ { \top } ,
$$

with varying strengths. For trajectory $m ,$ we denote these strengths at time t by $\lambda _ { \mathrm { s } } ^ { m } ( t ) , \lambda _ { + } ^ { m } ( t )$ , and $\lambda _ { - } ^ { m } ( t )$

We now compare the features selected by full-data training and by retraining on $D _ { r }$ . Although the forget set has a mass of only $\varepsilon ,$ the two procedures converge to qualitatively different solutions.

Theorem 4.1. As training time $t \to \infty ,$

• full-data training satisfies $\frac { \lambda _ { \mathrm { s } } ^ { \mathrm { f u l l } } ( t ) } { \lambda _ { + } ^ { \mathrm { f u l l } } ( t ) } $ ∞ and $\lambda _ { - } ^ { \mathrm { f u l l } } ( t ) = 0$ for all $t \geq 0 .$

• under retraining on $\begin{array} { r } { D _ { r } , \frac { \lambda _ { \mathrm { s } } ^ { \mathrm { r e t r } } ( t ) } { \lambda _ { \mathrm { + } } ^ { \mathrm { r e t r } } ( t ) } \to 0 , \frac { \lambda _ { \mathrm { s } } ^ { \mathrm { r e t r } } ( t ) } { \lambda _ { \mathrm { - } } ^ { \mathrm { r e t r } } ( t ) } \to 0 , a n d \frac { \lambda _ { \mathrm { + } } ^ { \mathrm { r e t r } } ( t ) } { \lambda _ { \mathrm { - } } ^ { \mathrm { r e t r } } ( t ) } \to ( 1 - 2 \varepsilon ) ^ { 2 } . } \end{array}$

Theorem 4.1 shows that deletion can require more than reweighting features learned by the full model. Under retraining, the shortcut becomes asymptotically negligible, while a core feature $q _ { - }$ absent from the full-data solution emerges. Thus, matching retraining will require learning a feature that the full model never learned.

Unlearning Can Miss The Retrained Feature Solution. We now ask whether unlearning, initialized from the full-data model, can reproduce the above feature substitution induced by retraining. We first consider Random Labels (RL) and its version with retain-set fine-tuning (RL+FT).

Theorem 4.2. As the unlearning time $t \to \infty$ , for $M \in \{ \mathrm { R L } , \mathrm { R L } + \mathrm { F T } \}$ ,

$$
\frac { \lambda _ { \mathrm { s } } ^ { M } ( t ) } { \lambda _ { - } ^ { M } ( t ) } \longrightarrow \frac { 1 - \varepsilon } { 2 \varepsilon } > \frac { 1 } { 2 } ,
$$

while the corresponding ratio converges to zero under retraining, as shown in Theorem 4.1.

For small ε, the ratio $\frac { 1 - \varepsilon } { 2 \varepsilon }$ is large, so RL and RL+FT remain shortcut-dominated like the full-data solution, unlike retraining where this ratio vanishes. Even combined with retain-set fine-tuning, RL does not recover the retraining feature transition. Appendix D further shows that RL generally does not recover the retraining ratio between the two core strengths, whereas $\mathrm { R L + F T }$ does.

Next, we analyse Gradient Ascent (GA). We show that GA behaves differently: it eventually recovers the retraining-like feature solution, but requires nearly as much optimization as retraining from scratch. Train the full-data and retrained models for the same duration $T .$ Let $\tau _ { \mathrm { G A } } ( T )$ be the first time gradient ascent, initialized from the full-data model, matches the retrained model’s feature-strength ratio $\lambda _ { s } / \lambda _ { - }$

Theorem 4.3. For every $\zeta \in ( 0 , 1 )$ , ifthe initialisation $\delta > 0$ is sufficiently small or the shortcut strength c is sufficiently large, keeping other parameters fixed, then

$$
\operatorname* { l i m } _ { T \to \infty } \frac { \tau _ { \mathrm { G A } } ( T ) } { T } \geq 1 - \zeta
$$

Thus, GA requires a non-vanishing fraction of the retraining time to recover the same feature balance, and this fraction can approach one. Together, these results expose two distinct difficulties for approximate unlearning: converging to the wrong feature solution, as RL does, or recovering the right one only with retraining-scale optimization, as can occur for GA.

The key point is that successful unlearning requires a specific feature transition, not arbitrary movement away from the full model. Some full-model features should vanish or weaken, while others may need to emerge. This motivates the directional comparisons in Section 5: we measure both what unlearning retains from the full-model solution and what it fails to recover from retraining.

## 5 DOES UNLEARNING RECOVER RETRAINED FEATURES?

We now turn to neural networks and ask whether approximate unlearning recovers the specific feature solution selected by retraining. We first introduce a directional comparison between feature solutions and then use it to compare full-data, retrained, and unlearned models.

To compare feature solutions, we ask which features of a reference solution are reproduced by another model. Since model-defined features need not recur across random seeds (Jiang et al., 2024), we compare sets of independently trained or unlearned models rather than individual runs. Let R denote the reference model set and M the evaluated model set $( \mathrm { e . g }$ ., retrained runs and runs of one unlearning method, respectively). We write $\{ c _ { t } \} _ { t = 1 } ^ { T }$ and $w _ { \mathcal { R } }$ for the feature clusters and strength vector of R, as defined in Section 2. All comparisons below are computed on a shared input set, on which the resulting metrics depend; we suppress this dependence in the notation for readability.

For each reference feature cluster with centroid $c _ { t }$ , we ask how well it is matched by any of the top-K PCA components in the evaluated group M. We quantify its match quality by

$$
\rho _ { t } ( \mathcal { R } , \mathcal { M } ) = \operatorname* { m a x } _ { \theta \in \mathcal { M } , \ j \in [ K ] } \left| \operatorname { c o r r } ( c _ { t } , u _ { \theta , j } ) \right| .
$$

We convert this match quality into a soft penalty,

$$
d _ { t } ( \mathcal { R } , \mathcal { M } ) = \sqrt { 1 - \operatorname* { m i n } \{ 1 , \rho _ { t } ( \mathcal { R } , \mathcal { M } ) / \gamma \} ^ { 2 } } ,
$$

where $\gamma$ is a matching threshold derived from the correlation threshold used for feature clustering (App. B). Features matched at or above $\gamma$ incur no penalty, while weaker matches incur larger penalties. Collecting these penalties in $d ( \mathcal { R } , \mathcal { M } )$ , we define the directional feature discrepancy as

$$
D _ { \mathrm { f e a t } } ( \mathcal { R } , \mathcal { M } ) = \frac { \| w _ { \mathcal { R } } \odot d ( \mathcal { R } , \mathcal { M } ) \| _ { 2 } } { \| w _ { \mathcal { R } } \| _ { 2 } } .\tag{2}
$$

Thus, intuitively, the metric asks whether each feature of the reference solution has a recognizable counterpart in $\dot { \mathcal { M } }$ . If some response in M resembles that feature, its contribution is small; if nothing does, its contribution is large, with stronger reference features weighted more heavily. The metric therefore provides a soft, strength-weighted measure of how much of the reference feature solution is missing from $\mathcal { M }$

The particular form of the discrepancy is motivated by the centered Gram-matrix discrepancy in Section 3. For a single model, $\begin{array} { r } { \widetilde { \Phi } \widetilde { \Phi } ^ { \top } = \sum _ { i } \lambda _ { i } u _ { i } u _ { i } ^ { \top } } \end{array}$ . Consider first the idealized case where a set $\mathcal { Q }$ of reference features is absent from the other solution, with their response directions orthogonal to those represented there. By orthonormality, their contribution to the squared Frobenius discrepancy is

$$
\Vert \sum _ { i \in \mathcal { Q } } \lambda _ { i } u _ { i } u _ { i } ^ { \top } \Vert _ { F } ^ { 2 } = \sum _ { i \in \mathcal { Q } } \lambda _ { i } ^ { 2 } .
$$

This motivates weighting each feature-recovery error by its strength and aggregating these errors in $L _ { 2 }$ . For partially matched features, the discrepancy between two unit response directions satisfies

$$
\| u u ^ { \top } - v v ^ { \top } \| _ { F } ^ { 2 } = 2 \big ( 1 - ( u ^ { \top } v ) ^ { 2 } \big ) .
$$

![](images/973a62207949586de256995dd893d778c5662311cad7f37a989383676c62bc85.jpg)

![](images/f71b6e802a55142956ff76530f9726ab5b3b12d08c42951cd10fbc9a79758e63.jpg)

![](images/2a6a63034148bb484126ee42874dddb81f53d9d4444b5ef739aa56943310b341.jpg)  
Figure 1: The left panel illustrates that retraining moves away from some full-data features while favoring others that were previously less well represented; for NegGrad+, incomplete recovery of retraining features dominates the mismatch, consistent with over-suppression. The middle and right panels show the two-sided feature discrepancy from the full-data and retrained model sets on the test and forget sets, respectively.

This motivates the squared-correlation dependence in $d _ { t }$ . We rescale the correlation by the matching threshold $\gamma ,$ so that sufficiently well-matched directions incur no penalty. Thus, $w _ { t }$ captures the strength of a reference feature, while $d _ { t }$ captures how well its response direction is recovered.

Because $D _ { \mathrm { f e a t } } ( \mathcal { R } , \mathcal { M } )$ scores only features of the reference solution, it is directional. To capture mismatch in both directions, we define

$$
D _ {  } ( \mathcal { R } , \mathcal { M } ) = \sqrt { \frac { \| w _ { \mathcal { R } } \| _ { 2 } ^ { 2 } ( D _ { \mathrm { f e a t } } ( \mathcal { R } , \mathcal { M } ) ) ^ { 2 } + \| w _ { \mathcal { M } } \| _ { 2 } ^ { 2 } ( D _ { \mathrm { f e a t } } ( \mathcal { M } , \mathcal { R } ) ) ^ { 2 } } { \| w _ { \mathcal { R } } \| _ { 2 } ^ { 2 } } } .\tag{3}
$$

The first term captures reference features missing from $\mathcal { M } .$ , while the second captures features of M that are absent from the reference solution. We combine the two contributions quadratically, mirroring the structure of the Gram-matrix discrepancy, and normalize by the total feature strength of the reference solution, which is fixed across methods and therefore does not affect their ordering. Thus, $D _ {  }$ measures two-sided feature mismatch relative to the reference solution.

Although motivated by the Gram-matrix discrepancy, our metric isolates feature recovery rather than full Gram-matrix mismatch. Once a feature is well matched, strength differences incur no additional penalty, and the metric does not account for how frequently that feature recurs across models within a group. We analyze feature-strength differences separately for matched features in Appendix E.

For unlearning, we compare each model group with both the full-data and retrained feature solutions. Discrepancy from retraining is our primary measure of feature recovery, while discrepancy from the full-data solution measures how much the feature solution has changed during unlearning. Together, the two comparisons distinguish methods that preserve the full-data solution from those that change it without recovering the solution selected by retraining.

Feature Change Does Not Imply Feature Recovery. We study ViT-Tiny on CIFAR-100, removing one fine-grained class from each of the 20 superclasses (20% of the training data). Each model set uses 20 seeds; feature results are averaged over three repetitions. We evaluate eight unlearning baselines: fine-tuning (FT), Random Labels (RL) (Warnecke et al., 2023), NegGrad+ (Kurmanji et al., 2023), SalUn (Fan et al., 2024), AMUN (Ebrahimpour-Boroojeny et al., 2025a), Influence Unlearning (IU) (Koh & Liang, 2017; Jia et al., 2023), LTD (Dolgova & Stich, 2026), and Representation Unlearning (ReprUnl) (Almudevar & Ortega, 2026), which explicitly modifies learned representa-´ tions. Most perform well under standard post-unlearning metrics, allowing us to ask whether they also recover the retrained feature solution. Results for ResNet-56, smaller deletion fractions, and SVHN are in Appendix E, with standard unlearning metrics in Appendix E.2.

Importantly, retraining changes the feature solution in both directions: some full-data features disappear, while others emerge only after deletion. In our main setting, the two directional discrepancies are comparable in magnitude (Figure 1). This echoes Section 4: matching retraining may require both suppressing existing features and recovering features absent from the full-data solution.

![](images/8c205823d8ca22985fdb9a119abdf1a058d72b240c12787a018921d169384f9d.jpg)  
(a) Partial relearning, Forget Set

![](images/0b74debe1cd9a7aa1ab613951b0038bbda1ed38b7b534de5dba9aa838179e83e.jpg)  
(b) Flowers-102

![](images/4184a47aa0c17715117936600a11f3670556f88695c001c3ce4d27dde7efc17e.jpg)  
(c) Oxford-IIIT Pets  
Figure 2: ViT-Tiny/CIFAR-100, Feature mismatch predicts behavior under subsequent adaptation. (a) Retraining-relative feature discrepancy on the forget distribution versus the absolute forget accuracy gap after partial relearning; colors show the sign of the gap. (b–c) Missing retraining features on downstream inputs versus downstream accuracy gap on Flowers-102 and Oxford-IIIT Pets.

Figure 1 compares each method with the full-data and retrained feature solutions on the test and forget distributions. Methods with similar performance under standard unlearning metrics can nevertheless reach very different feature solutions. Some remain close to the full-data model, indi cating feature persistence; others move far from it without approaching retraining, indicating offtarget change. This spread is especially clear on the forget distribution, where methods range from strongly full-like to substantially displaced, yet none closely recovers retraining. Thus, similar behavioral success can mask qualitatively different internal outcomes. The same pattern holds across additional architectures, datasets, and deletion settings; see Appendix E.

## 6 CONSEQUENCES OF FEATURE MISMATCH

The previous sections show that approximate unlearning can perform well on standard postunlearning evaluations without recovering the feature solution selected by retraining. We now ex amine how this mismatch relates to behavior under subsequent adaptation.

Throughout this section, we freeze the learned representation, discard the original prediction head, and train a new randomly initialized head for the task of interest. This removes information specific to the old head and isolates what remains encoded in the representation. We also report standard unlearning metrics after training a fresh head on the retain set in Appendix E.2; these generally improve, including for methods that did not initially match retraining.

Feature Mismatch Predicts Relearning. We first consider partial relearning. For each frozen representation, we train a fresh head on the retain set together with half of the forget examples, then evaluate on the full forget set. If forgotten information remains encoded in the representation, access to only part of the forget data may reveal it, producing behavior that differs from retraining under the same adaptation. We therefore evaluate feature discrepancy on the forget distribution and compare $D _ {  } ( \operatorname { r e t r } , \mathcal { M } )$ with the absolute forget-accuracy gap from retraining after partial relearning.

Figure 2a shows a clear relationship: models farther from the retrained feature solution exhibit larger behavioral deviations after relearning. The full-data model and IU form a separate cluster: as Figure 1 shows, IU remains very close to the full-data feature solution. This is consistent with fulllike representations retaining features that make the forgotten behavior easy to recover under partial relearning. Among the remaining methods, larger feature mismatch closely tracks larger deviation from retraining. The direction of the deviation varies across methods, but its magnitude grows with feature mismatch. Thus, representation differences that are hidden immediately after unlearning can be revealed by lightweight subsequent training, matching the mechanism identified in Section 3.

Missing Retraining Features Predict Downstream Loss. Feature mismatch can also have the opposite consequence: an unlearned representation may lack capabilities that are retained by retraining. We test this by freezing each backbone and training a fresh head on two downstream classification datasets, Oxford-IIIT Pets (Parkhi et al., 2012) and Flowers-102 (Nilsback & Zisserman, 2008).

Here the directional discrepancy is the relevant quantity: unlearning may suppress features that retraining would retain, and we ask whether such missing features are associated with downstream degradation. On each downstream distribution, we measure $D _ { \mathrm { f e a t } } ( \mathrm { r e t r } , \mathcal { M } )$ , capturing retraining features that are missing from the unlearned model on task-relevant inputs. We compare it with the downstream accuracy gap from retraining.

![](images/0c826b036a7b5836541bad73f3eb6a26d658f42f1ceca9bc6ea218eaa6ac45cb.jpg)

![](images/274727d0538b9f04c9be6afe6320b3bb2a0933e98e242b0b055abe16edca2543.jpg)

![](images/8098f11dda78716e8e79209028fb494a24ef0f248531df1d09626a52ac4133f3.jpg)  
Figure 3: Correcting retraining-relative feature mismatch moves behavior toward retraining. For SalUn, restoring missing retraining features improves transfer, while additionally removing unmatched unlearned features further reduces the relearning gap.

Across both tasks, missing retraining features closely track downstream performance loss: models with smaller directional discrepancy transfer similarly to retraining, while larger discrepancy corresponds to larger degradation.

Correcting Feature Mismatch Moves Behavior Toward Retraining. The correlations above show that feature mismatch predicts subsequent behavior but do not establish that the identified mismatch is responsible for it. We therefore intervene directly on the features identified by our comparison: we restore retraining features missing from the unlearned solution and remove unlearned features that are absent from retraining.

We apply these interventions at four increasing levels, progressively correcting the measured feature mismatch. After each intervention, we keep the modified representation fixed and train a fresh prediction head. We also include untargeted controls to distinguish correction of the identified mismatch from generic changes to the representation. Exact constructions are given in Appendix G.1.

Figure 3 shows a consistent movement toward retraining as the measured feature mismatch is corrected for SalUn (see Appendix G.2 for other methods). On Flowers-102 and Oxford-IIIT Pets, progressively injecting missing retraining features recovers most of the downstream performance gap, while random injection does not produce the same effect. Under partial relearning, correcting the identified mismatch also substantially reduces the gap from retraining. A control that combines the same ablation with random feature injection also improves behavior, but to a smaller extent. This suggests that both removing unmatched features and recovering the specific features selected by retraining contribute to the behavioral change.

These interventions provide stronger evidence than correlation alone. The feature discrepancy does not merely identify representations that behave differently later: directly correcting the specific mismatch measured by our comparison systematically moves subsequent behavior toward retraining. Together with the predictive results above, this supports feature recovery as a behaviorally meaningful target for unlearning rather than merely a descriptive notion of representation similarity.

## 7 CONCLUSION

We study machine unlearning through feature recovery rather than static agreement with retraining. Taken together, our results suggest that successful unlearning is not only about reproducing retraining at the moment of deletion, but also about recovering a model state that behaves similarly during subsequent use. Output-equivalent models can encode different feature solutions, and these differences can become consequential once training resumes.

This is particularly relevant to continual unlearning, where the model produced after one deletion request becomes the starting point for the next. Feature differences left by one update may therefore matter for subsequent requests, even when the model initially agrees with retraining. This motivates evaluating unlearning not only immediately after deletion, but also under subsequent updates, with feature recovery as a complementary criterion.

## ACKNOWLEDGEMENTS

This work was supported by the Helmholtz Association’s Initiative and Networking Fund on the HAICORE@FZJ partition. PD and SS gratefully acknowledge funding from the European Research Council (ERC) under the Horizon Europe Framework Programme (HORIZON) for proposal number 101170430 CollectiveMinds. Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them. AS also acknowledge support from VILLUM FONDEN via the Young Investigator program (72069) and the Novo Nordisk Foundation for support via the Startup grant (NNF24OC0087820).

## REFERENCES

Antonio Almudevar and Alfonso Ortega. Representation unlearning: Forgetting through information´ compression. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=CwjK5eUzcI.

Vikram S Chundawat, Ayush K Tarun, Murari Mandal, and Mohan Kankanhalli. Can bad teaching induce forgetting? unlearning in deep networks using an incompetent teacher. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 7210–7217, 2023.

Zahra Dehghani, Pablo Piantanida, and Mohammadhadi Shateri. Source-free class relearning: Diagnosing forgetting in class unlearning. arXiv preprint arXiv:2609.02018, 2026.

Polina Dolgova and Sebastian U. Stich. Forgetting has neighbors: Localized collateral forgetting in machine unlearning, 2026. URL https://arxiv.org/abs/2605.31317.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. CoRR, abs/2010.11929, 2020. URL https://arxiv.org/abs/2010.11929.

Ali Ebrahimpour-Boroojeny, Hari Sundaram, and Varun Chandrasekaran. Amun: Adversarial machine unlearning, 2025a. URL https://arxiv.org/abs/2503.00917.

Ali Ebrahimpour-Boroojeny, Yian Wang, and Hari Sundaram. On the necessity of output distribution reweighting for effective class unlearning, 2025b. URL https://arxiv.org/abs/2506. 20893.

Chongyu Fan, Jiancheng Liu, Yihua Zhang, Eric Wong, Dennis Wei, and Sijia Liu. Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=gn0mIhQGNM.

Yichen Gao, Altay Unal, Akshay Rangamani, and Zhihui Zhu. An illusion of unlearning? assessing machine unlearning through internal representations. arXiv preprint arXiv:2604.08271, 2026.

Shashwat Goel, Ameya Prabhu, Amartya Sanyal, Ser-Nam Lim, Philip Torr, and Ponnurangam Kumaraguru. Towards adversarial evaluations for inexact machine unlearning. arXiv preprint arXiv:2201.06640, 2022.

Shashwat Goel, Ameya Prabhu, Philip Torr, Ponnurangam Kumaraguru, and Amartya Sanyal. Corrective machine unlearning. arXiv preprint arXiv:2402.14015, 2024.

Satchel Grant, Simon Jerome Han, Alexa R. Tartaglini, and Christopher Potts. Addressing divergent representations from causal interventions on neural networks. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=cZrTMqYVL6.

Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, and Nicolas Papernot. Inexact unlearning needs more careful evaluations to avoid a false sense of privacy. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pp. 497–519. IEEE, 2025.

Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, 2015. URL https://api.semanticscholar.org/CorpusID:206594692.

Shengyuan Hu, Yiwei Fu, Steven Wu, and Virginia Smith. Unlearning or obfuscating? jogging the memory of unlearned LLMs via benign relearning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= fMNRYBvcQN.

Yurim Jang, Jaeung Lee, Dohyun Kim, Jaemin Jo, and Simon S Woo. Suppression or deletion: A restoration-based representation-level analysis of machine unlearning. In Proceedings ofthe ACM Web Conference 2026, pp. 8505–8508, 2026.

Jinghan Jia, Jiancheng Liu, Parikshit Ram, Yuguang Yao, Gaowen Liu, Yang Liu, Pranay Sharma, and Sijia Liu. Model sparsity can simplify machine unlearning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum? id=0jZH883i34.

Yiding Jiang, Christina Baek, and J Zico Kolter. On the joint interaction of models, data, and features. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=ze7DOLi394.

Yongwoo Kim, Sungmin Cha, and Donghyun Kim. Are we truly forgetting? a critical reexamination of machine unlearning evaluation protocols. Engineering Applications ofArtificial Intelligence, 167:113785, 2026.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 1885–1894. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/koh17a. html.

Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical Report 0, University of Toronto, Toronto, Ontario, 2009. URL https://www.cs. toronto.edu/<sub>˜</sub>kriz/learning-features-2009-TR.pdf.

Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, and Eleni Triantafillou. Towards unbounded machine unlearning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=OveBaTtUAT.

Seth Neel, Aaron Roth, and Saeed Sharifi-Malvajerdi. Descent-to-delete: Gradient-based methods for machine unlearning. In Vitaly Feldman, Katrina Ligett, and Sivan Sabato (eds.), Pro ceedings of the 32nd International Conference on Algorithmic Learning Theory, volume 132 of Proceedings of Machine Learning Research, pp. 931–962. PMLR, 16–19 Mar 2021. URL https://proceedings.mlr.press/v132/neel21a.html.

Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics and Image Processing, Dec 2008.

Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In IEEE Conference on Computer Vision and Pattern Recognition, 2012.

Nazanin Mohammadi Sepahvand, Eleni Triantafillou, Hugo Larochelle, Doina Precup, James J. Clark, Daniel M. Roy, and Gintare Karolina Dziugaite. Selective unlearning via representation erasure using domain adversarial training. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=KzSGJy1PIf.

Shoaib Ahmed Siddiqui, Adrian Weller, David Krueger, Gintare Karolina Dziugaite, Michael Curtis Mozer, and Eleni Triantafillou. From dormant to deleted: Tamper-resistant unlearning through weight-space regularization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=Zrqn7ZshXG.

Anvith Thudi, Gabriel Deza, Varun Chandrasekaran, and Nicolas Papernot. Unrolling SGD: Understanding Factors Influencing Machine Unlearning . In 2022 IEEE 7th European Symposium on Security and Privacy (EuroS&P), pp. 303–319, Los Alamitos, CA, USA, June 2022. IEEE Computer Society. doi: 10.1109/EuroSP53844.2022.00027. URL https://doi. ieeecomputersociety.org/10.1109/EuroSP53844.2022.00027.

Yiwen Tu, Pingbang Hu, and Jiaqi Ma. A reliable cryptographic framework for empirical machine unlearning evaluation. Advances in Neural Information Processing Systems, 38:131705–131736, 2026.

Alexander Warnecke, Lukas Pirch, Christian Wressnegger, and Konrad Rieck. Machine unlearning of features and labels. In Network and Distributed System Security (NDSS) Symposium 2023: 27 February - 3 March 2023, San Diego, CA, USA, 2023. ISBN 1-891562-83-5. doi: 10.14722/ndss. 2023.23087. 46.23.01; LK 01.

Heng Xu, Tianqing Zhu, Lefeng Zhang, Wanlei Zhou, and Philip S. Yu. Machine unlearning: A survey, 2023. URL https://arxiv.org/abs/2306.03558.

Xiaoyu Xu, Xiang Yue, Yang Liu, Qingqing Ye, Huadi Zheng, Peizhao Hu, Minxin Du, and Haibo Hu. Unlearning isn’t deletion: Investigating reversibility of machine unlearning in LLMs. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=E5SVowO13b.

Teresa Pui Yee Yong, Win Kent Ong, and Chee Seng Chan. Erased, but not gone: Output forgetting is not true forgetting, 2026. URL https://arxiv.org/abs/2606.25001.

Vladimir Zorich. Mathematical analysis. Part I. 01 2016. ISBN 978-3-662-48991-8. doi: 10.1007/ 978-3-662-48993-2.

A Additional Related Work 14   
B Feature Construction Details 14   
C Harmlessness 15   
C.1 Proof of Proposition 3.1 . 15   
C.2 Harmlessness Under a Shared Head 15   
C.3 Proofs for Harmlessness Under a Shared Head . 16   
C.3.1 Proof of Lemma C.1 16   
C.3.2 Proof of Proposition C.2 17   
C.3.3 Proof of Proposition C.3 17   
C.3.4 Proof of Proposition C.4 18   
D XOR Theory 18   
D.1 Connection to model features 19   
D.2 Proof roadmap 19   
D.3 Full-data training dynamics . 20   
D.4 Retain-data training dynamics 22   
D.5 Random Labels Unlearning dynamics 26   
D.6 Random Labels with Retain Fine-Tuning unlearning dynamics 30   
D.7 Gradient Ascent unlearning dynamics 33   
E Additional Results 39   
E.1 Experimental Details 39   
E.2 Standard Metrics 39   
E.3 Directional Discrepancies . 40   
E.4 Additional Settings 40   
E.5 Hyperparameter ablations . 42   
E.6 Feature-Strength Mismatch . 44   
F Relation to Linear CKA 45   
G Feature Injection and Ablation 46   
G.1 Experiment Details 46   
G.2 Additional Experiments . 48

## A ADDITIONAL RELATED WORK

Neel et al. (2021) distinguish statistical indistinguishability from retraining of the full maintained state from the weaker requirement of indistinguishability only at the observable output. Their setting concerns deletion guarantees for convex models; we instead study neural representations and ask when output agreement with retraining persists under subsequent head adaptation.

Approximate unlearning and evaluation. Approximate machine unlearning encompasses a broad range of practical approaches for modifying trained neural networks without retraining from scratch (Xu et al., 2023). Representative methods include fine-tuning and relabeling (Warnecke et al., 2023), gradient-based forgetting (Kurmanji et al., 2023), selective parameter updates (Fan et al., 2024), adversarial unlearning (Ebrahimpour-Boroojeny et al., 2025a), influence-based approximations (Koh & Liang, 2017; Jia et al., 2023), and representation-level objectives (Almudevar´ & Ortega, 2026).

Evaluation is similarly diverse. Standard protocols report retain/test utility and forget-set behavior, often relative to a retrained reference. Privacy-oriented evaluations test whether deleted examples remain distinguishable, commonly through membership-inference attacks (Hayes et al., 2025; Tu et al., 2026), while other criteria assess residual forget-set information without explicit comparison to retraining (Chundawat et al., 2023). Other works also test the removal of specifically instilled undesirable behavior through incorrect labels or data poisoning (Goel et al., 2022; 2024). Despite these different notions of successful deletion, evaluation remains predominantly snapshot-based, assessing the model immediately after unlearning rather than its agreement with retraining unde subsequent use.

## B FEATURE CONSTRUCTION DETAILS

Correlation threshold. Following Jiang et al. (2024), we determine the correlation threshold empirically from the reference ensemble. Let Λ collect the Pearson correlations between every pair of PCA component responses from different models, evaluated on the same shared inputs. We set

$$
\gamma _ { \mathrm { c o r r } } = \mathrm { p e r c e n t i l e } _ { 9 0 } ( | \Lambda | ) ,
$$

using the same 90th-percentile rule as Jiang et al. (2024).

Clustering. Components are matched using the absolute correlation of their response patterns. While Jiang et al. (2024) use greedy clustering, we use complete-linkage hierarchical clustering with threshold $\gamma _ { \mathrm { c o r r } }$ to avoid dependence on component ordering and make the resulting clusters reproducible for a fixed ensemble. Thus, components belong to the same cluster only if every pair within the cluster has absolute response correlation at least $\gamma _ { \mathrm { c o r r } }$

Feature centroids. Each cluster $C _ { t }$ defines a model-defined feature. Since PCA response directions are defined only up to sign, we first align the signs of the normalized responses $u _ { \theta , i }$ within each cluster and define its centroid as

$$
c _ { t } = \mathrm { u n i t } \left( \frac { 1 } { | C _ { t } | } \sum _ { ( \theta , i ) \in C _ { t } } \epsilon _ { \theta , i } u _ { \theta , i } \right) , \qquad \epsilon _ { \theta , i } \in \{ - 1 , + 1 \} ,
$$

where the signs $\epsilon _ { \theta , i }$ are aligned to the first member of $C _ { t }$ (ordered by model and component index): the anchor has $\epsilon = + 1$ , and each remaining response is flipped if necessary so that its inner product with the anchor is nonnegative. The arbitrary sign of the anchor only determines the overall sign of $c _ { t }$ , which is immaterial since subsequent comparisons use absolute correlation.

Matching threshold for feature discrepancy. The matching threshold $\gamma$ used in Section 5 is calibrated from the clustering threshold rather than chosen independently. We set

$$
\gamma = \operatorname* { m i n } ( 1 , \kappa \cdot \gamma _ { \mathrm { c o r r } } ) ,
$$

with $\kappa = 2 . 0$ by default in our experiments. Components within a reference feature are themselves only required to have pairwise correlation at least $\gamma _ { \mathrm { c o r r } } ,$ so matching at the same threshold would not imply close agreement with that feature. Our default $\kappa = 2$ avoids the strong compression observed at $\kappa = 1$ while keeping the discrepancy between independent retraining groups small. We report ablations over κ in Appendix E.

## C HARMLESSNESS

## C.1 PROOF OF PROPOSITION 3.1

Proof. A gradient-descent step on $( x _ { i } , y _ { i } )$ of step size η on the linear head gives

$$
\begin{array} { r l r } & { } & { W _ { \mathrm { r e t r } } ^ { + } = W _ { \mathrm { r e t r } } - \eta \phi _ { \mathrm { r e t r } } ( x _ { i } ) g _ { i } ^ { \top } , \qquad W _ { \mathrm { u n l } } ^ { + } = W _ { \mathrm { u n l } } - \eta \phi _ { \mathrm { u n l } } ( x _ { i } ) g _ { i } ^ { \top } . } \end{array}
$$

For $m \in \{ \mathrm { r e t r } , \mathrm { u n l } \}$ , let $f _ { m , i } ^ { + } ( x ) : = ( W _ { m , i } ^ { + } ) ^ { \top } \phi _ { m } ( x )$ denote the prediction at x after the head update on $( x _ { i } , y _ { i } )$ . Thus, predictive equivalence before the update gives, for every $x _ { j } \in { \mathcal { X } }$

$$
f _ { \mathrm { r e t r } , i } ^ { + } ( x _ { j } ) - f _ { \mathrm { u n l } , i } ^ { + } ( x _ { j } ) = - \eta g _ { i } \left( \phi _ { \mathrm { r e t r } } ( x _ { i } ) ^ { \top } \phi _ { \mathrm { r e t r } } ( x _ { j } ) - \phi _ { \mathrm { u n l } } ( x _ { i } ) ^ { \top } \phi _ { \mathrm { u n l } } ( x _ { j } ) \right) .
$$

Since the term in parentheses is scalar,

$$
\begin{array} { r } { \| f _ { \mathrm { r e t r } , i } ^ { + } ( x _ { j } ) - f _ { \mathrm { u n l } , i } ^ { + } ( x _ { j } ) \| ^ { 2 } = \eta ^ { 2 } \| g _ { i } \| ^ { 2 } \left( \phi _ { \mathrm { r e t r } } ( x _ { i } ) ^ { \top } \phi _ { \mathrm { r e t r } } ( x _ { j } ) - \phi _ { \mathrm { u n l } } ( x _ { i } ) ^ { \top } \phi _ { \mathrm { u n l } } ( x _ { j } ) \right) ^ { 2 } . } \end{array}
$$

Summing over j therefore gives

$$
\frac { \| F _ { \mathrm { r e t r } , i } ^ { + } - F _ { \mathrm { u n l } , i } ^ { + } \| _ { F } ^ { 2 } } { \eta ^ { 2 } \| g _ { i } \| _ { 2 } ^ { 2 } } = \sum _ { j = 1 } ^ { N } \left( \phi _ { \mathrm { r e t r } } ( x _ { i } ) ^ { \top } \phi _ { \mathrm { r e t r } } ( x _ { j } ) - \phi _ { \mathrm { u n l } } ( x _ { i } ) ^ { \top } \phi _ { \mathrm { u n l } } ( x _ { j } ) \right) ^ { 2 } .
$$

Summing over i yields

$$
\sum _ { i = 1 } ^ { N } \frac { \lVert F _ { \mathrm { r e t r } , i } ^ { + } - F _ { \mathrm { u n l } , i } ^ { + } \rVert _ { F } ^ { 2 } } { \eta ^ { 2 } \lVert g _ { i } \rVert _ { 2 } ^ { 2 } } = \left. \Phi _ { \mathrm { r e t r } } \Phi _ { \mathrm { r e t r } } ^ { \top } - \Phi _ { \mathrm { u n l } } \Phi _ { \mathrm { u n l } } ^ { \top } \right. _ { F } ^ { 2 } .
$$

Let $\begin{array} { r } { H = I - \frac { 1 } { N } \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}$ denote the centering matrix. Since

$$
\widetilde { \Phi } _ { \mathrm { r e t r } } \widetilde { \Phi } _ { \mathrm { r e t r } } ^ { \top } - \widetilde { \Phi } _ { \mathrm { u n l } } \widetilde { \Phi } _ { \mathrm { u n l } } ^ { \top } = H \left( \Phi _ { \mathrm { r e t r } } \Phi _ { \mathrm { r e t r } } ^ { \top } - \Phi _ { \mathrm { u n l } } \Phi _ { \mathrm { u n l } } ^ { \top } \right) H ,
$$

and H is an orthogonal projector,

$$
\left\| \widetilde { \Phi } _ { \mathrm { r e t r } } \widetilde { \Phi } _ { \mathrm { r e t r } } ^ { \top } - \widetilde { \Phi } _ { \mathrm { u n l } } \widetilde { \Phi } _ { \mathrm { u n l } } ^ { \top } \right\| _ { F } \leq \left\| \Phi _ { \mathrm { r e t r } } \Phi _ { \mathrm { r e t r } } ^ { \top } - \Phi _ { \mathrm { u n l } } \Phi _ { \mathrm { u n l } } ^ { \top } \right\| _ { F } .
$$

Combining the two displays proves the claim.

## C.2 HARMLESSNESS UNDER A SHARED HEAD

We formalize the fixed-head notion of harmlessness introduced in the main text. Fixing the unlearned head $W _ { \mathrm { u n l } }$ , the representational difference is harmless when

$$
W _ { \mathrm { u n l } } ^ { \top } \big ( \phi _ { \mathrm { r e t r } } ( x ) - \phi _ { \mathrm { u n l } } ( x ) \big ) = 0 \qquad \forall x \in \mathcal { X } .\tag{4}
$$

Thus, the two representations may differ in directions lying in ker $( W _ { \mathrm { u n l } } ^ { \top } )$ without affecting the current predictions.

We now ask whether this property is preserved when the head is adapted. The following lemma characterizes exactly when equation 4 survives a single gradient step.

Lemma C.1. Suppose equation 4 holds. After one nonzero gradient step on $W _ { \mathrm { u n l } }$ using an example $( x _ { \mathrm { f t } } , y _ { \mathrm { f t } } )$ and the representation $\phi _ { \mathrm { u n l } }$ , harmlessness is preserved if and only if

$$
\big ( \phi _ { \mathrm { r e t r } } ( x ) - \phi _ { \mathrm { u n l } } ( x ) \big ) ^ { \top } \phi _ { \mathrm { u n l } } ( x _ { \mathrm { f t } } ) = 0 \qquad \forall x \in \mathcal { X } .
$$

Hence, surviving one head update requires the representational difference to be orthogonal not only to the current head, but also to the representation of the adaptation point. Requiring this to hold across sufficiently many possible updates becomes highly restrictive.

Proposition C.2. Suppose there exist $d + 1$ fine-tuning points $x _ { 0 } , \ldots , x _ { d }$ such that

$$
\mathrm { a f f s p a n } \{ \phi _ { \mathrm { u n l } } ( x _ { 0 } ) , \dots , \phi _ { \mathrm { u n l } } ( x _ { d } ) \} = \mathbb { R } ^ { d } .
$$

Ifthe representational difference remains harmless after nonzero head updates on each ofthese points, then

$$
\phi _ { \mathrm { r e t r } } ( x ) = \phi _ { \mathrm { u n l } } ( x ) \qquad \forall x \in \mathcal { X } .
$$

In particular,

$$
\widetilde { \Phi } _ { \mathrm { r e t r } } = \widetilde { \Phi } _ { \mathrm { u n l } } ,
$$

so the two models have identical model-definedfeature directions and strengths, up to the usual PCA sign and eigenspace ambiguities.

Thus, when the representation has full affine dimension, no nontrivial mismatch can remain harmless under sufficiently rich head adaptation. Importantly, full affine dimension is not an exceptional condition: it can be established for a broad class of feed-forward networks.

To make this explicit, consider a representation $\phi : \mathbb { R } ^ { p }  \mathbb { R } ^ { d }$ defined by

$$
h _ { 0 } ( x ) = x , \qquad h _ { \ell } ( x ) = \sigma _ { \ell } ( W _ { \ell } h _ { \ell - 1 } ( x ) + b _ { \ell } ) , \quad W _ { \ell } \in \mathbb { R } ^ { m _ { \ell } \times m _ { \ell - 1 } } , \quad \ell = 1 , \dots , L - 1 ,
$$

and

$$
\phi ( x ) = W _ { L } h _ { L - 1 } ( x ) + b _ { L } , \qquad W _ { L } \in \mathbb { R } ^ { d \times m _ { L - 1 } } ,
$$

with

$$
p = m _ { 0 } \geq m _ { 1 } \geq \cdot \cdot \cdot \geq m _ { L - 1 } \geq d .
$$

We consider smooth activations with everywhere nonvanishing derivative, such as tanh, sigmoid, and softplus, together with Leaky-ReLU with nonzero negative slope.

Proposition C.3. For the network above, suppose that every weight matrix has full rank. Then there exist d + 1 inputs $x _ { 0 } , \ldots , x _ { d }$ such that

$$
\mathrm { a f f s p a n } \{ \phi ( x _ { 0 } ) , \ldots , \phi ( x _ { d } ) \} = \mathbb { R } ^ { d } .
$$

This provides a simple example in which harmlessness under the current head is maximally unstable: representational differences may be invisible before adaptation, yet requiring them to remain invisible across head updates forces the representations themselves to coincide.

Full affine dimension is not required to obtain a meaningful restriction. Even when nontrivial harmless directions remain, persistent harmlessness constrains the strengths of otherwise preserved features.

Proposition C.4 (Strength of preserved features). Suppose that the retrained and unlearned models have the same normalized response directions $u _ { i } ,$ and that their representational difference remains harmless under the nonzero head updates on every $x \in { \mathcal { X } } .$ Then, for every corresponding feature,

$$
\lambda _ { i } ^ { \mathrm { r e t r } } \geq \lambda _ { i } ^ { \mathrm { u n l } } .
$$

Taken together, these results show that equivalence under a fixed head can hide substantial representational mismatch, but this equivalence is generally unstable under subsequent adaptation.

## C.3 PROOFS FOR HARMLESSNESS UNDER A SHARED HEAD

## C.3.1 PROOF OF LEMMA C.1

Proof. The argument is identical to the one-step calculation in the proof of Proposition 3.1. Using the initial harmlessness condition, after the update we obtain

$$
( W _ { \mathrm { u n l } } ^ { + } ) ^ { \top } \big ( \phi _ { \mathrm { r e t r } } ( x ) - \phi _ { \mathrm { u n l } } ( x ) \big ) = - \eta g _ { \mathrm { f t } } \phi _ { \mathrm { u n l } } ( x _ { \mathrm { f t } } ) ^ { \top } \big ( \phi _ { \mathrm { r e t r } } ( x ) - \phi _ { \mathrm { u n l } } ( x ) \big ) .
$$

Lemma C.5. Let   
ϕ : R<sup>p</sup> → R<sup>d</sup>, p ≥ d,   
be continuously differentiable in a neighborhood of $x ^ { \star }$ , and suppose   
rank J<sub>ϕ</sub>(x<sup>⋆</sup>) = d.   
Then there exists a neighborhood U of $\boldsymbol { \cdot } \boldsymbol { x } ^ { \star }$ such that $\phi ( U )$ contains an open neighborhood of   
$\phi ( x ^ { \star } )$ in $\mathbb { R } ^ { d }$

Since $g _ { \mathrm { f t } } \neq 0$ and the remaining factor is scalar, the left-hand side vanishes for all x if and only if the stated orthogonality condition holds.

## C.3.2 PROOF OF PROPOSITION C.2

Proof. By Lemma C.1,

$$
\bigl ( \phi _ { \mathrm { r e t r } } ( x ) - \phi _ { \mathrm { u n l } } ( x ) \bigr ) ^ { \top } \phi _ { \mathrm { u n l } } ( x _ { i } ) = 0 \qquad \forall x , i = 0 , \ldots , d .
$$

Since $\{ \phi _ { \mathrm { u n l } } ( x _ { i } ) \} _ { i = 0 } ^ { d }$ affinely spans $\mathbb { R } ^ { d } .$ , it also linearly spans $\mathbb { R } ^ { d } .$ . Hence $\phi _ { \mathrm { r e t r } } ( x ) - \phi _ { \mathrm { u n l } } ( x ) =$ 0 for every x. The equality $\widetilde \Phi _ { \mathrm { r e t r } } = \widetilde \Phi _ { \mathrm { u n l } }$ and the stated feature-level consequences follow immediately.

## C.3.3 PROOF OF PROPOSITION C.3

Proof. Since $J _ { \phi } ( x ^ { \star } )$ has rank $d ,$ some $d \times$ d minor is nonzero at $x ^ { \star }$ . By continuity, this minor remains nonzero in a sufficiently small neighborhood $U$ of $x ^ { \star }$ . Since d is the dimension of the codomain,

The Constant Rank Theorem (Zorich, 2016) therefore applies on $U .$ Since the rank equals the dimension of the codomain, there exist local coordinates in which ϕ has the form

$$
( u _ { 1 } , \ldots , u _ { p } ) \longmapsto ( u _ { 1 } , \ldots , u _ { d } ) .
$$

Hence $\phi ( U )$ contains an open neighborhood of $\phi ( x ^ { \star } )$ in $\mathbb { R } ^ { d } .$

Lemma C.6. Under the assumptions of Proposition C.3, if σ is Leaky ReLU with nonzero negative slope, then there exists $\bar { \boldsymbol { x } } ^ { \star } \in \mathbb { R } ^ { \bar { p } }$ such that no hidden pre-activation vanishes at $x ^ { \star }$ , ϕ is smooth in a neighborhood of $\cdot _ { x ^ { \star } }$ , and

$$
\operatorname { r a n k } J _ { \phi } ( x ^ { \star } ) = d .
$$

Proof. We first construct a point away from all activation boundaries. For the first layer, each pre-activation

$$
z _ { 1 , j } ( x ) = w _ { 1 , j } ^ { \top } x + b _ { 1 , j }
$$

is a nonzero affine function, so its zero set is a proper hyperplane. Hence there exists a nonempty open set $U _ { 1 }$ on which all first-layer pre-activations have fixed nonzero signs.

Suppose we have a nonempty open set $U _ { \ell - 1 }$ on which all pre-activations in the first $\ell - 1$ layers are nonzero with fixed signs. Then $h _ { \ell - 1 }$ , and hence every $z _ { \ell , j }$ , is affine on $U _ { \ell - 1 }$ . Under the full-rank assumption, none of these pre-activations is identically zero there, so their zero sets

cannot cover $U _ { \ell - 1 }$ . Thus there exists a nonempty open subset

$$
U _ { \ell } \subset U _ { \ell - 1 }
$$

on which all layer-ℓ pre-activations have fixed nonzero signs.

Iterating through the hidden layers yields a nonempty open set $U$ on which the entire activation pattern is fixed. Hence the network is affine on $U ,$ , with

$$
J _ { \phi } ( x ) = W _ { L } D _ { L - 1 } W _ { L - 1 } \cdot \cdot \cdot D _ { 1 } W _ { 1 } , \qquad x \in U ,
$$

where each $D _ { \ell }$ is diagonal with entries in $\{ 1 , \alpha \}$ and is therefore invertible.

Under the full-rank and width decreasing assumptions of Proposition C.3, this product has rank $d .$ Thus any $x ^ { \star } \in U$ is away from all activation boundaries, $\phi$ is smooth in a neighborhood of $x ^ { \star }$ , and

$$
\operatorname { r a n k } J _ { \phi } ( x ^ { \star } ) = d .
$$

Proof of Proposition C.3. By Lemma C.5, it suffices to exhibit a point $x ^ { \star }$ at which ϕ is smooth and

$$
\operatorname { r a n k } J _ { \phi } ( x ^ { \star } ) = d .
$$

For smooth activations with nonvanishing derivative, such a point exists generically since all activation-derivative matrices are nonsingular. For Leaky-ReLU networks, the same conclusion holds at a point away from all activation boundaries; the existence of such a point with full-rank Jacobian is established in Lemma C.6. □

## C.3.4 PROOF OF PROPOSITION C.4

Proof. Let

$$
D : = \widetilde { \Phi } _ { \mathrm { u n l } } - \widetilde { \Phi } _ { \mathrm { r e t r } } .
$$

By Lemma C.1, applied to all considered update points and then centered,

$$
D \widetilde { \Phi } _ { \mathrm { u n l } } ^ { \top } = 0 .
$$

Since

$$
\begin{array} { r } { \widetilde { \Phi } _ { \mathrm { r e t r } } = \widetilde { \Phi } _ { \mathrm { u n l } } - D , } \end{array}
$$

we obtain

$$
\begin{array} { r } { \widetilde { \Phi } _ { \mathrm { r e t r } } \widetilde { \Phi } _ { \mathrm { r e t r } } ^ { \top } = ( \widetilde { \Phi } _ { \mathrm { u n l } } - D ) ( \widetilde { \Phi } _ { \mathrm { u n l } } - D ) ^ { \top } = \widetilde { \Phi } _ { \mathrm { u n l } } \widetilde { \Phi } _ { \mathrm { u n l } } ^ { \top } + D D ^ { \top } . } \end{array}
$$

Hence

$$
\widetilde { \Phi } _ { \mathrm { r e t r } } \widetilde { \Phi } _ { \mathrm { r e t r } } ^ { \top } - \widetilde { \Phi } _ { \mathrm { u n l } } \widetilde { \Phi } _ { \mathrm { u n l } } ^ { \top } = D D ^ { \top } \succeq 0 .
$$

Because the two models have the same normalized response directions $u _ { i }$

$$
\widetilde { \Phi } _ { \mathrm { r e t r } } \widetilde { \Phi } _ { \mathrm { r e t r } } ^ { \top } - \widetilde { \Phi } _ { \mathrm { u n l } } \widetilde { \Phi } _ { \mathrm { u n l } } ^ { \top } = \sum _ { i } \big ( \lambda _ { i } ^ { \mathrm { r e t r } } - \lambda _ { i } ^ { \mathrm { u n l } } \big ) u _ { i } u _ { i } ^ { \top } .
$$

Positive semidefiniteness therefore implies

$$
\lambda _ { i } ^ { \mathrm { r e t r } } \geq \lambda _ { i } ^ { \mathrm { u n l } }
$$

for every corresponding feature.

## D XOR THEORY

The model is trained for binary classification with logistic loss

$$
\begin{array} { r l r l r } { \mathcal { L } _ { S } ( \theta ) = \mathbb { E } _ { ( x , y ) \sim S } \left[ \ell \big ( m _ { \theta } ( x , y ) \big ) \right] , } & { { } } & { m _ { \theta } ( x , y ) : = y f _ { \theta } ( x ) , } & { { } } & { \ell ( m ) = \log ( 1 + e ^ { - m } ) , } \end{array}
$$

where $S = D$ for full-data training and $S = D _ { r }$ for retraining. Writing

$$
h ( m ) : = - \ell ^ { \prime } ( m ) = \frac { 1 } { 1 + e ^ { m } } ,
$$

full-batch gradient flow,

$$
\dot { \theta } _ { S } ( t ) = - \nabla _ { \theta } \mathcal { L } _ { S } \big ( \theta _ { S } ( t ) \big ) , \qquad \theta _ { S } ( 0 ) = \theta _ { 0 } ,
$$

satisfies

$$
\dot { a } _ { S } ( t ) = \mathbb { E } _ { ( x , y ) \sim S } \left[ h \big ( m _ { \theta _ { S } ( t ) } ( x , y ) \big ) y s \right] ,\tag{5}
$$

$$
\dot { u } ( t ) = \mathbb { E } _ { ( x , y ) \sim S } \left[ h \big ( m _ { \theta _ { S } ( t ) } ( x , y ) \big ) y \big ( u ( t ) ^ { \top } \widetilde { z } \big ) \widetilde { z } \right] ,\tag{6}
$$

$$
\dot { v } ( t ) = - \mathbb { E } _ { ( x , y ) \sim S } \left[ h \big ( m _ { \theta _ { S } ( t ) } ( x , y ) \big ) y \big ( v ( t ) ^ { \top } \widetilde { z } \big ) \widetilde { z } \right] .\tag{7}
$$

## D.1 CONNECTION TO MODEL FEATURES

We take the activations immediately before the fixed linear output layer as the model representation,

$$
\phi _ { \theta } ( x ) = \left( \begin{array} { c } { { a s } } \\ { { ( u ^ { \top } \widetilde { z } ) ^ { 2 } } } \\ { { ( v ^ { \top } \widetilde { z } ) ^ { 2 } } } \end{array} \right) .
$$

Let $\xi : = \chi ( z ) = z _ { 1 } z _ { 2 } \in \{ - 1 , 1 \}$ . Since the representation depends only on whether an example is retained or forgotten and on ξ, it takes four distinct values. Collecting them as rows gives

$$
\Phi = \left( \begin{array} { c c c } { { a c } } & { { \varepsilon ( u _ { 1 } + u _ { 2 } ) ^ { 2 } } } & { { \varepsilon ( v _ { 1 } + v _ { 2 } ) ^ { 2 } } } \\ { { - a c } } & { { \varepsilon ( u _ { 1 } - u _ { 2 } ) ^ { 2 } } } & { { \varepsilon ( v _ { 1 } - v _ { 2 } ) ^ { 2 } } } \\ { { - a c } } & { { ( 1 - \varepsilon ) ( u _ { 1 } + u _ { 2 } ) ^ { 2 } } } & { { ( 1 - \varepsilon ) ( v _ { 1 } + v _ { 2 } ) ^ { 2 } } } \\ { { a c } } & { { ( 1 - \varepsilon ) ( u _ { 1 } - u _ { 2 } ) ^ { 2 } } } & { { ( 1 - \varepsilon ) ( v _ { 1 } - v _ { 2 } ) ^ { 2 } } } \end{array} \right) ,
$$

corresponding to

$$
( D _ { r } , \xi = 1 ) , \qquad ( D _ { r } , \xi = - 1 ) , \qquad ( D _ { f } , \xi = 1 ) , \qquad ( D _ { f } , \xi = - 1 ) ,
$$

with probabilities

$$
\pi = { \frac { 1 } { 2 } } \left( { 1 - \varepsilon \atop \varepsilon } \right) .
$$

We characterize the learned feature solution by PCA of the centered, π-weighted representation. Since $\phi _ { \theta } ( x ) \in \mathbb { R } ^ { 3 }$ , there are at most three nonzero principal components. Each nonzero PCA direction induces a normalized response vector across the data, which is the quantity used to define model features in Section 2. The resulting principal directions and their ordering by eigenvalue depend on the particular model parameters θ. We analyze these directions separately for each training and unlearning trajectory below.

## D.2 PROOF ROADMAP

Although the derivations below are somewhat lengthy, they follow the same simple structure across all training and unlearning settings. For each trajectory—full-data training, retraining, random-label unlearning, random-label unlearning combined with fine-tuning, and gradient ascent—we first identify an invariant of the corresponding gradient-flow dynamics. In particular, retraining and all three unlearning trajectories preserve the same relation $u = J v$ , for a matrix J specified below. We then use the invariant to reduce the parameter dynamics and determine the resulting coordinates of the feature representation. Finally, we compute the eigenvalues and eigenvectors of the corresponding centered covariance matrix, which gives the model-defined feature directions and strengths. Thus, the proofs differ mainly in the trajectory-specific algebra needed to establish the invariant and its limiting parameter values; the subsequent feature calculation is essentially the same.

## D.3 FULL-DATA TRAINING DYNAMICS

We begin with the initial training stage, before any data are deleted, and characterize gradient flow on the full dataset $D .$ . The chosen scaling makes the retain and forget contributions to the quadratic branch cancel, so training changes only the shortcut parameter.

For later use, define

$$
K : = \binom { 0 } { 1 } \ \stackrel { 1 } { 0 } \Big ) .
$$

Since $\chi ( z ) = z _ { 1 } z _ { 2 }$ and $z _ { i } ^ { 2 } = 1$

$$
\mathbb { E } _ { z } \big [ \chi ( z ) z z ^ { \top } \big ] = K ,
$$

and therefore

$$
\mathbb { E } _ { D _ { r } } \big [ y \widetilde { z } \widetilde { z } ^ { \top } \big ] = \varepsilon K , \qquad \mathbb { E } _ { D _ { f } } \big [ y \widetilde { z } \widetilde { z } ^ { \top } \big ] = - ( 1 - \varepsilon ) K .
$$

Hence the retain and forget contributions cancel exactly in the full-data gradient,

$$
\mathbb { E } _ { D } \left[ y \widetilde { z } \widetilde { z } ^ { \top } \right] = 0 .\tag{8}
$$

Lemma D.1 (Frozen core under full-data training). Under full-data gradient flow in the exactly balanced setting,

$$
u ( t ) = v ( t ) = \delta e _ { 1 } \qquad f o r a l l t \geq 0 .
$$

Proof. At initialization,

$$
u ( 0 ) = v ( 0 ) = \delta e _ { 1 } .
$$

Suppose that at some point along the trajectory $u = v .$ Then the two quadratic terms cancel, and every training example has margin

$$
m _ { \theta } ( x , y ) = y f _ { \theta } ( x ) = y a s = a c ,
$$

where we used $y s = c$ for both retain and forget examples.

Hence the gradient-flow dynamics of the core parameters reduce to

$$
\dot { u } = h ( a c ) \mathbb { E } _ { D } \left[ y \widetilde { z } \widetilde { z } ^ { \top } \right] u , \quad \dot { v } = - h ( a c ) \mathbb { E } _ { D } \left[ y \widetilde { z } \widetilde { z } ^ { \top } \right] v .
$$

By equation 8,

$$
\mathbb { E } _ { D } \left[ y \widetilde { z z } ^ { \top } \right] = 0 .
$$

Therefore $\dot { u } = \dot { v } = 0$ whenever $u = v$

Thus the trajectory

$$
u ( t ) = v ( t ) = \delta e _ { 1 }
$$

is preserved by the gradient-flow vector field. By uniqueness of the gradient-flow solution, it holds for all $t \geq 0$

□

Lemma D.2. Under full-data gradient flow in the exactly balanced setting, $a ( t )$ is increasing and

$$
a ( t ) \to + \infty \qquad a s t \to \infty .
$$

Proof. By Lemma D.1, the quadratic branch cancels identically, so every example has margin $a ( t ) c .$ Thus

$$
\dot { a } ( t ) = c h ( a ( t ) c ) = \frac { c } { 1 + e ^ { c a ( t ) } } > 0 .
$$

Hence $a ( t )$ is increasing. If it were bounded, then $a ( t ) \leq M$ for some $M ,$ , implying

$$
\dot { a } ( t ) \geq \frac { c } { 1 + e ^ { c M } } > 0 ,
$$

which contradicts boundedness. Therefore

$$
a ( t ) \to + \infty .
$$

Theorem D.3 (Full-data feature solution). Along the full-data training trajectory, the two nonzero normalized feature responses are, up to sign,

$$
u _ { \mathrm { s } } = \frac { 1 } { \sqrt 2 } \left( \begin{array} { c } { { \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { \varepsilon } } } \\ { { \sqrt { \varepsilon } } } \end{array} \right) , \qquad u _ { \mathrm { + } } = \frac { 1 } { \sqrt 2 } \left( \begin{array} { c } { { \sqrt { \varepsilon } } } \\ { { \sqrt { \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \end{array} \right) ,
$$

with corresponding PCA directions

$$
q _ { \mathrm { s } } = { \binom { 1 } { 0 } } , \qquad q _ { + } = { \frac { 1 } { \sqrt { 2 } } } { \binom { 0 } { 1 } } ,
$$

and strengths

$$
\lambda _ { \mathrm { s } } ( t ) = a ( t ) ^ { 2 } c ^ { 2 } , \qquad \lambda _ { + } = 2 \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) ^ { 2 } \delta ^ { 4 } .
$$

The orthogonal PCA direction

$$
q _ { - } = { \frac { 1 } { \sqrt { 2 } } } \left( { \begin{array} { l } { 0 } \\ { 1 } \\ { - 1 } \end{array} } \right)
$$

has eigenvalue $\lambda _ { - } = 0 ,$ , so there is no corresponding normalized feature response.

Proof. Recall the feature matrix from Section D.1. By Lemmas D.2 and D.1,

$$
u ( t ) = v ( t ) = \delta e _ { 1 }
$$

along the entire full-data training trajectory. Hence

$$
u _ { 2 } = v _ { 2 } = 0 , \qquad u _ { 1 } = v _ { 1 } = \delta ,
$$

and the feature matrix reduces to

$$
\Phi ( t ) = \left( \begin{array} { c c c } { { a c } } & { { \varepsilon \delta ^ { 2 } } } & { { \varepsilon \delta ^ { 2 } } } \\ { { - a c } } & { { \varepsilon \delta ^ { 2 } } } & { { \varepsilon \delta ^ { 2 } } } \\ { { - a c } } & { { ( 1 - \varepsilon ) \delta ^ { 2 } } } & { { ( 1 - \varepsilon ) \delta ^ { 2 } } } \\ { { a c } } & { { ( 1 - \varepsilon ) \delta ^ { 2 } } } & { { ( 1 - \varepsilon ) \delta ^ { 2 } } } \end{array} \right) .
$$

Its probability-weighted mean is

$$
\begin{array} { r } { \bar { \phi } ( t ) = \left( 0 \quad 2 \varepsilon ( 1 - \varepsilon ) \delta ^ { 2 } \quad 2 \varepsilon ( 1 - \varepsilon ) \delta ^ { 2 } \right) . } \end{array}
$$

Therefore the centered covariance matrix is

$$
C _ { \mathrm { f u l l } } = \left( \begin{array} { c c c } { { a ( t ) ^ { 2 } c ^ { 2 } } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { \gamma } } & { { \gamma } } \\ { { 0 } } & { { \gamma } } & { { \gamma } } \end{array} \right) , \qquad \gamma = \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) ^ { 2 } \delta ^ { 4 } .
$$

The first coordinate immediately gives $q _ { \mathrm { s } }$ with eigenvalue $a ( t ) ^ { 2 } c ^ { 2 }$ , while the remaining block

$$
\gamma \left( \begin{array} { c c } { { 1 } } & { { 1 } } \\ { { 1 } } & { { 1 } } \end{array} \right)
$$

has normalized eigenvectors $( 1 , 1 ) ^ { \top } / \sqrt { 2 }$ and $( 1 , - 1 ) ^ { \top } / \sqrt { 2 }$ with eigenvalues $2 \gamma$ and $0 ,$ respectively. This yields the stated feature directions and strengths.

For each nonzero component, the corresponding normalized response vector is

$$
u _ { i } = \frac { \mathrm { d i a g } ( \pi ) ^ { 1 / 2 } \widetilde { \Phi } q _ { i } } { \sqrt { \lambda _ { i } } } .
$$

Substituting $q _ { \mathrm { s } }$ and $q _ { + }$ gives the stated $u _ { \mathrm { s } }$ and $u _ { + } ;$ since $\lambda _ { - } = 0$ , no normalized response vector is defined for $q _ { - } .$ □

## D.4 RETAIN-DATA TRAINING DYNAMICS

We next analyze retraining on the retain data after deletion. Unlike full-data training, deletion breaks the cancellation that kept the quadratic branch frozen, so the core parameters now evolve. We first show that the retraining trajectory preserves the symmetry $u = J v$ , which reduces the dynamics to a low-dimensional form. We then characterize the resulting parameter growth and use the same invariant to derive the corresponding model features and their asymptotic ordering.

Lemma D.4 (Symmetry of the retraining trajectory). Consider gradient flow on the retain distribution $D _ { r } ,$ and let

$$
J : = \left( \begin{array} { c c } { { 1 } } & { { 0 } } \\ { { 0 } } & { { - 1 } } \end{array} \right) .
$$

Then

$$
u ( t ) = J v ( t ) \qquad f o r a l l t \geq 0 .
$$

Proof. At initialization,

$$
u ( 0 ) = v ( 0 ) = \delta e _ { 1 } ,
$$

and hence $u ( 0 ) = J v ( 0 )$

We show that the manifold

$$
\mathcal { M } : = \{ ( u , v ) : u = J v \}
$$

is invariant under the retraining dynamics. Suppose $u = J v$ and write

$$
u = { \binom { p } { s } } , \qquad v = { \binom { p } { - s } } .
$$

For a retain example,

$$
\begin{array} { r } { \widetilde { z } = \sqrt { \varepsilon } z , \qquad y = \chi ( z ) = z _ { 1 } z _ { 2 } . } \end{array}
$$

Therefore, denoting the retain-data margin by $m _ { R } ,$

$$
m _ { R } = y \left( a s + \frac { 1 } { 2 } ( u ^ { \top } \widetilde { z } ) ^ { 2 } - \frac { 1 } { 2 } ( v ^ { \top } \widetilde { z } ) ^ { 2 } \right) = a c + 2 \varepsilon p s ,
$$

which is independent of the particular retain example.

Moreover,

$$
\begin{array} { r } { \mathbb { E } _ { D _ { r } } \left[ y \widetilde { z } \widetilde { z } ^ { \top } \right] = \varepsilon K , \qquad K : = \left( \begin{array} { l l } { 0 } & { 1 } \\ { 1 } & { 0 } \end{array} \right) . } \end{array}
$$

Hence, on $\mathcal { M } ,$

Since

$$
\dot { u } = \varepsilon h ( m _ { R } ) K u , \qquad \dot { v } = - \varepsilon h ( m _ { R } ) K v .
$$

$$
J K = - K J ,
$$

we obtain

$$
J \dot { v } = - \varepsilon h ( m _ { R } ) J K v = \varepsilon h ( m _ { R } ) K J v = \varepsilon h ( m _ { R } ) K u = \dot { u } .
$$

Thus the gradient-flow vector field is tangent to M. Since the gradient-flow solution is unique and the initialization lies in $\mathcal { M } .$ , the entire trajectory remains in $\mathcal { M } .$ □

Lemma D.5. Let $p ( t )$ and $s ( t )$ satisfy

$$
\dot { p } ( t ) = \gamma ( t ) s ( t ) , \qquad \dot { s } ( t ) = \gamma ( t ) p ( t ) ,
$$

for some continuous scalar function $\gamma ( t )$ , and define

$$
G ( t ) : = \int _ { 0 } ^ { t } \gamma ( \tau ) d \tau .
$$

Then

$$
p ( t ) + s ( t ) = \bigl ( p ( 0 ) + s ( 0 ) \bigr ) e ^ { G ( t ) } ,
$$

$$
p ( t ) - s ( t ) = \big ( p ( 0 ) - s ( 0 ) \big ) e ^ { - G ( t ) } .
$$

In particular, if

$$
p ( 0 ) = \delta , \qquad s ( 0 ) = 0 ,
$$

then

$$
p ( t ) = \delta \cosh G ( t ) , \qquad s ( t ) = \delta \sinh G ( t ) .
$$

Proof. Adding and subtracting the two differential equations gives

$$
\frac { d } { d t } ( p + s ) = \gamma ( t ) ( p + s ) , \qquad \frac { d } { d t } ( p - s ) = - \gamma ( t ) ( p - s ) .
$$

Integrating these scalar equations yields

$$
p ( t ) + s ( t ) = \bigl ( p ( 0 ) + s ( 0 ) \bigr ) e ^ { G ( t ) }
$$

and

$$
p ( t ) - s ( t ) = \bigl ( p ( 0 ) - s ( 0 ) \bigr ) e ^ { - G ( t ) } .
$$

The stated formulas follow immediately.

Lemma D.6. Under retraining on $D _ { r }$ , define

$$
G ( t ) : = \int _ { 0 } ^ { t } \varepsilon h ( m _ { R } ( \tau ) ) d \tau .
$$

Then

$$
u ( t ) = \delta \left( \cosh G ( t ) \right) ,
$$

$$
v ( t ) = \delta \left( \cosh G ( t ) \right) ,\tag{9}
$$

$$
a ( t ) = a ( 0 ) + \frac { c } { \varepsilon } G ( t ) .\tag{10}
$$

Consequently,

$$
m _ { R } ( t ) = c a ( 0 ) + \frac { c ^ { 2 } } { \varepsilon } G ( t ) + \varepsilon \delta ^ { 2 } \sinh \bigl ( 2 G ( t ) \bigr ) .
$$

Moreover,

$$
G ( t ) \to \infty \qquad a s t \to \infty .
$$

Proof. By Lemma D.4, write

$$
u ( t ) = { \binom { p ( t ) } { s ( t ) } } , \qquad v ( t ) = { \binom { p ( t ) } { - s ( t ) } } .
$$

The retraining dynamics therefore reduce to

$$
\dot { p } = \varepsilon h ( m _ { R } ) s , \qquad \dot { s } = \varepsilon h ( m _ { R } ) p .
$$

Applying Lemma D.5 with $\gamma ( t ) = \varepsilon h ( m _ { R } ( t ) )$ and using $p ( 0 ) = \delta , s ( 0 ) = 0 \mathrm { g i v e s }$ s

$$
p ( t ) = \delta \cosh G ( t ) , \qquad s ( t ) = \delta \sinh G ( t ) ,
$$

which proves equation 9. For the shortcut parameter,

$$
\dot { a } ( t ) = c h ( m _ { R } ( t ) ) = \frac { c } { \varepsilon } \dot { G } ( t ) .
$$

Hence

$$
a ( t ) = a ( 0 ) + \frac { c } { \varepsilon } G ( t ) .
$$

Finally, since

$$
2 p ( t ) s ( t ) = \delta ^ { 2 } \sinh \bigl ( 2 G ( t ) \bigr ) ,
$$

the retain margin becomes

$$
m _ { R } ( t ) = c a ( 0 ) + \frac { c ^ { 2 } } { \varepsilon } G ( t ) + \varepsilon \delta ^ { 2 } \sinh \bigl ( 2 G ( t ) \bigr ) .
$$

It remains to show that $G ( t )$ is unbounded. If $G ( t ) \leq M$ for all $t \geq 0$ , then the expression above implies that $m _ { R } ( t )$ is uniformly bounded above by some finite $M _ { R }$ . Therefore,

$$
\begin{array} { r } { \dot { G } ( t ) = \varepsilon h ( m _ { R } ( t ) ) \geq \varepsilon h ( M _ { R } ) > 0 , } \end{array}
$$

which implies that $G ( t )$ grows at least linearly in t, a contradiction. Hence

$$
G ( t ) \to \infty .
$$

Proposition D.7 (Feature solution under the J-invariant). Suppose that

$$
\boldsymbol { u } = \boldsymbol { J } \boldsymbol { v } , \qquad \boldsymbol { J } = \left( \begin{array} { c c } { 1 } & { 0 } \\ { 0 } & { - 1 } \end{array} \right) , \qquad \boldsymbol { u } = ( u _ { 1 } , u _ { 2 } ) ^ { \top } .
$$

Then the centered representation has PCA directions

$$
q _ { \mathrm { s } } = { \binom { 1 } { 0 } } , \qquad q _ { \mathrm { + } } = { \frac { 1 } { \sqrt { 2 } } } { \binom { 0 } { 1 } } , \qquad q _ { \mathrm { - } } = { \frac { 1 } { \sqrt { 2 } } } { \binom { 0 } { - 1 } } .
$$

For each nonzero component, the corresponding normalized response vector is, up to sign,

$$
u _ { \mathrm { s } } = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { c } { { \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { \varepsilon } } } \\ { { \sqrt { \varepsilon } } } \end{array} \right) , \qquad u _ { \mathrm { + } } = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { c } { { \sqrt { \varepsilon } } } \\ { { \sqrt { \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \end{array} \right) , \qquad u _ { \mathrm { - } } = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { c } { { \sqrt { \varepsilon } } } \\ { { - \sqrt { \varepsilon } } } \\ { { \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \end{array} \right) .
$$

Their corresponding strengths are

$$
\lambda _ { \mathrm { s } } ( t ) = a ( t ) ^ { 2 } c ^ { 2 } , \qquad \lambda _ { + } ( t ) = 2 \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) ^ { 2 } \big ( u _ { 1 } ( t ) ^ { 2 } + u _ { 2 } ( t ) ^ { 2 } \big ) ^ { 2 } ,
$$

and

$$
\lambda _ { - } ( t ) = 8 \varepsilon ( 1 - \varepsilon ) u _ { 1 } ( t ) ^ { 2 } u _ { 2 } ( t ) ^ { 2 } .
$$

Proof. Define

$$
s : = ( u _ { 1 } + u _ { 2 } ) ^ { 2 } , \qquad d : = ( u _ { 1 } - u _ { 2 } ) ^ { 2 } .
$$

Under the J-invariant, the feature matrix from Section D.1 becomes

$$
\Phi = \left( \begin{array} { c c c } { { a c } } & { { \varepsilon s } } & { { \varepsilon d } } \\ { { - a c } } & { { \varepsilon d } } & { { \varepsilon s } } \\ { { - a c } } & { { ( 1 - \varepsilon ) s } } & { { ( 1 - \varepsilon ) d } } \\ { { a c } } & { { ( 1 - \varepsilon ) d } } & { { ( 1 - \varepsilon ) s } } \end{array} \right) .
$$

Its probability-weighted mean is

$$
\bar { \phi } = ( 0 \quad \varepsilon ( 1 - \varepsilon ) ( s + d ) \quad \varepsilon ( 1 - \varepsilon ) ( s + d ) ) .
$$

For convenience, define

$$
A : = \varepsilon s - ( 1 - \varepsilon ) d , \qquad B : = \varepsilon d - ( 1 - \varepsilon ) s .
$$

Then the centered feature matrix is

$$
\widetilde { \Phi } = \left( \begin{array} { c c c } { { a c } } & { { \varepsilon A } } & { { \varepsilon B } } \\ { { - a c } } & { { \varepsilon B } } & { { \varepsilon A } } \\ { { - a c } } & { { - ( 1 - \varepsilon ) B } } & { { - ( 1 - \varepsilon ) A } } \\ { { a c } } & { { - ( 1 - \varepsilon ) A } } & { { - ( 1 - \varepsilon ) B } } \end{array} \right) .
$$

Using the row probabilities π from Section D.1, the centered feature covariance is

$$
C = \widetilde { \Phi } ^ { \top } \mathrm { d i a g } ( \pi ) \widetilde { \Phi } = \left( \begin{array} { c c c } { { a ^ { 2 } c ^ { 2 } } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { \frac { \varepsilon ( 1 - \varepsilon ) } { 2 } ( A ^ { 2 } + B ^ { 2 } ) } } & { { \varepsilon ( 1 - \varepsilon ) A B } } \\ { { 0 } } & { { \varepsilon ( 1 - \varepsilon ) A B } } & { { \frac { \varepsilon ( 1 - \varepsilon ) } { 2 } ( A ^ { 2 } + B ^ { 2 } ) } } \end{array} \right) .
$$

Thus $q _ { \mathrm { s } }$ is an eigenvector with eigenvalue $a ^ { 2 } c ^ { 2 }$ , while the symmetric $2 \times 2$ block has eigenvectors $q _ { + }$ and $q _ { - }$ with eigenvalues

$$
\lambda _ { + } = \frac { \varepsilon ( 1 - \varepsilon ) } { 2 } ( A + B ) ^ { 2 } , \qquad \lambda _ { - } = \frac { \varepsilon ( 1 - \varepsilon ) } { 2 } ( A - B ) ^ { 2 } .
$$

Since

$$
A + B = - ( 1 - 2 \varepsilon ) ( s + d ) , \qquad A - B = s - d ,
$$

and

$$
s + d = 2 ( u _ { 1 } ^ { 2 } + u _ { 2 } ^ { 2 } ) , \qquad s - d = 4 u _ { 1 } u _ { 2 } ,
$$

the stated eigenvalues follow.

For the corresponding feature responses, multiplying the centered representation by the PCA directions and applying the probability weighting gives

$$
\mathrm { d i a g } ( \pi ) ^ { 1 / 2 } \widetilde { \Phi } q _ { \mathrm { s } } = \frac { a c } { \sqrt { 2 } } \left( \begin{array} { c } { { \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { \varepsilon } } } \\ { { \sqrt { \varepsilon } } } \end{array} \right) ,
$$

$$
\mathrm { d i a g } ( \pi ) ^ { 1 / 2 } \widetilde { \Phi } q _ { + } = \frac { A + B } { 2 } \left( \begin{array} { c } { { \varepsilon \sqrt { 1 - \varepsilon } } } \\ { { \varepsilon \sqrt { 1 - \varepsilon } } } \\ { { - ( 1 - \varepsilon ) \sqrt { \varepsilon } } } \\ { { - ( 1 - \varepsilon ) \sqrt { \varepsilon } } } \end{array} \right) ,
$$

and

$$
\mathrm { d i a g } ( \pi ) ^ { 1 / 2 } \widetilde \Phi q _ { - } = \frac { A - B } { 2 } \left( \begin{array} { c } { { \varepsilon \sqrt { 1 - \varepsilon } } } \\ { { - \varepsilon \sqrt { 1 - \varepsilon } } } \\ { { ( 1 - \varepsilon ) \sqrt { \varepsilon } } } \\ { { - ( 1 - \varepsilon ) \sqrt { \varepsilon } } } \end{array} \right) .
$$

Normalizing each nonzero response therefore gives, up to sign,

$$
u _ { \mathrm { s } } = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { c } { { \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { \varepsilon } } } \\ { { \sqrt { \varepsilon } } } \end{array} \right) , \qquad u _ { \mathrm { + } } = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { c } { { \sqrt { \varepsilon } } } \\ { { \sqrt { \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \end{array} \right) , \qquad u _ { \mathrm { - } } = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { c } { { \sqrt { \varepsilon } } } \\ { { - \sqrt { \varepsilon } } } \\ { { \sqrt { 1 - \varepsilon } } } \\ { { - \sqrt { 1 - \varepsilon } } } \end{array} \right) .
$$

Theorem D.8 (Asymptotic ordering of retraining feature directions). Assume $0 < \varepsilon < { \frac { 1 } { 2 } } .$ Along the retraining trajectory, the two core-feature eigenvalues asymptotically dominate the shortcut eigenvalue:

$$
\frac { \lambda _ { \mathrm { s } } ( t ) } { \lambda _ { + } ( t ) } \longrightarrow 0 , \qquad \frac { \lambda _ { \mathrm { s } } ( t ) } { \lambda _ { - } ( t ) } \longrightarrow 0 \qquad a s t \to \infty .
$$

Moreover,

$$
\frac { \lambda _ { + } ( t ) } { \lambda _ { - } ( t ) } \longrightarrow ( 1 - 2 \varepsilon ) ^ { 2 } .
$$

Consequently, for all sufficiently large t,

$$
\lambda _ { - } ( t ) > \lambda _ { + } ( t ) > \lambda _ { \mathrm { s } } ( t ) ,
$$

and the fraction of the total feature variance explained by the shortcut direction vanishes:

$$
\frac { \lambda _ { \mathrm { s } } ( t ) } { \lambda _ { \mathrm { s } } ( t ) + \lambda _ { + } ( t ) + \lambda _ { - } ( t ) } \longrightarrow 0 .
$$

Proof. By Lemma D.6, along the retraining trajectory,

$$
u _ { 1 } ( t ) = \delta \cosh G ( t ) , \qquad u _ { 2 } ( t ) = \delta \sinh G ( t ) , \qquad a ( t ) = \frac { c } { \varepsilon } G ( t ) ,
$$

where

$$
G ( t ) \to \infty \qquad \mathrm { a s } t \to \infty .
$$

Since retraining preserves the invariant $u ( t ) ~ = ~ J v ( t )$ by Lemma D.4, the principal feature directions and their eigenvalues are given directly by Proposition D.7. Substituting these expressions into the eigenvalues gives

$$
\lambda _ { \mathrm { s } } ( t ) = \frac { c ^ { 4 } } { \varepsilon ^ { 2 } } G ( t ) ^ { 2 } .
$$

Moreover,

$$
( u _ { 1 } ^ { 2 } + u _ { 2 } ^ { 2 } ) = \delta ^ { 2 } \cosh \bigl ( 2 G ( t ) \bigr ) \qquad u _ { 1 } u _ { 2 } = \frac { 1 } { 2 } \delta ^ { 2 } \sinh \bigl ( 2 G ( t ) \bigr ) .
$$

Therefore,

$$
\lambda _ { + } ( t ) = 2 \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) ^ { 2 } \delta ^ { 4 } \cosh ^ { 2 } \bigl ( 2 G ( t ) \bigr )
$$

and

$$
\lambda _ { - } ( t ) = 2 \varepsilon ( 1 - \varepsilon ) \delta ^ { 4 } \sinh ^ { 2 } \bigl ( 2 G ( t ) \bigr ) .
$$

Since $G ( t )  \infty ,$ , both $\cosh ^ { 2 } ( 2 G ( t ) )$ and $\sinh ^ { 2 } ( 2 G ( t ) )$ grow exponentially in $G ( t )$ , whereas $\lambda _ { \mathrm { s } } ( t )$ grows only quadratically. Hence

$$
\frac { \lambda _ { \mathrm { s } } ( t ) } { \lambda _ { + } ( t ) }  0 , \qquad \frac { \lambda _ { \mathrm { s } } ( t ) } { \lambda _ { - } ( t ) }  0 .
$$

Furthermore,

$$
\frac { \lambda _ { + } ( t ) } { \lambda _ { - } ( t ) } = ( 1 - 2 \varepsilon ) ^ { 2 } \coth ^ { 2 } \bigl ( 2 G ( t ) \bigr ) \longrightarrow ( 1 - 2 \varepsilon ) ^ { 2 } .
$$

For $0 < \varepsilon < 1 / 2 ,$

$$
( 1 - 2 \varepsilon ) ^ { 2 } < 1 ,
$$

and therefore, for all sufficiently large t,

$$
\lambda _ { - } ( t ) > \lambda _ { + } ( t ) > \lambda _ { \mathrm { s } } ( t ) .
$$

The vanishing fraction of variance explained by the shortcut direction follows immediately from these asymptotic relations. □

## D.5 RANDOM LABELS UNLEARNING DYNAMICS

We next analyze Random Labels (RL) unlearning without retain fine-tuning, initialized from the full-data solution. By Lemma D.1,

$$
a ( 0 ) = A , \qquad u ( 0 ) = v ( 0 ) = \delta e _ { 1 } ,
$$

where $A > 0$ is the shortcut coefficient at the start of unlearning.

RL replaces the forget labels by independent random labels

$$
{ \hat { y } } \sim \operatorname { U n i f } \{ - 1 , 1 \} ,
$$

and performs gradient flow on

$$
\mathcal { L } _ { \mathrm { R L } } ( \theta ) = \mathbb { E } _ { ( x , y ) \sim D _ { f } } \mathbb { E } _ { \hat { y } } \left[ \ell \big ( \hat { y } f _ { \theta } ( x ) \big ) \right] .
$$

The corresponding dynamics are

$$
\begin{array} { r } { \dot { a } ( t ) = \mathbb { E } _ { ( x , y ) \sim D _ { f } } \mathbb { E } _ { \hat { y } } \left[ h \big ( \hat { y } f _ { \theta } ( x ) \big ) \hat { y } s \right] , } \end{array}\tag{11}
$$

$$
\begin{array} { r } { \dot { u } ( t ) = \mathbb { E } _ { ( x , y ) \sim D _ { f } } \mathbb { E } _ { \hat { y } } \left[ h \big ( \hat { y } f _ { \theta } ( x ) \big ) \hat { y } \big ( u ( t ) ^ { \top } \tilde { z } \big ) \tilde { z } \right] , } \end{array}\tag{12}
$$

$$
\begin{array} { r } { \dot { v } ( t ) = - \mathbb { E } _ { ( x , y ) \sim D _ { f } } \mathbb { E } _ { \hat { y } } \left[ h \big ( \hat { y } f _ { \theta } ( x ) \big ) \hat { y } \big ( v ( t ) ^ { \top } \widetilde { z } \big ) \widetilde { z } \right] . } \end{array}\tag{13}
$$

Since $\hat { y }$ is uniform, define

$$
\psi ( r ) : = \mathbb { E } _ { \hat { y } } \left[ \hat { y } h ( \hat { y } r ) \right] = \frac { 1 } { 2 } \big ( h ( r ) - h ( - r ) \big ) = - \frac { 1 } { 2 } \operatorname { t a n h } \left( \frac { r } { 2 } \right) .
$$

Averaging over the random labels gives

$$
\begin{array} { r } { \dot { a } ( t ) = \mathbb { E } _ { ( x , y ) \sim D _ { f } } \left[ \psi \big ( f _ { \theta } ( x ) \big ) s \right] , } \end{array}\tag{14}
$$

$$
\dot { u } ( t ) = \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) \sim D _ { f } } \left[ \psi \big ( f _ { \theta } ( \boldsymbol { x } ) \big ) \big ( u ( t ) ^ { \top } \widetilde { \boldsymbol { z } } \big ) \widetilde { \boldsymbol { z } } \right] ,\tag{15}
$$

$$
\begin{array} { r } { \dot { v } ( t ) = - \mathbb { E } _ { ( x , y ) \sim D _ { f } } \left[ \psi \big ( f _ { \theta } ( x ) \big ) \big ( v ( t ) ^ { \top } \widetilde { z } \big ) \widetilde { z } \right] . } \end{array}\tag{16}
$$

As in retraining, the RL trajectory preserves the invariant $u ( t ) ~ = ~ J v ( t )$ . We use this symmetry to reduce the dynamics, characterize their asymptotic behavior, and derive the resulting principal feature directions and strengths.

Lemma D.9 (Symmetry of the Random Labels trajectory). Consider Random Labels gradient flow initializedfrom

$$
a ( 0 ) = A , \qquad u ( 0 ) = v ( 0 ) = \delta e _ { 1 } .
$$

Let

$$
J = \left( \begin{array} { c c } { { 1 } } & { { 0 } } \\ { { 0 } } & { { - 1 } } \end{array} \right) .
$$

Then the relation

$$
u ( t ) = J v ( t )
$$

is preserved along the entire Random Labels trajectory.

Proof. The initialization satisfies $u ( 0 ) = J v ( 0 )$ . As in Lemma D.4, we show that the manifold

$$
\mathcal { M } : = \{ ( u , v ) : u = J v \}
$$

is invariant under the Random Labels dynamics. Suppose $u = J v$ and write

$$
u = { \binom { p } { s } } , \qquad v = { \binom { p } { - s } } .
$$

For a forget example,

$$
f _ { \theta } ( x ) = - a \chi ( z ) c + \frac { 1 - \varepsilon } { 2 } \left[ ( p z _ { 1 } + s z _ { 2 } ) ^ { 2 } - ( p z _ { 1 } - s z _ { 2 } ) ^ { 2 } \right] = \chi ( z ) \big ( - a c + 2 ( 1 - \varepsilon ) p s \big ) .
$$

Since ψ is odd,

$$
\psi { \big ( } f _ { \theta } ( x ) { \big ) } = \chi ( z ) \psi { \big ( } - a c + 2 ( 1 - \varepsilon ) p s { \big ) } .
$$

Then

$$
\begin{array} { r l } & { \dot { u } ( t ) = \mathbb { E } _ { ( x , y ) \sim D _ { f } } \left[ \psi \big ( f _ { \theta } ( x ) \big ) \big ( u ( t ) ^ { \top } \tilde { z } \big ) \tilde { z } \right] } \\ & { \quad \quad = \psi \big ( - a c + 2 ( 1 - \varepsilon ) p s \big ) \mathbb { E } _ { ( x , y ) \sim D _ { f } } \left[ \chi ( z ) \widetilde { z z } ^ { \top } \right] u ( t ) } \\ & { \quad \quad = ( 1 - \varepsilon ) \psi \big ( - a c + 2 ( 1 - \varepsilon ) p s \big ) K u ( t ) , } \end{array}
$$

where

$$
\mathbb { E } _ { ( x , y ) \sim D _ { f } } \left[ \chi ( z ) \widetilde { z } \widetilde { z } ^ { \top } \right] = ( 1 - \varepsilon ) K .
$$

Similarly,

$$
\dot { v } ( t ) = - ( 1 - \varepsilon ) \psi \bigl ( - a c + 2 ( 1 - \varepsilon ) p s \bigr ) K v ( t ) .
$$

Since $J K = - K J ,$ , the same calculation as in Lemma D.4 gives

$$
J { \dot { v } } = { \dot { u } } .
$$

Thus the gradient-flow vector field is tangent to M. Since the initialization lies in M and the solution is unique, the entire Random Labels trajectory remains in $\mathcal { M }$ □

Lemma D.10. Along the Random Labels trajectory, define

$$
G _ { \mathrm { R L } } ( t ) : = \int _ { 0 } ^ { t } ( 1 - \varepsilon ) \psi { \bigl ( } - a ( \tau ) c + 2 ( 1 - \varepsilon ) p ( \tau ) s ( \tau ) { \bigr ) } d \tau .
$$

Then

$$
u _ { 1 } ( t ) = \delta \cosh G _ { \mathrm { R L } } ( t ) , \qquad u _ { 2 } ( t ) = \delta \sinh G _ { \mathrm { R L } } ( t ) , \qquad a ( t ) = A - { \frac { c } { 1 - \varepsilon } } G _ { \mathrm { R L } } ( t ) .
$$

Proof. From the proof of Lemma D.9, the dynamics on the invariant manifold $u = J v$ satisfy

$$
\dot { p } = ( 1 - \varepsilon ) \psi \bigl ( - a c + 2 ( 1 - \varepsilon ) p s \bigr ) s , \qquad \dot { s } = ( 1 - \varepsilon ) \psi \bigl ( - a c + 2 ( 1 - \varepsilon ) p s \bigr ) p .
$$

Define

$$
G _ { \mathrm { R L } } ( t ) : = \int _ { 0 } ^ { t } ( 1 - \varepsilon ) \psi { \bigl ( } - a ( \tau ) c + 2 ( 1 - \varepsilon ) p ( \tau ) s ( \tau ) { \bigr ) } d \tau .
$$

Applying Lemma D.5 and using $p ( 0 ) = \delta$ and $s ( 0 ) = 0 ,$ , we obtain

$$
\begin{array} { r } { p ( t ) = \delta \cosh G _ { \mathrm { R L } } ( t ) , \qquad s ( t ) = \delta \sinh G _ { \mathrm { R L } } ( t ) . } \end{array}
$$

For the shortcut parameter,

$$
\dot { a } = - c \psi \bigl ( - a c + 2 ( 1 - \varepsilon ) p s \bigr ) = - \frac { c } { 1 - \varepsilon } \dot { G } _ { \mathrm { R L } } .
$$

Since $a ( 0 ) = A$ and $G _ { \mathrm { R L } } ( 0 ) = 0 .$

$$
a ( t ) = A - \frac { c } { 1 - \varepsilon } G _ { \mathrm { R L } } ( t ) .
$$

Lemma D.11 (Convergence of the Random Labels dynamics). Along the Random Labels trajectory,

$$
G _ { \mathrm { R L } } ( t ) \longrightarrow G _ { \star } \qquad a s t \to \infty ,
$$

where $G _ { \star } > 0$ is the unique solution of

$$
- A c + \frac { c ^ { 2 } } { 1 - \varepsilon } G _ { \star } + ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G _ { \star } ) = 0 .
$$

Equivalently, the limiting parameters satisfy

$$
a _ { \star } c = ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G _ { \star } ) ,
$$

where

$$
a _ { \star } = A - { \frac { c } { 1 - \varepsilon } } G _ { \star } .
$$

Proof. By Lemma D.10,

$$
\dot { G } _ { \mathrm { R L } } ( t ) = ( 1 - \varepsilon ) \psi \big ( \rho ( G _ { \mathrm { R L } } ( t ) ) \big ) ,
$$

where

$$
\rho ( G ) : = - A c + \frac { c ^ { 2 } } { 1 - \varepsilon } G + ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G ) .
$$

The function $\rho$ is strictly increasing, since

$$
\rho ^ { \prime } ( G ) = \frac { c ^ { 2 } } { 1 - \varepsilon } + 2 ( 1 - \varepsilon ) \delta ^ { 2 } \cosh ( 2 G ) > 0 .
$$

Moreover,

$$
\rho ( 0 ) = - A c < 0 , \qquad \rho ( G ) \to \infty \quad \mathrm { a s } G \to \infty .
$$

Hence, there exists a unique $G _ { \star } > 0$ such that

$$
\rho ( G _ { \star } ) = 0 .
$$

Because $\psi ( r ) > 0$ for $r < 0$ and $\psi ( r ) < 0$ for $r > 0 ,$

$$
\dot { G } _ { \mathrm { R L } } ( t ) > 0 \mathrm { f o r } G _ { \mathrm { R L } } ( t ) < G _ { \star } , \qquad \dot { G } _ { \mathrm { R L } } ( t ) < 0 \mathrm { f o r } G _ { \mathrm { R L } } ( t ) > G _ { \star } .
$$

Since $G _ { \mathrm { R L } } ( 0 ) = 0 < G _ { \star }$ , uniqueness prevents the trajectory from crossing the equilibrium $G _ { \star }$ Hence $G _ { \mathrm { R L } } ( t )$ is increasing and bounded above by $G _ { \star } ,$ , and therefore converges. Its limit must be a fixed point of the dynamics, so by uniqueness it equals $G _ { \star }$

Finally, $\rho ( \bar { G } _ { \star } ) = 0$ together with

$$
a _ { \star } = A - { \frac { c } { 1 - \varepsilon } } G _ { \star }
$$

gives

$$
a _ { \star } c = ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G _ { \star } ) .
$$

Theorem D.12 (Feature spectrum along the Random Labels trajectory). Along the Random Labels trajectory, the normalizedfeature responses and corresponding PCA directions are those of Proposition D.7. Their strengths are

$$
\lambda _ { \mathrm { s } } ( t ) = a ( t ) ^ { 2 } c ^ { 2 } ,
$$

$$
\lambda _ { + } ( t ) = 2 \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) ^ { 2 } \delta ^ { 4 } \cosh ^ { 2 } \bigl ( 2 G _ { \mathrm { R L } } ( t ) \bigr ) ,
$$

and

$$
\lambda _ { - } ( t ) = 2 \varepsilon ( 1 - \varepsilon ) \delta ^ { 4 } \sinh ^ { 2 } \bigl ( 2 G _ { \mathrm { R L } } ( t ) \bigr ) .
$$

Moreover, for $t > 0 ,$

$$
\frac { \lambda _ { \mathrm { s } } ( t ) } { \lambda _ { - } ( t ) } > \frac { 1 - \varepsilon } { 2 \varepsilon } , \qquad \frac { \lambda _ { \mathrm { s } } ( t ) } { \lambda _ { - } ( t ) } \longrightarrow \frac { 1 - \varepsilon } { 2 \varepsilon }
$$

monotonically from above as $t  \infty .$

Proof. By Lemma D.9, the Random Labels trajectory satisfies $u ( t ) = J v ( t )$ , so the principal directions and their eigenvalues are given by Proposition D.7. By Lemma D.10,

$$
\begin{array} { r } { p ( t ) = \delta \cosh G _ { \mathrm { R L } } ( t ) , \qquad s ( t ) = \delta \sinh G _ { \mathrm { R L } } ( t ) , } \end{array}
$$

and hence

$$
p ( t ) ^ { 2 } + s ( t ) ^ { 2 } = \delta ^ { 2 } \cosh \bigl ( 2 G _ { \mathrm { R L } } ( t ) \bigr ) , \qquad p ( t ) s ( t ) = \frac { \delta ^ { 2 } } { 2 } \sinh \bigl ( 2 G _ { \mathrm { R L } } ( t ) \bigr ) .
$$

Substituting these expressions into Proposition D.7 gives the stated eigenvalues.

For the shortcut-to-core ratio,

$$
\frac { \lambda _ { \mathrm { s } } ( t ) } { \lambda _ { - } ( t ) } = \frac { 1 - \varepsilon } { 2 \varepsilon } \left( \frac { a ( t ) c } { ( 1 - \varepsilon ) \delta ^ { 2 } \sinh \bigl ( 2 G _ { \mathrm { R L } } ( t ) \bigr ) } \right) ^ { 2 } .
$$

By Lemma D.11, $G _ { \mathrm { R L } } ( t )$ increases to $G _ { \star }$ . For $G _ { \mathrm { R L } } ( t ) < G _ { \star }$ ,

$$
\rho \big ( G _ { \mathrm { R L } } ( t ) \big ) = - a ( t ) c + ( 1 - \varepsilon ) \delta ^ { 2 } \sinh \big ( 2 G _ { \mathrm { R L } } ( t ) \big ) < 0 ,
$$

so

$$
a ( t ) c > ( 1 - \varepsilon ) \delta ^ { 2 } \sinh \bigl ( 2 G _ { \mathrm { R L } } ( t ) \bigr ) .
$$

Thus, for every finite $t > 0$

$$
\frac { \lambda _ { \mathrm { s } } ( t ) } { \lambda _ { - } ( t ) } > \frac { 1 - \varepsilon } { 2 \varepsilon } .
$$

Moreover, $a ( t )$ is strictly decreasing while $G _ { \mathrm { R L } } ( t )$ , and hence sinh $\cdot ( 2 G _ { \mathrm { R L } } ( t ) )$ , is strictly increasing, so the ratio is strictly decreasing. Finally,

$$
a _ { \star } c = ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G _ { \star } )
$$

by Lemma D.11, which gives the stated limit.

Corollary D.13 (Ordering of the core feature directions under Random Labels). Along the Random Labels trajectory,

$$
\frac { \lambda _ { + } ( t ) } { \lambda _ { - } ( t ) } = ( 1 - 2 \varepsilon ) ^ { 2 } \coth ^ { 2 } \bigl ( 2 G _ { \mathrm { R L } } ( t ) \bigr ) ,
$$

and this ratio is strictly decreasing for $t > 0 .$

In particular, at the beginning of unlearning,

$$
\lambda _ { + } ( 0 ) > \lambda _ { - } ( 0 ) = 0 .
$$

The two eigenvalues cross at most once, namely when

$$
\operatorname { t a n h } \left( 2 G _ { \mathrm { { R L } } } ( t ) \right) = 1 - 2 \varepsilon .
$$

Proof. The ratio follows directly from the expressions for $\lambda _ { + }$ and $\lambda _ { - }$ . Since $G _ { \mathrm { R L } } ( t )$ is strictly increasing and coth(x) is strictly decreasing for $x > 0$ , the ratio is strictly decreasing. At $t = 0 .$ , we have $G _ { \mathrm { R L } } \dot { ( } 0 \dot { ) } = 0 , { \bf s o } \dot { \lambda } _ { - } ( 0 ) = 0$ , while $\lambda _ { + } ( 0 ) > 0$ for $0 < \varepsilon < \dot { 1 } / 2$ . The crossing condition follows by setting $\lambda _ { + } = \lambda _ { - }$ □

## D.6 RANDOM LABELS WITH RETAIN FINE-TUNING UNLEARNING DYNAMICS

We next consider Random Labels unlearning with simultaneous retain fine-tuning. The initialization and random relabeling of the forget set are unchanged; in addition, we minimize the standard training loss on the retain set. This corresponds to the objective

$$
\begin{array} { r } { L _ { \mathrm { R L + F T } } ( \theta ) = \lambda L _ { \mathrm { R L } } ( \theta ; F ) + L _ { \mathrm { r e t } } ( \theta ; R ) , \qquad \lambda > 0 , } \end{array}
$$

so the resulting gradient-flow vector field is the sum of the Random Labels and retraining vector fields. This setting reflects the common use of retain fine-tuning to preserve retained-data perfor mance and also underlies several more elaborate unlearning methods.

Lemma D.14 (Symmetry of the RL+FT trajectory). Random Labels unlearning with simultaneous retain fine-tuning preserves the invariant

$$
u ( t ) = J v ( t ) \qquad f o r a l l t \geq 0 .
$$

Proof. The invariant holds at initialization since

$$
u ( 0 ) = v ( 0 ) = \delta e _ { 1 } .
$$

By Lemma D.4, the retraining vector field preserves the manifold

$$
{ \mathcal { M } } = \{ ( u , v ) : u = J v \} ,
$$

and by Lemma D.9, the Random Labels vector field preserves the same manifold. Since the combined dynamics are given by the sum of these two vector fields, their vector field is also tangent to M. Hence,

$$
u ( t ) = J v ( t )
$$

along the entire trajectory.

Lemma D.15. Define

$$
G _ { \mathrm { R L + F T } } ( t ) : = \int _ { 0 } ^ { t } \left[ \varepsilon h \bigl ( m _ { R } ( \tau ) \bigr ) + ( 1 - \varepsilon ) \lambda \psi \bigl ( r _ { F } ( \tau ) \bigr ) \right] d \tau ,
$$

where

$$
\begin{array} { r } { m _ { R } ( t ) = a ( t ) c + 2 \varepsilon p ( t ) s ( t ) , \qquad r _ { F } ( t ) = - a ( t ) c + 2 ( 1 - \varepsilon ) p ( t ) s ( t ) . } \end{array}
$$

Then

$$
p ( t ) = \delta \cosh G _ { \mathrm { R L + F T } } ( t ) , \qquad s ( t ) = \delta \sinh G _ { \mathrm { R L + F T } } ( t ) ,
$$

while the shortcut coefficient satisfies

$$
\dot { a } ( t ) = c h \big ( m _ { R } ( t ) \big ) - c \lambda \psi \big ( r _ { F } ( t ) \big ) .
$$

Proof. By Lemma D.14, write

$$
u ( t ) = { \binom { p ( t ) } { s ( t ) } } , \qquad v ( t ) = { \binom { p ( t ) } { - s ( t ) } } .
$$

The retraining and Random Labels vector fields derived above contribute

$$
\begin{array} { r } { \dot { p } _ { R } = \varepsilon h ( m _ { R } ) s , \qquad \dot { s } _ { R } = \varepsilon h ( m _ { R } ) p , \qquad \dot { a } _ { R } = c h ( m _ { R } ) , } \end{array}
$$

and

$$
\begin{array} { r } { \dot { p } _ { \mathrm { R L } } = ( 1 - \varepsilon ) \psi ( r _ { F } ) s , \qquad \dot { s } _ { \mathrm { R L } } = ( 1 - \varepsilon ) \psi ( r _ { F } ) p , \qquad \dot { a } _ { \mathrm { R L } } = - c \psi ( r _ { F } ) . } \end{array}
$$

Since the simultaneous dynamics are the weighted sum of the two vector fields,

$$
\dot { p } = \left[ \varepsilon h ( m _ { R } ) + ( 1 - \varepsilon ) \lambda \psi ( r _ { F } ) \right] s , \qquad \dot { s } = \left[ \varepsilon h ( m _ { R } ) + ( 1 - \varepsilon ) \lambda \psi ( r _ { F } ) \right] p .
$$

Lemma D.5, together with $p ( 0 ) = \delta$ and $s ( 0 ) = 0$ , therefore gives

$$
p ( t ) = \delta \cosh G _ { \mathrm { R L + F T } } ( t ) , \qquad s ( t ) = \delta \sinh G _ { \mathrm { R L + F T } } ( t ) .
$$

Finally,

$$
\dot { a } = \dot { a } _ { R } + \lambda \dot { a } _ { \mathrm { R L } } = c h ( m _ { R } ) - c \lambda \psi ( r _ { F } ) .
$$

Lemma D.16. Along the Random Labels trajectory with retain fine-tuning,

$$
G _ { \mathrm { R L + F T } } ( t ) \to \infty , \qquad r _ { F } ( t ) \to 0 \qquad a s t \to \infty .
$$

Proof. Define

$$
S ( t ) = \frac { 1 - \varepsilon } { c } a ( t ) + G _ { \mathrm { R L + F T } } ( t ) .
$$

By Lemma D.15,

$$
\dot { S } ( t ) = h \big ( m _ { R } ( t ) \big ) > 0 ,
$$

so $S ( t )$ is strictly increasing.

We first note that both $a ( t )$ and $G _ { \mathrm { R L + F T } } ( t )$ remain non-negative along the entire trajectory. Indeed, since $S ( 0 ) ~ > ~ 0$ , if $G _ { \mathrm { R L + F T } } ~ = ~ 0$ then $a \ > \ 0$ and $r _ { F } ~ = ~ - a c ~ < ~ 0$ Therefore $\psi ( - a c ) > 0$ and

$$
\dot { G } _ { \mathrm { R L + F T } } = \varepsilon h ( a c ) + ( 1 - \varepsilon ) \lambda \psi ( - a c ) > 0 .
$$

Similarly, if $a = 0 ,$ , then $G _ { \mathrm { R L + F T } } > 0$ and $r _ { F } > 0 .$ . Therefore $\psi ( r _ { F } ) < 0$ and

$$
\dot { a } = c \big ( h ( m _ { R } ) - \lambda \psi ( r _ { F } ) \big ) > 0 .
$$

Thus, the non-negative quadrant is preserved.

Suppose now that $S ( t )$ is bounded. Since both $a ( t )$ and $G _ { \mathrm { R L + F T } } ( t )$ are non-negative, they are bounded as well. Therefore,

$$
m _ { R } = a c + \varepsilon \delta ^ { 2 } \sinh ( 2 G _ { \mathrm { R L + F T } } )
$$

is bounded. Hence,

$$
{ \dot { S } } ( t ) = h ( m _ { R } ( t ) ) \geq h \left( \operatorname* { s u p } _ { t } m _ { R } ( t ) \right) > 0 ,
$$

which implies that $S ( t )$ is unbounded, contradicting our assumption. Therefore,

$$
S ( t ) \to \infty \qquad \mathrm { a s } t \to \infty .
$$

Boundedness of $r _ { F } ( t )$ . We next show that $r _ { F } ( t )$ remains bounded. Otherwise, there would exist a sequence of times $t _ { k }$ such that

$$
| r _ { F } ( t _ { k } ) |  \infty .
$$

The Random Labels component of $\boldsymbol { L } _ { \mathrm { R L + F T } }$ then diverges, and hence

$$
L _ { \mathrm { R L + F T } } ( t _ { k } ) \to \infty ,
$$

contradicting the monotonicity of gradient flow,

$$
\begin{array} { r } { L _ { \mathrm { R L + F T } } ( t ) \leq L _ { \mathrm { R L + F T } } ( 0 ) . } \end{array}
$$

Thus, $| r _ { F } ( t ) | \le C$ for some $C > 0$ . Boundedness of $r _ { F } ,$ together with $S ( t )  \infty$ , implies

$$
\boxed { G _ { \mathrm { R L + F T } } ( t ) \to \infty \qquad \mathrm { a n d } \qquad a ( t ) \to \infty . }
$$

Convergence of $r _ { F }$ to zero. It remains to show that $r _ { F } ( t ) \to 0$ . From the limits above,

$$
m _ { R } ( t ) = a ( t ) c + \varepsilon \delta ^ { 2 } \sinh ( 2 G _ { \mathrm { R L + F T } } ( t ) ) \to \infty ,
$$

and therefore $h ( m _ { R } ( t ) )  0$ . From the dynamics,

$$
\begin{array} { r } { \dot { r } _ { F } ( t ) = c ^ { 2 } \Big ( \lambda \psi ( r _ { F } ( t ) ) - h ( m _ { R } ( t ) ) \Big ) } \end{array}\tag{17}
$$

$$
+ 2 ( 1 - \varepsilon ) \delta ^ { 2 } \Bigl ( \varepsilon h ( m _ { R } ( t ) ) + ( 1 - \varepsilon ) \lambda \psi ( r _ { F } ( t ) ) \Bigr ) \cosh ( 2 G _ { \mathrm { R L + F T } } ( t ) ) .\tag{18}
$$

Fix any $\eta > 0$ . Since $h ( m _ { R } ( t ) )  0$ , for all sufficiently large t,

$$
\frac { \lambda } { 2 } | \psi \left( \frac { \eta } { 2 } \right) | > | h ( m _ { R } ( t ) ) | \ge \frac { \varepsilon } { 1 - \varepsilon } | h ( m _ { R } ( t ) ) | .
$$

From this time onward, whenever

$$
| r _ { F } ( t ) | \geq \frac { \eta } { 2 } ,
$$

the sign of $\dot { r } _ { F } ( t )$ is determined by $\psi ( r _ { F } ( t ) )$ , and is therefore opposite to the sign of $r _ { F } ( t )$ Moreover, $| \dot { r } _ { F } ( t ) |$ is bounded away from zero at such points. Therefore, for every $\eta > 0$ , there exists a time after which $r _ { F } ( t )$ remains in $[ - \eta , \eta ]$ . Hence,

$$
\boxed { r _ { F } ( t )  0 . }
$$

Theorem D.17 (Asymptotic feature spectrum under RL+FT). Along the Random Labels trajectory with retain fine-tuning, the normalized feature responses and corresponding PCA directions are those ofProposition D.7. Then their strengths satisfy

$$
\frac { \lambda _ { \mathrm { s } } ( t ) } { \lambda _ { - } ( t ) } \longrightarrow \frac { 1 - \varepsilon } { 2 \varepsilon } \qquad a s t \to \infty .
$$

Proof. By Proposition D.7,

$$
\lambda _ { \mathrm { s } } = a ^ { 2 } c ^ { 2 } , \qquad \lambda _ { \mathrm { - } } = 2 \varepsilon ( 1 - \varepsilon ) \delta ^ { 4 } \sinh ^ { 2 } ( 2 G _ { \mathrm { R L + F T } } ) ,
$$

and hence

$$
\frac { \lambda _ { \mathrm { s } } } { \lambda _ { - } } = \frac { 1 - \varepsilon } { 2 \varepsilon } \left( \frac { a c } { ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G _ { \mathrm { R L + F T } } ) } \right) ^ { 2 } .
$$

By Lemma D.16,

$$
r _ { F } = - a c + ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G _ { \mathrm { R L + F T } } ) \to 0 , \qquad G _ { \mathrm { R L + F T } } \to \infty .
$$

Therefore,

$$
\frac { a c } { ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G _ { \mathrm { R L + F T } } ) }  1 ,
$$

which proves the claim.

## D.7 GRADIENT ASCENT UNLEARNING DYNAMICS

We next analyze Gradient Ascent (GA) without retain fine-tuning, initialized from the full-data solution,

$$
a ( 0 ) = A , \qquad u ( 0 ) = v ( 0 ) = \delta e _ { 1 } ,
$$

with $A > 0$ . GA reverses the forget-set gradient, yielding

$$
\dot { a } ( t ) = - \mathbb { E } _ { ( x , y ) \sim D _ { f } } \left[ h \big ( m _ { \theta ( t ) } ( x , y ) \big ) y s \right] ,\tag{19}
$$

$$
\begin{array} { r } { \dot { u } ( t ) = - \mathbb { E } _ { ( x , y ) \sim D _ { f } } \left[ h \big ( m _ { \theta ( t ) } ( x , y ) \big ) y \big ( u ( t ) ^ { \top } \widetilde { z } \big ) \widetilde { z } \right] , } \end{array}\tag{20}
$$

$$
\dot { v } ( t ) = \mathbb { E } _ { ( x , y ) \sim D _ { f } } \left[ h \big ( m _ { \theta ( t ) } ( x , y ) \big ) y \big ( v ( t ) ^ { \top } \widetilde { z } \big ) \widetilde { z } \right] .\tag{21}
$$

As in retraining and Random Labels, GA preserves $u ( t ) = J v ( t )$ . We use this invariant to characterize the trajectory and compare its asymptotic time to retraining.

Lemma D.18 (Symmetry of the Gradient Ascent trajectory). Consider Gradient Ascent initial ized from

$$
a ( 0 ) = A , \qquad u ( 0 ) = v ( 0 ) = \delta e _ { 1 } ,
$$

and let

$$
J = \left( \begin{array} { c c } { { 1 } } & { { 0 } } \\ { { 0 } } & { { - 1 } } \end{array} \right) .
$$

Then

$$
u ( t ) = J v ( t ) \qquad f o r a l l t \geq 0 .
$$

Proof. The relation holds at initialization. Suppose $u = J v ,$ and write

$$
u = { \binom { p } { s } } , \qquad v = { \binom { p } { - s } } .
$$

For every forget example, the margin

$$
m _ { F } = y f _ { \theta } ( x ) = a c - 2 ( 1 - \varepsilon ) p s ,
$$

so the margin is independent of the particular data point. Moreover,

$$
\mathbb { E } _ { D _ { f } } \left[ y \widetilde { z } \widetilde { z } ^ { \top } \right] = - ( 1 - \varepsilon ) K , \qquad K = \binom { 0 } { 1 } \ 0 \sp { 1 } .
$$

Hence, on the manifold $u = J v ,$

$$
\dot { u } = ( 1 - \varepsilon ) h ( m _ { F } ) K u , \qquad \dot { v } = - ( 1 - \varepsilon ) h ( m _ { F } ) K v .
$$

Since $J K = - K J ,$

$$
J \dot { v } = - ( 1 - \varepsilon ) h ( m _ { F } ) J K v = ( 1 - \varepsilon ) h ( m _ { F } ) K J v = \dot { u } .
$$

Thus the Gradient Ascent vector field preserves the relation $u = J v$ . Since it holds at initialization, it holds along the entire trajectory. □

Proposition D.19 (Feature spectrum along the GA trajectory). Let

$$
G _ { \mathrm { G A } } ( t ) = \int _ { 0 } ^ { t } ( 1 - \varepsilon ) h \big ( m _ { F } ( \tau ) \big ) d \tau .
$$

Along the Gradient Ascent trajectory, the normalized feature responses and corresponding PCA directions are those of Proposition D.7. Then their strengths are

$$
\begin{array} { l } { \displaystyle \lambda _ { \mathrm { s } } ( t ) = c ^ { 2 } \left( A - \frac { c } { 1 - \varepsilon } G _ { \mathrm { G A } } ( t ) \right) ^ { 2 } , } \\ { \displaystyle \lambda _ { + } ( t ) = 2 \varepsilon ( 1 - \varepsilon ) ( 1 - 2 \varepsilon ) ^ { 2 } \delta ^ { 4 } \cosh ^ { 2 } \bigl ( 2 G _ { \mathrm { G A } } ( t ) \bigr ) , } \\ { \displaystyle \lambda _ { - } ( t ) = 2 \varepsilon ( 1 - \varepsilon ) \delta ^ { 4 } \sinh ^ { 2 } \bigl ( 2 G _ { \mathrm { G A } } ( t ) \bigr ) . } \end{array}
$$

Proof. By Lemma D.18, the Gradient Ascent trajectory satisfies the J-invariant. Moreover, applying Lemma D.5 to the reduced GA dynamics gives

$$
p ( t ) = \delta \cosh G _ { \mathrm { G A } } ( t ) , \qquad s ( t ) = \delta \sinh G _ { \mathrm { G A } } ( t ) ,
$$

while

$$
a ( t ) = A - \frac { c } { 1 - \varepsilon } G _ { \mathrm { G A } } ( t ) .
$$

The principal directions therefore follow from Proposition D.7. Using

$$
p ( t ) ^ { 2 } + s ( t ) ^ { 2 } = \delta ^ { 2 } \cosh \bigl ( 2 G _ { \mathrm { G A } } ( t ) \bigr ) , \qquad p ( t ) s ( t ) = \frac { \delta ^ { 2 } } { 2 } \sinh \bigl ( 2 G _ { \mathrm { G A } } ( t ) \bigr ) ,
$$

and substituting these expressions together with $a ( t )$ into the corresponding eigenvalues yields the result. □

Although Gradient Ascent can eventually make the core direction $q _ { - }$ dominate the shortcut, as retraining does, matching this qualitative behavior is not sufficient: unlearning should reach a retraining-like representation substantially faster than retraining itself.

To compare their timescales, let the full model be trained for time τ, reaching $a ( \tau ) = A$ , and retrain from the same initialization on $D _ { r }$ for time τ. We take its resulting feature ratio

$$
\frac { \lambda _ { - } } { \lambda _ { \mathrm { s } } } \bigg | _ { \mathrm { r e t r } , t = \tau }
$$

as the target. Starting GA from the corresponding full model with $a ( 0 ) = A { \bmod { \mathrm { \Omega } } }$ , let $\tau _ { \mathrm { G A } }$ be the time required to match this ratio:

$$
\left. \frac { \lambda _ { - } } { \lambda _ { \mathrm { s } } } \right| _ { \mathrm { G A } , t = \tau _ { \mathrm { G A } } } = \left. \frac { \lambda _ { - } } { \lambda _ { \mathrm { s } } } \right| _ { \mathrm { r e t r } , t = \tau } .
$$

We compare $\tau _ { \mathrm { G A } }$ with τ asymptotically as $A \to \infty$

Lemma D.20. Let $\tau ( A )$ be the time required by full-data training to reach $a ( \tau ( A ) ) = A .$ . Then

$$
\tau ( A ) = \frac { A } { c } + \frac { e ^ { c A } - 1 } { c ^ { 2 } } .
$$

Proof. By Lemma D.1,

$$
\dot { a } = \frac { c } { 1 + e ^ { c a } } , \qquad \frac { d t } { d a } = \frac { 1 + e ^ { c a } } { c } .
$$

Integrating from $a ( 0 ) = 0$ to $a ( \tau ( A ) ) = A$ gives

$$
\tau ( A ) = \int _ { 0 } ^ { A } { \frac { 1 + e ^ { c a } } { c } } d a = { \frac { A } { c } } + { \frac { e ^ { c A } - 1 } { c ^ { 2 } } } .
$$

Lemma D.21 (Unique Gradient Ascent matching point). Let

$$
G _ { \mathrm { r e t r } } ^ { A } = G _ { \mathrm { r e t r } } ( \tau ( A ) ) .
$$

There exists a unique

$$
G _ { \mathrm { G A } } ^ { A } \in \left( 0 , \frac { ( 1 - \varepsilon ) A } { c } \right)
$$

such that Gradient Ascent matches the core-to-shortcut ratio achieved by retraining at time $\tau ( A )$ . It is uniquely characterized by

$$
\frac { \sinh \bigl ( 2 G _ { \mathrm { G A } } ^ { A } \bigr ) } { A - \frac { c } { 1 - \varepsilon } G _ { \mathrm { G A } } ^ { A } } = \frac { \sinh \bigl ( 2 G _ { \mathrm { r e t r } } ^ { A } \bigr ) } { \frac { c } { \varepsilon } G _ { \mathrm { r e t r } } ^ { A } } .
$$

Consequently, there is a unique matching time $\tau _ { \mathrm { G A } }$ such that

$$
G _ { \mathrm { G A } } ( \tau _ { \mathrm { G A } } ) = G _ { \mathrm { G A } } ^ { A } .
$$

Proof. At time $\tau ( A )$ , retraining achieves the core-to-shortcut ratio

$$
\mathcal { R } _ { \mathrm { r e t r } } ( A ) = \frac { 2 \varepsilon ^ { 3 } ( 1 - \varepsilon ) \delta ^ { 4 } } { c ^ { 4 } } \frac { \sinh ^ { 2 } \left( 2 G _ { \mathrm { r e t r } } ^ { A } \right) } { \left( G _ { \mathrm { r e t r } } ^ { A } \right) ^ { 2 } } .
$$

For Gradient Ascent,

$$
a ( t ) = A - \frac { c } { 1 - \varepsilon } G _ { \mathrm { G A } } ( t ) ,
$$

and Proposition D.19 gives

$$
\mathcal { R } _ { \mathrm { G A } } ( t ) = \frac { 2 \varepsilon ( 1 - \varepsilon ) \delta ^ { 4 } \sinh ^ { 2 } \bigl ( 2 G _ { \mathrm { G A } } ( t ) \bigr ) } { c ^ { 2 } \biggl ( A - \frac { c } { 1 - \varepsilon } G _ { \mathrm { G A } } ( t ) \biggr ) ^ { 2 } } .
$$

Equating the two ratios and taking the positive square root yields

$$
\frac { \sinh \bigl ( 2 G _ { \mathrm { G A } } ^ { A } \bigr ) } { A - \frac { c } { 1 - \varepsilon } G _ { \mathrm { G A } } ^ { A } } = \frac { \sinh \bigl ( 2 G _ { \mathrm { r e t r } } ^ { A } \bigr ) } { \frac { c } { \varepsilon } G _ { \mathrm { r e t r } } ^ { A } } .\tag{22}
$$

It remains to show that this matching point is unique. Before the shortcut coefficient reaches zero,

$$
0 \leq G < \frac { ( 1 - \varepsilon ) A } { c } .
$$

On this interval, define

$$
F _ { A } ( G ) : = \frac { \sinh ( 2 G ) } { A - \frac { c } { 1 - \varepsilon } G } .
$$

The numerator is strictly increasing and the denominator strictly decreasing and positive, so $F _ { A }$ is strictly increasing, with

$$
F _ { A } ( 0 ) = 0 , \qquad F _ { A } ( G ) \to \infty \quad \mathrm { a s } \quad G \to \frac { ( 1 - \varepsilon ) A } { c } ^ { - } .
$$

Since the right-hand side of equation 22 is finite and positive, there is a unique

$$
G _ { \mathrm { G A } } ^ { A } \in \left( 0 , \frac { ( 1 - \varepsilon ) A } { c } \right)
$$

satisfying the matching condition.

Finally,

$$
\dot { G } _ { \mathrm { G A } } ( t ) = ( 1 - \varepsilon ) h ( m _ { F } ( t ) ) > 0 ,
$$

so $G _ { \mathrm { G A } } ( t )$ is strictly increasing and reaches $G _ { \mathrm { G A } } ^ { A }$ at a unique time $\tau _ { \mathrm { G A } }$

Lemma D.22 (Training-time representations). Let $\tau ( A )$ be the full-data training time required to reach shortcut coefficient A, let

$$
G _ { \mathrm { r e t r } } ^ { A } = G _ { \mathrm { r e t r } } ( \tau ( A ) ) ,
$$

and let $G _ { \mathrm { G A } } ^ { A }$ be the unique Gradient Ascent matching point from Lemma D.21. Then

$$
\tau _ { \mathrm { G A } } = \frac { 1 } { 1 - \varepsilon } \int _ { 0 } ^ { G _ { \mathrm { G A } } ^ { A } } \left( 1 + e ^ { m _ { \mathrm { G A } } ( G ) } \right) d G ,
$$

where

$$
m _ { \mathrm { G A } } ( G ) = A c - \frac { c ^ { 2 } } { 1 - \varepsilon } G - ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G ) .
$$

Similarly,

$$
\tau ( A ) = \frac { 1 } { \varepsilon } \int _ { 0 } ^ { G _ { \mathrm { r e t r } } ^ { A } } \Bigl ( 1 + e ^ { m _ { \mathrm { r e t r } } ( G ) } \Bigr ) \ d G ,
$$

where

$$
m _ { \mathrm { r e t r } } ( G ) = \frac { c ^ { 2 } } { \varepsilon } G + \varepsilon \delta ^ { 2 } \sinh ( 2 G ) .
$$

Proof. Along the Gradient Ascent trajectory,

$$
a = A - \frac { c } { 1 - \varepsilon } G _ { \mathrm { G A } } , 2 p s = \delta ^ { 2 } \sinh ( 2 G _ { \mathrm { G A } } ) .
$$

Since the forget margin is

$$
m _ { F } = a c - 2 ( 1 - \varepsilon ) p s ,
$$

we obtain

$$
m _ { \mathrm { G A } } ( G ) = A c - \frac { c ^ { 2 } } { 1 - \varepsilon } G - ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G ) .
$$

Therefore,

$$
\dot { G } _ { \mathrm { G A } } = ( 1 - \varepsilon ) h ( m _ { \mathrm { G A } } ) = \frac { 1 - \varepsilon } { 1 + e ^ { m _ { \mathrm { G A } } } } ,
$$

and hence

$$
\tau _ { \mathrm { G A } } = \frac { 1 } { 1 - \varepsilon } \int _ { 0 } ^ { G _ { \mathrm { G A } } ^ { A } } \left( 1 + e ^ { m _ { \mathrm { G A } } ( G ) } \right) d G .
$$

For retraining, Lemma D.6 gives

$$
m _ { \mathrm { r e t r } } ( G ) = \frac { c ^ { 2 } } { \varepsilon } G + \varepsilon \delta ^ { 2 } \sinh ( 2 G ) ,
$$

and

$$
\dot { G } _ { \mathrm { r e t r } } = \varepsilon h ( m _ { \mathrm { r e t r } } ) = \frac { \varepsilon } { 1 + e ^ { m _ { \mathrm { r e t r } } } } .
$$

Integrating up to $G _ { \mathrm { r e t r } } ^ { A } = G _ { \mathrm { r e t r } } ( \tau ( A ) )$ gives

$$
\tau ( A ) = \frac { 1 } { \varepsilon } \int _ { 0 } ^ { G _ { \mathrm { r e t r } } ^ { A } } \Bigl ( 1 + e ^ { m _ { \mathrm { r e t r } } ( G ) } \Bigr ) \ d G .
$$

Lemma D.23. As $A \to \infty ,$

$$
{ \cal G } _ { \mathrm { G A } } ^ { A }  \infty .
$$

Proof. By Lemma D.20,

$$
\tau ( A ) \to \infty .
$$

Together with the retraining-time representation from Lemma D.22, this forces

$$
G _ { \mathrm { r e t r } } ^ { A } \to \infty ,
$$

since otherwise the corresponding integral would remain bounded. Hence

$$
\frac { \sinh \bigl ( 2 G _ { \mathrm { r e t r } } ^ { A } \bigr ) } { \frac { c } { \varepsilon } G _ { \mathrm { r e t r } } ^ { A } }  \infty .
$$

By the matching condition in Lemma D.21,

$$
\frac { \sinh \bigl ( 2 G _ { \mathrm { G A } } ^ { A } \bigr ) } { A - \frac { c } { 1 - \varepsilon } G _ { \mathrm { G A } } ^ { A } } = \frac { \sinh \bigl ( 2 G _ { \mathrm { r e t r } } ^ { A } \bigr ) } { \frac { c } { \varepsilon } G _ { \mathrm { r e t r } } ^ { A } } ,
$$

so the left-hand side must also diverge.

If $G _ { \mathrm { G A } } ^ { A }$ did not tend to infinity, there would be a subsequence along which $G _ { \mathrm { G A } } ^ { A } \leq M$ for some finite M. Along this subsequence, the numerator remains bounded while

$$
A - \frac { c } { 1 - \varepsilon } G _ { \mathrm { G A } } ^ { A } \to \infty ,
$$

so the left-hand side tends to zero, a contradiction. Therefore

$$
{ \cal G } _ { \mathrm { G A } } ^ { A }  \infty .
$$

Theorem D.24 (Asymptotic runtime of Gradient Ascent). Let $\tau ( A )$ be the time required $b y$ full-data training to reach shortcut coefficient A, and let $\tau _ { \mathrm { G A } }$ be the time required by Gradient Ascent to match the core-to-shortcut feature ratio achieved by retraining after time $\tau ( A )$ . Then

$$
\operatorname* { l i m } _ { A \to \infty } \frac { \tau _ { \mathrm { G A } } } { \tau ( A ) } = R _ { \mathrm { G A } } ( \varepsilon , \delta , c ) ,
$$

where

$$
R _ { \mathrm { G A } } ( \varepsilon , \delta , c ) : = \int _ { 0 } ^ { \infty } \exp \left( - z - ( 1 - \varepsilon ) \delta ^ { 2 } \sinh \left( \frac { 2 ( 1 - \varepsilon ) } { c ^ { 2 } } z \right) \right) d z .
$$

For every fixed $\varepsilon \in ( 0 , 1 ) , \delta > 0 ,$ , and $c > 0 ,$

$$
0 < R _ { \mathrm { G A } } ( \varepsilon , \delta , c ) < 1 .
$$

Moreover,

$$
\operatorname * { l i m } _ { \delta  0 } R _ { \mathrm { G A } } ( \varepsilon , \delta , c ) = 1 , \qquad \operatorname * { l i m } _ { c  \infty } R _ { \mathrm { G A } } ( \varepsilon , \delta , c ) = 1 .
$$

Thus, the asymptotic runtime of Gradient Ascent can be made arbitrarily close to that of $r e \mathrm { - }$ training.

Proof. By Lemma D.22,

$$
\tau _ { \mathrm { G A } } = \frac { G _ { \mathrm { G A } } ^ { A } } { 1 - \varepsilon } + \frac { e ^ { A c } } { 1 - \varepsilon } \int _ { 0 } ^ { G _ { \mathrm { G A } } ^ { A } } \exp \left( - \frac { c ^ { 2 } } { 1 - \varepsilon } G - ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G ) \right) d G .
$$

Define

$$
I _ { \infty } : = \int _ { 0 } ^ { \infty } \exp \left( - \frac { c ^ { 2 } } { 1 - \varepsilon } G - ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G ) \right) d G , \qquad I _ { \infty } < \infty .
$$

Since $G _ { \mathrm { G A } } ^ { A } \to \infty$ by Lemma D.23, the remaining tail satisfies

$$
0 \leq { I _ { \infty } } - \int _ { 0 } ^ { G _ { \mathrm { G A } } ^ { A } } \exp \left( - \frac { c ^ { 2 } } { 1 - \varepsilon } G - ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G ) \right) d G \leq \frac { 1 - \varepsilon } { c ^ { 2 } } \exp \left( - \frac { c ^ { 2 } } { 1 - \varepsilon } G _ { \mathrm { G A } } ^ { A } \right) = o ( 1 ) .
$$

Hence

$$
\int _ { 0 } ^ { G _ { \mathrm { G A } } ^ { A } } \exp \left( - \frac { c ^ { 2 } } { 1 - \varepsilon } G - ( 1 - \varepsilon ) \delta ^ { 2 } \sinh ( 2 G ) \right) d G = I _ { \infty } + o ( 1 ) .
$$

Moreover,

$$
G _ { \mathrm { G A } } ^ { A } < \frac { 1 - \varepsilon } { c } A = o ( e ^ { A c } ) ,
$$

and hence

$$
\tau _ { \mathrm { G A } } = \frac { I _ { \infty } } { 1 - \varepsilon } e ^ { A c } ( 1 + o ( 1 ) ) .
$$

Combining this with Lemma D.20,

$$
\tau ( A ) = \frac { e ^ { A c } } { c ^ { 2 } } ( 1 + o ( 1 ) ) ,
$$

gives

$$
\frac { \tau _ { \mathrm { G A } } } { \tau ( A ) } \longrightarrow \frac { c ^ { 2 } } { 1 - \varepsilon } I _ { \infty } .
$$

With the change of variables

$$
z = { \frac { c ^ { 2 } } { 1 - \varepsilon } } G ,
$$

this is exactly $R _ { \mathrm { G A } } ( \varepsilon , \delta , c )$

The integrand defining $R _ { \mathrm { G A } }$ is strictly positive and, for $z > 0 ,$

$$
\exp \left( - z - ( 1 - \varepsilon ) \delta ^ { 2 } \sinh \left( \frac { 2 ( 1 - \varepsilon ) } { c ^ { 2 } } z \right) \right) < e ^ { - z } .
$$

Therefore

$$
0 < R _ { \mathrm { G A } } ( \varepsilon , \delta , c ) < \int _ { 0 } ^ { \infty } e ^ { - z } d z = 1 .
$$

Finally, as either $\delta \to 0 \mathrm { ~ o r ~ } c \to \infty$ , the integrand converges pointwise to $e ^ { - z }$ and remains bounded by $e ^ { - z }$ . The dominated convergence theorem therefore yields

$$
R _ { \mathrm { G A } } ( \varepsilon , \delta , c )  1
$$

in both limits.

Corollary D.25 (Fixed feature-ratio targets). $F i x \ : r > 0 ,$ and let $\tau _ { \mathrm { G A } } ^ { ( r ) } ( A )$ be the first time such   
that   
λ   
λ<sub>s</sub> = r.   
Then   
$\frac { \tau _ { \mathrm { G A } } ^ { ( r ) } ( A ) } { \tau ( A ) } \longrightarrow R _ { \mathrm { G A } } ( \varepsilon , \delta , c )$ as A → ∞.   
Thus, the leading-order GA runtime is independent of the fixed target ratio $^ { r } \cdot$

Theorem 4.3 follows by reparameterizing the limit in training time. By Lemma $\mathrm { D } . 2 , a ( T )$ is increasing and $a ( T ) \to \infty$ , so setting $A = a ( T )$ gives $\tau ( A ) = T$ and $A \to \infty$ as $T \to \infty$ . The claim then follows from Theorem D.24 and $R _ { \mathrm { G A } } ( \varepsilon , \delta , c )  1$ as $\delta \to 0 \mathrm { o r } c \to \infty$

Proof. The argument is identical to that of Theorem D.24. For any fixed $r > 0 ,$ , the corresponding matching point $G _ { r } ^ { A }$ satisfies $G _ { r } ^ { A } \to \infty$ and $G _ { r } ^ { A } = O ( A )$ , so the same asymptotic estimates apply. □

## E ADDITIONAL RESULTS

## E.1 EXPERIMENTAL DETAILS

Full and retrained models use the same training recipe and differ only in the training data. For CIFAR-100 (Krizhevsky & Hinton, 2009), we fine-tune an ImageNet-21k-pretrained ViT-Tiny (Dosovitskiy et al., 2020) for 50 epochs with AdamW $( \mathrm { l r } = 5 \times \mathsf { \bar { 1 0 ^ { - 5 } } }$ , weight decay 0.05), 5-epoch warmup, cosine decay, and batch size 128. On SVHN, ViT-Tiny is trained from scratch for 100 epochs with AdamW $( \mathrm { l r } \dot { = } 1 0 ^ { - 3 }$ , weight decay 0.05), cosine decay, and batch size 256. ResNet-56 (He et al., 2015) is trained on CIFAR-100 for 200 epochs with SGD $( \mathrm { l r } = 0 . 1$ , momentum 0.9, weight decay $5 \times 1 0 ^ { - 4 } )$ , cosine decay, and batch size 256.

For each method, we tune only the main hyperparameters with a small manual sweep. Across FT, AMUN, NegGrad+, RL, SalUn, LTD, and ReprUnl, learning rates are swept over method-specific grids within $\mathrm { 1 0 ^ { - 4 } { - } 1 0 ^ { - 1 } }$ ; AMUN, RL, and SalUn use $\lambda \in [ 1 , { \bar { 5 } } ]$ , while NegGrad+ uses $\lambda \in [ 0 . 0 1 , 1 ]$ IU sweeps $\alpha \in [ 0 . 3 , 5 0 ]$ ], and ReprUnl additionally sweeps $\beta \in [ 1 , 1 5 ]$ . For epoch-based methods, we use 5 unlearning epochs on ViT-Tiny/CIFAR-100 (versus 50 training epochs) and 10 epochs in the remaining settings; ReprUnl uses 300 update steps.

Each model set contains 20 independently trained seeds. We compute feature metrics for three independently constructed sets and report the mean and standard deviation across them in the scatter plots.

For each setting, we choose K so that the top-K principal components capture at least 80% of the total feature mass. This gives $K = 7 5$ for ViT-Tiny/CIFAR-100, $K = 2 0$ for ViT-Tiny/SVHN, and $K = 3 0$ for both ResNet-56 settings. For head fine-tuning, we freeze the model body and train a fresh linear head for 30 epochs with Adam $( \mathrm { l r } = 1 0 ^ { - 3 }$ , no weight decay).

## E.2 STANDARD METRICS

Table 2 reports standard post-unlearning accuracies for the main ViT-Tiny/CIFAR-100 setting. Most methods perform well under these metrics: they reduce forget accuracy close to the retraining baseline while largely preserving performance on the retained data. IU is the clear exception, leaving both forget and test accuracy almost unchanged from the full-data model. Thus, methods that perform well on these standard metrics can still differ substantially in how closely their learned features match those of retraining (Figure 1).

The adaptation results in Figure 2 further illustrate this distinction. ReprUnl has the smallest mean absolute accuracy gap from retraining across the three standard metrics, yet exhibits the largest feature discrepancy on the forget set and one of the largest accuracy gaps after partial relearning.

Conversely, IU has the largest mean absolute gap under the standard metrics but remains closest to retraining among unlearning methods in downstream accuracy on both transfer tasks.  
Table 2: Standard post-unlearning accuracy. Values are mean ± std over 5 runs, in %. Blue values in brackets show deviation from Retrain in percentage points.
<table><tr><td>Method</td><td>Retain Acc.</td><td>Forget Acc.</td><td>Test Acc.</td></tr><tr><td>Retrain</td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $7 0 . 8 \pm 0 . 1$ </td></tr><tr><td>FT</td><td> $9 4 . 1 \pm 0 . 7 \ : ( - 5 . 9 )$ </td><td> $0 . 3 \pm 0 . 2 \ ( + 0 . 3 )$ </td><td> $6 5 . 0 \pm 0 . 6 ~ ( - 5 . 8 )$ </td></tr><tr><td>AMUN</td><td> $9 3 . 7 \pm 0 . 4 ( - 6 . 2 )$ </td><td> $2 . 1 \pm 1 . 4 ( + 2 . 1 )$ </td><td> $6 5 . 4 \pm 0 . 5 \ ( - 5 . 4 )$ </td></tr><tr><td>NegGrad+</td><td> $9 7 . 2 \pm 0 . 2 ( - 2 . 8 )$ </td><td> $0 . 0 \pm 0 . 0 \ ( 0 . 0 )$ </td><td> $6 7 . 4 \pm 0 . 3 ( - 3 . 4 )$ </td></tr><tr><td>IU</td><td> $1 0 0 . 0 \pm 0 . 0 \ ( 0 . 0 ) $ </td><td> $1 0 0 . 0 \pm 0 . 0 \ ( + 1 0 0 . 0 )$ </td><td> $8 6 . 7 \pm 0 . 2 ( + 1 5 . 9 )$ </td></tr><tr><td>LTD</td><td> $9 7 . 4 \pm 0 . 3 ( - 2 . 5 )$ </td><td> $0 . 0 \pm 0 . 0 \ ( 0 . 0 )$ </td><td> $6 5 . 9 \pm 0 . 2 \ ( - 4 . 9 )$ </td></tr><tr><td>RL</td><td> $9 3 . 3 \pm 0 . 3 ( - 6 . 7 )$ </td><td> $0 . 4 \pm 0 . 1 \ ( + 0 . 4 )$ </td><td> $6 5 . 1 \pm 0 . 3 ( - 5 . 7 )$ </td></tr><tr><td>Rep. Unl.</td><td> $1 0 0 . 0 \pm 0 . 0 \ ( 0 . 0 ) $ </td><td> $2 . 8 \pm 0 . 5 \ ( + 2 . 8 )$ </td><td> $7 0 . 6 \pm 0 . 2 ( - 0 . 2 )$ </td></tr><tr><td>SalUn</td><td> $9 4 . 9 \pm 0 . 4 ( - 5 . 0 )$ </td><td> $0 . 9 \pm 0 . 2 \ ( + 0 . 9 )$ </td><td> $6 6 . 3 \pm 0 . 3 \ : ( - 4 . 5 )$ </td></tr></table>

To separate the learned representation from the particular prediction head produced by unlearning, we discard the original head, freeze the representation, and train a fresh head on the retain set. Table 3 shows that this largely removes the remaining output-level differences on the forget set: all methods achieve essentially zero forget accuracy, including IU, whose original head preserves the forgotten behavior almost entirely. Retain accuracy is also nearly saturated across methods, although some differences remain on the test set.

Table 3: Accuracy after training a fresh prediction head on the frozen representation. Values are mean ± std, in %. Blue values in brackets show deviation from Retrain in percentage points.
<table><tr><td>Method</td><td>Retain Acc.</td><td>Forget Acc.</td><td>Test Acc.</td></tr><tr><td>Retrain</td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $7 0 . 8 \pm 0 . 1$ </td></tr><tr><td>FT</td><td> $9 9 . 7 \pm 0 . 0 \ ( - 0 . 3 )$ </td><td> $0 . 0 \pm 0 . 0 \ ( 0 . 0 )$ </td><td> $6 6 . 7 \pm 0 . 4 ( - 4 . 1 )$ </td></tr><tr><td>AMUN</td><td> $9 9 . 6 \pm 0 . 1 \ : ( - 0 . 4 )$ </td><td> $0 . 0 \pm 0 . 0 \ ( 0 . 0 )$ </td><td> $6 6 . 7 \pm 0 . 3 \ : ( - 4 . 2 )$ </td></tr><tr><td>NegGrad+</td><td> $9 9 . 8 \pm 0 . 1 \ : ( - 0 . 1 )$ </td><td> $0 . 0 \pm 0 . 0 \ ( 0 . 0 )$ </td><td> $6 8 . 8 \pm 0 . 3 ( - 2 . 0 )$ </td></tr><tr><td>IU</td><td> $1 0 0 . 0 \pm 0 . 0 \ ( 0 . 0 ) $ </td><td> $0 . 0 \pm 0 . 0 \ ( 0 . 0 )$ </td><td> $7 0 . 3 \pm 0 . 2 ( - 0 . 5 )$ </td></tr><tr><td>LTD</td><td> $1 0 0 . 0 \pm 0 . 0 \ ( 0 . 0 ) $ </td><td> $0 . 0 \pm 0 . 0 \ ( 0 . 0 )$ </td><td> $6 8 . 1 \pm 0 . 3 ( - 2 . 7 )$ </td></tr><tr><td>RL</td><td> $9 9 . 0 \pm 0 . 2 \ ( - 1 . 0 ) $ </td><td> $0 . 0 \pm 0 . 0 \ ( 0 . 0 )$ </td><td> $6 6 . 6 \pm 0 . 3 ( - 4 . 2 )$ </td></tr><tr><td>Rep. Unl.</td><td> $1 0 0 . 0 \pm 0 . 0 \ ( 0 . 0 ) $ </td><td> $0 . 0 \pm 0 . 0 \ ( 0 . 0 )$ </td><td> $7 0 . 4 \pm 0 . 2 ( - 0 . 4 )$ </td></tr><tr><td>SalUn</td><td> $9 9 . 6 \pm 0 . 1 \ : ( - 0 . 4 )$ </td><td> $0 . 0 \pm 0 . 0 \ ( 0 . 0 )$ </td><td> $6 7 . 4 \pm 0 . 2 ( - 3 . 4 )$ </td></tr></table>

More generally, retain-only head training makes the methods appear nearly identical to retrain ing on the forget set, despite the feature mismatches identified in our analysis. This motivates the subsequent-use experiments in Section 6, where partial relearning and downstream adaptation expose differences encoded in the frozen representation.

## E.3 DIRECTIONAL DISCREPANCIES

Figure 4 separates the two directional terms of our feature discrepancy. For the full and retrained solutions, the two directions are comparable: retraining both loses features present in the full model and gains features absent from it. Thus, the feature change induced by deletion is genuinely bidirectional.

## E.4 ADDITIONAL SETTINGS

We complement the main-text analysis with three additional settings spanning a different architecture, a smaller feature-targeted deletion set, and a different dataset. These experiments test whether the qualitative patterns observed in the main setting persist beyond the specific ViT-Tiny/CIFAR-100 configuration. We do not assume that retraining changes the feature solution for every deletion request; some deletions may leave it essentially unchanged. Rather, the settings below test whether nontrivial full-to-retraining feature differences arise beyond the main configuration, and whether the same qualitative consequences persist when they do.

![](images/b3f79f3d0c91939bff5fc887bc0bf6003d3615089a321df2b5e9894bd5e50461.jpg)

![](images/7dc3ccf0721a000e6daeaf49349684038af9b76046305dd339b600b4a607d784.jpg)  
Figure 4: Directional feature discrepancies on ViT-Tiny/CIFAR-100, using the full-data (left) and retrained (right) solutions as reference.

ResNet-56 on CIFAR-100: class-wise deletion. We first repeat the main CIFAR-100 deletion setting with a ResNet-56, removing one fine-grained class from each of the 20 superclasses. Figure 5 shows the feature discrepancies on the test and forget distributions. As in the main ViT-Tiny experiment, methods with similar standard unlearning performance can occupy markedly different positions relative to the full-data and retrained feature solutions. Figure 6 reports the corresponding head fine-tuning results.

![](images/e35d3f0f0b9235c4c0cf084c8985b5d7d30ef23413e9f56808f9382a796bcf74.jpg)  
(a) Test Set

![](images/76f15dd00d1d1b35c00b9963888e9d82dec9c7ef03580ecb8470ef80abfd9965.jpg)  
(b) Forget Set  
Figure 5: Feature discrepancy for ResNet-56 on CIFAR-100 under class-wise deletion. Each panel compares unlearned models with the full-data and retrained feature solutions on (a) the test distribution and (b) the forget distribution.

ResNet-56 on CIFAR-100: blue-green shortcut deletion. We next consider a substantially smaller, feature-targeted deletion set. We select 20 fine-grained CIFAR-100 classes belonging to four superclasses: aquatic mammals and fish (the marine group), and trees and flowers (the plant group). Within these classes, an example is placed in the forget set only when its image border exhibits the corresponding color shortcut: blue for marine examples and green for plant examples. Concretely, we compute the mean RGB values over a five-pixel image border and remove marine examples whose blue channel exceeds both red and green by more than 10, and plant examples whose green channel exceeds both red and blue by more than 10.

This yields a forget set containing only 4.64% of the 50,000 training examples, distributed across the 20 classes. Unlike class-wise deletion, this setting targets a small set of examples sharing a simple, human-interpretable feature, allowing us to ask whether unlearning removes the corresponding feature structure rather than merely changing class-level behavior. Figure 7 shows the feature discrepancies on the test and forget distributions, and Figure 8 reports the head fine-tuning results.

![](images/2ebe5a6d0986fc6c5eabc5e3637c9336fc56e11438c99d015a80ec5550d018b2.jpg)  
(a) Partial relearning

![](images/b4617fc11bb73cd5ad4e05301ae9e0a073ff3cfc8a111ef46a96c191156fe43f.jpg)  
(b) Flowers-102

![](images/99c522144c8fb4866d05071b16621fbaaf10acf2cd9c0ffc8b6e1c56e9b68572.jpg)  
(c) Oxford-IIIT Pets

Figure 6: Feature mismatch predicts behavior under subsequent adaptation. (a) Retraining-relative feature discrepancy on the forget distribution versus the absolute forget-accuracy gap after partial relearning; colors show the sign of the gap. (b–c) Missing retraining features on downstream inputs versus downstream performance on Flowers-102 and Oxford-IIIT Pets.  
![](images/a5434a460eea7ef4c24377f4056fc9628994c2c18cc86ad92aec6b20af8426f3.jpg)  
(a) Test Set

![](images/9c23a220e858698ef3e194a8f298670600ed04df95707c10207195760cb9df17.jpg)  
(b) Forget Set  
Figure 7: Feature discrepancy for ResNet-56 on CIFAR-100 under blue-green deletion. Each panel compares unlearned models with the full-data and retrained feature solutions on (a) the test distribution and (b) the forget distribution.

ViT-Tiny on SVHN: single-class deletion. Finally, we consider ViT-Tiny on SVHN with one class removed. On both the test and forget distributions, most unlearned models remain concentrated near the full-data feature solution, providing an additional example of feature persistence (Figure 9). Because the separation between methods is comparatively small in this setting, we avoid drawing fine-grained conclusions about their relative ordering. Figure 10 reports the corresponding head fine-tuning result.

Overall, these results show that the main feature-recovery picture is not specific to the ViT-Tiny/CIFAR-100 setting. Across architectures and deletion mechanisms, standard behavioral success can coexist with substantial mismatch from retraining, while some settings—such as SVHN— are dominated instead by persistence of the full-data feature solution.

## E.5 HYPERPARAMETER ABLATIONS

Matching threshold γ. We set the matching threshold as

$$
\gamma = \kappa \cdot \gamma _ { \mathrm { c o r r } } ,
$$

where $\gamma _ { \mathrm { c o r r } }$ is the correlation threshold used to construct recurring feature clusters and κ controls the sensitivity of the discrepancy. $\mathbf { A } \mathbf { t } \kappa = 1$ , many moderately similar components already count as full matches, compressing the range of discrepancies. Increasing κ makes the metric more sensitive to imperfect feature recovery.

![](images/c6086b822185070828436d496d97f67d1148062c25d01435d707da8a8d5a64a6.jpg)  
(a) Partial relearning

![](images/3b0c23912e229abfb161d35a1bf83ba3f61833f854983bc98e1c27562892c873.jpg)  
(b) Flowers-102

![](images/c8b9a2467b0b5ba0d12b3c8e5d6505d7ba25dd378700112e2af2502e9d40a230.jpg)  
(c) Oxford-IIIT Pets

Figure 8: Feature mismatch predicts behavior under subsequent adaptation. (a) Retraining-relative feature discrepancy on the forget distribution versus the absolute forget-accuracy gap after partial relearning; colors show the sign of the gap. (b–c) Missing retraining features on downstream inputs versus downstream performance on Flowers-102 and Oxford-IIIT Pets.  
![](images/10c97c5d80950765874b7a92c45f58e5de7e7d8b9023e607b89ad9898dae9f39.jpg)  
(a) Test Set

![](images/3a0f280418293d9559aeb03ac726bc7a9b9e595b0253296dd8c0304683b6943c.jpg)  
(b) Forget Set  
Figure 9: Feature discrepancy for ViT-Tiny on SVHN class-wise deletion. Each panel compares unlearned models with the full-data and retrained feature solutions on (a) the test distribution and (b) the forget distribution.

We calibrate γ using independent retraining runs rather than unlearning results: two independently trained retraining groups should have a small discrepancy, since they represent samples from the same target solution. Figure 11 reports this sensitivity on the evaluation set. We additionally repeat the downstream comparison from the main text on Flowers-102 across different values of κ (Figure 12). The same qualitative relationship between feature discrepancy and downstream accuracy is preserved across matching thresholds. The calibration also depends on group size because matching takes the best component across all models in the evaluated group; with more independent models, retraining groups are more likely to contain close matches and therefore become closer under the metric.

Number of models. Because feature solutions are constructed from recurring components across independent runs, the comparison can depend on the number of models in each group. Figure 13 compares groups of 10, 20, and 30 models for ResNet-56 under class-wise deletion. Larger groups bring an independent retraining group closer to the reference retraining solution, consistent with improved coverage of recurring features. The broad method-level picture is preserved across group sizes, although individual discrepancies can vary, particularly for methods whose features recur less consistently across runs. We use 20 models as a compromise between feature coverage and computational cost.

![](images/bb6b954608a2acd10aa6bf615446ed3a6ca6e1a9d50ef9957cb63835e478dbc2.jpg)  
(a) Forget Set

![](images/c3d0ee9a5cd2f4232532d3f9a245eb524697022ec642aadf2676c59e3e0931aa.jpg)  
(b) Oxford-IIIT Pets

Figure 10: Feature mismatch and behavior under subsequent adaptation for ViT-Tiny/SVHN.  
![](images/7ca8f4e550167e0f905eccf6cfc0e0d0903b536b2cebfa30ae8392c83e517505.jpg)  
(a) $\kappa = 1$

![](images/97cf7a6c9c030b48d9bc4510f393fb6defd21139212c7fdada461fd6cc4b82cb.jpg)  
(b) $\kappa = 1 . 5$

![](images/2c96ec0b4cc03c6f16e800879890137b217ab68e3d54b39de75b7f7881911b49.jpg)  
(c) $\kappa = 2$ (default)

![](images/213e17be9c0f88c91c2fbc2eae486a2abc1568d9f884007cb79d3570a125d16a.jpg)  
(d) $\kappa = 2 . 5$

![](images/e967203cf5a14e181a5df0218f7a3ab0c640cc6b58209e484ecc8450c8dd3bca.jpg)  
(e) $\kappa = 3$  
Figure 11: Sensitivity of the feature discrepancy to the matching threshold $\gamma = \kappa \gamma _ { \mathrm { c o r r } }$ . Larger κ increases sensitivity to imperfect feature matches.

## E.6 FEATURE-STRENGTH MISMATCH

Our feature discrepancy primarily measures whether reference feature directions are recovered. Once a feature is well matched, differences in its strength receive little or no additional penalty. We therefore separately examine how the strengths of matched features vary across unlearning methods.

We take the ten strongest feature clusters of the full-data solution, ranked by their raw PCA eigenvalues. For each cluster and evaluated group M, we identify its best-matching component using the same correlation-based matching as in our main metric and record the raw eigenvalue $\hat { \lambda } _ { \mathcal { M } , i }$ of that component. We summarize the strength mismatch by

$$
\mathrm { N M S E } ( \mathcal { M } ) = \frac { \sum _ { i } ( \hat { \lambda } _ { \mathcal { M } , i } - \lambda _ { \mathrm { f u l l } , i } ) ^ { 2 } } { \sum _ { i } \lambda _ { \mathrm { f u l l } , i } ^ { 2 } } .
$$

![](images/04b94bacc44371d5c22c6f990340384460678e07b297778c3d9077fa117f13e0.jpg)  
(a) κ = 1

![](images/020c16688701de09ea84694a888cffce7243e797f6df77e1558be399a8b8e1b1.jpg)  
(b) $\kappa = 1 . 5$

![](images/1ac65024c0341cf5980bef20c8b7047826ecca0188488cb59d7b56ba1df7a085.jpg)  
(c) $\kappa = 2$ (default)

![](images/d56dbb25153a3ab2e9585162a03def8b47d144bc9d18660fd6489bf77d2f5742.jpg)  
(d) $\kappa = 2 . 5$

![](images/f853d47872c833eb8b341c5e33627762840d621de5499e455fb027ffcd9aa53c.jpg)  
(e) $\kappa = 3$

Figure 12: Sensitivity of the relationship between feature discrepancy and downstream accuracy on Flowers-102 to the matching threshold $\gamma = \kappa \gamma _ { \mathrm { c o r r } }$ . The low-sensitivity setting $\kappa = 1$ compresses the discrepancies, while the qualitative trend is preserved across larger values of κ.  
![](images/c2d61581daabbd8390e530869c65c1a3f49457ba2d3bae27c848d0f52e991f5a.jpg)  
(a) 10 models

![](images/0046b90767c3b1411b153510265e3f01454d68457948a8ab02d0fbad0d67390f.jpg)  
(b) 20 models

![](images/3f0218ff08e4479756971ef51eb92db971d1784735f08b9dcd49caad01d77a3b.jpg)  
(c) 30 models  
Figure 13: Sensitivity to the number of models used to construct each feature solution for ResNet-56 on CIFAR-100 under class-wise deletion. Larger groups reduce the discrepancy between independent retraining groups, while the broad relative structure of the unlearning methods is preserved despite some method-level variation.

Figure 14 shows that matching a feature direction does not imply matching its strength. Several unlearning methods exhibit substantial eigenvalue mismatch, which can become considerably larger on the forget distribution than on the standard test distribution. The effect is method- and settingdependent, but it demonstrates that feature identity alone does not fully characterize the representation: even recovered directions can appear with substantially different strengths.

## F RELATION TO LINEAR CKA

Our feature comparison is closely related to linear CKA. For centered representations with Gram decompositions

$$
K _ { A } = \sum _ { i } \lambda _ { i } ^ { A } u _ { i } ^ { A } ( u _ { i } ^ { A } ) ^ { \top } , \qquad K _ { B } = \sum _ { j } \lambda _ { j } ^ { B } u _ { j } ^ { B } ( u _ { j } ^ { B } ) ^ { \top } ,
$$

![](images/96dee052a5f98ef9b0f740a4ce2a1fdb12545ff77e8b049edb0db59d5b7f3120.jpg)  
(a) ViT-Tiny

![](images/0da9c4f5f528c8cdde278d093b12799337d24d080754e6ebcf9b6a05e5a0c1ad.jpg)  
(b) ResNet-56  
Figure 14: Feature-strength mismatch for the ten strongest full-data feature clusters. For each ref erence cluster, we match the closest component in each evaluated group and compare its raw PCA eigenvalue with the full-data reference. Bars show normalized mean-squared error on the test and forget distributions; lower values indicate better agreement in feature strength.

linear CKA can be written as

$$
\mathrm { C K A } ( A , B ) = \frac { \sum _ { i , j } \lambda _ { i } ^ { A } \lambda _ { j } ^ { B } \big ( ( u _ { i } ^ { A } ) ^ { \top } u _ { j } ^ { B } \big ) ^ { 2 } } { \sqrt { \sum _ { i } ( \lambda _ { i } ^ { A } ) ^ { 2 } } \sqrt { \sum _ { j } ( \lambda _ { j } ^ { B } ) ^ { 2 } } } .
$$

Because the response directions are centered and normalized, $( u _ { i } ^ { A } ) ^ { \top } u _ { j } ^ { B }$ is their correlation. Thus, CKA aggregates the same basic ingredients underlying our comparison—feature alignment and strength—but compresses them into a single symmetric similarity score.

Our metric instead retains directional and group-level structure. $D _ { \mathrm { f e a t } } ( \mathcal { R } , \mathcal { M } )$ asks which recurring features of a reference solution R are poorly represented by M. Directionality is particularly important for unlearning: comparing full data and retraining in both directions separates features that disappear after deletion from features that emerge only under retraining. Our analysis shows that this distinction matters, since successful unlearning may require not only removing features of the full-data solution but also recovering features newly selected by retraining. A symmetric aggregate such as CKA cannot distinguish these two changes.

The group-level construction is also important because individual retrained models need not provide a stable reference. Figure 15 illustrates this effect. For ViT-Tiny, independently retrained models have substantially higher CKA with the retraining reference than any unlearning method. For ResNet-56, however, pairwise CKA between independent retraining runs is itself much lower, and AMUN attains comparable or slightly higher similarity to the chosen retraining reference. A pairwise similarity score can therefore give the appearance of matching retraining even when the retraining solution varies substantially across runs. Our comparison instead identifies features that recur across the retraining ensemble and separately measures which of them are recovered by another group.

## G FEATURE INJECTION AND ABLATION

## G.1 EXPERIMENT DETAILS

We provide additional details for the feature interventions used in Section 6. The goal is to directly correct the retraining-relative feature mismatch identified by our comparison and test whether subsequent behavior moves toward retraining. We use two complementary operations: ablation removes feature responses present in the unlearned solution but poorly represented by retraining, while injection adds responses corresponding to retraining features that are poorly represented by the unlearned solution. Applying the two operations together yields substitution. All interventions modify the full penultimate-layer representation; PCA components are used only to identify the feature responses to be removed or added.

From feature clusters to seed-specific responses. Feature clusters are defined across model groups, whereas interventions are applied to individual models. We therefore pair each unlearned model with the retrained model of the same seed and identify, in each model, the top-K PCA components belonging to the selected clusters. For a component i, let

![](images/5162ecca9160d4be02de8048e984cb1a33fdfad10e06b5408789673a2dc81a5c.jpg)  
(a) ViT-Tiny

![](images/e4343d9a69462b1e1f8aea93bc6d73d5727522122a545ceecf4ead0536449304.jpg)  
(b) ResNet-56  
Figure 15: Linear CKA with a retrained reference model. Independent retraining runs are highly similar for ViT-Tiny, whereas pairwise retraining similarity is substantially lower for ResNet-56; in the latter case, AMUN attains CKA comparable to or higher than an independent retraining run. This illustrates why pairwise representation similarity can be misleading when retraining varies across runs.

$$
r _ { i } : = \widetilde \Phi v _ { i } = \sigma _ { i } u _ { i }
$$

denote its centered response across the intervention inputs. Since

$$
\widetilde { \Phi } = \sum _ { i } r _ { i } \boldsymbol { v } _ { i } ^ { \intercal } ,
$$

$r _ { i } v _ { i } ^ { \top }$ is the rank-one contribution of this component to the centered representation. Thus, intervention on a selected cluster acts through these seed-specific contributions: ablation removes those of the corresponding unlearned components, while injection uses the responses of the corresponding components from the paired retrained model. The top-K PCA components are used only to identify the features to modify; the interventions themselves act on the full representation rather than on a truncated PCA reconstruction.

Feature ablation. For a selected unlearned component i, we remove its response by modifying the representation as

$$
\Phi ^ { \mathrm { a b l } } = \Phi ^ { \mathrm { u n l } } - r _ { i } ^ { \mathrm { u n l } } \big ( v _ { i } ^ { \mathrm { u n l } } \big ) ^ { \top } .
$$

Since $r _ { i } ^ { \mathrm { u n l } }$ is centered, the representation mean is unchanged, and therefore

$$
\widetilde { \Phi } ^ { \mathrm { a b l } } = \widetilde { \Phi } ^ { \mathrm { u n l } } - r _ { i } ^ { \mathrm { u n l } } \big ( v _ { i } ^ { \mathrm { u n l } } \big ) ^ { \top } .
$$

For a set A of selected components, orthonormality of the PCA directions then gives

$$
\widetilde { \Phi } ^ { \mathrm { a b l } } ( \widetilde { \Phi } ^ { \mathrm { a b l } } ) ^ { \top } = \widetilde { \Phi } ^ { \mathrm { u n l } } ( \widetilde { \Phi } ^ { \mathrm { u n l } } ) ^ { \top } - \sum _ { i \in \cal A } r _ { i } ^ { \mathrm { u n l } } ( r _ { i } ^ { \mathrm { u n l } } ) ^ { \top } .
$$

Thus, ablation removes exactly the selected feature contributions.

Feature injection. For a selected component $j$ of the paired retrained model, we add its response as a new coordinate of the unlearned representation:

$$
\Phi ^ { \mathrm { i n j } } = \left[ \Phi ^ { \mathrm { u n l } } , r _ { j } ^ { \mathrm { r e t r } } \right] .
$$

Since $r _ { j } ^ { \mathrm { r e t r } }$ is centered,

$$
\widetilde { \Phi } ^ { \mathrm { i n j } } = \left[ \widetilde { \Phi } ^ { \mathrm { u n l } } , r _ { j } ^ { \mathrm { r e t r } } \right] .
$$

Thus, for a set I of injected components,

$$
\widetilde { \Phi } ^ { \mathrm { i n j } } ( \widetilde { \Phi } ^ { \mathrm { i n j } } ) ^ { \top } = \widetilde { \Phi } ^ { \mathrm { u n l } } ( \widetilde { \Phi } ^ { \mathrm { u n l } } ) ^ { \top } + \sum _ { j \in \mathcal { I } } r _ { j } ^ { \mathrm { r e t r } } ( r _ { j } ^ { \mathrm { r e t r } } ) ^ { \top } .
$$

Hence, injection adds exactly the selected retraining feature contributions without altering the existing unlearned ones.

Feature substitution. When ablation and injection are combined, we use an ablated PCA direction to carry an injected retraining response. After ablating component i, the centered representation has zero response along $v _ { i } ^ { \mathrm { u n l } }$ . We can therefore inject a retraining response $r _ { j } ^ { \mathrm { r e t r } }$ along this direction, replacing

$$
r _ { i } ^ { \mathrm { u n l } } \bigl ( v _ { i } ^ { \mathrm { u n l } } \bigr ) ^ { \top } \quad \mathrm { w i t h } \quad r _ { j } ^ { \mathrm { r e t r } } \bigl ( v _ { i } ^ { \mathrm { u n l } } \bigr ) ^ { \top } .
$$

By orthogonality of the PCA directions, this replaces the corresponding Gram contribution $r _ { i } ^ { \mathrm { u n l } } ( r _ { i } ^ { \mathrm { u n l } } ) ^ { \top }$ with $r _ { i } ^ { \mathrm { r e t r } } ( r _ { i } ^ { \mathrm { r e t r } } ) ^ { \top }$ . If there are more injected than ablated components, the remaining responses are appended as new dimensions; excess ablations are simply removed.

Intervention levels. The operations above specify how we modify a selected feature response. To test whether behavior changes progressively as more of the measured retraining-relative mismatch is corrected, we apply them to nested sets of feature clusters.

The two directional discrepancies determine which clusters are eligible for intervention: retraining clusters contributing to $D _ { \mathrm { f e a t } } ( \mathrm { r e t r } , \mathcal { M } )$ are candidates for injection, while unlearned clusters contributing to $D _ { \mathrm { f e a t } } ( \mathcal { M } , \mathrm { r e t r } )$ are candidates for ablation.

For each direction, we assign the eligible clusters a fixed random ordering and construct four nested intervention levels. Level $\bar { k } \in \{ 1 , 2 , \bar { 3 } , 4 \}$ contains the shortest prefix whose cumulative contribution reaches at least $k / 4$ of the total directional discrepancy, with level 4 containing all eligible clusters. Thus,

$$
C _ { 1 } \subseteq C _ { 2 } \subseteq C _ { 3 } \subseteq C _ { 4 } , \qquad D _ { 1 } \subseteq D _ { 2 } \subseteq D _ { 3 } \subseteq D _ { 4 } ,
$$

where $C _ { k }$ and $D _ { k }$ denote the retraining and unlearned clusters selected for injection and ablation, respectively.

Control interventions. To distinguish correction of the identified feature mismatch from generic changes of comparable form, we replace the targeted retraining responses with random responses. Each direction is sampled as $w \sim \mathcal { N } ( 0 , I )$ and normalized to unit norm in the full activation space; its response is obtained by projecting the centered unlearned-model activations onto it, with no additional variance or norm matching. For downstream transfer, we keep the same number of injected dimensions but use random response vectors instead of the selected retraining features. For partial relearning, we keep the targeted ablations fixed and replace only the injected retraining responses by random ones. Thus, the control preserves the amount of feature removal while changing which responses are added.

Evaluation after intervention. After each intervention, we freeze the modified representation and train a fresh prediction head using the same protocol as in Section 6: on the downstream task for transfer experiments, and on the retain set together with half of the forget examples for partial relearning. We also recompute the feature discrepancy on the intervened representation itself by centering it, recomputing its PCA decomposition, and applying the same feature-comparison procedure.

## G.2 ADDITIONAL EXPERIMENTS

We repeat the feature intervention experiments from Section 6 for LTD, AMUN and NegGrad+. As in the main text, we progressively substitute mismatched features with their retraining counterparts or inject missing retraining features, and evaluate the resulting representations under downstream transfer and partial relearning.

LTD. For LTD, the intervention produces the same qualitative behavior as in the main-text experiments: progressively correcting retraining-relative feature mismatch moves subsequent performance toward the retraining baseline across the evaluated settings. Figure 16 reports the corresponding intervention trajectories.

![](images/68bac196b296ce0a224e66cbdcd77e8ad7f79faa434ece6428e86f0891398f24.jpg)

![](images/f6e00e0cf60c0dd4bb0ea0791daef73ace306ba44b7c4ae537a29d448cd7ae35.jpg)

![](images/13ed446a6dcef439b3433227bce883e5106e0eb7fd37961cc0b78dcc11a66d11.jpg)  
Figure 16: Behavior under direct correction of retraining-relative feature mismatch for LTD. Progressively correcting the identified mismatch moves behavior toward retraining on Flowers-102, Oxford-IIIT Pets, and partial relearning.

AMUN. AMUN provides a complementary case in the partial-relearning experiment. Its initial deviation from retraining has the opposite sign, suggesting that missing retraining features may be more important here than removing AMUN-specific ones. Consistent with this hypothesis, targeted substitution moves relearning behavior toward retraining, whereas replacing the removed features with random directions does not. Figure 17 shows the corresponding trajectories.

![](images/2e3cf9efe640d35bcdd9feb7e49a6791923bc840da6e41ebb4dd7fd30d4bd1de.jpg)

![](images/6e656830aca354922e33e28599b1dfb39976292218b670b4db5f458b898bcb0b.jpg)

![](images/f086fd671e6621ca3732f0bec2cc87fb26ab7f00504e234ead2c0c96a93960db.jpg)  
Figure 17: Behavior under direct correction of retraining-relative feature mismatch for AMUN. Correcting the identified mismatch moves downstream performance toward retraining on Flowers-102 and Oxford-IIIT Pets. Under partial relearning, targeted substitution reduces the deviation from retraining, while the corresponding random-injection control increases it.

NegGrad+. NegGrad+ shows the clearest intervention trajectory across the three settings. Progressively correcting the identified feature mismatch moves behavior toward retraining on Flowers-102, Oxford-IIIT Pets, and partial relearning. On the downstream tasks, injecting missing retraining features alone is already sufficient to recover much of the gap. In contrast, the corresponding random-injection control does not show the same systematic improvement, supporting the claim that the effect depends on restoring the identified retraining features. Figure 18 shows the corresponding trajectories.

Results on held-out forget set. As an additional robustness check, we repeat the partial-relearning interventions on a held-out forget subset not used to construct the intervention. Figure 19 illustrates that SalUn, LTD, AMUN, and NegGrad+ show the same qualitative behavior as on the full forget set, with targeted correction again moving behavior toward retraining.

Finally, we repeat the SalUn intervention experiment under the blue-green deletion setting (Figure 20). The same qualitative behavior is observed, with targeted correction moving the adapted model toward retraining.

![](images/a2896a3f3bbaf0a134f7279cec59b4fcc6004be0ccba017231fed0130ef9916b.jpg)

![](images/ee49169eecf852256f7a89d0fac5d376638757a8193e3c7ab60933ddd518e094.jpg)

![](images/f1b979ec0be945ec9e76949b374f49b145d72021ff03dedde9cc99b96438a73d.jpg)  
Figure 18: Behavior under direct correction of retraining-relative feature mismatch for NegGrad+. Progressively correcting the identified mismatch moves behavior toward retraining on Flowers-102, Oxford-IIIT Pets, and partial relearning.

![](images/5997582913cb8ad920bc942c4b1f70d776801f2335ce4b037c08794ac9aa734e.jpg)  
(a) SalUn

![](images/bb80ad5dea0380c1d5e00a34a386cd32e4a5a51b9cb7fe9e42951e21897132b3.jpg)  
(b) AMUN

![](images/bc4c525ab5cd55629d1a87260980e5a912c7f80874ea9c4acd3431ece7714105.jpg)  
(c) NegGrad+

![](images/8eb08d241697276ded8babc35875c2f0119b3b751337e244b3812240daeae91f.jpg)  
(d) LTD  
Figure 19: Partial-relearning intervention trajectories on a held-out forget subset for SalUn, AMUN, NegGrad+, and LTD.

ResNet-56/CIFAR-100 (bg), Oxford-IIIT Pets  
![](images/fa54ab30cc71f581ab67366eddfc05cb9ea6ef7aa12e4aa6819b8b0899741b0d.jpg)  
(a) Oxford-IIIT Pets

ResNet-56/CIFAR-100 (bg), Flowers-102  
![](images/418238d1164985a373305b241adf85cec750abb06bdf1de4fd2a8ad1cd3810a3.jpg)  
(b) Flowers-102

![](images/d95aca389e5b0d824f4bcc26aad1ccd2ac2c4247751b9c8a0c56726afe07a615.jpg)  
(c) Forget set

![](images/2165a3e56aa7d154fefd47ac17c941890338b3ecf012e236b3b292bfff32ed85.jpg)  
(d) Held-out forget subset  
Figure 20: Intervention trajectories for SalUn under the blue-green deletion setting, evaluated on downstream transfer and partial relearning.