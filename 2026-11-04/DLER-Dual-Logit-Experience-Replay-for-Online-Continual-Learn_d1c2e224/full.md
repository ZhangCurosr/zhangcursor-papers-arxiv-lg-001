# DLER: Dual-Logit Experience Replay for Online Continual Learning

Huiwei Lin, Shanshan Feng, and Kong Aik Lee, Senior Member, IEEE

Abstract—Online continual learning aims to continuously train neural networks on a non-stationary data stream in a single pass. Rehearsal-based methods alleviate catastrophic forgetting by replaying stored samples, but the imbalance between current and previous data often leads to predictions biased toward current classes. Many approaches have been proposed to overcome the forgetting problem by correcting the bias; however, they still need to be improved in an online fashion. By decomposing the dotproduct logits into an angle factor and a norm factor, our analysis reveals that the angle factor efficiently captures novel knowledge but is particularly susceptible to prediction bias, whereas the norm factor helps preserve historical knowledge. Based on these observations, we propose Dual-Logit Experience Replay (DLER), which exploits dual-logits for both model optimization and memory management. Specifically, DLER learns current samples using cosine logits and incorporates additional dot-product logits when replaying previous samples to mitigate prediction bias. To strengthen dual-logit learning under limited replay memory, an auxiliary supervised contrastive objective promotes angular compactness within each class, extending corrections learned from limited replay samples to other samples of the same class. Meanwhile, we introduce a memory update strategy that preferentially retains samples for which dot-product prediction assigns a higher probability to the correct class than cosine prediction, aligning sample selection with dual-logit training. Extensive experiments on three datasets demonstrate the effectiveness of DLER, particularly under limited memory budgets.

Index Terms—Online Continual Learning, Rehearsal-based Methods, Cosine Logits, Dot-product Logits.

## I. INTRODUCTION

At present, deep neural networks [1]–[3] still lack the conventional ability of continuous learning like human beings. If the model adapts to new data through fine-tuning, it will significantly degrade the performance on old data. Such a phenomenon is named catastrophic forgetting (CF) [4]. Because solving this problem is highly significant, continual learning [5]–[7] has been proposed to train models to accumulate knowledge without forgetting. In particular, online continual learning (OCL) considers a practical setting in which models must mitigate CF while learning continuously from a potentially infinite, non-stationary data stream, with each incoming sample accessible only once.

Rehearsal-based methods have demonstrated strong performance in OCL [8]. These methods maintain a fixed-size episodic memory buffer to store previous samples and replay them while learning from current data [9]–[11]. However, the imbalance between current data and limited previous samples can bias predictions toward new classes, degrading performance on previously learned classes. Although existing approaches mitigate forgetting by correcting this bias, three challenges remain in online settings: 1) Real-time ability. Some methods [12]–[14] require additional training steps or post-processing, hindering real-time prediction. 2) Flexibility. Some methods [15], [16] rely heavily on explicit boundaries between old and new classes, and their correction mechanisms cannot be readily extended from the predictor to the feature extractor. 3) Adjustability. Other methods [16], [17] offer limited control over the strength of bias correction, potentially restricting model generalization. Hence, can a straightforward and efficient approach address prediction bias in OCL while simultaneously meeting these three challenges?

To answer this question, we examine existing rehearsalbased methods and empirically find that the biased predictions (dot-product logits) mainly change in two parts, as shown in Figure 1(a). Inspired by [18], we express dot-product logits as $w h + b = \| w \| \| h \| \cos ( w , h ) + b .$ , separating the norm and angle factors while retaining the additive bias term. Data imbalance leads to unbalanced competition during gradient propagation, contributing to CF and affecting both factors. Further experiments reveal that these two factors act differently during the training process. On the one hand, the angle factor suffers from substantial prediction bias but efficiently captures novel knowledge from current samples when used independently as cosine logits. On the other hand, the norm factor, which is abandoned by existing works [16], [19], is helpful in keeping historical knowledge of previous samples.

These observations suggest that it is feasible to leverage the norm factor to balance old and new knowledge and address the bias issue. Specifically, dot-product logits incorporate both norm and angular information, making them suitable for consolidating historical knowledge during replay. In contrast, cosine logits rely on angular information and facilitate the acquisition of new knowledge from current samples. Different from existing studies (Figure 1(b)) that calculate logits in the same way, we try to (Figure 1(c)) learn current samples by cosine logits and further replay previous samples by dotproduct logits. As illustrated in Figure 1(d), a sample $h _ { p }$ will be mistakenly classified to current class c by cosine logits that satisfies cos $( w _ { c } , h _ { p } ) > \cos ( w _ { p } , h _ { p } )$ . However, with the help of the norm factor, the model can correctly classify $h _ { p }$ to the previous class $p$ by dot-product logits that satisfy $\| \hat { \boldsymbol { h } } _ { p } \| \| \boldsymbol { w } _ { c } \| c o s ( \boldsymbol { w } _ { c } , \boldsymbol { h } _ { p } ) + b _ { c } < \| \boldsymbol { h } _ { p } \| \| \boldsymbol { w } _ { p } \| c o s ( \boldsymbol { w } _ { p } , \boldsymbol { h } _ { p } ) + b _ { p } .$ Empirical experiments demonstrate that this approach effectively solves the bias problem and achieves better performance.

![](images/fd6c59b888cf687d974acfdd97a18803efc27df722c9a34f62ed4b24545b52f1.jpg)  
Fig. 1. Illustration of our work. (a) The dot-product logits can be decomposed into a norm factor and an angle factor; (b) Existing studies, which learn all samples by calculating logits in the same way; (c) Different from existing studies, the model in our work learns current samples only by the angle factor, while replays previous samples by both the angle factor and the norm factor. (d) By leveraging the norm factor, the model can correctly classify samples.

Experience Replay (DLER), which alleviates CF in OCL by exploiting the complementary properties of cosine and dotproduct logits to address prediction bias. Its learning process comprises three components: 1) The learning component trains current samples using cosine logits to acquire novel knowledge through angular information. 2) The replaying component incorporates additional dot-product supervision for buffered samples, leveraging norm and angular information to consolidate historical knowledge. A weighting parameter controls the contribution of dot-product supervision, allowing adjustment of the balance between current learning and historical replay. 3) The testing component uses dot-product logits to predict over all observed classes, retaining the contribution of norm information without introducing additional modules.

To further exploit the complementary roles of dual-logits under limited replay memory, we strengthen dual-logit training and extend it to memory management. Specifically, an auxiliary supervised contrastive loss promotes angular compactness within classes and separation between classes, providing more consistent angular evidence to complement norm information in dot-product logits. This geometry can help corrections learned through dot-product supervision generalize from limited replay samples to unseen samples of the same historical class. Moreover, we align memory selection with dual-logit training by scoring samples using the latest model according to the positive probability advantage of dot-product prediction over cosine prediction for the correct class. Within a fixed memory budget, class quotas regulate allocation, while score-based replacement preferentially retains samples exhibiting this advantage. This helps memory better support dual-logit complementarity when few samples can be retained per class. Together, these extensions exploit dual logits in model optimization and memory management, while retaining direct inference (Real-time ability), applicability to the feature extractor (Flexibility), and controllable bias correction (Adjustability).

The main contributions are summarized as follows:

• We theoretically analyze the characteristics of dot-product logits and cosine logits, discovering that their coupling is beneficial. To the best of our knowledge, this work is the first to combine these two logits for the OCL.

• We propose DLER, a rehearsal-based framework that learns current samples using cosine logits and incorporates additional dot-product supervision when replaying previous samples.

• We further enhance dual-logit learning through an auxiliary supervised contrastive loss that promotes feature compactness and a memory update strategy that exploits the predictive complementarity of both logits.

• We conduct extensive experiments on Split CIFAR100, Split MiniImageNet, and Split TinyImageNet to evaluate the effectiveness of DLER. The code is publicly available at https://github.com/FelixHuiweiLin/UER.

This work extends our conference version presented in [20] in three aspects: (i) We introduce an auxiliary supervised contrastive objective to promote angular compactness and class separation, helping dual-logit corrections generalize beyond limited replay samples. (ii) We extend dual-logits to memory management through advantage-based scoring and replacement under balanced class quotas, prioritizing samples with a positive dot-product probability advantage for the correct class. (iii) We replace Split CIFAR10 with the more challenging Split TinyImageNet and conduct additional ablation studies and analyses to examine the effectiveness and mechanisms of the proposed components.

## II. RELATED WORK

## A. Continual Learning

Continual learning, also referred to as lifelong or incremental learning, enables models to acquire knowledge from sequentially arriving data while preserving previously learned knowledge. Its central goal is to balance adaptation to new data with resistance to catastrophic forgetting. Existing approaches can be categorized into three groups. 1) Architecturebased methods [21] mitigate CF through dynamic or static network designs. Dynamic networks progressively expand their capacity as learning proceeds, whereas static networks retain a fixed architecture and selectively allocate parameters. 2) Regularization-based methods [22] introduce additional constraints into the training objective to preserve historical knowledge, typically by penalizing changes to important parameters or applying knowledge distillation. 3) Rehearsalbased methods [23], [24] retain a subset of historical samples or generate synthetic samples, which are revisited alongside new data to alleviate forgetting.

Our proposed DLER belongs to the rehearsal-based category and focuses on the online setting. It exploits the complementary properties of dot-product and cosine logits for model optimization and memory management.

## B. Online Continual Learning

Experience Replay (ER) [25] serves as the foundation for many OCL methods, which can be broadly divided into sample management and model update strategies. Sample management strategies focus on selecting informative incoming samples for storage [26], [27] or identifying valuable samples in the memory buffer for replay [28], [29]. More recently, GRASP [30] improves rehearsal efficiency by initially prioritizing prototypical samples and progressively shifting toward more difficult ones, thereby reducing the number of model updates needed for continual learning.

Model update strategies refine the training objectives or optimization procedures for learning from incoming and buffered samples [14], [20], [31]–[34]. SS-IL [15] and ER-ACE [16] apply distinct cross-entropy objectives to these two groups of samples to alleviate prediction bias. PCR [11] examines their interaction through gradient propagation, whereas LODE [33] separates the loss functions associated with new and old classes. More recent work explores alternative optimization formulations for rehearsal-based OCL. GEC [35] formulates ER as an ϵ-constraint optimization problem and adapts the update direction based on an allowable tolerance for memory loss. LOR [36] mitigates conflicts between incoming and replay gradients by examining lookahead model states and optimizing a smooth worst objective across their trade-offs.

Building on UER [20], DLER extends the use of dual logits from model optimization to memory management. It introduces a supervised contrastive objective to explicitly encourage intra-class compactness and inter-class separation. Furthermore, its memory update strategy jointly uses the probability distributions produced by dot-product and cosine logits to score candidate samples and prioritize their retention.

## III. PRELIMINARY

OCL divides a data stream into a sequence of mini learning batches as $\mathcal { D } = \{ B _ { t } \} _ { t = 1 } ^ { T }$ , where $\boldsymbol { B } _ { t } = \{ \mathcal { X } _ { t } \times \mathcal { Y } _ { t } \}$ contains the samples $\mathcal { X } _ { t }$ and corresponding labels Y<sub>t</sub>. All learned classes are denoted as $\mathcal { C } _ { t }$ up to step t. The neural network is made up of a feature extractor $\begin{array} { r } { h = h _ { \Phi } ( { \pmb x } ) } \end{array}$ and a predictor $f ( h ) =$ $\bar { W h } ^ { T } + b$ , where $^ { h }$ is non-negative by Relu, $W = [ { \pmb w } _ { i } ]$ , and $b = [ b _ { i } ]$ . Since the output of predictor is dot-product logits, the categorical prediction of sample x is $p ^ { d o t } ( \pmb { x } ) = [ p _ { i } ^ { d o t } ]$ , and $p _ { i } ^ { d o t }$ is the probability that a sample belongs to class $i \in { \mathcal { C } } _ { t } \colon$

$$
p _ { i } ^ { d o t } = \frac { e x p ( { \pmb w } _ { i } h _ { \pmb \Phi } ( { \pmb x } ) ^ { T } + b _ { i } ) } { \sum _ { j \in \mathcal { C } _ { t } } e x p ( { \pmb w } _ { j } h _ { \pmb \Phi } ( { \pmb x } ) ^ { T } + b _ { j } ) } .\tag{1}
$$

During training, the model receives the current mini-batch $B t .$ with each sample appearing only once in the data stream. Without replaying, the training objective is

$$
\begin{array} { r } { L _ { o c l } = E _ { ( \pmb { x } , \pmb { y } ) \sim \mathcal { B } _ { t } } [ l ( p ^ { d o t } ( \pmb { x } ) , \pmb { y } ) ] , } \end{array}\tag{2}
$$

where $l ( \cdot )$ is the cross-entropy loss function.

ER [25], as a classic rehearsal-based method, allocates a memory buffer M to store part of previous samples, and selects them as $p _ { { \mathcal { M } } }$ to replay with current samples. Thus, the objective function can be changed to

$$
L _ { e r } = E _ { ( \pmb { x } , \pmb { y } ) \sim \pmb { B _ { t } } \cup \pmb { B _ { \mathcal { M } } } } [ l ( p ^ { d o t } ( \pmb { x } ) , \pmb { y } ) ] .\tag{3}
$$

LUCIR [19] improves replay-based learning by introducing a cosine-normalized classifier. The resulting class probability distribution is denoted by $p ^ { \mathrm { c o s } } ( { \pmb x } ) = [ p _ { i } ^ { \mathrm { c o s } } ]$ , and the corresponding classification objective is

$$
L _ { l u c i r } = E _ { ( \pmb { x } , \pmb { y } ) \sim \mathcal { B } _ { t } \cup \mathcal { B } _ { \mathcal { M } } } [ l ( p ^ { c o s } ( \pmb { x } ) , \pmb { y } ) ] ,\tag{4}
$$

where

$$
p _ { i } ^ { c o s } = \frac { e x p ( c o s ( { \pmb w } _ { i } , h _ { \Phi } ( { \pmb x } ) ) \cdot \gamma ) } { \sum _ { j \in \mathcal { C } _ { 1 : t } } e x p ( c o s ( { \pmb w } _ { j } , h _ { \Phi } ( { \pmb x } ) ) \cdot \gamma ) } .\tag{5}
$$

γ is a scale parameter and is usually set to 10.

## IV. DUAL-LOGIT EXPERIENCE REPLAY

Inspired by the preceding analyses in Appendix VII, we propose Dual-Logit Experience Replay (DLER), which exploits cosine and dot-product logits for model optimization and memory management in OCL. Algorithm 1 summarizes this procedure, whose three components are described below.

## A. Online Continual Training

1) Dual-logits training: DLER uses the cosine and dotproduct predictions defined in Eqs. (5) and (1) to construct complementary objectives. For each sample, we use an original view $\pmb { x } ^ { ( 1 ) } = \pmb { x }$ and an augmented view ${ \pmb x } ^ { ( 2 ) } = \mathcal { T } ( { \pmb x } )$ , where T denotes random data augmentation. For compactness, the loss averaged over two views $s \in$ {cos, dot} is written as

$$
\ell _ { s } ( \pmb { x } , y ) = \frac { 1 } { 2 } \sum _ { v = 1 } ^ { 2 } l \big ( p ^ { s } ( \pmb { x } ^ { ( v ) } ) , \pmb { y } \big ) .\tag{6}
$$

The learning component supervises current samples through cosine logits, exploiting angular information to acquire novel knowledge. Its objective is

$$
L _ { c } = \frac { 1 } { | \mathcal { B } _ { t } | } \sum _ { ( \pmb { x } , y ) \in \mathcal { B } _ { t } } \ell _ { \mathrm { c o s } } ( \pmb { x } , y ) .\tag{7}
$$

The replaying component incorporates additional dot-product supervision to consolidate historical knowledge by

$$
L _ { p } = \frac { 1 } { | \mathcal { B } _ { \mathcal { M } } | } \sum _ { ( \pmb { x } , \pmb { y } ) \in \mathcal { B } _ { \mathcal { M } } } \Big [ \alpha \ell _ { \mathrm { d o t } } ( \pmb { x } , \pmb { y } )\tag{8}
$$

where $\alpha ~ \in ~ [ 0 , 1 ]$ controls the relative contribution of dotproduct supervision. The objective of dual-logits training is

$$
{ \cal L } _ { d u a l } = { \cal L } _ { c } + { \cal L } _ { p }\tag{9}
$$

2) Dual-logits enhancement: To enhance the dual-logits mechanism, we introduce a supervised contrastive loss [14] over the joint current and replay samples. Specifically, $B _ { t }$ and $p _ { { \mathcal { M } } }$ are concatenated together with their labels, and both original and augmented views are processed by the shared feature extractor. Let $\mathcal { T } _ { t }$ index all resulting views, and let $\boldsymbol { h } _ { i }$ denote the feature of view i. For each anchor i, the comparison set is $\mathcal { A } ( i ) = \mathcal { T } _ { t } \setminus \{ i \}$ , and the positive set is ${ \mathcal { P } } ( i ) = \{ j \in { \mathcal { A } } ( i ) : y _ { j } = y _ { i } \}$ . The loss is

$$
L _ { \mathrm { s c } } = \mathrm { ~ - ~ } \frac { 1 } { | \mathcal { T } _ { t } | } \sum _ { i \in \mathcal { I } _ { t } } \frac { 1 } { | \mathcal { P } ( i ) | } \sum _ { j \in \mathcal { P } ( i ) } \log \frac { \exp \left( \cos ( h _ { i } , h _ { j } ) / \tau \right) } { \displaystyle \sum _ { a \in A ( i ) } \exp \left( \cos ( h _ { i } , h _ { a } ) / \tau \right) } ,\tag{10}
$$

where $\tau$ is the contrastive temperature.

Why use the dual-logits enhancement? Dual-logit training uses dot-product supervision during replay to help correct predictions biased toward current classes. However, the class weight norms are shared across samples, and their corrective effect depends on each sample’s angular relationships with the class weights. If same-class features are widely dispersed, a correction that benefits a replay sample may not benefit other samples of that class. Encouraging angular compactness can make these shared corrections more consistent across samples.

Consider a previous class p and a competing class c. For a feature h, define their dot-product margin as

$$
\begin{array} { r l } & { m _ { p c } ( \pmb { h } ) = \| \pmb { h } \| _ { 2 } \bigg [ \| \pmb { w } _ { p } \| _ { 2 } \cos ( \pmb { w } _ { p } , \pmb { h } ) } \\ & { \qquad - \ \| \pmb { w } _ { c } \| _ { 2 } \cos ( \pmb { w } _ { c } , \pmb { h } ) \bigg ] + b _ { p } - b _ { c } . } \end{array}\tag{11}
$$

A positive margin means that dot-product prediction favors class $p$ over class c, even when their cosine similarities have the opposite ordering. To examine how this margin transfers between samples within the same class, let $h _ { r }$ be the feature of a replay sample and $h _ { u }$ the feature of another sample from class p that is not included in the current replay mini-batch. Both features are evaluated under the same model. For nonzero features, define the normalized margin

$$
\widetilde { m } _ { p c } ( \boldsymbol { h } ) = \frac { m _ { p c } ( \boldsymbol { h } ) } { \lVert \boldsymbol { h } \rVert _ { 2 } } .\tag{12}
$$

Algorithm 1 Dual-Logit Experience Replay   
Input: Stream $\{ B _ { t } \} _ { t = 1 } ^ { T }$ , memory capacity M, replay size $b _ { m } ,$   
learning rate η, and parameters $\alpha , \lambda _ { \mathrm { s c } } , \gamma , \tau$   
Output: Network parameters θ and memory buffer M   
1: Initialize $\theta , \mathcal { M }  \varnothing , \mathcal { C }  \varnothing ,$ , and class counts $N _ { c } \gets 0$   
2: for $t = 1 , \dots , T$ do   
3: $\mathcal C  \mathcal C \cup \{ y : ( \pmb { x } , y ) \in \mathcal B _ { t } \}$   
4: Uniformly sample up to $b _ { m }$ examples without replace  
ment from M to obtain $p _ { { \mathcal { M } } }$   
5: Construct original and augmented views of current and   
replay samples   
6: Compute $L _ { c }$ using Eq. (7)   
7: if $\boldsymbol { B } _ { \mathcal { M } } \neq \boldsymbol { \mathcal { Q } }$ then   
8: Compute $L _ { p }$ using Eq. (8)   
9: Compute joint $L _ { \mathrm { s c } }$ using Eq. (10)   
10: $L _ { d l e r }  L _ { c } + L _ { p } + \lambda _ { \mathrm { s c } } L _ { \mathrm { s c } }$   
11: else   
12: $L _ { d l e r }  L _ { c }$   
13: end if   
14: $\pmb { \theta }  \pmb { \theta } - \eta \nabla _ { \pmb { \theta } } L$   
15: Update $\mathcal { M }$ and N using Algorithm 2   
16: end for   
17: Predict test labels using Eq. (17)

Their margin difference can be expressed as

$$
\begin{array} { l } { \widetilde { m } _ { p c } ( \boldsymbol { h } _ { u } ) - \widetilde { m } _ { p c } ( \boldsymbol { h } _ { r } ) } \\ { = ( w _ { p } - w _ { c } ) \left( \cfrac { \boldsymbol { h } _ { u } } { \| \boldsymbol { h } _ { u } \| _ { 2 } } - \cfrac { \boldsymbol { h } _ { r } } { \| \boldsymbol { h } _ { r } \| _ { 2 } } \right) ^ { T } } \\ { + ( b _ { p } - b _ { c } ) \left( \cfrac { 1 } { \| \boldsymbol { h } _ { u } \| _ { 2 } } - \cfrac { 1 } { \| \boldsymbol { h } _ { r } \| _ { 2 } } \right) . } \end{array}\tag{13}
$$

Applying the Cauchy–Schwarz inequality [37] gives

$$
\begin{array} { r l } & { \widetilde { m } _ { p c } ( \pmb { h } _ { u } ) \geq \widetilde { m } _ { p c } ( \pmb { h } _ { r } ) } \\ & { \qquad - \| \pmb { w } _ { p } - \pmb { w } _ { c } \| _ { 2 } \sqrt { 2 [ 1 - \cos ( \pmb { h } _ { u } , \pmb { h } _ { r } ) ] } } \\ & { \qquad -  | b _ { p } - b _ { c } | | \frac { 1 } { \| \pmb { h } _ { u } \| _ { 2 } } - \frac { 1 } { \| \pmb { h } _ { r } \| _ { 2 } } | . } \end{array}\tag{14}
$$

Equation (14) shows that, with the remaining quantities fixed, greater same-class cosine similarity reduces the angular discrepancy term. If the replay sample has a positive margin exceeding the two discrepancy terms, the other sample also has a positive margin against class c. The bias-dependent term vanishes when $b _ { p } = b _ { c }$ and becomes small when the feature norms are similar, while correct classification requires a positive margin against every competing class. These observations suggest that within-class angular compactness can support dual-logit correction by allowing previous samples with similar directions to benefit more consistently from the same norm factor. The supervised contrastive objective encourages this geometry using current and replay samples, potentially extending corrections learned from limited replay data to more samples of the same class.

Algorithm 2 Dual-Logit Memory Update   
Input: Updated model θ, memory M, mini-batch $B _ { t } ,$ ob  
served classes $\mathcal { C } _ { t } ,$ capacity M   
Output: Updated memory M   
1: Compute $\{ q _ { c } ^ { ( t ) } \} _ { c \in \mathcal { C } _ { t } }$ using Eq. (18)   
2: Score samples in M and $B _ { t }$ using Eq. (20)   
3: for each $c \in { \mathcal { C } } _ { t }$ do   
4: Retain the top min $\{ q _ { c } ^ { ( t ) } , | \mathcal { M } _ { c } | \}$ samples in $\mathcal { M } _ { c }$ by score   
5: end for   
6: for each $( { \pmb x } , y ) \in B _ { t }$ sequentially do   
7: if $q _ { y } ^ { ( t ) } = 0$ then   
8: continue   
9: end if   
10: if $| \mathcal { M } _ { y } | < q _ { y } ^ { ( t ) }$ then   
11: $\hat { \mathcal { M } } _ { y } \gets \mathcal { M } _ { y } \cup \{ ( \pmb { x } , y ) \}$   
12: else   
13: Select $( \pmb { x } ^ { - } , \pmb { y } )$ using Eq. (21)   
14: if $s ( \mathbf x , y ) > s ( \mathbf x ^ { - } , y )$ then   
15: Replace $( \pmb { x } ^ { - } , \pmb { y } )$ with $( { \pmb x } , y )$ using Eq. (22)   
16: end if   
17: end if   
18: end for

3) The overall objective: Based on the dual-logits training and its enhancement, the overall objective of online continual training is thus denoted as

$$
\begin{array} { r } { L _ { d l e r } = L _ { c } + L _ { p } + \lambda _ { \mathrm { s c } } L _ { \mathrm { s c } } , } \end{array}\tag{15}
$$

where $\lambda _ { \mathrm { s c } }$ controls the auxiliary loss. When memory is empty, only $L _ { c }$ is used. The model is updated by

$$
\pmb { \theta }  \pmb { \theta } - \eta \nabla _ { \pmb { \theta } } L ,\tag{16}
$$

where $\theta = \{ \Phi , W , b \}$ and $\eta > 0$ is the learning rate.

## B. Online Continual Testing

During testing, DLER uses the dot-product prediction defined in Eq. (1) to incorporate both angular and norm factors. For each test sample x, the predicted label is

$$
\hat { y } = \underset { c \in \mathcal { C } _ { t } } { \arg \operatorname* { m a x } } \quad p _ { c } ^ { d o t } ( \pmb { x } ) ,\tag{17}
$$

which is restricted to the classes observed up to step t.

## C. Memory buffer Management

1) Random memory retrieval: The memory buffer stores at most M original samples, regardless of the length of the data stream. Before each model update, up to $b _ { m }$ historical samples are retrieved uniformly without replacement to form $p _ { { \mathcal { M } } }$

2) Dual-logits memory update: After optimization, DLER updates the memory buffer using dual-logits. The update consists of balanced memory allocation, dual-logits scoring, and score-guided replacement, as summarized in Algorithm 2.

Balanced Memory Allocation. Let $K _ { t } = | \mathcal { C } _ { t } |$ denote the number of observed classes and $r _ { t } = M - K _ { t } \lfloor M / K _ { t } \rfloor$ the capacity after equal allocation. The quota for class c is

$$
q _ { c } ^ { ( t ) } = \left\lfloor \frac { M } { K _ { t } } \right\rfloor + \mathbb { I } [ \mathrm { r a n k } _ { t } ( c ) \leq r _ { t } ] ,\tag{18}
$$

where rank $_ t ( c )$ denotes the position of class c among the observed classes sorted by label index. The quotas sum to M. When newly observed classes reduce an existing class quota, we retain its highest-scoring samples up to the updated quota.

Dual-Logit Scoring. The scoring criterion follows from the complementary roles of cosine and dot-product logits in replay. Cosine predictions rely on angular information and may remain biased toward current classes. Dot-product predictions additionally incorporate norm information, which can increase the probability assigned to the correct previous class. We therefore prioritize samples that exhibit this positive probability advantage, aiming to preserve examples that support the correction mechanism of dual-logit training. For a labeled sample $( { \pmb x } , y )$ , the probability advantage is

$$
a ( { \pmb x } , y ) = \left[ p _ { y } ^ { \mathrm { d o t } } ( { \pmb x } ) - p _ { y } ^ { \mathrm { c o s } } ( { \pmb x } ) \right] _ { + } ,\tag{19}
$$

where $[ v ] _ { + } = \operatorname* { m a x } ( v , 0 )$ and both probabilities correspond to the true class $y .$ . We directly use this advantage as the score

$$
\begin{array} { r } { s ( \pmb { x } , y ) = a ( \pmb { x } , y ) . } \end{array}\tag{20}
$$

Since both probabilities lie in [0, 1], the score belongs to [0, 1].

Score-Guided Replacement. After optimization, we rescore the original samples in M and $B _ { t }$ using the updated model. For a sample $( { \pmb x } , y )$ , no insertion is performed if $q _ { y } ^ { ( t ) } = 0 .$ . If $| \mathcal { M } _ { y } | ~ < ~ q _ { y } ^ { ( t ) }$ , the sample is inserted directly. Otherwise, we select a sample stored in the memory buffer with the lowest score from the same class by

$$
( { \pmb x } ^ { - } , y ) \in \mathrm { ~ \ a r g \operatorname* { m i n } _ { \Sigma } ~ } s ( { \pmb x } ^ { \prime } , y ) ,\tag{21}
$$

with ties resolved uniformly at random. Replacement occurs only when the incoming sample has a higher score, as $s ( \mathbf x , y ) > s ( \mathbf x ^ { - } , y )$ , and

$$
\mathcal { M } _ { y } \gets \left( \mathcal { M } _ { y } \setminus \{ ( \boldsymbol { x } ^ { - } , \boldsymbol { y } ) \} \right) \cup \{ ( \boldsymbol { x } , \boldsymbol { y } ) \} .\tag{22}
$$

Otherwise, $\mathcal { M } _ { y }$ remains unchanged. This procedure uses the same predictive advantage to guide both quota reduction and sample replacement. Rescoring requires additional forward passes during training, which can be processed in minibatches, but introduces no additional inference modules or testtime computation. As summarized in Algorithm 1, memory retrieval precedes model optimization, and memory updates follow it.

Why use the dual-logits memory update? Dual-logit training incorporates additional dot-product supervision during replay, but reservoir sampling does not consider which samples benefit from this prediction mechanism. When only a few samples can be retained per class, we prioritize those for which dot-product prediction assigns a higher probability to the correct class than cosine prediction. Retaining such samples provides opportunities to exploit norm information during replay, potentially reducing the reliance on angular adjustments to preserve historical predictions. This may help limit interference with the angular learning of current samples through cosine supervision. Scoring with the latest model identifies this predictive advantage under competition among the currently observed classes, aligning limited replay memory with the dual-logit training mechanism.

TABLE I  
FINAL AVERAGE ACCURACY (%). BEST AND SECOND-BEST AVAILABLE RESULTS AMONG CONTINUAL LEARNING METHODS, EXCLUDING UER, ARE BOLDFACED AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td colspan="4">Split CIFAR100</td><td colspan="4">Split MiniImageNet</td><td colspan="4">Split TinyImageNet</td></tr><tr><td>100</td><td>200</td><td>500</td><td>1000</td><td>200</td><td>500</td><td>1000</td><td>2000</td><td>400</td><td>1000</td><td>2000</td><td>4000</td></tr><tr><td>IID</td><td colspan="4"> $1 7 . 3 \pm 0 . 8$ </td><td colspan="4"></td><td colspan="4"> $1 7 . 3 \pm { 1 . 7 }$ </td></tr><tr><td>IID++ [16]</td><td colspan="4"> $2 3 . 5 \pm 0 . 8$ </td><td colspan="4"> $1 8 . 2 \pm { 1 . 1 }$   $2 0 . 7 \pm 1 . 0$ </td><td colspan="4"> $1 9 . 1 \pm 1 . 3$ </td></tr><tr><td>Fine-tune</td><td></td><td> $5 . 9 \pm 0 . 2$ </td><td></td><td></td><td></td><td> $4 . 3 \pm 0 . 2$ </td><td></td><td></td><td></td><td> $4 . 3 \pm 0 . 2$ </td><td></td><td></td></tr><tr><td>ER [25]</td><td> $9 . 0 \pm 0 . 4$ </td><td>11.0 ± 0.8 14.5 ± 0.8</td><td></td><td> $1 7 . 6 \pm 0 . 9$ </td><td> $8 . 8 \pm 0 . 4$ </td><td></td><td>11.2 ± 0.6 13.4 ± 0.9</td><td>16.5 ± 0.9</td><td> $5 . 2 \pm 0 . 6$ </td><td> $7 . 6 \pm 0 . 5$ </td><td>10.1 ± 0.9</td><td> $1 2 . 7 \pm 1 . 1$ </td></tr><tr><td>GSS [26]</td><td> $8 . 5 \pm 0 . 8$ </td><td> $1 0 . 4 \pm 0 . 6$ </td><td> $1 4 . 6 \pm 1 . 3$ </td><td> $1 6 . 9 \pm 1 . 4$ </td><td> $8 . 5 \pm 0 . 7$ </td><td></td><td></td><td> $1 0 . 3 \pm 1 . 5 1 3 . 9 \pm 1 . 0 1 4 . 6 \pm 1 . 1 5 . 1 \pm 0 . 6$ </td><td></td><td>7.0 ± 0.5</td><td>10.1 ± 0.5</td><td>11.7 ± 0.6</td></tr><tr><td>MIR [28]</td><td> $8 . 9 \pm 0 . 5$ </td><td> $1 0 . 6 \pm 0 . 7$ </td><td> $1 4 . 8 \pm 0 . 7$ </td><td> $1 8 . 1 \pm 0 . 7$ </td><td> $8 . 8 \pm 0 . 6$ </td><td></td><td></td><td> $1 1 . 9 \pm 0 . 6 1 4 . 8 \pm 1 . 1 1 7 . 2 \pm 0 . 8$ </td><td>4.9 ± 0.4</td><td>7.3 ± 0.7</td><td>9.2 ± 1.3</td><td>11.3 ± 1.2</td></tr><tr><td>ASER [29]</td><td> $7 . 5 \pm 0 . 8$ </td><td>8.8 ± 0.6</td><td>13.0 ± 0.9</td><td> $1 6 . 1 \pm 1 . 1$ </td><td>7.0 ± 0.5</td><td></td><td></td><td> $1 0 . 5 \pm 1 . 1 1 3 . 8 \pm 0 . 9 1 6 . 1 \pm 1 . 1$ </td><td>4.9 ± 0.3</td><td>6.9 ± 0.7</td><td>8.8 ± 0.9</td><td>12.3 ± 0.8</td></tr><tr><td>GMED [27]</td><td> $8 . 2 \pm 1 . 9$ </td><td> $1 0 . 8 \pm 2 . 6$ </td><td>15.0 ± 0.9</td><td>18.8 ± 0.7</td><td>9.3 ± 0.7</td><td></td><td></td><td> $1 1 . 9 \pm 1 . 2 1 5 . 3 \pm 1 . 3 1 8 . 0 \pm 0 . 8$ </td><td>5.8 ± 0.4</td><td>8.2 ± 0.8</td><td>9.9 ± 2.5</td><td>9.9±4.6</td></tr><tr><td>A-GEM [38]</td><td> $5 . 4 \pm 0 . 4$ </td><td> $5 . 3 \pm 0 . 5$ </td><td>5.4 ± 0.6</td><td>5.6 ± 0.5</td><td>4.9 ± 0.7</td><td>5.0 ± 1.0</td><td>4.7 ± 1.1</td><td>5.0 ± 2.3</td><td>2.5 ± 0.2</td><td>2.6 ± 0.2</td><td>2.5 ± 0.2</td><td>2.6 ± 0.1</td></tr><tr><td>LUCIR [19]</td><td> $1 1 . 1 \pm 0 . 4$ </td><td>13.4 ± 0.9</td><td>14.5 ± 0.4</td><td>17.4±0.8</td><td>10.0 ± 0.8</td><td>12.4 ± 0.6</td><td>14.7 ± 0.9</td><td>16.9 ± 1.1</td><td>6.1 ± 0.4</td><td>6.7 ± 0.4</td><td>8.3 ± 0.6</td><td>9.8 ± 0.6</td></tr><tr><td>ER-WA [13]</td><td> $1 0 . 9 \pm 0 . 5$ </td><td>14.6 ± 0.9</td><td>18.3 ± 0.7</td><td>21.7 ± 1.2</td><td>11.0 ± 0.8</td><td>15.1 ± 0.7</td><td>17.1 ± 0.9</td><td>18.9 ± 1.4</td><td>7.9 ± 0.6</td><td>11.5 ± 0.7</td><td>14.3 ± 0.9</td><td>16.4 ± 1.0</td></tr><tr><td>DER++ [39]</td><td>10.5 ± 0.9</td><td>11.6 ± 0.7</td><td>15.5 ± 1.0</td><td>17.2 ± 1.1</td><td>9.0 ± 0.7</td><td>11.9 ± 1.0</td><td>14.8 ± 0.7</td><td>16.1 ± 1.3</td><td>5.1 ± 0.5</td><td>7.0 ± 0.9</td><td>9.3 ± 0.7</td><td>10.8 ± 1.0</td></tr><tr><td>SS-IL [15]</td><td> $1 3 . 0 \pm 0 . 6$ </td><td> $1 6 . 1 \pm 1 . 0 1 9 . 5 \pm 0 . 6 2 1 . 9 \pm 1 . 1$ </td><td></td><td></td><td>14.9 ± 0.8</td><td>18.0 ± 0.7</td><td>19.7 ± 0.9</td><td>21.7 ± 1.0</td><td>11.2 ± 0.8</td><td>14.1 ± 0.5</td><td>17.1 ± 0.8</td><td>19.6 ± 0.6</td></tr><tr><td>SCR [14]</td><td>7.3 ± 0.6</td><td>8.6 ± 0.4</td><td>13.3 ± 0.6 16.2 ± 1.3</td><td></td><td>7.7 ± 0.6</td><td>12.1 ± 0.7</td><td>14.7 ± 1.9</td><td>16.8 ± 0.6</td><td>5.2 ± 0.2</td><td>7.3 ± 0.2</td><td>8.7 ± 0.4</td><td>10.2 ± 0.3</td></tr><tr><td>ER-ACE [16]</td><td>12.1 ± 0.7</td><td> $1 5 . 7 \pm 0 . 7 1 9 . 7 \pm 0 . 8 2 3 . 1 \pm 0 . 8$ </td><td></td><td></td><td>14.6 ± 0.8</td><td>18.1 ± 0.5</td><td>20.3 ± 1.3</td><td>24.8 ± 1.1</td><td>9.3 ± 0.4</td><td>12.2 ± 0.6</td><td>14.6 ± 0.8</td><td>17.9 ± 0.7</td></tr><tr><td>ER-DVC [31]</td><td> $1 0 . 6 \pm 0 . 5$ </td><td> $1 2 . 9 \pm 0 . 7 1 6 . 8 \pm 0 . 8 1 9 . 7 \pm 0 . 7$ </td><td></td><td></td><td>10.1 ± 0.6</td><td>13.9 ± 0.6</td><td>15.4 ± 0.7</td><td>17.2 ± 0.8</td><td>6.6± 0.6</td><td>9.4 ± 0.6</td><td>11.4 ± 0.8</td><td>14.1 ± 1.0</td></tr><tr><td>OBC [17]</td><td> $9 . 5 \pm 0 . 6$ </td><td>13.3 ± 0.9 18.7 ± 0.9 22.1 ± 0.6</td><td></td><td></td><td> $9 . 2 \pm 0 . 3$ </td><td> $1 4 . 4 \pm 0 . 9$ </td><td>16.4 ±1.4</td><td>19.5 ± 1.5</td><td> $7 . 3 \pm 0 . 4$ </td><td>11.0 ± 0.7</td><td>13.2 ± 0.6</td><td> $1 6 . 2 \pm { 1 . 0 }$ </td></tr><tr><td>LOR [36]</td><td> $9 . 5 \pm 0 . 5$ </td><td> $1 1 . 6 \pm 0 . 7 1 5 . 8 \pm 0 . 9 1 9 . 2 \pm 0 . 8$ </td><td></td><td></td><td> $9 . 5 \pm 1 . 0$ </td><td> $1 1 . 2 \pm 0 . 6$ </td><td>15.2 ± 2.2</td><td> $1 9 . 1 \pm 1 . 8$ </td><td> $5 . 5 \pm 0 . 4$ </td><td>8.6 ± 0.3</td><td>11.9 ± 0.9</td><td> $1 3 . 9 \pm 1 . 0$ </td></tr><tr><td>UER [20]</td><td> $1 2 . 5 \pm 0 . 6$ </td><td> $1 5 . 4 \pm 0 . 7$ </td><td> $2 0 . 9 \pm 0 . 8$ </td><td> $2 4 . 6 \pm 0 . 8$ </td><td> $1 2 . 8 \pm 0 . 9$ </td><td> $1 8 . 4 \pm 0 . 8$ </td><td> $2 1 . 9 \pm 1 . 3$ </td><td> $2 5 . 1 \pm 1 . 1$ </td><td> $9 . 7 \pm 0 . 4$ </td><td> $1 3 . 4 \pm 0 . 5$ </td><td> $1 6 . 2 \pm { 0 . 7 }$ </td><td> $1 8 . 9 \pm 0 . 8$ </td></tr><tr><td>DLER</td><td> $1 4 . 4 \pm 0 . 5$ </td><td> ${ \bf 1 8 . 7 \pm 0 . 5 }$ </td><td> $2 3 . 6 \pm 0 . 8$ </td><td> $2 5 . 1 \pm 1 . 2$ </td><td> ${ \bf 1 5 . 0 \pm 0 . 9 }$ </td><td> $2 0 . 2 \pm 1 . 3$ </td><td>23.1 ± 1.0</td><td> $2 5 . 6 \pm 0 . 7$ </td><td> ${ \bf 1 1 . 9 \pm 0 . 3 }$ </td><td> ${ \bf 1 5 . 8 \pm 0 . 5 }$ </td><td> ${ \bf 1 8 . 5 \pm 0 . 5 }$ </td><td> ${ \bf 2 0 . 3 \pm 0 . 6 }$ </td></tr></table>

3) Implementation Details: The basic setting of the backbone model is the same as the existing work [35], [36]. In detail, we take the Reduced ResNet18 (the number of filters is 20) as the feature extractor on all datasets. All of the networks are randomly initialized rather than pre-trained. During training, the network is trained with the SGD optimizer, and the learning rate is set to 0.1. Meanwhile, we select α and $\lambda _ { s c }$ on a validation set that is obtained by sampling 10% of the training set. As for testing, we select 256 as the batch size. For all datasets, the class indexes are shuffled before division. The model receives 10 current samples from the data stream and 10 previous samples from the memory buffer at a time. We also combine multiple data augmentation techniques to generate augmented views of the training samples.

## V. EXPERIMENTS

## A. Experiment Setup

1) Datasets: We conduct extensive experiments on three publicly available datasets. Split CIFAR100 [40] and Split MiniImageNet [41] are made up of 10 learning stages, and each stage has 10 classes.Split TinyImageNet [42] consists of 20 learning stages, and each stage contains 10 classes.

2) Evaluation Metrics: We first define $a _ { i , j } ( j < = i )$ as the accuracy evaluated on the held-out test samples of the jth task after the network has learned the first i tasks. Similar to [29], we can further acquire the average accuracy rate

$$
A _ { i } = \frac { 1 } { i } \sum _ { j = 1 } ^ { i } a _ { i , j } ,
$$

at the ith task. The final accuracy rate for T tasks is A .

(23)

## B. Overall Performance

We evaluate DLER against state-of-the-art baselines. Table I reports the final average accuracy as the mean ± the half-width of the 95% confidence interval over 10 runs, using distinct, randomly generated class orders for each dataset.

DLER achieves the highest mean accuracy across all 12 experimental settings. On Split CIFAR100, DLER outperforms the strongest competing baseline in each setting by 1.4%, 2.6%, 3.9%, and 2.0% with buffer sizes of 100, 200, 500, and 1000, respectively. On Split MiniImageNet, the corresponding improvements are 0.1%, 2.1%, 2.8%, and 0.8% with buffer sizes of 200, 500, 1000, and 2000. On Split TinyImageNet, DLER consistently outperforms SS-IL by 0.7%, 1.7%, 1.4%, and 0.7% with buffer sizes of 400, 1000, 2000, and 4000, respectively. These results demonstrate the effectiveness of DLER across different datasets and memory capacities, including settings with limited memory.

Compared with UER, DLER improves mean accuracy in every evaluated setting, with gains ranging from 0.5% to 3.3%. Notably, the gains on Split TinyImageNet remain above one percentage point across all four memory capacities, demonstrating the benefits of the extensions under a longer learning sequence with more classes. These results support the combined effectiveness of enhancing dual-logit training through improved feature compactness and memory management.

Figure 2 reports accuracy after each learning stage. Although DLER does not achieve the highest accuracy initially, it leads the compared methods from stage 6 onward on Split CIFAR100 and from stage 5 onward on Split MiniImageNet and Split TinyImageNet. These gains suggest an improved balance between acquiring novel knowledge and retaining historical knowledge as more classes share the fixed memory.

![](images/402564339951fe93279263f7a141f988de63870686da753bac9715ed22c9275c.jpg)  
(a) Split CIFAR100

![](images/12f3e2263dd605567fbc96dd53b2d66801672e2087b80c02b028318dfd33a4be.jpg)  
(b) Split MiniImageNet

![](images/408cf873cde0b97d6d0bba054c0d2a46d6b1efc64229f8a909b63e75405e7d72.jpg)  
(c) Split TinyImageNet  
Fig. 2. Average accuracy rate on observed learning stages on three datasets when the memory buffer size is 1000.

TABLE II  
ABLATION STUDY ON SPLIT TINYIMAGENET WITH DIFFERENT MEMORY BUFFER SIZES.
<table><tr><td></td><td colspan="5">Components</td><td colspan="4">Buffer size</td></tr><tr><td>Method</td><td>Dual-logits</td><td>α</td><td>SC</td><td>BM</td><td>Advantage</td><td>400</td><td>1000</td><td>2000</td><td>4000</td></tr><tr><td>ER</td><td></td><td></td><td></td><td></td><td></td><td> $5 . 2 \pm 0 . 6$ </td><td> $7 . 6 \pm 0 . 5$ </td><td> $1 0 . 1 \pm 0 . 9$ </td><td> $1 2 . 7 \pm 1 . 1$ </td></tr><tr><td>Dual-logit training (Fixed)</td><td>√</td><td></td><td></td><td></td><td></td><td> $8 . 0 \pm 0 . 6$ </td><td> $1 0 . 9 \pm 0 . 7$ </td><td> $1 4 . 2 \pm 0 . 8$ </td><td> $1 6 . 1 \pm 0 . 8$ </td></tr><tr><td>Dual-logit training</td><td>√</td><td>√</td><td></td><td></td><td></td><td> $9 . 7 \pm 0 . 4$ </td><td> $1 3 . 4 \pm 0 . 5$ </td><td> $1 6 . 2 \pm 0 . 7$ </td><td> $1 8 . 9 \pm 0 . 8$ </td></tr><tr><td>Dual-logit training + SC</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td> $1 0 . 0 \pm 0 . 5$ </td><td> $1 3 . 9 \pm 0 . 3$ </td><td> $1 7 . 2 \pm 0 . 5$ </td><td> $1 9 . 6 \pm 0 . 7$ </td></tr><tr><td>Dual-logit training + SC + CB</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td> $1 0 . 2 \pm 0 . 5$ </td><td> $1 4 . 5 \pm 0 . 3$ </td><td> $1 7 . 4 \pm 0 . 8$ </td><td> $1 9 . 7 \pm 0 . 5$ </td></tr><tr><td>DLER</td><td>√</td><td>V</td><td>√</td><td>√</td><td>√</td><td> ${ \bf 1 1 . 9 \pm 0 . 3 }$ </td><td> ${ \bf 1 5 . 8 \pm 0 . 5 }$ </td><td> ${ \bf 1 8 . 5 \pm 0 . 5 }$ </td><td> ${ \bf 2 0 . 3 \pm 0 . 6 }$ </td></tr></table>

![](images/10b03c75d67a4e2967accc996280e886a6426e264b5174f70a0cbf046f7fce8f.jpg)  
(a) Previous Classes

![](images/ef19f92fbe4ff04872dc53ce89b85c1dfa31f7793a97e222b7f52ae2df09f381.jpg)  
(b) Current Classes

![](images/7cc301cc13e4412c0d24b828cc5e850862c47794800353a52059ef621ce47d8e.jpg)  
(c) Average of Previous and Current Classes  
Fig. 3. Average accuracy rate of previous classes, current classes, and their average on Split MiniImageNet with a 1000 memory buffer size.

## C. Ablation Study

Table II presents an ablation study on Split TinyImageNet with buffer sizes of 400, 1000, 2000, and 4000. Starting from ER, we progressively introduce dual-logit training, controllable α, supervised contrastive loss (SC), balanced memory (BM), and advantage-based sample selection. Dual-logit training combines cosine supervision for current samples with additional dot-product supervision during replay, whose relative contribution is controlled by α. SC promotes feature compactness, BM balances memory allocation, and advantagebased selection favors samples with a positive dot-product probability advantage for the correct class. Each successive configuration improves mean final accuracy across all four capacities, with DLER achieving the best results.

1) Dual-Logit Training: Dual-logit training incorporates norm information through dot-product supervision to balance current learning and historical retention. Figure 3 compares accuracy on current classes, previous classes, and their equally weighted average on Split MiniImageNet. Dual-logit training, represented by UER, maintains competitive accuracy on both groups and a higher equally weighted average than these baselines throughout learning. This balance is controlled by α, which adjusts the relative contributions of dot-product and cosine supervision during replay.

Meanwhile, dual-logit training offers flexibility in where the mechanism is applied. To examine this property, we add a linear layer to the feature extractor and apply the mechanism to this layer, the predictor, or both. Table III reports the final accuracy on all classes, current classes, and previous classes. Applying the mechanism to either layer improves overall accuracy over ER, while applying it to both achieves the highest overall accuracy. Compared with the predictor, using both layers improves the performance of previous classes while maintaining competitive performance of current classes.

![](images/a69d8c08227230c1e1ce8c66e26d38e8030a961a82e059eb85b87ebab4c3268a.jpg)  
(a) Split CIFAR100

![](images/d1cf165725bfaeadecf1c8b030d79d6ff37f00b2ed37bc6eca2af62094b6f0d0.jpg)  
(b) Split MiniImageNet

![](images/4e5afa91592cba1009801eacc8b6b2790a660e69989dfb1f90c06ca7245665ac.jpg)  
(c) Split TinyImageNet

Fig. 4. Improvement of existing methods with dual-logits training on three datasets while the buffer size is 1000.  
![](images/7c7afc41d03b3f682b28819978fceb6a8e7a9371515eb0949665f4c81c327057.jpg)  
Fig. 5. The performance of dual-logits training on Split CIFAR100 (buffer size=1000) with different values of α.

TABLE III  
FINAL ACCURACY (%) ON SPLIT CIFAR100 WITH A BUFFER OF 1,000 SAMPLES AFTER ADDING A LINEAR LAYER TO THE FEATURE EXTRACTOR.
<table><tr><td>Method Setting</td><td>All Classes</td><td>Current Classes</td><td>Previous Classes</td></tr><tr><td>ER</td><td> $1 5 . 6 \pm 2 . 6$ </td><td>50.5</td><td>11.7</td></tr><tr><td>ER-ACE</td><td> $1 8 . 9 \pm 0 . 9$ </td><td>17.3</td><td>19.1</td></tr><tr><td>OBC</td><td> $1 8 . 3 \pm 0 . 7$ </td><td>17.5</td><td>18.4</td></tr><tr><td>Dual-logit training (feature)</td><td> $1 8 . 6 \pm 1 . 0$ </td><td>30.6</td><td>17.2</td></tr><tr><td>Dual-logit training (predictor)</td><td> $1 9 . 6 \pm 1 . 5$ </td><td>34.7</td><td>18.0</td></tr><tr><td>Dual-logit training (both)</td><td> ${ \bf 2 0 . 1 \pm 1 . 2 }$ </td><td>31.8</td><td>18.8</td></tr></table>

Additionally, dual-logits training can be used with existing methods. Figure 4 compares six ER-based baselines with and without dual-logit training on the three datasets. The blue curves represent the original baselines, while the red curves show their performance after integration. All six baselines achieve higher final accuracy, with improvements throughout most of the learning process. These results suggest that incorporating norm information during replay can complement existing methods for mitigating prediction bias.

2) Controllable α: To examine how α influences this balance, we conduct experiments on Split CIFAR100 with different values of α in Figure 5. The accuracy on current classes generally decreases as α increases, whereas the accuracy on previous classes initially improves and then declines. Consequently, the overall accuracy first increases but saturates or decreases when α becomes too large. Consistently, Table II shows that allowing α to be adjusted improves performance over fixing α = 1 across all four memory capacities. These results support the importance of balancing the two objectives and demonstrate the adjustability of our proposed method.

![](images/5397d43123789a6e4dbe4bbed4de472376762ac9364d7f9896f55f72b540bc5f.jpg)

![](images/04545078701266bb353cd9d3447ff1be12e880a65078cfcacddbbb52727aff02.jpg)  
Fig. 6. Conditional coverage and previous-class accuracy of dual-logits training and its enhancement with SC on Split TinyImageNet.

3) Supervised contrastive objective: SC is proposed to enhance the dual-logits training mechanism. To investigate how SC enhances dual-logit training, Figure 7 compares training with and without SC on Split TinyImageNet using identical memory contents and test sample pairings. We measure conditional coverage and previous-class accuracy. Conditional coverage is the proportion of test samples from a previous class correctly distinguished from a given current class, conditioned on a stored reference from the same previous class being correctly distinguished from that current class. It measures whether separation achieved by stored samples extends to unseen samples of the same class. Previous-class accuracy evaluates previous-class test samples over all observed classes.

TABLE IV  
COMPARISON OF THE HIGHEST- AND LOWEST-SCORING TRAINING SAMPLES WITHIN THE 96 SELECTED CLASSES ON SPLIT TINYIMAGENET.
<table><tr><td>Metric</td><td>High-score</td><td>Low-score</td></tr><tr><td>Spatial JS divergence</td><td>0.2256</td><td>0.0463</td></tr><tr><td>Feature norm</td><td>15.7856</td><td>12.8475</td></tr><tr><td>Ground-truth cosine probability</td><td>0.1173</td><td>0.0260</td></tr><tr><td>Gradient norm (all)</td><td>7.3114</td><td>52.7980</td></tr><tr><td>Gradient norm (backbone)</td><td>7.0487</td><td>51.4370</td></tr><tr><td>Gradient conflict rate</td><td>41.5%</td><td>44.6%</td></tr></table>

SC generally improves both metrics, with more consistent gains under larger buffers and improvements at every evaluated stage for the largest buffer. With the smallest buffer, gains fluctuate, and final performance is slightly lower despite higher stage averages. These trends suggest that SC helps replay supervision generalize beyond stored samples. By promoting within-class angular compactness and between-class separation, SC provides more consistent angular representations for both logits, potentially extending corrections learned through dot-product supervision to other samples of the same class.

4) Balanced Memory component: Reservoir sampling approximately preserves the class distribution of the data stream, yielding balanced memory in expectation for a balanced stream. However, score-based selection may favor certain classes, motivating BM to regulate memory allocation. To separate its contribution from advantage-based selection, Table II includes dual-logit training with SC and BM as an intermediate configuration. Compared with SC and reservoir sampling, adding BM yields modest gains across all four memory capacities, suggesting that class balance alone does not explain the full improvement of DLER.

5) The Advantage component: Building on BM, the advantage component prioritizes samples for which dot-product prediction assigns a higher probability to the correct class than cosine prediction. As shown in Table II, this component further improves mean accuracy across all memory capacities, with larger gains under smaller buffers. To investigate the characteristics associated with this score, we analyze the final DLER model trained on Split TinyImageNet with a buffer size of 400. We focus on the 96 classes for which DLER achieves higher accuracy than the configuration using dual-logit training with SC and CB. Within each class, we select the highest- and lowest-scoring samples using the final model and compare their spatial responses, feature statistics, and gradients. The results in Table IV characterize these selected classes.

We compare high- and low-score samples from three perspectives. 1) Spatial responses. Class activation maps (CAMs) are extracted from layer4 using the ground-truth dot-product and cosine logits and normalized by their spatial sums. Highscore samples exhibit larger spatial Jensen–Shannon (JS) divergence, indicating more distinct responses between the two logits, as illustrated in Figure 8. 2) Feature characteristics. High-score samples have larger feature norms and higher ground-truth cosine probabilities, suggesting that their dotproduct advantage coexists with stronger feature magnitudes and useful angular evidence rather than simply weak cosine predictions. 3) Gradient behavior. Under combined cosine and dot-product supervision, high-score samples exhibit substantially smaller overall and backbone gradient norms and a slightly lower gradient conflict rate, measured as the proportion of sample pairs with negative gradient cosine similarity. This indicates milder updates and less conflict among evalu ated pairs. Overall, advantage-based selection identifies samples with stronger feature magnitudes, greater cosine support, and additional dot-product probability advantages, potentially supporting more effective replay under limited memory.

TABLE V  
THEORETICAL COMPARISON OF TRAINING COMPUTATION, MODEL PARAMETERS, AND INFERENCE COMPUTATION.
<table><tr><td>Method</td><td>Training per update</td><td>Model parameters</td><td>Inference per sample</td></tr><tr><td>ER</td><td>O(BQ)</td><td> $P$ </td><td> $O ( F _ { \mathrm { p r e d } } )$ </td></tr><tr><td>ER-ACE</td><td>O(BQ)</td><td> $P$ </td><td> $O ( F _ { \mathrm { p r e d } } )$ </td></tr><tr><td>SS-IL</td><td> $O ( B Q + B F _ { \mathrm { p r e d } } )$ </td><td> $P + P _ { \mathrm { o l d } }$ </td><td> $O ( F _ { \mathrm { p r e d } } )$ </td></tr><tr><td>UER</td><td> $O ( B Q )$ </td><td> $P$ </td><td> $O ( F _ { \mathrm { p r e d } } )$ </td></tr><tr><td>DLER</td><td> $O \big ( B Q + B ^ { 2 } d + M F _ { \mathrm { p r e d } } + T _ { \mathrm { s e l } } \big )$ </td><td> $P$ </td><td> $O ( F _ { \mathrm { p r e d } } )$ </td></tr></table>

## D. Complexity Analysis

Table V compares computational and parameter storage requirements. Here, B, M, and d denote the combined training batch size, memory capacity, and feature dimension; Q and $F _ { \mathrm { p r e d } }$ denote per-sample training and prediction costs. P and $P _ { \mathrm { o l d } }$ are the standard and teacher model parameter counts, and $T _ { \mathrm { s e l } }$ is the memory selection cost. ER, ER-ACE, and UER share the same asymptotic training complexity, while SS-IL requires additional teacher predictions and storage. DLER retains UER’s architecture without additional parameters.

DLER adds $O ( B ^ { 2 } d )$ contrastive computation, $O ( M F _ { \mathrm { p r e d } } )$ memory scoring, and sample selection. Scoring scales linearly with the fixed buffer capacity and can be batched without retaining backpropagation graphs. All methods require O(MS) replay storage, where $S$ is the storage per sample; DLER adds $O ( M + C )$ scores and metadata for C observed classes. These additional operations occur only during training. DLER thus improves replay selection while preserving the model size and inference cost of ER and UER, supporting its Real-time ability through direct prediction without additional inference modules or post-processing.

## VI. CONCLUSION

In this paper, we propose DLER to alleviate catastrophic forgetting in OCL by addressing prediction bias. DLER learns current samples using cosine logits and incorporates additional dot-product supervision during replay. To further strengthen the dual-logit mechanism, an auxiliary supervised contrastive objective promotes angular compactness, supporting the generalization of corrections beyond limited replay samples. We also extend dual-logits to memory management through advantage-based scoring, aligning sample retention with the dual-logit training. Extensive experiments on three datasets demonstrate the effectiveness of DLER, particularly under limited replay memory.

## REFERENCES

[1] L. Nie, L. Qu, D. Meng, M. Zhang, Q. Tian, and A. D. Bimbo, “Search-oriented micro-video captioning,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 3234–3243. 1

[2] L. Nie, W. Wang, R. Hong, M. Wang, and Q. Tian, “Multimodal dialog system: Generating responses via adaptive decoders,” in Proceedings of the 27th ACM International Conference on Multimedia, 2019, pp. 1098–1106. 1

[3] L. Nie, X. Wang, J. Zhang, X. He, H. Zhang, R. Hong, and Q. Tian, “Enhancing micro-video understanding by harnessing external sounds,” in Proceedings ofthe 25th ACM international conference on Multimedia, 2017, pp. 1192–1200. 1

[4] M. Delange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis, G. Slabaugh, and T. Tuytelaars, “A continual learning survey: Defying forgetting in classification tasks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021. 1

[5] B. Liu and S. Mazumder, “Lifelong and continual learning dialogue systems: learning during conversation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 17, 2021, pp. 15 058– 15 063. 1

[6] B. Liu, S. Mazumder, E. Robertson, and S. Grigsby, “Ai autonomy: Self-initiation, adaptation and continual learning,” arXiv preprint arXiv:2203.08994, 2022. 1

[7] B. Liu and C. Mei, “Lifelong knowledge learning in rule-based dialogue systems,” arXiv preprint arXiv:2011.09811, 2020. 1

[8] H. Lin, S. Feng, X. Li, W. Li, and Y. Ye, “Anchor assisted experience replay for online class-incremental learning,” IEEE Transactions on Circuits and Systems for Video Technology, 2022. 1

[9] H. Yin, P. Li et al., “Mitigating forgetting in online continual learning with neuron calibration,” Advances in Neural Information Processing Systems, vol. 34, 2021. 1

[10] Z. Mai, R. Li, J. Jeong, D. Quispe, H. Kim, and S. Sanner, “Online continual learning in image classification: An empirical survey,” Neurocomputing, vol. 469, pp. 28–51, 2022. 1

[11] H. Lin, B. Zhang, S. Feng, X. Li, and Y. Ye, “Pcr: Proxy-based contrastive replay for online class-incremental continual learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 24 246–24 255. 1, 3

[12] Y. Wu, Y. Chen, L. Wang, Y. Ye, Z. Liu, Y. Guo, and Y. Fu, “Large scale incremental learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 374–382. 1

[13] B. Zhao, X. Xiao, G. Gan, B. Zhang, and S.-T. Xia, “Maintaining discrimination and fairness in class incremental learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 13 208–13 217. 1, 6

[14] Z. Mai, R. Li, H. Kim, and S. Sanner, “Supervised contrastive replay: Revisiting the nearest class mean classifier in online class-incremental continual learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 3589–3599. 1, 3, 4, 6

[15] H. Ahn, J. Kwak, S. Lim, H. Bang, H. Kim, and T. Moon, “Ss-il: Separated softmax for incremental learning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 844–853. 1, 3, 6

[16] L. Caccia, R. Aljundi, N. Asadi, T. Tuytelaars, J. Pineau, and E. Belilovsky, “New insights on reducing abrupt representation change in online continual learning,” in International Conference on Learning Representations, 2021. 1, 3, 6

[17] A. Chrysakis and M.-F. Moens, “Online bias correction for taskfree continual learning,” in The Eleventh International Conference on Learning Representations. 1, 6

[18] W. Liu, Z. Liu, Z. Yu, B. Dai, R. Lin, Y. Wang, J. M. Rehg, and L. Song, “Decoupled networks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 2771–2779. 1, 11

[19] S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin, “Learning a unified classifier incrementally via rebalancing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 831–839. 1, 3, 6, 11

[20] H. Lin, S. Feng, B. Zhang, H. Qiao, X. Li, and Y. Ye, “Uer: A heuristic bias addressing approach for online continual learning,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 96–104. 2, 3, 6

[21] C. Savadikar, M. Dai, and T. Wu, “CHEEM: Continual learning by reuse, new, adapt and skip—a hierarchical exploration-exploitation approach,”

in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 25 066–25 076. 3

[22] X. Liu and X. Chang, “Elastic weight consolidation done right for continual learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 3930–3940. 3

[23] J. Deng, Q. Wu, P. Ju, S. Lin, Y. Liang, and N. Shroff, “Unlocking the power of rehearsal in continual learning: A theoretical perspective,” in Proceedings ofthe 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267, 2025, pp. 13 277–13 326. 3

[24] B. Cui, G. Hu, and S. Yu, “Deepcollaboration: Collaborative generative and discriminative models for class incremental learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 2, 2021, pp. 1175–1183. 3

[25] D. Rolnick, A. Ahuja, J. Schwarz, T. Lillicrap, and G. Wayne, “Experience replay for continual learning,” Advances in Neural Information Processing Systems, vol. 32, 2019. 3, 6

[26] R. Aljundi, M. Lin, B. Goujaud, and Y. Bengio, “Gradient based sample selection for online continual learning,” Advances in neural information processing systems, vol. 32, 2019. 3, 6

[27] X. Jin, A. Sadhu, J. Du, and X. Ren, “Gradient-based editing of memory examples for online task-free continual learning,” Advances in Neural Information Processing Systems, vol. 34, 2021. 3, 6

[28] R. Aljundi, E. Belilovsky, T. Tuytelaars, L. Charlin, M. Caccia, M. Lin, and L. Page-Caccia, “Online continual learning with maximal interfered retrieval,” Advances in Neural Information Processing Systems, vol. 32, pp. 11 849–11 860, 2019. 3, 6

[29] D. Shim, Z. Mai, J. Jeong, S. Sanner, H. Kim, and J. Jang, “Online class-incremental continual learning with adversarial shapley value,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 11, 2021, pp. 9630–9638. 3, 6

[30] M. Y. Harun, J. Gallardo, J. Chen, and C. Kanan, “GRASP: A rehearsal policy for efficient online continual learning,” in Proceedings of the 3rd Conference on Lifelong Learning Agents, ser. Proceedings of Machine Learning Research, vol. 274, 2025, pp. 1032–1052. 3

[31] Y. Gu, X. Yang, K. Wei, and C. Deng, “Not just selection, but exploration: Online class-incremental continual learning via dual view consistency,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 7442–7451. 3, 6

[32] Q. Wang, R. Wang, Y. Wu, X. Jia, and D. Meng, “Cba: Improving online continual learning via continual bias adaptor,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19 082–19 092. 3

[33] Y.-S. Liang and W.-J. Li, “Loss decoupling for task-agnostic continual learning,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. 3

[34] Y. Guo, B. Liu, and D. Zhao, “Online continual learning through mutual information maximization,” in International Conference on Machine Learning. PMLR, 2022, pp. 8109–8126. 3

[35] S. Lai, C. Ma, F. Zhu, Z. Zhao, X. Lin, G. Meng, and Q. Zhang, “Gradient-guided epsilon constraint method for online continual learning,” in Advances in Neural Information Processing Systems, vol. 38, 2025. 3, 6

[36] S. Lai, Z. Zhao, F. Zhu, J. Cheng, X. Lin, Q. Zhang, and G. Meng, “Beyond myopic alignment: Lookahead optimization for online classincremental learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 18 053–18 062. 3, 6

[37] J. M. Steele, The Cauchy-Schwarz master class: an introduction to the art of mathematical inequalities. Cambridge University Press, 2004. 4

[38] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny, “Efficient lifelong learning with a-gem,” in International Conference on Learning Representations, 2018. 6

[39] P. Buzzega, M. Boschini, A. Porrello, D. Abati, and S. Calderara, “Dark experience for general continual learning: a strong, simple baseline,” Advances in neural information processing systems, vol. 33, pp. 15 920– 15 930, 2020. 6

[40] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009. 6

[41] O. Vinyals, C. Blundell, T. Lillicrap, D. Wierstra et al., “Matching networks for one shot learning,” Advances in neural information processing systems, vol. 29, pp. 3630–3638, 2016. 6

[42] Y. Le and X. Yang, “Tiny imagenet visual recognition challenge,” CS 231N, vol. 7, no. 7, p. 3, 2015. 6

## VII. APPENDIX

## A. Analysis of Catastrophic Forgetting

There is a competition between samples of each class in the process of gradient propagation. By the chain rule, the gradient of the predictor for a training sample x of class y can be expressed as

$$
\frac { \partial L _ { e r } } { \partial \pmb { w } _ { j } } = \left\{ \begin{array} { c c } { ( p _ { y } ^ { d o t } - 1 ) \pmb { h } , } & { j = y } \\ { p _ { j } ^ { d o t } \pmb { h } , } & { j \neq y } \end{array} \right. ,\tag{24}
$$

and

$$
\frac { \partial L _ { e r } } { \partial b _ { j } } = \left\{ \begin{array} { r l } { p _ { y } ^ { d o t } - 1 , } & { j = y } \\ { p _ { j } ^ { d o t } , } & { j \neq y } \end{array} \right. .\tag{25}
$$

With a positive learning rate η and a non-negative feature vector h, the gradient updates increase the parameters associated with the ground-truth class y: $\pmb { w } _ { y }  \pmb { w } _ { y } - \eta ( p _ { y } - 1 ) \mathbf { h }$ and $b _ { y } ~  ~ b _ { y } - \eta ( p _ { y } - 1 )$ . Conversely, they decrease the parameters associated with other classes: $\mathbf { \boldsymbol { w } } _ { j } \gets \mathbf { \boldsymbol { w } } _ { j } - \eta \boldsymbol { p } _ { j } \mathbf { \boldsymbol { h } }$ and $b _ { j }  b _ { j } - \eta p _ { j }$ , for $j \neq y .$ . These updates move ${ \pmb w } _ { y }$ toward the direction of h and increase $b _ { y }$ . Thus, it increases the logits score of the y-th dimension as well as decreases the logits scores of other dimensions. And the process is the same when using cosine logits by

$$
\frac { \partial L _ { l u c i r } } { \partial \pmb { w } _ { j } } = \pmb { \left\{ \begin{array} { l l } { ( p _ { y } ^ { c o s } - 1 ) \hat { \pmb { h } } , } & { j = y } \\ { p _ { j } ^ { c o s } \hat { \pmb { h } } , } & { j \neq y } \end{array} , \right. }\tag{26}
$$

where

$$
\hat { \pmb { h } } = \frac { \frac { \pmb { h } } { \| \pmb { h } \| } - c o s ( \pmb { w } , \pmb { h } ) \frac { \pmb { w } } { \| \pmb { w } \| } } { \| \pmb { w } \| } .\tag{27}
$$

The prediction bias is a phenomenon of CF, mainly caused by the unbalanced competition of gradient propagation. Generally, the number of samples belonging to previous classes in the fixed buffer will decrease as the learning process continues. The number of samples belonging to current classes remains unchanged. If the model incrementally learns by Equation (3) or Equation (4), the associated parameters of current classes receive more positive changes, while the others obtain more negative changes. This means the gradients of previous classes lose their competitiveness. Besides, the logits of current classes tend to be larger than the previous ones. Hence, the model is more likely to classify most samples to the classes of current samples, reducing the performance of previous classes.

## B. Analysis of Biased Predictor

Imbalanced gradient mainly affects the predictor through the angle and norm factors. The norm factor accounts for intraclass variation, and the angle factor accounts for inter-class variation [18]. To study their changes in OCL, we conduct experiments on CIFAR100 with two incremental stages and 50 classes per stage. We set two general ways to calculate logits in the experiment: dot-product logits (both factors) and cosine logits (only angle factor). In Table VI, we take different ways to calculate logits and record their performance.

Experiment results show that the characteristics of these two factors differ from each other. On the one hand, the bias issue mainly occurs in the angle factor, which efficiently learns novel knowledge. Comparing index 1 and 2, replacing dot-product logits with cosine logits in the testing phase leads to worse final accuracy, particularly for previous classes. Meanwhile, the method in index 3 performs better for current classes but worse for previous classes compared to the one in index 1. On the other hand, the norm factor, which has been abandoned in existing studies [19], benefits previous classes. As seen in indices 3 and 4, switching from cosine to dotproduct logits during testing improves overall accuracy, with a substantial gain on previous classes.

With this observation, the two factors can be utilized in different occasions. The angle factor is effective in adapting novel knowledge with inter-class variation. Consequently, cosine logits that depend on the angle factor are suitable for learning current samples. And dot-product logits can be used for replaying previous samples since they can leverage the norm factor to enhance historical knowledge with intra-class variation. Motivated by these findings, we propose a method that learns current samples by cosine logits, replays previous samples, and predicts testing samples by dot-product logits.

## C. Analysis of Motivation

In addition to the results in Table VI, we present more detailed results for ER (blue), LUCIR (yellow) and our method (orange) in Figure 7. We calculate the difference between the parameters of the predictor before and after learning the current classes, taking it as the accumulated changes in parameters as shown in Figure 7 (a). Meanwhile, we report the average norm value of the weight vector for all classes in Figure 7 (b). Moreover, we calculate the average categorical probability of all testing samples by the latest model, illustrated in Figure 7 (c). Further analysis and experimental results verify the feasibility of our proposed method.

Firstly, learning current samples using cosine logits can save the novel knowledge into the angle factor. In indices 7 and 8, the performance on previous classes is almost zero when testing with cosine logits. The reason behind this is that the angular similarity is almost the determining factor for gradients in Equation (26). For one thing, all vectors in Equation (26) are normalized, and this greatly enhances the influence of the angular similarity. For another thing, the change of w is constrained by h<sup>ˆ</sup> , as large ∥w∥ can decrease gradients so much that the weights pass is not able to update effectively. Moreover, the dot-product logits is associated with a value range [−∞, +∞], while the value range of cosine logits is [−1, 1]. This causes the cosine logits to produce smaller changes for previous classes and larger changes for current classes (as seen in the blue and orange parts in Figure 7 (a)). It also raises the norm factor of previous classes and reduces the norm factor of current classes (as observed in the blue and orange parts in Figure 7 (b)), which helps learn novel knowledge by the angle factor.

Secondly, replaying previous samples with dot-product logits can enhance historical knowledge through the norm factor. Comparing index 7 and 8, using dot-product logits in the testing process results in the best performance for previous classes, even surpassing the performance of current classes.

TABLE VI  
COMPARISON OF LOGIT CONFIGURATIONS FOR TRAINING AND INFERENCE ON SPLIT CIFAR100 WITH A MEMORY BUFFER OF 5,000 SAMPLES. $A _ { \mathrm { a l l } } ,$ $A _ { p } ,$ AND $A _ { c }$ DENOTE THE FINAL ACCURACIES (%) ON ALL, PREVIOUS, AND CURRENT CLASSES, RESPECTIVELY. $\overline { { | | \boldsymbol { \boldsymbol { w } } | | } }$ DENOTES THE MEAN CLASS-WEIGHT VECTOR NORM; µ<sub>W</sub> AND σ<sub>W</sub> DENOTE THE MEAN AND STANDARD DEVIATION OF WEIGHT ENTRIES; AND $\mu _ { b }$ DENOTES THE MEAN BIAS. SUBSCRIPTS p AND c INDICATE PREVIOUS AND CURRENT CLASSES. DOT AND COS DENOTE DOT-PRODUCT AND COSINE LOGITS, RESPECTIVELY.
<table><tr><td></td><td colspan="3">Logit configuration</td><td colspan="3">Accuracy (%)</td><td colspan="2">Weight norm</td><td colspan="2">Weight statistics</td><td colspan="2">Bias</td></tr><tr><td>Configuration</td><td>Learning</td><td>Replaying</td><td>Testing</td><td> $A _ { \mathrm { a l l } }$ </td><td> $A _ { p }$ </td><td> $A _ { c }$ </td><td> $\overline { { \| \pmb { w } \| } } _ { \gamma }$  9</td><td> $\overline { { \| \pmb { w } \| } } _ { c }$ </td><td> $\mu _ { W _ { p } } \pm \sigma _ { W _ { p } }$ </td><td> $\mu _ { W _ { c } } \pm \sigma _ { W _ { c } }$ </td><td> $\mu _ { b _ { p } }$ </td><td> $\mu _ { b _ { c } }$ </td></tr><tr><td>1 (ER)</td><td>Dot</td><td>Dot</td><td>Dot</td><td>27.3</td><td>17.7</td><td>36.9</td><td>2.08</td><td>1.86</td><td> $0 . 0 0 4 4 \pm 0 . 0 2 7 4$ </td><td> $- 0 . 0 0 3 8 \pm 0 . 0 2 2 1$  一</td><td>-0.22</td><td>0.22</td></tr><tr><td>2</td><td>Dot</td><td>Dot</td><td>Cos</td><td>24.1</td><td>12.7</td><td>35.6</td><td>2.08</td><td>1.86</td><td> $0 . 0 0 4 4 \pm 0 . 0 2 7 4$ </td><td> $- 0 . 0 0 3 8 \pm 0 . 0 2 2 1$ </td><td>-0.22</td><td>0.22</td></tr><tr><td>3 (LUCIR)</td><td>Cos</td><td>Cos</td><td>Cos</td><td>26.9</td><td>14.9</td><td>38.9</td><td>2.00</td><td>1.64</td><td> $0 . 0 3 9 8 \pm 0 . 0 2 3 5$ </td><td> $0 . 0 4 7 1 \pm 0 . 0 1 4 6$ </td><td>-0.00</td><td>0.00</td></tr><tr><td>4</td><td>Cos</td><td>Cos</td><td>Dot</td><td>28.2</td><td>28.8</td><td>27.6</td><td>2.00</td><td>1.64</td><td> $0 . 0 3 9 8 \pm 0 . 0 2 3 5$ </td><td> $0 . 0 4 7 1 \pm 0 . 0 1 4 6$ </td><td>-0.00</td><td>0.00</td></tr><tr><td>5</td><td>Dot</td><td>Cos</td><td>Dot</td><td>19.6</td><td>0.0</td><td>39.1</td><td>1.37</td><td>2.09</td><td> $0 . 0 1 9 7 \pm 0 . 0 1 1 5$ </td><td> $0 . 0 2 1 9 \pm 0 . 0 2 7 0$ </td><td>-1.03</td><td>1.03</td></tr><tr><td>6</td><td>Dot</td><td>Cos</td><td>Cos</td><td>27.3</td><td>25.3</td><td>29.4</td><td>1.37</td><td>2.09</td><td> $0 . 0 1 9 7 \pm 0 . 0 1 1 5$ </td><td> $0 . 0 2 1 9 \pm 0 . 0 2 7 0$ </td><td>-1.03</td><td>1.03</td></tr><tr><td>7</td><td>Cos</td><td>Dot</td><td>Cos</td><td>21.3</td><td>0.5</td><td>42.2</td><td>2.52</td><td>1.51</td><td> $0 . 0 5 0 8 \pm 0 . 0 3 7 1$ </td><td> $0 . 0 5 6 5 \pm 0 . 0 1 1 2$ </td><td>0.20</td><td>-0.20</td></tr><tr><td>8 (UER)</td><td>Cos</td><td>Dot</td><td>Dot</td><td>29.5</td><td>30.9</td><td>28.1</td><td>2.52</td><td>1.51</td><td> $0 . 0 5 0 8 \pm 0 . 0 3 7 1$ </td><td> $0 . 0 5 6 5 \pm 0 . 0 1 1 2$ </td><td>0.20</td><td>-0.20</td></tr></table>

![](images/9fdf910739f69d52f9487b431db1e3658918bf29a281bd62c2d88c24b8ce557a.jpg)  
(a) The accumulate changes of parameters.

![](images/4770bab6cfd9b2084dae4e4ecb2b51b652e96fcf6c269b40009acc06648e6fed.jpg)  
(b) The average norm value of all classes.

![](images/cd2b1ecce752bf044bd57147ef60dac0a5005552731a379615210bafff3fb743.jpg)  
(c) The categorical probability of testing samples.  
Fig. 7. The analysis results of different ways during the learning process on Split CIFAR100 when the memory buffer size is 5000.

Different from cosine logits, the gradient in Equation (24) depends on both two factors. In the meantime, the gradient in Equation (24) is only associated with previous samples in the buffer. Since the angle factor is easily influenced by an unbalanced gradient, this process tends to pay more attention to the norm factor. As a result, replaying by dot-product logits not only decreases the changes for previous classes but also increases the changes for current classes (as seen in the yellow and orange parts in Figure 7 (a)). In the meantime, it raises the norm factor of previous classes as well as reduces the norm factor of current classes (indicated by the yellow and orange parts in Figure 7 (a)). It causes that $\| W _ { p } \|$ is larger than $\lVert \boldsymbol { W } _ { c } \rVert$ , and ${ \mathrm { M e a n } } ( b _ { c } )$ is smaller than ${ \mathrm { M e a n } } ( b _ { p } )$ . Although the angle factor suffers from cos $( w _ { c } , h ) > \cos ( w _ { p } , h )$ , it can be leveraged by $\| \pmb { w } _ { p } \| \| \pmb { h } \| + b _ { p } > \| \pmb { w } _ { c } \| \| \pmb { h } \| + b _ { c }$ , as we described in Figure 1 (d).

As shown in Table VI, our proposed method achieves the best performance among all combinations of different logits. Based on the combination of different logits, our method quickly learns novel knowledge and effectively retains historical knowledge. As shown in Figure 7 (c), the posterior probability distribution of our method is relatively balanced, whereas those of ER and LUCIR tend to favor the current classes. Therefore, our motivation for learning current samples by cosine logits while replaying previous samples by dotproduct logits is reasonable and practicable.

![](images/97eb78d57bebb3995fd3041859e77618ddfc5f8062da076273e2d089847a06fe.jpg)  
Fig. 8. Examples of high-score and low-score samples and their class activation maps obtained using dot-product and cosine logits.