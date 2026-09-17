# Multi-Appliance Non-Intrusive Load Monitoring via Label-Preserving Aggregate Recomposition and Prediction Consistency

Jiangfeng Liu and Yanfang Fan

Abstract—Non-intrusive load monitoring (NILM) estimates appliance power sequences from aggregate power, but models trained on source households commonly lose accuracy in unseen households. Aggregate power also contains loads from other appliances and measurement error, so predictions may depend on the residual background that co-occurs with sourcehousehold targets. Time-aligned submetered measurements and the additive decomposition of aggregate power expose a relation unused by window-wise supervision: an aggregate window can be recomposed by replacing only its residual background while preserving all modeled target-appliance power sequences pointwise. We combine label-preserving aggregate recomposition with prediction consistency. Both windows receive complete power and operating-state supervision. For each appliance, disagreement between the two power predictions is penalized only when both satisfy a fixed reliability criterion and only to the extent that it exceeds a fixed margin. The proposed method is implemented using a multi-appliance architecture with two-stage shared-tospecific mixture-of-experts routing. On REDD, UK-DALE, and REFIT, the proposed method lowers appliance-averaged mean absolute error relative to single-window training from 14.75 to 13.14 W, from 8.88 to 8.51 W, and from 15.83 to 14.55 W. Labelpreserving aggregate recomposition and prediction consistency are used only during training, and add no inference-time module or parameter.

Index Terms—Energy consumption, energy disaggregation, load monitoring, non-intrusive load monitoring.

## I. INTRODUCTION

B UILDINGS account for a substantial share of global energy demand, and residential use represents most building-sector consumption [1]. Information on when individual appliances operate and how much electricity they use can support residential energy management and inform demandside management [2]. Non-intrusive load monitoring (NILM) estimates appliance power consumption from aggregate smartmeter readings without requiring individual metering of every appliance [3], [4]. NILM models learn this mapping during training on labeled data from source households.

## A. Background and Motivation

This paper considers a multi-appliance NILM setting in which a single model jointly estimates the power sequences of several target appliances from the same aggregate power sequence [5]. NILM models can show performance degradation when applied to previously unseen households [4]. This degradation may arise from differences across households in the power consumption of both the target appliances and the other appliances. Even appliances of the same type can exhibit different power characteristics across brands or models [6], and household usage habits can change when and for how long the target appliances operate [4], [7]. The aggregate also contains power consumption from appliances outside the target set, and this contribution can vary across households [8], [9].

For a specified set of target appliances, we define the residual background as the difference between the aggregate power and the sum of the target-appliance power sequences. It comprises consumption from all other appliances together with measurement error. In the training data, patterns in the residual background may correlate with target-appliance operation, and a model may exploit these correlations [10]. If these correlations no longer hold in other households, predictions that rely on them may become less accurate [11]. During training, we therefore seek to limit how much the model relies on these correlations.

## B. Related Work and Research Gap

Studies in computer vision have examined models’ reliance on background cues by deliberately altering non-target context [12], [13]. SwapMix [14] swaps the features of questionirrelevant context objects identified using ground-truth question reasoning steps, whereas Counterfactual Generative Networks [15] construct training images by recomposing disentangled visual factors. Although these constructions differ, they share the idea of varying factors that should not determine the target label while retaining task-relevant semantics. Both constructions, however, depend on annotations or on a generative decomposition, and they preserve categorical target semantics rather than dense, pointwise regression targets.

In NILM, several studies constrain learned representations through relations defined over modified or paired observations. CR-MAT mixes the frequency magnitudes of an aggregate window with those from another household while retaining the original phase, and penalizes the representation change caused by this mixing [10]. Han et al. form positive pairs from matching timestamps across augmented contexts and construct hard negatives by mixing features in the embedding space [16]. DisCoV instead separates positive from negative samples according to whether a given appliance is active, and constrains appliance-specific latent variables [17]. Outside NILM, CoRe provides a related grouped-observation formulation that penalizes the conditional variance of predictions within groups sharing the same target and identifier [11].

To limit reliance on residual-background correlations during training, we seek a window pair whose two windows differ only in the residual background, thereby leaving every modeled target-appliance power sequence pointwise unchanged. The reviewed approaches do not provide such a pair. In the supervised multi-appliance setting, however, such a pair can be constructed directly from time-aligned submetered measurements using the additive decomposition of aggregate power. Because the two windows have the same targets, training can constrain the disagreement between their corresponding per-appliance power predictions. Conventional window-wise supervision evaluates each window against its targets but does not explicitly encode the relation between the paired predictions.

## C. Contributions of the Present Work

To address this gap, we propose a method that combines label-preserving aggregate recomposition with prediction consistency during training. We develop the feature-gated layered appliance mixture-of-experts (FLAME) architecture to share aggregate-level context across appliances while forming a distinct representation for each. We use FLAME to implement the proposed method. The key contributions of this work are as follows:

1) Label-Preserving Aggregate Recomposition and $P r e \mathrm { . }$ diction Consistency: During supervised multi-appliance training, the proposed method recomposes an aggregate window so that only its residual background changes, leaving every modeled target-appliance power sequence and its operating-state targets pointwise unchanged. Both windows of the resulting pair receive the same complete task supervision. For each appliance, the method additionally constrains disagreement between the two windows’ power predictions only when both predictions are accurate enough, without requiring them to be identical.

2) FLAME Multi-Appliance Architecture: FLAME treats each appliance as a task and organizes prediction through two-stage shared-to-specific expert routing. It first aggregates responses from shared and appliance-specific experts into a single shared representation and then refines that representation into one per appliance for joint power and operating-state prediction.

The remainder of this paper is organized as follows. Section II states the multi-appliance NILM problem and defines the residual background. Section III presents the FLAME architecture, two residual-based window constructions, prediction consistency, and the overall training objective. Section IV reports the experimental setup and results, and Section V concludes the paper.

## II. PROBLEM STATEMENT

We consider multi-appliance non-intrusive load monitoring (NILM), where the objective is to estimate the power sequences of K target appliances from aggregate active-power measurements. Let $\mathbf { x } _ { i } \in \mathbb { R } ^ { T }$ denote the i-th aggregate power window of length $T ,$ and let $\mathbf { y } _ { i , k } ~ \in ~ \mathbb { R } ^ { T }$ denote the timealigned power sequence of target appliance k, $k = 1 , \ldots , K$ We stack the target-appliance power sequences as $\begin{array} { r l } { \mathbf { Y } _ { i } } & { { } = } \end{array}$ $[ \mathbf { y } _ { i , 1 } , \dots , \mathbf { y } _ { i , K } ] ^ { \top } \in \mathbb { R } ^ { K \times T }$

The disaggregation mapping is

$$
\begin{array} { r } { \widehat { \mathbf { Y } } _ { i } = F _ { \theta } ( \mathbf { x } _ { i } ) , \qquad F _ { \theta } : \mathbb { R } ^ { T } \to \mathbb { R } ^ { K \times T } } \end{array}\tag{1}
$$

where $\theta$ denotes the learnable parameters. The model is trained only on labeled windows from source households and evaluated on unseen households that contribute no training data.

The aggregate window is written as

$$
\mathbf { x } _ { i } = \sum _ { k = 1 } ^ { K } \mathbf { y } _ { i , k } + \mathbf { b } _ { i } , \qquad \mathbf { b } _ { i } = \mathbf { x } _ { i } - \sum _ { k = 1 } ^ { K } \mathbf { y } _ { i , k }\tag{2}
$$

where $\mathbf { b } _ { i } \in \mathbb { R } ^ { T }$ is the residual background defined by (2), collecting aggregate power not represented by the targetappliance power sequences together with residual measurement and preprocessing discrepancies.

For fixed target-appliance power sequences $\mathbf { Y } _ { i } ,$ , the residual background can still vary, so the additive observation model admits many aggregate windows for the same targets. It therefore permits a label-preserving aggregate recomposition that leaves $\mathbf { Y } _ { i }$ exactly unchanged, with label preservation guaranteed by the construction itself. The additive form also permits a complementary construction: with the residual background held fixed, substituting a target-appliance power sequence yields another additively consistent aggregate window, with its targets recomputed from the substituted sequence. That construction produces additional training windows; only the direction that holds $\mathbf { Y } _ { i }$ fixed yields two windows carrying identical targets. This label-preserving relation arises from the additive decomposition and the availability of time-aligned submetered measurements, and is therefore a property of the labeled problem setting rather than of any particular training method. Section III describes how the proposed method uses this relation to construct anchor–recomposed pairs and selectively constrain their per-appliance power predictions.

## III. PROPOSED METHOD

In this section, we present the proposed method for multiappliance NILM under window-wise supervision. We first develop the FLAME architecture that produces one power and operating-state prediction per appliance, then define the two window constructions it is trained on, then define prediction consistency on the resulting outputs, and finally state the overall training objective.

## A. FLAME Multi-Appliance Architecture

FLAME is designed to share aggregate-level context across appliances while forming a distinct representation for each. It does so within a single model through two-stage sharedto-specific expert routing, illustrated in Fig. 1(a). A backbone extracts the initial shared representation. The many-toone aggregation stage routes the responses of all appliance branches into a single shared representation, and the oneto-many refinement stage expands it back into K appliance views, each refined into an appliance representation from which the power and operating-state predictions are produced. The two stages are ordered so that information is pooled across appliances before it is specialized. The design builds on the multi-gate expert-routing paradigm of multi-gate mixture-ofexperts (MMoE) [18] and the shared/specific expert organization of progressive layered extraction (PLE) [19]. Throughout this subsection the index k ranges over the K appliances, each carrying a power and an operating-state output, and the window index is omitted.

![](images/5f274e71ef8f86a15864883306ceba74535ddc1995e234d21ecf84c388e6f649.jpg)  
Fig. 1. FLAME architecture. (a) Forward topology from the backbone output $\mathbf { H } ^ { ( 0 ) }$ through the many-to-one aggregation levels and the one-to-many refinement level to the appliance representations $\mathbf { Z } _ { k }$ and their prediction heads; rows $k = 1$ and k = K stand for all K appliances, and line styles and colors are defined in the legend (SE: squeeze-and-excitation). (b) The ResTCN backbone of (3), with the stem and one residual block expanded (LN: layer normalization, GN: group normalization). (c) The routing operator Route(·) of (4) and (5), used with J = 2K+1 in (7) and J = 3 in (9).

A residual temporal-convolutional backbone (ResTCN) first maps the aggregate to the initial shared representation,

$$
{ \bf H } ^ { ( 0 ) } = { \boldsymbol B } ( { \bf x } ) \in \mathbb { R } ^ { C \times T }\tag{3}
$$

where B denotes the backbone, C the feature-channel dimension, and the superscript indexes routing levels. Its stem and multi-scale residual blocks, shown in Fig. 1(b), combine local and dilated temporal convolutions following the sequencemodeling principle of residual temporal convolutional networks [20].

Both stages use the same routing form with separately parameterized gates and different candidate sets. For a routed branch with feature-gated view He $\in \mathbb { R } ^ { C \times T }$ and J candidate expert responses $\mathbf { E } _ { 1 } , \ldots , \mathbf { E } _ { J }$ of the same shape, we write

Route(·) for the gated mixing operation of Fig. 1(c),

$$
\operatorname { R o u t e } \bigl ( \widetilde { \mathbf { H } } ; \mathbf { E } _ { 1 } , \ldots , \mathbf { E } _ { J } \bigr ) = \eta \widetilde { \mathbf { H } } + ( 1 - \eta ) \sum _ { m = 1 } ^ { J } w _ { m } \mathbf { E } _ { m }\tag{4}
$$

where the mixing weights $\mathbf { w } = [ w _ { 1 } , \ldots , w _ { J } ] ^ { \top }$ are produced by a routing gate and the interpolation coefficient η by a selfgate, both reading the temporally pooled view,

$$
{ \bf w } = \mathrm { s o f t m a x } \left( g ( \mathrm { a v g } _ { t } \widetilde { \bf H } ) \right) , \qquad \eta = \sigma \left( g ^ { \mathrm { s e l f } } ( \mathrm { a v g } _ { t } \widetilde { \bf H } ) \right)\tag{5}
$$

Here avg averages over time, $\sigma$ is the logistic sigmoid, and g and $g ^ { \mathrm { s e l f } }$ are multilayer perceptrons (MLPs) that output J routing logits and a single scalar, respectively. The routing weights form a sample-wise dense softmax, so every candidate response enters the mixture with a continuous weight. The scalar self-gate keeps a direct path from the branch’s own gated view, so the routed output interpolates between that view and the expert mixture. The two gates thus make separate decisions: how the candidate responses are composed, and how far the routed output departs from the branch’s own view. The level index is suppressed on intermediate views and expert responses within a level, and the same operator names are reused across levels and stages for independently parameterized modules. The configuration of these modules used in the experiments is given in Section IV-B.

1) Cross-Appliance Aggregation: Let N denote the total number of routing levels: the first $N - 1$ perform crossappliance aggregation and the final one performs appliancespecific refinement. Aggregation level $j \in \{ 1 , \ldots , N - 1 \}$ takes the shared representation $\mathbf { H } ^ { ( j - 1 ) }$ produced by the preceding level, the first level taking $\mathbf { H } ^ { ( 0 ) }$ from the backbone. Squeeze-and-excitation (SE) feature gates [21] first form a shared view, marked by the subscript s, and from it one appliance-conditioned view per appliance, rebuilt from the current shared representation at every level,

$$
\mathbf { H } _ { s } = \mathrm { S E } _ { s } \big ( \mathbf { H } ^ { ( j - 1 ) } \big ) , \qquad \mathbf { H } _ { k } = \mathrm { S E } _ { k } \big ( \mathbf { H } _ { s } \big )\tag{6}
$$

Because each gate pools its input over time and rescales the feature channels, conditioning leaves the temporal axis untouched and every view keeps the $C \times T$ interface of the shared representation. All experts preserve the $C \times T$ shape. A shared temporal-convolution expert, TConv, is evaluated on the shared view; in both stages it gives the router one response computed from the shared view alone, so aggregatelevel context that no appliance gate has reshaped remains a routing candidate. Two heterogeneous experts are evaluated on each appliance view, consistent with the heterogeneous-expert design of [22]: a bidirectional gated recurrent unit (BiGRU) expert [23], [24] and a bidirectional Mamba (BiMamba) expert based on input-dependent selective state-space recurrence [25]. The two experts present the router with responses of distinct temporal inductive bias computed from the same appliance view. The shared router mixes all candidates with the shared view,

$$
\begin{array} { r } { \begin{array} { r l } & { { \bf E } _ { 2 k - 1 } = \mathrm { B i G R U } _ { k } ( { \bf H } _ { k } ) , \qquad { \bf E } _ { 2 k } = \mathrm { B i M a m b a } _ { k } ( { \bf H } _ { k } ) , } \\ & { \qquad { \bf E } _ { 2 K + 1 } = \mathrm { T C o n v } ( { \bf H } _ { s } ) , } \\ & { \qquad { \bf H } ^ { ( j ) } = \mathrm { R o u t e } \big ( { \bf H } _ { s } ; { \bf E } _ { 1 } , \ldots , { \bf E } _ { 2 K + 1 } \big ) , \qquad j = 1 , \ldots , N - 1 } \end{array} } \end{array}\tag{7}
$$

The updated shared representation $\mathbf { H } ^ { ( j ) }$ is the only quantity passed to the next level.

2) Appliance-Specific Refinement: This stage, the final routing level, reverses the flow to one-to-many, starting from the single shared representation ${ \bf H } ^ { ( N - 1 ) }$ ; its views are written U to mark the change of stage,

$$
{ \bf U } _ { s } = \mathrm { S E } _ { s } ( { \bf H } ^ { ( N - 1 ) } ) , \qquad { \bf U } _ { k } = \mathrm { S E } _ { k } ( { \bf U } _ { s } )\tag{8}
$$

Each appliance now owns an independent three-way router whose candidate set pairs the shared response, common to all appliances, with that appliance’s own two expert responses, giving the appliance representation

$$
\begin{array} { c } { { \displaystyle { \bf E } _ { k , 1 } = \mathrm { T C o n v } ( { \bf U } _ { s } ) , \qquad { \bf E } _ { k , 2 } = \mathrm { B i G R U } _ { k } ( { \bf U } _ { k } ) , } } \\ { { \displaystyle { \bf E } _ { k , 3 } = \mathrm { B i M a m b a } _ { k } ( { \bf U } _ { k } ) , } } \\ { { \displaystyle { \bf Z } _ { k } = \mathrm { R o u t e } \big ( { \bf U } _ { k } ; { \bf E } _ { k , 1 } , { \bf E } _ { k , 2 } , { \bf E } _ { k , 3 } \big ) } } \end{array}\tag{9}
$$

Two prediction heads map each appliance representation $\mathbf { Z } _ { k }$ to predictions: a regression head $h _ { k } ^ { \mathrm { r e g } }$ produces the regression sequence $\widehat { \mathbf { r } } _ { k } .$ , a state-classification head $h _ { k } ^ { \mathrm { s t a t e } }$ produces the state-logit sequence $\widehat { \ell } _ { k }$ , and one deterministic combination yields the gated power,

$$
\begin{array} { c } { { { \widehat { \mathbf { r } } } _ { k } = h _ { k } ^ { \mathrm { r e g } } ( \mathbf { Z } _ { k } ) , \qquad \widehat { \ell } _ { k } = h _ { k } ^ { \mathrm { s t a t e } } ( \mathbf { Z } _ { k } ) , } } \\ { { { \widehat { \mathbf { y } } } _ { k } = { \widehat { \mathbf { r } } } _ { k } \odot \sigma ( \widehat { \ell } _ { k } ) } } \end{array}\tag{10}
$$

where ⊙ denotes elementwise multiplication; the gated power is thus a parameter-free function of the two head outputs. Stacking the K gated outputs gives $\widehat { \mathbf { Y } } = [ \widehat { \mathbf { y } } _ { 1 } , \ldots , \widehat { \mathbf { y } } _ { K } ] ^ { \top }$ , the output of $F _ { \theta }$ in (1). For training, each appliance is additionally associated with a binary operating-state target ${ \bf s } _ { k } \in \{ 0 , 1 \} ^ { T }$ obtained by applying the configured operating-state threshold to $\mathbf { y } _ { k }$ in the raw-power domain, and these targets are stacked as $\mathbf { S } = [ \mathbf { s } _ { 1 } , \hdots , \mathbf { s } _ { K } ] ^ { \top } \in \{ 0 , 1 \} ^ { K \times T }$ . They supervise the state heads in the training objective of Section III-D.

## B. Residual-Based Window Construction

Both training constructions run on the same operation. The residual background of a recorded window is separated by (2), one side of that sum is replaced, and the aggregate is rebuilt through the same equation. The additive relation therefore holds exactly on every constructed window. Holding the residual background fixed and substituting target-appliance sequences produces further training windows, whose targets are recomputed from the substituted sequences. Holding the target sequences fixed and substituting the residual background instead produces two windows that carry identical targets. These are the pairs used for prediction consistency.

1) Training-Window Synthesis: Several target appliances operate for only a small fraction of the recording. Windows drawn uniformly from the source pool are therefore dominated by intervals in which those appliances are off, leaving few examples of the behavior the model must learn to separate. We synthesize training windows in the fixed-background direction of this construction: the residual background of a recorded window, termed the host window, is retained, and one targetappliance power sequence is substituted.

Appliance k is treated as sparse when the fraction of samples in the training portion whose power exceeds its on power threshold $\tau _ { k }$ falls below a sparsity threshold $\rho _ { \mathrm { s p } }$ This threshold governs which segments and windows the construction draws on; it is a per-appliance power level and is not the operating-state threshold that produces the state targets s<sub>k</sub>. Each candidate window is labeled by the sparse appliance for which more than a minimum number of the window’s samples exceed $\tau _ { k } ;$ these samples need not be consecutive. Every mini-batch is then filled to a fixed per-appliance quota rather than to a probability in expectation. The appliance assigned to a position in the batch is resampled, over segments and scaling modes, until the synthesized window meets the same requirement for it. The assigned count and the activation it stands for therefore coincide by construction. The quota fixes how many positions are assigned to each appliance, not the number of windows in which that appliance is active. Each position carries a single assignment; any other sparse appliance active in the host window is left as recorded.

Within a host window, the residual background $\mathbf { b } _ { i }$ is held fixed. The power sequence of the assigned appliance is substituted by an activation segment drawn from a per-appliance pool of segments detected on the training portion, and the aggregate is rebuilt through (2). The additive relation continues to hold exactly, and both targets follow from the substituted sequence. Each segment may be rescaled in amplitude, in duration, or in both before insertion; duration scaling is implemented by interpolation, after which the segment is inserted with the partial-visibility semantics of [8].

![](images/aa67623e0825d70b0952f9a5222cd38df811ad4e8ea5f621685e4d6ef1a32213.jpg)  
Fig. 2. Label-preserving aggregate recomposition and training objective. The appliance power of window a is added to its own residual background ${ \bf b } _ { a }$ and to that of another recorded window $j ,$ giving two aggregates that differ only in the residual background (11) and share the targets $( \mathbf { Y } _ { a } , \mathbf { S } _ { a } )$ ; one model predicts both windows under the full task loss, and ${ \mathcal { L } } _ { \mathrm { c o n s } }$ uses predictions of the same pair from a second forward pass with dropout disabled. Traces: Reference Energy Disaggregation Data Set (REDD).

2) Label-Preserving Aggregate Recomposition: The anchor of a pair is itself produced by the synthesis of Section III-B1. The anchor retains its host window’s residual background, while the replacement residual background comes from another recorded source window. Both are computed from the measured aggregate and target-appliance sequences of their source windows as in (2), and the paired windows share the same synthesized target matrix. Fig. 2 illustrates such a pair.

Constructing the pair requires a replacement residual background that differs from the anchor window’s while leaving every modeled appliance target untouched. For anchor window $^ { a , }$ window j is drawn from a pool of recorded source windows.

Using (2), oriented pair $p = ( a , j )$ is

$$
\begin{array} { l l } { { { \displaystyle { \bf { b } } _ { a } } = { \bf { x } } _ { a } - \sum _ { k = 1 } ^ { K } { \bf { y } } _ { a , k } } , } & { { { \bf { b } } _ { j } } = { \bf { x } } _ { j } - \sum _ { k = 1 } ^ { K } { \bf { y } } _ { j , k } , }  \\ { { { \displaystyle \bf { x } } _ { p } ^ { A } = \sum _ { k = 1 } ^ { K } { \bf { y } } _ { a , k } + { \bf { b } } _ { a } } , } & { { { \bf { x } } _ { p } ^ { B } = \sum _ { k = 1 } ^ { K } { \bf { y } } _ { a , k } + { \bf { b } } _ { j } } , } \\ { ( { \bf Y } _ { p } ^ { A } , { \bf S } _ { p } ^ { A } ) = ( { \bf Y } _ { p } ^ { B } , { \bf S } _ { p } ^ { B } ) = ( { \bf Y } _ { a } , { \bf S } _ { a } ) } & { { } \qquad \quad , } \end{array}\tag{11}
$$

where the superscripts A and B index the anchor and the recomposed window of the pair. Every modeled appliance power sequence is preserved pointwise. The operating-state targets are obtained from those sequences by thresholding, so they are preserved with them. This label preservation is an arithmetic identity of the construction: the two aggregates differ exactly by $\mathbf { b } _ { j } - \mathbf { b } _ { a }$

Admissibility. The replacement residual background must be numerically valid, and the recomposed aggregate must stay within the configured physical range. Candidates failing either check are resampled; if no valid replacement is found, sampling stops with an error.

Scope. The recomposed window is obtained by labelpreserving aggregate recomposition; it is not presented as a sample from the true joint distribution or as an identified causal intervention.

## C. Prediction Consistency

Given the label-preserving pair of Section III-B2, both windows receive the same complete power and operating-state task supervision. The consistency term additionally penalizes disagreement between their corresponding per-appliance power predictions only when both predictions satisfy a fixed reliability criterion and only to the extent that their disagreement exceeds a fixed margin. That penalty is placed on the gated power outputs of (10) rather than on the representations that produce them. The two windows carry pointwise identical targets, so agreement between their predictions concerns the same quantity the task loss already scores. Agreement between their representations would additionally require the two windows to be encoded identically despite their different residual backgrounds, which is more than the construction establishes.

1) Reliability Gate and Margin: Exact label preservation makes output consistency well defined, but it does not determine when or how strictly agreement should be enforced during training. The consistency term therefore uses two appliance-specific thresholds fixed in advance: a reliability threshold $\gamma _ { k }$ determines whether pair $p$ enters the term for appliance k, and a margin $\varepsilon _ { k }$ determines how much disagreement the term tolerates.

The per-window task losses are evaluated on the ordinary training forward pass, including the configured dropout. For the consistency term, the same paired batch is evaluated a second time with the dropout modules disabled while all other modules remain in their training states. Throughout (12)– (14), $\widehat { y } _ { p , k , t } ^ { v }$ denotes a prediction from this dropout-disabled pass. Both windows’ predictions remain differentiable; only the binary gate defined below is detached.

Let $y _ { p , k , t }$ denote the common target of appliance k at time t in pair $p .$ For each appliance, the reliability gate is computed from the window-mean absolute error of each window’s prediction against that target,

$$
\begin{array} { c } { { \displaystyle e _ { p , k } ^ { v } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left| \widehat { y } _ { p , k , t } ^ { v } - y _ { p , k , t } \right| , } } \\ { { \omega _ { p , k } = \nVdash \left[ e _ { p , k } ^ { A } \leq \gamma _ { k } \right] \nVdash \left[ e _ { p , k } ^ { B } \leq \gamma _ { k } \right] } } \end{array}\tag{12}
$$

![](images/8ec756c2d50261d68d46d01539edbfd7143a7347d426334d1839be3f69dda017.jpg)  
Fig. 3. The consistency term drawn on the three window-mean quantities of (12)–(13), whose values the distances reproduce: the target is the center, the reliability gate is the circle of radius $\gamma _ { k } ,$ , which a pair passes only when both predictions lie inside it, and the disagreement is the side between the two predictions, penalized only beyond the margin $\varepsilon _ { k }$ . The four cases are the four combinations of the gate and the margin; only the first carries a penalty, and the gated excess is averaged over pairs and appliances (14).

so the pair enters the term for appliance k only when both windows are already predicted to within $\gamma _ { k } .$ This excludes any pair in which either window is predicted unreliably, so the term never couples a reliable prediction with an unreliable one; it is also the only place where the target enters the consistency term. The gate is evaluated without gradient. It depends on the model’s own prediction errors, so a gate that carried gradient would allow the term to be reduced by closing the gate rather than by reducing disagreement.

For a pair that passes the gate, the window-mean disagreement between the two predictions is defined as

$$
d _ { p , k } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } | \widehat { y } _ { p , k , t } ^ { A } - \widehat { y } _ { p , k , t } ^ { B } |\tag{13}
$$

and the consistency term penalizes it only in excess of the margin, as $( d _ { p , k } - \varepsilon _ { k } ) .$ with $( \cdot ) _ { + } = \operatorname* { m a x } ( \cdot , 0 )$ . Once disagreement is within the margin, the consistency penalty is zero and the task loss continues to fit each window to the common target. Fig. 3 draws the gate and the margin on these three windowmean quantities.

Both thresholds are set before training with the consistency term, from dropout-disabled predictions of checkpoints trained without it: $\gamma _ { k }$ is a quantile of max $( e _ { p , k } ^ { A } , e _ { p , k } ^ { B } )$ over sourceside training pairs of those checkpoints, and ε<sub>k</sub> is a quantile of $d _ { p , k }$ over the pairs that pass the resulting gate. Setting them updates no model parameters, and they are held fixed throughout training. Were they adapted while training, the term could shift toward pairs that are easier to satisfy, and any resulting change could not be attributed to a stated rule.

2) Output Consistency: Let $A \subseteq \{ 1 , \ldots , K \}$ denote the appliances to which the term is applied and P the number of pairs in a mini-batch. The consistency term averages the gated excess disagreement over pairs and then over appliances:

$$
\mathcal { L } _ { \mathrm { c o n s } } = \frac { 1 } { \left| \boldsymbol { A } \right| } \sum _ { k \in \mathcal { A } } \frac { 1 } { P } \sum _ { p = 1 } ^ { P } \omega _ { p , k } \left( d _ { p , k } - \varepsilon _ { k } \right) _ { + }\tag{14}
$$

A pair closed by the gate contributes zero to the numerator and stays in the denominator, so the term scales with how often the gate opens instead of inflating when few pairs qualify. Averaging within an appliance before combining appliances keeps an appliance with many gate-open pairs from dominating one with few.

Both windows carry the same targets and receive the same task loss, so ${ \mathcal { L } } _ { \mathrm { c o n s } }$ adds no target information. Write $e _ { p , k , t } ^ { v } = \widehat { y } _ { p , k , t } ^ { v } - y _ { p , k , t }$ for the pointwise prediction error against the common target; then $| \widehat { y } _ { p , k , t } ^ { A } - \widehat { y } _ { p , k , t } ^ { B } | = | e _ { p , k , t } ^ { A } - e _ { p , k , t } ^ { B } |$ at every position. The disagreement $d _ { p , k }$ is therefore the windowmean absolute difference between the two pointwise prediction errors and cancels any additive pointwise error component shared by both windows. This cancellation applies to $d _ { p , k }$ not to the full gated consistency loss, because the reliability gate depends on each window’s mean absolute error. The term is evaluated only on gate-open pairs and only in excess of the margin, but its gradient reaches the complete upstream computation that produces those predictions, including the shared stage of Section III-A1. The gate and the margin therefore determine whether the term is active, not which appliances or parameters the resulting updates can affect.

## D. Overall Training Objective

For $v \in \{ A , B \} , \operatorname { l e t } \widehat { \mathbf { R } } ^ { v } , \widehat { \ell } ^ { v }$ , and ${ \widehat { \mathbf { Y } } } ^ { v }$ stack the raw regression outputs, state logits, and gated power outputs of (10) across appliances and time. Using the common anchor targets (Y, S), the task and total objectives are

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { t a s k } } ^ { v } = \alpha _ { \mathrm { r e g } } \mathrm { M S E } ( \widehat { \mathbf { R } } ^ { v } , \mathbf { Y } ) + \alpha _ { \mathrm { s t a t e } } \mathrm { B C E } ( \widehat { \boldsymbol { \ell } } ^ { v } , \mathbf { S } ) } \\ & { \qquad + \alpha _ { \mathrm { g a t e } } \mathrm { M S E } ( \widehat { \mathbf { Y } } ^ { v } , \mathbf { Y } ) , } \\ & { \mathcal { L } = \frac { 1 } { 2 } \left( \mathcal { L } _ { \mathrm { t a s k } } ^ { A } + \mathcal { L } _ { \mathrm { t a s k } } ^ { B } \right) + \lambda _ { \mathrm { c o n s } } \mathcal { L } _ { \mathrm { c o n s } } } \end{array}\tag{15}
$$

where MSE denotes the mean squared error and BCE the binary cross-entropy evaluated from logits. Every component of the per-window task loss is applied to both windows, and the two per-window task losses are weighted equally. At each update, the task-loss gradient is computed first, the dropout-disabled consistency pass is then evaluated, and its gradient is accumulated into the same parameter gradients before one clipping operation and one optimizer step. This sequential accumulation implements the gradient sum in (15) without retaining both computation graphs simultaneously. The loss weights and optimizer settings are reported with the experimental settings.

Training uses the recomposed window, the two fixed thresholds, and ${ \mathcal { L } } _ { \mathrm { c o n s } } ;$ inference uses none of them and runs the standard single-window forward path. Relative to the same checkpoint, the proposed method therefore introduces no additional inference-time module or parameter.

## IV. EXPERIMENTS AND RESULTS

We evaluate the complete method of Section III against single-window FLAME on three public low-frequency data sets, and compare the FLAME architecture with four alternative multi-appliance architectures under a common training protocol.

## A. Datasets and Evaluation Metrics

Experiments are conducted on REDD [26], UK-DALE [27], and REFIT [28]. On REDD we jointly model the dishwasher, refrigerator, microwave, and washer–dryer; on UK-DALE and REFIT the kettle, microwave, refrigerator, dishwasher, and washing machine. REDD and UK-DALE are used at their $6 { - } \mathrm { s }$ resolution and REFIT at 8 s; all power channels are scaled by a constant of 612 W. Training and testing use disjoint households: House 3 of REDD, House 1 of UK-DALE, and Houses 3, 5, 9, and 20 of REFIT are used for training, and House 1 of REDD and House 2 of UK-DALE and REFIT for testing. Each training recording is split chronologically into a training portion and a validation portion (the final 30% on REDD and 5% on UK-DALE and REFIT) before windowing, and the test household contributes no training data. Target-appliance segments for training-window synthesis are drawn from additional training-only households (Houses 2, 4, 5, and 6 of REDD, House 5 of UK-DALE, and fifteen further houses of REFIT). Signals are divided into 720-sample windows with a stride of 120 samples.

Three metrics are computed for each appliance k over the $N _ { \mathrm { w } }$ test windows: mean absolute error (MAE), signal aggregate error (SAE), and the F1 score of the operating state,

$$
\begin{array} { r l } & { \mathrm { M A E } _ { k } = \displaystyle \frac { 1 } { N _ { \mathrm { w } } T } \sum _ { i = 1 } ^ { N _ { \mathrm { w } } } \sum _ { t = 1 } ^ { T } \left| \widehat { y } _ { i , k , t } - y _ { i , k , t } \right| , } \\ & { \mathrm { S A E } _ { k } = \displaystyle \frac { 1 } { N _ { \mathrm { w } } T } \sum _ { i = 1 } ^ { N _ { \mathrm { w } } } \left| \sum _ { t = 1 } ^ { T } \widehat { y } _ { i , k , t } - \sum _ { t = 1 } ^ { T } y _ { i , k , t } \right| , } \\ & { \quad \quad \quad \quad \quad \mathrm { F } 1 _ { k } = \displaystyle \frac { 2 P _ { k } R _ { k } } { P _ { k } + R _ { k } } } \end{array}\tag{16}
$$

where $\widehat { y } _ { i , k , t }$ and $y _ { i , k , t }$ are the predicted and target power of appliance k at sample t of test window i, both in watts, and $P _ { k }$ and $R _ { k }$ are the precision and recall of the predicted operating state $\widehat { s } _ { i , k , t } = \mathcal { k } \left[ \sigma ( \widehat { \ell } _ { i , k , t } ) > 0 . 3 \right]$ against the operating-state target $s _ { i , k , t }$ over all test-window samples, with 0.3 the stateprobability threshold. Macro values are unweighted means over appliances.

## B. Experimental Setup

We compare two variants. The baseline, single-window FLAME, trains FLAME on single windows with $\mathcal { L } _ { \mathrm { t a s k } }$ alone. The proposed method constructs the label-preserving pair, applies the complete task loss to both windows, evaluates the consistency term on the dropout-disabled pass of Section III-C1, and sets $\lambda _ { \mathrm { c o n s } } = 0 . 8$ . The comparison therefore evaluates the proposed method as a whole rather than attributing its result to an individual component. All reported values are averaged over three independent trials; for macro MAE we also report the sample standard deviation and the number of paired runs in which the proposed method has lower error than single-window FLAME.

Both variants use the same FLAME configuration: a ResTCN backbone of width 256 and depth 4 (parallel kernels 3/9/3, block dilations 7/11/17/23), $N = 2$ routing levels, a shared TConv expert with kernel 3, per-appliance BiGRU (hidden size 128) and BiMamba (state size 16) experts, SE gates without channel reduction, gate MLPs of width 128, and dropout 0.1. They are trained separately from scratch with AdamW (learning rate $1 0 ^ { - 3 } .$ , weight decay $1 0 ^ { - 2 } )$ for 45 epochs of 30 updates on REDD, 90 epochs of 50 updates on UK-DALE, and 60 epochs of 50 updates on REFIT; singlewindow FLAME uses 64 windows per batch, and the proposed method uses 64 pairs, or 128 windows, with complete power and state supervision on both members of every pair. Operating-state targets use a 10-W threshold, and the task-loss weights are $( \alpha _ { \mathrm { r e g } } , \alpha _ { \mathrm { s t a t e } } , \alpha _ { \mathrm { g a t e } } ) = ( 2 , 1 , 1 )$

For training-window synthesis (Section III-B1), activation segments are detected on the training portion before windowing with per-appliance on power thresholds $\tau _ { k } .$ , and the sparsity threshold is $\rho _ { \mathrm { s p } } ~ = ~ 0 . 2$ . For each paired example, the replacement residual background is taken from another recorded window. The consistency set A contains the dishwasher, microwave, and washer–dryer on REDD; the kettle, microwave, refrigerator, dishwasher, and washing machine on $\mathrm { U K - D A L E } ;$ and the kettle, microwave, dishwasher, and washing machine on REFIT. Appliances outside A remain fully supervised in both windows but have zero weight in $\mathcal { L } _ { \mathrm { c o n s } } .$

A separate experiment varies the architecture instead of the training method. Five multi-appliance architectures— HMMOE [22], MATNilm [8], BERT4NILM [29], MA-NILM [30], and FLAME—are trained and evaluated on each of the three data sets. All five follow one common protocol with the same households, windowing, task loss, optimizer settings, per-data-set training budget, and inference procedure; no pair is constructed, the consistency term is not applied, and no hyperparameter is tuned for an individual architecture. BERT4NILM is trained directly under this protocol and does not reproduce the masked training of the original model, and MA-NILM is a sequence-to-sequence adaptation of a sequence-to-point design.

## C. Experimental Results

Table I reports MAE, SAE, and F1 on the unseen test household of each data set. The proposed method attains lower macro MAE and SAE than single-window FLAME on all three data sets. On REDD, macro MAE falls from $1 4 . 7 5 \pm 0 . 3 0$ to $1 3 . 1 4 \pm 0 . 5 0 \mathrm { ~ W } ,$ a reduction of 1.61 W (10.9%) with lower error in three of three paired runs and a paired 95% confidence interval of $[ - 2 . 5 9 , - 0 . 6 2 ]$ W, and macro SAE falls from 12.46 to 10.64 W. On REFIT, macro MAE falls from $1 5 . 8 3 { \pm } 0 . 6 2 $ to $1 4 . 5 5 { \pm } 0 . 6 6 $ W (1.28 W, 8.1%; three of three runs) and macro SAE from 13.03 to 12.04 W. On UK-DALE, macro MAE falls from $8 . 8 8 \pm 1 . 5 3 $ to 8.51 ± 0.36 W (0.37 W, 4.2%) and macro SAE from 7.22 to 6.85 W; on UK-DALE and REFIT the paired 95% confidence intervals of the macro differences include zero.

TABLE I  
RESULTS ON THE UNSEEN TEST HOUSEHOLD OF REDD, UK-DALE, AND REFIT. DW, FR, MW, WD, KT, AND WM DENOTE THE DISHWASHER, REFRIGERATOR, MICROWAVE, WASHER–DRYER, KETTLE, AND WASHING MACHINE; AVE IS THE UNWEIGHTED MEAN OVER THE APPLIANCES OF EACH DATA SET. PROPOSED DENOTES THE COMPLETE METHOD TRAINED ON LABEL-PRESERVING PAIRS. BOLD MARKS THE BETTER VALUE.
<table><tr><td></td><td></td><td colspan="5">REDD</td><td colspan="5">UK-DALE</td><td colspan="5">REFIT</td><td colspan="5"></td></tr><tr><td>Metric</td><td>Model</td><td>DW</td><td>FR</td><td>MW</td><td>WD</td><td>Ave</td><td></td><td>KT</td><td>MW</td><td>FR</td><td>DW</td><td>WM</td><td>Ave</td><td>KT</td><td>MW</td><td></td><td>FR</td><td>DW</td><td>WM</td><td>Ave</td></tr><tr><td>MAE (W)</td><td>Single-window FLAME</td><td>12.33</td><td>22.77</td><td>14.70</td><td>9.18</td><td>14.75</td><td></td><td>3.85</td><td>3.23</td><td>16.32</td><td>16.18</td><td>4.81</td><td>8.88</td><td>16.19</td><td>2.69</td><td>26.08</td><td></td><td>25.11</td><td>9.07</td><td>15.83</td></tr><tr><td rowspan="2">SAE (W)</td><td>Proposed</td><td>10.22</td><td>20.48</td><td>14.45</td><td>7.40</td><td>13.14</td><td></td><td>4.37</td><td>2.90</td><td>17.47</td><td>14.34</td><td>3.46</td><td>8.51</td><td>16.80</td><td>2.71</td><td>22.18</td><td></td><td>21.78</td><td>9.28</td><td>14.55</td></tr><tr><td>Single-window FLAME</td><td>11.85</td><td>16.76</td><td>13.35</td><td>7.90</td><td>12.46</td><td></td><td>3.24</td><td>2.94</td><td>11.67</td><td>14.94</td><td>3.30</td><td>7.22</td><td>15.08</td><td>2.49</td><td>20.29</td><td></td><td>21.20</td><td>6.11</td><td>13.03</td></tr><tr><td rowspan="2">F1</td><td>Proposed</td><td>9.49</td><td>14.58</td><td>12.77</td><td>5.72</td><td>10.64</td><td></td><td>3.79</td><td>2.65</td><td>13.10</td><td>12.89</td><td>1.84</td><td>6.85</td><td>16.05</td><td>2.49</td><td>17.93</td><td></td><td>17.03</td><td>6.70</td><td>12.04</td></tr><tr><td>Single-window FLAME</td><td>0.75</td><td>0.86</td><td>0.53</td><td>0.91</td><td></td><td>0.76</td><td>0.98</td><td>0.42</td><td>0.72</td><td>0.75</td><td>0.80</td><td>0.73</td><td>0.70</td><td>0.26</td><td></td><td>0.77</td><td>0.73</td><td>0.79</td><td>0.65</td></tr><tr><td></td><td>Proposed</td><td>0.78</td><td>0.86</td><td>0.52</td><td></td><td>0.91</td><td>0.76</td><td>0.98</td><td>0.43</td><td>0.71</td><td>0.74</td><td>0.86</td><td>0.74</td><td>0.62</td><td>0.27</td><td></td><td>0.82</td><td>0.77</td><td>0.80</td><td>0.66</td></tr></table>

TABLE II

MACRO MAE (W) OF FIVE ARCHITECTURES UNDER THE COMMON TRAINING PROTOCOL, AS MEAN ± SAMPLE STANDARD DEVIATION OVER THREE INDEPENDENT TRIALS. BOLD MARKS THE LOWEST VALUE ON EACH DATA SET.
<table><tr><td>Architecture</td><td>REDD</td><td>UK-DALE</td><td>REFIT</td></tr><tr><td>HMMOE</td><td> $1 5 . 9 8 \pm 0 . 2 2$ </td><td> $1 0 . 8 8 \pm 0 . 8 2$ </td><td> $1 9 . 7 8 \pm 2 . 1 1$ </td></tr><tr><td>MATNilm</td><td> $1 7 . 0 4 \pm 1 . 7 9$ </td><td> $1 0 . 2 3 \pm 1 . 0 4$ </td><td> $1 8 . 7 5 \pm 0 . 2 6$ </td></tr><tr><td>BERT4NILM</td><td> $1 6 . 7 0 \pm 0 . 9 6$ </td><td> $1 3 . 6 0 \pm 1 . 3 1$ </td><td> $2 3 . 5 3 \pm 1 . 0 3$ </td></tr><tr><td>MA-NILM</td><td> $2 6 . 7 4 \pm 0 . 7 7$ </td><td> $1 4 . 6 3 \pm 0 . 7 8$ </td><td> $2 2 . 8 7 \pm 0 . 7 8$ </td></tr><tr><td>FLAME</td><td> $\mathbf { 1 4 . 0 7 \ : \pm { \ : 0 . 9 1 } }$ </td><td> ${ \bf 8 . 3 0 \pm 0 . 2 5 }$ </td><td> $\mathbf { 1 4 . 5 7 \ : \pm { \ : 0 . 2 4 } }$ </td></tr></table>

## D. Architecture Comparison

Table II reports the macro MAE of the five architectures on the unseen test household of each data set. FLAME attains the lowest macro MAE among the compared architectures under this common protocol on all three data sets. Because the four alternatives are reimplemented under this protocol without architecture-specific tuning, the table compares them under a single shared optimization setting rather than at the best configuration reported for each original method; the ranking may therefore reflect both architectural properties and compatibility with that setting.

## V. CONCLUSION

This paper introduced a method for multi-appliance NILM that combines label-preserving aggregate recomposition with prediction consistency during training. The construction uses the additive decomposition of aggregate power and timealigned submetered measurements to recompose an aggregate window so that only its residual background changes while every modeled appliance power and operating-state target is preserved pointwise. Training then constrains the relation between the per-appliance power predictions of the two windows under complete task supervision. The consistency term acts only on pairs that pass a reliability gate and only on disagreement beyond a margin; both thresholds are fixed in advance and together determine whether the term is active, without restricting which parameters its gradient can reach. The proposed method was implemented using FLAME, a multi-appliance architecture with two-stage shared-to-specific expert routing. At deployment the model runs a standard single-window forward pass, so the proposed method adds no inference-time module or parameter.

On REDD, UK-DALE, and REFIT, each evaluated on an unseen household, the proposed method with $\lambda _ { \mathrm { c o n s } } ~ = ~ 0 . 8$ lowers macro MAE from 14.75 to 13.14 W, from 8.88 to 8.51 W, and from 15.83 to 14.55 W relative to single-window training. In a separate comparison under a common training protocol that varies only the architecture, FLAME attains the lowest macro MAE among the compared architectures on all three data sets.

The scope of the method-comparison evidence is set by the protocol that produced it: three data sets, one unseen household each, and one fixed consistency weight. The recomposed window is label-preserving by construction, a property of that construction rather than a claim about the true joint distribution or a physically realized intervention.

Future work will extend the evaluation to further data sets and household protocols and examine how the fixed gate and margin shape performance across appliance conditions.

## REFERENCES

[1] International Energy Agency, “Energy efficiency 2025,” International Energy Agency, Paris, France, Tech. Rep., 2025. [Online]. Available: https://www.iea.org/reports/energy-efficiency-2025

[2] P. A. Schirmer and I. Mporas, “Non-intrusive load monitoring: A review,” IEEE Transactions on Smart Grid, vol. 14, no. 1, pp. 769–784, 2023.

[3] G. W. Hart, “Nonintrusive appliance load monitoring,” Proceedings of the IEEE, vol. 80, no. 12, pp. 1870–1891, 1992.

[4] J. Lin, J. Ma, J. Zhu, and H. Liang, “Deep domain adaptation for non-intrusive load monitoring based on a knowledge transfer learning network,” IEEE Transactions on Smart Grid, 2022.

[5] S. Dash and N. C. Sahoo, “A multi-task deep learning approach for nonintrusive load monitoring of multiple appliances,” IEEE Transactions on Smart Grid, vol. 15, no. 3, pp. 3337–3340, 2024.

[6] H. Rafiq, X. Shi, H. Zhang, H. Li, M. K. Ochani, and A. A. Shah, “Generalizability improvement of deep learning-based non-intrusive load monitoring system using data augmentation,” IEEE Transactions on Smart Grid, vol. 12, no. 4, pp. 3265–3277, 2021.

[7] Q. Luo, T. Yu, C. Lan, Y. Huang, Z. Wang, and Z. Pan, “A generalizable method for practical non-intrusive load monitoring via metric-based meta-learning,” IEEE Transactions on Smart Grid, 2024.

[8] J. Xiong, T. Hong, D. Zhao, and Y. Zhang, “MATNilm: Multi-appliancetask non-intrusive load monitoring with limited labeled data,” IEEE Transactions on Industrial Informatics, vol. 20, no. 3, pp. 3177–3187, 2024.

monitoring,” in Proceedings of the 5th International Workshop on Non-Intrusive Load Monitoring (NILM’20), 2020, pp. 89–93.

[9] C. Chen, G. Geng, H. Yu, Z. Liu, and Q. Jiang, “An end-cloud collaborated framework for transferable non-intrusive load monitoring,” IEEE Transactions on Cloud Computing, vol. 11, no. 2, pp. 1157–1169, 2023.

[30] K. Li, J. Feng, Y. Yao, Y. Xing, Q. Xiao, and J. Wang, “NILM model for multi-appliance power disaggregation based on the combination of common features and individual features,” Measurement, vol. 263, p. 120104, 2026.

[10] X. Li, S. Kong, J. Zeng, H. Dai, L. Zhang, W. Wang, Z. Zhang, and L. Xu, “CR-MAT: Causal representation learning for few-shot nonintrusive load monitoring,” Electronics, vol. 15, no. 6, p. 1195, 2026.

[11] C. Heinze-Deml and N. Meinshausen, “Conditional variance penalties and domain shift robustness,” Machine Learning, vol. 110, no. 2, pp. 303–348, 2021.

[12] K. Xiao, L. Engstrom, A. Ilyas, and A. Madry, “Noise or signal: The role of image backgrounds in object recognition,” in International Conference on Learning Representations (ICLR), 2021.

[13] C. Roder and K. Schweighofer, “Automated background swapping for robustness against spurious backgrounds,” arXiv preprint arXiv:2606.32018, 2026.

[14] V. Gupta, Z. Li, A. Kortylewski, C. Zhang, Y. Li, and A. Yuille, “SwapMix: Diagnosing and regularizing the over-reliance on visual context in visual question answering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5068–5078.

[15] A. Sauer and A. Geiger, “Counterfactual generative networks,” in International Conference on Learning Representations (ICLR), 2021.

[16] Y. Han, Y. Zhao, and Q. Zhao, “End-to-end load disaggregation via hard negative sample contrastive learning for multi-state appliances,” IEEE Transactions on Green Communications and Networking, vol. 10, pp. 3219–3231, 2026.

[17] K. Oublal, S. Ladjal, D. Benhaiem, E. Le-borgne, and F. Roueff, “DisCoV: Disentangling time series representations via contrastive based l-variational inference,” in Proceedings of UniReps: the First Workshop on Unifying Representations in Neural Models, ser. Proceedings of Machine Learning Research, vol. 243. PMLR, 2024, pp. 223–236. [Online]. Available: https://proceedings.mlr.press/v243/oublal24a.html

[18] J. Ma, Z. Zhao, X. Yi, J. Chen, L. Hong, and E. H. Chi, “Modeling task relationships in multi-task learning with multi-gate mixture-of-experts,” in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018, pp. 1930–1939.

[19] H. Tang, J. Liu, M. Zhao, and X. Gong, “Progressive layered extraction (PLE): A novel multi-task learning (MTL) model for personalized recommendations,” in Proceedings of the 14th ACM Conference on Recommender Systems, 2020, pp. 269–278.

[20] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv preprint arXiv:1803.01271, 2018.

[21] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132–7141.

[22] Z. Liang, C. Y. Chung, H. Yang, J. Liang, W. Zhang, H. Dong, and J. Zhu, “A heterogeneous multiple-experts approach to low-frequency nonintrusive load monitoring,” IEEE Transactions on Smart Grid, vol. 17, no. 1, pp. 746–765, 2026.

[23] K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in Proceedings ofthe 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014, pp. 1724–1734.

[24] M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Transactions on Signal Processing, vol. 45, no. 11, pp. 2673–2681, 1997.

[25] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[26] J. Z. Kolter and M. J. Johnson, “REDD: A public data set for energy disaggregation research,” in Workshop on Data Mining Applications in Sustainability (SustKDD), 2011.

[27] J. Kelly and W. Knottenbelt, “The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes,” Scientific Data, vol. 2, p. 150007, 2015.

[28] D. Murray, L. Stankovic, and V. Stankovic, “An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study,” Scientific Data, vol. 4, p. 160122, 2017.

[29] Z. Yue, C. Requena Witzig, D. Jorde, and H.-A. Jacobsen, “BERT4NILM: A bidirectional transformer model for non-intrusive load