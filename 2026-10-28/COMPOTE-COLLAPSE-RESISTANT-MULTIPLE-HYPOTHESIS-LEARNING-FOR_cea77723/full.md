# COMPOTE: COLLAPSE-RESISTANT MULTIPLE-HYPOTHESIS LEARNING FOR SEQUENTIAL RECOM-MENDATION

Louis Castres Saint Martin<sup>1,2</sup>, Clement Calauz´ enes\` <sup>1</sup>, Alain Rakotomamonjy<sup>1</sup>, Patrick Gallinari<sup>1,2</sup> <sup>1</sup>Criteo AI Lab, Paris, France

<sup>2</sup>Sorbonne Universite, CNRS, ISIR, 75005 Paris, France´

## ABSTRACT

Next-item prediction is inherently ambiguous: a single interaction history often admits multiple plausible continuations across distinct catalogue regions. Standard sequential recommenders compress these alternatives into a single prediction, conflating user intents. We introduce COllapse-resistant Multiple hyPOThesis lEarning (COMPOTE), a multi-hypothesis transformer trained with an annealed objective that transitions from uniform supervision to a winner-takes-all assignment. This mechanism forces hypotheses to specialize into distinct catalogue regions, while inference relies on an efficient, normalizer-free max-similarity rule. Theoretically, we show that in the population case, the optimal solutionspartition the conditional next-item distribution according to a max-entropy quantization principle, while pooled scores preserve the Bayes-optimal ranking. We also prove a generalization bound scaling logarithmically with the number of hypotheses. Across three Amazon Reviews 2023 benchmarks, COMPOTE outperforms strong ID-based, generative, and multi-interest baselines using item IDs alone. Matched-capacity ablations confirm that performance gains stem directly from the annealed objective, preventing the head collapse observed in conventional multihead architectures, rather than added capacity, ensembling, or routing.

## 1 INTRODUCTION

Recommender systems filter vast product catalogues where their ranking dictates item visibility, directly driving discovery, engagement, and revenue (Covington et al., 2016). As an example, Sequential recommendation predicts the next item from ordered interactions, closely mirroring real-world deployment where user preferences evolve and recent user’s interaction history is most informative (Rendle et al., 2010). There exists essentially two paradigms for sequential recommendation, both of which learn a single conditional model of the next item. The first family encodes the his tory with an autoregressive sequence model and scores items against a learned embedding table, moving from recurrent (Hidasi et al., 2016) and convolutional (Tang and Wang, 2018) encoders to causal self-attention (Kang and McAuley, 2018), backbone of later variants (Sun et al., 2019; Zhou et al., 2020; Zhai et al., 2024). The second family treats retrieval as generation (Tay et al., 2022): TIGER (Rajput et al., 2023) quantizes item content embeddings into semantic identifiers and decodes the next item’s identifier token by token, which helps on rare items at the cost of a content encoder, a learned tokenizer and beam search. Both families nonetheless learn a single conditional model of the next item: one query representation, one score vector, one ranked list.

Next-item prediction is ambiguous. An user’s interaction history rarely determines the next item: it points to a set of plausible continuations, often in unrelated parts of the catalogue, so the nextitem distribution is multimodal and a single query vector trained with cross-entropy settles between the modes instead of matching any one. Multi-interest learning addresses this by representing each user with several vectors, extracted by dynamic routing over capsules in MIND (Li et al., 2019), by multi-head attention with a diversity–accuracy aggregation step in ComiRec (Cen et al., 2020), or by activating a few concepts from a shared pool in SINE (Tan et al., 2021). REMI (Xie et al., 2023) keeps the same architecture and instead adds interest-aware hard negatives and a regularizer against routing collapse, where all interests attend to the same historical item. These methods already rely on a form of winner-takes-all (WTA) training, but it comes with the routing architecture and WTA is rarely examined as an objective in its own right. As a result, the optimal interests are not characterized, and diversity is usually encouraged through the architecture or additional penalties.

Our approach: COMPOTE. We frame next-item prediction as an inherently ambiguous learning problem and adopt multiple-choice learning (MCL) (Guzman-Rivera et al., 2012; Lee et al., 2016; Rupprecht et al., 2017), where a model outputs M candidate hypotheses, but optimizes only the best-performing head. While this winner-takes-all objective effectively quantizes the conditional next-item distribution (Letzelter et al., 2024), it is prone to sub-optimal local minima where heads that underperform early permanently deactivate. Annealed MCL (Perera et al., 2024) mitigates this initialization sensitivity by annealing a soft-assignment temperature during training. We adapt this formulation to discrete catalogues with COllapse-resistant Multiple hyPOThesis lEarning (COM-POTE; Section 2). A shared causal transformer feeds M prediction heads, each defining a distinct distribution over the catalogue, which are optimized via the objective function $\ell _ { \tau }$ in Equation (3) where τ is the temperature parameter. At high temperatures, all heads update uniformly; as the temperature vanishes, the objective smoothly converges to pure winner-takes-all. For inference and retrieval, we introduce max-similarity, a normalizer-free counterpart to the loss’s max-pooling operator (Proposition 5), which matches max-pooling accuracy on standard benchmarks while bypassing catalogue-wide softmax operations (Section 4.1).

Contributions. Methodologically, we propose COMPOTE, extending annealed multiple-choice learning to sequential recommendation alongside an efficient, normalizer-free retrieval rule (Section 2). On the theoretical front, we first establish a generalization bound scaling logarithmically in the number of hypotheses M (Theorem 4). We further characterize the optimal hypotheses in the winner-takes-all limit, showing they partition the conditional item distribution into maximumentropy regions (Theorem 7) and that pooled hypothesis scores preserve the Bayes ranking up to saturation ties (Proposition 6), proving head specialization emerges naturally without explicit diversity regularizers. Finally, across three Amazon Reviews 2023 benchmarks, COMPOTE consistently outperforms ID-based, generative, and multi-interest sequential baselines using item IDs alone, with capacity-matched ablations attributing these gains directly to the annealed MCL objective rather than model scaling, ensembling, or expert routing (Section 4).

## 2 PROBLEM STATEMENT AND ARCHITECTURE

We consider a supervised learning setting where a query $\mathbf { x } \in \mathcal { X }$ has an associated relevant item y in a catalog Y of N items, and the goal is to learn a model $g : \mathcal { X }  \mathfrak { S } _ { N }$ mapping a query to a ranking of the items, i.e., a permutation of $\{ 1 , \ldots , N \}$ . Data pairs $( \mathbf { x } , \mathbf { y } ) \sim \mathbb { P } _ { \mathbf { x } , \mathbf { y } }$ are drawn i.i.d. from an unknown underlying probability distribution over $\mathcal { X } \times \mathcal { V }$ . In recommendation, queries are user histories and items are the entities being recommended (products, movies, etc.). We define $\Delta ^ { N - 1 } = \{ \mathbf { p } \in [ 0 , 1 ] ^ { N } : \| \mathbf { p } \| _ { 1 } = 1 \}$ as the (N91)-dimensional probability simplex over Y.

Datasets. We instantiate this setting on three categories from the Amazon 2023 review corpus (Hou et al., 2024): MUSICAL INSTRUMENTS, INDUSTRIAL AND SCIENTIFIC, and VIDEO GAMES. A query x is a user’s chronologically ordered interaction history, and the target y is the next reviewed item. Each category contains approximately 25,000 items and between 50,000 and 95,000 users (Table 5). Because each user contributes few observations relative to the catalog size, several next items may be plausible given the same history.

Next-item Prediction with Multiple-Hypothesis. As motivated in the introduction, the model aims at performing next-item prediction via multiple hypotheses. It can be formalized as a function $f : \mathcal { X } \overset { \bullet } {  } ( \Delta ^ { N - 1 } ) ^ { \overline { { M } } }$ that returns, for every hypothesis $\dot { m } \leq M ,$ , a probability distribution $f _ { m } ( \mathbf { x } ) \in$ $\Delta ^ { N - 1 }$ over the catalog, where $\mathbf { p } _ { m , i } \triangleq f _ { m } ( \mathbf { x } ) _ { i }$ is the probability that hypothesis m assigns to item i being relevant to query x. The exact architecture of f used in COMPOTE is detailed in Figure 1. At inference, the M per-encoder distributions $\mathbf { p } = f ( { \dot { \mathbf { x } } } )$ are combined into a single score vector over the catalog, which is then sorted to output the ranking

$$
g = \operatorname { a r g s o r t } \circ \omega \circ f \quad \mathrm { w h e r e } \quad \omega ( \mathbf { p } ) = \left( \operatorname* { m a x } _ { m \leq M } \mathbf { p } _ { m , i } \right) _ { i = 1 } ^ { N } ( \operatorname* { m a x } \mathbf { - p } \mathrm { o o l i n g } ) .\tag{1}
$$

![](images/169e05d52dd6fcd99b646ab61a29dc4bcc06cb585f8689c2f85438947a183a53.jpg)  
Figure 1: Multiple-choice transformer. An item-id history x is embedded, summed with a learned positional embedding, and encoded by a stack of $L$ pre-norm causal transformer blocks shared by all heads; the representation z at the last (most recent) position is read off. Two modules consume z: the M hypothesis heads $h _ { 1 } , \ldots , h _ { M }$ , each producing $\mathbf { \dot { h } } _ { j } = h _ { j } ( \mathbf { z } ) \in \mathbb { R } ^ { d }$ , and the confidence score head $\hat { \gamma } .$ , whose j-th output is the predicted probability that head j is the best expert for x. Each hypothesis is turned into a distribution over the catalogue by $f _ { j , i } ( \mathbf { \bar { x } } ) = \mathrm { s o f t m a x } _ { i } \big ( \cos ( \mathbf { h } _ { j } , \mathbf { e } _ { i } ) / \beta \big )$ yielding one score vector $f _ { j , \cdot } ( \mathbf { x } ) \in \Delta ^ { N - 1 }$ per head. Everything left of z is a single shared trunk; right of it the M branches are independent, which is what makes head specialisation possible.

## 3 LEARNING PROBLEM

Ideally, to learn $f$ avoiding hypothesis collapse, we would like to use the winner-takes-all loss, defined for a given prediction $\mathbf { p } = f ( \mathbf { x } ) \in \left( \Delta ^ { N - 1 } \right) ^ { M }$ as

$\ell ^ { \mathrm { W I A } } ( \mathbf { y } , \mathbf { p } ) \triangleq \operatorname* { m i n } _ { m \leq M } - \log ( \mathbf { p } _ { m , \mathbf { y } } )$ with associated risk $\mathcal { L } ^ { \mathrm { w r A } } ( f , \mathbb { P } _ { \mathbf { x } , \mathbf { y } } ) = \mathbb { E } _ { \mathbf { x } , \mathbf { y } } [ \ell ^ { \mathrm { w T A } } ( \mathbf { y } , f ( \mathbf { x } ) ) ]$

(2)

Optimizing directly $\ell ^ { \mathrm { W T A } }$ is usually challenging because of the discontinuity introduced, which leads many people to resort to annealing methods. Following this path, we use the following loss function defined for temperature $\tau > 0$ (to be annealed towards 0) as

$$
\ell _ { \tau } ( \mathbf { y } , \mathbf { p } ) = - \tau \log \Big ( \sum _ { m \leq M } \mathbf { p } _ { m , \mathbf { y } } ^ { \frac { 1 } { \tau } } \Big ) \quad \mathrm { w i t h ~ a s s o c i a t e d ~ r i s k } \quad \mathcal { L } _ { \tau } ( f , \mathbb { P } _ { \mathbf { x } , \mathbf { y } } ) = \mathbb { E } _ { \mathbf { x } , \mathbf { y } } [ \ell _ { \tau } ( \mathbf { y } , f ( \mathbf { x } ) ) ]\tag{3}
$$

As the following result illustrates, annealing on $\ell _ { \tau }$ leads to optimizing $\ell ^ { \mathrm { W T A } }$

$$
\begin{array} { r } { \mathbf { P r o p o s i t i o n 1 . } \ F o r \ a n y \ ( \mathbf { y } , \mathbf { p } ) \in \mathcal { V } \times \left( \Delta ^ { N - 1 } \right) ^ { M } , \operatorname* { l i m } _ { \tau \to 0 ^ { + } } \ell _ { \tau } ( \mathbf { y } , \mathbf { p } ) = \ell ^ { \mathrm { w r a } } ( \mathbf { y } , \mathbf { p } ) . } \end{array}
$$

Proof can be found in Section $\mathrm { A } . 2 .$ . This allows to extend by continuity the definition of $\ell _ { \tau } \mathrm { t o } \tau = 0$ and consider $\tau \geq 0$ in the remainder for statements and proofs.

Parallel with Annealed Multiple-Choice Learning (MCL). Methods employing a winner-takesall strategy exist in other domains, such as audio source separation, under the framework of multiplechoice learning (MCL). In fact, our loss $\ell _ { \tau }$ recovers a formulation from this domain, namely annealed ${ \mathrm { { M C L } } } ,$ yet existing literature does not address how to recombine these hypotheses for inference. Our primary contribution is demonstrating, both theoretically and empirically, that $\omega - \mathrm { E q u a } -$ tion $( 1 ) - \mathrm { i s }$ the principled choice for hypothesis aggregation in recommender systems. Formally, the annealed MCL (Perera et al., 2024) objective function is defined as

$$
\ell _ { \tau } ^ { \mathrm { a M C L } } ( \mathbf { y } , \mathbf { p } ) = - \sum _ { m \leq M } \gamma _ { m } ( \mathbf { y } , \mathbf { p } ) \log ( \mathbf { p } _ { m , \mathbf { y } } ) \quad { \mathrm { w i t h } } \quad \gamma ( \mathbf { y } , \mathbf { p } ) = { \mathrm { s o f t m a x } } \left( { \frac { 1 } { \tau } } \log ( \mathbf { p } . \mathbf { y } ) \right) .\tag{4}
$$

Corollary 3 (Recovering the gradient of aMCL). For any $( \mathbf { y } , \mathbf { p } ^ { \theta } ) \in \mathcal { V } \times \left( \Delta ^ { N - 1 } \right) ^ { M }$ (θ is some   
parameter), for any $\tau \geq 0 ,$ , and $\alpha \in \Delta ^ { M - 1 }$   
∇<sub>θ</sub>ℓ<sub>τ</sub> (y, p<sup>θ</sup>) = ∇<sub>θ</sub>ψ<sub>τ</sub>  y, p<sup>θ</sup>, α<sup></sup><sub>α=γ (y,pθ)</sub> with ψ<sub>τ</sub> (y, p<sup>θ</sup>) = − X α<sub>m</sub> log(p<sup>θ</sup><sub>m,y</sub>). (6)   
m≤M

and the following propostion establishes the link between the two losses.

Proposition 2 (Recovering aMCL). For any $( \mathbf { y } , \mathbf { p } ) \in \mathcal { V } \times \left( \Delta ^ { N - 1 } \right) ^ { M }$ and for any $\tau \geq 0 ,$

$$
\ell _ { \tau } ( \mathbf { y } , \mathbf { p } ) = \ell _ { \tau } ^ { \mathrm { a M C L } } ( \mathbf { y } , \mathbf { p } ) - \tau H ( \gamma ( \mathbf { y } , \mathbf { p } ) )\tag{5}
$$

where $\begin{array} { r } { H ( \gamma ) = - \sum _ { m \leq M } \gamma _ { m } \log \gamma _ { m } } \end{array}$ is the Shannon entropy of the softmax weights $\gamma ( \mathbf { y } , \mathbf { p } )$

Proof can be found in Section A.3. An important detail strengthens the link: In Perera et al. (2024), during the optimization, the gradient is only backpropagated through $\log \mathbf { p } _ { m , \mathbf { y } }$ and not through $\gamma _ { m } ( \mathbf { y } , \mathbf { p } ) - i . e .$ . there is a stop-gradient operator on the $\gamma _ { j } ( \mathbf { y } , \mathbf { p } )$ term. The following result shows the two learning paradigm coincide exactly in practice. Proof can be found in Section A.4.

Learning Guarantees. We have discussed the loss function we use to train the model and have established a connection with the annealed MCL loss. We now turn to the theoretical analysis of the learning guarantees associated with this loss function. While in practice, we learn $f$ by minimizing the empirical risk $\widehat { \mathcal { L } } _ { \tau } ( f ) = \mathcal { L } _ { \tau } ( f , \hat { \mathbb { P } } _ { S } )$ over a training dataset. during training, τ goes to 0. Thus the learning guarantees need to be provided for the limiting loss $\ell ^ { \mathbf { \overline { { w } } _ { \mathrm { { T A } } } } }$ Over a training dataset $S = \{ ( { \bf x } _ { i } , { \bf y } _ { i } ) \} _ { i = 1 } ^ { n } \sim \mathbb { P } _ { { \bf x } , { \bf y } } ^ { n } ,$ , the empirical risk is denoted $\widehat { \mathcal { L } } _ { S } ^ { \mathrm { w r A } } ( f ) = \mathcal { L } ^ { \mathrm { w r A } } ( f , \hat { \mathbb { P } } _ { S } )$ where $\hat { \mathbb P } _ { S }$ is the empirical distribution of S. In order to get insights on the impact of the number of hypotheses M on the learning problem, we are here interested in the generalization gap between the empirical WTA loss and the expected WTA loss $\mathcal { L } ^ { \mathrm { w r a } } ( f , \mathbb { P } _ { S } )$ and provide a bound on this gap in terms of the Rademacher complexity of the hypothesis class $\mathcal { F }$ and the number of hypotheses M. Proof is given in Section D.2.

Theorem 4 (refined log-loss generalization bound). Let $\mathcal { F }$ be a hypothesis class ofprobabilitygenerating heads $f _ { m } : \overline { { \mathcal { X } } } \overset { } {  } \bar { \Delta } ^ { N - 1 }$ . Assume there exists a constant $\epsilon > 0$ such that $f _ { m , i } ( \mathbf { x } ) \geq \epsilon$ for all $m \ { \overset { \cdot } { \in } } \ [ M ] , { \overset { \cdot } { \mathbf { x } } } \in { \mathcal { X } } ,$ , and $i \in [ N ]$ . Then, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ over the sample $\begin{array} { r } { \left. S \sim P _ { \mathbf { x } , \mathbf { y } } ^ { n } , \right. } \end{array}$ , every ensemble $f \doteq ( f _ { 1 } , \cdot \cdot \cdot , f _ { M } ) \in \mathcal { F } ^ { M }$ satisfies:

$$
\mathcal { L } ^ { \mathrm { W T A } } ( f , \mathbb { P } _ { S } ) \leq \widehat { \mathcal { L } } _ { S } ^ { \mathrm { w T A } } ( f ) + \frac { 2 } { \epsilon } \operatorname* { m a x } _ { m \in [ M ] } \mathcal { R } _ { n } ( \mathcal { F } _ { m , \mathbf { y } } ) + 2 \log \left( \frac { 1 } { \epsilon } \right) \sqrt { \frac { 2 \ln M } { n } } + \log \left( \frac { 1 } { \epsilon } \right) \sqrt { \frac { \ln ( 2 / \delta ) } { 2 n } } .\tag{7}
$$

where $\mathcal { R } _ { n } ( \mathcal { F } _ { m , \mathbf { y } } )$ is the expected Rademacher complexity of ${ } ^ { \mathsf { c } } { \mathcal { F } } _ { m , \mathbf { y } } = \{ \mathbf { x } \mapsto f _ { m , \mathbf { y } } ( \mathbf { x } ) : f _ { m } \in { \mathcal { F } } \}$

## Insights from the bounds:

• The generalization gap is controlled by the Rademacher complexities of the individual heads, scaled by $1 \bar { / } \epsilon . ~ \mathrm { A s }$ the minimum predicted probability ϵ decreases, the bounds become looser, reflecting increased uncertainty in predictions. Ensuring a lower bound on predicted probabilities, via label smoothing, temperature scaling, or KL regularisation, therefore yields tighter guarantees.

• Theorem 4 shows that the number of hypotheses M impacts the bound only through a $\mathcal { O } ( \sqrt { \ln M / n } )$ term, on top of the complexity of the best head. Adding more hypotheses therefore does not linearly increase the risk of overfitting, provided that the individual head complexities are controlled. Hence, one can benefit from the diversity of multiple heads without a proportional increase in generalization error.

We provide an empirical validation of this theoretical insight in Section I in which we also discuss the practical implications of the saturation phenomenon described in Proposition 6 in Section 4.

Learning Stabilization. We observe that training $f$ can be stabilized through the joint learning of the confidence score head via a binary cross-entropy loss over $\hat { \gamma }$ (see Figure 1). This finding aligns with the fact that the min operator is non-smooth and can be challenging to optimize. While beneficial, this effect is not the primary driver of the overall performance gains (see ablation study in Section G). Consequently, the formalization and technical details of this stabilization term are deferred to the appendix (see Section H).

## 4 COMPOTE LEADS TO OPTIMAL RANKING

This section explains why the max-pooling aggregation ω defined in Equation (1) is the natural choice when learning f with $\ell _ { \tau }$ . In fact, the max-pooling aggregation ω can be seen as the limit for a temperature $\tau  0$ of some aggregation $\omega _ { \tau }$ (defined below), which cross-entropy is exactly $\ell _ { \tau }$

$$
\begin{array} { r l r } {  { ( \mathbf { P r o p o s i t i o n } \ \mathsf { S } ( \ell _ { \tau } \ \mathrm { v i a } \ \omega ^ { \tau } ) , \ F o r a n y \ \mathbf { ( y , p ) } \in \mathcal { V } \times ( \Delta ^ { N - 1 } ) ^ { M } a n d f o r a n y \ \tau \ge 0 ,  } } \\ & { } & \\ & {  \ell _ { \tau } ( \mathbf { y , p } ) = - \log \omega _ { \mathbf { y } } ^ { \tau } ( \mathbf { p } ) \quad w h e r e \quad \omega ^ { \tau } ( \mathbf { p } ) = ( \Big ( \sum _ { m \leq M } \mathbf { p } _ { m , i } ^ { 1 / \tau } \Big ) ^ { \tau } ) _ { i = 1 } ^ { N } \quad a n d \quad \omega ^ { \tau } \ \xrightarrow [ \tau  0 ] { \tau } \omega } \omega  \end{array}\tag{8}
$$

Proof can be found in Section B.1. Notice that $\omega ^ { \tau } ( \mathbf { p } )$ does not necessarily lie in $\Delta ^ { N - 1 }$ , thus $\mathscr { L } _ { \tau } ( f ) =$ $- \mathbb { E } _ { \mathbf { x } , \mathbf { y } } \left[ \log \omega _ { \mathbf { y } } ^ { \tau } ( f ( \mathbf { x } ) ) \right]$ can be lower than the Bayes risk $\begin{array} { r l } { \operatorname* { i n f } _ { q : \mathcal { X } \to \Delta ^ { N - 1 } } - \mathbb { E } _ { \mathbf { x } , \mathbf { y } } [ \log q _ { \mathbf { y } } ( \mathbf { x } ) ] } & { { } } \end{array}$ . Because $\omega ^ { \tau } ( \mathbf { p } )$ and $\omega ( \mathbf { p } )$ do not necessarily belong to the simplex, they do not recover the Bayes predictor (they are not a probability). However, for prediction $\mathbf { p } ^ { \star }$ that minimizes $\ell ^ { \mathrm { W T A } }$ in expectation, $\omega ( \mathbf { p } ^ { \star } )$ ranks the items in the same order as the Bayes predictor, as formalized by the following statement.

$$
\begin{array} { r l r } & { \mathbf { P r o p o s i t i o n ~ 6 ~ ( \omega \mathrm { ~ i n d u c e s ~ B a y e s ~ r a n k i n g ) . ~ } } L e t \textbf { q } ^ { \star } \in \Delta ^ { N - 1 } , a n d d e n o t e \ t h e \ o p t i m a l \ p r e d i c t i o n } & \\ & { \mathbf { p } ^ { \star } \in \mathrm { a r g m i n } _ { \mathbf { p } \in ( \Delta ^ { N - 1 } ) ^ { M } } \mathbb { E } _ { \mathbf { y } \sim \mathbf { q } ^ { \star } } [ \ell ^ { \mathrm { W I a } } ( \mathbf { y } , \mathbf { p } ) ] . \ T h e n } & \\ & { \qquad \forall a , b \in \mathcal { Y } , \qquad \mathbf { q } _ { a } ^ { \star } > \mathbf { q } _ { b } ^ { \star } \implies \omega _ { a } ( \mathbf { p } ^ { \star } ) > \omega _ { b } ( \mathbf { p } ^ { \star } ) \quad o r \quad \omega _ { a } ( \mathbf { p } ^ { \star } ) = \omega _ { b } ( \mathbf { p } ^ { \star } ) = 1 . \qquad ( 9 ) } & \end{array}
$$

Proof can be found in Section B.2. The ranking prediction associated to COMPOTE thus preserves the order of the Bayes predictor strictly, except in the single saturation case $\omega _ { a } ( \mathbf { p } ^ { \star } ) = \omega _ { b } ( \mathbf { p } ^ { \star } ) = 1 ;$ understanding this failure case is informative for the choice of M. Intuitively, it happens when (given some x) at least two items a and b have large probabilities $\mathbf { q } _ { a } ^ { \star } , \mathbf { q } _ { b } ^ { \star } \geq 1 / M \colon$ : for this $\mathbf { x } ,$ one hypothesis m will be such that $\mathbf { p } _ { m , i } ^ { \star } = \mathbb { 1 } _ { [ i = a ] }$ (see Theorem 7 for why this happens), another hypothesis m<sup>′</sup> will be such that $\mathbf { p } _ { m ^ { \prime } , i } ^ { \star } = \mathbb { 1 } _ { [ i = b ] }$ . Hence, even though $\mathbf { q } _ { a } ^ { \star } > \mathbf { q } _ { b } ^ { \star }$ , then $\phi _ { a } = \phi _ { b } = 1$ . This advocates for using values for M that are not too large, which is validated experimentally in Figure 5.

The performance of using the aggregation ω over the learning done via annealing of $\ell _ { \tau }$ is also validated empirically in the recommendation setting on several Amazon 2023 datasets.

Preprocessing and protocol. We evaluate on the three Amazon 2023 subsets introduced in Section 2. Following prior work (Rajput et al., 2023; Zhou et al., 2020; Zheng et al., 2025), we apply the benchmark 5-core filtering (Amazon Reviews’23 Benchmark, 2023), which iteratively discards users and items with fewer than five interactions, and truncate histories to 50 items at both training and test time. We adopt the leave-one-out split of Kang and McAuley (2018): the last interaction of each user is the test target, the penultimate one is used for validation and the remainder for training. We report Hit Rate (H@K) and NDCG (N@K) for $K \in \{ 5 , 1 0 \}$ , computed by ranking the entire catalog rather than sampled negatives; since each test instance has a single relevant item, H@K coincides with Recall@K.

Baselines. We compare against seven ID-based sequential recommenders (Caser (Tang and Wang, 2018), HGN (Ma et al., 2019), GRU4Rec (Hidasi et al., 2016), BERT4Rec (Sun et al., 2019), SAS-Rec (Kang and McAuley, 2018), HSTU (Zhai et al., 2024) and S3-Rec (Zhou et al., 2020)) and against the generative recommender TIGER (Rajput et al., 2023), whose numbers we take from Zheng et al. (2025). Because COMPOTE predicts several hypotheses per query, we also compare against six multi-interest recommenders that represent a user by several vectors: MIND (Li et al., 2019), ComiRec-DR and ComiRec-SA (Cen et al., 2020), SINE (Tan et al., 2021), Re4 (Zhang et al.,

Table 1: Performance (%) on three Amazon 2023 subsets (H@K: Hit Rate@K, N@K: NDCG@K). First-block numbers are from Zheng et al. (2025); multi-interest models are trained by us under the same protocol (Section F). COMPOTE uses max-similarity aggregation (Equation (10)), equivalent to max-softmax pooling (Section 4.1). Bold: best, underline: second best.
<table><tr><td></td><td colspan="4">Instrument</td><td colspan="4">Scientific</td><td colspan="4">Game</td></tr><tr><td>Method</td><td>H@5</td><td>H@10</td><td>N@5</td><td>N@10</td><td>H@5</td><td>H@10</td><td>N@5</td><td>N@10</td><td>H@5</td><td>H@10</td><td>N@5</td><td>N@10</td></tr><tr><td>Caser (Tang and Wang, 2018)</td><td>2.41</td><td>3.86</td><td>1.51</td><td>1.97</td><td>1.59</td><td>2.57</td><td>1.01</td><td>1.32</td><td>3.30</td><td>5.53</td><td>2.09</td><td>2.81</td></tr><tr><td>HGN (Ma et al., 2019)</td><td>3.21</td><td>5.17</td><td>2.02</td><td>2.65</td><td>2.12</td><td>3.51</td><td>1.31</td><td>1.76</td><td>4.24</td><td>6.87</td><td>2.71</td><td>3.56</td></tr><tr><td>GRU4Rec (Hidasi et al., 2016)</td><td>3.24</td><td>5.01</td><td>2.09</td><td>2.66</td><td>2.02</td><td>3.38</td><td>1.29</td><td>1.73</td><td>4.99</td><td>7.99</td><td>3.20</td><td>4.16</td></tr><tr><td>BERT4Rec (Sun et al., 2019)</td><td>3.07</td><td>4.85</td><td>1.95</td><td>2.52</td><td>1.86</td><td>2.96</td><td>1.19</td><td>1.55</td><td>4.60</td><td>7.35</td><td>2.98</td><td>3.86</td></tr><tr><td>SASRec (Kang and McAuley, 2018)</td><td>3.33</td><td>5.23</td><td>2.13</td><td>2.74</td><td>2.59</td><td>4.12</td><td>1.50</td><td>1.99</td><td>5.35</td><td>8.47</td><td>3.31</td><td>4.38</td></tr><tr><td>HSTU (Zhai et al., 2024)</td><td>3.43</td><td>5.77</td><td>1.91</td><td>2.71</td><td>2.71</td><td>4.29</td><td>1.47</td><td>1.98</td><td>5.78</td><td>9.03</td><td>3.34</td><td>4.42</td></tr><tr><td>S3-Rec (Zhou et al., 2020)</td><td>3.17</td><td>4.96</td><td>1.99</td><td>2.57</td><td>2.63</td><td>4.18</td><td>1.71</td><td>2.19</td><td>4.85</td><td>7.69</td><td>3.15</td><td>4.06</td></tr><tr><td>TIGER (Rajput et al., 2023)</td><td>3.70</td><td>5.64</td><td>2.44</td><td>3.06</td><td>2.64</td><td>4.22</td><td>1.75</td><td>2.26</td><td>5.59</td><td>8.68</td><td>3.66</td><td>4.67</td></tr><tr><td>MIND (Li et al., 2019)</td><td>2.71</td><td>4.42</td><td>1.65</td><td>2.20</td><td>2.11</td><td>3.29</td><td>1.27</td><td>1.65</td><td>3.22</td><td>5.20</td><td>1.99</td><td>2.65</td></tr><tr><td>ComiRec-DR (Cen et al., 2020)</td><td>2.67</td><td>4.37</td><td>1.64</td><td>2.19</td><td>2.14</td><td>3.41</td><td>1.30</td><td>1.72</td><td>3.21</td><td>5.23</td><td>1.96</td><td>2.60</td></tr><tr><td>ComiRec-SA (Cen et al., 2020)</td><td>2.64</td><td>4.31</td><td>1.62</td><td>2.15</td><td>2.08</td><td>3.31</td><td>1.26</td><td>1.65</td><td>3.20</td><td>5.19</td><td>1.95</td><td>2.61</td></tr><tr><td>SINE (Tan et al., 2021)</td><td>3.08</td><td>5.20</td><td>1.52</td><td>2.24</td><td>2.36</td><td>3.83</td><td>1.21</td><td>1.68</td><td>5.26</td><td>8.80</td><td>2.77</td><td>3.87</td></tr><tr><td>Re4 (Żhang et al., 2022)</td><td>2.13</td><td>3.74</td><td>1.23</td><td>1.66</td><td>1.61</td><td>2.66</td><td>0.95</td><td>1.27</td><td>3.15</td><td>5.61</td><td>1.98</td><td>2.68</td></tr><tr><td>REMI (Xie et al., 2023)</td><td>2.73</td><td>4.34</td><td>1.71</td><td>2.22</td><td>2.20</td><td>3.86</td><td>1.21</td><td>1.70</td><td>4.14</td><td>7.42</td><td>2.29</td><td>3.30</td></tr><tr><td>COMPOTE (ours)</td><td>3.75</td><td>5.83</td><td>2.45</td><td>3.10</td><td>2.84</td><td>4.20</td><td>1.92</td><td>2.35</td><td>6.06</td><td>9.26</td><td>4.01</td><td>5.04</td></tr></table>

2022) and REMI (Xie et al., 2023), which we train ourselves under the protocol above. Section F describes each method.

In Table 1, COMPOTE is best in 11 of the 12 columns, the exception being H@10 on Scientific where HSTU (4.29) and TIGER (4.22) edge it out (4.20). Against TIGER, the strongest baseline on NDCG, COMPOTE improves N@5 by 9.7% on Scientific and 9.6% on Game, and ties on Instrument (2.44 → 2.45). Note that TIGER performs autoregressive generative retrieval over semantic identifiers derived from item text and decodes them with beam search at inference, while COM-POTE sees item identifiers only and ranks the catalogue in a single forward pass: modelling the multimodality of the next-item distribution suffices to beat a content-aware generative recommender.

Multi-interest models trade precision for coverage. Multi-interest recommenders are the closest analogue of COMPOTE, yet none reaches the second-best value of any column, and only SINE on Game H@10 (8.80) exceeds SASRec. SINE is within 5% of COMPOTE on that column but 31% below on N@5 (2.77 against 4.01): maximising over interest vectors widens the candidate pool without placing the target near the top, whereas COMPOTE improves both.

## 4.1 FAST RETRIEVAL FOR COMPOTE

For large-scale recommender systems, it is desirable to have a model that is compatible with vector databases to enable a retrieval scaling sublinearly with N. As $\begin{array} { r l } { f _ { m , i } ( \mathbf { x } ) } & { { } = } \end{array}$ softmax $( \cos ( h _ { m } ( x ) , \mathbf { e } _ { i } ) / \beta )$ (see Figure 1), there is already a dependency linear in N with the computation of the softmax. This section shows that, for the purpose of ranking the items for recommendation, the softmax can be dropped at inference – it is possible to apply ω directly on the cosine similarities to obtain scores to rank the items. Formally, it means replace at inference g = argsort ◦ ω ◦ f by

$$
\tilde { g } = \mathrm { a r g s o r t } \circ \omega \circ s \quad \mathrm { w h e r e } \quad s _ { m , i } ( \mathbf { x } ) = \mathrm { c o s } ( h _ { m } ( \mathbf { x } ) , \mathbf { e } _ { i } ) / \beta )\tag{10}
$$

Due to the following identity: log max<sub>m≤M</sub> $\begin{array} { r } { ( \mathrm { s o f t m a x } _ { i } ( \mathbf { s } _ { m , \cdot } ) ) = \operatorname* { m a x } _ { m \leq M } \left( \mathbf { s } _ { m , i } - \log Z _ { m } \right) } \end{array}$ where $\begin{array} { r } { Z _ { m } = \sum _ { i = 1 } ^ { N } \exp ( \mathbf { s } _ { m , i } ) } \end{array}$ , g and $\tilde { g }$ rank items identically if $Z _ { m }$ is constant across the hypotheses. This turns out to happen in practice as the standard deviation of $Z _ { m }$ across hypotheses is largely lower that its value (Figure 3). Experiments confirm this as using $\tilde { g }$ rather than g comes at no loss in ranking accuracy (Table 2).

This equivalence is a key deployment result: the model requires neither a catalog-wide softmax nor its normalizer, making the model directly compatible with standard vector databases. For this reason, we report most of the results with $\tilde { g }$ (ex: Table 3)

Table 2: Effect of the inference aggregator on COMPOTE performance (%) on Game, for the same trained model with M = 3 heads and the same checkpoint. “Normalizer” indicates whether ranking requires the per-head catalogue partition function $Z _ { m }$ . Bold: best.
<table><tr><td>Aggregator</td><td>Normalizer</td><td>H@5</td><td>H@10</td><td>N@5</td><td>N@10</td></tr><tr><td>g – &quot;max softmax&quot;</td><td>required</td><td>6.04</td><td>9.27</td><td>4.01</td><td>5.05</td></tr><tr><td>ğ – &quot;max cosine similarity&quot;</td><td>not required</td><td>6.06</td><td>9.26</td><td>4.01</td><td>5.04</td></tr></table>

Table 3: Ablation over the prediction head and the training objective (%). Variants share the same encoder backbone and parameter budget, and rank items with max-similarity aggregation (Equation (10)): MCT-SASRec is a single-hypothesis iso-parameter head, MCT-EnsCE uses an ensemble cross-entropy objective, and MCT-Router routes queries to a single expert. Bold: best.
<table><tr><td></td><td colspan="4">Instrument</td><td colspan="4">Scientific</td><td colspan="4">Game</td></tr><tr><td>Method</td><td>H@5</td><td>H@10</td><td>N@5</td><td>N@10</td><td>H@5</td><td>H@10</td><td>N@5</td><td>N@10</td><td>H@5</td><td>H@10</td><td>N@5</td><td>N@10</td></tr><tr><td>MCT-SASRec</td><td>3.63</td><td>5.70</td><td>2.34</td><td>2.97</td><td>2.55</td><td>3.87</td><td>1.71</td><td>2.13</td><td>5.73</td><td>8.84</td><td>3.78</td><td>4.78</td></tr><tr><td>MCT-EnsCE</td><td>3.54</td><td>5.53</td><td>2.28</td><td>2.96</td><td>2.42</td><td>3.80</td><td>1.60</td><td>2.03</td><td>5.74</td><td>8.81</td><td>3.78</td><td>4.77</td></tr><tr><td>MCT-Router</td><td>3.57</td><td>5.48</td><td>2.28</td><td>2.89</td><td>2.40</td><td>3.60</td><td>1.64</td><td>2.00</td><td>5.74</td><td>8.81</td><td>3.78</td><td>4.77</td></tr><tr><td>COMPOTE (ours)</td><td>3.75</td><td>5.83</td><td>2.45</td><td>3.10</td><td>2.84</td><td>4.20</td><td>1.92</td><td>2.35</td><td>6.06</td><td>9.26</td><td>4.01</td><td>5.04</td></tr></table>

## 4.2 ABLATIONS: WHERE DO THE GAINS COME FROM?

COMPOTE could outperform a single-head model for three distinct reasons, additional capacity, naive ensembling, or mixture-of-experts (MoE) routing. We isolate each with a matched-capacity variant (Table 3). COMPOTE outperforms all three on every dataset and metric.

Not a bigger model: MCT-SASRec spends the parameter budget of COMPOTE on a single hypoth esis with a deeper prediction head. It is the strongest of the three variants and still trails COMPOTE everywhere (N@5 of 2.34 against 2.45 on Instrument, 1.71 against 1.92 on Scientific, 3.78 against 4.01 on Game). At equal capacity, the multi-hypothesis parameterization beats a deeper singlehypothesis head.

Not an ensemble: MCT-EnsCE keeps the architecture of COMPOTE but trains every head on every example with equal weight, without confidence supervision or annealing. It minimizes $\begin{array} { r } { \ell ^ { \mathrm { e n s } } ( { \bf y } , f ( { \bf x } ) ) = - \frac { 1 } { M } \sum _ { m < M } \mathrm { l o g } f _ { m } ( { \bf x } ) _ { \bf y } } \end{array}$ , the τ → ∞ limit of the soft assignment in Equation (4). It falls below COMPOTE on every dataset and metric, so the gains are not explained by architectural multiplicity or naive ensembling alone.

Not a mixture of experts: MCT-Router follows the routing recipe of multi-interest recommendation (Section 1): it keeps the shared encoder and replaces the training of COMPOTE by a learned gate that selects one expert per query. It trails COMPOTE everywhere (N@5 of 2.28 against 2.45 on Instrument, 1.64 against 1.92 on Scientific, 3.78 against 4.01 on Game). The next section traces this gap to head collapse: without the annealed winner-takes-all objective, routing fails to induce distinct hypotheses (Section 5.1).

## 5 HYPOTHESIS SPECIALIZATION

We first show that, unlike the matched-capacity variants of Section 4.2, COMPOTE learns hypotheses that do not collapse and carry complementary rankings. We then explain this phenomenon by characterizing what each hypothesis $\mathbf { p } _ { m } ^ { \star }$ learns at the winner-takes-all limit τ → 0.

## 5.1 COLLAPSE AND ORACLE EVIDENCE

We quantify collapse through the mean pairwise cosine similarity of the head outputs. For a query x, let $\bar { \mathbf { h } } _ { m } ( \mathbf { x } ) = \mathbf { h } _ { m } ( \mathbf { x } ) / \| \mathbf { h } _ { m } ( \mathbf { x } ) \| _ { 2 }$ and define $\begin{array} { r } { S ( \mathbf { x } ) = \frac { 1 } { M ( M - 1 ) } \sum _ { j , k = 1 } ^ { M } \bar { \mathbf { h } } _ { j } ( \mathbf { x } ) ^ { \top } \bar { \mathbf { h } } _ { k } ( \mathbf { x } ) } \end{array}$ . We report its test-set average; $S = 1$ means that all heads have the same direction. MCT-EnsCE and MCT-Router reach $S = 1 . 0 0$ on all three datasets, whereas COMPOTE obtains 0.64, 0.61 and 0.58 on INSTRU-

MENT, SCIENTIFIC and GAME, respectively (Table 6). Thus, neither architectural multiplicity nor learned routing is sufficient to prevent collapse.

Oracle head selection. To test whether COMPOTE’s distinct heads contain complementary rankings, we compare the max-similarity rule of Equation (10) with an oracle that observes the target and selects the head assigning it the highest similarity,

$$
m ^ { \mathrm { o r c } } ( { \bf x } , { \bf y } ) = \underset { m \leq M } { \mathrm { a r g m a x } } s _ { m , { \bf y } } ( { \bf x } ) ,\tag{11}
$$

then ranks every item using $s _ { m ^ { \mathrm { o r c } } ( \mathbf { x } , \mathbf { y } ) , i } ( \mathbf { x } )$ . The oracle is not deployable, but it measures whether one hypothesis ranks the target substantially better than the aggregated list. For MCT-EnsCE and MCT-Router, aggregated and oracle N@5 coincide exactly on every dataset: selecting among collapsed heads cannot help (Table 7). For COMPOTE, oracle selection raises N@5 by 53% on IN-STRUMENT $( 2 . 4 5  3 . 7 5 )$ , 83% on SCIENTIFIC $( 1 . 9 2  3 . 5 2 )$ and 94% on GAME $( 4 . 0 1  7 . 7 8 )$ The ranking gains therefore coincide with a qualitative change in the learned predictor: COMPOTE produces distinct heads whose rankings contain complementary signal.

These findings raise a structural question: why does COMPOTE’s training create complementary hypotheses, whereas ensemble and routed objectives collapse?

## 5.2 WHAT EACH HYPOTHESIS LEARNS

For a fixed query, Theorem 7 describes the optimal hypotheses through three properties:

(P1) Partition. The catalog splits into disjoint cells $y _ { 1 } , \ldots , y _ { M }$ , each hypothesis supported on its own cell.

(P2) Faithfulness within cells. On $\mathcal { \boldsymbol { \mathrm { y } } } _ { m } ,  { \mathbf { p } } _ { m } ^ { \star }$ is q<sup>⋆</sup> renormalized by $Q _ { m } \colon$ each hypothesis ranks the items of its cell exactly as the Bayes predictor does, and therefore still depends on the query through $\mathbf { q } ^ { \star }$

(P3) Balance. The partition maximizes the entropy of $( Q _ { m } ) _ { m = 1 } ^ { M }$ , so the cells split the mass of $\mathbf { q } ^ { \star }$ as evenly as the catalogue allows.

```latex
Theorem 7 (Extension of Th. 1 – Rupprecht et al. (2017)). Let $\mathbf { q } ^ { \star } \in \Delta ^ { N - 1 }$ , and denote the
optimal prediction $\mathbf { p } ^ { \star } \in$ argmin $\mathsf { \mathsf { l } } _ { \mathbf { p } \in ( \Delta ^ { N - 1 } ) ^ { M } } \mathbb { E } _ { \mathbf { y } \sim \mathbf { q } ^ { \star } } [ \ell ^ { \mathrm { W T A } } ( \mathbf { y } , \mathbf { p } ) ]$ . Then for every $m \le M$ and
$y \in \mathcal { V }$
p<sub>m,y</sub> Q<sub>m</sub> <sup>q⋆y</sup> 1<sub>[y∈Y ]</sub> , with Q<sub>m</sub> = X q<sup>⋆</sup><sub>y</sub> . (12)
y∈Y<sub>m</sub>
where the tessalation is $\mathcal { Y } _ { m } = \big \{ y \in \mathcal { Y } : \forall m ^ { \prime } \neq m , - \log ( \mathbf { p } _ { m , y } ^ { \star } ) < - \log ( \mathbf { p } _ { m ^ { \prime } , y } ^ { \star } ) \big \}$ . Moreover
the tessalation $( { \mathcal { V } } _ { m } ) _ { m = 1 } ^ { M }$ is entropy-maximizing.
```

Proof can be found in Section K.1.

COMPOTE is trained at $\tau > 0$ with softmax heads, so the exact zeros of Equation (12) cannot be reached. We test each property through its observable signature. Mirroring the tessellation of Theorem 7, the empirical cell of head m on query x is $\mathcal { Y } _ { m } ( \bar { \mathbf { x } } ) = \{ y \in \mathcal { Y } : m = \mathrm { a r g m a x } _ { m ^ { \prime } } f _ { m ^ { \prime } , y } ( \mathbf { x } ) \}$ and the winning head of a test pair $\displaystyle ( \mathbf { x } , \mathbf { y } )$ is the head whose cell contains y. All diagnostics are computed on GAME, with $M = 3$ unless stated otherwise, and Table 4 reports them for $M \in \{ 3 , 4 , 5 \}$ The protocol is detailed in Section L, which also reports the same partition on SCIENTIFIC.

(P1) The hypotheses partition the catalogue. The non-collapsed head directions of Table 6 are a first signature of (P1). The partition holds for the full distributions and not only for the top of the rankings: each head keeps between 91.5% and 93.7% of its softmax mass inside its own cell, for every M. The retrieved sets are therefore nearly disjoint. The top-10 lists of two heads on the same query have a mean Jaccard overlap of 0.008, and 91.8% of the catalogue items retrieved at test time are reached by a single head only. Both sharpen with M, reaching 0.001 and 98.6% at $M = 5$ Figure 2 shows the resulting partition: projected onto the plane of the head directions, the catalogue splits into three sectors, one per hypothesis.

Table 4: Signatures of Theorem 7 on GAME as the number of hypotheses M varies. All columns are computed on the full test set of 94,762 query–target pairs. Win frequencies and sector shares are reported as the range over heads.
<table><tr><td>M</td><td>In-cell mass</td><td>Win frequency (%)</td><td>Sector share (%)</td><td> $\hat { \mathcal { L } } ^ { \mathrm { w r a } }$ </td><td> $\hat { \mathcal { L } } ^ { \mathrm { w r a } }$  + log M</td><td>Sector agreement</td></tr><tr><td>3</td><td>0.929</td><td>[32.0, 34.4]</td><td>[27.9, 36.3]</td><td>7.32</td><td>8.41</td><td>98.5%</td></tr><tr><td>4</td><td>0.915</td><td>[24.5, 25.6]</td><td>[23.4, 27.1]</td><td>7.02</td><td>8.41</td><td>98.7%</td></tr><tr><td>5</td><td>0.937</td><td>[18.9, 21.1]</td><td>[15.5, 25.3]</td><td>6.71</td><td>8.32</td><td>99.4%</td></tr></table>

![](images/5895c49cb465a602b13d6eb5b975b06e6634efbc3344f4516811ac50d3ab367f.jpg)

![](images/7d0947835f6292d0d4d2d3b1478bbb97d62a09b9aeeaa8ab79dc48c7f90c74da.jpg)

![](images/75aa12fefdef3d0137dec94f1222548232c2d6a7774f50c252d92df47a6e32be.jpg)  
Figure 2: Catalogue tiling on GAME $( M = 3 )$ . Normalized items (dots) and head outputs (stars) are projected onto the plane spanned by the static head directions $\bar { \mathbf { h } } _ { m } .$ , the average normalized output of each head over the test set. Items are coloured by the direction they are most aligned with, and the green cross marks the target. Each panel is a test query won by a different head, and in each the target lies in the sector of its winning head.

(P2) Each hypothesis remains query-dependent. By (P2), a hypothesis is not a fixed list of items: its ranking follows $\mathbf { q } ^ { \star } ( \mathbf { x } )$ . Empirically, the top-10 of a given head overlaps by only 0.017 to 0.026 Jaccard between two different histories, and in Figure 2 the head closest to the target changes from query to query. (P1) and (P2) together explain the oracle gap in Table 7. At the optimum, the target y belongs to a single cell ${ \mathcal { V } } _ { m }$ , where it competes only with the other items of ${ \mathcal { V } } _ { m } .$ , ranked by $\mathbf { q } ^ { \star }$ . Selecting that head therefore ranks y no worse than the Bayes predictor over the full catalogue. The cosine oracle of Equation (11) makes exactly this choice, as it picks the winning head on 98.9% to 99.7% of test pairs. This is the behavior of COMPOTE, where the oracle-selected head ranks the target substantially higher than the pooled list, while collapsed models exhibit no oracle gain.

(P3) The cells split the mass evenly. The win frequency of a head (the fraction of test targets it wins), estimates $\mathbb { E } _ { \mathbf { x } } [ Q _ { m } ( \mathbf { x } ) ]$ without access to $\mathbf { q } ^ { \star }$ . Win frequencies stay within 1.6 points of $1 / M$ for every $M ,$ and their normalized entropy exceeds 0.999. Balance holds in mass and not in item counts: the heads own between 15.5% and 25.3% of the catalogue at $M = 5$ , yet each wins close to 20% of the targets. This is the balance (P3) predicts, since $\bar { Q } _ { m }$ weights items by $\mathbf { q } ^ { \star }$ , so a head with fewer items holds items that are more often the next one. Balance also matters for aggregation: if $Q _ { m } = 1 / M$ for all $m ,$ , Equation (12) gives $\omega ( \mathbf { p } ^ { \star } ) = M \mathbf { q } ^ { \star }$ , so max-pooling recovers the Bayes ranking exactly. Balance also fixes the value of the optimal loss, and Section K.3 checks that the test loss follows the predicted log M law.

## 6 DISCUSSION AND CONCLUSION

We introduced COMPOTE, which uses annealed multiple-hypothesis learning to produce complementary rankings for sequential recommendation. Our theoretical and empirical results show that it prevents head collapse, preserves ranking quality, and improves performance across three benchmarks. We leave combining COMPOTE with semantic identifiers for future work, which could improve recommendations for rare items. It also remains to explain why its theoretically query-dependent cells form an almost static catalogue tiling and how this structure relates to itemembedding geometry (Section M).

## AI USE STATEMENT

We used generative AI tools to polish the manuscript, primarily by correcting grammar and improving clarity, and to assist with portions of the experimental code. Parts of the theoretical proofs (the generalization bounds) were also drafted with the help of generative AI, but all proofs were verified and corrected by the authors.

## ACKNOWLEDGMENTS

This project was provided with computing HPC and storage resources by GENCI at IDRIS thanks to the grant 2026-A0191016862 on the supercomputer Jean Zay’s V100 partition.

## REFERENCES

Amazon Reviews’23 Benchmark. Pure IDs (5-core) data processing. https:// amazon-reviews-2023.github.io/data\_processing/5core.html, 2023. Accessed August 2026. 5, 19

Peter L Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. Journal ofmachine learning research, 3(Nov):463–482, 2002. 17

Olivier Bousquet and Andre Elisseeff. Stability and generalization. ´ Journal of machine learning research, 2(Mar):499–526, 2002. 16

Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang. Controllable multiinterest framework for recommendation. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2942–2951. ACM, 2020. doi: 10. 1145/3394486.3403344. 1, 5, 6, 20

Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for YouTube recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems, pages 191–198. ACM, 2016. doi: 10.1145/2959100.2959190. 1

Abner Guzman-Rivera, Dhruv Batra, and Pushmeet Kohli. Multiple choice learning: Learning to produce multiple structured outputs. In Advances in Neural Information Processing Systems 25, pages 1799–1807, 2012. URL https://proceedings.neurips.cc/paper/2012/ hash/cfbce4c1d7c425baf21d6b6f2babe6be-Abstract.html. 2

Balazs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. Session-based rec-´ ommendations with recurrent neural networks. In 4th International Conference on Learning Representations, 2016. URL http://arxiv.org/abs/1511.06939. 1, 5, 6, 20

Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian J. McAuley. Bridging language and items for retrieval and recommendation. CoRR, abs/2403.03952, 2024. URL https://doi.org/10.48550/ARXIV.2403.03952. 2, 19

Wang-Cheng Kang and Julian J. McAuley. Self-attentive sequential recommendation. In IEEE International Conference on Data Mining, pages 197–206. IEEE Computer Society, 2018. doi: 10.1109/ICDM.2018.00035. 1, 5, 6, 20

Michel Ledoux and Michel Talagrand. Probability in Banach Spaces: isoperimetry and processes, volume 23. Springer, 1991. 17

Stefan Lee, Senthil Purushwalkam, Michael Cogswell, Viresh Ranjan, David J. Crandall, and Dhruv Batra. Stochastic multiple choice learning for training diverse deep ensembles. In Advances in Neural Information Processing Systems 29, pages 2119– 2127, 2016. URL https://proceedings.neurips.cc/paper/2016/hash/ 20d135f0f28185b84a4cf7aa51f29500-Abstract.html. 2

Victor Letzelter, David Perera, Cedric Rommel, Mathieu Fontaine, Slim Essid, Ga´ el Richard, and¨ Patrick Perez. Winner-takes-all learners are geometry-aware conditional density estimators. In´ Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024. 2

Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. Multi-interest network with dynamic routing for recommendation at Tmall. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pages 2615–2623. ACM, 2019. doi: 10.1145/3357384.3357814. 1, 5, 6, 20

Chen Ma, Peng Kang, and Xue Liu. Hierarchical gating networks for sequential recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 825–833. ACM, 2019. doi: 10.1145/3292500.3330984. 5, 6, 20

Pascal Massart. Some applications of concentration inequalities to statistics. In Annales de la Faculte des sciences de Toulouse: Math´ ematiques´ , volume 9, pages 245–303, 2000. 19

Mehryar Mohri and Andres Munoz Medina. Learning theory and algorithms for revenue optimization in second price auctions with reserve. In Eric P. Xing and Tony Jebara, editors, Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 262–270, Bejing, China, 22–24 Jun 2014. PMLR. URL https://proceedings.mlr.press/v32/mohri14.html. 17

David Perera, Victor Letzelter, Theo Mariotte, Adrien Cort´ es, Mickael Chen, Slim Essid,´ and Gael Richard. Annealed multiple choice learning: Overcoming limitations of winner-¨ takes-all with annealing. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 10541–10574. Curran Associates, Inc., 2024. doi: 10.52202/079017-0337. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 1456560769bbc38e4f8c5055048ea712-Paper-Conference.pdf. 2, 3, 4

Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Mahesh Sathiamoorthy. Recommender systems with generative retrieval. In Advances in Neural Information Processing Systems 36, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html. 1, 5, 6, 19, 20

Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. Factorizing personalized Markov chains for next-basket recommendation. In Proceedings of the 19th International Conference on World Wide Web, pages 811–820. ACM, 2010. doi: 10.1145/1772690.1772773. 1

Christian Rupprecht, Iro Laina, Robert DiPietro, Maximilian Baust, Federico Tombari, Nassir Navab, and Gregory D. Hager. Learning in an uncertain world: Representing ambiguity through multiple hypotheses. In IEEE International Conference on Computer Vision, pages 3611–3620. IEEE Computer Society, 2017. doi: 10.1109/ICCV.2017.388. 2, 8, 23

Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pages 1441–1450. ACM, 2019. doi: 10.1145/3357384.3357895. 1, 5, 6, 20

Qiaoyu Tan, Jianwei Zhang, Jiangchao Yao, Ninghao Liu, Jingren Zhou, Hongxia Yang, and Xia Hu. Sparse-interest network for sequential recommendation. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining. ACM, 2021. doi: 10.1145/3437963. 3441811. 1, 5, 6, 20

Jiaxi Tang and Ke Wang. Personalized top-N sequential recommendation via convolutional sequence embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pages 565–573. ACM, 2018. doi: 10.1145/3159652.3159656. 1, 5, 6, 20

Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. Transformer memory as a differentiable search index. In Advances in Neural Information Processing Systems 35, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ 892840a6123b5ec99ebaab8be1530fba-Abstract-Conference.html. 1

Yueqi Xie, Jingqi Gao, Peilin Zhou, Qichen Ye, Yining Hua, Jae Boum Kim, Fangzhao Wu, and Sunghun Kim. Rethinking multi-interest learning for candidate matching in recommender systems. In Proceedings of the 17th ACM Conference on Recommender Systems. ACM, 2023. doi: 10.1145/3604915.3608766. 1, 6, 20

Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Jiayuan He, Yinghai Lu, and Yu Shi. Actions speak louder than words: Trillionparameter sequential transducers for generative recommendations. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id= xye7iNsgXn. 1, 5, 6, 20

Shengyu Zhang, Lingxiao Yang, Dong Yao, Yujie Lu, Fuli Feng, Zhou Zhao, Tat-Seng Chua, and Fei Wu. Re4: Learning to re-contrast, re-attend, re-construct for multi-interest recommendation. In Proceedings ofthe ACM Web Conference 2022, pages 2216–2226. ACM, 2022. doi: 10.1145/ 3485447.3512094. 5, 6, 20

Wayne Xin Zhao, Shanlei Mu, Yupeng Hou, Zihan Lin, Yushuo Chen, Xingyu Pan, Kaiyuan Li, Yujie Lu, Hui Wang, Changxin Tian, Yingqian Min, Zhichao Feng, Xinyan Fan, Xu Chen, Pengfei Wang, Wendi Ji, Yaliang Li, Xiaoling Wang, and Ji-Rong Wen. RecBole: Towards a unified, comprehensive and efficient framework for recommendation algorithms. In Proceedings of the 30th ACM International Conference on Information and Knowledge Management, pages 4653– 4664. ACM, 2021. doi: 10.1145/3459637.3482016. 19

Bowen Zheng, Enze Liu, Zhongfu Chen, Zhongrui Ma, Yue Wang, Wayne Xin Zhao, and Ji-Rong Wen. Pre-training generative recommender with multi-identifier item tokenization. arXiv preprint arXiv:2504.04400, 2025. URL https://arxiv.org/abs/2504.04400. 5, 6, 19

Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. S3-Rec: Self-supervised learning for sequential recommendation with mutual information maximization. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management, pages 1893–1902. ACM, 2020. doi: 10.1145/3340531.3411954. 1, 5, 6, 19, 20

## A PROOFS - SECTION 2

## A.1 PRELIMINARY RESULTS

Proposition 8 $( \ell _ { \tau }$ as a convex conjugate). For any $( \mathbf { y } , \mathbf { p } ) \in \mathcal { V } \times \left( \Delta ^ { N - 1 } \right) ^ { M }$ and for any $\tau \geq 0 ,$

$$
\ell _ { \tau } ( \mathbf { y } , \mathbf { p } ) = \operatorname* { m i n } _ { \gamma \in \Delta ^ { M - 1 } } \sum _ { m \leq M } - \gamma \log ( \mathbf { p } _ { m , \mathbf { y } } ) - \tau H ( \gamma ) .\tag{13}
$$

with optimal γ given by

$$
\gamma ( \mathbf { y } , \mathbf { p } ) = \mathrm { s o f t m a x } \left( \frac { 1 } { \tau } \log ( \mathbf { p } . , \mathbf { y } ) \right) .\tag{14}
$$

Proof. The log-sum-exp is the convex conjugate of the negative entropy.

## A.2 PROOF OF PROPOSITION 1

Proposition 1. For any $\begin{array} { r } { ( \mathbf { y } , \mathbf { p } ) \in \mathcal { V } \times ( \Delta ^ { N - 1 } ) ^ { M } , \operatorname* { l i m } _ { \tau  0 ^ { + } } \ell _ { \tau } ( \mathbf { y } , \mathbf { p } ) = \ell ^ { \mathrm { w T A } } ( \mathbf { y } , \mathbf { p } ) . } \end{array}$

Proof of Proposition 1. The proof is a direct consequence of Proposition 8 as lim $\cdot \tau \to 0 ^ { + }$ softmax $\begin{array} { r } { \left( \frac { 1 } { \tau } \log ( \mathbf { p } . \mathbf { , y } ) \right) = \operatorname* { m a x } _ { m \leq M } \frac { 1 } { \tau } \log ( \mathbf { p } _ { m , \mathbf { y } } ) } \end{array}$ □

A.3 PROOF OF PROPOSITION 2

Proposition 2 (Recovering aMCL). For any $( \mathbf { y } , \mathbf { p } ) \in \mathcal { V } \times \left( \Delta ^ { N - 1 } \right) ^ { M }$ and for any $\tau \geq 0 ,$

$$
\ell _ { \tau } ( \mathbf { y } , \mathbf { p } ) = \ell _ { \tau } ^ { \mathrm { a M C L } } ( \mathbf { y } , \mathbf { p } ) - \tau H ( \gamma ( \mathbf { y } , \mathbf { p } ) )\tag{5}
$$

where $\begin{array} { r } { H ( \gamma ) = - \sum _ { m \leq M } \gamma _ { m } \log \gamma _ { m } } \end{array}$ is the Shannon entropy of the softmax weights $\gamma ( \mathbf { y } , \mathbf { p } )$

ProofofProposition 2. This is a direct consequence of Proposition 8.

## A.4 PROOF OF COROLLARY 3

Corollary 3 (Recovering the gradient of aMCL). For any $( \mathbf { y } , \mathbf { p } ^ { \theta } ) \in \mathcal { V } \times \left( \Delta ^ { N - 1 } \right) ^ { M }$ (θ is some parameter), for any $\tau \geq 0 ,$ , and $\alpha \in \Delta ^ { M - 1 }$

$$
\nabla _ { \theta } \ell _ { \tau } ( \mathbf { y } , \mathbf { p } ^ { \theta } ) = \nabla _ { \theta } \psi _ { \tau } \left. \left( \mathbf { y } , \mathbf { p } ^ { \theta } , \alpha \right) \right| _ { \alpha = \gamma _ { m } ( \mathbf { y } , \mathbf { p } ^ { \theta } ) } \quad w i t h \quad \psi _ { \tau } ( \mathbf { y } , \mathbf { p } ^ { \theta } ) = - \sum _ { m \le M } \alpha _ { m } \log ( \mathbf { p } _ { m , \mathbf { y } } ^ { \theta } ) .\tag{6}
$$

Proof of Corollary 3.

$$
\nabla _ { \theta } \psi _ { \tau } ( \mathbf { y } , \mathbf { p } ^ { \theta } , \alpha ) | _ { \alpha = \gamma _ { m } ( \mathbf { y } , \mathbf { p } ^ { \theta } ) } = - \sum _ { m \leq M } \alpha _ { m } \frac { \nabla _ { \theta } \mathbf { p } _ { m , \mathbf { y } } ^ { \theta } } { \mathbf { p } _ { m , \mathbf { y } } ^ { \theta } } = - \sum _ { m \leq M } \frac { ( \mathbf { p } _ { m , \mathbf { y } } ^ { \theta } ) ^ { 1 / \tau - 1 } } { \sum _ { m ^ { \prime } \leq M } ( \mathbf { p } _ { m ^ { \prime } , \mathbf { y } } ^ { \theta } ) ^ { 1 / \tau } } \nabla _ { \theta } \mathbf { p } _ { m , \mathbf { y } } ^ { \theta } .\tag{15}
$$

On the other hand, differentiating $\begin{array} { r } { \ell _ { \tau } ( \mathbf { y } , \mathbf { p } ^ { \theta } ) = - \tau \log \left( \sum _ { m < M } ( \mathbf { p } _ { m , \mathbf { y } } ^ { \theta } ) ^ { 1 / \tau } \right) } \end{array}$ directly gives

$$
\nabla _ { \theta } \ell _ { \tau } ( \mathbf { y } , \mathbf { p } ^ { \theta } ) = - \tau \frac { \sum _ { m \le M } \frac { 1 } { \tau } ( \mathbf { p } _ { m , \mathbf { y } } ^ { \theta } ) ^ { 1 / \tau - 1 } \nabla _ { \theta } \mathbf { p } _ { m , \mathbf { y } } ^ { \theta } } { \sum _ { m ^ { \prime } \le M } ( \mathbf { p } _ { m ^ { \prime } , \mathbf { y } } ^ { \theta } ) ^ { 1 / \tau } } = - \sum _ { m \le M } \frac { ( \mathbf { p } _ { m , \mathbf { y } } ^ { \theta } ) ^ { 1 / \tau - 1 } } { \sum _ { m ^ { \prime } \le M } ( \mathbf { p } _ { m ^ { \prime } , \mathbf { y } } ^ { \theta } ) ^ { 1 / \tau } } \nabla _ { \theta } \mathbf { p } _ { m , \mathbf { y } } ^ { \theta } .\tag{16}
$$

This concludes the proof.

## B PROOFS OF SECTION 4

## B.1 PROOF OF PROPOSITION 5

$$
\begin{array} { r l r } { \mathbf { P r o p o s i t i o n } \ \mathsf { S } \left( \ell _ { \tau } \mathrm { \ v i a } \omega ^ { \tau } \right) . } & { { } { \mathit { F o r } a n y } \left( \mathbf { y } , \mathbf { p } \right) \in \mathcal { Y } \times \left( \Delta ^ { N - 1 } \right) ^ { M } a n d f o r a n y \tau \ge 0 , } & { } & { } \\ { \ell _ { \tau } \left( \mathbf { y } , \mathbf { p } \right) = - \log \omega _ { \mathbf { y } } ^ { \tau } \left( \mathbf { p } \right) } & { { } w h e r e } & { { } \omega ^ { \tau } ( \mathbf { p } ) = \left( \Big ( \displaystyle \sum _ { m \le M } \mathbf { p } _ { m , i } ^ { 1 / \tau } \Big ) ^ { \tau } \right) _ { i = 1 } ^ { N } a n d } & { \omega ^ { \tau } \xrightarrow { \displaystyle \ } \omega ^ { \tau } \omega ^ { \tau } } & { { } \displaystyle \omega ^ { \tau } \omega ^ { \tau } \left( \Delta ^ { N - 1 } \right) ^ { M } } \end{array}\tag{8}
$$

Proof of Proposition 5. Fix $\mathbf { \rho } ( \mathbf { y } , \mathbf { p } )$ . For $\tau > 0 .$ , by definition of $\omega ^ { \tau }$ in Equation (8),

$$
- \log \left( \omega _ { \mathbf { y } } ^ { \tau } \left( \mathbf { p } \right) \right) = - \log \left( \left( \sum _ { m \leq M } \mathbf { p } _ { m , \mathbf { y } } ^ { \frac { 1 } { \tau } } \right) ^ { \tau } \right) = - \tau \log \left( \sum _ { m \leq M } \mathbf { p } _ { m , \mathbf { y } } ^ { \frac { 1 } { \tau } } \right) = \ell _ { \tau } ( \mathbf { y } , \mathbf { p } ) ,\tag{17}
$$

the last equality being the definition of $\ell _ { \tau }$ in Equation (3).

For $\tau \ = \ 0 ,$ , understood as the limit $\tau  0 ^ { + }$ on both sides, $\omega ^ { 0 } = \omega \mathrm { g i v e s } - \log \left( \omega _ { \mathbf { y } } ^ { 0 } ( \mathbf { p } ) \right) =$ $- \log \operatorname* { m a x } _ { m \leq M } \mathbf { p } _ { m , \mathbf { y } } = \ell ^ { \mathrm { w T A } } ( \mathbf { y } , \mathbf { p } )$ , which equals lim ${ \bf \omega } _ { \tau \to 0 ^ { + } } \ell _ { \tau } ( { \bf y } , { \bf p } )$ by Proposition 1. □

## B.2 PROOF OF PROPOSITION 6

Proposition 6 (ω induces Bayes ranking). Let $\mathbf { q } ^ { \star } \in \Delta ^ { N - 1 }$ , and denote the optimal prediction $\begin{array} { r } { \mathbf { p } ^ { \star } \in \mathop { \mathrm { a r g m i n } } _ { \mathbf { p } \in ( \Delta ^ { N - 1 } ) ^ { M } } \mathbb { E } _ { \mathbf { y } \sim \mathbf { q } ^ { \star } } [ \ell ^ { \mathrm { W T A } } ( \mathbf { y } , \mathbf { \bar { p } } ) ] } \end{array}$ . Then

$$
\forall a , b \in \mathcal { V } , \qquad \mathbf { q } _ { a } ^ { \star } > \mathbf { q } _ { b } ^ { \star } \implies \omega _ { a } ( \mathbf { p } ^ { \star } ) > \omega _ { b } ( \mathbf { p } ^ { \star } ) \quad o r \quad \omega _ { a } ( \mathbf { p } ^ { \star } ) = \omega _ { b } ( \mathbf { p } ^ { \star } ) = 1 .\tag{9}
$$

Proof of Proposition 6. Denote the inner risk by $L ( \mathbf { p } ) = \mathbb { E } _ { \mathbf { y } \sim \mathbf { q } ^ { \star } } [ \ell ^ { \mathrm { W T A } } ( \mathbf { y } , \mathbf { p } ) ]$ ]. By Proposition 5,

$$
L ( \mathbf { p } ) = - \sum _ { y \in \mathcal { Y } } \mathbf { q } _ { y } ^ { \star } \log \omega _ { y } ( \mathbf { p } ) ,\tag{18}
$$

so that, $L ( \mathbf { p } ^ { \star } )$ being finite, $\omega _ { y } ( \mathbf { p } ^ { \star } ) > 0$ for every y with $\mathbf { q } _ { y } ^ { \star } > 0$ . Fix $a , b \in \mathcal { D }$ with $\mathbf { q } _ { a } ^ { \star } > \mathbf { q } _ { b } ^ { \star } ;$ ; in particular $\mathbf { q } _ { a } ^ { \star } > 0$ and $\omega _ { a } ( \mathbf { p } ^ { \star } ) > 0$

Weak inequality. Assume $\omega _ { b } ( \mathbf { p } ^ { \star } ) > 0 .$ , the conclusion being trivial otherwise, and let $\Pi _ { a b }$ be the operator associated with the transposition $\pi = ( a b )$ ). Then

$$
L ( \Pi _ { a b } { \bf p } ^ { \star } ) - L ( { \bf p } ^ { \star } ) = - { \bf q } _ { a } ^ { \star } \log \omega _ { b } ( { \bf p } ^ { \star } ) - { \bf q } _ { b } ^ { \star } \log \omega _ { a } ( { \bf p } ^ { \star } ) + { \bf q } _ { a } ^ { \star } \log \omega _ { a } ( { \bf p } ^ { \star } ) + { \bf q } _ { b } ^ { \star } \log \omega _ { b } ( { \bf p } ^ { \star } )\tag{19}
$$

$$
= ( { \bf q } _ { a } ^ { \star } - { \bf q } _ { b } ^ { \star } ) \log \frac { \omega _ { a } ( { \bf p } ^ { \star } ) } { \omega _ { b } ( { \bf p } ^ { \star } ) } ,\tag{20}
$$

the terms at $y \notin \{ a , b \}$ cancelling and the remaining logarithms being finite. Since ${ \cal L } ( \Pi _ { a b } \mathbf { p } ^ { \star } ) \geq$ $L ( \mathbf { p } ^ { \star } )$ and $\mathbf { q } _ { a } ^ { \star } > \mathbf { q } _ { b } ^ { \star }$ , this gives $\omega _ { a } ( \mathbf { p } ^ { \star } ) \geq \omega _ { b } ( \mathbf { p } ^ { \star } )$

Strictness up to saturation. Assume now $\omega _ { a } ( \mathbf { p } ^ { \star } ) = \omega _ { b } ( \mathbf { p } ^ { \star } )$ ; we show that this common value is 1. Let $( { \mathcal { V } } _ { m } ) _ { m \leq M }$ be the tessellation of Theorem $7 , m ( y )$ the index of the cell containing y and $Q _ { m }$ the mass of ${ \mathcal { N } } _ { m } ,$ so that $\omega _ { y } ( \mathbf { p } ^ { \star } ) = \mathbf { p } _ { m ( y ) , y } ^ { \star } = \mathbf { q } _ { y } ^ { \star } / Q _ { m ( y ) }$ whenever $Q _ { m ( y ) } > 0$ , and, by equation 18,

$$
L ( \mathbf { p ^ { \star } } ) = - \sum _ { y \in \mathcal { Y } } \mathbf { q } _ { y } ^ { \star } \log \frac { \mathbf { q } _ { y } ^ { \star } } { Q _ { m ( y ) } } = H ( \mathbf { q } ^ { \star } ) + \sum _ { m \leq M } \phi ( Q _ { m } ) , \qquad \phi ( t ) = t \log t , \quad \phi ( 0 ) = 0 .\tag{21}
$$

Suppose first $\mathbf { q } _ { b } ^ { \star } > 0$ , so that $Q _ { A } , Q _ { B } > 0$ where $A = m ( a )$ and $B = m ( b )$ . The items a and b lie in distinct cells: were they in the same cell, the ratio $\dot { \omega } _ { a } ( \mathbf { p } ^ { \star } ) / \omega _ { b } ( \mathbf { p } ^ { \star } ) = { \mathbf { q } _ { a } ^ { \star } } / { \mathbf { q } _ { b } ^ { \star } } > 1$ would contradict the assumed equality. Exchanging a and b between the two cells produces a partition of $\mathcal { V }$ with masses $Q _ { A } - u$ and $Q _ { B } + u$ , where $u = \mathbf { q } _ { a } ^ { \star } - \mathbf { q } _ { b } ^ { \star } > 0$ , the other cells being unchanged; the predictor it defines through equation 12 is admissible, so equation 21 and the optimality of $\mathbf { p } ^ { \star }$ give

$$
g ( u ) \geq g ( 0 ) , \qquad \mathrm { w h e r e } \qquad g ( t ) = \phi ( Q _ { A } - t ) + \phi ( Q _ { B } + t ) .\tag{22}
$$

Writing $\Delta = Q _ { A } - Q _ { B }$ , we have $\begin{array} { r } { g ^ { \prime } ( t ) = \log \frac { Q _ { B } + t } { Q _ { A } - t } } \end{array}$ , so g decreases on $( - \infty , \Delta / 2 ]$ and increases on $[ \Delta / 2 , Q _ { A } )$ ; being symmetric about $\Delta / 2$ , it satisfies $g ( u ) \ge g ( 0 )$ with $u > 0$ only if $u \ge \Delta$ Consequently

$$
\mathbf { q } _ { a } ^ { \star } Q _ { B } - \mathbf { q } _ { b } ^ { \star } Q _ { A } \ge ( \mathbf { q } _ { b } ^ { \star } + \Delta ) Q _ { B } - \mathbf { q } _ { b } ^ { \star } ( Q _ { B } + \Delta ) = \Delta ( Q _ { B } - \mathbf { q } _ { b } ^ { \star } ) \ge 0 ,\tag{23}
$$

using $\mathbf { q } _ { a } ^ { \star } \geq \mathbf { q } _ { b } ^ { \star } + \Delta$ and $Q _ { B } \geq { \bf q } _ { b } ^ { \star } \mathrm { i f } \Delta > 0 .$ , and $\mathbf { q } _ { a } ^ { \star } > \mathbf { q } _ { b } ^ { \star }$ with $Q _ { A } \leq Q _ { B } { \mathrm { ~ i f ~ } } \Delta \leq 0$ . The assumed equality reads $\omega _ { a } ( \mathbf { p } ^ { \star } ) - \omega _ { b } ( \mathbf { p } ^ { \star } ) = ( \mathbf { q } _ { a } ^ { \star } Q _ { B } - \mathbf { q } _ { b } ^ { \star } Q _ { A } ) / ( \bar { Q } _ { A } Q _ { B } ) = 0$ and therefore forces $\Delta > 0$ $\mathbf { q } _ { a } ^ { \star } = \mathbf { q } _ { b } ^ { \star } + \Delta$ and $Q _ { B } = \mathbf { q } _ { b } ^ { \star }$ , whence $\omega _ { b } ( \mathbf { p } ^ { \star } ) = \mathbf { q } _ { b } ^ { \star } / Q _ { B } = 1$

Suppose finally $\mathbf { q } _ { b } ^ { \star } = 0$ . If $Q _ { B } > 0$ then $\omega _ { b } ( \mathbf { p } ^ { \star } ) = \mathbf { q } _ { b } ^ { \star } / Q _ { B } = 0 < \omega _ { a } ( \mathbf { p } ^ { \star } )$ , excluding the equality. Otherwise every $y \in \mathcal { V } _ { B }$ has $\mathbf { q } _ { y } ^ { \star } = 0 ;$ let $\mathbf { p } ^ { \prime }$ be $\mathbf { p } ^ { \star }$ with head B replaced by $e _ { a }$ . For $y \notin \mathcal { Y } _ { B } \cup \{ a \}$ the maximum defining $\omega _ { y }$ is attained by a head other than B, so $\omega _ { y } ( \mathbf { p } ^ { \prime } ) = \omega _ { y } ( \mathbf { p } ^ { \star } )$ ; the items of $\mathcal { { V } } _ { B }$ carry no mass; and $\omega _ { a } ( \mathbf { p } ^ { \prime } ) = 1 \geq \omega _ { a } ( \dot { \mathbf { p } ^ { \star } } )$ . Hence $L ( \mathbf { p } ^ { \prime } ) \leq L ( \mathbf { p } ^ { \star } )$ with equality only if $\dot { { \boldsymbol { \omega } } } _ { a } ( { \bf p } ^ { \star } ) = 1$ and the optimality of $\mathbf { p } ^ { \star }$ forces $\omega _ { a } ( \mathbf { p } ^ { \star } ) = 1$ □

## C PER-HEAD NORMALIZER

Protocol. For the $\begin{array} { r l r } { M } & { { } = } & { 3 } \end{array}$ GAME checkpoint of Table 2, we compute log $\begin{array} { r l } { Z _ { m } ( \mathbf { x } ) } & { { } = } \end{array}$ log $\begin{array} { r } { \sum _ { i \leq N } \exp ( s _ { m , i } ( \mathbf { x } ) ) } \end{array}$ over all $N \ = \ 2 5 { , } 6 1 2$ items for each of the 94,762 test queries, with $s _ { m , i } = \cos ( h _ { m } , \mathbf { e } _ { i } ) / \beta$ and $\beta = 0 . 0 7$

The normalizer is head-independent. The three distributions nearly overlap (Figure $^ { 3 ) , }$ with means 13.67, 13.63, and 13.79. At fixed x, the cross-head standard deviation of log $Z _ { m } ( \mathbf { x } )$ is 0.13 on average; its range has median 0.28, 95th percentile 0.62, and maximum 1.10.

Effect on the ranking. Because log max<sub>m</sub> softmax $\begin{array} { r } { ( \mathbf { s } _ { m , \cdot } ) = \operatorname* { m a x } _ { m } ( \mathbf { s } _ { m , i } - \log Z _ { m } ) } \end{array}$ , a ranking change requires max-cosine scores within $\beta$ times the cross-head range: 0.020 for the median query, 0.043 at the 95th percentile, and 0.077 at most. Accordingly, Table 2 shows no measurable loss from dropping the normalizer.

![](images/972a44bf3062a8549e7f1d4aa9e01d92b613965797b29edffbb7db2b61a6425d.jpg)  
Figure 3: Distribution of the per-head log-normalizer log $Z _ { m } ( \mathbf { x } )$ . Histogram of log $Z _ { m } ( \mathbf { x } ) =$ log $\begin{array} { r } { \sum _ { i < N } \exp \left( s _ { m , i } ( \mathbf { x } ) \right) } \end{array}$  over the 94,762 test queries of GAME, one colour per head, for the COM-POTE checkpoint with $M = 3$ of Table $2 \left( \beta = 0 . 0 7 , N = 2 5 , 6 1 2 \right)$ . The three heads have almost superimposed distributions, concentrated between 12.2 and 15.0 on a scale where log $Z _ { m }$ could a priori take any value in [log $N - 1 / \beta ,$ , log $N + 1 / \beta ] = [ - 4 . 1 , 2 4 . 4 ]$ . For a fixed query, the standard deviation of log $Z _ { m } ( \mathbf { x } )$ across the three heads is 0.13 on average, against a mean value of 13.7.

## D GENERALIZATION BOUND

## D.1 PROOF OF THEOREM 9

We first provide here a generalization bound for the log-loss in Theorem 9, which is a direct consequence of the Rademacher complexity of the hypothesis class $\mathcal { F } ^ { M }$ . We then provide a refined version of the generalization bound in Theorem 4, which has a milder dependence on the number of heads M.

Theorem 9 (log-loss generalization bound). Let F be a hypothesis class of probabilitygenerating heads $f _ { m } : \breve { \mathcal { X } } \to \Delta ^ { N - 1 }$ . Assume there exists a constant $\epsilon > 0$ such that $f _ { m , i } ( \mathbf { x } ) \geq \epsilon$ for all $m \in [ M ] , \mathbf { x } \in { \mathcal { X } } ,$ and $i \in [ N ]$ . Then, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ over the sample $S \sim P _ { { \bf x } , { \bf y } } ^ { n }$ , every ensemble $f = ( f _ { 1 } , \ldots , f _ { M } ) \in \mathcal { F } ^ { M }$ satisfies:

$$
\mathcal { L } ^ { \mathrm { w r a } } ( f , P _ { \mathbf { x } , \mathbf { y } } ) \leq \widehat { \mathcal { L } } _ { S } ^ { \mathrm { w r a } } ( f ) + \frac { 2 } { \epsilon } \sum _ { m = 1 } ^ { M } \mathcal { R } _ { n } ( \mathcal { F } _ { m , \mathbf { y } } ) + \log \left( \frac { 1 } { \epsilon } \right) \sqrt { \frac { \ln ( 2 / \delta ) } { 2 n } } ,\tag{24}
$$

where $\mathcal { R } _ { n } ( \mathcal { F } _ { m , \mathbf { y } } )$ is the expected Rademacher complexity of the set of functions $\mathcal { F } _ { m , \mathbf { y } } = \{ \mathbf { x } \mapsto $ $f _ { m , \mathbf { y } } ( \mathbf { x } ) : f _ { m } \in \mathcal { F } \}$

Proof of Theorem 9. Step 1: Boundedness and Lipschitz continuity of the loss function. Consider the scalar loss function $- \log ( t )$ restricted to $t \in [ \epsilon , 1 ]$

• Upper Bound (B): Since $t \geq \epsilon$ , the loss is non-negative and bounded above by:

$$
B = \operatorname* { s u p } _ { t \in [ \epsilon , 1 ] } ( - \log t ) = - \log ( \epsilon ) = \log \left( { \frac { 1 } { \epsilon } } \right) .
$$

Thus, for every sample $\left( \mathbf { x } _ { i } , \mathbf { y } _ { i } \right)$ , min ${ \ L } _ { m \in [ M ] } ( - \log f _ { m , \mathbf { y } _ { i } } ( \mathbf { x } _ { i } ) ) \in [ 0 , B ]$

• Lipschitz Continuity $( L _ { \epsilon } ) \colon$ By the Mean Value Theorem, the Lipschitz constant is bounded by the supremum of the derivative:

$$
L _ { \epsilon } = \operatorname* { s u p } _ { t \in [ \epsilon , 1 ] } \left| { \frac { d } { d t } } ( - \log t ) \right| = \operatorname* { s u p } _ { t \in [ \epsilon , 1 ] } { \frac { 1 } { t } } = { \frac { 1 } { \epsilon } } .
$$

Therefore, for any $t _ { 1 } , t _ { 2 } \ge \epsilon ,$ , we have $\begin{array} { r } { | ( - \log t _ { 1 } ) - ( - \log t _ { 2 } ) | \leq \frac { 1 } { \epsilon } | t _ { 1 } - t _ { 2 } | . } \end{array}$

## Step 2: Concentration via McDiarmid’s Inequality.

We want to show that the empirical loss on n finite samples $\widehat { \mathcal { L } } _ { S } ^ { \mathrm { w r A } } ( f )$ is close to the true expected loss $\mathcal { L } ^ { \mathrm { w r A } } ( f )$ uniformly across all possible models in our hypothesis class $\mathcal { F } ^ { M }$ . Define the worst-case generalisation gap function:

$$
\Phi ( S ) = \operatorname* { s u p } _ { f \in { \mathcal { F } } ^ { M } } \left( { \mathcal { L } } ^ { \operatorname { w a } } ( f , P _ { \mathbf { x } , \mathbf { y } } ) - { \widehat { \mathcal { L } } } _ { S } ^ { \operatorname { w T A } } ( f ) \right) .
$$

Suppose we change a single data point in $S = \left\{ ( \mathbf { x } _ { 1 } , \mathbf { y } _ { 1 } ) , \dotsc , ( \mathbf { x } _ { n } , \mathbf { y } _ { n } ) \right\}$ to create a neighboring dataset $S ^ { \prime } = \{ ( \mathbf { x } _ { 1 } , \mathbf { y } _ { 1 } ) , \dotsc , ( \mathbf { x } _ { i } ^ { \prime } , \mathbf { y } _ { i } ^ { \prime } ) , \dotsc , ( \mathbf { x } _ { n } , \mathbf { y } _ { n } ) \}$ . The change in $\Phi ( S )$ can be bounded as:

$$
| \Phi ( S ) - \Phi ( S ^ { \prime } ) | \leq \operatorname* { s u p } _ { f \in \mathcal { F } ^ { M } } \frac { 1 } { n } \left| \operatorname* { m i n } _ { m } ( - \log f _ { m } ( \mathbf { x } _ { i } ) _ { \mathbf { y } _ { i } } ) - \operatorname* { m i n } _ { m } ( - \log f _ { m } ( \mathbf { x } _ { i } ^ { \prime } ) _ { \mathbf { y } _ { i } ^ { \prime } } ) \right| \leq \frac { B } { n } = \frac { \log ( 1 / \epsilon ) } { n } .
$$

Because changing one sample alters $\Phi ( S )$ by at most $\begin{array} { r } { c _ { i } = \frac { \log ( 1 / \epsilon ) } { n } } \end{array}$ , we can apply McDiarmid’s inequality to bound the deviation of $\Phi ( { \bar { S } } )$ from its expectation (Bousquet and Elisseeff, 2002):

$$
\mathbb { P } \left( \Phi ( S ) - \mathbb { E } _ { S } [ \Phi ( S ) ] \geq t \right) \leq \exp \left( { \frac { - 2 t ^ { 2 } } { \sum _ { i = 1 } ^ { n } c _ { i } ^ { 2 } } } \right) .
$$

Evaluating $\begin{array} { r } { \sum _ { i = 1 } ^ { n } c _ { i } ^ { 2 } = n \cdot \left( \frac { \log ( 1 / \epsilon ) } { n } \right) ^ { 2 } = \frac { \log ^ { 2 } ( 1 / \epsilon ) } { n } } \end{array}$ . Set the probability tail exp $\begin{array} { r } { \left( \frac { - 2 t ^ { 2 } n } { \log ^ { 2 } ( 1 / \epsilon ) } \right) = \frac { \delta } { 2 } } \end{array}$ Then, solving for t gives

$$
t = \log \left( \frac { 1 } { \epsilon } \right) \sqrt { \frac { \ln ( 2 / \delta ) } { 2 n } } .
$$

Thus, with probability at least $1 - { \textstyle { \frac { \delta } { 2 } } }$

$$
\operatorname* { s u p } _ { f \in \mathcal { F } ^ { M } } \left( \mathcal { L } ^ { \mathrm { w r a } } ( f , P _ { \mathbf { x } , \mathbf { y } } ) - \widehat { \mathcal { L } } _ { S } ^ { \mathrm { w r a } } ( f ) \right) \leq \mathbb { E } _ { S } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } ^ { M } } \left( \mathcal { L } ^ { \mathrm { w r a } } - \widehat { \mathcal { L } } _ { S } ^ { \mathrm { w r a } } \right) \right] + \log \left( \frac { 1 } { \epsilon } \right) \sqrt { \frac { \ln ( 2 / \delta ) } { 2 n } } .
$$

## Step 3: Symmetrization and min-operator decomposition.

Our goal now is to bound the expected supremum $\begin{array} { r } { \mathbb { E } _ { S } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } ^ { M } } \left( \mathcal { L } ^ { \mathrm { w T A } } - \widehat { \mathcal { L } } ^ { \mathrm { w T A } } \right) \right] } \end{array}$ in terms of the Rademacher complexity of the hypothesis class $\mathcal { F } ^ { M }$ . For doing so, we can redefine $\mathbb { E } _ { S } [ \Phi ( S ) ]$ as follows:

$$
\mathbb { E } _ { S } [ \Phi ( S ) ] = \mathbb { E } _ { S } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } ^ { M } } \Big ( \mathbb { E } _ { S ^ { \prime } } [ \widehat { \mathcal { L } } _ { S ^ { \prime } } ^ { \mathrm { w T A } } ( f ) ] - \widehat { \mathcal { L } } _ { S } ^ { \mathrm { w T A } } ( f ) \Big ) \right] ,
$$

where $S ^ { \prime }$ is an independent ghost sample drawn from the same distribution. Then by gathering the terms, and using Jensen’s inequality, we can pull the expectation over $S ^ { \prime }$ outside of the supremum:

$$
\mathbb { E } _ { S } [ \Phi ( S ) ] \leq \mathbb { E } _ { S , S ^ { \prime } } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } ^ { M } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \ell ( f ; \mathbf { x } _ { i } ^ { \prime } , \mathbf { y } _ { i } ^ { \prime } ) - \ell ( f ; \mathbf { x } _ { i } , \mathbf { y } _ { i } ) \right) \right] .
$$

Now, since $\left( \mathbf { x } _ { i } , \mathbf { y } _ { i } \right)$ and $( \mathbf { x } _ { i } ^ { \prime } , \mathbf { y } _ { i } ^ { \prime } )$ are identically distributed, introducing independent random sign variables $\sigma _ { i } \in \{ - 1 , + 1 \}$ with $\dot { \mathbb { P } } ( \sigma _ { i } = 1 ) = 1 / 2$ does not change the distribution of their difference:

$$
\mathbb { E } _ { S } [ \Phi ( S ) ] \leq \mathbb { E } _ { S , S ^ { \prime } } \left[ \operatorname* { s u p } _ { f } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \left( \ell ( f ; \mathbf { x } _ { i } ^ { \prime } , \mathbf { y } _ { i } ^ { \prime } ) - \ell ( f ; \mathbf { x } _ { i } , \mathbf { y } _ { i } ) \right) \right] ,
$$

and using the triangle inequality leads to the following decomposition:

$$
\begin{array} { r l } & { \mathbb { E } _ { S } [ \Phi ( S ) ] \le \mathbb { E } _ { S ^ { \prime } , \sigma } \left[ \displaystyle \operatorname* { s u p } _ { f } \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \sigma _ { i } \ell ( f ; \mathbf { x } _ { i } ^ { \prime } , \mathbf { y } _ { i } ^ { \prime } ) \right] } \\ & { \qquad + \mathbb { E } _ { S , \sigma } \left[ \displaystyle \operatorname* { s u p } _ { f } \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } ( - \sigma _ { i } ) \ell ( f ; \mathbf { x } _ { i } , \mathbf { y } _ { i } ) \right] } \\ & { = 2 \mathbb { E } _ { S } \left[ \widehat { \mathcal { R } } _ { S } ( \mathcal { G } _ { \mathrm { e m a x } } ) \right] = 2 \mathcal { R } _ { n } ( \mathcal { G } _ { \mathrm { e m a x } } ) , } \end{array}
$$

where $\widehat { \mathcal { R } } _ { S } ( \mathcal { G } _ { \mathrm { e m a x } } )$ is the empirical Rademacher complexity of the composite pointwise minimum class:

$$
\mathcal { G } _ { \mathrm { e m a x } } = \left\{ ( \mathbf { x } , \mathbf { y } ) \mapsto \operatorname* { m i n } _ { m \in [ M ] } ( - \log f _ { m , \mathbf { y } } ( \mathbf { x } ) ) : f _ { m , \cdot } \in \mathcal { F } \right\} ,
$$

and $\mathcal { R } _ { n } ( \mathcal { G } _ { \mathrm { e m a x } } ) = \mathbb { E } _ { S } [ \widehat { \mathcal { R } } _ { S } ( \mathcal { G } _ { \mathrm { e m a x } } ) ] .$

For a fixed sample S, using the structural property of Rademacher complexity under pointwise minimum operators over M hypothesis classes (decomposition of the pointwise minimum as a weighted sum of indicator functions and using results from Bartlett and Mendelson (2002)), we bound $\widehat { \mathcal { R } } _ { S } ( \mathcal { G } _ { \mathrm { e m a x } } )$ as:

$$
\widehat { \mathcal { R } } _ { S } ( \mathcal { G } _ { \mathrm { e m a x } } ) \leq \sum _ { m = 1 } ^ { M } \widehat { \mathcal { R } } _ { S } ( - \log \circ \mathcal { F } _ { m , y } ) ,\tag{25}
$$

where ${ \mathcal { F } } _ { m , y } = \{ \mathbf { x } \mapsto f _ { m , y } ( \mathbf { x } ) : f \in { \mathcal { F } } \}$ . Since $t \mapsto - \log t$ is $( 1 / \epsilon )$ -Lipschitz on $[ \epsilon , 1 ]$ , applying Talagrand’s Contraction Lemma (Mohri and Medina, 2014; Ledoux and Talagrand, 1991) yields:

$$
\widehat { \mathcal { R } } _ { S } ( - \log \circ \mathcal { F } _ { m , y } ) \leq \frac { 1 } { \epsilon } \widehat { \mathcal { R } } _ { S } ( \mathcal { F } _ { m , y } ) .
$$

Taking expectations $\mathbb { E } _ { S }$ on both sides gives $\begin{array} { r } { \mathcal { R } _ { n } ( \mathcal { G } _ { \mathrm { e m a x } } ) \leq \frac { 1 } { \epsilon } \sum _ { m = 1 } ^ { M } \mathcal { R } _ { n } ( \mathcal { F } _ { m , y } ) } \end{array}$

Final combination. Substituting into $\mathbb { E } _ { S } [ \Phi ( S ) ] \le 2 \mathcal { R } _ { n } ( \mathcal { G } _ { \mathrm { e m a x } } )$ and applying concentration yields the stated bound. □

## D.2 PROOF OF THEOREM 4

Theorem 4 (refined log-loss generalization bound). Let $\mathcal { F }$ be a hypothesis class ofprobabilitygenerating heads $f _ { m } : \overline { { \mathcal { X } } } \overset { } {  } \bar { \Delta } ^ { N - 1 }$ . Assume there exists a constant $\epsilon > 0$ such that $f _ { m , i } ( \mathbf { x } ) \geq \epsilon$ for all $m \ { \overset { \cdot } { \in } } \ [ M ] , { \overset { \cdot } { \mathbf { x } } } \in { \mathcal { X } } ,$ , and $i \in [ N ]$ . Then, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ over the sample $S \sim P _ { { \bf x } , { \bf y } } ^ { n } ,$ every ensemble $f = ( f _ { 1 } , \ldots , f _ { M } ) \in \mathcal { F } ^ { M }$ satisfies:

$$
\mathcal { L } ^ { \mathrm { W I A } } ( f , \mathbb { P } _ { S } ) \leq \widehat { \mathcal { L } } _ { S } ^ { \mathrm { W I A } } ( f ) + \frac { 2 } { \epsilon } \operatorname* { m a x } _ { m \in [ M ] } \mathcal { R } _ { n } ( \mathcal { F } _ { m , \mathbf { y } } ) + 2 \log \left( \frac { 1 } { \epsilon } \right) \sqrt { \frac { 2 \ln M } { n } } + \log \left( \frac { 1 } { \epsilon } \right) \sqrt { \frac { \ln ( 2 / \delta ) } { 2 n } } .\tag{7}
$$

where $\mathcal { R } _ { n } ( \mathcal { F } _ { m , \mathbf { y } } )$ is the expected Rademacher complexity of $\cdot \mathcal { F } _ { m , \mathbf { y } } = \{ \mathbf { x } \mapsto f _ { m , \mathbf { y } } ( \mathbf { x } ) : f _ { m } \in \mathcal { F } \}$

ProofofTheorem 4. To prove the theorem, we need to show that the expected Rademacher complexity of the minimum operator over M heads satisfies:

$$
\mathbb { E } _ { S , \sigma } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } ^ { M } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \operatorname* { m i n } _ { m \in [ M ] } \left( - \log f _ { m , \mathbf { y } } ( \mathbf { x } _ { i } ) \right) \right] \leq \frac { 1 } { \epsilon } \operatorname* { m a x } _ { m \in [ M ] } \mathcal { R } _ { n } ( \mathcal { F } _ { m , \mathbf { y } } ) + B \sqrt { \frac { 2 \ln M } { n } } .\tag{26}
$$

and plugging this bound into the generalization bound of Theorem 9 gives the desired result. Basically, we replace the crude sum over heads in Equation (25) by this Massart-type control of the minimum operator, which yields a sharper dependence on M. Hence, we only need to prove Equation (26). That is done in the following steps.

Let $\mathcal { G } _ { m } = \left\{ ( \mathbf { x } , \mathbf { y } ) \mapsto - \log f _ { m , \mathbf { y } } ( \mathbf { x } ) : f _ { m } \in \mathcal { F } \right\}$ be the loss function class for head $m ,$ where each function $g _ { m } \in \mathcal { G } _ { m }$ takes values in $[ 0 , B ]$ with $\tilde { B ^ { \prime } } = \log ( 1 / \epsilon )$

Let $g _ { m } ( \mathbf { z } _ { i } ) = - \log f _ { m , \mathbf { y } } ( \mathbf { x } _ { i } )$ , where $\mathbf { z } _ { i } = ( \mathbf { x } _ { i } , \mathbf { y } _ { i } )$ . We analyze the empirical Rademacher process for a fixed sample S:

$$
\widehat { \mathcal { R } } _ { S } ( \mathcal { G } _ { \mathrm { e m a x } } ) = \mathbb { E } _ { \pmb { \sigma } } \left[ \operatorname* { s u p } _ { g _ { 1 } \in \mathcal { G } _ { 1 } , \ldots , g _ { M } \in \mathcal { G } _ { M } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \operatorname* { m i n } _ { m \in [ M ] } g _ { m } ( \mathbf { z } _ { i } ) \right] .
$$

Pointwise minimum upper bound. For any given input $\mathbf { z } _ { i }$ and any choice of functions $\left( g _ { 1 } , \ldots , g _ { M } \right)$ , the minimum can be written as:

$$
\operatorname* { m i n } _ { m \in [ M ] } g _ { m } ( \mathbf { z } _ { i } ) = g _ { m ^ { * } } ( \mathbf { z } _ { i } ) \quad \mathrm { w h e r e } m ^ { * } = \arg \operatorname* { m i n } _ { m \in [ M ] } g _ { m } ( \mathbf { z } _ { i } ) .
$$

Instead of summing over all $m ,$ , we decouple the choice of the minimizing head index $m ^ { * }$ from the individual head function space. For any fixed sequence of functions $\left( g _ { 1 } , \ldots , g _ { M } \right)$ , we have:

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \operatorname* { m i n } _ { m \in [ M ] } g _ { m } ( \mathbf { z } _ { i } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } g _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) .
$$

α-cover discretization of the minima vector. To apply Massart’s Lemma over the finite M choices of heads, we construct an empirical $L _ { 2 }$ -net $\mathcal { N } ( \alpha , \mathcal { G } _ { m } , \bar { S } )$ for each function class $\mathcal { G } _ { m }$ at scale $\alpha > 0$ Hence, for every $g _ { m } \in \mathcal { G } _ { m }$ , there exists a net element $\bar { g } _ { m } \in \mathcal { N } ( \alpha , \mathcal { G } _ { m } , S )$ such that:

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } | g _ { m } ( \mathbf { z } _ { i } ) - \bar { g } _ { m } ( \mathbf { z } _ { i } ) | ^ { 2 } \leq \alpha ^ { 2 } .
$$

Then, using Cauchy–Schwarz, the discretization error introduced by replacing $g _ { m _ { i } ^ { * } }$ with its closest net element $\hat { g } _ { m _ { i } ^ { * } }$ is bounded by:

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \left( g _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) - \bar { g } _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) \right) \leq \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } ^ { 2 } } \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | g _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) - \bar { g } _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) | ^ { 2 } } \leq \alpha .
$$

Decomposing into single head and Massart’s finite maximum. For any tuple of true functions $( g _ { 1 } , \ldots , g _ { M } ) \in { \mathcal { G } } _ { 1 } \times \cdot \cdot \cdot \times { \mathcal { G } } _ { M }$ , we find their corresponding net approximations $( { \dot { g } } _ { 1 } , \dots , { \bar { g } } _ { M } ) \in { \mathcal { N } } ^ { M }$ The empirical sum splits into:

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } g _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \bar { g } _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) + \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \left( g _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) - \bar { g } _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) \right) .
$$

Taking the supremum over all valid choices of tuples:

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { ( g _ { 1 } , \ldots , g _ { M } ) \in \Pi _ { m } \mathcal { G } _ { m } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } g _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) } } \\ & { } & { \leq \operatorname* { s u p } _ { ( \bar { g } _ { 1 } , \ldots , \bar { g } _ { M } ) \in \mathcal { N } ^ { M } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \bar { g } _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) } \\ & { } & { + \operatorname* { s u p } _ { \bar { g } _ { m } \in \mathcal { S } _ { m } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } ( g _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) - \bar { g } _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) ) , } \end{array}
$$

where the final term is at most α, which gives us the following upper bound on the empirical Rademacher process:

$$
R ( S ) \leq \mathbb { E } _ { \pmb { \sigma } } \left[ \operatorname* { s u p } _ { ( \bar { g } _ { 1 } , \dots , \bar { g } _ { M } ) \in { \cal N } ^ { M } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \bar { g } _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) \right] + \alpha .
$$

Since $m _ { i } ^ { * } \in [ M ] ,$ , we decompose $\bar { g } _ { m _ { i } ^ { * } } \left( \mathbf { z } _ { i } \right)$ into an arbitrary reference head $( \operatorname { s a y } m = 1 )$ plus a residual vector in $\mathbb { R } ^ { n }$ representing the candidate selection among M heads:

$$
\bar { g } _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) = \bar { g } _ { 1 } ( \mathbf { z } _ { i } ) + \left( \bar { g } _ { m _ { i } ^ { * } } ( \mathbf { z } _ { i } ) - \bar { g } _ { 1 } ( \mathbf { z } _ { i } ) \right) .
$$

Table 5: Statistics of the preprocessed datasets (Zheng et al., 2025). Avg. len is the average length of the user interaction sequences.
<table><tr><td>Dataset</td><td>#Users</td><td>#Items</td><td>#Interactions</td><td>Sparsity</td><td> $\operatorname { A v g } .$  len</td></tr><tr><td>Instrument</td><td>57,439</td><td>24,587</td><td>511,836</td><td>99.964%</td><td>8.91</td></tr><tr><td>Scientific</td><td>50,985</td><td>25,848</td><td>412,947</td><td>99.969%</td><td>8.10</td></tr><tr><td>Game</td><td>94,762</td><td>25,612</td><td>814,586</td><td>99.966%</td><td>8.60</td></tr></table>

Then, taking expectation gives the following decomposition of the empirical Rademacher process:

$$
R ( S ) \leq \underbrace { \mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { g _ { 1 } \in \mathcal { G } _ { 1 } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } g _ { 1 } ( \mathbf { z } _ { i } ) \right] } _ { \mathrm { E m p i r i c a l ~ R a d e m a c h e r ~ o f ~ s i n g l e h e a d ~ } \widehat { \mathcal { R } } _ { S } ( g _ { 1 } ) } + \underbrace { \mathbb { E } _ { \sigma } \left[ \operatorname* { m a x } _ { m \in [ M ] } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \mathbf { v } _ { m } ( i ) \right] } _ { \mathrm { F i n i t e ~ m a x i m u m o v e r ~ } M \mathrm { ~ c a n d i d a t e ~ v e c t o r s } } + \alpha ,
$$

where $\mathbf v _ { m } ( i ) = \bar { g } _ { m } ( \mathbf z _ { i } ) - \bar { g } _ { 1 } ( \mathbf z _ { i } ) \in [ - B , B ]$ as $\bar { g } _ { m }$ and $\bar { g } _ { 1 }$ are both in the loss function class $\mathcal { G } _ { m }$ and hence bounded by $B = \log ( 1 / \epsilon )$

For M vectors $\mathbf { v } _ { 1 } , \dots , \mathbf { v } _ { M } \ \in \ \mathbb { R } ^ { n }$ with bounded norm $\| \mathbf { v } _ { m } \| _ { 2 } ~ \le ~ \sqrt { n } B$ , Massart’s Finite Class Lemma (Massart, 2000, Lemma 5.2) states that:

$$
\mathbb { E } _ { \pmb { \sigma } } \left[ \operatorname* { m a x } _ { m \in [ M ] } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \mathbf { v } _ { m } ( i ) \right] \leq \operatorname* { m a x } _ { m \in [ M ] } \frac { \| \mathbf { v } _ { m } \| _ { 2 } } { n } \sqrt { 2 \ln M } \leq B \sqrt { \frac { 2 \ln M } { n } } .
$$

We apply Talagrand’s Contraction Lemma on the single-head term and use the fact that the loss function is 1/ϵ-Lipschitz. In addition, by taking the infimum over all valid cover radii $\alpha > 0$ (or letting $\alpha = \hat { 1 / \sqrt { n } } )$ , the residual discretization error α is absorbed into the standard covering-number bound of the individual head complexity $\mathcal { R } _ { n } ( \mathcal { F } _ { 1 , \mathbf { y } } )$ . This gives us the following upper bound on the empirical Rademacher process:

$$
\mathbb { E } _ { S } [ \widehat { \mathcal { R } } _ { S } ( { \mathcal { G } } _ { 1 } ) ] \leq \frac { 1 } { \epsilon } \mathcal { R } _ { n } ( \mathcal { F } _ { 1 , { \mathbf { y } } } ) .
$$

Finally, since the choice of head $m = 1$ as reference was arbitrary, we take the maximum over all m $\in [ M ]$ :

$$
\mathbb { E } _ { S , \sigma } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } ^ { M } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \operatorname* { m i n } _ { m \in [ M ] } \left( - \log f _ { m } ( \mathbf { x } _ { i } ) _ { y _ { i } } \right) \right] \leq \frac { 1 } { \epsilon } \operatorname* { m a x } _ { m \in [ M ] } \mathcal { R } _ { n } ( \mathcal { F } _ { m , \mathbf { y } } ) + B \sqrt { \frac { 2 \ln M } { n } } .
$$

Substituting this refined bound for $\mathcal { R } _ { n } ( \mathcal { G } _ { \mathrm { e m a x } } )$ into the symmetrization argument of Section D.1 and combining with McDiarmid’s inequality yields the corollary. □

## E DATASET DESCRIPTION

We use three category subsets of the Amazon 2023 review corpus (Hou et al., 2024) MUSICAL INSTRUMENTS, INDUSTRIAL AND SCIENTIFIC, and VIDEO GAMES, abbreviated Instrument, Scientific, and Game which span May 1996 to September 2023. Following prior work (Rajput et al., 2023; Zhou et al., 2020; Zheng et al., 2025), we apply the benchmark 5-core filtering (Amazon Reviews’23 Benchmark, 2023), which iteratively discards users and items with fewer than five interactions. We then group the interactions by user, order them chronologically, and truncate the histories to 50 items at both training and test time. The resulting statistics are reported in Table 5.

## F COMPARED MODELS

We detail here the baselines summarized in Section 4. Numbers for the sequential and generative recommenders come from Zheng et al. (2025), who train them with RecBole (Zhao et al., 2021) under a common protocol and tune them by grid search. The multi-interest models are trained by us.

## ID-based sequential recommenders.

1. Caser (Tang and Wang, 2018) applies horizontal and vertical convolutions to the embedded interaction sequence, capturing both positional and skip-behaviour patterns.

2. HGN (Ma et al., 2019) combines feature-level and instance-level gating to modulate item embeddings according to the user representation.

3. GRU4Rec (Hidasi et al., 2016) encodes the interaction sequence with gated recurrent units trained for session-based next-item prediction.

4. BERT4Rec (Sun et al., 2019) uses a bidirectional transformer encoder trained with a masked item objective, so that each position is predicted from both past and future context.

5. SASRec (Kang and McAuley, 2018) uses a unidirectional (causal) self-attention encoder and predicts the next item by inner product with the item embeddings; it is the closest single-hypothesis analogue of our backbone.

6. HSTU (Zhai et al., 2024) introduces hierarchical sequential transducers that consume actions and timestamps jointly, and is designed for scalability while remaining ID-based.

7. S3-Rec (Zhou et al., 2020) pre-trains the sequence encoder with self-supervised objectives built from item–attribute correlations, then fine-tunes it for next-item prediction.

Generative recommender. TIGER (Rajput et al., 2023) quantizes item text embeddings into semantic identifiers with an RQ-VAE and casts recommendation as autoregressive generation of the target identifier, retrieving items by beam search over the identifier space.

Multi-interest recommenders. These models represent a user by M vectors instead of one, and are the closest existing analogue of our multi-hypothesis parameterization. We train them with the sampled-softmax objective and the readout rule of their reference implementations, and rank the catalogue by the maximum inner product over the M interest vectors, which is the retrieval rule all of them adopt.

1. MIND (Li et al., 2019) extracts interests with a dynamic-routing capsule network over the history and trains the interest whose inner product with the target is largest.

2. ComiRec (Cen et al., 2020) comes in two variants: ComiRec-DR keeps the capsule routing of MIND with a bilinear mapping, while ComiRec-SA replaces it by a multi-head self-attentive pooling of the history.

3. SINE (Tan et al., 2021) selects a sparse subset of concepts from a large learned concept pool, forms one interest per selected concept, and aggregates them back into a single user vector used for retrieval.

4. Re4 (Zhang et al., 2022) augments ComiRec-SA with three backward-flow regularizers— contrasting interests against each other, realigning the attention with the retrieval criterion, and reconstructing the assigned items from each interest.

5. REMI (Xie et al., 2023) augments ComiRec-SA with interest-aware hard-negative mining in the sampled softmax and a variance-based regularizer on the item-to-interest routing matrix that prevents routing collapse.

## G ABLATION ON THE SCORE-LOSS WEIGHT

This appendix measures how much of the performance of COMPOTE comes from the score loss of Equation (31), whose weight in the training objective is α (Equation (33)).

Protocol. We train COMPOTE on SCIENTIFIC with M = 3 for α ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7}, all other hyperparameters being fixed to the values of Section 4. Rankings use max-similarity aggregation (Equation (1)), which does not read γˆ at inference. The score loss can therefore only change the ranking through its gradient on the shared encoder; the oracle assignments it regresses are detached from the computation graph. We report N@5 at the best checkpoint.

The score loss helps. The best weight, α = 0.1, reaches 1.91 N@5 against 1.82 without the score loss (Figure 4), and four of the six positive weights improve over α = 0. We use α = 0.1 in all other experiments.

It is not the main contributor. With $\alpha \ : = \ : 0$ , COMPOTE already outperforms every singlehypothesis variant of Table 3 on SCIENTIFIC: 1.82 N@5 against 1.71 for MCT-SASRec, 1.64 for MCT-Router and 1.60 for MCT-EnsCE. The gain over single-hypothesis training is therefore obtained by the multiple-hypothesis objective alone. Moreover, the curve is not monotonic: $\alpha = 0 . 2$ and $\alpha = 0 . 5$ fall below $\alpha = 0$ , so the effect of the score loss is of the same order as the variation across weights. The score loss acts as an auxiliary regularizer of the shared encoder rather than as the source of the improvement.

![](images/f4b35a0faf58a309f30b2947f825ad1a06c17edfb09e77828ad5c5c32571f2b8.jpg)  
Figure 4: Effect of the score-loss weight α. N@5 (%) on SCIENTIFIC with $M = 3$ under maxsimilarity aggregation, as a function of the weight α of the score loss in Equation (33). Ticks are evenly spaced and do not follow the scale of α. Without the score loss $( \alpha = 0 )$ , COMPOTE already reaches 1.82, above every single-hypothesis variant of Table 3; the best weight $\alpha = 0 . 1$ adds 0.09 points.

## H THE CONFIDENCE-SCORE STABILIZATION

This appendix formalizes the regularization term mentioned at the end of Section 3. It is an auxiliary supervised objective attached to the confidence-score head $\hat { \gamma }$ of Figure 1, which is trained to predict which hypothesis the target item belongs to.

Confidence-score head. Recall the architecture of Figure 1: the shared causal encoder maps the history to the hidden state $\mathbf { z } = \operatorname { E n c } ( \mathbf { x } ) \in \mathbb { R } ^ { d }$ at the most recent position, the M hypothesis heads $h _ { 1 } , \ldots , h _ { M }$ map z to query representations $h _ { m } ( \mathbf { x } ) \ \in \ \mathbb { R } ^ { d }$ , and the catalogue scores $\mathbf { s } _ { m , i } ( \mathbf { x } )$ of Equation (10) are turned into the per-hypothesis distributions $f _ { m , \cdot } ( \mathbf { x } ) = \mathrm { s o f t m a x } _ { i } \left( \mathbf { s } _ { m , \cdot } ( \mathbf { x } ) \right) \in$ $\Delta ^ { N - 1 }$ . In parallel, an MLP taking the same z as input, followed by a coordinate-wise sigmoid, defines the confidence-score head

$$
{ \hat { \gamma } } : \mathcal { X } \to ( 0 , 1 ) ^ { M } ,\tag{27}
$$

where $\hat { \gamma } _ { m } ( \mathbf { x } )$ is read as the predicted probability that hypothesis m is the one containing the target of query x. Note that the $M$ outputs are independent sigmoids and not a softmax: $\hat { \gamma } ( \mathbf { x } )$ need not sum to one.

Supervision target: belonging to ${ \mathcal { V } } _ { m } .$ . Following the tessellation of Theorem 7, the empirical cell of hypothesis m on query x is the set of items for which m is the most confident hypothesis,

$$
\begin{array} { r } { \mathcal { V } _ { m } ( \mathbf { x } ) = \left\{ y \in \mathcal { V } : \forall m ^ { \prime } \neq m , - \log f _ { m , y } ( \mathbf { x } ) < - \log f _ { m ^ { \prime } , y } ( \mathbf { x } ) \right\} , } \end{array}\tag{28}
$$

so that $( \mathcal { V } _ { m } ( \mathbf { x } ) ) _ { m \leq M }$ is a partition of $\mathcal { V }$ up to ties. The winning hypothesis of a training pair $\displaystyle ( \mathbf { x } , \mathbf { y } )$ is the index of the cell that contains the target,

$$
m ^ { \star } ( { \bf x } , { \bf y } ) = \underset { m \leq M } { \mathrm { a r g m i n } } - \log f _ { m , { \bf y } } ( { \bf x } ) \quad \mathrm { i . e . } \quad { \bf y } \in \mathcal { Y } _ { m ^ { \star } ( { \bf x } , { \bf y } ) } ( { \bf x } ) ,\tag{29}
$$

which is exactly the hypothesis realizing the min in $\ell ^ { \mathrm { W T A } } ( \mathbf { y } , f ( \mathbf { x } ) )$ . The regression target of $\hat { \gamma }$ is the corresponding membership vector

$$
\gamma ^ { \star } ( \mathbf { x } , \mathbf { y } ) = \Big ( \mathbb { 1 } _ { [ \mathbf { y } \in \mathcal { Y } _ { m } ( \mathbf { x } ) ] } \Big ) _ { m = 1 } ^ { M } \in \{ 0 , 1 \} ^ { M } ,\tag{30}
$$

Table 6: Mean pairwise cosine similarity between hypothesis heads on the test set. A value of 1 means the heads are indistinguishable, lower is more specialized.
<table><tr><td>Method</td><td>Instrument</td><td>Scientific</td><td>Game</td></tr><tr><td>MCT-EnsCE</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>MCT-Router</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>COMPOTE (ours)</td><td>0.64</td><td>0.61</td><td>0.58</td></tr></table>

which is a one-hot vector, with its non-zero coordinate at $m ^ { \star } ( { \bf x } , { \bf y } )$

Binary cross-entropy on $\hat { \gamma } .$ . The regularizer is the binary cross-entropy between the membership indicators equation 30 and the predicted confidences, summed over the M coordinates,

$$
\ell ^ { \mathrm { s c o r e } } ( \mathbf { x } , \mathbf { y } , \hat { \gamma } ) = - \sum _ { m = 1 } ^ { M } \left[ \mathbb { 1 } _ { [ \mathbf { y } \in \mathcal { Y } _ { m } ( \mathbf { x } ) ] } \log \hat { \gamma } _ { m } ( \mathbf { x } ) + \mathbb { 1 } _ { [ \mathbf { y } \notin \mathcal { Y } _ { m } ( \mathbf { x } ) ] } \log \left( 1 - \hat { \gamma } _ { m } ( \mathbf { x } ) \right) \right] ,\tag{31}
$$

which, the target being one-hot, can be written equivalently as

$$
\ell ^ { \mathrm { s c o r e } } ( \mathbf { x } , \mathbf { y } , \hat { \gamma } ) = - \log \hat { \gamma } _ { m ^ { \star } ( \mathbf { x } , \mathbf { y } ) } ( \mathbf { x } ) - \sum _ { m \neq m ^ { \star } ( \mathbf { x } , \mathbf { y } ) } \log \left( 1 - \hat { \gamma } _ { m } ( \mathbf { x } ) \right) .\tag{32}
$$

The cells equation $^ { 2 8 , }$ and hence the labels equation 30, depend on $f \colon$ they are recomputed at every training step from the current hypotheses, and are detached from the computation graph, so that no gradient flows from $\ell ^ { \mathrm { s c o r e } }$ back to the heads $h _ { m }$ through $m ^ { \star }$ . The only gradient path of $\ell ^ { \mathrm { s c o r e } }$ is through γˆ, i.e. through the confidence MLP and the shared encoder producing z.

Full training objective. Weighting this term by $\alpha > 0$ , the objective minimized by COMPOTE is

$$
\mathcal { L } _ { \tau , \alpha } ( f , \hat { \gamma } ) = \mathbb { E } _ { \mathbf { x } , \mathbf { y } } \Big [ \ell _ { \tau } \big ( \mathbf { y } , f ( \mathbf { x } ) \big ) \Big ] + \alpha \mathbb { E } _ { \mathbf { x } , \mathbf { y } } \Big [ \ell ^ { \mathrm { s c o r e } } ( \mathbf { x } , \mathbf { y } , \hat { \gamma } ) \Big ] ,\tag{33}
$$

with $\ell _ { \tau }$ the annealed loss of Equation (3) – equivalently $\ell _ { \tau } ^ { \mathrm { a M C L } }$ of Equation (4) by Proposition 2 and Corollary 3. We use $\alpha = 0 .$ 1 in all experiments; the sensitivity to α is reported in Section G.

## I IMPACT OF THE NUMBER OF HYPOTHESES M ON LEARNING

In Theorem 4 in Section 3, we have established that the generalization gap between the empirical WTA loss and the expected WTA loss has a sublinear dependence on the number of hypotheses M. We present here an empirical validation of this theoretical insight.

Keeping the experimental setup used for performance evaluation described in Section 4, we vary the number of hypotheses M from 1 to 12 and measure the ranking quality on SCIENTIFIC using N@5. Figure 5 shows that increasing M from 1 to 3 improves ranking quality, which suggests that having multiple hypotheses is beneficial for capturing the ambiguity of the next-item distribution. However, beyond ${ \bar { M } } = 3 .$ , the performance plateaus, indicating that adding more hypotheses does not necessarily lead to better ranking quality. This empirical finding is consistent with the sublinear dependence on M in Theorem 4, and provides practical guidance for selecting an appropriate number of hypotheses in real-world applications.

For those larger value of M, a phenomenom we have described in Proposition 6 in Section 4 can also occur, which is the saturation of the max-similarity aggregation ω. Section 4. We have stated that this saturation can happen when at least two items have large probabilities $\mathbf { q } _ { a } ^ { \star } , \mathbf { q } _ { b } ^ { \star } \geq 1 / M$ . This means that as M increases, the probability of this saturation event increases, which can lead to a degradation of the ranking quality.

## J COLLAPSE AND ORACLE EVIDENCE

## K PROOFS OF SECTION 5

K.1 PROOF OF THEOREM 7

![](images/79b5de3e5f893970fbf504ab40eb29276dcb75b52245ab8edeca429dd0935f14.jpg)  
Figure 5: Effect of the number of hypotheses. N@5 on SCIENTIFIC under max-similarity aggregation, as a function of M. Performance improves sharply from $M = 1$ to $M = 3 .$ , then plateaus: increasing M from 3 to 12 does not degrade ranking quality, in line with the sublinear dependence on M in Theorem 4.

Table 7: NDCG@5 (%) under max-similarity aggregation (Equation (10)) and under oracle head selection (Equation (11)). Equal columns indicate that all heads induce the same ranking, i.e. complete collapse.
<table><tr><td></td><td colspan="2">Instrument</td><td colspan="2">Scientific</td><td colspan="2">Game</td></tr><tr><td>Method</td><td>Aggregated</td><td>Oracle</td><td>Aggregated</td><td>Oracle</td><td>Aggregated</td><td>Oracle</td></tr><tr><td>MCT-EnsCE</td><td>2.28</td><td>2.28</td><td>1.60</td><td>1.60</td><td>3.78</td><td>3.78</td></tr><tr><td>MCT-Router</td><td>2.28</td><td>2.28</td><td>1.64</td><td>1.64</td><td>3.78</td><td>3.78</td></tr><tr><td>COMPOTE (ours)</td><td>2.45</td><td>3.75</td><td>1.92</td><td>3.52</td><td>4.01</td><td>7.78</td></tr></table>

Theorem 7 (Extension of Th. 1 – Rupprecht et al. (2017)). Let $\mathbf { q } ^ { \star } \in \Delta ^ { N - 1 }$ , and denote the optimal prediction $\begin{array} { r } { \mathbf { p } ^ { \star } \in \mathrm { a r g m i n } _ { \mathbf { p } \in ( \Delta ^ { N - 1 } ) ^ { M } } \mathbb { E } _ { \mathbf { y } \sim \mathbf { q } ^ { \star } } [ \ell ^ { \mathrm { w T A } } ( \mathbf { y } , \mathbf { p } ) ] } \end{array}$ . Then for every $m \le M$ and $y \in \mathcal { V } _ { : }$

$$
{ \bf p } _ { m , y } ^ { \star } = \frac { { \bf q } _ { y } ^ { \star } } { Q _ { m } } \mathbb { 1 } _ { [ y \in \mathcal { V } _ { m } ] } , \quad w i t h \quad Q _ { m } = \sum _ { y \in \mathcal { V } _ { m } } { \bf q } _ { y } ^ { \star } .\tag{12}
$$

where the tessalation is $\mathcal { Y } _ { m } = \big \{ y \in \mathcal { Y } : \forall m ^ { \prime } \neq m , - \log ( \mathbf { p } _ { m , y } ^ { \star } ) < - \log ( \mathbf { p } _ { m ^ { \prime } , y } ^ { \star } ) \big \}$ . Moreover the tessalation $( \mathcal { V } _ { m } ) _ { m = 1 } ^ { M }$ is entropy-maximizing.

Proof of Theorem 7. The characterization of $\mathbf { p } ^ { \star }$ using $Q _ { m }$ and the tessalation ${ \mathcal { V } } _ { m }$ are a direct application of Theorem 7.

The only part remaining is to prove the tessalation ${ \mathcal { V } } _ { m }$ is entropy-maximizing. Given a tessalation $C = \{ \dot { C } _ { 1 } , \dots , C _ { m } \}$ of $\bar { y }$ , define

$$
\mathbf { p } _ { m , y } ^ { C } = { \frac { \mathbf { q } _ { y } ^ { \star } } { Q _ { m } ^ { C } } } \mathbb { 1 } _ { [ y \in C _ { m } ] } \quad { \mathrm { w h e r e } } \quad Q _ { m } ^ { C } = \sum _ { y \in C _ { m } } \mathbf { q } _ { y } ^ { \star }\tag{34}
$$

$$
\mathbb { E } _ { \mathbf { y } \sim \mathbf { q } ^ { \star } } [ \ell ^ { \mathrm { W T A } } ( \mathbf { y } , \mathbf { p } ^ { C } ) ] = - \sum _ { m = 1 } ^ { M } \sum _ { y \in C _ { m } } \mathbf { q } _ { y } ^ { \star } \log ( \mathbf { p } _ { m , y } ^ { C } )\tag{35}
$$

$$
= - \sum _ { m = 1 } ^ { M } \sum _ { y \in C _ { m } } \mathbf { q } _ { y } ^ { \star } \log \left( \frac { \mathbf { q } _ { y } ^ { \star } } { Q _ { m } ^ { C } } \right)\tag{36}
$$

$$
= H ( \mathbf { q } ^ { \star } ) - H ( Q ^ { C } )\tag{37}
$$

The choice $C ^ { \star } = ( \mathcal { V } _ { m } ) _ { m = } ^ { M }$ coincides with the optimal predictor – i.e. $\mathbf { p } ^ { C ^ { \star } } = \mathbf { p } ^ { \star }$ , leading to

$$
\mathbb { E } _ { { \mathbf { y } } \sim { \mathbf { q } } ^ { \star } } [ \ell ^ { \mathrm { W T A } } ( { \mathbf { y } } , { \mathbf { p } } ^ { C ^ { \star } } ) ] = \operatorname* { m i n } _ { { C : \mathrm { M } \cdot \mathrm { t e s s a l a t i o n } \ \mathrm { o f } { \mathbf { y } } } } \mathbb { E } _ { { \mathbf { y } } \sim { \mathbf { q } } ^ { \star } } [ \ell ^ { \mathrm { W T A } } ( { \mathbf { y } } , { \mathbf { p } } ^ { C } ) ]\tag{38}
$$

$$
\Leftrightarrow H ( \mathbf { q } ^ { \star } ) - H ( Q ^ { C ^ { \star } } ) = \operatorname* { m i n } _ { { C \cdot } \mathbf { M } \mathbf { \cdot } \mathbf { \Phi } \mathrm { e s s a l a t i o n } \mathrm { ~ o f } \mathcal { Y } } H ( \mathbf { q } ^ { \star } ) - H ( Q ^ { C } )\tag{39}
$$

$$
\Leftrightarrow H ( Q ^ { C ^ { \star } } ) = \operatorname* { m a x } _ { { C : \mathrm { M } } \mathrm { { \cdot } t e s s a l a t i o n } \mathrm { { \ o f } } \mathcal { V } } H ( Q ^ { C } )\tag{40}
$$

## K.2 OPTIMAL RISK

The following corollary makes explicit the value of the optimal winner-takes-all risk.

Corollary 10 (Optimal winner-takes-all risk). Under the setting of Theorem $^ 7 ,$ with $Q \ =$ $( Q _ { m } ) _ { m = 1 } ^ { M }$ the cell masses of the tessalation $( \mathcal { V } _ { m } ) _ { m = 1 } ^ { M }$

$$
\operatorname* { m i n } _ { \mathbf { p } \in ( \Delta ^ { N - 1 } ) ^ { M } } \mathbb { E } _ { \mathbf { y } \sim \mathbf { q } ^ { \star } } [ \ell ^ { \mathrm { W T A } } ( \mathbf { y } , \mathbf { p } ) ] = H ( \mathbf { q } ^ { \star } ) - H ( Q ) \geq H ( \mathbf { q } ^ { \star } ) - \log M ,\tag{41}
$$

with equality in the inequality if and only $i f Q _ { m } = 1 / M f o r e \nu e r y m \leq M .$

Proof of Corollary 10. By Theorem ${ } ^ { 7 } , \mathbf { p } ^ { \star } = \mathbf { p } ^ { C ^ { \star } }$ with $C ^ { \star } = ( \mathcal { Y } _ { m } ) _ { m = 1 } ^ { M }$ , and the computation in the proof of Theorem 7 gives

$$
\operatorname* { m i n } _ { \mathbf { p } \in ( \Delta ^ { N - 1 } ) ^ { M } } \mathbb { E } _ { \mathbf { y } \sim \mathbf { q } ^ { \star } } [ \ell ^ { \mathrm { w r a } } ( \mathbf { y } , \mathbf { p } ) ] = \mathbb { E } _ { \mathbf { y } \sim \mathbf { q } ^ { \star } } [ \ell ^ { \mathrm { w r a } } ( \mathbf { y } , \mathbf { p } ^ { C ^ { \star } } ) ] = H ( \mathbf { q } ^ { \star } ) - H ( Q ) .\tag{42}
$$

Since $Q \in \Delta ^ { M - 1 } , H ( Q ) \leq \log M$ , with equality if and only if Q is uniform.

## K.3 THE LOSS FOLLOWS THE log M LAW

Theorem 7 also predicts how the loss should change with M. For a given query, the optimal winnertakes-all loss is the Bayes loss minus the entropy $\bar { \cal H } ( Q ( { \bf x } ) )$ ) of the cell masses (Corollary 10). Exact querywise balance would therefore yield a log M reduction relative to the Bayes loss. Although the win frequencies of Section 5.2 establish only marginal balance, empirically $\hat { \mathcal { L } } ^ { \mathrm { w T A } } + \log M$ varies by less than 0.1 across $M \in \{ 3 , 4 , 5 \}$ (Table 4), showing that the learned models approximately follow this scaling law, where $\hat { \mathcal { L } } ^ { \mathrm { w T A } } = \mathbb { E } [ \operatorname* { m i n } _ { m } - \log f _ { m , \mathbf { y } } ( \mathbf { x } ) ]$ is the test loss. Equivalently, going from $M = 3 \tan { \bar { M } } = 5$ multiplies the geometric-mean probability that the winning head assigns to the target by $e ^ { 0 . 6 1 } \approx 1 . 8 3$ , close to and slightly above the ratio $\bar { 5 } / 3 \approx 1 . 6 7$ predicted under exact balance.

## L LATENT TILING ACROSS M

This appendix details the protocol behind Table 4, reports additional diagnostics on the three COM-POTE checkpoints trained on GAME with $M \in \{ 3 , { \dot { 4 } } , 5 \}$ , and checks that the partition also appears on SCIENTIFIC.

Protocol. Win frequencies, static sectors, and sector agreement use the full test set of 94,762 query–target pairs. In-cell mass and the winner-takes-all loss use the softmax $f _ { m } ( \mathbf { x } )$ over the whole catalogue, with $\beta = 0 . 0 7$ , on the same full test set. The in-cell mass is $\begin{array} { r } { \sum _ { y \in \mathcal { Y } _ { m } ( \mathbf { x } ) } \overset { \cdot } { f _ { m , y } } ( \mathbf { x } ) } \end{array}$ , averaged over heads and test pairs. The static direction $\widetilde { \mathbf { h } } _ { m }$ of head m is the renormalized average of its normalized outputs over the test set. Writing $\bar { \bf e } _ { i } = { \bf e } _ { i } / \| { \bf e } _ { i } \| _ { 2 }$ for the normalized embedding of item $i ,$ its sector is argmax $\mathrm { ~  ~ \Gamma ~ } _ { m } \langle \bar { \bf e } _ { i } , \widetilde { \bf h } _ { m } \rangle$ . Sector agreement is the fraction of test targets whose sector is their winning head. Cross-head Jaccard, region loyalty, and single-head reach are computed from the top-10 lists on 4,000 test queries sampled uniformly. Among catalogue items retrieved at least once, an item is reached by a single head if it appears in the top-10 of exactly one head. Within-head overlap compares the top-10 list of a head between randomly paired queries from this sample.

Table 8: Latent-space tiling on GAME as the number of hypotheses M varies. Mean off-diagonal cosine similarity is computed between the static head directions $\widetilde { \mathbf { h } } _ { m }$ (the dataset-wide average of normalized head outputs). The top-10 diagnostics use 4,000 test queries sampled uniformly without replacement. Cross-head Jaccard is the mean pairwise overlap of these lists across heads. Region loyalty is the fraction of a head’s top-10 items that fall in its catalogue sector.
<table><tr><td>M</td><td>Mean off-diag. cos()</td><td>Cross-head top-10 Jaccard</td><td>Region loyalty</td></tr><tr><td>3</td><td>-0.09</td><td>0.008</td><td>98.9%</td></tr><tr><td>4</td><td>-0.17</td><td>0.003</td><td>99.4%</td></tr><tr><td>5</td><td>-0.04</td><td>0.001</td><td>99.8%</td></tr></table>

Cosine and softmax winners agree. The oracle of Equation (11) selects heads by cosine similarity, whereas the cells of Theorem 7 are defined by probabilities, and each head carries its own softmax normalizer. The two definitions of the winning head agree on 98.9%, 99.4%, and 99.7% of test pairs for $M = 3 , 4 , 5$

The tiling sharpens with M. The static directions are well separated for every M (Figure 6). Cross-head top-10 Jaccard falls from 0.008 at M = 3 to 0.001 at $\dot { M } = 5$ , while region loyalty rises from 98.9% to 99.8% (Table 8). Theorem 7 predicts an exact partition for every M, so this trend describes how close training gets to the optimum rather than a property of the optimum itself.

The same partition appears on SCIENTIFIC. We repeat the overlap diagnostics on 4,000 sampled test queries for the COMPOTE checkpoint trained on SCIENTIFIC with M = 3. The top-10 lists of two heads on the same query have a mean Jaccard overlap of 0.001, and 94.5% of the catalogue items retrieved at test time are reached by a single head only. The top-10 of a given head overlaps by 0.025 to 0.044 Jaccard between two different histories, so each head still adapts to the query. Figure 7 shows three test queries. A plain PCA of the normalized catalogue already reveals three arms, and the heads sit along distinct arms, with the target close to one of them.

![](images/2313d0f1ffc09435f954039eae2825d352636857b0894d7b1dbff54317632fd1.jpg)  
(a) $M = 3$

![](images/1e4ae45c11596fc9a994bca72fb6b5218a1d7bac4bf1cee994af68ba0337beae.jpg)  
(b) M = 4

![](images/cf6132f12f533ff49ddc7739f86f1cb6b7a0e5f6297b7929f697f8c83dfc815a.jpg)  
(c) $M = 5$  
Figure 6: Latent-space tiling on GAME. Cosine similarity between the static head directions $\bar { \mathbf { h } } _ { m }$ (the dataset-wide average of normalized head outputs) for $M \in \{ 3 , 4 , 5 \}$ . Off-diagonal values remain near zero or negative, indicating that the heads occupy separated directions rather than collapsing onto one another.

## M BEYOND THE THEOREM: A STATIC TILING

Theorem 7 allows the cells to depend on the query. To test whether they do, we average the normalized outputs of each head over the test set and assign every catalogue item to the static sector of the direction it is most aligned with (protocol in Section L). The sector of the target predicts its winning head on 98.5% to 99.4% of test pairs (last column of Table 4), and more than 98.9% of each head’s top-10 items lie in its sector. Cell membership is thus almost a function of the item alone. COMPOTE learns a fixed tiling of the catalogue, and only the ranking inside each tile adapts to the history, as (P2) requires. A static tiling cannot balance every query: for a user whose likely next items all lie in one tile, Q(x) is concentrated on a single head, and that head gets no boost from

![](images/c4357c31ab2116941ac69b8bf1c8d5575c11db3fb1e964df0770c2043f7ddf90.jpg)

![](images/9fd2403e07db758a9d6538f753c5d5172459e5eb04b8171a99d8e1c34b443180.jpg)

![](images/fb990f969880d44c4ecb99ec46a71281f630f2b89bb43e1260154ed83ab315f6.jpg)  
Figure 7: Catalogue partition on SCIENTIFIC (M = 3). PCA projections on three test sequences (queries 4934, 23025, and 26314) under COMPOTE. Each panel shows the L2-normalized item catalogue in two dimensions (grey), the target next item (green cross), and the three head query embeddings (stars). The heads occupy distinct regions of the catalogue rather than collapsing onto a single direction.  
the partition. The balance reported in Section 5.2 holds on average over queries, which is what win frequencies measure.