# EXPLORING POST-HOC DISENTANGLEMENT OF AUDIO FOUNDATION MODELS FOR MUSIC SIMILARITY

Lucas de Souza Brandão Universidade de São Paulo lucasouza@usp.br

Gabriel da Costa Merlin Universidade de São Paulo gabrielcmerlin@usp.br

Diego Furtado Silva Universidade de São Paulo diegofsilva@usp.br

## ABSTRACT

Audio foundation models like CLAP, MuQ, and MERT excel at capturing general musical similarity. On the other hand, their entangled latent spaces limit interpretability and fine-grained attribute control. Conditional Similarity Networks (CSN) address this by pairing a backbone representation model with learned masks that gate the embedding space into attribute-specific subspaces. However, existing methods train the backbone from scratch, rather than leveraging pretrained foundation models. We instead present a systematic study of post-hoc disentanglement, adapting CSN to gate frozen pretrained embeddings. We evaluate adding trainable parameters to adapt the embedding under different masking and pooling regimes across two datasets. Our findings show that minimal adaptation capacity is sufficient for post-hoc gating to improve both attribute-specific disentanglement and overall retrieval performance, while pooling, masking, and depth choices yield limited additional gains.

## 1. INTRODUCTION

Musical similarity is inherently multi-faceted, as two tracks can be perceived as similar because they share characteristics such as genre, instrumentation, or emotional tone. Capturing this multiplicity requires representations rich enough to encode all these facets at once, making recent audio foundation models like CLAP [1], MuQ [2], and MERT [3] natural candidates. Learned from largescale pretraining, these foundation models excel at capturing overall musical similarity. Indeed, they capture perceptually dominant features like melody, closely aligning with human similarity judgments [4]. However, because these models collapse distinct musical characteristics into a single undifferentiated latent space, they lack disentanglement. In other words, the embedding cannot isolate or independently manipulate individual similarity factors. This makes it difficult to perform fine-grained tasks, such as retrieving tracks that match on one specific attribute, such as instrumentation, while differing on others. It also hinders the ability to explain why two tracks were retrieved as similar in the first place [5]. Disentangling these factors is essential not only for model interpretability, but also for building controllable music retrieval systems [6].

Conditional Similarity Networks (CSN) [7] offer a disentanglement mechanism by learning masks that gate the embedding space provided by a backbone model according to a chosen musical factor. CSN and related conditional or attribute-specific embedding approaches have already been applied to music similarity, exploring different learning objectives and architectures for disentangled similarity. Specifically, existing methods condition subspaces on factors like genre, mood, instrumentation [6], key, tempo, era [8], and structure representations across different hierarchy levels [9]. Furthermore, recent approaches extend these disentangled similarity spaces to specific instrumental parts from mixed signals [10]. These works, however, train their backbones directly from scratch on audio features. This approach discards the knowledge already captured by large pretrained foundation models and requires computational resources that are not always accessible. This raises the question of whether disentanglement can instead be accomplished post hoc by adapting or masking an embedding space from a pretrained model, rather than training a disentangled representation from scratch.

In this work, we empirically study post-hoc disentangled representation learning for musical similarity built on pretrained audio foundation models (CLAP, MuQ, and MERT). We begin with CSN, adapting it to gate the pretrained embedding space into attribute-conditioned subspaces for genre, instrument, and mood without touching the backbone, and systematically explore different lightweight strategies to adapt and mask the embedding space. We evaluate disentanglement across two datasets to assess generalization beyond the training distribution. We investigate the impact of adding trainable parameters to adapt the embedding under different masking and pooling regimes, aiming to obtain representations that are both disentangled and effective for similarity retrieval without requiring heavy retraining.

## 2. PROPOSED METHOD

We adapt CSN variants to extract disentangled, attributeconditioned representations for instrument, genre, and mood features from frozen audio backbones. Temporal pooling first aggregates features into global vectors. An adaptation module then transforms these vectors before applying CSN-like masks, which route them into distinct attribute-specific subspaces. We optimize all parameters end-to-end using a regularized triplet loss.

## 2.1 Backbones and Temporal Pooling

We extract frame-level representations from frozen pretrained audio backbones (CLAP, MuQ, or MERT), yielding a temporal sequence of shape $[ T , D _ { \mathrm { e m b } } ]$ , where $T$ denotes the number of temporal frames and $D _ { \mathrm { e m b } }$ is the backbone embedding dimension. Since $T$ remains large for multi-minute tracks, making downstream processing memory-intensive, we downsample the temporal axis via adaptive average pooling into 128 uniform segments. This yields a fixed sequence shape of $[ 1 2 8 , D _ { \mathrm { e m b } } ]$ , standardizing representations across tracks of varying durations.

## 2.2 Adaptation Modules and Masking Strategies

To aggregate the temporal sequence $[ 1 2 8 , D _ { \mathrm { e m b } } ]$ into a single $D _ { \mathrm { e m b } }$ -dimensional vector, we evaluate two architectural options prior to attribute gating: (i) non-parametric Mean Pooling, which averages the 128 temporal tokens, or (ii) parametric Attention Pooling, which learns attention scores over the 128 temporal tokens via a linear layer, normalizes them across time via Softmax, and computes a weighted sum to emphasize salient temporal regions.

The aggregated vector x is passed through one of three adaptation modules: Identity, which leaves x unchanged; the One Layer module, which applies LayerNorm, a linear projection, GELU activation, and a residual connection; and the MLP module, which applies LayerNorm, two linear projections with an intermediate GELU, dropout, and a residual connection. We denote the resulting vector by $\hat { \bf x }$

To capture distinct attributes, we define a conditionspecific mask vector $\mathbf { m } _ { c } ~ \in ~ \mathbb { R } ^ { D _ { e m b } }$ for each condition $c \in \mathcal { C } =$ {genre, instrument, mood}. We obtain attributeconditioned embeddings through element-wise gating:

$$
\mathbf { e } _ { c } = \mathbf { m } _ { c } \odot { \hat { \mathbf { x } } } .\tag{1}
$$

We follow [7] and compare two masking regimes. Fixed Masks use deterministic, non-overlapping binary masks $\mathbf { n } _ { c } \in \{ 0 , 1 \} ^ { D _ { e m b } }$ that define equally sized, contiguous partitions of embedding dimensions and remain frozen during training. Learned Masks are initialized randomly as $\mathbf { m } _ { c } \sim \mathcal { N } ( 0 . 9 , 0 . 7 ^ { 2 } )$ , updated via gradient descent, and passed through ReLU before gating.

## 2.3 Objective Function

The framework is optimized by combining a primary metric learning objective with auxiliary regularization terms that encourage mask sparsity, subspace orthogonality, and embedding stability. For a given condition $c \in { \mathcal { C } } .$ , the training signal is enforced via a margin-based triplet loss in the masked subspace:

$$
\mathcal { L } _ { \mathrm { t r i p l e t } } ^ { ( c ) } = \operatorname* { m a x } \left( 0 , d ( \mathbf { e } _ { c } ^ { a } , \mathbf { e } _ { c } ^ { p } ) - d ( \mathbf { e } _ { c } ^ { a } , \mathbf { e } _ { c } ^ { n } ) + \Delta \right) ,\tag{2}
$$

where $d ( \cdot , \cdot )$ denotes Euclidean distance, $\Delta$ is the margin, and $\mathbf { e } _ { c } ^ { a } , \mathbf { e } _ { c } ^ { p } , \mathbf { e } _ { c } ^ { n }$ represent the masked embeddings of the anchor, positive, and negative tracks, respectively. This objective pulls the anchor closer to tracks sharing condition c while pushing away non-matching tracks.

To prevent mask values from growing unbounded and inflating the gated embeddings, we apply an $L _ { 1 }$ penalty to each condition mask $\mathbf { m } _ { c }$ . This encourages sparse masks that retain only the dimensions relevant to each attribute:

$$
\mathcal { L } _ { \mathrm { m a s k } } ^ { ( c ) } = \| \mathbf { m } _ { c } \| _ { 1 } .\tag{3}
$$

To reduce redundancy across different attribute subspaces and encourage disentanglement, we apply an orthogonality penalty across all condition mask pairs:

$$
\mathcal { L } _ { \mathrm { o r t h o } } = \sum _ { c _ { i } \neq c _ { j } \in \mathcal { C } } \left( \frac { \mathbf { m } _ { c _ { i } } ^ { \top } \mathbf { m } _ { c _ { j } } } { \| \mathbf { m } _ { c _ { i } } \| _ { 2 } \| \mathbf { m } _ { c _ { j } } \| _ { 2 } } \right) ^ { 2 } .\tag{4}
$$

To regularize the latent space and discourage arbitrarily large embedding norms that could artificially shrink the effective triplet distance, we apply an $L _ { 2 }$ penalty to the unmasked representation xˆ before gating:

$$
\mathcal { L } _ { \mathrm { e m b } } = \| \hat { \mathbf { x } } \| _ { 2 } ^ { 2 } .\tag{5}
$$

Finally, all remaining models are optimized end-to-end, using the combined training objective:

$$
\mathcal { L } = \sum _ { c \in \mathcal { C } } \left( \mathcal { L } _ { \mathrm { t r i p l e t } } ^ { ( c ) } + \lambda _ { \mathrm { m a s k } } \mathcal { L } _ { \mathrm { m a s k } } ^ { ( c ) } \right) + \lambda _ { \mathrm { o r t h o } } \mathcal { L } _ { \mathrm { o r t h o } } + \lambda _ { \mathrm { e m b } } \mathcal { L } _ { \mathrm { e m b } } ,\tag{6}
$$

where $\lambda _ { \mathrm { m a s k } } , \lambda _ { \mathrm { o r t h o } }$ , and $\lambda _ { \mathrm { e m b } }$ control the relative contribution of the auxiliary regularization terms. Under the Fixed Masks regime, $\mathcal { L } _ { \mathrm { m a s k } } ^ { ( c ) }$ and ${ \mathcal { L } } _ { \mathrm { o r t h o } }$ remain constant and produce no gradients, leaving the adaptation module parameters as the only trainable parameters. Under the Identity module, $\mathcal { L } _ { \mathrm { e m b } }$ also produces no gradients.

## 3. EXPERIMENTAL DESIGN

To address the research questions, we designed the experiments described below, evaluating every combination of backbone, temporal pooling method, adaptation module, and masking regime introduced in Section 2.

## 3.1 Backbone Processing

Because these backbones have maximum input sequence lengths (e.g., CLAP accepts at most 10 seconds of audio per pass), long tracks are processed by extracting features from non-overlapping 10-second chunks and concatenating them along the time axis. For a given track, MERT and MuQ yield temporal feature sequences directly in the shape $[ T , D _ { \mathrm { e m b } } ]$ , while CLAP outputs dual-stream window representations of shape $[ 2 , T , D _ { \mathrm { e m b } } ]$ , which we average along the first dimension to obtain a uniform sequence.

<table><tr><td></td><td colspan="2">CLAP</td><td colspan="2">MERT</td><td colspan="2">MuQ</td></tr><tr><td>Configuration</td><td>Jamendo</td><td>MTT</td><td>Jamendo</td><td>MTT</td><td>Jamendo</td><td>MTT</td></tr><tr><td>Baseline</td><td> $6 9 . 3 \pm 0 . 0$ </td><td> $7 4 . 8 \pm 0 . 0$ </td><td> $5 8 . 1 \pm 0 . 0$ </td><td> $6 0 . 3 \pm 0 . 0$ </td><td> $6 4 . 3 \pm 0 . 0$ </td><td> $7 0 . 0 \pm 0 . 0$ </td></tr><tr><td> $\mathrm { I d e n t i t y } + \mathrm { M e a n } + \mathrm { L e a r n e d } \mathrm { M a s k }$ </td><td> $7 1 . 6 \pm 0 . 3$ </td><td> $7 6 . 5 \pm 0 . 1$ </td><td> $5 7 . 8 \pm 0 . 2$ </td><td> $6 0 . 2 \pm { 0 . 3 }$ </td><td> $6 4 . 6 \pm 1 . 1$ </td><td> $6 9 . 9 \pm 0 . 8$ </td></tr><tr><td> $\mathrm { I d e n t i t y } + \mathrm { A t t e n t i o n } + \mathrm { L e a r n e d } \ \mathrm { M a s k }$ </td><td> $7 0 . 0 \pm 0 . 3$ </td><td> $7 4 . 2 \pm 0 . 4$ </td><td> $5 8 . 1 \pm 0 . 3$ </td><td> $6 0 . 5 \pm 0 . 3$ </td><td> $6 4 . 0 \pm 0 . 4$ </td><td> $6 9 . 9 \pm 0 . 6$ </td></tr><tr><td> $\mathrm { O n e \ L a y e r + M e a n + F i x e d \ M a s k }$ </td><td> $7 1 . 5 \pm 0 . 3$ </td><td> $7 4 . 1 \pm 0 . 4$ </td><td> $6 8 . 4 \pm 0 . 7$ </td><td> $7 0 . 0 \pm 2 . 1$ </td><td> $7 1 . 0 \pm 0 . 3$ </td><td> $7 0 . 8 \pm 0 . 6$ </td></tr><tr><td> $\mathrm { O n e \ L a y e r + M e a n + L e a r n e d \ M a s k }$ </td><td> $7 1 . 2 \pm 0 . 4$ </td><td> $7 4 . 2 \pm 0 . 2$ </td><td> $6 8 . 4 \pm 0 . 7$ </td><td> $7 0 . 1 \pm 1 . 7$ </td><td> $7 0 . 9 \pm 0 . 3$ </td><td> $7 1 . 5 \pm 0 . 6$ </td></tr><tr><td> $\mathrm { O n e \ L a y e r + A t t e n t i o n + F i x e d \ M a s k }$ </td><td> $7 1 . 4 \pm 0 . 2$ </td><td> $7 4 . 0 \pm 0 . 4$ </td><td> $6 7 . 6 \pm 0 . 4$ </td><td> $6 8 . 5 \pm { 1 . 8 }$ </td><td> $7 1 . 0 \pm 0 . 3$ </td><td> $7 1 . 2 \pm 0 . 5$ </td></tr><tr><td> $\mathrm { O n e ~ L a y e r + A t t e n t i o n + L e a r n e d ~ M a s k }$ </td><td> $7 1 . 4 \pm 0 . 1$ </td><td> $7 4 . 3 \pm 0 . 2$ </td><td> $6 8 . 4 \pm 0 . 4$ </td><td> $6 8 . 4 \pm { 1 . 7 }$ </td><td> $7 1 . 2 \pm 0 . 2$ </td><td> ${ \bf 7 2 . 0 \pm 0 . 5 }$ </td></tr><tr><td> $\mathrm { M L P + M e a n + F i x e d M a s k }$ </td><td> $7 2 . 7 \pm 0 . 2$ </td><td> ${ \bf 7 6 . 9 \pm 0 . 4 }$ </td><td> ${ \bf 6 8 . 6 \pm 0 . 4 }$ </td><td> ${ \bf 7 0 . 3 \pm 0 . 9 }$ </td><td> $7 0 . 3 \pm 0 . 2$ </td><td> $6 9 . 1 \pm 0 . 9$ </td></tr><tr><td> $\mathrm { M L P + M e a n + L e a r n e d M a s k }$ </td><td> $7 2 . 4 \pm 0 . 2$ </td><td> $7 6 . 4 \pm 0 . 4$ </td><td> ${ \bf 6 8 . 6 \pm 0 . 4 }$ </td><td> $7 0 . 1 \pm 0 . 4$ </td><td> $7 0 . 2 \pm 0 . 1$ </td><td> $7 0 . 2 \pm 0 . 8$ </td></tr><tr><td> $\mathrm { M L P + A t t e n t i o n + F i x e d M a s k }$ </td><td> $7 2 . 7 \pm 0 . 1$ </td><td> $7 6 . 7 \pm 0 . 4$ </td><td> $6 8 . 4 \pm 0 . 3$ </td><td> $6 9 . 8 \pm 0 . 5$ </td><td> $7 0 . 2 \pm 0 . 3$ </td><td> $6 9 . 8 \pm 0 . 6$ </td></tr><tr><td> $\mathrm { M L P } + \mathrm { A t t e n t i o n } + \mathrm { L e a r n e d } \mathrm { M a s k }$ </td><td> $7 1 . 2 \pm 0 . 9$ </td><td> $7 3 . 3 \pm 1 . 9$ </td><td> $6 8 . 4 \pm 0 . 4$ </td><td> $6 9 . 4 \pm 0 . 7$ </td><td> $7 0 . 2 \pm 0 . 5$ </td><td> $7 0 . 5 \pm 0 . 7$ </td></tr></table>

Table 1. Macro-average performance per triplet type and run of all configurations across the Jamendo and MTT datasets. Values are reported as mean ± run standard deviation, in percentages. Bold marks the best mean values.

## 3.2 Datasets and Data Curation

We conduct evaluations across two standard music information retrieval datasets. The first, MTG-Jamendo [11], is used for both training and evaluation purposes. We utilize the raw\_30s preprocessing, in which each clip has a duration of at least 30 seconds, containing 55,701 tracks in total. Data is partitioned into training, evaluation, and test sets. Split partitions are strictly grouped by artist identity across over 3,500 artists to prevent data leakage.

The second dataset, MagnaTagATune (MTT) <sup>1</sup> [12], comprises 25,863 clips with noisy raw tag annotations. We perform text normalization to unify morphological variants <sup>2</sup> (e.g., merging “clasical”, “classical”, and “classic”) and manually group the top tags into genre, instrument, and mood categories. As MTT provides no standard training or validation split for our setup, we use it only as a heldout test set to assess generalization beyond MTG-Jamendo.

## 3.3 Triplet Mining and Optimization

Training and evaluating the masked embeddings described above requires constructing triplets $( x ^ { a } , x ^ { p } , x ^ { n } )$ for each similarity condition $c \in { \mathcal { C } } .$ Across both training and evaluation, all triplets satisfy a unified set of criteria: the anchor $x ^ { a }$ and positive $x ^ { p }$ share at least one tag in c, whereas negatives $x ^ { n }$ must possess at least one tag in $c ,$ share zero tags in c with $x ^ { a }$ , and originate from an artist distinct from both $x ^ { a }$ and $x ^ { p }$ . For deterministic and reproducible testing, these triplets are generated offline per attribute c. During training, they are instead mined dynamically using a semihard strategy [8] to adapt to the evolving embedding space.

## 3.4 Optimization and Implementation Details

All models are trained end-to-end using the Adam optimizer with a learning rate of $1 0 ^ { - 4 }$ , a batch size of 128, and cosine annealing scheduling. Training proceeds until convergence, monitored via early stopping on validation loss with a patience of 20 epochs and an improvement threshold of $\delta = 0$ . The loss function hyper-parameters are set in all experiments to $\lambda _ { \mathrm { m a s k } } = 1 0 ^ { - 5 } , \lambda _ { \mathrm { o r t h o } } = 1 0 ^ { - 3 }$ , and $\lambda _ { \mathrm { e m b } } = 1 0 ^ { - 5 }$ , with a triplet margin of $\Delta = 0 . 1$

To systematically evaluate the framework, we test a complete grid combining every backbone (CLAP, MuQ, MERT), temporal pooling method (Mean, Attention), adaptation module (Identity, One Layer, MLP), and masking regime (Fixed, Learned). Each configuration is executed across five independent seeds (0–4). For Identity configuration, only Learned Masks apply.

## 3.5 Evaluation Protocol and Baselines

To evaluate retrieval performance, we use Triplet Accuracy separately for each similarity condition, yielding three triplet types $c ^ { \prime } .$ We denote the resulting accuracy by $\boldsymbol { \mathcal { A } } ( \boldsymbol { e } , \boldsymbol { c } ^ { \prime } )$ , where e is an embedding vector. Given a fixed combination θ of backbone model, pooling method, adaptation module, and masking regime, we define the Effect E of a mask $m _ { c }$ on triplets of type $c ^ { \prime }$ as the change in triplet accuracy relative to the unmasked representation xˆ:

$$
\mathcal { E } ( \theta , m _ { c } , c ^ { \prime } ) = \boldsymbol { \mathcal { A } } ( e _ { c } , c ^ { \prime } ) - \boldsymbol { \mathcal { A } } ( \hat { \mathbf { x } } , c ^ { \prime } ) .\tag{7}
$$

A well-disentangled mask should have a positive effect on its target condition $\left( c ^ { \prime } \ = \ c \right)$ and a negative effect on non-matching conditions. Therefore, we define the Specificity S score to measure whether a mask $m _ { c }$ selectively improves retrieval for its target condition c while degrading retrieval for non-matching conditions in C:

$$
\mathcal { S } ( \theta , m _ { c } , \mathcal { C } ) = \mathcal { E } ( \theta , m _ { c } , c ) - \sum _ { c ^ { \prime \prime } \in \mathcal { C } \backslash \{ c \} } \mathcal { E } ( \theta , m _ { c } , c ^ { \prime \prime } ) .\tag{8}
$$

## 4. RESULTS

Table 1 reports macro-averaged triplet accuracy across the three triplet types and five runs. We also report the baseline performance, given by the unmasked, mean-pooled embedding of each backbone. Performance increases most for MERT and MuQ, suggesting that these models are less effective at direct similarity retrieval than CLAP, which shows smaller improvements. The Identity module, which applies no adaptation, generally yields little to no improvement and can even underperform the baseline, while the One Layer and MLP modules yield substantially larger gains. Thus, post-hoc disentanglement training can improve general similarity retrieval performance. However, neither the choice of adaptation module, pooling method, nor masking regime shows a consistent effect. These findings highlight the importance of adding trainable adaptation parameters and indicate that masking alone is insufficient to reshape the representation for improved retrieval.

<table><tr><td></td><td colspan="2">CLAP</td><td colspan="2">MERT</td><td colspan="2">MuQ</td></tr><tr><td>Configuration</td><td>Jamendo</td><td>MTT</td><td>Jamendo</td><td>MTT</td><td>Jamendo</td><td>MTT</td></tr><tr><td>Identity + Mean + Learned Mask</td><td> $2 . 1 4 \pm 0 . 3 0$ </td><td> $2 . 7 1 \pm 0 . 4 2$ </td><td> $- 0 . 0 3 \pm 0 . 1 5$ </td><td> $0 . 0 7 \pm 0 . 3 7$ </td><td> $0 . 4 9 \pm 0 . 8 1$ </td><td> $0 . 3 7 \pm 1 . 0 7$ </td></tr><tr><td>Identity + Attention + Learned Mask</td><td> $0 . 3 5 \pm 0 . 3 1$ </td><td> $0 . 2 4 \pm 0 . 3 6$ </td><td> $- 0 . 0 3 \pm 0 . 2 1$ </td><td> $0 . 2 4 \pm 0 . 4 7$ </td><td> $- 0 . 1 3 \pm 0 . 3 4$ </td><td> $0 . 5 7 \pm 0 . 8 4$ </td></tr><tr><td>One  ${ \mathrm { L a y e r + M e a n + F i x e d M a s k } }$ </td><td> $7 . 3 7 \pm 0 . 3 1$ </td><td> $4 . 8 4 \pm 0 . 4 0$ </td><td> $4 . 8 3 \pm 0 . 7 0$ </td><td> $5 . 7 4 \pm 0 . 7 6$ </td><td> $\mathbf { 7 . 6 9 \pm 0 . 3 0 }$ </td><td> $3 . 5 1 \pm 0 . 7 8$ </td></tr><tr><td>One  $\operatorname { L a y e r } + \operatorname { M e a n } + \operatorname { L e a r n e d } \operatorname { M a s k }$ </td><td> $7 . 0 2 \pm 0 . 3 2$ </td><td> $4 . 6 0 \pm 0 . 2 0$ </td><td> ${ \bf 5 . 0 7 \pm 0 . 4 5 }$ </td><td> $5 . 7 1 \pm 0 . 4 1$ </td><td> $7 . 1 2 \pm 0 . 4 1$ </td><td> $3 . 8 1 \pm 0 . 4 0$ </td></tr><tr><td>One  $\mathrm { L a y e r + A t t e n t i o n + F i x e d M a s k }$ </td><td> ${ \bf 7 . 4 1 \pm 0 . 2 1 }$ </td><td> $5 . 3 0 \pm 0 . 4 9$ </td><td> $4 . 1 3 \pm 1 . 1 4$ </td><td> $4 . 0 7 \pm 0 . 2 9$ </td><td> $6 . 9 7 \pm 0 . 3 3$ </td><td> $3 . 5 9 \pm 0 . 4 3$ </td></tr><tr><td>One Layer + Attention + Learned Mask</td><td> $7 . 1 2 \pm 0 . 3 4$ </td><td> $4 . 8 2 \pm 0 . 2 4$ </td><td> $5 . 0 4 \pm 0 . 6 5$ </td><td> $4 . 7 1 \pm 0 . 4 0$ </td><td> $6 . 7 7 \pm 0 . 2 2$ </td><td> $3 . 8 4 \pm 0 . 3 2$ </td></tr><tr><td> $\mathrm { M L P + M e a n + F i x e d M a s k }$ </td><td> $4 . 8 8 \pm 0 . 4 1$ </td><td> ${ \bf 6 . 0 3 \pm 0 . 3 3 }$ </td><td> $3 . 3 1 \pm 1 . 3 5$ </td><td> $4 . 2 1 \pm 2 . 0 1$ </td><td> $7 . 1 1 \pm 0 . 0 5$ </td><td> $4 . 6 6 \pm 0 . 5 0$ </td></tr><tr><td> $\mathrm { M L P + M e a n + L e a r n e d M a s k }$ </td><td> $3 . 4 6 \pm 0 . 5 3$ </td><td> $4 . 2 7 \pm 0 . 4 8$ </td><td> $4 . 5 5 \pm 0 . 1 9$ </td><td> ${ \bf 6 . 2 5 \pm 0 . 4 1 }$ </td><td> $6 . 5 6 \pm 0 . 3 1$ </td><td> $5 . 2 7 \pm 0 . 5 4$ </td></tr><tr><td> $\mathrm { M L P + A t t e n t i o n + F i x e d M a s k }$ </td><td> $4 . 7 0 \pm 0 . 2 7$ </td><td> $5 . 7 7 \pm 0 . 4 3$ </td><td> $2 . 4 0 \pm 0 . 3 1$ </td><td> $2 . 8 4 \pm 0 . 7 1$ </td><td> $7 . 0 8 \pm 0 . 1 8$ </td><td> ${ \bf 5 . 9 3 \pm 0 . 5 9 }$ </td></tr><tr><td> $\mathrm { M L P + A t t e n t i o n + L e a r n e d M a s k }$ </td><td> $6 . 2 9 \pm 1 . 2 4$ </td><td> $4 . 5 9 \pm 0 . 4 0$ </td><td> $4 . 5 4 \pm 0 . 2 3$ </td><td> $5 . 8 3 \pm 0 . 2 6$ </td><td> $6 . 6 4 \pm 0 . 2 5$ </td><td> $4 . 9 3 \pm 0 . 4 2$ </td></tr></table>

Table 2. Average Specificity performance of all masks and runs of the evaluated configurations across the Jamendo and MTT datasets. Values are reported as $\mathrm { m e a n } \pm \mathrm { r u n }$ standard deviation, in percentages. Bold marks the best mean values.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Mask</td><td colspan="3">Jamendo</td><td colspan="3">MTT</td></tr><tr><td>Inst.</td><td>Genre</td><td>Mood</td><td>Inst.</td><td>Genre</td><td>Mood</td></tr><tr><td rowspan="3">CLAP</td><td>None</td><td>73.2</td><td>75.5</td><td>66.4</td><td>73.7</td><td>86.2</td><td>64.8</td></tr><tr><td>Inst.</td><td>74.1 ↑</td><td>71.3↓</td><td>63.7↓</td><td>73.9↑</td><td>81.0↓</td><td>58.2↓</td></tr><tr><td>Genre Mood</td><td>71.6↓ 65.6↓</td><td>76.8↑ 69.6↓</td><td>65.2↓ 67.2↑</td><td>71.3↓ 70.1↓</td><td>86.5↑ 81.3↓</td><td>63.5↓ 70.3↑</td></tr><tr><td rowspan="4">MERT</td><td>None</td><td>69.2</td><td>71.7</td><td>64.2</td><td>65.8</td><td>79.6</td><td>63.6</td></tr><tr><td>Inst.</td><td>69.8 ↑</td><td>68.0↓</td><td>61.5↓</td><td>64.9↓</td><td>75.7↓</td><td>59.2↓</td></tr><tr><td>Genre</td><td>69.0↓</td><td>72.2↑</td><td>63.3↓</td><td>64.8↓</td><td>80.0 ↑</td><td>61.5↓</td></tr><tr><td>Mood</td><td>63.4↓</td><td>66.6↓</td><td>64.0↓</td><td>62.4↓</td><td>73.0↓</td><td>66.0↑</td></tr><tr><td rowspan="4">MuQ</td><td>None</td><td>68.6</td><td>72.9</td><td>66.8</td><td>65.4</td><td>78.0</td><td>64.5</td></tr><tr><td>Inst.</td><td>70.0↑</td><td>65.3↓</td><td>60.4↓</td><td>65.4 =</td><td>71.7↓</td><td>60.1 ↓</td></tr><tr><td>Genre</td><td>67.2↓</td><td>75.6↑</td><td>64.5↓</td><td>62.9↓</td><td>78.4↑</td><td>60.6↓</td></tr><tr><td>Mood</td><td>58.7↓</td><td>63.1↓</td><td>65.3↓</td><td>61.3↓</td><td>69.8↓</td><td>63.4↓</td></tr></table>

Table 3. Performance of the MLP configuration with Mean Pooling and Fixed Masks across the Jamendo and MTT datasets. Values are reported as percentages for instrument, genre, and mood retrieval. Bold values indicate the highest performance for each metric and backbone. Arrows indicate increases (↑) or decreases (↓) relative to the corresponding None-mask configuration.

To evaluate disentanglement, Table 2 reports the average Specificity score for each evaluated configuration, calculated by first averaging across the three masks and then across the five independent runs. Again, trained adaptation modules perform best, whereas further architectural design choices show inconclusive results and depend heavily on both the chosen backbone and evaluation dataset. This indicates that even a minimal number of trainable parameters is sufficient, yet necessary, to achieve both post-hoc disentanglement and retrieval performance gains.

We further examine triplet accuracy for each mask and triplet type in Table 3, considering only the MLP adaptation module with mean pooling and fixed masks. Triplet accuracy by type shows that genre is consistently the easiest to retrieve, while mood is the hardest across backbones and datasets. Analysis by mask shows that performance is very similar across masks for every triplet type. Since the fixed masks retain only one third of the embedding through a disjoint partition, these results suggest that the pretrained latent space is highly entangled, with attributerelevant information distributed redundantly across dimensions rather than confined to specific ones. Moreover, the largest performance effects come from decreases on nonmatching triplet types rather than gains on the matching one, suggesting that masking primarily suppresses interfering information rather than injecting new signal. This supports our central hypothesis that post-hoc disentanglement of pretrained backbones is achievable by reorganizing existing representations without backbone retraining.

## 5. CONCLUSION

In this work, we present a post-hoc strategy to disentangle representations from frozen audio foundation models (CLAP, MuQ, and MERT) for attribute-conditioned music similarity retrieval using Conditional Similarity Networks. We evaluate the addition of trainable parameters to adapt the embeddings and test different pooling methods and masking regimes across two datasets. We find that the original embeddings are highly entangled and redundant, allowing high triplet accuracy across all conditions using any third of the representation. We show that introducing even minimal adaptation capacity improves both triplet accuracy and disentanglement, whereas direct mask training without additional parameters performs poorly. Other architectural variations yield inconclusive results. Specifically, these disentanglement gains arise primarily from suppressing cross-attribute interference rather than from improving performance on the corresponding similarity condition. Overall, our findings show that post-hoc disentanglement training is a promising strategy for effectively adapting frozen foundation models to attribute-based music retrieval without backbone retraining.

## 6. ACKNOWLEDGMENTS

This study was financed, in part, by the São Paulo Research Foundation (FAPESP), Brasil. Process Number 2025/07753-1 and Process Number 2024/13907-9.

## 7. REFERENCES

[1] Y. Wu, K. Chen, T. Zhang, Y. Hui, M. Nezhurina, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” 2024. [Online]. Available: https://arxiv.org/abs/2211.06687

[2] H. Zhu, Y. Zhou, H. Chen, J. Yu, Z. Ma, R. Gu, Y. Luo, W. Tan, and X. Chen, “Muq: Self-supervised music representation learning with mel residual vector quantization,” IEEE Transactions on Audio, Speech and Language Processing, 2025.

[3] Y. Li, R. Yuan, G. Zhang, Y. Ma, X. Chen, H. Yin, C. Lin, A. Ragni, E. Benetos, N. Gyenge, R. Dannenberg, R. Liu, W. Chen, G. Xia, Y. Shi, W. Huang, Y. Guo, and J. Fu, “Mert: Acoustic music understanding model with large-scale self-supervised training,” 2023.

[4] C. Papaioannou, E. Benetos, and A. Potamianos, “Cross-cultural music similarity: Bridging human perception, signal processing, and foundation models,” Transactions ofthe International Societyfor Music Information Retrieval, vol. 9, no. 1, pp. 347–368, 2026.

[5] D. Afchar, A. Melchiorre, M. Schedl, R. Hennequin, E. Epure, and M. Moussallam, “Explainability in music recommender systems,” AI Magazine, vol. 43, no. 2, pp. 190–208, 2022.

[6] J. Lee, N. J. Bryan, J. Salamon, Z. Jin, and J. Nam, “Disentangled multidimensional metric learning for music similarity,” in ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp. 6–10.

[7] A. Veit, S. Belongie, and T. Karaletsos, “Conditional similarity networks,” 2017. [Online]. Available: https: //arxiv.org/abs/1603.07810

[8] S. Ribecky, J. Abeßer, and H. Lukashevich, “Multiinput architecture and disentangled representation learning for multi-dimensional modeling of music similarity,” arXiv preprint arXiv:2111.01710, 2021.

[9] M. Buisson, B. Mcfee, S. Essid, and H.-C. Crayencour, “Learning multi-level representations for hierarchical music structure analysis,” in International Society for Music Information Retrieval (ISMIR), 2022.

[10] Y. Hashizume, L. Li, A. Miyashita, and T. Toda, “Learning separated representations for instrument-based music similarity,” arXiv preprint arXiv:2503.17281, 2025.

[11] D. Bogdanov, M. Won, P. Tovstogan, A. Porter, and X. Serra, “The mtg-jamendo dataset for automatic music tagging,” in Machine Learning for Music Discovery Workshop, International Conference on Machine Learning (ICML 2019), Long Beach, CA, United States, 2019. [Online]. Available: http: //hdl.handle.net/10230/42015

[12] E. Law, K. West, M. I. Mandel, M. Bay, and J. S. Downie, “Evaluation of algorithms using games: The case of music tagging.” in ISMIR, 2009, pp. 387–392.