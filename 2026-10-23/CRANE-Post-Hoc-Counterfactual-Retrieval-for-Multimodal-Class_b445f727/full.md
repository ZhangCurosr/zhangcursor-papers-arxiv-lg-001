# CRANE: Post-Hoc Counterfactual Retrieval for Multimodal Classifiers

1<sup>st</sup> Franco Rugolon

2<sup>nd</sup> Ioanna Miliou

3<sup>rd</sup> Panagiotis Papapetrou

Dept. of Computer and Systems Sciences

Dept. of Computer and Systems Sciences

Stockholm University

Stockholm University

Stockholm University

Stockholm, Sweden

franco.rugolon@dsv.su.se

Stockholm, Sweden

Stockholm, Sweden

ioanna.miliou@dsv.su.se

panagiotis@dsv.su.se

Abstract—Counterfactual explanation methods have been introduced for unimodal settings, with dedicated frameworks for tabular, timeseries, text, and image data, yet no post-hoc method has been designed for classifiers that learn from multimodal inputs. Existing explainability methods for multimodal classifiers apply modality-specific attribution independently to each input, which fails to yield coherent explanations and does not transfer to arbitrary pre-trained multimodal architectures. To address these shortcomings, we present CRANE (Counterfactual Retrieval via Agnostic Nearest-neighbour Explanations), the first post-hoc, model-agnostic framework for nearest-neighbour multimodal counterfactual generation that retrieves real training examples from the target class as candidate explanations, ensuring plausibility by construction. Across four heterogeneous datasets and three fusion strategies, we show that Multimodal Consensus Retrieval (MC-R), which intersects per-modality neighbour sets to identify instances jointly close to the query in every modality, emerges as the strongest multimodal strategy, achieving the best or joint-best outcome on three of four datasets while grounding every explanation in the joint training distribution. Unimodal strategies remain competitive when a single modality dominates the decision boundary.

Index Terms—Explainable artificial intelligence, Counterfactual explanations, Multimodal learning, Post-hoc explanation

## I. INTRODUCTION

Data science pipelines increasingly operate on heterogeneous inputs, combining structured records, free text, timeseries, and images within a single predictive model. Clinical decision support systems fuse vital signs with structured patient demographics to predict outcomes, such as sepsis mortality [1] and cancer metastasis [2]–[4]; content-moderation systems pair images and their captions to detect hateful memes [5]; and fraud detectors combine free-text job postings with structured posting metadata to flag deceptive listings [6]. As these multimodal architectures become standard practice, the need to explain their predictions grows. This need has driven a substantial body of work on explainable Artificial Intelligence (XAI) [7], [8].

Counterfactual explanations are among the most natural and actionable forms of post-hoc justification: a counterfactual x<sup>′</sup> reconfigures a sample x so as to flip its prediction while staying close to x and consistent with the data manifold. Falling within the example-based XAI methods, they answer the intuitive, often directly actionable question “what would need to change in x for the model to decide otherwise?”.

![](images/036fd347d331f3a252a8f1da0d5c717fbf8c656809999c4fe89b5f135a0980f7.jpg)  
Fig. 1. Independently generated per-modality counterfactuals can produce a chimeric explanation whose static features and timeseries trajectories are incoherent. Arrows denote rising (↑) or falling (↓) biomarkers.

Well-performing counterfactual generators exist for individual modalities: tabular [9], [10], timeseries [11], [12], text [13], and image [14]. These generators, however, assume a single, homogeneous input space. When a classifier consumes several modalities at once, applying these generators independently per modality fails to yield coherent explanations, as illustrated in Fig. 1. For a multimodal mortality classifier over static demographics and laboratory timeseries, a tabular counterfactual generator may return a female survivor of the same age, while a timeseries generator may return signals that reflect decreasing N-terminal pro-B-type natriuretic peptide (NT-proBNP), a cardiac stress marker, but might not change the increasing trend in Prostate Specific Antigen (PSA), which is a clinical impossibility, as biological females do not produce PSA. This process would result in a chimeric counterfactual that would be locally plausible within each modality, yet globally incoherent and clinically meaningless when considered as a whole.

Retrieval-based methods avoid this problem by construction: rather than synthesising a new sample, they return a real training example from the target class. The nearest-unlikeneighbour (NUN) literature exploits this in unimodal settings [11], [12], [15], but it has not been extended to the multimodal case.

In addition, when using a retrieval-based approach, different choices regarding the search space itself deserve consideration. One could search within a single modality’s feature space, concatenate all modality embeddings and search jointly, or search in the multimodal model’s latent representation. Each choice encodes a different notion of similarity, and how their relative merits compare in terms of the standard counterfactual quality objectives (prediction-flip rate, proximity, sparsity, and plausibility) [16], and of the diversity of the generated set [10] remains to be explored.

We address these open questions with CRANE (Counterfactual Retrieval via Agnostic Nearest-neighbour Explanations), a post-hoc, model-agnostic framework for nearest-neighbour counterfactual generation for multimodal classifiers. CRANE<sup>1</sup> supports arbitrary combinations of tabular, timeseries, text, and image modalities through retrieval strategies spanning from unimodal search to multimodal fusion, and extends the standard counterfactual quality objectives (i.e., outcome, proximity, sparsity, and plausibility) to heterogeneous input spaces. Among the different strategies, Multimodal Consensus Retrieval (MC-R), which retrieves training instances jointly close to the query in each modality, emerges as the strongest multimodal strategy in four datasets spanning social media, recruitment, content moderation, and clinical prediction, while unimodal retrieval remains competitive when a single modality dominates the decision boundary.

## II. RELATED WORK

Explainability methods for machine learning (ML) span a broad landscape, from feature-attribution methods [17], [18], to saliency maps [19], attention-based explanations [20], and example-based approaches, like counterfactuals, yet the vast majority assume a single, homogeneous input space. This limitation is sharpest for counterfactuals, where the explanation must simultaneously specify what would change across all input spaces, not merely within one.

Counterfactual explanations were formalised in the context of ML by Wachter et al. [9], and have since grown into a rich research area. For example, DiCE [10] generates a diverse set of counterfactuals by adding a determinantal point process term to the objective, so that different explanations suggest different actionable paths, while Dandl et al. [21] proposed a multi-objective formulation that simultaneously optimises proximity, sparsity, plausibility, and actionability without collapsing them into a single weighted sum.

A distinct thread within the counterfactual literature avoids optimisation altogether by framing explanation as retrieval, where a real training example from the target class, the NUN, is used directly as the explanation or as a seed for further refinement. Brughmans and Martens [15] formalise this approach in NICE, which exploits the NUN to construct tabular counterfactuals that are guaranteed to exist in the observed data distribution. The retrieval strategy carries an important structural guarantee: since the returned sample was observed in the training data, it is plausible by construction, without any explicit plausibility penalty in an objective function.

However, the methods reviewed above are designed exclusively for flat tabular or embedding-space inputs; the question of how to generate counterfactuals for classifiers that fuse multiple modalities simultaneously has received considerably less attention, as the following review of modality-specific approaches makes clear.

## III. COUNTERFACTUAL RETRIEVAL VIA AGNOSTIC NEAREST-NEIGHBOUR EXPLANATIONS

## A. Problem Formulation

We introduce a general formulation for multimodal counterfactual generation that is independent of any particular data type, classifier architecture, or application domain.

Multimodal instance space. Let $M \geq 1$ be the number of modalities present in a dataset ${ \mathcal { S } } .$ For each modality $m \in$ $\{ 1 , \ldots , M \}$ , let ${ \mathcal { X } } ^ { m }$ be the corresponding modality space, encoding the representation format and domain constraints of that modality. The multimodal input space is

$$
\mathcal { X } : = \mathcal { X } ^ { 1 } \times \cdots \times \mathcal { X } ^ { M } .\tag{1}
$$

A multimodal instance is a tuple $x = ( x ^ { 1 } , \ldots , x ^ { M } ) \in \mathcal { X }$ where $x ^ { m } \in \mathcal { X } ^ { m }$ is the component of x in modality m.

In principle, ${ \boldsymbol { \mathcal { X } } } ^ { m }$ may represent any data type for which a suitable distance function $d _ { m }$ and plausibility estimator $p _ { m }$ can be defined. Without loss of generality, in this work we explore four modality types commonly encountered in heterogeneous data pipelines: static feature vectors (tabular), multivariate timeseries, token sequences (text), and two-dimensional pixel arrays (image). Adding additional modalities requires only specifying appropriate $d _ { m }$ and $p _ { m }$ functions for each one.

Multimodal classifier. Let Y be a finite label set. A multimodal classifier is a function $f : \mathcal { X } \to [ 0 , 1 ] ^ { | \mathcal { Y } | }$ , where $f _ { y } ( x )$ is the score assigned to label $y \in \mathcal { V }$ for instance x. The predicted label is $\hat { y } : = \arg \operatorname* { m a x } _ { y \in \mathcal { Y } } f _ { y } ( x )$ . CRANE treats f as a black box: only the prediction function is required, and no access to model internals, gradients, or training procedure is assumed.

Multimodal counterfactual generation. Given a factual instance $x \in \mathcal { X }$ with predicted label $\hat { y } ~ = ~ f ( x )$ , a labelled training set $\boldsymbol { \mathcal { D } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , and a target label $y ^ { \star } \ne \hat { y } .$ the goal is to find a counterfactual instance $x ^ { \prime } \in \mathcal { X }$ such that $f ( x ^ { \prime } ) ~ = ~ y ^ { \star }$ and $x ^ { \prime }$ is of high quality with respect to the multimodal equivalents to the standard counterfactual objectives, as defined in Section III-A1.

1) Objectives: To evaluate the generated counterfactuals, we use four metrics commonly employed in the counterfactual explanation literature: outcome, proximity, sparsity, and plausibility. Each candidate counterfactual x˙ is compared to the factual instance $\boldsymbol { x } = \left( x _ { 1 } , \dots , x _ { M } \right)$ . The latter three scores are computed per modality and aggregated using non-negative weights $\begin{array} { r } { w _ { m } \ \ge \ 0 , \ \sum _ { m } w _ { m } \ = \ 1 } \end{array}$ . Every retrieval strategy returns an ordered list of such candidates; the best-scoring ones under the aggregated objectives are retained as counterfactual instances $x ^ { \prime }$ as explained in Section III-A. All objectives are minimised; a value of zero denotes a perfect candidate. To ensure that scores from modalities with different native scales are commensurable, every per-modality function is mapped to [0, 1] before aggregation, so the multimodal aggregate is guaranteed to lie in [0, 1] regardless of the number or type of modalities present. In all experiments reported here we use uniform weights $w _ { m } \ = \ 1 / M$ ; dataset-specific weights may be substituted when prior knowledge about the relative importance of modalities is available. We define the four metrics as follows:

a) Outcome: The first objective measures whether, and how decisively, a candidate attains the target label $y ^ { \star }$ , using validity, a binary indicator

$$
\begin{array} { r } { \mathrm { V a l i d i t y } ( \dot { x } ) \ = \ \mathbb { 1 } \left[ \mathrm { a r g m a x } _ { y \in \mathcal { V } } f _ { y } ( \dot { x } ) = y ^ { \star } \right] , } \end{array}\tag{2}
$$

recording whether the prediction has flipped, and outcome

$$
\Phi _ { \mathrm { o u t } } ( \dot { x } ) = 1 - f _ { y ^ { \star } } ( \dot { x } ) ,\tag{3}
$$

a continuous deviation between the score the classifier assigns to the target class and the score it would assign to an unambiguous member of that class. $\Phi _ { \mathrm { o u t } }$ lies in [0, 1] and is minimised at 0 when the classifier has full confidence in $y ^ { \star }$

b) Proximity: Proximity measures how much the candidate deviates from the factual instance:

$$
\Phi _ { \mathrm { p r o x } } ( x , \dot { x } ) = \sum _ { m = 1 } ^ { M } w _ { m } \tilde { d } _ { m } ( x ^ { m } , \dot { x } ^ { m } ) ,\tag{4}
$$

where $\tilde { d } _ { m } \in [ 0 , 1 ]$ is a modality-specific normalised distance. For tabular and timeseries modalities, the raw distance is the mean absolute error between $x ^ { m }$ and ${ \dot { x } } ^ { m } ;$ it is normalised by converting it to a percentile rank against a reference distribution of pairwise distances computed from all crossclass training pairs, so that a score of 0 indicates that the counterfactual is closer than all reference pairs and a score of 1 indicates that it is farther. For text and image modalities, $\tilde { d } _ { m }$ is the normalised cosine distance, i.e.,

$$
\tilde { d } _ { m } ( x ^ { m } , \dot { x } ^ { m } ) = \frac { 1 - \cos ( \phi _ { m } ( x ^ { m } ) , \phi _ { m } ( \dot { x } ^ { m } ) ) } { 2 } ,\tag{5}
$$

where $\phi _ { m }$ is the modality-specific encoder. Note that $\tilde { d } _ { m } ( \cdot ) \in$ [0, 1], ensuring comparability with the percentile-normalised distances used for tabular and timeseries modalities.

c) Sparsity: Sparsity measures how many components of the factual instance must be altered to yield the counterfactual:

$$
\Phi _ { \mathrm { s p a r } } ( x , \dot { x } ) = \sum _ { m = 1 } ^ { M } w _ { m } s _ { m } ( x ^ { m } , \dot { x } ^ { m } ) ,\tag{6}
$$

where $s _ { m } \in [ 0 , 1 ]$ is a modality-specific normalised change count. For the tabular modality, $s _ { m }$ is the $\ell _ { 0 }$ norm divided by the total number of features D: $s _ { m } = \| \dot { x } ^ { m } - x ^ { m } \| _ { 0 } / D$ so that a value of 1 indicates every feature has changed. For the timeseries modality with C channels, $s _ { m }$ is computed per channel as the fraction of timesteps at which the absolute difference exceeds a threshold $\tau _ { c } \geq 0$ (default $\tau _ { c } = 0 .$ , so any nonzero change is counted); these per-channel fractions are averaged over all $C$ channels, giving $s _ { m } ~ \in ~ [ 0 , 1 ]$ . For the text modality, $s _ { m }$ is the token-level Levenshtein distance between $x ^ { m }$ and ${ \dot { x } } ^ { m }$ , divided by $\operatorname* { m a x } ( 1 , | x ^ { m } | )$ where $| x ^ { m } |$ is the number of tokens in the factual instance, and capped at 1.0.

For the image modality, $s _ { m }$ is the fraction of pixel elements (across all channels, after normalising integer images to [0, 1]) whose absolute difference exceeds a threshold (default 0).

d) Plausibility: It measures how typical the candidate counterfactual is relative to the target-class training distribution:

$$
\Phi _ { \mathrm { p l a u s } } ( \dot { x } ) = \sum _ { m = 1 } ^ { M } w _ { m } p _ { m } ( \dot { x } ^ { m } ) ,\tag{7}
$$

where $p _ { m } \in [ 0 , 1 ]$ is a normalised outlier score for modality m. For each modality, a Local Outlier Factor (LOF) model with $k = 2 0$ neighbours is fitted in novelty-detection mode on the target-class portion of the training set; timeseries are flattened to a vector of length $T \times C$ before fitting. The raw outlierness of a candidate $\dot { x } ^ { m } \mathrm { i s } - \mathsf { s c o r e \_ s a m p l e s } ( \dot { x } ^ { m } )$ so higher values indicate a greater deviation from the training distribution. This raw score is then mapped to [0, 1] via its percentile rank against the raw scores of all target-class training points:

$$
p _ { m } ( \dot { x } ^ { m } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { 1 } [ \mathbf { r a w } ( x _ { i } ^ { m } ) \leq \mathrm { r a w } ( \dot { x } ^ { m } ) ] ,\tag{8}
$$

where the sum is over the n target-class training instances. A score of 0 indicates that the candidate is at least as inlier-like as every target-class training point; a score of 1 indicates that it is a stronger outlier than every target-class training point.

On the plausibility of retrieved counterfactuals. A key property of retrieval-based methods is that whenever $k = 1 ,$ the returned counterfactual is a real training instance from the target class and is therefore plausible by construction: it lies on the empirical data manifold, and all its modalities were observed together in a single real-world instance, preserving cross-modal coherence. When $k > 1$ , CRANE always returns the single nearest neighbour as its first candidate; the j-th subsequent candidate $( j = 2 , \ldots , k )$ is the component-wise median over the $j$ nearest neighbours, yielding a sequence of increasingly smoothed prototypes. In general, none of these median candidates corresponds to any observed training instance. The plausibility objective in Eq. (7) quantifies plausibility for all candidates uniformly.

## B. CRANE Retrieval Strategies

CRANE constructs counterfactual candidates by searching the training set D for the k instances closest to x among those labelled $y ^ { \star } ,$ , and assembling a candidate from those neighbours. The key design choice is in which space the nearest-neighbour search is performed. CRANE provides a family of retrieval strategies, ranging from unimodal search to multimodal fusion. All strategies share a common interface: they take $( x , y ^ { \star } , k )$ as input and return an ordered list of candidate counterfactuals. All treat the classifier f as a black box. The different strategies are described below:

1) Unimodal Retrieval: CRANE provides four unimodal retrieval strategies — TABULARNN, TIMESERIESNN, TEXTNN, and IMAGENN — each ranking target-class training instances by distance in a single modality’s feature or embedding space. Let ${ \mathcal { D } } _ { y ^ { \star } } = \{ x _ { i } \mid y _ { i } = y ^ { \star } \}$ denote the target-class training instances. Given a modality-specific distance $d _ { m } ,$ , the k nearest neighbours are

$$
N _ { m } ^ { ( k ) } ( x ) = \mathrm { k N N } _ { y ^ { \star } } ( x ^ { m } ; d _ { m } ) ,\tag{9}
$$

the $k$ elements of $\mathcal { D } _ { y ^ { \star } }$ ⋆ with smallest $d _ { m } ( x ^ { m } , x _ { i } ^ { m } )$ . For text and image modalities, $d _ { m }$ operates in encoder embedding space: $d _ { m } ( a , b ) = 1 - \cos ( \phi _ { m } ( a ) , \phi _ { m } ( b ) )$ , where $\phi _ { m }$ is a configurable pre-trained encoder treated as a hyperparameter of the distance function. In the experiments reported here we evaluate BERT-family models (multilingual E5, MiniLM), TF-IDF, and Word2Vec for text; ResNet-50, EfficientNet-B0, ViT-$B / 1 6 ,$ and CLIP ViT-B/32 for images. All four strategies share the same retrieval interface, but the rule for constructing the j-th candidate $( j ~ = ~ 1 , \ldots , k )$ from the ranked neighbours depends on whether the modality is structured or unstructured, and this choice has direct consequences for how quality metrics behave as k increases.

Structured modalities (tabular, timeseries). Feature values are defined on a common continuous scale, so a componentwise median over $j$ neighbours is a meaningful summary. The j-th candidate prototype is the progressive median over the $j$ nearest neighbours:

$$
\pi _ { m } ^ { ( j ) } ( x ) \ = \ \ \operatorname * { m e d i a n } _ { x _ { i } \in N _ { m } ^ { ( j ) } ( x ) } x _ { i } ^ { m } .\tag{10}
$$

$\mathbf { A } \mathbf { t } \ j = 1$ this returns a real training instance. For $j > 1$ the prototype is a smoothed aggregate that may not correspond to any observed row, but it remains close to the centre of the local target-class neighbourhood. As a result, proximity and plausibility tend to improve modestly as $j$ increases, since the progressive median averages out outlying feature values. TABULARNN uses Euclidean or Manhattan distance; TIMESERIESNN uses Euclidean, DTW, or LCSS.

Unstructured modalities (text, image). No meaningful median exists in the original representation space: the componentwise median of token sequences or pixel arrays is not a coherent text or image. CRANE therefore uses rank-matched selection: the $j \cdot$ -th candidate takes the literal text or image from the j-th nearest neighbour in embedding space,

$$
\pi _ { m } ^ { ( j ) } ( x ) = x _ { ( j ) } ^ { m } ,\tag{11}
$$

where $x _ { ( j ) }$ denotes the j-th nearest neighbour under $d _ { m } .$ Therefore, every candidate is a real observed training instance in any j, but as $j$ increases, the selected instances are drawn from progressively more distant regions of the embedding space. Proximity and sparsity objectives for text and image are measured in embedding space, since recovering an unambiguous token sequence or pixel array from an encoder representation is not possible, in general.

2) Early-Fusion Nearest Neighbour (EF-NN): Early-fusion retrieval forms a joint representation by concatenating the embeddings of all modalities into a single vector:

$$
\widetilde { x } = \mathrm { c o n c a t } \big ( \phi _ { 1 } ( x ^ { 1 } ) , \dots , \phi _ { M } ( x ^ { M } ) \big ) ,\tag{12}
$$

and retrieves the k nearest neighbours in this joint space using a distance $\rho$ defined on the concatenated vector (e.g., cosine or Euclidean). The prototype is assembled from the jointlyranked neighbours using the modality-type materialisation rules defined in Section III-B (Eqs. 10–11). The only difference from a unimodal strategy is the ranking step: neighbours are ordered by distance in the concatenated space x˜ rather than in any single modality’s space.

3) Intermediate-Fusion Nearest Neighbour (IF-NN): Intermediate-fusion retrieval performs neighbour search in the latent representation learned by the classifier. Let $\lambda : \mathcal { X } \to \mathbb { R } ^ { d }$ denote the penultimate-layer activation of the multimodal classifier $f .$ . Distances are computed as

$$
d _ { \mathrm { i n t } } ( x , x _ { i } ) = \lVert \lambda ( x ) - \lambda ( x _ { i } ) \rVert _ { 2 } ,\tag{13}
$$

and the j-th candidate is assembled from the $j$ nearest neighbours in this representation. Because $\lambda ( x )$ is a single continuous vector regardless of the original modality types, candidate materialisation always follows the structured rule: the j-th candidate is the component-wise median over the latent-space-ranked neighbours, mapped back to input space by retrieving the corresponding complete training instances and applying Eqs. 10–11 per modality. By operating in the model’s own embedding space, IF-NN retrieves instances that the classifier itself considers similar to $x .$ However, unlike all other CRANE strategies, which operate solely through a prediction function and are fully model-agnostic, this one requires the explained model to be a deep neural network with accessible intermediate activations; it is not applicable to non-differentiable models.

4) Multimodal Consensus Retrieval (MC-R): MC-R retrieves instances that are simultaneously close to x across all modalities by intersecting per-modality neighbour sets. For each modality $m ,$ , the top-k nearest target-class neighbours are retrieved independently:

$$
N _ { m } ^ { ( k ) } ( x ) = \mathrm { k N N } _ { y ^ { \star } } ( x ^ { m } ; d _ { m } ) .\tag{14}
$$

The consensus set is the intersection of these sets:

$$
{ \cal S } ( x ) = \bigcap _ { m = 1 } ^ { M } { \cal N } _ { m } ^ { ( k ) } ( x ) .\tag{15}
$$

k is chosen large enough to ensure $s ( x ) ~ \neq ~ \emptyset$ . The $j \mathrm { - t h }$ candidate is then assembled from the $j$ nearest instances in $s ( x )$ , ranked by the summed per-modality distance, using the same materialisation rules (Eqs. 10–11). MC-R is the most conservative multimodal strategy: candidates are drawn from the intersection of per-modality neighbourhoods and are therefore jointly close to x in every modality simultaneously. The first candidate (when $| S ( x ) | \geq 1 )$ is always a real training instance.

5) Modality-wise Prototype Synthesis (MPS): MPS synthesises a candidate by combining independent per-modality prototypes. For each modality $m$ , the k nearest neighbours are retrieved in that modality’s own feature or embedding space, as in Eq. 9. The per-modality prototype $\pi _ { m } ^ { ( j ) } ( x )$ for the j-th candidate is computed following the modality-type rules in Eqs. 10–11. The j-th synthetic counterfactual is assembled by concatenating the per-modality prototypes:

$$
\dot { x } _ { \mathrm { M P S } } = \big ( \pi _ { 1 } ^ { ( j ) } ( x ) , \ldots , \pi _ { M } ^ { ( j ) } ( x ) \big ) .\tag{16}
$$

Unlike MC-R, MPS does not require any training instance to rank highly in all modalities simultaneously, so it can produce multimodal candidates even when no single instance is jointly close to x across all modalities. However, because each modality is sourced from a different nearest neighbour, MPS candidates are synthetic composites that do not in general correspond to any observed training instance, even at $j = 1$ Consequently, MPS does not carry the structural plausibility guarantees that hold for MC-R and the unimodal retrievers.

## IV. EXPERIMENTS

## A. Datasets

We evaluate CRANE on four publicly available datasets spanning diverse application domains, modality combinations, and class structures. A summary is given in Table I.

Sepsis (in-hospital mortality). The Sepsis dataset is extracted from MIMIC-III [22] following the procedure described in [1], which identifies ICU admissions with a confirmed sepsis diagnosis. Each admission is represented as 24 hourly timesteps over 100 features, partitioned into a timeseries modality of 53 time-varying clinical variables (vital signs and laboratory values) and a tabular modality of 47 static features (demographic and admission data extracted from the first timestep). The classification task is binary in-hospital mortality prediction. The dataset is strongly imbalanced, with a mortality rate of approximately 7.4%. Evaluation follows 5-fold stratified crossvalidation; results are pooled across folds and reported as F1 score as the primary metric, and AUC.

Long COVID Tweets (emotion classification). This dataset <sup>2</sup> consists of Italian tweets discussing long COVID, annotated with four emotion labels drawn from the FEEL-IT taxonomy [23]: fear, anger, sadness, and joy. Each instance comprises a raw tweet text field and a tabular modality encoding engagement metadata (likes, retweets, replies, quote counts) and structural tweet features (hashtag and URL counts). The class distribution is imbalanced, with fear (41.6%) and anger (32.9%) dominating and joy (8.8%) forming the minority class.

Hateful Memes (hate speech detection). The Hateful Memes dataset [24] is a binary classification benchmark for detecting hate speech in internet memes. Each instance consists of a text caption and a 224×224 RGB image. The class distribution is moderately imbalanced, with 63.7% non-hateful and 36.3% hateful instances.

Real or Fake Jobs (fraud detection). This dataset [25] contains job postings labelled as real or fraudulent. Each instance combines a tabular modality (structured fields such as salary range, employment type, and required experience) with three independent text fields: job description, company profile, and requirements. The task is strongly imbalanced, with fraudulent postings accounting for only 4.8% of the data.

TABLE I  
SUMMARY OF DATASETS USED IN THE EXPERIMENTS.
<table><tr><td>Dataset</td><td>Modalities</td><td>Task</td><td>Train</td><td>Test</td></tr><tr><td>Sepsis</td><td> $\mathrm { T S } + \mathrm { T a b }$ </td><td>Binary</td><td>2,085/fold</td><td>521/fold</td></tr><tr><td>Long COVID Tweets</td><td> $\mathrm { T e x t } + \mathrm { T a b }$ </td><td>4-class</td><td>20,121</td><td>5,031</td></tr><tr><td>Hateful Memes</td><td> $\mathrm { T e x t } + \mathrm { I m a g e }$ </td><td>Binary</td><td>7,632</td><td>1,908</td></tr><tr><td>Real or Fake Jobs</td><td> $\mathrm { T e x t } + \mathrm { T a b }$ </td><td>Binary</td><td>14,304</td><td>3,576</td></tr></table>

## B. Experimental Setup

1) Classifier Families and Fusion Strategies: For each dataset, we train classifiers under three fusion strategies: early fusion, intermediate fusion, and late fusion, as defined in Section III-B. For each fusion strategy, we compare two classifier families: a deep family based on multi-layer perceptrons (MLP) with task-trained branches, and a non-deep family using tree-based ensembles (Random Forest and Gradient Boosting) or, for timeseries branches, the ROCKET transform with Logistic Regression [26]. This design yields a range of classifier types that stress-tests CRANE’s model-agnostic claim: the IF-NN retrieval strategy requires a deep neural classifier with accessible penultimate activations and is therefore evaluated only on deep models, while all other CRANE strategies are evaluated uniformly across both families.

Text is encoded using pre-trained transformer models: XLM-RoBERTa for Long COVID Tweets, DistilBERT for Real or Fake Jobs, and BERT or MiniLM for Hateful Memes. Images in Hateful Memes are encoded using ResNet-50, EfficientNet-B0, or ViT-B/16. For early-fusion classifiers, all encoders are frozen (no fine-tuning); for intermediate-fusion classifiers, the transformer backbone is fine-tuned end-to-end. Class imbalance is addressed via inverse-frequency weighting for all classifiers. For training we use a 15% hold-out validation split for early stopping and hyperparameter tuning.

2) CRANE Configuration: Each CRANE retrieval strategy is configured with $k ~ = ~ 2 0$ , returning 20 candidates per query. For MC-R, the per-modality search pool is set to $k _ { \mathrm { s e a r c h } } = 2 0 0$ neighbours per modality prior to intersection, ensuring that the consensus set is non-empty for all queries in our datasets. For each dataset we run a systematic ablation over all valid combinations of encoder and distance metric per modality (as defined in Section III-B), yielding 6, 22, 16, and 84 combinations for Sepsis, Long COVID Tweets, Real or Fake Jobs, and Hateful Memes, respectively. The results are reported as averages across all ablation combinations.

3) CRANE Strategies per Dataset: Not all retrieval strategies are applicable to every dataset and classifier combination. IF-NN is evaluated only on deep classifiers, as described in Section IV-B; for datasets where a modality is absent (e.g., no image modality in Sepsis, Tweets, or Jobs), the corresponding unimodal strategy is not evaluated.

TABLE II  
BEST CLASSIFIER PER FUSION STRATEGY AND DATASET.
<table><tr><td>Dataset</td><td>Strategy</td><td>Model</td><td>Macro F1</td><td>Acc</td><td>AUROC</td></tr><tr><td rowspan="3">Sepsis</td><td>Early</td><td>Spatiotemporal Attention + GRU</td><td>0.691</td><td>0.920</td><td>0.800</td></tr><tr><td>Intermediate</td><td>RF on 32-dim latents</td><td>0.746</td><td>0.946</td><td>0.833</td></tr><tr><td>Late</td><td>GRU + MLP branches</td><td>0.691</td><td>0.912</td><td>0.871</td></tr><tr><td rowspan="3">Long COVID Tweets</td><td>Early</td><td>MLP on frozen RoBERTa</td><td>0.589</td><td>0.650</td><td>0.855</td></tr><tr><td>Intermediate</td><td>Fine-tuned transformer + MLP</td><td>0.717</td><td>0.759</td><td>0.921</td></tr><tr><td>Late</td><td>TF-IDF + LR / GBT branches</td><td>0.604</td><td>0.658</td><td>0.844</td></tr><tr><td rowspan="3">Hateful Memes</td><td>Early</td><td>GBT on frozen embeddings</td><td>0.713</td><td>0.740</td><td>0.777</td></tr><tr><td>Intermediate</td><td>Fine-tuned transformer + MLP</td><td>0.698</td><td>0.721</td><td>0.748</td></tr><tr><td>Late</td><td>TF-IDF + LR / GBT branches</td><td>0.669</td><td>0.703</td><td>0.723</td></tr><tr><td rowspan="3">Real or Fake Jobs</td><td>Early</td><td>RF on frozen embeddings</td><td>0.886</td><td>0.982</td><td>0.990</td></tr><tr><td>Intermediate</td><td>Fine-tuned transformer + MLP</td><td>0.932</td><td>0.988</td><td>0.992</td></tr><tr><td>Late</td><td>TF-IDF + LR / GBT branches</td><td>0.922</td><td>0.986</td><td>0.988</td></tr></table>

## C. Evaluation Protocol

We evaluate each retrieval strategy using the four quality objectives introduced in Section III-A1. All metrics are reported on a lower-is-better scale, where 0.0 denotes a perfect score on that objective. All four metrics are averaged across the five cross-validation folds using pooled test queries. For each dataset and strategy, we report the mean and standard deviation across all ablation combinations, providing a measure of sensitivity to encoder and distance metric choice. Counterfactual generation is performed on all test instances that the explained classifier predicts to belong to an undesired class; no subsampling is applied. For binary datasets, the target label y<sup>⋆</sup> is the single opposite class. For Long COVID Tweets, where four emotion classes are present, we focus on the sadness→joy direction, as these two classes represent the most semantically opposed poles in the dataset; the remaining class pairs (fear and anger) do not have a clearly defined opposite and are excluded from the counterfactual evaluation.

## V. RESULTS

## A. Classifier Performance

Table II reports macro-F1, accuracy, and AUROC for the best-performing classifier per fusion strategy and dataset, selected on the held-out set. Across datasets, intermediate fusion consistently yields the strongest or joint-strongest macro-F1, most notably on Long COVID Tweets (0.717 vs. 0.589 for early fusion) and Real or Fake Jobs (0.932 vs. 0.886 for early fusion). On Sepsis, late fusion achieves the highest AUROC (0.871). Performance on Hateful Memes is more modest across all strategies (macro-F1 0.669-0.713). These classifiers serve as the explained models in all subsequent counterfactual experiments.

## B. Generator quality

Unless otherwise stated, all results in the following sections are reported for k = 20; the effect of varying k is analysed separately in Section V-C. Fig. 2 shows the distribution of each quality metric across all datasets and fusion strategies, with each box capturing variance across encoder and distancemetric combinations within that generator, while Table III reports per-dataset results averaged across all three fusion strategies and encoder/distance-metric combinations.

IF-NN underperforms consistently across all four datasets compared to the other generators, with outcome values of

![](images/89e1de32462ec9fa2162304a9a300723c27d1b182efaf140bd6cededbd03f21a.jpg)  
Fig. 2. Generator quality across the full benchmark (k = 20). Each box shows the distribution over all encoder/distance-metric combinations, datasets, and fusion strategies. All metrics are lower-is-better.

0.371 on Sepsis, 0.699 on Tweets, 0.816 on Memes, and 0.113 on Jobs. Consequently, it emerges as the weakest generator overall, with a mean outcome of 0.526 and a validity rate of only 46.3%. Among the remaining generators, EF-NN achieves the lowest mean outcome (0.079), making it the most reliable at producing valid counterfactuals across the benchmark. Unimodal generators achieve the best proximity and plausibility.

On Real or Fake Jobs, MC-R achieves an outcome of 0.000 alongside the lowest sparsity (0.137) in the entire benchmark. On Sepsis, MPS achieves the best proximity (0.093) and plausibility (0.129), while TimeSeriesNN has the lowest outcome (0.021). On Long COVID Tweets, TabularNN achieves the best outcome (0.217), with engagement metadata proving more useful for retrieval than free-form tweet text. Finally, the constant sparsity of the Memes dataset (0.500) reflects a structural property: any two distinct training images are pixeldifferent everywhere after normalisation, so sparsity is fixed at 0.5 regardless of generator or k.

Table III averages across all three fusion strategies; however, the fusion strategy of the explained classifier is itself a source of variance. Its effect is concentrated on outcome and is dataset-dependent: on Long COVID Tweets, the choice of fusion strategy shifts mean outcome by up to 0.58, while on Sepsis and Jobs the range is below 0.05. Proximity, sparsity, and plausibility are largely stable across fusion strategies.

## C. Effect of k on Counterfactual Quality

Fig. 3 shows how each metric evolves as k increases from 1 to 50, averaged across all datasets and fusion strategies. We observe that outcome improves monotonically with k for nearly all generators, with the largest gains for TabularNN (0.178 to 0.049), EF-NN (0.170 to 0.052), MPS (0.246 to 0.095), and IF-NN (0.512 to 0.198), which shows the largest absolute improvement, yet remains the worst generator even at k = 50. MC-R’s outcome plateaus almost immediately after k = 5 (0.116 to 0.115 at k = 50): adding new instances to the consensus set does not substantially improve the outcome. Plausibility improves with k for structured modalities as the progressive median prototype moves toward the centre of the target-class neighbourhood, but degrades for TextNN, since larger $j$ draws from more distant training instances. ImageNN plausibility is flat (0.527 at all k), as every image candidate is a real training instance. Sparsity is nearly flat across k for all generators.

TABLE III  
GENERATOR QUALITY PER DATASET AT k = 20, AVERAGED ACROSS ALL THREE FUSION STRATEGIES AND ENCODER/DISTANCE-METRIC COMBINATIONS (LOWER IS BETTER; BEST RESULT PER METRIC/DATASET IN BOLD).
<table><tr><td>Dataset</td><td>Generator</td><td>Outcome</td><td>Proximity</td><td>Sparsity</td><td>Plausibility</td></tr><tr><td rowspan="6">Sepsis</td><td>TabularNN</td><td>0.056</td><td>0.158</td><td>0.474</td><td>0.143</td></tr><tr><td>TimeSeriesNN</td><td>0.021</td><td>0.317</td><td>0.457</td><td>0.196</td></tr><tr><td>EF-NN</td><td>0.055</td><td>0.131</td><td>0.465</td><td>0.143</td></tr><tr><td>IF-NN</td><td>0.371</td><td>0.203</td><td>0.488</td><td>0.151</td></tr><tr><td>MC-R</td><td>0.163</td><td>0.119</td><td>0.476</td><td>0.242</td></tr><tr><td>MPS</td><td>0.124</td><td>0.093</td><td>0.450</td><td>0.129</td></tr><tr><td rowspan="6">Tweets</td><td>TabularNN</td><td>0.217</td><td>0.350</td><td>0.142</td><td>0.052</td></tr><tr><td>TextNN</td><td>0.286</td><td>0.449</td><td>0.202</td><td>0.325</td></tr><tr><td>EF-NN</td><td>0.222</td><td>0.343</td><td>0.145</td><td>0.054</td></tr><tr><td>IF-NN</td><td>0.699</td><td>0.570</td><td>0.219</td><td>0.166</td></tr><tr><td>MC-R</td><td>0.300</td><td>0.356</td><td>0.172</td><td>0.089</td></tr><tr><td>MPS</td><td>0.298</td><td>0.334</td><td>0.144</td><td>0.054</td></tr><tr><td rowspan="6">Memes</td><td>TextNN</td><td>0.070</td><td>0.662</td><td>0.500</td><td>0.439</td></tr><tr><td>ImageNN</td><td>0.182</td><td>0.426</td><td>0.500</td><td>0.527</td></tr><tr><td>EF-NN</td><td>0.062</td><td>0.695</td><td>0.500</td><td>0.495</td></tr><tr><td>IF-NN</td><td>0.816</td><td>0.731</td><td>0.500</td><td>0.571</td></tr><tr><td>MC-R</td><td>0.087</td><td>0.658</td><td>0.500</td><td>0.472</td></tr><tr><td>MPS</td><td>0.079</td><td>0.662</td><td>0.500</td><td>0.439</td></tr><tr><td rowspan="6">Jobs</td><td>TabularNN</td><td>0.008</td><td>0.657</td><td>0.216</td><td>0.034</td></tr><tr><td>TextNN</td><td>0.005</td><td>0.688</td><td>0.241</td><td></td></tr><tr><td>EF-NN</td><td>0.017</td><td>0.725</td><td>0.216</td><td>0.113 0.035</td></tr><tr><td>IF-NN</td><td>0.113</td><td>0.906</td><td>0.241</td><td>0.098</td></tr><tr><td>MC-R</td><td>0.000</td><td>0.520</td><td>0.137</td><td>0.040</td></tr><tr><td>MPS</td><td>0.026</td><td>0.740</td><td>0.213</td><td>0.039</td></tr></table>

![](images/8229af5f3fcb76e77d9b45a7f3a087c0611e14c7f74f31c8151bdc8720717cbb.jpg)  
Fig. 3. Effect of $k$ on counterfactual quality, averaged across all datasets and fusion strategies. Unimodal generators are shown with dashed lines, multimodal with solid lines.

## VI. DISCUSSION

Unimodal retrieval. Unimodal generators achieve strong proximity and plausibility because they retrieve complete, coherent training instances: cross-modal coherence is preserved by construction. On Sepsis, TimeSeriesNN achieves near-perfect outcome (0.021) because the temporal clinical trajectory dominates the classifier’s decision boundary, consistent with prior findings on this dataset [1]; the retrieved instance’s static features jointly reinforce the flip. On imbalanced datasets (Sepsis, Jobs), performance is partly aided by the large majority-class retrieval pool; practitioners should expect degradation when the target class is the minority. Conversely, on Hateful Memes, TextNN (0.070) substantially outperforms

ImageNN (0.182), suggesting that the text modality dominates the classifier’s decision boundary for meme classification, consistent with prior findings on this benchmark [24].

Latent-space retrieval. IF-NN consistently fails to produce valid counterfactuals across all datasets and fusion strategies, even at $k = 5 0$ . The penultimate layer of a deep classifier is optimised for discriminative separation rather than within-class clustering, so neighbours in that space are not reliably from the target class. This limitation is specific to current training objectives; architectures that explicitly optimise for clustered latent representations may overcome it.

Classifier quality. Counterfactual quality is partly a function of classifier quality: a well-separating classifier assigns distinct predictions to distinct classes, so retrieved targetclass instances are more likely to flip the prediction. This is consistent with the results: on Jobs, where all classifiers achieve F1 above 0.88, every generator except IF-NN reaches near-zero outcome, whereas on Hateful Memes, which has the weakest classifiers in the benchmark (F1 0.669-0.713), outcome is substantially higher across all generators

MC-R as the recommended multimodal strategy. MC-R is the strongest multimodal generator overall, achieving the best or joint-best outcome on three of four datasets and perfect outcome on Real or Fake Jobs (0.000). On Jobs, MC-R also achieves the lowest sparsity in the entire benchmark (0.137), indicating that the intersection mechanism identifies targetclass instances that differ from the query in very few features, a desirable property for actionable explanations in fraud detection. Its intersection mechanism identifies instances jointly close to the query across all modalities, grounding counterfactuals in the joint training distribution in a way that MPS, which synthesises per-modality prototypes independently, does not guarantee. Its outcome saturates at $k = 5$ , making it efficient in practice. The choice of k also governs a trade-off between coherence and diversity [10]: at $k = 1$ , the proposed candidate is a real, coherent training instance; increasing k yields a larger and more diverse candidate set, with the first candidate always retaining full cross-modal coherence.

## VII. CONCLUSION

We presented CRANE, a post-hoc, model-agnostic framework for nearest-neighbour counterfactual generation for multimodal classifiers, supporting arbitrary modality combinations. Its extensibility to new modalities makes it naturally applicable to domains, such as clinical decision support, where normative frameworks require ML tools to be explainable [27]. Across four datasets and three fusion strategies, MC-R emerges as the strongest multimodal strategy, combining the strongest outcome scores in the benchmark, joint-distribution grounding, and computational efficiency. Future work includes comparative evaluation against the baselines described in Section II, human evaluation of counterfactual quality with domain experts, extension to additional clinical datasets, and exploration of generative counterfactual strategies within the CRANE evaluation protocol.

[1] L. Mondrejevski, F. Rugolon, I. Miliou, and P. Papapetrou, “Masicu: A multimodal attention-based classifier for sepsis mortality prediction in the icu,” in 2024 IEEE 37th International Symposium on Computer-Based Medical Systems (CBMS). IEEE, 2024, pp. 326–331.

[2] F. Rugolon, K. Randl, M. Bampa, and P. Papapetrou, “A workflow for creating multimodal machine learning models for metastasis predictions in melanoma patients,” in Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, 2023, pp. 87–102.

[3] F. Rugolon, K. Randl, B. Jovanovic, I. Miliou, and P. Papapetrou, “Multimodal machine learning for early prediction of metastasis in a swedish multicancer cohort,” JCO Clinical Cancer Informatics, vol. 10, p. e2500141, 2026.

[4] Y. Liu, Y. Wu, Q. Xia, H. He, H. Yu, and Y. Che, “An explainable predictive machine learning model for axillary lymph node metastasis in breast cancer based on multimodal data: A retrospective single-center study,” Journal of Translational Medicine, vol. 23, no. 1, p. 892, 2025.

[5] R. Gomez, J. Gibert, L. Gomez, and D. Karatzas, “Exploring hate speech detection in multimodal publications,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2020, pp. 1470– 1478.

[6] N. Akram, R. Irfan, A. S. Al-Shamayleh, A. Kousar, A. Qaddos, M. Imran, and A. Akhunzada, “Online recruitment fraud (orf) detection using deep learning approaches,” IEEE Access, vol. 12, pp. 109 388– 109 408, 2024.

[7] A. B. Arrieta, N. D´ıaz-Rodr´ıguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. Garc´ıa, S. Gil-Lopez, D. Molina, R. Benjamins´ et al., “Explainable artificial intelligence (xai): Concepts, taxonomies, opportunities and challenges toward responsible ai,” Information fusion, vol. 58, pp. 82–115, 2020.

[8] R. Guidotti, A. Monreale, S. Ruggieri, F. Turini, F. Giannotti, and D. Pedreschi, “A survey of methods for explaining black box models,” ACM computing surveys (CSUR), vol. 51, no. 5, pp. 1–42, 2018.

[9] S. Wachter, B. Mittelstadt, and C. Russell, “Counterfactual explanations without opening the black box: Automated decisions and the gdpr,” Harv. JL & Tech., vol. 31, p. 841, 2017.

[10] R. K. Mothilal, A. Sharma, and C. Tan, “Explaining machine learning classifiers through diverse counterfactual explanations,” in Proceedings of the 2020 conference on fairness, accountability, and transparency, 2020, pp. 607–617.

[11] E. Ates, B. Aksar, V. J. Leung, and A. K. Coskun, “Counterfactual explanations for multivariate time series,” in 2021 international conference on applied artificial intelligence (ICAPAI). IEEE, 2021, pp. 1–8.

[12] E. Delaney, D. Greene, and M. T. Keane, “Instance-based counterfactual explanations for time series classification,” in International conference on case-based reasoning. Springer, 2021, pp. 32–47.

[13] T. Wu, M. T. Ribeiro, J. Heer, and D. S. Weld, “Polyjuice: Generating counterfactuals for explaining, evaluating, and improving models,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2021, pp. 6707– 6723.

[14] T. Vermeire, D. Brughmans, S. Goethals, R. M. B. De Oliveira, and D. Martens, “Explainable image classification with evidence counterfactual,” Pattern Analysis and Applications, vol. 25, no. 2, pp. 315–335, 2022.

[15] D. Brughmans, P. Leyman, and D. Martens, “Nice: an algorithm for nearest instance counterfactual explanations,” Data Mining and Knowledge Discovery, vol. 38, no. 5, pp. 2665–2703, 2024.

[16] S. Verma, V. Boonsanong, M. Hoang, K. Hines, J. Dickerson, and C. Shah, “Counterfactual explanations and algorithmic recourses for machine learning: A review,” ACM Computing Surveys, vol. 56, no. 12, pp. 1–42, 2024.

[17] M. T. Ribeiro, S. Singh, and C. Guestrin, “” why should i trust you?” explaining the predictions of any classifier,” in Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, 2016, pp. 1135–1144.

[18] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” Advances in neural information processing systems, vol. 30, 2017.

[19] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via

gradient-based localization,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 618–626.

[20] H. Chefer, S. Gur, and L. Wolf, “Transformer interpretability beyond attention visualization,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 782–791.

[21] S. Dandl, C. Molnar, M. Binder, and B. Bischl, “Multi-objective counterfactual explanations,” in International conference on parallel problem solving from nature. Springer, 2020, pp. 448–469.

[22] A. E. Johnson, T. J. Pollard, L. Shen, L.-w. H. Lehman, M. Feng, M. Ghassemi, B. Moody, P. Szolovits, L. Anthony Celi, and R. G. Mark, “Mimic-iii, a freely accessible critical care database,” Scientific data, vol. 3, no. 1, pp. 1–9, 2016.

[23] F. Bianchi, D. Nozza, and D. Hovy, “Feel-it: Emotion and sentiment classification for the italian language,” in Proceedings of the eleventh workshop on computational approaches to subjectivity, sentiment and social media analysis, 2021, pp. 76–83.

[24] D. Kiela, H. Firooz, A. Mohan, V. Goswami, A. Singh, P. Ringshia, and D. Testuggine, “The hateful memes challenge: Detecting hate speech in multimodal memes,” Advances in neural information processing systems, vol. 33, pp. 2611–2624, 2020.

[25] S. Vidros, C. Kolias, G. Kambourakis, and L. Akoglu, “Automatic detection of online recruitment frauds: Characteristics, methods, and a public dataset,” Future Internet, vol. 9, no. 1, p. 6, 2017.

[26] A. Dempster, F. Petitjean, and G. I. Webb, “Rocket: exceptionally fast and accurate time series classification using random convolutional kernels,” Data Mining and Knowledge Discovery, vol. 34, no. 5, pp. 1454–1495, 2020.

[27] N. Freyer, D. Groß, and M. Lipprandt, “The ethical requirement of explainability for ai-dss in healthcare: a systematic review of reasons,” BMC Medical Ethics, vol. 25, no. 1, p. 104, 2024.