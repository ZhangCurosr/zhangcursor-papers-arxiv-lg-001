# When Does Self-Supervised Pretraining Help Tabular Models? A Study of Label Scarcity and Missing Data

Sahand Mazrouei

Faculty of Mathematics and Computer Science, Kharazmi University

sahand.mazrue@khu.ac.ir

## Abstract

Self-supervised learning (SSL) has emerged as a promising approach for tabular data, yet its efficacy under extreme label scarcity and test-time missingness remains under-explored. In this paper, we evaluate a mask-and-recover SSL pretraining objective against training from scratch and classical baselines across 14 diverse classification tasks. First, while SSL outperforms training from scratch on average and remains competitive with state-of-the-art tree ensembles (achieving ∼0.8954 AUC vs. Random Forest’s 0.9015 at 10% labels), the SSL-vs-scratch gains exhibit high inter-task variance and lack significance (p = 0.626 at both 5% and 10% labels). Second, contrary to the hypothesis that missing-value imputation objectives universally benefit datasets with native missingness, SSL yields the most reliable improvements on clean datasets, while frequently degrading performance on datasets with high inherent missingness. Third, despite this training variance, SSL-pretrained models achieve a higher average AUC than scratch-trained models under both test-time missingness completely at random (MCAR) injection (+0.0245 AUC, positive on 11 of 14 tasks) and structured missingness shifts (MNAR, +0.0418 AUC, positive on 8 of 14 tasks), though neither difference remains statistically significant after Holm-Bonferroni correction for multiple comparisons (adjusted p = 0.118 and p = 0.518, respectively). Fourth, comparing our mask-and-recover objective against three established tabular SSL baselines (VIME, SCARF, SubTab) under an identical encoder architecture, we find no significant difference from any of them (adjusted p = 0.459, p = 1.000, p = 1.000), indicating our findings reflect general properties of tabular SSL rather than idiosyncrasies of one particular pretext task.

Keywords: Tabular Data, Self-Supervised Learning, Missing Data, Label Scarcity, Robustness

## 1 Introduction

Tabular data remains the ubiquitous format for storing heterogeneous information in domains ranging from healthcare to financial forecasting. Despite the dominance of deep learning in vision and natural language processing, tree-based ensembles such as Random Forest and Gradient Boosted Trees frequently outperform deep neural networks on tabular tasks. This performance gap widens in real-world scenarios characterized by two distinct challenges: label scarcity, where annotating examples is costly or slow, and feature missingness, where data is lost due to sensor failures, privacy restrictions, or irregular reporting.

Self-supervised learning (SSL) provides a compelling framework to address these limitations. By optimizing a reconstruction or contrastive objective on abundant unlabeled features, a deep tabular encoder can learn the underlying data manifold before fine-tuning on a small labeled subset. Recent tabular SSL architectures typically employ a “mask, corrupt, and recover” paradigm. However, the existing literature primarily benchmarks these methods on fully observed, clean datasets. The interaction between SSL reconstruction objectives and datasets that natively contain missing values— as well as the model’s robustness when missingness distributions shift at test time—remains an open empirical question.

In this work, we present a comprehensive evaluation of tabular SSL under realistic constraints. We compare an SSL-pretrained tabular encoder against the exact same architecture trained from scratch, as well as against classical machine learning baselines, across a curated suite of OpenML classification tasks. Our experimental design explicitly tests the limits of SSL by enforcing label scarcity fractions of 1%, 5%, 10%, and 20%, and by evaluating resilience to test-time missingness injection.

Our analysis moves beyond aggregate leaderboards to uncover the dataset-dependent dynamics of tabular SSL. Our key contributions are as follows:

• Benchmarking under Label Scarcity: We demonstrate that SSL pretraining is highly competitive with classical baselines, achieving average AUC-ROC scores within roughly 0.006 of Random Forest at the 10% label fraction despite far less inductive bias. However, we show that the delta between SSL and scratch training exhibits high inter-task variance and is not itself statistically significant.

• The Missing vs. Clean Paradox: We uncover a nuanced interaction between SSL and missing data. While reconstruction objectives inherently learn imputation, our results show that SSL yields the most consistent downstream classification improvements on clean datasets. On datasets with high native missingness, SSL performance is highly dataset-dependent.

• Robustness to Test-Time Degradation: We show that SSL-pretrained models retain higher accuracy than scratch-trained models under input degradation at inference time. Under Missing Completely At Random (MCAR) injection (+30% missingness) and Missing Not At Random (MNAR) structured shifts, SSL models degrade more gracefully than models trained from scratch, though this advantage does not reach significance once corrected for multiple comparisons across our four primary hypothesis tests.

• Ablation of Tabular SSL Components: We ablate the consistency loss, group masking, and mask-visibility across four native-missingness datasets. An initial single-seed comparison suggested a large, consistent effect of removing the consistency loss; however, once we re-ran each variant across three independent pretraining seeds—a stricter protocol than elsewhere in this paper—this effect collapsed to statistical noise (mean ∆AUC = +0.0031, p = 0.625). None of the three ablated components show a reliable, generalizable effect within our four-task sample, underscoring the risk of drawing conclusions from single-seed ablation comparisons in small-sample tabular SSL studies.

• Comparison Against Prior Tabular SSL Objectives: Using the identical encoder architecture, we compare our mask-and-recover objective against three established tabular SSL pretext tasks—VIME, SCARF, and SubTab—at 10% labels across all 14 tasks. None of the three baselines differ significantly from our method (Wilcoxon $p = 0 . 1 5 3 , p = 0 . 6 7 0$ , and $p = 0 . 5 0 2 \mathrm { { : } }$ ; all remain non-significant after Holm-Bonferroni correction), indicating that our objective is empirically competitive with, rather than distinguishable from, existing tabular SSL formulations under a controlled, architecture-matched comparison.

## 2 Related Work

## 2.1 Deep Learning for Tabular Data

While deep neural networks have revolutionized unstructured data domains, tree-based models such as XGBoost [1], LightGBM [2], and Random Forest [3] remain the dominant paradigm for tabular data; Grinsztajn et al. [7] attribute this persistent gap in part to deep models’ sensitivity to uninformative features and to inductive biases poorly suited to tabular data. Recent efforts to adapt deep learning to tabular datasets have introduced specialized architectures. TabNet [4] utilizes sequential attention for feature selection, FT-Transformer [5] applies multi-head self-attention to tabular inputs, and TabTransformer [9], SAINT [6], and Non-Parametric Transformers [10] extend attention-based modeling across rows, columns, or both. Separately, Gorishniy et al. [8] show that learned numerical-feature embeddings can substantially affect downstream performance independent of encoder architecture. We adopt a deliberately simpler dual-branch MLP encoder throughout this work (Section 4) so that differences in downstream results can be attributed to the pretraining objective rather than architectural capacity.

## 2.2 Self-Supervised Learning on Tabular Data

To alleviate the data-hunger of deep tabular models, Self-Supervised Learning (SSL) has been adapted from natural language processing and computer vision [17, 18]. VIME [12] uses a pretext task of recovering corrupted features and estimating mask vectors. SCARF [13] adapts contrastive learning by forming positive pairs through marginal distribution corruption. SubTab [14] divides tabular rows into multiple subsets and learns representations by reconstructing the full row from partial views.

## 2.3 Handling Missing Data in Machine Learning

Missing data is traditionally categorized into three mechanisms: Missing Completely At Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR) [19]. Classical approaches rely on statistical imputation (e.g., mean imputation, K-Nearest Neighbors [21], or MICE [22]) prior to model training, while more recent work learns the imputation mechanism directly from data, such as MissForest’s iterative random-forest-based scheme [24] and GAIN’s adversarial imputation framework [23]. Our classical baselines (Section 5) use simple statistical imputation rather than these learned alternatives; we return to this choice in Appendix C.

## 3 Background and Problem Setup

## 3.1 Tabular prediction with missing values

We consider supervised classification on a tabular dataset with N examples and d features. Each example includes numeric features $x _ { n }$ , categorical features $x _ { c } ,$ and a label y (available only for a

subset).

Missingness is represented via binary masks: $m _ { n } ^ { ( i ) } \in \{ 0 , 1 \} ^ { d _ { n } }$ for numeric features and $m _ { c } ^ { ( i ) } \in$ $\{ 0 , 1 \} ^ { d _ { c } }$ for categorical features of instance i, where 1 denotes a missing entry. To characterize the degradation of the datasets, we define the numeric missingness rate over all N training instances as:

$$
\mathrm { m i s s i n g \_ p c t } = \frac { \sum _ { i = 1 } ^ { N } \sum m _ { n } ^ { ( i ) } } { N d _ { n } } \times 1 0 0 \%
$$

## 3.2 Label scarcity protocol

We use OpenML’s official train/test split (fold 0); preprocessing and pretraining are fit on the training split only, never on test data. We sample labeled subsets at fractions {1%, 5%, 10%, 20%} of the training set. For each fraction, we run 3 seeds (0, 1, 2) controlling the labeled-subset sample and the fine-tuning trajectory, and report mean (and where relevant, standard deviation); SSL pretraining itself is run once per task and shared across these 3 seeds. This differs from the ablation study and SSL-baseline comparison (Sections 6.4, 6.6), where 3 independent pretraining seeds are used instead.

## 3.3 Metric: AUC-ROC

We evaluate performance using test AUC-ROC. AUC-ROC measures the probability that the model ranks a random positive example above a random negative example, aggregated across all decision thresholds. AUC-ROC ranges from 0.5 (random) to 1.0 (perfect). For the subset of our 14 tasks that are multiclass, we compute AUC-ROC using the one-vs-rest formulation with macro-averaging across classes; binary tasks use the standard two-class AUC-ROC.

## 4 Method

## 4.1 Model Architecture

Let a tabular dataset be denoted as $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , where each instance $x _ { i }$ consists of a numeric feature vector $x _ { n } \in \mathbb { R } ^ { d _ { n } }$ and a categorical feature vector $x _ { c } \in \mathbb { Z } ^ { d _ { c } }$ . To explicitly inform the model of missingness, we define binary mask vectors $m _ { n } \in \{ 0 , 1 \} ^ { d _ { n } }$ and $m _ { c } \in \{ 0 , 1 \} ^ { \bar { d } _ { c } }$ , where a value of 1 indicates a missing entry.

We process these inputs using a dual-branch Tabular Encoder. The numeric branch concatenates the features and their masks, passing them through a two-layer Multi-Layer Perceptron (MLP):

$$
h _ { n } = \mathrm { M L P _ { n u m } } \big ( [ x _ { n } \oplus m _ { n } ] \big )
$$

where ⊕ denotes concatenation, and the MLP applies Layer Normalization, ReLU activation, and Dropout (rate $p = 0 . 1 )$ to project the input to a hidden dimension of 128.

Simultaneously, the categorical features are processed via learnable embedding dictionaries $E _ { j } \in \mathbb { R } ^ { | V _ { j } | \times 3 2 }$ , where $| V _ { j } |$ is the vocabulary size of the j-th categorical feature. The embeddings are concatenated and processed:

$$
h _ { c } = \mathbf { M L P _ { c a t } } \left( \bigoplus _ { j = 1 } ^ { d _ { c } } E _ { j } ( x _ { c , j } ) \right)
$$

The final representation $h \in \mathbb { R } ^ { 1 2 8 }$ is obtained by fusing the numeric and categorical representations:

$$
h = { \bf M L P _ { f u s i o n } } ( [ h _ { n } \oplus h _ { c } ] )
$$

## 4.2 Self-Supervised Pretraining Objective

Our SSL pretraining employs a “mask, corrupt, recover” paradigm. During pretraining, we sample an artificial corruption mask m˜ from a Bernoulli distribution with parameter $\rho = 0 . 3$ . The corrupted input x˜ replaces masked numeric values with 0.0 (the scaled mean) and categorical values with a dedicated [MASK] token.

The network produces two augmented views, $\tilde { x } ^ { ( 1 ) }$ and $\tilde { x } ^ { ( 2 ) }$ . The total loss is defined as:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { S S L } } = \mathcal { L } _ { \mathrm { r e c o n } } ( \tilde { x } ^ { ( 1 ) } , x ) + \mathcal { L } _ { \mathrm { r e c o n } } ( \tilde { x } ^ { ( 2 ) } , x ) } \\ & { \quad \quad + \lambda \mathcal { L } _ { \mathrm { c o n s } } ( h ( \tilde { x } ^ { ( 1 ) } ) , h ( \tilde { x } ^ { ( 2 ) } ) ) } \end{array}
$$

The reconstruction loss $\mathcal { L } _ { \mathrm { r e c o n } }$ is partitioned by feature type. For numeric features, we compute the Mean Squared Error (MSE) strictly over the artificially masked indices:

$$
\mathcal { L } _ { \mathrm { { n u m } } } = \frac { \sum _ { j = 1 } ^ { d _ { n } } \tilde { m } _ { n , j } ( \hat { x } _ { n , j } - x _ { n , j } ) ^ { 2 } } { \sum _ { j = 1 } ^ { d _ { n } } \tilde { m } _ { n , j } + \epsilon }
$$

For categorical features, we compute the mean Cross-Entropy (CE) independently for each masked categorical feature $j$ (averaged over the instances in the batch where $j$ was masked), then average these per-feature losses across all $d _ { c }$ features that had at least one masked instance in the batch:

$$
\mathcal { L } _ { \mathrm { c a t } } = \frac { 1 } { | \mathcal { T } | } \sum _ { j \in \mathcal { I } } \frac { \sum _ { i = 1 } ^ { B } \tilde { m } _ { c , j } ^ { ( i ) } \mathbf { C } \mathbf { E } ( \hat { x } _ { c , j } ^ { ( i ) } , x _ { c , j } ^ { ( i ) } ) } { \sum _ { i = 1 } ^ { B } \tilde { m } _ { c , j } ^ { ( i ) } }
$$

where $B$ is the batch size and $\mathcal { I }$ is the set of categorical features masked at least once in the batch. This weights each feature equally, rather than each masked token equally. When a batch contains both masked numeric and masked categorical entries, $\mathcal { L } _ { \mathrm { r e c o n } } = 0 . 5 ( \mathcal { L } _ { \mathrm { n u m } } + \mathcal { L } _ { \mathrm { c a t } } )$ ; if only one type is present in a given view, $\mathcal { L } _ { \mathrm { r e c o n } }$ reduces to that term alone. Finally, $\mathcal { L } _ { \mathrm { c o n s } }$ enforces representation invariance via cosine distance:

$$
\mathcal { L } _ { \mathrm { c o n s } } = 1 - \frac { h ^ { ( 1 ) } \cdot h ^ { ( 2 ) } } { \| h ^ { ( 1 ) } \| _ { 2 } \| h ^ { ( 2 ) } \| _ { 2 } }
$$

## 4.3 Group Masking

We optionally mask groups of features (rather than independent features) to encourage learning dependencies. Feature groups are discovered by calculating the absolute Spearman correlation matrix of the numeric features, converting it to a distance matrix $( 1 - | \mathrm { c o r r } | )$ , and applying hierarchical clustering with average linkage. During pretraining, an entire correlated group is masked simultaneously with a probability of 0.5.

## 4.4 Fine-tuning

After SSL pretraining for 50 epochs on the unlabeled features, we fine-tune the model on the labeled subset for 30 epochs using the AdamW optimizer [25]. We employ a learning rate of $1 0 ^ { - 3 } .$ , a weight decay of $1 0 ^ { - 4 }$ , and a batch size of up to 128 (capped by the number of labeled examples available).

## 5 Experiments

## 5.1 Datasets

We evaluate our approach on 14 diverse OpenML [15] classification tasks. The datasets vary significantly in training size and native missingness rates.

## 5.2 Baselines

We compare our approach against:

• Scratch (No Pretrain): The exact same neural architecture trained only on the labeled data.

• Classical baselines (10% labels): Logistic Regression, Random Forest, XGBoost, and LightGBM.

## 5.3 Robustness evaluation

To measure robustness to missing data, we evaluate the models under three conditions:

1. Normal test performance (no additional corruption).

2. MCAR +30% missingness: We inject additional missing entries uniformly at random (MCAR) with a 30% probability into the available test features.

3. Missingness shift (MNAR): We apply a structured Missing Not At Random (MNAR) shift by masking all numeric values that exceed the 70th percentile of their respective column distributions.

## 5.4 Statistical testing

We compute per-task performance deltas $\Delta = \mathrm { A U C } _ { \mathrm { S S L } } - \mathrm { A U C } _ { \mathrm { s c r a t c h } }$ . To assess statistical significance across datasets, we report the mean $\Delta$ and apply the Wilcoxon signed-rank test, following standard practice for comparing classifiers across multiple independent datasets [20]. Our main experiments involve four primary hypothesis tests—label scarcity at 5% and 10%, MCAR robustness, and MNAR shift robustness—which we treat as a single family and correct using the Holm-Bonferroni procedure at $\alpha = 0 . 0 5$ . We report both raw and Holm-adjusted p-values throughout; our comparison against prior tabular SSL baselines (Section 6.6) constitutes a separate family of three tests, corrected independently.

## 6 Results

## 6.1 Main results under label scarcity

Figure 1 summarizes the average AUC-ROC across tasks. At 5% labels, SSL improves the average AUC by +0.0074 over scratch training. At 10% labels, the average improvement is +0.0055. A Wilcoxon signed-rank test indicates that neither aggregate gain reaches significance $( p = 0 . 6 2 5 7$ at both 5% and 10%; Holm-adjusted $p = 1 . 0 0 0 ) .$ <sup>1</sup> This confirms that SSL pretraining is not universally beneficial across all tabular manifolds. Notably, this shrinking advantage crosses zero by 20% labels: the average SSL AUC (0.9145) falls slightly below the scratch baseline (0.9173) at this fraction, indicating that pretraining’s benefit is concentrated in the label-scarce regime rather than persisting once more labeled data is available (full per-task, per-fraction results are in Appendix A).

![](images/6d6e7e0b6c41f8dd6fc567286f427427a03fe53f8433507ecc374999981ff0b1.jpg)  
Figure 1: Average test AUC-ROC across tasks for SSL-pretrained vs. scratch training.

## 6.2 The Missing vs. Clean Paradox

A core hypothesis of tabular SSL is that mask-and-recover objectives teach the model to natively handle missing values. However, Figure 2 illustrates a counter-intuitive finding: SSL yields the most reliable performance improvements on clean datasets. On datasets with high inherent missingness, the impact of SSL is highly variable, frequently degrading performance.

## 6.3 Robustness to Test-Time Degradation

Despite the variance in clean test settings, SSL-pretrained models achieve higher raw AUC than scratch training on average under test-time data degradation. Under MCAR +30% missingness (injected into both numeric and categorical features), SSL outperforms scratch by an average of +0.0245 AUC (raw $p = 0 . 0 2 9 5 ;$ Holm-adjusted $p = 0 . 1 1 8$ across our family of four primary tests). Under a more severe MNAR missingness shift, SSL achieves a mean gain of +0.0418 AUC (raw $p = 0 . 1 7 2 6 ;$ Holm-adjusted $p = 0 . 5 1 8 )$ . Neither comparison remains significant after correcting for multiple comparisons, though both show the same consistent direction and are the two largest effect sizes among our four primary tests. Figure 3 demonstrates this graceful degradation on Task 3021 across increasing missingness rates.

![](images/9d82c9702eb51d1d705fe4338b3c6e39d95a05eefeea771c80eb5cdc0822770a.jpg)

Figure 2: Per-task $\Delta \mathrm { A U C }$ (SSL−Scratch) at 5% labels. Datasets with > 1% native missingness are colored “Has Missing”; datasets at or below that threshold are colored “Clean.”  
![](images/5083f5ec8100de0346458807e63c0a7e1ef1ffb1fe9c2945950d20351d60de45.jpg)  
Figure 3: AUC-ROC on Task 3021 as additional missingness is injected at test time.

## 6.4 Ablation study

We ablate the consistency loss, group masking, and mask-visibility components at 10% labels on the four tasks with non-trivial native missingness (Task 3021, 22.2%; Task 14954, 5.3%; Task 2079, 4.3%; Task 146800, 1.7%). An initial run trained each variant once per task using a single pretraining seed and appeared to show that removing the consistency loss improved performance on all four tasks. Given the small number of tasks involved, we treated this as provisional and re-ran every variant across three independent pretraining seeds (matching the n = 3 seed protocol used throughout the rest of this paper), averaging the resulting AUC-ROC before computing deltas (Figure 4).

Under this seed-averaged protocol, the apparent consistency-loss effect does not replicate. Removing consistency regularization no longer improves all four tasks: it helps on Task 3021 (+0.0088) and Task 146800 (+0.0098), but hurts on Task 2079 (−0.0037) and Task 14954 (−0.0025), for a mean $\Delta \mathrm { A U C ~ o f ~ } + 0 . 0 0 3 1$ across tasks (Wilcoxon p = 0.625, n = 4; p = 0.233 across all 12 seed-level pairs). Group masking similarly shows no reliable direction (mean $\Delta \mathrm { A U C } = + 0 . 0 0 0 4$ $p = 1 . 0 )$ : Task 3021 (−0.0022) and Task 14954 (−0.0024) degrade while Task 2079 (+0.0016) and Task 146800 (+0.0045) improve, all well within the noise observed across pretraining seeds (per-task standard deviations ranged from 0.004 to 0.015 AUC). We interpret this as evidence that our original single-seed ablation reflected uncontrolled variance between independently-initialized pretraining runs rather than a genuine effect of the consistency loss or group masking on how the encoder handles native missingness.

We separately tested whether the Missing vs. Clean Paradox (Section 6.2) stems from the encoder never observing the $( x { = } 0 , m { = } 1 )$ signature of true missingness during pretraining, since our default corruption scheme always pairs synthetic masking with m=0 (Section 4). We re-ran pretraining with the pretext mask also written into m (“Mask-Aware”), again averaged over three pretraining seeds. This effect was also inconsistent across the four tasks (mean $\Delta \mathrm { A U C } = + 0 . 0 0 1 1 , p = 0 . 8 7 5 \colon$ Task $3 0 2 1 + 0 . 0 1 0 5$ , Task 2079 −0.0031, Task 14954 +0.0074, Task 146800 −0.0104), indicating that mask-visibility during pretraining is not a reliable driver of the paradox either.

Overall, none of the three ablated components produced an effect distinguishable from noise once averaged across pretraining seeds. We view this as an important methodological finding in its own right: with only four native-missingness tasks and substantial per-seed variance, single-seed ablation comparisons in tabular SSL can produce confident-looking but spurious patterns, and multi-seed averaging is necessary before attributing an effect to a specific design choice.

![](images/0d4a3dba874a5d80576591bcc3fad218a1ee8ff293089c3160a41a659dd5129d.jpg)  
Figure 4: Ablation results at 10% labels across the four native-missingness tasks (missingness % in parentheses), averaged over three independent pretraining seeds; error bars show ±1 standard deviation across seeds. “Mask-Aware” denotes pretraining where the pretext corruption mask is also passed to the encoder via m.

## 6.5 Classical baselines at 10% labels

Table 1 compares our deep tabular models to classical tree-based ensembles and linear models. SSL pretraining elevates the neural network to be highly competitive with state-of-the-art methods: it outperforms Logistic Regression by a wide margin (+0.0688 AUC), edges out XGBoost by a smaller +0.0135 AUC, closely matches LightGBM (+0.0079 AUC), and narrowly trails Random Forest.

<table><tr><td>Method Avg AUC @ 10% labels</td></tr><tr><td>Random Forest 0.9015</td></tr><tr><td>SSL Pretrained (Ours) 0.8954</td></tr><tr><td>Scratch (No Pretrain) 0.8899</td></tr><tr><td>LightGBM 0.8875</td></tr><tr><td>XGBoost 0.8819</td></tr><tr><td>Logistic Regression 0.8266</td></tr></table>

Table 1: Baseline comparison averaged across tasks at 10% labels. Classical baselines use a single fixed seed (42); SSL Pretrained and Scratch use a single shared pretraining run fine-tuned across 3 independent seeds (see Limitations).

## 6.6 Comparison against prior tabular SSL objectives

To assess whether our mask-and-recover objective is empirically distinguishable from established tabular SSL pretext tasks, we re-implemented VIME [12], SCARF [13], and SubTab [14] using the identical dual-branch encoder architecture described in Section 4, differing only in the pretraining objective. This controls for architectural capacity so that any observed difference can be attributed to the pretraining objective itself. Each baseline was pretrained and fine-tuned at 10% labels across all 14 tasks, averaged over 3 independent pretraining seeds, each using its own (not our) batch size, LR schedule, and clipping defaults.

Table 2 reports the results. None of the three baselines differ significantly from our method: VIME (mean $\Delta = - 0 . 0 0 2 4 .$ , Wilcoxon $p = 0 . 1 5 3 )$ , SCARF (mean $\Delta = + 0 . 0 0 1 1 , p = 0 . 6 7 0 )$ and SubTab (mean $\Delta = + 0 . 0 0 4 1 , p = 0 . 5 0 2 )$ ; all remain non-significant after Holm-Bonferroni correction across this family of three tests (adjusted $p = 0 . 4 5 9 .$ $p ~ = ~ 1 . 0 0 0$ , and $p \ = \ 1 . 0 0 0$ respectively). We interpret this as evidence that our objective is empirically competitive with, rather than distinguishable from, existing tabular SSL formulations under a controlled comparison, rather than as evidence that the choice of pretraining objective is inconsequential—a larger and more diverse task sample would be needed to detect smaller effects, if they exist.

<table><tr><td>Method</td><td>Avg AUC @ 10%</td><td>∆ vs. Ours</td><td> $\operatorname { A d j . } p$ </td></tr><tr><td>Ours (mask-and-recover)</td><td>0.8954</td><td></td><td></td></tr><tr><td>VIME [12]</td><td>0.8978</td><td>-0.0024</td><td>0.459</td></tr><tr><td>SCARF [13]</td><td>0.8943</td><td>+0.0011</td><td>1.000</td></tr><tr><td>SubTab [14]</td><td>0.8913</td><td>+0.0041</td><td>1.000</td></tr></table>

Table 2: Comparison against prior tabular SSL objectives using an identical encoder architecture. $\Delta$ is Ours − baseline; adjusted p-values are Holm-Bonferroni corrected across this family of 3 tests.

## 7 Discussion

When does SSL help most? Improvements peak at 5% labels and shrink as more labels become available, reversing sign entirely by 20% labels (Section 6). More interestingly, our findings challenge the assumption that missing-value imputation objectives universally benefit datasets with heavy native missingness. Instead, we observed the strongest and most reliable gains on originally clean datasets.

Why does robustness improve under missingness? A mask-and-recover SSL objective forces the encoder to rely on redundant predictive structures across features, reducing sensitivity to missing entries at test time. This is supported by the positive average gain observed under structured MNAR missingness shifts (+0.0418 AUC, positive on 8 of 14 tasks), though this effect does not reach significance after Holm-Bonferroni correction (adjusted $p = 0 . 5 1 8 )$ ; we treat this as a suggestive rather than confirmatory finding.

Why are some components not consistently helpful? Extended ablations (Section 6.4) initially suggested that consistency regularization was consistently harmful on datasets with native missingness. However, this effect did not survive averaging over multiple independent pretraining seeds: across three seeds per variant, the sign of the effect varied by task and the aggregate delta was statistically indistinguishable from zero. The same was true for group masking and mask-aware pretraining. We take this as a caution against over-interpreting single-seed ablation comparisons in small-sample tabular SSL studies, rather than as evidence that these design choices are inert—a larger task sample or more pretraining seeds per variant would be needed to reliably detect effects of this magnitude, if they exist at all.

## 8 Limitations and Broader Impact

Limitations. While self-supervised pretraining demonstrates strong average performance and testtime robustness, our empirical evaluation highlights several limitations. First, the mask-and-recover objective does not universally benefit all tabular manifolds; on datasets with high native missingness, the injection of additional artificial missingness during pretraining can occasionally cause the model to underfit the available signal. Second, the computational overhead of training a deep tabular encoder for 50 pretraining epochs and 30 fine-tuning epochs is significantly higher than fitting a classical decision tree. Third, there is an asymmetry in evaluation rigor: our deep tabular models are averaged over 3 independent seeds for the label-scarcity, ablation, and SSL-baseline results, but the MCAR and MNAR robustness deltas (Section 6.3) use a single fixed fine-tuning seed (42) rather than a 3-seed average, so those two numbers carry additional seed-selection noise not present in our other primary tests; for the same reason, the "∆ 10%" column of Table 4 (single-seed) need not exactly match the 3-seed-averaged +0.0055 figure reported in Section 6.3. The classical baselines in Table 1 are likewise fit with a single fixed seed (Appendix C), so their reported scores do not reflect seed-to-seed variance the way the 3-seed deep-model numbers do. Finally, our group masking strategy relies on linear correlation metrics (Spearman rank), which may fail to capture complex, non-linear dependencies between features. Relatedly, three tasks (15 and 32 from 1% labels onward, and 6 from 5% labels onward) sit at or above roughly 0.98 AUC (Task 6 is the exception at 1% labels, where all methods score below 0.93); these near-ceiling tasks necessarily contribute small, low-variance deltas to our paired statistical tests, which slightly dilutes the effective sample even though n = 14 is used nominally throughout. We also tested, and could not confirm, a specific mechanistic explanation for the Missing vs. Clean Paradox (mask-visibility during pretraining, Section 6.4); the paradox’s root cause remains an open question for future work. Additionally, our ablation study is limited to four native-missingness tasks; even after averaging over three independent pretraining seeds per variant, this sample size affords limited statistical power to detect small effects (a Wilcoxon test on four paired samples cannot reach $p < 0 . 0 5$ regardless of effect size), so our null findings for the consistency loss, group masking, and mask-aware components should be read as an absence of detected effect rather than definitive proof of no effect. Similarly, our comparison against prior tabular SSL objectives (Section 6.6) is restricted to 10% labels and does not rule out differences that might emerge at other label fractions or with per-method hyperparameter tuning; we deliberately did not tune hyperparameters per dataset or per method (Appendix D) in order to isolate the pretraining objective as the variable of interest, which is a defensible but real constraint on how far these comparisons generalize. Finally, all p-values in this paper are corrected for multiple comparisons within their respective families (Section 5); readers should interpret the raw, uncorrected effect sizes (e.g., the +0.0245 AUC gain under MCAR) as suggestive rather than confirmatory, given that none of our primary robustness comparisons remain significant after correction.

Broader Impact. The ability to train robust models under extreme label scarcity has profound implications for domains where labeling requires expensive human expertise, such as medical diagnostics and fraud detection. Furthermore, achieving zero-shot robustness to test-time sensor degradation (MNAR shifts) reduces the likelihood of catastrophic algorithmic failures in production environments. However, practitioners must remain cautious; deep tabular models are inherently less interpretable than shallow decision trees, and deploying them in high-stakes environments requires secondary explainability audits to ensure predictions do not rely on spurious correlations.

## 9 Conclusion

In this paper, we presented a rigorous empirical evaluation of self-supervised pretraining for tabular deep learning under extreme label scarcity and test-time missingness. Our analysis across diverse OpenML datasets revealed that while mask-and-recover SSL objectives generally outperform scratch training and are highly competitive with classical tree ensembles, these performance gains exhibit significant inter-task variance. Notably, we uncovered a "clean vs. missing" paradox: imputationbased pretext tasks yield the most reliable downstream improvements on natively clean datasets, rather than those with inherent missingness. Despite this variance, SSL-pretrained models achieved a higher average AUC under test-time degradation, degrading more gracefully than scratch-trained models under both random (MCAR) missingness injections and structured (MNAR) test-time shifts on average, though neither difference remains significant after correcting for multiple comparisons, nor is the effect uniform across all 14 tasks. We additionally found that our mask-and-recover objective performs comparably to three established tabular SSL baselines—VIME, SCARF, and SubTab—under a controlled, architecture-matched comparison, indicating our findings are not an artifact of an unusually weak or strong pretraining objective. Finally, our extended, seed-averaged ablations found no reliable effect of view-consistency regularization, group masking, or maskaware pretraining on native-missingness tasks—an initial single-seed comparison had suggested a consistent harmful effect of the consistency loss, but this did not replicate once properly averaged across independent pretraining seeds, underscoring the importance of multi-seed evaluation in smallsample tabular SSL ablation studies. Ultimately, our findings establish a more nuanced understanding of when and why self-supervised learning succeeds, highlighting the need for dataset-adaptive SSL designs in real-world tabular data mining.

Future Work. Several directions follow naturally from these findings. First, given that our seedaveraged ablations found no reliable effect for the consistency loss or group masking on the four native-missingness tasks studied here, a larger and more diverse pool of native-missingness datasets would provide the statistical power needed to determine whether these components have a genuine, if small, effect, or whether they are truly inert with respect to missingness handling. Second, the root cause of the Missing vs. Clean Paradox remains open: future work could investigate whether alternative corruption schemes, such as explicitly modeling the joint distribution of synthetic and native missingness rather than treating them as interchangeable, might resolve the paradox where our mask-aware ablation did not. Third, extending this evaluation protocol to a broader set of pretraining objectives beyond mask-and-recover, such as contrastive or generative approaches, would help determine whether the robustness gains and paradoxical behavior we observe are specific to this pretraining paradigm or are a more general property of self-supervised tabular representations. We hope the dataset-adaptive perspective and the seed-averaging methodology adopted here prove useful to future empirical studies in this area.

## Declarations

Funding: This research was conducted independently; no external funding or grants were received.   
Conflicts of Interest: The author declares no competing interests.

Data Availability: All datasets utilized in this study are publicly accessible via the OpenML repository.

Code Availability: The complete PyTorch implementation, experimental pipeline, and evaluation scripts used to generate the results in this manuscript are available at https://github.com/ shnd23/ssl-tabular-label-scarcity.

## References

[1] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

[2] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30.

[3] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.

[4] Arik, S. O., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679–6687.

[5] Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. Advances in Neural Information Processing Systems, 34, 18932–18943.

[6] Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T. (2021). SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training. arXiv preprint arXiv:2106.01342.

[7] Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems, 35.

[8] Gorishniy, Y., Rubachev, I., & Babenko, A. (2022). On embeddings for numerical features in tabular deep learning. Advances in Neural Information Processing Systems, 35, 24991–25004.

[9] Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). TabTransformer: Tabular data modeling using contextual embeddings. arXiv preprint arXiv:2012.06678.

[10] Kossen, J., Band, N., Lyle, C., Gomez, A. N., Rainforth, T., & Gal, Y. (2021). Self-attention between datapoints: Going beyond individual input-output pairs in deep learning. Advances in Neural Information Processing Systems, 34.

[11] Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., & Kasneci, G. (2024). Deep neural networks and tabular data: A survey. IEEE Transactions on Neural Networks and Learning Systems, 35(6), 7499–7519.

[12] Yoon, J., Zhang, Y., Jordon, J., & van der Schaar, M. (2020). VIME: Extending the success of self-and semi-supervised learning to tabular domain. Advances in Neural Information Processing Systems, 33, 11033–11043.

[13] Bahri, D., Jiang, H., Tay, Y., & Metzler, D. (2021). SCARF: Self-supervised contrastive learning using random feature corruption. International Conference on Learning Representations.

[14] Ucar, T., Hajiramezanali, E., & Edwards, L. (2021). SubTab: Subsetting features of tabular data for self-supervised representation learning. Advances in Neural Information Processing Systems, 34, 18853–18865.

[15] Vanschoren, J., van Rijn, J. N., Bischl, B., & Torgo, L. (2013). OpenML: Networked science in machine learning. SIGKDD Explorations Newsletter, 15(2), 49–60.

[16] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825– 2830.

[17] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

[18] He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16000–16009.

[19] Little, R. J. A., & Rubin, D. B. (2002). Statistical analysis with missing data (2nd ed.). John Wiley & Sons.

[20] Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7(1), 1–30.

[21] Troyanskaya, O., Cantor, M., Sherlock, G., Brown, P., Hastie, T., Tibshirani, R., ... & Altman, R. B. (2001). Missing value estimation methods for DNA microarrays. Bioinformatics, 17(6), 520–525.

[22] van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. Journal ofStatistical Software, 45(3), 1–67.

[23] Yoon, J., Jordon, J., & van der Schaar, M. (2018). GAIN: Missing data imputation using generative adversarial nets. Proceedings ofthe 35th International Conference on Machine Learning, 80, 5689–5698.

[24] Stekhoven, D. J., & Bühlmann, P. (2012). MissForest—non-parametric missing value imputation for mixed-type data. Bioinformatics, 28(1), 112–118.

[25] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations (ICLR).

[26] Smith, L. N., & Topin, N. (2019). Super-convergence: Very fast training of neural networks using large learning rates. Artificial Intelligence and Machine Learningfor Multi-Domain Operations Applications, 11006, 369–386.

## A Per-task Results and Robustness Table

The following tables provide the exact, per-task evaluation metrics used to compute the aggregate summaries presented in the main text. Table 3 details the test AUC-ROC performance of both the Self-Supervised Learning (SSL) pretrained encoder and the Scratch baseline across four distinct label scarcity regimes (1%, 5%, 10%, and 20%). Table 4 details the exact ∆ AUC (SSL−Scratch) under the normal 10% labeled setting, alongside the two test-time degradation settings: Missing Completely At Random (MCAR +30%) and the structured Missing Not At Random (MNAR) shift.
<table><tr><td>Task</td><td>Missing(%)</td><td>SSL 1%</td><td>Scr 1%</td><td>SSL 5%</td><td>Scr 5%</td><td>SSL 10%</td><td>Scr 10%</td><td>SSL 20%</td><td>Scr 20%</td></tr><tr><td>2079</td><td>4.305</td><td>0.554</td><td>0.560</td><td>0.742</td><td>0.767</td><td>0.796</td><td>0.807</td><td>0.848</td><td>0.869</td></tr><tr><td>3021</td><td>22.216</td><td>0.938</td><td>0.872</td><td>0.934</td><td>0.930</td><td>0.957</td><td>0.948</td><td>0.970</td><td>0.970</td></tr><tr><td>14954</td><td>5.258</td><td>0.594</td><td>0.680</td><td>0.738</td><td>0.759</td><td>0.780</td><td>0.806</td><td>0.799</td><td>0.830</td></tr><tr><td>146800</td><td>1.732</td><td>0.688</td><td>0.685</td><td>0.881</td><td>0.912</td><td>0.943</td><td>0.956</td><td>0.987</td><td>0.992</td></tr><tr><td>2</td><td>0.000</td><td>0.700</td><td>0.594</td><td>0.864</td><td>0.776</td><td>0.976</td><td>0.923</td><td>0.991</td><td>0.982</td></tr><tr><td>15</td><td>0.247</td><td>0.993</td><td>0.988</td><td>0.991</td><td>0.985</td><td>0.987</td><td>0.989</td><td>0.987</td><td>0.983</td></tr><tr><td>29</td><td>0.617</td><td>0.843</td><td>0.858</td><td>0.866</td><td>0.846</td><td>0.866</td><td>0.842</td><td>0.885</td><td>0.887</td></tr><tr><td>3904</td><td>0.010</td><td>0.662</td><td>0.640</td><td>0.643</td><td>0.624</td><td>0.650</td><td>0.633</td><td>0.690</td><td>0.672</td></tr><tr><td>7592</td><td>0.000</td><td>0.873</td><td>0.858</td><td>0.882</td><td>0.881</td><td>0.885</td><td>0.886</td><td>0.893</td><td>0.895</td></tr><tr><td>6</td><td>0.000</td><td>0.925</td><td>0.923</td><td>0.981</td><td>0.982</td><td>0.994</td><td>0.994</td><td>0.998</td><td>0.998</td></tr><tr><td>32</td><td>0.000</td><td>0.989</td><td>0.987</td><td>0.997</td><td>0.999</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>37</td><td>0.000</td><td>0.835</td><td>0.832</td><td>0.859</td><td>0.816</td><td>0.840</td><td>0.805</td><td>0.881</td><td>0.875</td></tr><tr><td>43</td><td>0.000</td><td>0.929</td><td>0.924</td><td>0.938</td><td>0.946</td><td>0.943</td><td>0.953</td><td>0.953</td><td>0.965</td></tr><tr><td>53</td><td>0.000</td><td>0.614</td><td>0.661</td><td>0.841</td><td>0.831</td><td>0.919</td><td>0.916</td><td>0.921</td><td>0.924</td></tr></table>

Table 3: Per-task AUC-ROC results for SSL pretraining vs. scratch training across four label scarcity fractions. Metrics represent the mean over 3 random seeds and are rounded to 3 decimals for readability; statistical tests use full-precision values, available in the accompanying code.
<table><tr><td>Task</td><td>Missing(%)</td><td>∆10%</td><td>∆MCAR</td><td>∆ Shift</td></tr><tr><td>2079</td><td>4.3051</td><td>-0.0071</td><td>-0.0223</td><td>-0.0113</td></tr><tr><td>3021</td><td>22.2157</td><td>0.0085</td><td>0.0267</td><td>-0.0185</td></tr><tr><td>14954</td><td>5.2583</td><td>-0.0456</td><td>-0.0000</td><td>0.0063</td></tr><tr><td>146800</td><td>1.7316</td><td>-0.0133</td><td>-0.0203</td><td>-0.0466</td></tr><tr><td>2</td><td>0.0000</td><td>0.1100</td><td>0.1551</td><td>0.1954</td></tr><tr><td>15</td><td>0.2473</td><td>0.0109</td><td>0.0154</td><td>-0.0018</td></tr><tr><td>29</td><td>0.6173</td><td>-0.0026</td><td>0.0051</td><td>0.0068</td></tr><tr><td>3904</td><td>0.0097</td><td>0.0465</td><td>0.0418</td><td>0.1661</td></tr><tr><td>7592</td><td>0.0000</td><td>-0.0024</td><td>0.0117</td><td>-0.0014</td></tr><tr><td>6</td><td>0.0000</td><td>0.0007</td><td>0.0006</td><td>-0.0210</td></tr><tr><td>32</td><td>0.0000</td><td>-0.0001</td><td>0.0107</td><td>0.0347</td></tr><tr><td>37</td><td>0.0000</td><td>0.0193</td><td>0.0548</td><td>0.1052</td></tr><tr><td>43</td><td>0.0000</td><td>-0.0081</td><td>0.0160</td><td>0.0997</td></tr><tr><td>53</td><td>0.0000</td><td>0.0039</td><td>0.0475</td><td>0.0713</td></tr></table>

Table 4: Robustness deltas at 10% labels (seed 42). Values represent ∆AUC (SSL−Scratch) under normal, MCAR (+30% missingness), and MNAR structured shift.

## B Data Dictionary and Context

To ensure the broad applicability of our empirical findings, we curated 14 datasets from the OpenML repository spanning diverse domains, feature dimensionalities, and missingness properties. A brief context for each dataset is provided below:

Task 2079 (Eucalyptus): An ecological dataset measuring the survival and growth of Eucalyptus species across different environments, containing categorical soil and genetic features alongside substantial missing sensor readings (4.3% missingness).

Task 3021 (Sick): A medical dataset aimed at identifying sick patients based on thyroid disease indicators. It contains a high degree of missingness (22.2%) due to patients frequently lacking full diagnostic workups.

Task 14954 (Cylinder Bands): An industrial manufacturing dataset predicting rotogravure printing press delays (cylinder banding). It features a mix of nominal and numeric features with 5.3% missingness.

Task 146800 (MiceProtein): A biological dataset tracking expression levels of 77 proteins in the cerebral cortex of mice (1.7% missingness), used to identify subpopulations subject to specific learning contexts.

Task 2 (Anneal): A metallurgical dataset used to predict the annealing process of steel. Although the raw dataset is infamous for heavy structural missingness in its categorical variables, the preprocessed numerical subset evaluated here is clean (0.0% missingness).

Task 15 (Breast-W): The original Wisconsin Breast Cancer dataset, using nine cytological characteristics (e.g., clump thickness, uniformity of cell size and shape, bare nuclei) manually scored on a 1–10 scale by a cytologist, rather than features computed via image processing.

Task 29 (Credit-Approval): A financial dataset analyzing credit card applications, featuring mixed data types and minor missingness related to anonymized applicant history.

Task 3904 (JM1): A software engineering dataset predicting software defects based on Halstead and McCabe complexity metrics extracted from C code.

Task 7592 (Adult): A census-based demographic dataset used to predict whether a person’s income exceeds \$50K/year. It is a highly cited benchmark in both tabular learning and algorithmic fairness.

Task 6 (Letter): An image-derived tabular dataset where the objective is to identify English capital letters from 16 statistical features extracted from raster scan images.

Task 32 (Pendigits): Pen-based recognition of handwritten digits, utilizing coordinate information collected from a pressure-sensitive tablet.

Task 37 (Diabetes): The Pima Indians Diabetes dataset, containing diagnostic measurements to predict the onset of diabetes mellitus within 5 years.

Task 43 (Spambase): A cybersecurity dataset used to classify emails as spam or non-spam based on word frequencies and character run-length metrics.

Task 53 (Vehicle): A computer vision-derived tabular dataset classifying 3D vehicle silhouettes (e.g., Saab, Van, Bus) based on geometric features.

## C Baseline Implementation Details

In Section 6.5, we compared our SSL pretraining against state-of-the-art tree-based ensembles and linear models. To ensure maximum reproducibility, we detail the exact parameter configurations

below.

For all classical baselines, categorical variables were strictly isolated, cast to string data types to avoid float coercion errors, and processed using Scikit-Learn’s [16] OrdinalEncoder. Logistic Regression: We utilized Scikit-Learn’s LogisticRegression class. Due to the high dimensionality and variance of some datasets, the max\_iter parameter was explicitly raised to 1000 to ensure gradient descent convergence.

Random Forest: We employed Scikit-Learn’s RandomForestClassifier. To provide a highly competitive baseline, we fixed the ensemble size to n\_estimators=100 and fully utilized parallel processing via $\mathrm { n \_ j o b s = - 1 }$ . Node splitting criteria remained at the default Gini impurity.

XGBoost: We used the XGBClassifier from the native xgboost Python package. The evaluation metric was explicitly set to eval\_metric=’logloss’. The label encoder deprecation warning was bypassed via use\_label\_encoder=False. All other boosting hyperparameters (e.g., maximum depth, learning rate) were kept at their highly optimized default values.

LightGBM: We employed the LGBMClassifier. To prevent memory fragmentation and console spam during the iteration over the 14 datasets, the model was executed with verbose=-1.

All classical baselines (Logistic Regression, Random Forest, XGBoost, LightGBM) were initialized with a fixed random seed of 42 to guarantee deterministic behavior; this is distinct from the 3-seed protocol used for the deep tabular models (Section 5). We note that our classical baselines use simple statistical imputation rather than learned imputation methods such as GAIN [23] or MissForest [24]; the choice of imputation strategy for classical baselines, and the broader question of why tree ensembles remain competitive with deep tabular models [7], are surveyed at length elsewhere [11] and are outside the scope of this comparison.

## D Extended Hyperparameter Configurations

To ensure fair comparison and reproducibility, we detail the complete hyperparameter configurations and initialization bounds used for both the deep tabular models and the classical baselines.

## D.1 Deep Tabular Encoder and SSL Objective

The deep tabular architecture was kept strictly uniform across all 14 datasets to isolate the impact of the self-supervised pretraining objective. No dataset-specific architecture tuning was performed. Our dual-branch MLP encoder is deliberately simple relative to attention-based tabular architectures such as TabTransformer [9], SAINT [6], and Non-Parametric Transformers [10], and relative to learned numerical-feature embeddings [8]; this choice keeps the pretraining objective, rather than architectural capacity, as the primary variable under study. Hyperparameter values (corruption rate $\rho = 0 . 3 ,$ , embedding dimension 32, hidden dimension 128, learning rate 10<sup>−3</sup>) follow common choices in the tabular SSL literature rather than being tuned per dataset or per method; we did not perform a hyperparameter search on any individual task, since doing so would confound the cross-task comparisons that are the focus of this study. This is a deliberate methodological choice rather than an oversight, though we acknowledge in Section 8 that dataset-specific tuning could interact with the Missing vs. Clean Paradox we report.

• Numeric Multi-Layer Perceptron: 2 layers, hidden dimension 128, Layer Normalization applied before ReLU activations, Dropout rate $p = 0 . 1$

• Categorical Embeddings: Dimensionality E = 32 per feature. Unknown categories at test time mapped to an index of 0.

• Fusion Multi-Layer Perceptron: 2 layers, hidden dimension 128, Layer Normalization, Dropout rate p = 0.1.

• Pretraining Optimizer: AdamW [25], Learning Rate = 10<sup>−3</sup>, Weight Decay = 10<sup>−4</sup>, Batch Size = 256, Grad. Clip 1.0, OneCycleLR [26].

• Corruption Rate: Bernoulli masking parameter $\rho = 0 . 3$

• Consistency Weight: λ = 1.0 (ablated to 0.0; see Section 6.4 for results averaged over 3 independent pretraining seeds).

## D.2 Tree-Based Ensembles

The classical baselines were instantiated using their respective Python libraries. To provide a rigorous baseline, we ensured all tree ensembles utilized 100 estimators, providing sufficient capacity to model the tabular manifolds without excessive computational overhead.

• Random Forest (Scikit-Learn): n\_estimators = 100, criterion = ’gini’, max\_depth = None, min\_samples\_split = 2, min\_samples\_leaf = 1, max\_features = ’sqrt’, bootstrap = True.

• XGBoost: n\_estimators = 100, learning\_rate = 0.3, max\_depth = 6, subsample = 1.0, colsample\_bytree = 1.0, eval\_metric = ’logloss’.

• LightGBM: n\_estimators = 100, learning\_rate = 0.1, num\_leaves = 31, boosting\_type = ’gbdt’.