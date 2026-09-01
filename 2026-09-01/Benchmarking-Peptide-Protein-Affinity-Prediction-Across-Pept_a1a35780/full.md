# Benchmarking Peptide–Protein Affinity Prediction Across Peptide and Target Shifts

Jiaxin Tian<sup>1</sup>, Darren An<sup>2</sup>, and Jun Li<sup>2,\*</sup>

## Abstract

Peptide–protein affinity models are often evaluated with a single data split, obscuring whether they interpolate among measurements for observed targets or generalize across peptide or target shifts. We integrated three sources of quantitative peptide–protein binding data to obtain 11,349 deduplicated pairs and benchmarked ten peptide representations, ESM-2 protein embeddings, and six regressors under peptide-similarity, within-target, and leave-target-out partitions. Across 60 matched representation–regressor configurations, mean test Spearman correlations were 0.462, 0.669, and 0.530, respectively. The top configuration shifted from ECFP-16 count fingerprints with random forest in the first two settings to HELM-BERT with Extra Trees when exact target sequences were excluded. Representation-rank correlations ranged from −0.042 to 0.624 across partitions, whereas regressor-rank correlations ranged from 0.771 to 0.943. Learning curves showed that representation differences were largest with limited supervision and narrowed as training data increased. PeptideCLM-2 adaptation and simple element-wise interaction features provided no consistent gain over a frozen encoder and direct concatenation under the tested protocols. These conclusions are specific to a dataset that pools transformed K<sub>d</sub>, K<sub>i</sub>, and IC<sub>50</sub> measurements and to target exclusion at the exact-sequence level. Peptide–protein affinity benchmarks should therefore align data partitions with the intended use and jointly assess the effects of data scale, molecular representation, and downstream learner.

## 1. Introduction

Peptides can engage extended protein surfaces with high affinity and selectivity while remaining accessible to chemical synthesis [1, 2]. This makes them attractive for protein–protein interaction targets that are often difficult for small molecule [3, 4]. Their development is limited by proteolysis, short exposure, poor oral absorption, and low membrane permeability [5]. Cyclization, backbone modification, terminal capping, and non-canonical residues can mitigate these liabilities, but greatly enlarge the design space [6–8].

Computational models can reduce experimental screening by prioritizing peptide–target pairs. Peptides may be encoded as sequences, molecular graphs, circular fingerprints, or pretrained embeddings; proteins can likewise be represented by sequence language models. Earlier affinity models used physicochemical sequence kernels, while recent systems address motif–domain binding, interaction classification, and residue-level binding sites with sequence-, structure-, or attention-based architectures [9–14]. Structural resources such as PepBDB and larger activity collections support complementary analyses [15, 16]. However, results across these tasks are not directly comparable because their endpoints, negative examples, and held-out entities differ.

Representation comparisons also depend on data partitioning. Random record-level splits may place close molecular analogues, homologous sequences, or repeated biological entities in both training and test sets, favoring interpolation and potentially overstating prospective performance [17–21]. Fernández-Díaz et al. showed that representation rankings depended on peptide class and downstream learner, that chemical fingerprints remained competitive, and that transfer between canonical and modified peptides was harder than within-class interpolation [22]. These findings show that representation quality cannot be discussed independently of data partition and downstream learner. Pairwise affinity prediction also adds target novelty: a peptide may be chemically distinct while its target is represented in training, or the target sequence itself may be absent. These settings require separate evaluation.

We therefore considered three deployment regimes. Peptide-similarity partitioning tests transfer to structurally separated peptides while allowing targets to recur. Within-target prediction models follow-up measurements for observed proteins. Leave-target-out prediction tests exact protein sequences excluded from fitting. The first two settings can exploit target-specific information learned from other records; the third requires the protein representation and peptide–protein mapping to transfer jointly. It does not, however, guarantee separation between homologous proteins. More generally, molecular out-of-distribution studies show that conclusions can change with the source of distribution shift [21, 23].

![](images/c6893c2c725f2b24542a6d70aac00c1b19c7acb6e4e2e8974923764aea1efee4.jpg)  
Figure 1. Study design for generalization-aware peptide–protein affinity prediction. Canonical, non-canonical, and PPIKB records were merged and deduplicated at the peptide–target-pair level. Three partitions represented structurally separated peptides, additional measurements for observed targets, and unseen target sequences. Peptide and protein features were generated independently, fused, and supplied to downstream regressors. The benchmark compared generalization regimes, training-set sizes, peptide-encoder adaptation, and pairwise feature-fusion strategies, with SPCC as the primary metric.

Our benchmark asks whether performance and model rankings change across these regimes, how training-set size affects the relative performance of peptide representations and downstream learners, and whether adapting a pretrained peptide encoder is more useful than increasing supervision. We also test whether simple element-wise peptide–protein features improve direct concatenation. Three sources of quantitative peptide–protein binding data were integrated and deduplicated to obtain 11,349 measurements for 9,510 peptides and 2,659 target sequences. Ten peptide representations were paired with ESM-2 protein embeddings and six regressors. Learning curves, PeptideCLM-2 adaptation, and feature-fusion experiments then separated the contributions of representation, data scale, and model design. The objective was not a universal winner, but conclusions tha remain interpretable across intended uses.

## 2. Materials and Methods

## 2.1. Data sources and endpoint transformation

The benchmark combined three sources of quantitative peptide–protein binding data (Table 1). Canonical and non-canonica datasets contributed 1,002 and 299 records, respectively [22]. The PPIKB preprint reports 21,760 manually validated quantitative affinity entries overall [16]; the study-specific PPIKB-derived input used here comprised 15,248 canonical records. Records were keyed by peptide SMILES and complete target sequence, then duplicate pairs were removed. The final dataset comprised 11,349 pairs, 9,510 peptides, and 2,659 protein sequences.

The source records reported dissociation constants $( K _ { d } )$ , inhibition constants $( K _ { i } )$ , or half-maximal inhibitory concentrations $( I C _ { 5 0 } )$ . Values were first expressed in molar units and then mapped to

$$
S = - \log _ { 1 0 } ( C ) ,\tag{1}
$$

where C is the reported molar concentration. Larger S indicates stronger apparent affinity. The transformation supplies a common modeling scale without assuming thermodynamic equivalence among endpoints.

## 2.2. Molecular representations and pair construction

Ten peptide featurizations covered different molecular resolutions and pretraining domains (Table 2): 2,048-dimensional binary and count ECFP-16 fingerprints [24], PepFuNN [25], chemical language models ChemBERTa-2 and MolFormer-XL [26, 27], peptide language models HELM-BERT, PeptideCLM, PeptideCLM-2, and PepTune [28–31], and graph-based PepLand [32]. In the primary representation–regressor benchmark, pretrained neural encoders were frozen to produce fixed-length peptide features, and target proteins were represented by fixed-length ESM-2 features [33]. For each peptide–protein pair, the precomputed peptide and protein vectors were directly concatenated and supplied to each of the six downstream regressors. Task-specific adaptation of PeptideCLM-2 was evaluated separately in section 2.5.

The separate feature-fusion ablation used an independent neural prediction architecture with frozen PeptideCLM-2 representations and precomputed protein representations. Peptide vector p and target vector t were mapped to a shared dimensionality through separate learnable projections:

$$
\widetilde { \pmb { p } } = f _ { p } ( \pmb { p } ) , \qquad \widetilde { \pmb { t } } = f _ { t } ( \pmb { t } ) .\tag{2}
$$

Each projection consisted of LayerNorm, a linear mapping, GELU activation, and dropout, and mapped its input to a 512-dimensional latent space. The first fusion strategy concatenated the two projected vectors,

$$
z _ { \mathrm { c o n c a t } } = \Big [ \widetilde { p } , \widetilde { t } \Big ] .\tag{3}
$$

The interaction-aware alternative appended the element-wise product and absolute difference,

$$
z _ { \mathrm { i n t e r a c t i o n } } = \Big [ \widetilde { p } , \widetilde { t } , \widetilde { p } \odot \widetilde { t } , \Big | \widetilde { p } - \widetilde { t } \Big | \Big ] ,\tag{4}
$$

where ⊙ denotes element-wise multiplication. Both fusion rules used identical partitions, seeds, projection dimensions, and MLP prediction heads.

## 2.3. Generalization-oriented data partitions

The peptide-similarity partition represented structurally separated peptides. Following Hestia-GOOD/CCPart [23], similarity used 2,048-bit MAPc fingerprints of diameter 20 (radius 10) and the Jaccard index. Connected-component partitions were generated at thresholds 0.1–1.0 in 0.1 increments. The target test fraction was 20%; partitions were retained when test records comprised at least 18.5% of the data, and 10% of the remaining pool was used for validation. Peptides obeyed the similarity separation, while target sequences could recur. Cross-partition model comparisons and learning curves used threshold 0.5; the full threshold series was used only for the fusion sensitivity analysis.

The within-target partition represented follow-up measurements for known targets. Within each exact target sequence, records were assigned to training, validation, and test subsets at approximately 70/10/20. Targets with fewer than three records remained in training. Target sequences, but not individual records, could therefore recur across subsets.

The leave-target-out partition grouped all records for an exact protein sequence in one subset. Because target groups were imbalanced, 5,000 randomized assignments were evaluated and the assignment closest to 70/10/20 record proportions was retained (seed 42). Exact sequences could not overlap, but no homology or protein-family threshold was imposed.

Table 1. Sources and composition of the peptide–protein binding benchmark.
<table><tr><td>Dataset</td><td>Peptide class</td><td>Records</td><td>Ref.</td></tr><tr><td>Protein-peptide binding</td><td>Canonical</td><td>1,002</td><td>[22]</td></tr><tr><td>Protein-peptide binding</td><td>Non-canonical</td><td>299</td><td>[22]</td></tr><tr><td>PPIKB-derived benchmark input²</td><td>Canonical</td><td>15,248</td><td>[16]</td></tr><tr><td>Integrated benchmarkb</td><td>Mixed</td><td>11,349</td><td>[16, 22]</td></tr></table>

<sup>a</sup>Study-specific canonical subset; PPIKB reports 21,760 manually validated quantitative affinity entries overall. <sup>b</sup>Number remaining after merging the three sources and removing repeated peptide–target pairs using peptide SMILES and target sequence as the pair identifier.

Table 2. Peptide and protein representations included in the benchmark.
<table><tr><td>Entity</td><td>Representation family</td><td>Model or feature</td><td>Ref.</td></tr><tr><td rowspan="5">Peptide</td><td>Circular fingerprints</td><td>ECFP-16</td><td>[24]</td></tr><tr><td>Peptide fingerprint</td><td>ECFP-16 counts PepFuNN</td><td>[24]</td></tr><tr><td>Chemical language models</td><td>ChemBERTa-2</td><td>[25] [26]</td></tr><tr><td></td><td>MolFormer-XL</td><td>[27]</td></tr><tr><td rowspan="4">Peptide language models</td><td>HELM-BERT</td><td>[28]</td></tr><tr><td>PeptideCLM</td><td>[29]</td></tr><tr><td></td><td></td></tr><tr><td>PeptideCLM-2 PepTune</td><td>[30] [31]</td></tr><tr><td>Protein language model</td><td>Peptide graph model</td><td>PepLand</td><td>[32]</td></tr><tr><td>Protein</td><td></td><td>ESM-2</td><td>[33]</td></tr></table>

## 2.4. Downstream learning and model selection

Six regressors covered linear, kernel, bagged-tree, boosted-tree, and neural approaches: support vector regression (SVR) [34], LightGBM [35], random forest [36], ridge regression [37], Extra Trees [38], and a multilayer perceptron (MLP) [39].

Optuna optimized each representation–regressor combination and setting against training-only five-fold cross-validation SPCC [40]. Searches used 30 trials and stopped after 20 trials without improvement; predefined validation and test subsets were excluded from hyperparameter selection. Selected configurations were refitted on the full training subset and evaluated on the fixed test subset with five random seeds.

## 2.5. Training-set scaling and peptide-encoder adaptation

To examine the sample efficiency of pretrained peptide representations across supervision levels, learning curves combined five peptide-specific representations—HELM-BERT, PeptideCLM, PeptideCLM-2, PepTune, and PepLand—with all six regressors. Experiments used the fixed peptide-similarity partition at threshold 0.5. Training sizes were 100, 300, 1,000, 3,000, 6,000, and all 8,172 available training records. Validation and test sets were fixed, and optimization was repeated at each size. Model-seed results were first summarized within each training-set subsampling replicate and then averaged across available replicates. Because uncertainty intervals were not retained in the Figure 4 summary, comparisons among these learning curve were interpreted descriptively.

Encoder adaptation used PeptideCLM-2 hybrid-small, distinct from the MLM-small checkpoint in the fixed-representation benchmark. We compared a frozen encoder, unfreezing the final one, two, or four transformer blocks, full fine-tuning, and Low-Rank Adaptation (LoRA) [41]. The frozen strategy trained only the MLP; partial strategies also updated corresponding normalization layers. LoRA adapters were inserted into attention-related linear layers with rank r = 8, scaling α = 16, and dropout 0.05.

All strategies used fixed protein features, concatenation, the same MLP head and peptide-similarity partition at threshold 0.5, and all six training sizes. Three matched training-set subsampling replicates were evaluated. Validation loss controlled early stopping and model selection.

## 2.6. Evaluation and statistical analysis

Metrics were Spearman correlation (SPCC), Pearson correlation (PCC), root mean squared error (RMSE), and mean absolute error (MAE); mean squared error was also recorded. SPCC was primary because it measures agreement between the rankings of predicted and experimental values without requiring a linear relationship between them; all comparative analyses reported here use SPCC.

Five-seed results were summarized by mean and standard deviation. Seed-averaged test SPCC for each of the 60 matched representation–regressor combinations was the paired unit in scenario comparisons. Two-sided Wilcoxon signed-rank tests were used with Holm adjustment for three contrasts. This inference quantifies consistency across the evaluated model configurations rather than uncertainty from resampling the underlying dataset. Configurations were ranked by mean SPCC within each scenario, and scenario rank vectors were compared by Spearman correlation. Representation scores were averaged over regressors, regressor scores over representations, and ties received average ranks.

For the adaptation study, each learning curve was summarized by its area under the learning curve (AULC). Training sizes $n _ { i }$ were transformed to a normalized logarithmic coordinate,

$$
x _ { i } = \frac { \log _ { 1 0 } ( n _ { i } ) - \log _ { 1 0 } ( n _ { \operatorname* { m i n } } ) } { \log _ { 1 0 } ( n _ { \operatorname* { m a x } } ) - \log _ { 1 0 } ( n _ { \operatorname* { m i n } } ) } ,\tag{5}
$$

and integrated by the trapezoidal rule,

$$
\mathrm { A U L C } = \sum _ { i = 1 } ^ { K - 1 } \frac { y _ { i } + y _ { i + 1 } } { 2 } \left( x _ { i + 1 } - x _ { i } \right) ,\tag{6}
$$

where $y _ { i }$ is SPCC at size $n _ { i } .$ . All six sizes, including 8,172 records, contributed to AULC; only complete learning curves were analyzed. A Friedman test compared adaptation strategies, using strategy as the repeated condition and each of the three training-set subsampling replicates as a block.

## 3. Results

## 3.1. The held-out entity changes both performance and model choice

Across 60 seed-averaged representation–regressor combinations, mean (median) SPCC was 0.462 (0.466) for peptide-similarity at threshold 0.5, 0.669 (0.707) for within-target, and 0.530 (0.524) for leave-target-out evaluation (Figures 2 and 3). Relative to within-target prediction, median losses were 0.224 and 0.160 for peptide-similarity and leave-target-out; leave-target-out exceeded peptide-similarity by 0.067. All contrasts remained significant across the evaluated configurations after Holm adjustment (two-sided Wilcoxon tests, adjusted $p < 0 . 0 0 1 )$ .

The top configuration also changed. ECFP-16 count fingerprints with random forest led peptide-similarity (0.545) and within-target evaluation (0.771), whereas HELM-BERT with Extra Trees led leave-target-out evaluation (0.605). Thus, the selected configuration was partition dependent; these maxima do not establish intrinsic superiority of either representation family beyond the models evaluated here.

![](images/f800ae069b9819bc849043b69d2575dc0cc4f0de577651da7f1dfe901cc2b3b2.jpg)  
Figure 2. Model-performance landscapes under three definitions of generalization. Each cell reports five-seed mean full-data test SPCC for one peptide representation and downstream regressor under (A) peptide-similarity at threshold 0.5, (B) within-target, or (C) leave-target-ou evaluation. The row and column order is identical across panels, and the color scale is shared.

Complete-configuration rank correlations were 0.860 for peptide-similarity versus within-target, 0.801 for peptide-similarity versus leave-target-out, and 0.795 for within-target versus leave-target-out. In the same order, representation-rank correlation fell to 0.103, 0.624, and −0.042, whereas regressor-rank correlations remained 0.886, 0.771, and 0.943. Partition choice therefore affected representation rankings more strongly than regressor rankings.

![](images/e458feae1b719cb342a6aa9857b7c5522fe5ac65505d111286badfa3684f8c46.jpg)  
Figure 3. Distribution of full-data test SPCC across peptide representations under (A) peptide-similarity partitioning at threshold 0.5, (B) within-target evaluation, and (C) leave-target-out evaluation. For each representation, the half violin summarizes the six regressor values and the horizontal bar marks their median; the corresponding regressor points are shown alongside. Each marker is the five-seed mean for one representation–regressor combination; color identifies the peptide representation and marker shape identifies the regressor.

Tree ensembles occupied the upper part of each distribution but showed larger within-target to leave-target-out differences (Figure 3). Random forest, Extra Trees, MLP, and LightGBM lost 0.179, 0.177, 0.176, and 0.170 mean SPCC. SVR and ridge lost 0.067 and 0.065, although their lower within-target baselines limit interpretation of the smaller loss. Representation-level losses ranged from 0.122 (MolFormer-XL) and 0.123 (PepTune) to 0.164 (ECFP-16) and 0.165 (PepFuNN).

## 3.2. Additional supervision narrows representation differences but not learner differences

Learning curves used five pretrained peptide representations, all six regressors, and fixed validation and test sets under peptide-similarity threshold 0.5. Training sizes were 100, 300, 1,000, 3,000, 6,000, and all 8,172 available records (Figure 4).

Extra Trees, random forest, and LightGBM achieved the highest full-data means: 0.526, 0.518, and 0.513 SPCC. Their 100- record values were 0.186, 0.170, and 0.150. All three tree models continued to improve beyond 1,000 records (Figure 4A,C). From 100 records to all 8,172 records, their absolute SPCC gains were approximately 0.34–0.36. By comparison, MLP rose from 0.099 to 0.430, ridge from 0.168 to 0.424, and SVR from 0.151 to 0.403; the latter two flattened more clearly at larger sizes.

HELM-BERT had the highest representation mean at every displayed size (Figure 4B,D), with its clearest advantage under low-data conditions. At 100 records, SPCC was 0.210 for HELM-BERT, 0.155 for PepLand, 0.156 for PeptideCLM-2, 0.152 for PeptideCLM, and 0.099 for PepTune. Full-data values were 0.488, 0.455, 0.475, 0.462, and 0.464, reducing the best–worst gap from 0.111 to 0.033. PepTune gained the most (0.366), while HELM-BERT gained 0.279 from a stronger baseline. Among the five pretrained peptide representations, representation choice had its largest effect with limited data and became less influential as supervision increased; by contrast, substantial differences among downstream learners remained at larger training sizes.

![](images/c14750aaf5d151f8fad0d12f0d7ba7faac43d3a3156f763a7df9d7089173c7ff.jpg)

B  
![](images/08ac40b9e2ecc1c50372786570d7217511cd0fc2fb77976c0822b1052424caed.jpg)

C  
![](images/bed006ac18b9dc93ea38a68ac215205030656078c567700070e99e77533b1b17.jpg)

D  
![](images/6c48ee2b3da8a9d8c79e2112deee22de915315f20da7a1c511cafa939d0e8f25.jpg)  
Figure 4. Effect of training-set size on peptide–protein affinity prediction under peptide-similarity partitioning at threshold 0.5. (A) Regressorspecific curves averaged across five peptide representations, with Extra Trees highlighted as the highest-performing regressor at each shown size. (B) Representation-specific curves averaged across six regressors, with HELM-BERT highlighted as the highest-performing representation at each shown size. (C, D) Horizontal scaling traces for all regressors and peptide representations, respectively; marker size increases with training-set size. The display covers 100–6,000 records; full-data results for 8,172 records are reported in the text. Panels show descriptive means without uncertainty intervals. In panel D, PepCLM and PepCLM-2 denote PeptideCLM and PeptideCLM-2, respectively

## 3.3. PeptideCLM-2 adaptation is less influential than training-set size

We compared frozen PeptideCLM-2 hybrid-small with Last-1, Last-2, Last-4, full fine-tuning, and LoRA across the same partition and training sizes (Figure 5).

All strategies improved with more records, but differences among strategies were small relative to the training-size effect, and their relative ordering was unstable across data sizes. Between-replicate variability was greatest under low-data conditions and generally decreased as training size increased. Unfreezing one or two final blocks was competitive at several sizes; Last-4, full fine-tuning, and LoRA showed no systematic gain. At 6,000 records, mean SPCC ranged from 0.452 to 0.484 without a monotonic relation between the number of trainable blocks and performance.

The AULC comparison did not provide evidence of an overall difference among strategies (Friedman test, $\chi ^ { 2 } = 6 . 6 2 .$ $p = 0 . 2 5 1 )$ . With only three paired subsampling replicates, this test had limited sensitivity to small differences; the numerical increase across training sizes was nevertheless much larger than the separation among adaptation strategies.

B  
![](images/6ab3f9790705f1c98429485448129e5b47ededff6e6b952906e3bfd8550ec2d2.jpg)  
Figure 5. PeptideCLM-2 hybrid-small adaptation across training-set sizes. Test SPCC is shown for a frozen encoder, partial unfreezing of the final one, two, or four transformer blocks, full fine-tuning, and LoRA. All strategies used matched peptide-similarity partitions at threshold 0.5 and three subsampling replicates; error bars denote the between-replicate standard deviation, and opacity increases with training-set size. The display is limited to 100–6,000 records; the 8,172-record point was included in AULC but falls outside the plotted range.

## 3.4. Explicit pairwise features provide no consistent gain

To test whether explicit peptide–protein interaction features improved prediction, direct concatenation was compared with interaction-aware fusion containing the element-wise product and absolute difference of projected peptide and protein features. Both used five matched seeds at peptide-similarity thresholds 0.1–1.0 (Figure 6).

![](images/ee13d8fdffbcdae33a5e91f32b1ced4cb9cea131d681c2070419c9a6eccce099.jpg)

![](images/04b6c26c6b44b12caef283f2e69ff3755d1d23267c45782bdcff08507c013259.jpg)  
Figure 6. Comparison of peptide–protein feature-fusion rules across peptide-similarity thresholds. (A) Test SPCC for direct concatenation and interaction-aware fusion. (B) Matched difference $\Delta \mathrm { S P C C } = \mathrm { S P C C } _ { \mathrm { i n t e r a c t i o n } } - \mathrm { S P C C } _ { \mathrm { c o n c a t e n a t i o n } } ;$ the horizontal line marks equal performance. Shaded ribbons in A and vertical intervals in B show ±1 standard deviation across five matched seeds

The performance curves were highly similar throughout the threshold range: concatenation was higher at seven thresholds and interaction-aware fusion at three. The paired contrast

$$
\Delta \mathrm { S P C C } = \mathrm { S P C C } _ { \mathrm { i n t e r a c t i o n } } - \mathrm { S P C C } _ { \mathrm { c o n c a t e n a t i o n } }\tag{7}
$$

ranged from −0.0079 at threshold 0.1 to +0.0061 at 0.7, with no persistent advantage.

After averaging across thresholds within each seed, concatenation achieved $0 . 4 5 2 5 \pm 0 . 0 0 1 8 \mathrm { S P C C }$ versus $0 . 4 5 1 1 \pm 0 . 0 0 4 4$ for interaction-aware fusion (mean ± standard deviation across five seeds; $\Delta = - 0 . 0 0 1 4 )$ . Interaction-aware fusion achieved

the higher SPCC in 21 of the 50 matched threshold–seed comparisons. Thus, under the present model and data settings, adding element-wise products and absolute differences provided no consistent predictive gain.

## 4. Discussion

Within this benchmark, the same 60 model configurations produced different accuracies, selected configurations, and representation rankings when the held-out entity changed. Mean SPCC was lowest under peptide-similarity partitioning, highest within targets, and intermediate when exact target sequences were excluded. This ordering should not be interpreted as an intrinsic comparison of peptide and target novelty because the partitions differ in composition and constraints. More generally, model performance cannot be attributed to the peptide encoder alone: evaluation regime, training-data scale, and downstream learner must be considered jointly.

These findings extend the chemically informed peptide evaluation of Fernández-Díaz et al. [22]. ECFP-16 count fingerprints with random forest led peptide-similarity and within-target evaluation, whereas HELM-BERT with Extra Trees led leavetarget-out evaluation. Peptide-side structural novelty and target-side novelty therefore pose different generalization challenges and may favor different representation–learner combinations. The partitions map to different uses: within-target evaluation to follow-up peptide optimization or candidate ranking around an observed protein target, peptide-similarity evaluation to transfer toward new candidates chemically separated from the training peptides, and leave-target-out evaluation to protein targets absent from training. None is universally most realistic, and a single partition can conceal consequential model-selection changes [19–21, 23, 42].

Related studies predict continuous affinity, motif–domain binding, binary interactions, or residue-level binding sites [9–13]. Our benchmark is not a leaderboard comparison with these systems: it compares fixed representations within a unified quantitative affinity-prediction framework under controlled peptide and target shifts. Interaction-classification performance therefore cannot be assumed to imply accurate affinity ranking for an unseen target.

Data scale altered representation and learner comparisons differently. Among the five pretrained peptide representations, performance differences were largest under low-data conditions and narrowed as supervision increased, whereas tree ensembles continued to improve at larger training sizes. This agrees with the competitiveness of conventional chemical fingerprints in peptide prediction and the sensitivity of low-data chemical modeling to sampling and partition design [43, 44]. It also cautions against evaluating an encoder with only one learner or at one data scale, which can misattribute learner capacity and representation–learner interactions to the representation itself. Full fine-tuning and LoRA likewise did not consistently improve the PeptideCLM-2 learning curves. Although heterogeneous binding labels may have limited reliable estimation of the added parameters, that explanation was not tested directly. Under the tested protocol, increasing supervision was associated with the more consistent numerical improvement.

The feature-fusion finding is more narrowly bounded. Element-wise products and absolute differences expose simple pairwise relations but neither model residue-level contact geometry nor allow one partner to condition the other. Their lack of a consistent gain does not exclude cross-attention, residue-pair networks, bilinear coupling, or complex-structure models that may describe peptide–protein interactions more effectively [11–13]. It shows only that, within the present independentencoding and MLP prediction framework, expanding the fusion representation with element-wise products and absolute differences did not reliably improve prediction.

Several limitations define the scope of inference. Combining $K _ { d } , K _ { i }$ , and $I C _ { 5 0 } \mathrm { o n o n e - l o g _ { 1 0 } ( M ) }$ scale leaves assay and endpoint heterogeneity unmodeled. Leave-target-out excludes identical sequences but not homologous proteins, so it does not establish transfer across families or folds; within-target evaluation also emphasizes targets with enough records to populate held-out subsets. MAPc and threshold 0.5 define only one form of peptide novelty, and the merged dataset is dominated by canonical peptides. Primary results emphasize ranking by SPCC rather than calibration or endpoint-specific error, and no independent external dataset was used. Figure 4 learning-curve comparisons are descriptive and do not include uncertainty intervals. Finally, scenario tests treat model configurations as analysis units and adaptation uses only three subsampling blocks, so the reported p-values do not quantify uncertainty across independently sampled datasets. Sequence- and chemistry-derived features also omit resolved complex geometry; PepBDB could support a smaller structure-aware benchmark [15].

Future work should combine homology-aware target and alternative peptide-structure partitions, incorporate assay metadata, validate on independent sources, and test jointly trained partner-aware architectures. Learning curves should remain part of evaluation. The central conclusion is that representation superiority is conditional on the intended generalization regime; credible affinity benchmarks should align partitions with use, compare multiple learners, and report sensitivity to data availability.

## References

[1] Markus Muttenthaler et al. Trends in peptide drug discovery. Nature Reviews Drug Discovery, 20(4):309–325, 2021.

[2] Lei Wang et al. Therapeutic peptides: current applications and future directions. Signal Transduction and Targeted Therapy, 7(1):48, 2022.

[3] Duncan E. Scott et al. Small molecules, big targets: drug discovery faces the protein–protein interaction challenge. Nature Reviews Drug Discovery, 15(8):533–550, 2016.

[4] Suzanne P. van Wier and Andrew M. Beekman. Peptide design to control protein–protein interactions. Chemical Society Reviews, 54 (4):1684–1698, 2025.

[5] Bingyi Zheng et al. Therapeutic peptides: recent advances in discovery, synthesis, and clinical translation. International Journal of Molecular Sciences, 26(11):5131, 2025.

[6] Jennifer L. Hickey et al. Beyond 20 in the 21st century: prospects and challenges of non-canonical amino acids in peptide drug discovery. ACS Medicinal Chemistry Letters, 14(5):557–565, 2023.

[7] Tarsila G. Castro et al. Non-canonical amino acids as building blocks for peptidomimetics: structure, function, and applications Biomolecules, 13(6):981, 2023.

[8] Chen Deng et al. Engineering protease-resistant peptides via non-canonical amino acids: design strategies and biosynthetic advances. Bioengineering, 13(7):767, 2026.

[9] Sébastien Giguère, Mario Marchand, François Laviolette, Alexandre Drouin, and Jacques Corbeil. Learning a peptide–protein binding affinity predictor with kernel ridge regression. BMC Bioinformatics, 14:82, 2013.

[10] Joseph M. Cunningham, Grigoriy Koytiger, Peter K. Sorger, and Mohammed AlQuraishi. Biophysical prediction of protein–peptide interactions and signaling networks using machine learning. Nature Methods, 17(2):175–183, 2020.

[11] Yipin Lei, Shuya Li, Ziyi Liu, Fangping Wan, Tingzhong Tian, Shao Li, Dan Zhao, and Jianyang Zeng. A deep-learning framework for multi-level peptide–protein interaction prediction. Nature Communications, 12(1):5465, 2021.

[12] Osama Abdin, Satra Nim, Han Wen, and Philip M. Kim. PepNN: a deep attention model for the identification of peptide binding sites. Communications Biology, 5(1):503, 2022.

[13] Shizhuo Li et al. PepBAN: a deep learning framework with bilinear attention and adversarial learning for peptide–protein interaction prediction. Journal ofChemical Information and Modeling, 65(17):9061–9074, 2025.

[14] Song Yin, Xuenan Mi, and Diwakar Shukla. Leveraging machine learning models for peptide–protein interaction prediction. RSC Chemical Biology, 5(5):401–417, 2024.

[15] Zeyu Wen, Jiahua He, Huanyu Tao, and Sheng-You Huang. PepBDB: a comprehensive structural database of biological peptide–protein interactions. Bioinformatics, 35(1):175–177, 2019.

[16] Ning Zhu et al. PPIKB: a comprehensive knowledge base and analysis platform for protein–peptide interactions based on literature and patents. bioRxiv, 2025.

[17] Robert P. Sheridan. Time-split cross-validation as a method for estimating the goodness of prospective prediction. Journal ofChemica Information and Modeling, 53(4):783–790, 2013.

[18] Izhar Wallach and Abraham Heifets. Most ligand-based classification benchmarks reward memorization rather than generalization. Journal ofChemical Information and Modeling, 58(5):916–932, 2018.

[19] Ian Walsh et al. DOME: recommendations for supervised machine learning validation in biology. Nature Methods, 18(10):1122–1127, 2021.

[20] Felix Teufel et al. GraphPart: homology partitioning for biological sequence analysis. NAR Genomics and Bioinformatics, 5(4): lqad088, 2023.

[21] Prudencio Tossou et al. Real-world molecular out-of-distribution: specification and investigation. Journal of Chemical Information and Modeling, 64(3):697–711, 2024.

[22] Raúl Fernández-Díaz et al. How to build machine learning models able to extrapolate from standard to modified peptides. Journal of Cheminformatics, 17(1):185, 2025.

[23] Raúl Fernández-Díaz et al. A new framework for evaluating model out-of-distribution generalisation for the biochemical domain. In International Conference on Learning Representations, 2025.

[24] David Rogers and Mathew Hahn. Extended-connectivity fingerprints. Journal ofChemical Information and Modeling, 50(5):742–754, 2010.

[25] Rodrigo Ochoa and Kristine Deibler. PepFuNN: Novo Nordisk open-source toolkit to enable peptide in silico analysis. Journal of Peptide Science, 31(2):e3666, 2025.

[26] Walid Ahmad et al. ChemBERTa-2: towards chemical foundation models. arXiv preprint arXiv:2209.01712, 2022.

[27] Jerret Ross et al. Large-scale chemical language representations capture molecular structure and properties. Nature Machine Intelligence, 4(12):1256–1264, 2022.

[28] Seungeon Lee et al. HELM-BERT: a transformer for medium-sized peptide property prediction. arXiv preprint arXiv:2512.23175, 2025.

[29] Aaron L. Feller and Claus O. Wilke. Peptide-aware chemical language model successfully predicts membrane diffusion of cyclic peptides. Journal ofChemical Information and Modeling, 65(2):571–579, 2025.

[30] Aaron L. Feller et al. Scaling SMILES-based chemical language models for therapeutic peptide engineering. Journal ofChemical Information and Modeling, 66(14):8110–8122, 2026.

[31] Sophia Tang, Yinuo Zhang, and Pranam Chatterjee. PepTune: de novo generation of therapeutic peptides with multi-objective-guided discrete diffusion. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 59017–59065, 2025.

[32] Ruochi Zhang et al. PepLand: a large-scale pre-trained peptide representation model for a comprehensive landscape of both canonical and non-canonical amino acids. Briefings in Bioinformatics, 26(4):bbaf367, 2025.

[33] Zeming Lin et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637): 1123–1130, 2023.

[34] Corinna Cortes and Vladimir Vapnik. Support-vector networks. Machine Learning, 20(3):273–297, 1995.

[35] Guolin Ke et al. LightGBM: a highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems, volume 30, 2017.

[36] Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001.

[37] Arthur E. Hoerl and Robert W. Kennard. Ridge regression: biased estimation for nonorthogonal problems. Technometrics, 12(1): 55–67, 1970.

[38] Pierre Geurts, Damien Ernst, and Louis Wehenkel. Extremely randomized trees. Machine Learning, 63(1):3–42, 2006.

[39] David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representations by back-propagating errors. Nature, 323 (6088):533–536, 1986.

[40] Takuya Akiba et al. Optuna: a next-generation hyperparameter optimization framework. In Proceedings ofthe 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019.

[41] Edward J. Hu et al. LoRA: low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[42] Yasha Ektefaie et al. Evaluating generalizability of artificial intelligence models for molecular datasets. Nature Machine Intelligence, 6(12):1512–1524, 2024.

[43] Jakub Adamczyk, Piotr Ludynia, and Wojciech Czech. Molecular fingerprints are strong models for peptide function prediction. Bioinformatics, 42(5):btag179, 2026.

[44] Gary Tom et al. Calibration and generalizability of probabilistic models on low-data chemical datasets with DIONYSUS. Digital Discovery, 2:759–774, 2023.