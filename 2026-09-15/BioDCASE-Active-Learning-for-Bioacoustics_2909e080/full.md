# BioDCASE: Active Learning for Bioacoustics

Ben McEwen<sup>1</sup> , Rupa Kurinchi-Vendhan<sup>2</sup> , and Shiqi Zhang<sup>3</sup> Lukas Rauch<sup>4</sup> Marek Herde<sup>5</sup> Sara Beery<sup>2</sup>

<sup>1</sup> University of Amsterdam, Amsterdam, Netherlands

<sup>2</sup> Massachusetts Institute of Technology, Cambridge, Massachusetts, USA

<sup>3</sup> Tampere University, Finland

Earth Species Project

<sup>5</sup> Kassel University, Germany

Abstract. Ecological monitoring increasingly relies on machine learning models, whose performance depends on the quality and quantity of labelled data. However, obtaining these labels is costly, particularly in passive acoustic monitoring, where vast amounts of data are collected but only a small proportion can feasibly be annotated. Active learning addresses this bottleneck by prioritizing which samples should be labelled. However, progress is dificult to measure, because published methods are evaluated under diferent models, budgets, evaluation metrics and datasets. To address this challenge, we present the 2026 Active Learning for Bioacoustics BioDCASE challenge: a systematic evaluation of sampling methods designed to identify efective AL strategies. Participant methods were evaluated across four subsets composed of terrestrial and marine data. Across ten proposed sampling methods from seven teams, the top-ranked method achieved an area under the learning curve 26.4 % higher than random sampling at the same annotation budget, averaged over four data subsets. Significant variation in performance was observed across subsets, with the top-performing submission achieving a 67.1 % gain for the HSN subset over random sampling and a gain of 8 % for the ATBFL subset. Top-ranking submissions combined multiple acquisition signals, and diversity-based selection outperformed pure uncertainty sampling. Furthermore, there is evidence that transitioning from diversity-based to uncertainty-based selection and explicitly reducing redundancy within acquisition batches improve model training. There is also initial evidence that larger acquisition batch sizes may be increasingly beneficial later in the labelling process.

Keywords: Active Learning · Benchmarking · Bioacoustics · Data Challenge · Machine Learning

## 1 Introduction

A fundamental challenge across bioacoustics, in terrestrial and marine domains alike, is the cost of annotation. Passive acoustic monitoring (PAM) systems generate vast amounts of data [8], but only a small proportion can be feasibly annotated by human experts. Since model performance depends heavily on the quality and quantity of labelled data, this raises the following question: Given vast amounts of raw acoustic data and limited annotation resources, which data should be prioritised for labelling?

Active learning (AL) is an iterative process of model-guided sample selection and label acquisition that aims to maximise model performance with fewer labelled samples than random (passive) sampling [39]. Practical constraints on annotator availability and computation mean that AL for bioacoustics is generally applied in batch mode over a pool of unlabelled recordings: an acquisition function defines a notion of utility for unlabelled instances, which informs batch selection before the model is retrained. Acquisition function design must therefore balance the informativeness of individual instances against redundancy within the selected batch. Increasingly, this selection operates not on audio but on embeddings from a frozen, pretrained acoustic (foundation) model such as BirdNET [17], Bird-MAE [33], and PerchV2 [27], with only a lightweight classification head trained during the AL loop. This active fine-tuning regime is increasingly common in bioacoustics, rather than end-to-end fine-tuning [11, 18, 25, 35]. Bioacoustic applications additionally present severe class imbalance, sparse and rare events, and domain shift across spatial and temporal gradients [36].

The BioDCASE challenge [41] is an annual bioacoustics data challenge, within the long-running format of the DCASE (Detection and Classification of Acoustic Scenes and Events) challenge [28,40]. The 2026 BioDCASE challenge comprised six concurrent tasks, including AL for Bioacoustics as Task 4. The challenge consists of a development phase, during which the participants are provided development datasets, instructions, and baseline methods from which to develop their method, followed by an evaluation and reporting phase where participant submissions are ranked.

Growing interest in AL methods within bioacoustics and biodiversity monitoring creates a need for standardised evaluation across datasets, domains, and metrics. Evaluation should also consider practical objectives such as computational and annotation cost. The challenge design including dataset and evaluation metric selection is reported in Section 2. The challenge provides a consistent test suite for evaluation of AL methods through the development and use of BaseAL (discussed in Section 2.3). The format also provides an accessible entry-point to AL methods for non-practitioners.

Contributions. We report the first edition of the 2026 AL for Bioacoustics task. We discuss the challenge design for terrestrial and marine bioacoustics, and the development of two curated AL datasets publicly available on Zenodo [21, 34]. We report results for 14 sampling methods, including ten submissions from seven teams and four baselines. We analyse sampling methods on held-out test sets and identify common characteristics among the top-performing submissions, including combining multiple acquisition signals, shifting from diversity-based to uncertainty-based selection over the course of sampling, and explicitly reducing redundancy within acquisition batches. Additionally, we discuss broader considerations for future iterations of the challenge, such as the choice of pre-trained model and datasets, and balancing evaluation across broader objectives such as computational and annotation cost.

## 2 Challenge Design

The BioDCASE challenge attract a heterogeneous participant base with students, early-career researchers, domain experts and newcomers from across domains (e.g. ecology, acoustics, computer science). The challenge design, discussed here, must balance the flexibility and evaluation rigor with accessibility.

## 2.1 Datasets

The challenge includes both terrestrial and marine datasets to assess whether AL methods generalize across distinct bioacoustic monitoring settings. Despite diferences in species, recording conditions, and acoustic structure, both domains share the challenge of selecting informative samples under limited annotation budgets. Table 1 provides a corresponding overview of our challenge’s datasets.

BirdSet. The three terrestrial datasets are derived from the labelled PAM test subsets of BirdSet [34, 36]: High Sierra Nevada (HSN), Powdermill Nature Reserve (POW), and Hawai’i Island (UHH). HSN [6] originates from recordings collected in 2015 around ten high-elevation lakes in Sequoia and Kings Canyon National Parks, California, POW [4] from dawn chorus recordings collected in 2018 at Powdermill Nature Reserve in Pennsylvania, and UHH [31] from recordings acquired at four sites on Hawai’i Island between 2016 and 2022. In the BirdSet benchmark, models are trained on weakly labelled focal recordings and evaluated on strongly annotated soundscapes, introducing a domain shift between training and evaluation data. For this challenge, we instead derive new train, validation, and test splits exclusively from the soundscape recordings. The resulting setting is therefore not directly comparable to the BirdSet benchmark, but more closely reflects an AL scenario in which unlabelled and labelled samples originate from the same PAM domain.

BirdSet standardized the source recordings to 32kHz, aligned species labels using the eBird taxonomy, and divided the soundscapes into 5s segments. Each segment was propagated through PerchV2 [27], resulting in a 1,536-dimensional embedding. Species represented by fewer than three positive segments were removed because they could not be included in all three data splits, reducing the target space from 21 to 19 classes for HSN and from 48 to 41 for POW, while UHH retained all 25 classes. For each of the three datasets, we generated multiple multilabel-aware candidate splits and selected a splitting in which every retained class occurred in the train, validation, and test splits while keeping label distributions comparable. Splits were constructed at the level of individual 5s segments rather than complete source recordings. Hence, segments from the same source recording can occur in diferent splits.

Acoustic Trends Blue Fin Library (ATBFL). ATBFL, the representative marine dataset for the challenge, is an annotated library of PAM recordings collected around Antarctica between 2005 and 2017, designed to support the development and evaluation of automated detectors for Antarctic blue whales (Balaenoptera musculus intermedia) and fin whale (Balaenoptera physalus) vocalizations [29]. To capture diverse geographic, temporal, and instrumentation settings, recordings span 11 site–year deployments across the Atlantic, Pacific, and Indian sectors of the Southern Ocean and the Western Antarctic Peninsula [29].

For this challenge, we converted the strongly-annotated ATBFL recordings released for BioDCASE 2025 Task 2 into a segment-level multilabel classification dataset [16]. The original annotations specify the call type and start and end time of each vocalization, and the absolute timestamps were first converted to ofsets relative to the start of each recording. Recordings were then divided into consecutive, non-overlapping 5-second windows, and passed as input to PerchV2 [27] to generate input representations for the challenge. Strong event annotations were converted to window-level multilabel targets by assigning each segment all call types whose annotated intervals overlapped with the 5-second window, and segments with no overlapping call were treated as negative examples.

The resulting dataset contains 19,633 audio samples, corresponding to approximately 21.7 hours of audio [21]. Samples preserve the site–year metadata and are annotated in a multilabel setting (multiple call types to co-occur within a sample) for seven blue- and fin- whale call types: Antarctic blue whale A (bma), AB/B (bmb), and Z (bmz) stereotyped calls, blue whale frequency-modulated Dcalls (bmd), and fin whale downsweeps (bpd), 20-Hz pulses (bp20), and 20-Hz pulses with additional higher-frequency energy (bp20plus) [29]. Of the 19,633 samples, 15,667 (79.8%) contain at least one target call, with substantial diferences in prevalence across call types [21]. Table 1 summarizes the train, validation, and test splits for this dataset.

Table 1: Statistics of our challenge’s datasets after preprocessing.
<table><tr><td>Subset</td><td>Segments</td><td>Classes</td><td>Labels per Segment</td><td>Train/Validation/Test</td></tr><tr><td>HSN</td><td>12,000</td><td>19</td><td>0.52</td><td>6,600/1,800/ 3,600</td></tr><tr><td>POW</td><td>4,560</td><td>41</td><td>2.83</td><td>2,280/ 684/1,596</td></tr><tr><td>UHH</td><td>36,637</td><td>25</td><td>1.05</td><td>18,319/7,327/10,991</td></tr><tr><td>ATBFL</td><td>19,633</td><td>7</td><td>2.26</td><td>12,506/3,127/4,000</td></tr></table>

## 2.2 Representations and Model

Participants received pre-computed PerchV2 [27] embeddings (generated using bacpipe [19]) rather than the raw audio, matching the active fine-tuning regime increasingly common in bioacoustics. PerchV2 was selected for its common usage within the bioacoustics community, benchmarking on marine mammal and underwater acoustics [3], and competitive performance relative to other pretrained models [37]. This approach kept the datasets compact and the pipeline runnable on consumer-grade hardware without HPC access. The implications of this design are discussed in Section 4.4.

The classification head was fixed to a two-layer MLP (1536-dimension projection layer and output layer based on class count) to prevent trivial improvements unrelated to AL such as increasing the number of model parameters.

## 2.3 BaseAL: Active Learning Framework

BaseAL is an AL framework designed for the BioDCASE Active Learning for Bioacoustics challenge [26]. BaseAL provides a general AL pipeline split into modular components, user-friendly notebooks, and an accompanying web-application for visualising AL cycles in the embedding space and comparing sampling methods. Figure 1 shows the AL cycle and which components were fixed and which components participants could edit.

![](images/1e3af15b1f0f507fffc274c9943bd7a40e9d7b82c1099c67083b1ad0604f9e15.jpg)  
\*Repeated until budget exhausted

Fig. 1: Active learning loop showing fixed and editable components.

At each AL cycle the participant acquisition function received the current model predictions, the embeddings, dataset-specific metadata and the indices of the labelled and unlabelled sets. The function returns a utility score for each unlabelled sample. The highest-scoring samples were then revealed by oracle labelling, added to the labelled set and the classification head is retrained. This repeated until the annotation budget was exhausted.

Participant flexibility was bounded. Participants could edit the acquisition function and an optional warm-up function used for the first cycle before any labels are available. They could additionally set the AL batch size, and therefore the number of AL cycles. The core loop and experiment manager, the model, and the embeddings were fixed. The options to vary batch size and cycle counts afects computational cost, which is reported as a supplementary metric (Section 2.4).

## 2.4 Evaluation Protocol

The first edition of the challenges only ranks participants on model training eficiency. The primary ranking metric was Area Under the Learning Curve (AULC) of macro-averaged average precision (mAP). Learning curves were computed for participant submissions up to the maximum budget, submissions with a higher AULC mAP achieved a higher model performance within the budget. The final score was averaged over each of the four test data subsets, meaning that to achieve a high overall ranking, submissions must generalise (e.g. across locations or domains, marine vs. terrestrial). Test splits use the same stratified, label-aware procedure as train/validation (Section 2.1), so dificulty and label prevalence are comparable across splits.

A major consideration when applying AL is computational and annotation cost. For transparent ranking, these considerations were not included in the ranking score however they were reported as supplementary metrics. Computational cost was measured as the sampling time (s), the number of seconds on average require to assign utility scores and select samples. Additionally the computational cost cost\_method = model\_parameters ∗ epochs ∗ AL\_cycles relative to the baseline configurations relative\_cost = cost\_method/baseline\_cost. Participants could opt to increase the number of AL cycles (reduce the AL batch size) or increase the number of model training epochs. For large unlabelled datasets,repeated AL cycles can be extremely costly, leading to larger batch sizes. When using oracle labelling, annotation cost/time cannot be measured directly. Instead a proxy of the total number of labels per samples was used. The intuition was that audio segments with more vocalising species would be more time-consuming to annotate. All metrics were averaged over subsets and over five independent repeats.

## 2.5 Baselines and Configuration

Four baseline methods were provided to participants: random, margin (multilabel), CoreSet [38] and TypiClust [14]. Random sampling (passive sampling)

simulates model training if active learning had not been applied. Margin is a common uncertainty sampling method. In the multilabel setting, for each sample, the distance of each class probability from 0.5 (highest uncertainty) is averaged across classes and the samples with the smallest margin are selected. Margin represents a pure uncertainty-based sampling method, while CoreSet is a pure diversity criterion without explicit consideration of prediction uncertainty. Core-Set applies a k-center (farthest-first) selection in an embedding space, seeded by the labelled set, to minimize the maximum distance of unlabelled samples to the selected samples. TypiClust is also a diversity-based method which selects typical samples (i.e. highest local density) across clusters.

A fixed batch size of 50 samples, with 10 model training epochs per AL cycle and a learning rate of 1e-3 was used for all baselines, up to the maximum budget of 500 samples for each of the four data subsets. The maximum budget was selected by manually testing where model performance was starting to plateau for each subset. For simplicity this 500 samples was selected for each of the subsets. This budget simulates a realistic labelling budget. All baseline results and participant results were averaged over five independent runs and the baseline results are reported in Table 2 along with supplementary metrics. Core-Set achieved the highest baseline performance with an AULC (mAP) of 0.460, compared to passive sampling which achieved a score of 0.390.

Table 2: Baseline sampling method performance on the development set, aggregated across all subsets. All baselines use the reference configuration (batch size 50, 10 epochs per cycle, learning rate $1 0 ^ { - 3 }$ , budget 500 samples). Best AULC in bold.
<table><tr><td>Method</td><td colspan="4">AULC (mAP) Comp. cost Sampling time (s) Annotation cost</td></tr><tr><td>Random</td><td>0.390</td><td>1.0</td><td>0.00131</td><td>966.85</td></tr><tr><td>Margin</td><td>0.399</td><td>1.0</td><td>0.00307</td><td>1267.75</td></tr><tr><td>CoreSet</td><td>0.460</td><td>1.0</td><td>5.04364</td><td>1019.05</td></tr><tr><td>TypiClust</td><td>0.423</td><td>1.0</td><td>5.90751</td><td>958.10</td></tr></table>

## 2.6 Submissions

Submissions followed the general BioDCASE submission format [41], with some modifications specific to the challenges setting. Rather than submitting model predictions, participants submitted the file containing their acquisition function and warm-up function in some cases. This enabled evaluation of the acquisition function on a held-out test set and that the supplementary evaluation metric, sampling time, could be compared on the same hardware. For reproducibility, participants also provide a .yaml file containing their submissions output on the development set. The development results were reported to a running leaderboard during the development phase of the challenge. Participants were also required to include a short technical report (max. 5 pages), summarising their method and design consideration.

Optionally, participants could note any additional dependencies not already included in BaseAL. An optional notebook was also recommended for participants using non-default configurations or AL batchsize schedulers. As per the evaluation instructions, both participant-generated results and organisergenerated results on the test set were repeated over five independent runs, and the average was reported. Participant submissions were uploaded to the BioD-CASE organiser managed Microsoft Conference Management Toolkit (CMT).

## 3 Results

Fourteen sampling methods were ranked, comprised of ten participant submission and four baseline methods. Submissions were received from 12 academic institutes and companies spanning nine countries. Here we report the overall ranking (Section 3.1), performance across the four subsets (Section 3.2) and the awarded submissions (Section 3.3). The oficial results are available on the BioDCASE website.

## 3.1 Overall Ranking

Note that the following results are reported on the held out test dataset. Reported results will difer from those reported in the participant technical reports.

Table 3: Task 4 submission ranking. Score is the AULC of macro mAP averaged across the four evaluation subsets. Eficiency metrics are reported but not used for ranking. Baseline entries are italicised.
<table><tr><td>Rank Authors</td><td></td><td>System</td><td>Report Score Cost Sampling</td><td></td><td></td><td>g (s) Labels</td></tr><tr><td>1</td><td>Dubus et al.</td><td>ADU-MMR</td><td>[10]</td><td>0.507</td><td>2.0</td><td>1.006 1026.7</td></tr><tr><td>2</td><td>Magaldi et al.</td><td>CARE-DPP</td><td>[24]</td><td>0.505 1.1</td><td>0.215</td><td>1124.6</td></tr><tr><td>3</td><td>Wang et al.</td><td>PB-MFS</td><td>[42]</td><td>0.499 1.6</td><td>0.485</td><td>1109.9</td></tr><tr><td>4</td><td>Yang et al.</td><td>Safe Rarity</td><td>[43]</td><td>0.492 1.0</td><td>0.385</td><td>1116.0</td></tr><tr><td>5</td><td>Nihal et al.</td><td>Capped Rarity K-Center</td><td>[32]</td><td>0.481 0.9</td><td>0.128</td><td>1134.5</td></tr><tr><td>6</td><td>Garcia-Yi, J.</td><td>AFL</td><td>[13]</td><td>0.477 1.6</td><td>1.333</td><td>1031.5</td></tr><tr><td>7</td><td>Parcerisas et al. Coreset KMeans</td><td></td><td>[2]</td><td>0.469 1.0</td><td>4.095</td><td>1012.3</td></tr><tr><td>8</td><td>Baseline</td><td>CoreSet</td><td>=</td><td>0.464 1.0</td><td>5.838</td><td>1023.6</td></tr><tr><td>9</td><td></td><td>Parcerisas et al. Coreset Eigenvalues</td><td>[2]</td><td>0.435 0.8</td><td>7.800</td><td>1010.6</td></tr><tr><td>10</td><td>Baseline</td><td>TypiClust</td><td></td><td>0.421 1.0</td><td>5.076</td><td>959.0</td></tr><tr><td>11</td><td>Baseline</td><td>Margin</td><td></td><td>0.408 1.0</td><td>0.002</td><td>1263.7</td></tr><tr><td>12</td><td>Baseline</td><td>Random</td><td>1</td><td>0.401 1.0</td><td>0.001</td><td>966.1</td></tr><tr><td>13</td><td></td><td>Parcerisas et al. All Quantiles KMeans</td><td>[2]</td><td>0.397 1.0</td><td>0.330</td><td>992.1</td></tr><tr><td>14</td><td></td><td>Parcerisas et al. Balance Class (Eigen.)</td><td>[2]</td><td>0.393 0.8</td><td>0.110</td><td>1039.0</td></tr></table>

## 3.2 Per-Subset Performance

System performance was evaluated across data subsets (Table 4). The relative gain of the best-performing submission over random sampling varied substantially across subsets. The largest gain was 67.1% on HSN (0.625 versus 0.374), compared with 8.0% on ATBFL (0.502 versus 0.465), 13.2% on POW (0.490 versus 0.433), and 28.5% on UHH (0.428 versus 0.333). Submission rankings on ATBFL were strongly consistent with those obtained by averaging AULC across the three BirdSet subsets (HSN, POW, and UHH; Spearman’s $\rho = 0 . 8 5 5$ across ten submissions), although the rankings were not identical.

Table 4: Per-subset AULC (mAP). ATBFL is the marine dataset; HSN, POW and UHH are BirdSet subsets. Best value per subset in bold. Baseline entries are italicised.
<table><tr><td colspan="2" rowspan="2">Rank System</td><td rowspan="2">Score</td><td>Marine</td><td>Terrestrial (BirdSet)</td></tr><tr><td>ATBFL HSN</td><td>POW UHH</td></tr><tr><td>1 ADU-MMR</td><td>0.507</td><td>0.502</td><td>0.625</td><td>0.480 0.421</td></tr><tr><td>2</td><td>CARE-DPP</td><td>0.505 0.500</td><td>0.601</td><td>0.490 0.428</td></tr><tr><td>3</td><td>PB-MFS</td><td>0.499 0.491</td><td>0.601</td><td>0.481 0.422</td></tr><tr><td>4</td><td>Safe Rarity</td><td>0.492 0.492</td><td>0.589</td><td>0.475 0.412</td></tr><tr><td>5</td><td>Capped Rarity K-Center 0.481</td><td>0.489</td><td>0.568</td><td>0.474 0.392</td></tr><tr><td>6</td><td>AFL</td><td>0.477 0.483</td><td>0.552</td><td>0.462 0.411</td></tr><tr><td>7</td><td>Coreset KMeans</td><td>0.469 0.496</td><td>0.555</td><td>0.440 0.386</td></tr><tr><td>8</td><td>CoreSet</td><td>0.464 0.477</td><td>0.531</td><td>0.453 0.394</td></tr><tr><td>9</td><td>Coreset Eigenvalues</td><td>0.435 0.457</td><td>0.484</td><td>0.435 0.364</td></tr><tr><td>10</td><td>TypiClust</td><td>0.421 0.477</td><td>0.430</td><td>0.421 0.358</td></tr><tr><td>11</td><td>Margin</td><td>0.408 0.457</td><td>0.438</td><td>0.419 0.319</td></tr><tr><td>12 Random</td><td></td><td>0.401 0.465</td><td>0.374 0.433</td><td>0.333</td></tr><tr><td>13</td><td>All Quantiles KMeans</td><td>0.397 0.461</td><td>0.402</td><td>0.393 0.332</td></tr><tr><td>14</td><td>Balance Class (Eigen.)</td><td>0.393 0.447</td><td>0.389</td><td>0.408 0.328</td></tr><tr><td colspan="5">Range across all entries 0.055 0.251 0.097</td></tr></table>

## 3.3 Awards and Submissions

All ten systems were hybrid (diversity- and uncertainty-based) acquisition functions. Every team placed at least one entry above the strongest baseline method (CoreSet, 0.464).

Top Ranking. ADU-MMR (Dubus, Magaldi and Gros-Martial [10]) ranked first with a AULC mAP score of 0.507 ± 0.001 and was the best entry on two of the four subsets (ATBFL, 0.502; HSN, 0.625). Each sample is scored as a weighted sum of per-class binary entropies with the normalised squared distance to the nearest labelled neighbour. The weighting parameter $a _ { t }$ is a function of the classifiers confidence over the unlabelled pool. Uncertainty contributes nothing until the mean entropy falls below a threshold τ = 0.05 and its weight is capped at 0.5 so the diversity term never carries less weight than the uncertainty term. Batches are then diversified by greedy Maximum Marginal Relevance over the l2- normalised embeddings with λ = 0.3. The team provides an ablation of adaptive weighting compared to fixed settings demonstrating clear improvement using adaptive weighting. ADU-MMR achieved an average annotation cost of 1026.7, comparable to the CoreSet baseline. The relative cost was 2x higher than the baseline configuration, with a reduced acquisition batch of 25 samples (compared to 50), doubling the number AL cycles. This was the highest cost of all entries. The average sampling time was 1.006 s.

CARE-DPP (Magaldi and Dubus [24]) ranked second at 0.505±0.007 AULC mAP and was the best entry on the subsets POW (0.490) and UHH (0.428). The averaged ranking score across subsets was only 0.002 below ADU-MMR with an average SD of 0.007 meaning that the performance gap between the first and second ranking submissions are not statistically significant. CARE-DPP has a relative cost of only 1.1x higher than the baseline configuration and a sampling time of 0.215 s. This method uses a similar weighting between uncertainty and diversity but instead anneals based on budget rather than the model state. Novelty weight is dropped from 0.65 to 0.25 at 250 labels and uncertainty weight rising comparatively. CARE-DPP selects batches using a determinantal point process (DPP). The ablation notes a reduction in mean AULC of 0.038 (0.502 to 0.464) when removing DPP. This is a significant reduction relative to the other scores.

Jury Award. PB-MFS (Wang et al. [42]) ranked third at 0.499 and was selected for the novelty of their candidate filtering method and the completeness of the technical report. This method decouples filtering from selection. Four filters of ordered criteria: coverage, mean-margin uncertainty, class-frequency and disagreement with the K nearest labelled neighbours and label cardinality, which successively prune the pool at a retention ratio of 0.3 per stage. The votes based on each of the filters and each sample is ranked. The optimal annotation batch then balances uncertainty and diversity through a Pareto optimisation objective. The relative cost of this method was 1.6x the baseline configuration and the average sampling time was 0.485 s.

Notable Submissions Safe Rarity (Yang [43], rank 4, 0.492) mixes ranknormalised top-5 binary entropy, k-center diversity and a predicted-prevalence rarity term, with the rarity weight passed through a gate that collapses to zero when predictions are degenerate or the budget is small, reducing the sampler to entropy-filtered k-center. It achieved the best rank per unit compute of any entry, placing fourth at a relative cost of 1.0 and a sampling time of 0.385 s. Its ablation attributes the largest efect to k-center diversity. Capped Rarity K-Center (Nihal et al. [32], rank 5, 0.481) is the only submission to use dataset metadata, adding a location-stratification term alongside a capped class-deficit rarity term and a budget-ramped margin computed from a kNN label surrogate, with all three faded in or out by a single density gate keyed to the average number of labels per sample. AFL (Garcia-Yi [13], rank 6, 0.477) is the most distinctive, using a discrete particle-swarm search over candidate subsets for its 50-sample warm start and a front-loaded budget schedule, then coverage-only selection until 35% of the budget is spent before switching to a hybrid score. Its ablations rank the swarm warm start and diversity-aware batch reranking as the two largest contributors (-0.014 each), while the front-loaded scheduler and the density term were each worth under 0.001. The Parcerisas et al. [2] submitted four ranked entries and is the only team to have compared warm-up procedures systematically. Their entries ranked as Coreset KMeans placed 7th (0.469), above the CoreSet baseline, while Coreset Eigenvalues placed 9th (0.435) and their two confidence-quantile strategies placed 13th and 14th, below random sampling.

## 4 Discussion

## 4.1 Progress Towards Active Learning for Bioacoustics

The challenge provides further evidence that AL can improve label eficiency for bioacoustic classification. An earlier pilot study on HSN found that diversitybased sampling was particularly efective early in the AL process, while uncertainty became more useful once additional labels had been acquired [35]. A similar pattern emerged across the independent submissions in this challenge. CoreSet was the strongest baseline, outperforming both random and marginbased uncertainty sampling, while the highest-ranked submissions generally retained an explicit coverage or diversity component and combined it with uncertainty [10, 24, 42], rarity [32, 43], or class-balance information [24, 42]. Several methods further adapted these signals over the annotation budget [13, 24] or according to model confidence [10, 32, 43]. These results suggest that coverage of the representation space provides a robust foundation for acquisition when labels are scarce in bioacoustics, while model-dependent information becomes more useful as the classifier improves.

The benefit of AL varied substantially across subsets. Relative to random sampling, the best submission improved AULC by 67.1% on HSN and 28.5% on UHH, compared with 13.2% on POW and 8.0% on ATBFL (Table 4). These diferences indicate that the potential gains from AL depend strongly on characteristics of the underlying dataset. HSN, the sparsest BirdSet subset, showed the largest separation between methods, although the small number of subsets prevents attributing this efect to label density alone. Overall, the results support combining coverage with adaptively introduced model-based criteria rather than relying on a single acquisition method across the AL process.

Two teams varied acquisition batch size in opposite directions: CARE-DPP [24] increased it as labels accumulated (25 to 75), for a modest 0.014 gain, while AFL’s [13] ablation with larger early-cycle batches showed no improvement. The leading submission instead used a smaller fixed batch (25 vs. 50), doubling AL cycles and computational cost. Optimising batch size across the budget, or extending AL to larger batches [5], remains a valuable direction given the cost of repeated inference for large PAM deployments [1, 8].

Parcerisas et al. [2] systematically compared warm-up methods across ten acquisition strategies finding little efect (0.003 AULC diference) compared to that of the acquisition function (with a diference 0.121). Two other participants found a minor improvement in performance using warm-up with separate ablations showing a 0.026 [42] and 0.014 [13] increase in performance on the development set. Further investigation is required to determine the utility of warm-up in the case of high-quality pre-trained model embeddings applied within similar domains (bioacoustics-to-bioacoustics).

## 4.2 Single Metric or Aggregate

The leaderboard is based on a single metric in the form of AULC scores for macro mAP aggregated across datasets. While this provides a straightforward scalar ranking of the AL methods, there are several limitations: AULC is not directly comparable across datasets with diferent dificulty levels and label prevalences, a single metric captures only a single aspect of the systems performance. Finally the current ranking does not consider practical considerations such as segmentlevel labelling cost and the computational cost of diferent AL acquisition functions and configurations. Several of these limitations are implicitly considered by including proxies for them as separate statistics in Table 3. This leads to the question whether we can improve upon this by aggregating multiple metrics. One option would be to consider rank aggregation across datasets and evaluation criteria, as proposed for multi-task and multi-metric benchmarking [7]. These ranks could be complemented with normalized performance diferences, where we, for example, employ random sampling as lower and an oracle AL method [15] as upper references. Further additions may consider the actual segment-dependent labelling costs when computing AULC and incorporate computational eficiency either as a separate evaluation criterion or by expressing labelling and compute costs on the same scale. Estimating the labelling costs may leverage the number of labels per segment as a first crude proxy but requires further validation. A potentially supportive observation for such a proxy is that Margin, a purely exploitative AL method, selected on average segments with more labels than the remaining AL methods (cf. Table 3).

## 4.3 Configuration, Generalisability and Deployability

Label scarcity motivates AL but also makes acquisition configurations dificult to validate. Comparing settings often requires labelled validation data and repeated AL runs. Lüth et al. describe the AL validation paradox: configuration selection can consume the labels that AL aims to save [23]. Past acquisition decisions cannot be revised once their labels have been collected [22], and gains over random sampling may diminish under carefully controlled training and evaluation conditions [30]. Here, “deployability” means applying a complete acquisition policy to a new data source without label-intensive retuning.

Submitted configurations used fixed weights, schedules, gates, candidate-pool rules, or warm starts. We did not count raw hyperparameters, as this conflates choices with diferent roles. The submission format also did not standardise how values were chosen or whether development data informed them. The challenge fixed the classification head but allowed the acquisition configuration, query schedule, and warm start to vary (Section 2.3). The warm start directly afects the learning curve by determining the first labelled set included when the curve is integrated. Because the held-out evaluation data came from the same datasets and subsets represented during development, the ranking measures performance only on unseen samples within the predefined challenge domains, not transfer to a new site or data source.

Future challenge editions could test generalisability more directly by freezing each complete submitted configuration and applying it unchanged to sourcedisjoint evaluation data unavailable during development. They could also require standardised reports of configuration defaults, rationale, data dependencies, and tuning procedures. Rules relative to the annotation budget or properties of the unlabelled pool may accommodate changes in dataset scale, but their transfer would still require evaluation. Sensitivity analyses could identify which choices afect performance. Hyperparameter counts could remain descriptive metadata, but not stand-alone measures of generalisability or deployability.

## 4.4 Foundation-Model Embeddings as a Design Constraint

For this iteration of the challenge, we provided all participants with fixed PerchV2 embeddings. Fixing the representations allowed for a controlled comparison of acquisition functions by preventing gains from being driven by diferences in model training and architecture or output representation quality. The choice of foundation model itself afects AL performance, since many acquisition functions depend directly or indirectly on the geometry of the learned representations: changing the embedding model can change which samples are most uncertain redundant, representative, or distant from the labelled set. Dumoulin et al. 2025 compare multiple acoustic embedding models within an “agile modeling” workflow and find substantial diferences in downstream classification performance across representations. Domain-relevant bioacoustic embeddings generally outperforming more generic audio representations [11], with disproportionate efects for rare or easily-confused classes [20]. Future iterations of this challenge could explicitly explore this dimension further by including comparisons of the initial representations, or ensembles of them.

## 4.5 Dataset Selection and Representation Limits

The ATBFL recordings are sampled at 250 Hz and contain predominantly lowfrequency blue- and fin-whale vocalizations: several target calls have most of their energy below 60 $\mathrm { H z , }$ including blue whale Z-calls and fin whale 20 Hz pulses [29]. However, the PerchV2 frontend operates on a log-mel representation spanning approximately 60 $\mathrm { H z } - 1 6 \mathrm { k H z } \ [ 2 7 ]$ . Although some ATBFL calls contain higherfrequency components useful for discrimination, much of the signal energy falls below the PerchV2 range. However, this representation mismatch didn’t prevent strong performance on ATBFL (the results summarized in Sec. 3.2). This suggests that PerchV2 embeddings are still able to discriminate between the call types. This may be because some calls contain harmonics, broadband structure, or high-frequency components above the 60 Hz frontend cutof. The embeddings may also capture other acoustic characteristics correlated with call identity.

While future iterations of this challenge should continue to include both terrestrial and marine datasets to test cross-domain performance, it may be more valuable to replace ATBFL with an underwater acoustics dataset that better matches existing embedding models. If PerchV2 remains a common representation for bioacoustics, datasets with stronger kilohertz-range signals, such as the DCLDE killer whale dataset [9], would allow for cleaner comparisons of acquisition strategies.

## 4.6 Joint Allocation of the Annotation Budget

An important consideration for deploying ML models for bioacoustic monitoring is validating model performance to support downstream ecological inference. While the sampling objective for validation difers from that of AL for model training, both draw from the same finite labelling budget and cannot, in most scenarios, be used interchangeably without introducing bias [12, 20, 22]. This highlights a key limitation of the current data challenge format and evaluation of AL systems more generally: oracle sampling assumes a pre-existing validation set that is excluded from budget considerations. BaseAL V1.2 [26] now supports active testing, enabling iterative validation strategies within a budget. Future editions could leverage this to require participants to jointly optimise model performance and evaluation within a single annotation budget, better reflecting deployment scenarios.

## 5 Conclusions

The 2026 BioDCASE Active Learning for Bioacoustics challenge demonstrates that AL can substantially improve label eficiency across bioacoustics domains, with combining multiple acquisition signals, shifting from diversity- to uncertaintybased selection as labels accumulate, and explicitly reducing batch redundancy each improving over single-criterion methods. Future work should extend evaluation to better reflect deployment: systematic comparison of embedding models, computational and annotation cost, configuration generalisability, joint allocation of training and validation budgets, and coverage of more diverse taxa (anurans, bats, fish). The challenge format and BaseAL provide a foundation for continued progress in data-eficient bioacoustic monitoring.

## Acknowledgements

Thank you to the participants of the 2026 BioDCASE Active Learning for Bioacoustics challenge and to the BioDCASE organisers.

## References

1. Bernard, C., McEwen, B., Cretois, B., Glotin, H., Stowell, D., Marxer, R.: Datadriven sampling strategies for fine-tuning bird detection models. The Journal of the Acoustical Society of America 159(6), 4891–4903 (2026)

2. Bordoux, V., Cuyx, B., Parcerisas, C., Schall, E.: Are whales too big to fly? Tech. rep., BioDCASE 2026 Challenge (June 2026), https://biodcase.github.io/ documents/challenge2026/technical\_reports/Parcerisas\_task4.technical\_ report.pdf

3. Burns, A., Harrell, L., van Merriënboer, B., Dumoulin, V., Hamer, J., Denton, T.: Perch 2.0 transfers’ whale’to underwater tasks. arXiv preprint arXiv:2512.03219 (2025)

4. Chronister, L.M., Rhinehart, T.A., Place, A., Kitzes, J.: An annotated set of audio recordings of Eastern North American birds containing frequency, time, and species information (2022). https://doi.org/10.5061/dryad.d2547d81z

5. Citovsky, G., DeSalvo, G., Gentile, C., Karydas, L., Rajagopalan, A., Rostamizadeh, A., Kumar, S.: Batch active learning at scale. Advances in Neural Information Processing Systems 34, 11933–11944 (2021)

6. Clapp, M., Kahl, S., Meyer, E., McKenna, M., Klinck, H., Patricelli, G.: A collection of fully-annotated soundscape recordings from the southern Sierra Nevada mountain range (2023). https://doi.org/10.5281/zenodo.7525805

7. Colombo, P., Noiry, N., Irurozki, E., CLEMENCON, S.: What are the best Systems? New Perspectives on NLP Benchmarking. In: Oh, A.H., Agarwal, A., Belgrave, D., Cho, K. (eds.) Advances in Neural Information Processing Systems (2022), https://openreview.net/forum?id=kvtVrzQPvgb

8. Cretois, B., Rosten, C.M., Wiel, J., Barile, C., McEwen, B., Bernard, C., Boom, M.P., Bota, G., Brotons, L., Serrano-Davies, E., et al.: Tabmon: Design and deployment of a transnational passive acoustic monitoring network for european birds. Methods in Ecology and Evolution 17(6), 1867–1879 (2026)

9. Department of Fisheries and Oceans Canada: DCLDE 2027: Killer whale (Orcinus orca) ecotype and other species annotations for the Detection Classification Localization and Density Estimate (DCLDE) conference in 2027 (2025). https://doi.org/10.25921/15ey-mh50, https://doi.org/10.25921/15ey-mh50, accessed: 2026-08-09

10. Dubus, G., Magaldi, H., Gros-Martial, A.: Adaptive diversity-uncertainty active learning with redundancy control for bioacoustic event classification. Tech. rep., BioDCASE 2026 Challenge (June 2026), https : / / biodcase . github . io/documents/challenge2026/technical\_reports/Dubus\_task4.technical\_ report.pdf

11. Dumoulin, V., Stretcu, O., Hamer, J., Harrell, L., Laber, R., Larochelle, H., van Merriënboer, B., Navine, A., Hart, P., Williams, B., et al.: The search for squawk: Agile modeling in bioacoustics. arXiv preprint arXiv:2505.03071 (2025)

12. Farquhar, S., Gal, Y., Rainforth, T.: On statistical bias in active learning: How and when to fix it. In: International Conference on Learning Representations (2021)

13. Garcia-Yi, J.: Adaptive frontloaded learning. Tech. rep., BioDCASE 2026 Challenge (June 2026), https://biodcase.github.io/documents/challenge2026/ technical\_reports/Garcia-Yi\_Task4.technical\_report.pdf

14. Hacohen, G., Dekel, A., Weinshall, D.: Active learning on a budget: Opposite strategies suit high and low budgets. arXiv preprint arXiv:2202.02794 (2022)

15. Huseljic, D., Hahn, P., Herde, M., Sandrock, C., Sick, B.: BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning. Transactions on Machine Learning Research (2026), https://openreview.net/forum?id=qTs6spvhOS

16. Jean-Labadye, L., Parcerisas, C., Miller, B., Carvaillo, P., Dubus, G., Farrugia, N., Gros-Martial, A., Marmoret, A., Moummad, I., Napoli, A., Nguyen Hong Duc, P., Raumer, P.Y., Schall, E., White, E., Adam, O., Roch, M.A., White, P., Cazau, D.: Biodcase 2025 task 2 : Development set (Mar 2025). https://doi.org/10.5281/ zenodo.15092732, https://doi.org/10.5281/zenodo.15092732

17. Kahl, S., Wood, C.M., Eibl, M., Klinck, H.: Birdnet: A deep learning solution for avian diversity monitoring. Ecological Informatics 61, 101236 (2021)

18. Kath, H., Serafini, P.P., Campos, I.B., Gouvêa, T.S., Sonntag, D.: Leveraging transfer learning and active learning for data annotation in passive acoustic monitoring of wildlife. Ecological Informatics 82, 102710 (2024)

19. Kather, V.S., Haupert, S., Ghani, B., Stowell, D.: bacpipe: a python package to make bioacoustic deep learning models accessible. arXiv preprint arXiv:2604.11560 (2026)

20. Kurinchi-Vendhan, R., Beery, S.: Finding needles in the haystack: Transductive active labeling in ecology. arXiv preprint arXiv:2606.03821 (2026)

21. Kurinchi-Vendhan, R., Zhang, S., McEwen, B.: Biodcase 2026 task 4: Atbfl dataset (Mar 2026). https://doi.org/10.5281/zenodo.19133112, https://doi.org/10. 5281/zenodo.19133112

22. Lowell, D., Lipton, Z.C., Wallace, B.C.: Practical obstacles to deploying active learning. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). pp. 21–30 (2019)

23. Lüth, C.T., Bungert, T.J., Klein, L., Jaeger, P.F.: Navigating the pitfalls of active learning evaluation: A systematic framework for meaningful performance assessment. In: Advances in Neural Information Processing Systems. vol. 36, pp. 9789– 9836 (2023). https://doi.org/10.52202/075280-0428

24. Magaldi, H., Dubus, G.: Determinantal point process sampling for bioacoustic active learning. Tech. rep., BioDCASE 2026 Challenge (June 2026), https:// biodcase.github.io/documents/challenge2026/technical\_reports/Magaldi\_ task4.technical\_report.pdf

25. McEwen, B., Soltero, K., Gutschmidt, S., Bainbridge-Smith, A., Atlas, J., Green, R.: Active few-shot learning for rare bioacoustic feature annotation. Ecological Informatics 82, 102734 (2024)

26. McEwen, B., Zhang, S.: Baseal: Release v1.2.0 (Aug 2026). https://doi.org/10. 5281/zenodo.21806641, https://doi.org/10.5281/zenodo.21806641

27. van Merriënboer, B., Dumoulin, V., Hamer, J., Harrell, L., Burns, A., Denton, T.: Perch 2.0: The bittern lesson for bioacoustics. arXiv preprint arXiv:2508.04665 (2025)

28. Mesaros, A., Serizel, R., Heittola, T., Virtanen, T., Plumbley, M.D.: A decade of dcase: Achievements, practices, evaluations and future challenges. In: ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1–5. IEEE (2025)

29. Miller, B., Staford, K., Van Opzeeland, I., Harris, D., Samaran, F., Širovic, A., et al.: An annotated library of underwater acoustic recordings for testing and training automated algorithms for detecting antarctic blue and fin whale sounds. Dataset hosted by the Australian Antarctic Data Centre (2020)

30. Munjal, P., Hayat, N., Hayat, M., Sourati, J., Khan, S.: Towards robust and reproducible active learning using neural networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 223–232 (Jun 2022). https://doi.org/10.1109/CVPR52688.2022.00032

31. Navine, A., Kahl, S., Tanimoto-Johnson, A., Klinck, H., Hart, P.: A collection of fully-annotated soundscape recordings from the island of hawai’i (2022). https: //doi.org/10.5281/zenodo.7078499

32. Nihal, R.A., Yen, B., Ashizawa, T., Nakadai, K.: A density-gated coverage sampler for cross-domain active learning in bioacoustics. Tech. rep., BioDCASE 2026 Challenge (June 2026), https://biodcase.github.io/documents/challenge2026/ technical\_reports/Nihal\_task4.technical\_report.pdf

33. Rauch, L., Heinrich, R., Moummad, I., Joly, A., Sick, B., Scholz, C.: Can masked autoencoders also listen to birds? Transactions on Machine Learning Research (TMLR) (2025)

34. Rauch, L., Herde, M., McEwen, B.: BioDCASE 2026 Task 4: BirdSet Dataset (2026). https://doi.org/10.5281/zenodo.19340660

35. Rauch, L., Huseljic, D., Wirth, M., Decke, J., Sick, B., Scholz, C.: Towards deep active learning in avian bioacoustics. In: Workshop on Interactive Adaptive Learning (IAL@ECML-PKDD) (2024)

36. Rauch, L., Schwinger, R., Wirth, M., Heinrich, R., Huseljic, D., Herde, M., Lange, J., Kahl, S., Sick, B., Tomforde, S., Scholz, C.: BirdSet: A large-scale dataset for audio classification in avian bioacoustics. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id= dRXxFEY8ZE

37. Schwinger, R., Zadeh, P.V., Rauch, L., Kurz, M., Hauschild, T., Lapp, S., Tomforde, S.: Foundation models for bioacoustics–a comparative review. Ecological Informatics p. 103765 (2026)

38. Sener, O., Savarese, S.: Active learning for convolutional neural networks: A coreset approach. arXiv preprint arXiv:1708.00489 (2017)

39. Settles, B.: Active learning. Morgan & Claypool Publishers (2012)

40. Stowell, D., Giannoulis, D., Benetos, E., Lagrange, M., Plumbley, M.D.: Detection and classification of acoustic scenes and events. IEEE Transactions on Multimedia 17(10), 1733–1746 (2015)

41. Stowell, D., Vidaña-Vila, E., Nolasco, I., McEwen, B., Jean-Labadye, L., Benhamadi, Y., Dubus, G., Hofman, B., Linhart, P., Morandi, I., et al.: Biodcase: Using data challenges to make community advances in computational bioacoustics. bioRxiv pp. 2026–04 (2026)

42. Wang, H., Dou, H., Chen, H., Du, Y., Tu, L., Pan, N., Zhang, H., Wu, J., Li, G., Huang, G.: Pb-mfs: Multi-funnel selection with pareto-balanced uncertainty and diversity for bioacoustic active learning. Tech. rep., BioDCASE 2026 Challenge (June 2026), https://biodcase.github.io/documents/challenge2026/ technical\_reports/Huang\_task4.technical\_report.pdf

43. Yang, X.W.: Safe rarity-aware k-center active learning for bioacoustic classification. Tech. rep., BioDCASE 2026 Challenge (May 2026), https://biodcase.github. io / documents / challenge2026 / technical \_ reports / Yang \_ task4 . technical \_ report.pdf