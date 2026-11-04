# Beyond Foundation Models: PFN Priors Improve Tabular Hyperparameter Optimization

Artem Chistyakov<sup>\*,1,2</sup>, Renat Sergazinov<sup>\*,2</sup>, Artem Babenko<sup>2,1</sup>

<sup>1</sup>HSE University, <sup>2</sup>Yandex

## Abstract

Synthetic priors have so far served one purpose: pretraining tabular foundation models. We ask whether the same priors can improve conventional tabular models, which are trained from scratch on each dataset. We propose PriorTune, a method that meta-tunes hyperparameter configurations on synthetic tasks alone. Ofline, for a given model, we fit a pool of configurations on datasets sampled from a published PFN prior and greedily select the subset with the highest winrate over the random baseline. Online, a new dataset fits only the selected configurations, so the gain comes at no extra cost. Empirically, we observe that scaling both the number of configurations and synthetic datasets has an equal scaling efect on the performance of the final subset of configurations, which we also explain theoretically. On the full BeyondArena benchmark, LightGBM with TabPrep features with our configurations ranks above TabPFN-3 and RealMLP, while being much more eficient at inference. Overall, our results establish existing PFN priors as a scalable source of metatraining data for hyperparameter selection for non-foundational tabular models.

Code | Results

## 1. Introduction

Synthetic data drives the development of tabular foundation models. Since TabPFNv1 [1], every generation has come with a richer prior: TabPFNv2 added new edge types to the causal graph sampler [2], TabICLv2 added heavy-tailed random graphs and a filter on the generated datasets [3], and TabPFN-3 scaled the generated datasets to a million rows [4]. To date, these priors have mostly served one purpose: pretraining PFNs.

Conventional models, which are trained from scratch on each dataset, have no direct way to use these priors. For such models, e.g. GBDTs and DNNs, hyperparameter selection is the natural way for transferring knowledge across tasks. TabRepo demonstrates that a set of hyperparameter configurations selected on approximately 200 real datasets transfers efectively to unseen datasets [5]. Similarly, RealMLP identifies strong MLP architecture and training defaults through meta-tuning on 118 real datasets [6]. Both rely on a small, fixed collection of real datasets, which caps the scale and diversity of the meta-training tasks. This raises a natural question: can datasets sampled from a PFN prior surpass real datasets when learning transferable sets of hyperparameter configurations? To test this, we sample 4000 synthetic datasets from TabICLv2 prior, evaluate 4000 hyperparameter configurations on them, and propose a new algorithm to select K best-performing configurations. On a new dataset, we only train the selected subset of K configurations, so the meta-tuning gains transfer for free.

![](images/2853e8d09306fdaa42e9d9b845d4bdda45996da256595b273a5976f8a8a67061.jpg)

![](images/c8e255d675cb673487d8f732861c721de1425eae45de01a5808efe66bd911341.jpg)  
Figure 1 | Full BeyondArena. Left: Elo under default, tuned, and tuned-and-ensembled deployment. Our families are coloured; FT (half-colored) and FTE (full-colored) are the subsets meta-tuned under T and T+E. Right: Elo against inference latency at batch size 1. GBDTs run on 16 CPU threads, and DNNs and PFNs on one GPU. For PFNs, we run with recompute (light grey), KV cache on CPU (medium grey), on GPU (dark grey). Protocol in Appendix A.1.4.

We study four eficient tabular models: LightGBM [7], LightGBM with TabPrep [8], an MLP with cosine embeddings [9], and TabPack [10]. These four models are near the eficiency-performance Pareto frontier (Figure 1). We use the TabICLv2 prior, with TabPFNv1 and TabICLv1 in ablations, and evaluate transfer on TabArena’s 51 IID datasets [11] and BeyondArena’s 142 datasets with diverse splitting protocols [12].

As a source of meta-training tasks, synthetic priors have three advantages over real datasets. First, supply is unlimited and quality is controllable. Real repositories plateau: on TabRepo’s 182 datasets, adding configurations stops helping, whereas on synthetic tasks neither the number of tasks nor the number of configurations has saturated at 4,000 (Section 5). Second, synthetic priors can be continuously improved: foundational model have shown that improving priors yields better model performance. Third, synthetic tasks rule out test-set contamination by construction. Real data needs an overlap audit: for TabArena, we removed 18 of the 200 TabRepo datasets.

The contributions of this paper are as follows:

1. Meta-tuning hyperparameters on synthetic data. We show that existing PFN priors can improve conventional tabular models through ofline hyperparameter selection. Per dataset, real tasks help more, but synthetic tasks scale: given equal budget (configurations × datasets), hyperparameters selected on 4000 synthetic datasets outperform those selected on 182 real TabRepo datasets for every model and every K.

2. Improved tuning on real datasets. On both TabArena and BeyondArena, the selected sets of configurations outperform random sets of the same size. On TabArena, they also outperform random sets of any size, so meta-selection improves both the quality and the eficiency of downstream tuning. On BeyondArena, our LightGBM-TabPrep ranks 7th of 58 methods, ahead of TabPFN-3 and behind only recent foundation models.

3. An analysis of scaling and prior choice. We investigate how the quality of the configuration set depends on the number of synthetic tasks, the candidate pool size, and the generating prior, and derive a transfer bound and a record-count argument that motivate the observed scaling forms. Scaling number of datasets and configurations keeps paying of at 4,000, whereas the choice of prior appears secondary: even the TabPFNv1 prior is competitive with TabICLv2.

![](images/03c391dc87520e6a83d6e17f408bfbc4ee20207d9dfff0b47b002756f263263d.jpg)  
Figure 2 | Overview. Ofline, we sample hyperparameter configurations from a pool C, fit them on N synthetic tasks, and greedily select a subset of K by win rate over random subsets under the given deployment rule. Online, we fit the K selected configurations once per task.

## 2. Related Work

Transfer of hyperparameter configurations. Good hyperparameter configurations typically transfer across datasets. Feurer et al. [13] warm-start Bayesian optimization with configurations that performed well on similar datasets. Later work drops the similarity and learns one fixed set for all datasets: given a table of scores for every configuration on every dataset, greedily add the configuration that most improves the average best-member score over datasets [14–16]. TabRepo scales this ofline selection to 1,310 configurations on 200 datasets, and AutoGluon 1.0 ships the learned set as its default [5, 17]. RealMLP meta-learns several model components (e.g. preprocessing, activations) on 118 datasets [6]. All of this work is bounded by the supply of real data, 25 to 208 datasets. TabRepo reports a saturating efect around 150 datasets or configurations. We keep the greedy forward selection, but run it on synthetic tasks, whose supply is unlimited, and change the scoring function (Section 3.4).

A second line transfers knowledge into the optimizer rather than into a set of configurations. One line of work fits the Gaussian-process prior on evaluations from previous tasks [18–22]. OptFormer pre-trains a text transformer on optimization logs [23]. Recent works such as PFNs4BO and its successors, pretrain PFNs on synthetic optimization tasks [24–27]. All of these search sequentially and need the surrogate model at tuning time. Our configuration set is built once ofline and can be evaluated in parallel.

Synthetic priors for tabular data. Synthetic priors are the engine of tabular foundation models. Among others, the recent TabPFN, TabICL, Mitra, LimiX, Xiaomi-TabLDM, Causilo, and EXAONE Tabular are pretrained on synthetic tables alone [1–4, 28–33]. A typical synthetic prior has three parts: a directed graph over variables, an edge function that maps parents to a child, and an observation process. Graphs are random DAGs in TabPFN, Cauchy random graphs with varied connectivity in TabICLv2, and hierarchical causal models in LimiX. Edge functions started as small neural networks and grew to include decision trees and discretization in TabPFNv2, tree ensembles in TabICL, and Gaussian processes, polynomials, and products in TabICLv2 [2, 33]. Related in spirit, recent work distills a PFN into a tree ensemble or an MLP on a given dataset [34, 35]. However, distillation requires PFN during the training phase and has yet to demonstrate significant performance improvements.

## 3. Method

In this section, we present PriorTune, a method for meta-tuning strong hyperparameter configurations of conventional tabular models on synthetic data (Figure 2). Ofline, we sample N tasks from a synthetic prior Q (Section 3.2), fit a pool C of C configurations once per task (Section 3.3), and greedily select K of them (Section 3.4). Online, a real task fits only the K selected configurations and deploys them by validation, either the best one (tuned) or a greedy ensemble (tuned and ensembled).

Algorithm 1 PriorTune subset selection   
Require: cached validation and test predictions of every $c \in \mathcal { C }$ on synthetic tasks   
$D _ { 1 } , \ldots , D _ { N } ;$ subset size $K ;$ number of draws $R ;$ deployment rule $\mathcal { E } \in \{ \mathrm { T } , \mathrm { T } { + } \mathrm { E } \}$ equa  
tion 1   
1: Draw references $\mathcal { R } _ { 1 } , . . . , \mathcal { R } _ { R } \sim \mathrm { U n i f } ( \Pi _ { K } )$ and, for every task $i ,$ bootstrap resamples   
$D _ { \mathrm { v a l } , i } ^ { ( 1 ) } , \ldots , D _ { \mathrm { v a l } , i } ^ { ( R ) }$ of its validation rows   
2: $\quad S _ { 0 } \gets \emptyset$   
3: for $k = 1 , \ldots , K$ do   
4: for each candidate $c \in \mathcal { C } \setminus \mathcal { S } _ { k - 1 }$ do   
5: # deploy both with $\mathcal { E }$ on resample $^ { r , }$ score on test; $\begin{array} { r } { h ( a , b ) = \mathbf { 1 } _ { a < b } + \frac { 1 } { 2 } \mathbf { 1 } _ { a = b } } \end{array}$   
6: $\widetilde { U } ( c ) \gets \frac { 1 } { N R } \sum _ { i = 1 } ^ { N } \sum _ { \substack { n = 1 } } ^ { K } h \big ( L _ { i } ^ { ( r ) } ( \mathcal { S } _ { k - 1 } \cup \{ c \} ) , L _ { i } ^ { ( r ) } ( \mathcal { R } _ { r } ) \big )$   
7: end for   
8: $c _ { k } \gets \arg \operatorname* { m a x } _ { c } \widetilde { U } ( c )$ , ties broken by the lowest configuration id   
9: $S _ { k } \gets S _ { k - 1 } \cup \{ c _ { k } \}$   
10: end for   
11: return $\scriptstyle { S _ { K } }$

## 3.1 Setup

Notation. We consider regression and classification on tabular data. We denote the synthetic data generator by Q and a dataset by D. The loss ℓ is the native metric of the task, oriented so that lower is better: RMSE for regression, log-loss for multiclass, and 1−ROC-AUC for binary classification, as in TabArena and BeyondArena. We split each dataset into train, validation, and test subsets $D _ { \mathrm { t r a i n } } , D _ { \mathrm { v a l } } , D _ { \mathrm { t e s t } }$

Benchmarks. We use TabArena [11] and BeyondArena [12], which together cover 155 to 1.3 million rows, 3 to 22,215 features, and i.i.d., grouped, and temporal splits. Table 2 summarizes both suites. The main results (Section 4) use the full BeyondArena with every outer train-test split. The ablations of Section 5 use TabArena-Lite, the first train-test split. Each split has up to 8 inner train-validation splits. No benchmark data enters the selection.

## 3.2 Synthetic data generation

For all main experiments, we sample tasks from the TabICLv2 prior [3]. Three settings of the prior matter most, and we keep them close to the third pretraining stage of TabICLv2. Dataset size is log-uniform between 400 and 60,000 rows. The feature count is uniform between 1 and a cap that falls from 100 to 20 as the row count grows. The sampled datasets are filtered using both of the available filters: reject if the target node of the graph shares no ancestors with feature nodes, and reject if ExtraTrees cannot beat the constant predictor. We fix the task mix at 2,000 regression, 1,400 binary, and 600 multiclass datasets with 3 to 10 classes, and split each dataset 64/16/20 into train, validation, and test, stratified for classification. Appendix A.3 lists all settings.

## 3.3 Models

For our main runs, we use LightGBM [7], LightGBM with TabPrep [8], MLP with cosine embeddings [9], and TabPack [10]. We select these models for their tradeof between training and inference speed and performance (Figure 1). Each configuration in the pool C with $C = 4 { , } 0 0 0$ is an independent draw from the model’s published search space, which we list in Appendix A.2. TabPack is a packed ensemble of MLPs, so its unit of sampling is a single MLP configuration: it shares the pool of $C = 4 { , } 0 0 0$ MLP configurations, from which we select its $K = 6 4$ members.

## 3.4 Subset selection

The selection mimics the deployment rule and scores against random subsets of the same size. Following TabArena and BeyondArena, we distinguish two deployment rules: tuned (T), where the validation-best member is deployed, and tuned and ensembled (T+E), where validation builds a greedy ensemble [36]. We implement the whole procedure as a greedy forward selection algorithm: at each iteration starting from an empty set, we add a configuration, score it under the given deployment rule, and select the one that yields the highest winrate against the random subset (Algorithm 1). Appendix B states the objective formally. Below we motivate each choice on a small example (see Section 5.1 for ablation).

Table 1 gives a toy instance under the tuned rule: four configurations A to D, three synthetic tasks $D _ { 1 }$ to $D _ { 3 }$ for selection, one held-out task $D _ { 4 }$ standing for a real dataset, and a subset size of $K = 2$

Selecting on validation, scoring on test. Under both deployment rules, the model (or ensemble) is selected on the validation split and scored on the test split. Our objective mimics this, which selects configurations that are stable across validation and test. The alternative, selecting on test directly or on a single out-of-fold score as TabRepo and AutoGluon do [5], hides that instability. In Table 1, A and D are unstable: their validation and test losses disagree on most datasets, whereas B and C are stable. Both selections pick B first, then diverge: test-only selection returns {B, A}, validation-then-test returns the stable {B, C}. On the held-out $D _ { 4 } , \{ \mathrm { B } , \mathrm { A } \}$ deploys A and loses; {B, C} deploys C and wins.

Comparing against the random baseline. The alternative is to use losses directly: TabRepo adds the configuration that most lowers the mean error of the best member over datasets [5]. Raw losses mix metrics and scales across tasks. In Table 1, any rule on raw losses selects

Table 1 | A toy example under the tuned rule. Four configurations, three synthetic datasets $D _ { 1 } { - } D _ { 3 }$ used for selection $( D _ { 1 }$ scored by ${ \mathrm { R M S E } } ,$ the rest by log-loss) and one held-out dataset $D _ { 4 } ~ ( \mathrm { l o g \mathrm { - } l o s s ) }$ Cells are validation / test losses (lower is better); the last row is the test loss of the deployed member of a random reference subset. We select a subset of $K = 2$ from the empty set.
<table><tr><td></td><td colspan="2"> $D _ { 1 }$ </td><td colspan="2"> $D _ { 2 }$ </td><td colspan="2"> $D _ { 3 }$ </td><td colspan="2"> $D _ { 4 } ,$  held out</td></tr><tr><td>Configuration</td><td>val</td><td>test</td><td>val</td><td>test</td><td>val</td><td>test</td><td>val</td><td>test</td></tr><tr><td>A</td><td>105</td><td>103</td><td>0.28</td><td>0.36</td><td>0.60</td><td>0.45</td><td>0.38</td><td>0.47</td></tr><tr><td>B</td><td>96</td><td>96</td><td>0.30</td><td>0.31</td><td>0.53</td><td>0.53</td><td>0.41</td><td>0.41</td></tr><tr><td>C</td><td>95</td><td>97</td><td>0.34</td><td>0.34</td><td>0.55</td><td>0.55</td><td>0.40</td><td>0.40</td></tr><tr><td>D</td><td>60</td><td>60</td><td>0.29</td><td>0.40</td><td>0.52</td><td>0.60</td><td>0.39</td><td>0.50</td></tr><tr><td>Random reference</td><td>一</td><td>100</td><td>一</td><td>0.32</td><td>一</td><td>0.50</td><td>一</td><td>0.42</td></tr></table>

![](images/d87858f4061c7810fd4feedb44eac042a87b7aaa3466dd8c96d05a9bbaa3e656.jpg)  
Figure 3 | Elo on the full BeyondArena by dataset group. Top: tuned (T), our models with their FT subsets. Bottom: tuned and ensembled (T+E), our models with their FTE subsets. Under each panel, the bars show the Elo gain of each meta-tuned subset over the same model with its random subset.

D first, since it cuts the RMSE on $D _ { 1 }$ by 36 points, which outweighs every other gain in the table, but D has the worst test loss on every other dataset. Ranking objectives are preferred over regression objectives for the same reason in ofline model-based optimization [37].

Stability. The third design choice is to resample both the validation split and the random reference. A single validation split decides which member is deployed: in Table 1, C beats B on the validation split of $D _ { 1 }$ by one RMSE point, and another validation sample may deploy B instead. We therefore draw R = 16 bootstrap resamples of the validation rows, paired with R independent random references $\mathcal { R } _ { r \mathrm { i } }$ ; with cached predictions the cost is negligible, and averaging over references is unbiased with variance O(1/R) (Proposition 4).

## 4. Main results

Setup. We evaluate on the full BeyondArena core (142 datasets) under the benchmark’s protocol: every outer train-test split, 8-fold bagging within each, native metrics, and K = 25 configurations per dataset (the oficial budget). TabPack fits its K = 64 members. FT and FTE denote the subsets meta-tuned for T and T+E. A random subset has 25 draws from the same search space. We rank methods by Elo, which counts pairwise wins and losses against all other methods. Figure 3 reports Elo within each BeyondArena dataset group. Appendix A.4.1 reports alternative metric, the median improvement over the tuned MLP.

State of the art and gains over random subsets. Under T, LightGBM-TabPrep with its FT subset has the highest Elo of all tuned methods (Figure 1). Under T+E, LightGBM-TabPrep with its FTE subset ranks 7th, above RealMLP and TabPFN-3. The 6 methods above it are pretrained foundation models: TabPFN-3.5 (fast and regular), Causilo, and 3 sizes of Kumo-Tabular [38] (Figure 10 in the appendix shows the full leaderboard). Across groups (Figure 3), LightGBM-TabPrep leads RealMLP on small and medium datasets, and RealMLP leads on tiny and large datasets and on text features. Compared to Kumo-Tabular, LightGBM-TabPrep loses on i.i.d. splits and on tiny to medium datasets, but leads it on grouped and temporal splits and on large datasets (1313 against 1183 Elo). The bars of Figure 3 show the gain of each meta-tuned subset over its random subset. LightGBM-TabPrep gains in every group, by 21 to 72 Elo under T+E and by 5 to 76 under T. The other models gain 5 to 28 Elo overall, but lose to their random subsets in some groups, most often on grouped and temporal splits. These gains cost nothing at deployment, and Section 5.3 shows that, on TabArena, random subsets of any size up to the full pool of 4,000 do not close the gap.

![](images/2ae17abe1f6177ed6d6258be9471d071a7ffe1e488840ce0f7d9a84c1125276b.jpg)  
Figure 4 | Selection rules, synthetic out-of-fold, T deployment. Each of 200 repetitions selects on 3,000 random synthetic tasks and deploys on the other 1,000. Points are win rates against random-K baselines that no selection saw, ±1.96 standard errors over repetitions; the shaded row is our rule. Values in Table 10.

Accuracy and inference cost. The right panel of Figure 1 plots Elo against the latency of a single-row prediction. Each PFN appears in three modes: recompute, and a key-value (KV) cache kept on the CPU or on the GPU. Up to 1313 Elo, the Pareto front consists of LightGBM, CatBoost, TabPack, and LightGBM-TabPrep. LightGBM-TabPrep with its FTE subset under T+E reaches 1313 Elo at 4.4 ms. The references with a similar Elo are 2.6 to 740 times slower: RealMLP under T+E reaches 1297 Elo at 11 ms, TabPFN-3 1279 at 243 to 3,308 ms, and TabICLv2 1215 at 102 to 1,412 ms. Above 1313 Elo, the front consists of the 3 sizes of Kumo-Tabular with a KV cache, from 1361 Elo at 81 ms to 1455 Elo at 236 ms. Meta-tuning changes only which configurations are fitted, so the latency stays within a factor of 1.8 of the random subset.

## 5. Ablations

All ablations use subsets selected under T and report the win rate of the meta-tuned subset against random subsets of the same size on TabArena (details in Appendix A.1.2). We use T because T+E selection runs a Caruana ensemble per candidate at every step, which is expensive. TabPack is absent: it is an online ensemble of MLP members, so the MLP configurations cover it. The main text shows K = 200, TabArena’s default tuning budget, and we include K = 25, 64 in Appendix A.4.

![](images/79cfa8e6ce974ac718416326fe04c8090ccfa31e27250028d97450ed16b5b7ff.jpg)  
Figure 5 | TabArena, T deployment, win rate against a random subset of the same size. Synthetic versus TabRepo meta-training data at $K = 2 0 0$ . Dots are (C, N) cells; lines are the best cell within each factor-of-two budget bin. For TabRepo (182 datasets), we evaluated 16,000 configurations, so its curve stops at $B = 1 6 , 0 0 0 \times 1 8 2$

## 5.1 Selection rule

Figure 4 ablates the choices of Section 3.4. To avoid potential data leak, we evaluate out-of-fold on synthetic tasks: randomly select 3,000 datasets, deploy under T on the other 1,000, repeat 200 times. All rules choose from the same 4,000 configurations and are scored against the same random-K baselines (Appendix A.1.2.1).

Comparing to TabRepo. The TabRepo algorithm scores the best member of a subset directly, with no validation-then-test step, so we give it one score per task on the joined validation and test rows. We replace the raw loss with TabRepo’s normalized error, $( \ell { - } \ell _ { \mathrm { b e s t } } ) / ( \ell _ { \mathrm { m e d i a n } } { - } \ell _ { \mathrm { b e s t } } )$ clipped to [0, 1] per task [5], which removes the metric scale. From the empty set, each step adds the configuration that minimizes the mean over tasks of the best normalized error in the subset. It trails R = 16 for every model and K, by 0.010 to 0.038: even with the scale gone, the loss shape steers the selection into suboptimal regions. A naive improvement is to count instead the tasks on which a candidate improves the best member (i.e., a win rate against the current subset). The improved version beats the base in all nine cases, by 0.004 to 0.017. However, it still trails R = 16 by 0.014 to 0.024 for LightGBM and LightGBM-TabPrep, where scoring the best member hides configurations whose validation rank does not hold on test, and ties it on the MLP (−0.003 to +0.003).

Resampling validation and references. The plain hill-climb, with one validation split and one reference, trails R = 16 by 0.014 to 0.031. If we resample (16 times) just the validation, we close 60 to 84% of the gap. If we resample just the random references, we close 41 to 62%. Both matter, but validation noise appears to matter more. The win rate rises monotonically in R for every model and K: by 0.015 to 0.030 from R = 1 to R = 16, most at K = 200.

## 5.2 Synthetic versus real meta-training data

TabRepo [5] is the standard real-data source for meta-tuning: 200 datasets, of which we drop 18 that overlap with or share a source with TabArena (listed in Appendix A.4). We fit 16,000 configurations (our pool of 4,000 and 12,000 further draws) on the 182 datasets and select on them with the same protocol. Figure 5 compares the two sources at equal budget $B = C \cdot N$ . Per task, real data is worth more at smaller budgets: up to $2 ^ { 1 8 }$ subsets selected on the synthetic data show lower win rates, but from $2 ^ { 1 9 }$ to $2 ^ { 2 0 }$ on, synthetic data overtakes TabRepo. TabRepo saturates: over its last two doublings, to 16,000 configurations, its curve gains at most 0.008. We attribute this to the TabRepo size: with 182 datasets, the subsets start to overfit the source datasets. Synthetic data allows us to sidestep this, because we can continue scaling the data size indefinitely: at $2 ^ { 2 4 }$ it ends 0.02 to 0.08 above TabRepo’s best for every model and every K. At K = 25 and 64 the crossover lies between $2 ^ { 2 0 }$ and $2 ^ { 2 2 }$ (Appendix A.4).

![](images/490c4ae2cd6eb69a5ff409af36eae983c926563a385c7f2311cddb238772e53d.jpg)  
Random K0 of 16,000 (mean, 95% of 200 draws) Meta-tuned K (synthetic) Meta-tuned K (TabRepo)  
Figure 6 | TabArena, T deployment, $K = 2 0 0$ . Random subsets of growing size $K ^ { \prime }$ against the random-K baseline. Solid curve and band: mean and 95% CI over 200 random subsets of size $K ^ { \prime }$ . Horizontal lines are the K configurations meta-tuned on all 4,000 configurations, with 4,000 synthetic tasks or the 182 TabRepo datasets, averaged against the same baselines.

## 5.3 Meta-tuning versus more random search

A cheaper alternative to meta-tuning is to fit more random configurations. Figure 6 scores random subsets of size $K ^ { \prime }$ against the random-K baseline (details in Appendix A.1.2.3). The mean curve rises, peaks at 0.50 to 0.53 for $K ^ { \prime }$ between 500 and 750, and falls to 0.49 to 0.51 at $K ^ { \prime } = 4 { , } 0 0 0 { : }$ : larger random subsets overfit the validation split. At K = 25 and 64, the mean curve peaks higher, at 0.51 to 0.60, but the meta-tuned subset still sits at least 0.07 above it (Appendix A.4). For a practitioner with a fixed budget, the 200 meta-tuned configurations replace a random search over 4,000 configurations that fits twenty times more models and still scores lower on average.

## 5.4 Choice of prior

We repeat the selection on synthetic tasks from the TabICLv1 [28] and TabPFNv1 [1] priors (Appendix A.3), with 2,000 configurations on 2,000 datasets for LightGBM, LightGBM-TabPrep, and the MLP (Figure 7). With all 2,000 configurations and tasks, the prior matters most for plain LightGBM: TabICLv2 leads at every K, by up to 0.09 over TabPFNv1 (0.644 against 0.551 at $K = 2 5 )$ . With TabPrep, the three priors are within 0.02 of each other at every K. For the MLP, TabICLv2 wins at $K = 2 5$ and 64 and TabPFNv1 at $K = 2 0 0$ . No prior wins everywhere, and the prior of TabPFNv1 appears to be competitive. We use TabICLv2, the newest prior, for the main experiments.

## 5.5 Scaling

Figure 8 shows the win rate over the full (C, N) grid. It rises at almost every step along both axes, and neither axis has saturated at 4,000. Appendix B motivates two forms: the estimation error over N synthetic tasks decays as $1 / \sqrt { N }$ (Proposition 2), and the number of improvements available in a pool grows as log C (Proposition 1). We fit both forms to the curves at $C = 4 { , } 0 0 0$ and $N = 4 { , } 0 0 0$ and obtain $R ^ { 2 }$ between 0.92 and 1.00 for $a - b / \sqrt { N }$ and between 0.90 and 0.98 for $a + b \log _ { 2 } C$ , over all three models and all K. In Figures 17 and 18 in the appendix, the curve in C is steeper than the curve in N for every model: at $K = 2 0 0$ , a doubling of C is worth 1.2 to 1.7 doublings of N, and the curve in N tends to flatten above about 1,000 tasks. Once there are a few thousand synthetic tasks, more configurations are the better use of a fixed budget. Tasks reduce the estimation error of the objective, whereas configurations add candidates worth finding.

![](images/9af5512b52a59ce57cd49c85890c49ee7333ff64f727c1911d7f1f5e324c0e11.jpg)

Figure 7 | TabArena, T deployment, win rate against a random subset of the same size, $K = 2 0 0$ Three priors, each with $2 { , } 0 0 0$ configurations on 2,000 synthetic tasks, for LightGBM, LightGBM-TabPrep, and the MLP. Top: as C grows at $N = 2 { , } 0 0 0$ ; bottom: as N grows at $C = 2 { , } 0 0 0$ . Win rates are averaged over 2,000 repetitions.  
![](images/0f0305cc9ccd0846261e476f7666528958ec4e062022fb86c1e8f326eeaf1ec3.jpg)  
Figure 8 | TabArena, T deployment, win rate against a random subset of the same size over the (C, N) grid, $K = 2 0 0$ . Win rates are averaged over 2,000 repetitions.

## 5.6 Selected hyperparameters

Figure 9 compares the selected configurations with the distribution that we sample them from. For LightGBM, the selected configurations prefer more expressive trees, with a small minimum data in a leaf and weak L1 regularization, together with a small learning rate: 21% of the selected learning rates are below the 0.1 quantile of the sampling distribution. For the MLP, the pattern is similar. The selected configurations prefer small learning rates, with 45% of the Muon learning rates below the 0.1 quantile, and larger embeddings: the largest dimension, 32, has a share of 21% against 14% in the sampling. For these hyperparameters, the range or the shape of the search space is thus possibly a poor fit, and a wider or reshaped space can contain better configurations.

![](images/57fb40ed4ae99e1ab66c96ab9c33ac04e221de6982490239b25dc2d6ba0a6aaa.jpg)  
Sampling distribution 200 selected configurations (model colour)  
Figure 9 | Selected hyperparameters against their sampling distribution: K = 200, T regime, selected from the full pool of 4000 configurations on 4000 synthetic datasets. Lines: density of the 200 selected configurations divided by the sampling density (grey, flat at 1); ticks: selected values; bars: shares (grey: sampling). All hyperparameters, also of LightGBM-TabPrep, are in Figures 19–21.

## 6. Conclusion

We have shown that the synthetic priors built for PFN pretraining are a scalable source of meta-training tasks for conventional tabular models. PriorTune selects a subset of hyperparameter configurations on synthetic tasks alone, by win rate against random subsets under the deployment rule. We show that when scaled the synthetic tasks beat real datasets of TabRepo at equal budgets. The selected subsets beat random subsets of any size, improve LightGBM, LightGBM-TabPrep, the MLP, and TabPack on the TabArena and BeyondArena benchmarks, and make LightGBM-TabPrep the highest-ranked model in Elo on BeyondArena that is not a pretrained foundation model, above TabPFN-3 and RealMLP, at lower latency than both. The released subsets are drop-in search spaces: a practitioner replaces random search over the published space with the K selected configurations.

## 7. Limitations

We do not apply PriorTune to tabular foundation models. They are pretrained on tasks from the same kind of prior, so the selection can prefer configurations that change them the least. Newer priors do not always help PriorTune: the newest prior helps plain LightGBM most but gives no clear gain for LightGBM-TabPrep and the MLP (Section 5.4). The grid is fitted once per model and search space, so a new search space needs a new grid. The synthetic tasks have at most 60,000 rows and 100 features, so the selection does not see tasks as large as the largest real datasets. The ablations use the tuned rule only, because selection for ensembles is expensive. The error bars measure variation across repetitions, not uncertainty over new datasets. Our theory motivates the forms of the scaling curves but does not prove them; the fits are empirical.

## AI use statement

We used generative AI tools (Claude Code, Codex) to implement parts of the experiment and analysis code, to draft and edit parts of the text, and to search for and format references. We did not use them to generate data, develop the theory, prove claims, propose hypotheses, or design the method and experiments. The authors checked and tested all code, cross-checked every number against the experiment logs, verified all references, and reviewed all text. We take full responsibility for the content of this work.

## Reproducibility statement

Section 3 and Algorithm 1 specify the selection procedure and its fixed settings (R = 16, tie rule, 40 Caruana steps). Appendix A.1 gives the evaluation protocol on both benchmarks, the repetition scheme of the ablations, the compute used, and the latency measurement. Appendix A.2 lists the complete hyperparameter search space of every model. Appendix A.3 describes the three synthetic priors and Table 9 the exact sampling settings and seeds. Appendix A.4 lists the TabRepo datasets excluded for overlap with TabArena. Appendix B states all assumptions and gives complete proofs of the propositions. The benchmarks (TabArena, BeyondArena, TabRepo) are public. The supplementary material contains the code, the selected subsets for every model and K, the seeds, and the score caches of all fits, from which every table and figure can be regenerated.

## References

[1] Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer that solves small tabular classification problems in a second. In International Conference on Learning Representations (ICLR), 2023.

[2] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637:319–326, 2025.

[3] Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICLv2: A better, faster, scalable, and open tabular foundation model. In International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/abs/2602.11139.

[4] Léo Grinsztajn, Klemens Flöge, Oscar Key, et al. TabPFN-3: Technical report. arXiv preprint arXiv:2605.13986, 2026.

[5] David Salinas and Nick Erickson. TabRepo: A large scale repository of tabular model evaluations and its AutoML applications. In Proceedings of the Third International Conference on Automated Machine Learning, volume 256 of Proceedings of Machine Learning Research, pages 19/1–30, 2024. URL https://proceedings.mlr.press/ v256/salinas24a.html.

[6] David Holzmüller, Léo Grinsztajn, and Ingo Steinwart. Better by default: Strong pre-tuned MLPs and boosted trees on tabular data. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[7] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. LightGBM: A highly eficient gradient boosting decision tree. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

[8] Andrej Tschalzev, Nick Erickson, Yuyang Wang, Huzefa Rangwala, Stefan Lüdtke, Heiner Stuckenschmidt, and Christian Bartelt. TabPrep: Closing the feature engineering gap in tabular benchmarks. arXiv preprint arXiv:2606.02384, 2026.

[9] Yury Gorishniy, Ivan Rubachev, and Artem Babenko. On embeddings for numerical features in tabular deep learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[10] Yury Gorishniy, Akim Kotelnikov, Ivan Rubachev, and Artem Babenko. TabPack: Eficient hyperparameter ensembles for tabular deep learning. In International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/abs/2607.05380.

[11] Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas, and Frank Hutter. TabArena: A living benchmark for machine learning on tabular data. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025. URL https://arxiv.org/abs/ 2506.16791.

[12] Lennart Purucker, Andrej Tschalzev, Nick Erickson, Blayer, David Holzmüller, Arazi, Pfeferle, Tajjar, Gaël Varoquaux, and Frank Hutter. Beyond IID: How general are tabular foundation models, really? arXiv preprint arXiv:2606.30410, 2026.

[13] Matthias Feurer, Jost Tobias Springenberg, and Frank Hutter. Initializing Bayesian hyperparameter optimization via meta-learning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1128–1135, 2015. doi: 10.1609/aaai.v29i1.9354.

[14] Martin Wistuba, Nicolas Schilling, and Lars Schmidt-Thieme. Sequential model-free hyperparameter tuning. In IEEE International Conference on Data Mining (ICDM), pages 1033–1038, 2015. doi: 10.1109/ICDM.2015.20.

[15] Florian Pfisterer, Jan N. van Rijn, Philipp Probst, Andreas C. Müller, and Bernd Bischl. Learning multiple defaults for machine learning algorithms. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO), pages 241–242, 2021. doi: 10.1145/3449726.3459523. Extended version: arXiv:1811.09409.

[16] Fela Winkelmolen, Nikita Ivkin, H. Furkan Bozkurt, and Zohar Karnin. Practical and sample eficient zero-shot HPO. arXiv preprint arXiv:2007.13382, 2020. URL https://arxiv.org/abs/2007.13382.

[17] Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, Mu Li, and Alexander Smola. AutoGluon-Tabular: Robust and accurate AutoML for structured data. arXiv preprint arXiv:2003.06505, 2020. URL https://arxiv.org/abs/2003.06505.

[18] Valerio Perrone, Rodolphe Jenatton, Matthias W. Seeger, and Cédric Archambeau. Scalable hyperparameter transfer learning. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

[19] Matthias Feurer, Benjamin Letham, Frank Hutter, and Eytan Bakshy. Practical transfer learning for Bayesian optimization. arXiv preprint arXiv:1802.02219, 2018. URL https://arxiv.org/abs/1802.02219.

[20] Martin Wistuba and Josif Grabocka. Few-shot Bayesian optimization with deep kernel surrogates. In International Conference on Learning Representations (ICLR), 2021. URL https://arxiv.org/abs/2101.07667.

[21] Zi Wang, George E. Dahl, Kevin Swersky, Chansoo Lee, Zachary Nado, Justin Gilmer, Jasper Snoek, and Zoubin Ghahramani. Pre-trained Gaussian processes for Bayesian optimization. Journal of Machine Learning Research, 25(212):1–83, 2024. URL https://arxiv.org/abs/2109.08215.

[22] Zhou Fan, Xinran Han, and Zi Wang. HyperBO+: Pre-training a universal prior for Bayesian optimization with hierarchical Gaussian processes. arXiv preprint arXiv:2212.10538, 2023. URL https://arxiv.org/abs/2212.10538. Full version of a NeurIPS 2022 workshop paper.

[23] Yutian Chen, Xingyou Song, Chansoo Lee, Zi Wang, Qiuyi Zhang, David Dohan, Kazuya Kawakami, Greg Kochanski, Arnaud Doucet, Marc’Aurelio Ranzato, Sagi Perel, and Nando de Freitas. Towards learning universal hyperparameter optimizers with transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2205.13320.

[24] Samuel Müller, Matthias Feurer, Noah Hollmann, and Frank Hutter. PFNs4BO: Incontext learning for Bayesian optimization. In International Conference on Machine Learning (ICML), volume 202 of Proceedings of Machine Learning Research, pages 25444–25470, 2023. URL https://proceedings.mlr.press/v202/muller23a.html.

[25] Rosen Ting-Ying Yu, Cyril Picard, and Faez Ahmed. GIT-BO: High-dimensional Bayesian optimization with tabular foundation models. In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/2505.20685.

[26] Theodore Rogers and Srividya Ponnada. Zero-shot Bayesian optimization with TabPFN: Competitive with state-of-the-art without per-task training. In International Conference

on Automated Machine Learning (AutoML), 2026. Listed on the Amazon Science publications page; no arXiv version found.

[27] Xinyu Zhang, Conor Hassan, Julien Martinelli, Daolang Huang, and Samuel Kaski. In-context multi-objective optimization. In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/2512.11114.

[28] Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICL: A tabular foundation model for in-context learning on large data. In International Conference on Machine Learning (ICML), 2025.

[29] Xiyuan Zhang, Danielle C. Maddix, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W. Mahoney, Cuixiong Hu, Huzefa Rangwala, George Karypis, and Bernie Wang. Mitra: Mixed synthetic priors for enhancing tabular foundation models. In Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv.org/abs/2510. 21204.

[30] Xingxuan Zhang, Gang Ren, Han Yu, et al. LimiX: Unleashing structured-data modeling capability for generalist intelligence. arXiv preprint arXiv:2509.03505, 2025. URL https://arxiv.org/abs/2509.03505.

[31] Penghui Wang, Wei Liu, Hong Wang, Chengyue Huang, Yuxi Sun, Zirui Wang, Hongming Huang, Quan Wang, Zhenwei Xin, Ping Hou, Jie Yu, Chunxiao Liu, Erli Meng, and Bin Wang. Xiaomi-TabLDM: A tabular foundation model technical report. arXiv preprint arXiv:2609.03880, 2026. URL https://arxiv.org/abs/2609.03880.

[32] Minyong Cho, Minho Jeong, Dooho Lee, Jinmo Lee, and Jaemin Yoo. Causilo technical report. arXiv preprint arXiv:2609.22866, 2026. URL https://arxiv.org/abs/2609. 22866.

[33] Moonjung Eo, Min-Kook Suh, Hye-Seung Cho, Jiwon Kim, Seoyoon Kim, Sangjun Nam, and Soonyoung Lee. EXAONE Tabular 1.0: Technical report. arXiv preprint arXiv:2608.25774, 2026. URL https://arxiv.org/abs/2608.25774.

[34] Aditya Tanna, Nassim Bouarour, Mohamed Bouadi, Vinay Kumar Sankarapu, and Pratinav Seth. Pocket foundation models: Distilling TFMs into CPU-ready gradientboosted trees. arXiv preprint arXiv:2605.18654, 2026. URL https://arxiv.org/abs/ 2605.18654.

[35] Qi Qin, Jiajie Zhu, Dali Chen, Yuzhao Zhang, Jia-Xing Han, Peng Zhang, Ying Yan, Yifan Sun, and Yu Su. GEAR: Generative expansion and real anchoring for two-stage distillation of tabular foundation models. arXiv preprint arXiv:2608.18849, 2026. URL https://arxiv.org/abs/2608.18849.

[36] Rich Caruana, Alexandru Niculescu-Mizil, Geof Crew, and Alex Ksikes. Ensemble selection from libraries of models. In Proceedings of the Twenty-First International Conference on Machine Learning (ICML), 2004.

[37] Rong-Xi Tan, Ke Xue, Shen-Huan Lyu, Haopu Shang, Yao Wang, Yaoyuan Wang, Sheng Fu, and Chao Qian. Ofline model-based optimization by learning to rank. In International Conference on Learning Representations (ICLR), 2025. URL https: //arxiv.org/abs/2410.11502.

[38] NVIDIA. Kumo relational and kumo tabular: Overview. https://docs.nvidia.com/ sdgm/kumo-relational/about-kumo-relational/overview, 2026. Accessed 2026- 10-30.

[39] Andrej Tschalzev, Stefan Lüdtke, Heiner Stuckenschmidt, and Christian Bartelt. Lost in aggregation: How benchmarks overlook irreplaceable model strengths. arXiv preprint arXiv:2608.18919, 2026.

[40] Alfréd Rényi. Théorie des éléments saillants d’une suite d’observations. Annales scientifiques de l’Université de Clermont-Ferrand 2, 8(2):7–13, 1962.

## Appendix contents

A Computation 18   
A.1 Setup 18   
A.1.1 Main result 18   
A.1.2 Ablations 18   
A.1.3 Compute 20   
A.1.4 Inference latency 21   
A.2 Hyperparameter search spaces . 22   
A.3 Synthetic priors 23   
A.3.1 Graph structure 24   
A.3.2 Edge functions 24   
A.3.3 Transformations and filters 25   
A.3.4 Sampling settings 25   
A.4 Additional results 26   
A.4.1 Main results 26   
A.4.2 Selection rule 27   
A.4.3 Synthetic versus real meta-training data . 30   
A.4.4 Meta-tuning versus more random search . . 30   
A.4.5 Choice of prior 31   
A.4.6 Scaling 32   
A.4.7 Selected hyperparameters 36   
B Theory 39   
B.1 Setup and main propositions . 39   
B.1.1 Setup 39   
B.1.2 Upper bound on the improvement 40   
B.1.3 Transfer from synthetic to real tasks 40   
B.1.4 Analysis of the forward-selection heuristic 41   
B.2 Proofs 42

## A. Computation

## A.1 Setup

## A.1.1 Main results

Datasets. Table 2 shows distribution of features and sample sizes of both benchmarks. For full list of datasets, their selection criteria, and description, see the original work [11, 12].

Table 2 | Benchmark statistics. Row and feature counts are per dataset (minimum / median / maximum).
<table><tr><td></td><td>TabArena</td><td>BeyondArena</td></tr><tr><td>Datasets</td><td>51</td><td>142</td></tr><tr><td>Binary / multiclass / regression</td><td>30 / 8 /13</td><td>73 / 25 / 44</td></tr><tr><td>Splits: i.i.d. / grouped / temporal</td><td>51/0/0</td><td>103 /18 / 21</td></tr><tr><td>Rows</td><td>748 / 6,497 / 150,000</td><td>155 / 5,000 /1,296,316</td></tr><tr><td>Features</td><td>4/17/1,776</td><td>3 / 23 / 22,215</td></tr></table>

Evaluation. On BeyondArena, we use every outer split of the core benchmark. Within each outer split, every method is bagged over the 8 inner folds, and each configuration is trained once per fold. T deploys the validation-best configuration; T+E fits a 40-step Caruana ensemble on the same validation predictions. Both score the held-out test split with the native metric.

Subsets. FT subsets are selected with the T objective and FTE subsets with the T+E objective (40 Caruana steps inside the selection), both with R = 16 on all 4,000 configurations and 4,000 synthetic tasks. No benchmark data enters the selection. TabPack’s FTE subset is the K = 64 subset selected with the T+E objective on the MLP pool and deployed as one pack of those members.

## A.1.2 Ablations

All ablations select under the T deployment rule, deploy under T, and report the win rate against a random-K baseline.

## A.1.2.1 Selection rule (Section 5.1)

Protocol. We run 200 repetitions. Each splits the 4,000 synthetic tasks at random into 3,000 selection tasks and 1,000 held-out tasks. Every rule selects K out of 4,000 configurations on the selection tasks and is deployed under T on the held-out tasks. There it is compared with 200 random-K baselines, and its win rate is averaged over them. All rules in a repetition use the same tasks and baselines. Error bars in Figure 4 are ±1.96 standard errors of the win rate over the 200 repetitions.

Our rule. It draws, once per repetition, 16 bootstrap resamples of each task’s validation rows and 16 random-K references $\mathcal { R } _ { r }$ . R < 16 pairs the first R resamples with the first R references. Validation only pairs the 16 resamples with $\mathcal { R } _ { 1 }$ , and reference only pairs the original validation split with the 16 references.

TabRepo algorithms. They use no validation resamples and no references. They score the best member of a subset on each dataset’s joined validation and test rows, with the native metric recomputed on the union. The first version adds, at each step, the configuration that minimizes the mean over datasets of the best normalized error, $( \ell - \ell _ { \mathrm { b e s t } } ) / ( \ell _ { \mathrm { m e d i a n } } - \ell _ { \mathrm { b e s t } } )$ clipped to [0, 1], with $\ell _ { \mathrm { b e s t } }$ and $\ell _ { \mathrm { m e d i a n } }$ taken over the 4,000 configurations. The second starts from the configuration with the lowest mean rank, then adds the one that improves the best member on the most tasks. Both break ties by a random configuration order drawn per repetition.

## A.1.2.2 Synthetic versus real meta-training data (Section 5.2)

Cells. A cell (C, N) selects K of C configurations on N meta-training tasks by Algorithm 1 with R = 16. Synthetic cells use our pool of 4,000 configurations and 4,000 synthetic tasks. TabRepo cells use the 182 TabRepo datasets and a bank of 16,000 configurations: our pool of 4,000 plus 12,000 further draws from the same search spaces, fitted on TabRepo and TabArena. Validation resamples are drawn once per task and shared across repetitions. Ties in validation selection and in the greedy step go to the lowest configuration id.

Repetitions. We run 2,000 repetitions, shared by all cells. A repetition permutes the configurations and the tasks once; each cell takes the first C configurations and N tasks of that permutation.

Evaluation. The selected subset is deployed on TabArena under T and compared with the first K configurations of the repetition’s permutation of our 4,000-configuration pool. Both synthetic and TabRepo cells share this baseline. Also both synthetic and TabRepo $C = K$ cells therefore have 0.5 by construction.

Reported values. Each win rate is the mean over the 2,000 repetitions, its standard error is below 0.003 in every cell and is not drawn. It measures variation across repetitions on the fixed benchmark, not uncertainty over new datasets. In Figure 5, each cell $( C , N )$ is a dot at its budget $B = C \cdot N ,$ so the dots above one B are all cells with that budget. The line joins the best cells within each B.

## A.1.2.3 Meta-tuning versus more random search (Section 5.3)

Meta-tuned lines. The synthetic line is the cell $C = N = 4 { , } 0 0 0$ of Appendix A.1.2.2. The TabRepo line selects on all 182 TabRepo datasets from our pool of 4,000 configurations, with the same protocol, repetitions, and baselines. Each line is the mean win rate over the 2,000 repetitions.

Random subsets. For every $K ^ { \prime }$ , we draw 200 subsets of $K ^ { \prime }$ configurations uniformly from the 16,000-configuration bank, independently across $K ^ { \prime } { } _ { i }$ , and deploy each under T on TabArena. Each subset is compared with all 2,000 random-K baselines of the meta-tuned lines, and its win rate is averaged over them. The band therefore reflects only the draw of the random subset: we report the mean over the 200 subsets and the 95% CI band.

## A.1.2.4 Choice of prior (Section 5.4)

Cells. For LightGBM, LightGBM-TabPrep, and the MLP, a pool of 2,000 configurations is fitted on 2,000 tasks (datasets) from each of TabICLv2, TabICLv1, and TabPFNv1 (Appendix A.3). Within a model, the configuration pool and the TabArena evaluation are shared, so only the priors afect the selection of the K configurations. A cell $( C , N )$ selects K of C configurations on N tasks by Algorithm 1 with R = 16.

Repetitions and evaluation. As in Appendix A.1.2.2: 2,000 repetitions shared by all cells, each permuting the configurations and the tasks once. The selected subset is deployed on

TabArena under T and compared with the first K configurations of the permutation. Win rates are means over the repetitions, with standard errors below 0.003.

## A.1.2.5 Scaling (Section 5.5)

Cells. The synthetic cells of Appendix A.1.2.2, with all 4,000 configurations and 4,000 synthetic tasks available: C and N range from K to 4,000, with the same 2,000 repetitions, baselines, and reported values.

Fits. We fit $a - b / \sqrt { N }$ to the win rate against N at $C = 4 { , } 0 0 0$ and $a + b \log _ { 2 } C$ to the win rate against C at $N = 4 { , } 0 0 0$ , by least squares on the means; the $C = K$ cell, 0.5 by construction, is left out of the second fit.

## A.1.2.6 Selected hyperparameters (Section 5.6)

Subset. We take the K = 200 subset selected under T on all 4,000 configurations and 4,000 synthetic tasks. For TabPrep settings, only the selected configurations with TabPrep enabled count.

Distributions. For each hyperparameter, the sampling distribution F is given in Appendix A.2. For a continuous hyperparameter, we plot the density of the selected values divided by the sampling density, $f _ { \mathrm { s e l } } ( x ) / f ( x )$ . The sampling distribution is then flat at 1, and a value above 1 is selected more often than it is sampled from the prior. We estimate this ratio with a Gaussian kernel (Scott bandwidth) and plot it as the coloured line. The Kolmogorov–Smirnov (KS) distance is $\begin{array} { r } { \operatorname* { s u p } _ { x } | \hat { F } _ { \mathrm { s e l } } ( x ) - F ( x ) | } \end{array}$ , where $\hat { F } _ { \mathrm { s e l } }$ is the empirical distribution function of the selected values. It is in $[ 0 , 1 ] \colon 0$ when the selected values follow the sampling distribution exactly, and larger when they move away from it. Categorical hyperparameters are shown as shares.

## A.1.3 Compute

We report compute in A100 hours. CPU work is converted at AWS on-demand prices:<sup>1</sup> 1 A100 hour costs as much as 61.5 vCPU-hours. Table 3 summarizes the costs.

PriorTune, once per model. (i) Configuration×task grid. We fit MLPs with the tensorized code of TabPack [10]: one fit takes 0.1 A100 seconds on average on the synthetic tasks, so the 16 million fits of the $4 , 0 0 0 \times 4 , 0 0 0$ grid take 400 A100 hours. One LightGBM-TabPrep fit takes about 5 core-seconds, so its grid takes 20,000 core-hours, about 300 A100 hours. (ii) Selection. Under T, selecting a subset from the cached predictions takes seconds. Under T+E, selecting 200 configurations takes about 10 A100 hours.

PriorTune, per dataset. A new dataset fits only the K selected configurations, so the cost equals that of tuning with K random ones. As a proxy, we take the benchmark’s tuned LightGBM on BeyondArena-Lite (26 configurations, 8 folds, 16 vCPUs): 8 vCPU-hours per dataset on average (median 2.4), that is, 0.1 A100 hours (median 0.04).

PFNs, once. Pretraining TabICLv2 takes 24.5 H100-days, about 1,180 A100 hours [3]. TabICL takes 20 days on three A100s, 1,440 A100 hours [28]; TabPFNv2 takes about two weeks on eight RTX 2080 Ti, 2,690 GPU-hours [2], or about 460 A100 hours. TabPFN-3 does not report its pretraining compute [4].

PFNs, per dataset. In-context learning replaces fitting with one fit-and-predict pass. On BeyondArena-Lite, TabICLv2 takes 0.48 A100 hours per dataset on average (median 26 seconds). TabPFN-3 takes 0.70 GPU-hours on average (median 53 seconds).

Table 3 | Compute in A100 hours; CPU work converted at 61.5 vCPU-hours per A100 hour. Perdataset costs are means over the 142 BeyondArena-Lite datasets, medians in parentheses.
<table><tr><td>PriorTune</td><td></td><td>PFN</td></tr><tr><td rowspan="4">Once per model</td><td>Grid: 400 (MLP)</td><td>Pretraining: 1,180 (TabICLv2)</td></tr><tr><td>300 (LightGBM-TabPrep)</td><td>1,440 (TabICL)</td></tr><tr><td>Selection: &lt; 0.01 (T), 10 (T+E, K = 200)</td><td>460 (TabPFNv2)</td></tr><tr><td></td><td>not reported (TabPFN-3)</td></tr><tr><td>Per dataset</td><td>Fit K = 25: 0.1 (0.04), LightGBM proxy</td><td>Fit and predict: 0.48 (0.007), TabICLv2</td></tr></table>

## A.1.4 Inference latency

Datasets and fits. We measure latency (Figure 1) on 10 BeyondArena datasets that cover its dataset groups (Table 4). For each dataset, we fix one outer train-test split. Under T, we deploy the validation-best configuration. Under T+E, we build the 40-step greedy ensemble of TabArena on the validation predictions. Ensemble members predict sequentially, and the ensemble averages their predictions. TabPack tensorizes the ensemble. In-context models use the inner training and validation splits as context and run at default settings, which include their internal ensembling: 8 estimators for TabPFN-3 and TabICLv2, and 16, 8, and 8 for the large, medium, and small Kumo-Tabular. The Elo of Figure 1 comes from the full BeyondArena.

Table 4 | Datasets of the latency measurement. Rows are those of the outer training split.
<table><tr><td>Dataset</td><td>Task</td><td>Split</td><td>Rows</td><td>Features</td></tr><tr><td>credit-g</td><td>binary</td><td>i.i.d.</td><td>666</td><td>20</td></tr><tr><td>mice-protein</td><td>multiclass</td><td>grouped</td><td>720</td><td>77</td></tr><tr><td>coffee-rating</td><td>regression</td><td>temporal</td><td>2,137</td><td>12</td></tr><tr><td>predict-students</td><td>multiclass</td><td>i.i.d.</td><td>2,949</td><td>36</td></tr><tr><td>churn</td><td>binary</td><td>i.i.d.</td><td>3,333</td><td>19</td></tr><tr><td>emscad</td><td>binary</td><td>i.i.d.</td><td>11,640</td><td>17</td></tr><tr><td>video-transcoding</td><td>regression</td><td>grouped</td><td>44,897</td><td>19</td></tr><tr><td>sdss17</td><td>multiclass</td><td>i.i.d.</td><td>52,035</td><td>11</td></tr><tr><td>sf-permit-time</td><td>regression</td><td>temporal</td><td>60,784</td><td>37</td></tr><tr><td>kick</td><td>binary</td><td>temporal</td><td>68,012</td><td>32</td></tr></table>

Timing. For each method and dataset, the fitted model is kept in memory. We do 5 warm-up predict calls on single test rows and then time 50 calls. We report the median over calls and then over the 10 datasets. All methods run on one host. GBDTs run on 16 CPU threads. DNNs and in-context models run on one GPU.

KV cache. We time each PFN in three modes. In recompute, every call re-encodes the training context. With a key-value (KV) cache, the encoded context is computed once at fit time and kept on either the GPU or on the CPU. Table 5 lists the options. The KV cache of

TabPFN-3 is quantized to int8 by default, so its cached predictions difer from recompute by up to 0.14 in probability. For TabICLv2, they difer by up to 0.015.  
Table 5 | Options of the in-context models for the three timed modes. TabPFN-3 and TabICLv2 take them in the constructor (tabpfn 9.0.0, tabicl 2.2.0). Kumo-Tabular takes them in the TabArena wrapper.
<table><tr><td>Model</td><td>Recompute</td><td>KV cache on GPU</td><td>KV cache on CPU</td></tr><tr><td>TabPFN-3</td><td>fit_mode= &quot;fit_preprocessors&quot;a</td><td>fit_mode=&quot;fit_with_cache&quot; keep_cache_on_device=True keep_cache_on_device=False</td><td>fit_mode=&quot;fit_with_cache&quot;</td></tr><tr><td>TabICLv2</td><td>kv_cache=Falsea</td><td>kv_cache=True</td><td>kv_cache=Trueb</td></tr><tr><td></td><td>Kumo-Tabular cache_context=False</td><td> $\mathtt { c a c h e \_ c o n t e } \times t = \mathtt { T } r \mathsf { u e } ^ { c }$ </td><td>cache_context=Truea</td></tr></table>

<sup>a</sup>Default. <sup>b</sup>TabICL has no option to keep the cache on the CPU: we move it to the CPU after fit and copy it to the GPU in every call. <sup>c</sup>The wrapper keeps the cache on the CPU: we move it to the GPU once after fit.

Inference with TabPrep. For TabPrep, we compile the feature generation in Numba and make it faster in two ways: (i) the fitted feature generators become lookup tables, and (ii) we compute only the generated features that the booster (LightGBM) splits on. The compiled predictor gives identical predictions. Without compilation, LightGBM-TabPrep under T costs 44 to 51 ms per row through AutoGluon, against 0.8 to 0.9 ms compiled, 51 to 54 times slower.

## A.2 Hyperparameter search spaces

Every pool is a set of independent random draws from the search space below; no configuration is tuned on any dataset. The same draws are fitted on every synthetic and real task.

LightGBM. We use the LightGBM search space of TabArena [11], shown in Table 6. Training follows the AutoGluon LGBModel semantics: up to 10,000 boosting rounds with AutoGluon’s adaptive early stopping (ofset 20, rate 0.3, maximum patience 300 rounds, scaled down above 10,000 training rows) on the validation split. The pool has C = 4,000 configurations.

LightGBM with TabPrep. The same LightGBM space, with the TabPrep feature generators of Tschalzev et al. [8] prepended to the pipeline. Following the published protocol, TabPrep is enabled for one configuration in four; an enabled configuration also draws the eight TabPrep axes in Table 7. The pool has C = 4,000 configurations and shares its LightGBM draws with the plain LightGBM pool, so the two pools difer only in preprocessing.

MLP with cosine embeddings. The MLP is the independent member model of TabPack [10]: a feed-forward MLP with cosine numerical embeddings [9], trained with the Muon optimizer for the hidden weights and AdamW for the rest. The pool is a frozen registry of C = 4,000 draws from the space in Table 8. The seven sampled axes follow TabPack’s cosine base model; the hidden width is fixed at 384. Members are trained individually with per-member optimizer steps, up to 256 epochs, and early stopping after 16 non-improving validations, without TabPack’s online ensemble.

TabPack. TabPack [10] trains 64 MLP members as one packed tensor program and combines them with the greedy online ensemble of the TabPack paper. The members are the first 64 configurations of the MLP registry above. A meta-tuned TabPack is a pack of K = 64 configurations selected from the same pool of 4,000 MLP configurations with the T+E objective and deployed with the same online ensemble; Figure 1 and Figure 3 use this subset.

Table 6 | The hyperparameter tuning space for LightGBM.
<table><tr><td>Parameter</td><td>Distribution</td></tr><tr><td>Learning rate</td><td>LogUniform[5e-3, 1e-1]</td></tr><tr><td>Feature fraction</td><td>Uniform[0.4, 1.0]</td></tr><tr><td>Bagging fraction</td><td>Uniform[0.7, 1.0]</td></tr><tr><td>Bagging frequency</td><td>1</td></tr><tr><td># leaves Min. data in leaf</td><td>LogUniformInt[2, 200]</td></tr><tr><td>Extra trees</td><td>LogUniformInt[1, 64] {False, True}</td></tr><tr><td>Min. data per group</td><td></td></tr><tr><td>Categorical L2</td><td>LogUniformInt[2, 100]</td></tr><tr><td></td><td>LogUniform[5e-3, 2.0]</td></tr><tr><td>Categorical smoothing</td><td>LogUniform[1e-3, 100]</td></tr><tr><td>Max. categories to one-hot</td><td>LogUniformInt[8, 100]</td></tr><tr><td>L1 regularization</td><td>Uniform[1e-4, 1.0]</td></tr><tr><td>L2 regularization</td><td>Uniform[1e-4, 2.0]</td></tr><tr><td>Max. boosting rounds</td><td>10,000</td></tr><tr><td>Early stopping</td><td>AutoGluon adaptive (offset 20, rate 0.3, max. 300)</td></tr></table>

Table 7 | The TabPrep axes drawn for the one-in-four configurations with TabPrep enabled.
<table><tr><td>Parameter</td><td>Distribution</td></tr><tr><td>TabPrep enabled</td><td>Bernoulli(0.25)</td></tr><tr><td>OpenFE max. features</td><td>{2000, 1000}</td></tr><tr><td>OpenFE generation seed</td><td>{42, 84, 168, 336, 672}</td></tr><tr><td>Cat. combination max. features</td><td>{100, 500}</td></tr><tr><td>Cat. combination generation seed Random-subset feature combination # subsets</td><td>{42, 84, 168, 336, 672}</td></tr><tr><td>Random-subset generation seed</td><td>{50, 1, 100}</td></tr><tr><td>OOF target encoding generation seed</td><td>{42, 84, 168, 336, 672}</td></tr><tr><td></td><td>{42, 84, 168, 336, 672}</td></tr><tr><td>Group-by max. features</td><td>{500, 100, 1000}</td></tr></table>

## A.3 Synthetic priors

All three priors are structural causal models: a directed acyclic graph over latent variables, a random function per edge set, and an observation process that turns node values into a table. This section lists the three parts for each prior.

Table 8 | The hyperparameter tuning space for the MLP with cosine embeddings.
<table><tr><td>Parameter</td><td>Distribution</td></tr><tr><td># blocks</td><td>UniformInt[1, 4]</td></tr><tr><td>Block width</td><td>384</td></tr><tr><td>Dropout rate</td><td>{0.0, Uniform[0.0, 0.5]}</td></tr><tr><td>Embedding dim</td><td>UniformInt[8, 32], step=4</td></tr><tr><td>Embedding init. scale Learning rate (AdamW)</td><td>LogUniform[1e-2, 1e1]</td></tr><tr><td>Learning rate (Muon)</td><td>LogUniform[1e-4, 5e-3]</td></tr><tr><td></td><td>LogUniform[1e-3, 1e-1]</td></tr><tr><td>Weight decay Batch size</td><td>LogUniform[1e-3, 1.0]</td></tr><tr><td>Max epochs</td><td>dataset-dependent (TabArena rule)</td></tr><tr><td></td><td>256</td></tr><tr><td>Early-stopping patience</td><td>16</td></tr></table>

## A.3.1 Graph structure

TabICLv2. A random Cauchy DAG with LogInt(2, 32) nodes. The probability of an edge from node i to node j is $\sigma ( A + B _ { i } + C _ { j } )$ , where $A , B _ { i } , C _ { j }$ are independent standard Cauchy variables: A sets the overall density, $B _ { i }$ and $C _ { j }$ the out- and in-degree of each node. Features and the target are assigned to random nodes. A graph is resampled until the feature nodes and the target node share an ancestor.

TabICLv1. A layered MLP graph as in TabPFNv1: num\_layers layers of width hidden\_dim, both drawn from log-scaled truncated normals with per-dataset means in [1, 6] and [5, 130], with block-wise or Bernoulli dropout of edges. In causal mode (sampled per dataset) features and target are read from hidden activations; otherwise the root causes are the features and the last layer is the target.

TabPFNv1. The same layered MLP graph, with layer count and width from gamma metadistributions (means up to 6 and 300), num\_causes root variables, edge dropout, and the same causal or non-causal read-out. With probability $1 / ( 1 + e ^ { w } ) , w \sim U ( 2 , 1 0 )$ , the dataset is instead a draw from an RBF Gaussian process on Gaussian inputs [1].

## A.3.2 Edge functions

TabICLv2. Eight function types: MLP, tree ensemble (symmetric trees, LogInt(1, 128) trees of depth 1 to 7), discretization to random centers, Gaussian process with a random kernel, linear, quadratic, EM-style cluster assignment, and product. The first three follow TabPFNv2, and the other five are new. With several parents, the prior either concatenates them and applies one function, or applies a function per parent and aggregates by sum, product, max, or log-sum-exp. Each node output is standardized, rescaled by a random feature-importance vector, and multiplied by a random node importance.

TabICLv1. With probability 0.7 an MLP layer: a random afine map followed by an activation drawn from 19 fixed functions (Identity, Tanh, LeakyReLU, ELU, ReLU, ReLU6, SELU, SiLU, Softplus, Hardtanh, sign, sine, RBF, exponential, ${ \sqrt { | x | } } , 1 _ { | x | \leq 1 } , x ^ { 2 } , | x | )$ or a random Fourierfeature function. Activations may difer between layers. With probability 0.3 a tree layer:

an XGBoost multi-output regressor with at most 4 trees of depth at most 4, fitted on standard-normal targets and evaluated on the parent values.

TabPFNv1. A random afine map followed by one activation per dataset, shared by all layers, drawn from Tanh, Identity, or ReLU.

## A.3.3 Transformations and filters

TabICLv2. Each scalar feature is min-max scaled and then warped by a Kumaraswamy function, with exponents drawn from LogNum(0.2, 5). Categorical features come from discretizing sub-vectors of node values into 2 to 9 categories. Outliers are clamped at 4 standard deviations, and standardized values are clipped to [−100, 100]. Constant columns are removed, and the feature order and class labels are permuted. Two filters reject datasets. The ancestor filter rejects a graph whose target shares no ancestor with the features. The ExtraTrees test fits 25 trees of depth 6 with bootstrapping and keeps a dataset only if their out-of-bag error beats the mean predictor in at least 95% of 200 bootstrap resamples.

TabICLv1. Gaussian noise is added after every layer, with a standard deviation drawn per dataset from LogNum(10−<sup>4</sup>, 0.3). Root causes come from a normal, a uniform, or a mixed distribution. With probability 0.2, a dataset has categorical features. Each column then becomes categorical independently, with a uniformly drawn probability, and its values are binned by rank into max(2, round(Γ(1, 10))) categories. Outliers are clamped at 4 standard deviations, and the features and the regression target are standardized. Class labels are permuted. The original prior applies no filter. We apply the ExtraTrees test of TabICLv2, so that all three corpora are filtered alike.

TabPFNv1. Gaussian noise is added at every node, with a mean and a standard deviation drawn per node. The standard deviation comes from LogNum(10−<sup>4</sup>, 0.3). Categorical features are generated as in TabICLv1: with probability 0.2, binned by rank into Γ(1, 10) categories. Outliers are removed, and the features and the target are z-normalized. Features are rescaled by the fraction of columns in use. For classification, class boundaries are sampled from the data, and class labels are permuted. The original prior applies no filter. We apply the ExtraTrees test of TabICLv2.

## A.3.4 Sampling settings

Table 9 lists the sampling settings. TabICLv2 is the main prior of Section 3.2. TabICLv1 and TabPFNv1 share one shape specification: dataset i has the same task type, row count, feature count, and class count in both sets, so a diference between them is a diference of generator alone.

Table 9 | Sampling settings of the three synthetic priors. LogU: log-uniform. Splits are train/validation/test and stratified for classification.
<table><tr><td>Setting</td><td>TabICLv2</td><td>TabICLv1</td><td>TabPFNv1</td></tr><tr><td>Generator</td><td>graph_scm</td><td>mix_scm</td><td>prior_bag</td></tr><tr><td>Datasets</td><td>4,000</td><td>2,000</td><td>2,000</td></tr><tr><td>Task mix (reg/bin/multi)</td><td> $2 , 0 0 0 / 1 , 4 0 0 / 6 0 0$ </td><td> $0 . 4 / 0 . 3 / 0 . 3$ </td><td> $0 . 4 / 0 . 3 / 0 . 3$ </td></tr><tr><td>Multiclass classes</td><td> $U \{ 3 , \ldots , 1 0 \}$ </td><td> $U \{ 3 , \ldots , 1 0 \}$ </td><td> $U \{ 3 , \ldots , 1 0 \}$ </td></tr><tr><td>Rows</td><td> $\mathrm { L o g U } [ 4 0 0 , 6 0 , 0 0 0 ]$ </td><td>LogU[1,024, 32,768]</td><td>LogU[1,024, 32,768]</td></tr><tr><td>Features</td><td>U[1, cap(n)], cap 100 → 20</td><td>LogU[4, 64]</td><td>LogU[4, 64]</td></tr><tr><td>Graph nodes</td><td>LogInt(2, 32)</td><td>MLP, mean layers [1, 6]</td><td>MLP, gamma layers</td></tr><tr><td>Categorical share</td><td>clip(U(−0.5, 1.2), 0, 1)</td><td> $p = 0 . 2 \ \mathrm { p e r } \ \mathrm { d a t a s e t }$ </td><td> $p = 0 . 2$  per dataset</td></tr><tr><td>Node noise</td><td>none</td><td> $\mathrm { L o g N u m } ( 1 0 ^ { - 4 } , 0 . 3 )$ </td><td> $\mathrm { L o g N u m } ( 1 0 ^ { - 4 } , 0 . 3 )$ </td></tr><tr><td>Graph filter</td><td>ancestor test</td><td>none</td><td>none</td></tr><tr><td>Dataset filter</td><td>ExtraTrees, p &lt; 0.05</td><td>ExtraTrees,  $p < 0 . 0 5$ </td><td>ExtraTrees,  $p < 0 . 0 5$ </td></tr><tr><td>Split</td><td>64 /16/ 20</td><td> $6 4 / 1 6 / 2 0$ </td><td> $6 4 / 1 6 / 2 0$ </td></tr><tr><td>Attempts per dataset</td><td>≤ 100</td><td>≤ 64</td><td>≤ 64</td></tr><tr><td>Base seed</td><td>20260804</td><td>0</td><td>0</td></tr></table>

## A.4 Additional results

## A.4.1 Main results

Full leaderboard. Figure 10 shows the Elo of all 58 methods on the full BeyondArena.

![](images/f5e57b3619812b0eed9064ed67c4e2015c0b48503976c1b15b439f403eeec027.jpg)  
Figure 10 | Elo of all 58 methods on the full BeyondArena under default, tuned, and tuned-andensembled deployment. Our families are coloured as in Figure 1.

Improvement over the MLP. Figure 11 repeats Figure 3 with the improvement over the tuned MLP, $1 0 0 ( \ell _ { \mathrm { b a s e } } - \ell ) / \ell _ { \mathrm { b a s e } } .$ . We average across the splits of each datasets and then take the median across datasets. Unlike Elo, the improvement keeps the size of each gain [39] and does not depend on the set of compared methods. However, it depends on the choice of the baseline and averaging conceals skewness. Elo has the opposite properties, so neither metric suits every application [39]. Under T, LightGBM-TabPrep with its FT subset has the highest median (+0.71). Under T+E, Kumo-Tabular (+6.04) and RealMLP (+1.69) lead LightGBM-TabPrep (+1.34). LightGBM-TabPrep leads Kumo-Tabular on grouped splits and on large datasets.

![](images/e456b2d7627276d850578c5ac9eee935571fc6b50a7683537ce0b7a32992893b.jpg)  
Figure 11 | Median per-dataset improvement over the tuned MLP with a random subset on the full BeyondArena, by dataset group. Each dataset’s improvement is averaged over its splits. Top: tuned (T), our models with their FT subsets. Bottom: tuned and ensembled (T+E), our models with their FTE subsets. Under each panel, the bars show the gain of each meta-tuned subset over the same model with its random subset, as the diference of the two medians.

## A.4.2 Selection rule

Table 10 gives the win rate of every selection rule of Figure 4.

Table 10 | Win rate against random-K of the selection rules of Section 5.1, synthetic out-of-fold (1,000 held-out tasks), T deployment, against random-K baselines that no selection saw; mean over 200 repetitions (±1.96 standard errors of at most 0.0015). Draws R: our rule with R validation resamples paired with R references. Resampling kept: 16 validation resamples with one reference (Val.), 16 references on the original split (Ref.), or one of each (Neither). Bold: the best rule in the row.
<table><tr><td></td><td></td><td colspan="5">Draws R</td><td colspan="3">Resampling kept</td><td colspan="2">TabRepo</td></tr><tr><td>Model</td><td>K</td><td>16</td><td>8</td><td>4</td><td>2</td><td>1</td><td>Val.</td><td>Ref.</td><td>Neither</td><td>Norm.</td><td>Count</td></tr><tr><td>LightGBM</td><td>25</td><td>0.636</td><td>0.634</td><td>0.632</td><td>0.627</td><td>0.620</td><td>0.630</td><td>0.630</td><td>0.621</td><td>0.617</td><td>0.621</td></tr><tr><td></td><td>64</td><td>0.621</td><td>0.618</td><td>0.613</td><td>0.607</td><td>0.599</td><td>0.614</td><td>0.609</td><td>0.600</td><td>0.588</td><td>0.602</td></tr><tr><td></td><td>200</td><td>0.599</td><td>0.596</td><td>0.592</td><td>0.585</td><td>0.576</td><td>0.593</td><td>0.586</td><td>0.574</td><td>0.561</td><td>0.575</td></tr><tr><td>LightGBM-TabPrep</td><td>25</td><td>0.636</td><td>0.636</td><td>0.633</td><td>0.629</td><td>0.622</td><td>0.632</td><td>0.631</td><td>0.623</td><td>0.618</td><td>0.622</td></tr><tr><td></td><td>64</td><td>0.621</td><td>0.619</td><td>0.615</td><td>0.610</td><td>0.602</td><td>0.616</td><td>0.609</td><td>0.601</td><td>0.597</td><td>0.607</td></tr><tr><td></td><td>200</td><td>0.603</td><td>0.599</td><td>0.594</td><td>0.588</td><td>0.579</td><td>0.597</td><td>0.588</td><td>0.577</td><td>0.571</td><td>0.588</td></tr><tr><td>MLP</td><td>25</td><td>0.697</td><td>0.696</td><td>0.692</td><td>0.687</td><td>0.679</td><td>0.691</td><td>0.689</td><td>0.681</td><td>0.687</td><td>0.693</td></tr><tr><td></td><td>64</td><td>0.669</td><td>0.665</td><td>0.661</td><td>0.654</td><td>0.646</td><td>0.664</td><td>0.655</td><td>0.645</td><td>0.658</td><td>0.672</td></tr><tr><td></td><td>200</td><td>0.631</td><td>0.626</td><td>0.619</td><td>0.612</td><td>0.601</td><td>0.626</td><td>0.613</td><td>0.600</td><td>0.616</td><td>0.633</td></tr></table>

![](images/c0caeb1d0a95c25aded99cfd9055ad283004357b55fb9b247528ce69d5469d78.jpg)  
Figure 12 | Synthetic versus TabRepo meta-training data at equal budget, as in Figure 5, for K = 25, 64, and 200.

## A.4.3 Synthetic versus real meta-training data

Figure 12 repeats Figure 5 for K ∈ {25, 64, 200}. At K = 25, TabRepo is ahead at equal budget up to $2 ^ { 2 0 }$ to $2 ^ { 2 2 }$ for all three models. The synthetic tasks end above its best for all three (Figure 12).

TabRepo overlap. Of the 200 TabRepo datasets we exclude 18. Ten are content-identical to or listed as overlapping with a TabArena dataset: Amazon\_employee\_access, APSFailure, Bioresponse, blood-transfusion-service-center, diabetes, Diabetes130US, diamonds, hiva\_agnostic, jm1, splice. Eight share the source data of a TabArena dataset under a diferent curation: bank-marketing, churn, credit-g, houses, qsar-biodeg, wine-quality-red, wine-quality-white, wine\_quality.

![](images/d4289d99941a38735ac5051d9343effa021adfb87d704aaaec3673b10fa1a53a.jpg)  
Figure 13 | Random subsets of growing size $K ^ { \prime }$ against the random-K baseline, as in Figure 6, for K = 25, 64, and 200.

## A.4.4 Meta-tuning versus more random search

Figure 13 repeats Figure 6 for $K \in \{ 2 5 , 6 4 , 2 0 0 \}$ }.

## A.4.5 Choice of prior

Figures 14 and 15 repeat the top and bottom rows of Figure 7 for $K \in \{ 2 5 , 6 4 , 2 0 0 \}$

![](images/bc079f7154ca35805fdbbc4bfd335d12744530263b97589adf80734b5d914698.jpg)  
Figure 14 | Three priors with 2,000 configurations on 2,000 synthetic tasks, as in the top row of Figure 7: win rate against the number of configurations C at $N = 2 { , } 0 0 0 ,$ for $K = 2 5 , 6 4 ,$ and 200.

![](images/ebbd4cd5410d7837d78c5843a1232b22ac562c9b3e2e3513b0a6d9c3e2ff31e3.jpg)  
Figure 15 | As Figure 14, against the number of synthetic tasks N at C = 2,000 (bottom row of Figure 7).

## A.4.6 Scaling

Figure 16 shows the win rate over the (C, N) grid, as in Figure 8, for $K \in \{ 2 5 , 6 4 , 2 0 0 \}$ Figures 17 and 18 show the curves along N and along C.

![](images/cce31997a0e7274e555a5a404268467f147da8de58a898417e966b51447188b5.jpg)  
Figure 16 | Win rate over the (C, N) grid, as in Figure 8, for K = 25, 64, and 200.

![](images/ef42e562af7c82c5e2eed3bac8d85d5ef7d32eff300bf1b16860d4e4a8329d81.jpg)  
Figure 17 | Win rate against the number of synthetic datasets N, one curve per pool size C, with the fit $a - b / \sqrt { N }$ at $C = 4 { , } 0 0 0 ,$ for K = 25, 64, and 200.

N = 4000 Smaller N (lighter = fewer) Fit a + b log<sub>2</sub> C  
![](images/ecee241b9cbc417faa0892f0b8cd6692c525985f67ea57799212a340ceb272b3.jpg)

![](images/91140630b0ced8a81398c538f16961388c933ed5542566d3ef26de74ce29ef38.jpg)

![](images/7c4ec3f94fb5e04963a5e1d0bbc696b208f5c5d6a8e077fff33d2c764bab9bb5.jpg)  
Figure 18 | Win rate against the number of configurations C, one curve per number of synthetic datasets N, with the fit $a + b \log _ { 2 } C$ at N = 4,000, for K = 25, 64, and 200.

Sampling distribution 200 selected configurations (LightGBM)

## A.4.7 Selected hyperparameters

Figures 19–21 show all hyperparameters of each model, as in Figure 9.

![](images/a14cb44c17113a1ee91f3d7073c9e4d0a73424e2ee12fe368ea715850c4fdef0.jpg)  
Figure 19 | All hyperparameters of LightGBM: the 200 selected configurations against the sampling distribution, as in Figure 9.

![](images/288da5200ca855cc415e8706ca6d321201226f07c98ba6f328be483685b87ef4.jpg)  
Figure 20 | All hyperparameters of LightGBM-TabPrep, as in Figure 9. The TabPrep settings count only the 37 selected configurations with TabPrep enabled.

![](images/fc87d4180eaca300d26c79f35ea85365f12a76d889a4a019b08b09668a5602f0.jpg)

![](images/35b733154e7d136366c9530fbee849270df74280116ab4299a86eb5e5347cde6.jpg)

![](images/82591082f961200250fb87dbb8d16682a5630f26d3480a93734a492f2b4da596.jpg)

![](images/d78d0d822e4dea8e9d1079e00b7baaaa454aa7cff8854b64e9bb8da83fd864d7.jpg)

![](images/e3769c8a4699a7045a4e581e54f86aacc97ddf698e8be163da7b5abe70d3274f.jpg)

![](images/714ec20c78eb6d8eb33fbe2d4e50cd6d6bfe4a3d94fc4261dce08ea92da5f756.jpg)

![](images/94a5df893bb3a2d56a43c0cb1e03a4c545cdc1052feb13a77888c69ca3aeb555.jpg)  
Figure 21 | All hyperparameters of the MLP, as in Figure 9. The dropout rate is 0 with probability 0.5 and uniform otherwise; the label gives the share at 0 (sampled / selected), and the curve is the continuous part.

## B. Theory

## B.1 Setup and main propositions

## B.1.1 Setup

Fix a model $f ,$ a learning algorithm ${ \mathcal { A } } ,$ a candidate pool C of C hyperparameter configurations, and a deployment budget $1 \leq K \leq C$ . A subset is $\mathcal { S } \in \Pi _ { K } : = \{ \mathcal { S } \subseteq \mathcal { C } : | \mathcal { S } | = K \}$ The goal is to choose one subset that performs well on the target distribution of real tasks, using synthetic tasks for meta-training. Performance is win rate against a uniformly random subset of the same size. All statements condition on the candidate pool.

Tasks and deployment. Let $\mathcal { P }$ and $\mathcal { Q }$ be the target (e.g. TabArena or BeyondArena benchmarks [11, 12]) and source $( \mathrm { e } . g$ . sampled from the TabICLv2 prior [3]) distributions over tasks. Sample a target task $\tau \sim \mathcal { P }$ , which specifies a row distribution $P _ { \tau }$ . Sample the rows of $D$ independently from $P _ { \tau }$ and split the dataset into training, validation, and test sets of prescribed sizes. For each configuration, fit

$$
\hat { f } _ { c } = \mathcal { A } ( f , c , \xi , D ) ,
$$

where $\xi$ collects the randomness in the dataset split and in the training of every configuration (initialization, subsampling, and any other stochastic step of $\mathcal { A } )$ . Each configuration is fitted once per task, and every subset that contains $\mathbf { i t } ,$ including the random reference it is compared against, reuses that fit. A deployment rule $\mathcal { E }$ takes as an input a subset (its fitted members, train and validation subsets) and outputs a single predictor. We then score it on the test set:

$$
\begin{array} { r } { L ( S ; D , \hat { f } , \xi ) = \ell \Big ( \mathcal { E } ( S ; D , \hat { f } , \xi ) ; D _ { \mathrm { t e s t } } \Big ) . } \end{array}\tag{1}
$$

We require that $\mathcal { E }$ be deterministic given its inputs and that the same $\mathcal { E }$ be applied to a subset and to any subset it is compared against.

Two deployment rules. We use the two regimes of Erickson et al. [11], Purucker et al. [12]. Tuned (T) keeps the validation-best member,

$$
\begin{array} { r } { \mathcal { E } _ { \mathrm { T } } ( S ) = \hat { f } _ { c ^ { * } } , \qquad c ^ { * } = \underset { c \in S } { \arg \operatorname* { m i n } } \ell ( \hat { f } _ { c } ; D _ { \mathrm { v a l } } ) . } \end{array}
$$

Tuned and ensembled (T+E) runs B steps of greedy ensemble selection with replacement on the validation split [36]. Writing $w _ { c }$ for the share of the B slots taken by configuration $c ,$

$$
{ \mathcal { E } } _ { \mathrm { T + E } } ( S ) = \sum _ { c \in { \mathcal { S } } } w _ { c } { \hat { f } } _ { c } , \qquad w _ { c } \geq 0 , \quad \sum _ { c \in { \mathcal { S } } } w _ { c } = 1 .
$$

The tuned regime is the special case of tuned and ensembled that puts all weight on one member.

Population objective. Let $\begin{array} { r } { h ( a , b ) = \mathbf { 1 } _ { a < b } + \frac { 1 } { 2 } \mathbf { 1 } _ { a = b } } \end{array}$ . The per-task utility of a subset is

$$
\begin{array} { r } { u ( S , \tau ) = \mathbb { E } _ { D , \hat { f } , \xi | \tau , \mathcal { R } \sim \mathrm { U n i f } ( \Pi _ { K } ) } \left[ h \Big ( L ( S ; D , \hat { f } , \xi ) , L ( \mathcal { R } ; D , \hat { f } , \xi ) \Big ) \right] , } \end{array}\tag{2}
$$

and the population objective is

$$
U _ { \mathcal { P } } ( \mathcal { S } ) = \mathbb { E } _ { \tau \sim \mathcal { P } } [ u ( \mathcal { S } , \tau ) ] , \qquad \mathcal { S } _ { \mathcal { P } } ^ { \ast } \in \underset { \mathcal { S } \in \Pi _ { K } } { \arg \operatorname* { m a x } } U _ { \mathcal { P } } ( \mathcal { S } ) .\tag{3}
$$

A uniformly random subset scores $\mathbb { E } _ { S \sim \mathrm { U n i f } ( \Pi _ { K } ) } [ U _ { \mathcal { P } } ( S ) ] = 1 / 2$ by exchangeability (proof in §B.2), so $U _ { \mathcal { P } } ( S ) - 1 / 2$ is the gain over random search at the same budget.

Evaluations and statistics. Given a task $\tau _ { : }$ , let $\omega = ( D , \hat { f } , \xi )$ be the evaluation, with law $\mu _ { \tau }$ on a space $\Omega ,$ and write $L ( S ; \omega )$ for the deployed loss equation 1. The deployment rule is deterministic given its inputs, so once $\omega$ is fixed the deployed loss of every subset is fixed: $L ( S ; \cdot )$ is a map on $\Omega _ { z }$ , which we assume measurable, as the expectation in equation 2 already requires. Write $\bar { \mathcal { P } } = \mathbb { E } _ { \tau \sim \mathcal { P } } [ \mu _ { \tau } ]$ and $\bar { \mathcal { Q } } = \mathbb { E } _ { \tau \sim \mathcal { Q } } [ \mu _ { \tau } ]$ for the marginal laws of ω. A statistic is a measurable map $\Phi : \Omega  \mathcal { X }$ , and $\mathcal { P } _ { \Phi } = \bar { \mathcal { P } } \circ \Phi ^ { - 1 } , \mathcal { Q } _ { \Phi } = \bar { \mathcal { Q } } \circ \Phi ^ { - 1 }$ are the laws of $\Phi ( \omega )$ . In this notation, equation 2–equation 3 read

$$
\begin{array} { r } { U _ { P } ( \mathcal { S } ) = \mathbb { E } _ { \tau \sim \mathcal { P } } \mathbb { E } _ { \omega \sim \mu _ { \tau } } \mathbb { E } _ { \mathcal { R } } [ h ( L ( \mathcal { S } ; \omega ) , L ( \mathcal { R } ; \omega ) ) ] , \qquad \mathcal { R } \sim \mathrm { U n i f } ( \Pi _ { K } ) \mathrm { i n d e p e n d e n t o f } ( \tau , \omega ) . } \end{array}
$$

## B.1.2 Upper bound on the improvement

Suppose we have access to the target distribution $\mathcal { P }$ itself, so $U _ { \mathcal P }$ is known on all of $\Pi _ { K }$ and selection carries no estimation error. We ask how many of the subsets in $\Pi _ { K }$ improve upon the best subset seen so far. Suppose the $M = { \binom { C } { K } }$ subsets are revealed in uniformly random order. A subset improves the oracle’s choice precisely when its utility sets a new record.

Proposition 1 (Oracle record count) Ifthe oracle utilities are distinct, the ith revealed subset sets a record with probability $1 / i$ over the uniformly random reveal order. Counting the initial subset as a record,

$$
\mathbb { E } [ n u m b e r ~ o f r e c o r d s ~ a m o n g ~ t h e ~ M ~ s u b s e t s ] = \sum _ { i = 1 } ^ { M } \frac { 1 } { i } = \log M + O ( 1 ) .
$$

The count is a sum of independent Bernoulli variables with mean log $M + O ( 1 )$ [40]: improvements arrive at rate $1 / i$ and thin out as the search proceeds. Since log $M \ =$ $K \log ( C / K ) + O ( K )$ , at fixed $K \ll C$ doubling the pool adds about K log 2 of them. Metatuning thus is the problem of recovering enough of the log M improvements, having access only to the N source tasks.

## B.1.3 Transfer from synthetic to real tasks

The target distribution is not available for meta-tuning: we observe only a benchmark of real tasks, which we reserve for evaluation. We therefore select the subset on a source distribution Q of synthetic tasks and transfer it. We sample N independent source tasks $\tau _ { i } \sim \mathcal { Q }$ and one evaluation $( D _ { i } , \hat { f } _ { i } , \xi _ { i } )$ for each. Write $L _ { i } ( \boldsymbol { S } ) = L ( \boldsymbol { S } ; D _ { i } , \hat { f } _ { i } , \xi _ { i } )$ . The empirical counterpart of equation 3 is

$$
\widehat { U } _ { N } ( S ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { E } _ { \mathcal { R } \sim \mathrm { U n i f } ( \Pi _ { K } ) } \left[ h ( L _ { i } ( S ) , L _ { i } ( \mathcal { R } ) ) \right] ,\tag{4}
$$

and $\mathbb { E } [ \widehat { U } _ { N } ( S ) ] = U _ { \mathcal { Q } } ( S )$ for every fixed subset. Its expectation is $U _ { \mathcal { Q } }$ , not $U _ { \mathcal { P } } ;$ we quantify the price of this approximation.

Assumption 1 (Comparison fidelity) Source and target subset comparisons satisfy

$$
\operatorname* { s u p } _ { \mathcal { S } , \mathcal { T } \in \Pi _ { K } } | [ U _ { \mathcal { P } } ( \mathcal { S } ) - U _ { \mathcal { P } } ( \mathcal { T } ) ] - [ U _ { \mathcal { Q } } ( \mathcal { S } ) - U _ { \mathcal { Q } } ( \mathcal { T } ) ] | \leq \varepsilon .\tag{5}
$$

Proposition 2 (Transfer of subset selection) Let $\widehat { s }$ be the subset returned by the selector and

$$
\eta _ { N } = \operatorname* { m a x } _ { \mathcal { S } \in \Pi _ { K } } \widehat { U } _ { N } ( \mathcal { S } ) - \widehat { U } _ { N } ( \widehat { \mathcal { S } } )
$$

its suboptimality on the source tasks. Under Assumption 1, with probability at least $1 - \delta ,$

$$
U _ { \mathcal { P } } ( S _ { \mathcal { P } } ^ { * } ) - U _ { \mathcal { P } } ( \widehat { S } ) \leq \varepsilon + 2 \sqrt { \frac { \log ( 2 { \binom { C } { K } } / \delta ) } { 2 N } } + \eta _ { N } .\tag{6}
$$

The bound separates selection error, source-target mismatch, and finite-task uncertainty. The number of source tasks reduces the second term, of order $\sqrt { \log M / N }$ , but cannot remove systematic reversals of subset preferences (ε) or the error of the selection algorithm $( \eta _ { N } )$ . We next investigate when Assumption 1 holds.

Lemma 1 (Comparison discrepancy through a statistic) The comparison factors through Φ if for every $( \mathcal { S } , \mathcal { R } ) \in \Pi _ { K } \times \Pi _ { K }$ there is a measurable $g _ { S , \mathcal { R } } : \mathcal { X }  [ 0 , 1 ]$ , depending on $( S , R )$ only, such that

$$
h ( L ( S ; \omega ) , L ( \mathcal { R } ; \omega ) ) = g _ { S , \mathcal { R } } ( \Phi ( \omega ) ) \qquad f o r \ e v e r y \ \omega \in \Omega .\tag{7}
$$

If the comparison factors through Φ, then Assumption 1 holds with $\varepsilon = 2 \mathrm { T V } ( \mathcal { P } _ { \Phi } , \mathcal { Q } _ { \Phi } )$ , where $\mathrm { T V } ( P , Q ) = \operatorname* { s u p } _ { A } | P ( A ) - Q ( A ) |$ |. If moreover $\Phi = \kappa \circ \Psi f o r$ a statistic $\Psi : \Omega \to \mathcal { V }$ and a measurable $\kappa : \mathcal { y }  \mathcal { X } _ { : }$ , then the comparison also factors through Ψ, and $\mathrm { T V } ( \mathcal { P } _ { \Phi } , \mathcal { Q } _ { \Phi } ) \le$ $\mathrm { T V } ( \mathcal { P } _ { \Psi } , \mathcal { Q } _ { \Psi } )$

Proposition 3 (Rank suficiency) Let $\Phi ( \omega )$ be the ordering of the M losses $\{ L ( \boldsymbol { S } ; \omega ) \} _ { \boldsymbol { S } \in \Pi _ { K } } ,$ ties recorded. Then, for any deployment rule, Assumption 1 holds with $\varepsilon = 2 \mathrm { T V } ( \mathcal { P } _ { \Phi } , \mathcal { Q } _ { \Phi } )$ Under $\mathcal { E } _ { \mathrm { T } }$ the same holds for the finer statistic $\Psi ( \omega )$ , the validation and test orderings of the pool C.

Hence what must transfer is the ranking of subsets by deployed loss. Under T this reduces to the validation and test ordering of individual configurations. Whether the synthetic prior preserves these orderings is an empirical question.

## B.1.4 Analysis of the forward-selection heuristic

The inner expectation in equation 4 runs over $\textstyle { \binom { C } { K } }$ references, so in practice we draw R independent references $\mathcal { R } _ { r } \sim \operatorname { U n i f } ( \Pi _ { K } )$ and optimize

$$
\widetilde { U } _ { N , R } ( \boldsymbol { S } ) = \frac { 1 } { N R } \sum _ { i = 1 } ^ { N } \sum _ { r = 1 } ^ { R } h ( L _ { i } ( \boldsymbol { S } ) , L _ { i } ( \mathcal { R } _ { r } ) ) .\tag{8}
$$

Proposition 4 (Reference averaging) Write $\mathbb { E } _ { \mathcal { R } }$ and $\mathrm { V a r } _ { \mathcal { R } } f o r$ the average over the R reference draws with the source tasks held fixed. For every fixed subset,

$$
\mathbb { E } _ { \mathcal { R } } \Big [ \widetilde { U } _ { N , R } ( \boldsymbol { S } ) \Big ] = \widehat { U } _ { N } ( \boldsymbol { S } ) , \qquad \mathrm { V a r } _ { \mathcal { R } } \Big ( \widetilde { U } _ { N , R } ( \boldsymbol { S } ) \Big ) \le \frac { 1 } { 4 R } .\tag{9}
$$

References thus add no bias, and their sampling noise vanishes as $1 / R$

Forward selection. Exhaustive search over $\textstyle { \binom { C } { K } }$ subsets is impractical. Starting from $S _ { 0 } = \emptyset$ we run

$$
\begin{array} { r l } & { c _ { k } \in \underset { c \in \mathcal { C } \setminus \mathcal { S } _ { k - 1 } } { \arg \operatorname* { m a x } } \widetilde { U } _ { N , R } ( S _ { k - 1 } \cup \{ c \} ) , } \\ & { } \\ & { S _ { k } = S _ { k - 1 } \cup \{ c _ { k } \} , \qquad k = 1 , \ldots , K . } \end{array}
$$

We return exactly K configurations, even if an addition lowers the objective. The objective is not monotone in general, so forward selection is a heuristic with no generic greedyoptimality guarantee. In Proposition $^ { 2 , }$ both the gap between $\widetilde { U } _ { N , R }$ and ${ \widehat { U } } _ { N }$ and the greedy suboptimality enter through $\eta _ { N }$

## B.2 Proofs

Proof of the random-K baseline. Let S, R be i.i.d. uniform on $\Pi _ { K }$ , independent of $( \tau , \omega )$ and write $a = L ( \mathcal { S } ; \omega ) , b = L ( \mathcal { R } ; \omega )$ , so that $\mathbb { E } _ { S } [ U _ { \mathcal { P } } ( S ) ] = \mathbb { E } _ { \tau , \omega } \mathbb { E } _ { S , \mathcal { R } } [ h ( a , b ) ]$ . For fixed $( \tau , \omega )$ , the pairs $( \mathcal { S } , \mathcal { R } )$ and $( \mathcal { R } , S )$ have the same law, so $\mathbb { E } _ { S , \mathcal { R } } [ h ( a , b ) ] = \mathbb { E } _ { S , \mathcal { R } } [ h ( b , a ) ]$ , and adding the two with $h ( a , b ) + h ( b , a ) = 1$ gives $\begin{array} { r } { \mathbb { E } _ { S , \mathcal { R } } [ h ( a , b ) ] = \frac { 1 } { 2 } } \end{array}$ for every $( \tau , \omega )$ , hence $\begin{array} { r } { \mathbb { E } _ { S } [ U _ { \mathcal { P } } ( S ) ] = \frac { 1 } { 2 } } \end{array}$ □

Proof of Proposition 1. Write $M = { \binom { C } { K } }$ and let $U _ { i } = U _ { \mathcal { P } } ( S _ { i } )$ be the utility of the ith revealed subset. Let $\rho _ { i }$ be the rank of $U _ { i } ,$ from largest to smallest, among $U _ { 1 } , \ldots , U _ { i } ,$ so that $I _ { i } = { \bf 1 } _ { \rho _ { i } = 1 }$ indicates a record. We first show that $\rho _ { 1 } , \ldots , \rho _ { M }$ are independent with $\rho _ { i }$ uniform on $\{ 1 , \ldots , i \}$ ; this is the classical record theorem of Rényi [40].

Let π be the permutation that sorts $( U _ { 1 } , \dots , U _ { M } )$ . Because the reveal order is uniform and the utilities are distinct, every ordering is equally likely, so $\pi$ is uniform over the M! permutations. The rank vector $\left( \rho _ { 1 } , \ldots , \rho _ { M } \right)$ is a function of $\pi ,$ and this function is a bijection onto $\{ 1 \} \times \{ 1 , 2 \} \times \cdots \times \{ 1 , \ldots , M \}$ : it is injective because π is recovered by inserting $U _ { i }$ at position $\rho _ { i }$ among $U _ { 1 } , \dots , U _ { i - 1 } { \mathrm { f o r } } i = 1 , \dots , M$ , and both sets have $M !$ elements. Hence $( \rho _ { 1 } , \ldots , \rho _ { M } )$ is uniform on that product set, and for each marginal we can calculate

$$
\operatorname* { P r } ( \rho _ { i } = k _ { i } ) = \sum _ { k _ { 1 } = 1 } ^ { 1 } \sum _ { k _ { 2 } = 1 } ^ { 2 } \cdot \cdot \cdot \sum _ { k _ { i - 1 } = 1 } ^ { i - 1 } \sum _ { k _ { i + 1 } = 1 } ^ { i + 1 } \cdot \cdot \sum _ { k _ { M } = 1 } ^ { M } \frac { 1 } { M ! } = \frac { M ! } { i } \cdot \frac { 1 } { M ! } = \frac { 1 } { i } ,
$$

$$
1 \cdot 2 \cdot \cdot \cdot ( i - 1 ) \cdot ( i + 1 ) \cdot \cdot \cdot M = M ! / i
$$

$$
\operatorname* { P r } ( \rho _ { 1 } = k _ { 1 } , \dots , \rho _ { M } = k _ { M } ) = 1 / M ! = \operatorname* { P r } ( \rho _ { 1 } = k _ { 1 } ) \cdot \cdot \cdot \operatorname* { P r } ( \rho _ { M } = k _ { M } )
$$

Therefore we also have that $\mathrm { P r } ( I _ { i } = 1 ) = \mathrm { P r } ( \rho _ { i } = 1 ) = 1 / i .$ , and $I _ { 1 } , \ldots , I _ { M }$ are independent because each is a function of a diferent $\rho _ { i }$ . By linearity, $\begin{array} { r } { \mathbb { E } \sum _ { i } I _ { i } = \sum _ { i } \operatorname* { P r } ( I _ { i } = 1 ) = H _ { M } . } \end{array}$ and comparing the sum with $\ b { \int x ^ { - 1 } } d \ b { x }$ gives $\log ( M + 1 ) \ \leq \ H _ { M } \leq 1 + \log M$ . Finally $( C / K ) ^ { K } \overset { \cdot } { \leq } M \overset { \cdot } { \leq } ( C / K ) ^ { K } e ^ { K }$ gives log $M = K \log ( C / K ) + O ( K )$ □

Proof of Proposition 2. Fix $ { \mathcal { S } } \in \Pi _ { K }$ . By equation 4, $\hat { U } _ { N } ( \mathcal { S } )$ is the mean of N i.i.d. terms in [0, 1] with mean $U _ { \mathcal { Q } } ( S )$ . Hoefding’s inequality gives $\mathrm { P r } ( | \hat { U } _ { N } ( { \cal S } ) - U _ { \mathcal { Q } } ( { \cal S } ) | >$ $t ) \le 2 \exp ( - 2 N t ^ { 2 } )$ , and a union bound over the $\textstyle { \binom { C } { K } }$ subsets gives

$$
\operatorname* { P r } \Big ( \operatorname* { s u p } _ { S \in \Pi _ { K } } | \hat { U } _ { N } ( S ) - U _ { \mathcal { Q } } ( S ) | > t \Big ) \leq 2 \binom { C } { K } \exp ( - 2 N t ^ { 2 } ) .\tag{10}
$$

Setting the right-hand side to δ and solving for t yields $t _ { N } = \sqrt { \log ( 2 \binom { C } { K } / \delta ) } / ( 2 N )$ . On the complementary event, which has probability at least $1 - \delta _ { \cdot }$ , let ${ S ^ { * } }$ maximize $U _ { \mathcal P }$ . Then

$$
\begin{array} { r l r l } & { U _ { \mathcal { P } } ( \mathcal { S } ^ { * } ) - U _ { \mathcal { P } } ( \hat { S } ) \leq U _ { \mathcal { Q } } ( \mathcal { S } ^ { * } ) - U _ { \mathcal { Q } } ( \hat { S } ) + \varepsilon } & & { \mathrm { b y ~ e q u a t i o n ~ 5 } } \\ & { \leq \hat { U } _ { N } ( \mathcal { S } ^ { * } ) - \hat { U } _ { N } ( \hat { S } ) + \varepsilon + 2 t _ { N } } & & { \mathrm { b y ~ u n i f o r m ~ c o n c e n t r a t i o n } } \\ & { \leq \eta _ { N } + \varepsilon + 2 t _ { N } } & & { \mathrm { b y ~ t h e ~ d e f i n i t i o n ~ o f ~ } \eta _ { N } . } \end{array}
$$

The bound holds for any selector. Ours maximizes the Monte Carlo objective $\widetilde { U } _ { N , R }$ rather than ${ \widehat { U } } _ { N }$ , and both that gap and the greedy heuristic’s are part of $\eta _ { N }$ □

Proof of Lemma 1. Since $\mathcal { R } \sim \operatorname { U n i f } ( \Pi _ { K } )$ is independent of $( \tau , \omega )$ and all terms are bounded, the objective equation 3 and the factoring equation 7 give

$$
\begin{array} { r l } & { U _ { \mathcal { P } } ( \mathcal { S } ) = \mathbb { E } _ { \tau \sim \mathcal { P } } \mathbb { E } _ { \omega \sim \mu _ { \tau } } \mathbb { E } _ { \mathcal { R } } [ h ( L ( \mathcal { S } ; \omega ) , L ( \mathcal { R } ; \omega ) ) ] } \\ & { \quad \quad \quad = \mathbb { E } _ { \mathcal { R } } \mathbb { E } _ { \tau \sim \mathcal { P } } \mathbb { E } _ { \omega \sim \mu _ { \tau } } [ g _ { \mathcal { S } , \mathcal { R } } ( \Phi ( \omega ) ) ] = \mathbb { E } _ { \mathcal { R } } \mathbb { E } _ { \mathcal { P } _ { \Phi } } [ g _ { \mathcal { S } , \mathcal { R } } ] , } \end{array}\tag{11}
$$

where the last equality uses that $\bar { \mathcal { P } } = \mathbb { E } _ { \tau \sim \mathcal { P } } [ \mu _ { \tau } ]$ is the marginal law of $\omega$ and $\mathcal { P } _ { \Phi }$ is the law of $\Phi ( \omega )$ under $\mathrm { i t } ;$ likewise $U _ { \mathcal { Q } } ( S ) = \mathbb { E } _ { \mathcal { R } } \mathbb { E } _ { \mathcal { Q } _ { \Phi } } [ g _ { S , \mathcal { R } } ]$ . For any measurable $g : \mathcal { X }  [ 0 , 1 ]$ the layer-cake formula gives $\begin{array} { r } { | \mathbb { E } _ { P } g - \mathbb { E } _ { Q } g | = | \int _ { 0 } ^ { 1 } ( P ( g > t ) - Q ( g > t ) ) d t | \le \mathrm { T V } ( P , Q ) } \end{array}$ Applying it to each ${ { g } } _ { { S } , \mathcal { R } }$ and averaging over R gives $| U _ { \mathcal { P } } ( S ) - U _ { \mathcal { Q } } ( S ) | \le \mathrm { T V } \big ( \mathcal { P } _ { \Phi } , \mathcal { Q } _ { \Phi } \big )$ for every S. Applying this to $s$ and $\tau$ and using the triangle inequality yields equation 5 with $\varepsilon = 2 \mathrm { T V } ( \mathcal { P } _ { \Phi } , \mathcal { Q } _ { \Phi } )$

If $\Phi = \kappa \circ \Psi$ , then equation 7 holds with Ψ and $g _ { \mathcal { S } , \mathcal { R } } \circ \kappa$ in place of Φ and ${ \mathit { g } } _ { { \mathit { S } } , \mathcal { R } }$ . Since $\{ \Phi \in A \} = \{ \Psi \in \kappa ^ { - 1 } ( A ) \} , | \mathcal { P } _ { \Phi } ( A ) - Q _ { \Phi } ( A ) | = | \mathcal { P } _ { \Psi } ( \kappa ^ { - 1 } ( A ) ) - Q _ { \Psi } ( \kappa ^ { - 1 } ( A ) ) | \le \mathrm { T V } ( \mathcal { P } _ { \Psi } , Q _ { \Psi } ) | \le \frac { 1 } { \pi } \{ | Q _ { \Psi } | ^ { 2 } \} .$ for every A. □

Proof of Proposition 3. Let $\Phi ( \omega ) = ( \mathrm { r a n k } _ { \omega } ( \mathcal { S } ) ) _ { \mathcal { S } \in \Pi _ { K } }$ be the vector of ranks of the losses $\{ L ( \boldsymbol { S } ; \omega ) \} _ { \boldsymbol { S } \in \Pi _ { K } }$ , equal losses receiving equal rank; it takes values in a finite set. Since $h ( a , b )$ depends only on whether $a < b , a = b ,$ or $a \ > \ b _ { \mathrm { : } }$ , and dense ranks preserve this, $\begin{array} { r } { h ( L ( S ; \omega ) , L ( \mathcal { R } ; \omega ) ) = \mathbf { 1 } _ { \mathrm { r a n k } _ { \omega } ( S ) < \mathrm { r a n k } _ { \omega } ( \mathcal { R } ) } + \frac { 1 } { 2 } \mathbf { 1 } _ { \mathrm { r a n k } _ { \omega } ( S ) = \mathrm { r a n k } _ { \omega } ( \mathcal { R } ) } } \end{array}$ , which is equation 7. Lemma $1 \mathrm { g i v e s } \varepsilon = 2 \mathrm { T V } ( \mathcal { P } _ { \Phi } , \mathcal { Q } _ { \Phi } )$

Under $\mathcal { E } _ { \mathrm { T } }$ , let $\Psi ( \omega )$ be the pair of dense-rank vectors of the pool C under the validation losses and under the test losses. The deployed member of S is its validation-rank minimizer, ties broken by a fixed rule, and the deployed losses of two subsets compare as the test ranks of their deployed members, so $\Phi = \kappa \circ \Psi$ for a fixed κ. The second part of Lemma 1 applies. □

Proof of Proposition 4. Fix $s ,$ hold the source tasks fixed, and set $\begin{array} { r } { Z _ { r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } h ( L _ { i } ( S ) } \end{array}$ $L _ { i } ( \mathcal { R } _ { r } ) )$ , so that $\begin{array} { r } { \widetilde { U } _ { N , R } ( { \cal S } ) = R ^ { - 1 } \sum _ { r } Z _ { r } } \end{array}$ . The $Z _ { r }$ are i.i.d. functions of the independent draws $\mathcal { R } _ { r } ,$ lie in $[ 0 , 1 ] .$ , and have mean $\dot { U } _ { N } ( S )$ by equation 4. Hence $\mathbb { E } _ { \mathcal { R } } [ \widetilde { U } _ { N , R } ( S ) ] = \widehat { U } _ { N } ( S )$ , and $\mathrm { V a r } _ { \mathcal { R } } ( \widetilde { U } _ { N , R } ( S ) ) = R ^ { - 1 } \mathrm { V a r } ( Z _ { 1 } ) \leq 1 / ( 4 R )$ , since a variable supported on $[ 0 , 1 ]$ has variance at most $1 / 4$ □