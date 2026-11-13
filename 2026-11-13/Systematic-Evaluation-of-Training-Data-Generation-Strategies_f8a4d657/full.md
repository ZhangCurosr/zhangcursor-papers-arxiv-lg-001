# Systematic Evaluation of Training Data Generation Strategies for Machine Learning-Based Molecular and Materials Property Prediction

Minh Nghia Nguyen,<sup>1</sup> Beate Paulus,<sup>1</sup> and Akhil S. Nair<sup>2,</sup> <sup>∗</sup>

<sup>1</sup>Institute of Chemistry and Biochemistry, Freie Universit¨at Berlin, Arnimallee 22, 14195 Berlin, Germany <sup>2</sup>Department of Polymer and Functional Materials, CSIR-Indian Institute of Chemical Technology, 500007, Hyderabad, India (Dated: November 13, 2026)

Machine-learning (ML) models for molecular and materials property prediction are often trained on large datasets curated through arbitrary or generic sampling strategies. Such approaches can lead to the over- or under-representation of regions of chemical and materials space that are most relevan to the prediction task. An important challenge is therefore how to eficiently curate training data to maximize predictive performance while minimising the number of labelled examples required. Here, we systematically compare training data acquisition strategies for molecular and materials property prediction, ranging from random sampling to various active learning approaches. We evaluate their performance across diferent model classes, representations, datasets, and target properties. Our results show that the efectiveness of data acquisition strategies depends strongly on the interplay between the model, dataset, and prediction task, although uncertainty-based sampling generally outperforms alternative strategies. Based on these findings, we derive guidelines for selecting data acquisition strategies under a given labelling budget.

## I. INTRODUCTION

Machine learning (ML) can accelerate molecular and materials discovery by predicting properties that would otherwise require expensive experiments or high-fidelity simulations. The predictive performance of ML models depends not only on their architectures but also on the quality, diversity, and composition of the training data. However, datasets are often constructed through exhaustive or near-random sampling of chemical space, leading to substantial redundancy among chemically or structurally similar samples. Such redundancy increases both data-generation and model-training costs without necessarily yielding commensurate improvements in predictive accuracy. Recent studies have demonstrated that large molecular and materials datasets can be substantially reduced while retaining much of their predictive performance, although the extent of redundancy depends on the target property, dataset, and representation [1–3].

Active learning (AL) provides a systematic alternative to passive dataset construction by iteratively selecting samples for labeling according to criteria such as predictive uncertainty, structural diversity, or their combination. This approach is particularly valuable in atomistic modelling, where obtaining electronic-structure la bels is computationally expensive and adequately sampling diverse chemical environments remains challenging [4]. Nevertheless, the efectiveness of sampling strategies used in AL to acquire the data is not universal; it can vary with the dataset, target property, model class, data representation, and acquisition criterion [5, 6]. Consequently, identifying an eficient data-generation strategy for a given prediction task requires understanding how these factors interact. Despite growing interest in data-eficient learning, a systematic assessment of their individual and combined efects across diferent model classes, representations, and molecular and materials datasets remains limited.

In this work, we investigate three central questions: (i) To what extent are widely used molecular and materials datasets redundant across diferent ML model classes? (ii) Which sampling strategies reduce labelling require ments most efectively while preserving predictive accuracy? and (iii) How does the data generation eficiency depend on the representation used to describe molecules and materials? To address these questions, we benchmark conventional ML and deep-learning (DL) models across datasets spanning molecular systems and inorganic crystals, considering dataset sizes ranging from approximately 10<sup>3</sup> to 10<sup>5</sup> samples. By systematically comparing alternative sampling strategies, model classes, and data representations, we quantify the trade-ofs between training-set size, labeling efort, and predictive performance. This analysis provides a comparative assessment of data eficiency across diverse learning settings and identifies factors that govern the efectiveness of training-data generation methods for molecular and materials property prediction.

## II. METHODS

## A. Datasets

For molecular property prediction, we consider the QM9 dataset [8], comprising 133,881 small organic molecules after removing duplicate structures and invalid entries. The target property is the HOMO–LUMO gap $( \Delta E _ { g a p } )$ , calculated at the B3LYP level of theory. To probe the same domain at a much smaller scale, we use QM9small, a random 5% subset of QM9 containing 6,695 molecules with the same target property. For materials property prediction, we use the Matbench Materials Project (MP) dataset, which contains 132,752 inorganic crystals, with the DFT-PBE formation energy per atom $\left( \Delta E _ { f } \right)$ as the target property [9, 10]. As a smaller dataset for materials property prediction, we consider the Pourbaix Stability dataset introduced in our previous study [12], comprising 6,870 inorganic crystals with DFT-HSE06-calculated Pourbaix decomposition free energies $( \Delta G _ { p b x }$ in $\mathrm { e V / a t o m ) }$ as the target property. The four datasets thus form two large–small pairs, one for molecules and one for materials, which allows the efect of dataset size to be examined alongside that of the target property. The distributions of the target properties across all four datasets are shown in Figure 1.

## B. Machine Learning Models and Representations

For classical ML models, we employ descriptor-based representations to encode molecules and materials. For the QM9 dataset, molecules are represented by eight scalar properties such as $\mu$ (dipole moment), α (isotropic polarizability), ZPVE (zero-Point vibrational Energy), $U _ { 0 } , U _ { 2 9 8 }$ (internal energies at 0 and 298.15 K) $H _ { 0 } , H _ { 2 9 8 }$ (enthalpies at 0 and 298.15 K), $G _ { 2 9 8 }$ (Gibbs free energy at 298.15 K) and $C _ { v }$ (heat capacity at 298.15 K). These are further combined with Morgan fingerprints (radius of 2 and 2 048 bits) [13, 14] and MACCS keys [15], generated using RDKit [16]. For the Matbench MP dataset, we use composition-weighted elemental descriptors from the Magpie feature set [17], implemented through matminer [18]. For the Pourbaix stability dataset, we use the feature set provided as in Ref. [12], which augments the Magpie features with additional elemental descriptors.

We consider three classical ML algorithms: random forests (RF) [19], kernel ridge regression (KRR) with a Laplacian kernel, and extreme gradient boosting (XGB) [20]. For KRR, we employ the Nystr¨om approximation [21] for the two larger datasets to reduce computational costs. The QM9small and Pourbaix stability datasets use exact KRR. The predictions from the three models are combined using a voting regressor. We use the voting ensemble rather than a single member because it ofers a stronger and more stable representative of the classical model class. Details of the model configurations and hyperparameters are provided in Appendix Table IV.

For deep learning, we employ SchNet [7], a continuousfilter convolutional neural network model that takes atomic numbers and Cartesian coordinates as input and learns atom-centred representations through six continuous-filter interaction blocks, followed by an atomwise readout layer. Since both targets are intensive, so we use mean pooling for QM9 and Matbench MP. SchNet constructs the QM9 radius graph internally from the Cartesian coordinates with a cutof of 6.0 <sup>˚</sup>A. For the materials in the Matbench MP dataset, neighbour lists with a cutof of 5.0 <sup>˚</sup>A are constructed using ASE [22] and supplied to a periodic variant of SchNet. Further details of the training procedure are provided in Appendix A. SchNet is not evaluated on the Pourbaix stability and QM9small datasets because of their relatively small size.

## C. Sampling strategies

We considered a pool-based sampling framework in this work [11]. In this framework, an initial labelled set $\mathcal { L } ^ { ( 0 ) }$ of $N _ { 0 }$ samples is drawn at random from the pool. At iteration t the model is trained on $\boldsymbol { \mathcal { L } } ^ { ( t ) }$ , and a batch of b samples is selected from the unlabeled pool $\mathcal { U } ^ { ( t ) }$ , labeled, and added to $\boldsymbol { \mathcal { L } ^ { ( t ) } }$ [23]. All strategies and models share the same initial set within each repetition. Three acquisition strategies were compared:

Random sampling (RS) draws b candidates uniformly and serves as the baseline.

Uncertainty sampling (US) uses query-bycommittee [24], scoring each candidate by the variance of the committee predictions,

$$
a _ { \mathrm { U S } } ( x ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \bigl ( \hat { y } _ { m } ( x ) - \bar { y } ( x ) \bigr ) ^ { 2 } ,\tag{1}
$$

with $\bar { y }$ the committee mean. For the classical ML models, the committee consists of RF, KRR and XGB models with the disagreement among these diferent models providing an uncertainty signal rather than bootstrap resampling of a single model, which was used for SchNet $( M \ = \ 3$ models were trained on bootstrap resamples of $\mathcal { L } ^ { ( t ) } )$ . For Matbench MP, the US committee used a reduced SchNet configuration for faster training (fewer filters, epochs and a smaller batch size than the main model).

Diversity sampling (DS) applies TypiClust [25] as implemented in scikit-activeml package [26]. All samples are clustered with k-means [27] into $\dot { | \mathcal { L } ( t ) | } + b$ clusters; for each of the b acquisitions, the largest cluster without labelled members is selected, and its most typical sample is queried, where typicality is the inverse mean Euclidean distance to the $k = 5$ nearest neighbours [28] within the cluster,

$$
\mathrm { T y p } ( x ) = \Big ( \frac { 1 } { k } \sum _ { x _ { i } \in k - \mathrm { N N } ( x ) } \| x - x _ { i } \| _ { 2 } \Big ) ^ { - 1 } .\tag{2}
$$

Clustering operates in descriptor space for the classical ML models and in the learned, latent space for the SchNet model.

Initial set sizes, batch sizes and iteration counts for AL are listed in Table I. For QM9 and Matbench MP, the budgets, $\mathrm { i . e . , }$ final training-set sizes, were 52 106 and 50 000 labels, i.e. half of the training pool ( 40% of the full dataset); for the Pourbaix stability dataset, the entire pool (5,496 labels) was consumed. Ensemble experiments were repeated with five random seeds, each with its own initial set, shared across strategies within a run whereas SchNet experiments were run only once owing to computational cost.

![](images/4ef2531fbb1647504b688f09a51bfa75849e5f8f11ba3f7440a7819a4934883c.jpg)

(a)  
![](images/4b7b2576adbc2f0d42be34eb056c3aef6be1cb431c4858a17818b4e0f324035e.jpg)  
(c)

![](images/20dd45e6576cad8044066edf961a7dbabab877646eedd2de7d46f5a7e0e23456.jpg)

(b)  
![](images/d5a4fa3c490774ea814388119fd1e12af9490dd768c0e37603c0df4c6098b54c.jpg)  
(d)  
FIG. 1: Target property distributions of (a) QM9, (b) QM9small, (c) the Matbench MP and (d) the Pourbaix decomposition free energy dataset. The legend in each panel reports the total number of samples $N _ { \mathrm { t o t } }$ . All histograms use a constant bin width of 0.25 in the units of the respective x-axis. Symbols on the x-axes denote the target properties: $\Delta E _ { g a p } ,$ the HOMO-LUMO gap (eV); $\Delta E _ { f } ,$ , the formation energy per atom (eV/atom); and $\Delta G _ { p b x }$ , the Pourbaix decomposition free energy per atom (eV/atom).

The performance of sampling strategies is reported as test RMSE (primary metric), MAE and $R ^ { 2 }$ . Following Li et al. [1], data eficiency is quantified by

$$
N _ { 1 0 \% } = \mathrm { m i n } \{ N : { \mathrm { R M S E } } ( N ) \leq 1 . 1 \times { \mathrm { R M S E } } _ { \mathrm { f u l l } } \} ,\tag{3}
$$

where $\mathrm { R M S E } _ { \mathrm { f u l l } }$ is the error of the model trained on the full pool. The crossing was obtained by linear interpolation on a logarithmic N axis, counted only if the error stayed below the threshold thereafter, and is also given as a fraction of the pool. A smaller $N _ { 1 0 \% }$ than RS indicates high redundancy of the dataset for the given model and property. Because $N _ { 1 0 \% }$ is defined relative to each model’s own baseline, cross-model comparisons use absolute errors.

TABLE I: Data splits and AL settings: initial size $N _ { 0 }$ batch size $b ,$ number of iterations $n _ { \mathrm { A L } }$
<table><tr><td>Dataset</td><td>Train</td><td>Val.</td><td>Test</td><td> $N _ { 0 }$ </td><td></td><td> $\textit { b } \ n _ { \mathrm { A L } }$ </td></tr><tr><td>QM9</td><td>107,106</td><td>13,386</td><td>13,389</td><td>7,106</td><td>5000</td><td>9</td></tr><tr><td>QM9small</td><td>5,356</td><td>670</td><td>669</td><td>356</td><td>500</td><td>10</td></tr><tr><td>Matbench MP</td><td>106,201</td><td>13,276</td><td>13,275</td><td>5,000</td><td>5,000</td><td>9</td></tr><tr><td>Pourbaix Stability</td><td>5,496</td><td>687</td><td>687</td><td>496</td><td>500</td><td>10</td></tr></table>

## III. RESULTS AND DISCUSSION

## A. Full-data baselines

Table II lists the test set performance for models trained on the full training data pool. SchNet outperforms the ensemble on both large datasets with a smaller RMSE of 0.090 vs. 0.339 eV for the HOMO-LUMO gap and 0.070 vs. 0.268 eV/atom for the formation energy prediction tasks. This can be attributed to the fact that the composition-derived descriptors carry no structural information and thus impose an information ceiling on a strongly structure-dependent property like formation energy, whereas SchNet encodes bond lengths, local coordination environments and atomic packing. The QM9small ensemble, trained on the 5,356-sample pool, reaches $\mathrm { R M S E } = 0 . 3 8 7 \ : \mathrm { e V }$ , slightly higher than the larger QM9 dataset. On the Pourbaix stability, the ensemble obtains a reasonable accuracy with $\mathrm { R M S E } = 0 . 2 4 3 ~ \mathrm { e V }$ since $\Delta G _ { p b x }$ is computed across diferent compositions present in the Pourbaix diagram of a material and hence compositional information alone aids the learning task.

![](images/ba530ad85a542d7915d35903c5780eece4c411541fc6a170aa3bd156392cedf3.jpg)  
(a) QM9: Descriptor-based ensemble

![](images/80842e7d44af6a2d334ad8124cb2094733ac99a08068d237299e944e4b313b41.jpg)  
(b) QM9: SchNet  
FIG. 2: AL learning performance curves on QM9 for (a) the descriptor-based ensemble of ML models and (b) SchNet. Curves show random sampling (RS), uncertainty sampling (US) and diversity sampling (DS). Solid black line indicates the RMSE obtained using the full dataset $\left( \mathrm { R M S E } _ { \mathrm { f u l l } } \right)$ and the dashed grey line indicates $1 . 1 \times \mathrm { R M S E _ { f u l l } }$ . The x-axis represents the number of samples in the training dataset and the y-axis represents the RMSE evaluated on a hold-out test set. The solid lines represent the mean values, and the shaded region in (a) spans the minimum–maximum range over five independent runs.

TABLE II: Test-set performance of models trained on the full training pool. RMSE and MAE are reported in the target property’s units.
<table><tr><td>Dataset</td><td>Model</td><td>RMSE</td><td>MAE</td><td> $R ^ { 2 }$ </td></tr><tr><td>QM9</td><td>Ensemble SchNet</td><td>0.339 0.090</td><td>0.257 0.062</td><td>0.931 0.995</td></tr><tr><td>QM9small</td><td>Ensemble SchNet</td><td>0.387</td><td>0.303</td><td>0.907</td></tr><tr><td>Matbench MP</td><td>Ensemble SchNet</td><td>0.268 0.070</td><td>0.140 0.036</td><td>0.947 0.996</td></tr><tr><td>Pourbaix Stability</td><td>Ensemble SchNet</td><td>0.243</td><td>0.158</td><td>0.993</td></tr></table>

## B. Active Learning Performance

## 1. HOMO-LUMO Gap Prediction

The AL performance of ensemble-based classical mod els and SchNet for HOMO-LUMO gap is shown in Fig. 2a & b, respectively. In these plots, the RMSE evaluated on the hold-out set is compared with the number of samples acquired during AL iterations. For the ensemblebased models, all the sampling strategies start from a test RMSE of 0.415 eV at a training data size of 7 106 and decrease monotonically, indicating improved performance with more training data. It can be seen that the performance curve of US lies below RS at every iteration, reaching the 10% threshold at $N _ { 1 0 \% } \approx 1 9 0 0 0$ (17% of the pool) compared with  26 000 (25%) for RS, and at the end of the budget (52 106 labels), RMSE reaches 0.347 eV, compared with 0.354 eV for RS. DS occupies an intermediate position $( N _ { 1 0 \% } \approx 2 4 0 0 0$ , 23% of the pool; final $\mathrm { R M S E } = 0 . 3 5 3 ~ \mathrm { e V } )$ , indicating that it is systematically better than RS in terms of $N _ { 1 0 \% }$ , but the improvement is small and, at higher dataset sizes, is essentially indistinguishable from that of RS. The consis tently observed higher performance of US indicates that disagreement among diferent regressors is a meaningful informativeness signal for the HOMO-LUMO gap, steering acquisition toward underrepresented regions of chemical space rather than merely toward already acquired labels. Coverage-oriented DS in descriptor space, which does not use any information about the model’s current errors, captures only part of this benefit.

![](images/87c5405ddf09fdb0aa2758892c60ba35ad2d1af950bacc4f65baf3d86a27404c.jpg)  
FIG. 3: AL learning curves on QM9small for the descriptor-based ensemble; layout as in Fig. 2. The budget covers the full pool, so all strategies converge to the full-data limit.

On the contrary to the ensemble, SchNet starts from a lower error (0.276 vs. 0.415 eV at 7 106 labels) and improves by more than a factor of two within the budget (Fig. 2b). It is therefore more accurate than the ensembles even at the smallest budget. However, no sampling strategy reaches the 10% threshold $( 1 . 1 \times \mathrm { R M S E _ { f u l l } } \ =$ 0.099 eV), and the lowest error achieved after full AL iterations (0.114 eV observed for US) remains 26% above the full-data baseline. The performance curves are monotonic, with all three strategies decreasing at every cycle apart from a 0.0001 eV uptick for US at 42 106 labels. US attains the lowest final error (0.114 eV), followed by RS (0.119 eV) and DS (0.126 eV); the three values differ by at most 0.012 eV, small relative to the 0.16 eV reduction achieved over the budget. Note that we fixed the number of AL iterations to nine for QM9 and Matbench MP, stopping at training-set sizes of 52 106 and 50 000 labels, respectively, rather than training until the complete dataset is scanned.

Efect of pool size. To analyse the efect of pool size for a fixed property and model, we repeated the AL experiment on QM9small, a random 5% subset of QM9 (6 695 molecules; 5 356 in the training pool), using the descriptor-based ensemble only. The full-data baseline is 0.387 eV (Table II) and the AL settings are listed in Table I. All strategies start from 0.587 eV and converge to 0.387 eV at 5 356 labels (Fig. 3). Because the budget covers the full pool, all strategies end at the same accuracy; their benefit therefore lies in reaching the threshold with fewer labels, not in a better final model. US reaches the 10% threshold first, at $N _ { 1 0 \% } \approx 2 1 0 0$ labels (40% of the pool), compared with 2 500 (46%) for RS and 2 800 (52%) for DS. US therefore saves 400 labels relative to RS ( 16% of the RS budget; the threshold fraction drops from 46% to 40%), while DS requires  300 labels more than RS. US is the most data-eficient strategy, followed by RS and DS. This difers from the full QM9 dataset, where DS was slightly better than RS, consistent with the smaller pool ofering less redundancy for descriptorspace DS to exploit. It should be also noticed that the threshold fractions are larger than those on QM9 dataset (40–52% vs. 17–25%).

## 2. Formation Energy Prediction

In the case of the formation energy prediction task based on the Matbench MP dataset, US reaches the 10% threshold at $N _ { 1 0 \% } \approx 1 4 0 0 0$ labels (13% of the pool), RS at 16 000 (15%) and DS at 17 000 (16%) (Fig. 4a). This clearly shows the very high redundancy of this dataset, which is expected because of very similar inorganic crystal structures present in MaterialsProject datasets from high-throughput simulations. At the end of the budget, all curves are within 3% of the full-data baseline, and the mean RMSE increases in the order US, DS, RS (0.271, 0.273 and 0.276 eV/atom, respectively). The enhanced early performance of US suggests that querying models for highly uncertain configurations efectively isolates and corrects the most critical knowledge gaps in the materials space early in the training process. Interestingly, the lower final error of DS compared with RS emerges only at larger training-set sizes, although DS reaches the 10% threshold slightly later (17 000 vs. 16 000 labels; Table III): DS is marginally worse than RS up to 15 000 labels, essentially indistinguishable at 20 000, and lower on average from 25 000 labels onward, with the gap growing to 0.003 eV/atom at 50 000. This gradual separation indicates that at lower data regimes, RS provides suficiently broad phase-space coverage. However, as the dataset grows, DS begins to outpace RS by systematically avoiding the redundant sampling of overrepresented crystal structures, ensuring a more eficient exploration of the remaining chemical space.

The performance of the SchNet model on Matbench MP shows a diferent pattern compared to the QM9 dataset (Fig. 4b). Starting from 0.275 eV/atom at 5 000 labels, US is observed to show the highest performance through most of the AL iterations, except at the later stage, where it converges to similar performance to DS. It is to be noted that DS outperforms RS at every subsequent iteration, ending at 0.080 versus 0.090 eV/atom (11% lower error). No strategy reaches the threshold of

![](images/869cd549f1cb0be8f6f5c60382113ebf1abcba336f231edd2d136f2015dbfde4.jpg)  
(a) Matbench MP: Descriptor-based ensemble

![](images/0c09824d84a4b6eaab1703f089ee2770e678a211ffc04c1c5e3cf387735e30a5.jpg)  
(b) Matbench MP: SchNet  
FIG. 4: AL learning performance curves for formation energy prediction task based on the Matbench MP dataset, layout as in Fig. 2.

0.077 eV/atom within the 50 000-label budget, although DS comes closest. These results indicate that the relative eficacy of AL strategies is strongly data-dependent, meaning a strategy optimized for one dataset may not generalise as the most efective choice for another.

## 3. Pourbaix Decomposition Energy Prediction

Since the pool is small (5,496 labels) and structural relaxations were not performed at the HSE06 level for curating this dataset, we did not evaluate SchNet on this dataset, and hence the comparison is limited to sampling strategies for the ensemble. Here, the diferences between strategies are the largest among the three properties considered when expressed as a fraction of the pool (Fig. 5). US reaches the 10% threshold at 3 000 samples (54% of the pool), DS at 3 600 (66%) and RS at 4 200 (77%). US thus saves 1 200 labels, or about 22% of the pool, relative to RS. We extended the sampling to the full pool, where all strategies converge to the fulldata limit (5 496 labels). As observed for the QM9small dataset, this benefit lies in reaching the threshold earlier rather than in a better final model (Table III). This is particularly important here because for a property with expensive labels (HSE06-level calculations), reaching a target accuracy earlier translates directly into reduced computational cost.

![](images/be4e1bec283b17e0c5b28d417675beaf6a512dd3fe48e6564ef394fca277090a.jpg)  
FIG. 5: AL learning performance curves on Poubrbaix stability dataset for the ensemble; layout as in Fig. 2. The budget covers the full pool, so all strategies converge to the full-data limit.

TABLE III: Data eficiency $N _ { 1 0 \% }$ for random sampling (RS), uncertainty sampling (US) and diversity sampling (DS) sampling strategies for all the datasets considered in this study. The fraction of the training pool is indicated in parentheses
<table><tr><td>Dataset</td><td>Model</td><td>RS</td><td>US</td><td>DS</td></tr><tr><td>QM9</td><td>Ensemble</td><td>26 000 (25%)</td><td>19 000 (17%)</td><td>24000 (23%)</td></tr><tr><td>QM9small</td><td>Ensemble</td><td>2500 (46%)</td><td>2100 (40%)</td><td>2800 (52%)</td></tr><tr><td>Matbench MP</td><td>Ensemble</td><td>16 000 (15%)</td><td>14000 (13%)</td><td>17000 (16%)</td></tr><tr><td>Pourbaix stability</td><td>Ensemble</td><td>4200 (77%)</td><td>3 000 (54%)</td><td>3600 (66%)</td></tr></table>

Note: $N _ { 1 0 \% }$ is obtained by linear interpolation of the mean learning curve and rounded to the nearest 1 000 labels (100 for QM9small and Pourbaix stability). Percentages are relative to the training pools in Table I and are computed from the unrounded crossings, so subtracting the rounded $N _ { 1 0 \% }$ values may difer slightly.

## C. Dataset Redundancy Analysis

To further assess the performance of various strategies across all the datasets, we list their corresponding $N _ { 1 0 \% }$ values in Table III for the ensemble. It is particularly important to note that, even for RS, only 15– 25% of the pool is suficient on QM9 and Matbench MP datasets, so that ≳75% of the entries contribute little additional accuracy. This agrees with the finding of Li et al. that only about 13–30% of Materials Project entries are informative for composition-based tree models at the same threshold [1]. QM9small, a 5% subset of QM9, requires 46% of its pool under RS, between the two large datasets and the Pourbaix stability dataset. For the $\Delta G _ { p b x }$ prediction, RS needs 77% of the pool (4 200 samples). This sharp contrast can be attributed to the significantly smaller overall size and lower inherent redundancy of the HSE06 dataset compared to the massive scale of QM9 and Matbench MP. Because the total available pool for the Pourbaix stability dataset is highly constrained, RS struggles to capture the full structural diversity without exhausting the majority of the available data. However, it is precisely in this data-scarce regime that AL demonstrates its greatest value. Relative to RS, US reduces the threshold fraction from 77% to 54%, reaching the target with 3 000 labels instead of 4 200 (Table III). This emphasizes that when data are limited and computationally expensive to acquire, as is typical for high-fidelity properties like DFT-HSE06, uncertaintydriven sampling is crucial for maximizing model accuracy with a minimal number of calculations.

## D. Efect of Choice of Surrogate Models

Herein, we investigate the efect of model choice (ensemble vs SchNet) on the performance of AL performance for the same dataset. Because $N _ { 1 0 \% }$ is defined relative to each model’s own full-data error, the threshold is lower in absolute terms for the more accurate model. The SchNet curves never reach their threshold within the budget and hence $N _ { 1 0 \% }$ could not be determined for SchNet. It is important to note that $N _ { 1 0 \% }$ values are not directly comparable across model classes. We therefore compare the models using absolute errors (Fig. 6). For the QM9 dataset, SchNet is superior at every budget tested: it is already better at the initial 7 106 labels (0.276 vs. 0.415 eV) and the gap widens to 0.114 vs. 0.347 eV at 52 106 labels. On Matbench MP, SchNet is also superior at all budgets tested, and the gap widens to roughly a factor of three by the end of the 50 000- label budget. The ensemble is ultimately limited by its fixed descriptors with the composition-derived features that have limited representation capability for the structural information that SchNet learns, and for QM9 the fingerprints and scalar descriptors do not capture the HOMO-LUMO gap as accurately as the learned representation. These results ofer practical guidelines for AL deployments. When full structural information is unavailable, the descriptor-based ensemble combined with US is the most robust option tested. When structural data are available, SchNet is the more accurate model class at all budgets tested on both QM9 and Matbench MP; the ensemble is computationally cheaper but does not provide a low-budget accuracy advantage.

## E. Limitations

Due to GPU availability constraints, the SchNet models were trained using a single run per configuration. Consequently, we are unable to report standard deviations for the SchNet learning curves, which hinders a rigorous assessment of the model’s statistical variance across diferent random weight initializations.

## IV. CONCLUSIONS

We evaluated active learning (AL) strategies for curating training data across molecular and materials datasets, using both a descriptor-based ensemble and the structure-aware SchNet model. Our results show that how much training data is needed, and how much a well chosen selection strategy can save, are not fixed properties of a dataset but depend on the interplay of dataset size, target property, model and representation. For the descriptor-based models, only 15–25% of the two large datasets (QM9 and Matbench MP) was needed to come within 10% of the full-data error under random sampling, whereas the smaller datasets required 46–77%. Within this picture, uncertainty sampling was the most label eficient strategy for the descriptor-based models on all four datasets, reducing the number of labels required by roughly 13–30% relative to random sampling, with the largest gain on the smallest and most expensive dataset. Diversity sampling, by contrast, gave no consistent advantage, and the best strategy for SchNet changed with the dataset. Moreover, the accuracy of the model at the start of AL strongly shaped its performance throughout the sampling, and the model and its input representation often mattered more than the choice of sampling strategy. Building an eficient training set therefore requires balancing these factors rather than applying a single recipe. Future work will focus on a benchmarking platform that ofers the community an expanded suite of ML models, sampling strategies and performance metrics for quantifying the redundancy of target properties.

![](images/f62ef4e092082ef52bcd7a405f11f9e29bfcd668becad22f405991e11eff7765.jpg)  
(a) QM9: Ensemble vs. SchNet

![](images/1429fb58529f76a16eed667b58be17401003ab0bf03ef01c456f198aad0eb1fd.jpg)  
(b) Matbench MP: Ensemble vs. SchNet  
FIG. 6: Ensemble vs. SchNet with the best strategy per dataset and model: (a) QM9 (uncertainty sampling for both); (b) Matbench MP (uncertainty sampling for the ensemble and diversity sampling for SchNet).

## ACKNOWLEDGMENTS

A.S.N thanks the German Research Foundation (DFG) for their support through the Walter-Benjamin Fellowship Program (project No. 540316537).

## DATA AND CODE AVAILABILITY

The code used in this work is available at https:// github.com/minhnghian03/property-prediction-al. The datasets are publicly available from the cited sources.

## Appendix A: Data splits and training protocols

## 1. Data splits

Each dataset was divided into 80% training, 10% validation and 10% test subsets, stratified by quantile bins of the target property; the resulting sizes are listed in Table I. The training subset defines the AL pool, the validation subset is used for early stopping, and the test subset is held out for all reported metrics. Hyperparameters were selected by five-fold cross-validation on the training subset.

## 2. Classical ensemble

The classical model is an unweighted voting ensemble of RF, KRR and XGB. Continuous descriptors were standardized with a scaler fitted on the training subset, while fingerprint features were used without scaling. KRR used a Laplacian kernel, with the Nystr¨om approximation (2 500 components) for QM9 and Matbench MP, and exact KRR for HSE06 and QM9small. The ensemble prediction is the mean of the three members. Table IV lists the selected hyperparameters.

Targets were used in their native units: the QM9 and QM9small gap in Hartree and the formation and Pourbaix decomposition energies in eV/atom. QM9 predictions were converted to eV for all reported metrics.

## 3. SchNet

SchNet was trained with the MSE loss, an initial learning rate of $1 0 ^ { - 4 }$ , and mini-batches of 32 graphs for at most 150 epochs; the optimizer, weight decay and earlystopping settings are listed in Table V. The learning rate was halved after 5 epochs without validation improvement (minimum $1 0 ^ { - \bar { 6 } } )$ and early stopping used a patience of 20 epochs with the best-validation weights restored. QM9 used mean readout and an internal radius graph with a 6.0 <sup>˚</sup>A cutof. Matbench MP used mean readout and periodic neighbour lists with a $5 . 0 \mathrm { ~ \AA ~ }$ cutof constructed with ASE. QM9 targets were used in Hartree and converted to eV for reporting; Matbench MP targets were standardized to zero mean and unit variance before training and converted back to eV/atom for evaluation. Table V lists the architecture and training hyperparameters.

[1] K. Li, D. Persaud, K. Choudhary, et al., Nat. Commun. 14, 7283 (2023).

[2] E. Moerman, I. Poltavsky, A. Charkin-Gorbulin, and A. Tkatchenko, “Atoms that Matter: Largest Materials Datasets Are 99% Redundant, and Local Chemistry Can Guide the Required 1%,” arXiv preprint arXiv:2610.14392, 2026.

[3] Q. Li, N. Fu, S. S. Omee, J. Hu, npj Comput. Mater. 10, 245 (2024).

[4] M. Kulichenko, B. Nebgen, N. Lubbers, et al., Chem. Rev. 124, 13681 (2024).

[5] J. Bi, Y. Xu, F. Conrad, H. Wiemer, S. Ihlenfeldt, Sci. Rep. 15, 37167 (2025).

[6] A. S. Nair and L. Foppa, Digit. Discov. 5, 2366 (2026).

[7] K. T. Sch¨utt, P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. M¨uller, Adv. Neural Inf. Process. Syst. 30, 991–1001 (2017).

[8] R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld, Sci. Data 1, 140022 (2014).

[9] A. Jain, S. P. Ong, G. Hautier, W. Chen, et al., APL Mater. 1, 011002 (2013).

[10] A. Dunn, Q. Wang, A. Ganose, D. Dopp, A. Jain, npj Comput. Mater. 6, 138 (2020).

[11] D. D. Lewis, “A sequential algorithm for training text classifiers: Corrigendum and additional data,” ACM SI-GIR Forum, vol. 29, pp. 13–19, 1995.

[12] A. S. Nair, L. Foppa, M. Schefler, npj Comput. Mater. 11, 150 (2025).

[13] H. L. Morgan, J. Chem. Doc. 5, 107 (1965).

## 4. Implementation and hardware

The implementation used Python 3.13.9, scikit-learn 1.7.2, XGBoost 3.2.0, PyTorch 2.12.0 (CUDA 12.6), PyTorch Geometric 2.8.0 with the torch-scatter, torchsparse extensions and pyg-lib, Skorch 1.3.1, RDKit 2026.3.1, matminer 0.10.0, ASE 3.26.0 and scikitactiveml 1.0.0. SchNet was trained on NVIDIA GeForce RTX 3050, GeForce GTX 1080 and Quadro RTX 5000 Max-Q GPUs, and the ensembles on Intel Xeon Gold 6526Y CPUs.

[14] D. Rogers and M. Hahn, J. Chem. Inf. Model. 50, 742 (2010).

[15] J. L. Durant, B. A. Leland, D. R. Henry, J. G. Nourse, J. Chem. Inf. Comput. Sci. 42, 1273 (2002).

[16] G. Landrum, RDKit: Open-source cheminformatics, https://www.rdkit.org.

[17] L. Ward, A. Agrawal, A. Choudhary, C. Wolverton, npj Comput. Mater. 2, 16028 (2016).

[18] L. Ward, A. Dunn, A. Faghaninia, et al., Comput. Mater. Sci. 152, 60 (2018).

[19] L. Breiman, Mach. Learn. 45, 5 (2001).

[20] J. H. Friedman, Ann. Stat. 29, 1189 (2001).

[21] C. K. I. Williams and M. Seeger, Adv. Neural Inf. Process. Syst. 13 (2000).

[22] A. H. Larsen, J. J. Mortensen, J. Blomqvist, et al., J. Phys.: Condens. Matter 29, 273002 (2017).

[23] B. Settles, Active Learning Literature Survey, Computer Sciences Technical Report 1648, University of Wisconsin– Madison (2009).

[24] H. S. Seung, M. Opper, H. Sompolinsky, in Proc. 5th Annu. Workshop Comput. Learn. Theory (ACM, 1992), pp. 287–294.

[25] G. Hacohen, A. Dekel, D. Weinshall, in Proc. ICML (2022).

[26] M. Herde, M. T. Pham, D. Kottke, et al., scikit-activeml: A comprehensive and user-friendly active learning library, J. Mach. Learn. Res. 27(184), 1–20 (2026).

[27] S. P. Lloyd, IEEE Trans. Inf. Theory 28, 129 (1982).

[28] T. Cover and P. Hart, IEEE Trans. Inf. Theory 13, 21 (1967).

TABLE IV: Hyperparameters of the ensemble members (QM9, QM9small, Matbench MP and Pourbaix stability), selected by five-fold cross-validation on the training subset.
<table><tr><td>Model</td><td>Parameter</td><td>QM9</td><td>QM9small</td><td>MP</td><td>HSE06</td></tr><tr><td>RF</td><td>n_estimators</td><td>100</td><td>100</td><td>500</td><td>500</td></tr><tr><td></td><td>max_features</td><td>sqrt</td><td>sqrt</td><td>0.8</td><td>0.8</td></tr><tr><td></td><td>min_samples_leaf</td><td>2</td><td>2</td><td>1</td><td>1</td></tr><tr><td>KRR</td><td>kernel</td><td>Laplacian</td><td>Laplacian</td><td>Laplacian</td><td>Laplacian</td></tr><tr><td></td><td>α</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td></td><td> $\gamma$ </td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td></td><td>Nyström components</td><td>2500</td><td>一</td><td>2500</td><td>一</td></tr><tr><td>XGB</td><td>n_estimators</td><td>100</td><td>100</td><td>500</td><td>500</td></tr><tr><td></td><td>learning-rate</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td></td><td>max_depth</td><td>4</td><td>4</td><td>4</td><td>4</td></tr><tr><td></td><td>colsample_bytree</td><td>0.6</td><td>0.6</td><td>0.8</td><td>0.8</td></tr><tr><td></td><td>min_child_weight</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td></td><td>reg-alpha</td><td>0.1</td><td>0.1</td><td>1.0</td><td>1.0</td></tr><tr><td></td><td>reg_lambda</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td></td><td>subsample</td><td>0.8</td><td>0.8</td><td>0.6</td><td>0.6</td></tr></table>

TABLE V: SchNet architecture and training hyperparameters.
<table><tr><td>Parameter</td><td>QM9</td><td>Matbench MP</td></tr><tr><td>hidden channels</td><td>64</td><td>64</td></tr><tr><td>filters</td><td>256</td><td>128</td></tr><tr><td>interaction blocks</td><td>6</td><td>6</td></tr><tr><td>Gaussians</td><td>50</td><td>50</td></tr><tr><td>cutoff (Å)</td><td>6.0</td><td>5.0</td></tr><tr><td>readout</td><td>mean</td><td>mean</td></tr><tr><td>optimizer</td><td> $\mathrm { A d a m W }$ </td><td> $\mathrm { A d a m W }$ </td></tr><tr><td>learning rate</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>weight decay</td><td> $1 0 ^ { - 6 }$ </td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>batch size</td><td>32</td><td>32</td></tr><tr><td>max. epochs</td><td>150</td><td>150</td></tr><tr><td>early-stopping patience</td><td>20</td><td>20</td></tr><tr><td>LR-scheduler patience</td><td>5</td><td>5</td></tr></table>