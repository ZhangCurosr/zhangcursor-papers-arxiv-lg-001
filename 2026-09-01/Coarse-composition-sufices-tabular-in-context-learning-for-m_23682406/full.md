# Coarse composition sufices: tabular in-context learning for multi-activity antimicrobial peptide profiling

Raunak Kumar<sup>1,</sup> <sup>†</sup>, Anuj Pal<sup>1,</sup> <sup>†</sup>, Dhruvi Solanki<sup>2,</sup> <sup>†</sup>, Parikshit Pareek<sup>3</sup>, Juhi Singh<sup>2,∗</sup>, and Jitin Singla<sup>4,∗</sup>

<sup>1</sup>Department of Electronics and Communication Engineering, Indian Institute of Technology Roorkee, Roorkee 247667, Uttarakhand, India

<sup>2</sup>Department of Bioengineering, Indian Institute of Science, Bengaluru 560012, Karnataka, India <sup>3</sup>Department of Electrical Engineering, Indian Institute of Technology Roorkee, Roorkee 247667, Uttarakhand, India

<sup>4</sup>Department of Biosciences and Bioengineering, Indian Institute of Technology Roorkee, Roorkee 247667, Uttarakhand, India <sup>†</sup>These authors contributed equally. <sup>∗</sup>Corresponding authors.

## Abstract

Antimicrobial peptides (AMPs) often act against multiple pathogen classes, making multi-label activity prediction a more realistic screening target than binary antimicrobial classification. The ESCAPE benchmark formalizes this setting, but leading approaches typically rely on multimodal, structure-conditioned deep models that are costly to train and tune. We show that a simple, sequence-only pipeline can match and surpass these methods by combining 330 interpretable sequence descriptors with TabPFN, a tabular foundation model that performs in-context prediction in a single forward pass without gradient-based training or hyperparameter search. On ESCAPE (82,359 peptides; five labels), a label-powerset TabPFN model achieves mAP-5 = 77.8%, improving on the previously best reported 72.1%. A probabilistic classifier chain is the first method to match or exceed the best published average precision on each of the five labels simultaneously. The gains persist under the prior state-of-the-art single-fold training protocol, indicating they are not a training-set-size artefact, and are largest for remote homologues (+11.2 points below 30% sequence identity). Ablations further show that predicted structure is unnecessary at inference and that performance is not driven by any single descriptor family: ten global physicochemical scalars recover 91% of full-feature performance. Finally, explicitly modelling label dependence yields targeted benefits for scarce activities and supports ranking which activity to assay next from partial positive evidence.

Keywords: antimicrobial peptides; multi-label classification; tabular foundation models; benchmarking; sequence descriptors; assay prioritization

## 1 Introduction

Bacterial antimicrobial resistance was associated with an estimated 4.95 million deaths worldwide in 2019, of which 1.27 million were directly attributable [1]. Antimicrobial peptides (AMPs), typically cationic sequences of 10–50 residues, are among the most actively pursued alternatives to conventional antibiotics. Most AMPs act by physically disrupting microbial membranes rather than by engaging a single molecular target, a mechanism against which resistance is thought to evolve comparatively slowly [2–4]. A defining property of the class is that a single peptide is frequently active against more than one kind of pathogen: bacteria, fungi, viruses, and parasites [5]. The prediction target that matters for screening is therefore a multi-activity profile, not an isolated binary activity.

Computational AMP prediction has nonetheless been dominated by binary antimicrobial-versusnon-antimicrobial classification. Random forests over composition–transition–distribution descriptors [6, 7], attention-augmented recurrent networks [8], and fine-tuned protein language models [9] all address that task, and comparative assessments have repeatedly found them dificult to separate once evaluation is standardized [10]. Multi-activity prediction is a more recent development. Graph neural networks over molecular representations [11], transformer architectures with asymmetric losses for label imbalance [12], and multi-task convolutional networks [13] predict several activities jointly. The ESCAPE benchmark [14] has recently standardized the task by integrating 82,359 peptides from 27 curated repositories under a fixed split and reporting a multimodal transformer baseline that utilizes both sequence and predicted three-dimensional structure. That benchmark is a substantial community resource: it fixes the split, retrains every prior method under a single protocol, and reports per-label as well as macro results. An informative feature of those results is that no method dominates, i.e., the leading method on macro average precision is not the leading method on any of antibacterial, antiviral, or antifungal activity, and diferent architectures lead on diferent labels.

The prevailing response to this task has been to add capacity and modalities. Structureconditioned and language-model-conditioned architectures are now standard in peptide property prediction [15–17], and each addition carries real cost, i.e., structure prediction for every candidate, GPU training, architecture search, and hyperparameter tuning that must itself be validated. Recent critical assessments in adjacent domains have shown that apparently large deep-learning advantages can shrink or disappear once simple baselines, homology control, and evaluation protocol are examined [18–20], and the same examination has not been carried out for multi-activity AMP prediction.

In this study we ask how far a deliberately simple, sequence-only model can be taken on ESCAPE, and what that performance implies about the benchmark itself. We pair 330 interpretable sequence descriptors with TabPFN,a prior-data fitted network pre-trained on millions of synthetic tabular classification tasks that takes a labelled reference set as in-context samples and returns calibrated posteriors in a single forward pass, with no gradient updates and no hyperparameter search on the target data [21–23]. We (i) evaluate standard multi-label transforms on a fixed descriptor matrix under the ESCAPE split and protocol; (ii) separate model quality from the quantity of labelled data available, by reproducing ESCAPE’s released multimodal baseline and repeating every comparison under its single-fold protocol; (iii) stratify by sequence identity to quantify generalization to remote homologues; (iv) test whether performance is driven by any particular descriptor family and whether structural input contributes at inference; and (v) measure when modelling label dependence helps, including for prioritizing assays from partial positive evidence.

Overall, the results favour simplicity. A label-powerset TabPFN model improves five-label macro average precision to 77.8% versus 72.1% for the previously best-performing method, and a probabilistic classifier chain is the first approach to match or exceed the best published average precision on all five labels simultaneously. These gains persist under the matched single-fold protocol, are largest for peptides remote from the peptides supplied in context, are not attributable to any single descriptor family, and remain largely intact with only ten global physicochemical scalars (91% of full-feature performance).

Table 1: Label structure of the ESCAPE benchmark. Positive counts and prevalence are given for the held-out test fold $( n = 1 6 , 4 8 9 )$ ; the conditional column is the fraction of antimicrobial test peptides carrying each activity.
<table><tr><td>Label</td><td>Test positives</td><td>Prevalence (%)</td><td>P(label | AM) (%)</td></tr><tr><td>Antibacterial</td><td>3,187</td><td>19.33</td><td>75.3</td></tr><tr><td>Antifungal</td><td>1,316</td><td>7.98</td><td>31.1</td></tr><tr><td>Antiviral</td><td>946</td><td>5.74</td><td>22.3</td></tr><tr><td>Antiparasitic</td><td>77</td><td>0.47</td><td>1.8</td></tr><tr><td>Antimicrobial</td><td>4,235</td><td>25.68</td><td>100.0</td></tr></table>

## 2 Materials and methods

## 2.1 Dataset

We used the Expanded Standardized Collection for Antimicrobial Peptide Evaluation (ESCAPE) benchmark [14]. ESCAPE contains 82,359 peptides with four activity labels (antibacterial, antifungal, antiviral, antiparasitic) and a deterministic parent Antimicrobial label $\left( y _ { \mathrm { A M } } = \bigvee _ { j } y _ { j } \right)$ . We used the predefined split: folds 1–2 (65,870 peptides) supply the in-context samples and the held-out fold $( n = 1 6 , 4 8 9 )$ is used only for testing. Label counts and prevalence in the test fold are summarized in Table 1; antiparasitic is the rarest label and thus constrains macro-averaged performance (see Section 3.1).

## 2.2 Evaluation metrics and statistics

Three averaging scales are used and are not interchangeable. We report mAP-5 (all five labels) to enable direct comparison against published results. For dependence analyses we use mAP-4, which averages over the four real activities and excludes the deterministic Antimicrobial parent. We also report mAP-3, which averages over antibacterial, antifungal, and antiviral only (excluding both the parent and Antiparasitic), because Antiparasitic has only 77 test positives. Unless stated otherwise, every average precision, macro average, per-label diference, bootstrap interval, and threshold-dependent score is reported on the 0–100 scale.

Interval estimates use 2,000 bootstrap resamples of the test peptides. All transforms are scored on identical resamples, and within each resample the macro metric is averaged across the three seeds, treating seeds as repeated measurements of one test set rather than as additional independent data. A diference is called significant when its two-sided 95% percentile interval excludes zero. Point estimates are means over three seeds and ± denotes the sample standard deviation.

## 2.3 Feature representation

Each peptide was encoded as a 330-dimensional numeric vector partitioned into eight interpretable families (Table 2). Feature extraction was identical for every transform compared here, so no diference between transforms can arise from representation. Composition–transition–distribution (CTD) descriptors encode three complementary views of a physicochemical property along the sequence: composition is the fraction of residues in each property class, transition the frequency of switches between classes at adjacent positions, and distribution the sequence positions at which the first, 25%, 50%, 75%, and 100% of a class are reached. A descriptor such as Hydrophobicity\_D\_3\_0 is therefore spatial rather than compositional.

Table 2: Feature representation (330 numeric columns).
<table><tr><td>Family</td><td>Content</td><td>n</td></tr><tr><td>ESM</td><td>ESM-2 embeddings, mean-pooled and PCA-reduced [17]</td><td>128</td></tr><tr><td>CTD</td><td>7 properties × 21 (composition 3, transition 3, distribution 15)</td><td>147</td></tr><tr><td>AAC</td><td>Amino-acid composition</td><td>20</td></tr><tr><td>AAIndex</td><td>PCA-reduced AAindex residue-property matrices [24]</td><td>15</td></tr><tr><td>PhysChem</td><td>Length, net charge, charge density, pI, instability index, aromaticity, aliphatic index, Boman index, hydrophobic ratio, hydrophobic moment</td><td>10</td></tr><tr><td>Motif</td><td>R-X-R, RW repeat, RGD,  $\mathrm { C - X _ { 3 ^ { - } } C _ { 3 } }$  RKKR</td><td>5</td></tr><tr><td>Density</td><td>Arg, Trp, Cys, Pro residue densities</td><td>4</td></tr><tr><td>Cleavageª</td><td>Basic-cleavage-pair count</td><td>1</td></tr><tr><td>Total</td><td></td><td>330</td></tr></table>

<sup>a</sup>The basic-cleavage-pair count is held out as its own single-column family rather than folded into PhysChem

## 2.4 Backbone

TabPFN is a prior-data fitted network: a transformer pre-trained on millions of synthetic tabular classification tasks that receives an entire labelled reference set as in-context samples and returns test-row posteriors in a single forward pass, with no gradient updates and no hyperparameter search on the target data [21, 22]. Because such networks approximate the Bayesian posterior predictive distribution [23], their probability outputs are calibrated by construction, which is what licenses the conditional analyses in Sections 2.7 and 2.10.

Two terms, fit and inference, are used in a TabPFN-specific sense. A fit performs no weight updates, it only loads the labelled reference peptides into the model context, so the “fitted” object is the stored context matrix plus fixed pre-trained weights. Inference is a single forward pass that reads a test peptide with that context and returns a posterior over labels. We therefore treat the 65,870 peptides of folds 1–2 as in-context samples, and reserve training for gradient-based procedures (TabPFN’s one-of synthetic pre-training and the deep-learning baselines). No gradients are computed on peptide data. Reported fit time is context ingestion (not optimisation), hence typically much smaller than inference time.

We used TabPFN v3 (tabpfn 8.2.0) with $n _ { \mathrm { e s t i m a t o r s } } = 1 6$ and seeds {42, 1665, 8914}, matching the seed protocol of the published baseline. Ensemble classifier chains use $n _ { \mathrm { e s t i m a t o r s } } = 8$ to make eight chains afordable. No hyperparameter was tuned on ESCAPE at any stage. Experiments ran on two NVIDIA RTX PRO 5000 Blackwell GPUs (48 GB each) under PyTorch 2.13.0+cu130, scikit-learn 1.7.2, NumPy 2.2.6, and Python 3.10.20.

## 2.5 Multi-label transforms

Five standard transforms were compared on an identical feature matrix, split, and backbone, difering only in how the multi-label target is factorized. A worked out example for the transforms is given in Supplementary Section S1.

Notation. Let $\mathbf { x } \in \mathbb { R } ^ { 3 3 0 }$ denote a peptide’s descriptor vector and $\mathbf { y } \in \{ 0 , 1 \} ^ { L }$ its label vector (with L = 5 for the benchmark comparison and $L = 4$ in the dependence analyses). Labels are ordered frequency-descending, y = Antibacterial, y = Antifungal, y = Antiviral, $y _ { 4 }$ = Antiparasitic, $y _ { 5 } =$ Antimicrobial. $\hat { P }$ denotes a TabPFN posterior probability.

Binary relevance (BR) assumes the labels are conditionally independent given the features and factorizes the joint as a product of marginals,

$$
\hat { P } _ { \mathrm { B R } } ( \mathbf { y } \mid \mathbf { x } ) = \prod _ { j = 1 } ^ { L } \hat { P } ( y _ { j } \mid \mathbf { x } ) ,\tag{1}
$$

fitting one classifier per label. It encodes no dependence and is the conditional-independence floor against which the others are measured. BR costs L inference passes.

Label powerset (LP) treats a peptide’s whole activity profile as one category rather than as five separate questions [25]. A peptide active against bacteria and fungi but not against viruses or parasites has the profile $( 1 , 1 , 0 , 0 , 1 )$ , the last entry being Antimicrobial, which is 1 whenever any of the four activities is. Because that parent is fixed by the other four, only $2 ^ { 4 } = 1 6$ of the $2 ^ { 5 }$ conceivable profiles can occur, from all-negative to active against all four. LP therefore solves a single 16-class problem and returns one probability per profile. A per-label probability is recovered by adding up the profiles that contain the label of interest:

$$
\hat { P } _ { \mathrm { L P } } ( y _ { j } = 1 \mid \mathbf { x } ) = \sum _ { c \in \mathcal { C } : c _ { j } = 1 } \hat { P } ( c \mid \mathbf { x } ) ,\tag{2}
$$

where C is the set of profiles seen in context and $c _ { j }$ the jth entry of profile $c ,$ so the sum runs over every profile in which label $j$ is present. LP can represent any pattern of co-occurrence among the profiles it has seen, at the cost of assigning zero probability to a profile absent from context. LP only cost one inference pass.

Probabilistic classifier chains (PCC) predict the labels in a fixed order, each one conditioned on the answers already given, i.e., is the peptide antibacterial, then, given that answer, is it antifungal, and so on. Multiplying these conditionals gives the probability of one complete profile, and PCC evaluates all $2 ^ { 5 } = 3 2$ profiles rather than following a single path [26]:

$$
\hat { P } _ { \mathrm { P C C } } ( \mathbf { y } \mid \mathbf { x } ) = \prod _ { j = 1 } ^ { L } \hat { P } ( y _ { j } \mid \mathbf { x } , y _ { 1 } , \ldots , y _ { j - 1 } ) , \qquad \hat { P } ( y _ { j } = 1 \mid \mathbf { x } ) = \sum _ { \mathbf { y } : y _ { j } = 1 } \hat { P } ( \mathbf { y } \mid \mathbf { x } ) .\tag{3}
$$

As in LP, a per-label probability is a sum over the profiles containing that label. Enumerating 32 profiles makes PCC the most expensive transform here, but it yields the full joint posterior the conditional analyses require. Unlike LP it is not confined to the 16 profiles the Antimicrobial rule permits, which is why it can return incoherent ones (Section 3.5). Cost: L fits, $2 ^ { L }$ inference passes, i.e., 4,116 s against 128 s for LP.

Classifier chains (CC) keep the same label order as PCC, but they follow only one trajectory through the chain. After predicting $y _ { 1 }$ , the model feeds that prediction into the classifier for $y _ { 2 } .$ then feeds the prediction for $y _ { 2 }$ into the classifier for $y _ { 3 } .$ , and so on [27]. This avoids enumerating all profiles, making CC much cheaper than $\mathrm { P C C } ,$ but it also means mistakes made early can cascade downstream.

Formally,

$$
\hat { P } _ { \mathrm { C C } } ( \mathbf { y } \mid \mathbf { x } ) = \prod _ { j } \hat { P } ( y _ { j } \mid \mathbf { x } , y _ { < j } ) ,\tag{4}
$$

where $y _ { < j }$ are the earlier labels, given as ground truth in the in-context samples and replaced by the chain’s own predictions at inference. How the earlier predictions are encoded is therefore important. Hard context appends the binarized indicator $\mathbf { 1 } [ \hat { p } _ { < j } > \tau ] \in \{ 0 , 1 \}$ , matching the $0 / 1$ encoding of the in-context labels, while soft context appends the raw probability $\hat { p } _ { < j } \in [ 0 , 1 ]$

Ensemble classifier chains (ECC) average the per-label probabilities of eight chains under independently sampled label orders, so that no single arbitrary ordering decides the result. ECC run at $n _ { \mathrm { e s t i m a t o r s } } = 8$ to make eight chains afordable.

## 2.6 ESCAPE baseline re-run and matched-protocol evaluation

Published per-label numbers for the ESCAPE baseline [14] are point estimates from the original authors’ runs, so comparisons to them are unpaired. For a paired comparison, we re-ran the released ESCAPE baseline checkpoints (Best\_model\_Fold1.pth, Best\_model\_Fold2.pth) on our hardware without any retraining. Structural maps are available for 5,495 of the 16,489 test peptides (33.3%), so paired analyses are restricted to this subset.

Matched-protocol evaluation: The ESCAPE baseline averages two checkpoints trained on opposite folds, whereas our transforms take both folds as in-context samples. To match protocols, we supplied each fold separately as context (per seed) and averaged the two resulting probability matrices, mirroring the baseline. The gap to the full-context experiment isolates the efect of the additional in-context samples.

Structural-branch ablation: To test whether the ESCAPE baseline’s structural modality contributes at inference, we held the sequence input fixed and replaced only the structural input with (i) an all-zero image, (ii) uniform noise, and (iii) another peptide’s real map, which preserves input statistics while destroying the association between a peptide and its own structure. Changes were quantified both as per-label average precision and as the maximum and mean absolute change in predicted probability over every peptide × label pair, against the same checkpoint’s real-map run.

## 2.7 Quantifying label dependence

Two analyses establish whether conditional dependence between activities exists and what it is worth. Both are restricted to the four real activities, excluding the deterministic Antimicrobial parent, whose inclusion inflates apparent dependence two-to-four fold.

An oracle conditional-predictability ceiling scores each activity with its three true sibling labels appended to the features, and compares that against features alone. This upper-bounds the gain available to any method that exploits sibling labels. Recovery is reported as (transform − BR)/(oracle − BR) on the mAP-4 scale. This measures what dependence is worth for per-label ranking.

A joint-versus-product held-out likelihood comparison asks the complementary question, whether dependence exists at all independently of any ranking metric. We compare the mean held-out log-likelihood of the true four-activity test vectors under the BR product of marginals against the PCC joint, computed in log space with log-sum-exp normalization, in nats per peptide. Log-likelihood is a strictly proper scoring rule, so a joint-model advantage on it constitutes exploitable dependence by construction, whether or not that dependence is recoverable as average precision.

## 2.8 Homology stratification

Test peptides were searched against the in-context set (folds 1–2) with MMseqs2 [28] 18.8cc5c easy-search (-s 7.5 -e 10000 –max-seqs 4000 –alignment-mode 3) and each test peptide labelled with the maximum sequence identity among hits with query coverage ≥ 0.5. Peptides with no qualifying hit were assigned zero. There are no exactly duplicated sequences within or across splits.

## 2.9 Feature-group attribution

Three analyses answer diferent questions and are expected to disagree where features are redundant. Group-only refits on each family alone and measures standalone suficiency. Leave-one-group-out refits without each family and measures what is lost when it is unavailable from the start. Permutation shufles the family’s columns in the test matrix and re-infers, measuring the model’s reliance on it. All arms run on a stratified 4,000-peptide test subsample drawn once and shared across three analyses, retaining all 77 antiparasitic positives. Label prevalence in the subsample is consequently higher than in the full test ${ \mathrm { s e t } } ,$ so its absolute values are not comparable to Section 3.1 and are used only for within-experiment contrasts.

## 2.10 Positive-unlabelled assay prioritization

Curated activity repositories record what was found positive. An activity a peptide was never assayed for is unlabelled rather than known-negative [29, 30]. We therefore treated confirmed positives as reliable and all zeros as unlabelled, and scored the task a screening campaign actually faces: given a peptide already confirmed positive for m activities, rank the remaining activities by which to assay next.

For each test peptide with at least m confirmed positives and at least one remaining candidate, the m highest-prevalence positives were revealed and every unobserved activity scored by the joint posterior $\hat { P } ( a = 1 \ |$ x, observed positives). Rankers compared were BR (which cannot use the revealed positives and is therefore the do-nothing baseline) and the LP and PCC joint posteriors. Performance is reported as Hit@1, mean reciprocal rank, and a pooled positive-unlabelled AP, which is a lower bound because the unlabelled set contains an unknown number of true positives. As a negative control the conditioning was permuted across peptides, supplying the same quantity of label information while severing its link to the specific peptide. This control is a single donor assignment per m, reused across seeds, not a distribution over permutations. It is therefore a point comparison and yields no permutation p-value. Hit@1 is additionally reported decomposed by which activity is held out, because the pooled figure is dominated by the commonest target.

## 3 Results

## 3.1 Improved average precision across all activity labels

Table 3 places the five transforms against every published result on the ESCAPE benchmark. The label powerset transform reaches a five-label macro average precision of $7 7 . 7 9 \pm 0 . 2 3$ and the probabilistic classifier chain $7 7 . 7 6 \pm 0 . 1 2$ , against 72.12 for the previously best-performing method, a margin of 5.67 and 5.64 percentage points respectively. Binary relevance, which encodes no dependence between activities at all, reaches $7 7 . 0 1 \pm 0 . 1 7$ and is already ahead of every published method (Figure 1a). All five transforms use the same 65,870 peptides as in-context samples and involve no gradient training (Section 2.4). Supplementary Section S4 records what each published method’s own publication set out to predict, since several were binary or single-activity, classifiers were adapted to the five-label output by ESCAPE’s benchmark authors.

The per-label picture is more informative than the macro. Among published methods no single architecture leads on more than three labels. PEP-Net leads on Antibacterial, Antifungal, and Antimicrobial, AVP-IFT on Antiviral, and ESCAPE on Antiparasitic only, despite ranking first overall. Every transform reported here exceeds the best published value on Antibacterial, Antifungal, Antiviral, and Antimicrobial, and PCC additionally matches it on Antiparasitic (37.68 against 37.6). PCC is therefore the first model to match or exceed the best published average precision on all

Table 3: Average precision (%) on the ESCAPE test set $( n ~ = ~ 1 6 , 4 8 9 )$ , averaged over seeds {42, 1665, 8914}. Per-label published values are as reported by the benchmark authors. Bold marks the best value in each column. Powered macro averages Antibacterial, Antifungal, and Antiviral only. mAP-5 and Powered are recomputed here from the per-label columns so that every row is on the same footing. Per-seed values in Supplementary Section S2.
<table><tr><td>Method</td><td>Antibact.</td><td>Antifung.</td><td>Antiviral</td><td>Antiparas.</td><td>Antimicrob.</td><td>mAP-5</td><td>Powered</td></tr><tr><td>Published</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AMPs-Net [11]</td><td>82.5</td><td>53.1</td><td>51.2</td><td>5.3</td><td>82.1</td><td>54.8</td><td>62.3</td></tr><tr><td>TransImbAMP [12]</td><td>92.5</td><td>56.3</td><td>65.0</td><td>16.7</td><td>94.0</td><td>64.9</td><td>71.3</td></tr><tr><td>AMP-BERT [9]</td><td>92.3</td><td>61.5</td><td>65.9</td><td>21.4</td><td>93.6</td><td>66.9</td><td>73.2</td></tr><tr><td>PEP-Net [31]</td><td>95.2</td><td>72.6</td><td>61.2</td><td>16.2</td><td>96.7</td><td>68.4</td><td>76.3</td></tr><tr><td>amPEPpy [7]</td><td>93.9</td><td>62.2</td><td>67.7</td><td>23.8</td><td>95.2</td><td>68.6</td><td>74.6</td></tr><tr><td>AVP-IFT [32]</td><td>94.3</td><td>63.3</td><td>71.1</td><td>20.0</td><td>95.5</td><td>68.8</td><td>76.2</td></tr><tr><td>AMPlify [8]</td><td>94.0</td><td>68.3</td><td>66.1</td><td>27.7</td><td>95.3</td><td>70.3</td><td>76.1</td></tr><tr><td>ESCAPE [14]</td><td>94.2</td><td>63.4</td><td>69.8</td><td>37.6</td><td>95.6</td><td>72.12</td><td>75.8</td></tr><tr><td>This work</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CC</td><td>97.27</td><td>73.69</td><td>78.93</td><td>28.93</td><td>97.53</td><td>75.27</td><td>83.30</td></tr><tr><td>ECC</td><td>97.21</td><td>74.73</td><td>79.85</td><td>34.84</td><td>97.81</td><td>76.89</td><td>83.93</td></tr><tr><td>BR</td><td>97.27</td><td>74.98</td><td>79.95</td><td>34.76</td><td>98.10</td><td>77.01</td><td>84.07</td></tr><tr><td>PCC</td><td>97.27</td><td>75.37</td><td>80.41</td><td>37.68</td><td>98.06</td><td>77.76</td><td>84.35</td></tr><tr><td>LP</td><td>97.51</td><td>75.65</td><td>81.29</td><td>36.41</td><td>98.11</td><td>77.79</td><td>84.82</td></tr></table>

![](images/5b2d538df9afd8a713bc9cf41a33f392cc1be0e9b13cc8c6ee38bb78c0822798.jpg)

![](images/7d2dfdd3fafe8d9f526995136a9b70e43336ab51fe2f1a98ff01ae888152588b.jpg)  
Figure 1: Each transform leads the published benchmarks on both macro scales. (a) Five-label macro average precision, (b) The macro restricted to the three adequately powered activities (antibacterial, antifungal, antiviral). Values are recomputed from the per-label average precisions in Table 3.

Antibacterial (3,187 test positives) best published: PEP-Net 95.2

![](images/6dd42db003156a5dbf15c4cb8795278ac637e6eecca466b1508b4cee052baac6.jpg)

Antiparasitic (77 test positives) best published: ESCAPE 37.6  
![](images/5ed2a79aace1acd903b49d7306f13d678c88e9028c8155a7d64d15bb804d6bb0.jpg)

![](images/20803789c91e271eee58a8f3ac239b82d42a374525f75c64e7de55349fd40854.jpg)  
Antimicrobial (4,235 test positives) best published: PEP-Net 96.7

![](images/475149d12f288f3ded0515eed1f9aeb78c93b79e89a16efb6990fdad31fc8fa2.jpg)

![](images/adca5ff0282982d413646277246542a0b8ad97afb2bc2a65f8f5f4749b74bb0c.jpg)  
Figure 2: Per-label average precision, all eight published methods against all five proposed transforms. Each panel is one activity label, annotated with its test-positive count. The dashed line is the best published value for that label. Every transform exceeds the best published value on antibacterial, antifungal, antiviral and antimicrobial activity. Error bars are 95% bootstrap percentile intervals, shown for LP in every panel and additionally for PCC on Antiparasitic. They are narrower than the plotting symbol on Antibacterial and Antimicrobial. Note the difering y-axis range in each panel.

Table 4: Computational cost of each transform on the full benchmark, single GPU, $n _ { \mathrm { e s t i m a t o r s } } = 1 6$ (8 for ECC). Times exclude feature extraction, which is shared across all transforms. A fit is the ingestion of the 65,870 in-context samples and involves no gradient update; an inference pass is the forward pass that returns posteriors for the test peptides (Section 2.4). Counts are given as a function of the number of labels $L = 5$ . Times are means over seeds {42, 1665, 8914}
<table><tr><td>Transform</td><td>Fits</td><td>Inference passes</td><td>Fit (s)</td><td>Inference (s)</td><td>mAP-5</td></tr><tr><td>LP</td><td>1</td><td>1</td><td>18</td><td>128</td><td>77.79</td></tr><tr><td>BR</td><td>L</td><td>L</td><td>85</td><td>646</td><td>77.01</td></tr><tr><td>CC</td><td>L</td><td>L</td><td>124</td><td>907</td><td>75.27</td></tr><tr><td>ECC</td><td>8L</td><td>8L</td><td>255</td><td>3,791</td><td>76.89</td></tr><tr><td>PCC</td><td>L</td><td>2L</td><td>82</td><td>4,116</td><td>77.76</td></tr></table>

five labels simultaneously. The Antiparasitic comparison should be read with its power in mind: with 77 test positives, PCC’s 95% interval is [26.26, 48.50], so the correct statement is parity rather than superiority. Because published per-label values are point estimates from the original authors runs, they support only the question of whether each falls inside our 95% confidence interval. On Antibacterial, Antifungal, Antiviral, and Antimicrobial, every published value lies strictly below the lower bound of every transform’s interval. On Antiparasitic all published values lie inside our intervals for all five transforms (Figure 2). Building on the per-label results, we focus on the “powered” macro (averaging only adequately powered labels) because Antiparasitic is so rare (77 positives) that its wide uncertainty can dominate the equal-weight five-label metric (Figure 1b).

These results were obtained by forward passes through the pre-trained TabPFN network alone, with no gradient training and no hyperparameter search. The labelled peptides supplied as in-context samples are read in the same forward pass as the test query and directly yield its predicted labels. Because the transforms difer only in how the label is presented, their costs difer by orders of magnitude: LP is both the most accurate transform and the cheapest, 5.0 times cheaper at inference than BR, the next-cheapest, and 32 times cheaper than PCC (Table 4).

## 3.2 Separating Data-Size Efects from Model Gains

Our transforms take both non-test folds (65,870 peptides) as in-context samples, whereas ESCAPE, the previous state of the art, trains on one fold (≈32,900 peptides) at a time and averages the test metric. A margin measured that way therefore confounds model quality with twice as much labelled data. One observation motivated testing this directly: halving the number of in-context samples (a random half of folds 1–2) costs 5.01 points of mAP-5 for BR (4.71 for LP, 4.43 for CC; Supplementary Section S3), the same magnitude as BR’s full-context margin of +5.57 over the reproduced baseline. That subset was drawn at random across both folds, however, and is therefore not like-for-like with the baseline’s single-fold protocol, which the fold-matched experiment below supplies.

Running the released ESCAPE checkpoints over the 5,495 test peptides for which structural maps were released gives macro AP 72.18 against the published 72.12, a diference of 0.06 points, with every per-label value within 1.7 points of its published counterpart. The released checkpoints and evaluation code therefore reproduce the publication closely, and the comparison below (Table 5) is against a faithfully reproduced baseline.

Under the matched protocol LP retains a significant advantage of $+ 3 . 6 9 \ [ + 1 . 0 4 , + 6 . 4 6 ]$ points, which also holds on $\mathrm { m A P - 4 \ ( + 4 . 0 9 \ [ + 0 . 8 1 , + 7 . 5 3 ] ) }$ . BR retains +2.86, significant on mAP-5 but marginally so, and not significant on mAP-4. CC is indistinguishable from the baseline on the macro.

Table 5: Paired comparison against ESCAPE on the 5,495 test peptides with released structural maps, 2,000 shared bootstrap resamples. Fold-matched supplies one fold at a time as in-context samples and averages the two probability matrices, mirroring the ESCAPE’s own protocol. All entries are diferences in mAP-5, in percentage points.
<table><tr><td>Transform</td><td>∆ full-data</td><td>∆ fold-matched</td><td>95% CI (matched)</td><td>Significant</td></tr><tr><td>LP</td><td>+6.08</td><td>+3.69</td><td> $[ + 1 . 0 4 , + 6 . 4 6 ]$ </td><td>yes</td></tr><tr><td>BR</td><td>+5.57</td><td>+2.86</td><td> $[ + 0 . 0 4 , + 5 . 8 7 ]$ </td><td>yes (marginal)</td></tr><tr><td>CC</td><td>+3.32</td><td>-0.13</td><td> $[ - 3 . 2 1 , + 3 . 6 0 ]$ </td><td>no</td></tr><tr><td>PCC</td><td>+6.01</td><td>+3.26</td><td> $[ + 0 . 6 5 , + 6 . 0 3 ]$ </td><td>yes</td></tr></table>

PCC retains $+ 3 . 2 6 ~ [ + 0 . 6 5 , + 6 . 0 3 ]$ on mAP-5 and $+ 3 . 5 6 \ [ + 0 . 2 9 , + 7 . 0 0 ]$ on mAP-4. Direct paired measurement of the confound (full-data minus fold-matched, on identical test rows, $n = 1 6 { , } 4 8 9 )$ gives $+ 2 . 4 0 \ [ + 1 . 8 0 , + 3 . 0 2 ]$ for LP, $+ 3 . 0 8 \ [ + 2 . 1 6 , + 3 . 8 5 ]$ for BR, $+ 2 . 9 6 \ [ + 1 . 8 9 , + 3 . 8 8 ]$ for CC, and $+ 3 . 0 5 \ [ + 2 . 1 5 , + 3 . 8 5 ]$ for PCC; all four intervals exclude zero. Roughly half of the full-data margin is therefore attributable to the number of in-context samples, and the remainder to the model.

One secondary observation bears on architecture choice. The baseline gains +5.86 points of mAP-5 from averaging its two checkpoints, more than twice what any of our transforms gain from the same operation (LP +2.32, BR +2.27, CC +1.61), and its individual checkpoints are weak in isolation (65.03 and 67.62 against 72.18 ensembled). For an in-context learner the reverse holds: one model on 65,870 rows outperforms two on ≈32,900 each (LP 77.79 against 75.40), which is the expected behaviour when additional labelled rows are additional evidence at inference rather than additional gradient steps.

## 3.3 Remote-homologue peptides show strongest gains

A margin obtained on a benchmark in which a third of the test set is nearly identical to sequences the model has seen in context invites the objection that it reflects memorization. The predefined fold split does place 5,342 test peptides (32.4%) out of $1 6 { , } 4 8 9 \mathrm { ~ a t } \geq 9 0 \%$ identity to an in-context sequence.

Paired per-peptide comparison of 5,495 test peptides against the previous state of the art, by identity band, shows the margin growing as similarity falls. Below 30% identity it is +9.65 points (BR), +11.23 (LP), and +10.03 (PCC), against +3.8 (LP) and several non-significant values in the $\geq 9 0 \%$ band (Table 6). Pooled over the whole covered subset the paired margins are +6.08 $[ + 3 . 1 3 , + 9 . 1 0 ]$ for $\mathrm { L P , ~ + 6 . 0 1 ~ [ + 3 . 7 8 , + 8 . 1 5 ] }$ for PCC, and $+ 5 . 5 7 \ [ + 3 . 1 5 , + 7 . 8 6 ]$ for BR, so the remote-homologue margin is close to twice the pooled one. The advantage is therefore largest exactly where the benchmark is hardest and where a screening campaign would actually operate.

## 3.4 Sequence-only information is suficient

To determine which parts of our 330-dimensional descriptor set actually carry predictive signal (and thus whether the gains come from rich representations or from coarse composition), we ran an ablation-style attribution study on a stratified 4,000-peptide random subsample retaining all 77 antiparasitic positives (baseline mAP-5 80.41 on this subsample; its absolute values are not comparable to Section 3.1). Because rerunning TabPFN is computationally expensive, we restrict these refits to this smaller test subset. Table 7 summarizes the three attribution experiments (group-only refits, leave-one-family-out refits, and test-time permutation within each family).

Table 6: Homology stratification of the predefined test split. Margin is the paired per-peptide diference in mAP-5 against the previous state of the art, in percentage points, restricted within each band to labels with at least five positives in that band.
<table><tr><td rowspan="2">Identity band</td><td rowspan="2">Test peptides n (% of test)</td><td colspan="3">Margin over previous SOTA (points)</td></tr><tr><td>BR</td><td>LP</td><td>PCC</td></tr><tr><td>&lt; 30%</td><td>3,708 (22.5%)</td><td>+9.65</td><td>+11.23</td><td>+10.03</td></tr><tr><td>≥ 90%</td><td>5,342 (32.4%)</td><td>n.s.</td><td>+3.8</td><td>n.s.</td></tr><tr><td>All (paired)</td><td>5,495 (100%)</td><td>+5.57</td><td>+6.08</td><td>+6.01</td></tr></table>

Refitting on each descriptor family alone shows that the signal is broadly distributed and largely redundant. Ten global physicochemical scalars reach 73.52, recovering 91.4% of full-feature performance; 20 amino-acid-composition features reach 77.06 (95.8%); 15 AAindex components reach 76.05 (94.6%); the 147 CTD descriptors reach 77.30 (96.1%); and the 128-dimensional ESM-2 protein language model block [17] reaches 79.19 (98.5%), within 1.22 points of the full model. A learned embedding and a hand-crafted descriptor set are thus near-substitutable here. Only the three smallest families fall away: residue densities reach 51.12 and the motif and cleavage features, which are near-binary indicators, reach 16.26 and 16.48.

Leave-one-family-out shows that no single feature family is essential. Removing ESM gives the biggest drop (−0.74 points), followed by CTD (−0.67) and every other family changes mAP-5 by at most 0.21. Dropping motif (+0.51) or cleavage (+0.12) even gives a tiny increase. Permutation importance, however, makes CTD and ESM look much more important (−45.19 and −19.17 points). This happens because permuting a large, highly correlated block creates unrealistic (outof-distribution) inputs, so the score drop can be exaggerated. For CTD, the permutation and refit results difer by a factor of 67, so we trust the refit results more. The small gains from removing motif and cleavage are within noise for a 4,000-peptide subsample, and should not be taken as evidence that these features are harmful.

This redundancy is the most direct explanation of why a simple model sufices on this benchmark. The accessible signal is coarse composition, and it is close to saturated by the time a few dozen descriptors have been supplied.

The same conclusion is reached independently from the released multimodal checkpoints. Holding the sequence input fixed and replacing only the structural input with an all-zero image, with uniform noise, or with another peptide’s map moves ensemble macro AP by at most 0.056 points across all 5,495 covered peptides. Essentially, all of that comes from Antiparasitic, whose own per-label spread is 0.298 points and which therefore contributes up to 0.060 points to the macro. The other four labels contribute at most 0.005 points each. For the Fold-2 checkpoint, deleting the structural input entirely moves none of the predicted probability by more than $2 . 1 \times 1 0 ^ { - 5 }$ . We emphasize that an inference-level measurement does not establish what the structural branch contributed during training, where it may well have acted as a useful regularizer, nor does it speak to architectures other than the one examined here. Taken with the feature-attribution results, it indicates that sequence-derived descriptors are suficient for this task at present.

## 3.5 Dependence helps mainly for scarce activities

In this section we test whether the four activity labels are statistically dependent in a way that a model can use, and we measure (i) how much better a joint model scores complete multi-label outcomes than an independent-label baseline, and (ii) how much that dependence can improve

Table 7: Feature-group attribution on the stratified 4,000-peptide subsample (full-feature mAP-5 80.41; absolute values are not comparable to Table 3). Group-only is a refit on that family alone, leave-one-out is a refit without it, reported as the loss relative to the full model, and permutation shufles the family’s columns at test time and re-infers with the same fitted model. Negative leave-one-out and permutation values denote a loss relative to the full model and positive values an improvement. All values in percentage points. The cleavage feature is a single column held out from PhysChem (Table 2).
<table><tr><td>Family</td><td>n cols</td><td>Group-only</td><td>% of full</td><td>Leave-one-out</td><td>Permutation</td></tr><tr><td>ESM</td><td>128</td><td>79.19</td><td>98.5</td><td>-0.74</td><td>-19.17</td></tr><tr><td>CTD</td><td>147</td><td>77.30</td><td>96.1</td><td>-0.67</td><td>-45.19</td></tr><tr><td>AAC</td><td>20</td><td>77.06</td><td>95.8</td><td>-0.02</td><td>-2.43</td></tr><tr><td>AAIndex</td><td>15</td><td>76.05</td><td>94.6</td><td>-0.21</td><td>-1.70</td></tr><tr><td>PhysChem</td><td>10</td><td>73.52</td><td>91.4</td><td>-0.07</td><td>-6.45</td></tr><tr><td>Density</td><td>4</td><td>51.12</td><td>63.6</td><td>-0.06</td><td>-0.52</td></tr><tr><td>Cleavage</td><td>1</td><td>16.48</td><td>20.5</td><td>+0.12</td><td>-0.04</td></tr><tr><td>Motif</td><td>5</td><td>16.26</td><td>20.2</td><td>+0.51</td><td>-0.03</td></tr><tr><td>All</td><td>330</td><td>80.41</td><td>100.0</td><td></td><td></td></tr></table>

ranking each activity.

Dependence is present and measurable. On held-out test data, the joint posterior gives the true four-activity label vectors a mean log-likelihood of −0.28266, compared with −0.30271 for the product of binary-relevance (independent) marginals, a gain of +0.0200 nats per peptide. This is dependence that can be exploited by construction.

For per-label ranking, however, the upside is limited. An oracle that is told each peptide’s three true sibling activities reaches mAP-4 76.73, while the binary-relevance floor is 71.74, so the total headroom available to any method that uses sibling labels is only 4.99 points. LP captures 19.5% of this headroom and PCC 18.9%. CC captures none in its default form (CC-soft), but this is an encoding mismatch rather than a failure of the factorization. If we supply hard-thresholded labels as chain context (matching the 0/1 encoding of the in-context labels), CC-hard rises from 69.70 to 72.43 and recovers 13.7% of the ceiling (Figure 3a).

The benefit is not evenly spread across activities, and that distribution is the main practical takeaway. The oracle lift is +0.82 points for Antibacterial, +4.47 for Antiviral, +5.02 for Antifungal, and +9.66 for Antiparasitic (Figure 3b). Dependence tends to help most where marginal evidence is scarcest (antifungal and antiviral are efectively tied). Consistent with this, PCC’s only per-label gain over BR that is individually resolvable is on Antiparasitic (+2.91, [+0.81, +5.29]), and this is what brings PCC to parity on the one label where the previous state of the art led (Table 3).

One property of LP is worth noting for practitioners independent of accuracy. LP exactly respects the deterministic parent relation, i.e., it produced zero violations of the Antimicrobial OR-gate in 246,000 predictions across all seeds and both splits, versus violation rates of 11.3% (PCC), 34.8% (BR), 39.5% (ECC), and 44.9% (CC). A peptide called antifungal but not antimicrobial is not interpretable to a screening biologist (Table 8).

## 3.6 Ranking Activities from Partial Positive Evidence

Curated repositories record confirmed positives and an activity a peptide was never assayed for is unlabelled rather than known-negative. The question a screening campaign actually asks is therefore not "is this peptide antifungal?" but "given what we have already confirmed, which assay should we

(a) Headroom from label dependence is 4.99 points  
![](images/689ff1bf6af1950bea7bd3ba1755aa225abe09fa2c0ef533dae9250121dc980e.jpg)

(b) Dependence is worth most where evidence is scarcest  
![](images/61f2f3fdf76a590e44ee1cd49bd4f3e04996eae42951b76de6f0a07747c8d8bd.jpg)  
Figure 3: Conditional dependence between activities is real but bounded. (a) An oracle supplied with each peptide’s three true sibling activities gains 4.99 points of mAP-4 over the binary-relevance floor; the best transforms recover about a fifth of that. CC recovers nothing under soft chain context and +13.7% once the context encoding matches the 0/1 encoding of the in-context labels. Percentages are the fraction of the oracle headroom recovered. (b) The oracle lift per activity, against that activity’s test-set prevalence. Dependence is worth most on the rarest activities; antifungal and antiviral, whose prevalences difer by less than a factor of two, are efectively tied.

Table 8: OR-gate violations is the fraction of predictions at the honest threshold in which the predicted Antimicrobial call contradicts $\vee _ { j } y _ { j }$ over the four activities, across 246,000 predictions spanning all seeds and both splits.
<table><tr><td>Transform</td><td>mAP-5 (points)</td><td>OR-gate violations (%)</td></tr><tr><td>LP</td><td>77.79</td><td>0.00</td></tr><tr><td>PCC</td><td>77.76</td><td>11.33</td></tr><tr><td>BR</td><td>77.01</td><td>34.75</td></tr><tr><td>ECC</td><td>76.89</td><td>39.51</td></tr><tr><td>CC</td><td>75.27</td><td>44.85</td></tr></table>

run next?"

Scored that way the model is directly useful. The task is defined by concealing one confirmed activity and asking the model to rank it against the activities the peptide has not been confirmed for. Say, m is the number of other confirmed activities revealed to the model before it ranks. A peptide is therefore eligible only if it carries at least m+1 confirmed activities (m to reveal and one to conceal), so eligibility shrinks as m grows. 4,235 of the 16,489 test peptides carry at least one confirmed activity, 1,035 carry at least two, and 250 carry at least three. The remaining 12,254 test peptides have no confirmed activity at all, so there is nothing to conceal and no ranking to make.

At m = 0, with nothing revealed, the concealed activity is ranked first for 70.89% of the 4,235 eligible peptides against a chance rate of 25%, one of four candidate activities (mean reciprocal rank 0.840 on the 0–1 scale). Because nothing has been revealed, no ranker can exploit conditioning information at m = 0, and BR, LP and PCC accordingly agree to within 0.16 points; this figure measures what the sequence descriptors alone support, not what the joint model adds.

Revealing one confirmed positive raises the PCC joint posterior to 74.65% [71.95, 77.26] against a 33.3% chance rate, and revealing two raises it to 89.73% against 50% (Table 9). A permuted-label control, in which a peptide is conditioned on another peptide’s confirmed activity, drops to 61.45% at m = 1, indicating that the gain comes from the peptide-specific co-occurrence structure rather than from the generic presence of an additional input.

The joint posterior only marginally improves on binary relevance (and not consistently across m), consistent with the bounded ceiling in Section 3.5. The permuted control separates clearly at m = 1 and, on pooled positive-unlabelled average precision, remains well below the true-conditioning setting at both m = 1 (92.58 vs 68.88) and m = 2 (93.78 vs 87.59).

The pooled Hit@1 also conceals a strong dependence on which activity is concealed. At $m = 0$ the LP joint posterior ranks the held-out activity first for 97.77% of peptides when that activity is antibacterial $( n = 2 , 1 5 7 )$ but 64.11% when antiviral (n = 938), 26.59% when antifungal $( n = 1 , 0 6 3 )$ and 3.03% when antiparasitic (n = 77). The pooled figure is therefore dominated by the commonest target, and is the same averaging problem described in Section 3.1 in a diferent guise. The practical point is that for the three well-populated activities the ranking is far above chance in the regime where screening budgets are actually allocated.

## 4 Discussion

Because a single antimicrobial peptide can be active against multiple pathogen classes, screening needs to predict an activity profile, not a single label. Most work tackles this by building customized multi-output models (e.g., GNNs [11], imbalance-aware transformers [12], multi-task CNNs [13], or the ESCAPE multimodal transformer [14]). A smaller line instead keeps a single-label learner and changes the target via problem transformations: binary relevance, label powerset, and (deterministic/ensembled/probabilistic) classifier chains [25–27]. Although these transforms are well studied in general [33–35], they have not been compared head-to-head on a fixed multi-activity AMP benchmark with representation and split held constant. We take that controlled route: one 330-dimensional interpretable descriptor matrix, one protocol and split, five transforms, and TabPFN as a shared backbone. Because TabPFN returns its posterior in a single forward pass over the in-context samples, each arm runs in seconds with no gradient training, which is what makes the five-way comparison practical at all.

Across every metric we report, label powerset (LP) transform is the most accurate (77.79 mAP-5, ahead of every published method and the other four transforms). It is also the cheapest by a wide margin (one fit and one inference pass: 18 s and 128 s; ×5 faster at inference than the next-cheapest transform and ×32 faster than probabilistic classifier chains). Finally, it recovers the largest share of the oracle ceiling on label dependence (19.5% vs 18.9% for PCC and 13.7% for the best chain), so the speed is not bought by discarding structure. LP flips the usual approach, rather than explicitly modelling label relations, it collapses each peptide’s full activity profile into a single class, so dependence is encoded as class identity. This has two practical consequences that macro metrics obscure. First, any co-occurrence pattern observed among the in-context samples is represented exactly, which is why the simplest arm leads the oracle-recovery comparison. Second, incoherent predictions are impossible because Antimicrobial is the logical OR of the four activities.

Table 9: Assay prioritization: rank the activities a peptide has not yet been assayed for, given m already-confirmed positives. Hit@1 and pooled positive-unlabelled average precision (PU-AP) are in percentage points; mean reciprocal rank (MRR) is dimensionless and reported on the 0–1 scale. Chance rises with m because fewer candidate activities remain. BR cannot use the revealed positives and is the do-nothing baseline; PCC-perm is the permuted-label control, which is undefined at m = 0 because nothing has been revealed.
<table><tr><td>Metric</td><td>m (n eligible)</td><td>Chance</td><td>BR</td><td>LP-joint</td><td>PCC-joint</td><td>PCC-perm</td></tr><tr><td rowspan="3">Hit@1 (%)</td><td>0 (4,235)</td><td>25.0</td><td>70.89</td><td>70.73</td><td>70.80</td><td></td></tr><tr><td>1 (1,035)</td><td>33.3</td><td>73.33</td><td>71.56</td><td>74.65</td><td>61.45</td></tr><tr><td>2 (250)</td><td>50.0</td><td>90.00</td><td>89.87</td><td>89.73</td><td>88.40</td></tr><tr><td rowspan="3">MRR (0−1)</td><td>0</td><td></td><td>0.8395</td><td>0.8383</td><td>0.8389</td><td></td></tr><tr><td>1</td><td></td><td>0.8601</td><td>0.8513</td><td>0.8669</td><td>0.8002</td></tr><tr><td>2</td><td></td><td>0.9500</td><td>0.9493</td><td>0.9487</td><td>0.9420</td></tr><tr><td>PU-AP (%)</td><td>0</td><td></td><td>92.75</td><td>93.17</td><td>92.86</td><td></td></tr><tr><td></td><td>1</td><td></td><td>91.36</td><td>92.78</td><td>92.58</td><td>68.88</td></tr><tr><td></td><td>2</td><td></td><td>92.65</td><td>93.85</td><td>93.78</td><td>87.59</td></tr></table>

95% bootstrap intervals over eligible peptides for the Hit@1 rows: at m = 1, BR [70.66, 76.01], LP-joint [68.73, 74.30], PCC-joint [71.95, 77.26], PCC-perm [58.42, 64.35]. The LP permuted control gives 58.90 at m = 1 and 88.27 at m = 2.

More peptide predictors now ship a predicted 3D structure alongside sequence [14–16], on the reasonable premise that membrane disruption is fundamentally structural. Our results indicate that, for the released checkpoints examined here, that modality contributes little at the point of prediction. First, a sequence-only descriptor set is already state of the art, and its signal is highly redundant: ten global physicochemical scalars recover 91.4% of full-feature performance, and dropping any one feature family costs at most 0.74 mAP. Second, for released ESCAPE checkpoints, ablating the structural branch at inference (zero/noise/swapped maps with sequence fixed) changes ensemble macro average precision by at most 0.056, and for one checkpoint removing structure entirely changes no predicted probability by more than $2 . 1 \times 1 0 ^ { - 5 }$ . As distributed, these checkpoints behave as sequence-only at inference, which does not preclude the structural branch having helped during optimisation. The practical suggestion is procedural rather than critical: inference-time ablations of this kind are inexpensive, and reporting one alongside a multimodal architecture would settle the question directly, which matters most when a structure must be predicted for every screened candidate.

For experimental use, the more natural question is not per-label classification but which assay to run next. In curated repositories, a missing test is recorded as 0 but is better treated as unknown. With that reading, a model that outputs a posterior over full activity profiles can prioritize hidden activities. In our setting, the concealed activity is ranked first for 70.89% of eligible peptides (25% by chance), improving as more positives are revealed. The caveats are that joint inference adds little beyond calibrated marginals, and performance depends strongly on which activity is concealed (97.77% antibacterial vs 3.03% antiparasitic). To make this a laboratory tool would require prospective “freeze-and-test” validation, cost-aware ranking, and an explicit positive–unlabelled treatment [29, 30]. The same workflow should generalize to richer label sets (e.g., species-level panels or activity–liability profiles) without changing the architecture, it needs only a feature matrix, a transform matched to the label structure, and an in-context learner.

Key limitations suggest clear next steps. Antiparasitic activity is too sparse (77 test positives) to support firm conclusions. Homology is handled by stratification on the predefined split rather than by holding out entire identity clusters. Finally, the prioritization analysis treats all zeros as unlabelled (so PU-AP is a lower bound) and uses a single donor assignment as a negative control rather than a permutation distribution.

## 5 Conclusion

A tabular foundation model sets a new state of the art on the ESCAPE multi-activity AMP benchmark (five-label macro AP 77.8% vs 72.1%) using standard sequence descriptors, and is the first to meet or exceed the best published AP on all five labels at once. It requires neither task-specific training nor structural input, and its gains over published methods are largest for remote homologues. Ablations suggest the benchmark signal is largely coarse composition, and the resulting calibrated profile probabilities enable practical next-assay ranking.

## References

[1] Christopher J. L. Murray, Kevin Shunji Ikuta, Fablina Sharara, et al. Global burden of bacterial antimicrobial resistance in 2019: a systematic analysis. The Lancet, 399(10325):629–655, 2022. doi: 10.1016/S0140-6736(21)02724-0.

[2] Michael Zaslof. Antimicrobial peptides of multicellular organisms. Nature, 415(6870):389–395, 2002. doi: 10.1038/415389a.

[3] Kim A. Brogden. Antimicrobial peptides: pore formers or metabolic inhibitors in bacteria? Nature Reviews Microbiology, 3(3):238–250, 2005. doi: 10.1038/nrmicro1098.

[4] Robert E. W. Hancock and Hans-Georg Sahl. Antimicrobial and host-defense peptides as new anti-infective therapeutic strategies. Nature Biotechnology, 24(12):1551–1557, 2006. doi: 10.1038/nbt1267.

[5] Michael R. Yeaman and Nannette Y. Yount. Mechanisms of antimicrobial peptide action and resistance. Pharmacological Reviews, 55(1):27–55, 2003. doi: 10.1124/pr.55.1.2.

[6] Pratiti Bhadra, Jielu Yan, Jinyan Li, et al. AmPEP: sequence-based prediction of antimicrobial peptides using distribution patterns of amino acid properties and random forest. Scientific Reports, 8:1697, 2018. doi: 10.1038/s41598-018-19752-w.

[7] Travis J. Lawrence, Dana L. Carper, Margaret K. Spangler, et al. amPEPpy 1.0: a portable and accurate antimicrobial peptide prediction tool. Bioinformatics, 37(14):2058–2060, 2021. doi: 10.1093/bioinformatics/btaa917.

[8] Chenkai Li, Darcy Sutherland, S. Austin Hammond, et al. AMPlify: attentive deep learning model for discovery of novel antimicrobial peptides efective against WHO priority pathogens. BMC Genomics, 23(1):77, 2022. doi: 10.1186/s12864-022-08310-4.

[9] Hansol Lee, Songyeon Lee, Ingoo Lee, et al. AMP-BERT: prediction of antimicrobial peptide function based on a BERT model. Protein Science, 32(1):e4529, 2023. doi: 10.1002/pro.4529.

[10] Jing Xu, Fuyi Li, André Leier, et al. Comprehensive assessment of machine learning-based methods for predicting antimicrobial peptides. Briefings in Bioinformatics, 22(5):bbab083, 2021. doi: 10.1093/bib/bbab083.

[11] Paola Ruiz Puentes, Maria C. Henao, Javier Cifuentes, et al. Rational discovery of antimicrobial peptides by means of artificial intelligence. Membranes, 12(7):708, 2022. doi: 10.3390/membranes12070708.

[12] Yuxuan Pang, Lantian Yao, Jingyi Xu, et al. Integrating transformer and imbalanced multi-label learning to identify antimicrobial peptides and their functional activities. Bioinformatics, 38 (24):5368–5374, 2022. doi: 10.1093/bioinformatics/btac711.

[13] Jing Xu, Fuyi Li, Chen Li, et al. iAMPCN: a deep-learning approach for identifying antimicrobial peptides and their functional activities. Briefings in Bioinformatics, 24(4):bbad240, 2023. doi: 10.1093/bib/bbad240.

[14] Sebastian Ojeda, Rafael Velasquez, Nicolás Aparicio, et al. A standardized benchmark for multilabel antimicrobial peptide classification. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), Datasets and Benchmarks Track, 2025. arXiv:2511.04814.

[15] Zehua Sun, Jing Xu, Yumeng Zhang, et al. Multimodal geometric learning for antimicrobial peptide identification by leveraging AlphaFold2-predicted structures and surface features. Briefings in Bioinformatics, 26(3):bbaf261, 2025. doi: 10.1093/bib/bbaf261.

[16] Qiule Yu, Zhixing Zhang, Guixia Liu, et al. ToxGIN: an in silico prediction model for peptide toxicity via graph isomorphism networks integrating peptide sequence and structure information. Briefings in Bioinformatics, 25(6):bbae583, 2024. doi: 10.1093/bib/bbae583.

[17] Alexander Rives, Joshua Meier, Tom Sercu, et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences, 118(15):e2016239118, 2021. doi: 10.1073/pnas.2016239118.

[18] Judith Bernett, David B. Blumenthal, and Markus List. Cracking the black box of deep sequencebased protein–protein interaction prediction. Briefings in Bioinformatics, 25(2):bbae076, 2024. doi: 10.1093/bib/bbae076.

[19] Katarzyna Sidorczuk, Przemysław Gagat, Filip Pietluch, et al. Benchmarks in antimicrobial peptide prediction are biased due to the selection of negative data. Briefings in Bioinformatics, 23(5):bbac343, 2022. doi: 10.1093/bib/bbac343.

[20] Raúl Fernández-Díaz, Rodrigo Cossio-Pérez, Clement Agoni, et al. AutoPeptideML: a study on how to build more trustworthy peptide bioactivity predictors. Bioinformatics, 40(9):btae555, 2024. doi: 10.1093/bioinformatics/btae555.

[21] Noah Hollmann, Samuel Müller, Katharina Eggensperger, et al. TabPFN: a transformer that solves small tabular classification problems in a second. In International Conference on Learning Representations (ICLR), 2023.

[22] Noah Hollmann, Samuel Müller, Lennart Purucker, et al. Accurate predictions on small data with a tabular foundation model. Nature, 637:319–326, 2025. doi: 10.1038/s41586-024-08328-6.

[23] Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, et al. Transformers can do Bayesian inference. In International Conference on Learning Representations (ICLR), 2022.

[24] Shuichi Kawashima, Piotr Pokarowski, Maria Pokarowska, et al. AAindex: amino acid index database, progress report 2008. Nucleic Acids Research, 36(suppl\_1):D202–D205, 2008. doi: 10.1093/nar/gkm998.

[25] Grigorios Tsoumakas and Ioannis Katakis. Multi-label classification: an overview. International Journal of Data Warehousing and Mining, 3(3):1–13, 2007. doi: 10.4018/jdwm.2007070101.

[26] Krzysztof Dembczyński, Weiwei Cheng, and Eyke Hüllermeier. Bayes optimal multilabel classification via probabilistic classifier chains. In Proceedings of the 27th International Conference on Machine Learning (ICML), pages 279–286, 2010.

[27] Jesse Read, Bernhard Pfahringer, Geof Holmes, et al. Classifier chains for multi-label classification. Machine Learning, 85(3):333–359, 2011. doi: 10.1007/s10994-011-5256-5.

[28] Martin Steinegger and Johannes Söding. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nature Biotechnology, 35(11):1026–1028, 2017. doi: 10.1038/nbt.3988.

[29] Charles Elkan and Keith Noto. Learning classifiers from only positive and unlabeled data. In Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 213–220, 2008. doi: 10.1145/1401890.1401920.

[30] Jessa Bekker and Jesse Davis. Learning from positive and unlabeled data: a survey. Machine Learning, 109(4):719–760, 2020. doi: 10.1007/s10994-020-05877-5.

[31] Jiyun Han, Tongxin Kong, and Juntao Liu. PepNet: an interpretable neural network for anti-inflammatory and antimicrobial peptides prediction using a pre-trained protein language model. Communications Biology, 7(1):1198, 2024. doi: 10.1038/s42003-024-06911-1.

[32] Jiahui Guan, Lantian Yao, Peilin Xie, et al. A two-stage computational framework for identifying antiviral peptides and their functional types based on contrastive learning and multi-feature fusion strategy. Briefings in Bioinformatics, 25(3):bbae208, 2024. doi: 10.1093/bib/bbae208.

[33] Krzysztof Dembczyński, Willem Waegeman, Weiwei Cheng, et al. On label dependence and loss minimization in multi-label classification. Machine Learning, 88(1–2):5–45, 2012. doi: 10.1007/s10994-012-5285-8.

[34] Oscar Luaces, Jorge Díez, José Barranquero, et al. Binary relevance eficacy for multilabel classification. Progress in Artificial Intelligence, 1(4):303–313, 2012. doi: 10.1007/s13748-012-0030-x.

[35] Min-Ling Zhang and Zhi-Hua Zhou. A review on multi-label learning algorithms. IEEE Transactions on Knowledge and Data Engineering, 26(8):1819–1837, 2014. doi: 10.1109/TKDE. 2013.39.

# Supplementary Material

## S1 The multi-label transforms, with a worked example

## S1.1 Notation

Let $\mathbf { x } \in \mathbb { R } ^ { 3 3 0 }$ denote a peptide’s descriptor vector and $\mathbf { y } \in \{ 0 , 1 \} ^ { L }$ its label vector, with $L = 5$ in the benchmark comparison and $L = 4$ in every dependence analysis. Labels are ordered $y _ { 1 } =$ Antibacterial, $y _ { 2 } = \mathrm { A n t i f u n g a l }$ , y = Antiviral, $y _ { 4 } = \mathrm { A n t i p a r a s i t i c }$ $y _ { 5 } = \mathrm { A }$ ntimicrobial, and we call a complete assignment y a profile. $\hat { P }$ denotes a TabPFN posterior and 1[·] an indicator.

All five transforms wrap the same backbone on the same features and the same split. They difer only in how the five-label target is broken into problems TabPFN can solve, and therefore in what they can say about how activities co-occur.

## S1.2 A worked example

The diferences are easiest to see on two labels. Consider one peptide x and the first two links of the chain order used throughout, Antibacterial (AB) then Antifungal (AF), and suppose the fitted models give

$$
{ \hat { P } } ( { \mathrm { A B } } = 1 | { \bf x } ) = 0 . 6 , \qquad { \hat { P } } ( { \mathrm { A F } } = 1 | { \bf x } , { \mathrm { A B } } = 1 ) = 0 . 7 , \qquad { \hat { P } } ( { \mathrm { A F } } = 1 | { \bf x } , { \mathrm { A B } } = 0 ) = 0 . 1 .\tag{5}
$$

The peptide is moderately likely to be antibacterial, and antifungal activity is far more plausible if it is. Chaining these gives a probability for each of the four profiles:

<table><tr><td>Profile  ${ \hat { P } } ( \mathbf { y } \mid \mathbf { x } )$ </td></tr><tr><td>(AB = 1, AF = 1) 0.42  $= 0 . 6 \times 0 . 7$  (AB = 0, AF = 1) = 0.4 × 0.1</td></tr><tr><td>(AB = 1, AF = 0) 0.18 = 0.6 × 0.3 0.04</td></tr><tr><td>(AB = 0, AF = 0) 0.36 = 0.4 × 0.9</td></tr><tr><td>Total 1.00</td></tr></table>

so the correct marginals are $\hat { P } ( \mathrm { A B } = 1 ) = 0 . 4 2 + 0 . 1 8 = 0 . 6 0$ and $\hat { P } ( \mathrm { A F } = 1 ) = 0 . 4 2 + 0 . 0 4 = 0 . 4 6$ Each subsection below states what each transform does with this peptide. The numbers are for illustration purpose only.

## S1.3 Binary relevance (BR)

L independent classifiers, one per label, each seeing only x:

$$
\hat { P } _ { \mathrm { B R } } ( \mathbf { y } \mid \mathbf { x } ) = \prod _ { j = 1 } ^ { L } \hat { P } ( y _ { j } \mid \mathbf { x } )\tag{6}
$$

A well-fitted BR recovers both marginals exactly, 0.60 and 0.46, because each classifier is estimating a marginal and nothing else. What it cannot represent is the joint: BR assigns the doubly-active profile $0 . 6 0 \times 0 . 4 6 = 0 . 2 7 6$ against the 0.42 above, understating co-occurrence by 0.14. Per-label ranking metrics such as average precision are blind to this, which is why BR is competitive on mAP-5 (Table 3) yet loses to the joint models on held-out log-likelihood (Section 3.5).

## S1.4 Label powerset (LP)

Each observed profile is relabelled as one class of a multiclass problem over the set $\mathcal { C } = \{ 1 1 , 1 0 , 0 1 , 0 0 \}$ of profiles present among the in-context samples, and marginals are recovered by summation (Equation 2). On the example LP predicts the four-way distribution (0.42, 0.18, 0.04, 0.36) directly and returns $\hat { P } ( \mathrm { A F } = 1 ) = 0 . 4 2 + 0 . 0 4 = 0 . 4 6$ , the same answer as the joint models, reached in one fit instead of a chain.

The cost of this is that C is closed. Had the profile $( \mathrm { A B } = 0 , \mathrm { A F } = 1 )$ never occurred among the in-context samples, LP could not assign it any probability, and its 0.04 would be redistributed over the profiles that were observed. On the benchmark this is harmless: the Antimicrobial parent is a deterministic ${ \mathrm { O R } } ,$ so only $2 ^ { 4 } = 1 6$ profiles are realizable and all 16 occur, well inside TabPFN’s class ceiling. It is also why LP cannot violate the parent constraint, as no illegal profile is in its output space at all.

## S1.5 Probabilistic classifier chains (PCC)

At inference every profile is evaluated and the marginals recovered by summation. On the example this is the enumeration in Section S1.2, giving $\hat { P } ( \mathrm { A F } = 1 ) = 0 . 4 6$ . PCC is the only transform here that returns a full joint posterior over profiles, which the oracle-ceiling, joint-likelihood and assay-prioritization analyses require.

Unlike LP where we only use four primary activities and compute deterministic antimicrobial activity using OR gate, PCC’s output space is all $2 ^ { 5 }$ profiles, so it can place probability mass on contradictory profiles, for example, a peptide called antifungal but not antimicrobial. This is the mechanical origin of the violation rates in Section 3.5.

## S1.6 Classifier chains (CC)

The same chain-rule factorization as PCC above, but a single path is followed rather than every branch:

$$
\hat { P } _ { \mathrm { C C } } ( \mathbf { y } \mid \mathbf { x } ) = \prod _ { j } \hat { P } ( y _ { j } \mid \mathbf { x } , y _ { < j } ) ,\tag{7}
$$

with $y _ { < j }$ supplied as ground truth in the in-context samples and replaced at inference by the chain’s own prediction. On the example, CC thresholds AB at 0.5, commits to $\mathrm { A B } = 1$ , and reports link $2 \mathrm { { ^ { \circ } s } }$ answer for that branch alone, $\hat { P } ( \mathrm { A F } = 1 ) = 0 . 7 0$ against the correct 0.46. The 40% of probability mass in which the peptide is not antibacterial, and in which antifungal activity was unlikely, has been discarded. The error grows with depth, which is consistent with CC’s worst per-label result falling on Antiparasitic, the fourth link (Section 3.1).

What is appended to the next link’s features also matters. Every link sees true $0 / 1$ labels in its context. Hard context appends $\mathbf { 1 } [ \hat { p } _ { < j } > \tau ]$ and so matches that encoding. $S o f t$ context appends the raw probability $\hat { p } _ { < j } \in [ 0 , 1 ]$ , here 0.6, a value the second link never saw in context, which presents the model with an out-of-distribution input. The distinction moves CC’s recovery of the oracle ceiling from −40.8% to +13.7% (Section 3.5).

## S1.7 Ensemble classifier chains (ECC)

CC’s 0.70 above is partly an artefact of putting AB first, under the order $\mathrm { A F }  \mathrm { A B }$ the first link estimates ${ \hat { P } } ( \operatorname { A F } = 1 \mid \mathbf { x } )$ with no conditioning at all, and returns something diferent. ECC runs eight chains under independently sampled label orders and averages each label’s marginal across them, so that no single arbitrary ordering determines the result. It does not fix the committed-path problem, only spreads it over orderings. Run at TabPFN parameter $n _ { \mathrm { e s t i m a t o r s } } = 8$ to make eight chains afordable.

## S2 Per-seed results

Table S1 reports a per-seed benchmark of all five multi-label transforms. Every transform uses the same TabPFN v3 base classifier (tabpfn 8.2.0) on the same 330 sequence descriptors. We use all 65,870 peptides as in-context samples and evaluate on the held-out test set of 16,489 peptides. Three seeds (42, 1665, 8914) are reported individually together with their mean ± standard deviation. Bold marks the best transform mean in each column.

Average precision: mAP-5 macro-averages step-wise AP over all five labels. mAP-4 excludes Antimicrobial, which is the deterministic OR of the other four and is therefore trivially predictable. mAP-3 further excludes Antiparasitic $( n = 7 7$ test positives), leaving the three labels for which the test set is adequately powered.

Threshold metrics: Max-F1 macro-averages the per-label maximum $F _ { 1 }$ over thresholds chosen on the test set. It is reported only as an optimistic upper bound. Honest $F _ { 1 }$ uses per-label thresholds selected on 5-fold out-of-fold predictions over the in-context set and frozen before the test set is touched. The Max-F1 minus honest- $F _ { 1 }$ gap is the cost of not knowing the threshold in advance. Subset accuracy is the exact-match rate over the full five-label vector at a fixed 0.5 cut on every label. It understates performance on the low-prevalence labels and should be read as a conservative floor rather than a tuned operating point.

Wall-clock: OOF is the 5-fold out-of-fold (OOF) pass over the in-context set used to fix the honest thresholds. Fit is the final fit with all 65,870 in-context samples. Infer is chunked inference over the 16,489 test peptides. All times are seconds on a single GPU. TabPFN performs in-context learning with no gradient updates, so the fit time is dominated by ingesting the context matrix rather than by optimisation. This is why it is one to two orders of magnitude smaller than inference. Timings come from a shared machine and include contention. The CC OOF spread (±427 s) reflects competing jobs, not a property of the transform.

ECC uses 8 chains with n\_estimators = 8. CC and PCC use the default label order. All metric values are percentage points. The PCC subset-accuracy lead over LP is 0.01 points, well inside the seed-to-seed spread, so it should not be read as a separation.

## S3 Scaling with the number of in-context samples

BR, LP, and CC were refitted on nested random subsets of the in-context set drawn from both folds, on the geometric ladder $n \in \{ 2 5 0 , 1 , 2 5 0 , 6 , 2 5 0 , 3 1 , 2 5 0 , 6 5 , 8 7 0 \}$ . Between the top two rungs mAP-5 falls by 5.01 points for BR (76.94 to 71.93), 4.71 for LP (77.79 to 73.07), and 4.43 for CC (75.25 to 70.82). This motivated the matched-protocol experiment in Section 3.2, but is not the same ablation because a random subset spans both folds, whereas the fold-matched analysis uses one fold at a time as context, and the fold-matched confound in Section 3.2 is correspondingly about half as large.

Table S2 uses the same held-out test set of 16,489 peptides. Each cell is the mean over three seeds (42, 1665, 8914). mAP-5, Max-F1, and subset accuracy are reported in percentage points. Subset accuracy is the exact-match rate over the five-label vector at a fixed 0.5 cut, while Max-F1 uses test-tuned thresholds (an optimistic upper bound).

The 65,870 rows are an independent re-run of the full-data configuration and difer from Table ?? by at most 0.07 points, bounding run-to-run variation at fixed seed.

<sub>er-seed</sub> <sub>benchmark</sub> <sub>of</sub> a<sup>ll</sup> <sup>five</sup> <sup>multi-label</sup> <sup>t</sup>
<table><tr><td></td><td colspan="3">Average precision</td><td colspan="3">Threshold metrics</td><td colspan="3">Wall-clock (s)</td></tr><tr><td>Seed</td><td>mAP-5</td><td>mAP-4</td><td>mAP-3</td><td>Max-F1</td><td>Honest F1</td><td>Subset acc.</td><td>OOF</td><td>Train</td><td>Infer</td></tr><tr><td colspan="10">Binary relevance (BR)</td></tr><tr><td>42</td><td>76.93</td><td>71.64</td><td>84.07</td><td>74.17</td><td>72.43</td><td>90.56</td><td>1,599</td><td>85</td><td>647</td></tr><tr><td>1665</td><td>76.90</td><td>71.60</td><td>84.07</td><td>73.84</td><td>73.04</td><td>90.37</td><td>1,595</td><td>84</td><td>640</td></tr><tr><td>8914</td><td>77.20</td><td>71.98</td><td>84.06</td><td>74.09</td><td>72.48</td><td>90.46</td><td>1,596</td><td>85</td><td>652</td></tr><tr><td>mean</td><td>77.01 ± 0.17</td><td>71.74±0.21</td><td>84.07±0.00</td><td>74.03±0.17</td><td>72.65±0.34</td><td>90.46±0.10</td><td>1,597±2</td><td>85±1</td><td>646±6</td></tr><tr><td colspan="10">Label powerset (LP)</td></tr><tr><td>42</td><td>78.05</td><td>73.03</td><td>84.73</td><td>74.26</td><td>73.74</td><td>90.75</td><td>321</td><td>18</td><td>128</td></tr><tr><td>1665</td><td>77.60</td><td>72.47</td><td>84.95</td><td>74.19</td><td>73.40</td><td>90.82</td><td>319</td><td>18</td><td>128</td></tr><tr><td>8914</td><td>77.74</td><td>72.64</td><td>84.77</td><td>74.18</td><td>73.91</td><td>90.70</td><td>328</td><td>18</td><td>128</td></tr><tr><td>mean</td><td>77.79±0.23</td><td>72.71 ± 0.29 84.82± 0.11 74.21 ± 0.05 73.69 ± 0.26</td><td></td><td></td><td></td><td>90.76±0.06</td><td>323 ± 5</td><td>18±0</td><td>128±0</td></tr><tr><td colspan="10">Classifier chain (CC)</td></tr><tr><td>42</td><td>74.73</td><td>69.08</td><td>83.21</td><td>71.30</td><td>70.13</td><td>89.94</td><td>1,626</td><td>96</td><td>896</td></tr><tr><td>1665</td><td>75.36</td><td>69.79</td><td>83.26</td><td>72.19</td><td>70.31</td><td>90.04</td><td>2,296</td><td>115</td><td>907</td></tr><tr><td>8914</td><td>75.72</td><td>70.24</td><td>83.42</td><td>72.15</td><td>69.78</td><td>90.18</td><td>2,419</td><td>160</td><td>918</td></tr><tr><td>mean</td><td>75.27±0.50</td><td>69.70±0.58</td><td>83.30±0.11</td><td>71.88±0.50</td><td>70.07±0.27</td><td>90.06±0.12</td><td>2,114±427</td><td>124± 33</td><td>907±11</td></tr><tr><td colspan="10">Ensemble of classifier chains (ECC)</td></tr><tr><td>42</td><td>76.90</td><td>71.66</td><td>83.98</td><td>73.31</td><td>71.53</td><td>90.39</td><td>8,678</td><td>235</td><td>3,805</td></tr><tr><td>1665</td><td>76.97</td><td>71.76</td><td>83.81</td><td>73.27</td><td>73.01</td><td>90.11</td><td>8,523</td><td>267</td><td>3,779</td></tr><tr><td>8914</td><td>76.80</td><td>71.56</td><td>84.00</td><td>73.44</td><td>72.50</td><td>90.41</td><td>8,576</td><td>264</td><td>3,788</td></tr><tr><td>mean</td><td>76.89±0.09</td><td>71.66±0.10</td><td>83.93±0.11</td><td>73.34±0.09</td><td>72.34±0.75</td><td>90.30±0.17</td><td>8,592±79</td><td>255± 18 3,791 ±13</td><td></td></tr><tr><td colspan="10">Probabilistic classifier chain (PCC)</td></tr><tr><td>42</td><td>77.73</td><td>72.65</td><td>84.31</td><td>74.08</td><td>73.43</td><td>90.80</td><td>8,694</td><td>88</td><td>4,127</td></tr><tr><td>1665</td><td>77.66</td><td>72.56</td><td>84.35</td><td>74.00</td><td>73.35</td><td>90.72</td><td>8,690</td><td>81</td><td>4,092</td></tr><tr><td>8914</td><td>77.89</td><td>72.84</td><td>84.40</td><td>74.17</td><td>72.72</td><td>90.78</td><td>8,709</td><td>77</td><td>4,129</td></tr><tr><td>mean</td><td>77.76±0.12</td><td>72.68±0.14</td><td>84.35±0.04</td><td>74.08±0.09</td><td>73.16±0.39</td><td>90.76±0.04</td><td>8,698±10</td><td></td><td>82±6 4,116±21</td></tr></table>

Table S2: Context-size scaling sweep. Macro metrics versus the number of in-context samples for BR, LP and CC.
<table><tr><td></td><td colspan="3">mAP-5</td><td colspan="3">Max-F1</td><td colspan="3">Subset acc.</td></tr><tr><td>Context size</td><td>BR</td><td>LP</td><td>CC</td><td>BR</td><td>LP</td><td>CC</td><td>BR</td><td>LP</td><td>CC</td></tr><tr><td>250</td><td>44.18</td><td>46.15</td><td>43.89</td><td>44.84</td><td>47.08</td><td>44.69</td><td>79.04</td><td>79.39</td><td>80.29</td></tr><tr><td>1,250</td><td>53.93</td><td>54.73</td><td>53.93</td><td>53.27</td><td>53.55</td><td>53.08</td><td>82.26</td><td>82.80</td><td>83.25</td></tr><tr><td>6,250</td><td>62.05</td><td>62.23</td><td>61.52</td><td>60.27</td><td>59.45</td><td>60.01</td><td>84.75</td><td>85.23</td><td>85.12</td></tr><tr><td>31,250</td><td>71.93</td><td>73.07</td><td>70.82</td><td>69.17</td><td>69.69</td><td>67.59</td><td>88.73</td><td>89.15</td><td>88.51</td></tr><tr><td>65,870</td><td>76.94</td><td>77.79</td><td>75.25</td><td>74.16</td><td>74.29</td><td>71.71</td><td>90.46</td><td>90.75</td><td>90.03</td></tr></table>

## S4 Provenance of the published baselines

ESCAPE authors retrained all seven prior methods from scratch on the benchmark’s training folds and adapted each to emit five sigmoid outputs. Therefore no cross-dataset leakage in the published comparison. However, only two of the seven were natively designed for multi-activity prediction over these labels, and Table S3 documents what each method’s original publication actually addressed. A low row in the main-text frontier table should not be read as a verdict on the corresponding published method.

Table S3: What each published baseline’s original paper addressed. ✓ denotes an activity the original method predicted; × denotes one it did not. Sample counts are as reported in the original publication, not on the benchmark.
<table><tr><td>Method</td><td>Original dataset</td><td>n (original)</td><td>AB</td><td>AV</td><td>AF</td><td>AP</td><td>AM</td><td>Native task</td></tr><tr><td>AMPs-Net [11]</td><td>19 public repositories</td><td>23,967</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>binary + 4-label</td></tr><tr><td>TransImbAMP [12]</td><td>dbAMP/DRAMP/DBAASP, CD-HIT 40%</td><td>22,381</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>binary + 7-label</td></tr><tr><td>AMP-BERT [9]</td><td>Veltri benchmark (APD3 + UniProt)</td><td>3,556</td><td>X</td><td>X</td><td>X</td><td>X</td><td>1</td><td>binary</td></tr><tr><td>PEP-Net [31]</td><td>AMP and AIP benchmarks</td><td>multiple</td><td>√a</td><td>√a</td><td>√a</td><td>X</td><td>√</td><td>binary (separate tasks)</td></tr><tr><td>amPEPpy [7]</td><td>APD3/CAMPR3/LAMP + UniProt</td><td>3,268 pos.</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td><td>binary</td></tr><tr><td>AVP-IFT [32]</td><td>AVPdb/dbAMP/DRAMP/DBAASP/HPIdb + Uniprot</td><td>5,324</td><td>X</td><td>√</td><td>X</td><td>X</td><td>×</td><td>antiviral specialist</td></tr><tr><td>AMPlify [8] ESCAPE [14]</td><td>APD3 + DADP + UniProt</td><td>8,346</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td><td>binary</td></tr><tr><td></td><td>26 repositories + UniProt</td><td>82,359</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>5-label multimodal</td></tr><tr><td>This work</td><td>ESCAPE (unmodified)</td><td>82,359</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>5-label, sequence only</td></tr></table>

AB, antibacterial; AV, antiviral; AF, antifungal; AP, antiparasitic; AM, antimicrobial (binary AMP vs non-AMP). <sup>a</sup> As separate single-activity binary datasets, not a joint multi-label head.