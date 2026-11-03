# Anomaly-ranked confirmation slows new-class learning

Argyro Sasli1 and Michael Coughlin¹

1School of Physics and Astronomy, University of Minnesota, Minneapolis, MN 55455, USA (Dated: November 3, 2026)

Anomaly detectors rank unfamiliar examples for expert confirmation, and the confirmed examples teach a classifier to recognise a new class. Across five benchmarks from physics, medicine and computer vision, the members flagged by the detector are those the classifier learns first. In three systems, confirming the highest-ranked members of a new class gives lower recognition than confirming the same number of random members. Handing subsequent selection to the promoted classifier recovers part of this loss in two systems. A paired intervention further shows that replacing nearby known classes with distant ones sharply reduces the confirmations needed for recognition. These results distinguish finding a new class from selecting examples that teach its extent: within the tested classes, concentrating confirmations at the top of the anomaly ranking makes learning less efficient.

## INTRODUCTION

Anomaly detection is a route to discovering unfamiliar events. In time-domain astronomy, gravitational-wave detector characterisation and clinical monitoring, discovery pipelines pair a classifier trained on known classes with an anomaly detector that flags unfamiliar inputs [1– 6]. Experts label the selected candidates through spectroscopy, visual inspection or clinical review, and we call each such label a confirmation. Confirmed examples enter training as a new class, but telescope time and expert hours limit how many can be obtained.

Out-of-distribution detection and open-set recognition evaluate whether a system flags unfamiliar inputs [7-12]. Few-shot, class-incremental and novel-category learning evaluate how a system learns from labelled examples [13– 17]. Active learning addresses the value and selection bias of queried examples [18, 19], including pools containing unknown classes [20, 21]. Here we examine the link between discovery and subsequent learning: does the ranking that finds a new class select useful examples for teaching that same class?

Our FLARE transient pipeline [6] measures recognition after returning confirmed examples of a withheld class to training. Those examples were drawn independently of the anomaly ranking. In an anomaly-driven follow-up programme, however, the detector also influences which members receive labels.

We test that choice by withholding classes from five labelled benchmarks and returning their members as confirmations, using benchmark labels in place of expert labels. We first compare which members are flagged and learned, then intervene on confirmation selection and on the known classes surrounding the new class.

## RESULTS

## The detector flags the members the classifier learns first

For each eligible withheld class, we divided its held-out members into those flagged by the zero-label anomaly detector and those not flagged. Both groups were evaluated with the same classifier, promoted using random confirmations. Differences therefore concern which members are recognised, rather than differences in the confirmations used for training.

Across all five datasets, flagged members are recognised after fewer confirmations than unflagged members (Fig. 1). In Gravity Spy, recognition after one confirmation is 0.69 for flagged members and 0.24 for unflagged members; the groups converge only near forty confirmations. In CIFAR-100 and Food-101, flagged members reach 0.86 and 0.72 after one confirmation, whereas unflagged members reach 0.79 and 0.57 after forty. In the electrocardiograms, recognition of flagged members declines as confirmations accumulate but remains above recognition of unflagged members.

The detector thus selects a region of the new class that the classifier can already learn from very few confirmations. This motivates testing whether repeatedly confirming that region is an efficient way to learn the rest of the class.

## Detector-selected confirmations slow new-class learning

For each withheld class, we compared three choices of n confirmations: its highest anomaly scores, n random members, and its lowest anomaly scores. Only confirmation selection changed; the learner and evaluation protocol were fixed. We tested Gravity Spy with gradientboosted trees on image features, CIFAR-100 with a linear classifier on ResNet-50 features, and electrocardiograms with gradient-boosted trees on hand-crafted features.

![](images/392c2282aa703989a700d73fbf78d38153c8e1f3db4f84fa46c3086240ad923b.jpg)  
FIG. 1. Discovery and learning favour the same members of a new class. Top: an anomaly detector flags an unfamiliar population, experts confirm selected members, and the confirmed examples enter training as a new class. Recognition is measured at a fixed false-alarm rate; the label cost N(f) is the estimated number of confirmations needed to reach efficiency f. Bottom: probability that a held-out member is recognised at a 1% false-alarm rate after promotion with n random confirmations, separately for members flagged before any label (magenta) and members not flagged (dark). Curves average over withholds with at least three members in each group; parentheses give the number of withholds.

Detector-selected confirmations give lower recognition than random confirmations in all three systems (Fig. 2). After ten confirmations, the difference is negative for twenty of twenty-two Gravity Spy classes, all thirteen CIFAR-100 classes and ten of eleven ECG classes. Classbootstrap intervals exclude zero from five confirmations onwards in Gravity Spy, at every measured budget in CIFAR-100 and at five of six budgets in the ECGs. Averaged over three Gravity Spy training seeds, the difference at ten confirmations is -0.17 (95% interval -0.24 to -0.11). At twenty confirmations, recognition remains 0.66 against 0.82 for random selection.

The penalty is not confined to a few extreme examples. Drawing n confirmations at random from the 2n highestranked members also underperforms unrestricted random selection in all eight Gravity Spy classes tested. Selecting the least anomalous members gives recognition close to or slightly below random selection. These comparisons are consistent with the detector concentrating confirmations on members the classifier already recognises readily, while a random sample covers a broader part of the class.

To match the recognition obtained from ten random confirmations, detector-selected confirmations require approximately eighteen in Gravity Spy and nineteen in the ECGs, estimated by interpolation of the mean curves. In CIFAR-100, twenty detector-selected confirmations remain below the recognition obtained from five random ones. Random-confirmation benchmarks can therefore underestimate the label requirement under anomaly-ranked confirmation in these systems.

A classifier can recover part of the loss by taking over subsequent selection (Fig. 3). We first confirmed the detector's two highest-ranked members, promoted the class, and then sampled further confirmations among members flagged by the promoted classifier. At ten confirmations, this recovers 63% of the gap between detector-selected and random selection in Gravity Spy and 72% in CIFAR-100. It gives no corresponding improvement in the ECGs. Choosing only the members with the highest promotedclass probability performs worse than sampling among the classifier's alerts (Supplementary Sec. S3).

The same ordering holds when confirmations are drawn from a mixed stream that also contains a second withheld class and known-class events. At twenty confirmations, the detector's highest-ranked events include more members of the new class than random draws among its alerts, 15.6 against 12.9 in Gravity Spy and 10.8 against 7.2 in CIFAR-100, yet give lower mean recognition, 0.57 against 0.66 and 0.70 against 0.72. Handing selection to the promoted classifier gives the highest recognition in both systems, 0.68 and 0.77 (Supplementary Sec. S4).

![](images/582e659b888beb78ecca38ebac80c5d6e8050924ded222855c3f667fec7cb0f0.jpg)

FIG. 2. Detector-selected confirmations slow new-class learning. (a) Mean recognition at a 1% false-alarm rate after promotion with n random members (dark), the n highest-ranked members under the anomaly detector (magenta), or the n least anomalous members (light). Gravity Spy uses all twenty-two classes; CIFAR-100 uses thirteen classes with zerolabel detectability between 0.1 and 0.9; ECG uses thirty-five withholds of eleven classes in geometry-selected sub-taxonomies. (b) Class-averaged differences from random selection with 95% bootstrap intervals over classes. Gravity Spy at five, ten and twenty confirmations averages three training seeds. Each ECG class enters once, through its median over the sub-taxonomies containing it.  
![](images/05ce79486aa28fada840e75e49f5aca85b6c492a60f3a11ee5fff9a3ae63a3e4.jpg)  
FIG. 3. Handing later confirmations to the classifier recovers part of the penalty. Recognition at a 1% false-alarm rate for the hand-off cohorts: eight Gravity Spy classes with zero-label detectability between 0.1 and 0.9, thirteen CIFAR-100 classes, and thirty-five ECG withholds involving eleven classes. Following two detector-selected confirmations, the promoted classifier selects further members by sampling among its alerts (dashed) or by highest promoted-class probability (dotted). Random and detector-selected policies are evaluated on the same cohorts.

## Known-class context changes the cost of learning

We tested the influence of neighbouring known classes with a paired intervention. For each of twenty-one Gravity Spy classes, two six-class taxonomies shared the withheld class, two background classes and the confirmed examples. The remaining three classes were either its nearest or its farthest eligible neighbours in feature space.

Distant neighbours reduce the estimated number of confirmations needed for 80% recognition for eighteen of twenty-one classes. The median falls from 4.3 to 1.6 confirmations (Fig. 4a). The result persists across four replicate draws and alternative readings of label cost (Supplementary Sec. S2).

In CIFAR-100, we replaced a withheld class's four semantic siblings with four classes from other superclasses, holding five further background classes and the confirmation draws fixed. Best recognition through forty confirmations is higher with distant classes for ninety-nine of one hundred classes (Fig. 4b). These interventions establish a dependence on known-class context; they do not establish a geometric law predicting the size of the reduction.

## DISCUSSION

The examples that reveal a new class need not be those that teach its extent. In the tested systems, anomalyranked confirmation concentrates labels on members the promoted classifier recognises readily, producing lower recognition than random confirmation at the same budget. Handing later selection to the classifier's alerts recovers part of this loss in two systems, while selecting only its highest-probability members remains ineffective.

These results distinguish novelty from representativeness within a new class. Programmes such as the Bright Transient Survey select confirmations through a predefined brightness-based selection function rather than anomaly rank [22, 23]. Combining anomaly-driven discovery with broader confirmation sampling may similarly help pipelines learn newly identified populations.

The main experiments draw confirmations from members of the withheld class alone. In a mixed stream, where labels are also spent on known classes and a second unknown class, the detector finds more members of the new class and still teaches it less efficiently than random sampling among its alerts. Withholding an existing class also differs from encountering a genuinely new population. Noisy labels and unequal confirmation costs may alter the trade-off.

The neighbour intervention shows that label requirements depend on the surrounding taxonomy as well as on the class itself. It does not supply a general rule for predicting how many labels a class needs. A prospective evaluation should record candidate rankings, all confirmation costs and subsequent recognition, testing whether broader sampling retains its advantage under expert confirmation.

## METHODS

## Datasets and representations

We used five labelled benchmarks with fixed feature representations and train, validation and test partitions.

Gravity Spy contains twenty-two morphological classes of LIGO detector glitches [24]. Models use 984 pooledpixel features from four spectrogram windows and a stratified 70/15/15 split. Before withholding, the pipeline passed its specified accuracy criterion of 0.90, reaching 0.963. We removed all 37 records belonging to 18 exact-duplicate groups and repeated the experiments with otherwise unchanged designs and seeds. Four groups had copies across the test and training or validation partitions. All reported Gravity Spy results use the deduplicated data

PTB-XL provides twelve-lead electrocardiograms [25]. We retained records with a unique dominant diagnostic subclass at annotation likelihood $\geq 5 0$ , giving 15 041 of 21 799 records. Models use 108 per-lead summary features: moments, amplitude range, autocorrelation-based rhythm measures and spectral-band fractions. Official folds 1–8 supply training members, fold 9 validation and thresholds, and fold 10 evaluation. Over sixteen subclasses with at least fifty members, full-taxonomy macro AUC was 0.855, above the specified 0.80 criterion. Restricting to unambiguous labels limits extrapolation to mixed clinical cases.

Optical transients use 160 class-agnostic light-curve features for 10 338 quality-passing Bright Transient Survey objects and 37 classical novae, with the splits documented by the companion FLARE study [6, 22, 23]. Host and redshift context are excluded. Full-taxonomy macro AUC over sixteen classes was 0.909, above the specified 0.80 criterion.

CIFAR-100 has one hundred classes grouped into twenty superclasses [26]. We used ImageNet-pretrained ResNet-50 features [27], reduced from 2048 to 256 principal components fitted without labels on training images. Each class contributes 450 training, 50 validation and 100 test images. Full-taxonomy macro AUC was 0.990, above the specified 0.80 criterion.

Food-101 has one hundred and one classes [28]. We sampled 300 training and 50 validation images per class from its training split and 100 test images per class from its official validation split, with seed 0. Images were resized to 256 pixels on the shorter side and centre-cropped to 224 before feature extraction.

The member-level comparison includes full and sparse taxonomies. Gravity Spy uses four random six-class and three eleven-class taxonomies; ECG and light curves use four six-class and three nine-class taxonomies, retaining the normal ECG or Type Ia supernova background class. CIFAR-100 and Food-101 use six ten-class and three twenty-class taxonomies. Random taxonomy draws use seed 11.

## Models and withholding

Gravity Spy, ECG and transient models use multiclass LightGBM [29] with 63 leaves, learning rate 0.08 for Gravity Spy and 0.06 elsewhere, and early stopping after 80 validation rounds. Maximum rounds are 1500 and 1200, respectively. Gravity Spy uses feature and bagging fractions 0.7 and 0.8. Image models use multinomial logistic regression on frozen features, fitted by full-batch L-BFGS with an $L _ { 2 }$ penalty of $1 0 ^ { - 4 }$ . Except in Gravity Spy, class weights are proportional to $\sqrt { N / ( K N _ { k } ) }$ , where $N _ { k }$ is the number of training members of class k among K classes.

![](images/cd49aa1addef7e73eb22e166e29164d3f5481661247b7a3b63cc384893291712.jpg)

![](images/254df17d2537cf0c5b277c0b568eb09734d55d9ad05da45061a953ffd7b05ca0.jpg)  
FIG. 4. Known-class context changes new-class learning cost. (a) Estimated confirmations needed for 80% recognition at a 1% false-alarm rate for each Gravity Spy class, among its three nearest eligible known classes (dark) or its three farthest (magenta). Two further background classes are shared. Costs are computed separately in four replicates and averaged within each class. (b) CIFAR-100: best recognition through forty confirmations with semantic neighbours versus distant classes. Five further background classes and the confirmation draws are shared; ninety-nine of one hundred points lie above the diagonal.

For each withheld class $c ,$ we trained on the other $K - 1$ classes. The zero-label anomaly score is the energy of the known-class logits [10],

$$
E ( x ) = - \log \sum _ { k } e ^ { z _ { k } ( x ) } .\tag{1}
$$

Its threshold is the 99th percentile of scores on held-out known-class events. Zero-label detectability is the fraction of held-out members of c above this threshold.

We then added n confirmed members of c as a new training class and retrained. Recognition efficiency is the fraction of held-out target members whose promotedclass probability exceeds its 99th-percentile threshold on held-out known-class events. Thresholds are estimated separately for each promoted model. The nominal 1% false-alarm rate is a benchmark operating point, not a guarantee for a future stream. Target evaluation members are excluded from confirmation selection and training.

Random-confirmation curves average three draws per label count. The member-level analysis uses raw recognition at each measured budget. For the Gravity Spy swap, $N _ { 8 0 }$ is the first crossing of 80% on the running maximum of the mean recognition curve, linearly interpolated between measured budgets. If the first measured budget reaches the target, its label count is used; classes not reaching it are censored. Fractional costs are estimated crossings, not guaranteed integer requirements. Alternative readings are reported in Supplementary Sec. S2.

## Member-level comparison

For each withhold, we used the zero-label detector to partition held-out target members into flagged and unflagged groups. Eligible withholds contain at least ten held-out members and at least three in each group. Recognition after random confirmation is averaged over three training draws and then over withholds. The comparison includes 50 Gravity $\operatorname { S p y } .$ 23 ECG, 29 lightcurve, 159 CIFAR-100 and 113 Food-101 withholds. This exploratory analysis preceded and motivated the confirmation-policy experiments.

## Confirmation selection and hand-off

Candidates are members of the withheld class available for promotion: training and validation members in Gravity Spy, training images in CIFAR-100, and folds 1–8 in PTB-XL. Policies select the highest-energy candidates, the lowest-energy candidates, or random candidates. A stochastic detector policy samples n members from the 2n highest-energy candidates. Random policies average three draws.

The selection experiment uses all twenty-two Gravity Spy classes through twenty confirmations; thirteen CIFAR-100 classes with stored zero-label detectability between 0.1 and 0.9 through twenty confirmations; and thirty-five ECG withholds involving eleven classes through forty confirmations. CIFAR-100 eligibility was fixed before the selection runs. ECG uses seven geometry-selected six-class taxonomies because no class in its full taxonomy has zero-label detectability above 0.1. Their construction is described in Supplementary Sec. S1.

Hand-off policies first confirm the detector's two highest-ranked members. At budgets 2, 5 and 10, the refitted classifier selects the next confirmations from the remaining pool by sampling among its alerts, by highest promoted-class probability, or by probability closest to its threshold. These experiments use eight Gravity Spy classes with detectability between 0.1 and 0.9, the thirteen CIFAR-100 classes and the thirty-five ECG withholds.

The mixed-stream test withholds a second class, drawn at random among classes with at least fifty candidates, and moves 20% of each known class's training members into the stream. Confirmations of the second class or of known classes consume budget without entering training. The detector confirms the highest-energy stream events; the alerts policy samples among stream events above the 1% energy threshold; the hand-off policy follows the detector until at least two target members are confirmed and then samples among the promoted classifier's alerts. Budgets are 5, 10, 20 and 40 confirmations, with three draws per randomised policy, for the eight Gravity Spy hand-off classes and the thirteen CIFAR-100 classes.

Equivalent confirmation counts are interpolated in log n from the running maximum of the mean detectorselected curve. They compare equal recognition levels and are approximate.

## Neighbour intervention

Gravity Spy neighbours are ranked on training data by directional isolation (Supplementary Sec. S2). Eligible background classes have at least 100 training and 40 calibration members. For each target, the two six-class taxonomies contain the target, two shared classes at the median rank, and either its three nearest or three farthest remaining candidates. Background classes are subsampled to equal training and calibration counts; shared subsamples and confirmed-member indices are identical across arms.

The comparison uses four replicate draws. We compute $N _ { 8 0 }$ separately in each replicate and average within each class before taking the median across classes. Protocol thresholds are recomputed in each arm; a diagnostic comparison uses identical calibration members from the shared classes.

In CIFAR-100, the near arm contains the target's four semantic siblings, the far arm four classes from different superclasses, and both share five further background classes (K = 10). Training counts and confirmation draws are paired. The reported endpoint is best recognition through forty confirmations.

## Statistics and reproducibility

Confirmation-policy differences are summarised at class level. Each ECG class contributes its median over the sub-taxonomies containing it. Reported 95% intervals are percentile bootstrap intervals over classes; they do not include uncertainty from threshold estimation or new confirmation draws. Gravity Spy results at five, ten and twenty confirmations average training seeds 0, 1 and 2. Seed sensitivity and failed criteria are reported in Supplementary Sec. S1.

Neighbour-swap sign tests are paired, one-sided and censoring-aware, excluding ties. Spearman correlation describes the relation between isolation changes and cost changes. No multiple-comparison correction was applied. Sample sizes follow benchmark eligibility rather than a power calculation.

The member-level analysis was exploratory. Predictions for subsequent analyses were recorded in dated scripts before their corresponding measurements, without an external registry. Earlier undated file timestamps do not establish prior specification. Declaring scripts, digests and the full ledger, including failed predictions, accompany the code.

Runs used Python 3.12, LightGBM 4.6.0, scikit-learn 1.8.0, NumPy 2.3.5, SciPy 1.17.1, PyTorch 2.10.0 and torchvision 0.25.0. Feature extraction used a GPU and subsequent experiments used CPU nodes on NCSA Delta. Exact seeds, preprocessing and run configurations are recorded in the scripts.

## DATA AVAILABILITY

The transient benchmark is released with the companion paper [6|at https://github.com/applecider-ml/ f1are. Gravity Spy [24], CIFAR-100 [26], Food-101 [28] and PTB-XL [25] are public; PTB-XL is available at https://physionet.org/content/ptb-xl/. Derived features, embeddings and per-class outputs accompany the code.

## CODE AVAILABILITY

Feature extraction, withholding and promotion, confirmation policies, controls, prediction records and figuregeneration scripts are available at https://github. com/ asasli/label-cost.

## ACKNOWLEDGMENTS

The authors acknowledge support from the National Science Foundation through grants PHY-2117997, PHY-2308862 and PHY-2409481. This work uses observations from the Zwicky Transient Facility and its Bright Transient Survey; Gravity Spy classifications contributed by Zooniverse volunteers and LIGO detectorcharacterisation data; and PTB-XL data made available through PhysioNet by the Physikalisch-Technische Bundesanstalt.

During manuscript preparation, A.S. used Claude (Opus 5.5) to improve English language, clarity, and organization. The authors reviewed and edited all AI-

[1] K. L. Malanchev et al., Anomaly detection in the Zwicky Transient Facility DR3, Monthly Notices of the Royal Astronomical Society 502, 5147 (2021).

[2] M. Lochner and B. A. Bassett, Astronomaly: Personalised active anomaly detection in astronomical data, Astronomy and Computing 36, 100481 (2021)

[3] E. E. O. Ishida, M. V. Kornilov, K. L. Malanchev, et al., Active anomaly detection for time-domain discoveries, Astronomy & Astrophysics 650, A195 (2021).

[4] S. Soni et al., Discovering features in gravitational-wave data through detector characterization, citizen science and machine learning, Classical and Quantum Gravity 38, 195016 (2021).

[5] A. Sasli, M. Xu, A. Junell, et al., Catching disguised transients with astranet: Anomaly-aware spectroscopic classification and conformal calibration, arXiv:2607.08044 (2026).

[6] A. Sasli, F. Fontinele Nunes, X. Zhang, D. Kourtesis, S. Nikmehr, A. Junell, J. Sollerman, A. Le Calloch, S. Sharma Chaudhary, S. Maharjan, M. Xu, A. Raghuvanshi, S. L. Groom, J. Castaneda Jaimes, J. Purdum, R. Smith, M. M. Kasliwal, M. J. Graham, A. A. Mahabal, and M. Coughlin, Photometric classification and outof-taxonomy detection of ZTF transients with FLARE, arXiv preprint arXiv:2610.30656 (2026).

[7] W. J. Scheirer, A. de Rezende Rocha, A. Sapkota, and T. E. Boult, Toward open set recognition, IEEE Transactions on Pattern Analysis and Machine Intelligence 35, 1757 (2013).

[8] A. Bendale and T. E. Boult, Towards open set deep networks, in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016) pp. 1563-1572.

[9] D. Hendrycks and K. Gimpel, A baseline for detecting misclassified and out-of-distribution examples in neural networks, in International Conference on Learning Representations (2017).

[10] W. Liu, X. Wang, J. D. Owens, and Y. Li, Energy-based out-of-distribution detection, in Advances in Neural Information Processing Systems, Vol. 33 (2020) pp. 21464– 21475.

[11] J. Yang et al., OpenOOD: Benchmarking generalized outof-distribution detection, in Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track (2022).

[12] S. Vaze, K. Han, A. Vedaldi, and A. Zisserman, Open-set recognition: A good closed-set classifier is all you need?, in International Conference on Learning Representations (ICLR) (2022).

[13] Y. Wang, Q. Yao, J. T. Kwok, and L. M. Ni, Generalizing from a few examples: A survey on few-shot learning, ACM Computing Surveys 53, 63 (2020).

[14] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, iCaRL: Incremental classifier and representation learning, in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017) pp. 5533–5542.

[15] K. Han, A. Vedaldi, and A. Zisserman, Learning to discover novel visual categories via deep transfer clustering, in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2019) pp. 8401–8409.

[16] S. Vaze, K. Han, A. Vedaldi, and A. Zisserman, Generalized category discovery, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022) pp. 7492–7501.

[17] T. Viering and M. Loog, The shape of learning curves: A review, IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 7799 (2023).

[18] B. Settles, Active learning literature survey, in Computer Sciences Technical Report 1648 (University of Wisconsin–Madison, 2009).

[19] S. Dasgupta and D. Hsu, Hierarchical sampling for active learning, in Proceedings of the 25th International Conference on Machine Learning (2008) pp. 208–215.

[20] K.-P. Ning, X. Zhao, Y. Li, and S.-J. Huang, Active learning for open-set annotation, arXiv preprint arXiv:2201.06758 (2022).

[21] S. Schmidt, L. Schenk, L. Schwinn, and S. Günnemann, Joint out-of-distribution filtering and data discovery active learning, arXiv preprint arXiv:2503.02491 (2025).

[22] C. Fremling, A. A. Miller, Y. Sharma, et al., The zwicky transient facility bright transient survey. i. spectroscopic classification and the redshift completeness of local galaxy catalogs, The Astrophysical Journal 895, 32 (2020).

[23] D. A. Perley, C. Fremling, J. Sollerman, et al., The zwicky transient facility bright transient survey. ii. a public statistical sample for exploring supernova demographics, The Astrophysical Journal 904, 35 (2020).

[24] M. Zevin et al., Gravity spy: integrating advanced ligo detector characterization, machine learning, and citizen science, Classical and Quantum Gravity 34, 064003 (2017).

[25] P. Wagner, N. Strodthoff, R.-D. Bousseljot, D. Kreiseler, F. I. Lunze, W. Samek, and T. Schaeffter, PTB-XL, a large publicly available electrocardiography dataset, Scientific Data 7, 154 (2020).

[26] A. Krizhevsky, Learning Multiple Layers of Features from Tiny Images, Tech. Rep. (University of Toronto, 2009).

[27] K. He, X. Zhang, S. Ren, and J. Sun, Deep residual learning for image recognition, in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2016) pp. 770–778.

[28] L. Bossard, M. Guillaumin, and L. Van Gool, Food-101 – mining discriminative components with random forests, in European Conference on Computer Vision (2014).

[29] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, LightGBM: A highly efficient gradient boosting decision tree, in Advances in Neural Information Processing Systems, Vol. 30 (2017).

# Supplementary Information

## S1. CONFIRMATION-POLICY ROBUSTNESS

## A. ECG taxonomy selection

ECG sub-taxonomies were constructed without using recognition outcomes. Starting from each pathological seed class, we greedily added the class maximising the minimum pairwise directional isolation until five pathological subclasses joined the normal class. We retained the seven distinct sets with the highest minimum isolation and withheld each pathological member in turn. The resulting thirty-five withholds involve eleven distinct classes. Class-level summaries keep these repeated appearances together.

This design conditions the ECG result on geometrically separated taxonomies. The originally planned intermediate-detectability comparison could not be evaluated in the full ECG taxonomy because no class had detectability above 0.1.

## B. Training seeds

Gravity Spy's feature and row subsampling make its learner sensitive to the training seed. With seeds 1 and 2, detector-selected confirmation gives lower recognition than random confirmation after ten labels for seventeen of twenty-two classes in each case. Mean recognitions are 0.46 against 0.57 and 0.44 against 0.62, respectively.

Averaging seeds 0, 1 and 2, the difference is -0.17 (95% class-bootstrap interval -0.24 to -0.11), negative for twenty of twenty-two classes. The predeclared criterion of at least eighteen negative classes in each additional seed was not met. The detector arm's acrossseed standard deviation for a single class is 0.15, exceeding the proposed stability criterion of half the mean gap (0.09). The average penalty is therefore more stable than individual-class estimates.

The ECG trees use neither feature nor row subsampling and give identical predictions across seeds. The CIFAR-100 linear head solves a convex objective from a fixed start.

## C. Selection controls

Sampling n confirmations at random among the 2n highest-ranked candidates retains the penalty relative to unrestricted random sampling in all eight Gravity Spy classes tested. This argues against the effect being confined to the most extreme candidates.

The least anomalous policy did not consistently outperform random selection. The prediction that the detector penalty would shrink from five to twenty confirmations also failed in Gravity Spy, where it grew. Persistence over that interval replicated in CIFAR-100 but not in the ECG sub-taxonomies. These results limit any universal claim about how the penalty changes with budget.

Supplementary Table 1. Gravity Spy neighbour-swap controls on deduplicated data. Counts exclude ties. Costs are computed per replicate and averaged within class.
<table><tr><td>Comparison Result</td></tr><tr><td>Interpolated  $N _ { 8 0 } .$  cheaper : dearer 18:2</td></tr><tr><td>Median  $N _ { 8 0 } .$  near → far 4.3 → 1.6</td></tr><tr><td>Replicate cheaper : dearer counts 15 :5, 19 :1, 17 :1, 17 :3</td></tr><tr><td>First measured crossing, cheaper : dearer 19:1</td></tr><tr><td>Median first crossing, near → far 6.2 → 2.2</td></tr><tr><td>Two frequent far classes excluded 20:0</td></tr></table>

## S2. NEIGHBOUR-SWAP CONTROLS

## A. Neighbour ranking

Directional isolation is computed from standardised training features reduced to fifty principal components. For classes c and $j ,$ with means $\mu _ { c } , \mu _ { j }$ and Ledoit-Wolf covariance estimates $S _ { c } , S _ { j }$ , define

$$
S _ { \mathrm { p o o l e d } } = { \textstyle { \frac { 1 } { 2 } } } ( S _ { c } + S _ { j } ) , \qquad w = S _ { \mathrm { p o o l e d } } ^ { - 1 } ( \mu _ { c } - \mu _ { j } ) .\tag{2}
$$

The ranking uses

$$
\hat { I } _ { \mathrm { d i r } } ( c , j ) = \frac { | w ^ { \top } ( \mu _ { c } - \mu _ { j } ) | } { \sqrt { w ^ { \top } S _ { c } w + w ^ { \top } S _ { j } w } } .\tag{3}
$$

Training members of the withheld class are used to define this controlled intervention. It is an explanatory benchmark, not a geometry estimate available before confirming that class.

## B. Replicates, thresholds and cost definition

With the paper's interpolated cost, distant neighbours are cheaper for eighteen classes and dearer for two, with one tie. The direction persists in each of the four replicates. Excluding the two classes most frequently chosen as distant neighbours also retains the effect.

Using the first measured budget reaching 80%, rather than interpolation, gives the same direction with or without the running maximum. Computing cost from each class's mean curve instead of averaging replicate costs gives medians 4.19 and 1.33, with lower cost for seventeen of twenty-one classes.

With thresholds estimated from identical shared-class calibration members, recognition after five confirmations is higher among distant neighbours for sixteen classes and lower for three. The effect therefore remains under a common calibration reference.

Supplementary Table 2. Mean recognition at a 1% false-alarm rate for the hand-off cohorts. "Least" selects the lowest anomaly scores; "alerts", "top" and "margin" follow the two detector-selected confirmations. Randomised policies average three draws. The thirty-five ECG withholds involve eleven classes and are not independent class replicates.
<table><tr><td>System</td><td>n</td><td>Random</td><td>Detector</td><td>Least</td><td>Alerts</td><td>Top</td><td>Margin</td></tr><tr><td>Gravity Spy (8 classes)</td><td>5</td><td>0.42</td><td>0.30</td><td>0.48</td><td>0.42</td><td>0.35</td><td>0.42</td></tr><tr><td></td><td>10</td><td>0.72</td><td>0.43</td><td>0.66</td><td>0.61</td><td>0.42</td><td>0.53</td></tr><tr><td></td><td>20</td><td>0.89</td><td>0.67</td><td>0.79</td><td>0.82</td><td>0.73</td><td>0.83</td></tr><tr><td>CIFAR-100 (13 classes)</td><td>5</td><td>0.75</td><td>0.68</td><td>0.69</td><td>0.74</td><td>0.67</td><td>0.75</td></tr><tr><td></td><td>10</td><td>0.81</td><td>0.68</td><td>0.75</td><td>0.77</td><td>0.68</td><td>0.80</td></tr><tr><td></td><td>20</td><td>0.82</td><td>0.72</td><td>0.77</td><td>0.80</td><td>0.70</td><td>0.83</td></tr><tr><td>ECG (35 withholds)</td><td>5</td><td>0.46</td><td>0.35</td><td>0.44</td><td>0.34</td><td>0.30</td><td>0.37</td></tr><tr><td></td><td>10</td><td>0.49</td><td>0.43</td><td>0.51</td><td>0.43</td><td>0.38</td><td>0.49</td></tr><tr><td></td><td>20</td><td>0.56</td><td>0.49</td><td>0.56</td><td>0.53</td><td>0.45</td><td>0.57</td></tr></table>

The increase in directional isolation does not predict the size of the cost reduction across classes $( \rho = - 0 . 0 4 )$ Predictions of a corresponding, consistently weaker response in the anomaly channel were not all met. Energy detectability rises for twelve of twenty-one classes on replicate means. The intervention supports an effect of taxonomy composition on learning, without establishing a scalar geometric predictor or an absence of change in detectability.

## S3. HAND-OFF POLICIES AND THEIR LIMITS

The first two confirmations are detector-selected. Subsequent choices sample among the promoted classifier's alerts, take its highest-probability members, or take members closest to its threshold. All policies draw only from the withheld class's candidate pool.

Table 2 reports the results of this investigation. At ten confirmations, sampling among classifier alerts beats detector-only selection in seven of eight Gravity Spy classes, all thirteen CIFAR-100 classes and eighteen of thirty-five ECG withholds. The criterion of improvement in at least two thirds of the withholds was not met in the ECGs.

At twenty confirmations, highest-probability selection performs worse than sampling among alerts in five of eight Gravity Spy classes, thirteen of thirteen CIFAR-100 classes and twenty-seven of thirty-five ECG withholds. Margin selection approaches the random-policy mean in CIFAR-100 and ECG, but the corresponding predeclared random-level criterion passed only in ECG. These comparisons do not establish equivalence between policies or a universally effective hand-off rule.

## S4. MIXED CANDIDATE STREAMS

The stream for each target class contains its candidate members, the candidate members of a second withheld class and 20% of the training members of every known class. Only confirmed target members enter training and every confirmation counts against the budget.

Supplementary Table 3. Mixed-stream confirmation. Mean recognition of the target class at a 1% false-alarm rate, with the mean number of target members among the confirmations in parentheses. Randomised policies average three draws.
<table><tr><td>System</td><td>B</td><td>Detector</td><td>Alerts</td><td>Hand-off</td></tr><tr><td>Gravity Spy</td><td>5</td><td>0.23 (3.8)</td><td>0.31 (3.1)</td><td>0.23 (3.8)</td></tr><tr><td>(8 classes)</td><td>10</td><td>0.42 (7.8)</td><td>0.42 (6.2)</td><td>0.49 (7.0)</td></tr><tr><td></td><td>20</td><td>0.57 (15.6)</td><td>0.66 (12.9)</td><td>0.68 (14.4)</td></tr><tr><td>CIFAR-100</td><td>40</td><td>0.74 (29.1)</td><td>0.79 (25.2)</td><td>0.88 (31.3)</td></tr><tr><td>(13 classes)</td><td>5</td><td>0.54 (3.2)</td><td>0.54 (1.7)</td><td>0.54 (3.2)</td></tr><tr><td></td><td>10</td><td>0.67 (5.9)</td><td>0.67 (3.4)</td><td>0.72 (6.5)</td></tr><tr><td></td><td>20</td><td>0.70 (10.8)</td><td>0.72 (7.2)</td><td>0.77 (13.7)</td></tr><tr><td></td><td>40</td><td>0.72 (20.1)</td><td>0.74 (14.5)</td><td>0.83 (28.9)</td></tr></table>

At twenty confirmations, the detector confirms more target members than the alerts policy for six of eight Gravity Spy classes and eleven of thirteen CIFAR-100 classes (Table 3). The alerts policy gives recognition at least as high as the detector for eight of eight and eight of thirteen classes, and hand-off exceeds the detector for seven of eight and eleven of thirteen. All four predeclared criteria were met. In CIFAR-100 the difference between the detector and alerts policies is small, and most of the gain comes from hand-off.