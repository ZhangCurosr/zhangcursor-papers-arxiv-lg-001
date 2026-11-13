# Few-Shot Cross-Subject Neural–Image Retrieval via Stimulus-Anchored Latent Bias Calibration

Xizhang Yao, Li Zhang, Gan Huang, and Zhen Liang

Abstract—In cross-subject neural–image retrieval, comparing the mean of a few stimulus-matched target observations with the global source mean conflates subject-specific representation shifts with stimulus-composition effects. To address this problem, we propose Stimulus-Anchored Latent Bias Calibration (SA-LBC), which estimates the first-order shift under fixed preprocessing and a frozen source-trained representation by averaging residuals between target responses and same-stimulus source-population anchors, then shrinks the correction using source-derived uncertainty. Under uniform sampling without replacement from a finite stimulus population, we derive exact finite-population MSE expressions for the unshrunk anchored and global-mean estimators, including residual–anchor cross-covariance, and establish a necessary and sufficient condition for anchoring to reduce MSE relative to global-mean estimation. In a 10-subject leave-one-subject-out evaluation on THINGS-EEG2, SA-LBC increases mean reciprocal rank (MRR) for the Attention–CLIP ViT-B/32 configuration from 0.1867 without calibration to 0.2013 and 0.2336 with M = 2 and M = 16 stimulus-level calibration samples, respectively. Across six neural–visual configurations and all calibration budgets, stimulus-matched anchors yield higher MRR than anchors with shuffled stimulus identities, but retrieval gains over no calibration remain representation-dependent. At (M=2), shrinkage improves MRR over the unshrunk anchored estimator in all six configurations, but its effects at larger budgets vary across representations; the relative ranking of translation and orientation corrections also depends on regularization strength. Thus, lower shift-estimation MSE does not guarantee higher retrieval performance. The source code is publicly available at: https://github.com/Xizhang-Yao/SA-LBC.

Index Terms—Few-Shot Calibration, Cross-Subject Neural Decoding, Neural–Image Retrieval, Latent Bias Estimation, Geometric Calibration, Electroencephalography (EEG), Magnetoencephalography (MEG)

## 1 INTRODUCTION

Non-invasive neural–image retrieval aims to identify visual content from brain activity by aligning neural and visual representations in a shared space [1], [2], [3]. Recent EEG- and MEG-based methods have substantially advanced neural–image retrieval under subject-dependent protocols, with models trained and evaluated separately for each participant [3], [4], [5]. However, the learned neural–visual correspondence often transfers poorly to unseen subjects, partly because neural responses exhibit systematic intersubject variability [6], [7]. Reliable transfer of such correspondence to new subjects therefore remains a key obstacle to scalable cross-subject deployment of neural–image retrieval.

Zero-shot cross-subject generalization avoids targetspecific calibration but cannot use target observations to correct individual mismatch [8], [9], [10]. In contrast, adaptation with substantial new-subject data can accommodate individual variability but incurs additional acquisition and adaptation costs [11], [12]. A complementary setting uses a few stimulus-matched target observations to estimate a latent correction before testing, conditional on fixed preprocessing and a frozen source-trained representation. Here, shared stimulus identities link target observations to source responses elicited by the same stimuli, providing a reference for estimating subject mismatch [11], [13], [14]. These correspondences specify what to compare, but not which cross-subject corrections can be estimated reliably from so few observations.

Candidate corrections range from translation and scaling to orientation changes and nonlinear mappings [14], [15]. Positive global isotropic scaling leaves cosine-based retrieval rankings unchanged. Translation and orientation offer two contrasting cases: translation uses centroid differences, whereas orientation uses the geometry of matched, centered responses [7], [15]. With M matched observations in a d-dimensional space, the centered cross-covariance has rank at most M − 1, limiting its data-supported modes when $M \ \ll \ d$ . Translation avoids estimating this crosscovariance, but a small target-sample mean and a global source mean need not represent the same stimuli. Their difference therefore reflects sampled-stimulus composition as well as subject mismatch, creating stimulus-composition uncertainty even for first-order calibration.

To address this uncertainty, we introduce Stimulus-Anchored Latent Bias Calibration (SA-LBC), which estimates a target-specific first-order shift using stimulusmatched source-population anchors (Fig. 1). SA-LBC compares the target calibration mean with the source-population mean for the same stimulus set, equivalently averaging the matched residuals. This removes the explicit sampledstimulus composition term incurred in global-mean matching, but finite-sample residual variation remains. SA-LBC therefore applies reliability-aware shrinkage, attenuating poorly supported shifts rather than always applying them in full. To isolate anchoring from shrinkage, we analyze the unshrunk anchored and global-mean estimators under uniform sampling without replacement from a finite stimulus population. This yields exact mean-squared errors and a necessary and sufficient condition for anchoring to reduce estimation error. Finally, same-budget comparisons of translation, centroid-preserving regularized orientation, and their combination test whether paired geometry adds retrieval value.

Under leave-one-subject-out evaluation with sourceonly model selection and the stated preprocessing, SA-LBC improves MRR from 0.1867 to 0.2013 with $M = 2$ stimuluslevel calibration averages and to 0.2336 with $M = 1 6$ in the Attention–CLIP ViT-B/32 configuration on THINGS-EEG2. Across six neural–visual configurations, mean relative discrepancies between analytic and empirical MSE are below 1.7% for both unshrunk estimators. Matched source references outperform references constructed with shuffled stimulus identities in all six configurations, but calibration does not consistently improve on leaving the representation unchanged. The benefit of shrinkage varies across representations, and geometric ordering can reverse with regularization strength. These results distinguish estimating the shift, controlling its strength, and establishing its retrieval utility.

Our contributions are threefold:

We introduce SA-LBC, a few-shot pre-test latentcalibration method that combines stimulus-matched source-population references with source-derived shrinkage in a frozen neural–visual representation.

We derive exact finite-population MSE expressions for the unshrunk anchored and global-mean estimators, retaining residual–anchor covariance to establish when anchoring lowers estimation error.

We characterize the benefits and limits of sourcereference matching, shrinkage, and regularized geometric correction through same-budget comparisons across six neural–visual configurations.

## 2 RELATED WORK

Prior work relevant to cross-subject neural–image retrieval can be understood through two questions: how neural– visual representations are learned, and what target-subject information is available when transferring them to a new individual. We therefore review neural–image representation learning, cross-subject alignment and source-side generalization, unlabeled target-side calibration, and limited-data cross-subject adaptation.

## 2.1 Neural–Image Retrieval

Early visual brain-decoding studies established that perceptual content can be identified or reconstructed from neural activity [16], [17], [18], [19]. Later work adopted multimodal representation learning, embedding neural activity and visual content into directly comparable feature spaces [2], [20], [21], [22]. Recent EEG-to-image retrieval methods extend this formulation by aligning EEG with pretrained visual representations in a shared embedding space for similaritybased retrieval [3], [23], [24], [25].

One recent direction refines where and at what level neural and visual information should be matched, moving from a single global embedding toward multi-granularity, hierarchical, regional, and structured correspondences [4], [5], [26], [27]. A second direction enriches what visual information is used for alignment, incorporating semantic priors, complementary visual teachers, and saliency- or object-aware supervision [28], [29], [30]. These advances improve how neural and visual information is represented and aligned, but do not by themselves determine how the resulting representation should be transferred to an unseen subject.

## 2.2 Cross-Subject Alignment and Source-Side Generalization

Cross-subject variability has commonly been addressed in two ways: by explicitly aligning subject-specific representations or by learning representations that generalize across subjects during source training. The first strategy estimates mappings between subject-specific representational spaces using cross-subject correspondence or shared structure [13], [15]. Geometric formulations make such mappings explicit through changes in location, scale, and orientation [15]. Related alignment methods also standardize EEG trials in Euclidean space [31] or match source and target feature covariances [32]. The second strategy shifts the burden to source training, seeking representations that transfer across individuals through feature alignment, invariance learning, disentanglement, and related domaingeneralization objectives [6], [33]. This strategy has been pursued through subject-aware normalization, adversarially learned subject-invariant embeddings, and cross-subject reconstruction frameworks designed for unseen individuals [8], [9], [10].

Together, these approaches reduce subject variability either by explicitly aligning subject-specific representations or by learning representations that generalize across subjects. Yet dedicated benchmarks and empirical studies still show a substantial gap between cross-subject and within-subject decoding [6], [34], [35]. This motivates a complementary deployment-time question: given a trained representation, how can target-subject observations be used to correct the mismatch that remains?

## 2.3 Unlabeled Target-Side Calibration

Once a neural–visual representation has been trained, unlabeled target-side information can still be used to correct unseen-subject mismatch at deployment [7], [36]. SATTC follows this strategy by keeping the EEG and image encoders fixed and recalibrating retrieval from the unlabeled target test batch through statistical normalization and structure-aware similarity correction [36]. SCORE takes a more explicit geometric view, recovering target coordinates at deployment from unlabeled target observations through moment matching, pseudo-landmark discovery, and regularized orthogonal alignment [7].

Their target information, however, takes different forms: SATTC is naturally transductive over the unlabeled test batch, whereas SCORE can estimate a coordinate map from unlabeled target observations and apply the frozen map to a disjoint unseen gallery [7]. Both methods infer target mismatch from unlabeled target-side structure, without requiring explicit stimulus correspondence across subjects.

(a) Cross-Subject EEG–Image Retrieval

![](images/fe8a46c4b02fb7c433f17364e72586dc1505e2bc6ecc5e1343844ce1f45d3e91.jpg)

(b) Stimulus-Composition Effect  
![](images/512a27c8f4c957fea508063b34c3c6fc8cc3c7fe945ca55ea0ef197845c79a5d.jpg)  
Global Source Mean Composition Term Matched Source Mean

$$
\overline { { { \pmb { a } } } } _ { \mathrm { M } } = \overline { { { \pmb { p } } } } _ { \mathrm { M } } - { \pmb { \mu } } _ { \mathrm { P } } \qquad { \widehat { \pmb { b } } } _ { \mathrm { T } } ^ { \mathrm { G M } } = \widehat { \pmb { b } } _ { \mathrm { T } } + \overline { { { \pmb { a } } } } _ { \mathrm { M } }
$$

![](images/3ae8c0de58401b626d0fb7f82f0600f8220384967fd24b601b23f9e68d5be97c.jpg)

(d) Reliability-Aware Correction  
![](images/30c6e7079bc48126c547c52bc3f88627ee0f5a4ae99e1100caa32bb6a65cab8e.jpg)  
Fig. 1. Overview of Stimulus-Anchored Latent Bias Calibration (SA-LBC), illustrated for EEG–image retrieval after fixed preprocessing. (a) Source only representation training is followed by calibration from M pre-test target observations and retrieval on concept-disjoint TEST stimuli. (b) The global-mean and anchored estimates differ by the sampled-stimulus composition term $\bar { \mathbf { a } } _ { M } = \bar { \mathbf { p } } _ { M } - \pmb { \mu } _ { P } . \mathbf { \tau } ( \mathsf { c } )$ Source embeddings are averaged across subjects for each stimulus to form population anchors; averaging target–anchor residuals yields the shift estimate bb . (d) Source-only leave-one-out (LOO) residual uncertainty, the sampling budget, and the estimated shift energy determine the plug-in shrinkage factor β. The fixed correction is applied before $\ell _ { 2 }$ normalization and cosine ranking. Stimulus IDs select the source anchors; latent calibration uses no image pixels or visual features, target TEST statistics, or retrieval feedback, and does not update the representation. In THINGS-EEG2, each calibration observation averages four responses, and participant-specific TRAIN whitening uses additional target recordings outside the M-observation budget.

A different regime arises when a small set of stimulusmatched observations is available before evaluation: each target observation can then be directly related to the source representation of the same stimulus. This correspondence changes the calibration problem from inferring mismatch from target-side structure to estimating it from paired crosssubject evidence. Such settings motivate limited-data crosssubject adaptation methods, which we review next.

## 2.4 Limited-Data Cross-Subject Adaptation

Limited-data cross-subject adaptation differs in where the target-specific correction is introduced. One line estimates an explicit transformation from the new subject to a source or common representational space, using shared or related stimuli to establish cross-subject correspondence [11], [37]. Another adapts the pretrained decoding model or a lightweight subject-specific module using limited target data, as in MindShot [38], ENIGMA [39], and MindAdapter [14].

Together, these studies show that limited target data can support cross-subject transfer through functional mapping, model adaptation, or residual correction. They do not, however, directly ask which subject-specific correction can be estimated reliably under an extremely small, fixed budget of stimulus-matched pre-test observations. Our work treats this question as an explicit estimation problem.

## 3 METHOD

## 3.1 Few-Shot Pre-Test Calibration Problem

We consider K source subjects $s = \{ 1 , \ldots , K \}$ and an unseen target subject T that is excluded from source-model training. Let $f _ { \theta }$ be a pretrained frozen neural encoder, and

let $\mathbf { x } _ { s , i }$ and ${ \bf x } _ { T , i }$ denote the preprocessed neural observations of source subject s and target subject T for stimulus $i ,$ respectively. Their latent representations are

$$
\begin{array} { r } { \mathbf { z } _ { s , i } = f _ { \theta } ( \mathbf { x } _ { s , i } ) , \qquad \mathbf { z } _ { T , i } = f _ { \theta } ( \mathbf { x } _ { T , i } ) , \qquad \mathbf { z } _ { s , i } , \mathbf { z } _ { T , i } \in \mathbb { R } ^ { d } . } \end{array}\tag{1}
$$

Let $g _ { \phi }$ be the frozen visual encoder, with candidate-image representation

$$
\mathbf { v } _ { j } = g _ { \phi } ( \mathbf { I } _ { j } ) \in \mathbb { R } ^ { d } .\tag{2}
$$

After calibration, candidate images are ranked by cosine similarity between the calibrated target representation and the frozen visual representations.

The budget below applies to latent calibration after preprocessing. In THINGS-EEG2, whitening is participant specific and uses each participant’s TRAIN recordings, including those of the held-out target. These preprocessing observations are outside the M-observation calibration budget. LOSO excludes the target from representation fitting and checkpoint selection, not from participant-specific preprocessing.

Let the pre-test calibration pool for target subject $T$ contain $N _ { T }$ stimuli. Only a subset

$$
\mathcal { C } _ { M } \subseteq \{ 1 , \dots , N _ { T } \} , \qquad | \mathcal { C } _ { M } | = M , \qquad M \ll N _ { T }\tag{3}
$$

is available for estimating the latent correction. For each calibration observation, the stimulus identity is used only to index the corresponding source-population neural anchor; the calibration image, its visual embedding, and semantic category labels are not used. The calibration pool is conceptdisjoint from the final TEST set, and calibration uses neither target TEST statistics nor retrieval feedback and performs no target-side parameter updates. The calibration task is to estimate a subject-specific latent correction from $\mathcal { C } _ { M }$ and apply it to future target observations while keeping the neural–visual representation fixed (Fig. 1(a)).

## 3.2 Stimulus-Composition Uncertainty in First-Order Shift Estimation

We first consider a first-order correction that aligns the target calibration centroid with the source-population centroid. The target calibration centroid is

$$
\bar { \mathbf { z } } _ { T , M } = \frac { 1 } { M } \sum _ { i \in \mathcal { C } _ { M } } \mathbf { z } _ { T , i } .\tag{4}
$$

The source-population centroid over the same $N _ { T }$ -stimulus population is

$$
\pmb { \mu } _ { P } = \frac { 1 } { K N _ { T } } \sum _ { s = 1 } ^ { K } \sum _ { i = 1 } ^ { N _ { T } } \mathbf { z } _ { s , i } .\tag{5}
$$

Global-mean matching therefore estimates the targetspecific shift as

$$
\widehat { \mathbf { b } } _ { T } ^ { \mathrm { G M } } = \bar { \mathbf { z } } _ { T , M } - \pmb { \mu } _ { P } .\tag{6}
$$

Because $\bar { \mathbf { z } } _ { T , M }$ averages only the sampled calibration stimuli, it depends on both the target subject and the composition of $\mathcal { C } _ { M }$ . Consequently, two calibration subsets of the same size can yield different global-mean estimates for the same target subject because they contain different stimuli. We refer to this finite-sample contribution to firstorder estimation error as stimulus-composition uncertainty.

This formulation isolates a finite-sample estimation problem within first-order calibration; it does not assume that all cross-subject mismatch is additive.

## 3.3 Stimulus-Matched Source-Population Anchors

To match the sampled stimulus composition on the source side, we construct a stimulus-specific source-population anchor by averaging the frozen representations of stimulus i across source subjects,

$$
{ \bf p } _ { i } = \frac { 1 } { K } \sum _ { s = 1 } ^ { K } { \bf z } _ { s , i } .\tag{7}
$$

The resulting anchor is the source-population mean latent representation of stimulus i. The global source-population centroid is therefore

$$
{ \boldsymbol { \mu } } _ { P } = { \frac { 1 } { N _ { T } } } \sum _ { i = 1 } ^ { N _ { T } } \mathbf { p } _ { i } .\tag{8}
$$

For the sampled calibration stimuli, the matched sourcepopulation centroid is

$$
\bar { \mathbf { p } } _ { M } = \frac { 1 } { M } \sum _ { i \in \mathcal { C } _ { M } } \mathbf { p } _ { i } .\tag{9}
$$

Because $\bar { \mathbf { z } } _ { T , M }$ and p¯<sub>M</sub> average exactly the same stimulus identities, their difference cancels the explicit sourcepopulation composition term introduced by global-mean matching (Fig. 1(b)).

## 3.4 Stimulus-Anchored Latent Bias Calibration

With a source-population anchor matched to each calibration stimulus, we define the stimulus-wise residual

$$
\mathbf { r } _ { T , i } = \mathbf { z } _ { T , i } - \mathbf { p } _ { i } .\tag{10}
$$

SA-LBC estimates the target-specific first-order shift by averaging these residuals (Fig. 1(c)),

$$
\begin{array} { l } { { \displaystyle { \widehat { \bf b } } _ { T } ^ { \mathrm { S A } } = \frac { 1 } { M } \sum _ { i \in { \mathcal C } _ { M } } \left( { \bf z } _ { T , i } - { \bf p } _ { i } \right) } \ ~ } \\ { { \displaystyle ~ = { \bar { \bf z } } _ { T , M } - { \bar { \bf p } } _ { M } } . } \end{array}\tag{11}
$$

The translation estimate depends on the target and anchor means, not on the pairing order within the selected sets. Any permutation of the selected anchors leaves the estimate unchanged. Stimulus identities are therefore used to construct a source reference with the same stimulus composition as the target calibration set. Replacing the global source centroid $\pmb { \mu } _ { P }$ with the subset-matched centroid $\bar { \mathbf p } _ { M }$ removes the explicit source-population composition term identified above. The estimator nevertheless remains variable across finite calibration subsets because the matched residuals themselves vary across stimuli. Henceforth, we write $\widehat { \mathbf { b } } _ { T } \equiv \widehat { \mathbf { b } } _ { T } ^ { \mathrm { S A } }$

## 3.5 Reliability-Aware Shrinkage

The correction below is a source-derived plug-in shrinkage rule, related in form to classical shrinkage estimation [40]. To quantify the remaining finite-sample uncertainty without target TEST information, we treat each source subject in turn as a pseudo-target. For source subject s, we form a subjectwise leave-one-out (LOO) anchor by excluding s from the source population,

$$
{ \bf p } _ { i } ^ { ( - s ) } = \frac { 1 } { K - 1 } \sum _ { q \in \cal S } { \bf z } _ { q , i } .\tag{12}
$$

The corresponding LOO residual is

$$
\mathbf { e } _ { s , i } ^ { \mathrm { { L O O } } } = \mathbf { z } _ { s , i } - \mathbf { p } _ { i } ^ { ( - s ) } .\tag{13}
$$

Its mean over the $N _ { T }$ -stimulus population is

$$
\bar { \mathbf { e } } _ { s } ^ { \mathrm { L O O } } = \frac { 1 } { N _ { T } } \sum _ { i = 1 } ^ { N _ { T } } \mathbf { e } _ { s , i } ^ { \mathrm { L O O } } ,\tag{14}
$$

and the across-stimulus residual covariance trace is estimated by

$$
\hat { \tau } _ { s } ^ { \mathrm { L O O } } = \frac { 1 } { N _ { T } - 1 } \sum _ { i = 1 } ^ { N _ { T } } \left. \mathbf { e } _ { s , i } ^ { \mathrm { L O O } } - \bar { \mathbf { e } } _ { s } ^ { \mathrm { L O O } } \right. _ { 2 } ^ { 2 } .\tag{15}
$$

The LOO residual uses an anchor averaged over $K - 1$ subjects, whereas the deployed target residual uses an anchor averaged over all $\dot { K }$ source subjects. Under a working approximation of comparable subject-level residual covariance and negligible cross-subject covariance, subtracting an n-subject mean scales the covariance trace by approximately $1 + 1 / n$ . We therefore estimate the target-residual covariance trace as

$$
\widehat { \tau } _ { \mathrm { s r c } } = \frac { 1 + 1 / K } { 1 + 1 / ( K - 1 ) } \cdot \frac { 1 } { K } \sum _ { s = 1 } ^ { K } \widehat { \tau } _ { s } ^ { \mathrm { L O O } } .\tag{16}
$$

This variance-transfer approximation is used only for deployment-time shrinkage; the exact finite-population analysis below does not rely on it.

For a size-M calibration subset sampled without replacement from the $N _ { T }$ -stimulus population, the estimated sampling-error energy of the residual mean is

$$
\widehat { \nu } _ { M } = \widehat { \tau } _ { \mathrm { s r c } } \left( \frac { 1 } { M } - \frac { 1 } { N _ { T } } \right) .\tag{17}
$$

Under a signal-plus-sampling-error view, $\| \widehat { \mathbf { b } } _ { T } \| _ { 2 } ^ { 2 }$ contains shift energy plus finite-sample error energy in expectation. Subtracting the estimated error energy $\widehat { \nu } _ { M }$ and normalizing by the observed energy gives the plug-in reliability factor

$$
\beta = \left[ 1 - \frac { \widehat { \nu } _ { M } } { \Vert \widehat { \mathbf { b } } _ { T } \Vert _ { 2 } ^ { 2 } } \right] _ { 0 } ^ { 1 } ,\tag{18}
$$

where $[ x ] _ { 0 } ^ { 1 } \ = \ \operatorname* { m i n } \{ 1 , \operatorname* { m a x } \{ 0 , x \} \}$ . We set $\beta ~ = ~ 0$ when $\| \widehat { \mathbf { b } } _ { T } \| _ { 2 } ^ { 2 } = \widetilde { 0 }$ . Thus, $\beta$ approaches zero when the estimated shift is dominated by sampling uncertainty and approaches one when most of its observed energy is supported beyond that uncertainty.

SA-LBC therefore applies the shrunk shift to each subsequent target TEST observation q (Fig. 1(d)),

$$
\widetilde { \mathbf { z } } _ { T , q } = \mathbf { z } _ { T , q } - \beta \widehat { \mathbf { b } } _ { T } .\tag{19}
$$

Calibration is performed before $\ell _ { 2 }$ normalization; normalization is applied only for cosine-similarity retrieval.

## 3.6 Finite-Population Analysis of First-Order Estimation

We isolate the effect of stimulus anchoring from reliabilityaware shrinkage by analyzing the raw anchored and globalmean estimators. We condition on a fixed target subject and its full N<sub>T</sub>-stimulus calibration population; the only randomness is which M stimuli are sampled uniformly without replacement.

We define the full-population first-order shift as

$$
\mathbf { b } _ { T } ^ { \mathrm { f u l l } } = \frac { 1 } { N _ { T } } \sum _ { i = 1 } ^ { N _ { T } } \left( \mathbf { z } _ { T , i } - \mathbf { p } _ { i } \right) .\tag{20}
$$

This quantity serves only as the analysis reference and is not available to the deployed calibration procedure.

To separate stimulus composition from residual variation, define the centered anchor deviation and centered matched residual as

$$
\mathbf { a } _ { i } = \mathbf { p } _ { i } - \pmb { \mu } _ { P } , \qquad \delta _ { T , i } = ( \mathbf { z } _ { T , i } - \mathbf { p } _ { i } ) - \mathbf { b } _ { T } ^ { \mathrm { f u l l } } .\tag{21}
$$

By construction,

$$
\sum _ { i = 1 } ^ { N _ { T } } \mathbf { a } _ { i } = \mathbf { 0 } , \qquad \sum _ { i = 1 } ^ { N _ { T } } \delta _ { T , i } = \mathbf { 0 } .\tag{22}
$$

For a sampled calibration subset $\mathcal { C } _ { M }$ , let

$$
\bar { \mathbf { a } } _ { M } = \frac { 1 } { M } \sum _ { i \in \mathcal { C } _ { M } } \mathbf { a } _ { i } , \qquad \bar { \mathbf { \delta } } _ { M } = \frac { 1 } { M } \sum _ { i \in \mathcal { C } _ { M } } \delta _ { T , i } .\tag{23}
$$

The estimator errors then decompose exactly as

$$
\widehat { \mathbf { b } } _ { T } ^ { \mathrm { S A } } - \mathbf { b } _ { T } ^ { \mathrm { f u l l } } = \bar { \pmb { \delta } } _ { M } ,
$$

$$
\widehat { \mathbf { b } } _ { T } ^ { \mathrm { G M } } - \mathbf { b } _ { T } ^ { \mathrm { f u l l } } = \bar { \pmb { \delta } } _ { M } + \bar { \mathbf { a } } _ { M } .\tag{24}
$$

(25)

Hence,

$$
\widehat { \mathbf { b } } _ { T } ^ { \mathrm { G M } } - \widehat { \mathbf { b } } _ { T } ^ { \mathrm { S A } } = \bar { \mathbf { a } } _ { M } .\tag{26}
$$

Under uniform sampling, the composition term $\bar { \mathbf { a } } _ { M } = \bar { \mathbf { p } } _ { M } -$ $\pmb { \mu } _ { P }$ has zero expectation. The unshrunk anchored estimator therefore has the classical control-variate form with a unit coefficient [41]. This preserves unbiasedness but does not guarantee lower MSE: the removed composition term $\hat { \mathbf { a } } _ { M }$ can covary with the remaining residual error $\bar { \pmb { \delta } } _ { M }$

Their finite-population second-order structure is characterized by

$$
\Sigma _ { \delta } = \frac { 1 } { N _ { T } - 1 } \sum _ { i = 1 } ^ { N _ { T } } \delta _ { T , i } \delta _ { T , i } ^ { \top } ,\tag{27}
$$

$$
\Sigma _ { P } = \frac { 1 } { N _ { T } - 1 } \sum _ { i = 1 } ^ { N _ { T } } \mathbf { a } _ { i } \mathbf { a } _ { i } ^ { \top } ,\tag{28}
$$

$$
\boldsymbol { \Sigma } _ { \delta P } = \frac { 1 } { N _ { T } - 1 } \sum _ { i = 1 } ^ { N _ { T } } \delta _ { T , i } \mathbf { a } _ { i } ^ { \top } .\tag{29}
$$

Here, $\Sigma _ { \delta }$ measures residual variation, $\Sigma _ { P }$ measures stimulus-anchor variation, and $\Sigma _ { \delta P }$ captures their coupling.

Theorem 1 (Exact finite-population MSE). ${ \cal I } f { \cal C } _ { M }$ is sampled uniformly without replacement, both estimators are unbiased for $\mathbf { b } _ { T } ^ { \mathrm { f u l l } }$

$$
\mathbb { E } \left[ \widehat { \mathbf { b } } _ { T } ^ { \mathrm { S A } } \right] = \mathbb { E } \left[ \widehat { \mathbf { b } } _ { T } ^ { \mathrm { G M } } \right] = \mathbf { b } _ { T } ^ { \mathrm { f u l l } } .\tag{30}
$$

The anchored estimator has mean-squared error

$$
\begin{array} { r } { \mathbb { E } \left[ \left\| \widehat { \mathbf { b } } _ { T } ^ { \mathrm { S A } } - \mathbf { b } _ { T } ^ { \mathrm { f u l l } } \right\| _ { 2 } ^ { 2 } \right] = \kappa _ { M } \operatorname { t r } \left( \Sigma _ { \delta } \right) , } \end{array}\tag{31}
$$

where

$$
\kappa _ { M } = \frac { 1 } { M } - \frac { 1 } { N _ { T } } .\tag{32}
$$

The corresponding global-mean MSE is

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left\| \widehat { \mathbf { b } } _ { T } ^ { \mathrm { G M } } - { \mathbf { b } } _ { T } ^ { \mathrm { f u l l } } \right\| _ { 2 } ^ { 2 } \right] } \\ & { \quad = \kappa _ { M } \left[ \mathrm { t r } \left( \Sigma _ { \delta } \right) + \mathrm { t r } \left( \Sigma _ { P } \right) + 2 \mathrm { t r } \left( \Sigma _ { \delta P } \right) \right] . } \end{array}\tag{33}
$$

The proof uses the finite-population sample-mean covariance identity [42] and is given in Supplementary Section S2.

Subtracting the two MSEs gives

$$
\begin{array} { r l r } & { } & { \Delta _ { \mathrm { M S E } } = \mathrm { M S E } \left( \widehat { \bf b } _ { T } ^ { \mathrm { G M } } \right) - \mathrm { M S E } \left( \widehat { \bf b } _ { T } ^ { \mathrm { S A } } \right) } \\ & { } & { = \kappa _ { M } \left[ \mathrm { t r } \left( { \pmb { \Sigma } } _ { P } \right) + 2 \mathrm { t r } \left( { \pmb { \Sigma } } _ { \delta P } \right) \right] . } \end{array}\tag{34}
$$

Corollary 1 (Condition for lower anchored-estimator MSE). For $M < N _ { T }$

$$
\begin{array} { r l } & { \mathrm { M S E } \Big ( \widehat { \mathbf { b } } _ { T } ^ { \mathrm { S A } } \Big ) < \mathrm { M S E } \Big ( \widehat { \mathbf { b } } _ { T } ^ { \mathrm { G M } } \Big ) } \\ & { \Longleftrightarrow \mathrm { t r } ( { \pmb { \Sigma } } _ { \delta P } ) > - \frac { 1 } { 2 } \mathrm { t r } ( { \pmb { \Sigma } } _ { P } ) . } \end{array}\tag{35}
$$

Stimulus anchoring therefore has lower MSE only when residual–anchor coupling is not sufficiently negative to offset the anchor-variation term. In the special case $\begin{array} { r } { \pmb { \Sigma } _ { \delta P } = \mathbf { 0 } , } \end{array}$ anchoring has lower MSE whenever $\mathrm { t r } ( \Sigma _ { P } ) > 0$ . No residual–anchor orthogonality assumption is required by the general result.

The estimator difference in Eq. (26) also gives

$$
\begin{array} { r } { \mathbb { E } \left[ \left\| \widehat { \mathbf { b } } _ { T } ^ { \mathrm { G M } } - \widehat { \mathbf { b } } _ { T } ^ { \mathrm { S A } } \right\| _ { 2 } ^ { 2 } \right] = \kappa _ { M } \operatorname { t r } \left( \Sigma _ { P } \right) . } \end{array}\tag{36}
$$

This stimulus-composition energy decreases with calibration budget and vanishes at $M = N _ { T }$ . When the condition in Corollary 1 holds, the positive MSE gap is scaled by the same factor $\kappa _ { M }$ and therefore shrinks as M approaches $N _ { T } .$

The cross-covariance term $\Sigma _ { \delta P }$ also allows residual variation to remain coupled to stimulus geometry, so the theorem does not require purely additive cross-subject mismatch. For illustration, consider

$$
\begin{array} { c } { { \displaystyle { \bf z } _ { T , i } = { \pmb \mu } _ { P } + { \bf b } _ { T } ^ { \mathrm { f u l l } } + { \bf Q } _ { T } { \bf a } _ { i } + \epsilon _ { i } , } } \\ { { \displaystyle { \bf Q } _ { T } ^ { \top } { \bf Q } _ { T } = { \bf I } , \qquad \sum _ { i = 1 } ^ { N _ { T } } \epsilon _ { i } = { \bf 0 } . } } \end{array}\tag{37}
$$

Since ${ \bf p } _ { i } = \pmb { \mu } _ { P } + { \bf a } _ { i }$ , the matched residual becomes

$$
{ \bf z } _ { T , i } - { \bf p } _ { i } = { \bf b } _ { T } ^ { \mathrm { f u l l } } + ( { \bf Q } _ { T } - { \bf I } ) { \bf a } _ { i } + \epsilon _ { i } .\tag{38}
$$

Thus, reorientation can induce residual–anchor coupling through $( { \bf Q } _ { T } - { \bf I } ) { \bf a } _ { i }$ . This model is illustrative only and is not assumed by the theorem. The finite-population analysis characterizes estimation within the calibration pool; transfer to concept-disjoint TEST stimuli is assessed empirically.

## 3.7 Calibration Budget and Orientation Estimation

The finite-population analysis allows higher-order mismatch, but does not determine how much orientation information can be estimated from the same few-shot calibration set. Let $\mathcal { C } _ { M } = \{ i _ { 1 } , \dots , i _ { M } \}$ . To isolate centered geometry, we form the matched calibration matrices

$$
\mathbf { X } _ { M } = \left[ \begin{array} { c c c } { ( \mathbf { z } _ { T , i _ { 1 } } - \bar { \mathbf { z } } _ { T , M } ) ^ { \top } } \\ { \vdots } \\ { ( \mathbf { z } _ { T , i _ { M } } - \bar { \mathbf { z } } _ { T , M } ) ^ { \top } } \end{array} \right] , \qquad \mathbf { Y } _ { M } = \left[ \begin{array} { c c c } { ( \mathbf { p } _ { i _ { 1 } } - \bar { \mathbf { p } } _ { M } ) ^ { \top } } \\ { \vdots } \\ { ( \mathbf { p } _ { i _ { M } } - \bar { \mathbf { p } } _ { M } ) ^ { \top } } \end{array} \right] .
$$

Because the centered rows sum to zero,

(39)

$$
\operatorname { r a n k } ( \mathbf { X } _ { M } ) , \operatorname { r a n k } ( \mathbf { Y } _ { M } ) \leq M - 1 ,\tag{40}
$$

and therefore

$$
\mathrm { r a n k } \left( { \bf X } _ { M } ^ { \top } { \bf Y } _ { M } \right) \leq M - 1 .\tag{41}
$$

The cross-covariance therefore contains at most $M \ : - \ : 1$ nonzero singular modes. The union of the centered target and anchor spans can have dimension up to $2 ( M - 1 ) ,$ ; the rank bound does not restrict an orthogonal correction to only $M - 1$ ambient directions. In our main setting, $d = 7 6 8$ and $M \in \{ 2 , 4 , 8 , 1 6 \}$ , giving rank bounds of 1, 3, 7, and 15, respectively. This rank bound neither rules out reorientation nor implies that translation must perform better; it only limits the orientation structure directly supported by the observed pairs.

To evaluate orientation under the same information bud-${ \mathrm { g e t } } ,$ we estimate an identity-regularized orthogonal map

$$
\begin{array} { r } { \widehat { \bf R } = \arg \operatorname* { m i n } _ { { \bf R } ^ { \top } { \bf R } = { \bf I } } \Big \{ \| { \bf X } _ { M } { \bf R } - { \bf Y } _ { M } \| _ { F } ^ { 2 } } \\ { + \lambda _ { R } \| { \bf R } - { \bf I } \| _ { F } ^ { 2 } \Big \} . } \end{array}\tag{42}
$$

where

$$
\lambda _ { R } = \rho \left\| \mathbf { X } _ { M } ^ { \top } \mathbf { Y } _ { M } \right\| _ { 2 } .\tag{43}
$$

Here, $\rho ~ \geq ~ 0$ controls identity regularization and $\| \cdot \| _ { 2 }$ denotes the spectral norm. By the orthogonal Procrustes solution [43], if

$$
\mathbf { X } _ { M } ^ { \top } \mathbf { Y } _ { M } + \lambda _ { R } \mathbf { I } = \mathbf { U } \boldsymbol { \Sigma } \mathbf { V } ^ { \top } ,\tag{44}
$$

then

$$
\begin{array} { r } { \widehat { \bf R } = { \bf U } { \bf V } ^ { \top } . } \end{array}\tag{45}
$$

Because calibration observations are stacked as rows, the corresponding transform on column-vector embeddings is $\widehat { \mathbf { R } } ^ { \top }$ . The orientation-only correction therefore preserves the target calibration centroid,

$$
\widetilde { \mathbf { z } } _ { T , q } ^ { R } = \bar { \mathbf { z } } _ { T , M } + \widehat { \mathbf { R } } ^ { \top } \left( \mathbf { z } _ { T , q } - \bar { \mathbf { z } } _ { T , M } \right) .\tag{46}
$$

The combined correction shifts the centroid by SA-LBC while applying the same orientation map to centered structure,

$$
\begin{array} { r } { \widetilde { \mathbf { z } } _ { T , q } ^ { T + R } = \bar { \mathbf { z } } _ { T , M } - \beta \widehat { \mathbf { b } } _ { T } \quad \quad \quad \quad \quad \quad \quad \quad } \\ { + \widehat { \mathbf { R } } ^ { \top } \left( \mathbf { z } _ { T , q } - \bar { \mathbf { z } } _ { T , M } \right) . } \end{array}\tag{47}
$$

These controls use the same frozen representation, matched observations, and calibration budget, holding target information access fixed while varying the geometric form of the correction.

## 4 EXPERIMENTS

## 4.1 Experimental Setup

Data and information access: THINGS-EEG2 [44] is the primary benchmark, with 10 subjects evaluated by leave-one-subject-out (LOSO) training on the other nine. The 200-image TEST gallery is concept-disjoint from TRAIN. EEG is whitened within each participant and session using that participant’s retained TRAIN epochs, not TEST covariance; the encoder receives 17 channels and 250 samples. TRAIN and TEST observations average four and 80 repetitions per stimulus, respectively. Thus, M counts stimuluslevel calibration averages after participant-specific preprocessing, not total target recordings or single-trial inputs. For each of 200 random permutations of target TRAIN, prefixes of lengths $M \in \{ 2 , \overline { { 4 } } , 8 , 1 6 \}$ define nested calibration sets shared across methods and seeds. Stimulus identities index source neural anchors; calibration uses no target image pixels, image embeddings, semantic labels, TEST statistics, retrieval feedback, or representation updates. Alljoined-1.6M [45] provides a 20-subject EEG cohort; THINGS-MEG [46] provides four subjects and two predefined variants for descriptive boundary analysis. Preprocessing, validrepeat averaging, and dataset-specific training protocols are detailed in Supplementary Section S1; external results are in Section S6.

Frozen representations: The primary diagnostic configuration uses our attention-based EEG encoder (denoted Attention) with a frozen CLIP ViT-B/32 backbone [47]. Source-trained projections map both modalities into a 768- dimensional retrieval space; visual prompts are disabled. Source-support, shift-fidelity, and directional diagnostics use this configuration. Source-balanced multi-positive contrastive training uses source-TRAIN concept-disjoint checkpoint selection only. We compare Attention, EEGProject, and TSConv with CLIP ViT-B/32; Attention with CLIP RN101 and DINOv2 ViT-B/14 [48]; and TSConv with DINOv2. These are six configurations, counting Attention–CLIP ViT-B/32 once, not a complete Cartesian product. Each configuration independently fits its EEG encoder and projections with three seeds, while keeping its pretrained visual backbone frozen; fixing an architecture does not fix its learned weights. All six use the same source-only training/selection protocol and paired draws without architecture-specific tuning. Architecture and optimizer details are in Supplementary Tables S1 and S11.

Controls: We compare no adaptation, raw globalmean matching, adaptive global mean, bank-shuffled anchoring, raw anchoring, and SA-LBC. Adaptive global mean shares SA-LBC’s source-derived noise term but computes its own coefficient:

$$
\begin{array} { r } { \beta _ { \mathrm { G M } } = \left[ 1 - \frac { \widehat \nu _ { M } } { \| \widehat { \mathbf { b } } _ { T } ^ { \mathrm { G M } } \| _ { 2 } ^ { 2 } } \right] _ { 0 } ^ { 1 } , } \\ { \widetilde { \mathbf { z } } _ { T , q } ^ { \mathrm { G M } } = \mathbf { z } _ { T , q } - \beta _ { \mathrm { G M } } \widehat { \mathbf { b } } _ { T } ^ { \mathrm { G M } } . } \end{array}\tag{48}
$$

The coefficient is zero for a zero-norm estimate. This comparison changes reference construction and its induced coefficient, not the reference alone at fixed strength. The bank-shuffled control instead fixes the matched-condition coefficient and replaces anchors through four derangements of the full source bank per draw. It generally changes the selected anchor set, testing stimulus-composition matching rather than pairing order within a fixed set. Raw anchoring fixes $\beta \ = \ \bar { 1 }$ . The orientation controls in Section 3.7 use the same matched data, with primary $\rho \mathrm { ~  ~ { ~ = ~ } ~ } 0 . 1$ and a fixed nine-value sensitivity grid; no TEST-based deployment selection is performed. They isolate regularized orthogonal calibration, not the complete SCORE method [7]. Normmatched random and sign-flipped directions are additional controls for Attention–CLIP ${ \hat { \mathrm { V i T } } } { \cdot } \mathrm { B } / 3 2 ;$ full definitions are in Supplementary Section S7.7.

Statistics: MRR is the primary endpoint; Top-1 and Top-5 are secondary metrics. Draws and training seeds are averaged within each held-out subject before reporting subject-level means, SDs, and inference. THINGS-EEG2 contrasts use exact two-sided paired sign-flip tests and $2 0 { , } 0 0 0$ subject-bootstrap resamples [49] for $9 5 \%$ intervals. Holm adjustment [50] covers four budgets separately within each contrast and configuration, not all configurations or contrasts jointly. The $4 \times 9$ regularization grids and jointconfiguration contrasts are descriptive. Alljoined uses 20 subjects, 50,000 bootstrap resamples, exact sign flips, and the explicitly supplemental within-contrast Holm-4 analysis; MEG remains descriptive over four subjects. Full statistical families and convergence details are in Supplementary Table S3 and Fig. S1.

## 4.2 Calibration Utility and Source-Reference Matching

Table 1 reports all budgets and controls for Attention–CLIP ViT-B/32. SA-LBC raises MRR from 0.1867 to 0.2013 at $M = 2$ and 0.2336 at M = 16. At M = 16, Top-1 increases from 8.63% to 12.10% and Top-5 from 26.20% to 33.11%. MRR and Top-5 improve in all 10 subjects at every budget; Top-1 improves in $\mathsf { 9 } / 1 0$ subjects at $M = 2 , 4$ and 10/10 at $\bar { \mathbf { \xi } } _ { M } = \mathbf { \bar { \xi } } 8 , 1 6$ . Each SA-LBC–baseline MRR contrast has $p _ { \mathrm { H o l m } } = 0 . 0 0 7 8$ . These are configuration-specific gains, not representation-invariant benefits.

Against adaptive global mean, the paired MRR gains are 0.00151, 0.00133, 0.00204, and 0.00183 at $M = 2 , 4 , 8 , 1 6 ,$ respectively, with $p _ { \mathrm { H o l m } } ~ = ~ 0 . 0 1 1 7 , 0 . 0 2 7 3 , 0 . 0 0 7 8 , 0 . 0 0 7 8 .$ The increment is small and not monotonically decreasing with budget. Matched references exceed bank-shuffled references by $0 . 0 0 6 0 { \scriptstyle - 0 . 0 0 7 3 } \mathrm { ~ M R R } ,$ , with positive differences in all subjects and $p _ { \mathrm { H o l m } } ~ = ~ 0 . 0 0 7 8$ at each budget (Fig. 2). Omitting matching and supplying a mismatched reference are different controls; neither replaces the no-adaptation comparison. The finite-population decrease in composition energy does not impose a corresponding MRR trend after estimator-specific shrinkage and cosine normalization.

Increasing source support K improves group-average calibration gains under the same rule, but both the noise term and coefficient are recomputed as K changes. This is not an isolated test of anchor averaging at fixed strength (Supplementary Table S6 and Section S7.3).

## 4.3 Verification of First-Order Estimation

For each configuration, subject, and seed, analytic MSE uses the full target-TRAIN finite population, and empirical MSE uses 200 calibration draws of the unshrunk estimators. The full-TRAIN residual mean serves only as the retrospective analysis reference and is not used to estimate or select the deployed correction. Across 120 seed–subject–budget conditions per configuration, mean relative discrepancies are 0.90–1.61% for anchoring and 1.05–1.58% for global-mean estimation; ordering agreement is 90.0–100.0% (Supplementary Table S33). This is agreement between analytic and empirical error rankings, not the fraction in which anchoring wins. The audit verifies subset-sampling calculations, not a retrieval ordering after shrinkage, normalization, and transfer to concept-disjoint TEST stimuli.

TABLE 1  
Attention–CLIP ViT-B/32 retrieval after participant-specific preprocessing. MRR is the primary endpoint; values are $\mathtt { m e a n } \pm \mathtt { S D }$ over 10 subjects after averaging 200 paired draws and three seeds within subject. Both adaptive estimators share the noise term but compute their own coefficient; raw anchoring uses $\beta = 1$ . Bold marks the highest rounded mean within each calibration budget. Complete Top-1 and Top-5 results are in Supplementary Table S12.
<table><tr><td>Method</td><td> $M = 2$ </td><td> $M = 4$ </td><td> $M = 8$ </td><td> $M = 1 6$ </td></tr><tr><td>No adaptation</td><td> $0 . 1 8 6 7 \pm 0 . 0 4 9 5$ </td><td> $0 . 1 8 6 7 \pm 0 . 0 4 9 5$ </td><td> $0 . 1 8 6 7 \pm 0 . 0 4 9 5$ </td><td> $0 . 1 8 6 7 \pm 0 . 0 4 9 5$ </td></tr><tr><td>Global mean</td><td> $0 . 1 6 7 3 \pm 0 . 0 2 6 5$ </td><td> $0 . 1 9 9 4 \pm 0 . 0 3 3 8$ </td><td> $0 . 2 2 1 3 \pm 0 . 0 4 1 2$ </td><td> $0 . 2 3 4 7 \pm 0 . 0 4 5 9$ </td></tr><tr><td>Adaptive global mean</td><td> $0 . 1 9 9 8 \pm 0 . 0 4 5 9$ </td><td> $0 . 2 1 0 7 \pm 0 . 0 4 6 0$ </td><td> $0 . 2 2 1 9 \pm 0 . 0 4 7 5$ </td><td> $0 . 2 3 1 8 \pm 0 . 0 4 8 9$ </td></tr><tr><td>Shuffled anchor</td><td> $0 . 1 9 5 3 \pm 0 . 0 4 5 6$ </td><td> $0 . 2 0 5 6 \pm 0 . 0 4 5 1$ </td><td> $0 . 2 1 6 6 \pm 0 . 0 4 5 6$ </td><td> $0 . 2 2 7 5 \pm 0 . 0 4 7 0$ </td></tr><tr><td>Raw anchored  $( \beta = 1 )$ </td><td> $0 . 1 8 2 2 \pm 0 . 0 3 4 1$ </td><td> $0 . 2 0 9 0 \pm 0 . 0 4 0 1$ </td><td> $\mathbf { 0 . 2 2 7 1 \pm 0 . 0 4 5 0 }$ </td><td> $\mathbf { 0 . 2 3 8 1 \pm 0 . 0 4 8 1 }$ </td></tr><tr><td>SA-LBC</td><td> $\mathbf { 0 . 2 0 1 3 \pm 0 . 0 4 6 4 }$ </td><td> $\mathbf { 0 . 2 1 2 0 \pm 0 . 0 4 7 6 }$ </td><td> $0 . 2 2 3 9 \pm 0 . 0 4 9 5$ </td><td> $0 . 2 3 3 6 \pm 0 . 0 5 0 4$ </td></tr></table>

![](images/30f9c090c01787c891a4d95278b1e673116cb1f32806452ac3ee5b32731209c3.jpg)  
(a)

![](images/725c697e23f6d99939128724e30cd87736edcf3f04392eb7d025614497f8e923.jpg)  
(b)

![](images/87d81d49af2c4dc970a81d45c4bd7a3bc141b674df322a696540d74fa81b00b6.jpg)  
(c)  
Fig. 2. Paired MRR effects for Attention–CLIP ViT-B/32: SA-LBC minus (a) no adaptation, (b) adaptive global mean, and (c) bank-shuffled anchoring. Error bars are 95% subject-bootstrap intervals after within-subject draw/seed averaging. The controls test net utility, the adaptivereference increment, and reference-composition matching, respectively; their effects are not interchangeable.

## 4.4 Geometry and Correction Strength

We compare SA-LBC translation (T), centroid-preserving regularized orientation (R), and their combination $( T + R )$ using the same frozen representations and matched calibration observations. The centered cross-covariance respects rank bounds of 1, 3, 7, and 15 for M = 2, 4, 8, 16; unionspan dimensions do not exceed 2, 6, 14, and 30 in the 768-dimensional space. These constraints do not determine retrieval ordering. At $\rho \ : = \ : 0 . 1 , T > R$ and $T + R \textless T$ across all six configurations and four budgets, with the same direction in all subjects and $p _ { \mathrm { H o l m } } ~ = ~ 0 . 0 0 7 8$ within each contrast/configuration. $\mathrm { A t } \rho = 1$ , R instead exceeds T at every budget for both TSConv configurations in groupaverage MRR. At the same $\rho = 1$ and $M = 1 6 , { \bar { T } } + { \bar { R } }$ exceeds T in all six configurations by $0 . 0 0 3 7 7 { - } 0 . 0 0 7 9 0$ (Table 2). These grid results are descriptive, not TEST-selected deployment settings; the primary disadvantage does not make orientation intrinsically harmful.

Even within translation, correction strength remains a separate question. Shrinkage improves on raw anchoring in all six configurations at $M \equiv \mathrm { 2 } ~ ( p _ { \mathrm { H o l m } } < 0 . 0 5 $ within each configuration). For Attention–CLIP ViT-B/32, MRR is 0.2013 versus 0.1822 $( \Delta = + 0 . 0 1 9 1 , p _ { \mathrm { H o l m } } = 0 . 0 4 6 9 ) ;$ raw anchoring falls below no adaptation (0.1867). At $M = 1 6 ,$ raw anchoring instead reaches 0.2381 versus 0.2336 $( \Delta \ : =$ $- 0 . 0 0 4 5 , p _ { \mathrm { H o l m } } = 0 . 0 3 9 1 )$ . Both TSConv configurations still benefit from shrinkage at $M = 1 6$ , by 0.00567 with CLIP and 0.00376 with DINOv2 (both $p _ { \mathrm { H o l m } } = 0 . 0 0 7 8 )$ . Thus, sample count alone does not determine a common crossover, and improving on raw correction does not guarantee improving on no adaptation.

Diagnostics for Attention–CLIP ViT-B/32 show increasing fidelity to the within-model full-TRAIN reference as M grows. Norm-matched random directions remain near no adaptation, whereas sign reversal degrades retrieval; $\mathsf { S A - L B C }$ exceeds both controls in all subjects and budgets $( p _ { \mathrm { H o l m } } ~ = ~ 0 . 0 0 7 8$ within each family). Full magnitudes, fidelity, and direction results are retained in Supplementary Section S7.4 and Tables S8–S10.

## 4.5 Representation and External Boundaries

Table 3 compares each configuration with its own samecache baseline. With CLIP ViT-B/32, EEGProject achieves higher group-average MRR than Attention before calibration (0.21393 versus 0.18675) and after SA-LBC at $M = 1 6$ (0.25908 versus 0.23364). Its larger final MRR does not reflect a larger paired calibration gain (+0.04515 versus +0.04690).

TABLE 2  
Regularization-dependent geometric ordering at $M = 1 6$ . Entries are group-average MRR from the same cache within each configuration. Comparisons of correction types are made within each row. $\rho = 0 . 1$ is the primary setting; $\rho = 1$ is a fixed member of the sensitivity grid, not a TEST-selected deployment parameter. The complete grid is reported in the Supplementary Material.
<table><tr><td>EEG encoder</td><td>Visual backbone</td><td>T</td><td> $R _ { 0 . 1 }$ </td><td> $( T + R ) _ { 0 . 1 }$ </td><td> $R _ { 1 }$ </td><td> $( T + R ) _ { 1 }$ </td></tr><tr><td>Attention</td><td>CLIP ViT-B/32</td><td>0.23364</td><td>0.16749</td><td>0.20697</td><td>0.19326</td><td>0.24114</td></tr><tr><td>EEGProject</td><td>CLIP ViT-B/32</td><td>0.25908</td><td>0.17956</td><td>0.21624</td><td>0.22060</td><td>0.26695</td></tr><tr><td>TSConv</td><td>CLIP ViT-B/32</td><td>0.21047</td><td>0.18012</td><td>0.18171</td><td>0.21503</td><td>0.21582</td></tr><tr><td>Attention</td><td>CLIP RN101</td><td>0.21311</td><td>0.15846</td><td>0.19034</td><td>0.18487</td><td>0.22101</td></tr><tr><td>Attention</td><td>DINOv2 ViT-B/14</td><td>0.18318</td><td>0.13609</td><td>0.16186</td><td>0.15945</td><td>0.18855</td></tr><tr><td>TSConv</td><td>DINOv2 ViT-B/14</td><td>0.18264</td><td>0.15619</td><td>0.15840</td><td>0.18479</td><td>0.18641</td></tr></table>

SA-LBC improves Attention across all three evaluated visual backbones and EEGProject with CLIP across all budgets. TSConv–CLIP instead has a negative M = 2 effect in all subjects (∆MRR = −0.00330, p<sub>Holm</sub> = 0.0078), and only +0.00100 at $M = 1 6 ~ ( p _ { \mathrm { H o l m } } = 0 . 6 1 7 2 )$ . TSConv–DINOv2 effects are -0.00236 and +0.00143 at those budgets, neither significant after correction $( p _ { \mathrm { H o l m } } ~ = ~ 0 . 0 5 4 7 , 0 . 2 7 5 4 )$ . The configurations are separately source-trained; these comparisons do not isolate architecture causality or establish an encoder–backbone interaction. Matched references still exceed shuffled references in every configuration and budget, whereas their advantage over adaptive global mean is not significant at any budget for TSConv–DINOv2. Reference matching is therefore not a guarantee of useful calibration.

External results use three seeds and 200 paired draws (Supplementary Section S6 and Fig. S6). Alljoined MRR changes are -0.00048, +0.00044, +0.00120, and +0.00228 as M increases. At M = 16, 16/20 subjects improve; the unadjusted 95% interval is [0.00047,0.00418] and exact $p =$ 0.0281, but supplemental Holm-4 gives $p = 0 . 1 1 2 4$ . No SA-LBC–baseline MRR contrast passes that correction. Matched references do exceed shuffled references by 0.00025–0.00032 under their own Holm-4 correction, which does not establish a net calibration benefit. MEG gains at M = 16 are +0.00265 and +0.01009 for the baseline and full variants. With four subjects and multiple components changed together, this is descriptive evidence, not a causal or modalitygeneral result.

Finally, the dedicated SATTC comparison uses the same preprocessed inputs but different latent-calibration information: 200 unlabeled TEST queries versus SA-LBC’s matched pre-test observations. It is a regime comparison, not a matched-information algorithmic ranking (Supplementary Section S7.6 and Table S35).

## 5 DISCUSSION

Our results separate the information needed to estimate a cross-subject correction from the conditions under which that correction improves retrieval. Stimulus matching controls the source reference, shrinkage controls the applied strength, and geometric regularization controls flexibility. These choices remain constrained by the calibration budget, but their functional value also depends on the frozen representation. The resulting view is not that one correction is universally preferable, but that calibration must be evaluated under its information and representation conditions.

Stimulus anchoring has a specific estimation role: it replaces a fixed population centroid with the source reference for the sampled target stimuli. This removes the explicit composition term in the global-mean estimator, but does not guarantee lower MSE because residual–anchor crosscovariance can offset the removed variation. The finitepopulation analysis states that boundary exactly, and the sampling audits verify its numerical predictions across the six configurations. Retrieval supplies a separate test: matched anchors consistently outperform shuffled anchors, whereas their advantage over adaptive global mean is smaller and not significant in every configuration. The two controls cannot be conflated, because omitting correspondence differs from imposing an incorrect correspondence. Nor does decreasing composition energy imply a decreasing MRR gap after shrinkage and normalization. Relative to prior uses of shared stimuli for cross-subject alignment [11], [13], [14], [37], the contribution is therefore an explicit account of how reference construction changes few-shot estimation error, not the first use of correspondence itself.

Reference construction does not determine which geometric transformation the same observations can support. At the primary regularization setting, translation outperforms standalone orientation in all evaluated configurations and budgets. However, stronger identity regularization reverses this ordering for the TSConv configurations and allows orientation to complement translation. The low rank of the centered cross-covariance limits data-supported geometry, but is not a proof of translation’s superiority. These results qualify rather than contradict evidence that richer mappings can support cross-subject alignment [7], [15]. They show why geometric class and regularization must be considered together; the sensitivity grid does not provide a TEST-free rule for choosing either.

Even for a fixed transformation class, correction strength remains a separate estimation decision. At the smallest budget, shrinkage improves on the raw anchored correction in all six configurations. Its larger-budget behavior is not uniform: full-strength correction outperforms shrinkage for Attention–CLIP ViT-B/32 at M = 16, whereas both TSConv configurations still benefit from shrinkage. Sample count alone therefore does not determine when attenuation becomes unnecessary. These contrasts also distinguish limiting an unreliable correction from ensuring a net retrieval gain. The crossover in this configuration is consistent with improving shift fidelity, but does not distinguish uncertainty misestimation from a mismatch between squared-error estimation and ranking-based utility. The current coefficient remains a source-derived plug-in rule, not retrieval-optimal shrinkage.

TABLE 3  
Calibration utility across six source-trained neural–visual configurations. Baseline MRR and paired changes use the same re-encoded cache within each row; Attention–CLIP ViT-B/32 is counted once. ∆ is SA-LBC minus no adaptation. p<sub>H</sub> is Holm-adjusted across M ∈ {2, 4, 8, 16} separately within each configuration and contrast, not across the six rows. These tests do not compare encoders. The full four-budget results are provided in the Supplementary Material.
<table><tr><td></td><td>Visual backbone</td><td>Baseline MRR</td><td>∆MRR (M = 2)</td><td>pH</td><td>∆MRR  $( M = 1 6 )$ </td><td></td></tr><tr><td>EEG encoder</td><td></td><td></td><td>+0.01458</td><td>0.0078</td><td></td><td>pH</td></tr><tr><td>Attention</td><td>CLIP ViT-B/32</td><td>0.18675 0.21393</td><td>+0.01224</td><td>0.0234</td><td>+0.04690 +0.04515</td><td>0.0078 0.0078</td></tr><tr><td>EEGProject TSConv</td><td>CLIP ViT-B/32 CLIP ViT-B/32</td><td>0.20947</td><td>-0.00330</td><td>0.0078</td><td>+0.00100</td><td>0.6172</td></tr><tr><td>Attention</td><td>CLIP RN101</td><td>0.18022</td><td>+0.00944</td><td>0.0215</td><td>+0.03290</td><td>0.0078</td></tr><tr><td>Attention</td><td>DINOv2 ViT-B/14</td><td>0.15587</td><td>+0.00800</td><td>0.0078</td><td>+0.02730</td><td>0.0078</td></tr><tr><td>TSConv</td><td>DINOv2 ViT-B/14</td><td>0.18122</td><td>-0.00236</td><td>0.0547</td><td>+0.00143</td><td>0.2754</td></tr></table>

The six-configuration results make this distinction concrete. With CLIP ViT-B/32, EEGProject has higher groupaverage MRR than Attention before and after calibration, but smaller mean calibration gains at M = 2 and M = 16. SA-LBC benefits Attention with each evaluated visual backbone and EEGProject with CLIP, whereas TSConv yields small or negative calibration effects with both CLIP and DINOv2. Yet matched anchors remain preferable to shuffled anchors in those same TSConv configurations. Correct correspondence can therefore improve a calibration procedure without making that procedure better than no calibration. This is a configuration-dependent result, not evidence that neural architecture alone causes the difference: each configuration has separately learned source alignment, and visual changes also alter pretraining and preprocessing. The external results extend the boundary rather than remove it. Alljoined-1.6M shows smaller, heterogeneous gains, while the two THINGS-MEG variants provide only descriptive evidence from four subjects. Neither establishes universal cross-dataset or cross-modality benefit, and the mechanism linking representation to calibration utility remains unresolved.

These findings must also be interpreted within the information available to each method. Shared-space and functional-alignment approaches use shared responses or stimulus correspondence [11], [13], whereas SATTC and SCORE exploit unlabeled target-side structure at deployment [7], [36]. Conditional on the stated preprocessing, SA-LBC uses a frozen source-trained representation and a small set of matched observations to estimate the latent correction, without using future target TEST statistics or retrieval feedback. The dedicated SATTC comparison is therefore a comparison of information regimes, not a matched-information test of algorithmic superiority. Calibration methods should be distinguished by when target data become available, whether correspondence is supplied, and whether future TEST observations are jointly accessible. Under this distinction, SA-LBC is stimulus-matched pre-test calibration, not unrestricted label-free adaptation.

Several limits remain. Participant-specific TRAIN whitening uses target recordings beyond the latentcalibration subset. The present THINGS-EEG2 results therefore do not establish end-to-end M-observation adaptation. Testing that stronger claim would require budgeting all target-dependent preprocessing as well as calibration. Known stimulus identities are required to retrieve the source anchors; extending beyond that regime requires estimating correspondence without target TEST information. The geometric controls cover identity-regularized orthogonal maps, not arbitrary nonlinear transformations, and the six configurations do not constitute a full encoder– backbone factorial design. Neither the sensitivity grid nor the shrinkage comparisons establish a source-only selector of geometric class and correction strength. A stronger test would require such a selector, evaluated without target TEST feedback, as well as independent characterization of representation properties that predict calibration utility. Broader validity requires larger external cohorts and tests across sessions, devices, and modalities. Finally, the full-TRAIN residual mean is a retrospective finite-population reference, not ground-truth biological bias; fidelity to it does not establish recovery of a stable biological subject-specific state.

The practical implication is to distinguish estimating a correction, controlling its strength and flexibility, and establishing its utility relative to no calibration. The available target evidence constrains each decision, but the appropriate decision also depends on the representation being corrected. Cross-subject calibration should therefore ask not only what transformation can be expressed, but what correction the available evidence can support and whether applying it is useful.

## 6 CONCLUSION

We study stimulus-matched latent calibration for crosssubject neural–image retrieval conditional on fixed preprocessing, and introduce SA-LBC to estimate a first-order shift with source-derived shrinkage. The finite-population analysis gives exact errors for the unshrunk anchored and global-mean estimators and identifies when anchoring lowers MSE. Across six source-trained configurations, correct stimulus correspondence contributes consistently, but net retrieval gains, the benefit of shrinkage, and geometric ordering are not representation invariant. The primary translation advantage is conditional on regularization, and the TSConv results show that matching the source reference or attenuating the correction need not yield a net retrieval benefit over no calibration. These findings support evaluating reference construction, correction strength, and geometric flexibility against both the available target evidence and the frozen representation.

## 7 DATA AND CODE AVAILABILITY

The neural datasets used in this study are publicly available from their original repositories: THINGS-EEG2 [44], Alljoined-1.6M [45], and THINGS-MEG [46]. The source code, configuration files, and evaluation scripts required to reproduce the reported experiments are publicly available at: https://github.com/Xizhang-Yao/SA-LBC.

## 8 CONFLICTS OF INTEREST

The authors declare that they have no conflicts of interest.

## 9 ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China under Grant 62522608, the Guangdong Provincial Project (Grant No. 2024TQ08X330), the Medical-Engineering Interdisciplinary Research Foundation of Shenzhen University under Grant 2024YG008, Brain Science and Brain-like Intelligence Technology - National Science and Technology Major Project (2021ZD0200500), Shenzhen-Hong Kong Institute of Brain Science-Shenzhen Fundamental Research Institutions (2023SHIBS0003), the Shenzhen Science and Technology Program (No. JCYJ20241202124222027 and JCYJ20241202124209011), and the Key Research and Development Program of Hunan Province (2025QK3008).

## REFERENCES

[1] Y. Song, B. Liu, X. Li, N. Shi, Y. Wang, and X. Gao, “Decoding natural images from EEG for object recognition,” in Proc. 12th Int. Conf. Learn. Represent., 2024. [Online]. Available: https://openreview.net/forum?id=dhLIno8FmH

[2] C.-S. Chen and C.-S. Wei, “Mind’s Eye: Image recognition by EEG via multimodal similarity-keeping contrastive learning,” 2024, arXiv:2406.16910. [Online]. Available: https://arxiv.org/ abs/2406.16910

[3] J. Wang, X. Yao, H. Lin, Z. Li, Y. Liu, Q. Liu, G. Huang, L. Zhang, Z. Liang, and X. Wu, “NeuroCLIP: Brain-inspired prompt tuning for EEG-to-image multimodal contrastive learning,” 2025, arXiv:2511.09250v2; Revised September 2026. [Online]. Available: https://arxiv.org/abs/2511.09250v2

[4] L. Jiang, Q. She, J. Xu, H. Xu, D. Wu, and Z. Kuang, “Subject-aware multi-granularity alignment for zero-shot EEGto-image retrieval,” 2026, arXiv:2604.17782. [Online]. Available: https://arxiv.org/abs/2604.17782

[5] S. Jo, W. Jeong, D.-W. Heo, Y. Hwang, and H.-I. Suk, “HyFI: Hyperbolic feature interpolation for brain-vision alignment,” Proc. AAAI Conf. Artif. Intell., vol. 40, no. 7, pp. 5575–5583, 2026. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/ article/view/37476

[6] T. Li, Y. Yan, F. Dou, W. Song, and X. Zhang, “Cross-subject generalization for EEG decoding: A survey of deep learning methods,” Prog. Biomed. Eng., vol. 8, no. 2, 2026, art. no. 022013. [Online]. Available: https://doi.org/10.1088/2516-1091/ae65f0

[7] Z. Cui, S. Kan, S. Li, Z. Wang, and D. Wu, “SCORE: Subject coordinate recovery for label-free cross-subject EEGto-image retrieval,” 2026, arXiv:2608.19134. [Online]. Available: https://arxiv.org/abs/2608.19134

[8] W. Choi, J.-M. Kim, S. Kim, Y. Choi, and T.-E. Kam, “Subjectinvariant EEG embeddings via mixup and adversarial learning for semantic retrieval,” in Proc. 41st ACM/SIGAPP Symp. Appl. Comput. Association for Computing Machinery, 2026, pp. 1960–1961. [Online]. Available: https://doi.org/10.1145/3748522. 3779935

[9] S. Huang, H. Luo, H. Jing, Q. Zhang, L. Chang, Y. Feng, X. Lin, C. Qin, H. Chen, S. Jia, S. Sun, and Y. Wang, “NEED: Cross-subject and cross-task generalization for video and image reconstruction from EEG signals,” in Adv. Neural Inf. Process. Syst., vol. 38. Curran Associates, Inc., 2025, pp. 173 134–173 173. [Online]. Available: https://doi.org/10.52202/085713-5761

[10] J.-M. Kim, W. Choi, Y. Choi, S. Kim, and T.-E. Kam, “CroSS: Cross-subject visual brain decoding via soft alignment and subject-aware batch normalization,” in Proc. 41st ACM/SIGAPP Symp. Appl. Comput. Association for Computing Machinery, 2026, pp. 1917–1924. [Online]. Available: https://doi.org/10. 1145/3748522.3779936

[11] M. Ferrante, T. Boccato, F. Ozcelik, R. VanRullen, and N. Toschi, “Through their eyes: Multi-subject brain decoding with simple alignment techniques,” Imaging Neurosci., vol. 2, pp. 1–21, 2024. [Online]. Available: https://doi.org/10.1162/imag a 00170

[12] P. S. Scotti, M. Tripathy, C. Torrico, R. Kneeland, T. Chen, A. Narang, C. Santhirasegaran, J. Xu, T. Naselaris, K. A. Norman, and T. M. Abraham, “MindEye2: Shared-subject models enable fMRI-to-image with 1 hour of data,” in Proc. 41st Int. Conf. Mach. Learn., ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 2024, pp. 44 038–44 059. [Online]. Available: https://proceedings.mlr.press/v235/scotti24a.html

[13] J. V. Haxby, J. S. Guntupalli, A. C. Connolly, Y. O. Halchenko, B. R. Conroy, M. I. Gobbini, M. Hanke, and P. J. Ramadge, “A common, high-dimensional model of the representational space in human ventral temporal cortex,” Neuron, vol. 72, no. 2, pp. 404–416, 2011. [Online]. Available: https://doi.org/10.1016/j.neuron.2011.08.026

[14] J. Liu, J. Du, X. Chen, G. Li, J. Cai, S. Fong, and M. Xu, “MindAdapter: Few-shot parameter-efficient residual calibration of cross-subject brain-to-visual decoding models,” in Proc. 32nd ACM SIGKDD Conf. Knowl. Discov. Data Min. V.2. Association for Computing Machinery, 2026, pp. 11 456–11 467. [Online]. Available: https://dl.acm.org/doi/10.1145/3770855.3818949

[15] P. L. C. Rodrigues, C. Jutten, and M. Congedo, “Riemannian Procrustes analysis: Transfer learning for brain–computer interfaces,” IEEE Trans. Biomed. Eng., vol. 66, no. 8, pp. 2390–2401, 2019. [Online]. Available: https://doi.org/10.1109/TBME.2018. 2889705

[16] Y. Kamitani and F. Tong, “Decoding the visual and subjective contents of the human brain,” Nat. Neurosci., vol. 8, no. 5, pp. 679– 685, 2005. [Online]. Available: https://doi.org/10.1038/nn1444

[17] K. N. Kay, T. Naselaris, R. J. Prenger, and J. L. Gallant, “Identifying natural images from human brain activity,” Nature, vol. 452, no. 7185, pp. 352–355, 2008. [Online]. Available: https://doi.org/10.1038/nature06713

[18] T. Naselaris, R. J. Prenger, K. N. Kay, M. Oliver, and J. L. Gallant, “Bayesian reconstruction of natural images from human brain activity,” Neuron, vol. 63, no. 6, pp. 902–915, 2009. [Online]. Available: https://doi.org/10.1016/j.neuron.2009.09.006

[19] T. Horikawa and Y. Kamitani, “Generic decoding of seen and imagined objects using hierarchical visual features,” Nat. Commun., vol. 8, no. 1, 2017, art. no. 15037. [Online]. Available: https://doi.org/10.1038/ncomms15037

[20] S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, and M. Shah, “Decoding brain representations by multimodal learning of neural activity and visual features,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 11, pp. 3833–3849, 2021. [Online]. Available: https://doi.org/10.1109/TPAMI.2020.2995909

[21] D. Li, C. Wei, S. Li, J. Zou, and Q. Liu, “Visual decoding and reconstruction via EEG embeddings with guided diffusion,” in Adv. Neural Inf. Process. Syst., vol. 37. Curran Associates, Inc., 2024, pp. 102 822–102 864. [Online]. Available: https: //doi.org/10.52202/079017-3266

[22] Y. Li, Z. Kang, S. Gong, W. Dong, W. Zeng, H. Yan, W. T. Siok, and N. Wang, “Neural-MCRL: Neural multimodal contrastive representation learning for EEG-based visual decoding,” in Proc. IEEE Int. Conf. Multimedia Expo (ICME). IEEE, 2025, pp. 1–6. [Online]. Available: https://doi.org/10.1109/ICME59968. 2025.11210130

[23] W. Zhang, S. Wang, Y. Su, X. Li, C. Zhang, and S. Zhong, “NeuroBridge: Bio-inspired self-supervised EEGto-image decoding via cognitive priors and bidirectional semantic alignment,” Proc. AAAI Conf. Artif. Intell., vol. 40, no. 21, pp. 18 028–18 036, 2026. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/38863

[24] J. Zhou, C. Xu, W. Wang, E. Yang, and C. Deng, “EEGiT: Teaching vision transformers to understand the EEG signal,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2026, pp. 40 438–40 447.

[25] H. Xu and N. Lin, “Neurovista: A bidirectional masked cross-modal fusion network for robust EEG-to-image decoding,” Neural Netw., vol. 195, 2026, art. no. 108297. [Online]. Available: https://doi.org/10.1016/j.neunet.2025.108297

[26] J. Tang, S. Jiang, F. Su, and Z. Zhao, “Aligning what EEG can see: Structural representations for brain-vision matching,” 2026, arXiv:2603.07077. [Online]. Available: https://arxiv.org/ abs/2603.07077

[27] Y. Zhu, Z. Xiang, J. Wu, J. Guo, H. Zhang, C. Song, H. Fang, Y. Guo, and X. Liu, “Region-aware hierarchical sub-feature alignment for robust EEG-based visual decoding,” in Findings IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2026, pp. 6667–6676.

[28] D. Liu, W. Dai, J. Qian, H. Liu, H. Yi, and W. Kong, “Linguistic priors for visual decoupling: Towards symmetric vision-brain alignment,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2026, pp. 7869–7878.

[29] J. Wang and K. Chen, “CORTIVA: Candidate-score fusion of complementary visual teachers for EEG- and MEG-toimage retrieval,” 2026, arXiv:2608.01355. [Online]. Available: https://arxiv.org/abs/2608.01355

[30] Y. Lin, J.-H. Tsai, and C.-S. Wei, “SIMON: Saliency-aware integrative multi-view object-centric neural decoding,” 2026, arXiv:2605.00401. [Online]. Available: https://arxiv.org/abs/ 2605.00401

[31] H. He and D. Wu, “Transfer learning for brain–computer interfaces: A Euclidean space data alignment approach,” IEEE Trans. Biomed. Eng., vol. 67, no. 2, pp. 399–410, 2020. [Online]. Available: https://doi.org/10.1109/TBME.2019.2913914

[32] B. Sun, J. Feng, and K. Saenko, “Return of frustratingly easy domain adaptation,” in Proc. AAAI Conf. Artif. Intell., vol. 30, no. 1, 2016, pp. 2058–2065. [Online]. Available: https://doi.org/10.1609/aaai.v30i1.10306

[33] B. Lu, J. Chen, F. Wang, G. Wen, R. Fu, and C. Hua, “Causality-driven convolutional manifold attention network for electroencephalogram signal decoding,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 3, pp. 2253–2264, 2026. [Online]. Available: https://doi.org/10.1109/TPAMI.2025.3625631

[34] S. Zhu, Z. Ye, Q. Ai, and Y. Liu, “CrossPT-EEG: A benchmark for cross-participant and cross-time generalization of EEG-based visual decoding,” 2024, arXiv:2406.07151. [Online]. Available: https://arxiv.org/abs/2406.07151

[35] Y. Wang, T. Li, Y. Yan, W. Song, and X. Zhang, “What causes performance degradation in cross-subject EEG classification?” 2024, arXiv:2410.03057; Accepted to IEEE International Conference on Systems, Man, and Cybernetics (SMC) 2026. [Online]. Available: https://arxiv.org/abs/2410.03057

[36] Q. Huang and W. Zhu, “SATTC: Structure-aware label-free testtime calibration for cross-subject EEG-to-image retrieval,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2026, pp. 16 887–16 896.

[37] Y. Dai, Z. Yao, C. Song, Q. Zheng, W. Mai, K. Peng, S. Lu, W. Ouyang, J. Yang, and J. Wu, “MindAligner: Explicit brain functional alignment for cross-subject visual decoding from limited fMRI data,” in Proc. 42nd Int. Conf. Mach. Learn., ser. Proceedings of Machine Learning Research, vol. 267. PMLR, 2025, pp. 12 214–12 228. [Online]. Available: https://proceedings.mlr.press/v267/dai25m.html

[38] S. Jiang, Z. Meng, H. Li, D. Liu, F. Su, and Z. Zhao, “MindShot: A few-shot brain decoding framework via transferring cross-subject prior and distilling frequency domain knowledge,” Knowl.-Based Syst., vol. 332, 2026, art. no. 114867. [Online]. Available: https://doi.org/10.1016/j.knosys.2025.114867

[39] R. Kneeland, W. Jiang, U. B. Nunes, P. S. Scotti, A. Delorme, and J. Xu, “ENIGMA: EEG-to-image in 15 minutes using less than 1% of the parameters,” 2026, arXiv:2602.10361. [Online]. Available: https://arxiv.org/abs/2602.10361

[40] W. James and C. Stein, “Estimation with quadratic loss,” in Proc. 4th Berkeley Symp. Math. Statist. Probab., J. Neyman, Ed., vol. 1. Berkeley, CA, USA: University of California Press, 1961, pp. 361– 379.

[41] P. W. Glynn and R. Szechtman, “Some new perspectives on the method of control variates,” in Monte Carlo and Quasi-Monte Carlo

Methods 2000. Berlin, Heidelberg: Springer, 2002, pp. 27–49. [Online]. Available: https://doi.org/10.1007/978-3-642-56046-0 3

[42] W. G. Cochran, Sampling Techniques, 3rd ed. New York, NY, USA: John Wiley & Sons, 1977.

[43] P. H. Schonemann, “A generalized solution of the orthogonal¨ Procrustes problem,” Psychometrika, vol. 31, no. 1, pp. 1–10, 1966. [Online]. Available: https://doi.org/10.1007/BF02289451

[44] A. T. Gifford, K. Dwivedi, G. Roig, and R. M. Cichy, “A large and rich EEG dataset for modeling human visual object recognition,” NeuroImage, vol. 264, 2022, art. no. 119754. [Online]. Available: https://doi.org/10.1016/j.neuroimage.2022.119754

[45] J. Xu, U. B. Nunes, W. Jiang, S. Ryther, J. Pringle, P. S. Scotti, A. Delorme, and R. Kneeland, “Alljoined-1.6M: A million-trial EEG-image dataset for evaluating affordable braincomputer interfaces,” 2025, arXiv:2508.18571. [Online]. Available: https://arxiv.org/abs/2508.18571

[46] M. N. Hebart, O. Contier, L. Teichmann, A. H. Rockter, C. Y. Zheng, A. Kidder, A. Corriveau, M. Vaziri-Pashkam, and C. I. Baker, “THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior,” eLife, vol. 12, 2023, art. no. e82580. [Online]. Available: https://doi.org/10.7554/eLife.82580

[47] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proc. 38th Int. Conf. Mach. Learn., ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 8748–8763. [Online]. Available: https://proceedings.mlr.press/v139/radford21a.html

[48] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and´ P. Bojanowski, “DINOv2: Learning robust visual features without supervision,” Trans. Mach. Learn. Res., 2024. [Online]. Available: https://openreview.net/forum?id=a68SUt6zFt

[49] B. Efron, “Bootstrap methods: Another look at the jackknife,” Ann. Statist., vol. 7, no. 1, pp. 1–26, 1979. [Online]. Available: https://doi.org/10.1214/aos/1176344552

[50] S. Holm, “A simple sequentially rejective multiple test procedure,” Scand. J. Stat., vol. 6, no. 2, pp. 65–70, 1979. [Online]. Available: https://www.jstor.org/stable/4615733